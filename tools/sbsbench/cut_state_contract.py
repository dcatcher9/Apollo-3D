"""Canonical loader for authenticated V2 compact cut-state traces.

The compact evaluator trace projects cut pulse/generation state from ``CutBridgeState``. It is not
evidence for the deleted V1 subject, adaptive-pop, or zero-plane controllers.
"""

from __future__ import annotations

import json
from typing import Dict

import numpy as np

try:
    from .adaptive_state_contract import (
        COMPACT_CUT_TRACE_FIELDS,
        COMPACT_CUT_TRACE_ROLES,
        COUNTER_MAX,
        CUT_CONTRACT_TAG,
        FIELD_ENCODINGS,
        FIELD_NAMES,
        KNOWN_CUT_FLAG_MASK,
    )
except ImportError:
    from adaptive_state_contract import (  # type: ignore
        COMPACT_CUT_TRACE_FIELDS,
        COMPACT_CUT_TRACE_ROLES,
        COUNTER_MAX,
        CUT_CONTRACT_TAG,
        FIELD_ENCODINGS,
        FIELD_NAMES,
        KNOWN_CUT_FLAG_MASK,
    )


SCHEMA = int(COMPACT_CUT_TRACE_ROLES["production"]["schema"])
SOURCE = str(COMPACT_CUT_TRACE_ROLES["production"]["source"])
CAPTURE = str(COMPACT_CUT_TRACE_ROLES["production"]["capture"])
GPU_REPLAY_SCHEMA = int(COMPACT_CUT_TRACE_ROLES["gpu_replay"]["schema"])
GPU_REPLAY_SOURCE = str(COMPACT_CUT_TRACE_ROLES["gpu_replay"]["source"])
GPU_REPLAY_CAPTURE = str(COMPACT_CUT_TRACE_ROLES["gpu_replay"]["capture"])

FIELDS = COMPACT_CUT_TRACE_FIELDS
_ENCODINGS = {
    name: encoding for name, encoding in zip(FIELD_NAMES, FIELD_ENCODINGS)
}
UINT64_MAX = (1 << 64) - 1
ROOT_KEYS = frozenset({"schema", "source", "capture", "fields", "frames"})


_CONTRACTS = {
    SCHEMA: {"schema": SCHEMA, "source": SOURCE, "capture": CAPTURE},
    GPU_REPLAY_SCHEMA: {
        "schema": GPU_REPLAY_SCHEMA,
        "source": GPU_REPLAY_SOURCE,
        "capture": GPU_REPLAY_CAPTURE,
    },
}


def _required_contract(schema: int) -> dict:
    if schema not in _CONTRACTS:
        raise ValueError(f"unknown cut-state compatibility schema {schema!r}")
    return {**_CONTRACTS[schema], "fields": list(FIELDS)}


def _exact_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json_object(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream, object_pairs_hook=_exact_json_object)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid cut-state trace {path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != ROOT_KEYS:
        raise ValueError(
            f"invalid cut-state trace {path}: exact root contract required")
    return payload


def _frame_id_value(value: object, description: str) -> int:
    if (not isinstance(value, str) or not value or not value.isascii() or
            any(character < "0" or character > "9" for character in value) or
            len(value) > 20):
        raise ValueError(f"{description} must be an ASCII uint64 decimal string")
    numeric = int(value)
    if numeric > UINT64_MAX:
        raise ValueError(f"{description} exceeds uint64")
    return numeric


def contract_reference(*, gpu_replay: bool = False) -> dict:
    """Return the exact harness descriptor for one supported trace role."""

    required = _required_contract(GPU_REPLAY_SCHEMA if gpu_replay else SCHEMA)
    return {
        "file": "cut_state.json",
        "schema": required["schema"],
        "capture": required["capture"],
    }


def trace_schema(path: str) -> int:
    """Return a supported trace schema; callers must still use ``load_trace`` for validation."""

    payload = _load_json_object(path)
    schema = payload.get("schema")
    try:
        _required_contract(schema)
    except ValueError as exc:
        raise ValueError(f"invalid cut-state trace contract {path}: {exc}") from exc
    return schema


def load_trace(path: str) -> Dict[int, Dict[str, float]]:
    """Load a supported compatibility trace under its exact schema and frame identities."""

    payload = _load_json_object(path)
    try:
        required = _required_contract(payload.get("schema"))
    except ValueError as exc:
        raise ValueError(f"invalid cut-state trace contract {path}: {exc}") from exc
    mismatch = {key: (expected, payload.get(key)) for key, expected in required.items()
                if payload.get(key) != expected}
    frames = payload.get("frames")
    if mismatch or not isinstance(frames, list):
        raise ValueError(
            f"invalid cut-state trace contract {path}: fields={mismatch}, frames=list required")
    trace: Dict[int, Dict[str, float]] = {}
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or set(frame) != {"frame_id", "values"}:
            raise ValueError(
                f"invalid cut-state frame {index} in {path}: exact frame_id/values required")
        frame_text = frame["frame_id"]
        frame_id = _frame_id_value(
            frame_text, f"cut-state frame id at record {index} in {path}")
        values = frame["values"]
        if (frame_id in trace or not isinstance(values, list) or len(values) != len(FIELDS)):
            raise ValueError(f"invalid/duplicate cut-state frame {frame_text!r} in {path}")
        if (not isinstance(values[0], int) or isinstance(values[0], bool) or
                values[0] != CUT_CONTRACT_TAG):
            raise ValueError(
                f"invalid cut-state cut contract tag in frame {frame_text} of {path}")
        numeric = np.asarray(values)
        if (numeric.dtype.kind not in "iuf" or
                any(isinstance(value, (bool, np.bool_)) for value in values)):
            raise ValueError(f"non-numeric cut-state value in frame {frame_text} of {path}")
        numeric = numeric.astype(np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError(f"non-finite cut-state value in frame {frame_text} of {path}")
        for field, value in zip(FIELDS, values):
            if (_ENCODINGS[field] in {"uint_bits", "uint_valued_float"} and
                    (not isinstance(value, int) or isinstance(value, bool) or
                     value < 0 or value > 0xFFFFFFFF)):
                raise ValueError(
                    f"cut-state {field} is not a JSON uint32 in frame {frame_text} of {path}")
        trace[frame_id] = {field: float(value) for field, value in zip(FIELDS, numeric)}
        cut_flags = trace[frame_id]["cut_flags"]
        if cut_flags < 0.0 or abs(cut_flags - round(cut_flags)) > 1.0e-6:
            raise ValueError(
                "cut-state cut_flags is not a non-negative integer in frame "
                f"{frame_text} of {path}")
        if int(round(cut_flags)) & ~KNOWN_CUT_FLAG_MASK:
            raise ValueError(
                "cut-state cut_flags contains unknown schema bits in frame "
                f"{frame_text} of {path}")
        hard_cut_pulse = trace[frame_id]["hard_cut_pulse"]
        if hard_cut_pulse not in (0.0, 1.0):
            raise ValueError(
                "cut-state hard_cut_pulse is not exactly 0 or 1 in frame "
                f"{frame_text} of {path}")
        hard_cut_count = trace[frame_id]["hard_cut_count"]
        if (hard_cut_count < 0.0 or hard_cut_count > COUNTER_MAX or
                hard_cut_count != float(round(hard_cut_count))):
            raise ValueError(
                "cut-state hard_cut_count is not an exact uint32 generation in frame "
                f"{frame_text} of {path}")
    if not trace:
        raise ValueError(f"cut-state trace has no frames: {path}")
    return trace


__all__ = [
    "SCHEMA", "SOURCE", "CAPTURE", "GPU_REPLAY_SCHEMA", "GPU_REPLAY_SOURCE",
    "GPU_REPLAY_CAPTURE", "FIELDS", "CUT_CONTRACT_TAG", "COUNTER_MAX",
    "contract_reference", "trace_schema", "load_trace",
]
