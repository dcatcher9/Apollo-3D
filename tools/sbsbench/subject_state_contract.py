"""Canonical loader for authenticated scene-cut compatibility traces.

Legacy evaluation records the estimator's complete ``SubjectState`` buffer. Native coordinate-v2
replay does not execute that estimator; it records the authenticated cut pulse/generation in the
same fixed-width compatibility layout. Keeping both roles explicit prevents a cut-only trace from
being mistaken for evidence that legacy subject, zero-plane, or adaptive-pop state ran.
"""

from __future__ import annotations

import json
from typing import Dict

import numpy as np


SCHEMA = 2
GPU_REPLAY_SCHEMA = 3

LEGACY_SOURCE = "depth_subject_resolve_cs.SubjectState"
LEGACY_CAPTURE = "every-source-frame-after-estimator-update"
GPU_REPLAY_SOURCE = "depth-coordinate-v2-gpu-input.LegacyState-cut-compatibility"
GPU_REPLAY_CAPTURE = "every-source-frame-from-authenticated-cut-input"

FIELDS = (
    "subject_recenter_delta", "scene_age", "subject_depth_ema", "initialized",
    "stretch_lo", "stretch_inv_range", "depth_change_baseline_ema", "adaptive_pop_ratio",
    "zero_anchor_shift_px", "zero_anchor_valid", "cut_flags",
    "model_input_history_valid", "hard_cut_pulse", "hard_cut_count",
)


def _required_contract(schema: int) -> dict:
    if schema == SCHEMA:
        source = LEGACY_SOURCE
        capture = LEGACY_CAPTURE
    elif schema == GPU_REPLAY_SCHEMA:
        source = GPU_REPLAY_SOURCE
        capture = GPU_REPLAY_CAPTURE
    else:
        raise ValueError(f"unknown subject-state compatibility schema {schema!r}")
    return {
        "schema": schema,
        "source": source,
        "capture": capture,
        "fields": list(FIELDS),
    }


def contract_reference(schema: int = SCHEMA) -> dict:
    """Return the exact harness descriptor for one supported trace role."""

    required = _required_contract(schema)
    return {
        "file": "subject_state.json",
        "schema": schema,
        "capture": required["capture"],
    }


def trace_schema(path: str) -> int:
    """Return a supported trace schema; callers must still use ``load_trace`` for validation."""

    try:
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid subject-state trace {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid subject-state trace {path}: root must be an object")
    schema = payload.get("schema")
    try:
        _required_contract(schema)
    except ValueError as exc:
        raise ValueError(f"invalid subject-state trace contract {path}: {exc}") from exc
    return schema


def load_trace(path: str) -> Dict[int, Dict[str, float]]:
    """Load a supported compatibility trace under its exact schema and frame identities."""

    try:
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid subject-state trace {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid subject-state trace {path}: root must be an object")
    try:
        required = _required_contract(payload.get("schema"))
    except ValueError as exc:
        raise ValueError(f"invalid subject-state trace contract {path}: {exc}") from exc
    mismatch = {key: (expected, payload.get(key)) for key, expected in required.items()
                if payload.get(key) != expected}
    frames = payload.get("frames")
    if mismatch or not isinstance(frames, list):
        raise ValueError(
            f"invalid subject-state trace contract {path}: fields={mismatch}, frames=list required")
    trace: Dict[int, Dict[str, float]] = {}
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or set(frame) != {"frame_id", "values"}:
            raise ValueError(
                f"invalid subject-state frame {index} in {path}: exact frame_id/values required")
        frame_text = frame["frame_id"]
        if not isinstance(frame_text, str) or not frame_text.isdigit():
            raise ValueError(
                f"invalid subject-state frame id {frame_text!r} at record {index} in {path}")
        frame_id = int(frame_text)
        values = frame["values"]
        if (frame_id in trace or not isinstance(values, list) or len(values) != len(FIELDS)):
            raise ValueError(f"invalid/duplicate subject-state frame {frame_text!r} in {path}")
        numeric = np.asarray(values)
        if (numeric.dtype.kind not in "iuf" or
                any(isinstance(value, (bool, np.bool_)) for value in values)):
            raise ValueError(f"non-numeric subject-state value in frame {frame_text} of {path}")
        numeric = numeric.astype(np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError(f"non-finite subject-state value in frame {frame_text} of {path}")
        trace[frame_id] = {field: float(value) for field, value in zip(FIELDS, numeric)}
        cut_flags = trace[frame_id]["cut_flags"]
        if cut_flags < 0.0 or abs(cut_flags - round(cut_flags)) > 1.0e-6:
            raise ValueError(
                "subject-state cut_flags is not a non-negative integer in frame "
                f"{frame_text} of {path}")
        hard_cut_pulse = trace[frame_id]["hard_cut_pulse"]
        if hard_cut_pulse not in (0.0, 1.0):
            raise ValueError(
                "subject-state hard_cut_pulse is not exactly 0 or 1 in frame "
                f"{frame_text} of {path}")
        hard_cut_count = trace[frame_id]["hard_cut_count"]
        if (hard_cut_count < 0.0 or hard_cut_count > 0xFFFFFFFE or
                hard_cut_count != float(round(hard_cut_count))):
            raise ValueError(
                "subject-state hard_cut_count is not an exact uint32 generation in frame "
                f"{frame_text} of {path}")
    if not trace:
        raise ValueError(f"subject-state trace has no frames: {path}")
    return trace


__all__ = [
    "SCHEMA", "GPU_REPLAY_SCHEMA", "FIELDS",
    "LEGACY_SOURCE", "LEGACY_CAPTURE", "GPU_REPLAY_SOURCE", "GPU_REPLAY_CAPTURE",
    "contract_reference", "trace_schema", "load_trace",
]
