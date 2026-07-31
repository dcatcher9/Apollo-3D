"""Strict decoder for the whole-clip Scene Controller shadow trace.

The trace is diagnostic evidence only.  Its header binds every positional array to the
canonical Scene Controller ABI, and every frame binds a snapshot to the exact depth-completion
identity that produced it.  Importing the canonical contract here prevents the Python consumer
from growing a second, hand-maintained field layout.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

if __package__:
    from . import scene_controller_contract as controller_contract
else:
    import scene_controller_contract as controller_contract


TRACE_NAME = "scene_controller.jsonl"
TRACE_SCHEMA = 1
TRACE_SOURCE = "video_depth_estimator.scene_controller_snapshot"
TRACE_CAPTURE = "every-source-frame-after-estimator-update"
TRACE_TRANSPORT = "jsonl-v1"
ATOMIC_TRACE_TRANSPORT = "atomic-latest-v1"
ATOMIC_HEADER_NAME = "scene_controller_header.json"
ATOMIC_FRAME_NAME = "scene_controller_frame.json"
SUPPORTED_BACKENDS = frozenset({"off", "shadow_rules"})

GLOBAL_OUT_FIELDS = tuple(
    {"name": str(field["name"]), "type": "float32"}
    for field in controller_contract.CONTRACT["global_out"]
)
RULE_STATE_FIELDS = tuple(
    {
        "name": str(field["name"]),
        "type": str(field["type"]),
        "gpu_encoding": str(field["gpu_encoding"]),
    }
    for field in controller_contract.CONTRACT["rule_state"]
)
RULE_STATE_CONTRACT = tuple(controller_contract.CONTRACT["rule_state"])
GLOBAL_OUT_CONTRACT = tuple(controller_contract.CONTRACT["global_out"])

HEADER_KEYS = frozenset({
    "record",
    "trace_schema",
    "source",
    "capture",
    "backend",
    "controller_schema",
    "rule_revision",
    "ordered_abi_hash",
    "global_out_fields",
    "rule_state_fields",
    "config",
})
CONFIG_KEYS = frozenset({
    "model",
    "depth_reuse_interval",
    "active_roi_authority",
})
FRAME_KEYS = frozenset({
    "record",
    "frame_id",
    "source_index",
    "depth_updated",
    "snapshot_available",
    "controller_frame_id",
    "backend_generation",
    "shadow",
    "global_out",
    "rule_state",
})
DESCRIPTOR_KEYS = frozenset({
    "enabled",
    "backend",
    "active_roi_authority",
    "transport",
    "file",
    "header_file",
    "frame_file",
    "retained_history",
    "trace_schema",
    "controller_schema",
    "rule_revision",
    "ordered_abi_hash",
    "frame_count",
})

UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
FLOAT32_MAX = 3.4028234663852886e38


class SceneControllerTraceError(ValueError):
    """The Scene Controller trace or descriptor violates its exact contract."""


def _exact_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SceneControllerTraceError(
                f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _exact_keys(value: Any, keys: frozenset[str], description: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise SceneControllerTraceError(
            f"{description} must have exact keys {sorted(keys)}, got {actual}")
    return value


def _uint(value: Any, maximum: int, description: str) -> int:
    if (
        not isinstance(value, int) or
        isinstance(value, bool) or
        not 0 <= value <= maximum
    ):
        bits = 32 if maximum == UINT32_MAX else 64
        raise SceneControllerTraceError(
            f"{description} must be a uint{bits} JSON integer")
    return value


def _finite_float32(value: Any, description: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SceneControllerTraceError(f"{description} must be a JSON number")
    result = float(value)
    if not math.isfinite(result) or abs(result) > FLOAT32_MAX:
        raise SceneControllerTraceError(
            f"{description} must be a finite float32 value")
    return result


def _same_json_contract_type(actual: Any, expected: Any) -> bool:
    """Compare JSON values without Python's bool/int/float coercions."""
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, int):
        return (
            isinstance(actual, int) and
            not isinstance(actual, bool) and
            actual == expected
        )
    if isinstance(expected, float):
        return isinstance(actual, float) and actual == expected
    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected
    if isinstance(expected, list):
        return (
            isinstance(actual, list) and
            len(actual) == len(expected) and
            all(
                _same_json_contract_type(got, wanted)
                for got, wanted in zip(actual, expected)
            )
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict) and
            set(actual) == set(expected) and
            all(
                _same_json_contract_type(actual[key], wanted)
                for key, wanted in expected.items()
            )
        )
    return type(actual) is type(expected) and actual == expected


def _validate_config(value: Any) -> dict[str, Any]:
    config = _exact_keys(value, CONFIG_KEYS, "scene-controller trace config")
    if not isinstance(config["model"], str) or not config["model"]:
        raise SceneControllerTraceError(
            "scene-controller trace config.model must be a non-empty string")
    interval = config["depth_reuse_interval"]
    if (
        not isinstance(interval, int) or
        isinstance(interval, bool) or
        not 1 <= interval <= 8
    ):
        raise SceneControllerTraceError(
            "scene-controller trace config.depth_reuse_interval must be an integer from 1 to 8")
    if config["active_roi_authority"] is not False:
        raise SceneControllerTraceError(
            "scene-controller trace must attest active_roi_authority=false")
    return dict(config)


def _validate_header(value: Any) -> dict[str, Any]:
    header = _exact_keys(value, HEADER_KEYS, "scene-controller trace header")
    expected = {
        "record": "header",
        "trace_schema": TRACE_SCHEMA,
        "source": TRACE_SOURCE,
        "capture": TRACE_CAPTURE,
        "backend": "shadow_rules",
        "controller_schema": controller_contract.SCHEMA_VERSION,
        "rule_revision": controller_contract.RULE_REVISION,
        "ordered_abi_hash": controller_contract.ORDERED_ABI_HASH,
        "global_out_fields": list(GLOBAL_OUT_FIELDS),
        "rule_state_fields": list(RULE_STATE_FIELDS),
    }
    mismatches = {
        key: {"expected": wanted, "actual": header.get(key)}
        for key, wanted in expected.items()
        if not _same_json_contract_type(header.get(key), wanted)
    }
    if mismatches:
        raise SceneControllerTraceError(
            "scene-controller trace header contract mismatch: " +
            json.dumps(mismatches, sort_keys=True))
    return {**header, "config": _validate_config(header["config"])}


def _validate_global_out(value: Any, line_number: int) -> list[float]:
    if not isinstance(value, list) or len(value) != len(GLOBAL_OUT_CONTRACT):
        raise SceneControllerTraceError(
            f"scene-controller global_out at line {line_number} must contain exactly "
            f"{len(GLOBAL_OUT_CONTRACT)} values")
    decoded = [
        _finite_float32(
            item,
            f"scene-controller global_out[{index}] ({field['name']}) "
            f"at line {line_number}",
        )
        for index, (field, item) in enumerate(zip(GLOBAL_OUT_CONTRACT, value))
    ]
    for index, (field, item) in enumerate(zip(GLOBAL_OUT_CONTRACT, decoded)):
        if field.get("required_zero") is True and item != 0.0:
            raise SceneControllerTraceError(
                f"scene-controller global_out[{index}] ({field['name']}) "
                f"at line {line_number} must be zero")
    return decoded


def _validate_rule_state(
    value: Any,
    line_number: int,
    backend_generation: int,
) -> list[float | int]:
    if not isinstance(value, list) or len(value) != len(RULE_STATE_CONTRACT):
        raise SceneControllerTraceError(
            f"scene-controller rule_state at line {line_number} must contain exactly "
            f"{len(RULE_STATE_CONTRACT)} values")
    decoded: list[float | int] = []
    for index, (field, item) in enumerate(zip(RULE_STATE_CONTRACT, value)):
        description = (
            f"scene-controller rule_state[{index}] ({field['name']}) "
            f"at line {line_number}"
        )
        if field["type"] == "uint32":
            decoded_item: float | int = _uint(item, UINT32_MAX, description)
        else:
            decoded_item = _finite_float32(item, description)
            if field["gpu_encoding"] == "uint_valued_float":
                if (
                    decoded_item < 0.0 or
                    decoded_item > UINT32_MAX or
                    decoded_item != math.trunc(decoded_item)
                ):
                    raise SceneControllerTraceError(
                        f"{description} must be a uint32-valued float")
        if field.get("required_zero") is True and decoded_item != 0:
            raise SceneControllerTraceError(f"{description} must be zero")
        decoded.append(decoded_item)

    schema_index = controller_contract.RULE_STATE_NAMES.index("schema_version")
    generation_index = controller_contract.RULE_STATE_NAMES.index(
        "backend_generation")
    if decoded[schema_index] != controller_contract.SCHEMA_VERSION:
        raise SceneControllerTraceError(
            f"scene-controller rule_state schema at line {line_number} does not "
            "match the header")
    if decoded[generation_index] != backend_generation:
        raise SceneControllerTraceError(
            f"scene-controller backend generation at line {line_number} "
            "disagrees with rule_state")
    return decoded


class IncrementalSceneControllerTraceDecoder:
    """Decode one exact Scene Controller JSONL record at a time."""

    def __init__(
        self,
        expected_frame_ids: Iterable[str] | None = None,
        *,
        expected_frame_count: int | None = None,
        retain_frames: bool = True,
        initial_source_index: int = 0,
    ) -> None:
        if (
            not isinstance(initial_source_index, int) or
            isinstance(initial_source_index, bool) or
            initial_source_index < 0
        ):
            raise SceneControllerTraceError(
                "initial scene-controller source index must be non-negative")
        if (
            expected_frame_count is not None and
            (
                not isinstance(expected_frame_count, int) or
                isinstance(expected_frame_count, bool) or
                expected_frame_count < 0
            )
        ):
            raise SceneControllerTraceError(
                "expected scene-controller frame count must be non-negative")
        self.header: dict[str, Any] | None = None
        self.frames: list[dict[str, Any]] = []
        self.frame_count = initial_source_index
        self.line_number = 0
        self._finalized = False
        self._retain_frames = retain_frames
        self._expected_frame_ids = (
            list(expected_frame_ids) if expected_frame_ids is not None else None
        )
        if (
            self._expected_frame_ids is not None and
            expected_frame_count is not None and
            len(self._expected_frame_ids) != expected_frame_count
        ):
            raise SceneControllerTraceError(
                "expected scene-controller identities/frame count differ")
        self._expected_frame_count = (
            len(self._expected_frame_ids)
            if self._expected_frame_ids is not None
            else expected_frame_count
        )
        self._previous_controller_frame_id: int | None = None
        self._previous_backend_generation: int | None = None

    def feed_line(self, raw_line: str) -> dict[str, Any] | None:
        if self._finalized:
            raise SceneControllerTraceError(
                "cannot feed a scene-controller trace after finalization")
        self.line_number += 1
        if not isinstance(raw_line, str) or not raw_line.strip():
            raise SceneControllerTraceError(
                f"scene-controller trace contains an empty record at line "
                f"{self.line_number}")
        try:
            value = json.loads(raw_line, object_pairs_hook=_exact_json_object)
        except (json.JSONDecodeError, SceneControllerTraceError) as exc:
            raise SceneControllerTraceError(
                f"invalid JSON at scene-controller trace line "
                f"{self.line_number}: {exc}") from exc
        if self.line_number == 1:
            self.header = _validate_header(value)
            return None
        if self.header is None:
            raise SceneControllerTraceError(
                "scene-controller frame appeared before its header")

        frame = _exact_keys(
            value,
            FRAME_KEYS,
            f"scene-controller frame at line {self.line_number}",
        )
        if frame["record"] != "frame":
            raise SceneControllerTraceError(
                f"scene-controller record at line {self.line_number} must be 'frame'")
        source_index = frame["source_index"]
        if (
            not isinstance(source_index, int) or
            isinstance(source_index, bool) or
            source_index != self.frame_count
        ):
            raise SceneControllerTraceError(
                f"scene-controller source_index at line {self.line_number} must "
                f"be contiguous index {self.frame_count}")
        frame_id = frame["frame_id"]
        if not isinstance(frame_id, str) or not frame_id.isdigit():
            raise SceneControllerTraceError(
                f"scene-controller frame_id at line {self.line_number} must be "
                "a decimal string")
        if self._expected_frame_ids is not None:
            if source_index >= len(self._expected_frame_ids):
                raise SceneControllerTraceError(
                    "scene-controller trace contains more frames than the source timeline")
            expected_id = self._expected_frame_ids[source_index]
            if frame_id != expected_id:
                raise SceneControllerTraceError(
                    f"scene-controller frame identity mismatch at source_index "
                    f"{source_index}: {frame_id!r} != {expected_id!r}")
        depth_updated = frame["depth_updated"]
        if not isinstance(depth_updated, bool):
            raise SceneControllerTraceError(
                f"scene-controller depth_updated at line {self.line_number} "
                "must be a JSON boolean")
        interval = int(self.header["config"]["depth_reuse_interval"])
        terminal_forced_update = (
            self._expected_frame_count is not None and
            self._expected_frame_count > 0 and
            source_index == self._expected_frame_count - 1
        )
        expected_depth_updated = (
            source_index % interval == 0 or
            terminal_forced_update
        )
        if depth_updated != expected_depth_updated:
            raise SceneControllerTraceError(
                f"scene-controller depth_updated at line {self.line_number} "
                f"disagrees with depth_reuse_interval={interval}")
        snapshot_available = frame["snapshot_available"]
        shadow = frame["shadow"]
        if not isinstance(snapshot_available, bool) or not isinstance(shadow, bool):
            raise SceneControllerTraceError(
                f"scene-controller snapshot/shadow flags at line "
                f"{self.line_number} must be JSON booleans")
        if snapshot_available is not True:
            raise SceneControllerTraceError(
                f"shadow_rules scene-controller snapshot at line "
                f"{self.line_number} is unavailable")

        if snapshot_available:
            controller_frame_id = _uint(
                frame["controller_frame_id"],
                UINT64_MAX,
                f"scene-controller controller_frame_id at line {self.line_number}",
            )
            backend_generation = _uint(
                frame["backend_generation"],
                UINT32_MAX,
                f"scene-controller backend_generation at line {self.line_number}",
            )
            expected_controller_frame_id = (
                source_index
                if terminal_forced_update else
                source_index - (source_index % interval)
            )
            if controller_frame_id != expected_controller_frame_id:
                raise SceneControllerTraceError(
                    f"scene-controller source identity mismatch at line "
                    f"{self.line_number}: controller_frame_id={controller_frame_id}, "
                    f"expected={expected_controller_frame_id}")
            if backend_generation == 0:
                raise SceneControllerTraceError(
                    f"scene-controller backend_generation at line "
                    f"{self.line_number} must be positive")
            if shadow is not True:
                raise SceneControllerTraceError(
                    f"available scene-controller snapshot at line "
                    f"{self.line_number} must remain shadow-only")
            global_out = _validate_global_out(
                frame["global_out"], self.line_number)
            rule_state = _validate_rule_state(
                frame["rule_state"], self.line_number, backend_generation)
            if (
                self._previous_controller_frame_id == controller_frame_id and
                self._previous_backend_generation != backend_generation
            ):
                raise SceneControllerTraceError(
                    f"scene-controller generation changed without a new source "
                    f"identity at line {self.line_number}")
            if (
                self._previous_backend_generation is not None and
                backend_generation < self._previous_backend_generation
            ):
                raise SceneControllerTraceError(
                    f"scene-controller backend generation decreased at line "
                    f"{self.line_number}")
            self._previous_controller_frame_id = controller_frame_id
            self._previous_backend_generation = backend_generation
        else:
            if (
                frame["controller_frame_id"] is not None or
                frame["backend_generation"] is not None or
                frame["global_out"] is not None or
                frame["rule_state"] is not None
            ):
                raise SceneControllerTraceError(
                    f"unavailable scene-controller snapshot at line "
                    f"{self.line_number} must use null identity and payloads")
            controller_frame_id = None
            backend_generation = None
            global_out = None
            rule_state = None

        decoded = {
            **frame,
            "controller_frame_id": controller_frame_id,
            "backend_generation": backend_generation,
            "global_out": global_out,
            "rule_state": rule_state,
        }
        if self._retain_frames:
            self.frames.append(decoded)
        self.frame_count += 1
        return decoded

    def finalize(self, expected_frame_count: int | None = None) -> dict[str, Any]:
        if self.header is None:
            raise SceneControllerTraceError("scene-controller trace is empty")
        if expected_frame_count is not None:
            if (
                not isinstance(expected_frame_count, int) or
                isinstance(expected_frame_count, bool) or
                expected_frame_count < 0
            ):
                raise SceneControllerTraceError(
                    "expected scene-controller frame count must be non-negative")
            if (
                self._expected_frame_count is not None and
                expected_frame_count != self._expected_frame_count
            ):
                raise SceneControllerTraceError(
                    "expected scene-controller frame counts disagree")
            if self.frame_count != expected_frame_count:
                raise SceneControllerTraceError(
                    f"scene-controller trace contains {self.frame_count} frames; "
                    f"expected {expected_frame_count}")
        if (
            self._expected_frame_ids is not None and
            self.frame_count != len(self._expected_frame_ids)
        ):
            raise SceneControllerTraceError(
                "scene-controller trace/timeline frame counts differ")
        self._finalized = True
        return self.header


def load_trace(
    path: str | os.PathLike[str],
    *,
    expected_frame_ids: Iterable[str] | None = None,
    expected_frame_count: int | None = None,
    retain_frames: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace_path = Path(path)
    decoder = IncrementalSceneControllerTraceDecoder(
        expected_frame_ids,
        expected_frame_count=expected_frame_count,
        retain_frames=retain_frames,
    )
    try:
        stream = trace_path.open(encoding="utf-8")
    except OSError as exc:
        raise SceneControllerTraceError(
            f"cannot open scene-controller trace {trace_path}: {exc}") from exc
    with stream:
        for raw_line in stream:
            try:
                decoder.feed_line(raw_line)
            except SceneControllerTraceError as exc:
                raise SceneControllerTraceError(
                    f"{trace_path}:{decoder.line_number}: {exc}") from exc
    try:
        header = decoder.finalize(expected_frame_count)
    except SceneControllerTraceError as exc:
        raise SceneControllerTraceError(f"{trace_path}: {exc}") from exc
    return header, decoder.frames


def validate_atomic_latest(
    artifacts_dir: Path,
    *,
    expected_frame_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not expected_frame_ids:
        raise SceneControllerTraceError(
            "atomic scene-controller validation requires source frames")
    header_path = artifacts_dir / ATOMIC_HEADER_NAME
    frame_path = artifacts_dir / ATOMIC_FRAME_NAME
    try:
        header_text = header_path.read_text(encoding="utf-8")
        frame_text = frame_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SceneControllerTraceError(
            f"cannot read atomic scene-controller snapshots: {exc}") from exc
    decoder = IncrementalSceneControllerTraceDecoder(
        expected_frame_ids,
        retain_frames=False,
        initial_source_index=len(expected_frame_ids) - 1,
    )
    decoder.feed_line(header_text)
    frame = decoder.feed_line(frame_text)
    assert frame is not None
    header = decoder.finalize(len(expected_frame_ids))
    return header, frame


def validate_descriptor(
    descriptor: Any,
    artifacts_dir: Path,
    *,
    expected_frame_ids: Iterable[str],
    expected_model: str,
    expected_depth_reuse_interval: int,
    expected_backend: str,
    replay: bool = False,
) -> dict[str, Any]:
    """Validate the whole-clip descriptor and its trace, if enabled."""

    value = _exact_keys(
        descriptor, DESCRIPTOR_KEYS, "whole-clip scene-controller descriptor")
    expected_ids = list(expected_frame_ids)
    backend = value["backend"]
    if backend not in SUPPORTED_BACKENDS:
        raise SceneControllerTraceError(
            f"unsupported scene-controller backend {backend!r}")
    if expected_backend not in SUPPORTED_BACKENDS:
        raise SceneControllerTraceError(
            f"unsupported expected scene-controller backend {expected_backend!r}")
    if backend != expected_backend:
        raise SceneControllerTraceError(
            "whole-clip scene-controller backend mismatch: "
            f"{backend!r} != {expected_backend!r}")
    if replay and expected_backend != "off":
        raise SceneControllerTraceError(
            "scene-cache replay must expect scene-controller backend 'off'")
    common_expected = {
        "active_roi_authority": False,
        "controller_schema": controller_contract.SCHEMA_VERSION,
        "rule_revision": controller_contract.RULE_REVISION,
        "ordered_abi_hash": controller_contract.ORDERED_ABI_HASH,
    }
    common_mismatches = {
        key: {"expected": wanted, "actual": value.get(key)}
        for key, wanted in common_expected.items()
        if not _same_json_contract_type(value.get(key), wanted)
    }
    if common_mismatches:
        raise SceneControllerTraceError(
            "whole-clip scene-controller descriptor mismatch: " +
            json.dumps(common_mismatches, sort_keys=True))

    enabled = value["enabled"]
    if not isinstance(enabled, bool):
        raise SceneControllerTraceError(
            "whole-clip scene-controller enabled must be a JSON boolean")
    expected_enabled = expected_backend == "shadow_rules"
    if enabled != expected_enabled:
        raise SceneControllerTraceError(
            "whole-clip scene-controller enabled/backend mismatch: "
            f"enabled={enabled!r}, backend={expected_backend!r}")
    trace_path = artifacts_dir / TRACE_NAME
    header_path = artifacts_dir / ATOMIC_HEADER_NAME
    frame_path = artifacts_dir / ATOMIC_FRAME_NAME
    if not enabled:
        expected_disabled = {
            "transport": None,
            "file": None,
            "header_file": None,
            "frame_file": None,
            "retained_history": False,
            "trace_schema": None,
            "frame_count": 0,
        }
        mismatches = {
            key: {"expected": wanted, "actual": value.get(key)}
            for key, wanted in expected_disabled.items()
            if not _same_json_contract_type(value.get(key), wanted)
        }
        if mismatches:
            raise SceneControllerTraceError(
                "disabled scene-controller descriptor mismatch: " +
                json.dumps(mismatches, sort_keys=True))
        if trace_path.exists() or header_path.exists() or frame_path.exists():
            raise SceneControllerTraceError(
                "disabled scene-controller unexpectedly produced a trace")
        return dict(value)

    if replay:
        raise SceneControllerTraceError(
            "scene-cache replay must not enable the scene controller")
    common_enabled = {
        "backend": "shadow_rules",
        "trace_schema": TRACE_SCHEMA,
        "frame_count": len(expected_ids),
    }
    transport = value["transport"]
    if transport == TRACE_TRANSPORT:
        transport_expected = {
            "file": TRACE_NAME,
            "header_file": None,
            "frame_file": None,
            "retained_history": True,
        }
    elif transport == ATOMIC_TRACE_TRANSPORT:
        transport_expected = {
            "file": None,
            "header_file": ATOMIC_HEADER_NAME,
            "frame_file": ATOMIC_FRAME_NAME,
            "retained_history": False,
        }
    else:
        raise SceneControllerTraceError(
            f"unsupported enabled scene-controller transport {transport!r}")
    expected_enabled = {**common_enabled, **transport_expected}
    mismatches = {
        key: {"expected": wanted, "actual": value.get(key)}
        for key, wanted in expected_enabled.items()
        if not _same_json_contract_type(value.get(key), wanted)
    }
    if mismatches:
        raise SceneControllerTraceError(
            "enabled scene-controller descriptor mismatch: " +
            json.dumps(mismatches, sort_keys=True))
    if transport == TRACE_TRANSPORT:
        if header_path.exists() or frame_path.exists():
            raise SceneControllerTraceError(
                "JSONL scene-controller transport produced atomic snapshots")
        header, _frames = load_trace(
            trace_path,
            expected_frame_ids=expected_ids,
            expected_frame_count=len(expected_ids),
            retain_frames=False,
        )
        validated_frame_count = len(expected_ids)
    else:
        if trace_path.exists():
            raise SceneControllerTraceError(
                "atomic scene-controller transport produced a JSONL trace")
        header, _frame = validate_atomic_latest(
            artifacts_dir,
            expected_frame_ids=expected_ids,
        )
        validated_frame_count = len(expected_ids)
    header_mismatches = {}
    if header["config"]["model"] != expected_model:
        header_mismatches["config.model"] = {
            "expected": expected_model,
            "actual": header["config"]["model"],
        }
    if (
        header["config"]["depth_reuse_interval"] !=
        expected_depth_reuse_interval
    ):
        header_mismatches["config.depth_reuse_interval"] = {
            "expected": expected_depth_reuse_interval,
            "actual": header["config"]["depth_reuse_interval"],
        }
    if header_mismatches:
        raise SceneControllerTraceError(
            "scene-controller trace/native contract mismatch: " +
            json.dumps(header_mismatches, sort_keys=True))
    return {
        **value,
        "validated_header": header,
        "validated_frame_count": validated_frame_count,
    }
