#!/usr/bin/env python3
"""Analyze a continuous Host SBS V2 depth-health trace.

The native ``--sbs-bench`` whole-clip mode writes ``adaptive_state.jsonl``.  Its first
record freezes the exact GPU state layout and relevant Host SBS configuration; every later
record contains one source-frame sample.  This module validates that contract, joins source
presentation timing, and writes machine-readable CSV/JSON plus a self-contained HTML report.

The report is deliberately diagnostic. It describes depth readiness, valid-depth support, raw
range width, and model-health counters; it does not claim authority over V2 geometry or camera
policy.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import html
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable

if __package__:
    from .adaptive_state_contract import (
        CONFIG_KEYS,
        COUNTER_MAX,
        CUT_CONTRACT_TAG,
        FIELD_DESCRIPTORS,
        FIELD_ENCODINGS,
        FIELD_NAMES,
        FIELD_SPECS,
        FRAME_KEYS,
        HEADER_KEYS,
        INITIAL_VALUES,
        RETIRED_ZERO_FIELDS,
        TRACE_CAPTURE,
        TRACE_SCHEMA,
        TRACE_SOURCE,
    )
else:
    from adaptive_state_contract import (
        CONFIG_KEYS,
        COUNTER_MAX,
        CUT_CONTRACT_TAG,
        FIELD_DESCRIPTORS,
        FIELD_ENCODINGS,
        FIELD_NAMES,
        FIELD_SPECS,
        FRAME_KEYS,
        HEADER_KEYS,
        INITIAL_VALUES,
        RETIRED_ZERO_FIELDS,
        TRACE_CAPTURE,
        TRACE_SCHEMA,
        TRACE_SOURCE,
    )
UINT64_MAX = (1 << 64) - 1
_UINT64_MAX_DECIMAL = str(UINT64_MAX)


class TraceContractError(ValueError):
    """The JSONL or timing input does not satisfy its exact contract."""


def _exact_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TraceContractError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: Any, description: str) -> float:
    if not _is_number(value):
        raise TraceContractError(f"{description} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise TraceContractError(f"{description} must be finite")
    return result


def _uint32(value: Any, description: str) -> int:
    if (not isinstance(value, int) or isinstance(value, bool) or
            not 0 <= value <= 0xFFFFFFFF):
        raise TraceContractError(f"{description} must be a uint32 JSON integer")
    return value


def _frame_id_value(value: Any, description: str) -> int:
    """Parse an unsigned 64-bit identity while ignoring decimal zero-padding."""
    if (
        not isinstance(value, str) or
        not value or
        any(character < "0" or character > "9" for character in value)
    ):
        raise TraceContractError(
            f"{description} must be an ASCII decimal string")
    significant = value.lstrip("0") or "0"
    if (
        len(significant) > len(_UINT64_MAX_DECIMAL) or
        (
            len(significant) == len(_UINT64_MAX_DECIMAL) and
            significant > _UINT64_MAX_DECIMAL
        )
    ):
        raise TraceContractError(f"{description} exceeds the uint64 range")
    return int(significant)


def _boolean(value: Any, description: str) -> bool:
    if not isinstance(value, bool):
        raise TraceContractError(f"{description} must be a JSON boolean")
    return value


def _validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        actual = sorted(config) if isinstance(config, dict) else type(config).__name__
        raise TraceContractError(
            f"trace config must have exact keys {sorted(CONFIG_KEYS)}, got {actual}")
    if not isinstance(config["model"], str) or not config["model"]:
        raise TraceContractError("trace config.model must be a non-empty string")
    pop_strength = _finite_number(config["pop_strength"], "trace config.pop_strength")
    if not 0.25 <= pop_strength <= 2.0:
        raise TraceContractError("trace config.pop_strength must be in [0.25, 2.0]")
    interval = config["depth_reuse_interval"]
    if (not isinstance(interval, int) or isinstance(interval, bool) or
            interval != 1):
        raise TraceContractError(
            "trace config.depth_reuse_interval must be exactly 1")
    return dict(config)


def _validate_header(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != HEADER_KEYS:
        actual = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise TraceContractError(
            f"trace header must have exact keys {sorted(HEADER_KEYS)}, got {actual}")
    expected = {
        "record": "header",
        "schema": TRACE_SCHEMA,
        "source": TRACE_SOURCE,
        "capture": TRACE_CAPTURE,
        "fields": list(FIELD_DESCRIPTORS),
    }
    mismatches = {
        key: (value, payload.get(key))
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise TraceContractError(f"trace header contract mismatch: {mismatches}")
    return {**payload, "config": _validate_config(payload["config"])}


def _validate_frame(payload: Any, line_number: int, config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != FRAME_KEYS:
        actual = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise TraceContractError(
            f"trace frame at line {line_number} must have exact keys "
            f"{sorted(FRAME_KEYS)}, got {actual}")
    if payload["record"] != "frame":
        raise TraceContractError(
            f"trace record at line {line_number} must have record='frame'")
    frame_id = payload["frame_id"]
    _frame_id_value(frame_id, f"trace frame_id at line {line_number}")
    source_index = payload["source_index"]
    if (not isinstance(source_index, int) or isinstance(source_index, bool) or
            source_index < 0):
        raise TraceContractError(
            f"trace source_index at line {line_number} must be a non-negative integer")
    values = payload["values"]
    if not isinstance(values, list) or len(values) != len(FIELD_SPECS):
        raise TraceContractError(
            f"trace values at line {line_number} must contain exactly "
            f"{len(FIELD_SPECS)} entries")
    decoded: dict[str, float | int] = {}
    for index, ((name, kind), encoding, value) in enumerate(
            zip(FIELD_SPECS, FIELD_ENCODINGS, values)):
        description = f"trace values[{index}] ({name}) at line {line_number}"
        decoded[name] = (
            _uint32(value, description)
            if encoding in {"uint_bits", "uint_valued_float"} else
            _finite_number(value, description)
        )
    if decoded["cut_contract_tag_bits"] != CUT_CONTRACT_TAG:
        raise TraceContractError(
            f"trace cut contract tag at line {line_number} is not schema-{TRACE_SCHEMA}"
        )

    depth_updated = _boolean(
        payload["depth_updated"], f"trace depth_updated at line {line_number}")
    if not depth_updated:
        raise TraceContractError(
            f"trace depth_updated at line {line_number} must be true")
    for name, initial in INITIAL_VALUES.items():
        if name.startswith("reserved_") and decoded[name] != initial:
            raise TraceContractError(
                f"trace reserved field {name} at line {line_number} is non-default")
    for name in RETIRED_ZERO_FIELDS:
        if decoded[name] != 0:
            raise TraceContractError(
                f"trace retired field {name} at line {line_number} must be zero")
    for name in (
        "empty_raw_count",
        "collapsed_raw_count",
    ):
        if decoded[name] > COUNTER_MAX:
            raise TraceContractError(
                f"trace {name} at line {line_number} exceeds the counter contract")
        duplicate = _uint32(payload[name], f"trace {name} at line {line_number}")
        if duplicate != decoded[name]:
            raise TraceContractError(
                f"trace {name} duplicate disagrees with values at line {line_number}")

    return {
        "frame_id": frame_id,
        "source_index": source_index,
        "depth_updated": depth_updated,
        **decoded,
    }


class IncrementalTraceDecoder:
    """Validate an adaptive JSONL stream one complete record at a time.

    Native follow-mode progress is published only after its matching trace snapshot is atomically
    complete. The offline wrapper can therefore feed exactly one line per acknowledged frame
    without repeatedly reparsing an ever-growing file. ``load_trace`` uses this same decoder so
    streaming and final validation cannot drift.
    """

    def __init__(self) -> None:
        self.header: dict[str, Any] | None = None
        self.frame_count = 0
        self.line_number = 0
        self._seen_ids: set[int] = set()
        self._previous_numeric_id: int | None = None
        self._finalized = False

    def feed_line(self, raw_line: str) -> dict[str, Any] | None:
        """Consume one newline-delimited record; return a decoded frame or ``None`` for header."""
        if self._finalized:
            raise TraceContractError("cannot feed an adaptive trace after finalization")
        self.line_number += 1
        if not isinstance(raw_line, str) or not raw_line.strip():
            raise TraceContractError(
                f"adaptive trace contains an empty record at line {self.line_number}")
        try:
            payload = json.loads(raw_line, object_pairs_hook=_exact_json_object)
        except (json.JSONDecodeError, TraceContractError) as exc:
            raise TraceContractError(
                f"invalid JSON at adaptive trace line {self.line_number}: {exc}") from exc
        if self.line_number == 1:
            self.header = _validate_header(payload)
            return None
        if self.header is None:
            raise TraceContractError("adaptive trace frame appeared before its header")
        frame = _validate_frame(
            payload, self.line_number, self.header["config"])
        frame_id = frame["frame_id"]
        numeric_id = _frame_id_value(
            frame_id, f"trace frame_id at line {self.line_number}")
        if numeric_id in self._seen_ids:
            raise TraceContractError(
                f"duplicate trace frame_id {frame_id} at line {self.line_number}")
        if (
            self._previous_numeric_id is not None and
            numeric_id <= self._previous_numeric_id
        ):
            raise TraceContractError(
                f"trace frame ids must increase numerically; {frame_id} "
                f"at line {self.line_number} follows {self._previous_numeric_id}")
        if frame["source_index"] != self.frame_count:
            raise TraceContractError(
                f"trace source_index at line {self.line_number} is "
                f"{frame['source_index']}; expected contiguous index {self.frame_count}")
        self._seen_ids.add(numeric_id)
        self._previous_numeric_id = numeric_id
        self.frame_count += 1
        return frame

    def finalize(self, *, require_frames: bool = True) -> dict[str, Any]:
        """Seal the decoder and return its validated header."""
        if self.header is None:
            raise TraceContractError("adaptive trace is empty")
        if require_frames and self.frame_count == 0:
            raise TraceContractError("adaptive trace has no frame records")
        self._finalized = True
        return self.header


def load_trace(path: str | os.PathLike[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and validate an exact current-schema adaptive JSONL trace."""
    trace_path = Path(path)
    decoder = IncrementalTraceDecoder()
    frames: list[dict[str, Any]] = []
    try:
        stream = trace_path.open(encoding="utf-8")
    except OSError as exc:
        raise TraceContractError(f"cannot open adaptive trace {trace_path}: {exc}") from exc
    with stream:
        for raw_line in stream:
            try:
                frame = decoder.feed_line(raw_line)
            except TraceContractError as exc:
                raise TraceContractError(
                    f"{trace_path}:{decoder.line_number}: {exc}") from exc
            if frame is not None:
                frames.append(frame)
    try:
        header = decoder.finalize()
    except TraceContractError as exc:
        raise TraceContractError(f"{trace_path}: {exc}") from exc
    return header, frames


def _timeline_frames(timeline: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize the wrapper's schema-1 timeline or a direct list of timing rows."""
    if isinstance(timeline, (str, os.PathLike)):
        try:
            with Path(timeline).open(encoding="utf-8") as stream:
                timeline = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise TraceContractError(f"cannot load timing manifest {timeline}: {exc}") from exc
    if isinstance(timeline, list):
        return timeline, {"source": "caller-list"}
    if not isinstance(timeline, dict):
        raise TraceContractError("timeline must be a list, schema-1 object, path, or None")
    if timeline.get("schema") != 1 or not isinstance(timeline.get("frames"), list):
        raise TraceContractError("timeline object requires schema 1 and a frames list")
    metadata = {key: value for key, value in timeline.items() if key != "frames"}
    return timeline["frames"], metadata


def join_timeline(
    frames: list[dict[str, Any]],
    timeline: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach normalized presentation timestamps and durations to trace frames.

    The wrapper supplies source-video PTS records.  Direct callers may pass a list containing
    ``frame_id`` or zero-based ``index`` plus ``pts_seconds``/``pts_time`` and an optional
    ``duration_seconds``/``duration_time``.  With no timeline, frame index is the explicit clock.
    """
    if timeline is None:
        rows = [
            {
                **frame,
                "source_pts_seconds": float(index),
                "timestamp_seconds": float(index),
                "duration_seconds": 1.0,
            }
            for index, frame in enumerate(frames)
        ]
        return rows, {
            "source": "frame-index",
            "first_pts_seconds": 0.0,
            "nominal_fps": 1.0,
            "variable_frame_rate": False,
        }

    timing_rows, metadata = _timeline_frames(timeline)
    by_id: dict[int, dict[str, Any]] = {}
    by_index: dict[int, dict[str, Any]] = {}
    for position, row in enumerate(timing_rows):
        if not isinstance(row, dict):
            raise TraceContractError(f"timeline frame {position} must be an object")
        frame_id = row.get("frame_id")
        index = row.get("index")
        if frame_id is not None:
            numeric_frame_id = _frame_id_value(
                frame_id, f"timeline frame {position} frame_id")
            if numeric_frame_id in by_id:
                raise TraceContractError(
                    f"timeline frame {position} has invalid/duplicate frame_id")
            by_id[numeric_frame_id] = row
        if index is not None:
            if (not isinstance(index, int) or isinstance(index, bool) or index < 0 or
                    index in by_index):
                raise TraceContractError(
                    f"timeline frame {position} has invalid/duplicate index")
            by_index[index] = row
        if frame_id is None and index is None:
            raise TraceContractError(
                f"timeline frame {position} requires frame_id or index")

    joined: list[dict[str, Any]] = []
    used: set[int] = set()
    raw_pts: list[float] = []
    raw_durations: list[float | None] = []
    for frame in frames:
        trace_numeric_id = _frame_id_value(
            frame["frame_id"],
            f"trace frame_id at source_index={frame['source_index']}",
        )
        timing_by_id = by_id.get(trace_numeric_id)
        timing_by_index = by_index.get(frame["source_index"])
        if (timing_by_id is not None and timing_by_index is not None and
                timing_by_id is not timing_by_index):
            raise TraceContractError(
                f"timeline frame_id/index identities resolve to different records for "
                f"trace frame {frame['frame_id']} at source_index={frame['source_index']}")
        timing = timing_by_id if timing_by_id is not None else timing_by_index
        if timing is None:
            raise TraceContractError(
                f"timeline has no record for trace frame {frame['frame_id']}")
        if (
            "frame_id" in timing and
            _frame_id_value(
                timing["frame_id"],
                f"timeline frame_id for source_index={frame['source_index']}",
            ) != trace_numeric_id
        ):
            raise TraceContractError(
                f"timeline frame_id {timing['frame_id']!r} disagrees with trace frame "
                f"{frame['frame_id']!r} at source_index={frame['source_index']}")
        if ("index" in timing and timing["index"] != frame["source_index"]):
            raise TraceContractError(
                f"timeline index {timing['index']!r} disagrees with trace source_index "
                f"{frame['source_index']} for frame {frame['frame_id']}")
        identity = id(timing)
        if identity in used:
            raise TraceContractError(
                f"one timeline record matched multiple trace frames near {frame['frame_id']}")
        used.add(identity)
        pts_key = "pts_seconds" if "pts_seconds" in timing else "pts_time"
        if pts_key not in timing:
            raise TraceContractError(
                f"timeline frame {frame['frame_id']} has no pts_seconds/pts_time")
        pts = _finite_number(timing[pts_key], f"timeline PTS for {frame['frame_id']}")
        duration_value = timing.get(
            "duration_seconds",
            timing.get("duration_time"),
        )
        duration = (
            None if duration_value is None else
            _finite_number(duration_value, f"timeline duration for {frame['frame_id']}")
        )
        if duration is not None and duration < 0.0:
            raise TraceContractError(
                f"timeline duration for {frame['frame_id']} must be non-negative")
        raw_pts.append(pts)
        raw_durations.append(duration)

    if len(used) != len(timing_rows):
        raise TraceContractError(
            "timeline contains records that do not match the adaptive trace")
    if any(right < left for left, right in zip(raw_pts, raw_pts[1:])):
        raise TraceContractError("timeline presentation timestamps must be non-decreasing")
    positive_steps = [
        right - left for left, right in zip(raw_pts, raw_pts[1:]) if right > left
    ]
    fallback_duration = (
        statistics.median(positive_steps)
        if positive_steps else
        (1.0 / float(metadata["nominal_fps"])
         if _is_number(metadata.get("nominal_fps")) and
         float(metadata["nominal_fps"]) > 0.0 else 1.0)
    )
    first_pts = raw_pts[0]
    for index, frame in enumerate(frames):
        duration = raw_durations[index]
        if duration is None:
            duration = (
                raw_pts[index + 1] - raw_pts[index]
                if index + 1 < len(raw_pts) and raw_pts[index + 1] >= raw_pts[index]
                else fallback_duration
            )
        joined.append({
            **frame,
            "source_pts_seconds": raw_pts[index],
            "timestamp_seconds": raw_pts[index] - first_pts,
            "duration_seconds": duration,
        })
    inferred_fps = 1.0 / fallback_duration if fallback_duration > 0.0 else None
    timing_meta = {
        **metadata,
        "source": metadata.get("clock", metadata.get("source", "source-timeline")),
        "first_pts_seconds": first_pts,
        "nominal_fps": metadata.get("nominal_fps", inferred_fps),
        "variable_frame_rate": bool(metadata.get("variable_frame_rate", False)),
    }
    return joined, timing_meta


def _frame_end(frame: dict[str, Any]) -> float:
    return float(frame["timestamp_seconds"]) + float(frame["duration_seconds"])


def _counter_delta(previous: int, current: int) -> int:
    # Native counters saturate instead of wrapping. A decrease means this purportedly continuous
    # trace crossed an estimator reset without a new contract.
    if current < previous:
        raise TraceContractError(
            f"cumulative telemetry counter decreased ({previous} -> {current})")
    return current - previous


def _contiguous_runs(
    frames: list[dict[str, Any]],
    predicate,
) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for frame in frames:
        if predicate(frame):
            current.append(frame)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _anomaly(
    kind: str,
    severity: str,
    message: str,
    frame: dict[str, Any] | None = None,
    **details: Any,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "kind": kind,
        "severity": severity,
        "message": message,
    }
    if frame is not None:
        item.update({
            "frame_id": frame["frame_id"],
            "source_index": frame["source_index"],
            "timestamp_seconds": frame["timestamp_seconds"],
        })
    item.update(details)
    return item


def find_anomalies(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    ready = [frame for frame in frames if frame["depth_ready"] > 0.5]
    if not ready:
        anomalies.append(_anomaly(
            "depth_never_ready",
            "error",
            "The depth state never became ready.",
            frames[0],
        ))
    elif ready[0]["source_index"] > 1:
        anomalies.append(_anomaly(
            "depth_ready_delay",
            "warning",
            "Depth readiness took more than two source frames.",
            ready[0],
            delayed_source_frames=ready[0]["source_index"],
        ))

    first_ready_index = ready[0]["source_index"] if ready else None
    for run in _contiguous_runs(
            frames,
            lambda frame: first_ready_index is not None and
            frame["source_index"] > first_ready_index and
            frame["depth_updated"] and frame["depth_ready"] <= 0.5):
        anomalies.append(_anomaly(
            "depth_not_ready_after_ready",
            "error",
            f"Depth readiness was lost for {len(run)} depth update(s).",
            run[0],
            end_frame_id=run[-1]["frame_id"],
            duration_seconds=max(0.0, _frame_end(run[-1]) -
                                 float(run[0]["timestamp_seconds"])),
        ))

    for counter, kind, message in (
        ("empty_raw_count", "empty_raw_depth",
         "The first model result was empty/invalid raw depth."),
        ("collapsed_raw_count", "collapsed_raw_depth",
         "The first model result had a collapsed raw depth range."),
    ):
        initial_count = int(frames[0][counter])
        if initial_count:
            anomalies.append(_anomaly(
                kind,
                "error",
                message,
                frames[0],
                count_delta=initial_count,
                cumulative_count=initial_count,
            ))

    previous = frames[0]
    for frame in frames[1:]:
        for counter, kind, message in (
            ("empty_raw_count", "empty_raw_depth",
             "The model produced an empty/invalid raw depth result."),
            ("collapsed_raw_count", "collapsed_raw_depth",
             "The model produced a collapsed raw depth range."),
        ):
            delta = _counter_delta(int(previous[counter]), int(frame[counter]))
            if delta:
                anomalies.append(_anomaly(
                    kind,
                    "error",
                    message,
                    frame,
                    count_delta=delta,
                    cumulative_count=int(frame[counter]),
                ))
        previous = frame

    for run in _contiguous_runs(
            frames,
            lambda frame: frame["depth_updated"] and frame["range_collapsed"] > 0.5):
        anomalies.append(_anomaly(
            "range_collapsed",
            "error",
            f"The current raw depth range was collapsed for {len(run)} update(s).",
            run[0],
            end_frame_id=run[-1]["frame_id"],
        ))
    return anomalies


def analyze_trace(
    header: dict[str, Any],
    frames: list[dict[str, Any]],
    timing: dict[str, Any],
    source_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build aggregate and anomaly results from validated samples."""
    anomalies = find_anomalies(frames)

    first, last = frames[0], frames[-1]
    config = dict(header["config"])
    valid_fractions = [
        float(frame["valid_depth_fraction"]) for frame in frames
        if frame["depth_updated"]
    ]
    summary = {
        "schema": 2,
        "source_name": source_name,
        "generated_utc": _datetime.datetime.now(
            _datetime.timezone.utc).isoformat(timespec="seconds"),
        "trace_contract": {
            "schema": header["schema"],
            "source": header["source"],
            "capture": header["capture"],
            "fields": header["fields"],
        },
        "config": config,
        "timing": timing,
        "frame_count": len(frames),
        "depth_update_count": sum(bool(frame["depth_updated"]) for frame in frames),
        "duration_seconds": max(
            0.0, _frame_end(last) - float(first["timestamp_seconds"])),
        "min_valid_depth_fraction": min(valid_fractions) if valid_fractions else None,
        "empty_raw_count": int(last["empty_raw_count"]),
        "collapsed_raw_count": int(last["collapsed_raw_count"]),
        "anomaly_count": len(anomalies),
        "anomaly_counts": {
            kind: sum(item["kind"] == kind for item in anomalies)
            for kind in sorted({item["kind"] for item in anomalies})
        },
        "anomalies": anomalies,
    }
    return summary, anomalies


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, float):
        return format(value, ".10g")
    return value


FRAME_CSV_FIELDS = (
    "frame_id",
    "source_index",
    "source_pts_seconds",
    "timestamp_seconds",
    "duration_seconds",
    "depth_updated",
    *FIELD_NAMES,
)


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in writer.fieldnames})


def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}{suffix}"
    return str(value)


def _sample_for_chart(frames: list[dict[str, Any]], limit: int = 1200) -> list[dict[str, Any]]:
    if len(frames) <= limit:
        return frames
    step = (len(frames) - 1) / (limit - 1)
    indices = sorted({round(index * step) for index in range(limit)})
    return [frames[index] for index in indices]


def _chart_svg(
    frames: list[dict[str, Any]],
    key: str,
    label: str,
    color: str,
) -> str:
    sampled = _sample_for_chart(frames)
    points = [
        (float(frame["timestamp_seconds"]), float(frame[key]))
        for frame in sampled
        if frame.get(key) is not None
    ]
    if not points:
        return (
            f'<section class="chart"><h3>{html.escape(label)}</h3>'
            '<p class="muted">No applicable samples.</p></section>'
        )
    width, height = 960.0, 220.0
    pad_l, pad_r, pad_t, pad_b = 56.0, 18.0, 18.0, 34.0
    min_x, max_x = points[0][0], points[-1][0]
    min_y = min(value for _time, value in points)
    max_y = max(value for _time, value in points)
    if max_x <= min_x:
        max_x = min_x + 1.0
    if max_y <= min_y:
        margin = max(abs(min_y) * 0.05, 0.01)
        min_y -= margin
        max_y += margin

    def x(value: float) -> float:
        return pad_l + (value - min_x) / (max_x - min_x) * (width - pad_l - pad_r)

    def y(value: float) -> float:
        return pad_t + (max_y - value) / (max_y - min_y) * (height - pad_t - pad_b)

    polyline = " ".join(f"{x(time):.2f},{y(value):.2f}" for time, value in points)
    return (
        f'<section class="chart"><h3>{html.escape(label)}</h3>'
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="{html.escape(label)} from {_fmt(min_y)} to {_fmt(max_y)}">'
        f'<line class="axis" x1="{pad_l}" x2="{pad_l}" y1="{pad_t}" '
        f'y2="{height - pad_b}"/><line class="axis" x1="{pad_l}" '
        f'x2="{width - pad_r}" y1="{height - pad_b}" y2="{height - pad_b}"/>'
        f'<polyline points="{polyline}" fill="none" stroke="{color}" '
        'stroke-width="2" vector-effect="non-scaling-stroke"/>'
        f'<text x="6" y="{pad_t + 10}">{max_y:.3g}</text>'
        f'<text x="6" y="{height - pad_b}">{min_y:.3g}</text>'
        f'<text x="{pad_l}" y="{height - 8}">{min_x:.2f}s</text>'
        f'<text text-anchor="end" x="{width - pad_r}" y="{height - 8}">'
        f'{max_x:.2f}s</text></svg></section>'
    )


def _write_html(
    path: Path,
    summary: dict[str, Any],
    frames: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
) -> None:
    config = summary["config"]
    anomaly_rows = "".join(
        "<tr>"
        f"<td><span class=\"severity {html.escape(item['severity'])}\">"
        f"{html.escape(item['severity'])}</span></td>"
        f"<td><code>{html.escape(item['kind'])}</code></td>"
        f"<td>{_fmt(item.get('timestamp_seconds'))}</td>"
        f"<td>{html.escape(item['message'])}</td>"
        "</tr>"
        for item in anomalies
    ) or '<tr><td colspan="4" class="ok">No configured anomalies detected.</td></tr>'
    charts = "".join((
        _chart_svg(
            frames, "valid_depth_fraction", "Valid depth fraction", "#9be564"),
        _chart_svg(
            frames, "effective_raw_range_width", "Effective raw range width", "#64d3ff"),
        _chart_svg(
            frames, "depth_ready", "Depth ready", "#f5b942"),
        _chart_svg(
            frames, "range_collapsed", "Raw range collapsed", "#f06e9c"),
    ))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Host SBS whole-clip depth-health report</title>
<style>
:root{{--bg:#0b1017;--panel:#131b25;--line:#283747;--ink:#edf6ff;--muted:#94a8ba;
--accent:#64d3ff;--ok:#74d89b;--warn:#f5b942;--error:#ff7383}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.48 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1280px;margin:auto;padding:28px}}h1{{margin:0 0 4px;font-size:28px}}
h2{{margin:32px 0 12px}}h3{{margin:0 0 10px;font-size:14px}}code{{color:#b9e7ff}}
.muted{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);
gap:12px;margin-top:20px}}.card,.chart,.notice{{background:var(--panel);
border:1px solid var(--line);border-radius:12px;padding:16px}}.card b{{display:block;
font-size:23px;margin-top:4px}}.charts{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.chart svg{{display:block;width:100%;height:auto;background:#0d141d;border-radius:7px}}
.axis{{stroke:#526579;stroke-width:1}}
svg text{{fill:#91a7b9;font:11px ui-monospace,Consolas,monospace}}
.notice{{border-left:4px solid var(--warn)}}table{{width:100%;border-collapse:collapse;
background:var(--panel);border:1px solid var(--line)}}th,td{{padding:9px 10px;
border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{color:#b9c9d8}}
.severity{{display:inline-block;border-radius:99px;padding:2px 8px;font-size:11px;
text-transform:uppercase}}.severity.warning{{background:#4a3916;color:#ffd979}}
.severity.error{{background:#4b2029;color:#ff9dab}}.ok{{color:var(--ok)}}
@media(max-width:850px){{.cards,.charts{{grid-template-columns:1fr}}main{{padding:18px}}}}
</style></head><body><main>
<h1>Host SBS whole-clip depth-health report</h1>
<p class="muted">{html.escape(summary['source_name'])} · generated
{html.escape(summary['generated_utc'])}</p>
<div class="cards">
<div class="card">Frames<b>{summary['frame_count']}</b></div>
<div class="card">Duration<b>{_fmt(summary['duration_seconds'], 2, ' s')}</b></div>
<div class="card">Depth updates<b>{summary['depth_update_count']}</b></div>
<div class="card">Flagged intervals/events<b>{summary['anomaly_count']}</b></div>
</div>
<h2>Trace configuration</h2>
<div class="notice"><b>{html.escape(config['model'])}</b> · configured pop
{_fmt(config['pop_strength'], 2)} · depth interval {config['depth_reuse_interval']}
<p>This trace reports V2 depth and model health. It does not infer or override geometry.</p></div>
<h2>Depth-health timeline</h2>
<div class="charts">{charts}</div>
<h2>Anomalies</h2><div style="overflow:auto"><table><thead><tr>
<th>Severity</th><th>Kind</th><th>Time (s)</th><th>Observation</th>
</tr></thead><tbody>{anomaly_rows}</tbody></table></div>
<p class="muted">This report is diagnostic. It does not mutate Sunshine 3D configuration and
does not replace the authenticated core/extended evaluator gate.</p>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def generate_outputs(
    trace_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    timeline: Any = None,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Validate a trace and emit ``frames.csv``, JSON, and HTML.

    This is the stable wrapper-facing entry point. ``timeline`` may be the wrapper's schema-1
    object, a direct list of timing rows, a path to either JSON form, or ``None``.
    """
    header, raw_frames = load_trace(trace_path)
    frames, timing = join_timeline(raw_frames, timeline)
    name = source_name or Path(trace_path).parent.name or Path(trace_path).stem
    summary, anomalies = analyze_trace(header, frames, timing, name)

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "frames.csv", FRAME_CSV_FIELDS, frames)
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    _write_html(output / "report.html", summary, frames, anomalies)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Host SBS whole-clip depth-health report.")
    parser.add_argument("trace", help="adaptive_state.jsonl from the native harness")
    parser.add_argument("--out", required=True, help="report output directory")
    parser.add_argument("--timeline", help="optional schema-1 source PTS manifest")
    parser.add_argument("--source-name", help="human-readable source-video label")
    args = parser.parse_args()
    try:
        summary = generate_outputs(
            args.trace,
            args.out,
            timeline=args.timeline,
            source_name=args.source_name,
        )
    except (OSError, TraceContractError) as exc:
        parser.error(str(exc))
    print(f"wrote {Path(args.out) / 'report.html'}")
    print(
        f"{summary['frame_count']} frames, {summary['depth_update_count']} depth updates, "
        f"{summary['anomaly_count']} flagged observations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
