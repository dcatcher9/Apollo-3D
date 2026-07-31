#!/usr/bin/env python3
"""Score a strict Scene Controller shadow trace against exact browser-scene labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__:
    from . import browser_scene_compositor as compositor
    from . import scene_controller_contract as contract
    from . import scene_controller_trace as trace
else:
    import browser_scene_compositor as compositor
    import scene_controller_contract as contract
    import scene_controller_trace as trace


EVALUATOR = "scene-controller-shadow-eval-v1"
REPORT_SCHEMA = 1
THRESHOLDS_SCHEMA = 1
DEFAULT_THRESHOLDS = (
    Path(__file__).resolve().parent /
    "datasets" /
    "browser_scene_thresholds_v1.json"
)
THRESHOLD_KEYS = {
    "schema",
    "name",
    "analysis_canvas_size",
    "event_tolerance_updates",
    "max_acquisition_ms",
    "max_inactive_frames_after_acquisition",
    "max_roi_jitter_analysis_cells_p95",
    "max_unsafe_pixels",
    "min_active_layout_accuracy",
    "min_roi_iou_median",
    "min_roi_iou_p05",
    "min_target_coverage_p05",
}
GENERATED_CONTRACT_KEYS = {
    "schema",
    "generator",
    "scenario",
    "recipe_sha256",
    "recipe",
    "recipe_file_sha256",
    "frame_count",
    "labels",
    "labels_sha256",
    "timeline",
    "timeline_sha256",
    "artifact",
    "artifact_sha256",
    "rgb_sha256",
}
ARTIFACT_KEYS = {
    "schema",
    "generator",
    "scenario",
    "recipe_sha256",
    "recipe_file_sha256",
    "timeline_sha256",
    "sequence_npz_sha256",
    "arrays",
    "semantic_classes",
    "storage_note",
    "emitted_frame_files",
}
SHA256_FIELDS = {
    "recipe_sha256",
    "recipe_file_sha256",
    "labels_sha256",
    "timeline_sha256",
    "artifact_sha256",
    "rgb_sha256",
}

RULE_INDEX = {
    name: index for index, name in enumerate(contract.RULE_STATE_NAMES)
}
STATE_KIND = contract.CONTRACT["enums"]["state_kind"]
LAYOUT_DECISION = contract.CONTRACT["enums"]["layout_decision"]
EVENT_DECISION = contract.CONTRACT["enums"]["event_decision"]
STATE_KIND_NAMES = {value: name for name, value in STATE_KIND.items()}
LAYOUT_DECISION_NAMES = {
    value: name for name, value in LAYOUT_DECISION.items()
}
EVENT_DECISION_NAMES = {
    value: name for name, value in EVENT_DECISION.items()
}
UINT32_MAX = (1 << 32) - 1
RESOLVER_COUNTER_MAX = UINT32_MAX - 1
ROI_STATE_KINDS = frozenset({"video", "content"})
ROI_LAYOUTS = frozenset({"primary_video", "content_collage"})
FULL_FRAME_LAYOUTS = frozenset({
    "identity_fullscreen",
    "no_target",
    "ambiguous",
})
FULL_FRAME_COMMITTED_LAYOUTS = frozenset({
    "identity_fullscreen",
    "no_target",
})
STATE_FLAG_BITS = contract.CONTRACT["flag_bits"]["state_flags"]
RESET_FLAG_BITS = contract.CONTRACT["flag_bits"]["reset_flags"]
PROMOTION_FLAG_BITS = contract.CONTRACT["flag_bits"]["promotion_flags"]
HISTORY_FLAG_BITS = contract.CONTRACT["flag_bits"]["history_flags"]
STATE_INITIALIZED = 1 << STATE_FLAG_BITS["initialized"]
STATE_ROI_LOCKED = 1 << STATE_FLAG_BITS["roi_locked"]
STATE_SCROLL_HOLD = 1 << STATE_FLAG_BITS["scroll_hold_active"]
STATE_FALLBACK = 1 << STATE_FLAG_BITS["fallback_active"]
STATE_CUT_STABLE_MASK = (
    (1 << STATE_FLAG_BITS["cut_armed"]) |
    (1 << STATE_FLAG_BITS["cut_accepted_ever"])
)
RESET_LAYOUT = 1 << RESET_FLAG_BITS["layout"]
RESET_GEOMETRY = 1 << RESET_FLAG_BITS["geometry"]
RESET_BACKEND = 1 << RESET_FLAG_BITS["backend"]
RESET_DISPLAY_OR_HDR = 1 << RESET_FLAG_BITS["display_or_hdr"]
CONTROLLER_RESET_MASK = (
    RESET_LAYOUT |
    RESET_GEOMETRY |
    RESET_BACKEND |
    RESET_DISPLAY_OR_HDR
)
PROMOTE_LAYOUT = 1 << PROMOTION_FLAG_BITS["layout_history"]
PROMOTE_ROI = 1 << PROMOTION_FLAG_BITS["roi"]
HISTORY_LAYOUT_READ = 1 << HISTORY_FLAG_BITS["layout_read_bank"]
HISTORY_LAYOUT_WRITE = 1 << HISTORY_FLAG_BITS["layout_write_bank"]
HISTORY_DEPTH_READ = 1 << HISTORY_FLAG_BITS["depth_read_bank"]
SCROLL_PROMOTION_FLAGS = PROMOTE_LAYOUT
SCROLL_HISTORY_FLAGS = (
    HISTORY_LAYOUT_READ |
    HISTORY_LAYOUT_WRITE |
    HISTORY_DEPTH_READ
)
# Recipe FPS is the cadence of accepted captured-stream presentations, never
# the cadence of video embedded inside those presentations. Every duration in
# this evaluator is stream wall time derived from that timeline.
#
SCROLL_ENTRY_SECONDS = 0.05
SCROLL_RELEASE_SECONDS = 0.12
SCROLL_FROZEN_FIELDS = (
    "committed_roi_age_s",
    "layout_decision",
    "layout_confidence",
    "event_decision",
    "event_confidence",
    "pop_action",
    "pop_strength",
    "pop_confidence",
    "zero_plane_decision",
    "zero_plane_confidence",
    "scene_age_s",
    "cut_cooldown_s",
    "cut_rearm_evidence",
)
SCROLL_CLEARED_FIELDS = {
    "acquisition_roi_x0": 0.0,
    "acquisition_roi_y0": 0.0,
    "acquisition_roi_x1": 1.0,
    "acquisition_roi_y1": 1.0,
    "acquisition_score": 0.0,
    "acquisition_dwell_s": 0.0,
    "acquisition_layout": float(LAYOUT_DECISION["no_target"]),
    "acquisition_valid": 0.0,
    "challenger_roi_x0": 0.0,
    "challenger_roi_y0": 0.0,
    "challenger_roi_x1": 1.0,
    "challenger_roi_y1": 1.0,
    "challenger_score": 0.0,
    "challenger_dwell_s": 0.0,
    "challenger_layout": float(LAYOUT_DECISION["no_target"]),
    "challenger_valid": 0.0,
    "seconds_since_layout_evidence": 0.0,
}
SCROLL_ROI_RETAINED_FIELDS = (
    "committed_roi_x0",
    "committed_roi_y0",
    "committed_roi_x1",
    "committed_roi_y1",
    "committed_roi_confidence",
    "committed_mask_confidence",
    "committed_layout",
)


class SceneControllerEvalError(ValueError):
    """The labels, trace, or threshold policy cannot support a valid score."""


def _stream_update_seconds(
    rendered: compositor.RenderedSequence,
) -> float:
    """Return one accepted stream-presentation interval in seconds."""
    return (
        rendered.recipe["fps_den"] /
        rendered.recipe["fps_num"]
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(
    value: Any,
    *,
    where: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if (
        not isinstance(value, (int, float)) or isinstance(value, bool) or
        not math.isfinite(float(value))
    ):
        raise SceneControllerEvalError(f"{where} must be a finite JSON number")
    result = float(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise SceneControllerEvalError(
            f"{where} must be in [{minimum}, "
            f"{maximum if maximum is not None else 'infinity'}]"
        )
    return result


def validate_thresholds(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != THRESHOLD_KEYS:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise SceneControllerEvalError(
            f"threshold policy must have exact keys {sorted(THRESHOLD_KEYS)}, got {actual}"
        )
    if (
        not _is_int(value["schema"]) or
        value["schema"] != THRESHOLDS_SCHEMA
    ):
        raise SceneControllerEvalError(
            f"threshold policy schema must be {THRESHOLDS_SCHEMA}"
        )
    if not isinstance(value["name"], str) or not value["name"]:
        raise SceneControllerEvalError("threshold policy name must be non-empty")
    for name in (
        "analysis_canvas_size",
        "event_tolerance_updates",
        "max_inactive_frames_after_acquisition",
        "max_unsafe_pixels",
    ):
        if not _is_int(value[name]) or value[name] < 0:
            raise SceneControllerEvalError(
                f"threshold policy {name} must be a non-negative JSON integer"
            )
    if value["analysis_canvas_size"] < 1:
        raise SceneControllerEvalError(
            "threshold policy analysis_canvas_size must be positive"
        )
    _number(value["max_acquisition_ms"], where="max_acquisition_ms", minimum=0.0)
    _number(
        value["max_roi_jitter_analysis_cells_p95"],
        where="max_roi_jitter_analysis_cells_p95",
        minimum=0.0,
    )
    for name in (
        "min_active_layout_accuracy",
        "min_roi_iou_median",
        "min_roi_iou_p05",
        "min_target_coverage_p05",
    ):
        _number(value[name], where=name, minimum=0.0, maximum=1.0)
    if value["min_roi_iou_p05"] > value["min_roi_iou_median"]:
        raise SceneControllerEvalError(
            "min_roi_iou_p05 cannot exceed min_roi_iou_median"
        )
    return dict(value)


def load_thresholds(
    path: os.PathLike[str] | str = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(
                stream, object_pairs_hook=compositor._reject_duplicate_keys
            )
    except (OSError, json.JSONDecodeError, compositor.CompositorError) as exc:
        raise SceneControllerEvalError(
            f"cannot read scene-controller thresholds: {exc}"
        ) from exc
    return validate_thresholds(value)


def _state(frame: Mapping[str, Any]) -> dict[str, Any]:
    state = frame.get("rule_state")
    if (
        not isinstance(state, list) or
        len(state) != len(contract.RULE_STATE_NAMES)
    ):
        raise SceneControllerEvalError(
            f"frame {frame.get('source_index')} has no exact rule_state"
        )
    return {name: state[index] for name, index in RULE_INDEX.items()}


def _enum_name(
    value: Any,
    names: Mapping[int, str],
    *,
    where: str,
) -> str:
    if (
        not isinstance(value, (int, float)) or isinstance(value, bool) or
        not math.isfinite(float(value)) or float(value) != math.trunc(float(value))
    ):
        raise SceneControllerEvalError(f"{where} must be an integer-valued number")
    integer = int(value)
    if integer not in names:
        raise SceneControllerEvalError(f"{where} contains unknown enum value {integer}")
    return names[integer]


def _binary(value: Any, *, where: str) -> bool:
    if value not in {0, 0.0, 1, 1.0} or isinstance(value, bool):
        raise SceneControllerEvalError(f"{where} must be exactly 0.0 or 1.0")
    return bool(value)


def _uint32(value: Any, *, where: str) -> int:
    if not _is_int(value) or not 0 <= value <= UINT32_MAX:
        raise SceneControllerEvalError(
            f"{where} must be a uint32 JSON integer"
        )
    return value


def _layout_segments(
    timeline: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    start = 0
    while start < len(timeline):
        layout = timeline[start]["expected_layout"]
        end = start + 1
        while (
            end < len(timeline) and
            timeline[end]["expected_layout"] == layout and
            not timeline[end]["geometry_reset"]
        ):
            end += 1
        segments.append({
            "index": len(segments),
            "layout": layout,
            "start_index": start,
            "end_index": end,
            "start_frame": start + 1,
            "end_frame": end,
        })
        start = end
    return segments


def _roi_layout_pair_valid(state_kind: str, layout: str) -> bool:
    if state_kind == "video":
        return layout == "primary_video"
    if state_kind == "content":
        return layout == "content_collage"
    return False


def _roi_mask(
    state: Mapping[str, Any],
    *,
    width: int,
    height: int,
    where: str,
) -> tuple[np.ndarray, list[float]]:
    normalized = [
        float(state["committed_roi_x0"]),
        float(state["committed_roi_y0"]),
        float(state["committed_roi_x1"]),
        float(state["committed_roi_y1"]),
    ]
    if (
        not all(math.isfinite(value) for value in normalized) or
        not (
            0.0 <= normalized[0] < normalized[2] <= 1.0 and
            0.0 <= normalized[1] < normalized[3] <= 1.0
        )
    ):
        raise SceneControllerEvalError(
            f"{where} has invalid normalized committed ROI {normalized}"
        )
    x0 = int(math.floor(normalized[0] * width + 1e-7))
    y0 = int(math.floor(normalized[1] * height + 1e-7))
    x1 = int(math.ceil(normalized[2] * width - 1e-7))
    y1 = int(math.ceil(normalized[3] * height - 1e-7))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, width), min(y1, height)
    if x0 >= x1 or y0 >= y1:
        raise SceneControllerEvalError(f"{where} rasterizes to an empty ROI")
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[y0:y1, x0:x1] = True
    return mask, normalized


def _fraction(intersection: int, denominator: int) -> float:
    return float(intersection / denominator) if denominator else 0.0


def _quantile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _match_events(
    expected: Sequence[int],
    predicted: Sequence[int],
    tolerance: int,
) -> tuple[list[dict[str, int]], list[int], list[int]]:
    remaining = set(predicted)
    matches: list[dict[str, int]] = []
    missed: list[int] = []
    for expected_frame in expected:
        candidates = sorted(
            (
                (abs(actual - expected_frame), actual)
                for actual in remaining
                if abs(actual - expected_frame) <= tolerance
            ),
            key=lambda item: (item[0], item[1]),
        )
        if not candidates:
            missed.append(expected_frame)
            continue
        actual = candidates[0][1]
        remaining.remove(actual)
        matches.append({"expected": expected_frame, "predicted": actual})
    return matches, missed, sorted(remaining)


def _match_event_windows(
    expected: Sequence[Sequence[int]],
    predicted: Sequence[int],
) -> tuple[list[dict[str, Any]], list[list[int]], list[int]]:
    remaining = set(predicted)
    matches: list[dict[str, Any]] = []
    missed: list[list[int]] = []
    for raw_window in expected:
        earliest, latest = int(raw_window[0]), int(raw_window[1])
        window = [earliest, latest]
        center = (earliest + latest) // 2
        candidates = sorted(
            (
                (abs(actual - center), actual)
                for actual in remaining
                if earliest <= actual <= latest
            ),
            key=lambda item: (item[0], item[1]),
        )
        if not candidates:
            missed.append(window)
            continue
        actual = candidates[0][1]
        remaining.remove(actual)
        matches.append({"expected": window, "predicted": actual})
    return matches, missed, sorted(remaining)


def evaluate(
    rendered: compositor.RenderedSequence,
    frames: Sequence[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, Any] | None = None,
    depth_reuse_interval: int = 1,
    trace_frame_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    policy = validate_thresholds(
        dict(thresholds) if thresholds is not None else load_thresholds()
    )

    if depth_reuse_interval != 1:
        raise SceneControllerEvalError(
            "browser-scene shadow qualification requires depth_reuse_interval=1"
        )
    if len(frames) != len(rendered.timeline):
        raise SceneControllerEvalError(
            f"trace has {len(frames)} frames; labels have {len(rendered.timeline)}"
        )
    if trace_frame_ids is None:
        trace_frame_ids = [
            item["source_frame_id"] for item in rendered.timeline
        ]
    if (
        len(trace_frame_ids) != len(frames) or
        any(not isinstance(item, str) or not item.isdigit()
            for item in trace_frame_ids)
    ):
        raise SceneControllerEvalError(
            "trace_frame_ids must contain one decimal identity per frame"
        )
    width = int(rendered.recipe["width"])
    height = int(rendered.recipe["height"])
    expected_target = rendered.arrays["target"].copy()
    exclusion = rendered.arrays["exclusion"]
    if (
        expected_target.shape != (len(frames), height, width) or
        exclusion.shape != expected_target.shape
    ):
        raise SceneControllerEvalError("browser-scene dense labels have invalid shape")

    tolerance = policy["event_tolerance_updates"]
    reset_frames = rendered.recipe["events"]["geometry_reset_frames"]
    relocation_frames = rendered.recipe["events"].get(
        "relocation_frames",
        [],
    )
    relocation_windows = rendered.recipe["events"].get(
        "relocation_acceptance_windows",
        [],
    )
    for index, expected in enumerate(rendered.timeline):
        if (
            expected.get("relocation_observation_generation", 0) !=
            expected.get("relocation_generation", 0)
        ):
            rect = expected["expected_roi_px"]
            expected_target[index] = False
            if rect is not None:
                x0, y0, x1, y1 = rect
                expected_target[index, y0:y1, x0:x1] = True

    def near_geometry(frame_number: int) -> bool:
        return (
            any(
                abs(frame_number - reset) <= tolerance
                for reset in reset_frames
            ) or
            any(
                earliest <= frame_number <= latest
                for earliest, latest in relocation_windows
            )
        )

    frame_rows: list[dict[str, Any]] = []
    decoded_rows: list[dict[str, Any]] = []
    active_indices: list[int] = []
    ious: list[float] = []
    coverages: list[float] = []
    layout_matches: list[bool] = []
    unsafe_frames: list[dict[str, int]] = []
    predicted_hard_cuts: list[int] = []
    predicted_exposure: list[int] = []
    predicted_scroll: list[int] = []
    predicted_geometry: list[int] = []
    invalid_output_frames: list[dict[str, Any]] = []
    invalid_event_attempts: list[dict[str, Any]] = []
    invalid_generation_attempts: list[dict[str, Any]] = []
    authority_rejections: list[dict[str, Any]] = []
    state_flag_inconsistencies: list[dict[str, Any]] = []
    generation_changes: list[int] = []
    generation_transitions: list[dict[str, Any]] = []
    generation_resets: list[dict[str, Any]] = []
    geometry_generation_frames: list[int] = []
    roi_promotion_frames: list[int] = []
    committed_geometry_mismatches: list[dict[str, Any]] = []
    generation_unit_violations: list[dict[str, Any]] = []
    epoch_has_positive_generation = False
    prior_current_generation: int | None = None
    prior_current_update_count: int | None = None
    prior_current_backend_generation: int | None = None
    prior_current_committed_geometry: tuple[Any, ...] | None = None
    prior_active_roi: list[float] | None = None
    prior_active_layout: str | None = None
    jitter_cells: list[float] = []

    for index, (frame, expected) in enumerate(zip(frames, rendered.timeline)):
        frame_number = index + 1
        if frame.get("source_index") != index:
            raise SceneControllerEvalError(
                f"trace source_index {frame.get('source_index')} is not {index}"
            )
        if frame.get("frame_id") != trace_frame_ids[index]:
            raise SceneControllerEvalError(
                f"trace frame_id {frame.get('frame_id')!r} does not match "
                f"native identity {trace_frame_ids[index]!r}"
            )
        state = _state(frame)
        output_valid = _binary(
            state["output_valid"], where=f"frame {index + 1} output_valid"
        )
        state_kind = _enum_name(
            state["state_kind"],
            STATE_KIND_NAMES,
            where=f"frame {index + 1} state_kind",
        )
        layout = _enum_name(
            state["committed_layout"],
            LAYOUT_DECISION_NAMES,
            where=f"frame {index + 1} committed_layout",
        )
        event = _enum_name(
            state["event_decision"],
            EVENT_DECISION_NAMES,
            where=f"frame {frame_number} event_decision",
        )
        layout_decision = _enum_name(
            state["layout_decision"],
            LAYOUT_DECISION_NAMES,
            where=f"frame {frame_number} layout_decision",
        )
        backend_generation = _uint32(
            state["backend_generation"],
            where=f"frame {frame_number} backend_generation",
        )
        generation = _uint32(
            state["roi_generation"],
            where=f"frame {frame_number} roi_generation",
        )
        update_count = _uint32(
            state["update_count"],
            where=f"frame {frame_number} update_count",
        )
        state_flags = _uint32(
            state["state_flags"],
            where=f"frame {frame_number} state_flags",
        )
        reset_flags = _uint32(
            state["reset_flags"],
            where=f"frame {frame_number} reset_flags",
        )
        promotion_flags = _uint32(
            state["promotion_flags"],
            where=f"frame {frame_number} promotion_flags",
        )
        history_flags = _uint32(
            state["history_flags"],
            where=f"frame {frame_number} history_flags",
        )

        initialized = (state_flags & STATE_INITIALIZED) != 0
        roi_locked = (state_flags & STATE_ROI_LOCKED) != 0
        fallback_active = (state_flags & STATE_FALLBACK) != 0
        scroll_hold_flag = (state_flags & STATE_SCROLL_HOLD) != 0
        backend_changed = (
            prior_current_backend_generation is not None and
            backend_generation != prior_current_backend_generation
        )
        initial_epoch = prior_current_backend_generation is None
        controller_reset = (
            initial_epoch or
            (reset_flags & CONTROLLER_RESET_MASK) != 0 or
            backend_changed
        )
        current_reasons: list[str] = []
        if not output_valid:
            current_reasons.append("output_invalid")
        if not initialized:
            current_reasons.append("uninitialized")
        if fallback_active:
            current_reasons.append("fallback_active")
        if backend_generation == 0:
            current_reasons.append("backend_generation_zero")
        if update_count == 0:
            current_reasons.append("update_count_zero")
        if not current_reasons:
            if prior_current_backend_generation is None:
                if update_count != 1:
                    current_reasons.append(
                        "initial_update_count_not_one"
                    )
            elif backend_generation < prior_current_backend_generation:
                current_reasons.append("backend_generation_decreased")
            elif controller_reset:
                if update_count != 1:
                    current_reasons.append(
                        "reset_update_count_not_one"
                    )
            elif (
                prior_current_update_count is not None and
                update_count != min(
                    prior_current_update_count + 1,
                    RESOLVER_COUNTER_MAX,
                )
            ):
                current_reasons.append("update_count_sequence_mismatch")
        if not current_reasons and controller_reset:
            try:
                reset_roi = [
                    float(state["committed_roi_x0"]),
                    float(state["committed_roi_y0"]),
                    float(state["committed_roi_x1"]),
                    float(state["committed_roi_y1"]),
                ]
                reset_confidence = float(
                    state["committed_roi_confidence"]
                )
                reset_mask_confidence = float(
                    state["committed_mask_confidence"]
                )
            except (TypeError, ValueError):
                current_reasons.append("reset_committed_state_malformed")
            else:
                if generation != 0:
                    current_reasons.append(
                        "reset_roi_generation_not_zero"
                    )
                if roi_locked:
                    current_reasons.append("reset_roi_still_locked")
                if state_kind != "full_frame":
                    current_reasons.append(
                        "reset_state_kind_not_full_frame"
                    )
                if layout != "no_target":
                    current_reasons.append(
                        "reset_committed_layout_not_no_target"
                    )
                if (
                    reset_roi != [0.0, 0.0, 1.0, 1.0] or
                    not math.isfinite(reset_confidence) or
                    reset_confidence != 0.0 or
                    not math.isfinite(reset_mask_confidence) or
                    reset_mask_confidence != 0.0
                ):
                    current_reasons.append(
                        "reset_committed_roi_not_canonical"
                    )
        current_output = not current_reasons
        if not current_output:
            invalid_output_frames.append({
                "frame": frame_number,
                "reasons": current_reasons,
            })
        if controller_reset and generation != 0:
            generation_unit_violations.append({
                "frame": frame_number,
                "from": prior_current_generation,
                "to": generation,
                "reset_flags": reset_flags,
                "reason": "reset_generation_not_zero",
            })

        hold_state = scroll_hold_flag
        if state_kind == "full_frame" and roi_locked:
            state_flag_inconsistencies.append({
                "frame": frame_number,
                "reason": "full_frame_has_roi_lock",
                "state_kind": state_kind,
                "state_flags": state_flags,
            })
        if current_output and state_kind == "full_frame":
            try:
                full_frame_roi = [
                    float(state["committed_roi_x0"]),
                    float(state["committed_roi_y0"]),
                    float(state["committed_roi_x1"]),
                    float(state["committed_roi_y1"]),
                ]
                full_frame_confidence = float(
                    state["committed_roi_confidence"]
                )
                full_frame_mask_confidence = float(
                    state["committed_mask_confidence"]
                )
            except (TypeError, ValueError):
                state_flag_inconsistencies.append({
                    "frame": frame_number,
                    "reason": "full_frame_committed_state_malformed",
                })
            else:
                if (
                    layout not in FULL_FRAME_COMMITTED_LAYOUTS or
                    full_frame_roi != [0.0, 0.0, 1.0, 1.0] or
                    not math.isfinite(full_frame_confidence) or
                    full_frame_confidence != 0.0 or
                    not math.isfinite(full_frame_mask_confidence) or
                    full_frame_mask_confidence != 0.0
                ):
                    state_flag_inconsistencies.append({
                        "frame": frame_number,
                        "reason": (
                            "full_frame_committed_state_not_canonical"
                        ),
                        "committed_layout": layout,
                        "committed_roi": full_frame_roi,
                        "committed_roi_confidence": (
                            full_frame_confidence
                        ),
                        "committed_mask_confidence": (
                            full_frame_mask_confidence
                        ),
                    })

        pair_valid = _roi_layout_pair_valid(state_kind, layout)
        authority_reasons: list[str] = []
        if not current_output:
            authority_reasons.append("output_not_current")
        if not roi_locked:
            authority_reasons.append("roi_not_locked")
        if generation == 0:
            authority_reasons.append("roi_generation_zero")
        if state_kind not in ROI_STATE_KINDS:
            authority_reasons.append("state_kind_not_roi")
        if not pair_valid:
            authority_reasons.append("committed_layout_state_mismatch")

        active = not authority_reasons
        roi: list[float] | None = None
        predicted_mask: np.ndarray | None = None
        iou = 0.0
        coverage = 0.0
        unsafe_pixels = 0
        # A committed target remains authoritative while candidate evidence
        # legitimately reports a transient state such as ``scroll``. Score
        # target retention from the committed layout; full-frame segments
        # separately require the current decision to match their abstention.
        layout_match = layout == expected["expected_layout"]
        if active:
            try:
                predicted_mask, roi = _roi_mask(
                    state,
                    width=width,
                    height=height,
                    where=f"frame {frame_number}",
                )
            except SceneControllerEvalError as exc:
                active = False
                authority_reasons.append("committed_roi_invalid")
                authority_rejections.append({
                    "frame": frame_number,
                    "reasons": authority_reasons,
                    "detail": str(exc),
                })
        roi_attempted = roi_locked or state_kind in ROI_STATE_KINDS
        if current_output and roi_attempted and not active:
            if not any(
                rejection["frame"] == frame_number
                for rejection in authority_rejections
            ):
                authority_rejections.append({
                    "frame": frame_number,
                    "reasons": authority_reasons,
                })

        if active:
            assert predicted_mask is not None and roi is not None
            active_indices.append(index)
            unsafe_pixels = int(np.count_nonzero(
                predicted_mask & exclusion[index]
            ))
            if unsafe_pixels:
                unsafe_frames.append({
                    "frame": frame_number,
                    "pixels": unsafe_pixels,
                })
            if expected["expected_layout"] in ROI_LAYOUTS:
                target = expected_target[index]
                intersection = int(np.count_nonzero(
                    predicted_mask & target
                ))
                union = int(np.count_nonzero(predicted_mask | target))
                iou = _fraction(intersection, union)
                coverage = _fraction(
                    intersection, int(np.count_nonzero(target))
                )
                ious.append(iou)
                coverages.append(coverage)
                layout_matches.append(layout_match)
                if (
                    prior_active_roi is not None and
                    prior_active_layout == expected["expected_layout"] and
                    not near_geometry(frame_number)
                ):
                    size = float(policy["analysis_canvas_size"])
                    jitter_cells.append(max(
                        abs(roi[0] - prior_active_roi[0]) * size,
                        abs(roi[2] - prior_active_roi[2]) * size,
                        abs(roi[1] - prior_active_roi[1]) * size,
                        abs(roi[3] - prior_active_roi[3]) * size,
                    ))
                prior_active_roi = roi
                prior_active_layout = expected["expected_layout"]
            else:
                prior_active_roi = None
                prior_active_layout = None
        else:
            prior_active_roi = None
            prior_active_layout = None

        if current_output:
            committed_geometry = (
                layout,
                state["committed_roi_x0"],
                state["committed_roi_y0"],
                state["committed_roi_x1"],
                state["committed_roi_y1"],
            )
            epoch_start = (
                prior_current_backend_generation is None or
                controller_reset
            )
            if (
                not epoch_start and
                prior_current_generation is not None and
                prior_current_committed_geometry is not None
            ):
                generation_changed = (
                    generation != prior_current_generation
                )
                committed_geometry_changed = (
                    committed_geometry !=
                    prior_current_committed_geometry
                )
                if generation_changed != committed_geometry_changed:
                    committed_geometry_mismatches.append({
                        "frame": frame_number,
                        "generation_changed": generation_changed,
                        "committed_geometry_changed": (
                            committed_geometry_changed
                        ),
                        "generation": generation,
                        "prior_generation": prior_current_generation,
                    })
            if (promotion_flags & PROMOTE_ROI) != 0:
                roi_promotion_frames.append(frame_number)
            if epoch_start:
                generation_reset = {
                    "frame": frame_number,
                    "from": prior_current_generation,
                    "to": generation,
                    "reset_flags": reset_flags,
                    "backend_generation": backend_generation,
                }
                generation_resets.append(generation_reset)
                epoch_has_positive_generation = generation > 0
                if generation > 1:
                    generation_unit_violations.append({
                        **generation_reset,
                        "reason": "reset_generation_exceeds_one",
                    })
                if (
                    (reset_flags & RESET_GEOMETRY) != 0 and
                    generation <= 1
                ):
                    geometry_generation_frames.append(frame_number)
            elif (
                prior_current_generation is not None and
                generation != prior_current_generation
            ):
                transition = {
                    "frame": frame_number,
                    "from": prior_current_generation,
                    "to": generation,
                }
                generation_changes.append(frame_number)
                generation_transitions.append(transition)
                if generation != prior_current_generation + 1:
                    generation_unit_violations.append({
                        **transition,
                        "reason": "generation_change_not_unit_increment",
                    })
                if generation > 0 and not epoch_has_positive_generation:
                    if generation != 1:
                        generation_unit_violations.append({
                            **transition,
                            "reason": (
                                "first_positive_generation_in_epoch_not_one"
                            ),
                        })
                    epoch_has_positive_generation = True
            prior_current_generation = generation
            prior_current_update_count = update_count
            prior_current_backend_generation = backend_generation
            prior_current_committed_geometry = committed_geometry
            scroll_confidence = float(state["scroll_confidence"])
            if math.isfinite(scroll_confidence) and scroll_confidence > 0.0:
                predicted_scroll.append(frame_number)
            # A scroll hold deliberately retains the prior event decision. Do
            # not turn that frozen value into repeated detector credit.
            if not hold_state:
                if event == "hard_cut":
                    predicted_hard_cuts.append(frame_number)
                elif event == "flash_or_exposure":
                    predicted_exposure.append(frame_number)
                elif (
                    event == "scroll" and
                    frame_number not in predicted_scroll
                ):
                    predicted_scroll.append(frame_number)
                elif event == "geometry_reset":
                    predicted_geometry.append(frame_number)
        else:
            if event != "same_shot":
                invalid_event_attempts.append({
                    "frame": frame_number,
                    "event": event,
                    "reasons": current_reasons,
                })
            if (
                (
                    prior_current_generation is None and generation != 0
                ) or
                (
                    prior_current_generation is not None and
                    generation != prior_current_generation
                ) or
                (reset_flags & CONTROLLER_RESET_MASK) != 0 or
                (promotion_flags & PROMOTE_ROI) != 0
            ):
                invalid_generation_attempts.append({
                    "frame": frame_number,
                    "generation": generation,
                    "prior_current_generation": prior_current_generation,
                    "promotion_flags": promotion_flags,
                    "reset_flags": reset_flags,
                    "reasons": current_reasons,
                })

        decoded_rows.append({
            "active": active,
            "backend_generation": backend_generation,
            "committed_layout": layout,
            "controller_reset": controller_reset,
            "current_output": current_output,
            "event": event,
            "generation": generation,
            "history_flags": history_flags,
            "layout_decision": layout_decision,
            "promotion_flags": promotion_flags,
            "reset_flags": reset_flags,
            "roi": roi,
            "state": state,
            "state_flags": state_flags,
            "state_kind": state_kind,
        })
        frame_rows.append({
            "active_roi": active,
            "authority_reasons": authority_reasons,
            "content_cut_expected": expected["content_cut"],
            "controller_reset": controller_reset,
            "current_output": current_output,
            "current_output_reasons": current_reasons,
            "event": event,
            "expected_layout": expected["expected_layout"],
            "frame": frame_number,
            "source_frame_id": expected["source_frame_id"],
            "trace_frame_id": trace_frame_ids[index],
            "geometry_generation_expected": expected["geometry_generation"],
            "iou": iou,
            "layout": layout,
            "layout_decision": layout_decision,
            "layout_match": layout_match,
            "backend_generation": backend_generation,
            "reset_flags": reset_flags,
            "roi": roi,
            "roi_generation": generation,
            "state_flags": state_flags,
            "state_kind": state_kind,
            "target_coverage": coverage,
            "update_count": update_count,
            "unsafe_pixels": unsafe_pixels,
        })

    failures: list[dict[str, Any]] = []

    def fail(code: str, detail: Any) -> None:
        failures.append({"code": code, "detail": detail})

    if invalid_output_frames:
        fail("controller_output_not_current", invalid_output_frames)
    if invalid_event_attempts:
        fail("invalid_output_event_attempt", invalid_event_attempts)
    if invalid_generation_attempts:
        fail(
            "invalid_output_generation_attempt",
            invalid_generation_attempts,
        )
    if authority_rejections:
        fail("roi_authority_rejected", authority_rejections)
    if state_flag_inconsistencies:
        fail("state_flag_inconsistent", state_flag_inconsistencies)
    if generation_unit_violations:
        fail("roi_generation_invalid", generation_unit_violations)

    active_set = set(active_indices)
    segment_reports: list[dict[str, Any]] = []
    acquisition_indices: list[int] = []
    inactive_after_acquisition = 0
    for segment in _layout_segments(rendered.timeline):
        start = segment["start_index"]
        end = segment["end_index"]
        layout_name = segment["layout"]
        indices = list(range(start, end))
        if layout_name in ROI_LAYOUTS:
            segment_active = [
                index for index in indices if index in active_set
            ]
            acquisition_index = (
                segment_active[0] if segment_active else None
            )
            acquisition_ms = (
                (acquisition_index - start) * 1000.0 *
                _stream_update_seconds(rendered)
                if acquisition_index is not None else None
            )
            if acquisition_index is not None:
                acquisition_indices.append(acquisition_index)
            segment_rows = [
                frame_rows[index]
                for index in indices
                if index in active_set
            ]
            segment_ious = [row["iou"] for row in segment_rows]
            segment_coverages = [
                row["target_coverage"] for row in segment_rows
            ]
            segment_layout_matches = [
                row["layout_match"] for row in segment_rows
            ]
            inactive = 0
            if acquisition_index is not None:
                inactive = sum(
                    index not in active_set and
                    not near_geometry(index + 1)
                    for index in range(acquisition_index, end)
                )
                inactive_after_acquisition += inactive
            segment_metrics = {
                "acquisition_frame": (
                    acquisition_index + 1
                    if acquisition_index is not None else None
                ),
                "acquisition_ms": acquisition_ms,
                "active_frame_count": len(segment_rows),
                "active_layout_accuracy": (
                    _fraction(
                        sum(segment_layout_matches),
                        len(segment_layout_matches),
                    )
                    if segment_layout_matches else None
                ),
                "inactive_frames_after_acquisition": inactive,
                "roi_iou_median": _quantile(segment_ious, 0.50),
                "roi_iou_p05": _quantile(segment_ious, 0.05),
                "target_coverage_p05": _quantile(
                    segment_coverages, 0.05
                ),
            }
            segment_reports.append({**segment, "metrics": segment_metrics})
            detail = {
                "segment": segment["index"],
                "layout": layout_name,
                "start_frame": segment["start_frame"],
                "end_frame": segment["end_frame"],
            }
            if acquisition_index is None:
                fail("target_not_acquired", detail)
            elif acquisition_ms > policy["max_acquisition_ms"]:
                fail("acquisition_too_slow", {
                    **detail,
                    "actual_ms": acquisition_ms,
                    "maximum_ms": policy["max_acquisition_ms"],
                })
            if acquisition_index is not None:
                for code, name, minimum in (
                    (
                        "roi_iou_median_low",
                        "roi_iou_median",
                        policy["min_roi_iou_median"],
                    ),
                    (
                        "roi_iou_p05_low",
                        "roi_iou_p05",
                        policy["min_roi_iou_p05"],
                    ),
                    (
                        "target_coverage_p05_low",
                        "target_coverage_p05",
                        policy["min_target_coverage_p05"],
                    ),
                    (
                        "active_layout_mismatch",
                        "active_layout_accuracy",
                        policy["min_active_layout_accuracy"],
                    ),
                ):
                    value = segment_metrics[name]
                    if value is None or value < minimum:
                        fail(code, {**detail, "actual": value})
                if inactive > policy["max_inactive_frames_after_acquisition"]:
                    fail("target_released_after_acquisition", {
                        **detail,
                        "inactive_frames": inactive,
                    })
        elif layout_name in FULL_FRAME_LAYOUTS:
            checked = [
                index for index in indices
                if not near_geometry(index + 1)
            ]
            illegal_active = [
                index + 1 for index in checked if index in active_set
            ]
            decision_mismatch = []
            for index in checked:
                row = decoded_rows[index]
                if not row["current_output"]:
                    continue
                if layout_name == "ambiguous":
                    # Ambiguity is an authority contract, not a ban on
                    # short-lived internal proposals. A plausible VIDEO or
                    # CONTENT proposal may enter dwell while the controller
                    # remains canonically FULL_FRAME; illegal_active and
                    # state_mismatch below still fail any proposal that gains
                    # crop authority. Forcing provisional evidence to be
                    # relabelled "ambiguous" would hide useful diagnostics
                    # without improving safety.
                    allowed = FULL_FRAME_LAYOUTS | ROI_LAYOUTS
                else:
                    allowed = frozenset({layout_name})
                if (
                    rendered.timeline[index]["scroll"] or
                    (
                        row["state_flags"] &
                        STATE_SCROLL_HOLD
                    ) != 0
                ):
                    allowed = allowed | {"scroll"}
                if row["layout_decision"] not in allowed:
                    decision_mismatch.append({
                        "frame": index + 1,
                        "actual": row["layout_decision"],
                        "expected": sorted(allowed),
                    })
            state_mismatch = [
                {
                    "frame": index + 1,
                    "actual": decoded_rows[index]["state_kind"],
                    "expected": "full_frame",
                }
                for index in checked
                if (
                    decoded_rows[index]["current_output"] and
                    decoded_rows[index]["state_kind"] != "full_frame"
                )
            ]
            segment_reports.append({
                **segment,
                "metrics": {
                    "active_frame_count": len(illegal_active),
                    "checked_frame_count": len(checked),
                    "layout_decision_mismatch_count": len(
                        decision_mismatch
                    ),
                    "state_mismatch_count": len(state_mismatch),
                },
            })
            detail = {
                "segment": segment["index"],
                "layout": layout_name,
                "start_frame": segment["start_frame"],
                "end_frame": segment["end_frame"],
            }
            if illegal_active:
                fail(
                    (
                        "ambiguous_layout_selected_roi"
                        if layout_name == "ambiguous" else
                        f"{layout_name}_selected_roi"
                    ),
                    {**detail, "frames": illegal_active},
                )
            if decision_mismatch:
                fail("full_frame_layout_decision_mismatch", {
                    **detail,
                    "frames": decision_mismatch,
                })
            if state_mismatch:
                fail("full_frame_state_mismatch", {
                    **detail,
                    "frames": state_mismatch,
                })
        else:
            raise SceneControllerEvalError(
                f"unsupported expected layout {layout_name!r}"
            )

    cut_matches, missed_cuts, false_cuts = _match_events(
        rendered.recipe["events"]["content_cut_frames"],
        predicted_hard_cuts,
        tolerance,
    )
    exposure_matches, missed_exposure, false_exposure = _match_events(
        [
            frame["accepted_depth_update_index"]
            for frame in rendered.timeline
            if frame["exposure_only"]
        ],
        predicted_exposure,
        tolerance,
    )
    geometry_matches, missed_geometry_events, false_geometry_events = _match_events(
        rendered.recipe["events"]["geometry_reset_frames"],
        predicted_geometry,
        tolerance,
    )
    acquisition_frames = [index + 1 for index in acquisition_indices]
    (
        geometry_acquisition_matches,
        missing_acquisition_events,
        false_geometry_events,
    ) = _match_events(
        acquisition_frames,
        false_geometry_events,
        tolerance,
    )
    geometry_acquisition_events = [
        match["predicted"] for match in geometry_acquisition_matches
    ]
    (
        relocation_event_matches,
        missed_relocation_events,
        false_geometry_events,
    ) = _match_event_windows(
        relocation_windows,
        false_geometry_events,
    )
    scroll_interval = rendered.recipe["events"]["scroll_frames"]
    scroll_hold_frames = [
        index + 1
        for index, row in enumerate(decoded_rows)
        if (
            row["current_output"] and
            (row["state_flags"] & STATE_SCROLL_HOLD) != 0
        )
    ]
    scroll_hold_set = set(scroll_hold_frames)
    scroll_history_violations: list[dict[str, Any]] = []
    scroll_roi_violations: list[dict[str, Any]] = []
    for frame_number in scroll_hold_frames:
        index = frame_number - 1
        row = decoded_rows[index]
        state = row["state"]
        if row["promotion_flags"] != SCROLL_PROMOTION_FLAGS:
            scroll_history_violations.append({
                "frame": frame_number,
                "reason": "scroll_promotion_contract_mismatch",
                "promotion_flags": row["promotion_flags"],
                "expected": SCROLL_PROMOTION_FLAGS,
            })
        if row["history_flags"] != SCROLL_HISTORY_FLAGS:
            scroll_history_violations.append({
                "frame": frame_number,
                "reason": "scroll_history_bank_contract_mismatch",
                "history_flags": row["history_flags"],
                "expected": SCROLL_HISTORY_FLAGS,
            })
        uncleared_fields = {
            name: {
                "actual": state[name],
                "expected": expected,
            }
            for name, expected in SCROLL_CLEARED_FIELDS.items()
            if state[name] != expected
        }
        if uncleared_fields:
            scroll_history_violations.append({
                "frame": frame_number,
                "reason": "pending_geometry_not_cleared",
                "fields": uncleared_fields,
            })
        if index == 0 or not decoded_rows[index - 1]["current_output"]:
            scroll_history_violations.append({
                "frame": frame_number,
                "reason": "no_current_predecessor_for_frozen_state",
            })
            continue
        prior = decoded_rows[index - 1]
        prior_state = prior["state"]
        changed_roi_fields = [
            name for name in SCROLL_ROI_RETAINED_FIELDS
            if state[name] != prior_state[name]
        ]
        if (
            row["backend_generation"] != prior["backend_generation"] or
            row["generation"] != prior["generation"] or
            changed_roi_fields
        ):
            scroll_roi_violations.append({
                "frame": frame_number,
                "backend_generation": row["backend_generation"],
                "prior_backend_generation": prior["backend_generation"],
                "generation": row["generation"],
                "prior_generation": prior["generation"],
                "changed_fields": changed_roi_fields,
            })
        changed_frozen_fields = [
            name for name in SCROLL_FROZEN_FIELDS
            if state[name] != prior_state[name]
        ]
        if (
            row["state_flags"] & STATE_CUT_STABLE_MASK
        ) != (
            prior["state_flags"] & STATE_CUT_STABLE_MASK
        ):
            changed_frozen_fields.append("cut_state_flags")
        if changed_frozen_fields:
            scroll_history_violations.append({
                "frame": frame_number,
                "reason": "semantic_decision_state_changed",
                "changed_fields": changed_frozen_fields,
            })

    scroll_detected = False
    scroll_entry_updates: int | None = None
    scroll_entry_nominal: int | None = None
    scroll_entry_earliest: int | None = None
    scroll_entry_latest: int | None = None
    scroll_release_earliest: int | None = None
    scroll_release_deadline: int | None = None
    scroll_entry_frame: int | None = None
    scroll_release_frame: int | None = None
    scroll_hold_required = False
    if scroll_interval:
        first_scroll, last_scroll = scroll_interval
        scroll_entry_updates = max(
            2,
            1 + math.ceil(
                SCROLL_ENTRY_SECONDS /
                _stream_update_seconds(rendered)
            ),
        )
        scroll_entry_nominal = first_scroll + scroll_entry_updates - 1
        scroll_entry_earliest = max(
            1,
            scroll_entry_nominal - tolerance,
        )
        scroll_entry_latest = scroll_entry_nominal + tolerance
        scroll_hold_required = (
            last_scroll - first_scroll + 1 >= scroll_entry_updates
        )
        if scroll_hold_required:
            entry_candidates = [
                frame for frame in scroll_hold_frames
                if scroll_entry_earliest <= frame <= last_scroll
            ]
            if not entry_candidates:
                fail("scroll_hold_not_entered", scroll_interval)
            else:
                scroll_entry_frame = entry_candidates[0]
                if scroll_entry_frame > scroll_entry_latest:
                    fail("scroll_hold_entry_late", {
                        "actual": scroll_entry_frame,
                        "latest": scroll_entry_latest,
                        "nominal": scroll_entry_nominal,
                    })
                entry_evidence = [
                    frame for frame in predicted_scroll
                    if (
                        first_scroll - tolerance <= frame <=
                        scroll_entry_frame
                    )
                ]
                if not entry_evidence:
                    fail("scroll_hold_entry_without_evidence", {
                        "entry_frame": scroll_entry_frame,
                        "evidence_frames": predicted_scroll,
                        "earliest": first_scroll - tolerance,
                    })
                missing_hold = [
                    frame
                    for frame in range(
                        scroll_entry_frame,
                        last_scroll + 1,
                    )
                    if frame not in scroll_hold_set
                ]
                if missing_hold:
                    fail("scroll_hold_coverage_gap", missing_hold)
                elif entry_evidence:
                    scroll_detected = True

            if scroll_entry_frame is not None:
                release_updates = max(
                    1,
                    math.ceil(
                        SCROLL_RELEASE_SECONDS *
                        (1.0 / _stream_update_seconds(rendered))
                    ),
                )
                scroll_release_earliest = (
                    last_scroll + max(1, release_updates - tolerance)
                )
                scroll_release_deadline = (
                    last_scroll + release_updates + tolerance
                )
                if len(frames) < scroll_release_deadline:
                    fail("scroll_hold_release_unobservable", {
                        "frame_count": len(frames),
                        "required_frame": scroll_release_deadline,
                    })
                else:
                    scroll_release_frame = next(
                        (
                            frame
                            for frame in range(
                                last_scroll + 1,
                                scroll_release_deadline + 1,
                            )
                            if (
                                decoded_rows[frame - 1]["current_output"] and
                                frame not in scroll_hold_set
                            )
                        ),
                        None,
                    )
                    if scroll_release_frame is None:
                        fail("scroll_hold_release_late", {
                            "deadline": scroll_release_deadline,
                            "hold_frames": scroll_hold_frames,
                        })
                    else:
                        if scroll_release_frame < scroll_release_earliest:
                            fail("scroll_hold_release_early", {
                                "actual": scroll_release_frame,
                                "earliest": scroll_release_earliest,
                            })
                        release_gaps = [
                            frame
                            for frame in range(
                                last_scroll + 1,
                                scroll_release_frame,
                            )
                            if frame not in scroll_hold_set
                        ]
                        if release_gaps:
                            fail("scroll_hold_release_gap", release_gaps)
                        reentered = [
                            frame for frame in scroll_hold_frames
                            if frame > scroll_release_frame
                        ]
                        if reentered:
                            fail("scroll_hold_reentered", reentered)
            unexpected_hold = [
                frame for frame in scroll_hold_frames
                if (
                    frame < scroll_entry_earliest or
                    (
                        scroll_release_deadline is not None and
                        frame >= scroll_release_deadline
                    )
                )
            ]
            if unexpected_hold:
                fail("unexpected_scroll_hold", unexpected_hold)
        elif scroll_hold_frames:
            fail("unexpected_scroll_hold", scroll_hold_frames)
    elif scroll_hold_frames:
        fail("unexpected_scroll_hold", scroll_hold_frames)

    # Raw directional evidence is intentionally allowed to occur for one
    # update around cuts or moving widgets. It becomes a semantic scroll only
    # after the stream-time dwell activates the hold, so only an unexpected
    # hold is a false positive. Preserve unconfirmed evidence diagnostically.
    unconfirmed_scroll_evidence = [
        frame for frame in predicted_scroll
        if frame not in scroll_hold_set
    ]
    false_scroll = [
        frame for frame in scroll_hold_frames
        if not scroll_interval or not (
            scroll_interval[0] - tolerance <= frame <=
            scroll_interval[1] +
                math.ceil(
                    SCROLL_RELEASE_SECONDS /
                    _stream_update_seconds(rendered)
                ) +
                tolerance
        )
    ]
    scroll_event_detected = bool(
        scroll_interval and any(
            scroll_interval[0] - tolerance <= frame <=
            scroll_interval[1] + tolerance
            for frame in predicted_scroll
        )
    )
    if scroll_interval and not scroll_event_detected:
        fail("scroll_event_missed", scroll_interval)
    if scroll_interval and not scroll_hold_required:
        scroll_detected = scroll_event_detected
    if scroll_history_violations:
        fail("scroll_history_not_frozen", scroll_history_violations)
    if scroll_roi_violations:
        fail("scroll_roi_not_retained", scroll_roi_violations)

    allowed_generation_changes: set[int] = set()
    for acquisition_index in acquisition_indices:
        allowed_generation_changes.update(
            range(
                max(1, acquisition_index + 1 - tolerance),
                min(len(frames), acquisition_index + 1 + tolerance) + 1,
            )
        )
    for reset in rendered.recipe["events"]["geometry_reset_frames"]:
        allowed_generation_changes.update(
            range(max(1, reset - tolerance), min(len(frames), reset + tolerance) + 1)
        )
    for earliest, latest in relocation_windows:
        allowed_generation_changes.update(
            range(max(1, earliest), min(len(frames), latest) + 1)
        )
    for segment in segment_reports[1:]:
        boundary = segment["start_frame"]
        allowed_generation_changes.update(
            range(
                max(1, boundary - tolerance),
                min(len(frames), boundary + tolerance) + 1,
            )
        )
    unexpected_generation_changes = [
        frame for frame in generation_changes
        if frame not in allowed_generation_changes
    ]
    (
        geometry_generation_matches,
        missed_geometry_generations,
        false_geometry_generations,
    ) = _match_events(
        rendered.recipe["events"]["geometry_reset_frames"],
        geometry_generation_frames,
        tolerance,
    )
    (
        _acquisition_generation_matches,
        _missed_acquisition_generations,
        relocation_generation_candidates,
    ) = _match_events(
        acquisition_frames,
        generation_changes,
        tolerance,
    )
    (
        relocation_generation_matches,
        missed_relocation_generations,
        unexpected_relocation_generations,
    ) = _match_event_windows(
        relocation_windows,
        relocation_generation_candidates,
    )
    noninitial_controller_resets = [
        reset["frame"]
        for reset in generation_resets
        if reset["from"] is not None
    ]
    (
        controller_reset_matches,
        _missed_controller_resets,
        unexpected_controller_resets,
    ) = _match_events(
        rendered.recipe["events"]["geometry_reset_frames"],
        noninitial_controller_resets,
        tolerance,
    )
    geometry_authority_frames = sorted(set(
        generation_changes + geometry_generation_frames
    ))
    missing_geometry_events_for_generation = sorted(
        set(geometry_authority_frames) - set(predicted_geometry)
    )
    geometry_events_without_generation = sorted(
        set(predicted_geometry) - set(geometry_authority_frames)
    )
    missing_roi_promotions = sorted(
        set(generation_changes) - set(roi_promotion_frames)
    )
    roi_promotions_without_generation = sorted(
        set(roi_promotion_frames) - set(generation_changes)
    )

    first_acquisition_index = (
        acquisition_indices[0] if acquisition_indices else None
    )
    first_acquisition_segment = next(
        (
            segment for segment in segment_reports
            if (
                segment["layout"] in ROI_LAYOUTS and
                segment["metrics"]["acquisition_frame"] is not None
            )
        ),
        None,
    )
    metrics = {
        "acquisition_frame": (
            first_acquisition_index + 1
            if first_acquisition_index is not None else None
        ),
        "acquisition_ms": (
            first_acquisition_segment["metrics"]["acquisition_ms"]
            if first_acquisition_segment is not None else None
        ),
        "active_frame_count": len(active_indices),
        "active_layout_accuracy": (
            _fraction(sum(layout_matches), len(layout_matches))
            if layout_matches else None
        ),
        "inactive_frames_after_acquisition": inactive_after_acquisition,
        "roi_iou_median": _quantile(ious, 0.50),
        "roi_iou_p05": _quantile(ious, 0.05),
        "roi_jitter_analysis_cells_p95": _quantile(jitter_cells, 0.95),
        "target_coverage_p05": _quantile(coverages, 0.05),
        "unsafe_frame_count": len(unsafe_frames),
        "unsafe_pixel_max": max(
            (item["pixels"] for item in unsafe_frames), default=0
        ),
    }
    if metrics["unsafe_pixel_max"] > policy["max_unsafe_pixels"]:
        fail("unsafe_region_selected", unsafe_frames)
    if (
        metrics["roi_jitter_analysis_cells_p95"] is not None and
        metrics["roi_jitter_analysis_cells_p95"] >
        policy["max_roi_jitter_analysis_cells_p95"]
    ):
        fail(
            "roi_jitter_high",
            metrics["roi_jitter_analysis_cells_p95"],
        )
    if missed_cuts:
        fail("content_cut_missed", missed_cuts)
    if false_cuts:
        fail("false_content_cut", false_cuts)
    if missed_exposure:
        fail("exposure_event_missed", missed_exposure)
    if false_exposure:
        fail("false_exposure_event", false_exposure)
    if false_scroll:
        fail("false_scroll_event", false_scroll)
    if missed_geometry_events:
        fail("geometry_event_missed", missed_geometry_events)
    if missed_relocation_events:
        fail("target_relocation_event_missed", missed_relocation_events)
    if false_geometry_events:
        fail("false_geometry_event", false_geometry_events)
    if (
        missing_geometry_events_for_generation or
        geometry_events_without_generation
    ):
        fail("geometry_generation_event_mismatch", {
            "missing_geometry_events": (
                missing_geometry_events_for_generation
            ),
            "events_without_generation": (
                geometry_events_without_generation
            ),
        })
    if missing_roi_promotions or roi_promotions_without_generation:
        fail("roi_generation_promotion_mismatch", {
            "missing_roi_promotions": missing_roi_promotions,
            "promotions_without_generation": (
                roi_promotions_without_generation
            ),
        })
    if committed_geometry_mismatches:
        fail(
            "committed_geometry_generation_mismatch",
            committed_geometry_mismatches,
        )
    if missing_acquisition_events:
        fail(
            "acquisition_geometry_event_missed",
            missing_acquisition_events,
        )
    if missed_geometry_generations:
        fail("geometry_generation_missed", missed_geometry_generations)
    if false_geometry_generations:
        fail(
            "unexpected_geometry_generation_reset",
            false_geometry_generations,
        )
    if unexpected_controller_resets:
        fail("unexpected_controller_reset", unexpected_controller_resets)
    if unexpected_generation_changes:
        fail("unexpected_roi_generation_change", unexpected_generation_changes)
    if missed_relocation_generations:
        fail(
            "target_relocation_generation_missed",
            missed_relocation_generations,
        )
    if unexpected_relocation_generations:
        fail(
            "unexpected_target_relocation_generation",
            unexpected_relocation_generations,
        )

    recipe_bytes = json.dumps(
        rendered.recipe, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema": REPORT_SCHEMA,
        "evaluator": EVALUATOR,
        "scenario": rendered.recipe["name"],
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "controller_contract": {
            "schema": contract.SCHEMA_VERSION,
            "rule_revision": contract.RULE_REVISION,
            "ordered_abi_hash": contract.ORDERED_ABI_HASH,
            "active_roi_authority": False,
        },
        "thresholds": policy,
        "pass": not failures,
        "failures": failures,
        "metrics": metrics,
        "layout_segments": segment_reports,
        "events": {
            "content_cut": {
                "matches": cut_matches,
                "missed": missed_cuts,
                "false": false_cuts,
            },
            "exposure": {
                "matches": exposure_matches,
                "missed": missed_exposure,
                "false": false_exposure,
            },
            "scroll": {
                "expected_interval": scroll_interval,
                "predicted": predicted_scroll,
                "unconfirmed_evidence": unconfirmed_scroll_evidence,
                "detected": scroll_detected,
                "event_detected": scroll_event_detected,
                "false": false_scroll,
                "hold_frames": scroll_hold_frames,
                "entry_frame": scroll_entry_frame,
                "entry_seconds": SCROLL_ENTRY_SECONDS,
                "entry_updates": scroll_entry_updates,
                "entry_nominal": scroll_entry_nominal,
                "entry_earliest": scroll_entry_earliest,
                "entry_latest": scroll_entry_latest,
                "hold_required": scroll_hold_required,
                "release_earliest": scroll_release_earliest,
                "release_deadline": scroll_release_deadline,
                "release_frame": scroll_release_frame,
                "history_violations": scroll_history_violations,
                "roi_violations": scroll_roi_violations,
            },
            "geometry": {
                "relocation_observation_frames": relocation_frames,
                "relocation_acceptance_windows": relocation_windows,
                "matches": geometry_matches,
                "missed_events": missed_geometry_events,
                "false_events": false_geometry_events,
                "acquisition_events": geometry_acquisition_events,
                "acquisition_event_matches": (
                    geometry_acquisition_matches
                ),
                "missed_acquisition_events": (
                    missing_acquisition_events
                ),
                "relocation_event_matches": (
                    relocation_event_matches
                ),
                "missed_relocation_events": (
                    missed_relocation_events
                ),
                "relocation_generation_matches": (
                    relocation_generation_matches
                ),
                "missed_relocation_generations": (
                    missed_relocation_generations
                ),
                "unexpected_relocation_generations": (
                    unexpected_relocation_generations
                ),
                "generation_changes": generation_changes,
                "generation_transitions": generation_transitions,
                "generation_resets": generation_resets,
                "controller_reset_matches": controller_reset_matches,
                "generation_reset_matches": geometry_generation_matches,
                "generation_event_authority_frames": (
                    geometry_authority_frames
                ),
                "missing_generation_events": (
                    missing_geometry_events_for_generation
                ),
                "events_without_generation": (
                    geometry_events_without_generation
                ),
                "roi_promotion_frames": roi_promotion_frames,
                "missing_roi_promotions": missing_roi_promotions,
                "promotions_without_generation": (
                    roi_promotions_without_generation
                ),
                "committed_geometry_mismatches": (
                    committed_geometry_mismatches
                ),
                "missed_generations": missed_geometry_generations,
                "false_generation_resets": false_geometry_generations,
                "unexpected_controller_resets": (
                    unexpected_controller_resets
                ),
                "unexpected_generation_changes": unexpected_generation_changes,
            },
        },
        "frames": frame_rows,
    }


def _read_exact_json(path: Path, description: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream, object_pairs_hook=compositor._reject_duplicate_keys
            )
    except (OSError, json.JSONDecodeError, compositor.CompositorError) as exc:
        raise SceneControllerEvalError(
            f"cannot read {description} {path}: {exc}"
        ) from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and
        len(value) == 64 and
        all(character in "0123456789abcdef" for character in value)
    )


def _read_authenticated_file(
    root: Path,
    name: str,
    expected_sha256: str,
    description: str,
) -> tuple[Path, bytes]:
    path = root / name
    try:
        value = path.read_bytes()
    except OSError as exc:
        raise SceneControllerEvalError(
            f"cannot read generated {description} {path}: {exc}"
        ) from exc
    actual_sha256 = _sha256(value)
    if actual_sha256 != expected_sha256:
        raise SceneControllerEvalError(
            f"generated {description} hash mismatch: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return path, value


def _expected_artifact_arrays(
    rendered: compositor.RenderedSequence,
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": compositor.array_digest(array),
        }
        for name, array in sorted(rendered.arrays.items())
    }


def _authenticate_label_archive(
    path: Path,
    rendered: compositor.RenderedSequence,
) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = list(archive.files)
            expected_names = set(rendered.arrays)
            if (
                len(names) != len(set(names)) or
                set(names) != expected_names
            ):
                raise SceneControllerEvalError(
                    "generated label archive arrays do not exactly match "
                    f"the recipe: {names}"
                )
            for name, expected in rendered.arrays.items():
                actual = archive[name]
                if (
                    actual.dtype != expected.dtype or
                    actual.shape != expected.shape or
                    not np.array_equal(actual, expected)
                ):
                    raise SceneControllerEvalError(
                        f"generated label archive array {name!r} "
                        "disagrees with the recipe"
                    )
    except SceneControllerEvalError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise SceneControllerEvalError(
            f"cannot decode generated label archive {path}: {exc}"
        ) from exc


def load_generated_clip_contract(
    frames_dir: os.PathLike[str] | str,
    *,
    manifest_path: os.PathLike[str] | str = compositor.DEFAULT_MANIFEST,
) -> compositor.RenderedSequence | None:
    """Authenticate a generated frame directory and return its exact labels.

    A directory without ``scene_controller_contract`` is ordinary user input and returns ``None``.
    Once the contract exists, every mismatch is fatal rather than silently disabling the gate.
    """
    root = Path(frames_dir).resolve()
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        return None
    meta = _read_exact_json(meta_path, "generated clip metadata")
    if not isinstance(meta, dict):
        raise SceneControllerEvalError("generated clip metadata must be an object")
    generated = meta.get("scene_controller_contract")
    if generated is None:
        return None
    if not isinstance(generated, dict) or set(generated) != GENERATED_CONTRACT_KEYS:
        actual = (
            sorted(generated) if isinstance(generated, dict)
            else type(generated).__name__
        )
        raise SceneControllerEvalError(
            "scene_controller_contract must have exact keys "
            f"{sorted(GENERATED_CONTRACT_KEYS)}, got {actual}"
        )
    if (
        not _is_int(generated["schema"]) or
        generated["schema"] != compositor.ARTIFACT_SCHEMA
    ):
        raise SceneControllerEvalError(
            "scene_controller_contract schema mismatch"
        )
    if generated["generator"] != compositor.GENERATOR:
        raise SceneControllerEvalError(
            "scene_controller_contract generator mismatch"
        )
    expected_files = {
        "recipe": "recipe.json",
        "labels": "sequence.npz",
        "timeline": "timeline.jsonl",
        "artifact": "artifact.json",
    }
    for name, expected_name in expected_files.items():
        if generated[name] != expected_name:
            raise SceneControllerEvalError(
                f"scene_controller_contract {name} must be {expected_name}"
            )
    if not isinstance(generated["scenario"], str) or not generated["scenario"]:
        raise SceneControllerEvalError(
            "scene_controller_contract scenario must be non-empty"
        )
    for name in SHA256_FIELDS:
        if not _valid_sha256(generated[name]):
            raise SceneControllerEvalError(
                f"scene_controller_contract {name} must be a lowercase SHA-256"
            )
    if (
        not _is_int(generated["frame_count"]) or
        generated["frame_count"] < 1
    ):
        raise SceneControllerEvalError(
            "scene_controller_contract frame_count must be positive"
        )

    recipe_path, recipe_bytes = _read_authenticated_file(
        root,
        generated["recipe"],
        generated["recipe_file_sha256"],
        "recipe",
    )
    labels_path, _labels_bytes = _read_authenticated_file(
        root,
        generated["labels"],
        generated["labels_sha256"],
        "label archive",
    )
    _timeline_path, timeline_bytes = _read_authenticated_file(
        root,
        generated["timeline"],
        generated["timeline_sha256"],
        "timeline",
    )
    artifact_path, _artifact_bytes = _read_authenticated_file(
        root,
        generated["artifact"],
        generated["artifact_sha256"],
        "artifact descriptor",
    )

    manifest = compositor.load_manifest(manifest_path)
    rendered = compositor.render_scenario(manifest, generated["scenario"])
    canonical_recipe = json.dumps(
        rendered.recipe, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    actual_recipe_hash = hashlib.sha256(canonical_recipe).hexdigest()
    if generated["recipe_sha256"] != actual_recipe_hash:
        raise SceneControllerEvalError(
            "scene_controller_contract recipe hash does not match the locked manifest"
        )
    if generated["frame_count"] != len(rendered.timeline):
        raise SceneControllerEvalError(
            "scene_controller_contract frame count does not match the recipe"
        )
    if (
        generated["rgb_sha256"] !=
        compositor.array_digest(rendered.arrays["rgb"])
    ):
        raise SceneControllerEvalError(
            "scene_controller_contract RGB hash does not match the recipe"
        )

    expected_recipe_file = canonical_recipe + b"\n"
    if recipe_bytes != expected_recipe_file:
        raise SceneControllerEvalError(
            f"generated recipe file disagrees with the locked recipe: {recipe_path}"
        )
    expected_timeline = b"".join(
        compositor._canonical_json(item) + b"\n"
        for item in rendered.timeline
    )
    if timeline_bytes != expected_timeline:
        raise SceneControllerEvalError(
            "generated timeline disagrees with the locked recipe"
        )
    _authenticate_label_archive(labels_path, rendered)

    artifact = _read_exact_json(
        artifact_path, "generated browser artifact descriptor"
    )
    if not isinstance(artifact, dict) or set(artifact) != ARTIFACT_KEYS:
        actual = (
            sorted(artifact) if isinstance(artifact, dict)
            else type(artifact).__name__
        )
        raise SceneControllerEvalError(
            "generated artifact descriptor must have exact keys "
            f"{sorted(ARTIFACT_KEYS)}, got {actual}"
        )
    if (
        not _is_int(artifact["schema"]) or
        artifact["schema"] != compositor.ARTIFACT_SCHEMA
    ):
        raise SceneControllerEvalError(
            "generated artifact descriptor schema mismatch"
        )
    expected_artifact_scalars = {
        "generator": compositor.GENERATOR,
        "scenario": rendered.recipe["name"],
        "recipe_sha256": generated["recipe_sha256"],
        "recipe_file_sha256": generated["recipe_file_sha256"],
        "timeline_sha256": generated["timeline_sha256"],
        "sequence_npz_sha256": generated["labels_sha256"],
        "emitted_frame_files": True,
    }
    for name, expected in expected_artifact_scalars.items():
        actual = artifact[name]
        if type(actual) is not type(expected) or actual != expected:
            raise SceneControllerEvalError(
                f"generated artifact descriptor {name} mismatch"
            )
    if (
        not isinstance(artifact["storage_note"], str) or
        not artifact["storage_note"]
    ):
        raise SceneControllerEvalError(
            "generated artifact descriptor storage_note must be non-empty"
        )
    expected_arrays = _expected_artifact_arrays(rendered)
    if (
        compositor._canonical_json(artifact["arrays"]) !=
        compositor._canonical_json(expected_arrays)
    ):
        raise SceneControllerEvalError(
            "generated artifact descriptor arrays disagree with the recipe"
        )
    if (
        compositor._canonical_json(artifact["semantic_classes"]) !=
        compositor._canonical_json(compositor.SEMANTIC_CLASSES)
    ):
        raise SceneControllerEvalError(
            "generated artifact descriptor semantic classes disagree "
            "with the recipe"
        )

    expected_names = [
        f"frame_{index + 1:05d}.png"
        for index in range(len(rendered.timeline))
    ]
    actual_paths = sorted(root.glob("frame_*.*"))
    if [path.name for path in actual_paths] != expected_names:
        raise SceneControllerEvalError(
            "generated clip frame files do not exactly match the recipe identities"
        )
    try:
        from PIL import Image
    except ImportError as exc:
        raise SceneControllerEvalError(
            "Pillow is required to authenticate generated browser frames"
        ) from exc
    for index, path in enumerate(actual_paths):
        try:
            with Image.open(path) as image:
                actual = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except OSError as exc:
            raise SceneControllerEvalError(
                f"cannot decode generated browser frame {path}: {exc}"
            ) from exc
        if not np.array_equal(actual, rendered.arrays["rgb"][index]):
            raise SceneControllerEvalError(
                f"generated browser frame pixels disagree with the recipe: {path.name}"
            )
    return rendered


def evaluate_generated_clip_trace(
    frames_dir: os.PathLike[str] | str,
    artifacts_dir: os.PathLike[str] | str,
    native_contract: Mapping[str, Any],
    timeline: Mapping[str, Any],
    output: os.PathLike[str] | str,
    *,
    manifest_path: os.PathLike[str] | str = compositor.DEFAULT_MANIFEST,
    thresholds_path: os.PathLike[str] | str = DEFAULT_THRESHOLDS,
) -> dict[str, Any] | None:
    """Run the fail-closed browser qualification gate when generated labels are present."""
    rendered = load_generated_clip_contract(
        frames_dir, manifest_path=manifest_path
    )
    if rendered is None:
        return None
    descriptor = native_contract.get("scene_controller")
    if not isinstance(descriptor, dict):
        raise SceneControllerEvalError(
            "native contract lacks a scene-controller descriptor"
        )
    if (
        descriptor.get("enabled") is not True or
        descriptor.get("backend") != "shadow_rules" or
        descriptor.get("transport") != trace.TRACE_TRANSPORT or
        descriptor.get("file") != trace.TRACE_NAME
    ):
        raise SceneControllerEvalError(
            "browser qualification requires the complete shadow_rules JSONL trace"
        )
    timeline_frames = timeline.get("frames")
    if (
        not isinstance(timeline_frames, list) or
        len(timeline_frames) != len(rendered.timeline) or
        any(not isinstance(row, dict) or
            not isinstance(row.get("frame_id"), str) or
            not row["frame_id"].isdigit()
            for row in timeline_frames)
    ):
        raise SceneControllerEvalError(
            "whole-clip timeline cannot bind browser labels to source frames"
        )
    resolved_runtime = native_contract.get("resolved_runtime")
    follow_mode = (
        isinstance(resolved_runtime, dict) and
        resolved_runtime.get("follow_mode") is True
    )
    trace_ids = (
        [f"{index + 1:010d}" for index in range(len(rendered.timeline))]
        if follow_mode else
        [row["frame_id"] for row in timeline_frames]
    )
    header, frames = trace.load_trace(
        Path(artifacts_dir) / trace.TRACE_NAME,
        expected_frame_ids=trace_ids,
        expected_frame_count=len(trace_ids),
    )
    report = evaluate(
        rendered,
        frames,
        thresholds=load_thresholds(thresholds_path),
        depth_reuse_interval=header["config"]["depth_reuse_interval"],
        trace_frame_ids=trace_ids,
    )
    write_report(report, output)
    return {
        "enabled": True,
        "pass": report["pass"],
        "scenario": report["scenario"],
        "report": Path(output).name,
        "report_sha256": hashlib.sha256(Path(output).read_bytes()).hexdigest(),
        "failure_count": len(report["failures"]),
        "threshold_policy": report["thresholds"]["name"],
        "active_roi_authority": False,
    }


def write_report(report: Mapping[str, Any], output: os.PathLike[str] | str) -> None:
    path = Path(output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=compositor.DEFAULT_MANIFEST)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = compositor.load_manifest(args.manifest)
        rendered = compositor.render_scenario(manifest, args.scenario)
        expected_ids = [
            item["source_frame_id"] for item in rendered.timeline
        ]
        header, frames = trace.load_trace(
            args.trace,
            expected_frame_ids=expected_ids,
            expected_frame_count=len(expected_ids),
        )
        report = evaluate(
            rendered,
            frames,
            thresholds=load_thresholds(args.thresholds),
            depth_reuse_interval=header["config"]["depth_reuse_interval"],
        )
        write_report(report, args.output)
    except (
        compositor.CompositorError,
        trace.SceneControllerTraceError,
        SceneControllerEvalError,
        OSError,
    ) as exc:
        parser.error(str(exc))
    print(
        f"{args.scenario}: {'PASS' if report['pass'] else 'FAIL'} "
        f"({len(report['failures'])} findings)"
    )
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
