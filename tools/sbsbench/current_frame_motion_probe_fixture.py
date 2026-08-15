#!/usr/bin/env python3
"""Deterministic diagnostic evidence for the current-frame motion probe.

This oracle intentionally does not reproduce or authorize a production admission policy. It
compares exact authored current/baseline RGB pixels, reports tile and bottom-OCR-band evidence,
and calls every exact change a veto. Exact equality is only diagnostic quiet evidence; it is never
standalone hold authority. The two cadence phases expose changes that a fixed every-other-frame
policy could miss.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "host_sbs"
    / "current_frame_motion_probe_v1.json"
)

_EXPECTED_FIELD = {"width": 770, "height": 434}
_EXPECTED_TILE = {"width": 16, "height": 16}
_EXPECTED_BOTTOM_BAND = [0, 305, 770, 434]
_EXPECTED_CLASSIFICATION = {
    "changed": "veto_exact_change",
    "unchanged": "quiet_evidence_only",
    "quiet_authorizes_hold": False,
}
_EXPECTED_WIDTHS = [1, 2, 4, 8, 16]
_EXPECTED_EVENTS = ["appear", "move", "disappear"]
_EXPECTED_TIMELINE = ["A", "A", "B", "A", "A"]
_EXPECTED_SCENARIO_KINDS = [
    "object_pulse",
    "equal_sum_swap",
    "subtitle_pulse",
    "subtitle_shift",
    "full_frame_cut",
]
_EXPECTED_COUNTER_KEYS = {
    "report_rows",
    "changed_rows",
    "exact_equal_rows",
    "veto_rows",
    "quiet_evidence_rows",
    "would_hold_slots",
    "would_hold_vetoes",
    "would_hold_quiet_evidence",
    "bottom_band_vetoes",
    "small_object_vetoes",
    "localized_object_pulse_vetoes",
    "equal_sum_rearrangement_vetoes",
    "subtitle_onset_vetoes",
    "subtitle_shift_vetoes",
    "hard_cut_vetoes",
    "active_authorizations",
    "phase_0_rows",
    "phase_0_changed",
    "phase_0_would_hold",
    "phase_1_rows",
    "phase_1_changed",
    "phase_1_would_hold",
}


def _rect_is_valid(rect: Any, width: int, height: int) -> bool:
    return (
        isinstance(rect, list)
        and len(rect) == 4
        and all(isinstance(value, int) and not isinstance(value, bool) for value in rect)
        and 0 <= rect[0] < rect[2] <= width
        and 0 <= rect[1] < rect[3] <= height
    )


def _rects_intersect(left: list[int], right: list[int]) -> bool:
    return (
        left[0] < right[2]
        and left[2] > right[0]
        and left[1] < right[3]
        and left[3] > right[1]
    )


def _rgb_is_valid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(
            isinstance(channel, int)
            and not isinstance(channel, bool)
            and 0 <= channel <= 255
            for channel in value
        )
    )


def validate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    """Validate and return one parsed qualification fixture."""
    if fixture.get("schema") != 1:
        raise ValueError("current-frame motion probe fixture must use schema 1")
    if fixture.get("name") != "host-sbs-current-frame-motion-probe-v1":
        raise ValueError("current-frame motion probe fixture has an unknown identity")
    if fixture.get("purpose") != (
            "deterministic probe qualification evidence; "
            "never standalone active hold authority"):
        raise ValueError("current-frame motion probe fixture has unknown authority semantics")
    if fixture.get("model_field") != _EXPECTED_FIELD:
        raise ValueError("current-frame motion probe fixture must use the 770x434 field")
    if fixture.get("tile") != _EXPECTED_TILE:
        raise ValueError("current-frame motion probe fixture must use 16x16 evidence tiles")
    if fixture.get("bottom_ocr_band") != _EXPECTED_BOTTOM_BAND:
        raise ValueError("current-frame motion probe fixture has the wrong bottom OCR band")
    if fixture.get("classification") != _EXPECTED_CLASSIFICATION:
        raise ValueError("current-frame motion probe fixture may not authorize quiet evidence")

    palette = fixture.get("palette")
    expected_palette = {"background", "object", "subtitle", "swap_a", "swap_b"}
    if not isinstance(palette, dict) or set(palette) != expected_palette:
        raise ValueError("current-frame motion probe fixture has an invalid palette")
    if not all(_rgb_is_valid(value) for value in palette.values()):
        raise ValueError("current-frame motion probe fixture palette must contain uint8 RGB")
    if palette["swap_a"] == palette["swap_b"]:
        raise ValueError("equal-sum rearrangement colors must differ")

    if fixture.get("object_widths") != _EXPECTED_WIDTHS:
        raise ValueError("current-frame motion probe object ladder must be 1/2/4/8/16")
    if fixture.get("small_object_events") != _EXPECTED_EVENTS:
        raise ValueError("current-frame motion probe object events changed")
    objects = fixture.get("small_objects")
    if (not isinstance(objects, list) or
            [item.get("width") for item in objects] != _EXPECTED_WIDTHS):
        raise ValueError("current-frame motion probe object records do not match the ladder")
    field_width = _EXPECTED_FIELD["width"]
    field_height = _EXPECTED_FIELD["height"]
    bottom_top = _EXPECTED_BOTTOM_BAND[1]
    for item in objects:
        size = item["width"]
        start = item.get("start")
        moved = item.get("moved")
        if (not isinstance(start, list) or not isinstance(moved, list) or
                len(start) != 2 or len(moved) != 2 or
                not all(isinstance(value, int) for value in [*start, *moved])):
            raise ValueError(f"current-frame motion probe object {size} has invalid positions")
        start_rect = [start[0], start[1], start[0] + size, start[1] + size]
        moved_rect = [moved[0], moved[1], moved[0] + size, moved[1] + size]
        if (not _rect_is_valid(start_rect, field_width, field_height) or
                not _rect_is_valid(moved_rect, field_width, field_height) or
                _rects_intersect(start_rect, moved_rect) or
                start_rect[3] > bottom_top or moved_rect[3] > bottom_top):
            raise ValueError(
                f"current-frame motion probe object {size} must move disjointly outside OCR")

    cadence = fixture.get("cadence")
    if (not isinstance(cadence, dict) or cadence.get("phase_offsets") != [0, 1] or
            cadence.get("aba_timeline") != _EXPECTED_TIMELINE or
            cadence.get("would_hold_without_probe") !=
            "(transition_index + phase_offset) % 2 == 0"):
        raise ValueError("current-frame motion probe fixture has the wrong cadence contract")

    scenarios = fixture.get("aba_scenarios")
    if (not isinstance(scenarios, list) or
            [scenario.get("kind") for scenario in scenarios] != _EXPECTED_SCENARIO_KINDS):
        raise ValueError("current-frame motion probe A-B-A scenarios changed")
    if len({scenario.get("name") for scenario in scenarios}) != len(scenarios):
        raise ValueError("current-frame motion probe scenario names must be unique")
    bottom_band = fixture["bottom_ocr_band"]
    for scenario in scenarios:
        kind = scenario["kind"]
        rect_keys = {
            "object_pulse": ("rect",),
            "equal_sum_swap": ("first_rect", "second_rect"),
            "subtitle_pulse": ("rect",),
            "subtitle_shift": ("before_rect", "after_rect"),
            "full_frame_cut": (),
        }[kind]
        for key in rect_keys:
            if not _rect_is_valid(scenario.get(key), field_width, field_height):
                raise ValueError(f"current-frame motion probe {scenario['name']} has invalid {key}")
        if kind == "equal_sum_swap":
            first = scenario["first_rect"]
            second = scenario["second_rect"]
            if ((first[2] - first[0], first[3] - first[1]) !=
                    (second[2] - second[0], second[3] - second[1])):
                raise ValueError("equal-sum rectangles must have the same shape")
            if _rects_intersect(first, second):
                raise ValueError("equal-sum rectangles must be disjoint")
            tile_width = fixture["tile"]["width"]
            tile_height = fixture["tile"]["height"]
            first_tile = (first[0] // tile_width, first[1] // tile_height)
            second_tile = (second[0] // tile_width, second[1] // tile_height)
            if first_tile != second_tile:
                raise ValueError("equal-sum rearrangement must stay within one tile")
        if kind in {"subtitle_pulse", "subtitle_shift"}:
            subtitle_rects = [scenario[key] for key in rect_keys]
            if not all(_rects_intersect(rect, bottom_band) for rect in subtitle_rects):
                raise ValueError("subtitle evidence must intersect the exact bottom OCR band")
        if kind == "full_frame_cut":
            for key in ("seed_a", "seed_b"):
                if not isinstance(scenario.get(key), int):
                    raise ValueError("full-frame cut seeds must be integers")
            if scenario["seed_a"] == scenario["seed_b"]:
                raise ValueError("full-frame cut endpoints must differ")

    expected = fixture.get("expected_counters")
    if (not isinstance(expected, dict) or set(expected) != _EXPECTED_COUNTER_KEYS or
            not all(isinstance(value, int) and value >= 0 for value in expected.values())):
        raise ValueError("current-frame motion probe expected counters are malformed")
    if expected["active_authorizations"] != 0:
        raise ValueError("qualification evidence may not expect active authorization")
    return fixture


def load_fixture(path: str | Path = FIXTURE_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError("current-frame motion probe fixture root must be an object")
    return validate_fixture(document)


def _blank_frame(fixture: dict[str, Any]) -> np.ndarray:
    field = fixture["model_field"]
    result = np.empty((field["height"], field["width"], 3), dtype=np.uint8)
    result[...] = np.asarray(fixture["palette"]["background"], dtype=np.uint8)
    return result


def _paint_rect(frame: np.ndarray, rect: list[int], color: list[int]) -> None:
    frame[rect[1]:rect[3], rect[0]:rect[2], :] = np.asarray(color, dtype=np.uint8)


def _object_frame(fixture: dict[str, Any], item: dict[str, Any], position: str) -> np.ndarray:
    result = _blank_frame(fixture)
    x, y = item[position]
    size = item["width"]
    _paint_rect(result, [x, y, x + size, y + size], fixture["palette"]["object"])
    return result


def _cut_frame(fixture: dict[str, Any], seed: int) -> np.ndarray:
    field = fixture["model_field"]
    y, x = np.indices((field["height"], field["width"]), dtype=np.uint32)
    # Each seed changes the red channel by a nonzero constant modulo 256 for the authored pair,
    # while the other channels retain texture. Thus every pixel is an exact endpoint witness.
    return np.stack(
        (
            (3 * x + 5 * y + 17 * seed) & 0xFF,
            (7 * x + 11 * y + 29 * seed) & 0xFF,
            (13 * x + 19 * y + 43 * seed) & 0xFF,
        ),
        axis=2,
    ).astype(np.uint8)


def exact_evidence(
        fixture: dict[str, Any], baseline: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    """Return exact current-vs-baseline pixel, tile, and bottom-band evidence."""
    field = fixture["model_field"]
    expected_shape = (field["height"], field["width"], 3)
    if (baseline.shape != expected_shape or current.shape != expected_shape or
            baseline.dtype != np.uint8 or current.dtype != np.uint8):
        raise ValueError("probe evidence frames must be exact uint8 770x434 RGB")
    delta = current.astype(np.int16) - baseline.astype(np.int16)
    changed = np.any(delta != 0, axis=2)
    changed_y, changed_x = np.nonzero(changed)
    changed_count = int(changed_y.size)
    tile_width = fixture["tile"]["width"]
    tile_height = fixture["tile"]["height"]
    tile_columns = (field["width"] + tile_width - 1) // tile_width
    if changed_count:
        tile_ids = np.unique((changed_y // tile_height) * tile_columns + changed_x // tile_width)
        changed_tiles = [
            [int(tile_id % tile_columns), int(tile_id // tile_columns)]
            for tile_id in tile_ids.tolist()
        ]
        tile_counts = np.bincount(
            (changed_y // tile_height) * tile_columns + changed_x // tile_width)
        maximum_changed_in_tile = int(tile_counts.max())
    else:
        changed_tiles = []
        maximum_changed_in_tile = 0
    bottom = fixture["bottom_ocr_band"]
    bottom_changed = changed[bottom[1]:bottom[3], bottom[0]:bottom[2]]
    bottom_changed_count = int(np.count_nonzero(bottom_changed))
    exact_equal = changed_count == 0
    return {
        "pixel_exact_equal": exact_equal,
        "changed_pixel_count": changed_count,
        "changed_pixel_fraction": changed_count / (field["width"] * field["height"]),
        "changed_tile_count": len(changed_tiles),
        "changed_tiles": changed_tiles,
        "maximum_changed_pixels_in_tile": maximum_changed_in_tile,
        "bottom_band_changed_pixel_count": bottom_changed_count,
        "bottom_band_changed": bottom_changed_count > 0,
        "signed_channel_delta_sum": [
            int(value) for value in delta.astype(np.int64).sum(axis=(0, 1)).tolist()
        ],
        "absolute_channel_delta_sum": [
            int(value) for value in np.abs(delta.astype(np.int64)).sum(axis=(0, 1)).tolist()
        ],
        "maximum_absolute_channel_delta": int(np.abs(delta).max(initial=0)),
        "verdict": (
            fixture["classification"]["unchanged"] if exact_equal
            else fixture["classification"]["changed"]
        ),
        # Even exact equality is only a shadow observation in this fixture.
        "active_authorized": False,
    }


def _small_object_pairs(
        fixture: dict[str, Any], item: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    blank = _blank_frame(fixture)
    start = _object_frame(fixture, item, "start")
    moved = _object_frame(fixture, item, "moved")
    return {
        "appear": (blank, start),
        "move": (start, moved),
        "disappear": (moved, blank),
    }


def _aba_endpoints(
        fixture: dict[str, Any], scenario: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    kind = scenario["kind"]
    if kind == "full_frame_cut":
        return (
            _cut_frame(fixture, scenario["seed_a"]),
            _cut_frame(fixture, scenario["seed_b"]),
        )
    first = _blank_frame(fixture)
    second = _blank_frame(fixture)
    if kind == "object_pulse":
        _paint_rect(second, scenario["rect"], fixture["palette"]["object"])
    elif kind == "equal_sum_swap":
        _paint_rect(first, scenario["first_rect"], fixture["palette"]["swap_a"])
        _paint_rect(first, scenario["second_rect"], fixture["palette"]["swap_b"])
        _paint_rect(second, scenario["first_rect"], fixture["palette"]["swap_b"])
        _paint_rect(second, scenario["second_rect"], fixture["palette"]["swap_a"])
    elif kind == "subtitle_pulse":
        _paint_rect(second, scenario["rect"], fixture["palette"]["subtitle"])
    elif kind == "subtitle_shift":
        _paint_rect(first, scenario["before_rect"], fixture["palette"]["subtitle"])
        _paint_rect(second, scenario["after_rect"], fixture["palette"]["subtitle"])
    else:  # pragma: no cover - validate_fixture rejects unknown kinds.
        raise ValueError(f"unknown current-frame probe scenario {kind!r}")
    return first, second


def qualification_rows(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    """Build exact evidence rows for every event and both hypothetical cadence phases."""
    validate_fixture(fixture)
    phases = fixture["cadence"]["phase_offsets"]
    rows: list[dict[str, Any]] = []
    event_index = 0
    for item in fixture["small_objects"]:
        pairs = _small_object_pairs(fixture, item)
        for event in fixture["small_object_events"]:
            baseline, current = pairs[event]
            for phase in phases:
                rows.append({
                    "family": "small_object",
                    "scenario": f"small-object-{item['width']}",
                    "scenario_kind": "small_object_event",
                    "width": item["width"],
                    "event": event,
                    "phase_offset": phase,
                    "transition_index": event_index,
                    "would_hold_without_probe": (event_index + phase) % 2 == 0,
                    "evidence": exact_evidence(fixture, baseline, current),
                })
            event_index += 1

    timeline = fixture["cadence"]["aba_timeline"]
    for scenario in fixture["aba_scenarios"]:
        endpoint_a, endpoint_b = _aba_endpoints(fixture, scenario)
        frames = {"A": endpoint_a, "B": endpoint_b}
        for phase in phases:
            for transition_index, (before, after) in enumerate(zip(timeline, timeline[1:])):
                rows.append({
                    "family": "aba",
                    "scenario": scenario["name"],
                    "scenario_kind": scenario["kind"],
                    "event": f"{before}-to-{after}",
                    "phase_offset": phase,
                    "transition_index": transition_index,
                    "would_hold_without_probe": (transition_index + phase) % 2 == 0,
                    "evidence": exact_evidence(fixture, frames[before], frames[after]),
                })
    return rows


def _empty_counters() -> dict[str, int]:
    return {key: 0 for key in _EXPECTED_COUNTER_KEYS}


def qualification_report(fixture: dict[str, Any]) -> dict[str, Any]:
    """Return the fixture rows plus conservative expected veto counters."""
    rows = qualification_rows(fixture)
    counters = _empty_counters()
    counters["report_rows"] = len(rows)
    kind_counter = {
        "small_object_event": "small_object_vetoes",
        "object_pulse": "localized_object_pulse_vetoes",
        "equal_sum_swap": "equal_sum_rearrangement_vetoes",
        "subtitle_pulse": "subtitle_onset_vetoes",
        "subtitle_shift": "subtitle_shift_vetoes",
        "full_frame_cut": "hard_cut_vetoes",
    }
    changed_verdict = fixture["classification"]["changed"]
    for row in rows:
        evidence = row["evidence"]
        phase = row["phase_offset"]
        counters[f"phase_{phase}_rows"] += 1
        if evidence["pixel_exact_equal"]:
            counters["exact_equal_rows"] += 1
            counters["quiet_evidence_rows"] += 1
        else:
            counters["changed_rows"] += 1
            counters["veto_rows"] += 1
            counters[f"phase_{phase}_changed"] += 1
            counters[kind_counter[row["scenario_kind"]]] += 1
            if evidence["bottom_band_changed"]:
                counters["bottom_band_vetoes"] += 1
        if row["would_hold_without_probe"]:
            counters["would_hold_slots"] += 1
            counters[f"phase_{phase}_would_hold"] += 1
            if evidence["verdict"] == changed_verdict:
                counters["would_hold_vetoes"] += 1
            else:
                counters["would_hold_quiet_evidence"] += 1
        counters["active_authorizations"] += int(evidence["active_authorized"])
    return {
        "schema": 1,
        "source": fixture["name"],
        "authority": (
            "deterministic probe qualification only; "
            "never standalone active hold authority"
        ),
        "counters": counters,
        "rows": rows,
    }


if __name__ == "__main__":
    print(json.dumps(qualification_report(load_fixture()), indent=2, sort_keys=True))
