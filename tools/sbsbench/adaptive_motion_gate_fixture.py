#!/usr/bin/env python3
"""Deterministic small-object consequence oracle for adaptive Host SBS depth holds.

This helper does not duplicate the production admission policy. The native fixture calls the real
pure Windows helpers using the same JSON input. Here, authored ideal depth fields establish whether
the existing GT-lag metric can see a one-delivery hold at 1/2/4/8/16 DAV2 texels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from . import sbsbench
else:
    import sbsbench


FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "host_sbs"
    / "adaptive_motion_gate_v1.json"
)


def load_fixture(path: str | Path = FIXTURE_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        fixture = json.load(stream)
    if fixture.get("schema") != 1:
        raise ValueError("adaptive-motion fixture must use schema 1")
    field = fixture.get("model_field")
    if field != {"width": 770, "height": 434}:
        raise ValueError("adaptive-motion fixture must use the authenticated 770x434 field")
    if fixture.get("object_widths") != [1, 2, 4, 8, 16]:
        raise ValueError("adaptive-motion fixture object ladder must be 1/2/4/8/16")
    if fixture.get("events") != ["appear", "move", "disappear"]:
        raise ValueError("adaptive-motion fixture event order changed")
    if fixture.get("cadence", {}).get("phase_offsets") != [0, 1]:
        raise ValueError("adaptive-motion fixture requires both cadence phases")
    if fixture["cadence"].get("hard_cut_event_indices") != [0, 1]:
        raise ValueError("adaptive-motion fixture cuts must land on both would-hold phases")
    objects = fixture.get("objects")
    if not isinstance(objects, list) or [item.get("width") for item in objects] != [1, 2, 4, 8, 16]:
        raise ValueError("adaptive-motion object records do not match the width ladder")
    for item in objects:
        width = item["width"]
        start, moved = item.get("start"), item.get("moved")
        if (not isinstance(start, list) or not isinstance(moved, list) or
                len(start) != 2 or len(moved) != 2 or
                moved[0] < start[0] + width + 4 or moved[1] != start[1]):
            raise ValueError(f"adaptive-motion object {width} has an invalid move")
    return fixture


def rects_intersect(left: list[int], right: list[int]) -> bool:
    return (
        len(left) == 4 and len(right) == 4 and
        left[0] < right[2] and left[2] > right[0] and
        left[1] < right[3] and left[3] > right[1]
    )


def changed_texels(width: int, event: str) -> int:
    if event not in {"appear", "move", "disappear"}:
        raise ValueError(f"unknown adaptive-motion event {event!r}")
    return width * width * (2 if event == "move" else 1)


def localized_change_fractions(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    area = fixture["model_field"]["width"] * fixture["model_field"]["height"]
    return [
        {
            "width": item["width"],
            "event": event,
            "changed_texels": changed_texels(item["width"], event),
            "changed_fraction": changed_texels(item["width"], event) / area,
        }
        for item in fixture["objects"]
        for event in fixture["events"]
    ]


def _object_depth(
        fixture: dict[str, Any], item: dict[str, Any], position: str) -> np.ndarray:
    width = fixture["model_field"]["width"]
    height = fixture["model_field"]["height"]
    background = float(fixture["depth"]["background"])
    foreground = float(fixture["depth"]["foreground"])
    result = np.full((height, width), background, dtype=np.float32)
    x, y = item[position]
    size = item["width"]
    result[y:y + size, x:x + size] = foreground
    return result


def depth_lag_report(
        fixture: dict[str, Any], damage_mode: str, phase_offset: int) -> list[dict[str, Any]]:
    """Score an ideal current-depth path or a counterfactual one-hold broad-depth path.

    ``localized`` always infers. ``broad_single_rect`` follows the shadow cadence's alternating
    depth-hold opportunity; its JSON contract marks OCR as separately required on those holds.
    """
    if damage_mode not in fixture["damage_modes"]:
        raise ValueError(f"unknown adaptive-motion damage mode {damage_mode!r}")
    if phase_offset not in fixture["cadence"]["phase_offsets"]:
        raise ValueError(f"unknown adaptive-motion phase {phase_offset}")

    width = fixture["model_field"]["width"]
    height = fixture["model_field"]["height"]
    background = np.full(
        (height, width), float(fixture["depth"]["background"]), dtype=np.float32)
    previous_truth = background
    cached_depth = background
    rows: list[dict[str, Any]] = []
    event_index = 0
    for item in fixture["objects"]:
        targets = {
            "appear": _object_depth(fixture, item, "start"),
            "move": _object_depth(fixture, item, "moved"),
            "disappear": background,
        }
        for event in fixture["events"]:
            current_truth = targets[event]
            held = (
                damage_mode == "broad_single_rect" and
                (event_index + phase_offset) % 2 == 0
            )
            prediction = cached_depth if held else current_truth
            lag = sbsbench.depth_ground_truth_lag(
                prediction, current_truth, previous_truth)
            if lag is None:
                raise RuntimeError(
                    f"depth-lag metric abstained for width={item['width']} event={event}")
            rows.append({
                "index": event_index,
                "width": item["width"],
                "event": event,
                "damage_mode": damage_mode,
                "phase_offset": phase_offset,
                "held": held,
                "ocr_only_needed": bool(
                    held and not fixture["damage_modes"][damage_mode]["ocr_crop_unchanged"]),
                "depth_gt_lag_f1": float(lag),
            })
            if not held:
                cached_depth = current_truth
            previous_truth = current_truth
            event_index += 1
    return rows
