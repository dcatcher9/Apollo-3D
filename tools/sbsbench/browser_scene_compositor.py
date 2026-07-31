#!/usr/bin/env python3
"""Deterministic browser-scene compositor for the Host SBS ROI safety suite.

The committed input is a compact JSON recipe.  Materialized RGB frames and exact dense labels are
evaluation artifacts, not source assets: they are written as one deterministic compressed NPZ so
the repository never needs a directory full of generated PNG frames.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


GENERATOR = "browser-scene-compositor-v1"
MANIFEST_SCHEMA = 1
ARTIFACT_SCHEMA = 1
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "datasets" / "browser_synth_v1.json"

SEMANTIC_CLASSES = {
    "background": 0,
    "primary_video": 1,
    "accepted_content": 2,
    "chrome": 3,
    "ad_unsafe": 4,
}
EXPECTED_LAYOUTS = {
    "no_target",
    "primary_video",
    "content_collage",
    "identity_fullscreen",
    "ambiguous",
}
PATTERNS = {
    "solid",
    "chrome",
    "page",
    "video",
    "cards",
    "ad",
    "noise",
    "grayscale",
}
TAGS = {
    "subject",
    "stable_gutter",
    "ignore",
    "ambiguous_candidate",
    "unrelated",
}
CUT_SCOPES = {"none", "content", "exterior", "both"}
ANIMATIONS = {"static", "frame", "scroll"}
VISIBLE_PHASES = {"always", "before_geometry_reset", "after_geometry_reset"}

MANIFEST_KEYS = {"schema", "generator", "description", "scenarios"}
SCENARIO_KEYS = {
    "name",
    "description",
    "seed",
    "width",
    "height",
    "frames",
    "fps_num",
    "fps_den",
    "expected_layout",
    "selected_instance_id",
    "layers",
    "events",
}
SCENARIO_OPTIONAL_KEYS = {
    "expected_layout_after_geometry_reset",
    "expected_layout_transitions",
}
LAYER_REQUIRED_KEYS = {"name", "role", "instance_id", "rect", "pattern"}
LAYER_OPTIONAL_KEYS = {
    "z",
    "tags",
    "cut_scope",
    "animation",
    "animation_end_frame",
    "animation_pause_frames",
    "rect_after_geometry_reset",
    "rect_after_relocation",
    "visible_frames",
    "visible_phase",
}
EVENT_KEYS = {
    "content_cut_frames",
    "exterior_cut_frames",
    "exposure_gain_percent_by_frame",
    "scroll_frames",
    "scroll_step_px",
    "geometry_reset_frames",
}
EVENT_OPTIONAL_KEYS = {
    "relocation_acceptance_windows",
    "relocation_frames",
}


class CompositorError(ValueError):
    """A recipe cannot be rendered without weakening its label contract."""


@dataclass(frozen=True)
class RenderedSequence:
    recipe: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    timeline: Sequence[Mapping[str, Any]]


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CompositorError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    where: str,
) -> None:
    optional = optional or set()
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise CompositorError(f"{where} is missing keys: {sorted(missing)}")
    if unknown:
        raise CompositorError(f"{where} has unknown keys: {sorted(unknown)}")


def _require_int(
    value: Any,
    *,
    where: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if not _is_int(value) or value < minimum or (
        maximum is not None and value > maximum
    ):
        suffix = f"..{maximum}" if maximum is not None else " or greater"
        raise CompositorError(f"{where} must be an integer in {minimum}{suffix}")
    return value


def _validate_frame_list(value: Any, *, frames: int, where: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise CompositorError(f"{where} must be an array")
    parsed = tuple(
        _require_int(item, where=f"{where}[]", minimum=1, maximum=frames)
        for item in value
    )
    if tuple(sorted(set(parsed))) != parsed:
        raise CompositorError(f"{where} must be unique and strictly increasing")
    return parsed


def _validate_rect(
    value: Any,
    *,
    width: int,
    height: int,
    where: str,
) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4:
        raise CompositorError(f"{where} must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (
        _require_int(item, where=f"{where}[{index}]", minimum=0)
        for index, item in enumerate(value)
    )
    if not (x0 < x1 <= width and y0 < y1 <= height):
        raise CompositorError(
            f"{where} must be a non-empty half-open rectangle inside {width}x{height}"
        )
    return x0, y0, x1, y1


def _validate_events(value: Any, *, frames: int, where: str) -> None:
    if not isinstance(value, dict):
        raise CompositorError(f"{where} must be an object")
    _require_exact_keys(
        value,
        required=EVENT_KEYS,
        optional=EVENT_OPTIONAL_KEYS,
        where=where,
    )
    _validate_frame_list(
        value["content_cut_frames"], frames=frames, where=f"{where}.content_cut_frames"
    )
    _validate_frame_list(
        value["exterior_cut_frames"],
        frames=frames,
        where=f"{where}.exterior_cut_frames",
    )
    _validate_frame_list(
        value["geometry_reset_frames"],
        frames=frames,
        where=f"{where}.geometry_reset_frames",
    )
    relocation_frames = _validate_frame_list(
        value.get("relocation_frames", []),
        frames=frames,
        where=f"{where}.relocation_frames",
    )
    relocation_windows = value.get("relocation_acceptance_windows", [])
    if not isinstance(relocation_windows, list):
        raise CompositorError(
            f"{where}.relocation_acceptance_windows must be an array"
        )
    if len(relocation_windows) != len(relocation_frames):
        raise CompositorError(
            f"{where}.relocation_acceptance_windows must provide one "
            "window per relocation frame"
        )
    prior_window_end = 0
    for index, (relocation_frame, window) in enumerate(
        zip(relocation_frames, relocation_windows)
    ):
        window_where = (
            f"{where}.relocation_acceptance_windows[{index}]"
        )
        if not isinstance(window, list) or len(window) != 2:
            raise CompositorError(
                f"{window_where} must be [earliest, latest]"
            )
        earliest = _require_int(
            window[0],
            where=f"{window_where}[0]",
            minimum=relocation_frame,
            maximum=frames,
        )
        latest = _require_int(
            window[1],
            where=f"{window_where}[1]",
            minimum=earliest,
            maximum=frames,
        )
        if earliest <= prior_window_end:
            raise CompositorError(
                f"{where}.relocation_acceptance_windows must be "
                "strictly ordered and non-overlapping"
            )
        prior_window_end = latest

    gains = value["exposure_gain_percent_by_frame"]
    if not isinstance(gains, dict):
        raise CompositorError(f"{where}.exposure_gain_percent_by_frame must be an object")
    prior = 0
    for key, gain in gains.items():
        if not isinstance(key, str) or not key.isdigit():
            raise CompositorError(f"{where} exposure frame keys must be decimal strings")
        frame = int(key)
        _require_int(frame, where=f"{where} exposure frame", minimum=1, maximum=frames)
        _require_int(gain, where=f"{where} exposure gain", minimum=1, maximum=400)
        if frame <= prior:
            raise CompositorError(f"{where} exposure frames must be strictly increasing")
        prior = frame

    scroll_frames = value["scroll_frames"]
    if (
        not isinstance(scroll_frames, list) or
        len(scroll_frames) not in {0, 2}
    ):
        raise CompositorError(f"{where}.scroll_frames must be [] or [first, last]")
    if scroll_frames:
        first = _require_int(
            scroll_frames[0], where=f"{where}.scroll_frames[0]",
            minimum=1, maximum=frames
        )
        last = _require_int(
            scroll_frames[1], where=f"{where}.scroll_frames[1]",
            minimum=1, maximum=frames
        )
        if first > last:
            raise CompositorError(f"{where}.scroll_frames must be ordered")

    step = value["scroll_step_px"]
    if (
        not isinstance(step, list) or len(step) != 2 or
        any(not _is_int(item) for item in step)
    ):
        raise CompositorError(f"{where}.scroll_step_px must be [integer dx, integer dy]")
    if not scroll_frames and step != [0, 0]:
        raise CompositorError(f"{where} cannot have a scroll step without scroll frames")


def validate_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CompositorError("manifest must be an object")
    _require_exact_keys(value, required=MANIFEST_KEYS, where="manifest")
    if not _is_int(value["schema"]) or value["schema"] != MANIFEST_SCHEMA:
        raise CompositorError(f"manifest.schema must be {MANIFEST_SCHEMA}")
    if value["generator"] != GENERATOR:
        raise CompositorError(f"manifest.generator must be {GENERATOR}")
    if not isinstance(value["description"], str) or not value["description"]:
        raise CompositorError("manifest.description must be a non-empty string")
    scenarios = value["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise CompositorError("manifest.scenarios must be a non-empty array")

    names: set[str] = set()
    for index, scenario in enumerate(scenarios):
        where = f"manifest.scenarios[{index}]"
        if not isinstance(scenario, dict):
            raise CompositorError(f"{where} must be an object")
        _require_exact_keys(
            scenario,
            required=SCENARIO_KEYS,
            optional=SCENARIO_OPTIONAL_KEYS,
            where=where,
        )
        name = scenario["name"]
        if (
            not isinstance(name, str) or not name or
            any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in name)
        ):
            raise CompositorError(f"{where}.name must use lowercase snake_case")
        if name in names:
            raise CompositorError(f"duplicate scenario name: {name}")
        names.add(name)
        if not isinstance(scenario["description"], str) or not scenario["description"]:
            raise CompositorError(f"{where}.description must be non-empty")
        _require_int(scenario["seed"], where=f"{where}.seed", minimum=0, maximum=0xFFFFFFFF)
        width = _require_int(scenario["width"], where=f"{where}.width", minimum=64)
        height = _require_int(scenario["height"], where=f"{where}.height", minimum=64)
        frames = _require_int(scenario["frames"], where=f"{where}.frames", minimum=2)
        _require_int(scenario["fps_num"], where=f"{where}.fps_num", minimum=1)
        _require_int(scenario["fps_den"], where=f"{where}.fps_den", minimum=1)
        _validate_events(scenario["events"], frames=frames, where=f"{where}.events")
        if scenario["expected_layout"] not in EXPECTED_LAYOUTS:
            raise CompositorError(f"{where}.expected_layout is unsupported")
        if (
            "expected_layout_after_geometry_reset" in scenario and
            scenario["expected_layout_after_geometry_reset"] not in EXPECTED_LAYOUTS
        ):
            raise CompositorError(
                f"{where}.expected_layout_after_geometry_reset is unsupported"
            )
        if (
            "expected_layout_after_geometry_reset" in scenario and
            not scenario["events"]["geometry_reset_frames"]
        ):
            raise CompositorError(
                f"{where}.expected_layout_after_geometry_reset requires a geometry reset"
            )
        transitions = scenario.get("expected_layout_transitions", [])
        if not isinstance(transitions, list):
            raise CompositorError(
                f"{where}.expected_layout_transitions must be an array"
            )
        if transitions and "expected_layout_after_geometry_reset" in scenario:
            raise CompositorError(
                f"{where} cannot combine expected_layout_transitions with "
                "expected_layout_after_geometry_reset"
            )
        expected_layouts = [scenario["expected_layout"]]
        prior_transition_frame = 1
        for transition_index, transition in enumerate(transitions):
            transition_where = (
                f"{where}.expected_layout_transitions[{transition_index}]"
            )
            if not isinstance(transition, dict):
                raise CompositorError(
                    f"{transition_where} must be an object"
                )
            _require_exact_keys(
                transition,
                required={"frame", "layout"},
                where=transition_where,
            )
            transition_frame = _require_int(
                transition["frame"],
                where=f"{transition_where}.frame",
                minimum=2,
                maximum=frames,
            )
            if transition_frame <= prior_transition_frame:
                raise CompositorError(
                    f"{where}.expected_layout_transitions must be "
                    "strictly increasing"
                )
            if transition["layout"] not in EXPECTED_LAYOUTS:
                raise CompositorError(
                    f"{transition_where}.layout is unsupported"
                )
            if transition["layout"] == expected_layouts[-1]:
                raise CompositorError(
                    f"{transition_where}.layout must change the contract"
                )
            prior_transition_frame = transition_frame
            expected_layouts.append(transition["layout"])
        if "expected_layout_after_geometry_reset" in scenario:
            expected_layouts.append(
                scenario["expected_layout_after_geometry_reset"]
            )
        selected = _require_int(
            scenario["selected_instance_id"],
            where=f"{where}.selected_instance_id",
            minimum=0,
            maximum=65535,
        )
        layers = scenario["layers"]
        if not isinstance(layers, list) or not layers:
            raise CompositorError(f"{where}.layers must be a non-empty array")
        layer_names: set[str] = set()
        candidate_instances: set[int] = set()
        for layer_index, layer in enumerate(layers):
            layer_where = f"{where}.layers[{layer_index}]"
            if not isinstance(layer, dict):
                raise CompositorError(f"{layer_where} must be an object")
            _require_exact_keys(
                layer,
                required=LAYER_REQUIRED_KEYS,
                optional=LAYER_OPTIONAL_KEYS,
                where=layer_where,
            )
            layer_name = layer["name"]
            if not isinstance(layer_name, str) or not layer_name:
                raise CompositorError(f"{layer_where}.name must be non-empty")
            if layer_name in layer_names:
                raise CompositorError(f"{where} has duplicate layer name {layer_name}")
            layer_names.add(layer_name)
            if layer["role"] not in SEMANTIC_CLASSES:
                raise CompositorError(f"{layer_where}.role is unsupported")
            instance = _require_int(
                layer["instance_id"],
                where=f"{layer_where}.instance_id",
                minimum=0,
                maximum=65535,
            )
            _validate_rect(layer["rect"], width=width, height=height, where=f"{layer_where}.rect")
            if "rect_after_geometry_reset" in layer:
                if not scenario["events"]["geometry_reset_frames"]:
                    raise CompositorError(
                        f"{layer_where}.rect_after_geometry_reset requires a geometry reset"
                    )
                _validate_rect(
                    layer["rect_after_geometry_reset"],
                    width=width,
                    height=height,
                    where=f"{layer_where}.rect_after_geometry_reset",
                )
            if "rect_after_relocation" in layer:
                if len(scenario["events"].get("relocation_frames", [])) != 1:
                    raise CompositorError(
                        f"{layer_where}.rect_after_relocation requires exactly "
                        "one relocation frame"
                    )
                _validate_rect(
                    layer["rect_after_relocation"],
                    width=width,
                    height=height,
                    where=f"{layer_where}.rect_after_relocation",
                )
            if layer["pattern"] not in PATTERNS:
                raise CompositorError(f"{layer_where}.pattern is unsupported")
            _require_int(layer.get("z", layer_index), where=f"{layer_where}.z", minimum=0)
            tags = layer.get("tags", [])
            if (
                not isinstance(tags, list) or
                any(not isinstance(tag, str) or tag not in TAGS for tag in tags) or
                len(tags) != len(set(tags))
            ):
                raise CompositorError(f"{layer_where}.tags are invalid")
            if layer.get("cut_scope", "none") not in CUT_SCOPES:
                raise CompositorError(f"{layer_where}.cut_scope is unsupported")
            if layer.get("animation", "static") not in ANIMATIONS:
                raise CompositorError(f"{layer_where}.animation is unsupported")
            if "animation_end_frame" in layer:
                if layer.get("animation", "static") != "frame":
                    raise CompositorError(
                        f"{layer_where}.animation_end_frame requires frame animation"
                    )
                _require_int(
                    layer["animation_end_frame"],
                    where=f"{layer_where}.animation_end_frame",
                    minimum=1,
                    maximum=frames - 1,
                )
            if "animation_pause_frames" in layer:
                if layer.get("animation", "static") != "frame":
                    raise CompositorError(
                        f"{layer_where}.animation_pause_frames requires frame animation"
                    )
                pause = layer["animation_pause_frames"]
                if not isinstance(pause, list) or len(pause) != 2:
                    raise CompositorError(
                        f"{layer_where}.animation_pause_frames must be [first, last]"
                    )
                if "animation_end_frame" in layer:
                    raise CompositorError(
                        f"{layer_where} cannot combine animation_pause_frames "
                        "with animation_end_frame"
                    )
                first = _require_int(
                    pause[0],
                    where=f"{layer_where}.animation_pause_frames[0]",
                    minimum=2,
                    maximum=frames - 1,
                )
                last = _require_int(
                    pause[1],
                    where=f"{layer_where}.animation_pause_frames[1]",
                    minimum=first,
                    maximum=frames - 1,
                )
            if "visible_frames" in layer:
                visible_frames = layer["visible_frames"]
                if (
                    not isinstance(visible_frames, list) or
                    len(visible_frames) != 2
                ):
                    raise CompositorError(
                        f"{layer_where}.visible_frames must be [first, last]"
                    )
                first = _require_int(
                    visible_frames[0],
                    where=f"{layer_where}.visible_frames[0]",
                    minimum=1,
                    maximum=frames,
                )
                _require_int(
                    visible_frames[1],
                    where=f"{layer_where}.visible_frames[1]",
                    minimum=first,
                    maximum=frames,
                )
            if layer.get("visible_phase", "always") not in VISIBLE_PHASES:
                raise CompositorError(f"{layer_where}.visible_phase is unsupported")
            if instance and layer["role"] in {"primary_video", "accepted_content"}:
                candidate_instances.add(instance)

        relocation_frames = scenario["events"].get(
            "relocation_frames",
            [],
        )
        if relocation_frames:
            relocated_layers = [
                layer for layer in layers
                if "rect_after_relocation" in layer
            ]
            if (
                len(relocation_frames) != 1 or
                len(relocated_layers) != 1 or
                selected == 0 or
                relocated_layers[0]["instance_id"] != selected or
                relocated_layers[0]["rect_after_relocation"] ==
                    relocated_layers[0]["rect"]
            ):
                raise CompositorError(
                    f"{where} relocation must move exactly one selected target "
                    "to one different stable rectangle"
                )

        requires_target = any(
            layout in {"primary_video", "content_collage"}
            for layout in expected_layouts
        )
        if not requires_target and selected != 0:
            raise CompositorError(
                f"{where} full-frame-only layouts must abstain"
            )
        if requires_target and (
            selected == 0 or selected not in candidate_instances
        ):
            raise CompositorError(
                f"{where}.selected_instance_id must identify a content layer"
            )
    return value


def load_manifest(path: os.PathLike[str] | str = DEFAULT_MANIFEST) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError) as exc:
        raise CompositorError(f"cannot read browser compositor manifest: {exc}") from exc
    return validate_manifest(value)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _stable_hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _event_phase(frames: Sequence[int], frame_id: int) -> int:
    return sum(event_frame <= frame_id for event_frame in frames)


def _scroll_offset(events: Mapping[str, Any], frame_id: int) -> tuple[int, int]:
    interval = events["scroll_frames"]
    if not interval or frame_id < interval[0]:
        return 0, 0
    updates = min(frame_id, interval[1]) - interval[0] + 1
    return (
        updates * events["scroll_step_px"][0],
        updates * events["scroll_step_px"][1],
    )


def _base_color(seed: int) -> np.ndarray:
    """Return a dark neutral generated from style identity, never semantic role."""
    token = int(seed) & 0xFFFFFFFF
    return np.array([
        24 + ((token * 29 + 17) % 57),
        24 + ((token * 43 + 31) % 57),
        24 + ((token * 61 + 47) % 57),
    ], dtype=np.uint32)


def _style_transform(image: np.ndarray, seed: int) -> np.ndarray:
    """Deterministically vary palette/channel style without changing geometry."""
    permutations = (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )
    token = int(seed) & 0xFFFFFFFF
    permutation = permutations[(token >> 3) % len(permutations)]
    gains = np.array([
        76 + ((token >> 7) & 31),
        76 + ((token >> 12) & 31),
        76 + ((token >> 17) & 31),
    ], dtype=np.int32)
    offsets = np.array([
        ((token >> 2) & 15) - 7,
        ((token >> 9) & 15) - 7,
        ((token >> 16) & 15) - 7,
    ], dtype=np.int32)
    transformed = image[..., permutation].astype(np.int32)
    transformed = (transformed * gains + 50) // 100 + offsets
    return np.clip(transformed, 0, 255).astype(np.uint8)


def _pattern(
    *,
    width: int,
    height: int,
    role: str,
    pattern: str,
    seed: int,
    phase: int,
    frame_id: int,
    animation: str,
    animation_end_frame: int | None,
    animation_pause_frames: Sequence[int] | None,
    scroll: tuple[int, int],
) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    if animation == "scroll":
        xx = xx + scroll[0]
        yy = yy + scroll[1]
    animation_phase = 0
    if animation == "frame":
        animation_phase = (
            min(frame_id, animation_end_frame)
            if animation_end_frame is not None else
            frame_id
        )
        if animation_pause_frames is not None:
            pause_first, pause_last = animation_pause_frames
            if frame_id >= pause_first:
                paused_updates = min(
                    frame_id,
                    pause_last,
                ) - pause_first + 1
                animation_phase = frame_id - paused_updates
    # A playing video must preserve the current shot's appearance while objects move. Folding the
    # frame number into every video pixel would manufacture a full-frame photometric cut on each
    # update and make the event gate test a synthetic artifact instead of ROI scoping.
    # Frame animation moves a localized subject inside one stable shot. It
    # must not alter the base texture token: doing so makes ordinary playback
    # look like a full-tile cut and can algebraically cancel a real cut when
    # the animation and shot phases advance together.
    pattern_animation_phase = 0 if animation == "frame" else animation_phase
    token = np.uint64(
        seed + phase * 104729 + pattern_animation_phase * 1009
    )
    base = _base_color(seed)
    image = np.empty((height, width, 3), dtype=np.uint32)
    image[...] = base

    if pattern == "solid":
        pass
    elif pattern == "chrome":
        line = ((yy // 7) % 3 == 0) & (xx > width // 8)
        image[line] = np.minimum(base + np.array([38, 38, 38]), 255)
        accent = (xx < max(width // 16, 2)) & (yy < max(height // 2, 2))
        image[accent] = (93, 138, 219)
    elif pattern == "page":
        image[...] = np.minimum(base + np.array([168, 168, 164]), 255)
        text = ((yy % 14) < 3) & ((xx % 97) < 68)
        image[text] = np.maximum(base - np.array([22, 18, 10]), 0)
    elif pattern == "video":
        image[..., 0] = (xx * 3 + yy + int(token & 255)) % 156 + 40
        image[..., 1] = (xx + yy * 2 + int((token >> 8) & 255)) % 146 + 44
        image[..., 2] = (xx * 2 + yy * 3 + int((token >> 16) & 255)) % 136 + 54
        center_x = (animation_phase * 7 + phase * 31 + seed) % max(width, 1)
        center_y = (animation_phase * 3 + phase * 17 + seed // 3) % max(height, 1)
        radius = max(min(width, height) // 6, 2)
        subject = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
        image[subject] = (235, 181, 72)
    elif pattern == "cards":
        gutter = (xx % 52 >= 45) | (yy % 42 >= 34)
        checker = ((xx // 13 + yy // 11 + int(token & 3)) & 1) == 0
        image[~gutter & checker] = (92, 142, 198)
        image[~gutter & ~checker] = (201, 128, 79)
        image[gutter] = (224, 226, 229)
    elif pattern == "ad":
        stripes = ((xx + yy + int(token & 31)) // 9) & 1
        image[stripes == 0] = (238, 68, 91)
        image[stripes == 1] = (251, 193, 54)
    elif pattern == "noise":
        hashed = (
            xx.astype(np.uint64) * np.uint64(0x9E3779B1) ^
            yy.astype(np.uint64) * np.uint64(0x85EBCA77) ^
            token * np.uint64(0xC2B2AE3D)
        )
        image[..., 0] = 32 + (hashed & 191)
        image[..., 1] = 32 + ((hashed >> 8) & 191)
        image[..., 2] = 32 + ((hashed >> 16) & 191)
    elif pattern == "grayscale":
        gray = (
            24 +
            (
                xx * 17 +
                yy * 29 +
                ((xx // 11 + yy // 7) & 1) * 61 +
                int(token & 63)
            ) % 208
        )
        image[..., 0] = gray
        image[..., 1] = gray
        image[..., 2] = gray
    if animation == "frame" and pattern != "video":
        center_x = (animation_phase * 7 + phase * 31 + seed) % max(width, 1)
        center_y = (
            animation_phase * 3 + phase * 17 + seed // 3
        ) % max(height, 1)
        radius = max(min(width, height) // 7, 2)
        subject = (
            (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
        )
        image[subject] = (64, 204, 146)
    clipped = np.clip(image, 0, 255).astype(np.uint8)
    if pattern == "grayscale":
        return clipped
    return _style_transform(clipped, seed ^ 0xA5C31F27)


def _validate_rendered_event_observability(
    recipe: Mapping[str, Any],
    rgb: np.ndarray,
    instance: np.ndarray,
    timeline: Sequence[Mapping[str, Any]],
) -> None:
    events = recipe["events"]

    def changed(frame_id: int) -> np.ndarray:
        if frame_id <= 1:
            raise CompositorError(
                f"{recipe['name']} frame 1 cannot declare a visible transition"
            )
        index = frame_id - 1
        return np.any(rgb[index] != rgb[index - 1], axis=2)

    def scope_mask(frame_id: int, scopes: set[str]) -> np.ndarray:
        ids = {
            layer["instance_id"]
            for layer in recipe["layers"]
            if layer.get("cut_scope", "none") in scopes
        }
        if not ids:
            return np.zeros(instance.shape[1:], dtype=np.bool_)
        index = frame_id - 1
        return (
            np.isin(instance[index], tuple(ids)) |
            np.isin(instance[index - 1], tuple(ids))
        )

    for event_name, scopes in (
        ("content_cut_frames", {"content", "both"}),
        ("exterior_cut_frames", {"exterior", "both"}),
    ):
        for frame_id in events[event_name]:
            if not np.any(changed(frame_id) & scope_mask(frame_id, scopes)):
                raise CompositorError(
                    f"{recipe['name']} {event_name} frame {frame_id} "
                    "does not create a visible scoped pixel change"
                )

    for row in timeline:
        frame_id = row["accepted_depth_update_index"]
        if row["exposure_only"] and not np.any(changed(frame_id)):
            raise CompositorError(
                f"{recipe['name']} exposure transition frame {frame_id} "
                "does not create a visible pixel change"
            )

    scroll_interval = events["scroll_frames"]
    if scroll_interval:
        selected = recipe["selected_instance_id"]
        for frame_id in range(scroll_interval[0], scroll_interval[1] + 1):
            index = frame_id - 1
            if selected:
                mask = (
                    (instance[index] == selected) |
                    (instance[index - 1] == selected)
                )
            else:
                mask = np.ones(instance.shape[1:], dtype=np.bool_)
            if not np.any(changed(frame_id) & mask):
                raise CompositorError(
                    f"{recipe['name']} scroll frame {frame_id} "
                    "does not create a visible target pixel change"
                )

    for frame_id in events["geometry_reset_frames"]:
        if not np.any(changed(frame_id)):
            raise CompositorError(
                f"{recipe['name']} geometry reset frame {frame_id} "
                "does not create a visible pixel change"
            )
    for frame_id in events.get("relocation_frames", []):
        if not np.any(changed(frame_id)):
            raise CompositorError(
                f"{recipe['name']} relocation frame {frame_id} "
                "does not create a visible pixel change"
            )


def _boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, ((1, 1), (1, 1)), mode="constant")
    interior = padded[1:-1, 1:-1]
    eroded = (
        interior &
        padded[:-2, 1:-1] &
        padded[2:, 1:-1] &
        padded[1:-1, :-2] &
        padded[1:-1, 2:]
    )
    return interior & ~eroded


def _mask_bounds(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]


def _scenario_by_name(
    manifest: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    for scenario in manifest["scenarios"]:
        if scenario["name"] == name:
            return scenario
    raise CompositorError(f"unknown browser compositor scenario: {name}")


def render_scenario(
    manifest: Mapping[str, Any],
    name: str,
) -> RenderedSequence:
    validate_manifest(manifest)
    recipe = _scenario_by_name(manifest, name)
    frame_count = recipe["frames"]
    height = recipe["height"]
    width = recipe["width"]
    selected = recipe["selected_instance_id"]
    events = recipe["events"]

    rgb = np.empty((frame_count, height, width, 3), dtype=np.uint8)
    semantic = np.empty((frame_count, height, width), dtype=np.uint8)
    instance = np.empty((frame_count, height, width), dtype=np.uint16)
    tag_arrays = {
        tag: np.zeros((frame_count, height, width), dtype=np.bool_)
        for tag in TAGS
    }
    timeline: list[Mapping[str, Any]] = []

    ordered_layers = sorted(
        enumerate(recipe["layers"]),
        key=lambda item: (item[1].get("z", item[0]), item[0]),
    )
    content_cuts = tuple(events["content_cut_frames"])
    exterior_cuts = tuple(events["exterior_cut_frames"])
    geometry_resets = tuple(events["geometry_reset_frames"])
    relocations = tuple(events.get("relocation_frames", []))
    relocation_acceptance_windows = tuple(
        tuple(window)
        for window in events.get("relocation_acceptance_windows", [])
    )
    relocation_acceptance_frames = tuple(
        (earliest + latest) // 2
        for earliest, latest in relocation_acceptance_windows
    )
    exposure_gains = {
        int(frame): gain
        for frame, gain in events["exposure_gain_percent_by_frame"].items()
    }
    prior_exposure_gain = 100

    for frame_index in range(frame_count):
        frame_id = frame_index + 1
        geometry_phase = _event_phase(geometry_resets, frame_id)
        relocation_phase = _event_phase(relocations, frame_id)
        relocation_acceptance_phase = _event_phase(
            relocation_acceptance_frames,
            frame_id,
        )
        frame_rgb = np.empty((height, width, 3), dtype=np.uint8)
        frame_rgb[...] = _base_color(
            recipe["seed"] ^ 0x51633E2D
        ).astype(np.uint8)
        frame_semantic = np.zeros((height, width), dtype=np.uint8)
        frame_instance = np.zeros((height, width), dtype=np.uint16)
        frame_tags = {
            tag: np.zeros((height, width), dtype=np.bool_)
            for tag in TAGS
        }
        scroll = _scroll_offset(events, frame_id)

        for layer_index, layer in ordered_layers:
            visible_frames = layer.get("visible_frames")
            if visible_frames is not None and not (
                visible_frames[0] <= frame_id <= visible_frames[1]
            ):
                continue
            visible_phase = layer.get("visible_phase", "always")
            if (
                visible_phase == "before_geometry_reset" and geometry_phase > 0
            ) or (
                visible_phase == "after_geometry_reset" and geometry_phase == 0
            ):
                continue
            rect = (
                layer.get("rect_after_geometry_reset", layer["rect"])
                if geometry_phase > 0 else layer["rect"]
            )
            if relocation_phase > 0:
                rect = layer.get("rect_after_relocation", rect)
            x0, y0, x1, y1 = rect
            cut_scope = layer.get("cut_scope", "none")
            phase = 0
            if cut_scope in {"content", "both"}:
                phase += _event_phase(content_cuts, frame_id)
            if cut_scope in {"exterior", "both"}:
                phase += 4096 * _event_phase(exterior_cuts, frame_id)
            tile = _pattern(
                width=x1 - x0,
                height=y1 - y0,
                role=layer["role"],
                pattern=layer["pattern"],
                seed=recipe["seed"] + layer_index * 7919,
                phase=phase,
                frame_id=frame_id,
                animation=layer.get("animation", "static"),
                animation_end_frame=layer.get("animation_end_frame"),
                animation_pause_frames=layer.get(
                    "animation_pause_frames"
                ),
                scroll=scroll,
            )
            frame_rgb[y0:y1, x0:x1] = tile
            frame_semantic[y0:y1, x0:x1] = SEMANTIC_CLASSES[layer["role"]]
            frame_instance[y0:y1, x0:x1] = layer["instance_id"]
            for tag_mask in frame_tags.values():
                tag_mask[y0:y1, x0:x1] = False
            for tag in layer.get("tags", []):
                frame_tags[tag][y0:y1, x0:x1] = True

        exposure_gain = exposure_gains.get(frame_id, 100)
        exposure_changed = exposure_gain != prior_exposure_gain
        if exposure_gain != 100:
            frame_rgb = np.minimum(
                (frame_rgb.astype(np.uint32) * exposure_gain + 50) // 100,
                255,
            ).astype(np.uint8)

        rgb[frame_index] = frame_rgb
        semantic[frame_index] = frame_semantic
        instance[frame_index] = frame_instance
        for tag, tag_mask in frame_tags.items():
            tag_arrays[tag][frame_index] = tag_mask

        target = (
            frame_instance == selected
            if selected else np.zeros((height, width), dtype=np.bool_)
        )
        target &= ~frame_tags["ignore"]
        scroll_interval = events["scroll_frames"]
        scrolling = bool(
            scroll_interval and scroll_interval[0] <= frame_id <= scroll_interval[1]
        )
        expected_layout = recipe["expected_layout"]
        for transition in recipe.get("expected_layout_transitions", []):
            if frame_id >= transition["frame"]:
                expected_layout = transition["layout"]
        if geometry_phase > 0:
            expected_layout = recipe.get(
                "expected_layout_after_geometry_reset",
                expected_layout,
            )
        expected_roi_px = _mask_bounds(target)
        if (
            selected and
            relocation_phase != relocation_acceptance_phase
        ):
            selected_layers = [
                layer for _, layer in ordered_layers
                if layer["instance_id"] == selected
            ]
            if len(selected_layers) != 1:
                raise CompositorError(
                    f"{recipe['name']} relocation contract requires one "
                    "selected target layer"
                )
            selected_layer = selected_layers[0]
            expected_rect = (
                selected_layer.get(
                    "rect_after_geometry_reset",
                    selected_layer["rect"],
                )
                if geometry_phase > 0 else selected_layer["rect"]
            )
            if relocation_acceptance_phase > 0:
                expected_rect = selected_layer.get(
                    "rect_after_relocation",
                    expected_rect,
                )
            expected_roi_px = list(expected_rect)
        timeline.append({
            "accepted_depth_update_index": frame_id,
            "content_cut": frame_id in content_cuts,
            "exposure_gain_percent": exposure_gain,
            "exposure_only": exposure_changed,
            "exterior_cut": frame_id in exterior_cuts,
            "expected_layout": expected_layout,
            "expected_roi_px": expected_roi_px,
            "geometry_generation": geometry_phase,
            "geometry_reset": frame_id in geometry_resets,
            "relocation": frame_id in relocation_acceptance_frames,
            "relocation_generation": relocation_acceptance_phase,
            "relocation_observed": frame_id in relocations,
            "relocation_observation_generation": relocation_phase,
            "selected_instance_id": selected,
            "source_frame_id": f"{frame_id:05d}",
            "source_timestamp_ns": (
                frame_index * 1_000_000_000 * recipe["fps_den"] //
                recipe["fps_num"]
            ),
            "scroll": scrolling,
            "scroll_offset_px": list(scroll),
        })
        prior_exposure_gain = exposure_gain

    _validate_rendered_event_observability(recipe, rgb, instance, timeline)

    target = (
        instance == selected
        if selected else np.zeros_like(instance, dtype=np.bool_)
    )
    target &= ~tag_arrays["ignore"]
    exclusion = (
        (semantic == SEMANTIC_CLASSES["chrome"]) |
        (semantic == SEMANTIC_CLASSES["ad_unsafe"]) |
        tag_arrays["unrelated"]
    )
    focus_boundary = np.stack([_boundary(mask) for mask in target])
    arrays = {
        "rgb": rgb,
        "semantic": semantic,
        "instance": instance,
        "target": target,
        "exclusion": exclusion,
        "focus_boundary": focus_boundary,
        "stable_gutter": tag_arrays["stable_gutter"],
        "subject": tag_arrays["subject"],
        "ambiguity": tag_arrays["ambiguous_candidate"],
        "ignore": tag_arrays["ignore"],
    }
    return RenderedSequence(recipe=recipe, arrays=arrays, timeline=timeline)


def array_digest(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _write_deterministic_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, buffer.getvalue(), compresslevel=9)


def materialize(
    rendered: RenderedSequence,
    output: os.PathLike[str] | str,
    *,
    emit_frames: bool = False,
) -> Path:
    output_path = Path(output).resolve()
    if output_path.exists():
        raise CompositorError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    ))
    try:
        recipe_canonical = _canonical_json(rendered.recipe)
        recipe_bytes = recipe_canonical + b"\n"
        (temporary / "recipe.json").write_bytes(recipe_bytes)
        _write_deterministic_npz(temporary / "sequence.npz", rendered.arrays)
        timeline_bytes = b"".join(
            _canonical_json(item) + b"\n" for item in rendered.timeline
        )
        (temporary / "timeline.jsonl").write_bytes(timeline_bytes)
        if emit_frames:
            from PIL import Image

            for index, frame in enumerate(rendered.arrays["rgb"], start=1):
                Image.fromarray(frame).save(
                    temporary / f"frame_{index:05d}.png"
                )
        sequence_bytes = (temporary / "sequence.npz").read_bytes()
        artifact = {
            "schema": ARTIFACT_SCHEMA,
            "generator": GENERATOR,
            "scenario": rendered.recipe["name"],
            "recipe_sha256": _stable_hash(recipe_canonical),
            "recipe_file_sha256": _stable_hash(recipe_bytes),
            "timeline_sha256": _stable_hash(timeline_bytes),
            "sequence_npz_sha256": _stable_hash(sequence_bytes),
            "arrays": {
                name: {
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "sha256": array_digest(array),
                }
                for name, array in sorted(rendered.arrays.items())
            },
            "semantic_classes": SEMANTIC_CLASSES,
            "storage_note": (
                "Generated evaluation artifact; regenerate from recipe instead of committing."
            ),
            "emitted_frame_files": emit_frames,
        }
        artifact_bytes = (
            json.dumps(artifact, indent=2, sort_keys=True).encode("utf-8") +
            b"\n"
        )
        (temporary / "artifact.json").write_bytes(artifact_bytes)
        if emit_frames:
            meta = {
                "name": rendered.recipe["name"],
                "description": rendered.recipe["description"],
                "content_type": "synthetic",
                "frame_rate": (
                    f"{rendered.recipe['fps_num']}/"
                    f"{rendered.recipe['fps_den']}"
                ),
                "provenance_note": (
                    "Procedural browser safety probe generated by "
                    f"{GENERATOR}; generated frames are disposable."
                ),
                "scene_controller_contract": {
                    "schema": ARTIFACT_SCHEMA,
                    "generator": GENERATOR,
                    "scenario": rendered.recipe["name"],
                    "recipe_sha256": _stable_hash(recipe_canonical),
                    "recipe": "recipe.json",
                    "recipe_file_sha256": _stable_hash(recipe_bytes),
                    "frame_count": len(rendered.timeline),
                    "labels": "sequence.npz",
                    "labels_sha256": _stable_hash(sequence_bytes),
                    "timeline": "timeline.jsonl",
                    "timeline_sha256": _stable_hash(timeline_bytes),
                    "artifact": "artifact.json",
                    "artifact_sha256": _stable_hash(artifact_bytes),
                    "rgb_sha256": array_digest(rendered.arrays["rgb"]),
                },
            }
            (temporary / "meta.json").write_bytes(
                json.dumps(meta, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            )
        temporary.replace(output_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--emit-frames",
        action="store_true",
        help="also emit lossless frame_*.png files and meta.json for the native harness",
    )
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        names = [scenario["name"] for scenario in manifest["scenarios"]]
        if args.list:
            for name in names:
                print(name)
            return 0
        if args.all and args.scenario:
            raise CompositorError("--all and --scenario are mutually exclusive")
        selected = names if args.all else args.scenario
        if not selected:
            raise CompositorError("select --scenario NAME or --all")
        if args.output is None:
            raise CompositorError("--output is required when materializing")
        if len(selected) > 1:
            if args.output.exists():
                raise CompositorError(f"output already exists: {args.output.resolve()}")
            args.output.mkdir(parents=True)
            try:
                for name in selected:
                    materialize(
                        render_scenario(manifest, name),
                        args.output / name,
                        emit_frames=args.emit_frames,
                    )
            except Exception:
                shutil.rmtree(args.output, ignore_errors=True)
                raise
        else:
            materialize(
                render_scenario(manifest, selected[0]),
                args.output,
                emit_frames=args.emit_frames,
            )
    except CompositorError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
