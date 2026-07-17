"""Fail-closed reader for the harness's shipping SubjectState scene evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path


CONTRACT = "apollo-subject-state-runtime-scenes-v1"
TOP_LEVEL_KEYS = {
    "schema",
    "contract",
    "evidence_source",
    "cut_rule",
    "cadence",
    "completion_sequence_contract",
    "depth_reuse_interval",
    "source_frame_ids",
    "completed_source_frame_ids",
    "completed_depth_frame_count",
    "frames",
}
FRAME_KEYS = {
    "source_frame_ordinal",
    "source_frame_id",
    "runtime_scene_id",
    "scene_age",
    "subject_initialized",
    "hard_cut",
    "scene_start",
}


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _ids(value, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    result = [_integer(item, f"{name} item") for item in value]
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return result


def validate(payload):
    """Return canonical evidence or raise when any identity/cadence field is inconsistent."""
    if not isinstance(payload, dict) or set(payload) != TOP_LEVEL_KEYS:
        raise ValueError("runtime scene evidence has noncanonical top-level keys")
    if payload["schema"] != 1 or payload["contract"] != CONTRACT:
        raise ValueError("runtime scene evidence contract differs")
    if payload["evidence_source"] != (
            "SubjectState[0].y after completed depth postprocess"):
        raise ValueError("runtime scene evidence source differs")
    if payload["cut_rule"] != (
            "prior_scene_age_gte_7_and_current_scene_age_eq_0"):
        raise ValueError("runtime scene cut rule differs")
    if payload["cadence"] != "completed-depth-frames-only":
        raise ValueError("runtime scene evidence cadence differs")
    if payload["completion_sequence_contract"] != (
            "exact for this synchronous harness sequence; live busy-drop cadence is not replayed"):
        raise ValueError("runtime scene completion-sequence contract differs")

    depth_reuse = _integer(payload["depth_reuse_interval"],
                           "depth_reuse_interval", minimum=1)
    source_ids = _ids(payload["source_frame_ids"], "source_frame_ids")
    completed_ids = _ids(payload["completed_source_frame_ids"],
                         "completed_source_frame_ids")
    frames = payload["frames"]
    if not source_ids or not isinstance(frames, list) or not frames:
        raise ValueError("runtime scene evidence must contain source and completed frames")
    expected_ordinals = list(range(0, len(source_ids), depth_reuse))
    expected_ids = [source_ids[index] for index in expected_ordinals]
    if completed_ids != expected_ids:
        raise ValueError("completed source IDs differ from the declared depth cadence")
    if (payload["completed_depth_frame_count"] != len(frames) or
            len(frames) != len(expected_ordinals)):
        raise ValueError("completed depth-frame count differs")

    previous = None
    for index, (row, ordinal, source_id) in enumerate(
            zip(frames, expected_ordinals, expected_ids)):
        if not isinstance(row, dict) or set(row) != FRAME_KEYS:
            raise ValueError("runtime scene frame has noncanonical keys")
        if (_integer(row["source_frame_ordinal"],
                     "source_frame_ordinal") != ordinal or
                _integer(row["source_frame_id"], "source_frame_id") != source_id):
            raise ValueError("runtime scene frame identity differs")
        scene_id = _integer(row["runtime_scene_id"], "runtime_scene_id")
        if not isinstance(row["subject_initialized"], bool):
            raise ValueError("subject_initialized must be boolean")
        if not isinstance(row["hard_cut"], bool) or not isinstance(row["scene_start"], bool):
            raise ValueError("runtime scene boundary flags must be boolean")
        age = row["scene_age"]
        if (isinstance(age, bool) or not isinstance(age, (int, float)) or
                not math.isfinite(age) or age < 0.0 or age > 65535.0 or
                abs(age - round(age)) > 1e-5):
            raise ValueError("scene_age must be a finite shader age counter")
        age = float(age)

        if index == 0:
            if (scene_id != 0 or row["hard_cut"] or not row["scene_start"] or
                    age != 0.0):
                raise ValueError("first runtime scene row must start scene zero without a cut")
        elif row["hard_cut"]:
            if (not row["scene_start"] or scene_id != previous["scene_id"] + 1 or
                    not previous["initialized"] or
                    not row["subject_initialized"] or
                    previous["age"] < 7.0 or age != 0.0):
                raise ValueError("hard-cut row does not match the authoritative age reset")
        else:
            if row["scene_start"] or scene_id != previous["scene_id"]:
                raise ValueError("runtime scene changed without a hard cut")
            if previous["initialized"] and not row["subject_initialized"]:
                raise ValueError("runtime subject state deinitialized without a cut")
            if not row["subject_initialized"]:
                if age != 0.0:
                    raise ValueError("uninitialized runtime subject state must have zero age")
            elif not previous["initialized"]:
                if age != 0.0:
                    raise ValueError(
                        "newly initialized runtime subject state must start at zero age"
                    )
            else:
                next_age = min(previous["age"] + 1.0, 65535.0)
                if age != next_age:
                    raise ValueError(
                        "scene age must increment once per completed depth frame"
                    )
        if not row["subject_initialized"] and age != 0.0:
            raise ValueError("uninitialized runtime subject state must have zero age")
        previous = {
            "scene_id": scene_id,
            "age": age,
            "initialized": row["subject_initialized"],
        }
    return payload


def load(path):
    return validate(json.loads(Path(path).read_text(encoding="utf-8")))
