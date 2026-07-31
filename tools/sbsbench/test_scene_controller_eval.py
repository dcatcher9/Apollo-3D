#!/usr/bin/env python3

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import browser_scene_compositor as compositor  # noqa: E402
import scene_controller_contract as contract  # noqa: E402
import scene_controller_eval as evaluator  # noqa: E402
from PIL import Image  # noqa: E402


RULE_INDEX = {
    name: index for index, name in enumerate(contract.RULE_STATE_NAMES)
}


def _flag(group, name):
    return 1 << contract.CONTRACT["flag_bits"][group][name]


STATE_INITIALIZED = _flag("state_flags", "initialized")
STATE_ROI_LOCKED = _flag("state_flags", "roi_locked")
STATE_SCROLL_HOLD = _flag("state_flags", "scroll_hold_active")
STATE_LAYOUT_HISTORY_VALID = _flag("state_flags", "layout_history_valid")
STATE_DEPTH_HISTORY_VALID = _flag("state_flags", "depth_history_valid")
STATE_FALLBACK = _flag("state_flags", "fallback_active")
RESET_LAYOUT = _flag("reset_flags", "layout")
RESET_DEPTH_SHOT = _flag("reset_flags", "depth_shot")
RESET_GEOMETRY = _flag("reset_flags", "geometry")
RESET_BACKEND = _flag("reset_flags", "backend")
RESET_DISPLAY_OR_HDR = _flag("reset_flags", "display_or_hdr")
PROMOTE_LAYOUT = _flag("promotion_flags", "layout_history")
PROMOTE_DEPTH = _flag("promotion_flags", "depth_history")
PROMOTE_ROI = _flag("promotion_flags", "roi")
HISTORY_LAYOUT_READ = _flag("history_flags", "layout_read_bank")
HISTORY_LAYOUT_WRITE = _flag("history_flags", "layout_write_bank")
HISTORY_DEPTH_READ = _flag("history_flags", "depth_read_bank")
HISTORY_DEPTH_WRITE = _flag("history_flags", "depth_write_bank")


def _write_json(path, value):
    path.write_bytes(
        json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rule_state():
    return [
        int(field["initial"]) if field["type"] == "uint32"
        else float(field["initial"])
        for field in contract.CONTRACT["rule_state"]
    ]


def _ideal_frames(rendered):
    frames = []
    layout_enum = contract.CONTRACT["enums"]["layout_decision"]
    event_enum = contract.CONTRACT["enums"]["event_decision"]
    state_enum = contract.CONTRACT["enums"]["state_kind"]
    generation = 0
    epoch_update_count = 0
    has_target = False
    scroll_interval = rendered.recipe["events"]["scroll_frames"]
    scroll_enter_updates = 1 + math.ceil(
        evaluator.SCROLL_ENTRY_SECONDS *
        rendered.recipe["fps_num"] /
        rendered.recipe["fps_den"]
    )
    scroll_entry_frame = (
        scroll_interval[0] + scroll_enter_updates - 1
        if scroll_interval else None
    )
    scroll_hold_required = bool(
        scroll_interval and
        scroll_interval[1] - scroll_interval[0] + 1 >=
        scroll_enter_updates
    )
    scroll_release_updates = math.ceil(
        evaluator.SCROLL_RELEASE_SECONDS *
        rendered.recipe["fps_num"] /
        rendered.recipe["fps_den"]
    )
    for index, expected in enumerate(rendered.timeline):
        frame_number = index + 1
        state = _rule_state()
        expected_layout = expected["expected_layout"]
        target_layout = expected_layout in evaluator.ROI_LAYOUTS
        reset_flags = 0
        if index == 0:
            generation = 0
            epoch_update_count = 0
            has_target = False
            reset_flags = RESET_LAYOUT | RESET_DEPTH_SHOT | RESET_BACKEND
        elif expected["geometry_reset"]:
            generation = 0
            epoch_update_count = 0
            has_target = False
            reset_flags = (
                RESET_LAYOUT |
                RESET_DEPTH_SHOT |
                RESET_GEOMETRY |
                RESET_DISPLAY_OR_HDR
            )
        acquires_target = (
            target_layout and
            not has_target and
            reset_flags == 0
        )
        relocates_target = (
            target_layout and
            has_target and
            reset_flags == 0 and
            bool(expected.get("relocation", False))
        )
        if acquires_target:
            generation += 1
            has_target = True
        elif relocates_target:
            generation += 1
        authoritative_target = target_layout and has_target
        epoch_update_count += 1

        hold = bool(
            scroll_hold_required and
            scroll_entry_frame <= frame_number <= (
                scroll_interval[1] + scroll_release_updates - 1
            )
        )
        if authoritative_target and expected_layout == "primary_video":
            state_kind = "video"
        elif authoritative_target and expected_layout == "content_collage":
            state_kind = "content"
        else:
            state_kind = "full_frame"

        state[RULE_INDEX["output_valid"]] = 1.0
        state[RULE_INDEX["backend_generation"]] = 1
        state[RULE_INDEX["update_count"]] = epoch_update_count
        state[RULE_INDEX["reset_flags"]] = reset_flags
        state[RULE_INDEX["state_kind"]] = float(state_enum[state_kind])
        state[RULE_INDEX["committed_layout"]] = float(
            layout_enum[
                expected_layout if authoritative_target else "no_target"
            ]
        )
        state[RULE_INDEX["layout_decision"]] = float(
            layout_enum[
                "scroll"
                if (
                    scroll_interval and
                    frame_number == scroll_interval[0]
                ) else
                expected_layout
            ]
        )
        roi = (
            expected["expected_roi_px"]
            if authoritative_target else None
        ) or [
            0, 0, rendered.recipe["width"], rendered.recipe["height"]
        ]
        normalized = [
            roi[0] / rendered.recipe["width"],
            roi[1] / rendered.recipe["height"],
            roi[2] / rendered.recipe["width"],
            roi[3] / rendered.recipe["height"],
        ]
        for name, value in zip(
            (
                "committed_roi_x0",
                "committed_roi_y0",
                "committed_roi_x1",
                "committed_roi_y1",
            ),
            normalized,
        ):
            state[RULE_INDEX[name]] = value
        state[RULE_INDEX["committed_roi_confidence"]] = (
            1.0 if authoritative_target else 0.0
        )
        state[RULE_INDEX["committed_mask_confidence"]] = (
            1.0 if authoritative_target else 0.0
        )
        state[RULE_INDEX["layout_confidence"]] = 1.0
        state[RULE_INDEX["roi_generation"]] = generation

        state_flags = (
            STATE_INITIALIZED |
            STATE_LAYOUT_HISTORY_VALID |
            STATE_DEPTH_HISTORY_VALID
        )
        if authoritative_target:
            state_flags |= STATE_ROI_LOCKED
        if hold:
            state_flags |= STATE_SCROLL_HOLD
        state[RULE_INDEX["state_flags"]] = state_flags

        promotion_flags = PROMOTE_LAYOUT
        if not hold:
            promotion_flags |= PROMOTE_DEPTH
        if acquires_target or relocates_target:
            promotion_flags |= PROMOTE_ROI
        state[RULE_INDEX["promotion_flags"]] = promotion_flags
        history_flags = HISTORY_LAYOUT_READ | HISTORY_DEPTH_READ
        if promotion_flags & PROMOTE_LAYOUT:
            history_flags |= HISTORY_LAYOUT_WRITE
        if promotion_flags & PROMOTE_DEPTH:
            history_flags |= HISTORY_DEPTH_WRITE
        state[RULE_INDEX["history_flags"]] = history_flags

        if (
            expected["geometry_reset"] or
            acquires_target or
            relocates_target
        ):
            event = "geometry_reset"
        elif expected["content_cut"]:
            event = "hard_cut"
        elif expected["exposure_only"]:
            event = "flash_or_exposure"
        elif expected["scroll"]:
            event = "scroll"
        else:
            event = "same_shot"
        state[RULE_INDEX["event_decision"]] = float(event_enum[event])
        state[RULE_INDEX["event_confidence"]] = 1.0
        state[RULE_INDEX["scroll_confidence"]] = (
            1.0
            if expected["scroll"]
            else 0.0
        )
        if hold:
            if frames:
                # The resolver retains committed geometry and freezes adaptive
                # decisions from the update immediately preceding scroll hold.
                prior_state = frames[-1]["rule_state"]
                for name in evaluator.SCROLL_FROZEN_FIELDS:
                    state[RULE_INDEX[name]] = prior_state[RULE_INDEX[name]]
            # Pending acquisition/challenger geometry is invalidated, not
            # frozen, on the first held update and remains canonical.
            for name, value in evaluator.SCROLL_CLEARED_FIELDS.items():
                state[RULE_INDEX[name]] = value
        frames.append({
            "frame_id": expected["source_frame_id"],
            "source_index": index,
            "rule_state": state,
        })
    return frames


def _failure_codes(report):
    return {finding["code"] for finding in report["failures"]}


def _with_fps(rendered, fps, *, minimum_frames=None):
    recipe = copy.deepcopy(rendered.recipe)
    recipe["fps_num"] = fps
    recipe["fps_den"] = 1
    arrays = rendered.arrays
    timeline = rendered.timeline
    if minimum_frames is not None and minimum_frames > len(timeline):
        original_frames = len(timeline)
        added_frames = minimum_frames - original_frames
        recipe["frames"] = minimum_frames
        timeline = copy.deepcopy(list(timeline))
        for frame_index in range(original_frames, minimum_frames):
            row = copy.deepcopy(timeline[-1])
            row["source_frame_id"] = f"{frame_index + 1:05d}"
            row["source_timestamp_ns"] = (
                frame_index * 1_000_000_000 // fps
            )
            timeline.append(row)
        arrays = {
            name: (
                np.concatenate(
                    (
                        value,
                        np.repeat(value[-1:], added_frames, axis=0),
                    ),
                    axis=0,
                )
                if value.ndim > 0 and value.shape[0] == original_frames
                else value
            )
            for name, value in rendered.arrays.items()
        }
    return compositor.RenderedSequence(
        recipe=recipe,
        arrays=arrays,
        timeline=timeline,
    )


class SceneControllerEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = compositor.load_manifest()

    def test_ideal_clear_target_passes_all_hard_gates(self):
        rendered = compositor.render_scenario(
            self.manifest, "video_sidebar_ad"
        )
        report = evaluator.evaluate(rendered, _ideal_frames(rendered))
        self.assertTrue(report["pass"], report["failures"])
        self.assertEqual(report["metrics"]["unsafe_pixel_max"], 0)
        self.assertFalse(report["frames"][0]["active_roi"])
        self.assertEqual(report["frames"][0]["roi_generation"], 0)
        self.assertEqual(report["metrics"]["acquisition_frame"], 2)
        self.assertEqual(
            report["events"]["content_cut"]["matches"],
            [{"expected": 19, "predicted": 19}],
        )

    def test_every_ideal_scenario_passes_its_segment_contract(self):
        for scenario in self.manifest["scenarios"]:
            with self.subTest(scenario=scenario["name"]):
                rendered = compositor.render_scenario(
                    self.manifest, scenario["name"]
                )
                report = evaluator.evaluate(
                    rendered, _ideal_frames(rendered)
                )
                self.assertTrue(report["pass"], report["failures"])

    def test_ideal_ambiguity_abstains(self):
        rendered = compositor.render_scenario(
            self.manifest, "ambiguous_dual_video"
        )
        report = evaluator.evaluate(rendered, _ideal_frames(rendered))
        self.assertTrue(report["pass"], report["failures"])
        self.assertEqual(report["metrics"]["active_frame_count"], 0)

    def test_simplified_unsupported_policy_always_abstains(self):
        unsupported = {
            "cold_start_static_media",
            "ambiguous_dual_video",
            "partial_player",
            "cross_axis_disjoint_motion",
            "sparse_three_side_candidate",
            "animated_sidebar_no_dominant_player",
            "grayscale_static_media",
        }
        for name in unsupported:
            with self.subTest(scenario=name):
                rendered = compositor.render_scenario(
                    self.manifest,
                    name,
                )
                report = evaluator.evaluate(
                    rendered,
                    _ideal_frames(rendered),
                )
                self.assertTrue(report["pass"], report["failures"])
                self.assertEqual(
                    report["metrics"]["active_frame_count"],
                    0,
                )

    def test_stable_relocation_updates_roi_once_without_controller_reset(self):
        rendered = compositor.render_scenario(
            self.manifest,
            "stable_relocated_player",
        )
        report = evaluator.evaluate(rendered, _ideal_frames(rendered))
        self.assertTrue(report["pass"], report["failures"])
        geometry = report["events"]["geometry"]
        self.assertEqual(
            geometry["relocation_event_matches"],
            [{"expected": [59, 61], "predicted": 60}],
        )
        self.assertEqual(
            geometry["relocation_generation_matches"],
            [{"expected": [59, 61], "predicted": 60}],
        )
        self.assertEqual(
            report["frames"][58]["roi_generation"],
            1,
        )
        self.assertEqual(
            report["frames"][59]["roi_generation"],
            2,
        )
        self.assertFalse(report["frames"][59]["controller_reset"])

    def test_ambiguous_candidate_scroll_holds_releases_and_abstains(self):
        rendered = compositor.render_scenario(
            self.manifest,
            "ambiguous_candidate_during_scroll",
        )
        report = evaluator.evaluate(rendered, _ideal_frames(rendered))
        self.assertTrue(report["pass"], report["failures"])
        self.assertTrue(report["events"]["scroll"]["event_detected"])
        self.assertTrue(report["events"]["scroll"]["hold_required"])
        self.assertEqual(
            report["events"]["scroll"]["hold_frames"],
            list(range(4, 28)),
        )
        self.assertTrue(all(
            report["frames"][frame - 1]["state_kind"] == "full_frame"
            for frame in report["events"]["scroll"]["hold_frames"]
        ))
        self.assertEqual(report["events"]["scroll"]["release_frame"], 28)
        self.assertIsNone(report["metrics"]["acquisition_frame"])
        self.assertEqual(report["metrics"]["active_frame_count"], 0)
        self.assertEqual(
            [segment["layout"] for segment in report["layout_segments"]],
            ["ambiguous"],
        )
        self.assertTrue(all(
            frame["state_kind"] == "full_frame" and
            not frame["active_roi"]
            for frame in report["frames"]
        ))

        missing_hold = _ideal_frames(rendered)
        for frame in missing_hold:
            state = frame["rule_state"]
            state[RULE_INDEX["state_flags"]] &= ~STATE_SCROLL_HOLD
        missing_report = evaluator.evaluate(rendered, missing_hold)
        self.assertIn(
            "scroll_hold_not_entered",
            _failure_codes(missing_report),
        )

    def test_cold_static_envelope_commits_content_never_video(self):
        rendered = compositor.render_scenario(
            self.manifest,
            "cold_paused_video",
        )
        report = evaluator.evaluate(rendered, _ideal_frames(rendered))
        self.assertTrue(report["pass"], report["failures"])
        self.assertEqual(report["metrics"]["acquisition_frame"], 2)
        self.assertTrue(all(
            not row["active_roi"] or (
                row["state_kind"] == "content" and
                row["layout"] == "content_collage"
            )
            for row in report["frames"]
        ))

    def test_interrupted_challenger_retains_incumbent_generation(self):
        rendered = compositor.render_scenario(
            self.manifest,
            "interrupted_relocation",
        )
        report = evaluator.evaluate(rendered, _ideal_frames(rendered))
        self.assertTrue(report["pass"], report["failures"])
        self.assertEqual(report["metrics"]["acquisition_frame"], 2)
        self.assertEqual(
            report["events"]["geometry"]["generation_changes"],
            [2],
        )
        self.assertEqual(
            report["events"]["geometry"]["relocation_event_matches"],
            [],
        )
        self.assertTrue(all(
            row["roi_generation"] == 1
            for row in report["frames"][1:]
        ))

    def test_ambiguity_allows_proposals_but_keeps_full_frame_authority(self):
        rendered = compositor.render_scenario(
            self.manifest, "ambiguous_dual_video"
        )
        frames = _ideal_frames(rendered)
        layout_enum = contract.CONTRACT["enums"]["layout_decision"]
        for frame in frames:
            frame["rule_state"][RULE_INDEX["layout_decision"]] = float(
                layout_enum["no_target"]
            )
        for index in (0, 5, 11):
            frames[index]["rule_state"][
                RULE_INDEX["layout_decision"]
            ] = float(layout_enum["identity_fullscreen"])
        report = evaluator.evaluate(rendered, frames)
        self.assertTrue(report["pass"], report["failures"])

        frames[5]["rule_state"][RULE_INDEX["layout_decision"]] = float(
            layout_enum["primary_video"]
        )
        report = evaluator.evaluate(rendered, frames)
        self.assertNotIn(
            "full_frame_layout_decision_mismatch",
            _failure_codes(report),
        )
        self.assertTrue(report["pass"], report["failures"])

    def test_invalid_output_does_not_restrict_later_full_frame_attribution(self):
        rendered = compositor.render_scenario(
            self.manifest, "ambiguous_dual_video"
        )
        frames = _ideal_frames(rendered)
        layout_enum = contract.CONTRACT["enums"]["layout_decision"]
        event_enum = contract.CONTRACT["enums"]["event_decision"]
        first = frames[0]["rule_state"]
        first[RULE_INDEX["output_valid"]] = 0.0
        first[RULE_INDEX["state_flags"]] |= STATE_FALLBACK
        first[RULE_INDEX["event_decision"]] = float(event_enum["same_shot"])
        frames[1]["rule_state"][RULE_INDEX["layout_decision"]] = float(
            layout_enum["identity_fullscreen"]
        )
        for frame in frames[2:]:
            frame["rule_state"][RULE_INDEX["layout_decision"]] = float(
                layout_enum["no_target"]
            )
        report = evaluator.evaluate(rendered, frames)
        self.assertNotIn(
            "full_frame_layout_decision_mismatch",
            _failure_codes(report),
        )

    def test_sidebar_overlap_is_a_hard_failure(self):
        rendered = compositor.render_scenario(
            self.manifest, "video_sidebar_ad"
        )
        frames = _ideal_frames(rendered)
        state = frames[10]["rule_state"]
        state[RULE_INDEX["committed_roi_x1"]] = 1.0
        report = evaluator.evaluate(rendered, frames)
        codes = {finding["code"] for finding in report["failures"]}
        self.assertIn("unsafe_region_selected", codes)
        self.assertFalse(report["pass"])

    def test_false_cut_from_sidebar_is_a_hard_failure(self):
        rendered = compositor.render_scenario(
            self.manifest, "paused_video_animated_sidebar"
        )
        frames = _ideal_frames(rendered)
        frames[7]["rule_state"][RULE_INDEX["event_decision"]] = float(
            contract.CONTRACT["enums"]["event_decision"]["hard_cut"]
        )
        report = evaluator.evaluate(rendered, frames)
        codes = {finding["code"] for finding in report["failures"]}
        self.assertIn("false_content_cut", codes)

    def test_active_selection_in_ambiguous_layout_fails(self):
        rendered = compositor.render_scenario(
            self.manifest, "ambiguous_dual_video"
        )
        frames = _ideal_frames(rendered)
        state = frames[5]["rule_state"]
        state[RULE_INDEX["state_kind"]] = float(
            contract.CONTRACT["enums"]["state_kind"]["video"]
        )
        state[RULE_INDEX["committed_layout"]] = float(
            contract.CONTRACT["enums"]["layout_decision"]["primary_video"]
        )
        state[RULE_INDEX["roi_generation"]] = 1
        state[RULE_INDEX["state_flags"]] |= STATE_ROI_LOCKED
        for name, value in zip(
            (
                "committed_roi_x0",
                "committed_roi_y0",
                "committed_roi_x1",
                "committed_roi_y1",
            ),
            (0.1, 0.1, 0.5, 0.5),
        ):
            state[RULE_INDEX[name]] = value
        report = evaluator.evaluate(rendered, frames)
        self.assertIn(
            "ambiguous_layout_selected_roi",
            _failure_codes(report),
        )

    def test_active_roi_requires_the_complete_authority_predicate(self):
        rendered = compositor.render_scenario(
            self.manifest, "video_sidebar_ad"
        )
        mutations = {
            "output_invalid": lambda state: state.__setitem__(
                RULE_INDEX["output_valid"], 0.0
            ),
            "uninitialized": lambda state: state.__setitem__(
                RULE_INDEX["state_flags"],
                state[RULE_INDEX["state_flags"]] & ~STATE_INITIALIZED,
            ),
            "missing_lock": lambda state: state.__setitem__(
                RULE_INDEX["state_flags"],
                state[RULE_INDEX["state_flags"]] & ~STATE_ROI_LOCKED,
            ),
            "fallback": lambda state: state.__setitem__(
                RULE_INDEX["state_flags"],
                state[RULE_INDEX["state_flags"]] | STATE_FALLBACK,
            ),
            "wrong_state": lambda state: state.__setitem__(
                RULE_INDEX["state_kind"],
                float(contract.CONTRACT["enums"]["state_kind"]["full_frame"]),
            ),
            "wrong_layout": lambda state: state.__setitem__(
                RULE_INDEX["committed_layout"],
                float(
                    contract.CONTRACT["enums"]["layout_decision"][
                        "content_collage"
                    ]
                ),
            ),
            "malformed_roi": lambda state: state.__setitem__(
                RULE_INDEX["committed_roi_x1"],
                state[RULE_INDEX["committed_roi_x0"]],
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                frames = _ideal_frames(rendered)
                mutate(frames[10]["rule_state"])
                report = evaluator.evaluate(rendered, frames)
                self.assertFalse(report["frames"][10]["active_roi"])
                self.assertFalse(report["pass"])
                if name in {"output_invalid", "uninitialized", "fallback"}:
                    self.assertIn(
                        "controller_output_not_current",
                        _failure_codes(report),
                    )
                else:
                    self.assertIn(
                        "roi_authority_rejected",
                        _failure_codes(report),
                    )

    def test_missing_target_does_not_cascade_empty_quality_metrics(self):
        rendered = compositor.render_scenario(
            self.manifest, "video_sidebar_ad"
        )
        frames = _ideal_frames(rendered)
        state_enum = contract.CONTRACT["enums"]["state_kind"]
        layout_enum = contract.CONTRACT["enums"]["layout_decision"]
        event_enum = contract.CONTRACT["enums"]["event_decision"]
        for frame in frames:
            state = frame["rule_state"]
            state[RULE_INDEX["state_kind"]] = float(state_enum["full_frame"])
            state[RULE_INDEX["committed_layout"]] = float(
                layout_enum["no_target"]
            )
            state[RULE_INDEX["roi_generation"]] = 0
            state[RULE_INDEX["state_flags"]] &= ~STATE_ROI_LOCKED
            state[RULE_INDEX["promotion_flags"]] &= ~PROMOTE_ROI
            for name, value in zip(
                (
                    "committed_roi_x0",
                    "committed_roi_y0",
                    "committed_roi_x1",
                    "committed_roi_y1",
                ),
                (0.0, 0.0, 1.0, 1.0),
            ):
                state[RULE_INDEX[name]] = value
            state[RULE_INDEX["committed_roi_confidence"]] = 0.0
            state[RULE_INDEX["committed_mask_confidence"]] = 0.0
        frames[1]["rule_state"][RULE_INDEX["event_decision"]] = float(
            event_enum["same_shot"]
        )
        report = evaluator.evaluate(rendered, frames)
        codes = _failure_codes(report)
        self.assertIn("target_not_acquired", codes)
        self.assertTrue({
            "roi_iou_median_low",
            "roi_iou_p05_low",
            "target_coverage_p05_low",
            "active_layout_mismatch",
        }.isdisjoint(codes))

    def test_committed_wrong_roi_is_acquired_then_fails_quality(self):
        rendered = compositor.render_scenario(
            self.manifest, "video_sidebar_ad"
        )
        frames = _ideal_frames(rendered)
        for frame in frames[1:]:
            state = frame["rule_state"]
            for name, value in zip(
                (
                    "committed_roi_x0",
                    "committed_roi_y0",
                    "committed_roi_x1",
                    "committed_roi_y1",
                ),
                (0.01, 0.20, 0.20, 0.70),
            ):
                state[RULE_INDEX[name]] = value
        report = evaluator.evaluate(rendered, frames)
        codes = _failure_codes(report)
        self.assertNotIn("target_not_acquired", codes)
        self.assertEqual(
            report["layout_segments"][0]["metrics"]["acquisition_frame"],
            2,
        )
        self.assertIn("roi_iou_median_low", codes)
        self.assertIn("target_coverage_p05_low", codes)

    def test_full_frame_cannot_stage_committed_roi_before_activation(self):
        rendered = compositor.render_scenario(
            self.manifest, "video_sidebar_ad"
        )
        frames = _ideal_frames(rendered)
        state = frames[10]["rule_state"]
        state[RULE_INDEX["state_kind"]] = float(
            contract.CONTRACT["enums"]["state_kind"]["full_frame"]
        )
        state[RULE_INDEX["state_flags"]] &= ~STATE_ROI_LOCKED
        report = evaluator.evaluate(rendered, frames)
        self.assertIn(
            "state_flag_inconsistent",
            _failure_codes(report),
        )
        details = next(
            failure["detail"] for failure in report["failures"]
            if failure["code"] == "state_flag_inconsistent"
        )
        self.assertIn(
            "full_frame_committed_state_not_canonical",
            {finding["reason"] for finding in details},
        )

    def test_full_frame_accepts_canonical_identity_commit(self):
        rendered = compositor.render_scenario(
            self.manifest, "embedded_to_fullscreen"
        )
        frames = _ideal_frames(rendered)
        state = frames[46]["rule_state"]
        state[RULE_INDEX["committed_layout"]] = float(
            contract.CONTRACT["enums"]["layout_decision"][
                "identity_fullscreen"
            ]
        )
        report = evaluator.evaluate(rendered, frames)
        state_reasons = {
            finding["reason"]
            for failure in report["failures"]
            if failure["code"] == "state_flag_inconsistent"
            for finding in failure["detail"]
        }
        self.assertNotIn(
            "full_frame_committed_state_not_canonical",
            state_reasons,
        )

        state[RULE_INDEX["committed_mask_confidence"]] = 0.25
        report = evaluator.evaluate(rendered, frames)
        details = next(
            failure["detail"] for failure in report["failures"]
            if failure["code"] == "state_flag_inconsistent"
        )
        self.assertIn(
            "full_frame_committed_state_not_canonical",
            {finding["reason"] for finding in details},
        )

    def test_invalid_or_stale_output_cannot_earn_cut_credit(self):
        rendered = compositor.render_scenario(
            self.manifest, "video_sidebar_ad"
        )
        for name, mutate in (
            (
                "invalid",
                lambda state, prior: (
                    state.__setitem__(RULE_INDEX["output_valid"], 0.0),
                    state.__setitem__(
                        RULE_INDEX["state_flags"],
                        state[RULE_INDEX["state_flags"]] | STATE_FALLBACK,
                    ),
                ),
            ),
            (
                "stale",
                lambda state, prior: state.__setitem__(
                    RULE_INDEX["update_count"],
                    prior[RULE_INDEX["update_count"]],
                ),
            ),
            (
                "skipped",
                lambda state, prior: state.__setitem__(
                    RULE_INDEX["update_count"],
                    prior[RULE_INDEX["update_count"]] + 2,
                ),
            ),
        ):
            with self.subTest(name=name):
                frames = _ideal_frames(rendered)
                mutate(
                    frames[18]["rule_state"],
                    frames[17]["rule_state"],
                )
                report = evaluator.evaluate(rendered, frames)
                codes = _failure_codes(report)
                self.assertIn("controller_output_not_current", codes)
                self.assertIn("invalid_output_event_attempt", codes)
                self.assertIn("content_cut_missed", codes)
                self.assertEqual(
                    report["events"]["content_cut"]["matches"], []
                )
                self.assertNotIn(
                    19, report["events"]["content_cut"]["false"]
                )

    def test_invalid_output_cannot_earn_geometry_or_generation_credit(self):
        rendered = compositor.render_scenario(
            self.manifest, "embedded_to_fullscreen"
        )
        frames = _ideal_frames(rendered)
        reset = frames[44]["rule_state"]
        reset[RULE_INDEX["output_valid"]] = 0.0
        reset[RULE_INDEX["state_flags"]] |= STATE_FALLBACK
        for update_count, frame in enumerate(frames[45:], start=45):
            state = frame["rule_state"]
            state[RULE_INDEX["update_count"]] = update_count
            state[RULE_INDEX["roi_generation"]] = 1
        report = evaluator.evaluate(rendered, frames)
        codes = _failure_codes(report)
        self.assertIn("invalid_output_event_attempt", codes)
        self.assertIn("invalid_output_generation_attempt", codes)
        self.assertIn("geometry_event_missed", codes)
        self.assertIn("geometry_generation_missed", codes)
        self.assertEqual(
            report["events"]["geometry"]["generation_reset_matches"], []
        )

    def test_invalid_acquisition_is_diagnostic_only_until_valid_retry(self):
        rendered = compositor.render_scenario(
            self.manifest, "video_sidebar_ad"
        )
        frames = _ideal_frames(rendered)
        attempted = frames[1]["rule_state"]
        attempted[RULE_INDEX["output_valid"]] = 0.0
        attempted[RULE_INDEX["state_flags"]] |= STATE_FALLBACK

        retry = frames[2]["rule_state"]
        retry[RULE_INDEX["update_count"]] = 2
        retry[RULE_INDEX["promotion_flags"]] |= PROMOTE_ROI
        retry[RULE_INDEX["event_decision"]] = float(
            contract.CONTRACT["enums"]["event_decision"]["geometry_reset"]
        )
        for update_count, frame in enumerate(frames[3:], start=3):
            frame["rule_state"][RULE_INDEX["update_count"]] = update_count

        report = evaluator.evaluate(rendered, frames)
        self.assertIn(
            "invalid_output_generation_attempt",
            _failure_codes(report),
        )
        self.assertFalse(report["frames"][1]["active_roi"])
        self.assertEqual(
            report["events"]["geometry"]["generation_changes"], [3]
        )

    def test_reset_generation_must_return_to_zero(self):
        rendered = compositor.render_scenario(
            self.manifest, "embedded_to_fullscreen"
        )
        frames = _ideal_frames(rendered)
        frames[44]["rule_state"][RULE_INDEX["roi_generation"]] = 3
        for update_count, frame in enumerate(frames[45:], start=45):
            state = frame["rule_state"]
            state[RULE_INDEX["update_count"]] = update_count
            state[RULE_INDEX["roi_generation"]] = 1
        report = evaluator.evaluate(rendered, frames)
        codes = _failure_codes(report)
        self.assertIn("roi_generation_invalid", codes)
        self.assertIn("geometry_generation_missed", codes)

    def test_nonreset_roi_generation_must_advance_by_exactly_one(self):
        rendered = compositor.render_scenario(
            self.manifest, "video_sidebar_ad"
        )
        frames = _ideal_frames(rendered)
        for frame in frames[10:]:
            frame["rule_state"][RULE_INDEX["roi_generation"]] = 3
        report = evaluator.evaluate(rendered, frames)
        self.assertIn("roi_generation_invalid", _failure_codes(report))

    def test_roi_acquisition_requires_the_production_geometry_event(self):
        rendered = compositor.render_scenario(
            self.manifest, "video_sidebar_ad"
        )
        frames = _ideal_frames(rendered)
        frames[1]["rule_state"][RULE_INDEX["event_decision"]] = float(
            contract.CONTRACT["enums"]["event_decision"]["same_shot"]
        )
        report = evaluator.evaluate(rendered, frames)
        self.assertIn(
            "acquisition_geometry_event_missed",
            _failure_codes(report),
        )
        self.assertIn(
            "geometry_generation_event_mismatch",
            _failure_codes(report),
        )

    def test_every_generation_change_requires_same_update_geometry_event(self):
        rendered = compositor.render_scenario(
            self.manifest, "video_sidebar_ad"
        )
        frames = _ideal_frames(rendered)
        for frame in frames[2:]:
            frame["rule_state"][RULE_INDEX["roi_generation"]] = 2
        report = evaluator.evaluate(rendered, frames)
        self.assertIn(
            "geometry_generation_event_mismatch",
            _failure_codes(report),
        )
        self.assertEqual(
            report["events"]["geometry"]["missing_generation_events"],
            [3],
        )

    def test_generation_change_requires_roi_promotion(self):
        rendered = compositor.render_scenario(
            self.manifest, "video_sidebar_ad"
        )
        frames = _ideal_frames(rendered)
        frames[1]["rule_state"][RULE_INDEX["promotion_flags"]] &= (
            ~PROMOTE_ROI
        )
        report = evaluator.evaluate(rendered, frames)
        self.assertIn(
            "roi_generation_promotion_mismatch",
            _failure_codes(report),
        )

    def test_committed_geometry_change_requires_generation_change(self):
        rendered = compositor.render_scenario(
            self.manifest, "video_sidebar_ad"
        )
        frames = _ideal_frames(rendered)
        frames[10]["rule_state"][RULE_INDEX["committed_roi_x0"]] += 0.01
        report = evaluator.evaluate(rendered, frames)
        self.assertIn(
            "committed_geometry_generation_mismatch",
            _failure_codes(report),
        )

    def test_geometry_reset_requires_event_and_generation(self):
        rendered = compositor.render_scenario(
            self.manifest, "embedded_to_fullscreen"
        )
        frames = _ideal_frames(rendered)
        reset = frames[44]["rule_state"]
        reset[RULE_INDEX["event_decision"]] = float(
            contract.CONTRACT["enums"]["event_decision"]["same_shot"]
        )
        reset[RULE_INDEX["reset_flags"]] &= ~RESET_GEOMETRY
        report = evaluator.evaluate(rendered, frames)
        codes = {finding["code"] for finding in report["failures"]}
        self.assertIn("geometry_event_missed", codes)
        self.assertIn("geometry_generation_missed", codes)

    def test_reset_epoch_cannot_retain_roi_and_must_be_label_authorized(self):
        rendered = compositor.render_scenario(
            self.manifest, "video_sidebar_ad"
        )
        frames = _ideal_frames(rendered)
        stale = frames[10]["rule_state"]
        stale[RULE_INDEX["reset_flags"]] = RESET_LAYOUT
        stale[RULE_INDEX["update_count"]] = 1
        for update_count, frame in enumerate(frames[11:], start=11):
            frame["rule_state"][RULE_INDEX["update_count"]] = update_count
        report = evaluator.evaluate(rendered, frames)
        self.assertFalse(report["frames"][10]["active_roi"])
        self.assertIn(
            "controller_output_not_current", _failure_codes(report)
        )
        self.assertIn(
            "invalid_output_generation_attempt", _failure_codes(report)
        )

        frames = _ideal_frames(rendered)
        reset = frames[10]["rule_state"]
        reset[RULE_INDEX["reset_flags"]] = RESET_LAYOUT
        reset[RULE_INDEX["update_count"]] = 1
        reset[RULE_INDEX["roi_generation"]] = 0
        reset[RULE_INDEX["state_kind"]] = float(
            contract.CONTRACT["enums"]["state_kind"]["full_frame"]
        )
        reset[RULE_INDEX["committed_layout"]] = float(
            contract.CONTRACT["enums"]["layout_decision"]["no_target"]
        )
        reset[RULE_INDEX["state_flags"]] &= ~STATE_ROI_LOCKED
        for name, value in zip(
            (
                "committed_roi_x0",
                "committed_roi_y0",
                "committed_roi_x1",
                "committed_roi_y1",
            ),
            (0.0, 0.0, 1.0, 1.0),
        ):
            reset[RULE_INDEX[name]] = value
        reset[RULE_INDEX["committed_roi_confidence"]] = 0.0
        reset[RULE_INDEX["committed_mask_confidence"]] = 0.0
        for update_count, frame in enumerate(frames[11:], start=2):
            frame["rule_state"][RULE_INDEX["update_count"]] = update_count
        report = evaluator.evaluate(rendered, frames)
        self.assertIn(
            "unexpected_controller_reset", _failure_codes(report)
        )

    def test_embedded_to_fullscreen_scores_both_layout_segments(self):
        rendered = compositor.render_scenario(
            self.manifest, "embedded_to_fullscreen"
        )
        report = evaluator.evaluate(rendered, _ideal_frames(rendered))
        self.assertTrue(report["pass"], report["failures"])
        self.assertEqual(
            [segment["layout"] for segment in report["layout_segments"]],
            ["primary_video", "identity_fullscreen"],
        )
        self.assertEqual(
            report["layout_segments"][1]["metrics"]["active_frame_count"],
            0,
        )
        self.assertEqual(
            report["events"]["geometry"]["generation_reset_matches"],
            [{"expected": 45, "predicted": 45}],
        )
        reset_row = report["frames"][44]
        self.assertEqual(reset_row["update_count"], 1)
        self.assertEqual(reset_row["roi_generation"], 0)
        self.assertNotIn(
            "stale_update_count", reset_row["current_output_reasons"]
        )

    def test_geometry_reset_splits_and_reacquires_the_same_layout(self):
        original = compositor.render_scenario(
            self.manifest, "static_dense_collage"
        )
        recipe = copy.deepcopy(original.recipe)
        timeline = copy.deepcopy(original.timeline)
        recipe["events"]["geometry_reset_frames"] = [13]
        for index, row in enumerate(timeline):
            row["geometry_reset"] = index == 12
            row["geometry_generation"] = 1 if index >= 12 else 0
        rendered = compositor.RenderedSequence(
            recipe=recipe,
            arrays=original.arrays,
            timeline=timeline,
        )
        frames = _ideal_frames(rendered)
        for index in (13, 14):
            state = frames[index]["rule_state"]
            state[RULE_INDEX["state_kind"]] = float(
                contract.CONTRACT["enums"]["state_kind"]["full_frame"]
            )
            state[RULE_INDEX["committed_layout"]] = float(
                contract.CONTRACT["enums"]["layout_decision"]["no_target"]
            )
            state[RULE_INDEX["state_flags"]] &= ~STATE_ROI_LOCKED
            state[RULE_INDEX["roi_generation"]] = 0
            for name, value in zip(
                (
                    "committed_roi_x0",
                    "committed_roi_y0",
                    "committed_roi_x1",
                    "committed_roi_y1",
                ),
                (0.0, 0.0, 1.0, 1.0),
            ):
                state[RULE_INDEX[name]] = value
            state[RULE_INDEX["committed_roi_confidence"]] = 0.0
            state[RULE_INDEX["committed_mask_confidence"]] = 0.0
            state[RULE_INDEX["event_decision"]] = float(
                contract.CONTRACT["enums"]["event_decision"]["same_shot"]
            )
            state[RULE_INDEX["promotion_flags"]] &= ~PROMOTE_ROI
        acquired = frames[15]["rule_state"]
        acquired[RULE_INDEX["event_decision"]] = float(
            contract.CONTRACT["enums"]["event_decision"]["geometry_reset"]
        )
        acquired[RULE_INDEX["promotion_flags"]] |= PROMOTE_ROI

        report = evaluator.evaluate(rendered, frames)
        self.assertTrue(report["pass"], report["failures"])
        self.assertEqual(
            [segment["layout"] for segment in report["layout_segments"]],
            ["content_collage", "content_collage"],
        )
        self.assertEqual(
            [
                segment["metrics"]["acquisition_frame"]
                for segment in report["layout_segments"]
            ],
            [2, 16],
        )

    def test_identity_fullscreen_rejects_roi_after_reset_tolerance(self):
        rendered = compositor.render_scenario(
            self.manifest, "embedded_to_fullscreen"
        )
        frames = _ideal_frames(rendered)
        state = frames[46]["rule_state"]
        source = frames[1]["rule_state"]
        state[RULE_INDEX["state_kind"]] = float(
            contract.CONTRACT["enums"]["state_kind"]["video"]
        )
        state[RULE_INDEX["committed_layout"]] = float(
            contract.CONTRACT["enums"]["layout_decision"]["primary_video"]
        )
        state[RULE_INDEX["roi_generation"]] = 1
        state[RULE_INDEX["state_flags"]] |= STATE_ROI_LOCKED
        for name in (
            "committed_roi_x0",
            "committed_roi_y0",
            "committed_roi_x1",
            "committed_roi_y1",
        ):
            state[RULE_INDEX[name]] = source[RULE_INDEX[name]]
        report = evaluator.evaluate(rendered, frames)
        self.assertIn(
            "identity_fullscreen_selected_roi", _failure_codes(report)
        )

    def test_identity_fullscreen_checks_current_layout_decision(self):
        rendered = compositor.render_scenario(
            self.manifest, "embedded_to_fullscreen"
        )
        frames = _ideal_frames(rendered)
        frames[46]["rule_state"][RULE_INDEX["layout_decision"]] = float(
            contract.CONTRACT["enums"]["layout_decision"]["primary_video"]
        )
        report = evaluator.evaluate(rendered, frames)
        self.assertIn(
            "full_frame_layout_decision_mismatch", _failure_codes(report)
        )

    def test_scroll_hold_enters_freezes_retains_and_releases(self):
        rendered = compositor.render_scenario(
            self.manifest, "collage_scroll"
        )
        report = evaluator.evaluate(rendered, _ideal_frames(rendered))
        self.assertTrue(report["pass"], report["failures"])
        scroll = report["events"]["scroll"]
        self.assertEqual(scroll["entry_frame"], 10)
        self.assertEqual(scroll["release_frame"], 21)
        self.assertEqual(scroll["history_violations"], [])
        self.assertEqual(scroll["roi_violations"], [])

    def test_no_scroll_reports_false_instead_of_vacuous_detection(self):
        rendered = compositor.render_scenario(
            self.manifest, "video_sidebar_ad"
        )
        report = evaluator.evaluate(rendered, _ideal_frames(rendered))
        scroll = report["events"]["scroll"]
        self.assertEqual(scroll["expected_interval"], [])
        self.assertFalse(scroll["detected"])
        self.assertFalse(scroll["event_detected"])

    def test_scroll_entry_uses_stream_time_at_every_frame_rate(self):
        original = compositor.render_scenario(
            self.manifest, "collage_scroll"
        )
        for fps in (24, 60, 90):
            with self.subTest(fps=fps):
                release_updates = math.ceil(
                    evaluator.SCROLL_RELEASE_SECONDS * fps
                )
                minimum_frames = (
                    original.recipe["events"]["scroll_frames"][1] +
                    release_updates +
                    evaluator.load_thresholds()[
                        "event_tolerance_updates"
                    ]
                )
                rendered = _with_fps(
                    original,
                    fps,
                    minimum_frames=max(
                        len(original.timeline),
                        minimum_frames,
                    ),
                )
                report = evaluator.evaluate(
                    rendered, _ideal_frames(rendered)
                )
                self.assertTrue(report["pass"], report["failures"])
                scroll = report["events"]["scroll"]
                expected_updates = 1 + math.ceil(
                    evaluator.SCROLL_ENTRY_SECONDS * fps
                )
                self.assertEqual(
                    scroll["entry_updates"],
                    expected_updates,
                )
                self.assertEqual(
                    scroll["entry_nominal"],
                    8 + expected_updates - 1,
                )
                self.assertEqual(
                    scroll["entry_frame"],
                    8 + expected_updates - 1,
                )

    def test_single_update_scroll_does_not_enter_hold(self):
        original = compositor.render_scenario(
            self.manifest, "collage_scroll"
        )
        rendered = _with_fps(original, 60)
        recipe = copy.deepcopy(rendered.recipe)
        recipe["events"]["scroll_frames"] = [8, 8]
        timeline = copy.deepcopy(rendered.timeline)
        for frame_number, row in enumerate(timeline, start=1):
            row["scroll"] = frame_number == 8
        rendered = compositor.RenderedSequence(
            recipe=recipe,
            arrays=rendered.arrays,
            timeline=timeline,
        )
        frames = _ideal_frames(rendered)
        report = evaluator.evaluate(rendered, frames)
        self.assertTrue(report["pass"], report["failures"])
        scroll = report["events"]["scroll"]
        self.assertFalse(scroll["hold_required"])
        self.assertEqual(
            scroll["entry_updates"],
            1 + math.ceil(evaluator.SCROLL_ENTRY_SECONDS * 60),
        )
        self.assertIsNone(scroll["entry_frame"])
        self.assertEqual(scroll["hold_frames"], [])
        self.assertIsNone(scroll["release_frame"])
        self.assertEqual(scroll["predicted"], [8])
        self.assertTrue(scroll["detected"])

    def test_scroll_hold_is_required_and_history_must_freeze(self):
        rendered = compositor.render_scenario(
            self.manifest, "collage_scroll"
        )
        frames = _ideal_frames(rendered)
        for frame in frames:
            state = frame["rule_state"]
            if int(state[RULE_INDEX["state_flags"]]) & STATE_SCROLL_HOLD:
                state[RULE_INDEX["state_flags"]] &= ~STATE_SCROLL_HOLD
        report = evaluator.evaluate(rendered, frames)
        codes = _failure_codes(report)
        self.assertIn("scroll_hold_not_entered", codes)
        self.assertTrue({
            "scroll_hold_release_early",
            "scroll_hold_release_late",
            "scroll_hold_release_unobservable",
        }.isdisjoint(codes))

        frames = _ideal_frames(rendered)
        held = frames[9]["rule_state"]
        held[RULE_INDEX["promotion_flags"]] |= PROMOTE_DEPTH
        held[RULE_INDEX["history_flags"]] |= HISTORY_DEPTH_WRITE
        held[RULE_INDEX["pop_strength"]] = 0.5
        held[RULE_INDEX["acquisition_dwell_s"]] = 0.25
        report = evaluator.evaluate(rendered, frames)
        self.assertIn("scroll_history_not_frozen", _failure_codes(report))

    def test_scroll_entry_clears_pending_geometry_instead_of_freezing_it(self):
        rendered = compositor.render_scenario(
            self.manifest, "collage_scroll"
        )
        frames = _ideal_frames(rendered)
        pending = frames[6]["rule_state"]
        pending[RULE_INDEX["acquisition_roi_x0"]] = 0.20
        pending[RULE_INDEX["acquisition_roi_x1"]] = 0.80
        pending[RULE_INDEX["acquisition_score"]] = 0.75
        pending[RULE_INDEX["acquisition_dwell_s"]] = 0.10
        pending[RULE_INDEX["acquisition_layout"]] = float(
            contract.CONTRACT["enums"]["layout_decision"]["content_collage"]
        )
        pending[RULE_INDEX["acquisition_valid"]] = 1.0

        report = evaluator.evaluate(rendered, frames)
        self.assertTrue(report["pass"], report["failures"])

        entry = frames[9]["rule_state"]
        entry[RULE_INDEX["acquisition_dwell_s"]] = 0.10
        entry[RULE_INDEX["acquisition_valid"]] = 1.0
        report = evaluator.evaluate(rendered, frames)
        details = next(
            failure["detail"] for failure in report["failures"]
            if failure["code"] == "scroll_history_not_frozen"
        )
        self.assertIn(
            "pending_geometry_not_cleared",
            {finding["reason"] for finding in details},
        )

    def test_scroll_hold_entry_requires_causal_scroll_evidence(self):
        rendered = compositor.render_scenario(
            self.manifest, "collage_scroll"
        )
        frames = _ideal_frames(rendered)
        event_enum = contract.CONTRACT["enums"]["event_decision"]
        for frame in frames:
            frame["rule_state"][RULE_INDEX["scroll_confidence"]] = 0.0
        for frame in frames[6:10]:
            frame["rule_state"][RULE_INDEX["event_decision"]] = float(
                event_enum["same_shot"]
            )
        frames[15]["rule_state"][RULE_INDEX["scroll_confidence"]] = 1.0
        report = evaluator.evaluate(rendered, frames)
        self.assertIn(
            "scroll_hold_entry_without_evidence",
            _failure_codes(report),
        )

    def test_scroll_hold_must_retain_roi_and_generation(self):
        rendered = compositor.render_scenario(
            self.manifest, "collage_scroll"
        )
        frames = _ideal_frames(rendered)
        for frame in frames[9:]:
            frame["rule_state"][RULE_INDEX["roi_generation"]] = 2
        frames[9]["rule_state"][RULE_INDEX["committed_roi_x0"]] += 0.01
        report = evaluator.evaluate(rendered, frames)
        self.assertIn("scroll_roi_not_retained", _failure_codes(report))

    def test_scroll_hold_must_release_within_the_resolver_bound(self):
        rendered = compositor.render_scenario(
            self.manifest, "collage_scroll"
        )
        frames = _ideal_frames(rendered)
        for index in range(20, len(frames)):
            state = frames[index]["rule_state"]
            prior = frames[index - 1]["rule_state"]
            state[RULE_INDEX["state_flags"]] |= STATE_SCROLL_HOLD
            state[RULE_INDEX["promotion_flags"]] = PROMOTE_LAYOUT
            state[RULE_INDEX["history_flags"]] = (
                HISTORY_LAYOUT_READ |
                HISTORY_LAYOUT_WRITE |
                HISTORY_DEPTH_READ
            )
            for name in evaluator.SCROLL_FROZEN_FIELDS:
                state[RULE_INDEX[name]] = prior[RULE_INDEX[name]]
        report = evaluator.evaluate(rendered, frames)
        codes = _failure_codes(report)
        self.assertIn("scroll_hold_release_late", codes)
        self.assertIn("unexpected_scroll_hold", codes)

    def test_scroll_hold_cannot_release_early_or_reenter(self):
        rendered = compositor.render_scenario(
            self.manifest, "collage_scroll"
        )
        frames = _ideal_frames(rendered)
        for index in range(17, 20):
            state = frames[index]["rule_state"]
            state[RULE_INDEX["state_flags"]] &= ~STATE_SCROLL_HOLD
        report = evaluator.evaluate(rendered, frames)
        self.assertIn(
            "scroll_hold_release_early", _failure_codes(report)
        )

        frames = _ideal_frames(rendered)
        released = frames[19]["rule_state"]
        released[RULE_INDEX["state_flags"]] &= ~STATE_SCROLL_HOLD
        reentered = frames[20]["rule_state"]
        prior = frames[19]["rule_state"]
        reentered[RULE_INDEX["state_flags"]] |= STATE_SCROLL_HOLD
        reentered[RULE_INDEX["promotion_flags"]] = PROMOTE_LAYOUT
        reentered[RULE_INDEX["history_flags"]] = (
            HISTORY_LAYOUT_READ |
            HISTORY_LAYOUT_WRITE |
            HISTORY_DEPTH_READ
        )
        for name in evaluator.SCROLL_FROZEN_FIELDS:
            reentered[RULE_INDEX[name]] = prior[RULE_INDEX[name]]
        report = evaluator.evaluate(rendered, frames)
        self.assertIn("scroll_hold_reentered", _failure_codes(report))

    def test_rejects_depth_reuse_that_weakens_exact_event_alignment(self):
        rendered = compositor.render_scenario(
            self.manifest, "partial_player"
        )
        with self.assertRaisesRegex(
            evaluator.SceneControllerEvalError, "depth_reuse_interval=1"
        ):
            evaluator.evaluate(
                rendered, _ideal_frames(rendered), depth_reuse_interval=2
            )

    def test_threshold_contract_rejects_bool_as_integer(self):
        thresholds = copy.deepcopy(evaluator.load_thresholds())
        thresholds["max_unsafe_pixels"] = False
        rendered = compositor.render_scenario(
            self.manifest, "partial_player"
        )
        with self.assertRaisesRegex(
            evaluator.SceneControllerEvalError, "max_unsafe_pixels"
        ):
            evaluator.evaluate(
                rendered, _ideal_frames(rendered), thresholds=thresholds
            )

        thresholds = copy.deepcopy(evaluator.load_thresholds())
        thresholds["schema"] = True
        with self.assertRaisesRegex(
            evaluator.SceneControllerEvalError, "schema"
        ):
            evaluator.evaluate(
                rendered, _ideal_frames(rendered), thresholds=thresholds
            )

    def test_generated_clip_contract_authenticates_exact_pixels(self):
        rendered = compositor.render_scenario(
            self.manifest, "partial_player"
        )
        with tempfile.TemporaryDirectory() as root:
            output = compositor.materialize(
                rendered, Path(root) / "clip", emit_frames=True
            )
            authenticated = evaluator.load_generated_clip_contract(output)
            self.assertIsNotNone(authenticated)
            self.assertEqual(authenticated.recipe["name"], "partial_player")

            frame_path = output / "frame_00007.png"
            with Image.open(frame_path) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            pixels[0, 0, 0] ^= 1
            Image.fromarray(pixels).save(frame_path)
            with self.assertRaisesRegex(
                evaluator.SceneControllerEvalError, "pixels disagree"
            ):
                evaluator.load_generated_clip_contract(output)

    def test_generated_clip_contract_rejects_bool_schema_and_file_hash_drift(self):
        rendered = compositor.render_scenario(
            self.manifest, "partial_player"
        )
        with tempfile.TemporaryDirectory() as root:
            output = compositor.materialize(
                rendered, Path(root) / "clip", emit_frames=True
            )
            meta_path = output / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["scene_controller_contract"]["schema"] = True
            _write_json(meta_path, meta)
            with self.assertRaisesRegex(
                evaluator.SceneControllerEvalError, "schema mismatch"
            ):
                evaluator.load_generated_clip_contract(output)

            meta["scene_controller_contract"]["schema"] = (
                compositor.ARTIFACT_SCHEMA
            )
            _write_json(meta_path, meta)
            for name in (
                "recipe.json",
                "sequence.npz",
                "timeline.jsonl",
                "artifact.json",
            ):
                with self.subTest(name=name):
                    path = output / name
                    original = path.read_bytes()
                    path.write_bytes(original + b" ")
                    try:
                        with self.assertRaisesRegex(
                            evaluator.SceneControllerEvalError,
                            "hash mismatch",
                        ):
                            evaluator.load_generated_clip_contract(output)
                    finally:
                        path.write_bytes(original)

    def test_generated_clip_contract_rejects_rehashed_semantic_drift(self):
        rendered = compositor.render_scenario(
            self.manifest, "partial_player"
        )
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)

            recipe_output = compositor.materialize(
                rendered, root_path / "recipe", emit_frames=True
            )
            recipe_path = recipe_output / "recipe.json"
            recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            recipe["description"] += " tampered"
            recipe_path.write_bytes(compositor._canonical_json(recipe) + b"\n")
            recipe_meta_path = recipe_output / "meta.json"
            recipe_meta = json.loads(
                recipe_meta_path.read_text(encoding="utf-8")
            )
            recipe_meta["scene_controller_contract"][
                "recipe_file_sha256"
            ] = _sha256(recipe_path)
            artifact_path = recipe_output / "artifact.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["recipe_file_sha256"] = _sha256(recipe_path)
            _write_json(artifact_path, artifact)
            recipe_meta["scene_controller_contract"][
                "artifact_sha256"
            ] = _sha256(artifact_path)
            _write_json(recipe_meta_path, recipe_meta)
            with self.assertRaisesRegex(
                evaluator.SceneControllerEvalError,
                "recipe file disagrees",
            ):
                evaluator.load_generated_clip_contract(recipe_output)

            timeline_output = compositor.materialize(
                rendered, root_path / "timeline", emit_frames=True
            )
            timeline_path = timeline_output / "timeline.jsonl"
            timeline = [
                json.loads(line)
                for line in timeline_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            timeline[0]["source_timestamp_ns"] += 1
            timeline_path.write_bytes(b"".join(
                compositor._canonical_json(row) + b"\n"
                for row in timeline
            ))
            timeline_meta_path = timeline_output / "meta.json"
            timeline_meta = json.loads(
                timeline_meta_path.read_text(encoding="utf-8")
            )
            timeline_meta["scene_controller_contract"][
                "timeline_sha256"
            ] = _sha256(timeline_path)
            artifact_path = timeline_output / "artifact.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["timeline_sha256"] = _sha256(timeline_path)
            _write_json(artifact_path, artifact)
            timeline_meta["scene_controller_contract"][
                "artifact_sha256"
            ] = _sha256(artifact_path)
            _write_json(timeline_meta_path, timeline_meta)
            with self.assertRaisesRegex(
                evaluator.SceneControllerEvalError,
                "timeline disagrees",
            ):
                evaluator.load_generated_clip_contract(timeline_output)

            labels_output = compositor.materialize(
                rendered, root_path / "labels", emit_frames=True
            )
            labels_path = labels_output / "sequence.npz"
            with np.load(labels_path, allow_pickle=False) as archive:
                arrays = {
                    name: archive[name].copy() for name in archive.files
                }
            arrays["target"][0, 0, 0] = ~arrays["target"][0, 0, 0]
            compositor._write_deterministic_npz(labels_path, arrays)
            labels_meta_path = labels_output / "meta.json"
            labels_meta = json.loads(
                labels_meta_path.read_text(encoding="utf-8")
            )
            labels_meta["scene_controller_contract"][
                "labels_sha256"
            ] = _sha256(labels_path)
            artifact_path = labels_output / "artifact.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["sequence_npz_sha256"] = _sha256(labels_path)
            artifact["arrays"]["target"]["sha256"] = (
                compositor.array_digest(arrays["target"])
            )
            _write_json(artifact_path, artifact)
            labels_meta["scene_controller_contract"][
                "artifact_sha256"
            ] = _sha256(artifact_path)
            _write_json(labels_meta_path, labels_meta)
            with self.assertRaisesRegex(
                evaluator.SceneControllerEvalError,
                "array 'target' disagrees",
            ):
                evaluator.load_generated_clip_contract(labels_output)

            artifact_output = compositor.materialize(
                rendered, root_path / "artifact", emit_frames=True
            )
            artifact_path = artifact_output / "artifact.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["arrays"]["target"]["shape"][0] += 1
            _write_json(artifact_path, artifact)
            artifact_meta_path = artifact_output / "meta.json"
            artifact_meta = json.loads(
                artifact_meta_path.read_text(encoding="utf-8")
            )
            artifact_meta["scene_controller_contract"][
                "artifact_sha256"
            ] = _sha256(artifact_path)
            _write_json(artifact_meta_path, artifact_meta)
            with self.assertRaisesRegex(
                evaluator.SceneControllerEvalError,
                "artifact descriptor arrays disagree",
            ):
                evaluator.load_generated_clip_contract(artifact_output)

    def test_ordinary_frame_directory_has_no_implicit_gate(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(
                evaluator.load_generated_clip_contract(Path(root))
            )


if __name__ == "__main__":
    unittest.main()
