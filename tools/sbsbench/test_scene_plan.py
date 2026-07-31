#!/usr/bin/env python3

import math
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import scene_plan as planning  # noqa: E402


def sample(
    index: int,
    *,
    pulse: bool = False,
    depth: float = 0.05,
    raw: float = 0.02,
    structural: float = 0.01,
    edge: float = 0.10,
    anchor: float = 1.0,
    ready: float = 1.0,
    flags: int = 0,
    depth_updated: bool = True,
    requires_previous: bool = False,
) -> dict:
    return {
        "frame_id": f"{index + 1:010d}",
        "source_index": index,
        "depth_updated": depth_updated,
        "hard_cut_pulse": pulse,
        "current_depth_change_fraction": depth,
        "raw_rgb_change_fraction": raw,
        "structural_change_fraction": structural,
        "current_edge_fraction": edge,
        "current_zero_anchor_candidate_shift_px": anchor,
        "zero_anchor_shift_px": anchor + 0.25,
        "zero_anchor_valid": 1.0,
        "depth_ready": ready,
        "initialized": 1.0,
        "scene_age": float(index),
        "valid_depth_fraction": 1.0,
        "range_collapsed": 0.0,
        "current_structural_support_fraction": 0.5,
        "previous_structural_support_fraction": 0.5,
        "common_structural_support_fraction": 0.5,
        "analysis_flags": flags,
        "requires_previous_packed_frame": requires_previous,
    }


def config(**overrides) -> planning.ScenePlannerConfig:
    values = {
        "pop_strength": 1.0,
        "adaptive_pop": True,
        "adaptive_pop_max": 2.0,
        "zero_plane": "median",
        "lookbehind_depth_updates": 2,
        "lookahead_depth_updates": 2,
        "duplicate_pulse_distance_updates": 1,
        "settle_depth_updates": 0,
        "minimum_scene_frames": 2,
        "max_open_cache_bytes": 1024 * 1024,
    }
    values.update(overrides)
    return planning.ScenePlannerConfig(**values)


class StreamingScenePlannerTests(unittest.TestCase):
    def test_waits_for_future_evidence_before_committing(self):
        planner = planning.StreamingScenePlanner(config())
        for index in range(7):
            result = planner.feed(
                sample(index, pulse=index == 3, depth=0.65 if index == 3 else 0.05)
            )
            self.assertEqual(result, [])
        result = planner.feed(sample(7))
        self.assertEqual(len(result), 1)
        scene = result[0]
        self.assertEqual(
            (scene["start_sequence"], scene["end_sequence_exclusive"]),
            (1, 4),
        )
        self.assertEqual(scene["boundary"]["decision"], "confirmed")
        self.assertEqual(scene["boundary"]["final_sequence"], 4)

    def test_moves_to_stronger_correlated_transition_in_lookbehind(self):
        planner = planning.StreamingScenePlanner(config())
        finalized = []
        for index in range(8):
            if index == 2:
                frame = sample(
                    index,
                    depth=0.40,
                    raw=0.95,
                    structural=0.06,
                    flags=planning.ANALYSIS_APPEARANCE_PROPOSAL,
                )
            elif index == 3:
                frame = sample(index, pulse=True, depth=0.60)
            else:
                frame = sample(index)
            finalized.extend(planner.feed(frame))
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0]["end_sequence_exclusive"], 3)
        self.assertEqual(
            finalized[0]["boundary"]["decision"],
            "moved_to_correlated_evidence",
        )
        self.assertEqual(finalized[0]["boundary"]["revision_updates"], -1)

    def test_producer_localized_appearance_can_move_boundary(self):
        def run(producer_qualified: bool):
            planner = planning.StreamingScenePlanner(config())
            finalized = []
            for index in range(8):
                if index == 2:
                    frame = sample(
                        index,
                        depth=0.3864,
                        raw=0.2019,
                        structural=0.0396,
                        flags=(
                            planning.ANALYSIS_APPEARANCE_PROPOSAL
                            if producer_qualified else 0
                        ),
                    )
                elif index == 3:
                    frame = sample(index, pulse=True, depth=0.61)
                else:
                    frame = sample(index)
                finalized.extend(planner.feed(frame))
            return finalized

        qualified = run(True)
        self.assertEqual(len(qualified), 1)
        self.assertEqual(qualified[0]["end_sequence_exclusive"], 3)
        self.assertTrue(
            qualified[0]["boundary"]["evidence_window"]["selected"][
                "appearance_qualified"
            ]
        )
        self.assertEqual(
            qualified[0]["boundary"]["decision"],
            "moved_to_correlated_evidence",
        )

        unqualified = run(False)
        self.assertEqual(len(unqualified), 1)
        self.assertEqual(unqualified[0]["end_sequence_exclusive"], 4)
        self.assertFalse(
            unqualified[0]["boundary"]["evidence_window"]["selected"][
                "appearance_qualified"
            ]
        )

    def test_duplicate_pulses_merge_but_distinct_close_cuts_survive(self):
        planner = planning.StreamingScenePlanner(config(
            lookahead_depth_updates=1,
            duplicate_pulse_distance_updates=1,
        ))
        scenes = []
        for index in range(12):
            pulse = index in {3, 4, 8}
            scenes.extend(planner.feed(
                sample(index, pulse=pulse, depth=0.70 if pulse else 0.05)
            ))
        scenes.extend(planner.finish())
        self.assertEqual(
            [(scene["start_sequence"], scene["end_sequence_exclusive"])
             for scene in scenes],
            [(1, 4), (4, 9), (9, 13)],
        )
        self.assertEqual(
            scenes[0]["boundary"]["decision"],
            "merged_duplicate_proposals",
        )
        self.assertEqual(
            scenes[0]["boundary"]["proposal_sequences"], [4, 5])

    def test_supported_flash_return_rejects_provisional_cut(self):
        planner = planning.StreamingScenePlanner(config())
        veto = (
            planning.ANALYSIS_APPEARANCE_PROPOSAL |
            planning.ANALYSIS_EXPOSURE_LIKE |
            planning.ANALYSIS_APPEARANCE_VETO
        )
        same_return = planning.ANALYSIS_SAME_SCENE_RETURN
        for index in range(7):
            flags = veto if index == 3 else same_return if index == 5 else 0
            planner.feed(sample(
                index,
                pulse=index == 3,
                depth=0.70 if index == 3 else 0.05,
                raw=0.95 if index == 3 else 0.0,
                structural=0.0,
                flags=flags,
            ))
        scenes = planner.finish()
        self.assertEqual(len(scenes), 1)
        self.assertEqual((scenes[0]["start_sequence"],
                          scenes[0]["end_sequence_exclusive"]), (1, 8))
        rejected = planner.boundary_revisions[0]
        self.assertFalse(rejected["accepted"])
        self.assertEqual(
            rejected["decision"], "rejected_supported_flash_return")

    def test_held_depth_cut_pulse_is_not_reproposed(self):
        planner = planning.StreamingScenePlanner(config(
            lookbehind_depth_updates=0,
            lookahead_depth_updates=1,
        ))
        planner.feed(sample(0))
        planner.feed(sample(1))
        planner.feed(sample(2, pulse=True, depth=0.70))
        planner.feed(sample(
            3, pulse=True, depth=0.70, depth_updated=False))
        self.assertEqual(planner.pending_proposal_count, 1)
        scenes = planner.feed(sample(4))
        self.assertEqual(scenes[0]["boundary"]["proposal_sequences"], [3])

    def test_overlapping_proposal_windows_merge_before_commit(self):
        planner = planning.StreamingScenePlanner(config(
            lookbehind_depth_updates=2,
            lookahead_depth_updates=2,
            duplicate_pulse_distance_updates=0,
        ))
        finalized = []
        for index in range(13):
            pulse = index in {3, 7}
            finalized.extend(planner.feed(sample(
                index, pulse=pulse, depth=0.70 if pulse else 0.05)))
            if index < 11:
                self.assertEqual(finalized, [])
        self.assertEqual(len(finalized), 1)
        self.assertEqual(
            finalized[0]["boundary"]["proposal_sequences"], [4, 8])
        self.assertEqual(
            finalized[0]["boundary"]["decision"],
            "merged_duplicate_proposals",
        )

    def test_unsupported_proposal_rejects_and_geometry_only_peak_can_move(self):
        rejected = planning.StreamingScenePlanner(config(
            lookbehind_depth_updates=0,
            lookahead_depth_updates=1,
        ))
        for index in range(5):
            rejected.feed(sample(
                index,
                pulse=index == 2,
                depth=0.01,
                raw=0.01,
                structural=0.001,
            ))
        rejected.finish()
        self.assertEqual(
            rejected.boundary_revisions[0]["decision"],
            "rejected_unsupported_proposal",
        )

        moved = planning.StreamingScenePlanner(config())
        scenes = []
        for index in range(9):
            if index == 2:
                frame = sample(index, depth=0.90)
            elif index == 3:
                frame = sample(index, pulse=True, depth=0.61)
            else:
                frame = sample(index)
            scenes.extend(moved.feed(frame))
        self.assertEqual(scenes[0]["boundary"]["final_sequence"], 3)
        self.assertEqual(
            scenes[0]["boundary"]["decision"],
            "moved_to_correlated_evidence",
        )

    def test_eof_resolves_with_truncated_right_context(self):
        planner = planning.StreamingScenePlanner(config(
            lookahead_depth_updates=5))
        for index in range(6):
            planner.feed(sample(
                index, pulse=index == 4, depth=0.70 if index == 4 else 0.05))
        scenes = planner.finish()
        self.assertEqual(len(scenes), 2)
        self.assertTrue(scenes[0]["boundary"]["truncated"])
        self.assertEqual(scenes[0]["boundary"]["final_sequence"], 5)
        self.assertEqual(scenes[1]["boundary"]["decision"], "end_of_stream")

    def test_whole_scene_camera_uses_high_risk_quantile_and_median_anchor(self):
        planner = planning.StreamingScenePlanner(config())
        edges = [0.02, 0.04, 0.08, 0.12, 0.20]
        anchors = [4.0, 1.0, 3.0, 2.0, 8.0]
        for index, (edge, anchor) in enumerate(zip(edges, anchors)):
            planner.feed(sample(index, edge=edge, anchor=anchor))
        scene = planner.finish()[0]
        expected_risk = planning._quantile(edges, 0.90)
        expected_confidence = 1.0 - planning._smoothstep(
            planning.POP_RISK_LOW, planning.POP_RISK_HIGH, expected_risk)
        self.assertAlmostEqual(
            scene["absolute_pop_strength"], 1.0 + expected_confidence)
        self.assertEqual(scene["zero_anchor_shift_px"], 3.0)
        self.assertEqual(
            scene["render"]["pop_origin"],
            "whole-finalized-scene-edge-risk",
        )
        self.assertEqual(
            scene["render"]["zero_origin"],
            "whole-finalized-scene-median-candidate",
        )
        self.assertEqual(scene["evidence"]["valid_edge_sample_count"], 5)

    def test_unsettled_short_scene_falls_back_conservatively(self):
        planner = planning.StreamingScenePlanner(config(settle_depth_updates=8))
        for index in range(3):
            planner.feed(sample(index, edge=0.0, anchor=5.0))
        scene = planner.finish()[0]
        self.assertEqual(scene["absolute_pop_strength"], 1.0)
        self.assertEqual(
            scene["render"]["pop_origin"], "conservative-floor-fallback")
        self.assertEqual(scene["zero_anchor_shift_px"], 5.25)
        self.assertEqual(
            scene["render"]["zero_origin"], "production-latched-fallback")

    def test_invalid_collapsed_samples_cannot_select_camera(self):
        planner = planning.StreamingScenePlanner(config())
        for index in range(4):
            frame = sample(index, edge=0.0, anchor=8.0)
            frame.update({
                "initialized": 0.0,
                "depth_ready": 0.0,
                "valid_depth_fraction": 0.0,
                "range_collapsed": 1.0,
                "zero_anchor_valid": 0.0,
            })
            planner.feed(frame)
        scene = planner.finish()[0]
        self.assertEqual(scene["absolute_pop_strength"], 1.0)
        self.assertEqual(scene["zero_anchor_shift_px"], 0.0)
        self.assertEqual(scene["evidence"]["valid_edge_sample_count"], 0)
        self.assertEqual(scene["evidence"]["valid_anchor_sample_count"], 0)

    def test_fixed_pop_ignores_scene_edge_risk(self):
        planner = planning.StreamingScenePlanner(config(
            pop_strength=1.25,
            adaptive_pop=False,
            adaptive_pop_max=1.75,
        ))
        for index in range(4):
            planner.feed(sample(index, edge=0.0))
        scene = planner.finish()[0]
        self.assertEqual(scene["absolute_pop_strength"], 1.25)
        self.assertEqual(scene["render"]["pop_origin"], "configured-fixed")

    def test_budget_fails_by_default_with_machine_readable_context(self):
        planner = planning.StreamingScenePlanner(config(
            max_open_cache_bytes=100,
            budget_policy="fail",
        ))
        planner.feed(sample(0), frame_cache_bytes=60)
        with self.assertRaises(planning.SceneCacheBudgetExceeded) as caught:
            planner.feed(sample(1), frame_cache_bytes=60)
        self.assertEqual(caught.exception.limit_bytes, 100)
        self.assertEqual(caught.exception.live_bytes, 120)
        self.assertEqual(caught.exception.open_start_sequence, 1)
        self.assertEqual(caught.exception.current_sequence, 2)

    def test_opt_in_budget_split_is_explicitly_non_semantic(self):
        planner = planning.StreamingScenePlanner(config(
            minimum_scene_frames=1,
            max_open_cache_bytes=100,
            budget_policy="split",
        ))
        self.assertEqual(
            planner.feed(sample(0), frame_cache_bytes=60), [])
        scenes = planner.feed(sample(1), frame_cache_bytes=60)
        self.assertEqual(len(scenes), 1)
        scene = scenes[0]
        self.assertTrue(scene["boundary"]["budget_forced"])
        self.assertFalse(scene["boundary"]["semantic_cut"])
        self.assertEqual(
            scene["boundary"]["decision"], "administrative_cache_split")
        self.assertEqual(scene["semantic_scene_id"], 1)
        tail = planner.finish()[0]
        self.assertEqual(tail["semantic_scene_id"], 1)

    def test_semantic_boundary_moves_past_preserve_previous_frame(self):
        planner = planning.StreamingScenePlanner(config(
            lookbehind_depth_updates=0,
            lookahead_depth_updates=2,
        ))
        scenes = []
        for index in range(5):
            scenes.extend(planner.feed(sample(
                index,
                pulse=index == 2,
                depth=0.90 if index == 2 else 0.80 if index == 3 else 0.05,
                requires_previous=index == 2,
            )))

        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0]["end_sequence_exclusive"], 4)
        self.assertEqual(scenes[0]["boundary"]["final_sequence"], 4)
        self.assertEqual(
            scenes[0]["boundary"]["decision"],
            "moved_to_correlated_evidence",
        )

    def test_admin_split_retains_invalid_tail_behind_safe_start(self):
        planner = planning.StreamingScenePlanner(config(
            minimum_scene_frames=1,
            max_open_cache_bytes=100,
            budget_policy="split",
        ))
        self.assertEqual(
            planner.feed(sample(0), frame_cache_bytes=40), [])
        self.assertEqual(
            planner.feed(sample(1), frame_cache_bytes=40), [])
        scenes = planner.feed(
            sample(2, requires_previous=True), frame_cache_bytes=40)

        self.assertEqual(len(scenes), 1)
        self.assertEqual(
            (scenes[0]["start_sequence"], scenes[0]["end_sequence_exclusive"]),
            (1, 2),
        )
        self.assertEqual(planner.open_start_sequence, 2)
        self.assertEqual(planner.open_cache_bytes, 80)
        tail = planner.finish()[0]
        self.assertEqual(
            (tail["start_sequence"], tail["end_sequence_exclusive"]),
            (2, 4),
        )

    def test_admin_split_fails_without_replay_safe_boundary(self):
        planner = planning.StreamingScenePlanner(config(
            minimum_scene_frames=1,
            max_open_cache_bytes=100,
            budget_policy="split",
        ))
        planner.feed(sample(0), frame_cache_bytes=60)
        with self.assertRaises(planning.SceneCacheBudgetExceeded):
            planner.feed(
                sample(1, requires_previous=True),
                frame_cache_bytes=60,
            )
        self.assertEqual(planner.open_start_sequence, 1)

    def test_clip_cannot_begin_with_preserve_previous_state(self):
        planner = planning.StreamingScenePlanner(config(
            minimum_scene_frames=1))
        planner.feed(sample(0, requires_previous=True))
        with self.assertRaisesRegex(
                planning.ScenePlanError, "cannot begin with preserve-previous"):
            planner.finish()

    def test_admin_split_does_not_create_undersized_safe_prefix(self):
        planner = planning.StreamingScenePlanner(config(
            minimum_scene_frames=2,
            max_open_cache_bytes=100,
            budget_policy="split",
        ))
        planner.feed(sample(0), frame_cache_bytes=40)
        planner.feed(sample(1), frame_cache_bytes=40)
        with self.assertRaises(planning.SceneCacheBudgetExceeded):
            planner.feed(
                sample(2, requires_previous=True),
                frame_cache_bytes=40,
            )

    def test_opt_in_budget_split_fails_while_proposal_is_unresolved(self):
        planner = planning.StreamingScenePlanner(config(
            minimum_scene_frames=1,
            lookahead_depth_updates=5,
            max_open_cache_bytes=100,
            budget_policy="split",
        ))
        planner.feed(
            sample(0, pulse=True, depth=0.70), frame_cache_bytes=60)
        with self.assertRaises(planning.SceneCacheBudgetExceeded):
            planner.feed(sample(1), frame_cache_bytes=60)
        self.assertEqual(planner.pending_proposal_count, 1)
        self.assertEqual(planner.boundary_revisions, ())

    def test_eof_rejects_cut_that_would_leave_undersized_tail(self):
        planner = planning.StreamingScenePlanner(config(
            minimum_scene_frames=2,
            lookahead_depth_updates=5,
        ))
        for index in range(6):
            planner.feed(sample(
                index, pulse=index == 5, depth=0.70 if index == 5 else 0.05))
        scenes = planner.finish()
        self.assertEqual(len(scenes), 1)
        self.assertEqual((scenes[0]["start_sequence"],
                          scenes[0]["end_sequence_exclusive"]), (1, 7))
        self.assertEqual(
            planner.boundary_revisions[0]["decision"],
            "rejected_minimum_scene_length",
        )

    def test_flash_return_must_follow_the_proposal(self):
        planner = planning.StreamingScenePlanner(config())
        for index in range(8):
            flags = (
                planning.ANALYSIS_SAME_SCENE_RETURN if index == 1 else
                (planning.ANALYSIS_APPEARANCE_PROPOSAL |
                 planning.ANALYSIS_EXPOSURE_LIKE |
                 planning.ANALYSIS_APPEARANCE_VETO) if index == 3 else 0
            )
            planner.feed(sample(
                index,
                pulse=index == 3,
                depth=0.70 if index == 3 else 0.05,
                flags=flags,
            ))
        planner.finish()
        self.assertEqual(
            planner.boundary_revisions[0]["decision"],
            "rejected_unsupported_proposal",
        )

    def test_revision_reports_depth_updates_separately_from_source_frames(self):
        planner = planning.StreamingScenePlanner(config(
            lookbehind_depth_updates=2,
            lookahead_depth_updates=1,
        ))
        scenes = []
        for index in range(9):
            depth_updated = index % 2 == 0
            if index == 2:
                frame = sample(index, depth=0.90, depth_updated=True)
            elif index == 4:
                frame = sample(
                    index, pulse=True, depth=0.61, depth_updated=True)
            else:
                frame = sample(
                    index, depth_updated=depth_updated,
                    pulse=False,
                )
            scenes.extend(planner.feed(frame))
        scenes.extend(planner.finish())
        boundary = scenes[0]["boundary"]
        self.assertEqual(boundary["revision_updates"], -1)
        self.assertEqual(boundary["revision_source_frames"], -2)

    def test_committed_boundary_audit_is_deeply_immutable_to_callers(self):
        planner = planning.StreamingScenePlanner(config())
        scenes = []
        for index in range(9):
            scenes.extend(planner.feed(sample(
                index, pulse=index == 3, depth=0.70 if index == 3 else 0.05)))
        original = planner.boundary_revisions
        scenes[0]["boundary"]["proposal_sequences"].append(999)
        mutated_copy = planner.boundary_revisions
        mutated_copy[0]["evidence_window"]["selected"]["score"] = -100.0
        final = planner.boundary_revisions
        self.assertEqual(original, final)
        self.assertNotIn(999, final[0]["proposal_sequences"])

    def test_native_document_contains_one_immutable_scene_and_audit(self):
        planner = planning.StreamingScenePlanner(config())
        for index in range(3):
            planner.feed(sample(index))
        scene = planner.finish()[0]
        document = planning.native_scene_plan_document(scene)
        self.assertEqual(document["schema"], 1)
        self.assertEqual(document["version"], "scene-plan-v1")
        self.assertEqual(document["cache_contract_schema"], 2)
        self.assertEqual(len(document["scenes"]), 1)
        self.assertEqual(
            document["scenes"][0]["absolute_pop_strength"],
            scene["absolute_pop_strength"],
        )
        self.assertEqual(document["audit"]["semantics"]["ground_truth"], False)

    def test_frame_contract_and_configuration_fail_closed(self):
        with self.assertRaises(planning.ScenePlanError):
            config(pop_strength=0.1)
        planner = planning.StreamingScenePlanner(config())
        bad = sample(1)
        with self.assertRaises(planning.ScenePlanError):
            planner.feed(bad)
        self.assertEqual(planner.open_cache_bytes, 0)

    def test_quantile_and_smoothstep_are_deterministic_at_endpoints(self):
        self.assertEqual(planning._quantile([4.0, 1.0, 3.0, 2.0], 0.5), 2.5)
        self.assertEqual(planning._smoothstep(0.04, 0.20, 0.04), 0.0)
        self.assertEqual(planning._smoothstep(0.04, 0.20, 0.20), 1.0)
        self.assertTrue(math.isfinite(
            planning._smoothstep(0.04, 0.20, 0.10)))


if __name__ == "__main__":
    unittest.main()
