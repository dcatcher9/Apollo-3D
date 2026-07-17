#!/usr/bin/env python3

import unittest

from artistic_policy_ordinal_controller import (
    CausalOrdinalController,
    mix_style_scale,
    replay_desired_scales,
)


class ArtisticPolicyOrdinalControllerTests(unittest.TestCase):
    def test_lowers_on_the_same_completed_depth_update(self):
        controller = CausalOrdinalController(
            10, safe_hold_seconds=0.2, upward_cooldown_seconds=0.3
        )
        controller.update(1.20)
        raised = controller.update(1.20)
        self.assertEqual(raised.applied_safe_cap, 1.20)
        lowered = controller.update(1.04)
        self.assertEqual(lowered.applied_safe_cap, 1.04)
        self.assertEqual(lowered.change_kind, "safety_lower")

    def test_raises_only_after_sustained_minimum_safe_cap(self):
        controller = CausalOrdinalController(
            10, safe_hold_seconds=0.3, upward_cooldown_seconds=0.2
        )
        self.assertEqual(controller.update(1.30).applied_safe_cap, 1.0)
        self.assertEqual(controller.update(1.20).applied_safe_cap, 1.0)
        raised = controller.update(1.26)
        self.assertEqual(raised.applied_safe_cap, 1.20)
        self.assertEqual(raised.change_kind, "safe_raise")

    def test_upward_cooldown_prevents_pumping(self):
        controller = CausalOrdinalController(
            10, safe_hold_seconds=0.1, upward_cooldown_seconds=0.3
        )
        self.assertEqual(controller.update(1.10).applied_safe_cap, 1.10)
        for _ in range(3):
            self.assertEqual(controller.update(1.30).applied_safe_cap, 1.10)
        self.assertEqual(controller.update(1.30).applied_safe_cap, 1.30)

    def test_hard_cut_resets_without_using_future_frames(self):
        trace = replay_desired_scales(
            [1.20, 1.20, 1.40, 1.40],
            depth_fps=10,
            hard_cut_indices={2},
            safe_hold_seconds=0.2,
            upward_cooldown_seconds=0.1,
        )
        self.assertEqual(trace[1].applied_safe_cap, 1.20)
        self.assertEqual(trace[2].applied_safe_cap, 1.0)
        self.assertEqual(trace[2].change_kind, "cut_reset")
        self.assertEqual(trace[3].applied_safe_cap, 1.0)

    def test_time_constants_are_depth_fps_invariant(self):
        def first_raise(fps):
            trace = replay_desired_scales(
                [1.20] * fps,
                depth_fps=fps,
                safe_hold_seconds=0.5,
                upward_cooldown_seconds=2.0,
            )
            return next(
                index + 1 for index, row in enumerate(trace) if row.changed
            )

        self.assertAlmostEqual(first_raise(30) / 30.0,
                               first_raise(60) / 60.0, places=6)

    def test_style_is_applied_after_safety_selection(self):
        self.assertEqual(mix_style_scale(1.40, "clean"), 1.0)
        self.assertEqual(mix_style_scale(1.40, "immersive"), 1.40)
        self.assertAlmostEqual(mix_style_scale(1.40, "balanced"), 1.20)
        with self.assertRaises(RuntimeError):
            mix_style_scale(1.40, "unknown")


if __name__ == "__main__":
    unittest.main()
