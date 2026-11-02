"""Finite policy/mechanics tests; actual shader execution remains a separate check."""
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subtitle_plane_timing import PlaneTiming, advance
from depth_coordinate_v2_contract import SUBTITLE_OCR


class SubtitlePlaneTimingTests(unittest.TestCase):
    width = 2173

    def px(self, state):
        return state.applied * self.width

    def test_captured_7020_7021_twelve_pixel_rise_is_bounded_without_delaying_entry(self):
        # Literal authenticated trace times/levels; not a replay of absent Base pixels.
        entered = advance(PlaneTiming(), goal=24/self.width, now_us=73386636445,
                          source_width=self.width)
        moved = advance(entered, goal=36/self.width, now_us=73386668011,
                        source_width=self.width)
        self.assertAlmostEqual(self.px(entered), 24)
        self.assertAlmostEqual(self.px(moved) - 24, .94698, places=8)
        self.assertEqual(moved.goal * self.width, 36)
        self.assertAlmostEqual((moved.goal-moved.applied)*self.width, 11.05302, places=8)

    def test_direction_is_symmetric_and_each_step_respects_time_not_frame_count(self):
        for gap_us in (1, 8333, 11111, 16667, 33333, 50000, 250000, 2**33):
            up = advance(PlaneTiming(8/self.width, 0, 1, True),
                         goal=40/self.width, now_us=1+gap_us, source_width=self.width)
            down = advance(PlaneTiming(40/self.width, 0, 1, True),
                           goal=8/self.width, now_us=1+gap_us, source_width=self.width)
            allowance = 30 * min(gap_us, 50000) * 1e-6
            self.assertAlmostEqual(self.px(up)-8, 40-self.px(down), places=10)
            self.assertLessEqual(self.px(up)-8, allowance+1e-12)
            self.assertGreater(self.px(up), 8)

    def test_variable_fps_and_repeated_observations_do_not_accumulate_extra_time(self):
        end = []
        for fps in (20, 30, 60, 90, 120):
            state = PlaneTiming(8/self.width, 0, 1, True)
            for i in range(1, fps+1):
                now = 1 + round(i*1_000_000/fps)
                state = advance(state, goal=80/self.width, now_us=now, source_width=self.width)
                repeated = advance(state, goal=0, now_us=now+100000, source_width=self.width,
                                   distinct=False)
                self.assertIs(repeated, state)
            end.append(self.px(state))
        for value in end:
            self.assertAlmostEqual(value, 38, places=9)

    def test_small_goal_error_uses_exponential_without_overshoot(self):
        state = PlaneTiming(24/self.width, 0, 1, True)
        state = advance(state, goal=24.1/self.width, now_us=33334, source_width=self.width)
        expected = 24.1 - .1*math.exp(-.033333/.1)
        self.assertAlmostEqual(self.px(state), expected, places=10)

    def test_empty_clears_in_one_observation_and_reentry_uses_own_goal(self):
        for clock in (0, 1, 1000):
            state = advance(PlaneTiming(36/self.width, 0, 20, True), goal=0,
                            now_us=clock, source_width=self.width, visible=False,
                            valid_probe=False)
            self.assertEqual(state.applied, 0)
            self.assertFalse(state.presented_active)
            reentry = advance(state, goal=8/self.width, now_us=2000, source_width=self.width)
            self.assertAlmostEqual(self.px(reentry), 8)

    def test_abstention_probe_clock_fault_rearm_holds_instead_of_becoming_new_cue(self):
        previous = PlaneTiming(24/self.width, 24/self.width, 1000, True)
        for fault in ({'ready': False}, {'valid_ocr': False}, {'valid_probe': False},
                      {'now_us': 0}, {'now_us': 1000}, {'now_us': 900}):
            args = dict(goal=36/self.width, now_us=2000, source_width=self.width)
            args.update(fault)
            held = advance(previous, **args)
            self.assertEqual(held.applied, previous.applied)
            self.assertTrue(held.presented_active)
            self.assertEqual(held.accepted_us, 0)
            rearmed = advance(held, goal=36/self.width, now_us=10_000_000, source_width=self.width)
            self.assertEqual(rearmed.applied, previous.applied)
            self.assertEqual(rearmed.accepted_us, 10_000_000)
            next_ = advance(rearmed, goal=36/self.width, now_us=10_033_333, source_width=self.width)
            self.assertLessEqual(self.px(next_)-24, .99999+1e-12)

    def test_representation_shrink_clamps_and_domain_reset_reenters_explicitly(self):
        previous = PlaneTiming(.03, .03, 1, True)
        shrunk = advance(previous, goal=.04, now_us=33334, source_width=self.width, limit=.01)
        self.assertEqual(shrunk.applied, .01)  # Explicit validity exception to slew.
        reset = advance(previous, goal=.005, now_us=33334, source_width=self.width, reset=True)
        self.assertEqual(reset.applied, .005)

    def test_random_continuations_remain_between_old_and_current_goal_with_bounded_delta(self):
        rng = random.Random(124)
        for _ in range(2000):
            before, goal = rng.random()*.04, rng.random()*.04
            gap = rng.randint(1, 2_000_000)
            result = advance(PlaneTiming(before, 0, 1, True), goal=goal, now_us=1+gap,
                             source_width=self.width)
            self.assertGreaterEqual(result.applied, min(before, goal))
            self.assertLessEqual(result.applied, max(before, goal))
            self.assertLessEqual(abs(result.applied-before)*self.width,
                                 30*min(gap,50000)*1e-6+1e-12)

    def test_shader_and_generated_contract_use_the_registered_bounds_and_no_floor_clamp(self):
        self.assertEqual((SUBTITLE_OCR.plane_smoothing_time_constant_ms,
                          SUBTITLE_OCR.plane_slew_source_pixels_per_second,
                          SUBTITLE_OCR.plane_max_tick_gap_ms), (100,30,50))
        shader = (Path(__file__).resolve().parents[2]/'src_assets/windows/assets/shaders/directx/host_sbs_subtitle_locator_cs.hlsl').read_text()
        self.assertNotIn('max(applied, current_floor)', shader)
        for token in ('V2_SUBTITLE_PLANE_MAX_TICK_GAP_MS',
                      'V2_SUBTITLE_PLANE_SMOOTHING_TIME_CONSTANT_MS',
                      'V2_SUBTITLE_PLANE_SLEW_SOURCE_PIXELS_PER_SECOND',
                      'flags | ADAPTIVE_CLOCK | ADAPTIVE_PRESENTED'):
            self.assertIn(token, shader)


if __name__ == '__main__':
    unittest.main()
