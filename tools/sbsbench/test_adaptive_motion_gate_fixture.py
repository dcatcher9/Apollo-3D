import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adaptive_motion_gate_fixture as fixture_oracle  # noqa: E402


class AdaptiveMotionGateFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixture_oracle.load_fixture()

    def test_localized_object_ladder_stays_below_tiny_motion_bounds(self):
        rows = fixture_oracle.localized_change_fractions(self.fixture)
        self.assertEqual(len(rows), 15)
        self.assertEqual({row["width"] for row in rows}, {1, 2, 4, 8, 16})
        self.assertEqual(
            {row["event"] for row in rows}, {"appear", "move", "disappear"})
        # The largest case is a disjoint 16x16 move: 512 / (770*434) = 0.1532%.
        # It is deliberately inside both the legacy 0.25% gate and every delayed quiet fraction.
        self.assertLess(max(row["changed_fraction"] for row in rows), 1.0 / 400.0)
        self.assertLess(max(row["changed_fraction"] for row in rows), 0.005)

    def test_localized_damage_always_uses_current_depth(self):
        expected = self.fixture["depth"]["expected_current_lag_f1"]
        for phase in self.fixture["cadence"]["phase_offsets"]:
            with self.subTest(phase=phase):
                rows = fixture_oracle.depth_lag_report(
                    self.fixture, "localized", phase)
                self.assertEqual(len(rows), 15)
                self.assertFalse(any(row["held"] for row in rows))
                self.assertFalse(any(row["ocr_only_needed"] for row in rows))
                for row in rows:
                    self.assertAlmostEqual(row["depth_gt_lag_f1"], expected, places=6)

    def test_broad_depth_hold_is_visible_at_both_cadence_phases(self):
        reports = {
            phase: fixture_oracle.depth_lag_report(
                self.fixture, "broad_single_rect", phase)
            for phase in self.fixture["cadence"]["phase_offsets"]
        }
        self.assertEqual(sum(row["held"] for row in reports[0]), 8)
        self.assertEqual(sum(row["held"] for row in reports[1]), 7)
        held_lag = self.fixture["depth"]["expected_held_lag_f1"]
        current_lag = self.fixture["depth"]["expected_current_lag_f1"]
        for index, (phase_zero, phase_one) in enumerate(zip(reports[0], reports[1])):
            with self.subTest(index=index, width=phase_zero["width"], event=phase_zero["event"]):
                self.assertNotEqual(phase_zero["held"], phase_one["held"])
                for row in (phase_zero, phase_one):
                    expected = held_lag if row["held"] else current_lag
                    self.assertAlmostEqual(row["depth_gt_lag_f1"], expected, places=6)
                    self.assertEqual(row["ocr_only_needed"], row["held"])

        for rows in reports.values():
            for previous, current in zip(rows, rows[1:]):
                self.assertFalse(previous["held"] and current["held"])

    def test_subtitle_shift_and_cut_probes_cover_both_phases(self):
        probe = self.fixture["ocr_probe"]
        before = probe["subtitle_before"]
        after = probe["subtitle_after"]
        self.assertNotEqual(before, after)
        self.assertEqual(before[2] - before[0], after[2] - after[0])
        self.assertEqual(before[3] - before[1], after[3] - after[1])
        self.assertTrue(fixture_oracle.rects_intersect(probe["crop"], before))
        self.assertTrue(fixture_oracle.rects_intersect(probe["crop"], after))

        for phase, cut_index in zip(
                self.fixture["cadence"]["phase_offsets"],
                self.fixture["cadence"]["hard_cut_event_indices"]):
            rows = fixture_oracle.depth_lag_report(
                self.fixture, "broad_single_rect", phase)
            self.assertTrue(rows[cut_index]["held"])


if __name__ == "__main__":
    unittest.main()
