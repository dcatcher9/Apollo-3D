import copy
import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import current_frame_motion_probe_fixture as probe_fixture  # noqa: E402


class CurrentFrameMotionProbeFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = probe_fixture.load_fixture()
        cls.report = probe_fixture.qualification_report(cls.fixture)
        cls.rows = cls.report["rows"]

    def test_fixture_is_not_standalone_authority_and_matches_all_expected_counters(self):
        self.assertIn("never standalone active hold authority", self.report["authority"])
        self.assertEqual(
            self.report["counters"],
            self.fixture["expected_counters"],
        )
        self.assertEqual(self.report["counters"]["active_authorizations"], 0)
        self.assertFalse(any(row["evidence"]["active_authorized"] for row in self.rows))

    def test_small_object_ladder_vetoes_every_exact_event_in_both_phases(self):
        rows = [row for row in self.rows if row["family"] == "small_object"]
        self.assertEqual(len(rows), 30)
        self.assertEqual({row["width"] for row in rows}, {1, 2, 4, 8, 16})
        self.assertEqual({row["event"] for row in rows}, {"appear", "move", "disappear"})
        for row in rows:
            with self.subTest(
                    width=row["width"], event=row["event"], phase=row["phase_offset"]):
                expected_pixels = row["width"] ** 2 * (2 if row["event"] == "move" else 1)
                evidence = row["evidence"]
                self.assertEqual(evidence["changed_pixel_count"], expected_pixels)
                self.assertGreater(evidence["changed_tile_count"], 0)
                self.assertEqual(evidence["bottom_band_changed_pixel_count"], 0)
                self.assertEqual(evidence["verdict"], "veto_exact_change")
                self.assertFalse(evidence["active_authorized"])

        for width in self.fixture["object_widths"]:
            for event in self.fixture["small_object_events"]:
                event_rows = [
                    row for row in rows
                    if row["width"] == width and row["event"] == event
                ]
                self.assertEqual({row["phase_offset"] for row in event_rows}, {0, 1})
                self.assertEqual(sum(row["would_hold_without_probe"] for row in event_rows), 1)

    def test_object_moves_cannot_hide_behind_zero_global_signed_delta(self):
        moves = [
            row for row in self.rows
            if row["family"] == "small_object" and row["event"] == "move"
        ]
        self.assertEqual(len(moves), 10)
        for row in moves:
            with self.subTest(width=row["width"], phase=row["phase_offset"]):
                evidence = row["evidence"]
                self.assertEqual(evidence["signed_channel_delta_sum"], [0, 0, 0])
                self.assertGreater(evidence["absolute_channel_delta_sum"][0], 0)
                self.assertEqual(evidence["verdict"], "veto_exact_change")

    def test_aba_endpoints_cover_both_cadence_phases_without_endpoint_blindness(self):
        aba_rows = [row for row in self.rows if row["family"] == "aba"]
        scenario_names = {scenario["name"] for scenario in self.fixture["aba_scenarios"]}
        self.assertEqual({row["scenario"] for row in aba_rows}, scenario_names)
        for scenario in scenario_names:
            scenario_rows = [row for row in aba_rows if row["scenario"] == scenario]
            self.assertEqual(len(scenario_rows), 8)
            for phase in self.fixture["cadence"]["phase_offsets"]:
                phase_rows = sorted(
                    (row for row in scenario_rows if row["phase_offset"] == phase),
                    key=lambda row: row["transition_index"],
                )
                self.assertEqual(
                    [row["event"] for row in phase_rows],
                    ["A-to-A", "A-to-B", "B-to-A", "A-to-A"],
                )
                self.assertEqual(
                    [row["evidence"]["verdict"] for row in phase_rows],
                    [
                        "quiet_evidence_only",
                        "veto_exact_change",
                        "veto_exact_change",
                        "quiet_evidence_only",
                    ],
                )
            for transition in (1, 2):
                changed_endpoint_rows = [
                    row for row in scenario_rows if row["transition_index"] == transition
                ]
                self.assertEqual(
                    sum(row["would_hold_without_probe"] for row in changed_endpoint_rows),
                    1,
                )
                self.assertTrue(all(
                    row["evidence"]["verdict"] == "veto_exact_change"
                    for row in changed_endpoint_rows
                ))

    def test_same_tile_equal_sum_rearrangement_is_still_an_exact_veto(self):
        rows = [
            row for row in self.rows
            if row["scenario_kind"] == "equal_sum_swap"
            and not row["evidence"]["pixel_exact_equal"]
        ]
        self.assertEqual(len(rows), 4)
        for row in rows:
            with self.subTest(event=row["event"], phase=row["phase_offset"]):
                evidence = row["evidence"]
                self.assertEqual(evidence["changed_pixel_count"], 8)
                self.assertEqual(evidence["changed_tile_count"], 1)
                self.assertEqual(evidence["signed_channel_delta_sum"], [0, 0, 0])
                self.assertTrue(all(value > 0 for value in evidence["absolute_channel_delta_sum"]))
                self.assertEqual(evidence["verdict"], "veto_exact_change")

    def test_subtitle_onset_shift_removal_and_cut_veto_bottom_band(self):
        kinds = {"subtitle_pulse", "subtitle_shift", "full_frame_cut"}
        rows = [
            row for row in self.rows
            if row["scenario_kind"] in kinds and not row["evidence"]["pixel_exact_equal"]
        ]
        self.assertEqual(len(rows), 12)
        self.assertEqual({row["scenario_kind"] for row in rows}, kinds)
        for row in rows:
            with self.subTest(
                    kind=row["scenario_kind"], event=row["event"], phase=row["phase_offset"]):
                evidence = row["evidence"]
                self.assertTrue(evidence["bottom_band_changed"])
                self.assertGreater(evidence["bottom_band_changed_pixel_count"], 0)
                self.assertEqual(evidence["verdict"], "veto_exact_change")
        cut_rows = [row for row in rows if row["scenario_kind"] == "full_frame_cut"]
        field_area = (
            self.fixture["model_field"]["width"] * self.fixture["model_field"]["height"]
        )
        self.assertTrue(all(
            row["evidence"]["changed_pixel_count"] == field_area for row in cut_rows
        ))

    def test_exact_equality_is_evidence_but_never_authority(self):
        equal_rows = [row for row in self.rows if row["evidence"]["pixel_exact_equal"]]
        self.assertEqual(len(equal_rows), 20)
        for row in equal_rows:
            evidence = row["evidence"]
            self.assertEqual(evidence["changed_pixel_count"], 0)
            self.assertEqual(evidence["changed_tile_count"], 0)
            self.assertEqual(evidence["verdict"], "quiet_evidence_only")
            self.assertFalse(evidence["active_authorized"])

    def test_evidence_rejects_wrong_shape_and_fixture_rejects_authority_drift(self):
        frame = np.zeros((434, 770, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "exact uint8 770x434 RGB"):
            probe_fixture.exact_evidence(self.fixture, frame[:, :-1], frame[:, :-1])

        changed = copy.deepcopy(self.fixture)
        changed["classification"]["quiet_authorizes_hold"] = True
        with self.assertRaisesRegex(ValueError, "may not authorize quiet evidence"):
            probe_fixture.validate_fixture(changed)

        changed = copy.deepcopy(self.fixture)
        changed["bottom_ocr_band"][1] += 1
        with self.assertRaisesRegex(ValueError, "wrong bottom OCR band"):
            probe_fixture.validate_fixture(changed)


if __name__ == "__main__":
    unittest.main()
