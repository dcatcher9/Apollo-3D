"""CPU checks for evidence eligibility and known-background diagnostics."""
import unittest
from pathlib import Path
import tempfile

import numpy as np

from compare_stereo_parity import (Ineligible, compare, edge_diagnostic, final_aa_cases,
                                   landmarks, match_landmarks, numeric_runtime)


class MatchedEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.reference = np.array([[8., -8.], [-4., 4.], [2., -2.]])

    def test_weaker_stereo_cannot_be_an_eligible_quality_win(self):
        with self.assertRaisesRegex(Ineligible, "Unmatched endpoint"):
            match_landmarks(self.reference, self.reference * .5)

    def test_same_binocular_strength_with_common_image_shift_is_rejected(self):
        with self.assertRaisesRegex(Ineligible, "Unmatched endpoint"):
            match_landmarks(self.reference, self.reference + 1)

    def test_holdout_transfer_difference_is_reported_without_fitting_it_away(self):
        candidate = self.reference.copy()
        candidate[2] = [4, -4]
        result = match_landmarks(self.reference, candidate)
        self.assertEqual(result["holdout_binocular_delta_px"], 4)

    def test_zero_relief_is_not_a_quality_reference(self):
        flat = np.zeros((3, 2))
        with self.assertRaisesRegex(Ineligible, "too weak"):
            match_landmarks(flat, flat)

    def test_numeric_runtime_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = (Path(directory) / name for name in ("a.txt", "b.txt"))
            control.write_text("older numeric runtime\n")
            treatment.write_text("different numeric runtime\n")
            with self.assertRaisesRegex(Ineligible, "fingerprints differ"):
                numeric_runtime(control, treatment)

    def test_landmarks_are_measured_from_pixels(self):
        h, w = 144, 320
        pixels = np.full((h, 2 * w, 4), .04, dtype=np.float32)
        top = h * 3 // 4
        for plane in range(3):
            y = top + (2 * plane + 1) * (h - top) // 6
            for eye in range(2):
                x = eye * w + w // 2 + int(self.reference[plane, eye])
                pixels[y, x - 8:x + 8, :3] = .8
        np.testing.assert_allclose(landmarks(pixels), self.reference, atol=1e-10)
        pixels[:] = 0
        with self.assertRaisesRegex(Ineligible, "landmark"):
            landmarks(pixels)


class KnownBackgroundTest(unittest.TestCase):
    def test_added_halo_is_measured_and_foreground_retention_is_separate(self):
        h, w = 144, 320
        source = np.zeros((h, w, 4), dtype=np.float32)
        source[:, :w // 2, 0] = 1
        source[:, w // 2:, 2] = 1
        depth = np.full((h, w), .005, dtype=np.float32)
        depth[:, :w // 2] = .02
        clean = np.concatenate([source, source], axis=1)
        shifts = np.zeros((3, 2))
        result = edge_diagnostic(clean, source, depth, shifts)
        self.assertEqual(result["known_background_chroma_leakage_mean"], 0)
        self.assertEqual(result["foreground_retained_chroma_mean"], 1)
        halo = clean.copy()
        for eye in range(2):
            halo[:, eye * w + w // 2 + 4:eye * w + w // 2 + 12, 0] = .5
            halo[:, eye * w + w // 2 + 4:eye * w + w // 2 + 12, 2] = .5
        polluted = edge_diagnostic(halo, source, depth, shifts)
        self.assertGreater(polluted["known_background_chroma_leakage_mean"], .1)
        self.assertEqual(polluted["foreground_retained_chroma_mean"], 1)
        self.assertGreater(polluted["unexcluded_background_bands_descriptive"]["4_to_8_px"]["mean"], 0)

    def test_narrow_halo_is_visible_in_unexcluded_inner_band(self):
        h, w = 144, 320
        source = np.zeros((h, w, 4), dtype=np.float32)
        source[:, :w // 2, 0] = 1
        source[:, w // 2:, 2] = 1
        depth = np.full((h, w), .005, dtype=np.float32)
        depth[:, :w // 2] = .02
        output = np.concatenate([source, source], axis=1)
        for eye in range(2):
            output[:, eye * w + w // 2:eye * w + w // 2 + 2, 0] = .5
        metric = edge_diagnostic(output, source, depth, np.zeros((3, 2)))
        self.assertEqual(metric["known_background_chroma_leakage_mean"], 0)
        self.assertGreater(metric["unexcluded_background_bands_descriptive"]["0_to_2_px"]["mean"], 0)


class FinalAAEvidenceTest(unittest.TestCase):
    contract = b"schema=final-aa-parity-1\nfinal_aa=off\ncase_branches=1\n"

    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.control, self.treatment = [Path(self.workspace.name) / side for side in ("control", "treatment")]
        self.control.mkdir()
        self.treatment.mkdir()

    def write_contract(self, directory, contents=None):
        (directory / "final-aa-contract.txt").write_bytes(self.contract if contents is None else contents)

    def test_no_contract_keeps_both_legacy_branches(self):
        self.assertEqual(final_aa_cases(self.control, self.treatment), (0, 1))

    def test_exact_paired_contract_selects_aa0(self):
        for directory in (self.control, self.treatment):
            self.write_contract(directory)
        self.assertEqual(final_aa_cases(self.control, self.treatment), (0,))

    def test_one_sided_contract_is_rejected_in_either_direction(self):
        for directory in (self.control, self.treatment):
            with self.subTest(side=directory.name):
                self.write_contract(directory)
                with self.assertRaisesRegex(Ineligible, "only one side"):
                    final_aa_cases(self.control, self.treatment)
                (directory / "final-aa-contract.txt").unlink()

    def test_different_contracts_are_rejected(self):
        self.write_contract(self.control)
        self.write_contract(self.treatment, self.contract.replace(b"off", b"on"))
        with self.assertRaisesRegex(Ineligible, "contracts differ"):
            final_aa_cases(self.control, self.treatment)

    def test_matching_but_malformed_contracts_are_rejected(self):
        malformed = [b"", self.contract[:-1], self.contract.replace(b"\n", b"\r\n"),
                     self.contract.replace(b"case_branches=1", b"case_branches=2"),
                     self.contract.replace(b"parity-1", b"parity-2"), self.contract + b"extra=1\n"]
        for contents in malformed:
            with self.subTest(contents=contents):
                for directory in (self.control, self.treatment):
                    self.write_contract(directory, contents)
                with self.assertRaisesRegex(Ineligible, "malformed"):
                    final_aa_cases(self.control, self.treatment)

    def write_evidence(self, aa_cases, motion=False):
        h, w = 144, 320
        source = np.zeros((h, w, 4), dtype="<f2")
        source[:, :w // 2, 0] = 1
        source[:, w // 2:, 2] = 1
        depth = np.full((h, w), .005, dtype="<f4")
        depth[:, :w // 2] = .02
        pixels = np.full((h, 2 * w, 4), .04, dtype="<f4")
        for plane, positions in enumerate(((8, -8), (-4, 4), (2, -2))):
            y = h * 3 // 4 + (2 * plane + 1) * (h - h * 3 // 4) // 6
            for eye, shift in enumerate(positions):
                x = eye * w + w // 2 + shift
                pixels[y, x - 8:x + 8, :3] = .8
        names = [f"motion-step-aa{aa}-f{frame}" for aa in aa_cases for frame in range(7)] if motion else []
        captures = [f"{scene}-aa{aa}" for scene in ("step", "detail", "fringe") for aa in aa_cases] + names
        for directory in (self.control, self.treatment):
            (directory / "measurements.txt").write_text(
                f"renderer=fixture variant=known width={w} height={h} source_color=2\n"
                "dump=float32_little_endian_RGBA rgb=linear_Rec709\n")
            (directory / "provenance.txt").write_text("same fixture executable\nsame official runtime\n")
            for scene in ("calibration", "step", "detail", "fringe", *names):
                source.tofile(directory / f"{scene}.source.bin")
                depth.tofile(directory / f"{scene}.depth.f32")
            for capture in captures:
                pixels.tofile(directory / f"{capture}.rgba.f32")

    def test_aa0_contract_reads_only_three_static_captures(self):
        for directory in (self.control, self.treatment):
            self.write_contract(directory)
        self.write_evidence((0,))
        result = compare(self.control, self.treatment)
        self.assertEqual(result["final_aa_case_branches"], [0])
        self.assertEqual(len(result["cases"]), 3)
        self.assertTrue(all(case["identical_full_rgba"] for case in result["cases"]))

    def test_legacy_evidence_still_requires_aa1(self):
        self.write_evidence((0,))
        with self.assertRaises(FileNotFoundError):
            compare(self.control, self.treatment)

    def test_legacy_evidence_reads_six_static_captures(self):
        self.write_evidence((0, 1))
        result = compare(self.control, self.treatment)
        self.assertEqual(result["final_aa_case_branches"], [0, 1])
        self.assertEqual(len(result["cases"]), 6)

    def test_aa0_motion_requires_exactly_seven_frames_and_matching_inputs(self):
        for directory in (self.control, self.treatment):
            self.write_contract(directory)
        self.write_evidence((0,), motion=True)
        result = compare(self.control, self.treatment)
        self.assertEqual(len(result["cases"]), 10)
        self.assertEqual(result["moving_sequence_frames_per_aa"], 7)
        path = self.treatment / "motion-step-aa0-f6.source.bin"
        path.write_bytes(b"changed input")
        with self.assertRaisesRegex(Ineligible, "Source/depth bytes differ"):
            compare(self.control, self.treatment)
        (self.treatment / "motion-step-aa0-f6.rgba.f32").unlink()
        with self.assertRaisesRegex(Ineligible, "Moving sequence"):
            compare(self.control, self.treatment)

    def test_aa0_still_rejects_provenance_and_endpoint_mismatch(self):
        for directory in (self.control, self.treatment):
            self.write_contract(directory)
        self.write_evidence((0,))
        provenance = self.treatment / "provenance.txt"
        original = provenance.read_text()
        provenance.write_text("different fixture\nsame official runtime\n")
        with self.assertRaisesRegex(Ineligible, "provenance differs"):
            compare(self.control, self.treatment)
        provenance.write_text(original)
        path = self.treatment / "step-aa0.rgba.f32"
        pixels = np.fromfile(path, dtype="<f4").reshape(144, 640, 4)
        np.roll(pixels, 1, axis=1).tofile(path)
        with self.assertRaisesRegex(Ineligible, "Unmatched endpoint"):
            compare(self.control, self.treatment)


if __name__ == "__main__":
    unittest.main()
