import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sbsbench
import run_eval


class EvalContractTests(unittest.TestCase):
    def test_warp_override_uses_last_explicit_value(self):
        self.assertEqual(run_eval.extra_value(
            ["--warp", "apollo", "--warp", "vd3d"], "--warp", "apollo"), "vd3d")

    def test_warp_is_read_from_config(self):
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            fh.write("# sbs_3d_warp = apollo\nsbs_3d_warp = vd3d # active\n")
            path = fh.name
        try:
            self.assertEqual(run_eval.conf_value(path, "sbs_3d_warp", "apollo"), "vd3d")
        finally:
            os.unlink(path)

    def test_shift_profile_override_uses_last_explicit_value(self):
        self.assertEqual(run_eval.extra_value(
            ["--shift-profile", "apollo", "--shift-profile", "bestv2"],
            "--shift-profile", "apollo"), "bestv2")

    def test_shift_profile_is_read_from_config(self):
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            fh.write("sbs_3d_shift_profile = bestv2 # active\n")
            path = fh.name
        try:
            self.assertEqual(
                run_eval.conf_value(path, "sbs_3d_shift_profile", "apollo"), "bestv2")
        finally:
            os.unlink(path)

    def test_phase_shift_recovers_known_translation(self):
        rng = np.random.default_rng(1234)
        a = rng.random((64, 64))
        b = np.roll(a, shift=(2, -5), axis=(0, 1))
        dy, dx = sbsbench.phase_shift(a, b)
        self.assertAlmostEqual(dy, -2.0, places=5)
        self.assertAlmostEqual(dx, 5.0, places=5)

    def test_sequence_joins_by_frame_identity(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            sbs = np.zeros((16, 32, 3), dtype=np.uint8)
            src = np.zeros((16, 16, 3), dtype=np.uint8)
            Image.fromarray(sbs).save(os.path.join(seq, "sbs_00007.png"))
            Image.fromarray(src).save(os.path.join(frames, "frame_00007.png"))
            rows, agg = sbsbench.measure_sequence(seq, frames)
            self.assertEqual(rows[0]["_frame_id"], 7)
            self.assertEqual(agg["_n"], 1)

    def test_sequence_rejects_positional_mispairing(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            blank = np.zeros((16, 32, 3), dtype=np.uint8)
            Image.fromarray(blank).save(os.path.join(seq, "sbs_00008.png"))
            Image.fromarray(blank[:, :16]).save(os.path.join(frames, "frame_00007.png"))
            with self.assertRaisesRegex(ValueError, "frame-id mismatch"):
                sbsbench.measure_sequence(seq, frames)

    def test_duplicate_numeric_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(os.path.join(root, "frame_1.png"))
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(os.path.join(root, "frame_01.jpg"))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                sbsbench.indexed_files(os.path.join(root, "frame_*.*"), "frame_")

    def test_disocclusion_ratio_requires_minimum_support(self):
        eye = np.zeros((64, 64), dtype=np.float32)
        depth = np.full((16, 16), 0.5, dtype=np.float32)
        frac, smear = sbsbench.disocclusion_metrics(eye, depth)
        self.assertLess(frac, sbsbench.MIN_DISOCC_FRAC)
        self.assertIsNone(smear)

    def test_depth_is_diagnostic_not_part_of_artifact_score(self):
        clean = {"pop_spread_pct": 0.0}
        false_stereo = {"pop_spread_pct": 0.2}
        self.assertGreater(
            sbsbench.sbs_score(clean, expected_flat=True)["q_depth"],
            sbsbench.sbs_score(false_stereo, expected_flat=True)["q_depth"])
        self.assertLess(
            sbsbench.sbs_score(clean)["q_depth"],
            sbsbench.sbs_score(false_stereo)["q_depth"])
        self.assertEqual(sbsbench.sbs_score(clean)["score"],
                         sbsbench.sbs_score(false_stereo)["score"])

    def test_metric_delta_class_uses_gate_tolerance_and_direction(self):
        lower = {"better": "lower", "rel_tol": 0.25, "abs_floor": 0.5}
        self.assertEqual(sbsbench.metric_delta_class(2.0, 2.4, lower), "noise")
        self.assertEqual(sbsbench.metric_delta_class(2.0, 2.6, lower), "regressed")
        self.assertEqual(sbsbench.metric_delta_class(2.0, 1.4, lower), "improved")

    def test_metric_roles_control_committed_gate(self):
        diagnostic = {"role": "diagnostic", "better": "lower",
                      "rel_tol": 0.0, "abs_floor": 0.1}
        hard = {"role": "hard", "better": "lower", "hard_max": 0.5,
                "rel_tol": 0.0, "abs_floor": 0.1}
        self.assertFalse(sbsbench.metric_gate_failed(0.0, 99.0, diagnostic))
        self.assertFalse(sbsbench.metric_gate_failed(0.0, 0.49, hard))
        self.assertTrue(sbsbench.metric_gate_failed(0.0, 0.51, hard))

    def test_ab_decision_preserves_primary_axis_tradeoff(self):
        specs = {
            "pop": {"role": "primary", "axis": "stereo", "better": "higher",
                    "rel_tol": 0.0, "abs_floor": 0.5},
            "halo": {"role": "primary", "axis": "warp", "better": "lower",
                     "rel_tol": 0.0, "abs_floor": 0.5},
            "legacy_proxy": {"role": "diagnostic", "axis": "warp", "better": "lower",
                             "rel_tol": 0.0, "abs_floor": 0.1},
        }
        result = sbsbench.evaluate_ab_decision(
            {"clip": {"pop": 4.0, "halo": 2.0, "legacy_proxy": 0.0}},
            {"clip": {"pop": 5.0, "halo": 3.0, "legacy_proxy": 99.0}},
            ["clip"], specs)
        self.assertEqual(result["verdict"], "tradeoff")
        self.assertEqual(result["improved"], 1)
        self.assertEqual(result["regressed"], 1)

    def test_ab_decision_hard_constraint_cannot_be_traded(self):
        specs = {
            "vmis": {"role": "hard", "axis": "comfort", "hard_max": 0.5,
                     "better": "lower", "rel_tol": 0.0, "abs_floor": 0.1},
            "pop": {"role": "primary", "axis": "stereo", "better": "higher",
                    "rel_tol": 0.0, "abs_floor": 0.5},
        }
        result = sbsbench.evaluate_ab_decision(
            {"clip": {"vmis": 0.1, "pop": 4.0}},
            {"clip": {"vmis": 0.6, "pop": 8.0}}, ["clip"], specs)
        self.assertEqual(result["verdict"], "reject_hard")

    def test_source_residual_accepts_horizontal_parallax_and_detects_corruption(self):
        rng = np.random.default_rng(42)
        src = np.round(rng.random((96, 160), dtype=np.float32) * 255.0) / 255.0
        shifted = sbsbench._shift_x_edge(src, 5)
        clean = sbsbench.source_match_residual(shifted, src, max_shift=8)
        corrupted = shifted.copy()
        corrupted[24:72, 60:100] = 0.0
        damaged = sbsbench.source_match_residual(corrupted, src, max_shift=8)
        self.assertLess(clean[1], 0.01)
        self.assertGreater(damaged[1], clean[1] + 5.0)


if __name__ == "__main__":
    unittest.main()
