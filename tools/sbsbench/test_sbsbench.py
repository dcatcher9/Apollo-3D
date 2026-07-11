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
        hard_min = {"role": "hard", "better": "higher", "hard_min": 90.0,
                    "rel_tol": 0.0, "abs_floor": 1.0}
        self.assertFalse(sbsbench.metric_gate_failed(95.0, 91.0, hard_min))
        self.assertTrue(sbsbench.metric_gate_failed(95.0, 89.0, hard_min))

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

    def test_static_region_jitter_ignores_source_motion_but_detects_static_warp_change(self):
        rng = np.random.default_rng(9)
        src = np.round(rng.random((64, 96), dtype=np.float32) * 255.0) / 255.0
        stable, support = sbsbench.static_region_jitter(src, src, src, src, src, src,
                                                        min_support=0.5)
        self.assertAlmostEqual(stable, 0.0)
        self.assertEqual(support, 1.0)
        changed = src.copy()
        changed[16:48, 30:66] = np.clip(changed[16:48, 30:66] + 0.2, 0, 1)
        jitter, _ = sbsbench.static_region_jitter(changed, changed, src, src, src, src,
                                                  min_support=0.5)
        self.assertGreater(jitter, 20.0)
        moving_src = np.roll(src, 8, axis=1)
        skipped, moving_support = sbsbench.static_region_jitter(
            moving_src, moving_src, src, src, moving_src, src, min_support=0.5)
        self.assertIsNone(skipped)
        self.assertLess(moving_support, 0.5)

    def test_comfort_disparity_reports_both_signed_tails(self):
        dx = np.array([-12.0, -8.0, 0.0, 6.0, 10.0])
        weights = np.ones_like(dx)
        positive, negative = sbsbench.comfort_disparity(dx, weights, eye_width=400, tail=0.8)
        self.assertAlmostEqual(positive, 1.5)
        self.assertAlmostEqual(negative, 3.0)

    def test_hard_integrity_aggregates_worst_frame_not_mean(self):
        agg = sbsbench.aggregate([
            {"source_coverage_pct": 100.0, "positive_disparity_pct": 0.5},
            {"source_coverage_pct": 70.0, "positive_disparity_pct": 4.0},
        ])
        self.assertEqual(agg["source_coverage_pct"], 70.0)
        self.assertEqual(agg["positive_disparity_pct"], 4.0)

    def test_source_coverage_and_integrity_detect_missing_content(self):
        rng = np.random.default_rng(22)
        src = np.round(rng.random((96, 160), dtype=np.float32) * 255.0) / 255.0
        clean = sbsbench._shift_x_edge(src, 5)
        good = sbsbench.source_relative_metrics(clean, src, max_shift=8)
        damaged = clean.copy()
        damaged[20:76, 60:110] = 0.0
        bad = sbsbench.source_relative_metrics(damaged, src, max_shift=8)
        self.assertGreater(good["source_coverage_pct"], 99.0)
        self.assertGreater(good["image_integrity_pct"], 99.0)
        self.assertLess(bad["source_coverage_pct"], good["source_coverage_pct"] - 15.0)
        self.assertLess(bad["image_integrity_pct"], good["image_integrity_pct"] - 10.0)

    def test_source_relative_halo_and_stretch_subtract_real_source_structure(self):
        y, x = np.mgrid[:96, :160]
        src = (0.35 + 0.2 * np.sin(x * 0.55) + 0.15 * np.sin(y * 0.3)).astype(np.float32)
        depth = np.full((24, 40), 0.2, np.float32)
        depth[:, 20:] = 0.8
        clean = sbsbench.source_relative_metrics(src, src, depth, max_shift=4)
        halo_eye = src.copy()
        halo_eye[:, 79:82] = 1.0
        halo = sbsbench.source_relative_metrics(halo_eye, src, depth, max_shift=4)
        stretch_eye = src.copy()
        stretch_eye[:, 82:115] = stretch_eye[:, 82:83]
        stretch = sbsbench.source_relative_metrics(stretch_eye, src, depth, max_shift=4)
        self.assertGreater(halo["source_halo_p95"], clean["source_halo_p95"] + 3.0)
        self.assertGreater(stretch["source_stretch_pct"], clean["source_stretch_pct"] + 10.0)

    def test_ground_truth_depth_metrics_reward_aligned_structure(self):
        gt = np.full((96, 160), 0.25, np.float32)
        gt[:, 80:] = 0.75
        equivalent = gt * 0.8 + 0.1  # monocular scale/shift ambiguity is intentionally free
        flat = np.full_like(gt, 0.5)
        good = sbsbench.depth_ground_truth_metrics(equivalent, gt)
        bad = sbsbench.depth_ground_truth_metrics(flat, gt)
        self.assertLess(good["depth_gt_si_rmse"], 0.01)
        self.assertGreater(good["depth_gt_edge_f1"], 99.0)
        self.assertGreater(bad["depth_gt_si_rmse"], 40.0)
        self.assertLess(bad["depth_gt_edge_f1"], 1.0)

    def test_optical_flow_temporal_metric_compensates_motion(self):
        rng = np.random.default_rng(31)
        previous = np.round(rng.random((96, 160), dtype=np.float32) * 255.0) / 255.0
        current = np.roll(previous, 5, axis=1)
        stable, _, support = sbsbench.flow_temporal_metrics(
            current, current, previous, previous, current, previous, min_support=0.1)
        corrupted = current.copy()
        corrupted[20:75, 60:110] = 0.0
        unstable, _, _ = sbsbench.flow_temporal_metrics(
            corrupted, corrupted, previous, previous, current, previous, min_support=0.1)
        self.assertGreater(support, 0.8)
        self.assertLess(stable, 2.0)
        self.assertGreater(unstable, stable + 100.0)


if __name__ == "__main__":
    unittest.main()
