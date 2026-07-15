#!/usr/bin/env python3
"""Synthetic correctness tests for the deterministic stereo label fitter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import artistic_stereo_label_fitter  # noqa: E402


class ArtisticStereoLabelFitterTests(unittest.TestCase):
    def test_recovers_positive_subpixel_disparity(self):
        rng = np.random.default_rng(19)
        height, width = 160, 384
        left = rng.integers(0, 256, (height, width), dtype=np.uint8)
        left = cv2.GaussianBlur(left, (0, 0), 0.7)
        disparity = 11
        right = np.empty_like(left)
        right[:, :-disparity] = left[:, disparity:]
        right[:, -disparity:] = left[:, -1:]
        config = artistic_stereo_label_fitter.LabelFitterConfig(
            analysis_width=width, search_fraction=0.125,
            min_support=0.02, min_fit_pixels=128,
        )
        recovered, confidence, valid, diagnostics = (
            artistic_stereo_label_fitter.estimate_disparity(left, right, config)
        )
        interior = valid[:, 64:-64]
        self.assertGreater(float(interior.mean()), 0.50)
        self.assertAlmostEqual(
            float(np.median(recovered[:, 64:-64][interior])), disparity,
            delta=0.35,
        )
        self.assertGreater(float(confidence[valid].mean()), 0.10)
        self.assertGreater(diagnostics["lr_consistency_pct"], 50.0)

        depth = np.tile(
            np.linspace(0.0, 1.0, width, dtype=np.float32), (height, 1)
        )
        reference = np.full((height, width), 12.0, np.float32)
        analysis = artistic_stereo_label_fitter.frame_analysis(
            left, right, depth, config, reference
        )
        self.assertIsNotNone(analysis)
        self.assertTrue(analysis["diagnostics"]["reference_disparity_used"])
        self.assertAlmostEqual(
            float(np.median(analysis["disparity"][analysis["valid"]]) * width),
            12.0,
            delta=0.05,
        )
        bad_reference = np.full((height, width), 30.0, np.float32)
        self.assertIsNone(artistic_stereo_label_fitter.frame_analysis(
            left, right, depth, config, bad_reference
        ))

    def test_latches_shot_and_rejects_inverted_depth(self):
        rng = np.random.default_rng(23)
        height, width = 192, 384
        left = rng.integers(0, 256, (height, width), dtype=np.uint8)
        disparity = np.full((height, width), 6, np.int32)
        disparity[:, width // 2:] = 18
        baseline = disparity.astype(np.float32) / width
        right = np.empty_like(left)
        right[:] = left[:, -1:]
        for x in range(width):
            target = x - disparity[0, x]
            if 0 <= target < width:
                right[:, target] = left[:, x]
        config = artistic_stereo_label_fitter.LabelFitterConfig(
            analysis_width=width, search_fraction=0.125,
            min_support=0.02, min_fit_pixels=128,
        )
        analysis = artistic_stereo_label_fitter.frame_analysis(
            left, right, baseline, config
        )
        outputs = artistic_stereo_label_fitter.finalize_shot(
            [analysis, analysis], config
        )
        self.assertIsNotNone(outputs)
        self.assertAlmostEqual(
            outputs[0]["baseline_multiplier"],
            outputs[1]["baseline_multiplier"],
        )
        self.assertGreater(outputs[0]["confidence"], 0.05)
        self.assertIn("shot_scale_raw", outputs[0]["diagnostics"])
        self.assertIn("scale_clamped", outputs[0]["diagnostics"])
        inverted = artistic_stereo_label_fitter.frame_analysis(
            left, right, -baseline, config
        )
        self.assertIsNone(artistic_stereo_label_fitter.finalize_shot(
            [inverted], config
        ))


if __name__ == "__main__":
    unittest.main()
