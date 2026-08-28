#!/usr/bin/env python3
"""Focused tests for the evaluation-only high-resolution camera/statistics A/B."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from compare_prod_zipdepth_highres_statistics import (  # noqa: E402
    compare_pair,
    map_refined_with_statistics,
)
from depth_mapping_v2 import MappingV2Config, generate_depth_mapping_v2  # noqa: E402


class ProdZipDepthHighresStatisticsTests(unittest.TestCase):
    def test_treatment_is_exact_existing_v2_oracle_on_refined_field(self):
        refined = np.asarray([
            [0.0, 0.2, 1.2, 2.4, 2.8, 3.0],
            [0.1, 0.5, 1.4, 2.2, 2.7, 3.1],
            [0.3, 0.7, 1.1, 2.0, 2.5, 3.2],
            [0.4, 0.8, 1.0, 1.8, 2.3, 3.3],
        ], dtype=np.float32)
        config = MappingV2Config(pop_strength=1.75)

        treatment = map_refined_with_statistics(refined, refined, config)
        existing = generate_depth_mapping_v2(refined, config)

        self.assertEqual(treatment.center, existing.diagnostics.selected_center)
        self.assertEqual(treatment.collapsed, existing.diagnostics.collapsed)
        np.testing.assert_array_equal(treatment.pre_spatial, existing.pre_limiter_parallax)
        np.testing.assert_array_equal(treatment.final, existing.parallax)

    def test_identical_convex_replication_changes_neither_camera_nor_parallax(self):
        coarse = np.asarray([[0.0, 1.0, 2.0], [0.5, 1.5, 3.0]], dtype=np.float32)
        refined = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)

        result = compare_pair(coarse, refined)

        self.assertEqual(result["camera"]["treatment_minus_control_center_raw"], 0.0)
        self.assertEqual(
            result["post_spatial_parallax_delta"]["source_u"]["maximum_absolute"], 0.0)

    def test_center_shift_is_measured_on_common_refined_geometry(self):
        coarse = np.asarray([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)
        refined = np.asarray([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 3.0, 3.0],
            [2.0, 2.0, 3.0, 3.0],
        ], dtype=np.float32)

        result = compare_pair(coarse, refined, display_widths=(100,))

        self.assertGreater(
            result["camera"]["treatment_minus_control_center_raw"], 0.0)
        self.assertGreater(
            result["pre_spatial_parallax_delta"]["source_u"]["mean_absolute"], 0.0)
        post = result["post_spatial_parallax_delta"]
        self.assertGreater(post["source_u"]["maximum_absolute"], 0.0)
        self.assertAlmostEqual(
            post["equivalent_source_pixels"]["100"]["maximum_absolute"],
            post["source_u"]["maximum_absolute"] * 100.0)

    def test_collapse_decision_belongs_to_selected_statistics_source(self):
        coarse = np.ones((2, 2), dtype=np.float32)
        refined = np.asarray([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 3.0, 3.0],
            [2.0, 2.0, 3.0, 3.0],
        ], dtype=np.float32)

        result = compare_pair(coarse, refined)

        self.assertTrue(result["camera"]["control_collapsed"])
        self.assertFalse(result["camera"]["treatment_collapsed"])
        self.assertGreater(
            result["post_spatial_parallax_delta"]["source_u"]["maximum_absolute"], 0.0)

    def test_rejects_non_exact_2x_pair_or_invalid_display_width(self):
        coarse = np.ones((2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "exactly twice"):
            compare_pair(coarse, np.ones((4, 5), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "unique positive"):
            compare_pair(coarse, np.ones((4, 6), dtype=np.float32), display_widths=(0,))


if __name__ == "__main__":
    unittest.main()
