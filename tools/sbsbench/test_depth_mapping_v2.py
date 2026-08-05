#!/usr/bin/env python3
"""Focused unit tests for the simplified depth-coordinate-v2 NumPy oracle."""

from dataclasses import fields
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from depth_coordinate_v2_contract import CALIBRATED_DEFAULTS  # noqa: E402
from depth_mapping_v2 import (  # noqa: E402
    MappingV2Config,
    SCENE_HISTOGRAM_BINS,
    SCENE_HISTOGRAM_SOURCE_BINS,
    _otsu_three_class_split,
    _scene_histogram_128,
    asymmetric_curve,
    calibrate_coordinate,
    curve_relative_coordinate,
    decode_direct_parallax,
    encode_direct_parallax,
    generate_depth_mapping_v2,
    horizontal_lipschitz_majorant,
    pointwise_soft_container,
    select_scene_coordinate,
    vertical_lipschitz_majorant,
    vertical_lipschitz_minorant,
)


class DepthMappingV2Test(unittest.TestCase):
    def test_rejects_invalid_raw_input(self):
        for raw in (
                np.asarray([]), np.empty((0, 2)), np.zeros((1, 1, 1)),
                np.asarray([[np.nan]]), np.asarray([[np.inf]]),
                np.asarray([[1.0 + 2.0j]]), np.asarray([["not-depth"]])):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    generate_depth_mapping_v2(raw)

    def test_calibration_latches_mean_with_fixed_authenticated_scale(self):
        raw = np.asarray([[1.0, 2.0], [3.0, 8.0]], dtype=np.float64)
        expected_mean = float(np.mean(raw))
        expected_std = float(np.std(raw))
        calibration = calibrate_coordinate(
            raw, MappingV2Config(raw_coordinate_scale=0.5))
        self.assertAlmostEqual(calibration.center, expected_mean)
        self.assertAlmostEqual(calibration.observed_std, expected_std)
        self.assertEqual(calibration.scale, 0.5)

    @staticmethod
    def _separated_three_stage_field() -> np.ndarray:
        rng = np.random.default_rng(123)
        return np.concatenate((
            rng.normal(1.0, 0.2, 7680),
            rng.normal(2.0, 0.2, 3200),
            rng.normal(4.0, 0.2, 1920),
        )).reshape(100, 128)

    def test_scene_histogram_exactly_merges_existing_gpu_256_bin_layout(self):
        raw = np.arange(SCENE_HISTOGRAM_SOURCE_BINS, dtype=np.float32).reshape(16, 16)
        counts, raw_min, raw_max = _scene_histogram_128(raw)
        self.assertEqual(counts.shape, (SCENE_HISTOGRAM_BINS,))
        np.testing.assert_array_equal(counts, np.full(SCENE_HISTOGRAM_BINS, 2))
        self.assertEqual(raw_min, 0.0)
        self.assertEqual(raw_max, 255.0)
        # Inclusive classes and strict ascending tie retention are part of GPU parity.
        self.assertEqual(_otsu_three_class_split(counts)[:2], (41, 84))

    def test_scene_selector_adopts_only_a_deep_upper_valley(self):
        raw = self._separated_three_stage_field()
        selection = select_scene_coordinate(raw)
        self.assertTrue(selection.adopted)
        self.assertEqual(selection.reason, "accepted-upper-stage-boundary")
        self.assertEqual(selection.lower_split_bin, 36)
        self.assertEqual(selection.upper_split_bin, 72)
        self.assertEqual(selection.class_counts, (7654, 3226, 1920))
        self.assertLessEqual(selection.valley_ratio, 0.75)
        self.assertAlmostEqual(
            selection.candidate_center + MappingV2Config().raw_coordinate_scale,
            selection.upper_split_raw)
        self.assertEqual(selection.selected_center, selection.upper_split_raw)
        self.assertGreater(selection.candidate_center, selection.observed_mean)
        self.assertEqual(selection.convergence_curve, 0.0)
        canonical_zero, curved_zero = curve_relative_coordinate(
            np.asarray([[selection.selected_center]]),
            selection.selected_center,
            MappingV2Config().raw_coordinate_scale,
            convergence_curve=selection.convergence_curve,
        )
        self.assertEqual(float(canonical_zero[0, 0]), 0.0)
        self.assertEqual(float(curved_zero[0, 0]), 0.0)

        result = generate_depth_mapping_v2(raw)
        self.assertAlmostEqual(result.diagnostics.center_mean, selection.observed_mean)
        self.assertAlmostEqual(result.diagnostics.selected_center, selection.selected_center)
        self.assertEqual(result.diagnostics.scene_coordinate, selection)
        self.assertEqual(result.diagnostics.convergence_curve, 0.0)

    def test_scene_selector_abstains_on_shallow_valley_and_preserves_mean_semantics(self):
        raw = np.linspace(1.0, 4.0, 12800, dtype=np.float32).reshape(100, 128)
        selection = select_scene_coordinate(raw)
        self.assertFalse(selection.adopted)
        self.assertEqual(selection.reason, "upper-valley-not-separated")
        self.assertEqual(selection.valley_ratio, 1.0)
        self.assertEqual(selection.selected_center, selection.observed_mean)
        self.assertEqual(selection.convergence_curve, 0.0)

    def test_scene_selector_abstains_when_boundary_center_is_not_above_mean(self):
        rng = np.random.default_rng(456)
        raw = np.concatenate((
            rng.normal(1.0, 0.2, 1500),
            rng.normal(2.0, 0.25, 1800),
            rng.normal(4.0, 0.25, 9500),
        )).reshape(100, 128)
        selection = select_scene_coordinate(raw)
        self.assertFalse(selection.adopted)
        self.assertEqual(selection.reason, "candidate-not-above-mean")
        self.assertLessEqual(selection.valley_ratio, 0.75)
        self.assertLess(selection.candidate_center, selection.observed_mean)
        self.assertEqual(selection.selected_center, selection.observed_mean)

    def test_scene_selector_falls_back_if_reused_gpu_histogram_is_incomplete(self):
        raw = np.asarray([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float64)
        selection = select_scene_coordinate(raw)
        self.assertFalse(selection.adopted)
        self.assertEqual(selection.reason, "incomplete-gpu-histogram")
        self.assertEqual(selection.histogram_total, 3)
        self.assertEqual(selection.selected_center, float(np.mean(raw)))

    def test_collapse_is_absolute_and_keeps_requested_gain_diagnostic(self):
        config = MappingV2Config(
            raw_coordinate_scale=0.25, collapse_abs_epsilon=1.0e-3,
            pop_strength=2.5, gain_per_pop=0.007)
        # A huge DC offset must not alter the absolute population-standard-deviation test.
        raw = 1.0e6 + np.asarray([[0.0, 2.0e-4], [-2.0e-4, 0.0]])
        result = generate_depth_mapping_v2(raw, config)
        self.assertTrue(result.diagnostics.collapsed)
        self.assertAlmostEqual(result.diagnostics.requested_gain, config.parallax_gain)
        self.assertEqual(result.diagnostics.effective_gain, 0.0)
        self.assertEqual(result.diagnostics.container_scale, 1.0)
        self.assertEqual(result.diagnostics.convergence_curve, 0.0)
        np.testing.assert_array_equal(result.parallax, np.zeros(raw.shape, np.float32))

    def test_contract_surface_has_fixed_near_curve_without_occupancy_state(self):
        self.assertEqual(
            tuple(field.name for field in fields(MappingV2Config)),
            ("raw_coordinate_scale", "collapse_abs_epsilon", "far_tau", "near_log_tau",
             "pop_strength", "gain_per_pop", "max_horizontal_slope",
             "max_vertical_shear", "vertical_majorant_share",
             "direct_container_limit"))
        result = generate_depth_mapping_v2(np.asarray([[-1.0, 0.0, 1.0]]))
        names = set(result.diagnostics.to_dict())
        for removed in (
                "upper_l4_semideviation", "spatial_support_ratio",
                "spatial_support_neighbors", "source_u_budget",
                "source_u_safety_scale", "collapse_relative_epsilon"):
            self.assertNotIn(removed, names)

    def test_fixed_near_curve_feeds_the_pointwise_soft_container(self):
        canonical = np.concatenate((np.full(25, 12.0), np.full(75, -4.0))).reshape(10, 10)
        config = MappingV2Config(
            raw_coordinate_scale=0.5, pop_strength=20.0, gain_per_pop=0.01,
            max_horizontal_slope=0.99)
        raw = canonical * config.raw_coordinate_scale
        result = generate_depth_mapping_v2(raw, config)
        base_curve = asymmetric_curve(canonical, config)
        self.assertEqual(result.diagnostics.container_scale, 1.0)
        np.testing.assert_allclose(
            result.desired_parallax,
            (base_curve * config.parallax_gain).astype(np.float32), rtol=1.0e-6)
        np.testing.assert_allclose(
            result.pre_limiter_parallax,
            pointwise_soft_container(
                base_curve * config.parallax_gain,
                config.direct_container_limit).astype(np.float32),
            rtol=1.0e-6)

    def test_fallback_convergence_is_separate_curve_coordinate_and_exactly_zero(self):
        raw = np.asarray([[-2.0, 0.0, 2.0]], dtype=np.float64)
        result = generate_depth_mapping_v2(
            raw, MappingV2Config(raw_coordinate_scale=0.5, max_horizontal_slope=0.99))
        self.assertEqual(CALIBRATED_DEFAULTS.convergence_curve_default, 0.0)
        self.assertEqual(result.diagnostics.convergence_curve, 0.0)
        self.assertAlmostEqual(result.diagnostics.center_mean, 0.0)
        self.assertEqual(float(result.desired_parallax[0, 1]), 0.0)
        self.assertAlmostEqual(
            result.diagnostics.curve_far_limit,
            -MappingV2Config().far_tau - result.diagnostics.convergence_curve)

    def test_asymmetric_curve_is_monotone_far_bounded_and_near_unbounded(self):
        config = MappingV2Config(far_tau=0.25, near_log_tau=2.0)
        x = np.asarray([-100.0, -1.0, 0.0, 1.0, 4.0, 1000.0])
        y = asymmetric_curve(x, config)
        self.assertTrue(np.all(np.diff(y) > 0.0))
        # At an extreme finite input expm1 may round to its asymptote exactly.
        self.assertGreaterEqual(float(y[0]), -config.far_tau)
        self.assertGreater(float(y[1]), -config.far_tau)
        self.assertEqual(float(y[2]), 0.0)
        self.assertEqual(float(y[3]), 1.0)
        self.assertGreater(float(y[-1]), float(y[-2]))

    def test_additive_raw_offset_preserves_mapping_with_fixed_scale(self):
        raw = np.asarray([[-2.0, -0.5, 0.0], [0.5, 1.0, 4.0]])
        config = MappingV2Config(raw_coordinate_scale=0.5)
        original = generate_depth_mapping_v2(raw, config)
        transformed = generate_depth_mapping_v2(raw + 1234.0, config)
        np.testing.assert_allclose(
            original.canonical, transformed.canonical, rtol=2.0e-5, atol=2.0e-5)
        np.testing.assert_allclose(
            original.parallax, transformed.parallax, rtol=2.0e-5, atol=2.0e-5)

    def test_pointwise_container_bounds_without_mutating_requested_gain(self):
        raw = np.asarray([[-100.0, 0.0, 100.0]], dtype=np.float64)
        config = MappingV2Config(
            raw_coordinate_scale=0.5, pop_strength=20.0, gain_per_pop=0.01,
            max_horizontal_slope=0.99, direct_container_limit=0.04)
        result = generate_depth_mapping_v2(raw, config)
        self.assertEqual(result.diagnostics.requested_gain, config.parallax_gain)
        self.assertEqual(result.diagnostics.container_scale, 1.0)
        self.assertEqual(result.diagnostics.effective_gain, config.parallax_gain)
        self.assertLessEqual(
            float(np.max(np.abs(result.pre_limiter_parallax))), 0.04000001)
        self.assertLessEqual(float(np.max(np.abs(result.parallax))), 0.04000001)

    def test_pointwise_container_is_odd_monotone_and_does_not_scale_other_values(self):
        values = np.asarray([-1000.0, -0.01, 0.0, 0.01, 1000.0])
        contained = pointwise_soft_container(values, 0.04)
        self.assertTrue(np.all(np.diff(contained) > 0.0))
        self.assertLessEqual(float(np.max(np.abs(contained))), 0.04)
        self.assertEqual(float(contained[2]), 0.0)
        self.assertAlmostEqual(float(contained[1]), -float(contained[3]))
        unchanged = pointwise_soft_container(np.asarray([0.01]), 0.04)
        with_outlier = pointwise_soft_container(np.asarray([0.01, 1.0e6]), 0.04)
        self.assertEqual(float(unchanged[0]), float(with_outlier[0]))

    def test_horizontal_majorant_never_lowers_and_honors_step(self):
        field = np.asarray([[0.0, -1.0, -1.0, 0.5], [-0.2, 0.7, -0.8, -0.9]])
        limited = horizontal_lipschitz_majorant(field, 0.1)
        self.assertTrue(np.all(limited >= field))
        self.assertLessEqual(float(np.max(np.abs(np.diff(limited, axis=1)))), 0.10000001)
        # High/near samples are preserved exactly; only the low/background side is raised to form
        # the bounded no-fill collar.
        self.assertEqual(float(limited[0, 3]), 0.5)
        self.assertEqual(float(limited[1, 1]), 0.7)
        self.assertGreater(float(limited[0, 1]), -1.0)

    def test_vertical_majorant_never_lowers_and_honors_aspect_matched_step(self):
        field = np.asarray([
            [-1.0, -0.2, -1.0],
            [-1.0, 0.8, -1.0],
            [-1.0, -0.4, -1.0],
        ])
        limited = vertical_lipschitz_majorant(field, 0.1)
        self.assertTrue(np.all(limited >= field))
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(limited, axis=0)))), 0.10000001)
        self.assertEqual(float(limited[1, 1]), 0.8)
        self.assertGreater(float(limited[0, 1]), -0.2)

    def test_vertical_minorant_never_raises_and_honors_aspect_matched_step(self):
        field = np.asarray([
            [-1.0, -0.2, -1.0],
            [-1.0, 0.8, -1.0],
            [-1.0, -0.4, -1.0],
        ])
        limited = vertical_lipschitz_minorant(field, 0.1)
        self.assertTrue(np.all(limited <= field))
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(limited, axis=0)))), 0.10000001)
        self.assertAlmostEqual(float(limited[0, 1]), -0.2)
        self.assertLess(float(limited[1, 1]), 0.8)

    def test_mapping_composes_vertical_share_then_horizontal_majorant(self):
        raw = np.zeros((7, 9), dtype=np.float64)
        raw[3, 4] = 8.0
        config = MappingV2Config(
            raw_coordinate_scale=0.5,
            max_horizontal_slope=0.5,
            max_vertical_shear=0.01,
        )
        result = generate_depth_mapping_v2(raw, config)
        candidate = result.pre_limiter_parallax.astype(np.float64)
        vertical = result.post_vertical_parallax.astype(np.float64)
        final = result.parallax.astype(np.float64)
        vertical_step = config.max_vertical_shear / raw.shape[1]
        upper = vertical_lipschitz_majorant(candidate, vertical_step)
        lower = vertical_lipschitz_minorant(candidate, vertical_step)
        expected_vertical = (
            config.vertical_majorant_share * upper +
            (1.0 - config.vertical_majorant_share) * lower)
        np.testing.assert_allclose(vertical, expected_vertical, atol=1.0e-8)
        self.assertTrue(np.all(vertical <= upper + 1.0e-8))
        self.assertTrue(np.all(vertical >= lower - 1.0e-8))
        self.assertTrue(np.any(vertical < candidate - 1.0e-8))
        self.assertTrue(np.all(final >= vertical - 1.0e-8))
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(vertical, axis=0)))) * raw.shape[1],
            config.max_vertical_shear + 1.0e-6)
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(final, axis=0)))) * raw.shape[1],
            config.max_vertical_shear + 1.0e-6)
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(final, axis=1)))) * raw.shape[1],
            config.max_horizontal_slope + 1.0e-6)
        self.assertEqual(result.diagnostics.horizontal_limiter_illegal_lower_count, 0)
        self.assertGreater(result.diagnostics.conditioner_lowered_fraction, 0.0)

    def test_direct_parallax_interchange_round_trips_without_clipping(self):
        field = np.asarray([[-0.04, -0.01, 0.0, 0.02, 0.04]], dtype=np.float32)
        encoded = encode_direct_parallax(field)
        self.assertTrue(np.all((encoded >= 0.0) & (encoded <= 1.0)))
        np.testing.assert_allclose(decode_direct_parallax(encoded), field, atol=1.0e-8)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            encode_direct_parallax(np.asarray([[0.041]]))

    def test_invalid_configuration_fails_closed(self):
        invalid = (
            MappingV2Config(raw_coordinate_scale=-1.0),
            MappingV2Config(collapse_abs_epsilon=0.0),
            MappingV2Config(far_tau=0.0),
            MappingV2Config(near_log_tau=float("nan")),
            MappingV2Config(pop_strength=0.0),
            MappingV2Config(gain_per_pop=-1.0),
            MappingV2Config(max_horizontal_slope=0.0),
            MappingV2Config(max_horizontal_slope=1.0),
            MappingV2Config(max_vertical_shear=0.0),
            MappingV2Config(max_vertical_shear=float("inf")),
            MappingV2Config(vertical_majorant_share=0.0),
            MappingV2Config(vertical_majorant_share=1.0),
            MappingV2Config(vertical_majorant_share=1.0e-50),
            MappingV2Config(vertical_majorant_share=1.0 - 1.0e-12),
            MappingV2Config(vertical_majorant_share=float("nan")),
            MappingV2Config(direct_container_limit=0.0),
        )
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    generate_depth_mapping_v2(np.asarray([[0.0, 1.0]]), config)

    def test_decimal_share_uses_shader_float32_coefficients(self):
        config = MappingV2Config(vertical_majorant_share=0.7)
        raw = np.asarray([
            [-1.0, 0.2, 0.9],
            [0.8, -0.4, 0.1],
            [-0.2, 1.0, -0.7],
        ])
        result = generate_depth_mapping_v2(raw, config)
        candidate = result.pre_limiter_parallax.astype(np.float64)
        step = config.max_vertical_shear / raw.shape[1]
        upper = vertical_lipschitz_majorant(candidate, step)
        lower = vertical_lipschitz_minorant(candidate, step)
        majorant_share = np.float32(0.7)
        minorant_share = np.float32(np.float32(1.0) - majorant_share)
        expected = float(majorant_share) * upper + float(minorant_share) * lower
        np.testing.assert_allclose(
            result.post_vertical_parallax.astype(np.float64), expected, atol=1.0e-8)

    def test_zero_raw_coordinate_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "raw_coordinate_scale"):
            generate_depth_mapping_v2(
                np.asarray([[-1.0, 1.0]]),
                MappingV2Config(raw_coordinate_scale=0.0))


if __name__ == "__main__":
    unittest.main()
