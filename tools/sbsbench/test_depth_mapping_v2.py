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
    asymmetric_curve,
    calibrate_coordinate,
    container_scale_for_curve_range,
    dense_near_weight,
    decode_direct_parallax,
    encode_direct_parallax,
    effective_near_log_tau,
    generate_depth_mapping_v2,
    horizontal_lipschitz_majorant,
    near_tail_count_and_coverage,
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

    def test_contract_surface_has_only_scene_near_tail_adaptation(self):
        self.assertEqual(
            tuple(field.name for field in fields(MappingV2Config)),
            ("raw_coordinate_scale", "collapse_abs_epsilon", "far_tau", "near_log_tau",
             "near_tail_probe_u", "near_tail_coverage_low", "near_tail_coverage_high",
             "near_log_tau_dense",
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

    def test_near_tail_coverage_selects_dense_tau_with_smoothstep(self):
        config = MappingV2Config()
        canonical = np.concatenate((np.full(20, 2.0), np.full(80, -0.5)))
        count, coverage = near_tail_count_and_coverage(canonical, config)
        self.assertEqual(count, 20)
        self.assertAlmostEqual(coverage, 0.20)
        position = ((coverage - config.near_tail_coverage_low) /
                    (config.near_tail_coverage_high - config.near_tail_coverage_low))
        expected_weight = position * position * (3.0 - 2.0 * position)
        self.assertAlmostEqual(dense_near_weight(coverage, config), expected_weight)
        self.assertAlmostEqual(
            effective_near_log_tau(coverage, config),
            config.near_log_tau + expected_weight *
            (config.near_log_tau_dense - config.near_log_tau))
        self.assertEqual(dense_near_weight(0.10, config), 0.0)
        self.assertEqual(dense_near_weight(0.25, config), 1.0)
        self.assertEqual(effective_near_log_tau(0.10, config), config.near_log_tau)
        self.assertEqual(effective_near_log_tau(0.25, config), config.near_log_tau_dense)

    def test_dense_render_curve_cannot_loosen_base_tau_container(self):
        # Keep 25% of the zero-mean field in a very long near tail so the dense tau is selected.
        canonical = np.concatenate((np.full(25, 12.0), np.full(75, -4.0))).reshape(10, 10)
        config = MappingV2Config(
            raw_coordinate_scale=0.5, pop_strength=20.0, gain_per_pop=0.01,
            max_horizontal_slope=0.99)
        raw = canonical * config.raw_coordinate_scale
        result = generate_depth_mapping_v2(raw, config)
        self.assertAlmostEqual(result.diagnostics.near_tail_coverage, 0.25)
        self.assertEqual(result.diagnostics.dense_near_weight, 1.0)
        self.assertEqual(
            result.diagnostics.effective_near_log_tau, config.near_log_tau_dense)
        base_curve = asymmetric_curve(canonical, config)
        dense_curve = asymmetric_curve(
            canonical, config, near_log_tau=config.near_log_tau_dense)
        expected_base, _ = container_scale_for_curve_range(
            float(np.min(base_curve)), float(np.max(base_curve)), config)
        relaxed_dense, _ = container_scale_for_curve_range(
            float(np.min(dense_curve)), float(np.max(dense_curve)), config)
        self.assertAlmostEqual(result.diagnostics.container_scale, expected_base)
        self.assertLess(result.diagnostics.container_scale, relaxed_dense)

    def test_convergence_is_separate_curve_coordinate_and_exactly_zero(self):
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

    def test_hard_container_is_derived_without_mutating_requested_gain(self):
        raw = np.asarray([[-100.0, 0.0, 100.0]], dtype=np.float64)
        config = MappingV2Config(
            raw_coordinate_scale=0.5, pop_strength=20.0, gain_per_pop=0.01,
            max_horizontal_slope=0.99, direct_container_limit=0.04)
        result = generate_depth_mapping_v2(raw, config)
        self.assertEqual(result.diagnostics.requested_gain, config.parallax_gain)
        self.assertLess(result.diagnostics.container_scale, 1.0)
        self.assertAlmostEqual(
            result.diagnostics.effective_gain,
            config.parallax_gain * result.diagnostics.container_scale)
        self.assertLessEqual(float(np.max(np.abs(result.parallax))), 0.04000001)

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
            MappingV2Config(near_tail_probe_u=0.5),
            MappingV2Config(near_tail_coverage_low=-0.1),
            MappingV2Config(near_tail_coverage_high=1.1),
            MappingV2Config(
                near_tail_coverage_low=0.22, near_tail_coverage_high=0.15),
            MappingV2Config(near_log_tau_dense=0.0),
            MappingV2Config(near_log_tau=1.0, near_log_tau_dense=1.0),
            MappingV2Config(near_log_tau=1.0, near_log_tau_dense=2.0),
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
