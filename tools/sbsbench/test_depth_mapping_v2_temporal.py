#!/usr/bin/env python3
"""Focused temporal-contract tests for the simplified V2 coordinate camera."""

import copy
from dataclasses import fields
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import depth_mapping_v2_temporal as temporal  # noqa: E402
from depth_coordinate_v2_contract import (  # noqa: E402
    CALIBRATED_DEFAULTS, CONTRACT_CANONICAL_SHA256, MODEL_CALIBRATIONS)
from depth_mapping_v2 import (  # noqa: E402
    MappingV2Config,
    asymmetric_curve, curve_relative_coordinate, generate_depth_mapping_v2,
    horizontal_lipschitz_majorant,
    pointwise_soft_container,
    vertical_lipschitz_majorant,
    vertical_lipschitz_minorant)
from depth_mapping_v2_temporal import (  # noqa: E402
    CoordinateState,
    TemporalConfig,
    TemporalCoordinateController,
    V2_GPU_SHADER_SEQUENCE,
    V2_STATE_TRACE_FIELDS,
    V2_STATE_TRACE_SCHEMA,
    generate_first_latch_exact_sequence,
    map_timeline_flat_on_unusable,
    moment_candidate,
    parallax_for_state,
    validate_v2_state_trace,
)


class DepthMappingV2TemporalTest(unittest.TestCase):
    @staticmethod
    def _separated_three_stage_field(offset: float = 0.0) -> np.ndarray:
        rng = np.random.default_rng(123)
        return np.concatenate((
            rng.normal(1.0 + offset, 0.2, 7680),
            rng.normal(2.0 + offset, 0.2, 3200),
            rng.normal(4.0 + offset, 0.2, 1920),
        )).reshape(100, 128)

    def test_exact_selected_reference_does_not_construct_research_controller(self):
        raw = np.asarray([[-1.0, 0.0, 1.0]])
        with mock.patch.object(
                temporal, "TemporalCoordinateController",
                side_effect=AssertionError("research controller must not own exact replay")):
            result = generate_first_latch_exact_sequence(
                [raw], [0], [False], "unit-test")
        self.assertEqual(result.state_trace["schema"], V2_STATE_TRACE_SCHEMA)

    def test_moment_candidate_uses_fixed_scale_and_observes_moments(self):
        raw = np.asarray([[1.0, 2.0], [3.0, 8.0]])
        floor = float(np.std(raw)) * 2.0
        candidate = moment_candidate(raw, MappingV2Config(raw_coordinate_scale=floor))
        self.assertAlmostEqual(candidate.center, float(np.mean(raw)))
        self.assertAlmostEqual(candidate.observed_std, float(np.std(raw)))
        self.assertAlmostEqual(candidate.scale, floor)
        self.assertEqual(candidate.raw_min, 1.0)
        self.assertEqual(candidate.raw_max, 8.0)
        self.assertEqual(
            tuple(field.name for field in fields(type(candidate))),
            ("center", "scale", "observed_std", "raw_min", "raw_max", "collapsed"))

    def test_first_policy_holds_camera_across_collapse_until_cut(self):
        config = MappingV2Config(raw_coordinate_scale=0.5)
        first = moment_candidate(np.asarray([[-1.0, 0.0, 1.0]]), config)
        moved = moment_candidate(np.asarray([[9.0, 10.0, 11.0]]), config)
        collapsed = moment_candidate(np.full((1, 3), 5.0), config)
        controller = TemporalCoordinateController(TemporalConfig("first"), config)

        seeded = controller.update(first, 0.0, False)
        held = controller.update(moved, 0.1, False)
        self.assertTrue(seeded.reset)
        self.assertEqual(held.state.center, seeded.state.center)
        self.assertEqual(held.state.scale, seeded.state.scale)
        self.assertEqual(held.state.convergence_curve, 0.0)

        unavailable = controller.update(collapsed, 0.2, False)
        self.assertTrue(unavailable.state.valid)
        self.assertEqual(unavailable.state.center, seeded.state.center)
        resumed = controller.update(moved, 0.3, False)
        self.assertFalse(resumed.reset)
        self.assertEqual(resumed.state.center, seeded.state.center)
        cleared = controller.update(collapsed, 0.4, True)
        self.assertFalse(cleared.state.valid)
        relatched = controller.update(first, 0.5, False)
        self.assertTrue(relatched.reset)
        self.assertAlmostEqual(relatched.state.center, first.center)

    def test_exact_sequence_map_uses_pointwise_soft_container(self):
        canonical = np.concatenate((np.full(25, 12.0), np.full(75, -4.0))).reshape(10, 10)
        config = MappingV2Config(
            raw_coordinate_scale=0.5, pop_strength=2.0, gain_per_pop=0.10)
        result = generate_first_latch_exact_sequence(
            [canonical * config.raw_coordinate_scale], [0], [False], "unit-cut", config)
        row = result.state_trace["frames"][0]
        base_curve = asymmetric_curve(canonical, config)
        mapping = generate_depth_mapping_v2(
            canonical * config.raw_coordinate_scale,
            config,
        )
        self.assertEqual(row["container_scale"], 1.0)
        np.testing.assert_allclose(
            mapping.pre_limiter_parallax,
            pointwise_soft_container(
                base_curve * config.parallax_gain,
                config.direct_container_limit).astype(np.float32),
            atol=2.0e-6)
        np.testing.assert_allclose(
            result.parallax_fields[0], mapping.parallax, atol=2.0e-6)

    def test_exact_sequence_latches_arithmetic_mean_camera_until_confirmed_cut(self):
        accepted = self._separated_three_stage_field()
        moved = accepted + 8.0
        ambiguous = np.linspace(20.0, 23.0, accepted.size).reshape(accepted.shape)
        result = generate_first_latch_exact_sequence(
            [accepted, moved, ambiguous], [0, 1, 2], [False, False, True],
            "unit-cut", MappingV2Config(pop_strength=1.0))
        first, held, relatched = result.state_trace["frames"]

        self.assertAlmostEqual(first["center"], first["observed_mean"])
        self.assertEqual(first["convergence_curve"], 0.0)
        self.assertEqual(held["center"], first["center"])
        self.assertEqual(held["convergence_curve"], first["convergence_curve"])
        self.assertFalse(held["confirmed_cut"])
        self.assertTrue(relatched["confirmed_cut"])
        self.assertAlmostEqual(relatched["center"], relatched["observed_mean"])
        self.assertEqual(relatched["convergence_curve"], 0.0)
        validate_v2_state_trace(result.state_trace, result.frame_ids)

    def test_exact_sequence_never_relatches_after_initial_camera(self):
        initial = self._separated_three_stage_field()
        moved = initial + 2.0
        fields_in = [initial, *([moved] * 10)]
        config = MappingV2Config(pop_strength=1.0)
        result = generate_first_latch_exact_sequence(
            fields_in, list(range(len(fields_in))), [False] * len(fields_in),
            "unit-cut", config)
        rows = result.state_trace["frames"]

        self.assertTrue(all(row["center"] == rows[0]["center"] for row in rows[1:]))
        self.assertEqual([row["calibration_revision"] for row in rows], [1] * len(rows))
        self.assertEqual(rows[0]["cut_attribution"], "initialization")
        self.assertTrue(all(row["cut_attribution"] == "none" for row in rows[1:]))
        validate_v2_state_trace(result.state_trace, result.frame_ids)

    def test_trace_rejects_corrupt_stage_selection_without_histogram_fields(self):
        accepted = self._separated_three_stage_field()
        result = generate_first_latch_exact_sequence(
            [accepted], [0], [False], "unit-cut")

        changed = copy.deepcopy(result.state_trace)
        changed["frames"][0]["convergence_curve"] = -0.05
        with self.assertRaisesRegex(ValueError, "unknown convergence selection"):
            validate_v2_state_trace(changed, result.frame_ids)

        changed = copy.deepcopy(result.state_trace)
        changed["frames"][0]["center"] = (
            changed["frames"][0]["observed_mean"] - 0.25)
        with self.assertRaisesRegex(ValueError, "invalid camera center"):
            validate_v2_state_trace(changed, result.frame_ids)

    def test_convergence_is_separate_exact_zero_and_invalid_maps_flat(self):
        raw = np.asarray([[0.0, 1.0, 2.0]])
        state = CoordinateState(center=1.0, scale=1.0)
        self.assertEqual(CALIBRATED_DEFAULTS.convergence_curve_default, 0.0)
        self.assertEqual(state.convergence_curve, 0.0)
        mapped = parallax_for_state(raw, state)
        self.assertEqual(float(mapped[0, 1]), 0.0)
        np.testing.assert_array_equal(
            parallax_for_state(raw, CoordinateState(0.0, 1.0, 0.0, False)),
            np.zeros(raw.shape))

    def test_exact_sequence_latches_soft_bounds_flattens_and_relatches(self):
        base = np.asarray([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
        spike = np.asarray([[-1000.0, 0.0, 1000.0], [-1000.0, 0.0, 1000.0]])
        collapsed = np.full(base.shape, 7.0)
        next_shot = base + 10.0
        cut_shot = base + 20.0
        fields_in = [base, spike, base, collapsed, next_shot, cut_shot]
        config = MappingV2Config(raw_coordinate_scale=0.5)
        result = generate_first_latch_exact_sequence(
            fields_in, list(range(len(fields_in))),
            [False, False, False, False, False, True], "unit-cut", config)
        rows = result.state_trace["frames"]

        first_scale = config.raw_coordinate_scale
        self.assertAlmostEqual(rows[0]["center"], float(np.mean(base)))
        self.assertAlmostEqual(rows[0]["latched_scale"], first_scale)
        self.assertEqual(rows[1]["center"], rows[0]["center"])
        self.assertEqual(rows[1]["latched_scale"], rows[0]["latched_scale"])

        self.assertTrue(all(row["container_scale"] == 1.0 for row in rows))
        requested = config.parallax_gain
        self.assertTrue(all(row["requested_gain"] == requested for row in rows))
        self.assertEqual(rows[1]["effective_gain"], requested)
        self.assertLessEqual(rows[1]["pre_limiter_max_abs_source_u"], 0.04)

        self.assertTrue(rows[3]["collapsed"])
        self.assertFalse(rows[3]["frame_valid"])
        self.assertTrue(rows[3]["camera_valid"])
        self.assertEqual(rows[3]["effective_gain"], 0.0)
        self.assertEqual(rows[3]["requested_gain"], requested)
        np.testing.assert_array_equal(result.parallax_fields[3], np.zeros(base.shape, np.float32))

        self.assertEqual(
            [row["calibration_revision"] for row in rows], [1, 1, 1, 1, 1, 2])
        self.assertAlmostEqual(rows[4]["center"], rows[0]["center"])
        self.assertTrue(rows[5]["confirmed_cut"])
        self.assertEqual(rows[5]["confirmed_cut_count"], 1)
        self.assertAlmostEqual(rows[5]["center"], float(np.mean(cut_shot)))
        self.assertTrue(all(row["convergence_curve"] == 0.0 for row in rows))
        self.assertTrue(all(
            row["final_vertical_shear_max"] <= config.max_vertical_shear + 1.0e-6
            for row in rows))
        validate_v2_state_trace(result.state_trace, result.frame_ids)

    def test_exact_sequence_decimal_share_matches_shader_float32_coefficients(self):
        # This deterministic witness crosses a float32 rounding boundary while remaining small.
        # Recreate both the HLSL coefficients and the old float64 mistake independently,
        # including the production row majorant.
        row_values = np.asarray([
            3.700167660523528,
            -6.901352286814722,
            -2.1227203094893454,
            2.7232882788932873,
            -0.4017646507734014,
        ])
        raw = np.repeat(row_values[:, None], 67, axis=1)
        config = MappingV2Config(
            raw_coordinate_scale=0.5,
            far_tau=0.15,
            near_log_tau=2.0,
            pop_strength=1.0,
            gain_per_pop=0.0075,
            vertical_majorant_share=0.7,
            direct_container_limit=10.0,
        )
        result = generate_first_latch_exact_sequence(
            [raw], [0], [False], "unit-decimal-share", config)
        row = result.state_trace["frames"][0]
        _, curved = curve_relative_coordinate(
            raw,
            row["center"],
            row["latched_scale"],
            config,
            convergence_curve=row["convergence_curve"],
        )
        candidate = pointwise_soft_container(
            curved * row["effective_gain"], config.direct_container_limit)
        max_vertical_step = config.max_vertical_shear / raw.shape[1]
        upper = vertical_lipschitz_majorant(candidate, max_vertical_step)
        lower = vertical_lipschitz_minorant(candidate, max_vertical_step)
        majorant_f32 = np.float32(config.vertical_majorant_share)
        minorant_f32 = np.float32(np.float32(1.0) - majorant_f32)
        expected = (
            horizontal_lipschitz_majorant(
                float(majorant_f32) * upper + float(minorant_f32) * lower,
                config.max_horizontal_slope / raw.shape[1],
            )
        ).astype("<f4")
        float64_coefficients = horizontal_lipschitz_majorant(
            config.vertical_majorant_share * upper +
            (1.0 - config.vertical_majorant_share) * lower,
            config.max_horizontal_slope / raw.shape[1],
        ).astype("<f4")

        np.testing.assert_array_equal(result.parallax_fields[0], expected)
        self.assertTrue(np.any(expected != float64_coefficients))

    def test_invalid_trace_row_keeps_requested_gain_but_has_zero_effective_gain(self):
        base = np.asarray([[-1.0, 0.0, 1.0]])
        invalid = np.full(base.shape, np.nan)
        result = generate_first_latch_exact_sequence(
            [base, invalid, base + 5.0], [0, 1, 2],
            [False, False, False], "unit-cut")
        trace = copy.deepcopy(result.state_trace)
        row = trace["frames"][1]
        validate_v2_state_trace(trace, result.frame_ids)
        self.assertEqual(row["requested_gain"], trace["frames"][0]["requested_gain"])
        self.assertEqual(row["effective_gain"], 0.0)

        changed = copy.deepcopy(trace)
        changed["frames"][1]["requested_gain"] = 0.0
        with self.assertRaisesRegex(ValueError, "requested gain"):
            validate_v2_state_trace(changed, result.frame_ids)

    def test_unusable_confirmed_cut_clears_camera(self):
        dense_u = np.concatenate((np.full(25, 2.0), np.full(75, -2.0 / 3.0)))
        dense = dense_u.reshape(10, 10) * 0.5
        invalid = np.full(dense.shape, np.nan)
        result = generate_first_latch_exact_sequence(
            [dense, invalid], [0, 1], [False, True], "unit-cut")
        first, cleared = result.state_trace["frames"]
        self.assertTrue(first["camera_valid"])
        self.assertFalse(cleared["camera_valid"])
        validate_v2_state_trace(result.state_trace, result.frame_ids)

    def test_native_trace_binds_calibrated_tensor_shape(self):
        calibration = MODEL_CALIBRATIONS[0]
        width, height = calibration.calibrated_input_shapes[0]
        texels = width * height
        near_count = texels // 10
        canonical = np.empty(texels, dtype=np.float64)
        canonical[:near_count] = 2.0
        canonical[near_count:] = -2.0 * near_count / (texels - near_count)
        raw = canonical.reshape(height, width) * calibration.raw_coordinate_scale
        result = generate_first_latch_exact_sequence(
            [raw], [0], [False], "unit-cut",
            MappingV2Config(raw_coordinate_scale=calibration.raw_coordinate_scale))
        native = copy.deepcopy(result.state_trace)
        native["diagnostic_method"] = (
            "gpu-frame-moments-and-rendered-fields-v5")
        native["producer"] = {
            "authority":
                "authenticated-raw-depth-plus-six-v2-compute-shaders-persistent-gpu-state-v9",
            "manifest_sha256": "0" * 64,
            "contract_canonical_sha256": CONTRACT_CANONICAL_SHA256,
            "tensor_shape": {"width": width, "height": height},
            "shader_sequence": list(V2_GPU_SHADER_SEQUENCE),
            "state_persistence": "single-buffer-whole-sequence",
            "numpy_role": "comparison-only-not-render-authority",
        }
        validate_v2_state_trace(native, result.frame_ids)

        # The active fused model exposes the sole refined output on the exact 2x grid while
        # retaining this calibration for its embedded DAV2 input. Native replay must accept
        # that exact relationship without admitting arbitrary enlarged tensors.
        high = copy.deepcopy(native)
        high["producer"]["tensor_shape"] = {
            "width": width * 2, "height": height * 2,
        }
        validate_v2_state_trace(high, result.frame_ids)

        arbitrary_high = copy.deepcopy(high)
        arbitrary_high["producer"]["tensor_shape"]["width"] += 16
        with self.assertRaisesRegex(ValueError, "unauthenticated replay tensor shape"):
            validate_v2_state_trace(arbitrary_high, result.frame_ids)

        # The native producer contains exactly the six passes that consume or produce V2
        # state/geometry. Reintroducing either former histogram-only pass must fail attribution.
        self.assertNotIn("depth_minmax_cs.hlsl", V2_GPU_SHADER_SEQUENCE)
        self.assertNotIn("depth_hist_cs.hlsl", V2_GPU_SHADER_SEQUENCE)
        extra_histogram_input = copy.deepcopy(native)
        extra_histogram_input["producer"]["shader_sequence"] = [
            "depth_hist_cs.hlsl", *V2_GPU_SHADER_SEQUENCE]
        with self.assertRaisesRegex(ValueError, "invalid native GPU producer evidence"):
            validate_v2_state_trace(extra_histogram_input, result.frame_ids)

        changed = copy.deepcopy(native)
        changed["producer"]["tensor_shape"]["width"] += 16
        with self.assertRaisesRegex(ValueError, "unauthenticated replay tensor shape"):
            validate_v2_state_trace(changed, result.frame_ids)

    def test_trace_rejects_impossible_collapse_and_frame_validity_combinations(self):
        raw = np.asarray([[-1.0, 0.0, 1.0]])
        result = generate_first_latch_exact_sequence(
            [raw], [0], [False], "unit-cut")

        changed = copy.deepcopy(result.state_trace)
        changed["frames"][0]["collapsed"] = True
        with self.assertRaisesRegex(ValueError, "input/collapse/frame validity"):
            validate_v2_state_trace(changed, result.frame_ids)

        changed = copy.deepcopy(result.state_trace)
        changed["frames"][0]["frame_valid"] = False
        with self.assertRaisesRegex(ValueError, "input/collapse/frame validity"):
            validate_v2_state_trace(changed, result.frame_ids)

        changed = copy.deepcopy(result.state_trace)
        changed["frames"][0]["observed_std"] = 0.0
        with self.assertRaisesRegex(ValueError, "input/collapse/frame validity"):
            validate_v2_state_trace(changed, result.frame_ids)

    def test_trace_layout_records_vertical_conditioner_attribution(self):
        self.assertEqual(len(V2_STATE_TRACE_FIELDS), 38)
        self.assertIn("vertical_majorant_raised_fraction", V2_STATE_TRACE_FIELDS)
        self.assertIn("vertical_majorant_sha256", V2_STATE_TRACE_FIELDS)
        self.assertIn("vertical_conditioned_sha256", V2_STATE_TRACE_FIELDS)
        self.assertIn("conditioner_raised_fraction", V2_STATE_TRACE_FIELDS)
        self.assertIn("conditioner_lowered_fraction", V2_STATE_TRACE_FIELDS)
        self.assertIn("final_vertical_shear_max", V2_STATE_TRACE_FIELDS)
        for removed in (
                "upper_l4", "source_u_budget", "source_u_safety_scale",
                "observed_coordinate_scale", "supported_raw_maximum",
                "collapse_relative_epsilon", "spatial_support_ratio"):
            self.assertNotIn(removed, V2_STATE_TRACE_FIELDS)
        self.assertEqual(
            tuple(field.name for field in fields(CoordinateState)),
            ("center", "scale", "convergence_curve", "valid"))

    def test_timeline_flattens_unusable_depth_without_mutating_valid_neighbors(self):
        valid = np.asarray([[-1.0, 0.0, 1.0]])
        flat = np.zeros_like(valid)
        candidates = [moment_candidate(valid), moment_candidate(flat), moment_candidate(valid)]
        states = [
            CoordinateState(0.0, 1.0),
            CoordinateState(0.0, 1.0, 0.0, False),
            CoordinateState(0.0, 1.0),
        ]
        output = map_timeline_flat_on_unusable(
            [valid, flat, valid], candidates, states, [False, False, False])
        self.assertTrue(np.any(output[0] != 0.0))
        np.testing.assert_array_equal(output[1], np.zeros_like(flat))
        np.testing.assert_array_equal(output[0], output[2])


if __name__ == "__main__":
    unittest.main()
