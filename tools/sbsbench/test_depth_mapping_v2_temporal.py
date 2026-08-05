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
    ownership_refine_candidate,
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

    def test_source_ownership_matches_one_fractional_far_boundary_edit(self):
        candidate = np.zeros((5, 5), dtype=np.float32)
        candidate[2:, :] = np.float32(0.01)
        source = np.zeros((25, 25, 3), dtype=np.uint8)
        source[8:, :, :] = 255
        config = MappingV2Config(max_horizontal_slope=0.001)

        refined = ownership_refine_candidate(candidate, source, config)

        expected = candidate.copy()
        # The filtered profile authenticates one contour; bounded raw refinement recovers the
        # exact 40% foreground coverage of this synthetic source edge.
        expected[1, :] = np.float32(0.004)
        np.testing.assert_allclose(refined, expected, rtol=0.0, atol=1.0e-9)

    def test_source_ownership_720p_has_no_pixel_phase_holes(self):
        target_shape = (434, 770)
        source_scale = np.float32(1280.0 / 770.0)

        # Sweep every interior target-cell phase from the 10% admission floor through the 99%
        # ceiling. Quantization may move an intended edge just outside that interval; those cases
        # must abstain, while every actually admitted case must recover geometric occupancy.
        for nominal_coverage in (0.10, 0.20, 0.40, 0.60, 0.80,
                                 0.90, 0.95, 0.98, 0.99):
            source = np.ones((720, 1280, 3), dtype=np.float32)
            previous_edge = 0
            for split in range(2, target_shape[1] - 1):
                source_edge = int(round(float(
                    np.float32(split) * source_scale -
                    np.float32(nominal_coverage) * source_scale)))
                source[:, previous_edge:source_edge, :] = np.float32(0.0)
                previous_edge = source_edge
                coverage = temporal._ownership_foreground_coverage(
                    source, target_shape, split - 1, target_shape[0] // 2, (1, 0))
                expected_coverage = float(np.float32(
                    (np.float32(split) * source_scale - np.float32(source_edge)) /
                    source_scale))
                with self.subTest(
                        nominal_coverage=nominal_coverage,
                        split=split,
                        source_edge=source_edge):
                    if 0.0998 <= expected_coverage <= 0.9902:
                        self.assertIsNotNone(coverage)
                        self.assertAlmostEqual(
                            float(coverage),
                            float(np.clip(expected_coverage, 0.10, 0.99)),
                            delta=2.0e-4)
                    else:
                        self.assertIsNone(coverage)

    def test_source_ownership_coverage_range_is_resolution_invariant(self):
        # Isolate the normal-axis raster phase while keeping the tangent axis uniform. Native GPU
        # tests separately use full 2D 720p-4K sources and active contours on every allowlisted
        # landscape/portrait tensor shape.
        cases = (
            ("720p", (434, 770), (434, 1280), 0),
            ("1080p", (434, 770), (434, 1920), 0),
            ("1440p", (434, 770), (434, 2560), 0),
            ("4k", (434, 770), (434, 3840), 0),
            ("ultrawide-1022", (434, 1022), (434, 3440), 0),
            ("ultrawide-1036", (434, 1036), (434, 3840), 0),
            ("portrait-770", (770, 434), (1280, 434), 1),
            ("portrait-1022", (1022, 434), (3440, 434), 1),
            ("portrait-1036", (1036, 434), (3840, 434), 1),
        )
        for name, target_shape, source_shape, axis in cases:
            target_extent = target_shape[1] if axis == 0 else target_shape[0]
            source_extent = source_shape[1] if axis == 0 else source_shape[0]
            source_scale = np.float32(source_extent / target_extent)
            for nominal_coverage in (0.10, 0.40, 0.80, 0.98):
                source = np.ones((*source_shape, 3), dtype=np.float32)
                previous_edge = 0
                for split in range(2, target_extent - 1, 31):
                    source_edge = int(round(float(
                        np.float32(split) * source_scale -
                        np.float32(nominal_coverage) * source_scale)))
                    if axis == 0:
                        source[:, previous_edge:source_edge, :] = np.float32(0.0)
                        position = (split - 1, target_shape[0] // 2)
                        direction = (1, 0)
                    else:
                        source[previous_edge:source_edge, :, :] = np.float32(0.0)
                        position = (target_shape[1] // 2, split - 1)
                        direction = (0, 1)
                    previous_edge = source_edge
                    coverage = temporal._ownership_foreground_coverage(
                        source, target_shape, position[0], position[1], direction)
                    expected_coverage = float(np.float32(
                        (np.float32(split) * source_scale - np.float32(source_edge)) /
                        source_scale))
                    with self.subTest(
                            case=name,
                            nominal_coverage=nominal_coverage,
                            split=split):
                        if 0.0998 <= expected_coverage <= 0.9902:
                            self.assertIsNotNone(coverage)
                            self.assertAlmostEqual(
                                float(coverage),
                                float(np.clip(expected_coverage, 0.10, 0.99)),
                                delta=2.0e-4)
                        else:
                            self.assertIsNone(coverage)

    def test_source_ownership_abstains_on_ambiguous_constant_color(self):
        candidate = np.zeros((5, 5), dtype=np.float32)
        candidate[2:, :] = np.float32(0.01)
        ambiguous = np.full((25, 25, 3), 127, dtype=np.uint8)

        refined = ownership_refine_candidate(
            candidate, ambiguous, MappingV2Config(max_horizontal_slope=0.001))

        np.testing.assert_array_equal(refined, candidate)

    def test_source_ownership_abstains_when_source_is_smaller_than_model_grid(self):
        candidate = np.zeros((5, 5), dtype=np.float32)
        candidate[2:, :] = np.float32(0.01)
        for source_shape in ((5, 4), (4, 5)):
            with self.subTest(source_shape=source_shape):
                undersized = np.zeros((*source_shape, 3), dtype=np.uint8)
                undersized[source_shape[0] // 2:, :, :] = 255
                refined = ownership_refine_candidate(
                    candidate, undersized,
                    MappingV2Config(max_horizontal_slope=0.001))
                np.testing.assert_array_equal(refined, candidate)

    def test_source_ownership_competing_contours_have_positive_controls(self):
        target_shape = (434, 770)
        split = 385
        source_a = np.zeros((434, 3840, 3), dtype=np.uint8)
        source_a[:, 1916:, :] = 208
        source_b = np.full((434, 3840, 3), 160, dtype=np.uint8)
        source_b[:, 1918:, :] = 255
        combined = np.zeros((434, 3840, 3), dtype=np.uint8)
        combined[:, 1916:1917, :] = 208
        combined[:, 1917:1918, :] = 160
        combined[:, 1918:, :] = 255

        coverage_a = temporal._ownership_foreground_coverage(
            temporal._ownership_source_rgb(source_a), target_shape,
            split - 1, target_shape[0] // 2, (1, 0))
        coverage_b = temporal._ownership_foreground_coverage(
            temporal._ownership_source_rgb(source_b), target_shape,
            split - 1, target_shape[0] // 2, (1, 0))
        ambiguous = temporal._ownership_foreground_coverage(
            temporal._ownership_source_rgb(combined), target_shape,
            split - 1, target_shape[0] // 2, (1, 0))

        source_scale = 3840.0 / target_shape[1]
        expected_a = (split * source_scale - 1916.0) / source_scale
        expected_b = (split * source_scale - 1918.0) / source_scale
        self.assertAlmostEqual(float(coverage_a), expected_a, delta=2.0e-4)
        self.assertAlmostEqual(float(coverage_b), expected_b, delta=2.0e-4)
        self.assertIsNone(ambiguous)

    def test_source_ownership_rejects_subprofile_opposite_transitions(self):
        target_shape = (434, 770)
        split = 385
        first = np.zeros((434, 3840, 3), dtype=np.uint8)
        first[:, 1917:, :] = 255
        second = np.zeros((434, 3840, 3), dtype=np.uint8)
        second[:, 1919:, :] = 255
        combined = np.zeros((434, 3840, 3), dtype=np.uint8)
        combined[:, 1917:1918, :] = 255
        combined[:, 1919:, :] = 255
        refinement_only = np.zeros((434, 3840, 3), dtype=np.uint8)
        refinement_only[:, 1916:1917, :] = 255
        refinement_only[:, 1918:, :] = 255

        position = (split - 1, target_shape[0] // 2, (1, 0))
        coverage_first = temporal._ownership_foreground_coverage(
            temporal._ownership_source_rgb(first), target_shape, *position)
        coverage_second = temporal._ownership_foreground_coverage(
            temporal._ownership_source_rgb(second), target_shape, *position)
        ambiguous = temporal._ownership_foreground_coverage(
            temporal._ownership_source_rgb(combined), target_shape, *position)
        refinement_ambiguous = temporal._ownership_foreground_coverage(
            temporal._ownership_source_rgb(refinement_only), target_shape, *position)

        self.assertIsNotNone(coverage_first)
        self.assertIsNotNone(coverage_second)
        self.assertIsNone(ambiguous)
        self.assertIsNone(refinement_ambiguous)

    def test_source_ownership_is_stable_across_capture_resolutions_and_directions(self):
        config = MappingV2Config(max_horizontal_slope=0.001)
        # The 5-cell oracle represents calibrated model space. These source extents cover the
        # normal-axis ratios of 720p, 1080p, 1440p, and 4K capture without depending on a lucky
        # integer pixel phase.
        for source_extent in (8, 12, 17, 25):
            for axis in (0, 1):
                for reverse in (False, True):
                    with self.subTest(
                            source_extent=source_extent, axis=axis, reverse=reverse):
                        candidate = np.zeros((5, 5), dtype=np.float32)
                        if axis == 0:
                            candidate[:, :2] = np.float32(0.01) if reverse else 0.0
                            candidate[:, 2:] = 0.0 if reverse else np.float32(0.01)
                        else:
                            candidate[:2, :] = np.float32(0.01) if reverse else 0.0
                            candidate[2:, :] = 0.0 if reverse else np.float32(0.01)

                        source = np.zeros(
                            (source_extent, source_extent, 3), dtype=np.uint8)
                        scale = np.float32(source_extent / 5.0)
                        boundary = np.float32(2.0) * scale
                        edge_offset = np.float32(0.4) * scale
                        edge = int(round(float(
                            boundary + edge_offset if reverse else
                            boundary - edge_offset)))
                        if axis == 0:
                            if reverse:
                                source[:, :edge, :] = 255
                            else:
                                source[:, edge:, :] = 255
                        elif reverse:
                            source[:edge, :, :] = 255
                        else:
                            source[edge:, :, :] = 255

                        refined = ownership_refine_candidate(candidate, source, config)
                        if axis == 0:
                            boundary_slice = refined[:, 2] if reverse else refined[:, 1]
                            original_slice = candidate[:, 2] if reverse else candidate[:, 1]
                        else:
                            boundary_slice = refined[2, :] if reverse else refined[1, :]
                            original_slice = candidate[2, :] if reverse else candidate[1, :]
                        self.assertTrue(np.all(boundary_slice > original_slice))
                        self.assertTrue(np.all(boundary_slice < np.float32(0.01)))

    def test_exact_sequence_threads_source_ownership_into_final_geometry(self):
        raw = np.zeros((5, 5), dtype=np.float64)
        raw[2:, :] = 1.0
        source = np.zeros((25, 25, 3), dtype=np.uint8)
        source[8:, :, :] = 255
        config = MappingV2Config(
            raw_coordinate_scale=0.5,
            pop_strength=1.0,
            gain_per_pop=0.01,
            max_horizontal_slope=0.001,
            max_vertical_shear=0.004,
        )

        identity = generate_first_latch_exact_sequence(
            [raw], [0], [False], "unit-ownership", config)
        refined = generate_first_latch_exact_sequence(
            [raw], [0], [False], "unit-ownership", config,
            source_rgb_fields=[source])
        identity_row = identity.state_trace["frames"][0]
        refined_row = refined.state_trace["frames"][0]

        self.assertEqual(identity_row["ownership_raised_fraction"], 0.0)
        self.assertEqual(identity_row["ownership_max_raise_source_u"], 0.0)
        self.assertGreater(refined_row["ownership_raised_fraction"], 0.0)
        self.assertGreater(refined_row["ownership_max_raise_source_u"], 0.0)
        self.assertNotEqual(
            refined_row["ownership_refined_sha256"],
            identity_row["ownership_refined_sha256"])
        self.assertGreater(
            float(np.max(np.abs(
                refined.parallax_fields[0] - identity.parallax_fields[0]))),
            0.0)

    def test_exact_sequence_rejects_source_count_or_shape_mismatch(self):
        raw = np.asarray([[-1.0, 0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "source RGB fields"):
            generate_first_latch_exact_sequence(
                [raw], [0], [False], "unit-ownership", source_rgb_fields=[])
        with self.assertRaisesRegex(ValueError, "HxWx3"):
            generate_first_latch_exact_sequence(
                [raw], [0], [False], "unit-ownership",
                source_rgb_fields=[np.zeros((4, 4), dtype=np.uint8)])

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

    def test_exact_sequence_latches_accepted_stage_camera_until_confirmed_cut(self):
        accepted = self._separated_three_stage_field()
        moved = accepted + 8.0
        ambiguous = np.linspace(20.0, 23.0, accepted.size).reshape(accepted.shape)
        result = generate_first_latch_exact_sequence(
            [accepted, moved, ambiguous], [0, 1, 2], [False, False, True],
            "unit-cut", MappingV2Config(pop_strength=1.0))
        first, held, relatched = result.state_trace["frames"]

        self.assertGreater(first["center"], first["observed_mean"])
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
                "authenticated-raw-source-color-histogram-plus-seven-v2-compute-shaders-persistent-gpu-state-v7",
            "manifest_sha256": "0" * 64,
            "contract_canonical_sha256": CONTRACT_CANONICAL_SHA256,
            "tensor_shape": {"width": width, "height": height},
            "shader_sequence": list(V2_GPU_SHADER_SEQUENCE),
            "state_persistence": "single-buffer-whole-sequence",
            "numpy_role": "comparison-only-not-render-authority",
        }
        validate_v2_state_trace(native, result.frame_ids)

        # The native producer authenticates the shared raw-range and histogram passes before the
        # seven coordinate passes. Omitting those inputs used to make every real native trace fail
        # the Python gate even though this synthetic fixture (built from the Python constant)
        # passed.  Keep the exact boundary explicit and prove that the former six-only trace is
        # rejected.
        self.assertEqual(
            V2_GPU_SHADER_SEQUENCE[:2],
            ("depth_minmax_cs.hlsl", "depth_hist_cs.hlsl"),
        )
        missing_histogram_inputs = copy.deepcopy(native)
        missing_histogram_inputs["producer"]["shader_sequence"] = list(
            V2_GPU_SHADER_SEQUENCE[2:])
        with self.assertRaisesRegex(ValueError, "invalid native GPU producer evidence"):
            validate_v2_state_trace(missing_histogram_inputs, result.frame_ids)

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
        self.assertEqual(len(V2_STATE_TRACE_FIELDS), 41)
        self.assertIn("ownership_raised_fraction", V2_STATE_TRACE_FIELDS)
        self.assertIn("ownership_max_raise_source_u", V2_STATE_TRACE_FIELDS)
        self.assertIn("ownership_refined_sha256", V2_STATE_TRACE_FIELDS)
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
