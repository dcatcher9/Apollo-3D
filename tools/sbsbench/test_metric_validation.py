"""Adversarial qualification tests for evaluator metrics used as model labels.

These tests deliberately construct geometry that image matching can miss.  They exercise the
harness-only exact inverse-warp contract and the fail-closed/no-evidence behavior of supporting
detectors.  Keep this module independent of the large repository contract suite so it can be run
quickly while changing a metric implementation.
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import sbsbench  # noqa: E402
import run_eval  # noqa: E402


CHANNELS = [
    "raw_reproject_source_u_normalized",
]


def mapping_shape(eye_width=64, height=32):
    """Return a complete unit-content warp-map shape contract."""
    return {
        "schema": 1,
        "dtype": "float32-le",
        "layout": "row-major",
        "channels": CHANNELS.copy(),
        "width": eye_width * 2,
        "height": height,
        "eye_width": eye_width,
        "eye_height": height,
        "source_width": eye_width,
        "source_height": height,
        "content_scale_x": 1.0,
        "content_scale_y": 1.0,
    }


def identity_mapping(shape):
    """Create the exact-map output for two unwarped, fully covered eyes."""
    height = shape["height"]
    eye_width = shape["eye_width"]
    mapping = np.zeros((height, eye_width * 2), dtype=np.float32)
    source_u = (np.arange(eye_width, dtype=np.float32) + 0.5) / eye_width
    for eye_index in range(2):
        eye = mapping[:, eye_index * eye_width:(eye_index + 1) * eye_width]
        eye[...] = source_u[None, :]
    return mapping


def aspect_fitted_identity_mapping(shape):
    """Create the exact source-U map for an unwarped aspect-fitted source."""
    height = shape["height"]
    eye_width = shape["eye_width"]
    scale_x = float(shape["content_scale_x"])
    lo_x = 0.5 * (1.0 - scale_x)
    output_u = (np.arange(eye_width, dtype=np.float32) + 0.5) / eye_width
    content = (output_u >= lo_x) & (output_u <= lo_x + scale_x)
    source_u = np.zeros(eye_width, dtype=np.float32)
    source_u[content] = (output_u[content] - lo_x) / scale_x
    mapping = np.zeros((height, eye_width * 2), dtype=np.float32)
    mapping[:, :eye_width] = source_u
    mapping[:, eye_width:] = source_u
    return mapping


def shift_eye_output(mapping, shape, eye_index, pixels):
    """Translate one eye's recovered output position while preserving the source contract."""
    eye_width = shape["eye_width"]
    scale_x = float(shape["content_scale_x"])
    eye = mapping[:, eye_index * eye_width:(eye_index + 1) * eye_width]
    eye -= float(pixels) / (scale_x * eye_width)


def set_signed_disparity(mapping, disparity, mask=None):
    """Encode signed disparity by offsetting the exact sampled-source-U coordinate."""
    height, packed_width = mapping.shape[:2]
    eye_width = packed_width // 2
    disparity = np.broadcast_to(np.asarray(disparity, dtype=np.float32),
                                (height, eye_width))
    if mask is None:
        mask = np.ones((height, eye_width), dtype=bool)
    for eye_index, eye_sign in ((0, -1.0), (1, 1.0)):
        sampled_u = mapping[:, eye_index * eye_width:(eye_index + 1) * eye_width]
        sampled_u[mask] += disparity[mask] / (2.0 * eye_sign * eye_width)


def textured_source(width, height):
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    return np.clip(
        0.45 + 0.16 * np.sin(xx * 0.19) + 0.12 * np.cos(yy * 0.27)
        + 0.08 * np.sin((xx + 1.7 * yy) * 0.11), 0.02, 0.98).astype(np.float32)


def render_exact_eyes(source, mapping, shape):
    height, width = shape["height"], shape["eye_width"]
    v = np.broadcast_to(
        ((np.arange(height, dtype=np.float32) + 0.5) / height)[:, None],
        (height, width))
    return [sbsbench._sample_scalar_uv(
        source, np.clip(mapping[:, index * width:(index + 1) * width], 0.0, 1.0), v)
            for index in range(2)]


def shift_y(image, pixels):
    result = np.empty_like(image)
    if pixels == 0:
        return image.copy()
    if pixels > 0:
        result[pixels:] = image[:-pixels]
        result[:pixels] = image[:1]
    else:
        pixels = -pixels
        result[:-pixels] = image[pixels:]
        result[-pixels:] = image[-1:]
    return result


def exact_metrics(mapping, shape, depth=None):
    """Evaluate with a fully covered companion warp mask."""
    warp_mask = np.zeros(mapping.shape, dtype=np.float32)
    return sbsbench.exact_warp_mapping_metrics(
        mapping, shape, depth=depth, warp_mask=warp_mask)


class ExactWarpMapLoaderTests(unittest.TestCase):
    def write_mapping(self, directory, values):
        path = os.path.join(directory, "warp_map_0.f32")
        np.asarray(values, dtype="<f4").tofile(path)
        return path

    def test_loader_rejects_wrong_size(self):
        shape = mapping_shape(8, 4)
        with tempfile.TemporaryDirectory() as directory:
            expected_values = shape["width"] * shape["height"]
            path = self.write_mapping(directory, np.zeros(expected_values - 1))
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                sbsbench.load_warp_mapping(path, shape)

    def test_loader_rejects_nonfinite_values(self):
        shape = mapping_shape(8, 4)
        mapping = identity_mapping(shape)
        mapping[0, 0] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_mapping(directory, mapping)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                sbsbench.load_warp_mapping(path, shape)

    def test_loader_rejects_unknown_contract(self):
        shape = mapping_shape(8, 4)
        shape["channels"][0] = "ambiguous_coordinate"
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_mapping(directory, identity_mapping(mapping_shape(8, 4)))
            with self.assertRaisesRegex(ValueError, "shape contract"):
                sbsbench.load_warp_mapping(path, shape)

    def test_loader_preserves_finite_raw_coordinates_outside_image(self):
        shape = mapping_shape(8, 4)
        mapping = identity_mapping(shape)
        mapping[0, 0] = -0.125
        mapping[0, -1] = 1.25
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_mapping(directory, mapping)
            loaded = sbsbench.load_warp_mapping(path, shape)
        self.assertEqual(float(loaded[0, 0]), -0.125)
        self.assertEqual(float(loaded[0, -1]), 1.25)


class ExactGeometryMetricTests(unittest.TestCase):
    def test_raw_offscreen_demand_is_excluded_from_actual_binocular_disparity(self):
        shape = mapping_shape(100, 24)
        clean = identity_mapping(shape)
        offscreen = clean.copy()
        boundary_subject = np.zeros((shape["height"], shape["eye_width"]), dtype=bool)
        boundary_subject[:, 80:] = True
        set_signed_disparity(offscreen, 8.0, boundary_subject)

        clean_metrics = exact_metrics(clean, shape)
        metrics = exact_metrics(offscreen, shape)

        # Comfort measures only the part of the shifted subject that remains mutually visible;
        # it does not double either eye's raw off-image request.
        self.assertAlmostEqual(metrics["exact_positive_disparity_pct"], 0.0, places=6)
        pixel_pct = sbsbench.perceived_disparity_pct(1.0, 100, 24)
        self.assertAlmostEqual(
            metrics["exact_negative_disparity_pct"] / pixel_pct, 8.0, delta=0.02)
        # Consumed-map topology exposes the boundary collapse without a redundant clamp alias.
        self.assertGreater(metrics["exact_mapping_stretch_pct"],
                           clean_metrics["exact_mapping_stretch_pct"])
        self.assertNotIn("exact_source_clamp_pct", metrics)
        self.assertNotIn("exact_forward_coverage_pct", metrics)

    def test_extreme_tail_catches_small_local_three_percent_disparity(self):
        shape = mapping_shape(100, 100)
        mapping = identity_mapping(shape)
        # One percent of the eye violates the 3% comfort boundary.  It is intentionally too
        # small to move the p5/p95 stereo-volume statistic, but must move the p99.9 hard tail.
        local_subject = np.zeros((100, 100), dtype=bool)
        local_subject[20:30, 20:30] = True
        disparity_px = 3.1 * shape["height"] * sbsbench.REFERENCE_STREAM_ASPECT / 100.0
        set_signed_disparity(mapping, disparity_px, local_subject)

        metrics = exact_metrics(mapping, shape)

        # Apollo near disparity is negative in the standard xR-xL convention.
        self.assertGreater(metrics["exact_negative_disparity_pct"], 3.0)
        self.assertGreater(metrics["exact_over_3pct_area_pct"], 0.1)

    def test_common_translation_is_not_binocular_disparity(self):
        shape = mapping_shape(120, 60)
        mapping = identity_mapping(shape)
        shift_eye_output(mapping, shape, 0, 5.0)
        shift_eye_output(mapping, shape, 1, 5.0)

        metrics = exact_metrics(mapping, shape)

        self.assertAlmostEqual(metrics["exact_positive_disparity_pct"], 0.0, places=6)
        self.assertAlmostEqual(metrics["exact_negative_disparity_pct"], 0.0, places=6)
        self.assertAlmostEqual(metrics["exact_symmetry_residual_p95_px"], 5.0, delta=0.02)
        self.assertGreater(metrics["exact_binocular_support_count"], 16)

    def test_one_eye_ten_pixel_shift_measures_ten_not_twenty(self):
        shape = mapping_shape(160, 80)
        mapping = identity_mapping(shape)
        shift_eye_output(mapping, shape, 1, 10.0)

        metrics = exact_metrics(mapping, shape)
        pixel_pct = sbsbench.perceived_disparity_pct(1.0, 160, 80)

        self.assertAlmostEqual(
            metrics["exact_positive_disparity_pct"] / pixel_pct, 10.0, delta=0.03)
        self.assertAlmostEqual(metrics["exact_negative_disparity_pct"], 0.0, places=6)
        self.assertAlmostEqual(metrics["exact_symmetry_residual_p95_px"], 5.0, delta=0.02)

    def test_symmetric_near_geometry_uses_xright_minus_xleft_sign(self):
        shape = mapping_shape(160, 80)
        mapping = identity_mapping(shape)
        depth = np.broadcast_to(
            np.linspace(0.0, 1.0, shape["eye_width"], dtype=np.float32),
            (shape["height"], shape["eye_width"])).copy()
        set_signed_disparity(mapping, 8.0 * (depth - 0.5))

        metrics = exact_metrics(mapping, shape, depth=depth)

        self.assertGreater(metrics["exact_negative_disparity_pct"], 0.0)
        self.assertEqual(metrics["exact_polarity_ok"], 100.0)
        self.assertLess(metrics["exact_symmetry_residual_p95_px"], 0.1)

    def test_fold_overlap_and_map_gap_are_not_mutual_visibility(self):
        shape = mapping_shape(160, 60)
        clean = identity_mapping(shape)
        corrupt = clean.copy()
        right = corrupt[:, shape["eye_width"]:]
        # This extra increasing run overlaps the later identity run in source U, then releases
        # through a fold. The shared inversion must reject the ambiguous source interval.
        right[:, 48:80] = np.linspace(0.55, 0.75, 32, dtype=np.float32)

        clean_metrics = exact_metrics(clean, shape)
        corrupt_metrics = exact_metrics(corrupt, shape)

        self.assertGreater(corrupt_metrics["exact_mapping_fold_pct"], 0.0)
        self.assertLess(corrupt_metrics["exact_binocular_support_pct"],
                        clean_metrics["exact_binocular_support_pct"] - 5.0)

    def test_forward_holes_are_excluded_from_mutual_visibility(self):
        shape = mapping_shape(120, 60)
        mapping = identity_mapping(shape)
        clean = exact_metrics(mapping, shape)
        mask = np.zeros(mapping.shape, dtype=np.float32)
        mask[:, 30:50] = 1.0

        holed = sbsbench.exact_warp_mapping_metrics(
            mapping, shape, warp_mask=mask)

        self.assertLess(holed["exact_binocular_support_pct"],
                        clean["exact_binocular_support_pct"] - 10.0)

    def test_binocular_support_measures_common_output_area_not_only_source_samples(self):
        shape = mapping_shape(160, 60)
        clean = exact_metrics(identity_mapping(shape), shape)
        collapsed = identity_mapping(shape)
        output_u = ((np.arange(shape["eye_width"], dtype=np.float32) + 0.5)
                    / shape["eye_width"])
        narrow = np.clip(output_u * 2.0, 0.0, 1.0)
        collapsed[:, :shape["eye_width"]] = narrow
        collapsed[:, shape["eye_width"]:] = narrow

        measured = exact_metrics(collapsed, shape)

        # The first half still spans nearly every source-grid sample, but it represents only half
        # an eye of rendered support. A source-count percentage would incorrectly stay near 100%.
        self.assertGreater(measured["exact_binocular_support_count"],
                           clean["exact_binocular_support_count"] * 0.8)
        self.assertLess(measured["exact_binocular_support_pct"], 60.0)

    def test_actual_geometry_is_aspect_fit_robust(self):
        for scale_x, scale_y in ((0.75, 1.0), (1.0, 0.75)):
            with self.subTest(scale_x=scale_x, scale_y=scale_y):
                shape = mapping_shape(160, 80)
                shape["content_scale_x"] = scale_x
                shape["content_scale_y"] = scale_y
                mapping = aspect_fitted_identity_mapping(shape)
                shift_eye_output(mapping, shape, 1, 10.0)

                metrics = exact_metrics(mapping, shape)
                pixel_pct = sbsbench.perceived_disparity_pct(1.0, 160, 80)

                self.assertAlmostEqual(
                    metrics["exact_positive_disparity_pct"] / pixel_pct, 10.0, delta=0.05)
                self.assertAlmostEqual(
                    metrics["exact_symmetry_residual_p95_px"], 5.0, delta=0.03)

    def test_depth_polarity_inversion_fails(self):
        shape = mapping_shape(64, 32)
        mapping = identity_mapping(shape)
        depth = np.broadcast_to(np.linspace(0.0, 1.0, shape["eye_width"], dtype=np.float32),
                                (shape["height"], shape["eye_width"])).copy()
        # High-is-near depth is deliberately assigned decreasing signed disparity.
        set_signed_disparity(mapping, -4.0 * (depth - 0.5))

        metrics = exact_metrics(mapping, shape, depth=depth)

        self.assertEqual(metrics["exact_polarity_ok"], 0.0)
        self.assertGreater(metrics["exact_polarity_support_pct"], 0.0)

    def test_local_polarity_detects_small_inversion_hidden_from_global_medians(self):
        shape = mapping_shape(96, 48)
        mapping = identity_mapping(shape)
        depth = np.broadcast_to(
            np.linspace(0.0, 1.0, shape["eye_width"], dtype=np.float32),
            (shape["height"], shape["eye_width"])).copy()
        disparity = 5.0 * (depth - 0.5)
        local = np.zeros(depth.shape, dtype=bool)
        local[16:32, 42:54] = True
        disparity[local] *= -1.0
        set_signed_disparity(mapping, disparity)

        metrics = exact_metrics(mapping, shape, depth=depth)

        self.assertEqual(metrics["exact_polarity_ok"], 100.0)
        self.assertGreater(metrics["exact_local_polarity_support_pct"], 0.0)
        self.assertGreater(metrics["exact_local_polarity_component_pct"], 0.0)

    def test_local_polarity_is_clean_for_order_preserving_geometry(self):
        shape = mapping_shape(96, 48)
        mapping = identity_mapping(shape)
        depth = np.broadcast_to(
            np.linspace(0.0, 1.0, shape["eye_width"], dtype=np.float32),
            (shape["height"], shape["eye_width"])).copy()
        set_signed_disparity(mapping, 5.0 * (depth - 0.5))

        metrics = exact_metrics(mapping, shape, depth=depth)

        self.assertEqual(metrics["exact_polarity_ok"], 100.0)
        self.assertAlmostEqual(
            metrics["exact_local_polarity_component_pct"], 0.0, places=6)

    def test_identity_mapping_has_no_stretch_or_fold(self):
        shape = mapping_shape(100, 24)
        metrics = exact_metrics(identity_mapping(shape), shape)

        self.assertAlmostEqual(metrics["exact_mapping_stretch_pct"], 0.0, places=5)
        self.assertAlmostEqual(metrics["exact_mapping_fold_pct"], 0.0, places=5)

    def test_repeated_and_folded_columns_increase_topology_metrics(self):
        shape = mapping_shape(100, 24)
        clean = identity_mapping(shape)
        repeated = clean.copy()
        folded = clean.copy()

        # A repeated block has zero source-coordinate Jacobian.  The stronger corruption repeats
        # a wider block and then steps backward, representing a fold at its release boundary.
        for eye_index in range(2):
            offset = eye_index * shape["eye_width"]
            repeated[:, offset + 20:offset + 31] = repeated[:, offset + 20:offset + 21]
            folded[:, offset + 15:offset + 41] = folded[:, offset + 15:offset + 16]
            folded[:, offset + 41] = folded[:, offset + 14]

        clean_metrics = exact_metrics(clean, shape)
        repeated_metrics = exact_metrics(repeated, shape)
        folded_metrics = exact_metrics(folded, shape)

        self.assertLess(clean_metrics["exact_mapping_stretch_pct"],
                        repeated_metrics["exact_mapping_stretch_pct"])
        self.assertLess(repeated_metrics["exact_mapping_stretch_pct"],
                        folded_metrics["exact_mapping_stretch_pct"])
        self.assertLessEqual(clean_metrics["exact_mapping_fold_pct"],
                             repeated_metrics["exact_mapping_fold_pct"])
        self.assertLess(repeated_metrics["exact_mapping_fold_pct"],
                        folded_metrics["exact_mapping_fold_pct"])


class ExactVerticalMisalignmentTests(unittest.TestCase):
    def evaluate(self, source, mapping, shape, eyes):
        width = shape["eye_width"]
        return sbsbench.exact_vertical_misalignment(
            eyes[0], eyes[1], source,
            mapping[:, :width], mapping[:, width:], shape)

    def test_full_frame_vertical_fault_is_accurate_and_resolution_normalized(self):
        measured = []
        for width, height in ((192, 96), (384, 192)):
            shape = mapping_shape(width, height)
            source = textured_source(width, height)
            mapping = identity_mapping(shape)
            # Exercise real horizontal parallax at the same time. The exact map must cancel it.
            disparity = 5.0 * np.sin(
                np.linspace(0.0, 2.0 * np.pi, width, dtype=np.float32))[None, :]
            set_signed_disparity(mapping, disparity)
            eyes = render_exact_eyes(source, mapping, shape)
            fault = max(2, round(height * 0.0208))
            eyes[1] = shift_y(eyes[1], fault)

            result = self.evaluate(source, mapping, shape, eyes)

            self.assertIsNotNone(result)
            native_px, normalized_pct, support = result
            self.assertAlmostEqual(native_px, fault, delta=0.25)
            self.assertGreater(support, 20.0)
            measured.append(normalized_pct)
        self.assertAlmostEqual(measured[0], measured[1], delta=0.15)

    def test_small_salient_patch_is_not_diluted_by_whole_frame_matching(self):
        width, height = 256, 128
        shape = mapping_shape(width, height)
        source = textured_source(width, height)
        mapping = identity_mapping(shape)
        eyes = render_exact_eyes(source, mapping, shape)
        shifted = shift_y(eyes[1], 4)
        # Only 8% of the frame is corrupt, but it covers a high-contrast central subject-sized
        # region. A single whole-frame/tile median misses this failure.
        y0, y1 = 42, 84
        x0, x1 = 88, 152
        eyes[1][y0:y1, x0:x1] = shifted[y0:y1, x0:x1]

        result = self.evaluate(source, mapping, shape, eyes)

        self.assertIsNotNone(result)
        self.assertGreater(result[0], 2.0)
        self.assertGreater(result[1], 1.5)

    def test_one_percent_vertical_fault_reaches_the_localized_tail(self):
        width, height = 256, 128
        shape = mapping_shape(width, height)
        source = textured_source(width, height)
        mapping = identity_mapping(shape)
        eyes = render_exact_eyes(source, mapping, shape)
        shifted = shift_y(eyes[1], 4)
        # 16x20 / 256x128 = 0.977%.  A p95 tile statistic is structurally blind to this
        # footprint; the retained p99 statistic must cross the candidate 0.10% hard limit.
        eyes[1][50:66, 104:124] = shifted[50:66, 104:124]

        _, normalized_pct, support = self.evaluate(source, mapping, shape, eyes)

        self.assertGreaterEqual(normalized_pct, 0.10)
        self.assertGreater(support, 20.0)

    def test_horizontal_parallax_does_not_become_vertical_fault(self):
        width, height = 256, 128
        shape = mapping_shape(width, height)
        source = textured_source(width, height)
        mapping = identity_mapping(shape)
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        set_signed_disparity(mapping, 7.0 * np.sin(xx * 0.035) * np.cos(yy * 0.021))
        eyes = render_exact_eyes(source, mapping, shape)

        result = self.evaluate(source, mapping, shape, eyes)

        self.assertIsNotNone(result)
        self.assertLess(result[0], 0.10)
        self.assertLess(result[1], 0.10)

    def test_common_vertical_motion_cancels(self):
        width, height = 192, 96
        shape = mapping_shape(width, height)
        source = textured_source(width, height)
        mapping = identity_mapping(shape)
        eyes = [shift_y(eye, 3) for eye in render_exact_eyes(source, mapping, shape)]

        result = self.evaluate(source, mapping, shape, eyes)

        self.assertIsNotNone(result)
        self.assertLess(result[0], 0.10)

    def test_low_texture_abstains(self):
        shape = mapping_shape(192, 96)
        source = np.full((96, 192), 0.4, np.float32)
        mapping = identity_mapping(shape)
        eyes = render_exact_eyes(source, mapping, shape)

        severity_px, severity_pct, support = self.evaluate(source, mapping, shape, eyes)
        self.assertIsNone(severity_px)
        self.assertIsNone(severity_pct)
        self.assertLess(support, 2.0)

    def test_sequence_frame_uses_exact_estimator_instead_of_cross_eye_fallback(self):
        width, height = 192, 96
        shape = mapping_shape(width, height)
        source = textured_source(width, height)
        mapping = identity_mapping(shape)
        eyes = render_exact_eyes(source, mapping, shape)
        eyes[1] = shift_y(eyes[1], 2)
        rgb = np.repeat(np.concatenate(eyes, axis=1)[..., None], 3, axis=2)
        depth = np.broadcast_to(
            np.linspace(0.1, 0.9, width, dtype=np.float32), (height, width)).copy()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "sbs_00000.png")
            Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8)).save(path)
            with mock.patch.object(
                    sbsbench, "disparity_field",
                    side_effect=AssertionError("legacy fallback must not run")):
                metrics, _, _ = sbsbench.measure_seq_frame(
                    path, depth=depth, src_gray=source,
                    warp_mapping=mapping, warp_mapping_shape=shape)

        self.assertAlmostEqual(metrics["vmisalign_p99_px"], 2.0, delta=0.35)
        self.assertGreater(metrics["vmisalign_support_pct"], 20.0)
        self.assertNotIn("source_residual_p50", metrics)
        self.assertNotIn("source_residual_p95", metrics)


class VisibleDisparityMetricTests(unittest.TestCase):
    def test_nonfinite_exact_map_is_a_structural_error(self):
        shape = mapping_shape(100, 48)
        mapping = identity_mapping(shape)
        mapping[0, 0] = np.inf
        source = np.full((shape["height"], shape["eye_width"]), 0.5, np.float32)

        with self.assertRaisesRegex(ValueError, "non-finite"):
            sbsbench.exact_warp_mapping_metrics(mapping, shape)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            sbsbench.exact_visible_disparity_metrics(mapping, shape, source)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            sbsbench.exact_source_relative_metrics(
                source, source, mapping[:, :shape["eye_width"]], shape)

    def test_uniform_source_abstains_instead_of_claiming_invisible_pop(self):
        shape = mapping_shape(100, 48)
        mapping = identity_mapping(shape)
        set_signed_disparity(mapping, 6.0)
        source = np.full((shape["height"], shape["eye_width"]), 0.5, np.float32)

        metrics = sbsbench.exact_visible_disparity_metrics(mapping, shape, source)

        self.assertEqual(metrics["exact_visible_support_pct"], 0.0)
        self.assertNotIn("exact_visible_pop_spread_pct", metrics)

    def test_horizontal_line_does_not_masquerade_as_horizontal_correspondence(self):
        shape = mapping_shape(100, 48)
        mapping = identity_mapping(shape)
        source = np.full((shape["height"], shape["eye_width"]), 0.5, np.float32)
        source[23:25, :] = 0.1
        set_signed_disparity(mapping, 6.0)

        metrics = sbsbench.exact_visible_disparity_metrics(mapping, shape, source)

        self.assertEqual(metrics["exact_visible_support_pct"], 0.0)
        self.assertNotIn("exact_visible_pop_spread_pct", metrics)

    def test_small_vertical_subject_contributes_visible_spread(self):
        shape = mapping_shape(100, 100)
        mapping = identity_mapping(shape)
        source = np.full((100, 100), 0.25, np.float32)
        subject = np.zeros((100, 100), dtype=bool)
        subject[20:40, 20:30] = True
        source[subject] = 0.8
        source[20:40, 24:26] = 0.6
        # Independent background structure at zero disparity supplies the second visible plane;
        # a single constant-disparity contour alone is not a disparity *spread*.
        source[:, 70:72] = 0.45
        disparity_px = 3.1 * shape["height"] * sbsbench.REFERENCE_STREAM_ASPECT / 100.0
        set_signed_disparity(mapping, disparity_px, subject)

        metrics = sbsbench.exact_visible_disparity_metrics(mapping, shape, source)

        self.assertGreater(metrics["exact_visible_support_pct"], 0.0)
        self.assertGreater(metrics["exact_visible_pop_spread_pct"], 3.0)


class EvidenceSemanticsTests(unittest.TestCase):
    def test_conservative_resize_preserves_one_pixel_silhouette(self):
        depth = np.zeros((512, 512), dtype=np.float32)
        depth[:, 257:] = 1.0

        edge = sbsbench.silhouette_edges(depth, 32, 32)

        self.assertTrue(edge.any())
        self.assertGreater(np.count_nonzero(edge), 0)

    def test_image_integrity_rejects_excess_edge_energy(self):
        shape = mapping_shape(96, 48)
        source = np.broadcast_to(
            np.where(np.arange(96) % 2, 0.6, 0.4).astype(np.float32), (48, 96)).copy()
        mapping = identity_mapping(shape)[:, :shape["eye_width"]]
        clean = sbsbench.exact_source_relative_metrics(source, source, mapping, shape)
        sharpened = np.where(source > 0.5, 1.0, 0.0).astype(np.float32)
        excessive = sbsbench.exact_source_relative_metrics(
            sharpened, source, mapping, shape)
        self.assertGreater(clean["image_integrity_pct"], 99.0)
        self.assertLess(excessive["image_integrity_pct"], clean["image_integrity_pct"])

    def test_exact_coverage_catches_missing_dark_content(self):
        shape = mapping_shape(96, 48)
        source = np.full((48, 96), 0.04, np.float32)
        mapping = identity_mapping(shape)[:, :shape["eye_width"]]
        clean = sbsbench.exact_source_relative_metrics(source, source, mapping, shape)
        damaged = source.copy()
        damaged[:, 30:50] = 0.0
        missing = sbsbench.exact_source_relative_metrics(damaged, source, mapping, shape)

        self.assertEqual(clean["source_coverage_pct"], 100.0)
        self.assertLess(missing["source_coverage_pct"], 85.0)

    def test_image_integrity_rejects_wrong_gradient_orientation(self):
        shape = mapping_shape(96, 48)
        x = np.arange(96, dtype=np.float32)[None, :]
        source = np.broadcast_to(0.5 + 0.2 * np.sin(x * 0.5), (48, 96)).copy()
        # Construct equal-energy horizontal bands at the same raster size.
        wrong = np.broadcast_to(
            (0.5 + 0.2 * np.sin(np.arange(48, dtype=np.float32)[:, None] * 0.5)),
            source.shape).copy()
        mapping = identity_mapping(shape)[:, :shape["eye_width"]]
        metrics = sbsbench.exact_source_relative_metrics(wrong, source, mapping, shape)
        self.assertLess(metrics["image_integrity_pct"], 25.0)

    def test_flow_temporal_subtracts_legitimate_source_change(self):
        previous = np.full((48, 80), 0.4, np.float32)
        current = previous + 4.0 / 255.0
        with mock.patch.object(
                sbsbench, "dense_source_flow",
                return_value=(np.zeros_like(current), np.zeros_like(current),
                              np.ones_like(current, dtype=bool))):
            temporal, _, support = sbsbench.flow_temporal_metrics(
                current, current, previous, previous, current, previous)
        self.assertGreater(support, 0.9)
        self.assertLess(temporal, 1e-4)

    def test_hdr_exact_source_uses_production_linear_half_float_preview_contract(self):
        shape = mapping_shape(16, 8)
        x = np.linspace(0.0, 1.0, shape["eye_width"], dtype=np.float32)
        y = np.linspace(0.0, 1.0, shape["height"], dtype=np.float32)[:, None]
        source_rgb = np.stack((
            np.broadcast_to(0.05 + 0.85 * x, (shape["height"], shape["eye_width"])),
            np.broadcast_to(0.1 + 0.65 * y, (shape["height"], shape["eye_width"])),
            np.broadcast_to(0.9 - 0.55 * x, (shape["height"], shape["eye_width"])),
        ), axis=-1).astype(np.float32)
        hdr_scale = 4.0

        # Independent transcription of the harness contract: sRGB decode, scRGB scale, FP16
        # source storage, luminance-preserving preview tone map, then sRGB encode.
        linear = np.where(
            source_rgb <= 0.04045, source_rgb / 12.92,
            ((source_rgb + 0.055) / 1.055) ** 2.4)
        linear = (linear * hdr_scale).astype(np.float16).astype(np.float32)
        luminance = (linear[..., 0] * 0.2126 + linear[..., 1] * 0.7152
                     + linear[..., 2] * 0.0722)
        preview_linear = linear / (1.0 + luminance[..., None])
        preview_linear /= np.maximum(1.0, np.max(preview_linear, axis=-1))[..., None]
        preview_rgb = np.where(
            preview_linear <= 0.0031308, 12.92 * preview_linear,
            1.055 * preview_linear ** (1.0 / 2.4) - 0.055).astype(np.float32)

        sampled_u = identity_mapping(shape)[:, :shape["eye_width"]]
        metrics = sbsbench.exact_source_relative_metrics(
            sbsbench.rgb_luma(preview_rgb), sbsbench.rgb_luma(source_rgb),
            sampled_u, shape, eye_rgb=preview_rgb, src_rgb=source_rgb,
            hdr_scale=hdr_scale)
        wrong_transfer = sbsbench.exact_source_relative_metrics(
            sbsbench.rgb_luma(preview_rgb), sbsbench.rgb_luma(source_rgb),
            sampled_u, shape, eye_rgb=preview_rgb, src_rgb=source_rgb)

        self.assertLess(metrics["source_residual_p95"], 1e-3)
        self.assertLess(metrics["source_color_residual_p95"], 1e-3)
        self.assertGreater(wrong_transfer["source_color_residual_p95"], 5.0)


class FrameLabelContractTests(unittest.TestCase):
    SPECS = {
        "volume": {"label": "reward", "requires": "always"},
        "polarity": {
            "label": "hard", "requires": "exact_polarity_support_pct",
        },
        "warp_risk": {
            "label": "risk", "requires": "warp_cross_row_shear_support_count",
        },
        "diagnostic": {"requires": "always"},
    }

    def test_supported_frame_is_eligible_and_exports_only_label_metrics(self):
        row = {
            "volume": 1.5,
            "polarity": 100.0,
            "exact_polarity_support_pct": 40.0,
            "warp_risk": 2.0,
            "warp_cross_row_shear_support_count": 512.0,
            "diagnostic": 123.0,
        }

        labels = sbsbench.frame_label_evidence(row, self.SPECS)

        self.assertTrue(labels["eligible"])
        self.assertEqual(labels["missing_required"], [])
        self.assertEqual(set(labels["metrics"]), {"volume", "polarity", "warp_risk"})
        self.assertTrue(all(item["state"] == "valid"
                            for item in labels["metrics"].values()))

    def test_measured_zero_support_is_explicit_unsupported_not_invented_zero(self):
        row = {
            "volume": 1.5,
            "exact_polarity_support_pct": 0.0,
            "warp_cross_row_shear_support_count": 0.0,
        }

        labels = sbsbench.frame_label_evidence(row, self.SPECS)

        self.assertTrue(labels["eligible"])
        self.assertEqual(labels["metrics"]["polarity"]["state"], "unsupported")
        self.assertEqual(labels["metrics"]["warp_risk"]["state"], "unsupported")
        self.assertNotIn("value", labels["metrics"]["polarity"])
        self.assertNotIn("value", labels["metrics"]["warp_risk"])

    def test_warp_support_below_detector_minimum_is_unsupported(self):
        row = {
            "volume": 1.5,
            "polarity": 100.0,
            "exact_polarity_support_pct": 40.0,
            "warp_cross_row_shear_support_count": 511.0,
        }

        labels = sbsbench.frame_label_evidence(row, self.SPECS)

        self.assertTrue(labels["eligible"])
        self.assertEqual(labels["metrics"]["warp_risk"]["state"], "unsupported")
        self.assertFalse(sbsbench.metric_evidence_applicable(
            "warp_risk", self.SPECS["warp_risk"], row))

    def test_metric_specific_support_minima_match_detector_contracts(self):
        cases = (
            ("binocular", "exact_binocular_support_count", 1023.0, 1024.0),
            ("visible", "exact_visible_support_count", 255.0, 256.0),
            ("local_polarity", "exact_local_polarity_support_count", 255.0, 256.0),
            ("row_shear", "warp_cross_row_shear_support_count", 511.0, 512.0),
            ("integrity", "image_integrity_support", 0.099, 0.1),
        )
        for metric, support_name, below, enough in cases:
            spec = {"requires": support_name}
            with self.subTest(metric=metric):
                self.assertFalse(sbsbench.metric_evidence_applicable(
                    metric, spec, {support_name: below}))
                self.assertTrue(sbsbench.metric_evidence_applicable(
                    metric, spec, {support_name: enough}))

    def test_missing_or_invalid_required_support_abstains_even_if_metric_exists(self):
        for support in (None, np.nan, -1.0):
            with self.subTest(support=support):
                row = {"volume": 1.5, "polarity": 100.0, "warp_risk": 1.0,
                       "warp_cross_row_shear_support_count": 512.0}
                if support is not None:
                    row["exact_polarity_support_pct"] = support

                labels = sbsbench.frame_label_evidence(row, self.SPECS)

                self.assertFalse(labels["eligible"])
                self.assertIn("polarity", labels["missing_required"])
                self.assertEqual(labels["metrics"]["polarity"]["state"], "missing")
                self.assertIsNone(labels["metrics"]["polarity"]["support"])

    def test_positive_support_without_metric_abstains(self):
        row = {
            "volume": 1.5,
            "exact_polarity_support_pct": 20.0,
            "warp_risk": 1.0,
            "warp_cross_row_shear_support_count": 512.0,
        }

        labels = sbsbench.frame_label_evidence(row, self.SPECS)

        self.assertFalse(labels["eligible"])
        self.assertEqual(labels["missing_required"], ["polarity"])

    def test_hard_gate_fails_closed_on_missing_support_or_supported_metric(self):
        spec = {
            "role": "hard", "requires": "exact_polarity_support_pct",
            "better": "higher", "hard_min": 100.0,
        }
        thresholds = {"metrics": {"exact_polarity_ok": spec}}
        cases = (
            ({"_frame_id": 7}, {}, [7]),
            ({"_frame_id": 8, "exact_polarity_support_pct": 10.0},
             {"exact_polarity_support_pct": 10.0}, [8]),
        )
        for row, aggregate, missing_frames in cases:
            with self.subTest(row=row):
                _, _, failures = run_eval.score_clip_gates(
                    [row], aggregate, thresholds, {})
                self.assertEqual(len(failures), 1)
                self.assertTrue(failures[0]["missing"])
                self.assertEqual(failures[0]["missing_frames"], missing_frames)

        _, _, unsupported = run_eval.score_clip_gates(
            [{"_frame_id": 9, "exact_polarity_support_pct": 0.0}],
            {"exact_polarity_support_pct": 0.0}, thresholds, {})
        self.assertEqual(unsupported, [])

    def test_temporal_primary_evidence_checks_transitions_not_aggregate_keys(self):
        thresholds = {"metrics": {
            "static_jitter_p95": {
                "role": "primary", "requires": "static_support", "better": "lower"},
            "depth_gt_lag_f1_p95": {
                "role": "primary", "requires": "gt_depth_temporal", "better": "lower"},
        }}
        meta = {"source_frame_count": 3, "required_gt_depth": True}
        aggregate = {
            "static_support": 0.5, "static_jitter_p95": 1.2,
            "depth_gt_lag_f1_p95": 0.0,
        }
        rows = [
            {"_frame_id": 1},
            {"_frame_id": 2, "static_support": 0.5, "static_jitter": 1.0,
             "depth_gt_lag_f1": 0.0},
            {"_frame_id": 3, "static_support": 0.5, "static_jitter": 1.4,
             "depth_gt_lag_f1": 0.0},
        ]

        self.assertEqual(run_eval.primary_evidence_failures(
            aggregate, thresholds, "clip", meta, rows=rows), [])

        del rows[-1]["static_jitter"]
        failures = run_eval.primary_evidence_failures(
            aggregate, thresholds, "clip", meta, rows=rows)
        self.assertEqual([item["metric"] for item in failures], ["static_jitter_p95"])
        self.assertEqual(failures[0]["missing_frames"], [3])

    def test_synthetic_temporal_gt_does_not_opt_itself_into_depth_accuracy_gate(self):
        spec = {
            "role": "hard", "requires": "gt_depth_accuracy",
            "better": "higher", "hard_min": 100.0,
        }
        thresholds = {"metrics": {"depth_gt_polarity_ok": spec}}
        row = {"_frame_id": 3, "depth_gt_polarity_ok": 0.0}
        aggregate = {"depth_gt_polarity_ok": 0.0}

        _, _, synthetic_failures = run_eval.score_clip_gates(
            [row], aggregate, thresholds,
            {"gt_depth_kind": "disparity", "required_gt_depth": False})
        _, _, authenticated_failures = run_eval.score_clip_gates(
            [row], aggregate, thresholds,
            {"gt_depth_kind": "disparity", "required_gt_depth": True})

        self.assertEqual(synthetic_failures, [])
        self.assertEqual(len(authenticated_failures), 1)
        self.assertEqual(authenticated_failures[0]["metric"], "depth_gt_polarity_ok")

    def test_numeric_metric_cannot_self_authenticate_temporal_gt_evidence(self):
        self.assertEqual(
            sbsbench.metric_evidence_state(
                "depth_gt_lag_f1_p95", {"requires": "gt_depth_temporal"},
                {"depth_gt_lag_f1_p95": 1.0}, {}),
            "unsupported")

    def test_numeric_but_unsupported_primary_metric_cannot_vote(self):
        spec = {"role": "primary", "axis": "warp", "better": "lower",
                "rel_tol": 0.0, "abs_floor": 0.1,
                "requires": "warp_cross_row_shear_support_count"}
        control = {"clip": {
            "quality": 1.0, "warp_cross_row_shear_support_count": 100.0}}
        treatment = {"clip": {
            "quality": 9.0, "warp_cross_row_shear_support_count": 100.0}}

        decision = sbsbench.evaluate_ab_decision(
            control, treatment, ["clip"], {"quality": spec})

        self.assertEqual(decision["verdict"], "screen_neutral")
        self.assertEqual(decision["improved"], 0)
        self.assertEqual(decision["regressed"], 0)

    def test_policy_aggregate_excludes_subthreshold_numeric_rows(self):
        spec = {"requires": "exact_visible_support_count"}
        rows = [
            {"visible": 99.0, "exact_visible_support_count": 16.0},
            {"visible": 2.0, "exact_visible_support_count": 512.0},
        ]
        raw = sbsbench.aggregate(rows)

        filtered = sbsbench.filter_aggregate_by_evidence(
            rows, raw, {"visible": spec}, {})

        self.assertEqual(raw["visible"], 50.5)
        self.assertEqual(filtered["visible"], 2.0)

    def test_binocular_support_floor_cannot_be_abstained_away(self):
        thresholds = {"metrics": {"exact_binocular_support_pct": {
            "role": "hard", "better": "higher", "hard_min": 80.0,
        }}}
        row = {"_frame_id": 2, "exact_binocular_support_pct": 79.9}

        _, _, failures = run_eval.score_clip_gates(
            [row], {"exact_binocular_support_pct": 0.2}, thresholds, {})

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["metric"], "exact_binocular_support_pct")


class LabelProvenanceTests(unittest.TestCase):
    def test_label_contract_hash_changes_with_metrics_thresholds_or_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {
                "sbsbench.py": "metric-v1\n",
                "sbs_interocular_metrics.py": "registration-v1\n",
                "sbs_interocular_phase_chroma.py": "phase-chroma-v1\n",
                "sbs_interocular_photometric_rivalry.py": "photometric-v1\n",
                "sbs_stereo_window_metrics.py": "window-v1\n",
                "sbs_warp_shear_metrics.py": "shear-v1\n",
                "thresholds.json": "{}\n",
                "run_eval.py": "runner-v1\n",
                "rescore_run.py": "rescorer-v1\n",
            }
            for name, contents in paths.items():
                with open(os.path.join(directory, name), "w", encoding="utf-8") as stream:
                    stream.write(contents)
            with mock.patch.object(run_eval, "SCRIPT_DIR", directory):
                hashes = [run_eval.label_contract_sha()]
                for name in paths:
                    with open(os.path.join(directory, name), "a", encoding="utf-8") as stream:
                        stream.write("semantic-change\n")
                    hashes.append(run_eval.label_contract_sha())

        self.assertEqual(len(hashes), len(set(hashes)))

    def test_label_context_hash_changes_for_renderer_model_source_and_candidate(self):
        base = {
            "label_contract_sha256": "labels",
            "clip_set_sha1": {"clip": "source-and-gt"},
            "conf_sha256": "conf",
            "executable_sha256": "exe",
            "runtime_shader_sha256": "shader",
            "model": "dav2-small",
            "engine_name": "engine",
            "engine_sha256": "engine-sha",
            "onnx_sha256": "onnx-sha",
            "profile": "apollo",
            "extra_args": ["--pop-strength", "1.25"],
            "depth_step": 1,
            "depth_compensation": "none",
            "literal_bestv2": False,
            "adaptive_pop": False,
            "adaptive_pop_max": 1.3,
            "zero_plane": 0.5,
            "metric_runtime": {"python": "3.x", "numpy": "2.x", "pillow": "11.x"},
            "scored_artifact_sha256": {"clip": "artifacts"},
            "training_label_gate": {"passed": True},
        }
        original = run_eval.label_context_sha(base)
        for key, changed in (
                ("clip_set_sha1", {"clip": "different-source-or-gt"}),
                ("conf_sha256", "different-conf"),
                ("executable_sha256", "different-exe"),
                ("runtime_shader_sha256", "different-shader"),
                ("metric_runtime", {"python": "3.x", "numpy": "different", "pillow": "11.x"}),
                ("scored_artifact_sha256", {"clip": "changed-artifacts"}),
                ("training_label_gate", {"passed": False}),
                ("model", "different-model"),
                ("extra_args", ["--pop-strength", "1.3"])):
            with self.subTest(key=key):
                candidate = dict(base)
                candidate[key] = changed
                self.assertNotEqual(original, run_eval.label_context_sha(candidate))
        irrelevant = dict(base)
        irrelevant["report_sha256"] = "presentation-only"
        self.assertEqual(original, run_eval.label_context_sha(irrelevant))


class ReportEvidenceContractTests(unittest.TestCase):
    def test_source_evidence_uses_exact_map_and_never_legacy_matcher(self):
        path = os.path.join(SCRIPT_DIR, "build_report.py")
        with open(path, encoding="utf-8") as stream:
            tree = ast.parse(stream.read(), filename=path)
        functions = {node.name: node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

        def calls(function):
            names = set()
            for node in ast.walk(function):
                if not isinstance(node, ast.Call):
                    continue
                target = node.func
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, ast.Attribute):
                    names.add(target.attr)
            return names

        exact_calls = calls(functions["_exact_source_evidence_for_run"])
        residual_calls = calls(functions["source_residual_evidence"])
        self.assertIn("load_warp_mapping", exact_calls)
        self.assertIn("_sample_rgb_uv", exact_calls)
        self.assertIn("_exact_source_evidence_for_run", residual_calls)
        for legacy in ("source_match_map", "source_align_map"):
            self.assertNotIn(legacy, exact_calls)
            self.assertNotIn(legacy, residual_calls)

        stereo_literals = {
            value.value
            for function_name in ("visual_evidence_images", "visual_evidence_section")
            for value in ast.walk(functions[function_name])
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        }
        self.assertIn("exact_visible_pop_spread_pct", stereo_literals)
        self.assertNotIn("pop_spread_px", stereo_literals)

    def test_current_schema_report_renders_exact_topology_and_missing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            clips_root = os.path.join(directory, "clips")
            clip_dir = os.path.join(clips_root, "demo")
            control_dir = os.path.join(directory, "control")
            treatment_dir = os.path.join(directory, "treatment")
            os.makedirs(clip_dir)
            os.makedirs(os.path.join(control_dir, "demo"))
            os.makedirs(os.path.join(treatment_dir, "demo"))

            height, eye_width = 32, 64
            x = np.linspace(0, 255, eye_width, dtype=np.uint8)
            source = np.stack((
                np.broadcast_to(x, (height, eye_width)),
                np.broadcast_to(np.flip(x), (height, eye_width)),
                np.full((height, eye_width), 96, np.uint8),
            ), axis=-1)
            source[:, eye_width // 2:] = np.clip(
                source[:, eye_width // 2:].astype(np.int16) + 32, 0, 255).astype(np.uint8)
            Image.fromarray(source, "RGB").save(os.path.join(clip_dir, "frame_00000.png"))
            with open(os.path.join(clip_dir, "meta.json"), "w", encoding="utf-8") as stream:
                json.dump({
                    "name": "current-schema report fixture",
                    "content_type": "synthetic",
                    "source_artifacts": "A baked source edge used to verify audit ordering.",
                }, stream)

            shape = mapping_shape(eye_width, height)
            control_map = identity_mapping(shape)
            treatment_map = control_map.copy()
            for eye_index in range(2):
                offset = eye_index * eye_width
                treatment_map[:, offset + 20:offset + 34] = treatment_map[
                    :, offset + 20:offset + 21]

            control_sbs = np.concatenate((source, source), axis=1)
            treatment_sbs = control_sbs.copy()
            treatment_sbs[:, 29:31] = np.array([255, 255, 255], np.uint8)
            warp_mask = np.zeros((height, eye_width * 2, 3), np.uint8)
            warp_mask[12:20, 26:31, 0] = 255
            depth = np.full((height, eye_width), 72, np.uint8)
            depth[:, eye_width // 2:] = 208
            for run_dir, output, mapping in (
                    (control_dir, control_sbs, control_map),
                    (treatment_dir, treatment_sbs, treatment_map)):
                artifact_dir = os.path.join(run_dir, "demo")
                Image.fromarray(output, "RGB").save(
                    os.path.join(artifact_dir, "sbs_00000.png"))
                Image.fromarray(warp_mask, "RGB").save(
                    os.path.join(artifact_dir, "warp_mask_00000.png"))
                Image.fromarray(depth, "L").save(
                    os.path.join(artifact_dir, "depth_00000.png"))
                mapping.astype("<f4").tofile(
                    os.path.join(artifact_dir, "warp_map_00000.f32"))
                with open(os.path.join(artifact_dir, "warp_map_shape.json"), "w",
                          encoding="utf-8") as stream:
                    json.dump(shape, stream)
                with open(os.path.join(artifact_dir, "contract.json"), "w",
                          encoding="utf-8") as stream:
                    json.dump({
                        "schema": 16,
                        "model": "depth_anything_v2_fp16",
                        "profile": "apollo",
                        "depth_step": "current-once",
                        "depth_reuse_interval": 1,
                        "depth_compensation": "none",
                        "literal_bestv2": False,
                        "cuda_graph": True,
                        "adaptive_pop": True,
                        "adaptive_pop_max": 1.3,
                        "zero_plane": "legacy",
                    }, stream)
                with open(os.path.join(artifact_dir, "sbs_perf.json"), "w",
                          encoding="utf-8") as stream:
                    json.dump({"stages": {
                        "depth_infer": {"p50_ms": 1.0},
                        "warp_infer": {"p50_ms": 1.0},
                    }}, stream)

            clip_hash = run_eval.sha1_dir(clip_dir)
            with open(os.path.join(SCRIPT_DIR, "thresholds.json"),
                      encoding="utf-8") as stream:
                thresholds = json.load(stream)
            common_meta = {
                "git_sha": "fixture",
                "git_dirty": False,
                "clip_set_sha1": {"demo": clip_hash},
                "mode": "profile",
                "suite": "core",
                "clips_root": clips_root,
                "extra_args": [],
                "conf_sha256": "conf",
                "metric_sha256": run_eval.metric_contract_sha(),
                "label_contract_sha256": run_eval.label_contract_sha(),
                "metric_runtime": run_eval.metric_runtime_provenance(),
                "executable_sha256": "exe",
                "runtime_shader_sha256": "shader",
                "engine_sha256": "engine",
                "onnx_sha256": "onnx",
                "model": "depth_anything_v2_fp16",
                "profile": "apollo",
                "eval_schema": run_eval.EVAL_SCHEMA,
                "depth_step": "current-once",
                "depth_reuse_interval": 1,
                "depth_compensation": "none",
                "literal_bestv2": False,
                "cuda_graph": True,
                "adaptive_pop": True,
                "adaptive_pop_max": 1.3,
                "zero_plane": "legacy",
                "training_labels": run_eval.training_label_manifest(thresholds),
                "run_kind": "comparison-only",
                "timestamp": "2026-07-17T00:00:00",
            }

            def result(run_name, run_dir):
                meta = dict(common_meta)
                meta["run_name"] = run_name
                artifact_hash = run_eval.scored_artifact_sha256(os.path.join(run_dir, "demo"))
                meta["scored_artifact_sha256"] = {"demo": artifact_hash}
                entry_meta = {
                    "name": "current-schema report fixture",
                    "model": "depth_anything_v2_fp16",
                    "profile": "apollo",
                    "depth_compensation": "none",
                    "literal_bestv2": False,
                    "cuda_graph": True,
                    "adaptive_pop": True,
                    "adaptive_pop_max": 1.3,
                    "zero_plane": "legacy",
                    "source_frame_count": 1,
                    "scored_artifact_sha256": artifact_hash,
                    "content_type": "synthetic",
                    "source_artifacts": (
                        "A baked source edge used to verify audit ordering."),
                }
                rows, aggregate = sbsbench.measure_sequence(
                    os.path.join(run_dir, "demo"), clip_dir)
                aggregate = sbsbench.filter_aggregate_by_evidence(
                    rows, aggregate, thresholds["metrics"], entry_meta)
                worst, clip_issues, clip_hard = run_eval.score_clip_gates(
                    rows, aggregate, thresholds, entry_meta)
                frames = run_eval.build_frame_records(rows, thresholds, entry_meta)
                perf = run_eval.load_perf_metrics(
                    os.path.join(run_dir, "demo", "sbs_perf.json"))
                evidence = run_eval.primary_evidence_failures(
                    aggregate, thresholds, "demo", entry_meta, worst=worst, rows=rows)
                evidence.extend(run_eval.perf_evidence_failures(
                    None, perf, thresholds, "demo"))
                hard = [{"clip": "demo", **item} for item in clip_hard]
                payload = {
                    "meta": meta,
                    "clips": {
                        "demo": {
                            "aggregate": aggregate,
                            "perf_ms": perf,
                            "meta": entry_meta,
                            "worst_frame": worst,
                            "frames": frames,
                            "label_summary": run_eval.summarize_frame_labels(
                                frames, thresholds),
                        },
                    },
                    "issues": [{"clip": "demo", **item} for item in clip_issues],
                    "hard_failures": hard,
                    "evidence_failures": evidence,
                    "regressions": [],
                    "verdict": ("hard_failures" if hard else
                                "evidence_failures" if evidence else "comparison_only"),
                }
                run_eval.bind_training_labels_to_evidence_gate(payload, thresholds)
                return payload

            control_payload = result("control", control_dir)
            treatment_payload = result("treatment", treatment_dir)
            run_eval.verify_results_against_artifacts(
                treatment_payload, treatment_dir, clips_root, thresholds)
            forged_labels = json.loads(json.dumps(treatment_payload))
            forged_labels["clips"]["demo"]["frames"][0]["labels"]["forged"] = True
            with self.assertRaisesRegex(ValueError, "frames"):
                run_eval.verify_results_against_artifacts(
                    forged_labels, treatment_dir, clips_root, thresholds)
            self.assertTrue(treatment_payload["hard_failures"])
            forged_hard = json.loads(json.dumps(treatment_payload))
            forged_hard["hard_failures"] = []
            with self.assertRaisesRegex(ValueError, "hard_failures"):
                run_eval.verify_results_against_artifacts(
                    forged_hard, treatment_dir, clips_root, thresholds)
            self.assertTrue(treatment_payload["issues"])
            forged_issues = json.loads(json.dumps(treatment_payload))
            forged_issues["issues"] = []
            with self.assertRaisesRegex(ValueError, "issues"):
                run_eval.verify_results_against_artifacts(
                    forged_issues, treatment_dir, clips_root, thresholds)

            # A baseline-gated result must reconstruct its decision from the authenticated
            # baseline snapshot. Cached evidence/regression lists are not authoritative merely
            # because the current candidate pixels remeasure correctly.
            baseline_dir = os.path.join(directory, "baselines")
            os.makedirs(baseline_dir)
            gated_payload = result("gated-treatment", treatment_dir)
            gated_payload["meta"]["run_kind"] = "baseline-gated"
            baseline_manifest = {
                "aggregate": control_payload["clips"]["demo"]["aggregate"],
                "perf_ms": control_payload["clips"]["demo"]["perf_ms"],
                "meta": {
                    **control_payload["meta"],
                    **control_payload["clips"]["demo"]["meta"],
                    "run_kind": "baseline-update",
                    "clip_sha1": clip_hash,
                    "extra_args": [],
                },
            }
            baseline_path = os.path.join(baseline_dir, "demo.json")
            with open(baseline_path, "w", encoding="utf-8") as stream:
                json.dump(baseline_manifest, stream)
            snapshot = run_eval.build_baseline_snapshot(
                baseline_dir, {"demo": baseline_manifest})
            with open(os.path.join(treatment_dir, run_eval.BASELINE_SNAPSHOT_FILE), "w",
                      encoding="utf-8") as stream:
                json.dump(snapshot, stream)
            gated_payload["meta"]["baseline_snapshot_sha256"] = run_eval.sha256_json(snapshot)
            gated_thresholds = json.loads(json.dumps(thresholds))
            gated_thresholds["metrics"]["exact_mapping_stretch_pct"].update({
                "role": "primary", "rel_tol": 0.0, "abs_floor": 0.0})
            gated_rows, _ = sbsbench.measure_sequence(
                os.path.join(treatment_dir, "demo"), clip_dir)
            gated_evidence, gated_regressions = run_eval.score_baseline_comparison(
                gated_payload["clips"]["demo"]["aggregate"],
                gated_payload["clips"]["demo"]["perf_ms"], gated_rows,
                gated_payload["clips"]["demo"]["worst_frame"],
                gated_payload["clips"]["demo"]["meta"], baseline_manifest,
                gated_thresholds, "demo")
            self.assertTrue(gated_regressions)
            gated_payload["evidence_failures"] = gated_evidence
            gated_payload["regressions"] = gated_regressions
            gated_payload["verdict"] = (
                "hard_failures" if gated_payload["hard_failures"] else
                "evidence_failures" if gated_evidence else
                "regressions" if gated_regressions else "pass")
            for record in gated_payload["clips"]["demo"]["frames"]:
                record["labels"] = run_eval.canonical_frame_labels(
                    record["metrics"], gated_thresholds,
                    gated_payload["clips"]["demo"]["meta"])
            gated_payload["clips"]["demo"]["label_summary"] = (
                run_eval.summarize_frame_labels(
                    gated_payload["clips"]["demo"]["frames"], gated_thresholds))
            run_eval.bind_training_labels_to_evidence_gate(gated_payload, gated_thresholds)
            run_eval.verify_results_against_artifacts(
                gated_payload, treatment_dir, clips_root, gated_thresholds)

            forged_regressions = json.loads(json.dumps(gated_payload))
            forged_regressions["regressions"] = []
            with self.assertRaisesRegex(ValueError, "regressions"):
                run_eval.verify_results_against_artifacts(
                    forged_regressions, treatment_dir, clips_root, gated_thresholds)
            forged_evidence = json.loads(json.dumps(gated_payload))
            forged_evidence["evidence_failures"] = [
                {"clip": "demo", "metric": "forged", "missing": True}]
            with self.assertRaisesRegex(ValueError, "evidence_failures"):
                run_eval.verify_results_against_artifacts(
                    forged_evidence, treatment_dir, clips_root, gated_thresholds)

            forged_snapshot = json.loads(json.dumps(snapshot))
            forged_snapshot["clips"]["demo"]["manifest"]["aggregate"] = {}
            with open(os.path.join(treatment_dir, run_eval.BASELINE_SNAPSHOT_FILE), "w",
                      encoding="utf-8") as stream:
                json.dump(forged_snapshot, stream)
            forged_snapshot_payload = json.loads(json.dumps(gated_payload))
            forged_snapshot_payload["meta"]["baseline_snapshot_sha256"] = (
                run_eval.sha256_json(forged_snapshot))
            with self.assertRaisesRegex(ValueError, "manifest digest"):
                run_eval.verify_results_against_artifacts(
                    forged_snapshot_payload, treatment_dir, clips_root, gated_thresholds)
            with open(os.path.join(treatment_dir, run_eval.BASELINE_SNAPSHOT_FILE), "w",
                      encoding="utf-8") as stream:
                json.dump(snapshot, stream)

            for run_dir, payload in (
                    (control_dir, control_payload),
                    (treatment_dir, treatment_payload)):
                with open(os.path.join(run_dir, "results.json"), "w",
                          encoding="utf-8") as stream:
                    json.dump(payload, stream)

            report_path = os.path.join(directory, "report.html")
            completed = subprocess.run(
                [sys.executable, os.path.join(SCRIPT_DIR, "build_report.py"),
                 control_dir, treatment_dir, report_path],
                text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            with open(report_path, encoding="utf-8") as stream:
                report = stream.read()

            ordered_sections = (
                "<h2>Conclusion</h2>",
                "Metric definitions and decision roles",
                "<h2>Metrics by group</h2>",
                "<h2>Original-source artifact audit</h2>",
                "<h2>Quality-axis visual evidence</h2>",
                "<h2>Reproduce</h2>",
                "Per-clip bar comparison",
            )
            positions = [report.index(section) for section in ordered_sections]
            self.assertEqual(positions, sorted(positions))
            self.assertIn("exact production-selected source", report)
            self.assertIn("visible_pop", report)
            self.assertIn("left | right", report)
            self.assertIn("Source-structure-weighted visible relief", report)
            self.assertIn("renderer-conformance diagnostics", report)
            self.assertIn("mapping_stretch", report)
            self.assertIn("delta: red worse / blue better", report)
            self.assertIn("not applicable", report)
            self.assertNotIn("9876.54", report)
            self.assertIn("render_integrity", report)
            self.assertNotIn("__CONCLUSION__", report)
            self.assertNotIn("__GROUP_RADARS__", report)
            self.assertTrue(os.path.exists(os.path.join(directory, "decision.json")))

            # A report may never trust a hand-edited aggregate, even when all artifact hashes are
            # still valid. The full metric stack must be re-run before any conclusion is shown.
            forged = result("forged-treatment", treatment_dir)
            forged_stretch = forged["clips"]["demo"]["aggregate"][
                "exact_mapping_stretch_pct"]
            forged["clips"]["demo"]["aggregate"]["exact_mapping_stretch_pct"] = (
                forged_stretch + 7.0)
            with open(os.path.join(treatment_dir, "results.json"), "w",
                      encoding="utf-8") as stream:
                json.dump(forged, stream)
            residual_report_path = os.path.join(directory, "residual-report.html")
            residual_run = subprocess.run(
                [sys.executable, os.path.join(SCRIPT_DIR, "build_report.py"),
                 control_dir, treatment_dir, residual_report_path],
                text=True, capture_output=True, check=False)
            self.assertNotEqual(residual_run.returncode, 0)
            self.assertIn("authoritative remeasurement", residual_run.stderr)
            self.assertIn("aggregate", residual_run.stderr)


if __name__ == "__main__":
    unittest.main()
