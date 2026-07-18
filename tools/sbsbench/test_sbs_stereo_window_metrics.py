"""Focused synthetic qualification for the experimental stereo-window metric."""

import os
import sys
import unittest

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import sbs_stereo_window_metrics as window_metrics  # noqa: E402


def mapping_shape(eye_width, eye_height, *, source_width=None, source_height=None,
                  scale_x=1.0, scale_y=1.0):
    return {
        "width": 2 * eye_width,
        "height": eye_height,
        "eye_width": eye_width,
        "eye_height": eye_height,
        "source_width": source_width or eye_width,
        "source_height": source_height or eye_height,
        "content_scale_x": scale_x,
        "content_scale_y": scale_y,
    }


def identity_eye(shape):
    width, height = shape["eye_width"], shape["eye_height"]
    output_u = (np.arange(width, dtype=np.float32) + 0.5) / width
    lo_x = 0.5 * (1.0 - shape["content_scale_x"])
    source_u = (output_u - lo_x) / shape["content_scale_x"]
    return np.broadcast_to(source_u, (height, width)).copy()


def signed_maps(shape, disparity_pct, *, y0=0.0, y1=1.0, x0=0.0, x1=1.0):
    """Construct inverse maps with requested actual disparity xR-xL."""
    base = identity_eye(shape)
    left, right = base.copy(), base.copy()
    height, width = base.shape
    region = np.zeros(base.shape, dtype=bool)
    region[int(round(y0 * height)):int(round(y1 * height)),
           int(round(x0 * width)):int(round(x1 * width))] = True
    # Forward shifts are xL=-d/2 and xR=+d/2.  Inverse maps have the opposite shift.
    left[region] += disparity_pct / 200.0
    right[region] -= disparity_pct / 200.0
    return np.concatenate((left, right), axis=1)


def stripe_source(width, height, *, amplitude=0.40, frequency=14.0,
                  orientation="vertical"):
    x = (np.arange(width, dtype=np.float32) + 0.5) / width
    y = (np.arange(height, dtype=np.float32) + 0.5) / height
    if orientation == "vertical":
        pattern = np.sin(2.0 * np.pi * frequency * x)[None, :]
        pattern = np.broadcast_to(pattern, (height, width))
    elif orientation == "horizontal":
        pattern = np.sin(2.0 * np.pi * frequency * y)[:, None]
        pattern = np.broadcast_to(pattern, (height, width))
    else:
        raise ValueError(orientation)
    return np.clip(0.5 + amplitude * pattern, 0.0, 1.0).astype(np.float32)


def measure(mapping, shape, source, **kwargs):
    return window_metrics.measure_stereo_window_violation(
        mapping, shape, source, **kwargs)


class StereoWindowMetricTests(unittest.TestCase):
    def setUp(self):
        self.shape = mapping_shape(640, 360)
        self.source = stripe_source(640, 360)

    def test_identity_has_no_signed_window_contact(self):
        metrics = measure(signed_maps(self.shape, 0.0), self.shape, self.source)
        self.assertGreater(metrics["experimental_stereo_window_support_pct"], 99.0)
        self.assertEqual(metrics["experimental_stereo_window_crossed_burden_pct"], 0.0)
        self.assertEqual(metrics["experimental_stereo_window_crossed_area_pct"], 0.0)
        self.assertEqual(metrics["experimental_stereo_window_uncrossed_burden_pct"], 0.0)

    def test_crossed_foreground_and_uncrossed_contact_are_separate(self):
        crossed = measure(signed_maps(self.shape, -2.0), self.shape, self.source)
        uncrossed = measure(signed_maps(self.shape, 2.0), self.shape, self.source)

        self.assertGreater(crossed["experimental_stereo_window_crossed_burden_pct"], 0.01)
        self.assertGreater(crossed["experimental_stereo_window_crossed_area_pct"], 0.1)
        self.assertGreater(
            crossed["experimental_stereo_window_crossed_largest_component_pct"], 0.05)
        self.assertEqual(crossed["experimental_stereo_window_uncrossed_area_pct"], 0.0)
        self.assertGreater(uncrossed["experimental_stereo_window_uncrossed_burden_pct"], 0.01)
        self.assertEqual(uncrossed["experimental_stereo_window_crossed_area_pct"], 0.0)

    def test_central_pop_is_not_a_window_violation(self):
        mapping = signed_maps(self.shape, -2.0, x0=0.30, x1=0.70)
        metrics = measure(mapping, self.shape, self.source)
        self.assertEqual(metrics["experimental_stereo_window_crossed_burden_pct"], 0.0)
        self.assertEqual(metrics["experimental_stereo_window_crossed_area_pct"], 0.0)
        self.assertEqual(
            metrics["experimental_stereo_window_crossed_largest_component_pct"], 0.0)

    def test_perceptual_burden_responds_to_contrast_not_geometric_area(self):
        mapping = signed_maps(self.shape, -2.0)
        high = measure(mapping, self.shape, stripe_source(640, 360, amplitude=0.42))
        low = measure(mapping, self.shape, stripe_source(640, 360, amplitude=0.02))
        high_burden = high["experimental_stereo_window_crossed_burden_pct"]
        low_burden = low["experimental_stereo_window_crossed_burden_pct"]
        self.assertGreater(high_burden, low_burden * 3.0)
        self.assertAlmostEqual(
            high["experimental_stereo_window_crossed_area_pct"],
            low["experimental_stereo_window_crossed_area_pct"], places=6)

    def test_horizontal_orientation_is_downweighted(self):
        mapping = signed_maps(self.shape, -2.0)
        vertical = measure(
            mapping, self.shape, stripe_source(640, 360, orientation="vertical"))
        horizontal = measure(
            mapping, self.shape, stripe_source(640, 360, orientation="horizontal"))
        self.assertGreater(
            vertical["experimental_stereo_window_crossed_burden_pct"],
            horizontal["experimental_stereo_window_crossed_burden_pct"] * 1.10)

    def test_spatial_frequency_changes_perceptual_burden(self):
        mapping = signed_maps(self.shape, -2.0)
        very_low = measure(
            mapping, self.shape, stripe_source(640, 360, frequency=1.0))
        mid = measure(
            mapping, self.shape, stripe_source(640, 360, frequency=48.0))
        very_high = measure(
            mapping, self.shape, stripe_source(640, 360, frequency=96.0))
        low_value = very_low["experimental_stereo_window_crossed_burden_pct"]
        mid_value = mid["experimental_stereo_window_crossed_burden_pct"]
        high_value = very_high["experimental_stereo_window_crossed_burden_pct"]
        self.assertGreater(mid_value, low_value * 10.0)
        self.assertGreater(mid_value, high_value * 1.10)

    def test_longer_border_connected_subject_increases_component_and_area(self):
        short = measure(
            signed_maps(self.shape, -2.0, y0=0.42, y1=0.58), self.shape, self.source)
        tall = measure(
            signed_maps(self.shape, -2.0, y0=0.20, y1=0.80), self.shape, self.source)
        self.assertGreater(
            tall["experimental_stereo_window_crossed_area_pct"],
            short["experimental_stereo_window_crossed_area_pct"] * 2.5)
        self.assertGreater(
            tall["experimental_stereo_window_crossed_largest_component_pct"],
            short["experimental_stereo_window_crossed_largest_component_pct"] * 2.5)

    def test_proportional_resolution_change_is_stable(self):
        low_shape = mapping_shape(640, 360)
        high_shape = mapping_shape(1280, 720)
        low = measure(
            signed_maps(low_shape, -2.0), low_shape, stripe_source(640, 360))
        high = measure(
            signed_maps(high_shape, -2.0), high_shape, stripe_source(1280, 720))
        for suffix in ("crossed_burden_pct", "crossed_area_pct",
                       "crossed_largest_component_pct"):
            low_value = low["experimental_stereo_window_" + suffix]
            high_value = high["experimental_stereo_window_" + suffix]
            relative = abs(low_value - high_value) / max(0.5 * (low_value + high_value), 1e-6)
            self.assertLess(relative, 0.12, (suffix, low_value, high_value))

    def test_letterbox_rows_do_not_change_the_measurement(self):
        full_shape = mapping_shape(640, 360, source_width=640, source_height=360)
        boxed_shape = mapping_shape(
            640, 480, source_width=640, source_height=360, scale_y=0.75)
        full = measure(
            signed_maps(full_shape, -2.0), full_shape, stripe_source(640, 360))
        boxed = measure(
            signed_maps(boxed_shape, -2.0), boxed_shape, stripe_source(640, 360))
        # Area topology is unchanged, while the disparity-weighted burden follows the shared
        # full-eye/reference-aspect normalization: the same pixel shift occupies less of a taller
        # requested eye even when the extra rows are bars.
        for suffix in ("crossed_area_pct", "crossed_largest_component_pct"):
            self.assertAlmostEqual(
                full["experimental_stereo_window_" + suffix],
                boxed["experimental_stereo_window_" + suffix], delta=0.02)
        self.assertAlmostEqual(
            boxed["experimental_stereo_window_crossed_burden_pct"] /
            full["experimental_stereo_window_crossed_burden_pct"],
            360.0 / 480.0, delta=0.03)

    def test_invalid_forward_coverage_cannot_vote(self):
        mapping = signed_maps(self.shape, -2.0)
        valid = np.ones(mapping.shape, dtype=bool)
        # Remove both physical side regions from both eyes.  Plenty of central support remains,
        # but no exact border-connected component is qualified.
        edge = int(round(0.10 * self.shape["eye_width"]))
        for eye_index in range(2):
            offset = eye_index * self.shape["eye_width"]
            valid[:, offset:offset + edge] = False
            valid[:, offset + self.shape["eye_width"] - edge:
                  offset + self.shape["eye_width"]] = False
        metrics = measure(mapping, self.shape, self.source, coverage_mask=valid)
        self.assertGreater(metrics["experimental_stereo_window_support_count"], 128)
        self.assertLess(metrics["experimental_stereo_window_support_pct"], 90.0)
        self.assertEqual(metrics["experimental_stereo_window_crossed_area_pct"], 0.0)

    def test_clamps_and_folds_reduce_support_but_cannot_manufacture_risk(self):
        identity = signed_maps(self.shape, 0.0)
        clamped = identity.copy()
        width = self.shape["eye_width"]
        clamped[:, :30] = -0.2
        clamped[:, width - 30:width] = 1.2
        clamped[:, width:width + 30] = -0.2
        clamped[:, 2 * width - 30:] = 1.2
        clamp_metrics = measure(clamped, self.shape, self.source)

        folded = identity.copy()
        folded[:, :width // 4] = folded[:, :width // 4][:, ::-1]
        fold_metrics = measure(folded, self.shape, self.source)
        for metrics in (clamp_metrics, fold_metrics):
            self.assertLess(metrics["experimental_stereo_window_support_pct"], 99.0)
            self.assertEqual(metrics["experimental_stereo_window_crossed_area_pct"], 0.0)
            self.assertEqual(metrics["experimental_stereo_window_uncrossed_area_pct"], 0.0)

    def test_returned_maps_localize_left_and_right_cut_bands(self):
        metrics, maps = measure(
            signed_maps(self.shape, -2.0), self.shape, self.source, return_maps=True)
        expected_shape = maps["support"].shape
        self.assertEqual(expected_shape, (288, 512))
        for key in ("disparity_pct", "perceptual_weight", "crossed_risk",
                    "crossed_left_cut", "crossed_right_cut", "crossed_contribution"):
            self.assertEqual(maps[key].shape, expected_shape)
        self.assertTrue(maps["crossed_left_cut"].any())
        self.assertTrue(maps["crossed_right_cut"].any())
        self.assertFalse(np.any(maps["crossed_left_cut"] & maps["crossed_right_cut"]))
        self.assertGreater(metrics["experimental_stereo_window_crossed_burden_pct"], 0.0)

    def test_fractional_hdr_sample_precedes_nonlinear_transform(self):
        width, height = 513, 257
        shape = mapping_shape(width, height)
        mapping = signed_maps(shape, -2.0)
        linear = np.empty((height, width, 3), dtype=np.float32)
        linear[:, 0::2] = 0.04
        linear[:, 1::2] = 8.0

        def tonemap(rgb):
            rgb = np.asarray(rgb, dtype=np.float32)
            return rgb / (1.0 + rgb)

        _, maps = measure(
            mapping, shape, linear, source_sample_transform=tonemap, return_maps=True)
        sampled = window_metrics._sample_grid(linear, 512, 256)
        correct = tonemap(sampled)[..., 0]
        wrong = window_metrics._sample_grid(tonemap(linear), 512, 256)[..., 0]
        self.assertLess(float(np.max(np.abs(maps["source_luma"] - correct))), 1e-6)
        self.assertGreater(float(np.mean(np.abs(maps["source_luma"] - wrong))), 0.02)

    def test_insufficient_exact_support_abstains(self):
        mapping = signed_maps(self.shape, -2.0)
        valid = np.zeros(mapping.shape, dtype=bool)
        valid[:, 300:340] = True
        valid[:, 640 + 300:640 + 340] = True
        metrics = measure(
            mapping, self.shape, self.source, coverage_mask=valid, min_support_count=10000)
        self.assertLess(metrics["experimental_stereo_window_support_count"], 10000)
        self.assertIsNone(metrics["experimental_stereo_window_crossed_burden_pct"])
        self.assertIsNone(metrics["experimental_stereo_window_uncrossed_area_pct"])

    def test_bad_inputs_fail_closed(self):
        mapping = signed_maps(self.shape, -2.0)
        with self.assertRaises(ValueError):
            measure(mapping[:, :-1], self.shape, self.source)
        with self.assertRaises(ValueError):
            measure(mapping, self.shape, self.source[:, :-1])
        with self.assertRaises(ValueError):
            measure(mapping, self.shape, self.source,
                    coverage_mask=np.ones((10, 10), dtype=bool))
        rgb_source = np.repeat(self.source[..., None], 3, axis=2)
        with self.assertRaises(ValueError):
            measure(mapping, self.shape, rgb_source,
                    source_sample_transform=lambda value: value[..., 0])
        with self.assertRaises(ValueError):
            measure(mapping, self.shape, self.source,
                    source_sample_transform=lambda value: value)


if __name__ == "__main__":
    unittest.main()
