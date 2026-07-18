"""Qualification tests for exact-map interocular photometric rivalry metrics."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import sbs_interocular_phase_chroma as exact  # noqa: E402
import sbs_interocular_photometric_rivalry as rivalry  # noqa: E402


EXPOSURE = "interocular_exposure_rivalry_burden_pct"
COLOR = "interocular_color_gain_rivalry_burden_pct"


def _scene(width=256, height=144):
    y, x = np.mgrid[:height, :width].astype(np.float32)
    x /= max(width - 1, 1)
    y /= max(height - 1, 1)
    checker = ((np.floor(x * 12) + np.floor(y * 8)) % 2) * 0.08
    return np.stack((
        0.08 + 0.72 * x + checker,
        0.10 + 0.62 * y + 0.5 * checker,
        0.12 + 0.46 * (1.0 - x) + 0.25 * checker,
    ), axis=2).astype(np.float32)


def _shape(width, height):
    return {
        "source_width": width,
        "source_height": height,
        "eye_width": width,
        "eye_height": height,
        "content_scale_x": 1.0,
        "content_scale_y": 1.0,
    }


def _maps(width, height, displacement=0.025):
    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    left = np.broadcast_to(u[None, :] + displacement, (height, width)).copy()
    right = np.broadcast_to(u[None, :] - displacement, (height, width)).copy()
    return left, right


def _render(source, maps, shape):
    return tuple(exact._sample_source_eye(source, mapping, shape) for mapping in maps)


def _linear_exposure(image, gain):
    linear = rivalry._linearize_srgb(image) * float(gain)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _measure(source, eyes, maps, shape, **kwargs):
    return rivalry.measure_interocular_photometric_rivalry(
        source, eyes[0], eyes[1], maps[0], maps[1], shape,
        min_pixels=32, **kwargs)


class PhotometricRivalryMetricTests(unittest.TestCase):
    def setUp(self):
        self.source = _scene()
        self.shape = _shape(self.source.shape[1], self.source.shape[0])
        self.maps = _maps(self.source.shape[1], self.source.shape[0])
        self.eyes = _render(self.source, self.maps, self.shape)

    def test_clean_exact_render_is_zero(self):
        result = _measure(self.source, self.eyes, self.maps, self.shape)
        self.assertEqual(result["interocular_exposure_rivalry_evidence_sufficient"], 100.0)
        self.assertEqual(result["interocular_color_gain_rivalry_evidence_sufficient"], 100.0)
        self.assertLess(result[EXPOSURE], 1e-4)
        self.assertLess(result[COLOR], 1e-4)

    def test_unilateral_global_exposure_is_detected(self):
        eyes = (self.eyes[0], _linear_exposure(self.eyes[1], 1.20))
        result = _measure(self.source, eyes, self.maps, self.shape)
        self.assertGreater(result[EXPOSURE], 14.0)
        self.assertLess(result[COLOR], 0.25)

    def test_unilateral_rgb_gain_is_detected_as_colour(self):
        gain = np.asarray((1.30, 0.90, 0.82), dtype=np.float32)
        eyes = (self.eyes[0], self.eyes[1] * gain)
        result = _measure(self.source, eyes, self.maps, self.shape)
        self.assertGreater(result[COLOR], 20.0)
        self.assertGreater(result[EXPOSURE], 1.0)

    def test_shared_exposure_and_rgb_gain_cancel(self):
        gain = np.asarray((1.16, 0.94, 0.86), dtype=np.float32)
        eyes = tuple(eye * gain for eye in self.eyes)
        result = _measure(self.source, eyes, self.maps, self.shape)
        self.assertLess(result[EXPOSURE], 0.15)
        self.assertLess(result[COLOR], 0.15)

    def test_shared_colour_matrix_cancels(self):
        matrix = np.asarray((
            (0.88, 0.10, 0.02),
            (0.04, 0.91, 0.05),
            (0.08, 0.08, 0.84),
        ), dtype=np.float32)
        eyes = tuple(np.maximum(eye @ matrix.T, 0.0) for eye in self.eyes)
        result = _measure(self.source, eyes, self.maps, self.shape)
        self.assertLess(result[EXPOSURE], 0.25)
        self.assertLess(result[COLOR], 0.25)

    def test_localized_exposure_and_hue_are_detected(self):
        right = self.eyes[1].copy()
        patch = (slice(42, 82), slice(92, 156))
        right[patch] *= 1.35
        exposure = _measure(
            self.source, (self.eyes[0], right), self.maps, self.shape)
        self.assertGreater(exposure[EXPOSURE], 4.0)

        right = self.eyes[1].copy()
        local = right[patch].copy()
        right[patch] = local[..., (1, 2, 0)]
        hue = _measure(self.source, (self.eyes[0], right), self.maps, self.shape)
        self.assertGreater(hue[COLOR], 10.0)

    def test_forward_hole_mask_excludes_unauthenticated_patch(self):
        right = self.eyes[1].copy()
        patch = (slice(42, 82), slice(92, 156))
        right[patch] *= np.asarray((1.8, 0.3, 1.5), dtype=np.float32)
        masks = [np.zeros(self.source.shape[:2], dtype=np.float32) for _ in range(2)]
        masks[1][patch] = 1.0
        result = _measure(
            self.source, (self.eyes[0], right), self.maps, self.shape,
            warp_mask=tuple(masks))
        self.assertLess(result[EXPOSURE], 0.1)
        self.assertLess(result[COLOR], 0.1)

    def test_return_maps_localizes_one_eye_corruption(self):
        right = self.eyes[1].copy()
        patch = (slice(42, 82), slice(92, 156))
        right[patch] *= 1.35
        metrics, evidence = _measure(
            self.source, (self.eyes[0], right), self.maps, self.shape,
            return_maps=True)
        self.assertGreater(metrics[EXPOSURE], 4.0)
        values = evidence["exposure_rivalry_pct"]
        active = np.isfinite(values) & (values > 5.0)
        self.assertGreater(np.count_nonzero(active), 0)

    def test_dark_frame_abstains(self):
        source = np.zeros_like(self.source)
        eyes = _render(source, self.maps, self.shape)
        result = _measure(source, eyes, self.maps, self.shape)
        self.assertIsNone(result[EXPOSURE])
        self.assertIsNone(result[COLOR])
        self.assertEqual(result["interocular_exposure_rivalry_evidence_sufficient"], 0.0)

    def test_invalid_packed_mask_fails_closed(self):
        bad = np.zeros((10, 10), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "packed warp_mask"):
            _measure(
                self.source, self.eyes, self.maps, self.shape, warp_mask=bad)


if __name__ == "__main__":
    unittest.main()
