"""Equivalence tests for shared interocular exact-map registration."""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import sbs_interocular_phase_chroma as phase  # noqa: E402
import sbs_interocular_photometric_rivalry as photometric  # noqa: E402


def _scene(width=96, height=48):
    y, x = np.mgrid[:height, :width].astype(np.float32)
    x /= max(width - 1, 1)
    y /= max(height - 1, 1)
    return np.stack((
        0.12 + 0.64 * x + 0.07 * np.sin(11.0 * y),
        0.10 + 0.58 * y + 0.06 * np.cos(13.0 * x),
        0.18 + 0.42 * (1.0 - x) + 0.05 * np.sin(9.0 * (x + y)),
    ), axis=2).astype(np.float32)


def _maps(width, height):
    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    vertical = np.sin(np.linspace(0.0, np.pi, height, dtype=np.float32))[:, None]
    shift = (0.008 + 0.006 * vertical).astype(np.float32)
    base = np.broadcast_to(u[None, :], (height, width)).copy()
    return base + shift, base - shift


def _assert_maps_equal(test, direct, prepared):
    test.assertEqual(set(direct), set(prepared))
    for key in direct:
        np.testing.assert_allclose(
            direct[key], prepared[key], rtol=0.0, atol=0.0, equal_nan=True,
            err_msg=key)


class PreparedInterocularEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.source = _scene()
        height, width = self.source.shape[:2]
        self.shape = {
            "source_width": width,
            "source_height": height,
            "eye_width": width,
            "eye_height": height,
            "content_scale_x": 1.0,
            "content_scale_y": 1.0,
        }
        self.maps = _maps(width, height)

        def display_transform(image):
            return image / (1.0 + image)

        self.display_transform = display_transform
        linear_source = self.source * 2.5
        self.source = linear_source
        self.eyes = tuple(
            display_transform(phase._sample_source_eye(linear_source, mapping, self.shape))
            for mapping in self.maps)
        right = self.eyes[1].copy()
        right[14:31, 37:59] *= np.asarray((1.18, 0.87, 1.05), dtype=np.float32)
        self.eyes = self.eyes[0], right
        self.mask = np.zeros((height, 2 * width, 3), dtype=np.float32)
        self.mask[3:9, width + 8:width + 19, 0] = 1.0

    def _prepare(self):
        return phase.prepare_interocular_evidence(
            self.source, *self.eyes, *self.maps, self.shape,
            warp_mask=self.mask, source_sample_transform=self.display_transform,
            max_analysis_width=160, max_analysis_height=90)

    def test_one_preparation_matches_both_standalone_apis_exactly(self):
        shared = self._prepare()
        phase_direct = phase.measure_interocular_phase_chroma(
            self.source, *self.eyes, *self.maps, self.shape,
            warp_mask=self.mask, source_sample_transform=self.display_transform,
            max_analysis_width=160, max_analysis_height=90,
            min_phase_pixels=16, return_maps=True)
        phase_shared = phase.measure_interocular_phase_chroma_prepared(
            shared, min_phase_pixels=16, return_maps=True)
        self.assertEqual(phase_direct[0], phase_shared[0])
        _assert_maps_equal(self, phase_direct[1], phase_shared[1])

        photo_direct = photometric.measure_interocular_photometric_rivalry(
            self.source, *self.eyes, *self.maps, self.shape,
            warp_mask=self.mask, source_sample_transform=self.display_transform,
            max_analysis_width=160, max_analysis_height=90,
            min_pixels=16, return_maps=True)
        photo_shared = photometric.measure_interocular_photometric_rivalry_prepared(
            shared, min_pixels=16, return_maps=True)
        self.assertEqual(photo_direct[0], photo_shared[0])
        _assert_maps_equal(self, photo_direct[1], photo_shared[1])

    def test_prepared_consumers_do_not_repeat_registration(self):
        with mock.patch.object(
                phase, "_registered_evidence", wraps=phase._registered_evidence) as register:
            shared = self._prepare()
            self.assertEqual(register.call_count, 2)
            phase.measure_interocular_phase_chroma_prepared(
                shared, min_phase_pixels=16)
            photometric.measure_interocular_photometric_rivalry_prepared(
                shared, min_pixels=16)
            self.assertEqual(register.call_count, 2)

    def test_prepared_entry_points_reject_untrusted_objects(self):
        with self.assertRaisesRegex(ValueError, "invalid type"):
            phase.measure_interocular_phase_chroma_prepared({})
        with self.assertRaisesRegex(ValueError, "invalid type"):
            photometric.measure_interocular_photometric_rivalry_prepared({})


if __name__ == "__main__":
    unittest.main()
