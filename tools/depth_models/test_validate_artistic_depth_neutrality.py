#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import validate_artistic_depth_neutrality as neutrality


class ProductionNormalizationTests(unittest.TestCase):
    def test_prepare_image_respects_native_low_resolution_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, shape, expected in (
                    ("landscape", (180, 320), (1, 3, 168, 294)),
                    ("portrait", (320, 180), (1, 3, 294, 168))):
                with self.subTest(name=name):
                    path = Path(directory) / f"{name}.png"
                    cv2.imwrite(str(path), np.zeros((*shape, 3), np.uint8))
                    image, width, height, source_width, source_height = (
                        neutrality.prepare_image(path)
                    )
                    self.assertEqual(tuple(image.shape), expected)
                    self.assertEqual((width, height), (expected[3], expected[2]))
                    self.assertEqual(
                        (source_width, source_height), (shape[1], shape[0])
                    )

    def test_flat_depth_maps_to_zero(self):
        depth = np.full((1, 4, 5), 3.0, dtype=np.float32)
        actual = neutrality.production_percentile_normalize(depth)
        np.testing.assert_array_equal(actual, np.zeros_like(depth))

    def test_outliers_saturate_outside_percentile_bounds(self):
        depth = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
        actual = neutrality.production_percentile_normalize(depth)
        self.assertEqual(float(actual[0]), 0.0)
        self.assertEqual(float(actual[-1]), 1.0)
        self.assertGreater(float(actual[500]), 0.49)
        self.assertLess(float(actual[500]), 0.51)

    def test_affine_equivalent_depth_is_neutral(self):
        depth = np.linspace(2.0, 9.0, 2048, dtype=np.float32)
        reference = neutrality.production_percentile_normalize(depth)
        candidate = neutrality.production_percentile_normalize(depth * 3.5 + 11.0)
        np.testing.assert_allclose(reference, candidate, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
