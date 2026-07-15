#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from scipy.io import savemat

import prepare_inria_3dmovie_training as inria


class InriaPreparationTests(unittest.TestCase):
    @staticmethod
    def write_jpeg(path, image):
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("cannot encode test JPEG")
        path.write_bytes(encoded.tobytes())

    def test_stereo_pairs_require_complete_numeric_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((8, 12, 3), np.uint8)
            cv2.imwrite(str(root / "00000001.jpg"), image)
            self.write_jpeg(root / "00000001.jpg.right", image)
            pairs = inria.stereo_pairs(root)
            self.assertEqual(pairs[0][0].name, "00000001.jpg")
            self.assertEqual(pairs[0][1].name, "00000001.jpg.right")

            cv2.imwrite(str(root / "00000002.jpg"), image)
            with self.assertRaisesRegex(RuntimeError, "missing right eye"):
                inria.stereo_pairs(root)

    def test_orphan_right_eye_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((8, 12, 3), np.uint8)
            cv2.imwrite(str(root / "00000001.jpg"), image)
            self.write_jpeg(root / "00000001.jpg.right", image)
            self.write_jpeg(root / "00000002.jpg.right", image)
            with self.assertRaisesRegex(RuntimeError, "orphan right eye"):
                inria.stereo_pairs(root)

    def test_reference_flow_is_converted_to_apollo_polarity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uv = np.zeros((8, 12, 2), np.float32)
            uv[..., 0] = 3.5
            source = root / "reference.mat"
            output = root / "reference.npz"
            savemat(source, {"uv": uv})
            inria.write_disparity(source, output, (12, 8))
            with np.load(output) as payload:
                np.testing.assert_allclose(payload["disparity_px"], -3.5)


if __name__ == "__main__":
    unittest.main()
