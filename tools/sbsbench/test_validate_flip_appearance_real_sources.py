import importlib.util
import unittest

import numpy as np

try:
    import validate_flip_appearance_real_sources as validator
except ImportError:
    from . import validate_flip_appearance_real_sources as validator


@unittest.skipUnless(
    importlib.util.find_spec("flip_evaluator") is not None,
    "official optional flip-evaluator package is not installed",
)
class RealSourceFlipValidatorTests(unittest.TestCase):
    def test_controlled_corruption_contract_passes_on_photo_like_source(self):
        height, width = 192, 320
        rows, columns = np.indices((height, width))
        source = np.stack((
            0.15 + 0.70 * columns / width,
            0.10 + 0.75 * rows / height,
            0.45 + 0.25 * np.sin(columns / 7.0) * np.cos(rows / 11.0),
        ), axis=2).astype(np.float32)
        source[35:160, 55:145] = (0.82, 0.18, 0.12)
        source[60:145, 80:120] = (0.10, 0.75, 0.35)
        source = np.clip(source, 0.0, 1.0)
        result = validator.validate_source(source, {
            "clip": "unit",
            "frame": "synthetic-photo-like",
            "sha256": "unit",
        })
        self.assertEqual("pass", result["status"])
        self.assertTrue(all(check["status"] == "pass" for check in result["checks"]))
        self.assertFalse(
            result["scenarios"]["clean_exact_geometry"]["training_label_eligible"])


if __name__ == "__main__":
    unittest.main()
