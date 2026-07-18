"""Contract tests for the authenticated-real-source stereo-window validator."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import validate_real_stereo_window_metric as validator  # noqa: E402


def fixture_image(width=256, height=144):
    x = (np.arange(width, dtype=np.float32) + 0.5) / width
    y = (np.arange(height, dtype=np.float32) + 0.5) / height
    xx, yy = np.meshgrid(x, y)
    image = np.stack((
        0.5 + 0.35 * np.sin(2.0 * np.pi * 14.0 * xx),
        0.5 + 0.30 * np.sin(2.0 * np.pi * (7.0 * xx + 2.0 * yy)),
        0.5 + 0.25 * np.cos(2.0 * np.pi * 11.0 * yy),
    ), axis=-1)
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


class RealStereoWindowValidationTests(unittest.TestCase):
    def test_report_records_authenticated_samples_and_explicit_counts(self):
        with tempfile.TemporaryDirectory() as root:
            clip = os.path.join(root, "fixture")
            os.makedirs(clip)
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as stream:
                json.dump({"name": "validator fixture"}, stream)
            frame = os.path.join(clip, "frame_00001.png")
            Image.fromarray(fixture_image(), mode="RGB").save(frame)

            report = validator.build_report(
                [root], frames_per_clip=1, max_width=256, workers=1)
            summary = report["summary"]
            self.assertEqual(report["schema"], validator.SCHEMA)
            self.assertEqual(report["training_label_qualification"], "blocked")
            self.assertFalse(report["auto_promotes_labels"])
            self.assertEqual(summary["clips"], 1)
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(summary["checks"], 13)
            self.assertEqual(
                summary["checks"],
                summary["passed"] + summary["failed"] + summary["abstained"])
            sample = report["samples"][0]
            self.assertEqual(sample["clip"], "fixture")
            self.assertEqual(len(sample["source_sha256"]), 64)
            self.assertIn("graded_crossed_perceptual_burden", report["checks_by_name"])
            self.assertIn("source_derived_frequency", report["checks_by_name"])

    def test_missing_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            clip = os.path.join(root, "fixture")
            os.makedirs(clip)
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as stream:
                json.dump({}, stream)
            Image.fromarray(fixture_image(), mode="RGB").save(
                os.path.join(clip, "frame_00001.png"))
            with self.assertRaisesRegex(ValueError, "unauthenticated"):
                validator.build_report([root], workers=1)


if __name__ == "__main__":
    unittest.main()
