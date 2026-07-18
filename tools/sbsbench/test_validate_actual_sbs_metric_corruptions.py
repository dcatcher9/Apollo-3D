"""Contract tests for the actual-harness controlled-corruption validator."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import validate_actual_sbs_metric_corruptions as validator  # noqa: E402


class ActualSbsMetricCorruptionValidatorTests(unittest.TestCase):
    @staticmethod
    def _textured_fixture(height=72, width=128):
        y, x = np.mgrid[:height, :width].astype(np.float32)
        image = (0.5 + 0.20 * np.sin(x * 0.37) + 0.16 * np.sin(y * 0.29)
                 + 0.08 * np.sin((x + y) * 0.71))
        return np.clip(image, 0.0, 1.0).astype(np.float32)

    def test_shared_integrity_map_localizes_blur_and_ringing(self):
        expected = self._textured_fixture()
        valid = np.ones(expected.shape, dtype=bool)
        patch = np.zeros(expected.shape, dtype=bool)
        patch[20:52, 40:88] = True
        smooth = validator.sbsbench._local_mean(expected, 4)
        blurred = expected.copy()
        blurred[patch] = smooth[patch]
        ringing = expected.copy()
        sharpened = np.clip(
            expected + 3.0 * (expected - validator.sbsbench._local_mean(expected, 2)),
            0.0, 1.0)
        ringing[patch] = sharpened[patch]

        clean_bad, support = validator.sbsbench.exact_image_integrity_maps(
            expected, expected, valid)
        blur_bad, blur_support = validator.sbsbench.exact_image_integrity_maps(
            blurred, expected, valid)
        ringing_bad, ringing_support = validator.sbsbench.exact_image_integrity_maps(
            ringing, expected, valid)

        self.assertFalse(clean_bad.any())
        self.assertTrue(np.array_equal(support, blur_support))
        self.assertTrue(np.array_equal(support, ringing_support))
        target = patch & support
        self.assertGreater(float(np.mean(blur_bad[target])), 0.80)
        self.assertGreater(float(np.mean(ringing_bad[target])), 0.80)
        halo = validator.sbsbench.dilate2d(patch, 3)
        self.assertGreater(float(np.mean(halo[blur_bad])), 0.95)
        self.assertGreater(float(np.mean(halo[ringing_bad])), 0.95)

    def test_source_localization_calls_evaluator_integrity_map(self):
        height, width = 48, 96
        source_luma = self._textured_fixture(height, width)
        source_rgb = np.repeat(source_luma[..., None], 3, axis=-1)
        mapping = np.broadcast_to(
            (np.arange(width, dtype=np.float32) + 0.5) / width,
            (height, width)).copy()
        shape = {
            "source_width": width,
            "source_height": height,
            "content_scale_x": 1.0,
            "content_scale_y": 1.0,
        }
        patch = np.zeros((height, width), dtype=bool)
        patch[12:36, 30:66] = True
        blurred = source_rgb.copy()
        smooth = np.stack([
            validator.sbsbench._local_mean(source_rgb[..., channel], 4)
            for channel in range(3)
        ], axis=2)
        blurred[patch] = smooth[patch]
        sample = {
            "source_rgb": source_rgb,
            "right": source_rgb,
            "map_right": mapping,
            "shape": shape,
        }

        with mock.patch.object(
                validator.sbsbench, "exact_image_integrity_maps",
                wraps=validator.sbsbench.exact_image_integrity_maps) as shared_detector:
            evidence = validator._source_localization(
                sample, blurred, patch, "integrity")

        self.assertEqual(shared_detector.call_count, 2)
        self.assertGreater(evidence["active_pixels"], 0)

    def test_footprint_ladder_is_nested_and_close_to_requested_area(self):
        height, width = 180, 320
        x = np.linspace(0.0, 8.0 * np.pi, width, dtype=np.float32)
        expected = np.broadcast_to(0.5 + 0.4 * np.sin(x)[None, :], (height, width)).copy()
        masks = validator._nested_fault_masks(expected, np.ones(expected.shape, dtype=bool))

        self.assertEqual(len(masks), len(validator.FOOTPRINT_FRACTIONS))
        for requested, mask in zip(validator.FOOTPRINT_FRACTIONS, masks):
            self.assertLess(abs(float(mask.mean()) - requested), max(0.0002, requested * 0.12))
        for before, after in zip(masks, masks[1:]):
            self.assertFalse(np.any(before & ~after))

    def test_ladder_check_fails_nonmonotonic_or_missing_response(self):
        passed = validator._ladder_check(
            "good", 100.0, [99.9, 99.5, 99.0, 98.0, 95.0],
            direction="lower", min_final=3.0, tolerance=0.05)
        reversed_response = validator._ladder_check(
            "bad", 100.0, [99.9, 99.5, 99.7, 98.0, 95.0],
            direction="lower", min_final=3.0, tolerance=0.05)
        absent = validator._ladder_check(
            "absent", 100.0, [99.9, None, 99.0, 98.0, 95.0],
            direction="lower", min_final=3.0, tolerance=0.05)

        self.assertEqual(passed["status"], "pass")
        self.assertEqual(reversed_response["status"], "fail")
        self.assertEqual(absent["status"], "fail")

    def test_phase_metric_contract_rejects_retired_percentile_names(self):
        fake = {
            validator.PHASE: 0.0,
            "interocular_phase_orientation_p95_pct": 2.0,
            "interocular_chroma_conflict_p95_pct": 1.0,
            validator.PHASE_OK: 100.0,
        }
        sample = {
            "source_rgb": np.zeros((8, 16, 3), dtype=np.float32),
            "left": np.zeros((8, 16, 3), dtype=np.float32),
            "map_left": np.zeros((8, 16), dtype=np.float32),
            "map_right": np.zeros((8, 16), dtype=np.float32),
            "warp_mask": np.zeros((8, 32, 3), dtype=np.float32),
            "shape": {},
        }
        with mock.patch.object(
                validator.phase_chroma, "measure_interocular_phase_chroma", return_value=fake):
            with self.assertRaisesRegex(ValueError, "metric-name contract drift"):
                validator._phase_metrics(sample, sample["left"])

    def test_report_can_never_promote_training_labels(self):
        with tempfile.TemporaryDirectory() as root:
            clips_root = os.path.join(root, "clips")
            os.makedirs(clips_root)
            for clip in ("c647", "c339"):
                os.makedirs(os.path.join(clips_root, clip))
            results = {
                "meta": {
                    "eval_schema": validator.EXPECTED_EVAL_SCHEMA,
                    "suite": "core",
                    "clips_root": clips_root,
                },
                "clips": {"c647": {}, "c339": {}},
            }
            with open(os.path.join(root, "results.json"), "w", encoding="utf-8") as stream:
                json.dump(results, stream)

            def validated(sample):
                return {
                    "clip_id": sample["clip_id"],
                    "checks": [{"name": "fixture", "status": "pass"}],
                    "passed": True,
                }

            with mock.patch.object(
                    validator, "_authenticate_sample",
                    side_effect=lambda *args: {"clip_id": args[2]}), mock.patch.object(
                        validator, "validate_sample", side_effect=validated):
                report = validator.build_report(root, max_clips=2)

        self.assertEqual(report["training_label_qualification"], "blocked")
        self.assertEqual(report["eligible_training_labels"], [])
        self.assertFalse(report["auto_promotes_thresholds"])
        self.assertTrue(report["summary"]["overall_pass"])

    def test_wrong_eval_schema_fails_before_artifact_discovery(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "results.json"), "w", encoding="utf-8") as stream:
                json.dump({"meta": {"eval_schema": 31}}, stream)
            with self.assertRaisesRegex(ValueError, "eval schema"):
                validator.build_report(root)


if __name__ == "__main__":
    unittest.main()
