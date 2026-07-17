#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_simple_artistic_controller as simple  # noqa: E402


def condition_target_fields(scale):
    variant = simple.input_color.sdr_input_variant()
    return {
        "condition_target_contract": simple.label_merge.CONDITION_TARGET_CONTRACT,
        "input_condition_targets": [{
            "schema": simple.label_merge.CONDITION_TARGET_SCHEMA,
            "contract": simple.label_merge.CONDITION_TARGET_CONTRACT,
            "input_variant": variant,
            "input_variant_sha256": simple.input_color.input_variant_sha256(
                variant
            ),
            "safe_scale_ceiling": scale,
        }],
    }


class SimpleArtisticControllerTests(unittest.TestCase):
    def test_tree_finds_deterministic_separation(self):
        features = np.asarray([
            [0.0, 0.0], [0.1, 1.0], [0.2, 0.0], [0.3, 1.0],
            [0.7, 0.0], [0.8, 1.0], [0.9, 0.0], [1.0, 1.0],
        ])
        targets = np.asarray([False] * 4 + [True] * 4)
        tree = simple.fit_tree(features, targets, max_depth=1, min_leaf=2)
        probabilities = simple.predict_probabilities(tree, features)
        self.assertFalse(tree.leaf)
        self.assertEqual(tree.feature, 0)
        np.testing.assert_array_equal(probabilities >= 0.5, targets)

    def test_tree_refuses_too_small_dataset(self):
        with self.assertRaisesRegex(ValueError, "minimum-size leaves"):
            simple.fit_tree(
                np.zeros((3, 2)), np.asarray([False, True, True]),
                min_leaf=2,
            )

    def test_classification_treats_false_positive_as_unsafe_exposure(self):
        metrics = simple.classification_metrics(
            np.asarray([True, True, False, False]),
            np.asarray([True, False, True, False]),
        )
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["safe_recall_pct"], 50.0)
        self.assertEqual(metrics["unsafe_exposure_pct"], 50.0)

    def test_binary_action_does_not_claim_headroom_above_target_scale(self):
        samples = [
            {"safe_scale_ceiling": 1.3},
            {"safe_scale_ceiling": 1.2},
        ]
        captured = simple.captured_safe_headroom_pct(
            samples, np.asarray([True, False])
        )
        self.assertAlmostEqual(captured, 20.0)

    def test_development_loader_uses_first_frame_and_shot_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / "clip"
            clip.mkdir()
            for frame in (0, 10):
                cv2.imwrite(
                    str(clip / f"frame_{frame:05d}.png"),
                    np.zeros((4, 6, 3), dtype=np.uint8),
                )
            labels = root / "labels.jsonl"
            rows = [{
                "label_schema": simple.label_merge.LABEL_SCHEMA,
                "policy_contract": simple.selector.POLICY_CONTRACT,
                "metric_sha256": "0123456789abcdef",
                "policy_warp_source_sha256": "a" * 64,
                "split": "development",
                "clip": "clip",
                "film_id": "film",
                "domain": "movie",
                "source": str(clip / "frame_00010.png"),
                "model_input_width": 14,
                "model_input_height": 14,
                "safe_scale_ceiling": 1.2,
                "deployment_geometry_allowlist_sha256": "b" * 64,
                **condition_target_fields(1.2),
            }]
            labels.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            samples, _provenance = simple.load_development_shots(
                labels, "0123456789abcdef", "a" * 64
            )
            self.assertEqual(len(samples), 1)
            self.assertTrue(samples[0]["safe"])
            self.assertTrue(samples[0]["source"].endswith("frame_00000.png"))

    def test_development_loader_rejects_inconsistent_shot_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / "clip"
            clip.mkdir()
            cv2.imwrite(
                str(clip / "frame_00000.png"),
                np.zeros((4, 6, 3), dtype=np.uint8),
            )
            base = {
                "label_schema": simple.label_merge.LABEL_SCHEMA,
                "policy_contract": simple.selector.POLICY_CONTRACT,
                "metric_sha256": "0123456789abcdef",
                "policy_warp_source_sha256": "a" * 64,
                "split": "development",
                "clip": "clip",
                "film_id": "film",
                "domain": "movie",
                "source": str(clip / "frame_00000.png"),
                "model_input_width": 14,
                "model_input_height": 14,
                "deployment_geometry_allowlist_sha256": "b" * 64,
            }
            labels = root / "labels.jsonl"
            labels.write_text(
                json.dumps({
                    **base, "safe_scale_ceiling": 1.0,
                    **condition_target_fields(1.0),
                }) + "\n" +
                json.dumps({
                    **base, "safe_scale_ceiling": 1.1,
                    **condition_target_fields(1.1),
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "inconsistent"):
                simple.load_development_shots(
                    labels, "0123456789abcdef", "a" * 64
                )

    def test_development_loader_requires_current_multigeometry_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / "clip"
            clip.mkdir()
            cv2.imwrite(
                str(clip / "frame_00000.png"),
                np.zeros((4, 6, 3), dtype=np.uint8),
            )
            labels = root / "labels.jsonl"
            labels.write_text(json.dumps({
                "label_schema": 8,
                "policy_contract": simple.selector.POLICY_CONTRACT,
                "metric_sha256": "0123456789abcdef",
                "policy_warp_source_sha256": "a" * 64,
                "split": "development",
                "clip": "clip",
                "film_id": "film",
                "domain": "movie",
                "source": str(clip / "frame_00000.png"),
                "model_input_width": 14,
                "model_input_height": 14,
                "safe_scale_ceiling": 1.1,
                "deployment_geometry_allowlist_sha256": "b" * 64,
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                    RuntimeError,
                    f"schema-{simple.label_merge.LABEL_SCHEMA}"):
                simple.load_development_shots(
                    labels, "0123456789abcdef", "a" * 64
                )

    def test_development_loader_rejects_missing_native_sdr_target(self):
        row = {
            "condition_target_contract": (
                simple.label_merge.CONDITION_TARGET_CONTRACT
            ),
            "input_condition_targets": [],
        }
        with self.assertRaisesRegex(RuntimeError, "native-SDR target"):
            simple.native_sdr_condition_target(row, "test row")


if __name__ == "__main__":
    unittest.main()
