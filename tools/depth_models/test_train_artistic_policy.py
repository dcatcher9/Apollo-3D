#!/usr/bin/env python3

import json
import hashlib
import math
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import artistic_policy_model as policy_model  # noqa: E402
import prepare_stereo_movie_training as prepare_movie  # noqa: E402
import train_artistic_policy as train  # noqa: E402
import evaluate_artistic_policy as evaluate  # noqa: E402
import export_artistic_policy as export_policy  # noqa: E402
import artistic_geometry_contract as geometry_contract  # noqa: E402


class ArtisticLabelLoadingTests(unittest.TestCase):
    @staticmethod
    def write_rows(path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def write_bundle(root, name, rows, thresholds_payload=None):
        bundle = root / name
        bundle.mkdir()
        labels = bundle / "labels.jsonl"
        geometry = {
            "source_width": 1920, "source_height": 1080,
            "model_input_width": 770, "model_input_height": 434,
            "depth_short_side": 432, "depth_max_aspect": 4.0,
            "eye_width": 1920, "eye_height": 1080,
            "content_scale_x": 1.0, "content_scale_y": 1.0,
            "disparity_raster_width": 1920,
            "disparity_raster_height": 1080,
            "color_mode": geometry_contract.COLOR_MODE_SDR,
        }
        allowlist = geometry_contract.build_allowlist([geometry])
        allowlist_hash = geometry_contract.allowlist_sha256(allowlist)
        rows = [
            {
                **row,
                "source_sha256": row.get(
                    "source_sha256", hashlib.sha256(
                        f"{name}-{row['clip']}-{row['frame']}".encode()
                    ).hexdigest()
                ),
                "deployment_geometry_allowlist_sha256": allowlist_hash,
                "deployment_geometry_variants": [{"geometry": geometry}],
            }
            for row in rows
        ]
        ArtisticLabelLoadingTests.write_rows(labels, rows)
        code_path = bundle / "fitter.py"
        code_path.write_text("# frozen\n", encoding="utf-8")
        code_hash = hashlib.sha256(code_path.read_bytes()).hexdigest()
        thresholds_path = bundle / "thresholds.json"
        thresholds_path.write_text(
            json.dumps(thresholds_payload or {"metrics": {}}),
            encoding="utf-8",
        )
        metric_sha256 = train.semantic_file_hash((
            code_path, thresholds_path, code_path,
        ))
        control_path = bundle / "control.json"
        control_path.write_text(
            json.dumps({"meta": {"metric_sha256": metric_sha256}}),
            encoding="utf-8",
        )
        fitter = {
            "schema": 9,
            "label_fitter": "test",
            "policy_contract": "safe-frontier-multistyle-apollo-v1",
            "label_fitter_config": {
                "analysis_width": 512,
                "objective": (
                    "multi-geometry-connected-safe-frontier-intersection-multistyle"
                ),
                "confidence_semantics": (
                    "hard actionable 0/1 probability target"
                ),
                "reliability_semantics": (
                    "soft safety-margin reliability from render evidence"
                ),
            },
            "model_limits": {"scale_delta_max": 0.5},
            "rendered_disparity_supervision": {"artifact": "test"},
            "thresholds": {
                "path": str(thresholds_path),
                "sha256": train.sha256(thresholds_path),
            },
            "control": {
                "path": str(control_path),
                "sha256": train.sha256(control_path),
            },
            "policy_baseline": {
                "profile": "apollo", "depth_model": "dav2-small",
                "pop_strength": 1.25, "warp_contract": "apollo-safe-frontier-v1",
                "depth_short_side": 432, "depth_max_aspect": 4.0,
            },
            "code": {
                role: {"path": str(code_path), "sha256": code_hash}
                for role in ("label_fitter", "policy_contract",
                             "label_preparation", "image_loader",
                             "geometry_merge", "evaluator_runner")
            },
            "deployment_geometry_allowlist": allowlist,
            "deployment_geometry_allowlist_sha256": allowlist_hash,
        }
        fitter_path = bundle / "label_fitter_contract.json"
        fitter_path.write_text(json.dumps(fitter), encoding="utf-8")
        summary = {
            "schema": 9,
            "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
            "label_fitter_contract_sha256": hashlib.sha256(
                fitter_path.read_bytes()
            ).hexdigest(),
        }
        (bundle / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return labels

    @staticmethod
    def write_active_split(root):
        catalog = root / "catalog.json"
        catalog.write_text(json.dumps({"schema": 1}), encoding="utf-8")
        assignments = {
            "train_film": "training",
            "dev_film": "development",
            "test_a": "test",
            "test_b": "test",
        }
        production_rows = []
        datasets = {}
        for index, (production, split) in enumerate(assignments.items()):
            video_hash = hashlib.sha256(f"video-{index}".encode()).hexdigest()
            dataset = root / f"{production}.json"
            dataset.write_text(json.dumps({
                "schema": 1,
                "film_id": production,
                "split": split,
                "video_sha256": video_hash,
            }), encoding="utf-8")
            datasets[production] = dataset
            production_rows.append({
                "production_id": production,
                "split": split,
                "dataset_manifest": str(dataset),
                "dataset_manifest_sha256": train.sha256(dataset),
                "video_sha256": video_hash,
            })
        active = root / "active.json"
        payload = {
            "schema": 1,
            "catalog": str(catalog),
            "catalog_sha256": train.sha256(catalog),
            "productions": production_rows,
            "split_productions": {
                "training": ["train_film"],
                "development": ["dev_film"],
                "test": ["test_a", "test_b"],
            },
        }
        active.write_text(json.dumps(payload), encoding="utf-8")
        return active, payload, catalog, datasets

    def test_multiple_label_sources_are_combined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.write_bundle(
                root, "first", [{"clip": "a", "frame": 0}]
            )
            second = self.write_bundle(
                root, "second", [{"clip": "b", "frame": 0}]
            )
            rows = train.load_rows([first, second])
            self.assertEqual([row["clip"] for row in rows], ["a", "b"])

            sources, digest = train.labels_contract([first, second])
            self.assertEqual(len(sources), 2)
            self.assertEqual(len(digest), 64)
            self.assertEqual(sources[0]["policy_baseline"]["profile"], "apollo")
            self.assertEqual(len(sources[0]["metric_sha256"]), 16)

    def test_label_bundles_reject_different_metric_implementations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.write_bundle(
                root, "first", [{"clip": "a", "frame": 0}]
            )
            second = self.write_bundle(
                root, "second", [{"clip": "b", "frame": 0}],
                {"metrics": {"changed": {"role": "diagnostic"}}},
            )
            with self.assertRaisesRegex(RuntimeError, "metric implementations"):
                train.labels_contract([first, second])

    def test_label_bundle_requires_every_matching_destination_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = self.write_bundle(
                root, "incomplete_geometry", [{"clip": "a", "frame": 0}]
            )
            bundle = labels.parent
            fitter_path = bundle / "label_fitter_contract.json"
            fitter = json.loads(fitter_path.read_text(encoding="utf-8"))
            base = fitter["deployment_geometry_allowlist"]["tuples"][0]
            wide = {
                **base,
                "eye_width": 2560,
                "eye_height": 1080,
                "content_scale_x": 0.75,
                "disparity_raster_width": 2560,
                "disparity_raster_height": 1080,
            }
            allowlist = geometry_contract.build_allowlist([base, wide])
            allowlist_hash = geometry_contract.allowlist_sha256(allowlist)
            fitter["deployment_geometry_allowlist"] = allowlist
            fitter["deployment_geometry_allowlist_sha256"] = allowlist_hash
            fitter_path.write_text(json.dumps(fitter), encoding="utf-8")

            rows = [json.loads(line) for line in labels.read_text(
                encoding="utf-8"
            ).splitlines() if line]
            rows[0]["deployment_geometry_allowlist_sha256"] = allowlist_hash
            self.write_rows(labels, rows)
            summary_path = bundle / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["labels_sha256"] = train.sha256(labels)
            summary["label_fitter_contract_sha256"] = train.sha256(fitter_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(
                    RuntimeError, "omits a matching deployment geometry variant"):
                train.labels_contract([labels])

    def test_export_rejects_depth_weight_or_metric_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "depth.pth"
            weights.write_bytes(b"official-depth-weights")
            checkpoint = {
                "depth_weights_sha256": train.sha256(weights),
                "metric_sha256": "0123456789abcdef",
            }
            allowlist = geometry_contract.build_allowlist([{
                "source_width": 1920, "source_height": 1080,
                "model_input_width": 770, "model_input_height": 434,
                "depth_short_side": 432, "depth_max_aspect": 4.0,
                "eye_width": 1920, "eye_height": 1080,
                "content_scale_x": 1.0, "content_scale_y": 1.0,
                "disparity_raster_width": 1920,
                "disparity_raster_height": 1080,
                "color_mode": geometry_contract.COLOR_MODE_SDR,
            }])
            checkpoint["deployment_geometry_allowlist"] = allowlist
            checkpoint["deployment_geometry_allowlist_sha256"] = (
                geometry_contract.allowlist_sha256(allowlist)
            )
            actual, metric, actual_allowlist, geometry_hash = (
                export_policy.validate_export_provenance(
                    checkpoint, weights
                )
            )
            self.assertEqual(actual, checkpoint["depth_weights_sha256"])
            self.assertEqual(metric, checkpoint["metric_sha256"])
            self.assertEqual(actual_allowlist, allowlist)
            self.assertEqual(
                geometry_hash,
                checkpoint["deployment_geometry_allowlist_sha256"],
            )

            checkpoint["depth_weights_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "DA-V2 weights"):
                export_policy.validate_export_provenance(checkpoint, weights)
            checkpoint["depth_weights_sha256"] = train.sha256(weights)
            checkpoint["metric_sha256"] = "missing"
            with self.assertRaisesRegex(RuntimeError, "metric provenance"):
                export_policy.validate_export_provenance(checkpoint, weights)

    def test_export_requires_matching_accepted_sealed_test_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.pt"
            policy.write_bytes(b"exact-policy-checkpoint")
            evaluation = root / "evaluation.json"
            checkpoint = {
                "active_split_sha256": "1" * 64,
                "metric_sha256": "2" * 16,
                "label_fitter_identity_sha256": "3" * 64,
                "sealed_test_productions": ["film_b", "film_a"],
            }
            allowlist = geometry_contract.build_allowlist([{
                "source_width": 1920, "source_height": 1080,
                "model_input_width": 770, "model_input_height": 434,
                "depth_short_side": 432, "depth_max_aspect": 4.0,
                "eye_width": 1920, "eye_height": 1080,
                "content_scale_x": 1.0, "content_scale_y": 1.0,
                "disparity_raster_width": 1920,
                "disparity_raster_height": 1080,
                "color_mode": geometry_contract.COLOR_MODE_SDR,
            }])
            checkpoint["deployment_geometry_allowlist"] = allowlist
            checkpoint["deployment_geometry_allowlist_sha256"] = (
                geometry_contract.allowlist_sha256(allowlist)
            )
            unsafe_overshoot = {
                "maximum_scale": 0.04,
                "maximum_limit_scale": 0.05,
                "film_balanced_mean_scale": 0.008,
                "film_balanced_mean_limit_scale": 0.01,
                "film_balanced_overshoot_rate_pct": 25.0,
                "by_film_mean_scale": {"film_a": 0.006, "film_b": 0.01},
                "by_film_overshoot_rate_pct": {
                    "film_a": 0.0, "film_b": 50.0,
                },
                "maximum_pass": True,
                "film_balanced_mean_pass": True,
            }
            payload = {
                "schema": export_policy.EVALUATION_SCHEMA,
                "split": "test",
                "checkpoint_sha256": train.sha256(policy),
                "active_split_sha256": checkpoint["active_split_sha256"],
                "metric_sha256": checkpoint["metric_sha256"],
                "label_fitter_identity_sha256": checkpoint[
                    "label_fitter_identity_sha256"
                ],
                "deployment_geometry_allowlist": allowlist,
                "deployment_geometry_allowlist_sha256": checkpoint[
                    "deployment_geometry_allowlist_sha256"
                ],
                "test_labels_sha256": "4" * 64,
                "val_films": ["film_a", "film_b"],
                "unsafe_ceiling_overshoot": unsafe_overshoot,
                "decision": {
                    "accepted": True,
                    "guards": {
                        "unsafe_ceiling_maximum": True,
                        "unsafe_ceiling_film_balanced_mean": True,
                    },
                    "unsafe_overshoot_guard_required": True,
                    "unsafe_ceiling_overshoot": unsafe_overshoot,
                },
            }

            def write_evaluation():
                evaluation.write_text(
                    json.dumps(payload) + "\n", encoding="utf-8"
                )

            write_evaluation()
            evaluation_hash, approval = (
                export_policy.validate_sealed_test_approval(
                    checkpoint, train.sha256(policy), evaluation
                )
            )
            self.assertEqual(evaluation_hash, train.sha256(evaluation))
            self.assertEqual(
                approval["contract"],
                export_policy.SEALED_TEST_APPROVAL_CONTRACT,
            )
            self.assertEqual(
                approval["checkpoint_sha256"], train.sha256(policy)
            )
            self.assertEqual(
                approval["sealed_test_productions"], ["film_a", "film_b"]
            )
            self.assertEqual(
                approval["unsafe_ceiling_overshoot"], unsafe_overshoot
            )

            payload["schema"] = 10
            write_evaluation()
            with self.assertRaisesRegex(RuntimeError, "evaluation schema"):
                export_policy.validate_sealed_test_approval(
                    checkpoint, train.sha256(policy), evaluation
                )
            payload["schema"] = export_policy.EVALUATION_SCHEMA

            for key, value, message in (
                    ("split", "development", "sealed test split"),
                    ("checkpoint_sha256", "5" * 64, "checkpoint_sha256"),
                    ("active_split_sha256", "5" * 64,
                     "active_split_sha256"),
                    ("metric_sha256", "5" * 16, "metric_sha256"),
                    ("label_fitter_identity_sha256", "5" * 64,
                     "label_fitter_identity_sha256")):
                with self.subTest(key=key):
                    original = payload[key]
                    payload[key] = value
                    write_evaluation()
                    with self.assertRaisesRegex(RuntimeError, message):
                        export_policy.validate_sealed_test_approval(
                            checkpoint, train.sha256(policy), evaluation
                        )
                    payload[key] = original

            payload["decision"]["accepted"] = False
            write_evaluation()
            with self.assertRaisesRegex(RuntimeError, "did not accept"):
                export_policy.validate_sealed_test_approval(
                    checkpoint, train.sha256(policy), evaluation
                )
            payload["decision"]["accepted"] = True
            payload["val_films"] = ["film_a"]
            write_evaluation()
            with self.assertRaisesRegex(RuntimeError, "productions"):
                export_policy.validate_sealed_test_approval(
                    checkpoint, train.sha256(policy), evaluation
                )

            payload["val_films"] = ["film_a", "film_b"]
            payload["unsafe_ceiling_overshoot"]["maximum_scale"] = 0.051
            payload["unsafe_ceiling_overshoot"]["maximum_pass"] = False
            payload["decision"]["unsafe_ceiling_overshoot"] = (
                payload["unsafe_ceiling_overshoot"]
            )
            payload["decision"]["guards"]["unsafe_ceiling_maximum"] = False
            write_evaluation()
            with self.assertRaisesRegex(RuntimeError, "unsafe-ceiling"):
                export_policy.validate_sealed_test_approval(
                    checkpoint, train.sha256(policy), evaluation
                )

    def test_schema9_row_trains_geometry_intersection_ceiling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "frame.png"
            baseline = root / "baseline.f32"
            unclamped = root / "unclamped.f32"
            unclamped_wide = root / "unclamped-wide.f32"
            source.write_bytes(b"source")
            baseline.write_bytes(b"baseline")
            unclamped.write_bytes(
                np.asarray([1, 1], dtype="<u4").tobytes()
                + np.asarray([0.01], dtype="<f4").tobytes()
            )
            unclamped_wide.write_bytes(
                np.asarray([2, 1], dtype="<u4").tobytes()
                + np.asarray([0.01, 0.02], dtype="<f4").tobytes()
            )

            def digest(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            row = {
                "label_schema": 9,
                "policy_contract": "safe-frontier-multistyle-apollo-v1",
                "source": str(source), "source_sha256": digest(source),
                "baseline_disparity": str(baseline),
                "baseline_disparity_sha256": digest(baseline),
                "baseline_unclamped_disparity": str(unclamped),
                "baseline_unclamped_disparity_sha256": digest(unclamped),
                "baseline_multiplier": 1.2, "confidence": 1.0,
                "safe_scale_ceiling": 1.2, "ceiling_confidence": 1.0,
                "safety_margin_reliability": 0.8,
                "render_evidence_confidence": 0.8,
                "safe_scale_min": 0.9, "safe_scale_max": 1.2,
                "style_targets": {
                    "clean": 1.0, "balanced": 1.15, "immersive": 1.2,
                },
                "style_render_targets": {},
                "safe_ceiling_render_target": {
                    "comfort_clamp_abs_pct": 3.0,
                    "hlsl_full_clamp_abs": 0.02,
                },
                "source_width": 10, "source_height": 10,
                "artistic_full_clamp_abs": 0.02,
                "safe_ceiling_exact_pop_spread_pct": 2.0,
                "baseline_disparity_mean_abs_pct": 1.0,
                "baseline_unclamped_disparity_mean_abs_pct": 1.1,
                "render_grid_key": "shot", "clip": "shot", "frame": 0,
                "split": "training", "film_id": "film",
                "global_policy_weight": 1.0,
            }
            geometries = [{
                "source_width": 10, "source_height": 10,
                "model_input_width": 14, "model_input_height": 14,
                "depth_short_side": 432, "depth_max_aspect": 4.0,
                "eye_width": 1, "eye_height": 1,
                "content_scale_x": 1.0, "content_scale_y": 1.0,
                "disparity_raster_width": 1,
                "disparity_raster_height": 1,
                "color_mode": geometry_contract.COLOR_MODE_SDR,
            }, {
                "source_width": 10, "source_height": 10,
                "model_input_width": 14, "model_input_height": 14,
                "depth_short_side": 432, "depth_max_aspect": 4.0,
                "eye_width": 2, "eye_height": 1,
                "content_scale_x": 0.5, "content_scale_y": 1.0,
                "disparity_raster_width": 2,
                "disparity_raster_height": 1,
                "color_mode": geometry_contract.COLOR_MODE_SDR,
            }]
            allowlist = geometry_contract.build_allowlist(geometries)
            row["deployment_geometry_allowlist_sha256"] = (
                geometry_contract.allowlist_sha256(allowlist)
            )
            row["deployment_geometry_variants"] = [
                {
                    "geometry": geometry,
                    "baseline_unclamped_disparity": str(path),
                    "baseline_unclamped_disparity_sha256": digest(path),
                    "artistic_full_clamp_abs": 0.02,
                }
                for geometry, path in zip(
                    geometries, (unclamped, unclamped_wide)
                )
            ]
            train.validate_row(row)
            image = np.zeros((10, 10, 3), np.uint8)
            cv2.imwrite(str(source), image)
            row["source_sha256"] = digest(source)
            _image, target, raw, clamp_abs = train.PolicyDataset([row])[0]
            self.assertAlmostEqual(float(target[0]), 1.2)
            self.assertEqual(float(target[1]), 1.0)
            self.assertAlmostEqual(float(target[4]), 0.8)
            self.assertEqual(tuple(target.shape), (5,))
            self.assertEqual([tuple(field.shape) for field in raw], [(1, 1), (1, 2)])
            self.assertEqual(clamp_abs, [0.02, 0.02])
            row["baseline_multiplier"] = 1.15
            with self.assertRaisesRegex(RuntimeError, "safe ceiling"):
                train.validate_row(row)
            row["baseline_multiplier"] = 1.2
            row["confidence"] = 0.8
            row["ceiling_confidence"] = 0.8
            with self.assertRaisesRegex(RuntimeError, "hard actionable"):
                train.validate_row(row)

    def test_global_head_is_exact_identity_when_untrained(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pretrained = type("Pretrained", (), {"embed_dim": 4})()

        model = policy_model.ArtisticPolicyModel(Backbone())
        self.assertEqual(model.policy_feature_size, 40)
        output = model.forward_policy_features(
            torch.zeros((2, model.policy_feature_size))
        )
        self.assertTrue(torch.equal(output[:, 0], torch.ones(2)))
        self.assertTrue(torch.allclose(output[:, 1], torch.full((2,), 0.02)))
        self.assertNotIn("local_head", " ".join(policy_model.policy_state_dict(model)))

        raw = torch.tensor(0.0, requires_grad=True)
        ceiling = model._safe_ceiling(raw)
        ceiling.backward()
        self.assertEqual(float(ceiling.detach()), 1.0)
        self.assertGreater(float(raw.grad.detach()), 0.0)
        negative = torch.tensor(-2.0, requires_grad=True)
        negative_ceiling = model._safe_ceiling(negative)
        negative_ceiling.backward()
        self.assertEqual(float(negative_ceiling.detach()), 1.0)
        self.assertGreater(float(negative.grad.detach()), 0.0)
        self.assertEqual(float(model._safe_ceiling(torch.tensor(1.0))), 1.5)
        probe = model._safe_ceiling(torch.linspace(-100.0, 100.0, 101))
        self.assertTrue(torch.all(probe >= 1.0))
        self.assertTrue(torch.all(probe <= 1.5))

    def test_preprocessing_dimensions_match_production_examples(self):
        self.assertEqual(
            geometry_contract.aspect_aligned_dims(5120, 2160), (994, 420)
        )
        self.assertEqual(
            geometry_contract.aspect_aligned_dims(1920, 1080), (770, 434)
        )
        self.assertEqual(
            geometry_contract.aspect_aligned_dims(8000, 1000), (1008, 252)
        )
        self.assertEqual(
            geometry_contract.aspect_aligned_dims(1000, 8000), (252, 1008)
        )
        self.assertEqual(
            geometry_contract.aspect_aligned_dims(
                5120, 1080, depth_short_side=280, depth_max_aspect=2.0
            ),
            (560, 280),
        )

    def test_preprocessing_respects_native_low_resolution_bounds(self):
        # Match video_depth_estimator.cpp's min(1008, native dimension) profile bounds.
        # Without these bounds the offline loader would silently upscale both sources to a
        # 434-pixel short side and train the policy on a feature grid production never sees.
        self.assertEqual(
            geometry_contract.aspect_aligned_dims(
                320, 180, max_width=min(train.MAX_WIDTH, 320),
                max_height=min(train.MAX_HEIGHT, 180),
            ),
            (294, 168),
        )
        self.assertEqual(
            geometry_contract.aspect_aligned_dims(
                180, 320, max_width=min(train.MAX_WIDTH, 180),
                max_height=min(train.MAX_HEIGHT, 320),
            ),
            (168, 294),
        )

    def test_policy_dataset_uses_native_bounds_for_low_resolution_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.f32"
            raw.write_bytes(
                np.asarray([1, 1], dtype="<u4").tobytes()
                + np.asarray([0.0], dtype="<f4").tobytes()
            )

            for name, source_shape, expected_shape in (
                    ("landscape", (180, 320), (3, 168, 294)),
                    ("portrait", (320, 180), (3, 294, 168))):
                with self.subTest(name=name):
                    source = root / f"{name}.png"
                    cv2.imwrite(
                        str(source), np.zeros((*source_shape, 3), np.uint8)
                    )
                    row = {
                        "source": str(source),
                        "safe_scale_ceiling": 1.0,
                        "ceiling_confidence": 0.0,
                        "safe_scale_min": 1.0,
                        "safe_scale_max": 1.0,
                        "safety_margin_reliability": 0.0,
                        "baseline_unclamped_disparity": str(raw),
                        "artistic_full_clamp_abs": 0.03,
                        "deployment_geometry_variants": [
                            {
                                "baseline_unclamped_disparity": str(raw),
                                "artistic_full_clamp_abs": 0.03,
                                "geometry": {
                                    "source_width": source_shape[1],
                                    "source_height": source_shape[0],
                                    "model_input_width": expected_shape[2],
                                    "model_input_height": expected_shape[1],
                                    "depth_short_side": 432,
                                    "depth_max_aspect": 4.0,
                                    "eye_width": 1,
                                    "eye_height": 1,
                                    "disparity_raster_height": 1,
                                    "disparity_raster_width": 1,
                                    "content_scale_x": 1.0,
                                    "content_scale_y": 1.0,
                                },
                            },
                        ],
                    }
                    image, _target, _raw, _clamp = train.PolicyDataset([row])[0]
                    self.assertEqual(tuple(image.shape), expected_shape)

    def test_movie_keeps_full_cadence_context_before_label_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "stereo.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 6.0, (128, 48)
            )
            if not writer.isOpened():
                self.skipTest("OpenCV MJPG writer is unavailable")
            for frame in range(6):
                image = np.full((48, 128, 3), 20 + frame, np.uint8)
                writer.write(image)
            writer.release()
            output = root / "prepared"
            manifest = prepare_movie.prepare(
                video, output, "test", "test", layout="side-by-side",
                sample_fps=2.0, cut_threshold=1.0, output_width=0,
                split="training", film_id="test",
            )
            clip = output / "test_shot_0000"
            self.assertEqual(manifest["context_frame_count"], 6)
            self.assertEqual(manifest["sample_count"], 2)
            self.assertEqual(len(list(clip.glob("frame_*.png"))), 6)
            self.assertEqual(len(list((clip / "gt_right").glob("frame_*.png"))), 2)
            self.assertTrue((clip / "gt_right" / "frame_00000.png").is_file())

    def test_movie_restores_anamorphic_eye_aspect(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "half_sbs.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (128, 72)
            )
            if not writer.isOpened():
                self.skipTest("OpenCV MJPG writer is unavailable")
            writer.write(np.full((72, 128, 3), 80, np.uint8))
            writer.release()
            output = root / "prepared"
            manifest = prepare_movie.prepare(
                video, output, "test", "test", layout="side-by-side",
                sample_fps=2.0, cut_threshold=1.0, output_width=160,
                split="training", film_id="test", eye_aspect_ratio=16 / 9,
            )
            image = cv2.imread(str(output / "test_shot_0000" / "frame_00000.png"))
            self.assertEqual(image.shape[1::-1], (160, 90))
            self.assertAlmostEqual(manifest["display_eye_aspect_ratio"], 16 / 9)

    def test_duplicate_clip_frame_is_rejected_across_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            row = {"clip": "a", "frame": 0}
            self.write_rows(first, [row])
            self.write_rows(second, [row])
            with self.assertRaisesRegex(RuntimeError, "duplicate artistic label"):
                train.load_rows([first, second])

    def test_global_validation_rejects_film_leakage(self):
        training = [{
            "clip": "film_a_shot_0", "film_id": "film_a",
            "global_policy_weight": 1.0,
        }]
        validation = [{
            "clip": "film_a_shot_1", "film_id": "film_a",
            "global_policy_weight": 1.0,
        }]
        with self.assertRaisesRegex(RuntimeError, "leaks complete films"):
            train.validate_global_film_split(training, validation)

        validation[0]["global_policy_weight"] = 0.0
        train.validate_global_film_split(training, validation)

    def test_sampler_applies_global_policy_weight_after_domain_balance(self):
        rows = [
            {"domain": "cinema", "clip": "a", "global_policy_weight": 1.0},
            {"domain": "cinema", "clip": "a", "global_policy_weight": 1.0},
            {"domain": "supplement", "clip": "b", "global_policy_weight": 0.25},
        ]
        weights = train.balanced_sample_weights(rows)
        self.assertAlmostEqual(sum(weights[:2]) / weights[2], 4.0)

    def test_same_shot_pairing_never_crosses_clip_boundaries(self):
        rows = [
            {"clip": "a", "frame": 0}, {"clip": "a", "frame": 2},
            {"clip": "b", "frame": 0}, {"clip": "b", "frame": 2},
            {"clip": "single", "frame": 0},
        ]
        peers = train.same_shot_peer_indices(rows)
        self.assertEqual(peers, [1, 0, 3, 2, 4])
        for index, peer in enumerate(peers):
            self.assertEqual(rows[index]["clip"], rows[peer]["clip"])

    def test_shot_consistency_penalizes_divergent_pair_predictions(self):
        target = torch.tensor([
            [1.2, 1.0, 0.9, 1.2, 0.9],
            [1.0, 0.0, 0.9, 1.0, 0.0],
        ])
        predicted = torch.tensor([[1.2, 0.9], [1.0, 0.1]])
        same = predicted.clone()
        divergent = torch.tensor([[1.2, 0.1], [0.8, 0.9]])
        raw = [torch.tensor([0.01]), torch.tensor([0.01])]
        clamp_abs = torch.tensor([0.03, 0.03])
        same_loss, same_parts = train.losses(
            predicted, target, same, raw, clamp_abs
        )
        divergent_loss, divergent_parts = train.losses(
            predicted, target, divergent, raw, clamp_abs
        )
        self.assertGreater(divergent_loss, same_loss)
        self.assertGreater(divergent_parts["shot_consistency"],
                           same_parts["shot_consistency"])

    def test_policy_checkpoint_cannot_overwrite_frozen_depth_model(self):
        class StubPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.depth_model = torch.nn.Linear(1, 1, bias=False)
                self.global_head = torch.nn.Linear(1, 1)

        model = StubPolicy()
        depth_before = model.depth_model.weight.detach().clone()
        payload = {
            "schema": policy_model.POLICY_CHECKPOINT_SCHEMA,
            "policy_contract": policy_model.POLICY_CONTRACT,
            "policy_feature_contract": policy_model.POLICY_FEATURE_CONTRACT,
            "output_semantics": policy_model.POLICY_OUTPUT_SEMANTICS,
            "policy_baseline": {},
            "metric_sha256": "0123456789abcdef",
            "policy_state": {
                **{
                    key: value.detach().clone()
                    for key, value in policy_model.policy_state_dict(model).items()
                },
                "depth_model.weight": torch.full_like(
                    model.depth_model.weight, 42.0
                ),
            },
        }
        with self.assertRaisesRegex(RuntimeError, "frozen depth-model weights"):
            policy_model.load_policy_state(model, Path("unused.pt"), payload)
        self.assertTrue(torch.equal(model.depth_model.weight, depth_before))

    def test_checkpoint_metrics_macro_average_clips_then_films(self):
        rows = [
            {"film_id": "large", "clip": "many"},
            {"film_id": "large", "clip": "many"},
            {"film_id": "large", "clip": "many"},
            {"film_id": "small", "clip": "one"},
        ]
        target = np.asarray([[0.9, 1.0]] * 4)
        predicted = np.asarray([
            [0.9, 0.9], [0.9, 0.9], [0.9, 0.9], [1.0, 0.1]
        ])
        metrics = train.film_balanced_acceptance(predicted, target, rows)
        self.assertAlmostEqual(
            metrics["macro"]["first_frame_effective_scale_mae_pct"], 5.0
        )
        self.assertEqual(set(metrics["films"]), {"large", "small"})

    def test_active_split_rejects_test_rows_during_training(self):
        active = {"split_productions": {
            "training": ["train_film"],
            "development": ["dev_film"],
            "test": ["test_a", "test_b"],
        }}
        rows = [
            {"split": "training", "film_id": "train_film"},
            {"split": "development", "film_id": "dev_film"},
        ]
        train.validate_rows_against_active_split(
            rows, active, {"training", "development"}
        )
        rows.append({"split": "test", "film_id": "test_a"})
        with self.assertRaisesRegex(RuntimeError, "disallowed splits"):
            train.validate_rows_against_active_split(
                rows, active, {"training", "development"}
            )

    def test_active_split_authenticates_sources_and_is_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active, payload, catalog, datasets = self.write_active_split(root)
            loaded, identity = train.load_active_split(active)
            self.assertEqual(loaded["split_productions"], payload["split_productions"])
            self.assertEqual(identity, train.sha256(active))

            catalog.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "catalog hash is stale"):
                train.load_active_split(active)
            catalog.write_text(json.dumps({"schema": 1}), encoding="utf-8")

            datasets["train_film"].write_text(
                json.dumps({
                    "schema": 1, "film_id": "train_film",
                    "split": "training", "video_sha256": "f" * 64,
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "dataset manifest hash is stale"):
                train.load_active_split(active)

            self.write_active_split(root)
            payload = json.loads(active.read_text(encoding="utf-8"))
            payload["split_productions"]["test"].append("train_film")
            active.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "appears in both"):
                train.load_active_split(active)

    def test_active_split_rejects_duplicate_video_across_productions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active, payload, _catalog, datasets = self.write_active_split(root)
            shared = payload["productions"][0]["video_sha256"]
            payload["productions"][1]["video_sha256"] = shared
            dataset = json.loads(
                datasets["dev_film"].read_text(encoding="utf-8")
            )
            dataset["video_sha256"] = shared
            datasets["dev_film"].write_text(
                json.dumps(dataset), encoding="utf-8"
            )
            payload["productions"][1]["dataset_manifest_sha256"] = train.sha256(
                datasets["dev_film"]
            )
            active.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "same source video"):
                train.load_active_split(active)

    def test_policy_decision_requires_held_out_film_majority(self):
        def summary(trained, neutral):
            keys = [item[0] for item in evaluate.ALL_METRICS]
            trained_values = {key: trained for key in keys}
            neutral_values = {key: neutral for key in keys}
            trained_values["identity_false_action_pct"] = 0.0
            neutral_values["identity_false_action_pct"] = 0.0
            return {
                "trained": trained_values,
                "neutral": neutral_values,
            }

        decision = evaluate.policy_decision(
            summary(0.5, 1.0),
            {"c1": summary(0.5, 1.0), "c2": summary(0.5, 1.0)},
            {"domain": summary(0.5, 1.0)},
            {"film_a": summary(0.5, 1.0), "film_b": summary(1.5, 1.0)},
            ["c1", "c2"], ["domain"], ["film_a", "film_b"],
        )
        self.assertFalse(decision["accepted"])
        self.assertTrue(all(decision["aggregate_wins"].values()))
        self.assertEqual(
            decision["film_count"]["effective_scale_mae_pct"], 2
        )
        self.assertEqual(
            decision["film_wins"]["effective_scale_mae_pct"], 1
        )

        one_film = evaluate.policy_decision(
            summary(0.5, 1.0), {"c1": summary(0.5, 1.0)},
            {"domain": summary(0.5, 1.0)},
            {"film_a": summary(0.5, 1.0)},
            ["c1"], ["domain"], ["film_a"],
        )
        self.assertFalse(one_film["accepted"])
        self.assertEqual(
            one_film["minimum_film_count"]["effective_scale_mae_pct"], 2
        )

    def test_sealed_decision_rejects_one_sided_unsafe_ceiling_overshoot(self):
        def measurement(film, clip, target, prediction):
            return {
                "film_id": film,
                "clip": clip,
                "prediction": {"scale": prediction, "confidence": 1.0},
                "target": {"scale": target, "confidence": 1.0},
            }

        maximum_rows = [
            measurement("film_a", f"a{index}", 1.2, 1.251 if index == 0 else 1.2)
            for index in range(6)
        ] + [measurement("film_b", "b", 1.2, 1.2)]
        maximum_guard = evaluate.unsafe_ceiling_overshoot(maximum_rows)
        self.assertFalse(maximum_guard["maximum_pass"])
        self.assertTrue(maximum_guard["film_balanced_mean_pass"])

        mean_rows = [
            measurement("film_a", "a", 1.2, 1.22),
            measurement("film_b", "b", 1.2, 1.22),
        ]
        mean_guard = evaluate.unsafe_ceiling_overshoot(mean_rows)
        self.assertTrue(mean_guard["maximum_pass"])
        self.assertFalse(mean_guard["film_balanced_mean_pass"])
        self.assertEqual(mean_guard["film_balanced_overshoot_rate_pct"], 100.0)

        def summary():
            trained = {key: 0.5 for key, _, _ in evaluate.ALL_METRICS}
            neutral = {key: 1.0 for key, _, _ in evaluate.ALL_METRICS}
            trained["identity_false_action_pct"] = None
            neutral["identity_false_action_pct"] = None
            return {"trained": trained, "neutral": neutral}

        summaries = {"a": summary(), "b": summary()}
        rejected = evaluate.policy_decision(
            summary(), summaries, {"domain": summary()},
            {"film_a": summary(), "film_b": summary()},
            ["a", "b"], ["domain"], ["film_a", "film_b"],
            require_identity_guard=False,
            unsafe_overshoot=mean_guard,
            require_unsafe_overshoot_guard=True,
        )
        self.assertFalse(rejected["accepted"])
        self.assertFalse(
            rejected["guards"]["unsafe_ceiling_film_balanced_mean"]
        )

    def test_rendered_disparity_loss_uses_exact_mean_baseline_magnitude(self):
        predicted = torch.tensor([[1.2, 0.9]])
        target = torch.tensor([[1.0, 0.0, 0.8, 1.0, 0.0]])
        _, low_parts = train.losses(
            predicted, target, raw_disparities=[torch.tensor([0.005])],
            clamp_abs=torch.tensor([0.03]),
        )
        _, high_parts = train.losses(
            predicted, target, raw_disparities=[torch.tensor([0.02])],
            clamp_abs=torch.tensor([0.03]),
        )
        self.assertAlmostEqual(
            float(high_parts["rendered_disparity"]),
            float(low_parts["rendered_disparity"]) * 4.0,
            places=5,
        )

    def test_exact_rendered_loss_accounts_for_comfort_saturation(self):
        predicted = torch.tensor([1.5], requires_grad=True)
        target = torch.tensor([1.0])
        exact = train.exact_clamped_disparity_errors(
            predicted, target, [torch.tensor([0.04, -0.04])],
            torch.tensor([0.03]),
        )
        factorized = 0.04 * abs(float(predicted.detach()) - 1.0) / 0.03
        self.assertEqual(float(exact.detach()), 0.0)
        self.assertGreater(factorized, 0.0)

    def test_exact_rendered_loss_matches_post_clamp_disparity_edges(self):
        predicted = torch.tensor([[1.2, 0.9]])
        target = torch.tensor([[1.0, 0.0, 0.8, 1.0, 0.0]])
        _, parts = train.losses(
            predicted, target,
            raw_disparities=[torch.tensor([[0.0, 0.01], [0.0, 0.02]])],
            clamp_abs=torch.tensor([0.03]),
        )
        self.assertGreater(float(parts["rendered_gradient"]), 0.0)

    def test_exact_rendered_loss_uses_worst_deployment_geometry(self):
        predicted = torch.tensor([1.2])
        target = torch.tensor([1.0])
        easy = torch.tensor([0.005])
        hard = torch.tensor([0.02])
        combined = train.exact_clamped_disparity_errors(
            predicted, target, [[easy, hard]], [[0.03, 0.03]],
        )
        hard_only = train.exact_clamped_disparity_errors(
            predicted, target, [hard], torch.tensor([0.03]),
        )
        self.assertAlmostEqual(float(combined), float(hard_only), places=6)

    def test_identity_examples_constrain_raw_scale_despite_zero_confidence(self):
        target = torch.tensor([[1.0, 0.0, 0.9, 1.0, 0.0]])
        identity, identity_parts = train.losses(
            torch.tensor([[1.0, 0.01]]), target,
            raw_disparities=[torch.tensor([0.01])],
            clamp_abs=torch.tensor([0.03]),
        )
        unsafe, unsafe_parts = train.losses(
            torch.tensor([[1.3, 0.01]]), target,
            raw_disparities=[torch.tensor([0.01])],
            clamp_abs=torch.tensor([0.03]),
        )
        self.assertGreater(unsafe, identity)
        self.assertGreater(unsafe_parts["global_style"],
                           identity_parts["global_style"])
        self.assertGreater(unsafe_parts["safe_frontier"],
                           identity_parts["safe_frontier"])

    def test_confidence_uses_hard_action_not_margin_reliability(self):
        target = torch.tensor([[1.2, 1.0, 1.0, 1.2, 0.5]])
        _loss, parts = train.losses(
            torch.tensor([[1.2, 0.75]]), target,
            raw_disparities=[torch.tensor([0.01])],
            clamp_abs=torch.tensor([0.03]),
        )
        self.assertAlmostEqual(
            float(parts["global_conf"]), -math.log(0.75), places=6
        )

    def test_acceptance_uses_first_available_frame_for_shot_latch(self):
        rows = [
            {"film_id": "film", "clip": "shot", "frame": 5},
            {"film_id": "film", "clip": "shot", "frame": 0},
            {"film_id": "film", "clip": "shot", "frame": 10},
        ]
        target = np.asarray([[1.2, 1.0]] * 3)
        predicted = np.asarray([
            [1.2, 0.9], [1.0, 0.1], [1.2, 0.9],
        ])
        metrics = train.film_balanced_acceptance(predicted, target, rows)
        self.assertAlmostEqual(
            metrics["macro"]["first_frame_effective_scale_mae_pct"], 20.0
        )
        self.assertEqual(
            metrics["macro"]["first_frame_action_recall_pct"], 0.0
        )

    def test_evaluator_aggregates_only_the_latched_first_frame(self):
        def measurement(frame, trained_error):
            metrics = {
                key: trained_error for key, _, _ in evaluate.ALL_METRICS
            }
            neutral = {
                key: 10.0 for key, _, _ in evaluate.ALL_METRICS
            }
            return {
                "film_id": "film", "clip": "shot", "domain": "movie",
                "frame": frame, "trained": metrics, "neutral": neutral,
            }

        # The first frame is poor and later frames are perfect. Averaging the later
        # frames would incorrectly approve behavior the runtime never gets to latch.
        rows = [measurement(10, 0.0), measurement(0, 20.0)]
        first, by_clip, _by_domain, _by_film, overall = (
            evaluate.shot_latched_aggregates(rows)
        )
        self.assertEqual([row["frame"] for row in first], [0])
        self.assertEqual(
            by_clip["shot"]["trained"]["effective_scale_mae_pct"], 20.0
        )
        self.assertEqual(
            overall["trained"]["effective_scale_mae_pct"], 20.0
        )


if __name__ == "__main__":
    unittest.main()
