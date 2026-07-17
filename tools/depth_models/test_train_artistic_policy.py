#!/usr/bin/env python3

import json
import hashlib
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
import test_artistic_policy_evaluation_contract as evaluation_contract_test  # noqa: E402
import test_audit_artistic_dataset_splits as split_contract_test  # noqa: E402


class ArtisticLabelLoadingTests(unittest.TestCase):
    @staticmethod
    def write_rows(path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def attach_four_condition_targets(row, geometries, disparity_paths,
                                      scales=(1.2, 1.1, 1.3, 1.15),
                                      reliabilities=(0.8, 0.7, 0.9, 0.6)):
        """Attach the exact schema-10 image-condition target/evidence shape."""
        variants = [
            value for value in train.label_merge.policy_input_variants()
            if value["kind"] != train.input_color.INPUT_KIND_NATIVE_PQ
        ]
        manifest = train.label_merge.build_input_variant_manifest(
            train.label_merge.policy_input_variants()
        )
        row["condition_target_contract"] = (
            train.label_merge.CONDITION_TARGET_CONTRACT
        )
        row["input_variant_manifest"] = manifest
        row["input_variant_manifest_sha256"] = (
            train.label_merge.input_variant_manifest_sha256(manifest)
        )
        row["depth_input_color_contract_sha256"] = (
            train.input_color.color_contract_sha256()
        )
        row["input_condition_targets"] = []
        row["deployment_geometry_variants"] = []

        def render(scale, clamp):
            return {
                "scale": scale, "hlsl_full_clamp_abs": clamp,
                "comfort_clamp_abs_pct": clamp * 100.0,
                "mean_abs_disparity_pct": scale,
                "p95_abs_disparity_pct": scale,
                "exact_pop_spread_pct": scale,
                "clamped_pixel_pct": 0.0,
            }

        clamp = float(row.get("artistic_full_clamp_abs", 0.02))
        for condition_index, input_variant in enumerate(variants):
            input_hash = train.input_color.input_variant_sha256(input_variant)
            scale = float(scales[condition_index])
            actionable = train.is_actionable_scale(scale)
            reliability = (
                float(reliabilities[condition_index]) if actionable else 0.0
            )
            confidence = 1.0 if actionable else 0.0
            style_targets = {
                "clean": 1.0,
                "balanced": 1.0 + 0.5 * (scale - 1.0),
                "immersive": scale,
            }
            target = {
                "schema": train.label_merge.CONDITION_TARGET_SCHEMA,
                "contract": train.label_merge.CONDITION_TARGET_CONTRACT,
                "input_variant": input_variant,
                "input_variant_sha256": input_hash,
                "deployment_geometry_variant_count": 2,
                "safe_scale_min": 0.9,
                "safe_scale_max": scale,
                "safe_scale_ceiling": scale,
                "baseline_multiplier": scale,
                "ceiling_confidence": confidence,
                "confidence": confidence,
                "safety_margin_reliability": reliability,
                "render_evidence_confidence": reliability,
                "identity_feasible": True,
                "identity_infeasible_variants": [],
                "style_targets": style_targets,
                "style_render_targets": {
                    name: render(value, clamp)
                    for name, value in style_targets.items()
                },
                "safe_ceiling_render_target": render(scale, clamp),
                "safe_ceiling_exact_pop_spread_pct": scale,
            }
            row["input_condition_targets"].append(target)
            for geometry, path in zip(geometries, disparity_paths):
                condition_geometry = json.loads(json.dumps(geometry))
                condition_geometry["color_mode"] = input_variant["color_mode"]
                row["deployment_geometry_variants"].append({
                    "geometry": condition_geometry,
                    "input_variant": input_variant,
                    "input_variant_sha256": input_hash,
                    "baseline_unclamped_disparity": str(path),
                    "baseline_unclamped_disparity_sha256": (
                        hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    ),
                    "artistic_full_clamp_abs": clamp,
                    "safe_scale_min": 0.9,
                    "safe_scale_max": scale,
                    "safety_margin_reliability": reliability,
                    "identity_feasible": True,
                    "identity_violations": [],
                    "safe_ceiling_render_target": render(scale, clamp),
                    "style_render_targets": {
                        name: render(value, clamp)
                        for name, value in style_targets.items()
                    },
                })
        return row

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
        second_geometry = dict(geometry, eye_width=1280, eye_height=720,
                               disparity_raster_width=1280,
                               disparity_raster_height=720)
        geometries = (geometry, second_geometry)
        allowlist = geometry_contract.build_allowlist([
            dict(item, color_mode=color_mode)
            for item in geometries
            for color_mode in (
                geometry_contract.COLOR_MODE_SDR,
                geometry_contract.COLOR_MODE_HDR,
            )
        ])
        allowlist_hash = geometry_contract.allowlist_sha256(allowlist)
        input_manifest = train.label_merge.build_input_variant_manifest(
            train.label_merge.policy_input_variants()
        )
        input_manifest_hash = train.label_merge.input_variant_manifest_sha256(
            input_manifest
        )
        disparity_paths = (bundle / "g0.f32", bundle / "g1.f32")
        for path in disparity_paths:
            path.write_bytes(b"test-disparity")
        prepared_rows = []
        for row in rows:
            prepared = {
                **row,
                "source_sha256": row.get(
                    "source_sha256", hashlib.sha256(
                        f"{name}-{row['clip']}-{row['frame']}".encode()
                    ).hexdigest()
                ),
                "deployment_geometry_allowlist_sha256": allowlist_hash,
            }
            ArtisticLabelLoadingTests.attach_four_condition_targets(
                prepared, geometries, disparity_paths
            )
            prepared_rows.append(prepared)
        rows = prepared_rows
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
            "schema": 10,
            "label_fitter": "test",
            "policy_contract": "safe-frontier-multistyle-apollo-v1",
            "label_fitter_config": {
                "analysis_width": 512,
                "objective": train.label_merge.OBJECTIVE,
                "confidence_semantics": (
                    "hard actionable 0/1 probability target"
                ),
                "reliability_semantics": (
                    "soft safety-margin reliability from render evidence"
                ),
                "condition_target_contract":
                    train.label_merge.CONDITION_TARGET_CONTRACT,
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
                for role in sorted(
                    train.label_merge.MERGED_LABEL_FITTER_CODE_ROLES
                )
            },
            "deployment_geometry_allowlist": allowlist,
            "deployment_geometry_allowlist_sha256": allowlist_hash,
            "input_variant_manifest": input_manifest,
            "input_variant_manifest_sha256": input_manifest_hash,
            "depth_input_color_contract_sha256":
                train.input_color.color_contract_sha256(),
            "condition_target_contract":
                train.label_merge.CONDITION_TARGET_CONTRACT,
        }
        fitter_path = bundle / "label_fitter_contract.json"
        fitter_path.write_text(json.dumps(fitter), encoding="utf-8")
        summary = {
            "schema": 10,
            "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
            "label_fitter_contract_sha256": hashlib.sha256(
                fitter_path.read_bytes()
            ).hexdigest(),
            "condition_target_contract":
                train.label_merge.CONDITION_TARGET_CONTRACT,
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

    def test_label_bundle_rejects_missing_or_changed_code_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = self.write_bundle(
                root, "code_identity", [{"clip": "a", "frame": 0}]
            )
            fitter_path = labels.parent / "label_fitter_contract.json"
            summary_path = labels.parent / "summary.json"
            fitter = json.loads(fitter_path.read_text(encoding="utf-8"))
            identity = fitter["code"].pop("depth_input_color_contract")
            fitter_path.write_text(json.dumps(fitter), encoding="utf-8")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["label_fitter_contract_sha256"] = train.sha256(fitter_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "code roles differ"):
                train.labels_contract([labels])

            fitter["code"]["depth_input_color_contract"] = identity
            fitter["code"]["image_loader"]["sha256"] = "0" * 64
            fitter_path.write_text(json.dumps(fitter), encoding="utf-8")
            summary["label_fitter_contract_sha256"] = train.sha256(fitter_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing or changed.*image_loader"):
                train.labels_contract([labels])

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
                    RuntimeError,
                    "unapproved deployment geometry|omits a matching deployment"):
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
            input_manifest = train.label_merge.build_input_variant_manifest([
                train.input_color.sdr_input_variant(),
                *(train.input_color.windows_hdr_input_variant(value)
                  for value in (1000, 2500, 6000)),
            ])
            checkpoint["input_variant_manifest"] = input_manifest
            checkpoint["input_variant_manifest_sha256"] = (
                train.label_merge.input_variant_manifest_sha256(input_manifest)
            )
            checkpoint["depth_input_color_contract_sha256"] = (
                train.input_color.color_contract_sha256()
            )
            checkpoint["condition_target_contract"] = (
                train.label_merge.CONDITION_TARGET_CONTRACT
            )
            (actual, metric, actual_allowlist, geometry_hash,
             actual_manifest, actual_manifest_hash, color_hash,
             condition_target_contract) = (
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
            self.assertEqual(actual_manifest, input_manifest)
            self.assertEqual(
                actual_manifest_hash, checkpoint["input_variant_manifest_sha256"]
            )
            self.assertEqual(
                color_hash, checkpoint["depth_input_color_contract_sha256"]
            )
            self.assertEqual(
                condition_target_contract,
                train.label_merge.CONDITION_TARGET_CONTRACT,
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
                "sealed_test_productions": ["film-b", "film-a"],
            }
            input_manifest = train.label_merge.build_input_variant_manifest([
                train.input_color.sdr_input_variant(),
                *(train.input_color.windows_hdr_input_variant(value)
                  for value in (1000, 2500, 6000)),
            ])
            checkpoint["input_variant_manifest"] = input_manifest
            checkpoint["input_variant_manifest_sha256"] = (
                train.label_merge.input_variant_manifest_sha256(input_manifest)
            )
            checkpoint["depth_input_color_contract_sha256"] = (
                train.input_color.color_contract_sha256()
            )
            checkpoint["condition_target_contract"] = (
                train.label_merge.CONDITION_TARGET_CONTRACT
            )
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
            runtime_payload, runtime_decision = (
                evaluation_contract_test.accepted_payload()
            )
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
                "input_variant_manifest": input_manifest,
                "input_variant_manifest_sha256": checkpoint[
                    "input_variant_manifest_sha256"
                ],
                "depth_input_color_contract_sha256": checkpoint[
                    "depth_input_color_contract_sha256"
                ],
                "condition_target_contract": checkpoint[
                    "condition_target_contract"
                ],
                "test_labels_sha256": "4" * 64,
                "val_films": runtime_payload["val_films"],
                "unsafe_ceiling_overshoot": unsafe_overshoot,
                "runtime_regime_evaluation": runtime_payload[
                    "runtime_regime_evaluation"
                ],
                "decision": {
                    "accepted": True,
                    **runtime_decision,
                    "guards": {
                        **runtime_decision["guards"],
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
                approval["sealed_test_productions"], ["film-a", "film-b"]
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
            with self.assertRaisesRegex(RuntimeError, "productions|film_count"):
                export_policy.validate_sealed_test_approval(
                    checkpoint, train.sha256(policy), evaluation
                )

            payload["val_films"] = ["film-a", "film-b"]
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

    def test_schema10_row_trains_each_condition_intersection_ceiling(self):
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
                "label_schema": 10,
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
            self.attach_four_condition_targets(
                row, geometries, (unclamped, unclamped_wide),
                scales=(1.2, 1.1, 1.3, 1.15),
            )
            train.validate_row(row)
            row["deployment_geometry_variants"][0][
                "input_variant_sha256"
            ] = "0" * 64
            with self.assertRaisesRegex(
                    RuntimeError, "stale or undeclared input identity"):
                train.validate_row(row)
            row["deployment_geometry_variants"][0][
                "input_variant_sha256"
            ] = train.input_color.input_variant_sha256(
                train.input_color.sdr_input_variant()
            )
            image = np.full((10, 10, 3), 128, np.uint8)
            cv2.imwrite(str(source), image)
            row["source_sha256"] = digest(source)
            dataset = train.PolicyDataset([row])
            self.assertEqual(len(dataset), 4)
            observed = {}
            images = []
            for index in range(len(dataset)):
                image, target, raw, clamp_abs = dataset[index]
                images.append(image)
                observed[dataset.rows[index]["_input_variant_sha256"]] = float(
                    target[0]
                )
                self.assertEqual(tuple(target.shape), (5,))
                self.assertEqual(
                    [tuple(field.shape) for field in raw], [(1, 1), (1, 2)]
                )
                self.assertEqual(clamp_abs, [0.02, 0.02])
                self.assertAlmostEqual(
                    dataset.rows[index]["safe_scale_ceiling"], float(target[0])
                )
            expected = {
                target["input_variant_sha256"]: target["safe_scale_ceiling"]
                for target in row["input_condition_targets"]
            }
            self.assertEqual(set(observed), set(expected))
            for key in expected:
                self.assertAlmostEqual(observed[key], expected[key])
            self.assertTrue(any(
                not torch.equal(images[0], image) for image in images[1:]
            ))
            native_row = json.loads(json.dumps(row))
            native_variant = train.input_color.native_pq_input_variant()
            native_hash = train.input_color.input_variant_sha256(native_variant)
            hdr_target = next(
                target for target in native_row["input_condition_targets"]
                if target["input_variant"]["color_mode"] ==
                train.input_color.COLOR_MODE_HDR
            )
            hdr_hash = hdr_target["input_variant_sha256"]
            hdr_target["input_variant"] = native_variant
            hdr_target["input_variant_sha256"] = native_hash
            native_row["input_condition_targets"] = [hdr_target]
            native_row["deployment_geometry_variants"] = [
                variant for variant in native_row["deployment_geometry_variants"]
                if variant["input_variant_sha256"] == hdr_hash
            ]
            for variant in native_row["deployment_geometry_variants"]:
                variant["input_variant"] = native_variant
                variant["input_variant_sha256"] = native_hash
            model_source = root / "frame.scrgb16"
            np.zeros((10, 10, 4), dtype="<f2").tofile(model_source)
            native_row.update({
                "model_source": str(model_source),
                "model_source_sha256": digest(model_source),
                "model_source_encoding":
                    train.native_hdr_capture.CAPTURE_ENCODING,
            })
            train.validate_row(native_row)
            native_dataset = train.PolicyDataset([native_row])
            self.assertEqual(len(native_dataset), 1)
            _image, _target, native_raw, _clamp = native_dataset[0]
            self.assertEqual(
                [tuple(field.shape) for field in native_raw], [(1, 1), (1, 2)]
            )
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
            raw_wide = root / "raw-wide.f32"
            raw_wide.write_bytes(
                np.asarray([2, 1], dtype="<u4").tobytes()
                + np.asarray([0.0, 0.0], dtype="<f4").tobytes()
            )

            for name, source_shape, expected_shape in (
                    ("landscape", (180, 320), (3, 168, 294)),
                    ("portrait", (320, 180), (3, 294, 168))):
                with self.subTest(name=name):
                    eye_shapes = (
                        ((16, 9), (32, 18)) if name == "landscape"
                        else ((9, 16), (18, 32))
                    )
                    local_raw = []
                    for geometry_index, (eye_width, eye_height) in enumerate(
                            eye_shapes):
                        path = root / f"{name}-raw-{geometry_index}.f32"
                        path.write_bytes(
                            np.asarray([eye_width, eye_height], dtype="<u4").tobytes()
                            + np.zeros(eye_width * eye_height, dtype="<f4").tobytes()
                        )
                        local_raw.append(path)
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
                                    "color_mode": geometry_contract.COLOR_MODE_SDR,
                                    "eye_width": eye_shapes[0][0],
                                    "eye_height": eye_shapes[0][1],
                                    "disparity_raster_height": eye_shapes[0][1],
                                    "disparity_raster_width": eye_shapes[0][0],
                                    "content_scale_x": 1.0,
                                    "content_scale_y": 1.0,
                                },
                            },
                        ],
                    }
                    first_geometry = row["deployment_geometry_variants"][0][
                        "geometry"
                    ]
                    second_geometry = json.loads(json.dumps(first_geometry))
                    second_geometry.update({
                        "eye_width": eye_shapes[1][0],
                        "eye_height": eye_shapes[1][1],
                        "disparity_raster_width": eye_shapes[1][0],
                        "disparity_raster_height": eye_shapes[1][1],
                    })
                    self.attach_four_condition_targets(
                        row, (first_geometry, second_geometry),
                        tuple(local_raw), scales=(1.0, 1.0, 1.0, 1.0),
                    )
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
            with mock.patch.object(
                    prepare_movie.video_color, "probe_sdr_input",
                    return_value={
                        "dataset_color_contract": "decoded-sdr-bgr8",
                        "admission": "probed-no-hdr-signals",
                    }):
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
            # Prepared movie shots require at least one adjacent frame so the
            # temporal evidence contract is meaningful.
            writer.write(np.full((72, 128, 3), 80, np.uint8))
            writer.write(np.full((72, 128, 3), 81, np.uint8))
            writer.release()
            output = root / "prepared"
            with mock.patch.object(
                    prepare_movie.video_color, "probe_sdr_input",
                    return_value={
                        "dataset_color_contract": "decoded-sdr-bgr8",
                        "admission": "probed-no-hdr-signals",
                    }):
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

    def test_sampler_balances_sdr_against_three_hdr_white_anchors(self):
        sdr = train.input_color.sdr_input_variant()
        hdr = [
            train.input_color.windows_hdr_input_variant(white)
            for white in (1000, 2500, 6000)
        ]
        rows = []
        for variant in (sdr, *hdr):
            rows.append({
                "domain": "cinema", "clip": "shot",
                "global_policy_weight": 1.0,
                "_input_variant": variant,
                "_input_variant_sha256": (
                    train.input_color.input_variant_sha256(variant)
                ),
            })
        weights = train.balanced_sample_weights(rows)
        self.assertAlmostEqual(weights[0], sum(weights[1:]))
        self.assertAlmostEqual(weights[1], weights[2])
        self.assertAlmostEqual(weights[2], weights[3])

    def test_sampler_balances_actions_inside_each_runtime_regime(self):
        sdr = train.input_color.sdr_input_variant()
        hdr = [
            train.input_color.windows_hdr_input_variant(white)
            for white in (1000, 2500, 6000)
        ]
        rows = []
        for variant in (sdr, *hdr):
            for action, ceiling in ((False, 1.0), (True, 1.2)):
                rows.append({
                    "domain": "cinema",
                    "clip": f"shot-{action}",
                    "global_policy_weight": 1.0,
                    "safe_scale_ceiling": ceiling,
                    "_input_variant": variant,
                    "_input_variant_sha256":
                        train.input_color.input_variant_sha256(variant),
                })
        weights = train.balanced_sample_weights(rows)
        self.assertAlmostEqual(sum(weights[:2]), sum(weights[2:]))
        self.assertAlmostEqual(weights[0], weights[1])
        self.assertAlmostEqual(sum(weights[2::2]), sum(weights[3::2]))

    def test_paired_sdr_hdr_source_targets_may_differ_but_tensor_must_match(self):
        scales = (1.2, 1.1, 1.3, 1.15)
        rows = []
        tensors = []
        for variant, scale in zip(train.label_merge.policy_input_variants(), scales):
            variant_hash = train.input_color.input_variant_sha256(variant)
            condition = {
                "input_variant_sha256": variant_hash,
                "safe_scale_ceiling": scale,
                "ceiling_confidence": 1.0,
                "safe_scale_min": 0.9,
                "safe_scale_max": scale,
                "safety_margin_reliability": 0.8,
            }
            rows.append({
                "source_sha256": "a" * 64,
                "film_id": "film", "clip": "shot", "frame": 7,
                "_input_variant_sha256": variant_hash,
                "_condition_target": condition,
            })
            tensors.append([scale, 1.0, 0.9, scale, 0.8])
        targets = torch.tensor(tensors)
        train.validate_expanded_variant_targets(rows, targets)
        stale = targets.clone()
        stale[3, 0] = 1.1
        with self.assertRaisesRegex(RuntimeError, "differs from its condition target"):
            train.validate_expanded_variant_targets(rows, stale)

    def test_equivalent_condition_disparity_rejects_conflicting_frontiers(self):
        geometries = [{"eye_width": 2, "eye_height": 1, "color_mode": mode}
                      for mode in ("sdr-srgb-8bit", "sdr-srgb-8bit")]

        def condition(name, ceiling, color_mode):
            render = []
            for index, geometry in enumerate(geometries):
                value = dict(geometry, eye_width=index + 1, color_mode=color_mode)
                render.append({"geometry": value, "safe_scale_max": ceiling})
            return {
                "film_id": "film", "clip": "shot", "frame": 0,
                "source_sha256": "a" * 64,
                "_input_variant_sha256": name,
                "safe_scale_ceiling": ceiling,
                "_render_variants": render,
            }

        rows = [
            condition("sdr", 1.3, "sdr-srgb-8bit"),
            condition("hdr", 1.2, "hdr-scrgb-fp16"),
        ]
        exact = [
            [torch.tensor([[0.01]]), torch.tensor([[0.02]])],
            [torch.tensor([[0.01]]), torch.tensor([[0.02]])],
        ]
        with self.assertRaisesRegex(RuntimeError, "conflicting condition targets"):
            train.validate_expanded_variant_frontiers(rows, exact)

        near = [
            exact[0],
            [torch.tensor([[0.01005]]), torch.tensor([[0.0201]])],
        ]
        with self.assertRaisesRegex(RuntimeError, "conflicting condition targets"):
            train.validate_expanded_variant_frontiers(rows, near)

        distinct = [
            exact[0],
            [torch.tensor([[0.02]]), torch.tensor([[0.04]])],
        ]
        train.validate_expanded_variant_frontiers(rows, distinct)

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

    def test_active_split_rejects_boolean_schema_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active, payload, _catalog, datasets = self.write_active_split(root)

            payload["schema"] = True
            active.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "active split manifest"):
                train.load_active_split(active)

            active, payload, _catalog, datasets = self.write_active_split(root)
            dataset = datasets["train_film"]
            dataset_payload = json.loads(dataset.read_text(encoding="utf-8"))
            dataset_payload["schema"] = True
            dataset.write_text(json.dumps(dataset_payload), encoding="utf-8")
            payload["productions"][0]["dataset_manifest_sha256"] = train.sha256(
                dataset
            )
            active.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "identity disagrees"):
                train.load_active_split(active)

            active, payload, _catalog, _datasets = self.write_active_split(root)
            payload["productions"][0]["dataset_manifest_schema"] = True
            active.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "identity disagrees"):
                train.load_active_split(active)

    def test_active_split_accepts_schema_2_monocular_dataset_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active, payload, _catalog, datasets = self.write_active_split(root)
            dataset = datasets["train_film"]
            dataset_payload = json.loads(dataset.read_text(encoding="utf-8"))
            dataset_payload.update({
                "schema": 2,
                "production_id": dataset_payload.pop("film_id"),
                "source_kind": "mono-video",
            })
            dataset.write_text(json.dumps(dataset_payload), encoding="utf-8")
            payload["productions"][0].update({
                "dataset_manifest_schema": 2,
                "source_kind": "mono-video",
                "dataset_manifest_sha256": train.sha256(dataset),
            })
            active.write_text(json.dumps(payload), encoding="utf-8")

            loaded, _identity = train.load_active_split(active)

            self.assertEqual(
                loaded["productions"][0]["source_kind"], "mono-video"
            )

            payload["productions"][0]["source_kind"] = "authored-stereo"
            active.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "identity disagrees"):
                train.load_active_split(active)

            payload["productions"][0]["source_kind"] = "mono-video"
            payload["productions"][0].pop("dataset_manifest_schema")
            active.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "identity disagrees"):
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

    def test_active_split_authenticates_native_hdr_video_collections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active, payload, _catalog, datasets = self.write_active_split(root)
            for row in payload["productions"]:
                production = row["production_id"]
                if production not in {"train_film", "dev_film"}:
                    continue
                split = row["split"]
                manifest, _, _ = (
                    split_contract_test.ArtisticDatasetSplitAuditTests.
                    write_native_manifest(root, production, split)
                )
                datasets[production] = manifest
                dataset = json.loads(manifest.read_text(encoding="utf-8"))
                identity = train.split_audit.native_hdr_source_identity(
                    dataset, manifest, production, verify_media=False
                )
                row.pop("video_sha256")
                row.update({
                    "dataset_manifest": str(manifest),
                    "dataset_manifest_schema": 2,
                    "dataset_manifest_sha256": train.sha256(manifest),
                    "source_kind": "native-hdr-video",
                    **identity,
                })
            active.write_text(json.dumps(payload), encoding="utf-8")

            loaded, _ = train.load_active_split(active)

            native = [
                row for row in loaded["productions"]
                if row.get("source_kind") == "native-hdr-video"
            ]
            self.assertEqual(len(native), 2)
            self.assertTrue(all(
                row["source_identity_kind"] ==
                train.split_audit.NATIVE_HDR_COLLECTION_IDENTITY_KIND
                for row in native
            ))

    def test_active_split_rejects_overlapping_native_capture_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active, payload, _catalog, _datasets = self.write_active_split(root)
            for row in payload["productions"]:
                production = row["production_id"]
                if production not in {"train_film", "dev_film"}:
                    continue
                split = row["split"]
                manifest, _, _ = (
                    split_contract_test.ArtisticDatasetSplitAuditTests.
                    write_native_manifest(
                        root, production, split,
                        video_id="shared_video",
                        capture_group="shared_capture",
                        media_bytes=b"shared",
                    )
                )
                dataset = json.loads(manifest.read_text(encoding="utf-8"))
                identity = train.split_audit.native_hdr_source_identity(
                    dataset, manifest, production, verify_media=False
                )
                row.pop("video_sha256")
                row.update({
                    "dataset_manifest": str(manifest),
                    "dataset_manifest_schema": 2,
                    "dataset_manifest_sha256": train.sha256(manifest),
                    "source_kind": "native-hdr-video",
                    **identity,
                })
            active.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "multiple productions"):
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

    def test_action_coverage_requires_both_actionable_and_identity_shots(self):
        sdr = train.input_color.sdr_input_variant()
        hdr = train.input_color.windows_hdr_input_variant(1000)

        def row(clip, ceiling, variant):
            return {
                "film_id": "film",
                "clip": clip,
                "frame": 0,
                "safe_scale_ceiling": ceiling,
                "_input_variant": variant,
                "_input_variant_sha256":
                    train.input_color.input_variant_sha256(variant),
            }

        with self.assertRaisesRegex(RuntimeError, "no actionable"):
            train.validate_action_coverage([
                row("sdr_identity", 1.0, sdr),
                row("hdr_identity", 1.0, hdr),
            ], "training")
        with self.assertRaisesRegex(RuntimeError, "no identity"):
            train.validate_action_coverage([
                row("sdr_action", 1.2, sdr),
                row("hdr_action", 1.2, hdr),
            ], "training")
        train.validate_action_coverage(
            [
                row("sdr_identity", 1.0, sdr),
                row("sdr_action", 1.2, sdr),
                row("hdr_identity", 1.0, hdr),
                row("hdr_action", 1.2, hdr),
            ], "training"
        )

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

    def test_acceptance_classifies_input_conditions_independently(self):
        rows = [
            {"film_id": "film", "clip": "shot", "frame": frame,
             "_input_variant_sha256": variant}
            for frame in (0, 1) for variant in ("bright", "dim")
        ]
        target = np.asarray([[1.2, 1.0]] * 4)
        predicted = np.asarray([
            [1.2, 0.9], [1.0, 0.1],
            [1.2, 0.9], [1.0, 0.1],
        ])

        metrics = train.film_balanced_acceptance(
            predicted, target, rows
        )

        self.assertEqual(
            metrics["macro"]["first_frame_action_recall_pct"], 50.0
        )
        self.assertAlmostEqual(
            metrics["macro"]["first_frame_effective_scale_mae_pct"], 10.0
        )
        self.assertAlmostEqual(
            metrics["macro"]["within_shot_scale_std_pct"], 0.0
        )

    def test_acceptance_reduces_temporal_risk_within_each_fixed_variant(self):
        rows = [
            {"film_id": "film", "clip": "shot", "frame": frame,
             "_input_variant_sha256": variant}
            for frame in (0, 1) for variant in ("stable", "unstable")
        ]
        target = np.asarray([[1.2, 1.0]] * 4)
        predicted = np.asarray([
            [1.2, 0.9], [1.2, 0.9],
            [1.2, 0.9], [1.0, 0.1],
        ])

        metrics = train.film_balanced_acceptance(predicted, target, rows)

        self.assertAlmostEqual(
            metrics["macro"]["within_shot_scale_std_pct"], 5.0
        )
        self.assertAlmostEqual(
            metrics["macro"]["within_shot_action_flip_pct"], 25.0
        )
        self.assertEqual(
            metrics["macro"]["first_frame_action_recall_pct"], 100.0
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

    def test_runtime_regime_acceptance_is_fail_closed_and_reports_whites(self):
        def measurement(regime, white, clip, actionable, improved=True):
            target_scale = 1.2 if actionable else 1.0
            prediction_scale = target_scale if improved else 1.0
            prediction_confidence = 0.9 if actionable and improved else 0.02
            trained = {}
            neutral = {}
            for key, _label, _unit in evaluate.ALL_METRICS:
                if key in ("actionable_scale_mae_pct", "action_miss_pct"):
                    trained[key] = (0.0 if actionable else None)
                    neutral[key] = (20.0 if key == "actionable_scale_mae_pct"
                                    and actionable else
                                    100.0 if actionable else None)
                elif key == "rendered_disparity_mae_pct":
                    trained[key] = 0.0 if improved else (1.0 if actionable else 0.0)
                    neutral[key] = 1.0 if actionable else 0.0
                elif key == "identity_false_action_pct":
                    trained[key] = 0.0 if not actionable else None
                    neutral[key] = 0.0 if not actionable else None
                elif key == "action_brier":
                    trained[key] = ((prediction_confidence - float(actionable)) ** 2)
                    neutral[key] = ((0.02 - float(actionable)) ** 2)
                else:
                    trained[key] = (abs(prediction_scale - target_scale) * 100.0)
                    neutral[key] = (abs(1.0 - target_scale) * 100.0)
            return {
                "film_id": "film", "clip": clip, "domain": "cinema",
                "frame": 0, "runtime_regime": regime,
                "hdr_white_level_raw": white,
                "input_variant_sha256": f"{regime}-{white}",
                "prediction": {
                    "scale": prediction_scale,
                    "confidence": prediction_confidence,
                    "rendered_disparity_mean_abs_pct": prediction_scale,
                },
                "target": {
                    "scale": target_scale,
                    "confidence": float(actionable),
                    "rendered_disparity_mean_abs_pct": target_scale,
                },
                "trained": trained, "neutral": neutral,
            }

        rows = [
            measurement("sdr", None, "action_a", True),
            measurement("sdr", None, "action_b", True),
            measurement("sdr", None, "identity", False),
        ]
        for white in (1000, 2500, 6000):
            rows.extend((
                measurement("hdr", white, "action_a", True),
                measurement("hdr", white, "action_b", True),
                measurement("hdr", white, "identity", False),
            ))
        result = evaluate.evaluate_runtime_regimes(
            rows, (1000, 2500, 6000), minimum_films=1,
            require_identity_guard=False,
        )
        self.assertTrue(result["accepted"], result)
        self.assertEqual(
            set(result["hdr_by_white_level_raw"]), {"1000", "2500", "6000"}
        )
        self.assertEqual(
            result["regimes"]["sdr"]["primary"]["variant_sample_count"], 3
        )
        self.assertEqual(
            result["regimes"]["hdr"]["primary"]["variant_sample_count"], 9
        )

        missing = evaluate.evaluate_runtime_regimes(
            [row for row in rows if row["hdr_white_level_raw"] != 6000],
            (1000, 2500, 6000), minimum_films=1,
            require_identity_guard=False,
        )
        self.assertFalse(missing["accepted"])
        self.assertEqual(missing["missing_hdr_white_levels_raw"], [6000])

        failed_rows = [dict(row) for row in rows]
        for row in failed_rows:
            if row["runtime_regime"] == "hdr" and "action" in row["clip"]:
                row["prediction"] = dict(row["prediction"], scale=1.0,
                                         confidence=0.02)
                row["trained"] = dict(row["neutral"])
        failed = evaluate.evaluate_runtime_regimes(
            failed_rows, (1000, 2500, 6000), minimum_films=1,
            require_identity_guard=False,
        )
        self.assertTrue(failed["regime_pass"]["sdr"])
        self.assertFalse(failed["regime_pass"]["hdr"])
        self.assertFalse(failed["accepted"])

    def test_hdr_worst_variant_reduces_neutral_and_trained_metrics(self):
        rows = []
        for trained, neutral in ((9.0, 2.0), (3.0, 8.0)):
            metrics_trained = {
                key: trained for key, _label, _unit in evaluate.ALL_METRICS
            }
            metrics_neutral = {
                key: neutral for key, _label, _unit in evaluate.ALL_METRICS
            }
            rows.append({
                "film_id": "film", "clip": "shot", "frame": 0,
                "input_variant_sha256": str(trained),
                "prediction": {
                    "scale": 1.0, "confidence": 0.0,
                    "rendered_disparity_mean_abs_pct": trained,
                },
                "target": {
                    "scale": 1.0, "confidence": 0.0,
                    "rendered_disparity_mean_abs_pct": neutral,
                },
                "trained": metrics_trained, "neutral": metrics_neutral,
            })
        collapsed = evaluate.worst_input_variant_measurements(rows)[0]
        self.assertEqual(collapsed["trained"]["effective_scale_mae_pct"], 9.0)
        self.assertEqual(collapsed["neutral"]["effective_scale_mae_pct"], 2.0)


if __name__ == "__main__":
    unittest.main()
