#!/usr/bin/env python3

import copy
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import artistic_geometry_contract as geometry
import artistic_policy_evaluation_contract as evaluation_contract
import promote_artistic_policy as promote
from test_artistic_policy_evaluation_contract import accepted_group


class ArtisticPolicyPromotionTests(unittest.TestCase):
    def test_host_runtime_accepts_current_export_and_approval_contracts(self):
        source = (
            promote.REPO_ROOT / "src" / "video_depth_estimator.cpp"
        ).read_text(encoding="utf-8")
        expected = (
            f'metadata.value("schema", 0) != '
            f'{promote.EXPORT_METADATA_SCHEMA}',
            f'approval->value("contract", "") != '
            f'"{promote.SEALED_APPROVAL_CONTRACT}"',
            f'approval->value("evaluation_schema", 0) != '
            f'{promote.SEALED_EVALUATION_SCHEMA}',
            f'{{"harness_schema", {promote.HARNESS_SCHEMA}}}',
            f'gate->value("harness_schema", 0) != '
            f'{promote.HARNESS_SCHEMA}',
        )
        for contract in expected:
            with self.subTest(contract=contract):
                self.assertIn(contract, source)
        evaluator = (
            promote.SBSBENCH_DIR / "run_eval.py"
        ).read_text(encoding="utf-8")
        self.assertIn(f"EVAL_SCHEMA = {promote.EVAL_SCHEMA}", evaluator)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.onnx = self.root / "depth_anything_v2_artistic.onnx"
        self.reference = self.root / "depth_anything_v2_fp16.onnx"
        self.checkpoint = self.root / "policy.pt"
        self.image = self.root / "frame.png"
        self.onnx.write_bytes(b"candidate-onnx")
        self.reference.write_bytes(b"reference-onnx")
        self.checkpoint.write_bytes(b"policy-checkpoint")
        if not cv2.imwrite(
                str(self.image), np.zeros((1080, 1920, 3), dtype=np.uint8)):
            raise RuntimeError("cannot create promotion-test image")

        current_contracts = promote.current_contract_identities()
        self.metric = current_contracts["metric_sha256"]
        self.warp = current_contracts["policy_warp_source_sha256"]
        self.active_split = "3" * 64
        self.fitter = "4" * 64
        self.test_labels = "5" * 64
        self.geometry_allowlist = geometry.build_allowlist([{
            "source_width": 1920,
            "source_height": 1080,
            "model_input_width": 770,
            "model_input_height": 434,
            "depth_short_side": 432,
            "depth_max_aspect": 4.0,
            "eye_width": 1920,
            "eye_height": 1080,
            "content_scale_x": 1.0,
            "content_scale_y": 1.0,
            "disparity_raster_width": 1920,
            "disparity_raster_height": 1080,
            "color_mode": geometry.COLOR_MODE_SDR,
        }])
        self.geometry_hash = geometry.allowlist_sha256(
            self.geometry_allowlist
        )
        self.input_manifest = promote.label_merge.build_input_variant_manifest([
            promote.input_color.sdr_input_variant(),
            *(promote.input_color.windows_hdr_input_variant(value)
              for value in (1000, 2500, 6000)),
        ])
        self.input_manifest_hash = (
            promote.label_merge.input_variant_manifest_sha256(
                self.input_manifest
            )
        )
        self.color_contract_hash = promote.input_color.color_contract_sha256()
        self.condition_target_contract = (
            promote.label_merge.CONDITION_TARGET_CONTRACT
        )
        self.baseline = {
            "profile": "apollo",
            "depth_model": self.reference.stem,
            "pop_strength": 1.25,
            "adaptive_pop": True,
            "adaptive_pop_max": 1.3,
            "ema": 0.5,
            "ema_edge_change": 0.05,
            "ema_edge_gradient": 0.02,
            "ema_edge_strength": 0.25,
            "minmax_ema": 0.18,
            "subject_lock": 0.5,
            "subject_recenter": 0.35,
            "subject_stretch": True,
            "depth_short_side": 432,
            "depth_max_aspect": 4.0,
            "zero_plane": "legacy",
            "depth_step": "current-once",
            "depth_compensation": "none",
            "literal_bestv2": False,
            "harness_schema": promote.HARNESS_SCHEMA,
            "eval_schema": promote.EVAL_SCHEMA,
            "warp_contract": promote.POLICY_WARP_CONTRACT,
            "metric_sha256": self.metric,
            "policy_warp_source_sha256": self.warp,
        }
        self.unsafe_overshoot = {
            "maximum_scale": 0.04,
            "maximum_limit_scale": 0.05,
            "film_balanced_mean_scale": 0.008,
            "film_balanced_mean_limit_scale": 0.01,
            "film_balanced_overshoot_rate_pct": 25.0,
            "by_film_mean_scale": {
                "sealed-film-a": 0.006, "sealed-film-b": 0.01,
            },
            "by_film_overshoot_rate_pct": {
                "sealed-film-a": 0.0, "sealed-film-b": 50.0,
            },
            "maximum_pass": True,
            "film_balanced_mean_pass": True,
        }
        self.runtime_acceptance = {
            "contract": evaluation_contract.RUNTIME_REGIME_ACCEPTANCE_CONTRACT,
            "condition_target_contract": self.condition_target_contract,
            "hdr_aggregation_contract": evaluation_contract.HDR_AGGREGATION_CONTRACT,
            "required_regimes": ["sdr", "hdr"],
            "expected_hdr_white_levels_raw": [1000, 2500, 6000],
            "missing_regimes": [],
            "missing_hdr_white_levels_raw": [],
            "unexpected_hdr_white_levels_raw": [],
            "incomplete_source_frame_count": 0,
            "source_condition_coverage_complete": True,
            "hdr_white_pass": {"1000": True, "2500": True, "6000": True},
            "hdr_aggregate_pass": True,
            "regime_pass": {"sdr": True, "hdr": True},
            "accepted": True,
        }
        self.evaluation = self.root / "evaluation.json"
        self.write_json(self.evaluation, {
            "schema": promote.SEALED_EVALUATION_SCHEMA,
            "split": "test",
            "checkpoint_sha256": promote.sha256(self.checkpoint),
            "active_split_sha256": self.active_split,
            "metric_sha256": self.metric,
            "label_fitter_identity_sha256": self.fitter,
            "test_labels_sha256": self.test_labels,
            "deployment_geometry_allowlist": self.geometry_allowlist,
            "deployment_geometry_allowlist_sha256": self.geometry_hash,
            "input_variant_manifest": self.input_manifest,
            "input_variant_manifest_sha256": self.input_manifest_hash,
            "depth_input_color_contract_sha256": self.color_contract_hash,
            "condition_target_contract": self.condition_target_contract,
            "val_films": ["sealed-film-a", "sealed-film-b"],
            "unsafe_ceiling_overshoot": self.unsafe_overshoot,
            "runtime_regime_evaluation": {
                **self.runtime_acceptance,
                "incomplete_source_frames": [],
                "regimes": {
                    "sdr": accepted_group(),
                    "hdr": accepted_group(
                        samples=6, unique=2, shots=2, shot_conditions=6,
                        actionable_samples=3, actionable_shots=3,
                    ),
                },
                "hdr_by_white_level_raw": {
                    "1000": accepted_group(),
                    "2500": accepted_group(),
                    "6000": accepted_group(),
                },
            },
            "decision": {
                "accepted": True,
                "overall_diagnostic_accepted": True,
                "runtime_regime_acceptance": self.runtime_acceptance,
                "guards": {
                    "unsafe_ceiling_maximum": True,
                    "unsafe_ceiling_film_balanced_mean": True,
                    "runtime_regimes_present": True,
                    "hdr_white_levels_present": True,
                    "hdr_white_levels_accepted": True,
                    "no_unexpected_hdr_white_levels": True,
                    "source_condition_coverage_complete": True,
                    "condition_target_contract": True,
                    "sdr_and_hdr_accepted": True,
                },
                "unsafe_overshoot_guard_required": True,
                "unsafe_ceiling_overshoot": self.unsafe_overshoot,
            },
        })
        self.metadata = self.root / "depth_anything_v2_artistic.json"
        approval = {
            "contract": promote.SEALED_APPROVAL_CONTRACT,
            "evaluation_sha256": promote.sha256(self.evaluation),
            "evaluation_schema": promote.SEALED_EVALUATION_SCHEMA,
            "split": "test",
            "decision_accepted": True,
            "checkpoint_sha256": promote.sha256(self.checkpoint),
            "active_split_sha256": self.active_split,
            "metric_sha256": self.metric,
            "label_fitter_identity_sha256": self.fitter,
            "test_labels_sha256": self.test_labels,
            "deployment_geometry_allowlist_sha256": self.geometry_hash,
            "input_variant_manifest_sha256": self.input_manifest_hash,
            "depth_input_color_contract_sha256": self.color_contract_hash,
            "sealed_test_productions": ["sealed-film-a", "sealed-film-b"],
            "unsafe_ceiling_overshoot": self.unsafe_overshoot,
            "runtime_regime_acceptance": self.runtime_acceptance,
            "condition_target_contract": self.condition_target_contract,
        }
        self.write_json(self.metadata, {
            "schema": promote.EXPORT_METADATA_SCHEMA,
            "deployed_model": self.onnx.stem,
            "base_depth_model": self.reference.stem,
            "onnx_sha256": promote.sha256(self.onnx),
            "depth_weights_sha256": "6" * 64,
            "metric_sha256": self.metric,
            "evaluation_sha256": promote.sha256(self.evaluation),
            "policy_contract": promote.POLICY_CONTRACT,
            "policy_feature_contract": promote.POLICY_FEATURE_CONTRACT,
            "input": {
                "name": "pixel_values",
                "dtype": "float32",
                "shape": [1, 3, "H", "W"],
            },
            "outputs": {
                "predicted_depth": {
                    "dtype": "float32",
                    "shape": [1, "H", "W"],
                },
                "artistic_global": {
                    "dtype": "float32",
                    "shape": [1, 2],
                    "channels": [
                        "safe_scale_ceiling",
                        "safe_ceiling_confidence",
                    ],
                },
            },
            "output_semantics": copy.deepcopy(
                promote.POLICY_OUTPUT_SEMANTICS
            ),
            "bounds": {"scale_delta_max": 0.5},
            "runtime": copy.deepcopy(promote.POLICY_RUNTIME),
            "approval_contract": approval,
            "policy_baseline": self.baseline,
            "deployment_geometry_allowlist": self.geometry_allowlist,
            "deployment_geometry_allowlist_sha256": self.geometry_hash,
            "input_variant_manifest": self.input_manifest,
            "input_variant_manifest_sha256": self.input_manifest_hash,
            "depth_input_color_contract_sha256": self.color_contract_hash,
            "condition_target_contract": self.condition_target_contract,
        })
        self.neutrality = self.root / "neutrality.json"
        self.write_json(self.neutrality, self.neutrality_payload())
        self.core = self.root / "core-results.json"
        self.extended = self.root / "extended-results.json"
        self.balanced_core = self.root / "balanced-core-results.json"
        self.balanced_extended = self.root / "balanced-extended-results.json"
        self.write_json(self.core, self.gate_payload("core", "immersive"))
        self.write_json(self.extended, self.gate_payload("extended", "immersive"))
        self.write_json(self.balanced_core, self.gate_payload("core", "balanced"))
        self.write_json(
            self.balanced_extended, self.gate_payload("extended", "balanced")
        )
        self.headset = {
            "approved": True,
            "reviewer": "human-reviewer",
            "device": "Galaxy XR",
            "resolution": "3840x1080",
            "refresh_hz": 90.0,
            "color_mode": geometry.COLOR_MODE_SDR,
            "geometry_index": 0,
            "notes": "Immersive style passed comfort and artifact review.",
        }

    @staticmethod
    def write_json(path, payload):
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    def neutrality_payload(self):
        return {
            "schema": 4,
            "preprocessing_contract": promote.NEUTRALITY_CONTRACT,
            "preprocessing": {
                "depth_short_side": 432,
                "depth_max_aspect": 4.0,
                "max_width": 1008,
                "max_height": 1008,
                "resize_interpolation": "opencv-inter-linear",
                "color_conversion": "opencv-bgr8-to-rgb-srgb",
            },
            "reference": {
                "path": str(self.reference),
                "sha256": promote.sha256(self.reference),
            },
            "candidate": {
                "path": str(self.onnx),
                "sha256": promote.sha256(self.onnx),
            },
            "limits": {
                "production_normalized_mean_abs": 1.0 / 1024.0,
                "production_normalized_p99_abs": 2.0 / 1024.0,
            },
            "passed": True,
            "images": [{
                "image": str(self.image),
                "image_sha256": promote.sha256(self.image),
                "source_width": 1920,
                "source_height": 1080,
                "input_shape": [1, 3, 434, 770],
                "production_normalized": {"mean_abs": 0.0, "p99_abs": 0.0},
                "passed": True,
            }],
        }

    def gate_payload(self, suite, style="immersive"):
        metadata_hash = promote.sha256(self.metadata)
        common = {
            **{
                key: self.baseline[key]
                for key in promote.BASELINE_RESULT_FIELDS
            },
            "model": self.onnx.stem,
            "metric_sha256": self.metric,
            "policy_warp_source_sha256": self.warp,
            "model_onnx_sha256": promote.sha256(self.onnx),
            "policy_metadata_sha256": metadata_hash,
            "deployment_geometry_allowlist_sha256": self.geometry_hash,
            "source_width": 1920,
            "source_height": 1080,
            "model_input_width": 770,
            "model_input_height": 434,
            "eye_width": 1920,
            "eye_height": 1080,
            "content_scale_x": 1.0,
            "content_scale_y": 1.0,
            "disparity_raster_width": 1920,
            "disparity_raster_height": 1080,
            "color_mode": geometry.COLOR_MODE_SDR,
            "metric_preview_encoding": promote.sbs_contract.METRIC_PREVIEW_SDR,
            "artistic_policy": True,
            "artistic_policy_consumed": True,
            "artistic_policy_authorization": "candidate-evaluation",
            "artistic_style": style,
            "artistic_scale_override": 0,
            "depth_reuse_interval": 1,
            "output_interval": 1,
            "output_gt_right_only": False,
        }
        clip_meta = {
            **common,
            "harness_schema": promote.HARNESS_SCHEMA,
            "artifact_mode": "full",
            "warp_disparity": promote.EXACT_DISPARITY_CONTRACT,
            "warp_unclamped_disparity": promote.UNCLAMPED_DISPARITY_CONTRACT,
            "artistic_disparity_contract": promote.ARTISTIC_DISPARITY_CONTRACT,
            "clip_sha1": "7" * 12,
        }
        return {
            "meta": {
                **common,
                "run_kind": "policy_candidate_gate",
                "suite": suite,
                "eval_schema": promote.EVAL_SCHEMA,
                "gpu_contention": False,
                "timestamp": "2026-07-15T12:00:00",
                "clip_set_sha1": {"shot": "7" * 12},
                "baseline_identities": {"shot": "8" * 64},
            },
            "verdict": "pass",
            "regressions": [],
            "hard_failures": [],
            "issues": [],
            "clips": {"shot": {"meta": clip_meta, "aggregate": {}}},
        }

    def build(self, **changes):
        arguments = {
            "onnx": self.onnx,
            "metadata": self.metadata,
            "checkpoint": self.checkpoint,
            "evaluation": self.evaluation,
            "reference_depth_onnx": self.reference,
            "neutrality_report": self.neutrality,
            "core_results": self.core,
            "extended_results": self.extended,
            "balanced_core_results": self.balanced_core,
            "balanced_extended_results": self.balanced_extended,
            "headset_review": self.headset,
            "expected_core_clips": {"shot"},
            "expected_extended_clips": {"shot"},
            "expected_neutrality_images": {
                "shot": promote.sha256(self.image)
            },
        }
        arguments.update(changes)
        return promote.build_manifest(**arguments)

    def replace_allowlist(self, allowlist):
        self.geometry_allowlist = allowlist
        self.geometry_hash = geometry.allowlist_sha256(allowlist)
        evaluation = json.loads(self.evaluation.read_text(encoding="utf-8"))
        evaluation["deployment_geometry_allowlist"] = allowlist
        evaluation["deployment_geometry_allowlist_sha256"] = self.geometry_hash
        self.write_json(self.evaluation, evaluation)
        metadata = json.loads(self.metadata.read_text(encoding="utf-8"))
        metadata["deployment_geometry_allowlist"] = allowlist
        metadata["deployment_geometry_allowlist_sha256"] = self.geometry_hash
        metadata["evaluation_sha256"] = promote.sha256(self.evaluation)
        metadata["approval_contract"]["evaluation_sha256"] = promote.sha256(
            self.evaluation
        )
        metadata["approval_contract"][
            "deployment_geometry_allowlist_sha256"
        ] = self.geometry_hash
        self.write_json(self.metadata, metadata)
        self.write_json(self.core, self.gate_payload("core"))
        self.write_json(self.extended, self.gate_payload("extended"))
        self.write_json(self.balanced_core, self.gate_payload("core", "balanced"))
        self.write_json(
            self.balanced_extended, self.gate_payload("extended", "balanced")
        )

    def test_manifest_binds_every_approval_identity(self):
        manifest = self.build()
        self.assertTrue(manifest["approved"])
        self.assertEqual(manifest["stage"], "production")
        self.assertEqual(manifest["contract"], promote.DEPLOYMENT_CONTRACT)
        self.assertEqual(
            manifest["model"]["onnx_sha256"], promote.sha256(self.onnx)
        )
        self.assertEqual(
            manifest["neutrality"]["report_sha256"],
            promote.sha256(self.neutrality),
        )
        self.assertEqual(
            manifest["render_gates"]["core"]["results_sha256"],
            promote.sha256(self.core),
        )
        self.assertEqual(
            manifest["render_gates"]["extended"]["results_sha256"],
            promote.sha256(self.extended),
        )
        self.assertEqual(
            manifest["render_gates"]["balanced_core"]["results_sha256"],
            promote.sha256(self.balanced_core),
        )
        self.assertEqual(manifest["headset_review"]["style"], "immersive")
        self.assertEqual(
            manifest["headset_review"]["deployment_geometry"],
            self.geometry_allowlist["tuples"][0],
        )
        self.assertEqual(
            manifest["deployment_geometry_coverage"],
            self.geometry_allowlist["tuples"],
        )
        for gate_name in ("core", "extended", "balanced_core", "balanced_extended"):
            gate = manifest["render_gates"][gate_name]
            self.assertEqual(gate["timestamp"], "2026-07-15T12:00:00")
            self.assertEqual(gate["clip_set_sha1"], {"shot": "7" * 12})
            self.assertEqual(
                gate["observed_deployment_geometries"],
                self.geometry_allowlist["tuples"],
            )

    def test_staged_manifest_allows_live_review_without_claiming_approval(self):
        manifest = self.build(stage="headset-review", headset_review=None)
        self.assertFalse(manifest["approved"])
        self.assertEqual(manifest["stage"], "headset-review")
        self.assertNotIn("headset_review", manifest)
        self.assertEqual(
            set(manifest["render_gates"]),
            {"core", "extended", "balanced_core", "balanced_extended"},
        )

    def test_manifest_shape_matches_the_fail_closed_runtime_contract(self):
        staged = self.build(stage="headset-review", headset_review=None)
        production = self.build()
        common_fields = {
            "schema", "contract", "stage", "approved", "created_at", "model",
            "neutrality", "render_gates", "deployment_geometry_coverage",
        }
        self.assertEqual(set(staged), common_fields)
        self.assertEqual(set(production), common_fields | {"headset_review"})
        self.assertEqual(set(production["model"]), {
            "deployed_model", "base_depth_model", "onnx_sha256",
            "metadata_sha256", "checkpoint_sha256", "evaluation_sha256",
            "metric_sha256", "policy_warp_source_sha256",
            "active_split_sha256", "label_fitter_identity_sha256",
            "test_labels_sha256", "deployment_geometry_allowlist",
            "deployment_geometry_allowlist_sha256", "input_variant_manifest",
            "input_variant_manifest_sha256",
            "depth_input_color_contract_sha256", "condition_target_contract",
            "sealed_test_productions",
        })
        self.assertEqual(set(production["neutrality"]), {
            "report_sha256", "reference_model", "reference_onnx_sha256",
            "candidate_onnx_sha256", "preprocessing_contract", "limits",
            "canonical_core_first_frames", "evidence_image_count",
        })
        gate_fields = {
            "results_sha256", "suite", "artistic_style", "verdict",
            "eval_schema", "harness_schema", "metric_sha256",
            "policy_warp_source_sha256", "model_onnx_sha256",
            "policy_metadata_sha256", "deployment_geometry_allowlist_sha256",
            "artistic_policy_consumed", "artistic_policy_authorization",
            "timestamp", "clip_set_sha1", "baseline_identities",
            "observed_deployment_geometries", "metric_preview_encodings",
        }
        for gate in production["render_gates"].values():
            self.assertEqual(set(gate), gate_fields)
        self.assertEqual(set(production["headset_review"]), {
            "approved", "reviewer", "device", "resolution", "refresh_hz",
            "color_mode", "deployment_geometry_index", "deployment_geometry",
            "deployment_geometry_allowlist_sha256", "style", "notes",
            "reviewed_at",
        })

    def test_rejects_sidecar_contract_runtime_would_reject(self):
        original = json.loads(self.metadata.read_text(encoding="utf-8"))
        mutations = {
            "policy contract": lambda value: value.update(
                {"policy_contract": "stale-policy"}
            ),
            "policy feature contract": lambda value: value.update(
                {"policy_feature_contract": "stale-features"}
            ),
            "policy outputs contract": lambda value: value["outputs"][
                "artistic_global"
            ].update({"shape": [1, 3]}),
            "policy output semantics": lambda value: value[
                "output_semantics"
            ].update({"action_threshold": 0.25}),
            "policy bounds": lambda value: value["bounds"].update(
                {"scale_delta_max": 0.75}
            ),
            "policy runtime": lambda value: value["runtime"].update(
                {"inactive_ceiling": 1.1}
            ),
        }
        for expected, mutate in mutations.items():
            with self.subTest(expected=expected):
                payload = copy.deepcopy(original)
                mutate(payload)
                self.write_json(self.metadata, payload)
                with self.assertRaisesRegex(RuntimeError, expected):
                    self.build()
        self.write_json(self.metadata, original)

    def test_rejects_stale_runtime_schema_or_compiled_identity(self):
        original = json.loads(self.metadata.read_text(encoding="utf-8"))
        cases = (
            ("harness_schema", promote.HARNESS_SCHEMA - 1, "harness schema"),
            ("eval_schema", promote.EVAL_SCHEMA - 1, "evaluation schema"),
            ("policy_warp_source_sha256", "0" * 64,
             "current compiled warp contract"),
            ("metric_sha256", "0" * 16,
             "current compiled metric contract"),
        )
        for field, stale, expected in cases:
            with self.subTest(field=field):
                payload = copy.deepcopy(original)
                if field == "metric_sha256":
                    payload["metric_sha256"] = stale
                payload["policy_baseline"][field] = stale
                self.write_json(self.metadata, payload)
                with self.assertRaisesRegex(RuntimeError, expected):
                    self.build()
        self.write_json(self.metadata, original)

    def test_rejects_unconsumed_policy_or_rescored_render_gate(self):
        payload = json.loads(self.core.read_text(encoding="utf-8"))
        payload["clips"]["shot"]["meta"]["artistic_policy_consumed"] = False
        self.write_json(self.core, payload)
        with self.assertRaisesRegex(RuntimeError, "artistic_policy_consumed"):
            self.build()

        self.write_json(self.core, self.gate_payload("core"))
        payload = json.loads(self.core.read_text(encoding="utf-8"))
        payload["meta"]["artifact_metric_sha256"] = self.metric
        self.write_json(self.core, payload)
        with self.assertRaisesRegex(RuntimeError, "rescored"):
            self.build()

        self.write_json(self.core, self.gate_payload("core"))
        payload = json.loads(self.core.read_text(encoding="utf-8"))
        payload["meta"]["timestamp"] = "not-a-render-timestamp"
        self.write_json(self.core, payload)
        with self.assertRaisesRegex(RuntimeError, "ISO-8601 render timestamp"):
            self.build()

    def test_rejects_missing_or_wrong_metric_preview_provenance(self):
        for value in (None, promote.sbs_contract.METRIC_PREVIEW_HDR):
            payload = self.gate_payload("core")
            clip_meta = payload["clips"]["shot"]["meta"]
            if value is None:
                del clip_meta["metric_preview_encoding"]
            else:
                clip_meta["metric_preview_encoding"] = value
            self.write_json(self.core, payload)
            with self.assertRaisesRegex(RuntimeError, "metric preview encoding"):
                self.build()
        self.write_json(self.core, self.gate_payload("core"))

    def test_rejects_render_geometry_outside_the_allowlist(self):
        payload = self.gate_payload("core")
        clip_meta = payload["clips"]["shot"]["meta"]
        clip_meta.update({
            "source_width": 1280,
            "source_height": 720,
            "eye_width": 1280,
            "eye_height": 720,
            "disparity_raster_width": 1280,
            "disparity_raster_height": 720,
        })
        self.write_json(self.core, payload)
        with self.assertRaisesRegex(RuntimeError, "outside the allow-list"):
            self.build()

    def test_rejects_stale_reported_model_input_geometry(self):
        payload = self.gate_payload("core")
        payload["clips"]["shot"]["meta"]["model_input_width"] = 756
        self.write_json(self.core, payload)
        with self.assertRaisesRegex(RuntimeError, "stale model-input dimensions"):
            self.build()

    def test_requires_fresh_render_coverage_of_every_allowed_geometry(self):
        other = {
            "source_width": 1280,
            "source_height": 720,
            "model_input_width": 770,
            "model_input_height": 434,
            "depth_short_side": 432,
            "depth_max_aspect": 4.0,
            "eye_width": 1280,
            "eye_height": 720,
            "content_scale_x": 1.0,
            "content_scale_y": 1.0,
            "disparity_raster_width": 1280,
            "disparity_raster_height": 720,
            "color_mode": geometry.COLOR_MODE_SDR,
        }
        self.replace_allowlist(geometry.build_allowlist([
            self.geometry_allowlist["tuples"][0], other,
        ]))
        with self.assertRaisesRegex(RuntimeError, "do not cover every"):
            self.build()

    def test_rejects_relaxed_or_stale_neutrality_evidence(self):
        payload = self.neutrality_payload()
        payload["limits"]["production_normalized_mean_abs"] = 0.5
        self.write_json(self.neutrality, payload)
        with self.assertRaisesRegex(RuntimeError, "relaxed"):
            self.build()

        self.write_json(self.neutrality, self.neutrality_payload())
        self.image.write_bytes(b"changed-after-neutrality")
        with self.assertRaisesRegex(RuntimeError, "bytes"):
            self.build()

    def test_rejects_nonpositive_neutrality_limit(self):
        for value in (0.0, -0.001):
            with self.subTest(value=value):
                payload = self.neutrality_payload()
                payload["limits"]["production_normalized_mean_abs"] = value
                self.write_json(self.neutrality, payload)
                with self.assertRaisesRegex(RuntimeError, "relaxed"):
                    self.build()

    def test_rejects_wrong_neutrality_preprocessing_or_input_shape(self):
        payload = self.neutrality_payload()
        payload["preprocessing"]["depth_short_side"] = 518
        self.write_json(self.neutrality, payload)
        with self.assertRaisesRegex(RuntimeError, "neutrality preprocessing"):
            self.build()

        payload = self.neutrality_payload()
        payload["images"][0]["input_shape"] = [1, 3, 518, 910]
        self.write_json(self.neutrality, payload)
        with self.assertRaisesRegex(RuntimeError, "shape is invalid"):
            self.build()

    def test_rejects_checkpoint_or_evaluation_identity_drift(self):
        self.checkpoint.write_bytes(b"different-checkpoint")
        with self.assertRaisesRegex(RuntimeError, "checkpoint_sha256"):
            self.build()

        self.checkpoint.write_bytes(b"policy-checkpoint")
        evaluation = copy.deepcopy(json.loads(
            self.evaluation.read_text(encoding="utf-8")
        ))
        evaluation["decision"]["accepted"] = False
        self.write_json(self.evaluation, evaluation)
        with self.assertRaisesRegex(RuntimeError, "evaluation_sha256"):
            self.build()

    def test_requires_explicit_headset_approval(self):
        review = dict(self.headset, approved=False)
        with self.assertRaisesRegex(RuntimeError, "headset review"):
            self.build(headset_review=review)

    def test_headset_review_must_match_an_approved_geometry(self):
        review = dict(self.headset, color_mode="hdr-pq-10bit")
        with self.assertRaisesRegex(RuntimeError, "headset color mode"):
            self.build(headset_review=review)

        review = dict(self.headset, resolution="5120x2160")
        with self.assertRaisesRegex(RuntimeError, "headset SBS resolution"):
            self.build(headset_review=review)

        review = dict(self.headset, geometry_index=1)
        with self.assertRaisesRegex(RuntimeError, "geometry index"):
            self.build(headset_review=review)


if __name__ == "__main__":
    unittest.main()
