#!/usr/bin/env python3
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest

import numpy as np

try:
    from . import subject_state_contract, whole_clip_raw_contract
    from .depth_coordinate_v2_contract import CALIBRATED_DEFAULTS, MODEL_CALIBRATIONS
    from .depth_mapping_v2_temporal import (
        _clip_sequence_input_contract,
        _load_cut_indices,
        _load_run_model_contract,
        _load_shape,
        _raw_files,
        generate_first_latch_exact_sequence,
    )
    from .replay_depth_mapping_v2_sequence import (
        IMPLEMENTATION_SOURCE_FILES,
        GPU_INPUT_MANIFEST_MODE, GPU_INPUT_MANIFEST_SCHEMA,
        PRODUCER_EVIDENCE_BINDING, PRODUCER_EVIDENCE_SCHEMA,
        SEQUENCE_CONTRACT_SCHEMA,
        LEGACY_STATE_METRICS, RENDERER_SCORECARD_FILE, RENDERER_SCORE_CONTRACT_FILE,
        RENDERER_SCORE_CONTRACT_SCHEMA, MappingV2Config,
        _diagnostic_summary,
        _load_authenticated_cut_pulses, _materialize_gpu_replay_inputs,
        _materialize_producer_evidence_bundle,
        _metric_contract_evidence,
        _publish_renderer_quality_score, _resolve_pop_strength, _source_frames,
        _validate_frame_source_attestation, _validate_gpu_input_manifest_evidence,
        _validate_metric_contract_evidence, _validate_producer_evidence_bundle)
except ImportError:  # Direct execution from tools/sbsbench.
    import subject_state_contract  # type: ignore
    import whole_clip_raw_contract  # type: ignore
    from depth_coordinate_v2_contract import (  # type: ignore
        CALIBRATED_DEFAULTS, MODEL_CALIBRATIONS)
    from depth_mapping_v2_temporal import (  # type: ignore
        _clip_sequence_input_contract,
        _load_cut_indices,
        _load_run_model_contract,
        _load_shape,
        _raw_files,
        generate_first_latch_exact_sequence,
    )
    from replay_depth_mapping_v2_sequence import (  # type: ignore
        IMPLEMENTATION_SOURCE_FILES,
        GPU_INPUT_MANIFEST_MODE, GPU_INPUT_MANIFEST_SCHEMA,
        PRODUCER_EVIDENCE_BINDING, PRODUCER_EVIDENCE_SCHEMA,
        SEQUENCE_CONTRACT_SCHEMA,
        LEGACY_STATE_METRICS, RENDERER_SCORECARD_FILE, RENDERER_SCORE_CONTRACT_FILE,
        RENDERER_SCORE_CONTRACT_SCHEMA, MappingV2Config,
        _diagnostic_summary,
        _load_authenticated_cut_pulses, _materialize_gpu_replay_inputs,
        _materialize_producer_evidence_bundle,
        _metric_contract_evidence,
        _publish_renderer_quality_score, _resolve_pop_strength, _source_frames,
        _validate_frame_source_attestation, _validate_gpu_input_manifest_evidence,
        _validate_metric_contract_evidence, _validate_producer_evidence_bundle)


RAW_STAGE = "raw model output before transform/normalization/EMA/curvature"


class DepthMappingV2SequenceReplayTest(unittest.TestCase):
    def test_diagnostic_summary_uses_frame_validity_without_adaptive_scale_fields(self):
        common = {
            "candidate_center_drift_u": 0.01,
            "predicted_zero_translation_source_u": 0.002,
            "effective_gain": 0.005,
            "conditioner_raised_fraction": 0.1,
            "conditioner_max_raise_source_u": 0.003,
            "conditioner_lowered_fraction": 0.02,
            "conditioner_max_lower_source_u": 0.001,
            "final_horizontal_slope_max": 0.2,
            "final_vertical_shear_max": 0.3,
            "collapsed": False,
            "confirmed_cut": False,
            "calibration_revision": 1,
        }
        active = dict(common, frame_valid=True, camera_valid=True)
        retained = dict(
            common, frame_valid=False, camera_valid=True, effective_gain=0.0,
            candidate_center_drift_u=0.0, predicted_zero_translation_source_u=0.0)
        cleared = dict(
            common, frame_valid=False, camera_valid=False, effective_gain=0.0,
            candidate_center_drift_u=0.0, predicted_zero_translation_source_u=0.0,
            collapsed=True, confirmed_cut=True)

        summary = _diagnostic_summary([active, retained, cleared])

        self.assertEqual(summary["role"], "non-controlling-fixed-scale-camera-audit-v4")
        self.assertEqual(summary["maximum_final_vertical_shear"], 0.3)
        self.assertEqual(summary["minimum_active_effective_gain"], 0.005)
        self.assertEqual(summary["flat_unusable_frames"], 2)
        self.assertEqual(summary["retained_camera_unusable_frames"], 1)
        self.assertEqual(summary["camera_clear_unusable_frames"], 1)
        self.assertEqual(summary["collapsed_frames"], 1)
        self.assertEqual(summary["confirmed_cuts"], 1)
        self.assertNotIn("maximum_abs_candidate_log_scale_drift", summary)
        self.assertNotIn("maximum_abs_candidate_scale_ratio_deviation", summary)

    def test_native_cut_compatibility_trace_has_a_distinct_fail_closed_role(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "subject_state.json"
            values = [0.0] * len(subject_state_contract.FIELDS)
            values[subject_state_contract.FIELDS.index("hard_cut_pulse")] = 1.0
            values[subject_state_contract.FIELDS.index("hard_cut_count")] = 4.0
            payload = {
                "schema": subject_state_contract.GPU_REPLAY_SCHEMA,
                "source": subject_state_contract.GPU_REPLAY_SOURCE,
                "capture": subject_state_contract.GPU_REPLAY_CAPTURE,
                "fields": list(subject_state_contract.FIELDS),
                "frames": [{"frame_id": "00001", "values": values}],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            trace = subject_state_contract.load_trace(str(path))
            self.assertEqual(subject_state_contract.trace_schema(str(path)), 3)
            self.assertEqual(trace[1]["hard_cut_pulse"], 1.0)
            self.assertEqual(trace[1]["hard_cut_count"], 4.0)
            self.assertEqual(
                subject_state_contract.contract_reference(3),
                {"file": "subject_state.json", "schema": 3,
                 "capture": subject_state_contract.GPU_REPLAY_CAPTURE},
            )

            payload["source"] = subject_state_contract.LEGACY_SOURCE
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid subject-state trace contract"):
                subject_state_contract.load_trace(str(path))

    def test_implementation_hash_surface_covers_transitive_gpu_and_renderer_sources(self):
        self.assertEqual(len(IMPLEMENTATION_SOURCE_FILES), len(set(IMPLEMENTATION_SOURCE_FILES)))
        self.assertTrue({
            "tools/sbsbench/contracts/depth-coordinate-v2-v1.json",
            "src/generated/depth_coordinate_v2_contract.h",
            "src_assets/windows/assets/shaders/directx/include/depth_coordinate_v2_contract.generated.hlsl",
            "src_assets/windows/assets/shaders/directx/include/depth_coordinate_v2.hlsl",
            "src_assets/windows/assets/shaders/directx/include/depth_constants.hlsl",
            "src_assets/windows/assets/shaders/directx/include/sbs_warp_common.hlsl",
            "src_assets/windows/assets/shaders/directx/sbs_reprojection_ps.hlsl",
            "src_assets/windows/assets/shaders/directx/sbs_forward_coverage_cs.hlsl",
            "tools/sbsbench/direct_geometry_contract.py",
            "tools/sbsbench/run_eval.py",
            "tools/sbsbench/subject_state_contract.py",
            "tools/sbsbench/whole_clip_raw_contract.py",
        }.issubset(IMPLEMENTATION_SOURCE_FILES))

    def _write_run(self, root: Path) -> tuple[Path, dict]:
        calibration = MODEL_CALIBRATIONS[0]
        digest_a = calibration.onnx_sha256
        digest_b = "2" * 64
        width, height = calibration.calibrated_input_shapes[0]
        clip = root / "unit_sequence_contract"
        clip.mkdir()
        producer_identity = {
            "schema": 1,
            "model": calibration.depth_model,
            "depth_model_url": calibration.depth_model_url,
            "onnx_sha256": calibration.onnx_sha256,
            "preprocess_profile": calibration.preprocess.profile,
            "preprocess_source_closure_sha256":
                calibration.preprocess.source_closure_sha256,
            "raw_width": width,
            "raw_height": height,
        }
        (clip / "contract.json").write_text(json.dumps({
            "schema": whole_clip_raw_contract.HARNESS_CONTRACT_SCHEMA,
            "model": calibration.depth_model,
            "pop_strength": 1.2,
            "depth_step": "current-once",
            "depth_reuse_interval": 1,
            "raw_model_provenance": producer_identity,
        }), encoding="utf-8")
        (clip / "raw_shape.json").write_text(json.dumps({
            "schema": whole_clip_raw_contract.RAW_SHAPE_SCHEMA,
            "width": width, "height": height, "dtype": "float32-le",
            "layout": "row-major", "stage": RAW_STAGE,
        }), encoding="utf-8")
        np.linspace(-1.0, 1.0, width * height, dtype="<f4").tofile(
            clip / "raw_00001.f32")
        (clip / "sbs_00001.png").write_bytes(b"sbs")
        (clip / "depth_00001.png").write_bytes(b"depth")
        np.zeros(width * height, dtype="<f4").tofile(clip / "warp_map_00001.f32")
        (clip / "subject_state.json").write_text(json.dumps({
            "schema": subject_state_contract.SCHEMA,
            "source": "depth_subject_resolve_cs.SubjectState",
            "capture": "every-source-frame-after-estimator-update",
            "fields": list(subject_state_contract.FIELDS),
            "frames": [{"frame_id": "00001", "values": [0.0] * len(
                subject_state_contract.FIELDS)}],
        }), encoding="utf-8")
        raw_manifest = whole_clip_raw_contract.build_manifest(clip, [1], {
            "model": calibration.depth_model,
            "depth_model_url": calibration.depth_model_url,
            "onnx_sha256": calibration.onnx_sha256,
            "preprocess_profile": calibration.preprocess.profile,
            "preprocess_source_closure_sha256":
                calibration.preprocess.source_closure_sha256,
            "calibration_id": calibration.calibration_id,
        })
        raw_summary = {
            "calibration_status": raw_manifest["calibration_status"],
            "calibration_id": raw_manifest["calibration_id"],
            "preprocess_profile": producer_identity["preprocess_profile"],
            "raw_shape": raw_manifest["raw_shape"],
        }
        (root / "results.json").write_text(json.dumps({
            "meta": {
                "eval_schema": whole_clip_raw_contract.EVALUATOR_SCHEMA,
                "model": calibration.depth_model,
                "depth_model_url": calibration.depth_model_url,
                "preprocess_profile": calibration.preprocess.profile,
                "preprocess_source_closure_sha256":
                    calibration.preprocess.source_closure_sha256,
                "depth_coordinate_v2_calibration_id": calibration.calibration_id,
                "depth_coordinate_v2_raw_shape": {"height": height, "width": width},
                "onnx_sha256": digest_a,
                "engine_name": "unit.engine",
                "engine_sha256": digest_b,
                "pop_strength": 1.2,
                whole_clip_raw_contract.RESULTS_META_KEY: {
                    clip.name: raw_manifest,
                },
            },
            "clips": {clip.name: {"meta": {"raw_model_identity": raw_summary}}},
        }), encoding="utf-8")
        return clip, _load_run_model_contract(root)

    def test_sequence_input_contract_binds_complete_ordered_artifact_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            clip, run_model = self._write_run(Path(temporary))
            shape = _load_shape(clip)
            evidence = _clip_sequence_input_contract(
                clip, [1], shape, run_model)
            self.assertEqual(evidence["selected_frame_ids"], ["00001"])
            self.assertEqual(
                evidence["model_hash_authority"], "schema-36-run-level-results-json")
            self.assertEqual(
                evidence["input_shape_authority"], "schema-36-per-clip-raw-manifest")
            self.assertEqual(evidence["eval_schema"], 36)
            self.assertEqual(evidence["contract_schema"], 18)
            self.assertEqual(evidence["raw_hash_authority"]["binding"],
                             whole_clip_raw_contract.BINDING)
            self.assertEqual(evidence["depth_model_url"], MODEL_CALIBRATIONS[0].depth_model_url)
            self.assertEqual(
                evidence["preprocess_profile"], MODEL_CALIBRATIONS[0].preprocess.profile)
            self.assertEqual(
                evidence["preprocess_source_closure_sha256"],
                MODEL_CALIBRATIONS[0].preprocess.source_closure_sha256)
            self.assertEqual(
                evidence["cut_control_evidence"]["kind"],
                "subject-state-hard-cut-generation")
            self.assertEqual(evidence["cut_expectation_evidence"], {
                "kind": "none", "sha256": None,
            })
            width, height = MODEL_CALIBRATIONS[0].calibrated_input_shapes[0]
            self.assertEqual(evidence["raw_shape"], {"height": height, "width": width})
            self.assertRegex(evidence["sequence_input_sha256"], r"^[0-9a-f]{64}$")

            contract_path = clip / "contract.json"
            original_contract = contract_path.read_text(encoding="utf-8")
            contract = json.loads(original_contract)
            contract["pop_strength"] = 1.3
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model/pop/schema"):
                _clip_sequence_input_contract(clip, [1], shape, run_model)
            contract_path.write_text(original_contract, encoding="utf-8")

            (clip / "warp_map_00001.f32").unlink()
            with self.assertRaisesRegex(ValueError, "complete output identities"):
                _clip_sequence_input_contract(clip, [1], shape, run_model)

    def test_run_model_contract_rejects_url_and_preprocess_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root)
            results_path = root / "results.json"
            original = json.loads(results_path.read_text(encoding="utf-8"))
            for key, replacement in (
                    ("depth_model_url", "https://example.invalid/same-bytes.onnx"),
                    ("preprocess_profile", "lookalike-preprocess-v1"),
                    ("preprocess_source_closure_sha256", "3" * 64)):
                with self.subTest(key=key):
                    payload = json.loads(json.dumps(original))
                    payload["meta"][key] = replacement
                    results_path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        _load_run_model_contract(root)
            results_path.write_text(json.dumps(original), encoding="utf-8")
            self.assertEqual(
                _load_run_model_contract(root)["calibration_id"],
                MODEL_CALIBRATIONS[0].calibration_id,
            )

    def test_run_and_clip_contract_schemas_and_capture_hashes_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clip, run_model = self._write_run(root)
            results_path = root / "results.json"
            original_results = results_path.read_text(encoding="utf-8")
            payload = json.loads(original_results)
            payload["meta"]["eval_schema"] -= 1
            results_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale evaluator schema"):
                _load_run_model_contract(root)
            results_path.write_text(original_results, encoding="utf-8")

            contract_path = clip / "contract.json"
            original_contract = contract_path.read_text(encoding="utf-8")
            contract = json.loads(original_contract)
            contract["schema"] -= 1
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model/pop/schema"):
                _clip_sequence_input_contract(clip, [1], _load_shape(clip), run_model)
            contract_path.write_text(original_contract, encoding="utf-8")

            raw_path = clip / "raw_00001.f32"
            with raw_path.open("r+b") as stream:
                stream.write(b"\x00\x00\x80?")
            with self.assertRaisesRegex(ValueError, "changed after evaluator capture"):
                _clip_sequence_input_contract(clip, [1], _load_shape(clip), run_model)

    def test_pop_strength_defaults_to_run_profile_and_attests_override(self):
        run_model = {"pop_strength": 1.2}
        selected, authority = _resolve_pop_strength(run_model, None)
        self.assertEqual(selected, 1.2)
        self.assertEqual(authority, {
            "source": "run-results-profile-config",
            "selected": 1.2,
            "run_configured": 1.2,
        })
        selected, authority = _resolve_pop_strength(run_model, 1.6)
        self.assertEqual(selected, 1.6)
        self.assertEqual(authority["source"], "explicit-cli-override")
        self.assertEqual(authority["run_configured"], 1.2)
        with self.assertRaisesRegex(ValueError, "between 0.25 and 2"):
            _resolve_pop_strength(run_model, 2.1)

    def test_raw_shape_and_raw_ids_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            clip = Path(temporary)
            (clip / "raw_shape.json").write_text(json.dumps({
                "schema": whole_clip_raw_contract.RAW_SHAPE_SCHEMA,
                "width": 4, "height": 2, "dtype": "float32-le",
                "layout": "row-major", "stage": RAW_STAGE,
                "unexpected": True,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing/unknown"):
                _load_shape(clip)

            (clip / "raw_1.f32").write_bytes(b"")
            (clip / "raw_00001.f32").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "duplicate numeric"):
                _raw_files(clip)

    def test_raw_manifest_requires_exact_shape_bytes_and_producer_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clip, run_model = self._write_run(root)
            manifest = run_model["whole_clip_raw_artifacts"][clip.name]

            shape_path = clip / "raw_shape.json"
            original_shape = shape_path.read_text(encoding="utf-8")
            shape = json.loads(original_shape)
            shape["layout"] = "column-major"
            shape_path.write_text(json.dumps(shape), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tensor semantics"):
                whole_clip_raw_contract.build_manifest(clip, [1], {
                    "model": MODEL_CALIBRATIONS[0].depth_model,
                    "depth_model_url": MODEL_CALIBRATIONS[0].depth_model_url,
                    "onnx_sha256": MODEL_CALIBRATIONS[0].onnx_sha256,
                    "preprocess_profile": MODEL_CALIBRATIONS[0].preprocess.profile,
                    "preprocess_source_closure_sha256":
                        MODEL_CALIBRATIONS[0].preprocess.source_closure_sha256,
                    "calibration_id": MODEL_CALIBRATIONS[0].calibration_id,
                })
            shape_path.write_text(original_shape, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "runtime identity"):
                whole_clip_raw_contract.build_manifest(clip, [1], {
                    "model": MODEL_CALIBRATIONS[0].depth_model,
                    "depth_model_url": MODEL_CALIBRATIONS[0].depth_model_url,
                    "onnx_sha256": MODEL_CALIBRATIONS[0].onnx_sha256,
                    "preprocess_profile": MODEL_CALIBRATIONS[0].preprocess.profile,
                    "preprocess_source_closure_sha256": "c" * 64,
                    "calibration_id": MODEL_CALIBRATIONS[0].calibration_id,
                })

            raw_path = clip / "raw_00001.f32"
            raw_path.write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "expected"):
                whole_clip_raw_contract.authenticate_manifest_files(clip, manifest, [1])

    def test_uncalibrated_empty_profile_and_local_url_are_authenticated_abstentions(self):
        with tempfile.TemporaryDirectory() as temporary:
            cases = (
                ("custom", "custom-local-model", "", ""),
                ("base", "depth_anything_v2_base_fp16",
                 "https://huggingface.co/onnx-community/depth-anything-v2-base/resolve/"
                 "main/onnx/model_fp16.onnx", ""),
            )
            for name, model, producer_url, evaluator_expected_url in cases:
                with self.subTest(model=model):
                    clip = Path(temporary) / name
                    clip.mkdir()
                    width, height = 4, 2
                    producer = {
                        "schema": 1, "model": model,
                        "depth_model_url": producer_url,
                        "onnx_sha256": "a" * 64, "preprocess_profile": "",
                        "preprocess_source_closure_sha256": "b" * 64,
                        "raw_width": width, "raw_height": height,
                    }
                    (clip / "contract.json").write_text(json.dumps({
                        "schema": whole_clip_raw_contract.HARNESS_CONTRACT_SCHEMA,
                        "raw_model_provenance": producer,
                    }), encoding="utf-8")
                    (clip / "raw_shape.json").write_text(json.dumps({
                        "schema": whole_clip_raw_contract.RAW_SHAPE_SCHEMA,
                        "width": width, "height": height, "dtype": "float32-le",
                        "layout": "row-major", "stage": whole_clip_raw_contract.RAW_STAGE,
                    }), encoding="utf-8")
                    (clip / "raw_00001.f32").write_bytes(b"\0" * (width * height * 4))
                    manifest = whole_clip_raw_contract.build_manifest(clip, [1], {
                        "model": model, "depth_model_url": evaluator_expected_url,
                        "onnx_sha256": producer["onnx_sha256"], "preprocess_profile": "",
                        "preprocess_source_closure_sha256":
                            producer["preprocess_source_closure_sha256"],
                        "calibration_id": None,
                    })
                    self.assertEqual(
                        manifest["calibration_status"],
                        "abstain-unsupported-model-contract")
                    self.assertEqual(
                        manifest["producer_model_identity"]["depth_model_url"], producer_url)
                    self.assertIsNone(manifest["calibration_id"])
                    whole_clip_raw_contract.authenticate_manifest_files(
                        clip, manifest, [1])

    def test_mixed_supported_and_unsupported_shapes_admit_only_supported_clip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            supported, _ = self._write_run(root)
            results_path = root / "results.json"
            results = json.loads(results_path.read_text(encoding="utf-8"))
            calibration = MODEL_CALIBRATIONS[0]
            width, height = 658, 434
            unsupported = root / "unsupported_shape"
            unsupported.mkdir()
            producer = {
                "schema": 1, "model": calibration.depth_model,
                "depth_model_url": calibration.depth_model_url,
                "onnx_sha256": calibration.onnx_sha256,
                "preprocess_profile": calibration.preprocess.profile,
                "preprocess_source_closure_sha256":
                    calibration.preprocess.source_closure_sha256,
                "raw_width": width, "raw_height": height,
            }
            (unsupported / "contract.json").write_text(json.dumps({
                "schema": whole_clip_raw_contract.HARNESS_CONTRACT_SCHEMA,
                "model": calibration.depth_model, "pop_strength": 1.2,
                "depth_step": "current-once", "depth_reuse_interval": 1,
                "raw_model_provenance": producer,
            }), encoding="utf-8")
            (unsupported / "raw_shape.json").write_text(json.dumps({
                "schema": whole_clip_raw_contract.RAW_SHAPE_SCHEMA,
                "width": width, "height": height, "dtype": "float32-le",
                "layout": "row-major", "stage": RAW_STAGE,
            }), encoding="utf-8")
            np.zeros(width * height, dtype="<f4").tofile(unsupported / "raw_00001.f32")
            (unsupported / "sbs_00001.png").write_bytes(b"sbs")
            (unsupported / "depth_00001.png").write_bytes(b"depth")
            np.zeros(width * height, dtype="<f4").tofile(
                unsupported / "warp_map_00001.f32")
            (unsupported / "subject_state.json").write_text(json.dumps({
                "schema": subject_state_contract.SCHEMA,
                "source": "depth_subject_resolve_cs.SubjectState",
                "capture": "every-source-frame-after-estimator-update",
                "fields": list(subject_state_contract.FIELDS),
                "frames": [{"frame_id": "00001", "values": [0.0] * len(
                    subject_state_contract.FIELDS)}],
            }), encoding="utf-8")
            manifest = whole_clip_raw_contract.build_manifest(unsupported, [1], {
                "model": calibration.depth_model,
                "depth_model_url": calibration.depth_model_url,
                "onnx_sha256": calibration.onnx_sha256,
                "preprocess_profile": calibration.preprocess.profile,
                "preprocess_source_closure_sha256":
                    calibration.preprocess.source_closure_sha256,
                "calibration_id": calibration.calibration_id,
            })
            self.assertEqual(manifest["calibration_status"], "abstain-unsupported-shape")
            results["meta"][whole_clip_raw_contract.RESULTS_META_KEY][unsupported.name] = manifest
            results["meta"]["depth_coordinate_v2_raw_shape"] = None
            results["clips"][unsupported.name] = {"meta": {"raw_model_identity": {
                "calibration_status": manifest["calibration_status"],
                "calibration_id": manifest["calibration_id"],
                "preprocess_profile": producer["preprocess_profile"],
                "raw_shape": manifest["raw_shape"],
            }}}
            results_path.write_text(json.dumps(results), encoding="utf-8")

            run_model = _load_run_model_contract(root)
            supported_evidence = _clip_sequence_input_contract(
                supported, [1], _load_shape(supported), run_model)
            self.assertEqual(
                supported_evidence["calibration_id"], calibration.calibration_id)
            with self.assertRaisesRegex(ValueError, "replay abstains for this clip"):
                _clip_sequence_input_contract(
                    unsupported, [1], _load_shape(unsupported), run_model)

    def test_source_frame_padding_alias_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "frame_1.png").write_bytes(b"one")
            (root / "frame_00001.jpg").write_bytes(b"alias")
            with self.assertRaisesRegex(ValueError, "duplicate source frame ID"):
                _source_frames(root)

    def test_cut_labels_never_control_replay_and_generation_recovers_missing_pulse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene_cut = root / "scene_cut"
            scene_cut.mkdir()
            # The committed scene_cut metadata expects a pulse, but labels are scoring-only. With
            # no live trace, the controller sees initialization and no invented cut.
            cuts, counts, source = _load_cut_indices(scene_cut, [1, 2])
            self.assertEqual(cuts, [True, False])
            self.assertEqual(counts, [0, 0])
            self.assertEqual(source, "first-frame-only")

            # Persistent generation remains authoritative even if the one-frame pulse is absent.
            values = [0.0] * len(subject_state_contract.FIELDS)
            pulse_index = subject_state_contract.FIELDS.index("hard_cut_pulse")
            count_index = subject_state_contract.FIELDS.index("hard_cut_count")
            frames = []
            for frame_id, count in ((1, 4), (2, 5), (3, 5)):
                row = list(values)
                row[pulse_index] = 0.0
                row[count_index] = float(count)
                frames.append({"frame_id": f"{frame_id:05d}", "values": row})
            (scene_cut / "subject_state.json").write_text(json.dumps({
                "schema": subject_state_contract.SCHEMA,
                "source": "depth_subject_resolve_cs.SubjectState",
                "capture": "every-source-frame-after-estimator-update",
                "fields": list(subject_state_contract.FIELDS),
                "frames": frames,
            }), encoding="utf-8")
            cuts, counts, source = _load_cut_indices(scene_cut, [1, 2, 3])
            self.assertEqual(cuts, [True, True, False])
            self.assertEqual(counts, [4, 5, 5])
            self.assertEqual(source, "subject-state-hard-cut-generation")
            self.assertEqual(
                _load_authenticated_cut_pulses(scene_cut, [1, 2, 3], counts),
                [False, False, False],
            )

            raw = np.tile(np.linspace(-1.0, 1.0, 8), (2, 1))
            mapped = generate_first_latch_exact_sequence(
                [raw, raw, raw],
                [1, 2, 3],
                cuts,
                source,
                confirmed_cut_counts=counts,
            )
            self.assertEqual(
                [row["confirmed_cut_count"] for row in mapped.state_trace["frames"]],
                [4, 5, 5],
            )

            malformed = root / "unit_malformed_trace"
            malformed.mkdir()
            (malformed / "subject_state.json").write_text(
                json.dumps({"schema": subject_state_contract.SCHEMA, "frames": []}),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid subject-state trace contract"):
                _load_cut_indices(malformed, [1])

    def test_gpu_manifest_materializes_current_color_and_preserves_count_only_cut(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            source_one = root / "source_one.png"
            source_two = root / "source_two.png"
            source_one.write_bytes(b"matched-first-color")
            source_two.write_bytes(b"current-second-color")
            raw_one_path = root / "raw_00001.f32"
            raw_two_path = root / "raw_00002.f32"
            width, height = MODEL_CALIBRATIONS[0].calibrated_input_shapes[0]
            raw_one = np.linspace(-1.0, 1.0, width * height, dtype="<f4").reshape(
                height, width)
            raw_two = np.zeros_like(raw_one)
            raw_one.astype("<f4").tofile(raw_one_path)
            raw_two.astype("<f4").tofile(raw_two_path)
            calibration = MODEL_CALIBRATIONS[0]
            run_model = {
                "calibration_id": calibration.calibration_id,
                "model": calibration.depth_model,
                "depth_model_url": calibration.depth_model_url,
                "onnx_sha256": calibration.onnx_sha256,
                "preprocess_profile": calibration.preprocess.profile,
                "preprocess_source_closure_sha256":
                    calibration.preprocess.source_closure_sha256,
            }
            rows, manifest_path = _materialize_gpu_replay_inputs(
                output,
                {1: source_one, 2: source_two},
                {1: raw_one_path, 2: raw_two_path},
                [1, 2],
                [4, 5],
                [False, False],
                "subject-state-hard-cut-generation",
                (height, width),
                run_model,
                MappingV2Config(
                    raw_coordinate_scale=calibration.raw_coordinate_scale),
            )
            self.assertNotIn("output_hold", rows[0])
            self.assertEqual(rows[1]["rendered_source_frame_id"], "00002")
            self.assertEqual(
                (output / "frames" / "frame_00002.png").read_bytes(),
                b"current-second-color")
            self.assertEqual(
                (output / "input_frames" / "frame_00002.png").read_bytes(),
                b"current-second-color")
            self.assertEqual(
                rows[1]["input_source_sha256"], rows[1]["rendered_source_sha256"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], GPU_INPUT_MANIFEST_SCHEMA)
            self.assertEqual(manifest["mode"], GPU_INPUT_MANIFEST_MODE)
            self.assertEqual(
                {key: manifest["mapping_config"][key] for key in (
                    "near_tail_probe_u", "near_tail_coverage_low",
                    "near_tail_coverage_high", "near_log_tau_dense",
                    "vertical_majorant_share")},
                {
                    "near_tail_probe_u": CALIBRATED_DEFAULTS.near_tail_probe_u,
                    "near_tail_coverage_low": CALIBRATED_DEFAULTS.near_tail_coverage_low,
                    "near_tail_coverage_high": CALIBRATED_DEFAULTS.near_tail_coverage_high,
                    "near_log_tau_dense": CALIBRATED_DEFAULTS.near_log_tau_dense,
                    "vertical_majorant_share":
                        CALIBRATED_DEFAULTS.vertical_majorant_share,
                })
            self.assertEqual(
                manifest["model_identity"]["preprocess_source_closure_sha256"],
                calibration.preprocess.source_closure_sha256)
            self.assertEqual(
                [(frame["hard_cut_count"], frame["hard_cut_pulse"])
                 for frame in manifest["frames"]],
                [(4, False), (5, False)],
            )
            input_contract = {
                "calibration_id": calibration.calibration_id,
                "model": calibration.depth_model,
                "depth_model_url": calibration.depth_model_url,
                "onnx_sha256": calibration.onnx_sha256,
                "preprocess_profile": calibration.preprocess.profile,
                "preprocess_source_closure_sha256":
                    calibration.preprocess.source_closure_sha256,
                "raw_shape": {"height": height, "width": width},
                "selected_frame_ids": ["00001", "00002"],
                "ordered_raw_fields": [
                    {"frame_id": row["frame_id"], "sha256": row["raw_depth_sha256"]}
                    for row in rows
                ],
            }
            reference = {
                "file": "gpu_input/manifest.json",
                "schema": GPU_INPUT_MANIFEST_SCHEMA,
                "sha256": whole_clip_raw_contract.file_sha256(manifest_path),
            }
            validated = _validate_gpu_input_manifest_evidence(
                output, reference, manifest["mapping_config"],
                "subject-state-hard-cut-generation", input_contract)
            self.assertEqual(validated["schema"], GPU_INPUT_MANIFEST_SCHEMA)
            for name in ("model", "shape", "frame_hash"):
                with self.subTest(corruption=name):
                    corrupt = json.loads(json.dumps(manifest))
                    if name == "model":
                        corrupt["model_identity"]["model"] = "different"
                    elif name == "shape":
                        corrupt["raw_shape"]["width"] += 1
                    else:
                        corrupt["frames"][0]["raw_sha256"] = "0" * 64
                    manifest_path.write_text(json.dumps(corrupt), encoding="utf-8")
                    reference["sha256"] = whole_clip_raw_contract.file_sha256(manifest_path)
                    with self.assertRaises(ValueError):
                        _validate_gpu_input_manifest_evidence(
                            output, reference, manifest["mapping_config"],
                            "subject-state-hard-cut-generation", input_contract)
            raw_copy = output / "gpu_input" / "raw_00001.f32"
            original_raw = raw_copy.read_bytes()
            raw_copy.write_bytes(b"tampered")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            reference["sha256"] = whole_clip_raw_contract.file_sha256(manifest_path)
            with self.assertRaisesRegex(ValueError, "raw tensor"):
                _validate_gpu_input_manifest_evidence(
                    output, reference, manifest["mapping_config"],
                    "subject-state-hard-cut-generation", input_contract)
            raw_copy.write_bytes(original_raw)
            stale = json.loads(manifest_path.read_text(encoding="utf-8"))
            stale["schema"] = 1
            manifest_path.write_text(json.dumps(stale), encoding="utf-8")
            reference["sha256"] = whole_clip_raw_contract.file_sha256(manifest_path)
            with self.assertRaisesRegex(ValueError, "does not bind"):
                _validate_gpu_input_manifest_evidence(
                    output, reference, manifest["mapping_config"],
                    "subject-state-hard-cut-generation", input_contract)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            _validate_frame_source_attestation(output, rows[1], "00002")
            (output / "input_frames" / "frame_00002.png").write_bytes(b"tampered-current-input")
            with self.assertRaisesRegex(ValueError, "input source hash mismatch"):
                _validate_frame_source_attestation(output, rows[1], "00002")

    def test_producer_bundle_survives_source_deletion_and_authenticates_cut_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            eval_root = workspace / "eval"
            eval_root.mkdir()
            clip, run_model = self._write_run(eval_root)
            output = workspace / "replay"
            output.mkdir()
            shape = _load_shape(clip)
            input_contract = _clip_sequence_input_contract(clip, [1], shape, run_model)
            producer_reference = _materialize_producer_evidence_bundle(
                output, clip, run_model)
            source = workspace / "frame_00001.png"
            source.write_bytes(b"source")
            calibration = MODEL_CALIBRATIONS[0]
            config = MappingV2Config(raw_coordinate_scale=calibration.raw_coordinate_scale)
            _, gpu_path = _materialize_gpu_replay_inputs(
                output, {1: source}, dict(_raw_files(clip)), [1], [0], [False],
                "subject-state-hard-cut-generation", shape, run_model, config)
            gpu_reference = {
                "file": "gpu_input/manifest.json", "schema": GPU_INPUT_MANIFEST_SCHEMA,
                "sha256": whole_clip_raw_contract.file_sha256(gpu_path),
            }
            gpu_manifest = _validate_gpu_input_manifest_evidence(
                output, gpu_reference, json.loads(gpu_path.read_text())["mapping_config"],
                "subject-state-hard-cut-generation", input_contract)
            validated = _validate_producer_evidence_bundle(
                output, producer_reference, input_contract, gpu_manifest)
            self.assertEqual(validated["calibration_status"], "calibrated")
            corrupt_cut = json.loads(json.dumps(gpu_manifest))
            corrupt_cut["frames"][0]["hard_cut_count"] = 1
            with self.assertRaisesRegex(ValueError, "cut evidence disagrees"):
                _validate_producer_evidence_bundle(
                    output, producer_reference, input_contract, corrupt_cut)

            shutil.rmtree(eval_root)
            _validate_producer_evidence_bundle(
                output, producer_reference, input_contract, gpu_manifest)
            subject_path = output / producer_reference["subject_state"]["file"]
            subject_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "subject_state hash mismatch"):
                _validate_producer_evidence_bundle(
                    output, producer_reference, input_contract, gpu_manifest)

    def test_renderer_score_excludes_recomputed_legacy_state_from_voting(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            raw_path = output / "legacy_state_diagnostic_scorecard.json"
            aggregate = {"source_coverage_pct": 100.0}
            frame = {"_dump": "sbs_00001.png", "source_coverage_pct": 100.0}
            for index, key in enumerate(LEGACY_STATE_METRICS):
                aggregate[key] = float(index)
                frame[key] = float(index)
            raw_path.write_text(json.dumps({
                "aggregate": aggregate, "frames": [frame]}), encoding="utf-8")

            evidence = _publish_renderer_quality_score(raw_path, output)
            sanitized = json.loads((output / RENDERER_SCORECARD_FILE).read_text())
            scope_contract = json.loads(
                (output / RENDERER_SCORE_CONTRACT_FILE).read_text())
            self.assertEqual(evidence["scope"], "renderer-output-quality-only-v1")
            self.assertNotIn("scorer_source_sha256", evidence)
            self.assertEqual(evidence["metric_contract"], _metric_contract_evidence())
            self.assertEqual(scope_contract["schema"], RENDERER_SCORE_CONTRACT_SCHEMA)
            self.assertEqual(scope_contract["metric_contract"], evidence["metric_contract"])
            self.assertEqual(sanitized["aggregate"]["source_coverage_pct"], 100.0)
            for key in LEGACY_STATE_METRICS:
                self.assertNotIn(key, sanitized["aggregate"])
                self.assertNotIn(key, sanitized["frames"][0])
            original = json.loads(raw_path.read_text())
            self.assertTrue(all(key in original["aggregate"] for key in LEGACY_STATE_METRICS))

    def test_score_provenance_uses_full_canonical_metric_contract(self):
        evidence = _metric_contract_evidence()
        self.assertEqual(
            [source["file"] for source in evidence["sources"]],
            [
                "tools/sbsbench/sbsbench.py",
                "tools/sbsbench/sbs_interocular_metrics.py",
                "tools/sbsbench/sbs_interocular_phase_chroma.py",
                "tools/sbsbench/sbs_interocular_photometric_rivalry.py",
                "tools/sbsbench/sbs_stereo_window_metrics.py",
                "tools/sbsbench/sbs_warp_shear_metrics.py",
                "tools/sbsbench/direct_geometry_contract.py",
                "tools/sbsbench/subject_state_contract.py",
                "tools/sbsbench/thresholds.json",
            ])
        self.assertRegex(evidence["sha256"], r"^[0-9a-f]{16}$")
        self.assertTrue(all(
            set(source) == {"file", "sha256"} and
            re.fullmatch(r"[0-9a-f]{16}", source["sha256"])
            for source in evidence["sources"]))
        _validate_metric_contract_evidence(evidence)

        tampered = json.loads(json.dumps(evidence))
        tampered["sources"][-1]["sha256"] = "0" * 16
        with self.assertRaisesRegex(ValueError, "stale or incomplete"):
            _validate_metric_contract_evidence(tampered)


if __name__ == "__main__":
    unittest.main()
