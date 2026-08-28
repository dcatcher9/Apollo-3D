from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import prod_zipdepth_convex2x_diagnostics_contract as diagnostics  # noqa: E402
import run_eval  # noqa: E402


class ProdZipDepthConvex2xDiagnosticContractTests(unittest.TestCase):
    embedded_provenance = {
        "schema": 1,
        "model": "depth_anything_v2_fp16",
        "depth_model_url": "https://example.invalid/dav2.onnx",
        "onnx_sha256": "d" * 64,
        "preprocess_profile": "imagenet-test",
        "preprocess_source_closure_sha256": "b" * 64,
        "raw_width": 4,
        "raw_height": 2,
    }
    sidecar_embedded_provenance = {
        key: value for key, value in embedded_provenance.items()
        if key not in {"schema", "raw_width", "raw_height"}
    }
    composite_sidecar = {
        "model": "prod_dav2_zipdepth_c2x_high_opset18",
        "onnx_sha256": "f" * 64,
        "embedded_dav2_onnx_sha256": "d" * 64,
        "zipdepth_checkpoint_sha256": "c" * 64,
        "guidance_preprocess_source_closure_sha256": "b" * 64,
        "engine_recipe": "trt-6high-point-l5-v2",
        "engine_artifact": "fused.test.engine",
        "active_engine_manifest": "fused.active-engine.json",
    }

    def setUp(self) -> None:
        # The production contract admits only six calibrated megapixel profiles. Keep these tiny
        # byte-level fixtures cheap through a private test-only monkeypatch; no production reader
        # or evaluator entry point can request this exception.
        self.production_active_grid_validator = diagnostics._validate_active_grid_calibration

        def validate_fixture_grid(width, height, embedded):
            if ((width, height) == (4, 2) and
                    embedded == self.sidecar_embedded_provenance):
                return 2, 1
            return self.production_active_grid_validator(width, height, embedded)

        patcher = mock.patch.object(
            diagnostics, "_validate_active_grid_calibration",
            side_effect=validate_fixture_grid)
        patcher.start()
        self.addCleanup(patcher.stop)

    @classmethod
    def expected_runtime(cls):
        return {
            "schema": 2,
            "runtime": "dav2_zipdepth_convex2x_composite",
            **cls.composite_sidecar,
            "engine_sha256": "e" * 64,
            "active_engine_manifest_sha256": "a" * 64,
        }

    @staticmethod
    def write_json(path: Path, value) -> None:
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def make_artifacts(self, root: Path, frame_ids=(1, 2)) -> None:
        sidecar = {
            "schema": diagnostics.SIDECAR_SCHEMA,
            "authority": {
                "role": "authenticated-single-high-model-io",
                "live_geometry_source": "refined_depth",
                "scoring_depth_source": "raw_<frame-id>.f32",
                "coarse_dav2_public_binding": False,
            },
            "same_frame_binding": "completed_frame_id-to-decimal-frame-id",
            "frame_count": len(frame_ids),
            "model_input": {
                "width": 4,
                "height": 2,
                "tensor_shape_nchw": [1, 3, 2, 4],
                "dtype": "float32-le",
                "layout": "nchw-contiguous",
                "file_pattern": "model_input_<frame-id>.f32",
                "stage": (
                    "sole fused ONNX high-resolution pixel_values binding after "
                    "authenticated preprocess"),
            },
            "refined_depth": {
                "width": 4,
                "height": 2,
                "tensor_shape_nchw": [1, 1, 2, 4],
                "dtype": "float32-le",
                "layout": "nchw-contiguous",
                "file_pattern": "raw_<frame-id>.f32",
                "stage": (
                    "sole fused ONNX refined_depth binding and live high-resolution depth source"),
            },
            "diagnostic_aliases": {
                "model_input_primary": "model_input_snapshot",
                "model_input_compatibility_alias": "guidance_model_input_snapshot",
                "refined_depth_primary": "raw_model_depth_snapshot",
                "refined_depth_compatibility_alias": "refined_model_depth_snapshot",
                "gpu_resource_policy": "compatibility-aliases-reference-primary-resources",
                "duplicate_gpu_resources": False,
            },
            "composite_runtime_provenance": dict(self.composite_sidecar),
            "embedded_dav2_provenance": dict(self.sidecar_embedded_provenance),
        }
        sidecar_path = root / diagnostics.SIDECAR_FILENAME
        self.write_json(sidecar_path, sidecar)
        contract = {
            "schema": diagnostics.HARNESS_CONTRACT_SCHEMA,
            "raw_model_provenance": dict(self.embedded_provenance),
            "prod_zipdepth_convex2x_diagnostics": {
                "schema": diagnostics.SIDECAR_SCHEMA,
                "sidecar": diagnostics.SIDECAR_FILENAME,
                "sidecar_sha256": diagnostics.file_sha256(sidecar_path),
                "frame_count": len(frame_ids),
                "input_file_pattern": "model_input_<frame-id>.f32",
                "output_file_pattern": "raw_<frame-id>.f32",
                "authority": "single-high-input-output-boundary",
            },
        }
        self.write_json(root / "contract.json", contract)
        for frame_id in frame_ids:
            identity = f"{frame_id:05d}"
            (root / f"model_input_{identity}.f32").write_bytes(
                np.linspace(-1.0, 1.0, 24, dtype="<f4").tobytes())
            (root / f"raw_{identity}.f32").write_bytes(
                np.linspace(0.0, 1.0, 8, dtype="<f4").tobytes())

    def make_legacy_artifacts(self, root: Path, frame_ids=(1,)) -> None:
        legacy_provenance = dict(self.embedded_provenance)
        legacy_provenance["raw_width"] = 2
        legacy_provenance["raw_height"] = 1
        sidecar = {
            "schema": diagnostics.LEGACY_SIDECAR_SCHEMA,
            "authority": {
                "role": "diagnostic-only-stage-1",
                "live_geometry_authority": False,
                "scoring_depth_authority": False,
                "coarse_dav2_remains_live_authority": True,
            },
            "same_frame_binding": "completed_frame_id-to-decimal-frame-id",
            "frame_count": len(frame_ids),
            "coarse_dav2": {
                "width": 2, "height": 1, "tensor_shape_nchw": [1, 1, 1, 2],
                "raw_file_pattern": "raw_<frame-id>.f32",
                "role": "existing-prod-raw-output-and-only-live-depth-authority",
            },
            "zip_model_input": {
                "width": 4, "height": 2, "tensor_shape_nchw": [1, 3, 2, 4],
                "dtype": "float32-le", "layout": "nchw-contiguous",
                "file_pattern": "zip_model_input_<frame-id>.f32",
                "stage": "matched native RGB independently preprocessed on the exact 2x grid",
            },
            "refined_depth": {
                "width": 4, "height": 2, "tensor_shape_nchw": [1, 1, 2, 4],
                "dtype": "float32-le", "layout": "nchw-contiguous",
                "file_pattern": "refined_<frame-id>.f32",
                "stage": "fused ONNX refined_depth after frozen ZipDepth convex2x",
            },
            "composite_runtime_provenance": dict(self.composite_sidecar),
            "raw_dav2_provenance": dict(self.sidecar_embedded_provenance),
        }
        sidecar_path = root / diagnostics.SIDECAR_FILENAME
        self.write_json(sidecar_path, sidecar)
        contract = {
            "schema": diagnostics.HARNESS_CONTRACT_SCHEMA,
            "raw_model_provenance": legacy_provenance,
            "prod_zipdepth_convex2x_diagnostics": {
                "schema": diagnostics.LEGACY_SIDECAR_SCHEMA,
                "sidecar": diagnostics.SIDECAR_FILENAME,
                "sidecar_sha256": diagnostics.file_sha256(sidecar_path),
                "frame_count": len(frame_ids),
                "refined_file_pattern": "refined_<frame-id>.f32",
                "guidance_file_pattern": "zip_model_input_<frame-id>.f32",
                "authority": "diagnostic-only-never-live-or-scoring-depth",
            },
        }
        self.write_json(root / "contract.json", contract)
        for frame_id in frame_ids:
            identity = f"{frame_id:05d}"
            (root / f"raw_{identity}.f32").write_bytes(
                np.asarray([frame_id, frame_id + 0.25], dtype="<f4").tobytes())
            (root / f"zip_model_input_{identity}.f32").write_bytes(
                np.linspace(-1.0, 1.0, 24, dtype="<f4").tobytes())
            (root / f"refined_{identity}.f32").write_bytes(
                np.linspace(0.0, 1.0, 8, dtype="<f4").tobytes())

    def test_builds_single_high_manifest_without_duplicate_output_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifacts(root)
            manifest = diagnostics.build_manifest(root, [1, 2], self.expected_runtime())

            self.assertEqual(manifest["schema"], diagnostics.MANIFEST_SCHEMA)
            self.assertEqual(set(manifest["tensor_shapes"]), {"input", "output"})
            self.assertEqual(manifest["tensor_shapes"]["input"]["width"], 4)
            self.assertEqual(manifest["tensor_shapes"]["output"]["width"], 4)
            self.assertEqual(manifest["frames"][0]["output"]["file"], "raw_00001.f32")
            self.assertFalse(any(root.glob("refined_*.f32")))
            self.assertEqual(
                diagnostics.authenticate_manifest_files(
                    root, manifest, [1, 2], self.expected_runtime()),
                manifest["frames"])

    def test_historical_schema_one_artifacts_remain_authenticatable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_legacy_artifacts(root)
            manifest = diagnostics.build_manifest(root, [1], self.expected_runtime())

            self.assertEqual(manifest["schema"], diagnostics.LEGACY_MANIFEST_SCHEMA)
            self.assertEqual(manifest["binding"], diagnostics.LEGACY_BINDING)
            self.assertEqual(set(manifest["tensor_shapes"]), {"raw", "guidance", "refined"})
            diagnostics.authenticate_manifest_files(
                root, manifest, [1], self.expected_runtime())

    def test_production_schema_two_gate_accepts_only_six_calibrated_high_profiles(self):
        calibration = diagnostics.coordinate_contract.MODEL_CALIBRATIONS[0]
        embedded = {
            "model": calibration.depth_model,
            "depth_model_url": calibration.depth_model_url,
            "onnx_sha256": calibration.onnx_sha256,
            "preprocess_profile": calibration.preprocess.profile,
            "preprocess_source_closure_sha256":
                calibration.preprocess.source_closure_sha256,
        }
        supported = diagnostics.convex2x_contract.supported_high_shapes()
        self.assertEqual(len(supported), 6)
        for high in supported:
            with self.subTest(shape=(high.width, high.height)):
                self.assertEqual(
                    self.production_active_grid_validator(
                        high.width, high.height, embedded),
                    (high.width // 2, high.height // 2))

        with self.assertRaisesRegex(ValueError, "six supported high profiles"):
            self.production_active_grid_validator(2016, 868, embedded)
        wrong = dict(embedded)
        wrong["preprocess_source_closure_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "calibrated DAV2 half-shape"):
            self.production_active_grid_validator(
                supported[0].width, supported[0].height, wrong)

    def test_rejects_nonfinite_or_wrong_sized_tensor_before_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifacts(root)
            values = np.linspace(0.0, 1.0, 8, dtype="<f4")
            values[3] = np.nan
            (root / "raw_00001.f32").write_bytes(values.tobytes())
            with self.assertRaisesRegex(ValueError, "non-finite"):
                diagnostics.build_manifest(root, [1, 2], self.expected_runtime())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifacts(root)
            (root / "model_input_00002.f32").write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "expected 96"):
                diagnostics.build_manifest(root, [1, 2], self.expected_runtime())

    def test_reauthentication_rejects_mutation_or_extra_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifacts(root)
            manifest = diagnostics.build_manifest(root, [1, 2], self.expected_runtime())
            (root / "raw_00001.f32").write_bytes(np.full(8, 0.5, dtype="<f4").tobytes())
            with self.assertRaisesRegex(ValueError, "changed after evaluator capture"):
                diagnostics.authenticate_manifest_files(
                    root, manifest, [1, 2], self.expected_runtime())

    def test_schema_two_rejects_stale_split_artifacts(self):
        for stale_name in ("zip_model_input_00001.f32", "refined_00001.f32"):
            with self.subTest(stale_name=stale_name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.make_artifacts(root)
                (root / stale_name).write_bytes(np.zeros(8, dtype="<f4").tobytes())
                with self.assertRaisesRegex(ValueError, "stale split tensors"):
                    diagnostics.build_manifest(root, [1, 2], self.expected_runtime())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifacts(root)
            manifest = diagnostics.build_manifest(root, [1, 2], self.expected_runtime())
            (root / "refined_00009.f32").write_bytes(np.zeros(8, dtype="<f4").tobytes())
            with self.assertRaisesRegex(ValueError, "stale split tensors"):
                diagnostics.authenticate_manifest_files(
                    root, manifest, [1, 2], self.expected_runtime())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifacts(root)
            manifest = diagnostics.build_manifest(root, [1, 2], self.expected_runtime())
            (root / "model_input_00003.f32").write_bytes(
                np.ones(24, dtype="<f4").tobytes())
            with self.assertRaisesRegex(ValueError, "files disagree"):
                diagnostics.authenticate_manifest_files(
                    root, manifest, [1, 2], self.expected_runtime())

    def test_rejects_sidecar_schema_or_runtime_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifacts(root)
            sidecar_path = root / diagnostics.SIDECAR_FILENAME
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["unexpected"] = True
            self.write_json(sidecar_path, sidecar)
            contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
            contract["prod_zipdepth_convex2x_diagnostics"]["sidecar_sha256"] = (
                diagnostics.file_sha256(sidecar_path))
            self.write_json(root / "contract.json", contract)
            with self.assertRaisesRegex(ValueError, "schema-2 semantics"):
                diagnostics.build_manifest(root, [1, 2], self.expected_runtime())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifacts(root)
            expected = self.expected_runtime()
            expected["engine_artifact"] = "another.engine"
            with self.assertRaisesRegex(ValueError, "runtime identity"):
                diagnostics.build_manifest(root, [1, 2], expected)

    def test_model_boundary_files_change_authenticated_not_numeric_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sbs_00001.png").write_bytes(b"numeric")
            (root / diagnostics.SIDECAR_FILENAME).write_bytes(b"sidecar")
            (root / "raw_00001.f32").write_bytes(b"output-v1")
            (root / "model_input_00001.f32").write_bytes(b"input-v1")
            before = run_eval.scored_artifact_digests(root)

            (root / "model_input_00001.f32").write_bytes(b"input-v2")
            after = run_eval.scored_artifact_digests(root)

            self.assertNotEqual(before[0], after[0])
            self.assertEqual(before[1], after[1])


if __name__ == "__main__":
    unittest.main()
