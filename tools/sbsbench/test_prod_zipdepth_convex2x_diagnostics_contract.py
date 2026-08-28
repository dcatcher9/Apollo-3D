from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import prod_zipdepth_convex2x_diagnostics_contract as diagnostics  # noqa: E402
import run_eval  # noqa: E402


class ProdZipDepthConvex2xDiagnosticContractTests(unittest.TestCase):
    raw_provenance = {
        "schema": 1,
        "model": "depth_anything_v2_fp16",
        "depth_model_url": "https://example.invalid/dav2.onnx",
        "onnx_sha256": "d" * 64,
        "preprocess_profile": "imagenet-test",
        "preprocess_source_closure_sha256": "b" * 64,
        "raw_width": 2,
        "raw_height": 1,
    }
    sidecar_raw_provenance = {
        key: value for key, value in raw_provenance.items()
        if key not in {"schema", "raw_width", "raw_height"}
    }
    composite_sidecar = {
        "model": "prod_dav2_zipdepth_convex2x_dynamic_opset18",
        "onnx_sha256": "f" * 64,
        "embedded_dav2_onnx_sha256": "d" * 64,
        "zipdepth_checkpoint_sha256": "c" * 64,
        "guidance_preprocess_source_closure_sha256": "b" * 64,
        "engine_recipe": "trt-six-point-profiles-level5-v1",
        "engine_artifact": "fused.test.engine",
        "active_engine_manifest": "fused.active-engine.json",
    }

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
            "schema": 1,
            "authority": {
                "role": "diagnostic-only-stage-1",
                "live_geometry_authority": False,
                "scoring_depth_authority": False,
                "coarse_dav2_remains_live_authority": True,
            },
            "same_frame_binding": "completed_frame_id-to-decimal-frame-id",
            "frame_count": len(frame_ids),
            "coarse_dav2": {
                "width": 2,
                "height": 1,
                "tensor_shape_nchw": [1, 1, 1, 2],
                "raw_file_pattern": "raw_<frame-id>.f32",
                "role": "existing-prod-raw-output-and-only-live-depth-authority",
            },
            "zip_model_input": {
                "width": 4,
                "height": 2,
                "tensor_shape_nchw": [1, 3, 2, 4],
                "dtype": "float32-le",
                "layout": "nchw-contiguous",
                "file_pattern": "zip_model_input_<frame-id>.f32",
                "stage": (
                    "matched native RGB independently preprocessed on the exact 2x grid"),
            },
            "refined_depth": {
                "width": 4,
                "height": 2,
                "tensor_shape_nchw": [1, 1, 2, 4],
                "dtype": "float32-le",
                "layout": "nchw-contiguous",
                "file_pattern": "refined_<frame-id>.f32",
                "stage": "fused ONNX refined_depth after frozen ZipDepth convex2x",
            },
            "composite_runtime_provenance": dict(self.composite_sidecar),
            "raw_dav2_provenance": dict(self.sidecar_raw_provenance),
        }
        sidecar_path = root / diagnostics.SIDECAR_FILENAME
        self.write_json(sidecar_path, sidecar)
        sidecar_sha = diagnostics.file_sha256(sidecar_path)
        contract = {
            "schema": diagnostics.HARNESS_CONTRACT_SCHEMA,
            "raw_model_provenance": dict(self.raw_provenance),
            "prod_zipdepth_convex2x_diagnostics": {
                "schema": 1,
                "sidecar": diagnostics.SIDECAR_FILENAME,
                "sidecar_sha256": sidecar_sha,
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

    def test_builds_and_reauthenticates_exact_complete_finite_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifacts(root)

            manifest = diagnostics.build_manifest(
                root, [1, 2], self.expected_runtime())

            self.assertEqual(manifest["frame_count"], 2)
            self.assertEqual(manifest["tensor_shapes"]["raw"]["width"], 2)
            self.assertEqual(manifest["tensor_shapes"]["guidance"]["channels"], 3)
            self.assertEqual(
                diagnostics.authenticate_manifest_files(
                    root, manifest, [1, 2], self.expected_runtime()),
                manifest["frames"])
            self.assertEqual(
                {row["frame_id"] for row in manifest["frames"]},
                {"00001", "00002"})

    def test_rejects_nonfinite_or_wrong_sized_tensor_before_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifacts(root)
            values = np.linspace(0.0, 1.0, 8, dtype="<f4")
            values[3] = np.nan
            (root / "refined_00001.f32").write_bytes(values.tobytes())
            with self.assertRaisesRegex(ValueError, "non-finite"):
                diagnostics.build_manifest(root, [1, 2], self.expected_runtime())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifacts(root)
            (root / "zip_model_input_00002.f32").write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "expected 96"):
                diagnostics.build_manifest(root, [1, 2], self.expected_runtime())

    def test_reauthentication_rejects_mutation_or_extra_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifacts(root)
            manifest = diagnostics.build_manifest(
                root, [1, 2], self.expected_runtime())
            (root / "refined_00001.f32").write_bytes(
                np.full(8, 0.5, dtype="<f4").tobytes())
            with self.assertRaisesRegex(ValueError, "changed after evaluator capture"):
                diagnostics.authenticate_manifest_files(
                    root, manifest, [1, 2], self.expected_runtime())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifacts(root)
            manifest = diagnostics.build_manifest(
                root, [1, 2], self.expected_runtime())
            (root / "raw_00003.f32").write_bytes(
                np.ones(2, dtype="<f4").tobytes())
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
            with self.assertRaisesRegex(ValueError, "schema-1 semantics"):
                diagnostics.build_manifest(root, [1, 2], self.expected_runtime())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifacts(root)
            expected = self.expected_runtime()
            expected["engine_artifact"] = "another.engine"
            with self.assertRaisesRegex(ValueError, "runtime identity"):
                diagnostics.build_manifest(root, [1, 2], expected)

    def test_diagnostic_files_change_only_authenticated_not_numeric_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sbs_00001.png").write_bytes(b"numeric")
            (root / diagnostics.SIDECAR_FILENAME).write_bytes(b"sidecar")
            (root / "raw_00001.f32").write_bytes(b"raw-v1")
            (root / "zip_model_input_00001.f32").write_bytes(b"guidance-v1")
            (root / "refined_00001.f32").write_bytes(b"refined-v1")
            before = run_eval.scored_artifact_digests(root)

            (root / "refined_00001.f32").write_bytes(b"refined-v2")
            after = run_eval.scored_artifact_digests(root)

            self.assertNotEqual(before[0], after[0])
            self.assertEqual(before[1], after[1])


if __name__ == "__main__":
    unittest.main()
