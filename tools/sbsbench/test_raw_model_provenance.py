#!/usr/bin/env python3
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import raw_model_provenance as provenance  # noqa: E402
import replay_depth_mapping_v2 as replay  # noqa: E402


class RawModelProvenanceTests(unittest.TestCase):
    CALIBRATION = provenance.coordinate_contract.MODEL_CALIBRATIONS[0]
    WIDTH, HEIGHT = CALIBRATION.calibrated_input_shapes[0]

    def _dump(self, root: Path, proof=None) -> tuple[Path, dict]:
        dump = root / "dump"
        dump.mkdir()
        with (dump / "raw_depth.f32").open("wb") as stream:
            stream.truncate(self.WIDTH * self.HEIGHT * 4)
        (dump / "raw_shape.json").write_text(json.dumps({
            "schema": 1,
            "width": self.WIDTH,
            "height": self.HEIGHT,
            "dtype": "float32-le",
            "layout": "row-major",
        }), encoding="utf-8")
        with (dump / "model_input.f32").open("wb") as stream:
            stream.truncate(3 * self.WIDTH * self.HEIGHT * 4)
        preprocess = self.CALIBRATION.preprocess
        (dump / "model_input_shape.json").write_text(json.dumps({
            "schema": preprocess.model_input_schema,
            "width": self.WIDTH,
            "height": self.HEIGHT,
            "dtype": preprocess.dtype,
            "layout": preprocess.layout,
            "channels": list(preprocess.channels),
            "stage": preprocess.stage,
            "imagenet_mean": list(preprocess.imagenet_mean),
            "imagenet_std": list(preprocess.imagenet_std),
        }), encoding="utf-8")
        manifest = {
            "schema": 1,
            "depth_model": self.CALIBRATION.depth_model,
            "config": {
                "depth_model": self.CALIBRATION.depth_model,
                "configured_depth_model": self.CALIBRATION.depth_model,
                "depth_model_url": self.CALIBRATION.depth_model_url,
            },
        }
        if proof is not None:
            manifest[provenance.PROVENANCE_KEY] = proof
        (dump / "dump_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        return dump, manifest

    def _proof(self, dump: Path) -> dict:
        return {
            "schema": provenance.PROVENANCE_SCHEMA,
            "binding": provenance.BINDING,
            "depth_model": self.CALIBRATION.depth_model,
            "depth_model_url": self.CALIBRATION.depth_model_url,
            "onnx_sha256": self.CALIBRATION.onnx_sha256,
            "preprocess_profile": self.CALIBRATION.preprocess.profile,
            "preprocess_source_closure_sha256":
                self.CALIBRATION.preprocess.source_closure_sha256,
            "raw_depth_sha256": provenance.file_sha256(dump / "raw_depth.f32"),
            "model_input_sha256": provenance.file_sha256(dump / "model_input.f32"),
            "model_input_shape_sha256": provenance.file_sha256(
                dump / "model_input_shape.json"),
        }

    def test_legacy_dump_is_unverified_and_requires_explicit_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            dump, _ = self._dump(Path(temporary))
            observed = provenance.inspect_dump(dump)
            self.assertEqual(observed.status, "unverified")
            self.assertFalse(observed.authoritative)
            self.assertIsNone(observed.attestation_schema)
            self.assertIsNone(observed.binding)
            self.assertIsNone(observed.onnx_sha256)
            self.assertIn("descriptive only", observed.reason)
            with self.assertRaisesRegex(ValueError, "--allow-unverified-model-provenance"):
                provenance.require_authoritative(observed, False)
            provenance.require_authoritative(observed, True)

    def test_capture_time_onnx_and_raw_hash_binding_is_authoritative(self):
        with tempfile.TemporaryDirectory() as temporary:
            dump, manifest = self._dump(Path(temporary))
            manifest[provenance.PROVENANCE_KEY] = self._proof(dump)
            (dump / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            observed = provenance.inspect_dump(dump)
            self.assertTrue(observed.authoritative)
            self.assertEqual(observed.status, "authoritative")
            self.assertEqual(observed.attestation_schema, provenance.PROVENANCE_SCHEMA)
            self.assertEqual(observed.binding, provenance.BINDING)
            self.assertEqual(observed.onnx_sha256, self.CALIBRATION.onnx_sha256)
            self.assertEqual(observed.calibration_id, self.CALIBRATION.calibration_id)
            self.assertEqual(
                observed.preprocess_source_closure_sha256,
                self.CALIBRATION.preprocess.source_closure_sha256)
            self.assertEqual(
                observed.calibrated_raw_coordinate_scale,
                self.CALIBRATION.raw_coordinate_scale)
            self.assertEqual(observed.model_input_width, self.WIDTH)
            self.assertEqual(observed.model_input_height, self.HEIGHT)
            self.assertIsNone(observed.reason)
            provenance.require_authoritative(observed, False)

    def test_corrupt_or_mismatched_proof_never_downgrades_to_unverified(self):
        for mutation, message in (("unknown", "missing or unknown"),
                                  ("model", "disagrees"),
                                  ("profile", "unknown calibrated preprocess"),
                                  ("source", "unknown calibrated preprocess source"),
                                  ("raw", "does not match")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                dump, manifest = self._dump(Path(temporary))
                proof = self._proof(dump)
                if mutation == "unknown":
                    proof["extra"] = True
                elif mutation == "model":
                    proof["depth_model"] = "different"
                elif mutation == "profile":
                    proof["preprocess_profile"] = "different-profile"
                elif mutation == "source":
                    proof["preprocess_source_closure_sha256"] = "c" * 64
                else:
                    proof["raw_depth_sha256"] = "b" * 64
                manifest[provenance.PROVENANCE_KEY] = proof
                (dump / "dump_manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    provenance.inspect_dump(dump)

    def test_valid_but_unknown_onnx_sha_cannot_claim_small_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            dump, manifest = self._dump(Path(temporary))
            proof = self._proof(dump)
            proof["onnx_sha256"] = "a" * 64
            manifest[provenance.PROVENANCE_KEY] = proof
            (dump / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match a calibrated model"):
                provenance.inspect_dump(dump)

    def test_authenticated_input_shape_must_match_preprocess_contract(self):
        mutations = {
            "layout": ("layout", "NHWC", "preprocess contract"),
            "mean": ("imagenet_mean", [0.0, 0.0, 0.0], "preprocess contract"),
            "patch": ("width", 13, "spatial contract"),
            "uncalibrated_shape": ("width", 1008, "not an exact shape"),
        }
        for name, (field, value, message) in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                dump, manifest = self._dump(Path(temporary))
                shape_path = dump / "model_input_shape.json"
                shape = json.loads(shape_path.read_text(encoding="utf-8"))
                shape[field] = value
                shape_path.write_text(json.dumps(shape), encoding="utf-8")
                proof = self._proof(dump)
                manifest[provenance.PROVENANCE_KEY] = proof
                (dump / "dump_manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    provenance.inspect_dump(dump)

    def test_bound_input_artifacts_cannot_change_after_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            dump, manifest = self._dump(Path(temporary))
            manifest[provenance.PROVENANCE_KEY] = self._proof(dump)
            (dump / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with (dump / "model_input.f32").open("r+b") as stream:
                stream.write(b"\x00\x00\x80?")
            with self.assertRaisesRegex(ValueError, "model_input.f32 SHA-256"):
                provenance.inspect_dump(dump)

    def test_replay_refuses_unverified_dump_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dump, _ = self._dump(root)
            (dump / "source.png").write_bytes(b"preview")
            output = root / "output"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                replay.main(["--dump", str(dump), "--out", str(output), "--skip-score"])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("raw model provenance is unverified", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_model_flag_is_harness_only_and_old_ambiguous_flag_is_rejected(self):
        parser = replay._build_parser()
        parsed = parser.parse_args([
            "--dump", "dump", "--out", "out", "--harness-model", "midas",
            "--allow-unverified-model-provenance",
        ])
        self.assertEqual(parsed.harness_model, "midas")
        self.assertTrue(parsed.allow_unverified_model_provenance)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            parser.parse_args(["--dump", "dump", "--out", "out", "--model", "midas"])
        self.assertEqual(raised.exception.code, 2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            parser.parse_args([
                "--dump", "dump", "--out", "out", "--raw-scale-prior", "0.5"])
        self.assertEqual(raised.exception.code, 2)

    def test_unverified_report_separates_raw_bytes_from_harness_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dump, _ = self._dump(root)
            (dump / "source.png").write_bytes(b"diagnostic preview")
            build = root / "build"
            build.mkdir()
            (build / "sunshine.exe").write_bytes(b"placeholder")
            conf = root / "bench.conf"
            conf.write_text("", encoding="utf-8")
            output = root / "output"
            with mock.patch.object(replay, "_run_checked", side_effect=RuntimeError("stop")):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    replay.main([
                        "--dump", str(dump), "--out", str(output),
                        "--build-dir", str(build), "--conf", str(conf),
                        "--harness-model", "midas", "--allow-unverified-model-provenance",
                        "--experimental-raw-coordinate-scale", "0.5",
                        "--skip-score",
                    ])
            report = json.loads(
                (output / "mapping_v2_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], 7)
            self.assertEqual(
                report["source_geometry"]["authority"],
                "exact-raw-depth-input-to-non-captured-state-recomputation")
            self.assertEqual(report["raw_model_provenance"]["status"], "unverified")
            self.assertEqual(
                report["mapping_calibration"]["status"], "experiment-unverified")
            self.assertFalse(report["mapping_calibration"]["authoritative"])
            self.assertEqual(report["mapping_calibration"]["raw_coordinate_scale"], 0.5)
            self.assertIsNone(
                report["mapping_calibration"]["preprocess_source_closure_sha256"])
            self.assertEqual(report["config"]["raw_coordinate_scale"], 0.5)
            self.assertEqual(report["harness_legacy_estimator"]["model"], "midas")
            self.assertEqual(report["harness_legacy_estimator"]["geometry_role"],
                             "none; direct canonical order and conditioned parallax replace its geometry")
            self.assertEqual(report["harness_legacy_estimator"]["provenance_role"], "none")

    def test_unverified_nonzero_floor_requires_experiment_only_flag(self):
        provenance_record = provenance.RawModelProvenance(
            status="unverified", source="dump_manifest.json", attestation_schema=None,
            binding=None, declared_model="custom", configured_model="custom",
            declared_url=None, onnx_sha256=None, raw_depth_sha256="a" * 64,
            model_input_sha256=None, model_input_shape_sha256=None,
            preprocess_profile=None, preprocess_source_closure_sha256=None,
            calibration_id=None,
            calibrated_raw_coordinate_scale=None, model_input_width=None,
            model_input_height=None, reason="legacy")
        floor, report = replay._resolve_mapping_calibration(provenance_record, 0.75)
        self.assertEqual(floor, 0.75)
        self.assertEqual(report["status"], "experiment-unverified")
        self.assertFalse(report["authoritative"])

    def test_authoritative_floor_comes_only_from_bound_manifest_calibration(self):
        provenance_record = provenance.RawModelProvenance(
            status="authoritative", source="dump_manifest.json",
            attestation_schema=provenance.PROVENANCE_SCHEMA, binding=provenance.BINDING,
            declared_model=self.CALIBRATION.depth_model,
            configured_model=self.CALIBRATION.depth_model,
            declared_url=self.CALIBRATION.depth_model_url,
            onnx_sha256=self.CALIBRATION.onnx_sha256, raw_depth_sha256="a" * 64,
            model_input_sha256="b" * 64, model_input_shape_sha256="c" * 64,
            preprocess_profile=self.CALIBRATION.preprocess.profile,
            preprocess_source_closure_sha256=
                self.CALIBRATION.preprocess.source_closure_sha256,
            calibration_id=self.CALIBRATION.calibration_id,
            calibrated_raw_coordinate_scale=self.CALIBRATION.raw_coordinate_scale,
            model_input_width=self.WIDTH, model_input_height=self.HEIGHT, reason=None)
        floor, report = replay._resolve_mapping_calibration(provenance_record, None)
        self.assertEqual(floor, self.CALIBRATION.raw_coordinate_scale)
        self.assertEqual(report["status"], "authoritative")
        self.assertEqual(
            report["preprocess_source_closure_sha256"],
            self.CALIBRATION.preprocess.source_closure_sha256)
        with self.assertRaisesRegex(ValueError, "only valid for unverified"):
            replay._resolve_mapping_calibration(provenance_record, 0.75)


if __name__ == "__main__":
    unittest.main()
