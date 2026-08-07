import io
import ast
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import argparse
import threading
import time
import unittest
from unittest import mock
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_depth_transform  # noqa: E402
import audit_depth_confidence  # noqa: E402
import make_synth_clips  # noqa: E402
import prepare_public_datasets  # noqa: E402
import run_eval  # noqa: E402
import rescore_run  # noqa: E402
import sbsbench  # noqa: E402
import eval_parallel  # noqa: E402


class EvalContractTests(unittest.TestCase):
    @staticmethod
    def png_bytes(value=64, mode="RGB"):
        shape = (8, 12, 3) if mode == "RGB" else (8, 12)
        array = np.full(shape, value, np.uint8)
        stream = io.BytesIO()
        Image.fromarray(array, mode=mode).save(stream, "PNG")
        return stream.getvalue()

    def test_rgb_sampler_reuses_uv_math_without_numeric_drift(self):
        rng = np.random.default_rng(23)
        image = rng.random((19, 31, 3), dtype=np.float32)
        u = rng.uniform(-0.2, 1.2, (13, 17)).astype(np.float32)
        v = rng.uniform(-0.2, 1.2, (13, 17)).astype(np.float32)
        expected = np.stack([
            sbsbench._sample_scalar_uv(image[..., channel], u, v)
            for channel in range(3)
        ], axis=-1)

        actual = sbsbench._sample_rgb_uv(image, u, v)
        self.assertTrue(np.array_equal(actual, expected))

    def test_multi_weighted_percentile_reuses_sort_without_numeric_drift(self):
        rng = np.random.default_rng(29)
        values = rng.integers(-20, 21, 4096).astype(np.float32)
        weights = rng.random(values.size, dtype=np.float32) + np.float32(1e-4)
        quantiles = (0.001, 0.01, 0.20, 0.50, 0.80, 0.99, 0.999)
        expected = tuple(
            sbsbench.weighted_pct(values, weights, quantile)
            for quantile in quantiles)

        actual = sbsbench.weighted_pcts(values, weights, quantiles)
        self.assertEqual(actual, expected)

    def test_metric_hash_is_independent_of_text_line_endings(self):
        paths = []
        try:
            for data in (b"alpha\nbeta\n", b"alpha\r\nbeta\r\n"):
                with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as fh:
                    fh.write(data)
                    paths.append(fh.name)
            # sha256_files includes the basename, so give both temp files the same logical name.
            with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
                left_path = os.path.join(left, "metric.py")
                right_path = os.path.join(right, "metric.py")
                with open(paths[0], "rb") as src, open(left_path, "wb") as dst:
                    dst.write(src.read())
                with open(paths[1], "rb") as src, open(right_path, "wb") as dst:
                    dst.write(src.read())
                self.assertEqual(run_eval.sha256_files([left_path]),
                                 run_eval.sha256_files([right_path]))
        finally:
            for path in paths:
                os.unlink(path)

    def test_scored_artifact_hash_ignores_reports_and_optional_oracles(self):
        with tempfile.TemporaryDirectory() as run:
            with open(os.path.join(run, "sbs_00001.png"), "wb") as stream:
                stream.write(b"canonical-sbs")
            with open(os.path.join(run, "warp_map_00001.f32"), "wb") as stream:
                stream.write(b"canonical-map")
            oracle_dir = os.path.join(run, "offline_oracles")
            os.makedirs(oracle_dir)
            report_path = os.path.join(run, "report.html")
            oracle_path = os.path.join(oracle_dir, "flip.json")
            for path, value in ((report_path, b"report-v1"), (oracle_path, b"oracle-v1")):
                with open(path, "wb") as stream:
                    stream.write(value)
            original, numeric_original = run_eval.scored_artifact_digests(run)

            for path, value in ((report_path, b"report-v2"), (oracle_path, b"oracle-v2")):
                with open(path, "wb") as stream:
                    stream.write(value)
            self.assertEqual(original, run_eval.scored_artifact_sha256(run))
            self.assertEqual(numeric_original, run_eval.scored_artifact_digests(run)[1])

            with open(os.path.join(run, "sbs_perf.json"), "wb") as stream:
                stream.write(b"changed-performance")
            self.assertNotEqual(original, run_eval.scored_artifact_sha256(run))
            self.assertEqual(numeric_original, run_eval.scored_artifact_digests(run)[1])

            with open(os.path.join(run, "sbs_00001.png"), "wb") as stream:
                stream.write(b"changed-sbs")
            self.assertNotEqual(numeric_original, run_eval.scored_artifact_digests(run)[1])

    def test_direct_parallax_manifest_is_authenticated_and_geometry_bound(self):
        with tempfile.TemporaryDirectory() as artifact_dir:
            depth_files = {}
            fields = []
            for frame_id, encoded, maximum, order in (
                    (1, 0.5, 0.0, -0.75), (2, 0.75, 0.02, 1.25)):
                field_bytes = np.full((8, 12), encoded, dtype="<f4").tobytes()
                field_sha = hashlib.sha256(field_bytes).hexdigest()
                order_bytes = np.full((8, 12), order, dtype="<f4").tobytes()
                order_sha = hashlib.sha256(order_bytes).hexdigest()
                depth_path = os.path.join(artifact_dir, f"depth_{frame_id:05d}.f32")
                parallax_path = os.path.join(
                    artifact_dir, f"parallax_{frame_id:05d}.f32")
                with open(depth_path, "wb") as stream:
                    stream.write(order_bytes)
                with open(parallax_path, "wb") as stream:
                    stream.write(field_bytes)
                depth_files[frame_id] = depth_path
                fields.append({
                    "frame_id": f"{frame_id:05d}",
                    "width": 12,
                    "height": 8,
                    "parallax_sha256": field_sha,
                    "maximum_absolute_source_u": maximum,
                    "order_sha256": order_sha,
                    "order_minimum": order,
                    "order_maximum": order,
                })
            manifest = {
                **run_eval._DIRECT_GEOMETRY_MANIFEST_V6,
                "fields": fields,
            }
            manifest_path = os.path.join(
                artifact_dir, "direct_parallax_manifest.json")
            with open(manifest_path, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(manifest, stream, indent=2)
                stream.write("\n")
            contract = {
                "schema": run_eval.direct_geometry.CONTRACT_SCHEMA,
                "warp_input": run_eval.direct_geometry.WARP_INPUT,
                "direct_parallax_frames": 2,
                "direct_parallax": run_eval._DIRECT_GEOMETRY_DESCRIPTOR_V6,
                "direct_parallax_manifest": {
                    "file": "direct_parallax_manifest.json",
                    "schema": run_eval.direct_geometry.MANIFEST_SCHEMA,
                    "sha256": run_eval.file_sha256(manifest_path),
                },
            }

            loaded = run_eval.validate_direct_parallax_manifest(
                artifact_dir, contract, {1, 2}, depth_files)
            self.assertEqual(loaded["fields"][0]["maximum_absolute_source_u"], 0.0)
            self.assertEqual(loaded["fields"][1]["maximum_absolute_source_u"], 0.02)

            old_contract = json.loads(json.dumps(contract))
            old_contract["schema"] = 20
            with self.assertRaisesRegex(ValueError, "requires harness contract schema 25"):
                run_eval.validate_direct_parallax_manifest(
                    artifact_dir, old_contract, {1, 2}, depth_files)
            old_contract = json.loads(json.dumps(contract))
            old_contract["warp_input"] = "external-final-parallax-and-order-v3"
            with self.assertRaisesRegex(ValueError, "unknown warp_input"):
                run_eval.validate_direct_parallax_manifest(
                    artifact_dir, old_contract, {1, 2}, depth_files)
            old_contract = json.loads(json.dumps(contract))
            old_contract["direct_parallax_manifest"]["schema"] = 3
            with self.assertRaisesRegex(ValueError, "invalid manifest reference"):
                run_eval.validate_direct_parallax_manifest(
                    artifact_dir, old_contract, {1, 2}, depth_files)

            # The manifest is provenance rather than a numeric metric input: changing it must
            # invalidate the full artifact digest without pretending the scored pixels changed.
            with open(os.path.join(artifact_dir, "sbs_00001.png"), "wb") as stream:
                stream.write(b"numeric")
            before = run_eval.scored_artifact_digests(artifact_dir)
            contract["direct_parallax_manifest"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                run_eval.validate_direct_parallax_manifest(
                    artifact_dir, contract, {1, 2}, depth_files)
            with open(manifest_path, "ab") as stream:
                stream.write(b" ")
            after = run_eval.scored_artifact_digests(artifact_dir)
            self.assertNotEqual(before[0], after[0])
            self.assertEqual(before[1], after[1])

    def test_direct_parallax_manifest_rejects_invalid_bound_and_dimensions(self):
        with tempfile.TemporaryDirectory() as artifact_dir:
            depth_path = os.path.join(artifact_dir, "depth_00001.f32")
            parallax_path = os.path.join(artifact_dir, "parallax_00001.f32")
            order_values = np.ones((8, 12), dtype="<f4")
            parallax_values = np.zeros((8, 12), dtype="<f4")
            order_values.tofile(depth_path)
            parallax_values.tofile(parallax_path)
            manifest = {
                **run_eval._DIRECT_GEOMETRY_MANIFEST_V6,
                "fields": [{
                    "frame_id": "00001",
                    "width": 12,
                    "height": 8,
                    "parallax_sha256": hashlib.sha256(parallax_values.tobytes()).hexdigest(),
                    "maximum_absolute_source_u": 0.0401,
                    "order_sha256": hashlib.sha256(order_values.tobytes()).hexdigest(),
                    "order_minimum": -1.0,
                    "order_maximum": 1.0,
                }],
            }
            manifest_path = os.path.join(
                artifact_dir, "direct_parallax_manifest.json")

            def write_contract():
                with open(manifest_path, "w", encoding="utf-8", newline="\n") as stream:
                    json.dump(manifest, stream, indent=2)
                    stream.write("\n")
                return {
                    "schema": run_eval.direct_geometry.CONTRACT_SCHEMA,
                    "warp_input": run_eval.direct_geometry.WARP_INPUT,
                    "direct_parallax_frames": 1,
                    "direct_parallax": run_eval._DIRECT_GEOMETRY_DESCRIPTOR_V6,
                    "direct_parallax_manifest": {
                        "file": "direct_parallax_manifest.json",
                        "schema": run_eval.direct_geometry.MANIFEST_SCHEMA,
                        "sha256": run_eval.file_sha256(manifest_path),
                    },
                }

            with self.assertRaisesRegex(ValueError, "invalid displacement bound"):
                run_eval.validate_direct_parallax_manifest(
                    artifact_dir, write_contract(), {1}, {1: depth_path})
            manifest["fields"][0]["maximum_absolute_source_u"] = 0.02
            manifest["fields"][0]["order_minimum"] = 2.0
            with self.assertRaisesRegex(ValueError, "invalid order range"):
                run_eval.validate_direct_parallax_manifest(
                    artifact_dir, write_contract(), {1}, {1: depth_path})
            manifest["fields"][0]["order_minimum"] = 1.0
            parallax_values.fill(0.0)
            parallax_values[:, 1::2] = 1.0
            parallax_values.tofile(parallax_path)
            manifest["fields"][0]["parallax_sha256"] = hashlib.sha256(
                parallax_values.tobytes()).hexdigest()
            manifest["fields"][0]["maximum_absolute_source_u"] = \
                run_eval.direct_geometry.SOURCE_U_LIMIT
            with self.assertRaisesRegex(ValueError, "horizontal slope bound"):
                run_eval.validate_direct_parallax_manifest(
                    artifact_dir, write_contract(), {1}, {1: depth_path})
            manifest["fields"][0]["width"] = 11
            with self.assertRaisesRegex(ValueError, "artifact size mismatch"):
                run_eval.validate_direct_parallax_manifest(
                    artifact_dir, write_contract(), {1}, {1: depth_path})

    def test_conditioned_parallax_cannot_replace_canonical_depth_order(self):
        # The near-preserving Lipschitz majorant raises background samples around a nearby
        # foreground cliff. That conditioned displacement is safe warp geometry, but its
        # numeric ordering can therefore disagree with the semantic order that produced it.
        canonical_order = np.array([[2.0, 0.5, 1.0]], dtype=np.float32)
        pre_limiter = np.array([[0.020, 0.010, 0.015]], dtype=np.float32)
        texel_slope = 0.001
        conditioned = np.max(np.stack([
            pre_limiter[:, source:source + 1] - texel_slope * np.abs(
                np.arange(pre_limiter.shape[1]) - source)
            for source in range(pre_limiter.shape[1])
        ], axis=0), axis=0)

        self.assertGreater(canonical_order[0, 2], canonical_order[0, 1])
        self.assertLess(conditioned[0, 2], conditioned[0, 1])
        self.assertEqual(
            run_eval._DIRECT_GEOMETRY_DESCRIPTOR_V6["depth_artifact_semantics"],
            "canonical-pre-limiter-order-float32-v1")
        self.assertFalse(
            run_eval._DIRECT_GEOMETRY_DESCRIPTOR_V6["renderer_uses_order"])
        self.assertEqual(
            run_eval._DIRECT_GEOMETRY_DESCRIPTOR_V6["order_role"],
            "diagnostic-semantic-depth-only-v1")
        self.assertNotIn(
            "occlusion_order", run_eval._DIRECT_GEOMETRY_DESCRIPTOR_V6)
        self.assertNotIn(
            "displacement_high_is_near", run_eval._DIRECT_GEOMETRY_DESCRIPTOR_V6)

    def test_clip_hash_covers_stereo_reference_and_contract(self):
        with tempfile.TemporaryDirectory() as clip:
            gt_right = os.path.join(clip, "gt_right")
            os.makedirs(gt_right)
            Image.fromarray(np.zeros((8, 12, 3), np.uint8)).save(
                os.path.join(clip, "frame_00000.png"))
            reference_path = os.path.join(gt_right, "frame_00000.png")
            Image.fromarray(np.zeros((8, 12, 3), np.uint8)).save(reference_path)
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"reference_stereo_available": True}, fh)
            original = run_eval.sha1_dir(clip)
            Image.fromarray(np.full((8, 12, 3), 255, np.uint8)).save(reference_path)
            self.assertNotEqual(original, run_eval.sha1_dir(clip))
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"reference_stereo_available": False}, fh)
            changed_pixels = run_eval.sha1_dir(clip)
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"reference_stereo_available": True}, fh)
            self.assertNotEqual(changed_pixels, run_eval.sha1_dir(clip))

    def test_clip_hash_authenticates_content_classification(self):
        with tempfile.TemporaryDirectory() as clip:
            Image.fromarray(np.zeros((8, 12, 3), np.uint8)).save(
                os.path.join(clip, "frame_00000.png"))
            meta_path = os.path.join(clip, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({"content_type": "real-capture"}, fh)
            capture_hashes = run_eval.source_evidence_digests(clip)
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({"content_type": "synthetic"}, fh)
            self.assertNotEqual(capture_hashes, run_eval.source_evidence_digests(clip))

    def test_clip_hash_excludes_labels_but_authenticates_shot_state_semantics(self):
        with tempfile.TemporaryDirectory() as clip:
            for frame_id in (1, 2):
                Image.fromarray(np.zeros((8, 12, 3), np.uint8)).save(
                    os.path.join(clip, f"frame_{frame_id:05d}.png"))
            meta_path = os.path.join(clip, "meta.json")
            contract = {
                "kind": "hard-cut", "monitor_from_frame": 2,
                "expected_pulse_frames": [2],
            }
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "name": "human label A", "description": "wording A",
                    "content_type": "synthetic", "shot_state_contract": contract,
                }, fh)
            original = run_eval.source_evidence_digests(clip)
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "name": "human label B", "description": "wording B",
                    "content_type": "synthetic", "shot_state_contract": contract,
                }, fh)
            self.assertEqual(original, run_eval.source_evidence_digests(clip))
            contract["expected_pulse_frames"] = []
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "name": "human label B", "description": "wording B",
                    "content_type": "synthetic", "shot_state_contract": contract,
                }, fh)
            self.assertNotEqual(original, run_eval.source_evidence_digests(clip))

    def test_exposure_probe_is_reproducible_and_pixel_authenticated(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        committed = os.path.join(
            repo, "tools", "sbsbench", "clips", "exposure_flash_strobe")
        committed_meta = run_eval.load_clip_metadata(committed, suite="core")
        self.assertEqual(committed_meta["content_type"], "synthetic")
        self.assertEqual(committed_meta["evaluation_role"], "conformance-only")
        self.assertEqual(
            committed_meta["shot_state_contract"]["monitor_from_frame"], 2)
        self.assertEqual(
            committed_meta["shot_state_contract"]["stable_from_frame"], 10)
        self.assertFalse(os.path.exists(os.path.join(committed, "gt_depth")))
        committed_frames = sbsbench.indexed_files(
            os.path.join(committed, "frame_*.*"), "frame_")
        sbsbench.validate_exposure_only_source(
            committed_frames, committed_meta["shot_state_contract"])

        with tempfile.TemporaryDirectory() as generated_root, mock.patch.object(
                make_synth_clips, "CLIPS", generated_root):
            make_synth_clips.exposure_flash_strobe()
            make_synth_clips.write_meta(
                "exposure_flash_strobe",
                **make_synth_clips.clip_metadata("exposure_flash_strobe"))
            generated = os.path.join(generated_root, "exposure_flash_strobe")
            generated_meta = run_eval.load_clip_metadata(generated, suite="core")
            generated_frames = sbsbench.indexed_files(
                os.path.join(generated, "frame_*.*"), "frame_")
            self.assertEqual(generated_meta, committed_meta)
            self.assertEqual(set(generated_frames), set(committed_frames))
            for frame_id in generated_frames:
                with open(generated_frames[frame_id], "rb") as actual, open(
                        committed_frames[frame_id], "rb") as expected:
                    self.assertEqual(actual.read(), expected.read(), frame_id)

            # A single non-exposure pixel invalidates the semantic contract before state metrics.
            tampered = np.asarray(Image.open(generated_frames[12]).convert("RGB")).copy()
            tampered[0, 0, 0] ^= 1
            Image.fromarray(tampered).save(generated_frames[12])
            with self.assertRaisesRegex(ValueError, "global-RGB-gain"):
                sbsbench.validate_exposure_only_source(
                    generated_frames, generated_meta["shot_state_contract"])

    def test_sustained_motion_cut_is_reproducible_and_source_authenticated(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        committed = os.path.join(
            repo, "tools", "sbsbench", "clips", "sustained_motion_scene_cut")
        committed_meta = run_eval.load_clip_metadata(committed, suite="core")
        contract = committed_meta["shot_state_contract"]
        self.assertEqual(committed_meta["evaluation_role"], "conformance-only")
        self.assertEqual(contract["kind"], "latched-motion-hard-cut")
        self.assertEqual(contract["monitor_from_frame"], 2)
        self.assertEqual(contract["expected_pulse_frames"], [11, 28])
        self.assertEqual(contract["escape_candidate_frame"], 27)
        self.assertEqual(contract["escape_pulse_frame"], 28)
        committed_frames = sbsbench.indexed_files(
            os.path.join(committed, "frame_*.*"), "frame_")
        sbsbench.validate_latched_motion_hard_cut_source(
            committed_frames, contract)

        with tempfile.TemporaryDirectory() as generated_root, mock.patch.object(
                make_synth_clips, "CLIPS", generated_root):
            # The generator reads its two source anchors from the clip root, so copy only those
            # authenticated inputs into the isolated regeneration tree.
            for source, frame in (("c841", 1), ("c647", 13)):
                source_dir = os.path.join(generated_root, source)
                os.makedirs(source_dir)
                shutil.copyfile(
                    os.path.join(repo, "tools", "sbsbench", "clips", source,
                                 f"frame_{frame:05d}.jpg"),
                    os.path.join(source_dir, f"frame_{frame:05d}.jpg"))
            make_synth_clips.sustained_motion_scene_cut()
            make_synth_clips.write_meta(
                "sustained_motion_scene_cut",
                **make_synth_clips.clip_metadata("sustained_motion_scene_cut"))
            generated = os.path.join(generated_root, "sustained_motion_scene_cut")
            generated_meta = run_eval.load_clip_metadata(generated, suite="core")
            generated_frames = sbsbench.indexed_files(
                os.path.join(generated, "frame_*.*"), "frame_")
            self.assertEqual(generated_meta, committed_meta)
            self.assertEqual(set(generated_frames), set(committed_frames))
            for frame_id in generated_frames:
                with open(generated_frames[frame_id], "rb") as actual, open(
                        committed_frames[frame_id], "rb") as expected:
                    self.assertEqual(actual.read(), expected.read(), frame_id)

            tampered = np.asarray(Image.open(generated_frames[18]).convert("RGB")).copy()
            tampered[0, 0, 0] ^= 1
            Image.fromarray(tampered).save(generated_frames[18])
            with self.assertRaisesRegex(ValueError, "horizontal-roll"):
                sbsbench.validate_latched_motion_hard_cut_source(
                    generated_frames, generated_meta["shot_state_contract"])

    def assert_structureless_bridge_is_reproducible_and_source_authenticated(
            self, clip_name, uniform_rgb):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        committed = os.path.join(
            repo, "tools", "sbsbench", "clips", clip_name)
        committed_meta = run_eval.load_clip_metadata(committed, suite="core")
        contract = committed_meta["shot_state_contract"]
        self.assertEqual(committed_meta["evaluation_role"], "conformance-only")
        self.assertEqual(contract["kind"], "structureless-history-bridge")
        self.assertEqual(contract["expected_pulse_frames"], [18, 22])
        self.assertEqual(contract["uniform_rgb"], list(uniform_rgb))
        committed_frames = sbsbench.indexed_files(
            os.path.join(committed, "frame_*.*"), "frame_")
        run_eval.validate_structureless_history_bridge_source(
            committed, contract)

        with tempfile.TemporaryDirectory() as generated_root, mock.patch.object(
                make_synth_clips, "CLIPS", generated_root):
            source_anchors = (
                make_synth_clips.BRIDGE_SCENE_A_BY_CLIP[clip_name],
                make_synth_clips.BRIDGE_SCENE_B,
            )
            for source, frame in source_anchors:
                source_dir = os.path.join(generated_root, source)
                os.makedirs(source_dir, exist_ok=True)
                shutil.copyfile(
                    os.path.join(repo, "tools", "sbsbench", "clips", source,
                                 f"frame_{frame:05d}.jpg"),
                    os.path.join(source_dir, f"frame_{frame:05d}.jpg"))
            make_synth_clips.GENERATORS[clip_name]()
            make_synth_clips.write_meta(
                clip_name, **make_synth_clips.clip_metadata(clip_name))
            generated = os.path.join(generated_root, clip_name)
            generated_meta = run_eval.load_clip_metadata(generated, suite="core")
            generated_frames = sbsbench.indexed_files(
                os.path.join(generated, "frame_*.*"), "frame_")
            self.assertEqual(generated_meta, committed_meta)
            self.assertEqual(set(generated_frames), set(committed_frames))
            for frame_id in generated_frames:
                with open(generated_frames[frame_id], "rb") as actual, open(
                        committed_frames[frame_id], "rb") as expected:
                    self.assertEqual(actual.read(), expected.read(), frame_id)

            tampered = np.asarray(Image.open(generated_frames[18]).convert("RGB")).copy()
            tampered[0, 0, 0] = 1
            Image.fromarray(tampered).save(generated_frames[18])
            with self.assertRaisesRegex(ValueError, "structureless-history"):
                run_eval.validate_structureless_history_bridge_source(
                    generated, generated_meta["shot_state_contract"])

    def test_black_structureless_bridge_is_reproducible_and_source_authenticated(self):
        self.assert_structureless_bridge_is_reproducible_and_source_authenticated(
            "structureless_history_bridge", (0, 0, 0))

    def test_white_structureless_bridge_is_reproducible_and_source_authenticated(self):
        self.assert_structureless_bridge_is_reproducible_and_source_authenticated(
            "structureless_white_history_bridge", (255, 255, 255))

    @staticmethod
    def cut_state(age, flags, history=1.0):
        return {
            "cut_contract_tag_bits": sbsbench.cut_state_contract.CUT_CONTRACT_TAG,
            "scene_age": float(age),
            "reserved_cut_bridge_2": 0.0, "initialized": 1.0,
            "reserved_cut_bridge_4": 0.0, "reserved_cut_bridge_5": 0.0,
            "depth_change_baseline_ema": 0.08, "reserved_cut_bridge_7": 0.0,
            "reserved_cut_bridge_8": 0.0, "reserved_cut_bridge_9": 0.0,
            "cut_flags": float(flags), "model_input_history_state": float(history),
        }

    def test_shot_state_contract_observes_exactly_one_real_cut(self):
        rows = [{"_frame_id": frame_id} for frame_id in range(1, 7)]
        trace = {
            1: self.cut_state(0, 3),
            2: self.cut_state(1, 3),
            3: self.cut_state(2, 3),
            4: self.cut_state(0, 16),
            5: self.cut_state(1, 16),
            6: self.cut_state(2, 16),
        }
        summary = sbsbench.apply_shot_state_contract(
            rows, list(range(1, 7)), trace, {
                "kind": "hard-cut", "monitor_from_frame": 2,
                "expected_pulse_frames": [4],
            })
        self.assertEqual(summary["shot_state_accepted_pulse"], 1.0)
        self.assertEqual(summary["shot_state_expected_pulse"], 1.0)
        self.assertEqual(max(row.get("shot_state_pulse_mismatch", 0.0) for row in rows), 0.0)
        self.assertEqual(
            max(row.get("shot_state_trace_inconsistent", 0.0) for row in rows), 0.0)
        self.assertEqual(
            sbsbench.aggregate(rows)["shot_state_accepted_pulse"], 1.0)

    def test_v2_cut_state_requires_model_input_history(self):
        rows = [{"_frame_id": frame_id} for frame_id in range(1, 5)]
        trace = {
            frame_id: self.cut_state(frame_id - 1, 3)
            for frame_id in range(1, 5)
        }
        sbsbench.apply_shot_state_contract(
            rows, list(range(1, 5)), trace, {
                "kind": "hard-cut", "monitor_from_frame": 2,
                "expected_pulse_frames": [],
            })
        self.assertEqual(
            min(row["shot_state_initialized_ok"] for row in rows
                if "shot_state_initialized_ok" in row), 100.0)
        uninitialized_rows = [{"_frame_id": frame_id} for frame_id in range(1, 5)]
        trace[3] = dict(trace[3], model_input_history_state=0.0)
        sbsbench.apply_shot_state_contract(
            uninitialized_rows, list(range(1, 5)), trace, {
                "kind": "hard-cut", "monitor_from_frame": 2,
                "expected_pulse_frames": [],
            })
        broken = next(row for row in uninitialized_rows if row["_frame_id"] == 3)
        self.assertEqual(broken["shot_state_initialized_ok"], 0.0)

    def test_structureless_bridge_contract_observes_bounded_slate_and_return_cuts(self):
        rows = [{"_frame_id": frame_id} for frame_id in range(1, 23)]
        trace = {
            frame_id: self.cut_state(frame_id - 1, 3)
            for frame_id in range(1, 18)
        }
        trace[18] = self.cut_state(0, 16)
        trace[19] = self.cut_state(1, 16)
        trace[20] = self.cut_state(2, 19)
        trace[21] = self.cut_state(0, 16)
        trace[22] = self.cut_state(1, 16)
        contract = {
            "kind": "structureless-history-bridge", "monitor_from_frame": 2,
            "expected_pulse_frames": [18, 21], "flash_frame": 11,
            "flash_return_frame": 12, "slate_frame": 17,
            "persistent_frame": 18, "new_scene_frame": 21,
        }
        summary = sbsbench.apply_shot_state_contract(
            rows, list(range(1, 23)), trace, contract)
        self.assertEqual(summary["shot_state_accepted_pulse"], 2.0)
        self.assertEqual(max(
            row.get("shot_state_trace_inconsistent", 0.0) for row in rows), 0.0)

    def test_exposure_contract_rejects_relatched_state_and_latch_drift(self):
        rows = [{"_frame_id": frame_id} for frame_id in range(1, 7)]
        trace = {frame_id: self.cut_state(frame_id - 1, 3)
                 for frame_id in range(1, 7)}
        contract = {
            "kind": "exposure-only", "monitor_from_frame": 2,
            "stable_from_frame": 3,
            "expected_pulse_frames": [],
        }
        summary = sbsbench.apply_shot_state_contract(
            rows, list(range(1, 7)), trace, contract)
        self.assertEqual(summary["shot_state_accepted_pulse"], 0.0)
        reset_rows = [{"_frame_id": frame_id} for frame_id in range(1, 7)]
        trace[5] = self.cut_state(0, 16)
        sbsbench.apply_shot_state_contract(
            reset_rows, list(range(1, 7)), trace, contract)
        reset = next(row for row in reset_rows if row["_frame_id"] == 5)
        self.assertEqual(reset["shot_state_pulse_mismatch"], 1.0)

    def test_exposure_pulse_monitoring_cannot_hide_startup_relatched_state(self):
        rows = [{"_frame_id": frame_id} for frame_id in range(1, 6)]
        trace = {
            1: self.cut_state(1, 3),
            2: self.cut_state(0, 16),
            3: self.cut_state(1, 16),
            4: self.cut_state(2, 16),
            5: self.cut_state(3, 16),
        }
        summary = sbsbench.apply_shot_state_contract(
            rows, list(range(1, 6)), trace, {
                "kind": "exposure-only", "monitor_from_frame": 2,
                "stable_from_frame": 4, "expected_pulse_frames": [],
            })
        self.assertEqual(summary["shot_state_accepted_pulse"], 1.0)
        startup = next(row for row in rows if row["_frame_id"] == 2)
        self.assertEqual(startup["shot_state_pulse_mismatch"], 1.0)

    def test_shot_trace_recognizes_later_cut_after_independent_rearm(self):
        rows = [{"_frame_id": frame_id} for frame_id in range(1, 9)]
        trace = {
            1: self.cut_state(0, 3),
            2: self.cut_state(1, 3),
            3: self.cut_state(2, 3),
            4: self.cut_state(0, 16),
            5: self.cut_state(1, 17),
            6: self.cut_state(2, 19),
            7: self.cut_state(0, 16),
            8: self.cut_state(1, 16),
        }
        summary = sbsbench.apply_shot_state_contract(
            rows, list(range(1, 9)), trace, {
                "kind": "hard-cut", "monitor_from_frame": 2,
                "expected_pulse_frames": [4, 7],
            })
        self.assertEqual(summary["shot_state_accepted_pulse"], 2.0)
        self.assertEqual(
            sbsbench.aggregate(rows)["shot_state_accepted_pulse"], 2.0)
        self.assertEqual(max(
            row.get("shot_state_pulse_mismatch", 0.0) for row in rows), 0.0)
        self.assertEqual(max(
            row.get("shot_state_trace_inconsistent", 0.0) for row in rows), 0.0)

    def test_shot_trace_accepts_relative_cut_with_latched_flags_unchanged(self):
        rows = [{"_frame_id": frame_id} for frame_id in range(1, 13)]
        trace = {
            1: self.cut_state(0, 3),
            2: self.cut_state(1, 3),
            3: self.cut_state(0, 16),
        }
        for frame_id in range(4, 11):
            trace[frame_id] = self.cut_state(frame_id - 3, 16)
        trace[11] = self.cut_state(8, 80, history=4.0)
        trace[12] = self.cut_state(0, 16)
        summary = sbsbench.apply_shot_state_contract(
            rows, list(range(1, 13)), trace, {
                "kind": "hard-cut", "monitor_from_frame": 2,
                "expected_pulse_frames": [3, 12],
            })
        self.assertEqual(summary["shot_state_accepted_pulse"], 2.0)
        self.assertEqual(max(
            row.get("shot_state_pulse_mismatch", 0.0) for row in rows), 0.0)
        self.assertEqual(max(
            row.get("shot_state_trace_inconsistent", 0.0) for row in rows), 0.0)

    def test_latched_motion_contract_proves_relative_escape_precondition(self):
        rows = [{"_frame_id": frame_id} for frame_id in range(1, 13)]
        trace = {
            1: self.cut_state(0, 3),
            2: self.cut_state(1, 3),
            3: self.cut_state(0, 16),
        }
        for frame_id in range(4, 11):
            trace[frame_id] = self.cut_state(frame_id - 3, 16)
        trace[11] = self.cut_state(8, 80, history=4.0)
        trace[12] = self.cut_state(0, 16)
        contract = {
            "kind": "latched-motion-hard-cut", "monitor_from_frame": 2,
            "expected_pulse_frames": [3, 12],
            "setup_pulse_frame": 3,
            "persistent_motion_frames": [3, 11],
            "escape_candidate_frame": 11,
            "escape_pulse_frame": 12,
        }
        summary = sbsbench.apply_shot_state_contract(
            rows, list(range(1, 13)), trace, contract)
        self.assertEqual(summary["shot_state_relative_escape_ok"], 100.0)
        self.assertEqual(min(
            row.get("shot_state_relative_escape_ok", 100.0)
            for row in rows if row["_frame_id"] >= 2), 100.0)

        trace[11] = self.cut_state(8, 16)
        failed_rows = [{"_frame_id": frame_id} for frame_id in range(1, 13)]
        failed = sbsbench.apply_shot_state_contract(
            failed_rows, list(range(1, 13)), trace, contract)
        self.assertEqual(failed["shot_state_relative_escape_ok"], 0.0)

    def test_unattributed_core_clips_are_explicitly_unclassified(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for clip in ("c525", "c747", "c841"):
            with self.subTest(clip=clip), open(
                    os.path.join(repo, "tools", "sbsbench", "clips", clip, "meta.json"),
                    encoding="utf-8") as fh:
                self.assertEqual(json.load(fh).get("content_type"), "unclassified")

    def test_clip_hash_covers_all_semantic_reference_sidecars(self):
        sidecars = (
            "gt_depth_valid", "gt_depth_valid_all", "gt_depth_valid_nonocc",
            "gt_occlusion", "gt_outofframe", "gt_right_disparity", "gt_detail",
            "gt_match", "gt_sky",
        )
        with tempfile.TemporaryDirectory() as clip:
            Image.fromarray(np.zeros((8, 12, 3), np.uint8)).save(
                os.path.join(clip, "frame_00000.png"))
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({}, fh)
            for directory in sidecars:
                path = os.path.join(clip, directory)
                os.makedirs(path)
                sidecar = os.path.join(path, "frame_00000.png")
                Image.fromarray(np.zeros((8, 12), np.uint8)).save(sidecar)
                original = run_eval.sha1_dir(clip)
                Image.fromarray(np.full((8, 12), 255, np.uint8)).save(sidecar)
                self.assertNotEqual(original, run_eval.sha1_dir(clip), directory)

    def test_metric_contract_excludes_runner_diagnostics(self):
        metric_files = run_eval.metric_contract_files()
        self.assertEqual(run_eval.metric_contract_sha(),
                         run_eval.sha256_files(metric_files))
        self.assertNotEqual(
            run_eval.metric_contract_sha(),
            run_eval.sha256_files(metric_files + [os.path.abspath(run_eval.__file__)]))

    def test_metric_contract_covers_delegated_metric_modules(self):
        names = {os.path.basename(path) for path in run_eval.metric_contract_files()}
        self.assertEqual(names, {
            "sbsbench.py", "sbs_interocular_metrics.py",
            "sbs_interocular_phase_chroma.py",
            "sbs_interocular_photometric_rivalry.py", "sbs_stereo_window_metrics.py",
            "sbs_warp_shear_metrics.py", "direct_geometry_contract.py",
            "cut_state_contract.py", "thresholds.json",
        })

    def test_label_contract_covers_parallel_result_association(self):
        names = {os.path.basename(path) for path in run_eval.label_contract_files()}
        self.assertIn("eval_parallel.py", names)
        self.assertNotIn(
            "eval_parallel.py",
            {os.path.basename(path) for path in run_eval.metric_contract_files()})
        self.assertEqual(
            run_eval.label_contract_sha(),
            run_eval.sha256_files(run_eval.label_contract_files()))

    def test_canonical_model_is_the_single_authenticated_calibration(self):
        self.assertEqual(len(run_eval.MODEL_CALIBRATIONS), 1)
        self.assertEqual(
            run_eval.expected_depth_model(),
            run_eval.MODEL_CALIBRATIONS[0].depth_model)
        self.assertEqual(
            run_eval.expected_depth_model_url(),
            run_eval.MODEL_CALIBRATIONS[0].depth_model_url)

    def test_baseline_update_refuses_gpu_contention(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "tools", "sbsbench", "run_eval.py"),
                  encoding="utf-8") as fh:
            evaluator = fh.read()
        self.assertIn("if args.update_baselines:", evaluator)
        self.assertIn("refusing --update-baselines while another sunshine.exe is running",
                      evaluator)
        self.assertIn("--update-baselines requires the canonical config", evaluator)
        self.assertIn('meta.get("extra_args") != []', evaluator)

    def test_baseline_update_requires_clean_stable_head(self):
        dirty = subprocess.CompletedProcess([], 0, stdout=" M src/video.cpp\n", stderr="")
        with mock.patch.object(run_eval.subprocess, "run", return_value=dirty):
            with self.assertRaises(SystemExit):
                run_eval.require_clean_baseline_update(True)

        clean = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        head_a = subprocess.CompletedProcess([], 0, stdout="a" * 40 + "\n", stderr="")
        with mock.patch.object(run_eval.subprocess, "run", side_effect=[clean, head_a]):
            self.assertEqual(run_eval.require_clean_baseline_update(True), "a" * 40)
        head_b = subprocess.CompletedProcess([], 0, stdout="b" * 40 + "\n", stderr="")
        with mock.patch.object(run_eval.subprocess, "run", side_effect=[clean, head_b]):
            with self.assertRaises(SystemExit):
                run_eval.require_clean_baseline_update(True, expected_head="a" * 40)

    def test_runtime_identity_stability_guard_names_changed_component(self):
        engine = {
            "engine_name": "unit.engine", "engine_sha256": "d" * 64,
            "onnx_sha256": "e" * 64,
        }
        with (mock.patch.object(run_eval, "file_sha256", return_value="a" * 64),
              mock.patch.object(run_eval, "runtime_shader_sha256", return_value="b" * 64),
              mock.patch.object(
                  run_eval, "shader_source_closure_sha256", return_value="c" * 64),
              mock.patch.object(run_eval, "engine_provenance", return_value=engine)):
            expected = run_eval.runtime_identity_snapshot("sunshine.exe", "build", "model")
            run_eval.require_runtime_identity_unchanged(
                expected, "sunshine.exe", "build", "model")
            changed = dict(engine, engine_sha256="f" * 64)
            with mock.patch.object(run_eval, "engine_provenance", return_value=changed):
                with self.assertRaisesRegex(ValueError, "engine_sha256"):
                    run_eval.require_runtime_identity_unchanged(
                        expected, "sunshine.exe", "build", "model")

        runtime = {
            "executable_sha256": "a" * 64,
            "runtime_shader_sha256": "b" * 64,
            "preprocess_source_closure_sha256": "c" * 64,
            **engine,
        }
        with (mock.patch.object(run_eval, "runtime_identity_snapshot", return_value=runtime),
              mock.patch.object(run_eval, "sha256_files", return_value="conf-a"),
              mock.patch.object(run_eval, "metric_contract_sha", return_value="metric-a"),
              mock.patch.object(run_eval, "label_contract_sha", return_value="label-a")):
            expected = run_eval.evaluation_identity_snapshot(
                "sunshine.exe", "build", "model", "bench.conf")
            run_eval.require_evaluation_identity_unchanged(
                expected, "sunshine.exe", "build", "model", "bench.conf")
            with mock.patch.object(run_eval, "sha256_files", return_value="conf-b"):
                with self.assertRaisesRegex(ValueError, "conf_sha256"):
                    run_eval.require_evaluation_identity_unchanged(
                        expected, "sunshine.exe", "build", "model", "bench.conf")

    def test_uncalibrated_remote_or_local_url_comes_from_harness_identity(self):
        for model, url in (
                ("custom-remote-model", "https://example.invalid/custom-remote-model.onnx"),
                ("custom-midas", "")):
            with self.subTest(model=model):
                identity = {
                    "schema": 1, "model": model, "depth_model_url": url,
                    "onnx_sha256": "a" * 64, "preprocess_profile": "",
                    "preprocess_source_closure_sha256": "b" * 64,
                    "raw_width": 256, "raw_height": 256,
                }
                manifest = {
                    "schema": run_eval.whole_clip_raw_contract.MANIFEST_SCHEMA,
                    "binding": run_eval.whole_clip_raw_contract.BINDING,
                    "evaluator_schema": run_eval.EVAL_SCHEMA,
                    "harness_contract_schema":
                        run_eval.whole_clip_raw_contract.HARNESS_CONTRACT_SCHEMA,
                    "contract_json_sha256": "c" * 64,
                    "raw_shape": {"height": 256, "width": 256},
                    "raw_shape_json_sha256": "d" * 64,
                    "producer_model_identity": identity,
                    "calibration_status": "abstain-unsupported-model-contract",
                    "calibration_id": None,
                    "abstention_reason": "no exact calibration",
                    "frames": [{
                        "frame_id": "00001", "file": "raw_00001.f32",
                        "sha256": "e" * 64,
                    }],
                }
                meta = {
                    "model": model, "depth_model_url": "",
                    "onnx_sha256": "a" * 64,
                    "preprocess_source_closure_sha256": "b" * 64,
                }
                summaries = run_eval.finalize_whole_clip_raw_identity(
                    meta, {"clip": manifest})
                self.assertEqual(meta["depth_model_url"], url)
                self.assertIsNone(meta["preprocess_profile"])
                self.assertIsNone(meta["depth_coordinate_v2_calibration_id"])
                self.assertEqual(
                    summaries["clip"]["calibration_status"],
                    "abstain-unsupported-model-contract")

    def test_missing_metric_and_perf_evidence_fail_closed(self):
        thresholds = {
            "metrics": {
                "comfort": {"role": "hard", "hard_max": 3.0, "better": "lower"},
                "quality": {"role": "primary", "better": "higher",
                            "abs_floor": 0.1, "rel_tol": 0.1},
            },
            "perf_ms": {"warp": {"abs_floor": 0.1, "rel_tol": 0.1}},
        }
        _, _, hard = run_eval.score_clip_gates([], {}, thresholds, {})
        self.assertEqual(hard[0]["metric"], "comfort")
        self.assertTrue(hard[0]["missing"])
        evidence = run_eval.primary_evidence_failures(
            {}, thresholds, "clip", {}, baseline={"quality": 10.0})
        self.assertEqual(evidence[0]["metric"], "quality")
        perf = run_eval.perf_evidence_failures({"warp": 1.0}, {}, thresholds, "clip")
        self.assertEqual(perf[0]["metric"], "perf:warp")
        self.assertEqual(perf[0]["source"], "current")

    def test_engine_preflight_requires_fresh_exact_runtime_manifest(self):
        with tempfile.TemporaryDirectory() as build:
            assets = os.path.join(build, "assets")
            os.makedirs(assets)
            model = "depth_model"
            exe = os.path.join(build, "sunshine.exe")
            onnx = os.path.join(assets, model + ".onnx")
            engine_name = model + ".exact-compatible.engine"
            engine = os.path.join(assets, engine_name)
            manifest = os.path.join(assets, model + ".active-engine.json")
            for path, payload in ((exe, b"binary"), (onnx, b"onnx"), (engine, b"engine")):
                with open(path, "wb") as fh:
                    fh.write(payload)
            with open(manifest, "w", encoding="utf-8") as fh:
                json.dump({"schema": 1, "model": model, "engine": engine_name,
                           "onnx_sha256": run_eval.file_sha256(onnx)}, fh)
            exe_time = os.path.getmtime(exe)
            os.utime(manifest, (exe_time + 2, exe_time + 2))
            self.assertEqual(run_eval.check_engines(build, model), [])

            with open(os.path.join(assets, model + ".unrelated.engine"), "wb") as fh:
                fh.write(b"unrelated")
            os.unlink(engine)
            issues = run_eval.check_engines(build, model)
            self.assertTrue(any("exact engine missing/empty" in issue for issue in issues))

    def test_engine_preflight_rejects_stale_manifest_and_onnx_mismatch(self):
        with tempfile.TemporaryDirectory() as build:
            assets = os.path.join(build, "assets")
            os.makedirs(assets)
            model = "depth_model"
            exe = os.path.join(build, "sunshine.exe")
            onnx = os.path.join(assets, model + ".onnx")
            engine_name = model + ".exact.engine"
            engine = os.path.join(assets, engine_name)
            manifest = os.path.join(assets, model + ".active-engine.json")
            for path, payload in ((exe, b"new binary"), (onnx, b"new onnx"),
                                  (engine, b"engine")):
                with open(path, "wb") as fh:
                    fh.write(payload)
            with open(manifest, "w", encoding="utf-8") as fh:
                json.dump({"schema": 1, "model": model, "engine": engine_name,
                           "onnx_sha256": "0" * 64}, fh)
            now = os.path.getmtime(exe)
            os.utime(manifest, (now - 2, now - 2))
            issues = run_eval.check_engines(build, model)
            self.assertTrue(any("predates sunshine.exe" in issue for issue in issues))
            self.assertTrue(any("ONNX SHA-256" in issue for issue in issues))

    def test_runtime_pipeline_provenance_covers_shaders_engine_and_onnx(self):
        with tempfile.TemporaryDirectory() as build:
            assets = os.path.join(build, "assets")
            shaders = os.path.join(assets, "shaders", "directx", "include")
            os.makedirs(shaders)
            model = "depth_model"
            shader = os.path.join(shaders, "common.hlsl")
            onnx = os.path.join(assets, model + ".onnx")
            engine_name = model + ".exact.engine"
            engine = os.path.join(assets, engine_name)
            manifest = os.path.join(assets, model + ".active-engine.json")
            with open(shader, "wb") as fh:
                fh.write(b"float x;\r\n")
            with open(onnx, "wb") as fh:
                fh.write(b"onnx")
            with open(engine, "wb") as fh:
                fh.write(b"engine")
            with open(manifest, "w", encoding="utf-8") as fh:
                json.dump({"engine": engine_name,
                           "onnx_sha256": run_eval.file_sha256(onnx)}, fh)

            shader_sha = run_eval.runtime_shader_sha256(build)
            with open(shader, "wb") as fh:
                fh.write(b"float x;\n")
            self.assertEqual(shader_sha, run_eval.runtime_shader_sha256(build))
            with open(shader, "ab") as fh:
                fh.write(b"float y;\n")
            self.assertNotEqual(shader_sha, run_eval.runtime_shader_sha256(build))

            provenance = run_eval.engine_provenance(build, model)
            self.assertEqual(provenance["engine_name"], engine_name)
            self.assertEqual(provenance["engine_sha256"], run_eval.file_sha256(engine))
            self.assertEqual(provenance["onnx_sha256"], run_eval.file_sha256(onnx))

    def test_allow_build_engine_preflight_is_untimed_and_revalidates_manifest(self):
        with tempfile.TemporaryDirectory() as build:
            frames = os.path.join(build, "frames")
            os.makedirs(frames)
            exe = os.path.join(build, "sunshine.exe")
            conf = os.path.join(build, "bench.conf")
            for path in (exe, conf):
                with open(path, "wb") as fh:
                    fh.write(b"x")
            completed = subprocess.CompletedProcess([], 0, "", "")
            with mock.patch.object(run_eval.subprocess, "run", return_value=completed) as launch, \
                    mock.patch.object(run_eval, "check_engines", return_value=[]):
                run_eval.run_engine_preflight(exe, conf, build, frames, "model")
            command = launch.call_args.args[0]
            self.assertIn("--limit", command)
            self.assertEqual(command[command.index("--limit") + 1], "1")
            self.assertEqual(launch.call_args.kwargs["timeout"], 900)

    def test_ab_decision_rejects_asymmetric_primary_evidence(self):
        specs = {"quality": {"role": "primary", "axis": "warp", "better": "higher"},
                 "comfort": {"role": "hard", "hard_max": 3.0, "better": "lower"}}
        decision = sbsbench.evaluate_ab_decision(
            {"clip": {"quality": 1.0, "comfort": 1.0}},
            {"clip": {"comfort": 1.0}}, ["clip"], specs)
        self.assertEqual(decision["verdict"], "reject_evidence")
        self.assertEqual(decision["missing_evidence"][0]["missing"], "treatment")

    def test_report_candidate_is_bound_to_both_canonical_run_gates(self):
        metric_decision = {"verdict": "screen_candidate", "hard_failures": [],
                           "missing_evidence": [], "axes": {}, "improved": 1,
                           "regressed": 0, "perceptual_qualification": "experimental"}

        def run(kind="comparison-only", verdict="comparison_only", **overrides):
            result = {"meta": {"run_kind": kind}, "verdict": verdict,
                      "hard_failures": [], "evidence_failures": [], "regressions": []}
            result.update(overrides)
            return result

        accepted = sbsbench.gate_ab_decision(metric_decision, run(), run())
        self.assertEqual(accepted["verdict"], "screen_candidate")
        self.assertTrue(accepted["screen_candidate"])
        self.assertFalse(accepted["perceptual_qualified_candidate"])

        for field, value in (
                ("evidence_failures", [{"metric": "perf:warp"}]),
                ("regressions", [{"metric": "perf:warp"}]),
                ("hard_failures", [{"metric": "comfort"}])):
            with self.subTest(field=field):
                verdict = ("evidence_failures" if field == "evidence_failures" else
                           "regressions" if field == "regressions" else "hard_failures")
                rejected_run = run(verdict=verdict, **{field: value})
                rejected = sbsbench.gate_ab_decision(metric_decision, run(), rejected_run)
                self.assertEqual(rejected["verdict"], "reject_run_gate")
                self.assertFalse(rejected["screen_candidate"])
                self.assertFalse(rejected["canonical_gate"]["passed"])

        malformed = run()
        del malformed["regressions"]
        self.assertFalse(sbsbench.canonical_run_gate(malformed)["passed"])

    def test_primary_evidence_applicability_is_explicit(self):
        specs = {
            "always": {"role": "primary"},
            "depth": {"role": "primary", "requires": "gt_depth"},
            "temporal": {"role": "primary", "requires": "multi_frame"},
            "warp": {
                "role": "primary", "requires": "warp_cross_row_shear_support_count",
            },
        }
        failures = run_eval.primary_evidence_failures(
            {}, {"metrics": specs}, "clip", {"source_frame_count": 1})
        self.assertEqual({failure["metric"] for failure in failures}, {"always", "warp"})
        explicitly_unsupported = run_eval.primary_evidence_failures(
            {"warp_cross_row_shear_support_count": 0.0}, {"metrics": specs}, "clip",
            {"source_frame_count": 1})
        self.assertEqual([failure["metric"] for failure in explicitly_unsupported], ["always"])
        failures = run_eval.primary_evidence_failures(
            {"warp_cross_row_shear_support_count": 511.0}, {"metrics": specs}, "clip",
            {"source_frame_count": 2, "gt_depth_kind": "disparity"})
        self.assertEqual({failure["metric"] for failure in failures},
                         {"always", "depth", "temporal"})
        failures = run_eval.primary_evidence_failures(
            {"warp_cross_row_shear_support_count": 512.0}, {"metrics": specs}, "clip",
            {"source_frame_count": 2, "gt_depth_kind": "disparity"})
        self.assertEqual({failure["metric"] for failure in failures},
                         {"always", "depth", "temporal", "warp"})
        failures = run_eval.primary_evidence_failures(
            {"always": float("nan")}, {"metrics": specs}, "clip",
            {"source_frame_count": 1})
        self.assertEqual({failure["metric"] for failure in failures}, {"always", "warp"})
        with self.assertRaisesRegex(ValueError, "unknown metric evidence requirement"):
            sbsbench.metric_evidence_applicable(
                "metric", {"requires": "typo"}, {}, {})

    def test_temporal_evidence_requires_measured_reliable_support(self):
        metrics = {
            "static_jitter_p95": {"role": "primary", "requires": "static_support"},
            "flow_temporal_p95": {"role": "primary", "requires": "flow_support"},
        }
        thresholds = {"metrics": metrics}
        clip_meta = {"source_frame_count": 8}
        insufficient = run_eval.primary_evidence_failures(
            {"static_support": 0.099, "flow_support": 0.0}, thresholds,
            "moving", clip_meta)
        self.assertEqual(insufficient, [])
        sufficient = run_eval.primary_evidence_failures(
            {"static_support": 0.1, "flow_support": 0.8}, thresholds,
            "supported", clip_meta)
        self.assertEqual({failure["metric"] for failure in sufficient}, set(metrics))
        missing_support = run_eval.primary_evidence_failures(
            {}, thresholds, "broken", clip_meta)
        self.assertEqual({failure["metric"] for failure in missing_support}, set(metrics))
        single_frame = run_eval.primary_evidence_failures(
            {}, thresholds, "still", {"source_frame_count": 1})
        self.assertEqual(single_frame, [])
        asymmetric = run_eval.primary_evidence_failures(
            {"static_support": 0.01}, thresholds, "asymmetric", clip_meta,
            baseline={"static_support": 0.7, "flow_support": 0.0})
        # Static support changed from applicable to unsupported; flow support is missing in the
        # current run while the baseline explicitly measured it as unsupported.  Neither
        # asymmetric evidence state is safe to silently compare/skip.
        self.assertEqual([failure["metric"] for failure in asymmetric],
                         ["static_jitter_p95", "flow_temporal_p95"])

    def test_gt_depth_lag_is_required_only_for_temporal_reference_clips(self):
        thresholds = {"metrics": {
            "depth_gt_lag_f1_p95": {
                "role": "primary", "requires": "gt_depth_temporal"},
        }}
        single = run_eval.primary_evidence_failures(
            {}, thresholds, "single", {"required_gt_depth": True, "source_frame_count": 1})
        self.assertEqual(single, [])
        temporal = run_eval.primary_evidence_failures(
            {}, thresholds, "temporal", {"required_gt_depth": True, "source_frame_count": 2})
        self.assertEqual([item["metric"] for item in temporal], ["depth_gt_lag_f1_p95"])

    def test_missing_perf_median_remains_missing_evidence(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"stages": {"warp": {"samples": 12}}}, fh)
            path = fh.name
        try:
            perf = run_eval.load_perf_metrics(path)
            self.assertIsNone(perf["warp"])
            failures = run_eval.perf_evidence_failures(
                {"warp": 1.0}, perf,
                {"perf_ms": {"warp": {"abs_floor": 0.1, "rel_tol": 0.1}}}, "clip")
            self.assertEqual(failures[0]["metric"], "perf:warp")
            self.assertEqual(failures[0]["source"], "current")
        finally:
            os.unlink(path)

    def test_clip_metadata_preflight_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as clip:
            with self.assertRaisesRegex(ValueError, "missing clip metadata"):
                run_eval.load_clip_metadata(clip, suite="extended")
            meta_path = os.path.join(clip, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump([], fh)
            with self.assertRaisesRegex(ValueError, "root must be an object"):
                run_eval.load_clip_metadata(clip)
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({"content_type": "capture"}, fh)
            with self.assertRaisesRegex(ValueError, "content_type .* is not one of"):
                run_eval.load_clip_metadata(clip)
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({}, fh)
            with self.assertRaisesRegex(ValueError, "explicit content_type"):
                run_eval.load_clip_metadata(clip)
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({"dataset": "public"}, fh)
            with self.assertRaisesRegex(ValueError, "consumed depth/flow GT"):
                run_eval.load_clip_metadata(clip, suite="extended")

            gt_right = os.path.join(clip, "gt_right")
            os.makedirs(gt_right)
            Image.fromarray(np.zeros((8, 12, 3), np.uint8)).save(
                os.path.join(gt_right, "frame_00000.png"))
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "dataset": "public", "content_type": "animation",
                    "reference_stereo_available": True,
                }, fh)
            inferred_reference = run_eval.load_clip_metadata(clip, suite="extended")
            self.assertEqual(inferred_reference["evaluation_role"], "reference-only")
            self.assertFalse(inferred_reference.get("required_gt_depth", False))

            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "dataset": "public", "content_type": "animation",
                    "reference_stereo_available": True,
                    "evaluation_role": "reference-only",
                }, fh)
            reference_meta = run_eval.load_clip_metadata(clip, suite="extended")
            self.assertEqual(reference_meta["evaluation_role"], "reference-only")
            self.assertFalse(reference_meta.get("required_gt_depth", False))
            self.assertEqual(
                sbsbench.metric_evidence_state(
                    "unused", {"requires": "gt_depth"}, {}, reference_meta),
                "unsupported")

            gt_depth = os.path.join(clip, "gt_depth")
            os.makedirs(gt_depth)
            np.save(os.path.join(gt_depth, "frame_00000.npy"), np.ones((8, 12), np.float32))
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "dataset": "public", "content_type": "real-capture",
                    "required_gt_depth": True,
                    "gt_depth_kind": "metric", "reference_stereo_available": True,
                }, fh)
            self.assertTrue(run_eval.load_clip_metadata(
                clip, suite="extended")["reference_stereo_available"])

            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "dataset": "public", "content_type": "real-capture",
                    "required_gt_depth": True,
                    "gt_depth_kind": "metric", "required_gt_stereo": True,
                }, fh)
            migrated = run_eval.load_clip_metadata(clip, suite="extended")
            self.assertNotIn("required_gt_stereo", migrated)
            self.assertTrue(migrated["reference_stereo_available"])

    def test_baseline_context_is_validated_before_harness_use(self):
        with tempfile.TemporaryDirectory() as baseline_dir:
            context = {
                "eval_schema": run_eval.EVAL_SCHEMA,
                "metric_sha256": "metric",
                "parallax_v2_renderer_source_closure_sha256": "renderer-a",
                "run_kind": "baseline-update",
            }
            payload = {
                "meta": {**context, "clip_sha1": "cliphash", "extra_args": [],
                         "git_dirty": False},
                "aggregate": {"quality": 1.0},
                "perf_ms": {"warp": 1.0},
            }
            path = os.path.join(baseline_dir, "clip.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            loaded = run_eval.preflight_baselines(
                baseline_dir, ["clip"], context, {"clip": "cliphash"})
            self.assertEqual(loaded["clip"]["aggregate"]["quality"], 1.0)
            payload["meta"]["git_dirty"] = True
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            with self.assertRaisesRegex(ValueError, "dirty or unauthenticated"):
                run_eval.preflight_baselines(
                    baseline_dir, ["clip"], context, {"clip": "cliphash"})
            payload["meta"]["git_dirty"] = False
            payload["meta"]["eval_schema"] -= 1
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            with self.assertRaisesRegex(ValueError, "stale/incompatible"):
                run_eval.preflight_baselines(
                    baseline_dir, ["clip"], context, {"clip": "cliphash"})
            payload["meta"]["eval_schema"] = run_eval.EVAL_SCHEMA
            payload["meta"]["parallax_v2_renderer_source_closure_sha256"] = "renderer-b"
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            with self.assertRaisesRegex(ValueError, "stale/incompatible"):
                run_eval.preflight_baselines(
                    baseline_dir, ["clip"], context, {"clip": "cliphash"})

    def test_numeric_baseline_context_excludes_reusable_label_implementation(self):
        candidate = {
            "suite": "core",
            "model": "model",
            "parallax_v2_source_closure_sha256": "producer-sha",
            "parallax_v2_renderer_source_closure_sha256": "renderer-sha",
            "eval_schema": run_eval.EVAL_SCHEMA,
            "depth_step": "current-once",
            "depth_compensation": "none",
            "cuda_graph": True,
            "parallax_v2_shadow": False,
            "parallax_v2_render": True,
            "conf_sha256": "config",
            "metric_sha256": "numeric-metrics",
            "label_contract_sha256": "labels-only",
            "metric_runtime": {"python": "test"},
        }
        context = run_eval.baseline_required_context(candidate)
        self.assertEqual(context["mode"], "canonical-v2")
        self.assertNotIn("profile", context)
        self.assertEqual(context["metric_sha256"], "numeric-metrics")
        self.assertEqual(
            context["parallax_v2_source_closure_sha256"], "producer-sha")
        self.assertEqual(
            context["parallax_v2_renderer_source_closure_sha256"], "renderer-sha")
        self.assertTrue(context["cuda_graph"])
        self.assertFalse(context["parallax_v2_shadow"])
        self.assertTrue(context["parallax_v2_render"])
        self.assertNotIn("label_contract_sha256", context)

    def test_cuda_graph_mode_is_part_of_fail_closed_baseline_context(self):
        with tempfile.TemporaryDirectory() as baseline_dir:
            context = {
                "eval_schema": run_eval.EVAL_SCHEMA,
                "metric_sha256": "metric",
                "run_kind": "baseline-update",
                "cuda_graph": False,
            }
            payload = {
                "meta": {
                    **context,
                    "cuda_graph": True,
                    "clip_sha1": "cliphash",
                    "extra_args": [],
                    "git_dirty": False,
                },
                "aggregate": {"quality": 1.0},
                "perf_ms": {"warp": 1.0},
            }
            path = os.path.join(baseline_dir, "clip.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)

            with self.assertRaisesRegex(ValueError, "stale/incompatible"):
                run_eval.preflight_baselines(
                    baseline_dir, ["clip"], context, {"clip": "cliphash"})

            payload["meta"]["cuda_graph"] = False
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            loaded = run_eval.preflight_baselines(
                baseline_dir, ["clip"], context, {"clip": "cliphash"})
            self.assertFalse(loaded["clip"]["meta"]["cuda_graph"])

    def test_cuda_graph_actual_execution_mode_fails_closed(self):
        self.assertTrue(run_eval.validate_cuda_graph_execution_mode(
            True, True, True))
        self.assertFalse(run_eval.validate_cuda_graph_execution_mode(
            False, False, False))

        for requested, captured in ((True, False), (False, True)):
            with self.subTest(requested=requested, captured=captured):
                with self.assertRaisesRegex(ValueError, "refusing to mix"):
                    run_eval.validate_cuda_graph_execution_mode(
                        requested, captured)

        for captured in (None, 1, "true"):
            with self.subTest(invalid_candidate=captured):
                with self.assertRaisesRegex(ValueError, "must be a boolean"):
                    run_eval.validate_cuda_graph_execution_mode(True, captured)

        for baseline in (None, 1, "true"):
            with self.subTest(invalid_baseline=baseline):
                with self.assertRaisesRegex(ValueError, "baseline.*must be boolean"):
                    run_eval.validate_cuda_graph_execution_mode(
                        True, True, baseline)

        with self.assertRaisesRegex(ValueError, "baseline CUDA graph.*differs"):
            run_eval.validate_cuda_graph_execution_mode(True, True, False)

    def test_only_conformance_only_clips_are_exempt_from_committed_baselines(self):
        clips = ["decisive", "reference", "probe"]
        metadata = {
            "decisive": {"content_type": "real-capture"},
            "reference": {
                "content_type": "animation",
                "evaluation_role": "reference-only",
            },
            "probe": {
                "content_type": "synthetic",
                "evaluation_role": "conformance-only",
                "shot_state_contract": {"kind": "hard-cut"},
            },
        }
        self.assertEqual(
            run_eval.clips_requiring_committed_baselines(clips, metadata),
            ["decisive", "reference"])
        metadata["probe"]["evaluation_role"] = "ground-truth"
        self.assertEqual(
            run_eval.clips_requiring_committed_baselines(clips, metadata),
            clips)
        del metadata["reference"]
        with self.assertRaisesRegex(ValueError, "missing source metadata"):
            run_eval.clips_requiring_committed_baselines(clips, metadata)

        metadata["reference"] = {"evaluation_role": "reference-only"}
        metadata["probe"] = {"evaluation_role": "conformance-only"}
        with self.assertRaisesRegex(ValueError, "no authenticated hard contract"):
            run_eval.clips_requiring_committed_baselines(clips, metadata)

    def test_baseline_snapshot_exactly_covers_only_required_clips(self):
        context = {"eval_schema": run_eval.EVAL_SCHEMA, "run_kind": "baseline-update"}
        manifest = {
            "meta": {
                **context,
                "clip_sha1": "decisive-hash",
                "extra_args": [],
                "git_dirty": False,
            },
            "aggregate": {"quality": 1.0},
            "perf_ms": {"warp": 1.0},
        }
        with tempfile.TemporaryDirectory() as baseline_dir:
            path = os.path.join(baseline_dir, "decisive.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            snapshot = run_eval.build_baseline_snapshot(
                baseline_dir, {"decisive": manifest})
            validated = run_eval.validate_baseline_snapshot(
                snapshot, ["decisive"], context,
                {"decisive": "decisive-hash", "probe": "probe-hash"})
            self.assertEqual(set(validated), {"decisive"})
            with self.assertRaisesRegex(
                    ValueError, "baseline-required clip set"):
                run_eval.validate_baseline_snapshot(
                    snapshot, ["decisive", "probe"], context,
                    {"decisive": "decisive-hash", "probe": "probe-hash"})

    def test_hard_constraint_summary_uses_worst_clip_not_mean(self):
        aggregates = {"safe": {"comfort": 1.0}, "unsafe": {"comfort": 5.0}}
        self.assertEqual(
            sbsbench.worst_hard_metric(
                aggregates, "comfort", {"hard_max": 4.0}, ["safe", "unsafe"]),
            (5.0, "unsafe"))
        self.assertEqual(
            sbsbench.worst_hard_metric(
                aggregates, "comfort", {"hard_min": 2.0}, ["safe", "unsafe"]),
            (1.0, "safe"))
        report = os.path.join(os.path.dirname(__file__), "build_report.py")
        with open(report, encoding="utf-8") as fh:
            report_text = fh.read()
        self.assertIn("worst_hard_metric", report_text)
        self.assertIn("safety-worst per-clip aggregate", report_text)

    def test_live_sbs_contract_has_one_canonical_configuration(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "video.h"), encoding="utf-8") as fh:
            video_header = fh.read()
        self.assertIn("SBS_AI = 1", video_header)
        self.assertNotIn("SBS_GAME", video_header)
        self.assertNotIn("SBS_MOVIE", video_header)

        with open(os.path.join(repo, "src", "config.cpp"), encoding="utf-8") as fh:
            config = fh.read()
        self.assertIn(
            'double_between_f(vars, "sbs_3d_pop_strength", video.sbs.pop_strength, '
            '{0.25, 2.0})', config)
        self.assertIn(
            'int_between_f(vars, "sbs_3d_max_encode_width", video.sbs.max_encode_width, '
            '{256, 16384})', config)
        self.assertIn('bool_f(vars, "sbs_3d_cuda_graph", video.sbs.cuda_graph)', config)
        self.assertNotIn("video.sbs_profiles", config)
        self.assertNotIn("apply_sbs_values", config)

        with open(os.path.join(repo, "src", "stream.cpp"), encoding="utf-8") as fh:
            stream = fh.read()
        self.assertNotIn("IDX_SET_SBS_PROFILE", stream)
        self.assertNotIn("IDX_SBS_PROFILE_LIST", stream)
        self.assertIn("mail::sbs_depth_status", stream)
        self.assertNotIn("depth_engine_phase", stream)
        self.assertNotIn("set_active_depth_model(id)", stream)

        with open(os.path.join(repo, "src", "main.cpp"), encoding="utf-8") as fh:
            main = fh.read()
        self.assertIn("prepare_tensorrt_model", main)
        self.assertIn("std::jthread model_prepare_thread", main)
        self.assertLess(main.index("if (!config::sunshine.cmd.name.empty())"),
                        main.index("std::jthread model_prepare_thread"))

        with open(os.path.join(repo, "src", "video_depth_estimator.cpp"),
                  encoding="utf-8") as fh:
            estimator = fh.read()
        self.assertIn("cuda_device_for_configured_adapter", estimator)
        self.assertIn("if (!warmup_execution_context", estimator)
        self.assertIn("const bool synchronized = enqueued", estimator)

    def test_relative_cli_paths_are_resolved_before_subprocess_cwd(self):
        args = argparse.Namespace(build_dir="cmake-build-relwithdebinfo", conf="bench.conf",
                                  clips_root=None, baseline_dir=None,
                                  report_control=None, report_out=None)
        run_eval.normalize_cli_paths(args)
        self.assertTrue(os.path.isabs(args.build_dir))
        self.assertTrue(os.path.isabs(args.conf))

    def test_eval_clip_names_and_labels_are_single_safe_path_components(self):
        self.assertEqual(run_eval.safe_path_component("scene_cut", "clip name"), "scene_cut")
        for unsafe in ("", ".", "..", "../clip", "nested/clip", r"..\clip",
                       r"nested\clip", "/absolute", r"C:\absolute"):
            with self.subTest(unsafe=unsafe), self.assertRaisesRegex(
                    ValueError, "basename|path separators"):
                run_eval.safe_path_component(unsafe, "clip name")

    def test_eval_output_child_is_confined_before_recursive_delete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = os.path.join(temporary, "sbs_eval")
            outside = os.path.join(temporary, "source-frames")
            os.makedirs(root)
            os.makedirs(outside)
            self.assertEqual(
                run_eval.confined_child(root, "safe-label", "run label"),
                os.path.realpath(os.path.join(root, "safe-label")))

            link = os.path.join(root, "reused-label")
            try:
                os.symlink(outside, link, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("directory symlinks are unavailable")
            with self.assertRaisesRegex(ValueError, "escapes evaluator output root"):
                run_eval.confined_child(root, "reused-label", "run label")

    def test_eval_builds_production_binary_and_fails_closed_on_build_error(self):
        current = mock.Mock(returncode=0, stdout="ninja: no work to do.\n", stderr="")
        ninja = os.path.join("C:\\", "msys64", "ucrt64", "bin", "ninja.exe")
        with mock.patch.object(run_eval.shutil, "which", return_value=ninja), \
                mock.patch.object(run_eval.subprocess, "run", return_value=current) as run:
            run_eval.require_current_build("build")
        self.assertEqual(run.call_args.args[0], [ninja, "-C", "build", "sunshine"])
        build_env = run.call_args.kwargs["env"]
        for key, original in run_eval._ORIGINAL_NUMERIC_THREAD_ENV.items():
            if original is None:
                self.assertNotIn(key, build_env)
            else:
                self.assertEqual(build_env[key], original)
        if os.name == "nt":
            self.assertEqual(build_env["MSYSTEM"], "UCRT64")
            self.assertEqual(build_env["MSYS2_PATH_TYPE"], "inherit")
            self.assertIn(os.path.dirname(ninja).lower(), build_env["PATH"].lower())
        failed = mock.Mock(returncode=1, stdout="compile failed\n", stderr="")
        with mock.patch.object(run_eval.shutil, "which", return_value="ninja"), \
                mock.patch.object(run_eval.subprocess, "run", return_value=failed), \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                run_eval.require_current_build("build")

        with open(run_eval.__file__, encoding="utf-8") as stream:
            runner = stream.read()
        self.assertGreaterEqual(runner.count("env=production_subprocess_env()"), 2)

    def test_eval_records_exact_runtime_pipeline_provenance_for_reports(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "tools", "sbsbench", "run_eval.py"),
                  encoding="utf-8") as fh:
            runner = fh.read()
        with open(os.path.join(repo, "tools", "sbsbench", "build_report.py"),
                  encoding="utf-8") as fh:
            report = fh.read()
        self.assertIn('"executable_sha256": file_sha256(exe)', runner)
        self.assertIn('"runtime_shader_sha256": shader_sha', runner)
        self.assertIn('"engine_sha256": file_sha256(engine_path)', runner)
        self.assertIn('"executable_sha256"', report)
        self.assertIn('"runtime_shader_sha256"', report)
        self.assertIn('"engine_sha256"', report)
        self.assertIn('"onnx_sha256"', report)
        self.assertIn('--allow-executable-diff', report)

    def test_production_v2_pop_strength_has_one_configured_authority(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "config.cpp"), encoding="utf-8") as fh:
            config = fh.read()
        self.assertIn(
            '"sbs_3d_pop_strength", video.sbs.pop_strength, {0.25, 2.0}', config)
        with open(os.path.join(repo, "src", "config.h"), encoding="utf-8") as fh:
            config_header = fh.read()
        self.assertIn("double pop_strength = 1.75;", config_header)
        with open(os.path.join(repo, "src_assets", "common", "assets", "web",
                               "config.html"), encoding="utf-8") as fh:
            web_config = fh.read()
        self.assertIn('"sbs_3d_pop_strength": 1.75,', web_config)
        self.assertIn(": 1.75;", web_config)
        with open(os.path.join(repo, "docs", "configuration.md"),
                  encoding="utf-8") as fh:
            configuration_doc = fh.read()
        pop_doc = configuration_doc[configuration_doc.index("### sbs_3d_pop_strength"):]
        self.assertIn("<td><code>1.75</code></td>", pop_doc.split("\n## ", 1)[0])
        with open(os.path.join(repo, "tools", "sbsbench", "run_eval.py"),
                  encoding="utf-8") as fh:
            run_eval_source = fh.read()
        self.assertIn(
            'args.conf, "pop_strength", 1.75, args.extra, "--pop-strength"',
            run_eval_source,
        )
        self.assertIn(
            '"renderer_source_closure_sha256": LIVE_RENDERER_SOURCE_CLOSURE_SHA256',
            run_eval_source,
        )

        with open(os.path.join(repo, "src", "depth_coordinate_v2.h"),
                  encoding="utf-8") as fh:
            coordinate = fh.read()
        self.assertIn("requested_pop_strength(const float configured_pop)", coordinate)
        self.assertIn(
            "return gain_per_pop * requested_pop_strength(configured_pop);", coordinate)
        self.assertNotIn("adaptive_pop_max", coordinate)
        self.assertNotIn("zero_plane", coordinate)

        with open(os.path.join(repo, "src", "video_depth_estimator.cpp"),
                  encoding="utf-8") as fh:
            estimator = fh.read()
        self.assertIn("static_cast<float>(cfg.pop_strength)", estimator)
        self.assertIn(".requested_gain = parallax_v2_requested_gain", estimator)
        self.assertIn(
            "r.parallax_v2_requested_pop_strength = parallax_v2_requested_pop_strength;",
            estimator,
        )
        self.assertIn("r.parallax_v2_requested_gain = parallax_v2_requested_gain;", estimator)

        live_shader_path = os.path.join(
            repo, "src_assets", "windows", "assets", "shaders", "directx",
            "sbs_reprojection_v2_live_ps.hlsl")
        with open(live_shader_path, encoding="utf-8") as fh:
            live_shader = fh.read()
        self.assertIn("Texture2D<float> FinalParallax : register(t1);", live_shader)
        self.assertNotIn("pop_strength", live_shader)
        self.assertNotIn("adaptive_ratio", live_shader)

    def test_depth_runtime_seeds_history_and_sanitizes_nonfinite_model_output(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "video_depth_estimator.cpp"),
                  encoding="utf-8") as fh:
            estimator = fh.read()
        self.assertNotIn("cbuffer_first_frame", estimator)
        self.assertNotIn("depth_history_valid", estimator)
        self.assertIn("depth_valid_history_cs", estimator)
        self.assertIn("tensor_previous_input_uav", estimator)

        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders",
                                  "directx")
        for name in ("depth_minmax_cs.hlsl", "depth_hist_cs.hlsl",
                     "depth_ema_motion_cs.hlsl", "buffer_to_tex_cs.hlsl"):
            shader_path = os.path.join(shader_dir, name)
            with self.subTest(shader=name), open(shader_path, encoding="utf-8") as fh:
                self.assertIn("isinf", fh.read())

        with open(os.path.join(shader_dir, "depth_minmax_cs.hlsl"), encoding="utf-8") as fh:
            reduction = fh.read()
        self.assertIn("g_valid", reduction)
        self.assertIn("MinMaxOut.InterlockedAdd(8, g_valid[0])", reduction)
        with open(os.path.join(shader_dir, "depth_minmax_ema_cs.hlsl"),
                  encoding="utf-8") as fh:
            bounds = fh.read()
        self.assertIn("valid_count > 0u", bounds)
        self.assertIn("MinMaxRaw.Store(8, 0u)", bounds)
        self.assertIn("s.w = 0.0f", bounds)
        self.assertIn("s.w = 2.0f", bounds)
        with open(os.path.join(shader_dir, "buffer_to_tex_cs.hlsl"),
                  encoding="utf-8") as fh:
            mapper = fh.read()
        self.assertIn("OutputTexture[DTid.xy] = PreviousDepth[DTid.xy]", mapper)
        self.assertIn("scale.w < 0.5f", mapper)
        self.assertIn("scale.w > 1.5f ? 1.0f : ema_alpha", mapper)
        with open(os.path.join(shader_dir, "depth_ema_motion_cs.hlsl"),
                  encoding="utf-8") as fh:
            motion = fh.read()
        self.assertIn("return PreviousDepth[p]", motion)
        with open(os.path.join(shader_dir, "depth_valid_history_cs.hlsl"),
                  encoding="utf-8") as fh:
            history = fh.read()
        self.assertIn("MinMaxEma[0].w < 0.5f", history)
        self.assertIn("PreviousModelInput[idx + 2u * plane]", history)

    def test_failed_tensorrt_warmups_are_quarantined_and_not_advertised(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "video_depth_estimator.cpp"),
                  encoding="utf-8") as fh:
            estimator = fh.read()
        self.assertIn(
            "quarantine_execution_context_locked(engine_key, exec_context,",
            estimator,
        )
        self.assertIn("--slot.warmed_context_count", estimator)
        self.assertIn("quarantined_context_count", estimator)
        self.assertIn("warmed_context_count", estimator)
        self.assertIn("if (context_warmed && !execution_context_poisoned)", estimator)
        self.assertIn('model.name + ".active-engine.json"', estimator)
        self.assertIn('{"onnx_sha256", artifact.source_sha256}', estimator)

    def test_harness_contract_is_v2_only_and_machine_verified(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                               "sbs_reprojection_v2_live_ps.hlsl"), encoding="utf-8") as fh:
            shader = fh.read()
        self.assertNotIn("literal_mode", shader)
        self.assertIn("StructuredBuffer<float4> ParallaxState", shader)

        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"), encoding="utf-8") as fh:
            harness = fh.read()
        self.assertNotIn('a == "--literal-bestv2"', harness)
        self.assertIn('fs::path(o.out) / "contract.json"', harness)
        self.assertIn(
            '(direct_parallax_mode ? direct_geometry_contract_schema : 20u)',
            harness)
        self.assertIn('direct_geometry_contract_schema = 25u', harness)
        self.assertIn('direct_geometry_manifest_schema = 6u', harness)
        self.assertIn(
            '"external-final-parallax-with-diagnostic-order-v6"',
            harness)
        self.assertNotIn('\\"depth_override_frames\\"', harness)
        self.assertIn('\\"pop_strength\\"', harness)
        self.assertNotIn('\\"zero_plane\\"', harness)
        self.assertIn('"mapping_ps"', harness)
        self.assertIn('"warp_map_%s.f32"', harness)
        self.assertIn('fs::path(o.out) / "warp_map_shape.json"', harness)
        self.assertIn('raw_reproject_source_u_normalized', harness)
        self.assertIn('live_sample_transform', harness)
        self.assertIn('\\"warp_mapping\\"', harness)
        self.assertIn('fs::path(o.out) / "cut_state.json"', harness)

        with open(os.path.join(repo, "tools", "sbsbench", "run_eval.py"),
                  encoding="utf-8") as fh:
            evaluator = fh.read()
        self.assertIn('contract_path = os.path.join(out_dir, "contract.json")', evaluator)
        self.assertIn('mapping_path = mapping_by_id[frame_id]', evaluator)
        self.assertIn('"raw_reproject_source_u_normalized"', evaluator)
        self.assertNotIn("profile ([a-z0-9_-]+)", evaluator)

        with open(os.path.join(repo, "src", "stream.cpp"), encoding="utf-8") as fh:
            stream = fh.read()
        self.assertNotIn("SBS_PRESENTATION_FIXED_HEIGHT", stream)

    def test_native_offline_adaptive_trace_transport_is_bounded(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"),
                  encoding="utf-8") as fh:
            harness = fh.read()
        self.assertIn('a == "--bounded-adaptive-state"', harness)
        self.assertIn('\\"transport\\":\\"atomic-latest-v1\\"', harness)
        self.assertIn('\\"retained_history\\":false', harness)
        whole_contract = harness[harness.index(
            'std::ofstream contract(fs::path(o.out) / "whole_clip_contract.json")'):]
        self.assertNotIn('<< "  \\"cut_state\\":', whole_contract)

        with open(os.path.join(repo, "src", "offline_sbs_worker.cpp"),
                  encoding="utf-8") as fh:
            worker = fh.read()
        self.assertEqual(worker.count('"--bounded-adaptive-state"'), 1)
        self.assertIn("trace_tail_t trace(analysis_output, true)", worker)
        self.assertIn(
            'read_snapshot(child, "adaptive_state_frame.json")',
            worker)
        self.assertIn(
            'remove_file_checked(snapshot)',
            worker)

    def test_depth_reuse_cadence_is_fixed_at_one_inference_per_frame(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"), encoding="utf-8") as fh:
            harness = fh.read()
        self.assertNotIn('a == "--depth-every"', harness)
        self.assertNotIn('a == "--depth-override-root"', harness)
        self.assertIn("int effective_depth_every = 1;", harness)
        self.assertIn("depth_reuse_interval", harness)
        with open(os.path.join(repo, "tools", "sbsbench", "run_eval.py"),
                  encoding="utf-8") as fh:
            evaluator = fh.read()
        self.assertIn('"depth_compensation": depth_compensation', evaluator)
        self.assertIn('depth_reuse_interval = 1', evaluator)
        self.assertIn('depth_step = "current-once"', evaluator)
        self.assertGreaterEqual(
            evaluator.count("whole_clip_raw_contract.HARNESS_CONTRACT_SCHEMA"), 2)
        self.assertIn('"cut_state.json"', evaluator)
        self.assertIn('"warp_map_*.f32"', evaluator)
        self.assertIn('expected_mapping_bytes = width * height * 4', evaluator)

    def test_direct_parallax_is_harness_only_and_machine_verified(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"),
                  encoding="utf-8") as fh:
            harness = fh.read()
        self.assertIn('a == "--direct-parallax-root"', harness)
        self.assertNotIn('a == "--direct-candidate-root"', harness)
        self.assertNotIn('a == "--direct-candidate-fill"', harness)
        self.assertIn('o.output_every != 1', harness)
        self.assertNotIn('o.literal_bestv2', harness)
        self.assertNotIn('depth_override_root', harness)
        self.assertNotIn('direct_parallax_max_abs * 1.10f', harness)
        self.assertIn('words remain reserved to preserve the 16-byte constant-buffer ABI', harness)
        self.assertIn(
            'direct_parallax_source_u_limit =\n'
            '      models::depth_coordinate_v2::direct_container_limit', harness)
        self.assertIn(
            'direct_parallax_max_horizontal_slope =\n'
            '      models::depth_coordinate_v2::max_horizontal_slope', harness)
        self.assertIn('violates the generated horizontal slope contract', harness)
        self.assertIn('"order_" + output_id + ".f32"', harness)
        self.assertIn('"direct_parallax_manifest.json"', harness)
        self.assertNotIn('"direct_candidate_manifest.json"', harness)
        self.assertNotIn('SBS_DIRECT_CANDIDATE_PARALLAX', harness)
        self.assertNotIn('SBS_CANDIDATE_GAP_FILL', harness)
        self.assertIn(
            '(direct_parallax_mode ? direct_geometry_contract_schema : 20u)',
            harness)
        self.assertIn('{"renderer_uses_order", false}', harness)
        self.assertIn(
            '{"order_role", "diagnostic-semantic-depth-only-v1"}',
            harness)
        self.assertIn('("depth_" + output_id + ".f32")', harness)
        self.assertNotIn('"displacement_high_is_near"', harness)

        shader_root = os.path.join(
            repo, "src_assets", "windows", "assets", "shaders", "directx")
        with open(os.path.join(shader_root, "sbs_direct_replay_ps.hlsl"),
                  encoding="utf-8") as fh:
            reprojection = fh.read()
        self.assertNotIn("CanonicalOrderTexture", reprojection)
        self.assertNotIn("SBS_DIRECT_CANDIDATE_PARALLAX", reprojection)
        self.assertNotIn("SBS_CANDIDATE_GAP_FILL", reprojection)
        direct_inverse = reprojection[reprojection.index("float2 Reproject"):
                                      reprojection.index("float4 main_ps")]
        self.assertIn("for (int iteration = 0; iteration < 11; ++iteration)",
                      direct_inverse)
        self.assertIn(
            "source_x = destination_uv.x + eye_sign *",
            direct_inverse)
        self.assertNotIn("ForwardCoverageTexture", direct_inverse)
        self.assertNotIn("bgOrder", direct_inverse)
        self.assertNotIn("SBS_DIRECT_PARALLAX", reprojection)
        self.assertIn("/sbs_direct_replay_ps.hlsl", harness)

        evaluator_path = os.path.join(repo, "tools", "sbsbench", "run_eval.py")
        with open(evaluator_path, encoding="utf-8") as fh:
            evaluator = fh.read()
        self.assertIn(
            'direct_parallax_root and not args.comparison_only', evaluator)
        self.assertIn(
            'validate_direct_parallax_manifest(', evaluator)

    def test_live_depth_pairing_is_bounded_with_retained_source_completion_owner(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "config.cpp"), encoding="utf-8") as fh:
            config = fh.read()
        self.assertNotIn('"depth_frame_mode"', config)
        self.assertNotIn('"depth_fps"', config)

        with open(os.path.join(repo, "src", "platform", "windows", "display_vram.cpp"),
                  encoding="utf-8") as fh:
            production = fh.read()
        self.assertIn("std::array<matched_frame_slot_t, 2>", production)
        self.assertIn("repeat_matched_output", production)
        self.assertNotIn("finish_pending_depth_for_evaluation", production)
        self.assertIn("finish_pending_depth_for_idle_recovery", production)
        self.assertIn("bootstrap_current_output", production)
        self.assertIn("stale_prior_completion", production)
        self.assertIn("retained_source_pending_slot", production)
        self.assertIn("depth_completion_poll_pending", production)
        self.assertIn("needs_conversion_poll() const override", production)
        self.assertNotIn("depth_frame_mode", production)

        cleanup = production[
            production.index("const auto release_unknown_completion"):
            production.index("if (retained_source_pending_slot)")
        ]
        self.assertIn("if (&slot != preserve_slot)", cleanup)
        self.assertIn("slot.pending = false;", cleanup)

        retained = production[
            production.index("if (retained_source_pending_slot)"):
            production.index("matched_candidate_slot = available_matched_slot();")
        ]
        self.assertLess(
            retained.index("depth_completion_poll_pending = false;"),
            retained.index("find_pending_matched_slot(recovered.completed_frame_id)"))
        self.assertIn(
            "release_unknown_completion(recovered.completed_frame_id, nullptr);",
            retained)
        compact_production = "".join(production.split())
        self.assertIn(
            "release_unknown_completion(est.completed_frame_id,matched_candidate_slot);",
            compact_production)

        bootstrap = production[
            production.index("const bool bootstrap_current_output"):
            production.index("// A poisoned CUDA/TensorRT producer")
        ]
        self.assertLess(
            bootstrap.index("depth_completion_poll_pending = false;"),
            bootstrap.index("find_pending_matched_slot(recovered.completed_frame_id)"))
        self.assertIn(
            "release_unknown_completion(recovered.completed_frame_id, nullptr);",
            bootstrap)

        with open(os.path.join(repo, "src", "video.cpp"), encoding="utf-8") as fh:
            encoder = fh.read()
        self.assertIn("session->needs_conversion_poll()", encoder)
        self.assertIn("session->convert(*last_img)", encoder)
        self.assertIn("img->frame_timestamp = frame_timestamp", encoder)

        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"),
                  encoding="utf-8") as fh:
            harness = fh.read()
        self.assertIn("finish_pending_depth_for_evaluation", harness)

    def test_live_debug_dump_snapshots_exact_model_io_before_cuda_reuse(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "video_depth_estimator.cpp"),
                  encoding="utf-8") as fh:
            estimator = fh.read()
        live_estimate = estimator[
            estimator.index("estimate_result estimate("):
            estimator.index("video_depth_estimator::video_depth_estimator")
        ]
        normalized = live_estimate.index("normalize_depth_output(d3d_timer);")
        production_post_timing_ended = live_estimate.index(
            "mark_d3d_post_end(d3d_timer);", normalized)
        raw_snapshotted = live_estimate.index(
            "tensor_out_buf.Get(),", normalized)
        input_snapshotted = live_estimate.index(
            "tensor_in_buf.Get(),", raw_snapshotted)
        current_preprocess = live_estimate.index(
            "context->CSSetShader(rgb_to_nchw_cs.Get()", input_snapshotted)
        reused_by_cuda = live_estimate.index(
            "cuda.cuGraphicsMapResources(2, resources, cu_stream)", current_preprocess)
        self.assertLess(normalized, raw_snapshotted)
        self.assertLess(production_post_timing_ended, raw_snapshotted)
        self.assertLess(raw_snapshotted, input_snapshotted)
        self.assertLess(input_snapshotted, current_preprocess)
        self.assertLess(current_preprocess, reused_by_cuda)
        self.assertIn(
            "context->CopyResource(snapshot.Get(), source);", estimator)

        with open(os.path.join(repo, "src", "platform", "windows", "display_vram.cpp"),
                  encoding="utf-8") as fh:
            production = fh.read()
        requested = production.index(
            "const bool snapshot_debug_inputs = sbs_dumper.snapshot_requested();")
        estimate_call = production.index("depth_estimator->estimate_depth(", requested)
        self.assertLess(requested, estimate_call)
        self.assertIn(
            "perf && !snapshot_debug_inputs ? begin_sbs_gpu_timer() : nullptr",
            production)
        self.assertIn("if (perf && !snapshot_debug_inputs)", production)
        self.assertIn("est.raw_model_depth_snapshot.Get()", production[estimate_call:])
        self.assertIn("est.model_input_snapshot.Get()", production[estimate_call:])
        # The authenticated final-parallax field is retained for live rendering and dump replay.
        self.assertIn("dump_warp_depth = warp_depth;", production)
        preflight = production.index("sbs_dumper.preflight_requested_v2_frame(")
        self.assertIn("render_sbs_debug_geometry(", production)
        self.assertLess(
            preflight,
            production.index("render_sbs_debug_geometry(", preflight))
        self.assertLess(
            production.index("end_sbs_gpu_timer(gpu_timer);"),
            production.index("render_sbs_debug_geometry("))
        self.assertLess(
            production.index('sbs_perf::add_sample_ms("sbs_convert_cpu"', requested),
            production.index("render_sbs_debug_geometry(", requested))
        # Color interpretation belongs to the buffered source/depth pair, not whichever capture
        # frame happens to be current when the asynchronous inference completes.
        self.assertIn(
            "models::input_color_space color_space = models::input_color_space::srgb;",
            production)
        self.assertIn("slot.color_space = color_space;", production)
        self.assertIn("matched_render_slot->color_space", production)

        with open(os.path.join(repo, "src", "platform", "windows",
                               "sbs_debug_dump.cpp"), encoding="utf-8") as fh:
            dumper = fh.read()
        for artifact in (
                "model_input.f32", "model_input.png", "model_input_shape.json",
                "raw_depth.f32", "raw_depth.png", "raw_depth_heat.png",
                "raw_shape.json", "depth.f32", "warp_depth.f32",
                "adaptive_state.json", "warp_map.f32",
                "warp_displacement_heat.png", "warp_mask.png",
                "dump_manifest.json"):
            self.assertIn(artifact, dumper)
        self.assertIn("matched_frame_id", dumper)
        self.assertIn("catch (const std::exception &exception)", dumper)
        self.assertIn(
            "color_space == models::input_color_space::srgb", dumper)
        self.assertIn("r = encode_unit(rf);", dumper)

    def test_live_debug_dump_is_session_scoped_and_host_sbs_only(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "stream.cpp"), encoding="utf-8") as fh:
            stream = fh.read()
        with open(os.path.join(repo, "src", "video.cpp"), encoding="utf-8") as fh:
            video = fh.read()
        with open(os.path.join(repo, "src", "platform", "windows",
                               "sbs_debug_dump.cpp"), encoding="utf-8") as fh:
            dumper = fh.read()
        self.assertIn("sbs_debug_dump_request_allowed(", stream)
        self.assertIn("requested_sbs_mode", stream)
        self.assertIn("session->video->sbs_debug_dump_pending->store(true", stream)
        self.assertNotIn("std::atomic<bool> sbs_debug_dump_pending", video)
        self.assertNotIn("extern std::atomic<bool> sbs_debug_dump_pending", dumper)
        self.assertIn("button_request_", dumper)

    def test_cuda_graph_replay_is_signature_safe_and_falls_back(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "config.h"), encoding="utf-8") as fh:
            config = fh.read()
        self.assertIn("bool cuda_graph = true;", config)
        with open(os.path.join(repo, "src", "video_depth_estimator.cpp"),
                  encoding="utf-8") as fh:
            estimator = fh.read()
        self.assertIn("input != graph_input || output != graph_output", estimator)
        self.assertIn("target_w != graph_width || target_h != graph_height", estimator)
        self.assertIn("if (!graph_signature_warmed)", estimator)
        self.assertIn("destroy_inference_graph(cuda);", estimator)
        self.assertIn("return exec_context->enqueueV3(cu_stream);", estimator)
        with open(os.path.join(repo, "src", "cuda_driver_api.h"), encoding="utf-8") as fh:
            driver = fh.read()
        for symbol in ("cuStreamBeginCapture", "cuStreamEndCapture",
                       "cuGraphInstantiateWithFlags", "cuGraphLaunch",
                       "cuGraphExecDestroy"):
            self.assertIn(symbol, driver)

    def test_shared_eval_controls_read_explicit_top_level_values(self):
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            fh.write("sbs_3d_cuda_graph = false\n"
                     "sbs_3d_pop_strength = 1.45\n")
            path = fh.name
        try:
            self.assertFalse(run_eval.expected_shared_bool(
                path, "cuda_graph", True, [], "--cuda-graph"))
            self.assertEqual(run_eval.expected_shared_number(
                path, "pop_strength", 1.2, [], "--pop-strength"), 1.45)
        finally:
            os.unlink(path)

    def test_edge_selective_ema_uses_immutable_history_and_exports_locality_mask(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders",
                                  "directx")
        with open(os.path.join(shader_dir, "depth_ema_motion_cs.hlsl"),
                  encoding="utf-8") as fh:
            mask_shader = fh.read()
        self.assertIn("PreviousDepth", mask_shader)
        self.assertIn("ema_edge_change", mask_shader)
        self.assertNotIn("ema_edge_dilation", mask_shader)
        self.assertIn("MotionMask[DTid.xy] = IsMovingEdge", mask_shader)
        with open(os.path.join(repo, "src", "video_depth_estimator.cpp"),
                  encoding="utf-8") as fh:
            estimator = fh.read()
        self.assertIn("CopyResource(depth_previous_tex.Get(), depth_tex.Get())", estimator)
        self.assertIn("ema_motion_mask_srv", estimator)
        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"),
                  encoding="utf-8") as fh:
            harness = fh.read()
        self.assertIn('"ema_mask_%s.png"', harness)

    def test_rescore_derives_depth_compensation_for_schema_upgrade(self):
        self.assertEqual(rescore_run.depth_compensation_from_meta({}), "none")
        self.assertEqual(rescore_run.depth_compensation_from_meta(
            {"extra_args": ["--depth-override-root", "reference"]}),
            "external-reference")
        self.assertEqual(rescore_run.depth_compensation_from_meta(
            {"depth_compensation": "nvof-1x1"}), "nvof-1x1")
        self.assertEqual(rescore_run.depth_compensation_from_meta(
            {"extra_args": ["--depth-override-root", "reference",
                            "--depth-override-all"]}),
            "external-treatment")

    def test_rescore_requires_current_comparison_only_provenance(self):
        valid = {"meta": {"run_kind": "comparison-only",
                          "eval_schema": run_eval.EVAL_SCHEMA,
                          "preprocess_profile": None,
                          "preprocess_source_closure_sha256": "0" * 64,
                          "depth_coordinate_v2_calibration_id": None}}
        rescore_run.validate_rescore_provenance(valid)
        with self.assertRaisesRegex(SystemExit, "comparison-only provenance"):
            rescore_run.validate_rescore_provenance(
                {"meta": {"run_kind": "baseline-gated",
                          "eval_schema": run_eval.EVAL_SCHEMA}})
        with self.assertRaisesRegex(SystemExit, "evaluator schema"):
            rescore_run.validate_rescore_provenance(
                {"meta": {"run_kind": "comparison-only",
                          "eval_schema": run_eval.EVAL_SCHEMA - 1}})

    def test_warp_and_coverage_apply_per_eye_aspect_mapping(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx")
        for name in ("sbs_reprojection_v2_live_ps.hlsl", "sbs_direct_replay_ps.hlsl"):
            with self.subTest(shader=name), open(os.path.join(shader_dir, name), encoding="utf-8") as fh:
                self.assertIn("ContentToSourceUV", fh.read())

    def test_host_sbs_intermediate_follows_observed_capture_transfer(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        display = os.path.join(repo, "src", "platform", "windows", "display_vram.cpp")
        with open(display, encoding="utf-8") as fh:
            pipeline = fh.read()
        self.assertIn("const DXGI_FORMAT required_format = input_is_linear ?", pipeline)
        self.assertIn("DXGI_FORMAT_R16G16B16A16_FLOAT", pipeline)
        self.assertIn("DXGI_FORMAT_B8G8R8A8_UNORM", pipeline)
        self.assertIn("display->capture_format != DXGI_FORMAT_UNKNOWN", pipeline)
        self.assertIn("if (!input_is_linear &&", pipeline)
        self.assertNotIn("sbs_sharpen", pipeline)
        self.assertIn("input_is_linear ? convert_Y_or_YUV_fp16_ps.get()", pipeline)
        self.assertIn("models::input_color_space::linear_sdr", pipeline)
        common = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "include", "common.hlsl")
        with open(common, encoding="utf-8") as fh:
            color = fh.read()
        self.assertIn("rgb = Rec709toRec2020(rgb)", color)
        self.assertIn("rgb *= 80", color)
        self.assertIn("return NitsToPQ(rgb)", color)

    def test_local_ar_fp16_sdr_presentation_applies_srgb_transfer(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        display_path = os.path.join(repo, "src", "platform", "windows", "display_vram.cpp")
        shader_path = os.path.join(
            repo, "src_assets", "windows", "assets", "shaders", "directx",
            "rgb_present_linear_to_srgb_ps.hlsl")
        with open(display_path, encoding="utf-8") as fh:
            display = fh.read()
        with open(shader_path, encoding="utf-8") as fh:
            shader = fh.read()
        self.assertIn("bool rgb_present_target_is_linear = false;", display)
        self.assertIn("if (input_is_linear != rgb_present_target_is_linear)", display)
        self.assertIn("rgb_present_srgb_to_linear_ps.get()", display)
        self.assertIn("rgb_present_linear_to_srgb_ps.get()", display)
        self.assertNotIn("frame_is_hdr = d3d_image.format", display)
        self.assertIn("ApplySRGBCurve(saturate(source.rgb))", shader)

    def test_hdr_debug_preview_preserves_hue(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(repo, "src", "platform", "windows", "sbs_debug_dump.cpp")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("0.2126f * r + 0.7152f * g", text)
        self.assertIn("const float tone_scale = 1.0f / (1.0f + luminance)", text)
        self.assertNotIn("c = c / (1.0f + c)", text)

    def test_report_preserves_missing_unsupported_and_support_units(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(repo, "tools", "sbsbench", "build_report.py")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        tree = ast.parse(text, filename=path)
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

        state_ns = {
            "THR": {"image_integrity_pct": {
                "requires": "image_integrity_support", "role": "hard"}},
            "sbsbench": sbsbench,
        }
        exec(compile(ast.Module(
            body=[functions["_metric_state_value"]], type_ignores=[]), path, "exec"),
             state_ns)
        metric_state = state_ns["_metric_state_value"]

        def payload(aggregate):
            return {"clips": {"clip": {"aggregate": aggregate, "meta": {}}}}

        self.assertEqual(metric_state(payload({}), "clip", "image_integrity_pct"),
                         ("missing", None))
        self.assertEqual(metric_state(
            payload({"image_integrity_support": 0.0}), "clip", "image_integrity_pct"),
            ("unsupported", None))
        self.assertEqual(metric_state(payload({
            "image_integrity_support": 21.4, "image_integrity_pct": 99.0,
        }), "clip", "image_integrity_pct"), ("applicable", 99.0))

        support_ns = {
            "HARD_SUPPORT_KEYS": {"image_integrity_pct": "image_integrity_support"},
            "sbsbench": sbsbench,
        }
        exec(compile(ast.Module(
            body=[functions["_hard_support"]], type_ignores=[]), path, "exec"),
             support_ns)
        self.assertEqual(support_ns["_hard_support"](
            payload({"image_integrity_support": 21.4}), "clip", "image_integrity_pct"),
            "21.4%")

        self.assertIn('class="heat-clip"', text)
        self.assertNotIn(".heatmap th:first-child", text)
        self.assertIn('class="heat-missing"', text)

    def test_report_runtime_and_artifact_provenance_fail_closed(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(repo, "tools", "sbsbench", "build_report.py")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        tree = ast.parse(text, filename=path)
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

        runtime_ns = {}
        exec(compile(ast.Module(
            body=[functions["_validate_metric_runtime"]], type_ignores=[]), path, "exec"),
             runtime_ns)
        validate_runtime = runtime_ns["_validate_metric_runtime"]
        expected_runtime = {"python": "3", "numpy": "2", "pillow": "11"}
        validate_runtime({"meta": {"metric_runtime": expected_runtime}},
                         "control", expected_runtime)
        with self.assertRaisesRegex(SystemExit, "different numeric runtime"):
            validate_runtime({"meta": {"metric_runtime": {"python": "old"}}},
                             "control", expected_runtime)

        artifact_runtime = mock.Mock()
        artifact_runtime.new_remeasurement_session.return_value = object()
        artifact_runtime.verify_results_against_artifacts.return_value = {
            "passed": True, "clips": ["clip"], "frame_count": 1}
        artifact_ns = {"run_eval": artifact_runtime, "THRESHOLD_CFG": {"metrics": {}}}
        exec(compile(ast.Module(
            body=[functions["_validate_authoritative_results"]], type_ignores=[]), path, "exec"),
             artifact_ns)
        validate_results = artifact_ns["_validate_authoritative_results"]
        run = {"meta": {}, "clips": {"clip": {}}}
        validate_results(run, "run", "control", "clips")
        artifact_runtime.verify_results_against_artifacts.assert_called_once_with(
            run, "run", "clips", {"metrics": {}}, remeasurement_session=None)
        artifact_runtime.verify_results_against_artifacts.side_effect = ValueError(
            "clips.clip.aggregate differs")
        with self.assertRaisesRegex(SystemExit, "authoritative remeasurement"):
            validate_results(run, "run", "control", "clips")

    def test_report_dashboard_covers_decision_hard_and_supporting_metrics(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        report_path = os.path.join(repo, "tools", "sbsbench", "build_report.py")
        with open(report_path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=report_path)

        def assignment_keys(name):
            assignment = next(
                node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name
                        for target in node.targets))
            return [entry.elts[0].value for entry in assignment.value.elts]

        primary_style = assignment_keys("PRIMARY_STYLE_AXES")
        hard = assignment_keys("HARD_DISPLAY")
        supporting = assignment_keys("SUPPORTING_HEATMAP_AXES")

        with open(os.path.join(repo, "tools", "sbsbench", "thresholds.json"),
                  encoding="utf-8") as fh:
            specs = json.load(fh)["metrics"]
        configured_primary = {key for key, spec in specs.items()
                              if spec.get("role") == "primary"}
        configured_hard = {key for key, spec in specs.items()
                           if spec.get("role") == "hard"}
        configured_diagnostic = {key for key, spec in specs.items()
                                 if spec.get("role") == "diagnostic"}
        self.assertEqual(set(primary_style),
                         configured_primary | {"exact_visible_pop_spread_pct"})
        self.assertEqual(set(hard), configured_hard)
        self.assertEqual(len(hard), 15)
        self.assertEqual(set(supporting),
                         configured_diagnostic - {"exact_visible_pop_spread_pct"})
        self.assertIn("exact_local_polarity_component_pct", supporting)
        self.assertIn("flow_temporal_p95", supporting)

    def test_live_trt_contexts_are_bounded_and_engine_io_fails_closed(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(repo, "src", "video_depth_estimator.cpp")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("kMaxContextsPerEngine = 4", text)
        self.assertIn("allocated_context_count(slot) >= kMaxContextsPerEngine", text)
        self.assertIn("g_trt_context_available.wait_for", text)
        self.assertIn("slot.io_compatible = have_in && have_out && input_fp32 && output_fp32", text)
        self.assertIn("validate_engine_io_locked", text)

    def test_nvhttp_reads_coherent_locked_process_status(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "process.h"), encoding="utf-8") as fh:
            header = fh.read()
        with open(os.path.join(repo, "src", "process.cpp"), encoding="utf-8") as fh:
            implementation = fh.read()
        with open(os.path.join(repo, "src", "nvhttp.cpp"), encoding="utf-8") as fh:
            nvhttp = fh.read()
        public = header[header.index("class proc_t"):header.index("  private:")]
        self.assertNotIn("bool virtual_display", public)
        self.assertNotIn("bool allow_client_commands", public)
        self.assertIn("struct process_status_t", header)
        self.assertIn("process_status_t get_status()", header)
        status_start = implementation.index("process_status_t proc_t::get_status()")
        status_end = implementation.index("#ifdef _WIN32", status_start)
        status_implementation = implementation[status_start:status_end]
        self.assertIn("std::lock_guard lock(process_state_mutex);", status_implementation)
        self.assertIn("return {\n      _app_id,\n      _app_name,", status_implementation)
        self.assertIn("_virtual_display,", status_implementation)
        self.assertNotIn("running_locked()", status_implementation)
        self.assertIn(
            "set_display_name_locked(platf::to_utf8(_virtual_display_gdi_name));",
            implementation,
        )
        self.assertNotIn("proc::proc.virtual_display", nvhttp)
        self.assertNotIn("proc::proc.allow_client_commands", nvhttp)
        self.assertIn("proc::proc.get_status()", nvhttp)

    def test_remote_virtual_display_lease_is_renewed_after_slow_launch(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "process.cpp"), encoding="utf-8") as fh:
            process = fh.read()
        with open(os.path.join(repo, "src", "platform", "windows", "ar_glasses.cpp"),
                  encoding="utf-8") as fh:
            ownership = fh.read()
        launch_tail = process[process.index("start_hdr_worker(launch_session->enable_hdr);"):
                              process.index("fg.disable();")]
        self.assertIn(
            "remote_virtual_display_awaiting_client(\n"
            "        *_remote_virtual_display_lease,\n"
            "        config::stream.ping_timeout",
            launch_tail,
        )
        self.assertIn("std::chrono::steady_clock::now() + "
                      "detail::remote_pending_duration(connect_timeout)", ownership)

    def test_live_gpu_timer_tail_is_bounded_and_generation_safe(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        display_path = os.path.join(repo, "src", "platform", "windows", "display_vram.cpp")
        with open(display_path, encoding="utf-8") as fh:
            display = fh.read()
        with open(os.path.join(repo, "src", "sbs_perf.cpp"), encoding="utf-8") as fh:
            perf = fh.read()
        self.assertIn("drain_sbs_gpu_timers();", display)
        self.assertIn("std::chrono::milliseconds(100)", display)
        self.assertIn("sbs_perf::add_sample_ms_if_current", display)
        self.assertIn("g_generation.fetch_add", perf)

    def test_depth_transform_audit_preserves_16bit_precision(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "depth.png")
            values = np.linspace(0, 65535, 100, dtype=np.uint16).reshape(10, 10)
            Image.fromarray(values).save(path)
            stats = audit_depth_transform.frame_stats(path)
        self.assertAlmostEqual(stats["spread_p95_p05"], 0.9, delta=0.02)
        self.assertGreater(stats["saturated_low_pct"], 0.0)
        self.assertGreater(stats["saturated_high_pct"], 0.0)

    def test_depth_transform_audit_uses_image_mode_for_dark_16bit_png(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            path = fh.name
        try:
            values = np.linspace(0, 255, 100, dtype=np.uint16).reshape(10, 10)
            Image.fromarray(values).save(path)
            stats = audit_depth_transform.frame_stats(path)
            self.assertLess(stats["p99"], 0.005)
            self.assertLess(stats["spread_p95_p05"], 0.005)
        finally:
            os.unlink(path)

    def test_expected_flat_exemption_is_derived_from_stereo_axis(self):
        flat = {"expected_flat": True}
        self.assertTrue(run_eval.metric_exempt_for_clip({"axis": "stereo"}, flat))
        self.assertFalse(run_eval.metric_exempt_for_clip({"axis": "comfort"}, flat))
        self.assertFalse(run_eval.metric_exempt_for_clip({"axis": "stereo"}, {}))

    def test_production_v2_pipeline_is_mandatory_and_fails_flat(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        paths = {
            "estimator": os.path.join(repo, "src", "video_depth_estimator.cpp"),
            "display": os.path.join(repo, "src", "platform", "windows", "display_vram.cpp"),
        }
        text = {}
        for key, path in paths.items():
            with open(path, encoding="utf-8") as fh:
                text[key] = fh.read()
        self.assertIn("const bool producer_shaders_ok", text["estimator"])
        self.assertIn("if (!producer_shaders_ok)", text["estimator"])
        self.assertIn(
            "if (!valid || terminal_failure || live_v2_producer_unavailable() || !input_srv)",
            text["estimator"],
        )
        self.assertIn("depth_estimator->is_valid()", text["display"])
        self.assertIn("models::parallax_v2_result_is_authenticated(est)", text["display"])
        self.assertIn(
            "host_sbs_renderer = models::fail_host_sbs_renderer_flat(host_sbs_renderer);",
            text["display"],
        )

    def test_tensorrt_level_is_part_of_engine_recipe(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "model_manager.h"), encoding="utf-8") as fh:
            manager = fh.read()
        with open(os.path.join(repo, "src", "video_depth_estimator.cpp"),
                  encoding="utf-8") as fh:
            estimator = fh.read()
        self.assertIn("depth_engine_builder_level = 5", manager)
        self.assertIn("trt-opt770x434-max1036-level5-v3", manager)
        # The bound belongs in the tag: the cached engine filename encodes only the opt
        # shape and builder level, so a kMAX change would otherwise reuse a stale engine.
        self.assertIn("depth_engine_max_dim = 1036", manager)
        self.assertIn("setBuilderOptimizationLevel(depth_engine_builder_level)", estimator)

    def test_live_and_eval_shaders_use_level3_optimization(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "platform", "windows", "display_vram.cpp"),
                  encoding="utf-8") as fh:
            live = fh.read()
        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"),
                  encoding="utf-8") as fh:
            harness = fh.read()
        self.assertIn("flags |= D3DCOMPILE_OPTIMIZATION_LEVEL3", live)
        self.assertIn("D3DCOMPILE_OPTIMIZATION_LEVEL3", harness)

    def test_production_warp_uses_authenticated_inverse_mapping(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "sbs_reprojection_v2_live_ps.hlsl")
        with open(shader, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("sample_uv = WarpAvailable() ? Reproject(source_uv, eye_sign)", text)

    def test_report_evidence_is_bounded_and_accepts_zero_based_frames(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        report = os.path.join(repo, "tools", "sbsbench", "build_report.py")
        with open(report, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("load_warp_mapping", text)
        self.assertIn("warp_map_shape.json", text)
        self.assertNotIn("match_scale = min(1.0, 256.0 / ew)", text)
        self.assertIn("prev_idx = sbsbench.predecessor_frame_id(source_files(clip), idx)", text)
        self.assertIn("if prev_idx is None:", text)
        self.assertNotIn("if prev_idx < 1:", text)
        self.assertIn("source_eye_width = image.width // 2", text)
        self.assertIn("packed.paste(eyes[1], (eye_w, 0))", text)
        self.assertNotIn("image.resize(size, Image.LANCZOS)", text)
        self.assertIn("run_eval.EVAL_SCHEMA", text)

    def test_report_resolves_artifacts_by_numeric_index_not_fixed_padding(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        report = os.path.join(repo, "tools", "sbsbench", "build_report.py")
        with open(report, encoding="utf-8") as fh:
            text = fh.read()
        self.assertNotIn(":05d", text)
        self.assertNotIn("glob.glob", text)
        self.assertIn("def run_files(run, clip, prefix):", text)
        with tempfile.TemporaryDirectory() as root:
            expected = os.path.join(root, "sbs_7.preview.png")
            with open(expected, "wb") as fh:
                fh.write(b"evidence")
            Image.fromarray(np.zeros((2, 4), dtype=np.uint8)).save(
                os.path.join(root, "sbs_00042.png"))
            indexed = sbsbench.indexed_files(os.path.join(root, "sbs_*.*"), "sbs_")
            self.assertEqual(indexed[7], expected)
            self.assertIn(42, indexed)
            self.assertEqual(sbsbench.predecessor_frame_id(indexed, 42), 7)
            self.assertIsNone(sbsbench.predecessor_frame_id(indexed, 7))
        self.assertIn("predecessor_frame_id(gt_depth_files(clip), idx)", text)
        self.assertIn("predecessor_frame_id(source_files(clip), idx)", text)
        self.assertNotIn("idx - 1", text)

    def test_harness_orders_numeric_frames_and_rejects_mixed_dimensions(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"), encoding="utf-8") as fh:
            harness = fh.read()
        self.assertIn("numeric_frame_less", harness)
        self.assertIn("std::sort(frames.begin(), frames.end(), numeric_frame_less)", harness)
        self.assertIn("mixed source dimensions are not a valid clip", harness)

    def test_phase_shift_recovers_known_translation(self):
        rng = np.random.default_rng(1234)
        a = rng.random((64, 64))
        b = np.roll(a, shift=(2, -5), axis=(0, 1))
        dy, dx = sbsbench.phase_shift(a, b)
        self.assertAlmostEqual(dy, -2.0, places=5)
        self.assertAlmostEqual(dx, 5.0, places=5)

    def test_translation_residual_validates_nonwrapping_alignment(self):
        rng = np.random.default_rng(4321)
        source = rng.random((48, 64), dtype=np.float32)
        shifted = np.zeros_like(source)
        shifted[:, 7:] = source[:, :-7]
        self.assertLess(sbsbench.translation_residual(source, shifted, 0, -7),
                        sbsbench.translation_residual(source, shifted, 0, 0) * 0.1)

    def test_disparity_field_rejects_photometrically_invalid_peak(self):
        rng = np.random.default_rng(987)
        left = rng.random((64, 64), dtype=np.float32)
        with mock.patch.object(sbsbench, "phase_shift", return_value=(15.0, -30.0)):
            field = sbsbench.disparity_field(left, left.copy(), tile=64, stride=64)
        self.assertIsNone(field)

    def test_disparity_field_covers_tile_sized_frame_and_final_borders(self):
        rng = np.random.default_rng(2026)
        left = rng.random((192, 320), dtype=np.float32)
        right = np.roll(left, 3, axis=1)
        field = sbsbench.disparity_field(left, right, tile=192, stride=128)
        self.assertIsNotNone(field)
        self.assertEqual(len(field[0]), 2)  # x=0 and the border-aligned x=128 tile
        self.assertEqual(sbsbench._tile_positions(320, 192, 128), [0, 128])

    def test_split_eyes_rejects_malformed_odd_width(self):
        with self.assertRaisesRegex(ValueError, "width must be even"):
            sbsbench.split_eyes(np.zeros((8, 17), np.float32))

    def test_float_resize_does_not_quantize_metric_evidence(self):
        source = np.array([[0.5001, 0.5002], [0.5003, 0.5004]], np.float32)
        resized = sbsbench.resize_to(source, 4, 4)
        self.assertGreater(float(np.ptp(resized)), 1e-4)

    def test_silhouette_downscale_preserves_one_pixel_edge_support(self):
        depth = np.zeros((64, 192), np.float32)
        depth[:, 97:] = 1.0
        edge = sbsbench.silhouette_edges(depth, 64, 32)
        self.assertTrue(edge.any())

    def test_horizontal_morphology_does_not_wrap_opposite_border(self):
        mask = np.zeros((3, 12), bool)
        mask[:, -3:] = True
        self.assertFalse(sbsbench.hdilate(mask, 1)[:, 0].any())
        image = mask.astype(np.float32)
        self.assertEqual(float(sbsbench._hopen(image, 1)[:, 0].max()), 0.0)

    def test_sequence_joins_by_frame_identity(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            sbs = np.zeros((16, 32, 3), dtype=np.uint8)
            src = np.zeros((16, 16, 3), dtype=np.uint8)
            Image.fromarray(sbs).save(os.path.join(seq, "sbs_00007.png"))
            Image.fromarray(src).save(os.path.join(frames, "frame_00007.png"))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({}, fh)
            rows, agg = sbsbench.measure_sequence(seq, frames)
            self.assertEqual(rows[0]["_frame_id"], 7)
            self.assertEqual(agg["_n"], 1)

    def test_direct_sequence_scores_canonical_order_not_conditioned_parallax(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            Image.fromarray(np.zeros((4, 8, 3), dtype=np.uint8)).save(
                os.path.join(seq, "sbs_00001.png"))
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                os.path.join(frames, "frame_00001.png"))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as stream:
                json.dump({}, stream)

            # Canonical order says sample 2 is nearer than sample 1, while the cliff limiter has
            # raised sample 1's final displacement above sample 2. The scorer must receive the
            # former; parallax remains authenticated but is never substituted as semantic depth.
            order = np.array([[1.0, -1.0, 0.9], [1.0, -1.0, 0.9]], dtype="<f4")
            encoded = np.array([[0.75, 0.7375, 0.725],
                                [0.75, 0.7375, 0.725]], dtype="<f4")
            depth_path = os.path.join(seq, "depth_00001.f32")
            parallax_path = os.path.join(seq, "parallax_00001.f32")
            order.tofile(depth_path)
            encoded.tofile(parallax_path)
            field = {
                "frame_id": "00001", "width": 3, "height": 2,
                "parallax_sha256": run_eval.file_sha256(parallax_path),
                "maximum_absolute_source_u": float(np.max(np.abs(
                    (encoded.astype(np.float64) * 2.0 - 1.0) *
                    run_eval.direct_geometry.SOURCE_U_LIMIT))),
                "order_sha256": run_eval.file_sha256(depth_path),
                "order_minimum": float(np.min(order)),
                "order_maximum": float(np.max(order)),
            }
            manifest = {**run_eval._DIRECT_GEOMETRY_MANIFEST_V6, "fields": [field]}
            manifest_path = os.path.join(seq, "direct_parallax_manifest.json")
            with open(manifest_path, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(manifest, stream, indent=2)
                stream.write("\n")
            contract = {
                "schema": run_eval.direct_geometry.CONTRACT_SCHEMA,
                "warp_input": run_eval.direct_geometry.WARP_INPUT,
                "direct_parallax_frames": 1,
                "direct_parallax": run_eval._DIRECT_GEOMETRY_DESCRIPTOR_V6,
                "direct_parallax_manifest": {
                    "file": "direct_parallax_manifest.json",
                    "schema": run_eval.direct_geometry.MANIFEST_SCHEMA,
                    "sha256": run_eval.file_sha256(manifest_path),
                },
            }
            with open(os.path.join(seq, "contract.json"), "w", encoding="utf-8") as stream:
                json.dump(contract, stream)

            def inspect_job(jobs):
                self.assertEqual(len(jobs), 1)
                self.assertEqual(jobs[0]["depth_shape"], (2, 3))
                np.testing.assert_array_equal(
                    sbsbench.load_depth(jobs[0]["depth_path"], jobs[0]["depth_shape"]), order)
                self.assertNotEqual(jobs[0]["depth_path"], parallax_path)
                return [{"_frame_id": 1}]

            with mock.patch.object(
                    sbsbench, "_measure_sequence_spatial_rows", side_effect=inspect_job):
                rows, agg = sbsbench.measure_sequence(seq, frames)
            self.assertEqual(rows, [{"_frame_id": 1}])
            self.assertEqual(agg["_n"], 1)

    def _write_parallel_sequence_fixture(self, root, frame_count=8):
        seq = os.path.join(root, "seq")
        frames = os.path.join(root, "frames")
        os.makedirs(seq)
        os.makedirs(frames)
        yy, xx = np.mgrid[:32, :48]
        for frame_id in range(frame_count):
            source = np.stack((
                (xx * 5 + frame_id * 7) % 256,
                (yy * 9 + xx * 2 + frame_id * 3) % 256,
                ((xx + yy) * 4 + frame_id * 11) % 256,
            ), axis=2).astype(np.uint8)
            sbs = np.concatenate((source, source), axis=1)
            if frame_id == frame_count - 2:
                sbs[9:18, 12:23] = 0
            Image.fromarray(sbs, "RGB").save(
                os.path.join(seq, f"sbs_{frame_id:05d}.png"))
            Image.fromarray(source, "RGB").save(
                os.path.join(frames, f"frame_{frame_id:05d}.png"))
        with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as stream:
            json.dump({}, stream)
        return seq, frames

    def test_sequence_parallel_spatial_rows_are_serial_equivalent(self):
        with tempfile.TemporaryDirectory() as root:
            seq, frames = self._write_parallel_sequence_fixture(root)
            with mock.patch.dict(
                    os.environ, {sbsbench.SEQUENCE_SPATIAL_WORKERS_ENV: "1"}):
                serial_rows, serial_agg = sbsbench.measure_sequence(seq, frames)
            with mock.patch.dict(
                    os.environ, {sbsbench.SEQUENCE_SPATIAL_WORKERS_ENV: "2"}):
                parallel_rows, parallel_agg = sbsbench.measure_sequence(seq, frames)

            self.assertEqual(serial_rows, parallel_rows)
            self.assertEqual(serial_agg, parallel_agg)
            self.assertEqual(
                [row["_frame_id"] for row in parallel_rows], list(range(8)))
            self.assertEqual(parallel_agg["_n"], 8)

    def test_sequence_thread_spatial_rows_are_process_equivalent(self):
        with tempfile.TemporaryDirectory() as root:
            seq, frames = self._write_parallel_sequence_fixture(root)
            with mock.patch.dict(os.environ, {
                    sbsbench.SEQUENCE_SPATIAL_WORKERS_ENV: "2",
                    sbsbench.SEQUENCE_SPATIAL_BACKEND_ENV: "process"}):
                process_rows, process_agg = sbsbench.measure_sequence(seq, frames)
            with mock.patch.dict(os.environ, {
                    sbsbench.SEQUENCE_SPATIAL_WORKERS_ENV: "2",
                    sbsbench.SEQUENCE_SPATIAL_BACKEND_ENV: "thread"}):
                thread_rows, thread_agg = sbsbench.measure_sequence(seq, frames)

            self.assertEqual(process_rows, thread_rows)
            self.assertEqual(process_agg, thread_agg)

    def test_sequence_parallel_worker_failure_is_frame_local_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            seq, frames = self._write_parallel_sequence_fixture(root)
            broken = os.path.join(seq, "sbs_00004.png")
            with open(broken, "wb") as stream:
                stream.write(b"not a PNG")
            with mock.patch.dict(
                    os.environ, {sbsbench.SEQUENCE_SPATIAL_WORKERS_ENV: "2"}):
                with self.assertRaisesRegex(RuntimeError, "failed for frame 4"):
                    sbsbench.measure_sequence(seq, frames)

    def test_sequence_worker_count_is_bounded_and_environment_controlled(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(sbsbench._sequence_spatial_worker_count(7), 1)
            self.assertEqual(
                sbsbench._sequence_spatial_worker_count(8),
                min(sbsbench.SEQUENCE_SPATIAL_MAX_WORKERS,
                    os.cpu_count() or 1, 8))
            self.assertEqual(
                sbsbench._sequence_spatial_worker_count(24, 12_000_000),
                min(sbsbench.SEQUENCE_SPATIAL_MAX_WORKERS,
                    os.cpu_count() or 1, 24, 2))
        for configured, frames, expected in (("1", 24, 1), ("24", 2, 2)):
            with self.subTest(configured=configured), mock.patch.dict(
                    os.environ, {sbsbench.SEQUENCE_SPATIAL_WORKERS_ENV: configured}):
                self.assertEqual(
                    sbsbench._sequence_spatial_worker_count(frames), expected)
        for invalid in ("0", "65", "many"):
            with self.subTest(invalid=invalid), mock.patch.dict(
                    os.environ, {sbsbench.SEQUENCE_SPATIAL_WORKERS_ENV: invalid}):
                with self.assertRaisesRegex(ValueError, "must be"):
                    sbsbench._sequence_spatial_worker_count(8)
        with mock.patch.dict(os.environ, {
                sbsbench.SEQUENCE_SPATIAL_WORKERS_ENV: "2",
                sbsbench.SEQUENCE_SPATIAL_BACKEND_ENV: "invalid"}):
            with self.assertRaisesRegex(ValueError, "must be 'process' or 'thread'"):
                sbsbench._measure_sequence_spatial_rows([{}] * 2)

    def test_clip_parallelism_reserves_the_inner_peak_from_one_global_budget(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            for frame_id in range(8):
                Image.new("RGB", (100, 100)).save(
                    os.path.join(seq, f"sbs_{frame_id:05d}.png"))
            with open(os.path.join(seq, "warp_map_shape.json"), "w",
                      encoding="utf-8") as shape_file:
                json.dump({"width": 100, "height": 100}, shape_file)
            job = ("clip", seq, frames)
            with mock.patch.dict(os.environ, {
                    sbsbench.SEQUENCE_SPATIAL_PIXEL_BUDGET_ENV: "0.03",
            }, clear=True):
                budget = eval_parallel._configured_pixel_budget_pixels()
                self.assertEqual(budget, 30_000)
                # Eight frames select three 10k-pixel workers under a 30k global allowance.
                self.assertEqual(
                    eval_parallel._clip_spatial_reservation(job, budget),
                    30_000)

            # An explicit worker override cannot bypass the memory contract.
            with mock.patch.dict(os.environ, {
                    sbsbench.SEQUENCE_SPATIAL_PIXEL_BUDGET_ENV: "0.03",
                    sbsbench.SEQUENCE_SPATIAL_WORKERS_ENV: "4",
            }, clear=True), self.assertRaisesRegex(ValueError, "run-global"):
                eval_parallel._clip_spatial_reservation(job, 30_000)

            no_map = os.path.join(root, "no-map")
            os.makedirs(no_map)
            for frame_id in range(8):
                Image.new("RGB", (100, 100)).save(
                    os.path.join(no_map, f"sbs_{frame_id:05d}.png"))
            with mock.patch.dict(os.environ, {
                    sbsbench.SEQUENCE_SPATIAL_PIXEL_BUDGET_ENV: "0.03",
            }, clear=True), mock.patch.object(
                    sbsbench.os, "cpu_count", return_value=8
            ), self.assertRaisesRegex(ValueError, "run-global"):
                # A legacy sequence without mapping_shape passes ``None`` to the real inner
                # worker calculation. Reserve that actual eight-worker peak, not the three
                # workers a guessed 10k-pixel shape would have selected.
                eval_parallel._clip_spatial_reservation(
                    ("no-map", no_map, frames), 30_000)

            oversized = os.path.join(root, "oversized")
            os.makedirs(oversized)
            Image.new("RGB", (201, 150)).save(
                os.path.join(oversized, "sbs_00000.png"))
            with mock.patch.dict(os.environ, {
                    sbsbench.SEQUENCE_SPATIAL_PIXEL_BUDGET_ENV: "0.03",
            }, clear=True), self.assertRaisesRegex(
                    ValueError, "packed frame above"):
                eval_parallel._clip_spatial_reservation(
                    ("oversized", oversized, frames), 30_000)

    def test_concurrent_clip_scoring_never_exceeds_global_pixel_budget(self):
        weights = {"first": 60, "second": 60, "third": 40}
        active = 0
        peak = 0
        lock = threading.Lock()

        def fake_measure(job):
            nonlocal active, peak
            weight = weights[job[0]]
            with lock:
                active += weight
                peak = max(peak, active)
            time.sleep(0.03)
            with lock:
                active -= weight
            return job[0], {"measured": True}

        jobs = [(clip, "seq-" + clip, "frames-" + clip) for clip in weights]
        with mock.patch.object(
                eval_parallel, "_configured_pixel_budget_pixels", return_value=100
        ), mock.patch.object(
                eval_parallel, "_clip_spatial_reservation",
                side_effect=lambda job, _: weights[job[0]]
        ), mock.patch.object(
                eval_parallel, "_measure_clip_sequence_job", side_effect=fake_measure
        ), mock.patch.object(
                sbsbench, "enable_reusable_spatial_executor"
        ):
            measured = eval_parallel.measure_clip_sequences(jobs, jobs=3)

        self.assertEqual([clip for clip, _ in measured], list(weights))
        self.assertLessEqual(peak, 100)
        self.assertEqual(active, 0)

    def test_sequence_reports_complete_temporal_transition_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            flow_dir = os.path.join(frames, "gt_flow")
            os.makedirs(flow_dir)
            for frame_id in range(3):
                Image.fromarray(np.zeros((16, 32, 3), np.uint8)).save(
                    os.path.join(seq, f"sbs_{frame_id:05d}.png"))
                Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(
                    os.path.join(frames, f"frame_{frame_id:05d}.png"))
                if frame_id > 0:
                    np.savez_compressed(
                        os.path.join(flow_dir, f"frame_{frame_id:05d}.npz"),
                        flow=np.zeros((16, 16, 2), np.float32),
                        valid=np.ones((16, 16), bool))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({}, fh)
            with mock.patch.object(
                    sbsbench, "static_region_jitter", return_value=(2.0, 0.5)), \
                    mock.patch.object(
                        sbsbench, "flow_temporal_metrics", return_value=(3.0, None, 0.5)):
                _, agg = sbsbench.measure_sequence(seq, frames)
            self.assertEqual(agg["temporal_expected_transition_count"], 2.0)
            self.assertEqual(agg["source_temporal_transition_count"], 2.0)
            self.assertEqual(agg["static_applicable_transition_count"], 2.0)
            self.assertEqual(agg["static_measured_transition_count"], 2.0)
            self.assertEqual(agg["flow_applicable_transition_count"], 2.0)
            self.assertEqual(agg["flow_measured_transition_count"], 2.0)

    def test_sequence_rejects_missing_applicable_middle_temporal_metric(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            for frame_id in range(2):
                Image.fromarray(np.zeros((16, 32, 3), np.uint8)).save(
                    os.path.join(seq, f"sbs_{frame_id:05d}.png"))
                Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(
                    os.path.join(frames, f"frame_{frame_id:05d}.png"))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({}, fh)
            with mock.patch.object(
                    sbsbench, "static_region_jitter", return_value=(None, 0.5)), \
                    mock.patch.object(
                        sbsbench, "flow_temporal_metrics", return_value=(0.0, None, 0.5)):
                with self.assertRaisesRegex(ValueError, "static temporal metric missing"):
                    sbsbench.measure_sequence(seq, frames)

    def test_sequence_rejects_positional_mispairing(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            blank = np.zeros((16, 32, 3), dtype=np.uint8)
            Image.fromarray(blank).save(os.path.join(seq, "sbs_00008.png"))
            Image.fromarray(blank[:, :16]).save(os.path.join(frames, "frame_00007.png"))
            with self.assertRaisesRegex(ValueError, "frame-id mismatch"):
                sbsbench.measure_sequence(seq, frames)

    def test_public_clip_rejects_missing_required_ground_truth(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            Image.fromarray(np.zeros((16, 32, 3), np.uint8)).save(
                os.path.join(seq, "sbs_00000.png"))
            Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(
                os.path.join(frames, "frame_00000.png"))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as fh:
                fh.write('{"dataset":"Example Public Dataset","required_gt_depth":true}')
            with self.assertRaisesRegex(ValueError, "requires GT depth"):
                sbsbench.measure_sequence(seq, frames)

    def test_public_clip_rejects_missing_required_optical_flow(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            for frame_id in range(2):
                Image.fromarray(np.zeros((16, 32, 3), np.uint8)).save(
                    os.path.join(seq, f"sbs_{frame_id:05d}.png"))
                Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(
                    os.path.join(frames, f"frame_{frame_id:05d}.png"))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as fh:
                fh.write('{"required_gt_flow":true}')
            with self.assertRaisesRegex(ValueError, "requires GT optical flow"):
                sbsbench.measure_sequence(seq, frames)

    def test_public_clip_rejects_missing_declared_stereo_reference(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            Image.fromarray(np.zeros((16, 32, 3), np.uint8)).save(
                os.path.join(seq, "sbs_00000.png"))
            Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(
                os.path.join(frames, "frame_00000.png"))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as fh:
                fh.write('{"reference_stereo_available":true}')
            with self.assertRaisesRegex(ValueError, "diagnostic stereo reference"):
                sbsbench.measure_sequence(seq, frames)

    def test_public_clip_rejects_missing_required_depth_lag_metric(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            gt_dir = os.path.join(frames, "gt_depth")
            valid_dir = os.path.join(frames, "gt_depth_valid")
            os.makedirs(seq)
            os.makedirs(gt_dir)
            os.makedirs(valid_dir)
            for frame_id in range(2):
                Image.fromarray(np.zeros((16, 32, 3), np.uint8)).save(
                    os.path.join(seq, f"sbs_{frame_id:05d}.png"))
                Image.fromarray(np.full((8, 16), 32768, np.uint16)).save(
                    os.path.join(seq, f"depth_{frame_id:05d}.png"))
                Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(
                    os.path.join(frames, f"frame_{frame_id:05d}.png"))
                Image.fromarray(np.full((8, 16), 32768, np.uint16)).save(
                    os.path.join(gt_dir, f"frame_{frame_id:05d}.png"))
                Image.fromarray(np.full((8, 16), 255, np.uint8)).save(
                    os.path.join(valid_dir, f"frame_{frame_id:05d}.png"))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"required_gt_depth": True, "gt_depth_kind": "disparity"}, fh)
            with mock.patch.object(sbsbench, "depth_ground_truth_lag", return_value=None):
                with self.assertRaisesRegex(ValueError, "depth_gt_lag_f1_p95"):
                    sbsbench.measure_sequence(seq, frames)

    def test_duplicate_numeric_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(os.path.join(root, "frame_1.png"))
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(os.path.join(root, "frame_01.jpg"))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                sbsbench.indexed_files(os.path.join(root, "frame_*.*"), "frame_")

    def test_metric_delta_class_uses_gate_tolerance_and_direction(self):
        lower = {"better": "lower", "rel_tol": 0.25, "abs_floor": 0.5}
        self.assertEqual(sbsbench.metric_delta_class(2.0, 2.4, lower), "noise")
        self.assertEqual(sbsbench.metric_delta_class(2.0, 2.6, lower), "regressed")
        self.assertEqual(sbsbench.metric_delta_class(2.0, 1.4, lower), "improved")

    def test_metric_roles_control_committed_gate(self):
        diagnostic = {"role": "diagnostic", "better": "lower",
                      "rel_tol": 0.0, "abs_floor": 0.1}
        hard = {"role": "hard", "better": "lower", "hard_max": 0.5,
                "rel_tol": 0.0, "abs_floor": 0.1}
        self.assertFalse(sbsbench.metric_gate_failed(0.0, 99.0, diagnostic))
        self.assertFalse(sbsbench.metric_gate_failed(0.0, 0.49, hard))
        self.assertTrue(sbsbench.metric_gate_failed(0.0, 0.51, hard))
        hard_min = {"role": "hard", "better": "higher", "hard_min": 90.0,
                    "rel_tol": 0.0, "abs_floor": 1.0}
        self.assertFalse(sbsbench.metric_gate_failed(95.0, 91.0, hard_min))
        self.assertTrue(sbsbench.metric_gate_failed(95.0, 89.0, hard_min))

    def test_unrelated_artistic_stereo_metrics_are_not_in_metric_policy(self):
        with open(os.path.join(run_eval.SCRIPT_DIR, "thresholds.json"),
                  encoding="utf-8") as fh:
            specs = json.load(fh)["metrics"]
        artistic = {key: spec for key, spec in specs.items()
                    if key.startswith("stereo_art_")}
        self.assertEqual(artistic, {})

    def test_ab_decision_preserves_primary_axis_tradeoff(self):
        specs = {
            "pop": {"role": "primary", "axis": "stereo", "better": "higher",
                    "rel_tol": 0.0, "abs_floor": 0.5},
            "halo": {"role": "primary", "axis": "warp", "better": "lower",
                     "rel_tol": 0.0, "abs_floor": 0.5},
            "legacy_proxy": {"role": "diagnostic", "axis": "warp", "better": "lower",
                             "rel_tol": 0.0, "abs_floor": 0.1},
        }
        result = sbsbench.evaluate_ab_decision(
            {"clip": {"pop": 4.0, "halo": 2.0, "legacy_proxy": 0.0}},
            {"clip": {"pop": 5.0, "halo": 3.0, "legacy_proxy": 99.0}},
            ["clip"], specs)
        self.assertEqual(result["verdict"], "tradeoff")
        self.assertEqual(result["improved"], 1)
        self.assertEqual(result["regressed"], 1)

    def test_ab_decision_hard_constraint_cannot_be_traded(self):
        specs = {
            "vmis": {"role": "hard", "axis": "comfort", "hard_max": 0.5,
                     "better": "lower", "rel_tol": 0.0, "abs_floor": 0.1},
            "pop": {"role": "primary", "axis": "stereo", "better": "higher",
                    "rel_tol": 0.0, "abs_floor": 0.5},
        }
        result = sbsbench.evaluate_ab_decision(
            {"clip": {"vmis": 0.1, "pop": 4.0}},
            {"clip": {"vmis": 0.6, "pop": 8.0}}, ["clip"], specs)
        self.assertEqual(result["verdict"], "reject_hard")

    def test_static_region_jitter_ignores_source_motion_but_detects_static_warp_change(self):
        rng = np.random.default_rng(9)
        src = np.round(rng.random((64, 96), dtype=np.float32) * 255.0) / 255.0
        stable, support = sbsbench.static_region_jitter(src, src, src, src, src, src,
                                                        min_support=0.5)
        self.assertAlmostEqual(stable, 0.0)
        self.assertEqual(support, 1.0)
        changed = src.copy()
        changed[16:48, 30:66] = np.clip(changed[16:48, 30:66] + 0.2, 0, 1)
        jitter, _ = sbsbench.static_region_jitter(changed, changed, src, src, src, src,
                                                  min_support=0.5)
        self.assertGreater(jitter, 20.0)
        grain = np.where(np.indices(src.shape).sum(axis=0) % 2, 2.0, -2.0) / 255.0
        noisy_src = np.clip(src + grain, 0.0, 1.0)
        noise_only, _ = sbsbench.static_region_jitter(
            noisy_src, noisy_src, src, src, noisy_src, src, min_support=0.5)
        self.assertLess(noise_only, 1e-4)
        moving_src = np.roll(src, 8, axis=1)
        skipped, moving_support = sbsbench.static_region_jitter(
            moving_src, moving_src, src, src, moving_src, src, min_support=0.5)
        self.assertIsNone(skipped)
        self.assertLess(moving_support, 0.5)

    def test_static_jitter_uses_signed_source_conditioning(self):
        previous_source = np.full((48, 80), 0.5, np.float32)
        source_step = 2.0 / 255.0  # retained by the default static-region threshold
        current_source = previous_source + source_step
        previous_eye = np.full_like(previous_source, 0.4)

        matched_eye = previous_eye + source_step
        matched, support = sbsbench.static_region_jitter(
            matched_eye, matched_eye, previous_eye, previous_eye,
            current_source, previous_source)
        self.assertEqual(support, 1.0)
        self.assertLess(matched, 1e-4)

        opposite_eye = previous_eye - source_step
        opposite, _ = sbsbench.static_region_jitter(
            opposite_eye, opposite_eye, previous_eye, previous_eye,
            current_source, previous_source)
        self.assertAlmostEqual(opposite, 4.0, places=4)

        worse_eye, _ = sbsbench.static_region_jitter(
            matched_eye, opposite_eye, previous_eye, previous_eye,
            current_source, previous_source)
        self.assertAlmostEqual(worse_eye, 4.0, places=4)

    def test_static_jitter_resizes_each_source_frame_only_once(self):
        rng = np.random.default_rng(42)
        previous_source = rng.random((72, 120), dtype=np.float32)
        current_source = previous_source + 1.0 / 255.0
        previous_eye = sbsbench.resize_to(previous_source, 80, 48)
        current_eye = sbsbench.resize_to(current_source, 80, 48)

        with mock.patch.object(
                sbsbench, "resize_to", wraps=sbsbench.resize_to) as resize:
            measured = sbsbench.static_region_jitter(
                current_eye, current_eye, previous_eye, previous_eye,
                current_source, previous_source)

        self.assertEqual(resize.call_count, 2)
        self.assertEqual(measured[1], 1.0)
        self.assertLess(measured[0], 1e-4)

    def test_perceived_disparity_is_client_aspect_invariant(self):
        ref = sbsbench.perceived_disparity_pct(51.2, 5120, 2160)
        # The aspect correction keeps pixel disparity constant when pixel height is unchanged;
        # at a taller raster it grows in direct proportion to height.
        uhd = sbsbench.perceived_disparity_pct(51.2, 3840, 2160)
        tall = sbsbench.perceived_disparity_pct(51.2 * 3840.0 / 2160.0, 3552, 3840)
        self.assertAlmostEqual(ref, uhd, places=6)
        self.assertAlmostEqual(ref, tall, places=6)

    def test_hard_integrity_aggregates_worst_frame_not_mean(self):
        agg = sbsbench.aggregate([
            {"source_coverage_pct": 100.0, "exact_positive_disparity_pct": 0.5,
             "vmisalign_p99_pct": 0.01,
             "source_coverage_worst_patch_bad_pct": 0.0,
             "depth_gt_polarity_ok": 100.0},
            {"source_coverage_pct": 70.0, "exact_positive_disparity_pct": 4.0,
             "vmisalign_p99_pct": 0.2,
             "source_coverage_worst_patch_bad_pct": 80.0,
             "depth_gt_polarity_ok": 0.0},
        ])
        self.assertEqual(agg["source_coverage_pct"], 70.0)
        self.assertEqual(agg["exact_positive_disparity_pct"], 4.0)
        self.assertEqual(agg["vmisalign_p99_pct"], 0.2)
        self.assertEqual(agg["source_coverage_worst_patch_bad_pct"], 80.0)
        self.assertEqual(agg["depth_gt_polarity_ok"], 0.0)

    def test_print_diff_does_not_repeat_temporal_metrics(self):
        values = {"static_jitter_p95": 2.0}
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            sbsbench.print_diff(values, {"static_jitter_p95": 1.0}, sbsbench.SEQ_FMT)
        self.assertEqual(output.getvalue().count("static_jitter_p95"), 1)

    def test_resolution_independent_metrics_preserve_normalized_geometry(self):
        small = sbsbench.perceived_disparity_pct(4.0, 400, 200)
        large = sbsbench.perceived_disparity_pct(8.0, 800, 400)
        self.assertAlmostEqual(small, large)

    def test_source_coverage_and_integrity_detect_missing_content(self):
        rng = np.random.default_rng(22)
        src = np.round(rng.random((96, 160), dtype=np.float32) * 255.0) / 255.0
        clean = sbsbench._shift_x_edge(src, 5)
        good = sbsbench.source_relative_metrics(clean, src, max_shift=8)
        damaged = clean.copy()
        damaged[20:76, 60:110] = 0.0
        bad = sbsbench.source_relative_metrics(damaged, src, max_shift=8)
        self.assertGreater(good["source_coverage_pct"], 99.0)
        self.assertGreater(good["image_integrity_pct"], 99.0)
        self.assertLess(bad["source_coverage_pct"], good["source_coverage_pct"] - 15.0)
        self.assertLess(bad["image_integrity_pct"], good["image_integrity_pct"] - 10.0)

    def test_ground_truth_depth_metrics_reward_aligned_structure(self):
        gt = np.full((96, 160), 0.25, np.float32)
        gt[:, 80:] = 0.75
        equivalent = gt * 0.8 + 0.1  # monocular scale/shift ambiguity is intentionally free
        flat = np.full_like(gt, 0.5)
        good = sbsbench.depth_ground_truth_metrics(equivalent, gt)
        bad = sbsbench.depth_ground_truth_metrics(flat, gt)
        self.assertLess(good["depth_gt_affine_nrmse_pct"], 0.01)
        self.assertEqual(good["depth_gt_polarity_ok"], 100.0)
        self.assertGreater(good["depth_gt_valid_pct"], 99.9)
        self.assertGreater(good["depth_gt_edge_f1"], 99.0)
        self.assertGreater(bad["depth_gt_affine_nrmse_pct"], 40.0)
        self.assertLess(bad["depth_gt_edge_f1"], 1.0)

    def test_ground_truth_depth_metrics_reject_inverted_polarity(self):
        gt = np.full((96, 160), 0.2, np.float32)
        gt[:, 80:] = 0.8
        inverted = 1.0 - gt
        metrics = sbsbench.depth_ground_truth_metrics(inverted, gt)
        self.assertEqual(metrics["depth_gt_polarity_ok"], 0.0)
        self.assertGreater(metrics["depth_gt_affine_nrmse_pct"], 20.0)
        self.assertLess(metrics["depth_gt_edge_f1"], 1.0)

    def test_ground_truth_depth_lag_detects_previous_frame_geometry(self):
        previous = np.zeros((32, 48), np.float32)
        previous[8:24, 8:20] = 1.0
        current = np.zeros_like(previous)
        current[8:24, 18:30] = 1.0
        self.assertGreater(
            sbsbench.depth_ground_truth_lag(previous, current, previous), 50.0)
        self.assertEqual(
            sbsbench.depth_ground_truth_lag(current, current, previous), 0.0)

    def test_ground_truth_depth_lag_reuses_current_spatial_edge_score(self):
        previous = np.zeros((32, 48), np.float32)
        previous[8:24, 8:20] = 1.0
        current = np.zeros_like(previous)
        current[8:24, 18:30] = 1.0
        current_metrics = sbsbench.depth_ground_truth_metrics(previous, current)
        expected = sbsbench.depth_ground_truth_lag(previous, current, previous)

        with mock.patch.object(
                sbsbench, "depth_ground_truth_metrics",
                wraps=sbsbench.depth_ground_truth_metrics) as measure_gt:
            reused = sbsbench.depth_ground_truth_lag(
                previous, current, previous,
                current_edge_f1=current_metrics["depth_gt_edge_f1"])

        self.assertEqual(measure_gt.call_count, 1)
        self.assertEqual(reused, expected)

    def test_ground_truth_edge_tolerance_works_in_both_axes(self):
        gt = np.full((96, 160), 0.25, np.float32)
        gt[48:, :] = 0.75
        shifted = np.full_like(gt, 0.25)
        shifted[49:, :] = 0.75
        metrics = sbsbench.depth_ground_truth_metrics(shifted, gt)
        self.assertGreater(metrics["depth_gt_edge_f1"], 99.0)

    def test_metric_depth_resize_does_not_invert_interpolated_invalid_holes(self):
        gt = np.full((48, 80), 2.0, np.float32)
        gt[12:36, 38:42] = 0.0
        resized, valid = sbsbench.resize_metric_depth(gt, 40, 24)
        self.assertTrue(np.all(resized[valid] > 1.9))
        self.assertTrue(np.all(resized[valid] < 2.1))
        prediction = np.full((24, 40), 0.5, np.float32)
        metrics = sbsbench.depth_ground_truth_metrics(prediction, gt, "metric")
        self.assertLess(metrics["depth_gt_affine_nrmse_pct"], 0.01)
        self.assertGreater(metrics["depth_gt_edge_f1"], 99.0)

    def test_metric_depth_edges_are_invariant_to_distant_inverse_depth_units(self):
        gt = np.full((64, 96), 50.0, np.float32)
        gt[:, 48:] = 55.0
        matching_inverse_depth = 1.0 / gt
        matching = sbsbench.depth_ground_truth_metrics(
            matching_inverse_depth, gt, "metric")
        flat = sbsbench.depth_ground_truth_metrics(
            np.full_like(gt, float(np.mean(matching_inverse_depth))), gt, "metric")
        self.assertGreater(matching["depth_gt_edge_f1"], 99.0)
        self.assertLess(flat["depth_gt_edge_f1"], 1.0)

    def test_optical_flow_temporal_metric_compensates_motion(self):
        rng = np.random.default_rng(31)
        previous = np.round(rng.random((96, 160), dtype=np.float32) * 255.0) / 255.0
        current = np.roll(previous, 5, axis=1)
        stable, _, support = sbsbench.flow_temporal_metrics(
            current, current, previous, previous, current, previous, min_support=0.1)
        corrupted = current.copy()
        corrupted[20:75, 60:110] = 0.0
        unstable, _, _ = sbsbench.flow_temporal_metrics(
            corrupted, corrupted, previous, previous, current, previous, min_support=0.1)
        self.assertGreater(support, 0.8)
        self.assertLess(stable, 2.0)
        self.assertGreater(unstable, stable + 100.0)

    def test_flow_temporal_metric_uses_edge_preserving_depth_transport(self):
        rng = np.random.default_rng(311)
        source = rng.random((48, 80), dtype=np.float32)
        depth = np.zeros((48, 80), np.float32)
        depth[:, 30:] = 1.0
        with mock.patch.object(
                sbsbench, "warp_previous_nearest_with_flow",
                wraps=sbsbench.warp_previous_nearest_with_flow) as nearest:
            sbsbench.flow_temporal_metrics(
                source, source, source, source, source, source,
                depth, depth, min_support=0.1)
        self.assertTrue(nearest.called)

    def test_nearest_flow_warp_preserves_depth_steps(self):
        previous = np.zeros((16, 24), np.float32)
        previous[:, 8:] = 1.0
        u = np.full_like(previous, 3.0)
        v = np.zeros_like(previous)
        warped, valid = sbsbench.warp_previous_nearest_with_flow(previous, u, v)
        self.assertTrue(valid[:, 3:].all())
        self.assertEqual(set(np.unique(warped)), {0.0, 1.0})
        self.assertTrue((warped[:, 11:] == 1.0).all())

    def test_depth_confidence_ignores_flat_depth(self):
        source = np.tile(np.linspace(0.0, 1.0, 96, dtype=np.float32), (48, 1))
        depth = np.full((48, 96), 0.5, np.float32)
        result = audit_depth_confidence.depth_confidence_map(depth, source)
        self.assertFalse(result["band"].any())
        self.assertTrue(np.all(result["risk"] == 0.0))
        self.assertTrue(np.all(result["confidence"] == 1.0))

    def test_depth_confidence_prefers_sharp_aligned_edges(self):
        source = np.zeros((48, 96), np.float32)
        source[:, 48:] = 1.0
        sharp = np.zeros_like(source)
        sharp[:, 48:] = 1.0
        shifted = np.zeros_like(source)
        shifted[:, 56:] = 1.0
        soft = np.zeros_like(source)
        soft[:, 44:53] = np.linspace(0.0, 1.0, 9, dtype=np.float32)
        soft[:, 53:] = 1.0
        sharp_result = audit_depth_confidence.depth_confidence_map(sharp, source)
        shifted_result = audit_depth_confidence.depth_confidence_map(shifted, source)
        soft_result = audit_depth_confidence.depth_confidence_map(soft, source)
        sharp_risk = sharp_result["model_risk"]
        shifted_risk = shifted_result["model_risk"]
        soft_risk = soft_result["model_risk"]
        self.assertLess(float(sharp_risk.max()), 0.1)
        self.assertGreater(float(shifted_risk.max()), 0.6)
        self.assertGreater(float(soft_risk.max()), float(sharp_risk.max()) + 0.2)
        self.assertGreater(float(sharp_result["warp_risk"].max()), 0.5)

    def test_depth_confidence_detects_flow_compensated_temporal_change(self):
        rng = np.random.default_rng(91)
        source = rng.random((64, 128), dtype=np.float32)
        previous = np.zeros_like(source)
        previous[:, 48:] = 1.0
        current = previous.copy()
        current[16:48, 48:] = 0.25
        stable = audit_depth_confidence.depth_confidence_map(
            previous, source, previous_depth=previous, previous_src=source)
        changed = audit_depth_confidence.depth_confidence_map(
            current, source, previous_depth=previous, previous_src=source)
        valid = changed["band"] & changed["temporal_valid"]
        self.assertTrue(valid.any())
        self.assertLess(float(stable["temporal"].max()), 0.01)
        self.assertGreater(float(changed["temporal"][valid].max()), 0.9)

    def test_confidence_audit_auc_is_tie_aware(self):
        labels = np.array([False, True, False, True])
        self.assertEqual(
            audit_depth_confidence.rank_auc(np.array([0.0, 1.0, 0.0, 1.0]), labels), 1.0)
        self.assertEqual(
            audit_depth_confidence.rank_auc(np.ones(4), labels), 0.5)
        self.assertIsNone(
            audit_depth_confidence.rank_auc(np.arange(4), np.zeros(4, bool)))

    def test_confidence_audit_rejects_tiny_pixel_classes(self):
        risk = np.zeros((16, 16), np.float32)
        risk[:, 8:] = 1.0
        confidence = {"risk": risk, "band": np.ones_like(risk, bool)}
        severity = np.zeros_like(risk)
        severity[:2, :8] = 2.0  # only 16 artifact pixels despite perfect ranking
        row, _, _, _ = audit_depth_confidence.validation_row(
            confidence, severity, np.ones_like(risk, bool))
        self.assertEqual(row["artifact_positive_px"], 16)
        self.assertIsNone(row["artifact_auc"])

    def test_confidence_audit_fails_closed_when_gt_evidence_is_missing(self):
        rows = [{"artifact_auc": 0.8, "artifact_capture_pct": 90.0} for _ in range(4)]
        stats = audit_depth_confidence.calibration_decision(rows, 4, 4)
        self.assertTrue(stats["warp_screening_validated"])
        self.assertFalse(stats["model_boundary_validated"])
        self.assertEqual(stats["gt_auc_frames"], 0)
        rows[0]["gt_bad_edge_auc"] = 0.7
        rows[1]["gt_bad_edge_auc"] = 0.6
        stats = audit_depth_confidence.calibration_decision(rows, 4, 4)
        self.assertTrue(stats["model_boundary_validated"])

    def test_confidence_audit_allows_flat_gt_without_boundary_auc(self):
        rows = [{"artifact_auc": 0.8, "artifact_capture_pct": 90.0} for _ in range(4)]
        stats = audit_depth_confidence.calibration_decision(rows, 0, 4)
        self.assertTrue(stats["warp_screening_validated"])
        self.assertIsNone(stats["model_boundary_validated"])
        self.assertEqual(stats["gt_frames_available"], 4)
        self.assertEqual(stats["gt_frames_eligible"], 0)

    def test_confidence_audit_rejects_frame_identity_drift(self):
        with self.assertRaisesRegex(ValueError, "missing=\\[2\\], extra=\\[3\\]"):
            audit_depth_confidence.require_frame_ids("depth", [1, 2], [1, 3])

    def test_exact_forward_flow_temporal_metric_compensates_motion(self):
        rng = np.random.default_rng(71)
        previous = np.round(rng.random((96, 160), dtype=np.float32) * 255.0) / 255.0
        current = np.zeros_like(previous)
        current[:, 5:] = previous[:, :-5]
        flow = np.zeros((96, 160, 2), np.float32)
        flow[..., 0] = 5.0
        valid = np.ones((96, 160), bool)
        valid[:, -5:] = False
        stable, _, support = sbsbench.flow_temporal_metrics(
            current, current, previous, previous, current, previous, min_support=0.1,
            reference_flow=flow, reference_valid=valid)
        self.assertGreater(support, 0.8)
        self.assertLess(stable, 2.0)

    def test_flow_temporal_uses_registered_signed_source_conditioning(self):
        height, width = 48, 80
        previous_source = np.broadcast_to(
            np.linspace(0.35, 0.45, width, dtype=np.float32),
            (height, width)).copy()
        previous_eye = previous_source.copy()
        shift = 2
        source_step = 4.0 / 255.0

        current_source = np.zeros_like(previous_source)
        current_source[:, shift:] = previous_source[:, :-shift] + source_step
        matched_eye = current_source.copy()
        opposite_eye = np.zeros_like(previous_eye)
        opposite_eye[:, shift:] = previous_eye[:, :-shift] - source_step

        flow = np.zeros((height, width, 2), np.float32)
        flow[..., 0] = float(shift)
        valid = np.ones((height, width), bool)
        valid[:, -shift:] = False

        matched, _, support = sbsbench.flow_temporal_metrics(
            matched_eye, matched_eye, previous_eye, previous_eye,
            current_source, previous_source, min_support=0.1,
            reference_flow=flow, reference_valid=valid)
        self.assertGreater(support, 0.9)
        self.assertLess(matched, 1e-4)

        opposite, _, _ = sbsbench.flow_temporal_metrics(
            opposite_eye, opposite_eye, previous_eye, previous_eye,
            current_source, previous_source, min_support=0.1,
            reference_flow=flow, reference_valid=valid)
        self.assertAlmostEqual(opposite, 8.0, places=3)

        worse_eye, _, _ = sbsbench.flow_temporal_metrics(
            matched_eye, opposite_eye, previous_eye, previous_eye,
            current_source, previous_source, min_support=0.1,
            reference_flow=flow, reference_valid=valid)
        self.assertAlmostEqual(worse_eye, 8.0, places=3)

    def test_npy_metric_depth_preserves_native_values(self):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as fh:
            path = fh.name
        try:
            expected = np.array([[0.25, 4.0], [12.5, 200.0]], np.float32)
            np.save(path, expected)
            np.testing.assert_array_equal(sbsbench.load_depth(path), expected)
        finally:
            os.unlink(path)

    def test_public_dataset_timestamp_association_is_nearest_and_unique(self):
        rgb = [(0.00, "r0"), (0.10, "r1"), (0.20, "r2")]
        depth = [(0.009, "d0"), (0.105, "d1"), (0.35, "far")]
        pairs = prepare_public_datasets.associate_timestamps(rgb, depth, 0.03)
        self.assertEqual([(p[1], p[3]) for p in pairs], [("r0", "d0"), ("r1", "d1")])

    def test_public_dataset_v4_canvases_fit_only_authenticated_production_shapes(self):
        manifest = prepare_public_datasets.load_manifest(
            prepare_public_datasets.MANIFEST_PATH)
        self.assertEqual(manifest["schema"], 3)
        self.assertEqual(manifest["prepared_suite"], "extended-v4")
        self.assertEqual(
            prepare_public_datasets.PRODUCTION_V2_TENSOR_SHAPES,
            frozenset({
                (770, 434), (1022, 434), (1036, 434),
                (434, 770), (434, 1022), (434, 1036),
            }),
        )
        padded = {
            "bonn_person_walk": ((640, 480), (854, 480), (770, 434)),
            "bonn_person_close": ((640, 480), (854, 480), (770, 434)),
            "tartanair_house_easy": ((640, 640), (1138, 640), (770, 434)),
            "tartanair_house_motion": ((640, 640), (1138, 640), (770, 434)),
            "vkitti_drive_clone": ((1242, 375), (1242, 520), (1036, 434)),
            "vkitti_drive_rain": ((1242, 375), (1242, 520), (1036, 434)),
        }
        for clip_id, clip in manifest["clips"].items():
            with self.subTest(clip=clip_id):
                geometry = prepare_public_datasets.preparation_geometry_contract(
                    clip_id, clip)
                canvas = geometry["canvas_shape"]
                tensor = geometry["depth_tensor_shape"]
                fitted = prepare_public_datasets.fit_host_sbs_v2_depth_tensor_shape(
                    canvas["width"], canvas["height"])
                self.assertIn(fitted, prepare_public_datasets.PRODUCTION_V2_TENSOR_SHAPES)
                self.assertEqual(fitted, (tensor["width"], tensor["height"]))
                if clip_id in padded:
                    source_expected, canvas_expected, tensor_expected = padded[clip_id]
                    source = geometry["source_shape"]
                    self.assertEqual((source["width"], source["height"]), source_expected)
                    self.assertEqual((canvas["width"], canvas["height"]), canvas_expected)
                    self.assertEqual(fitted, tensor_expected)
                    self.assertEqual(geometry["method"], "center-pad-black-no-resize")
                else:
                    self.assertEqual(geometry["method"], "identity")

    def test_public_dataset_python_fitter_is_bound_to_native_production_constants(self):
        with open(os.path.join(run_eval.REPO, "src", "config.h"), encoding="utf-8") as stream:
            config_source = stream.read()
        with open(os.path.join(run_eval.REPO, "src", "model_manager.h"),
                  encoding="utf-8") as stream:
            model_source = stream.read()
        self.assertRegex(config_source, r"depth_short_side\s*=\s*432\s*;")
        self.assertRegex(config_source, r"depth_max_aspect\s*=\s*4\.0\s*;")
        self.assertRegex(model_source, r"depth_engine_max_dim\s*=\s*1036\s*;")
        self.assertEqual(prepare_public_datasets.PRODUCTION_DEPTH_PATCH, 14)

    def test_public_dataset_center_padding_preserves_pixels_and_sidecar_coordinates(self):
        geometry = {
            "method": "center-pad-black-no-resize",
            "source_shape": {"width": 3, "height": 2},
            "canvas_shape": {"width": 7, "height": 5},
            "content_offset": {"x": 2, "y": 1},
        }
        rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        depth = (np.arange(6, dtype=np.uint16).reshape(2, 3) + 1) * 100
        valid = np.array([[True, False, True], [True, True, False]])
        flow = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
        with tempfile.TemporaryDirectory() as directory:
            for subdirectory in (
                    "gt_depth", "gt_depth_valid", "gt_outofframe", "gt_flow"):
                os.makedirs(os.path.join(directory, subdirectory))
            Image.fromarray(rgb, "RGB").save(os.path.join(directory, "frame_00000.png"))
            Image.fromarray(depth).save(
                os.path.join(directory, "gt_depth", "frame_00000.png"))
            Image.fromarray(valid.astype(np.uint8) * 255, "L").save(
                os.path.join(directory, "gt_depth_valid", "frame_00000.png"))
            Image.fromarray(np.zeros((2, 3), np.uint8), "L").save(
                os.path.join(directory, "gt_outofframe", "frame_00000.png"))
            np.savez_compressed(
                os.path.join(directory, "gt_flow", "frame_00000.npz"),
                flow=flow, valid=valid)

            prepare_public_datasets.apply_center_padding("fixture", directory, geometry)
            prepare_public_datasets.validate_prepared_evidence_geometry(
                "fixture", directory, 7, 5)

            with Image.open(os.path.join(directory, "frame_00000.png")) as image:
                padded_rgb = np.asarray(image)
            with Image.open(os.path.join(
                    directory, "gt_depth", "frame_00000.png")) as image:
                padded_depth = np.asarray(image)
            with Image.open(os.path.join(
                    directory, "gt_depth_valid", "frame_00000.png")) as image:
                padded_valid = np.asarray(image) != 0
            with Image.open(os.path.join(
                    directory, "gt_outofframe", "frame_00000.png")) as image:
                padded_outside = np.asarray(image) != 0
            with np.load(os.path.join(
                    directory, "gt_flow", "frame_00000.npz"), allow_pickle=False) as archive:
                padded_flow = archive["flow"]
                padded_flow_valid = archive["valid"]

            interior = np.s_[1:3, 2:5]
            np.testing.assert_array_equal(padded_rgb[interior], rgb)
            np.testing.assert_array_equal(padded_depth[interior], depth)
            np.testing.assert_array_equal(padded_valid[interior], valid)
            np.testing.assert_array_equal(padded_flow[interior], flow)
            np.testing.assert_array_equal(padded_flow_valid[interior], valid)
            self.assertFalse(np.any(padded_rgb[:, :2]))
            self.assertFalse(np.any(padded_depth[:, :2]))
            self.assertFalse(np.any(padded_valid[:, :2]))
            self.assertFalse(np.any(padded_flow[:, :2]))
            self.assertFalse(np.any(padded_flow_valid[:, :2]))
            self.assertTrue(np.all(padded_outside[:, :2]))
            self.assertFalse(np.any(padded_outside[interior]))

    def test_public_dataset_native_unsupported_shape_requires_a_canvas(self):
        clip = {"source_shape": {"width": 640, "height": 480}}
        with self.assertRaisesRegex(RuntimeError, r"fits unsupported V2 tensor 574x434"):
            prepare_public_datasets.preparation_geometry_contract("bonn", clip)

    def test_public_dataset_manifest_paths_are_single_safe_components(self):
        self.assertEqual(
            prepare_public_datasets.safe_path_component("clip-name", "clip ID"),
            "clip-name")
        for unsafe in ("", ".", "..", "../clip", "nested/clip", r"..\clip",
                       r"nested\clip", "/absolute", r"C:\absolute", "C:relative"):
            with self.subTest(unsafe=unsafe), self.assertRaisesRegex(
                    ValueError, "basename|path separators"):
                prepare_public_datasets.safe_path_component(unsafe, "clip ID")

    def test_public_dataset_prepared_child_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = os.path.join(temporary, "prepared")
            outside = os.path.join(temporary, "outside")
            os.makedirs(root)
            os.makedirs(outside)
            link = os.path.join(root, "clip")
            try:
                os.symlink(outside, link, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("directory symlinks are unavailable")
            with self.assertRaisesRegex(ValueError, "escapes dataset cache root"):
                prepare_public_datasets.confined_child(root, "clip", "clip ID")

    def test_public_dataset_manifest_rejects_escaping_clip_id(self):
        manifest = {
            "schema": 2,
            "prepared_suite": "extended-v3",
            "datasets": {"demo": {"archives": {}}},
            "clips": {"..": {
                "dataset": "demo", "archives": [], "content_type": "simulation",
            }},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
            path = stream.name
            json.dump(manifest, stream)
        try:
            with self.assertRaisesRegex(ValueError, "clip ID must be a non-empty basename"):
                prepare_public_datasets.load_manifest(path)
        finally:
            os.unlink(path)

    def test_public_dataset_metadata_refresh_is_authenticated_and_atomic(self):
        with tempfile.TemporaryDirectory() as root:
            prepared = os.path.join(root, "prepared")
            clip_dir = os.path.join(prepared, "clip")
            depth_dir = os.path.join(clip_dir, "gt_depth")
            os.makedirs(depth_dir)
            for frame_id in range(2):
                pixels = np.full((4, 6, 3), frame_id, np.uint8)
                Image.fromarray(pixels, "RGB").save(
                    os.path.join(clip_dir, f"frame_{frame_id:05d}.png"))
                Image.fromarray(pixels[..., 0], "L").save(
                    os.path.join(depth_dir, f"frame_{frame_id:05d}.png"))
            with open(os.path.join(clip_dir, "meta.json"), "w", encoding="utf-8") as stream:
                json.dump({
                    "name": "demo", "dataset": "Demo Dataset", "suite": "extended-v3",
                    "selection": [
                        {
                            "source_index": 4,
                            "rgb_timestamp": 1.0,
                            "depth_timestamp": 1.1,
                        },
                        {
                            "source_index": 5,
                            "rgb_timestamp": 2.0,
                            "depth_timestamp": 2.1,
                        },
                    ],
                    "legacy_annotation": "preserve me",
                }, stream)
            clip = {
                "dataset": "demo", "adapter": "tum_rgbd_zip", "archives": ["frames"],
                "name": "demo", "description": "fixture", "content_type": "real-capture",
                "start": 4, "stride": 1, "count": 2,
            }
            manifest = {
                "prepared_suite": "extended-v3",
                "datasets": {"demo": {
                    "title": "Demo Dataset", "homepage": "https://example.invalid",
                    "citation": "Fixture", "license_note": "Fixture only",
                    "archives": {"frames": {"filename": "unused.zip"}},
                }},
            }

            prepare_public_datasets.refresh_prepared_clip_metadata(
                manifest, "clip", clip, prepared)

            with open(os.path.join(clip_dir, "meta.json"), encoding="utf-8") as stream:
                refreshed = json.load(stream)
            self.assertEqual(refreshed["content_type"], "real-capture")
            self.assertEqual(refreshed["evaluation_role"], "ground-truth")
            self.assertEqual(refreshed["gt_depth_kind"], "metric")
            self.assertEqual(refreshed["legacy_annotation"], "preserve me")
            self.assertFalse(glob.glob(os.path.join(clip_dir, "meta.*.json.tmp")))

    def test_public_dataset_metadata_refresh_rejects_noncontiguous_frame_identities(self):
        with tempfile.TemporaryDirectory() as clip_dir:
            for frame_id in (0, 2):
                Image.new("RGB", (2, 2)).save(
                    os.path.join(clip_dir, f"frame_{frame_id:05d}.png"))
            with self.assertRaisesRegex(
                    RuntimeError, r"source frame identities.*missing=\[1\].*unexpected=\[2\]"):
                prepare_public_datasets.require_prepared_frame_ids(
                    "clip", clip_dir, range(2), "source frame", ".png")

    def test_public_dataset_metadata_refresh_rejects_stale_source_window(self):
        clip = {
            "adapter": "tum_rgbd_zip", "start": 4, "stride": 1, "count": 2,
        }
        selection = [
            {"source_index": 4, "rgb_timestamp": 1.0, "depth_timestamp": 1.1},
            {"source_index": 5, "rgb_timestamp": 2.0, "depth_timestamp": 2.1},
        ]
        self.assertIs(
            prepare_public_datasets.validate_prepared_selection(
                "clip", clip, selection),
            selection,
        )
        for field, changed in (("start", 5), ("stride", 2)):
            changed_clip = {**clip, field: changed}
            with self.subTest(field=field), self.assertRaisesRegex(
                    RuntimeError, r"selection row \d+ source_index=.*expected"):
                prepare_public_datasets.validate_prepared_selection(
                    "clip", changed_clip, selection)

    def test_public_dataset_selection_requires_exact_adapter_identity(self):
        clip = {
            "adapter": "sintel_stereo_zip", "start": 5, "stride": 1, "count": 2,
            "sequence": "ambush_4", "pass": "final",
        }
        selection = [
            {
                "source_index": 5, "dataset_frame": 6,
                "sequence": "ambush_4", "pass": "final",
            },
            {
                "source_index": 6, "dataset_frame": 7,
                "sequence": "ambush_4", "pass": "final",
            },
        ]
        prepare_public_datasets.validate_prepared_selection("clip", clip, selection)

        wrong_identity = [dict(row) for row in selection]
        wrong_identity[0]["sequence"] = "market_2"
        with self.assertRaisesRegex(RuntimeError, r"sequence='market_2'.*'ambush_4'"):
            prepare_public_datasets.validate_prepared_selection(
                "clip", clip, wrong_identity)

        missing_identity = [dict(row) for row in selection]
        missing_identity[0].pop("pass")
        with self.assertRaisesRegex(RuntimeError, r"fields do not match.*missing=\['pass'\]"):
            prepare_public_datasets.validate_prepared_selection(
                "clip", clip, missing_identity)

        extra_identity = [dict(row) for row in selection]
        extra_identity[0]["camera"] = "Camera_0"
        with self.assertRaisesRegex(RuntimeError, r"unexpected=\['camera'\]"):
            prepare_public_datasets.validate_prepared_selection(
                "clip", clip, extra_identity)

        tartan_clip = {
            "adapter": "tartanair_v2_zip", "start": 40, "stride": 1, "count": 1,
            "trajectory": "P000", "camera": "lcam_front",
        }
        tartan_row = prepare_public_datasets.make_selection_entry(
            tartan_clip, 40, dataset_frame=40)
        self.assertEqual(
            {key: tartan_row[key] for key in ("trajectory", "camera")},
            {"trajectory": "P000", "camera": "lcam_front"},
        )
        stale_tartan = {**tartan_row, "trajectory": "P002"}
        with self.assertRaisesRegex(RuntimeError, r"trajectory='P002'.*'P000'"):
            prepare_public_datasets.validate_prepared_selection(
                "clip", tartan_clip, [stale_tartan])

    def test_public_dataset_evidence_rejects_wrong_extension(self):
        with tempfile.TemporaryDirectory() as clip_dir:
            Image.new("RGB", (2, 2)).save(
                os.path.join(clip_dir, "frame_00000.jpg"))
            with self.assertRaisesRegex(
                    RuntimeError, r"wrong_extensions=\['frame_00000.jpg'\].*\.png"):
                prepare_public_datasets.require_prepared_frame_ids(
                    "clip", clip_dir, range(1), "source frame", ".png")

    def test_tartanair_flow_evidence_rejects_npy_instead_of_npz(self):
        with tempfile.TemporaryDirectory() as flow_dir:
            np.save(
                os.path.join(flow_dir, "frame_00001.npy"),
                np.zeros((2, 2, 2), np.float32),
            )
            with self.assertRaisesRegex(
                    RuntimeError, r"wrong_extensions=\['frame_00001.npy'\].*\.npz"):
                prepare_public_datasets.require_prepared_frame_ids(
                    "clip", flow_dir, range(1, 2), "flow evidence", ".npz")

    def test_public_dataset_evidence_rejects_directory_masquerading_as_frame(self):
        with tempfile.TemporaryDirectory() as clip_dir:
            os.makedirs(os.path.join(clip_dir, "frame_00000.png"))
            with self.assertRaisesRegex(
                    RuntimeError, r"non_regular=\['frame_00000.png'\]"):
                prepare_public_datasets.require_prepared_frame_ids(
                    "clip", clip_dir, range(1), "source frame", ".png")

    def test_tartanair_metadata_refresh_requires_every_flow_sidecar(self):
        with tempfile.TemporaryDirectory() as root:
            prepared = os.path.join(root, "prepared")
            clip_dir = os.path.join(prepared, "clip")
            depth_dir = os.path.join(clip_dir, "gt_depth")
            flow_dir = os.path.join(clip_dir, "gt_flow")
            os.makedirs(depth_dir)
            os.makedirs(flow_dir)
            for frame_id in range(3):
                Image.new("RGB", (2, 2)).save(
                    os.path.join(clip_dir, f"frame_{frame_id:05d}.png"))
                np.save(
                    os.path.join(depth_dir, f"frame_{frame_id:05d}.npy"),
                    np.ones((2, 2), np.float32))
            # A three-frame sequence requires flow sidecars 1 and 2. Leave 2 absent.
            np.savez_compressed(
                os.path.join(flow_dir, "frame_00001.npz"),
                flow=np.zeros((2, 2, 2), np.float32),
                valid=np.ones((2, 2), bool))
            with open(os.path.join(clip_dir, "meta.json"), "w", encoding="utf-8") as stream:
                json.dump({
                    "name": "demo", "dataset": "Demo Dataset", "suite": "extended-v3",
                    "selection": [
                        {
                            "source_index": i,
                            "dataset_frame": i,
                            "trajectory": "P000",
                            "camera": "lcam_front",
                        }
                        for i in range(3)
                    ],
                }, stream)
            clip = {
                "dataset": "demo", "adapter": "tartanair_v2_zip",
                "archives": ["frames"], "name": "demo", "description": "fixture",
                "trajectory": "P000", "camera": "lcam_front",
                "content_type": "simulation", "start": 0, "stride": 1, "count": 3,
            }
            manifest = {
                "prepared_suite": "extended-v3",
                "datasets": {"demo": {
                    "title": "Demo Dataset", "homepage": "https://example.invalid",
                    "citation": "Fixture", "license_note": "Fixture only",
                    "archives": {"frames": {"filename": "unused.zip"}},
                }},
            }

            with self.assertRaisesRegex(
                    RuntimeError, r"flow evidence identities.*missing=\[2\]"):
                prepare_public_datasets.refresh_prepared_clip_metadata(
                    manifest, "clip", clip, prepared)

    def test_range_dataset_manifest_requires_pinned_prepared_evidence(self):
        manifest = {
            "schema": 2,
            "prepared_suite": "extended-v3",
            "datasets": {"demo": {"archives": {
                "frames": {"access": "http_range_zip", "url": "https://example.invalid"},
            }}},
            "clips": {"clip": {
                "dataset": "demo", "archives": ["frames"], "content_type": "animation",
            }},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
            path = stream.name
            json.dump(manifest, stream)
        try:
            with self.assertRaisesRegex(RuntimeError, "prepared_evidence_sha256"):
                prepare_public_datasets.load_manifest(path)
            manifest["clips"]["clip"]["prepared_evidence_sha256"] = "a" * 64
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            self.assertEqual(
                prepare_public_datasets.load_manifest(path)["clips"]["clip"]
                ["prepared_evidence_sha256"], "a" * 64)
            manifest["clips"]["clip"]["content_type"] = "capture"
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            with self.assertRaisesRegex(RuntimeError, "content_type must be one of"):
                prepare_public_datasets.load_manifest(path)
        finally:
            os.unlink(path)

    def test_suite_defaults_keep_core_and_extended_baselines_separate(self):
        core_clips, core_baselines = run_eval.suite_defaults("core")
        extended_clips, extended_baselines = run_eval.suite_defaults("extended")
        self.assertTrue(core_clips.endswith(os.path.join("sbsbench", "clips")))
        self.assertTrue(core_baselines.endswith(os.path.join("sbsbench", "baselines")))
        self.assertIn(os.path.join("prepared", "extended-v4"), extended_clips)
        self.assertTrue(extended_baselines.endswith("baselines_extended"))

    def test_dataset_suite_revision_cannot_overwrite_evaluator_suite(self):
        published = run_eval.published_clip_metadata({
            "suite": "extended-v4", "dataset": "Example", "name": "clip",
        })
        self.assertNotIn("suite", published)
        self.assertEqual(published["source_suite"], "extended-v4")
        self.assertEqual({**{"suite": "extended"}, **published}["suite"], "extended")

    def test_rescore_uses_canonical_metric_contract_hash(self):
        data = {"meta": {"eval_schema": run_eval.EVAL_SCHEMA - 1}}
        runtime = {"python": "new", "numpy": "new", "pillow": "new"}
        with mock.patch.object(run_eval, "metric_contract_sha", return_value="canonical"), \
                mock.patch.object(run_eval, "label_contract_sha", return_value="labels"), \
                mock.patch.object(run_eval, "metric_runtime_provenance", return_value=runtime):
            rescore_run.refresh_contract_metadata(data)
        self.assertEqual(data["meta"]["metric_sha256"], "canonical")
        self.assertEqual(data["meta"]["label_contract_sha256"], "labels")
        self.assertEqual(data["meta"]["metric_runtime"], runtime)
        self.assertEqual(data["meta"]["eval_schema"], run_eval.EVAL_SCHEMA - 1)

    def test_sintel_adapter_preserves_left_and_rendered_right_frames(self):
        with tempfile.TemporaryDirectory() as root:
            archive = os.path.join(root, "sintel.zip")
            with zipfile.ZipFile(archive, "w") as zf:
                for i in range(3):
                    for eye, value in (("left", 40 + i), ("right", 80 + i)):
                        zf.writestr(f"training/final_{eye}/demo/frame_{i + 1:04d}.png",
                                    self.png_bytes(value))
                    zf.writestr(f"training/disparities/demo/frame_{i + 1:04d}.png",
                                self.png_bytes(10 + i))
                    zf.writestr(f"training/occlusions/demo/frame_{i + 1:04d}.png",
                                self.png_bytes(0, "L"))
                    zf.writestr(f"training/outofframe/demo/frame_{i + 1:04d}.png",
                                self.png_bytes(0, "L"))
            out = os.path.join(root, "out")
            os.makedirs(out)
            clip = {"adapter": "sintel_stereo_zip", "archives": ["stereo"],
                    "sequence": "demo", "pass": "final",
                    "start": 0, "stride": 1, "count": 2}
            rows = prepare_public_datasets.prepare_sintel(
                "demo", clip, {}, {"stereo": archive}, out, "test")
            self.assertEqual(len(rows), 2)
            self.assertTrue(os.path.exists(os.path.join(out, "frame_00000.png")))
            self.assertTrue(os.path.exists(os.path.join(out, "gt_right", "frame_00001.png")))
            self.assertTrue(os.path.exists(os.path.join(out, "gt_depth", "frame_00001.npy")))
            self.assertTrue(os.path.exists(
                os.path.join(out, "gt_depth_valid", "frame_00001.png")))
            self.assertTrue(os.path.exists(
                os.path.join(out, "gt_depth_valid_all", "frame_00001.png")))
            self.assertTrue(os.path.exists(
                os.path.join(out, "gt_depth_valid_nonocc", "frame_00001.png")))
            self.assertTrue(os.path.exists(
                os.path.join(out, "gt_occlusion", "frame_00001.png")))
            self.assertTrue(os.path.exists(
                os.path.join(out, "gt_outofframe", "frame_00001.png")))

            range_out = os.path.join(root, "range-out")
            os.makedirs(range_out)
            clip["prepared_evidence_sha256"] = (
                prepare_public_datasets.prepared_evidence_sha256(out))

            def memory_reader(url, expected_size):
                self.assertEqual(url, archive)
                self.assertEqual(expected_size, os.path.getsize(archive))
                with open(url, "rb") as archive_file:
                    return io.BytesIO(archive_file.read())

            remote = {"stereo": {
                "access": "http_range_zip", "url": archive,
                "size": os.path.getsize(archive),
            }}
            with mock.patch.object(
                    prepare_public_datasets, "HTTPRangeReader",
                    side_effect=memory_reader):
                range_rows = prepare_public_datasets.prepare_sintel(
                    "demo", clip, {}, remote, range_out, "test")
            self.assertEqual(rows, range_rows)
            self.assertTrue(os.path.exists(
                os.path.join(range_out, "gt_depth_valid", "frame_00001.png")))

            mismatch_out = os.path.join(root, "range-mismatch")
            os.makedirs(mismatch_out)
            bad_clip = {**clip, "prepared_evidence_sha256": "0" * 64}
            with mock.patch.object(
                    prepare_public_datasets, "HTTPRangeReader",
                    side_effect=memory_reader), self.assertRaisesRegex(
                        RuntimeError, "prepared evidence SHA-256 mismatch"):
                prepare_public_datasets.prepare_sintel(
                    "demo", bad_clip, {}, remote, mismatch_out, "test")

    def test_spring_adapter_range_selects_matching_stereo_frames(self):
        with tempfile.TemporaryDirectory() as root:
            archives = {}
            for side, value in (("left", 40), ("right", 80)):
                path = os.path.join(root, side + ".zip")
                with zipfile.ZipFile(path, "w") as zf:
                    for i in range(3):
                        zf.writestr(f"spring/test/0003/frame_{side}/frame_{side}_{i + 1:04d}.png",
                                    self.png_bytes(value + i))
                archives["test_" + side] = {"url": path, "size": os.path.getsize(path)}
            out = os.path.join(root, "out")
            os.makedirs(out)
            clip = {
                "adapter": "spring_http_range_zip", "sequence": "0003",
                "start": 1, "stride": 1, "count": 2,
            }

            def memory_reader(url, expected_size):
                with open(url, "rb") as archive_file:
                    return io.BytesIO(archive_file.read())

            bootstrap = os.path.join(root, "bootstrap")
            os.makedirs(bootstrap)
            with mock.patch.object(
                    prepare_public_datasets, "HTTPRangeReader",
                    side_effect=memory_reader), mock.patch.object(
                        prepare_public_datasets, "authenticate_prepared_range_evidence"):
                prepare_public_datasets.prepare_spring(
                    "demo", clip, {}, archives, bootstrap, "test")
            clip["prepared_evidence_sha256"] = (
                prepare_public_datasets.prepared_evidence_sha256(bootstrap))

            with mock.patch.object(
                    prepare_public_datasets, "HTTPRangeReader",
                    side_effect=memory_reader):
                rows = prepare_public_datasets.prepare_spring(
                    "demo", clip, {}, archives, out, "test")
            self.assertEqual([row["dataset_frame"] for row in rows], [2, 3])
            self.assertTrue(os.path.exists(os.path.join(out, "frame_00000.png")))
            self.assertTrue(os.path.exists(os.path.join(out, "gt_right", "frame_00001.png")))

    def test_vkitti_adapter_selects_matching_rgb_and_depth(self):
        with tempfile.TemporaryDirectory() as root:
            archives = {}
            for modality in ("rgb", "depth"):
                path = os.path.join(root, modality + ".tar")
                archives[modality] = path
                with tarfile.open(path, "w") as tf:
                    for i in range(3):
                        suffix = f"rgb_{i:05d}.jpg" if modality == "rgb" else f"depth_{i:05d}.png"
                        folder = "rgb" if modality == "rgb" else "depth"
                        data = self.png_bytes(50 + i, "RGB" if modality == "rgb" else "L")
                        info = tarfile.TarInfo(
                            f"vkitti/{'Scene01'}/clone/frames/{folder}/Camera_0/{suffix}")
                        info.size = len(data)
                        tf.addfile(info, io.BytesIO(data))
            out = os.path.join(root, "out")
            os.makedirs(out)
            clip = {"adapter": "vkitti2_tar", "scene": "Scene01",
                    "variant": "clone", "camera": "Camera_0",
                    "start": 1, "stride": 1, "count": 2}
            rows = prepare_public_datasets.prepare_vkitti2(
                "demo", clip, {}, archives, out, "test")
            self.assertEqual([r["dataset_frame"] for r in rows], [1, 2])
            self.assertTrue(os.path.exists(os.path.join(out, "frame_00000.png")))
            self.assertTrue(os.path.exists(os.path.join(out, "gt_depth", "frame_00001.png")))


if __name__ == "__main__":
    unittest.main()
