#!/usr/bin/env python3

import json
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import generate_artistic_depth_run as generate


class ArtisticDepthRunTests(unittest.TestCase):
    @staticmethod
    def write_frames(root, frame_ids):
        root.mkdir(parents=True, exist_ok=True)
        for frame_id in frame_ids:
            (root / f"frame_{frame_id:05d}.png").write_bytes(
                f"rgb-{frame_id}".encode("ascii")
            )

    @staticmethod
    def write_model_assets(executable, model="model", engine=b"engine", onnx=b"onnx"):
        assets = executable.parent / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        recipe = generate.selected_depth_engine_recipe()
        engine_path = assets / f"{model}.{recipe}.engine"
        engine_path.write_bytes(engine)
        if onnx is not None:
            (assets / f"{model}.onnx").write_bytes(onnx)
        return engine_path

    @staticmethod
    def write_depth_artifacts(root, payload=b"baseline"):
        root.mkdir(parents=True, exist_ok=True)
        (root / "contract.json").write_text("{}", encoding="utf-8")
        (root / "generation_identity.json").write_text("{}", encoding="utf-8")
        (root / "depth_00000.png").write_bytes(b"depth")
        (root / "baseline_disparity_00000.f32").write_bytes(payload)
        (root / "baseline_unclamped_disparity_00000.f32").write_bytes(
            b"unclamped"
        )

    def test_sparse_labels_include_preceding_temporal_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory)
            self.write_frames(clip, range(6))
            (clip / "label_frames.json").write_text(json.dumps({
                "schema": 1, "frame_ids": [0, 3, 5],
            }), encoding="utf-8")
            selection = generate.output_selection(clip)
            self.assertEqual(selection["mode"], "label-frames")
            self.assertEqual(selection["label_frame_ids"], [0, 3, 5])
            self.assertEqual(selection["output_frame_ids"], [0, 1, 2, 3, 4, 5])
            self.assertEqual(len(selection["label_frames_sha256"]), 64)

    def test_sparse_label_without_adjacent_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory)
            self.write_frames(clip, [0, 2])
            (clip / "label_frames.json").write_text(json.dumps({
                "schema": 1, "frame_ids": [0],
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "no consecutive"):
                generate.output_selection(clip)

    def test_clip_without_manifest_keeps_all_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory)
            self.write_frames(clip, [0, 1, 2])
            self.assertEqual(generate.output_selection(clip), {
                "mode": "interval",
                "label_frame_ids": [],
                "output_frame_ids": [0, 1, 2],
                "label_frames_sha256": "",
            })

    def test_native_hdr_manifest_is_not_misclassified_as_a_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory)
            self.write_frames(clip, [0, 1])
            (clip / generate.native_hdr_capture.MANIFEST_NAME).write_text(
                "{}", encoding="utf-8"
            )
            self.assertEqual(set(generate.source_frame_files(clip)), {0, 1})
            (clip / "frame_unexpected.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid frame_"):
                generate.source_frame_files(clip)

    def test_explicit_gt_right_selection_uses_exact_source_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory)
            self.write_frames(clip, [0, 1, 2, 3])
            self.write_frames(clip / "gt_right", [0, 3])
            self.assertEqual(generate.output_selection(clip, True), {
                "mode": "gt-right",
                "label_frame_ids": [],
                "output_frame_ids": [0, 3],
                "label_frames_sha256": "",
            })
            gt_selection = generate.output_selection(clip, True)
            identity = generate.generation_identity(
                {
                    "source_identity_method":
                        generate.SOURCE_IDENTITY_FINGERPRINT,
                    "source_identity_value": "a" * 64,
                },
                gt_selection, "e" * 64, "c" * 16, "model",
            )
            self.assertEqual(identity["output_selection"], gt_selection)
            self.assertEqual(
                generate.output_selection(clip)["output_frame_ids"],
                [0, 1, 2, 3],
            )

    def test_explicit_gt_right_selection_rejects_missing_or_invalid_gt(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory)
            self.write_frames(clip, [0, 1])
            with self.assertRaisesRegex(RuntimeError, "missing GT-right"):
                generate.output_selection(clip, True)

            gt_right = clip / "gt_right"
            self.write_frames(gt_right, [2])
            with self.assertRaisesRegex(RuntimeError, "missing source RGB"):
                generate.output_selection(clip, True)

            (gt_right / "frame_00002.png").unlink()
            (gt_right / "frame_invalid.png").write_bytes(b"invalid")
            with self.assertRaisesRegex(RuntimeError, "invalid frame_"):
                generate.output_selection(clip, True)

    def test_explicit_gt_right_selection_rejects_label_manifest_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory)
            self.write_frames(clip, [0, 1])
            self.write_frames(clip / "gt_right", [0])
            (clip / "label_frames.json").write_text(json.dumps({
                "schema": 1, "frame_ids": [0],
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "cannot override"):
                generate.output_selection(clip, True)

    def test_frozen_clip_manifest_supplies_cheap_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            clip = suite / "shot"
            self.write_frames(clip, [0, 1])
            manifest, manifest_path = generate.clip_hashes.build_and_write(
                suite, workers=1
            )
            with mock.patch.object(
                    generate, "source_fingerprint",
                    side_effect=AssertionError("full fallback must not run")):
                identities, provenance = generate.resolve_source_identities(
                    suite, ["shot"]
                )
            self.assertEqual(
                identities["shot"]["source_identity_value"],
                manifest["clips"]["shot"]["clip_sha1"],
            )
            self.assertEqual(
                identities["shot"]["source_identity_method"],
                generate.SOURCE_IDENTITY_MANIFEST,
            )
            self.assertEqual(provenance["clip_hash_source"], "manifest")
            self.assertEqual(provenance["clip_hash_verification"], "stat")
            self.assertEqual(
                provenance["clip_hash_manifest_content_sha256"],
                manifest[generate.clip_hashes.MANIFEST_CONTENT_SHA256_FIELD],
            )
            self.assertEqual(
                provenance["clip_hash_manifest_file_sha256"],
                generate.sha256(manifest_path),
            )

    def test_rebuilt_unchanged_manifest_preserves_depth_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            clip = suite / "shot"
            self.write_frames(clip, [0, 1])
            _manifest, manifest_path = generate.clip_hashes.build_and_write(
                suite, workers=1
            )
            first, first_provenance = generate.resolve_source_identities(
                suite, ["shot"]
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["created_utc"] = "2099-01-01T00:00:00+00:00"
            generate.clip_hashes.write_manifest_atomic(payload, manifest_path)
            second, second_provenance = generate.resolve_source_identities(
                suite, ["shot"]
            )
            self.assertEqual(first, second)
            self.assertEqual(
                first_provenance["clip_hash_manifest_content_sha256"],
                second_provenance["clip_hash_manifest_content_sha256"],
            )
            self.assertNotEqual(
                first_provenance["clip_hash_manifest_file_sha256"],
                second_provenance["clip_hash_manifest_file_sha256"],
            )
            selection = generate.output_selection(clip)
            identity = generate.generation_identity(
                first["shot"], selection, "e" * 64, "c" * 16, "model"
            )
            self.assertTrue(generate.source_identity_matches(
                identity, second["shot"]
            ))

    def test_rebuilt_changed_manifest_invalidates_depth_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            clip = suite / "shot"
            self.write_frames(clip, [0, 1])
            generate.clip_hashes.build_and_write(suite, workers=1)
            first, _provenance = generate.resolve_source_identities(
                suite, ["shot"]
            )
            selection = generate.output_selection(clip)
            cached = generate.generation_identity(
                first["shot"], selection, "e" * 64, "c" * 16, "model"
            )

            (clip / "frame_00000.png").write_bytes(b"changed-source")
            generate.clip_hashes.build_and_write(suite, workers=1)
            second, _provenance = generate.resolve_source_identities(
                suite, ["shot"]
            )
            self.assertNotEqual(first, second)
            self.assertFalse(generate.source_identity_matches(
                cached, second["shot"]
            ))

    def test_present_but_stale_clip_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            clip = suite / "shot"
            self.write_frames(clip, [0])
            generate.clip_hashes.build_and_write(suite, workers=1)
            (clip / "frame_00000.png").write_bytes(b"changed-size")
            with self.assertRaisesRegex(RuntimeError, "stale clip hash manifest"):
                generate.resolve_source_identities(suite, ["shot"])

    def test_full_clip_hash_verification_catches_same_stat_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            clip = suite / "shot"
            self.write_frames(clip, [0])
            generate.clip_hashes.build_and_write(suite, workers=1)
            frame = clip / "frame_00000.png"
            previous = frame.stat()
            frame.write_bytes(b"RGB-0")
            os.utime(frame, ns=(previous.st_atime_ns, previous.st_mtime_ns))
            identities, provenance = generate.resolve_source_identities(
                suite, ["shot"], verify_clip_hashes=False
            )
            self.assertEqual(identities["shot"]["source_identity_method"],
                             generate.SOURCE_IDENTITY_MANIFEST)
            self.assertEqual(provenance["clip_hash_verification"], "stat")
            with self.assertRaisesRegex(RuntimeError, "content hash changed"):
                generate.resolve_source_identities(
                    suite, ["shot"], verify_clip_hashes=True
                )

    def test_missing_manifest_preserves_full_source_fingerprint_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            clip = suite / "shot"
            self.write_frames(clip, [0])
            with mock.patch.object(
                    generate, "source_fingerprint", return_value="f" * 64) as fingerprint:
                identities, provenance = generate.resolve_source_identities(
                    suite, ["shot"], verify_clip_hashes=True
                )
            fingerprint.assert_called_once_with(clip)
            self.assertEqual(identities["shot"], {
                "source_identity_method": generate.SOURCE_IDENTITY_FINGERPRINT,
                "source_identity_value": "f" * 64,
            })
            self.assertEqual(provenance["clip_hash_source"], "direct")
            self.assertEqual(provenance["clip_hash_verification"], "direct-content")

    def test_generation_identity_versions_manifest_and_fallback_sources(self):
        selection = {
            "mode": "interval", "label_frame_ids": [],
            "output_frame_ids": [0], "label_frames_sha256": "",
        }
        frozen = {
            "source_identity_method": generate.SOURCE_IDENTITY_MANIFEST,
            "source_identity_value": "a" * 12,
            "clip_hash_manifest_content_sha256": "b" * 64,
        }
        frozen_payload = generate.generation_identity(
            frozen, selection, "c" * 64, "d" * 16, "model"
        )
        self.assertEqual(frozen_payload["schema"], 5)
        self.assertEqual(
            frozen_payload["input_variant"],
            generate.input_color.sdr_input_variant(),
        )
        self.assertTrue(generate.source_identity_matches(frozen_payload, frozen))

        fallback = {
            "source_identity_method": generate.SOURCE_IDENTITY_FINGERPRINT,
            "source_identity_value": "e" * 64,
        }
        fallback_payload = generate.generation_identity(
            fallback, selection, "c" * 64, "d" * 16, "model"
        )
        self.assertEqual(fallback_payload["schema"], 5)
        self.assertEqual(fallback_payload["source_sha256"], "e" * 64)
        self.assertTrue(generate.source_identity_matches(fallback_payload, fallback))

    def test_selected_model_asset_identity_changes_with_plan_or_onnx(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "sunshine.exe"
            executable.write_bytes(b"exe")
            engine = self.write_model_assets(executable)
            first = generate.selected_depth_model_identity(executable, "model")
            self.assertEqual(
                first["contract"], generate.MODEL_ASSET_IDENTITY_CONTRACT
            )
            self.assertEqual(first["engine_sha256"], generate.sha256(engine))
            self.assertEqual(first["onnx_sha256"], generate.sha256(
                executable.parent / "assets" / "model.onnx"
            ))

            engine.write_bytes(b"changed-engine")
            second = generate.selected_depth_model_identity(executable, "model")
            self.assertNotEqual(first, second)

            (executable.parent / "assets" / "model.onnx").write_bytes(
                b"changed-onnx"
            )
            third = generate.selected_depth_model_identity(executable, "model")
            self.assertNotEqual(second, third)

    def test_reusable_rows_require_exact_model_assets_and_artifact_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "sunshine.exe"
            executable.write_bytes(b"exe")
            engine = self.write_model_assets(executable)
            model_identity = generate.selected_depth_model_identity(
                executable, "model"
            )
            clip = root / "output" / "shot"
            self.write_depth_artifacts(clip)
            artifact_identity = generate.depth_artifact_identity(clip)
            manifest_path = root / "output" / "depth_run_manifest.json"
            generate.write_json_atomic(manifest_path, {
                "schema": generate.DEPTH_RUN_MANIFEST_SCHEMA,
                "model_asset_identity": model_identity,
                "model_asset_identity_sha256":
                    generate.clip_hashes.canonical_json_sha256(model_identity),
                "input_variant": generate.input_color.sdr_input_variant(),
                "input_variant_sha256": generate.input_color.input_variant_sha256(
                    generate.input_color.sdr_input_variant()
                    ),
                "metric_preview_encoding": "native-srgb-v1",
                "clips": [{"clip": "shot", **artifact_identity}],
            })

            rows = generate.reusable_artifact_rows(
                manifest_path, model_identity
            )
            self.assertTrue(generate.artifact_identity_matches(
                rows["shot"], generate.depth_artifact_identity(clip)
            ))

            (clip / "baseline_disparity_00000.f32").write_bytes(b"corrupt!")
            self.assertFalse(generate.artifact_identity_matches(
                rows["shot"], generate.depth_artifact_identity(clip)
            ))

            engine.write_bytes(b"replacement-plan")
            replacement = generate.selected_depth_model_identity(
                executable, "model"
            )
            self.assertEqual(
                generate.reusable_artifact_rows(manifest_path, replacement), {}
            )

    def test_gt_right_generation_requires_stereo_manifest_and_passes_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite"
            clip = suite / "shot"
            self.write_frames(clip, [0])
            self.write_frames(clip / "gt_right", [0])
            manifest_path = suite / "dataset_manifest.json"
            manifest_path.write_text(json.dumps({
                "schema": 2,
                "source_kind": "mono-video",
                "sequences": [{"clip": "shot"}],
            }), encoding="utf-8")
            executable = root / "sunshine.exe"
            executable.write_bytes(b"exe")
            self.write_model_assets(executable)
            conf = root / "bench.conf"
            conf.write_text("conf", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "authored-stereo"):
                generate.generate(
                    suite, root / "out", executable, conf, "model", 10,
                    False, output_gt_right_only=True,
                )

            manifest_path.write_text(json.dumps({
                "schema": 1,
                "layout": "side-by-side",
                "eye_order": "first-left",
                "sequences": [{"clip": "shot"}],
            }), encoding="utf-8")
            failed = mock.Mock(returncode=1, stdout="", stderr="failure")
            with mock.patch.object(
                    generate.subprocess, "run", return_value=failed) as run:
                with self.assertRaisesRegex(RuntimeError, "harness failed"):
                    generate.generate(
                        suite, root / "out", executable, conf, "model", 10,
                        False, output_gt_right_only=True,
                    )
            command = run.call_args.args[0]
            self.assertIn("--output-gt-right-only", command)
            self.assertNotIn("--output-label-frames", command)

    def test_sequence_paths_reject_traversal_and_duplicates_before_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite"
            output = root / "output"
            self.write_frames(suite / "shot", [0])
            with self.assertRaisesRegex(RuntimeError, "unsafe dataset clip name"):
                generate.validate_sequence_paths(
                    suite, output, [{"clip": "../outside"}]
                )
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(RuntimeError, "duplicate dataset clip name"):
                generate.validate_sequence_paths(
                    suite, output, [{"clip": "shot"}, {"clip": "shot"}]
                )
            with self.assertRaisesRegex(RuntimeError, "paths overlap"):
                generate.validate_sequence_paths(
                    suite, suite, [{"clip": "shot"}]
                )
            for unsafe_output in (suite / "generated", root):
                with self.subTest(output=unsafe_output):
                    with self.assertRaisesRegex(RuntimeError, "paths overlap"):
                        generate.validate_sequence_paths(
                            suite, unsafe_output, [{"clip": "shot"}]
                        )
                    self.assertTrue((suite / "shot" / "frame_00000.png").is_file())

    def test_source_fingerprint_supports_exact_png_jpg_and_jpeg_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "frame_00000.png").write_bytes(b"png")
            (source / "frame_00001.jpg").write_bytes(b"jpg")
            (source / "frame_00002.jpeg").write_bytes(b"jpeg")
            self.assertEqual(set(generate.source_frame_files(source)), {0, 1, 2})
            self.assertEqual(len(generate.source_fingerprint(source)), 64)
            self.assertEqual(
                generate.output_selection(source)["output_frame_ids"], [0, 1, 2]
            )
            (source / "frame_00001.png").write_bytes(b"duplicate")
            with self.assertRaisesRegex(RuntimeError, "duplicate numeric"):
                generate.source_fingerprint(source)

    def test_completed_clip_rejects_wrong_artifact_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            self.write_frames(source, [0])
            output.mkdir()
            selection = generate.output_selection(source)
            source_identity = {
                "source_identity_method": generate.SOURCE_IDENTITY_FINGERPRINT,
                "source_identity_value": generate.source_fingerprint(source),
            }
            identity = generate.generation_identity(
                source_identity, selection, "e" * 64, "c" * 16, "model"
            )
            (output / "generation_identity.json").write_text(
                json.dumps(identity), encoding="utf-8"
            )
            contract = {
                "schema": generate.harness_contract.HARNESS_SCHEMA,
                "model": "model",
                "artifact_mode": "depth+baseline-disparity",
                "depth_step": "current-once",
                "artistic_policy": False,
                "artistic_policy_consumed": False,
                "artistic_policy_authorization": "none",
                "model_onnx_sha256": "",
                "policy_metadata_sha256": "",
                "deployment_geometry_allowlist_sha256": "",
                "artistic_scale_override": 0.0,
                "color_mode": "sdr-srgb-8bit",
                "hdr_source_kind":
                    generate.harness_contract.HDR_SOURCE_SDR,
                "metric_preview_encoding": "native-srgb-v1",
                "hdr_input_scale": 0.0,
                "sdr_white_level_raw": 0,
                "warp_disparity": (
                    "exact_clamped_full_binocular_normalized_at_output_eye_raster_zero_bars"
                ),
                "warp_unclamped_disparity": (
                    "unclamped_full_binocular_normalized_at_artistic_scale_1_"
                    "output_eye_raster_zero_bars"
                ),
                "artistic_disparity_contract": (
                    "clamp(raw_baseline_times_scale_to_plus_or_minus_0.142_"
                    "times_aspect_scale_times_content_scale_x)"
                ),
                "source_width": 16,
                "source_height": 16,
                "eye_width": 16,
                "eye_height": 16,
                "disparity_raster_width": 16,
                "disparity_raster_height": 16,
                "policy_warp_source_sha256": "w" * 64,
                "artistic_full_clamp_abs": 0.1,
                "output_selection_mode": selection["mode"],
                "label_frame_ids": selection["label_frame_ids"],
                "output_selected_frame_ids": selection["output_frame_ids"],
                "output_label_frames_sha256": selection["label_frames_sha256"],
            }
            (output / "contract.json").write_text(
                json.dumps(contract), encoding="utf-8"
            )
            (output / "depth_00000.png").write_bytes(b"depth")
            (output / "baseline_disparity_00000.f32").write_bytes(b"exact")
            (output / "baseline_unclamped_disparity_00000.f32").write_bytes(
                b"unclamped"
            )
            self.assertTrue(generate.valid_completed_clip(
                source, output, "model", "e" * 64, "c" * 16, source_identity
            ))
            (output / "depth_00000.png").rename(output / "depth_00000.jpg")
            self.assertFalse(generate.valid_completed_clip(
                source, output, "model", "e" * 64, "c" * 16, source_identity
            ))

    def test_failed_staging_keeps_old_published_run_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite"
            self.write_frames(suite / "shot", [0])
            (suite / "dataset_manifest.json").write_text(json.dumps({
                "sequences": [{"clip": "shot"}],
            }), encoding="utf-8")
            output = root / "output"
            old_clip = output / "shot"
            old_clip.mkdir(parents=True)
            old_marker = old_clip / "old.txt"
            old_marker.write_text("old", encoding="utf-8")
            old_manifest = output / "depth_run_manifest.json"
            old_manifest.write_text("old-manifest", encoding="utf-8")
            executable = root / "sunshine.exe"
            conf = root / "bench.conf"
            executable.write_bytes(b"exe")
            self.write_model_assets(executable)
            conf.write_text("conf", encoding="utf-8")

            failed = mock.Mock(returncode=1, stdout="", stderr="failure")
            with mock.patch.object(generate.subprocess, "run", return_value=failed):
                with self.assertRaisesRegex(RuntimeError, "harness failed"):
                    generate.generate(
                        suite, output, executable, conf, "model", 10, False
                    )

            self.assertEqual(old_marker.read_text(encoding="utf-8"), "old")
            self.assertEqual(
                old_manifest.read_text(encoding="utf-8"), "old-manifest"
            )
            self.assertFalse(any(
                path.name.startswith(".shot.partial-")
                for path in output.iterdir()
            ))

    def test_partial_regeneration_removes_manifest_before_mixed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite"
            for name in ("a", "b"):
                self.write_frames(suite / name, [0])
            (suite / "dataset_manifest.json").write_text(json.dumps({
                "sequences": [{"clip": "a"}, {"clip": "b"}],
            }), encoding="utf-8")
            output = root / "output"
            for name in ("a", "b"):
                destination = output / name
                destination.mkdir(parents=True)
                (destination / "old.txt").write_text("old", encoding="utf-8")
            manifest = output / "depth_run_manifest.json"
            manifest.write_text("old-manifest", encoding="utf-8")
            executable = root / "sunshine.exe"
            conf = root / "bench.conf"
            executable.write_bytes(b"exe")
            self.write_model_assets(executable)
            conf.write_text("conf", encoding="utf-8")
            invocation = 0

            def fake_run(command, **_kwargs):
                nonlocal invocation
                invocation += 1
                staging = Path(command[command.index("--out") + 1])
                if invocation == 1:
                    (staging / "contract.json").write_text(json.dumps({
                        "policy_warp_source_sha256": "p" * 64,
                        "metric_sha256": "m" * 16,
                        "metric_preview_encoding": "native-srgb-v1",
                    }), encoding="utf-8")
                    return mock.Mock(returncode=0, stdout="", stderr="")
                return mock.Mock(returncode=1, stdout="", stderr="failure")

            def completed(_source, candidate, *_args, **_kwargs):
                return (
                    candidate.name.startswith(".a.partial-") and
                    (candidate / "contract.json").is_file()
                )

            with mock.patch.object(
                    generate.subprocess, "run", side_effect=fake_run), \
                    mock.patch.object(
                        generate, "valid_completed_clip", side_effect=completed
                    ):
                with self.assertRaisesRegex(RuntimeError, "harness failed"):
                    generate.generate(
                        suite, output, executable, conf, "model", 10, False
                    )

            self.assertFalse(manifest.exists())
            self.assertFalse((output / "a" / "old.txt").exists())
            self.assertTrue((output / "a" / "contract.json").is_file())
            self.assertEqual(
                (output / "b" / "old.txt").read_text(encoding="utf-8"), "old"
            )
            self.assertFalse(any(
                path.name.startswith((".a.partial-", ".b.partial-"))
                for path in output.iterdir()
            ))


if __name__ == "__main__":
    unittest.main()
