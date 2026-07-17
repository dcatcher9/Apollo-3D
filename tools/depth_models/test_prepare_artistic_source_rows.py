import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import prepare_artistic_source_rows as source_rows  # noqa: E402


def write_float_texture(path, width=2, height=2):
    header = np.array([width, height], dtype="<u4").tobytes()
    values = np.linspace(-0.02, 0.02, width * height, dtype="<f4").tobytes()
    path.write_bytes(header + values)


def write_native_hdr_manifest(clip, frame_ids, width=2, height=2):
    model_root = clip / source_rows.native_hdr_capture.MODEL_SOURCE_DIRECTORY
    model_root.mkdir()
    frame_rows = []
    semantic_rows = []
    for frame_id in frame_ids:
        model_path = model_root / f"frame_{frame_id:05d}.scrgb16"
        pixels = np.full(
            (height, width, 4), frame_id / 16.0, dtype="<f2"
        )
        model_path.write_bytes(pixels.tobytes())
        stat = model_path.stat()
        preview_path = clip / f"frame_{frame_id:05d}.png"
        timestamp = frame_id / 30.0
        row = {
            "frame": frame_id,
            "path": model_path.relative_to(clip).as_posix(),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": source_rows.sha256(model_path),
            "preview": preview_path.relative_to(clip).as_posix(),
            "preview_sha256": source_rows.sha256(preview_path),
            "timestamp_seconds": timestamp,
            "stats": {},
        }
        frame_rows.append(row)
        semantic_rows.append({
            key: row[key] for key in (
                "frame", "path", "size", "sha256", "preview",
                "preview_sha256", "timestamp_seconds",
            )
        })
    source_video = {"sha256": "f" * 64}
    conversion = {"contract_sha256": "d" * 64}
    semantic = {
        "contract": source_rows.native_hdr_capture.MANIFEST_CONTRACT,
        "capture_encoding": source_rows.native_hdr_capture.CAPTURE_ENCODING,
        "preview_encoding": source_rows.native_hdr_capture.PREVIEW_ENCODING,
        "width": width,
        "height": height,
        "row_pitch_bytes": width * 8,
        "source_video": source_video,
        "conversion": conversion,
        "frames": semantic_rows,
    }
    payload = {
        "schema": source_rows.native_hdr_capture.MANIFEST_SCHEMA,
        **semantic,
        "frames": frame_rows,
        "frame_count": len(frame_rows),
        "content_sha256": source_rows.native_hdr_capture.canonical_sha256(
            semantic
        ),
    }
    path = clip / source_rows.native_hdr_capture.MANIFEST_NAME
    path.write_bytes(
        source_rows.native_hdr_capture.canonical_json_bytes(payload)
    )
    return payload


def depth_contract(label_ids=(1, 2), selected_ids=(0, 1, 2)):
    return {
        "schema": source_rows.HARNESS_SCHEMA,
        "model": "model",
        "artifact_mode": "depth+baseline-disparity",
        "depth_step": "current-once",
        "depth_reuse_interval": 1,
        "depth_compensation": "none",
        "depth_override_frames": 0,
        "artistic_policy": False,
        "artistic_policy_consumed": False,
        "artistic_policy_authorization": "none",
        "model_onnx_sha256": "",
        "policy_metadata_sha256": "",
        "deployment_geometry_allowlist_sha256": "",
        "artistic_scale_override": 0.0,
        "color_mode": "sdr-srgb-8bit",
        "metric_preview_encoding": "native-srgb-v1",
        "hdr_source_kind": "native-sdr",
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
        "source_width": 2,
        "source_height": 2,
        "model_input_width": 14,
        "model_input_height": 14,
        "eye_width": 2,
        "eye_height": 2,
        "disparity_raster_width": 2,
        "disparity_raster_height": 2,
        "content_scale_x": 1.0,
        "content_scale_y": 1.0,
        "artistic_full_clamp_abs": 0.04,
        "policy_warp_source_sha256": "a" * 64,
        "metric_sha256": "b" * 16,
        "output_selection_mode": "label-frames",
        "label_frame_ids": list(label_ids),
        "output_selected_frame_ids": list(selected_ids),
        "output_label_frames_sha256": "",
    }


class ArtisticSourceRowsTests(unittest.TestCase):
    def make_fixture(self, root, source_ids=(0, 1, 2), label_ids=(1, 2),
                     selected_ids=None, source_suffix=".png",
                     input_variant=None, native_dimensions=(2, 2),
                     native_frame_ids=None):
        input_variant = (
            input_variant or source_rows.input_color.sdr_input_variant()
        )
        is_native_hdr = (
            input_variant["kind"] ==
            source_rows.input_color.INPUT_KIND_NATIVE_PQ
        )
        if is_native_hdr and source_suffix != ".png":
            raise ValueError("native-HDR fixtures require PNG previews")
        source_kind = "native-hdr-video" if is_native_hdr else "mono-video"
        clips = root / "clips"
        run = root / "run"
        clip = clips / "shot"
        run_clip = run / "shot"
        clip.mkdir(parents=True)
        run_clip.mkdir(parents=True)
        if selected_ids is None:
            selected_ids = source_rows.evidence_frame_ids(label_ids, source_ids)
        for frame_id in source_ids:
            Image.new("RGB", (2, 2), (frame_id, 2, 3)).save(
                clip / f"frame_{frame_id:05d}{source_suffix}"
            )
        for frame_id in selected_ids:
            Image.new("L", (2, 2), frame_id).save(
                run_clip / f"depth_{frame_id:05d}.png"
            )
            write_float_texture(
                run_clip / f"baseline_disparity_{frame_id:05d}.f32"
            )
            write_float_texture(
                run_clip / f"baseline_unclamped_disparity_{frame_id:05d}.f32"
            )
        label_frames = clip / "label_frames.json"
        label_frames.write_text(json.dumps({
            "schema": 1,
            "frame_ids": list(label_ids),
        }), encoding="utf-8")
        (clip / "meta.json").write_text(json.dumps({
            "split": "training",
            "film_id": "film",
            "dataset": "fixture",
            "source_kind": source_kind,
            "global_policy_weight": 1.0,
        }), encoding="utf-8")
        dataset_manifest = clips / "dataset_manifest.json"
        dataset_manifest.write_text(json.dumps({
            "schema": 2,
            "dataset": "fixture",
            "domain": "fixture",
            "production_id": "film",
            "source_kind": source_kind,
            "split": "training",
            "global_policy_weight": 1.0,
            "license": "test-only",
            "policy_role": "cinematic_training",
            "sequences": [{"clip": "shot"}],
        }), encoding="utf-8")
        contract = depth_contract(label_ids, selected_ids)
        contract.update({
            "color_mode": input_variant["color_mode"],
            "metric_preview_encoding": (
                source_rows.input_variant_metric_preview_encoding(
                    input_variant
                )
            ),
            "hdr_source_kind": source_rows.input_variant_hdr_source_kind(
                input_variant
            ),
            "hdr_input_scale": float(input_variant["scrgb_white_scale"] or 0.0),
            "sdr_white_level_raw": int(
                input_variant["windows_sdr_white_level_raw"] or 0
            ),
        })
        contract["output_label_frames_sha256"] = source_rows.sha256(label_frames)
        (run_clip / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
        selection = {
            "mode": "label-frames",
            "label_frame_ids": list(label_ids),
            "output_frame_ids": list(selected_ids),
            "label_frames_sha256": source_rows.sha256(label_frames),
        }
        if is_native_hdr:
            write_native_hdr_manifest(
                clip, source_ids if native_frame_ids is None else native_frame_ids,
                width=native_dimensions[0], height=native_dimensions[1],
            )
            frozen, frozen_path = (
                source_rows.depth_run.clip_hashes.build_and_write(
                    clips, clips=["shot"], workers=1
                )
            )
            frozen_content = frozen[
                source_rows.depth_run.clip_hashes.MANIFEST_CONTENT_SHA256_FIELD
            ]
            source_identity = {
                "source_identity_method": (
                    source_rows.depth_run.SOURCE_IDENTITY_MANIFEST
                ),
                "source_identity_value": frozen["clips"]["shot"]["clip_sha1"],
                "clip_hash_manifest_content_sha256": frozen_content,
            }
            clip_hash_provenance = {
                "clip_hash_source": "manifest",
                "clip_hash_verification": "full",
                "clip_hash_manifest": str(frozen_path.resolve()),
                "clip_hash_manifest_content_sha256": frozen_content,
                "clip_hash_manifest_file_sha256": source_rows.sha256(
                    frozen_path
                ),
            }
        else:
            source_identity = {
                "source_identity_method": (
                    source_rows.depth_run.SOURCE_IDENTITY_FINGERPRINT
                ),
                "source_identity_value": source_rows.depth_run.source_fingerprint(
                    clip
                ),
            }
            clip_hash_provenance = {
                "clip_hash_source": "direct",
                "clip_hash_verification": "direct-content",
                "clip_hash_manifest": None,
                "clip_hash_manifest_content_sha256": None,
                "clip_hash_manifest_file_sha256": None,
            }
        generation_identity = source_rows.depth_run.generation_identity(
            source_identity, selection, "e" * 64, "c" * 16, "model",
            input_variant,
        )
        (run_clip / "generation_identity.json").write_text(
            json.dumps(generation_identity), encoding="utf-8"
        )
        if input_variant["color_mode"] == source_rows.input_color.COLOR_MODE_HDR:
            (run_clip / "hdr_output_stats.json").write_text(json.dumps({
                "format": "linear-scRGB-fp16",
                "hdr_source_kind": source_rows.input_variant_hdr_source_kind(
                    input_variant
                ),
                "input_scale": float(input_variant["scrgb_white_scale"] or 0.0),
                "sdr_white_level_raw": int(
                    input_variant["windows_sdr_white_level_raw"] or 0
                ),
                "output_min": 0.0,
                "output_max": 1.0,
                "nonfinite_components": 0,
            }), encoding="utf-8")
        artifact_identity = source_rows.depth_run.depth_artifact_identity(
            run_clip
        )
        native_run_provenance = None
        if is_native_hdr:
            native_authentication = source_rows.native_hdr_capture.validate_clip(
                clip, full=False
            )
            native_run_provenance = {
                key: native_authentication[key] for key in (
                    "manifest", "manifest_sha256", "content_sha256",
                    "width", "height", "frame_count", "verification",
                )
            }
        (run / "depth_run_manifest.json").write_text(json.dumps({
            "schema": source_rows.depth_run.DEPTH_RUN_MANIFEST_SCHEMA,
            "purpose": "artistic-policy depth supervision",
            "suite": str(clips.resolve()),
            **clip_hash_provenance,
            "source_identities": {"shot": source_identity},
            "model": "model",
            "harness_schema": source_rows.HARNESS_SCHEMA,
            "input_variant": input_variant,
            "input_variant_sha256": source_rows.input_color.input_variant_sha256(
                input_variant
            ),
            "depth_input_color_contract_sha256":
                source_rows.input_color.color_contract_sha256(),
            "metric_preview_encoding": (
                source_rows.input_variant_metric_preview_encoding(
                    input_variant
                )
            ),
            "hdr_source_kind": source_rows.input_variant_hdr_source_kind(
                input_variant
            ),
            "executable_sha256": "e" * 64,
            "conf_sha256": "c" * 16,
            "suite_manifest_sha256": source_rows.sha256(dataset_manifest),
            "policy_warp_source_sha256": "a" * 64,
            "metric_sha256": "b" * 16,
            "clips": [{
                "clip": "shot",
                "frames": len(selected_ids),
                "label_frames": len(label_ids),
                "output_selection_mode": "label-frames",
                "output_label_frames_sha256": source_rows.sha256(label_frames),
                "source_identity_method": source_identity[
                    "source_identity_method"
                ],
                "source_identity_value": source_identity[
                    "source_identity_value"
                ],
                "metric_preview_encoding": (
                    source_rows.input_variant_metric_preview_encoding(
                        input_variant
                    )
                ),
                "contract_sha256": source_rows.sha256(
                    run_clip / "contract.json"
                ),
                **({"native_hdr_model_source": native_run_provenance}
                   if is_native_hdr else {}),
                **artifact_identity,
            }],
            "clip_count": 1,
            "frame_count": len(selected_ids),
        }), encoding="utf-8")
        return clips, run, clip

    def test_prepares_authenticated_native_hdr_model_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant = source_rows.input_color.native_pq_input_variant()
            clips, run, clip = self.make_fixture(
                root, input_variant=variant
            )

            summary = source_rows.prepare(run, clips, root / "output")
            rows = [
                json.loads(line) for line in
                (root / "output" / "labels.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            contract = json.loads(
                (root / "output" / "source_contract.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(summary["native_hdr_rows"], 2)
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["source_schema"], 2)
                self.assertEqual(
                    row["source_contract"],
                    "full-cadence-artistic-source-v2",
                )
                frame_id = row["frame"]
                model_path = (
                    clip / "model_source" / f"frame_{frame_id:05d}.scrgb16"
                )
                self.assertEqual(row["model_source"], str(model_path.resolve()))
                self.assertEqual(
                    row["model_source_sha256"], source_rows.sha256(model_path)
                )
                self.assertEqual(
                    row["model_source_encoding"],
                    source_rows.native_hdr_capture.CAPTURE_ENCODING,
                )
                self.assertEqual(
                    row["metric_preview_encoding"],
                    source_rows.native_hdr_capture.PREVIEW_ENCODING,
                )
                self.assertEqual(
                    row["hdr_source_kind"],
                    source_rows.input_color.INPUT_KIND_NATIVE_PQ,
                )
                provenance = row["native_hdr_source_provenance"]
                self.assertEqual(provenance["frame"], frame_id)
                self.assertEqual(provenance["verification"], "full")
                self.assertEqual(
                    provenance["capture_encoding"],
                    source_rows.native_hdr_capture.CAPTURE_ENCODING,
                )
            self.assertIn(
                "shot", contract["depth_authentication"]["native_hdr_sources"]
            )
            self.assertEqual(
                contract["dataset_manifest"]["source_kind"],
                "native-hdr-video",
            )

    def test_native_hdr_run_color_identity_must_be_exact(self):
        mutations = {
            "hdr_source_kind": "sdr-in-windows-hdr",
            "metric_preview_encoding": "native-srgb-v1",
        }
        for field, value in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                clips, run, _clip = self.make_fixture(
                    root,
                    input_variant=(
                        source_rows.input_color.native_pq_input_variant()
                    ),
                )
                manifest_path = run / "depth_run_manifest.json"
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                manifest[field] = value
                manifest_path.write_text(
                    json.dumps(manifest), encoding="utf-8"
                )

                with self.assertRaisesRegex(
                        RuntimeError, "input color identity is stale"):
                    source_rows.prepare(run, clips, root / "output")

    def test_native_hdr_sidecar_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, clip = self.make_fixture(
                root,
                input_variant=source_rows.input_color.native_pq_input_variant(),
            )
            sidecar = clip / "model_source" / "frame_00001.scrgb16"
            original_stat = sidecar.stat()
            payload = bytearray(sidecar.read_bytes())
            payload[-1] ^= 1
            sidecar.write_bytes(payload)
            os.utime(
                sidecar,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )

            with self.assertRaisesRegex(
                    RuntimeError, "model-source hash differs"):
                source_rows.prepare(run, clips, root / "output")

    def test_native_hdr_sidecar_is_in_late_mutation_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, clip = self.make_fixture(
                root,
                input_variant=source_rows.input_color.native_pq_input_variant(),
            )
            sidecar = clip / "model_source" / "frame_00001.scrgb16"
            original_prepare_clip = source_rows.prepare_clip

            def mutate_after_rows(*args, **kwargs):
                result = original_prepare_clip(*args, **kwargs)
                stat = sidecar.stat()
                os.utime(
                    sidecar,
                    ns=(stat.st_atime_ns, stat.st_mtime_ns + 100),
                )
                return result

            with mock.patch.object(
                    source_rows, "prepare_clip", side_effect=mutate_after_rows):
                with self.assertRaisesRegex(
                        RuntimeError, "authenticated input changed"):
                    source_rows.prepare(run, clips, root / "output")

    def test_native_hdr_sidecar_cadence_must_match_previews(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, _clip = self.make_fixture(
                root,
                input_variant=source_rows.input_color.native_pq_input_variant(),
                native_frame_ids=(0, 1),
            )

            with self.assertRaisesRegex(
                    RuntimeError, "exactly match preview cadence"):
                source_rows.prepare(run, clips, root / "output")

    def test_native_hdr_dimensions_must_match_preview_and_depth_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, _clip = self.make_fixture(
                root,
                input_variant=source_rows.input_color.native_pq_input_variant(),
                native_dimensions=(1, 2),
            )

            with self.assertRaisesRegex(
                    RuntimeError, "preview dimensions differ"):
                source_rows.prepare(run, clips, root / "output")

    def test_prepares_authenticated_windows_hdr_source_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant = source_rows.input_color.windows_hdr_input_variant(2500)
            clips, run, _clip = self.make_fixture(
                root, input_variant=variant
            )
            output = root / "output"
            source_rows.prepare(run, clips, output)
            rows = [
                json.loads(line) for line in
                (output / "labels.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertTrue(rows)
            self.assertTrue(all(row["input_variant"] == variant for row in rows))
            self.assertTrue(all(
                row["color_mode"] == source_rows.input_color.COLOR_MODE_HDR
                for row in rows
            ))

    def test_prepares_mono_rows_and_preserves_optional_stereo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, clip = self.make_fixture(root)
            (clip / "gt_right").mkdir()
            Image.new("RGB", (2, 2), (9, 8, 7)).save(
                clip / "gt_right" / "frame_00001.png"
            )
            output = root / "output"
            summary = source_rows.prepare(run, clips, output)
            rows = [json.loads(line) for line in
                    (output / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["accepted"], 2)
            self.assertEqual(summary["stereo_rows"], 1)
            self.assertEqual([row["frame"] for row in rows], [1, 2])
            self.assertTrue(all(row["film_id"] == "film" for row in rows))
            self.assertTrue(all(row["source_kind"] == "mono-video" for row in rows))
            self.assertTrue(all(
                row["source_contract"] == source_rows.SOURCE_CONTRACT
                for row in rows
            ))
            self.assertIn("right_eye", rows[0])
            self.assertNotIn("right_eye", rows[1])
            contract = json.loads(
                (output / "source_contract.json").read_text(encoding="utf-8")
            )
            self.assertEqual(contract["run_contract"]["model"], "model")
            self.assertEqual(
                contract["dataset_manifest"]["source_kind"], "mono-video"
            )

    def test_clip_metadata_cannot_reassign_authenticated_production(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, clip = self.make_fixture(root)
            metadata_path = clip / "meta.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update({"film_id": "other", "split": "development"})
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(
                    RuntimeError, "differs from dataset manifest"):
                source_rows.prepare(run, clips, root / "output")

    def test_depth_run_must_authenticate_exact_dataset_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, _clip = self.make_fixture(root)
            manifest_path = clips / "dataset_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["license"] = "changed"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(
                    RuntimeError, "does not authenticate"):
                source_rows.prepare(run, clips, root / "output")

    def test_legacy_stereo_manifest_normalizes_source_kind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, clip = self.make_fixture(root)
            manifest_path = clips / "dataset_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update({
                "schema": 1,
                "film_id": manifest.pop("production_id"),
            })
            manifest.pop("source_kind")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            metadata_path = clip / "meta.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["source_kind"] = "authored-stereo"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            run_manifest_path = run / "depth_run_manifest.json"
            run_manifest = json.loads(
                run_manifest_path.read_text(encoding="utf-8")
            )
            run_manifest["suite_manifest_sha256"] = source_rows.sha256(
                manifest_path
            )
            run_manifest_path.write_text(
                json.dumps(run_manifest), encoding="utf-8"
            )

            source_rows.prepare(run, clips, root / "output")
            row = json.loads(
                (root / "output" / "labels.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()[0]
            )

            self.assertEqual(row["source_kind"], "authored-stereo")

    def test_depth_run_clip_set_must_exactly_match_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, _clip = self.make_fixture(root)
            run_manifest_path = run / "depth_run_manifest.json"
            run_manifest = json.loads(
                run_manifest_path.read_text(encoding="utf-8")
            )
            run_manifest["clips"].append({"clip": "extra"})
            run_manifest_path.write_text(
                json.dumps(run_manifest), encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "clip set/order differs"):
                source_rows.prepare(run, clips, root / "output")

    def test_unlabelled_cadence_rgb_must_match_depth_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, clip = self.make_fixture(root)
            Image.new("RGB", (2, 2), (99, 98, 97)).save(
                clip / "frame_00000.png"
            )

            with self.assertRaisesRegex(RuntimeError, "source identity differs"):
                source_rows.prepare(run, clips, root / "output")

    def test_swapped_contract_or_artifact_fails_manifest_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, _clip = self.make_fixture(root)
            contract_path = run / "shot" / "contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["content_scale_x"] = 0.75
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "contract SHA-256 differs"):
                source_rows.prepare(run, clips, root / "contract-output")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, _clip = self.make_fixture(root)
            write_float_texture(
                run / "shot" / "baseline_disparity_00001.f32",
                width=2, height=2,
            )
            artifact = run / "shot" / "baseline_disparity_00001.f32"
            payload = bytearray(artifact.read_bytes())
            payload[-1] ^= 1
            artifact.write_bytes(payload)
            with self.assertRaisesRegex(RuntimeError, "artifact identity differs"):
                source_rows.prepare(run, clips, root / "artifact-output")

    def test_late_source_mutation_fails_before_atomic_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, clip = self.make_fixture(root)
            output = root / "output"
            output.mkdir()
            marker = output / "previous.txt"
            marker.write_text("previous", encoding="utf-8")
            original = source_rows.prepare_clip

            def mutate_after_rows(*args, **kwargs):
                result = original(*args, **kwargs)
                Image.new("RGB", (2, 2), (40, 41, 42)).save(
                    clip / "frame_00000.png"
                )
                return result

            with mock.patch.object(
                    source_rows, "prepare_clip", side_effect=mutate_after_rows):
                with self.assertRaisesRegex(
                        RuntimeError, "authenticated input changed"):
                    source_rows.prepare(
                        run, clips, output, overwrite=True
                    )

            self.assertEqual(marker.read_text(encoding="utf-8"), "previous")
            self.assertFalse(any(
                path.name.startswith(".output.partial-")
                for path in root.iterdir()
            ))

    def test_prepared_clip_set_must_exactly_match_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, clip = self.make_fixture(root)
            extra = clips / "extra"
            extra.mkdir()
            (extra / "label_frames.json").write_text(
                (clip / "label_frames.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "sequence set differs"):
                source_rows.prepare(run, clips, root / "output")

    def test_single_target_uses_adjacent_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, _ = self.make_fixture(
                root, source_ids=(0, 1), label_ids=(0,)
            )
            summary = source_rows.prepare(run, clips, root / "output")
            row = json.loads(
                (root / "output" / "labels.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(row["label_frame_ids"], [0])
            self.assertEqual(row["output_selected_frame_ids"], [0, 1])

    def test_jpeg_sources_and_different_right_eye_extension_are_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, clip = self.make_fixture(
                root, source_suffix=".jpg"
            )
            (clip / "gt_right").mkdir()
            Image.new("RGB", (2, 2), (9, 8, 7)).save(
                clip / "gt_right" / "frame_00001.png"
            )

            source_rows.prepare(run, clips, root / "output")
            rows = [
                json.loads(line) for line in
                (root / "output" / "labels.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

            self.assertTrue(rows[0]["source"].endswith("frame_00001.jpg"))
            self.assertTrue(rows[0]["right_eye"].endswith("frame_00001.png"))

    def test_duplicate_numeric_source_identity_across_extensions_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, clip = self.make_fixture(root)
            Image.new("RGB", (2, 2), (4, 5, 6)).save(
                clip / "frame_00001.jpg"
            )

            with self.assertRaisesRegex(RuntimeError, "duplicate numeric"):
                source_rows.prepare(run, clips, root / "output")

    def test_sparse_targets_preserve_only_target_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, _ = self.make_fixture(
                root, source_ids=(0, 1, 2, 3), label_ids=(0, 3)
            )
            source_rows.prepare(run, clips, root / "output")
            rows = [json.loads(line) for line in
                    (root / "output" / "labels.jsonl").read_text(
                        encoding="utf-8"
                    ).splitlines()]
            self.assertEqual([row["frame"] for row in rows], [0, 3])
            self.assertTrue(all(
                row["output_selected_frame_ids"] == [0, 1, 2, 3]
                for row in rows
            ))

    def test_actual_still_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, _ = self.make_fixture(
                root, source_ids=(0,), label_ids=(0,), selected_ids=(0,)
            )
            with self.assertRaisesRegex(
                    RuntimeError, "still images|no consecutive"):
                source_rows.prepare(run, clips, root / "output")

    def test_non_contiguous_source_sequence_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, _clip = self.make_fixture(
                root, source_ids=(0, 2), label_ids=(0, 1), selected_ids=(0, 2)
            )
            with self.assertRaisesRegex(
                    RuntimeError, "not full cadence|references missing RGB"):
                source_rows.prepare(run, clips, root / "output")

    def test_output_root_cannot_overlap_either_input_before_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, _clip = self.make_fixture(root)
            clips_guard = clips / "do-not-delete.txt"
            run_guard = run / "do-not-delete.txt"
            clips_guard.write_text("clips", encoding="utf-8")
            run_guard.write_text("run", encoding="utf-8")

            for output in (clips, clips / "nested", run, run / "nested", root):
                with self.subTest(output=output):
                    with self.assertRaisesRegex(RuntimeError, "overlaps"):
                        source_rows.prepare(
                            run, clips, output, overwrite=True
                        )
                    self.assertEqual(
                        clips_guard.read_text(encoding="utf-8"), "clips"
                    )
                    self.assertEqual(
                        run_guard.read_text(encoding="utf-8"), "run"
                    )

    def test_late_publication_failure_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, _clip = self.make_fixture(root)
            output = root / "output"
            output.mkdir()
            old = output / "old-contract.json"
            old.write_text("old", encoding="utf-8")

            with mock.patch.object(
                    source_rows, "publish_staged_directory",
                    side_effect=RuntimeError("publish failed")):
                with self.assertRaisesRegex(RuntimeError, "publish failed"):
                    source_rows.prepare(run, clips, output, overwrite=True)

            self.assertEqual(old.read_text(encoding="utf-8"), "old")
            self.assertFalse(any(
                path.name.startswith(".output.partial-")
                for path in root.iterdir()
            ))

    def test_overwrite_publishes_complete_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips, run, _clip = self.make_fixture(root)
            output = root / "output"
            output.mkdir()
            (output / "old.txt").write_text("old", encoding="utf-8")

            source_rows.prepare(run, clips, output, overwrite=True)

            self.assertFalse((output / "old.txt").exists())
            self.assertTrue((output / "labels.jsonl").is_file())
            self.assertTrue((output / "source_contract.json").is_file())
            self.assertTrue((output / "summary.json").is_file())
            self.assertFalse(any(
                path.name.startswith((".output.partial-", ".output.backup-"))
                for path in root.iterdir()
            ))


if __name__ == "__main__":
    unittest.main()
