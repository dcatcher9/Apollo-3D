#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import artistic_geometry_contract as geometry_contract
import artistic_policy_ordinal_contract as ordinal_contract
import build_ordinal_frame_label_bundle as bundle_builder
import depth_input_color as input_color
import native_hdr_capture
import prepare_ordinal_full_frame_source_rows as source_rows


METRICS = {
    "exact_pop_spread_pct": {
        "role": "primary", "axis": "stereo", "better": "higher",
        "rel_tol": 0.1, "abs_floor": 0.1,
    },
    "source_coverage_pct": {
        "role": "hard", "axis": "integrity", "better": "higher",
        "rel_tol": 0.0, "abs_floor": 1.0, "hard_min": 90.0,
    },
}


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_geometry(source_width, source_height, eye_width, eye_height,
                  color_mode):
    scale_x, scale_y = geometry_contract.source_content_scales(
        source_width, source_height, eye_width, eye_height
    )
    return geometry_contract.geometry_tuple({
        "source_width": source_width,
        "source_height": source_height,
        "eye_width": eye_width,
        "eye_height": eye_height,
        "content_scale_x": scale_x,
        "content_scale_y": scale_y,
        "disparity_raster_width": eye_width,
        "disparity_raster_height": eye_height,
        "color_mode": color_mode,
    })


def write_source_frames(clip_root, count=2):
    clip_root.mkdir(parents=True)
    paths = []
    for frame_id in range(count):
        image = np.full((252, 448, 3), 30 + frame_id * 40, np.uint8)
        path = clip_root / f"frame_{frame_id:05d}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError("test PNG write failed")
        paths.append(path)
    return paths


def write_bundle(path, clip, sources, variant, label_frame_ids=None):
    variant_sha256 = input_color.input_variant_sha256(variant)
    source_frame_ids = list(range(len(sources)))
    label_frame_ids = list(
        [source_frame_ids[-1]] if label_frame_ids is None else label_frame_ids
    )
    output_frame_ids = list(label_frame_ids)
    label_frames_sha256 = digest("label-frames:" + repr(label_frame_ids))
    geometries = (
        make_geometry(448, 252, 1280, 720, variant["color_mode"]),
        make_geometry(448, 252, 960, 540, variant["color_mode"]),
    )
    common = {
        "clip": clip,
        "metric_sha256": bundle_builder.run_eval.metric_contract_sha(),
        "thresholds_sha256": bundle_builder.sha256_file(
            bundle_builder.THRESHOLDS_PATH
        ),
        "clip_sha1": "a" * 12,
        "expected_flat": False,
        "pipeline_without_scale": {"model": "depth_anything_v2_fp16"},
        "output_selection_contract":
            bundle_builder.run_eval.SELECTED_FRAME_GATE_OUTPUT_SELECTION_CONTRACT,
        "source_frame_count": len(source_frame_ids),
        "source_frame_ids": source_frame_ids,
        "source_frame_ids_sha256":
            bundle_builder.run_eval.frame_id_sequence_sha256(source_frame_ids),
        "label_frame_count": len(label_frame_ids),
        "label_frame_ids": label_frame_ids,
        "output_frame_count": len(output_frame_ids),
        "output_selected_frame_ids": output_frame_ids,
        "output_label_frames_sha256": label_frames_sha256,
        "runtime_scene_count": 1,
        "completion_sequence_contract": (
            "exact for this synchronous harness sequence; live busy-drop cadence "
            "is not replayed"
        ),
        "executable_sha256": digest("sunshine"),
    }
    runtime_scene_trace = [{
        "source_frame_ordinal": frame_id,
        "source_frame_id": frame_id,
        "runtime_scene_id": 0,
        "scene_age": float(frame_id),
        "subject_initialized": True,
        "hard_cut": False,
        "scene_start": frame_id == 0,
    } for frame_id in source_frame_ids]
    runs = []
    for geometry_index, geometry in enumerate(geometries):
        geometry_sha256 = bundle_builder.canonical_sha256(geometry)
        for scale in ordinal_contract.SCALES:
            frames = []
            for ordinal, frame_id in enumerate(label_frame_ids):
                source = sources[frame_id]
                frames.append({
                    "frame_id": frame_id,
                    "ordinal": ordinal,
                    "source_ordinal": frame_id,
                    "runtime_scene_id": 0,
                    "runtime_scene_evidence": {
                        "source_frame_ordinal": frame_id,
                        "source_frame_id": frame_id,
                        "runtime_scene_id": 0,
                        "scene_age": float(frame_id),
                        "subject_initialized": True,
                        "hard_cut": False,
                        "scene_start": frame_id == 0,
                    },
                    "artifact_sha256": {
                        "source": source_rows.sha256_file(source),
                        "depth": digest(f"depth:{frame_id}"),
                    },
                    "metrics": {
                        "exact_pop_spread_pct": 2.0 + scale - 1.0,
                        "source_coverage_pct": 99.0,
                    },
                })
            runs.append({
                "scale": scale,
                "clip": clip,
                "geometry": geometry,
                "geometry_sha256": geometry_sha256,
                "input_variant": variant,
                "input_variant_sha256": variant_sha256,
                "common_identity": common,
                "run_identity": {
                    "frame_gate_evidence_sha256": digest(
                        f"gate:{geometry_index}:{scale}"
                    ),
                    "results_sha256": digest(
                        f"results:{geometry_index}:{scale}"
                    ),
                    "harness_contract_sha256": digest(
                        f"harness:{geometry_index}:{scale}"
                    ),
                    "runtime_scene_evidence_sha256": digest("scenes"),
                    "multiscale_batch_manifest_sha256": digest(
                        f"batch:{geometry_index}"
                    ),
                    "multiscale_harness_contract_sha256": digest(
                        f"multiscale-harness:{geometry_index}"
                    ),
                    "multiscale_scale_contract_sha256": digest(
                        f"harness:{geometry_index}:{scale}"
                    ),
                    "run_name": f"g{geometry_index}-s{scale}",
                    "geometry_sha256": geometry_sha256,
                    "scale": scale,
                },
                "frames": frames,
                "_runtime_scene_trace": runtime_scene_trace,
            })
    header, frames = bundle_builder.build_frame_label_bundle(
        runs, METRICS, variant_sha256
    )
    path.mkdir(parents=True)
    bundle_builder.write_frame_label_bundle(
        path / "labels.jsonl", path / "summary.json", header, frames
    )
    return path / "labels.jsonl"


def write_dataset(root, split="training", source_kind="mono-video",
                  frame_count=2):
    manifest = {
        "schema": 2,
        "production_id": f"fixture_{split}",
        "source_kind": source_kind,
        "split": split,
        "global_policy_weight": 1.0,
        "dataset": "fixture",
        "domain": "fixture-scenes",
        "license": "test-only",
        "policy_role": "cinematic_training",
        "sequences": [{
            "clip": "shot-a",
            "split": split,
            ("context_frames" if source_kind == "mono-video" else "frames"):
                frame_count,
        }],
    }
    if source_kind == "mono-video":
        manifest["context_fps"] = 24.0
    else:
        manifest["sequences"][0]["source_frame_rate"] = 24.0
    manifest[
        "context_frame_count" if source_kind == "mono-video" else "frame_count"
    ] = frame_count
    path = root / "dataset_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def write_native_manifest(clip_root, sources):
    model_root = clip_root / native_hdr_capture.MODEL_SOURCE_DIRECTORY
    model_root.mkdir()
    frames = []
    semantic_frames = []
    for frame_id, preview in enumerate(sources):
        model = model_root / f"frame_{frame_id:05d}.scrgb16"
        np.zeros((252, 448, 4), dtype="<f2").tofile(model)
        stat = model.stat()
        relative_model = model.relative_to(clip_root).as_posix()
        relative_preview = preview.relative_to(clip_root).as_posix()
        row = {
            "frame": frame_id,
            "path": relative_model,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": source_rows.sha256_file(model),
            "preview": relative_preview,
            "preview_sha256": source_rows.sha256_file(preview),
            "timestamp_seconds": frame_id / 24.0,
            "stats": {},
        }
        frames.append(row)
        semantic_frames.append({
            key: row[key] for key in (
                "frame", "path", "size", "sha256", "preview",
                "preview_sha256", "timestamp_seconds",
            )
        })
    source_video = {"sha256": "b" * 64}
    conversion = {"contract_sha256": "c" * 64}
    semantic = {
        "contract": native_hdr_capture.MANIFEST_CONTRACT,
        "capture_encoding": native_hdr_capture.CAPTURE_ENCODING,
        "preview_encoding": native_hdr_capture.PREVIEW_ENCODING,
        "width": 448,
        "height": 252,
        "row_pitch_bytes": 448 * 8,
        "source_video": source_video,
        "conversion": conversion,
        "frames": semantic_frames,
    }
    payload = {
        "schema": native_hdr_capture.MANIFEST_SCHEMA,
        **semantic,
        "frames": frames,
        "frame_count": len(frames),
        "content_sha256": native_hdr_capture.canonical_sha256(semantic),
    }
    (clip_root / native_hdr_capture.MANIFEST_NAME).write_bytes(
        native_hdr_capture.canonical_json_bytes(payload)
    )


def swap_native_preview_join(clip_root):
    manifest_path = clip_root / native_hdr_capture.MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    first, second = payload["frames"][:2]
    first["preview"], second["preview"] = second["preview"], first["preview"]
    first["preview_sha256"], second["preview_sha256"] = (
        second["preview_sha256"], first["preview_sha256"]
    )
    semantic_frames = [{
        key: row[key] for key in (
            "frame", "path", "size", "sha256", "preview",
            "preview_sha256", "timestamp_seconds",
        )
    } for row in payload["frames"]]
    semantic = {
        "contract": payload["contract"],
        "capture_encoding": payload["capture_encoding"],
        "preview_encoding": payload["preview_encoding"],
        "width": payload["width"],
        "height": payload["height"],
        "row_pitch_bytes": payload["row_pitch_bytes"],
        "source_video": payload["source_video"],
        "conversion": payload["conversion"],
        "frames": semantic_frames,
    }
    payload["content_sha256"] = native_hdr_capture.canonical_sha256(semantic)
    manifest_path.write_bytes(native_hdr_capture.canonical_json_bytes(payload))


class OrdinalFullFrameSourceRowsTests(unittest.TestCase):
    def test_publishes_only_authenticated_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            manifest = write_dataset(dataset)
            sources = write_source_frames(dataset / "shot-a")
            bundle = write_bundle(
                root / "bundle", "shot-a", sources,
                input_color.windows_hdr_input_variant(2500),
            )
            output = root / "sources"
            summary = source_rows.publish(manifest, [bundle], output)
            rows = source_rows.validate_full_frame_source_bundle(
                output / "labels.jsonl"
            )
            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(summary["source_frame_count"], 2)
            self.assertEqual(summary["label_frame_count"], 1)
            self.assertEqual(summary["target_row_count"], 1)
            self.assertEqual(summary["context_row_count"], 0)
            self.assertEqual(summary["output_frame_count"], 1)
            self.assertEqual(summary["source_frame_rates"], [24.0])
            self.assertEqual([row["frame"] for row in rows], [1])
            self.assertEqual([row["row_role"] for row in rows], ["target"])
            self.assertEqual(rows[0]["label_ordinal"], 0)
            self.assertEqual([row["source_ordinal"] for row in rows], [1])
            self.assertEqual(
                {row["source_frame_rate"] for row in rows}, {24.0}
            )
            self.assertTrue(all(
                row["ordinal_bundle_sha256"] ==
                source_rows.sha256_file(bundle) for row in rows
            ))
            original_unselected = sources[0].read_bytes()
            sources[0].unlink()
            with self.assertRaisesRegex(RuntimeError, "cadence"):
                source_rows.validate_full_frame_source_bundle(
                    output / "labels.jsonl"
                )
            sources[0].write_bytes(original_unselected)
            sources[0].write_bytes(b"changed-unselected")
            with self.assertRaisesRegex(RuntimeError, "cadence content"):
                source_rows.validate_full_frame_source_bundle(
                    output / "labels.jsonl"
                )
            sources[0].write_bytes(original_unselected)
            sources[1].write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "cadence content"):
                source_rows.validate_full_frame_source_bundle(
                    output / "labels.jsonl"
                )

    def test_rejects_bundle_summary_aggregate_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            manifest = write_dataset(dataset)
            sources = write_source_frames(dataset / "shot-a")
            bundle = write_bundle(
                root / "bundle", "shot-a", sources,
                input_color.sdr_input_variant(),
            )
            summary_path = bundle.parent / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["frontier_bounds"] = {"fabricated-frontier": 1}
            summary_path.write_bytes(bundle_builder.canonical_bytes(summary))

            with self.assertRaisesRegex(RuntimeError, "authenticated bundle"):
                source_rows.build_rows(manifest, [bundle])

    def test_native_pq_rows_use_authenticated_linear_model_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            manifest = write_dataset(dataset, source_kind="native-hdr-video")
            sources = write_source_frames(dataset / "shot-a")
            write_native_manifest(dataset / "shot-a", sources)
            bundle = write_bundle(
                root / "bundle", "shot-a", sources,
                input_color.native_pq_input_variant(),
            )
            output = root / "sources"
            summary = source_rows.publish(manifest, [bundle], output)
            rows = source_rows.validate_full_frame_source_bundle(
                output / "labels.jsonl"
            )
            self.assertEqual(summary["native_hdr_rows"], 1)
            self.assertTrue(all(
                row["model_source_encoding"] ==
                native_hdr_capture.CAPTURE_ENCODING for row in rows
            ))

    def test_native_pq_rejects_valid_manifest_with_swapped_preview_join(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            manifest = write_dataset(dataset, source_kind="native-hdr-video")
            sources = write_source_frames(dataset / "shot-a")
            write_native_manifest(dataset / "shot-a", sources)
            swap_native_preview_join(dataset / "shot-a")
            bundle = write_bundle(
                root / "bundle", "shot-a", sources,
                input_color.native_pq_input_variant(),
            )
            with self.assertRaisesRegex(
                    RuntimeError, "preview (?:identity|join) differs"):
                source_rows.build_rows(manifest, [bundle])

    def test_rejects_sealed_test_before_opening_clip_or_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            manifest = write_dataset(dataset, split="test")
            with self.assertRaisesRegex(RuntimeError, "train/development only"):
                source_rows.build_rows(manifest, [root / "missing-bundle"])

    def test_rejects_sparse_source_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            manifest = write_dataset(dataset, frame_count=3)
            sources = write_source_frames(dataset / "shot-a", count=2)
            bundle = write_bundle(
                root / "bundle", "shot-a", sources,
                input_color.sdr_input_variant(),
            )
            with self.assertRaisesRegex(RuntimeError, "cadence"):
                source_rows.build_rows(manifest, [bundle])

    def test_rejects_missing_or_nonpositive_authenticated_cadence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_dataset(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload.pop("context_fps")
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid context_fps"):
                source_rows.load_dataset_manifest(manifest)
            payload["context_fps"] = 0.0
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid context_fps"):
                source_rows.load_dataset_manifest(manifest)

    def test_authenticated_clip_subset_is_marked_smoke_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            manifest = write_dataset(dataset)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["sequences"].append({
                "clip": "shot-b", "split": "training", "context_frames": 2,
            })
            payload["context_frame_count"] = 4
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            write_source_frames(dataset / "shot-a")
            sources = write_source_frames(dataset / "shot-b")
            bundle = write_bundle(
                root / "bundle", "shot-b", sources,
                input_color.sdr_input_variant(),
            )
            output = root / "sources"
            summary = source_rows.publish(
                manifest, [bundle], output, selected_clips=["shot-b"]
            )
            rows = source_rows.validate_full_frame_source_bundle(
                output / "labels.jsonl"
            )
            self.assertEqual(summary["scope"],
                             "smoke-subset-not-training-eligible")
            self.assertEqual(summary["selected_clips"], ["shot-b"])
            self.assertEqual({row["clip"] for row in rows}, {"shot-b"})
            with self.assertRaisesRegex(RuntimeError, "exactly cover"):
                source_rows.build_rows(manifest, [bundle])


if __name__ == "__main__":
    unittest.main()
