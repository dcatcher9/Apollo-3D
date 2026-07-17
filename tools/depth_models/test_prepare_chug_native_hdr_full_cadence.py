import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import prepare_chug_native_hdr_full_cadence as full  # noqa: E402
import audit_chug_native_hdr_full_cadence as full_audit  # noqa: E402


TRAIN_COUNTS = [144, 242, 270, 210, 301, 300, 300, 300, 240, 300, 300, 181]
DEV_COUNTS = [180, 300, 150, 300]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _with_content_hash(payload):
    payload = dict(payload)
    payload["content_sha256"] = full.native_hdr_capture.canonical_sha256(payload)
    return payload


def _clip_fixture(root, *, cut_threshold=full.CUT_THRESHOLD):
    video_id = "1" * 32
    capture_group = "2" * 64
    content_id = "3" * 64
    source_sha = "4" * 64
    conversion_hash = "5" * 64
    frame_count = 3
    label_frame_ids = [0, 1, 2]
    timing = _with_content_hash({
        "schema": full.TIMING_SCHEMA,
        "contract": full.TIMING_CONTRACT,
        "video_id": video_id,
        "source_video_sha256": source_sha,
        "frame_count": frame_count,
        "source_frame_rate": {
            "numerator": 30, "denominator": 1,
            "rational": "30/1", "decimal": 30.0,
        },
        "time_base": {
            "numerator": 1, "denominator": 30000,
            "rational": "1/30000", "decimal": 1 / 30000,
        },
        "frames": [{
            "frame": frame_id,
            "timestamp_ticks": frame_id * 1000,
            "timestamp_seconds": frame_id / 30,
        } for frame_id in range(frame_count)],
    })
    cut_rows = [{
        "frame": frame_id,
        "timestamp_ticks": frame_id * 1000,
        "timestamp_seconds": frame_id / 30,
        "scene_start": frame_id == 0,
        "preview_mean_absolute_delta": None if frame_id == 0 else (
            0.25 if frame_id == 2 else 0.01
        ),
        "cut_candidate": frame_id == 2,
    } for frame_id in range(frame_count)]
    cuts = _with_content_hash({
        "schema": full.CUT_SCHEMA,
        "contract": full.CUT_CONTRACT,
        "video_id": video_id,
        "source_video_sha256": source_sha,
        "frame_count": frame_count,
        "threshold": cut_threshold,
        "cut_candidate_count": 1,
        "frames": cut_rows,
    })
    destination = root / full._clip_name(video_id)
    destination.mkdir(parents=True)
    timing_path = destination / "source_timing.json"
    cuts_path = destination / "source_cut_evidence.json"
    _write(timing_path, timing)
    _write(cuts_path, cuts)
    source_video = {
        "video_id": video_id,
        "split": "training",
        "capture_group_id": capture_group,
        "content_id": content_id,
        "bytes": 1234,
        "sha256": source_sha,
        "source_frame_count": frame_count,
        "source_timing_sha256": _sha(timing_path),
        "source_timing_content_sha256": timing["content_sha256"],
        "source_cut_evidence_sha256": _sha(cuts_path),
        "source_cut_evidence_content_sha256": cuts["content_sha256"],
    }
    model_root = destination / full.native_hdr_capture.MODEL_SOURCE_DIRECTORY
    model_root.mkdir()
    records = []
    for frame_id in range(frame_count):
        suffix = f"{frame_id:05d}"
        model = model_root / f"frame_{suffix}.scrgb16"
        model.write_bytes(bytes([frame_id]) * (8 * 4 * 8))
        preview = destination / f"frame_{suffix}.png"
        Image.new("RGB", (8, 4), (frame_id * 10,) * 3).save(preview)
        records.append({
            "frame": frame_id,
            "path": f"model_source/frame_{suffix}.scrgb16",
            "size": model.stat().st_size,
            "mtime_ns": model.stat().st_mtime_ns,
            "sha256": _sha(model),
            "preview": preview.name,
            "preview_sha256": _sha(preview),
            "timestamp_seconds": frame_id / 30,
            "stats": {
                "nonfinite_components": 0,
                "luminance_nits_max": 100.0,
                "preview_saturated_fraction": 0.0,
                "preview_black_fraction": 0.0,
            },
        })
    semantic_rows = [{
        key: value for key, value in record.items()
        if key not in {"mtime_ns", "stats"}
    } for record in records]
    semantic = {
        "contract": full.native_hdr_capture.MANIFEST_CONTRACT,
        "capture_encoding": full.native_hdr_capture.CAPTURE_ENCODING,
        "preview_encoding": full.native_hdr_capture.PREVIEW_ENCODING,
        "width": 8,
        "height": 4,
        "row_pitch_bytes": 8 * full.SCRGB_BYTES_PER_PIXEL,
        "source_video": source_video,
        "conversion": {"contract_sha256": conversion_hash},
        "frames": semantic_rows,
    }
    _write(destination / full.native_hdr_capture.MANIFEST_NAME, {
        "schema": full.native_hdr_capture.MANIFEST_SCHEMA,
        "contract": full.native_hdr_capture.MANIFEST_CONTRACT,
        "capture_encoding": full.native_hdr_capture.CAPTURE_ENCODING,
        "preview_encoding": full.native_hdr_capture.PREVIEW_ENCODING,
        "width": 8,
        "height": 4,
        "row_pitch_bytes": 8 * full.SCRGB_BYTES_PER_PIXEL,
        "source_video": source_video,
        "conversion": {"contract_sha256": conversion_hash},
        "frames": records,
        "frame_count": len(records),
        "content_sha256": full.native_hdr_capture.canonical_sha256(semantic),
    })
    _write(destination / "label_frames.json", {
        "schema": 1, "frame_ids": label_frame_ids,
    })
    metadata = {
        "preparation_contract": full.PREPARATION_CONTRACT,
        "full_cadence_contract": full.FULL_CADENCE_CONTRACT,
        "production_id": f"{full.PRODUCTION_PREFIX}_training",
        "split": "training",
        "capture_group_id": capture_group,
        "source_video_id": video_id,
        "frame_count": frame_count,
        "source_timing_sha256": _sha(timing_path),
        "source_timing_content_sha256": timing["content_sha256"],
        "source_cut_evidence_sha256": _sha(cuts_path),
        "source_cut_evidence_content_sha256": cuts["content_sha256"],
        "curated_diagnostic_label_frame_ids": label_frame_ids,
    }
    _write(destination / "meta.json", metadata)
    row = {
        "video_id": video_id,
        "split": "training",
        "capture_group_id": capture_group,
        "content_id": content_id,
        "source_frame_count": frame_count,
        "label_frame_ids": label_frame_ids,
        "download": {"bytes": 1234, "sha256": source_sha},
        "timing": timing,
    }
    sequence = {
        "clip": destination.name,
        "frames": frame_count,
        "label_frame_ids": label_frame_ids,
        "split": "training",
        "capture_group_id": capture_group,
        "video_id": video_id,
        "source_frame_rate_rational": "30/1",
        "source_time_base_rational": "1/30000",
        "source_timing_content_sha256": timing["content_sha256"],
        "source_cut_candidate_count": 1,
    }
    return destination, row, sequence, conversion_hash


def _sparse_fixture(root):
    datasets = {}
    identity = 0
    rows = {}
    for split, counts in (("training", TRAIN_COUNTS),
                          ("development", DEV_COUNTS)):
        sequences = []
        split_rows = []
        for count in counts:
            video_id = f"{identity + 1:032x}"
            group = f"{identity + 1000:064x}"
            identity += 1
            labels = [round((index + 1) * (count - 1) / 6) for index in range(5)]
            row = {
                "video_id": video_id,
                "split": split,
                "capture_group_id": group,
                "source_frame_count": count,
                "labels": labels,
            }
            split_rows.append(row)
            for window, label in enumerate(labels):
                sequences.append({
                    "clip": f"clip-{video_id}-{window}",
                    "frames": 3,
                    "source_frames": 3,
                    "master_source_frames": count,
                    "source_frame_rate": 24.0 if identity <= 2 else 30.0,
                    "label_frames": 1,
                    "split": split,
                    "capture_group_id": group,
                    "video_id": video_id,
                    "window_index": window,
                    "source_label_frame_id": label,
                    "temporal_evidence_selection": {"contract": "fixture"},
                })
        manifest = {
            "schema": 2,
            "dataset": "chug-native-pq-v1",
            "production_id": f"chug_native_pq_v1_{split}",
            "split": split,
            "preparation_contract": full.SOURCE_BOOTSTRAP_CONTRACT,
            "sequences": sequences,
            "master_source_frame_count": sum(counts),
            "label_frame_count": len(counts) * 5,
        }
        manifest_path = root / split / "dataset_manifest.json"
        _write(manifest_path, manifest)
        datasets[split] = {
            "dataset_manifest": str(manifest_path),
            "dataset_manifest_sha256": _sha(manifest_path),
        }
        rows[split] = split_rows
    bootstrap = {
        "schema": full.SOURCE_BOOTSTRAP_SCHEMA,
        "contract": full.SOURCE_BOOTSTRAP_CONTRACT,
        "sealed_test_policy": "CHUG test masters were not decoded or opened",
        "datasets": datasets,
    }
    bootstrap_path = root / "native_hdr_bootstrap_manifest.json"
    _write(bootstrap_path, bootstrap)
    return bootstrap_path, rows


class FullCadencePreparationTest(unittest.TestCase):
    def test_sparse_windows_collapse_to_frozen_video_cardinality(self):
        with tempfile.TemporaryDirectory() as directory:
            bootstrap, expected = _sparse_fixture(Path(directory))
            selected, provenance = full._load_sparse_selection(bootstrap)
        self.assertEqual(len(selected["training"]), 12)
        self.assertEqual(len(selected["development"]), 4)
        self.assertEqual(sum(row["source_frame_count"]
                             for row in selected["training"]), 3088)
        self.assertEqual(sum(row["source_frame_count"]
                             for row in selected["development"]), 930)
        self.assertEqual(selected["training"][0]["label_frame_ids"],
                         expected["training"][0]["labels"])
        self.assertIn("sparse_bootstrap_manifest_sha256", provenance)

    def test_sparse_selection_rejects_cross_split_capture_group(self):
        with tempfile.TemporaryDirectory() as directory:
            bootstrap, _ = _sparse_fixture(Path(directory))
            document = json.loads(bootstrap.read_text(encoding="utf-8"))
            train_path = Path(document["datasets"]["training"]["dataset_manifest"])
            dev_path = Path(document["datasets"]["development"]["dataset_manifest"])
            train = json.loads(train_path.read_text(encoding="utf-8"))
            dev = json.loads(dev_path.read_text(encoding="utf-8"))
            dev_group = train["sequences"][0]["capture_group_id"]
            for row in dev["sequences"][:5]:
                row["capture_group_id"] = dev_group
            _write(dev_path, dev)
            document["datasets"]["development"]["dataset_manifest_sha256"] = _sha(dev_path)
            _write(bootstrap, document)
            with self.assertRaisesRegex(RuntimeError, "crosses train/dev"):
                full._load_sparse_selection(bootstrap)

    def test_selected_rows_never_touch_sealed_test_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            videos.mkdir()
            selected = {"training": [], "development": []}
            selection_rows = []
            receipt_rows = []
            for index, split in enumerate(("training", "development")):
                video_id = f"{index + 1:032x}"
                group = f"{index + 10:064x}"
                video = videos / f"{video_id}.mp4"
                video.write_bytes(f"video-{index}".encode())
                base = {
                    "video_id": video_id,
                    "split": split,
                    "capture_group_id": group,
                    "content_id": f"content-{index}",
                    "source_frame_count": 3,
                    "sparse_source_frame_rate": 30.0,
                    "label_frame_ids": [0, 1, 2, 1, 0],
                }
                selected[split].append(base)
                selection_rows.append({**base})
                receipt_rows.append({
                    **base,
                    "orientation": "Landscape",
                    "content_id": f"content-{index}",
                    "audit": {
                        "codec": "hevc", "color_range": "tv",
                        "color_primaries": "bt2020", "color_space": "bt2020nc",
                        "color_transfer": "smpte2084", "pixel_format": "yuv420p10le",
                        "width": 1920, "height": 1080,
                    },
                    "download": {
                        "video_id": video_id, "bytes": video.stat().st_size,
                        "sha256": _sha(video),
                    },
                })
            test_id = "f" * 32
            selection_rows.append({
                "video_id": test_id, "split": "test",
                "capture_group_id": "e" * 64,
            })
            receipt_rows.append({
                "video_id": test_id, "split": "test",
                "capture_group_id": "e" * 64,
                "download": {"video_id": test_id, "path": "videos/sealed.mp4"},
            })
            _write(root / "selection_manifest.json", {
                "schema": 1, "clips": selection_rows,
            })
            _write(root / "download_receipt.json", {
                "schema": 1, "license": "CC BY-NC-SA 4.0",
                "accepted": receipt_rows,
            })
            touched = []
            original = full.sparse.sha256

            def tracked(path):
                touched.append(Path(path))
                return original(path)

            with mock.patch.object(full.sparse, "sha256", side_effect=tracked):
                resolved, _ = full._selected_source_rows(root, selected)
            self.assertEqual(len(resolved["training"]), 1)
            self.assertFalse(any(path.name == "sealed.mp4" for path in touched))

    def test_timing_probe_preserves_30000_over_1001_ticks(self):
        stream = {
            "codec_name": "hevc", "pix_fmt": "yuv420p10le",
            "color_range": "tv", "color_space": "bt2020nc",
            "color_transfer": "smpte2084", "color_primaries": "bt2020",
            "width": 1920, "height": 1080,
            "avg_frame_rate": "30000/1001", "r_frame_rate": "30000/1001",
            "time_base": "1/30000", "start_time": "0.000000",
            "duration": "0.100100", "nb_frames": "3",
        }
        frames = [{
            "best_effort_timestamp": value,
            "best_effort_timestamp_time": text,
            "key_frame": int(index == 0), "pict_type": "I" if index == 0 else "P",
        } for index, (value, text) in enumerate((
            (0, "0.000000"), (1001, "0.033367"), (2002, "0.066733"),
        ))]
        completed = subprocess.CompletedProcess(
            [], 0, stdout=json.dumps({"streams": [stream], "frames": frames}),
            stderr="",
        )
        row = {
            "video_id": "1" * 32, "video_path": Path("video.mp4"),
            "source_frame_count": 3,
            "sparse_source_frame_rate": 30000 / 1001,
            "source_receipt": {"probed_frame_rate": 30000 / 1001},
            "audit": {"width": 1920, "height": 1080},
            "download": {"sha256": "a" * 64},
        }
        with mock.patch.object(full.subprocess, "run", return_value=completed):
            timing = full._probe_source_timing(Path("ffprobe"), row)
        self.assertEqual(timing["source_frame_rate"]["rational"], "30000/1001")
        self.assertEqual(timing["time_base"]["rational"], "1/30000")
        self.assertEqual(timing["unique_frame_duration_ticks"], [1001])
        self.assertEqual(timing["frames"][2]["timestamp_seconds_numerator"], 1001)
        self.assertEqual(timing["frames"][2]["timestamp_seconds_denominator"], 15000)

    def test_timing_probe_rejects_nonmonotonic_pts(self):
        stream = {
            "codec_name": "hevc", "pix_fmt": "yuv420p10le",
            "color_range": "tv", "color_space": "bt2020nc",
            "color_transfer": "smpte2084", "color_primaries": "bt2020",
            "width": 1920, "height": 1080, "avg_frame_rate": "30/1",
            "r_frame_rate": "30/1", "time_base": "1/30000", "nb_frames": "2",
        }
        completed = subprocess.CompletedProcess([], 0, stdout=json.dumps({
            "streams": [stream],
            "frames": [
                {"best_effort_timestamp": 1001, "key_frame": 1, "pict_type": "I"},
                {"best_effort_timestamp": 1001, "key_frame": 0, "pict_type": "P"},
            ],
        }), stderr="")
        row = {
            "video_id": "1" * 32, "video_path": Path("video.mp4"),
            "source_frame_count": 2, "sparse_source_frame_rate": 30.0,
            "source_receipt": {"probed_frame_rate": 30.0},
            "audit": {"width": 1920, "height": 1080},
            "download": {"sha256": "a" * 64},
        }
        with mock.patch.object(full.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "nonmonotonic"):
                full._probe_source_timing(Path("ffprobe"), row)

    def test_full_decode_has_no_sparse_select_filter(self):
        command = full.raw_native.full_decode_command(
            Path("ffmpeg"), Path("source.mp4"), "format=gbrpf32le"
        )
        self.assertNotIn("select=", " ".join(map(str, command)))
        self.assertEqual(command[command.index("-fps_mode") + 1], "passthrough")

    def test_dataset_manifest_keeps_one_sequence_per_video(self):
        results = []
        for index, count in enumerate(TRAIN_COUNTS):
            results.append({
                "clip": f"clip-{index}", "frames": count,
                "source_frames": count, "label_frames": 5,
                "label_frame_ids": [1, 2, 3, 4, 5], "split": "training",
                "capture_group_id": f"{index + 1000:064x}",
                "video_id": f"{index + 1:032x}", "source_frame_rate": 30.0,
                "source_frame_rate_rational": "30/1",
                "source_time_base_rational": "1/15360",
                "source_timing_content_sha256": f"{index + 2000:064x}",
                "source_cut_candidate_count": index % 2,
            })
        manifest = full._dataset_manifest(
            "training", results,
            {"selection_manifest": "selection.json",
             "selection_manifest_sha256": "a" * 64,
             "download_receipt": "receipt.json",
             "download_receipt_sha256": "b" * 64},
            {"sparse_bootstrap_manifest": "sparse.json"},
            "c" * 64, full.CUT_THRESHOLD,
        )
        self.assertEqual(len(manifest["sequences"]), 12)
        self.assertEqual(manifest["frame_count"], 3088)
        self.assertEqual(manifest["source_frame_count"], 3088)
        self.assertEqual(manifest["label_frame_count"], 60)
        self.assertEqual(manifest["full_cadence_contract"],
                         full.FULL_CADENCE_CONTRACT)

    def test_interrupted_resume_rejects_a_changed_cut_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            split_root = Path(directory)
            destination, row, _, conversion_hash = _clip_fixture(split_root)
            with mock.patch.object(
                    full.native_hdr_capture, "validate_clip",
                    return_value={"frame_count": row["source_frame_count"]}), \
                    mock.patch.object(full.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "stale"):
                    full._prepare_clip(
                        row, split_root, Path("ffmpeg"), {}, conversion_hash,
                        8, 4, full.CUT_THRESHOLD + 0.01,
                    )
            self.assertTrue(destination.is_dir())
            popen.assert_not_called()

    def test_raw_cache_reuses_native_frames_when_labels_and_threshold_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_root = root / "training"
            destination, row, _, conversion_hash = _clip_fixture(split_root)
            source = root / "source.mp4"
            source.write_bytes(b"fixture source")
            row["video_path"] = source
            row["audit"] = {"width": 8, "height": 4}
            cache = root / "cache"
            conversion = {"contract": "fixture-color"}
            first_events = []
            with mock.patch.object(
                    full.native_hdr_capture, "validate_clip",
                    return_value={"frame_count": row["source_frame_count"]}):
                first = full._prepare_clip(
                    row, split_root, Path("ffmpeg"), conversion,
                    conversion_hash, 8, 4, full.CUT_THRESHOLD, cache,
                    cache_observer=first_events.append,
                )
            self.assertEqual(first["status"], "reused")
            self.assertEqual(
                first_events[0]["status"], "unseeded-existing"
            )
            code_identity = full.raw_native.code_identity()
            runtime_identity = full.raw_native.runtime_identity()
            raw_identity = full._raw_native_identity(
                row, conversion_hash, 8, 4,
                code_identity=code_identity,
                runtime_identity_value=runtime_identity,
            )
            _, published = full._publish_raw_native_from_clip(
                full.artifact_cache.DirectoryArtifactCache(cache),
                raw_identity, destination, row, conversion_hash,
                split_root, code_identity,
            )
            self.assertEqual(published["status"], "published")
            shutil.rmtree(destination)
            row["label_frame_ids"] = [0, 2]
            second_events = []
            with (mock.patch.object(
                      full.native_hdr_capture, "validate_clip",
                      return_value={"frame_count": row["source_frame_count"]},
                  ), mock.patch.object(full.subprocess, "Popen") as popen):
                second = full._prepare_clip(
                    row, split_root, Path("ffmpeg"), conversion,
                    conversion_hash, 8, 4, 0.30, cache,
                    cache_observer=second_events.append,
                )
            self.assertEqual(second["status"], "raw-cache-reused")
            self.assertEqual(second_events[0]["status"], "hit")
            self.assertEqual(
                published["key_sha256"],
                second_events[0]["key_sha256"],
            )
            self.assertTrue(destination.is_dir())
            popen.assert_not_called()
            labels = json.loads(
                (destination / "label_frames.json").read_text(encoding="utf-8")
            )
            cuts = json.loads(
                (destination / "source_cut_evidence.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(labels["frame_ids"], [0, 2])
            self.assertEqual(cuts["threshold"], 0.30)
            self.assertEqual(cuts["cut_candidate_count"], 0)
            payloads = list((cache / "objects").glob("*/*/payload"))
            self.assertEqual(len(payloads), 1)
            self.assertTrue(
                (payloads[0] / "raw_native_hdr_frames.json").is_file()
            )
            self.assertFalse((payloads[0] / "meta.json").exists())
            self.assertFalse((payloads[0] / "label_frames.json").exists())
            self.assertFalse(
                (payloads[0] / "source_cut_evidence.json").exists()
            )
            cache_manifest = json.loads(
                (payloads[0].parent / "cache_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(set(cache_manifest["identity"]["code"]), {
                "depth_input_color",
                "native_color_functions",
                "native_hdr_capture",
                "native_raw_contract",
                "native_runtime_identity",
            })

    def test_raw_cache_identity_binds_cut_analysis_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            _, row, _, conversion_hash = _clip_fixture(Path(directory))
            row["audit"] = {"width": 8, "height": 4}
            code = full.raw_native.code_identity()
            first = full.raw_native.identity(
                row=row, conversion_hash=conversion_hash,
                width=8, height=4,
                cut_analysis_width=160, cut_analysis_height=90,
                code_identity=code,
            )
            second = full.raw_native.identity(
                row=row, conversion_hash=conversion_hash,
                width=8, height=4,
                cut_analysis_width=320, cut_analysis_height=180,
                code_identity=code,
            )
            self.assertNotEqual(
                full.artifact_cache.DirectoryArtifactCache.key(first),
                full.artifact_cache.DirectoryArtifactCache.key(second),
            )

    def test_raw_cache_publication_rehashes_linked_frame_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_root = root / "training"
            destination, row, _, conversion_hash = _clip_fixture(split_root)
            source = root / "source.mp4"
            source.write_bytes(b"fixture source")
            row["video_path"] = source
            row["audit"] = {"width": 8, "height": 4}
            model = (
                destination / full.native_hdr_capture.MODEL_SOURCE_DIRECTORY /
                "frame_00001.scrgb16"
            )
            model.write_bytes(b"x" * model.stat().st_size)
            code_identity = full.raw_native.code_identity()
            raw_identity = full._raw_native_identity(
                row, conversion_hash, 8, 4,
                code_identity=code_identity,
                runtime_identity_value=full.raw_native.runtime_identity(),
            )
            with self.assertRaisesRegex(
                    RuntimeError, "receipt differs from frame manifest"):
                full._publish_raw_native_from_clip(
                    full.artifact_cache.DirectoryArtifactCache(root / "cache"),
                    raw_identity, destination, row, conversion_hash,
                    split_root, code_identity,
                )

    def test_raw_cache_materialization_rehashes_original_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_root = root / "training"
            destination, row, _, conversion_hash = _clip_fixture(split_root)
            source = root / "source.mp4"
            source.write_bytes(b"fixture source")
            row["video_path"] = source
            row["audit"] = {"width": 8, "height": 4}
            cache = root / "cache"
            with mock.patch.object(
                    full.native_hdr_capture, "validate_clip",
                    return_value={"frame_count": row["source_frame_count"]}):
                full._prepare_clip(
                    row, split_root, Path("ffmpeg"), {}, conversion_hash,
                    8, 4, full.CUT_THRESHOLD, cache,
                )
            code_identity = full.raw_native.code_identity()
            raw_identity = full._raw_native_identity(
                row, conversion_hash, 8, 4,
                code_identity=code_identity,
                runtime_identity_value=full.raw_native.runtime_identity(),
            )
            full._publish_raw_native_from_clip(
                full.artifact_cache.DirectoryArtifactCache(cache),
                raw_identity, destination, row, conversion_hash,
                split_root, code_identity,
            )
            shutil.rmtree(destination)
            original_copy = shutil.copy2
            corrupted = False

            def corrupt_then_copy(source_path, destination_path, *args, **kwargs):
                nonlocal corrupted
                source_path = Path(source_path)
                if not corrupted and source_path.suffix == ".scrgb16":
                    source_path.write_bytes(
                        b"x" * source_path.stat().st_size
                    )
                    corrupted = True
                return original_copy(
                    source_path, destination_path, *args, **kwargs
                )

            with mock.patch.object(
                    full.shutil, "copy2", side_effect=corrupt_then_copy):
                with self.assertRaisesRegex(RuntimeError, "hash differs"):
                    full._prepare_clip(
                        row, split_root, Path("ffmpeg"), {}, conversion_hash,
                        8, 4, full.CUT_THRESHOLD, cache,
                    )
            self.assertFalse(destination.exists())

    def test_native_cache_rejects_test_before_source_path_access(self):
        row = {"split": "test"}
        with self.assertRaisesRegex(RuntimeError, "train/development only"):
            full._prepare_clip(
                row, Path("unused"), Path("ffmpeg"), {}, "1" * 64,
                8, 4, full.CUT_THRESHOLD, Path("cache"),
            )

    def test_full_audit_rejects_rehashed_stale_sidecar_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            destination, row, sequence, conversion_hash = _clip_fixture(
                Path(directory)
            )
            manifest_path = destination / full.native_hdr_capture.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_video"]["content_id"] = "7" * 64
            _write(manifest_path, manifest)
            with mock.patch.object(
                    full.native_hdr_capture, "validate_clip",
                    return_value={"frame_count": row["source_frame_count"]}):
                with self.assertRaisesRegex(
                        RuntimeError, "source-video sidecar identity differs"):
                    full_audit._clip_audit(
                        destination, sequence, verify_content=False,
                        production_id=f"{full.PRODUCTION_PREFIX}_training",
                        conversion_hash=conversion_hash,
                        cut_threshold=full.CUT_THRESHOLD,
                        source_row={
                            "content_id": row["content_id"],
                            "bytes": row["download"]["bytes"],
                            "sha256": row["download"]["sha256"],
                        },
                    )

    def test_full_audit_rejects_rehashed_cut_count_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            destination, row, sequence, conversion_hash = _clip_fixture(
                Path(directory)
            )
            cuts_path = destination / "source_cut_evidence.json"
            cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
            cuts["cut_candidate_count"] = 0
            cuts.pop("content_sha256")
            cuts = _with_content_hash(cuts)
            _write(cuts_path, cuts)
            metadata_path = destination / "meta.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["source_cut_evidence_sha256"] = _sha(cuts_path)
            metadata["source_cut_evidence_content_sha256"] = cuts[
                "content_sha256"
            ]
            _write(metadata_path, metadata)
            manifest_path = destination / full.native_hdr_capture.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_video"]["source_cut_evidence_sha256"] = _sha(
                cuts_path
            )
            manifest["source_video"][
                "source_cut_evidence_content_sha256"
            ] = cuts["content_sha256"]
            _write(manifest_path, manifest)
            with mock.patch.object(
                    full.native_hdr_capture, "validate_clip",
                    return_value={"frame_count": row["source_frame_count"]}):
                with self.assertRaisesRegex(
                        RuntimeError, "candidate count differs"):
                    full_audit._clip_audit(
                        destination, sequence, verify_content=False,
                        production_id=f"{full.PRODUCTION_PREFIX}_training",
                        conversion_hash=conversion_hash,
                        cut_threshold=full.CUT_THRESHOLD,
                        source_row={
                            "content_id": row["content_id"],
                            "bytes": row["download"]["bytes"],
                            "sha256": row["download"]["sha256"],
                        },
                    )


if __name__ == "__main__":
    unittest.main()
