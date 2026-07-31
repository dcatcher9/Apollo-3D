#!/usr/bin/env python3
"""Focused unit tests for the continuous whole-clip orchestrator."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import struct
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from fractions import Fraction
from pathlib import Path
from unittest import mock
from PIL import Image


sys.path.insert(0, os.path.dirname(__file__))
import run_whole_clip as whole  # noqa: E402


SHOWINFO_VFR = """
[Parsed_showinfo_0 @ 000001] config in time_base: 1/1000, frame_rate: 25/1
[Parsed_showinfo_0 @ 000001] n:   0 pts:    100 pts_time:0.1 duration:40 duration_time:0.04 fmt:rgb24
[Parsed_showinfo_0 @ 000001] n:   1 pts:    140 pts_time:0.14 duration:60 duration_time:0.06 fmt:rgb24
[Parsed_showinfo_0 @ 000001] n:   2 pts:    200 pts_time:0.2 duration:40 duration_time:0.04 fmt:rgb24
"""


class WholeClipTimelineTests(unittest.TestCase):
    @staticmethod
    def _hdr_summary(side_data=None, transfer="smpte2084"):
        common = {
            "pix_fmt": "yuv420p10le",
            "color_range": "tv",
            "color_space": "bt2020nc",
            "color_transfer": transfer,
            "color_primaries": "bt2020",
        }
        return {
            "streams": [{
                **common,
                "time_base": "1/1000",
                "avg_frame_rate": "24/1",
                "side_data_list": list(side_data or []),
            }],
            "frames": [{
                **common,
                "side_data_list": list(side_data or []),
            }],
        }

    def test_showinfo_keeps_integer_pts_and_detects_vfr(self):
        time_base, rows = whole.parse_showinfo(SHOWINFO_VFR)
        self.assertEqual(time_base, Fraction(1, 1000))
        timeline = whole.build_video_timeline(time_base, rows, {"fps": 25.0})

        self.assertEqual(timeline["frame_count"], 3)
        self.assertEqual(timeline["first_pts"], 100)
        self.assertAlmostEqual(timeline["first_pts_time"], 0.1)
        self.assertTrue(timeline["variable_frame_rate"])
        self.assertEqual(
            [frame["duration"] for frame in timeline["frames"]],
            [40, 60, 40],
        )
        self.assertEqual(
            [frame["frame_id"] for frame in timeline["frames"]],
            ["0000000001", "0000000002", "0000000003"],
        )

    def test_showinfo_rejects_non_monotonic_presentation_pts(self):
        text = SHOWINFO_VFR.replace(
            "n:   2 pts:    200 pts_time:0.2",
            "n:   2 pts:    140 pts_time:0.14",
        )
        with self.assertRaisesRegex(whole.WholeClipError, "strictly increasing"):
            whole.parse_showinfo(text)

    def test_frame_directory_timeline_matches_native_frame_id_rules(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("plain.png", "frame_00020.jpg", "frame_00010.png"):
                (root / name).write_bytes(b"x")
            frames = whole.frame_directory_files(root)
            timeline = whole.build_frame_directory_timeline(frames, 30000 / 1001)

        self.assertEqual(
            [row["frame_id"] for row in timeline["frames"]],
            ["00010", "00020", "00002"],
        )

    def test_frame_directory_rejects_fallback_id_collision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Numeric identities sort first. The non-numeric frame falls back to index 1,
            # colliding with the literal numeric identity "00001".
            (root / "frame_00001.png").write_bytes(b"x")
            (root / "plain.png").write_bytes(b"x")
            frames = whole.frame_directory_files(root)
            with self.assertRaisesRegex(whole.WholeClipError, "duplicate"):
                whole.build_frame_directory_timeline(frames, 30)

    def test_frame_directory_requires_an_explicit_or_metadata_fps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(whole.WholeClipError, "requires --fps"):
                whole.frame_directory_fps(root, None)
            (root / "meta.json").write_text(
                '{"frame_rate":"30000/1001"}', encoding="utf-8")
            self.assertAlmostEqual(
                whole.frame_directory_fps(root, None), 30000 / 1001)
            self.assertEqual(whole.frame_directory_fps(root, 24), 24)

    def test_concat_uses_vfr_durations_and_per_file_source_clock(self):
        _, rows = whole.parse_showinfo(SHOWINFO_VFR)
        timeline = whole.build_video_timeline(Fraction(1, 1000), rows, {"fps": 25})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = []
            for index in range(3):
                path = root / f"sbs_{index + 1}.png"
                path.write_bytes(b"x")
                frames.append(path)
            concat = root / "frames.ffconcat"
            whole.write_concat_file(concat, frames, timeline)
            text = concat.read_text(encoding="utf-8")

        self.assertIn("duration 0.040000000000", text)
        self.assertIn("duration 0.060000000000", text)
        self.assertEqual(text.count("file '"), 3)
        self.assertEqual(text.count("option framerate 1000/1"), 3)

    def test_hdr_and_rotation_fail_closed(self):
        with self.assertRaisesRegex(whole.WholeClipError, "rotated"):
            whole.reject_unsupported_video({"rotate": 90, "pix_fmt": "yuv420p"})
        with self.assertRaisesRegex(whole.WholeClipError, "implicitly"):
            whole.reject_unsupported_video({
                "rotate": 0,
                "pix_fmt": "yuv420p",
                "source_size": [1920, 1080],
                "size": [1080, 1920],
            })
        with self.assertRaisesRegex(whole.WholeClipError, "HDR"):
            whole.reject_unsupported_video({
                "rotate": 0,
                "pix_fmt": "yuv420p10le(tv, bt2020nc/bt2020/smpte2084)",
            })

    def test_ffprobe_color_contract_accepts_static_pq_and_hlg(self):
        mastering = {
            "side_data_type": "Mastering display metadata",
            "red_x": "34000/50000",
            "max_luminance": "10000000/10000",
        }
        cll = {
            "side_data_type": "Content light level metadata",
            "max_content": 1000,
            "max_average": 400,
        }
        pq = whole.classify_video_color(self._hdr_summary([mastering, cll]))
        self.assertEqual(pq["mode"], "hdr-pq")
        self.assertEqual(pq["mastering_display"]["red_x"], "17/25")
        self.assertEqual(pq["content_light_level"]["max_content"], 1000)
        hlg = whole.classify_video_color(
            self._hdr_summary(transfer="arib-std-b67"))
        self.assertEqual(hlg["mode"], "hdr-hlg")
        self.assertIsNone(hlg["mastering_display"])

    def test_ffprobe_color_contract_rejects_dynamic_and_ambiguous_hdr(self):
        dovi = {
            "side_data_type": "DOVI configuration record",
            "dv_profile": 8,
        }
        with self.assertRaisesRegex(whole.WholeClipError, "Dolby Vision"):
            whole.classify_video_color(self._hdr_summary([dovi]))
        hdr10_plus = {
            "side_data_type": "HDR Dynamic Metadata SMPTE2094-40 (HDR10+)",
        }
        with self.assertRaisesRegex(whole.WholeClipError, "HDR10"):
            whole.classify_video_color(self._hdr_summary([hdr10_plus]))
        high_bit_sdr = self._hdr_summary(transfer="bt709")
        with self.assertRaisesRegex(whole.WholeClipError, "high-bit-depth"):
            whole.classify_video_color(high_bit_sdr)

    def test_full_timestamp_probe_detects_late_dynamic_hdr_metadata(self):
        summary = self._hdr_summary()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "timeline.csv"
            output.write_text(
                "0,0,40,Mastering display metadata\n"
                "40,40,40,HDR Dynamic Metadata SMPTE2094-40 (HDR10+)\n",
                encoding="utf-8",
            )
            with mock.patch.object(whole, "run_probe_command"):
                with self.assertRaisesRegex(whole.WholeClipError, "dynamic HDR"):
                    whole.probe_video_timeline(
                        "ffprobe", root / "source.mkv", summary,
                        output, root / "probe.log")


class WholeClipProgressTests(unittest.TestCase):
    def test_live_progress_retries_transient_windows_sharing_violation(self):
        child = mock.Mock()
        child.poll.return_value = None
        child.log_path = Path("native.log")
        progress = {
            "schema": 1,
            "processed_count": 1,
            "status": "running",
        }
        with mock.patch.object(
                whole,
                "read_json_object",
                side_effect=[PermissionError("sharing violation"), progress]
        ) as read, mock.patch.object(whole.time, "sleep"):
            result = whole.wait_for_progress(
                Path("follow_progress.json"),
                "processed_count",
                1,
                child,
                timeout_seconds=1.0,
                expected={"schema": 1},
            )

        self.assertEqual(result, progress)
        self.assertEqual(read.call_count, 2)
        self.assertTrue(
            read.call_args.kwargs["allow_transient_permission_error"])

    def test_pipeline_prefers_the_matching_build_local_ffmpeg(self):
        with tempfile.TemporaryDirectory() as temporary:
            build_dir = Path(temporary)
            tools = build_dir / "tools"
            tools.mkdir()
            ffmpeg = tools / "ffmpeg.exe"
            ffmpeg.write_bytes(b"test")
            with mock.patch.object(whole, "resolve_ffmpeg") as fallback:
                resolved = whole.resolve_pipeline_ffmpeg(build_dir)

        self.assertEqual(resolved, os.fspath(ffmpeg.resolve()))
        fallback.assert_not_called()

    def test_pipeline_ffmpeg_falls_back_when_build_has_no_tool(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
                whole,
                "resolve_ffmpeg",
                return_value="system-ffmpeg",
        ) as fallback:
            resolved = whole.resolve_pipeline_ffmpeg(Path(temporary))

        self.assertEqual(resolved, "system-ffmpeg")
        fallback.assert_called_once_with()


class SceneControllerSourceTimeTests(unittest.TestCase):
    @staticmethod
    def _video_timeline(pts):
        rows = [
            {
                "n": index,
                "pts": value,
                "pts_time_text": str(value / 1000),
                "duration": None,
                "duration_time_text": None,
            }
            for index, value in enumerate(pts)
        ]
        return whole.build_video_timeline(
            Fraction(1, 1000),
            rows,
            {"fps": 30.0},
        )

    def test_vfr_source_time_preserves_exact_presentation_deltas(self):
        timeline = self._video_timeline([100, 140, 240, 280])
        records = list(whole.build_scene_controller_source_time(timeline))
        header = records[0]
        self.assertEqual(header["schema"], 2)
        self.assertEqual(
            header["clock"],
            whole.SCENE_CONTROLLER_SOURCE_TIME_CLOCK,
        )
        self.assertEqual(header["time_base"], {"num": 1, "den": 1000})
        self.assertEqual(
            [frame["pts_ticks"] for frame in records[1:]],
            [100, 140, 240, 280],
        )
        self.assertEqual(
            [frame["duration_ticks"] for frame in records[1:]],
            [40, 100, 40, 40],
        )

    def test_cfr_source_time_is_deterministic_and_hashes_exact_jsonl_bytes(self):
        timeline = whole.build_frame_directory_timeline(
            [
                Path("frame_00001.png"),
                Path("frame_00002.png"),
                Path("frame_00003.png"),
            ],
            30.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timeline_path = root / whole.TIMELINE_NAME
            sidecar_path = (
                root / whole.SCENE_CONTROLLER_SOURCE_TIME_NAME
            )
            whole.write_json_atomic(timeline_path, timeline)
            first = whole.write_scene_controller_source_time(
                sidecar_path, timeline)
            first_bytes = sidecar_path.read_bytes()
            second = whole.write_scene_controller_source_time(
                sidecar_path, timeline)
            second_bytes = sidecar_path.read_bytes()
            records = [
                json.loads(line)
                for line in sidecar_path.read_text(
                    encoding="utf-8").splitlines()
            ]
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first, second)
        self.assertEqual(first["sha256"], hashlib.sha256(first_bytes).hexdigest())
        self.assertEqual(first["frame_count"], 3)
        self.assertEqual(first["file_bytes"], len(first_bytes))
        self.assertEqual(first["disk_reservation_bytes"], len(first_bytes))
        self.assertAlmostEqual(first["total_elapsed_seconds"], 2 / 30)
        self.assertEqual(
            [frame["pts_ticks"] for frame in records[1:]],
            [0, 1, 2],
        )

    def test_elapsed_attestation_mirrors_native_sequential_double_sum(self):
        timeline = whole.build_frame_directory_timeline(
            [
                Path(f"frame_{index:05d}.png")
                for index in range(1, 1002)
            ],
            30.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timeline_path = root / whole.TIMELINE_NAME
            sidecar_path = (
                root / whole.SCENE_CONTROLLER_SOURCE_TIME_NAME
            )
            whole.write_json_atomic(timeline_path, timeline)
            provenance = whole.write_scene_controller_source_time(
                sidecar_path,
                timeline,
            )
            records = [
                json.loads(line)
                for line in sidecar_path.read_text(
                    encoding="utf-8").splitlines()
            ]

        native_style_total = 0.0
        previous_pts = None
        for frame in records[1:]:
            delta = (
                0 if previous_pts is None
                else frame["pts_ticks"] - previous_pts
            )
            native_style_total += float(delta) / 30.0
            previous_pts = frame["pts_ticks"]
        self.assertEqual(
            provenance["total_elapsed_seconds"],
            native_style_total,
        )
        self.assertNotEqual(
            provenance["total_elapsed_seconds"],
            float(Fraction(1000, 30)),
        )

    def test_source_time_rejects_timeline_identity_and_clock_corruption(self):
        valid = self._video_timeline([0, 40])
        mutations = []
        wrong_count = copy.deepcopy(valid)
        wrong_count["frame_count"] = True
        mutations.append(("exact timeline", wrong_count))
        wrong_index = copy.deepcopy(valid)
        wrong_index["frames"][0]["index"] = False
        mutations.append(("indices", wrong_index))
        duplicate_pts = copy.deepcopy(valid)
        duplicate_pts["frames"][1]["pts"] = 0
        mutations.append(("increase strictly", duplicate_pts))
        wrong_time_base = copy.deepcopy(valid)
        wrong_time_base["time_base"]["num"] = True
        mutations.append(("exact JSON integers", wrong_time_base))
        bad_identity = copy.deepcopy(valid)
        bad_identity["frames"][0]["frame_id"] = "frame-one"
        mutations.append(("frame identity", bad_identity))
        bad_duration = copy.deepcopy(valid)
        bad_duration["frames"][0]["duration"] = 0
        mutations.append(("positive signed 64-bit", bad_duration))
        oversized_delta = copy.deepcopy(valid)
        oversized_delta["frames"][1]["pts"] = whole.INT64_MAX + 1
        mutations.append(("signed 64-bit", oversized_delta))
        oversized_clock = copy.deepcopy(valid)
        oversized_clock["time_base"] = {
            "num": whole.INT64_MAX + 1,
            "den": 1,
        }
        mutations.append(("signed 64-bit clock", oversized_clock))
        for message, timeline in mutations:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                        whole.WholeClipError, message):
                    list(whole.build_scene_controller_source_time(timeline))

    def test_follow_source_time_uses_native_global_sequence_identities(self):
        timeline = self._video_timeline([0, 40])
        timeline["frames"][0]["frame_id"] = "00001"
        timeline["frames"][1]["frame_id"] = "00002"

        legacy_records = list(
            whole.build_scene_controller_source_time(timeline))
        records = list(whole.build_scene_controller_source_time(
            timeline,
            follow_frame_ids=True,
        ))
        self.assertEqual(
            [record["frame_id"] for record in legacy_records[1:]],
            ["00001", "00002"],
        )
        self.assertEqual(
            [record["frame_id"] for record in records[1:]],
            ["0000000001", "0000000002"],
        )
        self.assertEqual(
            [frame["frame_id"] for frame in timeline["frames"]],
            ["00001", "00002"],
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / whole.SCENE_CONTROLLER_SOURCE_TIME_NAME
            provenance = whole.write_scene_controller_source_time(
                path,
                timeline,
                follow_frame_ids=True,
            )
            written = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(
            [record["frame_id"] for record in written[1:]],
            ["0000000001", "0000000002"],
        )
        self.assertEqual(provenance["frame_count"], 2)


class WholeClipNativeContractTests(unittest.TestCase):
    @staticmethod
    def _capabilities(*, state_schema=whole.SCENE_CACHE_STATE_SCHEMA,
                      state_words=whole.SCENE_CACHE_STATE_WORDS):
        return {
            "schema": 1,
            "native_whole_clip": {
                "artifact_modes": ["adaptive", "conversion"],
                "source_formats": ["png", "bmp", "pfm"],
                "follow_protocol_schema": 1,
                "follow_global_first_sequence": True,
                "adaptive_state_schema": whole.ADAPTIVE_TRACE_SCHEMA,
                "scene_controller_trace": {
                    "trace_schema":
                        whole.scene_controller_trace.TRACE_SCHEMA,
                    "controller_schema":
                        whole.scene_controller_trace.controller_contract.SCHEMA_VERSION,
                    "rule_revision":
                        whole.scene_controller_trace.controller_contract.RULE_REVISION,
                    "ordered_abi_hash":
                        whole.scene_controller_trace.controller_contract.ORDERED_ABI_HASH,
                    "backends": ["off", "shadow_rules"],
                    "active_roi_authority": False,
                    "file": whole.scene_controller_trace.TRACE_NAME,
                    "transports": [
                        whole.scene_controller_trace.TRACE_TRANSPORT,
                        whole.scene_controller_trace.ATOMIC_TRACE_TRANSPORT,
                    ],
                    "atomic_header_file":
                        whole.scene_controller_trace.ATOMIC_HEADER_NAME,
                    "atomic_frame_file":
                        whole.scene_controller_trace.ATOMIC_FRAME_NAME,
                    "global_out_word_count":
                        len(whole.scene_controller_trace.GLOBAL_OUT_CONTRACT),
                    "rule_state_word_count":
                        len(whole.scene_controller_trace.RULE_STATE_CONTRACT),
                    "source_time_override":
                        whole.SCENE_CONTROLLER_SOURCE_TIME_CLOCK,
                },
                "scene_cache_contract_schema":
                    whole.SCENE_CACHE_CONTRACT_SCHEMA,
                "scene_cache_packed_sbs_contract": True,
                "scene_cache_depth": {
                    "dtype": "float32-le",
                    "layout": "row-major",
                    "dxgi_format": "R32_FLOAT",
                    "dimensions": "per-frame-metadata",
                    "bytes_per_frame": None,
                },
                "scene_cache_frame_metadata": {
                    "schema": whole.SCENE_CACHE_METADATA_SCHEMA,
                    "word_count": whole.SCENE_CACHE_METADATA_WORDS,
                    "roi_transform_word_offset":
                        whole.SCENE_CACHE_METADATA_ROI_OFFSET,
                    "roi_transform_word_count": whole.SCENE_CACHE_ROI_WORDS,
                    "roi_transform_contract_schema":
                        whole.FRAME_ROI_TRANSFORM_SCHEMA,
                },
                "scene_cache_state": {
                    "schema": state_schema,
                    "subject_word_count": whole.SCENE_CACHE_SUBJECT_WORDS,
                    "depth_frame_state_word_count":
                        whole.SCENE_CACHE_DEPTH_FRAME_STATE_WORDS,
                    "word_count": state_words,
                    "dtype": "uint32-le",
                },
                "scene_plan": {
                    "schema": 1,
                    "version": "scene-plan-v1",
                    "one_scene_per_replay": True,
                    "absolute_pop_strength": True,
                    "source_pixel_zero_anchor": True,
                },
                "render_cache_follow": True,
                "render_skips_tensorrt": True,
                "whole_clip_inference_attestation": {
                    "depth_inference_enabled": True,
                    "scheduled_depth_update_count": True,
                    "tensorrt_enqueue_count": True,
                },
                "atomic_sbs_publication": True,
            },
        }

    def _write_contract(
        self,
        root,
        mode="conversion",
        count=2,
        *,
        frame_ids=None,
        scene_controller_backend="shadow_rules",
        follow_mode=None,
    ):
        source_time = {
            "file": whole.SCENE_CONTROLLER_SOURCE_TIME_NAME,
            "sha256": "1" * 64,
            "schema": whole.SCENE_CONTROLLER_SOURCE_TIME_SCHEMA,
            "clock": whole.SCENE_CONTROLLER_SOURCE_TIME_CLOCK,
            "frame_count": count,
            "total_elapsed_seconds": float(max(0, count - 1)),
        }
        (root / whole.TRACE_NAME).write_text(
            '{"schema":2,"frame_id":"00001"}\n', encoding="utf-8")
        if frame_ids is None:
            frame_ids = [
                f"{index + 1:05d}" for index in range(count)
            ]
        self.assertEqual(len(frame_ids), count)
        if scene_controller_backend == "shadow_rules":
            controller = whole.scene_controller_trace
            rule_state = []
            for field in controller.RULE_STATE_CONTRACT:
                value = 0
                if field["name"] == "schema_version":
                    value = controller.controller_contract.SCHEMA_VERSION
                elif field["name"] == "backend_generation":
                    value = 1
                rule_state.append(
                    value if field["type"] == "uint32" else float(value)
                )
            header = {
                "record": "header",
                "trace_schema": controller.TRACE_SCHEMA,
                "source": controller.TRACE_SOURCE,
                "capture": controller.TRACE_CAPTURE,
                "backend": "shadow_rules",
                "controller_schema":
                    controller.controller_contract.SCHEMA_VERSION,
                "rule_revision":
                    controller.controller_contract.RULE_REVISION,
                "ordered_abi_hash":
                    controller.controller_contract.ORDERED_ABI_HASH,
                "global_out_fields": list(controller.GLOBAL_OUT_FIELDS),
                "rule_state_fields": list(controller.RULE_STATE_FIELDS),
                "config": {
                    "model": "test-model",
                    "depth_reuse_interval": 1,
                    "active_roi_authority": False,
                },
            }
            frames = [
                {
                    "record": "frame",
                    "frame_id": frame_id,
                    "source_index": index,
                    "depth_updated": True,
                    "snapshot_available": True,
                    "controller_frame_id": index,
                    "backend_generation": 1,
                    "shadow": True,
                    "global_out": [
                        0.0 for _field in controller.GLOBAL_OUT_CONTRACT
                    ],
                    "rule_state": rule_state,
                }
                for index, frame_id in enumerate(frame_ids)
            ]
            (root / controller.TRACE_NAME).write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in [header, *frames]
                ),
                encoding="utf-8",
            )
            scene_controller_descriptor = {
                "enabled": True,
                "backend": "shadow_rules",
                "active_roi_authority": False,
                "transport": controller.TRACE_TRANSPORT,
                "file": controller.TRACE_NAME,
                "header_file": None,
                "frame_file": None,
                "retained_history": True,
                "trace_schema": controller.TRACE_SCHEMA,
                "controller_schema":
                    controller.controller_contract.SCHEMA_VERSION,
                "rule_revision":
                    controller.controller_contract.RULE_REVISION,
                "ordered_abi_hash":
                    controller.controller_contract.ORDERED_ABI_HASH,
                "frame_count": count,
            }
        else:
            self.assertEqual(scene_controller_backend, "off")
            scene_controller_descriptor = {
                "enabled": False,
                "backend": "off",
                "active_roi_authority": False,
                "transport": None,
                "file": None,
                "header_file": None,
                "frame_file": None,
                "retained_history": False,
                "trace_schema": None,
                "controller_schema": (
                    whole.scene_controller_trace.controller_contract.SCHEMA_VERSION
                ),
                "rule_revision": (
                    whole.scene_controller_trace.controller_contract.RULE_REVISION
                ),
                "ordered_abi_hash": (
                    whole.scene_controller_trace.controller_contract.ORDERED_ABI_HASH
                ),
                "frame_count": 0,
            }
        contract = {
            "schema": 1,
            "artifact_mode": mode,
            "source_frame_count": count,
            "inference_mode": "single-pass-tensorrt",
            "depth_inference_enabled": True,
            "scheduled_depth_update_count": count,
            "tensorrt_enqueue_count": count,
            "depth_reuse_interval": 1,
            "model": "test-model",
            "adaptive_state": {
                "file": whole.TRACE_NAME,
                "schema": whole.ADAPTIVE_TRACE_SCHEMA,
                "frame_count": count,
            },
            "scene_controller": scene_controller_descriptor,
            "scene_controller_source_time": {
                "enabled": True,
                "schema": whole.SCENE_CONTROLLER_SOURCE_TIME_SCHEMA,
                "clock": whole.SCENE_CONTROLLER_SOURCE_TIME_CLOCK,
                "file_sha256": source_time["sha256"],
                "frame_count": count,
                "total_elapsed_seconds":
                    source_time["total_elapsed_seconds"],
            },
            "sbs": {
                "enabled": mode == "conversion",
                "file_pattern": (
                    "sbs_<frame-id>.png" if mode == "conversion" else None),
                "frame_count": count if mode == "conversion" else 0,
            },
        }
        if follow_mode is not None:
            contract["resolved_runtime"] = {"follow_mode": follow_mode}
        (root / whole.NATIVE_CONTRACT_NAME).write_text(
            json.dumps(contract), encoding="utf-8")
        return source_time

    def test_conversion_requires_exact_sbs_identity_set(self):
        timeline = {
            "frame_count": 2,
            "frames": [{"frame_id": "00001"}, {"frame_id": "00002"}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_time = self._write_contract(root)
            (root / "sbs_00001.png").write_bytes(b"x")
            (root / "sbs_00002.png").write_bytes(b"x")
            contract = whole.validate_native_outputs(
                root, timeline, "conversion", source_time)
            self.assertEqual(contract["source_frame_count"], 2)
            (root / "sbs_00002.png").unlink()
            with self.assertRaisesRegex(whole.WholeClipError, "identities"):
                whole.validate_native_outputs(
                    root, timeline, "conversion", source_time)

    def test_conversion_accepts_mixed_numeric_and_fallback_identity_order(self):
        timeline = {
            "frame_count": 3,
            "frames": [
                {"frame_id": "00010"},
                {"frame_id": "00020"},
                {"frame_id": "00002"},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_time = self._write_contract(
                root,
                count=3,
                frame_ids=["00010", "00020", "00002"],
            )
            for frame_id in ("00010", "00020", "00002"):
                (root / f"sbs_{frame_id}.png").write_bytes(b"x")
            whole.validate_native_outputs(
                root, timeline, "conversion", source_time)

    def test_followed_analysis_uses_native_sequence_ids_for_controller_trace(self):
        timeline = {
            "frame_count": 3,
            "frames": [
                {"frame_id": "00010"},
                {"frame_id": "00020"},
                {"frame_id": "00002"},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_time = self._write_contract(
                root,
                mode="adaptive",
                count=3,
                frame_ids=[
                    "0000000001", "0000000002", "0000000003",
                ],
                follow_mode=True,
            )
            whole.validate_native_outputs(
                root, timeline, "adaptive", source_time)

    def test_analysis_rejects_disabled_scene_controller_descriptor(self):
        timeline = {
            "frame_count": 2,
            "frames": [{"frame_id": "00001"}, {"frame_id": "00002"}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_time = self._write_contract(
                root,
                scene_controller_backend="off",
            )
            (root / "sbs_00001.png").write_bytes(b"x")
            (root / "sbs_00002.png").write_bytes(b"x")
            with self.assertRaisesRegex(
                    whole.WholeClipError,
                    "scene-controller backend mismatch"):
                whole.validate_native_outputs(
                    root, timeline, "conversion", source_time)

    def test_native_inference_attestation_rejects_analysis_and_replay_drift(self):
        timeline = {
            "frame_count": 2,
            "frames": [{"frame_id": "00001"}, {"frame_id": "00002"}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_time = self._write_contract(root)
            (root / "sbs_00001.png").write_bytes(b"x")
            (root / "sbs_00002.png").write_bytes(b"x")
            contract_path = root / whole.NATIVE_CONTRACT_NAME
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["tensorrt_enqueue_count"] = 1
            contract_path.write_text(
                json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(
                    whole.WholeClipError,
                    "analysis inference attestation"):
                whole.validate_native_outputs(
                    root, timeline, "conversion", source_time)

        one_frame_analysis = {
            "inference_mode": "single-pass-tensorrt",
            "depth_inference_enabled": True,
            "scheduled_depth_update_count": 1,
            "tensorrt_enqueue_count": 1,
            "depth_reuse_interval": 1,
        }
        whole._validate_native_inference_attestation(
            one_frame_analysis,
            replay=False,
            source_frame_count=1,
        )
        for key in (
                "scheduled_depth_update_count",
                "tensorrt_enqueue_count"):
            wrong_type = dict(one_frame_analysis)
            wrong_type[key] = True
            with self.assertRaisesRegex(
                    whole.WholeClipError,
                    "analysis inference attestation"):
                whole._validate_native_inference_attestation(
                    wrong_type,
                    replay=False,
                    source_frame_count=1,
                )

        replay_contract = {
            "inference_mode": "scene-cache-replay",
            "depth_inference_enabled": False,
            "scheduled_depth_update_count": 0,
            "tensorrt_enqueue_count": 0,
        }
        whole._validate_native_inference_attestation(
            replay_contract,
            replay=True,
            source_frame_count=2,
        )
        for key, wrong_value in (
                ("depth_inference_enabled", 0),
                ("scheduled_depth_update_count", False),
                ("tensorrt_enqueue_count", False)):
            wrong_type = dict(replay_contract)
            wrong_type[key] = wrong_value
            with self.assertRaisesRegex(
                    whole.WholeClipError,
                    "scene replay inference attestation"):
                whole._validate_native_inference_attestation(
                    wrong_type,
                    replay=True,
                    source_frame_count=2,
                )

    def test_source_time_attestation_is_exact_and_replay_is_disabled(self):
        source_time = {
            "file": whole.SCENE_CONTROLLER_SOURCE_TIME_NAME,
            "sha256": "1" * 64,
            "schema": whole.SCENE_CONTROLLER_SOURCE_TIME_SCHEMA,
            "clock": whole.SCENE_CONTROLLER_SOURCE_TIME_CLOCK,
            "frame_count": 2,
            "total_elapsed_seconds": 0.04,
        }
        enabled = {
            "scene_controller_source_time": {
                "enabled": True,
                "schema": whole.SCENE_CONTROLLER_SOURCE_TIME_SCHEMA,
                "clock": whole.SCENE_CONTROLLER_SOURCE_TIME_CLOCK,
                "file_sha256": source_time["sha256"],
                "frame_count": 2,
                "total_elapsed_seconds": 0.04,
            },
        }
        whole._validate_scene_controller_source_time_attestation(
            enabled, source_time, enabled=True)
        for key, value in (
                ("enabled", 1),
                ("schema", 1.0),
                ("file_sha256", "3" * 64),
                ("clock", "forged-presentation-clock"),
                ("frame_count", 2.0),
                ("total_elapsed_seconds", 0)):
            with self.subTest(enabled_field=key):
                mutated = copy.deepcopy(enabled)
                mutated["scene_controller_source_time"][key] = value
                with self.assertRaisesRegex(
                        whole.WholeClipError,
                        "source-time attestation"):
                    whole._validate_scene_controller_source_time_attestation(
                        mutated, source_time, enabled=True)

        disabled = {
            "scene_controller_source_time": {
                "enabled": False,
                "schema": None,
                "clock": None,
                "file_sha256": None,
                "frame_count": 0,
                "total_elapsed_seconds": 0.0,
            },
        }
        whole._validate_scene_controller_source_time_attestation(
            disabled, None, enabled=False)
        disabled["scene_controller_source_time"][
            "total_elapsed_seconds"] = 0
        with self.assertRaisesRegex(
                whole.WholeClipError, "source-time attestation"):
            whole._validate_scene_controller_source_time_attestation(
                disabled, None, enabled=False)

    def test_scene_replay_command_disables_controller_without_sidecar(self):
        import scene_plan

        class ReplayCommandCaptured(RuntimeError):
            pass

        captured = []

        def capture_command(command, _cwd, _log_path):
            captured.append(command)
            raise ReplayCommandCaptured

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            producer = mock.Mock()
            producer.extension = "png"
            with mock.patch.object(
                        scene_plan,
                        "native_scene_plan_document",
                        return_value={"schema": 1}), \
                    mock.patch.object(
                        whole,
                        "LoggedSubprocess",
                        side_effect=capture_command):
                with self.assertRaises(ReplayCommandCaptured):
                    whole.render_cached_scene(
                        sunshine=root / "sunshine.exe",
                        conf=root / "bench.conf",
                        build_dir=root,
                        cache_dir=root / "cache",
                        scene={
                            "scene_id": 1,
                            "start_sequence": 1,
                            "end_sequence_exclusive": 2,
                        },
                        render_producer=producer,
                        bridge=mock.Mock(),
                        encoder=mock.Mock(),
                        work_dir=root / "work",
                        plans_dir=root / "plans",
                        timeout_seconds=1.0,
                        expected_sbs_dimensions=(32, 8),
                    )

        self.assertEqual(len(captured), 1)
        command = captured[0]
        controller_index = command.index("--scene-controller")
        self.assertEqual(command[controller_index + 1], "off")
        self.assertNotIn("--scene-controller-source-time", command)

    def test_browser_qualification_requires_the_analysis_source_time_attestation(self):
        source_time = {
            "file": whole.SCENE_CONTROLLER_SOURCE_TIME_NAME,
            "sha256": "1" * 64,
            "schema": whole.SCENE_CONTROLLER_SOURCE_TIME_SCHEMA,
            "clock": whole.SCENE_CONTROLLER_SOURCE_TIME_CLOCK,
            "frame_count": 2,
            "total_elapsed_seconds": 0.04,
        }
        native_contract = {
            "scene_controller_source_time": {
                "enabled": True,
                "schema": whole.SCENE_CONTROLLER_SOURCE_TIME_SCHEMA,
                "clock": whole.SCENE_CONTROLLER_SOURCE_TIME_CLOCK,
                "file_sha256": source_time["sha256"],
                "frame_count": source_time["frame_count"],
                "total_elapsed_seconds":
                    source_time["total_elapsed_seconds"],
            },
        }
        timeline = {"frame_count": 2}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(
                        whole.scene_controller_eval,
                        "evaluate_generated_clip_trace",
                        return_value={"schema": 1, "pass": True}) as evaluate:
                result = whole.browser_scene_qualification(
                    root / "frames",
                    root / "artifacts",
                    native_contract,
                    timeline,
                    source_time,
                    root,
                )
                self.assertTrue(result["pass"])
                evaluate.assert_called_once_with(
                    root / "frames",
                    root / "artifacts",
                    native_contract,
                    timeline,
                    root / "browser_scene_controller_report.json",
                )

                mismatched = copy.deepcopy(native_contract)
                mismatched["scene_controller_source_time"][
                    "file_sha256"] = "3" * 64
                with self.assertRaisesRegex(
                        whole.WholeClipError,
                        "source-time attestation"):
                    whole.browser_scene_qualification(
                        root / "frames",
                        root / "artifacts",
                        mismatched,
                        timeline,
                        source_time,
                        root,
                    )
                self.assertEqual(evaluate.call_count, 1)

    def test_capability_gate_requires_schema_3_sixteen_word_cache_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "capabilities.json"

            def publish_capabilities(command, _cwd, _log_path):
                Path(command[-1]).write_text(
                    json.dumps(self._capabilities()), encoding="utf-8")

            with mock.patch.object(
                    whole, "run_logged_command",
                    side_effect=publish_capabilities):
                result = whole.query_native_capabilities(
                    root / "sunshine.exe",
                    root,
                    output,
                    root / "capabilities.log",
                )
            state = result["value"]["native_whole_clip"]["scene_cache_state"]
            self.assertEqual(state["schema"], 3)
            self.assertEqual(state["word_count"], 16)
            self.assertEqual(state["dtype"], "uint32-le")

            def publish_stale_capabilities(command, _cwd, _log_path):
                Path(command[-1]).write_text(
                    json.dumps(self._capabilities(
                        state_schema=1, state_words=12)),
                    encoding="utf-8",
                )

            with mock.patch.object(
                    whole, "run_logged_command",
                    side_effect=publish_stale_capabilities):
                with self.assertRaisesRegex(
                        whole.WholeClipError, "required bounded scene replay"):
                    whole.query_native_capabilities(
                        root / "sunshine.exe",
                        root,
                        output,
                        root / "capabilities.log",
                    )

            def publish_wrong_depth_capabilities(command, _cwd, _log_path):
                value = self._capabilities()
                value["native_whole_clip"]["scene_cache_depth"]["dtype"] = "float16"
                Path(command[-1]).write_text(
                    json.dumps(value), encoding="utf-8")

            with mock.patch.object(
                    whole, "run_logged_command",
                    side_effect=publish_wrong_depth_capabilities):
                with self.assertRaisesRegex(
                        whole.WholeClipError, "required bounded scene replay"):
                    whole.query_native_capabilities(
                        root / "sunshine.exe",
                        root,
                        output,
                        root / "capabilities.log",
                    )

    def test_capability_gate_rejects_json_scalar_type_coercions(self):
        mutations = [
            (("schema",), True),
            (("native_whole_clip", "scene_controller_trace",
              "controller_schema"), float(
                  whole.scene_controller_trace.controller_contract.SCHEMA_VERSION)),
            (("native_whole_clip", "scene_controller_trace",
              "active_roi_authority"), 0),
            (("native_whole_clip", "scene_controller_trace",
              "source_time_override"), "wall-clock-delta-v1"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "capabilities.json"
            for path, replacement in mutations:
                with self.subTest(path=path):
                    value = self._capabilities()
                    target = value
                    for key in path[:-1]:
                        target = target[key]
                    target[path[-1]] = replacement

                    def publish_capabilities(
                            command, _cwd, _log_path, document=value):
                        Path(command[-1]).write_text(
                            json.dumps(document), encoding="utf-8")

                    with mock.patch.object(
                            whole, "run_logged_command",
                            side_effect=publish_capabilities):
                        with self.assertRaisesRegex(
                                whole.WholeClipError,
                                "required bounded scene replay"):
                            whole.query_native_capabilities(
                                root / "sunshine.exe",
                                root,
                                output,
                                root / "capabilities.log",
                            )

    def test_bounded_adaptive_transport_cannot_be_overridden(self):
        with self.assertRaisesRegex(
                whole.WholeClipError, "owned by the whole-clip wrapper"):
            whole.validate_native_extra(["--bounded-adaptive-state"])
        with self.assertRaisesRegex(
                whole.WholeClipError, "owned by the whole-clip wrapper"):
            whole.validate_native_extra([
                "--scene-controller-source-time", "forged.json",
            ])

    def test_capability_gate_rejects_roi_and_inference_contract_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "capabilities.json"

            def publish_wrong_roi_schema(command, _cwd, _log_path):
                value = self._capabilities()
                value["native_whole_clip"]["scene_cache_frame_metadata"][
                    "roi_transform_contract_schema"] += 1
                Path(command[-1]).write_text(
                    json.dumps(value), encoding="utf-8")

            with mock.patch.object(
                    whole, "run_logged_command",
                    side_effect=publish_wrong_roi_schema):
                with self.assertRaisesRegex(
                        whole.WholeClipError, "required bounded scene replay"):
                    whole.query_native_capabilities(
                        root / "sunshine.exe",
                        root,
                        output,
                        root / "capabilities.log",
                    )

            def publish_missing_inference_capability(
                    command, _cwd, _log_path):
                value = self._capabilities()
                value["native_whole_clip"][
                    "whole_clip_inference_attestation"
                ]["tensorrt_enqueue_count"] = False
                Path(command[-1]).write_text(
                    json.dumps(value), encoding="utf-8")

            with mock.patch.object(
                    whole, "run_logged_command",
                    side_effect=publish_missing_inference_capability):
                with self.assertRaisesRegex(
                        whole.WholeClipError,
                        "required bounded scene replay"):
                    whole.query_native_capabilities(
                        root / "sunshine.exe",
                        root,
                        output,
                        root / "capabilities.log",
                    )


class WholeClipSceneCacheTests(unittest.TestCase):
    @staticmethod
    def _float_word(value: float) -> int:
        return struct.unpack("<I", struct.pack("<f", value))[0]

    @classmethod
    def _full_frame_transform(
        cls,
        *,
        sequence: int,
        source_width: int,
        source_height: int,
        model_width: int,
        model_height: int,
    ) -> list[int]:
        retained = sequence - 1
        words = [0] * whole.SCENE_CACHE_ROI_WORDS
        words[0] = whole.FRAME_ROI_TRANSFORM_SCHEMA
        words[1] = (1 << 0) | (1 << 1)
        words[2:4] = [retained & 0xffffffff, retained >> 32]
        words[6:10] = [
            source_width,
            source_height,
            model_width,
            model_height,
        ]
        words[10] = model_width * model_height
        full_rect = [
            cls._float_word(0.0),
            cls._float_word(0.0),
            cls._float_word(1.0),
            cls._float_word(1.0),
        ]
        words[12:16] = full_rect
        words[16:20] = full_rect
        words[20:24] = [0, 0, model_width, model_height]
        words[28:30] = [sequence, 0]
        words[30] = sequence % whole.FRAME_ROI_TRANSFORM_BANK_COUNT
        return words

    @classmethod
    def _write_triplet(
        cls,
        root: Path,
        *,
        sequence: int = 1,
        source_width: int = 14,
        source_height: int = 14,
        width: int = 14,
        height: int = 14,
        depth_bytes: bytes | None = None,
        metadata_mutator=None,
        state_mutator=None,
    ) -> None:
        words = [0] * whole.SCENE_CACHE_METADATA_WORDS
        words[0:4] = [
            whole.SCENE_CACHE_METADATA_MAGIC,
            whole.SCENE_CACHE_METADATA_SCHEMA,
            whole.SCENE_CACHE_METADATA_WORDS,
            whole.SCENE_CACHE_METADATA_ROI_OFFSET,
        ]
        words[4:8] = [width, height, width * height, 4]
        words[8:10] = [sequence, 0]
        retained = sequence - 1
        words[10:12] = [retained & 0xffffffff, retained >> 32]
        words[12:16] = [source_width, source_height, width, height]
        words[whole.SCENE_CACHE_METADATA_ROI_OFFSET:] = (
            cls._full_frame_transform(
                sequence=sequence,
                source_width=source_width,
                source_height=source_height,
                model_width=width,
                model_height=height,
            )
        )
        if metadata_mutator is not None:
            metadata_mutator(words)
        stem = f"frame_{sequence:010d}"
        (root / f"{stem}.meta.u32").write_bytes(
            struct.pack(f"<{len(words)}I", *words))
        if depth_bytes is None:
            depth_bytes = b"d" * (width * height * 4)
        (root / f"{stem}.depth.r32f").write_bytes(depth_bytes)
        state_words = [cls._float_word(0.0)] * whole.SCENE_CACHE_STATE_WORDS
        state_words[whole.SCENE_CACHE_SUBJECT_WORDS + 2] = cls._float_word(1.0)
        state_words[whole.SCENE_CACHE_SUBJECT_WORDS + 3] = cls._float_word(1.0)
        if state_mutator is not None:
            state_mutator(state_words)
        (root / f"{stem}.state.u32").write_bytes(
            struct.pack(f"<{len(state_words)}I", *state_words))

    @staticmethod
    def _contract(*, processed_count=1, source_width=14,
                  source_height=14) -> dict:
        return {
            "schema": whole.SCENE_CACHE_CONTRACT_SCHEMA,
            "status": "running",
            "first_sequence": 1,
            "processed_count": processed_count,
            "atomic_frame_publication": True,
            "source": {
                "width": source_width,
                "height": source_height,
            },
            "render_config": {
                "depth_reuse_interval": 1,
            },
            "depth": {
                "dimensions": "per-frame-metadata",
                "dtype": "float32-le",
                "dxgi_format": "R32_FLOAT",
                "bytes_per_frame": None,
                "bytes_per_sample": 4,
            },
            "frame_metadata": {
                "schema": whole.SCENE_CACHE_METADATA_SCHEMA,
                "magic": whole.SCENE_CACHE_METADATA_MAGIC,
                "word_count": whole.SCENE_CACHE_METADATA_WORDS,
                "bytes_per_frame": whole.SCENE_CACHE_METADATA_BYTES,
                "roi_transform_word_offset":
                    whole.SCENE_CACHE_METADATA_ROI_OFFSET,
                "roi_transform_word_count": whole.SCENE_CACHE_ROI_WORDS,
                "roi_transform_contract_schema":
                    whole.FRAME_ROI_TRANSFORM_SCHEMA,
            },
            "state": {
                "schema": whole.SCENE_CACHE_STATE_SCHEMA,
                "subject_word_count": whole.SCENE_CACHE_SUBJECT_WORDS,
                "depth_frame_state_word_count":
                    whole.SCENE_CACHE_DEPTH_FRAME_STATE_WORDS,
                "word_count": whole.SCENE_CACHE_STATE_WORDS,
                "bytes_per_frame": whole.SCENE_CACHE_STATE_WORDS * 4,
            },
        }

    def test_ledger_enforces_pressure_force_block_and_exact_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = whole.SceneCacheLedger(root, 1200)
            contract = self._contract()
            depth = root / "frame_0000000001.depth.r32f"
            state = root / "frame_0000000001.state.u32"
            metadata = root / "frame_0000000001.meta.u32"
            self._write_triplet(root)
            self.assertEqual(ledger.acknowledge_pair(1, contract), 1040)
            self.assertEqual(ledger.pressure_state(), "pressure")
            self.assertEqual(ledger.pressure_state(100), "force-finalize")
            self.assertEqual(ledger.pressure_state(200), "blocked")
            ledger.record_forced_segment(1, 2)
            released = ledger.release_through(2)

        self.assertEqual(released, {"pairs": 1, "bytes": 1040})
        self.assertEqual(ledger.current_bytes, 0)
        self.assertEqual(ledger.high_water_bytes, 1040)
        self.assertFalse(depth.exists())
        self.assertFalse(state.exists())
        self.assertFalse(metadata.exists())
        self.assertFalse(
            ledger.snapshot()["forced_segments"][0]["semantic_boundary"])

    def test_preflight_reserves_largest_dynamic_triplet_before_publish(self):
        maximum = whole.scene_cache_max_triplet_bytes(3840, 2160)
        self.assertEqual(
            maximum,
            1036 * 1036 * 4 +
            whole.SCENE_CACHE_STATE_WORDS * 4 +
            whole.SCENE_CACHE_METADATA_BYTES,
        )
        self.assertEqual(
            whole.scene_cache_max_triplet_bytes(1920, 1080),
            maximum,
        )
        self.assertEqual(
            whole.preflight_scene_cache_hard_cap(
                3840, 2160, maximum * 10),
            maximum,
        )
        with self.assertRaisesRegex(
                whole.WholeClipError, "before native publication"):
            whole.preflight_scene_cache_hard_cap(
                3840, 2160, maximum * 10 - 1)
        with self.assertRaisesRegex(
                whole.WholeClipError, "valid scene-cache depth shape"):
            whole.scene_cache_max_triplet_bytes(13, 1080)

    def test_ledger_rejects_contract_size_mismatch_before_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_triplet(
                root, depth_bytes=b"d")
            ledger = whole.SceneCacheLedger(root, 1000)
            with self.assertRaisesRegex(whole.WholeClipError, "size mismatch"):
                ledger.acknowledge_pair(1, self._contract())
            self.assertEqual(ledger.current_bytes, 0)

    def test_ledger_accepts_variable_schema_2_triplet_sizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = whole.SceneCacheLedger(root, 100_000)
            self._write_triplet(
                root,
                sequence=1,
                source_width=56,
                source_height=28,
                width=28,
                height=14,
            )
            first_bytes = ledger.acknowledge_pair(
                1,
                self._contract(
                    processed_count=1,
                    source_width=56,
                    source_height=28,
                ),
            )
            self._write_triplet(
                root,
                sequence=2,
                source_width=56,
                source_height=28,
                width=56,
                height=28,
            )
            second_bytes = ledger.acknowledge_pair(
                2,
                self._contract(
                    processed_count=2,
                    source_width=56,
                    source_height=28,
                ),
            )

            self.assertEqual(first_bytes, 28 * 14 * 4 + 64 + 192)
            self.assertEqual(second_bytes, 56 * 28 * 4 + 64 + 192)
            self.assertNotEqual(first_bytes, second_bytes)
            self.assertEqual(
                ledger.release_through(3),
                {"pairs": 2, "bytes": first_bytes + second_bytes},
            )

    def test_ledger_rejects_corrupt_state_and_schema_2_frame_identity(self):
        cases = (
            (
                "state",
                {},
                lambda state: state.__setitem__(
                    whole.SCENE_CACHE_SUBJECT_WORDS + 3,
                    self._float_word(0.5),
                ),
                "invalid validity flags",
            ),
            (
                "source",
                {"metadata_mutator":
                    lambda words: words.__setitem__(12, words[12] + 1)},
                None,
                "source identity",
            ),
            (
                "retained",
                {"metadata_mutator":
                    lambda words: words.__setitem__(10, 1)},
                None,
                "retained source identity",
            ),
            (
                "transform",
                {"metadata_mutator":
                    lambda words: words.__setitem__(
                        whole.SCENE_CACHE_METADATA_ROI_OFFSET + 1,
                        1 << 0,
                    )},
                None,
                "ROI transform",
            ),
        )
        for name, triplet_options, state_mutator, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self._write_triplet(
                    root,
                    state_mutator=state_mutator,
                    **triplet_options,
                )
                ledger = whole.SceneCacheLedger(root, 10_000)
                with self.assertRaisesRegex(whole.WholeClipError, message):
                    ledger.acknowledge_pair(1, self._contract())
                self.assertEqual(ledger.current_bytes, 0)

    def test_ledger_rejects_consistently_stale_reused_depth_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_triplet(root, sequence=5)
            contract = self._contract(processed_count=5)
            contract["render_config"]["depth_reuse_interval"] = 4
            ledger = whole.SceneCacheLedger(root, 10_000)
            self.assertEqual(
                ledger.acknowledge_pair(5, contract),
                14 * 14 * 4 + 64 + 192,
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def make_consistently_stale(words):
                words[10:12] = [0, 0]
                offset = whole.SCENE_CACHE_METADATA_ROI_OFFSET
                words[offset + 2:offset + 4] = [0, 0]

            self._write_triplet(
                root,
                sequence=5,
                metadata_mutator=make_consistently_stale,
            )
            contract = self._contract(processed_count=5)
            contract["render_config"]["depth_reuse_interval"] = 4
            ledger = whole.SceneCacheLedger(root, 10_000)
            with self.assertRaisesRegex(
                    whole.WholeClipError, "retained source identity"):
                ledger.acknowledge_pair(5, contract)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def make_retained_previous(words):
                words[10:12] = [0, 0]
                offset = whole.SCENE_CACHE_METADATA_ROI_OFFSET
                words[offset + 2:offset + 4] = [0, 0]

            self._write_triplet(
                root,
                sequence=5,
                metadata_mutator=make_retained_previous,
                state_mutator=lambda state: state.__setitem__(
                    whole.SCENE_CACHE_SUBJECT_WORDS + 3,
                    self._float_word(0.0),
                ),
            )
            contract = self._contract(processed_count=5)
            contract["render_config"]["depth_reuse_interval"] = 4
            ledger = whole.SceneCacheLedger(root, 10_000)
            self.assertEqual(
                ledger.acknowledge_pair(5, contract),
                14 * 14 * 4 + 64 + 192,
            )
            self.assertTrue(
                ledger.requires_previous_packed_frame(5))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def make_non_cadence_previous(words):
                words[10:12] = [1, 0]
                offset = whole.SCENE_CACHE_METADATA_ROI_OFFSET
                words[offset + 2:offset + 4] = [1, 0]

            self._write_triplet(
                root,
                sequence=5,
                metadata_mutator=make_non_cadence_previous,
                state_mutator=lambda state: state.__setitem__(
                    whole.SCENE_CACHE_SUBJECT_WORDS + 3,
                    self._float_word(0.0),
                ),
            )
            contract = self._contract(processed_count=5)
            contract["render_config"]["depth_reuse_interval"] = 4
            ledger = whole.SceneCacheLedger(root, 10_000)
            with self.assertRaisesRegex(
                    whole.WholeClipError, "retained source identity"):
                ledger.acknowledge_pair(5, contract)

    def test_ledger_allows_only_explicit_terminal_forced_current_depth(self):
        def contract_for_sequence() -> dict:
            contract = self._contract(processed_count=6)
            contract["render_config"]["depth_reuse_interval"] = 4
            return contract

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_triplet(root, sequence=6)
            ledger = whole.SceneCacheLedger(root, 10_000)
            with self.assertRaisesRegex(
                    whole.WholeClipError, "retained source identity"):
                ledger.acknowledge_pair(6, contract_for_sequence())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_triplet(root, sequence=6)
            ledger = whole.SceneCacheLedger(root, 10_000)
            self.assertEqual(
                ledger.acknowledge_pair(
                    6,
                    contract_for_sequence(),
                    allow_forced_current=True,
                ),
                14 * 14 * 4 + 64 + 192,
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_triplet(
                root,
                sequence=6,
                state_mutator=lambda state: state.__setitem__(
                    whole.SCENE_CACHE_SUBJECT_WORDS + 3,
                    self._float_word(0.0),
                ),
            )
            ledger = whole.SceneCacheLedger(root, 10_000)
            with self.assertRaisesRegex(
                    whole.WholeClipError, "retained source identity"):
                ledger.acknowledge_pair(
                    6,
                    contract_for_sequence(),
                    allow_forced_current=True,
                )

    def test_ledger_accepts_first_valid_and_reports_preserve_previous_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_triplet(
                root,
                state_mutator=lambda state: state.__setitem__(
                    whole.SCENE_CACHE_SUBJECT_WORDS + 3,
                    self._float_word(2.0),
                ),
            )
            ledger = whole.SceneCacheLedger(root, 10_000)
            ledger.acknowledge_pair(1, self._contract())
            self.assertFalse(ledger.requires_previous_packed_frame(1))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def retain_first_frame(words):
                words[10:12] = [0, 0]
                offset = whole.SCENE_CACHE_METADATA_ROI_OFFSET
                words[offset + 2:offset + 4] = [0, 0]

            self._write_triplet(
                root,
                sequence=2,
                metadata_mutator=retain_first_frame,
                state_mutator=lambda state: state.__setitem__(
                    whole.SCENE_CACHE_SUBJECT_WORDS + 3,
                    self._float_word(0.0),
                ),
            )
            ledger = whole.SceneCacheLedger(root, 10_000)
            ledger.acknowledge_pair(
                2,
                self._contract(processed_count=2),
            )
            self.assertTrue(ledger.requires_previous_packed_frame(2))

    def test_ledger_rejects_each_invalid_cached_state_field(self):
        invalid_fields = (
            (
                "subject initialized",
                whole.ADAPTIVE_INITIALIZED_WORD,
                0.5,
            ),
            (
                "zero anchor valid",
                whole.ADAPTIVE_ZERO_ANCHOR_VALID_WORD,
                0.5,
            ),
            (
                "cut flags fractional",
                whole.ADAPTIVE_CUT_FLAGS_WORD,
                0.5,
            ),
            (
                "cut flags unknown",
                whole.ADAPTIVE_CUT_FLAGS_WORD,
                float(whole.ADAPTIVE_KNOWN_CUT_FLAG_MASK + 1),
            ),
            (
                "history state",
                whole.ADAPTIVE_MODEL_INPUT_HISTORY_STATE_WORD,
                5.0,
            ),
        )
        for name, index, value in invalid_fields:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self._write_triplet(
                    root,
                    state_mutator=lambda state, index=index, value=value:
                        state.__setitem__(index, self._float_word(value)),
                )
                ledger = whole.SceneCacheLedger(root, 10_000)
                with self.assertRaisesRegex(
                        whole.WholeClipError, "invalid validity flags"):
                    ledger.acknowledge_pair(1, self._contract())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def invalid_first_valid(state):
                state[whole.SCENE_CACHE_SUBJECT_WORDS + 2] = (
                    self._float_word(0.0))
                state[whole.SCENE_CACHE_SUBJECT_WORDS + 3] = (
                    self._float_word(2.0))

            self._write_triplet(root, state_mutator=invalid_first_valid)
            ledger = whole.SceneCacheLedger(root, 10_000)
            with self.assertRaisesRegex(
                    whole.WholeClipError, "invalid validity flags"):
                ledger.acknowledge_pair(1, self._contract())

    def test_ledger_rejects_unsafe_depth_dimensions_before_file_accounting(self):
        cases = (
            ("engine maximum", 1050, 1050, 1050, 1050),
            ("source bounds", 28, 28, 42, 28),
            ("patch alignment", 28, 28, 15, 14),
        )
        for name, source_width, source_height, width, height in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self._write_triplet(
                    root,
                    source_width=source_width,
                    source_height=source_height,
                    width=width,
                    height=height,
                    depth_bytes=b"",
                )
                ledger = whole.SceneCacheLedger(root, 10_000_000)
                with self.assertRaisesRegex(
                        whole.WholeClipError, "depth identity"):
                    ledger.acknowledge_pair(
                        1,
                        self._contract(
                            source_width=source_width,
                            source_height=source_height,
                        ),
                    )
                self.assertEqual(ledger.current_bytes, 0)

    def test_ledger_requires_exact_running_ack_contract(self):
        mutations = (
            ("stale count", lambda value: value.__setitem__("processed_count", 0)),
            ("ahead count", lambda value: value.__setitem__("processed_count", 2)),
            ("terminal", lambda value: value.__setitem__("status", "complete")),
            ("wrong first", lambda value: value.__setitem__("first_sequence", 2)),
            (
                "non-atomic",
                lambda value: value.__setitem__("atomic_frame_publication", False),
            ),
            (
                "depth dtype",
                lambda value: value["depth"].__setitem__("dtype", "float16-le"),
            ),
            (
                "depth format",
                lambda value: value["depth"].__setitem__(
                    "dxgi_format", "R16_FLOAT"),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self._write_triplet(root)
                contract = copy.deepcopy(self._contract())
                mutate(contract)
                ledger = whole.SceneCacheLedger(root, 10_000)
                with self.assertRaises(whole.WholeClipError):
                    ledger.acknowledge_pair(1, contract)
                self.assertEqual(ledger.current_bytes, 0)

    def test_trace_tail_uses_the_same_incremental_contract_and_ack_identity(self):
        import test_adaptive_clip_report as fixtures

        class LiveChild:
            log_path = Path("native.log")

            @staticmethod
            def poll():
                return None

            @staticmethod
            def close_log():
                return None

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / whole.TRACE_NAME
            fixtures.write_trace(
                path,
                [fixtures.frame_record(0), fixtures.frame_record(1)],
            )
            tail = whole.AdaptiveTraceTail(path)
            try:
                first = tail.read_frame(1, LiveChild(), timeout_seconds=1)
                second = tail.read_frame(2, LiveChild(), timeout_seconds=1)
                header = tail.finish(2)
            finally:
                tail.close()

        self.assertEqual(first["frame_id"], "0000000001")
        self.assertEqual(second["source_index"], 1)
        self.assertEqual(header["schema"], whole.ADAPTIVE_TRACE_SCHEMA)


class WholeClipOrchestrationTests(unittest.TestCase):
    @staticmethod
    def _write_frame(path, color=(10, 20, 30)):
        Image.new("RGB", (16, 8), color).save(path)

    @staticmethod
    def _write_runtime_assets(build_dir, model="test_model"):
        assets = build_dir / "assets"
        shaders = assets / "shaders" / "directx"
        shaders.mkdir(parents=True)
        (shaders / "test.hlsl").write_text("float4 main() : SV_Target { return 0; }\n")
        onnx = assets / f"{model}.onnx"
        engine = assets / f"{model}.engine"
        onnx.write_bytes(b"onnx")
        engine.write_bytes(b"engine")
        (assets / f"{model}.active-engine.json").write_text(json.dumps({
            "schema": 1,
            "model": model,
            "engine": engine.name,
            "onnx_sha256": whole.sha256_file(onnx),
        }), encoding="utf-8")

    def _args(self, source, output, sunshine, conf, *extra):
        return argparse.Namespace(
            input=os.fspath(source),
            out=os.fspath(output),
            build_dir=os.fspath(sunshine.parent),
            sunshine=os.fspath(sunshine),
            conf=os.fspath(conf),
            fps=12.0,
            sbs_video=None,
            codec="hevc_nvenc",
            keep_work=False,
            keep_sbs_frames=False,
            extra=list(extra),
        )

    def test_parser_keeps_normal_options_after_input_and_bounds_native_extra(self):
        args = whole.parse_args([
            "movie.mkv",
            "--out", "result",
            "--fps", "24",
            "--extra", "--", "--zero-plane", "subject",
        ])
        self.assertEqual(args.out, "result")
        self.assertEqual(args.fps, 24)
        self.assertEqual(args.extra, ["--zero-plane", "subject"])
        with self.assertRaisesRegex(whole.WholeClipError, "owned"):
            whole.validate_native_extra(["--limit=5"])
        with self.assertRaisesRegex(whole.WholeClipError, "owned"):
            whole.validate_native_extra(["--scene-controller", "off"])

    def test_run_uses_bounded_scene_pipeline_and_cleans_success_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "frames"
            output = root / "out"
            source.mkdir()
            self._write_frame(source / "frame_00001.png")
            self._write_frame(source / "frame_00002.png")
            sunshine = root / "sunshine.exe"
            conf = root / "bench.conf"
            sunshine.write_bytes(b"exe")
            conf.write_text("config", encoding="utf-8")
            self._write_runtime_assets(root)
            calls = []

            def fake_pipeline(**kwargs):
                calls.append(kwargs)
                source_time = kwargs["scene_controller_source_time"]
                artifacts = output / "artifacts"
                (artifacts / whole.TRACE_NAME).write_text(
                    '{"schema":3}\n', encoding="utf-8")
                (artifacts / whole.SCENE_CONTROLLER_TRACE_NAME).write_text(
                    '{"record":"header"}\n', encoding="utf-8")
                native_contract = {
                    "schema": 1,
                    "artifact_mode": "adaptive",
                    "model": "test_model",
                    "resolved_runtime": {"model": "test_model"},
                    "source_frame_count": 2,
                    "adaptive_state": {
                        "file": whole.TRACE_NAME,
                        "schema": whole.ADAPTIVE_TRACE_SCHEMA,
                        "frame_count": 2,
                    },
                    "sbs": {
                        "enabled": False,
                        "file_pattern": None,
                        "frame_count": 0,
                    },
                    "scene_controller": {
                        "enabled": True,
                        "backend": "shadow_rules",
                        "active_roi_authority": False,
                        "transport":
                            whole.scene_controller_trace.TRACE_TRANSPORT,
                        "file": whole.SCENE_CONTROLLER_TRACE_NAME,
                        "header_file": None,
                        "frame_file": None,
                        "retained_history": True,
                        "trace_schema":
                            whole.scene_controller_trace.TRACE_SCHEMA,
                        "controller_schema": (
                            whole.scene_controller_trace.controller_contract.SCHEMA_VERSION
                        ),
                        "rule_revision": (
                            whole.scene_controller_trace.controller_contract.RULE_REVISION
                        ),
                        "ordered_abi_hash": (
                            whole.scene_controller_trace.controller_contract.ORDERED_ABI_HASH
                        ),
                        "frame_count": 2,
                    },
                    "scene_controller_source_time": {
                        "enabled": True,
                        "schema":
                            whole.SCENE_CONTROLLER_SOURCE_TIME_SCHEMA,
                        "clock":
                            whole.SCENE_CONTROLLER_SOURCE_TIME_CLOCK,
                        "file_sha256": source_time["sha256"],
                        "frame_count": source_time["frame_count"],
                        "total_elapsed_seconds":
                            source_time["total_elapsed_seconds"],
                    },
                }
                (artifacts / whole.NATIVE_CONTRACT_NAME).write_text(json.dumps({
                    **native_contract,
                }), encoding="utf-8")
                (output / "scene_audit.json").write_text(
                    '{"schema":1}', encoding="utf-8")
                return {
                    "analysis": {
                        "command": ["sunshine", "--sbs-bench", *kwargs["native_extra"]],
                        "native_contract": native_contract,
                    },
                    "scene_audit": {
                        "file": os.fspath(output / "scene_audit.json"),
                        "sha256": whole.sha256_file(output / "scene_audit.json"),
                        "scene_count": 1,
                        "boundary_revision_count": 0,
                    },
                    "cache": {"enabled": False},
                    "render_scenes": [],
                    "scene_controller_qualification": {
                        "schema": 1,
                        "pass": True,
                    },
                    "encoder": None,
                }

            capabilities = {
                "value": {"schema": 1},
                "file": "capabilities.json",
                "sha256": "00",
                "command": ["sunshine", "--sbs-bench", "--capabilities"],
            }
            with mock.patch.object(
                        whole, "query_native_capabilities",
                        return_value=capabilities), \
                    mock.patch.object(
                        whole, "run_streaming_scene_pipeline",
                        side_effect=fake_pipeline), \
                    mock.patch.object(
                        whole, "runtime_provenance",
                        return_value={"validated": True}), \
                    mock.patch.object(
                        whole, "generate_report_outputs",
                        return_value={"available": True}):
                result = whole.run(self._args(
                    source, output, sunshine, conf, "--zero-plane", "subject"))

            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(calls), 1)
            self.assertIsInstance(
                calls[0]["analysis_producer"],
                whole.FrameDirectoryFollowProducer,
            )
            self.assertEqual(
                calls[0]["native_extra"], ["--zero-plane", "subject"])
            source_time_path = calls[0][
                "scene_controller_source_time_path"]
            self.assertEqual(
                source_time_path,
                output / whole.SCENE_CONTROLLER_SOURCE_TIME_NAME,
            )
            self.assertTrue(source_time_path.is_file())
            self.assertEqual(
                calls[0]["scene_controller_source_time"]["sha256"],
                whole.sha256_file(source_time_path),
            )
            self.assertEqual(
                calls[0]["qualification_frames_dir"],
                source,
            )
            self.assertFalse((output / "work").exists())
            manifest = json.loads(
                (output / whole.MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["timeline"]["frame_count"], 2)
            self.assertEqual(
                manifest["scene_controller_source_time"]["sha256"],
                whole.sha256_file(source_time_path),
            )
            self.assertTrue(
                manifest["scene_controller_qualification"]["pass"])

    def test_streaming_pipeline_authenticates_fixture_before_native_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "generated"
            source.mkdir()
            producer = mock.Mock()
            producer.extension = "png"
            with mock.patch.object(
                    whole.scene_controller_eval,
                    "load_generated_clip_contract",
                    side_effect=whole.scene_controller_eval.SceneControllerEvalError(
                        "forged generated fixture"
                    )), mock.patch.object(
                        whole, "LoggedSubprocess"
                    ) as launch:
                with self.assertRaisesRegex(
                        whole.WholeClipError,
                        "source authentication failed"):
                    whole.run_streaming_scene_pipeline(
                        sunshine=root / "sunshine.exe",
                        conf=root / "bench.conf",
                        build_dir=root,
                        timeline={
                            "frame_count": 1,
                            "frames": [{
                                "frame_id": "0000000001",
                                "pts_time": 0.0,
                                "duration_time": 1.0 / 30.0,
                            }],
                        },
                        source_width=320,
                        source_height=180,
                        analysis_producer=producer,
                        render_producer=None,
                        output_dir=root / "out",
                        work_dir=root / "work",
                        artifacts_dir=root / "artifacts",
                        scene_controller_source_time_path=(
                            root / whole.SCENE_CONTROLLER_SOURCE_TIME_NAME
                        ),
                        scene_controller_source_time={},
                        native_extra=[],
                        cache_max_bytes=whole.DEFAULT_SCENE_CACHE_MAX_BYTES,
                        cache_budget_policy="segment",
                        ffmpeg=None,
                        ffprobe=None,
                        source_video=None,
                        output_video=None,
                        codec="hevc_nvenc",
                        color={"mode": "sdr"},
                        qualification_frames_dir=source,
                    )
            launch.assert_not_called()

    def test_failed_native_process_retains_work_and_failure_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "frames"
            output = root / "out"
            source.mkdir()
            self._write_frame(source / "frame_00001.png")
            sunshine = root / "sunshine.exe"
            conf = root / "bench.conf"
            sunshine.write_bytes(b"exe")
            conf.write_text("config", encoding="utf-8")

            with mock.patch.object(
                    whole, "run_logged_command",
                    side_effect=whole.WholeClipError("native failed")):
                with self.assertRaisesRegex(whole.WholeClipError, "native failed"):
                    whole.run(self._args(source, output, sunshine, conf))

            self.assertTrue((output / "work").is_dir())
            manifest = json.loads(
                (output / whole.MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error"], "native failed")

    def test_audio_remux_maps_every_audio_stream_without_transcoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            work = root / "work"
            artifacts.mkdir()
            work.mkdir()
            source = root / "source.mkv"
            source.write_bytes(b"source")
            output = root / "converted.mkv"
            timeline = {
                "time_base": {"num": 1, "den": 1000},
                "frame_count": 2,
                "first_pts": 1250,
                "first_pts_time": 1.25,
                "frames": [
                    {
                        "frame_id": "00001", "pts": 1250,
                        "duration": 40, "duration_time": 0.04,
                    },
                    {
                        "frame_id": "00002", "pts": 1290,
                        "duration": 60, "duration_time": 0.06,
                    },
                ],
            }
            for frame_id in ("00001", "00002"):
                (artifacts / f"sbs_{frame_id}.png").write_bytes(b"x")
            calls = []

            def fake_command(command, cwd, log_path):
                calls.append(command)
                if log_path.name == "encode.log":
                    output.write_bytes(b"muxed")

            with mock.patch.object(whole, "run_logged_command", fake_command):
                commands = whole.encode_sbs_video(
                    "ffmpeg", artifacts, timeline, source, output,
                    "hevc_nvenc", work)

        self.assertEqual(commands, calls)
        encode = calls[0]
        self.assertIn("-copyts", encode)
        self.assertIn("1:a?", encode)
        self.assertIn("-map_metadata", encode)
        self.assertIn("-map_chapters", encode)
        self.assertEqual(encode[encode.index("-c:a") + 1], "copy")
        self.assertEqual(
            encode[encode.index("-itsoffset") + 1], "1.250000000000")
        self.assertIn("-enc_time_base", encode)
        self.assertNotIn("-frames:v", encode)

    def test_success_cleanup_removes_only_timeline_bound_sbs_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for frame_id in ("00001", "00002"):
                (root / f"sbs_{frame_id}.png").write_bytes(b"x")
            unrelated = root / "sbs_unrelated.png"
            unrelated.write_bytes(b"keep")
            removed = whole.remove_expected_sbs_frames(root, {
                "frames": [{"frame_id": "00001"}, {"frame_id": "00002"}],
            })
            self.assertEqual(removed, 2)
            self.assertTrue(unrelated.is_file())

    def test_disk_preflight_fails_closed_and_reports_required_space(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(whole.DiskSpaceError) as caught:
                whole.disk_spool_preflight(
                    Path(temporary),
                    frame_count=100,
                    source_width=3840,
                    source_height=2160,
                    source_spooled=True,
                    conversion=True,
                    native_extra=[],
                    free_bytes=1,
                )
        result = caught.exception.preflight
        self.assertFalse(result["passed"])
        self.assertGreater(result["required_free_bytes"], result["free_bytes"])
        self.assertEqual(result["sbs"]["width"], 7680)

    def test_output_preflight_rejects_cleanup_scope_and_non_delivery_codec(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with self.assertRaisesRegex(whole.WholeClipError, "inside <out>/work"):
                whole.validate_output_target(
                    root / "work" / "clip.mkv", "hevc_nvenc", root / "work")
            with self.assertRaisesRegex(whole.WholeClipError, "unsupported codec"):
                whole.validate_output_target(
                    root / "clip.mkv", "ffv1", root / "work")

    def test_runtime_provenance_fails_on_onnx_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_runtime_assets(root)
            contract = {
                "model": "test_model",
                "resolved_runtime": {"model": "test_model"},
            }
            provenance = whole.runtime_provenance(root, contract)
            self.assertEqual(provenance["engine"]["sha256"], whole.sha256_file(
                root / "assets" / "test_model.engine"))
            manifest = root / "assets" / "test_model.active-engine.json"
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["onnx_sha256"] = "0" * 64
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(whole.WholeClipError, "ONNX SHA-256"):
                whole.runtime_provenance(root, contract)

    def test_mp4_frame_input_is_encoded_directly_into_mp4_container(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            work = root / "work"
            artifacts.mkdir()
            work.mkdir()
            output = root / "converted.mp4"
            timeline = {
                "time_base": {"num": 1, "den": 30},
                "first_pts": 0,
                "first_pts_time": 0.0,
                "frame_count": 1,
                "frames": [{
                    "frame_id": "00001",
                    "pts": 0,
                    "duration": 1,
                    "duration_time": 1 / 30,
                }],
            }
            self._write_frame(artifacts / "sbs_00001.png")
            calls = []

            def fake_command(command, cwd, log_path):
                calls.append(command)
                output.write_bytes(b"mp4")

            with mock.patch.object(whole, "run_logged_command", fake_command):
                whole.encode_sbs_video(
                    "ffmpeg", artifacts, timeline, None, output,
                    "libx265", work)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][calls[0].index("-f", 8) + 1], "concat")
        self.assertIn("mp4", calls[0])
        self.assertEqual(calls[0][-1], os.fspath(output))


class WholeClipFfmpegTimingTests(unittest.TestCase):
    @staticmethod
    def _ffmpeg_or_skip(test_case):
        try:
            return whole.resolve_ffmpeg()
        except RuntimeError as exc:
            test_case.skipTest(str(exc))

    def test_vfr_h265_round_trip_preserves_pts_count_and_final_duration(self):
        ffmpeg = self._ffmpeg_or_skip(self)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            work = root / "work"
            artifacts.mkdir()
            work.mkdir()
            for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255)), 1):
                Image.new("RGB", (256, 144), color).save(
                    artifacts / f"sbs_{index:05d}.png")
            timeline = {
                "time_base": {"num": 1, "den": 1000},
                "first_pts": 100,
                "first_pts_time": 0.1,
                "frame_count": 3,
                "frames": [
                    {
                        "frame_id": "00001", "pts": 100,
                        "duration": 40, "duration_time": 0.04,
                    },
                    {
                        "frame_id": "00002", "pts": 140,
                        "duration": 60, "duration_time": 0.06,
                    },
                    {
                        "frame_id": "00003", "pts": 200,
                        "duration": 40, "duration_time": 0.04,
                    },
                ],
            }
            output = root / "converted.mkv"
            whole.encode_sbs_video(
                ffmpeg, artifacts, timeline, None, output, "libx265", work)
            validation = whole.validate_encoded_timeline(
                ffmpeg, output, timeline, work / "validate.log")

        self.assertEqual(validation["frame_count"], 3)
        self.assertEqual(validation["duration_error_seconds"], 0.0)
        self.assertEqual(validation["max_pts_error_seconds"], 0.0)

    def test_cfr_12_does_not_regress_to_image_demuxer_25_fps(self):
        ffmpeg = self._ffmpeg_or_skip(self)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            work = root / "work"
            artifacts.mkdir()
            work.mkdir()
            frames = []
            for index in range(36):
                frame_id = f"{index + 1:05d}"
                Image.new("RGB", (256, 144), (index, 0, 0)).save(
                    artifacts / f"sbs_{frame_id}.png")
                frames.append({
                    "frame_id": frame_id,
                    "pts": index,
                    "duration": 1,
                    "duration_time": 1 / 12,
                })
            timeline = {
                "time_base": {"num": 1, "den": 12},
                "first_pts": 0,
                "first_pts_time": 0.0,
                "frame_count": 36,
                "frames": frames,
            }
            output = root / "converted.mkv"
            whole.encode_sbs_video(
                ffmpeg, artifacts, timeline, None, output, "libx265", work)
            validation = whole.validate_encoded_timeline(
                ffmpeg, output, timeline, work / "validate.log")

        self.assertEqual(validation["frame_count"], 36)
        self.assertAlmostEqual(validation["actual_duration_seconds"], 3.0)
        self.assertLessEqual(
            validation["duration_error_seconds"],
            validation["tolerance_seconds"],
        )

    def test_streaming_decoder_publishes_atomic_bounded_follow_frames(self):
        ffmpeg = self._ffmpeg_or_skip(self)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mkv"
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner", "-loglevel", "error", "-nostdin",
                    "-f", "lavfi",
                    "-i", "testsrc2=size=64x48:rate=3:duration=1",
                    "-c:v", "ffv1",
                    os.fspath(source),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )
            decoder = whole.VideoFollowFrameDecoder(
                ffmpeg,
                source,
                64,
                48,
                {"mode": "sdr"},
                root / "decode.log",
            )
            try:
                paths = [
                    decoder.publish_next(root / "follow", sequence)
                    for sequence in range(1, 4)
                ]
                decoder.finish(3)
            finally:
                decoder.abort()

            self.assertEqual(
                [path.name for path in paths],
                [f"frame_{sequence:010d}.png" for sequence in range(1, 4)],
            )
            self.assertFalse(any((root / "follow").glob("*.part")))
            for path in paths:
                with Image.open(path) as image:
                    self.assertEqual(image.size, (64, 48))

    def test_frame_directory_producer_normalizes_global_png_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            png = root / "arbitrary.png"
            jpeg = root / "frame_17.jpg"
            Image.new("RGB", (16, 8), (1, 2, 3)).save(png)
            Image.new("RGB", (16, 8), (4, 5, 6)).save(jpeg)
            producer = whole.FrameDirectoryFollowProducer([png, jpeg])
            first = producer.publish_next(root / "follow", 1)
            second = producer.publish_next(root / "follow", 2)
            producer.finish(2)

            self.assertEqual(first.name, "frame_0000000001.png")
            self.assertEqual(second.name, "frame_0000000002.png")
            with Image.open(second) as image:
                self.assertEqual(image.size, (16, 8))


class WholeClipHttpBridgeTests(unittest.TestCase):
    @staticmethod
    def _wait_for(predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError("timed out waiting for bridge state")

    def test_bridge_is_loopback_tokenized_and_sends_exact_content_length(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = Path(temporary) / "sbs_0000000001.png"
            payload = b"exact-frame-payload"
            frame.write_bytes(payload)
            bridge = whole.SbsFrameHttpBridge(
                1, "png", wait_seconds=2,
                token="A_secure_test_token_0123456789")
            try:
                with self.assertRaises(urllib.error.HTTPError) as wrong:
                    urllib.request.urlopen(
                        bridge.base_url.replace(bridge.token, "x" * 32) +
                        "/frame_0000000001.png",
                        timeout=2,
                    )
                self.assertEqual(wrong.exception.code, 404)
                wrong.exception.close()
                self.assertIsNone(bridge.error)
                bridge.publish(1, frame)
                with urllib.request.urlopen(bridge.frame_url(1), timeout=2) as response:
                    self.assertEqual(
                        int(response.headers["Content-Length"]), len(payload))
                    self.assertEqual(response.read(), payload)
                self._wait_for(lambda: bridge.served_count == 1)
                bridge.close(encoder_succeeded=True)
                self.assertFalse(frame.exists())
            finally:
                if bridge._server_thread.is_alive():
                    bridge.close(encoder_succeeded=False)

    def test_bridge_rejects_non_monotonic_request_and_times_out_missing_frame(self):
        bridge = whole.SbsFrameHttpBridge(
            2, "png", wait_seconds=0.1,
            token="B_secure_test_token_0123456789")
        try:
            with self.assertRaises(urllib.error.HTTPError) as out_of_order:
                urllib.request.urlopen(bridge.frame_url(2), timeout=2)
            self.assertEqual(out_of_order.exception.code, 409)
            out_of_order.exception.close()
            self.assertIn("non-monotonic", bridge.error)
        finally:
            bridge.close(encoder_succeeded=False)

        timeout_bridge = whole.SbsFrameHttpBridge(
            1, "png", wait_seconds=0.05,
            token="C_secure_test_token_0123456789")
        try:
            with self.assertRaises(urllib.error.HTTPError) as timed_out:
                urllib.request.urlopen(timeout_bridge.frame_url(1), timeout=2)
            self.assertEqual(timed_out.exception.code, 503)
            timed_out.exception.close()
            self.assertIn("timed out", timeout_bridge.error)
        finally:
            timeout_bridge.close(encoder_succeeded=False)

    def test_http_backpressure_keeps_one_encoder_and_exact_vfr_timeline(self):
        try:
            ffmpeg = whole.resolve_ffmpeg()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            work.mkdir()
            paths = []
            for index, color in enumerate(
                    ((255, 0, 0), (0, 255, 0), (0, 0, 255)), 1):
                path = root / f"sbs_{index:010d}.png"
                Image.new("RGB", (256, 144), color).save(path)
                paths.append(path)
            timeline = {
                "time_base": {"num": 1, "den": 1000},
                "first_pts": 0,
                "first_pts_time": 0.0,
                "frame_count": 3,
                "frames": [
                    {
                        "frame_id": "0000000001", "pts": 0,
                        "duration": 40, "duration_time": 0.04,
                    },
                    {
                        "frame_id": "0000000002", "pts": 40,
                        "duration": 60, "duration_time": 0.06,
                    },
                    {
                        "frame_id": "0000000003", "pts": 100,
                        "duration": 40, "duration_time": 0.04,
                    },
                ],
            }
            bridge = whole.SbsFrameHttpBridge(
                3, "png", wait_seconds=5,
                token="D_secure_test_token_0123456789")
            concat = work / "http.ffconcat"
            whole.write_http_concat_file(concat, bridge, timeline)
            output = root / "output.mkv"
            command = [
                ffmpeg,
                "-hide_banner", "-loglevel", "warning", "-xerror", "-nostdin",
                "-copyts",
                "-protocol_whitelist", "file,http,tcp",
                "-f", "concat", "-safe", "0", "-i", os.fspath(concat),
                "-map", "0:v:0",
                "-fps_mode", "passthrough",
                "-c:v", "libx265", "-preset", "ultrafast",
                "-x265-params", "log-level=error:bframes=0",
                "-enc_time_base", "demux",
                "-bsf:v", whole._last_duration_bsf(timeline),
                "-f", "matroska", os.fspath(output),
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            encoder_succeeded = False
            try:
                for sequence, path in enumerate(paths, 1):
                    self._wait_for(
                        lambda sequence=sequence:
                        bridge.requested_count == sequence)
                    bridge.publish(sequence, path)
                    if sequence > 1:
                        self._wait_for(
                            lambda sequence=sequence:
                            bridge.released_count == sequence - 1)
                        self.assertFalse(paths[sequence - 2].exists())
                stdout, stderr = process.communicate(timeout=20)
                self.assertEqual(
                    process.returncode, 0,
                    (stdout + stderr).decode("utf-8", errors="replace"))
                encoder_succeeded = True
                bridge.close(encoder_succeeded=True)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                if bridge._server_thread.is_alive():
                    bridge.close(encoder_succeeded=encoder_succeeded)
            validation = whole.validate_encoded_timeline(
                ffmpeg, output, timeline, work / "validate.log")

        self.assertEqual(validation["frame_count"], 3)
        self.assertEqual(validation["max_pts_error_seconds"], 0.0)
        self.assertEqual(validation["duration_error_seconds"], 0.0)

    def test_xerror_propagates_invalid_served_frame_as_encoder_failure(self):
        try:
            ffmpeg = whole.resolve_ffmpeg()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = root / "sbs_0000000001.png"
            frame.write_bytes(b"not a png")
            timeline = {
                "time_base": {"num": 1, "den": 30},
                "first_pts": 0,
                "first_pts_time": 0.0,
                "frame_count": 1,
                "frames": [{
                    "frame_id": "0000000001", "pts": 0,
                    "duration": 1, "duration_time": 1 / 30,
                }],
            }
            bridge = whole.SbsFrameHttpBridge(
                1, "png", wait_seconds=5,
                token="E_secure_test_token_0123456789")
            concat = root / "http.ffconcat"
            whole.write_http_concat_file(concat, bridge, timeline)
            output = root / "bad.mkv"
            command = [
                ffmpeg,
                "-hide_banner", "-loglevel", "warning", "-xerror", "-nostdin",
                "-protocol_whitelist", "file,http,tcp",
                "-f", "concat", "-safe", "0", "-i", os.fspath(concat),
                "-map", "0:v:0",
                "-c:v", "libx265",
                "-f", "matroska", os.fspath(output),
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                self._wait_for(lambda: bridge.requested_count == 1)
                bridge.publish(1, frame)
                process.communicate(timeout=10)
                self.assertNotEqual(process.returncode, 0)
                bridge.close(encoder_succeeded=False)
                self.assertIn("encoder did not complete", bridge.error)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                if bridge._server_thread.is_alive():
                    bridge.close(encoder_succeeded=False)


if __name__ == "__main__":
    unittest.main()
