#!/usr/bin/env python3
"""Focused unit tests for the continuous whole-clip orchestrator."""

import argparse
import json
import os
import subprocess
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
                side_effect=[PermissionError("sharing violation"), progress],
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

    def test_ffprobe_prefers_the_resolved_ffmpeg_sibling_over_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ffmpeg = root / "ffmpeg.exe"
            ffprobe = root / "ffprobe.exe"
            ffmpeg.write_bytes(b"test")
            ffprobe.write_bytes(b"test")
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                    whole.shutil,
                    "which",
                    return_value=os.fspath(root / "system" / "ffprobe.exe"),
            ):
                resolved = whole.resolve_ffprobe(ffmpeg=os.fspath(ffmpeg))

        self.assertEqual(resolved, os.fspath(ffprobe.resolve()))


class WholeClipCapabilityTests(unittest.TestCase):
    @staticmethod
    def _capabilities():
        return {
            "schema": 1,
            "native_whole_clip": {
                "follow_protocol_schema": 1,
                "follow_global_first_sequence": True,
                "adaptive_state_schema": whole.ADAPTIVE_TRACE_SCHEMA,
                "scene_cache_contract_schema": 1,
                "scene_cache_state": {
                    "schema": 1,
                    "word_count": 12,
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
                "atomic_sbs_publication": True,
                "artifact_modes": ["adaptive", "conversion"],
                "source_formats": ["png", "pfm"],
            },
        }

    def test_capability_gate_rejects_json_scalar_type_coercions(self):
        mutations = (
            (("schema",), True),
            (("native_whole_clip", "follow_protocol_schema"), True),
            (("native_whole_clip", "follow_global_first_sequence"), 1),
            (("native_whole_clip", "scene_plan", "schema"), 1.0),
        )
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
                        command, _cwd, _log_path, document=value
                    ):
                        Path(command[-1]).write_text(
                            json.dumps(document), encoding="utf-8")

                    with mock.patch.object(
                        whole,
                        "run_logged_command",
                        side_effect=publish_capabilities,
                    ):
                        with self.assertRaisesRegex(
                            whole.WholeClipError,
                            "required bounded scene replay",
                        ):
                            whole.query_native_capabilities(
                                root / "sunshine.exe",
                                root,
                                output,
                                root / "capabilities.log",
                            )


class WholeClipNativeContractTests(unittest.TestCase):
    def _write_contract(self, root, mode="conversion", count=2):
        (root / whole.TRACE_NAME).write_text(
            '{"schema":2,"frame_id":"00001"}\n', encoding="utf-8")
        (root / whole.NATIVE_CONTRACT_NAME).write_text(json.dumps({
            "schema": 1,
            "artifact_mode": mode,
            "source_frame_count": count,
            "adaptive_state": {
                "file": whole.TRACE_NAME,
                "schema": whole.ADAPTIVE_TRACE_SCHEMA,
                "frame_count": count,
            },
            "sbs": {
                "enabled": mode == "conversion",
                "file_pattern": (
                    "sbs_<frame-id>.png" if mode == "conversion" else None),
                "frame_count": count if mode == "conversion" else 0,
            },
        }), encoding="utf-8")

    def test_conversion_requires_exact_sbs_identity_set(self):
        timeline = {
            "frame_count": 2,
            "frames": [{"frame_id": "00001"}, {"frame_id": "00002"}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_contract(root)
            (root / "sbs_00001.png").write_bytes(b"x")
            (root / "sbs_00002.png").write_bytes(b"x")
            contract = whole.validate_native_outputs(root, timeline, "conversion")
            self.assertEqual(contract["source_frame_count"], 2)
            (root / "sbs_00002.png").unlink()
            with self.assertRaisesRegex(whole.WholeClipError, "identities"):
                whole.validate_native_outputs(root, timeline, "conversion")

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
            self._write_contract(root, count=3)
            for frame_id in ("00010", "00020", "00002"):
                (root / f"sbs_{frame_id}.png").write_bytes(b"x")
            whole.validate_native_outputs(root, timeline, "conversion")


class WholeClipSceneCacheTests(unittest.TestCase):
    def test_ledger_enforces_pressure_force_block_and_exact_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = whole.SceneCacheLedger(root, 1000)
            contract = {
                "schema": 1,
                "processed_count": 1,
                "depth": {"bytes_per_frame": 752},
                "state": {"bytes_per_frame": 48},
            }
            depth = root / "frame_0000000001.depth.r32f"
            state = root / "frame_0000000001.state.u32"
            depth.write_bytes(b"d" * 752)
            state.write_bytes(b"s" * 48)
            self.assertEqual(ledger.acknowledge_pair(1, contract), 800)
            self.assertEqual(ledger.pressure_state(), "pressure")
            self.assertEqual(ledger.pressure_state(100), "force-finalize")
            self.assertEqual(ledger.pressure_state(200), "blocked")
            ledger.record_forced_segment(1, 2)
            released = ledger.release_through(2)

        self.assertEqual(released, {"pairs": 1, "bytes": 800})
        self.assertEqual(ledger.current_bytes, 0)
        self.assertEqual(ledger.high_water_bytes, 800)
        self.assertFalse(depth.exists())
        self.assertFalse(state.exists())
        self.assertFalse(
            ledger.snapshot()["forced_segments"][0]["semantic_boundary"])

    def test_ledger_rejects_contract_size_mismatch_before_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "frame_0000000001.depth.r32f").write_bytes(b"d")
            (root / "frame_0000000001.state.u32").write_bytes(b"s" * 48)
            ledger = whole.SceneCacheLedger(root, 1000)
            with self.assertRaisesRegex(whole.WholeClipError, "size mismatch"):
                ledger.acknowledge_pair(1, {
                    "schema": 1,
                    "processed_count": 1,
                    "depth": {"bytes_per_frame": 2},
                    "state": {"bytes_per_frame": 48},
                })
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
                artifacts = output / "artifacts"
                (artifacts / whole.TRACE_NAME).write_text(
                    '{"schema":3}\n', encoding="utf-8")
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
            self.assertFalse((output / "work").exists())
            manifest = json.loads(
                (output / whole.MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["timeline"]["frame_count"], 2)

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
