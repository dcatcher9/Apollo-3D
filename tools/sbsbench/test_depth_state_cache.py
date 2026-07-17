#!/usr/bin/env python3

from pathlib import Path
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))

import depth_state_cache  # noqa: E402


class DepthStateCacheIdentityTests(unittest.TestCase):
    @staticmethod
    def _runtime(tag="a"):
        return {
            "schema": depth_state_cache.RUNTIME_SCHEMA,
            "contract": depth_state_cache.RUNTIME_CONTRACT,
            "runtime_namespace": "windows-nvidia-d3d11-tensorrt-inference-v1",
            "platform": {"system": "Windows", "tag": tag},
            "native_files": {
                "build/nvinfer_11.dll": {
                    "name": "nvinfer_11.dll", "bytes": 1,
                    "sha256": tag * 64,
                },
            },
            "gpus": [{"uuid": f"GPU-{tag}", "driver_version": tag}],
        }

    def _fixture(self, root):
        root = Path(root)
        repo = Path(__file__).resolve().parents[2]
        build = root / "build"
        shader_root = build / "assets" / "shaders" / "directx"
        for relative in depth_state_cache.DEPTH_SHADER_FILES:
            path = shader_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("shader:" + relative).encode())
        header = (repo / "src" / "model_manager.h").read_text(encoding="utf-8")
        recipe = re.search(
            r'depth_engine_recipe\[\]\s*=\s*"([^"]+)"', header
        ).group(1)
        (build / "assets" / f"depth_anything_v2_fp16.{recipe}.engine").write_bytes(
            b"engine"
        )
        clip = root / "clip"
        clip.mkdir()
        (clip / "model_source").mkdir()
        source_by_id = {}
        native_rows = []
        for frame_id in (0, 1, 2):
            path = clip / f"frame_{frame_id:05d}.png"
            path.write_bytes(f"frame-{frame_id}".encode())
            native_path = clip / "model_source" / f"frame_{frame_id:05d}.scrgb16"
            native_path.write_bytes(bytes([frame_id]) * 8)
            native_rows.append({
                "frame": frame_id,
                "path": f"model_source/frame_{frame_id:05d}.scrgb16",
                "size": 8,
                "mtime_ns": native_path.stat().st_mtime_ns,
                "sha256": depth_state_cache.artifact_cache.sha256_file(native_path),
                "preview": path.name,
                "preview_sha256": depth_state_cache.artifact_cache.sha256_file(path),
                "timestamp_seconds": frame_id / 30.0,
                "stats": {},
            })
            source_by_id[frame_id] = path
        native = depth_state_cache.native_hdr_capture
        semantic = {
            "contract": native.MANIFEST_CONTRACT,
            "capture_encoding": native.CAPTURE_ENCODING,
            "preview_encoding": native.PREVIEW_ENCODING,
            "width": 1,
            "height": 1,
            "row_pitch_bytes": 8,
            "source_video": {"sha256": "3" * 64},
            "conversion": {"contract_sha256": "4" * 64},
            "frames": [{
                key: row[key] for key in (
                    "frame", "path", "size", "sha256", "preview",
                    "preview_sha256", "timestamp_seconds",
                )
            } for row in native_rows],
        }
        manifest = {
            "schema": native.MANIFEST_SCHEMA,
            **semantic,
            "frames": native_rows,
            "frame_count": len(native_rows),
            "content_sha256": native.canonical_sha256(semantic),
        }
        (clip / native.MANIFEST_NAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return repo, build, clip, source_by_id

    @staticmethod
    def _content_rows(source_by_id):
        return [{
            "path": path.name,
            "size": path.stat().st_size,
            "sha256": depth_state_cache.artifact_cache.sha256_file(path),
        } for _frame_id, path in sorted(source_by_id.items())]

    def _identity(self, root, *, extra=()):
        repo, build, clip, source_by_id = self._fixture(root)
        return depth_state_cache.identity(
            repo=repo,
            build_dir=build,
            conf_sha256="1" * 16,
            executable_sha256="2" * 64,
            model="depth_anything_v2_fp16",
            clip_dir=clip,
            source_content_rows=self._content_rows(source_by_id),
            source_ids=[0, 1, 2],
            selected_frame_ids=[1, 2],
            extra=extra,
            runtime=self._runtime(),
        )

    def test_output_geometry_is_excluded_from_identity(self):
        with tempfile.TemporaryDirectory() as first_dir, \
                tempfile.TemporaryDirectory() as second_dir:
            first = self._identity(
                first_dir, extra=("--eye-w", "1280", "--eye-h", "720")
            )
            second = self._identity(
                second_dir, extra=("--eye-w", "960", "--eye-h", "540")
            )
            self.assertEqual(first, second)

    def test_input_condition_and_white_level_are_bound(self):
        with tempfile.TemporaryDirectory() as first_dir, \
                tempfile.TemporaryDirectory() as second_dir:
            hdr80 = self._identity(
                first_dir,
                extra=("--simulate-hdr", "--sdr-white-level-raw", "1000"),
            )
            hdr200 = self._identity(
                second_dir,
                extra=("--simulate-hdr", "--sdr-white-level-raw", "2500"),
            )
            self.assertNotEqual(hdr80, hdr200)

    def test_source_bytes_are_bound_without_absolute_path(self):
        with tempfile.TemporaryDirectory() as first_dir, \
                tempfile.TemporaryDirectory() as second_dir:
            first_fixture = self._fixture(first_dir)
            second_fixture = self._fixture(second_dir)

            def make(fixture):
                repo, build, clip, source_by_id = fixture
                return depth_state_cache.identity(
                    repo=repo, build_dir=build, conf_sha256="1" * 16,
                    executable_sha256="2" * 64,
                    model="depth_anything_v2_fp16", clip_dir=clip,
                    source_content_rows=self._content_rows(source_by_id),
                    source_ids=[0, 1, 2],
                    selected_frame_ids=[1, 2], extra=(), runtime=self._runtime(),
                )

            first = make(first_fixture)
            second = make(second_fixture)
            self.assertEqual(first, second)
            second_fixture[3][1].write_bytes(b"changed")
            changed = make(second_fixture)
            self.assertNotEqual(first, changed)

    def test_native_hdr_preview_shape_source_is_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._identity(root / "first", extra=("--native-hdr-scrgb",))
            fixture_root = root / "second"
            repo, build, clip, source_by_id = self._fixture(fixture_root)
            source_by_id[1].write_bytes(b"same-sidecar-different-preview")
            second = depth_state_cache.identity(
                repo=repo, build_dir=build, conf_sha256="1" * 16,
                executable_sha256="2" * 64,
                model="depth_anything_v2_fp16", clip_dir=clip,
                source_content_rows=self._content_rows(source_by_id),
                source_ids=[0, 1, 2],
                selected_frame_ids=[1, 2], extra=("--native-hdr-scrgb",),
                runtime=self._runtime(),
            )
            self.assertNotEqual(first, second)

    def test_full_native_admission_detects_same_stat_sidecar_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            _repo, _build, clip, _source_by_id = self._fixture(directory)
            sidecar = clip / "model_source" / "frame_00001.scrgb16"
            previous = sidecar.stat()
            sidecar.write_bytes(b"x" * previous.st_size)
            os.utime(
                sidecar,
                ns=(previous.st_atime_ns, previous.st_mtime_ns),
            )
            # The normal warm-path stat receipt is intentionally cheap, but a
            # cold cache admission must perform the full byte check.
            depth_state_cache.native_hdr_capture.validate_clip(clip, full=False)
            with self.assertRaisesRegex(RuntimeError, "model-source hash differs"):
                depth_state_cache.native_hdr_capture.validate_clip(clip, full=True)

    def test_native_runtime_is_bound(self):
        with tempfile.TemporaryDirectory() as first_dir, \
                tempfile.TemporaryDirectory() as second_dir:
            first_fixture = self._fixture(first_dir)
            second_fixture = self._fixture(second_dir)

            def make(fixture, runtime):
                repo, build, clip, source_by_id = fixture
                return depth_state_cache.identity(
                    repo=repo, build_dir=build, conf_sha256="1" * 16,
                    executable_sha256="2" * 64,
                    model="depth_anything_v2_fp16", clip_dir=clip,
                    source_content_rows=self._content_rows(source_by_id),
                    source_ids=[0, 1, 2],
                    selected_frame_ids=[1, 2], extra=(), runtime=runtime,
                )

            self.assertNotEqual(
                make(first_fixture, self._runtime("a")),
                make(second_fixture, self._runtime("b")),
            )

    def test_forced_runtime_refresh_bypasses_valid_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            runtime_file = build / "nvinfer.dll"
            runtime_file.write_bytes(b"runtime")
            paths = {"inference": runtime_file}

            def snapshot(tag):
                return {
                    "identity": {"runtime": tag},
                    "files": {
                        "inference": {
                            "path": str(runtime_file.resolve()),
                            "tag": tag,
                        },
                    },
                }

            with mock.patch.object(
                    depth_state_cache, "_runtime_paths",
                    return_value=(build, paths)), \
                    mock.patch.object(
                        depth_state_cache, "_build_runtime_snapshot",
                        side_effect=(snapshot("first"), snapshot("second")),
                    ) as build_snapshot, \
                    mock.patch.object(
                        depth_state_cache, "verify_runtime_snapshot",
                        side_effect=lambda value: value["identity"],
                    ):
                first = depth_state_cache.runtime_snapshot(build)
                warm = depth_state_cache.runtime_snapshot(build)
                refreshed = depth_state_cache.runtime_snapshot(
                    build, force_refresh=True
                )

            self.assertEqual(first, warm)
            self.assertNotEqual(first, refreshed)
            self.assertEqual(build_snapshot.call_count, 2)

    def test_transitive_depth_shader_includes_are_discovered_and_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, build, clip, source_by_id = self._fixture(directory)
            entry = (
                build / "assets" / "shaders" / "directx" /
                depth_state_cache.DEPTH_SHADER_ENTRY_FILES[0]
            )
            nested = entry.parent / "include" / "future_nested.hlsl"
            nested.parent.mkdir(parents=True, exist_ok=True)
            entry.write_text(
                '#include "include/future_nested.hlsl"\n', encoding="utf-8"
            )
            nested.write_text("static const float nested = 1;\n", encoding="utf-8")

            def make():
                return depth_state_cache.identity(
                    repo=repo, build_dir=build, conf_sha256="1" * 16,
                    executable_sha256="2" * 64,
                    model="depth_anything_v2_fp16", clip_dir=clip,
                    source_content_rows=self._content_rows(source_by_id),
                    source_ids=[0, 1, 2], selected_frame_ids=[1, 2],
                    extra=(), runtime=self._runtime(),
                )

            first = make()
            self.assertIn(
                "include/future_nested.hlsl",
                first["code"]["depth_shader_sources"],
            )
            nested.write_text("static const float nested = 2;\n", encoding="utf-8")
            self.assertNotEqual(first, make())

    def test_cpp_reader_validates_runtime_scene_types_before_replay(self):
        source = (
            Path(__file__).resolve().parents[2] /
            "src" / "sbs_depth_state_sequence.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn('frame, "runtime_scene_id"', source)
        self.assertIn('frame, "scene_age_float32_bits"', source)
        self.assertIn('frame["subject_initialized"].is_boolean()', source)
        self.assertIn('frame["hard_cut"].is_boolean()', source)
        self.assertIn('frame["scene_start"].is_boolean()', source)

    def test_cpp_reader_bounds_unsigned_values_before_narrowing(self):
        source = (
            Path(__file__).resolve().parents[2] /
            "src" / "sbs_depth_state_sequence.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("json_unsigned_bounded", source)
        self.assertIn("std::numeric_limits<std::uint32_t>::max()", source)
        self.assertNotIn('state.value("depth_width", 0u)', source)

    def test_validated_sequence_returns_outer_pinned_inner_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = self._identity(root / "identity")
            sequence = root / "sequence"
            sequence.mkdir()
            manifest = sequence / depth_state_cache.INNER_MANIFEST
            manifest.write_bytes(b'{"schema":1}\n')
            (sequence / "payload.bin").write_bytes(b"payload")
            store = depth_state_cache.cache(root / "cache")
            store.publish(identity, sequence)

            hit = depth_state_cache.validated_sequence(store, identity)
            self.assertEqual(
                hit["inner_manifest_sha256"],
                depth_state_cache.artifact_cache.sha256_file(
                    hit["payload"] / depth_state_cache.INNER_MANIFEST
                ),
            )
            self.assertRegex(
                hit["outer_payload_manifest_sha256"], r"^[0-9a-f]{64}$"
            )


if __name__ == "__main__":
    unittest.main()
