#!/usr/bin/env python3

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import prepare_monocular_movie_training as mono


class FakeCapture:
    def __init__(self, frames, fps):
        self.frames = [frame.copy() for frame in frames]
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, name):
        if name == cv2.CAP_PROP_FPS:
            return self.fps
        if name == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        return 0

    def set(self, name, value):
        if name == cv2.CAP_PROP_POS_FRAMES:
            self.index = int(value)
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def release(self):
        self.released = True


class DeferredFuture:
    def __init__(self, executor):
        self.executor = executor
        self.completed = False

    def result(self):
        if not self.completed:
            self.completed = True
            self.executor.outstanding -= 1

    def cancel(self):
        self.result()


class DeferredExecutor:
    def __init__(self):
        self.outstanding = 0
        self.peak_outstanding = 0

    def submit(self, function, *args):
        del function, args
        self.outstanding += 1
        self.peak_outstanding = max(
            self.peak_outstanding, self.outstanding
        )
        return DeferredFuture(self)

    def shutdown(self, wait=True, cancel_futures=False):
        del wait, cancel_futures


class MonocularMoviePreparationTests(unittest.TestCase):
    @staticmethod
    def directory_bytes(root):
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*")) if path.is_file()
        }

    def run_prepare(self, root, frames, **kwargs):
        video = root / "movie.bin"
        video.write_bytes(b"test-video")
        output = root / "prepared"
        capture = FakeCapture(frames, fps=4.0)
        color_probe = {
            "dataset_color_contract": "decoded-sdr-bgr8",
            "admission": "probed-no-hdr-signals",
        }
        with (mock.patch.object(mono.cv2, "VideoCapture", return_value=capture),
              mock.patch.object(
                  mono.video_color, "probe_sdr_input", return_value=color_probe
              )):
            manifest = mono.prepare(
                video, output, "Test movie", "test_movie",
                sample_fps=2.0, cut_threshold=0.18, output_width=12,
                min_context_frames=kwargs.get("min_context_frames", 2),
                write_workers=kwargs.get("write_workers", 4),
            )
        return output, manifest, capture

    def test_preserves_full_cadence_and_selects_sparse_label_frames(self):
        black = np.zeros((8, 12, 3), np.uint8)
        white = np.full((8, 12, 3), 255, np.uint8)
        frames = [black] * 4 + [white] * 4
        with tempfile.TemporaryDirectory() as directory:
            output, manifest, capture = self.run_prepare(
                Path(directory), frames
            )
            self.assertTrue(capture.released)
            self.assertEqual(manifest["context_frame_count"], 8)
            self.assertEqual(manifest["label_frame_count"], 4)
            self.assertEqual(manifest["write_worker_count"], 4)
            self.assertEqual(len(manifest["sequences"]), 2)
            for row in manifest["sequences"]:
                clip = output / row["clip"]
                self.assertEqual(len(list(clip.glob("frame_*.png"))), 4)
                labels = json.loads(
                    (clip / "label_frames.json").read_text(encoding="utf-8")
                )
                self.assertEqual(labels["schema"], 1)
                self.assertEqual(labels["frame_ids"], [0, 2])
                meta = json.loads(
                    (clip / "meta.json").read_text(encoding="utf-8")
                )
                self.assertEqual(meta["source_kind"], "mono-video")
                self.assertEqual(meta["temporal_contract"], "full-cadence-shot")
                self.assertTrue(meta["required_temporal_evidence"])
                self.assertEqual(meta["write_worker_count"], 4)
                self.assertFalse((clip / "gt_right").exists())

    def test_single_worker_mode_is_synchronous(self):
        frame = np.zeros((8, 12, 3), np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            output, manifest, _ = self.run_prepare(
                Path(directory), [frame] * 3, write_workers=1
            )
            self.assertEqual(manifest["write_worker_count"], 1)
            self.assertEqual(
                len(list((output / "test_movie_shot_0000").glob("frame_*.png"))),
                3,
            )

    def test_writer_never_queues_more_than_twice_worker_count(self):
        executor = DeferredExecutor()
        with mock.patch.object(
                mono, "ThreadPoolExecutor", return_value=executor):
            writer = mono._BoundedPngWriter(2)
            for index in range(20):
                writer.submit(Path(f"frame_{index}.png"), np.zeros((1, 1, 3)), [])
            writer.close()
        self.assertLessEqual(executor.peak_outstanding, 4)
        self.assertEqual(writer.peak_pending, 4)

    def test_write_failure_removes_incomplete_dataset(self):
        frame = np.zeros((8, 12, 3), np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "movie.bin"
            video.write_bytes(b"test-video")
            output = root / "prepared"
            capture = FakeCapture([frame] * 4, fps=4.0)
            color_probe = {
                "dataset_color_contract": "decoded-sdr-bgr8",
                "admission": "probed-no-hdr-signals",
            }

            def fail_one(path, image, params):
                del image, params
                return "frame_00001" not in str(path)

            with (mock.patch.object(
                      mono.cv2, "VideoCapture", return_value=capture
                  ), mock.patch.object(
                      mono.video_color, "probe_sdr_input",
                      return_value=color_probe,
                  ), mock.patch.object(
                      mono.cv2, "imwrite", side_effect=fail_one
                  )):
                with self.assertRaisesRegex(RuntimeError, "cannot write"):
                    mono.prepare(
                        video, output, "Test", "test",
                        min_context_frames=2, write_workers=2,
                    )
            self.assertTrue(capture.released)
            self.assertFalse(output.exists())

    def test_drops_single_frame_shot_from_temporal_supervision(self):
        black = np.zeros((8, 12, 3), np.uint8)
        white = np.full((8, 12, 3), 255, np.uint8)
        frames = [black] * 3 + [white] + [black] * 3
        with tempfile.TemporaryDirectory() as directory:
            output, manifest, _ = self.run_prepare(
                Path(directory), frames, min_context_frames=2
            )
            self.assertEqual(len(manifest["sequences"]), 2)
            self.assertEqual(manifest["dropped_shot_count"], 1)
            self.assertEqual(
                manifest["dropped_shots"][0]["reason"],
                "insufficient-temporal-context",
            )
            self.assertFalse(any(output.glob(".*.partial")))

    def test_rejects_nonempty_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "prepared"
            output.mkdir()
            (output / "existing").write_text("keep", encoding="utf-8")
            video = root / "movie.bin"
            video.write_bytes(b"test-video")
            with self.assertRaisesRegex(RuntimeError, "output must be empty"):
                mono.prepare(video, output, "Test", "test")

    def test_rejects_unsafe_domain_before_touching_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "movie.bin"
            video.write_bytes(b"test-video")
            for index, domain in enumerate((
                    "../escape", "..\\escape", "nested/name", "C:escape",
                    ".", "CON")):
                output = root / f"prepared-{index}"
                with self.subTest(domain=domain):
                    with self.assertRaisesRegex(
                            RuntimeError, "safe .*path component"):
                        mono.prepare(video, output, "Test", domain)
                    self.assertFalse(output.exists())

    def test_direct_child_guard_rejects_external_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "prepared"
            output.mkdir()
            with self.assertRaisesRegex(RuntimeError, "outside output"):
                mono._assert_direct_child(root / "external", output)

    def test_rejects_hdr_before_decoding_or_writing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "movie.bin"
            video.write_bytes(b"test-video")
            output = root / "prepared"
            with mock.patch.object(
                    mono.video_color, "probe_sdr_input",
                    side_effect=RuntimeError("HDR transfer 'smpte2084'")):
                with self.assertRaisesRegex(RuntimeError, "HDR transfer"):
                    mono.prepare(video, output, "Test", "test")
            self.assertFalse(output.exists())

    def test_authenticated_cache_reuses_prepared_frames_without_decoding(self):
        frame = np.zeros((8, 12, 3), np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "movie.bin"
            video.write_bytes(b"test-video")
            output = root / "prepared"
            cache = root / "cache"
            events = []
            color_probe = {
                "dataset_color_contract": "decoded-sdr-bgr8",
                "admission": "probed-no-hdr-signals",
            }
            with (mock.patch.object(
                      mono.cv2, "VideoCapture",
                      return_value=FakeCapture([frame] * 3, fps=4.0),
                  ), mock.patch.object(
                      mono.video_color, "probe_sdr_input",
                      return_value=color_probe,
                  )):
                first = mono.prepare(
                    video, output, "Test", "test", output_width=12,
                    min_context_frames=2, preprocess_cache=cache,
                    cache_observer=events.append,
                )
            prepared_bytes = self.directory_bytes(output)
            shutil.rmtree(output)
            with (mock.patch.object(
                      mono.cv2, "VideoCapture",
                      side_effect=AssertionError("cache hit decoded source"),
                  ), mock.patch.object(
                      mono.video_color, "probe_sdr_input",
                      return_value=color_probe,
                  )):
                second = mono.prepare(
                    video, output, "Test", "test", output_width=12,
                    min_context_frames=2, preprocess_cache=cache,
                    cache_observer=events.append,
                )
            self.assertEqual(second, first)
            self.assertEqual(
                self.directory_bytes(output), prepared_bytes
            )
            self.assertEqual([event["status"] for event in events], [
                "published", "hit",
            ])
            self.assertEqual(events[0]["key_sha256"], events[1]["key_sha256"])
            self.assertEqual(events[0]["payload_bytes"],
                             events[1]["payload_bytes"])
            self.assertGreater(events[0]["payload_bytes"], 0)
            self.assertEqual(
                len(list((output / "test_shot_0000").glob("frame_*.png"))),
                3,
            )

    def test_raw_cache_reuses_decode_when_labels_threshold_and_metadata_change(self):
        black = np.zeros((8, 12, 3), np.uint8)
        gray = np.full((8, 12, 3), 128, np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "movie.bin"
            video.write_bytes(b"test-video")
            cache = root / "cache"
            color_probe = {
                "dataset_color_contract": "decoded-sdr-bgr8",
                "admission": "probed-no-hdr-signals",
            }
            first_output = root / "prepared-first"
            with (mock.patch.object(
                      mono.cv2, "VideoCapture",
                      return_value=FakeCapture([black] * 4 + [gray] * 4, fps=4.0),
                  ), mock.patch.object(
                      mono.video_color, "probe_sdr_input",
                      return_value=color_probe,
                  )):
                first = mono.prepare(
                    video, first_output, "First title", "movie_a",
                    sample_fps=2.0, cut_threshold=0.18, output_width=12,
                    min_context_frames=2, write_workers=1,
                    preprocess_cache=cache,
                )
            self.assertEqual(first["shot_count"], 2)

            second_output = root / "prepared-second"
            with (mock.patch.object(
                      mono.cv2, "VideoCapture",
                      side_effect=AssertionError("raw cache reopened video"),
                  ), mock.patch.object(
                      mono.video_color, "probe_sdr_input",
                      return_value=color_probe,
                  )):
                second = mono.prepare(
                    video, second_output, "Retitled", "movie_b",
                    sample_fps=1.0, cut_threshold=0.8, output_width=12,
                    production_id="new-production", homepage="https://example.test",
                    license_name="new-license", policy_role="new-role",
                    global_policy_weight=2.0, min_context_frames=3,
                    write_workers=3, preprocess_cache=cache,
                )
            self.assertEqual(second["shot_count"], 1)
            self.assertEqual(second["requested_label_fps"], 1.0)
            self.assertEqual(second["dataset"], "Retitled")
            clip = second_output / second["sequences"][0]["clip"]
            labels = json.loads(
                (clip / "label_frames.json").read_text(encoding="utf-8")
            )
            self.assertEqual(labels["frame_ids"], [0, 4])
            self.assertEqual(len(list(clip.glob("frame_*.png"))), 8)
            payloads = list((cache / "objects").glob("*/*/payload"))
            self.assertEqual(len(payloads), 1)
            self.assertTrue((payloads[0] / "raw_full_cadence.json").is_file())
            self.assertFalse((payloads[0] / "meta.json").exists())
            self.assertFalse(any(payloads[0].glob("*_shot_*")))
            cache_manifest = json.loads(
                (payloads[0].parent / "cache_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                set(cache_manifest["identity"]["code"]),
                {"movie_raw_producer", "native_runtime_identity"},
            )

    def test_cached_and_direct_cold_paths_are_byte_identical(self):
        black = np.zeros((8, 12, 3), np.uint8)
        gray = np.full((8, 12, 3), 96, np.uint8)
        frames = [black] * 3 + [gray] * 3
        color_probe = {
            "dataset_color_contract": "decoded-sdr-bgr8",
            "admission": "probed-no-hdr-signals",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "movie.bin"
            video.write_bytes(b"test-video")
            direct = root / "direct"
            cached = root / "cached"
            cache = root / "cache"
            common = {
                "sample_fps": 2.0,
                "cut_threshold": 0.18,
                "output_width": 12,
                "min_context_frames": 2,
                "write_workers": 1,
            }
            with (mock.patch.object(
                      mono.cv2, "VideoCapture",
                      return_value=FakeCapture(frames, fps=4.0),
                  ), mock.patch.object(
                      mono.video_color, "probe_sdr_input",
                      return_value=color_probe,
                  )):
                mono.prepare(video, direct, "Test", "test", **common)
            with (mock.patch.object(
                      mono.cv2, "VideoCapture",
                      return_value=FakeCapture(frames, fps=4.0),
                  ), mock.patch.object(
                      mono.video_color, "probe_sdr_input",
                      return_value=color_probe,
                  )):
                mono.prepare(
                    video, cached, "Test", "test",
                    preprocess_cache=cache, **common,
                )
            self.assertEqual(
                self.directory_bytes(cached), self.directory_bytes(direct)
            )

    def test_cache_publication_rejects_source_mutation_during_decode(self):
        black = np.zeros((8, 12, 3), np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "movie.bin"
            video.write_bytes(b"authenticated source")
            capture = FakeCapture([black] * 3, fps=4.0)
            original_read = capture.read
            changed = False

            def mutating_read():
                nonlocal changed
                if not changed:
                    video.write_bytes(b"changed source")
                    changed = True
                return original_read()

            capture.read = mutating_read
            color_probe = {
                "dataset_color_contract": "decoded-sdr-bgr8",
                "admission": "probed-no-hdr-signals",
            }
            with (mock.patch.object(
                      mono.cv2, "VideoCapture", return_value=capture,
                  ), mock.patch.object(
                      mono.video_color, "probe_sdr_input",
                      return_value=color_probe,
                  )):
                with self.assertRaisesRegex(RuntimeError, "source changed"):
                    mono.prepare(
                        video, root / "prepared", "Test", "test",
                        output_width=12, min_context_frames=2,
                        write_workers=1, preprocess_cache=root / "cache",
                    )
            self.assertFalse((root / "prepared").exists())
            self.assertFalse(any((root / "cache").glob("objects/*/*")))

    def test_cache_copy_rehashes_against_original_receipt(self):
        black = np.zeros((8, 12, 3), np.uint8)
        color_probe = {
            "dataset_color_contract": "decoded-sdr-bgr8",
            "admission": "probed-no-hdr-signals",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "movie.bin"
            video.write_bytes(b"authenticated source")
            output = root / "prepared"
            cache = root / "cache"
            with (mock.patch.object(
                      mono.cv2, "VideoCapture",
                      return_value=FakeCapture([black] * 3, fps=4.0),
                  ), mock.patch.object(
                      mono.video_color, "probe_sdr_input",
                      return_value=color_probe,
                  )):
                mono.prepare(
                    video, output, "Test", "test", output_width=12,
                    min_context_frames=2, write_workers=1,
                    preprocess_cache=cache,
                )
            shutil.rmtree(output)
            original_copy = shutil.copy2
            corrupted = False

            def corrupt_then_copy(source, destination, *args, **kwargs):
                nonlocal corrupted
                source = Path(source)
                if not corrupted and source.suffix == ".png":
                    source.write_bytes(b"corrupt cache payload")
                    corrupted = True
                return original_copy(source, destination, *args, **kwargs)

            with (mock.patch.object(
                      mono.cv2, "VideoCapture",
                      side_effect=AssertionError("cache hit decoded source"),
                  ), mock.patch.object(
                      mono.video_color, "probe_sdr_input",
                      return_value=color_probe,
                  ), mock.patch.object(
                      mono.shutil, "copy2", side_effect=corrupt_then_copy,
                  )):
                with self.assertRaisesRegex(RuntimeError, "changed during"):
                    mono.prepare(
                        video, output, "Test", "test", output_width=12,
                        min_context_frames=2, write_workers=1,
                        preprocess_cache=cache,
                    )
            self.assertFalse(output.exists())

    def test_cache_rejects_test_split_before_touching_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "train/development only"):
                mono.prepare(
                    root / "sealed-does-not-exist.mp4", root / "output",
                    "Test", "test", split="test",
                    preprocess_cache=root / "cache",
                )


if __name__ == "__main__":
    unittest.main()
