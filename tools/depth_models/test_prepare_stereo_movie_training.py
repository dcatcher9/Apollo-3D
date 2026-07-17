#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import prepare_stereo_movie_training as stereo


class FakeCapture:
    def __init__(self, frames, fps=4.0):
        self.frames = frames
        self.fps = fps
        self.index = 0

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


class StereoMoviePreparationTests(unittest.TestCase):
    def test_rejects_invalid_programmatic_layout_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "movie.bin"
            video.write_bytes(b"stereo-video")
            for keyword, value, message in (
                ("layout", "diagonal", "stereo layout"),
                ("eye_order", "unknown-first", "stereo eye order"),
            ):
                with self.subTest(keyword=keyword):
                    with self.assertRaisesRegex(RuntimeError, message):
                        stereo.prepare(
                            video, root / f"prepared-{keyword}",
                            "Stereo", "stereo_test", **{keyword: value},
                        )

    def test_rejects_unsafe_domain_before_touching_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "movie.bin"
            video.write_bytes(b"stereo-video")
            for index, domain in enumerate((
                    "../escape", "..\\escape", "nested/name", "C:escape",
                    ".", "CON")):
                output = root / f"prepared-{index}"
                with self.subTest(domain=domain):
                    with self.assertRaisesRegex(
                            RuntimeError, "safe .*path component"):
                        stereo.prepare(video, output, "Stereo", domain)
                    self.assertFalse(output.exists())

    def test_direct_child_guard_rejects_external_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "prepared"
            output.mkdir()
            with self.assertRaisesRegex(RuntimeError, "outside output"):
                stereo._assert_direct_child(root / "external", output)

    def test_writes_sparse_targets_with_full_cadence_context(self):
        first_eye = np.zeros((8, 12, 3), np.uint8)
        second_eye = np.full((8, 12, 3), 32, np.uint8)
        packed = np.concatenate((first_eye, second_eye), axis=1)
        frames = [packed] * 5
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "movie.bin"
            video.write_bytes(b"stereo-video")
            output = root / "prepared"
            color_probe = {
                "dataset_color_contract": "decoded-sdr-bgr8",
                "admission": "probed-no-hdr-signals",
            }
            with (mock.patch.object(
                      stereo.cv2, "VideoCapture",
                      return_value=FakeCapture(frames)
                  ), mock.patch.object(
                      stereo.video_color, "probe_sdr_input",
                      return_value=color_probe,
                  )):
                manifest = stereo.prepare(
                    video, output, "Stereo", "stereo_test",
                    layout="side-by-side", sample_fps=2.0,
                    output_width=12,
                )
            clip = output / manifest["sequences"][0]["clip"]
            self.assertEqual(len(list(clip.glob("frame_*.png"))), 5)
            labels = json.loads(
                (clip / "label_frames.json").read_text(encoding="utf-8")
            )
            self.assertEqual(labels, {"schema": 1, "frame_ids": [0, 2, 4]})
            self.assertEqual(
                sorted(int(path.stem.removeprefix("frame_"))
                       for path in (clip / "gt_right").glob("frame_*.png")),
                labels["frame_ids"],
            )
            meta = json.loads(
                (clip / "meta.json").read_text(encoding="utf-8")
            )
            self.assertEqual(meta["temporal_contract"], "full-cadence-shot")
            self.assertEqual(meta["source_kind"], "authored-stereo")
            self.assertEqual(meta["write_worker_count"], 4)
            self.assertEqual(meta["source_color_probe"], color_probe)
            self.assertEqual(manifest["write_worker_count"], 4)
            self.assertEqual(manifest["source_color_probe"], color_probe)

    def test_write_failure_is_not_published(self):
        eye = np.zeros((8, 12, 3), np.uint8)
        packed = np.concatenate((eye, eye), axis=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "movie.bin"
            video.write_bytes(b"stereo-video")
            output = root / "prepared"
            capture = FakeCapture([packed] * 4)
            color_probe = {
                "dataset_color_contract": "decoded-sdr-bgr8",
            }

            def fail_one(path, image):
                del image
                return "frame_00001" not in str(path)

            with (mock.patch.object(
                      stereo.cv2, "VideoCapture", return_value=capture
                  ), mock.patch.object(
                      stereo.video_color, "probe_sdr_input",
                      return_value=color_probe,
                  ), mock.patch.object(
                      stereo.cv2, "imwrite", side_effect=fail_one
                  )):
                with self.assertRaisesRegex(RuntimeError, "cannot write"):
                    stereo.prepare(
                        video, output, "Stereo", "stereo_test",
                        layout="side-by-side", output_width=12,
                        write_workers=2,
                    )
            self.assertTrue(capture.released)
            self.assertFalse(output.exists())

    def test_rejects_hdr_before_decoding_or_writing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "movie.bin"
            video.write_bytes(b"stereo-video")
            output = root / "prepared"
            with (mock.patch.object(
                      stereo.video_color, "probe_sdr_input",
                      side_effect=RuntimeError("HDR transfer 'smpte2084'")
                  ), mock.patch.object(stereo.cv2, "VideoCapture") as capture):
                with self.assertRaisesRegex(RuntimeError, "HDR transfer"):
                    stereo.prepare(video, output, "Stereo", "stereo_test")
            capture.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
