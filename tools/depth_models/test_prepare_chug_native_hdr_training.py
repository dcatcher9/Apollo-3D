#!/usr/bin/env python3

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import prepare_chug_native_hdr_training as chug


class ChugNativeHdrPreparationTests(unittest.TestCase):
    def test_each_window_writes_its_own_name_and_label_frames(self):
        selection = {
            "contract": chug.FRAME_SELECTION_CONTRACT,
            "flow_support_selection_contract":
                chug.FLOW_SUPPORT_SELECTION_CONTRACT,
            "source_frame_count": 6,
            "temporal_window_radius": 1,
            "source_frame_ids": list(range(6)),
            "source_label_frame_ids": [0, 5],
            "windows": [
                {
                    "window_index": 0,
                    "source_frame_ids": [0, 1, 2],
                    "source_label_frame_id": 0,
                    "frame_ids": [0, 1, 2],
                    "label_frame_ids": [0],
                    "temporal_evidence_selection": {
                        "contract": chug.FLOW_SUPPORT_SELECTION_CONTRACT,
                        "nominal_source_label_frame_id": 0,
                        "selected_pair_flow_support": 0.5,
                    },
                },
                {
                    "window_index": 1,
                    "source_frame_ids": [3, 4, 5],
                    "source_label_frame_id": 5,
                    "frame_ids": [0, 1, 2],
                    "label_frame_ids": [2],
                    "temporal_evidence_selection": {
                        "contract": chug.FLOW_SUPPORT_SELECTION_CONTRACT,
                        "nominal_source_label_frame_id": 5,
                        "selected_pair_flow_support": 0.5,
                    },
                },
            ],
        }

        class FakeProcess:
            def __init__(self):
                # Six 1x1 planar GBR float32 frames.
                self.stdout = io.BytesIO(b"\0" * (6 * 3 * 4))
                self.stderr = io.BytesIO()

            @staticmethod
            def wait(timeout):
                return 0

            @staticmethod
            def kill():
                return None

        row = {
            "video_id": "video-a",
            "video_path": Path("source.mp4"),
            "source_frame_count": 6,
            "probed_frame_rate": 30.0,
            "split": "training",
            "capture_group_id": "group-a",
            "content_id": "content-a",
            "download": {"bytes": 1, "sha256": "a" * 64},
            "audit": {"width": 1, "height": 1},
        }
        conversion = {"decoder": {"filter": "format=gbrpf32le"}}

        with tempfile.TemporaryDirectory() as temporary:
            split_root = Path(temporary) / "training"
            with (
                mock.patch.object(
                    chug, "_frame_selection_plan", return_value=selection
                ),
                mock.patch.object(
                    chug.subprocess, "Popen", return_value=FakeProcess()
                ),
            ):
                chug._prepare_clip(  # noqa: SLF001
                    row,
                    split_root,
                    Path("ffmpeg.exe"),
                    conversion,
                    "b" * 64,
                    1,
                    1,
                    2,
                    True,
                    selection,
                )

            expected = {
                0: [0],
                1: [2],
            }
            for window_index, frame_ids in expected.items():
                clip_root = split_root / f"chug_pq_video-a_w{window_index:02d}"
                metadata = json.loads(
                    (clip_root / "meta.json").read_text(encoding="utf-8")
                )
                labels = json.loads(
                    (clip_root / "label_frames.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    metadata["name"],
                    f"CHUG native PQ video-a window {window_index:02d}",
                )
                self.assertEqual(labels["frame_ids"], frame_ids)
                self.assertEqual(
                    metadata["frame_selection"]["window_index"],
                    window_index,
                )
                self.assertEqual(
                    metadata["frame_selection"]["label_frame_ids"],
                    labels["frame_ids"],
                )

    def test_label_windows_preserve_source_time_and_local_cadence(self):
        plan = chug._frame_selection_plan(300, 5)  # noqa: SLF001

        self.assertEqual(
            plan["source_label_frame_ids"], [50, 100, 150, 199, 249]
        )
        self.assertEqual(len(plan["source_frame_ids"]), 15)
        self.assertEqual(len(plan["windows"]), 5)
        for source_label, window in zip(
                plan["source_label_frame_ids"], plan["windows"]):
            self.assertEqual(
                window["source_frame_ids"],
                [source_label - 1, source_label, source_label + 1],
            )
            self.assertEqual(window["frame_ids"], [0, 1, 2])
            self.assertEqual(window["label_frame_ids"], [1])

        metadata = chug._frame_selection_metadata(  # noqa: SLF001
            plan, plan["windows"][0], 30.0
        )
        self.assertEqual(metadata["frames"][1], {
            "frame": 1,
            "source_frame": 50,
            "source_timestamp_seconds": 50.0 / 30.0,
        })

    def test_label_windows_never_use_source_endpoints_as_centers(self):
        plan = chug._frame_selection_plan(7, 5)  # noqa: SLF001

        self.assertEqual(plan["source_label_frame_ids"], [1, 2, 3, 4, 5])
        self.assertEqual(plan["source_frame_ids"], list(range(7)))
        self.assertEqual(
            [window["label_frame_ids"] for window in plan["windows"]],
            [[1], [1], [1], [1], [1]],
        )

    def test_flow_curation_uses_nearest_valid_previous_pair(self):
        row = {
            "video_id": "fireworks",
            "video_path": Path("source.mp4"),
            "source_frame_count": 100,
            "audit": {"width": 1, "height": 1},
        }

        def decode(_row, _ffmpeg, _conversion, _width, _height, frame_ids):
            return {frame_id: object() for frame_id in frame_ids}

        supports = {
            33: 0.05,
            32: 0.25,
            66: 0.40,
        }
        with (
            mock.patch.object(
                chug, "_decode_flow_previews", side_effect=decode
            ),
            mock.patch.object(
                chug, "_previous_pair_support",
                side_effect=lambda _previews, center: supports.get(center, 0.0),
            ),
        ):
            plan = chug._curated_frame_selection_plan(  # noqa: SLF001
                row, Path("ffmpeg.exe"), {"decoder": {"filter": "test"}},
                1280, 720, 2,
            )

        self.assertEqual(plan["initial_source_label_frame_ids"], [33, 66])
        self.assertEqual(plan["source_label_frame_ids"], [32, 66])
        first = plan["windows"][0]
        self.assertEqual(first["source_frame_ids"], [31, 32, 33])
        evidence = first["temporal_evidence_selection"]
        self.assertEqual(evidence["nominal_source_label_frame_id"], 33)
        self.assertEqual(evidence["selected_source_label_frame_id"], 32)
        self.assertEqual(evidence["selected_offset_frames"], -1)
        self.assertEqual(evidence["selected_pair_flow_support"], 0.25)
        self.assertEqual(
            evidence["flow_support_contract"],
            chug.FLOW_SUPPORT_CONTRACT,
        )
        self.assertEqual(
            evidence["flow_support_metric_sha256"],
            chug.flow_support_metric_sha256(),
        )

    def test_flow_curation_fails_when_bounded_search_has_no_valid_pair(self):
        row = {
            "video_id": "untrackable",
            "video_path": Path("source.mp4"),
            "source_frame_count": 30,
            "audit": {"width": 1, "height": 1},
        }

        def decode(_row, _ffmpeg, _conversion, _width, _height, frame_ids):
            return {frame_id: object() for frame_id in frame_ids}

        with (
            mock.patch.object(
                chug, "_decode_flow_previews", side_effect=decode
            ),
            mock.patch.object(
                chug, "_previous_pair_support", return_value=0.01
            ),
            self.assertRaisesRegex(RuntimeError, "no flow-valid center"),
        ):
            chug._curated_frame_selection_plan(  # noqa: SLF001
                row, Path("ffmpeg.exe"), {"decoder": {"filter": "test"}},
                1280, 720, 1,
            )

    def test_decode_selects_only_authenticated_source_frames(self):
        command = chug._decode_command(  # noqa: SLF001
            Path("ffmpeg.exe"), Path("clip.mp4"), "format=gbrpf32le",
            [49, 50, 51],
        )

        filter_value = command[command.index("-vf") + 1]
        self.assertEqual(
            filter_value,
            "select=eq(n\\,49)+eq(n\\,50)+eq(n\\,51),format=gbrpf32le",
        )

    def test_retention_summary_reports_exact_raw_storage(self):
        selected = {
            "training": [
                {"source_frame_count": 300},
                {"source_frame_count": 180},
            ],
            "development": [{"source_frame_count": 150}],
        }
        summary = chug._retention_summary(  # noqa: SLF001
            selected, 1280, 720, 5
        )

        self.assertEqual(summary["total"]["source_frames"], 630)
        self.assertEqual(summary["total"]["retained_frames"], 45)
        self.assertEqual(summary["splits"]["training"]["window_clips"], 10)
        self.assertEqual(
            summary["splits"]["development"]["window_clips"], 5
        )
        expected_bytes = 45 * 1280 * 720 * 8
        self.assertEqual(
            summary["total"]["raw_scrgb16_retained_bytes"], expected_bytes
        )
        self.assertEqual(
            summary["total"]["raw_scrgb16_full_bytes"],
            630 * 1280 * 720 * 8,
        )

    def test_receipt_frame_count_uses_authenticated_cadence(self):
        row = {
            "probed_duration_seconds": 10.01,
            "probed_frame_rate": 30000 / 1001,
        }
        self.assertEqual(chug._receipt_frame_count(row), 300)  # noqa: SLF001

    def test_dataset_counts_master_once_and_keeps_window_grouping(self):
        rows = []
        for window_index in range(2):
            rows.append({
                "clip": f"clip_w{window_index:02d}",
                "frames": 3,
                "source_frames": 3,
                "master_source_frames": 300,
                "source_frame_rate": 30.0,
                "label_frames": 1,
                "capture_group_id": "group-a",
                "video_id": "video-a",
                "window_index": window_index,
                "source_label_frame_id": 50 + window_index * 50,
            })
        manifest = chug._dataset_manifest(  # noqa: SLF001
            "training", rows, {}, "conversion-sha"
        )

        self.assertEqual(manifest["window_clip_count"], 2)
        self.assertEqual(manifest["source_video_count"], 1)
        self.assertEqual(manifest["frame_count"], 6)
        self.assertEqual(manifest["master_source_frame_count"], 300)
        self.assertEqual(
            {row["capture_group_id"] for row in manifest["sequences"]},
            {"group-a"},
        )
        self.assertEqual(
            {row["split"] for row in manifest["sequences"]}, {"training"}
        )


if __name__ == "__main__":
    unittest.main()
