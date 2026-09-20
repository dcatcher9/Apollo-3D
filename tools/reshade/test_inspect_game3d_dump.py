import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from PIL import Image

import inspect_game3d_dump as reader


class GameDumpReaderTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def artifact(self, kind, values, fmt=41, channels=1, artifact_id=None):
        payload = np.array(values, dtype=reader.FORMATS[fmt][0]).tobytes()
        name = kind + ".bin"
        (self.root / name).write_bytes(payload)
        return dict(kind=kind, file=name, dxgi_format=fmt, width=len(values) // channels,
                    height=1, row_bytes=len(payload), byte_count=len(payload), artifact_id=artifact_id)

    def inspect(self, artifacts, metadata=None):
        (self.root / "manifest.json").write_text(json.dumps(dict(
            schema="sunshine.game3d.dump.v1", status="complete", artifacts=artifacts,
            producer_metadata=metadata or {})))
        return reader.inspect(self.root, previews=True)

    def png(self, name):
        with Image.open(self.root / "previews" / name) as image:
            return np.array(image).reshape(-1, 3).tolist()

    def test_raw_float_depth_is_not_clamped_or_inverted(self):
        values = (-1.25, 0.0, 0.75, 12.5)
        (self.root / "depth.bin").write_bytes(struct.pack("<4f", *values))
        desc = dict(kind="raw_depth", file="depth.bin", dxgi_format=41,
                    width=2, height=2, row_bytes=8, byte_count=16)
        self.assertEqual(reader.read_artifact(self.root, desc).ravel().tolist(), list(values))
        (self.root / "manifest.json").write_text(json.dumps(dict(
            schema="sunshine.game3d.dump.v1", status="complete", artifacts=[desc])))
        report = reader.inspect(self.root, previews=True)
        self.assertEqual(report["artifacts"][0]["min"], -1.25)
        self.assertEqual(report["artifacts"][0]["max"], 12.5)
        self.assertTrue((self.root / "previews/depth.png").read_bytes().startswith(b"\x89PNG"))
        self.assertEqual((self.root / "depth.bin").read_bytes(), struct.pack("<4f", *values))

    def test_packed_color_and_rejected_lengths(self):
        (self.root / "color.bin").write_bytes(struct.pack("<I", 1023 | (512 << 10) | (3 << 30)))
        desc = dict(kind="source_color", file="color.bin", dxgi_format=24,
                    width=1, height=1, row_bytes=4, byte_count=4)
        pixel = reader.read_artifact(self.root, desc)[0, 0]
        self.assertEqual(pixel[0], 1)
        self.assertAlmostEqual(pixel[1], 512 / 1023)
        self.assertEqual(pixel[2], 0)
        self.assertEqual(pixel[3], 1)
        with self.assertRaises(ValueError):
            reader.read_artifact(self.root, {**desc, "byte_count": 8})
        with self.assertRaises(ValueError):
            reader.read_artifact(self.root, {**desc, "file": "../outside.bin"})

    def test_optional_fixed_masks_alpha_unknown_transfer_and_own_rect(self):
        zeros = self.artifact("sl_no_warp_mask", [0, 0], fmt=61, artifact_id=11)
        ones = self.artifact("sl_ui_alpha", [255, 255], fmt=61, artifact_id=19)
        ui = self.artifact("sl_ui_color_alpha", [.5, .5, .5, .25, .5, .5, .5, np.nan],
                           fmt=2, channels=4, artifact_id=10)
        before = (self.root / ui["file"]).read_bytes()
        metadata = dict(color_space=2, render_parameters=dict(depth_rect=[0, 0, .1, .1]),
                        ui_resources=[dict(file_stem=ui["kind"], role="color_alpha"),
                                      dict(file_stem="sl_hudless_color", state="null")],
                        optional_captures=[dict(artifact_id=10, transfer_status="unknown",
                                                active_rect=[1, 0, 1, 1])])
        report = self.inspect([zeros, ones, ui], metadata)
        self.assertEqual(self.png("sl_no_warp_mask.png"), [[0, 0, 0]] * 2)
        self.assertEqual(self.png("sl_ui_alpha.png"), [[255, 255, 255]] * 2)
        self.assertEqual(self.png("sl_ui_color_alpha.png"), [[128, 128, 128]] * 2)
        self.assertEqual(self.png("sl_ui_color_alpha_alpha.png"), [[64, 64, 64], [255, 0, 255]])
        self.assertEqual(report["ui_resources"], metadata["ui_resources"])
        self.assertEqual(report["artifacts"][2]["capture"]["active_rect"], [1, 0, 1, 1])
        self.assertIn("Unknown transfer", report["artifacts"][2]["preview_mapping"])
        self.assertEqual((self.root / ui["file"]).read_bytes(), before)

        metadata["optional_captures"][0].update(transfer_status="declared", color_space=2)
        self.inspect([ui], metadata)
        self.assertEqual(self.png("sl_ui_color_alpha.png"), [[156, 156, 156]] * 2)

    def test_optional_multichannel_mask_and_missing_alpha_are_explicit(self):
        mask = self.artifact("ngx_transparency_mask", [0, .5, 1, np.inf], fmt=2, channels=4)
        layer = self.artifact("sl_ui_color_alpha", [.2, .8], fmt=16, channels=2)
        metadata = dict(ui_resources=[dict(file_stem=mask["kind"], role="mask"),
                                      dict(file_stem=layer["kind"], role="color_alpha")])
        self.inspect([mask, layer], metadata)
        self.assertEqual(self.png("ngx_transparency_mask.png"), [[0, 0, 0]])
        self.assertEqual(self.png("ngx_transparency_mask_G.png"), [[128, 128, 128]])
        self.assertEqual(self.png("ngx_transparency_mask_B.png"), [[255, 255, 255]])
        self.assertEqual(self.png("ngx_transparency_mask_A.png"), [[255, 0, 255]])
        self.assertFalse((self.root / "previews/sl_ui_color_alpha_alpha.png").exists())
        self.assertEqual(self.png("sl_ui_color_alpha_G.png"), [[204, 204, 204]])

    def test_sl_backbuffer_alpha_is_preserved_and_not_assumed_to_be_ui(self):
        payload = struct.pack("<4I", *(1023 | (alpha << 30) for alpha in range(4)))
        (self.root / "sl_backbuffer.bin").write_bytes(payload)
        image = dict(kind="sl_backbuffer", file="sl_backbuffer.bin", dxgi_format=24,
                     width=4, height=1, row_bytes=16, byte_count=16, artifact_id=32)
        semantic = "final_game_color_before_fg_candidate_not_verified_ui_mask"
        metadata = dict(color_space=2, ui_resources=[dict(file_stem="sl_backbuffer", role="color_alpha", semantic=semantic)],
                        optional_captures=[dict(artifact_id=32, transfer_status="unknown")])
        report = self.inspect([image], metadata)
        self.assertEqual(self.png("sl_backbuffer_alpha.png"), [[v] * 3 for v in (0, 85, 170, 255)])
        self.assertEqual(self.png("sl_backbuffer.png"), [[255, 0, 0]] * 4)
        self.assertEqual((self.root / "sl_backbuffer.bin").read_bytes(), payload)
        self.assertEqual(report["ui_resources"][0]["semantic"], semantic)
        self.assertIn("Unknown transfer", report["artifacts"][0]["preview_mapping"])

    def test_bad_optional_does_not_hide_primary_and_v2_capacity_is_bounded(self):
        depth = self.artifact("raw_depth", [.25, .75])
        broken = dict(kind="sl_no_warp_mask", file="absent.bin", dxgi_format=999,
                      width=1, height=1, row_bytes=1, byte_count=1)
        report = self.inspect([broken, depth])
        self.assertEqual(report["artifacts"][0]["kind"], "raw_depth")
        self.assertEqual(report["artifacts"][1]["status"], "unavailable")
        self.assertTrue((self.root / "previews/raw_depth.png").exists())
        optional = [dict(broken, kind=f"future_optional_{i}") for i in range(20)]
        self.assertEqual(len(self.inspect([depth] + optional)["artifacts"]), 21)
        with self.assertRaises(ValueError):
            self.inspect([depth] * (reader.MAX_ARTIFACTS + 1))

    def test_uint_mask_is_not_misread_as_unorm(self):
        mask = self.artifact("sl_no_warp_mask", [0, 1, 255], fmt=62)
        report = self.inspect([mask])
        self.assertEqual(report["artifacts"][0]["max"], 255)
        self.assertEqual(self.png("sl_no_warp_mask.png"), [[0, 0, 0], [255, 255, 255], [255, 255, 255]])
        self.assertEqual((self.root / mask["file"]).read_bytes(), bytes([0, 1, 255]))

    def test_source_alpha_is_unmapped_raw_channel_and_not_sbs_alpha(self):
        source = self.artifact("source_color", [4, 2, -1, 0, 4, 2, -1, .5,
                                                4, 2, -1, 1, 4, 2, -1, np.nan], fmt=2, channels=4)
        sbs = self.artifact("sbs", [1, 2, 3, 255] * 4, fmt=28, channels=4)
        original = (self.root / source["file"]).read_bytes()
        report = self.inspect([source, sbs], dict(color_space=2))
        self.assertEqual(self.png("source_alpha.png"),
                         [[0, 0, 0], [128, 128, 128], [255, 255, 255], [255, 0, 255]])
        alpha = report["artifacts"][0]["source_alpha_preview"]
        self.assertEqual(alpha["source_artifact"], "source_color")
        self.assertEqual(alpha["semantic"], "uninterpreted_source_alpha")
        self.assertEqual(alpha["nonfinite_count"], 1)
        self.assertNotIn("source_alpha_preview", report["artifacts"][1])
        self.assertFalse((self.root / "previews/sbs_alpha.png").exists())
        self.assertEqual((self.root / source["file"]).read_bytes(), original)

    def test_source_alpha_keeps_constants_and_requires_real_component(self):
        for alpha in (0, 255):
            source = self.artifact("source_color", [1, 2, 3, alpha] * 2, fmt=87, channels=4)
            self.inspect([source])
            self.assertEqual(self.png("source_alpha.png"), [[alpha] * 3] * 2)
        (self.root / "previews/source_alpha.png").unlink()
        source = self.artifact("source_color", [.25, .75], fmt=16, channels=2)
        sbs = self.artifact("sbs", [0, 0, 0, 255] * 2, fmt=28, channels=4)
        report = self.inspect([source, sbs])
        self.assertFalse((self.root / "previews/source_alpha.png").exists())
        self.assertNotIn("source_alpha_preview", report["artifacts"][0])

    def test_consumed_ui_is_required_input_and_has_separate_unmapped_alpha(self):
        source = self.artifact("source_color", [8, 16, 32, 255] * 2, fmt=28, channels=4)
        consumed = self.artifact("ui_source_color", [64, 128, 255, 0, 64, 128, 255, 128], fmt=28, channels=4, artifact_id=33)
        optional = self.artifact("sl_backbuffer", [0, 0, 0, 255] * 2, fmt=28, channels=4, artifact_id=32)
        metadata = dict(color_space=2, ui_source=dict(sequence=42), replay=dict(ui_alpha_source="ui_source_color"))
        report = self.inspect([source, consumed, optional], metadata)
        self.assertEqual(self.png("ui_source_alpha.png"), [[0, 0, 0], [128, 128, 128]])
        self.assertEqual(self.png("ui_source_color.png"), [[64, 128, 255]] * 2)
        self.assertEqual(self.png("source_alpha.png"), [[255, 255, 255]] * 2)
        self.assertEqual(report["ui_source"], dict(sequence=42))
        self.assertEqual(report["artifacts"][1]["ui_source_alpha_preview"]["semantic"], "consumed_ui_alpha")
        (self.root / consumed["file"]).unlink()
        with self.assertRaises(OSError):
            self.inspect([source, consumed, optional], metadata)
        with self.assertRaises(ValueError):
            self.inspect([source, optional], metadata)


if __name__ == "__main__":
    unittest.main()
