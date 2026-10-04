import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
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

    def test_identity_tickets_are_reported_and_absent_before_s3(self):
        depth = self.artifact("raw_depth", [0.5])
        self.assertIsNone(self.inspect([depth])["identity_tickets"])
        ticket = dict(kind="sl_tag", boundary="at_tag", token=dict(space="token", value=7, valid=True, reason="none"),
                      present=dict(space="present", value=0, valid=False, reason="none"), proof="sl_declared",
                      refusal="none", stamped=True)
        layer = dict(kind="layer_copy", boundary="before_clear", proof="clear_precall", refusal="none", stamped=True,
                     stamp=dict(present_read=41, token_read=7, present_label=41, token_label=7))
        metadata = dict(
            ui_source=dict(candidates=[dict(source="sl_hudless", ticket=ticket), dict(source="none")],
                           identity_shadow=dict(today=dict(batch=True), ticket=dict(batch=False))),
            ui_layer_census=dict(candidates=[dict(kind="ui_layer_0", ticket=layer)]),
            optional_captures=[dict(name="Backbuffer", ticket=dict(kind="sl_tag", stamp=None))])
        report = self.inspect([self.artifact("raw_depth", [0.5])], metadata)["identity_tickets"]
        self.assertEqual([r["source"] for r in report["candidates"]], ["sl_hudless"])
        self.assertEqual(report["candidates"][0]["token"]["value"], 7)
        self.assertIsNone(report["candidates"][0]["present_read"])
        self.assertEqual((report["census"][0]["present_label"], report["census"][0]["token_read"]), (41, 7))
        self.assertEqual(report["optional"][0]["source"], "Backbuffer")
        self.assertEqual(report["identity_shadow"]["today"]["batch"], True)

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

    def test_consumed_red_mask_uses_declared_channel_without_stretch(self):
        metadata = dict(replay=dict(ui_alpha_source="ui_source_color", ui_constant_binding=dict(mask_channel="red")))
        for fmt, values, channels in ((61, [255, 255], 1), (54, [0, .5], 1),
                                      (41, [.25, .75], 1), (28, [0, 0, 0, 255, 255, 0, 0, 0], 4)):
            with self.subTest(fmt=fmt):
                desc = self.artifact("ui_source_color", values, fmt=fmt, channels=channels, artifact_id=33)
                report = self.inspect([desc], metadata)
                mask = reader.read_artifact(self.root, desc)[:, :, 0].ravel()
                expected = [[round(float(v) * 255)] * 3 for v in mask]
                self.assertEqual(self.png("ui_source_mask.png"), expected)
                self.assertEqual(report["artifacts"][0]["ui_source_mask_preview"]["channel"], "R")
                self.assertFalse((self.root / "previews/ui_source_alpha.png").exists())

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

    def test_production_nested_inventory_controls_preview_and_discovery(self):
        image = self.artifact("sl_ui_color_alpha", [.2, .4, .6, .5], fmt=2, channels=4, artifact_id=10)
        row = dict(artifact_id=10, file_stem="sl_ui_color_alpha", provider="streamline",
                   role="color_alpha", semantic="ui_color_and_alpha", state="non_null")
        metadata = dict(ui_resources=[dict(row, role="mask", state="null")],
                        latest_observations=dict(ui_resources=[row], status="observed-window"),
                        optional_captures=[dict(artifact_id=10, file_stem="sl_ui_color_alpha", status="captured")])
        report = self.inspect([image], metadata)
        self.assertEqual(report["ui_resources"], [row])
        self.assertEqual(report["artifacts"][0]["role"], "color_alpha")
        self.assertEqual(self.png("sl_ui_color_alpha_alpha.png"), [[128, 128, 128]])
        self.assertIn("ui_discovery", report)
        # An intentionally empty canonical catalog must not revive stale flat data.
        metadata["latest_observations"]["ui_resources"] = []
        self.assertEqual(self.inspect([image], metadata)["ui_resources"], [])

    def test_review_fingerprint_uses_native_bytes_before_swizzle_and_decode(self):
        for fmt, values in ((87, [1, 2, 3, 128]), (24, [1023 | (2 << 30)])):
            with self.subTest(fmt=fmt):
                image = self.artifact("source_color", values, fmt=fmt, channels=4 if fmt == 87 else 1)
                native_bytes = (self.root / image["file"]).read_bytes()
                decoded, digest = reader.read_artifact(self.root, image, include_digest=True)
                self.assertEqual(digest, hashlib.sha256(native_bytes).hexdigest())
                self.assertEqual(decoded.shape, (1, 1, 4))
                self.assertFalse(np.array_equal(decoded.ravel(), values))
                (self.root / "manifest.json").write_text(json.dumps(dict(
                    schema="sunshine.game3d.dump.v1", status="complete", artifacts=[image])))
                report = reader.inspect(self.root)
                self.assertIn("ui_discovery", report)
                self.assertFalse((self.root / "previews").exists())

    def image(self, kind, values, fmt):
        array = np.asarray(values, dtype=reader.FORMATS[fmt][0])
        (self.root / (kind + ".bin")).write_bytes(array.tobytes())
        return dict(kind=kind, file=kind + ".bin", dxgi_format=fmt, width=array.shape[1], height=array.shape[0],
                    row_bytes=array.nbytes // array.shape[0], byte_count=array.nbytes)

    @staticmethod
    def pin_metadata(mode=5, fraction=0.5, ready=1, strength=100.0, limit_uv=0.04, rect=(0.0, 0.0, 1.0, 1.0)):
        # b0: strength, depth_view, depth_ready, camera_ready, basis, depth_scale, blend, limit_uv, projection
        # (A, 1/B), raw depth range, convergence (reference ZPD, zero inverse distance), jitter, depth rect.
        parameters = struct.pack("<fiIIifff2f2f2f2f4f", strength, 0, ready, ready, 0, 1.0, 1.0, limit_uv,
                                 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, *rect)
        word2, = struct.unpack("<I", struct.pack("<f", fraction))
        return dict(color_space=3, replay=dict(
            parameter_hex=parameters.hex(), ui_alpha_source="ui_source_color",
            ui_constant_binding=dict(uint32=[1, mode, word2, 1], mask_channel="red")),
            ui_layer_census=dict(candidates=[dict(kind="ui_layer_candidate_0", active=True)]))

    def pin_dump(self):
        """A 64x8 HDR10 frame: a light glyph pinned on row 4, and a scene step the pin did not cause on row 1."""
        width, height = 64, 8
        plane = np.float32(0.5) * np.float32(0.04)
        final = np.zeros((height, width), np.float32)
        final[4, :10] = plane
        final[1, 20:30] = 0.01  # 0.64 px; the unpinned field is zero everywhere.
        mask = np.zeros((height, width), np.float32)
        mask[4, :10] = 1
        layer = np.zeros((height, width, 4), np.uint8)
        layer[4, :10] = 255
        layer[4, 40] = (2, 2, 2, 255)  # Dark UI is neither light nor a glyph body.
        codes = np.repeat(np.arange(width, dtype=np.uint32)[None, :] * 10, height, axis=0)
        color = codes | codes << 10 | codes << 20 | np.uint32(3) << 30
        artifacts = [self.image("vertical_field", np.zeros((height, width)), 41), self.image("final_field", final, 41),
                     self.image("ui_source_color", mask, 41), self.image("ui_layer_candidate_0", layer, 28),
                     self.image("source_color", color, 24)]
        (self.root / "manifest.json").write_text(json.dumps(dict(
            schema="sunshine.game3d.dump.v1", status="complete", artifacts=artifacts,
            producer_metadata=self.pin_metadata())))
        return width, plane

    def test_unpinned_field_is_the_horizontal_limiter(self):
        vertical = np.zeros((1, 8))
        vertical[0, 3] = 0.2
        # 0.5/W is exactly 2**26 in Q30 at W = 8.
        expected = [0.0125, 0.075, 0.1375, 0.2, 0.1375, 0.075, 0.0125, 0]
        np.testing.assert_allclose(reader.unpinned_field(vertical)[0], expected, rtol=0, atol=1e-15)

    def test_ui_plane_follows_the_shader_modes(self):
        bound = np.float32(0.04) * np.float32(100) * np.float32(0.01) * np.float32(1)
        self.assertEqual(reader.ui_plane_uv(self.pin_metadata(5, 0.5)), float(np.float32(0.5) * bound))
        self.assertEqual(reader.ui_plane_uv(self.pin_metadata(4)), float(np.float32(0.25) * bound))
        self.assertEqual(reader.ui_plane_uv(self.pin_metadata(3)), float(bound))
        for metadata in (self.pin_metadata(0), self.pin_metadata(5, 0.8), self.pin_metadata(5, ready=0),
                         self.pin_metadata(5, strength=0.0)):
            self.assertEqual(reader.ui_plane_uv(metadata), 0.0)
        # SunshineCameraActive rejects a display limit outside (0, 0.04] and a depth rect outside the
        # allocation, which pins UI to the screen plane rather than to a clamped limit.
        for metadata in (self.pin_metadata(3, limit_uv=0.05), self.pin_metadata(3, limit_uv=0.0),
                         self.pin_metadata(3, limit_uv=float("nan")), self.pin_metadata(3, rect=(0.5, 0.0, 0.6, 1.0)),
                         self.pin_metadata(3, rect=(0.0, 0.0, 0.0, 1.0))):
            self.assertEqual(reader.ui_plane_uv(metadata), 0.0)
        smaller = np.float32(0.02) * np.float32(100) * np.float32(0.01) * np.float32(1)
        self.assertEqual(reader.ui_plane_uv(self.pin_metadata(3, limit_uv=0.02)), float(smaller))
        with self.assertRaises(ValueError):
            reader.ui_plane_uv(self.pin_metadata(2))

    def test_pin_metrics_count_new_steps_tear_flattening_and_torn_glyphs(self):
        width, plane = self.pin_dump()
        metrics = reader.pin_metrics(self.root, bands=[(0, 2), (3, 5)])
        self.assertEqual(metrics["layer"], "ui_layer_candidate_0")
        self.assertAlmostEqual(metrics["plane_px"], float(plane) * width)
        # Each edge of the glyph row and of the scene step is a new step in 10 columns.
        self.assertEqual(metrics["bands"], [{"rows": [0, 2], "max_new_step_columns": 10, "row": 0},
                                            {"rows": [3, 5], "max_new_step_columns": 10, "row": 3}])
        # Only the scene step tears: 20 pairs of 0.64 px times luma texture 10/1023.
        self.assertAlmostEqual(metrics["scene_tear"], 20 * 0.64 * 10 / 1023, places=5)
        # Masked UI pixels are not flattened scene: 10 scene pixels moved 0.64 px.
        self.assertAlmostEqual(metrics["flattening_px_per_pixel"], 10 * 0.64 / (width * 8), places=6)
        # Moved glyph bodies land off the plane; a move out of the frame tears nothing.
        self.assertEqual(metrics["torn_glyph_pixels"],
                         {"up_1": 10, "up_2": 10, "up_4": 10, "down_4": 0, "right_4": 4, "up_8": 0, "down_8": 0,
                          "right_8": 8, "up_8_right_8": 0, "right_16": 10})

    def test_pin_metrics_measure_a_replayed_field(self):
        self.pin_dump()
        unpinned = self.root / "replayed_final_field.bin"
        unpinned.write_bytes(np.zeros((8, 64), "<f4").tobytes())
        metrics = reader.pin_metrics(self.root, field=unpinned)
        self.assertEqual(metrics["bands"][0]["max_new_step_columns"], 0)
        self.assertEqual(metrics["scene_tear"], 0)
        self.assertEqual(metrics["torn_glyph_pixels"]["up_1"], 10)
        unpinned.write_bytes(np.zeros((8, 63), "<f4").tobytes())
        with self.assertRaises(ValueError):
            reader.pin_metrics(self.root, field=unpinned)

    def test_pin_metrics_exclude_half_pixel_float32_rounding(self):
        width, _ = self.pin_dump()
        # The pin collar's own bound, 0.5 px from the plane, rounded to float32.
        final = np.zeros((8, width), np.float32)
        final[5, :] = np.nextafter(np.float32(0.5 / width), np.float32(1))
        self.assertGreater(float(final[5, 0]) * width, 0.5)
        field = self.root / "collar.bin"
        field.write_bytes(final.tobytes())
        self.assertEqual(reader.pin_metrics(self.root, field=field)["bands"][0]["max_new_step_columns"], 0)

    def late_layer_dump(self, mask, flags=7):
        """pin_dump whose consumed mask came from the one-frame-late UI layer (flag 0x4), with a pin on rows 2-6."""
        width, plane = self.pin_dump()
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        (self.root / "ui_source_color.bin").write_bytes(np.asarray(mask, np.float32).tobytes())
        final = np.fromfile(self.root / "final_field.bin", np.float32).reshape(8, width)
        final[2:7, :10] = plane
        (self.root / "final_field.bin").write_bytes(final.tobytes())
        replay = manifest["producer_metadata"]["replay"]
        replay["ui_detection"] = dict(flags=flags)
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        return width, plane

    def test_pin_metrics_weigh_a_late_layer_by_its_consumed_alpha(self):
        texture = 10 / 1023
        width, plane = self.pin_dump()
        layer = np.fromfile(self.root / "ui_layer_candidate_0.bin", np.uint8).reshape(8, width, 4)
        raw = layer[:, :, 3] / np.float32(255)
        # The layer's raw alpha leaves pinned rows 2, 3, 5 and 6 as scene: their outer edges tear it, and their
        # 4 x 10 pinned pixels are flattened scene.
        self.late_layer_dump(raw)
        metrics = reader.pin_metrics(self.root)
        self.assertEqual(metrics["alpha"], "ui_source_color")
        margin_px = float(plane) * width
        self.assertAlmostEqual(metrics["scene_tear"], (20 * 0.64 + 20 * margin_px) * texture, places=5)
        self.assertAlmostEqual(metrics["flattening_px_per_pixel"], (10 * 0.64 + 40 * margin_px) / (width * 8),
                               places=6)
        # The reader never reconstructs a layer: a wider consumed mask is the alpha, and its rows count as UI.
        wide = np.zeros((8, width), np.float32)
        wide[2:7, :12] = 1
        self.late_layer_dump(wide)
        metrics = reader.pin_metrics(self.root)
        self.assertAlmostEqual(metrics["scene_tear"], 20 * 0.64 * texture, places=5)
        self.assertAlmostEqual(metrics["flattening_px_per_pixel"], 10 * 0.64 / (width * 8), places=6)

    def test_cli_reports_pin_metrics(self):
        self.pin_dump()
        command = [sys.executable, str(Path(reader.__file__)), str(self.root), "--pin-metrics", "--pin-band", "3:5"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        metrics = json.loads(result.stdout)["pin_metrics"]
        self.assertEqual(metrics["bands"], [{"rows": [3, 5], "max_new_step_columns": 10, "row": 3}])

    # Hidden-scene statistic D: a 512x288 frame (2x2 pixels per cell) whose depth has two discs nearer than the
    # background by 100 pixels of parallax per 2160 rows.
    SCENE_SIZE = (288, 512)

    @staticmethod
    def scene_parameters(camera_ready=1):
        return struct.pack("<fiIIifff8f4f", 100.0, 0, 1, camera_ready, 1, 100000.0, 1.0, .04,
                           0.0, 1.0, 0.0, 1.0, .05, .03, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)

    def scene_inputs(self):
        height, width = self.SCENE_SIZE
        y, x = np.mgrid[0:height, 0:width]
        discs = (((x - .3 * width) ** 2 + (y - .5 * height) ** 2 <= (.25 * height) ** 2) |
                 ((x - .72 * width) ** 2 + (y - .45 * height) ** 2 <= (.18 * height) ** 2))
        return np.where(discs, np.float32(.0302), np.float32(.03)), discs

    def scene(self, *colors, transfer=1, depth=None, parameters=None):
        height, width = self.SCENE_SIZE
        raw, _ = self.scene_inputs()
        parameters = parameters or self.scene_parameters()
        parallax = reader.scene_parallax(raw if depth is None else depth, parameters, width, height)
        active = reader.camera_active(parameters)
        return reader.scene_evidence(parallax, [reader.scene_code_luma(color, transfer) for color in colors], active)

    def test_scene_evidence_reads_whether_color_shows_the_depth_edges(self):
        _, discs = self.scene_inputs()
        silhouette = np.repeat(np.where(discs, .75, .25)[..., None], 3, axis=-1).astype(np.float32)
        noise = np.random.default_rng(7).random((*self.SCENE_SIZE, 3), dtype=np.float32)
        black = np.zeros_like(silhouette)
        visible, hidden, unrelated, faint = self.scene(silhouette, black, noise, silhouette * .01)
        self.assertGreaterEqual(visible["n"], reader.SCENE_MIN_EDGES)
        self.assertTrue(all(result["valid"] and result["n"] == visible["n"] for result in (hidden, unrelated, faint)))
        self.assertEqual(visible["verdict"], "visible")
        self.assertGreater(visible["d"], .9)
        # Black has no activity anywhere: every comparison ties, so D is exactly 0 and nothing is decided.
        self.assertEqual((hidden["verdict"], hidden["d"], hidden["decided"]), ("hidden", 0.0, 0))
        self.assertLess(abs(unrelated["d"]), .1)
        self.assertGreaterEqual(unrelated["decided"], reader.SCENE_MIN_EDGES)
        self.assertEqual(unrelated["verdict"], "hidden")
        # A dimmed picture keeps its verdict.
        self.assertEqual(faint["verdict"], "visible")
        self.assertGreater(faint["d"], .9)
        for transfer in (2, 3):
            with self.subTest(transfer=transfer):
                bright, dark = self.scene(silhouette, black, transfer=transfer)
                self.assertEqual((bright["verdict"], dark["verdict"], dark["d"]), ("visible", "hidden", 0.0))

    def test_scene_evidence_needs_depth_edges_and_a_ready_camera(self):
        _, discs = self.scene_inputs()
        silhouette = np.repeat(np.where(discs, .75, .25)[..., None], 3, axis=-1).astype(np.float32)
        flat, = self.scene(silhouette, depth=np.full(self.SCENE_SIZE, .03, np.float32))
        self.assertEqual((flat["n"], flat["valid"], flat["verdict"]), (0, False, "none"))
        unready, = self.scene(silhouette, parameters=self.scene_parameters(camera_ready=0))
        self.assertEqual((unready["n"], unready["valid"], unready["verdict"]), (0, False, "none"))
        # A frame smaller than the grid leaves cells without a pixel; the shader measures none beyond 3840.
        small, = reader.scene_evidence(np.zeros((142, 250), np.float32), [np.zeros((142, 250), np.float32)])
        self.assertEqual((small["n"], small["valid"]), (0, False))
        steps = np.tile((np.arange(3842) // 15 % 2).astype(np.float32) * 100, (144, 1))
        wide, = reader.scene_evidence(steps, [np.zeros_like(steps)])
        self.assertEqual((wide["n"], wide["valid"]), (0, False))

    def test_scene_cells_map_pixels_by_floor_and_round_means_half_away_from_zero(self):
        # 300 pixels over 256 columns: cell c holds pixels floor(c * 300 / 256) up to the next cell's first pixel.
        values = np.tile(np.arange(300, dtype=np.float32), (144, 1)) / 4096
        means = reader.scene_cell_means(values, reader.SCENE_PARALLAX_SCALE)
        columns = np.arange(300) * 256 // 300
        expected = [round(np.arange(300)[columns == c].mean() + 1e-9) for c in range(256)]
        self.assertEqual(means[0].tolist(), expected)
        # Two pixels per cell of 1 and 2 (or -1 and -2) units: a mean of 1.5 rounds to 2 (-2).
        pairs = np.tile(np.array([1, 2], np.float32), (144, 256)) / 4096
        self.assertEqual(reader.scene_cell_means(pairs, 4096)[0, 0], 2)
        self.assertEqual(reader.scene_cell_means(-pairs, 4096)[0, 0], -2)

    def test_cli_reports_scene_evidence(self):
        height, width = self.SCENE_SIZE
        raw, discs = self.scene_inputs()
        gray = np.where(discs, 192, 64)[..., None]
        color = (gray * np.array([1, 1, 1, 0]) + np.array([0, 0, 0, 255])).astype(np.uint8)
        artifacts = []
        for kind, payload, fmt in (("raw_depth", raw.astype("<f4").tobytes(), 41),
                                   ("source_color", color.tobytes(), 28)):
            (self.root / (kind + ".bin")).write_bytes(payload)
            artifacts.append(dict(kind=kind, file=kind + ".bin", dxgi_format=fmt, width=width, height=height,
                                  row_bytes=len(payload) // height, byte_count=len(payload)))
        metadata = dict(color_space=1, replay=dict(parameter_hex=self.scene_parameters().hex()))
        (self.root / "manifest.json").write_text(json.dumps(dict(
            schema="sunshine.game3d.dump.v1", status="complete", artifacts=artifacts, producer_metadata=metadata)))
        result = subprocess.run([sys.executable, str(Path(reader.__file__)), str(self.root), "--scene-evidence"],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        presented = json.loads(result.stdout)["scene_evidence"]["presented"]
        self.assertEqual(presented["verdict"], "visible")
        self.assertGreaterEqual(presented["n"], reader.SCENE_MIN_EDGES)

    def test_scene_evidence_reads_the_pre_ui_image_beside_the_presented_frame(self):
        # A settings menu presented over a hidden scene: the presented frame is a flat panel, while the cleared UI
        # layer still holds the pre-UI scene at alpha 0 (Stellar Blade SDR) or a HUD-less image shows it.
        height, width = self.SCENE_SIZE
        raw, discs = self.scene_inputs()
        gray = np.where(discs, 192, 64)[..., None]
        scene = (gray * np.array([1, 1, 1, 0])).astype(np.uint8)
        menu = np.full((height, width, 4), (90, 90, 90, 255), np.uint8)

        def write(kind, payload, fmt):
            (self.root / (kind + ".bin")).write_bytes(payload)
            return dict(kind=kind, file=kind + ".bin", dxgi_format=fmt, width=width, height=height,
                        row_bytes=len(payload) // height, byte_count=len(payload))

        census = dict(candidates=[dict(kind="ui_layer_candidate_0", active=True, dxgi_format=87)])
        metadata = dict(color_space=1, replay=dict(parameter_hex=self.scene_parameters().hex()), ui_layer_census=census)
        base = [write("raw_depth", raw.astype("<f4").tobytes(), 41), write("source_color", menu.tobytes(), 28),
                write("ui_layer_candidate_0", scene[:, :, [2, 1, 0, 3]].tobytes(), 87)]
        for artifacts, image, artifact in ((base, "layer", "ui_layer_candidate_0"),
                                           (base + [write("sl_hudless_color", scene.tobytes(), 28)], "hudless",
                                            "sl_hudless_color")):
            with self.subTest(image=image):
                (self.root / "manifest.json").write_text(json.dumps(dict(
                    schema="sunshine.game3d.dump.v1", status="complete", artifacts=artifacts,
                    producer_metadata=metadata)))
                evidence = reader.dump_scene_evidence(self.root)
                self.assertEqual(evidence["presented"]["verdict"], "hidden")
                self.assertEqual((evidence["pre_ui"]["image"], evidence["pre_ui"]["artifact"]), (image, artifact))
                self.assertEqual(evidence["pre_ui"]["verdict"], "visible")
                self.assertGreater(evidence["pre_ui"]["d"], .9)
                # The HUD-less key stays for compatibility, only when the HUD-less image is the pre-UI image.
                self.assertEqual("hudless" in evidence, image == "hudless")
                if image == "hudless":
                    self.assertEqual(evidence["hudless"]["d"], evidence["pre_ui"]["d"])
                    self.assertNotIn("image", evidence["hudless"])
        # Without a HUD-less image or an active census layer there is no pre-UI image.
        (self.root / "manifest.json").write_text(json.dumps(dict(
            schema="sunshine.game3d.dump.v1", status="complete", artifacts=base[:2],
            producer_metadata=dict(metadata, ui_layer_census=dict(candidates=[])))))
        self.assertEqual(set(reader.dump_scene_evidence(self.root)), {"presented"})

    def test_cli_creates_review_and_report_without_overwriting_evidence(self):
        source = self.artifact("source_color", [1, 2, 3, 0, 4, 5, 6, 255], fmt=28, channels=4)
        (self.root / "manifest.json").write_text(json.dumps(dict(
            schema="sunshine.game3d.dump.v1", status="complete", artifacts=[source])))
        review_path = self.root / "review.json"
        report_path = self.root / "ui-report.md"
        command = [sys.executable, str(Path(reader.__file__)), str(self.root),
                   "--write-ui-review", str(review_path), "--ui-report", str(report_path)]
        first = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(first.returncode, 0, first.stderr)
        report = json.loads(first.stdout)
        template = json.loads(review_path.read_text(encoding="utf-8"))
        self.assertEqual(template["manifest_sha256"], report["ui_discovery"]["manifest_sha256"])
        self.assertIn("source\\_color", report_path.read_text(encoding="utf-8"))
        original = {path: path.read_bytes() for path in
                    (review_path, report_path, self.root / "manifest.json", self.root / source["file"])}
        second = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(second.returncode, 0)
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
