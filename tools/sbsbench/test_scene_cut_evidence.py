import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import measure_scene_cut_evidence as evidence


class SceneCutEvidenceTests(unittest.TestCase):
    def test_d3d_model_resize_identity_preserves_rgb(self):
        source = np.arange(5 * 7 * 3, dtype=np.float32).reshape(5, 7, 3)
        source /= float(source.max())
        resized = evidence.d3d_model_resize_rgb(source, 7, 5)
        np.testing.assert_allclose(resized, source, atol=1e-7)
        np.testing.assert_allclose(
            evidence.point_resize_max_rgb(source, 7, 5),
            np.max(source, axis=2),
            atol=0.0,
        )

    def test_d3d_model_resize_exactly_averages_four_by_four_footprint(self):
        source = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
        source = source[..., None].repeat(3, axis=2)
        resized = evidence.d3d_model_resize_rgb(source, 1, 1)
        np.testing.assert_allclose(
            resized[0, 0],
            np.full(3, np.mean(source[..., 0]), dtype=np.float32),
            atol=1e-6,
        )

    def test_d3d_model_resize_preserves_center_impulse_at_five_to_one(self):
        source = np.zeros((5, 5, 3), dtype=np.float32)
        source[2, 2, :] = 1.0
        resized = evidence.d3d_model_resize_rgb(source, 1, 1)
        np.testing.assert_allclose(
            resized[0, 0],
            np.full(3, 1.0 / 25.0, dtype=np.float32),
            atol=1e-7,
        )

    def test_d3d_model_resize_uses_fractional_source_cell_overlap(self):
        source = np.arange(3, dtype=np.float32).reshape(1, 3, 1)
        source = source.repeat(3, axis=2)
        resized = evidence.d3d_model_resize_rgb(source, 2, 1)
        expected = np.array([1.0 / 3.0, 5.0 / 3.0], dtype=np.float32)
        np.testing.assert_allclose(resized[0, :, 0], expected, atol=1e-6)
        np.testing.assert_allclose(resized[0, :, 1], expected, atol=1e-6)
        np.testing.assert_allclose(resized[0, :, 2], expected, atol=1e-6)

    def test_live_noninteger_footprints_preserve_a_uniform_field(self):
        source = np.ones((1, 3840, 3), dtype=np.float32)
        resized = evidence.d3d_model_resize_rgb(source, 770, 1)
        np.testing.assert_allclose(resized, 1.0, rtol=0.0, atol=2e-7)

    def test_d3d_model_resize_retains_bilinear_upscale_fallback(self):
        source = np.array([0.0, 1.0], dtype=np.float32).reshape(1, 2, 1)
        source = source.repeat(3, axis=2)
        resized = evidence.d3d_model_resize_rgb(source, 4, 1)
        expected = np.array([0.0, 0.25, 0.75, 1.0], dtype=np.float32)
        np.testing.assert_allclose(resized[0, :, 0], expected, atol=1e-7)

    def test_point_ordinal_uses_exact_target_center_source_texel(self):
        source = np.arange(4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3)
        ordinal = evidence.point_resize_max_rgb(source, 3, 2)
        expected = np.max(source[np.ix_([1, 3], [1, 3, 5])], axis=2)
        np.testing.assert_array_equal(ordinal, expected)

    def test_clipped_monotone_exposure_cannot_reverse_ordinal_pairs(self):
        width, height = 96, 64
        x = np.arange(width, dtype=np.float32)[None, :]
        y = np.arange(height, dtype=np.float32)[:, None]
        previous = np.clip(
            (120.0 + 60.0 * np.sin(np.pi * x / 3.0) +
             60.0 * np.sin(2.0 * np.pi * y / 5.0)) / 255.0,
            0.0,
            1.0,
        )
        current = np.clip(2.0 * previous, 0.0, 1.0)
        self.assertEqual(
            evidence.structural_change_fraction(current, previous),
            0.0,
        )
        self.assertGreater(
            evidence.raw_rgb_change_fraction(
                current[..., None].repeat(3, axis=2),
                previous[..., None].repeat(3, axis=2),
            ),
            0.25,
        )

    def test_all_pair_census_detects_horizontal_to_vertical_ramp(self):
        width, height = 32, 24
        x = np.arange(width, dtype=np.float32)[None, :]
        y = np.arange(height, dtype=np.float32)[:, None]
        # Keep adjacent contrast above the production 4% relative reliability floor across the
        # field; an additive ramp deliberately becomes unreliable toward its bright end.
        horizontal = np.broadcast_to(0.05 * np.power(1.08, x), (height, width))
        vertical = np.broadcast_to(0.05 * np.power(1.08, y), (height, width))
        self.assertGreater(
            evidence.structural_change_fraction(vertical, horizontal),
            0.75,
        )

    def test_structure_support_distinguishes_black_gap_from_preserved_exposure(self):
        width, height = 32, 24
        x = np.arange(width, dtype=np.float32)[None, :]
        y = np.arange(height, dtype=np.float32)[:, None]
        structured = np.mod(0.17 + 0.031 * (3 * x + 5 * y), 0.71)
        exposed = structured * 1.25 + 0.04
        black = np.zeros_like(structured)

        exposure_fractions = evidence.structural_evidence_fractions(
            exposed, structured)
        self.assertEqual(exposure_fractions[0], 0.0)
        self.assertGreater(min(exposure_fractions[1:]), 0.01)

        black_fractions = evidence.structural_evidence_fractions(
            black, structured)
        self.assertEqual(black_fractions[0], 0.0)
        self.assertEqual(black_fractions[1], 0.0)
        self.assertGreater(black_fractions[2], 0.01)
        self.assertEqual(black_fractions[3], 0.0)

    def test_measure_clip_emits_adjacent_source_depth_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clip = root / "clip"
            artifacts = root / "artifacts"
            clip.mkdir()
            artifacts.mkdir()
            for frame_id, value in enumerate((32, 64, 192), start=1):
                Image.new("RGB", (8, 6), (value, value, value)).save(
                    clip / f"frame_{frame_id:05d}.png")
                depth = np.full((6, 8), value * 257, dtype=np.uint16)
                Image.fromarray(depth).save(
                    artifacts / f"depth_{frame_id:05d}.png")
            (artifacts / "raw_shape.json").write_text(
                '{"width": 8, "height": 6}', encoding="utf-8")

            records = evidence.measure_clip(clip, artifacts)

        self.assertEqual([row["frame"] for row in records], [2, 3])
        self.assertEqual(records[0]["raw_rgb_change_fraction"], 0.0)
        self.assertEqual(records[0]["structural_change_fraction"], 0.0)
        self.assertEqual(records[0]["current_structural_support_fraction"], 0.0)
        self.assertEqual(records[0]["previous_structural_support_fraction"], 0.0)
        self.assertEqual(records[0]["common_structural_support_fraction"], 0.0)
        self.assertEqual(records[0]["depth_change_fraction"], 1.0)

    def test_measure_clip_retains_supported_appearance_and_depth_across_one_black(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clip = root / "clip"
            artifacts = root / "artifacts"
            clip.mkdir()
            artifacts.mkdir()
            width, height = 32, 24
            x = np.arange(width, dtype=np.uint8)[None, :]
            y = np.arange(height, dtype=np.uint8)[:, None]
            horizontal = np.broadcast_to(24 + x * 5, (height, width))
            vertical = np.broadcast_to(24 + y * 7, (height, width))
            frames = (horizontal, np.zeros((height, width), dtype=np.uint8), vertical)
            for frame_id, values in enumerate(frames, start=1):
                rgb = np.repeat(values[..., None], 3, axis=2)
                Image.fromarray(rgb).save(clip / f"frame_{frame_id:05d}.png")
                Image.fromarray(
                    np.full((height, width), frame_id * 1000, dtype=np.uint16)
                ).save(artifacts / f"depth_{frame_id:05d}.png")
            (artifacts / "raw_shape.json").write_text(
                f'{{"width": {width}, "height": {height}}}', encoding="utf-8")

            records = evidence.measure_clip(clip, artifacts)

        black, returned = records
        self.assertEqual(black["appearance_previous_frame"], 1)
        self.assertEqual(black["appearance_history_held"], 1)
        self.assertEqual(returned["previous_frame"], 1)
        self.assertEqual(returned["appearance_previous_frame"], 1)
        self.assertEqual(returned["appearance_history_held"], 0)
        self.assertGreater(returned["structural_change_fraction"], 0.5)
        self.assertGreater(returned["common_structural_support_fraction"], 0.5)

    def test_measure_clip_releases_history_after_second_structureless_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clip = root / "clip"
            artifacts = root / "artifacts"
            clip.mkdir()
            artifacts.mkdir()
            width, height = 32, 24
            x = np.arange(width, dtype=np.uint8)[None, :]
            structured = np.broadcast_to(24 + x * 5, (height, width))
            flat = np.zeros((height, width), dtype=np.uint8)
            for frame_id, values in enumerate(
                    (structured, flat, flat, structured), start=1):
                rgb = np.repeat(values[..., None], 3, axis=2)
                Image.fromarray(rgb).save(clip / f"frame_{frame_id:05d}.png")
                Image.fromarray(
                    np.full((height, width), frame_id * 1000, dtype=np.uint16)
                ).save(artifacts / f"depth_{frame_id:05d}.png")
            (artifacts / "raw_shape.json").write_text(
                f'{{"width": {width}, "height": {height}}}', encoding="utf-8")

            records = evidence.measure_clip(clip, artifacts)

        first_flat, persistent_flat, returned = records
        self.assertEqual(first_flat["appearance_previous_frame"], 1)
        self.assertEqual(first_flat["previous_frame"], 1)
        self.assertEqual(first_flat["appearance_history_held"], 1)
        self.assertEqual(persistent_flat["appearance_previous_frame"], 1)
        self.assertEqual(persistent_flat["previous_frame"], 1)
        self.assertEqual(persistent_flat["appearance_history_held"], 0)
        self.assertEqual(returned["appearance_previous_frame"], 3)
        self.assertEqual(returned["previous_frame"], 3)


if __name__ == "__main__":
    unittest.main()
