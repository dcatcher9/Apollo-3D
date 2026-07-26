import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import measure_scene_cut_evidence as evidence


class SceneCutEvidenceTests(unittest.TestCase):
    def test_d3d_resize_identity_preserves_rgb(self):
        source = np.arange(5 * 7 * 3, dtype=np.float32).reshape(5, 7, 3)
        source /= float(source.max())
        resized = evidence.d3d_bilinear_resize_rgb(source, 7, 5)
        np.testing.assert_allclose(resized, source, atol=1e-7)
        np.testing.assert_allclose(
            evidence.point_resize_max_rgb(source, 7, 5),
            np.max(source, axis=2),
            atol=0.0,
        )

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
        horizontal = np.broadcast_to(0.1 + 0.03 * x, (height, width))
        vertical = np.broadcast_to(0.1 + 0.03 * y, (height, width))
        self.assertGreater(
            evidence.structural_change_fraction(vertical, horizontal),
            0.75,
        )

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
        self.assertEqual(records[0]["depth_change_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
