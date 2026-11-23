import unittest

import numpy as np

try:
    from .direct_glyph_normalization import normalize_region, normalized_tiles, score_region
except ImportError:
    from direct_glyph_normalization import normalize_region, normalized_tiles, score_region


def echo_alpha(rgb):
    fill = rgb[:, :1].clip(1e-7, 1 - 1e-7)
    zero = np.full_like(fill, -100)
    return np.concatenate((np.log1p(-fill), np.log(fill), zero, zero), axis=1)


class ExpandedBoxNormalizationTests(unittest.TestCase):
    def test_compact_encoder_preserves_the_owned_rows_and_their_halo_across_tiles(self):
        source = np.arange(28 * 881).reshape(28, 881)
        original = normalized_tiles(source, 512, 144)
        compact = normalized_tiles(source, 512, 100)
        for (left, right, owned, tile), (new_left, new_right, new_owned, cropped) in zip(original, compact, strict=True):
            self.assertEqual((new_left, new_right), (left, right))
            self.assertEqual(new_owned[0], slice(36, 64))
            np.testing.assert_array_equal(cropped[new_owned], tile[owned])
            np.testing.assert_array_equal(cropped, tile[20:120])

    def test_every_native_pixel_including_last_row_and_long_tile_seams(self):
        for h, w, target in [(1, 1, 28), (23, 997, 49), (107, 3840, 28), (179, 973, 49)]:
            with self.subTest(shape=(h, w), target=target):
                src = np.full((h, w, 3), 255, np.uint8)
                q, tiles = score_region(src, target, echo_alpha)
                self.assertEqual(q.shape, (h, w))
                np.testing.assert_array_equal(q, 255)
                if w in (997, 3840):
                    self.assertGreater(tiles, 1)

    def test_rgb_and_continuous_training_labels_share_area_and_pixel_centers(self):
        src = np.array([[0, 0, 255], [0, 0, 255], [255, 255, 0]], np.uint8)
        expected = np.array([[0, 170], [170, 113]], np.uint8)
        np.testing.assert_array_equal(normalize_region(src, 2), expected)
        labels = normalize_region(src.astype(np.float32) / 255, 2)
        np.testing.assert_array_equal(np.rint(labels * 255), expected)

    def test_nonuniform_matte_is_identical_across_tile_widths(self):
        y, x = np.indices((73, 1921))
        patterned = (((x // 17 + y // 5) % 2) * 220 + 20).astype(np.uint8)
        src = np.repeat(patterned[:, :, None], 3, axis=2)
        large, _ = score_region(src, 49, echo_alpha, tile_width=2048)
        split, n = score_region(src, 49, echo_alpha, tile_width=384)
        self.assertGreater(n, 1)
        np.testing.assert_array_equal(large, split)

    def test_true_openings_survive_at_native_scale(self):
        src = np.zeros((28, 620, 3), np.uint8)
        src[4:24, 424:458] = 255
        src[8:20, 431:451] = 0
        q, _ = score_region(src, 28, echo_alpha)
        np.testing.assert_array_equal(q, src[:, :, 0])
