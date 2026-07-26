import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sbsbench  # noqa: E402


def legacy_local_vertical_offsets(eye, expected, valid, max_height=540,
                                  min_std=3.0 / 255.0):
    """Pre-optimization implementation retained as an exact test oracle."""
    eye = np.asarray(eye, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    original_height, original_width = eye.shape
    scale = min(1.0, float(max_height) / max(original_height, 1))
    height = max(24, int(round(original_height * scale)))
    width = max(32, int(round(original_width * scale)))
    if (height, width) != eye.shape:
        eye = sbsbench.resize_to(eye, width, height)
        expected = sbsbench.resize_to(expected, width, height)
        valid = sbsbench.resize_mask_conservative(valid, width, height)

    tile = max(16, min(64, int(round(height * 0.10))))
    max_shift = max(2, min(12, int(np.ceil(height * 0.03))))
    if height < tile + 2 * max_shift:
        tile = max(8, height - 2 * max_shift)
    stride = max(4, tile // 2)
    y_positions = [y for y in sbsbench._tile_positions(height, tile, stride)
                   if y >= max_shift and y + tile + max_shift <= height]
    x_positions = sbsbench._tile_positions(width, tile, stride)
    if not y_positions:
        return {}, 0

    offsets = {}
    attempted = len(y_positions) * len(x_positions)
    min_pixels = max(32, int(round(tile * tile * 0.90)))
    candidate_offsets = range(-max_shift, max_shift + 1)
    for y in y_positions:
        for x in x_positions:
            output_valid = valid[y:y + tile, x:x + tile]
            if int(output_valid.sum()) < min_pixels:
                continue
            output = eye[y:y + tile, x:x + tile]
            output_values = output[output_valid]
            output_centered = output_values - float(output_values.mean())
            output_norm = float(np.linalg.norm(output_centered))
            output_std = output_norm / np.sqrt(max(output_centered.size, 1))
            if output_std < min_std:
                continue

            scores = []
            texture = []
            for shift in candidate_offsets:
                ry = y - shift
                reference_valid = valid[ry:ry + tile, x:x + tile]
                joint = output_valid & reference_valid
                if int(joint.sum()) < min_pixels:
                    scores.append(float("-inf"))
                    texture.append(0.0)
                    continue
                a = output[joint]
                b = expected[ry:ry + tile, x:x + tile][joint]
                a = a - float(a.mean())
                b = b - float(b.mean())
                na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
                reference_std = nb / np.sqrt(max(b.size, 1))
                if min(na, nb) <= 1e-8 or reference_std < min_std:
                    scores.append(float("-inf"))
                    texture.append(0.0)
                    continue
                scores.append(float(np.dot(a, b) / (na * nb)))
                texture.append(reference_std)

            best_index = int(np.argmax(scores))
            best_score = scores[best_index]
            if not np.isfinite(best_score) or best_score < 0.55:
                continue
            separated = [score for index, score in enumerate(scores)
                         if abs(index - best_index) > 1 and np.isfinite(score)]
            if separated and best_score - max(separated) < 0.004:
                continue

            shift = float(best_index - max_shift)
            if 0 < best_index < len(scores) - 1:
                before, peak, after = scores[best_index - 1:best_index + 2]
                denominator = before - 2.0 * peak + after
                if (np.isfinite(before) and np.isfinite(after)
                        and abs(denominator) > 1e-8):
                    shift += float(0.5 * (before - after) / denominator)
            weight = min(0.10, np.sqrt(max(output_std * texture[best_index], 0.0)))
            offsets[(y, x)] = (shift, max(weight, 1e-6))
    return offsets, attempted


class VerticalOffsetOutputReuseTests(unittest.TestCase):
    @staticmethod
    def random_case(height, width, seed, edge_mask=False, sparse_mask=False):
        rng = np.random.default_rng(seed)
        expected = rng.random((height, width), dtype=np.float32)
        eye = np.roll(expected, 2, axis=0)
        eye = np.clip(
            eye * np.float32(0.83) + np.float32(0.07)
            + rng.normal(0.0, 0.002, eye.shape).astype(np.float32),
            0.0, 1.0)
        valid = np.ones((height, width), dtype=bool)
        if edge_mask:
            margin = max(2, height // 40)
            valid[:margin] = False
            valid[-margin:] = False
            valid[:, :2] = False
            valid[:, -3:] = False
        if sparse_mask:
            valid[rng.random(valid.shape) < 0.012] = False
            valid[height // 3:height // 3 + 3, width // 4:3 * width // 4] = False
        return eye, expected, valid

    def assert_matches_legacy(self, *args, **kwargs):
        expected = legacy_local_vertical_offsets(*args, **kwargs)
        actual = sbsbench._local_vertical_offsets(*args, **kwargs)
        self.assertEqual(actual, expected)

    def test_random_images_and_masks_are_bit_exact(self):
        cases = (
            self.random_case(128, 192, 7),
            self.random_case(128, 192, 11, edge_mask=True),
            self.random_case(144, 208, 23, edge_mask=True, sparse_mask=True),
        )
        for index, case in enumerate(cases):
            with self.subTest(case=index):
                self.assert_matches_legacy(*case)

    def test_resized_analysis_raster_is_bit_exact(self):
        case = self.random_case(
            600, 96, 31, edge_mask=True, sparse_mask=True)
        self.assert_matches_legacy(*case, max_height=540)

    def test_low_texture_and_small_edge_cases_are_bit_exact(self):
        constant = np.full((48, 64), 0.4, dtype=np.float32)
        self.assert_matches_legacy(
            constant, constant, np.ones(constant.shape, dtype=bool))
        tiny = self.random_case(12, 17, 41, edge_mask=True)
        self.assert_matches_legacy(*tiny)

    def test_full_valid_tiles_avoid_repeated_output_norms(self):
        case = self.random_case(128, 192, 53)
        real_norm = np.linalg.norm
        with mock.patch.object(
                np.linalg, "norm", side_effect=real_norm) as legacy_norm:
            legacy = legacy_local_vertical_offsets(*case)
        with mock.patch.object(
                np.linalg, "norm", side_effect=real_norm) as optimized_norm:
            optimized = sbsbench._local_vertical_offsets(*case)

        self.assertEqual(optimized, legacy)
        self.assertLess(optimized_norm.call_count, legacy_norm.call_count)


if __name__ == "__main__":
    unittest.main()
