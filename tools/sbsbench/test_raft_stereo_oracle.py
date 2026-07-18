import importlib.util
import os
import unittest

import numpy as np


MODULE_PATH = os.path.join(os.path.dirname(__file__), "raft_stereo_oracle.py")
SPEC = importlib.util.spec_from_file_location("raft_stereo_oracle", MODULE_PATH)
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


def textured_image(height=48, width=96):
    yy, xx = np.mgrid[:height, :width]
    gray = ((17 * xx + 11 * yy + (xx * yy) % 37) % 251).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def shifted_right(left, displacement, vertical=0):
    """Create R such that R[y+vertical, x+displacement] == L[y,x]."""
    height, width = left.shape[:2]
    right = np.zeros_like(left)
    dx = int(displacement)
    for y in range(height):
        target_y = y + int(vertical)
        if not 0 <= target_y < height:
            continue
        for x in range(width):
            target_x = x + dx
            if 0 <= target_x < width:
                right[target_y, target_x] = left[y, x]
    return right


class RaftStereoOracleTests(unittest.TestCase):
    def test_orientation_selection_abstains_on_coequal_hypotheses(self):
        selected = {
            "status": "ok",
            "raft_supported_texture_pct": 80.0,
            "raft_correspondence_residual_p50": 2.0,
        }
        coequal = {
            "status": "ok",
            "raft_supported_texture_pct": 70.0,
            "raft_correspondence_residual_p50": 2.2,
        }
        decisive = {
            "status": "ok",
            "raft_supported_texture_pct": 10.0,
            "raft_correspondence_residual_p50": 1.0,
        }
        self.assertTrue(oracle._orientation_is_ambiguous(selected, coequal))
        self.assertFalse(oracle._orientation_is_ambiguous(selected, decisive))

    def test_signed_exact_comparison_preserves_polarity(self):
        left = textured_image()
        right = shifted_right(left, -3)
        forward = np.full(left.shape[:2], -3.0, np.float32)
        reverse = np.full(left.shape[:2], 3.0, np.float32)
        exact = np.full(left.shape[:2], -3.0, np.float32)
        result = oracle.correspondence_metrics(left, right, forward, reverse, exact)
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["raft_signed_disparity_p50_px"], -3.0)
        self.assertLess(result["raft_exact_mae_px"], 1e-5)
        self.assertEqual(result["raft_exact_polarity_agreement_pct"], 100.0)

        inverted = oracle.correspondence_metrics(
            left, right, forward, reverse, -exact)
        self.assertGreater(inverted["raft_exact_mae_px"], 5.9)
        self.assertEqual(inverted["raft_exact_polarity_agreement_pct"], 0.0)

    def test_left_right_inconsistency_fails_closed(self):
        left = textured_image()
        right = shifted_right(left, -2)
        forward = np.full(left.shape[:2], -2.0, np.float32)
        wrong_reverse = np.full(left.shape[:2], -2.0, np.float32)
        result = oracle.correspondence_metrics(left, right, forward, wrong_reverse)
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["support_pixels"], 0)

    def test_textureless_pair_abstains(self):
        left = np.full((48, 96, 3), 128, np.uint8)
        forward = np.full((48, 96), -2.0, np.float32)
        reverse = -forward
        result = oracle.correspondence_metrics(left, left, forward, reverse)
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["raft_texture_support_pct"], 0.0)

    def test_vertical_search_detects_epipolar_offset(self):
        left = textured_image(64, 128)
        right = shifted_right(left, -3, vertical=2)
        forward = np.full(left.shape[:2], -3.0, np.float32)
        reverse = np.full(left.shape[:2], 3.0, np.float32)
        result = oracle.correspondence_metrics(left, right, forward, reverse)
        self.assertGreater(result["raft_vertical_abs_p50_px"], 1.5)
        self.assertGreater(result["raft_vertical_abs_p99_px"], 1.5)
        self.assertGreater(result["raft_vertical_nonzero_pct"], 90.0)
        self.assertLess(result["raft_vertical_aligned_residual_p95"],
                        result["raft_correspondence_residual_p95"])

    def test_exact_mapping_inversion_recovers_left_to_right_displacement(self):
        height, width = 12, 40
        x = np.arange(width, dtype=np.float32)
        # Left content is shifted right by two pixels and right content left by two: a source
        # coordinate appearing at x in the left eye appears at x-4 in the right eye.
        map_left = np.broadcast_to((x - 2.0) / width, (height, width)).copy()
        map_right = np.broadcast_to((x + 2.0) / width, (height, width)).copy()
        mapping = np.concatenate((map_left, map_right), axis=1)
        shape = {
            "height": height,
            "width": 2 * width,
            "eye_width": width,
            "content_scale_x": 1.0,
            "content_scale_y": 1.0,
        }
        disparity, valid = oracle.exact_left_to_right_disparity(mapping, shape)
        self.assertGreater(np.mean(valid), 0.75)
        self.assertAlmostEqual(float(np.median(disparity[valid])), -4.0, places=4)

    def test_folded_exact_mapping_abstains_instead_of_sorting_fold(self):
        height, width = 8, 32
        x = np.arange(width, dtype=np.float32) / width
        left = np.broadcast_to(x, (height, width)).copy()
        right = left.copy()
        right[:, 8:16] = right[:, 8:16][:, ::-1]
        mapping = np.concatenate((left, right), axis=1)
        shape = {
            "height": height,
            "width": 2 * width,
            "eye_width": width,
            "content_scale_x": 1.0,
            "content_scale_y": 1.0,
        }
        disparity, valid = oracle.exact_left_to_right_disparity(mapping, shape)
        self.assertFalse(valid.any())
        self.assertTrue(np.isnan(disparity).all())


if __name__ == "__main__":
    unittest.main()
