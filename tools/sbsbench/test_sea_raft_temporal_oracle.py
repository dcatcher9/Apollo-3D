import importlib.util
import os
import unittest

import numpy as np


MODULE_PATH = os.path.join(os.path.dirname(__file__), "sea_raft_temporal_oracle.py")
SPEC = importlib.util.spec_from_file_location("sea_raft_temporal_oracle", MODULE_PATH)
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


def textured(height=72, width=128):
    yy, xx = np.mgrid[:height, :width]
    value = (0.12 + 0.42 * ((xx // 9 + yy // 7) % 2)
             + 0.18 * np.sin(xx * 0.19) + 0.12 * np.cos(yy * 0.27))
    value = np.clip(value, 0.0, 1.0).astype(np.float32)
    return np.repeat(value[..., None], 3, axis=2)


def translate(image, dx, dy=0):
    result = np.zeros_like(image)
    height, width = image.shape[:2]
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    target_x0 = source_x0 + dx
    target_x1 = source_x1 + dx
    target_y0 = source_y0 + dy
    target_y1 = source_y1 + dy
    result[target_y0:target_y1, target_x0:target_x1] = image[
        source_y0:source_y1, source_x0:source_x1]
    return result


def flow_pair(height, width, dx, dy=0):
    forward = np.zeros((height, width, 2), np.float32)
    backward = np.zeros_like(forward)
    forward[..., 0] = dx
    forward[..., 1] = dy
    backward[..., 0] = -dx
    backward[..., 1] = -dy
    return forward, backward


def measure(previous_source, current_source, previous_eye, current_eye,
            forward, backward, **kwargs):
    return oracle.temporal_artifact_metrics(
        previous_source, current_source,
        previous_eye, current_eye, previous_eye, current_eye,
        forward, backward, min_support_pixels=64, **kwargs)


class SeaRaftTemporalMetricTests(unittest.TestCase):
    def test_clean_translation_has_near_zero_artifact(self):
        previous = textured()
        current = translate(previous, 5)
        forward, backward = flow_pair(*previous.shape[:2], 5)
        result = measure(previous, current, previous, current, forward, backward)
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["sea_flow_support_pct"], 80.0)
        self.assertLess(result["sea_flow_edge_ghost_p95"], 0.05)
        self.assertLess(result["sea_flow_flicker_p95"], 0.05)

    def test_held_and_ema_outputs_form_monotonic_ghost_ladder(self):
        previous = textured()
        current = translate(previous, 7)
        forward, backward = flow_pair(*previous.shape[:2], 7)
        clean = measure(previous, current, previous, current, forward, backward)
        ema = measure(previous, current, previous, 0.5 * current + 0.5 * previous,
                      forward, backward)
        held = measure(previous, current, previous, previous, forward, backward)
        self.assertEqual({clean["status"], ema["status"], held["status"]}, {"ok"})
        self.assertLess(clean["sea_flow_edge_ghost_p95"],
                        ema["sea_flow_edge_ghost_p95"])
        self.assertLess(ema["sea_flow_edge_ghost_p95"],
                        held["sea_flow_edge_ghost_p95"])
        self.assertGreater(held["sea_flow_gradient_ghost_p95"], 8.0)

    def test_output_flicker_is_detected_without_calling_it_edge_ghost(self):
        source = textured()
        forward, backward = flow_pair(*source.shape[:2], 0)
        brighter = np.clip(source + 0.12, 0.0, 1.0)
        result = measure(source, source, source, brighter, forward, backward)
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["sea_flow_flicker_p95"], 20.0)
        self.assertLess(result["sea_flow_gradient_ghost_p95"], 0.1)

    def test_forward_backward_inconsistent_occlusion_is_excluded(self):
        previous = textured()
        current = translate(previous, 4)
        forward, backward = flow_pair(*previous.shape[:2], 4)
        corrupted = current.copy()
        corrupted[20:52, 44:84] = 1.0 - corrupted[20:52, 44:84]
        # Mark the entire corruption plus a guard band as a disocclusion by sending its backward
        # match out of bounds. It must reduce support, not inflate the temporal artifact score.
        backward[16:56, 40:88, 0] = 1000.0
        result = measure(previous, current, previous, corrupted, forward, backward)
        self.assertEqual(result["status"], "ok")
        self.assertLess(result["sea_flow_support_pct"], 75.0)
        self.assertLess(result["sea_flow_edge_ghost_p95"], 0.1)

    def test_scene_cut_abstains_instead_of_reporting_a_ghost(self):
        previous = np.full((72, 128, 3), 0.08, np.float32)
        current = np.full_like(previous, 0.92)
        forward, backward = flow_pair(*previous.shape[:2], 0)
        result = measure(previous, current, previous, current, forward, backward)
        self.assertEqual(result["status"], "cut")
        self.assertEqual(result["reason"], "scene_cut")
        self.assertFalse(result["training_label_eligible"])
        self.assertNotIn("sea_flow_edge_ghost_p95", result)

    def test_static_output_jitter_uses_static_source_support(self):
        source = textured()
        jittered = translate(source, 2)
        forward, backward = flow_pair(*source.shape[:2], 0)
        result = measure(source, source, source, jittered, forward, backward)
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["sea_flow_static_support_pct"], 80.0)
        self.assertGreater(result["sea_static_jitter_p95"], 10.0)

    def test_source_exposure_change_is_subtracted(self):
        previous = textured()
        current = np.clip(previous + 0.08, 0.0, 1.0)
        forward, backward = flow_pair(*previous.shape[:2], 0)
        result = measure(previous, current, previous, current, forward, backward,
                         source_residual_limit=0.10)
        self.assertEqual(result["status"], "ok")
        self.assertLess(result["sea_flow_flicker_p95"], 0.05)

    def test_opposite_signed_change_cannot_cancel_source_change(self):
        previous = textured()
        current_source = np.clip(previous + 0.06, 0.0, 1.0)
        current_eye = np.clip(previous - 0.06, 0.0, 1.0)
        forward, backward = flow_pair(*previous.shape[:2], 0)
        result = measure(
            previous, current_source, previous, current_eye, forward, backward,
            source_residual_limit=0.08)
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["sea_flow_flicker_p95"], 20.0)

    def test_opposite_signed_spatial_changes_cannot_cancel_gradient_or_log(self):
        height, width = 72, 128
        previous = np.full((height, width, 3), 0.5, np.float32)
        yy, xx = np.mgrid[:height, :width]
        pattern = (0.07 * np.sin(xx * 0.72) * np.cos(yy * 0.43)).astype(np.float32)
        current_source = previous + pattern[..., None]
        current_eye = previous - pattern[..., None]
        zero_flow = np.zeros((height, width, 2), np.float32)
        maps = oracle._eye_residual_maps(
            previous, current_eye, previous, current_source,
            zero_flow, np.ones((height, width), bool))
        interior = maps["valid"][3:-3, 3:-3]
        self.assertGreater(float(np.percentile(maps["gradient"][3:-3, 3:-3][interior], 95)),
                           0.04)
        self.assertGreater(float(np.percentile(maps["log"][3:-3, 3:-3][interior], 95)),
                           0.04)

    def test_eye_motion_mismatch_excludes_cycle_inconsistent_outlier(self):
        height, width = 72, 128
        source_backward = np.zeros((height, width, 2), np.float32)
        eye_forward = np.zeros_like(source_backward)
        eye_backward = np.zeros_like(source_backward)
        eye_backward[20:52, 40:88, 0] = 15.0
        support = np.ones((height, width), bool)
        mismatch, support_pct = oracle._motion_mismatch(
            source_backward, eye_forward, eye_backward, support)
        self.assertLess(mismatch, 0.01)
        self.assertLess(support_pct, 80.0)

    def test_low_flow_support_abstains_without_zero_artifact_values(self):
        source = textured()
        forward, backward = flow_pair(*source.shape[:2], 0)
        backward[..., 0] = 1000.0
        result = measure(source, source, source, source, forward, backward)
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["reason"], "insufficient_reliable_flow_support")
        self.assertNotIn("sea_flow_flicker_p95", result)
        self.assertNotIn("sea_flow_edge_ghost_p95", result)


if __name__ == "__main__":
    unittest.main()
