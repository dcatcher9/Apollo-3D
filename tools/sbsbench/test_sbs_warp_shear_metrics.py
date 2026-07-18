"""Focused qualification tests for exact-map cross-row shear detection."""

import json
import os
import sys
import unittest

import numpy as np
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import sbs_warp_shear_metrics  # noqa: E402


def mapping_shape(width, height, *, source_width=None, source_height=None,
                  scale_x=1.0, scale_y=1.0):
    return {
        "width": 2 * width,
        "height": height,
        "eye_width": width,
        "eye_height": height,
        "source_width": source_width or width,
        "source_height": source_height or height,
        "content_scale_x": scale_x,
        "content_scale_y": scale_y,
    }


def identity_map(shape):
    width = shape["eye_width"]
    output_u = (np.arange(width, dtype=np.float32) + 0.5) / width
    lo_x = 0.5 * (1.0 - shape["content_scale_x"])
    source_u = (output_u - lo_x) / shape["content_scale_x"]
    return np.broadcast_to(source_u, (shape["eye_height"], width)).copy()


def vertical_feature_source(width, height):
    source = np.full((height, width), 0.2, dtype=np.float32)
    source[:, width // 2:] = 0.8
    return source


def add_row_ramp(mapping, shape, amplitude_ref_px, *, y0=0.30, y1=0.38,
                 x0=0.15, x1=0.85):
    """Add a normalized row-shear ramp that remains constant below the ramp."""
    out = mapping.copy()
    height, width = out.shape
    first = int(round(y0 * height))
    last = max(first + 1, int(round(y1 * height)))
    left = int(round(x0 * width))
    right = int(round(x1 * width))
    # Convert the 854x480 reference derivative back to this image geometry.
    per_row_px = (amplitude_ref_px * (width / height) /
                  (854.0 / 480.0))
    for y in range(first, height):
        row_count = min(y - first + 1, last - first)
        out[y, left:right] += row_count * per_row_px / (
            shape["content_scale_x"] * width)
    return out


def add_row_step(mapping, shape, shift_ref_px, row_fraction=0.5):
    out = mapping.copy()
    height, width = out.shape
    row = int(round(row_fraction * height))
    shift_px = shift_ref_px * (width / height) / (854.0 / 480.0)
    out[row:] += shift_px / (shape["content_scale_x"] * width)
    return out


def measure(mapping, shape, source, **kwargs):
    return sbs_warp_shear_metrics.measure_cross_row_shear(
        mapping, shape, source=source, **kwargs)


class WarpShearMetricTests(unittest.TestCase):
    def test_localized_row_shift_ladder_is_monotonic(self):
        shape = mapping_shape(854, 480)
        identity = identity_map(shape)
        source = vertical_feature_source(854, 480)
        clean = measure(identity, shape, source)
        mild = measure(add_row_ramp(identity, shape, 0.7), shape, source)
        strong = measure(add_row_ramp(identity, shape, 1.4), shape, source)

        self.assertEqual(clean["warp_cross_row_shear_severity_pct"], 0.0)
        self.assertGreater(mild["warp_cross_row_shear_severity_pct"], 1.0)
        self.assertGreater(strong["warp_cross_row_shear_severity_pct"],
                           mild["warp_cross_row_shear_severity_pct"] * 1.7)
        self.assertGreater(strong["warp_cross_row_shear_bad_area_pct"], 0.5)
        self.assertGreater(strong["warp_cross_row_shear_largest_run_pct"], 50.0)

    def test_source_supported_horizontal_step_is_benign(self):
        width, height = 854, 480
        shape = mapping_shape(width, height)
        shifted = add_row_step(identity_map(shape), shape, 2.0)
        unsupported_source = vertical_feature_source(width, height)
        supported_source = unsupported_source.copy()
        supported_source[height // 2:] += 0.15

        unsupported = measure(shifted, shape, unsupported_source)
        supported = measure(shifted, shape, supported_source)
        self.assertGreater(unsupported["warp_cross_row_shear_severity_pct"], 50.0)
        self.assertEqual(supported["warp_cross_row_shear_severity_pct"], 0.0)
        self.assertEqual(supported["warp_cross_row_shear_bad_area_pct"], 0.0)

    def test_480p_and_1080p_are_resolution_invariant(self):
        values = []
        for width, height in ((854, 480), (1920, 1080)):
            shape = mapping_shape(width, height)
            source = vertical_feature_source(width, height)
            mapping = add_row_ramp(identity_map(shape), shape, 1.1)
            values.append(measure(mapping, shape, source))

        low, high = values
        for key, delta in (("warp_cross_row_shear_severity_pct", 0.8),
                           ("warp_cross_row_shear_bad_area_pct", 0.15),
                           ("warp_cross_row_shear_largest_run_pct", 0.15)):
            self.assertAlmostEqual(low[key], high[key], delta=delta, msg=key)

    def test_aspect_fit_bars_are_excluded(self):
        cases = (
            mapping_shape(640, 480, source_width=640, source_height=360,
                          scale_x=1.0, scale_y=0.75),
            mapping_shape(640, 360, source_width=480, source_height=360,
                          scale_x=0.75, scale_y=1.0),
        )
        for shape in cases:
            with self.subTest(scale=(shape["content_scale_x"], shape["content_scale_y"])):
                mapping = identity_map(shape)
                height, width = mapping.shape
                output_u = (np.arange(width, dtype=np.float32) + 0.5) / width
                output_v = (np.arange(height, dtype=np.float32) + 0.5) / height
                lo_x = 0.5 * (1.0 - shape["content_scale_x"])
                lo_y = 0.5 * (1.0 - shape["content_scale_y"])
                content = ((output_u[None, :] >= lo_x) &
                           (output_u[None, :] <= lo_x + shape["content_scale_x"]) &
                           (output_v[:, None] >= lo_y) &
                           (output_v[:, None] <= lo_y + shape["content_scale_y"]))
                # Deliberately hostile row changes outside the fitted content rectangle.
                yy = np.arange(height, dtype=np.float32)[:, None]
                mapping[~content] = np.broadcast_to(4.0 * np.sin(yy), mapping.shape)[~content]
                source = vertical_feature_source(
                    shape["source_width"], shape["source_height"])
                metrics, maps = measure(mapping, shape, source, return_maps=True)
                self.assertEqual(metrics["warp_cross_row_shear_severity_pct"], 0.0)
                self.assertFalse(np.any(maps["support"][~content]))
                self.assertFalse(np.any(maps["bad"][~content]))

    def test_clamps_and_folds_do_not_duplicate_other_topology_metrics(self):
        shape = mapping_shape(854, 480)
        source = vertical_feature_source(854, 480)
        mapping = identity_map(shape)
        first, last = 160, 240
        # A large row-dependent change is present, but the first region is clamped and the second
        # is reversed horizontally. Both belong to existing clamp/fold metrics, not this one.
        mapping[first:last, 80:260] = 1.3 + np.arange(last - first)[:, None] * 0.01
        mapping[first:last, 500:700] = np.linspace(
            0.8, 0.2, 200, dtype=np.float32)[None, :] + (
                np.arange(last - first, dtype=np.float32)[:, None] * 0.002)
        metrics = measure(mapping, shape, source)
        self.assertEqual(metrics["warp_cross_row_shear_severity_pct"], 0.0)

    def test_packed_input_uses_worst_eye(self):
        shape = mapping_shape(854, 480)
        source = vertical_feature_source(854, 480)
        clean = identity_map(shape)
        damaged = add_row_ramp(clean, shape, 1.2)
        single = measure(damaged, shape, source)
        packed = measure(np.concatenate((clean, damaged), axis=1), shape, source)
        for key in ("warp_cross_row_shear_severity_pct",
                    "warp_cross_row_shear_bad_area_pct",
                    "warp_cross_row_shear_largest_run_pct"):
            self.assertAlmostEqual(single[key], packed[key], places=6, msg=key)

    def test_optional_real_c647_tear_exceeds_c525(self):
        repo_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
        run_root = os.environ.get(
            "SBSBENCH_REAL_ARTIFACT_ROOT",
            os.path.join(repo_root, "cmake-build-relwithdebinfo", "sbs_eval",
                         "metric27-oracle-repeat"))
        required = []
        for clip in ("c525", "c647"):
            required.extend((
                os.path.join(run_root, clip, "warp_map_shape.json"),
                os.path.join(run_root, clip, "warp_map_00013.f32"),
                os.path.join(repo_root, "tools", "sbsbench", "clips", clip,
                             "frame_00013.jpg"),
            ))
        if not all(os.path.exists(path) for path in required):
            self.skipTest("set SBSBENCH_REAL_ARTIFACT_ROOT to a run containing c525/c647")

        results = {}
        for clip in ("c525", "c647"):
            clip_root = os.path.join(run_root, clip)
            with open(os.path.join(clip_root, "warp_map_shape.json"),
                      encoding="utf-8") as stream:
                shape = json.load(stream)
            mapping = np.fromfile(
                os.path.join(clip_root, "warp_map_00013.f32"), dtype="<f4").reshape(
                    shape["height"], shape["width"])
            source = np.asarray(Image.open(
                os.path.join(repo_root, "tools", "sbsbench", "clips", clip,
                             "frame_00013.jpg")).convert("RGB"))
            results[clip] = measure(mapping, shape, source)

        self.assertGreater(results["c647"]["warp_cross_row_shear_severity_pct"],
                           results["c525"]["warp_cross_row_shear_severity_pct"] * 3.0)
        self.assertGreater(results["c647"]["warp_cross_row_shear_largest_run_pct"],
                           results["c525"]["warp_cross_row_shear_largest_run_pct"] * 3.0)


if __name__ == "__main__":
    unittest.main()
