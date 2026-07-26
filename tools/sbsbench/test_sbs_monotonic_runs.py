"""Exact-equivalence tests for vectorized inverse-map run segmentation."""

import itertools
import os
import sys
import unittest
from unittest import mock

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import sbs_interocular_metrics as geometry  # noqa: E402
import sbs_stereo_window_metrics as window_metrics  # noqa: E402


def legacy_monotonic_runs(source_u, content):
    """Scalar implementation retained here as the exact compatibility oracle."""
    source_u = np.asarray(source_u, dtype=np.float32)
    content = np.asarray(content, dtype=bool)
    indices = np.flatnonzero(content & np.isfinite(source_u))
    if indices.size < 2:
        return
    adjacent = indices[1:] == indices[:-1] + 1
    differences = source_u[indices[1:]] - source_u[indices[:-1]]
    ordinary = differences[adjacent & (differences > 1e-8)]
    typical_step = float(np.median(ordinary)) if ordinary.size else 1.0 / source_u.size
    maximum_step = max(4.0 / source_u.size, 8.0 * typical_step)
    start = 0
    for offset in range(1, indices.size):
        previous, current = indices[offset - 1], indices[offset]
        step = float(source_u[current] - source_u[previous])
        if current != previous + 1 or step <= 1e-8 or step > maximum_step:
            run = indices[start:offset]
            if run.size >= 2:
                yield run
            start = offset
    run = indices[start:]
    if run.size >= 2:
        yield run


def run_lists(implementation, source_u, content):
    return [run.tolist() for run in implementation(source_u, content)]


class MonotonicRunEquivalenceTests(unittest.TestCase):
    def assert_matches_legacy(self, source_u, content):
        self.assertEqual(
            run_lists(geometry._monotonic_runs, source_u, content),
            run_lists(legacy_monotonic_runs, source_u, content),
        )

    def test_exhaustive_small_maps_and_content_gaps(self):
        # Includes increases, folds, equal coordinates, ordinary/large jumps, and NaNs.
        values = (np.nan, -0.25, 0.0, 0.1, 0.9, 1.25)
        for size in range(0, 5):
            for source_u in itertools.product(values, repeat=size):
                for content in itertools.product((False, True), repeat=size):
                    with self.subTest(size=size, source_u=source_u, content=content):
                        self.assert_matches_legacy(source_u, content)

    def test_explicit_clamps_folds_jumps_and_noncontent(self):
        source_u = np.asarray(
            (-0.20, 0.00, 0.02, 0.04, 0.55, 0.57, 0.40, 0.42,
             np.nan, 0.44, 0.46, 1.00, 1.20),
            dtype=np.float64,
        )
        content = np.asarray(
            (True, True, True, False, True, True, True, True,
             True, False, True, True, True),
            dtype=bool,
        )
        # Match the production callers: clamped coordinates are non-invertible.
        invertible = content & (source_u >= 0.0) & (source_u <= 1.0)
        self.assert_matches_legacy(source_u, invertible)

    def test_randomized_unusual_sizes(self):
        rng = np.random.default_rng(0x5B5)
        for size in (0, 1, 2, 3, 7, 31, 257, 1021, 4093):
            for _ in range(40):
                steps = rng.normal(1.0 / max(size, 1), 0.02, size=size)
                source_u = np.cumsum(steps, dtype=np.float64)
                if size:
                    source_u -= source_u[0]
                    # Inject folds, large occlusion jumps, clamps, and non-finite samples.
                    for index in rng.choice(size, size=min(size, 8), replace=False):
                        source_u[index] = rng.choice(
                            (-0.4, 1.4, np.nan, source_u[index] - 0.3))
                content = rng.random(size) > 0.18
                self.assert_matches_legacy(source_u, content)
                invertible = content & (source_u >= 0.0) & (source_u <= 1.0)
                self.assert_matches_legacy(source_u, invertible)

    def test_float32_threshold_boundaries_match(self):
        for size in (5, 17, 513):
            baseline = np.arange(size, dtype=np.float32) / np.float32(size)
            candidates = (
                np.float32(0.0),
                np.nextafter(np.float32(1e-8), np.float32(0.0)),
                np.float32(1e-8),
                np.nextafter(np.float32(1e-8), np.float32(np.inf)),
            )
            for step in candidates:
                source_u = baseline.copy()
                if size >= 3:
                    source_u[2] = source_u[1] + step
                self.assert_matches_legacy(source_u, np.ones(size, dtype=bool))

    def test_stereo_window_routes_through_shared_segmentation(self):
        output_x = np.arange(8, dtype=np.float32)
        source_u = (output_x + 0.5) / output_x.size
        usable = np.ones(output_x.size, dtype=bool)
        target_u = source_u.copy()
        original = geometry._monotonic_runs
        with mock.patch.object(
                window_metrics._geometry, "_monotonic_runs", wraps=original) as shared:
            window_metrics._invert_row(output_x, source_u, usable, target_u)
        shared.assert_called_once()

    def test_identity_value_inversion_reuses_positions_bit_exactly(self):
        rng = np.random.default_rng(0x1D3)
        for width in (8, 31, 257):
            output_x = np.arange(width, dtype=np.float32)
            target_u = (np.arange(width // 2 + 3, dtype=np.float32) + 0.5) / (
                width // 2 + 3)
            for _ in range(20):
                source_u = np.linspace(-0.1, 1.1, width, dtype=np.float32)
                source_u += rng.normal(0.0, 0.01, width).astype(np.float32)
                usable = rng.random(width) > 0.08
                expected, expected_valid = geometry._invert_row(
                    output_x, source_u, usable, target_u)
                actual, actual_valid = geometry._invert_row(
                    output_x, source_u, usable, target_u,
                    values_are_output_positions=True)
                self.assertTrue(np.array_equal(actual, expected, equal_nan=True))
                self.assertTrue(np.array_equal(actual_valid, expected_valid))


if __name__ == "__main__":
    unittest.main()
