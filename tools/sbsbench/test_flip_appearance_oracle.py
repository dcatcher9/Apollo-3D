import importlib.util
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageFilter

try:
    import flip_appearance_oracle as oracle
    import sbs_interocular_phase_chroma as exact
except ImportError:  # Package discovery from the repository root.
    from . import flip_appearance_oracle as oracle
    from . import sbs_interocular_phase_chroma as exact


HAS_FLIP = importlib.util.find_spec("flip_evaluator") is not None
P99 = "flip_worst_eye_p99"
AREA = "flip_worst_eye_area_gt_050_pct"
IMBALANCE = "flip_interocular_error_imbalance_p99"


def _source(height=160, width=256):
    rows, columns = np.indices((height, width))
    checker = ((columns // 8 + rows // 8) % 2).astype(np.float32)
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[:] = (0.08, 0.12, 0.16)
    image[:, width // 2:] = (0.75, 0.65, 0.45)
    image[..., 0] += checker * 0.08
    image[..., 1] += checker * 0.05
    image[30:130, 40:110] = (0.20, 0.75, 0.35)
    image[45:115, 58:93] = (0.90, 0.20, 0.15)
    image[20:140:9, 125:128] = 1.0
    image[75:77, 10:245] = (0.95, 0.95, 0.95)
    return np.clip(image, 0.0, 1.0)


def _maps(height, width, disparity_px=0.0, fractional_px=0.0):
    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    left = np.broadcast_to(u[None, :], (height, width)).copy()
    right = left.copy()
    amount = 0.5 * disparity_px / width + fractional_px / width
    left += amount
    right -= amount
    return left, right


def _render(source, maps, shape):
    return tuple(exact._sample_source_eye(source, mapping, shape) for mapping in maps)


def _blur(image, radius):
    encoded = Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0))
    return np.asarray(encoded.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0


@unittest.skipUnless(HAS_FLIP, "official optional flip-evaluator package is not installed")
class FlipAppearanceQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = _source()
        cls.height, cls.width = cls.source.shape[:2]
        cls.shape = {"content_scale_x": 1.0, "content_scale_y": 1.0}
        cls.maps = _maps(cls.height, cls.width)
        cls.clean = _render(cls.source, cls.maps, cls.shape)

    def _measure(self, left, right, maps=None, shape=None, source=None):
        result = oracle.measure_flip_appearance(
            self.source if source is None else source,
            left, right,
            *(self.maps if maps is None else maps),
            self.shape if shape is None else shape,
            min_support_pixels=64,
        )
        self.assertEqual("ok", result["status"], result)
        self.assertFalse(result["training_label_eligible"])
        self.assertEqual("experimental_diagnostic_only", result["qualification"])
        self.assertNotIn("mean", " ".join(result["metrics"]).lower())
        return result["metrics"]

    def test_clean_and_flat_reference_are_zero(self):
        clean = self._measure(*self.clean)
        self.assertEqual(0.0, clean[P99])
        self.assertEqual(0.0, clean[AREA])
        self.assertEqual(0.0, clean[IMBALANCE])

        flat = np.full_like(self.source, 0.31)
        flat_eyes = _render(flat, self.maps, self.shape)
        flat_metrics = self._measure(*flat_eyes, source=flat)
        self.assertEqual(0.0, flat_metrics[P99])
        self.assertEqual(0.0, flat_metrics[AREA])

    def test_exact_geometry_and_fractional_reference_regeneration_are_benign(self):
        maps = _maps(self.height, self.width, disparity_px=8.0, fractional_px=0.43)
        eyes = _render(self.source, maps, self.shape)
        metrics = self._measure(*eyes, maps=maps)
        self.assertLess(metrics[P99], 1e-6)
        self.assertLess(metrics[AREA], 1e-6)

    def test_bars_and_invalid_map_regions_are_excluded(self):
        scale = 0.72
        lo = 0.5 * (1.0 - scale)
        output_u = (np.arange(self.width, dtype=np.float32) + 0.5) / self.width
        mapped_u = (output_u - lo) / scale
        maps = tuple(
            np.broadcast_to(mapped_u[None, :], (self.height, self.width)).copy()
            for _ in range(2)
        )
        shape = {"content_scale_x": scale, "content_scale_y": 1.0}
        eyes = [eye.copy() for eye in _render(self.source, maps, shape)]
        bars = (output_u < lo) | (output_u > 1.0 - lo)
        rows, columns = np.indices((self.height, self.width))
        hostile = ((rows + columns) % 2).astype(np.float32)
        for index, eye in enumerate(eyes):
            eye[:, bars] = np.stack((
                hostile[:, bars] if index == 0 else 1.0 - hostile[:, bars],
                np.ones_like(hostile[:, bars]),
                1.0 - hostile[:, bars] if index == 0 else hostile[:, bars],
            ), axis=2)
        metrics = self._measure(*eyes, maps=maps, shape=shape)
        self.assertLess(metrics[P99], 1e-6)
        self.assertLess(metrics[AREA], 1e-6)

        invalid_maps = [mapping.copy() for mapping in self.maps]
        invalid_maps[0][55:105, 100:150] = -1.0
        invalid_maps[1][55:105, 100:150] = -1.0
        invalid_eyes = [eye.copy() for eye in _render(self.source, invalid_maps, self.shape)]
        invalid_eyes[0][55:105, 100:150] = (1.0, 0.0, 1.0)
        invalid_eyes[1][55:105, 100:150] = (0.0, 1.0, 0.0)
        masked = self._measure(*invalid_eyes, maps=tuple(invalid_maps))
        self.assertLess(masked[P99], 1e-6)
        self.assertLess(masked[AREA], 1e-6)

    def test_blur_ladder_is_monotonic(self):
        scores = [
            self._measure(self.clean[0], _blur(self.clean[1], radius))[P99]
            for radius in (1, 2, 4)
        ]
        self.assertGreater(scores[0], 0.10)
        self.assertTrue(all(later > earlier + 0.02
                            for earlier, later in zip(scores, scores[1:])), scores)

    def test_thin_line_deletion_ladder_is_monotonic(self):
        scores = []
        areas = []
        for thickness in (1, 2, 3):
            corrupted = self.clean[1].copy()
            corrupted[20:140, 125:125 + thickness] = corrupted[20:140, 120:121]
            metrics = self._measure(self.clean[0], corrupted)
            scores.append(metrics[P99])
            areas.append(metrics[AREA])
        self.assertGreater(scores[0], 0.05)
        self.assertTrue(all(later > earlier for earlier, later in zip(scores, scores[1:])), scores)
        self.assertTrue(all(later >= earlier for earlier, later in zip(areas, areas[1:])), areas)

    def test_halo_and_ringing_ladders_are_monotonic(self):
        halo_scores = []
        ring_scores = []
        for strength in (0.10, 0.20, 0.35):
            halo = self.clean[1].copy()
            halo[30:130, 36:40] = np.clip(
                halo[30:130, 36:40] + strength, 0.0, 1.0)
            halo_scores.append(self._measure(self.clean[0], halo)[P99])

            ringing = self.clean[1].copy()
            ringing[30:130, 110:113] = np.clip(
                ringing[30:130, 110:113] + strength, 0.0, 1.0)
            ringing[30:130, 113:116] = np.clip(
                ringing[30:130, 113:116] - strength, 0.0, 1.0)
            ring_scores.append(self._measure(self.clean[0], ringing)[P99])
        self.assertTrue(all(later > earlier
                            for earlier, later in zip(halo_scores, halo_scores[1:])), halo_scores)
        self.assertTrue(all(later > earlier
                            for earlier, later in zip(ring_scores, ring_scores[1:])), ring_scores)

    def test_jagged_and_double_edge_ladders_are_monotonic(self):
        jagged_scores = []
        double_scores = []
        for shift in (1, 2, 4):
            jagged = self.clean[1].copy()
            patch = jagged[30:130, 35:115].copy()
            for row in range(patch.shape[0]):
                patch[row] = np.roll(
                    patch[row], shift if row % 4 < 2 else -shift, axis=0)
            jagged[30:130, 35:115] = patch
            jagged_scores.append(self._measure(self.clean[0], jagged)[P99])
        for alpha in (0.15, 0.30, 0.50):
            shifted = np.roll(self.clean[1], 3, axis=1)
            doubled = (1.0 - alpha) * self.clean[1] + alpha * shifted
            double_scores.append(self._measure(self.clean[0], doubled)[P99])
        self.assertTrue(all(later > earlier
                            for earlier, later in zip(jagged_scores, jagged_scores[1:])), jagged_scores)
        self.assertTrue(all(later > earlier
                            for earlier, later in zip(double_scores, double_scores[1:])), double_scores)

    def test_one_eye_fault_reports_interocular_imbalance(self):
        one_eye = self._measure(self.clean[0], _blur(self.clean[1], 2))
        both_eyes = self._measure(_blur(self.clean[0], 2), _blur(self.clean[1], 2))
        self.assertGreater(one_eye[IMBALANCE], 0.20)
        self.assertLess(both_eyes[IMBALANCE], one_eye[IMBALANCE] * 0.10)

    def test_filter_footprint_is_measured_and_conservatively_erosion_padded(self):
        observed, erosion = oracle.measure_filter_support()
        self.assertGreater(observed, 0)
        self.assertEqual(observed + 2, erosion)
        result = oracle.measure_flip_appearance(
            self.source, *self.clean, *self.maps, self.shape, min_support_pixels=64)
        self.assertEqual(observed, result["support"]["observed_impulse_radius_px"])
        self.assertEqual(erosion, result["support"]["erosion_radius_px"])


class FlipAppearanceFailClosedTests(unittest.TestCase):
    def test_hdr_preview_abstains_before_loading_optional_dependency(self):
        tiny = np.zeros((32, 64, 3), dtype=np.uint8)
        mapping = np.broadcast_to(
            (np.arange(64, dtype=np.float32) + 0.5)[None, :] / 64.0,
            (32, 64),
        ).copy()
        with mock.patch.object(
                oracle, "load_official_flip", side_effect=AssertionError("must not load")):
            result = oracle.measure_flip_appearance(
                tiny, tiny, tiny, mapping, mapping,
                {"content_scale_x": 1.0, "content_scale_y": 1.0},
                hdr_output_stats={
                    "format": "linear-scRGB-fp16",
                    "hdr_source_kind": "native-pq-in-windows-hdr",
                },
            )
        self.assertEqual("abstained", result["status"])
        self.assertFalse(result["training_label_eligible"])
        self.assertIn("preview", result["reason"].lower())
        self.assertIsNone(result["metrics"][P99])

    def test_missing_package_is_explicitly_unavailable(self):
        tiny = np.zeros((32, 64, 3), dtype=np.uint8)
        mapping = np.broadcast_to(
            (np.arange(64, dtype=np.float32) + 0.5)[None, :] / 64.0,
            (32, 64),
        ).copy()
        with mock.patch.object(
                oracle, "load_official_flip",
                side_effect=oracle.FlipUnavailable("not installed")):
            result = oracle.measure_flip_appearance(
                tiny, tiny, tiny, mapping, mapping,
                {"content_scale_x": 1.0, "content_scale_y": 1.0},
            )
        self.assertEqual("unavailable", result["status"])
        self.assertFalse(result["training_label_eligible"])
        self.assertIn("not installed", result["reason"])


if __name__ == "__main__":
    unittest.main()
