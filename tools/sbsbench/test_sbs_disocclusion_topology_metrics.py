"""Focused corruption qualification for disocclusion/topology metrics."""

import os
import sys
import unittest

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import sbs_disocclusion_topology_metrics as topology  # noqa: E402


def _shape(eye_width=160, height=90, scale_x=1.0, scale_y=1.0):
    return {
        "width": eye_width * 2,
        "height": height,
        "eye_width": eye_width,
        "eye_height": height,
        "source_width": 160,
        "source_height": 90,
        "content_scale_x": scale_x,
        "content_scale_y": scale_y,
    }


def _sample_uv(image, u, v):
    return topology._sample_uv(image, u, v)


def _scene(eye_width=160, height=90, scale_x=1.0, scale_y=1.0,
           corruption=0.0, unrelated=False, split=False, constant_depth=False,
           low_near=False):
    shape = _shape(eye_width, height, scale_x, scale_y)
    source_h, source_w = shape["source_height"], shape["source_width"]
    yy, xx = np.mgrid[:source_h, :source_w].astype(np.float32)
    texture = 0.018 * np.sin(xx * 0.23) + 0.012 * np.cos(yy * 0.31)
    background = np.stack((
        0.12 + texture, 0.24 + 0.5 * texture, 0.58 - texture), axis=2)
    foreground_color = np.asarray((0.88, 0.68, 0.16), np.float32)
    foreground = ((xx >= 0.34 * source_w) & (xx < 0.58 * source_w)
                  & (yy >= 0.13 * source_h) & (yy < 0.87 * source_h))
    source = background.copy()
    source[foreground] = foreground_color + np.stack((
        0.012 * np.sin(yy[foreground] * 0.2),
        0.008 * np.cos(yy[foreground] * 0.17),
        np.zeros(np.count_nonzero(foreground), np.float32)), axis=1)
    source = np.clip(source, 0.0, 1.0).astype(np.float32)

    depth = np.full((source_h, source_w), 0.12, np.float32)
    if not constant_depth:
        depth[foreground] = 0.88
    if low_near:
        depth = 1.0 - depth

    output_u = (np.arange(eye_width, dtype=np.float32) + 0.5) / eye_width
    output_v = (np.arange(height, dtype=np.float32) + 0.5) / height
    lo_x, lo_y = 0.5 * (1.0 - scale_x), 0.5 * (1.0 - scale_y)
    source_u = (output_u - lo_x) / scale_x
    source_v = (output_v - lo_y) / scale_y
    content = ((output_u[None, :] >= lo_x) &
               (output_u[None, :] <= lo_x + scale_x) &
               (output_v[:, None] >= lo_y) &
               (output_v[:, None] <= lo_y + scale_y))
    eye_map = np.broadcast_to(source_u[None, :], (height, eye_width)).copy()
    eye = _sample_uv(
        source, np.clip(eye_map, 0.0, 1.0),
        np.broadcast_to(np.clip(source_v, 0.0, 1.0)[:, None], eye_map.shape))
    eye[~content] = 0.0

    # Output location of the foreground's right edge.  The forward hole sits on its background
    # side and is therefore independently supported by the high-near depth step.
    edge_x = int(round((lo_x + 0.58 * scale_x) * eye_width))
    y0 = int(round((lo_y + 0.18 * scale_y) * height))
    y1 = int(round((lo_y + 0.82 * scale_y) * height))
    hole_width = max(3, int(round(eye_width * scale_x * 0.01875)))
    hole = np.zeros((height, eye_width), dtype=bool)
    hole[y0:y1, edge_x:edge_x + hole_width] = True
    if split:
        hole[1::2] = False

    corrupted = eye.copy()
    if corruption > 0.0:
        if unrelated:
            direction = foreground_color - np.asarray((0.12, 0.24, 0.58), np.float32)
            orthogonal = np.asarray((direction[1], -direction[0], 0.0), np.float32)
            orthogonal /= np.linalg.norm(orthogonal)
            replacement = np.clip(corrupted[hole] + corruption * 0.28 * orthogonal,
                                  0.0, 1.0)
            corrupted[hole] = replacement
        else:
            corrupted[hole] = ((1.0 - corruption) * corrupted[hole]
                               + corruption * foreground_color)

    mapping = np.concatenate((eye_map, eye_map), axis=1)
    mask = np.zeros((height, 2 * eye_width, 3), np.uint8)
    mask[:, :eye_width, 0][hole] = 255
    mask[:, eye_width:, 0][hole] = 255
    return {
        "source": source,
        "left": corrupted.copy(),
        "right": corrupted.copy(),
        "mapping": mapping,
        "mask": mask,
        "depth": depth,
        "shape": shape,
        "hole": hole,
        "clean_eye": eye,
        "near_is_high": not low_near,
    }


def _measure(case, return_maps=False, **kwargs):
    return topology.measure_disocclusion_topology(
        case["source"], case["left"], case["right"], case["mapping"],
        case["mask"], case["depth"], case["shape"],
        near_is_high=case["near_is_high"], min_supported_hole_pixels=4,
        min_foreground_support_pixels=8, return_maps=return_maps, **kwargs)


class DisocclusionTopologyTests(unittest.TestCase):
    def test_clean_background_fill_is_not_an_artifact(self):
        metrics, maps = _measure(_scene(corruption=0.0), return_maps=True)

        self.assertEqual(metrics["disocclusion_bad_fill_abstained"], 0.0)
        self.assertEqual(metrics["disocclusion_bad_fill_evidence_sufficient"], 100.0)
        self.assertEqual(metrics["foreground_leak_evidence_sufficient"], 100.0)
        self.assertEqual(metrics["disocclusion_bad_fill_pct"], 0.0)
        self.assertEqual(metrics["foreground_leak_burden_pct"], 0.0)
        self.assertGreater(metrics["disocclusion_topology_support_count"], 0)
        self.assertTrue(np.any(maps["disoccluded_raw"]))
        self.assertTrue(np.any(maps["disoccluded_supported"]))
        self.assertFalse(np.any(maps["bad_fill"]))
        self.assertFalse(np.any(maps["foreground_leak"]))
        categories = (maps["non_occluded"].astype(np.uint8)
                      + maps["disoccluded_raw"].astype(np.uint8)
                      + maps["out_of_frame"].astype(np.uint8))
        self.assertTrue(np.array_equal(categories > 0, maps["content"]))
        self.assertLessEqual(int(categories.max()), 1)

    def test_foreground_smear_increases_continuous_leak_burden(self):
        levels = [_measure(_scene(corruption=value)) for value in (0.0, 0.25, 0.5, 1.0)]
        burdens = [level["foreground_leak_burden_pct"] for level in levels]

        self.assertEqual(burdens[0], 0.0)
        self.assertTrue(all(a < b for a, b in zip(burdens, burdens[1:])), burdens)
        self.assertGreater(levels[-1]["disocclusion_bad_fill_area_pct"], 0.5)
        self.assertGreater(levels[-1]["foreground_leak_area_pct"], 0.5)

    def test_visible_unrelated_colour_error_is_bad_fill_not_foreground_leak(self):
        metrics = _measure(_scene(corruption=1.0, unrelated=True))

        self.assertGreater(metrics["disocclusion_bad_fill_pct"], 50.0)
        self.assertEqual(metrics["foreground_leak_burden_pct"], 0.0)
        self.assertEqual(metrics["foreground_leak_area_pct"], 0.0)

    def test_foreground_selecting_map_cannot_define_its_own_perfect_reference(self):
        case = _scene(corruption=1.0)
        width = case["shape"]["eye_width"]
        for eye_index in range(2):
            eye_map = case["mapping"][:, eye_index * width:(eye_index + 1) * width]
            # The final eye already contains foreground colour. Make the suspect exact map point
            # at that same foreground. A map-relative residual alone would now be exactly zero.
            eye_map[case["hole"]] = 0.55
        metrics, maps = _measure(case, return_maps=True)

        self.assertGreater(metrics["disocclusion_bad_fill_pct"], 50.0)
        self.assertGreater(metrics["foreground_leak_burden_pct"], 0.1)
        selected = maps["disoccluded_supported"]
        self.assertFalse(np.any(maps["map_background_authenticated"][selected]))

    def test_map_authenticated_background_texture_is_not_forced_to_edge_colour(self):
        case = _scene(corruption=0.0)
        width, height = case["shape"]["eye_width"], case["shape"]["eye_height"]
        source_v = np.broadcast_to(
            ((np.arange(height, dtype=np.float32) + 0.5) / height)[:, None],
            (height, width))
        alternate_u = np.full((height, width), 0.76, np.float32)
        alternate = _sample_uv(case["source"], alternate_u, source_v)
        for eye_index, eye_name in enumerate(("left", "right")):
            eye_map = case["mapping"][:, eye_index * width:(eye_index + 1) * width]
            eye_map[case["hole"]] = alternate_u[case["hole"]]
            case[eye_name][case["hole"]] = alternate[case["hole"]]
        metrics = _measure(case)

        self.assertEqual(metrics["disocclusion_bad_fill_pct"], 0.0)
        self.assertEqual(metrics["foreground_leak_burden_pct"], 0.0)

    def test_mask_without_independent_depth_edge_abstains(self):
        metrics, maps = _measure(
            _scene(corruption=1.0, constant_depth=True), return_maps=True)

        self.assertGreater(metrics["disocclusion_raw_hole_count"], 0)
        self.assertEqual(metrics["disocclusion_topology_support_count"], 0)
        self.assertEqual(metrics["disocclusion_bad_fill_abstained"], 1.0)
        self.assertEqual(metrics["disocclusion_bad_fill_evidence_sufficient"], 0.0)
        self.assertEqual(metrics["foreground_leak_evidence_sufficient"], 0.0)
        self.assertIsNone(metrics["disocclusion_bad_fill_pct"])
        self.assertIsNone(metrics["foreground_leak_burden_pct"])
        self.assertFalse(np.any(maps["disoccluded_supported"]))

    def test_clamped_region_is_out_of_frame_not_disocclusion(self):
        case = _scene(corruption=1.0)
        width = case["shape"]["eye_width"]
        for eye_index in range(2):
            eye_map = case["mapping"][:, eye_index * width:(eye_index + 1) * width]
            eye_map[case["hole"]] = -0.02
        metrics, maps = _measure(case, return_maps=True)

        self.assertEqual(metrics["disocclusion_raw_hole_count"], 0)
        self.assertEqual(metrics["disocclusion_bad_fill_pct"], 0.0)
        self.assertTrue(np.any(maps["out_of_frame"]))
        self.assertFalse(np.any(maps["disoccluded_raw"]))

    def test_largest_component_distinguishes_coherent_from_split_corruption(self):
        coherent = _measure(_scene(corruption=1.0, split=False))
        split = _measure(_scene(corruption=1.0, split=True))

        self.assertGreater(
            coherent["foreground_leak_largest_component_pct"],
            split["foreground_leak_largest_component_pct"] * 3.0)

    def test_metrics_are_resolution_normalized(self):
        low = _measure(_scene(160, 90, corruption=0.6))
        high = _measure(_scene(320, 180, corruption=0.6))

        for key in ("disocclusion_bad_fill_area_pct", "foreground_leak_burden_pct",
                    "foreground_leak_area_pct",
                    "foreground_leak_largest_component_pct"):
            with self.subTest(key=key):
                self.assertAlmostEqual(low[key], high[key], delta=0.35)

    def test_aspect_fit_bars_are_excluded(self):
        full = _measure(_scene(160, 90, corruption=0.75))
        pillar = _measure(_scene(200, 90, scale_x=0.8, corruption=0.75))
        letter = _measure(_scene(160, 120, scale_y=0.75, corruption=0.75))

        for candidate in (pillar, letter):
            self.assertAlmostEqual(
                full["foreground_leak_burden_pct"],
                candidate["foreground_leak_burden_pct"], delta=0.4)
            self.assertAlmostEqual(
                full["disocclusion_bad_fill_area_pct"],
                candidate["disocclusion_bad_fill_area_pct"], delta=0.4)

    def test_authenticated_low_near_depth_contract_matches_high_near(self):
        high = _measure(_scene(corruption=0.65, low_near=False))
        low = _measure(_scene(corruption=0.65, low_near=True))

        self.assertAlmostEqual(
            high["foreground_leak_burden_pct"], low["foreground_leak_burden_pct"],
            places=5)
        self.assertAlmostEqual(
            high["disocclusion_bad_fill_area_pct"],
            low["disocclusion_bad_fill_area_pct"], places=5)

    def test_no_hole_is_verified_zero_but_no_edge_opportunity_abstains(self):
        case = _scene(corruption=0.0, constant_depth=True)
        case["mask"][...] = 0
        metrics = _measure(case)

        self.assertEqual(metrics["disocclusion_bad_fill_pct"], 0.0)
        self.assertEqual(metrics["disocclusion_bad_fill_abstained"], 0.0)
        self.assertEqual(metrics["disocclusion_bad_fill_evidence_sufficient"], 100.0)
        self.assertIsNone(metrics["foreground_leak_burden_pct"])
        self.assertEqual(metrics["foreground_leak_abstained"], 1.0)
        self.assertEqual(metrics["foreground_leak_evidence_sufficient"], 0.0)

    def test_zero_mask_abstains_when_exact_map_predicts_depth_edge_opportunity(self):
        case = _scene(corruption=1.0)
        width = case["shape"]["eye_width"]
        source_v = ((np.arange(case["shape"]["eye_height"], dtype=np.float32) + 0.5)
                    / case["shape"]["eye_height"])
        source_u = ((np.arange(width, dtype=np.float32) + 0.5) / width)
        sampled_depth = _sample_uv(
            case["depth"], np.broadcast_to(source_u[None, :], case["hole"].shape),
            np.broadcast_to(source_v[:, None], case["hole"].shape))
        foreground = sampled_depth >= 0.5
        for eye_index, direction in enumerate((-1.0, 1.0)):
            eye_map = case["mapping"][:, eye_index * width:(eye_index + 1) * width]
            eye_map[foreground] += direction * 3.0 / width
        case["mask"][...] = 0

        metrics, maps = _measure(case, return_maps=True)

        self.assertEqual(metrics["disocclusion_raw_hole_count"], 0)
        self.assertGreater(metrics["disocclusion_predicted_opportunity_count"], 0)
        self.assertEqual(metrics["disocclusion_bad_fill_evidence_sufficient"], 0.0)
        self.assertEqual(metrics["disocclusion_bad_fill_abstained"], 1.0)
        self.assertIsNone(metrics["disocclusion_bad_fill_burden_pct"])
        self.assertTrue(np.any(maps["predicted_disocclusion_opportunity"]))

    def test_zero_mask_allows_common_translation_without_depth_dependent_jump(self):
        case = _scene(corruption=0.0)
        width = case["shape"]["eye_width"]
        case["mapping"] += 2.0 / width
        case["mask"][...] = 0

        metrics = _measure(case)

        self.assertEqual(metrics["disocclusion_predicted_opportunity_count"], 0)
        self.assertEqual(metrics["disocclusion_bad_fill_evidence_sufficient"], 100.0)
        self.assertEqual(metrics["disocclusion_bad_fill_abstained"], 0.0)
        self.assertEqual(metrics["disocclusion_bad_fill_burden_pct"], 0.0)

    def test_mismatched_contract_fails_closed(self):
        case = _scene()
        case["shape"] = dict(case["shape"], source_width=159)
        with self.assertRaisesRegex(ValueError, "source"):
            _measure(case)


if __name__ == "__main__":
    unittest.main()
