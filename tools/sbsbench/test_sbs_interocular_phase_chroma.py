"""Synthetic validation for the experimental interocular phase/orientation metric."""

import os
import sys
import unittest

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import sbs_interocular_phase_chroma as phase_chroma  # noqa: E402


def color_scene(width=192, height=96):
    x = (np.arange(width, dtype=np.float32) + 0.5) / width
    y = (np.arange(height, dtype=np.float32) + 0.5) / height
    xx, yy = np.meshgrid(x, y)
    red = (0.39 + 0.14 * np.sin(2.0 * np.pi * (4.0 * xx + 0.3 * yy))
           + 0.08 * np.cos(2.0 * np.pi * 3.0 * yy))
    green = (0.41 + 0.13 * np.sin(2.0 * np.pi * (2.0 * xx - 1.7 * yy) + 0.7)
             + 0.07 * ((xx > 0.25) & (xx < 0.68)))
    blue = (0.36 + 0.12 * np.cos(2.0 * np.pi * (3.2 * xx + 1.1 * yy))
            - 0.06 * ((yy > 0.52) & (xx > 0.58)))
    return np.clip(np.stack((red, green, blue), axis=2), 0.08, 0.76).astype(np.float32)


def gray_scene(width=192, height=96):
    rgb = color_scene(width, height)
    gray = rgb @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    return np.repeat(gray[..., None], 3, axis=2)


def stereo_maps(width, height, disparity_pct=2.0):
    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    v = (np.arange(height, dtype=np.float32) + 0.5) / height
    disparity = disparity_pct / 100.0 * (0.45 + 0.55 * np.sin(np.pi * v) ** 2)
    left = np.broadcast_to(u[None, :], (height, width)).copy()
    right = left.copy()
    left += 0.5 * disparity[:, None]
    right -= 0.5 * disparity[:, None]
    return left.astype(np.float32), right.astype(np.float32)


def sample_horizontal(source, source_u):
    source_height, source_width = source.shape[:2]
    if source_u.shape[0] != source_height:
        raise ValueError("test sampler requires equal source/output heights")
    x = np.clip(source_u * source_width - 0.5, 0.0, source_width - 1.0)
    lo = np.floor(x).astype(np.int32)
    hi = np.minimum(lo + 1, source_width - 1)
    fraction = x - lo
    rows = np.arange(source_height, dtype=np.int32)[:, None]
    return ((1.0 - fraction[..., None]) * source[rows, lo]
            + fraction[..., None] * source[rows, hi]).astype(np.float32)


def measure(source, left, right, left_map, right_map, shape=None, return_maps=False,
            warp_mask=None):
    if shape is None:
        shape = {"content_scale_x": 1.0, "content_scale_y": 1.0}
    return phase_chroma.measure_interocular_phase_chroma(
        source, left, right, left_map, right_map, shape,
        warp_mask=warp_mask,
        max_analysis_width=320, max_analysis_height=180,
        min_phase_pixels=64, return_maps=return_maps)


class InterocularPhaseOrientationTests(unittest.TestCase):
    def setUp(self):
        self.source = color_scene()
        self.left_map, self.right_map = stereo_maps(192, 96)
        self.left = sample_horizontal(self.source, self.left_map)
        self.right = sample_horizontal(self.source, self.right_map)

    def test_exact_maps_remove_legitimate_disparity(self):
        metrics, evidence = measure(
            self.source, self.left, self.right, self.left_map, self.right_map,
            return_maps=True)
        self.assertGreater(metrics["interocular_phase_orientation_support_pct"], 45.0)
        self.assertEqual(
            metrics["interocular_phase_orientation_evidence_sufficient"], 100.0)
        self.assertLess(metrics["interocular_phase_orientation_burden_pct"], 0.5)
        self.assertEqual(
            evidence["phase_orientation_conflict_pct"].shape, (160, 320))
        self.assertEqual(set(metrics), {
            "interocular_phase_orientation_burden_pct",
            "interocular_phase_orientation_support_pct",
            "interocular_phase_orientation_support_count",
            "interocular_phase_orientation_evidence_sufficient",
        })

    def test_multichannel_registration_matches_scalar_reference(self):
        combined = np.concatenate((self.left, self.right), axis=2)
        target_u = (np.arange(200, dtype=np.float32) + 0.5) / 200.0
        content = np.ones(combined.shape[1], dtype=bool)
        registered, valid = phase_chroma._invert_row_channels(
            combined[48], self.left_map[48], content, target_u)
        for channel in range(combined.shape[2]):
            scalar, scalar_valid = phase_chroma._registration._invert_row(
                combined[48, :, channel], self.left_map[48], content, target_u)
            np.testing.assert_array_equal(valid, scalar_valid)
            np.testing.assert_allclose(
                registered[:, channel][valid], scalar[valid], atol=1e-7)

    def test_default_analysis_raster_is_640_wide(self):
        identity = stereo_maps(192, 96, disparity_pct=0.0)
        _, evidence = phase_chroma.measure_interocular_phase_chroma(
            self.source, self.source, self.source, *identity,
            {"content_scale_x": 1.0, "content_scale_y": 1.0}, return_maps=True)
        self.assertEqual(evidence["phase_orientation_conflict_pct"].shape, (320, 640))

    def test_equal_detail_phase_shift_is_detected(self):
        source = gray_scene()
        left_map, right_map = stereo_maps(192, 96, disparity_pct=0.0)
        left = source.copy()
        clean_right = source.copy()
        faulty_right = clean_right.copy()
        patch = faulty_right[16:80, 36:156].copy()
        faulty_right[16:80, 36:156] = np.roll(patch, 4, axis=1)

        clean = measure(source, left, clean_right, left_map, right_map)
        faulty = measure(source, left, faulty_right, left_map, right_map)
        self.assertGreater(
            faulty["interocular_phase_orientation_burden_pct"],
            clean["interocular_phase_orientation_burden_pct"] + 2.0,
        )

    def test_forward_hole_fill_is_excluded_from_phase_conflict(self):
        faulty = self.right.copy()
        faulty[20:76, 55:140] = np.roll(faulty[20:76, 55:140], 5, axis=1)
        packed_mask = np.zeros((96, 384, 3), dtype=np.float32)
        packed_mask[20:76, 192 + 55:192 + 140, 0] = 1.0

        unmasked = measure(
            self.source, self.left, faulty, self.left_map, self.right_map)
        masked = measure(
            self.source, self.left, faulty, self.left_map, self.right_map,
            warp_mask=packed_mask)

        self.assertGreater(
            unmasked["interocular_phase_orientation_burden_pct"],
            masked["interocular_phase_orientation_burden_pct"] + 1.0)
        self.assertLess(
            masked["interocular_phase_orientation_support_pct"],
            unmasked["interocular_phase_orientation_support_pct"])

    def test_localized_pooling_has_no_five_percent_footprint_cliff(self):
        height, width = 200, 500
        support = np.ones((height, width), dtype=bool)
        weights = np.ones((height, width), dtype=np.float32)
        footprints = (0.0008, 0.005, 0.01, 0.02, 0.05)
        burdens = []
        for footprint in footprints:
            pixel_count = int(round(height * width * footprint))
            values = np.zeros((height, width), dtype=np.float32)
            # One coherent thin rectangle, including the sub-0.1% case.
            rectangle_height = 20
            rectangle_width = max(1, pixel_count // rectangle_height)
            values[40:40 + rectangle_height, 60:60 + rectangle_width] = 55.0
            actual_fraction = rectangle_height * rectangle_width / float(height * width)
            burden = phase_chroma._localized_burden(
                values, weights, support, visibility_floor=5.0)
            self.assertAlmostEqual(
                burden, 50.0 * np.sqrt(actual_fraction), places=5)
            burdens.append(burden)
        self.assertTrue(all(a < b for a, b in zip(burdens, burdens[1:])), burdens)

    def test_one_percent_phase_fault_is_localized(self):
        left_map, right_map = stereo_maps(192, 96, disparity_pct=0.0)
        x0, y0, patch_width, patch_height = 92, 36, 8, 24

        gray = gray_scene()
        phase_fault = gray.copy()
        patch = phase_fault[y0:y0 + patch_height, x0:x0 + patch_width].copy()
        phase_fault[y0:y0 + patch_height, x0:x0 + patch_width] = np.roll(
            patch, 4, axis=1)
        phase_metrics, phase_evidence = measure(
            gray, gray, phase_fault, left_map, right_map, return_maps=True)
        self.assertGreater(
            phase_metrics["interocular_phase_orientation_burden_pct"], 1.0)
        conflict = np.nan_to_num(
            phase_evidence["phase_orientation_conflict_pct"], nan=0.0)
        active = conflict > 5.0
        analysis_height, analysis_width = active.shape
        scale_x, scale_y = analysis_width / 192.0, analysis_height / 96.0
        expected = np.zeros(active.shape, dtype=bool)
        margin = 12
        left = max(0, int(x0 * scale_x) - margin)
        right = min(
            analysis_width, int((x0 + patch_width) * scale_x) + margin)
        top = max(0, int(y0 * scale_y) - margin)
        bottom = min(
            analysis_height, int((y0 + patch_height) * scale_y) + margin)
        expected[top:bottom, left:right] = True
        self.assertGreater(np.count_nonzero(active), 0)
        precision = np.count_nonzero(active & expected) / np.count_nonzero(active)
        self.assertGreater(precision, 0.95)

    def test_equal_detail_orientation_mismatch_is_detected(self):
        source = gray_scene()
        identity = stereo_maps(192, 96, disparity_pct=0.0)
        faulty = source.copy()
        patch = faulty[12:84, 48:144].copy()
        faulty[12:84, 48:144] = np.flip(patch, axis=0)
        baseline = measure(source, source, source, *identity)
        changed = measure(source, source, faulty, *identity)
        self.assertGreater(
            changed["interocular_phase_orientation_burden_pct"],
            baseline["interocular_phase_orientation_burden_pct"] + 2.0,
        )

    def test_global_exposure_and_white_balance_are_benign(self):
        left = self.left * np.asarray((0.78, 0.70, 0.74), dtype=np.float32)
        right = self.right * np.asarray((1.10, 1.24, 1.16), dtype=np.float32)
        metrics = measure(self.source, left, right, self.left_map, self.right_map)
        self.assertLess(metrics["interocular_phase_orientation_burden_pct"], 1.0)

    def test_global_detail_imbalance_abstains_instead_of_becoming_phase(self):
        yy, xx = np.mgrid[:96, :192].astype(np.float32)
        detail = (0.42 + 0.13 * np.sin(0.72 * xx + 0.11 * yy)
                  + 0.11 * np.cos(0.63 * yy - 0.08 * xx))
        source = np.repeat(np.clip(detail, 0.08, 0.78)[..., None], 3, axis=2)
        left_map, right_map = stereo_maps(192, 96, disparity_pct=0.0)
        blurred_right = np.stack([
            phase_chroma._registration._box_mean(source[..., channel], 4)
            for channel in range(3)
        ], axis=2)
        metrics = measure(
            source, source, blurred_right, left_map, right_map)
        self.assertEqual(
            metrics["interocular_phase_orientation_evidence_sufficient"], 0.0)
        self.assertIsNone(metrics["interocular_phase_orientation_burden_pct"])

    def test_proportional_phase_fault_is_resolution_stable(self):
        low = gray_scene()
        low_maps = stereo_maps(192, 96, disparity_pct=0.0)
        low_fault = low.copy()
        low_patch = low_fault[16:80, 36:156].copy()
        low_fault[16:80, 36:156] = np.roll(low_patch, 4, axis=1)
        low_value = measure(
            low, low, low_fault, *low_maps)["interocular_phase_orientation_burden_pct"]

        high = phase_chroma._sample_source_rgb(low, 384, 192)
        high_maps = stereo_maps(384, 192, disparity_pct=0.0)
        high_fault = high.copy()
        high_patch = high_fault[32:160, 72:312].copy()
        high_fault[32:160, 72:312] = np.roll(high_patch, 8, axis=1)
        high_value = measure(
            high, high, high_fault, *high_maps)["interocular_phase_orientation_burden_pct"]
        relative_difference = abs(low_value - high_value) / (0.5 * (low_value + high_value))
        self.assertLess(relative_difference, 0.25)

    def test_fractional_hdr_source_is_transformed_after_sampling(self):
        height, width = 96, 192
        source = color_scene(width, height)
        linear = np.where(source < 0.42, 0.04, source * 8.0).astype(np.float32)
        u = (np.arange(width, dtype=np.float32) + 0.5) / width
        left_map = np.broadcast_to((u - 0.43 / width)[None, :], (height, width)).copy()
        right_map = np.broadcast_to((u + 0.43 / width)[None, :], (height, width)).copy()

        def tonemap(rgb):
            return rgb / (1.0 + rgb)

        correct_left = tonemap(sample_horizontal(linear, left_map))
        correct_right = tonemap(sample_horizontal(linear, right_map))
        wrong_left = sample_horizontal(tonemap(linear), left_map)
        wrong_right = sample_horizontal(tonemap(linear), right_map)
        shape = {"content_scale_x": 1.0, "content_scale_y": 1.0}
        kwargs = {
            "source_sample_transform": tonemap,
            "max_analysis_width": 320,
            "max_analysis_height": 180,
            "min_phase_pixels": 64,
        }
        correct = phase_chroma.measure_interocular_phase_chroma(
            linear, correct_left, correct_right, left_map, right_map, shape, **kwargs)
        wrong = phase_chroma.measure_interocular_phase_chroma(
            linear, wrong_left, wrong_right, left_map, right_map, shape, **kwargs)
        self.assertEqual(
            correct["interocular_phase_orientation_evidence_sufficient"], 100.0)
        self.assertLess(correct["interocular_phase_orientation_burden_pct"], 0.1)
        self.assertGreater(
            wrong["interocular_phase_orientation_burden_pct"],
            correct["interocular_phase_orientation_burden_pct"] + 0.1,
        )

    def test_invalid_source_transform_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "invalid RGB evidence"):
            phase_chroma.measure_interocular_phase_chroma(
                self.source, self.left, self.right, self.left_map, self.right_map,
                {"content_scale_x": 1.0, "content_scale_y": 1.0},
                source_sample_transform=lambda value: value[..., 0])

    def test_pillarbox_pixels_cannot_vote(self):
        source = color_scene(128, 64)
        output_width = 160
        output_u = (np.arange(output_width, dtype=np.float32) + 0.5) / output_width
        source_u = (output_u - 0.10) / 0.80
        source_map = np.broadcast_to(source_u[None, :], (64, output_width)).copy()
        left = sample_horizontal(source, source_map)
        right = left.copy()
        random = np.random.default_rng(7)
        left[:, :16] = random.random(left[:, :16].shape)
        left[:, 144:] = random.random(left[:, 144:].shape)
        right[:, :16] = random.random(right[:, :16].shape)
        right[:, 144:] = random.random(right[:, 144:].shape)
        shape = {"content_scale_x": 0.80, "content_scale_y": 1.0}
        metrics = measure(source, left, right, source_map, source_map, shape)
        self.assertLess(metrics["interocular_phase_orientation_burden_pct"], 0.5)

    def test_clamped_and_folded_corruption_is_excluded(self):
        source = color_scene()
        identity = stereo_maps(192, 96, disparity_pct=0.0)[0]

        clamped_map = identity.copy()
        clamped_map[:, :32] = -0.5
        clamped_eye = source.copy()
        clamped_eye[:, :32] = 1.0 - clamped_eye[:, :32]
        clamped = measure(source, source, clamped_eye, identity, clamped_map)
        self.assertLess(clamped["interocular_phase_orientation_burden_pct"], 0.5)

        folded_map = identity.copy()
        folded_map[:, 96:] = folded_map[:, 96:][:, ::-1]
        folded_eye = source.copy()
        folded_eye[:, 96:] = 1.0 - folded_eye[:, 96:]
        folded = measure(source, source, folded_eye, identity, folded_map)
        self.assertLess(folded["interocular_phase_orientation_burden_pct"], 0.5)
        self.assertLess(
            folded["interocular_phase_orientation_support_pct"],
            clamped["interocular_phase_orientation_support_pct"],
        )

    def test_flat_scene_abstains_on_phase(self):
        source = np.full((64, 128, 3), 0.4, dtype=np.float32)
        identity = stereo_maps(128, 64, disparity_pct=0.0)
        metrics = measure(source, source, source, *identity)
        self.assertEqual(metrics["interocular_phase_orientation_support_count"], 0)
        self.assertEqual(
            metrics["interocular_phase_orientation_evidence_sufficient"], 0.0)
        self.assertIsNone(metrics["interocular_phase_orientation_burden_pct"])


if __name__ == "__main__":
    unittest.main()
