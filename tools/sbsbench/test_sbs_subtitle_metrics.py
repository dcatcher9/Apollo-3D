"""Deterministic tests for explicit-ground-truth subtitle-region metrics."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import authenticated_metric_sources  # noqa: E402
import run_eval  # noqa: E402
import sbs_subtitle_metrics as subtitle_metrics  # noqa: E402
import sbsbench  # noqa: E402


def mapping_shape(width, height):
    return {
        "schema": 1,
        "dtype": "float32-le",
        "layout": "row-major",
        "channels": ["raw_reproject_source_u_normalized"],
        "width": 2 * width,
        "height": height,
        "eye_width": width,
        "eye_height": height,
        "source_width": width,
        "source_height": height,
        "content_scale_x": 1.0,
        "content_scale_y": 1.0,
    }


def binocular_geometry(width, height, disparity=None, valid=None, support_weight=None):
    if disparity is None:
        disparity = np.zeros((height, width), dtype=np.float32)
    if valid is None:
        valid = np.ones((height, width), dtype=bool)
    if support_weight is None:
        support_weight = np.where(valid, 1.0, 0.0)
    return {
        "target_u": (np.arange(width, dtype=np.float32) + 0.5) / width,
        "source_v": (np.arange(height, dtype=np.float32) + 0.5) / height,
        "disparity": np.asarray(disparity, dtype=np.float32),
        "symmetry": np.zeros((height, width), dtype=np.float32),
        "weight": np.where(valid, 1.0, 0.0).astype(np.float32),
        "support_weight": np.asarray(support_weight, dtype=np.float32),
        "valid": np.asarray(valid, dtype=bool),
        "possible_count": height * width,
    }


def stripe_frame(width, height):
    columns = (np.arange(width) % 2).astype(np.float32)
    return np.broadcast_to(columns[None, :], (height, width)).copy()


def horizontal_blur(image):
    padded = np.pad(np.asarray(image, dtype=np.float32), ((0, 0), (1, 1)), mode="edge")
    return (0.25 * padded[:, :-2] + 0.5 * padded[:, 1:-1]
            + 0.25 * padded[:, 2:]).astype(np.float32)


class SubtitleMetricTests(unittest.TestCase):
    def test_constant_nonzero_disparity_has_zero_variance_but_nonzero_target_error(self):
        width, height = 32, 16
        source = stripe_frame(width, height)
        geometry = binocular_geometry(
            width, height, np.full((height, width), 3.5, dtype=np.float32))
        measured = subtitle_metrics.measure_subtitle_region(
            source, source, source, np.ones(source.shape, bool),
            mapping_shape(width, height), geometry, target_disparity_pct=0.0)
        self.assertEqual(measured["subtitle_disparity_variance_pct2"], 0.0)
        self.assertGreater(measured["subtitle_target_disparity_rms_error_pct"], 0.0)

    def test_disparity_variance_is_resolution_normalized_and_region_limited(self):
        def measured(width, height, scale):
            source = stripe_frame(width, height)
            disparity = np.zeros(source.shape, dtype=np.float32)
            disparity[:, width // 4:width // 2] = -scale
            disparity[:, width // 2:3 * width // 4] = scale
            # Large defects outside the authored loose region must not vote.
            disparity[:, :width // 4] = -10.0 * scale
            disparity[:, 3 * width // 4:] = 10.0 * scale
            region = np.zeros(source.shape, bool)
            region[:, width // 4:3 * width // 4] = True
            metrics = subtitle_metrics.measure_subtitle_region(
                source, source, source, region, mapping_shape(width, height),
                binocular_geometry(width, height, disparity),
                target_disparity_pct=0.0)
            return (metrics["subtitle_disparity_variance_pct2"],
                    metrics["subtitle_target_disparity_rms_error_pct"])

        low = measured(32, 16, 1.0)
        high = measured(64, 32, 2.0)
        self.assertGreater(low[0], 0.0)
        self.assertGreater(low[1], 0.0)
        self.assertAlmostEqual(low[0], high[0], places=5)
        self.assertAlmostEqual(low[1], high[1], places=5)

    def test_identical_eyes_preserve_horizontal_gradient_energy(self):
        width, height = 32, 16
        source = stripe_frame(width, height)
        measured = subtitle_metrics.measure_subtitle_region(
            source, source, source, np.ones(source.shape, bool),
            mapping_shape(width, height), binocular_geometry(width, height),
            target_disparity_pct=0.0)
        self.assertEqual(measured["subtitle_target_disparity_rms_error_pct"], 0.0)
        self.assertAlmostEqual(
            measured["subtitle_sharpness_preservation_pct"], 100.0, places=5)

    def test_matching_nonzero_authored_target_has_zero_error(self):
        width, height = 32, 16
        source = stripe_frame(width, height)
        one_pixel_pct = subtitle_metrics.sbs_interocular_metrics.perceived_disparity_pct(
            1.0, width, height)
        target = 1.25
        disparity = np.full(source.shape, target / one_pixel_pct, dtype=np.float32)
        measured = subtitle_metrics.measure_subtitle_region(
            source, source, source, np.ones(source.shape, bool),
            mapping_shape(width, height), binocular_geometry(width, height, disparity),
            target_disparity_pct=target)
        self.assertAlmostEqual(
            measured["subtitle_target_disparity_rms_error_pct"], 0.0, places=6)

    def test_worse_eye_horizontal_blur_reduces_sharpness(self):
        width, height = 32, 16
        source = stripe_frame(width, height)
        measured = subtitle_metrics.measure_subtitle_region(
            source, horizontal_blur(source), source, np.ones(source.shape, bool),
            mapping_shape(width, height), binocular_geometry(width, height))
        self.assertLess(measured["subtitle_sharpness_preservation_pct"], 10.0)

    def test_empty_explicit_region_abstains_even_when_frame_contains_text(self):
        width, height = 32, 16
        source = stripe_frame(width, height)
        measured = subtitle_metrics.measure_subtitle_region(
            source, source, source, np.zeros(source.shape, bool),
            mapping_shape(width, height), binocular_geometry(width, height),
            target_disparity_pct=0.0)
        self.assertEqual(measured, {
            "subtitle_region_authored_count": 0,
            "subtitle_region_support_count": 0,
        })

    def test_flat_authored_region_abstains_from_sharpness_only(self):
        width, height = 32, 16
        source = np.full((height, width), 0.5, dtype=np.float32)
        measured = subtitle_metrics.measure_subtitle_region(
            source, source, source, np.ones(source.shape, bool),
            mapping_shape(width, height), binocular_geometry(width, height),
            target_disparity_pct=0.0)
        self.assertIn("subtitle_target_disparity_rms_error_pct", measured)
        self.assertIn("subtitle_disparity_variance_pct2", measured)
        self.assertNotIn("subtitle_sharpness_preservation_pct", measured)

    def test_rejected_nan_correspondences_do_not_enter_dense_sampling(self):
        width, height = 32, 16
        source = stripe_frame(width, height)
        valid = np.ones(source.shape, dtype=bool)
        valid[:, :2] = False
        geometry = binocular_geometry(width, height, valid=valid)
        geometry["disparity"][~valid] = np.nan
        geometry["symmetry"][~valid] = np.nan
        measured = subtitle_metrics.measure_subtitle_region(
            source, source, source, np.ones(source.shape, bool),
            mapping_shape(width, height), geometry, target_disparity_pct=0.0)
        self.assertEqual(
            measured["subtitle_region_support_count"], int(np.count_nonzero(valid)))
        self.assertAlmostEqual(
            measured["subtitle_sharpness_preservation_pct"], 100.0, places=5)

    def test_evidence_requires_authenticated_metadata_and_support(self):
        spec = {"requires": "subtitle_region_support_count"}
        enough = {
            "subtitle_region_support_count":
                subtitle_metrics.MIN_SUBTITLE_REGION_SAMPLES,
        }
        self.assertEqual(
            sbsbench.metric_evidence_state("subtitle", spec, enough, {}),
            "unsupported")
        self.assertEqual(
            sbsbench.metric_evidence_state(
                "subtitle", spec, {}, {"required_gt_subtitle_region": True}),
            "missing")
        self.assertEqual(
            sbsbench.metric_evidence_state(
                "subtitle", spec, enough, {"required_gt_subtitle_region": True}),
            "applicable")

    def test_binocular_support_gate_rejects_tiny_surviving_fraction(self):
        width, height = 32, 16
        source = stripe_frame(width, height)
        valid = np.zeros(source.shape, dtype=bool)
        valid[0, :subtitle_metrics.MIN_SUBTITLE_REGION_SAMPLES] = True
        measured = subtitle_metrics.measure_subtitle_region(
            source, source, source, np.ones(source.shape, bool),
            mapping_shape(width, height), binocular_geometry(width, height, valid=valid),
            target_disparity_pct=0.0)
        self.assertEqual(measured["subtitle_region_authored_count"], width * height)
        self.assertEqual(
            measured["subtitle_region_support_count"],
            subtitle_metrics.MIN_SUBTITLE_REGION_SAMPLES)
        self.assertAlmostEqual(
            measured["subtitle_region_binocular_support_pct"],
            subtitle_metrics.MIN_SUBTITLE_REGION_SAMPLES / (width * height) * 100.0)

        with open(os.path.join(SCRIPT_DIR, "thresholds.json"), encoding="utf-8") as stream:
            spec = json.load(stream)["metrics"][
                "subtitle_region_binocular_support_pct"]
        _, _, failures = run_eval.score_clip_gates(
            [measured], measured,
            {"metrics": {"subtitle_region_binocular_support_pct": spec}},
            {"required_gt_subtitle_region": True})
        self.assertEqual(
            [failure["metric"] for failure in failures],
            ["subtitle_region_binocular_support_pct"])

    def test_binocular_support_gate_detects_output_area_collapse(self):
        width, height = 32, 16
        source = stripe_frame(width, height)
        collapsed_weight = np.full(source.shape, 0.01, dtype=np.float32)
        measured = subtitle_metrics.measure_subtitle_region(
            source, source, source, np.ones(source.shape, bool),
            mapping_shape(width, height), binocular_geometry(
                width, height, support_weight=collapsed_weight),
            target_disparity_pct=0.0)
        self.assertEqual(measured["subtitle_region_support_count"], width * height)
        self.assertAlmostEqual(
            measured["subtitle_region_binocular_support_pct"], 1.0, places=6)

    def test_target_disparity_validation_is_finite_numeric_and_bounded(self):
        for value in (True, "0", float("nan"), -3.5001, 3.0001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                subtitle_metrics.validate_subtitle_target_disparity_pct(value)
        for value in (-3.5, 0, 3.0):
            with self.subTest(value=value):
                self.assertEqual(
                    subtitle_metrics.validate_subtitle_target_disparity_pct(value),
                    float(value))

    def test_nonzero_target_error_fails_the_hard_gate(self):
        with open(os.path.join(SCRIPT_DIR, "thresholds.json"), encoding="utf-8") as stream:
            spec = json.load(stream)["metrics"][
                "subtitle_target_disparity_rms_error_pct"]
        thresholds = {"metrics": {"subtitle_target_disparity_rms_error_pct": spec}}
        row = {
            "subtitle_region_support_count": 64,
            "subtitle_target_disparity_rms_error_pct": spec["hard_max"] + 0.001,
        }
        _, _, failures = run_eval.score_clip_gates(
            [row], row, thresholds, {"required_gt_subtitle_region": True})
        self.assertEqual(
            [failure["metric"] for failure in failures],
            ["subtitle_target_disparity_rms_error_pct"])


class SubtitleSidecarContractTests(unittest.TestCase):
    @staticmethod
    def _write_source(path, value=0):
        Image.fromarray(np.full((16, 32, 3), value, dtype=np.uint8)).save(path)

    @staticmethod
    def _transition_contract():
        return {
            "kind": "highres-empty-appear-replace-disappear-hard-cut",
            "source_size_px": [32, 16],
            "detector_target_size_px": [16, 8],
            "authored_glyph_stroke_width_px": 3,
            "authored_outline_radius_px": 2,
            "outside_overlay_rgb_delta_threshold": 40,
            "subtitle_only_outside_overlay_max_changed_fraction": 0.05,
            "broad_scene_cut_outside_overlay_min_changed_fraction": 0.90,
            "empty_frame_ranges": [[1, 4], [13, 16]],
            "appear_frames": [5, 17],
            "subtitle_only_replacement_frames": [9],
            "disappear_frames": [13],
            "broad_scene_cut_frames": [21],
            "overlay_replacement_at_scene_cut_frames": [21],
            "subtitle_state_by_frame": (
                ["empty"] * 4 + ["cue-a"] * 4 + ["cue-b"] * 4 +
                ["empty"] * 4 + ["cue-c"] * 4 + ["cue-d"] * 4),
            "scene_state_by_frame": ["scene-a"] * 20 + ["scene-b"] * 4,
        }

    def _write_transition_clip(self, clip):
        region_dir = os.path.join(clip, "gt_subtitle_region")
        overlay_dir = os.path.join(clip, "gt_subtitle_overlay_mask")
        oracle_dir = os.path.join(clip, "gt_subtitle_free")
        for directory in (region_dir, overlay_dir, oracle_dir):
            os.makedirs(directory, exist_ok=True)
        contract = self._transition_contract()
        for frame_id, state in enumerate(contract["subtitle_state_by_frame"], 1):
            scene = contract["scene_state_by_frame"][frame_id - 1]
            oracle_value = 0 if scene == "scene-a" else 96
            oracle = np.full((16, 32, 3), oracle_value, dtype=np.uint8)
            source = oracle.copy()
            region = np.zeros((16, 32), dtype=np.uint8)
            overlay = np.zeros((16, 32), dtype=np.uint8)
            if state != "empty":
                region[4:14, 6:26] = 255
                overlay[6:12, 8:24] = 255
                source[overlay != 0] = (245, 245, 240)
            Image.fromarray(source).save(
                os.path.join(clip, f"frame_{frame_id:05d}.png"))
            Image.fromarray(region).save(
                os.path.join(region_dir, f"frame_{frame_id:05d}.png"))
            Image.fromarray(overlay).save(
                os.path.join(overlay_dir, f"frame_{frame_id:05d}.png"))
            Image.fromarray(oracle).save(
                os.path.join(oracle_dir, f"frame_{frame_id:05d}.png"))
        metadata = {
            "name": "subtitle-transition-probe",
            "content_type": "synthetic",
            "required_gt_subtitle_region": True,
            "required_gt_subtitle_sanitizer_oracle": True,
            "subtitle_target_disparity_pct": 0.0,
            "subtitle_transition_contract": contract,
        }
        with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as stream:
            json.dump(metadata, stream)
        return metadata

    def test_clip_identity_covers_region_pixels_and_requirement_semantics(self):
        with tempfile.TemporaryDirectory() as clip:
            self._write_source(os.path.join(clip, "frame_00000.png"))
            region_dir = os.path.join(clip, "gt_subtitle_region")
            os.makedirs(region_dir)
            region_path = os.path.join(region_dir, "frame_00000.png")
            Image.fromarray(np.zeros((16, 32), dtype=np.uint8)).save(region_path)
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as stream:
                json.dump({
                    "required_gt_subtitle_region": True,
                    "subtitle_target_disparity_pct": 0.0,
                }, stream)
            original = run_eval.source_evidence_digests(clip)
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as stream:
                json.dump({
                    "required_gt_subtitle_region": True,
                    "subtitle_target_disparity_pct": 0.5,
                }, stream)
            changed_target = run_eval.source_evidence_digests(clip)
            self.assertNotEqual(original, changed_target)
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as stream:
                json.dump({
                    "required_gt_subtitle_region": True,
                    "subtitle_target_disparity_pct": 0.0,
                }, stream)
            Image.fromarray(np.full((16, 32), 255, dtype=np.uint8)).save(region_path)
            changed_pixels = run_eval.source_evidence_digests(clip)
            self.assertNotEqual(original, changed_pixels)
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as stream:
                json.dump({
                    "required_gt_subtitle_region": False,
                    "subtitle_target_disparity_pct": 0.0,
                }, stream)
            self.assertNotEqual(changed_pixels, run_eval.source_evidence_digests(clip))

    def test_published_metadata_preserves_subtitle_requirement(self):
        published = run_eval.published_clip_metadata({
            "name": "subtitle",
            "required_gt_subtitle_region": True,
            "subtitle_target_disparity_pct": 0.0,
        })
        self.assertIs(published["required_gt_subtitle_region"], True)
        self.assertEqual(published["subtitle_target_disparity_pct"], 0.0)

    def test_transition_contract_is_validated_published_and_hashed(self):
        with tempfile.TemporaryDirectory() as clip:
            metadata = self._write_transition_clip(clip)
            loaded = run_eval.load_clip_metadata(clip)
            self.assertEqual(
                loaded["subtitle_transition_contract"],
                metadata["subtitle_transition_contract"])
            published = run_eval.published_clip_metadata(loaded)
            self.assertEqual(
                published["subtitle_transition_contract"],
                metadata["subtitle_transition_contract"])
            self.assertIs(
                published["required_gt_subtitle_sanitizer_oracle"], True)
            original = run_eval.source_evidence_digests(clip)
            metadata["subtitle_transition_contract"]["detector_target_size_px"] = [15, 8]
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as stream:
                json.dump(metadata, stream)
            self.assertNotEqual(original, run_eval.source_evidence_digests(clip))

    def test_sanitizer_oracle_pixels_and_mask_are_part_of_clip_identity(self):
        with tempfile.TemporaryDirectory() as clip:
            self._write_transition_clip(clip)
            original = run_eval.source_evidence_digests(clip)

            # Add one valid authored difference inside the loose region but outside the prior
            # tight support. Source pixels stay unchanged; only the tight mask and clean oracle
            # change, so the identity change cannot be attributed to the rendered source.
            frame_id = 5
            overlay_path = os.path.join(
                clip, "gt_subtitle_overlay_mask", f"frame_{frame_id:05d}.png")
            oracle_path = os.path.join(
                clip, "gt_subtitle_free", f"frame_{frame_id:05d}.png")
            overlay = np.asarray(Image.open(overlay_path)).copy()
            oracle = np.asarray(Image.open(oracle_path)).copy()
            overlay[4, 6] = 255
            oracle[4, 6] = (1, 1, 1)
            Image.fromarray(overlay).save(overlay_path)
            Image.fromarray(oracle).save(oracle_path)

            changed = run_eval.source_evidence_digests(clip)
            self.assertNotEqual(original, changed)

    def test_sanitizer_oracle_rejects_missing_or_inconsistent_sidecars(self):
        with tempfile.TemporaryDirectory() as clip:
            self._write_transition_clip(clip)
            os.remove(os.path.join(
                clip, "gt_subtitle_overlay_mask", "frame_00005.png"))
            with self.assertRaisesRegex(ValueError, "frame-id mismatch"):
                run_eval.load_clip_metadata(clip)

            self._write_transition_clip(clip)
            source_path = os.path.join(clip, "frame_00005.png")
            source = np.asarray(Image.open(source_path)).copy()
            source[0, 0] = (1, 1, 1)
            Image.fromarray(source).save(source_path)
            with self.assertRaisesRegex(ValueError, "difference support"):
                run_eval.load_clip_metadata(clip)

            self._write_transition_clip(clip)
            region_path = os.path.join(
                clip, "gt_subtitle_region", "frame_00005.png")
            region = np.asarray(Image.open(region_path)).copy()
            region[6, 8] = 0
            Image.fromarray(region).save(region_path)
            with self.assertRaisesRegex(ValueError, "escapes loose subtitle region"):
                run_eval.load_clip_metadata(clip)

    def test_sanitizer_oracle_requires_rgb_and_retains_real_broad_cut(self):
        with tempfile.TemporaryDirectory() as clip:
            self._write_transition_clip(clip)
            oracle_path = os.path.join(
                clip, "gt_subtitle_free", "frame_00005.png")
            Image.fromarray(np.zeros((16, 32), dtype=np.uint8)).save(oracle_path)
            with self.assertRaisesRegex(ValueError, "must be an RGB PNG"):
                run_eval.load_clip_metadata(clip)

            self._write_transition_clip(clip)
            appear_oracle_path = os.path.join(
                clip, "gt_subtitle_free", "frame_00005.png")
            appear_source_path = os.path.join(clip, "frame_00005.png")
            appear_oracle = np.full((16, 32, 3), 96, dtype=np.uint8)
            appear_overlay = np.asarray(Image.open(os.path.join(
                clip, "gt_subtitle_overlay_mask", "frame_00005.png"))) != 0
            appear_source = appear_oracle.copy()
            appear_source[appear_overlay] = (245, 245, 240)
            Image.fromarray(appear_oracle).save(appear_oracle_path)
            Image.fromarray(appear_source).save(appear_source_path)
            with self.assertRaisesRegex(ValueError, "subtitle-only transition"):
                run_eval.load_clip_metadata(clip)

            self._write_transition_clip(clip)
            before_oracle_path = os.path.join(
                clip, "gt_subtitle_free", "frame_00020.png")
            cut_oracle_path = os.path.join(
                clip, "gt_subtitle_free", "frame_00021.png")
            cut_source_path = os.path.join(clip, "frame_00021.png")
            before_oracle = np.asarray(Image.open(before_oracle_path)).copy()
            overlay = np.asarray(Image.open(os.path.join(
                clip, "gt_subtitle_overlay_mask", "frame_00021.png"))) != 0
            cut_source = before_oracle.copy()
            cut_source[overlay] = (245, 245, 240)
            Image.fromarray(before_oracle).save(cut_oracle_path)
            Image.fromarray(cut_source).save(cut_source_path)
            with self.assertRaisesRegex(ValueError, "does not retain a broad scene cut"):
                run_eval.load_clip_metadata(clip)

    def test_authenticated_discovery_requires_exact_sanitizer_sidecar_id_parity(self):
        with tempfile.TemporaryDirectory() as root:
            clip = os.path.join(root, "subtitle-probe")
            os.makedirs(clip)
            self._write_transition_clip(clip)
            discovered = authenticated_metric_sources.discover_clips([root])
            self.assertEqual([item["id"] for item in discovered], ["subtitle-probe"])

            os.remove(os.path.join(
                clip, "gt_subtitle_free", "frame_00005.png"))
            with self.assertRaisesRegex(ValueError, "frame-id mismatch"):
                authenticated_metric_sources.discover_clips([root])

            self._write_transition_clip(clip)
            Image.fromarray(np.zeros((16, 32), dtype=np.uint8)).save(os.path.join(
                clip, "gt_subtitle_overlay_mask", "frame_00025.png"))
            with self.assertRaisesRegex(ValueError, r"extra=\[25\]"):
                authenticated_metric_sources.discover_clips([root])

    def test_sanitizer_oracle_requirement_is_boolean_and_requires_region_contract(self):
        with tempfile.TemporaryDirectory() as clip:
            self._write_source(os.path.join(clip, "frame_00001.png"))
            meta_path = os.path.join(clip, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as stream:
                json.dump({
                    "content_type": "synthetic",
                    "required_gt_subtitle_sanitizer_oracle": "yes",
                }, stream)
            with self.assertRaisesRegex(ValueError, "must be boolean"):
                run_eval.load_clip_metadata(clip)

            with open(meta_path, "w", encoding="utf-8") as stream:
                json.dump({
                    "content_type": "synthetic",
                    "required_gt_subtitle_sanitizer_oracle": True,
                }, stream)
            with self.assertRaisesRegex(
                    ValueError, "requires required_gt_subtitle_region=true"):
                run_eval.load_clip_metadata(clip)

    def test_transition_contract_rejects_schedule_or_mask_disagreement(self):
        with tempfile.TemporaryDirectory() as clip:
            metadata = self._write_transition_clip(clip)
            metadata["subtitle_transition_contract"]["appear_frames"] = [6, 17]
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as stream:
                json.dump(metadata, stream)
            with self.assertRaisesRegex(ValueError, "event lists disagree"):
                run_eval.load_clip_metadata(clip)

            metadata = self._write_transition_clip(clip)
            Image.fromarray(np.zeros((16, 32), dtype=np.uint8)).save(
                os.path.join(clip, "gt_subtitle_region", "frame_00005.png"))
            with self.assertRaisesRegex(ValueError, "state disagrees with authored mask"):
                run_eval.load_clip_metadata(clip)

            metadata = self._write_transition_clip(clip)
            metadata["subtitle_transition_contract"][
                "subtitle_only_outside_overlay_max_changed_fraction"] = 0.95
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as stream:
                json.dump(metadata, stream)
            with self.assertRaisesRegex(ValueError, "must be below broad-cut minimum"):
                run_eval.load_clip_metadata(clip)

    def test_load_metadata_rejects_partial_region_frame_set(self):
        with tempfile.TemporaryDirectory() as clip:
            self._write_source(os.path.join(clip, "frame_00000.png"))
            self._write_source(os.path.join(clip, "frame_00001.png"))
            region_dir = os.path.join(clip, "gt_subtitle_region")
            os.makedirs(region_dir)
            Image.fromarray(np.zeros((16, 32), dtype=np.uint8)).save(
                os.path.join(region_dir, "frame_00000.png"))
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as stream:
                json.dump({
                    "content_type": "synthetic",
                    "required_gt_subtitle_region": True,
                    "subtitle_target_disparity_pct": 0.0,
                }, stream)
            with self.assertRaisesRegex(ValueError, "frame-id mismatch"):
                run_eval.load_clip_metadata(clip)

    def test_binary_mask_loader_rejects_gray_values_and_wrong_dimensions(self):
        with tempfile.TemporaryDirectory() as root:
            gray = os.path.join(root, "gray.png")
            wrong = os.path.join(root, "wrong.png")
            Image.fromarray(np.full((16, 32), 128, dtype=np.uint8)).save(gray)
            Image.fromarray(np.zeros((8, 32), dtype=np.uint8)).save(wrong)
            with self.assertRaisesRegex(ValueError, "other than 0/255"):
                sbsbench.load_binary_source_mask(gray, (16, 32))
            with self.assertRaisesRegex(ValueError, "does not match source"):
                sbsbench.load_binary_source_mask(wrong, (16, 32))

    def test_binary_mask_loader_rejects_rgb_and_16bit_png_modes(self):
        with tempfile.TemporaryDirectory() as root:
            rgb = os.path.join(root, "rgb.png")
            gray16 = os.path.join(root, "gray16.png")
            Image.fromarray(np.zeros((16, 32, 3), dtype=np.uint8)).save(rgb)
            Image.fromarray(np.zeros((16, 32), dtype=np.uint16)).save(gray16)
            for path in (rgb, gray16):
                with self.subTest(path=path), self.assertRaisesRegex(
                        ValueError, "8-bit single-channel"):
                    sbsbench.load_binary_source_mask(path, (16, 32))

    def test_required_metadata_rejects_missing_or_invalid_target(self):
        with tempfile.TemporaryDirectory() as clip:
            self._write_source(os.path.join(clip, "frame_00000.png"))
            region_dir = os.path.join(clip, "gt_subtitle_region")
            os.makedirs(region_dir)
            Image.fromarray(np.zeros((16, 32), dtype=np.uint8)).save(
                os.path.join(region_dir, "frame_00000.png"))
            meta_path = os.path.join(clip, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as stream:
                json.dump({
                    "content_type": "synthetic",
                    "required_gt_subtitle_region": True,
                }, stream)
            with self.assertRaisesRegex(ValueError, "needs explicit"):
                run_eval.load_clip_metadata(clip)
            for target in (True, "0", -3.5001, 3.0001):
                with self.subTest(target=target):
                    with open(meta_path, "w", encoding="utf-8") as stream:
                        json.dump({
                            "content_type": "synthetic",
                            "required_gt_subtitle_region": True,
                            "subtitle_target_disparity_pct": target,
                        }, stream)
                    with self.assertRaisesRegex(
                            ValueError, "subtitle_target_disparity_pct"):
                        run_eval.load_clip_metadata(clip)

    def test_metadata_preflight_rejects_nonbinary_region(self):
        with tempfile.TemporaryDirectory() as clip:
            self._write_source(os.path.join(clip, "frame_00000.png"))
            region_dir = os.path.join(clip, "gt_subtitle_region")
            os.makedirs(region_dir)
            Image.fromarray(np.full((16, 32), 128, dtype=np.uint8)).save(
                os.path.join(region_dir, "frame_00000.png"))
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as stream:
                json.dump({
                    "content_type": "synthetic",
                    "required_gt_subtitle_region": True,
                    "subtitle_target_disparity_pct": 0.0,
                }, stream)
            with self.assertRaisesRegex(ValueError, "other than 0/255"):
                run_eval.load_clip_metadata(clip)

    def test_required_clip_allows_empty_frames_when_one_frame_has_evidence(self):
        rows = [
            {
                "subtitle_region_authored_count": 0,
                "subtitle_region_support_count": 0,
            },
            {
                "subtitle_region_authored_count": 64,
                "subtitle_region_support_count": 64,
                "subtitle_region_binocular_support_pct": 100.0,
                "subtitle_target_disparity_rms_error_pct": 0.0,
                "subtitle_disparity_variance_pct2": 0.0,
                "subtitle_sharpness_preservation_pct": 100.0,
            },
        ]
        aggregate = sbsbench.aggregate(rows)
        specs = {
            "subtitle_region_binocular_support_pct": {
                "requires": "subtitle_region_authored_count"},
            "subtitle_target_disparity_rms_error_pct": {
                "requires": "subtitle_region_support_count"},
            "subtitle_disparity_variance_pct2": {
                "requires": "subtitle_region_support_count"},
            "subtitle_sharpness_preservation_pct": {
                "requires": "subtitle_region_support_count"},
        }
        filtered = sbsbench.filter_aggregate_by_evidence(
            rows, aggregate, specs, {"required_gt_subtitle_region": True})
        self.assertEqual(filtered["subtitle_region_authored_count"], 64.0)
        self.assertEqual(filtered["subtitle_region_support_count"], 64.0)
        self.assertEqual(filtered["subtitle_region_binocular_support_pct"], 100.0)
        self.assertEqual(filtered["subtitle_target_disparity_rms_error_pct"], 0.0)
        self.assertEqual(filtered["subtitle_disparity_variance_pct2"], 0.0)
        self.assertEqual(filtered["subtitle_sharpness_preservation_pct"], 100.0)

    def test_measure_sequence_consumes_explicit_region_sidecar(self):
        with tempfile.TemporaryDirectory() as root:
            frames = os.path.join(root, "frames")
            sequence = os.path.join(root, "sequence")
            region_dir = os.path.join(frames, "gt_subtitle_region")
            os.makedirs(region_dir)
            os.makedirs(sequence)
            width, height = 32, 16
            source_luma = (stripe_frame(width, height) * 255.0).astype(np.uint8)
            source_rgb = np.repeat(source_luma[..., None], 3, axis=2)
            Image.fromarray(source_rgb).save(os.path.join(frames, "frame_00000.png"))
            Image.fromarray(np.concatenate((source_rgb, source_rgb), axis=1)).save(
                os.path.join(sequence, "sbs_00000.png"))
            Image.fromarray(np.full((height, width), 255, dtype=np.uint8)).save(
                os.path.join(region_dir, "frame_00000.png"))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as stream:
                json.dump({
                    "required_gt_subtitle_region": True,
                    "subtitle_target_disparity_pct": 0.0,
                }, stream)
            source_u = (np.arange(width, dtype=np.float32) + 0.5) / width
            eye_map = np.broadcast_to(source_u, (height, width))
            np.concatenate((eye_map, eye_map), axis=1).astype("<f4").tofile(
                os.path.join(sequence, "warp_map_00000.f32"))
            with open(os.path.join(sequence, "warp_map_shape.json"), "w",
                      encoding="utf-8") as stream:
                json.dump(mapping_shape(width, height), stream)

            rows, aggregate = sbsbench.measure_sequence(sequence, frames)
            self.assertEqual(rows[0]["subtitle_region_authored_count"], width * height)
            self.assertEqual(rows[0]["subtitle_region_support_count"], width * height)
            self.assertGreaterEqual(
                aggregate["subtitle_region_binocular_support_pct"], 95.0)
            self.assertEqual(
                aggregate["subtitle_target_disparity_rms_error_pct"], 0.0)
            self.assertAlmostEqual(
                aggregate["subtitle_disparity_variance_pct2"], 0.0, places=7)
            self.assertAlmostEqual(
                aggregate["subtitle_sharpness_preservation_pct"], 100.0, places=4)


if __name__ == "__main__":
    unittest.main()
