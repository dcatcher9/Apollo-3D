import io
import os
import sys
import tarfile
import tempfile
import argparse
import unittest
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_depth_transform  # noqa: E402
import prepare_public_datasets  # noqa: E402
import run_eval  # noqa: E402
import sbsbench  # noqa: E402


class EvalContractTests(unittest.TestCase):
    @staticmethod
    def png_bytes(value=64, mode="RGB"):
        shape = (8, 12, 3) if mode == "RGB" else (8, 12)
        array = np.full(shape, value, np.uint8)
        stream = io.BytesIO()
        Image.fromarray(array, mode=mode).save(stream, "PNG")
        return stream.getvalue()

    def test_warp_override_uses_last_explicit_value(self):
        self.assertEqual(run_eval.extra_value(
            ["--warp", "apollo", "--warp", "vd3d"], "--warp", "apollo"), "vd3d")

    def test_warp_is_read_from_config(self):
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            fh.write("# sbs_3d_warp = apollo\nsbs_3d_warp = vd3d # active\n")
            path = fh.name
        try:
            self.assertEqual(run_eval.conf_value(path, "sbs_3d_warp", "apollo"), "vd3d")
        finally:
            os.unlink(path)

    def test_named_profiles_and_explicit_overrides_share_production_precedence(self):
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            fh.write("sbs_3d_profile = vd3d\nsbs_3d_warp = apollo\n")
            path = fh.name
        try:
            self.assertEqual(run_eval.expected_profile(path, []), ("vd3d", "apollo"))
            self.assertEqual(run_eval.expected_profile(
                path, ["--warp", "vd3d"]), ("vd3d", "vd3d"))
        finally:
            os.unlink(path)

    def test_custom_profile_values_need_no_evaluator_code_change(self):
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            fh.write("sbs_3d_profile = cinema\n"
                     "sbs_3d_profile_cinema_warp = vd3d\n"
                     "sbs_3d_profile_cinema_depth_model = da3mono_large_fp16\n")
            path = fh.name
        try:
            self.assertEqual(run_eval.expected_profile(path, []), ("cinema", "vd3d"))
            self.assertEqual(run_eval.expected_profile(
                path, ["--warp", "apollo"]), ("cinema", "apollo"))
            self.assertEqual(run_eval.expected_depth_model(path, "cinema"),
                             "da3mono_large_fp16")
        finally:
            os.unlink(path)

    def test_live_sbs_contract_is_off_ai_and_profile_owned(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "video.h"), encoding="utf-8") as fh:
            video_header = fh.read()
        self.assertIn("SBS_AI = 1", video_header)
        self.assertNotIn("SBS_GAME", video_header)
        self.assertNotIn("SBS_MOVIE", video_header)

        with open(os.path.join(repo, "src", "config.cpp"), encoding="utf-8") as fh:
            config = fh.read()
        self.assertIn('apply_sbs_values(profile, "sbs_3d_profile_" + name + "_")', config)
        self.assertIn("video.sbs_profiles", config)

        with open(os.path.join(repo, "src", "stream.cpp"), encoding="utf-8") as fh:
            stream = fh.read()
        self.assertIn("IDX_SET_SBS_PROFILE", stream)
        self.assertIn("IDX_SBS_PROFILE_LIST", stream)
        self.assertIn("mail::sbs_depth_status", stream)
        self.assertNotIn("depth_engine_phase", stream)
        self.assertNotIn("set_active_depth_model(id)", stream)

    def test_relative_cli_paths_are_resolved_before_subprocess_cwd(self):
        args = argparse.Namespace(build_dir="cmake-build-relwithdebinfo", conf="bench.conf",
                                  clips_root=None, baseline_dir=None,
                                  report_control=None, report_out=None)
        run_eval.normalize_cli_paths(args)
        self.assertTrue(os.path.isabs(args.build_dir))
        self.assertTrue(os.path.isabs(args.conf))

    def test_apollo_bestv2_normalizes_pixel_shifts_by_source_width(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "sbs_reprojection_ps.hlsl")
        with open(shader, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("LeftColorTexture.GetDimensions(sourceWidth, sourceHeight)", text)
        self.assertIn("Bestv2SearchRadius((float)sourceWidth)", text)
        self.assertGreaterEqual(text.count("shaped, (float)sourceWidth)"), 2)
        self.assertNotIn("Bestv2SearchRadius((float)dw)", text)

    def test_vd3d_bestv2_normalizes_pixel_shifts_by_source_width(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx")
        for name in ("sbs_vd3d_forward_cs.hlsl", "sbs_vd3d_reprojection_ps.hlsl"):
            with self.subTest(shader=name), open(os.path.join(shader_dir, name), encoding="utf-8") as fh:
                text = fh.read()
                self.assertIn("LeftColorTexture.GetDimensions(source_w, source_h)", text)
                self.assertIn("shaped, (float)source_w)", text)
                self.assertNotIn("shaped, (float)eye_w)", text)

    def test_bestv2_scales_wide_sources_from_validated_calibration_width(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        common = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "include", "sbs_warp_common.hlsl")
        with open(common, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("BESTV2_CALIBRATION_WIDTH = 854.0f", text)
        self.assertIn("return min(max(source_width, 1.0f), BESTV2_CALIBRATION_WIDTH)", text)
        self.assertGreaterEqual(text.count("/ parallax_width"), 3)

    def test_pop_strength_scales_shared_parallax_and_apollo_probe_radius(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        common = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "include", "sbs_warp_common.hlsl")
        with open(common, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("float pop_strength;", text)
        self.assertIn("clamp(parallax * pop_strength", text)
        self.assertIn("return pop_strength *", text)

        with open(os.path.join(repo, "src", "config.cpp"), encoding="utf-8") as fh:
            config = fh.read()
        self.assertIn('prefix + "pop_strength", target.pop_strength, {0.25, 2.0}', config)
        with open(os.path.join(repo, "src", "config.h"), encoding="utf-8") as fh:
            config_header = fh.read()
        self.assertIn("double pop_strength = 1.25;", config_header)

        with open(os.path.join(repo, "src", "platform", "windows", "display_vram.cpp"),
                  encoding="utf-8") as fh:
            production = fh.read()
        self.assertIn("(float) sbs_config.pop_strength", production)

    def test_vd3d_fill_radius_scales_from_source_to_output_pixels(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "sbs_vd3d_reprojection_ps.hlsl")
        with open(shader, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("96.0f * source_to_output", text)

    def test_bestv2_sharpen_scales_taps_from_source_to_output_pixels(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader = os.path.join(repo, "src_assets", "windows", "assets",
                              "shaders", "directx", "sbs_sharpen_ps.hlsl")
        with open(shader, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("float tap = max(source_to_output", text)
        self.assertIn("EyeSample((float)x - tap", text)

    def test_warps_apply_per_eye_aspect_mapping(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx")
        for name in ("sbs_reprojection_ps.hlsl", "sbs_vd3d_forward_cs.hlsl",
                     "sbs_vd3d_reprojection_ps.hlsl", "sbs_sharpen_ps.hlsl"):
            with self.subTest(shader=name), open(os.path.join(shader_dir, name), encoding="utf-8") as fh:
                self.assertIn("ContentToSourceUV", fh.read())

    def test_hdr_depth_input_uses_validated_color_transform(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx")
        with open(os.path.join(shader_dir, "include", "depth_color.hlsl"), encoding="utf-8") as fh:
            color = fh.read()
        self.assertIn("DepthHdrScRgbToSrgb", color)
        self.assertIn("dot(c, float3(0.2126f, 0.7152f, 0.0722f))", color)
        self.assertNotIn("c / (1.0f + c)", color)
        for name in ("rgb_to_nchw_cs.hlsl",):
            with self.subTest(shader=name), open(os.path.join(shader_dir, name), encoding="utf-8") as fh:
                text = fh.read()
                self.assertIn('include/depth_color.hlsl', text)
                self.assertIn("DepthColorToSrgb", text)

    def test_hdr_warp_stays_linear_fp16_until_pq_conversion(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        display = os.path.join(repo, "src", "platform", "windows", "display_vram.cpp")
        with open(display, encoding="utf-8") as fh:
            pipeline = fh.read()
        self.assertIn("tex_desc.Format = sbs_intermediate_linear ? DXGI_FORMAT_R16G16B16A16_FLOAT", pipeline)
        self.assertIn("if (!sbs_intermediate_linear && sbs_config.bestv2_sharpen)", pipeline)
        self.assertIn("input_is_linear ? convert_Y_or_YUV_fp16_ps.get()", pipeline)
        self.assertIn("models::input_color_space::linear_sdr", pipeline)
        common = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "include", "common.hlsl")
        with open(common, encoding="utf-8") as fh:
            color = fh.read()
        self.assertIn("rgb = Rec709toRec2020(rgb)", color)
        self.assertIn("rgb *= 80", color)
        self.assertIn("return NitsToPQ(rgb)", color)

    def test_hdr_debug_preview_preserves_hue(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(repo, "src", "platform", "windows", "sbs_debug_dump.cpp")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("const float luminance = std::max(0.2126f * r + 0.7152f * g", text)
        self.assertIn("const float tone_scale = 1.0f / (1.0f + luminance)", text)
        self.assertNotIn("c = c / (1.0f + c)", text)

    def test_report_reuses_one_aggregate_decision_and_writes_sidecar(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(repo, "tools", "sbsbench", "build_report.py")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("AB_DECISION = sbsbench.evaluate_ab_decision(\n    ctrl_agg, treat_agg", text)
        self.assertIn("decision = AB_DECISION", text)
        self.assertIn('"decision_clips": DECISION_CLIPS', text)
        self.assertIn('"decision_scope": DECISION_SCOPE', text)
        self.assertIn('"source_artifact_clips": SOURCE_ARTIFACT_CLIPS', text)
        self.assertIn('AB_DECISION["verdict"]', text)

    def test_depth_transform_audit_preserves_16bit_precision(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "depth.png")
            values = np.linspace(0, 65535, 100, dtype=np.uint16).reshape(10, 10)
            Image.fromarray(values).save(path)
            stats = audit_depth_transform.frame_stats(path)
        self.assertAlmostEqual(stats["spread_p95_p05"], 0.9, delta=0.02)
        self.assertGreater(stats["saturated_low_pct"], 0.0)
        self.assertGreater(stats["saturated_high_pct"], 0.0)

    def test_depth_transform_audit_uses_image_mode_for_dark_16bit_png(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            path = fh.name
        try:
            values = np.linspace(0, 255, 100, dtype=np.uint16).reshape(10, 10)
            Image.fromarray(values).save(path)
            stats = audit_depth_transform.frame_stats(path)
            self.assertLess(stats["p99"], 0.005)
            self.assertLess(stats["spread_p95_p05"], 0.005)
        finally:
            os.unlink(path)

    def test_expected_flat_exemption_is_derived_from_stereo_axis(self):
        flat = {"expected_flat": True}
        self.assertTrue(run_eval.metric_exempt_for_clip({"axis": "stereo"}, flat))
        self.assertFalse(run_eval.metric_exempt_for_clip({"axis": "comfort"}, flat))
        self.assertFalse(run_eval.metric_exempt_for_clip({"axis": "stereo"}, {}))

    def test_rejected_processors_and_ema_order_are_permanently_removed(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx")
        for name in ("depth_guided_upsample_cs.hlsl", "depth_guide_downsample_cs.hlsl",
                     "depth_curvature_cs.hlsl", os.path.join("include", "band_curve.hlsl")):
            self.assertFalse(os.path.exists(os.path.join(shader_dir, name)))
        with open(os.path.join(repo, "src", "config.cpp"), encoding="utf-8") as fh:
            config = fh.read()
        for key in ("sbs_3d_ema_pixel_first", "sbs_3d_guided_upsample",
                    "sbs_3d_foreground_curvature", "sbs_3d_minmax_snap",
                    "sbs_3d_range_floor", "sbs_3d_shift_profile",
                    "sbs_3d_subject_track"):
            self.assertNotIn(key, config)

    def test_bestv2_subject_pipeline_is_mandatory_and_validated(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        paths = {
            "estimator": os.path.join(repo, "src", "video_depth_estimator.cpp"),
            "display": os.path.join(repo, "src", "platform", "windows", "display_vram.cpp"),
            "harness": os.path.join(repo, "src", "sbs_bench_harness.cpp"),
        }
        text = {}
        for key, path in paths.items():
            with open(path, encoding="utf-8") as fh:
                text[key] = fh.read()
        self.assertIn("const bool core_shaders_ok", text["estimator"])
        self.assertIn("if (!valid || !input_srv)", text["estimator"])
        self.assertIn("depth_estimator->is_valid()", text["display"])
        self.assertNotIn("--no-subject-track", text["harness"])
        self.assertNotIn("sbs_cfg.subject_track", text["harness"])

    def test_bestv2_plane_reduce_reuses_shared_curve_and_stretch_switch(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx")
        with open(os.path.join(shader_dir, "depth_plane_reduce_cs.hlsl"), encoding="utf-8") as fh:
            reduce_shader = fh.read()
        self.assertIn('include "include/bestv2_curve.hlsl"', reduce_shader)
        self.assertIn("plane_subject_stretch != 0u", reduce_shader)
        self.assertNotIn("Bestv2RawShiftPxPlane", reduce_shader)

    def test_disabled_plane_lock_has_no_probe_loop_texture_fetch(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "sbs_reprojection_ps.hlsl")
        with open(shader, encoding="utf-8") as fh:
            text = fh.read()
        self.assertEqual(text.count("PlaneLockTexture.SampleLevel"), 1)
        self.assertIn("if (subject_plane_lock > 0.0f)", text)

    def test_vd3d_live_forward_splat_is_cached_by_geometry_update(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        display = os.path.join(repo, "src", "platform", "windows", "display_vram.cpp")
        estimator = os.path.join(repo, "src", "video_depth_estimator.cpp")
        with open(display, encoding="utf-8") as fh:
            display_text = fh.read()
        with open(estimator, encoding="utf-8") as fh:
            estimator_text = fh.read()
        self.assertIn("est.geometry_updated || !sbs_vd3d_winner_valid", display_text)
        self.assertIn("geometry_updated = true", estimator_text)

    def test_report_evidence_is_bounded_and_accepts_zero_based_frames(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        report = os.path.join(repo, "tools", "sbsbench", "build_report.py")
        with open(report, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("match_scale = min(1.0, 256.0 / ew)", text)
        self.assertIn("if prev_idx < 0:", text)
        self.assertNotIn("if prev_idx < 1:", text)

    def test_phase_shift_recovers_known_translation(self):
        rng = np.random.default_rng(1234)
        a = rng.random((64, 64))
        b = np.roll(a, shift=(2, -5), axis=(0, 1))
        dy, dx = sbsbench.phase_shift(a, b)
        self.assertAlmostEqual(dy, -2.0, places=5)
        self.assertAlmostEqual(dx, 5.0, places=5)

    def test_disparity_field_covers_tile_sized_frame_and_final_borders(self):
        rng = np.random.default_rng(2026)
        left = rng.random((192, 320), dtype=np.float32)
        right = np.roll(left, 3, axis=1)
        field = sbsbench.disparity_field(left, right, tile=192, stride=128)
        self.assertIsNotNone(field)
        self.assertEqual(len(field[0]), 2)  # x=0 and the border-aligned x=128 tile
        self.assertEqual(sbsbench._tile_positions(320, 192, 128), [0, 128])

    def test_sequence_joins_by_frame_identity(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            sbs = np.zeros((16, 32, 3), dtype=np.uint8)
            src = np.zeros((16, 16, 3), dtype=np.uint8)
            Image.fromarray(sbs).save(os.path.join(seq, "sbs_00007.png"))
            Image.fromarray(src).save(os.path.join(frames, "frame_00007.png"))
            rows, agg = sbsbench.measure_sequence(seq, frames)
            self.assertEqual(rows[0]["_frame_id"], 7)
            self.assertEqual(agg["_n"], 1)

    def test_sequence_rejects_positional_mispairing(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            blank = np.zeros((16, 32, 3), dtype=np.uint8)
            Image.fromarray(blank).save(os.path.join(seq, "sbs_00008.png"))
            Image.fromarray(blank[:, :16]).save(os.path.join(frames, "frame_00007.png"))
            with self.assertRaisesRegex(ValueError, "frame-id mismatch"):
                sbsbench.measure_sequence(seq, frames)

    def test_public_clip_rejects_missing_required_ground_truth(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            Image.fromarray(np.zeros((16, 32, 3), np.uint8)).save(
                os.path.join(seq, "sbs_00000.png"))
            Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(
                os.path.join(frames, "frame_00000.png"))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as fh:
                fh.write('{"dataset":"Example Public Dataset","required_gt_depth":true}')
            with self.assertRaisesRegex(ValueError, "requires GT depth"):
                sbsbench.measure_sequence(seq, frames)

    def test_public_clip_rejects_missing_required_optical_flow(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            for frame_id in range(2):
                Image.fromarray(np.zeros((16, 32, 3), np.uint8)).save(
                    os.path.join(seq, f"sbs_{frame_id:05d}.png"))
                Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(
                    os.path.join(frames, f"frame_{frame_id:05d}.png"))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as fh:
                fh.write('{"required_gt_flow":true}')
            with self.assertRaisesRegex(ValueError, "requires GT optical flow"):
                sbsbench.measure_sequence(seq, frames)

    def test_duplicate_numeric_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(os.path.join(root, "frame_1.png"))
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(os.path.join(root, "frame_01.jpg"))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                sbsbench.indexed_files(os.path.join(root, "frame_*.*"), "frame_")

    def test_disocclusion_ratio_requires_minimum_support(self):
        eye = np.zeros((64, 64), dtype=np.float32)
        depth = np.full((16, 16), 0.5, dtype=np.float32)
        frac, smear = sbsbench.disocclusion_metrics(eye, depth)
        self.assertLess(frac, sbsbench.MIN_DISOCC_FRAC)
        self.assertIsNone(smear)

    def test_depth_is_diagnostic_not_part_of_artifact_score(self):
        clean = {"pop_spread_pct": 0.0}
        false_stereo = {"pop_spread_pct": 0.2}
        self.assertGreater(
            sbsbench.sbs_score(clean, expected_flat=True)["q_depth"],
            sbsbench.sbs_score(false_stereo, expected_flat=True)["q_depth"])
        self.assertLess(
            sbsbench.sbs_score(clean)["q_depth"],
            sbsbench.sbs_score(false_stereo)["q_depth"])
        self.assertEqual(sbsbench.sbs_score(clean)["score"],
                         sbsbench.sbs_score(false_stereo)["score"])

    def test_metric_delta_class_uses_gate_tolerance_and_direction(self):
        lower = {"better": "lower", "rel_tol": 0.25, "abs_floor": 0.5}
        self.assertEqual(sbsbench.metric_delta_class(2.0, 2.4, lower), "noise")
        self.assertEqual(sbsbench.metric_delta_class(2.0, 2.6, lower), "regressed")
        self.assertEqual(sbsbench.metric_delta_class(2.0, 1.4, lower), "improved")

    def test_metric_roles_control_committed_gate(self):
        diagnostic = {"role": "diagnostic", "better": "lower",
                      "rel_tol": 0.0, "abs_floor": 0.1}
        hard = {"role": "hard", "better": "lower", "hard_max": 0.5,
                "rel_tol": 0.0, "abs_floor": 0.1}
        self.assertFalse(sbsbench.metric_gate_failed(0.0, 99.0, diagnostic))
        self.assertFalse(sbsbench.metric_gate_failed(0.0, 0.49, hard))
        self.assertTrue(sbsbench.metric_gate_failed(0.0, 0.51, hard))
        hard_min = {"role": "hard", "better": "higher", "hard_min": 90.0,
                    "rel_tol": 0.0, "abs_floor": 1.0}
        self.assertFalse(sbsbench.metric_gate_failed(95.0, 91.0, hard_min))
        self.assertTrue(sbsbench.metric_gate_failed(95.0, 89.0, hard_min))

    def test_ab_decision_preserves_primary_axis_tradeoff(self):
        specs = {
            "pop": {"role": "primary", "axis": "stereo", "better": "higher",
                    "rel_tol": 0.0, "abs_floor": 0.5},
            "halo": {"role": "primary", "axis": "warp", "better": "lower",
                     "rel_tol": 0.0, "abs_floor": 0.5},
            "legacy_proxy": {"role": "diagnostic", "axis": "warp", "better": "lower",
                             "rel_tol": 0.0, "abs_floor": 0.1},
        }
        result = sbsbench.evaluate_ab_decision(
            {"clip": {"pop": 4.0, "halo": 2.0, "legacy_proxy": 0.0}},
            {"clip": {"pop": 5.0, "halo": 3.0, "legacy_proxy": 99.0}},
            ["clip"], specs)
        self.assertEqual(result["verdict"], "tradeoff")
        self.assertEqual(result["improved"], 1)
        self.assertEqual(result["regressed"], 1)

    def test_ab_decision_hard_constraint_cannot_be_traded(self):
        specs = {
            "vmis": {"role": "hard", "axis": "comfort", "hard_max": 0.5,
                     "better": "lower", "rel_tol": 0.0, "abs_floor": 0.1},
            "pop": {"role": "primary", "axis": "stereo", "better": "higher",
                    "rel_tol": 0.0, "abs_floor": 0.5},
        }
        result = sbsbench.evaluate_ab_decision(
            {"clip": {"vmis": 0.1, "pop": 4.0}},
            {"clip": {"vmis": 0.6, "pop": 8.0}}, ["clip"], specs)
        self.assertEqual(result["verdict"], "reject_hard")

    def test_source_residual_accepts_horizontal_parallax_and_detects_corruption(self):
        rng = np.random.default_rng(42)
        src = np.round(rng.random((96, 160), dtype=np.float32) * 255.0) / 255.0
        shifted = sbsbench._shift_x_edge(src, 5)
        clean = sbsbench.source_match_residual(shifted, src, max_shift=8)
        corrupted = shifted.copy()
        corrupted[24:72, 60:100] = 0.0
        damaged = sbsbench.source_match_residual(corrupted, src, max_shift=8)
        self.assertLess(clean[1], 0.01)
        self.assertGreater(damaged[1], clean[1] + 5.0)

    def test_static_region_jitter_ignores_source_motion_but_detects_static_warp_change(self):
        rng = np.random.default_rng(9)
        src = np.round(rng.random((64, 96), dtype=np.float32) * 255.0) / 255.0
        stable, support = sbsbench.static_region_jitter(src, src, src, src, src, src,
                                                        min_support=0.5)
        self.assertAlmostEqual(stable, 0.0)
        self.assertEqual(support, 1.0)
        changed = src.copy()
        changed[16:48, 30:66] = np.clip(changed[16:48, 30:66] + 0.2, 0, 1)
        jitter, _ = sbsbench.static_region_jitter(changed, changed, src, src, src, src,
                                                  min_support=0.5)
        self.assertGreater(jitter, 20.0)
        moving_src = np.roll(src, 8, axis=1)
        skipped, moving_support = sbsbench.static_region_jitter(
            moving_src, moving_src, src, src, moving_src, src, min_support=0.5)
        self.assertIsNone(skipped)
        self.assertLess(moving_support, 0.5)

    def test_comfort_disparity_reports_both_signed_tails(self):
        dx = np.array([-12.0, -8.0, 0.0, 6.0, 10.0])
        weights = np.ones_like(dx)
        positive, negative = sbsbench.comfort_disparity(dx, weights, eye_width=400, tail=0.8)
        self.assertAlmostEqual(positive, 1.5)
        self.assertAlmostEqual(negative, 3.0)

    def test_hard_integrity_aggregates_worst_frame_not_mean(self):
        agg = sbsbench.aggregate([
            {"source_coverage_pct": 100.0, "positive_disparity_pct": 0.5,
             "vmisalign_pct": 0.01},
            {"source_coverage_pct": 70.0, "positive_disparity_pct": 4.0,
             "vmisalign_pct": 0.2},
        ])
        self.assertEqual(agg["source_coverage_pct"], 70.0)
        self.assertEqual(agg["positive_disparity_pct"], 4.0)
        self.assertEqual(agg["vmisalign_pct"], 0.2)

    def test_resolution_independent_metrics_preserve_normalized_geometry(self):
        dx_small = np.array([-4.0, 0.0, 4.0])
        dx_large = dx_small * 2.0
        weights = np.ones(3)
        spread_small = sbsbench.pop_spread(dx_small, weights) / 400.0 * 100.0
        spread_large = sbsbench.pop_spread(dx_large, weights) / 800.0 * 100.0
        self.assertAlmostEqual(spread_small, spread_large)

    def test_source_coverage_and_integrity_detect_missing_content(self):
        rng = np.random.default_rng(22)
        src = np.round(rng.random((96, 160), dtype=np.float32) * 255.0) / 255.0
        clean = sbsbench._shift_x_edge(src, 5)
        good = sbsbench.source_relative_metrics(clean, src, max_shift=8)
        damaged = clean.copy()
        damaged[20:76, 60:110] = 0.0
        bad = sbsbench.source_relative_metrics(damaged, src, max_shift=8)
        self.assertGreater(good["source_coverage_pct"], 99.0)
        self.assertGreater(good["image_integrity_pct"], 99.0)
        self.assertLess(bad["source_coverage_pct"], good["source_coverage_pct"] - 15.0)
        self.assertLess(bad["image_integrity_pct"], good["image_integrity_pct"] - 10.0)

    def test_source_relative_halo_and_stretch_subtract_real_source_structure(self):
        y, x = np.mgrid[:96, :160]
        src = (0.35 + 0.2 * np.sin(x * 0.55) + 0.15 * np.sin(y * 0.3)).astype(np.float32)
        depth = np.full((24, 40), 0.2, np.float32)
        depth[:, 20:] = 0.8
        clean = sbsbench.source_relative_metrics(src, src, depth, max_shift=4)
        halo_eye = src.copy()
        halo_eye[:, 79:82] = 1.0
        halo = sbsbench.source_relative_metrics(halo_eye, src, depth, max_shift=4)
        stretch_eye = src.copy()
        stretch_eye[:, 82:115] = stretch_eye[:, 82:83]
        stretch = sbsbench.source_relative_metrics(stretch_eye, src, depth, max_shift=4)
        self.assertGreater(halo["source_halo_p95"], clean["source_halo_p95"] + 3.0)
        self.assertGreater(stretch["source_stretch_pct"], clean["source_stretch_pct"] + 10.0)

    def test_ground_truth_depth_metrics_reward_aligned_structure(self):
        gt = np.full((96, 160), 0.25, np.float32)
        gt[:, 80:] = 0.75
        equivalent = gt * 0.8 + 0.1  # monocular scale/shift ambiguity is intentionally free
        flat = np.full_like(gt, 0.5)
        good = sbsbench.depth_ground_truth_metrics(equivalent, gt)
        bad = sbsbench.depth_ground_truth_metrics(flat, gt)
        self.assertLess(good["depth_gt_si_rmse"], 0.01)
        self.assertGreater(good["depth_gt_edge_f1"], 99.0)
        self.assertGreater(bad["depth_gt_si_rmse"], 40.0)
        self.assertLess(bad["depth_gt_edge_f1"], 1.0)

    def test_ground_truth_depth_metrics_reject_inverted_polarity(self):
        gt = np.full((96, 160), 0.2, np.float32)
        gt[:, 80:] = 0.8
        inverted = 1.0 - gt
        metrics = sbsbench.depth_ground_truth_metrics(inverted, gt)
        self.assertGreater(metrics["depth_gt_si_rmse"], 20.0)
        self.assertLess(metrics["depth_gt_edge_f1"], 1.0)

    def test_ground_truth_edge_tolerance_works_in_both_axes(self):
        gt = np.full((96, 160), 0.25, np.float32)
        gt[48:, :] = 0.75
        shifted = np.full_like(gt, 0.25)
        shifted[49:, :] = 0.75
        metrics = sbsbench.depth_ground_truth_metrics(shifted, gt)
        self.assertGreater(metrics["depth_gt_edge_f1"], 99.0)

    def test_metric_depth_resize_does_not_invert_interpolated_invalid_holes(self):
        gt = np.full((48, 80), 2.0, np.float32)
        gt[12:36, 38:42] = 0.0
        resized, valid = sbsbench.resize_metric_depth(gt, 40, 24)
        self.assertTrue(np.all(resized[valid] > 1.9))
        self.assertTrue(np.all(resized[valid] < 2.1))
        prediction = np.full((24, 40), 0.5, np.float32)
        metrics = sbsbench.depth_ground_truth_metrics(prediction, gt, "metric")
        self.assertLess(metrics["depth_gt_si_rmse"], 0.01)
        self.assertGreater(metrics["depth_gt_edge_f1"], 99.0)

    def test_optical_flow_temporal_metric_compensates_motion(self):
        rng = np.random.default_rng(31)
        previous = np.round(rng.random((96, 160), dtype=np.float32) * 255.0) / 255.0
        current = np.roll(previous, 5, axis=1)
        stable, _, support = sbsbench.flow_temporal_metrics(
            current, current, previous, previous, current, previous, min_support=0.1)
        corrupted = current.copy()
        corrupted[20:75, 60:110] = 0.0
        unstable, _, _ = sbsbench.flow_temporal_metrics(
            corrupted, corrupted, previous, previous, current, previous, min_support=0.1)
        self.assertGreater(support, 0.8)
        self.assertLess(stable, 2.0)
        self.assertGreater(unstable, stable + 100.0)

    def test_exact_forward_flow_temporal_metric_compensates_motion(self):
        rng = np.random.default_rng(71)
        previous = np.round(rng.random((96, 160), dtype=np.float32) * 255.0) / 255.0
        current = np.zeros_like(previous)
        current[:, 5:] = previous[:, :-5]
        flow = np.zeros((96, 160, 2), np.float32)
        flow[..., 0] = 5.0
        valid = np.ones((96, 160), bool)
        valid[:, -5:] = False
        stable, _, support = sbsbench.flow_temporal_metrics(
            current, current, previous, previous, current, previous, min_support=0.1,
            reference_flow=flow, reference_valid=valid)
        self.assertGreater(support, 0.8)
        self.assertLess(stable, 2.0)

    def test_npy_metric_depth_preserves_native_values(self):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as fh:
            path = fh.name
        try:
            expected = np.array([[0.25, 4.0], [12.5, 200.0]], np.float32)
            np.save(path, expected)
            np.testing.assert_array_equal(sbsbench.load_depth(path), expected)
        finally:
            os.unlink(path)

    def test_public_dataset_timestamp_association_is_nearest_and_unique(self):
        rgb = [(0.00, "r0"), (0.10, "r1"), (0.20, "r2")]
        depth = [(0.009, "d0"), (0.105, "d1"), (0.35, "far")]
        pairs = prepare_public_datasets.associate_timestamps(rgb, depth, 0.03)
        self.assertEqual([(p[1], p[3]) for p in pairs], [("r0", "d0"), ("r1", "d1")])

    def test_suite_defaults_keep_core_and_extended_baselines_separate(self):
        core_clips, core_baselines = run_eval.suite_defaults("core")
        extended_clips, extended_baselines = run_eval.suite_defaults("extended")
        self.assertTrue(core_clips.endswith(os.path.join("sbsbench", "clips")))
        self.assertTrue(core_baselines.endswith(os.path.join("sbsbench", "baselines")))
        self.assertIn(os.path.join("prepared", "extended-v2"), extended_clips)
        self.assertTrue(extended_baselines.endswith("baselines_extended"))

    def test_sintel_adapter_preserves_left_and_rendered_right_frames(self):
        with tempfile.TemporaryDirectory() as root:
            archive = os.path.join(root, "sintel.zip")
            with zipfile.ZipFile(archive, "w") as zf:
                for i in range(3):
                    for eye, value in (("left", 40 + i), ("right", 80 + i)):
                        zf.writestr(f"training/final_{eye}/demo/frame_{i + 1:04d}.png",
                                    self.png_bytes(value))
                    zf.writestr(f"training/disparities/demo/frame_{i + 1:04d}.png",
                                self.png_bytes(10 + i))
            out = os.path.join(root, "out")
            os.makedirs(out)
            clip = {"archives": ["stereo"], "sequence": "demo", "pass": "final",
                    "start": 0, "stride": 1, "count": 2}
            rows = prepare_public_datasets.prepare_sintel(
                "demo", clip, {}, {"stereo": archive}, out, "test")
            self.assertEqual(len(rows), 2)
            self.assertTrue(os.path.exists(os.path.join(out, "frame_00000.png")))
            self.assertTrue(os.path.exists(os.path.join(out, "gt_right", "frame_00001.png")))
            self.assertTrue(os.path.exists(os.path.join(out, "gt_depth", "frame_00001.npy")))

    def test_vkitti_adapter_selects_matching_rgb_and_depth(self):
        with tempfile.TemporaryDirectory() as root:
            archives = {}
            for modality in ("rgb", "depth"):
                path = os.path.join(root, modality + ".tar")
                archives[modality] = path
                with tarfile.open(path, "w") as tf:
                    for i in range(3):
                        suffix = f"rgb_{i:05d}.jpg" if modality == "rgb" else f"depth_{i:05d}.png"
                        folder = "rgb" if modality == "rgb" else "depth"
                        data = self.png_bytes(50 + i, "RGB" if modality == "rgb" else "L")
                        info = tarfile.TarInfo(
                            f"vkitti/{'Scene01'}/clone/frames/{folder}/Camera_0/{suffix}")
                        info.size = len(data)
                        tf.addfile(info, io.BytesIO(data))
            out = os.path.join(root, "out")
            os.makedirs(out)
            clip = {"scene": "Scene01", "variant": "clone", "camera": "Camera_0",
                    "start": 1, "stride": 1, "count": 2}
            rows = prepare_public_datasets.prepare_vkitti2(
                "demo", clip, {}, archives, out, "test")
            self.assertEqual([r["dataset_frame"] for r in rows], [1, 2])
            self.assertTrue(os.path.exists(os.path.join(out, "frame_00000.png")))
            self.assertTrue(os.path.exists(os.path.join(out, "gt_depth", "frame_00001.png")))


if __name__ == "__main__":
    unittest.main()
