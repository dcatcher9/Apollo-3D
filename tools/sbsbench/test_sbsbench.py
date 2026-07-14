import io
import json
import os
import sys
import tarfile
import tempfile
import argparse
import unittest
from unittest import mock
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_depth_transform  # noqa: E402
import audit_depth_confidence  # noqa: E402
import prepare_public_datasets  # noqa: E402
import prepare_flow_ema_reference  # noqa: E402
import run_eval  # noqa: E402
import rescore_run  # noqa: E402
import sbsbench  # noqa: E402


class EvalContractTests(unittest.TestCase):
    @staticmethod
    def png_bytes(value=64, mode="RGB"):
        shape = (8, 12, 3) if mode == "RGB" else (8, 12)
        array = np.full(shape, value, np.uint8)
        stream = io.BytesIO()
        Image.fromarray(array, mode=mode).save(stream, "PNG")
        return stream.getvalue()

    def test_metric_hash_is_independent_of_text_line_endings(self):
        paths = []
        try:
            for data in (b"alpha\nbeta\n", b"alpha\r\nbeta\r\n"):
                with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as fh:
                    fh.write(data)
                    paths.append(fh.name)
            # sha256_files includes the basename, so give both temp files the same logical name.
            with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
                left_path = os.path.join(left, "metric.py")
                right_path = os.path.join(right, "metric.py")
                with open(paths[0], "rb") as src, open(left_path, "wb") as dst:
                    dst.write(src.read())
                with open(paths[1], "rb") as src, open(right_path, "wb") as dst:
                    dst.write(src.read())
                self.assertEqual(run_eval.sha256_files([left_path]),
                                 run_eval.sha256_files([right_path]))
        finally:
            for path in paths:
                os.unlink(path)

    def test_clip_hash_covers_stereo_reference_and_requirement(self):
        with tempfile.TemporaryDirectory() as clip:
            gt_right = os.path.join(clip, "gt_right")
            os.makedirs(gt_right)
            Image.fromarray(np.zeros((8, 12, 3), np.uint8)).save(
                os.path.join(clip, "frame_00000.png"))
            reference_path = os.path.join(gt_right, "frame_00000.png")
            Image.fromarray(np.zeros((8, 12, 3), np.uint8)).save(reference_path)
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"required_gt_stereo": True}, fh)
            original = run_eval.sha1_dir(clip)
            Image.fromarray(np.full((8, 12, 3), 255, np.uint8)).save(reference_path)
            self.assertNotEqual(original, run_eval.sha1_dir(clip))
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"required_gt_stereo": False}, fh)
            changed_pixels = run_eval.sha1_dir(clip)
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"required_gt_stereo": True}, fh)
            self.assertNotEqual(changed_pixels, run_eval.sha1_dir(clip))

    def test_metric_contract_excludes_runner_diagnostics(self):
        metric_files = [os.path.join(run_eval.SCRIPT_DIR, "sbsbench.py"),
                        os.path.join(run_eval.SCRIPT_DIR, "thresholds.json")]
        self.assertEqual(run_eval.metric_contract_sha(),
                         run_eval.sha256_files(metric_files))
        self.assertNotEqual(
            run_eval.metric_contract_sha(),
            run_eval.sha256_files(metric_files + [os.path.abspath(run_eval.__file__)]))

    def test_named_profiles_and_explicit_overrides_share_production_precedence(self):
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            fh.write("sbs_3d_profile = cinema\n")
            path = fh.name
        try:
            self.assertEqual(run_eval.expected_profile(path, []), "cinema")
        finally:
            os.unlink(path)

    def test_apollo_is_the_unconfigured_default_profile(self):
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            path = fh.name
        try:
            self.assertEqual(run_eval.expected_profile(path, []), "apollo")
        finally:
            os.unlink(path)

    def test_committed_gate_tracks_the_production_default_profile(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        bench_conf = os.path.join(repo, "tools", "sbsbench", "bench.conf")
        self.assertEqual(run_eval.expected_profile(bench_conf, []), "apollo")
        with open(os.path.join(repo, "src", "config.h"), encoding="utf-8") as fh:
            self.assertIn('std::string profile = "apollo"', fh.read())

    def test_baseline_update_refuses_gpu_contention(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "tools", "sbsbench", "run_eval.py"),
                  encoding="utf-8") as fh:
            evaluator = fh.read()
        self.assertIn("if args.update_baselines:", evaluator)
        self.assertIn("refusing --update-baselines while another sunshine.exe is running",
                      evaluator)

    def test_custom_profile_values_need_no_evaluator_code_change(self):
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            fh.write("sbs_3d_profile = Cinema\n"
                     "sbs_3d_profile_Cinema_depth_model = depth_anything_v2_base_fp16\n")
            path = fh.name
        try:
            self.assertEqual(run_eval.expected_profile(path, []), "Cinema")
            self.assertEqual(run_eval.expected_depth_model(path, "Cinema", []),
                             "depth_anything_v2_base_fp16")
            self.assertEqual(
                run_eval.expected_depth_model(
                    path, "Cinema", ["--model", "depth_anything_v2_fp8"]),
                "depth_anything_v2_fp8")
        finally:
            os.unlink(path)

    def test_live_sbs_contract_is_off_ai_with_startup_profile(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "video.h"), encoding="utf-8") as fh:
            video_header = fh.read()
        self.assertIn("SBS_AI = 1", video_header)
        self.assertNotIn("SBS_GAME", video_header)
        self.assertNotIn("SBS_MOVIE", video_header)

        with open(os.path.join(repo, "src", "config.cpp"), encoding="utf-8") as fh:
            config = fh.read()
        self.assertIn('apply_sbs_values(video.sbs, "sbs_3d_profile_" + sbs_profile + "_")', config)
        self.assertNotIn("video.sbs_profiles", config)
        self.assertIn('if (sbs_profile == "vd3d")', config)

        with open(os.path.join(repo, "src", "stream.cpp"), encoding="utf-8") as fh:
            stream = fh.read()
        self.assertNotIn("IDX_SET_SBS_PROFILE", stream)
        self.assertNotIn("IDX_SBS_PROFILE_LIST", stream)
        self.assertIn("mail::sbs_depth_status", stream)
        self.assertNotIn("depth_engine_phase", stream)
        self.assertNotIn("set_active_depth_model(id)", stream)

        with open(os.path.join(repo, "src", "main.cpp"), encoding="utf-8") as fh:
            main = fh.read()
        self.assertIn("prepare_tensorrt_model", main)
        self.assertIn("std::jthread model_prepare_thread", main)
        self.assertLess(main.index("if (!config::sunshine.cmd.name.empty())"),
                        main.index("std::jthread model_prepare_thread"))

        with open(os.path.join(repo, "src", "video_depth_estimator.cpp"),
                  encoding="utf-8") as fh:
            estimator = fh.read()
        self.assertIn("cuda_device_for_configured_adapter", estimator)
        self.assertIn("if (!warmup_execution_context", estimator)
        self.assertIn("const bool synchronized = enqueued", estimator)

    def test_relative_cli_paths_are_resolved_before_subprocess_cwd(self):
        args = argparse.Namespace(build_dir="cmake-build-relwithdebinfo", conf="bench.conf",
                                  clips_root=None, baseline_dir=None,
                                  report_control=None, report_out=None)
        run_eval.normalize_cli_paths(args)
        self.assertTrue(os.path.isabs(args.build_dir))
        self.assertTrue(os.path.isabs(args.conf))

    def test_eval_builds_production_binary_and_fails_closed_on_build_error(self):
        current = mock.Mock(returncode=0, stdout="ninja: no work to do.\n", stderr="")
        with mock.patch.object(run_eval.shutil, "which", return_value="ninja"), \
                mock.patch.object(run_eval.subprocess, "run", return_value=current) as run:
            run_eval.require_current_build("build")
        self.assertEqual(run.call_args.args[0], ["ninja", "-C", "build", "sunshine"])
        failed = mock.Mock(returncode=1, stdout="compile failed\n", stderr="")
        with mock.patch.object(run_eval.shutil, "which", return_value="ninja"), \
                mock.patch.object(run_eval.subprocess, "run", return_value=failed), \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                run_eval.require_current_build("build")

    def test_apollo_bestv2_normalizes_pixel_shifts_by_source_geometry(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "sbs_reprojection_ps.hlsl")
        with open(shader, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("LeftColorTexture.GetDimensions(sourceWidth, sourceHeight)", text)
        self.assertIn("Bestv2SearchRadius((float)sourceWidth, (float)sourceHeight)", text)
        self.assertIn("s0, s1, (float)sourceWidth, (float)sourceHeight", text)
        self.assertEqual(text.count("DepthParallax("), 2)
        self.assertNotIn("Bestv2SearchRadius((float)dw)", text)

    def test_forward_coverage_diagnostic_uses_source_geometry(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx")
        with open(os.path.join(shader_dir, "sbs_forward_coverage_cs.hlsl"),
                  encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("LeftColorTexture.GetDimensions(source_w, source_h)", text)
        self.assertIn("s0, s1, (float)source_w, (float)source_h", text)
        self.assertNotIn("s0, s1, (float)eye_w", text)

    def test_bestv2_scales_wide_sources_from_validated_calibration_width(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        common = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "include", "sbs_warp_common.hlsl")
        with open(common, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("BESTV2_CALIBRATION_WIDTH = 854.0f", text)
        self.assertIn("return min(max(source_width, 1.0f), BESTV2_CALIBRATION_WIDTH)", text)
        self.assertGreaterEqual(text.count("/ parallax_width"), 2)

    def test_bestv2_preserves_angular_pop_across_source_aspects(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        common = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "include", "sbs_warp_common.hlsl")
        with open(common, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("BESTV2_REFERENCE_ASPECT = 5120.0f / 2160.0f", text)
        self.assertIn("BESTV2_REFERENCE_ASPECT / aspect", text)
        self.assertNotIn("fixed_height", text)
        reference_aspect = 5120.0 / 2160.0
        self.assertAlmostEqual(reference_aspect / (5120.0 / 2160.0), 1.0)
        self.assertAlmostEqual(reference_aspect / (3840.0 / 2160.0), 4.0 / 3.0)
        self.assertAlmostEqual(reference_aspect / (3552.0 / 3840.0), 2.562562563, places=6)

    def test_pop_strength_scales_shared_parallax_and_apollo_probe_radius(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        common = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "include", "sbs_warp_common.hlsl")
        with open(common, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("float pop_strength;", text)
        self.assertIn("pop_strength * adaptive_ratio", text)
        self.assertIn("clamp(parallax * p.output_scale", text)

        with open(os.path.join(repo, "src", "config.cpp"), encoding="utf-8") as fh:
            config = fh.read()
        self.assertIn('prefix + "pop_strength", target.pop_strength, {0.25, 2.0}', config)
        with open(os.path.join(repo, "src", "config.h"), encoding="utf-8") as fh:
            config_header = fh.read()
        self.assertIn("double pop_strength = 1.25;", config_header)
        self.assertIn("bool adaptive_pop = true;", config_header)
        self.assertIn("double adaptive_pop_max = 1.30;", config_header)

        with open(os.path.join(repo, "src", "platform", "windows", "display_vram.cpp"),
                  encoding="utf-8") as fh:
            production = fh.read()
        self.assertIn("(float) sbs_config.pop_strength", production)
        self.assertIn("sbs_config.adaptive_pop ? 1.0f : 0.0f", production)

        with open(os.path.join(repo, "src_assets", "windows", "assets", "shaders",
                               "directx", "depth_subject_resolve_cs.hlsl"),
                  encoding="utf-8") as fh:
            adaptive = fh.read()
        self.assertIn("change_fraction >= 0.65f", adaptive)
        self.assertIn("scene_age >= 8.0f", adaptive)
        self.assertIn("smoothstep(0.007f, 0.016f, edge_fraction)", adaptive)
        self.assertNotIn("lerp(pop_ratio, target_ratio", adaptive)

    def test_adaptive_pop_last_flag_wins(self):
        conf = os.path.join(os.path.dirname(__file__), "bench.conf")
        self.assertFalse(run_eval.expected_adaptive_pop(
            conf, "apollo", ["--adaptive-pop", "--no-adaptive-pop"]))
        self.assertTrue(run_eval.expected_adaptive_pop(
            conf, "apollo", ["--no-adaptive-pop", "--adaptive-pop"]))

    def test_literal_bestv2_is_harness_only_and_machine_verified(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                               "include", "sbs_warp_common.hlsl"), encoding="utf-8") as fh:
            shader = fh.read()
        self.assertIn("float literal_bestv2;", shader)
        self.assertIn("literal_mode > 0.5f", shader)

        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"), encoding="utf-8") as fh:
            harness = fh.read()
        self.assertIn('a == "--literal-bestv2"', harness)
        self.assertIn('fs::path(o.out) / "contract.json"', harness)
        self.assertIn('"  \\"schema\\": 14,\\n"', harness)
        self.assertIn('\\"depth_override_frames\\"', harness)

        with open(os.path.join(repo, "tools", "sbsbench", "run_eval.py"),
                  encoding="utf-8") as fh:
            evaluator = fh.read()
        self.assertIn('contract_path = os.path.join(out_dir, "contract.json")', evaluator)
        self.assertNotIn("profile ([a-z0-9_-]+)", evaluator)

        with open(os.path.join(repo, "src", "stream.cpp"), encoding="utf-8") as fh:
            stream = fh.read()
        self.assertNotIn("SBS_PRESENTATION_FIXED_HEIGHT", stream)

    def test_depth_reuse_cadence_is_explicit_and_machine_verified(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"), encoding="utf-8") as fh:
            harness = fh.read()
        self.assertIn('a == "--depth-every"', harness)
        self.assertIn('a == "--depth-override-root"', harness)
        self.assertIn("depth_compensation", harness)
        with open(os.path.join(repo, "tools", "sbsbench", "run_eval.py"),
                  encoding="utf-8") as fh:
            evaluator = fh.read()
        self.assertIn('"depth_compensation": depth_compensation', evaluator)
        self.assertIn("depth_reuse_interval", harness)
        with open(os.path.join(repo, "tools", "sbsbench", "run_eval.py"),
                  encoding="utf-8") as fh:
            evaluator = fh.read()
        self.assertIn('extra_value(args.extra, "--depth-every", 1)', evaluator)
        self.assertIn('f"reuse-{depth_reuse_interval}"', evaluator)
        self.assertIn('"schema": 14', evaluator)
        self.assertIn('depth_override_root and not args.comparison_only', evaluator)

    def test_live_depth_pairing_is_bounded_and_sync_is_evaluation_only(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "config.cpp"), encoding="utf-8") as fh:
            config = fh.read()
        self.assertNotIn('"depth_frame_mode"', config)
        self.assertNotIn('"depth_fps"', config)

        with open(os.path.join(repo, "src", "platform", "windows", "display_vram.cpp"),
                  encoding="utf-8") as fh:
            production = fh.read()
        self.assertIn("std::array<matched_frame_slot_t, 2>", production)
        self.assertIn("repeat_matched_output", production)
        self.assertNotIn("finish_pending_depth", production)
        self.assertNotIn("depth_frame_mode", production)

        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"),
                  encoding="utf-8") as fh:
            harness = fh.read()
        self.assertIn("finish_pending_depth_for_evaluation", harness)

    def test_cuda_graph_replay_is_signature_safe_and_falls_back(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "config.h"), encoding="utf-8") as fh:
            config = fh.read()
        self.assertIn("bool cuda_graph = true;", config)
        with open(os.path.join(repo, "src", "video_depth_estimator.cpp"),
                  encoding="utf-8") as fh:
            estimator = fh.read()
        self.assertIn("input != graph_input || output != graph_output", estimator)
        self.assertIn("target_w != graph_width || target_h != graph_height", estimator)
        self.assertIn("if (!graph_signature_warmed)", estimator)
        self.assertIn("destroy_inference_graph(cuda);", estimator)
        self.assertIn("return exec_context->enqueueV3(cu_stream);", estimator)
        with open(os.path.join(repo, "src", "cuda_driver_api.h"), encoding="utf-8") as fh:
            driver = fh.read()
        for symbol in ("cuStreamBeginCapture", "cuStreamEndCapture",
                       "cuGraphInstantiateWithFlags", "cuGraphLaunch",
                       "cuGraphExecDestroy"):
            self.assertIn(symbol, driver)

    def test_cuda_graph_eval_override_matches_profile_precedence(self):
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            fh.write("sbs_3d_profile = cinema\n"
                     "sbs_3d_profile_cinema_cuda_graph = false\n")
            path = fh.name
        try:
            self.assertFalse(run_eval.expected_profile_bool(
                path, "cinema", "cuda_graph", True, [], "--cuda-graph"))
            self.assertTrue(run_eval.expected_profile_bool(
                path, "cinema", "cuda_graph", True,
                ["--cuda-graph", "on"], "--cuda-graph"))
        finally:
            os.unlink(path)

    def test_edge_selective_ema_uses_immutable_history_and_exports_locality_mask(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders",
                                  "directx")
        with open(os.path.join(shader_dir, "depth_ema_motion_cs.hlsl"),
                  encoding="utf-8") as fh:
            mask_shader = fh.read()
        self.assertIn("PreviousDepth", mask_shader)
        self.assertIn("ema_edge_change", mask_shader)
        self.assertNotIn("ema_edge_dilation", mask_shader)
        self.assertIn("MotionMask[DTid.xy] = IsMovingEdge", mask_shader)
        with open(os.path.join(repo, "src", "video_depth_estimator.cpp"),
                  encoding="utf-8") as fh:
            estimator = fh.read()
        self.assertIn("CopyResource(depth_previous_tex.Get(), depth_tex.Get())", estimator)
        self.assertIn("ema_motion_mask_srv", estimator)
        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"),
                  encoding="utf-8") as fh:
            harness = fh.read()
        self.assertIn('"ema_mask_%s.png"', harness)

    def test_depth_override_manifest_requires_exact_frames_and_source_hash(self):
        with tempfile.TemporaryDirectory() as root:
            clips_root = os.path.join(root, "clips")
            clip_dir = os.path.join(clips_root, "sample")
            override_root = os.path.join(root, "override")
            override_clip = os.path.join(override_root, "sample")
            os.makedirs(clip_dir)
            os.makedirs(override_clip)
            for frame_id in range(3):
                Image.fromarray(np.full((8, 12, 3), frame_id, np.uint8)).save(
                    os.path.join(clip_dir, f"frame_{frame_id:05d}.png"))
            Image.fromarray(np.full((4, 6), 32768, np.uint16)).save(
                os.path.join(override_clip, "depth_00001.png"))
            manifest = {
                "schema": 3,
                "method": "classical-tile-phase-flow",
                "frame_policy": "held",
                "depth_every": 2,
                "clips": {"sample": {
                    "override_frames": 1,
                    "override_frame_ids": [1],
                    "clip_sha1": run_eval.sha1_dir(clip_dir),
                }},
            }
            with open(os.path.join(override_root, "manifest.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(manifest, fh)
            self.assertEqual(run_eval.validate_depth_override_manifest(
                override_root, clips_root, ["sample"], 2), {"sample": 1})
            os.remove(os.path.join(override_clip, "depth_00001.png"))
            original_stderr = sys.stderr
            try:
                sys.stderr = io.StringIO()
                with self.assertRaises(SystemExit):
                    run_eval.validate_depth_override_manifest(
                        override_root, clips_root, ["sample"], 2)
            finally:
                sys.stderr = original_stderr

    def test_all_frame_depth_treatment_manifest_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            clips_root = os.path.join(root, "clips")
            clip_dir = os.path.join(clips_root, "sample")
            override_root = os.path.join(root, "override")
            override_clip = os.path.join(override_root, "sample")
            os.makedirs(clip_dir)
            os.makedirs(override_clip)
            frame_ids = list(range(3))
            for frame_id in frame_ids:
                Image.fromarray(np.full((8, 12, 3), frame_id, np.uint8)).save(
                    os.path.join(clip_dir, f"frame_{frame_id:05d}.png"))
                Image.fromarray(np.full((4, 6), 32768, np.uint16)).save(
                    os.path.join(override_clip, f"depth_{frame_id:05d}.png"))
            manifest = {
                "schema": 3,
                "method": "flow-aware-ema-oracle",
                "frame_policy": "all",
                "depth_every": 1,
                "clips": {"sample": {
                    "override_frames": 3,
                    "override_frame_ids": frame_ids,
                    "clip_sha1": run_eval.sha1_dir(clip_dir),
                }},
            }
            with open(os.path.join(override_root, "manifest.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(manifest, fh)
            self.assertEqual(run_eval.validate_depth_override_manifest(
                override_root, clips_root, ["sample"], 1, True), {"sample": 3})
            os.remove(os.path.join(override_clip, "depth_00002.png"))
            original_stderr = sys.stderr
            try:
                sys.stderr = io.StringIO()
                with self.assertRaises(SystemExit):
                    run_eval.validate_depth_override_manifest(
                        override_root, clips_root, ["sample"], 1, True)
            finally:
                sys.stderr = original_stderr

    def test_rescore_derives_depth_compensation_for_schema_upgrade(self):
        self.assertEqual(rescore_run.depth_compensation_from_meta({}), "none")
        self.assertEqual(rescore_run.depth_compensation_from_meta(
            {"extra_args": ["--depth-override-root", "reference"]}),
            "external-reference")
        self.assertEqual(rescore_run.depth_compensation_from_meta(
            {"depth_compensation": "nvof-1x1"}), "nvof-1x1")
        self.assertEqual(rescore_run.depth_compensation_from_meta(
            {"extra_args": ["--depth-override-root", "reference",
                            "--depth-override-all"]}),
            "external-treatment")

    def test_warp_and_coverage_apply_per_eye_aspect_mapping(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx")
        for name in ("sbs_reprojection_ps.hlsl", "sbs_forward_coverage_cs.hlsl"):
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
        self.assertNotIn("sbs_sharpen", pipeline)
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
        self.assertIn('"schema": 2', text)
        self.assertIn('"report_sha256": REPORT_SHA', text)
        self.assertIn('AB_DECISION["verdict"]', text)
        self.assertIn("IS_PROFILE_CMP", text)
        self.assertIn("IS_TRADEOFF_CMP = IS_MODE_CMP or IS_PROFILE_CMP", text)

    def test_live_trt_contexts_are_bounded_and_engine_io_fails_closed(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(repo, "src", "video_depth_estimator.cpp")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("kMaxContextsPerEngine = 4", text)
        self.assertIn("slot.context_count >= kMaxContextsPerEngine", text)
        self.assertIn("g_trt_context_available.wait_for", text)
        self.assertIn("slot.io_compatible = have_in && have_out && input_fp32 && output_fp32", text)
        self.assertIn("validate_engine_io_locked", text)

    def test_live_gpu_timer_tail_is_bounded_and_generation_safe(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        display_path = os.path.join(repo, "src", "platform", "windows", "display_vram.cpp")
        with open(display_path, encoding="utf-8") as fh:
            display = fh.read()
        with open(os.path.join(repo, "src", "sbs_perf.cpp"), encoding="utf-8") as fh:
            perf = fh.read()
        self.assertIn("drain_sbs_gpu_timers();", display)
        self.assertIn("std::chrono::milliseconds(100)", display)
        self.assertIn("sbs_perf::add_sample_ms_if_current", display)
        self.assertIn("g_generation.fetch_add", perf)

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

    def test_bestv2_fast_curve_is_subpixel_and_live_only(self):
        depth = np.linspace(0.0, 1.0, 100001, dtype=np.float64)
        near = np.exp(-0.5 * ((depth - 0.85) / 0.24) ** 2)
        middle = np.exp(-0.5 * ((depth - 0.50) / 0.28) ** 2)
        far = np.exp(-0.5 * ((depth - 0.15) / 0.24) ** 2)
        exact = (near * 9.99 + middle * 3.0 - far * 2.52) / (near + middle + far + 1e-6)
        coeffs = (-1.39635933, 2.776208766, 21.04503417, -94.6673759,
                  376.6610774, -645.141824, 482.8701123, -133.5645677)
        approx = np.full_like(depth, coeffs[-1])
        for coefficient in reversed(coeffs[:-1]):
            approx = approx * depth + coefficient
        self.assertLess(np.max(np.abs(approx - exact)), 0.01)

        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders",
                                  "directx")
        with open(os.path.join(shader_dir, "include", "sbs_warp_common.hlsl"),
                  encoding="utf-8") as fh:
            warp_common = fh.read()
        with open(os.path.join(shader_dir, "include", "bestv2_curve.hlsl"),
                  encoding="utf-8") as fh:
            curve = fh.read()
        self.assertIn("Bestv2RawShiftPxFast(shaped_depth)", warp_common)
        self.assertNotIn("Bestv2RawShiftPx(float", curve)

    def test_tensorrt_level_is_part_of_engine_recipe(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "model_manager.h"), encoding="utf-8") as fh:
            manager = fh.read()
        with open(os.path.join(repo, "src", "video_depth_estimator.cpp"),
                  encoding="utf-8") as fh:
            estimator = fh.read()
        self.assertIn("depth_engine_builder_level = 5", manager)
        self.assertIn("trt-opt770x434-level5-v2", manager)
        self.assertIn("setBuilderOptimizationLevel(depth_engine_builder_level)", estimator)

    def test_live_and_eval_shaders_use_level3_optimization(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "platform", "windows", "display_vram.cpp"),
                  encoding="utf-8") as fh:
            live = fh.read()
        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"),
                  encoding="utf-8") as fh:
            harness = fh.read()
        self.assertIn("flags |= D3DCOMPILE_OPTIMIZATION_LEVEL3", live)
        self.assertIn("D3DCOMPILE_OPTIMIZATION_LEVEL3", harness)

    def test_production_warp_has_no_retired_plane_lock_path(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "sbs_reprojection_ps.hlsl")
        with open(shader, encoding="utf-8") as fh:
            text = fh.read()
        self.assertNotIn("PlaneLockTexture", text)
        self.assertNotIn("subject_plane_lock", text)

        common = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "include", "sbs_warp_common.hlsl")
        with open(common, encoding="utf-8") as fh:
            common_text = fh.read()
        self.assertNotIn("use_plane_lock", common_text)
        self.assertIn("sample_uv = Reproject(src_uv, eyeSign, true)", text)
        self.assertIn("sample_uv = Reproject(src_uv, eyeSign, false)", text)
        self.assertIn("MakeBestv2Params", text)
        self.assertIn(
            "DepthParallax(\n            d, s0, s1, params, use_subject_stretch)",
            text)
        self.assertNotIn("DepthParallax(d, s0, s1, shaped", text)

    def test_retired_geometry_is_absent_but_forward_coverage_diagnostic_remains(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        display = os.path.join(repo, "src", "platform", "windows", "display_vram.cpp")
        with open(display, encoding="utf-8") as fh:
            display_text = fh.read()
        self.assertNotIn("sbs_vd3d", display_text)
        self.assertNotIn("sbs_sharpen", display_text)
        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"),
                  encoding="utf-8") as fh:
            harness_text = fh.read()
        self.assertIn("sbs_forward_coverage_cs.hlsl", harness_text)
        self.assertIn("dispatch_coverage", harness_text)
        for retired in ("subject_plane_lock", "subject_plane_width", "bestv2_sharpen",
                        "ema_edge_dilation"):
            self.assertNotIn(retired, harness_text)
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders",
                                  "directx")
        for retired_shader in ("depth_plane_band_cs.hlsl", "depth_plane_combine_cs.hlsl",
                               "depth_plane_filter_cs.hlsl", "depth_plane_reduce_cs.hlsl",
                               "depth_plane_resolve_cs.hlsl", "sbs_sharpen_ps.hlsl"):
            self.assertFalse(os.path.exists(os.path.join(shader_dir, retired_shader)))
        self.assertFalse(os.path.exists(os.path.join(
            shader_dir, "include", "depth_plane_constants.hlsl")))

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

    def test_public_clip_rejects_missing_required_stereo_reference(self):
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
                fh.write('{"required_gt_stereo":true}')
            with self.assertRaisesRegex(ValueError, "requires GT stereo"):
                sbsbench.measure_sequence(seq, frames)

    def test_public_clip_rejects_missing_required_depth_lag_metric(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            gt_dir = os.path.join(frames, "gt_depth")
            os.makedirs(seq)
            os.makedirs(gt_dir)
            for frame_id in range(2):
                Image.fromarray(np.zeros((16, 32, 3), np.uint8)).save(
                    os.path.join(seq, f"sbs_{frame_id:05d}.png"))
                Image.fromarray(np.full((8, 16), 32768, np.uint16)).save(
                    os.path.join(seq, f"depth_{frame_id:05d}.png"))
                Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(
                    os.path.join(frames, f"frame_{frame_id:05d}.png"))
                Image.fromarray(np.full((8, 16), 32768, np.uint16)).save(
                    os.path.join(gt_dir, f"frame_{frame_id:05d}.png"))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"required_gt_depth": True, "gt_depth_kind": "disparity"}, fh)
            with mock.patch.object(sbsbench, "depth_ground_truth_lag", return_value=None):
                with self.assertRaisesRegex(ValueError, "depth_gt_lag_f1_p95"):
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

    def test_exact_warp_mask_restricts_fill_error_and_artifact_overlap(self):
        source = np.zeros((64, 64), dtype=np.float32)
        left = source.copy()
        right = source.copy()
        left[20:30, 25:35] = 1.0
        right[20:30, 25:35] = 1.0
        mask = np.zeros((64, 128, 3), dtype=np.float32)
        mask[20:30, 25:35, 0] = 1.0
        mask[20:30, 64 + 25:64 + 35, 0] = 1.0
        metrics = sbsbench.warp_hole_metrics(left, right, mask, source)
        self.assertGreater(metrics["warp_hole_pct"], 1.0)
        self.assertNotIn("warp_unresolved_pct", metrics)
        self.assertGreater(metrics["hole_source_residual_p95"], 100.0)
        self.assertGreater(metrics["hole_bad_fill_pct"], 80.0)
        self.assertGreater(metrics["artifact_in_hole_pct"], 80.0)

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
        positive, negative = sbsbench.comfort_disparity(
            dx, weights, eye_width=400,
            eye_height=400 / sbsbench.REFERENCE_STREAM_ASPECT, tail=0.8)
        self.assertAlmostEqual(positive, 1.5)
        self.assertAlmostEqual(negative, 3.0)

    def test_perceived_disparity_is_client_aspect_invariant(self):
        ref = sbsbench.perceived_disparity_pct(51.2, 5120, 2160)
        # The aspect correction keeps pixel disparity constant when pixel height is unchanged;
        # at a taller raster it grows in direct proportion to height.
        uhd = sbsbench.perceived_disparity_pct(51.2, 3840, 2160)
        tall = sbsbench.perceived_disparity_pct(51.2 * 3840.0 / 2160.0, 3552, 3840)
        self.assertAlmostEqual(ref, uhd, places=6)
        self.assertAlmostEqual(ref, tall, places=6)

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

    def test_ground_truth_stereo_ignores_only_global_horizontal_offset(self):
        rng = np.random.default_rng(73)
        reference = rng.random((96, 160), dtype=np.float32)
        shifted = sbsbench._shift_x_edge(reference, 7)
        good = sbsbench.stereo_ground_truth_metrics(shifted, reference)
        vertically_wrong = np.roll(shifted, 4, axis=0)
        bad = sbsbench.stereo_ground_truth_metrics(vertically_wrong, reference)
        self.assertGreater(good["stereo_gt_psnr"], 80.0)
        self.assertGreater(good["stereo_gt_ssim"], 0.999)
        self.assertLess(good["stereo_gt_residual_p95"], 1.0)
        self.assertGreater(good["stereo_gt_coverage_pct"], 99.9)
        self.assertLess(bad["stereo_gt_psnr"], good["stereo_gt_psnr"] - 20.0)
        self.assertLess(bad["stereo_gt_ssim"], good["stereo_gt_ssim"] - 0.2)
        self.assertGreater(bad["stereo_gt_residual_p95"],
                           good["stereo_gt_residual_p95"] + 10.0)

    def test_ground_truth_stereo_detects_local_content_corruption(self):
        rng = np.random.default_rng(91)
        reference = rng.random((96, 160), dtype=np.float32)
        clean = sbsbench._shift_x_edge(reference, -5)
        corrupted = clean.copy()
        corrupted[24:72, 60:110] = 0.0
        good = sbsbench.stereo_ground_truth_metrics(clean, reference)
        bad = sbsbench.stereo_ground_truth_metrics(corrupted, reference)
        self.assertLess(bad["stereo_gt_psnr"], good["stereo_gt_psnr"] - 10.0)
        self.assertLess(bad["stereo_gt_ssim"], good["stereo_gt_ssim"] - 0.05)
        self.assertLess(bad["stereo_gt_coverage_pct"],
                        good["stereo_gt_coverage_pct"] - 5.0)

    def test_ground_truth_depth_lag_detects_previous_frame_geometry(self):
        previous = np.zeros((32, 48), np.float32)
        previous[8:24, 8:20] = 1.0
        current = np.zeros_like(previous)
        current[8:24, 18:30] = 1.0
        self.assertGreater(
            sbsbench.depth_ground_truth_lag(previous, current, previous), 50.0)
        self.assertEqual(
            sbsbench.depth_ground_truth_lag(current, current, previous), 0.0)

    def test_ground_truth_ghost_detects_previous_only_boundary(self):
        previous = np.zeros((32, 48), np.float32)
        previous[8:24, 8:20] = 1.0
        current = np.zeros_like(previous)
        current[8:24, 18:30] = 1.0
        self.assertEqual(
            sbsbench.depth_ground_truth_ghost(current, current, previous), 0.0)
        self.assertGreater(
            sbsbench.depth_ground_truth_ghost(previous, current, previous), 40.0)

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

    def test_nearest_flow_warp_preserves_depth_steps(self):
        previous = np.zeros((16, 24), np.float32)
        previous[:, 8:] = 1.0
        u = np.full_like(previous, 3.0)
        v = np.zeros_like(previous)
        warped, valid = sbsbench.warp_previous_nearest_with_flow(previous, u, v)
        self.assertTrue(valid[:, 3:].all())
        self.assertEqual(set(np.unique(warped)), {0.0, 1.0})
        self.assertTrue((warped[:, 11:] == 1.0).all())

    def test_flow_aware_ema_tracks_translated_depth_edge(self):
        height, width = 24, 40
        previous = np.zeros((height, width), np.float32)
        previous[:, 12:] = 1.0
        current = np.zeros_like(previous)
        current[:, 15:] = 1.0
        flow = np.zeros((height, width, 2), np.float32)
        flow[..., 0] = 3.0
        valid = np.ones((height, width), bool)
        filtered, reliable, _ = prepare_flow_ema_reference.flow_aware_ema(
            current, previous, previous, current, flow, valid,
            0.5, 0.05, 0.02, 0.25)
        self.assertGreater(float(reliable.mean()), 0.85)
        self.assertTrue(np.array_equal(filtered, current))

    def test_depth_confidence_ignores_flat_depth(self):
        source = np.tile(np.linspace(0.0, 1.0, 96, dtype=np.float32), (48, 1))
        depth = np.full((48, 96), 0.5, np.float32)
        result = audit_depth_confidence.depth_confidence_map(depth, source)
        self.assertFalse(result["band"].any())
        self.assertTrue(np.all(result["risk"] == 0.0))
        self.assertTrue(np.all(result["confidence"] == 1.0))

    def test_depth_confidence_prefers_sharp_aligned_edges(self):
        source = np.zeros((48, 96), np.float32)
        source[:, 48:] = 1.0
        sharp = np.zeros_like(source)
        sharp[:, 48:] = 1.0
        shifted = np.zeros_like(source)
        shifted[:, 56:] = 1.0
        soft = np.zeros_like(source)
        soft[:, 44:53] = np.linspace(0.0, 1.0, 9, dtype=np.float32)
        soft[:, 53:] = 1.0
        sharp_result = audit_depth_confidence.depth_confidence_map(sharp, source)
        shifted_result = audit_depth_confidence.depth_confidence_map(shifted, source)
        soft_result = audit_depth_confidence.depth_confidence_map(soft, source)
        sharp_risk = sharp_result["model_risk"]
        shifted_risk = shifted_result["model_risk"]
        soft_risk = soft_result["model_risk"]
        self.assertLess(float(sharp_risk.max()), 0.1)
        self.assertGreater(float(shifted_risk.max()), 0.6)
        self.assertGreater(float(soft_risk.max()), float(sharp_risk.max()) + 0.2)
        self.assertGreater(float(sharp_result["warp_risk"].max()), 0.5)

    def test_depth_confidence_detects_flow_compensated_temporal_change(self):
        rng = np.random.default_rng(91)
        source = rng.random((64, 128), dtype=np.float32)
        previous = np.zeros_like(source)
        previous[:, 48:] = 1.0
        current = previous.copy()
        current[16:48, 48:] = 0.25
        stable = audit_depth_confidence.depth_confidence_map(
            previous, source, previous_depth=previous, previous_src=source)
        changed = audit_depth_confidence.depth_confidence_map(
            current, source, previous_depth=previous, previous_src=source)
        valid = changed["band"] & changed["temporal_valid"]
        self.assertTrue(valid.any())
        self.assertLess(float(stable["temporal"].max()), 0.01)
        self.assertGreater(float(changed["temporal"][valid].max()), 0.9)

    def test_confidence_audit_auc_is_tie_aware(self):
        labels = np.array([False, True, False, True])
        self.assertEqual(
            audit_depth_confidence.rank_auc(np.array([0.0, 1.0, 0.0, 1.0]), labels), 1.0)
        self.assertEqual(
            audit_depth_confidence.rank_auc(np.ones(4), labels), 0.5)
        self.assertIsNone(
            audit_depth_confidence.rank_auc(np.arange(4), np.zeros(4, bool)))

    def test_confidence_audit_rejects_tiny_pixel_classes(self):
        risk = np.zeros((16, 16), np.float32)
        risk[:, 8:] = 1.0
        confidence = {"risk": risk, "band": np.ones_like(risk, bool)}
        severity = np.zeros_like(risk)
        severity[:2, :8] = 2.0  # only 16 artifact pixels despite perfect ranking
        row, _, _, _ = audit_depth_confidence.validation_row(
            confidence, severity, np.ones_like(risk, bool))
        self.assertEqual(row["artifact_positive_px"], 16)
        self.assertIsNone(row["artifact_auc"])

    def test_confidence_audit_fails_closed_when_gt_evidence_is_missing(self):
        rows = [{"artifact_auc": 0.8, "artifact_capture_pct": 90.0} for _ in range(4)]
        stats = audit_depth_confidence.calibration_decision(rows, 4, 4)
        self.assertTrue(stats["warp_screening_validated"])
        self.assertFalse(stats["model_boundary_validated"])
        self.assertEqual(stats["gt_auc_frames"], 0)
        rows[0]["gt_bad_edge_auc"] = 0.7
        rows[1]["gt_bad_edge_auc"] = 0.6
        stats = audit_depth_confidence.calibration_decision(rows, 4, 4)
        self.assertTrue(stats["model_boundary_validated"])

    def test_confidence_audit_allows_flat_gt_without_boundary_auc(self):
        rows = [{"artifact_auc": 0.8, "artifact_capture_pct": 90.0} for _ in range(4)]
        stats = audit_depth_confidence.calibration_decision(rows, 0, 4)
        self.assertTrue(stats["warp_screening_validated"])
        self.assertIsNone(stats["model_boundary_validated"])
        self.assertEqual(stats["gt_frames_available"], 4)
        self.assertEqual(stats["gt_frames_eligible"], 0)

    def test_confidence_audit_rejects_frame_identity_drift(self):
        with self.assertRaisesRegex(ValueError, "missing=\\[2\\], extra=\\[3\\]"):
            audit_depth_confidence.require_frame_ids("depth", [1, 2], [1, 3])

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

    def test_rescore_uses_canonical_metric_contract_hash(self):
        data = {"meta": {}}
        with mock.patch.object(run_eval, "metric_contract_sha", return_value="canonical"):
            rescore_run.refresh_contract_metadata(data)
        self.assertEqual(data["meta"]["metric_sha256"], "canonical")
        self.assertEqual(data["meta"]["eval_schema"], run_eval.EVAL_SCHEMA)

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
