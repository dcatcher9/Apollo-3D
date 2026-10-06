import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_adaptive_replay as replay


class AdaptiveReplayPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.build = self.root / "build"
        self.build.mkdir()
        self.exe = self.build / "sunshine.exe"
        self.exe.write_bytes(b"preflight-test-executable")
        self.conf = self.root / "mode3.conf"
        self.conf.write_text("sbs_3d_pop_strength = 1.2\n", encoding="utf-8")
        self.frames = self.root / "frames"
        self.frames.mkdir()
        self.timeline = self.root / "observation_timeline.sbsotl"
        replay.write_observation_timeline(self.timeline, [567890, 601223, 634556])
        self.model = "depth_anything_v2_fp16"
        self.environment = {"PATH": "production-runtime", "TEST_PREFLIGHT": "preserved"}
        self.subprocess = self.enterContext(mock.patch.object(replay.subprocess, "run"))
        self.env = self.enterContext(mock.patch.object(
            replay.run_eval, "production_subprocess_env", return_value=self.environment))
        self.engines = self.enterContext(mock.patch.object(
            replay.run_eval, "check_engines", return_value=[]))
        self.subprocess.return_value = mock.Mock(returncode=0, stdout="", stderr="")

    def run_preflight(self):
        replay.run_engine_preflight(
            self.exe, self.conf, self.build, self.frames, self.timeline, self.model)

    def test_uses_exact_one_timestamp_prefix_and_same_runtime_without_frame_copies(self):
        original_timeline = self.timeline.read_bytes()
        observed_paths = []

        def native(command, **kwargs):
            timeline = Path(command[command.index("--observation-timeline") + 1])
            out = Path(command[command.index("--out") + 1])
            observed_paths.append(timeline.parent)
            self.assertEqual(command, [
                str(self.exe), str(self.conf.resolve()), "--sbs-bench", "--frames",
                str(self.frames), "--out", str(out), "--limit", "1",
                "--observation-timeline", str(timeline),
            ])
            self.assertEqual(replay.read_observation_timeline(timeline), [567890])
            self.assertEqual(timeline.parent.parent, self.frames.parent)
            self.assertEqual(set(path.name for path in timeline.parent.iterdir()),
                             {"out", "observation_timeline.sbsotl"})
            self.assertEqual(kwargs, {
                "cwd": self.build, "capture_output": True, "text": True,
                "timeout": 900, "env": self.environment,
            })
            self.assertEqual(self.conf.read_text(encoding="utf-8"),
                             "sbs_3d_pop_strength = 1.2\n")
            self.engines.assert_not_called()
            return mock.Mock(returncode=0, stdout="ready", stderr="")

        self.subprocess.side_effect = native
        self.run_preflight()
        self.subprocess.assert_called_once()
        self.env.assert_called_once_with()
        self.engines.assert_called_once_with(str(self.build), self.model)
        self.assertEqual(self.timeline.read_bytes(), original_timeline)
        self.assertFalse(observed_paths[0].exists())

    def test_native_failure_does_not_authenticate_engines(self):
        self.subprocess.return_value = mock.Mock(
            returncode=7, stdout="native output\n", stderr="mode3 rejected context")
        with self.assertRaisesRegex(replay.EvidenceError, "exit 7.*") as caught:
            self.run_preflight()
        self.assertIn("mode3 rejected context", str(caught.exception))
        self.engines.assert_not_called()
        self.assertEqual(list(self.root.glob("sbs-engine-preflight-*")), [])

    def test_timeout_does_not_authenticate_engines(self):
        self.subprocess.side_effect = replay.subprocess.TimeoutExpired("native", 900)
        with self.assertRaisesRegex(replay.EvidenceError, "preflight timed out"):
            self.run_preflight()
        self.engines.assert_not_called()
        self.assertEqual(list(self.root.glob("sbs-engine-preflight-*")), [])

    def test_exact_engine_authentication_failure_is_not_waived(self):
        self.engines.return_value = ["wrong ONNX identity", "missing runtime manifest"]
        with self.assertRaisesRegex(replay.EvidenceError, "valid exact-engine manifest") as caught:
            self.run_preflight()
        self.assertIn("wrong ONNX identity; missing runtime manifest", str(caught.exception))
        self.engines.assert_called_once_with(str(self.build), self.model)

    def test_invalid_timeline_fails_before_native_submission(self):
        self.timeline.write_bytes(struct.pack(
            "<8sIIQQ", replay.OBSERVATION_TIMELINE_MAGIC,
            replay.OBSERVATION_TIMELINE_SCHEMA, replay.OBSERVATION_TIMELINE_HEADER_BYTES,
            1, 0))
        with self.assertRaisesRegex(replay.EvidenceError, "timestamps are zero"):
            self.run_preflight()
        self.subprocess.assert_not_called()
        self.env.assert_not_called()
        self.engines.assert_not_called()

    def test_main_passes_selected_timeline_to_local_preflight_before_measured_legs(self):
        for index in (1, 2):
            Image.new("RGB", (2, 2), color=(index, 2, 3)).save(
                self.frames / f"frame_{index:06d}.png")
        output = self.root / "replay"
        with mock.patch.object(replay.run_eval, "require_current_build"), \
                mock.patch.object(replay.run_eval, "expected_depth_model", return_value=self.model), \
                mock.patch.object(replay.run_eval, "run_engine_preflight") as generic, \
                mock.patch.object(replay, "run_harness") as harness, \
                mock.patch.object(replay, "run_engine_preflight", side_effect=
                                  replay.EvidenceError("preflight sentinel")) as preflight, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            result = replay.main([
                str(self.frames), "--build-dir", str(self.build), "--conf", str(self.conf),
                "--prepared-fps", "30/1", "--max-frames", "2", "--out", str(output),
            ])
        self.assertEqual(result, 2)
        self.assertIn("preflight sentinel", errors.getvalue())
        preflight.assert_called_once()
        exe, conf, build, frames, timeline, model = preflight.call_args.args
        self.assertEqual((exe, conf, build, model),
                         (self.exe.resolve(), self.conf.resolve(), self.build.resolve(), self.model))
        self.assertEqual(frames.parent.parent, output)
        self.assertEqual(timeline.parent, frames.parent)
        self.assertEqual(replay.read_observation_timeline(timeline), [1, 33334])
        generic.assert_not_called()
        harness.assert_not_called()
        self.subprocess.assert_not_called()


class AdaptiveReplayContractTests(unittest.TestCase):
    @staticmethod
    def mutate_trace_word(treatment: Path, frame_id: int, word: int, value: int):
        path = treatment / "device_conditional_gpu_trace_ring.u32"
        words = list(struct.unpack(f"<{path.stat().st_size // 4}I", path.read_bytes()))
        base = replay.TRACE_HEADER_WORDS + (frame_id - 1) * replay.TRACE_RECORD_WORDS
        words[base + word] = value
        path.write_bytes(struct.pack(f"<{len(words)}I", *words))

    def bind_authenticated_high_grid(
            self, control: Path, treatment: Path,
            source_shape=(1920, 1080), field_shape=(1540, 868)):
        """Upgrade the small trace fixture to one exact production fused-grid identity."""

        calibration = replay.depth_coordinate_v2_contract.MODEL_CALIBRATIONS[0]
        fused_contract = (
            replay.depth_coordinate_v2_dump_contract.convex2x_contract.load_contract())
        sources = fused_contract["sources"]
        fused = sources["fused_onnx"]
        recipe = fused_contract["tensorrt"]["engine_recipe"]
        raw_provenance = {
            "schema": 1,
            "model": calibration.depth_model,
            "depth_model_url": calibration.depth_model_url,
            "onnx_sha256": calibration.onnx_sha256,
            "preprocess_profile": calibration.preprocess.profile,
            "preprocess_source_closure_sha256":
                calibration.preprocess.source_closure_sha256,
            "raw_width": field_shape[0],
            "raw_height": field_shape[1],
        }
        composite = {
            "schema": fused_contract["schema"],
            "runtime": "dav2_zipdepth_convex2x_composite",
            "model": fused["logical_model"],
            "onnx_sha256": fused["sha256"],
            "embedded_dav2_onnx_sha256": sources["dav2"]["onnx_sha256"],
            "zipdepth_checkpoint_sha256": sources["zipdepth"]["checkpoint_sha256"],
            "guidance_preprocess_source_closure_sha256":
                calibration.preprocess.source_closure_sha256,
            "engine_recipe": recipe,
            "engine_artifact": (
                f"{fused['logical_model']}.{recipe}.cache-{'a' * 64}.engine"),
            "active_engine_manifest": f"{fused['logical_model']}.active-engine.json",
        }
        raw_shape = {
            "schema": 1,
            "width": field_shape[0],
            "height": field_shape[1],
            "dtype": "float32-le",
            "layout": "row-major",
            "stage": "raw model output before transform/normalization/EMA/curvature",
        }
        for directory in (control, treatment):
            path = directory / "contract.json"
            contract = json.loads(path.read_text(encoding="utf-8"))
            contract["raw_model_provenance"] = raw_provenance
            contract["composite_runtime_provenance"] = composite
            path.write_text(json.dumps(contract), encoding="utf-8")
            (directory / "raw_shape.json").write_text(
                json.dumps(raw_shape), encoding="utf-8")
        metadata_path = treatment / "device_conditional_replay.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["capture_match"].update({
            "source_width": source_shape[0],
            "source_height": source_shape[1],
            "field_width": field_shape[0],
            "field_height": field_shape[1],
        })
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        for frame_id in (1, 2):
            for word, value in (
                    (replay.TRACE_RECORD_SOURCE_WIDTH, source_shape[0]),
                    (replay.TRACE_RECORD_SOURCE_HEIGHT, source_shape[1]),
                    (replay.TRACE_RECORD_FIELD_WIDTH, field_shape[0]),
                    (replay.TRACE_RECORD_FIELD_HEIGHT, field_shape[1])):
                self.mutate_trace_word(treatment, frame_id, word, value)
        engine_sha256 = "e" * 64
        manifest_sha256 = "f" * 64
        return {
            "engine_name": composite["engine_artifact"],
            "engine_sha256": engine_sha256,
            "onnx_sha256": calibration.onnx_sha256,
            "preprocess_source_closure_sha256":
                calibration.preprocess.source_closure_sha256,
            "composite_runtime_provenance": {
                **composite,
                "engine_sha256": engine_sha256,
                "active_engine_manifest_sha256": manifest_sha256,
            },
        }

    def make_pair(self, root: Path, modes=("force", "reuse"), corrupt_hold=False,
                  invalid_hold_disposition=False, corrupt_target_hold=False,
                  corrupt_display=False, timestamps=None, analysis_generation=9):
        control = root / "control"
        treatment = root / "treatment"
        control.mkdir()
        treatment.mkdir()
        frame_count = len(modes)
        if timestamps is None:
            timestamps = [1 + index * 16_667 for index in range(frame_count)]
        self.assertEqual(len(timestamps), frame_count)
        timeline = root / "observation_timeline.sbsotl"
        replay.write_observation_timeline(timeline, timestamps)
        timeline_descriptor = {
            "schema": replay.OBSERVATION_TIMELINE_SCHEMA,
            "timestamp_unit": "monotonic-source-us-plus-one",
            "count": frame_count,
            "sha256": replay.sha256_file(timeline),
        }
        final_contract = replay.depth_coordinate_v2_contract.FINAL_PARALLAX
        shared = {
            "model": "depth_anything_v2_fp16",
            "pop_strength": 1.75,
            "cuda_graph": True,
            "cuda_graph_captured": True,
            "raw_model_provenance": {
                "sha256": "a" * 64, "raw_width": 2, "raw_height": 2,
            },
            "parallax_v2_live": {"producer": "b" * 64},
            "parallax_v2_shadow": False,
            "parallax_v2_render": True,
            "cut_state": {"schema": 5},
            "warp_mapping": {"schema": 1},
            "observation_timeline": timeline_descriptor,
            "adaptive_conditional": {
                "request_policy_schema": replay.ADAPTIVE_REQUEST_POLICY_SCHEMA,
                "near_identical_detector_source_closure_sha256":
                    replay.host_sbs_shader_manifest.NEAR_IDENTICAL_DETECTOR_GROUP.
                    source_closure_sha256,
            },
            "final_parallax_field": {
                "file_pattern": "final_parallax_<frame-id>.f32",
                "dtype": "float32-le",
                "layout": "row-major",
                "authority": final_contract.authority,
                "contract_schema": final_contract.schema,
                "publication_policy": final_contract.publication_policy,
                "reuse_policy": final_contract.reuse_policy,
                "invalid_policy": final_contract.invalid_policy,
                "current_rgb_policy": final_contract.current_rgb_policy,
            },
        }
        (control / "contract.json").write_text(json.dumps({
            **shared, "schema": replay.CONTROL_HARNESS_SCHEMA,
            "depth_step": "force-current-adaptive-replay",
            "depth_reuse_interval": 1,
            "device_conditional_replay_control": {
                "enabled": True, "scope": replay.CONTROL_SCOPE,
            },
        }), encoding="utf-8")

        force_count = sum(mode in ("force", "force_no_ocr", "suppress") for mode in modes)
        gpu_count = len(modes) - force_count
        (treatment / "contract.json").write_text(json.dumps({
            **shared, "schema": replay.CONDITIONAL_HARNESS_SCHEMA,
            "depth_step": "gpu-device-conditional",
            "depth_reuse_interval": None,
            "device_conditional_replay": {
                "enabled": True,
                "scope": replay.TREATMENT_SCOPE,
                "bootstrap": "force-infer",
                "followup": "known-publication-actual-depth-owner",
                "metadata": replay.METADATA_FILENAME,
                "raw_trace": replay.TRACE_FILENAME,
                "force_submissions": force_count,
                "gpu_undecided_submissions": gpu_count,
            },
        }), encoding="utf-8")

        capacity = replay.MAX_TRACE_FRAMES
        record_words = replay.TRACE_RECORD_WORDS
        words = [0] * (replay.TRACE_HEADER_WORDS + capacity * record_words)
        words[:8] = [
            replay.TRACE_RING_SCHEMA, replay.TRACE_RING_TAG, capacity, record_words,
            frame_count + 1, 0, frame_count, frame_count,
        ]
        locator = [0] * replay.TRACE_LOCATOR_WORDS
        locator[replay.TRACE_LOCATOR_CUT_EPOCH_WORD] = 7
        condition = [0] * replay.TRACE_CONDITION_WORDS
        subtitle_counts = {
            "suppressed": 0, "optional_ocr": 0,
            "abstention": 0, "held_with_depth": 0,
        }
        infer_count = 0
        reuse_count = 0
        previous_locator = locator.copy()
        previous_condition = condition.copy()
        row_subtitles = []
        for slot, mode in enumerate(modes):
            frame_id = slot + 1
            timestamp = timestamps[slot]
            row_locator = previous_locator.copy()
            row_condition = previous_condition.copy()
            if mode == "suppress":
                submission = replay.TRACE_SUBMISSION_FORCE
                depth = replay.TRACE_DEPTH_INFER
                expected_work = replay.WORK_NONE
                subtitle = replay.TRACE_SUBTITLE_SUPPRESSED
                flags = replay.TRACE_FLAG_SUBTITLE_SUPPRESSED
                host_outcome = replay.TRACE_HOST_SUBTITLE_SUPPRESSED
                optional_executed = False
                infer_count += 1
                subtitle_counts["suppressed"] += 1
            else:
                is_force = mode in ("force", "force_no_ocr")
                is_reuse = mode.startswith("reuse")
                if mode not in (
                        "force", "force_no_ocr", "opaque", "cut", "reset",
                        "reuse", "opaque_no_ocr", "reuse_no_ocr"):
                    raise AssertionError(mode)
                submission = (replay.TRACE_SUBMISSION_FORCE if is_force else
                              replay.TRACE_SUBMISSION_GPU_UNDECIDED)
                depth = replay.TRACE_DEPTH_REUSE if is_reuse else replay.TRACE_DEPTH_INFER
                optional_ready = not mode.endswith("_no_ocr")
                expected_work = (replay.WORK_OPTIONAL_OCR if optional_ready else
                                 replay.WORK_SUBTITLE_OBSERVATION)
                optional_executed = optional_ready and depth == replay.TRACE_DEPTH_INFER
                if depth == replay.TRACE_DEPTH_REUSE:
                    subtitle = replay.TRACE_SUBTITLE_HELD_WITH_DEPTH
                    subtitle_counts["held_with_depth"] += 1
                elif optional_executed:
                    subtitle = replay.TRACE_SUBTITLE_OPTIONAL_OCR
                    subtitle_counts["optional_ocr"] += 1
                else:
                    subtitle = replay.TRACE_SUBTITLE_ABSTENTION
                    subtitle_counts["abstention"] += 1
                flags = (replay.TRACE_FLAG_OCR_RECORD_SUBMITTED |
                         replay.TRACE_FLAG_CONDITION_EXECUTED if is_force else
                         replay.TRACE_FLAG_SUBTITLE_BRANCH_GATED)
                host_outcome = replay.TRACE_HOST_SUBTITLE_ORDINARY
                if mode == "reset":
                    flags |= replay.TRACE_FLAG_INPUT_DOMAIN_RESET
                if depth == replay.TRACE_DEPTH_INFER:
                    infer_count += 1
                else:
                    reuse_count += 1
                if subtitle != replay.TRACE_SUBTITLE_HELD_WITH_DEPTH:
                    row_locator[replay.TRACE_LOCATOR_FRAME_WORD] = frame_id
                    row_locator[replay.TRACE_LOCATOR_FRAME_WORD + 1] = 0
                    row_locator[20] = 1
                    row_locator[64:68] = [0, 0, 2, 2]
                    row_condition = [100 + frame_id + index
                                     for index in range(replay.TRACE_CONDITION_WORDS)]
                if mode == "cut":
                    row_locator[replay.TRACE_LOCATOR_CUT_EPOCH_WORD] += 1

            authentic_subtitle = subtitle
            if invalid_hold_disposition and subtitle == replay.TRACE_SUBTITLE_HELD_WITH_DEPTH:
                subtitle = replay.TRACE_SUBTITLE_INVALID
            base = replay.TRACE_HEADER_WORDS + slot * record_words
            words[base] = replay.TRACE_RING_SCHEMA
            words[base + 1] = replay.TRACE_RECORD_TAG
            words[base + 2] = frame_id
            words[base + replay.TRACE_RECORD_FRAME] = frame_id
            words[base + replay.TRACE_RECORD_ANALYSIS_GENERATION] = (
                analysis_generation & 0xFFFFFFFF)
            words[base + replay.TRACE_RECORD_ANALYSIS_GENERATION + 1] = (
                analysis_generation >> 32)
            words[base + replay.TRACE_RECORD_DOMAIN_TAG] = 0x1234
            words[base + replay.TRACE_RECORD_SUBMISSION] = submission
            words[base + replay.TRACE_RECORD_DEPTH] = depth
            words[base + replay.TRACE_RECORD_EXPECTED_WORK] = expected_work
            words[base + replay.TRACE_RECORD_SUBTITLE] = subtitle
            words[base + replay.TRACE_RECORD_FLAGS] = flags
            words[base + replay.TRACE_RECORD_HOST_SUBTITLE_OUTCOME] = host_outcome
            words[base + replay.TRACE_RECORD_SOURCE_WIDTH] = 2
            words[base + replay.TRACE_RECORD_SOURCE_HEIGHT] = 2
            words[base + replay.TRACE_RECORD_FIELD_WIDTH] = 2
            words[base + replay.TRACE_RECORD_FIELD_HEIGHT] = 2
            words[base + replay.TRACE_RECORD_TRANSACTION_WORDS] = replay.TRACE_TRANSACTION_WORDS
            token = frame_id
            branch = 0 if depth == replay.TRACE_DEPTH_REUSE else 1
            optional_marker = (replay.OPTIONAL_OCR_RECEIPT_MAGIC
                               if optional_executed else 0)
            transaction = [0] * replay.TRACE_TRANSACTION_WORDS
            transaction[:8] = [
                branch, branch ^ 0xD1EC15A5 ^ optional_marker,
                token, 0, token ^ 0xA3756C91, 0x5C8A936E,
                0x47524243, optional_marker,
            ]
            transaction[8:16] = [
                token, 0, token ^ 0xA3756C91, 0x5C8A936E,
                0x54535152, expected_work,
                0 if expected_work == 0 else expected_work ^ 0x6F435257, 0,
            ]
            words[
                base + replay.TRACE_RECORD_TRANSACTION_BEGIN:
                base + replay.TRACE_RECORD_TRANSACTION_BEGIN + replay.TRACE_TRANSACTION_WORDS
            ] = transaction
            if corrupt_hold and authentic_subtitle == replay.TRACE_SUBTITLE_HELD_WITH_DEPTH:
                row_locator[0] += 1
            words[
                base + replay.TRACE_RECORD_LOCATOR_BEGIN:
                base + replay.TRACE_RECORD_LOCATOR_BEGIN + replay.TRACE_LOCATOR_WORDS
            ] = row_locator
            words[
                base + replay.TRACE_RECORD_CONDITION_BEGIN:
                base + replay.TRACE_RECORD_CONDITION_BEGIN + replay.TRACE_CONDITION_WORDS
            ] = row_condition
            words[base + replay.TRACE_RECORD_OBSERVATION_TIMESTAMP] = timestamp & 0xFFFFFFFF
            words[base + replay.TRACE_RECORD_OBSERVATION_TIMESTAMP_HIGH] = timestamp >> 32
            previous_locator = row_locator
            previous_condition = row_condition
            row_subtitles.append(authentic_subtitle)
        trace_name = "device_conditional_gpu_trace_ring.u32"
        (treatment / trace_name).write_bytes(struct.pack(f"<{len(words)}I", *words))
        metadata = {
            "schema": replay.CONDITIONAL_METADATA_SCHEMA,
            "role": replay.TRACE_ROLE,
            "raw_trace": replay.TRACE_FILENAME,
            "ring": {
                "schema": replay.TRACE_RING_SCHEMA,
                "tag": replay.TRACE_RING_TAG,
                "capacity": capacity,
                "record_words": record_words,
                "committed_count": frame_count,
                "next_sequence": frame_count + 1,
            },
            "submission_counts": {"force": force_count, "gpu_undecided": gpu_count},
            "authenticated_device_dispositions": {
                "infer": infer_count, "reuse": reuse_count,
            },
            "authenticated_subtitle_dispositions": {
                **subtitle_counts,
            },
            "capture_match": {
                "matched_frame_id": frame_count,
                "analysis_generation": analysis_generation,
                "source_width": 2,
                "source_height": 2,
                "field_width": 2,
                "field_height": 2,
                "domain_tag": 0x1234,
                "input_domain_reset": bool(row_subtitles and
                    words[replay.TRACE_HEADER_WORDS +
                          (frame_count - 1) * record_words +
                          replay.TRACE_RECORD_FLAGS] &
                    replay.TRACE_FLAG_INPUT_DOMAIN_RESET),
            },
            "per_frame_artifact_scope": replay.PER_FRAME_ARTIFACT_SCOPE,
            "gpu_trace_source": {
                "closure_schema": replay.host_sbs_shader_manifest.SOURCE_CLOSURE_SCHEMA,
                "compile_flags": replay.host_sbs_shader_manifest.SHADER_COMPILE_FLAGS,
                "macro_count": replay.host_sbs_shader_manifest.SOURCE_MACRO_COUNT,
                "closure_sha256":
                    replay.host_sbs_shader_manifest.GPU_TRACE_GROUP.source_closure_sha256,
            },
        }
        (treatment / "device_conditional_replay.json").write_text(
            json.dumps(metadata), encoding="utf-8")

        treatment_final = None
        previous_raw = None
        for frame_id, (mode, row_subtitle) in enumerate(zip(modes, row_subtitles), 1):
            raw = np.full(4, frame_id / 100.0, dtype="<f4")
            if mode.startswith("reuse"):
                raw = previous_raw
            control_raw = np.full(4, frame_id / 100.0, dtype="<f4")
            # Long reuse chains still need legal synthetic parallax inside the V2 container.
            parallax_step = frame_id % 32
            control_final = np.array([
                parallax_step / 1000.0, -parallax_step / 1200.0,
                parallax_step / 1400.0, -parallax_step / 1600.0,
            ], dtype="<f4")
            if row_subtitle != replay.TRACE_SUBTITLE_HELD_WITH_DEPTH:
                treatment_final = control_final.copy()
            elif corrupt_target_hold:
                treatment_final = treatment_final.copy()
                treatment_final[0] = np.float32(treatment_final[0] + 0.001)
            if corrupt_display and frame_id == len(modes):
                treatment_final = treatment_final.copy()
                treatment_final[1] = np.float32(treatment_final[1] + 0.001)

            suffix = f"{frame_id:010d}.f32"
            (control / f"raw_{suffix}").write_bytes(control_raw.tobytes())
            (treatment / f"raw_{suffix}").write_bytes(raw.tobytes())
            (control / f"final_parallax_{suffix}").write_bytes(
                control_final.tobytes())
            (treatment / f"final_parallax_{suffix}").write_bytes(
                treatment_final.tobytes())
            previous_raw = raw

            image = Image.fromarray(
                np.full((9, 12, 3), (frame_id * 20) % 256, dtype=np.uint8))
            image.save(control / f"sbs_{frame_id:010d}.png")
            image.save(treatment / f"sbs_{frame_id:010d}.png")
        return control, treatment

    def validate_pair(self, control: Path, treatment: Path, count: int):
        metadata, records = replay._validate_contract_and_trace(
            control, treatment, count, control.parent / "observation_timeline.sbsotl")
        checks = replay.validate_adaptive_artifacts(control, treatment, records)
        return metadata, records, checks

    def test_authenticates_atomic_final_hold_and_direct_infer_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory))
            metadata, records, checks = self.validate_pair(control, treatment, 2)
            self.assertEqual(metadata["authenticated_device_dispositions"]["reuse"], 1)
            self.assertEqual([row["frame_id"] for row in records], [1, 2])
            self.assertEqual(checks["infer_current_raw_bit_exact_frames"], 1)
            self.assertEqual(checks["reuse_previous_raw_bit_exact_frames"], 1)
            self.assertEqual(
                checks["held_previous_final_parallax_bit_exact_frames"], 1)
            self.assertEqual(checks["authenticated_reuse_owner_ages"], {"2": 1})
            self.assertEqual(replay.final_field_gate(checks), {
                "status": "pass",
                "authenticated_reuse_frames": 1,
                "bit_exact_atomic_final_holds": 1,
            })
            comparison = replay.comparison_metrics(control, treatment, 2)
            self.assertEqual(comparison["summary"]["residual_mae"]["max"], 0.0)

    def test_authenticates_each_infer_branch_as_direct_atomic_publication(self):
        for mode in ("opaque", "cut", "reset"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                control, treatment = self.make_pair(
                    Path(directory), modes=("force", mode))
                _, _, checks = self.validate_pair(control, treatment, 2)
                self.assertEqual(checks["infer_current_raw_bit_exact_frames"], 2)
                self.assertEqual(
                    checks["held_previous_final_parallax_bit_exact_frames"], 0)
                self.assertEqual(
                    checks["transition_metrics"]["treatment"]
                    ["final_step_mae"]["count"], 1)

    def test_rejects_non_bit_exact_ordinary_reuse_locator_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory), corrupt_hold=True)
            with self.assertRaisesRegex(replay.EvidenceError, "bit-exactly hold"):
                replay._validate_contract_and_trace(control, treatment, 2)

    def test_rejects_treatment_infer_raw_that_differs_from_force_control(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory))
            _, records = replay._validate_contract_and_trace(control, treatment, 2)
            (treatment / "raw_0000000001.f32").write_bytes(
                np.ones(4, dtype="<f4").tobytes())
            with self.assertRaisesRegex(replay.EvidenceError, "differs from force control"):
                replay.validate_adaptive_artifacts(control, treatment, records)

    def test_rejects_ordinary_reuse_without_held_with_depth_disposition(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(
                Path(directory), invalid_hold_disposition=True)
            with self.assertRaisesRegex(replay.EvidenceError, "disposition disagrees"):
                replay._validate_contract_and_trace(control, treatment, 2)

    def test_rejects_non_bit_exact_atomic_final_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(
                Path(directory), corrupt_target_hold=True)
            _, records = replay._validate_contract_and_trace(control, treatment, 2)
            with self.assertRaisesRegex(replay.EvidenceError, "atomic final field"):
                replay.validate_adaptive_artifacts(control, treatment, records)

    def test_rejects_changed_final_field_on_ordinary_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(
                Path(directory), modes=("force", "reuse"),
                corrupt_display=True)
            _, records = replay._validate_contract_and_trace(control, treatment, 2)
            with self.assertRaisesRegex(replay.EvidenceError, "atomic final field"):
                replay.validate_adaptive_artifacts(control, treatment, records)

    def test_rejects_missing_or_misaligned_final_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory))
            _, records = replay._validate_contract_and_trace(control, treatment, 2)
            path = treatment / "final_parallax_0000000002.f32"
            path.unlink()
            with self.assertRaisesRegex(replay.EvidenceError, "do not cover"):
                replay.validate_adaptive_artifacts(control, treatment, records)
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory))
            _, records = replay._validate_contract_and_trace(control, treatment, 2)
            path = treatment / "final_parallax_0000000002.f32"
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(replay.EvidenceError, "misaligned"):
                replay.validate_adaptive_artifacts(control, treatment, records)

    def test_real_infer_replaces_the_owner_for_later_joint_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(
                Path(directory), modes=("force", "reuse", "reuse", "force", "reuse"),
                timestamps=[1, 11_112, 22_223, 33_334, 44_445])
            metadata, records, checks = self.validate_pair(control, treatment, 5)
            self.assertEqual(metadata["authenticated_device_dispositions"],
                             {"infer": 2, "reuse": 3})
            self.assertEqual(checks["authenticated_reuse_owner_ages"],
                             {"2": 1, "3": 2, "5": 1})
            self.assertEqual(checks["held_previous_final_parallax_bit_exact_frames"], 3)
            self.assertEqual([row["expected_work"] for row in records],
                             [replay.WORK_OPTIONAL_OCR] * 5)

    def test_joint_reuse_has_no_source_age_or_hold_count_expiration(self):
        count = 66
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(
                Path(directory), modes=("force",) + ("reuse",) * (count - 1),
                timestamps=[1 + index * (1 << 32) for index in range(count)])
            metadata, _, checks = self.validate_pair(control, treatment, count)
            self.assertEqual(metadata["authenticated_device_dispositions"],
                             {"infer": 1, "reuse": count - 1})
            self.assertEqual(checks["authenticated_reuse_owner_ages"],
                             {str(frame): frame - 1 for frame in range(2, count + 1)})
            self.assertEqual(checks["reuse_previous_raw_bit_exact_frames"], count - 1)
            self.assertEqual(checks["held_previous_final_parallax_bit_exact_frames"], count - 1)

    def test_changed_30fps_observations_can_reuse_the_same_infer_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(
                Path(directory), modes=("force", "reuse", "reuse", "reuse"),
                timestamps=[1, 33_334, 66_667, 100_001])
            metadata, _, checks = self.validate_pair(control, treatment, 4)
            self.assertEqual(metadata["authenticated_device_dispositions"],
                             {"infer": 1, "reuse": 3})
            self.assertEqual(checks["authenticated_reuse_owner_ages"],
                             {"2": 1, "3": 2, "4": 3})

    def test_long_joint_hold_still_rejects_changed_depth_or_final_field(self):
        for prefix, error in (
                ("raw_", "retain previous raw depth"),
                ("final_parallax_", "atomic final field")):
            with self.subTest(artifact=prefix), tempfile.TemporaryDirectory() as directory:
                control, treatment = self.make_pair(
                    Path(directory), modes=("force",) + ("reuse",) * 8,
                    timestamps=[1 + index * 1_000_000 for index in range(9)])
                path = treatment / f"{prefix}0000000009.f32"
                values = np.frombuffer(path.read_bytes(), dtype="<f4").copy()
                values[0] += np.float32(0.001)
                path.write_bytes(values.tobytes())
                with self.assertRaisesRegex(replay.EvidenceError, error):
                    self.validate_pair(control, treatment, 9)

    def test_long_joint_hold_still_rejects_changed_subtitle_state(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(
                Path(directory), modes=("force",) + ("reuse",) * 8,
                timestamps=[1 + index * 1_000_000 for index in range(9)])
            self.mutate_trace_word(
                treatment, 9, replay.TRACE_RECORD_CONDITION_BEGIN, 999)
            with self.assertRaisesRegex(replay.EvidenceError, "bit-exactly hold"):
                self.validate_pair(control, treatment, 9)

    def test_offline_joint_replay_still_rejects_subtitle_suppression(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(
                Path(directory), modes=("force", "suppress"))
            with self.assertRaisesRegex(replay.EvidenceError, "ordinary joint analysis work"):
                self.validate_pair(control, treatment, 2)

    def test_completed_conditional_infer_is_the_next_reuse_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(
                Path(directory), modes=("force", "opaque", "opaque", "reuse"),
                timestamps=[1, 1_001, 2_001, 3_001])
            _, _, checks = self.validate_pair(control, treatment, 4)
            self.assertEqual(checks["authenticated_reuse_owner_ages"], {"4": 1})
            self.assertEqual(checks["infer_current_raw_bit_exact_frames"], 3)

    def test_rejects_missing_or_regressed_reuse_owner_timestamps(self):
        for owner_time, current_time in ((0, 1), (1, 0), (2, 1)):
            with self.subTest(owner_time=owner_time, current_time=current_time):
                records = [
                    {"frame_id": 1, "depth": replay.TRACE_DEPTH_INFER,
                     "observation_timestamp_us": owner_time, "flags": 0},
                    {"frame_id": 2, "depth": replay.TRACE_DEPTH_REUSE,
                     "observation_timestamp_us": current_time, "flags": 0},
                ]
                with self.assertRaisesRegex(replay.EvidenceError, "time ordering"):
                    replay._authenticated_reuse_owner_ages(records)

    def test_force_without_ocr_infers_and_publishes_current_abstention(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(
                Path(directory), modes=("force", "force_no_ocr"), timestamps=[1, 40_001])
            metadata, records, checks = self.validate_pair(control, treatment, 2)
            self.assertEqual(records[1]["expected_work"], replay.WORK_SUBTITLE_OBSERVATION)
            self.assertEqual(records[1]["depth"], replay.TRACE_DEPTH_INFER)
            self.assertEqual(records[1]["subtitle"], replay.TRACE_SUBTITLE_ABSTENTION)
            self.assertEqual(metadata["authenticated_subtitle_dispositions"]["abstention"], 1)
            self.assertEqual(checks["held_previous_final_parallax_bit_exact_frames"], 0)

    def test_rejects_tampered_observation_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory))
            timeline = Path(directory) / "observation_timeline.sbsotl"
            timeline.write_bytes(timeline.read_bytes() + b"\0")
            with self.assertRaisesRegex(replay.EvidenceError, "timeline"):
                replay._validate_contract_and_trace(control, treatment, 2, timeline)

    def test_rejects_trace_timestamp_that_differs_from_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory))
            self.mutate_trace_word(
                treatment, 2, replay.TRACE_RECORD_OBSERVATION_TIMESTAMP, 12345)
            with self.assertRaisesRegex(replay.EvidenceError, "timestamps disagree"):
                replay._validate_contract_and_trace(control, treatment, 2)

    def test_accepts_zero_as_canonical_full_frame_analysis_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(
                Path(directory), analysis_generation=0)
            metadata, records = replay._validate_contract_and_trace(control, treatment, 2)
            self.assertEqual(metadata["capture_match"]["analysis_generation"], 0)
            self.assertEqual([record["analysis_generation"] for record in records], [0, 0])

    def test_rejects_analysis_generation_outside_unsigned_64_bit_range(self):
        for generation in (-1, replay.UINT64_MAX + 1):
            with self.subTest(generation=generation), tempfile.TemporaryDirectory() as directory:
                control, treatment = self.make_pair(Path(directory))
                metadata_path = treatment / replay.METADATA_FILENAME
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["capture_match"]["analysis_generation"] = generation
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaisesRegex(
                        replay.EvidenceError, "unsigned 64-bit value"):
                    replay._validate_contract_and_trace(control, treatment, 2)

    def test_rejects_retired_independent_due_work(self):
        for work in (8, 16):
            with self.subTest(work=work), tempfile.TemporaryDirectory() as directory:
                control, treatment = self.make_pair(Path(directory))
                self.mutate_trace_word(treatment, 2, replay.TRACE_RECORD_EXPECTED_WORK, work)
                with self.assertRaises(replay.EvidenceError):
                    replay._validate_contract_and_trace(control, treatment, 2)

    def test_rejects_previous_bounded_refresh_policy_even_when_pair_agrees(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory))
            for output in (control, treatment):
                path = output / "contract.json"
                contract = json.loads(path.read_text(encoding="utf-8"))
                contract["adaptive_conditional"]["request_policy_schema"] = 5
                path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(replay.EvidenceError, "adaptive request policy.*stale"):
                replay._validate_contract_and_trace(control, treatment, 2)

    def test_rejects_stale_trace_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory))
            trace = treatment / "device_conditional_gpu_trace_ring.u32"
            words = list(struct.unpack(f"<{trace.stat().st_size // 4}I", trace.read_bytes()))
            words[0] = 2
            trace.write_bytes(struct.pack(f"<{len(words)}I", *words))
            metadata_path = treatment / "device_conditional_replay.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["ring"]["schema"] = 2
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(replay.EvidenceError, "unexpected trace identity"):
                replay._validate_contract_and_trace(control, treatment, 2)

    def test_decodes_current_native_layout_and_complete_nonzero_adaptive_tail(self):
        self.assertEqual((
            replay.TRACE_RING_SCHEMA, replay.TRACE_RECORD_WORDS, replay.TRACE_LOCATOR_WORDS,
            replay.TRACE_RECORD_LOCATOR_BEGIN, replay.TRACE_RECORD_CONDITION_BEGIN,
            replay.TRACE_RECORD_OBSERVATION_TIMESTAMP,
            replay.TRACE_RECORD_OBSERVATION_TIMESTAMP_HIGH, replay.TRACE_RECORD_RESERVED_BEGIN,
        ), (4, 192, 96, 88, 184, 190, 191, 192))
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory))
            trace = treatment / replay.TRACE_FILENAME
            words = list(struct.unpack(f"<{trace.stat().st_size // 4}I", trace.read_bytes()))
            tail = list(range(0x40000000, 0x40000010))
            for slot in (0, 1):
                base = 16 + slot * 192
                words[base + 88 + 80:base + 88 + 96] = tail
            trace.write_bytes(struct.pack(f"<{len(words)}I", *words))
            _, records = replay._validate_contract_and_trace(control, treatment, 2)
            for record in records:
                self.assertEqual(len(record["locator"]), 96)
                self.assertEqual(record["locator"][80:96], tuple(tail))
                self.assertEqual(record["condition"], tuple(range(101, 107)))
            self.assertEqual([row["observation_timestamp_us"] for row in records], [1, 16668])

    def test_rejects_changed_first_or_last_adaptive_tail_word_on_reuse(self):
        for tail_word in (80, 95):
            with self.subTest(word=tail_word), tempfile.TemporaryDirectory() as directory:
                control, treatment = self.make_pair(Path(directory))
                self.mutate_trace_word(treatment, 2, 88 + tail_word, 0x40000000)
                with self.assertRaisesRegex(replay.EvidenceError, "bit-exactly hold SLR96"):
                    replay._validate_contract_and_trace(control, treatment, 2)

    def test_decodes_shifted_timestamp_low_and_high_words_and_rejects_mutations(self):
        timestamps = [(1 << 32) + 1, (1 << 32) + 33334]
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory), timestamps=timestamps)
            _, records = replay._validate_contract_and_trace(control, treatment, 2)
            self.assertEqual([row["observation_timestamp_us"] for row in records], timestamps)
            trace = treatment / replay.TRACE_FILENAME
            original = trace.read_bytes()
            for timestamp_word in (190, 191):
                with self.subTest(word=timestamp_word):
                    trace.write_bytes(original)
                    self.mutate_trace_word(treatment, 2, timestamp_word, 0)
                    with self.assertRaisesRegex(replay.EvidenceError, "media timeline"):
                        replay._validate_contract_and_trace(control, treatment, 2)

    def test_rejects_obsolete_layout_even_when_binary_and_metadata_agree(self):
        for schema, record_words in ((3, 192), (4, 176), (3, 176)):
            with self.subTest(schema=schema, words=record_words), \
                    tempfile.TemporaryDirectory() as directory:
                _, treatment = self.make_pair(Path(directory))
                trace = treatment / replay.TRACE_FILENAME
                native = list(struct.unpack(f"<{trace.stat().st_size // 4}I", trace.read_bytes()))
                words = native[:16] + [0] * (300 * record_words)
                words[0], words[3] = schema, record_words
                for slot in (0, 1):
                    row = native[16 + slot * 192:16 + (slot + 1) * 192]
                    if record_words == 176:
                        row = row[:168] + row[184:190] + row[190:192]
                    row[0] = schema
                    words[16 + slot * record_words:16 + (slot + 1) * record_words] = row
                trace.write_bytes(struct.pack(f"<{len(words)}I", *words))
                metadata = replay.load_json(treatment / replay.METADATA_FILENAME)
                metadata["ring"].update(schema=schema, record_words=record_words)
                with self.assertRaisesRegex(replay.EvidenceError, "unexpected trace identity"):
                    replay.decode_trace(trace, metadata, 2)

    def test_current_layout_still_rejects_header_record_and_size_mutations(self):
        mutations = (
            ("tag", 1, 0, "tag"),
            ("capacity", 2, 299, "capacity"),
            ("committed_count", 7, 1, "committed_count"),
            ("reserved_header", 8, 1, None),
            ("record_schema", 16, 3, None),
            ("torn_commit", 17, 0, None),
            ("reserved_record", 16 + 23, 1, None),
        )
        with tempfile.TemporaryDirectory() as directory:
            _, treatment = self.make_pair(Path(directory))
            trace = treatment / replay.TRACE_FILENAME
            original = trace.read_bytes()
            metadata = replay.load_json(treatment / replay.METADATA_FILENAME)
            for name, word, value, metadata_key in mutations:
                with self.subTest(mutation=name):
                    words = list(struct.unpack(f"<{len(original) // 4}I", original))
                    words[word] = value
                    trace.write_bytes(struct.pack(f"<{len(words)}I", *words))
                    changed_metadata = {**metadata, "ring": dict(metadata["ring"])}
                    if metadata_key:
                        changed_metadata["ring"][metadata_key] = value
                    with self.assertRaises(replay.EvidenceError):
                        replay.decode_trace(trace, changed_metadata, 2)
            for name, payload in (("truncated", original[:-4]), ("extra_word", original + b"\0" * 4)):
                with self.subTest(mutation=name):
                    trace.write_bytes(payload)
                    with self.assertRaisesRegex(replay.EvidenceError, "GPU trace byte size"):
                        replay.decode_trace(trace, metadata, 2)
            trace.write_bytes(original)
            changed_metadata = {**metadata, "ring": {**metadata["ring"], "schema": 3}}
            with self.assertRaisesRegex(replay.EvidenceError, "trace header disagrees with metadata"):
                replay.decode_trace(trace, changed_metadata, 2)

    def test_stages_numeric_subset_once_with_canonical_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "frame_10.png").write_bytes(b"ten")
            (source / "frame_2.png").write_bytes(b"two")
            (source / "frame_1.png").write_bytes(b"one")
            staged = replay.stage_prepared_corpus(source, root / "staged", 2)
            self.assertEqual(
                [path.name for path in staged],
                ["frame_000001.png", "frame_000002.png"])
            self.assertEqual([path.read_bytes() for path in staged], [b"one", b"two"])

    def test_public_validation_binds_exact_source_derived_fused_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory))
            runtime_identity = self.bind_authenticated_high_grid(control, treatment)
            metadata, records = replay.validate_contract_and_trace(
                control, treatment, 2,
                Path(directory) / "observation_timeline.sbsotl", (1920, 1080),
                runtime_identity)
            self.assertEqual(metadata["capture_match"]["field_width"], 1540)
            self.assertEqual(records[0]["field_width"], 1540)

    def test_public_validation_rejects_same_byte_count_transpose(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory))
            runtime_identity = self.bind_authenticated_high_grid(
                control, treatment, field_shape=(868, 1540))
            with self.assertRaisesRegex(replay.EvidenceError, "preflight-selected fused runtime"):
                replay.validate_contract_and_trace(
                    control, treatment, 2,
                    Path(directory) / "observation_timeline.sbsotl", (1920, 1080),
                    runtime_identity)

    def test_public_validation_rejects_missing_fused_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory))
            runtime_identity = self.bind_authenticated_high_grid(control, treatment)
            for path in (control / "contract.json", treatment / "contract.json"):
                contract = json.loads(path.read_text(encoding="utf-8"))
                del contract["composite_runtime_provenance"]
                path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(replay.EvidenceError, "composite runtime provenance"):
                replay.validate_contract_and_trace(
                    control, treatment, 2,
                    Path(directory) / "observation_timeline.sbsotl", (1920, 1080),
                    runtime_identity)

    def test_public_validation_rejects_preflight_fused_to_legacy_downgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory))
            runtime_identity = self.bind_authenticated_high_grid(control, treatment)
            for output in (control, treatment):
                contract_path = output / "contract.json"
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
                contract["raw_model_provenance"]["raw_width"] = 770
                contract["raw_model_provenance"]["raw_height"] = 434
                contract["composite_runtime_provenance"] = None
                contract_path.write_text(json.dumps(contract), encoding="utf-8")
                shape_path = output / "raw_shape.json"
                shape = json.loads(shape_path.read_text(encoding="utf-8"))
                shape["width"], shape["height"] = 770, 434
                shape_path.write_text(json.dumps(shape), encoding="utf-8")
            metadata_path = treatment / replay.METADATA_FILENAME
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["capture_match"]["field_width"] = 770
            metadata["capture_match"]["field_height"] = 434
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            for frame_id in (1, 2):
                self.mutate_trace_word(
                    treatment, frame_id, replay.TRACE_RECORD_FIELD_WIDTH, 770)
                self.mutate_trace_word(
                    treatment, frame_id, replay.TRACE_RECORD_FIELD_HEIGHT, 434)
            with self.assertRaisesRegex(replay.EvidenceError, "downgraded"):
                replay.validate_contract_and_trace(
                    control, treatment, 2,
                    Path(directory) / "observation_timeline.sbsotl", (1920, 1080),
                    runtime_identity)

    def test_rejects_arbitrary_shader_provenance_and_out_of_tree_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory))
            for output in (control, treatment):
                contract_path = output / "contract.json"
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
                contract["adaptive_conditional"][
                    "near_identical_detector_source_closure_sha256"] = "0" * 64
                contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(replay.EvidenceError, "detector identity"):
                replay._validate_contract_and_trace(control, treatment, 2)
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory))
            metadata_path = treatment / replay.METADATA_FILENAME
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["gpu_trace_source"] = {
                "macro_count": 0, "closure_sha256": "1" * 64,
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(replay.EvidenceError, "shader provenance"):
                replay._validate_contract_and_trace(control, treatment, 2)
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory))
            contract_path = treatment / "contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["device_conditional_replay"]["raw_trace"] = "../shadow/trace.u32"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(replay.EvidenceError, "replay authority"):
                replay._validate_contract_and_trace(control, treatment, 2)

    def test_public_validation_binds_contract_to_actual_preflight_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(Path(directory))
            runtime_identity = self.bind_authenticated_high_grid(control, treatment)
            for output in (control, treatment):
                contract_path = output / "contract.json"
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
                contract["composite_runtime_provenance"]["engine_artifact"] = (
                    contract["composite_runtime_provenance"]["engine_artifact"].replace(
                        f".cache-{'a' * 64}", f".cache-{'b' * 64}"))
                contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(replay.EvidenceError, "preflight-selected fused engine"):
                replay.validate_contract_and_trace(
                    control, treatment, 2,
                    Path(directory) / "observation_timeline.sbsotl", (1920, 1080),
                    runtime_identity)

    def test_reports_aligned_per_stage_timing_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "control"
            treatment = root / "treatment"
            control.mkdir()
            treatment.mkdir()
            base = {
                "min_ms": 1.0, "p50_ms": 2.0, "p95_ms": 3.0,
                "max_ms": 4.0, "mean_ms": 2.5, "n": 300, "total": 300,
            }
            changed = {
                "min_ms": 0.5, "p50_ms": 1.5, "p95_ms": 2.5,
                "max_ms": 3.5, "mean_ms": 2.0, "n": 300, "total": 300,
            }
            (control / "sbs_perf.json").write_text(
                json.dumps({"stages": {"transaction": base}}), encoding="utf-8")
            (treatment / "sbs_perf.json").write_text(
                json.dumps({"stages": {"transaction": changed}}), encoding="utf-8")
            report = replay.performance_comparison(control, treatment)
            self.assertAlmostEqual(
                report["stages"]["transaction"]["mean_delta_percent"], -20.0)
            self.assertAlmostEqual(
                report["stages"]["transaction"]["p95_delta_ms"], -0.5)

    def test_authenticates_manifest_selected_model_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            assets = build / "assets"
            models = assets / "models"
            models.mkdir(parents=True)
            onnx = b"contract-onnx"
            onnx_sha = hashlib.sha256(onnx).hexdigest()
            (models / "ocr.onnx").write_bytes(onnx)
            (assets / "ocr.engine").write_bytes(b"engine")
            (assets / "ocr.active-engine.json").write_text(json.dumps({
                "schema": 1,
                "model": "ocr",
                "engine": "ocr.engine",
                "onnx_sha256": onnx_sha,
            }), encoding="utf-8")
            provenance = replay.model_artifact_provenance(
                build, "ocr", "models/ocr.onnx", onnx_sha)
            self.assertEqual(provenance["onnx_sha256"], onnx_sha)
            self.assertEqual(provenance["engine_name"], "ocr.engine")


if __name__ == "__main__":
    unittest.main()
