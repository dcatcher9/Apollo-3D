import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_adaptive_replay as replay


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
                f"{fused['logical_model']}.{recipe}.fixture-"
                f"onnx{fused['sha256']}.engine"),
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

        force_count = sum(mode in ("force", "suppress") for mode in modes)
        gpu_count = len(modes) - force_count
        (treatment / "contract.json").write_text(json.dumps({
            **shared, "schema": replay.CONDITIONAL_HARNESS_SCHEMA,
            "depth_step": "gpu-device-conditional",
            "depth_reuse_interval": None,
            "device_conditional_replay": {
                "enabled": True,
                "scope": replay.TREATMENT_SCOPE,
                "bootstrap": "force-infer",
                "followup": "gpu-owned-infer-or-reuse",
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
        last_guaranteed = 0
        dirty_holds = 0
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
                last_guaranteed = 0
                dirty_holds = 0
            else:
                is_force = mode == "force"
                is_reuse = mode.startswith("reuse")
                if mode not in (
                        "force", "opaque", "cut", "reset",
                        "reuse", "opaque_no_ocr", "reuse_no_ocr"):
                    raise AssertionError(mode)
                submission = (replay.TRACE_SUBMISSION_FORCE if is_force else
                              replay.TRACE_SUBMISSION_GPU_UNDECIDED)
                depth = replay.TRACE_DEPTH_REUSE if is_reuse else replay.TRACE_DEPTH_INFER
                optional_ready = not mode.endswith("_no_ocr")
                due = (last_guaranteed == 0 or timestamp < last_guaranteed or
                       timestamp - last_guaranteed >= replay.OCR_MAX_OBSERVATION_AGE_US or
                       dirty_holds >= replay.OCR_MAX_DIRTY_HOLDS)
                expected_work = (
                    replay.WORK_OPTIONAL_OCR_DUE if due and optional_ready else
                    replay.WORK_SUBTITLE_OBSERVATION_DUE if due else
                    replay.WORK_OPTIONAL_OCR if optional_ready else
                    replay.WORK_SUBTITLE_OBSERVATION)
                optional_executed = optional_ready and (
                    expected_work == replay.WORK_OPTIONAL_OCR_DUE or
                    (expected_work == replay.WORK_OPTIONAL_OCR and
                     depth == replay.TRACE_DEPTH_INFER))
                if (expected_work in (replay.WORK_OPTIONAL_OCR,
                                     replay.WORK_SUBTITLE_OBSERVATION) and
                        depth == replay.TRACE_DEPTH_REUSE):
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
                if due or is_force:
                    last_guaranteed = timestamp
                    dirty_holds = 0
                else:
                    dirty_holds = min(dirty_holds + 1, replay.OCR_MAX_DIRTY_HOLDS)

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
            words[base + replay.TRACE_RECORD_OBSERVATION_TIMESTAMP + 1] = timestamp >> 32
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
            control_final = np.array([
                frame_id / 1000.0, -frame_id / 1200.0,
                frame_id / 1400.0, -frame_id / 1600.0,
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
                np.full((9, 12, 3), frame_id * 20, dtype=np.uint8))
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
                "independent_subtitle_publications_on_reuse": 0,
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

    def test_accepts_four_reuses_with_due_subtitle_publications(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(
                Path(directory), modes=("force", "reuse", "reuse", "reuse", "reuse"))
            metadata, records, checks = self.validate_pair(control, treatment, 5)
            self.assertEqual(metadata["authenticated_device_dispositions"]["reuse"], 4)
            self.assertEqual(
                checks["authenticated_reuse_owner_ages"],
                {"2": 1, "3": 2, "4": 3, "5": 4})
            self.assertEqual(checks["held_previous_final_parallax_bit_exact_frames"], 2)
            self.assertEqual(checks["reuse_subtitle_publication_frames"], 2)
            self.assertEqual(
                [row["expected_work"] for row in records],
                [replay.WORK_OPTIONAL_OCR_DUE, replay.WORK_OPTIONAL_OCR,
                 replay.WORK_OPTIONAL_OCR_DUE, replay.WORK_OPTIONAL_OCR,
                 replay.WORK_OPTIONAL_OCR_DUE])

    def test_rejects_more_than_four_consecutive_reuses(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(
                Path(directory), modes=(
                    "force", "reuse", "reuse", "reuse", "reuse", "reuse"))
            with self.assertRaisesRegex(replay.EvidenceError, "history-owner age"):
                replay._validate_contract_and_trace(control, treatment, 6)

    def test_rejects_reuse_at_strict_100ms_owner_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(
                Path(directory), timestamps=[1, 100_001])
            with self.assertRaisesRegex(replay.EvidenceError, "strict authenticated.*time bound"):
                replay._validate_contract_and_trace(control, treatment, 2)

    def test_accepts_due_abstention_publication_on_reused_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(
                Path(directory), modes=("force", "reuse_no_ocr"),
                timestamps=[1, 40_001])
            metadata, records, checks = self.validate_pair(control, treatment, 2)
            self.assertEqual(records[1]["expected_work"],
                             replay.WORK_SUBTITLE_OBSERVATION_DUE)
            self.assertEqual(records[1]["subtitle"], replay.TRACE_SUBTITLE_ABSTENTION)
            self.assertEqual(metadata["authenticated_subtitle_dispositions"]["abstention"], 1)
            self.assertEqual(checks["reuse_subtitle_publication_frames"], 1)
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

    def test_rejects_work_that_skips_shared_due_cadence(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.make_pair(
                Path(directory), modes=("force", "reuse", "reuse"))
            self.mutate_trace_word(
                treatment, 3, replay.TRACE_RECORD_EXPECTED_WORK,
                replay.WORK_OPTIONAL_OCR)
            with self.assertRaisesRegex(replay.EvidenceError, "due-OCR cadence"):
                replay._validate_contract_and_trace(control, treatment, 3)

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
                        ".fixture-", ".other-"))
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
