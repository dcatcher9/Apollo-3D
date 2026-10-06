"""Native adaptive trace admission; authored records never claim GPU execution."""

import copy
from dataclasses import asdict
import hashlib
import struct
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import depth_mapping_v2_temporal as temporal
import depth_coordinate_v2_contract as coordinate
import depth_coordinate_v2_dump_contract as dump
from depth_mapping_v2 import MappingV2Config


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def authored_trace():
    contract = coordinate.load_contract()
    tag = dump.generator.contract_tag(contract)
    descriptors = contract["shadow_state"]["fields"]
    config = MappingV2Config()
    rows = []
    for index in range(2):
        row = {name: 0.0 for name in temporal.V2_STATE_TRACE_FIELDS}
        row.update(frame_id=f"{index + 1:05d}", observation_timestamp_us=1 + index * 33333,
                   joint_plane_mode=3, input_source_frame_id=f"{index + 1:05d}",
                   rendered_source_frame_id=f"{index + 1:05d}", calibration_revision=index + 1,
                   confirmed_cut_count=index, confirmed_cut=bool(index),
                   cut_attribution="authored-cut" if index else "initialization",
                   input_valid=True, frame_valid=True, camera_valid=True, collapsed=False,
                   center=3.5, inverse_scale=f32(0.4), latched_scale=f32(2.5),
                   convergence_curve=0.0, requested_gain=f32(config.parallax_gain),
                   container_scale=1.0, effective_gain=f32(config.parallax_gain),
                   observed_mean=3.5, observed_std=2.0, observed_raw_minimum=1.0,
                   observed_raw_maximum=6.0, observed_percentile_low=1.0,
                   observed_percentile_high=6.0, observed_percentile_valid=1.0,
                   observed_percentile_bin_width=5.0 / 256)
        values = {field["name"]: 0 for field in descriptors}
        values.update(center=3.5, inverse_scale=f32(.4), convergence_curve=0.0,
                      container_scale=1.0, calibration_revision=index + 1,
                      frame_valid=1.0, confirmed_cut_count=index, contract_tag_bits=tag,
                      renderer_authorization_bits=tag, joint_plane_mode_bits=3,
                      gain_last_observation_low=row["observation_timestamp_us"],
                      gain_clock_armed=1, gain_seed_count=1, gain_target_zero=3.5,
                      gain_target_inverse_scale=f32(.4), gain_target_nearest=2.5,
                      gain_display_limit=f32(.04), gain_seed_first_low=1,
                      gain_seed_last_low=1, gain_seed_mean_nearest=2.5, gain_seed_mean_zero=3.5)
        values["camera_center_integrity_bits"] = dump.camera_center_integrity_for_state_values(values)
        row["state_words_u32"] = [
            struct.unpack("<I", struct.pack("<f", values[field["name"]]))[0]
            if field["gpu_encoding"] == "float" else values[field["name"]]
            for field in descriptors]
        for name in ("order_sha256", "vertical_majorant_sha256",
                     "vertical_conditioned_sha256", "parallax_sha256"):
            row[name] = "0" * 64
        rows.append(row)
    shape = coordinate.MODEL_CALIBRATIONS[0].calibrated_input_shapes[0]
    return {
        "schema": temporal.V2_STATE_TRACE_SCHEMA, "policy": temporal.V2_STATE_TRACE_POLICY,
        "calibration_contract": {
            "file": coordinate.CONTRACT_PATH.relative_to(coordinate.CONTRACT_PATH.parent.parent).as_posix(),
            "schema": coordinate.CONTRACT_SCHEMA,
            "sha256": hashlib.sha256(coordinate.CONTRACT_PATH.read_bytes()).hexdigest(),
        },
        "cut_source": "authored-cut", "diagnostic_role": "non-controlling-mode-selected-camera-audit-v6",
        "diagnostic_method": "gpu-frame-moments-and-rendered-fields-v6",
        "mapping_config": {**asdict(config), "joint_plane_mode": 3},
        "adaptation_semantics": temporal._trace_adaptation_semantics(3),
        "frame_fields": list(temporal.V2_STATE_TRACE_FIELDS), "frames": rows,
        "producer": {
            "authority": "authenticated-raw-depth-plus-eight-v2-compute-shaders-persistent-gpu-state-v11",
            "manifest_sha256": "0" * 64, "contract_canonical_sha256": coordinate.CONTRACT_CANONICAL_SHA256,
            "tensor_shape": {"width": shape[0], "height": shape[1]},
            "shader_sequence": list(temporal.V2_GPU_SHADER_SEQUENCE),
            "state_persistence": "single-buffer-whole-sequence",
            "numpy_role": "comparison-only-not-render-authority",
        },
    }


class NativeAdaptiveTraceTests(unittest.TestCase):
    def setUp(self):
        self.trace = authored_trace()

    def test_native_admission_keeps_camera_across_a_confirmed_cut(self):
        temporal.validate_v2_state_trace(self.trace, [1, 2])
        self.assertEqual(self.trace["frames"][0]["center"], self.trace["frames"][1]["center"])

    def test_removed_pipeline_cannot_be_admitted_as_another_mode(self):
        for mode in (0, 1, 2, 4, True):
            bad = copy.deepcopy(self.trace)
            bad["mapping_config"]["joint_plane_mode"] = mode
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                temporal.validate_v2_state_trace(bad, [1, 2])

    def test_old_curve_parameters_are_unknown_mapping_fields(self):
        for name in ("far_tau", "near_log_tau"):
            bad = copy.deepcopy(self.trace)
            bad["mapping_config"][name] = 0.5
            with self.subTest(name=name), self.assertRaises(ValueError):
                temporal.validate_v2_state_trace(bad, [1, 2])

    def test_numpy_cannot_claim_native_temporal_or_render_authority(self):
        bad = copy.deepcopy(self.trace)
        bad["producer"] = {"authority": "numpy-reference-comparison-only-v3",
                           "numpy_role": "comparison-only-not-render-authority"}
        with self.assertRaisesRegex(ValueError, "producer authority"):
            temporal.validate_v2_state_trace(bad, [1, 2])

    def test_changed_source_clock_cannot_authorize_a_repeated_observation(self):
        bad = copy.deepcopy(self.trace)
        bad["frames"][1]["observation_timestamp_us"] = 1
        with self.assertRaisesRegex(ValueError, "source evidence"):
            temporal.validate_v2_state_trace(bad, [1, 2])

    def test_current_json_view_must_match_sealed_gpu_words(self):
        for key in ("center", "inverse_scale", "calibration_revision", "confirmed_cut_count"):
            bad = copy.deepcopy(self.trace)
            bad["frames"][0][key] += 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "authenticated state"):
                temporal.validate_v2_state_trace(bad, [1, 2])

    def test_raw_distribution_cannot_masquerade_as_the_owned_amplitude_target(self):
        bad = copy.deepcopy(self.trace)
        bad["frames"][1]["observed_percentile_high"] = 5.0
        with self.assertRaisesRegex(ValueError, "captured moments"):
            temporal.validate_v2_state_trace(bad, [1, 2])

    def test_incomplete_or_reordered_shader_authority_is_rejected(self):
        bad = copy.deepcopy(self.trace)
        bad["producer"]["shader_sequence"].pop(2)
        with self.assertRaisesRegex(ValueError, "native GPU"):
            temporal.validate_v2_state_trace(bad, [1, 2])
