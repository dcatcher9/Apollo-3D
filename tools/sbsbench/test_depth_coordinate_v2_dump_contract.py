#!/usr/bin/env python3
"""Strict JSON-dump tests for the generated algorithm and independent dump schemas."""

import copy
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import depth_coordinate_v2_contract as coordinate  # noqa: E402
import depth_coordinate_v2_dump_contract as dump_contract  # noqa: E402
import generate_depth_coordinate_v2_contract as generator  # noqa: E402
from replay_depth_mapping_v2 import (  # noqa: E402
    _authenticate_replay_coordinate_scale,
    _inspect_optional_shadow_state,
    _inspect_optional_v2_dump_manifest,
    _require_supported_replay_domain,
)


PRODUCTION_SOURCE_PROFILE_CASES = (
    (1920, 1080, 770, 434),
    (2560, 1080, 1022, 434),
    (3440, 1440, 1036, 434),
    (1080, 1920, 434, 770),
    (1080, 2560, 434, 1022),
    (1440, 3440, 434, 1036),
    (2048, 1536, 574, 434),
    (2388, 1668, 616, 434),
    (2360, 1640, 630, 434),
    (2266, 1488, 658, 434),
    (2560, 1600, 700, 434),
    (2160, 1080, 868, 434),
    (2340, 1080, 938, 434),
    (2400, 1080, 966, 434),
    (2520, 1120, 980, 434),
    (1536, 2048, 434, 574),
    (1668, 2388, 434, 616),
    (1640, 2360, 434, 630),
    (1488, 2266, 434, 658),
    (1600, 2560, 434, 700),
    (1080, 2160, 434, 868),
    (1080, 2340, 434, 938),
    (1080, 2400, 434, 966),
    (1120, 2520, 434, 980),
)


class DepthCoordinateV2DumpContractTests(unittest.TestCase):
    def test_renderer_closure_constants_match_native_authenticated_pins(self):
        pins = generator.validate_renderer_source_closure_pins()
        self.assertEqual(
            dump_contract.LIVE_RENDERER_SOURCE_CLOSURE_SHA256,
            pins["parallax_v2_live_renderer_source_closure_sha256"],
        )
        self.assertEqual(
            dump_contract.DIAGNOSTIC_SOURCE_CLOSURE_SHA256,
            pins["parallax_v2_diagnostic_source_closure_sha256"],
        )

    @staticmethod
    def _set_state_word(document, name, value):
        document["named_values"][name] = value
        field = next(item for item in document["fields"] if item["name"] == name)
        field["value"] = value

    @classmethod
    def _seal_camera_center(cls, document):
        checksum = dump_contract.camera_center_integrity_for_state_values(
            document["named_values"])
        cls._set_state_word(document, "camera_center_integrity_bits", checksum)
        document["decoded"]["camera_center_integrity_bits"] = checksum
        authorization = (document["decoded"]["contract_tag"]
                         if document["named_values"]["frame_valid"] > 0.5 else 0)
        cls._set_state_word(document, "renderer_authorization_bits", authorization)
        document["decoded"]["renderer_authorization_bits"] = authorization


    @staticmethod
    def _state_words(document):
        return [struct.unpack("<I", struct.pack("<f", field["value"]))[0]
                if field["type"] == "float32" else field["value"]
                for field in document["fields"]]

    def _host_adapter_state(self):
        document = copy.deepcopy(self.state)
        self._set_state_word(document, "joint_plane_mode_bits", 3)
        changes = {
            "gain_last_observation_low": 1000001, "gain_clock_armed": 1,
            "gain_seed_count": 1, "gain_target_zero": document["named_values"]["center"],
            "gain_target_inverse_scale": document["named_values"]["inverse_scale"],
            "gain_target_nearest": document["decoded"]["latched_scale"],
            "gain_display_limit": dump_contract._float32(0.04),
            "gain_seed_first_low": 1000001, "gain_seed_last_low": 1000001,
            "gain_seed_mean_nearest": document["decoded"]["latched_scale"],
            "gain_seed_mean_zero": document["named_values"]["center"],
        }
        for name, value in changes.items():
            self._set_state_word(document, name, value)
        document["decoded"]["joint_plane_mode"] = 3
        document["adaptation_semantics"] = dump_contract.adaptation_semantics_for_mode(3)
        document["units"]["gain"] = "one-eye source-U per coordinate unit"
        self._seal_camera_center(document)
        return document

    @staticmethod
    def _set_manifest_capture_grid(manifest, width, height):
        for name in (
                "shadow_coordinate", "shadow_candidate_parallax",
                "shadow_vertical_majorant",
                "shadow_vertical_conditioned", "shadow_final_parallax"):
            manifest["dimensions"][name].update({"width": width, "height": height})
        manifest["dimensions"]["model_input"].update({"width": width, "height": height})
        manifest["dimensions"]["raw_depth"].update({"width": width, "height": height})

    @staticmethod
    def _composite_runtime_provenance():
        contract = dump_contract.convex2x_contract.load_contract()
        fused = contract["sources"]["fused_onnx"]
        model = fused["logical_model"]
        onnx_sha256 = fused["sha256"]
        recipe = contract["tensorrt"]["engine_recipe"]
        return {
            "schema": contract["schema"],
            "runtime": "dav2_zipdepth_convex2x_composite",
            "model": model,
            "onnx_sha256": onnx_sha256,
            "embedded_dav2_onnx_sha256":
                contract["sources"]["dav2"]["onnx_sha256"],
            "zipdepth_checkpoint_sha256":
                contract["sources"]["zipdepth"]["checkpoint_sha256"],
            "guidance_preprocess_source_closure_sha256":
                coordinate.MODEL_CALIBRATIONS[0].preprocess.source_closure_sha256,
            "engine_recipe": recipe,
            "engine_artifact": f"{model}.{recipe}.cache-{'a' * 64}.engine",
            "active_engine_manifest": f"{model}.active-engine.json",
        }

    def setUp(self):
        # Production readers reject every non-calibrated tensor extent. Arithmetic/byte-chain
        # tests keep their historical 16x12 payloads through this private test-only monkeypatch;
        # no production validation entry point accepts a fixture-mode flag.
        self.production_capture_grid_relation = dump_contract._capture_grid_relation
        self.production_source_capture_grid_relation = (
            dump_contract.capture_grid_relation_for_source)

        def fixture_capture_grid_relation(width, height):
            if (width, height) == (16, 12):
                return "test-only-fixture", 1, width, height
            return self.production_capture_grid_relation(width, height)

        patcher = mock.patch.object(
            dump_contract, "_capture_grid_relation",
            side_effect=fixture_capture_grid_relation)
        patcher.start()
        self.addCleanup(patcher.stop)

        def fixture_source_capture_grid_relation(
                source_width, source_height, width, height):
            if (width, height) == (16, 12):
                return "test-only-fixture", 1, width, height
            return self.production_source_capture_grid_relation(
                source_width, source_height, width, height)

        source_patcher = mock.patch.object(
            dump_contract, "capture_grid_relation_for_source",
            side_effect=fixture_source_capture_grid_relation)
        source_patcher.start()
        self.addCleanup(source_patcher.stop)

        manifest = coordinate.load_contract()
        tag = generator.contract_tag(manifest)
        defaults = coordinate.CALIBRATED_DEFAULTS
        calibration = coordinate.MODEL_CALIBRATIONS[0]
        width, height = calibration.calibrated_input_shapes[0]
        raw_scale = calibration.raw_coordinate_scale
        texel_count = width * height
        values = {
            "center": 2.0,
            "inverse_scale": 1.0 / raw_scale,
            "convergence_curve": 0.0,
            "container_scale": 1.0,
            "calibration_revision": 4,
            "frame_valid": 1.0,
            "confirmed_cut_count": 3,
            "contract_tag_bits": tag,
            "camera_center_integrity_bits": 0,
            "renderer_authorization_bits": tag,
            "joint_plane_mode_bits": 3,
            "mapping_state_reserved_2": 0,
        }
        for descriptor in manifest["shadow_state"]["fields"][12:]:
            values[descriptor["name"]] = (0.0 if descriptor["gpu_encoding"] == "float" else 0)
        values.update({
            "gain_last_observation_low": 1000001, "gain_clock_armed": 1,
            "gain_seed_count": 1, "gain_target_zero": values["center"],
            "gain_target_inverse_scale": values["inverse_scale"],
            "gain_target_nearest": raw_scale,
            "gain_display_limit": dump_contract._float32(defaults.direct_container_limit),
            "gain_seed_first_low": 1000001, "gain_seed_last_low": 1000001,
            "gain_seed_mean_nearest": raw_scale, "gain_seed_mean_zero": values["center"],
        })
        values["camera_center_integrity_bits"] = (
            dump_contract.camera_center_integrity_for_state_values(values))
        fields = []
        for descriptor in manifest["shadow_state"]["fields"]:
            name = descriptor["name"]
            fields.append({
                "index": descriptor["index"],
                "name": name,
                "type": ("float32" if descriptor["gpu_encoding"] == "float"
                         else "uint32-bitcast"),
                "value": values[name],
            })
        shader = coordinate.SHADER_IMPLEMENTATION
        coordinate_binding = {
            "schema": coordinate.CONTRACT_SCHEMA,
            "tag": tag,
            "source_closure_schema": shader.source_closure_schema,
            "source_compile_flags": shader.source_compile_flags,
            "source_macro_count": shader.source_macro_count,
            "source_closure_sha256": shader.source_closure_sha256,
        }
        requested_pop = 1.0
        requested_gain = defaults.gain_per_pop * requested_pop
        constants = {
            "raw_coordinate_scale": raw_scale,
            "collapse_abs_epsilon": defaults.collapse_abs_epsilon,
            "gain_per_pop": defaults.gain_per_pop,
            "reference_pop_strength": defaults.reference_pop_strength,
            "reference_gain_at_reference_pop":
                defaults.gain_per_pop * defaults.reference_pop_strength,
            "requested_gain": requested_gain,
            "requested_pop_strength": requested_pop,
            "direct_container_limit": defaults.direct_container_limit,
            "max_horizontal_slope": defaults.max_horizontal_slope,
            "max_vertical_shear": defaults.max_vertical_shear,
            "vertical_majorant_share": defaults.vertical_majorant_share,
            "convergence_curve_default": defaults.convergence_curve_default,
        }
        self.state = {
            "schema": dump_contract.SHADOW_STATE_DUMP_SCHEMA,
            "coordinate_contract": {
                **coordinate_binding,
                "state_word_count": len(fields),
            },
            "source": manifest["shadow_state"]["source"],
            "capture": manifest["shadow_state"]["capture"],
            "rendered_output_selected": False,
            "wire_contract":
                "experimental diagnostic shadow; not selected by the renderer or client",
            "units": {
                "coordinate": "dimensionless canonical coordinate derived from raw depth",
                "gain": "one-eye source-U per coordinate unit",
                "parallax": "signed one-eye source-U",
            },
            "constants": constants,
            "fields": fields,
            "named_values": values,
            "decoded": {
                "joint_plane_mode": 3,
                "frame_valid": True,
                "camera_valid": True,
                "calibration_revision": values["calibration_revision"],
                "confirmed_cut_count": values["confirmed_cut_count"],
                "contract_tag": tag,
                "requested_gain": constants["requested_gain"],
                "requested_pop_strength": constants["requested_pop_strength"],
                "latched_scale": raw_scale,
                "convergence_curve": values["convergence_curve"],
                "container_scale": values["container_scale"],
                "effective_gain": requested_gain,
                "camera_center_integrity_bits":
                    values["camera_center_integrity_bits"],
                "renderer_authorization_bits": values["renderer_authorization_bits"],
            },
            "adaptation_semantics": dump_contract.adaptation_semantics_for_mode(3),
        }
        self.live_state = copy.deepcopy(self.state)
        self.live_state["rendered_output_selected"] = True
        self.live_state["wire_contract"] = (
            "authenticated live Host-SBS renderer input; not a client wire contract")
        self.live_state["units"] = {
            "coordinate": "dimensionless canonical coordinate derived from raw depth",
            "gain": "one-eye full-source-U per coordinate unit",
            "parallax": "signed one-eye full-source-U",
        }
        stats = {
            "mean": 2.0,
            "population_std": 0.5,
            "minimum": 1.0,
            "maximum": 3.0,
            "valid_count": float(texel_count),
            "texel_count": float(texel_count),
            "valid": 1.0,
            "reserved": 0.0,
            "percentile_low": 1.1,
            "percentile_high": 2.9,
            "percentile_valid": 1.0,
            "percentile_bin_width": 2.0 / 2048.0,
        }
        self.frame_stats = {
            "schema": dump_contract.SHADOW_FRAME_STATS_DUMP_SCHEMA,
            "coordinate_contract": {
                **coordinate_binding,
                "frame_stats_word_count": len(stats),
            },
            "source": manifest["frame_stats"]["source"],
            "named_values": stats,
        }
        vertical_descriptions = {
            "shadow_coordinate.f32":
                ("parallax-v2 canonical coordinate diagnostic", True),
            "shadow_candidate_parallax.f32":
                ("parallax-v2 pre-limiter candidate displacement", True),
            "shadow_vertical_majorant.f32":
                ("parallax-v2 vertical shear-limiter intermediate", False),
            "shadow_vertical_conditioned.f32":
                ("parallax-v2 orientation-selective vertical conditioner", False),
            "shadow_final_parallax.f32":
                ("parallax-v2 atomic final displacement field", True),
        }
        subtitle_none = {
            "mode": "none",
            "request": None,
            "producer": None,
            "resolver": None,
            "artifacts": {},
        }
        self.manifest = {
            "schema": dump_contract.DUMP_MANIFEST_SCHEMA,
            "capture_status": "complete",
            "published_atomically": True,
            "matched_frame_id": 41,
            "matched_observation_timestamp_us": 1000001,
            "color_mode": "srgb",
            "float_previews": {
                "packaged": False,
                "generator": "tools/sbsbench/generate_dump_previews.py",
                "normalization": "finite p2-p98 computed on demand",
            },
            "subtitle_conditioning": subtitle_none,
            "renderer": {
                "authority":
                    "authenticated-parallax-v2-atomic-final-field",
                "parallax_v2_render_requested": True,
                "parallax_v2_render_selected": True,
                "mapping_artifacts_match_selected_renderer": False,
                "parallax_v2_position_field": "shadow_final_parallax",
                "parallax_v2_coordinate_role":
                    "shadow_coordinate is diagnostic only; it has no renderer authority",
                "parallax_v2_vertical_majorant_role":
                    "least column-wise upper envelope v+ >= candidate with "
                    "adjacent-row source-U change <= max_vertical_shear/content_width; "
                    "diagnostic evidence only",
                "parallax_v2_vertical_conditioned_role":
                    "fixed 75/25 share of column upper/lower envelopes; may raise or lower "
                    "candidate and feeds the row majorant",
                "parallax_v2_conditioner_role":
                    "least row-wise q >= shadow_vertical_conditioned with horizontal slope <= "
                    "max_horizontal_slope and vertical shear <= max_vertical_shear publishes "
                    "shadow_final_parallax atomically as direct live authority",
                "parallax_v2_inverse":
                    "11-step contractive fixed point; no forward-warp owner/visibility splat and no synthetic fill",
                "collar_defocus": {
                    "enabled": False,
                    "role": ("disabled after live hand-boundary halo regression; live color "
                             "uses one linear sample at the inverse-warped coordinate"),
                    "kernel": "none",
                    "hdr": ("native source sample; no clamp, tone map, or gamma "
                            "conversion"),
                },
                "live_shader_source": {
                    "source_closure_schema": generator.SOURCE_CLOSURE_SCHEMA,
                    "source_compile_flags": generator.SHADER_COMPILE_FLAGS,
                    "source_macro_count": 0,
                    "source_closure_sha256":
                        dump_contract.LIVE_RENDERER_SOURCE_CLOSURE_SHA256,
                    "source_file": "sbs_reprojection_v2_live_ps.hlsl",
                    "entrypoint": "main_ps",
                    "target": "ps_5_0",
                    "diagnostic_source_closure_sha256":
                        dump_contract.DIAGNOSTIC_SOURCE_CLOSURE_SHA256,
                    "mapping_source_file": "sbs_reprojection_v2_diagnostics_ps.hlsl",
                    "mapping_entrypoint": "mapping_ps",
                    "mask_source_file": "sbs_reprojection_v2_diagnostics_ps.hlsl",
                    "mask_entrypoint": "mask_ps",
                },
            },
            "parallax_v2_shadow": {
                "requested": False,
                "active": True,
                "rendered_output_selected": True,
                "shader_source": {
                    "source_closure_schema": shader.source_closure_schema,
                    "source_compile_flags": shader.source_compile_flags,
                    "source_macro_count": shader.source_macro_count,
                    "source_closure_sha256": shader.source_closure_sha256,
                },
                "state": {
                    **copy.deepcopy(self.live_state["decoded"]),
                    "raw_coordinate_scale": raw_scale,
                    "rendered_output_selected": True,
                },
            },
            "dimensions": {
                "source": {
                    "width": 1920, "height": 1080,
                    "format": "DXGI_FORMAT_B8G8R8A8_UNORM", "format_value": 87,
                },
                "analysis_source": {
                    "width": 1920, "height": 1080,
                    "format": "DXGI_FORMAT_B8G8R8A8_UNORM", "format_value": 87,
                },
                "model_input": {
                    "width": 770, "height": 434, "channels": 3,
                    "layout": "NCHW", "dtype": "float32-le",
                },
                "raw_depth": {
                    "width": 770, "height": 434,
                    "format": "float32-le structured buffer",
                },
                "shadow_coordinate": {
                    "width": 770, "height": 434,
                    "format": "DXGI_FORMAT_R32_FLOAT", "format_value": 41,
                },
                "shadow_candidate_parallax": {
                    "width": 770, "height": 434,
                    "format": "DXGI_FORMAT_R32_FLOAT", "format_value": 41,
                },
                "shadow_vertical_majorant": {
                    "width": 770, "height": 434,
                    "format": "DXGI_FORMAT_R32_FLOAT", "format_value": 41,
                },
                "shadow_vertical_conditioned": {
                    "width": 770, "height": 434,
                    "format": "DXGI_FORMAT_R32_FLOAT", "format_value": 41,
                },
                "shadow_final_parallax": {
                    "width": 770, "height": 434,
                    "format": "DXGI_FORMAT_R32_FLOAT", "format_value": 41,
                },
            },
            "artifacts": {
                name: {
                    "available": True,
                    "required": required,
                    "stage": stage,
                    "description": "authenticated test artifact",
                    **({"sha256": "0" * 64} if name.endswith(".f32") else {}),
                }
                for name, (stage, required) in vertical_descriptions.items()
            },
            "depth_input_region": {
                "available": True,
                "artifact": "depth_input_region.json",
                "mode": "full-source",
                "geometry_authority": True,
                "renderer_authority": True,
            },
            "window_region": {
                "available": False,
                "artifact": None,
                "observer_status": "not-observed",
                "mapping_status": "not-mapped",
                "geometry_authority": False,
                "renderer_authority": False,
            },
            "gpu_trace": {
                "available": False,
                "required": False,
                "rendering_authority": False,
                "raw_artifact": None,
                "decoded_artifact": None,
                "contract_artifact": None,
            },
            "final_parallax": {
                "contract_schema": coordinate.FINAL_PARALLAX.schema,
                "artifact": "shadow_final_parallax.f32",
                "warp_input_artifact": "shadow_final_parallax.f32",
                "authority": coordinate.FINAL_PARALLAX.authority,
                "publication_policy": coordinate.FINAL_PARALLAX.publication_policy,
                "reuse_policy": coordinate.FINAL_PARALLAX.reuse_policy,
                "invalid_policy": coordinate.FINAL_PARALLAX.invalid_policy,
                "current_rgb_policy": coordinate.FINAL_PARALLAX.current_rgb_policy,
                "warp_relation": "same authenticated artifact",
            },
            "warp_map_contract": {
                "available": False,
                "schema": 2,
                "artifact": None,
            },
        }
        self.manifest["artifacts"].update({
            "subtitle_conditioning.json": {
                "available": True,
                "required": True,
                "stage": "subtitle conditioning authority",
                "description": "canonical current-schema ordinary subtitle state",
                "sha256": hashlib.sha256(
                    json.dumps(subtitle_none).encode("utf-8")).hexdigest(),
            },
            "depth_input_region.json": {
                "available": True,
                "required": True,
                "stage": "depth analysis input region",
                "description": "authoritative test input-region artifact",
                "sha256": "0" * 64,
            },
            "depth_input_source.png": {
                "available": True,
                "required": False,
                "stage": "model-depth input source preview",
                "description": (
                    "Spatially exact full-source preview or post-completion reconstruction of "
                    "the logical ROI source. Live ROI preprocessing samples the retained full "
                    "matched texture directly; this transfer-aware PNG is diagnostic only, "
                    "while model_input.f32 is numeric authority."),
            },
            "shadow_state.json": {
                "available": True,
                "required": True,
                "stage": "parallax-v2 shot calibration and attenuation state",
                "description": "authenticated test V2 state",
                "sha256": "0" * 64,
            },
            "shadow_frame_stats.json": {
                "available": True,
                "required": True,
                "stage": "parallax-v2 current-frame moments",
                "description": "authenticated test V2 frame statistics",
                "sha256": "0" * 64,
            },
            "window_region.json": {
                "available": False,
                "required": False,
                "stage": "matched-frame window region provenance",
                "description": "diagnostic test artifact",
            },
            "gpu_trace_ring.u32": {
                "available": False,
                "required": False,
                "stage": "diagnostic GPU accepted-root completion history",
                "description": "optional diagnostic test artifact",
            },
            "gpu_trace.json": {
                "available": False,
                "required": False,
                "stage": "decoded diagnostic GPU history",
                "description": "optional diagnostic test artifact",
            },
            "gpu_trace_contract.json": {
                "available": False,
                "required": False,
                "stage": "diagnostic GPU trace wire contract",
                "description": "optional diagnostic test artifact",
            },
            "warp_map.f32": {
                "available": False,
                "required": False,
                "stage": "exact inverse-warp mapping",
                "description": "optional diagnostic test artifact",
            },
            "warp_mask.png": {
                "available": False,
                "required": False,
                "stage": "V2 boundary-extrapolation mask",
                "description": "optional diagnostic test artifact",
            },
        })
        self.depth_input_region = {
            "schema": dump_contract.DEPTH_INPUT_REGION_SCHEMA,
            "capture":
                "same matched source/color/model/depth/render frame as the parent Dump 3D package",
            "role":
                "authoritative analysis-domain placement and live-render embedding contract",
            "matched_frame_id": 41,
            "mode": "full-source",
            "authorization": None,
            "coordinate_space": {
                "name": "matched-source-pixels",
                "rect_semantics": "half-open [left, top, right, bottom)",
                "source_extent_px": {"width": 1920, "height": 1080},
                "semantic_rect_px": None,
                "inference_rect_px": {
                    "left": 0, "top": 0, "right": 1920, "bottom": 1080,
                },
            },
            "analysis": {
                "analysis_generation": 0,
                "input_domain_reset": False,
                "tensor_extent_px": {"width": 770, "height": 434},
                "tensor_content_rect_px": {
                    "left": 0, "top": 0, "right": 770, "bottom": 434,
                },
                "padded_area_fraction": 0.0,
                "fit_method": "full-tensor",
                "crop_method": "full-source",
                "scene_analysis_domain": "full-source",
            },
            "renderer": {
                "final_parallax_units": "full-source-u",
                "full_source_parallax_scale": 1.0,
                "inside_inference_rect": "no taper",
                "outside": None,
                "source_sampling":
                    "full matched source; never clamp to inference rectangle",
                "inverse_iterations": 11,
            },
        }

    def test_valid_state_and_frame_stats_are_accepted(self):
        decoded = dump_contract.validate_shadow_state_document(self.state)
        self.assertEqual(decoded["calibration_revision"], 4)
        self.assertEqual(decoded["convergence_curve"], 0.0)
        self.assertEqual(set(decoded), {
            descriptor["name"] for descriptor in coordinate.load_contract()["shadow_state"]["fields"]})
        stats = dump_contract.validate_shadow_frame_stats_document(self.frame_stats)
        self.assertEqual(stats["valid"], 1.0)

    def test_current_manifest_attributes_vertical_share_and_row_majorant(self):
        decoded = dump_contract.validate_v2_dump_manifest_document(self.manifest)
        self.assertTrue(decoded["active"])
        self.assertTrue(decoded["rendered_output_selected"])
        self.assertTrue(decoded["vertical_majorant_available"])
        self.assertTrue(decoded["vertical_conditioned_available"])
        self.assertEqual(decoded["position_field"], "shadow_final_parallax")
        self.assertFalse(decoded["mapping_artifacts_match_selected_renderer"])

    def test_manifest_accepts_exact_native_capture_color_format_allowlist(self):
        for name, value in (
                ("DXGI_FORMAT_B8G8R8A8_UNORM", 87),
                ("DXGI_FORMAT_B8G8R8X8_UNORM", 88),
                ("DXGI_FORMAT_R8G8B8A8_UNORM", 28),
                ("DXGI_FORMAT_R16G16B16A16_FLOAT", 10)):
            changed = copy.deepcopy(self.manifest)
            for key in ("source", "analysis_source"):
                changed["dimensions"][key]["format"] = name
                changed["dimensions"][key]["format_value"] = value
            with self.subTest(name=name):
                dump_contract.validate_v2_dump_manifest_document(changed)

        unsupported = copy.deepcopy(self.manifest)
        for key in ("source", "analysis_source"):
            unsupported["dimensions"][key]["format"] = "DXGI_FORMAT_UNKNOWN_24"
            unsupported["dimensions"][key]["format_value"] = 24
        with self.assertRaisesRegex(ValueError, "analysis-source dimensions"):
            dump_contract.validate_v2_dump_manifest_document(unsupported)

    def test_reader_rejects_previous_schema_without_a_compatibility_path(self):
        retired = copy.deepcopy(self.manifest)
        retired["schema"] = dump_contract.DUMP_MANIFEST_SCHEMA - 1
        with self.assertRaisesRegex(ValueError, "unknown serialization schema"):
            dump_contract.validate_v2_dump_manifest_document(retired)

    def test_reader_requires_complete_atomic_publication(self):
        for key, forged in (
                ("capture_status", "partial"),
                ("published_atomically", False)):
            manifest = copy.deepcopy(self.manifest)
            manifest[key] = forged
            with self.subTest(key=key), self.assertRaisesRegex(
                    ValueError, "complete atomic publication"):
                dump_contract.validate_v2_dump_manifest_document(manifest)
        missing = copy.deepcopy(self.manifest)
        missing.pop("published_atomically")
        with self.assertRaisesRegex(ValueError, "complete atomic publication"):
            dump_contract.validate_v2_dump_manifest_document(missing)

    def test_manifest_requires_exact_selected_v2_shadow_attribution(self):
        mutations = {
            "unknown-key": lambda value: value.update({"future": 1}),
            "requested-shadow": lambda value: value.update({"requested": True}),
            "inactive-shadow": lambda value: value.update({"active": False}),
            "unselected-shadow": lambda value: value.update(
                {"rendered_output_selected": False}),
            "shader-drift": lambda value: value["shader_source"].update(
                {"source_closure_sha256": "0" * 64}),
        }
        for name, mutate in mutations.items():
            changed = copy.deepcopy(self.manifest)
            mutate(changed["parallax_v2_shadow"])
            with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "shadow attribution"):
                dump_contract.validate_v2_dump_manifest_document(changed)

        for name, mutate in {
                "unknown-key": lambda value: value.update({"future": 1}),
                "invalid-frame": lambda value: value.update({"frame_valid": False}),
                "wrong-contract": lambda value: value.update({"contract_tag": 0}),
                "unselected-state": lambda value: value.update(
                    {"rendered_output_selected": False}),
        }.items():
            changed = copy.deepcopy(self.manifest)
            mutate(changed["parallax_v2_shadow"]["state"])
            with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "state summary"):
                dump_contract.validate_v2_dump_manifest_document(changed)

    def test_manifest_requires_hashed_v2_state_artifacts_with_exact_roles(self):
        for artifact in ("shadow_state.json", "shadow_frame_stats.json"):
            for field, replacement in (
                    ("sha256", "not-a-hash"),
                    ("required", False)):
                changed = copy.deepcopy(self.manifest)
                changed["artifacts"][artifact][field] = replacement
                with self.subTest(artifact=artifact, field=field), self.assertRaisesRegex(
                        ValueError, "artifact descriptor"):
                    dump_contract.validate_v2_dump_manifest_document(changed)

            changed = copy.deepcopy(self.manifest)
            changed["artifacts"][artifact]["stage"] = "wrong-stage"
            with self.subTest(artifact=artifact, field="stage"), self.assertRaisesRegex(
                    ValueError, "state artifact attribution"):
                dump_contract.validate_v2_dump_manifest_document(changed)


    def test_schema46_accepts_exact_single_high_grid_and_rejects_split_or_arbitrary_high(self):
        high = copy.deepcopy(self.manifest)
        self._set_manifest_capture_grid(high, 1540, 868)
        high["composite_runtime_provenance"] = self._composite_runtime_provenance()
        summary = dump_contract.validate_v2_dump_manifest_document(high)
        self.assertEqual(summary["capture_grid_kind"], "single-high-convex2x")
        self.assertEqual(summary["capture_grid_scale"], 2)
        self.assertEqual(
            (summary["embedded_dav2_width"], summary["embedded_dav2_height"]),
            (770, 434))
        subtitle_high = self._active_uip1_manifest(high)
        self.assertTrue(dump_contract.validate_v2_dump_manifest_document(
            subtitle_high)["subtitle_conditioning"]["live"])

        split = copy.deepcopy(high)
        split["dimensions"]["raw_depth"].update({"width": 770, "height": 434})
        with self.assertRaisesRegex(ValueError, "tensor dimensions disagree"):
            dump_contract.validate_v2_dump_manifest_document(split)

        for width, height in ((1568, 868), (1008, 800)):
            with self.subTest(arbitrary_shape=(width, height)):
                arbitrary = copy.deepcopy(self.manifest)
                self._set_manifest_capture_grid(arbitrary, width, height)
                with self.assertRaisesRegex(ValueError, "neither calibrated DAV2 nor an exact"):
                    dump_contract.validate_v2_dump_manifest_document(arbitrary)

        missing = copy.deepcopy(high)
        del missing["composite_runtime_provenance"]
        with self.assertRaisesRegex(ValueError, "schema-2 fused composite"):
            dump_contract.validate_v2_dump_manifest_document(missing)

        stale = copy.deepcopy(high)
        stale["composite_runtime_provenance"]["onnx_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "unauthenticated schema-2 fused composite"):
            dump_contract.validate_v2_dump_manifest_document(stale)

        legacy_claim = copy.deepcopy(self.manifest)
        legacy_claim["composite_runtime_provenance"] = (
            self._composite_runtime_provenance())
        with self.assertRaisesRegex(ValueError, "legacy DAV2 dump"):
            dump_contract.validate_v2_dump_manifest_document(legacy_claim)

    def test_source_fitter_binds_all_24_exact_coarse_and_high_profiles(self):
        cases = PRODUCTION_SOURCE_PROFILE_CASES
        self.assertEqual(
            tuple((width, height) for _, _, width, height in cases),
            coordinate.MODEL_CALIBRATIONS[0].calibrated_input_shapes)
        for source_width, source_height, coarse_width, coarse_height in cases:
            with self.subTest(source=(source_width, source_height), scale=1):
                self.assertEqual(
                    dump_contract.fit_host_sbs_v2_depth_tensor_shape(
                        source_width, source_height),
                    (coarse_width, coarse_height))
                self.assertEqual(
                    dump_contract.expected_capture_grid_for_source(
                        source_width, source_height, scale=1),
                    (coarse_width, coarse_height))
                coarse = copy.deepcopy(self.manifest)
                for name in ("source", "analysis_source"):
                    coarse["dimensions"][name].update({
                        "width": source_width, "height": source_height})
                self._set_manifest_capture_grid(
                    coarse, coarse_width, coarse_height)
                summary = dump_contract.validate_v2_dump_manifest_document(coarse)
                self.assertEqual(summary["capture_grid_kind"], "legacy-dav2")

            high_width, high_height = 2 * coarse_width, 2 * coarse_height
            with self.subTest(source=(source_width, source_height), scale=2):
                self.assertEqual(
                    dump_contract.expected_capture_grid_for_source(
                        source_width, source_height, scale=2),
                    (high_width, high_height))
                high = copy.deepcopy(coarse)
                self._set_manifest_capture_grid(high, high_width, high_height)
                high["composite_runtime_provenance"] = (
                    self._composite_runtime_provenance())
                summary = dump_contract.validate_v2_dump_manifest_document(high)
                self.assertEqual(summary["capture_grid_kind"], "single-high-convex2x")
                self.assertEqual(
                    (summary["embedded_dav2_width"], summary["embedded_dav2_height"]),
                    (coarse_width, coarse_height))

    def test_manifest_rejects_a_different_supported_profile_for_the_source(self):
        for width, height, composite in (
                (1022, 434, False),
                (2044, 868, True),
                (868, 1540, True),
                (574, 434, False),
                (1148, 868, True),
                (868, 1960, True)):
            with self.subTest(grid=(width, height)):
                changed = copy.deepcopy(self.manifest)
                self._set_manifest_capture_grid(changed, width, height)
                if composite:
                    changed["composite_runtime_provenance"] = (
                        self._composite_runtime_provenance())
                with self.assertRaisesRegex(ValueError, "exact source-derived production profile"):
                    dump_contract.validate_v2_dump_manifest_document(changed)

    def test_expected_capture_grid_rejects_unsupported_source_and_scale_types(self):
        self.assertIsNone(dump_contract.expected_capture_grid_for_source(
            1160, 496, scale=1))
        self.assertIsNone(dump_contract.expected_capture_grid_for_source(
            1160, 496, scale=2))
        self.assertIsNone(dump_contract.expected_capture_grid_for_source(
            5121, 1080, scale=1))
        self.assertIsNone(dump_contract.expected_capture_grid_for_source(
            True, 1080, scale=1))
        for scale in (True, 0, 3, 2.0):
            with self.subTest(scale=scale), self.assertRaisesRegex(
                    ValueError, "scale must be exactly 1 or 2"):
                dump_contract.expected_capture_grid_for_source(
                    1920, 1080, scale=scale)

    def test_production_readers_reject_subcalibration_fixture_grid(self):
        document = copy.deepcopy(self.depth_input_region)
        document["analysis"]["tensor_extent_px"] = {"width": 16, "height": 12}
        document["analysis"]["tensor_content_rect_px"] = {
            "left": 0, "top": 0, "right": 16, "bottom": 12,
        }
        with mock.patch.object(
                dump_contract, "_capture_grid_relation",
                self.production_capture_grid_relation), mock.patch.object(
                    dump_contract, "capture_grid_relation_for_source",
                    self.production_source_capture_grid_relation):
            with self.assertRaisesRegex(ValueError, "exact source-derived"):
                dump_contract.validate_depth_input_region_document(document)

            manifest = copy.deepcopy(self.manifest)
            self._set_manifest_capture_grid(manifest, 16, 12)
            with self.assertRaisesRegex(ValueError, "neither calibrated DAV2"):
                dump_contract.validate_v2_dump_manifest_document(manifest)


    @staticmethod
    def _valid_ocr_record_words():
        words = [0] * dump_contract.SUBTITLE_OCR_RECORD_WORD_COUNT
        words[:16] = [
            dump_contract.SUBTITLE_OCR_RECORD_SCHEMA,
            dump_contract.SUBTITLE_OCR_RECORD_TAG,
            1, 1, 1,
            41, 0,
            17, 0,
            1920, 1080,
            770, 434,
            325, 430,
            0,
        ]
        score_bits = struct.unpack("<I", struct.pack("<f", 0.875))[0]
        box = [120, 350, 650, 401, score_bits, 0, 1, 0]
        raw = dump_contract.SUBTITLE_OCR_RAW_BOX_WORD_OFFSET
        final = dump_contract.SUBTITLE_OCR_FINAL_BOX_WORD_OFFSET
        words[raw:raw + len(box)] = box
        words[final:final + len(box)] = box
        return words

    @staticmethod
    def _pack_uint32_words(words):
        return struct.pack(f"<{len(words)}I", *words)

    def _active_uip1_manifest(self, base_manifest=None):
        manifest = copy.deepcopy(
            self.manifest if base_manifest is None else base_manifest)
        subtitle = {
            "mode": "subtitle-contours-v1",
            "request": True,
            "producer": dump_contract._subtitle_ocr_producer_contract(),
            "resolver": dump_contract._subtitle_plane_resolver_contract(),
            "artifacts": {
                "ocr_record": "subtitle_ocr_record.u32",
                "locator_state": "subtitle_locator_state.u32",
                "base_field": "shadow_base_final_parallax.f32",
                "conditioned_field": "shadow_final_parallax.f32",
                **dump_contract._SUBTITLE_CONTOUR_ARTIFACT_FILES,
            },
        }
        manifest["subtitle_conditioning"] = subtitle
        manifest["dimensions"]["shadow_base_final_parallax"] = copy.deepcopy(
            manifest["dimensions"]["shadow_final_parallax"])
        manifest["artifacts"].update({
            "subtitle_ocr_record.u32": {
                "available": True,
                "required": True,
                "stage": "OCR8 subtitle boxes for atomic target",
                "description": "authenticated test OCR8 record",
                "sha256": "0" * 64,
            },
            "subtitle_locator_state.u32": {
                "available": True,
                "required": True,
                "stage": "compact UIP4 current subtitle plane state",
                "description": "authenticated test UIP4 state",
                "sha256": "0" * 64,
            },
            "shadow_base_final_parallax.f32": {
                "available": True,
                "required": True,
                "stage": "ordinary post-limiter V2 field before UIP4 conditioning",
                "description": "authenticated test unconditioned base field",
                "sha256": "0" * 64,
            },
        })
        layouts = dump_contract._subtitle_contour_artifact_layouts(
            manifest["dimensions"]["analysis_source"]["width"],
            manifest["dimensions"]["analysis_source"]["height"],
            manifest["dimensions"]["shadow_final_parallax"]["width"],
            manifest["dimensions"]["shadow_final_parallax"]["height"])
        for role, (layout, byte_count) in layouts.items():
            manifest["dimensions"][role] = layout
            manifest["artifacts"][dump_contract._SUBTITLE_CONTOUR_ARTIFACT_FILES[role]] = {
                "available": True, "required": True,
                "stage": dump_contract._SUBTITLE_CONTOUR_ARTIFACT_STAGES[role],
                "description": "complete independently authored test contour tuple",
                "sha256": "0" * 64, "byte_count": byte_count,
            }
        manifest["renderer"]["parallax_v2_conditioner_role"] = (
            "least row-wise q >= shadow_vertical_conditioned with horizontal slope <= "
            "max_horizontal_slope and vertical shear <= max_vertical_shear produces "
            "shadow_base_final_parallax; UIP4 publishes shadow_final_parallax atomically as "
            "direct live authority")
        payload = json.dumps(subtitle).encode("utf-8")
        manifest["artifacts"]["subtitle_conditioning.json"]["sha256"] = (
            hashlib.sha256(payload).hexdigest())
        return manifest


    def test_current_uip1_manifest_rejects_provenance_roles_and_base_field_drift(self):
        mutations = {
            "retired-mode": (
                lambda manifest: manifest["subtitle_conditioning"].update(
                    {"mode": "subtitle-slr8"}),
                "unsupported subtitle-conditioning authority"),
            "request": (
                lambda manifest: manifest["subtitle_conditioning"].update({"request": False}),
                "enabled request"),
            "target-policy": (
                lambda manifest: manifest["subtitle_conditioning"]["resolver"][
                    "target_policy"].update(
                        {"approach_dwell_us": 99999}),
                "resolver provenance"),
            "current-cover-policy": (
                lambda manifest: manifest["subtitle_conditioning"]["resolver"][
                    "current_cover_policy"].update({"selection": "largest-component-only"}),
                "resolver provenance"),
            "model": (
                lambda manifest: manifest["subtitle_conditioning"]["producer"]["model"].update(
                    {"artifact_onnx_sha256": "0" * 64}),
                "OCR8 provenance"),
            "resolver": (
                lambda manifest: manifest["subtitle_conditioning"]["resolver"].update(
                    {"state_tag": 0}),
                "resolver provenance"),
            "artifact-role": (
                lambda manifest: manifest["subtitle_conditioning"]["artifacts"].update(
                    {"plane_state": "wrong.u32"}),
                "artifact roles"),
            "artifact-stage": (
                lambda manifest: manifest["artifacts"]["subtitle_ocr_record.u32"].update(
                    {"stage": "diagnostic only"}),
                "misattributes"),
            "base-dimension": (
                lambda manifest: manifest["dimensions"]
                ["shadow_base_final_parallax"].update({"width": 769}),
                "dimensions disagree"),
            "retired-base-preview": (
                lambda manifest: manifest["artifacts"].update({
                    "shadow_base_final_parallax.png": {
                        "available": True, "required": False,
                        "stage": "retired preview", "description": "retired"}}),
                "retired schema-38 artifact"),
        }
        for name, (mutate, error) in mutations.items():
            with self.subTest(name=name):
                manifest = self._active_uip1_manifest()
                mutate(manifest)
                with self.assertRaisesRegex(ValueError, error):
                    dump_contract.validate_v2_dump_manifest_document(manifest)

    def test_current_contour_manifest_requires_complete_ordered_temporal_roots(self):
        for fault in ("missing-temporal", "legacy-seed", "reordered-temporal"):
            with self.subTest(fault=fault):
                manifest = self._active_uip1_manifest()
                specs = manifest["subtitle_conditioning"]["resolver"]["shader_contract"]["source_specs"]
                index = next(i for i, spec in enumerate(specs) if spec["entrypoint"] == "temporal_main")
                if fault == "missing-temporal":
                    del specs[index]
                elif fault == "legacy-seed":
                    specs[index]["entrypoint"] = "seed_main"
                else:
                    specs[index], specs[index + 1] = specs[index + 1], specs[index]
                with self.assertRaisesRegex(ValueError, "resolver provenance"):
                    dump_contract.validate_v2_dump_manifest_document(manifest)

    def test_current_ocr8_record_validates_exact_identity_boxes_and_empty_authority(self):
        decoded = dump_contract.validate_subtitle_ocr_record(
            self._pack_uint32_words(self._valid_ocr_record_words()),
            matched_frame_id=41,
            analysis_generation=17,
            source_width=1920,
            source_height=1080,
            field_width=770,
            field_height=434,
            roi_top=325,
            roi_bottom=430)
        self.assertTrue(decoded["authoritative"])
        self.assertEqual(decoded["raw_count"], 1)
        self.assertEqual(decoded["final_count"], 1)
        self.assertEqual(
            decoded["ordinary_cover_horizontal_guard"], {
                "cells": 1,
                "policy": "ordinary-only-add-guard-cells-times-producer-ocr-cell-width-to-horizontal-db-unclip-cover-clamped-detector-bounds-before-map-current-canonical-cover-all-consumers-no-second-mask-vertical-and-ribbon-unchanged-v1",
            })
        self.assertEqual(decoded["line_centering"]["minimum_vertical_span_iou_ratio"], [1, 2])
        self.assertEqual(decoded["line_centering"]["maximum_horizontal_gap_height_ratio"], [4, 1])
        self.assertEqual(
            decoded["line_centering"]["horizontal_gap_metric"],
            "native-source-horizontal-gap-to-current-union-edge-relative-to-max-candidate-and-fixed-anchor-height-exact-cross-multiplied-v1")
        self.assertEqual(decoded["line_centering"]["center_tolerance_ratio"], [1, 10])
        self.assertEqual(decoded["line_centering"]["search_bottom_height_ratio"], [1, 4])
        self.assertFalse(decoded["line_centering"]["preselection_geometry_verified"])
        self.assertIn("full-screen", decoded["line_centering"]["center_reference"])
        self.assertEqual(
            decoded["final_boxes"][0], {
                "left": 120,
                "top": 350,
                "right": 650,
                "bottom": 401,
                "score": 0.875,
                "score_bits": struct.unpack("<I", struct.pack("<f", 0.875))[0],
                "box_flags": 0,
                "kind": "text",
                "island_count": 1,
                "structural_gap_count": 0,
            })

        empty = self._valid_ocr_record_words()
        empty[3] = 0
        empty[4] = 0
        empty[dump_contract.SUBTITLE_OCR_RAW_BOX_WORD_OFFSET:] = [
            0
        ] * (dump_contract.SUBTITLE_OCR_RECORD_WORD_COUNT -
             dump_contract.SUBTITLE_OCR_RAW_BOX_WORD_OFFSET)
        decoded = dump_contract.validate_subtitle_ocr_record(
            self._pack_uint32_words(empty),
            matched_frame_id=41,
            analysis_generation=17,
            source_width=1920,
            source_height=1080,
            field_width=770,
            field_height=434,
            roi_top=325,
            roi_bottom=430)
        self.assertTrue(decoded["authoritative"])
        self.assertEqual(decoded["final_boxes"], [])


    def test_current_ocr8_record_rejects_identity_capacity_slots_and_box_corruption(self):
        mutations = {
            "tag": (lambda words: words.__setitem__(1, 0), "schema or tag"),
            "frame": (lambda words: words.__setitem__(5, 42), "identity"),
            "capacity": (
                lambda words: words.__setitem__(3, 17), "fixed box capacity"),
            "final-count": (
                lambda words: words.__setitem__(4, 2), "must match exactly"),
            "unused-slot": (
                lambda words: words.__setitem__(
                    dump_contract.SUBTITLE_OCR_RAW_BOX_WORD_OFFSET +
                    dump_contract.SUBTITLE_OCR_BOX_WORD_COUNT, 1),
                "canonical zero"),
            "outside-roi": (
                lambda words: words.__setitem__(
                    dump_contract.SUBTITLE_OCR_FINAL_BOX_WORD_OFFSET + 1, 100),
                "inside the field"),
            "nan-score": (
                lambda words: words.__setitem__(
                    dump_contract.SUBTITLE_OCR_FINAL_BOX_WORD_OFFSET + 4,
                    0x7FC00000),
                "finite float32"),
            "low-score": (
                lambda words: words.__setitem__(
                    dump_contract.SUBTITLE_OCR_FINAL_BOX_WORD_OFFSET + 4,
                    struct.unpack("<I", struct.pack("<f", 0.2))[0]),
                r"\[0.4,1.0\]"),
            "box-flags": (
                lambda words: words.__setitem__(
                    dump_contract.SUBTITLE_OCR_FINAL_BOX_WORD_OFFSET + 5, 1),
                "ribbon topology"),
        }
        for name, (mutate, error) in mutations.items():
            with self.subTest(name=name):
                words = self._valid_ocr_record_words()
                mutate(words)
                with self.assertRaisesRegex(ValueError, error):
                    dump_contract.validate_subtitle_ocr_record(
                        self._pack_uint32_words(words),
                        matched_frame_id=41,
                        analysis_generation=17,
                        source_width=1920,
                        source_height=1080,
                        field_width=770,
                        field_height=434,
                        roi_top=325,
                        roi_bottom=430)

    @staticmethod
    def _valid_uip1_state_words(*, frame=41, generation=17, timestamp=1000001,
                               source=(1920, 1080), field=(770, 434), content=None,
                               count=1, current_valid=True, invalid_probe=0):
        content = (0, 0, *field) if content is None else content
        roi = dump_contract._subtitle_dynamic_roi(*source, content)
        assert roi is not None
        target = struct.unpack("<I", struct.pack("<f", 0.00075))[0]
        limit = struct.unpack("<I", struct.pack("<f", 0.04))[0]
        mask = (1 << count) - 1 if current_valid else 0
        words = [dump_contract.SUBTITLE_PLANE_STATE_SCHEMA, dump_contract.SUBTITLE_PLANE_STATE_TAG,
                 int(current_valid), mask, frame & 0xffffffff, frame >> 32,
                 generation & 0xffffffff, generation >> 32, timestamp & 0xffffffff, timestamp >> 32,
                 *source, *field, *roi, *content, 3, 0, 0, 0,
                 1, 8, 1, target, timestamp & 0xffffffff, timestamp >> 32, 0, 0, 0, 0,
                 0, 0, limit, max(count, invalid_probe), 0, invalid_probe]
        assert len(words) == 40
        return words

    @staticmethod
    def _uip1_context(*, source=(1920, 1080), field=(770, 434), content=None):
        return {"matched_frame_id": 41, "analysis_generation": 17,
                "source_width": source[0], "source_height": source[1],
                "field_width": field[0], "field_height": field[1],
                "tensor_content": (0, 0, *field) if content is None else content,
                "expected_scene_epoch": 3,
                "expected_observation_timestamp_us": 1000001}

    def _parse_uip1_ocr_pairs(self, pairs=(), *, authoritative=True,
                              source=(1920, 1080), field=(770, 434), content=None):
        """Author current OCR8 bytes, then pass through its independent strict reader."""
        content = (0, 0, *field) if content is None else content
        roi = dump_contract._subtitle_dynamic_roi(*source, content)
        words = [0] * dump_contract.SUBTITLE_OCR_RECORD_WORD_COUNT
        words[:16] = [dump_contract.SUBTITLE_OCR_RECORD_SCHEMA,
                      dump_contract.SUBTITLE_OCR_RECORD_TAG, int(authoritative),
                      len(pairs), len(pairs), 41, 0, 17, 0, *source, *field, *roi, 0]
        score_bits = struct.unpack("<I", struct.pack("<f", 0.875))[0]
        for slot, (core, cover, ribbon) in enumerate(pairs):
            metadata = [score_bits, int(ribbon), 4 if ribbon else 1, 3 if ribbon else 0]
            for offset, rectangle in ((dump_contract.SUBTITLE_OCR_RAW_BOX_WORD_OFFSET, core),
                                      (dump_contract.SUBTITLE_OCR_FINAL_BOX_WORD_OFFSET, cover)):
                begin = offset + slot * dump_contract.SUBTITLE_OCR_BOX_WORD_COUNT
                words[begin:begin + 8] = [*rectangle, *metadata]
        return dump_contract.validate_subtitle_ocr_record(
            self._pack_uint32_words(words), matched_frame_id=41, analysis_generation=17,
            source_width=source[0], source_height=source[1],
            field_width=field[0], field_height=field[1], roi_top=roi[0], roi_bottom=roi[1],
            tensor_content=content)

    def test_uip1_grants_every_current_pair_including_short_square_disjoint_and_ribbon(self):
        pairs = [((10 + 80 * slot, 335 + 7 * slot, 12 + 80 * slot, 337 + 7 * slot),
                  (9 + 80 * slot, 334 + 7 * slot, 13 + 80 * slot, 338 + 7 * slot), False)
                 for slot in range(7)]
        pairs.append(((100, 420, 670, 430), (0, 418, 770, 434), True))
        ocr = self._parse_uip1_ocr_pairs(pairs)
        words = self._valid_uip1_state_words(count=8)
        snapshot = dump_contract.validate_subtitle_plane_snapshot(
            self._pack_uint32_words(words), **self._uip1_context())
        self.assertEqual(snapshot["current_count"], 8)
        self.assertFalse(snapshot["cover_geometry_verified"])
        self.assertNotIn("current_rectangles", snapshot)
        state = dump_contract.validate_subtitle_plane_state(
            self._pack_uint32_words(words), ocr_record=ocr, **self._uip1_context())
        self.assertTrue(state["cover_geometry_verified"])
        self.assertEqual(state["current_slot_mask"], 255)
        self.assertEqual(state["current_rectangles"], ocr["final_boxes"])
        self.assertEqual(state["current_ribbon_mask"], 128)
        # UIP4 current geometry may refresh while controller statistics remain held.
        self.assertEqual(state["adaptive"]["covered_count"], 8)
        params = [4, coordinate.SUBTITLE_OCR.condition_param_tag, 8, 128, 2, words[27]]
        dump_contract.validate_subtitle_condition_words(
            self._pack_uint32_words(params), plane_state=state, geometry_ready=True)
        with self.assertRaisesRegex(ValueError, "actual current OCR8 covers"):
            dump_contract.validate_subtitle_condition_words(
                self._pack_uint32_words(params), plane_state=snapshot, geometry_ready=True)

    def test_uip1_preserves_current_ocr_slot_order_without_a_component_winner(self):
        pairs = [((600, 330, 605, 335), (598, 329, 607, 337), False),
                 ((40, 390, 45, 395), (38, 388, 47, 397), False),
                 ((300, 350, 304, 354), (298, 348, 306, 356), False)]
        words = self._valid_uip1_state_words(count=3)
        for ordered in (pairs, list(reversed(pairs))):
            ocr = self._parse_uip1_ocr_pairs(ordered)
            state = dump_contract.validate_subtitle_plane_state(
                self._pack_uint32_words(words), ocr_record=ocr, **self._uip1_context())
            self.assertEqual(state["current_rectangles"], ocr["final_boxes"])
            self.assertEqual(state["current_count"], 3)

    def test_uip1_valid_empty_abstention_invalid_geometry_and_invalid_probe_are_distinct(self):
        empty = self._parse_uip1_ocr_pairs()
        abstention = self._parse_uip1_ocr_pairs(authoritative=False)
        nonempty = self._parse_uip1_ocr_pairs([
            ((120, 350, 650, 401), (120, 350, 650, 401), False)])
        cases = [(empty, True, 0, 0, True), (abstention, False, 0, 0, True),
                 (nonempty, False, 0, 0, False), (nonempty, True, 1, 1, True)]
        for ocr, current_valid, count, invalid_probe, geometry_ready in cases:
            with self.subTest(authoritative=ocr["authoritative"], count=count,
                              current_valid=current_valid, invalid_probe=invalid_probe):
                words = self._valid_uip1_state_words(count=count, current_valid=current_valid,
                                                      invalid_probe=invalid_probe)
                state = dump_contract.validate_subtitle_plane_state(
                    self._pack_uint32_words(words), ocr_record=ocr, **self._uip1_context())
                self.assertEqual(state["current_valid"], current_valid)
                self.assertEqual(state["current_count"], count)
                self.assertEqual(len(state["current_rectangles"]), count)
                params = [4, coordinate.SUBTITLE_OCR.condition_param_tag, 0, 0, 0, 0]
                dump_contract.validate_subtitle_condition_words(
                    self._pack_uint32_words(params), plane_state=state,
                    geometry_ready=geometry_ready)

    def test_uip1_current_mask_must_match_all_exact_current_ocr8_pairs(self):
        one = self._parse_uip1_ocr_pairs([
            ((120, 350, 650, 401), (120, 350, 650, 401), False)])
        for count, ocr in ((0, one), (2, one), (1, self._parse_uip1_ocr_pairs()),
                           (1, self._parse_uip1_ocr_pairs(authoritative=False))):
            with self.subTest(count=count, final_count=ocr["final_count"]):
                with self.assertRaisesRegex(ValueError, "authoritative current OCR8"):
                    dump_contract.validate_subtitle_plane_state(
                        self._pack_uint32_words(self._valid_uip1_state_words(count=count)),
                        ocr_record=ocr, **self._uip1_context())

    def test_uip4_header_authenticates_exact_current_identity_and_reserved_words(self):
        good = self._valid_uip1_state_words()
        # Words 21/22 are the current reset mask and packed warmup; tested separately.
        for index in (0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                      16, 17, 18, 19, 20, 23):
            changed = good.copy()
            changed[index] ^= 1
            with self.subTest(word=index), self.assertRaises(ValueError):
                dump_contract.validate_subtitle_plane_snapshot(
                    self._pack_uint32_words(changed), **self._uip1_context())
        for packed in (0, 1, 2, 3, 4, 8, 9, 10, 11, 12):
            changed = good.copy()
            changed[coordinate.SUBTITLE_OCR.plane_warmup_word] = packed
            decoded = dump_contract.validate_subtitle_plane_snapshot(
                self._pack_uint32_words(changed), **self._uip1_context())
            self.assertEqual(decoded["warmup_packed"], packed)
            self.assertEqual(decoded["warmup_remaining"], packed & coordinate.SUBTITLE_OCR.plane_warmup_remaining_mask)
        for packed in (5, 6, 7, 13, 14, 15, 16, 0xffffffff):
            changed = good.copy()
            changed[coordinate.SUBTITLE_OCR.plane_warmup_word] = packed
            with self.subTest(packed=packed), self.assertRaises(ValueError):
                dump_contract.validate_subtitle_plane_snapshot(
                    self._pack_uint32_words(changed), **self._uip1_context())
        payload = self._pack_uint32_words(good)
        for bad in (payload[:-4], payload + bytes(4), list(good)):
            with self.subTest(kind=type(bad).__name__, length=len(bad)), self.assertRaises(ValueError):
                dump_contract.validate_subtitle_plane_snapshot(bad, **self._uip1_context())

    def test_uip1_rejects_unknown_flag_and_noncanonical_current_slot_masks(self):
        for flags, mask in ((2, 1), (3, 1), (0, 1), (1, 2), (1, 5), (1, 256), (1, 511)):
            words = self._valid_uip1_state_words()
            words[2:4] = [flags, mask]
            with self.subTest(flags=flags, mask=mask), self.assertRaisesRegex(ValueError, "flags/slot mask"):
                dump_contract.validate_subtitle_plane_snapshot(
                    self._pack_uint32_words(words), **self._uip1_context())

    def test_uip1_full_state_requires_current_ocr8_source_domain_join(self):
        one = self._parse_uip1_ocr_pairs([
            ((120, 350, 650, 401), (120, 350, 650, 401), False)])
        for key in ("matched_frame_id", "analysis_generation", "source_width", "source_height",
                    "field_width", "field_height", "roi_top", "roi_bottom", "tensor_content_rect"):
            changed = copy.deepcopy(one)
            changed[key] = (1, 0, 770, 434) if key == "tensor_content_rect" else changed[key] + 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "frame/domain identity"):
                dump_contract.validate_subtitle_plane_state(
                    self._pack_uint32_words(self._valid_uip1_state_words()),
                    ocr_record=changed, **self._uip1_context())

    def test_uip1_accepts_independent_held_or_disarmed_controller_clock(self):
        for accepted_clock, flags in ((999900, 8), (0, 0)):
            words = self._valid_uip1_state_words()
            words[25], words[28], words[29] = flags, accepted_clock, 0
            state = dump_contract.validate_subtitle_plane_snapshot(
                self._pack_uint32_words(words), **self._uip1_context())
            self.assertEqual(state["observation_timestamp_us"], 1000001)
            self.assertEqual(state["adaptive"]["accepted_observation_timestamp_us"], accepted_clock)
            self.assertEqual(state["current_count"], 1)

    def test_uip1_adaptive_controller_flags_clocks_levels_and_counts_are_canonical(self):
        float_bits = lambda value: struct.unpack("<I", struct.pack("<f", value))[0]
        faults = ((24, 2), (25, 16), (25, 3), (26, 21), (27, 0x7fc00000),
                  (27, float_bits(-0.001)), (27, float_bits(0.05)), (28, 0),
                  (30, 1), (32, 1), (34, 21), (35, 21), (36, float_bits(0.039)),
                  (37, 770 * 434 + 1), (38, 2), (39, 2), (25, 12))
        for index, value in faults:
            words = self._valid_uip1_state_words()
            words[index] = value
            with self.subTest(word=index, value=value), self.assertRaisesRegex(ValueError, "adaptive UI"):
                dump_contract.validate_subtitle_plane_snapshot(
                    self._pack_uint32_words(words), **self._uip1_context())
        for flagged, offset in ((9, 30), (10, 32)):
            words = self._valid_uip1_state_words()
            words[25], words[offset] = flagged, 1000002
            with self.subTest(flags=flagged), self.assertRaisesRegex(ValueError, "clock"):
                dump_contract.validate_subtitle_plane_snapshot(
                    self._pack_uint32_words(words), **self._uip1_context())

    def test_uip4_history_reset_mask_is_bound_to_current_slots(self):
        words = self._valid_uip1_state_words(count=8)
        words[coordinate.SUBTITLE_OCR.plane_history_reset_mask_word] = 0b10001001
        plane = dump_contract.validate_subtitle_plane_snapshot(
            self._pack_uint32_words(words), **self._uip1_context())
        self.assertEqual(plane["history_reset_mask"], 0b10001001)
        self.assertEqual(plane["current_slot_mask"], 255)
        for count, reset in ((0, 1), (1, 2), (8, 256), (8, 0xffffffff)):
            changed = self._valid_uip1_state_words(count=count)
            changed[coordinate.SUBTITLE_OCR.plane_history_reset_mask_word] = reset
            with self.subTest(count=count, reset=reset), self.assertRaisesRegex(ValueError, "canonical"):
                dump_contract.validate_subtitle_plane_snapshot(
                    self._pack_uint32_words(changed), **self._uip1_context())

    def test_uip4_invalid_base_probe_cannot_publish_history_reset(self):
        words = self._valid_uip1_state_words()
        words[coordinate.SUBTITLE_OCR.plane_history_reset_mask_word] = 1
        words[coordinate.SUBTITLE_OCR.plane_adaptive_offset + 15] = 1
        with self.assertRaisesRegex(ValueError, "invalid Base probe"):
            dump_contract.validate_subtitle_plane_snapshot(
                self._pack_uint32_words(words), **self._uip1_context())

    def test_uip1_inactive_controller_is_canonical_and_cannot_grant_current_covers(self):
        words = self._valid_uip1_state_words(count=0)
        words[24:] = [0] * 16
        dump_contract.validate_subtitle_plane_snapshot(
            self._pack_uint32_words(words), **self._uip1_context())
        for index in range(25, 40):
            changed = words.copy()
            changed[index] = 1
            with self.subTest(word=index), self.assertRaisesRegex(ValueError, "inactive adaptive"):
                dump_contract.validate_subtitle_plane_snapshot(
                    self._pack_uint32_words(changed), **self._uip1_context())
        words[3] = 1
        with self.assertRaisesRegex(ValueError, "active adaptive UI tail"):
            dump_contract.validate_subtitle_plane_snapshot(
                self._pack_uint32_words(words), **self._uip1_context())

    def test_scp4_active_params_require_exact_count_kind_level_and_applied_target_bits(self):
        ocr = self._parse_uip1_ocr_pairs([
            ((120, 350, 650, 401), (120, 350, 650, 401), False)])
        plane = dump_contract.validate_subtitle_plane_state(
            self._pack_uint32_words(self._valid_uip1_state_words()),
            ocr_record=ocr, **self._uip1_context())
        params = [4, coordinate.SUBTITLE_OCR.condition_param_tag, 1, 0, 2, plane["target_bits"]]
        for index, value in ((0, 3), (1, 0), (2, 0), (2, 2), (3, 1), (3, 2),
                             (4, 1), (4, 3), (5, plane["target_bits"] ^ 1)):
            changed = params.copy()
            changed[index] = value
            with self.subTest(word=index), self.assertRaisesRegex(ValueError, "SCP4"):
                dump_contract.validate_subtitle_condition_words(
                    self._pack_uint32_words(changed), plane_state=plane, geometry_ready=True)

    def test_scp4_inactive_payload_is_fresh_tagged_positive_zero(self):
        plane = dump_contract.validate_subtitle_plane_snapshot(
            self._pack_uint32_words(self._valid_uip1_state_words(count=0)), **self._uip1_context())
        params = [4, coordinate.SUBTITLE_OCR.condition_param_tag, 0, 0, 0, 0]
        for index in range(2, 6):
            for value in (1, 0x80000000):
                changed = params.copy()
                changed[index] = value
                with self.subTest(word=index, value=value), self.assertRaisesRegex(ValueError, "positive zero"):
                    dump_contract.validate_subtitle_condition_words(
                        self._pack_uint32_words(changed), plane_state=plane, geometry_ready=True)
        with self.assertRaisesRegex(ValueError, "schema/tag"):
            dump_contract.validate_subtitle_condition_words(bytes(24), plane_state=plane, geometry_ready=True)

    def test_uip1_ocr_projection_covers_all_calibrated_source_shapes_and_native_grids(self):
        for source_width, source_height, logical_width, logical_height in PRODUCTION_SOURCE_PROFILE_CASES:
            for multiplier in (1, 2):
                source = (source_width, source_height)
                field = (logical_width * multiplier, logical_height * multiplier)
                with self.subTest(source=source, field=field):
                    ocr = self._parse_uip1_ocr_pairs(source=source, field=field)
                    words = self._valid_uip1_state_words(source=source, field=field, count=0)
                    state = dump_contract.validate_subtitle_plane_state(
                        self._pack_uint32_words(words), ocr_record=ocr,
                        **self._uip1_context(source=source, field=field))
                    self.assertEqual((state["roi_top"], state["roi_bottom"]),
                                     (ocr["roi_top"], ocr["roi_bottom"]))
                    self.assertEqual(state["current_rectangles"], [])

    def test_fresh_resolver_metadata_rejects_retired_owner_qualification_and_provisional_policy(self):
        for key in ("qualification", "owner_policy", "pending_policy", "provisional_current_policy"):
            manifest = self._active_uip1_manifest()
            manifest["subtitle_conditioning"]["resolver"][key] = {"enabled": True}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "resolver provenance"):
                dump_contract.validate_v2_dump_manifest_document(manifest)

    def test_retired_locator_owner_and_rectangular_fade_apis_are_removed(self):
        for name in ("validate_subtitle_locator_state", "SUBTITLE_LOCATOR_STATE_SCHEMA",
                     "SUBTITLE_LOCATOR_FLAG_OWNER", "SUBTITLE_LOCATOR_FLAG_PENDING",
                     "_subtitle_selected_ocr_indices", "_subtitle_qualified_ocr_core",
                     "_replay_slr13_conditioner", "_replay_slr13_conditioner_sm5_candidates"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(dump_contract, name))


    def test_production_ui_state_cannot_inherit_an_adaptive_tail(self):
        words = self._valid_uip1_state_words()
        with self.assertRaisesRegex(ValueError, "invalid joint plane mode"):
            dump_contract._validate_adaptive_ui_tail(
                tuple(words), 1920, (0, 0, 770, 434), joint_plane_mode=0)


    def test_depth_input_region_accepts_real_integer_floor_roi_and_float32_fitter(self):
        region = copy.deepcopy(self.depth_input_region)
        region["mode"] = "window-region"
        region["authorization"] = {
            "authority_kind": "chromium-video",
            "observer_generation": 901,
            "hwnd": "0x60736",
            "process_id": 34056,
            "document_id": -28681,
            "video_id": -28624,
        }
        coordinate_space = region["coordinate_space"]
        coordinate_space["source_extent_px"] = {"width": 3840, "height": 2160}
        coordinate_space["semantic_rect_px"] = {
            "left": 277, "top": 415, "right": 2813, "bottom": 1842,
        }
        coordinate_space["inference_rect_px"] = {
            "left": 277, "top": 415, "right": 2813, "bottom": 1842,
        }
        plan = dump_contract._plan_host_sbs_v2_window_region(
            (277, 415, 2813, 1842), 3840, 2160, 770, 434)
        self.assertIsNotNone(plan)
        _, content, padding = plan
        region["analysis"].update({
            "analysis_generation": 17,
            "input_domain_reset": True,
            "tensor_content_rect_px": dict(zip(
                ("left", "top", "right", "bottom"), content)),
            "padded_area_fraction": padding,
            "fit_method":
                "centered-integer-contain-with-edge-replicated-excluded-padding",
            "crop_method": "direct retained-source rectangle sampling",
            "scene_analysis_domain": "inference-rectangle-only",
        })
        runtime_scale, vertical_slope = dump_contract._roi_renderer_constants(
            (277, 415, 2813, 1842), 3840, 2160, content)
        region["renderer"].update({
            "final_parallax_units": "roi-local-source-u",
            "full_source_parallax_scale": runtime_scale,
            "outside": {
                "construction": "signed soft-threshold collar",
                "horizontal_slope_source_u_per_source_u": 0.5,
                "vertical_slope_source_u_per_source_v": vertical_slope,
                "beyond_collar": "exact zero parallax",
            },
        })

        decoded = dump_contract.validate_depth_input_region_document(
            region, matched_frame_id=41, source_width=3840, source_height=2160,
            tensor_width=770, tensor_height=434)
        self.assertEqual(decoded["inference_width"], 2536)
        self.assertEqual(decoded["tensor_content_rect"], content)
        self.assertEqual(decoded["padded_area_fraction"], padding)
        self.assertEqual(decoded["analysis_generation"], 17)
        self.assertEqual(decoded["authorization"]["observer_generation"], 901)
        self.assertNotEqual(
            decoded["authorization"]["observer_generation"],
            decoded["analysis_generation"])
        self.assertEqual(
            dump_contract._fit_host_sbs_v2_depth_tensor_shape(1160, 496),
            (1008, 434))
        self.assertEqual(
            dump_contract._fit_host_sbs_v2_depth_tensor_shape(496, 872),
            (434, 756))

        foreground = copy.deepcopy(region)
        foreground["authorization"].update({
            "authority_kind": "foreground-client",
            "document_id": 0,
            "video_id": 0,
        })
        foreground_decoded = dump_contract.validate_depth_input_region_document(
            foreground, matched_frame_id=41, source_width=3840, source_height=2160,
            tensor_width=770, tensor_height=434)
        self.assertEqual(
            foreground_decoded["authorization"]["authority_kind"],
            "foreground-client")

        forged_foreground = copy.deepcopy(foreground)
        forged_foreground["authorization"]["video_id"] = -28624
        with self.assertRaisesRegex(ValueError, "foreground authority carries DOM identity"):
            dump_contract.validate_depth_input_region_document(forged_foreground)

        moved = copy.deepcopy(region)
        moved["coordinate_space"]["inference_rect_px"]["left"] += 1
        moved["coordinate_space"]["inference_rect_px"]["right"] += 1
        with self.assertRaisesRegex(ValueError, "deterministic authenticated integer contain fit"):
            dump_contract.validate_depth_input_region_document(moved)

        high_plan = dump_contract._plan_host_sbs_v2_window_region(
            (277, 415, 2813, 1842), 3840, 2160, 1540, 868)
        self.assertIsNotNone(high_plan)
        _, high_content, high_padding = high_plan
        self.assertEqual(high_content, tuple(2 * value for value in content))
        self.assertEqual(high_padding, padding)
        high_region = copy.deepcopy(region)
        high_region["analysis"]["tensor_extent_px"] = {
            "width": 1540, "height": 868}
        high_region["analysis"]["tensor_content_rect_px"] = dict(zip(
            ("left", "top", "right", "bottom"), high_content))
        high_region["analysis"]["padded_area_fraction"] = high_padding
        _, high_vertical_slope = dump_contract._roi_renderer_constants(
            (277, 415, 2813, 1842), 3840, 2160, high_content)
        high_region["renderer"]["outside"][
            "vertical_slope_source_u_per_source_v"] = high_vertical_slope
        high_decoded = dump_contract.validate_depth_input_region_document(
            high_region, matched_frame_id=41, source_width=3840, source_height=2160,
            tensor_width=1540, tensor_height=868)
        self.assertEqual(high_decoded["tensor_content_rect"], high_content)
        self.assertIsNone(dump_contract._plan_host_sbs_v2_window_region(
            (277, 415, 2813, 1842), 3840, 2160, 1568, 868))

    def test_depth_input_region_rejects_forged_tensor_content_scale_slope_and_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_synthetic_roi_geometry_dump(root)
            valid = json.loads((root / "depth_input_region.json").read_text())
            mutations = {
                "unknown-key": (lambda value: value.update({"future": 1}), "unknown layout"),
                "noncanonical-hwnd": (
                    lambda value: value["authorization"].update({"hwnd": "0xabcdef"}),
                    "canonical uppercase hexadecimal"),
                "wrong-tensor": (
                    lambda value: value["analysis"]["tensor_extent_px"].update({"width": 756}),
                    "does not match the exact source-derived"),
                "wrong-content": (
                    lambda value: value["analysis"]["tensor_content_rect_px"].update(
                        {"top": value["analysis"]["tensor_content_rect_px"]["top"] + 1}),
                    "deterministic authenticated integer contain fit"),
                "wrong-padding": (
                    lambda value: value["analysis"].update({"padded_area_fraction": 0.01}),
                    "inconsistent window-region analysis"),
                "wrong-scale": (
                    lambda value: value["renderer"].update({
                        "full_source_parallax_scale":
                            value["renderer"]["full_source_parallax_scale"] + 1.0e-5}),
                    "inconsistent window-region analysis"),
                "wrong-slope": (
                    lambda value: value["renderer"]["outside"].update({
                        "vertical_slope_source_u_per_source_v":
                            value["renderer"]["outside"]
                            ["vertical_slope_source_u_per_source_v"] + 1.0e-5}),
                    "inconsistent exterior collar semantics"),
            }
            for name, (mutate, expected) in mutations.items():
                with self.subTest(name=name):
                    changed = copy.deepcopy(valid)
                    mutate(changed)
                    with self.assertRaisesRegex(ValueError, expected):
                        dump_contract.validate_depth_input_region_document(changed)

    def test_manifest_rejects_missing_or_misattributed_vertical_intermediate(self):
        changed = copy.deepcopy(self.manifest)
        changed["artifacts"].pop("shadow_vertical_majorant.f32")
        with self.assertRaisesRegex(ValueError, "geometry artifact"):
            dump_contract.validate_v2_dump_manifest_document(changed)

    def test_manifest_requires_shadow_coordinate_descriptor_and_dimensions(self):
        changed = copy.deepcopy(self.manifest)
        changed["artifacts"].pop("shadow_coordinate.f32")
        with self.assertRaisesRegex(ValueError, "geometry artifact"):
            dump_contract.validate_v2_dump_manifest_document(changed)

        changed = copy.deepcopy(self.manifest)
        changed["dimensions"].pop("shadow_coordinate")
        with self.assertRaisesRegex(ValueError, "geometry dimension"):
            dump_contract.validate_v2_dump_manifest_document(changed)

    def test_manifest_requires_content_hashes_for_geometry_fields(self):
        for field in (
                "shadow_coordinate.f32",
                "shadow_candidate_parallax.f32",
                "shadow_final_parallax.f32"):
            changed = copy.deepcopy(self.manifest)
            del changed["artifacts"][field]["sha256"]
            with self.assertRaisesRegex(ValueError, "geometry artifact"):
                dump_contract.validate_v2_dump_manifest_document(changed)
            changed = copy.deepcopy(self.manifest)
            changed["artifacts"][field]["sha256"] = "not-a-hash"
            with self.assertRaisesRegex(ValueError, "content sha256"):
                dump_contract.validate_v2_dump_manifest_document(changed)
        # The current schema must not advertise packaged scalar previews at all.
        changed = copy.deepcopy(self.manifest)
        changed["artifacts"]["shadow_vertical_majorant.png"] = {
            "available": True, "required": False, "stage": "retired preview",
            "description": "retired", "sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "retired schema-38 artifact"):
            dump_contract.validate_v2_dump_manifest_document(changed)

    @staticmethod
    def _write_subtitle_conditioning_document(root, manifest):
        payload = json.dumps(manifest["subtitle_conditioning"]).encode("utf-8")
        (root / "subtitle_conditioning.json").write_bytes(payload)
        manifest["artifacts"]["subtitle_conditioning.json"]["sha256"] = (
            hashlib.sha256(payload).hexdigest())

    @staticmethod
    def _write_hashed_payload(root, manifest, name, payload):
        (root / name).write_bytes(payload)
        manifest["artifacts"][name]["sha256"] = hashlib.sha256(payload).hexdigest()

    def _gpu_trace_ring_payload(
            self, input_region_document, field_width, field_height, *,
            slot=0, sequence=1, dump_forced_at_enqueue=False):
        parsed = dump_contract.validate_depth_input_region_document(
            input_region_document,
            matched_frame_id=self.manifest["matched_frame_id"],
            source_width=input_region_document["coordinate_space"]["source_extent_px"]["width"],
            source_height=input_region_document["coordinate_space"]["source_extent_px"]["height"],
            tensor_width=field_width,
            tensor_height=field_height,
        )
        domain_tag = dump_contract._gpu_trace_domain_tag(
            parsed, "srgb", field_width, field_height)
        token = 0x1020304050607080
        token_low, token_high = token & 0xFFFFFFFF, token >> 32
        work = dump_contract.GPU_TRACE_WORK_SUBTITLE_OBSERVATION
        transaction = [0] * dump_contract.GPU_TRACE_TRANSACTION_WORD_COUNT
        transaction[:8] = [
            1,
            1 ^ dump_contract.GPU_TRACE_DECISION_COOKIE,
            token_low,
            token_high,
            token_low ^ dump_contract.GPU_TRACE_TOKEN_LOW_COOKIE,
            token_high ^ dump_contract.GPU_TRACE_TOKEN_HIGH_COOKIE,
            dump_contract.GPU_TRACE_RECEIPT_MAGIC,
            0,
        ]
        transaction[8:16] = [
            token_low,
            token_high,
            token_low ^ dump_contract.GPU_TRACE_TOKEN_LOW_COOKIE,
            token_high ^ dump_contract.GPU_TRACE_TOKEN_HIGH_COOKIE,
            dump_contract.GPU_TRACE_REQUEST_MAGIC,
            work,
            work ^ dump_contract.GPU_TRACE_WORK_FLAGS_COOKIE,
            0,
        ]
        timestamp = self.manifest["matched_observation_timestamp_us"]
        locator = self._valid_uip1_state_words(
            frame=self.manifest["matched_frame_id"], generation=parsed["analysis_generation"],
            timestamp=timestamp, source=(parsed["inference_width"], parsed["inference_height"]),
            field=(field_width, field_height), content=parsed["tensor_content_rect"],
            count=0, current_valid=False)
        condition = [coordinate.SUBTITLE_OCR.condition_param_schema,
                     coordinate.SUBTITLE_OCR.condition_param_tag, 0, 0, 0, 0]
        flags = (dump_contract.GPU_TRACE_FLAG_OCR_RECORD_SUBMITTED |
                 dump_contract.GPU_TRACE_FLAG_CONDITION_EXECUTED)
        if parsed["input_domain_reset"]:
            flags |= dump_contract.GPU_TRACE_FLAG_INPUT_DOMAIN_RESET
        if dump_forced_at_enqueue:
            flags |= dump_contract.GPU_TRACE_FLAG_DUMP_FORCED_AT_ENQUEUE
        record = [0] * dump_contract.GPU_TRACE_RECORD_WORD_COUNT
        record[:24] = [
            dump_contract.GPU_TRACE_RING_SCHEMA,
            dump_contract.GPU_TRACE_RECORD_TAG,
            sequence & 0xFFFFFFFF,
            sequence >> 32,
            self.manifest["matched_frame_id"],
            0,
            parsed["analysis_generation"] & 0xFFFFFFFF,
            parsed["analysis_generation"] >> 32,
            domain_tag & 0xFFFFFFFF,
            domain_tag >> 32,
            token_low,
            token_high,
            1,
            2,
            work,
            2,
            flags,
            1,
            parsed["inference_width"],
            parsed["inference_height"],
            field_width,
            field_height,
            dump_contract.GPU_TRACE_TRANSACTION_WORD_COUNT,
            0,
        ]
        record[24:88] = transaction
        offsets = dump_contract.GPU_TRACE_RECORD_OFFSETS
        record[offsets["subtitle_locator_begin"]:offsets["subtitle_condition_begin"]] = locator
        record[offsets["subtitle_condition_begin"]:offsets["observation_timestamp_low"]] = condition
        observation_timestamp_us = self.manifest["matched_observation_timestamp_us"]
        record[offsets["observation_timestamp_low"]] = observation_timestamp_us & 0xFFFFFFFF
        record[offsets["observation_timestamp_high"]] = observation_timestamp_us >> 32
        ring = [0] * dump_contract.GPU_TRACE_RING_WORD_COUNT
        ring[:8] = [
            dump_contract.GPU_TRACE_RING_SCHEMA,
            dump_contract.GPU_TRACE_RING_TAG,
            dump_contract.GPU_TRACE_CAPACITY,
            dump_contract.GPU_TRACE_RECORD_WORD_COUNT,
            (sequence + 1) & 0xFFFFFFFF,
            (sequence + 1) >> 32,
            (slot + 1) % dump_contract.GPU_TRACE_CAPACITY,
            1,
        ]
        base = (dump_contract.GPU_TRACE_HEADER_WORD_COUNT +
                slot * dump_contract.GPU_TRACE_RECORD_WORD_COUNT)
        ring[base:base + dump_contract.GPU_TRACE_RECORD_WORD_COUNT] = record
        return self._pack_uint32_words(ring), parsed, domain_tag


    @staticmethod
    def _set_gpu_trace_reuse(words, *, work, host_outcome):
        base = dump_contract.GPU_TRACE_HEADER_WORD_COUNT
        transaction = base + dump_contract.GPU_TRACE_RECORD_OFFSETS["transaction_begin"]
        words[base + dump_contract.GPU_TRACE_RECORD_OFFSETS["submission_class"]] = 2
        words[base + dump_contract.GPU_TRACE_RECORD_OFFSETS["depth_disposition"]] = 1
        words[base + dump_contract.GPU_TRACE_RECORD_OFFSETS["expected_work"]] = work
        flags = base + dump_contract.GPU_TRACE_RECORD_OFFSETS["flags"]
        words[flags] &= ~dump_contract.GPU_TRACE_FLAG_SUBTITLE_SUPPRESSED
        if work == dump_contract.GPU_TRACE_WORK_NONE:
            disposition = 0
            words[flags] &= ~(
                dump_contract.GPU_TRACE_FLAG_OCR_RECORD_SUBMITTED |
                dump_contract.GPU_TRACE_FLAG_CONDITION_EXECUTED |
                dump_contract.GPU_TRACE_FLAG_SUBTITLE_BRANCH_GATED)
            words[flags] |= dump_contract.GPU_TRACE_FLAG_SUBTITLE_SUPPRESSED
        else:
            words[flags] &= ~(
                dump_contract.GPU_TRACE_FLAG_OCR_RECORD_SUBMITTED |
                dump_contract.GPU_TRACE_FLAG_CONDITION_EXECUTED)
            words[flags] |= dump_contract.GPU_TRACE_FLAG_SUBTITLE_BRANCH_GATED
            disposition = 5
            locator = base + dump_contract.GPU_TRACE_RECORD_OFFSETS[
                "subtitle_locator_begin"]
            frame = words[base + dump_contract.GPU_TRACE_RECORD_OFFSETS["frame_low"]]
            words[locator + 4] = frame - 1
            words[locator + 5] = 0
            words[locator + 8] -= 1
            words[locator + 28] -= 1
        words[base + dump_contract.GPU_TRACE_RECORD_OFFSETS[
            "subtitle_disposition"]] = disposition
        words[base + dump_contract.GPU_TRACE_RECORD_OFFSETS["host_subtitle_outcome"]] = (
            host_outcome)
        words[transaction + 0] = 0
        words[transaction + 1] = dump_contract.GPU_TRACE_DECISION_COOKIE
        words[transaction + 7] = 0
        words[transaction + 13] = work
        words[transaction + 14] = (
            0 if work == dump_contract.GPU_TRACE_WORK_NONE else
            work ^ dump_contract.GPU_TRACE_WORK_FLAGS_COOKIE)
        return base, transaction

    @staticmethod
    def _gpu_trace_contract_document():
        return {
            "schema": dump_contract.GPU_TRACE_CONTRACT_SCHEMA,
            "role": (
                "diagnostic-only accepted-root completion history; never rendering authority"),
            "byte_order": "little-endian",
            **copy.deepcopy(dump_contract._gpu_trace_contract_expected_sections()),
        }

    def _attach_gpu_trace(self, root, manifest, input_region_document, width, height):
        raw, parsed, domain_tag = self._gpu_trace_ring_payload(
            input_region_document, width, height)
        decoded = dump_contract.validate_gpu_trace_ring(
            raw,
            matched_frame_id=manifest["matched_frame_id"],
            analysis_generation=parsed["analysis_generation"],
            source_width=parsed["inference_width"],
            source_height=parsed["inference_height"],
            field_width=width,
            field_height=height,
            expected_domain_tag=domain_tag,
            input_domain_reset=parsed["input_domain_reset"],
        )
        documents = {
            "gpu_trace_ring.u32": raw,
            "gpu_trace.json": (json.dumps(decoded["decoded"], indent=2) + "\n").encode(),
            "gpu_trace_contract.json": (
                json.dumps(self._gpu_trace_contract_document(), indent=2) + "\n").encode(),
        }
        for name, payload in documents.items():
            (root / name).write_bytes(payload)
            manifest["artifacts"][name].update({
                "available": True,
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
        manifest["gpu_trace"] = {
            "available": True,
            "required": False,
            "rendering_authority": False,
            "raw_artifact": "gpu_trace_ring.u32",
            "decoded_artifact": "gpu_trace.json",
            "contract_artifact": "gpu_trace_contract.json",
            "record_count": decoded["record_count"],
            "oldest_sequence": decoded["oldest_sequence"],
            "next_sequence": decoded["next_sequence"],
            "matched_sequence": decoded["matched_sequence"],
            "source_closure_sha256": dump_contract.GPU_TRACE_SOURCE_CLOSURE_SHA256,
        }
        (root / "dump_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return raw, decoded

    def _write_synthetic_geometry_dump(self, root, width=16, height=12):
        import numpy as np

        rng = np.random.default_rng(20260804)
        candidate = (rng.uniform(-0.002, 0.03, (height, width))).astype(np.float32)

        majorant, conditioned, final = dump_contract._replay_v2_limiter_fields(
            candidate, width)

        manifest = copy.deepcopy(self.manifest)
        for name in (
                "model_input", "raw_depth",
                "shadow_coordinate",
                "shadow_candidate_parallax",
                "shadow_vertical_majorant", "shadow_vertical_conditioned",
                "shadow_final_parallax"):
            manifest["dimensions"][name] = dict(
                manifest["dimensions"][name], width=width, height=height)
        input_region = copy.deepcopy(self.depth_input_region)
        input_region["analysis"]["tensor_extent_px"] = {
            "width": width, "height": height,
        }
        input_region["analysis"]["tensor_content_rect_px"] = {
            "left": 0, "top": 0, "right": width, "bottom": height,
        }
        fields = {
            "shadow_coordinate": np.broadcast_to(
                np.linspace(0.0, 1.0, width, dtype=np.float32),
                (height, width)).copy(),
            "shadow_candidate_parallax": candidate,
            "shadow_vertical_majorant": majorant,
            "shadow_vertical_conditioned": conditioned,
            "shadow_final_parallax": final,
        }
        for name, values in fields.items():
            payload = values.astype("<f4").tobytes()
            (root / f"{name}.f32").write_bytes(payload)
            manifest["artifacts"][f"{name}.f32"]["sha256"] = hashlib.sha256(
                payload).hexdigest()
        region_payload = json.dumps(input_region).encode("utf-8")
        (root / "depth_input_region.json").write_bytes(region_payload)
        manifest["artifacts"]["depth_input_region.json"]["sha256"] = hashlib.sha256(
            region_payload).hexdigest()
        self._write_subtitle_conditioning_document(root, manifest)
        state_payload = json.dumps(self.live_state).encode("utf-8")
        (root / "shadow_state.json").write_bytes(state_payload)
        manifest["artifacts"]["shadow_state.json"]["sha256"] = hashlib.sha256(
            state_payload).hexdigest()
        frame_stats = copy.deepcopy(self.frame_stats)
        frame_stats["named_values"]["valid_count"] = float(width * height)
        frame_stats["named_values"]["texel_count"] = float(width * height)
        stats_payload = json.dumps(frame_stats).encode("utf-8")
        (root / "shadow_frame_stats.json").write_bytes(stats_payload)
        manifest["artifacts"]["shadow_frame_stats.json"]["sha256"] = hashlib.sha256(
            stats_payload).hexdigest()
        (root / "dump_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        return manifest, fields

    def _write_synthetic_contour_tuple(
            self, root, manifest, ocr, locator, *, geometry_ready=True):
        import numpy as np

        policy = coordinate.EXPECTED_SUBTITLE_OCR["contour_policy"]
        source_width = manifest["dimensions"]["analysis_source"]["width"]
        source_height = manifest["dimensions"]["analysis_source"]["height"]
        field_width = manifest["dimensions"]["shadow_final_parallax"]["width"]
        field_height = manifest["dimensions"]["shadow_final_parallax"]["height"]
        crop_height = min(source_height, (source_width + 5) // 6)
        crop_top = source_height - crop_height
        region = json.loads((root / "depth_input_region.json").read_text(encoding="utf-8"))
        content = tuple(region["analysis"]["tensor_content_rect_px"][key]
                        for key in ("left", "top", "right", "bottom"))
        eligible = max(source_width, source_height) <= policy["source_max_extent"]
        shape = ((crop_height, source_width) if eligible else (1, 1))
        snapshot_gate = eligible and geometry_ready and ocr[2] == 1
        valid = bool(snapshot_gate and locator[2] == 1 and locator[3] and locator[39] == 0)
        support = np.zeros(shape, dtype="<u4")
        source = np.zeros((*shape, 4), dtype="<f2")
        history = np.zeros((*shape, 4), dtype="<f4")
        probability = np.zeros(coordinate.SUBTITLE_OCR.output_shape, dtype="<f4")
        if snapshot_gate:
            source[..., :3] = np.float16(0.25)
            source[..., 3] = np.float16(1)
            probability.fill(np.float32(0.6))
        if valid:
            offset = dump_contract.SUBTITLE_OCR_FINAL_BOX_WORD_OFFSET
            left, top, right, bottom = ocr[offset:offset + 4]
            # A single independently authored stable pixel, strictly inside its cover.
            px = ((left + right) // 2 - content[0]) * source_width // (content[2] - content[0])
            py = ((top + bottom) // 2 - content[1]) * source_height // (content[3] - content[1]) - crop_top
            support[py, px] = 1
            source[py, px, :3] = np.float16(0.875)
        if snapshot_gate:
            history[..., :3] = source[..., :3].astype(np.float32)
            history[..., 3] = np.float32(policy["history_feature_valid_bit"])
            history[..., 3][support != 0] = np.float32(
                policy["history_feature_valid_bit"] | policy["history_feature_mask_bit"] |
                (policy["history_confirmation_observations"] << policy["history_feature_count_shift"]))
        pin = (dump_contract._subtitle_contour_project_support(
            support != 0, source_height, crop_top, field_width, field_height, content)
               if valid else np.zeros((field_height, field_width), dtype=bool))
        distance = dump_contract._subtitle_contour_squared_source_distance(
            support != 0, source_width, source_height, crop_top, field_width, field_height, content).astype("<f4")
        timestamp = manifest["matched_observation_timestamp_us"]
        identity = [policy["history_schema"], policy["history_tag"], int(valid), locator[20] if snapshot_gate else 0,
                    ocr[5], ocr[6], ocr[7], ocr[8], source_width, source_height, crop_top, crop_height,
                    field_width, field_height, ocr[13], ocr[14], *content,
                    timestamp & 0xffffffff, timestamp >> 32, policy["recipe"], int(snapshot_gate), *([0] * 8)]
        payloads = {
            "contour_support": support.tobytes(), "contour_field_pin": pin.astype("<u4").tobytes(),
            "contour_distance": distance.astype("<f4").tobytes(), "contour_identity": self._pack_uint32_words(identity),
            "contour_history": history.tobytes(), "contour_source": source.tobytes(), "ocr_probability": probability.tobytes(),
        }
        base = np.frombuffer((root / "shadow_base_final_parallax.f32").read_bytes(), dtype="<f4").reshape(field_height, field_width)
        cx = np.clip(np.arange(field_width), content[0], content[2] - 1)
        cy = np.clip(np.arange(field_height), content[1], content[3] - 1)
        joined_base = base[np.ix_(cy, cx)]
        enabled = bool(valid and np.isfinite(distance[content[1], content[0]]))
        weight = (dump_contract._subtitle_gaussian_weight_from_q(distance[np.ix_(cy, cx)], source_width, source_height)
                  if enabled else np.zeros_like(base))
        target = np.asarray(locator[27], dtype=np.uint32).view(np.float32)
        proposal = dump_contract._subtitle_gaussian_literal_blend(joined_base, weight, target) if enabled else joined_base.copy()
        if enabled:
            _, vertical, final = dump_contract._replay_v2_limiter_fields(proposal, content[2] - content[0])
        else:
            vertical = final = joined_base.copy()
        payloads.update(gaussian_weight=weight.astype("<f4").tobytes(), gaussian_blend=proposal.astype("<f4").tobytes(),
            gaussian_vertical_conditioned=vertical.astype("<f4").tobytes(), gaussian_control=struct.pack("<I", int(enabled)))
        for role, payload in payloads.items():
            self._write_hashed_payload(root, manifest, dump_contract._SUBTITLE_CONTOUR_ARTIFACT_FILES[role], payload)
        self._write_hashed_payload(root, manifest, "shadow_final_parallax.f32", final.astype("<f4").tobytes())
        return distance, valid


    def _activate_synthetic_uip1_dump(self, root, manifest, fields):
        manifest = self._active_uip1_manifest(manifest)
        self._write_hashed_payload(root, manifest, "shadow_base_final_parallax.f32",
                                   fields["shadow_final_parallax"].astype("<f4").tobytes())
        ocr = [0] * dump_contract.SUBTITLE_OCR_RECORD_WORD_COUNT
        ocr[:16] = [dump_contract.SUBTITLE_OCR_RECORD_SCHEMA, dump_contract.SUBTITLE_OCR_RECORD_TAG,
                    1, 0, 0, 41, 0, 0, 0, 1920, 1080, 770, 434, 325, 430, 0]
        plane = self._valid_uip1_state_words(generation=0, count=0,
                                             timestamp=manifest["matched_observation_timestamp_us"])
        for name, words in (("subtitle_ocr_record.u32", ocr), ("subtitle_locator_state.u32", plane)):
            self._write_hashed_payload(root, manifest, name, self._pack_uint32_words(words))
        self._write_synthetic_contour_tuple(root, manifest, ocr, plane)
        self._write_subtitle_conditioning_document(root, manifest)
        (root / "dump_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest


    def _activate_synthetic_uip1_current_dump(self, root, manifest, fields):
        import numpy as np

        manifest = self._activate_synthetic_uip1_dump(root, manifest, fields)
        ocr = self._valid_ocr_record_words()
        ocr[7:9] = [0, 0]
        plane = self._valid_uip1_state_words(generation=0,
                                             timestamp=manifest["matched_observation_timestamp_us"])
        for name, words in (("subtitle_ocr_record.u32", ocr), ("subtitle_locator_state.u32", plane)):
            self._write_hashed_payload(root, manifest, name, self._pack_uint32_words(words))
        self._write_synthetic_contour_tuple(root, manifest, ocr, plane)
        conditioned = np.frombuffer((root / "shadow_final_parallax.f32").read_bytes(), dtype="<f4").reshape(fields["shadow_final_parallax"].shape)
        self._write_subtitle_conditioning_document(root, manifest)
        (root / "dump_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest, conditioned, plane, ocr


    def _write_synthetic_roi_geometry_dump(self, root, *, near_full=False):
        import hashlib

        import numpy as np

        source_width, source_height = ((774, 436) if near_full else (960, 540))
        tensor_width, tensor_height = 770, 434
        if near_full:
            semantic = {"left": 1, "top": 1, "right": 773, "bottom": 435}
        else:
            # A deliberately tall arbitrary-aspect client: production keeps all 300x434 source
            # pixels and centers them in a 300-cell-wide content strip inside the 770x434 tensor.
            semantic = {"left": 330, "top": 53, "right": 630, "bottom": 487}
        inference = dict(semantic)
        plan = dump_contract._plan_host_sbs_v2_window_region(
            tuple(semantic[key] for key in ("left", "top", "right", "bottom")),
            source_width, source_height, tensor_width, tensor_height)
        self.assertIsNotNone(plan)
        _, tensor_content, padded_fraction = plan

        manifest = copy.deepcopy(self.manifest)
        manifest["depth_input_region"]["mode"] = "window-region"
        manifest["renderer"].update({
            "authority":
                "authenticated crop-local atomic final field plus depth-input-region embedding",
            "mapping_artifacts_match_selected_renderer": True,
            "parallax_v2_position_field":
                "shadow_final_parallax + depth_input_region embedding",
            "parallax_v2_conditioner_role":
                "least row-wise crop-local q >= shadow_vertical_conditioned with horizontal "
                "slope <= max_horizontal_slope and vertical shear <= max_vertical_shear "
                "publishes shadow_final_parallax atomically as direct live authority with "
                "depth_input_region embedding",
        })
        manifest["dimensions"].update({
            "source": {
                "width": source_width, "height": source_height,
                "format": "DXGI_FORMAT_R16G16B16A16_FLOAT", "format_value": 10,
            },
            "analysis_source": {
                "width": inference["right"] - inference["left"],
                "height": inference["bottom"] - inference["top"],
                "format": "DXGI_FORMAT_R16G16B16A16_FLOAT", "format_value": 10,
            },
            "packed_sbs": {
                "width": 2 * source_width, "height": source_height,
                "format": "DXGI_FORMAT_B8G8R8A8_UNORM", "format_value": 87,
            },
            "eye": {"width": source_width, "height": source_height},
            "content_fit": {"scale_x": 1.0, "scale_y": 1.0},
            "warp_map": {
                "width": 2 * source_width, "height": source_height,
                "format": "DXGI_FORMAT_R32_FLOAT", "format_value": 41,
            },
            "warp_mask": {
                "width": 2 * source_width, "height": source_height,
                "format": "DXGI_FORMAT_B8G8R8A8_UNORM", "format_value": 87,
            },
        })
        for name in (
                "model_input", "raw_depth",
                "shadow_coordinate",
                "shadow_candidate_parallax",
                "shadow_vertical_majorant", "shadow_vertical_conditioned",
                "shadow_final_parallax"):
            manifest["dimensions"][name] = dict(
                manifest["dimensions"][name],
                width=tensor_width, height=tensor_height)

        constant = np.full(
            (tensor_height, tensor_width), 0.01 if near_full else 0.001,
            dtype=np.float32)
        majorant, conditioned, final = dump_contract._replay_v2_limiter_fields(
            constant, tensor_content[2] - tensor_content[0])
        fields = {
            "shadow_coordinate": np.broadcast_to(
                np.linspace(0.0, 1.0, tensor_width, dtype=np.float32),
                (tensor_height, tensor_width)).copy(),
            "shadow_candidate_parallax": constant,
            "shadow_vertical_majorant": majorant,
            "shadow_vertical_conditioned": conditioned,
            "shadow_final_parallax": final,
        }
        for name, values in fields.items():
            payload = values.astype("<f4").tobytes()
            (root / f"{name}.f32").write_bytes(payload)
            manifest["artifacts"][f"{name}.f32"]["sha256"] = hashlib.sha256(
                payload).hexdigest()
        runtime_scale, vertical_slope = dump_contract._roi_renderer_constants(
            (inference["left"], inference["top"],
             inference["right"], inference["bottom"]),
            source_width, source_height, tensor_content)
        region = copy.deepcopy(self.depth_input_region)
        region.update({
            "mode": "window-region",
            "authorization": {
                "authority_kind": "chromium-video",
                "observer_generation": 901,
                "hwnd": "0x60736",
                "process_id": 34056,
                "document_id": -28681,
                "video_id": -28624,
            },
        })
        region["coordinate_space"].update({
            "source_extent_px": {"width": source_width, "height": source_height},
            "semantic_rect_px": semantic,
            "inference_rect_px": inference,
        })
        region["analysis"].update({
            "analysis_generation": 17,
            "input_domain_reset": True,
            "tensor_extent_px": {"width": tensor_width, "height": tensor_height},
            "tensor_content_rect_px": dict(zip(
                ("left", "top", "right", "bottom"), tensor_content)),
            "padded_area_fraction": padded_fraction,
            "fit_method":
                "centered-integer-contain-with-edge-replicated-excluded-padding",
            "crop_method": "direct retained-source rectangle sampling",
            "scene_analysis_domain": "inference-rectangle-only",
        })
        region["renderer"].update({
            "final_parallax_units": "roi-local-source-u",
            "full_source_parallax_scale": runtime_scale,
            "outside": {
                "construction": "signed soft-threshold collar",
                "horizontal_slope_source_u_per_source_u": 0.5,
                "vertical_slope_source_u_per_source_v": vertical_slope,
                "beyond_collar": "exact zero parallax",
            },
        })
        region_payload = json.dumps(region).encode("utf-8")
        (root / "depth_input_region.json").write_bytes(region_payload)
        manifest["artifacts"]["depth_input_region.json"]["sha256"] = hashlib.sha256(
            region_payload).hexdigest()

        border = {
            "schema": dump_contract.WINDOW_REGION_SCHEMA,
            "capture":
                "same matched source/color/depth/render frame as the parent Dump 3D package",
            "role":
                "matched-window region provenance; no independent geometry or renderer authority",
            "authority_kind": "chromium-video",
            "matched_frame_id": 41,
            "coordinate_space": {
                "name": "matched-source-pixels",
                "rect_semantics": "half-open [left, top, right, bottom)",
                "source_extent_px": {"width": source_width, "height": source_height},
                "capture_rect_px": semantic,
            },
            "identity": {
                "hwnd": "0x60736",
                "process_id": 34056,
                "document_id": -28681,
                "video_id": -28624,
                "generation": 901,
            },
            "freshness": {
                "latest_observation_age_ms_at_capture": 3,
                "maximum_observation_age_ms": 2500,
                "geometry_continuity_ms_at_capture": 1000,
                "source_content_age_ms_at_capture": 3,
                "fresh": True,
                "causal_geometry": True,
            },
        }
        border_payload = json.dumps(border).encode("utf-8")
        (root / "window_region.json").write_bytes(border_payload)
        manifest["window_region"] = {
            "available": True,
            "artifact": "window_region.json",
            "observer_status": "ok",
            "mapping_status": "ok",
            "geometry_authority": False,
            "renderer_authority": False,
        }
        manifest["artifacts"]["window_region.json"] = {
            "available": True,
            "required": True,
            "stage": "matched-frame window region provenance",
            "description": "authenticated test window region",
            "sha256": hashlib.sha256(border_payload).hexdigest(),
        }

        packed_u = ((np.arange(2 * source_width, dtype=np.float32) + np.float32(0.5)) /
                    np.float32(2 * source_width))
        right_eye = packed_u > np.float32(0.5)
        unwarped = np.where(
            right_eye,
            (packed_u - np.float32(0.5)) * np.float32(2.0),
            packed_u * np.float32(2.0),
        ).astype(np.float32)
        eye_sign = np.where(right_eye, np.float32(1.0), np.float32(-1.0))
        roi_left = np.float32(inference["left"]) / np.float32(source_width)
        roi_top = np.float32(inference["top"]) / np.float32(source_height)
        roi_right = np.float32(inference["right"]) / np.float32(source_width)
        roi_bottom = np.float32(inference["bottom"]) / np.float32(source_height)
        embedded = np.float32(runtime_scale) * constant[0, 0]
        warp_map = np.empty((source_height, 2 * source_width), dtype=np.float32)
        for row in range(source_height):
            source_v = np.float32((row + 0.5) / source_height)
            projected_v = np.clip(source_v, roi_top, roi_bottom)
            outside_y = np.abs(source_v - projected_v)
            source_x = unwarped.copy()
            for _ in range(11):
                projected_x = np.clip(source_x, roi_left, roi_right)
                budget = (np.float32(0.5) * np.abs(source_x - projected_x) +
                          np.float32(vertical_slope) * outside_y)
                effective = np.where(
                    budget < np.abs(embedded),
                    np.copysign(np.abs(embedded) - budget, embedded),
                    np.float32(0.0),
                ).astype(np.float32)
                source_x = (unwarped + eye_sign * effective).astype(np.float32)
            warp_map[row] = source_x
        map_payload = warp_map.astype("<f4").tobytes()
        (root / "warp_map.f32").write_bytes(map_payload)
        manifest["artifacts"]["warp_map.f32"] = {
            "available": True,
            "required": True,
            "stage": "exact full-source inverse-warp mapping",
            "description": "authenticated test full-source inverse map",
            "sha256": hashlib.sha256(map_payload).hexdigest(),
        }
        manifest["warp_map_contract"] = {
            "available": True,
            "schema": 2,
            "artifact": "warp_map.f32",
            "width": 2 * source_width,
            "height": source_height,
            "eye_width": source_width,
            "eye_height": source_height,
            "source_width": source_width,
            "source_height": source_height,
            "content_scale_x": 1.0,
            "content_scale_y": 1.0,
            "dtype": "float32-le",
            "layout": "row-major",
            "channels": ["raw_reproject_source_u_normalized"],
            "validity": {
                "content":
                    "derive from content_scale_x/content_scale_y and packed output coordinate",
                "inverse":
                    "11-step contractive fixed-point solution of crop-local q embedded by "
                    "depth_input_region.json scale and outside-only zero-plane collar",
                "mask": (
                    "warp_mask.png red marks finite-source boundary extrapolation; V2 has no "
                    "internal owner or synthetic-fill path"),
            },
            "live_sample_source_u_normalized":
                "clamp(raw_reproject_source_u_normalized, 0, 1)",
            "derived_inverse_displacement_output_eye_px":
                "(raw_reproject_source_u_normalized - aspect_fitted_unwarped_source_u) * content_scale_x * eye_width",
            "derived_signed_binocular_disparity_px":
                "invert both eye maps at common source-U samples; x_right - x_left",
        }
        import io
        from PIL import Image

        mask = np.zeros((source_height, 2 * source_width, 3), dtype=np.uint8)
        mask[:, :, 0] = np.where((warp_map < 0) | (warp_map > 1), 255, 0).astype(np.uint8)
        encoded_mask = io.BytesIO()
        Image.fromarray(mask).save(encoded_mask, format="PNG")
        mask_payload = encoded_mask.getvalue()
        (root / "warp_mask.png").write_bytes(mask_payload)
        manifest["artifacts"]["warp_mask.png"] = {
            "available": True,
            "required": True,
            "stage": "V2 boundary-extrapolation mask",
            "description": "authenticated test boundary mask",
            "sha256": hashlib.sha256(mask_payload).hexdigest(),
        }
        self._write_subtitle_conditioning_document(root, manifest)
        roi_state = copy.deepcopy(self.live_state)
        roi_state["units"] = {
            "coordinate": "dimensionless canonical coordinate derived from raw depth",
            "gain": "one-eye ROI-local source-U per coordinate unit",
            "parallax": (
                "signed one-eye ROI-local source-U; full-source renderer authority "
                "additionally requires depth_input_region embedding"),
        }
        state_payload = json.dumps(roi_state).encode("utf-8")
        (root / "shadow_state.json").write_bytes(state_payload)
        manifest["artifacts"]["shadow_state.json"]["sha256"] = hashlib.sha256(
            state_payload).hexdigest()
        frame_stats = copy.deepcopy(self.frame_stats)
        content_texel_count = (
            (tensor_content[2] - tensor_content[0]) *
            (tensor_content[3] - tensor_content[1]))
        frame_stats["named_values"]["valid_count"] = float(content_texel_count)
        frame_stats["named_values"]["texel_count"] = float(content_texel_count)
        stats_payload = json.dumps(frame_stats).encode("utf-8")
        (root / "shadow_frame_stats.json").write_bytes(stats_payload)
        manifest["artifacts"]["shadow_frame_stats.json"]["sha256"] = hashlib.sha256(
            stats_payload).hexdigest()
        (root / "dump_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        return manifest, region, warp_map

    @staticmethod
    def _advertise_window_region(root, manifest, *, available=True):
        manifest["matched_frame_id"] = 41
        manifest["dimensions"]["source"] = {
            "width": 1920,
            "height": 1080,
            "format": "DXGI_FORMAT_B8G8R8A8_UNORM",
            "format_value": 87,
        }
        manifest["window_region"] = {
            "available": available,
            "artifact": "window_region.json" if available else None,
            "observer_status": "ok" if available else "not-observed",
            "mapping_status": "ok" if available else "not-mapped",
            "geometry_authority": False,
            "renderer_authority": False,
        }
        manifest["artifacts"]["window_region.json"] = {
            "available": available,
            "required": False,
            "stage": "matched-frame window region provenance",
            "description": "diagnostic test artifact",
        }
        border = {
            "schema": dump_contract.WINDOW_REGION_SCHEMA,
            "capture":
                "same matched source/color/depth/render frame as the parent Dump 3D package",
            "role":
                "matched-window region provenance; no independent geometry or renderer authority",
            "authority_kind": "chromium-video",
            "matched_frame_id": 41,
            "coordinate_space": {
                "name": "matched-source-pixels",
                "rect_semantics": "half-open [left, top, right, bottom)",
                "source_extent_px": {"width": 1920, "height": 1080},
                "capture_rect_px": {
                    "left": 160, "top": 90, "right": 1760, "bottom": 990,
                },
            },
            "identity": {
                "hwnd": "0x1234",
                "process_id": 55,
                "document_id": -7,
                "video_id": -9,
                "generation": 3,
            },
            "freshness": {
                "latest_observation_age_ms_at_capture": 120,
                "maximum_observation_age_ms": 1000,
                "geometry_continuity_ms_at_capture": 5000,
                "source_content_age_ms_at_capture": 1000,
                "fresh": True,
                "causal_geometry": True,
            },
        }
        if available:
            (root / "window_region.json").write_text(
                json.dumps(border), encoding="utf-8")
        (root / "dump_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        return border

    def test_geometry_verifier_accepts_consistent_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_synthetic_geometry_dump(root)
            summary = dump_contract.verify_v2_dump_geometry(root)
            self.assertEqual(summary["width"], 16)
            self.assertEqual(summary["height"], 12)
            self.assertFalse(summary["window_region_verified"])

    def test_schema46_generates_float_previews_on_demand_outside_package(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dump"
            previews = Path(temporary) / "previews"
            root.mkdir()
            self._write_synthetic_geometry_dump(root)

            summary = dump_contract.generate_float_artifact_previews(
                root, "shadow_final_parallax.f32", previews)

            self.assertEqual(summary["normalization"], "finite p2-p98")
            self.assertEqual(len(summary["files"]), 2)
            self.assertFalse((root / "shadow_final_parallax.png").exists())
            for filename in summary["files"]:
                with Image.open(filename) as image:
                    self.assertEqual(image.size, (16, 12))

            with self.assertRaisesRegex(ValueError, "outside the atomic dump package"):
                dump_contract.generate_float_artifact_previews(
                    root, "shadow_final_parallax.f32", root / "previews")

    def test_model_input_preview_authenticates_shape_sidecar_before_normalization(self):
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dump"
            previews = Path(temporary) / "model-previews"
            rejected_previews = Path(temporary) / "rejected-model-previews"
            root.mkdir()
            manifest, _ = self._write_synthetic_geometry_dump(root)
            width = manifest["dimensions"]["model_input"]["width"]
            height = manifest["dimensions"]["model_input"]["height"]
            model_input = np.zeros((3, height, width), dtype="<f4").tobytes()
            (root / "model_input.f32").write_bytes(model_input)
            manifest["artifacts"]["model_input.f32"] = {
                "available": True,
                "required": True,
                "stage": "exact neural-network input",
                "description": "authenticated test model input",
                "sha256": hashlib.sha256(model_input).hexdigest(),
            }
            shape = {
                "schema": 1,
                "width": width,
                "height": height,
                "dtype": "float32-le",
                "layout": "NCHW",
                "channels": ["R", "G", "B"],
                "stage": "after calibrated preprocessing and ImageNet normalization",
                "imagenet_mean": [0.485, 0.456, 0.406],
                "imagenet_std": [0.229, 0.224, 0.225],
            }
            shape_payload = json.dumps(shape).encode("utf-8")
            (root / "model_input_shape.json").write_bytes(shape_payload)
            manifest["artifacts"]["model_input_shape.json"] = {
                "available": True,
                "required": True,
                "stage": "model-input contract",
                "description": "authenticated test preprocess contract",
                "sha256": hashlib.sha256(shape_payload).hexdigest(),
            }
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")

            summary = dump_contract.generate_float_artifact_previews(
                root, "model_input.f32", previews)
            self.assertEqual(summary["normalization"],
                             "channel * imagenet_std + imagenet_mean; clamped to [0,1]")
            with Image.open(summary["files"][0]) as image:
                self.assertEqual(image.size, (width, height))

            shape["imagenet_mean"][0] = 0.0
            (root / "model_input_shape.json").write_text(
                json.dumps(shape), encoding="utf-8")
            with self.assertRaisesRegex(
                    ValueError, "model_input_shape.json content hash mismatch"):
                dump_contract.generate_float_artifact_previews(
                    root, "model_input.f32", rejected_previews)
            self.assertFalse(rejected_previews.exists())

    def test_preview_generation_requires_complete_authenticated_geometry_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dump"
            previews = Path(temporary) / "previews"
            root.mkdir()
            self._write_synthetic_geometry_dump(root)
            candidate = root / "shadow_candidate_parallax.f32"
            payload = bytearray(candidate.read_bytes())
            payload[0] ^= 1
            candidate.write_bytes(payload)

            with self.assertRaisesRegex(
                    ValueError, "shadow_candidate_parallax.f32 content hash mismatch"):
                dump_contract.generate_float_artifact_previews(
                    root, "shadow_final_parallax.f32", previews)
            self.assertFalse(previews.exists())

    def test_geometry_verifier_requires_exact_shadow_coordinate_file(self):
        for mutation, expected in (
                ("missing", "shadow_coordinate.f32 is missing"),
                ("corrupt", "shadow_coordinate.f32 content hash mismatch")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self._write_synthetic_geometry_dump(root)
                coordinate_path = root / "shadow_coordinate.f32"
                if mutation == "missing":
                    coordinate_path.unlink()
                else:
                    payload = bytearray(coordinate_path.read_bytes())
                    payload[-1] ^= 1
                    coordinate_path.write_bytes(payload)
                with self.assertRaisesRegex(ValueError, expected):
                    dump_contract.verify_v2_dump_geometry(root)

    def test_geometry_verifier_authenticates_v2_state_artifact_bytes(self):
        for artifact in ("shadow_state.json", "shadow_frame_stats.json"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self._write_synthetic_geometry_dump(root)
                path = root / artifact
                path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, f"{artifact} content hash mismatch"):
                    dump_contract.verify_v2_dump_geometry(root)

    def test_geometry_verifier_validates_v2_state_artifact_payloads(self):
        mutations = {
            "shadow_state.json": (
                lambda value: value.update({"future": 1}),
                "missing or unknown root fields"),
            "shadow_frame_stats.json": (
                lambda value: value["named_values"].pop("reserved"),
                "exact stats layout"),
        }
        for artifact, (mutate, expected) in mutations.items():
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, _ = self._write_synthetic_geometry_dump(root)
                path = root / artifact
                document = json.loads(path.read_text(encoding="utf-8"))
                mutate(document)
                payload = json.dumps(document).encode("utf-8")
                path.write_bytes(payload)
                manifest["artifacts"][artifact]["sha256"] = hashlib.sha256(
                    payload).hexdigest()
                (root / "dump_manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    dump_contract.verify_v2_dump_geometry(root)

    def test_geometry_verifier_cross_checks_state_summary_and_exact_frame_stats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self._write_synthetic_geometry_dump(root)
            manifest["parallax_v2_shadow"]["state"]["confirmed_cut_count"] += 1
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "summary disagrees"):
                dump_contract.verify_v2_dump_geometry(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self._write_synthetic_geometry_dump(root)
            path = root / "shadow_frame_stats.json"
            stats = json.loads(path.read_text(encoding="utf-8"))
            stats["named_values"]["population_std"] = 0.0
            payload = json.dumps(stats).encode("utf-8")
            path.write_bytes(payload)
            manifest["artifacts"]["shadow_frame_stats.json"]["sha256"] = hashlib.sha256(
                payload).hexdigest()
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact-frame statistics"):
                dump_contract.verify_v2_dump_geometry(root)

    def test_geometry_verifier_accepts_current_uip1_empty_authority_as_exact_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, fields = self._write_synthetic_geometry_dump(
                root,
                width=770,
                height=434)
            self._activate_synthetic_uip1_dump(root, manifest, fields)

            summary = dump_contract.verify_v2_dump_geometry(root)

            subtitle = summary["subtitle_conditioning"]
            self.assertEqual(subtitle["mode"], "subtitle-contours-v1")
            self.assertTrue(subtitle["subtitle_evidence_verified"])
            self.assertTrue(subtitle["ocr_authoritative"])
            self.assertEqual(subtitle["ocr_ordinary_cover_horizontal_guard"]["cells"], 1)
            self.assertIn(
                "vertical-and-ribbon-unchanged",
                subtitle["ocr_ordinary_cover_horizontal_guard"]["policy"])
            self.assertEqual(subtitle["current_count"], 0)
            self.assertIn(
                "shadow_base_final_parallax",
                summary["chain_fields_verified"])

    def test_dump46_geometry_verifier_authenticates_every_contour_artifact_hash_and_byte_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, fields = self._write_synthetic_geometry_dump(root, width=770, height=434)
            manifest = self._activate_synthetic_uip1_dump(root, manifest, fields)
            for role, name in dump_contract._SUBTITLE_CONTOUR_ARTIFACT_FILES.items():
                original = (root / name).read_bytes()
                changed = bytearray(original)
                changed[0] ^= 1
                (root / name).write_bytes(changed)
                with self.subTest(role=role, fault="hash"), self.assertRaisesRegex(ValueError, "content hash mismatch"):
                    dump_contract.verify_v2_dump_geometry(root)
                self._write_hashed_payload(root, manifest, name, original[:-1])
                (root / "dump_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.subTest(role=role, fault="byte_count"), self.assertRaisesRegex(ValueError, "byte_count"):
                    dump_contract.verify_v2_dump_geometry(root)
                self._write_hashed_payload(root, manifest, name, original)
                (root / "dump_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(dump_contract.verify_v2_dump_geometry(root)["subtitle_conditioning"]["contour"]["tuple_verified"])

    def test_full_current_uip1_artifacts_preserve_independent_controller_clock(self):
        for accepted_clock, clock_flags in ((999900, 8), (0, 0)):
            with self.subTest(accepted_clock=accepted_clock), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, fields = self._write_synthetic_geometry_dump(root, width=770, height=434)
                manifest, _, plane, _ = self._activate_synthetic_uip1_current_dump(root, manifest, fields)
                plane[25], plane[28], plane[29] = clock_flags, accepted_clock, 0
                self._write_hashed_payload(root, manifest, "subtitle_locator_state.u32",
                                           self._pack_uint32_words(plane))
                (root / "dump_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                result = dump_contract.verify_v2_dump_geometry(root)["subtitle_conditioning"]
                self.assertTrue(result["geometry_ready"])
                self.assertEqual(result["active_level"], 2)
                self.assertEqual(result["geometry_publication_timestamp_us"], 1000001)
                self.assertEqual(result["adaptive"]["accepted_observation_timestamp_us"], accepted_clock)
                self.assertEqual(result["adaptive"]["covered_count"], 1)

    def test_full_held_uip1_artifacts_accept_equal_source_clock_and_independently_retained_ocr(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, fields = self._write_synthetic_geometry_dump(root, width=770, height=434)
            manifest, _, plane, ocr = self._activate_synthetic_uip1_current_dump(root, manifest, fields)
            plane[4], ocr[5] = 40, 40
            self._write_hashed_payload(root, manifest, "subtitle_locator_state.u32", self._pack_uint32_words(plane))
            self._write_hashed_payload(root, manifest, "subtitle_ocr_record.u32", self._pack_uint32_words(ocr))
            identity_name = dump_contract._SUBTITLE_CONTOUR_ARTIFACT_FILES["contour_identity"]
            identity = list(struct.unpack("<32I", (root / identity_name).read_bytes()))
            identity[4] = 40
            self._write_hashed_payload(root, manifest, identity_name, self._pack_uint32_words(identity))
            input_region = json.loads((root / "depth_input_region.json").read_text(encoding="utf-8"))
            raw, _ = self._attach_gpu_trace(root, manifest, input_region, 770, 434)
            words = list(struct.unpack(f"<{dump_contract.GPU_TRACE_RING_WORD_COUNT}I", raw))
            base, _ = self._set_gpu_trace_reuse(
                words, work=dump_contract.GPU_TRACE_WORK_SUBTITLE_OBSERVATION, host_outcome=1)
            words[base + 88:base + 128] = plane
            words[base + 128:base + 134] = [4, coordinate.SUBTITLE_OCR.condition_param_tag,
                                           1, 0, 2, plane[27]]
            raw = self._pack_uint32_words(words)
            parsed = dump_contract.validate_depth_input_region_document(
                input_region, matched_frame_id=41, source_width=1920, source_height=1080,
                tensor_width=770, tensor_height=434)
            trace = dump_contract.validate_gpu_trace_ring(
                raw, matched_frame_id=41, analysis_generation=0, source_width=1920,
                source_height=1080, field_width=770, field_height=434,
                expected_domain_tag=dump_contract._gpu_trace_domain_tag(parsed, "srgb", 770, 434),
                input_domain_reset=False)
            self._write_hashed_payload(root, manifest, "gpu_trace_ring.u32", raw)
            self._write_hashed_payload(root, manifest, "gpu_trace.json",
                                       (json.dumps(trace["decoded"], indent=2) + "\n").encode("utf-8"))
            (root / "dump_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            result = dump_contract.verify_v2_dump_geometry(root)["subtitle_conditioning"]
            self.assertTrue(result["held_with_depth"])
            self.assertEqual(result["evidence_frame_id"], 40)
            self.assertEqual(result["geometry_publication_timestamp_us"], 1000001)
            self.assertEqual(result["current_count"], 1)
            self.assertTrue(result["contour"]["tuple_verified"])


    def test_schema46_requires_an_independent_positive_publication_clock(self):
        for timestamp in (None, 0, -1, True, 1.0, 1 << 64):
            changed = copy.deepcopy(self.manifest)
            if timestamp is None:
                changed.pop("matched_observation_timestamp_us")
            else:
                changed["matched_observation_timestamp_us"] = timestamp
            with self.subTest(timestamp=timestamp), self.assertRaisesRegex(
                    ValueError, "observation timestamp|publication timestamp"):
                dump_contract.validate_v2_dump_manifest_document(changed)


    def test_geometry_verifier_rejects_uip1_record_identity_and_nonbase_empty_output(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, fields = self._write_synthetic_geometry_dump(
                root,
                width=770,
                height=434)
            manifest = self._activate_synthetic_uip1_dump(root, manifest, fields)

            record = list(struct.unpack(
                f"<{dump_contract.SUBTITLE_OCR_RECORD_WORD_COUNT}I",
                (root / "subtitle_ocr_record.u32").read_bytes()))
            record[5] = 42
            self._write_hashed_payload(
                root, manifest, "subtitle_ocr_record.u32",
                self._pack_uint32_words(record))
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity disagrees"):
                dump_contract.verify_v2_dump_geometry(root)

            record[5] = 41
            self._write_hashed_payload(
                root, manifest, "subtitle_ocr_record.u32",
                self._pack_uint32_words(record))
            changed = fields["shadow_final_parallax"].copy()
            changed[0, 0] = np.nextafter(
                changed[0, 0], np.float32(np.inf), dtype=np.float32)
            self._write_hashed_payload(
                root, manifest, "shadow_final_parallax.f32",
                changed.astype("<f4").tobytes())
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inactive Base copy"):
                dump_contract.verify_v2_dump_geometry(root)

    def test_geometry_verifier_replays_gaussian_chain_and_rejects_controller_and_wrong_ocr(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, fields = self._write_synthetic_geometry_dump(
                root,
                width=770,
                height=434)
            manifest, fade_one, locator, ocr = (
                self._activate_synthetic_uip1_current_dump(
                    root, manifest, fields))

            summary = dump_contract.verify_v2_dump_geometry(root)
            self.assertEqual(summary["subtitle_conditioning"]["current_count"], 1)
            base = fields["shadow_final_parallax"]
            self.assertTrue(np.array_equal(fade_one[0, 0], base[0, 0]))
            self.assertTrue(np.any(
                fade_one[350:401, 120:650] != base[350:401, 120:650]))

            locator[24] = 0
            self._write_hashed_payload(
                root, manifest, "subtitle_locator_state.u32",
                self._pack_uint32_words(locator))
            (root / "dump_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inactive adaptive UI tail"):
                dump_contract.verify_v2_dump_geometry(root)
            locator[24] = 1
            self._write_hashed_payload(
                root, manifest, "subtitle_locator_state.u32",
                self._pack_uint32_words(locator))
            # Restore authenticated full strength; actual Gaussian M/proposal/C75 are saved
            # independently. Final is the shared post-conditioned field, not a repeated pin.
            fade_two = fade_one.copy()
            fade_two_payload = fade_two.astype("<f4").tobytes()
            (root / "dump_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            summary = dump_contract.verify_v2_dump_geometry(root)
            self.assertEqual(summary["subtitle_conditioning"]["active_level"], 2)
            self.assertTrue(summary["subtitle_conditioning"]["gaussian"]["gaussian_final_verified"])

            tampered = fade_two.copy()
            tampered[360, 200] = np.nextafter(
                tampered[360, 200], np.float32(np.inf), dtype=np.float32)
            tampered_payload = tampered.astype("<f4").tobytes()
            self._write_hashed_payload(
                root, manifest, "shadow_final_parallax.f32", tampered_payload)
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Gaussian Final"):
                dump_contract.verify_v2_dump_geometry(root)

            self._write_hashed_payload(
                root, manifest, "shadow_final_parallax.f32", fade_two_payload)
            final = dump_contract.SUBTITLE_OCR_FINAL_BOX_WORD_OFFSET
            ocr[final] += 1
            self._write_hashed_payload(
                root, manifest, "subtitle_ocr_record.u32",
                self._pack_uint32_words(ocr))
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not contain|not exact"):
                dump_contract.verify_v2_dump_geometry(root)


    def test_roi_geometry_verifier_authenticates_region_border_map_and_zero_plane(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_synthetic_roi_geometry_dump(root)
            summary = dump_contract.verify_v2_dump_geometry(root)

            self.assertEqual(summary["depth_input_region"]["mode"], "window-region")
            self.assertTrue(summary["window_region_verified"])
            self.assertEqual(
                summary["depth_input_region"]["authorization"]["observer_generation"],
                901)
            self.assertNotEqual(
                summary["depth_input_region"]["analysis_generation"], 901)
            evidence = summary["roi_exterior_zero_evidence"]
            self.assertTrue(evidence["applicable"])
            self.assertTrue(evidence["has_exterior_zero_plane"])
            self.assertGreater(evidence["beyond_collar_sample_count"], 0)
            self.assertEqual(evidence["max_abs_identity_error_output_eye_px"], 0.0)
        self.assertEqual(
            summary["depth_input_region"]["inference_rect"],
            (330, 53, 630, 487))
        self.assertEqual(
            summary["depth_input_region"]["tensor_content_rect"],
            (235, 0, 535, 434))

    def test_roi_geometry_verifier_binds_frame_stats_to_nonpadding_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, region, _ = self._write_synthetic_roi_geometry_dump(root)
            content = region["analysis"]["tensor_content_rect_px"]
            expected_count = float(
                (content["right"] - content["left"]) *
                (content["bottom"] - content["top"]))
            stats_path = root / "shadow_frame_stats.json"
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual(stats["named_values"]["texel_count"], expected_count)

            # A full-tensor count incorrectly includes synthetic letterbox padding. Even with a
            # matching descriptor hash, it is not the exact analysis population for this frame.
            full_tensor_count = float(770 * 434)
            stats["named_values"]["valid_count"] = full_tensor_count
            stats["named_values"]["texel_count"] = full_tensor_count
            payload = json.dumps(stats).encode("utf-8")
            stats_path.write_bytes(payload)
            manifest["artifacts"]["shadow_frame_stats.json"]["sha256"] = hashlib.sha256(
                payload).hexdigest()
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact analysis content"):
                dump_contract.verify_v2_dump_geometry(root)

    def test_roi_geometry_verifier_uses_content_width_for_full_tensor_recurrences(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, region, _ = self._write_synthetic_roi_geometry_dump(root)
            content = region["analysis"]["tensor_content_rect_px"]
            left, right = content["left"], content["right"]
            width, height = 770, 434

            candidate = np.zeros((height, width), dtype=np.float32)
            split = left + (right - left) // 2
            candidate[:, split:] = np.float32(0.001)
            majorant, conditioned, final = dump_contract._replay_v2_limiter_fields(
                candidate, right - left)

            fields = {
                "shadow_candidate_parallax.f32": candidate,
                "shadow_vertical_majorant.f32": majorant,
                "shadow_vertical_conditioned.f32": conditioned,
                "shadow_final_parallax.f32": final,
            }
            for name, values in fields.items():
                self._write_hashed_payload(
                    root, manifest, name, values.astype("<f4").tobytes())
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            summary = dump_contract.verify_v2_dump_geometry(root)
            self.assertEqual(
                summary["depth_input_region"]["tensor_content_rect"],
                (235, 0, 535, 434))

            # The old whole-tensor denominator would propagate the 0.001 cliff leftward. It is
            # a different exact recurrence and must now fail closed.
            _, _, legacy = dump_contract._replay_v2_limiter_fields(candidate, width)
            self.assertFalse(np.array_equal(final, legacy))
            self._write_hashed_payload(
                root, manifest, "shadow_final_parallax.f32",
                legacy.astype("<f4").tobytes())
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact recurrence"):
                dump_contract.verify_v2_dump_geometry(root)

    def test_roi_geometry_verifier_accepts_foreground_window_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _, _ = self._write_synthetic_roi_geometry_dump(root)

            depth_input = json.loads((root / "depth_input_region.json").read_text())
            depth_input["authorization"].update({
                "authority_kind": "foreground-client",
                "document_id": 0,
                "video_id": 0,
            })
            depth_payload = json.dumps(depth_input).encode("utf-8")
            (root / "depth_input_region.json").write_bytes(depth_payload)
            manifest["artifacts"]["depth_input_region.json"]["sha256"] = (
                hashlib.sha256(depth_payload).hexdigest())

            provenance = json.loads((root / "window_region.json").read_text())
            provenance["authority_kind"] = "foreground-client"
            provenance["identity"].update({"document_id": 0, "video_id": 0})
            provenance_payload = json.dumps(provenance).encode("utf-8")
            (root / "window_region.json").write_bytes(provenance_payload)
            manifest["artifacts"]["window_region.json"]["sha256"] = (
                hashlib.sha256(provenance_payload).hexdigest())
            (root / "dump_manifest.json").write_text(json.dumps(manifest))

            summary = dump_contract.verify_v2_dump_geometry(root)
            self.assertEqual(
                summary["window_region"]["authority_kind"], "foreground-client")
            self.assertEqual(
                summary["depth_input_region"]["authorization"]["document_id"], 0)

    def test_roi_geometry_verifier_rejects_hashed_authority_and_zero_map_tampering(self):
        import hashlib

        import numpy as np

        mutations = (
            "region-hash", "border-hash", "map-hash", "mask-hash", "map-nonzero",
            "authorization-mismatch", "semantic-mismatch")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, _, warp_map = self._write_synthetic_roi_geometry_dump(root)
                if mutation == "region-hash":
                    document = json.loads((root / "depth_input_region.json").read_text())
                    document["analysis"]["analysis_generation"] += 1
                    (root / "depth_input_region.json").write_text(json.dumps(document))
                    expected = "depth_input_region.json content hash mismatch"
                elif mutation == "mask-hash":
                    (root / "warp_mask.png").write_bytes(b"tampered mask")
                    expected = "warp_mask.png content hash mismatch"
                elif mutation in {
                        "border-hash", "authorization-mismatch", "semantic-mismatch"}:
                    document = json.loads((root / "window_region.json").read_text())
                    if mutation == "semantic-mismatch":
                        document["coordinate_space"]["capture_rect_px"]["right"] -= 1
                    else:
                        document["identity"]["generation"] += 1
                    payload = json.dumps(document).encode("utf-8")
                    (root / "window_region.json").write_bytes(payload)
                    expected = ("window_region.json content hash mismatch"
                                if mutation == "border-hash" else
                                "authorization disagrees" if mutation == "authorization-mismatch"
                                else "semantic rectangle disagrees")
                    if mutation != "border-hash":
                        manifest["artifacts"]["window_region.json"]["sha256"] = (
                            hashlib.sha256(payload).hexdigest())
                        (root / "dump_manifest.json").write_text(json.dumps(manifest))
                else:
                    changed = warp_map.copy()
                    changed[0, 0] = np.float32(changed[0, 0] + 0.01)
                    payload = changed.astype("<f4").tobytes()
                    (root / "warp_map.f32").write_bytes(payload)
                    expected = ("warp_map.f32 content hash mismatch" if mutation == "map-hash"
                                else "nonzero beyond conservative collar support")
                    if mutation == "map-nonzero":
                        manifest["artifacts"]["warp_map.f32"]["sha256"] = hashlib.sha256(
                            payload).hexdigest()
                        (root / "dump_manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, expected):
                    dump_contract.verify_v2_dump_geometry(root)

    def test_roi_geometry_verifier_reports_when_collar_covers_all_exterior_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_synthetic_roi_geometry_dump(root, near_full=True)
            summary = dump_contract.verify_v2_dump_geometry(root)
            evidence = summary["roi_exterior_zero_evidence"]
            self.assertFalse(evidence["applicable"])
            self.assertFalse(evidence["has_exterior_zero_plane"])
            self.assertEqual(evidence["beyond_collar_sample_count"], 0)
            self.assertGreater(evidence["exterior_content_sample_count"], 0)
            self.assertGreater(evidence["max_horizontal_collar_source_px"], 0.0)

    def test_roi_manifest_dimensions_and_replay_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _, _ = self._write_synthetic_roi_geometry_dump(root)
            decoded = dump_contract.validate_v2_dump_manifest_document(manifest)
            self.assertEqual(decoded["position_authority"], [
                "shadow_final_parallax", "depth_input_region"])
            self.assertEqual(
                manifest["dimensions"]["source"]["format"],
                "DXGI_FORMAT_R16G16B16A16_FLOAT")
            with self.assertRaisesRegex(ValueError, "ROI-active Dump 3D replay"):
                _require_supported_replay_domain({"status": "validated", **decoded})

            changed = copy.deepcopy(manifest)
            changed["dimensions"]["analysis_source"]["width"] += 1
            (root / "dump_manifest.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "analysis-source dimensions"):
                dump_contract.verify_v2_dump_geometry(root)

            changed = copy.deepcopy(manifest)
            changed["dimensions"]["analysis_source"].update({
                "format": "DXGI_FORMAT_B8G8R8A8_UNORM",
                "format_value": 87,
            })
            with self.assertRaisesRegex(ValueError, "analysis-source dimensions"):
                dump_contract.validate_v2_dump_manifest_document(changed)

            changed = copy.deepcopy(manifest)
            changed["dimensions"]["model_input"]["layout"] = "NHWC"
            with self.assertRaisesRegex(ValueError, "crop-local tensor dimensions"):
                dump_contract.validate_v2_dump_manifest_document(changed)

    def test_geometry_verifier_cross_validates_advertised_window_region(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self._write_synthetic_geometry_dump(root)
            self._advertise_window_region(root, manifest)
            summary = dump_contract.verify_v2_dump_geometry(root)
            self.assertTrue(summary["window_region_verified"])
            self.assertEqual(summary["window_region"]["matched_frame_id"], 41)
            self.assertEqual(summary["window_region"]["right"], 1760)

    def test_full_source_window_statuses_remain_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self._write_synthetic_geometry_dump(root)
            self._advertise_window_region(root, manifest)
            manifest["window_region"].update({
                "observer_status": "desktop-or-unsupported-client",
                "mapping_status": "outside-capture-monitor",
            })
            (root / "dump_manifest.json").write_text(json.dumps(manifest))
            summary = dump_contract.verify_v2_dump_geometry(root)
            self.assertEqual(summary["depth_input_region"]["mode"], "full-source")
            self.assertTrue(summary["window_region_verified"])

    def test_ok_fullscreen_provenance_is_chromium_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self._write_synthetic_geometry_dump(root)
            provenance = self._advertise_window_region(root, manifest)
            manifest["window_region"]["observer_status"] = "ok-fullscreen"
            provenance["coordinate_space"]["capture_rect_px"] = {
                "left": 0, "top": 0, "right": 1920, "bottom": 1080,
            }
            (root / "window_region.json").write_text(json.dumps(provenance))
            (root / "dump_manifest.json").write_text(json.dumps(manifest))
            summary = dump_contract.verify_v2_dump_geometry(root)
            self.assertEqual(summary["window_region"]["authority_kind"], "chromium-video")

            provenance["authority_kind"] = "foreground-client"
            provenance["identity"].update({"document_id": 0, "video_id": 0})
            (root / "window_region.json").write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, "no matched Chromium provenance"):
                dump_contract.verify_v2_dump_geometry(root)

    def test_geometry_verifier_rejects_missing_malformed_and_mismatched_border(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self._write_synthetic_geometry_dump(root)
            border = self._advertise_window_region(root, manifest)

            (root / "window_region.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing or malformed"):
                dump_contract.verify_v2_dump_geometry(root)

            (root / "window_region.json").write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or malformed"):
                dump_contract.verify_v2_dump_geometry(root)

            changed = copy.deepcopy(border)
            changed["matched_frame_id"] = 42
            (root / "window_region.json").write_text(
                json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parent frame"):
                dump_contract.verify_v2_dump_geometry(root)

            changed = copy.deepcopy(border)
            changed["coordinate_space"]["source_extent_px"]["width"] = 1919
            (root / "window_region.json").write_text(
                json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source extent"):
                dump_contract.verify_v2_dump_geometry(root)

    def test_unavailable_window_region_has_no_package_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self._write_synthetic_geometry_dump(root)
            self._advertise_window_region(root, manifest, available=False)
            # An unadvertised leftover is intentionally ignored: only the atomic manifest grants
            # diagnostic-package authority to this optional artifact.
            (root / "window_region.json").write_text("{", encoding="utf-8")
            summary = dump_contract.verify_v2_dump_geometry(root)
            self.assertFalse(summary["window_region_verified"])
            self.assertIsNone(summary["window_region"])

    def test_manifest_rejects_inconsistent_window_region_advertisement(self):
        changed = copy.deepcopy(self.manifest)
        changed["window_region"] = {
            "available": True,
            "artifact": "window_region.json",
            "observer_status": "ok",
            "mapping_status": "ok",
            "geometry_authority": False,
            "renderer_authority": False,
        }
        changed["artifacts"]["window_region.json"] = {
            "available": False,
            "required": False,
            "stage": "matched-frame window region provenance",
            "description": "diagnostic test artifact",
        }
        with self.assertRaisesRegex(ValueError, "inconsistent window-region contract"):
            dump_contract.validate_v2_dump_manifest_document(changed)

    def test_geometry_verifier_rejects_content_and_chain_tampering(self):
        import hashlib

        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, fields = self._write_synthetic_geometry_dump(root)

            # 1) Content byte flip: hash mismatch.
            final_path = root / "shadow_final_parallax.f32"
            payload = bytearray(final_path.read_bytes())
            payload[3] ^= 0x40
            final_path.write_bytes(bytes(payload))
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                dump_contract.verify_v2_dump_geometry(root)

            # 2) Consistent hash but broken recurrence: chain mismatch.
            tampered = fields["shadow_final_parallax"].copy()
            tampered[5, 5] = np.float32(tampered[5, 5] + 0.001)
            tampered_bytes = tampered.astype("<f4").tobytes()
            final_path.write_bytes(tampered_bytes)
            manifest["artifacts"]["shadow_final_parallax.f32"]["sha256"] = (
                hashlib.sha256(tampered_bytes).hexdigest())
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact recurrence"):
                dump_contract.verify_v2_dump_geometry(root)

    def test_gpu_trace_contract_and_wrapped_raw_ring_are_exact(self):
        contract = self._gpu_trace_contract_document()
        summary = dump_contract.validate_gpu_trace_contract_document(contract)
        self.assertEqual(summary["capacity"], 300)
        self.assertEqual(dump_contract.GPU_TRACE_RING_SCHEMA, 5)
        self.assertEqual(dump_contract.GPU_TRACE_CONTRACT_SCHEMA, 6)
        self.assertEqual(dump_contract.GPU_TRACE_DECODED_SCHEMA, 7)
        self.assertEqual(summary["record_words"], 136)
        self.assertEqual(dump_contract.GPU_TRACE_RECORD_OFFSETS["subtitle_locator_begin"], 88)
        self.assertEqual(dump_contract.GPU_TRACE_RECORD_OFFSETS["subtitle_condition_begin"], 128)
        self.assertEqual(dump_contract.GPU_TRACE_RECORD_OFFSETS["observation_timestamp_low"], 134)
        self.assertEqual(dump_contract.GPU_TRACE_RECORD_OFFSETS["observation_timestamp_high"], 135)
        self.assertEqual(
            summary["source_closure_sha256"],
            dump_contract.GPU_TRACE_SOURCE_CLOSURE_SHA256)
        self.assertEqual(
            contract["enums"]["subtitle_disposition"]["invalid"], 6)
        input_region = copy.deepcopy(self.depth_input_region)
        input_region["analysis"]["tensor_extent_px"] = {"width": 16, "height": 12}
        input_region["analysis"]["tensor_content_rect_px"] = {
            "left": 0, "top": 0, "right": 16, "bottom": 12}
        raw, parsed, domain_tag = self._gpu_trace_ring_payload(
            input_region, 16, 12, slot=0, sequence=501,
            dump_forced_at_enqueue=False)
        words = list(struct.unpack(
            f"<{dump_contract.GPU_TRACE_RING_WORD_COUNT}I", raw))
        newest_base = dump_contract.GPU_TRACE_HEADER_WORD_COUNT
        oldest_base = (dump_contract.GPU_TRACE_HEADER_WORD_COUNT +
                       299 * dump_contract.GPU_TRACE_RECORD_WORD_COUNT)
        words[oldest_base:oldest_base + dump_contract.GPU_TRACE_RECORD_WORD_COUNT] = (
            words[newest_base:newest_base + dump_contract.GPU_TRACE_RECORD_WORD_COUNT])
        words[oldest_base + 2] = 500
        words[oldest_base + 3] = 0
        words[oldest_base + 4] = 40
        words[oldest_base + 5] = 0
        words[oldest_base + 14] = dump_contract.GPU_TRACE_WORK_NONE
        words[oldest_base + 15] = 0
        words[oldest_base + 16] = (
            dump_contract.GPU_TRACE_FLAG_INPUT_DOMAIN_RESET |
            dump_contract.GPU_TRACE_FLAG_SUBTITLE_SUPPRESSED)
        words[oldest_base + 17] = 0
        transaction = oldest_base + 24
        words[transaction + 13] = dump_contract.GPU_TRACE_WORK_NONE
        words[transaction + 14] = 0
        words[oldest_base + 88:oldest_base + 134] = [0xFFFFFFFF] * 46
        words[4:8] = [502, 0, 1, 2]
        raw = struct.pack(f"<{len(words)}I", *words)
        decoded = dump_contract.validate_gpu_trace_ring(
            raw,
            matched_frame_id=41,
            analysis_generation=0,
            source_width=1920,
            source_height=1080,
            field_width=16,
            field_height=12,
            expected_domain_tag=domain_tag,
            input_domain_reset=False,
        )
        self.assertEqual(decoded["record_count"], 2)
        self.assertEqual(decoded["oldest_sequence"], 500)
        self.assertEqual(decoded["next_sequence"], 502)
        self.assertEqual(decoded["matched_sequence"], 501)
        records = decoded["decoded"]["records"]
        self.assertEqual([record["ring_slot"] for record in records], [299, 0])
        self.assertTrue(records[0]["flags"]["subtitle_suppressed"])
        self.assertTrue(records[0]["flags"]["input_domain_reset"])
        self.assertTrue(records[0]["subtitle_condition"]["unused"])
        record = records[1]
        self.assertEqual(record["ring_slot"], 0)
        self.assertFalse(record["flags"]["dump_forced_at_enqueue"])
        self.assertEqual(record["depth_disposition"]["name"], "infer")
        self.assertEqual(record["subtitle_disposition"]["name"], "abstention")

    def test_gpu_trace_authenticated_reuse_preserves_explicit_suppression(self):
        input_region = copy.deepcopy(self.depth_input_region)
        input_region["analysis"]["tensor_extent_px"] = {"width": 16, "height": 12}
        input_region["analysis"]["tensor_content_rect_px"] = {
            "left": 0, "top": 0, "right": 16, "bottom": 12}
        raw, _, domain_tag = self._gpu_trace_ring_payload(input_region, 16, 12)
        words = list(struct.unpack(
            f"<{dump_contract.GPU_TRACE_RING_WORD_COUNT}I", raw))
        base, _ = self._set_gpu_trace_reuse(
            words, work=dump_contract.GPU_TRACE_WORK_NONE, host_outcome=0)
        words[base + dump_contract.GPU_TRACE_RECORD_OFFSETS["subtitle_disposition"]] = 0
        words[base + dump_contract.GPU_TRACE_RECORD_OFFSETS["flags"]] |= (
            dump_contract.GPU_TRACE_FLAG_SUBTITLE_SUPPRESSED)

        decoded = dump_contract.validate_gpu_trace_ring(
            struct.pack(f"<{len(words)}I", *words),
            matched_frame_id=41, analysis_generation=0,
            source_width=1920, source_height=1080,
            field_width=16, field_height=12,
            expected_domain_tag=domain_tag, input_domain_reset=False)
        record = decoded["decoded"]["records"][0]
        self.assertEqual(record["subtitle_disposition"]["name"], "suppressed")
        self.assertIn("subtitle-suppressed", record["subtitle_condition"]["reason"])

    def test_gpu_trace_authenticated_ordinary_reuse_holds_coherent_subtitle_tuple(self):
        input_region = copy.deepcopy(self.depth_input_region)
        input_region["analysis"]["tensor_extent_px"] = {"width": 16, "height": 12}
        input_region["analysis"]["tensor_content_rect_px"] = {
            "left": 0, "top": 0, "right": 16, "bottom": 12}
        raw, _, domain_tag = self._gpu_trace_ring_payload(input_region, 16, 12)
        words = list(struct.unpack(
            f"<{dump_contract.GPU_TRACE_RING_WORD_COUNT}I", raw))
        base, _ = self._set_gpu_trace_reuse(
            words,
            work=dump_contract.GPU_TRACE_WORK_SUBTITLE_OBSERVATION,
            host_outcome=1)

        decoded = dump_contract.validate_gpu_trace_ring(
            struct.pack(f"<{len(words)}I", *words),
            matched_frame_id=41, analysis_generation=0,
            source_width=1920, source_height=1080,
            field_width=16, field_height=12,
            expected_domain_tag=domain_tag, input_domain_reset=False)
        record = decoded["decoded"]["records"][0]
        self.assertEqual(
            record["subtitle_disposition"]["name"], "held-with-depth")
        self.assertTrue(record["flags"]["subtitle_branch_gated"])
        self.assertFalse(record["flags"]["condition_executed"])
        self.assertTrue(record["subtitle_condition"]["held_with_depth"])
        self.assertEqual(record["subtitle_plane"]["frame_id"], 40)

        locator = base + dump_contract.GPU_TRACE_RECORD_OFFSETS[
            "subtitle_locator_begin"]
        words[locator + 4] = 41
        with self.assertRaisesRegex(ValueError, "invalid finalized UIP4|UIP4"):
            dump_contract.validate_gpu_trace_ring(
                struct.pack(f"<{len(words)}I", *words),
                matched_frame_id=41, analysis_generation=0,
                source_width=1920, source_height=1080,
                field_width=16, field_height=12,
                expected_domain_tag=domain_tag, input_domain_reset=False)

    def test_gpu_trace_held_tuple_must_equal_immediately_prior_record(self):
        input_region = copy.deepcopy(self.depth_input_region)
        input_region["analysis"]["tensor_extent_px"] = {"width": 16, "height": 12}
        input_region["analysis"]["tensor_content_rect_px"] = {
            "left": 0, "top": 0, "right": 16, "bottom": 12}
        raw, _, domain_tag = self._gpu_trace_ring_payload(
            input_region, 16, 12, slot=0, sequence=2)
        words = list(struct.unpack(
            f"<{dump_contract.GPU_TRACE_RING_WORD_COUNT}I", raw))
        newest_base = dump_contract.GPU_TRACE_HEADER_WORD_COUNT
        oldest_base = (dump_contract.GPU_TRACE_HEADER_WORD_COUNT +
                       (dump_contract.GPU_TRACE_CAPACITY - 1) *
                       dump_contract.GPU_TRACE_RECORD_WORD_COUNT)
        words[oldest_base:oldest_base + dump_contract.GPU_TRACE_RECORD_WORD_COUNT] = (
            words[newest_base:newest_base + dump_contract.GPU_TRACE_RECORD_WORD_COUNT])
        words[oldest_base + 2] = 1
        words[oldest_base + 3] = 0
        words[oldest_base + 4] = 40
        words[oldest_base + 5] = 0
        oldest_locator = oldest_base + dump_contract.GPU_TRACE_RECORD_OFFSETS[
            "subtitle_locator_begin"]
        words[oldest_locator + 4] = 40
        words[oldest_locator + 8] = 1_000_000
        words[oldest_locator + 28] = 1_000_000
        words[oldest_base + 134] = 1_000_000
        words[oldest_base + 135] = 0
        words[4:8] = [3, 0, 1, 2]
        self._set_gpu_trace_reuse(
            words,
            work=dump_contract.GPU_TRACE_WORK_SUBTITLE_OBSERVATION,
            host_outcome=1)

        decoded = dump_contract.validate_gpu_trace_ring(
            struct.pack(f"<{len(words)}I", *words),
            matched_frame_id=41, analysis_generation=0,
            source_width=1920, source_height=1080,
            field_width=16, field_height=12,
            expected_domain_tag=domain_tag, input_domain_reset=False)
        self.assertEqual(
            decoded["decoded"]["records"][1]["subtitle_disposition"]["name"],
            "held-with-depth")

        changed = words.copy()
        newest_locator = newest_base + dump_contract.GPU_TRACE_RECORD_OFFSETS[
            "subtitle_locator_begin"]
        changed[newest_locator + 27] ^= 1
        with self.assertRaisesRegex(ValueError, "immediately prior record"):
            dump_contract.validate_gpu_trace_ring(
                struct.pack(f"<{len(changed)}I", *changed),
                matched_frame_id=41, analysis_generation=0,
                source_width=1920, source_height=1080,
                field_width=16, field_height=12,
                expected_domain_tag=domain_tag, input_domain_reset=False)

    def test_gpu_trace_held_tuple_permits_equal_clock_but_requires_older_frame_and_no_newer_clock(self):
        input_region = copy.deepcopy(self.depth_input_region)
        input_region["analysis"]["tensor_extent_px"] = {"width": 16, "height": 12}
        input_region["analysis"]["tensor_content_rect_px"] = {
            "left": 0, "top": 0, "right": 16, "bottom": 12}
        raw, _, domain_tag = self._gpu_trace_ring_payload(input_region, 16, 12)
        words = list(struct.unpack(f"<{dump_contract.GPU_TRACE_RING_WORD_COUNT}I", raw))
        base, _ = self._set_gpu_trace_reuse(
            words, work=dump_contract.GPU_TRACE_WORK_SUBTITLE_OBSERVATION, host_outcome=1)
        plane = base + dump_contract.GPU_TRACE_RECORD_OFFSETS["subtitle_locator_begin"]
        words[plane + 8] = self.manifest["matched_observation_timestamp_us"]
        decoded = dump_contract.validate_gpu_trace_ring(
            self._pack_uint32_words(words), matched_frame_id=41, analysis_generation=0,
            source_width=1920, source_height=1080, field_width=16, field_height=12,
            expected_domain_tag=domain_tag, input_domain_reset=False)
        row = decoded["decoded"]["records"][0]
        self.assertEqual(row["subtitle_plane"]["frame_id"], 40)
        self.assertEqual(row["subtitle_plane"]["observation_timestamp_us"],
                         row["observation_timestamp_us"])
        self.assertFalse(row["subtitle_condition"]["executed"])
        words[plane + 8] += 1
        with self.assertRaisesRegex(ValueError, "newer than the transaction"):
            dump_contract.validate_gpu_trace_ring(
                self._pack_uint32_words(words), matched_frame_id=41, analysis_generation=0,
                source_width=1920, source_height=1080, field_width=16, field_height=12,
                expected_domain_tag=domain_tag, input_domain_reset=False)

    def test_gpu_trace_abstention_cannot_grant_current_uip1_slots(self):
        input_region = copy.deepcopy(self.depth_input_region)
        input_region["analysis"]["tensor_extent_px"] = {"width": 16, "height": 12}
        input_region["analysis"]["tensor_content_rect_px"] = {
            "left": 0, "top": 0, "right": 16, "bottom": 12}
        raw, _, domain_tag = self._gpu_trace_ring_payload(input_region, 16, 12)
        words = list(struct.unpack(f"<{dump_contract.GPU_TRACE_RING_WORD_COUNT}I", raw))
        plane = dump_contract.GPU_TRACE_HEADER_WORD_COUNT + 88
        for mask in (0, 1):
            changed = words.copy()
            changed[plane + 2], changed[plane + 3] = 1, mask
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "abstention cannot grant"):
                dump_contract.validate_gpu_trace_ring(
                    self._pack_uint32_words(changed), matched_frame_id=41, analysis_generation=0,
                    source_width=1920, source_height=1080, field_width=16, field_height=12,
                    expected_domain_tag=domain_tag, input_domain_reset=False)

    def test_gpu_trace_optional_current_tuple_authenticates_layout_without_inventing_covers(self):
        input_region = copy.deepcopy(self.depth_input_region)
        input_region["analysis"]["tensor_extent_px"] = {"width": 16, "height": 12}
        input_region["analysis"]["tensor_content_rect_px"] = {
            "left": 0, "top": 0, "right": 16, "bottom": 12}
        raw, _, domain_tag = self._gpu_trace_ring_payload(input_region, 16, 12)
        words = list(struct.unpack(f"<{dump_contract.GPU_TRACE_RING_WORD_COUNT}I", raw))
        base = dump_contract.GPU_TRACE_HEADER_WORD_COUNT
        transaction = base + 24
        work = dump_contract.GPU_TRACE_WORK_OPTIONAL_OCR
        words[base + 14], words[base + 15] = work, 1
        words[transaction + 7] = dump_contract.GPU_TRACE_OPTIONAL_RECEIPT_MAGIC
        words[transaction + 1] = (1 ^ dump_contract.GPU_TRACE_DECISION_COOKIE ^
                                  dump_contract.GPU_TRACE_OPTIONAL_RECEIPT_MAGIC)
        words[transaction + 13] = work
        words[transaction + 14] = work ^ dump_contract.GPU_TRACE_WORK_FLAGS_COOKIE
        words[base + 90], words[base + 91] = 1, 1
        words[base + 128:base + 134] = [4, coordinate.SUBTITLE_OCR.condition_param_tag,
                                       1, 1, 2, words[base + 88 + 27]]
        decoded = dump_contract.validate_gpu_trace_ring(
            self._pack_uint32_words(words), matched_frame_id=41, analysis_generation=0,
            source_width=1920, source_height=1080, field_width=16, field_height=12,
            expected_domain_tag=domain_tag, input_domain_reset=False)
        row = decoded["decoded"]["records"][0]
        self.assertEqual(row["subtitle_disposition"]["name"], "optional-ocr")
        self.assertTrue(row["subtitle_condition"]["active"])
        self.assertEqual(row["subtitle_plane"]["current_count"], 1)
        self.assertNotIn("current_rectangles", row["subtitle_plane"])
        self.assertNotIn("subtitle_locator", row)
        self.assertNotIn("subtitle_locator_words", row)

    def test_gpu_trace_rejects_retired_due_work_and_ocr_on_reuse(self):
        input_region = copy.deepcopy(self.depth_input_region)
        input_region["analysis"]["tensor_extent_px"] = {"width": 16, "height": 12}
        input_region["analysis"]["tensor_content_rect_px"] = {
            "left": 0, "top": 0, "right": 16, "bottom": 12}
        for work in (8, 16):
            with self.subTest(work=work):
                raw, _, domain_tag = self._gpu_trace_ring_payload(input_region, 16, 12)
                words = list(struct.unpack(
                    f"<{dump_contract.GPU_TRACE_RING_WORD_COUNT}I", raw))
                self._set_gpu_trace_reuse(words, work=work, host_outcome=1)
                with self.assertRaisesRegex(ValueError, "depth disposition disagrees"):
                    dump_contract.validate_gpu_trace_ring(
                        struct.pack(f"<{len(words)}I", *words), matched_frame_id=41,
                        analysis_generation=0, source_width=1920, source_height=1080,
                        field_width=16, field_height=12, expected_domain_tag=domain_tag,
                        input_domain_reset=False)

        raw, _, domain_tag = self._gpu_trace_ring_payload(input_region, 16, 12)
        words = list(struct.unpack(
            f"<{dump_contract.GPU_TRACE_RING_WORD_COUNT}I", raw))
        base, transaction = self._set_gpu_trace_reuse(
            words, work=dump_contract.GPU_TRACE_WORK_OPTIONAL_OCR, host_outcome=1)
        marker = dump_contract.GPU_TRACE_OPTIONAL_RECEIPT_MAGIC
        words[transaction + 1] = dump_contract.GPU_TRACE_DECISION_COOKIE ^ marker
        words[transaction + 7] = marker
        with self.assertRaisesRegex(ValueError, "depth disposition disagrees"):
            dump_contract.validate_gpu_trace_ring(
                struct.pack(f"<{len(words)}I", *words),
                matched_frame_id=41, analysis_generation=0,
                source_width=1920, source_height=1080,
                field_width=16, field_height=12,
                expected_domain_tag=domain_tag, input_domain_reset=False)

        words[transaction + 7] = 0
        words[transaction + 1] = dump_contract.GPU_TRACE_DECISION_COOKIE
        words[base + dump_contract.GPU_TRACE_RECORD_OFFSETS["expected_work"]] = 4
        words[transaction + 13] = 4
        words[transaction + 14] = 4 ^ dump_contract.GPU_TRACE_WORK_FLAGS_COOKIE
        with self.assertRaisesRegex(ValueError, "depth disposition disagrees"):
            dump_contract.validate_gpu_trace_ring(
                struct.pack(f"<{len(words)}I", *words),
                matched_frame_id=41, analysis_generation=0,
                source_width=1920, source_height=1080,
                field_width=16, field_height=12,
                expected_domain_tag=domain_tag, input_domain_reset=False)

    def test_gpu_trace_reuse_cannot_claim_direct_subtitle_publication(self):
        for work in (dump_contract.GPU_TRACE_WORK_OPTIONAL_OCR,
                     dump_contract.GPU_TRACE_WORK_SUBTITLE_OBSERVATION):
            with self.subTest(work=work):
                self.assertEqual(dump_contract._gpu_trace_subtitle_disposition(
                    work, 1, 1, True, False,
                    dump_contract.GPU_TRACE_FLAG_OCR_RECORD_SUBMITTED |
                    dump_contract.GPU_TRACE_FLAG_CONDITION_EXECUTED), 6)
                self.assertEqual(dump_contract._gpu_trace_subtitle_disposition(
                    work, 1, 1, True, False,
                    dump_contract.GPU_TRACE_FLAG_SUBTITLE_BRANCH_GATED), 5)

    def test_gpu_trace_requires_nonzero_nondecreasing_observation_timestamps(self):
        input_region = copy.deepcopy(self.depth_input_region)
        input_region["analysis"]["tensor_extent_px"] = {"width": 16, "height": 12}
        input_region["analysis"]["tensor_content_rect_px"] = {
            "left": 0, "top": 0, "right": 16, "bottom": 12}
        raw, _, domain_tag = self._gpu_trace_ring_payload(
            input_region, 16, 12, slot=0, sequence=2)
        words = list(struct.unpack(
            f"<{dump_contract.GPU_TRACE_RING_WORD_COUNT}I", raw))
        newest_base = dump_contract.GPU_TRACE_HEADER_WORD_COUNT
        words[newest_base + 134] = 0
        words[newest_base + 135] = 0
        with self.assertRaisesRegex(ValueError, "observation timestamp"):
            dump_contract.validate_gpu_trace_ring(
                struct.pack(f"<{len(words)}I", *words),
                matched_frame_id=41, analysis_generation=0,
                source_width=1920, source_height=1080,
                field_width=16, field_height=12,
                expected_domain_tag=domain_tag, input_domain_reset=False)

        words = list(struct.unpack(
            f"<{dump_contract.GPU_TRACE_RING_WORD_COUNT}I", raw))
        oldest_base = (dump_contract.GPU_TRACE_HEADER_WORD_COUNT +
                       (dump_contract.GPU_TRACE_CAPACITY - 1) *
                       dump_contract.GPU_TRACE_RECORD_WORD_COUNT)
        words[oldest_base:oldest_base + dump_contract.GPU_TRACE_RECORD_WORD_COUNT] = (
            words[newest_base:newest_base + dump_contract.GPU_TRACE_RECORD_WORD_COUNT])
        words[oldest_base + 2] = 1
        words[oldest_base + 3] = 0
        words[oldest_base + 4] = 40
        words[oldest_base + 5] = 0
        oldest_locator = oldest_base + dump_contract.GPU_TRACE_RECORD_OFFSETS[
            "subtitle_locator_begin"]
        words[oldest_locator + 4] = 40
        newer_timestamp = 1_000_041
        older_timestamp = newer_timestamp + 1
        words[oldest_locator + 8] = older_timestamp
        words[oldest_locator + 28] = older_timestamp
        newest_locator = newest_base + dump_contract.GPU_TRACE_RECORD_OFFSETS["subtitle_locator_begin"]
        words[newest_locator + 8] = newer_timestamp
        words[newest_locator + 28] = newer_timestamp
        words[oldest_base + 134] = older_timestamp
        words[oldest_base + 135] = 0
        words[newest_base + 134] = newer_timestamp
        words[newest_base + 135] = 0
        words[4:8] = [3, 0, 1, 2]
        with self.assertRaisesRegex(ValueError, "observation timestamp"):
            dump_contract.validate_gpu_trace_ring(
                struct.pack(f"<{len(words)}I", *words),
                matched_frame_id=41, analysis_generation=0,
                source_width=1920, source_height=1080,
                field_width=16, field_height=12,
                expected_domain_tag=domain_tag, input_domain_reset=False)

    def test_gpu_trace_receipt_auth_rejects_force_reuse(self):
        input_region = copy.deepcopy(self.depth_input_region)
        input_region["analysis"]["tensor_extent_px"] = {"width": 16, "height": 12}
        input_region["analysis"]["tensor_content_rect_px"] = {
            "left": 0, "top": 0, "right": 16, "bottom": 12}
        raw, _, domain_tag = self._gpu_trace_ring_payload(input_region, 16, 12)
        words = list(struct.unpack(
            f"<{dump_contract.GPU_TRACE_RING_WORD_COUNT}I", raw))
        base = dump_contract.GPU_TRACE_HEADER_WORD_COUNT
        transaction = base + dump_contract.GPU_TRACE_RECORD_OFFSETS["transaction_begin"]
        words[transaction] = 0
        words[transaction + 1] = dump_contract.GPU_TRACE_DECISION_COOKIE
        words[base + dump_contract.GPU_TRACE_RECORD_OFFSETS["depth_disposition"]] = 0
        forged = struct.pack(f"<{len(words)}I", *words)
        decoded = dump_contract.validate_gpu_trace_ring(
            forged,
            matched_frame_id=41,
            analysis_generation=0,
            source_width=1920,
            source_height=1080,
            field_width=16,
            field_height=12,
            expected_domain_tag=domain_tag,
            input_domain_reset=False,
        )
        self.assertEqual(
            decoded["decoded"]["records"][0]["depth_disposition"]["name"],
            "invalid")
        record = decoded["decoded"]["records"][0]
        self.assertEqual(record["subtitle_disposition"]["name"], "abstention")
        self.assertFalse(record["subtitle_condition"]["unused"])
        self.assertTrue(record["subtitle_condition"]["executed"])

        words[base + dump_contract.GPU_TRACE_RECORD_OFFSETS["flags"]] &= ~(
            dump_contract.GPU_TRACE_FLAG_CONDITION_EXECUTED)
        with self.assertRaisesRegex(ValueError, "subtitle disposition disagrees"):
            dump_contract.validate_gpu_trace_ring(
                struct.pack(f"<{len(words)}I", *words),
                matched_frame_id=41, analysis_generation=0,
                source_width=1920, source_height=1080,
                field_width=16, field_height=12,
                expected_domain_tag=domain_tag, input_domain_reset=False)
        words[base + dump_contract.GPU_TRACE_RECORD_OFFSETS["flags"]] |= (
            dump_contract.GPU_TRACE_FLAG_CONDITION_EXECUTED)
        words[base + dump_contract.GPU_TRACE_RECORD_OFFSETS["subtitle_disposition"]] = 6
        with self.assertRaisesRegex(ValueError, "subtitle disposition disagrees"):
            dump_contract.validate_gpu_trace_ring(
                struct.pack(f"<{len(words)}I", *words),
                matched_frame_id=41, analysis_generation=0,
                source_width=1920, source_height=1080,
                field_width=16, field_height=12,
                expected_domain_tag=domain_tag, input_domain_reset=False)


    def test_geometry_verifier_accepts_optional_gpu_trace_and_matches_raw_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self._write_synthetic_geometry_dump(root)
            input_region = json.loads((root / "depth_input_region.json").read_text())
            self._attach_gpu_trace(root, manifest, input_region, 16, 12)
            summary = dump_contract.verify_v2_dump_geometry(root)
            self.assertTrue(summary["gpu_trace"]["available"])
            self.assertEqual(summary["gpu_trace"]["record_count"], 1)
            self.assertEqual(summary["gpu_trace"]["matched_sequence"], 1)

    def test_gpu_trace_verifier_rejects_hash_auth_decode_contract_and_domain_tampering(self):
        mutations = ("raw-hash", "receipt", "decoded", "contract", "domain", "torn-tag")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, _ = self._write_synthetic_geometry_dump(root)
                input_region = json.loads((root / "depth_input_region.json").read_text())
                self._attach_gpu_trace(root, manifest, input_region, 16, 12)
                if mutation in {"raw-hash", "receipt", "domain", "torn-tag"}:
                    path = root / "gpu_trace_ring.u32"
                    words = list(struct.unpack(
                        f"<{dump_contract.GPU_TRACE_RING_WORD_COUNT}I", path.read_bytes()))
                    base = dump_contract.GPU_TRACE_HEADER_WORD_COUNT
                    if mutation == "raw-hash":
                        words[base + 24 + 1] ^= 1
                        expected = "content hash mismatch"
                    elif mutation == "receipt":
                        words[base + 24 + 1] ^= 1
                        expected = "depth disposition disagrees"
                    elif mutation == "domain":
                        words[base + 8] ^= 1
                        expected = "disagrees with the dump domain"
                    else:
                        words[base + 1] = 0
                        expected = "invalid committed record"
                    payload = struct.pack(f"<{len(words)}I", *words)
                    path.write_bytes(payload)
                    if mutation != "raw-hash":
                        manifest["artifacts"]["gpu_trace_ring.u32"]["sha256"] = (
                            hashlib.sha256(payload).hexdigest())
                        (root / "dump_manifest.json").write_text(json.dumps(manifest))
                elif mutation == "decoded":
                    path = root / "gpu_trace.json"
                    document = json.loads(path.read_text())
                    document["records"][0]["depth_disposition"]["name"] = "reuse"
                    payload = json.dumps(document).encode()
                    path.write_bytes(payload)
                    manifest["artifacts"]["gpu_trace.json"]["sha256"] = (
                        hashlib.sha256(payload).hexdigest())
                    (root / "dump_manifest.json").write_text(json.dumps(manifest))
                    expected = "disagrees with the authenticated raw ring"
                else:
                    path = root / "gpu_trace_contract.json"
                    document = json.loads(path.read_text())
                    document["record_word_offsets"]["subtitle_locator_begin"] += 1
                    payload = json.dumps(document).encode()
                    path.write_bytes(payload)
                    manifest["artifacts"]["gpu_trace_contract.json"]["sha256"] = (
                        hashlib.sha256(payload).hexdigest())
                    (root / "dump_manifest.json").write_text(json.dumps(manifest))
                    expected = "stale record_word_offsets"
                with self.assertRaisesRegex(ValueError, expected):
                    dump_contract.verify_v2_dump_geometry(root)

    def test_manifest_authenticates_live_and_diagnostic_renderer_sources(self):
        for key, replacement in (
                ("source_closure_sha256", "0" * 64),
                ("diagnostic_source_closure_sha256", "0" * 64),
                ("source_compile_flags", 0),
                ("entrypoint", "wrong")):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.manifest)
                changed["renderer"]["live_shader_source"][key] = replacement
                with self.assertRaisesRegex(ValueError, "conditioner attribution"):
                    dump_contract.validate_v2_dump_manifest_document(changed)

        changed = copy.deepcopy(self.manifest)
        changed["renderer"].pop("live_shader_source")
        with self.assertRaisesRegex(ValueError, "conditioner attribution"):
            dump_contract.validate_v2_dump_manifest_document(changed)

    def test_manifest_requires_complete_renderer_attribution(self):
        for key in (
                "parallax_v2_render_requested",
                "mapping_artifacts_match_selected_renderer",
                "parallax_v2_coordinate_role"):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.manifest)
                changed["renderer"].pop(key)
                with self.assertRaises(ValueError):
                    dump_contract.validate_v2_dump_manifest_document(changed)

        changed = copy.deepcopy(self.manifest)
        changed["renderer"]["parallax_v2_position_field"] = (
            "shadow_vertical_majorant")
        with self.assertRaisesRegex(ValueError, "conditioner attribution"):
            dump_contract.validate_v2_dump_manifest_document(changed)

        changed = copy.deepcopy(self.manifest)
        changed["dimensions"]["shadow_vertical_majorant"] = None
        with self.assertRaisesRegex(ValueError, "geometry dimension"):
            dump_contract.validate_v2_dump_manifest_document(changed)

    def test_current_manifest_rejects_inactive_shadow_compatibility(self):
        inactive = copy.deepcopy(self.manifest)
        inactive["parallax_v2_shadow"]["active"] = False
        with self.assertRaisesRegex(ValueError, "shadow attribution"):
            dump_contract.validate_v2_dump_manifest_document(inactive)

    def test_manifest_rejects_selection_or_geometry_dimension_disagreement(self):
        changed = copy.deepcopy(self.manifest)
        changed["parallax_v2_shadow"]["rendered_output_selected"] = False
        with self.assertRaisesRegex(ValueError, "shadow attribution"):
            dump_contract.validate_v2_dump_manifest_document(changed)

        changed = copy.deepcopy(self.manifest)
        changed["dimensions"]["shadow_final_parallax"]["width"] += 1
        with self.assertRaisesRegex(ValueError, "geometry dimensions disagree"):
            dump_contract.validate_v2_dump_manifest_document(changed)

    def test_manifest_rejects_source_extents_beyond_production_bounds(self):
        at_limit = copy.deepcopy(self.manifest)
        for name in ("source", "analysis_source"):
            at_limit["dimensions"][name].update({"width": 5120, "height": 2160})
        at_limit_shape = dump_contract.expected_capture_grid_for_source(
            5120, 2160, scale=1)
        self.assertIsNotNone(at_limit_shape)
        self._set_manifest_capture_grid(at_limit, *at_limit_shape)
        dump_contract.validate_v2_dump_manifest_document(at_limit)

        invalid_extents = (
            (5121, 1),
            (5120, 2161),
        )
        for name in ("source", "analysis_source"):
            for width, height in invalid_extents:
                with self.subTest(name=name, width=width, height=height):
                    changed = copy.deepcopy(self.manifest)
                    changed["dimensions"][name].update({
                        "width": width,
                        "height": height,
                    })
                    with self.assertRaisesRegex(ValueError, "supported Host SBS V2 bounds"):
                        dump_contract.validate_v2_dump_manifest_document(changed)

    def test_live_rendered_state_is_accepted_with_distinct_wire_semantics(self):
        live = copy.deepcopy(self.state)
        live["rendered_output_selected"] = True
        live["wire_contract"] = (
            "authenticated live Host-SBS renderer input; not a client wire contract")
        decoded = dump_contract.validate_shadow_state_document(live)
        self.assertEqual(decoded["contract_tag_bits"], self.state["decoded"]["contract_tag"])

    def test_shadow_state_accepts_exact_writer_unit_variants(self):
        variants = (
            {
                "coordinate": "dimensionless canonical coordinate derived from raw depth",
                "gain": "one-eye full-source-U per coordinate unit",
                "parallax": "signed one-eye full-source-U",
            },
            {
                "coordinate": "dimensionless canonical coordinate derived from raw depth",
                "gain": "one-eye ROI-local source-U per coordinate unit",
                "parallax": (
                    "signed one-eye ROI-local source-U; full-source renderer authority "
                    "additionally requires depth_input_region embedding"),
            },
        )
        for units in variants:
            with self.subTest(gain=units["gain"]):
                writer_state = copy.deepcopy(self.state)
                writer_state["units"] = units
                decoded = dump_contract.validate_shadow_state_document(writer_state)
                self.assertEqual(decoded["confirmed_cut_count"], 3)

    def test_host_linear_state_requires_coordinate_units_and_hard_cap_semantics(self):
        state = self._host_adapter_state()
        for name, value in (("joint_plane_mode_bits", 3), ("gain_seed_count", 1),
                            ("gain_display_limit", dump_contract._float32(0.04)),
                            ("gain_seed_first_low", 1000001)):
            self._set_state_word(state, name, value)
        state["decoded"].update(joint_plane_mode=3,
                                effective_gain=state["constants"]["requested_gain"])
        state["adaptation_semantics"] = dump_contract.adaptation_semantics_for_mode(3)
        state["units"]["gain"] = "one-eye source-U per coordinate unit"
        self._seal_camera_center(state)
        dump_contract.validate_shadow_state_document(state)
        curve_units = copy.deepcopy(state)
        curve_units["units"]["gain"] = "one-eye source-U per curve unit"
        with self.assertRaisesRegex(ValueError, "unknown units"):
            dump_contract.validate_shadow_state_document(curve_units)
        for key, previous in (("container_scale", "abi-retained-identity-pointwise-soft-container-is-map-local"),
                              ("near_curve", "fixed-contract-logarithmic-tau-independent-of-content-occupancy")):
            changed = copy.deepcopy(state)
            changed["adaptation_semantics"][key] = previous
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "adaptation semantics"):
                dump_contract.validate_shadow_state_document(changed)

    def test_single_dump_replay_reads_frame_and_derived_camera_validity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "shadow_state.json").write_text(
                json.dumps(self.state), encoding="utf-8")
            (root / "shadow_frame_stats.json").write_text(
                json.dumps(self.frame_stats), encoding="utf-8")

            summary = _inspect_optional_shadow_state(root)

            self.assertEqual(summary["status"], "validated")
            self.assertTrue(summary["frame_valid"])
            self.assertTrue(summary["camera_valid"])
            self.assertEqual(summary["calibration_revision"], 4)

    def test_single_dump_replay_validates_the_current_schema_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "dump_manifest.json").write_text(
                json.dumps(self.manifest), encoding="utf-8")

            summary = _inspect_optional_v2_dump_manifest(root)

            self.assertEqual(summary["status"], "validated")
            self.assertEqual(
                summary["manifest_schema"], dump_contract.DUMP_MANIFEST_SCHEMA)
            self.assertTrue(summary["active"])

    def test_single_dump_replay_joins_state_and_manifest_scale_to_model_provenance(self):
        state = {
            "status": "validated",
            "raw_coordinate_scale": 2.25,
        }
        manifest = {
            "status": "validated",
            "shadow_state_summary": {"raw_coordinate_scale": 2.25},
        }
        _authenticate_replay_coordinate_scale(2.25, state, manifest)

        for owner in (state, manifest["shadow_state_summary"]):
            changed_state = copy.deepcopy(state)
            changed_manifest = copy.deepcopy(manifest)
            target = (changed_state if owner is state else
                      changed_manifest["shadow_state_summary"])
            target["raw_coordinate_scale"] = 3.0
            with self.assertRaisesRegex(ValueError, "model provenance"):
                _authenticate_replay_coordinate_scale(
                    2.25, changed_state, changed_manifest)

    def test_paired_state_rejects_both_directions_of_frame_validity_mismatch(self):
        collapsed = copy.deepcopy(self.frame_stats)
        collapsed["named_values"]["population_std"] = (
            self.state["constants"]["collapse_abs_epsilon"])
        # Each document remains structurally valid in isolation; the pair is impossible under
        # the frame-resolve shader's exact validity predicate and must fail closed.
        dump_contract.validate_shadow_state_document(self.state)
        dump_contract.validate_shadow_frame_stats_document(collapsed)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "shadow_state.json").write_text(
                json.dumps(self.state), encoding="utf-8")
            (root / "shadow_frame_stats.json").write_text(
                json.dumps(collapsed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees with its current-frame"):
                _inspect_optional_shadow_state(root)

        # The serialized threshold may contain more binary64 precision than its
        # authenticated native word.  Pair validation must compare float32 words,
        # not let that extra JSON precision turn equality into a valid frame.
        native_epsilon = struct.unpack(
            "<f", struct.pack(
                "<f", self.state["constants"]["collapse_abs_epsilon"]))[0]
        same_word_state = copy.deepcopy(self.state)
        same_word_state["constants"]["collapse_abs_epsilon"] = (
            native_epsilon - 1.0e-15)
        same_word_stats = copy.deepcopy(self.frame_stats)
        same_word_stats["named_values"]["population_std"] = native_epsilon
        dump_contract.validate_shadow_state_document(same_word_state)
        dump_contract.validate_shadow_frame_stats_document(same_word_stats)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "shadow_state.json").write_text(
                json.dumps(same_word_state), encoding="utf-8")
            (root / "shadow_frame_stats.json").write_text(
                json.dumps(same_word_stats), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees with its current-frame"):
                _inspect_optional_shadow_state(root)

        flat_state = copy.deepcopy(self.state)
        for name, value in (("frame_valid", 0.0), ("container_scale", 1.0)):
            flat_state["named_values"][name] = value
            field = next(item for item in flat_state["fields"] if item["name"] == name)
            field["value"] = value
        flat_state["decoded"]["frame_valid"] = False
        flat_state["decoded"]["container_scale"] = 1.0
        flat_state["decoded"]["effective_gain"] = 0.0
        self._seal_camera_center(flat_state)
        dump_contract.validate_shadow_state_document(flat_state)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "shadow_state.json").write_text(
                json.dumps(flat_state), encoding="utf-8")
            (root / "shadow_frame_stats.json").write_text(
                json.dumps(self.frame_stats), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees with its current-frame"):
                _inspect_optional_shadow_state(root)

    def test_serialization_schema_is_independent_and_both_bindings_are_required(self):
        self.assertNotEqual(
            dump_contract.SHADOW_STATE_DUMP_SCHEMA, coordinate.CONTRACT_SCHEMA)
        changed = copy.deepcopy(self.state)
        changed["schema"] = coordinate.CONTRACT_SCHEMA
        with self.assertRaisesRegex(ValueError, "serialization schema"):
            dump_contract.validate_shadow_state_document(changed)
        changed = copy.deepcopy(self.state)
        changed["coordinate_contract"]["schema"] += 1
        with self.assertRaisesRegex(ValueError, "contract binding"):
            dump_contract.validate_shadow_state_document(changed)

    def test_same_count_reorder_and_tag_change_fail_closed(self):
        changed = copy.deepcopy(self.state)
        changed["fields"][0], changed["fields"][1] = changed["fields"][1], changed["fields"][0]
        changed["fields"][0]["index"] = 0
        changed["fields"][1]["index"] = 1
        with self.assertRaisesRegex(ValueError, "field 0"):
            dump_contract.validate_shadow_state_document(changed)
        changed = copy.deepcopy(self.state)
        changed["coordinate_contract"]["tag"] ^= 1
        with self.assertRaisesRegex(ValueError, "contract binding"):
            dump_contract.validate_shadow_state_document(changed)

    def test_reserved_calibration_revision_fails_closed(self):
        changed = copy.deepcopy(self.state)
        self._set_state_word(changed, "calibration_revision", 0xFFFFFFFF)
        changed["decoded"]["calibration_revision"] = 0xFFFFFFFF
        with self.assertRaisesRegex(ValueError, "reserved calibration revision"):
            dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.state)
        changed["decoded"]["calibration_revision"] = 0xFFFFFFFF
        with self.assertRaisesRegex(ValueError, "reserved calibration revision"):
            dump_contract.validate_shadow_state_document(changed)

    def test_shader_source_identity_change_fails_closed(self):
        for key, replacement in (
                ("source_closure_schema", 3),
                ("source_compile_flags", 0),
                ("source_macro_count", 1),
                ("source_closure_sha256", "0" * 64)):
            changed = copy.deepcopy(self.state)
            changed["coordinate_contract"][key] = replacement
            with self.subTest(key=key), self.assertRaisesRegex(
                    ValueError, "contract binding"):
                dump_contract.validate_shadow_state_document(changed)

    def test_derived_gain_and_convergence_must_match_words(self):
        changed = copy.deepcopy(self.state)
        changed["decoded"]["effective_gain"] += 0.001
        with self.assertRaisesRegex(ValueError, "decoded values"):
            dump_contract.validate_shadow_state_document(changed)
        changed = copy.deepcopy(self.state)
        changed["named_values"]["convergence_curve"] = 0.1
        changed["fields"][2]["value"] = 0.1
        changed["decoded"]["convergence_curve"] = 0.1
        self._seal_camera_center(changed)
        with self.assertRaises(ValueError):
            dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.state)
        changed["named_values"]["container_scale"] = 0.9
        changed["fields"][3]["value"] = 0.9
        changed["decoded"]["container_scale"] = 0.9
        # V2's effective gain is the literal request; attenuation is pointwise in the map.
        # Keep the derived field internally consistent so validation reaches the invalid
        # compatibility-state value rather than failing earlier on a stale V1 derivation.
        changed["decoded"]["effective_gain"] = changed["constants"]["requested_gain"]
        with self.assertRaises(ValueError):
            dump_contract.validate_shadow_state_document(changed)

    def test_zero_requested_pop_and_gain_fail_closed_at_both_dump_boundaries(self):
        changed = copy.deepcopy(self.state)
        changed["constants"]["requested_gain"] = 0.0
        changed["constants"]["requested_pop_strength"] = 0.0
        changed["decoded"]["requested_gain"] = 0.0
        changed["decoded"]["requested_pop_strength"] = 0.0
        changed["decoded"]["effective_gain"] = 0.0
        with self.assertRaisesRegex(ValueError, "invalid runtime constants"):
            dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.manifest)
        summary = changed["parallax_v2_shadow"]["state"]
        summary["requested_gain"] = 0.0
        summary["requested_pop_strength"] = 0.0
        summary["effective_gain"] = 0.0
        with self.assertRaisesRegex(ValueError, "state summary"):
            dump_contract.validate_v2_dump_manifest_document(changed)

    def test_raw_scale_authentication_uses_the_native_absolute_tolerance(self):
        changed = copy.deepcopy(self.state)
        # The declared decoded value is deliberately closer to the raw scale
        # than the reciprocal encoded by the authenticated float32 word.  The
        # reader must authenticate the word itself rather than compose two
        # independent tolerances through the decoded mirror.
        inverse_scale = 0.44444364309310913
        self._set_state_word(changed, "inverse_scale", inverse_scale)
        changed["decoded"]["latched_scale"] = 2.250002
        self._seal_camera_center(changed)
        with self.assertRaisesRegex(ValueError, "decoded values|out of range"):
            dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.manifest)
        changed["parallax_v2_shadow"]["state"]["latched_scale"] = (
            2.250002145767212)
        dump_contract.validate_v2_dump_manifest_document(changed)

    def test_requested_gain_authentication_uses_the_native_absolute_tolerance(self):
        changed = copy.deepcopy(self.state)
        requested_gain = (
            changed["constants"]["requested_pop_strength"] *
            changed["constants"]["gain_per_pop"] + 5.0e-8)
        changed["constants"]["requested_gain"] = requested_gain
        changed["decoded"]["requested_gain"] = requested_gain
        changed["decoded"]["effective_gain"] = requested_gain
        dump_contract.validate_shadow_state_document(changed)

    def test_convergence_curve_is_exact_at_both_dump_boundaries(self):
        changed = copy.deepcopy(self.state)
        self._set_state_word(changed, "convergence_curve", 1.0e-9)
        changed["decoded"]["convergence_curve"] = 1.0e-9
        self._seal_camera_center(changed)
        with self.assertRaises(ValueError):
            dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.manifest)
        changed["parallax_v2_shadow"]["state"]["convergence_curve"] = 1.0e-9
        with self.assertRaisesRegex(ValueError, "state summary"):
            dump_contract.validate_v2_dump_manifest_document(changed)

        for name, value in (("convergence_curve", 1.0e-9),
                            ("container_scale", 1.0 + 1.0e-9)):
            changed = copy.deepcopy(self.state)
            changed["decoded"][name] = value
            with self.subTest(decoded=name), self.assertRaisesRegex(
                    ValueError, "decoded values"):
                dump_contract.validate_shadow_state_document(changed)

    def test_generated_constants_and_state_mirrors_are_exact_float32_identities(self):
        changed = copy.deepcopy(self.state)
        changed["constants"]["max_horizontal_slope"] *= 1.00001
        with self.assertRaisesRegex(ValueError, "generated contract"):
            dump_contract.validate_shadow_state_document(changed)

        for location in ("fields", "named_values"):
            changed = copy.deepcopy(self.state)
            if location == "fields":
                field = next(item for item in changed["fields"]
                             if item["name"] == "inverse_scale")
                field["value"] += 1.0e-9
            else:
                changed["named_values"]["inverse_scale"] += 1.0e-9
            with self.subTest(location=location), self.assertRaisesRegex(
                    ValueError, "disagrees"):
                dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.state)
        changed["named_values"]["container_scale"] = True
        with self.assertRaisesRegex(ValueError, "named_values.container_scale"):
            dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.state)
        field = next(item for item in changed["fields"]
                     if item["name"] == "convergence_curve")
        field["value"] = -0.0
        # Seal with the mutated native word while leaving both redundant mirrors
        # at +0.0.  The reader must preserve the sign bit in its exact join.
        checksum = dump_contract.camera_center_integrity_bits(
            changed["named_values"]["center"],
            changed["named_values"]["inverse_scale"],
            field["value"],
            changed["named_values"]["calibration_revision"],
        )
        self._set_state_word(changed, "camera_center_integrity_bits", checksum)
        changed["decoded"]["camera_center_integrity_bits"] = checksum
        with self.assertRaisesRegex(ValueError, "disagrees"):
            dump_contract.validate_shadow_state_document(changed)

        for location in ("constants", "field"):
            changed = copy.deepcopy(self.state)
            if location == "constants":
                changed["constants"]["max_horizontal_slope"] = 1.0e300
            else:
                self._set_state_word(changed, "center", 1.0e300)
            with self.subTest(huge=location), self.assertRaisesRegex(
                    ValueError, "float32"):
                dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.state)
        changed["constants"]["requested_pop_strength"] = 1.0e300
        changed["constants"]["requested_gain"] = 3.75e297
        changed["decoded"]["requested_pop_strength"] = 1.0e300
        changed["decoded"]["requested_gain"] = 3.75e297
        changed["decoded"]["effective_gain"] = 3.75e297
        with self.assertRaisesRegex(ValueError, "float32"):
            dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.manifest)
        summary = changed["parallax_v2_shadow"]["state"]
        summary["requested_pop_strength"] = 1.0e300
        summary["requested_gain"] = 3.75e297
        summary["effective_gain"] = 3.75e297
        with self.assertRaisesRegex(ValueError, "float32"):
            dump_contract.validate_v2_dump_manifest_document(changed)

        changed = copy.deepcopy(self.state)
        changed["constants"]["requested_pop_strength"] = 1.0e-100
        changed["constants"]["requested_gain"] = 3.75e-103
        changed["decoded"]["requested_pop_strength"] = 1.0e-100
        changed["decoded"]["requested_gain"] = 3.75e-103
        changed["decoded"]["effective_gain"] = 3.75e-103
        with self.assertRaisesRegex(ValueError, "invalid runtime constants"):
            dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.manifest)
        summary = changed["parallax_v2_shadow"]["state"]
        summary["requested_pop_strength"] = 1.0e-100
        summary["requested_gain"] = 3.75e-103
        summary["effective_gain"] = 3.75e-103
        with self.assertRaisesRegex(ValueError, "state summary"):
            dump_contract.validate_v2_dump_manifest_document(changed)

        # Authentication uses the native float32 values, not the wider JSON
        # intermediates.  This pair is exactly related after float32 rounding.
        changed = copy.deepcopy(self.state)
        native_pop = 3930.8535162710605
        native_gain = 14.740700897328566
        changed["constants"]["requested_pop_strength"] = native_pop
        changed["constants"]["requested_gain"] = native_gain
        changed["decoded"]["requested_pop_strength"] = native_pop
        changed["decoded"]["requested_gain"] = native_gain
        changed["decoded"]["effective_gain"] = native_gain
        dump_contract.validate_shadow_state_document(changed)

    def test_center_integrity_authorization_and_reserved_mapping_state_fail_closed(self):
        changed = copy.deepcopy(self.state)
        self._set_state_word(changed, "center", 2.25)
        with self.assertRaisesRegex(ValueError, "center integrity checksum"):
            dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.state)
        self._set_state_word(changed, "renderer_authorization_bits", 1)
        with self.assertRaisesRegex(ValueError, "renderer authorization"):
            dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.state)
        self._set_state_word(changed, "joint_plane_mode_bits", 1)
        with self.assertRaisesRegex(ValueError, "center integrity checksum"):
            dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.state)
        self._set_state_word(changed, "mapping_state_reserved_2", 1)
        with self.assertRaisesRegex(ValueError, "reserved mapping state"):
            dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.state)
        for name, value in (
                ("center", 0.0), ("inverse_scale", 0.0),
                ("container_scale", 1.0), ("frame_valid", 0.0)):
            self._set_state_word(changed, name, value)
        changed["decoded"].update({
            "frame_valid": False,
            "camera_valid": False,
            "latched_scale": 0.0,
            "container_scale": 1.0,
            "effective_gain": 0.0,
        })
        for descriptor in coordinate.load_contract()["shadow_state"]["fields"][12:]:
            self._set_state_word(changed, descriptor["name"],
                                 0.0 if descriptor["gpu_encoding"] == "float" else 0)
        self._seal_camera_center(changed)
        dump_contract.validate_shadow_state_document(changed)

    def test_invalid_state_preserves_requested_gain_but_effective_is_zero(self):
        changed = copy.deepcopy(self.state)
        replacements = {
            "center": 0.0, "inverse_scale": 0.0, "convergence_curve": 0.0,
            "container_scale": 1.0, "frame_valid": 0.0,
            "camera_center_integrity_bits": dump_contract.camera_center_integrity_bits(
                0.0, 0.0, 0.0,
                changed["named_values"]["calibration_revision"]),
            "renderer_authorization_bits": 0,
        }
        for name, value in replacements.items():
            changed["named_values"][name] = value
            changed["fields"][[field["name"] for field in changed["fields"]].index(name)][
                "value"] = value
        changed["decoded"]["frame_valid"] = False
        changed["decoded"]["camera_valid"] = False
        changed["decoded"]["latched_scale"] = 0.0
        changed["decoded"]["container_scale"] = 1.0
        changed["decoded"]["effective_gain"] = 0.0
        changed["decoded"]["camera_center_integrity_bits"] = (
            replacements["camera_center_integrity_bits"])
        changed["decoded"]["renderer_authorization_bits"] = 0
        for descriptor in coordinate.load_contract()["shadow_state"]["fields"][12:]:
            self._set_state_word(changed, descriptor["name"],
                                 0.0 if descriptor["gpu_encoding"] == "float" else 0)
        self._seal_camera_center(changed)
        dump_contract.validate_shadow_state_document(changed)
        self.assertEqual(
            changed["decoded"]["requested_gain"], changed["constants"]["requested_gain"])

        retained = copy.deepcopy(self.state)
        retained["named_values"]["frame_valid"] = 0.0
        retained["fields"][5]["value"] = 0.0
        retained["named_values"]["container_scale"] = 1.0
        retained["fields"][3]["value"] = 1.0
        retained["decoded"]["frame_valid"] = False
        retained["decoded"]["container_scale"] = 1.0
        retained["decoded"]["effective_gain"] = 0.0
        self._set_state_word(retained, "renderer_authorization_bits", 0)
        retained["decoded"]["renderer_authorization_bits"] = 0
        dump_contract.validate_shadow_state_document(retained)
        self.assertTrue(retained["decoded"]["camera_valid"])

    def test_valid_state_requires_a_real_camera_and_consistent_requested_gain(self):
        changed = copy.deepcopy(self.state)
        changed["named_values"]["inverse_scale"] = 0.0
        changed["fields"][1]["value"] = 0.0
        changed["decoded"]["camera_valid"] = False
        changed["decoded"]["latched_scale"] = 0.0
        self._seal_camera_center(changed)
        with self.assertRaisesRegex(ValueError, "out of range"):
            dump_contract.validate_shadow_state_document(changed)
        changed = copy.deepcopy(self.state)
        changed["constants"]["requested_gain"] += 0.001
        changed["decoded"]["requested_gain"] = changed["constants"]["requested_gain"]
        changed["decoded"]["effective_gain"] = (
            changed["constants"]["requested_gain"] *
            changed["named_values"]["container_scale"])
        with self.assertRaisesRegex(ValueError, "requested gain"):
            dump_contract.validate_shadow_state_document(changed)

    def test_removed_state_and_constant_fields_are_rejected(self):
        for removed in (
                "upper_l4", "source_u_safety_scale", "source_u_budget",
                "collapse_relative_epsilon", "spatial_support_ratio"):
            self.assertNotIn(removed, self.state["named_values"])
            self.assertNotIn(removed, self.state["constants"])
        changed = copy.deepcopy(self.state)
        changed["constants"]["source_u_budget"] = 0.01
        with self.assertRaisesRegex(ValueError, "constants object"):
            dump_contract.validate_shadow_state_document(changed)

    def test_frame_stats_schema_and_exact_layout_fail_closed(self):
        changed = copy.deepcopy(self.frame_stats)
        changed["schema"] += 1
        with self.assertRaisesRegex(ValueError, "serialization schema"):
            dump_contract.validate_shadow_frame_stats_document(changed)
        changed = copy.deepcopy(self.frame_stats)
        changed["named_values"].pop("reserved")
        with self.assertRaisesRegex(ValueError, "exact stats layout"):
            dump_contract.validate_shadow_frame_stats_document(changed)
        changed = copy.deepcopy(self.frame_stats)
        changed["named_values"]["valid_count"] = 15.0
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            dump_contract.validate_shadow_frame_stats_document(changed)
        changed = copy.deepcopy(self.frame_stats)
        changed["named_values"]["population_std"] = -0.1
        with self.assertRaisesRegex(ValueError, "counts or spread"):
            dump_contract.validate_shadow_frame_stats_document(changed)
        changed = copy.deepcopy(self.frame_stats)
        changed["named_values"].update({
            "valid": 0.0,
            "valid_count": 15.0,
            "mean": 1.0,
            "population_std": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
        })
        with self.assertRaisesRegex(ValueError, "canonical zero"):
            dump_contract.validate_shadow_frame_stats_document(changed)


class SubtitleContourTupleContractTests(unittest.TestCase):
    """Small independently authored tuple fixtures; no detector or native-renderer replica."""

    @staticmethod
    def _legacy_word(kind, dx=0, dy=0, weak=False):
        return kind | (int(weak) << 3) | ((dx + 4) << 4) | ((dy + 4) << 8)

    def _tuple(self, *, rectangles=True, ocr_authoritative=True, geometry_ready=True,
               source_width=24, source_height=12, invalid_probe=False, current_kind_mask=0):
        from fractions import Fraction
        import numpy as np

        policy = coordinate.EXPECTED_SUBTITLE_OCR["contour_policy"]
        content = (1, 1, 6, 5)
        field_width, field_height = 7, 6
        crop_height = min(source_height, (source_width + 5) // 6)
        crop_top = source_height - crop_height
        eligible = max(source_width, source_height) <= policy["source_max_extent"]
        snapshot = eligible and ocr_authoritative and geometry_ready
        current_count = int(rectangles and snapshot)
        valid = bool(snapshot and current_count and not invalid_probe)
        shape = ((crop_height, source_width) if eligible else (1, 1))
        source = np.zeros((*shape, 4), dtype="<f2")
        history = np.zeros((*shape, 4), dtype="<f4")
        support = np.zeros(shape, dtype="<u4")
        probability = np.zeros(coordinate.SUBTITLE_OCR.output_shape, dtype="<f4")
        if snapshot:
            source[..., :3] = np.float16(0.25)
            source[..., 3] = np.float16(1)
            probability.fill(np.float32(0.6))
        if valid:
            support[1, 12] = 1
            support[1, 13] = 1
            support[1, 15] = 1
            source[1, 12, :3] = np.float16(0.875)
            source[1, 13:16, :3] = np.float16(0.0625)
        if snapshot:
            history[..., :3] = source[..., :3].astype(np.float32)
            history[..., 3] = np.float32(policy["history_feature_valid_bit"])
            history[..., 3][support != 0] = np.float32(
                policy["history_feature_valid_bit"] | policy["history_feature_mask_bit"] |
                (policy["history_confirmation_observations"] << policy["history_feature_count_shift"]))
        # Brute geometric intersection of each source pixel's rational area with each node's
        # open +/-1 sampling support. This does not use the reader's summed-area gather.
        pin = np.zeros((field_height, field_width), dtype="<u4")
        for py, px in zip(*np.nonzero(support)):
            xlow = content[0] + Fraction(int(px) * 5, source_width) - Fraction(1, 2)
            xhigh = content[0] + Fraction((int(px) + 1) * 5, source_width) - Fraction(1, 2)
            ylow = content[1] + Fraction((int(py) + crop_top) * 4, source_height) - Fraction(1, 2)
            yhigh = content[1] + Fraction((int(py) + crop_top + 1) * 4, source_height) - Fraction(1, 2)
            for y in range(content[1], content[3]):
                for x in range(content[0], content[2]):
                    if xlow < x + 1 and xhigh > x - 1 and ylow < y + 1 and yhigh > y - 1:
                        pin[y, x] = 1
        distance = np.full(pin.shape, np.inf, dtype="<f4")
        offset = policy["gaussian_center_offset_source_pixels"]
        centers = [(int(px) + dx, int(py) + dy) for py, px in zip(*np.nonzero(support))
                   for dx, dy in ((0, 0), (-offset, 0), (offset, 0), (0, -offset), (0, offset))]
        for y in range(content[1], content[3]):
            for x in range(content[0], content[2]):
                if centers:
                    qx = (x - content[0] + .5) * source_width / (content[2] - content[0]) - .5
                    qy = (y - content[1] + .5) * source_height / (content[3] - content[1]) - .5 - crop_top
                    distance[y, x] = min((qx - px) ** 2 + (qy - py) ** 2 for px, py in centers)
        cx = np.clip(np.arange(field_width), content[0], content[2] - 1)
        cy = np.clip(np.arange(field_height), content[1], content[3] - 1)
        weight = (dump_contract._subtitle_gaussian_weight_from_q(distance[np.ix_(cy, cx)], source_width, source_height)
                  if centers else np.zeros(pin.shape, dtype="<f4"))
        common = {"evidence_frame_id": 17, "analysis_generation": (1 << 32) + 2,
                  "source_width": source_width, "source_height": source_height,
                  "field_width": field_width, "field_height": field_height, "tensor_content": content,
                  "roi_top": 2, "roi_bottom": 5, "scene_epoch": 7,
                  "observation_timestamp_us": (1 << 32) + 1234,
                  "geometry_ready": geometry_ready, "ocr_authoritative": ocr_authoritative,
                  "current_rectangles": ([{"left": 2, "top": 3, "right": 5, "bottom": 5}]
                                         if current_count else [])}
        common["plane_state"] = {
            "current_count": current_count, "current_slot_mask": (1 << current_count) - 1,
            "current_ribbon_mask": current_kind_mask, "flags": int(snapshot),
            "warmup_packed": 0,
            "current_valid": bool(snapshot), "cover_geometry_verified": True,
            "current_rectangles": copy.deepcopy(common["current_rectangles"]),
            "matched_frame_id": common["evidence_frame_id"],
            "observation_timestamp_us": common["observation_timestamp_us"],
            "analysis_generation": common["analysis_generation"], "scene_epoch": common["scene_epoch"],
            "source_width": source_width, "source_height": source_height,
            "field_width": field_width, "field_height": field_height, "tensor_content_rect": content,
            "adaptive": {"invalid_count": int(invalid_probe)},
        }
        identity = [policy["history_schema"], policy["history_tag"], int(valid), 7 if snapshot else 0,
                    17, 0, 2, 1, source_width, source_height, crop_top, crop_height, 7, 6, 2, 5,
                    *content, 1234, 1, policy["recipe"], int(snapshot), *([0] * 8)]
        payloads = {"contour_support": support.tobytes(), "contour_field_pin": pin.tobytes(),
                    "contour_distance": distance.tobytes(), "contour_identity": struct.pack("<32I", *identity),
                    "contour_history": history.tobytes(), "contour_source": source.tobytes(),
                    "ocr_probability": probability.tobytes(), "gaussian_weight": weight.tobytes(),
                    "gaussian_blend": bytes(field_width * field_height * 4),
                    "gaussian_vertical_conditioned": bytes(field_width * field_height * 4),
                    "gaussian_control": struct.pack("<I", int(bool(centers)))}
        return payloads, common


    def test_nonempty_tuple_matches_independent_rational_footprint_and_exhaustive_distance(self):
        payloads, common = self._tuple()
        result = dump_contract.validate_subtitle_contour_tuple(payloads, **common)
        self.assertTrue(result["tuple_verified"])
        self.assertTrue(result["authority_valid"])
        self.assertEqual(result["support_pixels"], 3)
        self.assertEqual(result["weak_support_pixels"], 0)
        self.assertEqual(result["history_rgb_valid_pixels"], 96)
        self.assertEqual(result["history_confirmation_count_histogram"],
                         {"0": 93, "1": 0, "2": 0, "3": 0, "4": 3})
        self.assertEqual(result["history_anchor_current_bit_mismatch_pixels"], 0)
        self.assertFalse(result["history_confirmation_chain_verified"])
        self.assertIn("not-semantic-text", result["support_semantics"])
        self.assertTrue(result["integer_bilinear_projection_verified"])
        self.assertTrue(result["squared_source_distance_verified"])
        self.assertEqual(result["inference_owner_frame_id"], 17)
        self.assertEqual(result["inference_owner_observation_timestamp_us"], (1 << 32) + 1234)
        self.assertIn("no-current-presentation", result["source_pixel_relation"])

    def test_all_eleven_roles_reject_missing_extra_truncated_appended_and_untyped_payloads(self):
        payloads, common = self._tuple()
        for role in payloads:
            for fault in ("missing", "short", "long", "untyped"):
                changed = dict(payloads)
                if fault == "missing":
                    changed.pop(role)
                elif fault == "short":
                    changed[role] = changed[role][:-1]
                elif fault == "long":
                    changed[role] += b"\x00"
                else:
                    changed[role] = list(changed[role])
                with self.subTest(role=role, fault=fault), self.assertRaisesRegex(ValueError, "eleven|byte length/type"):
                    dump_contract.validate_subtitle_contour_tuple(changed, **common)
        changed = {**payloads, "alpha": b""}
        with self.assertRaisesRegex(ValueError, "eleven"):
            dump_contract.validate_subtitle_contour_tuple(changed, **common)

    def test_every_sct5_identity_word_binds_source_clock_shape_cut_roi_recipe_snapshot_and_reserved_zero(self):
        payloads, common = self._tuple()
        words = list(struct.unpack("<32I", payloads["contour_identity"]))
        for index in range(32):
            changed = dict(payloads)
            modified = words.copy()
            modified[index] ^= 1
            changed["contour_identity"] = struct.pack("<32I", *modified)
            with self.subTest(word=index), self.assertRaisesRegex(ValueError, "SCT5 temporal-mask identity"):
                dump_contract.validate_subtitle_contour_tuple(changed, **common)

    def test_sct5_confirmation_snapshot_does_not_depend_on_slot_order_count_kind_or_split(self):
        payloads, common = self._tuple()
        whole = common["current_rectangles"]
        split = [{"left": 2, "top": 3, "right": 4, "bottom": 5},
                 {"left": 4, "top": 3, "right": 5, "bottom": 5}]
        expanded = [{"left": 1, "top": 3, "right": 6, "bottom": 5}]
        for rectangles, kinds in ((whole, 1), (split, 0), (list(reversed(split)), 3),
                                  (expanded, 0), (whole * 8, 255)):
            context = copy.deepcopy(common)
            context["current_rectangles"] = copy.deepcopy(rectangles)
            plane = context["plane_state"]
            plane["current_rectangles"] = copy.deepcopy(rectangles)
            plane["current_count"] = len(rectangles)
            plane["current_slot_mask"] = (1 << len(rectangles)) - 1
            plane["current_ribbon_mask"] = kinds
            with self.subTest(rectangles=rectangles, kinds=kinds):
                result = dump_contract.validate_subtitle_contour_tuple(payloads, **context)
                self.assertEqual(result["support_pixels"], 3)
                self.assertEqual(result["history_confirmation_count_histogram"]["4"], 3)
                self.assertFalse(result["history_confirmation_chain_verified"])
        self.assertEqual(struct.unpack("<32I", payloads["contour_identity"])[24:], (0,) * 8)

    def test_contour_requires_actual_current_ocr_geometry_and_exact_uip1_header_context(self):
        payloads, common = self._tuple()
        mutations = {"no-ocr-geometry": ("cover_geometry_verified", False),
                     "frame": ("matched_frame_id", 18),
                     "clock": ("observation_timestamp_us", (1 << 32) + 1235),
                     "generation": ("analysis_generation", 3), "cut": ("scene_epoch", 8),
                     "source": ("source_width", 25), "field": ("field_height", 7),
                     "content": ("tensor_content_rect", (0, 1, 6, 5)),
                     "count": ("current_count", 0), "mask": ("current_slot_mask", 2),
                     "flag": ("flags", 0), "current-verdict": ("current_valid", False),
                     "covers": ("current_rectangles", [])}
        for name, (key, value) in mutations.items():
            context = copy.deepcopy(common)
            context["plane_state"][key] = value
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "authenticated UIP4"):
                dump_contract.validate_subtitle_contour_tuple(payloads, **context)

    def test_invalid_base_probe_retains_current_covers_and_source_rgb_but_no_confirmation_or_gaussian(self):
        payloads, common = self._tuple(invalid_probe=True)
        result = dump_contract.validate_subtitle_contour_tuple(payloads, **common)
        self.assertTrue(common["plane_state"]["current_valid"])
        self.assertEqual(common["plane_state"]["current_count"], 1)
        self.assertFalse(result["authority_valid"])
        self.assertEqual(result["history_rgb_valid_pixels"], 96)
        self.assertEqual(result["support_pixels"], 0)
        self.assertEqual(result["history_confirmation_count_histogram"]["0"], 96)
        self.assertFalse(result["gaussian_smoothing_enabled"])
        identity = struct.unpack("<32I", payloads["contour_identity"])
        self.assertEqual((identity[2], identity[23]), (0, 1))


    def test_native_binary_support_rejects_legacy_kinds_ages_anchors_and_reserved_bits(self):
        import numpy as np

        payloads, common = self._tuple()
        original = np.frombuffer(payloads["contour_support"], dtype="<u4").reshape(4, 24)
        faults = {"kind-two": 2, "kind-four": 4, "weak": 8,
                  "reserved": 1 | (1 << 12), "all-ones": 0xffffffff,
                  "legacy-self-body": self._legacy_word(1),
                  "legacy-outline": self._legacy_word(3, dx=-1),
                  "legacy-weak-shadow": self._legacy_word(4, dx=-3, weak=True)}
        for fault, word in faults.items():
            changed = dict(payloads)
            values = original.copy()
            values[1, 13] = word
            changed["contour_support"] = values.tobytes()
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, "nonbinary support"):
                dump_contract.validate_subtitle_contour_tuple(changed, **common)
        changed = dict(payloads)
        values = original.copy()
        values[0, 0] = 1
        changed["contour_support"] = values.tobytes()
        with self.assertRaisesRegex(ValueError, "escapes current covers"):
            dump_contract.validate_subtitle_contour_tuple(changed, **common)

    def test_pin_distance_and_history_reject_same_sized_unjoined_payloads(self):
        import numpy as np

        payloads, common = self._tuple()
        for role in ("contour_field_pin", "contour_distance", "contour_history"):
            changed = dict(payloads)
            if role == "contour_history":
                values = np.frombuffer(payloads[role], dtype="<u4").copy()
                values[0] ^= 1
            elif role == "contour_distance":
                values = np.frombuffer(payloads[role], dtype="<f4").copy()
                values[2 * 7 + 3] += np.float32(.05)
            else:
                values = np.frombuffer(payloads[role], dtype="<u4").copy()
                values[2 * 7 + 3] ^= 1
            changed[role] = values.tobytes()
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "footprint|IEEE coordinate bound|history"):
                dump_contract.validate_subtitle_contour_tuple(changed, **common)

    def test_valid_empty_source_snapshots_preserve_opaque_probability_fault_bits(self):
        import numpy as np

        payloads, common = self._tuple(rectangles=False)
        probability = np.frombuffer(payloads["ocr_probability"], dtype="<u4").copy()
        fault_words = np.array([0x7fc12345, 0x7f800000, 0xff800000, 0xbf800000, 0x40000000], dtype="<u4")
        probability[:5] = fault_words
        payloads["ocr_probability"] = probability.tobytes()
        result = dump_contract.validate_subtitle_contour_tuple(payloads, **common)
        self.assertFalse(result["authority_valid"])
        self.assertTrue(result["snapshot_gate_valid"])
        self.assertEqual(result["history_rgb_valid_pixels"], 96)
        identity = struct.unpack("<32I", payloads["contour_identity"])
        self.assertEqual((identity[2], identity[3], identity[23]), (0, 7, 1))
        self.assertEqual(result["probability_nonfinite_words"], 3)
        self.assertEqual(result["probability_out_of_range_finite_words"], 2)
        self.assertTrue(np.array_equal(np.frombuffer(payloads["ocr_probability"], dtype="<u4")[:5], fault_words))
        self.assertIn("not-glyph-alpha", result["probability_semantics"])

    def test_valid_empty_mask_keeps_source_history_and_empty_distance(self):
        import numpy as np

        payloads, common = self._tuple()
        source = np.frombuffer(payloads["contour_source"], dtype="<f2").reshape(4, 24, 4)
        history = np.zeros((4, 24, 4), dtype="<f4")
        history[..., :3] = source[..., :3].astype(np.float32)
        history[..., 3] = np.float32(coordinate.EXPECTED_SUBTITLE_OCR["contour_policy"]["history_feature_valid_bit"])
        payloads["contour_support"] = bytes(4 * 24 * 4)
        payloads["contour_field_pin"] = bytes(7 * 6 * 4)
        payloads["contour_distance"] = np.full((6, 7), np.inf, dtype="<f4").tobytes()
        payloads["contour_history"] = history.tobytes()
        payloads["gaussian_weight"] = bytes(6 * 7 * 4)
        payloads["gaussian_control"] = struct.pack("<I", 0)
        result = dump_contract.validate_subtitle_contour_tuple(payloads, **common)
        self.assertTrue(result["authority_valid"])
        self.assertFalse(result["gaussian_smoothing_enabled"])
        self.assertEqual(result["support_pixels"], 0)
        self.assertEqual(result["field_pin_nodes"], 0)

    def test_empty_rgb_history_is_required_without_pin_authority_and_does_not_authorize_support(self):
        import numpy as np

        payloads, common = self._tuple(rectangles=False)
        result = dump_contract.validate_subtitle_contour_tuple(payloads, **common)
        self.assertFalse(result["authority_valid"])
        self.assertEqual(result["support_pixels"], 0)
        self.assertEqual(result["history_rgb_valid_pixels"], 96)
        with self.assertRaisesRegex(ValueError, "temporal history"):
            dump_contract.validate_subtitle_contour_tuple(
                {**payloads, "contour_history": bytes(96 * 16)}, **common)
        support = np.frombuffer(payloads["contour_support"], dtype="<u4").copy()
        support[0] = 1
        with self.assertRaisesRegex(ValueError, "invalid contour pin authority"):
            dump_contract.validate_subtitle_contour_tuple(
                {**payloads, "contour_support": support.tobytes()}, **common)

    def test_history_confirmation_codes_and_support_correspondence_reject_unknown_or_unjoined_states(self):
        import numpy as np

        payloads, common = self._tuple()
        original = np.frombuffer(payloads["contour_history"], dtype="<f4").reshape(4, 24, 4)
        self.assertEqual((float(original[0, 0, 3]), float(original[1, 12, 3])), (2.0, 19.0))
        for y, x, code in ((0, 0, 0), (0, 0, 1), (0, 0, 3), (1, 12, 2),
                           (1, 12, 6), (1, 12, 10), (1, 12, 14),
                           (0, 0, 18), (0, 0, 19), (0, 0, 22),
                           (0, 0, 2.5), (0, 0, 11), (0, 0, float("nan")),
                           (0, 0, float("inf")), (0, 0, -0.0)):
            changed = original.copy()
            changed[y, x, 3] = np.float32(code)
            with self.subTest(pixel=(y, x), code=code), self.assertRaisesRegex(ValueError, "temporal history"):
                dump_contract.validate_subtitle_contour_tuple(
                    {**payloads, "contour_history": changed.tobytes()}, **common)
        changed = original.view(np.uint32).copy()
        changed[0, 0, 0] ^= 1
        with self.assertRaisesRegex(ValueError, "temporal history"):
            dump_contract.validate_subtitle_contour_tuple(
                {**payloads, "contour_history": changed.tobytes()}, **common)

    def test_legal_streaks_keep_current_first_anchor_then_bounded_fixed_anchor_and_report_counts(self):
        import numpy as np

        payloads, common = self._tuple()
        history = np.frombuffer(payloads["contour_history"], dtype="<f4").reshape(4, 24, 4).copy()
        for y, code in ((1, 6), (2, 10), (3, 14)):
            history[y, 11, 3] = np.float32(code)
        history[2, 11, 0] += np.float32(0.0625)
        history[3, 11, 1] -= np.float32(0.0625)
        history[1, 12, 0] -= np.float32(0.0625)
        result = dump_contract.validate_subtitle_contour_tuple(
            {**payloads, "contour_history": history.tobytes()}, **common)
        self.assertEqual(result["history_confirmation_count_histogram"],
                         {"0": 90, "1": 1, "2": 1, "3": 1, "4": 3})
        self.assertEqual(result["history_anchor_current_bit_mismatch_pixels"], 3)
        self.assertEqual(result["history_anchor_current_numeric_mismatch_pixels"], 3)
        self.assertFalse(result["history_confirmation_chain_verified"])
        for y, x, lane, value in ((1, 11, 0, 0.3125), (2, 11, 0, 0.375),
                                  (3, 11, 1, -0.001), (1, 12, 0, float("nan")),
                                  (1, 12, 0, float("inf")), (1, 12, 0, 1.001)):
            changed = history.copy()
            changed[y, x, lane] = np.float32(value)
            with self.subTest(pixel=(y, x), lane=lane, value=value), self.assertRaisesRegex(ValueError, "temporal history"):
                dump_contract.validate_subtitle_contour_tuple(
                    {**payloads, "contour_history": changed.tobytes()}, **common)

    def test_streak_requires_current_cover_and_empty_snapshots_only_admit_count_zero(self):
        import numpy as np

        payloads, common = self._tuple()
        history = np.frombuffer(payloads["contour_history"], dtype="<f4").reshape(4, 24, 4).copy()
        history[0, 0, 3] = np.float32(6)
        with self.assertRaisesRegex(ValueError, "streak escapes current covers"):
            dump_contract.validate_subtitle_contour_tuple(
                {**payloads, "contour_history": history.tobytes()}, **common)
        payloads, common = self._tuple(rectangles=False)
        history = np.frombuffer(payloads["contour_history"], dtype="<f4").reshape(4, 24, 4).copy()
        for code in (6, 10, 14):
            history[1, 11, 3] = np.float32(code)
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, "count0 only"):
                dump_contract.validate_subtitle_contour_tuple(
                    {**payloads, "contour_history": history.tobytes()}, **common)

    def test_color_agnostic_binary_support_has_no_body_kind_or_anchor_requirement(self):
        import numpy as np

        payloads, common = self._tuple()
        source = np.frombuffer(payloads["contour_source"], dtype="<f2").reshape(4, 24, 4).copy()
        for color in ((0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 0), (0.2, 0.1, 0.9)):
            source[1, 12:16, :3] = np.array(color, dtype=np.float16)
            history = np.frombuffer(payloads["contour_history"], dtype="<f4").reshape(4, 24, 4).copy()
            history[..., :3] = source[..., :3].astype(np.float32)
            with self.subTest(color=color):
                result = dump_contract.validate_subtitle_contour_tuple(
                    {**payloads, "contour_source": source.tobytes(), "contour_history": history.tobytes()}, **common)
                self.assertEqual(result["support_pixels"], 3)

    def test_invalid_source_pixel_has_canonical_zero_history_and_valid_signed_zero_is_preserved(self):
        import numpy as np

        payloads, common = self._tuple()
        source = np.frombuffer(payloads["contour_source"], dtype="<f2").reshape(4, 24, 4).copy()
        history = np.frombuffer(payloads["contour_history"], dtype="<f4").reshape(4, 24, 4).copy()
        source[0, 0] = 0
        history[0, 0] = 0
        source[0, 1, :3] = np.array((-0.0, 0.5, 1.0), dtype=np.float16)
        history[0, 1, :3] = source[0, 1, :3].astype(np.float32)
        joined = {**payloads, "contour_source": source.tobytes(), "contour_history": history.tobytes()}
        result = dump_contract.validate_subtitle_contour_tuple(joined, **common)
        self.assertEqual(result["history_rgb_valid_pixels"], 95)
        changed = history.copy()
        changed[0, 0, 3] = -0.0
        with self.assertRaisesRegex(ValueError, "temporal history"):
            dump_contract.validate_subtitle_contour_tuple({**joined, "contour_history": changed.tobytes()}, **common)
        changed = history.copy()
        changed[0, 1, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "temporal history"):
            dump_contract.validate_subtitle_contour_tuple({**joined, "contour_history": changed.tobytes()}, **common)

    def test_globally_invalid_gates_require_bitwise_cleared_snapshots_and_empty_tuple(self):
        for options in ({"ocr_authoritative": False}, {"geometry_ready": False},
                        {"source_width": coordinate.EXPECTED_SUBTITLE_OCR["contour_policy"]["source_max_extent"] + 1}):
            payloads, common = self._tuple(**options)
            result = dump_contract.validate_subtitle_contour_tuple(payloads, **common)
            self.assertFalse(result["authority_valid"])
            self.assertFalse(result["snapshot_gate_valid"])
            if "source_width" in options:
                self.assertEqual((result["native_width"], result["native_height"]), (1, 1))
                self.assertFalse(result["native_source_eligible"])
            for role in ("contour_support", "contour_field_pin", "contour_history", "contour_source", "ocr_probability"):
                changed = dict(payloads)
                # Negative zero also cannot masquerade as a canonical cleared snapshot.
                values = bytearray(changed[role])
                values[-1] = 0x80
                changed[role] = bytes(values)
                with self.subTest(gate=options, role=role), self.assertRaises(ValueError):
                    dump_contract.validate_subtitle_contour_tuple(changed, **common)

    def test_source_alpha_zero_requires_canonical_zero_and_cannot_authorize_support(self):
        import numpy as np

        payloads, common = self._tuple()
        source = np.frombuffer(payloads["contour_source"], dtype="<f2").reshape(4, 24, 4).copy()
        source[1, 12, 3] = np.float16(0)
        changed = {**payloads, "contour_source": source.tobytes()}
        with self.assertRaisesRegex(ValueError, "alpha-zero"):
            dump_contract.validate_subtitle_contour_tuple(changed, **common)
        source[1, 12] = np.float16(0)
        changed["contour_source"] = source.tobytes()
        with self.assertRaisesRegex(ValueError, "valid current source"):
            dump_contract.validate_subtitle_contour_tuple(changed, **common)

    def test_literal_gaussian_no_hardcore_untruncated_tail_and_exact_signed_endpoints(self):
        import numpy as np

        base = np.full((12, 24), np.float32(0.0395))
        q = np.full(base.shape, np.inf, dtype=np.float32)
        q[2, 2] = np.float32(1e8)
        q[5, 10:14] = np.array((0, .01, 1, 64), dtype=np.float32)
        for target_bits in (0, 0x80000000):
            subtitle = {"contour": {"authority_valid": True}, "joint_plane_mode": 3,
                        "tensor_content_rect": (2, 2, 22, 10), "source_width": 120, "source_height": 120,
                        "target": np.asarray(target_bits, dtype=np.uint32).view(np.float32), "active_level": 2}
            original = base.copy()
            result = dump_contract._replay_subtitle_contour_conditioner(base, subtitle, q)
            self.assertEqual(int(result[5, 10].view(np.uint32)), target_bits)
            self.assertGreater(float(result[5, 11]), 0)
            self.assertLess(float(result[5, 11]), float(base[5, 11]))
            weight = dump_contract._subtitle_gaussian_weight_from_q(q, 120, 120)
            self.assertGreater(float(weight[5, 13]), 0)  # exp(-32), no four-sigma cutoff.
            self.assertEqual(int(result[6, 14].view(np.uint32)), int(base[6, 14].view(np.uint32)))
            self.assertEqual(int(weight[2, 2].view(np.uint32)), 0)  # Finite underflow is legal.
            self.assertTrue(np.array_equal(original.view(np.uint32), base.view(np.uint32)))
            self.assertTrue(np.array_equal(result[0].view(np.uint32), result[2].view(np.uint32)))

    def test_warmup_current_cover_gaussian_allows_empty_pixel_q_and_expires_without_promoting_raw_support(self):
        import math
        import numpy as np
        from fractions import Fraction

        payloads, common = self._tuple()
        source = np.frombuffer(payloads["contour_source"], dtype="<f2").reshape(4, 24, 4)
        history = np.zeros((4, 24, 4), dtype="<f4")
        history[..., :3] = source[..., :3].astype(np.float32)
        history[..., 3] = 2
        content = common["tensor_content"]
        rectangle = common["current_rectangles"][0]
        for y in range(4):
            for x in range(24):
                fx = content[0] + Fraction(2 * x + 1, 2) * 5 / 24
                fy = content[1] + (20 + Fraction(2 * y + 1, 2)) * 4 / 24
                if rectangle["left"] <= fx < rectangle["right"] and rectangle["top"] <= fy < rectangle["bottom"]:
                    history[y, x, 3] = 6  # Current first observation, never confirmed support.
        payloads["contour_history"] = history.tobytes()
        payloads["contour_support"] = bytes(4 * 24 * 4)
        payloads["contour_field_pin"] = bytes(7 * 6 * 4)
        payloads["contour_distance"] = np.full((6, 7), np.inf, dtype="<f4").tobytes()
        weight = np.zeros((6, 7), dtype="<f4")
        sigma = 24 / 60
        # Independent continuous field-edge geometry; no temporal mask or detector replica.
        for y in range(6):
            for x in range(7):
                fx = min(max(x, content[0]), content[2] - 1) + .5
                fy = min(max(y, content[1]), content[3] - 1) + .5
                dx = max(rectangle["left"] - fx, 0, fx - rectangle["right"]) * 24 / 5
                dy = max(rectangle["top"] - fy, 0, fy - rectangle["bottom"]) * 24 / 4
                weight[y, x] = math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
        payloads["gaussian_weight"] = weight.tobytes()
        payloads["gaussian_control"] = struct.pack("<I", 1)
        for packed in (1, 2, 3, 4, 9, 10, 11, 12):
            common["plane_state"]["warmup_packed"] = packed
            result = dump_contract.validate_subtitle_contour_tuple(payloads, **common)
            self.assertEqual(result["support_pixels"], 0)
            self.assertEqual(result["field_pin_nodes"], 0)
            self.assertTrue(result["gaussian_smoothing_enabled"])
            self.assertEqual(result["warmup_remaining"], packed & coordinate.SUBTITLE_OCR.plane_warmup_remaining_mask)
        for packed in (0, 8):
            common["plane_state"]["warmup_packed"] = packed
            with self.assertRaisesRegex(ValueError, "smoothing control"):
                dump_contract.validate_subtitle_contour_tuple(payloads, **common)
        common["plane_state"]["warmup_packed"] = 0
        payloads["gaussian_weight"] = bytes(7 * 6 * 4)
        payloads["gaussian_control"] = struct.pack("<I", 0)
        result = dump_contract.validate_subtitle_contour_tuple(payloads, **common)
        self.assertFalse(result["gaussian_smoothing_enabled"])
        self.assertEqual(result["support_pixels"], 0)

    def test_warmup_rejects_unknown_packed_bits_empty_authority_and_invalid_probe(self):
        payloads, common = self._tuple()
        del common["plane_state"]["warmup_packed"]
        with self.assertRaisesRegex(ValueError, "packed warmup"):
            dump_contract.validate_subtitle_contour_tuple(payloads, **common)
        for packed in (5, 6, 7, 13, 14, 15, 16, 0xffffffff):
            common["plane_state"]["warmup_packed"] = packed
            with self.assertRaisesRegex(ValueError, "packed warmup"):
                dump_contract.validate_subtitle_contour_tuple(payloads, **common)
        for context in ({"rectangles": False}, {"invalid_probe": True}, {"ocr_authoritative": False}):
            payloads, common = self._tuple(**context)
            common["plane_state"]["warmup_packed"] = 4
            with self.assertRaisesRegex(ValueError, "packed warmup"):
                dump_contract.validate_subtitle_contour_tuple(payloads, **common)

    def test_gaussian_own_q_exp_and_control_faults_reject_without_changing_temporal_admission(self):
        import numpy as np

        payloads, common = self._tuple()
        result = dump_contract.validate_subtitle_contour_tuple(payloads, **common)
        self.assertTrue(result["squared_source_distance_verified"])
        self.assertTrue(result["gaussian_weight_verified"])
        for role, values in (("contour_distance", (float("nan"), -1, -float("inf"))),
                             ("gaussian_weight", (float("nan"), -1, 1.01)),
                             ("gaussian_blend", (float("inf"), .041)),
                             ("gaussian_vertical_conditioned", (float("nan"), -.041))):
            original = np.frombuffer(payloads[role], dtype="<f4").copy()
            for value in values:
                changed = original.copy(); changed[8] = np.float32(value)
                with self.subTest(role=role, value=value), self.assertRaises(ValueError):
                    dump_contract.validate_subtitle_contour_tuple({**payloads, role: changed.tobytes()}, **common)
        for control in (0, 2, 0xffffffff):
            with self.subTest(control=control), self.assertRaisesRegex(ValueError, "control"):
                dump_contract.validate_subtitle_contour_tuple({**payloads, "gaussian_control": struct.pack("<I", control)}, **common)
        q = np.frombuffer(payloads["contour_distance"], dtype="<f4").reshape(6, 7).copy()
        q[3, 3] += np.float32(.05)
        with self.assertRaisesRegex(ValueError, "IEEE coordinate bound"):
            dump_contract.validate_subtitle_contour_tuple({**payloads, "contour_distance": q.tobytes()}, **common)
        q = np.frombuffer(payloads["contour_distance"], dtype="<f4").reshape(6, 7).copy()
        q[0, 0] = np.float32(0)
        with self.assertRaisesRegex(ValueError, "empty/padding"):
            dump_contract.validate_subtitle_contour_tuple({**payloads, "contour_distance": q.tobytes()}, **common)

    def test_actual_saved_blend_c75_final_replay_does_not_repin_after_shared_conditioning(self):
        import numpy as np

        payloads, _ = self._tuple()
        base = np.full((12, 24), np.float32(.0395))
        m = np.zeros_like(base); m[5, 10] = np.float32(1); m[5, 11] = np.float32(.5)
        subtitle = {"contour": {"authority_valid": True, "gaussian_smoothing_enabled": True},
                    "joint_plane_mode": 3, "tensor_content_rect": (0, 0, 24, 12),
                    "source_width": 120, "source_height": 120, "target": np.float32(0), "active_level": 2}
        proposal = dump_contract._subtitle_gaussian_literal_blend(base, m, subtitle["target"])
        _, vertical, final = dump_contract._replay_v2_limiter_fields(proposal, 24)
        payloads.update(gaussian_weight=m.tobytes(), gaussian_blend=proposal.tobytes(),
            gaussian_vertical_conditioned=vertical.tobytes(), gaussian_control=struct.pack("<I", 1))
        result = dump_contract.validate_subtitle_gaussian_fields(payloads, base, subtitle, final_field=final)
        self.assertTrue(result["gaussian_final_verified"])
        self.assertGreater(float(final[5, 10]), 0)  # A post-smoothing center need not stay target.
        for role in ("gaussian_blend", "gaussian_vertical_conditioned"):
            changed = np.frombuffer(payloads[role], dtype="<f4").copy()
            changed[5 * 24 + 10] += np.float32(.001)
            with self.subTest(role=role), self.assertRaises(ValueError):
                dump_contract.validate_subtitle_gaussian_fields({**payloads, role: changed.tobytes()}, base, subtitle, final_field=final)
        changed = final.copy(); changed[5, 10] = np.float32(0)
        with self.assertRaisesRegex(ValueError, "Final"):
            dump_contract.validate_subtitle_gaussian_fields(payloads, base, subtitle, final_field=changed)

    def test_gaussian_subnormal_interior_ftz_is_bounded_and_cannot_waive_endpoint_or_zero_sign(self):
        import numpy as np

        payloads, _ = self._tuple()
        base = np.zeros((6, 7), dtype=np.float32)
        m = np.full_like(base, np.float32(1e-36))
        target = np.float32(.01)
        literal = dump_contract._subtitle_gaussian_literal_blend(base, m, target)
        self.assertTrue(np.all((literal > 0) & (literal < np.finfo(np.float32).tiny)))
        subtitle = {"contour": {"authority_valid": True, "gaussian_smoothing_enabled": True},
                    "joint_plane_mode": 3, "tensor_content_rect": (0, 0, 7, 6),
                    "source_width": 120, "source_height": 120, "target": target, "active_level": 2}
        zero = np.zeros_like(base)
        payloads.update(gaussian_weight=m.tobytes(), gaussian_blend=zero.tobytes(),
            gaussian_vertical_conditioned=zero.tobytes(), gaussian_control=struct.pack("<I", 1))
        self.assertTrue(dump_contract.validate_subtitle_gaussian_fields(payloads, base, subtitle,
            final_field=zero)["gaussian_final_verified"])
        negative = zero.copy(); negative.view(np.uint32)[2, 2] = 0x80000000
        with self.assertRaisesRegex(ValueError, "arithmetic"):
            dump_contract.validate_subtitle_gaussian_fields({**payloads, "gaussian_blend": negative.tobytes()},
                base, subtitle, final_field=zero)
        wrong = zero.copy(); wrong[2, 2] = np.float32(1e-30)
        with self.assertRaisesRegex(ValueError, "arithmetic"):
            dump_contract.validate_subtitle_gaussian_fields({**payloads, "gaussian_blend": wrong.tobytes()},
                base, subtitle, final_field=zero)
        m[2, 2] = np.float32(1)
        with self.assertRaisesRegex(ValueError, "endpoints"):
            dump_contract.validate_subtitle_gaussian_fields({**payloads, "gaussian_weight": m.tobytes()},
                base, subtitle, final_field=zero)

    def test_disabled_gaussian_keeps_base_bits_and_nearest_padding_without_requantization(self):
        import numpy as np

        payloads, _ = self._tuple(rectangles=False)
        base = np.full((6, 7), np.asarray(0x33000001, dtype=np.uint32).view(np.float32))
        base[2, 2] = np.float32(-0.0)
        base[0] = np.float32(.03)
        subtitle = {"contour": {"authority_valid": False, "gaussian_smoothing_enabled": False},
                    "tensor_content_rect": (1, 1, 6, 5)}
        joined, _ = dump_contract._subtitle_gaussian_joined_base(base, subtitle)
        payloads.update(gaussian_blend=joined.tobytes(), gaussian_vertical_conditioned=joined.tobytes())
        result = dump_contract.validate_subtitle_gaussian_fields(payloads, base, subtitle, final_field=joined)
        self.assertFalse(result["gaussian_smoothing_enabled"])
        for role in ("gaussian_blend", "gaussian_vertical_conditioned"):
            changed = np.frombuffer(payloads[role], dtype="<u4").copy(); changed[2 * 7 + 2] = 0
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "exact Base"):
                dump_contract.validate_subtitle_gaussian_fields({**payloads, role: changed.tobytes()}, base, subtitle, final_field=joined)


class WindowRegionDumpContractTests(unittest.TestCase):
    def setUp(self):
        self.border = {
            "schema": dump_contract.WINDOW_REGION_SCHEMA,
            "capture":
                "same matched source/color/depth/render frame as the parent Dump 3D package",
            "role":
                "matched-window region provenance; no independent geometry or renderer authority",
            "authority_kind": "chromium-video",
            "matched_frame_id": 41,
            "coordinate_space": {
                "name": "matched-source-pixels",
                "rect_semantics": "half-open [left, top, right, bottom)",
                "source_extent_px": {"width": 3840, "height": 2160},
                "capture_rect_px": {
                    "left": 320, "top": 180, "right": 3520, "bottom": 1980,
                },
            },
            "identity": {
                "hwnd": "0x1234",
                "process_id": 55,
                "document_id": -7,
                "video_id": -9,
                "generation": 3,
            },
            "freshness": {
                "latest_observation_age_ms_at_capture": 120,
                "maximum_observation_age_ms": 1000,
                "geometry_continuity_ms_at_capture": 5000,
                "source_content_age_ms_at_capture": 1000,
                "fresh": True,
                "causal_geometry": True,
            },
        }

    def test_valid_border_is_bound_to_parent_frame_and_source_extent(self):
        decoded = dump_contract.validate_window_region_document(
            self.border, matched_frame_id=41, source_width=3840, source_height=2160)
        self.assertEqual(decoded["authority_kind"], "chromium-video")
        self.assertEqual(decoded["left"], 320)
        self.assertEqual(decoded["right"], 3520)
        self.assertEqual(decoded["hwnd"], 0x1234)

    def test_frame_extent_and_half_open_bounds_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "parent frame"):
            dump_contract.validate_window_region_document(
                self.border, matched_frame_id=40, source_width=3840, source_height=2160)
        with self.assertRaisesRegex(ValueError, "source extent"):
            dump_contract.validate_window_region_document(
                self.border, matched_frame_id=41, source_width=1920, source_height=1080)
        changed = copy.deepcopy(self.border)
        changed["coordinate_space"]["capture_rect_px"]["right"] = 3841
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            dump_contract.validate_window_region_document(changed)

    def test_stale_or_incomplete_identity_fails_closed(self):
        changed = copy.deepcopy(self.border)
        changed["identity"]["video_id"] = 0
        with self.assertRaisesRegex(ValueError, "incomplete Chromium identity"):
            dump_contract.validate_window_region_document(changed)
        changed = copy.deepcopy(self.border)
        changed["freshness"]["latest_observation_age_ms_at_capture"] = 1001
        with self.assertRaisesRegex(ValueError, "stale"):
            dump_contract.validate_window_region_document(changed)
        changed = copy.deepcopy(self.border)
        changed["freshness"]["geometry_continuity_ms_at_capture"] = 999
        with self.assertRaisesRegex(ValueError, "postdates"):
            dump_contract.validate_window_region_document(changed)

    def test_unknown_fields_and_authority_claims_are_rejected(self):
        changed = copy.deepcopy(self.border)
        changed["normalized_rect"] = [0.0, 0.0, 1.0, 1.0]
        with self.assertRaisesRegex(ValueError, "unknown layout"):
            dump_contract.validate_window_region_document(changed)
        changed = copy.deepcopy(self.border)
        changed["role"] = "renderer authority"
        with self.assertRaisesRegex(ValueError, "unknown authority"):
            dump_contract.validate_window_region_document(changed)

    def test_foreground_authority_requires_native_identity_without_dom_ids(self):
        foreground = copy.deepcopy(self.border)
        foreground["authority_kind"] = "foreground-client"
        foreground["identity"]["document_id"] = 0
        foreground["identity"]["video_id"] = 0
        decoded = dump_contract.validate_window_region_document(foreground)
        self.assertEqual(decoded["authority_kind"], "foreground-client")
        self.assertEqual(decoded["document_id"], 0)
        self.assertEqual(decoded["video_id"], 0)

        foreground["identity"]["document_id"] = -7
        with self.assertRaisesRegex(ValueError, "foreground authority carries DOM identity"):
            dump_contract.validate_window_region_document(foreground)

        unknown = copy.deepcopy(self.border)
        unknown["authority_kind"] = "desktop"
        with self.assertRaisesRegex(ValueError, "unknown authority kind"):
            dump_contract.validate_window_region_document(unknown)


if __name__ == "__main__":
    unittest.main()
