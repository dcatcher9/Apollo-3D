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
        checksum = dump_contract.camera_center_integrity_bits(
            document["named_values"]["center"],
            document["named_values"]["inverse_scale"],
            document["named_values"]["convergence_curve"],
            document["named_values"]["calibration_revision"],
        )
        cls._set_state_word(document, "camera_center_integrity_bits", checksum)
        document["decoded"]["camera_center_integrity_bits"] = checksum
        authorization = (document["decoded"]["contract_tag"]
                         if document["named_values"]["frame_valid"] > 0.5 else 0)
        cls._set_state_word(document, "renderer_authorization_bits", authorization)
        document["decoded"]["renderer_authorization_bits"] = authorization

    def setUp(self):
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
            "mapping_state_reserved_1": 0,
            "mapping_state_reserved_2": 0,
        }
        values["camera_center_integrity_bits"] = (
            dump_contract.camera_center_integrity_bits(
                values["center"], values["inverse_scale"],
                values["convergence_curve"],
                values["calibration_revision"]))
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
            "far_tau": defaults.far_tau,
            "near_log_tau": defaults.near_log_tau,
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
                "gain": "one-eye source-U per curve unit",
                "parallax": "signed one-eye source-U",
            },
            "constants": constants,
            "fields": fields,
            "named_values": values,
            "decoded": {
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
            "adaptation_semantics": {
                "coordinate": (
                    "immediate-first-usable-center-latched-until-cut-fixed-"
                    "authenticated-scale-retained-across-unusable"
                ),
                "convergence_curve":
                    "arithmetic-mean-center-is-zero-plane",
                "requested_gain": "immutable-cfg-pop-strength",
                "container_scale":
                    "abi-retained-identity-pointwise-soft-container-is-map-local",
                "near_curve":
                    "fixed-contract-logarithmic-tau-independent-of-content-occupancy",
                "spatial_conditioner":
                    "fixed-75pct-vertical-majorant-share-then-horizontal-majorant",
            },
        }
        self.live_state = copy.deepcopy(self.state)
        self.live_state["rendered_output_selected"] = True
        self.live_state["wire_contract"] = (
            "authenticated live Host-SBS renderer input; not a client wire contract")
        self.live_state["units"] = {
            "coordinate": "dimensionless canonical coordinate derived from raw depth",
            "gain": "one-eye full-source-U per curve unit",
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
            "shadow_ownership_refined_parallax.f32":
                ("parallax-v2 full-resolution contour ownership refinement", True),
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
                "parallax_v2_ownership_refined_role":
                    "conservative full-resolution source-contour foreground ownership applied "
                    "to candidate before the vertical conditioner; may only raise uniquely "
                    "owned far-side boundary texels",
                "parallax_v2_vertical_majorant_role":
                    "least column-wise upper envelope v+ >= ownership-refined candidate with "
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
                "shadow_ownership_refined_parallax": {
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
            "center", "inverse_scale", "convergence_curve", "container_scale",
            "calibration_revision", "frame_valid", "confirmed_cut_count", "contract_tag_bits",
            "camera_center_integrity_bits", "renderer_authorization_bits",
            "mapping_state_reserved_1", "mapping_state_reserved_2",
        })
        stats = dump_contract.validate_shadow_frame_stats_document(self.frame_stats)
        self.assertEqual(stats["valid"], 1.0)

    def test_current_manifest_attributes_ownership_then_vertical_share_and_row_majorant(self):
        decoded = dump_contract.validate_v2_dump_manifest_document(self.manifest)
        self.assertTrue(decoded["active"])
        self.assertTrue(decoded["rendered_output_selected"])
        self.assertTrue(decoded["ownership_refined_available"])
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

    def test_current_subtitle_record_and_state_abi_partition_exact_word_counts(self):
        self.assertEqual(dump_contract.DUMP_MANIFEST_SCHEMA, 39)
        self.assertEqual(dump_contract.DEPTH_INPUT_REGION_SCHEMA, 4)
        self.assertEqual(dump_contract.SUBTITLE_OCR_RECORD_SCHEMA, 3)
        self.assertEqual(dump_contract.SUBTITLE_LOCATOR_STATE_SCHEMA, 13)
        self.assertEqual(
            dump_contract.SUBTITLE_OCR_RAW_BOX_WORD_OFFSET,
            dump_contract.SUBTITLE_OCR_HEADER_WORD_COUNT)
        self.assertEqual(
            dump_contract.SUBTITLE_OCR_FINAL_BOX_WORD_OFFSET,
            dump_contract.SUBTITLE_OCR_RAW_BOX_WORD_OFFSET +
            dump_contract.SUBTITLE_OCR_RAW_BOX_CAPACITY *
            dump_contract.SUBTITLE_OCR_BOX_WORD_COUNT)
        self.assertEqual(
            dump_contract.SUBTITLE_OCR_RECORD_WORD_COUNT,
            dump_contract.SUBTITLE_OCR_FINAL_BOX_WORD_OFFSET +
            dump_contract.SUBTITLE_OCR_FINAL_BOX_CAPACITY *
            dump_contract.SUBTITLE_OCR_BOX_WORD_COUNT)
        self.assertEqual(
            struct.pack("<I", dump_contract.SUBTITLE_OCR_RECORD_TAG), b"OCR8")

        rectangle_words = 4 * dump_contract.SUBTITLE_LOCATOR_RECT_CAPACITY
        self.assertEqual(
            dump_contract.SUBTITLE_LOCATOR_OWNER_WORD_OFFSET,
            dump_contract.SUBTITLE_LOCATOR_HEADER_WORD_COUNT)
        self.assertEqual(
            dump_contract.SUBTITLE_LOCATOR_PENDING_WORD_OFFSET,
            dump_contract.SUBTITLE_LOCATOR_OWNER_WORD_OFFSET + rectangle_words)
        self.assertEqual(
            dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET,
            dump_contract.SUBTITLE_LOCATOR_PENDING_WORD_OFFSET + rectangle_words)
        self.assertEqual(
            dump_contract.SUBTITLE_LOCATOR_STATE_WORD_COUNT,
            dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET + rectangle_words)
        self.assertEqual(
            struct.pack("<I", dump_contract.SUBTITLE_LOCATOR_STATE_TAG), b"SL13")

    def test_subtitle_selection_mirrors_strict_symmetric_corner_filter(self):
        policy = coordinate.SUBTITLE_OCR
        content = (0, 0, 770, 434)
        roi_bottom = 430
        edge = (content[2] - content[0]) // policy.locator_corner_edge_divisor
        bottom = roi_bottom - policy.locator_corner_bottom_rows
        self.assertEqual(edge, 24)

        def core(left, right, core_bottom=bottom, kind="text"):
            return {
                "left": left,
                "top": core_bottom - 10,
                "right": right,
                "bottom": core_bottom,
                "kind": kind,
            }

        self.assertFalse(dump_contract._subtitle_qualified_ocr_core(
            core(edge - 1, edge + 115), content, roi_bottom))
        self.assertTrue(dump_contract._subtitle_qualified_ocr_core(
            core(edge, edge + 116), content, roi_bottom))
        self.assertFalse(dump_contract._subtitle_qualified_ocr_core(
            core(content[2] - edge - 115, content[2] - edge + 1),
            content, roi_bottom))
        self.assertTrue(dump_contract._subtitle_qualified_ocr_core(
            core(content[2] - edge - 116, content[2] - edge),
            content, roi_bottom))
        self.assertTrue(dump_contract._subtitle_qualified_ocr_core(
            core(edge - 1, edge + 115, bottom - 1), content, roi_bottom))
        self.assertTrue(dump_contract._subtitle_qualified_ocr_core(
            core(0, 700, roi_bottom, "ribbon"), content, roi_bottom))

        # Clearance is relative to the authenticated content rectangle, not tensor x=0.
        offset_content = (111, 0, 659, 434)
        offset_edge = ((offset_content[2] - offset_content[0]) //
                       policy.locator_corner_edge_divisor)
        self.assertEqual(offset_edge, 17)
        self.assertFalse(dump_contract._subtitle_qualified_ocr_core(
            core(offset_content[0] + offset_edge - 1,
                 offset_content[0] + offset_edge + 115),
            offset_content, roi_bottom))
        self.assertTrue(dump_contract._subtitle_qualified_ocr_core(
            core(offset_content[2] - offset_edge - 116,
                 offset_content[2] - offset_edge),
            offset_content, roi_bottom))

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

    def _active_slr13_manifest(self, base_manifest=None):
        manifest = copy.deepcopy(
            self.manifest if base_manifest is None else base_manifest)
        subtitle = {
            "mode": "subtitle-slr13",
            "request": True,
            "producer": dump_contract._subtitle_ocr_producer_contract(),
            "resolver": dump_contract._subtitle_locator_resolver_contract(),
            "artifacts": {
                "ocr_record": "subtitle_ocr_record.u32",
                "locator_state": "subtitle_locator_state.u32",
                "base_field": "shadow_base_final_parallax.f32",
                "conditioned_field": "shadow_final_parallax.f32",
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
                "stage": "compact SLR13 subtitle authority state",
                "description": "authenticated test SLR13 state",
                "sha256": "0" * 64,
            },
            "shadow_base_final_parallax.f32": {
                "available": True,
                "required": True,
                "stage": "ordinary post-limiter V2 field before SLR13 conditioning",
                "description": "authenticated test unconditioned base field",
                "sha256": "0" * 64,
            },
        })
        manifest["renderer"]["parallax_v2_conditioner_role"] = (
            "least row-wise q >= shadow_vertical_conditioned with horizontal slope <= "
            "max_horizontal_slope and vertical shear <= max_vertical_shear produces "
            "shadow_base_final_parallax; SLR13 publishes shadow_final_parallax atomically as "
            "direct live authority")
        payload = json.dumps(subtitle).encode("utf-8")
        manifest["artifacts"]["subtitle_conditioning.json"]["sha256"] = (
            hashlib.sha256(payload).hexdigest())
        return manifest

    def test_current_slr13_manifest_binds_exact_model_shader_and_artifact_roles(self):
        manifest = self._active_slr13_manifest()
        decoded = dump_contract.validate_v2_dump_manifest_document(manifest)
        subtitle = decoded["subtitle_conditioning"]
        self.assertEqual(subtitle["mode"], "subtitle-slr13")
        self.assertTrue(subtitle["live"])
        self.assertTrue(subtitle["subtitle_evidence_complete"])
        self.assertEqual(
            subtitle["artifact_files"]["conditioned_field"],
            "shadow_final_parallax.f32")
        self.assertEqual(
            manifest["artifacts"]["shadow_base_final_parallax.f32"]["stage"],
            "ordinary post-limiter V2 field before SLR13 conditioning")
        producer = manifest["subtitle_conditioning"]["producer"]
        self.assertEqual(producer["contract_schema"], coordinate.SUBTITLE_OCR.schema)
        self.assertEqual(set(producer["model"]), {
            "name", "asset_path", "artifact_onnx_sha256", "source_url",
            "source_onnx_sha256", "conversion_tool", "conversion_version",
            "conversion_recipe", "conversion_calibration_profile", "engine_recipe",
            "preprocess_profile", "source_crop", "input", "output",
        })
        self.assertEqual(
            manifest["subtitle_conditioning"]["resolver"]["qualification_policy"],
            {
                "corner_filter_applies_to": "non-ribbon-ordinary-cores",
                "corner_edge_clearance": (
                    "strictly-less-than-floor-content-width-over-divisor"),
                "corner_edge_divisor": 32,
                "corner_bottom": "at-or-below-dynamic-roi-bottom-minus-rows",
                "corner_bottom_rows": 16,
                "edge_threshold_equality": "accepted",
                "ribbon_exempt": True,
            })
        self.assertEqual(
            manifest["subtitle_conditioning"]["resolver"]["target_policy"],
            {
                "units": "binocular-source-pixels",
                "placement": {
                    "primary": "aggregate-owner-median-member-center",
                    "fallback_on_primary_failure": True,
                    "fallback_span": (
                        "ordinary-core-horizontal-bounds-else-owner-core-horizontal-bounds"),
                    "fallback_top": "ordinary-core-top-else-owner-core-top",
                    "fallback_step_denominator": 16,
                    "fallback_max_radius_steps": 2,
                    "fallback_order_within_radius": ["negative", "positive"],
                    "fallback_radius_policy": "first-reliable-radius",
                    "fallback_requires_unclamped_sample_strip": True,
                    "fallback_minimum_coherent_rows": 2,
                    "fallback_row_median_delta_max": 4.0,
                    "fallback_probe_target": "mean-medians",
                    "fallback_pair_target_delta_max": 4.0,
                    "fallback_pair_conflict": "unreliable-stop-search",
                    "fallback_within_radius_policy": "maximum-mean-within-delta",
                    "ribbon_places_fallback_with_ordinary": False,
                },
                "selection": {
                    "applies_to": "primary",
                    "samples_per_row": 16,
                    "median_indices": [7, 8],
                    "iqr_lower_indices": [3, 4],
                    "iqr_upper_indices": [11, 12],
                    "row_validity": "independent-finite-direct-container",
                    "both_valid_row_iqr": "ignored",
                    "single_valid_row": "median-if-iqr-at-most-row-iqr-max",
                    "both_valid_within_delta": "mean-medians",
                    "both_valid_beyond_delta": "maximum-median",
                },
                "evidence": {
                    "row_iqr_max": 8.0,
                    "row_median_delta_max": 4.0,
                },
                "deadband": 1.0,
                "ema_alpha": 0.125,
                "maximum_slew": 0.25,
                "maximum_residual": 8.0,
                "unreliable_hold": {
                    "owner_state_word": 25,
                    "maximum_distinct_observations": 2,
                    "increment_requires": (
                        "continuing-same-scene-owner-current-authority-valid-target"),
                    "preserve_without_current_authority": True,
                    "duplicate_observation_ages": False,
                    "hard_cut_allowed": False,
                },
                "representation_limit": "direct-parallax-container",
            })
        self.assertEqual(
            [
                source["entrypoint"] for source in
                manifest["subtitle_conditioning"]["resolver"]
                ["shader_contract"]["source_specs"]
            ],
            ["resolve_main", "condition_main"],
        )

    def test_current_slr13_manifest_rejects_provenance_roles_and_base_field_drift(self):
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
                    "target_policy"]["evidence"].update(
                        {"row_median_delta_max": 3.999}),
                "resolver provenance"),
            "qualification-policy": (
                lambda manifest: manifest["subtitle_conditioning"]["resolver"][
                    "qualification_policy"].update({"corner_edge_divisor": 31}),
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
                    {"locator_state": "wrong.u32"}),
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
                manifest = self._active_slr13_manifest()
                mutate(manifest)
                with self.assertRaisesRegex(ValueError, error):
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

    def test_current_ocr8_slr13_empty_records_accept_all_calibrated_fields(self):
        cases = (
            (1920, 1080, 770, 434),
            (2560, 1080, 1022, 434),
            (3440, 1440, 1036, 434),
            (1080, 1920, 434, 770),
            (1080, 2560, 434, 1022),
            (1440, 3440, 434, 1036),
        )
        for source_width, source_height, field_width, field_height in cases:
            with self.subTest(field=(field_width, field_height)):
                roi = coordinate.subtitle_ocr_dynamic_roi(
                    source_width, source_height, field_width, field_height)
                self.assertIsNotNone(roi)
                roi_top, roi_bottom = roi
                ocr = [0] * dump_contract.SUBTITLE_OCR_RECORD_WORD_COUNT
                ocr[:16] = [
                    dump_contract.SUBTITLE_OCR_RECORD_SCHEMA,
                    dump_contract.SUBTITLE_OCR_RECORD_TAG,
                    1, 0, 0,
                    41, 0,
                    17, 0,
                    source_width, source_height,
                    field_width, field_height,
                    roi_top, roi_bottom,
                    0,
                ]
                decoded_ocr = dump_contract.validate_subtitle_ocr_record(
                    self._pack_uint32_words(ocr),
                    matched_frame_id=41,
                    analysis_generation=17,
                    source_width=source_width,
                    source_height=source_height,
                    field_width=field_width,
                    field_height=field_height,
                    roi_top=roi_top,
                    roi_bottom=roi_bottom)
                self.assertEqual(
                    (decoded_ocr["field_width"], decoded_ocr["field_height"]),
                    (field_width, field_height))

                locator = [0] * dump_contract.SUBTITLE_LOCATOR_STATE_WORD_COUNT
                locator[:32] = [
                    dump_contract.SUBTITLE_LOCATOR_STATE_SCHEMA,
                    dump_contract.SUBTITLE_LOCATOR_STATE_TAG,
                    0, 0, 0,
                    0, 0, 0, 0, 0,
                    17, 0,
                    0,
                    0, 0, 0, 0, 0,
                    0, 0,
                    0,
                    dump_contract.SUBTITLE_LOCATOR_EVENT_NONE,
                    41, 0,
                    0, 0, 0,
                    field_width, field_height,
                    0, 0, 0,
                ]
                decoded_locator = dump_contract.validate_subtitle_locator_state(
                    self._pack_uint32_words(locator),
                    matched_frame_id=41,
                    analysis_generation=17,
                    source_width=source_width,
                    source_height=source_height,
                    field_width=field_width,
                    field_height=field_height)
                self.assertEqual(decoded_locator["current_rectangles"], [])

    def test_ocr8_and_slr13_project_and_confine_geometry_to_tensor_content(self):
        source_width, source_height = 400, 1200
        field_width, field_height = 770, 434
        content = (313, 0, 457, 434)
        roi = dump_contract._subtitle_dynamic_roi(
            source_width, source_height, content)
        minimum_bottom = dump_contract._subtitle_ribbon_min_bottom(
            source_width, source_height, content)
        self.assertEqual(roi, (414, 434))
        self.assertEqual(minimum_bottom, 433)

        score_bits = struct.unpack("<I", struct.pack("<f", 0.875))[0]
        core = [323, 420, 447, 433, score_bits,
                dump_contract.SUBTITLE_OCR_BOX_FLAG_RIBBON, 7, 4]
        cover = [313, 418, 457, 434, score_bits,
                 dump_contract.SUBTITLE_OCR_BOX_FLAG_RIBBON, 7, 4]
        ocr = [0] * dump_contract.SUBTITLE_OCR_RECORD_WORD_COUNT
        ocr[:16] = [
            dump_contract.SUBTITLE_OCR_RECORD_SCHEMA,
            dump_contract.SUBTITLE_OCR_RECORD_TAG,
            1, 1, 1,
            41, 0,
            17, 0,
            source_width, source_height,
            field_width, field_height,
            roi[0], roi[1],
            0,
        ]
        raw_offset = dump_contract.SUBTITLE_OCR_RAW_BOX_WORD_OFFSET
        final_offset = dump_contract.SUBTITLE_OCR_FINAL_BOX_WORD_OFFSET
        ocr[raw_offset:raw_offset + 8] = core
        ocr[final_offset:final_offset + 8] = cover
        decoded_ocr = dump_contract.validate_subtitle_ocr_record(
            self._pack_uint32_words(ocr),
            matched_frame_id=41,
            analysis_generation=17,
            source_width=source_width,
            source_height=source_height,
            field_width=field_width,
            field_height=field_height,
            roi_top=roi[0],
            roi_bottom=roi[1],
            tensor_content=content)
        self.assertEqual(decoded_ocr["tensor_content_rect"], content)
        self.assertEqual(
            tuple(decoded_ocr["final_boxes"][0][key]
                  for key in ("left", "top", "right", "bottom")),
            (313, 418, 457, 434))

        target_bits = struct.unpack("<I", struct.pack("<f", 0.00075))[0]
        locator = [0] * dump_contract.SUBTITLE_LOCATOR_STATE_WORD_COUNT
        locator[:32] = [
            dump_contract.SUBTITLE_LOCATOR_STATE_SCHEMA,
            dump_contract.SUBTITLE_LOCATOR_STATE_TAG,
            (dump_contract.SUBTITLE_LOCATOR_FLAG_OWNER |
             dump_contract.SUBTITLE_LOCATOR_FLAG_TARGET_VALID),
            9, 1,
            323, 420, 447, 433, 124 * 13,
            17, 0,
            0,
            0, 0, 0, 0, 0,
            target_bits, 9,
            1,
            dump_contract.SUBTITLE_LOCATOR_EVENT_BIRTH,
            41, 0,
            1, 0, 3,
            field_width, field_height,
            0, 0,
            ((1 << dump_contract.SUBTITLE_LOCATOR_OWNER_KIND_SHIFT) |
             (1 << dump_contract.SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT)),
        ]
        owner_offset = dump_contract.SUBTITLE_LOCATOR_OWNER_WORD_OFFSET
        current_offset = dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET
        locator[owner_offset:owner_offset + 4] = core[:4]
        locator[current_offset:current_offset + 4] = cover[:4]
        decoded_locator = dump_contract.validate_subtitle_locator_state(
            self._pack_uint32_words(locator),
            matched_frame_id=41,
            analysis_generation=17,
            source_width=source_width,
            source_height=source_height,
            field_width=field_width,
            field_height=field_height,
            tensor_content=content)
        self.assertEqual(decoded_locator["tensor_content_rect"], content)

        escaped = ocr.copy()
        escaped[final_offset] = 0
        with self.assertRaisesRegex(ValueError, "inside the field|canonical bottom strip"):
            dump_contract.validate_subtitle_ocr_record(
                self._pack_uint32_words(escaped),
                matched_frame_id=41, analysis_generation=17,
                source_width=source_width, source_height=source_height,
                field_width=field_width, field_height=field_height,
                roi_top=roi[0], roi_bottom=roi[1], tensor_content=content)

    def test_ribbon_bottom_tolerance_boundary_is_shared_by_strict_ocr_and_slr_readers(self):
        source_width, source_height = 2560, 1080
        field_width, field_height = 1022, 434
        roi_top, roi_bottom = coordinate.subtitle_ocr_dynamic_roi(
            source_width, source_height, field_width, field_height)
        minimum_bottom = coordinate.subtitle_ocr_ribbon_min_bottom(
            source_width, source_height, field_width, field_height)
        self.assertEqual((roi_top, minimum_bottom, roi_bottom), (289, 427, 429))

        score_bits = struct.unpack("<I", struct.pack("<f", 0.875))[0]

        def ocr_record(core_bottom):
            words = [0] * dump_contract.SUBTITLE_OCR_RECORD_WORD_COUNT
            words[:16] = [
                dump_contract.SUBTITLE_OCR_RECORD_SCHEMA,
                dump_contract.SUBTITLE_OCR_RECORD_TAG,
                1, 1, 1,
                41, 0,
                17, 0,
                source_width, source_height,
                field_width, field_height,
                roi_top, roi_bottom,
                0,
            ]
            core = [
                100, 417, 900, core_bottom, score_bits,
                dump_contract.SUBTITLE_OCR_BOX_FLAG_RIBBON, 7, 4,
            ]
            cover = [
                0, 413, field_width, field_height, score_bits,
                dump_contract.SUBTITLE_OCR_BOX_FLAG_RIBBON, 7, 4,
            ]
            raw = dump_contract.SUBTITLE_OCR_RAW_BOX_WORD_OFFSET
            final = dump_contract.SUBTITLE_OCR_FINAL_BOX_WORD_OFFSET
            words[raw:raw + len(core)] = core
            words[final:final + len(cover)] = cover
            return words

        arguments = {
            "matched_frame_id": 41,
            "analysis_generation": 17,
            "source_width": source_width,
            "source_height": source_height,
            "field_width": field_width,
            "field_height": field_height,
            "roi_top": roi_top,
            "roi_bottom": roi_bottom,
        }
        decoded = dump_contract.validate_subtitle_ocr_record(
            self._pack_uint32_words(ocr_record(minimum_bottom)), **arguments)
        self.assertEqual(decoded["raw_boxes"][0]["bottom"], 427)
        with self.assertRaisesRegex(ValueError, "projected bottom tolerance"):
            dump_contract.validate_subtitle_ocr_record(
                self._pack_uint32_words(ocr_record(minimum_bottom - 1)), **arguments)

        target_bits = struct.unpack("<I", struct.pack("<f", 0.00075))[0]

        def slr_state(core_bottom):
            words = [0] * dump_contract.SUBTITLE_LOCATOR_STATE_WORD_COUNT
            core_height = core_bottom - 417
            words[:32] = [
                dump_contract.SUBTITLE_LOCATOR_STATE_SCHEMA,
                dump_contract.SUBTITLE_LOCATOR_STATE_TAG,
                (dump_contract.SUBTITLE_LOCATOR_FLAG_OWNER |
                 dump_contract.SUBTITLE_LOCATOR_FLAG_TARGET_VALID),
                9,
                1,
                100, 417, 900, core_bottom,
                800 * core_height,
                17, 0,
                0,
                0, 0, 0, 0, 0,
                target_bits,
                9,
                1,
                dump_contract.SUBTITLE_LOCATOR_EVENT_BIRTH,
                41, 0,
                1,
                0,
                3,
                field_width, field_height,
                0, 0,
                ((1 << dump_contract.SUBTITLE_LOCATOR_OWNER_KIND_SHIFT) |
                 (1 << dump_contract.SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT)),
            ]
            owner = dump_contract.SUBTITLE_LOCATOR_OWNER_WORD_OFFSET
            current = dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET
            words[owner:owner + 4] = [100, 417, 900, core_bottom]
            words[current:current + 4] = [0, 413, field_width, field_height]
            return words

        locator_arguments = {
            key: arguments[key] for key in (
                "matched_frame_id", "analysis_generation", "source_width", "source_height",
                "field_width", "field_height")
        }
        decoded = dump_contract.validate_subtitle_locator_state(
            self._pack_uint32_words(slr_state(minimum_bottom)), **locator_arguments)
        self.assertEqual(decoded["owner_rectangles"][0]["bottom"], 427)
        with self.assertRaisesRegex(ValueError, "projected bottom tolerance"):
            dump_contract.validate_subtitle_locator_state(
                self._pack_uint32_words(slr_state(minimum_bottom - 1)), **locator_arguments)

    def test_slr13_provisional_state_authenticates_geometry_and_ephemeral_words(self):
        owner = [203, 369, 566, 384]
        pending = [342, 368, 415, 386]
        cover = [337, 364, 420, 390]
        target_bits = struct.unpack("<I", struct.pack("<f", 2.0 / (2.0 * 1920.0)))[0]
        provisional_bits = struct.unpack("<I", struct.pack("<f", 6.0 / (2.0 * 1920.0)))[0]
        words = [0] * dump_contract.SUBTITLE_LOCATOR_STATE_WORD_COUNT
        words[:32] = [
            dump_contract.SUBTITLE_LOCATOR_STATE_SCHEMA,
            dump_contract.SUBTITLE_LOCATOR_STATE_TAG,
            (dump_contract.SUBTITLE_LOCATOR_FLAG_OWNER |
             dump_contract.SUBTITLE_LOCATOR_FLAG_PENDING |
             dump_contract.SUBTITLE_LOCATOR_FLAG_TARGET_VALID |
             dump_contract.SUBTITLE_LOCATOR_FLAG_PROVISIONAL_CURRENT),
            9, 1, *owner, (owner[2] - owner[0]) * (owner[3] - owner[1]),
            17, 0, 1, *pending,
            (pending[2] - pending[0]) * (pending[3] - pending[1]),
            target_bits, 9, 1, dump_contract.SUBTITLE_LOCATOR_EVENT_NONE,
            41, 0, 2, 0, 3, 770, 434, provisional_bits, 2, 0,
        ]
        words[dump_contract.SUBTITLE_LOCATOR_OWNER_WORD_OFFSET:
              dump_contract.SUBTITLE_LOCATOR_OWNER_WORD_OFFSET + 4] = owner
        words[dump_contract.SUBTITLE_LOCATOR_PENDING_WORD_OFFSET:
              dump_contract.SUBTITLE_LOCATOR_PENDING_WORD_OFFSET + 4] = pending
        words[dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET:
              dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET + 4] = cover
        arguments = {
            "matched_frame_id": 41, "analysis_generation": 17,
            "source_width": 1920, "source_height": 1080,
            "field_width": 770, "field_height": 434,
            "expected_scene_epoch": 3,
        }
        decoded = dump_contract.validate_subtitle_locator_state(
            self._pack_uint32_words(words), **arguments)
        self.assertTrue(decoded["provisional_current"])
        self.assertEqual(decoded["provisional_target_bits"], provisional_bits)
        self.assertEqual(decoded["provisional_fade"], 2)

        wrong_cover = words.copy()
        wrong_cover[dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET] = pending[0] + 1
        with self.assertRaisesRegex(ValueError, "does not contain"):
            dump_contract.validate_subtitle_locator_state(
                self._pack_uint32_words(wrong_cover), **arguments)

        wrong_geometry = words.copy()
        unrelated = [100, 330, 500, 340]
        wrong_geometry[5:9] = unrelated
        wrong_geometry[9] = (unrelated[2] - unrelated[0]) * (unrelated[3] - unrelated[1])
        wrong_geometry[dump_contract.SUBTITLE_LOCATOR_OWNER_WORD_OFFSET:
                       dump_contract.SUBTITLE_LOCATOR_OWNER_WORD_OFFSET + 4] = unrelated
        with self.assertRaisesRegex(ValueError, "provisional current cover"):
            dump_contract.validate_subtitle_locator_state(
                self._pack_uint32_words(wrong_geometry), **arguments)

        iou_equality = words.copy()
        equal_owner = [100, 360, 200, 370]
        equal_pending = [125, 360, 225, 370]
        equal_cover = [121, 356, 229, 374]
        iou_equality[5:9] = equal_owner
        iou_equality[9] = 1000
        iou_equality[13:17] = equal_pending
        iou_equality[17] = 1000
        iou_equality[dump_contract.SUBTITLE_LOCATOR_OWNER_WORD_OFFSET:
                     dump_contract.SUBTITLE_LOCATOR_OWNER_WORD_OFFSET + 4] = equal_owner
        iou_equality[dump_contract.SUBTITLE_LOCATOR_PENDING_WORD_OFFSET:
                     dump_contract.SUBTITLE_LOCATOR_PENDING_WORD_OFFSET + 4] = equal_pending
        iou_equality[dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET:
                     dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET + 4] = equal_cover
        with self.assertRaisesRegex(ValueError, "provisional current cover"):
            dump_contract.validate_subtitle_locator_state(
                self._pack_uint32_words(iou_equality), **arguments)

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
    def _valid_slr13_state_words():
        words = [0] * dump_contract.SUBTITLE_LOCATOR_STATE_WORD_COUNT
        target_bits = struct.unpack("<I", struct.pack("<f", 0.00075))[0]
        words[:32] = [
            dump_contract.SUBTITLE_LOCATOR_STATE_SCHEMA,
            dump_contract.SUBTITLE_LOCATOR_STATE_TAG,
            (dump_contract.SUBTITLE_LOCATOR_FLAG_OWNER |
             dump_contract.SUBTITLE_LOCATOR_FLAG_TARGET_VALID),
            9,
            1,
            120, 350, 650, 401,
            530 * 51,
            17, 0,
            0,
            0, 0, 0, 0,
            0,
            target_bits,
            9,
            1,
            dump_contract.SUBTITLE_LOCATOR_EVENT_BIRTH,
            41, 0,
            1,
            0,
            3,
            770,
            434,
            0,
            0,
            0,
        ]
        rectangle = [120, 350, 650, 401]
        owner = dump_contract.SUBTITLE_LOCATOR_OWNER_WORD_OFFSET
        current = dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET
        words[owner:owner + 4] = rectangle
        words[current:current + 4] = rectangle
        return words

    def test_current_slr13_state_validates_identity_rectangles_and_target(self):
        decoded = dump_contract.validate_subtitle_locator_state(
            self._pack_uint32_words(self._valid_slr13_state_words()),
            matched_frame_id=41,
            analysis_generation=17,
            source_width=1920,
            source_height=1080,
            field_width=770,
            field_height=434,
            expected_scene_epoch=3)
        self.assertEqual(decoded["owner_count"], 1)
        self.assertEqual(decoded["current_count"], 1)
        self.assertEqual(decoded["target"], struct.unpack("<f", struct.pack("<f", 0.00075))[0])

        wrong_epoch = self._valid_slr13_state_words()
        wrong_epoch[26] = 4
        with self.assertRaisesRegex(ValueError, "scene epoch"):
            dump_contract.validate_subtitle_locator_state(
                self._pack_uint32_words(wrong_epoch),
                matched_frame_id=41,
                analysis_generation=17,
                source_width=1920,
                source_height=1080,
                field_width=770,
                field_height=434,
                expected_scene_epoch=3)
        self.assertEqual(decoded["last_event"], dump_contract.SUBTITLE_LOCATOR_EVENT_BIRTH)
        self.assertEqual(decoded["unreliable_target_holds"], 0)
        self.assertEqual(decoded["current_rectangles"], [{
            "left": 120, "top": 350, "right": 650, "bottom": 401,
            "kind": "text", "ribbon": False,
        }])

        empty = [0] * dump_contract.SUBTITLE_LOCATOR_STATE_WORD_COUNT
        empty[:32] = [
            dump_contract.SUBTITLE_LOCATOR_STATE_SCHEMA,
            dump_contract.SUBTITLE_LOCATOR_STATE_TAG,
            0, 0, 0,
            0, 0, 0, 0, 0,
            17, 0,
            0,
            0, 0, 0, 0, 0,
            0, 0,
            0,
            dump_contract.SUBTITLE_LOCATOR_EVENT_NONE,
            41, 0,
            0, 0, 3,
            770, 434,
            0, 0, 0,
        ]
        decoded = dump_contract.validate_subtitle_locator_state(
            self._pack_uint32_words(empty),
            matched_frame_id=41,
            analysis_generation=17,
            source_width=1920,
            source_height=1080,
            field_width=770,
            field_height=434)
        self.assertEqual(decoded["current_rectangles"], [])
        self.assertIsNone(decoded["target"])

        grace = empty.copy()
        cached_bits = struct.unpack("<I", struct.pack("<f", 0.0006))[0]
        grace[18] = cached_bits
        grace[25] = (
            coordinate.SUBTITLE_OCR.locator_death_grace_observations)
        grace[29] = 120 | (650 << 16)
        grace[30] = 350 | (401 << 16)
        decoded = dump_contract.validate_subtitle_locator_state(
            self._pack_uint32_words(grace),
            matched_frame_id=41,
            analysis_generation=17,
            source_width=1920,
            source_height=1080,
            field_width=770,
            field_height=434)
        self.assertEqual(
            decoded["target_grace"],
            coordinate.SUBTITLE_OCR.locator_death_grace_observations)
        self.assertEqual(decoded["grace_bounds"], {
            "left": 120, "top": 350, "right": 650, "bottom": 401,
        })

        held = self._valid_slr13_state_words()
        held[25] = coordinate.SUBTITLE_OCR.locator_target_max_unreliable_holds
        held[21] = dump_contract.SUBTITLE_LOCATOR_EVENT_NONE
        decoded = dump_contract.validate_subtitle_locator_state(
            self._pack_uint32_words(held),
            matched_frame_id=41, analysis_generation=17,
            source_width=1920, source_height=1080,
            field_width=770, field_height=434)
        self.assertEqual(
            decoded["unreliable_target_holds"],
            coordinate.SUBTITLE_OCR.locator_target_max_unreliable_holds)
        self.assertEqual(decoded["target_grace"], 0)

        # A missing current OCR authority frame preserves an existing hold counter and target but
        # cannot age or condition with them. The serialized state therefore has no current cover.
        held_without_current = held.copy()
        held_without_current[20] = 0
        current = dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET
        held_without_current[current:current + 4] = [0, 0, 0, 0]
        held_without_current[31] &= ~(
            dump_contract.SUBTITLE_LOCATOR_KIND_MASK <<
            dump_contract.SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT)
        decoded = dump_contract.validate_subtitle_locator_state(
            self._pack_uint32_words(held_without_current),
            matched_frame_id=41, analysis_generation=17,
            source_width=1920, source_height=1080,
            field_width=770, field_height=434)
        self.assertEqual(decoded["current_count"], 0)
        self.assertEqual(
            decoded["unreliable_target_holds"],
            coordinate.SUBTITLE_OCR.locator_target_max_unreliable_holds)

        too_many_holds = held.copy()
        too_many_holds[25] += 1
        with self.assertRaisesRegex(ValueError, "unreliable-target hold"):
            dump_contract.validate_subtitle_locator_state(
                self._pack_uint32_words(too_many_holds),
                matched_frame_id=41, analysis_generation=17,
                source_width=1920, source_height=1080,
                field_width=770, field_height=434)

        held_with_event = held.copy()
        held_with_event[21] = dump_contract.SUBTITLE_LOCATOR_EVENT_BIRTH
        with self.assertRaisesRegex(ValueError, "unreliable-target hold"):
            dump_contract.validate_subtitle_locator_state(
                self._pack_uint32_words(held_with_event),
                matched_frame_id=41, analysis_generation=17,
                source_width=1920, source_height=1080,
                field_width=770, field_height=434)

        for cached in (-0.0401, 0.0401):
            invalid_grace = grace.copy()
            invalid_grace[18] = struct.unpack("<I", struct.pack("<f", cached))[0]
            with self.subTest(cached_target=cached), self.assertRaisesRegex(
                    ValueError, "death-grace target"):
                dump_contract.validate_subtitle_locator_state(
                    self._pack_uint32_words(invalid_grace),
                    matched_frame_id=41,
                    analysis_generation=17,
                    source_width=1920,
                    source_height=1080,
                    field_width=770,
                    field_height=434)

    def test_slr13_target_accepts_signed_local_planes_within_direct_container(self):
        import numpy as np

        locator_arguments = {
            "matched_frame_id": 41,
            "analysis_generation": 17,
            "source_width": 3440,
            "source_height": 1440,
            "field_width": 770,
            "field_height": 434,
        }
        permitted = (-0.03, -0.0005, 0.0, 0.0005, 0.03)
        for target in permitted:
            target_bits = int(np.asarray(target, dtype=np.float32).view(np.uint32))
            with self.subTest(target=target, state="live"):
                live = self._valid_slr13_state_words()
                live[18] = target_bits
                decoded = dump_contract.validate_subtitle_locator_state(
                    self._pack_uint32_words(live), **locator_arguments)
                self.assertEqual(decoded["target_bits"], target_bits)

            with self.subTest(target=target, state="grace"):
                grace = [0] * dump_contract.SUBTITLE_LOCATOR_STATE_WORD_COUNT
                grace[:32] = [
                    dump_contract.SUBTITLE_LOCATOR_STATE_SCHEMA,
                    dump_contract.SUBTITLE_LOCATOR_STATE_TAG,
                    0, 0, 0,
                    0, 0, 0, 0, 0,
                    17, 0,
                    0,
                    0, 0, 0, 0,
                    0,
                    target_bits,
                    0,
                    0,
                    dump_contract.SUBTITLE_LOCATOR_EVENT_NONE,
                    41, 0,
                    0,
                    1,
                    3,
                    770,
                    434,
                    120 | (650 << 16),
                    350 | (401 << 16),
                    0,
                ]
                decoded = dump_contract.validate_subtitle_locator_state(
                    self._pack_uint32_words(grace), **locator_arguments)
                self.assertEqual(decoded["target_bits"], target_bits)

        rejected_bits = int(np.asarray(0.0401, dtype=np.float32).view(np.uint32))
        rejected = self._valid_slr13_state_words()
        rejected[18] = rejected_bits
        with self.assertRaisesRegex(ValueError, "representation"):
            dump_contract.validate_subtitle_locator_state(
                self._pack_uint32_words(rejected), **locator_arguments)

        base = np.full((434, 770), np.float32(0.02), dtype=np.float32)
        subtitle = {
            "current_rectangles": [{
                "left": 120, "top": 350, "right": 650, "bottom": 401,
            }],
            "source_width": 3440,
            "field_width": 770,
            "field_height": 434,
            "fade": 2,
        }
        for target in permitted:
            with self.subTest(target=target, state="conditioner"):
                subtitle["target"] = np.float32(target)
                dump_contract._replay_slr13_conditioner(base, subtitle)
        subtitle["target"] = np.asarray(
            [rejected_bits], dtype=np.uint32).view(np.float32)[0]
        with self.assertRaisesRegex(ValueError, "representation limit"):
            dump_contract._replay_slr13_conditioner(base, subtitle)

    def test_current_slr13_state_rejects_identity_flags_slots_and_aggregates(self):
        def add_second_current(words):
            words[20] = 2
            offset = dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET + 4
            words[offset:offset + 4] = [130, 352, 220, 380]

        def add_pending(words, *, valid_aggregate):
            words[2] |= dump_contract.SUBTITLE_LOCATOR_FLAG_PENDING
            words[12] = 1
            words[13:17] = [140, 360, 610, 398]
            words[17] = 470 * 38 if valid_aggregate else 1
            offset = dump_contract.SUBTITLE_LOCATOR_PENDING_WORD_OFFSET
            words[offset:offset + 4] = [140, 360, 610, 398]

        def make_target_reset_with_stale_target(words):
            words[2] = (dump_contract.SUBTITLE_LOCATOR_FLAG_OWNER |
                        dump_contract.SUBTITLE_LOCATOR_FLAG_TARGET_RESET)
            words[20] = 0
            words[24] = 0
            offset = dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET
            words[offset:offset + 4] = [0, 0, 0, 0]

        def make_invalid_grace(words):
            words[2] = 0
            words[3] = 0
            words[4] = 0
            words[5:10] = [0, 0, 0, 0, 0]
            owner = dump_contract.SUBTITLE_LOCATOR_OWNER_WORD_OFFSET
            words[owner:owner + 4] = [0, 0, 0, 0]
            words[19] = 0
            words[20] = 0
            current = dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET
            words[current:current + 4] = [0, 0, 0, 0]
            words[24] = 0
            words[25] = 2
            words[29] = 650 | (120 << 16)
            words[30] = 350 | (401 << 16)

        def make_grace_above_limit(words):
            words[2] = 0
            words[3] = 0
            words[4] = 0
            words[5:10] = [0, 0, 0, 0, 0]
            owner = dump_contract.SUBTITLE_LOCATOR_OWNER_WORD_OFFSET
            words[owner:owner + 4] = [0, 0, 0, 0]
            words[19] = 0
            words[20] = 0
            current = dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET
            words[current:current + 4] = [0, 0, 0, 0]
            words[24] = 0
            words[25] = (
                coordinate.SUBTITLE_OCR.locator_death_grace_observations + 1)
            words[29] = 120 | (650 << 16)
            words[30] = 350 | (401 << 16)

        def make_owner_without_target_with_stale_words(words):
            words[2] = dump_contract.SUBTITLE_LOCATOR_FLAG_OWNER
            words[20] = 0
            words[24] = 0
            current = dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET
            words[current:current + 4] = [0, 0, 0, 0]

        mutations = {
            "tag": (lambda words: words.__setitem__(1, 0), "schema or tag"),
            "retired-slr8-identity": (
                lambda words: words.__setitem__(slice(0, 2), [8, 0x38524C53]),
                "schema or tag"),
            "generation": (lambda words: words.__setitem__(10, 18), "identity"),
            "unknown-flags": (lambda words: words.__setitem__(2, 0x20), "unknown flags"),
            "flag-count": (
                lambda words: words.__setitem__(2, dump_contract.SUBTITLE_LOCATOR_FLAG_TARGET_VALID),
                "flags disagree"),
            "owner-bbox": (lambda words: words.__setitem__(8, 651), "bbox or area"),
            "owner-area": (lambda words: words.__setitem__(9, 1), "bbox or area"),
            "owner-generation": (
                lambda words: words.__setitem__(3, 0),
                "owner generation"),
            "target-generation": (
                lambda words: words.__setitem__(19, 8),
                "valid target identity"),
            "current-exceeds-owner": (
                add_second_current,
                "cannot exceed its owner count"),
            "pending-area": (
                lambda words: add_pending(words, valid_aggregate=False),
                "pending bbox or area"),
            "unused-owner": (
                lambda words: words.__setitem__(
                    dump_contract.SUBTITLE_LOCATOR_OWNER_WORD_OFFSET + 4, 1),
                "canonical zero"),
            "outside-field": (
                lambda words: words.__setitem__(
                    dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET + 2, 771),
                "field coordinates"),
            "nan-target": (
                lambda words: words.__setitem__(18, 0x7FC00000),
                "valid target"),
            "target-below-direct-container": (
                lambda words: words.__setitem__(
                    18, struct.unpack("<I", struct.pack("<f", -0.0401))[0]),
                "representation"),
            "target-above-direct-container": (
                lambda words: words.__setitem__(
                    18, struct.unpack("<I", struct.pack("<f", 0.0401))[0]),
                "representation"),
            "zero-fade-with-current": (
                lambda words: words.__setitem__(24, 0),
                "current rectangles require"),
            "unknown-fade": (
                lambda words: words.__setitem__(24, 3),
                "fade step"),
            "event": (lambda words: words.__setitem__(21, 4), "unknown last event"),
            "target-reset-stale-target": (
                make_target_reset_with_stale_target,
                "target reset must clear"),
            "owner-hold-above-contract-limit": (
                lambda words: words.__setitem__(
                    25, coordinate.SUBTITLE_OCR.locator_target_max_unreliable_holds + 1),
                "unreliable-target hold"),
            "invalid-grace-bounds": (
                make_invalid_grace,
                "death-grace target or packed bounds"),
            "grace-above-contract-limit": (
                make_grace_above_limit,
                "death-grace exceeds the authenticated observation limit"),
            "owner-without-target-stale-words": (
                make_owner_without_target_with_stale_words,
                "owner without target authority"),
            "unknown-kind-bit": (
                lambda words: words.__setitem__(31, 1 << 12),
                "unknown packed-kind bits"),
        }
        for name, (mutate, error) in mutations.items():
            with self.subTest(name=name):
                words = self._valid_slr13_state_words()
                mutate(words)
                with self.assertRaisesRegex(ValueError, error):
                    dump_contract.validate_subtitle_locator_state(
                        self._pack_uint32_words(words),
                        matched_frame_id=41,
                        analysis_generation=17,
                        source_width=1920,
                        source_height=1080,
                        field_width=770,
                        field_height=434)

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
                    "deterministic authenticated integer contain fit"),
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
        # Schema 39 must not advertise packaged scalar previews at all.
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
        target_bits = struct.unpack("<I", struct.pack("<f", 0.01))[0]
        locator = [0] * dump_contract.GPU_TRACE_LOCATOR_WORD_COUNT
        locator[0] = dump_contract.SUBTITLE_LOCATOR_STATE_SCHEMA
        locator[1] = dump_contract.SUBTITLE_LOCATOR_STATE_TAG
        locator[2] = (dump_contract.SUBTITLE_LOCATOR_FLAG_OWNER |
                      dump_contract.SUBTITLE_LOCATOR_FLAG_TARGET_VALID)
        locator[3] = 1
        locator[4] = 1
        locator[10] = parsed["analysis_generation"] & 0xFFFFFFFF
        locator[11] = parsed["analysis_generation"] >> 32
        locator[18] = target_bits
        locator[19] = 1
        locator[20] = 1
        locator[21] = dump_contract.SUBTITLE_LOCATOR_EVENT_BIRTH
        locator[22] = self.manifest["matched_frame_id"]
        locator[24] = 1
        locator[26] = 3
        locator[27] = field_width
        locator[28] = field_height
        condition = [
            coordinate.SUBTITLE_OCR.condition_param_schema,
            coordinate.SUBTITLE_OCR.condition_param_tag,
            1,
            0,
            1,
            target_bits,
        ]
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
        record[88:168] = locator
        record[168:174] = condition
        observation_timestamp_us = 1_000_000 + self.manifest["matched_frame_id"]
        record[174] = observation_timestamp_us & 0xFFFFFFFF
        record[175] = observation_timestamp_us >> 32
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
            words[locator + 22] = frame - 1
            words[locator + 23] = 0
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
        ownership = candidate.copy()
        ownership[4, 7] = np.float32(ownership[4, 7] + 0.005)  # raise-only refinement

        majorant, conditioned, final = dump_contract._replay_v2_limiter_fields(
            ownership, width)

        manifest = copy.deepcopy(self.manifest)
        for name in (
                "model_input", "raw_depth",
                "shadow_coordinate",
                "shadow_candidate_parallax", "shadow_ownership_refined_parallax",
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
            "shadow_ownership_refined_parallax": ownership,
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

    def _activate_synthetic_slr13_dump(self, root, manifest, fields):
        manifest = self._active_slr13_manifest(manifest)
        base_payload = fields["shadow_final_parallax"].astype("<f4").tobytes()
        self._write_hashed_payload(
            root, manifest, "shadow_base_final_parallax.f32", base_payload)

        ocr = [0] * dump_contract.SUBTITLE_OCR_RECORD_WORD_COUNT
        ocr[:16] = [
            dump_contract.SUBTITLE_OCR_RECORD_SCHEMA,
            dump_contract.SUBTITLE_OCR_RECORD_TAG,
            1, 0, 0,
            41, 0,
            0, 0,
            1920, 1080,
            770, 434,
            325, 430,
            0,
        ]
        self._write_hashed_payload(
            root, manifest, "subtitle_ocr_record.u32",
            self._pack_uint32_words(ocr))

        locator = [0] * dump_contract.SUBTITLE_LOCATOR_STATE_WORD_COUNT
        locator[:32] = [
            dump_contract.SUBTITLE_LOCATOR_STATE_SCHEMA,
            dump_contract.SUBTITLE_LOCATOR_STATE_TAG,
            0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0,
            0,
            0, 0, 0, 0, 0,
            0, 0,
            0,
            dump_contract.SUBTITLE_LOCATOR_EVENT_NONE,
            41, 0,
            0, 0, 3,
            770, 434,
            0, 0, 0,
        ]
        self._write_hashed_payload(
            root, manifest, "subtitle_locator_state.u32",
            self._pack_uint32_words(locator))
        self._write_subtitle_conditioning_document(root, manifest)
        (root / "dump_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        return manifest

    def _activate_synthetic_slr13_current_dump(
            self, root, manifest, fields, *, fade=1):
        manifest = self._activate_synthetic_slr13_dump(root, manifest, fields)
        ocr = self._valid_ocr_record_words()
        ocr[7] = 0
        ocr[8] = 0
        self._write_hashed_payload(
            root, manifest, "subtitle_ocr_record.u32",
            self._pack_uint32_words(ocr))

        locator = self._valid_slr13_state_words()
        locator[10] = 0
        locator[11] = 0
        locator[24] = fade
        self._write_hashed_payload(
            root, manifest, "subtitle_locator_state.u32",
            self._pack_uint32_words(locator))
        rectangle = {
            "left": 120, "top": 350, "right": 650, "bottom": 401,
        }
        target = struct.unpack("<f", struct.pack("<I", locator[18]))[0]
        subtitle = {
            "current_rectangles": [rectangle],
            "source_width": 1920,
            "field_width": 770,
            "field_height": 434,
            "target": target,
            "fade": fade,
        }
        conditioned = dump_contract._replay_slr13_conditioner(
            fields["shadow_final_parallax"], subtitle)
        payload = conditioned.astype("<f4").tobytes()
        self._write_hashed_payload(
            root, manifest, "shadow_final_parallax.f32", payload)
        self._write_subtitle_conditioning_document(root, manifest)
        (root / "dump_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        return manifest, conditioned, locator, ocr

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
            "parallax_v2_ownership_refined_role":
                "conservative full-resolution crop-local source-contour foreground ownership "
                "applied to candidate before the vertical conditioner; may only raise uniquely "
                "owned far-side boundary texels",
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
                "shadow_candidate_parallax", "shadow_ownership_refined_parallax",
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
            "shadow_ownership_refined_parallax": constant,
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
        mask_payload = b"synthetic authenticated warp mask"
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
            "gain": "one-eye ROI-local source-U per curve unit",
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

    def test_schema39_generates_float_previews_on_demand_outside_package(self):
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

    def test_geometry_verifier_accepts_current_slr13_empty_authority_as_exact_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, fields = self._write_synthetic_geometry_dump(
                root,
                width=770,
                height=434)
            self._activate_synthetic_slr13_dump(root, manifest, fields)

            summary = dump_contract.verify_v2_dump_geometry(root)

            subtitle = summary["subtitle_conditioning"]
            self.assertEqual(subtitle["mode"], "subtitle-slr13")
            self.assertTrue(subtitle["subtitle_evidence_verified"])
            self.assertTrue(subtitle["ocr_authoritative"])
            self.assertEqual(subtitle["current_count"], 0)
            self.assertIn(
                "shadow_base_final_parallax",
                summary["chain_fields_verified"])

    def test_geometry_verifier_rejects_slr13_scene_epoch_not_bound_to_shadow_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, fields = self._write_synthetic_geometry_dump(
                root,
                width=770,
                height=434)
            manifest = self._activate_synthetic_slr13_dump(root, manifest, fields)
            locator = list(struct.unpack(
                f"<{dump_contract.SUBTITLE_LOCATOR_STATE_WORD_COUNT}I",
                (root / "subtitle_locator_state.u32").read_bytes()))
            locator[26] = 4
            self._write_hashed_payload(
                root, manifest, "subtitle_locator_state.u32",
                self._pack_uint32_words(locator))
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "scene epoch"):
                dump_contract.verify_v2_dump_geometry(root)

    def test_geometry_verifier_accepts_abstaining_ocr_with_slr13_target_grace_as_exact_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, fields = self._write_synthetic_geometry_dump(
                root,
                width=770,
                height=434)
            manifest = self._activate_synthetic_slr13_dump(root, manifest, fields)

            ocr = list(struct.unpack(
                f"<{dump_contract.SUBTITLE_OCR_RECORD_WORD_COUNT}I",
                (root / "subtitle_ocr_record.u32").read_bytes()))
            ocr[2] = 0
            self._write_hashed_payload(
                root, manifest, "subtitle_ocr_record.u32",
                self._pack_uint32_words(ocr))

            locator = list(struct.unpack(
                f"<{dump_contract.SUBTITLE_LOCATOR_STATE_WORD_COUNT}I",
                (root / "subtitle_locator_state.u32").read_bytes()))
            locator[18] = struct.unpack("<I", struct.pack("<f", 0.0006))[0]
            locator[21] = dump_contract.SUBTITLE_LOCATOR_EVENT_DEATH
            locator[25] = 6
            locator[29] = 120 | (650 << 16)
            locator[30] = 350 | (401 << 16)
            self._write_hashed_payload(
                root, manifest, "subtitle_locator_state.u32",
                self._pack_uint32_words(locator))
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")

            summary = dump_contract.verify_v2_dump_geometry(root)
            subtitle = summary["subtitle_conditioning"]
            self.assertFalse(subtitle["ocr_authoritative"])
            self.assertEqual(subtitle["current_count"], 0)
            self.assertEqual(subtitle["target_grace"], 6)
            self.assertAlmostEqual(subtitle["cached_target"], 0.0006)
            self.assertEqual(subtitle["grace_bounds"], {
                "left": 120, "top": 350, "right": 650, "bottom": 401,
            })

    def test_geometry_verifier_rejects_slr13_record_identity_and_nonbase_empty_output(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, fields = self._write_synthetic_geometry_dump(
                root,
                width=770,
                height=434)
            manifest = self._activate_synthetic_slr13_dump(root, manifest, fields)

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
            with self.assertRaisesRegex(ValueError, "not the exact content-clamped Base"):
                dump_contract.verify_v2_dump_geometry(root)

    def test_geometry_verifier_replays_nonempty_slr13_rectangle_fade_and_ocr_binding(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, fields = self._write_synthetic_geometry_dump(
                root,
                width=770,
                height=434)
            manifest, fade_one, locator, ocr = (
                self._activate_synthetic_slr13_current_dump(
                    root, manifest, fields, fade=1))

            summary = dump_contract.verify_v2_dump_geometry(root)
            self.assertEqual(summary["subtitle_conditioning"]["current_count"], 1)
            base = fields["shadow_final_parallax"]
            self.assertTrue(np.array_equal(fade_one[0, 0], base[0, 0]))
            self.assertTrue(np.any(
                fade_one[350:401, 120:650] != base[350:401, 120:650]))

            locator[24] = 2
            self._write_hashed_payload(
                root, manifest, "subtitle_locator_state.u32",
                self._pack_uint32_words(locator))
            target = struct.unpack("<f", struct.pack("<I", locator[18]))[0]
            fade_two = dump_contract._replay_slr13_conditioner(base, {
                "current_rectangles": [{
                    "left": 120, "top": 350, "right": 650, "bottom": 401,
                }],
                "source_width": 1920,
                "field_width": 770,
                "field_height": 434,
                "target": target,
                "fade": 2,
            })
            fade_two_payload = fade_two.astype("<f4").tobytes()
            self._write_hashed_payload(
                root, manifest, "shadow_final_parallax.f32", fade_two_payload)
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            summary = dump_contract.verify_v2_dump_geometry(root)
            self.assertEqual(summary["subtitle_conditioning"]["fade"], 2)
            self.assertTrue(np.any(fade_two != fade_one))

            tampered = fade_two.copy()
            tampered[360, 200] = np.nextafter(
                tampered[360, 200], np.float32(np.inf), dtype=np.float32)
            tampered_payload = tampered.astype("<f4").tobytes()
            self._write_hashed_payload(
                root, manifest, "shadow_final_parallax.f32", tampered_payload)
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact SLR13 rectangle-conditioning"):
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

    def test_slr13_sm5_replay_accepts_the_roi_width_1101_division_bit(self):
        import numpy as np

        exact_candidates = dump_contract._sm5_power_of_two_division_candidates(
            0.5, 1024)
        exact_bits = {
            int(np.asarray(value, dtype=np.float32).view(np.uint32))
            for value in exact_candidates
        }
        self.assertEqual(exact_bits, {0x39FFFFFF, 0x3A000000, 0x3A000001})

        exact_base = np.full((434, 770), np.float32(0.02), dtype=np.float32)
        exact_subtitle = {
            "current_rectangles": [{
                "left": 210, "top": 393, "right": 561, "bottom": 430,
            }],
            "source_width": 1024,
            "field_width": 770,
            "field_height": 434,
            "target": 0.0,
            "fade": 2,
        }
        exact_replay_bits = {
            int(replay[393, 210].view(np.uint32))
            for replay in dump_contract._replay_slr13_conditioner_sm5_candidates(
                exact_base, exact_subtitle)
        }
        self.assertEqual(exact_replay_bits, exact_bits)

        candidates = dump_contract._sm5_power_of_two_division_candidates(0.5, 1101)
        candidate_bits = {
            int(np.asarray(value, dtype=np.float32).view(np.uint32))
            for value in candidates
        }
        self.assertEqual(candidate_bits, {0x39EE18A5, 0x39EE18A6})

        target = np.asarray([0x39FA3F70], dtype=np.uint32).view(np.float32)[0]
        base = np.full((434, 770), np.float32(0.02), dtype=np.float32)
        subtitle = {
            "current_rectangles": [{
                "left": 210, "top": 393, "right": 561, "bottom": 430,
            }],
            "source_width": 1101,
            "field_width": 770,
            "field_height": 434,
            "target": target,
            "fade": 2,
        }
        replay_bits = {
            int(replay[393, 210].view(np.uint32))
            for replay in dump_contract._replay_slr13_conditioner_sm5_candidates(
                base, subtitle)
        }
        # 0x3A742C0A is the production NVIDIA field bit from the supplied ROI dump;
        # 0x3A742C0B is the correctly rounded WARP/NumPy alternative.
        self.assertEqual(replay_bits, {0x3A742C0A, 0x3A742C0B})

    def test_slr13_replay_uses_content_width_and_boundary_extends_padding(self):
        import numpy as np

        content = (100, 10, 670, 424)
        base = np.full((434, 770), np.float32(-0.02), dtype=np.float32)
        base[content[1]:content[3], content[0]:content[2]] = np.float32(0.03)
        subtitle = {
            "current_rectangles": [{
                "left": 300, "top": 200, "right": 400, "bottom": 250,
            }],
            "source_width": 1000,
            "field_width": 770,
            "field_height": 434,
            "tensor_content_rect": content,
            "target": 0.0,
            "fade": 2,
        }
        replay = dump_contract._replay_slr13_conditioner(base, subtitle)
        horizontal_step = np.float32(
            np.float32(coordinate.CALIBRATED_DEFAULTS.max_horizontal_slope) /
            np.float32(content[2] - content[0]))
        core_range = np.float32(np.float32(0.5) / np.float32(1000))
        expected_one_cell = np.add(core_range, horizontal_step, dtype=np.float32)
        self.assertEqual(replay[220, 299], expected_one_cell)

        # Synthetic tensor padding is never conditioned from its own model value. It repeats the
        # nearest content result exactly on all four sides.
        np.testing.assert_array_equal(
            replay[:, :content[0]],
            np.repeat(replay[:, content[0]:content[0] + 1], content[0], axis=1))
        np.testing.assert_array_equal(
            replay[:, content[2]:],
            np.repeat(replay[:, content[2] - 1:content[2]], 770 - content[2], axis=1))
        np.testing.assert_array_equal(
            replay[:content[1], :],
            np.repeat(replay[content[1]:content[1] + 1, :], content[1], axis=0))
        np.testing.assert_array_equal(
            replay[content[3]:, :],
            np.repeat(replay[content[3] - 1:content[3], :], 434 - content[3], axis=0))

    def test_slr13_ribbon_replay_has_only_a_top_edge_collar(self):
        import numpy as np

        base = np.full((434, 770), np.float32(0.02), dtype=np.float32)
        common = {
            "source_width": 1920,
            "field_width": 770,
            "field_height": 434,
            "target": 0.0,
            "fade": 2,
        }
        geometry = {
            "left": 120, "top": 350, "right": 650, "bottom": 401,
        }
        ribbon = dump_contract._replay_slr13_conditioner(base, {
            **common,
            "current_rectangles": [{**geometry, "kind": "ribbon", "ribbon": True}],
        })
        ordinary = dump_contract._replay_slr13_conditioner(base, {
            **common,
            "current_rectangles": [{**geometry, "kind": "text", "ribbon": False}],
        })

        # Ribbon authority ignores the nominal side and bottom edges: every column and
        # every row at/below the corrected top receives the same core budget.
        self.assertEqual(ribbon[360, 0], ribbon[360, 200])
        self.assertEqual(ribbon[360, 769], ribbon[360, 200])
        self.assertEqual(ribbon[433, 200], ribbon[360, 200])
        self.assertNotEqual(ribbon[360, 0], base[360, 0])
        self.assertNotEqual(ribbon[433, 200], base[433, 200])

        # Above the corrected top there is one finite vertical collar and no horizontal
        # component.  The equivalent ordinary rectangle still has side/bottom collars.
        self.assertNotEqual(ribbon[349, 0], base[349, 0])
        self.assertEqual(ribbon[340, 0], base[340, 0])
        self.assertEqual(ribbon[349, 0], ribbon[349, 769])
        self.assertEqual(ordinary[360, 0], base[360, 0])
        self.assertEqual(ordinary[433, 200], base[433, 200])

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
            ownership = candidate.copy()
            majorant, conditioned, final = dump_contract._replay_v2_limiter_fields(
                ownership, right - left)

            fields = {
                "shadow_candidate_parallax.f32": candidate,
                "shadow_ownership_refined_parallax.f32": ownership,
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
            _, _, legacy = dump_contract._replay_v2_limiter_fields(ownership, width)
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
        self.assertEqual(summary["record_words"], 176)
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
        words[oldest_base + 88:oldest_base + 174] = [0xFFFFFFFF] * 86
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
        self.assertEqual(record["subtitle_locator"]["frame_id"], 40)

        locator = base + dump_contract.GPU_TRACE_RECORD_OFFSETS[
            "subtitle_locator_begin"]
        words[locator + 22] = 41
        with self.assertRaisesRegex(ValueError, "invalid finalized SLR13 state"):
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
        words[oldest_locator + 22] = 40
        words[oldest_base + 174] = 1_000_040
        words[oldest_base + 175] = 0
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
        changed[newest_locator + 18] ^= 1
        with self.assertRaisesRegex(ValueError, "immediately prior record"):
            dump_contract.validate_gpu_trace_ring(
                struct.pack(f"<{len(changed)}I", *changed),
                matched_frame_id=41, analysis_generation=0,
                source_width=1920, source_height=1080,
                field_width=16, field_height=12,
                expected_domain_tag=domain_tag, input_domain_reset=False)

    def test_gpu_trace_due_observation_advances_on_reuse_without_reauthorizing_ordinary(self):
        input_region = copy.deepcopy(self.depth_input_region)
        input_region["analysis"]["tensor_extent_px"] = {"width": 16, "height": 12}
        input_region["analysis"]["tensor_content_rect_px"] = {
            "left": 0, "top": 0, "right": 16, "bottom": 12}
        for work, marker, expected_name, expected_work_name in (
                (dump_contract.GPU_TRACE_WORK_OPTIONAL_OCR_DUE,
                 dump_contract.GPU_TRACE_OPTIONAL_RECEIPT_MAGIC,
                 "optional-ocr", "optional-ocr-due"),
                (dump_contract.GPU_TRACE_WORK_SUBTITLE_OBSERVATION_DUE,
                 0, "abstention", "subtitle-observation-due")):
            with self.subTest(work=work):
                raw, _, domain_tag = self._gpu_trace_ring_payload(
                    input_region, 16, 12)
                words = list(struct.unpack(
                    f"<{dump_contract.GPU_TRACE_RING_WORD_COUNT}I", raw))
                base, transaction = self._set_gpu_trace_reuse(
                    words, work=work, host_outcome=1)
                words[base + dump_contract.GPU_TRACE_RECORD_OFFSETS[
                    "subtitle_disposition"]] = 1 if marker else 2
                locator = base + dump_contract.GPU_TRACE_RECORD_OFFSETS[
                    "subtitle_locator_begin"]
                words[locator + 22] = 41
                words[transaction + 1] = (
                    dump_contract.GPU_TRACE_DECISION_COOKIE ^ marker)
                words[transaction + 7] = marker
                decoded = dump_contract.validate_gpu_trace_ring(
                    struct.pack(f"<{len(words)}I", *words),
                    matched_frame_id=41, analysis_generation=0,
                    source_width=1920, source_height=1080,
                    field_width=16, field_height=12,
                    expected_domain_tag=domain_tag, input_domain_reset=False)
                record = decoded["decoded"]["records"][0]
                self.assertEqual(record["expected_work"]["name"], expected_work_name)
                self.assertEqual(record["subtitle_disposition"]["name"], expected_name)
                self.assertTrue(record["flags"]["condition_executed"])
                self.assertFalse(record["subtitle_condition"]["held_with_depth"])

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
        words[newest_base + 174] = 0
        words[newest_base + 175] = 0
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
        words[oldest_locator + 22] = 40
        newer_timestamp = 1_000_041
        older_timestamp = newer_timestamp + 1
        words[oldest_base + 174] = older_timestamp
        words[oldest_base + 175] = 0
        words[newest_base + 174] = newer_timestamp
        words[newest_base + 175] = 0
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

    def test_gpu_trace_inactive_locator_requires_canonical_zero_condition(self):
        input_region = copy.deepcopy(self.depth_input_region)
        input_region["analysis"]["tensor_extent_px"] = {"width": 16, "height": 12}
        input_region["analysis"]["tensor_content_rect_px"] = {
            "left": 0, "top": 0, "right": 16, "bottom": 12}
        raw, _, domain_tag = self._gpu_trace_ring_payload(input_region, 16, 12)
        words = list(struct.unpack(
            f"<{dump_contract.GPU_TRACE_RING_WORD_COUNT}I", raw))
        base = dump_contract.GPU_TRACE_HEADER_WORD_COUNT
        locator = base + dump_contract.GPU_TRACE_RECORD_OFFSETS["subtitle_locator_begin"]
        condition = base + dump_contract.GPU_TRACE_RECORD_OFFSETS["subtitle_condition_begin"]

        inactive = words.copy()
        for index in (2, 3, 4, 12, 18, 19, 20, 21, 24, 25,
                      dump_contract.SUBTITLE_LOCATOR_KIND_WORD):
            inactive[locator + index] = 0
        inactive[condition:condition + dump_contract.GPU_TRACE_CONDITION_WORD_COUNT] = [0] * 6
        inactive_payload = struct.pack(f"<{len(inactive)}I", *inactive)
        decoded = dump_contract.validate_gpu_trace_ring(
            inactive_payload,
            matched_frame_id=41,
            analysis_generation=0,
            source_width=1920,
            source_height=1080,
            field_width=16,
            field_height=12,
            expected_domain_tag=domain_tag,
            input_domain_reset=False,
        )
        summary = decoded["decoded"]["records"][0]["subtitle_condition"]
        self.assertFalse(summary["active"])
        self.assertFalse(summary["valid"])

        stale = inactive.copy()
        stale[condition] = coordinate.SUBTITLE_OCR.condition_param_schema
        with self.assertRaisesRegex(ValueError, "condition params disagree"):
            dump_contract.validate_gpu_trace_ring(
                struct.pack(f"<{len(stale)}I", *stale),
                matched_frame_id=41, analysis_generation=0,
                source_width=1920, source_height=1080,
                field_width=16, field_height=12,
                expected_domain_tag=domain_tag, input_domain_reset=False)

        missing_active = words.copy()
        missing_active[condition:condition + dump_contract.GPU_TRACE_CONDITION_WORD_COUNT] = [0] * 6
        with self.assertRaisesRegex(ValueError, "condition params disagree"):
            dump_contract.validate_gpu_trace_ring(
                struct.pack(f"<{len(missing_active)}I", *missing_active),
                matched_frame_id=41, analysis_generation=0,
                source_width=1920, source_height=1080,
                field_width=16, field_height=12,
                expected_domain_tag=domain_tag, input_domain_reset=False)

    def test_gpu_trace_provisional_condition_uses_ephemeral_tuple(self):
        input_region = copy.deepcopy(self.depth_input_region)
        input_region["analysis"]["tensor_extent_px"] = {"width": 16, "height": 12}
        input_region["analysis"]["tensor_content_rect_px"] = {
            "left": 0, "top": 0, "right": 16, "bottom": 12}
        raw, _, domain_tag = self._gpu_trace_ring_payload(input_region, 16, 12)
        words = list(struct.unpack(
            f"<{dump_contract.GPU_TRACE_RING_WORD_COUNT}I", raw))
        base = dump_contract.GPU_TRACE_HEADER_WORD_COUNT
        locator = base + dump_contract.GPU_TRACE_RECORD_OFFSETS["subtitle_locator_begin"]
        condition = base + dump_contract.GPU_TRACE_RECORD_OFFSETS["subtitle_condition_begin"]
        owner = [2, 2, 12, 4]
        pending = [6, 2, 8, 4]
        cover = [5, 1, 9, 5]
        provisional_target = struct.unpack("<I", struct.pack("<f", 0.005))[0]
        words[locator + 2] = (
            dump_contract.SUBTITLE_LOCATOR_FLAG_OWNER |
            dump_contract.SUBTITLE_LOCATOR_FLAG_PENDING |
            dump_contract.SUBTITLE_LOCATOR_FLAG_TARGET_VALID |
            dump_contract.SUBTITLE_LOCATOR_FLAG_PROVISIONAL_CURRENT)
        words[locator + 12] = 1
        words[locator + 20] = 1
        words[locator + 21] = dump_contract.SUBTITLE_LOCATOR_EVENT_NONE
        words[locator + 24] = 2
        words[locator + dump_contract.SUBTITLE_LOCATOR_PROVISIONAL_TARGET_WORD] = (
            provisional_target)
        words[locator + dump_contract.SUBTITLE_LOCATOR_PROVISIONAL_FADE_WORD] = 2
        words[locator + dump_contract.SUBTITLE_LOCATOR_OWNER_WORD_OFFSET:
              locator + dump_contract.SUBTITLE_LOCATOR_OWNER_WORD_OFFSET + 4] = owner
        words[locator + dump_contract.SUBTITLE_LOCATOR_PENDING_WORD_OFFSET:
              locator + dump_contract.SUBTITLE_LOCATOR_PENDING_WORD_OFFSET + 4] = pending
        words[locator + dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET:
              locator + dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET + 4] = cover
        words[condition + 4] = 2
        words[condition + 5] = provisional_target
        payload = struct.pack(f"<{len(words)}I", *words)
        decoded = dump_contract.validate_gpu_trace_ring(
            payload, matched_frame_id=41, analysis_generation=0,
            source_width=1920, source_height=1080,
            field_width=16, field_height=12,
            expected_domain_tag=domain_tag, input_domain_reset=False)
        self.assertEqual(
            decoded["decoded"]["schema"],
            dump_contract.GPU_TRACE_DECODED_SCHEMA)
        summary = decoded["decoded"]["records"][0]
        self.assertTrue(summary["subtitle_locator"]["flags"]["provisional_current"])
        self.assertEqual(summary["subtitle_locator"]["provisional_fade_step"], 2)
        self.assertEqual(summary["subtitle_condition"]["target_bits"], provisional_target)

        durable_condition = words.copy()
        durable_condition[condition + 5] = durable_condition[locator + 18]
        with self.assertRaisesRegex(ValueError, "condition params disagree"):
            dump_contract.validate_gpu_trace_ring(
                struct.pack(f"<{len(durable_condition)}I", *durable_condition),
                matched_frame_id=41, analysis_generation=0,
                source_width=1920, source_height=1080,
                field_width=16, field_height=12,
                expected_domain_tag=domain_tag, input_domain_reset=False)

        invalid_cover = words.copy()
        invalid_cover[locator + dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET] = 7
        with self.assertRaisesRegex(ValueError, "provisional"):
            dump_contract.validate_gpu_trace_ring(
                struct.pack(f"<{len(invalid_cover)}I", *invalid_cover),
                matched_frame_id=41, analysis_generation=0,
                source_width=1920, source_height=1080,
                field_width=16, field_height=12,
                expected_domain_tag=domain_tag, input_domain_reset=False)

        iou_equality = words.copy()
        equal_owner = [1, 2, 5, 4]
        equal_pending = [2, 2, 6, 4]
        equal_cover = [1, 1, 7, 5]
        iou_equality[locator + dump_contract.SUBTITLE_LOCATOR_OWNER_WORD_OFFSET:
                     locator + dump_contract.SUBTITLE_LOCATOR_OWNER_WORD_OFFSET + 4] = equal_owner
        iou_equality[locator + dump_contract.SUBTITLE_LOCATOR_PENDING_WORD_OFFSET:
                     locator + dump_contract.SUBTITLE_LOCATOR_PENDING_WORD_OFFSET + 4] = equal_pending
        iou_equality[locator + dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET:
                     locator + dump_contract.SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET + 4] = equal_cover
        with self.assertRaisesRegex(ValueError, "provisional"):
            dump_contract.validate_gpu_trace_ring(
                struct.pack(f"<{len(iou_equality)}I", *iou_equality),
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

    def test_geometry_verifier_rejects_ownership_lowering(self):
        import hashlib

        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, fields = self._write_synthetic_geometry_dump(root)
            lowered = fields["shadow_ownership_refined_parallax"].copy()
            lowered[2, 2] = np.float32(lowered[2, 2] - 0.001)
            payload = lowered.astype("<f4").tobytes()
            (root / "shadow_ownership_refined_parallax.f32").write_bytes(payload)
            manifest["artifacts"]["shadow_ownership_refined_parallax.f32"]["sha256"] = (
                hashlib.sha256(payload).hexdigest())
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lowered the candidate"):
                dump_contract.verify_v2_dump_geometry(root)

    def test_manifest_requires_the_full_resolution_ownership_intermediate(self):
        changed = copy.deepcopy(self.manifest)
        changed["artifacts"].pop("shadow_ownership_refined_parallax.f32")
        with self.assertRaisesRegex(ValueError, "geometry artifact"):
            dump_contract.validate_v2_dump_manifest_document(changed)

        changed = copy.deepcopy(self.manifest)
        changed["dimensions"]["shadow_ownership_refined_parallax"] = None
        with self.assertRaisesRegex(ValueError, "geometry dimension"):
            dump_contract.validate_v2_dump_manifest_document(changed)

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
                "parallax_v2_coordinate_role",
                "parallax_v2_ownership_refined_role"):
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
                "gain": "one-eye full-source-U per curve unit",
                "parallax": "signed one-eye full-source-U",
            },
            {
                "coordinate": "dimensionless canonical coordinate derived from raw depth",
                "gain": "one-eye ROI-local source-U per curve unit",
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
        with self.assertRaisesRegex(ValueError, "state summary"):
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
        self._set_state_word(changed, "mapping_state_reserved_1", 1)
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
