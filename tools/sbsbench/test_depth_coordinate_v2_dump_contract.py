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
    _inspect_optional_shadow_state,
    _inspect_optional_v2_dump_manifest,
    _require_supported_replay_domain,
)


class DepthCoordinateV2DumpContractTests(unittest.TestCase):
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
            "shadow_candidate_parallax.f32":
                ("parallax-v2 pre-limiter candidate displacement", True),
            "shadow_ownership_refined_parallax.f32":
                ("parallax-v2 full-resolution contour ownership refinement", True),
            "shadow_ownership_refined_parallax_shape.json":
                ("parallax-v2 full-resolution contour ownership refinement contract", False),
            "shadow_ownership_refined_parallax.png":
                ("parallax-v2 full-resolution contour ownership refinement preview", False),
            "shadow_ownership_refined_parallax_heat.png":
                ("parallax-v2 full-resolution contour ownership refinement preview", False),
            "shadow_vertical_majorant.f32":
                ("parallax-v2 vertical shear-limiter intermediate", False),
            "shadow_vertical_majorant_shape.json":
                ("parallax-v2 vertical shear-limiter intermediate contract", False),
            "shadow_vertical_majorant.png":
                ("parallax-v2 vertical shear-limiter intermediate preview", False),
            "shadow_vertical_majorant_heat.png":
                ("parallax-v2 vertical shear-limiter intermediate preview", False),
            "shadow_vertical_conditioned.f32":
                ("parallax-v2 orientation-selective vertical conditioner", False),
            "shadow_vertical_conditioned_shape.json":
                ("parallax-v2 orientation-selective vertical conditioner contract", False),
            "shadow_vertical_conditioned.png":
                ("parallax-v2 orientation-selective vertical conditioner preview", False),
            "shadow_vertical_conditioned_heat.png":
                ("parallax-v2 orientation-selective vertical conditioner preview", False),
            "shadow_final_parallax.f32":
                ("parallax-v2 final conditioned displacement field", True),
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
            "subtitle_conditioning": subtitle_none,
            "renderer": {
                "authority":
                    "authenticated-parallax-v2-orientation-selective-conditioned-field",
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
                    "adjacent-row source-U change <= max_vertical_shear/target_width; "
                    "diagnostic evidence only",
                "parallax_v2_vertical_conditioned_role":
                    "fixed 75/25 share of column upper/lower envelopes; may raise or lower "
                    "candidate and feeds the row majorant",
                "parallax_v2_conditioner_role":
                    "least row-wise q >= shadow_vertical_conditioned with horizontal slope <= "
                    "max_horizontal_slope and vertical shear <= max_vertical_shear; q may "
                    "raise or lower candidate and is the live position authority",
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
                "warp_depth": {
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
                    "Spatially exact full-source or cropped color input submitted to the "
                    "calibrated preprocess; transfer-aware PNG is diagnostic only and never "
                    "numeric model authority."),
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

        native = (REPO / "src" / "platform" / "windows" /
                  "sbs_debug_dump.cpp").read_text(encoding="utf-8")
        self.assertIn(
            f'{{"schema", {dump_contract.DUMP_MANIFEST_SCHEMA}}}', native)
        self.assertIn('"shadow_ownership_refined_parallax"', native)
        self.assertIn("completed.shadow_ownership_refined_parallax", native)
        self.assertIn('"shadow_vertical_majorant"', native)
        self.assertIn("completed.shadow_vertical_majorant", native)
        self.assertIn('"shadow_vertical_conditioned"', native)
        self.assertIn("completed.shadow_vertical_conditioned", native)
        self.assertIn(
            'artifacts["shadow_state.json"] = hashed_artifact_description(', native)
        self.assertIn(
            'artifacts["shadow_frame_stats.json"] = hashed_artifact_description(', native)
        self.assertIn(
            'models::file_sha256_hex(paths.temporary / "shadow_state.json")', native)
        self.assertIn(
            'models::file_sha256_hex(paths.temporary / "shadow_frame_stats.json")', native)
        self.assertIn('{"requested", false}', native)
        self.assertIn('{"rendered_output_selected", true}', native)

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
        self.assertEqual(dump_contract.DUMP_MANIFEST_SCHEMA, 29)
        self.assertEqual(dump_contract.SUBTITLE_OCR_RECORD_SCHEMA, 3)
        self.assertEqual(dump_contract.SUBTITLE_LOCATOR_STATE_SCHEMA, 9)
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
            struct.pack("<I", dump_contract.SUBTITLE_LOCATOR_STATE_TAG), b"SLR9")

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

    def _active_slr9_manifest(self, base_manifest=None):
        manifest = copy.deepcopy(
            self.manifest if base_manifest is None else base_manifest)
        subtitle = {
            "mode": "subtitle-slr9",
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
                "stage": "same-frame OCR8 subtitle boxes",
                "description": "authenticated test OCR8 record",
                "sha256": "0" * 64,
            },
            "subtitle_locator_state.u32": {
                "available": True,
                "required": True,
                "stage": "compact SLR9 subtitle authority state",
                "description": "authenticated test SLR9 state",
                "sha256": "0" * 64,
            },
            "shadow_base_final_parallax.f32": {
                "available": True,
                "required": True,
                "stage": "ordinary post-limiter V2 field before SLR9 conditioning",
                "description": "authenticated test unconditioned base field",
                "sha256": "0" * 64,
            },
        })
        manifest["renderer"]["parallax_v2_conditioner_role"] = (
            "least row-wise q >= shadow_vertical_conditioned with horizontal slope <= "
            "max_horizontal_slope and vertical shear <= max_vertical_shear produces "
            "shadow_base_final_parallax; SLR9 applies the analytic anisotropic rectangle "
            "budget/fade from same-frame current authority and publishes "
            "shadow_final_parallax as live position authority")
        payload = json.dumps(subtitle).encode("utf-8")
        manifest["artifacts"]["subtitle_conditioning.json"]["sha256"] = (
            hashlib.sha256(payload).hexdigest())
        return manifest

    def test_current_slr9_manifest_binds_exact_model_shader_and_artifact_roles(self):
        manifest = self._active_slr9_manifest()
        decoded = dump_contract.validate_v2_dump_manifest_document(manifest)
        subtitle = decoded["subtitle_conditioning"]
        self.assertEqual(subtitle["mode"], "subtitle-slr9")
        self.assertTrue(subtitle["live"])
        self.assertTrue(subtitle["subtitle_evidence_complete"])
        self.assertEqual(
            subtitle["artifact_files"]["conditioned_field"],
            "shadow_final_parallax.f32")
        producer = manifest["subtitle_conditioning"]["producer"]
        self.assertEqual(producer["contract_schema"], coordinate.SUBTITLE_OCR.schema)
        self.assertEqual(set(producer["model"]), {
            "name", "asset_path", "artifact_onnx_sha256", "source_url",
            "source_onnx_sha256", "conversion_tool", "conversion_version",
            "conversion_recipe", "conversion_calibration_profile", "engine_recipe",
            "preprocess_profile", "source_crop", "input", "output",
        })
        native = (REPO / "src" / "platform" / "windows" /
                  "sbs_debug_dump.cpp").read_text(encoding="utf-8")
        for token in (
                '"subtitle-slr9"', '"subtitle_ocr_record.u32"',
                '"subtitle_locator_state.u32"',
                '"shadow_base_final_parallax.f32"',
                "subtitle_ocr_artifact_onnx_sha256",
                "subtitle_ocr_source_onnx_sha256",
                "subtitle_ocr_conversion_recipe",
                "subtitle_ocr_record_word_count",
                "subtitle_locator_state_word_count"):
            self.assertIn(token, native)

    def test_current_slr9_manifest_rejects_provenance_roles_and_base_field_drift(self):
        mutations = {
            "retired-mode": (
                lambda manifest: manifest["subtitle_conditioning"].update(
                    {"mode": "subtitle-slr8"}),
                "unsupported subtitle-conditioning authority"),
            "request": (
                lambda manifest: manifest["subtitle_conditioning"].update({"request": False}),
                "enabled request"),
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
        }
        for name, (mutate, error) in mutations.items():
            with self.subTest(name=name):
                manifest = self._active_slr9_manifest()
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

    def test_current_ocr8_slr9_empty_records_accept_all_calibrated_fields(self):
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

    def test_ocr8_and_slr9_project_and_confine_geometry_to_tensor_content(self):
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

        target_bits = struct.unpack("<I", struct.pack("<f", 0.0075))[0]
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

        target_bits = struct.unpack("<I", struct.pack("<f", 0.0075))[0]

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
    def _valid_slr9_state_words():
        words = [0] * dump_contract.SUBTITLE_LOCATOR_STATE_WORD_COUNT
        target_bits = struct.unpack("<I", struct.pack("<f", 0.0075))[0]
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

    def test_current_slr9_state_validates_identity_rectangles_and_target(self):
        decoded = dump_contract.validate_subtitle_locator_state(
            self._pack_uint32_words(self._valid_slr9_state_words()),
            matched_frame_id=41,
            analysis_generation=17,
            source_width=1920,
            source_height=1080,
            field_width=770,
            field_height=434)
        self.assertEqual(decoded["owner_count"], 1)
        self.assertEqual(decoded["current_count"], 1)
        self.assertEqual(decoded["target"], struct.unpack("<f", struct.pack("<f", 0.0075))[0])
        self.assertEqual(decoded["last_event"], dump_contract.SUBTITLE_LOCATOR_EVENT_BIRTH)
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
        cached_bits = struct.unpack("<I", struct.pack("<f", 0.006))[0]
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

    def test_current_slr9_state_rejects_identity_flags_slots_and_aggregates(self):
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
            "unknown-flags": (lambda words: words.__setitem__(2, 0x10), "unknown flags"),
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
            "owner-grace": (
                lambda words: words.__setitem__(25, 1),
                "live owner cannot retain"),
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
                words = self._valid_slr9_state_words()
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

    def test_native_slr9_prevalidation_mirrors_python_mutation_invariants(self):
        native = (REPO / "src" / "platform" / "windows" /
                  "sbs_debug_dump.cpp").read_text(encoding="utf-8")
        start = native.index("bool subtitle_locator_state_is_canonical(")
        end = native.index("bool subtitle_records_match_frame(", start)
        validator = native[start:end]
        for token in (
                "subtitle_rectangle_summary_matches(locator, 5u, 9u, owner_summary)",
                "subtitle_rectangle_summary_matches(locator, 13u, 17u, pending_summary)",
                "owner != (owner_count != 0u)",
                "pending != (pending_count != 0u)",
                "owner != (owner_generation != 0u)",
                "current_count > owner_count",
                "fade > subtitle_locator_max_fade",
                "last_event > subtitle_locator_max_event",
                "grace > subtitle_locator_death_grace_observations",
                "target_generation != owner_generation",
                "grace != 0u || !packed_grace_zero",
                "ocr_flags == 0u && current_count != 0u",
                "subtitle_current_rectangles_match_ocr_final"):
            with self.subTest(token=token):
                self.assertIn(token, validator)

        membership_start = native.index(
            "bool subtitle_current_rectangles_match_ocr_final(")
        membership = native[membership_start:start]
        self.assertIn("subtitle_ocr_final_box_offset", membership)
        self.assertIn("subtitle_locator_current_offset", membership)
        for component in range(4):
            suffix = "" if component == 0 else f" + {component}u"
            self.assertIn(
                f"subtitle_word(locator, current{suffix}) == "
                f"subtitle_word(ocr, final{suffix})",
                membership)

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
            "crop_method": "same-format D3D11 CopySubresourceRegion",
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

    def test_manifest_requires_content_hashes_for_geometry_fields(self):
        for field in ("shadow_candidate_parallax.f32", "shadow_final_parallax.f32"):
            changed = copy.deepcopy(self.manifest)
            del changed["artifacts"][field]["sha256"]
            with self.assertRaisesRegex(ValueError, "geometry artifact"):
                dump_contract.validate_v2_dump_manifest_document(changed)
            changed = copy.deepcopy(self.manifest)
            changed["artifacts"][field]["sha256"] = "not-a-hash"
            with self.assertRaisesRegex(ValueError, "content sha256"):
                dump_contract.validate_v2_dump_manifest_document(changed)
        # Preview and shape descriptors must not silently grow a hash claim.
        changed = copy.deepcopy(self.manifest)
        changed["artifacts"]["shadow_vertical_majorant.png"] = dict(
            changed["artifacts"]["shadow_vertical_majorant.png"], sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "geometry artifact"):
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

    def _write_synthetic_geometry_dump(self, root, width=16, height=12):
        import numpy as np

        rng = np.random.default_rng(20260804)
        candidate = (rng.uniform(-0.002, 0.03, (height, width))).astype(np.float32)
        ownership = candidate.copy()
        ownership[4, 7] = np.float32(ownership[4, 7] + 0.005)  # raise-only refinement

        defaults = coordinate.CALIBRATED_DEFAULTS
        vertical_step = np.float32(defaults.max_vertical_shear / width)
        horizontal_step = np.float32(defaults.max_horizontal_slope / width)
        share = np.float32(defaults.vertical_majorant_share)
        majorant = ownership.copy()
        for row in range(1, height):
            majorant[row] = np.maximum(majorant[row], majorant[row - 1] - vertical_step)
        for row in range(height - 2, -1, -1):
            majorant[row] = np.maximum(majorant[row], majorant[row + 1] - vertical_step)
        minorant = ownership.copy()
        for row in range(1, height):
            minorant[row] = np.minimum(ownership[row], minorant[row - 1] + vertical_step)
        for row in range(height - 2, -1, -1):
            minorant[row] = np.minimum(minorant[row], minorant[row + 1] + vertical_step)
        conditioned = (share * majorant + np.float32(1.0 - float(share)) * minorant
                       ).astype(np.float32)
        final = conditioned.copy()
        for col in range(1, width):
            final[:, col] = np.maximum(final[:, col], final[:, col - 1] - horizontal_step)
        for col in range(width - 2, -1, -1):
            final[:, col] = np.maximum(final[:, col], final[:, col + 1] - horizontal_step)

        manifest = copy.deepcopy(self.manifest)
        for name in (
                "model_input", "raw_depth", "warp_depth",
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
        warp_payload = final.astype("<f4").tobytes()
        (root / "warp_depth.f32").write_bytes(warp_payload)
        manifest["artifacts"]["warp_depth.f32"] = {
            "available": True,
            "required": True,
            "stage": "actual orientation-selective conditioned field sampled by live V2 reprojection",
            "description": "authenticated test warp field",
            "sha256": hashlib.sha256(warp_payload).hexdigest(),
        }
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

    def _activate_synthetic_slr9_dump(self, root, manifest, fields):
        manifest = self._active_slr9_manifest(manifest)
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

    def _activate_synthetic_slr9_current_dump(
            self, root, manifest, fields, *, fade=1):
        manifest = self._activate_synthetic_slr9_dump(root, manifest, fields)
        ocr = self._valid_ocr_record_words()
        ocr[7] = 0
        ocr[8] = 0
        self._write_hashed_payload(
            root, manifest, "subtitle_ocr_record.u32",
            self._pack_uint32_words(ocr))

        locator = self._valid_slr9_state_words()
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
        conditioned = dump_contract._replay_slr9_conditioner(
            fields["shadow_final_parallax"], subtitle)
        payload = conditioned.astype("<f4").tobytes()
        self._write_hashed_payload(
            root, manifest, "shadow_final_parallax.f32", payload)
        self._write_hashed_payload(root, manifest, "warp_depth.f32", payload)
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
                "authenticated crop-local parallax-v2 conditioned field plus depth-input-region embedding",
            "mapping_artifacts_match_selected_renderer": True,
            "parallax_v2_position_field":
                "shadow_final_parallax + depth_input_region embedding",
            "parallax_v2_ownership_refined_role":
                "conservative full-resolution crop-local source-contour foreground ownership "
                "applied to candidate before the vertical conditioner; may only raise uniquely "
                "owned far-side boundary texels",
            "parallax_v2_conditioner_role":
                "least row-wise crop-local q >= shadow_vertical_conditioned with horizontal "
                "slope <= max_horizontal_slope and vertical shear <= max_vertical_shear; q "
                "plus depth_input_region embedding is live position authority",
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
        })
        for name in (
                "model_input", "raw_depth", "warp_depth",
                "shadow_candidate_parallax", "shadow_ownership_refined_parallax",
                "shadow_vertical_majorant", "shadow_vertical_conditioned",
                "shadow_final_parallax"):
            manifest["dimensions"][name] = dict(
                manifest["dimensions"][name],
                width=tensor_width, height=tensor_height)

        constant = np.full(
            (tensor_height, tensor_width), 0.01 if near_full else 0.001,
            dtype=np.float32)
        fields = {
            "shadow_candidate_parallax": constant,
            "shadow_ownership_refined_parallax": constant,
            "shadow_vertical_majorant": constant,
            "shadow_vertical_conditioned": constant,
            "shadow_final_parallax": constant,
        }
        for name, values in fields.items():
            payload = values.astype("<f4").tobytes()
            (root / f"{name}.f32").write_bytes(payload)
            manifest["artifacts"][f"{name}.f32"]["sha256"] = hashlib.sha256(
                payload).hexdigest()
        warp_depth_payload = constant.astype("<f4").tobytes()
        (root / "warp_depth.f32").write_bytes(warp_depth_payload)
        manifest["artifacts"]["warp_depth.f32"] = {
            "available": True,
            "required": True,
            "stage":
                "crop-local orientation-selective conditioned field embedded by live V2 reprojection",
            "description": "authenticated test crop-local warp field",
            "sha256": hashlib.sha256(warp_depth_payload).hexdigest(),
        }

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
            "crop_method": "same-format D3D11 CopySubresourceRegion",
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
        manifest["artifacts"]["warp_map_shape.json"] = {
            "available": True,
            "required": True,
            "stage": "inverse-warp mapping contract",
            "description": "authenticated test map shape",
        }
        shape = {
            "schema": 2,
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
            "displacement_preview": {
                "file": "warp_displacement_heat.png",
                "range_px": [-1.0, 1.0],
                "normalization": "symmetric finite-content p98 absolute displacement",
                "negative": "blue",
                "zero": "green",
                "positive": "red",
                "bars": "black",
                "nonfinite": "magenta",
            },
        }
        (root / "warp_map_shape.json").write_text(
            json.dumps(shape), encoding="utf-8")
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

    def test_geometry_verifier_accepts_current_slr9_empty_authority_as_exact_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, fields = self._write_synthetic_geometry_dump(
                root,
                width=770,
                height=434)
            self._activate_synthetic_slr9_dump(root, manifest, fields)

            summary = dump_contract.verify_v2_dump_geometry(root)

            subtitle = summary["subtitle_conditioning"]
            self.assertEqual(subtitle["mode"], "subtitle-slr9")
            self.assertTrue(subtitle["subtitle_evidence_verified"])
            self.assertTrue(subtitle["ocr_authoritative"])
            self.assertEqual(subtitle["current_count"], 0)
            self.assertIn(
                "shadow_base_final_parallax",
                summary["chain_fields_verified"])

    def test_geometry_verifier_accepts_abstaining_ocr_with_slr9_target_grace_as_exact_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, fields = self._write_synthetic_geometry_dump(
                root,
                width=770,
                height=434)
            manifest = self._activate_synthetic_slr9_dump(root, manifest, fields)

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
            locator[18] = struct.unpack("<I", struct.pack("<f", 0.006))[0]
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
            self.assertAlmostEqual(subtitle["cached_target"], 0.006)
            self.assertEqual(subtitle["grace_bounds"], {
                "left": 120, "top": 350, "right": 650, "bottom": 401,
            })

    def test_geometry_verifier_rejects_slr9_record_identity_and_nonbase_empty_output(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, fields = self._write_synthetic_geometry_dump(
                root,
                width=770,
                height=434)
            manifest = self._activate_synthetic_slr9_dump(root, manifest, fields)

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
            self._write_hashed_payload(
                root, manifest, "warp_depth.f32",
                changed.astype("<f4").tobytes())
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not the exact content-clamped Base"):
                dump_contract.verify_v2_dump_geometry(root)

    def test_geometry_verifier_replays_nonempty_slr9_rectangle_fade_and_ocr_binding(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, fields = self._write_synthetic_geometry_dump(
                root,
                width=770,
                height=434)
            manifest, fade_one, locator, ocr = (
                self._activate_synthetic_slr9_current_dump(
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
            fade_two = dump_contract._replay_slr9_conditioner(base, {
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
            self._write_hashed_payload(
                root, manifest, "warp_depth.f32", fade_two_payload)
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
            self._write_hashed_payload(
                root, manifest, "warp_depth.f32", tampered_payload)
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact SLR9 rectangle-conditioning"):
                dump_contract.verify_v2_dump_geometry(root)

            self._write_hashed_payload(
                root, manifest, "shadow_final_parallax.f32", fade_two_payload)
            self._write_hashed_payload(
                root, manifest, "warp_depth.f32", fade_two_payload)
            final = dump_contract.SUBTITLE_OCR_FINAL_BOX_WORD_OFFSET
            ocr[final] += 1
            self._write_hashed_payload(
                root, manifest, "subtitle_ocr_record.u32",
                self._pack_uint32_words(ocr))
            (root / "dump_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not contain|not exact"):
                dump_contract.verify_v2_dump_geometry(root)

    def test_slr9_sm5_replay_accepts_the_roi_width_1101_division_bit(self):
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
            for replay in dump_contract._replay_slr9_conditioner_sm5_candidates(
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
            for replay in dump_contract._replay_slr9_conditioner_sm5_candidates(
                base, subtitle)
        }
        # 0x3A742C0A is the production NVIDIA field bit from the supplied ROI dump;
        # 0x3A742C0B is the correctly rounded WARP/NumPy alternative.
        self.assertEqual(replay_bits, {0x3A742C0A, 0x3A742C0B})

    def test_slr9_replay_uses_content_width_and_boundary_extends_padding(self):
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
        replay = dump_contract._replay_slr9_conditioner(base, subtitle)
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

    def test_slr9_ribbon_replay_has_only_a_top_edge_collar(self):
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
        ribbon = dump_contract._replay_slr9_conditioner(base, {
            **common,
            "current_rectangles": [{**geometry, "kind": "ribbon", "ribbon": True}],
        })
        ordinary = dump_contract._replay_slr9_conditioner(base, {
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
            vertical_step = np.float32(
                coordinate.CALIBRATED_DEFAULTS.max_vertical_shear / (right - left))
            horizontal_step = np.float32(
                coordinate.CALIBRATED_DEFAULTS.max_horizontal_slope / (right - left))
            share = np.float32(coordinate.CALIBRATED_DEFAULTS.vertical_majorant_share)

            majorant = ownership.copy()
            for row in range(1, height):
                majorant[row] = np.maximum(
                    majorant[row], majorant[row - 1] - vertical_step)
            for row in range(height - 2, -1, -1):
                majorant[row] = np.maximum(
                    majorant[row], majorant[row + 1] - vertical_step)
            minorant = ownership.copy()
            for row in range(1, height):
                minorant[row] = np.minimum(
                    ownership[row], minorant[row - 1] + vertical_step)
            for row in range(height - 2, -1, -1):
                minorant[row] = np.minimum(
                    minorant[row], minorant[row + 1] + vertical_step)
            conditioned = (
                share * majorant + np.float32(1.0 - float(share)) * minorant
            ).astype(np.float32)
            final = conditioned.copy()
            for column in range(1, width):
                final[:, column] = np.maximum(
                    final[:, column], final[:, column - 1] - horizontal_step)
            for column in range(width - 2, -1, -1):
                final[:, column] = np.maximum(
                    final[:, column], final[:, column + 1] - horizontal_step)

            fields = {
                "shadow_candidate_parallax.f32": candidate,
                "shadow_ownership_refined_parallax.f32": ownership,
                "shadow_vertical_majorant.f32": majorant,
                "shadow_vertical_conditioned.f32": conditioned,
                "shadow_final_parallax.f32": final,
                "warp_depth.f32": final,
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
            legacy_step = np.float32(
                coordinate.CALIBRATED_DEFAULTS.max_horizontal_slope / width)
            legacy = conditioned.copy()
            for column in range(1, width):
                legacy[:, column] = np.maximum(
                    legacy[:, column], legacy[:, column - 1] - legacy_step)
            for column in range(width - 2, -1, -1):
                legacy[:, column] = np.maximum(
                    legacy[:, column], legacy[:, column + 1] - legacy_step)
            self.assertFalse(np.array_equal(final, legacy))
            for name in ("shadow_final_parallax.f32", "warp_depth.f32"):
                self._write_hashed_payload(
                    root, manifest, name, legacy.astype("<f4").tobytes())
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
            "region-hash", "border-hash", "map-hash", "map-nonzero",
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
            replay_source = (SCRIPT_DIR / "replay_depth_mapping_v2.py").read_text(
                encoding="utf-8")
            self.assertLess(
                replay_source.index("_require_supported_replay_domain(dump_manifest_capture)"),
                replay_source.index("_new_output_directory(output)"))

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

        native_cache = (REPO / "src" / "host_sbs_shader_cache.h").read_text(
            encoding="utf-8")
        self.assertIn(dump_contract.LIVE_RENDERER_SOURCE_CLOSURE_SHA256, native_cache)
        self.assertIn(dump_contract.DIAGNOSTIC_SOURCE_CLOSURE_SHA256, native_cache)

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

        native = (REPO / "src" / "platform" / "windows" /
                  "sbs_debug_dump.cpp").read_text(encoding="utf-8")
        self.assertIn("state_frame_valid != expected_state_frame_valid", native)

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
        native = (REPO / "src" / "depth_coordinate_v2.h").read_text(encoding="utf-8")
        self.assertIn(
            f"shadow_state_dump_schema = {dump_contract.SHADOW_STATE_DUMP_SCHEMA}u", native)
        self.assertIn(
            "shadow_frame_stats_dump_schema = "
            f"{dump_contract.SHADOW_FRAME_STATS_DUMP_SCHEMA}u", native)

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

        native = (REPO / "src" / "platform" / "windows" /
                  "sbs_debug_dump.cpp").read_text(encoding="utf-8")
        self.assertIn(
            "calibration_revision_is_valid(calibration_revision_value)", native)

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
        with self.assertRaisesRegex(ValueError, "out of range"):
            dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.state)
        changed["named_values"]["container_scale"] = 0.9
        changed["fields"][3]["value"] = 0.9
        changed["decoded"]["container_scale"] = 0.9
        # V2's effective gain is the literal request; attenuation is pointwise in the map.
        # Keep the derived field internally consistent so validation reaches the invalid
        # compatibility-state value rather than failing earlier on a stale V1 derivation.
        changed["decoded"]["effective_gain"] = changed["constants"]["requested_gain"]
        with self.assertRaisesRegex(ValueError, "out of range"):
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

        native = (REPO / "src" / "platform" / "windows" /
                  "sbs_debug_dump.cpp").read_text(encoding="utf-8")
        self.assertIn("camera_center_integrity_is_valid", native)
        self.assertIn("mapping_state_reserved_valid", native)

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
