#!/usr/bin/env python3
"""Strict JSON-dump tests for the generated algorithm and independent dump schemas."""

import copy
import json
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
                "coordinate":
                    "immediate-first-usable-center-latched-until-cut-fixed-authenticated-scale-retained-across-unusable",
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
        self.manifest = {
            "schema": dump_contract.DUMP_MANIFEST_SCHEMA,
            "matched_frame_id": 41,
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
                "active": True,
                "rendered_output_selected": True,
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
            "window_video_border": {
                "available": False,
                "artifact": None,
                "observer_status": "no-video",
                "mapping_status": "invalid-video-rect",
                "geometry_authority": False,
                "renderer_authority": False,
            },
        }
        self.manifest["artifacts"].update({
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
            "window_video_border.json": {
                "available": False,
                "required": False,
                "stage": "matched-frame window-video border",
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
                "trimmed_area_fraction": 0.0,
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

    def test_depth_input_region_accepts_real_integer_floor_roi_and_float32_fitter(self):
        region = copy.deepcopy(self.depth_input_region)
        region["mode"] = "video-region"
        region["authorization"] = {
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
            "left": 279, "top": 415, "right": 2810, "bottom": 1842,
        }
        trim = 1.0 - (2531 * 1427) / (2536 * 1427)
        region["analysis"].update({
            "analysis_generation": 17,
            "input_domain_reset": True,
            "trimmed_area_fraction": trim,
            "crop_method": "same-format D3D11 CopySubresourceRegion",
            "scene_analysis_domain": "inference-rectangle-only",
        })
        runtime_scale, vertical_slope = dump_contract._roi_renderer_constants(
            (279, 415, 2810, 1842), 3840, 2160, 770, 434)
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
        self.assertEqual(decoded["inference_width"], 2531)
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

        moved = copy.deepcopy(region)
        moved["coordinate_space"]["inference_rect_px"]["left"] += 1
        moved["coordinate_space"]["inference_rect_px"]["right"] += 1
        with self.assertRaisesRegex(ValueError, "deterministic authenticated inward fit"):
            dump_contract.validate_depth_input_region_document(moved)

    def test_depth_input_region_rejects_forged_tensor_trim_scale_slope_and_identity(self):
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
                    "deterministic authenticated inward fit"),
                "wrong-trim": (
                    lambda value: value["analysis"].update({"trimmed_area_fraction": 0.01}),
                    "inconsistent video-region analysis"),
                "wrong-scale": (
                    lambda value: value["renderer"].update({
                        "full_source_parallax_scale":
                            value["renderer"]["full_source_parallax_scale"] + 1.0e-5}),
                    "inconsistent video-region analysis"),
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

    def _write_synthetic_geometry_dump(self, root):
        import hashlib

        import numpy as np

        width, height = 16, 12
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
        (root / "dump_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        return manifest, fields

    def _write_synthetic_roi_geometry_dump(self, root, *, near_full=False):
        import hashlib

        import numpy as np

        source_width, source_height = ((774, 436) if near_full else (960, 540))
        tensor_width, tensor_height = 770, 434
        if near_full:
            semantic = {"left": 1, "top": 1, "right": 773, "bottom": 435}
            inference = {"left": 2, "top": 1, "right": 772, "bottom": 435}
        else:
            semantic = {"left": 94, "top": 53, "right": 866, "bottom": 487}
            inference = {"left": 95, "top": 53, "right": 865, "bottom": 487}
        trim = np.float32(1.0 - (770 * 434) / (772 * 434)).item()

        manifest = copy.deepcopy(self.manifest)
        manifest["depth_input_region"]["mode"] = "video-region"
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
                "width": 770, "height": 434,
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
            source_width, source_height,
            tensor_width, tensor_height)
        region = copy.deepcopy(self.depth_input_region)
        region.update({
            "mode": "video-region",
            "authorization": {
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
            "trimmed_area_fraction": trim,
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
            "schema": dump_contract.WINDOW_VIDEO_BORDER_SCHEMA,
            "capture":
                "same matched source/color/depth/render frame as the parent Dump 3D package",
            "role":
                "diagnostic-only window-video border evidence; no geometry or renderer authority",
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
                "latest_heartbeat_age_ms_at_capture": 3,
                "maximum_heartbeat_age_ms": 2500,
                "geometry_continuity_ms_at_capture": 1000,
                "source_content_age_ms_at_capture": 3,
                "fresh": True,
                "causal_geometry": True,
            },
        }
        border_payload = json.dumps(border).encode("utf-8")
        (root / "window_video_border.json").write_bytes(border_payload)
        manifest["window_video_border"] = {
            "available": True,
            "artifact": "window_video_border.json",
            "observer_status": "ok",
            "mapping_status": "ok",
            "geometry_authority": False,
            "renderer_authority": False,
        }
        manifest["artifacts"]["window_video_border.json"] = {
            "available": True,
            "required": True,
            "stage": "matched-frame window-video border",
            "description": "authenticated test semantic border",
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
        (root / "dump_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        return manifest, region, warp_map

    @staticmethod
    def _advertise_window_video_border(root, manifest, *, available=True):
        manifest["matched_frame_id"] = 41
        manifest["dimensions"]["source"] = {
            "width": 1920,
            "height": 1080,
            "format": "DXGI_FORMAT_B8G8R8A8_UNORM",
            "format_value": 87,
        }
        manifest["window_video_border"] = {
            "available": available,
            "artifact": "window_video_border.json" if available else None,
            "observer_status": "ok" if available else "no-video",
            "mapping_status": "ok" if available else "invalid-video-rect",
            "geometry_authority": False,
            "renderer_authority": False,
        }
        manifest["artifacts"]["window_video_border.json"] = {
            "available": available,
            "required": False,
            "stage": "matched-frame window-video border",
            "description": "diagnostic test artifact",
        }
        border = {
            "schema": dump_contract.WINDOW_VIDEO_BORDER_SCHEMA,
            "capture":
                "same matched source/color/depth/render frame as the parent Dump 3D package",
            "role":
                "diagnostic-only window-video border evidence; no geometry or renderer authority",
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
                "latest_heartbeat_age_ms_at_capture": 120,
                "maximum_heartbeat_age_ms": 1000,
                "geometry_continuity_ms_at_capture": 5000,
                "source_content_age_ms_at_capture": 1000,
                "fresh": True,
                "causal_geometry": True,
            },
        }
        if available:
            (root / "window_video_border.json").write_text(
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
            self.assertFalse(summary["window_video_border_verified"])

    def test_roi_geometry_verifier_authenticates_region_border_map_and_zero_plane(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_synthetic_roi_geometry_dump(root)
            summary = dump_contract.verify_v2_dump_geometry(root)

            self.assertEqual(summary["depth_input_region"]["mode"], "video-region")
            self.assertTrue(summary["window_video_border_verified"])
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
                (95, 53, 865, 487))

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
                    document = json.loads((root / "window_video_border.json").read_text())
                    if mutation == "semantic-mismatch":
                        document["coordinate_space"]["capture_rect_px"]["right"] -= 1
                    else:
                        document["identity"]["generation"] += 1
                    payload = json.dumps(document).encode("utf-8")
                    (root / "window_video_border.json").write_bytes(payload)
                    expected = ("window_video_border.json content hash mismatch"
                                if mutation == "border-hash" else
                                "authorization disagrees" if mutation == "authorization-mismatch"
                                else "semantic rectangle disagrees")
                    if mutation != "border-hash":
                        manifest["artifacts"]["window_video_border.json"]["sha256"] = (
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

    def test_geometry_verifier_cross_validates_advertised_window_video_border(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self._write_synthetic_geometry_dump(root)
            self._advertise_window_video_border(root, manifest)
            summary = dump_contract.verify_v2_dump_geometry(root)
            self.assertTrue(summary["window_video_border_verified"])
            self.assertEqual(summary["window_video_border"]["matched_frame_id"], 41)
            self.assertEqual(summary["window_video_border"]["right"], 1760)

    def test_geometry_verifier_rejects_missing_malformed_and_mismatched_border(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self._write_synthetic_geometry_dump(root)
            border = self._advertise_window_video_border(root, manifest)

            (root / "window_video_border.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing or malformed"):
                dump_contract.verify_v2_dump_geometry(root)

            (root / "window_video_border.json").write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or malformed"):
                dump_contract.verify_v2_dump_geometry(root)

            changed = copy.deepcopy(border)
            changed["matched_frame_id"] = 42
            (root / "window_video_border.json").write_text(
                json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parent frame"):
                dump_contract.verify_v2_dump_geometry(root)

            changed = copy.deepcopy(border)
            changed["coordinate_space"]["source_extent_px"]["width"] = 1919
            (root / "window_video_border.json").write_text(
                json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source extent"):
                dump_contract.verify_v2_dump_geometry(root)

    def test_unavailable_window_video_border_has_no_package_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self._write_synthetic_geometry_dump(root)
            self._advertise_window_video_border(root, manifest, available=False)
            # An unadvertised leftover is intentionally ignored: only the atomic manifest grants
            # diagnostic-package authority to this optional artifact.
            (root / "window_video_border.json").write_text("{", encoding="utf-8")
            summary = dump_contract.verify_v2_dump_geometry(root)
            self.assertFalse(summary["window_video_border_verified"])
            self.assertIsNone(summary["window_video_border"])

    def test_manifest_rejects_inconsistent_window_video_border_advertisement(self):
        changed = copy.deepcopy(self.manifest)
        changed["window_video_border"] = {
            "available": True,
            "artifact": "window_video_border.json",
            "observer_status": "ok",
            "mapping_status": "ok",
            "geometry_authority": False,
            "renderer_authority": False,
        }
        changed["artifacts"]["window_video_border.json"] = {
            "available": False,
            "required": False,
            "stage": "matched-frame window-video border",
            "description": "diagnostic test artifact",
        }
        with self.assertRaisesRegex(ValueError, "inconsistent window-video border"):
            dump_contract.validate_v2_dump_manifest_document(changed)

        changed = copy.deepcopy(self.manifest)
        changed["window_video_border"] = {
            "available": True,
            "artifact": "window_video_border.json",
            "observer_status": "stale",
            "mapping_status": "ok",
            "geometry_authority": False,
            "renderer_authority": False,
        }
        changed["artifacts"]["window_video_border.json"] = {
            "available": True,
            "required": False,
            "stage": "matched-frame window-video border",
            "description": "diagnostic test artifact",
        }
        with self.assertRaisesRegex(ValueError, "inconsistent window-video border"):
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

    def test_inactive_manifest_requires_unavailable_vertical_artifacts(self):
        inactive = copy.deepcopy(self.manifest)
        inactive["renderer"].update({
            "authority": None,
            "parallax_v2_render_requested": False,
            "parallax_v2_render_selected": False,
            "mapping_artifacts_match_selected_renderer": False,
            "parallax_v2_position_field": None,
            "parallax_v2_coordinate_role": None,
            "parallax_v2_ownership_refined_role": None,
            "parallax_v2_vertical_majorant_role": None,
            "parallax_v2_vertical_conditioned_role": None,
            "parallax_v2_conditioner_role": None,
            "parallax_v2_inverse": None,
            "collar_defocus": None,
            "live_shader_source": None,
        })
        inactive["parallax_v2_shadow"]["active"] = False
        inactive["parallax_v2_shadow"]["rendered_output_selected"] = False
        for name in (
                "shadow_candidate_parallax", "shadow_ownership_refined_parallax",
                "shadow_vertical_majorant",
                "shadow_vertical_conditioned",
                "shadow_final_parallax"):
            inactive["dimensions"][name] = None
        for name, descriptor in inactive["artifacts"].items():
            if name.startswith("shadow_"):
                descriptor["available"] = False
        decoded = dump_contract.validate_v2_dump_manifest_document(inactive)
        self.assertFalse(decoded["vertical_majorant_available"])

    def test_manifest_rejects_selection_or_geometry_dimension_disagreement(self):
        changed = copy.deepcopy(self.manifest)
        changed["parallax_v2_shadow"]["rendered_output_selected"] = False
        with self.assertRaisesRegex(ValueError, "selection flags disagree"):
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


class WindowVideoBorderDumpContractTests(unittest.TestCase):
    def setUp(self):
        self.border = {
            "schema": dump_contract.WINDOW_VIDEO_BORDER_SCHEMA,
            "capture":
                "same matched source/color/depth/render frame as the parent Dump 3D package",
            "role":
                "diagnostic-only window-video border evidence; no geometry or renderer authority",
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
                "latest_heartbeat_age_ms_at_capture": 120,
                "maximum_heartbeat_age_ms": 1000,
                "geometry_continuity_ms_at_capture": 5000,
                "source_content_age_ms_at_capture": 1000,
                "fresh": True,
                "causal_geometry": True,
            },
        }

    def test_valid_border_is_bound_to_parent_frame_and_source_extent(self):
        decoded = dump_contract.validate_window_video_border_document(
            self.border, matched_frame_id=41, source_width=3840, source_height=2160)
        self.assertEqual(decoded["left"], 320)
        self.assertEqual(decoded["right"], 3520)
        self.assertEqual(decoded["hwnd"], 0x1234)

    def test_frame_extent_and_half_open_bounds_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "parent frame"):
            dump_contract.validate_window_video_border_document(
                self.border, matched_frame_id=40, source_width=3840, source_height=2160)
        with self.assertRaisesRegex(ValueError, "source extent"):
            dump_contract.validate_window_video_border_document(
                self.border, matched_frame_id=41, source_width=1920, source_height=1080)
        changed = copy.deepcopy(self.border)
        changed["coordinate_space"]["capture_rect_px"]["right"] = 3841
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            dump_contract.validate_window_video_border_document(changed)

    def test_stale_or_incomplete_identity_fails_closed(self):
        changed = copy.deepcopy(self.border)
        changed["identity"]["video_id"] = 0
        with self.assertRaisesRegex(ValueError, "incomplete identity"):
            dump_contract.validate_window_video_border_document(changed)
        changed = copy.deepcopy(self.border)
        changed["freshness"]["latest_heartbeat_age_ms_at_capture"] = 1001
        with self.assertRaisesRegex(ValueError, "stale"):
            dump_contract.validate_window_video_border_document(changed)
        changed = copy.deepcopy(self.border)
        changed["freshness"]["geometry_continuity_ms_at_capture"] = 999
        with self.assertRaisesRegex(ValueError, "postdates"):
            dump_contract.validate_window_video_border_document(changed)

    def test_unknown_fields_and_authority_claims_are_rejected(self):
        changed = copy.deepcopy(self.border)
        changed["normalized_rect"] = [0.0, 0.0, 1.0, 1.0]
        with self.assertRaisesRegex(ValueError, "unknown layout"):
            dump_contract.validate_window_video_border_document(changed)
        changed = copy.deepcopy(self.border)
        changed["role"] = "renderer authority"
        with self.assertRaisesRegex(ValueError, "unknown authority"):
            dump_contract.validate_window_video_border_document(changed)


if __name__ == "__main__":
    unittest.main()
