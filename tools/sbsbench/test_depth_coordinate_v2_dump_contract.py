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
            document["named_values"]["calibration_revision"],
        )
        cls._set_state_word(document, "camera_center_integrity_bits", checksum)
        document["decoded"]["camera_center_integrity_bits"] = checksum

    def setUp(self):
        manifest = coordinate.load_contract()
        tag = generator.contract_tag(manifest)
        defaults = coordinate.CALIBRATED_DEFAULTS
        width, height = coordinate.MODEL_CALIBRATIONS[0].calibrated_input_shapes[0]
        texel_count = width * height
        near_tail_coverage = 0.20
        near_tail_count = int(near_tail_coverage * texel_count)
        blend = ((near_tail_coverage - defaults.near_tail_coverage_low) /
                 (defaults.near_tail_coverage_high - defaults.near_tail_coverage_low))
        dense_weight = blend * blend * (3.0 - 2.0 * blend)
        effective_near_log_tau = (
            defaults.near_log_tau +
            (defaults.near_log_tau_dense - defaults.near_log_tau) * dense_weight)
        values = {
            "center": 2.0,
            "inverse_scale": 2.0,
            "convergence_curve": 0.0,
            "container_scale": 0.9,
            "calibration_revision": 4,
            "frame_valid": 1.0,
            "confirmed_cut_count": 3,
            "contract_tag_bits": tag,
            "latched_near_tail_coverage": near_tail_coverage,
            "effective_near_log_tau": effective_near_log_tau,
            "latched_near_tail_count": near_tail_count,
            "camera_center_integrity_bits": 0,
        }
        values["camera_center_integrity_bits"] = (
            dump_contract.camera_center_integrity_bits(
                values["center"], values["inverse_scale"],
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
        requested_pop = 2.0
        requested_gain = defaults.gain_per_pop * requested_pop
        constants = {
            "raw_coordinate_scale": 0.5,
            "collapse_abs_epsilon": defaults.collapse_abs_epsilon,
            "far_tau": defaults.far_tau,
            "near_log_tau": defaults.near_log_tau,
            "near_tail_probe_u": defaults.near_tail_probe_u,
            "near_tail_coverage_low": defaults.near_tail_coverage_low,
            "near_tail_coverage_high": defaults.near_tail_coverage_high,
            "near_log_tau_dense": defaults.near_log_tau_dense,
            "gain_per_pop": defaults.gain_per_pop,
            "reference_pop_strength": defaults.reference_pop_strength,
            "reference_gain_at_pop_2":
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
                "latched_scale": 0.5,
                "convergence_curve": values["convergence_curve"],
                "container_scale": values["container_scale"],
                "effective_gain": requested_gain * values["container_scale"],
                "latched_near_tail_coverage": values["latched_near_tail_coverage"],
                "effective_near_log_tau": values["effective_near_log_tau"],
                "latched_near_tail_count": values["latched_near_tail_count"],
                "camera_center_integrity_bits":
                    values["camera_center_integrity_bits"],
            },
            "adaptation_semantics": {
                "coordinate":
                    "center-latched-until-cut-fixed-authenticated-scale-retained-across-unusable",
                "convergence_curve":
                    "separately-scene-latched-curve-offset-currently-zero",
                "requested_gain": "immutable-cfg-pop-strength",
                "container_scale":
                    "frame-local-hard-direct-parallax-attenuation-recoverable-next-frame",
                "near_shoulder":
                    "shot-latched-near-tail-coverage-and-effective-tau-reset-on-confirmed-cut",
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
            "renderer": {
                "authority":
                    "authenticated-parallax-v2-orientation-selective-conditioned-field",
                "parallax_v2_render_requested": True,
                "parallax_v2_render_selected": True,
                "mapping_artifacts_match_selected_renderer": False,
                "parallax_v2_position_field": "shadow_final_parallax",
                "parallax_v2_coordinate_role":
                    "shadow_coordinate is diagnostic only; it has no renderer authority",
                "parallax_v2_vertical_majorant_role":
                    "least column-wise upper envelope v+ >= candidate with adjacent-row "
                    "source-U change <= max_vertical_shear/target_width; diagnostic evidence "
                    "only",
                "parallax_v2_vertical_conditioned_role":
                    "fixed 75/25 share of column upper/lower envelopes; may raise or lower "
                    "candidate and feeds the row majorant",
                "parallax_v2_conditioner_role":
                    "least row-wise q >= shadow_vertical_conditioned with horizontal slope <= "
                    "max_horizontal_slope and vertical shear <= max_vertical_shear; q may "
                    "raise or lower candidate and is the live position authority",
                "parallax_v2_inverse":
                    "12-step contractive fixed point; no owner pass or synthetic fill",
                "collar_defocus": {
                    "enabled": True,
                    "role": ("positive conditioner-deviation color-only background defocus; "
                             "geometry is unchanged"),
                    "deviation": ("max(shadow_final_parallax - "
                                  "shadow_candidate_parallax, 0) in source-color pixels at "
                                  "the inverse-warped source coordinate"),
                    "onset_source_px": 4.0,
                    "full_response_source_px": 20.0,
                    "gaussian_sigma_source_px": 6.0,
                    "kernel": ("one-pass 3x3 binomial approximation with "
                               "sqrt(2)-sigma-spaced taps and smoothstep opacity"),
                    "resolution_basis": ("current-source-color-pixels; "
                                         "depth-grid-independent, not "
                                         "stream-resolution-invariant"),
                    "hdr": ("weighted native source values; no clamp, tone map, or gamma "
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
                }
                for name, (stage, required) in vertical_descriptions.items()
            },
        }

    def test_valid_state_and_frame_stats_are_accepted(self):
        decoded = dump_contract.validate_shadow_state_document(self.state)
        self.assertEqual(decoded["calibration_revision"], 4)
        self.assertEqual(decoded["convergence_curve"], 0.0)
        self.assertAlmostEqual(
            decoded["effective_near_log_tau"],
            self.state["named_values"]["effective_near_log_tau"])
        self.assertEqual(set(decoded), {
            "center", "inverse_scale", "convergence_curve", "container_scale",
            "calibration_revision", "frame_valid", "confirmed_cut_count", "contract_tag_bits",
            "latched_near_tail_coverage", "effective_near_log_tau",
            "latched_near_tail_count", "camera_center_integrity_bits",
        })
        stats = dump_contract.validate_shadow_frame_stats_document(self.frame_stats)
        self.assertEqual(stats["valid"], 1.0)

    def test_schema_8_manifest_attributes_vertical_share_then_row_majorant(self):
        decoded = dump_contract.validate_v2_dump_manifest_document(self.manifest)
        self.assertTrue(decoded["active"])
        self.assertTrue(decoded["rendered_output_selected"])
        self.assertTrue(decoded["vertical_majorant_available"])
        self.assertTrue(decoded["vertical_conditioned_available"])
        self.assertEqual(decoded["position_field"], "shadow_final_parallax")
        self.assertFalse(decoded["mapping_artifacts_match_selected_renderer"])

        native = (REPO / "src" / "platform" / "windows" /
                  "sbs_debug_dump.cpp").read_text(encoding="utf-8")
        self.assertIn(
            f'{{"schema", {dump_contract.DUMP_MANIFEST_SCHEMA}}}', native)
        self.assertIn('"shadow_vertical_majorant"', native)
        self.assertIn("completed.shadow_vertical_majorant", native)
        self.assertIn('"shadow_vertical_conditioned"', native)
        self.assertIn("completed.shadow_vertical_conditioned", native)

    def test_manifest_rejects_missing_or_misattributed_vertical_intermediate(self):
        changed = copy.deepcopy(self.manifest)
        changed["artifacts"].pop("shadow_vertical_majorant.f32")
        with self.assertRaisesRegex(ValueError, "conditioner artifact"):
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

    def test_inactive_manifest_requires_unavailable_vertical_artifacts(self):
        inactive = copy.deepcopy(self.manifest)
        inactive["renderer"].update({
            "authority": None,
            "parallax_v2_render_requested": False,
            "parallax_v2_render_selected": False,
            "mapping_artifacts_match_selected_renderer": False,
            "parallax_v2_position_field": None,
            "parallax_v2_coordinate_role": None,
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
                "shadow_candidate_parallax", "shadow_vertical_majorant",
                "shadow_vertical_conditioned",
                "shadow_final_parallax"):
            inactive["dimensions"][name] = None
        for descriptor in inactive["artifacts"].values():
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

    def test_single_dump_replay_validates_the_schema_8_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "dump_manifest.json").write_text(
                json.dumps(self.manifest), encoding="utf-8")

            summary = _inspect_optional_v2_dump_manifest(root)

            self.assertEqual(summary["status"], "validated")
            self.assertEqual(summary["manifest_schema"], 8)
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
        with self.assertRaisesRegex(ValueError, "out of range"):
            dump_contract.validate_shadow_state_document(changed)
        changed = copy.deepcopy(self.state)
        changed["named_values"]["container_scale"] = 1.1
        changed["fields"][3]["value"] = 1.1
        changed["decoded"]["container_scale"] = 1.1
        changed["decoded"]["effective_gain"] = (
            changed["constants"]["requested_gain"] * 1.1)
        with self.assertRaisesRegex(ValueError, "out of range"):
            dump_contract.validate_shadow_state_document(changed)

    def test_near_shoulder_and_center_integrity_are_exact_scene_latched_state(self):
        changed = copy.deepcopy(self.state)
        self._set_state_word(changed, "center", 2.25)
        with self.assertRaisesRegex(ValueError, "center integrity checksum"):
            dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.state)
        self._set_state_word(
            changed, "latched_near_tail_count",
            changed["named_values"]["latched_near_tail_count"] + 1)
        changed["decoded"]["latched_near_tail_count"] += 1
        with self.assertRaisesRegex(ValueError, "inconsistent near shoulder"):
            dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.state)
        wrong_tau = changed["named_values"]["effective_near_log_tau"] + 0.05
        self._set_state_word(changed, "effective_near_log_tau", wrong_tau)
        changed["decoded"]["effective_near_log_tau"] = wrong_tau
        with self.assertRaisesRegex(ValueError, "inconsistent near shoulder"):
            dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.state)
        self._set_state_word(changed, "latched_near_tail_coverage", 1.1)
        changed["decoded"]["latched_near_tail_coverage"] = 1.1
        with self.assertRaisesRegex(ValueError, "inconsistent near shoulder"):
            dump_contract.validate_shadow_state_document(changed)

        changed = copy.deepcopy(self.state)
        changed["constants"]["near_tail_probe_u"] = 1.1
        with self.assertRaisesRegex(ValueError, "generated contract"):
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
        with self.assertRaisesRegex(ValueError, "out of range"):
            dump_contract.validate_shadow_state_document(changed)

        native = (REPO / "src" / "platform" / "windows" /
                  "sbs_debug_dump.cpp").read_text(encoding="utf-8")
        self.assertIn("camera_center_integrity_is_valid", native)
        self.assertIn("near_tail_effective_tau_for_coverage", native)
        self.assertIn('{"near_tail_probe_u", near_tail_probe_u}', native)

    def test_invalid_state_preserves_requested_gain_but_effective_is_zero(self):
        changed = copy.deepcopy(self.state)
        replacements = {
            "center": 0.0, "inverse_scale": 0.0, "convergence_curve": 0.0,
            "container_scale": 1.0, "frame_valid": 0.0,
            "latched_near_tail_coverage": 0.0,
            "effective_near_log_tau": coordinate.CALIBRATED_DEFAULTS.near_log_tau,
            "latched_near_tail_count": 0,
            "camera_center_integrity_bits": dump_contract.camera_center_integrity_bits(
                0.0, 0.0, changed["named_values"]["calibration_revision"]),
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
        changed["decoded"]["latched_near_tail_coverage"] = 0.0
        changed["decoded"]["effective_near_log_tau"] = (
            coordinate.CALIBRATED_DEFAULTS.near_log_tau)
        changed["decoded"]["latched_near_tail_count"] = 0
        changed["decoded"]["camera_center_integrity_bits"] = (
            replacements["camera_center_integrity_bits"])
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


if __name__ == "__main__":
    unittest.main()
