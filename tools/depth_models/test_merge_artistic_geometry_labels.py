import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import artistic_geometry_contract as geometry_contract  # noqa: E402
import depth_input_color as input_color  # noqa: E402
import merge_artistic_geometry_labels as merge  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_float_texture(path, width, height, value):
    values = np.full((height, width), value, dtype="<f4")
    Path(path).write_bytes(
        np.asarray([width, height], dtype="<u4").tobytes() + values.tobytes()
    )


class MultiGeometryMergeTests(unittest.TestCase):
    def test_label_fitter_code_roles_are_stage_exact_and_byte_authenticated(self):
        with tempfile.TemporaryDirectory() as directory:
            code_path = Path(directory) / "frozen.py"
            code_path.write_text("# frozen\n", encoding="utf-8")
            identity = {"path": str(code_path), "sha256": digest(code_path)}
            selected = {
                role: dict(identity)
                for role in merge.SELECTED_LABEL_FITTER_CODE_ROLES
            }
            self.assertEqual(
                set(merge.validate_label_fitter_code(
                    selected, merge.SOURCE_LABEL_SCHEMA
                )),
                set(merge.SELECTED_LABEL_FITTER_CODE_ROLES),
            )
            merged = {**selected, "geometry_merge": dict(identity)}
            self.assertEqual(
                set(merge.validate_label_fitter_code(
                    merged, merge.LABEL_SCHEMA
                )),
                set(merge.MERGED_LABEL_FITTER_CODE_ROLES),
            )
            with self.assertRaisesRegex(RuntimeError, "code roles differ"):
                merge.validate_label_fitter_code(selected, merge.LABEL_SCHEMA)
            code_path.write_text("# changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing or changed"):
                merge.validate_label_fitter_code(merged, merge.LABEL_SCHEMA)

    def test_render_summary_uses_output_eye_aspect_not_source_aspect(self):
        raw = np.asarray([[0.0, 0.01]], dtype=np.float32)
        geometry = {
            "source_width": 16,
            "source_height": 9,
            "eye_width": 4,
            "eye_height": 4,
            "content_scale_x": 1.0,
            "content_scale_y": 9.0 / 16.0,
        }

        target = merge.render_target(raw, geometry, 1.0, 0.03)

        expected_scale = (4.0 / 4.0) / (5120.0 / 2160.0)
        self.assertAlmostEqual(
            target["mean_abs_disparity_pct"],
            0.005 * expected_scale * 100.0,
        )

    def make_row(self, root, source, name, eye_width, eye_height,
                 ceiling, reliability, input_variant=None):
        raw = root / f"raw-{name}.f32"
        write_float_texture(raw, eye_width, eye_height, 0.01)
        target = {
            "scale": ceiling,
            "hlsl_full_clamp_abs": 0.03,
            "comfort_clamp_abs_pct": 3.0,
            "mean_abs_disparity_pct": ceiling,
            "p95_abs_disparity_pct": ceiling,
            "exact_pop_spread_pct": ceiling,
            "clamped_pixel_pct": 0.0,
        }
        scale_x, scale_y = geometry_contract.source_content_scales(
            16, 9, eye_width, eye_height
        )
        row = {
            "label_schema": 8,
            "policy_contract": merge.POLICY_CONTRACT,
            "source": str(source), "source_sha256": digest(source),
            "source_width": 16, "source_height": 9,
            "eye_width": eye_width, "eye_height": eye_height,
            "content_scale_x": scale_x, "content_scale_y": scale_y,
            "disparity_raster_width": eye_width,
            "disparity_raster_height": eye_height,
            "baseline_unclamped_disparity": str(raw),
            "baseline_unclamped_disparity_sha256": digest(raw),
            "baseline_disparity": str(raw),
            "baseline_disparity_sha256": digest(raw),
            "artistic_full_clamp_abs": 0.03,
            "safe_scale_min": 0.9,
            "safe_scale_max": ceiling,
            "safe_scale_ceiling": ceiling,
            "baseline_multiplier": ceiling,
            "ceiling_confidence": 1.0,
            "confidence": 1.0,
            "safety_margin_reliability": reliability,
            "render_evidence_confidence": reliability,
            "style_targets": {
                "clean": 1.0,
                "balanced": 1.0 + 0.5 * (ceiling - 1.0),
                "immersive": ceiling,
            },
            "style_render_targets": {},
            "safe_ceiling_render_target": target,
            "safe_ceiling_exact_pop_spread_pct": target[
                "exact_pop_spread_pct"
            ],
            "baseline_disparity_mean_abs_pct": 1.0,
            "baseline_unclamped_disparity_mean_abs_pct": 1.0,
            "render_grid_key": "shot", "clip": "shot", "frame": 0,
            "split": "training", "film_id": "film", "domain": "movie",
            "global_policy_weight": 1.0,
        }
        if input_variant is not None:
            row.update({
                "color_mode": input_variant["color_mode"],
                "input_variant": input_variant,
                "input_variant_sha256":
                    input_color.input_variant_sha256(input_variant),
                "depth_input_color_contract_sha256":
                    input_color.color_contract_sha256(),
            })
            if input_variant["kind"] == input_color.INPUT_KIND_NATIVE_PQ:
                model_source = source.with_suffix(".scrgb16")
                if not model_source.exists():
                    model_source.write_bytes(
                        np.zeros((9, 16, 4), dtype="<f2").tobytes()
                    )
                row.update({
                    "source_kind": "native-hdr-video",
                    "model_source": str(model_source),
                    "model_source_sha256": digest(model_source),
                    "model_source_encoding":
                        merge.native_hdr_capture.CAPTURE_ENCODING,
                })
            else:
                row["source_kind"] = "movie-video"
        return row

    def write_schema8_bundle(self, root, name, rows):
        bundle = root / name
        bundle.mkdir()
        labels = bundle / "labels.jsonl"
        labels.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        fitter = {
            "schema": 8,
            "policy_contract": merge.POLICY_CONTRACT,
            "policy_baseline": {
                "profile": "apollo", "depth_model": "dav2-small",
                "depth_short_side": 432, "depth_max_aspect": 4.0,
            },
            "model_limits": {"scale_delta_max": 0.5},
            "rendered_disparity_supervision": {"artifact": "raw"},
            "label_fitter_config": {
                "candidate_scales": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                "max_candidate_scale_step": 0.1,
                "protected_primary_axes": ["stability", "warp"],
                "protected_metric_reduction": "worst frame",
                "exact_pop_metric": "exact_pop_spread_pct",
                "connected_frontier": "identity connected",
                "confidence_semantics": "hard actionable target",
                "reliability_semantics": "safety margin reliability",
            },
            "code": {},
            "thresholds": {"path": "thresholds", "sha256": "a" * 64},
            "control": {"path": "control", "sha256": "b" * 64},
        }
        fitter_path = bundle / "label_fitter_contract.json"
        fitter_path.write_text(json.dumps(fitter), encoding="utf-8")
        (bundle / "summary.json").write_text(json.dumps({
            "schema": 8,
            "labels_sha256": digest(labels),
            "label_fitter_contract_sha256": digest(fitter_path),
        }), encoding="utf-8")
        return labels

    @staticmethod
    def write_input_manifest(root, variants, name="input-variants.json"):
        path = root / name
        path.write_text(json.dumps(
            merge.build_input_variant_manifest(variants)
        ), encoding="utf-8")
        return path

    def complete_policy_grid(self, root, source, ceilings=None,
                             reliabilities=None):
        """Create the exact SDR-origin 4-condition x 2-geometry input grid."""
        policy_variants = merge.policy_input_variants()
        variants = policy_variants[:-1]
        rows = []
        for geometry_index, (width, height) in enumerate(((4, 2), (8, 4))):
            for condition_index, variant in enumerate(variants):
                index = geometry_index * len(variants) + condition_index
                ceiling = (
                    ceilings[index] if ceilings is not None
                    else 1.3 - 0.01 * index
                )
                reliability = (
                    reliabilities[index] if reliabilities is not None
                    else 0.8 - 0.02 * index
                )
                rows.append(self.make_row(
                    root, source, f"g{geometry_index}-c{condition_index}",
                    width, height, ceiling, reliability, variant,
                ))
        bundles = [
            self.write_schema8_bundle(root, f"grid-{index}", [row])
            for index, row in enumerate(rows)
        ]
        geometry_payload = geometry_contract.build_allowlist([
            geometry_contract.geometry_tuple(row, row["color_mode"])
            for row in rows
        ])
        geometry_path = root / "policy-geometries.json"
        geometry_path.write_text(
            json.dumps(geometry_payload), encoding="utf-8"
        )
        input_path = self.write_input_manifest(root, policy_variants)
        return rows, bundles, geometry_payload, geometry_path, input_path

    def test_intersects_frontiers_and_keeps_every_geometry_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            cv2.imwrite(str(source), np.zeros((9, 16, 3), np.uint8))
            rows, bundles, manifest_payload, manifest, input_path = (
                self.complete_policy_grid(
                    root, source,
                    ceilings=(1.3, 1.4, 1.4, 1.4,
                              1.2, 1.35, 1.35, 1.35),
                    reliabilities=(0.8, 0.9, 0.9, 0.9,
                                   0.6, 0.8, 0.8, 0.8),
                )
            )
            output = root / "merged"

            summary = merge.write_bundle(
                bundles, manifest, output, color_mode=None,
                input_variant_manifest_path=input_path,
            )
            row = json.loads((output / "labels.jsonl").read_text(encoding="utf-8"))
            contract = json.loads(
                (output / "label_fitter_contract.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["schema"], 10)
            self.assertEqual(summary["unique_rgb_count"], 1)
            self.assertEqual(row["safe_scale_ceiling"], 1.2)
            self.assertEqual(row["safe_scale_max"], 1.2)
            self.assertEqual(row["safety_margin_reliability"], 0.6)
            self.assertEqual(row["style_targets"]["balanced"], 1.1)
            self.assertEqual(len(row["deployment_geometry_variants"]), 8)
            self.assertEqual(len(row["input_condition_targets"]), 4)
            condition_targets = {
                target["input_variant_sha256"]: target
                for target in row["input_condition_targets"]
            }
            sdr_hash = input_color.input_variant_sha256(
                input_color.sdr_input_variant()
            )
            self.assertEqual(
                condition_targets[sdr_hash]["safe_scale_ceiling"], 1.2
            )
            self.assertEqual(
                condition_targets[sdr_hash]["safety_margin_reliability"], 0.6
            )
            self.assertEqual(
                condition_targets[sdr_hash]["deployment_geometry_variant_count"], 2
            )
            self.assertTrue(all(
                variant["safe_ceiling_render_target"]["scale"] ==
                condition_targets[variant["input_variant_sha256"]][
                    "safe_scale_ceiling"
                ]
                for variant in row["deployment_geometry_variants"]
            ))
            self.assertEqual(
                contract["deployment_geometry_allowlist"], manifest_payload
            )
            self.assertEqual(
                row["deployment_geometry_allowlist_sha256"],
                geometry_contract.allowlist_sha256(manifest_payload),
            )
            policy_manifest = merge.build_input_variant_manifest(
                merge.policy_input_variants()
            )
            self.assertEqual(row["input_variant_manifest"], policy_manifest)
            self.assertEqual(contract["input_variant_manifest"], policy_manifest)
            self.assertEqual(summary["input_variant_manifest"], policy_manifest)
            self.assertEqual(summary["input_variant_count"], 4)
            self.assertEqual(summary["geometry_input_variant_count"], 8)
            self.assertEqual(
                row["condition_target_contract"],
                merge.CONDITION_TARGET_CONTRACT,
            )

    def test_rejects_missing_geometry_and_duplicate_rgb_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            cv2.imwrite(str(source), np.zeros((9, 16, 3), np.uint8))
            first_row = self.make_row(root, source, "a", 4, 2, 1.2, 0.7)
            second_row = self.make_row(root, source, "b", 8, 4, 1.2, 0.7)
            first = self.write_schema8_bundle(root, "first", [first_row])
            second = self.write_schema8_bundle(root, "second", [second_row])
            incomplete = geometry_contract.build_allowlist([
                geometry_contract.geometry_tuple(first_row)
            ])
            manifest = root / "geometry.json"
            manifest.write_text(json.dumps(incomplete), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exactly native SDR"):
                merge.write_bundle([first, second], manifest, root / "missing")

            duplicate = dict(first_row, clip="another-shot")
            duplicated_bundle = self.write_schema8_bundle(
                root, "duplicated", [first_row, duplicate]
            )
            with self.assertRaisesRegex(
                    RuntimeError, "duplicate identical RGB.*conflicting"):
                merge.row_map(merge.bundle_contract(duplicated_bundle))

            repeated = dict(first_row, frame=4)
            repeated_bundle = self.write_schema8_bundle(
                root, "repeated", [repeated, first_row]
            )
            collapsed = merge.row_map(merge.bundle_contract(repeated_bundle))
            self.assertEqual(len(collapsed), 1)
            self.assertEqual(next(iter(collapsed.values()))["frame"], 0)

    def test_hard_failing_identity_forces_cross_geometry_noop_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            cv2.imwrite(str(source), np.zeros((9, 16, 3), np.uint8))
            rows, _bundles, manifest_payload, manifest, input_path = (
                self.complete_policy_grid(root, source)
            )
            infeasible = rows[4]
            infeasible.update({
                "safe_scale_min": 1.0,
                "safe_scale_max": 1.0,
                "safe_scale_ceiling": 1.0,
                "baseline_multiplier": 1.0,
                "ceiling_confidence": 0.0,
                "confidence": 0.0,
                "safety_margin_reliability": 0.0,
                "render_evidence_confidence": 0.0,
                "style_targets": {
                    "clean": 1.0, "balanced": 1.0, "immersive": 1.0,
                },
                "identity_feasible": False,
                "identity_violations": ["source_coverage_pct:hard"],
                "selection_reason": "identity-hard-failure-nonactionable",
            })
            bundles = [
                self.write_schema8_bundle(root, f"failure-grid-{index}", [item])
                for index, item in enumerate(rows)
            ]
            output = root / "merged"

            merge.write_bundle(
                bundles, manifest, output, color_mode=None,
                input_variant_manifest_path=input_path,
            )
            row = json.loads(
                (output / "labels.jsonl").read_text(encoding="utf-8")
            )
            self.assertFalse(row["identity_feasible"])
            self.assertEqual(row["safe_scale_ceiling"], 1.0)
            self.assertEqual(row["ceiling_confidence"], 0.0)
            self.assertEqual(row["safety_margin_reliability"], 0.0)
            self.assertEqual(row["style_targets"], {
                "clean": 1.0, "balanced": 1.0, "immersive": 1.0,
            })
            self.assertEqual(len(row["identity_infeasible_variants"]), 1)
            self.assertEqual(
                row["identity_infeasible_variants"][0]["violations"],
                ["source_coverage_pct:hard"],
            )
            by_condition = {
                target["input_variant_sha256"]: target
                for target in row["input_condition_targets"]
            }
            sdr_hash = input_color.input_variant_sha256(
                input_color.sdr_input_variant()
            )
            self.assertFalse(by_condition[sdr_hash]["identity_feasible"])
            self.assertEqual(
                by_condition[sdr_hash]["safe_scale_ceiling"], 1.0
            )
            self.assertTrue(all(
                target["safe_scale_ceiling"] > 1.0
                for key, target in by_condition.items() if key != sdr_hash
            ))

    def test_hdr_cross_product_intersects_every_white_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            cv2.imwrite(str(source), np.zeros((9, 16, 3), np.uint8))
            low_white = input_color.windows_hdr_input_variant(1000)
            high_white = input_color.windows_hdr_input_variant(6000)
            rows = [
                self.make_row(
                    root, source, "g1-w1", 4, 2, 1.3, 0.8, low_white
                ),
                self.make_row(
                    root, source, "g1-w6", 4, 2, 1.15, 0.5, high_white
                ),
                self.make_row(
                    root, source, "g2-w1", 8, 4, 1.25, 0.7, low_white
                ),
                self.make_row(
                    root, source, "g2-w6", 8, 4, 1.2, 0.6, high_white
                ),
            ]
            bundles = [
                self.write_schema8_bundle(root, f"bundle-{index}", [row])
                for index, row in enumerate(rows)
            ]
            geometry_payload = geometry_contract.build_allowlist([
                geometry_contract.geometry_tuple(
                    rows[0], input_color.COLOR_MODE_HDR
                ),
                geometry_contract.geometry_tuple(
                    rows[2], input_color.COLOR_MODE_HDR
                ),
            ])
            geometry_path = root / "geometry.json"
            geometry_path.write_text(
                json.dumps(geometry_payload), encoding="utf-8"
            )
            input_path = self.write_input_manifest(
                root, [high_white, low_white]
            )
            output = root / "merged"

            with self.assertRaisesRegex(RuntimeError, "exactly native SDR"):
                merge.write_bundle(
                    bundles, geometry_path, output,
                    color_mode=input_color.COLOR_MODE_HDR,
                    input_variant_manifest_path=input_path,
                )

    def test_hdr_multi_white_requires_manifest_and_complete_cross_product(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            cv2.imwrite(str(source), np.zeros((9, 16, 3), np.uint8))
            low_white = input_color.windows_hdr_input_variant(1000)
            high_white = input_color.windows_hdr_input_variant(6000)
            first_row = self.make_row(
                root, source, "g1-w1", 4, 2, 1.2, 0.7, low_white
            )
            second_row = self.make_row(
                root, source, "g1-w6", 4, 2, 1.2, 0.7, high_white
            )
            first = self.write_schema8_bundle(root, "first", [first_row])
            second = self.write_schema8_bundle(root, "second", [second_row])
            geometry_payload = geometry_contract.build_allowlist([
                geometry_contract.geometry_tuple(
                    first_row, input_color.COLOR_MODE_HDR
                )
            ])
            geometry_path = root / "geometry.json"
            geometry_path.write_text(
                json.dumps(geometry_payload), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                    RuntimeError, "explicit input-variant manifest"):
                merge.write_bundle(
                    [first, second], geometry_path, root / "no-manifest",
                    color_mode=input_color.COLOR_MODE_HDR,
                )

            input_path = self.write_input_manifest(
                root, [low_white, high_white]
            )
            other_geometry = self.make_row(
                root, source, "g2-w1", 8, 4, 1.2, 0.7, low_white
            )
            other_bundle = self.write_schema8_bundle(
                root, "other", [other_geometry]
            )
            complete_geometry = geometry_contract.build_allowlist([
                geometry_contract.geometry_tuple(
                    first_row, input_color.COLOR_MODE_HDR
                ),
                geometry_contract.geometry_tuple(
                    other_geometry, input_color.COLOR_MODE_HDR
                ),
            ])
            complete_geometry_path = root / "complete-geometry.json"
            complete_geometry_path.write_text(
                json.dumps(complete_geometry), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "exactly native SDR"):
                merge.write_bundle(
                    [first, second, other_bundle], complete_geometry_path,
                    root / "incomplete", color_mode=input_color.COLOR_MODE_HDR,
                    input_variant_manifest_path=input_path,
                )

            duplicate = self.write_schema8_bundle(
                root, "duplicate", [dict(first_row)]
            )
            one_variant = self.write_input_manifest(
                root, [low_white], "one-variant.json"
            )
            with self.assertRaisesRegex(RuntimeError, "exactly native SDR"):
                merge.write_bundle(
                    [first, duplicate], geometry_path, root / "duplicate-out",
                    color_mode=input_color.COLOR_MODE_HDR,
                    input_variant_manifest_path=one_variant,
                )

    def test_mixed_sdr_hdr_requires_only_color_compatible_exact_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            cv2.imwrite(str(source), np.zeros((9, 16, 3), np.uint8))
            variants = [
                input_color.sdr_input_variant(),
                input_color.windows_hdr_input_variant(1000),
                input_color.windows_hdr_input_variant(2500),
                input_color.windows_hdr_input_variant(6000),
            ]
            rows = []
            for geometry_index, (width, height) in enumerate(((4, 2), (8, 4))):
                for variant_index, variant in enumerate(variants):
                    rows.append(self.make_row(
                        root, source,
                        f"g{geometry_index}-{variant['kind']}-{variant_index}",
                        width, height,
                        1.3 - 0.01 * (geometry_index * 4 + variant_index),
                        0.8 - 0.02 * (geometry_index * 4 + variant_index),
                        variant,
                    ))
            bundles = [
                self.write_schema8_bundle(root, f"mixed-{index}", [row])
                for index, row in enumerate(rows)
            ]
            geometry_payload = geometry_contract.build_allowlist([
                geometry_contract.geometry_tuple(row, row["color_mode"])
                for row in rows
            ])
            self.assertEqual(len(geometry_payload["tuples"]), 4)
            geometry_path = root / "mixed-geometries.json"
            geometry_path.write_text(
                json.dumps(geometry_payload), encoding="utf-8"
            )
            input_path = self.write_input_manifest(
                root, merge.policy_input_variants()
            )

            output = root / "mixed-output"
            summary = merge.write_bundle(
                bundles, geometry_path, output,
                color_mode=None,
                input_variant_manifest_path=input_path,
            )
            row = json.loads(
                (output / "labels.jsonl").read_text(encoding="utf-8")
            )
            evidence = row["deployment_geometry_variants"]
            self.assertEqual(row["label_schema"], 10)
            self.assertEqual(len(evidence), 8)
            self.assertEqual(len(row["input_condition_targets"]), 4)
            self.assertEqual(summary["geometry_variant_count"], 4)
            self.assertEqual(summary["input_variant_count"], 4)
            self.assertEqual(summary["geometry_input_variant_count"], 8)
            self.assertEqual(
                summary["geometry_input_variant_count_by_color_mode"],
                {
                    input_color.COLOR_MODE_SDR: 2,
                    input_color.COLOR_MODE_HDR: 6,
                },
            )
            self.assertTrue(all(
                item["geometry"]["color_mode"] ==
                item["input_variant"]["color_mode"]
                for item in evidence
            ))
            counts = {}
            for item in evidence:
                key = item["input_variant_sha256"]
                counts[key] = counts.get(key, 0) + 1
            self.assertEqual(set(counts.values()), {2})
            condition_targets = {
                target["input_variant_sha256"]: target
                for target in row["input_condition_targets"]
            }
            self.assertEqual(set(condition_targets), set(counts))
            expected_ceilings = {
                input_color.input_variant_sha256(variant): 1.26 - 0.01 * index
                for index, variant in enumerate(variants)
            }
            for key, target in condition_targets.items():
                self.assertAlmostEqual(
                    target["safe_scale_ceiling"], expected_ceilings[key]
                )
                self.assertEqual(target["deployment_geometry_variant_count"], 2)
            self.assertEqual(row["safe_scale_ceiling"], 1.23)

            with self.assertRaisesRegex(RuntimeError, "cross-product"):
                merge.write_bundle(
                    bundles[:-1], geometry_path, root / "mixed-incomplete",
                    color_mode=None,
                    input_variant_manifest_path=input_path,
                )

    def test_native_pq_uses_its_own_two_geometry_safety_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "native-preview.png"
            cv2.imwrite(str(source), np.zeros((9, 16, 3), np.uint8))
            native = input_color.native_pq_input_variant()
            rows = [
                self.make_row(
                    root, source, "native-0", 4, 2, 1.35, 0.8, native
                ),
                self.make_row(
                    root, source, "native-1", 8, 4, 1.22, 0.6, native
                ),
            ]
            bundles = [
                self.write_schema8_bundle(root, f"native-{index}", [row])
                for index, row in enumerate(rows)
            ]
            geometries = geometry_contract.build_allowlist([
                geometry_contract.geometry_tuple(row, row["color_mode"])
                for row in rows
            ])
            geometry_path = root / "native-geometries.json"
            geometry_path.write_text(json.dumps(geometries), encoding="utf-8")
            input_path = self.write_input_manifest(
                root, merge.policy_input_variants()
            )

            output = root / "native-output"
            summary = merge.write_bundle(
                bundles, geometry_path, output, color_mode=None,
                input_variant_manifest_path=input_path,
            )
            row = json.loads(
                (output / "labels.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(len(row["input_condition_targets"]), 1)
            self.assertEqual(len(row["deployment_geometry_variants"]), 2)
            self.assertEqual(row["safe_scale_ceiling"], 1.22)
            self.assertEqual(summary["input_variant_count"], 1)
            self.assertEqual(summary["declared_input_variant_count"], 5)
            self.assertEqual(summary["geometry_input_variant_count"], 2)
            self.assertEqual(summary["condition_target_count_per_rgb"], 1)


if __name__ == "__main__":
    unittest.main()
