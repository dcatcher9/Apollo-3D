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
import merge_artistic_geometry_labels as merge  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_float_texture(path, width, height, value):
    values = np.full((height, width), value, dtype="<f4")
    Path(path).write_bytes(
        np.asarray([width, height], dtype="<u4").tobytes() + values.tobytes()
    )


class MultiGeometryMergeTests(unittest.TestCase):
    def make_row(self, root, source, name, eye_width, eye_height,
                 ceiling, reliability):
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
        return {
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

    def test_intersects_frontiers_and_keeps_every_geometry_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            cv2.imwrite(str(source), np.zeros((9, 16, 3), np.uint8))
            first_row = self.make_row(root, source, "1080", 4, 2, 1.3, 0.8)
            second_row = self.make_row(root, source, "4k", 8, 4, 1.2, 0.6)
            first = self.write_schema8_bundle(root, "first", [first_row])
            second = self.write_schema8_bundle(root, "second", [second_row])
            manifest_payload = geometry_contract.build_allowlist([
                geometry_contract.geometry_tuple(first_row),
                geometry_contract.geometry_tuple(second_row),
            ])
            manifest = root / "geometry.json"
            manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
            output = root / "merged"

            summary = merge.write_bundle([first, second], manifest, output)
            row = json.loads((output / "labels.jsonl").read_text(encoding="utf-8"))
            contract = json.loads(
                (output / "label_fitter_contract.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["schema"], 9)
            self.assertEqual(summary["unique_rgb_count"], 1)
            self.assertEqual(row["safe_scale_ceiling"], 1.2)
            self.assertEqual(row["safe_scale_max"], 1.2)
            self.assertEqual(row["safety_margin_reliability"], 0.6)
            self.assertEqual(row["style_targets"]["balanced"], 1.1)
            self.assertEqual(len(row["deployment_geometry_variants"]), 2)
            self.assertTrue(all(
                variant["safe_ceiling_render_target"]["scale"] == 1.2
                for variant in row["deployment_geometry_variants"]
            ))
            self.assertEqual(
                contract["deployment_geometry_allowlist"], manifest_payload
            )
            self.assertEqual(
                row["deployment_geometry_allowlist_sha256"],
                geometry_contract.allowlist_sha256(manifest_payload),
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
            with self.assertRaisesRegex(RuntimeError, "every exact geometry"):
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


if __name__ == "__main__":
    unittest.main()
