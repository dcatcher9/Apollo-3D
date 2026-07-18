"""Focused tests for the fail-closed model-label qualification boundary."""

import ast
import json
import os
import sys
import unittest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import run_eval  # noqa: E402


class TrainingLabelQualificationTests(unittest.TestCase):
    def test_only_explicitly_qualified_label_is_exported(self):
        thresholds = {"metrics": {
            "qualified": {
                "label": "reward", "label_status": "qualified", "requires": "always",
                "scope": "perceptual", "role": "primary",
            },
            "experimental": {
                "label": "risk", "label_status": "experimental", "requires": "always",
            },
            "missing_status": {"label": "hard", "requires": "always"},
            "conformance": {"label_status": "conformance", "requires": "always"},
        }}

        records = run_eval.build_frame_records(
            [{"_frame_id": 4, "qualified": 1.5, "experimental": 2.0,
              "missing_status": 3.0, "conformance": 4.0}], thresholds, {})

        labels = records[0]["labels"]
        self.assertTrue(labels["eligible"])
        self.assertEqual(labels["required_status"], "qualified")
        self.assertEqual(labels["qualified_metric_count"], 1)
        self.assertEqual(labels["valid_metric_count"], 1)
        self.assertEqual(set(labels["metrics"]), {"qualified"})
        self.assertEqual(records[0]["metrics"]["experimental"], 2.0)

        manifest = run_eval.training_label_manifest(thresholds)
        self.assertEqual([item["metric"] for item in manifest["qualified_metrics"]],
                         ["qualified"])
        self.assertEqual({item["metric"] for item in manifest["excluded_metrics"]},
                         {"experimental", "missing_status"})

    def test_no_qualified_metrics_abstains_without_dropping_numeric_evidence(self):
        thresholds = {"metrics": {
            "candidate": {
                "label": "reward", "label_status": "experimental", "requires": "always",
            },
        }}

        records = run_eval.build_frame_records(
            [{"_frame_id": 8, "candidate": 1.25}], thresholds, {})
        summary = run_eval.summarize_frame_labels(records, thresholds)

        self.assertFalse(records[0]["labels"]["eligible"])
        self.assertEqual(records[0]["labels"]["reason"],
                         "no_qualified_training_labels")
        self.assertEqual(records[0]["labels"]["metrics"], {})
        self.assertEqual(records[0]["metrics"]["candidate"], 1.25)
        self.assertEqual(summary["eligible_frames"], 0)
        self.assertEqual(summary["abstained_frames"], 1)
        self.assertEqual(summary["qualified_metrics"], [])

        result = {
            "meta": {
                "run_kind": "comparison-only", "eval_schema": run_eval.EVAL_SCHEMA,
                "metric_sha256": run_eval.metric_contract_sha(),
                "metric_runtime": run_eval.metric_runtime_provenance(),
                "label_contract_sha256": run_eval.label_contract_sha(),
                "clip_set_sha1": {"clip": "2" * 12},
                "scored_artifact_sha256": {"clip": "3" * 64},
                "training_labels": run_eval.training_label_manifest(thresholds),
            },
            "verdict": "comparison_only", "hard_failures": [],
            "evidence_failures": [], "regressions": [],
            "clips": {"clip": {
                "frames": records, "label_summary": summary,
                "meta": {"source_frame_count": 1,
                         "scored_artifact_sha256": "3" * 64},
            }},
        }
        gate = run_eval.bind_training_labels_to_evidence_gate(result, thresholds)
        self.assertFalse(gate["passed"])
        self.assertIn("no_qualified_training_labels", gate["blockers"])

    def test_qualified_specs_fail_closed_on_scope_role_or_requirement(self):
        valid = {
            "label": "risk", "label_status": "qualified", "requires": "always",
            "scope": "perceptual", "role": "diagnostic",
        }
        for change in (
                {"scope": "conformance"}, {"label": "unknown"},
                {"role": "reported"}, {"requires": "typo"}):
            spec = {**valid, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                run_eval.training_label_manifest({"metrics": {"candidate": spec}})

    def test_frame_label_eligibility_is_revoked_when_evidence_or_provenance_fails(self):
        thresholds = {"metrics": {"candidate": {
            "label": "risk", "label_status": "qualified", "requires": "always",
            "scope": "perceptual", "role": "diagnostic",
        }}}
        records = run_eval.build_frame_records(
            [{"_frame_id": 1, "candidate": 2.0}], thresholds, {})
        result = {
            "meta": {
                "run_kind": "comparison-only", "metric_runtime": {},
                "label_contract_sha256": "1" * 16,
                "clip_set_sha1": {"clip": "2" * 12},
                "scored_artifact_sha256": {"clip": "3" * 64},
                "training_labels": run_eval.training_label_manifest(thresholds),
            },
            "verdict": "evidence_failures", "hard_failures": [],
            "evidence_failures": [{"metric": "missing"}], "regressions": [],
            "clips": {"clip": {
                "frames": records,
                "meta": {"source_frame_count": 1,
                         "scored_artifact_sha256": "3" * 64},
            }},
        }

        gate = run_eval.bind_training_labels_to_evidence_gate(result, thresholds)

        self.assertFalse(gate["passed"])
        self.assertFalse(records[0]["labels"]["eligible"])
        self.assertEqual(records[0]["labels"]["reason"],
                         "training_label_evidence_gate_failed")
        self.assertEqual(result["clips"]["clip"]["label_summary"]["eligible_frames"], 0)

    def test_unsafe_candidate_remains_a_valid_negative_training_example(self):
        thresholds = {"metrics": {"candidate": {
            "label": "risk", "label_status": "qualified", "requires": "always",
            "scope": "perceptual", "role": "hard", "hard_max": 10.0,
        }}}
        records = run_eval.build_frame_records(
            [{"_frame_id": 1, "candidate": 99.0}], thresholds, {})
        result = {
            "meta": {
                "run_kind": "comparison-only",
                "eval_schema": run_eval.EVAL_SCHEMA,
                "metric_sha256": run_eval.metric_contract_sha(),
                "metric_runtime": run_eval.metric_runtime_provenance(),
                "label_contract_sha256": run_eval.label_contract_sha(),
                "clip_set_sha1": {"clip": "2" * 12},
                "scored_artifact_sha256": {"clip": "3" * 64},
                "training_labels": run_eval.training_label_manifest(thresholds),
            },
            "verdict": "hard_failures",
            "hard_failures": [{"metric": "candidate"}],
            "evidence_failures": [], "regressions": [{"metric": "quality"}],
            "clips": {"clip": {
                "frames": records,
                "label_summary": run_eval.summarize_frame_labels(records, thresholds),
                "meta": {"source_frame_count": 1,
                         "scored_artifact_sha256": "3" * 64},
            }},
        }

        gate = run_eval.bind_training_labels_to_evidence_gate(result, thresholds)

        self.assertTrue(gate["passed"])
        self.assertTrue(records[0]["labels"]["eligible"])

    def test_forged_frame_label_is_rejected_against_numeric_metric_evidence(self):
        thresholds = {"metrics": {"candidate": {
            "label": "risk", "label_status": "qualified", "requires": "always",
            "scope": "perceptual", "role": "diagnostic",
        }}}
        records = run_eval.build_frame_records(
            [{"_frame_id": 1, "candidate": 2.0}], thresholds, {})
        records[0]["labels"]["metrics"]["candidate"]["value"] = 999.0
        result = {
            "meta": {
                "run_kind": "comparison-only", "eval_schema": run_eval.EVAL_SCHEMA,
                "metric_sha256": run_eval.metric_contract_sha(),
                "metric_runtime": run_eval.metric_runtime_provenance(),
                "label_contract_sha256": run_eval.label_contract_sha(),
                "clip_set_sha1": {"clip": "2" * 12},
                "scored_artifact_sha256": {"clip": "3" * 64},
                "training_labels": run_eval.training_label_manifest(thresholds),
            },
            "verdict": "comparison_only", "hard_failures": [],
            "evidence_failures": [], "regressions": [],
            "clips": {"clip": {
                "frames": records,
                "label_summary": run_eval.summarize_frame_labels(records, thresholds),
                "meta": {"source_frame_count": 1,
                         "scored_artifact_sha256": "3" * 64},
            }},
        }

        gate = run_eval.bind_training_labels_to_evidence_gate(result, thresholds)

        self.assertFalse(gate["passed"])
        self.assertIn("clips.clip.frame_labels", gate["blockers"])
        self.assertFalse(records[0]["labels"]["eligible"])

    def test_renderer_conformance_hard_failure_blocks_negative_labels(self):
        thresholds = {"metrics": {
            "candidate": {
                "label": "risk", "label_status": "qualified", "requires": "always",
                "scope": "perceptual", "role": "diagnostic",
            },
            "integrity": {
                "scope": "conformance", "role": "hard", "requires": "always",
                "hard_min": 99.0,
            },
        }}
        records = run_eval.build_frame_records(
            [{"_frame_id": 1, "candidate": 2.0, "integrity": 20.0}], thresholds, {})
        result = {
            "meta": {
                "run_kind": "comparison-only", "eval_schema": run_eval.EVAL_SCHEMA,
                "metric_sha256": run_eval.metric_contract_sha(),
                "metric_runtime": run_eval.metric_runtime_provenance(),
                "label_contract_sha256": run_eval.label_contract_sha(),
                "clip_set_sha1": {"clip": "2" * 12},
                "scored_artifact_sha256": {"clip": "3" * 64},
                "training_labels": run_eval.training_label_manifest(thresholds),
            },
            "verdict": "hard_failures",
            "hard_failures": [{"metric": "integrity"}],
            "evidence_failures": [], "regressions": [],
            "clips": {"clip": {
                "frames": records,
                "label_summary": run_eval.summarize_frame_labels(records, thresholds),
                "meta": {"source_frame_count": 1,
                         "scored_artifact_sha256": "3" * 64},
            }},
        }

        gate = run_eval.bind_training_labels_to_evidence_gate(result, thresholds)

        self.assertFalse(gate["passed"])
        self.assertIn("nonperceptual_or_unqualified_hard_failure", gate["blockers"])
        self.assertFalse(records[0]["labels"]["eligible"])

    def test_repository_contract_has_no_premature_qualified_labels(self):
        with open(os.path.join(SCRIPT_DIR, "thresholds.json"), encoding="utf-8") as stream:
            thresholds = json.load(stream)

        manifest = run_eval.training_label_manifest(thresholds)

        self.assertEqual(manifest["qualified_metrics"], [])
        self.assertTrue(manifest["excluded_metrics"])
        self.assertTrue(all(item["status"] == "experimental"
                            for item in manifest["excluded_metrics"]))
        for metric in ("source_coverage_pct", "image_integrity_pct",
                       "source_coverage_worst_patch_bad_pct",
                       "image_integrity_worst_patch_bad_pct",
                       "vmisalign_p99_pct",
                       "exact_polarity_ok", "exact_local_polarity_component_pct"):
            self.assertNotIn("label", thresholds["metrics"][metric])
            self.assertEqual(thresholds["metrics"][metric]["scope"], "conformance")

        self.assertEqual(
            {spec["scope"] for spec in thresholds["metrics"].values()},
            {"style", "perceptual", "conformance", "gt-only", "temporal-only"})
        for metric, spec in thresholds["metrics"].items():
            if "label" in spec:
                self.assertEqual(spec["scope"], "perceptual", metric)

    def test_repository_manifest_excludes_redundant_or_unrelated_proxies(self):
        with open(os.path.join(SCRIPT_DIR, "thresholds.json"), encoding="utf-8") as stream:
            thresholds = json.load(stream)
        metrics = set(thresholds["metrics"])
        removed = {
            "pop_spread_pct", "pop_spread_px", "positive_disparity_pct",
            "negative_disparity_pct", "exact_visible_bulk_pop_spread_pct",
            "exact_visible_abs_disparity_p995_pct", "exact_bulk_pop_spread_pct",
            "exact_extreme_pop_spread_pct", "exact_local_polarity_violation_pct",
            "exact_over_1pct_area_pct", "exact_over_2pct_area_pct",
            "exact_over_4pct_area_pct", "exact_over_3pct_component_pct",
            "exact_pop_spread_pct", "exact_forward_coverage_pct",
            "exact_hole_stretch_pct", "source_stretch_pct", "source_halo_p95",
            "source_chroma_residual_p95", "depth_gt_edge_f1_fine",
            "stereo_gt_psnr", "stereo_gt_ssim", "stereo_gt_residual_p95",
            "stereo_gt_coverage_pct", "stereo_art_scale_error_pct",
            "stereo_art_zero_error_pct", "stereo_art_ddc_iou", "depth_spread",
            "disocc_smear", "stretch_area", "rim_over_p95", "warp_hole_pct",
            "hole_source_residual_p95", "hole_bad_fill_pct", "artifact_in_hole_pct",
            "flicker_p50", "flicker_disocc_p50", "swim_p50",
            "depth_gt_si_p95", "depth_gt_boundary_p95", "edge_acc_p50",
            "vmisalign_pct", "exact_source_clamp_excess_p95_px",
            "edge_acc_p95", "edge_blur_excess_pct", "edge_ringing_excess_pct",
            "edge_color_fringe_excess_score", "source_color_residual_p95",
            "edge_double_edge_excess_pct", "edge_jagged_excess_pct",
            "edge_missing_pct", "interocular_luma_asymmetry_p95_pct",
            "interocular_detail_asymmetry_p95_pct",
            "depth_gt_bad_5pct", "flow_depth_p95",
            "depth_gt_bad_1px", "warp_cross_row_shear_largest_run_pct",
            "experimental_stereo_window_crossed_largest_component_pct",
            "disocclusion_bad_fill_largest_component_pct",
            "foreground_leak_largest_component_pct",
            "foreground_leak_burden_pct", "exact_source_clamp_pct",
            "source_residual_p95", "depth_gt_valid_pct", "stereo_gt_siou",
            "interocular_phase_orientation_p95_pct",
            "interocular_chroma_conflict_p95_pct",
            "interocular_chroma_conflict_burden_pct",
        }
        self.assertFalse(metrics & removed, sorted(metrics & removed))

        report_path = os.path.join(SCRIPT_DIR, "build_report.py")
        with open(report_path, encoding="utf-8") as stream:
            tree = ast.parse(stream.read())
        assignment = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "METRIC_DEFS"
                    for target in node.targets))
        report_metrics = {entry.elts[0].value for entry in assignment.value.elts}
        self.assertEqual(report_metrics, metrics)


if __name__ == "__main__":
    unittest.main()
