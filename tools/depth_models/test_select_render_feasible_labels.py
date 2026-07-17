import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import select_render_feasible_labels as selector  # noqa: E402
import run_eval  # noqa: E402


SPECS = {
    "source_halo_p95": {
        "role": "primary", "axis": "warp", "better": "lower",
        "rel_tol": 0.2, "abs_floor": 0.5,
    },
    "static_jitter_p95": {
        "role": "primary", "axis": "stability", "better": "lower",
        "rel_tol": 0.2, "abs_floor": 0.5,
    },
    "depth_gt_lag_f1_p95": {
        "role": "primary", "axis": "stability", "better": "lower",
        "rel_tol": 0.2, "abs_floor": 5.0,
    },
    "source_coverage_pct": {
        "role": "hard", "axis": "integrity", "better": "higher",
        "rel_tol": 0.0, "abs_floor": 1.0, "hard_min": 90.0,
    },
}


def aggregate(psnr, pop=1.0, halo=4.0, jitter=2.0, coverage=99.0):
    return {
        "stereo_gt_psnr": psnr,
        "exact_pop_spread_pct": pop,
        "source_halo_p95": halo,
        "static_jitter_p95": jitter,
        "source_coverage_pct": coverage,
    }


def policy_clip_contract(scale=1.0, input_variant=None, **changes):
    input_variant = input_variant or selector.input_color.sdr_input_variant()
    selector.input_color.validate_input_variant(input_variant)
    contract = {
        "harness_schema": selector.EXPECTED_HARNESS_SCHEMA,
        "model": "model", "profile": "apollo",
        "metric_sha256": "metrics", "policy_warp_source_sha256": "a" * 64,
        "source_width": 16, "source_height": 9,
        "model_input_width": 14, "model_input_height": 14,
        "eye_width": 2, "eye_height": 2,
        "color_mode": input_variant["color_mode"],
        "hdr_source_kind": (
            selector.sbs_contract.input_variant_hdr_source_kind(input_variant)
        ),
        "metric_preview_encoding": (
            selector.sbs_contract.input_variant_metric_preview_encoding(
                input_variant
            )
        ),
        "hdr_input_scale": float(input_variant["scrgb_white_scale"] or 0.0),
        "sdr_white_level_raw": int(
            input_variant["windows_sdr_white_level_raw"] or 0
        ),
        "content_scale_x": 1.0, "content_scale_y": 1.0,
        "disparity_raster_width": 2, "disparity_raster_height": 2,
        "artistic_full_clamp_abs": 0.04,
        "depth_step": "current-once", "depth_reuse_interval": 1,
        "depth_compensation": "none", "depth_override_frames": 0,
        "ema": 0.5, "ema_edge_change": 0.05,
        "ema_edge_gradient": 0.02, "ema_edge_strength": 0.25,
        "minmax_ema": 0.18, "subject_lock": 0.5,
        "subject_recenter": 0.35, "subject_stretch": True,
        "depth_short_side": 432, "depth_max_aspect": 4.0,
        "pop_strength": 1.25, "adaptive_pop": True,
        "adaptive_pop_max": 1.3, "zero_plane": "legacy",
        "artistic_style": "immersive", "artistic_policy": False,
        "artistic_policy_consumed": False,
        "artistic_policy_authorization": "none",
        "model_onnx_sha256": "", "policy_metadata_sha256": "",
        "deployment_geometry_allowlist_sha256": "",
        "artistic_scale_override": scale,
        "output_interval": 1, "output_gt_right_only": True,
        "literal_bestv2": False, "cuda_graph": True,
        "artifact_mode": "full", "warp_mask": selector.WARP_MASK_CONTRACT,
        "warp_disparity": selector.WARP_DISPARITY_CONTRACT,
        "warp_unclamped_disparity": selector.WARP_UNCLAMPED_DISPARITY_CONTRACT,
        "artistic_disparity_contract": selector.ARTISTIC_DISPARITY_CONTRACT,
    }
    contract.update(changes)
    return contract


def top_meta_from_clip(contract, **changes):
    meta = {field: contract[field] for field in selector.CLIP_TOP_META_FIELDS}
    meta.update({
        "clip_set_sha1": "clips", "eval_schema": 30,
        "conf_sha256": "conf", "extra_args": [],
    })
    meta.update(changes)
    return meta


def covered_grid(candidates):
    """Fill uninteresting upper samples with a hard failure through scale 1.5."""
    candidates = dict(candidates)
    for index in range(1, 6):
        scale = round(1.0 + 0.1 * index, 1)
        candidates.setdefault(
            scale, aggregate(0.0, pop=scale, coverage=80.0)
        )
    return candidates


def worst_frame_for(aggregate_metrics):
    return {
        metric: {"frame": 1, "worst_value": aggregate_metrics[metric]}
        for metric, spec in SPECS.items()
        if (spec.get("role") == "primary" and
            spec.get("axis") in selector.PROTECTED_PRIMARY_AXES and
            metric in aggregate_metrics and aggregate_metrics[metric] is not None)
    }


class RenderFeasibilityTests(unittest.TestCase):
    def test_sparse_label_targets_authenticate_adjacent_evidence(self):
        meta = policy_clip_contract(
            output_gt_right_only=False,
            output_selection_mode="label-frames",
            label_frame_ids=[0, 3],
            output_selected_frame_ids=[0, 1, 2, 3],
            output_label_frames_sha256="c" * 64,
        )
        selection = selector.output_selection_contract(meta, "fixture")
        self.assertEqual(selection["label_frame_ids"], (0, 3))
        self.assertEqual(selection["selected_frame_ids"], (0, 1, 2, 3))

    def test_label_target_without_adjacent_evidence_fails_closed(self):
        meta = policy_clip_contract(
            output_gt_right_only=False,
            output_selection_mode="label-frames",
            label_frame_ids=[4],
            output_selected_frame_ids=[4],
            output_label_frames_sha256="c" * 64,
        )
        with self.assertRaisesRegex(RuntimeError, "no adjacent evidence"):
            selector.output_selection_contract(meta, "fixture")

    def test_label_selection_rejects_unrelated_extra_artifact_frame(self):
        meta = policy_clip_contract(
            output_gt_right_only=False,
            output_selection_mode="label-frames",
            label_frame_ids=[4],
            output_selected_frame_ids=[3, 4, 8],
            output_label_frames_sha256="c" * 64,
        )
        with self.assertRaisesRegex(RuntimeError, "unauthenticated evidence"):
            selector.output_selection_contract(meta, "fixture")

    def test_generic_source_requires_exact_target_and_emitted_identities(self):
        meta = policy_clip_contract(
            output_gt_right_only=False,
            output_selection_mode="label-frames",
            label_frame_ids=[0, 3],
            output_selected_frame_ids=[0, 1, 2, 3],
            output_label_frames_sha256="c" * 64,
        )
        source = {
            "source_contract": selector.GENERIC_SOURCE_CONTRACT,
            "frame": 3,
            "label_frame_ids": [0, 3],
            "output_selected_frame_ids": [0, 1, 2, 3],
            "label_frames_sha256": "c" * 64,
        }
        selector.validate_source_render_selection(source, meta, "fixture")
        source["output_selected_frame_ids"] = [0, 1, 3]
        with self.assertRaisesRegex(RuntimeError, "differs"):
            selector.validate_source_render_selection(source, meta, "fixture")

    def test_explicit_gt_right_selection_keeps_legacy_source_compatible(self):
        meta = policy_clip_contract(
            output_gt_right_only=True,
            output_selection_mode="gt-right",
            label_frame_ids=[],
            output_selected_frame_ids=[2, 7],
            output_label_frames_sha256="",
        )
        selector.validate_source_render_selection({"frame": 7}, meta, "fixture")
        with self.assertRaisesRegex(RuntimeError, "was not rendered"):
            selector.validate_source_render_selection({"frame": 3}, meta, "fixture")

    def test_generic_source_bundle_contract_is_admitted_without_stereo_fitter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_variant = selector.input_color.sdr_input_variant()
            labels = root / "labels.jsonl"
            labels.write_text("{}\n", encoding="utf-8")
            contract_path = root / "source_contract.json"
            contract_path.write_text(json.dumps({
                "schema": selector.GENERIC_SOURCE_SCHEMA,
                "source_contract": selector.GENERIC_SOURCE_CONTRACT,
                "run_contract": {"model": "model", "conf_sha256": "conf"},
                "input_variant": input_variant,
                "input_variant_sha256":
                    selector.input_color.input_variant_sha256(input_variant),
                "depth_input_color_contract_sha256":
                    selector.input_color.color_contract_sha256(),
            }), encoding="utf-8")
            (root / "summary.json").write_text(json.dumps({
                "schema": selector.GENERIC_SOURCE_SCHEMA,
                "source_contract": selector.GENERIC_SOURCE_CONTRACT,
                "labels_sha256": selector.sha256(labels),
                "source_contract_sha256": selector.sha256(contract_path),
            }), encoding="utf-8")
            admitted = selector.source_label_contract(
                labels, {"model": "model", "conf_sha256": "conf"}
            )
            self.assertEqual(admitted["kind"], "generic-source")
            self.assertNotIn("fitter_contract", admitted)
            self.assertEqual(admitted["input_variant"], input_variant)

    def test_generic_source_bundle_rejects_stale_input_variant_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels.jsonl"
            labels.write_text("{}\n", encoding="utf-8")
            input_variant = selector.input_color.windows_hdr_input_variant(2500)
            contract_path = root / "source_contract.json"
            contract = {
                "schema": selector.GENERIC_SOURCE_SCHEMA,
                "source_contract": selector.GENERIC_SOURCE_CONTRACT,
                "run_contract": {"model": "model", "conf_sha256": "conf"},
                "input_variant": input_variant,
                "input_variant_sha256":
                    selector.input_color.input_variant_sha256(input_variant),
                "depth_input_color_contract_sha256":
                    selector.input_color.color_contract_sha256(),
            }

            def publish_contract():
                contract_path.write_text(json.dumps(contract), encoding="utf-8")
                (root / "summary.json").write_text(json.dumps({
                    "schema": selector.GENERIC_SOURCE_SCHEMA,
                    "source_contract": selector.GENERIC_SOURCE_CONTRACT,
                    "labels_sha256": selector.sha256(labels),
                    "source_contract_sha256": selector.sha256(contract_path),
                }), encoding="utf-8")

            publish_contract()
            admitted = selector.source_label_contract(
                labels, {"model": "model", "conf_sha256": "conf"}
            )
            self.assertEqual(admitted["input_variant"], input_variant)
            contract["input_variant_sha256"] = "0" * 64
            publish_contract()
            with self.assertRaisesRegex(RuntimeError, "variant hash is stale"):
                selector.source_label_contract(
                    labels, {"model": "model", "conf_sha256": "conf"}
                )

    def test_hdr_source_row_is_authenticated_and_color_bound(self):
        input_variant = selector.input_color.windows_hdr_input_variant(6000)
        row = {
            "source_schema": selector.GENERIC_SOURCE_SCHEMA,
            "source_contract": selector.GENERIC_SOURCE_CONTRACT,
            "color_mode": selector.input_color.COLOR_MODE_HDR,
            "hdr_source_kind": selector.sbs_contract.HDR_SOURCE_SIMULATED,
            "metric_preview_encoding": selector.sbs_contract.METRIC_PREVIEW_HDR,
            "input_variant": input_variant,
            "input_variant_sha256":
                selector.input_color.input_variant_sha256(input_variant),
        }
        self.assertEqual(
            selector.validate_source_input_variant(
                row, input_variant, "HDR source row"
            ),
            input_variant,
        )
        hdr_contract = policy_clip_contract(input_variant=input_variant)
        self.assertEqual(
            selector.validate_source_raster_contract(
                hdr_contract, hdr_contract
            ),
            (2, 2),
        )

        wrong_color = dict(row, color_mode=selector.input_color.COLOR_MODE_SDR)
        with self.assertRaisesRegex(RuntimeError, "color mode differs"):
            selector.validate_source_input_variant(
                wrong_color, input_variant, "HDR source row"
            )

    def test_native_pq_harness_provenance_is_not_simulated_hdr(self):
        native = selector.input_color.native_pq_input_variant()
        harness = policy_clip_contract(input_variant=native)
        self.assertEqual(
            selector.input_variant_from_harness(harness, "native HDR"),
            native,
        )
        self.assertEqual(harness["hdr_input_scale"], 0.0)
        self.assertEqual(harness["sdr_white_level_raw"], 0)
        simulated = dict(
            harness,
            hdr_source_kind=selector.sbs_contract.HDR_SOURCE_SIMULATED,
        )
        with self.assertRaisesRegex(RuntimeError, "SDR-white"):
            selector.input_variant_from_harness(simulated, "native HDR")

    def test_missing_variant_fails_closed_except_legacy_sdr_default(self):
        with self.assertRaisesRegex(RuntimeError, "missing authenticated"):
            selector.authenticated_input_variant({}, "generic source")
        variant, variant_hash = selector.authenticated_input_variant(
            {}, "legacy source", allow_legacy_sdr=True
        )
        self.assertEqual(variant, selector.input_color.sdr_input_variant())
        self.assertEqual(
            variant_hash, selector.input_color.input_variant_sha256(variant)
        )

    def test_default_immersive_never_follows_low_pop_authored_target(self):
        result = selector.select_clip(
            aggregate(26.0),
            covered_grid({
                0.9: aggregate(32.0, pop=0.9),
                1.0: aggregate(26.0, pop=1.0),
                1.1: aggregate(25.0, pop=1.1),
            }),
            SPECS,
        )
        self.assertEqual(result["style_targets"]["immersive"], 1.1)
        self.assertEqual(result["authored_fit_scale"], 0.9)
        self.assertGreaterEqual(result["style_targets"]["immersive"], 1.0)

    def test_unsafe_hole_disconnects_every_farther_candidate(self):
        result = selector.select_clip(
            aggregate(26.0),
            covered_grid({
                1.0: aggregate(26.0, pop=1.0),
                1.1: aggregate(25.0, pop=1.1, halo=6.0),
                1.2: aggregate(25.0, pop=1.3, halo=4.1),
            }),
            SPECS,
        )
        self.assertEqual(result["style_targets"]["immersive"], 1.0)
        self.assertFalse(result["candidate_grid"][1.2]["connected"])
        self.assertTrue(result["candidate_grid"][1.2]["individually_feasible"])

    def test_identity_hard_failure_becomes_explicit_nonactionable_negative(self):
        candidates = covered_grid({
            0.9: aggregate(26.0, pop=0.9, coverage=82.0),
            1.0: aggregate(26.0, pop=1.0, coverage=82.0),
            1.1: aggregate(26.0, pop=1.1, coverage=95.0),
        })
        result = selector.select_clip(
            aggregate(26.0, coverage=82.0), candidates, SPECS
        )
        self.assertFalse(result["identity_feasible"])
        self.assertEqual(result["identity_violations"], [
            "source_coverage_pct:hard"
        ])
        self.assertEqual(result["safe_scale_ceiling"], 1.0)
        self.assertEqual(result["ceiling_confidence"], 0.0)
        self.assertEqual(result["safety_margin_reliability"], 0.0)
        self.assertEqual(result["connected_safe_scales"], [])
        self.assertTrue(all(
            not evidence["connected"]
            for evidence in result["candidate_grid"].values()
        ))
        self.assertEqual(result["style_targets"], {
            "clean": 1.0, "balanced": 1.0, "immersive": 1.0,
        })

    def test_identity_missing_hard_evidence_is_not_a_negative_label(self):
        identity = aggregate(26.0)
        del identity["source_coverage_pct"]
        with self.assertRaisesRegex(RuntimeError, "incomplete or inconsistent"):
            selector.select_clip(
                aggregate(26.0), covered_grid({1.0: identity}), SPECS
            )

    def test_stability_regression_stops_connected_frontier(self):
        result = selector.select_clip(
            aggregate(26.0),
            covered_grid({
                1.0: aggregate(26.0),
                1.1: aggregate(27.0, pop=1.1, jitter=3.0),
            }),
            SPECS,
        )
        self.assertEqual(result["style_targets"]["immersive"], 1.0)
        self.assertIn(
            "static_jitter_p95:regression",
            result["candidate_grid"][1.1]["violations"],
        )

    def test_one_bad_frame_blocks_average_safe_candidate(self):
        control_aggregate = aggregate(26.0, halo=4.0)
        candidate_aggregate = aggregate(25.0, pop=1.1, halo=4.1)
        control = selector.project_protected_worst_metrics({
            "aggregate": control_aggregate,
            "worst_frame": worst_frame_for(control_aggregate),
        }, SPECS, "control")
        candidate_worst = worst_frame_for(candidate_aggregate)
        candidate_worst["source_halo_p95"]["worst_value"] = 6.0
        candidate = selector.project_protected_worst_metrics({
            "aggregate": candidate_aggregate,
            "worst_frame": candidate_worst,
        }, SPECS, "candidate")
        result = selector.select_clip(
            control,
            covered_grid({1.0: control, 1.1: candidate}),
            SPECS,
        )
        self.assertEqual(result["safe_scale_ceiling"], 1.0)
        self.assertIn(
            "source_halo_p95:regression",
            result["candidate_grid"][1.1]["violations"],
        )
        with self.assertRaisesRegex(RuntimeError, "worst-frame evidence"):
            selector.project_protected_worst_metrics({
                "aggregate": candidate_aggregate, "worst_frame": {},
            }, SPECS, "candidate")

    def test_unavailable_gt_depth_stability_is_not_required(self):
        result = selector.select_clip(
            aggregate(26.0),
            covered_grid({
                1.0: aggregate(26.0), 1.1: aggregate(25.0, pop=1.1),
            }),
            SPECS,
            {"required_gt_depth": False},
        )
        self.assertEqual(result["safe_scale_ceiling"], 1.1)
        self.assertNotIn(
            "depth_gt_lag_f1_p95",
            result["candidate_grid"][1.1]["constraint_margins"],
        )

    def test_declared_gt_depth_stability_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "depth_gt_lag_f1_p95"):
            selector.select_clip(
                aggregate(26.0),
                covered_grid({
                    1.0: aggregate(26.0), 1.1: aggregate(26.0, pop=1.1),
                }),
                SPECS,
                {"required_gt_depth": True},
            )

    def test_unavailable_control_primary_is_skipped_but_lost_evidence_fails(self):
        specs = dict(SPECS, source_stretch_pct={
            "role": "primary", "axis": "warp", "better": "lower",
            "rel_tol": 0.2, "abs_floor": 1.0,
        })
        no_control_evidence = selector.select_clip(
            aggregate(26.0),
            covered_grid({
                1.0: aggregate(26.0), 1.1: aggregate(25.0, pop=1.1),
            }),
            specs,
        )
        self.assertEqual(no_control_evidence["safe_scale_ceiling"], 1.1)
        control = aggregate(26.0)
        control["source_stretch_pct"] = 2.0
        identity = aggregate(26.0)
        identity["source_stretch_pct"] = 2.0
        lost_evidence = selector.select_clip(
            control,
            covered_grid({
                1.0: identity, 1.1: aggregate(25.0, pop=1.1),
            }),
            specs,
        )
        self.assertEqual(lost_evidence["safe_scale_ceiling"], 1.0)
        self.assertIn(
            "source_stretch_pct:missing",
            lost_evidence["candidate_grid"][1.1]["violations"],
        )

    def test_styles_have_explicit_distinct_rules(self):
        result = selector.select_clip(
            aggregate(26.0),
            covered_grid({
                0.9: aggregate(30.0, pop=0.95, halo=3.0, jitter=1.5),
                1.0: aggregate(26.0, pop=1.0),
                1.1: aggregate(25.0, pop=1.1, halo=4.2, jitter=2.1),
                1.2: aggregate(24.0, pop=1.2, halo=4.4, jitter=2.2),
            }),
            SPECS,
        )
        self.assertEqual(result["style_targets"], {
            "immersive": 1.2,
            "balanced": 1.1,
            "clean": 1.0,
        })
        self.assertEqual(result["safe_scale_ceiling"], 1.2)
        self.assertEqual(result["authored_fit_scale"], 0.9)

    def test_confidence_uses_safety_evidence_not_psnr(self):
        candidates = covered_grid({
            1.0: aggregate(26.0),
            1.1: aggregate(20.0, pop=1.1),
            1.2: aggregate(10.0, pop=1.2),
        })
        first = selector.select_clip(aggregate(26.0), candidates, SPECS)
        candidates[1.2]["stereo_gt_psnr"] = 50.0
        second = selector.select_clip(aggregate(26.0), candidates, SPECS)
        self.assertEqual(
            first["safety_margin_reliability"],
            second["safety_margin_reliability"],
        )
        self.assertEqual(first["ceiling_confidence"], 1.0)
        self.assertGreaterEqual(first["safety_margin_reliability"], 0.5)
        neutral = selector.select_clip(
            aggregate(26.0),
            covered_grid({
                1.0: aggregate(26.0),
                1.1: aggregate(25.0, pop=1.1, coverage=80.0),
            }),
            SPECS,
        )
        self.assertEqual(neutral["ceiling_confidence"], 0.0)
        self.assertEqual(neutral["safety_margin_reliability"], 0.0)

    def test_missing_exact_pop_fails_closed(self):
        identity = aggregate(26.0)
        del identity["exact_pop_spread_pct"]
        with self.assertRaisesRegex(RuntimeError, "exact_pop_spread_pct"):
            selector.select_clip(
                identity,
                covered_grid({
                    1.0: identity, 1.1: aggregate(26.0, pop=1.1),
                }),
                SPECS,
            )

    def test_sparse_scale_grid_is_not_a_connected_frontier(self):
        with self.assertRaisesRegex(RuntimeError, "too sparse"):
            selector.select_clip(
                aggregate(26.0),
                {1.0: aggregate(26.0), 1.5: aggregate(20.0, pop=1.5)},
                SPECS,
            )

    def test_truncated_high_pop_grid_cannot_claim_a_safe_ceiling(self):
        with self.assertRaisesRegex(RuntimeError, "full upward model contract"):
            selector.select_clip(
                aggregate(26.0),
                {
                    1.0: aggregate(26.0),
                    1.1: aggregate(25.0, pop=1.1),
                },
                SPECS,
            )

    def test_metric_thresholds_must_match_render_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            thresholds = Path(temporary) / "thresholds.json"
            thresholds.write_text(json.dumps({"metrics": SPECS}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "differs from the render grid"):
                selector.validate_metric_contract(
                    {"metric_sha256": "not-the-contract"}, thresholds
                )

    def test_metric_hash_matches_the_canonical_evaluator(self):
        thresholds = selector.SBSBENCH_DIR / "thresholds.json"
        self.assertEqual(
            selector.metric_contract_hash(thresholds),
            run_eval.metric_contract_sha(),
        )

    def test_candidate_policy_and_geometry_context_must_match(self):
        control_clip = policy_clip_contract()
        candidate_clip = policy_clip_contract(1.1)
        common = top_meta_from_clip(
            control_clip, extra_args=["--output-gt-right-only"]
        )
        control = {
            "meta": common, "clips": {"shot": {"meta": control_clip}},
        }
        changed_policy = dict(common, artistic_scale_override=1.1, subject_lock=0.9)
        with self.assertRaisesRegex(RuntimeError, "policy/geometry"):
            selector.validate_context(
                control,
                {"meta": changed_policy,
                 "clips": {"shot": {"meta": candidate_clip}}},
                1.1, "candidate",
            )
        changed_geometry = dict(
            common, artistic_scale_override=1.1,
            extra_args=["--output-gt-right-only", "--eye-h", "720",
                        "--artistic-scale-override", "1.1"],
        )
        with self.assertRaisesRegex(RuntimeError, "policy/geometry"):
            selector.validate_context(
                control,
                {"meta": changed_geometry,
                 "clips": {"shot": {"meta": candidate_clip}}},
                1.1, "candidate",
            )

    def test_compensated_or_overridden_depth_runs_are_rejected(self):
        control_clip = policy_clip_contract()
        control = {
            "meta": top_meta_from_clip(control_clip),
            "clips": {"shot": {"meta": control_clip}},
        }
        candidate_clip = policy_clip_contract(1.1)
        candidate_meta = top_meta_from_clip(
            candidate_clip,
            extra_args=["--artistic-scale-override", "1.1"],
        )
        candidate = {
            "meta": candidate_meta,
            "clips": {"shot": {"meta": candidate_clip}},
        }
        selector.validate_context(
            control, candidate, 1.1, "candidate"
        )
        compensated_clip = policy_clip_contract(
            1.1, depth_compensation="nvof-1x1"
        )
        compensated = {
            "meta": top_meta_from_clip(compensated_clip),
            "clips": {"shot": {"meta": compensated_clip}},
        }
        with self.assertRaisesRegex(RuntimeError, "depth_compensation=none"):
            selector.validate_context(
                control, compensated, 1.1, "candidate"
            )
        overridden_clip = policy_clip_contract(1.1, depth_override_frames=1)
        overridden = {
            "meta": candidate_meta,
            "clips": {"shot": {"meta": overridden_clip}},
        }
        with self.assertRaisesRegex(RuntimeError, "depth_override_frames=0"):
            selector.validate_context(
                control, overridden, 1.1, "candidate"
            )

    def test_per_clip_policy_and_raster_contract_is_strict(self):
        control_clip = policy_clip_contract()
        candidate_clip = policy_clip_contract(1.1)
        control = {
            "meta": top_meta_from_clip(control_clip),
            "clips": {"shot": {
                "meta": control_clip, "aggregate": aggregate(26.0),
            }},
        }

        def candidate_payload(clip_meta):
            return {
                "meta": top_meta_from_clip(
                    clip_meta,
                    extra_args=["--artistic-scale-override", "1.1"],
                ),
                "clips": {"shot": {
                    "meta": clip_meta, "aggregate": aggregate(25.0, pop=1.1),
                }},
            }

        selector.validate_context(
            control, candidate_payload(candidate_clip), 1.1, "candidate"
        )
        changed_geometry = policy_clip_contract(
            1.1, eye_width=4, disparity_raster_width=4
        )
        with self.assertRaisesRegex(RuntimeError, "policy/raster contract differs"):
            selector.validate_context(
                control, candidate_payload(changed_geometry), 1.1, "candidate"
            )
        changed_metric = policy_clip_contract(1.1, metric_sha256="stale")
        stale_metric = candidate_payload(changed_metric)
        stale_metric["meta"]["metric_sha256"] = "metrics"
        with self.assertRaisesRegex(RuntimeError, "differs from run metadata"):
            selector.validate_context(
                control, stale_metric, 1.1, "candidate"
            )
        stale_semantics = policy_clip_contract(1.1, warp_disparity="legacy")
        with self.assertRaisesRegex(RuntimeError, "invalid clip harness contract"):
            selector.validate_context(
                control, candidate_payload(stale_semantics), 1.1, "candidate"
            )
        missing_semantics = policy_clip_contract(1.1)
        del missing_semantics["warp_mask"]
        with self.assertRaisesRegex(RuntimeError, "clip harness contract lacks"):
            selector.validate_context(
                control, candidate_payload(missing_semantics), 1.1, "candidate"
            )

    def test_hdr_context_requires_exact_white_level_and_scale(self):
        input_variant = selector.input_color.windows_hdr_input_variant(2500)
        control_clip = policy_clip_contract(
            1.0, input_variant=input_variant
        )
        candidate_clip = policy_clip_contract(
            1.1, input_variant=input_variant
        )
        control = {
            "meta": top_meta_from_clip(control_clip),
            "clips": {"shot": {"meta": control_clip}},
        }

        def candidate_payload(clip_meta):
            return {
                "meta": top_meta_from_clip(
                    clip_meta,
                    extra_args=["--artistic-scale-override", "1.1"],
                ),
                "clips": {"shot": {"meta": clip_meta}},
            }

        selector.validate_context(
            control, candidate_payload(candidate_clip), 1.1, "candidate"
        )

        wrong_scale = policy_clip_contract(
            1.1, input_variant=input_variant, hdr_input_scale=3.75
        )
        with self.assertRaisesRegex(RuntimeError, "variant provenance differs"):
            selector.validate_context(
                control, candidate_payload(wrong_scale), 1.1, "candidate"
            )

        noncanonical_white = policy_clip_contract(
            1.1, input_variant=input_variant, sdr_white_level_raw=3000
        )
        with self.assertRaisesRegex(RuntimeError, "invalid HDR SDR-white"):
            selector.validate_context(
                control, candidate_payload(noncanonical_white), 1.1, "candidate"
            )

        missing_white = policy_clip_contract(1.1, input_variant=input_variant)
        del missing_white["sdr_white_level_raw"]
        with self.assertRaisesRegex(RuntimeError, "color provenance lacks"):
            selector.validate_context(
                control, candidate_payload(missing_white), 1.1, "candidate"
            )

        sdr_candidate = policy_clip_contract(1.1)
        with self.assertRaisesRegex(RuntimeError, "differs from control"):
            selector.validate_context(
                control, candidate_payload(sdr_candidate), 1.1, "candidate"
            )

    def test_control_and_scale_one_candidate_are_exact_identity(self):
        control_clip = policy_clip_contract(1.0)
        control = {
            "meta": top_meta_from_clip(
                control_clip,
                extra_args=["--artistic-scale-override", "1.0"],
            ),
            "clips": {"shot": {
                "meta": control_clip, "aggregate": aggregate(26.0),
            }},
        }
        candidate_clip = policy_clip_contract(1.0)
        candidate = {
            "meta": top_meta_from_clip(
                candidate_clip,
                extra_args=["--artistic-scale-override", "1.0"],
            ),
            "clips": {"shot": {
                "meta": candidate_clip, "aggregate": aggregate(26.0),
            }},
        }
        selector.validate_context(control, candidate, 1.0, "candidate")
        legacy_clip = policy_clip_contract(0.0)
        legacy_control = {
            "meta": top_meta_from_clip(legacy_clip),
            "clips": {"shot": {
                "meta": legacy_clip, "aggregate": aggregate(26.0),
            }},
        }
        with self.assertRaisesRegex(RuntimeError, "requires artistic_scale_override=1.0"):
            selector.validate_context(
                legacy_control, candidate, 1.0, "candidate"
            )
        mismatched = json.loads(json.dumps(candidate))
        mismatched["clips"]["shot"]["aggregate"]["exact_pop_spread_pct"] = 1.01
        with self.assertRaisesRegex(RuntimeError, "aggregate differs from control"):
            selector.validate_context(control, mismatched, 1.0, "candidate")

    def test_output_eye_raster_contract_and_texture_headers_are_exact(self):
        contract = {
            "source_width": 16, "source_height": 9,
            "model_input_width": 14, "model_input_height": 14,
            "eye_width": 4, "eye_height": 2,
            "color_mode": "sdr-srgb-8bit",
            "hdr_source_kind": selector.sbs_contract.HDR_SOURCE_SDR,
            "metric_preview_encoding": selector.sbs_contract.METRIC_PREVIEW_SDR,
            "hdr_input_scale": 0.0, "sdr_white_level_raw": 0,
            "content_scale_x": 0.5, "content_scale_y": 1.0,
            "disparity_raster_width": 4,
            "disparity_raster_height": 2,
            "artistic_full_clamp_abs": 0.04,
        }
        row = dict(contract)
        expected = selector.validate_source_raster_contract(row, contract)
        self.assertEqual(expected, (2, 4))
        bad_content_y = dict(row, content_scale_y=0.75)
        with self.assertRaisesRegex(RuntimeError, "content_scale_y"):
            selector.validate_source_raster_contract(bad_content_y, contract)
        bad_contract = dict(contract, disparity_raster_width=3)
        bad_row = dict(row, disparity_raster_width=3)
        with self.assertRaisesRegex(RuntimeError, "full output-eye raster"):
            selector.validate_source_raster_contract(bad_row, bad_contract)

    def test_output_eye_zero_bars_are_excluded_by_pixel_center(self):
        raster = np.array([
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 3.0, 4.0, 0.0],
        ], dtype=np.float32)
        content = selector.content_raster_values(raster, {
            "content_scale_x": 0.5, "content_scale_y": 1.0,
        })
        np.testing.assert_array_equal(content, np.array([
            [1.0, 2.0], [3.0, 4.0],
        ], dtype=np.float32))

    def test_hard_integrity_failure_is_never_traded_for_pop(self):
        result = selector.select_clip(
            aggregate(26.0),
            covered_grid({
                1.0: aggregate(26.0),
                1.1: aggregate(32.0, pop=1.3, coverage=80.0),
            }),
            SPECS,
        )
        self.assertEqual(result["style_targets"]["immersive"], 1.0)
        self.assertIn(
            "source_coverage_pct:hard",
            result["candidate_grid"][1.1]["violations"],
        )

    def test_clamp_aware_summary_uses_unclamped_field(self):
        raw = np.array([[-0.10, -0.01], [0.01, 0.10]], dtype=np.float32)
        identity = selector.clamp_aware_summary(raw, 1.0, 0.03)
        stronger = selector.clamp_aware_summary(raw, 1.5, 0.03)
        self.assertGreater(stronger["clamped_pixel_pct"], 0.0)
        self.assertGreaterEqual(stronger["exact_pop_spread_pct"],
                                identity["exact_pop_spread_pct"])
        # Saturation means the rendered field is not raw disparity times scale.
        self.assertLess(stronger["exact_pop_spread_pct"],
                        identity["exact_pop_spread_pct"] * 1.5)

    def test_schema8_bundle_retains_complete_grid_and_raw_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            thresholds = root / "thresholds.json"
            thresholds.write_text(json.dumps({"metrics": SPECS}), encoding="utf-8")
            metric_hash = selector.metric_contract_hash(thresholds)
            baseline = root / "baseline_disparity_00001.f32"
            raw = root / "baseline_unclamped_disparity_00001.f32"
            header = np.array([2, 2], dtype="<u4").tobytes()
            baseline.write_bytes(
                header + np.array([-.02, -.01, .01, .02], dtype="<f4").tobytes()
            )
            raw.write_bytes(
                header + np.array([-.04, -.01, .01, .04], dtype="<f4").tobytes()
            )
            warp_hash = "a" * 64
            input_variant = selector.input_color.sdr_input_variant()
            input_variant_hash = selector.input_color.input_variant_sha256(
                input_variant
            )
            harness_contract = {
                "schema": selector.EXPECTED_HARNESS_SCHEMA,
                "model": "model", "profile": "apollo",
                "depth_step": "current-once", "ema": 0.5,
                "ema_edge_change": 0.05, "ema_edge_gradient": 0.02,
                "ema_edge_strength": 0.25, "minmax_ema": 0.18,
                "subject_lock": 0.5, "subject_recenter": 0.35,
                "subject_stretch": True, "depth_short_side": 432,
                "depth_max_aspect": 4.0, "pop_strength": 1.25,
                "adaptive_pop": True, "adaptive_pop_max": 1.3,
                "zero_plane": "legacy", "literal_bestv2": False,
                "artistic_policy": False, "artistic_scale_override": 0.0,
                "artistic_policy_consumed": False,
                "artistic_policy_authorization": "none",
                "model_onnx_sha256": "", "policy_metadata_sha256": "",
                "deployment_geometry_allowlist_sha256": "",
                "source_width": 16, "source_height": 9,
                "model_input_width": 14, "model_input_height": 14,
                "eye_width": 2, "eye_height": 2,
                "color_mode": "sdr-srgb-8bit",
                "hdr_source_kind": selector.sbs_contract.HDR_SOURCE_SDR,
                "metric_preview_encoding": selector.sbs_contract.METRIC_PREVIEW_SDR,
                "hdr_input_scale": 0.0, "sdr_white_level_raw": 0,
                "content_scale_x": 1.0, "content_scale_y": 1.0,
                "disparity_raster_width": 2,
                "disparity_raster_height": 2,
                "depth_compensation": "none", "depth_override_frames": 0,
                "artistic_full_clamp_abs": 0.04,
                "policy_warp_source_sha256": warp_hash,
                "metric_sha256": metric_hash,
                "warp_disparity": (
                    "exact_clamped_full_binocular_normalized_at_output_eye_"
                    "raster_zero_bars"
                ),
                "warp_unclamped_disparity": (
                    "unclamped_full_binocular_normalized_at_artistic_scale_1_"
                    "output_eye_raster_zero_bars"
                ),
                "artistic_disparity_contract": (
                    "clamp(raw_baseline_times_scale_to_plus_or_minus_0.142_"
                    "times_aspect_scale_times_content_scale_x)"
                ),
            }
            harness_path = root / "contract.json"
            harness_path.write_text(
                json.dumps(harness_contract), encoding="utf-8"
            )
            source_labels = root / "source.jsonl"
            source_labels.write_text(json.dumps({
                "label_schema": 7, "policy_contract": "stereo-fit-source-v2",
                "clip": "shot-1", "frame": 1,
                "source": str(root / "frame.png"), "source_sha256": "unused",
                "baseline_disparity": str(baseline),
                "baseline_disparity_sha256": selector.sha256(baseline),
                "baseline_unclamped_disparity": str(raw),
                "baseline_unclamped_disparity_sha256": selector.sha256(raw),
                "source_width": 16, "source_height": 9,
                "model_input_width": 14, "model_input_height": 14,
                "eye_width": 2, "eye_height": 2,
                "color_mode": "sdr-srgb-8bit",
                "hdr_source_kind": selector.sbs_contract.HDR_SOURCE_SDR,
                "metric_preview_encoding": selector.sbs_contract.METRIC_PREVIEW_SDR,
                "input_variant": input_variant,
                "input_variant_sha256": input_variant_hash,
                "content_scale_x": 1.0, "content_scale_y": 1.0,
                "disparity_raster_width": 2,
                "disparity_raster_height": 2,
                "artistic_full_clamp_abs": 0.04,
                "harness_contract_sha256": selector.sha256(harness_path),
                "baseline_multiplier": 0.7, "confidence": 0.2,
            }) + "\n", encoding="utf-8")
            source_fitter = root / "label_fitter_contract.json"
            source_fitter.write_text(json.dumps({
                "schema": 7,
                "run_contract": {
                    "kind": "depth_run_manifest", "model": "model",
                    "conf_sha256": "conf",
                },
                "input_variant": input_variant,
                "input_variant_sha256": input_variant_hash,
                "depth_input_color_contract_sha256":
                    selector.input_color.color_contract_sha256(),
            }), encoding="utf-8")
            (root / "summary.json").write_text(json.dumps({
                "schema": 7,
                "labels_sha256": selector.sha256(source_labels),
                "label_fitter_contract_sha256": selector.sha256(source_fitter),
            }), encoding="utf-8")
            control_clip_meta = policy_clip_contract(
                1.0, metric_sha256=metric_hash,
                policy_warp_source_sha256=warp_hash,
                eye_width=4, disparity_raster_width=4,
            )
            common_meta = top_meta_from_clip(
                control_clip_meta,
                conf_sha256="conf",
                extra_args=["--output-gt-right-only",
                            "--artistic-scale-override", "1.0"],
            )
            control = root / "control.json"
            control_aggregate = aggregate(26.0)
            control.write_text(json.dumps({
                "meta": common_meta,
                "clips": {"shot-1": {
                    "aggregate": control_aggregate,
                    "worst_frame": worst_frame_for(control_aggregate),
                    "meta": control_clip_meta,
                }},
            }), encoding="utf-8")
            candidates = []
            for scale in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
                path = root / f"candidate-{scale}.json"
                candidate_clip_meta = policy_clip_contract(
                    scale, metric_sha256=metric_hash,
                    policy_warp_source_sha256=warp_hash,
                    eye_width=4, disparity_raster_width=4,
                )
                meta = top_meta_from_clip(
                    candidate_clip_meta, conf_sha256="conf",
                    extra_args=["--output-gt-right-only",
                                "--artistic-scale-override", str(scale)],
                )
                candidate_aggregate = aggregate(
                    26.0, pop=scale, coverage=99.0 if scale <= 1.1 else 80.0
                )
                path.write_text(json.dumps({
                    "meta": meta,
                    "clips": {"shot-1": {
                        "aggregate": candidate_aggregate,
                        "worst_frame": worst_frame_for(candidate_aggregate),
                        "meta": candidate_clip_meta,
                    }},
                }), encoding="utf-8")
                candidates.append((scale, path))
                if scale == 1.0:
                    clip_root = root / "shot-1"
                    clip_root.mkdir()
                    disk_contract = dict(candidate_clip_meta)
                    disk_contract["schema"] = disk_contract.pop("harness_schema")
                    (clip_root / "contract.json").write_text(
                        json.dumps(disk_contract), encoding="utf-8"
                    )
                    grid_header = np.array([4, 2], dtype="<u4").tobytes()
                    grid_clamped = np.array(
                        [-.02, -.02, -.01, -.01, .01, .01, .02, .02],
                        dtype="<f4",
                    ).tobytes()
                    grid_raw = np.array(
                        [-.04, -.04, -.01, -.01, .01, .01, .04, .04],
                        dtype="<f4",
                    ).tobytes()
                    (clip_root / "warp_disparity_00001.f32").write_bytes(
                        grid_header + grid_clamped
                    )
                    (clip_root / "warp_unclamped_disparity_00001.f32").write_bytes(
                        grid_header + grid_raw
                    )
            output = root / "output"
            summary = selector.write_bundle(
                source_labels, control, candidates, output, thresholds
            )
            row = json.loads((output / "labels.jsonl").read_text(encoding="utf-8"))
            evidence = json.loads(
                (output / "render_grid_evidence.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["schema"], 8)
            self.assertEqual(row["policy_contract"], selector.POLICY_CONTRACT)
            contract = json.loads(
                (output / "label_fitter_contract.json").read_text(encoding="utf-8")
            )
            self.assertEqual(contract["policy_baseline"]["depth_model"], "model")
            self.assertEqual(contract["policy_baseline"]["warp_contract"],
                             selector.POLICY_WARP_CONTRACT)
            self.assertEqual(
                contract["policy_baseline"]["metric_sha256"],
                common_meta["metric_sha256"],
            )
            self.assertEqual(row["input_variant"], input_variant)
            self.assertEqual(row["input_variant_sha256"], input_variant_hash)
            self.assertEqual(row["hdr_input_scale"], 0.0)
            self.assertEqual(row["sdr_white_level_raw"], 0)
            self.assertEqual(contract["input_variant"], input_variant)
            self.assertEqual(
                contract["input_variant_sha256"], input_variant_hash
            )
            self.assertEqual(contract["color_mode"], "sdr-srgb-8bit")
            self.assertEqual(contract["hdr_input_scale"], 0.0)
            self.assertEqual(contract["sdr_white_level_raw"], 0)
            self.assertEqual(evidence["input_variant"], input_variant)
            self.assertEqual(evidence["color_mode"], "sdr-srgb-8bit")
            self.assertEqual(summary["input_variant_sha256"], input_variant_hash)
            self.assertEqual(row["baseline_multiplier"], 1.1)
            self.assertEqual(row["ceiling_confidence"], 1.0)
            self.assertEqual(row["confidence"], 1.0)
            self.assertGreaterEqual(row["safety_margin_reliability"], 0.5)
            self.assertEqual(row["render_evidence_confidence"],
                             row["safety_margin_reliability"])
            self.assertEqual(row["eye_width"], 4)
            self.assertEqual(row["disparity_raster_width"], 4)
            self.assertEqual(
                row["baseline_unclamped_disparity_sha256"],
                selector.sha256(root / "shot-1" /
                                "warp_unclamped_disparity_00001.f32"),
            )
            self.assertEqual(row["source_depth_baseline_disparity"], str(baseline))
            self.assertEqual(set(evidence["clips"]["shot-1"]["candidate_grid"]),
                             {"1.0", "1.1", "1.2", "1.3", "1.4", "1.5"})
            self.assertIn("metrics",
                          evidence["clips"]["shot-1"]["candidate_grid"]["1.1"])


if __name__ == "__main__":
    unittest.main()
