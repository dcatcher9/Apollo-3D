#!/usr/bin/env python3

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import artistic_geometry_contract as geometry_contract
import artistic_policy_ordinal_contract as ordinal_contract
import build_ordinal_frame_label_bundle as builder
import depth_input_color as input_color
import select_render_feasible_labels as scalar_selector


METRICS = {
    "exact_pop_spread_pct": {
        "role": "primary", "axis": "stereo", "better": "higher",
        "rel_tol": 0.1, "abs_floor": 0.1,
    },
    "source_halo_p95": {
        "role": "primary", "axis": "warp", "better": "lower",
        "rel_tol": 0.2, "abs_floor": 0.5, "trigger": 5.0,
    },
    "flow_temporal_error_p95": {
        "role": "primary", "axis": "stability", "better": "lower",
        "rel_tol": 0.2, "abs_floor": 0.1, "required_evidence": True,
        "min_frames": 2, "temporal_evidence": True,
    },
    "source_coverage_pct": {
        "role": "hard", "axis": "integrity", "better": "higher",
        "rel_tol": 0.0, "abs_floor": 1.0, "hard_min": 90.0,
    },
}


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_geometry(eye_width, eye_height):
    scale_x, scale_y = geometry_contract.source_content_scales(
        1920, 1080, eye_width, eye_height
    )
    return geometry_contract.geometry_tuple({
        "source_width": 1920,
        "source_height": 1080,
        "eye_width": eye_width,
        "eye_height": eye_height,
        "content_scale_x": scale_x,
        "content_scale_y": scale_y,
        "disparity_raster_width": eye_width,
        "disparity_raster_height": eye_height,
    })


def write_multiscale_provenance(
        root, clip, source_frame_ids, label_frame_ids, output_frame_ids,
        scale, conf_sha256, metric_sha256, *, cached=False):
    destination = root / "multiscale_provenance" / clip
    destination.mkdir(parents=True)
    slug = builder.multiscale_batch.scale_slug(scale)
    bits = builder.multiscale_batch.scale_float32_bits(scale)
    label_frames_sha256 = digest("label-frame-manifest")
    depth_origin = {
        "mode": "scored-result-cache" if cached else "disabled",
        "key_sha256": digest("depth-key") if cached else "",
        "manifest_sha256": digest("depth-manifest") if cached else "",
        "boundary": "completed-production-depth-state-before-warp-prefilter",
        "selected_state_frame_count": len(output_frame_ids),
        "runtime_scene_frame_count": len(source_frame_ids),
    }
    contract = {
        "artistic_policy": False,
        "artistic_scale_override": scale,
        "depth_reuse_interval": 1,
        "depth_step": "current-once",
        "metric_sha256": metric_sha256,
        "multiscale_batch": True,
        "multiscale_batch_contract":
            builder.multiscale_batch.HARNESS_CONTRACT,
        "multiscale_scale_float32_bits": bits,
        "multiscale_scale_index": 0,
        "output_selection_mode": "label-frames",
        "label_frame_ids": label_frame_ids,
        "output_selected_frame_ids": output_frame_ids,
        "output_label_frames_sha256": label_frames_sha256,
        "depth_state_cache_mode": depth_origin["mode"],
        "depth_state_cache_key_sha256": depth_origin["key_sha256"],
        "depth_state_manifest_sha256": depth_origin["manifest_sha256"],
    }
    contract_path = destination / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    contract_sha256 = builder.sha256_file(contract_path)
    scale_rows = [{
        "index": 0, "scale": scale, "float32_bits": bits,
        "directory": f"scales/{slug}",
    }]
    harness = {
        "schema": builder.multiscale_batch.HARNESS_SCHEMA,
        "contract": builder.multiscale_batch.HARNESS_CONTRACT,
        "scope": "offline-sbs-bench-only",
        "shipping_estimator_calls_per_source_frame": 0 if cached else 1,
        "depth_state_cache": depth_origin,
        "common_directory": "common",
        "source_frame_ids": source_frame_ids,
        "label_frame_ids": label_frame_ids,
        "output_selected_frame_ids": output_frame_ids,
        "output_selection_mode": "label-frames",
        "output_label_frames_sha256": label_frames_sha256,
        "source_frame_count": len(source_frame_ids),
        "output_frame_count_per_scale": len(output_frame_ids),
        "scale_rows": scale_rows,
    }
    harness_path = destination / builder.multiscale_batch.HARNESS_MANIFEST
    harness_path.write_text(json.dumps(harness), encoding="utf-8")
    manifest = {
        "schema": builder.multiscale_batch.SCHEMA,
        "contract": builder.multiscale_batch.CONTRACT,
        "clip": clip,
        "clip_sha1": "a" * 12,
        "executable_sha256": digest("sunshine"),
        "conf_sha256": conf_sha256,
        "metric_sha256": metric_sha256,
        "harness_contract_sha256": builder.sha256_file(harness_path),
        "depth_state_cache": depth_origin,
        "source_frame_ids": source_frame_ids,
        "label_frame_ids": label_frame_ids,
        "output_selected_frame_ids": output_frame_ids,
        "output_selection_mode": "label-frames",
        "output_label_frames_sha256": label_frames_sha256,
        "common_artifacts": [],
        "scale_rows": [{
            **scale_rows[0], "contract_sha256": contract_sha256,
            "artifacts": [],
        }],
    }
    manifest_path = destination / builder.multiscale_batch.MANIFEST
    manifest_path.write_bytes(
        builder.multiscale_batch.canonical_bytes(manifest)
    )
    receipt = {
        "schema": builder.run_multiscale_eval.RENDER_IDENTITY_SCHEMA,
        "contract": builder.run_multiscale_eval.RENDER_IDENTITY_CONTRACT,
        "render_identity_sha256": digest("render-identity"),
        "batch_manifest_sha256": builder.sha256_file(manifest_path),
    }
    (destination / builder.run_multiscale_eval.RENDER_IDENTITY_FILENAME).write_bytes(
        builder.multiscale_batch.canonical_bytes(receipt)
    )
    return {
        "manifest_sha256": builder.sha256_file(manifest_path),
        "contract_sha256": contract_sha256,
        "label_frames_sha256": label_frames_sha256,
    }


def make_grid():
    variant = input_color.sdr_input_variant()
    variant_sha256 = input_color.input_variant_sha256(variant)
    geometries = (make_geometry(1280, 720), make_geometry(1920, 1080))
    source_frame_ids = list(range(10, 16))
    label_frame_ids = [10, 11, 12]
    output_frame_ids = [10, 11, 12]
    common = {
        "clip": "movie-shot",
        "metric_sha256": builder.run_eval.metric_contract_sha(),
        "thresholds_sha256": builder.sha256_file(
            builder.THRESHOLDS_PATH
        ),
        "clip_sha1": "a" * 12,
        "expected_flat": False,
        "pipeline_without_scale": {"model": "dav2-small"},
        "output_selection_contract":
            builder.run_eval.SELECTED_FRAME_GATE_OUTPUT_SELECTION_CONTRACT,
        "source_frame_count": len(source_frame_ids),
        "source_frame_ids": source_frame_ids,
        "source_frame_ids_sha256":
            builder.run_eval.frame_id_sequence_sha256(source_frame_ids),
        "label_frame_count": len(label_frame_ids),
        "label_frame_ids": label_frame_ids,
        "output_frame_count": len(output_frame_ids),
        "output_selected_frame_ids": output_frame_ids,
        "output_label_frames_sha256": digest("label-frames"),
        "runtime_scene_count": 1,
        "completion_sequence_contract":
            "exact for this synchronous harness sequence; live "
            "busy-drop cadence is not replayed",
        "executable_sha256": digest("sunshine"),
    }
    runtime_scene_trace = [{
        "source_frame_ordinal": ordinal,
        "source_frame_id": frame_id,
        "runtime_scene_id": 0,
        "scene_age": float(ordinal),
        "subject_initialized": True,
        "hard_cut": False,
        "scene_start": ordinal == 0,
    } for ordinal, frame_id in enumerate(source_frame_ids)]
    runs = []
    for geometry_index, geometry in enumerate(geometries):
        geometry_sha256 = builder.canonical_sha256(geometry)
        for scale in ordinal_contract.SCALES:
            frames = []
            for ordinal, frame_id in enumerate((10, 11, 12)):
                halo = 4.0
                coverage = 99.0
                if frame_id == 11:
                    failure_scale = 1.06 if geometry_index == 0 else 1.04
                    if scale >= failure_scale - 1e-8:
                        halo = 6.0
                if frame_id == 12:
                    coverage = 80.0
                frames.append({
                    "frame_id": frame_id,
                    "ordinal": ordinal,
                    "source_ordinal": ordinal,
                    "runtime_scene_id": 0,
                    "runtime_scene_evidence": {
                        "source_frame_ordinal": ordinal,
                        "source_frame_id": frame_id,
                        "runtime_scene_id": 0,
                        "scene_age": float(ordinal),
                        "subject_initialized": True,
                        "hard_cut": False,
                        "scene_start": frame_id == 10,
                    },
                    "artifact_sha256": {
                        "source": digest(f"source:{frame_id}"),
                        "depth": digest(f"depth:{frame_id}"),
                    },
                    "metrics": {
                        "exact_pop_spread_pct": 2.0 + (scale - 1.0) * 3.0,
                        "source_halo_p95": halo,
                        "flow_temporal_error_p95": (
                            None if frame_id == 10 else 0.2
                        ),
                        "source_coverage_pct": coverage,
                    },
                })
            runs.append({
                "scale": scale,
                "clip": "movie-shot",
                "geometry": geometry,
                "geometry_sha256": geometry_sha256,
                "input_variant": variant,
                "input_variant_sha256": variant_sha256,
                "common_identity": common,
                "run_identity": {
                    "frame_gate_evidence_sha256": digest(
                        f"sidecar:{geometry_index}:{scale:.2f}"
                    ),
                    "results_sha256": digest(
                        f"results:{geometry_index}:{scale:.2f}"
                    ),
                    "harness_contract_sha256": digest(
                        f"harness:{geometry_index}:{scale:.2f}"
                    ),
                    "runtime_scene_evidence_sha256": digest("runtime-scenes"),
                    "multiscale_batch_manifest_sha256": digest(
                        f"batch:{geometry_index}"
                    ),
                    "multiscale_harness_contract_sha256": digest(
                        f"multiscale-harness:{geometry_index}"
                    ),
                    "multiscale_scale_contract_sha256": digest(
                        f"harness:{geometry_index}:{scale:.2f}"
                    ),
                    "run_name": f"g{geometry_index}-s{scale:.2f}",
                    "geometry_sha256": geometry_sha256,
                    "scale": scale,
                },
                "frames": frames,
                "_runtime_scene_trace": runtime_scene_trace,
            })
    return runs, variant_sha256


class OrdinalFrameLabelBundleTests(unittest.TestCase):
    def test_build_rejects_render_evidence_from_stale_metric_contract(self):
        runs, variant_sha256 = make_grid()
        runs[0]["common_identity"]["metric_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "current metric thresholds"):
            builder.build_frame_label_bundle(runs, METRICS, variant_sha256)

    def test_frame_gate_adapter_authenticates_results_geometry_and_input(self):
        variant = input_color.sdr_input_variant()
        geometry = make_geometry(1280, 720)
        thresholds = {"metrics": METRICS}
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            results = root / "results.json"
            source_frame_ids = list(range(3, 9))
            label_frame_ids = [4, 7]
            output_frame_ids = [4, 7]
            provenance = write_multiscale_provenance(
                root, "movie-shot", source_frame_ids, label_frame_ids,
                output_frame_ids, 1.0, "e" * 16, "d" * 16,
                cached=True,
            )
            cache_contract = builder.ordinal_result_cache.PACKET_CONTRACT
            score_cache = {
                "contract": cache_contract,
                "retained_provenance": "cached-semantic-origin",
            }
            results_payload = {
                "meta": {
                    "precomputed_multiscale": True,
                    "multiscale_batch_manifest_sha256":
                        provenance["manifest_sha256"],
                    "ordinal_score_cache": score_cache,
                },
            }
            results.write_text(
                json.dumps(results_payload) + "\n", encoding="utf-8"
            )
            artifacts = {}
            for frame_id in output_frame_ids:
                artifacts[frame_id] = {}
                for name in (
                        "source", "sbs", "depth", "warp_mask",
                        "warp_disparity"):
                    path = root / f"{name}_{frame_id}.bin"
                    path.write_bytes(f"{name}:{frame_id}".encode("ascii"))
                    artifacts[frame_id][name] = str(path)
            context = {
                "source_frame_ids": source_frame_ids,
                "output_selection": {
                    "mode": "label-frames",
                    "output_frame_ids": output_frame_ids,
                    "label_frame_ids": label_frame_ids,
                    "label_frames_sha256": provenance[
                        "label_frames_sha256"
                    ],
                },
                "artifact_paths": artifacts,
                "clip_sha1": "a" * 12,
                "harness_contract_sha256": provenance["contract_sha256"],
                "expected_flat": False,
                "geometry": geometry,
                "color": scalar_selector._input_variant_harness_context(variant),
                "pipeline": {
                    "model": "dav2-small", "profile": "apollo",
                    "depth_step": "current-once", "depth_reuse_interval": 1,
                    "adaptive_pop": False, "adaptive_pop_max": 1.3,
                    "pop_strength": 1.25, "artistic_style": "immersive",
                    "artistic_policy": False, "artistic_scale_override": 1.0,
                    "policy_warp_source_sha256": "c" * 64,
                },
            }
            rows = [{
                "_frame_id": frame_id,
                "exact_pop_spread_pct": 2.0,
                "source_halo_p95": 4.0,
                "flow_temporal_error_p95": None if frame_id == 3 else 0.2,
                "source_coverage_pct": 99.0,
            } for frame_id in output_frame_ids]
            scene_dir = root / "movie-shot"
            scene_dir.mkdir()
            (scene_dir / builder.RUNTIME_SCENE_FILENAME).write_text(
                json.dumps({
                    "schema": 1,
                    "contract": builder.scene_contract.CONTRACT,
                    "evidence_source":
                        "SubjectState[0].y after completed depth postprocess",
                    "cut_rule":
                        "prior_scene_age_gte_7_and_current_scene_age_eq_0",
                    "cadence": "completed-depth-frames-only",
                    "completion_sequence_contract":
                        "exact for this synchronous harness sequence; live "
                        "busy-drop cadence is not replayed",
                    "depth_reuse_interval": 1,
                    "source_frame_ids": source_frame_ids,
                    "completed_source_frame_ids": source_frame_ids,
                    "completed_depth_frame_count": len(source_frame_ids),
                    "frames": [{
                        "source_frame_ordinal": ordinal,
                        "source_frame_id": frame_id,
                        "runtime_scene_id": 0,
                        "scene_age": float(ordinal),
                        "subject_initialized": True,
                        "hard_cut": False,
                        "scene_start": ordinal == 0,
                    } for ordinal, frame_id in enumerate(source_frame_ids)],
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            scene_path = scene_dir / builder.RUNTIME_SCENE_FILENAME
            context["runtime_scene_evidence"] = builder.scene_contract.load(
                scene_path
            )
            context["runtime_scene_evidence_path"] = str(scene_path)
            records = builder.run_eval.build_frame_gate_clip_records(
                "movie-shot", rows, thresholds, context
            )
            sidecar = root / builder.run_eval.FRAME_GATE_EVIDENCE_FILENAME
            builder.run_eval.write_frame_gate_evidence(
                str(sidecar),
                {
                    "metric_sha256": "d" * 16,
                    "conf_sha256": "e" * 16,
                    "clip_hash_manifest_sha256": "f" * 64,
                    "clip_set_sha1": {"movie-shot": "a" * 12},
                    "run_name": "adapter-test", "suite": "core",
                    "hdr_source_kind": "native-sdr",
                    "precomputed_multiscale": True,
                    "multiscale_batch_manifest_sha256":
                        provenance["manifest_sha256"],
                },
                thresholds, [records], builder.sha256_file(results),
                evidence_contract=(
                    builder.run_eval.SELECTED_FRAME_GATE_EVIDENCE_CONTRACT
                ),
            )
            parsed = builder.parse_frame_gate_evidence(
                sidecar, "movie-shot"
            )
            self.assertEqual(parsed["geometry"], geometry)
            self.assertEqual(
                parsed["input_variant_sha256"],
                input_color.input_variant_sha256(variant),
            )
            self.assertEqual(
                [frame["frame_id"] for frame in parsed["frames"]], [4, 7]
            )
            self.assertEqual(
                [frame["source_ordinal"] for frame in parsed["frames"]],
                [1, 4],
            )
            self.assertEqual(
                parsed["common_identity"]["source_frame_ids"],
                source_frame_ids,
            )
            self.assertEqual(
                parsed["common_identity"]["output_selected_frame_ids"],
                output_frame_ids,
            )
            harness_path = (
                root / "multiscale_provenance" / "movie-shot" /
                builder.multiscale_batch.HARNESS_MANIFEST
            )
            original_harness = harness_path.read_bytes()
            harness_path.write_bytes(original_harness + b"\n")
            with self.assertRaisesRegex(
                    RuntimeError, "multiscale harness contract"):
                builder.parse_frame_gate_evidence(sidecar, "movie-shot")
            harness_path.write_bytes(original_harness)
            results.write_text('{"tampered":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "results identity"):
                builder.parse_frame_gate_evidence(sidecar, "movie-shot")

    def test_builds_per_frame_intersections_and_retains_lineage(self):
        runs, variant_sha256 = make_grid()
        header, frames = builder.build_frame_label_bundle(
            runs, METRICS, variant_sha256
        )
        self.assertEqual(header["frame_count"], 3)
        self.assertEqual(header["schema"], 6)
        self.assertEqual(header["source_frame_ids"], list(range(10, 16)))
        self.assertEqual(
            [row["source_frame_id"] for row in
             header["runtime_scene_trace"]],
            list(range(10, 16)),
        )
        self.assertEqual(header["label_frame_ids"], [10, 11, 12])
        self.assertEqual(header["output_selected_frame_ids"], [10, 11, 12])
        self.assertEqual(
            [frame["source_ordinal"] for frame in frames], [0, 1, 2]
        )
        self.assertEqual(header["code_identity"], builder.current_code_identity())
        self.assertEqual(
            header["code_identity_sha256"],
            builder.canonical_sha256(header["code_identity"]),
        )
        self.assertEqual(len(header["scale_run_identities"]), 52)
        self.assertEqual(
            len(header["deployment_geometry_allowlist"]["tuples"]), 2
        )

        safe, bounded, identity_failure = (
            frame["geometry_intersection"] for frame in frames
        )
        self.assertTrue(safe["right_censored"])
        self.assertEqual(bounded["highest_proven_safe_scale"], 1.02)
        self.assertEqual(bounded["first_proven_unsafe_scale"], 1.04)
        self.assertTrue(identity_failure["left_censored"])
        self.assertFalse(identity_failure["identity_feasible"])
        self.assertEqual([frame["runtime_scene_id"] for frame in frames], [0, 0, 0])
        self.assertEqual(frames[0]["frame_safety_evidence"]["status"], "proven")
        self.assertFalse(frames[0]["frame_safety_evidence"][
            "scene_boundary_exemption"
        ])
        self.assertEqual(
            frames[0]["model_input_provenance"]["source_artifact_sha256"],
            digest("source:10"),
        )
        self.assertEqual(
            frames[0]["model_input_provenance"]["model_input_artifact_sha256"],
            None,
        )
        self.assertEqual(
            set(frames[0]["scale_run_evidence_sha256_by_geometry"]),
            {
                builder.canonical_sha256(value)
                for value in
                header["deployment_geometry_allowlist"]["tuples"]
            },
        )

    def test_temporal_null_is_excluded_from_target_only_safety(self):
        runs, variant_sha256 = make_grid()
        for run in runs:
            run["frames"][1]["metrics"]["flow_temporal_error_p95"] = None
        _header, frames = builder.build_frame_label_bundle(
            runs, METRICS, variant_sha256
        )
        self.assertEqual(frames[1]["frame_safety_evidence"]["status"], "proven")
        self.assertIsNotNone(frames[1]["geometry_intersection"])
        self.assertEqual(
            frames[1]["frame_safety_evidence"]["missing_required_evidence"], []
        )

    def test_partial_temporal_publication_is_ignored_target_only(self):
        runs, variant_sha256 = make_grid()
        runs[0]["frames"][0]["metrics"]["flow_temporal_error_p95"] = 0.2
        _header, frames = builder.build_frame_label_bundle(
            runs, METRICS, variant_sha256
        )
        self.assertEqual(frames[0]["frame_safety_evidence"]["status"], "proven")
        self.assertFalse(frames[0]["frame_safety_evidence"][
            "scene_boundary_exemption"
        ])
        self.assertIsNotNone(frames[0]["geometry_intersection"])

    def test_diagnostic_absolute_violations_are_retained_not_reclassified(self):
        runs, variant_sha256 = make_grid()
        runs[0]["frames"][0]["diagnostic_violations"] = [{
            "metric": "source_halo_p95", "kind": "trigger_max",
            "bound": 3.5, "value": 4.0,
        }]
        _header, frames = builder.build_frame_label_bundle(
            runs, METRICS, variant_sha256
        )
        self.assertEqual(frames[0]["diagnostic_absolute_violations"][0][
            "violations"
        ][0]["kind"], "trigger_max")
        self.assertTrue(frames[0]["geometry_intersection"]["right_censored"])

    def test_grid_fails_closed_on_missing_scale_frame_or_source(self):
        runs, variant_sha256 = make_grid()
        cases = []
        missing_scale = copy.deepcopy(runs)
        del missing_scale[-1]
        cases.append((missing_scale, "every 1.00"))
        missing_frame = copy.deepcopy(runs)
        missing_frame[-1]["frames"].pop()
        cases.append((missing_frame, "same ordered targets"))
        changed_source = copy.deepcopy(runs)
        changed_source[-1]["frames"][0]["artifact_sha256"]["source"] = digest(
            "different"
        )
        cases.append((changed_source, "source-frame bytes"))
        changed_depth = copy.deepcopy(runs)
        changed_depth[-1]["frames"][0]["artifact_sha256"]["depth"] = digest(
            "different-depth"
        )
        cases.append((changed_depth, "model depth"))
        for candidate, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(RuntimeError, pattern):
                    builder.build_frame_label_bundle(
                        candidate, METRICS, variant_sha256
                    )

    def test_scene_identity_must_be_present_and_equal_in_every_run(self):
        runs, variant_sha256 = make_grid()
        runs[-1]["frames"][1]["runtime_scene_id"] = None
        with self.assertRaisesRegex(RuntimeError, "scene identity"):
            builder.build_frame_label_bundle(runs, METRICS, variant_sha256)

    def test_full_runtime_scene_trace_must_match_every_run_and_target(self):
        runs, variant_sha256 = make_grid()
        runs[-1]["_runtime_scene_trace"][4]["scene_age"] = 99.0
        with self.assertRaisesRegex(RuntimeError, "runtime scene trace"):
            builder.build_frame_label_bundle(runs, METRICS, variant_sha256)

        runs, variant_sha256 = make_grid()
        runs[-1]["frames"][1]["runtime_scene_evidence"]["scene_age"] = 8.0
        with self.assertRaisesRegex(
                RuntimeError, "(?:scene evidence|full trace)"):
            builder.build_frame_label_bundle(runs, METRICS, variant_sha256)

    def test_bundle_is_canonical_digested_and_tamper_evident(self):
        runs, variant_sha256 = make_grid()
        header, frames = builder.build_frame_label_bundle(
            runs, METRICS, variant_sha256
        )
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "labels.jsonl"
            summary_path = Path(root) / "summary.json"
            summary = builder.write_frame_label_bundle(
                output, summary_path, header, frames
            )
            records = builder.validate_frame_label_bundle(output)
            self.assertEqual(len(records), 5)
            self.assertEqual(summary["frame_count"], 3)
            self.assertEqual(summary["source_frame_count"], 6)
            self.assertEqual(summary["output_frame_count"], 3)
            self.assertEqual(
                json.loads(summary_path.read_text(encoding="utf-8")), summary
            )
            self.assertEqual(
                builder.validate_frame_label_summary(
                    output, summary_path, records=records
                ),
                summary,
            )
            raw = output.read_bytes().replace(
                b'"frame_id":10', b'"frame_id":19', 1
            )
            output.write_bytes(raw)
            with self.assertRaisesRegex(RuntimeError, "digest/count"):
                builder.validate_frame_label_bundle(output)

    def test_summary_rejects_identity_aggregate_and_encoding_corruption(self):
        runs, variant_sha256 = make_grid()
        header, frames = builder.build_frame_label_bundle(
            runs, METRICS, variant_sha256
        )
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "labels.jsonl"
            summary_path = Path(root) / "summary.json"
            summary = builder.write_frame_label_bundle(
                output, summary_path, header, frames
            )
            corruptions = {
                "schema": builder.SUMMARY_SCHEMA + 1,
                "contract": builder.SUMMARY_CONTRACT + "-stale",
                "frame_count": summary["frame_count"] + 1,
                "frontier_bounds": {"fabricated-frontier": 1},
                "unproven_required_metrics": {"fabricated-metric": 1},
                "payload_sha256": "0" * 64,
                "label_bundle_sha256": "0" * 64,
            }
            for field, value in corruptions.items():
                with self.subTest(field=field):
                    candidate = copy.deepcopy(summary)
                    candidate[field] = value
                    summary_path.write_bytes(builder.canonical_bytes(candidate))
                    with self.assertRaisesRegex(
                            RuntimeError, "authenticated bundle"):
                        builder.validate_frame_label_summary(
                            output, summary_path
                        )

            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "not canonical"):
                builder.validate_frame_label_summary(output, summary_path)


if __name__ == "__main__":
    unittest.main()
