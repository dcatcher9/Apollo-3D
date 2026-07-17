#!/usr/bin/env python3

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import torch

import artistic_geometry_contract as geometry_contract
import artistic_policy_ordinal_contract as ordinal_contract
import depth_input_color as input_color
import merge_ordinal_geometry_frontiers as ordinal_merge
import train_artistic_policy_ordinal as trainer


SOURCE = "a" * 64


def geometry(width, height):
    scale_x, scale_y = geometry_contract.source_content_scales(
        1920, 1080, width, height
    )
    return geometry_contract.geometry_tuple({
        "source_width": 1920,
        "source_height": 1080,
        "eye_width": width,
        "eye_height": height,
        "content_scale_x": scale_x,
        "content_scale_y": scale_y,
        "disparity_raster_width": width,
        "disparity_raster_height": height,
    })


def frontier(failure_index=None, gain=0.1, cause="halo:hard"):
    end = (
        ordinal_contract.FRONTIER_SIZE - 1
        if failure_index is None else failure_index
    )
    tested = []
    for index in range(end + 1):
        safe = failure_index is None or index < failure_index
        tested.append({
            "scale": ordinal_contract.SCALES[index],
            "safe": safe,
            "realized_pop_pct": gain * min(index, max(0, end - 1)),
            "failure_causes": [] if safe else [cause],
        })
    return ordinal_contract.build_frontier_evidence(tested)


def intersection(failure_index=None, gain=0.1, cause="halo:hard"):
    records = []
    for width, height in ((1280, 720), (1920, 1080)):
        records.append(ordinal_merge.build_geometry_frontier(
            SOURCE,
            "b" * 64,
            geometry(width, height),
            frontier(failure_index, gain, cause),
        ))
    return ordinal_merge.intersect_geometry_frontiers(records)


def row(variant, target, *, film="film", clip="clip", frame=0,
        domain="movie"):
    return {
        "row_role": "target",
        "_ordinal_intersection": target,
        "_ordinal_safety_evidence": {
            "schema": 1,
            "contract": "apollo-ordinal-frame-safety-evidence-v1",
            "status": "proven",
            "scene_boundary_exemption": False,
            "missing_required_evidence": [],
            "exempt_missing_temporal_evidence": [],
        },
        "_input_variant": variant,
        "_input_variant_sha256": input_color.input_variant_sha256(variant),
        "_runtime_scene_id": "scene-0",
        "_runtime_scene_evidence": {
            "source_frame_id": frame,
            "runtime_scene_id": 0,
            "hard_cut": False,
        },
        "film_id": film,
        "clip": clip,
        "frame": frame,
        "source_frame_rate": 1.0,
        "domain": domain,
        "global_policy_weight": 1.0,
    }


def context_row(variant, *, film="film", clip="clip", frame=0,
                domain="movie"):
    value = row(
        variant, intersection(4), film=film, clip=clip, frame=frame,
        domain=domain,
    )
    value["row_role"] = "context"
    value["_ordinal_intersection"] = None
    value["_ordinal_safety_evidence"] = None
    return value


def safe_probabilities(highest_safe_index, value=0.99):
    probabilities = np.full(ordinal_contract.FRONTIER_SIZE, 0.01, np.float64)
    probabilities[:highest_safe_index + 1] = value
    return probabilities


class OrdinalTrainerTests(unittest.TestCase):
    def test_intersection_adapter_preserves_finite_and_censored_bounds(self):
        cases = (
            (intersection(0), True, False, None, 1.0),
            (intersection(4), False, False, 1.06, 1.08),
            (intersection(None), False, True, 1.5, None),
        )
        for target, left, right, highest, first_unsafe in cases:
            with self.subTest(left=left, right=right):
                evidence = trainer.intersection_loss_evidence(target)
                self.assertEqual(evidence["left_censored"], left)
                self.assertEqual(evidence["right_censored"], right)
                self.assertEqual(evidence["highest_proven_safe_scale"], highest)
                self.assertEqual(
                    evidence["first_proven_unsafe_scale"], first_unsafe
                )

    def test_sampling_gives_sdr_and_hdr_equal_total_mass(self):
        sdr = input_color.sdr_input_variant()
        hdrs = [
            input_color.windows_hdr_input_variant(level)
            for level in (1000, 2500, 6000)
        ]
        rows = [
            row(sdr, intersection(3), clip="sdr-a", frame=index)
            for index in range(5)
        ]
        for variant_index, variant in enumerate(hdrs):
            rows.extend(
                row(variant, intersection(3 + variant_index),
                    clip=f"hdr-{variant_index}", frame=index)
                for index in range(2 + variant_index)
            )
        weights = trainer.balanced_ordinal_sample_weights(rows)
        totals = {"sdr": 0.0, "hdr": 0.0}
        for sample, weight in zip(rows, weights):
            totals[trainer.runtime_regime(sample["_input_variant"])] += weight
        self.assertAlmostEqual(totals["sdr"], 0.5)
        self.assertAlmostEqual(totals["hdr"], 0.5)

    def test_sampling_balances_frontier_strata_inside_regime(self):
        sdr = input_color.sdr_input_variant()
        rows = [row(sdr, intersection(2), clip="finite", frame=0)]
        rows.extend(
            row(sdr, intersection(None), clip="right", frame=index)
            for index in range(4)
        )
        weights = trainer.balanced_ordinal_sample_weights(rows)
        totals = {}
        for sample, weight in zip(rows, weights):
            key = trainer.frontier_stratum(sample)
            totals[key] = totals.get(key, 0.0) + weight
        self.assertEqual(len(totals), 2)
        self.assertAlmostEqual(*totals.values())

    def test_frontier_stratum_keeps_exact_ceiling_and_failure_family(self):
        variant = input_color.sdr_input_variant()
        ceiling_two = row(
            variant, intersection(3, cause="halo:hard"), clip="two"
        )
        ceiling_three = row(
            variant, intersection(4, cause="halo:hard"), clip="three"
        )
        coverage = row(
            variant, intersection(3, cause="coverage:hard"), clip="coverage"
        )
        self.assertEqual(
            trainer.frontier_stratum(ceiling_two),
            "finite:safe-1.04:halo",
        )
        self.assertEqual(
            len({trainer.frontier_stratum(item) for item in (
                ceiling_two, ceiling_three, coverage
            )}),
            3,
        )

    def test_sampling_distinguishes_same_clip_name_across_films(self):
        variant = input_color.sdr_input_variant()
        rows = [row(variant, intersection(3), film="film-a", frame=0)]
        rows.extend(
            row(variant, intersection(3), film="film-b", frame=index)
            for index in range(3)
        )
        weights = trainer.balanced_ordinal_sample_weights(rows)
        totals = {"film-a": 0.0, "film-b": 0.0}
        for sample, weight in zip(rows, weights):
            totals[sample["film_id"]] += weight
        self.assertAlmostEqual(totals["film-a"], totals["film-b"])

    def test_selection_abstains_and_never_jumps_a_failed_bin(self):
        probabilities = np.linspace(0.99, 0.70, ordinal_contract.FRONTIER_SIZE)
        probabilities[3] = 0.90
        probabilities[4:] = 0.98
        # Non-monotone predictions fail closed instead of allowing a jump.
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            trainer.select_scale(probabilities)
        self.assertIsNone(trainer.select_scale(np.full(26, 0.94)))
        connected = np.r_[np.full(3, 0.99), np.full(23, 0.90)]
        self.assertEqual(trainer.select_scale(connected), 1.04)

    def test_positive_affine_calibration_preserves_monotonicity(self):
        evidence = [
            trainer.intersection_loss_evidence(intersection(4)),
            trainer.intersection_loss_evidence(intersection(None)),
        ]
        probability = torch.stack((
            torch.linspace(0.95, 0.05, 26),
            torch.linspace(0.99, 0.80, 26),
        ))
        calibration = trainer.fit_shared_affine_calibration(
            probability, evidence, steps=5
        )
        calibrated = calibration.apply(probability)
        self.assertGreater(calibration.slope, 0.0)
        self.assertTrue(torch.all(calibrated[:, 1:] <= calibrated[:, :-1]))

    def test_development_selection_prioritizes_zero_overshoot(self):
        sdr = input_color.sdr_input_variant()
        hdr = input_color.windows_hdr_input_variant(2500)
        target = intersection(4, gain=0.25)
        rows = [
            row(sdr, target, film="sdr-film"),
            row(hdr, target, film="hdr-film"),
        ]
        safe = np.stack((safe_probabilities(3), safe_probabilities(3)))
        risky = np.stack((safe_probabilities(4), safe_probabilities(4)))
        safe_eval = trainer.evaluate_predictions(safe, rows)
        risky_eval = trainer.evaluate_predictions(risky, rows)
        self.assertEqual(trainer.checkpoint_selection_key(safe_eval)[2], 0)
        self.assertGreater(trainer.checkpoint_selection_key(risky_eval)[2], 0)
        self.assertLess(
            trainer.checkpoint_selection_key(safe_eval),
            trainer.checkpoint_selection_key(risky_eval),
        )
        self.assertFalse(safe_eval["calibration_deployable"])
        self.assertFalse(safe_eval["development_candidate_status"][
            "training_checkpoint_eligible"
        ])
        self.assertIsNone(safe_eval["development_candidate_status"][
            "minimum_realized_pop_gain_pct"
        ])
        self.assertEqual(
            safe_eval["calibration_evidence"]["sdr"][
                "minimum_independent_groups_required"
            ],
            59,
        )

    def test_zero_gain_abstention_cannot_be_checkpoint_eligible(self):
        sdr = input_color.sdr_input_variant()
        hdr = input_color.windows_hdr_input_variant(2500)
        target = intersection(4, gain=0.25)
        rows = [
            row(sdr, target, film="sdr-film"),
            row(hdr, target, film="hdr-film"),
        ]
        abstain = np.full((2, ordinal_contract.FRONTIER_SIZE), 0.90)
        result = trainer.evaluate_predictions(
            abstain, rows,
            minimum_development_pop_gain_pct=0.01,
            minimum_development_pop_gain_rationale=(
                "frozen unit-test gain threshold before predictions"
            ),
        )
        status = result["development_candidate_status"]
        self.assertTrue(status["zero_overshoot_pass"])
        self.assertFalse(status["material_gain_pass"])
        self.assertFalse(status["training_checkpoint_eligible"])
        self.assertFalse(status["production_policy_accepted"])

    def test_unproven_frame_is_excluded_and_forces_controller_identity(self):
        sdr = input_color.sdr_input_variant()
        hdr = input_color.windows_hdr_input_variant(2500)
        rows = [
            row(sdr, intersection(4), film="sdr-film", frame=0),
            row(hdr, intersection(4), film="hdr-film", frame=0),
        ]
        unproven = row(sdr, None, film="sdr-film", frame=1)
        unproven["_ordinal_safety_evidence"]["status"] = "unproven"
        unproven["_ordinal_safety_evidence"]["missing_required_evidence"] = [{
            "metric": "flow_temporal_error_p95",
            "geometry_sha256": "c" * 64,
            "scales": [1.0],
        }]
        rows.insert(1, unproven)
        probabilities = np.stack((
            safe_probabilities(3), safe_probabilities(25),
            safe_probabilities(3),
        ))
        result = trainer.evaluate_predictions(
            probabilities, rows,
            minimum_development_pop_gain_pct=0.01,
            minimum_development_pop_gain_rationale=(
                "frozen unit-test gain threshold before predictions"
            ),
        )
        self.assertEqual(result["regimes"]["sdr"]["macro"][
            "unproven_samples"
        ], 1)
        self.assertAlmostEqual(result["regimes"]["sdr"]["macro"][
            "safety_evidence_unproven"
        ], 0.5)
        self.assertFalse(result["development_candidate_status"][
            "complete_evidence_pass"
        ])
        self.assertFalse(result["development_candidate_status"][
            "training_checkpoint_eligible"
        ])

    def test_context_rows_are_rejected_from_target_only_evaluation(self):
        sdr = input_color.sdr_input_variant()
        hdr = input_color.windows_hdr_input_variant(2500)
        rows = [
            context_row(sdr, film="sdr-film", frame=0),
            row(sdr, intersection(4), film="sdr-film", frame=1),
            context_row(hdr, film="hdr-film", frame=0),
            row(hdr, intersection(4), film="hdr-film", frame=1),
        ]
        probabilities = np.stack([safe_probabilities(3)] * len(rows))
        with self.assertRaisesRegex(RuntimeError, "target row"):
            trainer.evaluate_predictions(probabilities, rows)

    def test_target_only_selection_is_applied_directly(self):
        sdr = input_color.sdr_input_variant()
        hdr = input_color.windows_hdr_input_variant(2500)
        rows = [
            row(sdr, intersection(4), film="sdr-film", frame=7),
            row(hdr, intersection(4), film="hdr-film", frame=23),
        ]
        probabilities = np.stack([safe_probabilities(3)] * 2)
        result = trainer.evaluate_predictions(probabilities, rows)
        self.assertEqual(result["application"], {
            "mode": "direct-target-only",
            "target_rows": 2,
            "context_rows": 0,
            "validation_scope": "independent authenticated safety targets",
        })
        self.assertEqual(
            result["overall"]["macro"]["realized_pop_gain_pct"],
            result["overall"]["macro"]["selected_realized_pop_gain_pct"],
        )

    def test_left_censored_identity_failure_cannot_hide_in_abstention(self):
        sdr = input_color.sdr_input_variant()
        hdr = input_color.windows_hdr_input_variant(2500)
        rows = [
            row(sdr, intersection(0), film="sdr-film"),
            row(hdr, intersection(4), film="hdr-film"),
        ]
        abstain = np.full((2, ordinal_contract.FRONTIER_SIZE), 0.10)
        result = trainer.evaluate_predictions(
            abstain, rows,
            minimum_development_pop_gain_pct=0.01,
            minimum_development_pop_gain_rationale=(
                "frozen unit-test gain threshold before predictions"
            ),
        )
        self.assertEqual(
            result["regimes"]["sdr"]["macro"][
                "identity_hard_failure_count"
            ],
            1,
        )
        status = result["development_candidate_status"]
        self.assertFalse(status["zero_identity_failure_pass"])
        self.assertFalse(status["training_checkpoint_eligible"])

    def test_plateau_is_diagnostic_and_does_not_modify_selected_action(self):
        target = intersection(None)
        self.assertAlmostEqual(
            trainer._plateau_excess_scale(1.50, target), 0.02
        )
        self.assertEqual(trainer._plateau_excess_scale(1.48, target), 0.0)
        rows = [
            row(input_color.sdr_input_variant(), target, film="sdr-film"),
            row(input_color.windows_hdr_input_variant(2500), target,
                film="hdr-film"),
        ]
        predictions = np.full((2, ordinal_contract.FRONTIER_SIZE), 0.99)
        result = trainer.evaluate_predictions(predictions, rows)
        self.assertGreater(
            result["regimes"]["sdr"]["macro"][
                "plateau_excess_selection_count"
            ],
            0,
        )
        self.assertAlmostEqual(
            result["regimes"]["sdr"]["macro"][
                "selected_plateau_excess_scale"
            ],
            0.02,
        )
        first_max_predictions = predictions.copy()
        first_max_predictions[:, -1] = 0.90
        first_max = trainer.evaluate_predictions(first_max_predictions, rows)
        self.assertEqual(trainer.checkpoint_selection_key(first_max)[6], 0.0)
        self.assertGreater(trainer.checkpoint_selection_key(result)[6], 0.0)
        self.assertLess(
            trainer.checkpoint_selection_key(first_max),
            trainer.checkpoint_selection_key(result),
        )

    def test_known_bin_ece_ignores_unknown_tail(self):
        target = intersection(4)
        probabilities = np.zeros(ordinal_contract.FRONTIER_SIZE)
        probabilities[:4] = 1.0
        self.assertEqual(trainer.known_bin_ece(probabilities, target), 0.0)
        probabilities[5:] = 0.95
        self.assertEqual(trainer.known_bin_ece(probabilities, target), 0.0)

    def test_paired_variant_conflict_is_diagnostic_not_label_rejection(self):
        sdr = row(input_color.sdr_input_variant(), intersection(4))
        hdr = row(
            input_color.windows_hdr_input_variant(1000), intersection(5)
        )
        for sample in (sdr, hdr):
            sample.update({
                "source_kind": "mono-video",
                "source_sha256": SOURCE,
                "_model_depth_artifact_sha256": "d" * 64,
            })
        dataset = trainer.CachedOrdinalDataset(
            torch.ones(2, 4), [sdr, hdr]
        )
        audit = trainer.audit_paired_variant_identifiability(
            dataset, "training"
        )
        self.assertEqual(audit["contradictory_near_identical_pairs"], 1)
        self.assertTrue(audit["variant_specific_safety_targets_retained"])
        self.assertFalse(
            audit["label_admission_blocked_by_feature_similarity"]
        )

    def test_coverage_requires_both_regimes_failure_and_gain(self):
        sdr = input_color.sdr_input_variant()
        hdr = input_color.windows_hdr_input_variant(1000)
        valid = [
            row(sdr, intersection(4, gain=0.1)),
            row(hdr, intersection(4, gain=0.1)),
        ]
        trainer.validate_ordinal_coverage(valid, "development")
        with self.assertRaisesRegex(RuntimeError, "native SDR and HDR"):
            trainer.validate_ordinal_coverage(valid[:1], "development")
        no_failure = copy.deepcopy(valid)
        no_failure[1]["_ordinal_intersection"] = intersection(None, gain=0.1)
        with self.assertRaisesRegex(RuntimeError, "measured_failure"):
            trainer.validate_ordinal_coverage(no_failure, "development")

    def test_complete_cardinality_rejects_one_missing_condition_frame(self):
        specifications = (
            ("train-film", "training", "mono-video", 3, 1),
            ("dev-film", "development", "mono-video", 2, 1),
            ("train-hdr-film", "training", "native-hdr-video", 2, 1),
            ("dev-hdr-film", "development", "native-hdr-video", 1, 1),
        )
        active = {
            "productions": [{
                "production_id": film,
                "source_kind": kind,
                "split": split,
                "context_frames": frames,
                "label_frames": targets,
            } for film, split, kind, frames, targets in specifications],
            "_ordinal_expected_sequences": {
                film: {film + "-clip": {
                    "first_frame": 0,
                    "frame_count": frames,
                    "target_count": targets,
                    "target_frame_ids": [frames - 1],
                }} for film, _split, _kind, frames, targets in specifications
            },
        }
        variants = [
            input_color.sdr_input_variant(),
            *(input_color.windows_hdr_input_variant(level)
              for level in (1000, 2500, 6000)),
        ]
        rows = []
        for film, split, kind, frame_count, _targets in specifications:
            conditions = (
                variants if kind == "mono-video"
                else [input_color.native_pq_input_variant()]
            )
            for variant in conditions:
                sample = row(
                    variant, intersection(4), film=film,
                    clip=film + "-clip", frame=frame_count - 1,
                )
                sample["split"] = split
                rows.append(sample)
        evidence = trainer.validate_complete_active_cardinality(rows, active)
        self.assertEqual(
            evidence["training"]["regimes"]["sdr"]["target_rows"], 1
        )
        self.assertEqual(
            evidence["training"]["regimes"]["hdr"]["target_rows"], 4
        )
        with self.assertRaisesRegex(RuntimeError, "cardinality is incomplete"):
            trainer.validate_complete_active_cardinality(rows[:-1], active)

        extra = copy.deepcopy(rows)
        extra.append(context_row(
            input_color.sdr_input_variant(), film="train-film",
            clip="train-film-clip", frame=3,
        ))
        extra[-1]["split"] = "training"
        with self.assertRaisesRegex(RuntimeError, "target row"):
            trainer.validate_complete_active_cardinality(extra, active)

        duplicated = copy.deepcopy(rows)
        duplicated[1] = copy.deepcopy(duplicated[0])
        with self.assertRaisesRegex(RuntimeError, "repeats a frame/condition"):
            trainer.validate_complete_active_cardinality(duplicated, active)

        wrong_split = copy.deepcopy(rows)
        wrong_split[0]["split"] = "development"
        with self.assertRaisesRegex(RuntimeError, "outside the active"):
            trainer.validate_complete_active_cardinality(wrong_split, active)

    def test_source_map_rejects_conflicting_model_inputs(self):
        base = {
            "source_sha256": SOURCE,
            "clip": "clip",
            "frame": 0,
            "input_variant_sha256": "b" * 64,
            "source": "source.png",
            "source_width": 1920,
            "source_height": 1080,
            "model_input_width": 770,
            "model_input_height": 434,
            "model_source": None,
            "model_source_sha256": None,
            "input_variant": input_color.sdr_input_variant(),
            "split": "training",
            "film_id": "film",
            "domain": "movie",
            "source_kind": "mono-video",
            "global_policy_weight": 1.0,
        }
        changed = dict(base)
        changed["model_input_width"] = 756
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            trainer.source_row_map([base, changed])

    def test_image_dataset_reads_and_hashes_only_authoritative_media_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_read_bytes = Path.read_bytes
            calls = []

            def counted_read_bytes(path):
                calls.append(Path(path))
                return original_read_bytes(path)

            native_path = root / "native.f16"
            native_payload = np.arange(8, dtype="<f2").tobytes()
            native_path.write_bytes(native_payload)
            native = {
                "_input_variant": input_color.native_pq_input_variant(),
                "model_source": str(native_path),
                "model_source_sha256": trainer.hashlib.sha256(
                    native_payload
                ).hexdigest(),
                "source": str(root / "preview-must-not-be-opened.png"),
                "source_sha256": "0" * 64,
                "source_width": 2,
                "source_height": 1,
                "model_input_width": 2,
                "model_input_height": 1,
            }
            with (mock.patch.object(Path, "read_bytes", counted_read_bytes),
                  mock.patch.object(
                      trainer.input_color, "preprocess_scrgb_f16_to_nchw",
                      return_value=np.zeros((3, 1, 2), np.float32),
                  )):
                image, index = trainer.OrdinalImageDataset([native])[0]
            self.assertEqual(index, 0)
            self.assertEqual(tuple(image.shape), (3, 1, 2))
            self.assertEqual(calls, [native_path])

            encoded, png = cv2.imencode(
                ".png", np.zeros((2, 3, 3), np.uint8)
            )
            self.assertTrue(encoded)
            source_path = root / "source.png"
            source_payload = png.tobytes()
            source_path.write_bytes(source_payload)
            sdr = {
                "_input_variant": input_color.sdr_input_variant(),
                "source": str(source_path),
                "source_sha256": trainer.hashlib.sha256(
                    source_payload
                ).hexdigest(),
                "source_width": 3,
                "source_height": 2,
                "model_input_width": 3,
                "model_input_height": 2,
            }
            calls.clear()
            with (mock.patch.object(Path, "read_bytes", counted_read_bytes),
                  mock.patch.object(
                      trainer.input_color, "preprocess_rgb8_to_nchw",
                      return_value=np.zeros((3, 2, 3), np.float32),
                  )):
                trainer.OrdinalImageDataset([sdr])[0]
            self.assertEqual(calls, [source_path])
            sdr["source_sha256"] = "f" * 64
            with self.assertRaisesRegex(RuntimeError, "changed after admission"):
                trainer.OrdinalImageDataset([sdr])[0]

    def test_ordinal_split_loader_never_opens_sealed_test_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.json"
            catalog.write_text("{}\n", encoding="utf-8")

            def dataset(production, split):
                path = root / f"{production}.json"
                path.write_text(json.dumps({
                    "schema": 2,
                    "production_id": production,
                    "split": split,
                    "source_kind": "mono-video",
                    "sequences": [{
                        "clip": production + "-clip",
                        "context_frames": 2,
                        "label_frames": 1,
                        "source_start_frame": 11,
                    }],
                }), encoding="utf-8")
                return path

            train = dataset("train", "training")
            development = dataset("development", "development")
            sealed = root / "sealed-must-not-exist.json"

            def production(name, split, manifest, manifest_hash):
                return {
                    "production_id": name,
                    "split": split,
                    "source_kind": "mono-video",
                    "context_frames": 2,
                    "label_frames": 1,
                    "dataset_manifest": str(manifest),
                    "dataset_manifest_sha256": manifest_hash,
                }

            active = root / "active.json"
            active.write_text(json.dumps({
                "schema": 1,
                "catalog": str(catalog),
                "catalog_sha256": trainer.sha256(catalog),
                "split_productions": {
                    "training": ["train"],
                    "development": ["development"],
                    "test": ["sealed"],
                },
                "productions": [
                    production("train", "training", train,
                               trainer.sha256(train)),
                    production("development", "development", development,
                               trainer.sha256(development)),
                    production("sealed", "test", sealed, "a" * 64),
                ],
            }), encoding="utf-8")
            loaded, digest = trainer.load_ordinal_active_split(active)
            self.assertEqual(digest, trainer.sha256(active))
            self.assertFalse(sealed.exists())
            self.assertIn("train", loaded["_ordinal_expected_sequences"])
            train_sequence = loaded["_ordinal_expected_sequences"][
                "train"
            ]["train-clip"]
            self.assertEqual(train_sequence["first_frame"], 0)
            self.assertEqual(train_sequence["source_first_frame"], 11)

    def test_history_envelope_authenticates_exact_completed_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_hash = "a" * 64
            history = [{
                "epoch": 1,
                "checkpoint_selection_contract":
                    trainer.CHECKPOINT_SELECTION_CONTRACT,
            }]
            empty = trainer.build_history_envelope(
                history, 1, contract_hash, root / "missing.pt"
            )
            self.assertTrue(empty["completed"])
            self.assertIsNone(empty["best_checkpoint"])
            checkpoint_path = root / "best.pt"
            torch.save({
                "schema": trainer.ORDINAL_CHECKPOINT_SCHEMA,
                "training_contract_sha256": contract_hash,
                "checkpoint_selection_contract":
                    trainer.CHECKPOINT_SELECTION_CONTRACT,
                "epoch": 1,
            }, checkpoint_path)
            complete = trainer.build_history_envelope(
                history, 1, contract_hash, checkpoint_path
            )
            self.assertEqual(
                complete["best_checkpoint"]["sha256"],
                trainer.sha256(checkpoint_path),
            )
            with self.assertRaisesRegex(RuntimeError, "exact epochs"):
                trainer.build_history_envelope(
                    history, 2, contract_hash, checkpoint_path
                )

    def test_training_contract_derives_depth_identity_only_from_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "depth.pth"
            weights.write_bytes(b"weights")
            args = mock.Mock(
                depth_weights=weights,
                split_manifest=root / "active.json",
                minimum_development_pop_gain_pct=0.1,
                minimum_development_pop_gain_rationale=(
                    "frozen unit-test gain threshold before predictions"
                ),
                seed=7,
            )
            sdr = input_color.sdr_input_variant()
            hdr = input_color.windows_hdr_input_variant(1000)
            train_target = row(
                sdr, intersection(4), film="train", clip="train", frame=1
            )
            development_target = row(
                hdr, intersection(4), film="development",
                clip="development", frame=1,
            )
            for sample, split in (
                    (train_target, "training"),
                    (development_target, "development")):
                sample["split"] = split
            provenance = {"depth_model": "depth_anything_v2_fp16"}
            train_target["_model_input_provenance"] = provenance
            development_target["_model_input_provenance"] = provenance
            train_rows = [train_target]
            development_rows = [development_target]
            allowlist = geometry_contract.build_allowlist([
                geometry(1280, 720), geometry(1920, 1080),
            ])
            common = {
                "metric_specs_sha256": "1" * 64,
                "metric_contract_sha256": "2" * 64,
                "thresholds_sha256": "3" * 64,
                "deployment_geometry_allowlist": allowlist,
                "deployment_geometry_allowlist_sha256":
                    geometry_contract.allowlist_sha256(allowlist),
                "deployment_geometry_structure_sha256":
                    trainer.geometry_structure_sha256(allowlist),
            }
            active = {"split_productions": {
                "training": ["train"],
                "development": ["development"],
                "test": ["sealed"],
            }}
            all_rows = train_rows + development_rows
            contract = trainer.build_training_contract(
                args, [], [], common, active, "4" * 64,
                train_rows, development_rows, [train_target],
                [development_target], {}, {},
                trainer.build_expected_runtime_evidence(all_rows),
            )
            self.assertEqual(
                contract["depth_model"], "depth_anything_v2_fp16"
            )
            self.assertEqual(contract["train_samples"], 1)
            self.assertEqual(
                contract["checkpoint_selection_contract"],
                trainer.CHECKPOINT_SELECTION_CONTRACT,
            )

    def test_training_catalog_authenticates_all_inputs_before_caching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "active.json"
            frontier_labels = root / "frontier.jsonl"
            frontier_summary = root / "frontier-summary.json"
            source_labels = root / "source.jsonl"
            source_summary = root / "source-summary.json"
            source_contract = root / "source-contract.json"
            catalog = root / "catalog.json"
            active.write_text("{}\n", encoding="utf-8")
            frontier_labels.write_text("frontier\n", encoding="utf-8")
            frontier_summary.write_text("{}\n", encoding="utf-8")
            source_labels.write_text("source\n", encoding="utf-8")
            source_summary.write_text("{}\n", encoding="utf-8")
            source_contract.write_text("{}\n", encoding="utf-8")
            code_path = Path(trainer.__file__).resolve()
            payload = {
                "schema": trainer.CATALOG_SCHEMA,
                "contract": trainer.CATALOG_CONTRACT,
                "scope": "full-active-train-development",
                "training_eligible": True,
                "active_split": str(active.resolve()),
                "active_split_sha256": trainer.sha256(active),
                "metric_sha256": trainer.run_eval.metric_contract_sha(),
                "thresholds_sha256": trainer.sha256(
                    trainer.THRESHOLDS_PATH
                ),
                "sbsbench_sha256": trainer.sha256(
                    trainer.ordinal_bundle.SBSBENCH_DIR / "sbsbench.py"
                ),
                "run_eval_sha256": trainer.sha256(
                    Path(trainer.run_eval.__file__)
                ),
                "conf_sha256": "8" * 16,
                "executable_sha256": "9" * 64,
                "code_identities": {
                    "trainer": {
                        "path": str(code_path),
                        "sha256": trainer.sha256(code_path),
                    },
                },
                "bundles": [{
                    "labels": str(frontier_labels.resolve()),
                    "labels_sha256": trainer.sha256(frontier_labels),
                    "summary": str(frontier_summary.resolve()),
                    "summary_sha256": trainer.sha256(frontier_summary),
                }],
                "sources": [{
                    "labels": str(source_labels.resolve()),
                    "labels_sha256": trainer.sha256(source_labels),
                    "summary": str(source_summary.resolve()),
                    "summary_sha256": trainer.sha256(source_summary),
                    "source_contract": str(source_contract.resolve()),
                    "source_contract_sha256": trainer.sha256(
                        source_contract
                    ),
                }],
            }
            catalog.write_text(json.dumps(payload), encoding="utf-8")
            identity = trainer.validate_training_catalog(
                catalog, [frontier_labels], [source_labels], active
            )
            self.assertEqual(identity["sha256"], trainer.sha256(catalog))
            payload["conf_sha256"] = "8" * 64
            catalog.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                    RuntimeError, "authenticated conf_sha256"):
                trainer.validate_training_catalog(
                    catalog, [frontier_labels], [source_labels], active
                )
            payload["conf_sha256"] = "8" * 16
            catalog.write_text(json.dumps(payload), encoding="utf-8")
            source_labels.write_text("stale\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "publication is stale"):
                trainer.validate_training_catalog(
                    catalog, [frontier_labels], [source_labels], active
                )

    def test_ordinal_trainer_rejects_legacy_sparse_source_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels.jsonl"
            labels.write_text(json.dumps({
                "source_schema": 2,
                "source_contract": "full-cadence-artistic-source-v2",
            }) + "\n", encoding="utf-8")
            (root / "summary.json").write_text("{}\n", encoding="utf-8")
            (root / "source_contract.json").write_text(
                "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "stale"):
                trainer.load_source_bundles([labels])

    def test_join_rejects_source_rows_from_a_different_frontier_bundle(self):
        variant = input_color.sdr_input_variant()
        variant_hash = input_color.input_variant_sha256(variant)
        provenance = {
            "source_artifact_sha256": SOURCE,
            "input_variant_sha256": variant_hash,
            "depth_model": "depth_anything_v2_fp16",
            "model_input_artifact_sha256": None,
        }
        provenance_hash = trainer.canonical_sha256(provenance)
        source = {
            "row_role": "target",
            "source_sha256": SOURCE,
            "clip": "clip",
            "frame": 0,
            "input_variant": variant,
            "input_variant_sha256": variant_hash,
            "ordinal_bundle_sha256": "c" * 64,
            "ordinal_frame_model_input_provenance": provenance,
            "ordinal_frame_model_input_provenance_sha256": provenance_hash,
            "ordinal_model_depth_artifact_sha256": "d" * 64,
            "runtime_scene_evidence": {
                "source_frame_id": 0,
                "runtime_scene_id": 0,
                "hard_cut": False,
            },
        }
        frontier_row = {
            "row_role": "target",
            "_join_key": (SOURCE, "clip", 0, variant_hash),
            "_input_variant": variant,
            "_input_variant_sha256": variant_hash,
            "_model_input_provenance": provenance,
            "_model_input_provenance_sha256": provenance_hash,
            "_model_depth_artifact_sha256": "d" * 64,
            "_ordinal_bundle_sha256": "e" * 64,
            "_ordinal_intersection": intersection(4),
            "_ordinal_safety_evidence": {
                "status": "proven",
            },
            "_runtime_scene_evidence": source["runtime_scene_evidence"],
        }
        with self.assertRaisesRegex(RuntimeError, "provenance differs"):
            trainer.join_ordinal_sources([source], [frontier_row])

    def test_join_requires_exact_target_only_subset(self):
        variant = input_color.sdr_input_variant()
        variant_hash = input_color.input_variant_sha256(variant)
        scene = {
            "source_frame_id": 0,
            "runtime_scene_id": 0,
            "hard_cut": False,
        }
        provenance = {
            "source_artifact_sha256": SOURCE,
            "input_variant_sha256": variant_hash,
            "depth_model": "depth_anything_v2_fp16",
            "model_input_artifact_sha256": None,
        }
        provenance_hash = trainer.canonical_sha256(provenance)
        common = {
            "source": "source.png",
            "source_width": 1920,
            "source_height": 1080,
            "model_input_width": 770,
            "model_input_height": 434,
            "input_variant": variant,
            "input_variant_sha256": variant_hash,
            "clip": "clip",
            "split": "training",
            "film_id": "film",
            "domain": "movie",
            "source_kind": "mono-video",
            "global_policy_weight": 1.0,
            "source_frame_rate": 24.0,
            "runtime_scene_trace_sha256": "9" * 64,
            "ordinal_bundle_sha256": "e" * 64,
        }
        target_source = {
            **common,
            "row_role": "target",
            "source_sha256": SOURCE,
            "frame": 0,
            "runtime_scene_evidence": scene,
            "ordinal_frame_model_input_provenance": provenance,
            "ordinal_frame_model_input_provenance_sha256": provenance_hash,
            "ordinal_model_depth_artifact_sha256": "d" * 64,
        }
        context_scene = dict(scene, source_frame_id=1)
        context_source = {
            **common,
            "row_role": "context",
            "source_sha256": "c" * 64,
            "frame": 1,
            "runtime_scene_evidence": context_scene,
        }
        frontier_row = {
            "row_role": "target",
            "_join_key": (SOURCE, "clip", 0, variant_hash),
            "_input_variant": variant,
            "_input_variant_sha256": variant_hash,
            "_runtime_scene_id": 0,
            "_runtime_scene_evidence": scene,
            "_ordinal_safety_evidence": {"status": "proven"},
            "_ordinal_intersection": intersection(4),
            "_model_input_provenance": provenance,
            "_model_input_provenance_sha256": provenance_hash,
            "_model_depth_artifact_sha256": "d" * 64,
            "_ordinal_bundle_sha256": "e" * 64,
        }
        joined = trainer.join_ordinal_sources([target_source], [frontier_row])
        self.assertEqual([item["row_role"] for item in joined], ["target"])
        self.assertEqual(joined[0]["_input_variant"], variant)
        self.assertEqual(joined[0]["_input_variant_sha256"], variant_hash)
        self.assertEqual(
            trainer.runtime_variant_name(joined[0]["_input_variant"]),
            "native_sdr",
        )
        unexpected = copy.deepcopy(context_source)
        unexpected["row_role"] = "target"
        unexpected["ordinal_frame_model_input_provenance"] = provenance
        unexpected["ordinal_frame_model_input_provenance_sha256"] = (
            provenance_hash
        )
        unexpected["ordinal_model_depth_artifact_sha256"] = "d" * 64
        with self.assertRaisesRegex(RuntimeError, "target subset differs"):
            trainer.join_ordinal_sources(
                [target_source, unexpected], [frontier_row]
            )

    def test_geometry_structure_matches_across_sdr_and_hdr_conditions(self):
        sdr_tuples = [geometry(1280, 720), geometry(1920, 1080)]
        hdr_tuples = []
        for value in sdr_tuples:
            changed = dict(value)
            changed["color_mode"] = geometry_contract.COLOR_MODE_HDR
            hdr_tuples.append(changed)
        sdr = geometry_contract.build_allowlist(sdr_tuples)
        hdr = geometry_contract.build_allowlist(hdr_tuples)
        self.assertNotEqual(
            geometry_contract.allowlist_sha256(sdr),
            geometry_contract.allowlist_sha256(hdr),
        )
        self.assertEqual(
            trainer.geometry_structure_sha256(sdr),
            trainer.geometry_structure_sha256(hdr),
        )

    def test_cached_smoke_epoch_updates_only_ordinal_head(self):
        class DummyDepth(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = "vits"
                self.intermediate_layer_idx = {"vits": [0]}
                self.pretrained = type("Pretrained", (), {"embed_dim": 4})()

        # Build without invoking DA-V2: the cached path only needs the head and
        # feature width, making this a fast CPU smoke test of optimizer wiring.
        model = trainer.OrdinalArtisticPolicyModel(DummyDepth())
        model.freeze_base()
        target = intersection(4)
        rows = [
            row(input_color.sdr_input_variant(), target),
            row(input_color.windows_hdr_input_variant(1000), target,
                clip="hdr"),
        ]
        features = torch.randn(2, model.policy_feature_size)
        dataset = trainer.CachedOrdinalDataset(features, rows)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, collate_fn=trainer.collate_ordinal_samples
        )
        before = {
            key: value.detach().clone()
            for key, value in model.ordinal_head.state_dict().items()
        }
        optimizer = torch.optim.AdamW(model.ordinal_head.parameters(), lr=1e-2)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        result = trainer.run_epoch(
            model, loader, torch.device("cpu"), optimizer, scaler
        )
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertTrue(any(
            not torch.equal(before[key], value)
            for key, value in model.ordinal_head.state_dict().items()
        ))
        self.assertEqual(
            {name for name, parameter in model.named_parameters()
             if parameter.requires_grad},
            {name for name, _parameter in model.named_parameters()
             if name.startswith("ordinal_head.")},
        )

    def test_feature_cache_batches_same_shape_and_restores_row_order(self):
        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                row_value = float(self.rows[index]["value"])
                height = self.rows[index]["model_input_height"]
                width = self.rows[index]["model_input_width"]
                return torch.full((3, height, width), row_value), index

        class FakeModel:
            def __init__(self):
                self.batch_sizes = []

            def eval(self):
                return self

            def policy_features(self, images):
                self.batch_sizes.append(images.shape[0])
                mean = images.float().mean(dim=(1, 2, 3))
                return torch.stack((mean, mean + 1.0), dim=1)

        target = intersection(4)
        rows = []
        for value, shape in ((3, (8, 6)), (1, (4, 4)), (2, (8, 6))):
            item = row(input_color.sdr_input_variant(), target, frame=value)
            item.update({
                "value": value,
                "model_input_width": shape[0],
                "model_input_height": shape[1],
            })
            rows.append(item)
        model = FakeModel()
        with mock.patch.object(trainer, "OrdinalImageDataset", FakeDataset):
            cached = trainer.cache_ordinal_dataset(
                model, rows, torch.device("cpu"), batch_size=2
            )
        self.assertEqual(model.batch_sizes, [1, 2])
        self.assertEqual(cached.features[:, 0].tolist(), [3.0, 1.0, 2.0])
        self.assertEqual(cached.features.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
