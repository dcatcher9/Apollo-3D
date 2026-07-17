import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import build_artistic_policy_ordinal_report as report_builder
import depth_input_color as input_color


VARIANTS = (
    "hdr_raw1000", "hdr_raw2500", "hdr_raw6000", "native_pq",
    "native_sdr",
)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group(*, samples=20, selected_over=0, applied_over=0, gain=8.0,
          groups=12, group_overshoots=0, unproven=0,
          identity_failures=0, plateau_excess=0):
    return {
        "macro": {
            "samples": samples,
            "proven_samples": samples - unproven,
            "unproven_samples": unproven,
            "shots": 5,
            "films": 3,
            "independent_source_groups": groups,
            "independent_source_groups_with_overshoot": group_overshoots,
            "selected_unproven_overshoot_count": selected_over,
            "unproven_overshoot_count": applied_over,
            "selected_realized_pop_gain_pct": gain + 1.0,
            "realized_pop_gain_pct": gain,
            "abstained": 0.1,
            "scale_underreach": 0.04,
            "interval_nll": 0.2,
            "known_bin_brier": 0.03,
            "known_bin_ece": 0.02,
            "identity_hard_failure_count": identity_failures,
            "plateau_excess_selection_count": plateau_excess,
            "plateau_excess_applied_count": plateau_excess,
            "selected_plateau_excess_scale": 0.02 * plateau_excess,
            "plateau_excess_scale": 0.02 * plateau_excess,
        },
        "films": {},
    }


def calibration_evidence(value):
    measured = {
        "sdr": value["regimes"]["sdr"],
        "hdr": value["regimes"]["hdr"],
        **{
            "variant:" + name: item
            for name, item in value["input_variants"].items()
            if name != "native_sdr"
        },
    }
    required = 59
    result = {}
    for name, item in measured.items():
        macro = item["macro"]
        groups = macro["independent_source_groups"]
        failures = macro["independent_source_groups_with_overshoot"]
        upper = 1.0 - 0.05 ** (1.0 / groups) if groups and not failures else 1.0
        result[name] = {
            "contract": "zero-failure-one-sided-binomial-bound-v1",
            "confidence": 0.95,
            "maximum_failure_rate": 0.05,
            "independent_development_groups": groups,
            "observed_groups_with_overshoot": failures,
            "zero-failure_upper_bound": upper,
            "minimum_independent_groups_required": required,
            "deployable": failures == 0 and groups >= required,
        }
    return result


def acceptance(*, sdr=None, hdr=None, hdr_variant=None):
    sdr = sdr or group()
    hdr = hdr or group(samples=80, gain=7.0)
    hdr_variant = hdr_variant or group(gain=6.0)
    value = {
        "overall": group(samples=100, gain=7.5),
        "regimes": {"sdr": sdr, "hdr": hdr},
        "input_variants": {
            "native_sdr": copy.deepcopy(sdr),
            "hdr_raw1000": hdr_variant,
            "hdr_raw2500": group(gain=7.0),
            "hdr_raw6000": group(gain=7.5),
            "native_pq": group(gain=6.5),
        },
        "application": {
            "mode": "direct-target-only", "context_rows": 0,
            "target_rows": 100,
            "validation_scope": "independent authenticated safety targets",
        },
        "promotion_prerequisites": {
            "independent_target_render_gate_passed": False,
            "sealed_test_target_gate_passed": False,
            "promotion_blocked_until_both_true": True,
        },
    }
    value["calibration_evidence"] = calibration_evidence(value)
    value["calibration_deployable"] = all(
        item["deployable"]
        for item in value["calibration_evidence"].values()
    )
    return value


def add_candidate_status(value, *, threshold=5.0, passed=None):
    measured = {
        "sdr": value["regimes"]["sdr"],
        "hdr": value["regimes"]["hdr"],
        **{
            "variant:" + name: item
            for name, item in value["input_variants"].items()
            if name != "native_sdr"
        },
    }
    gains = {
        name: item["macro"]["realized_pop_gain_pct"]
        for name, item in measured.items()
    }
    unproven = {
        name: item["macro"]["unproven_samples"]
        for name, item in measured.items()
    }
    identity = {
        name: item["macro"]["identity_hard_failure_count"]
        for name, item in measured.items()
    }
    overshoot_safe = all(
        item["macro"]["selected_unproven_overshoot_count"] == 0 and
        item["macro"]["unproven_overshoot_count"] == 0
        for item in measured.values()
    )
    complete = all(count == 0 for count in unproven.values())
    zero_identity = all(count == 0 for count in identity.values())
    observed_pass = min(gains.values()) >= threshold
    if passed is None:
        passed = observed_pass
    value["development_candidate_status"] = {
        "contract": "ordinal-development-candidate-status-v2",
        "minimum_realized_pop_gain_pct": threshold,
        "minimum_realized_pop_gain_rationale": (
            "Predeclared development usefulness floor for unit testing."
        ),
        "group_realized_pop_gain_pct": gains,
        "worst_group_realized_pop_gain_pct": min(gains.values()),
        "group_unproven_samples": unproven,
        "group_identity_hard_failures": identity,
        "complete_evidence_pass": complete,
        "zero_identity_failure_pass": zero_identity,
        "zero_overshoot_pass": overshoot_safe,
        "material_gain_pass": passed,
        "training_checkpoint_eligible": (
            complete and zero_identity and overshoot_safe and passed
        ),
        "production_policy_accepted": False,
    }
    return value


def epoch(number, *, accepted=None, eligible=None):
    accepted = accepted or acceptance()
    metrics = {
        "loss": 0.5 / number,
        "interval_nll": 0.2,
        "known_bin_brier": 0.03,
    }
    record = {
        "epoch": number,
        "checkpoint_selection_contract": (
            report_builder.CHECKPOINT_SELECTION_CONTRACT
        ),
        "training": dict(metrics),
        "development": {**metrics, "acceptance": accepted},
        "checkpoint_eligible": False,
        "checkpoint_selection_key": [],
    }
    if eligible is None:
        status = accepted.get("development_candidate_status", {})
        eligible = status.get("training_checkpoint_eligible") is True
    record["checkpoint_eligible"] = eligible
    record["checkpoint_selection_key"] = list(
        report_builder._recompute_epoch_key(record)
    )
    return record


def evidence_bucket(*, target=20, unproven=0):
    return {
        "target_rows": target,
        "proven_targets": target - unproven,
        "unproven_targets": unproven,
    }


def runtime_split(variants):
    variants = sorted(variants)
    buckets = {name: evidence_bucket() for name in variants}
    sdr_names = [name for name in variants if name == "native_sdr"]
    hdr_names = [name for name in variants if name != "native_sdr"]

    def aggregate(names):
        return {
            key: sum(buckets[name][key] for name in names)
            for key in (
                "target_rows", "proven_targets", "unproven_targets",
            )
        }

    input_variants = {
        "native_sdr": input_color.sdr_input_variant(),
        "hdr_raw1000": input_color.windows_hdr_input_variant(1000),
        "hdr_raw2500": input_color.windows_hdr_input_variant(2500),
        "hdr_raw6000": input_color.windows_hdr_input_variant(6000),
        "native_pq": input_color.native_pq_input_variant(),
    }
    return {
        "expected_variants": variants,
        "expected_conditions": [
            {
                "name": name,
                "input_variant": input_variants[name],
                "input_variant_sha256": input_color.input_variant_sha256(
                    input_variants[name]
                ),
            }
            for name in variants
        ],
        "regimes": {"sdr": aggregate(sdr_names), "hdr": aggregate(hdr_names)},
        "variants": buckets,
    }


def contract(*, threshold=None):
    value = {
        "schema": report_builder.CHECKPOINT_SCHEMA,
        "training_contract": report_builder.TRAINING_CONTRACT,
        "checkpoint_selection_contract": (
            report_builder.CHECKPOINT_SELECTION_CONTRACT
        ),
        "policy_contract": "test-policy",
        "output_semantics": "test-output",
        "policy_feature_contract": "test-features",
        "active_split_sha256": "a" * 64,
        "source_bundles_sha256": "b" * 64,
        "frontier_bundles_sha256": "c" * 64,
        "metric_specs_sha256": "d" * 64,
        "metric_contract_sha256": "3" * 16,
        "thresholds_sha256": "4" * 64,
        "expected_runtime_evidence": {
            "training": runtime_split(VARIANTS[:-2] + ("native_sdr",)),
            "development": runtime_split(VARIANTS),
        },
        "orchestration_catalog": {
            "sha256": "5" * 64,
            "metric_sha256": "3" * 16,
            "thresholds_sha256": "4" * 64,
            "sbsbench_sha256": "6" * 64,
            "run_eval_sha256": "7" * 64,
            "conf_sha256": "8" * 16,
            "executable_sha256": "9" * 64,
        },
        "paired_variant_identifiability": {
            split: {
                "contract": "paired-image-evidence-identifiability-v1",
                "split": split,
                "paired_source_frames": 10,
                "variant_pair_comparisons": 60,
                "near_identical_feature_pairs": 2,
                "near_identical_pairs_with_exact_depth": 1,
                "contradictory_near_identical_pairs": 1,
                "contradiction_examples": [],
                "variant_specific_safety_targets_retained": True,
                "label_admission_blocked_by_feature_similarity": False,
                "runtime_condition_metadata_model_input": False,
            }
            for split in ("training", "development")
        },
        "plateau_policy": {
            "oracle_pop_tolerance_pct": 1e-6,
            "oracle_used_to_modify_direct_selection": False,
        },
        "deployment_geometry_allowlist_sha256": "e" * 64,
        "deployment_geometry_structure_sha256": "f" * 64,
        "depth_weights_sha256": "1" * 64,
        "depth_input_color_contract_sha256": "2" * 64,
        "sealed_test_productions": ["held-out-one", "held-out-two"],
        "promotion_prerequisites": {
            "promotion_blocked_until_both_true": True,
        },
    }
    if threshold is not None:
        value.update({
            "minimum_development_realized_pop_gain_pct": threshold,
            "minimum_development_realized_pop_gain_rationale": (
                "Predeclared development usefulness floor for unit testing."
            ),
        })
    return value


class OrdinalReportTest(unittest.TestCase):
    def write_run(self, directory, records, *, threshold=None,
                  contract_value=None):
        directory = Path(directory)
        contract_value = contract_value or contract(threshold=threshold)
        contract_path = directory / "training_contract.json"
        contract_path.write_text(json.dumps(contract_value), encoding="utf-8")
        eligible = [item for item in records if item["checkpoint_eligible"]]
        receipt = None
        if eligible:
            best = min(eligible, key=lambda item: (
                tuple(item["checkpoint_selection_key"]), item["epoch"]
            ))
            checkpoint = directory / "artistic_policy_ordinal_best.pt"
            checkpoint.write_bytes(b"authenticated checkpoint fixture")
            receipt = {
                "path": checkpoint.name,
                "sha256": file_sha256(checkpoint),
                "epoch": best["epoch"],
            }
        history = {
            "schema": report_builder.HISTORY_SCHEMA,
            "contract": report_builder.HISTORY_CONTRACT,
            "training_contract_sha256": file_sha256(contract_path),
            "completed": True,
            "epoch_count": len(records),
            "epochs": records,
            "best_checkpoint": receipt,
        }
        (directory / "history.json").write_text(
            json.dumps(history), encoding="utf-8"
        )
        return history

    def test_builds_authenticated_development_only_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.write_run(temporary, [epoch(1), epoch(2)])
            result = report_builder.build_report(temporary)
            rendered = report_builder.render_html(result)
            self.assertEqual(result["schema"], report_builder.REPORT_SCHEMA)
            self.assertEqual(result["best_epoch"]["epoch"], 1)
            self.assertTrue(result["conclusion"]["development_safety_pass"])
            self.assertFalse(result["conclusion"]["deployable"])
            self.assertFalse(result["provenance"]["sealed_test_content_read"])
            self.assertTrue(
                result["acceptance"]["native_pq_development_holdout"]
            )
            self.assertEqual(rendered.count('<div class="card">'), 3)
            self.assertIn("development-only holdout", rendered)

    def test_authenticated_candidate_receipt_is_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            accepted = add_candidate_status(acceptance())
            self.write_run(
                temporary, [epoch(1, accepted=accepted)], threshold=5.0
            )
            result = report_builder.build_report(temporary)
            self.assertTrue(
                result["conclusion"]["development_candidate_accepted"]
            )
            checkpoint = Path(temporary) / "artistic_policy_ordinal_best.pt"
            checkpoint.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "receipt authentication"):
                report_builder.build_report(temporary)

    def test_missing_development_condition_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            accepted = acceptance()
            accepted["input_variants"].pop("native_pq")
            accepted["calibration_evidence"].pop("variant:native_pq")
            record = epoch(1, accepted=accepted)
            self.write_run(temporary, [record])
            with self.assertRaisesRegex(RuntimeError, "condition set"):
                report_builder.build_report(temporary)

    def test_tampered_expected_condition_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.write_run(temporary, [epoch(1)])
            path = Path(temporary) / "training_contract.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["expected_runtime_evidence"]["development"][
                "expected_conditions"
            ][0]["input_variant_sha256"] = "0" * 64
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(
                    RuntimeError, "expected conditions differ"):
                report_builder.build_report(temporary)

    def test_training_contract_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.write_run(temporary, [epoch(1)])
            history_path = Path(temporary) / "history.json"
            history = json.loads(history_path.read_text(encoding="utf-8"))
            history["training_contract_sha256"] = "0" * 64
            history_path.write_text(json.dumps(history), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "authenticate"):
                report_builder.build_report(temporary)

    def test_stale_selection_key_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = epoch(1)
            record["checkpoint_selection_key"][4] -= 1.0
            self.write_run(temporary, [record])
            with self.assertRaisesRegex(RuntimeError, "selection key is stale"):
                report_builder.build_report(temporary)

    def test_stale_calibration_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            accepted = acceptance()
            accepted["calibration_evidence"]["sdr"]["deployable"] = True
            self.write_run(temporary, [epoch(1, accepted=accepted)])
            with self.assertRaisesRegex(RuntimeError, "calibration evidence"):
                report_builder.build_report(temporary)

    def test_application_or_target_count_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            accepted = acceptance()
            accepted["application"]["context_rows"] += 1
            self.write_run(temporary, [epoch(1, accepted=accepted)])
            with self.assertRaisesRegex(RuntimeError, "target-only application"):
                report_builder.build_report(temporary)

        with tempfile.TemporaryDirectory() as temporary:
            accepted = acceptance()
            accepted["input_variants"]["native_sdr"]["macro"]["samples"] += 1
            accepted["input_variants"]["native_sdr"]["macro"][
                "proven_samples"
            ] += 1
            self.write_run(temporary, [epoch(1, accepted=accepted)])
            with self.assertRaisesRegex(RuntimeError, "target evidence count"):
                report_builder.build_report(temporary)

    def test_exact_history_envelope_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.write_run(temporary, [epoch(1)])
            path = Path(temporary) / "history.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["unexpected"] = True
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "envelope keys"):
                report_builder.build_report(temporary)

    def test_sdr_and_hdr_fail_independently(self):
        with tempfile.TemporaryDirectory() as temporary:
            bad_sdr = group(selected_over=1, gain=50.0)
            accepted = acceptance(sdr=bad_sdr)
            accepted["calibration_evidence"] = calibration_evidence(accepted)
            self.write_run(temporary, [epoch(1, accepted=accepted)])
            result = report_builder.build_report(temporary)
            self.assertFalse(result["acceptance"]["sdr_pass"])
            self.assertTrue(result["acceptance"]["hdr_pass"])
            self.assertFalse(
                result["conclusion"]["overall_average_used_for_acceptance"]
            )


if __name__ == "__main__":
    unittest.main()
