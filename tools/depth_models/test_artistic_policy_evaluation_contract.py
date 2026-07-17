#!/usr/bin/env python3

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import artistic_policy_evaluation_contract as contract  # noqa: E402


CONDITION_TARGET_CONTRACT = "per-input-condition-two-geometry-safe-frontier-v1"


def unsafe_evidence():
    return {
        "maximum_scale": 0.0,
        "maximum_limit_scale": 0.05,
        "film_balanced_mean_scale": 0.0,
        "film_balanced_mean_limit_scale": 0.01,
        "film_balanced_overshoot_rate_pct": 0.0,
        "by_film_mean_scale": {"film": 0.0},
        "by_film_overshoot_rate_pct": {"film": 0.0},
        "maximum_pass": True,
        "film_balanced_mean_pass": True,
    }


def accepted_group(samples=2, unique=2, shots=2, shot_conditions=2,
                   actionable_samples=1, actionable_shots=1, films=2):
    evidence = unsafe_evidence()
    metrics = {key: 1.0 for key in contract.ALL_METRICS}
    neutral = {key: 2.0 for key in contract.ALL_METRICS}
    primary = {
        "variant_sample_count": samples,
        "unique_rgb_sample_count": unique,
        "shot_count": shots,
        "shot_condition_count": shot_conditions,
        "film_count": films,
        "actionable_variant_sample_count": actionable_samples,
        "identity_variant_sample_count": samples - actionable_samples,
        "actionable_shot_condition_count": actionable_shots,
        "identity_shot_condition_count": shot_conditions - actionable_shots,
        "predicted_scale_mean": 1.1,
        "target_scale_mean": 1.1,
        "predicted_confidence_mean": 0.5,
        "target_confidence_mean": 0.5,
        "action_brier": 1.0,
        "action_ece": 0.1,
        "identity_false_action_pct": 1.0,
        "rendered_disparity_mean_abs_pct": 1.0,
        "target_rendered_disparity_mean_abs_pct": 1.0,
        "rendered_disparity_mae_pct": 1.0,
        "maximum_unsafe_ceiling_overshoot_scale": 0.0,
        "film_balanced_mean_unsafe_ceiling_overshoot_scale": 0.0,
        "film_balanced_unsafe_ceiling_overshoot_rate_pct": 0.0,
    }
    ones = {key: 1 for key in contract.METRICS}
    film_counts = {key: films for key in contract.METRICS}
    required_films = {key: films // 2 + 1 for key in contract.METRICS}
    decision = {
        "accepted": True,
        "aggregate_wins": {key: True for key in contract.METRICS},
        "sequence_count": dict(ones),
        "sequence_wins": dict(ones),
        "required_sequence_wins": dict(ones),
        "domain_count": dict(ones),
        "domain_wins": dict(ones),
        "required_domain_wins": dict(ones),
        "film_count": film_counts,
        "film_wins": film_counts,
        "required_film_wins": required_films,
        "minimum_film_count": film_counts,
        "guards": {
            "identity_examples_present": True,
            "identity_false_action_pct": True,
            "unsafe_ceiling_maximum": True,
            "unsafe_ceiling_film_balanced_mean": True,
        },
        "identity_guard_required": True,
        "unsafe_overshoot_guard_required": True,
        "unsafe_ceiling_overshoot": evidence,
    }
    return {
        "evaluation": {"trained": metrics, "neutral": neutral},
        "unsafe_ceiling_overshoot": evidence,
        "decision": decision,
        "primary": primary,
    }


def accepted_payload():
    whites = {
        str(value): accepted_group()
        for value in contract.EXPECTED_HDR_WHITE_LEVELS_RAW
    }
    hdr = accepted_group(
        samples=6, unique=2, shots=2, shot_conditions=6,
        actionable_samples=3, actionable_shots=3,
    )
    runtime_summary = {
        "contract": contract.RUNTIME_REGIME_ACCEPTANCE_CONTRACT,
        "condition_target_contract": CONDITION_TARGET_CONTRACT,
        "hdr_aggregation_contract": contract.HDR_AGGREGATION_CONTRACT,
        "required_regimes": ["sdr", "hdr"],
        "expected_hdr_white_levels_raw": [1000, 2500, 6000],
        "missing_regimes": [],
        "missing_hdr_white_levels_raw": [],
        "unexpected_hdr_white_levels_raw": [],
        "incomplete_source_frame_count": 0,
        "source_condition_coverage_complete": True,
        "hdr_white_pass": {"1000": True, "2500": True, "6000": True},
        "hdr_aggregate_pass": True,
        "regime_pass": {"sdr": True, "hdr": True},
        "accepted": True,
    }
    runtime = {
        **runtime_summary,
        "incomplete_source_frames": [],
        "regimes": {"sdr": accepted_group(), "hdr": hdr},
        "hdr_by_white_level_raw": whites,
    }
    decision = {
        "overall_diagnostic_accepted": True,
        "runtime_regime_acceptance": runtime_summary,
        "guards": {
            "runtime_regimes_present": True,
            "hdr_white_levels_present": True,
            "hdr_white_levels_accepted": True,
            "no_unexpected_hdr_white_levels": True,
            "source_condition_coverage_complete": True,
            "condition_target_contract": True,
            "sdr_and_hdr_accepted": True,
        },
    }
    return {
        "runtime_regime_evaluation": runtime,
        "val_films": ["film-a", "film-b"],
    }, decision


class ArtisticPolicyEvaluationContractTests(unittest.TestCase):
    def test_accepts_complete_real_group_evidence(self):
        payload, decision = accepted_payload()
        summary = contract.validate_runtime_regime_acceptance(
            payload, decision, CONDITION_TARGET_CONTRACT
        )
        self.assertTrue(summary["accepted"])

    def test_rejects_empty_group_placeholders(self):
        payload, decision = accepted_payload()
        payload["runtime_regime_evaluation"]["regimes"]["sdr"] = {}
        with self.assertRaisesRegex(RuntimeError, "sealed SDR"):
            contract.validate_runtime_regime_acceptance(
                payload, decision, CONDITION_TARGET_CONTRACT
            )

    def test_rejects_white_failure_hidden_by_summary(self):
        payload, decision = accepted_payload()
        white = payload["runtime_regime_evaluation"][
            "hdr_by_white_level_raw"
        ]["2500"]
        white["decision"]["accepted"] = False
        with self.assertRaisesRegex(RuntimeError, "white 2500"):
            contract.validate_runtime_regime_acceptance(
                payload, decision, CONDITION_TARGET_CONTRACT
            )

    def test_rejects_hdr_count_not_equal_to_white_sum(self):
        payload, decision = accepted_payload()
        payload["runtime_regime_evaluation"]["regimes"]["hdr"]["primary"][
            "variant_sample_count"
        ] = 5
        with self.assertRaisesRegex(RuntimeError, "sample class counts"):
            contract.validate_runtime_regime_acceptance(
                payload, decision, CONDITION_TARGET_CONTRACT
            )

    def test_rejects_stale_condition_target_contract(self):
        payload, decision = accepted_payload()
        with self.assertRaisesRegex(RuntimeError, "per-condition acceptance"):
            contract.validate_runtime_regime_acceptance(
                payload, decision, "shared-target-v0"
            )


if __name__ == "__main__":
    unittest.main()
