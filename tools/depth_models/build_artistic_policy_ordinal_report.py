#!/usr/bin/env python3
"""Build a static development report for an ordinal artistic-policy run.

The report is deliberately separate from training.  It authenticates the
frozen ``training_contract.json``, completed ``history.json`` receipt, and the
published checkpoint bytes (without deserializing them).  It never opens
source media, labels, or the sealed test set.  Runtime-regime acceptance is
fail-closed: native SDR and every frozen HDR condition must pass independently.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path

import depth_input_color as input_color


REPORT_SCHEMA = "offline-ordinal-artistic-policy-report-v3"
TRAINING_CONTRACT = "offline-ordinal-artistic-policy-training-v5"
CHECKPOINT_SCHEMA = 3
CHECKPOINT_SELECTION_CONTRACT = "ordinal-safety-first-selection-key-v1"
HISTORY_SCHEMA = 1
HISTORY_CONTRACT = "offline-ordinal-artistic-policy-history-v1"
REQUIRED_REGIMES = ("sdr", "hdr")
CALIBRATION_ALPHA = 0.05
CALIBRATION_MAX_FAILURE_RATE = 0.05
HARD_ZERO_FIELDS = (
    "selected_unproven_overshoot_count",
    "unproven_overshoot_count",
    "identity_hard_failure_count",
)
PRIMARY_FINITE_FIELDS = (
    "selected_realized_pop_gain_pct",
    "realized_pop_gain_pct",
    "abstained",
    "scale_underreach",
    "interval_nll",
    "known_bin_brier",
    "known_bin_ece",
    "selected_plateau_excess_scale",
    "plateau_excess_scale",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path, expected_type):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read authenticated report input {path}") from error
    if not isinstance(value, expected_type):
        raise RuntimeError(f"invalid report input type: {path}")
    return value


def _finite_number(value):
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool) and
        math.isfinite(float(value))
    )


def _sha256_value(value):
    return (
        isinstance(value, str) and len(value) == 64 and
        all(character in "0123456789abcdef" for character in value)
    )


def _contract_sha256_value(value):
    return (
        isinstance(value, str) and len(value) == 16 and
        all(character in "0123456789abcdef" for character in value)
    )


def _runtime_variant_name(value):
    try:
        input_color.validate_input_variant(value)
    except (RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(
            "ordinal expected condition input variant differs"
        ) from error
    kind = value["kind"]
    if kind == input_color.INPUT_KIND_SDR:
        return "native_sdr"
    if kind == input_color.INPUT_KIND_WINDOWS_HDR:
        return "hdr_raw" + str(value["windows_sdr_white_level_raw"])
    if kind == input_color.INPUT_KIND_NATIVE_PQ:
        return "native_pq"
    raise RuntimeError("ordinal expected condition input variant differs")


def _validate_count(value, field):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"ordinal report has invalid {field}")


def _validate_identifiability(value, split):
    if (not isinstance(value, dict) or
            value.get("contract") !=
            "paired-image-evidence-identifiability-v1" or
            value.get("split") != split or
            value.get("variant_specific_safety_targets_retained") is not True or
            value.get("label_admission_blocked_by_feature_similarity") is not False or
            value.get("runtime_condition_metadata_model_input") is not False):
        raise RuntimeError(
            f"ordinal paired-variant identifiability is invalid: {split}"
        )
    count_fields = (
        "paired_source_frames", "variant_pair_comparisons",
        "near_identical_feature_pairs",
        "near_identical_pairs_with_exact_depth",
        "contradictory_near_identical_pairs",
    )
    for field in count_fields:
        _validate_count(value.get(field), f"{split}.{field}")
    comparisons = value["variant_pair_comparisons"]
    near = value["near_identical_feature_pairs"]
    if (near > comparisons or
            value["near_identical_pairs_with_exact_depth"] > near or
            value["contradictory_near_identical_pairs"] > near or
            not isinstance(value.get("contradiction_examples"), list)):
        raise RuntimeError(
            f"ordinal paired-variant identifiability counts disagree: {split}"
        )
    return value


def _validate_macro(macro, name):
    if not isinstance(macro, dict):
        raise RuntimeError(f"ordinal report group {name} lacks macro metrics")
    for field in (
            "samples", "shots", "films", "independent_source_groups",
            "independent_source_groups_with_overshoot",
            "proven_samples", "unproven_samples", *HARD_ZERO_FIELDS,
            "plateau_excess_selection_count",
            "plateau_excess_applied_count"):
        if field not in macro:
            raise RuntimeError(f"ordinal report group {name} lacks {field}")
        _validate_count(macro[field], f"{name}.{field}")
    if macro["samples"] <= 0 or macro["shots"] <= 0 or macro["films"] <= 0:
        raise RuntimeError(f"ordinal report group {name} is empty")
    if macro["proven_samples"] + macro["unproven_samples"] != macro["samples"]:
        raise RuntimeError(f"ordinal report group {name} sample counts disagree")
    if (macro["independent_source_groups_with_overshoot"] >
            macro["independent_source_groups"]):
        raise RuntimeError(
            f"ordinal report group {name} calibration counts disagree"
        )
    for field in PRIMARY_FINITE_FIELDS:
        if not _finite_number(macro.get(field)):
            raise RuntimeError(f"ordinal report group {name} lacks finite {field}")
    return macro


def _group_summary(group, name):
    if not isinstance(group, dict):
        raise RuntimeError(f"ordinal report lacks group {name}")
    macro = _validate_macro(group.get("macro"), name)
    failures = []
    if macro["proven_samples"] <= 0:
        failures.append("no proven frame-safety evidence")
    if macro["unproven_samples"] != 0:
        failures.append(f"unproven_samples={macro['unproven_samples']}")
    if macro["independent_source_groups"] <= 0:
        failures.append("no independent source groups")
    for field in HARD_ZERO_FIELDS:
        if macro[field] != 0:
            failures.append(f"{field}={macro[field]}")
    return {
        "name": name,
        "pass": not failures,
        "failures": failures,
        "samples": macro["samples"],
        "proven_samples": macro["proven_samples"],
        "unproven_samples": macro["unproven_samples"],
        "shots": macro["shots"],
        "films": macro["films"],
        "independent_groups": macro["independent_source_groups"],
        "independent_groups_with_overshoot": macro[
            "independent_source_groups_with_overshoot"
        ],
        "selected_pop_gain_pct": float(
            macro["selected_realized_pop_gain_pct"]
        ),
        "applied_pop_gain_pct": float(macro["realized_pop_gain_pct"]),
        "selected_overshoot_count": macro[
            "selected_unproven_overshoot_count"
        ],
        "applied_overshoot_count": macro["unproven_overshoot_count"],
        "identity_hard_failure_count": macro["identity_hard_failure_count"],
        "selected_plateau_excess_count": macro[
            "plateau_excess_selection_count"
        ],
        "applied_plateau_excess_count": macro[
            "plateau_excess_applied_count"
        ],
        "selected_plateau_excess_scale": float(
            macro["selected_plateau_excess_scale"]
        ),
        "applied_plateau_excess_scale": float(macro["plateau_excess_scale"]),
        "abstention_rate": float(macro["abstained"]),
        "scale_underreach": float(macro["scale_underreach"]),
        "interval_nll": float(macro["interval_nll"]),
        "known_bin_brier": float(macro["known_bin_brier"]),
        "known_bin_ece": float(macro["known_bin_ece"]),
    }


def _epoch_key(record):
    key = record.get("checkpoint_selection_key")
    if not isinstance(key, list) or len(key) != 12 or not all(
            _finite_number(value) for value in key):
        raise RuntimeError("ordinal history has invalid checkpoint selection key")
    return tuple(float(value) for value in key)


def _selection_metrics(acceptance):
    regimes = acceptance.get("regimes")
    variants = acceptance.get("input_variants")
    overall = acceptance.get("overall")
    if (not isinstance(regimes, dict) or
            set(regimes) != set(REQUIRED_REGIMES) or
            not isinstance(variants, dict) or
            "native_sdr" not in variants or
            not isinstance(overall, dict)):
        raise RuntimeError("ordinal epoch lacks complete selection metrics")
    metrics = [
        _validate_macro(regimes[name].get("macro"), name)
        for name in REQUIRED_REGIMES
    ]
    metrics.extend(
        _validate_macro(variants[name].get("macro"), "variant:" + name)
        for name in sorted(variants) if name != "native_sdr"
    )
    overall_macro = _validate_macro(overall.get("macro"), "overall")
    return metrics, overall_macro


def _recompute_epoch_key(record):
    metrics, overall = _selection_metrics(
        record["development"]["acceptance"]
    )
    identity_failures = [
        item["identity_hard_failure_count"] for item in metrics
    ]
    overshoots = [
        count for item in metrics for count in (
            item["selected_unproven_overshoot_count"],
            item["unproven_overshoot_count"],
        )
    ]
    return (
        float(max(identity_failures)),
        float(sum(identity_failures)),
        float(max(overshoots)),
        float(sum(overshoots)),
        -float(min(item["realized_pop_gain_pct"] for item in metrics)),
        -float(overall["realized_pop_gain_pct"]),
        float(max(item["selected_plateau_excess_scale"]
                  for item in metrics)),
        float(max(item["plateau_excess_scale"] for item in metrics)),
        float(overall["interval_nll"]),
        float(overall["known_bin_brier"]),
        float(overall["known_bin_ece"]),
        float(overall["scale_underreach"]),
    )


def _validate_epoch_acceptance_evidence(acceptance, expected, contract):
    if (not isinstance(acceptance, dict) or
            set(acceptance.get("regimes", {})) != set(REQUIRED_REGIMES) or
            set(acceptance.get("input_variants", {})) !=
            set(expected["expected_variants"])):
        raise RuntimeError("ordinal epoch acceptance condition set differs")
    regime_groups = {
        name: _group_summary(acceptance["regimes"][name], name)
        for name in REQUIRED_REGIMES
    }
    variants = {
        name: _group_summary(
            acceptance["input_variants"][name], "variant:" + name
        ) for name in expected["expected_variants"]
    }
    for name in REQUIRED_REGIMES:
        _require_target_counts(
            regime_groups[name], expected["regimes"][name], name
        )
    for name in expected["expected_variants"]:
        _require_target_counts(
            variants[name], expected["variants"][name], "variant:" + name,
        )
    overall = _group_summary(acceptance.get("overall"), "overall")
    overall_expected = {
        key: sum(item[key] for item in expected["variants"].values())
        for key in (
            "target_rows", "proven_targets", "unproven_targets",
        )
    }
    _require_target_counts(overall, overall_expected, "overall")
    target_rows = sum(
        item["target_rows"] for item in expected["variants"].values()
    )
    application = acceptance.get("application")
    if (not isinstance(application, dict) or
            application.get("mode") != "direct-target-only" or
            application.get("context_rows") != 0 or
            application.get("target_rows") != target_rows):
        raise RuntimeError(
            "ordinal target-only application evidence counts disagree"
        )
    measured = [
        {**regime_groups["sdr"], "name": "sdr"},
        {**regime_groups["hdr"], "name": "hdr"},
    ] + [
        {**variants[name], "name": "variant:" + name}
        for name in expected["expected_variants"] if name != "native_sdr"
    ]
    _recompute_calibration(acceptance, measured)
    usefulness = _material_gain_evidence(acceptance, contract, measured)
    complete = all(group["unproven_samples"] == 0 for group in measured)
    identity_safe = all(
        group["identity_hard_failure_count"] == 0 for group in measured
    )
    overshoot_safe = all(
        group["selected_overshoot_count"] == 0 and
        group["applied_overshoot_count"] == 0 for group in measured
    )
    expected_eligible = bool(
        complete and identity_safe and overshoot_safe and usefulness["pass"]
    )
    if usefulness["authenticated_threshold_present"]:
        reported = (
            usefulness["reported_complete_evidence_pass"],
            usefulness["reported_zero_identity_failure_pass"],
            usefulness["reported_zero_overshoot_pass"],
            usefulness["reported_training_checkpoint_eligible"],
        )
        if reported != (
                complete, identity_safe, overshoot_safe, expected_eligible):
            raise RuntimeError(
                "ordinal epoch candidate gate status is stale"
            )
    return expected_eligible


def _validate_epoch_records(records, development_expected, contract):
    if not records:
        raise RuntimeError("ordinal history is empty")
    exact_keys = {
        "epoch", "checkpoint_selection_contract", "training", "development",
        "checkpoint_eligible", "checkpoint_selection_key",
    }
    for expected_epoch, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise RuntimeError("ordinal history contains a non-object epoch")
        if set(record) != exact_keys:
            raise RuntimeError("ordinal history epoch keys differ from contract")
        epoch = record.get("epoch")
        if epoch != expected_epoch:
            raise RuntimeError("ordinal history epochs are not exact and consecutive")
        if (record.get("checkpoint_selection_contract") !=
                CHECKPOINT_SELECTION_CONTRACT or
                not isinstance(record.get("checkpoint_eligible"), bool)):
            raise RuntimeError("ordinal history selection contract differs")
        for split in ("training", "development"):
            metrics = record.get(split)
            if not isinstance(metrics, dict):
                raise RuntimeError(f"ordinal epoch {epoch} lacks {split}")
            for field in ("loss", "interval_nll", "known_bin_brier"):
                if not _finite_number(metrics.get(field)):
                    raise RuntimeError(
                        f"ordinal epoch {epoch} lacks finite {split}.{field}"
                    )
        acceptance = record["development"].get("acceptance")
        if not isinstance(acceptance, dict):
            raise RuntimeError(f"ordinal epoch {epoch} lacks acceptance metrics")
        regimes = acceptance.get("regimes")
        if not isinstance(regimes, dict) or set(regimes) != set(REQUIRED_REGIMES):
            raise RuntimeError(f"ordinal epoch {epoch} lacks SDR/HDR regimes")
        expected_eligible = _validate_epoch_acceptance_evidence(
            acceptance, development_expected, contract
        )
        if record["checkpoint_eligible"] is not expected_eligible:
            raise RuntimeError(
                f"ordinal epoch {epoch} checkpoint eligibility is stale"
            )
        serialized = _epoch_key(record)
        recomputed = _recompute_epoch_key(record)
        if serialized != recomputed:
            raise RuntimeError(
                f"ordinal epoch {epoch} checkpoint selection key is stale"
            )


def _load_and_validate_history(history, contract_path, training_output,
                               development_expected, contract):
    exact_keys = {
        "schema", "contract", "training_contract_sha256", "completed",
        "epoch_count", "epochs", "best_checkpoint",
    }
    if not isinstance(history, dict) or set(history) != exact_keys:
        raise RuntimeError("ordinal history envelope keys differ from contract")
    if (history.get("schema") != HISTORY_SCHEMA or
            history.get("contract") != HISTORY_CONTRACT or
            history.get("completed") is not True or
            history.get("training_contract_sha256") != sha256(contract_path)):
        raise RuntimeError("ordinal history does not authenticate completed training")
    records = history.get("epochs")
    if (not isinstance(records, list) or
            not isinstance(history.get("epoch_count"), int) or
            isinstance(history.get("epoch_count"), bool) or
            history["epoch_count"] != len(records)):
        raise RuntimeError("ordinal history epoch count disagrees")
    _validate_epoch_records(records, development_expected, contract)
    best, has_eligible = _best_record(records)
    receipt = history.get("best_checkpoint")
    if not has_eligible:
        if receipt is not None:
            raise RuntimeError("ordinal history publishes an ineligible checkpoint")
        return records, best, False, None
    if (not isinstance(receipt, dict) or
            set(receipt) != {"path", "sha256", "epoch"} or
            receipt.get("epoch") != best["epoch"] or
            not _sha256_value(receipt.get("sha256")) or
            receipt.get("path") != "artistic_policy_ordinal_best.pt"):
        raise RuntimeError("ordinal best checkpoint receipt is inconsistent")
    checkpoint_path = (training_output / receipt["path"]).resolve()
    if (checkpoint_path.parent != training_output or
            not checkpoint_path.is_file() or
            sha256(checkpoint_path) != receipt["sha256"]):
        raise RuntimeError("ordinal best checkpoint receipt authentication failed")
    return records, best, True, receipt


def _best_record(history):
    eligible = [
        record for record in history
        if record.get("checkpoint_eligible") is True
    ]
    pool = eligible or history
    return min(pool, key=lambda record: (_epoch_key(record), record["epoch"])), bool(
        eligible
    )


def _curve_record(record):
    acceptance = record["development"]["acceptance"]
    result = {
        "epoch": record["epoch"],
        "checkpoint_eligible": record.get("checkpoint_eligible") is True,
        "training": {
            field: float(record["training"][field])
            for field in ("loss", "interval_nll", "known_bin_brier")
        },
        "development": {
            field: float(record["development"][field])
            for field in ("loss", "interval_nll", "known_bin_brier")
        },
        "regimes": {},
    }
    for regime in REQUIRED_REGIMES:
        summary = _group_summary(acceptance["regimes"][regime], regime)
        result["regimes"][regime] = {
            field: summary[field]
            for field in (
                "selected_pop_gain_pct", "applied_pop_gain_pct",
                "abstention_rate", "scale_underreach",
                "selected_overshoot_count", "applied_overshoot_count",
            )
        }
    return result


def _validate_evidence_bucket(value, field):
    keys = {"target_rows", "proven_targets", "unproven_targets"}
    if not isinstance(value, dict) or set(value) != keys:
        raise RuntimeError(f"ordinal runtime evidence bucket differs: {field}")
    for key in keys:
        _validate_count(value[key], f"{field}.{key}")
    if value["proven_targets"] + value["unproven_targets"] != value["target_rows"]:
        raise RuntimeError(f"ordinal runtime target counts disagree: {field}")
    return value


def _validate_expected_runtime_evidence(contract):
    evidence = contract.get("expected_runtime_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
            "training", "development"}:
        raise RuntimeError("ordinal expected runtime evidence is incomplete")
    validated = {}
    for split in ("training", "development"):
        value = evidence[split]
        if (not isinstance(value, dict) or set(value) != {
                "expected_variants", "expected_conditions", "regimes",
                "variants"}):
            raise RuntimeError(
                f"ordinal expected runtime evidence differs: {split}"
            )
        variants = value["expected_variants"]
        if (not isinstance(variants, list) or not variants or
                variants != sorted(set(variants)) or
                not all(isinstance(item, str) and item for item in variants)):
            raise RuntimeError(f"ordinal expected variants differ: {split}")
        conditions = value["expected_conditions"]
        if (not isinstance(conditions, list) or
                [item.get("name") for item in conditions
                 if isinstance(item, dict)] != variants or any(
                    set(item) != {
                        "name", "input_variant", "input_variant_sha256"
                    } or not _sha256_value(item["input_variant_sha256"])
                    for item in conditions
                )):
            raise RuntimeError(f"ordinal expected conditions differ: {split}")
        for item in conditions:
            variant = item["input_variant"]
            if (_runtime_variant_name(variant) != item["name"] or
                    input_color.input_variant_sha256(variant) !=
                    item["input_variant_sha256"]):
                raise RuntimeError(
                    f"ordinal expected conditions differ: {split}"
                )
        regimes = value["regimes"]
        variant_buckets = value["variants"]
        if (not isinstance(regimes, dict) or
                set(regimes) != set(REQUIRED_REGIMES) or
                not isinstance(variant_buckets, dict) or
                set(variant_buckets) != set(variants)):
            raise RuntimeError(f"ordinal runtime evidence groups differ: {split}")
        validated[split] = {
            "expected_variants": variants,
            "expected_conditions": conditions,
            "regimes": {
                name: _validate_evidence_bucket(
                    regimes[name], f"{split}.regime.{name}"
                ) for name in REQUIRED_REGIMES
            },
            "variants": {
                name: _validate_evidence_bucket(
                    variant_buckets[name], f"{split}.variant.{name}"
                ) for name in variants
            },
        }
        sums = {
            key: sum(validated[split]["variants"][name][key]
                     for name in variants)
            for key in (
                "target_rows", "proven_targets", "unproven_targets",
            )
        }
        regime_sums = {
            key: sum(validated[split]["regimes"][name][key]
                     for name in REQUIRED_REGIMES)
            for key in sums
        }
        if sums != regime_sums:
            raise RuntimeError(f"ordinal runtime regime totals disagree: {split}")
    development_variants = validated["development"]["expected_variants"]
    if "native_sdr" not in development_variants:
        raise RuntimeError("ordinal development evidence lacks native SDR")
    return validated


def _require_target_counts(summary, expected, name):
    if (summary["samples"] != expected["target_rows"] or
            summary["proven_samples"] != expected["proven_targets"] or
            summary["unproven_samples"] != expected["unproven_targets"]):
        raise RuntimeError(
            f"ordinal target evidence count disagrees for {name}"
        )


def _recompute_calibration(acceptance, measured_groups):
    raw = acceptance.get("calibration_evidence")
    expected_names = {group["name"] for group in measured_groups}
    if not isinstance(raw, dict) or set(raw) != expected_names:
        raise RuntimeError("ordinal calibration group set differs")
    required = math.ceil(
        math.log(CALIBRATION_ALPHA) /
        math.log(1.0 - CALIBRATION_MAX_FAILURE_RATE)
    )
    recomputed = {}
    exact_keys = {
        "contract", "confidence", "maximum_failure_rate",
        "independent_development_groups",
        "observed_groups_with_overshoot", "zero-failure_upper_bound",
        "minimum_independent_groups_required", "deployable",
    }
    for group in measured_groups:
        name = group["name"]
        value = raw[name]
        independent = group["independent_groups"]
        failures = group["independent_groups_with_overshoot"]
        upper = (
            1.0 - CALIBRATION_ALPHA ** (1.0 / independent)
            if independent and failures == 0 else 1.0
        )
        deployable = failures == 0 and independent >= required
        if (not isinstance(value, dict) or set(value) != exact_keys or
                value.get("contract") !=
                "zero-failure-one-sided-binomial-bound-v1" or
                value.get("confidence") != 1.0 - CALIBRATION_ALPHA or
                value.get("maximum_failure_rate") !=
                CALIBRATION_MAX_FAILURE_RATE or
                value.get("independent_development_groups") != independent or
                value.get("observed_groups_with_overshoot") != failures or
                value.get("minimum_independent_groups_required") != required or
                value.get("deployable") is not deployable or
                not _finite_number(value.get("zero-failure_upper_bound")) or
                not math.isclose(
                    float(value["zero-failure_upper_bound"]), upper,
                    rel_tol=1e-12, abs_tol=1e-12,
                )):
            raise RuntimeError(
                f"ordinal calibration evidence is stale for {name}"
            )
        recomputed[name] = dict(value)
    deployable = all(value["deployable"] for value in recomputed.values())
    if acceptance.get("calibration_deployable") is not deployable:
        raise RuntimeError("ordinal calibration deployable status is stale")
    return recomputed, deployable


def _material_gain_evidence(acceptance, contract, measured_groups):
    """Return authenticated usefulness evidence, or an explicit pending state.

    Zero-overshoot safety alone can be satisfied by an all-abstain identity
    policy, so this report must not upgrade that safety result into candidate
    acceptance.  Current runs publish the canonical candidate-status object;
    if it is absent in a historical/partial run, usefulness remains pending
    regardless of the observed mean gain.
    """
    raw = acceptance.get("development_candidate_status")
    applied = [group["applied_pop_gain_pct"] for group in measured_groups]
    selected = [group["selected_pop_gain_pct"] for group in measured_groups]
    result = {
        "status": "pending",
        "pass": False,
        "authenticated_threshold_present": False,
        "minimum_required_direct_gain_pct": None,
        "worst_direct_gain_pct": min(applied),
        "worst_selected_gain_pct": min(selected),
        "reason": (
            "trainer emitted no authenticated material-gain threshold/status; "
            "safety eligibility alone cannot accept an all-abstain policy"
        ),
    }
    if raw is None:
        return result
    if not isinstance(raw, dict):
        raise RuntimeError("ordinal development candidate status is invalid")
    if raw.get("contract") != "ordinal-development-candidate-status-v2":
        raise RuntimeError("unknown ordinal development candidate contract")
    required = raw.get("minimum_realized_pop_gain_pct")
    rationale = raw.get("minimum_realized_pop_gain_rationale")
    passed = raw.get("material_gain_pass")
    groups = raw.get("group_realized_pop_gain_pct")
    if (not _finite_number(required) or float(required) <= 0.0 or
            not isinstance(rationale, str) or not rationale.strip() or
            not isinstance(passed, bool) or not isinstance(groups, dict) or
            not isinstance(raw.get("group_unproven_samples"), dict) or
            not isinstance(raw.get("group_identity_hard_failures"), dict) or
            not isinstance(raw.get("complete_evidence_pass"), bool) or
            not isinstance(raw.get("zero_identity_failure_pass"), bool) or
            not isinstance(raw.get("zero_overshoot_pass"), bool) or
            not isinstance(raw.get("training_checkpoint_eligible"), bool) or
            raw.get("production_policy_accepted") is not False):
        raise RuntimeError(
            "ordinal development candidate status is incomplete"
        )
    expected_groups = {
        group["name"]: group["applied_pop_gain_pct"]
        for group in measured_groups
    }
    if set(groups) != set(expected_groups):
        raise RuntimeError(
            "ordinal development candidate gain groups are incomplete"
        )
    for name, expected in expected_groups.items():
        if (not _finite_number(groups[name]) or
                not math.isclose(float(groups[name]), expected,
                                 rel_tol=1e-9, abs_tol=1e-9)):
            raise RuntimeError(
                "ordinal development candidate group gain disagrees with metrics"
            )
    unproven = raw["group_unproven_samples"]
    expected_unproven = {
        group["name"]: group["unproven_samples"]
        for group in measured_groups
    }
    if unproven != expected_unproven:
        raise RuntimeError(
            "ordinal candidate unproven counts disagree with group metrics"
        )
    identity_failures = raw["group_identity_hard_failures"]
    expected_identity_failures = {
        group["name"]: group["identity_hard_failure_count"]
        for group in measured_groups
    }
    if identity_failures != expected_identity_failures:
        raise RuntimeError(
            "ordinal candidate identity-hard-failure counts disagree with metrics"
        )
    zero_identity_failure_pass = all(
        count == 0 for count in expected_identity_failures.values()
    )
    if raw["zero_identity_failure_pass"] != zero_identity_failure_pass:
        raise RuntimeError(
            "ordinal candidate identity-hard-failure status disagrees with counts"
        )
    complete_evidence_pass = all(
        count == 0 for count in expected_unproven.values()
    )
    if raw["complete_evidence_pass"] != complete_evidence_pass:
        raise RuntimeError(
            "ordinal candidate evidence-completeness status disagrees with counts"
        )
    worst = raw.get("worst_group_realized_pop_gain_pct")
    if (not _finite_number(worst) or
            not math.isclose(float(worst), min(expected_groups.values()),
                             rel_tol=1e-9, abs_tol=1e-9)):
        raise RuntimeError(
            "ordinal development candidate worst gain disagrees with metrics"
        )
    contract_threshold = contract.get(
        "minimum_development_realized_pop_gain_pct"
    )
    contract_rationale = contract.get(
        "minimum_development_realized_pop_gain_rationale"
    )
    if (contract_threshold != required or contract_rationale != rationale):
        raise RuntimeError(
            "ordinal development candidate threshold is not frozen in contract"
        )
    observed_pass = float(worst) >= float(required)
    if passed != observed_pass:
        raise RuntimeError(
            "ordinal material-gain status disagrees with reported regime metrics"
        )
    result.update({
        "status": "pass" if passed else "fail",
        "pass": passed,
        "authenticated_threshold_present": True,
        "contract": raw["contract"],
        "rationale": rationale,
        "minimum_required_direct_gain_pct": float(required),
        "reason": (
            "worst runtime-regime/variant direct target gain meets the "
            "authenticated threshold" if passed else
            "worst runtime-regime/variant direct target gain is below the "
            "authenticated threshold"
        ),
    })
    result["reported_zero_overshoot_pass"] = raw["zero_overshoot_pass"]
    result["reported_zero_identity_failure_pass"] = raw[
        "zero_identity_failure_pass"
    ]
    result["reported_complete_evidence_pass"] = raw[
        "complete_evidence_pass"
    ]
    result["reported_training_checkpoint_eligible"] = raw[
        "training_checkpoint_eligible"
    ]
    return result


def build_report(training_output):
    training_output = Path(training_output).resolve()
    contract_path = training_output / "training_contract.json"
    history_path = training_output / "history.json"
    contract = _load_json(contract_path, dict)
    history_envelope = _load_json(history_path, dict)
    if (contract.get("schema") != CHECKPOINT_SCHEMA or
            contract.get("training_contract") != TRAINING_CONTRACT or
            contract.get("checkpoint_selection_contract") !=
            CHECKPOINT_SELECTION_CONTRACT):
        raise RuntimeError("ordinal training/checkpoint contract differs")
    expected_evidence = _validate_expected_runtime_evidence(contract)
    history, best, has_eligible_checkpoint, checkpoint_receipt = (
        _load_and_validate_history(
            history_envelope, contract_path, training_output,
            expected_evidence["development"],
            contract,
        )
    )
    catalog = contract.get("orchestration_catalog")
    catalog_hash_fields = (
        "sha256", "thresholds_sha256",
        "sbsbench_sha256", "run_eval_sha256", "executable_sha256",
    )
    if (not isinstance(catalog, dict) or
            any(not _sha256_value(catalog.get(field))
                for field in catalog_hash_fields) or
            not _contract_sha256_value(catalog.get("conf_sha256")) or
            not _contract_sha256_value(catalog.get("metric_sha256")) or
            catalog.get("metric_sha256") !=
            contract.get("metric_contract_sha256") or
            catalog.get("thresholds_sha256") !=
            contract.get("thresholds_sha256")):
        raise RuntimeError("ordinal training contract lacks catalog authentication")
    identifiability = contract.get("paired_variant_identifiability")
    if (not isinstance(identifiability, dict) or
            set(identifiability) != {"training", "development"}):
        raise RuntimeError("ordinal paired-variant identifiability contract differs")
    identifiability = {
        split: _validate_identifiability(identifiability[split], split)
        for split in ("training", "development")
    }
    if contract.get("promotion_prerequisites", {}).get(
            "promotion_blocked_until_both_true") is not True:
        raise RuntimeError("ordinal contract does not fail closed on promotion")

    acceptance = best["development"]["acceptance"]
    development_expected = expected_evidence["development"]
    if set(acceptance.get("regimes", {})) != set(REQUIRED_REGIMES):
        raise RuntimeError("ordinal acceptance runtime regime set differs")
    regime_groups = {
        name: _group_summary(acceptance["regimes"][name], name)
        for name in REQUIRED_REGIMES
    }
    variants_raw = acceptance.get("input_variants")
    expected_variant_names = development_expected["expected_variants"]
    if (not isinstance(variants_raw, dict) or
            set(variants_raw) != set(expected_variant_names)):
        raise RuntimeError("ordinal acceptance input-condition set differs")
    variants = {
        name: _group_summary(group, "variant:" + name)
        for name, group in sorted(variants_raw.items())
    }
    for name in REQUIRED_REGIMES:
        _require_target_counts(
            regime_groups[name], development_expected["regimes"][name], name
        )
    for name in expected_variant_names:
        _require_target_counts(
            variants[name], development_expected["variants"][name],
            "variant:" + name,
        )
    application = acceptance.get("application")
    expected_target_rows = sum(
        bucket["target_rows"]
        for bucket in development_expected["variants"].values()
    )
    if (not isinstance(application, dict) or
            application.get("mode") != "direct-target-only" or
            application.get("context_rows") != 0 or
            application.get("target_rows") != expected_target_rows):
        raise RuntimeError(
            "ordinal target-only application evidence counts disagree"
        )
    hdr_variant_names = [name for name in variants if name != "native_sdr"]
    if not hdr_variant_names:
        raise RuntimeError("ordinal acceptance lacks HDR variant evidence")

    sdr_pass = regime_groups["sdr"]["pass"] and variants["native_sdr"]["pass"]
    hdr_pass = regime_groups["hdr"]["pass"] and all(
        variants[name]["pass"] for name in hdr_variant_names
    )
    development_safety_pass = bool(sdr_pass and hdr_pass)
    measured_for_gain = [
        {**regime_groups["sdr"], "name": "sdr"},
        {**regime_groups["hdr"], "name": "hdr"},
    ] + [
        {**variants[name], "name": "variant:" + name}
        for name in hdr_variant_names
    ]
    calibration_evidence, calibration_deployable = _recompute_calibration(
        acceptance, measured_for_gain
    )
    usefulness = _material_gain_evidence(
        acceptance, contract,
        measured_for_gain,
    )
    complete_evidence_pass = all(
        group["unproven_samples"] == 0 for group in measured_for_gain
    )
    overshoot_only_pass = all(
        group["selected_overshoot_count"] == 0 and
        group["applied_overshoot_count"] == 0
        for group in measured_for_gain
    )
    zero_identity_failure_pass = all(
        group["identity_hard_failure_count"] == 0
        for group in measured_for_gain
    )
    plateau_utility = {
        "contract": "oracle-first-equal-pop-plateau-diagnostic-v1",
        "pop_tolerance_pct": contract.get("plateau_policy", {}).get(
            "oracle_pop_tolerance_pct"
        ),
        "selected_excess_count": sum(
            group["selected_plateau_excess_count"] for group in measured_for_gain
        ),
        "applied_excess_count": sum(
            group["applied_plateau_excess_count"] for group in measured_for_gain
        ),
        "optimal_utility_claim": all(
            group["selected_plateau_excess_count"] == 0 and
            group["applied_plateau_excess_count"] == 0
            for group in measured_for_gain
        ),
        "oracle_used_to_modify_direct_selection": False,
    }
    if usefulness["authenticated_threshold_present"]:
        expected_checkpoint_eligible = bool(
            complete_evidence_pass and
            zero_identity_failure_pass and overshoot_only_pass and
            usefulness["pass"]
        )
        if (usefulness["reported_complete_evidence_pass"] !=
                complete_evidence_pass):
            raise RuntimeError(
                "ordinal candidate complete-evidence status disagrees with "
                "group metrics"
            )
        if usefulness["reported_zero_overshoot_pass"] != overshoot_only_pass:
            raise RuntimeError(
                "ordinal candidate safety status disagrees with group metrics"
            )
        if (usefulness["reported_zero_identity_failure_pass"] !=
                zero_identity_failure_pass):
            raise RuntimeError(
                "ordinal candidate identity safety status disagrees with group metrics"
            )
        if (usefulness["reported_training_checkpoint_eligible"] !=
                expected_checkpoint_eligible):
            raise RuntimeError(
                "ordinal candidate checkpoint status is internally inconsistent"
            )
        if (best.get("checkpoint_eligible") is True) != expected_checkpoint_eligible:
            raise RuntimeError(
                "ordinal epoch checkpoint eligibility disagrees with candidate status"
            )
    candidate_accepted = bool(
        has_eligible_checkpoint and development_safety_pass and
        usefulness["pass"]
    )
    prerequisites = acceptance.get("promotion_prerequisites")
    if not isinstance(prerequisites, dict):
        raise RuntimeError("ordinal acceptance lacks promotion prerequisites")
    promotion_prerequisites_pass = bool(
        prerequisites.get(
            "independent_target_render_gate_passed"
        ) is True and
        prerequisites.get(
            "sealed_test_target_gate_passed"
        ) is True
    )
    failed = []
    if not has_eligible_checkpoint:
        failed.append(
            "no checkpoint passed the trainer's complete safety-and-material-"
            "gain gate"
        )
    if not sdr_pass:
        failed.append("native SDR failed its independent development gate")
    if not hdr_pass:
        failed.append("HDR or an HDR variant failed its independent development gate")
    if not usefulness["pass"]:
        failed.append(usefulness["reason"])
    if not plateau_utility["optimal_utility_claim"]:
        failed.append(
            "selected/applied scales exceed the first equal-pop safe action; "
            "optimal-utility claim is blocked"
        )
    if not calibration_deployable:
        failed.append("calibration evidence is not deployment-qualified")
    if not promotion_prerequisites_pass:
        failed.append("independent render and sealed-test gates are incomplete")
    if not development_safety_pass:
        conclusion = "Development safety gate failed closed: " + "; ".join(failed)
    elif usefulness["status"] == "pending":
        conclusion = (
            "Development safety evidence passed, but candidate usefulness and "
            "acceptance remain pending: " + "; ".join(failed)
        )
    elif not usefulness["pass"]:
        conclusion = (
            "Development safety evidence passed, but the candidate failed the "
            "material-gain gate: " + "; ".join(failed)
        )
    else:
        conclusion = (
            "Development safety and material-gain gates passed, but this "
            "experiment remains non-deployable: " + "; ".join(failed)
        )
    conclusion += (
        f" Worst direct target gain across SDR, HDR, and HDR variants: "
        f"{usefulness['worst_direct_gain_pct']:.3f}%."
    )
    # This V3 report is always marked development-only.  Even a future run with
    # every prerequisite true still requires a separate promotion transaction.
    deployable = False
    native_pq_development_holdout = bool(
        "native_pq" in development_expected["expected_variants"] and
        "native_pq" not in
        expected_evidence["training"]["expected_variants"]
    )
    warnings = []
    if not calibration_deployable:
        warnings.append(
            "Calibration is not deployment-qualified; development safety "
            "does not establish a production failure-rate bound."
        )
    if native_pq_development_holdout:
        warnings.append(
            "Native HDR PQ is a development-only holdout condition and was "
            "not admitted to training."
        )

    sealed = contract.get("sealed_test_productions")
    if not isinstance(sealed, list) or not all(isinstance(v, str) for v in sealed):
        raise RuntimeError("ordinal contract lacks sealed-test holdout provenance")
    provenance = {
        key: contract.get(key) for key in (
            "training_contract", "policy_contract", "output_semantics",
            "policy_feature_contract", "active_split_sha256",
            "source_bundles_sha256", "frontier_bundles_sha256",
            "orchestration_catalog", "metric_specs_sha256",
            "paired_variant_identifiability",
            "metric_contract_sha256", "thresholds_sha256",
            "deployment_geometry_allowlist_sha256",
            "deployment_geometry_structure_sha256", "depth_weights_sha256",
            "depth_input_color_contract_sha256",
            "minimum_development_realized_pop_gain_pct",
            "minimum_development_realized_pop_gain_rationale",
        )
    }
    provenance.update({
        "training_contract_json_sha256": sha256(contract_path),
        "history_json_sha256": sha256(history_path),
        "history_contract": HISTORY_CONTRACT,
        "checkpoint_receipt": checkpoint_receipt,
        "sealed_test_production_count": len(sealed),
        "sealed_test_content_read": False,
        "report_input_scope": (
            "training_contract.json and history.json plus checkpoint bytes for "
            "receipt hashing only; no checkpoint deserialization, media, "
            "labels, external data, or sealed-test content"
        ),
    })
    return {
        "schema": REPORT_SCHEMA,
        "scope": "development-only; not deployable",
        "conclusion": {
            "status": (
                "development_candidate_accept" if candidate_accepted else
                "safety_pass_usefulness_pending"
                if development_safety_pass and usefulness["status"] == "pending"
                else "failed"
            ),
            "text": conclusion,
            "development_safety_pass": development_safety_pass,
            "development_candidate_accepted": candidate_accepted,
            "deployable": deployable,
            "failed_conditions": failed,
            "warnings": warnings,
            "overall_average_used_for_acceptance": False,
        },
        "best_epoch": {
            "epoch": best["epoch"],
            "checkpoint_eligible": has_eligible_checkpoint,
            "selection_key": list(_epoch_key(best)),
            "selection_rule": (
                "lowest safety-first development selection key among eligible "
                "epochs; diagnostic best if none is eligible"
            ),
        },
        "acceptance": {
            "sdr_pass": sdr_pass,
            "hdr_pass": hdr_pass,
            "both_regimes_required": True,
            "regimes": regime_groups,
            "input_variants": variants,
            "expected_runtime_evidence": development_expected,
            "native_pq_development_holdout": native_pq_development_holdout,
            "calibration_evidence": calibration_evidence,
            "calibration_deployable": calibration_deployable,
            "promotion_prerequisites_pass": promotion_prerequisites_pass,
            "material_gain": usefulness,
            "plateau_utility": plateau_utility,
            "paired_variant_identifiability": identifiability,
            "candidate_gates": {
                "complete_evidence_pass": complete_evidence_pass,
                "zero_identity_failure_pass": zero_identity_failure_pass,
                "zero_overshoot_pass": overshoot_only_pass,
                "direct_target_safety_pass": development_safety_pass,
                "material_gain_status": usefulness["status"],
                "material_gain_pass": usefulness["pass"],
                "training_checkpoint_eligible": has_eligible_checkpoint,
                "production_policy_accepted": False,
            },
        },
        "curves": [_curve_record(record) for record in history],
        "provenance": provenance,
    }


def _fmt(value, digits=3):
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):.{digits}f}"


def _status_badge(passed):
    return (
        '<span class="badge pass">PASS</span>' if passed else
        '<span class="badge fail">FAIL</span>'
    )


def _group_table(groups):
    rows = []
    for label, group in groups:
        rows.append(
            "<tr>"
            f"<th>{html.escape(label)}</th>"
            f"<td>{_status_badge(group['pass'])}</td>"
            f"<td>{_fmt(group['samples'])}</td>"
            f"<td>{_fmt(group['proven_samples'])}</td>"
            f"<td>{_fmt(group['unproven_samples'])}</td>"
            f"<td>{_fmt(group['shots'])}</td>"
            f"<td>{_fmt(group['films'])}</td>"
            f"<td>{_fmt(group['independent_groups'])}</td>"
            f"<td>{_fmt(group['selected_pop_gain_pct'])}%</td>"
            f"<td>{_fmt(group['applied_pop_gain_pct'])}%</td>"
            f"<td>{_fmt(group['selected_overshoot_count'])}</td>"
            f"<td>{_fmt(group['applied_overshoot_count'])}</td>"
            f"<td>{_fmt(group['identity_hard_failure_count'])}</td>"
            f"<td>{_fmt(group['applied_plateau_excess_count'])}</td>"
            f"<td>{_fmt(100.0 * group['abstention_rate'], 2)}%</td>"
            f"<td>{_fmt(group['scale_underreach'])}</td>"
            f"<td>{_fmt(group['interval_nll'])}</td>"
            f"<td>{_fmt(group['known_bin_brier'])}</td>"
            f"<td>{_fmt(group['known_bin_ece'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def _polyline(values, width=640, height=170, pad=18):
    if not values:
        return ""
    low = min(values)
    high = max(values)
    span = high - low or 1.0
    count = len(values)
    points = []
    for index, value in enumerate(values):
        x = pad if count == 1 else pad + index * (width - 2 * pad) / (count - 1)
        y = height - pad - (value - low) * (height - 2 * pad) / span
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _curve_svg(curves, field, label):
    train = [item["training"][field] for item in curves]
    development = [item["development"][field] for item in curves]
    low = min(train + development)
    high = max(train + development)
    # Use the same scale for both lines by normalizing before the SVG helper.
    span = high - low or 1.0
    train_n = [(value - low) / span for value in train]
    development_n = [(value - low) / span for value in development]
    return (
        f'<h3>{html.escape(label)}</h3>'
        '<svg class="curve" viewBox="0 0 640 170" role="img" '
        f'aria-label="Training and development {html.escape(label)} by epoch">'
        '<line x1="18" y1="152" x2="622" y2="152" class="axis"/>'
        f'<polyline points="{_polyline(train_n)}" class="line train"/>'
        f'<polyline points="{_polyline(development_n)}" class="line dev"/>'
        '</svg>'
        f'<div class="legend"><span class="train-dot"></span>training '
        f'<span class="dev-dot"></span>development &nbsp; range '
        f'{_fmt(low)}–{_fmt(high)}</div>'
    )


def render_html(report):
    conclusion = report["conclusion"]
    acceptance = report["acceptance"]
    best = report["best_epoch"]
    sdr = acceptance["regimes"]["sdr"]
    hdr = acceptance["regimes"]["hdr"]
    variants = acceptance["input_variants"]
    variant_labels = {
        "native_sdr": "Native SDR",
        "hdr_raw1000": "HDR simulated SDR white 80 nit",
        "hdr_raw2500": "HDR simulated SDR white 200 nit",
        "hdr_raw6000": "HDR simulated SDR white 480 nit",
        "native_pq": "Native HDR PQ",
    }
    variant_rows = [
        (variant_labels.get(name, name), group)
        for name, group in variants.items()
    ]
    identifiability_rows = "".join(
        "<tr>"
        f"<th>{html.escape(split.title())}</th>"
        f"<td>{_fmt(value['paired_source_frames'])}</td>"
        f"<td>{_fmt(value['variant_pair_comparisons'])}</td>"
        f"<td>{_fmt(value['near_identical_feature_pairs'])}</td>"
        f"<td>{_fmt(value['near_identical_pairs_with_exact_depth'])}</td>"
        f"<td>{_fmt(value['contradictory_near_identical_pairs'])}</td>"
        "</tr>"
        for split, value in acceptance[
            "paired_variant_identifiability"
        ].items()
    )
    provenance = html.escape(json.dumps(
        report["provenance"], indent=2, sort_keys=True
    ))
    status_class = (
        "fail" if not conclusion["development_safety_pass"] else
        "pass" if conclusion["development_candidate_accepted"] else "pending"
    )
    material_status = acceptance["material_gain"]["status"]
    material_badge = (
        '<span class="badge pending">PENDING</span>'
        if material_status == "pending" else
        _status_badge(acceptance["material_gain"]["pass"])
    )
    learned_safety_pass = bool(
        acceptance["candidate_gates"]["zero_overshoot_pass"] and
        acceptance["candidate_gates"]["zero_identity_failure_pass"]
    )
    if acceptance["material_gain"]["authenticated_threshold_present"]:
        material_detail = (
            f"worst applied gain "
            f"{_fmt(acceptance['material_gain']['worst_direct_gain_pct'])}% "
            f"vs {_fmt(acceptance['material_gain']['minimum_required_direct_gain_pct'])}% floor"
        )
        material_rationale = (
            '<p class="scope">Frozen material-gain rationale: ' +
            html.escape(acceptance["material_gain"]["rationale"]) + "</p>"
        )
    else:
        material_detail = (
            "no authenticated material-gain threshold/status"
        )
        material_rationale = ""
    warning_html = "".join(
        '<p class="scope warning">Warning: ' + html.escape(message) + "</p>"
        for message in conclusion.get("warnings", [])
    )
    definitions = (
        "<dl>"
        "<dt>Selected pop gain</dt><dd>Conservative realized-pop gain for "
        "the model-selected scale on an independent target image.</dd>"
        "<dt>Applied pop gain</dt><dd>The same selected action applied directly; "
        "temporal controller behavior is outside this experiment.</dd>"
        "<dt>Overshoot</dt><dd>Count of frames authorized above the highest "
        "proven-safe two-geometry scale. Both selected and applied counts must "
        "be zero independently in SDR, HDR aggregate, and each HDR variant.</dd>"
        "<dt>Identity hard failure</dt><dd>Measured frame where scale 1.0 "
        "violates an absolute hard constraint. Abstention cannot hide it; every "
        "acceptance group requires zero.</dd>"
        "<dt>Plateau excess</dt><dd>Safe selected/applied action above the first "
        "scale attaining the same maximum realized pop. It does not weaken "
        "safety, but blocks an optimal-utility claim.</dd>"
        "<dt>Abstention</dt><dd>Fraction with no learned scale authorization; "
        "direct evaluation falls back to identity.</dd>"
        "<dt>Underreach</dt><dd>Mean scale interval left below the highest "
        "proven-safe frontier. Lower is more complete, after safety.</dd>"
        "<dt>Interval NLL / known-bin Brier / ECE</dt><dd>Fit and calibration "
        "metrics for interval-censored ordinal safety evidence. ECE is computed "
        "only over measured bins, then balanced scene-to-film. Lower is better.</dd>"
        "<dt>Independent groups</dt><dd>Authenticated source groups used for "
        "the zero-failure calibration bound; frames are not counted as "
        "independent trials.</dd>"
        "</dl>"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Apollo ordinal artistic policy — development report</title>
<style>
:root{{--bg:#0b1015;--panel:#121a22;--panel2:#17222c;--text:#e9f1f5;
--muted:#9bb0bc;--cyan:#42d7e8;--green:#55d68b;--red:#ff6f7d;--line:#293844}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);
font:14px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}}
main{{max-width:1500px;margin:auto;padding:28px}} h1,h2{{margin:0 0 14px}}
h1{{font-size:25px}} h2{{font-size:17px;color:var(--cyan);text-transform:uppercase;
letter-spacing:.08em}} .scope{{color:var(--muted);margin:0 0 22px}}
.conclusion{{border:1px solid var(--line);border-left:5px solid var(--red);
background:var(--panel);padding:20px;margin:0 0 18px;border-radius:9px}}
.conclusion.pass{{border-left-color:var(--green)}} .conclusion strong{{font-size:17px}}
.conclusion.pending{{border-left-color:#f0b65a}}
.cards{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px;margin:18px 0 28px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:16px}}
.card .label{{color:var(--muted);text-transform:uppercase;font-size:12px}}
.card .value{{font-size:21px;margin-top:7px}} .badge{{font-weight:800;padding:3px 8px;
border-radius:999px;font-size:11px}} .badge.pass{{color:#07180f;background:var(--green)}}
.badge.fail{{color:#21080c;background:var(--red)}} section{{margin:0 0 30px}}
.badge.pending{{color:#2b1b03;background:#f0b65a}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:9px}}
table{{border-collapse:collapse;width:100%;background:var(--panel);white-space:nowrap}}
th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right}}
thead th{{color:var(--muted);font-size:11px;text-transform:uppercase;background:var(--panel2)}}
tbody th,thead th:first-child{{text-align:left}} tbody tr:last-child th,
tbody tr:last-child td{{border-bottom:0}} .curve{{width:100%;max-width:900px;height:220px;
background:var(--panel);border:1px solid var(--line);border-radius:9px}}
.axis{{stroke:var(--line)}} .line{{fill:none;stroke-width:3}} .line.train{{stroke:var(--cyan)}}
.line.dev{{stroke:var(--red)}} .legend{{color:var(--muted)}}
.train-dot,.dev-dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}}
.train-dot{{background:var(--cyan)}} .dev-dot{{background:var(--red);margin-left:15px}}
details{{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:14px 16px}}
summary{{cursor:pointer;color:var(--cyan);font-weight:700}} dt{{color:var(--text);font-weight:700;
margin-top:12px}} dd{{color:var(--muted);margin-left:0}} pre{{overflow:auto;color:var(--muted)}}
.warning{{color:var(--red)}} @media(max-width:850px){{.cards{{grid-template-columns:1fr}}main{{padding:16px}}}}
</style></head><body><main>
<h1>Ordinal artistic policy V2</h1>
<p class="scope">Development-only report · not a deployment approval · sealed test untouched</p>
<div class="conclusion {status_class}"><strong>Conclusion</strong><br>
{html.escape(conclusion['text'])}</div>
{warning_html}
<div class="cards">
  <div class="card"><div class="label">Evidence completeness</div><div class="value">
  {_status_badge(acceptance['candidate_gates']['complete_evidence_pass'])}</div>
  <small>zero unproven frames in every acceptance group</small></div>
  <div class="card"><div class="label">Absolute + learned safety</div><div class="value">
  {_status_badge(learned_safety_pass)}</div>
  <small>zero overshoot and zero identity hard failures in every group</small></div>
  <div class="card"><div class="label">Material gain</div><div class="value">
  {material_badge}</div>
  <small>{html.escape(material_detail)}</small></div>
</div>
<p class="scope">Selected epoch {best['epoch']} · training checkpoint
{'eligible' if best['checkpoint_eligible'] else 'diagnostic only'} · deployment BLOCKED</p>
{material_rationale}
<section><h2>Metrics by runtime regime</h2><div class="table-wrap"><table>
<thead><tr><th>Group</th><th>Gate</th><th>Samples</th><th>Proven</th><th>Unproven</th>
<th>Shots</th><th>Films</th><th>Independent</th><th>Selected gain</th><th>Applied gain</th><th>Selected over</th>
<th>Applied over</th><th>Identity hard</th><th>Plateau excess</th><th>Abstain</th>
<th>Underreach</th><th>NLL</th><th>Brier</th><th>ECE</th></tr></thead>
<tbody>{_group_table([('Native SDR aggregate', sdr), ('HDR aggregate', hdr)])}</tbody>
</table></div></section>
<section><h2>HDR and input-condition breakdown</h2><div class="table-wrap"><table>
<thead><tr><th>Condition</th><th>Gate</th><th>Samples</th><th>Proven</th><th>Unproven</th>
<th>Shots</th><th>Films</th><th>Independent</th><th>Selected gain</th><th>Applied gain</th><th>Selected over</th>
<th>Applied over</th><th>Identity hard</th><th>Plateau excess</th><th>Abstain</th>
<th>Underreach</th><th>NLL</th><th>Brier</th><th>ECE</th></tr></thead>
<tbody>{_group_table(variant_rows)}</tbody></table></div></section>
<section><h2>Paired input identifiability diagnostic</h2>
<p class="scope">Each condition keeps its own safety target. Similar features never merge or
reject labels, and no runtime-condition metadata is a model input.</p>
<div class="table-wrap"><table><thead><tr><th>Split</th><th>Paired frames</th>
<th>Variant pairs</th><th>Near-identical features</th><th>Same depth</th>
<th>Conflicting safety bins</th></tr></thead><tbody>{identifiability_rows}</tbody>
</table></div></section>
<section><h2>Training curves</h2>
{_curve_svg(report['curves'], 'loss', 'Combined ordinal loss')}
{_curve_svg(report['curves'], 'interval_nll', 'Interval NLL')}
{_curve_svg(report['curves'], 'known_bin_brier', 'Known-bin Brier')}
</section>
<section><details><summary>Metric definitions</summary>{definitions}</details></section>
<section><details><summary>Provenance and authentication</summary><pre>{provenance}</pre></details></section>
</main></body></html>"""


def _atomic_write(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or args.training_output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = build_report(args.training_output)
    _atomic_write(
        output / "report.json",
        json.dumps(report, indent=2, allow_nan=False) + "\n",
    )
    _atomic_write(output / "report.html", render_html(report))
    print(output / "report.html")


if __name__ == "__main__":
    main()
