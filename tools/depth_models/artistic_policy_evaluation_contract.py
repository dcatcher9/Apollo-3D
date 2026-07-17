#!/usr/bin/env python3
"""Fail-closed structure checks for artistic-policy sealed evaluation evidence."""

from __future__ import annotations

import math


EVALUATION_SCHEMA = 13
EXPORT_METADATA_SCHEMA = 5
SEALED_APPROVAL_CONTRACT = "sealed-test-artistic-policy-v3"
RUNTIME_REGIME_ACCEPTANCE_CONTRACT = (
    "native-sdr-and-every-hdr-white-plus-coherent-hdr-aggregate-v1"
)
HDR_AGGREGATION_CONTRACT = (
    "per-source-frame-single-coherent-worst-risk-white-v1"
)
EXPECTED_HDR_WHITE_LEVELS_RAW = (1000, 2500, 6000)
MAX_UNSAFE_CEILING_OVERSHOOT_SCALE = 0.05
MAX_FILM_BALANCED_UNSAFE_CEILING_OVERSHOOT_SCALE = 0.01

METRICS = (
    "effective_scale_mae_pct",
    "actionable_scale_mae_pct",
    "rendered_disparity_mae_pct",
    "action_miss_pct",
)
ALL_METRICS = METRICS + (
    "raw_scale_mae_pct",
    "action_brier",
    "identity_false_action_pct",
)
COUNT_FIELDS = (
    "variant_sample_count",
    "unique_rgb_sample_count",
    "shot_count",
    "shot_condition_count",
    "film_count",
    "actionable_variant_sample_count",
    "identity_variant_sample_count",
    "actionable_shot_condition_count",
    "identity_shot_condition_count",
)
NUMERIC_PRIMARY_FIELDS = (
    "predicted_scale_mean",
    "target_scale_mean",
    "predicted_confidence_mean",
    "target_confidence_mean",
    "action_brier",
    "action_ece",
    "identity_false_action_pct",
    "rendered_disparity_mean_abs_pct",
    "target_rendered_disparity_mean_abs_pct",
    "rendered_disparity_mae_pct",
    "maximum_unsafe_ceiling_overshoot_scale",
    "film_balanced_mean_unsafe_ceiling_overshoot_scale",
    "film_balanced_unsafe_ceiling_overshoot_rate_pct",
)


def _number(value, origin, *, lower=None, upper=None):
    if (not isinstance(value, (int, float)) or isinstance(value, bool) or
            not math.isfinite(float(value))):
        raise RuntimeError(f"{origin} is not finite numeric evidence")
    value = float(value)
    if lower is not None and value < lower - 1e-12:
        raise RuntimeError(f"{origin} is below its valid range")
    if upper is not None and value > upper + 1e-12:
        raise RuntimeError(f"{origin} is above its valid range")
    return value


def _positive_int(value, origin):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(f"{origin} is not a positive integer")
    return value


def validate_unsafe_ceiling_overshoot(payload, decision, origin):
    """Validate one overall or group-local sealed safety guard."""
    evidence = payload.get("unsafe_ceiling_overshoot")
    if not isinstance(evidence, dict):
        raise RuntimeError(f"{origin} lacks unsafe-ceiling evidence")
    maximum = _number(
        evidence.get("maximum_scale"), f"{origin} maximum unsafe ceiling",
        lower=0.0,
    )
    maximum_limit = _number(
        evidence.get("maximum_limit_scale"),
        f"{origin} maximum unsafe-ceiling limit", lower=0.0,
    )
    film_mean = _number(
        evidence.get("film_balanced_mean_scale"),
        f"{origin} film-balanced unsafe ceiling", lower=0.0,
    )
    film_limit = _number(
        evidence.get("film_balanced_mean_limit_scale"),
        f"{origin} film-balanced unsafe-ceiling limit", lower=0.0,
    )
    _number(
        evidence.get("film_balanced_overshoot_rate_pct"),
        f"{origin} unsafe-ceiling rate", lower=0.0, upper=100.0,
    )
    if (abs(maximum_limit - MAX_UNSAFE_CEILING_OVERSHOOT_SCALE) > 1e-12 or
            abs(film_limit -
                MAX_FILM_BALANCED_UNSAFE_CEILING_OVERSHOOT_SCALE) > 1e-12):
        raise RuntimeError(f"{origin} uses different unsafe-ceiling limits")
    if (evidence.get("maximum_pass") is not True or
            evidence.get("film_balanced_mean_pass") is not True or
            maximum > MAX_UNSAFE_CEILING_OVERSHOOT_SCALE + 1e-9 or
            film_mean >
            MAX_FILM_BALANCED_UNSAFE_CEILING_OVERSHOOT_SCALE + 1e-9):
        raise RuntimeError(f"{origin} failed unsafe-ceiling guards")
    guards = decision.get("guards")
    if (not isinstance(guards, dict) or
            decision.get("unsafe_overshoot_guard_required") is not True or
            guards.get("unsafe_ceiling_maximum") is not True or
            guards.get("unsafe_ceiling_film_balanced_mean") is not True):
        raise RuntimeError(f"{origin} decision lacks unsafe-ceiling guards")
    if decision.get("unsafe_ceiling_overshoot") != evidence:
        raise RuntimeError(f"{origin} decision has inconsistent safety evidence")
    return evidence


def _validate_metric_mapping(value, origin):
    if not isinstance(value, dict):
        raise RuntimeError(f"{origin} is not a metric mapping")
    for metric in ALL_METRICS:
        _number(value.get(metric), f"{origin} {metric}", lower=0.0)


def _validate_majority_axis(decision, axis, origin):
    counts = decision.get(f"{axis}_count")
    wins = decision.get(f"{axis}_wins")
    required = decision.get(f"required_{axis}_wins")
    if not all(isinstance(value, dict) for value in (counts, wins, required)):
        raise RuntimeError(f"{origin} lacks {axis} majority evidence")
    for metric in METRICS:
        count = _positive_int(counts.get(metric), f"{origin} {axis} {metric} count")
        win = _positive_int(wins.get(metric), f"{origin} {axis} {metric} wins")
        need = _positive_int(
            required.get(metric), f"{origin} {axis} {metric} required wins"
        )
        if need != count // 2 + 1 or win < need or win > count:
            raise RuntimeError(f"{origin} failed {axis} majority for {metric}")


def validate_accepted_measurement_group(group, origin):
    """Reject empty, self-inconsistent, or unsafe accepted group placeholders."""
    if not isinstance(group, dict):
        raise RuntimeError(f"{origin} result group is missing")
    decision = group.get("decision")
    primary = group.get("primary")
    evaluation = group.get("evaluation")
    if (not isinstance(decision, dict) or decision.get("accepted") is not True or
            not isinstance(primary, dict) or not isinstance(evaluation, dict)):
        raise RuntimeError(f"{origin} result group was not accepted")
    aggregate_wins = decision.get("aggregate_wins")
    if (not isinstance(aggregate_wins, dict) or
            any(aggregate_wins.get(metric) is not True for metric in METRICS)):
        raise RuntimeError(f"{origin} lacks aggregate wins on every primary axis")
    for axis in ("sequence", "domain", "film"):
        _validate_majority_axis(decision, axis, origin)
    minimum_films = decision.get("minimum_film_count")
    film_counts = decision.get("film_count")
    if not isinstance(minimum_films, dict) or not isinstance(film_counts, dict):
        raise RuntimeError(f"{origin} lacks minimum-film evidence")
    for metric in METRICS:
        minimum = _positive_int(
            minimum_films.get(metric), f"{origin} {metric} minimum films"
        )
        count = _positive_int(
            film_counts.get(metric), f"{origin} {metric} film count"
        )
        if count < minimum:
            raise RuntimeError(f"{origin} has insufficient films for {metric}")
    guards = decision.get("guards")
    if (not isinstance(guards, dict) or
            decision.get("identity_guard_required") is not True or
            guards.get("identity_examples_present") is not True or
            guards.get("identity_false_action_pct") is not True):
        raise RuntimeError(f"{origin} lacks the identity guard")

    for field in COUNT_FIELDS:
        _positive_int(primary.get(field), f"{origin} primary {field}")
    if (primary["actionable_variant_sample_count"] +
            primary["identity_variant_sample_count"] !=
            primary["variant_sample_count"]):
        raise RuntimeError(f"{origin} sample class counts are inconsistent")
    if (primary["actionable_shot_condition_count"] +
            primary["identity_shot_condition_count"] !=
            primary["shot_condition_count"]):
        raise RuntimeError(f"{origin} shot class counts are inconsistent")
    if (primary["film_count"] > primary["shot_count"] or
            primary["shot_count"] > primary["shot_condition_count"] or
            primary["unique_rgb_sample_count"] >
            primary["variant_sample_count"]):
        raise RuntimeError(f"{origin} primary counts are not nested")
    for field in NUMERIC_PRIMARY_FIELDS:
        _number(primary.get(field), f"{origin} primary {field}", lower=0.0)
    for candidate in ("trained", "neutral"):
        _validate_metric_mapping(
            evaluation.get(candidate), f"{origin} {candidate} evaluation"
        )
    if not math.isclose(
            float(primary["rendered_disparity_mae_pct"]),
            float(evaluation["trained"]["rendered_disparity_mae_pct"]),
            abs_tol=1e-9):
        raise RuntimeError(f"{origin} primary rendered disparity is inconsistent")
    evidence = validate_unsafe_ceiling_overshoot(group, decision, origin)
    if (not math.isclose(
            float(primary["maximum_unsafe_ceiling_overshoot_scale"]),
            float(evidence["maximum_scale"]), abs_tol=1e-9) or
            not math.isclose(
                float(primary[
                    "film_balanced_mean_unsafe_ceiling_overshoot_scale"
                ]),
                float(evidence["film_balanced_mean_scale"]), abs_tol=1e-9) or
            not math.isclose(
                float(primary[
                    "film_balanced_unsafe_ceiling_overshoot_rate_pct"
                ]),
                float(evidence["film_balanced_overshoot_rate_pct"]),
                abs_tol=1e-9)):
        raise RuntimeError(f"{origin} primary safety summary is inconsistent")
    return primary


def _same_count(primary, expected, field, origin):
    if primary.get(field) != expected:
        raise RuntimeError(f"{origin} has inconsistent {field}")


def validate_runtime_regime_acceptance(
        payload, decision, condition_target_contract,
        expected_hdr_whites=EXPECTED_HDR_WHITE_LEVELS_RAW):
    """Validate every condition group and recompute sealed runtime acceptance."""
    runtime = payload.get("runtime_regime_evaluation")
    summary = decision.get("runtime_regime_acceptance")
    expected_hdr_whites = tuple(sorted(int(value) for value in expected_hdr_whites))
    if not isinstance(runtime, dict) or not isinstance(summary, dict):
        raise RuntimeError("sealed-test evaluation lacks runtime-regime evidence")
    fixed = {
        "contract": RUNTIME_REGIME_ACCEPTANCE_CONTRACT,
        "condition_target_contract": condition_target_contract,
        "hdr_aggregation_contract": HDR_AGGREGATION_CONTRACT,
        "required_regimes": ["sdr", "hdr"],
        "expected_hdr_white_levels_raw": list(expected_hdr_whites),
        "missing_regimes": [],
        "missing_hdr_white_levels_raw": [],
        "unexpected_hdr_white_levels_raw": [],
        "incomplete_source_frame_count": 0,
        "source_condition_coverage_complete": True,
        "hdr_white_pass": {str(value): True for value in expected_hdr_whites},
        "hdr_aggregate_pass": True,
        "regime_pass": {"sdr": True, "hdr": True},
        "accepted": True,
    }
    if summary != fixed or any(runtime.get(key) != value
                               for key, value in fixed.items()):
        raise RuntimeError("sealed-test evaluation failed per-condition acceptance")
    if runtime.get("incomplete_source_frames") != []:
        raise RuntimeError("sealed-test source-condition coverage is incomplete")
    regimes = runtime.get("regimes")
    whites = runtime.get("hdr_by_white_level_raw")
    if (not isinstance(regimes, dict) or set(regimes) != {"sdr", "hdr"} or
            not isinstance(whites, dict) or
            set(whites) != {str(value) for value in expected_hdr_whites}):
        raise RuntimeError("sealed-test evaluation lacks exact runtime result groups")

    sdr = validate_accepted_measurement_group(regimes["sdr"], "sealed SDR")
    hdr = validate_accepted_measurement_group(regimes["hdr"], "sealed HDR aggregate")
    white_primary = {
        key: validate_accepted_measurement_group(
            whites[key], f"sealed HDR white {key}"
        )
        for key in sorted(whites, key=int)
    }
    val_films = payload.get("val_films")
    if (not isinstance(val_films, list) or not val_films or
            any(not isinstance(value, str) or not value for value in val_films) or
            len(set(val_films)) != len(val_films)):
        raise RuntimeError("sealed-test evaluation has invalid film identities")
    expected_film_count = len(val_films)
    for origin, primary in (
            ("sealed SDR", sdr), ("sealed HDR aggregate", hdr),
            *((f"sealed HDR white {key}", primary)
              for key, primary in white_primary.items())):
        _same_count(primary, expected_film_count, "film_count", origin)
    first = white_primary[str(expected_hdr_whites[0])]
    for key, primary in white_primary.items():
        _same_count(primary, primary["unique_rgb_sample_count"],
                    "variant_sample_count", f"sealed HDR white {key}")
        for field in ("unique_rgb_sample_count", "shot_count", "film_count"):
            _same_count(primary, first[field], field, f"sealed HDR white {key}")
    _same_count(sdr, sdr["unique_rgb_sample_count"],
                "variant_sample_count", "sealed SDR")
    for field in ("unique_rgb_sample_count", "shot_count", "film_count"):
        _same_count(sdr, first[field], field, "sealed SDR")
        _same_count(hdr, first[field], field, "sealed HDR aggregate")
    for field in (
            "variant_sample_count", "shot_condition_count",
            "actionable_variant_sample_count", "identity_variant_sample_count",
            "actionable_shot_condition_count", "identity_shot_condition_count"):
        expected = sum(primary[field] for primary in white_primary.values())
        _same_count(hdr, expected, field, "sealed HDR aggregate")

    guards = decision.get("guards")
    required_guards = (
        "runtime_regimes_present", "hdr_white_levels_present",
        "hdr_white_levels_accepted", "no_unexpected_hdr_white_levels",
        "source_condition_coverage_complete", "condition_target_contract",
        "sdr_and_hdr_accepted",
    )
    if (decision.get("overall_diagnostic_accepted") is not True or
            not isinstance(guards, dict) or
            any(guards.get(key) is not True for key in required_guards)):
        raise RuntimeError("sealed-test decision lacks fail-closed condition guards")
    return summary
