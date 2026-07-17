#!/usr/bin/env python3
"""Evaluate a policy checkpoint against a neutral head and write an HTML report."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
import torch

import merge_artistic_geometry_labels as label_merge
from artistic_policy_evaluation_contract import (
    EVALUATION_SCHEMA,
    HDR_AGGREGATION_CONTRACT,
    RUNTIME_REGIME_ACCEPTANCE_CONTRACT,
)
from artistic_policy_model import (
    ArtisticPolicyModel,
    load_depth_anything_small,
    load_policy_state,
    use_dynamic_onnx_position_encoding,
)
from artistic_geometry_contract import allowlist_sha256, validate_allowlist
from train_artistic_policy import (
    PolicyDataset,
    film_balanced_acceptance,
    input_variant_acceptance_risk,
    input_variant_runtime_regime,
    is_actionable_scale,
    labels_contract,
    load_active_split,
    load_rows,
    sha256,
    validate_rows_against_active_split,
)


METRICS = (
    ("effective_scale_mae_pct", "First-frame effective ceiling MAE", "percentage points"),
    ("actionable_scale_mae_pct", "First-frame actionable ceiling MAE", "percentage points"),
    ("rendered_disparity_mae_pct", "Exact rendered-disparity MAE", "eye-width percent"),
    ("action_miss_pct", "First-frame actionable miss rate", "percent"),
)
DIAGNOSTICS = (
    ("raw_scale_mae_pct", "First-frame raw ceiling MAE", "percentage points"),
    ("action_brier", "First-frame shot-action Brier score", "probability squared"),
    ("identity_false_action_pct", "First-frame identity false-action rate", "percent"),
)
ALL_METRICS = METRICS + DIAGNOSTICS
MAX_IDENTITY_FALSE_ACTION_PCT = 5.0
MAX_UNSAFE_CEILING_OVERSHOOT_SCALE = 0.05
MAX_FILM_BALANCED_UNSAFE_CEILING_OVERSHOOT_SCALE = 0.01


def _rendered_disparity_stats(raw_disparities, clamp_abs, scale,
                              target_scale):
    """Return worst-geometry exact post-clamp magnitude and target error."""
    if len(raw_disparities) != len(clamp_abs) or not raw_disparities:
        raise RuntimeError("rendered-disparity geometry evidence is incomplete")
    magnitudes = []
    errors = []
    for raw, limit in zip(raw_disparities, clamp_abs):
        if isinstance(raw, torch.Tensor):
            raw = raw.detach().cpu().numpy()
        raw = np.asarray(raw, dtype=np.float32)
        limit = float(limit)
        if (raw.size == 0 or not np.isfinite(raw).all() or
                not np.isfinite(limit) or limit <= 0.0):
            raise RuntimeError("rendered-disparity geometry evidence is invalid")
        predicted = np.clip(raw * float(scale), -limit, limit)
        target = np.clip(raw * float(target_scale), -limit, limit)
        magnitudes.append(float(np.mean(np.abs(predicted))) * 100.0)
        errors.append(float(np.mean(np.abs(predicted - target))) * 100.0)
    return max(magnitudes), max(errors)


def measure(model, row, sample, device):
    image, global_target, raw_disparity, clamp_abs = sample
    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
            pred_global = model.forward_policy(image[None].to(device))
    pred_global = pred_global[0].float().cpu().numpy()
    global_target = global_target.numpy()

    action = is_actionable_scale(global_target[0])
    target_effective = float(global_target[0] if action else 1.0)

    def candidate_metrics(scale, confidence):
        predicted_action = confidence >= 0.5
        predicted_effective = float(scale if predicted_action else 1.0)
        rendered_magnitude, rendered_error = _rendered_disparity_stats(
            raw_disparity, clamp_abs, predicted_effective, target_effective
        )
        return {
            "effective_scale_mae_pct": (
                abs(predicted_effective - target_effective) * 100.0
            ),
            "raw_scale_mae_pct": abs(float(scale) - float(global_target[0])) * 100.0,
            "actionable_scale_mae_pct": (
                abs(float(scale) - float(global_target[0])) * 100.0
                if action else None
            ),
            "action_miss_pct": (
                0.0 if predicted_action else 100.0
            ) if action else None,
            "rendered_disparity_mae_pct": rendered_error,
            "rendered_disparity_mean_abs_pct": rendered_magnitude,
            "identity_false_action_pct": (
                100.0 if predicted_action else 0.0
            ) if not action else None,
            "action_brier": float((confidence - float(action)) ** 2),
        }

    trained = candidate_metrics(pred_global[0], pred_global[1])
    neutral = candidate_metrics(1.0, 0.02)
    target_rendered, _target_error = _rendered_disparity_stats(
        raw_disparity, clamp_abs, target_effective, target_effective
    )
    variant = row.get("_input_variant")
    if variant is None:
        raise RuntimeError("expanded evaluation sample lacks authenticated input variant")
    regime = input_variant_runtime_regime(variant)
    return {"clip": row["clip"], "frame": row["frame"],
            "domain": row.get("domain") or "unknown",
            "film_id": row.get("film_id") or row["clip"],
            "input_variant_sha256": row.get("_input_variant_sha256"),
            "runtime_regime": regime,
            "hdr_white_level_raw": (
                variant["windows_sdr_white_level_raw"] if regime == "hdr" else None
            ),
            "prediction": {
                "scale": float(pred_global[0]),
                "confidence": float(pred_global[1]),
                "rendered_disparity_mean_abs_pct": trained[
                    "rendered_disparity_mean_abs_pct"
                ],
            },
            "target": {
                "scale": float(global_target[0]),
                "confidence": float(global_target[1]),
                "rendered_disparity_mean_abs_pct": target_rendered,
            },
            "trained": trained, "neutral": neutral}


def worst_input_variant_measurements(rows):
    """Select one coherent worst-risk input condition for every source frame.

    The trained prediction, neutral baseline, target, and rendered evidence must
    all come from the same authenticated condition.  Independently maximizing
    trained and neutral errors can manufacture a comparison that occurred at no
    real white level and can hide a white-level regression.
    """
    groups = {}
    for row in rows:
        groups.setdefault(
            (row["film_id"], row["clip"], int(row["frame"])), []
        ).append(row)
    collapsed = []
    for _key, variants in sorted(groups.items()):
        if len(variants) == 1:
            collapsed.append(variants[0])
            continue

        def measurement_risk(row):
            core = tuple(
                float(row["trained"][metric])
                if row["trained"][metric] is not None else -1.0
                for metric, _label, _unit in METRICS
            )
            return (
                *input_variant_acceptance_risk(
                    (row["prediction"]["scale"],
                     row["prediction"]["confidence"]),
                    (row["target"]["scale"], row["target"]["confidence"]),
                ),
                *core,
                str(row.get("input_variant_sha256", "")),
            )

        representative = max(
            variants,
            key=measurement_risk,
        )
        merged = dict(representative)
        merged["input_variant_sha256s"] = sorted({
            str(row.get("input_variant_sha256")) for row in variants
        })
        merged["input_variant_aggregation"] = HDR_AGGREGATION_CONTRACT
        collapsed.append(merged)
    return collapsed


def aggregate(rows):
    def mean_metric(candidate, key):
        values = [
            row[candidate][key]
            for row in rows
            if row[candidate][key] is not None
        ]
        return float(np.mean(values)) if values else None

    result = {}
    for candidate in ("trained", "neutral"):
        result[candidate] = {
            key: mean_metric(candidate, key)
            for key, _, _ in ALL_METRICS
        }
    return result


def aggregate_summaries(summaries):
    if not summaries:
        raise RuntimeError("cannot aggregate an empty summary set")

    def mean_metric(candidate, key):
        values = [
            summary[candidate][key]
            for summary in summaries
            if summary[candidate][key] is not None
        ]
        return float(np.mean(values)) if values else None

    return {
        candidate: {
            key: mean_metric(candidate, key)
            for key, _, _ in ALL_METRICS
        }
        for candidate in ("trained", "neutral")
    }


def first_frame_measurements(rows):
    """Keep the earliest available label used by the runtime shot latch."""
    first = {}
    for row in rows:
        key = (row["film_id"], row["clip"])
        candidate = (int(row["frame"]), row)
        if key not in first or candidate[0] < first[key][0]:
            first[key] = candidate
    return [candidate[1] for _, candidate in sorted(first.items())]


def aggregate_domains(rows, by_clip):
    return {
        domain: aggregate_summaries([
            by_clip[clip]
            for clip in sorted({row["clip"] for row in rows
                                if row["domain"] == domain})
        ])
        for domain in sorted({row["domain"] for row in rows})
    }


def aggregate_films(rows, by_clip):
    return {
        film: aggregate_summaries([
            by_clip[clip]
            for clip in sorted({row["clip"] for row in rows
                                if row["film_id"] == film})
        ])
        for film in sorted({row["film_id"] for row in rows})
    }


def shot_latched_aggregates(rows):
    """Aggregate only the first available frame that can drive each shot latch."""
    first_rows = first_frame_measurements(
        worst_input_variant_measurements(rows)
    )
    by_clip = {
        clip: aggregate([row for row in first_rows if row["clip"] == clip])
        for clip in sorted({row["clip"] for row in first_rows})
    }
    by_domain = aggregate_domains(first_rows, by_clip)
    by_film = aggregate_films(first_rows, by_clip)
    overall = aggregate_summaries(
        [by_domain[domain] for domain in sorted(by_domain)]
    )
    return first_rows, by_clip, by_domain, by_film, overall


def unsafe_ceiling_overshoot(first_rows):
    """Measure only runtime-effective predictions above the labelled safe cap."""
    if not first_rows:
        raise RuntimeError("cannot evaluate unsafe overshoot without shot labels")
    by_film = {}
    for row in first_rows:
        prediction = row["prediction"]
        target = row["target"]
        predicted_effective = (
            float(prediction["scale"])
            if float(prediction["confidence"]) >= 0.5 else 1.0
        )
        overshoot = max(predicted_effective - float(target["scale"]), 0.0)
        by_film.setdefault(row["film_id"], []).append(overshoot)
    film_means = {
        film: float(np.mean(values)) for film, values in by_film.items()
    }
    film_rates = {
        film: float(np.mean(np.asarray(values) > 1e-9) * 100.0)
        for film, values in by_film.items()
    }
    maximum = max(max(values) for values in by_film.values())
    film_balanced_mean = float(np.mean(list(film_means.values())))
    film_balanced_rate = float(np.mean(list(film_rates.values())))
    return {
        "maximum_scale": maximum,
        "maximum_limit_scale": MAX_UNSAFE_CEILING_OVERSHOOT_SCALE,
        "film_balanced_mean_scale": film_balanced_mean,
        "film_balanced_mean_limit_scale": (
            MAX_FILM_BALANCED_UNSAFE_CEILING_OVERSHOOT_SCALE
        ),
        "film_balanced_overshoot_rate_pct": film_balanced_rate,
        "by_film_mean_scale": film_means,
        "by_film_overshoot_rate_pct": film_rates,
        "maximum_pass": maximum <= MAX_UNSAFE_CEILING_OVERSHOOT_SCALE + 1e-9,
        "film_balanced_mean_pass": (
            film_balanced_mean <=
            MAX_FILM_BALANCED_UNSAFE_CEILING_OVERSHOOT_SCALE + 1e-9
        ),
    }


def _prediction_arrays(rows):
    predicted = np.asarray([
        [row["prediction"]["scale"], row["prediction"]["confidence"]]
        for row in rows
    ], dtype=np.float64)
    target = np.asarray([
        [row["target"]["scale"], row["target"]["confidence"]]
        for row in rows
    ], dtype=np.float64)
    neutral = np.tile(np.asarray([[1.0, 0.02]]), (len(rows), 1))
    return predicted, target, neutral


def first_condition_measurements(rows):
    """Keep the first labelled frame for each shot and authenticated condition."""
    first = {}
    for row in rows:
        variant = row.get("input_variant_sha256")
        if not isinstance(variant, str) or not variant:
            raise RuntimeError("measurement lacks authenticated input-variant identity")
        key = (row["film_id"], row["clip"], variant)
        candidate = (int(row["frame"]), row)
        if key not in first or candidate[0] < first[key][0]:
            first[key] = candidate
    return [candidate[1] for _key, candidate in sorted(first.items())]


def group_primary_summary(sample_rows, first_rows, evaluation, calibration,
                          overshoot):
    """Summarize policy outputs and exact render safety for one input regime."""
    if not sample_rows or not first_rows:
        raise RuntimeError("cannot summarize an empty input regime")

    def mean_nested(candidate, key):
        return float(np.mean([
            float(row[candidate][key]) for row in first_rows
        ]))

    unique_rgb = {
        (row["film_id"], row["clip"], int(row["frame"]))
        for row in sample_rows
    }
    unique_shots = {
        (row["film_id"], row["clip"]) for row in sample_rows
    }
    condition_first = first_condition_measurements(sample_rows)
    sample_actions = [
        is_actionable_scale(row["target"]["scale"]) for row in sample_rows
    ]
    shot_actions = [
        is_actionable_scale(row["target"]["scale"])
        for row in condition_first
    ]
    macro = calibration["trained"]["macro"]
    return {
        "variant_sample_count": len(sample_rows),
        "unique_rgb_sample_count": len(unique_rgb),
        "shot_count": len(unique_shots),
        "shot_condition_count": len(condition_first),
        "film_count": len({row["film_id"] for row in sample_rows}),
        "actionable_variant_sample_count": sum(sample_actions),
        "identity_variant_sample_count": len(sample_actions) - sum(sample_actions),
        "actionable_shot_condition_count": sum(shot_actions),
        "identity_shot_condition_count": len(shot_actions) - sum(shot_actions),
        "predicted_scale_mean": mean_nested("prediction", "scale"),
        "target_scale_mean": mean_nested("target", "scale"),
        "predicted_confidence_mean": mean_nested("prediction", "confidence"),
        "target_confidence_mean": mean_nested("target", "confidence"),
        "action_brier": evaluation["trained"]["action_brier"],
        "action_ece": macro["action_ece"],
        "identity_false_action_pct": evaluation["trained"][
            "identity_false_action_pct"
        ],
        "rendered_disparity_mean_abs_pct": mean_nested(
            "prediction", "rendered_disparity_mean_abs_pct"
        ),
        "target_rendered_disparity_mean_abs_pct": mean_nested(
            "target", "rendered_disparity_mean_abs_pct"
        ),
        "rendered_disparity_mae_pct": evaluation["trained"][
            "rendered_disparity_mae_pct"
        ],
        "maximum_unsafe_ceiling_overshoot_scale": overshoot["maximum_scale"],
        "film_balanced_mean_unsafe_ceiling_overshoot_scale": overshoot[
            "film_balanced_mean_scale"
        ],
        "film_balanced_unsafe_ceiling_overshoot_rate_pct": overshoot[
            "film_balanced_overshoot_rate_pct"
        ],
    }


def evaluate_measurement_group(rows, minimum_films=2,
                               require_identity_guard=True,
                               require_unsafe_overshoot_guard=False):
    """Evaluate one non-empty SDR, HDR, or fixed-white measurement group."""
    first_rows, by_clip, by_domain, by_film, evaluation = (
        shot_latched_aggregates(rows)
    )
    clips = sorted(by_clip)
    domains = sorted(by_domain)
    films = sorted(by_film)
    overshoot = unsafe_ceiling_overshoot(first_rows)
    predicted, target, neutral = _prediction_arrays(rows)
    calibration = {
        "trained": film_balanced_acceptance(predicted, target, rows),
        "neutral": film_balanced_acceptance(neutral, target, rows),
    }
    decision = policy_decision(
        evaluation, by_clip, by_domain, by_film, clips, domains, films,
        minimum_films=minimum_films,
        require_identity_guard=require_identity_guard,
        unsafe_overshoot=overshoot,
        require_unsafe_overshoot_guard=require_unsafe_overshoot_guard,
    )
    return {
        "evaluation": evaluation,
        "unsafe_ceiling_overshoot": overshoot,
        "film_balanced_acceptance": calibration,
        "by_clip": by_clip,
        "validation_by_domain": by_domain,
        "validation_by_film": by_film,
        "decision": decision,
        "primary": group_primary_summary(
            rows, first_rows, evaluation, calibration, overshoot
        ),
    }


def evaluate_runtime_regimes(rows, expected_hdr_whites, minimum_films=2,
                             require_identity_guard=True,
                             require_unsafe_overshoot_guard=False):
    """Evaluate native SDR, every HDR white, and coherent aggregate HDR."""
    expected_hdr_whites = tuple(sorted(int(value) for value in expected_hdr_whites))
    if not expected_hdr_whites or len(set(expected_hdr_whites)) != len(
            expected_hdr_whites):
        raise RuntimeError("expected HDR white anchors must be unique and non-empty")
    unexpected_white_set = set()
    for row in rows:
        regime = row.get("runtime_regime")
        white = row.get("hdr_white_level_raw")
        if regime not in {"sdr", "hdr"}:
            raise RuntimeError("measurement has an invalid runtime regime")
        if regime == "sdr" and white is not None:
            raise RuntimeError("native SDR measurement has an HDR white level")
        if regime == "hdr":
            if (not isinstance(white, int) or isinstance(white, bool) or
                    white <= 0):
                raise RuntimeError("HDR measurement has an invalid white level")
            if white not in expected_hdr_whites:
                unexpected_white_set.add(white)
    by_regime = {
        regime: [row for row in rows if row.get("runtime_regime") == regime]
        for regime in ("sdr", "hdr")
    }
    by_white = {
        white: [row for row in by_regime["hdr"]
                if row.get("hdr_white_level_raw") == white]
        for white in expected_hdr_whites
    }
    unexpected_whites = sorted(unexpected_white_set)
    source_conditions = {}
    for row in rows:
        identity = (row["film_id"], row["clip"], int(row["frame"]))
        condition = (
            "sdr" if row.get("runtime_regime") == "sdr" else
            f"hdr-{row.get('hdr_white_level_raw')}"
        )
        variants = source_conditions.setdefault(identity, {})
        if condition in variants:
            raise RuntimeError(
                f"duplicate runtime condition {condition} for source frame {identity}"
            )
        variants[condition] = row.get("input_variant_sha256")
    expected_conditions = {"sdr"} | {
        f"hdr-{white}" for white in expected_hdr_whites
    }
    incomplete_source_frames = [
        {
            "film_id": identity[0], "clip": identity[1], "frame": identity[2],
            "missing_conditions": sorted(expected_conditions - set(conditions)),
            "unexpected_conditions": sorted(set(conditions) - expected_conditions),
        }
        for identity, conditions in sorted(source_conditions.items())
        if set(conditions) != expected_conditions
    ]
    missing_regimes = [regime for regime, values in by_regime.items() if not values]
    missing_whites = [white for white, values in by_white.items() if not values]
    regimes = {
        regime: evaluate_measurement_group(
            values, minimum_films, require_identity_guard,
            require_unsafe_overshoot_guard,
        )
        for regime, values in by_regime.items() if values
    }
    hdr_by_white = {
        str(white): evaluate_measurement_group(
            values, minimum_films, require_identity_guard,
            require_unsafe_overshoot_guard,
        )
        for white, values in by_white.items() if values
    }
    hdr_white_pass = {
        str(white): (
            hdr_by_white.get(str(white), {}).get("decision", {}).get("accepted")
            is True
        )
        for white in expected_hdr_whites
    }
    hdr_aggregate_pass = (
        regimes.get("hdr", {}).get("decision", {}).get("accepted") is True
    )
    regime_pass = {
        "sdr": regimes.get("sdr", {}).get("decision", {}).get("accepted") is True,
        "hdr": hdr_aggregate_pass and all(hdr_white_pass.values()),
    }
    return {
        "contract": RUNTIME_REGIME_ACCEPTANCE_CONTRACT,
        "condition_target_contract": label_merge.CONDITION_TARGET_CONTRACT,
        "hdr_aggregation_contract": HDR_AGGREGATION_CONTRACT,
        "required_regimes": ["sdr", "hdr"],
        "expected_hdr_white_levels_raw": list(expected_hdr_whites),
        "missing_regimes": missing_regimes,
        "missing_hdr_white_levels_raw": missing_whites,
        "unexpected_hdr_white_levels_raw": unexpected_whites,
        "incomplete_source_frame_count": len(incomplete_source_frames),
        "incomplete_source_frames": incomplete_source_frames,
        "source_condition_coverage_complete": not incomplete_source_frames,
        "hdr_white_pass": hdr_white_pass,
        "hdr_aggregate_pass": hdr_aggregate_pass,
        "regime_pass": regime_pass,
        "accepted": (
            not missing_regimes and not missing_whites and not unexpected_whites
            and not incomplete_source_frames and all(regime_pass.values())
        ),
        "regimes": regimes,
        "hdr_by_white_level_raw": hdr_by_white,
    }


def policy_decision(validation, by_clip, by_domain, by_film, val_clips,
                    val_domains, val_films, minimum_films=2,
                    require_identity_guard=True, unsafe_overshoot=None,
                    require_unsafe_overshoot_guard=False):
    aggregate_wins = {
        key: (validation["trained"][key] is not None
              and validation["neutral"][key] is not None
              and validation["trained"][key] < validation["neutral"][key])
        for key, _, _ in METRICS
    }
    eligible_clips = {
        key: [clip for clip in val_clips
              if by_clip[clip]["trained"][key] is not None
              and by_clip[clip]["neutral"][key] is not None]
        for key, _, _ in METRICS
    }
    sequence_wins = {
        key: sum(
            by_clip[clip]["trained"][key] < by_clip[clip]["neutral"][key]
            for clip in eligible_clips[key]
        )
        for key, _, _ in METRICS
    }
    required_sequence_wins = {
        key: len(eligible_clips[key]) // 2 + 1
        for key, _, _ in METRICS
    }
    eligible_domains = {
        key: [domain for domain in val_domains
              if by_domain[domain]["trained"][key] is not None
              and by_domain[domain]["neutral"][key] is not None]
        for key, _, _ in METRICS
    }
    domain_wins = {
        key: sum(
            by_domain[domain]["trained"][key]
            < by_domain[domain]["neutral"][key]
            for domain in eligible_domains[key]
        )
        for key, _, _ in METRICS
    }
    required_domain_wins = {
        key: len(eligible_domains[key]) // 2 + 1
        for key, _, _ in METRICS
    }
    eligible_films = {
        key: [film for film in val_films
              if by_film[film]["trained"][key] is not None
              and by_film[film]["neutral"][key] is not None]
        for key, _, _ in METRICS
    }
    film_wins = {
        key: sum(
            by_film[film]["trained"][key]
            < by_film[film]["neutral"][key]
            for film in eligible_films[key]
        )
        for key, _, _ in METRICS
    }
    required_film_wins = {
        key: len(eligible_films[key]) // 2 + 1
        for key, _, _ in METRICS
    }
    minimum_film_count = {key: minimum_films for key, _, _ in METRICS}
    identity_false_action = validation["trained"].get(
        "identity_false_action_pct"
    )
    guards = {
        "identity_examples_present": (
            identity_false_action is not None or not require_identity_guard
        ),
        "identity_false_action_pct": (
            (identity_false_action is None and not require_identity_guard) or
            (identity_false_action is not None and
             identity_false_action <= MAX_IDENTITY_FALSE_ACTION_PCT)
        ),
        "unsafe_ceiling_maximum": (
            not require_unsafe_overshoot_guard or
            (isinstance(unsafe_overshoot, dict) and
             unsafe_overshoot.get("maximum_pass") is True)
        ),
        "unsafe_ceiling_film_balanced_mean": (
            not require_unsafe_overshoot_guard or
            (isinstance(unsafe_overshoot, dict) and
             unsafe_overshoot.get("film_balanced_mean_pass") is True)
        ),
    }
    accepted = (all(aggregate_wins.values())
                and all(guards.values())
                and all(eligible_clips[key]
                        and sequence_wins[key] >= required_sequence_wins[key]
                        for key, _, _ in METRICS)
                and all(eligible_domains[key]
                        and domain_wins[key] >= required_domain_wins[key]
                        for key, _, _ in METRICS)
                and all(len(eligible_films[key]) >= minimum_film_count[key]
                        and film_wins[key] >= required_film_wins[key]
                        for key, _, _ in METRICS))
    return {
        "accepted": accepted,
        "aggregate_wins": aggregate_wins,
        "sequence_wins": sequence_wins,
        "sequence_count": {key: len(value)
                           for key, value in eligible_clips.items()},
        "required_sequence_wins": required_sequence_wins,
        "domain_wins": domain_wins,
        "domain_count": {key: len(value)
                         for key, value in eligible_domains.items()},
        "required_domain_wins": required_domain_wins,
        "film_wins": film_wins,
        "film_count": {key: len(value)
                       for key, value in eligible_films.items()},
        "required_film_wins": required_film_wins,
        "minimum_film_count": minimum_film_count,
        "guards": guards,
        "identity_false_action_limit_pct": MAX_IDENTITY_FALSE_ACTION_PCT,
        "identity_guard_required": require_identity_guard,
        "unsafe_overshoot_guard_required": require_unsafe_overshoot_guard,
        "unsafe_ceiling_overshoot": unsafe_overshoot,
        "rule": (
            "lower shot-first actionable errors on aggregate and strict majorities "
            f"of held-out sequences, at least {minimum_films} film(s), and domains, "
            f"with identity false actions <= {MAX_IDENTITY_FALSE_ACTION_PCT:.1f}%, "
            f"maximum unsafe ceiling overshoot <= "
            f"{MAX_UNSAFE_CEILING_OVERSHOOT_SCALE:.2f}, and film-balanced mean "
            f"unsafe overshoot <= "
            f"{MAX_FILM_BALANCED_UNSAFE_CEILING_OVERSHOOT_SCALE:.2f}"
        ),
    }


def _format_metric(value, digits=3, suffix=""):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}{suffix}"


def _runtime_group_html(title, group):
    primary = group["primary"]
    decision = group["decision"]
    status = "PASS" if decision["accepted"] else "FAIL"
    primary_rows = (
        ("Variant samples", primary["variant_sample_count"], ""),
        ("Unique RGB samples", primary["unique_rgb_sample_count"], ""),
        ("Shots", primary["shot_count"], ""),
        ("Shot-condition decisions", primary["shot_condition_count"], ""),
        ("Films", primary["film_count"], ""),
        ("Actionable / identity samples", (
            f"{primary['actionable_variant_sample_count']} / "
            f"{primary['identity_variant_sample_count']}"
        ), ""),
        ("Actionable / identity shot-conditions", (
            f"{primary['actionable_shot_condition_count']} / "
            f"{primary['identity_shot_condition_count']}"
        ), ""),
        ("Predicted / target scale", (
            f"{primary['predicted_scale_mean']:.4f} / "
            f"{primary['target_scale_mean']:.4f}"
        ), ""),
        ("Predicted / target confidence", (
            f"{primary['predicted_confidence_mean']:.4f} / "
            f"{primary['target_confidence_mean']:.4f}"
        ), ""),
        ("Action Brier", primary["action_brier"], ""),
        ("Action calibration ECE", primary["action_ece"] * 100.0, "%"),
        ("Identity false action", primary["identity_false_action_pct"], "%"),
        ("Rendered disparity / target", (
            f"{primary['rendered_disparity_mean_abs_pct']:.4f}% / "
            f"{primary['target_rendered_disparity_mean_abs_pct']:.4f}%"
        ), ""),
        ("Rendered-disparity MAE", primary["rendered_disparity_mae_pct"], "%"),
        ("Maximum unsafe ceiling overshoot", (
            primary["maximum_unsafe_ceiling_overshoot_scale"] * 100.0
        ), " pp"),
        ("Film-balanced unsafe overshoot", (
            primary["film_balanced_mean_unsafe_ceiling_overshoot_scale"] * 100.0
        ), " pp"),
        ("Film-balanced overshoot rate", (
            primary["film_balanced_unsafe_ceiling_overshoot_rate_pct"]
        ), "%"),
    )
    cells = []
    for label, value, suffix in primary_rows:
        display = (html.escape(value) if isinstance(value, str) else
                   _format_metric(value, suffix=suffix))
        cells.append(
            f"<tr><th>{html.escape(label)}</th><td>{display}</td></tr>"
        )
    comparisons = []
    for key, label, unit in ALL_METRICS:
        trained = group["evaluation"]["trained"][key]
        neutral = group["evaluation"]["neutral"][key]
        comparisons.append(
            f"<tr><th>{html.escape(label)}</th>"
            f"<td>{_format_metric(neutral)}</td>"
            f"<td>{_format_metric(trained)}</td>"
            f"<td>{html.escape(unit)}</td></tr>"
        )
    return f"""
      <section class="regime"><h2>{html.escape(title)} <span class="{status.lower()}">{status}</span></h2>
        <div class="regime-grid">
          <table><tr><th colspan="2">Primary outputs and safety</th></tr>{''.join(cells)}</table>
          <table><tr><th>Metric</th><th>Neutral</th><th>Trained</th><th>Unit</th></tr>
            {''.join(comparisons)}</table>
        </div>
      </section>"""


def write_html(path: Path, payload):
    decision_payload = payload["decision"]
    split = payload["split"]
    if split == "test":
        decision = ("Candidate passes sealed label screen"
                    if decision_payload["accepted"] else "REJECT checkpoint")
    else:
        decision = ("Development screen improves on neutral"
                    if decision_payload["accepted"] else
                    "Development screen does not beat neutral")
    aggregate_wins = sum(decision_payload["aggregate_wins"].values())
    calibration = payload["film_balanced_acceptance"]["trained"]["macro"]
    identity_false_action = payload["evaluation"]["trained"].get(
        "identity_false_action_pct"
    )
    identity_false_action_text = (
        f"{identity_false_action:.1f}%" if identity_false_action is not None else "n/a"
    )
    overshoot = payload["unsafe_ceiling_overshoot"]
    runtime = payload["runtime_regime_evaluation"]
    failed_regimes = [
        regime.upper() for regime, passed in runtime["regime_pass"].items()
        if not passed
    ]
    failed_whites = [
        white for white, passed in runtime["hdr_white_pass"].items()
        if not passed
    ]
    if decision_payload["accepted"]:
        regime_conclusion = (
            "Native SDR, coherent worst-risk aggregate HDR, and every expected "
            "HDR white anchor pass against their own authenticated condition "
            "targets; no runtime condition is hidden by an overall average."
        )
    else:
        reasons = []
        if runtime["missing_regimes"]:
            reasons.append("missing regimes: " + ", ".join(runtime["missing_regimes"]))
        if runtime["missing_hdr_white_levels_raw"]:
            reasons.append(
                "missing HDR white anchors: " + ", ".join(
                    str(value) for value in runtime["missing_hdr_white_levels_raw"]
                )
            )
        if runtime["unexpected_hdr_white_levels_raw"]:
            reasons.append(
                "unexpected HDR white anchors: " + ", ".join(
                    str(value)
                    for value in runtime["unexpected_hdr_white_levels_raw"]
                )
            )
        if runtime["incomplete_source_frame_count"]:
            reasons.append(
                f"{runtime['incomplete_source_frame_count']} source frame(s) "
                "lack the complete SDR/HDR condition set"
            )
        if failed_whites:
            reasons.append("failed HDR whites: " + ", ".join(failed_whites))
        if failed_regimes:
            reasons.append("failed regime decisions: " + ", ".join(failed_regimes))
        if not decision_payload["overall_diagnostic_accepted"]:
            reasons.append("overall conservative decision failed")
        regime_conclusion = "Reject: " + "; ".join(reasons) + "."
    minimum_films = min(
        decision_payload["minimum_film_count"].values(), default=0
    )
    sequence_summary = ", ".join(
        f"{key.removesuffix('_mae_pct')} "
        f"{decision_payload['sequence_wins'][key]}/"
        f"{decision_payload['sequence_count'][key]}"
        for key, _, _ in METRICS
    )
    film_summary = ", ".join(
        f"{key.removesuffix('_mae_pct')} "
        f"{decision_payload['film_wins'][key]}/"
        f"{decision_payload['film_count'][key]}"
        for key, _, _ in METRICS
    )
    cards = f"""
      <div class="card"><small>Decision</small><strong>{decision}</strong></div>
      <div class="card"><small>Held-out clips</small><strong>{len(payload['val_clips'])}</strong></div>
      <div class="card"><small>Held-out films</small><strong>{len(payload['val_films'])}</strong></div>
      <div class="card"><small>Aggregate axes won</small><strong>{aggregate_wins}/{len(METRICS)}</strong></div>
      <div class="card"><small>Action calibration ECE</small>
        <strong>{calibration['action_ece'] * 100.0:.2f}%</strong></div>
      <div class="card"><small>Identity false actions</small>
        <strong>{identity_false_action_text}</strong></div>
      <div class="card"><small>Maximum unsafe overshoot</small>
        <strong>{overshoot['maximum_scale'] * 100.0:.2f} pp</strong></div>
      <div class="card"><small>Film-balanced mean unsafe overshoot</small>
        <strong>{overshoot['film_balanced_mean_scale'] * 100.0:.2f} pp</strong></div>
      <div class="card"><small>Film-balanced overshoot rate</small>
        <strong>{overshoot['film_balanced_overshoot_rate_pct']:.2f}%</strong></div>
      <div class="card"><small>Native SDR</small>
        <strong>{'PASS' if runtime['regime_pass']['sdr'] else 'FAIL'}</strong></div>
      <div class="card"><small>HDR aggregate</small>
        <strong>{'PASS' if runtime['regime_pass']['hdr'] else 'FAIL'}</strong></div>
    """
    agg = payload["evaluation"]
    rows = []
    for key, label, unit in ALL_METRICS:
        neutral = agg["neutral"][key]
        trained = agg["trained"][key]
        if neutral is None or trained is None:
            rows.append(f"""
              <div class="metric"><h3>{html.escape(label)}</h3>
                <p>No eligible labels in this split.</p></div>""")
            continue
        maximum = max(neutral, trained, 1e-6)
        rows.append(f"""
          <div class="metric"><h3>{html.escape(label)}</h3>
            <p>{html.escape(unit)}; lower is better</p>
            <div class="bar neutral" style="width:{neutral / maximum * 100:.1f}%">neutral {neutral:.3f}</div>
            <div class="bar trained" style="width:{trained / maximum * 100:.1f}%">trained {trained:.3f}</div>
          </div>""")
    section = f"<section><h2>{split.title()}</h2>{''.join(rows)}</section>"
    regime_sections = []
    for regime, title in ((
            "sdr", "Native SDR"),
            ("hdr", "HDR aggregate (coherent worst-risk white per RGB)")):
        group = runtime["regimes"].get(regime)
        if group is not None:
            regime_sections.append(_runtime_group_html(title, group))
    for white, group in runtime["hdr_by_white_level_raw"].items():
        regime_sections.append(_runtime_group_html(
            f"HDR white level {white} ({int(white) * 80.0 / 1000.0:g} nits)",
            group,
        ))
    clip_rows = []
    for clip, values in payload["by_clip"].items():
        held = "held out" if clip in payload["val_clips"] else "trained"
        cells = []
        for key, _, _ in ALL_METRICS:
            neutral = values["neutral"][key]
            trained = values["trained"][key]
            cells.append(
                "<td>not eligible</td>" if neutral is None or trained is None else
                f"<td>{neutral:.3f} &rarr; {trained:.3f}</td>"
            )
        clip_rows.append(
            f"<tr><th>{html.escape(clip)}<small>{held}</small></th>{''.join(cells)}</tr>"
        )
    document = f"""<!doctype html><meta charset="utf-8"><title>DA-V2 artistic policy</title>
<style>
body{{font:15px system-ui;background:#101116;color:#ececf2;margin:0;padding:32px;max-width:1050px}}
h1{{margin:0 0 8px}} h2{{margin-top:36px}} p{{color:#b8bac6}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:24px 0}}
.card,section{{background:#191b23;border:1px solid #2b2e3a;border-radius:14px;padding:18px}}
.card small,.card strong{{display:block}} .card strong{{font-size:22px;margin-top:8px}}
.metric{{margin:18px 0}} .metric h3,.metric p{{margin:4px 0}}
.bar{{box-sizing:border-box;min-width:150px;padding:7px 10px;margin:5px 0;border-radius:6px;white-space:nowrap}}
.neutral{{background:#555b6d}} .trained{{background:#6c5ce7}}
table{{width:100%;border-collapse:collapse;background:#191b23}} th,td{{padding:10px;border-bottom:1px solid #30333e;text-align:left}}
th small{{display:block;color:#999;font-weight:400}}
.regime-grid{{display:grid;grid-template-columns:minmax(300px,1fr) minmax(420px,1.6fr);gap:18px}}
.pass{{color:#50d890;font-size:14px}} .fail{{color:#ff6b6b;font-size:14px}}
@media(max-width:850px){{.regime-grid{{grid-template-columns:1fr}}.cards{{grid-template-columns:1fr}}}}
</style>
<h1>DA-V2 artistic policy evaluation</h1>
<p><strong>SDR vs HDR conclusion:</strong> {html.escape(regime_conclusion)}</p>
<p><strong>Target contract:</strong> {html.escape(payload['condition_target_contract'])}.
Each authenticated input condition owns its two-geometry safety target; runtime mode and
white-level metadata are used only for evaluation grouping, never as a model input.</p>
<p>The checkpoint is compared with Apollo's identity policy at the fixed runtime action
threshold of 0.5. Each clip is decided by its earliest labelled frame, matching the runtime
shot latch. Complete films are sealed; domains and clips receive equal aggregate weight.</p>
<p>Held-out sequence wins: {html.escape(sequence_summary)}. Held-out film wins:
{html.escape(film_summary)}. Acceptance requires lower aggregate error and strict majorities of
held-out sequences, at least {minimum_films} film(s), and domains on every axis. Sealed-test
acceptance also caps one-sided runtime-effective ceiling overshoot at 5 percentage points per
shot and 1 percentage point as a film-balanced mean.</p>
<div class="cards">{cards}</div>{''.join(regime_sections)}
<h2>Overall conservative diagnostic</h2>{section}
<h2>Per clip: neutral &rarr; trained</h2>
<table><tr><th>Clip</th>{''.join(f'<th>{label}</th>' for _, label, _ in ALL_METRICS)}</tr>
{''.join(clip_rows)}</table>
"""
    path.write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, type=Path, nargs="+")
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--depth-anything-root", required=True, type=Path)
    parser.add_argument("--depth-weights", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("development", "test"),
                        default="test")
    args = parser.parse_args()

    rows = load_rows(args.labels, validate=True)
    active_split, active_split_hash = load_active_split(args.split_manifest)
    validate_rows_against_active_split(rows, active_split, {args.split})
    val_clips = {row["clip"] for row in rows if row["split"] == args.split}
    if not val_clips:
        raise RuntimeError(f"evaluation requires a non-empty {args.split} split")
    dataset = PolicyDataset(rows)
    model = ArtisticPolicyModel(load_depth_anything_small(
        args.depth_anything_root, args.depth_weights
    ))
    checkpoint_bytes = args.policy.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint_payload = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False
    )
    checkpoint = load_policy_state(model, args.policy, checkpoint_payload)
    if args.split == "test":
        expected_test = set(checkpoint.get("sealed_test_productions", ()))
        actual_test = {row["film_id"] for row in rows}
        if expected_test != actual_test:
            raise RuntimeError(
                "evaluation test split does not match the checkpoint contract: "
                f"{sorted(actual_test)} != {sorted(expected_test)}"
            )
    if checkpoint.get("active_split_sha256") != active_split_hash:
        raise RuntimeError("checkpoint was trained with a different active split")
    label_sources, labels_digest = labels_contract(args.labels)
    if (checkpoint.get("label_fitter_identity_sha256") !=
            label_sources[0]["label_fitter_identity_sha256"]):
        raise RuntimeError("test labels use a different label fitter contract")
    if checkpoint.get("metric_sha256") != label_sources[0]["metric_sha256"]:
        raise RuntimeError("test labels use a different metric implementation")
    geometry_allowlist = label_sources[0]["deployment_geometry_allowlist"]
    validate_allowlist(geometry_allowlist)
    geometry_hash = allowlist_sha256(geometry_allowlist)
    if (checkpoint.get("deployment_geometry_allowlist_sha256") != geometry_hash or
            checkpoint.get("deployment_geometry_allowlist") != geometry_allowlist):
        raise RuntimeError(
            "test labels use a different deployment geometry allow-list"
        )
    input_manifest = label_sources[0]["input_variant_manifest"]
    input_manifest_hash = label_sources[0]["input_variant_manifest_sha256"]
    if (checkpoint.get("input_variant_manifest") != input_manifest or
            checkpoint.get("input_variant_manifest_sha256") !=
            input_manifest_hash or
            checkpoint.get("depth_input_color_contract_sha256") !=
            label_sources[0]["depth_input_color_contract_sha256"]):
        raise RuntimeError("test labels use a different input color contract")
    condition_target_contract = label_sources[0].get(
        "condition_target_contract"
    )
    if (condition_target_contract != label_merge.CONDITION_TARGET_CONTRACT or
            checkpoint.get("condition_target_contract") !=
            condition_target_contract):
        raise RuntimeError(
            "test labels or checkpoint use a stale shared-target contract"
        )
    if checkpoint.get("depth_weights_sha256") != sha256(args.depth_weights):
        raise RuntimeError("checkpoint depth-weight provenance does not match")
    use_dynamic_onnx_position_encoding(model)
    model.freeze_base()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    measured = [measure(model, row, dataset[index], device)
                for index, row in enumerate(dataset.rows)]
    all_val_rows = [row for row in measured if row["clip"] in val_clips]
    if not all_val_rows:
        raise RuntimeError(f"need a non-empty {args.split} split")
    minimum_films = 2 if args.split == "test" else 1
    require_sealed_guards = args.split == "test"
    overall = evaluate_measurement_group(
        all_val_rows, minimum_films=minimum_films,
        require_identity_guard=require_sealed_guards,
        require_unsafe_overshoot_guard=require_sealed_guards,
    )
    expected_hdr_whites = sorted(
        int(variant["windows_sdr_white_level_raw"])
        for variant in input_manifest["variants"]
        if variant["kind"] == "simulated-sdr-in-windows-hdr"
    )
    regime_evaluation = evaluate_runtime_regimes(
        all_val_rows, expected_hdr_whites, minimum_films=minimum_films,
        require_identity_guard=require_sealed_guards,
        require_unsafe_overshoot_guard=require_sealed_guards,
    )
    decision = dict(overall["decision"])
    decision["overall_diagnostic_accepted"] = decision["accepted"]
    decision["runtime_regime_acceptance"] = {
        key: regime_evaluation[key]
        for key in (
            "contract", "condition_target_contract",
            "hdr_aggregation_contract",
            "required_regimes", "expected_hdr_white_levels_raw",
            "missing_regimes", "missing_hdr_white_levels_raw",
            "unexpected_hdr_white_levels_raw",
            "incomplete_source_frame_count",
            "source_condition_coverage_complete", "hdr_white_pass",
            "hdr_aggregate_pass", "regime_pass", "accepted",
        )
    }
    decision["guards"] = dict(decision["guards"])
    decision["guards"]["runtime_regimes_present"] = not regime_evaluation[
        "missing_regimes"
    ]
    decision["guards"]["hdr_white_levels_present"] = not regime_evaluation[
        "missing_hdr_white_levels_raw"
    ]
    decision["guards"]["hdr_white_levels_accepted"] = all(
        regime_evaluation["hdr_white_pass"].values()
    )
    decision["guards"]["no_unexpected_hdr_white_levels"] = not (
        regime_evaluation["unexpected_hdr_white_levels_raw"]
    )
    decision["guards"]["source_condition_coverage_complete"] = (
        regime_evaluation["source_condition_coverage_complete"]
    )
    decision["guards"]["condition_target_contract"] = (
        condition_target_contract == label_merge.CONDITION_TARGET_CONTRACT
    )
    decision["guards"]["sdr_and_hdr_accepted"] = regime_evaluation["accepted"]
    decision["accepted"] = (
        decision["overall_diagnostic_accepted"] and regime_evaluation["accepted"]
    )
    decision["rule"] += (
        "; native SDR, coherent aggregate HDR, and every authenticated HDR "
        "white anchor must each pass against its own condition target"
    )
    val_films = sorted(overall["validation_by_film"])
    payload = {
        "schema": EVALUATION_SCHEMA,
        "split": args.split,
        "checkpoint_sha256": checkpoint_sha256,
        "output_semantics": checkpoint.get("output_semantics"),
        "active_split_sha256": active_split_hash,
        "test_labels": label_sources,
        "test_labels_sha256": labels_digest,
        "metric_sha256": label_sources[0]["metric_sha256"],
        "deployment_geometry_allowlist": geometry_allowlist,
        "deployment_geometry_allowlist_sha256": geometry_hash,
        "input_variant_manifest": input_manifest,
        "input_variant_manifest_sha256": input_manifest_hash,
        "depth_input_color_contract_sha256": label_sources[0][
            "depth_input_color_contract_sha256"
        ],
        "condition_target_contract": condition_target_contract,
        "label_fitter_identity_sha256": label_sources[0][
            "label_fitter_identity_sha256"
        ],
        "val_clips": sorted(val_clips),
        "val_films": val_films,
        "evaluation": overall["evaluation"],
        "unsafe_ceiling_overshoot": overall["unsafe_ceiling_overshoot"],
        "runtime_action_threshold": 0.5,
        "decision_sampling": "earliest available labelled frame per complete shot",
        "film_balanced_acceptance": overall["film_balanced_acceptance"],
        "by_clip": overall["by_clip"],
        "validation_by_domain": overall["validation_by_domain"],
        "validation_by_film": overall["validation_by_film"],
        "runtime_regime_evaluation": regime_evaluation,
        "decision": decision,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_html(args.output / "report.html", payload)
    print(json.dumps({
        args.split: payload["evaluation"],
        "decision": payload["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
