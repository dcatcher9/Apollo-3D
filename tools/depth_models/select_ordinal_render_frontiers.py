#!/usr/bin/env python3
"""Build experimental ordinal safe-scale frontiers from rendered metrics.

This module deliberately does not replace ``select_render_feasible_labels``.
The shipping/scalar label path continues to emit schema-8 ceilings.  This
experimental path reuses the same fail-closed feasibility rules, but requires
the 1.00..1.50/0.02 ordinal lattice and preserves the first unsafe sample as an
interval boundary instead of pretending that the last safe sample is an exact
ceiling.

Authentication of render runs and extraction of directional worst-frame
metrics remain the caller's responsibility.  In particular, callers should
pass aggregates after ``project_protected_worst_metrics`` has been applied.
Keeping this first slice pure makes it testable before the expensive full-frame
render orchestration is changed.
"""

from __future__ import annotations

import math

import artistic_policy_ordinal_contract as ordinal_contract
import select_render_feasible_labels as scalar_selector


EXACT_POP_METRIC = scalar_selector.EXACT_POP_METRIC


def ordinal_feasibility_violations(control, candidate, metric_specs,
                                   clip_meta=None):
    """Return scalar protections plus explicit ordinal-only absolute limits.

    ``trigger`` and ``trigger_min`` remain report diagnostics. They are not
    approved safety policy and intentionally do not participate here.
    """
    clip_meta = clip_meta or {}
    effective_specs = metric_specs
    if clip_meta.get("temporal_boundary_exemption") is True:
        # Per-frame temporal metrics legitimately have no predecessor at scene
        # starts/cuts. Exempt only those whose identity evidence is actually
        # null; all finite temporal evidence remains testable.
        effective_specs = {
            metric: spec for metric, spec in metric_specs.items()
            if not ("min_frames" in spec and control.get(metric) is None)
        }
    violations = scalar_selector.feasibility_violations(
        control, candidate, effective_specs, clip_meta
    )
    for metric, spec in effective_specs.items():
        if ("ordinal_hard_min" not in spec and
                "ordinal_hard_max" not in spec):
            continue
        value = candidate.get(metric)
        if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                not math.isfinite(float(value))):
            violations.append(metric + ":missing-ordinal-bound")
            continue
        value = float(value)
        if ("ordinal_hard_min" in spec and
                value < float(spec["ordinal_hard_min"])):
            violations.append(metric + ":ordinal-hard-min")
        if ("ordinal_hard_max" in spec and
                value > float(spec["ordinal_hard_max"])):
            violations.append(metric + ":ordinal-hard-max")
    return sorted(set(violations))


def _ordered_candidate_prefix(candidates):
    """Return an exact identity-anchored prefix of the ordinal scale lattice."""
    if not isinstance(candidates, dict) or not candidates:
        raise RuntimeError("ordinal candidate evidence is empty")
    normalized = {}
    for raw_scale, metrics in candidates.items():
        if (not isinstance(raw_scale, (int, float)) or
                isinstance(raw_scale, bool) or
                not math.isfinite(float(raw_scale))):
            raise RuntimeError("ordinal candidate scale is not finite")
        index = ordinal_contract.scale_index(raw_scale)
        scale = ordinal_contract.SCALES[index]
        if scale in normalized:
            raise RuntimeError(f"duplicate ordinal candidate scale: {scale}")
        if not isinstance(metrics, dict):
            raise RuntimeError(
                f"ordinal candidate metrics are not an object: {scale}"
            )
        normalized[scale] = metrics
    ordered = sorted(normalized.items())
    expected = list(ordinal_contract.SCALES[:len(ordered)])
    if [scale for scale, _metrics in ordered] != expected:
        raise RuntimeError(
            "ordinal candidate scales must be contiguous from identity in "
            "0.02 steps"
        )
    return ordered


def select_clip_frontier(control_aggregate, candidates, metric_specs,
                         clip_meta=None):
    """Classify a rendered scale prefix and return canonical ordinal evidence.

    A caller may provide all 26 rendered candidates or stop after the first
    unsafe candidate.  A safe prefix that stops before 1.50 is rejected because
    it proves neither an upper interval boundary nor right-censoring.

    Candidates after the first unsafe point are intentionally ignored.  Their
    labels remain ``unknown``: the learned quantity is the identity-connected
    maximum-safe frontier, not independent safety at disconnected scales.
    """
    if not isinstance(control_aggregate, dict):
        raise RuntimeError("ordinal control aggregate is not an object")
    ordered = _ordered_candidate_prefix(candidates)
    clip_meta = clip_meta or {}
    tested_bins = []
    for index, (scale, aggregate) in enumerate(ordered):
        violations = ordinal_feasibility_violations(
            control_aggregate, aggregate, metric_specs, clip_meta
        )
        pop = aggregate.get(EXACT_POP_METRIC)
        if (not isinstance(pop, (int, float)) or isinstance(pop, bool) or
                not math.isfinite(float(pop))):
            raise RuntimeError(
                f"scale {scale:.2f} render has no finite {EXACT_POP_METRIC}"
            )

        # At identity, only a fully measured absolute hard-bound failure is a
        # valid left-censored target. Missing evidence and relative mismatches
        # indicate an evaluator inconsistency, just as in the scalar selector.
        if index == 0 and violations and not all(
                violation.endswith((
                    ":hard", ":ordinal-hard-min", ":ordinal-hard-max",
                )) for violation in violations):
            raise RuntimeError(
                "identity render has incomplete or inconsistent ordinal "
                "feasibility evidence: " + ", ".join(sorted(violations))
            )
        safe = not violations
        tested_bins.append({
            "scale": scale,
            "safe": safe,
            "realized_pop_pct": float(pop),
            "failure_causes": sorted(set(violations)),
        })
        if not safe:
            break
    return ordinal_contract.build_frontier_evidence(tested_bins)
