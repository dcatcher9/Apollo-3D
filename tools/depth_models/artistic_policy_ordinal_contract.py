"""Dependency-free ordinal maximum-safe scale contract for Apollo experiments."""

from __future__ import annotations

import math


FRONTIER_SCHEMA = 1
FRONTIER_CONTRACT = "apollo-ordinal-safe-scale-frontier-v1"
SCALE_MIN_HUNDREDTHS = 100
SCALE_MAX_HUNDREDTHS = 150
SCALE_STEP_HUNDREDTHS = 2
SCALE_HUNDREDTHS = tuple(range(
    SCALE_MIN_HUNDREDTHS,
    SCALE_MAX_HUNDREDTHS + 1,
    SCALE_STEP_HUNDREDTHS,
))
SCALES = tuple(value / 100.0 for value in SCALE_HUNDREDTHS)
FRONTIER_SIZE = len(SCALES)
STATES = frozenset({"safe", "unsafe", "unknown"})


def scale_index(scale):
    """Return the exact ordinal-bin index for ``scale`` or fail closed."""
    if (not isinstance(scale, (int, float)) or isinstance(scale, bool) or
            not math.isfinite(float(scale))):
        raise RuntimeError("frontier scale is not finite")
    hundredths = int(round(float(scale) * 100.0))
    if abs(float(scale) - hundredths / 100.0) > 1e-8:
        raise RuntimeError("frontier scale is not on the 0.01 lattice")
    offset = hundredths - SCALE_MIN_HUNDREDTHS
    if (offset < 0 or hundredths > SCALE_MAX_HUNDREDTHS or
            offset % SCALE_STEP_HUNDREDTHS):
        raise RuntimeError(
            "frontier scale is outside the 1.00..1.50/0.02 contract"
        )
    return offset // SCALE_STEP_HUNDREDTHS


def _finite_number(value, description):
    if (not isinstance(value, (int, float)) or isinstance(value, bool) or
            not math.isfinite(float(value))):
        raise RuntimeError(f"{description} is not finite")
    return float(value)


def _build_frontier_evidence(tested_bins, validate_result):
    """Build one conservative interval/censoring target from sequential tests."""
    if not isinstance(tested_bins, (list, tuple)) or not tested_bins:
        raise RuntimeError("frontier evidence has no tested bins")
    states = ["unknown"] * FRONTIER_SIZE
    realized_pop = [None] * FRONTIER_SIZE
    previous_safe_pop = None
    highest_safe = None
    first_unsafe = None
    first_failure_causes = []
    normalized_tests = []
    for expected_index, item in enumerate(tested_bins):
        if not isinstance(item, dict):
            raise RuntimeError("frontier tested bin is not an object")
        index = scale_index(item.get("scale"))
        if index != expected_index:
            raise RuntimeError(
                "frontier tests must be contiguous from identity in 0.02 steps"
            )
        safe = item.get("safe")
        if not isinstance(safe, bool):
            raise RuntimeError(
                "frontier tested bin lacks a boolean safety result"
            )
        pop = _finite_number(item.get("realized_pop_pct"), "realized pop")
        if pop < 0.0:
            raise RuntimeError("realized pop cannot be negative")
        causes = item.get("failure_causes", [])
        if (not isinstance(causes, list) or
                any(not isinstance(value, str) or not value for value in causes) or
                len(causes) != len(set(causes))):
            raise RuntimeError("frontier failure causes are invalid")
        if safe and causes:
            raise RuntimeError("a safe frontier bin cannot carry failure causes")
        if not safe and not causes:
            raise RuntimeError(
                "an unsafe frontier bin requires measured failure causes"
            )
        if first_unsafe is not None:
            raise RuntimeError(
                "frontier evidence continues after the first unsafe bin"
            )
        scale = SCALES[index]
        states[index] = "safe" if safe else "unsafe"
        realized_pop[index] = pop
        normalized_tests.append({
            "scale": scale,
            "safe": safe,
            "realized_pop_pct": pop,
            "failure_causes": sorted(causes),
        })
        if safe:
            if (previous_safe_pop is not None and
                    pop + 1e-6 < previous_safe_pop):
                raise RuntimeError(
                    "realized pop materially decreases along the safe frontier"
                )
            previous_safe_pop = pop
            highest_safe = scale
        else:
            first_unsafe = scale
            first_failure_causes = sorted(causes)

    reached_maximum = len(tested_bins) == FRONTIER_SIZE
    right_censored = first_unsafe is None and reached_maximum
    if first_unsafe is None and not right_censored:
        raise RuntimeError(
            "frontier evidence stops without an unsafe bin or a proven 1.50 endpoint"
        )
    identity_feasible = states[0] == "safe"
    left_censored = not identity_feasible
    if left_censored:
        if first_unsafe != SCALES[0] or highest_safe is not None:
            raise RuntimeError("identity failure has inconsistent frontier bounds")
    elif highest_safe is None:
        raise RuntimeError("identity-feasible frontier has no proven safe bin")

    safe_pop = [value for state, value in zip(states, realized_pop)
                if state == "safe"]
    identity_pop = safe_pop[0] if safe_pop else None
    maximum_pop = max(safe_pop) if safe_pop else None
    first_maximum_scale = None
    if maximum_pop is not None:
        first_maximum_scale = next(
            scale for scale, state, value in zip(SCALES, states, realized_pop)
            if state == "safe" and value >= maximum_pop - 1e-6
        )
    payload = {
        "schema": FRONTIER_SCHEMA,
        "contract": FRONTIER_CONTRACT,
        "scale_thresholds": list(SCALES),
        "states": states,
        "realized_pop_pct": realized_pop,
        "tested_bins": normalized_tests,
        "highest_proven_safe_scale": highest_safe,
        "first_proven_unsafe_scale": first_unsafe,
        "left_censored": left_censored,
        "right_censored": right_censored,
        "identity_feasible": identity_feasible,
        "first_unsafe_failure_causes": first_failure_causes,
        "identity_realized_pop_pct": identity_pop,
        "maximum_safe_realized_pop_pct": maximum_pop,
        "first_maximum_pop_scale": first_maximum_scale,
        "realized_pop_gain_over_identity_pct": (
            maximum_pop - identity_pop
            if maximum_pop is not None and identity_pop is not None else None
        ),
    }
    if validate_result:
        validate_frontier_evidence(payload)
    return payload


def build_frontier_evidence(tested_bins):
    """Build a canonical tri-state frontier from contiguous rendered evidence.

    Testing must start at identity and advance one 0.02 bin at a time. It stops
    at the first unsafe bin, or after proving 1.50 safe. Bins beyond a failure
    remain unknown rather than being invented as negative labels.
    """
    return _build_frontier_evidence(tested_bins, True)


def validate_frontier_evidence(value):
    """Validate the canonical ordinal interval target without re-running renders."""
    if not isinstance(value, dict):
        raise RuntimeError("frontier evidence is not an object")
    required = {
        "schema", "contract", "scale_thresholds", "states",
        "realized_pop_pct", "tested_bins", "highest_proven_safe_scale",
        "first_proven_unsafe_scale", "left_censored", "right_censored",
        "identity_feasible", "first_unsafe_failure_causes",
        "identity_realized_pop_pct", "maximum_safe_realized_pop_pct",
        "first_maximum_pop_scale", "realized_pop_gain_over_identity_pct",
    }
    if set(value) != required:
        raise RuntimeError("frontier evidence fields are incomplete")
    if (value.get("schema") != FRONTIER_SCHEMA or
            value.get("contract") != FRONTIER_CONTRACT or
            value.get("scale_thresholds") != list(SCALES)):
        raise RuntimeError("frontier evidence contract or thresholds differ")
    states = value.get("states")
    pops = value.get("realized_pop_pct")
    if (not isinstance(states, list) or len(states) != FRONTIER_SIZE or
            any(state not in STATES for state in states) or
            not isinstance(pops, list) or len(pops) != FRONTIER_SIZE):
        raise RuntimeError("frontier state/pop vectors have the wrong shape")
    tested = value.get("tested_bins")
    if not isinstance(tested, list) or not tested:
        raise RuntimeError("frontier tested-bin evidence is empty")
    rebuilt = _build_frontier_evidence(tested, False)
    if value != rebuilt:
        raise RuntimeError("frontier evidence is not canonical")
    return value


def select_contiguous_safe_scale(probabilities, threshold):
    """Select the highest contiguous authorized scale, or abstain.

    ``probabilities`` are calibrated point probabilities unless the caller has
    explicitly supplied separately validated lower confidence bounds.  Identity
    is not an implicit safe fallback: when its probability misses the threshold,
    the selector returns ``None`` so the controller can abstain/hold identity.
    """
    if (not isinstance(probabilities, (list, tuple)) or
            len(probabilities) != FRONTIER_SIZE):
        raise RuntimeError("frontier probability vector has the wrong shape")
    threshold = _finite_number(threshold, "frontier probability threshold")
    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError("frontier probability threshold is outside [0,1]")
    selected = None
    for scale, probability in zip(SCALES, probabilities):
        probability = _finite_number(probability, "frontier probability")
        if not 0.0 <= probability <= 1.0:
            raise RuntimeError("frontier probability is outside [0,1]")
        if probability < threshold:
            break
        selected = scale
    return selected
