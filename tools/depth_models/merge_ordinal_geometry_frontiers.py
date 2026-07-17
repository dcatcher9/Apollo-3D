#!/usr/bin/env python3
"""Intersect two experimental ordinal frontiers for one input condition.

This is deliberately separate from the shipping schema-10 scalar merge.  It
combines only the two deployment geometries of one authenticated source/input
condition.  Different SDR/HDR input variants keep independent safety targets.

The component realized-pop curves are retained verbatim.  The merged gain at
each proven-safe scale is ``min(pop_g(scale) - pop_g(identity))`` across the two
geometries.  This avoids the invalid shortcut ``min(pop_g(scale)) -
min(pop_g(identity))``, whose two minima may come from different geometries.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

import artistic_geometry_contract as geometry_contract
import artistic_policy_ordinal_contract as ordinal_contract


GEOMETRY_FRONTIER_SCHEMA = 1
GEOMETRY_FRONTIER_CONTRACT = "apollo-ordinal-geometry-frontier-v1"
INTERSECTION_SCHEMA = 1
INTERSECTION_CONTRACT = "apollo-ordinal-two-geometry-intersection-v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(value):
    encoded = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_geometry_frontier(source_sha256, input_variant_sha256, geometry,
                            frontier):
    """Bind one canonical frontier to its source, condition, and geometry."""
    if not isinstance(source_sha256, str) or not SHA256.fullmatch(source_sha256):
        raise RuntimeError("ordinal geometry frontier has invalid source identity")
    if (not isinstance(input_variant_sha256, str) or
            not SHA256.fullmatch(input_variant_sha256)):
        raise RuntimeError(
            "ordinal geometry frontier has invalid input-condition identity"
        )
    geometry = geometry_contract.canonical_geometry_tuple(geometry)
    ordinal_contract.validate_frontier_evidence(frontier)
    result = {
        "schema": GEOMETRY_FRONTIER_SCHEMA,
        "contract": GEOMETRY_FRONTIER_CONTRACT,
        "source_sha256": source_sha256,
        "input_variant_sha256": input_variant_sha256,
        "deployment_geometry": geometry,
        "deployment_geometry_sha256": _canonical_sha256(geometry),
        "frontier": frontier,
    }
    return result


def validate_geometry_frontier(value):
    required = {
        "schema", "contract", "source_sha256", "input_variant_sha256",
        "deployment_geometry", "deployment_geometry_sha256", "frontier",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RuntimeError("ordinal geometry frontier fields are incomplete")
    rebuilt = build_geometry_frontier(
        value.get("source_sha256"), value.get("input_variant_sha256"),
        value.get("deployment_geometry"), value.get("frontier"),
    )
    if value != rebuilt:
        raise RuntimeError("ordinal geometry frontier is not canonical")
    return value


def _build_intersection(records, validate_result):
    if not isinstance(records, (list, tuple)) or len(records) != 2:
        raise RuntimeError(
            "ordinal safety intersection requires exactly two geometries"
        )
    records = [validate_geometry_frontier(record) for record in records]
    sources = {record["source_sha256"] for record in records}
    conditions = {record["input_variant_sha256"] for record in records}
    if len(sources) != 1:
        raise RuntimeError("ordinal geometry frontiers mix source identities")
    if len(conditions) != 1:
        raise RuntimeError("ordinal geometry frontiers mix input conditions")
    geometry_ids = {
        record["deployment_geometry_sha256"] for record in records
    }
    if len(geometry_ids) != 2:
        raise RuntimeError("ordinal safety intersection repeats a geometry")
    records = sorted(
        records, key=lambda record: record["deployment_geometry_sha256"]
    )

    states = []
    conservative_gains = []
    first_unsafe_index = None
    first_unsafe_failures = []
    for index, _scale in enumerate(ordinal_contract.SCALES):
        component_states = [
            record["frontier"]["states"][index] for record in records
        ]
        if first_unsafe_index is not None:
            states.append("unknown")
            conservative_gains.append(None)
            continue
        if "unsafe" in component_states:
            states.append("unsafe")
            conservative_gains.append(None)
            first_unsafe_index = index
            for record, state in zip(records, component_states):
                if state == "unsafe":
                    first_unsafe_failures.append({
                        "deployment_geometry_sha256": record[
                            "deployment_geometry_sha256"
                        ],
                        "failure_causes": list(
                            record["frontier"]["first_unsafe_failure_causes"]
                        ),
                    })
            continue
        if all(state == "safe" for state in component_states):
            states.append("safe")
            gains = []
            for record in records:
                pops = record["frontier"]["realized_pop_pct"]
                gains.append(float(pops[index]) - float(pops[0]))
            gain = min(gains)
            if not math.isfinite(gain) or gain < -1e-6:
                raise RuntimeError(
                    "ordinal geometry frontier has invalid safe pop gain"
                )
            conservative_gains.append(max(0.0, gain))
            continue
        # A canonical component can become unknown only after its own first
        # unsafe bin. The merged frontier would already have stopped there.
        raise RuntimeError(
            "ordinal geometry frontiers contain an unbounded evidence gap"
        )

    safe_indices = [
        index for index, state in enumerate(states) if state == "safe"
    ]
    highest_safe = (
        ordinal_contract.SCALES[safe_indices[-1]] if safe_indices else None
    )
    first_unsafe = (
        ordinal_contract.SCALES[first_unsafe_index]
        if first_unsafe_index is not None else None
    )
    right_censored = first_unsafe_index is None
    left_censored = first_unsafe_index == 0
    identity_feasible = states[0] == "safe"
    if not right_censored and not first_unsafe_failures:
        raise RuntimeError("ordinal geometry intersection lost failure evidence")
    if right_censored and not all(state == "safe" for state in states):
        raise RuntimeError("ordinal geometry intersection is not fully bounded")

    safe_gains = [
        gain for state, gain in zip(states, conservative_gains)
        if state == "safe"
    ]
    maximum_gain = max(safe_gains) if safe_gains else None
    first_maximum_gain_scale = None
    if maximum_gain is not None:
        first_maximum_gain_scale = next(
            scale for scale, state, gain in zip(
                ordinal_contract.SCALES, states, conservative_gains
            )
            if state == "safe" and gain >= maximum_gain - 1e-6
        )
    result = {
        "schema": INTERSECTION_SCHEMA,
        "contract": INTERSECTION_CONTRACT,
        "source_sha256": records[0]["source_sha256"],
        "input_variant_sha256": records[0]["input_variant_sha256"],
        "scale_thresholds": list(ordinal_contract.SCALES),
        "states": states,
        "highest_proven_safe_scale": highest_safe,
        "first_proven_unsafe_scale": first_unsafe,
        "left_censored": left_censored,
        "right_censored": right_censored,
        "identity_feasible": identity_feasible,
        "first_unsafe_failures": first_unsafe_failures,
        "conservative_safe_pop_gain_over_identity_pct": conservative_gains,
        "maximum_conservative_safe_pop_gain_pct": maximum_gain,
        "first_maximum_conservative_gain_scale": first_maximum_gain_scale,
        "geometry_frontiers": records,
    }
    if validate_result:
        validate_geometry_intersection(result)
    return result


def intersect_geometry_frontiers(records):
    """Return the canonical safety intersection of two geometry frontiers."""
    return _build_intersection(records, True)


def validate_geometry_intersection(value):
    required = {
        "schema", "contract", "source_sha256", "input_variant_sha256",
        "scale_thresholds", "states", "highest_proven_safe_scale",
        "first_proven_unsafe_scale", "left_censored", "right_censored",
        "identity_feasible", "first_unsafe_failures",
        "conservative_safe_pop_gain_over_identity_pct",
        "maximum_conservative_safe_pop_gain_pct",
        "first_maximum_conservative_gain_scale", "geometry_frontiers",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RuntimeError("ordinal geometry intersection fields are incomplete")
    if (value.get("schema") != INTERSECTION_SCHEMA or
            value.get("contract") != INTERSECTION_CONTRACT or
            value.get("scale_thresholds") != list(ordinal_contract.SCALES)):
        raise RuntimeError("ordinal geometry intersection contract differs")
    rebuilt = _build_intersection(value.get("geometry_frontiers"), False)
    if value != rebuilt:
        raise RuntimeError("ordinal geometry intersection is not canonical")
    return value
