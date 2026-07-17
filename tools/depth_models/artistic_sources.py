#!/usr/bin/env python3
"""Validate generic artistic-policy source catalogs.

Schema 2 catalogs describe the input evidence independently of how labels are
generated.  ``artistic_sources.json`` is the active/default catalog.  Schema 1
is retained as a read-only compatibility contract for the legacy stereo
retrieval catalog and is normalized to ``authored-stereo`` in memory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


SPLITS = {"training", "development", "test"}
DEFAULT_ACTIVE_CATALOG_NAME = "artistic_sources.json"
SOURCE_KINDS = {
    "mono-video",
    "native-hdr-video",
    "authored-stereo",
    "gt-depth-flow",
    "still-spatial",
}
EYE_ORDERS = {"first-left", "first-right"}


def schema_is(value, *versions):
    """Match JSON schema versions without treating bool as an integer."""
    return type(value) is int and value in versions


def _non_empty(value):
    return isinstance(value, str) and bool(value.strip())


def _positive_number(value):
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def validate_catalog(payload, source="source catalog"):
    """Return a normalized catalog or raise ``RuntimeError``.

    Only sources admitted to the global policy are required to carry the full
    production, split, weight, and license contract.  This preserves excluded
    historical rows in the legacy catalog while making active evidence strict.
    """

    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid source catalog: {source}")
    schema = payload.get("schema")
    if not schema_is(schema, 1, 2):
        raise RuntimeError(f"unsupported source catalog schema: {source}")
    rows = payload.get("sources")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("source catalog has no sources")

    identifiers = set()
    production_splits = {}
    admitted_productions = {}
    sealed_productions = set()
    sealed_groups = set()
    normalized_rows = []

    for index, original in enumerate(rows):
        if not isinstance(original, dict):
            raise RuntimeError(f"source catalog row {index} is not an object")
        row = dict(original)
        identifier = row.get("id")
        if not _non_empty(identifier) or identifier in identifiers:
            raise RuntimeError(f"missing or duplicate source id: {identifier!r}")
        identifiers.add(identifier)

        if schema == 1:
            kind = "authored-stereo"
            row["source_kind"] = kind
        else:
            kind = row.get("source_kind")
            if kind not in SOURCE_KINDS:
                raise RuntimeError(
                    f"{identifier}: invalid source kind {kind!r}"
                )

        split = row.get("split")
        production = row.get("production_id")
        admission = row.get("admission")
        if split is not None and split not in SPLITS:
            raise RuntimeError(f"{identifier}: invalid split {split!r}")
        if production and split:
            previous = production_splits.setdefault(production, split)
            if previous != split:
                raise RuntimeError(
                    f"production {production!r} leaks across "
                    f"{previous} and {split}"
                )

        if kind == "still-spatial":
            if admission != "spatial_auxiliary":
                raise RuntimeError(
                    f"{identifier}: still-spatial evidence must use "
                    "spatial_auxiliary admission"
                )
            if (not _non_empty(row.get("license")) or
                    not _non_empty(row.get("license_url"))):
                raise RuntimeError(f"{identifier}: missing license provenance")

        if admission == "global_policy":
            if split not in SPLITS:
                raise RuntimeError(f"{identifier}: invalid global-policy split")
            if not _non_empty(production):
                raise RuntimeError(f"{identifier}: missing production id")
            if row.get("complete_production") is not True:
                raise RuntimeError(
                    f"{identifier}: global-policy source is not a complete "
                    "production"
                )
            if not _non_empty(row.get("source_group")):
                raise RuntimeError(f"{identifier}: missing source group")
            if not _positive_number(row.get("global_policy_weight")):
                raise RuntimeError(
                    f"{identifier}: global-policy weight is not positive"
                )
            if (not _non_empty(row.get("license")) or
                    not _non_empty(row.get("license_url"))):
                raise RuntimeError(f"{identifier}: missing license provenance")
            if kind == "authored-stereo":
                if row.get("eye_order") not in EYE_ORDERS:
                    raise RuntimeError(f"{identifier}: eye order is not verified")
                if not _positive_number(row.get("eye_display_aspect_ratio")):
                    raise RuntimeError(
                        f"{identifier}: display eye aspect is missing"
                    )

            owner = admitted_productions.setdefault(production, identifier)
            if owner != identifier:
                raise RuntimeError(
                    f"production {production!r} has duplicate admitted sources "
                    f"{owner!r} and {identifier!r}"
                )
            if split == "test":
                sealed_productions.add(production)
                sealed_groups.add(row["source_group"])

        normalized_rows.append(row)

    # A schema-2 catalog may deliberately start as an auxiliary-only migration
    # scaffold while real mono productions are prepared and registered.  Once
    # it admits any global-policy source, it must already describe a complete
    # auditable split.  The active-split audit independently enforces the same
    # requirement over the exact supplied dataset manifests.
    if admitted_productions:
        if len(sealed_productions) < 2:
            raise RuntimeError("source catalog needs two sealed test productions")
        if len(sealed_groups) < 2:
            raise RuntimeError("sealed tests need two independent source groups")

    normalized = dict(payload)
    normalized["sources"] = normalized_rows
    return normalized


def load_catalog(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read source catalog: {path}") from error
    return validate_catalog(payload, str(path))
