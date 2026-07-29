"""Canonical Host SBS Scene Controller ABI v1 shared by native, shader, and Python code."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


CONTRACT_PATH = (
    Path(__file__).resolve().parent
    / "contracts"
    / "sbs-scene-controller-v1.json"
)

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_INDEXED_SECTIONS = (
    "inputs",
    "outputs",
    "integer_sidecar",
    "analysis_grid",
    "layout_history",
    "depth_history",
    "meta",
    "dense_out",
    "global_out",
    "rule_state",
)


def canonical_abi_bytes(contract: dict[str, Any]) -> bytes:
    """Return the ordered ABI payload covered by ``ordered_abi_hash``.

    JSON object keys are canonicalized, while array order remains significant. Consequently a
    field reorder, rename, type/shape change, default change, or enum/flag change changes the
    digest even when every section retains the same element count.
    """

    payload = copy.deepcopy(contract)
    payload.pop("ordered_abi_hash", None)
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def compute_ordered_abi_hash(contract: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_abi_bytes(contract)).hexdigest()


def _require_indexed_section(contract: dict[str, Any], section: str) -> list[dict[str, Any]]:
    entries = contract.get(section)
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"scene-controller contract requires non-empty {section}")
    if any(not isinstance(entry, dict) for entry in entries):
        raise RuntimeError(f"scene-controller {section} entries must be objects")
    indices = [entry.get("index") for entry in entries]
    if indices != list(range(len(entries))):
        raise RuntimeError(
            f"scene-controller {section} indices must be contiguous and ordered"
        )
    names = [entry.get("name") for entry in entries]
    if any(not isinstance(name, str) or not name for name in names):
        raise RuntimeError(f"scene-controller {section} names must be non-empty strings")
    if len(set(names)) != len(names):
        raise RuntimeError(f"scene-controller {section} names must be unique")
    return entries


def validate_contract(
    contract: dict[str, Any],
    *,
    verify_hash: bool = True,
) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise RuntimeError("scene-controller contract must be a JSON object")
    if contract.get("schema") != 1:
        raise RuntimeError("scene-controller contract schema must be 1")
    if contract.get("name") != "sbs_scene_controller":
        raise RuntimeError("scene-controller contract name mismatch")
    if contract.get("rule_revision") != "rules_v1":
        raise RuntimeError("scene-controller contract rule revision mismatch")
    if (
        contract.get("abi_hash_algorithm")
        != "sha256-canonical-json-without-ordered_abi_hash"
    ):
        raise RuntimeError("scene-controller ABI hash algorithm mismatch")

    dimensions = contract.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise RuntimeError("scene-controller contract requires dimensions")
    dimension_names: list[str] = []
    dimension_values: dict[str, int] = {}
    for dimension in dimensions:
        if not isinstance(dimension, dict):
            raise RuntimeError("scene-controller dimensions must be objects")
        name = dimension.get("name")
        value = dimension.get("value")
        if not isinstance(name, str) or not name:
            raise RuntimeError("scene-controller dimension names must be non-empty")
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise RuntimeError(f"invalid scene-controller dimension {name}")
        dimension_names.append(name)
        dimension_values[name] = value
    if len(set(dimension_names)) != len(dimension_names):
        raise RuntimeError("scene-controller dimension names must be unique")

    indexed = {
        section: _require_indexed_section(contract, section)
        for section in _INDEXED_SECTIONS
    }

    dynamic_dimensions = contract.get("dynamic_dimensions")
    if (
        not isinstance(dynamic_dimensions, dict)
        or set(dynamic_dimensions) != {"Hd", "Wd"}
    ):
        raise RuntimeError("scene-controller dynamic dimensions must be exactly Hd and Wd")
    allowed_shape_values = {
        value for value in dimension_values.values()
    } | {1, 3, "Hd", "Wd"}
    for section in ("inputs", "outputs"):
        for tensor in indexed[section]:
            if tensor.get("type") != "float32":
                raise RuntimeError(
                    f"scene-controller tensor {tensor['name']} must use float32 interop"
                )
            shape = tensor.get("shape")
            if (
                not isinstance(shape, list)
                or len(shape) not in {2, 4}
                or any(value not in allowed_shape_values for value in shape)
            ):
                raise RuntimeError(
                    f"invalid scene-controller tensor shape for {tensor['name']}"
                )

    sidecar_types = {"uint32", "uint64", "sha256"}
    for field in indexed["integer_sidecar"]:
        if field.get("type") not in sidecar_types:
            raise RuntimeError(
                f"invalid integer-sidecar type for {field['name']}"
            )

    count_bindings = {
        "analysis_grid": "analysis_grid_channel_count",
        "layout_history": "layout_history_channel_count",
        "depth_history": "depth_history_channel_count",
        "meta": "meta_word_count",
        "dense_out": "dense_out_channel_count",
        "global_out": "global_out_word_count",
        "rule_state": "rule_state_word_count",
    }
    for section, dimension_name in count_bindings.items():
        if len(indexed[section]) != dimension_values.get(dimension_name):
            raise RuntimeError(
                f"scene-controller {section} count does not match {dimension_name}"
            )

    rule_state = indexed["rule_state"]
    if len(rule_state) % 4:
        raise RuntimeError("scene-controller rule state must contain complete float4 vectors")
    for field in rule_state:
        if field.get("type") not in {"float32", "uint32"}:
            raise RuntimeError(f"invalid rule-state type for {field['name']}")
        if field.get("gpu_encoding") not in {
            "float",
            "uint_bits",
            "uint_valued_float",
        }:
            raise RuntimeError(f"invalid rule-state GPU encoding for {field['name']}")
        initial = field.get("initial")
        if (
            not isinstance(initial, (int, float))
            or isinstance(initial, bool)
            or not math.isfinite(float(initial))
        ):
            raise RuntimeError(f"invalid rule-state initial value for {field['name']}")
        if field.get("required_zero") and float(initial) != 0.0:
            raise RuntimeError(f"reserved rule-state field {field['name']} must initialize to zero")

    for section in ("meta", "global_out"):
        reserved = [entry for entry in indexed[section] if entry["name"].startswith("reserved_")]
        if not reserved or any(entry.get("required_zero") is not True for entry in reserved):
            raise RuntimeError(
                f"scene-controller {section} reserved tail must be marked required-zero"
            )

    enums = contract.get("enums")
    if not isinstance(enums, dict) or not enums:
        raise RuntimeError("scene-controller contract requires enums")
    for enum_name, values in enums.items():
        if not isinstance(values, dict) or not values:
            raise RuntimeError(f"scene-controller enum {enum_name} must be non-empty")
        if any(not isinstance(name, str) or not name for name in values):
            raise RuntimeError(f"scene-controller enum {enum_name} has an invalid name")
        numeric_values = list(values.values())
        if (
            any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in numeric_values
            )
            or len(set(numeric_values)) != len(numeric_values)
        ):
            raise RuntimeError(f"scene-controller enum {enum_name} values must be unique uints")

    flag_bits = contract.get("flag_bits")
    if not isinstance(flag_bits, dict) or not flag_bits:
        raise RuntimeError("scene-controller contract requires flag bits")
    for flag_name, bits in flag_bits.items():
        if not isinstance(bits, dict) or not bits:
            raise RuntimeError(f"scene-controller flag set {flag_name} must be non-empty")
        numeric_bits = list(bits.values())
        if (
            any(
                not isinstance(bit, int)
                or isinstance(bit, bool)
                or not 0 <= bit < 32
                for bit in numeric_bits
            )
            or len(set(numeric_bits)) != len(numeric_bits)
        ):
            raise RuntimeError(
                f"scene-controller flag set {flag_name} must use unique uint32 bits"
            )

    stored_hash = contract.get("ordered_abi_hash")
    if verify_hash:
        if not isinstance(stored_hash, str) or not _HASH_PATTERN.fullmatch(stored_hash):
            raise RuntimeError("scene-controller ordered ABI hash is not a SHA-256 digest")
        computed_hash = compute_ordered_abi_hash(contract)
        if stored_hash != computed_hash:
            raise RuntimeError(
                "scene-controller ordered ABI hash mismatch: "
                f"stored {stored_hash}, computed {computed_hash}"
            )
    return contract


def load_contract(
    path: Path = CONTRACT_PATH,
    *,
    verify_hash: bool = True,
) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        contract = json.load(stream)
    return validate_contract(contract, verify_hash=verify_hash)


CONTRACT = load_contract()
SCHEMA_VERSION = int(CONTRACT["schema"])
RULE_REVISION = str(CONTRACT["rule_revision"])
ORDERED_ABI_HASH = str(CONTRACT["ordered_abi_hash"])
DIMENSIONS = {
    str(dimension["name"]): int(dimension["value"])
    for dimension in CONTRACT["dimensions"]
}
INPUT_NAMES = tuple(str(entry["name"]) for entry in CONTRACT["inputs"])
OUTPUT_NAMES = tuple(str(entry["name"]) for entry in CONTRACT["outputs"])
ANALYSIS_GRID_NAMES = tuple(str(entry["name"]) for entry in CONTRACT["analysis_grid"])
LAYOUT_HISTORY_NAMES = tuple(str(entry["name"]) for entry in CONTRACT["layout_history"])
DEPTH_HISTORY_NAMES = tuple(str(entry["name"]) for entry in CONTRACT["depth_history"])
META_NAMES = tuple(str(entry["name"]) for entry in CONTRACT["meta"])
DENSE_OUT_NAMES = tuple(str(entry["name"]) for entry in CONTRACT["dense_out"])
GLOBAL_OUT_NAMES = tuple(str(entry["name"]) for entry in CONTRACT["global_out"])
RULE_STATE_NAMES = tuple(str(entry["name"]) for entry in CONTRACT["rule_state"])
