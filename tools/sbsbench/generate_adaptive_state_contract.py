#!/usr/bin/env python3
"""Generate native and HLSL adaptive-state declarations from the canonical manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT / "tools" / "sbsbench" / "contracts" / "sbs-adaptive-state-v7.json"
)
CPP_TARGET = ROOT / "src" / "generated" / "sbs_adaptive_state_contract.h"
HLSL_TARGET = (
    ROOT
    / "src_assets"
    / "windows"
    / "assets"
    / "shaders"
    / "directx"
    / "include"
    / "sbs_adaptive_state_contract.generated.hlsl"
)
CONTRACT_TAG_SENTINEL = "contract_tag"


def canonical_bytes(contract: dict[str, Any]) -> bytes:
    return json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def contract_digest(contract: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(contract)).hexdigest()


def tag_is_finite_normal(tag: int) -> bool:
    exponent = (tag >> 23) & 0xFF
    return 0 < exponent < 0xFF


def contract_tag(contract: dict[str, Any]) -> int:
    digest = hashlib.sha256(canonical_bytes(contract)).digest()
    tag = int.from_bytes(digest[:4], "big")
    nonce = 0
    while not tag_is_finite_normal(tag):
        nonce += 1
        candidate = hashlib.sha256(
            digest + b"\0sbs-adaptive-state-tag\0" + nonce.to_bytes(4, "big")
        ).digest()
        tag = int.from_bytes(candidate[:4], "big")
    return tag


def _identifier(value: str) -> str:
    identifier = re.sub(r"[^a-zA-Z0-9_]", "_", value)
    if not identifier or identifier[0].isdigit():
        raise ValueError(f"cannot turn {value!r} into an identifier")
    return identifier


def _upper_identifier(value: str) -> str:
    return _identifier(value).upper()


def _float_literal(value: float | int) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return f"{int(numeric)}.0f"
    return f"{numeric:.9g}f"


def _string_list(contract: dict[str, Any], name: str) -> list[str]:
    values = contract.get(name)
    if (not isinstance(values, list) or not values or
            any(not isinstance(value, str) or not value for value in values) or
            len(set(values)) != len(values)):
        raise ValueError(f"manifest {name} must contain unique non-empty strings")
    return values


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete canonical manifest before generation or decoding."""

    required_root = {
        "schema", "source", "capture", "counter_max", "header_keys", "config_keys", "frame_keys",
        "retired_zero_fields", "fields",
    }
    if not isinstance(contract, dict) or set(contract) != required_root:
        raise ValueError("adaptive-state manifest has missing or unknown root keys")
    if (not isinstance(contract["schema"], int) or isinstance(contract["schema"], bool) or
            contract["schema"] <= 0):
        raise ValueError("adaptive-state schema must be a positive integer")
    for name in ("source", "capture"):
        if not isinstance(contract[name], str) or not contract[name]:
            raise ValueError(f"adaptive-state {name} must be a non-empty string")
    if (not isinstance(contract["counter_max"], int) or
            isinstance(contract["counter_max"], bool) or
            not 0 <= contract["counter_max"] <= 0xFFFFFFFF):
        raise ValueError("adaptive-state counter_max must be uint32")
    for name in ("header_keys", "config_keys", "frame_keys"):
        _string_list(contract, name)

    fields = contract.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("manifest fields must be a non-empty array")
    if any(not isinstance(field, dict) or set(field) != {
            "index", "name", "type", "gpu_encoding", "initial"
            } for field in fields):
        raise ValueError("every manifest field must use the exact field schema")
    if [field["index"] for field in fields] != list(range(len(fields))):
        raise ValueError("manifest fields must be ordered by contiguous index")
    if len(fields) % 4:
        raise ValueError("manifest field count must be divisible by four")
    field_names: list[str] = []
    identifiers: list[str] = []
    upper_identifiers: list[str] = []
    for field in fields:
        name = field["name"]
        if not isinstance(name, str) or not name:
            raise ValueError("manifest field names must be non-empty strings")
        field_names.append(name)
        identifiers.append(_identifier(name))
        upper_identifiers.append(_upper_identifier(name))
        if field["type"] not in {"float32", "uint32"}:
            raise ValueError(f"invalid JSON field type for {name}")
        if field["gpu_encoding"] not in {"float", "uint_bits", "uint_valued_float"}:
            raise ValueError(f"invalid GPU field encoding for {name}")
        if field["gpu_encoding"] == "uint_bits" and field["type"] != "uint32":
            raise ValueError(f"uint_bits field {name} must have uint32 JSON type")
        initial = field["initial"]
        if initial == CONTRACT_TAG_SENTINEL:
            continue
        if (not isinstance(initial, (int, float)) or isinstance(initial, bool) or
                not math.isfinite(float(initial))):
            raise ValueError(f"field {name} must have a finite numeric initial value")
        if field["gpu_encoding"] in {"uint_bits", "uint_valued_float"} and (
                not isinstance(initial, int) or initial < 0 or initial > 0xFFFFFFFF):
            raise ValueError(f"integer-encoded field {name} must have a uint32 initial value")
    if (len(set(field_names)) != len(field_names) or
            len(set(identifiers)) != len(identifiers) or
            len(set(upper_identifiers)) != len(upper_identifiers)):
        raise ValueError("manifest field names must be unique after code generation")

    tag_fields = [field for field in fields if field.get("initial") == CONTRACT_TAG_SENTINEL]
    if (
        len(tag_fields) != 1 or
        tag_fields[0].get("index") != 0 or
        tag_fields[0].get("name") != "cut_contract_tag_bits" or
        tag_fields[0].get("type") != "uint32" or
        tag_fields[0].get("gpu_encoding") != "uint_bits"
    ):
        raise ValueError(
            "cut_contract_tag_bits must be the sole contract tag sentinel at word zero"
        )

    # A retired word keeps its slot and name so equality joins stay layout-stable, but no
    # producer writes it: it is permanently its zero initial value and readers enforce that.
    retired = contract.get("retired_zero_fields")
    if (not isinstance(retired, list) or
            any(not isinstance(name, str) for name in retired) or
            len(set(retired)) != len(retired)):
        raise ValueError("retired_zero_fields must be a unique list of field names")
    by_name = {field["name"]: field for field in fields}
    for name in retired:
        field = by_name.get(name)
        if field is None or field["initial"] != 0 or name.startswith("reserved_"):
            raise ValueError(
                f"retired field {name!r} must be a named field with a zero initial value")

    contract_tag(contract)
    return contract


def _load() -> dict[str, Any]:
    with MANIFEST.open(encoding="utf-8") as stream:
        contract = json.load(stream)
    return validate_contract(contract)


def _initial_float_cpp(field: dict[str, Any]) -> str:
    if field["initial"] == CONTRACT_TAG_SENTINEL:
        return "std::bit_cast<float>(cut_contract_tag)"
    return _float_literal(field["initial"])


def _initial_word_cpp(field: dict[str, Any]) -> str:
    if field["initial"] == CONTRACT_TAG_SENTINEL:
        return "cut_contract_tag"
    if field["gpu_encoding"] == "uint_bits":
        return f"{int(field['initial'])}u"
    return f"std::bit_cast<std::uint32_t>({_float_literal(field['initial'])})"


def _render_string_array(name: str, values: list[str]) -> list[str]:
    lines = [
        f"  inline constexpr std::array<std::string_view, {len(values)}> {name} {{{{",
    ]
    lines.extend(f'    "{value}",' for value in values)
    lines.append("  }};")
    return lines


def render_cpp(contract: dict[str, Any]) -> str:
    fields = contract["fields"]
    tag = contract_tag(contract)
    digest = contract_digest(contract)
    lines = [
        "// Generated by tools/sbsbench/generate_adaptive_state_contract.py.",
        "// Edit tools/sbsbench/contracts/sbs-adaptive-state-v7.json instead.",
        "#pragma once",
        "",
        "#include <array>",
        "#include <bit>",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <string_view>",
        "",
        "namespace sbs_adaptive_state {",
        f"  inline constexpr std::uint32_t schema_version = {contract['schema']}u;",
        f"  inline constexpr std::uint32_t cut_contract_tag = 0x{tag:08X}u;",
        f"  inline constexpr std::uint32_t counter_max = {contract['counter_max']}u;",
        f'  inline constexpr std::string_view contract_canonical_sha256 = "{digest}";',
        f'  inline constexpr std::string_view source = "{contract["source"]}";',
        f'  inline constexpr std::string_view capture = "{contract["capture"]}";',
        "  enum class gpu_encoding_e {",
        "    float_value,",
        "    uint_bits,",
        "    uint_valued_float,",
        "  };",
        "",
        "  enum class word_e : std::size_t {",
    ]
    lines.extend(
        f"    {_identifier(field['name'])} = {field['index']}u,"
        for field in fields
    )
    lines.extend([
        f"    count = {len(fields)}u,",
        "  };",
        "",
        "  constexpr std::size_t index(const word_e word) {",
        "    return static_cast<std::size_t>(word);",
        "  }",
        "",
        "  inline constexpr std::size_t word_count = index(word_e::count);",
        "  inline constexpr std::size_t vector_count = word_count / 4u;",
        "  using words_t = std::array<std::uint32_t, word_count>;",
        "",
        "  struct field_descriptor_t {",
        "    word_e word;",
        "    std::string_view name;",
        "    std::string_view json_type;",
        "    gpu_encoding_e gpu_encoding;",
        "    float initial_value;",
        "  };",
        "",
        f"  inline constexpr std::array<field_descriptor_t, {len(fields)}> fields {{{{",
    ])
    encoding = {
        "float": "float_value",
        "uint_bits": "uint_bits",
        "uint_valued_float": "uint_valued_float",
    }
    lines.extend(
        "    {word_e::%s, \"%s\", \"%s\", gpu_encoding_e::%s, %s},"
        % (
            _identifier(field["name"]),
            field["name"],
            field["type"],
            encoding[field["gpu_encoding"]],
            _initial_float_cpp(field),
        )
        for field in fields
    )
    lines.extend([
        "  }};",
        "",
        "  inline constexpr std::array<float, word_count> initial_values {{",
    ])
    lines.extend(
        f"    {_initial_float_cpp(field)}," for field in fields
    )
    lines.extend([
        "  }};",
        "",
        "  inline constexpr words_t initial_words {{",
    ])
    lines.extend(
        f"    {_initial_word_cpp(field)}," for field in fields
    )
    lines.extend([
        "  }};",
        "",
        "  constexpr bool field_layout_matches_indices() {",
        "    for (std::size_t position = 0; position < fields.size(); ++position) {",
        "      if (index(fields[position].word) != position) {",
        "        return false;",
        "      }",
        "    }",
        "    return true;",
        "  }",
        "",
        "  static_assert(word_count % 4u == 0u);",
        "  static_assert(field_layout_matches_indices());",
        "  static_assert(cut_contract_tag != 0u);",
        "  static_assert(((cut_contract_tag >> 23u) & 0xffu) > 0u);",
        "  static_assert(((cut_contract_tag >> 23u) & 0xffu) < 0xffu);",
        "  static_assert(initial_words[index(word_e::cut_contract_tag_bits)] ==",
        "                cut_contract_tag);",
        "",
    ])
    lines.extend(_render_string_array("header_keys", contract["header_keys"]))
    lines.append("")
    lines.extend(_render_string_array("config_keys", contract["config_keys"]))
    lines.append("")
    lines.extend(_render_string_array("frame_keys", contract["frame_keys"]))
    reserved = [field for field in fields if field["name"].startswith("reserved_")]
    lines.extend([
        "",
        "  inline constexpr std::array reserved_words {",
    ])
    lines.extend(
        f"    word_e::{_identifier(field['name'])}," for field in reserved
    )
    lines.append("  };")
    lines.extend([
        "",
        "  // Retired words: no producer writes them, so they are always zero.",
        "  inline constexpr std::array retired_zero_words {",
    ])
    lines.extend(
        f"    word_e::{_identifier(name)}," for name in contract["retired_zero_fields"]
    )
    lines.append("  };")
    lines.extend(["}  // namespace sbs_adaptive_state", ""])
    return "\n".join(lines)


def render_hlsl(contract: dict[str, Any]) -> str:
    fields = contract["fields"]
    component_names = ("x", "y", "z", "w")
    tag = contract_tag(contract)
    lines = [
        "// Generated by tools/sbsbench/generate_adaptive_state_contract.py.",
        "// Edit tools/sbsbench/contracts/sbs-adaptive-state-v7.json instead.",
        "#ifndef SBS_ADAPTIVE_STATE_CONTRACT_GENERATED_HLSL",
        "#define SBS_ADAPTIVE_STATE_CONTRACT_GENERATED_HLSL",
        "",
        f"#define SBS_ADAPTIVE_STATE_SCHEMA {contract['schema']}u",
        f"#define SBS_CUT_CONTRACT_TAG 0x{tag:08X}u",
        f"#define SBS_ADAPTIVE_STATE_COUNTER_MAX {contract['counter_max']}u",
        f"#define SBS_ADAPTIVE_STATE_WORD_COUNT {len(fields)}u",
        f"#define SBS_ADAPTIVE_STATE_VECTOR_COUNT {len(fields) // 4}u",
        "",
    ]
    for field in fields:
        upper = _upper_identifier(field["name"])
        vector = int(field["index"]) // 4
        component = component_names[int(field["index"]) % 4]
        lines.extend([
            f"#define SBS_STATE_WORD_{upper} {field['index']}u",
            f"#define SBS_STATE_VECTOR_{upper} {vector}u",
            f"#define SBS_STATE_{upper}(value) ((value).{component})",
        ])
    lines.append("")
    lines.append("// Retired words: no producer writes them, so they are always zero.")
    for name in contract["retired_zero_fields"]:
        lines.append(f"#define SBS_STATE_RETIRED_ZERO_{_upper_identifier(name)} 1")
    lines.extend(["", "float4 SbsAdaptiveStateInitialVector(uint index) {"])
    for vector in range(len(fields) // 4):
        values = []
        for field in fields[vector * 4:(vector + 1) * 4]:
            if field["initial"] == CONTRACT_TAG_SENTINEL:
                values.append("asfloat(SBS_CUT_CONTRACT_TAG)")
            elif field["gpu_encoding"] == "uint_bits":
                values.append(f"asfloat({int(field['initial'])}u)")
            else:
                values.append(_float_literal(field["initial"]))
        lines.append(
            f"    if (index == {vector}u) return float4({', '.join(values)});"
        )
    lines.extend(["    return 0.0f;", "}", "", "#endif", ""])
    return "\n".join(lines)


def _write_or_check(path: Path, expected: str, check: bool) -> bool:
    if check:
        try:
            actual = path.read_text(encoding="utf-8")
        except OSError:
            print(f"missing generated contract: {path}", file=sys.stderr)
            return False
        if actual != expected:
            print(f"stale generated contract: {path}", file=sys.stderr)
            return False
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8", newline="\n")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if committed generated files differ from the manifest",
    )
    args = parser.parse_args()
    contract = _load()
    valid = _write_or_check(CPP_TARGET, render_cpp(contract), args.check)
    valid &= _write_or_check(HLSL_TARGET, render_hlsl(contract), args.check)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
