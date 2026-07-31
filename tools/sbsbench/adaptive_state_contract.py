"""Canonical Host SBS adaptive-state contract shared by Python tools.

The native and HLSL declarations are generated from the same JSON manifest by
``generate_adaptive_state_contract.py``. Import-time validation keeps malformed edits from
degrading into a positional decode with the wrong meaning.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


CONTRACT_PATH = (
    Path(__file__).resolve().parent
    / "contracts"
    / "sbs-adaptive-state-v4.json"
)


def _load_contract() -> dict[str, Any]:
    with CONTRACT_PATH.open(encoding="utf-8") as stream:
        contract = json.load(stream)
    fields = contract.get("fields")
    if not isinstance(fields, list) or not fields:
        raise RuntimeError("adaptive-state contract requires a non-empty fields list")
    indices = [field.get("index") for field in fields]
    if indices != list(range(len(fields))):
        raise RuntimeError("adaptive-state contract field indices must be contiguous and ordered")
    names = [field.get("name") for field in fields]
    if any(not isinstance(name, str) or not name for name in names):
        raise RuntimeError("adaptive-state contract field names must be non-empty strings")
    if len(set(names)) != len(names):
        raise RuntimeError("adaptive-state contract field names must be unique")
    for field in fields:
        if field.get("type") not in {"float32", "uint32"}:
            raise RuntimeError(f"invalid adaptive-state JSON type for {field.get('name')}")
        if field.get("gpu_encoding") not in {
            "float", "uint_bits", "uint_valued_float"
        }:
            raise RuntimeError(f"invalid adaptive-state GPU encoding for {field.get('name')}")
        initial = field.get("initial")
        if (not isinstance(initial, (int, float)) or isinstance(initial, bool) or
                not math.isfinite(float(initial))):
            raise RuntimeError(f"invalid adaptive-state initial value for {field.get('name')}")
    if len(fields) % 4:
        raise RuntimeError("adaptive-state word count must contain complete float4 vectors")
    render_prefix = contract.get("render_prefix_words")
    if (not isinstance(render_prefix, int) or
            not 0 < render_prefix <= len(fields) or render_prefix % 4):
        raise RuntimeError("adaptive-state render prefix must contain complete float4 vectors")
    for section in ("cut_flag_bits", "analysis_flag_bits"):
        bits = contract.get(section)
        if not isinstance(bits, dict) or not bits:
            raise RuntimeError(f"adaptive-state contract requires {section}")
        values = list(bits.values())
        if (any(not isinstance(bit, int) or isinstance(bit, bool) or not 0 <= bit < 32
                for bit in values) or len(set(values)) != len(values)):
            raise RuntimeError(f"adaptive-state {section} must contain unique uint32 bit indices")
    return contract


CONTRACT = _load_contract()
TRACE_SCHEMA = int(CONTRACT["schema"])
TRACE_SOURCE = str(CONTRACT["source"])
TRACE_CAPTURE = str(CONTRACT["capture"])
RENDER_PREFIX_WORDS = int(CONTRACT["render_prefix_words"])
FIELD_SPECS = tuple(
    (str(field["name"]), str(field["type"]))
    for field in CONTRACT["fields"]
)
FIELD_NAMES = tuple(name for name, _kind in FIELD_SPECS)
FIELD_DESCRIPTORS = tuple(
    {"name": name, "type": kind} for name, kind in FIELD_SPECS
)
HEADER_KEYS = frozenset(str(key) for key in CONTRACT["header_keys"])
CONFIG_KEYS = frozenset(str(key) for key in CONTRACT["config_keys"])
FRAME_KEYS = frozenset(str(key) for key in CONTRACT["frame_keys"])
CUT_FLAG_BITS = {
    str(name): int(bit) for name, bit in CONTRACT["cut_flag_bits"].items()
}
ANALYSIS_FLAG_BITS = {
    str(name): int(bit) for name, bit in CONTRACT["analysis_flag_bits"].items()
}
KNOWN_CUT_FLAG_MASK = sum(1 << bit for bit in CUT_FLAG_BITS.values())
KNOWN_ANALYSIS_FLAG_MASK = sum(1 << bit for bit in ANALYSIS_FLAG_BITS.values())


def flag_mask(bits: dict[str, int], name: str) -> int:
    return 1 << bits[name]


CUT_FLAG_GEOMETRY_ARMED = flag_mask(CUT_FLAG_BITS, "geometry_armed")
CUT_FLAG_APPEARANCE_ARMED = flag_mask(CUT_FLAG_BITS, "appearance_armed")
CUT_FLAG_GEOMETRY_LOW_ONCE = flag_mask(CUT_FLAG_BITS, "geometry_low_once")
CUT_FLAG_APPEARANCE_QUIET_ONCE = flag_mask(CUT_FLAG_BITS, "appearance_quiet_once")
CUT_FLAG_LATCHED = flag_mask(CUT_FLAG_BITS, "latched")
CUT_FLAG_APPEARANCE_RECOVERY = flag_mask(CUT_FLAG_BITS, "appearance_recovery")
CUT_FLAG_GEOMETRY_CONFIRMATION_PENDING = flag_mask(
    CUT_FLAG_BITS, "geometry_confirmation_pending"
)
