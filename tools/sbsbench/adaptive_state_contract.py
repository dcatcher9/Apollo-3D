"""Canonical Host SBS adaptive-state contract shared by Python tools.

The native and HLSL declarations are generated from the same JSON manifest by
``generate_adaptive_state_contract.py``. Import-time validation keeps malformed edits from
degrading into a positional decode with the wrong meaning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONTRACT_PATH = (
    Path(__file__).resolve().parent
    / "contracts"
    / "sbs-adaptive-state-v6.json"
)

try:
    from .generate_adaptive_state_contract import (  # type: ignore
        contract_digest,
        contract_tag,
        validate_contract,
    )
except ImportError:
    from generate_adaptive_state_contract import (  # type: ignore
        contract_digest,
        contract_tag,
        validate_contract,
    )


def _load_contract() -> dict[str, Any]:
    with CONTRACT_PATH.open(encoding="utf-8") as stream:
        contract = json.load(stream)
    try:
        return validate_contract(contract)
    except ValueError as exc:
        raise RuntimeError(f"invalid adaptive-state contract: {exc}") from exc


CONTRACT = _load_contract()
CUT_CONTRACT_TAG = contract_tag(CONTRACT)
CONTRACT_CANONICAL_SHA256 = contract_digest(CONTRACT)
COUNTER_MAX = int(CONTRACT["counter_max"])
TRACE_SCHEMA = int(CONTRACT["schema"])
TRACE_SOURCE = str(CONTRACT["source"])
TRACE_CAPTURE = str(CONTRACT["capture"])
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
COMPACT_CUT_TRACE_FIELDS = tuple(CONTRACT["compact_cut_trace"]["fields"])
COMPACT_CUT_TRACE_ROLES = {
    name: dict(role)
    for name, role in CONTRACT["compact_cut_trace"]["roles"].items()
}


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
FIELD_ENCODINGS = tuple(
    str(field["gpu_encoding"]) for field in CONTRACT["fields"]
)
INITIAL_VALUES = {
    str(field["name"]): field["initial"] for field in CONTRACT["fields"]
}
