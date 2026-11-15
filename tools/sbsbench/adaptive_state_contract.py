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
    / "sbs-adaptive-state-v7.json"
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
RETIRED_ZERO_FIELDS = tuple(str(name) for name in CONTRACT["retired_zero_fields"])
FIELD_ENCODINGS = tuple(
    str(field["gpu_encoding"]) for field in CONTRACT["fields"]
)
INITIAL_VALUES = {
    str(field["name"]): field["initial"] for field in CONTRACT["fields"]
}
