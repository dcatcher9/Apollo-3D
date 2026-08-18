#!/usr/bin/env python3
"""Generate non-authoritative diagnostic previews from a schema-39 Dump 3D package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import depth_coordinate_v2_dump_contract as dump_contract
except ImportError:  # Direct execution from tools/sbsbench.
    import depth_coordinate_v2_dump_contract as dump_contract  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path, help="schema-39 Dump 3D directory")
    parser.add_argument("artifact", help="authenticated .f32 artifact filename")
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="directory for generated PNGs (kept outside the atomic package contract)")
    args = parser.parse_args()
    summary = dump_contract.generate_float_artifact_previews(
        args.dump, args.artifact, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
