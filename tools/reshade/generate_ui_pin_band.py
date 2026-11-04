#!/usr/bin/env python3
"""Materialize the shared UI pin band in Game 3D's self-contained embedded shader."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
OWNER = ROOT / "src_assets/windows/assets/shaders/directx/include/ui_pin_band_shared.hlsl"
GAME3D = ROOT / "tools/reshade/game3d_native.hlsl"
BEGIN = "// BEGIN GENERATED SHARED UI PIN BAND"
END = "// END GENERATED SHARED UI PIN BAND"


def normalized(raw: bytes) -> str:
    return raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def materialized(source: str, owner: str) -> str:
    """Replace exactly one marked block, preserving all surrounding source text."""
    if BEGIN in owner or END in owner:
        raise ValueError("Shared UI pin owner must not contain generated-block markers")
    if not owner.endswith("\n"):
        raise ValueError("Shared UI pin owner must end with a newline")
    if source.count(BEGIN) != 1 or source.count(END) != 1:
        raise ValueError("Game 3D must contain exactly one shared UI pin begin/end marker")
    begin = source.index(BEGIN)
    end = source.index(END)
    if end < begin:
        raise ValueError("Shared UI pin end marker precedes its begin marker")
    end += len(END)
    return source[:begin] + BEGIN + "\n" + owner + END + source[end:]


def update(owner_path: Path, game3d_path: Path, check: bool) -> bool:
    owner = normalized(owner_path.read_bytes())
    raw = game3d_path.read_bytes()
    source = normalized(raw)
    result = materialized(source, owner)
    if source == result:
        return True
    if check:
        return False
    # The standalone embedded shader can have CRLF in a Windows checkout. Do not
    # rewrite its line endings while refreshing only the authoritative block.
    newline = "\r\n" if b"\r\n" in raw else "\n"
    game3d_path.write_bytes(result.replace("\n", newline).encode("utf-8"))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="reject drift without writing")
    parser.add_argument("--owner", type=Path, default=OWNER)
    parser.add_argument("--game3d", type=Path, default=GAME3D)
    args = parser.parse_args()
    try:
        if not update(args.owner, args.game3d, args.check):
            print("Game 3D UI pin band differs from its shared owner; run "
                  "tools/reshade/generate_ui_pin_band.py", file=sys.stderr)
            return 1
    except (OSError, UnicodeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
