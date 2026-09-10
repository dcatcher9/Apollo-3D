#!/usr/bin/env python3
"""Generate the shared, labeled Sunshine 3D + Moonlight 3D README diagram.

Run with Python 3.10+ and Pillow 10.1+; no external fonts or downloads are needed.
The SVG is the primary accessible asset. PNG and single-frame GIF companions
use the same drawing commands and Pillow's embedded font. The diagram is static
so all paths can be compared without motion or waiting for an animation.

Use --check to verify the checked-in assets. The checked-in raster assets were
generated with Pillow 12.3.0; use that version for byte-identical raster output.
"""

from __future__ import annotations

import argparse
import hashlib
import io
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 960
HEIGHT = 688
SCALE = 2
OUTPUT_BASENAME = "sunshine3d-moonlight3d-workflow"

BACKGROUND = "#101722"
SURFACE = "#182332"
BORDER = "#34455c"
TEXT = "#f1f5fc"
MUTED = "#bccadd"
GOLD = "#f0c65b"
GOLD_SURFACE = "#302b1d"
BLUE = "#91c4ff"
BLUE_SURFACE = "#1b3149"

TITLE = "Sunshine 3D + Moonlight 3D: where 3D happens"
DESCRIPTION = (
    "Host 3D: a flat picture goes to Sunshine 3D on the PC, which adds AI depth; "
    "Moonlight 3D shows stereo on the headset. "
    "Client 3D: Sunshine sends the flat picture; Moonlight adds AI depth on the headset. "
    "Raw SBS: Sunshine passes through the source's authored left and right views, "
    "and Moonlight shows them in 3D. In 2D mode, the picture stays flat. "
    "Separate PC options are local presentation to connected AR glasses and "
    "offline conversion of video into a saved side-by-side file."
)


class Canvas:
    """A shared scene keeps the SVG and raster fallback wording identical."""

    def __init__(self) -> None:
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title description">',
            f'<title id="title">{escape(TITLE)}</title>',
            f'<desc id="description">{escape(DESCRIPTION)}</desc>',
        ]
        self.image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), BACKGROUND)
        self.draw = ImageDraw.Draw(self.image)
        self.fonts: dict[int, ImageFont.FreeTypeFont] = {}

    def rect(self, x: int, y: int, width: int, height: int,
             fill: str, radius: int = 0, stroke: str | None = None) -> None:
        outline = f' stroke="{stroke}"' if stroke else ""
        self.svg.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" fill="{fill}"{outline}/>'
        )
        self.draw.rounded_rectangle(
            (x * SCALE, y * SCALE, (x + width) * SCALE, (y + height) * SCALE),
            radius=radius * SCALE, fill=fill, outline=stroke, width=SCALE,
        )

    def text(self, x: int, y: int, value: str, size: int = 17,
             fill: str = TEXT, bold: bool = False, anchor: str = "start") -> None:
        """Use baselines and explicit lines instead of renderer-dependent wrapping."""
        weight = ' font-weight="700"' if bold else ""
        self.svg.append(
            f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" '
            f'font-family="Arial, Helvetica, sans-serif" text-anchor="{anchor}"'
            f'{weight}>{escape(value)}</text>'
        )
        if size not in self.fonts:
            self.fonts[size] = ImageFont.load_default(size=size * SCALE)
        self.draw.text(
            (x * SCALE, y * SCALE), value, font=self.fonts[size], fill=fill,
            anchor="ms" if anchor == "middle" else "ls",
            stroke_width=1 if bold else 0, stroke_fill=fill,
        )

    def arrow(self, x1: int, x2: int, y: int, color: str = MUTED) -> None:
        self.svg.append(
            f'<path d="M {x1} {y} H {x2 - 7} M {x2 - 9} {y - 5} '
            f'L {x2 - 3} {y} L {x2 - 9} {y + 5}" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        self.draw.line([(x1 * SCALE, y * SCALE), ((x2 - 7) * SCALE, y * SCALE)],
                       fill=color, width=2 * SCALE)
        self.draw.line([((x2 - 9) * SCALE, (y - 5) * SCALE),
                        ((x2 - 3) * SCALE, y * SCALE),
                        ((x2 - 9) * SCALE, (y + 5) * SCALE)],
                       fill=color, width=2 * SCALE, joint="curve")

    def card(self, x: int, y: int, width: int, title: str, subtitle: str,
             accent: str | None = None) -> None:
        fill = GOLD_SURFACE if accent == GOLD else BLUE_SURFACE if accent == BLUE else SURFACE
        self.rect(x, y, width, 80, fill, radius=12, stroke=accent or BORDER)
        self.text(x + width // 2, y + 33, title, size=19,
                  fill=accent or TEXT, bold=True, anchor="middle")
        self.text(x + width // 2, y + 59, subtitle, size=16,
                  fill=MUTED, anchor="middle")

    def finish(self) -> dict[str, bytes]:
        svg = ("\n".join(self.svg) + "\n</svg>\n").encode("utf-8")
        raster = self.image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        png = io.BytesIO()
        raster.save(png, format="PNG", optimize=True, compress_level=9)
        gif = io.BytesIO()
        raster.quantize(colors=128, dither=Image.Dither.NONE).save(gif, format="GIF")
        return {"svg": svg, "png": png.getvalue(), "gif": gif.getvalue()}


def encode_outputs() -> dict[str, bytes]:
    canvas = Canvas()
    canvas.rect(0, 0, WIDTH, HEIGHT, BACKGROUND, radius=18)
    canvas.rect(30, 30, 5, 54, GOLD, radius=2)
    canvas.text(49, 53, "Sunshine 3D + Moonlight 3D", size=28, bold=True)
    canvas.text(49, 81, "Choose where a flat picture becomes 3D.", size=18, fill=MUTED)

    # Equal columns make depth ownership explicit in every streaming lane.
    canvas.text(256, 127, "SOURCE", size=15, fill=MUTED, bold=True, anchor="middle")
    canvas.text(504, 119, "SUNSHINE 3D", size=17, fill=GOLD, bold=True, anchor="middle")
    canvas.text(504, 140, "Windows PC", size=15, fill=MUTED, anchor="middle")
    canvas.text(796, 119, "MOONLIGHT 3D", size=17, fill=BLUE, bold=True, anchor="middle")
    canvas.text(796, 140, "Galaxy XR headset", size=15, fill=MUTED, anchor="middle")

    rows = [
        (158, "Host 3D", "Depth on PC", "Flat picture", "one view",
         "Adds AI depth", "creates stereo", GOLD, "Shows 3D", "one view per eye", None),
        (263, "Client 3D", "Depth on headset", "Flat picture", "one view",
         "Sends flat video", "one view", None, "Adds AI depth", "then shows 3D", BLUE),
        (368, "Raw SBS", "Already in 3D", "Stereo picture", "left + right views",
         "Passes stereo", "keeps both views", None, "Shows 3D", "one view per eye", None),
    ]
    for (y, mode, note, source, source_note, host, host_note, host_accent,
         client, client_note, client_accent) in rows:
        canvas.text(30, y + 33, mode, size=20, bold=True)
        canvas.text(30, y + 58, note, size=14, fill=MUTED)
        canvas.card(173, y, 166, source, source_note)
        canvas.arrow(349, 382, y + 40)
        canvas.card(392, y, 224, host, host_note, host_accent)
        canvas.arrow(628, 664, y + 40)
        canvas.card(676, y, 240, client, client_note, client_accent)

    canvas.text(173, 477, "SBS = side-by-side: left and right eye views in one video frame.",
                size=15, fill=MUTED)
    canvas.rect(30, 497, 886, 58, SURFACE, radius=12, stroke=BORDER)
    canvas.text(48, 533, "2D", size=20, bold=True)
    canvas.text(111, 533, "Flat picture", size=17)
    canvas.arrow(219, 260, 527)
    canvas.text(272, 533, "Sunshine sends 2D", size=17)
    canvas.arrow(452, 493, 527)
    canvas.text(505, 533, "Moonlight shows a flat screen", size=17)

    # These are separate PC routes, not branches of the headset streaming path.
    canvas.text(30, 590, "MORE OPTIONS ON THE PC", size=14, fill=MUTED, bold=True)
    canvas.rect(30, 607, 432, 57, SURFACE, radius=10)
    canvas.rect(478, 607, 438, 57, SURFACE, radius=10)
    canvas.text(46, 630, "Local AR glasses", size=17, bold=True)
    canvas.text(46, 652, "PC to connected glasses: 2D or Host 3D", size=15, fill=MUTED)
    canvas.text(494, 630, "Offline video conversion", size=17, bold=True)
    canvas.text(494, 652, "Video file to Host 3D to a saved SBS file", size=15, fill=MUTED)
    return canvas.finish()


def write_or_check(path: Path, data: bytes, check: bool) -> bool:
    digest = hashlib.sha256(data).hexdigest()
    if check:
        if not path.exists() or path.read_bytes() != data:
            print(f"OUTDATED {path}")
            return False
        print(f"OK {path} sha256={digest}")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"WROTE {path} bytes={len(data)} sha256={digest}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify that checked-in SVG, PNG, and GIF match this generator")
    args = parser.parse_args()
    output_dir = Path(__file__).resolve().parents[2] / "docs" / "assets" / "readme"
    results = [
        write_or_check(output_dir / f"{OUTPUT_BASENAME}.{extension}", data, args.check)
        for extension, data in encode_outputs().items()
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
