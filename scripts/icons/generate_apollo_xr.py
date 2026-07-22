#!/usr/bin/env python3
"""Generate the Apollo XR Windows, tray, and web icon family."""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
WEB_IMAGES = ROOT / "src_assets" / "common" / "assets" / "web" / "public" / "images"

CANVAS = 1024
# Windows reserves the full ICO frame for a tray icon. Keep only a slim optical
# margin so the badge reads at the same size as neighboring system icons.
VISIBLE_INSET = 52
ICO_SIZES = [(16, 16), (20, 20), (24, 24), (28, 28), (32, 32),
             (40, 40), (48, 48), (64, 64), (128, 128), (256, 256)]

CHARCOAL = "#111820"
WHITE = "#FFFFFF"
GOLD = "#F8B400"
BLUE = "#1677FF"
GREEN = "#22C55E"


def cubic(p0, p1, p2, p3, steps=36):
    points = []
    for index in range(steps + 1):
        t = index / steps
        u = 1.0 - t
        points.append((
            u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0],
            u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1],
        ))
    return points


def visor_points():
    points = []
    segments = [
        ((256, 446), (326, 414), (426, 398), (512, 398)),
        ((512, 398), (598, 398), (698, 414), (768, 446)),
        ((768, 446), (768, 446), (768, 526), (768, 566)),
        ((768, 566), (768, 620), (728, 650), (672, 670)),
        ((672, 670), (648, 679), (630, 685), (612, 692)),
        ((612, 692), (574, 706), (552, 687), (536, 650)),
        ((536, 650), (528, 632), (524, 622), (522, 616)),
        ((522, 616), (518, 606), (506, 606), (502, 616)),
        ((502, 616), (500, 622), (496, 632), (488, 650)),
        ((488, 650), (472, 687), (450, 706), (412, 692)),
        ((412, 692), (394, 685), (376, 679), (352, 670)),
        ((352, 670), (296, 650), (256, 620), (256, 566)),
        ((256, 566), (256, 526), (256, 446), (256, 446)),
    ]
    for index, segment in enumerate(segments):
        sampled = cubic(*segment)
        points.extend(sampled if index == 0 else sampled[1:])
    return points


def draw_round_line(draw, points, fill, width):
    draw.line(points, fill=fill, width=width, joint="curve")
    radius = width // 2
    for x, y in (points[0], points[-1]):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def draw_master(state="idle"):
    image = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((72, 72, 952, 952), radius=206, fill=CHARCOAL)

    upper_orbit = cubic((292, 406), (404, 258), (632, 236), (734, 324))
    draw_round_line(draw, upper_orbit, GOLD, 58)
    draw.ellipse((720, 286, 832, 398), fill=GOLD)

    lower_orbit = cubic((286, 642), (336, 828), (610, 846), (758, 652))
    draw_round_line(draw, lower_orbit, BLUE, 66)

    draw.rounded_rectangle((220, 476, 274, 592), radius=25, fill=WHITE)
    draw.rounded_rectangle((750, 476, 804, 592), radius=25, fill=WHITE)
    draw.polygon(visor_points(), fill=WHITE)

    if state == "playing":
        draw.ellipse((678, 678, 918, 918), fill=GREEN)
        draw.polygon(((766, 740), (766, 856), (862, 798)), fill=WHITE)
    elif state == "locked":
        draw.ellipse((678, 678, 918, 918), fill=GOLD)
        draw.rounded_rectangle((746, 784, 850, 864), radius=20, fill=WHITE)
        draw.arc((760, 716, 836, 812), start=180, end=360, fill=WHITE, width=24)

    visible_box = (VISIBLE_INSET, VISIBLE_INSET,
                   CANVAS - VISIBLE_INSET, CANVAS - VISIBLE_INSET)
    return image.crop(visible_box).resize(
        (CANVAS, CANVAS), Image.Resampling.LANCZOS
    )


def svg_for(state="idle"):
    state_markup = ""
    if state == "playing":
        state_markup = """
  <circle cx="798" cy="798" r="120" fill="#22C55E"/>
  <path d="M766 740V856L862 798Z" fill="#FFFFFF"/>"""
    elif state == "locked":
        state_markup = """
  <circle cx="798" cy="798" r="120" fill="#F8B400"/>
  <rect x="746" y="784" width="104" height="80" rx="20" fill="#FFFFFF"/>
  <path d="M772 784V760A26 26 0 0 1 824 760V784" fill="none" stroke="#FFFFFF" stroke-width="24" stroke-linecap="round"/>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="{VISIBLE_INSET} {VISIBLE_INSET} {CANVAS - 2 * VISIBLE_INSET} {CANVAS - 2 * VISIBLE_INSET}" role="img" aria-label="Apollo XR">
  <rect x="72" y="72" width="880" height="880" rx="206" fill="#111820"/>
  <path d="M292 406C404 258 632 236 734 324" fill="none" stroke="#F8B400" stroke-width="58" stroke-linecap="round"/>
  <circle cx="776" cy="342" r="56" fill="#F8B400"/>
  <path d="M286 642C336 828 610 846 758 652" fill="none" stroke="#1677FF" stroke-width="66" stroke-linecap="round"/>
  <rect x="220" y="476" width="54" height="116" rx="25" fill="#FFFFFF"/>
  <rect x="750" y="476" width="54" height="116" rx="25" fill="#FFFFFF"/>
  <path d="M256 446C326 414 426 398 512 398S698 414 768 446V566C768 620 728 650 672 670L612 692C574 706 552 687 536 650L522 616C518 606 506 606 502 616L488 650C472 687 450 706 412 692L352 670C296 650 256 620 256 566Z" fill="#FFFFFF"/>{state_markup}
</svg>
"""


def save_png(master, path, size):
    resized = master.resize((size, size), Image.Resampling.LANCZOS)
    resized.save(path, format="PNG", optimize=True)


def save_ico(master, path):
    master.save(path, format="ICO", sizes=ICO_SIZES, bitmap_format="png")


def main():
    WEB_IMAGES.mkdir(parents=True, exist_ok=True)

    idle = draw_master("idle")
    playing = draw_master("playing")
    locked = draw_master("locked")

    (ROOT / "apollo.svg").write_text(svg_for("idle"), encoding="utf-8", newline="\n")
    (WEB_IMAGES / "logo-apollo.svg").write_text(svg_for("idle"), encoding="utf-8", newline="\n")
    (WEB_IMAGES / "apollo-playing.svg").write_text(svg_for("playing"), encoding="utf-8", newline="\n")
    (WEB_IMAGES / "apollo-locked.svg").write_text(svg_for("locked"), encoding="utf-8", newline="\n")

    save_png(idle, ROOT / "apollo.png", 1024)
    save_png(idle, WEB_IMAGES / "logo-apollo-16.png", 16)
    save_png(idle, WEB_IMAGES / "logo-apollo-45.png", 45)
    save_ico(idle, ROOT / "apollo.ico")
    shutil.copyfile(ROOT / "apollo.ico", WEB_IMAGES / "apollo.ico")

    for name, master in (("apollo-playing", playing), ("apollo-locked", locked)):
        save_png(master, WEB_IMAGES / f"{name}.png", 1024)
        save_png(master, WEB_IMAGES / f"{name}-16.png", 16)
        save_png(master, WEB_IMAGES / f"{name}-45.png", 45)
        save_ico(master, WEB_IMAGES / f"{name}.ico")

    print("Generated Apollo XR icon family.")


if __name__ == "__main__":
    main()
