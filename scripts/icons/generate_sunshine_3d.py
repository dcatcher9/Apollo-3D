#!/usr/bin/env python3
"""Generate the Sunshine 3D Windows, tray, and web icon family."""

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

ACCENT = "#8AB4F8"
ACCENT_BRIGHT = "#D7E5FF"
GOLD = "#E0B020"
STATUS_OK = "#5CD65C"
STATUS_WARN = "#E0B020"
DANGER = "#FFB4AB"
DANGER_CONTAINER = "#6D3A3E"
ON_STATUS = "#0C0F14"


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


def draw_round_line(draw, points, fill, width):
    draw.line(points, fill=fill, width=width, joint="curve")
    radius = width // 2
    # Pillow can leave hairline wedges between steep polyline segments even
    # with joint="curve". Filling every sampled joint keeps the depth bands
    # solid in both the 1024 px source and the 16 px tray reduction.
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def draw_master(state="idle"):
    image = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Five heavy rays survive the 16 px tray-icon reduction.
    for start, end in (
        ((512, 160), (512, 236)),
        ((286, 222), (344, 280)),
        ((738, 222), (680, 280)),
        ((170, 414), (250, 414)),
        ((854, 414), (774, 414)),
    ):
        draw_round_line(draw, (start, end), GOLD, 52)

    # The sun sits behind two offset horizon bands: one symbol for
    # "Sunshine" and a readable near/far cue for "3D".
    draw.ellipse((332, 244, 692, 604), fill=GOLD)
    far_plane = cubic((188, 704), (342, 582), (682, 582), (836, 704))
    draw_round_line(draw, far_plane, ACCENT_BRIGHT, 104)
    near_plane = cubic((164, 810), (338, 650), (686, 650), (860, 810))
    draw_round_line(draw, near_plane, ACCENT, 112)

    if state == "playing":
        draw.ellipse((678, 678, 918, 918), fill=STATUS_OK)
        draw.polygon(((766, 740), (766, 856), (862, 798)), fill=ON_STATUS)
    elif state == "locked":
        draw.ellipse((678, 678, 918, 918), fill=DANGER_CONTAINER)
        draw.rounded_rectangle((746, 784, 850, 864), radius=20, fill=DANGER)
        draw.arc((760, 716, 836, 812), start=180, end=360, fill=DANGER, width=24)
    elif state == "pausing":
        draw.ellipse((678, 678, 918, 918), fill=STATUS_WARN)
        draw.rounded_rectangle((752, 744, 790, 852), radius=12, fill=ON_STATUS)
        draw.rounded_rectangle((806, 744, 844, 852), radius=12, fill=ON_STATUS)

    visible_box = (VISIBLE_INSET, VISIBLE_INSET,
                   CANVAS - VISIBLE_INSET, CANVAS - VISIBLE_INSET)
    return image.crop(visible_box).resize(
        (CANVAS, CANVAS), Image.Resampling.LANCZOS
    )


def svg_for(state="idle"):
    state_markup = ""
    if state == "playing":
        state_markup = """
  <circle cx="798" cy="798" r="120" fill="#5CD65C"/>
  <path d="M766 740V856L862 798Z" fill="#0C0F14"/>"""
    elif state == "locked":
        state_markup = """
  <circle cx="798" cy="798" r="120" fill="#6D3A3E"/>
  <rect x="746" y="784" width="104" height="80" rx="20" fill="#FFB4AB"/>
  <path d="M772 784V760A26 26 0 0 1 824 760V784" fill="none" stroke="#FFB4AB" stroke-width="24" stroke-linecap="round"/>"""
    elif state == "pausing":
        state_markup = """
  <circle cx="798" cy="798" r="120" fill="#E0B020"/>
  <rect x="752" y="744" width="38" height="108" rx="12" fill="#0C0F14"/>
  <rect x="806" y="744" width="38" height="108" rx="12" fill="#0C0F14"/>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="{VISIBLE_INSET} {VISIBLE_INSET} {CANVAS - 2 * VISIBLE_INSET} {CANVAS - 2 * VISIBLE_INSET}" role="img" aria-label="Sunshine 3D">
  <g fill="none" stroke="#E0B020" stroke-width="52" stroke-linecap="round">
    <path d="M512 160V236"/>
    <path d="M286 222L344 280"/>
    <path d="M738 222L680 280"/>
    <path d="M170 414H250"/>
    <path d="M854 414H774"/>
  </g>
  <circle cx="512" cy="424" r="180" fill="#E0B020"/>
  <path d="M188 704C342 582 682 582 836 704" fill="none" stroke="#D7E5FF" stroke-width="104" stroke-linecap="round"/>
  <path d="M164 810C338 650 686 650 860 810" fill="none" stroke="#8AB4F8" stroke-width="112" stroke-linecap="round"/>{state_markup}
</svg>
"""


def save_png(master, path, size):
    resized = master.resize((size, size), Image.Resampling.LANCZOS)
    resized.save(path, format="PNG", optimize=True)


def save_ico(master, path):
    master.save(path, format="ICO", sizes=ICO_SIZES, bitmap_format="png")


def require_transparent_canvas(image, label, minimum_fraction=0.5):
    alpha = image.getchannel("A")
    transparent_fraction = alpha.histogram()[0] / (image.width * image.height)
    if transparent_fraction < minimum_fraction or alpha.getpixel((0, 0)) != 0:
        raise RuntimeError(
            f"{label} must retain a transparent canvas; "
            f"fully transparent area was only {transparent_fraction:.1%}"
        )


def require_saved_icon_transparency(path):
    with Image.open(path) as source:
        if source.format == "ICO":
            frames = (
                source.ico.getimage((16, 16)).convert("RGBA"),
                source.ico.getimage((256, 256)).convert("RGBA"),
            )
        else:
            frames = (source.convert("RGBA"),)
        for frame in frames:
            require_transparent_canvas(
                frame,
                f"{path.name} {frame.width}px",
                0.25 if frame.width <= 16 else 0.5,
            )


def main():
    WEB_IMAGES.mkdir(parents=True, exist_ok=True)

    idle = draw_master("idle")
    playing = draw_master("playing")
    locked = draw_master("locked")
    pausing = draw_master("pausing")
    for label, master in (
        ("idle", idle),
        ("playing", playing),
        ("locked", locked),
        ("pausing", pausing),
    ):
        require_transparent_canvas(master, label)

    (ROOT / "sunshine3d.svg").write_text(svg_for("idle"), encoding="utf-8", newline="\n")
    (WEB_IMAGES / "logo-sunshine3d.svg").write_text(svg_for("idle"), encoding="utf-8", newline="\n")
    (WEB_IMAGES / "sunshine3d-playing.svg").write_text(svg_for("playing"), encoding="utf-8", newline="\n")
    (WEB_IMAGES / "sunshine3d-locked.svg").write_text(svg_for("locked"), encoding="utf-8", newline="\n")
    (WEB_IMAGES / "sunshine3d-pausing.svg").write_text(svg_for("pausing"), encoding="utf-8", newline="\n")

    save_png(idle, ROOT / "sunshine3d.png", 1024)
    save_png(idle, WEB_IMAGES / "logo-sunshine3d-16.png", 16)
    save_png(idle, WEB_IMAGES / "logo-sunshine3d-45.png", 45)
    save_ico(idle, ROOT / "sunshine3d.ico")
    shutil.copyfile(ROOT / "sunshine3d.ico", WEB_IMAGES / "sunshine3d.ico")

    for name, master in (
        ("sunshine3d-playing", playing),
        ("sunshine3d-locked", locked),
        ("sunshine3d-pausing", pausing),
    ):
        save_png(master, WEB_IMAGES / f"{name}.png", 1024)
        save_png(master, WEB_IMAGES / f"{name}-16.png", 16)
        save_png(master, WEB_IMAGES / f"{name}-45.png", 45)
        save_ico(master, WEB_IMAGES / f"{name}.ico")

    for path in (
        ROOT / "sunshine3d.ico",
        WEB_IMAGES / "logo-sunshine3d-16.png",
        WEB_IMAGES / "sunshine3d-playing.ico",
        WEB_IMAGES / "sunshine3d-locked.ico",
        WEB_IMAGES / "sunshine3d-pausing.ico",
    ):
        require_saved_icon_transparency(path)

    print("Generated Sunshine 3D icon family.")


if __name__ == "__main__":
    main()
