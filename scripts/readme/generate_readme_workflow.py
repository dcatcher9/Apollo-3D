#!/usr/bin/env python3
"""Generate the deterministic Sunshine 3D + Moonlight 3D README animation."""

from __future__ import annotations

import argparse
import hashlib
import io
import math
from pathlib import Path

from PIL import Image, ImageDraw


WIDTH = 960
HEIGHT = 400
SCALE = 2
FRAME_COUNT = 48
FRAME_DURATION_MS = 100

BACKGROUND = (12, 15, 20, 255)
SURFACE = (22, 29, 40, 255)
SURFACE_2 = (31, 42, 58, 255)
MUTED = (82, 99, 124, 255)
PALE = (215, 229, 255, 255)
BLUE = (138, 180, 248, 255)
DEEP_BLUE = (67, 107, 174, 255)
GOLD = (224, 176, 32, 255)

OUTPUT_BASENAME = "sunshine3d-moonlight3d-workflow"


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    x = clamp((value - edge0) / (edge1 - edge0))
    return x * x * (3.0 - 2.0 * x)


def window(value: float, start: float, attack_end: float,
           release_start: float, end: float) -> float:
    return smoothstep(start, attack_end, value) * (1.0 - smoothstep(release_start, end, value))


def mix(left: float, right: float, amount: float) -> float:
    return left + (right - left) * amount


def color_mix(left: tuple[int, int, int, int],
              right: tuple[int, int, int, int],
              amount: float,
              alpha: float = 1.0) -> tuple[int, int, int, int]:
    amount = clamp(amount)
    alpha = clamp(alpha)
    mixed = (
        mix(left[0], right[0], amount),
        mix(left[1], right[1], amount),
        mix(left[2], right[2], amount),
    )
    return (
        round(mix(BACKGROUND[0], mixed[0], alpha)),
        round(mix(BACKGROUND[1], mixed[1], alpha)),
        round(mix(BACKGROUND[2], mixed[2], alpha)),
        255,
    )


class Canvas:
    def __init__(self) -> None:
        self.image = Image.new("RGBA", (WIDTH * SCALE, HEIGHT * SCALE), BACKGROUND)
        self.draw = ImageDraw.Draw(self.image, "RGBA")

    @staticmethod
    def point(x: float, y: float) -> tuple[int, int]:
        return round(x * SCALE), round(y * SCALE)

    @staticmethod
    def box(bounds: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        return tuple(round(value * SCALE) for value in bounds)

    def rounded_rectangle(self, bounds: tuple[float, float, float, float],
                          radius: float, fill, outline=None, width: float = 1.0) -> None:
        self.draw.rounded_rectangle(
            self.box(bounds),
            radius=round(radius * SCALE),
            fill=fill,
            outline=outline,
            width=max(1, round(width * SCALE)),
        )

    def rectangle(self, bounds: tuple[float, float, float, float],
                  fill, outline=None, width: float = 1.0) -> None:
        self.draw.rectangle(
            self.box(bounds),
            fill=fill,
            outline=outline,
            width=max(1, round(width * SCALE)),
        )

    def ellipse(self, bounds: tuple[float, float, float, float],
                fill, outline=None, width: float = 1.0) -> None:
        self.draw.ellipse(
            self.box(bounds),
            fill=fill,
            outline=outline,
            width=max(1, round(width * SCALE)),
        )

    def line(self, points: list[tuple[float, float]], fill,
             width: float = 1.0, joint: str = "curve") -> None:
        self.draw.line(
            [self.point(x, y) for x, y in points],
            fill=fill,
            width=max(1, round(width * SCALE)),
            joint=joint,
        )

    def polygon(self, points: list[tuple[float, float]], fill) -> None:
        self.draw.polygon([self.point(x, y) for x, y in points], fill=fill)

    def arc(self, bounds: tuple[float, float, float, float],
            start: float, end: float, fill, width: float) -> None:
        self.draw.arc(
            self.box(bounds),
            start=start,
            end=end,
            fill=fill,
            width=max(1, round(width * SCALE)),
        )

    def finish(self) -> Image.Image:
        return self.image.convert("RGB").resize(
            (WIDTH, HEIGHT), Image.Resampling.LANCZOS
        )


def quadratic(start: tuple[float, float], control: tuple[float, float],
              end: tuple[float, float], amount: float) -> tuple[float, float]:
    inv = 1.0 - amount
    return (
        inv * inv * start[0] + 2.0 * inv * amount * control[0] + amount * amount * end[0],
        inv * inv * start[1] + 2.0 * inv * amount * control[1] + amount * amount * end[1],
    )


def draw_curve(canvas: Canvas, start, control, end, fill, width: float,
               dashed: bool = False) -> None:
    points = [quadratic(start, control, end, index / 64.0) for index in range(65)]
    if not dashed:
        canvas.line(points, fill=fill, width=width)
        return
    for index in range(0, 64, 6):
        canvas.line(points[index:min(index + 4, 65)], fill=fill, width=width)


def draw_sun_badge(canvas: Canvas, cx: float, cy: float) -> None:
    canvas.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=GOLD)
    for angle in range(0, 360, 45):
        radians = math.radians(angle)
        start = (cx + math.cos(radians) * 16, cy + math.sin(radians) * 16)
        end = (cx + math.cos(radians) * 23, cy + math.sin(radians) * 23)
        canvas.line([start, end], fill=GOLD, width=4)


def draw_moon_badge(canvas: Canvas, cx: float, cy: float) -> None:
    canvas.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), fill=PALE)
    canvas.ellipse((cx - 4, cy - 17, cx + 18, cy + 5), fill=BACKGROUND)


def draw_generic_scene(canvas: Canvas, bounds: tuple[float, float, float, float],
                       alpha: float = 1.0, x_shift: float = 0.0) -> None:
    x0, y0, x1, y1 = bounds
    x0 += x_shift
    x1 += x_shift
    outline = color_mix(MUTED, PALE, 0.35, alpha)
    fill = color_mix(BACKGROUND, SURFACE_2, 0.80, alpha)
    canvas.rounded_rectangle((x0, y0, x1, y1), 10, fill=fill, outline=outline, width=3)
    canvas.line([(x0, y0 + 22), (x1, y0 + 22)], fill=outline, width=2)
    for offset, dot_color in ((0, GOLD), (13, BLUE), (26, PALE)):
        canvas.ellipse(
            (x0 + 12 + offset, y0 + 8, x0 + 20 + offset, y0 + 16),
            fill=color_mix(BACKGROUND, dot_color, 0.85, alpha),
        )
    video_fill = color_mix(SURFACE, DEEP_BLUE, 0.40, alpha)
    canvas.rounded_rectangle(
        (x0 + 15, y0 + 34, x0 + (x1 - x0) * 0.66, y1 - 14),
        8,
        fill=video_fill,
    )
    play_x = x0 + (x1 - x0) * 0.36
    play_y = y0 + (y1 - y0) * 0.60
    canvas.polygon(
        [(play_x - 9, play_y - 13), (play_x - 9, play_y + 13), (play_x + 14, play_y)],
        fill=color_mix(BACKGROUND, PALE, 0.95, alpha),
    )
    card_x0 = x0 + (x1 - x0) * 0.71
    for index in range(3):
        top = y0 + 35 + index * 24
        canvas.rounded_rectangle(
            (card_x0, top, x1 - 12, top + 14),
            5,
            fill=color_mix(SURFACE, BLUE if index == 0 else PALE, 0.35, alpha),
        )


def draw_pc(canvas: Canvas, scan: float, scan_alpha: float) -> None:
    canvas.rounded_rectangle((48, 62, 372, 306), 28, fill=SURFACE, outline=MUTED, width=5)
    canvas.rounded_rectangle((66, 80, 354, 283), 16, fill=BACKGROUND, outline=PALE, width=3)
    draw_generic_scene(canvas, (82, 96, 338, 267))
    canvas.rounded_rectangle((188, 306, 232, 343), 8, fill=SURFACE_2)
    canvas.rounded_rectangle((143, 337, 277, 354), 8, fill=MUTED)
    draw_sun_badge(canvas, 333, 75)
    if scan_alpha > 0.01:
        x = mix(88, 332, scan)
        for offset, opacity in ((-10, 0.08), (-5, 0.18), (0, 0.80), (5, 0.18), (10, 0.08)):
            canvas.line(
                [(x + offset, 100), (x + offset, 263)],
                fill=color_mix(BACKGROUND, GOLD, 1.0, scan_alpha * opacity),
                width=3 if offset else 5,
            )


def draw_depth_stack(canvas: Canvas, amount: float, drift: float) -> None:
    if amount <= 0.01:
        return
    base = (416, 124, 570, 236)
    layers = [
        (-18 - drift * 2, -12, DEEP_BLUE, 0.42),
        (-3 + drift * 3, -2, BLUE, 0.58),
        (14 + drift * 7, 10, GOLD, 0.82),
    ]
    for x_offset, y_offset, accent, opacity in layers:
        x0, y0, x1, y1 = base
        alpha = amount * opacity
        canvas.rounded_rectangle(
            (x0 + x_offset, y0 + y_offset, x1 + x_offset, y1 + y_offset),
            12,
            fill=color_mix(BACKGROUND, SURFACE_2, 0.92, alpha),
            outline=color_mix(BACKGROUND, accent, 0.95, alpha),
            width=4,
        )
        canvas.line(
            [(x0 + x_offset + 14, y0 + y_offset + 26),
             (x1 + x_offset - 14, y0 + y_offset + 26)],
            fill=color_mix(BACKGROUND, accent, 0.85, alpha),
            width=3,
        )
        canvas.rounded_rectangle(
            (x0 + x_offset + 18, y0 + y_offset + 42,
             x0 + x_offset + 88, y1 + y_offset - 16),
            7,
            fill=color_mix(BACKGROUND, accent, 0.48, alpha),
        )
        canvas.rounded_rectangle(
            (x0 + x_offset + 99, y0 + y_offset + 44,
             x1 + x_offset - 16, y0 + y_offset + 58),
            5,
            fill=color_mix(BACKGROUND, PALE, 0.42, alpha),
        )
        canvas.rounded_rectangle(
            (x0 + x_offset + 99, y0 + y_offset + 68,
             x1 + x_offset - 28, y0 + y_offset + 82),
            5,
            fill=color_mix(BACKGROUND, BLUE, 0.32, alpha),
        )


def draw_xr_headset(canvas: Canvas, activity: float, drift: float) -> None:
    outline = color_mix(MUTED, PALE, 0.75, 0.40 + activity * 0.60)
    glow = color_mix(DEEP_BLUE, BLUE, activity, 0.35 + activity * 0.65)
    canvas.arc((702, 42, 916, 176), 202, 338, fill=outline, width=10)
    canvas.rounded_rectangle((720, 79, 898, 160), 30, fill=SURFACE, outline=outline, width=6)
    canvas.rounded_rectangle((737, 96, 881, 143), 20, fill=BACKGROUND, outline=glow, width=4)
    center = 809 + drift * 5
    canvas.arc((center - 56, 102, center + 20, 138), 195, 345, fill=PALE, width=7)
    canvas.arc((center - 24, 104, center + 54, 141), 195, 345, fill=BLUE, width=7)
    draw_moon_badge(canvas, 885, 76)


def draw_ar_glasses(canvas: Canvas, activity: float, drift: float) -> None:
    outline = color_mix(MUTED, PALE, 0.78, 0.42 + activity * 0.58)
    lens = color_mix(DEEP_BLUE, BLUE, activity, 0.25 + activity * 0.50)
    left = (724, 273, 800, 325)
    right = (818, 273, 894, 325)
    canvas.rounded_rectangle(left, 18, fill=SURFACE, outline=outline, width=6)
    canvas.rounded_rectangle(right, 18, fill=SURFACE, outline=outline, width=6)
    canvas.line([(800, 292), (818, 292)], fill=outline, width=6)
    canvas.line([(724, 282), (690, 267)], fill=outline, width=6)
    canvas.line([(894, 282), (925, 267)], fill=outline, width=6)
    for bounds, direction in ((left, -1), (right, 1)):
        x0, y0, x1, y1 = bounds
        shift = drift * 4 * direction
        canvas.arc((x0 + 11 + shift, y0 + 12, x1 - 6 + shift, y1 - 4),
                   195, 345, fill=lens, width=6)
        canvas.arc((x0 + 6 - shift, y0 + 18, x1 - 12 - shift, y1 + 2),
                   195, 345, fill=PALE, width=5)


def draw_pulse(canvas: Canvas, start, control, end, amount: float,
               color, strength: float) -> None:
    if strength <= 0.04:
        return
    x, y = quadratic(start, control, end, amount % 1.0)
    canvas.ellipse(
        (x - 9, y - 9, x + 9, y + 9),
        fill=color_mix(BACKGROUND, color, 1.0, strength * 0.20),
    )
    canvas.ellipse(
        (x - 4, y - 4, x + 4, y + 4),
        fill=color_mix(BACKGROUND, color, 1.0, strength),
    )


def render_frame(index: int, poster: bool = False) -> Image.Image:
    t = index / FRAME_COUNT
    canvas = Canvas()

    # Quiet background depth arcs echo both app icons without adding words.
    canvas.arc(
        (120, 285, 600, 520), 195, 345,
        fill=color_mix(BACKGROUND, DEEP_BLUE, 0.55, 0.32), width=8,
    )
    canvas.arc(
        (90, 310, 640, 570), 195, 345,
        fill=color_mix(BACKGROUND, DEEP_BLUE, 0.42, 0.24), width=7,
    )

    scan_phase = clamp((t - 0.03) / 0.23)
    scan_alpha = window(t, 0.02, 0.07, 0.23, 0.29)
    if poster:
        scan_alpha = 0.0
    draw_pc(canvas, scan_phase, scan_alpha)

    appear = smoothstep(0.10, 0.27, t)
    disappear = 1.0 - smoothstep(0.91, 0.99, t)
    stack_amount = appear * disappear
    if poster:
        stack_amount = 1.0
    drift = math.sin(t * math.tau) * 0.85
    draw_depth_stack(canvas, stack_amount, drift)

    remote_active = window(t, 0.24, 0.31, 0.47, 0.52)
    local_active = window(t, 0.58, 0.64, 0.86, 0.93)
    if poster:
        remote_active = 0.88
        local_active = 0.10
    remote_visibility = stack_amount * (0.18 + 0.82 * remote_active)
    local_visibility = stack_amount * (0.18 + 0.82 * local_active)

    remote_start = (574, 159)
    remote_control = (652, 82)
    remote_end = (728, 118)
    local_start = (574, 215)
    local_control = (646, 290)
    local_end = (724, 294)

    draw_curve(
        canvas, remote_start, remote_control, remote_end,
        fill=color_mix(BACKGROUND, BLUE, 1.0, remote_visibility),
        width=6, dashed=True,
    )
    draw_curve(
        canvas, local_start, local_control, local_end,
        fill=color_mix(BACKGROUND, PALE, 1.0, local_visibility),
        width=7, dashed=False,
    )

    draw_xr_headset(canvas, remote_active * stack_amount, drift)
    draw_ar_glasses(canvas, local_active * stack_amount, drift)

    pulse_phase = (t * 4.0) % 1.0
    draw_pulse(canvas, remote_start, remote_control, remote_end,
               pulse_phase, BLUE, remote_active * stack_amount)
    draw_pulse(canvas, remote_start, remote_control, remote_end,
               (pulse_phase + 0.52) % 1.0, BLUE, remote_active * stack_amount)
    draw_pulse(canvas, local_start, local_control, local_end,
               pulse_phase, PALE, local_active * stack_amount)
    draw_pulse(canvas, local_start, local_control, local_end,
               (pulse_phase + 0.52) % 1.0, PALE, local_active * stack_amount)

    return canvas.finish()


def encode_outputs() -> tuple[bytes, bytes]:
    frames = [render_frame(index) for index in range(FRAME_COUNT)]
    poster = render_frame(round(FRAME_COUNT * 0.52), poster=True)

    # A shared palette avoids frame-to-frame color shimmer and keeps the fallback GIF compact.
    samples = [frames[round(index * (FRAME_COUNT - 1) / 9)] for index in range(10)]
    palette_sheet = Image.new("RGB", (WIDTH * 5, HEIGHT * 2), BACKGROUND[:3])
    for index, sample in enumerate(samples):
        palette_sheet.paste(sample, ((index % 5) * WIDTH, (index // 5) * HEIGHT))
    palette = palette_sheet.quantize(
        colors=96,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )
    gif_frames = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in frames
    ]

    gif_buffer = io.BytesIO()
    gif_frames[0].save(
        gif_buffer,
        format="GIF",
        save_all=True,
        append_images=gif_frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )

    poster_buffer = io.BytesIO()
    poster.save(poster_buffer, format="PNG", optimize=True, compress_level=9)
    return gif_buffer.getvalue(), poster_buffer.getvalue()


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the checked-in animation matches this generator",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "docs" / "assets" / "readme"
    gif_data, poster_data = encode_outputs()
    results = [
        write_or_check(output_dir / f"{OUTPUT_BASENAME}.gif", gif_data, args.check),
        write_or_check(output_dir / f"{OUTPUT_BASENAME}.png", poster_data, args.check),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
