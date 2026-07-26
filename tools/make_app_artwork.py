"""Regenerates the built-in Desktop / Virtual Desktop card artwork.

The shipped assets are 600x800 GREYSCALE rasters with the app name baked in. The client card
already draws the name over the bottom third, so the baked text is a duplicate, and greyscale
line art on a grey gradient gives the tile nothing to be recognised by at a glance.

These are drawn at 4x and downsampled, so every edge is anti-aliased rather than stair-stepped,
and they carry no text: the label belongs to the card, not the artwork.
"""
import colorsys
import os
import sys

from PIL import Image, ImageDraw, ImageFilter

OUT_W, OUT_H = 1200, 1600     # 3:4, matching the card and the assets being replaced
SS = 4                        # supersample factor


def _lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _hsv(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _background(draw, w, h, top, bottom):
    # Diagonal-ish gradient: pure vertical reads flat at this size.
    for y in range(h):
        draw.line([(0, y), (w, y)], fill=_lerp(top, bottom, y / max(h - 1, 1)))


def _monitor(draw, cx, cy, w, h, stroke, colour, radius):
    """Rounded monitor body with a stand, stroked rather than filled."""
    left, top = cx - w / 2, cy - h / 2
    draw.rounded_rectangle([left, top, left + w, top + h], radius=radius,
                           outline=colour, width=stroke)
    neck_w, neck_h = w * 0.10, h * 0.13
    draw.rectangle([cx - neck_w / 2, top + h, cx + neck_w / 2, top + h + neck_h], fill=colour)
    foot_w = w * 0.34
    draw.rounded_rectangle([cx - foot_w / 2, top + h + neck_h,
                            cx + foot_w / 2, top + h + neck_h + stroke * 1.6],
                           radius=stroke, fill=colour)


def _screen_glow(base, cx, cy, w, h, radius, tint):
    """Lit glass: a vertical wash plus one soft diagonal sheen, clipped to the panel.

    A concentric stepped fill was tried first and read as a blurred rectangular blob with visible
    banding — the eye sees the steps, not a screen. A gradient wash with a Gaussian-blurred sheen
    has no steps to see.
    """
    left, top = cx - w / 2, cy - h / 2

    panel = Image.new("RGB", (int(w), int(h)))
    pd = ImageDraw.Draw(panel)
    for y in range(int(h)):
        t = y / max(h - 1, 1)
        # Brightest just above centre, falling off toward both edges.
        k = 1.0 - abs(t - 0.42) * 1.7
        pd.line([(0, y), (w, y)], fill=_lerp((6, 8, 14), tint, max(k, 0.0) ** 1.5))

    sheen = Image.new("L", (int(w), int(h)), 0)
    ImageDraw.Draw(sheen).polygon(
        [(0, h * 0.92), (w * 0.62, 0), (w * 0.92, 0), (0, h * 1.30)], fill=64)
    sheen = sheen.filter(ImageFilter.GaussianBlur(radius=w * 0.05))
    panel.paste(Image.new("RGB", panel.size, (255, 255, 255)), (0, 0), sheen)

    mask = Image.new("L", (int(w), int(h)), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    base.paste(panel, (int(left), int(top)), mask)


def desktop(path):
    w, h = OUT_W * SS, OUT_H * SS
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    _background(d, w, h, _hsv(212, 0.55, 0.26), _hsv(228, 0.68, 0.11))

    cx, cy = w / 2, h * 0.44
    mw, mh = w * 0.56, h * 0.30
    _screen_glow(img, cx, cy, mw, mh, h * 0.016, _hsv(199, 0.62, 0.52))
    d = ImageDraw.Draw(img)
    _monitor(d, cx, cy, mw, mh, int(w * 0.018), (255, 255, 255), int(h * 0.018))
    img.resize((OUT_W, OUT_H), Image.LANCZOS).save(path, optimize=True)


def virtual_desktop(path):
    """A second, offset panel behind the real one: an added screen that is not physically there."""
    w, h = OUT_W * SS, OUT_H * SS
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    _background(d, w, h, _hsv(268, 0.52, 0.28), _hsv(284, 0.66, 0.12))

    cx, cy = w / 2, h * 0.44
    mw, mh = w * 0.52, h * 0.28
    off = w * 0.055

    # Ghost panel first, dimmer and behind.
    ghost = _hsv(280, 0.30, 0.62)
    d.rounded_rectangle([cx - mw / 2 + off, cy - mh / 2 - off,
                         cx + mw / 2 + off, cy + mh / 2 - off],
                        radius=int(h * 0.016), outline=ghost, width=int(w * 0.012))

    _screen_glow(img, cx - off, cy + off, mw, mh, h * 0.016, _hsv(276, 0.55, 0.55))
    d = ImageDraw.Draw(img)
    _monitor(d, cx - off, cy + off, mw, mh, int(w * 0.018), (255, 255, 255), int(h * 0.018))
    img.resize((OUT_W, OUT_H), Image.LANCZOS).save(path, optimize=True)


def desktop_alt(path):
    """Same subject as `desktop`, cooler and flatter, for hosts that prefer the alternate."""
    w, h = OUT_W * SS, OUT_H * SS
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    _background(d, w, h, _hsv(190, 0.42, 0.30), _hsv(205, 0.58, 0.13))

    cx, cy = w / 2, h * 0.44
    mw, mh = w * 0.56, h * 0.30
    _screen_glow(img, cx, cy, mw, mh, h * 0.016, _hsv(186, 0.55, 0.50))
    d = ImageDraw.Draw(img)
    _monitor(d, cx, cy, mw, mh, int(w * 0.018), (255, 255, 255), int(h * 0.018))
    img.resize((OUT_W, OUT_H), Image.LANCZOS).save(path, optimize=True)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    desktop(os.path.join(out, "desktop.png"))
    desktop_alt(os.path.join(out, "desktop-alt.png"))
    virtual_desktop(os.path.join(out, "virtual_desktop.png"))
    print("wrote desktop.png, desktop-alt.png, virtual_desktop.png to " + out)
