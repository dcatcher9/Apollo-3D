"""Reference sampling for Direct Glyph training labels and offline inference on expanded OCR boxes."""

import numpy as np


def _area_axis(values, axis, source_unit, target_unit):
    pixels = np.moveaxis(values, axis, 0)
    size = len(pixels)
    count = (size * target_unit + source_unit - 1) // source_unit
    edges = np.arange(count + 1, dtype=np.int64) * source_unit
    whole, remainder = np.divmod(edges, target_unit)
    dtype = np.int64 if pixels.dtype.kind in 'ui' else np.float64
    integral = np.zeros((size + 1, *pixels.shape[1:]), dtype=dtype)
    np.cumsum(pixels, axis=0, dtype=dtype, out=integral[1:])
    # The final cell may extend a fraction of a pixel beyond the crop.
    remainder += (whole - np.minimum(whole, size)) * target_unit
    remainder = remainder.reshape((-1,) + (1,) * (pixels.ndim - 1))
    samples = integral[np.minimum(whole, size)] * target_unit + pixels[np.minimum(whole, size - 1)] * remainder
    return np.moveaxis(np.diff(samples, axis=0), 0, axis)


def normalize_region(region, target_height):
    """Use the complete extended crop as the coordinate system for RGB or aligned alpha labels."""
    if region.ndim not in (2, 3) or min(region.shape[:2]) < 1 or target_height < 1:
        raise ValueError('Expected a nonempty region and a positive target height')
    height = region.shape[0]
    if height == target_height:
        return region
    sums = _area_axis(_area_axis(region, 0, height, target_height), 1, height, target_height)
    area = height * height
    if region.dtype.kind in 'ui':
        return ((sums + area // 2) // area).astype(region.dtype)
    return (sums / area).astype(region.dtype)


def _native_axis(size, height, target_height, limit):
    numerator = (2 * np.arange(size, dtype=np.int64) + 1) * target_height - height
    low, remainder = np.divmod(numerator, 2 * height)
    return low.clip(0, limit - 1), (low + 1).clip(0, limit - 1), remainder.astype(np.float32) / (2 * height)


def normalized_tiles(normalized, tile_width, tile_height):
    """Yield the owned horizontal interval, its crop slices, and the edge-padded input tile."""
    target_height, width = normalized.shape[:2]
    halo = 36
    body = tile_width - 2 * halo
    if tile_width % 4 or body < 4 or tile_height < target_height:
        raise ValueError('Tile dimensions cannot accommodate the normalized crop')
    top = -4 * ((tile_height - target_height) // 8)
    ys = (top + np.arange(tile_height)).clip(0, target_height - 1)
    for start in range(0, width, body):
        origin = start - halo
        xs = (origin + np.arange(tile_width)).clip(0, width - 1)
        stop = min(width, start + body)
        owned = (slice(-top, target_height - top), slice(halo, halo + stop - start))
        yield start, stop, owned, normalized[ys[:, None], xs]


def score_region(rgb, target_height, infer, tile_width=512, tile_height=144):
    """Score one expanded box, tile long lines, and return native-size opacity and tile count."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected an RGB8 crop')
    normalized = normalize_region(rgb, target_height)
    width = normalized.shape[1]
    scores = np.empty((target_height, width), np.float32)
    tiles = 0
    for start, stop, owned, tile in normalized_tiles(normalized, tile_width, tile_height):
        tensor = np.ascontiguousarray(tile.transpose(2, 0, 1)[None], dtype=np.float32)
        logits = infer(tensor / np.float32(255))[0]
        odds = np.exp(logits - logits.max(axis=0, keepdims=True))
        ink = (odds[1] + odds[2]) + odds[3]
        alpha = ink / (odds[0] + ink)
        scores[:, start:stop] = alpha[owned]
        tiles += 1
    h, w = rgb.shape[:2]
    x0, x1, wx = _native_axis(w, h, target_height, width)
    y0, y1, wy = _native_axis(h, h, target_height, target_height)
    horizontal = scores[:, x0] + wx * (scores[:, x1] - scores[:, x0])
    native = horizontal[y0] + wy[:, None] * (horizontal[y1] - horizontal[y0])
    return np.rint(native * 255).astype(np.uint8), tiles
