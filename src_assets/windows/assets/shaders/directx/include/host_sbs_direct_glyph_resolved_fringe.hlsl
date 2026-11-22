#ifndef DIRECT_GLYPH_RESOLVED_FRINGE_HLSL
#define DIRECT_GLYPH_RESOLVED_FRINGE_HLSL
static const uint DirectResolvedFringeFlag = 32u;
static const uint DirectResolvedFringeCutoff = 64u;
static const uint DirectResolvedFringeCapacity = 6u;
ByteAddressBuffer DirectResolvedFringeBounds : register(t22);
StructuredBuffer<int> DirectResolvedPlacement : register(t23);

bool DirectOriginalScoreReady(uint scale, out uint count) {
    uint words, stride;
    DirectResolvedPlacement.GetDimensions(words, stride);
    count = 0u;
    if (words != 4u + 8u * DirectResolvedFringeCapacity || stride != 4u) return false;

    count = uint(DirectResolvedPlacement[0]);
    if (count == 0u || count > DirectResolvedFringeCapacity) return false;

    int scaleWord = DirectResolvedPlacement[10u];
    return ((scale == 1u && scaleWord == 0) || (scale == 2u && scaleWord == 2) ||
            (scale == 4u && scaleWord == 4)) &&
           uint(DirectResolvedPlacement[1]) == (1u << count) - 1u &&
           DirectResolvedPlacement[2] == 0 && DirectResolvedPlacement[3] == 0;
}

bool DirectOwnsResolvedTap(int2 tap, int4 window) {
    return all(tap >= window.xy) && all(tap < window.zw);
}

bool DirectOriginalTapsCovered(uint2 pixel, uint2 size, uint scale, uint count) {
    // Same actual floor and clamping as DirectNativeAlpha. On the native
    // lattice it reads only this pixel. Integer factors 2/4 use four taps.
    int2 cells = int2((size + scale - 1u) / scale);
    int2 low = int2(pixel);
    int2 high = low;
    if (scale != 1u) {
        float2 p = (float2(pixel) + 0.5f) / float(scale) - 0.5f;
        int2 corner = int2(floor(p));
        low = clamp(corner, 0, cells - 1);
        high = clamp(corner + 1, 0, cells - 1);
    }
    uint covered = 0u;
    [loop] for (uint tile = 0u; tile < count; ++tile) {
        uint at = 4u + 8u * tile;
        int4 window = int4(DirectResolvedPlacement[at + 2u], DirectResolvedPlacement[at + 3u],
                           DirectResolvedPlacement[at + 4u], DirectResolvedPlacement[at + 5u]);
        if (any(window.xy < 0) || any(window.zw > cells) || any(window.xy >= window.zw))
            return false;
        covered |= DirectOwnsResolvedTap(low, window) ? 1u : 0u;
        covered |= DirectOwnsResolvedTap(int2(high.x, low.y), window) ? 2u : 0u;
        covered |= DirectOwnsResolvedTap(int2(low.x, high.y), window) ? 4u : 0u;
        covered |= DirectOwnsResolvedTap(high, window) ? 8u : 0u;
    }
    return covered == 15u;
}

bool DirectCurrentResolvedFringeContains(uint2 pixel, uint slots, uint2 token) {
    uint bytes;
    DirectResolvedFringeBounds.GetDimensions(bytes);
    if (bytes != 144u || all(token == 0u) ||
        any(DirectResolvedFringeBounds.Load4(0u) != uint4(1u, slots, token))) return false;
    [loop] for (uint i = 0u; i < slots; ++i) {
        uint4 rect = DirectResolvedFringeBounds.Load4(16u + 16u * i);
        if (all(pixel >= rect.xy) && all(pixel < rect.zw)) return true;
    }
    return false;
}
#endif
