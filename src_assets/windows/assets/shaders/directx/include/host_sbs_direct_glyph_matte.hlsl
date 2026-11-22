#ifndef HOST_SBS_DIRECT_GLYPH_MATTE_HLSL
#define HOST_SBS_DIRECT_GLYPH_MATTE_HLSL

// The transport publishes native Q8 at scale 1 and float opacity bits at 2 or 4.
bool DirectScoreReady(uint scale) { return scale == 1u || scale == 2u || scale == 4u; }

uint DirectNativeAlpha(Texture2D<uint> candidate, uint2 pixel, uint2 size, uint scale) {
    if (scale == 1u) return candidate.Load(int3(pixel, 0));
    int2 last = int2((size + scale - 1u) / scale) - 1;
    float2 p = (float2(pixel) + 0.5f) / float(scale) - 0.5f;
    int2 lower = int2(floor(p));
    int2 first = clamp(lower, 0, last);
    int2 second = clamp(lower + 1, 0, last);
    float2 fraction = p - float2(lower);
    float a = asfloat(candidate.Load(int3(first, 0)));
    float b = asfloat(candidate.Load(int3(second.x, first.y, 0)));
    float c = asfloat(candidate.Load(int3(first.x, second.y, 0)));
    float d = asfloat(candidate.Load(int3(second, 0)));
    return uint(round(255.0f * lerp(lerp(a, b, fraction.x), lerp(c, d, fraction.x), fraction.y)));
}

#endif
