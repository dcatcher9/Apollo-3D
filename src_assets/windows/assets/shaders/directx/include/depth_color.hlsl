#ifndef DEPTH_COLOR_HLSL
#define DEPTH_COLOR_HLSL

// Depth models and the guided filter are calibrated on display-referred sRGB. HDR desktop
// capture is linear scRGB (Rec.709 primaries, 1.0 = 80 nits), so compress its absolute luminance
// before applying the sRGB OETF. A luminance-preserving Reinhard scale avoids the hue shifts of
// applying x/(1+x) independently to R, G and B. The final uniform peak scale maps an occasional
// out-of-sRGB-gamut highlight without changing chromaticity.
float3 DepthLinearToSrgb(float3 c) {
    c = saturate(c);
    float3 lo = c * 12.92f;
    float3 hi = 1.055f * pow(c, 1.0f / 2.4f) - 0.055f;
    return (c <= 0.0031308f) ? lo : hi;
}

float3 DepthHdrScRgbToSrgb(float3 c) {
    c = max(c, 0.0f);  // map Rec.709-out-of-gamut negative components into the model gamut
    float luminance = max(dot(c, float3(0.2126f, 0.7152f, 0.0722f)), 0.0f);
    c /= 1.0f + luminance;
    float peak = max(c.r, max(c.g, c.b));
    c /= max(peak, 1.0f);
    return DepthLinearToSrgb(c);
}

float3 DepthColorToSrgb(float3 c, uint color_mode) {
    if (color_mode == 2u) return DepthHdrScRgbToSrgb(c);
    if (color_mode == 1u) return DepthLinearToSrgb(c);
    return saturate(c);
}

#endif
