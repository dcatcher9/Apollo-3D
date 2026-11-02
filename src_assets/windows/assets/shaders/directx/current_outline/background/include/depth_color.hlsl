#ifndef DEPTH_COLOR_HLSL
#define DEPTH_COLOR_HLSL

// Depth preprocessing and ownership contour sampling are calibrated on display-referred sRGB.
// HDR desktop capture is linear scRGB (Rec.709 primaries, 1.0 = 80 nits), so compress its luminance
// before applying the sRGB OETF. A luminance-preserving Reinhard scale avoids the hue shifts of
// applying x/(1+x) independently to R, G and B. The final uniform peak scale maps an occasional
// out-of-sRGB-gamut highlight without changing chromaticity.
float3 DepthLinearToSrgb(float3 c) {
    c = saturate(c);
    float3 lo = c * 12.92f;
    float3 hi = 1.055f * pow(c, 1.0f / 2.4f) - 0.055f;
    return (c <= 0.0031308f) ? lo : hi;
}

float3 DepthSrgbToLinear(float3 color) {
    color = saturate(color);
    float3 low = color / 12.92f;
    float3 high = pow((color + 0.055f) / 1.055f, 2.4f);
    return color <= 0.04045f ? low : high;
}

float DepthSrgbToLinear(float value) {
    return DepthSrgbToLinear(float3(value, value, value)).x;
}

// Put the point-sampled max-RGB ordinal into one common scene-linear scale. max() commutes with
// the sRGB transfer because the same monotone curve is applied to every channel, so SDR encoded
// and SDR linear capture now produce the same ordinal. scRGB is already linear (1.0 = 80 nits).
float DepthAppearanceOrdinal(float3 capture_rgb, uint color_mode) {
    float ordinal = max(capture_rgb.r, max(capture_rgb.g, capture_rgb.b));
    ordinal = max(ordinal, 0.0f);
    return color_mode == 0u ? DepthSrgbToLinear(ordinal) : ordinal;
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
