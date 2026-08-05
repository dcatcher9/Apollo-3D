Texture2D<float4> InputTexture : register(t0);
RWStructuredBuffer<float> OutputBuffer : register(u0);
RWStructuredBuffer<float> OutputAppearanceOrdinal : register(u1);

#include "include/depth_constants.hlsl"
#include "include/depth_color.hlsl"

// Integrate the source texel cells covered by one target texel. The capture texture exposes only
// mip 0, so one bilinear SampleLevel tap aliases whenever the source is appreciably larger than
// the model grid (3840x2160 -> 770x434 is almost 5x in each axis). Exact overlap weights preserve
// thin features that sparse quadrant taps can miss at that ratio.
float3 LoadModelColor(int2 source_pixel, uint2 source_size) {
    source_pixel = clamp(source_pixel, int2(0, 0), int2(source_size) - 1);
    return DepthColorToSrgb(InputTexture.Load(int3(source_pixel, 0)).rgb, color_mode);
}

float3 SampleModelColorBilinear(float2 center_uv, uint2 source_size) {
    float2 source_position = center_uv * float2(source_size) - 0.5f;
    int2 lo = int2(floor(source_position));
    float2 blend = frac(source_position);
    float3 top = lerp(
        LoadModelColor(lo, source_size),
        LoadModelColor(lo + int2(1, 0), source_size),
        blend.x);
    float3 bottom = lerp(
        LoadModelColor(lo + int2(0, 1), source_size),
        LoadModelColor(lo + int2(1, 1), source_size),
        blend.x);
    return lerp(top, bottom, blend.y);
}

float3 SampleModelFootprint(uint2 target_pixel, uint2 source_size, float2 center_uv) {
    uint2 target_size = uint2(target_w, target_h);
    if (any(source_size < target_size)) {
        // The authenticated production shapes never upscale. Retain pixel-center bilinear
        // behavior for diagnostic/native edge cases, but interpolate in model (display-referred)
        // color so capture transfer cannot change the result.
        return SampleModelColorBilinear(center_uv, source_size);
    }

    float2 source_scale =
        float2(source_size) / float2(target_size);
    float2 source_lo = float2(target_pixel) * source_scale;
    float2 source_hi = float2(target_pixel + 1u) * source_scale;
    int2 first = max(int2(floor(source_lo)), int2(0, 0));
    int2 end = min(int2(ceil(source_hi)), int2(source_size));

    float3 weighted_sum = 0.0f;
    [loop]
    for (int source_y = first.y; source_y < end.y; ++source_y) {
        float y_coverage = max(
            min(source_hi.y, (float)(source_y + 1)) -
                max(source_lo.y, (float)source_y),
            0.0f);
        [loop]
        for (int source_x = first.x; source_x < end.x; ++source_x) {
            float x_coverage = max(
                min(source_hi.x, (float)(source_x + 1)) -
                    max(source_lo.x, (float)source_x),
                0.0f);
            // The model was trained on display-referred images resized in encoded color. Apply
            // linear/HDR conversion to every source cell before spatial integration; averaging
            // capture-domain linear light first makes isolated HDR highlights contaminate an
            // entire model texel and makes equivalent SDR capture modes disagree.
            weighted_sum += LoadModelColor(
                int2(source_x, source_y), source_size) * (x_coverage * y_coverage);
        }
    }

    float footprint_area =
        max((source_hi.x - source_lo.x) * (source_hi.y - source_lo.y), 1e-6f);
    return weighted_sum / footprint_area;
}

// Compute shader to area-resize RGB interleaved image to NCHW Float32 with ImageNet normalization.
[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= target_w || DTid.y >= target_h)
        return;

    // Calculate normalized UV coordinates based on target dimensions (center of the pixel)
    float2 uv = float2((DTid.x + 0.5f) / (float)target_w, (DTid.y + 0.5f) / (float)target_h);
    uint base_idx = DTid.y * target_w + DTid.x;

    uint source_w, source_h;
    InputTexture.GetDimensions(source_w, source_h);
    uint2 source_size = uint2(source_w, source_h);
    float3 pixel = SampleModelFootprint(DTid.xy, source_size, uv);

    // Preserve one exposure-ordinal signal independently from the model's HDR tonemapper.
    //
    // Point sampling is deliberate. A nonlinear exposure curve followed by spatial mixing can
    // also reverse the ordering of two mixtures even when the curve is globally monotone. At a
    // fixed capture texel, however, maxRGB commutes with an identical monotone channel curve; a
    // pair can retain its order or collapse to a tie, never reverse it. The later census requires
    // reliable contrast in both frames, so ties abstain.
    uint2 source_point = min(
        uint2(uv * float2(source_size)),
        source_size - 1u);
    float3 capture_rgb = InputTexture.Load(int3(source_point, 0)).rgb;
    OutputAppearanceOrdinal[base_idx] =
        DepthAppearanceOrdinal(capture_rgb, color_mode);

    // ImageNet Normalization
    float r = (pixel.r - 0.485f) / 0.229f;
    float g = (pixel.g - 0.456f) / 0.224f;
    float b = (pixel.b - 0.406f) / 0.225f;

    // NCHW Layout mapping: Output shape is [1, 3, target_h, target_w]
    uint channel_stride = target_w * target_h;

    // Write R, G, B channels
    OutputBuffer[base_idx] = r;
    OutputBuffer[base_idx + channel_stride] = g;
    OutputBuffer[base_idx + 2 * channel_stride] = b;
}
