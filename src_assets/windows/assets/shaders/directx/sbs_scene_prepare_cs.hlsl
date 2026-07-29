Texture2D<float4> InputTexture : register(t0);
RWStructuredBuffer<float> SceneRgb : register(u0);
RWStructuredBuffer<float> SceneOrdinal : register(u1);
SamplerState LinearSampler : register(s0);

#include "include/sbs_scene_controller_constants.hlsl"
#include "include/depth_color.hlsl"

float4 SampleSourceFootprint(
    uint2 viewport_pixel,
    uint2 viewport_size,
    uint2 source_size,
    float2 center_uv)
{
    if (any(source_size < viewport_size)) {
        return InputTexture.SampleLevel(LinearSampler, center_uv, 0);
    }

    float2 source_scale = float2(source_size) / float2(viewport_size);
    float2 source_lo = float2(viewport_pixel) * source_scale;
    float2 source_hi = float2(viewport_pixel + 1u) * source_scale;
    int2 first = max(int2(floor(source_lo)), int2(0, 0));
    int2 end = min(int2(ceil(source_hi)), int2(source_size));

    float4 weighted_sum = 0.0f;
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
            weighted_sum +=
                InputTexture.Load(int3(source_x, source_y, 0)) *
                (x_coverage * y_coverage);
        }
    }

    float area = max(
        (source_hi.x - source_lo.x) * (source_hi.y - source_lo.y),
        1e-6f);
    return weighted_sum / area;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    const uint canvas = SBS_SCENE_APPEARANCE_CANVAS_SIZE;
    if (dispatch_id.x >= canvas || dispatch_id.y >= canvas) {
        return;
    }

    uint2 pixel = dispatch_id.xy;
    float2 viewport_uv;
    bool valid = SceneViewportUv(pixel, canvas, viewport_uv);
    uint base = pixel.y * canvas + pixel.x;
    uint plane = canvas * canvas;
    if (!valid) {
        SceneRgb[base] = 0.0f;
        SceneRgb[base + plane] = 0.0f;
        SceneRgb[base + 2u * plane] = 0.0f;
        SceneOrdinal[base] = 0.0f;
        return;
    }

    uint2 source_size;
    InputTexture.GetDimensions(source_size.x, source_size.y);
    uint2 viewport_origin;
    uint2 viewport_size;
    SceneViewport(canvas, viewport_origin, viewport_size);
    uint2 viewport_pixel = pixel - viewport_origin;
    float4 source_pixel = SampleSourceFootprint(
        viewport_pixel,
        viewport_size,
        source_size,
        viewport_uv);
    bool source_pixel_finite =
        !any(isnan(source_pixel)) && !any(isinf(source_pixel));
    if (!source_pixel_finite) {
        const float invalid = asfloat(0x7fc00000u);
        SceneRgb[base] = invalid;
        SceneRgb[base + plane] = invalid;
        SceneRgb[base + 2u * plane] = invalid;
        SceneOrdinal[base] = invalid;
        return;
    }
    float3 display_rgb = DepthColorToSrgb(source_pixel.rgb, scene_color_mode);
    SceneRgb[base] = display_rgb.r;
    SceneRgb[base + plane] = display_rgb.g;
    SceneRgb[base + 2u * plane] = display_rgb.b;

    // Keep the ordinal before HDR tone mapping and spatial mixing. At a fixed capture texel,
    // max(R,G,B) cannot reverse its ordering under one monotone per-channel exposure curve.
    uint2 source_point = min(
        uint2(viewport_uv * float2(source_size)),
        source_size - 1u);
    float3 capture_rgb = InputTexture.Load(int3(source_point, 0)).rgb;
    SceneOrdinal[base] =
        any(isnan(capture_rgb)) || any(isinf(capture_rgb)) ?
            asfloat(0x7fc00000u) :
            max(capture_rgb.r, max(capture_rgb.g, capture_rgb.b));
}
