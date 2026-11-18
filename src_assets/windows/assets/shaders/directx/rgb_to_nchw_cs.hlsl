Texture2D<float4> InputTexture : register(t0);
RWStructuredBuffer<float> OutputBuffer : register(u0);
RWTexture2D<uint> OutputTensorExclusion : register(u1);

#include "include/depth_constants.hlsl"
#include "include/depth_color.hlsl"
#include "include/depth_source_region.hlsl"

// Integrate the source texel cells covered by one target texel. The capture texture exposes only
// mip 0, so one bilinear SampleLevel tap aliases whenever the source is appreciably larger than
// the model grid (3840x2160 -> 770x434 is almost 5x in each axis). Exact overlap weights preserve
// thin features that sparse quadrant taps can miss at that ratio.
float3 LoadModelColorDirect(int2 source_pixel, uint2 source_size) {
    int2 physical_pixel = DepthSourceRegionLoadPosition(source_pixel, source_size);
    return DepthColorToSrgb(InputTexture.Load(int3(physical_pixel, 0)).rgb, color_mode);
}

// This side length fits alongside the fused comparison's 4 KiB group array.
static const uint ModelSourceTileSide = 48u;
groupshared float3 ModelSourceTile[ModelSourceTileSide * ModelSourceTileSide];

float3 LoadModelColor(int2 source_pixel, uint2 source_size, bool use_tile, int2 tile_origin) {
    if (use_tile) {
        // The area loop clips its footprint to the source before using the shared tile.
        uint2 offset = uint2(source_pixel - tile_origin);
        return ModelSourceTile[offset.y * ModelSourceTileSide + offset.x];
    }
    return LoadModelColorDirect(source_pixel, source_size);
}

float3 SampleModelColorBilinear(float2 center_uv, uint2 source_size) {
    float2 source_position = center_uv * float2(source_size) - 0.5f;
    int2 lo = int2(floor(source_position));
    float2 blend = frac(source_position);
    float3 top = lerp(
        LoadModelColorDirect(lo, source_size),
        LoadModelColorDirect(lo + int2(1, 0), source_size),
        blend.x);
    float3 bottom = lerp(
        LoadModelColorDirect(lo + int2(0, 1), source_size),
        LoadModelColorDirect(lo + int2(1, 1), source_size),
        blend.x);
    return lerp(top, bottom, blend.y);
}

float3 SampleModelFootprint(uint2 target_pixel, uint2 source_size, float2 center_uv, bool use_tile, int2 tile_origin) {
    uint2 target_size = uint2(target_w, target_h);
    if (any(source_size < target_size)) {
        // Upsampling uses pixel-center bilinear interpolation in display-referred model color
        // so capture transfer cannot change the result.
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
                int2(source_x, source_y), source_size, use_tile, tile_origin) * (x_coverage * y_coverage);
        }
    }

    float footprint_area =
        max((source_hi.x - source_lo.x) * (source_hi.y - source_lo.y), 1e-6f);
    return weighted_sum / footprint_area;
}

float2 LetterboxSourceUv(float2 model_uv) {
    float2 content_min = DepthAnalysisContentMinUv();
    return saturate((model_uv - content_min) / DepthAnalysisContentScale());
}

float3 SampleLetterboxedModelFootprint(uint2 target_pixel, uint2 source_size, bool use_tile, int2 tile_origin) {
    float2 target_size = float2(target_w, target_h);
    float2 content_min = DepthAnalysisContentMinUv();
    float2 content_scale = DepthAnalysisContentScale();
    float2 source_lo =
        ((float2(target_pixel) / target_size) - content_min) /
        content_scale * float2(source_size);
    float2 source_hi =
        ((float2(target_pixel + 1u) / target_size) - content_min) /
        content_scale * float2(source_size);
    float2 source_extent = source_hi - source_lo;
    float2 center_uv = LetterboxSourceUv(
        (float2(target_pixel) + 0.5f) / target_size);
    if (any(float2(source_size) < target_size * content_scale)) {
        return SampleModelColorBilinear(center_uv, source_size);
    }

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
            weighted_sum += LoadModelColor(
                int2(source_x, source_y), source_size, use_tile, tile_origin) * (x_coverage * y_coverage);
        }
    }
    return weighted_sum / max(source_extent.x * source_extent.y, 1e-6f);
}

// Compute shader to area-resize RGB interleaved image to NCHW Float32 with ImageNet normalization.
void WriteModelInput(uint3 DTid, bool use_tile, int2 tile_origin) {
    if (DTid.x >= target_w || DTid.y >= target_h)
        return;

    // Calculate normalized UV coordinates based on target dimensions (center of the pixel)
    float2 uv = float2((DTid.x + 0.5f) / (float)target_w, (DTid.y + 0.5f) / (float)target_h);
    uint base_idx = DTid.y * target_w + DTid.x;

    uint texture_w, texture_h;
    InputTexture.GetDimensions(texture_w, texture_h);
    uint2 texture_size = uint2(texture_w, texture_h);
    bool source_region_valid = DepthSourceRegionValid(texture_size);
    if (!source_region_valid) {
        uint invalid_stride = target_w * target_h;
        OutputBuffer[base_idx] = 0.0f;
        OutputBuffer[base_idx + invalid_stride] = 0.0f;
        OutputBuffer[base_idx + 2u * invalid_stride] = 0.0f;
        OutputTensorExclusion[DTid.xy] = 1u;
        return;
    }
    uint2 source_size = DepthSourceRegionSize();
    bool content_valid = DepthAnalysisContentValid();
    bool content_cell = content_valid && DepthAnalysisCellIsContent(DTid.xy);
    bool full_content = content_valid && analysis_content_left == 0u &&
        analysis_content_top == 0u && analysis_content_right == target_w &&
        analysis_content_bottom == target_h;
    float3 pixel = content_valid ?
        (full_content ? SampleModelFootprint(DTid.xy, source_size, uv, use_tile, tile_origin) :
         (content_cell ? SampleLetterboxedModelFootprint(DTid.xy, source_size, use_tile, tile_origin) :
                         SampleLetterboxedModelFootprint(
                             DepthAnalysisClampCell(DTid.xy), source_size, use_tile, tile_origin))) :
        float3(0.0f, 0.0f, 0.0f);
    OutputTensorExclusion[DTid.xy] = content_cell ? 0u : 1u;

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

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    WriteModelInput(DTid, false, int2(0, 0));
}
