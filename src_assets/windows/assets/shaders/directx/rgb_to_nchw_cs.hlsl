Texture2D<float4> InputTexture : register(t0);
StructuredBuffer<uint4> FrameRoiTransform : register(t1);
RWStructuredBuffer<float> OutputBuffer : register(u0);
RWStructuredBuffer<float> OutputAppearanceOrdinal : register(u1);
SamplerState LinearSampler : register(s0);

#include "include/depth_constants.hlsl"
#include "include/depth_color.hlsl"
#include "include/sbs_frame_roi_transform.hlsl"

// Integrate the source texel cells covered by one target texel. The capture texture exposes only
// mip 0, so one bilinear SampleLevel tap aliases whenever the source is appreciably larger than
// the model grid (3840x2160 -> 770x434 is almost 5x in each axis). Exact overlap weights preserve
// thin features that sparse quadrant taps can miss at that ratio.
float4 SampleModelFootprint(uint2 target_pixel, uint2 source_size, float2 center_uv) {
    uint2 target_size = uint2(target_w, target_h);
    if (any(source_size < target_size)) {
        // The normal estimator contract never upscales, but retain the old pixel-center bilinear
        // behavior for a capped/native edge case rather than applying a box smaller than a texel.
        return InputTexture.SampleLevel(LinearSampler, center_uv, 0);
    }

    float2 source_scale =
        float2(source_size) / float2(target_size);
    float2 source_lo = float2(target_pixel) * source_scale;
    float2 source_hi = float2(target_pixel + 1u) * source_scale;
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

    float footprint_area =
        max((source_hi.x - source_lo.x) * (source_hi.y - source_lo.y), 1e-6f);
    return weighted_sum / footprint_area;
}

// Exact-area sampling for an active transform. ABI v1 never pads the model tensor: the builder
// expands and shifts one enclosing source crop until its physical aspect matches target_w/target_h,
// so every model texel integrates real source pixels. Keeping this separate from
// SampleModelFootprint preserves the canonical full-frame arithmetic byte-for-byte.
float4 SampleRoiModelFootprint(
    SbsFrameRoiTransformData transform,
    uint2 target_pixel,
    uint2 source_size,
    float2 model_center_uv)
{
    float4 crop = FrameRoiDataCrop(transform);
    float2 crop_extent = crop.zw - crop.xy;
    float2 model_lo =
        float2(target_pixel) / float2(target_w, target_h);
    float2 model_hi =
        float2(target_pixel + 1u) / float2(target_w, target_h);
    float2 source_lo =
        (crop.xy + model_lo * crop_extent) * float2(source_size);
    float2 source_hi =
        (crop.xy + model_hi * crop_extent) * float2(source_size);
    source_lo = clamp(source_lo, 0.0f.xx, float2(source_size));
    source_hi = clamp(source_hi, source_lo, float2(source_size));

    float2 source_footprint = source_hi - source_lo;
    float2 source_center_uv;
    FrameRoiDataModelToSourceUv(
        transform,
        model_center_uv,
        source_center_uv);
    if (any(source_footprint < 1.0f.xx)) {
        // An ROI may devote more model texels than the source crop contains. Preserve the
        // historical upscaling behavior (pixel-center bilinear) rather than turning a sub-texel
        // box into nearest-neighbor replication.
        return InputTexture.SampleLevel(
            LinearSampler,
            saturate(source_center_uv),
            0);
    }

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

    float footprint_area =
        max(source_footprint.x * source_footprint.y, 1e-6f);
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
    SbsFrameRoiTransformData transform = FrameRoiTransformLoad();
    uint channel_stride = target_w * target_h;
    bool active_transform =
        FrameRoiDataActive(transform) &&
        all(FrameRoiDataSourceDimensions(transform) == source_size) &&
        all(FrameRoiDataModelDimensions(transform) ==
            uint2(target_w, target_h));
    bool full_transform =
        FrameRoiDataUnboundZero(transform) ||
        (FrameRoiDataCanonicalFullFrame(transform) &&
         all(FrameRoiDataSourceDimensions(transform) == source_size) &&
         all(FrameRoiDataModelDimensions(transform) ==
             uint2(target_w, target_h)));
    bool full_shape_matches_source =
        FrameRoiPhysicalAspectMatches(
            float4(0.0f, 0.0f, 1.0f, 1.0f),
            source_size,
            uint2(target_w, target_h),
            SBS_FRAME_ROI_LEGACY_ASPECT_REL_TOLERANCE);
    if (!active_transform &&
        !(full_transform && full_shape_matches_source)) {
        // A stale/malformed transform paired with an ROI-specific tensor shape must never
        // reinterpret the full source at that shape (for example, stretch 16:9 into a square).
        // Emit a finite neutral tensor, leave the transform invalid, and let the post-process
        // retain the previous depth/transform pair while the shape handshake recovers.
        OutputBuffer[base_idx] = 0.0f;
        OutputBuffer[base_idx + channel_stride] = 0.0f;
        OutputBuffer[base_idx + 2u * channel_stride] = 0.0f;
        OutputAppearanceOrdinal[base_idx] = 0.0f;
        return;
    }
    float4 pixel;
    uint2 source_point;
    if (!active_transform) {
        // Off, shadow, valid full-frame, and the explicit all-zero legacy binding take the
        // literal validated legacy path. Never interpret "non-canonical" as "safe active ROI".
        pixel = SampleModelFootprint(DTid.xy, source_size, uv);
        source_point = min(
            uint2(uv * float2(source_size)),
            source_size - 1u);
    } else {
        pixel = SampleRoiModelFootprint(
            transform,
            DTid.xy,
            source_size,
            uv);
        float2 source_uv;
        bool mapped =
            FrameRoiDataModelToSourceUv(transform, uv, source_uv);
        // Strict validation above makes this true for every dispatched model center. Keep a
        // deterministic legacy-coordinate fallback for defensive shader compilation only.
        source_uv = mapped ? saturate(source_uv) : uv;
        source_point = min(
            uint2(source_uv * float2(source_size)),
            source_size - 1u);
    }

    // Preserve one exposure-ordinal signal BEFORE the model's HDR tonemapper. Reinhard uses a
    // per-pixel luminance divisor, so maxRGB reconstructed from the post-tone-map NCHW tensor is
    // not rank invariant: a global exposure change can reverse two differently coloured pixels.
    //
    // Point sampling is deliberate. A nonlinear exposure curve followed by spatial mixing can
    // also reverse the ordering of two mixtures even when the curve is globally monotone. At a
    // fixed capture texel, however, maxRGB commutes with an identical monotone channel curve; a
    // pair can retain its order or collapse to a tie, never reverse it. The later census requires
    // reliable contrast in both frames, so ties abstain.
    float3 capture_rgb = InputTexture.Load(int3(source_point, 0)).rgb;
    OutputAppearanceOrdinal[base_idx] =
        max(capture_rgb.r, max(capture_rgb.g, capture_rgb.b));

    // HDR capture is scRGB: LINEAR light with Rec.709 primaries (so primaries already match
    // the SDR-trained model; only the transfer function differs). Compress highlights with
    // luminance-preserving Reinhard (1.0 scRGB = 80 nits) so highlights don't blind the first conv,
    // then gamma-encode to sRGB so midtones land where ImageNet normalization expects them
    // (feeding linear light makes the image far too dark and degrades the depth estimate).
    pixel.rgb = DepthColorToSrgb(pixel.rgb, color_mode);

    // ImageNet Normalization
    float r = (pixel.r - 0.485f) / 0.229f;
    float g = (pixel.g - 0.456f) / 0.224f;
    float b = (pixel.b - 0.406f) / 0.225f;

    // NCHW Layout mapping: Output shape is [1, 3, target_h, target_w]
    // Write R, G, B channels
    OutputBuffer[base_idx] = r;
    OutputBuffer[base_idx + channel_stride] = g;
    OutputBuffer[base_idx + 2 * channel_stride] = b;
}
