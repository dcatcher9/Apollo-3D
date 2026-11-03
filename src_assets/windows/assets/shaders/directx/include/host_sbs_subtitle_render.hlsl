// Native glyph selection is a color-source selector, not a continuous parallax field.
// Both source candidates preserve V; only the ordinary Base uses a bounded inverse solve.
#include "include/depth_color.hlsl"

Texture2D<uint> NativeGlyphMatte : register(t3);
Texture2D<float4> CompletedDisplayBand : register(t4);
StructuredBuffer<uint> SubtitleCondition : register(t5);
StructuredBuffer<uint> ContourIdentity : register(t6);

cbuffer HostSbsSubtitleRender : register(b3) {
    uint4 subtitle_render_control; // enabled, color mode, analysis generation lo/hi
    uint4 subtitle_render_source;  // exact analysis source left/top/width/height
};

bool SubtitleRenderAuthorized(out uint2 source_size, out float full_source_plane) {
    source_size = 0u;
    full_source_plane = 0.0f;
    // Reject an absent b3 before touching optional resources.
    if (subtitle_render_control.x != 1u || subtitle_render_control.y > 2u ||
        !WarpAuthorized()) return false;
    uint condition_words, condition_stride, identity_words, identity_stride;
    SubtitleCondition.GetDimensions(condition_words, condition_stride);
    ContourIdentity.GetDimensions(identity_words, identity_stride);
    if (condition_words != V2_SUBTITLE_CONDITION_PARAM_WORD_COUNT || condition_stride != 4u ||
        identity_words != V2_SUBTITLE_CONTOUR_HISTORY_WORD_COUNT || identity_stride != 4u)
        return false;
    SourceColor.GetDimensions(source_size.x, source_size.y);
    uint2 analysis_size = subtitle_render_source.zw;
    if (any(source_size == 0u) || any(analysis_size == 0u) ||
        any(analysis_size > V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT) ||
        any(subtitle_render_source.xy > source_size) ||
        any(analysis_size > source_size - subtitle_render_source.xy))
        return false;
    uint2 matte_size, snapshot_size, field_size;
    NativeGlyphMatte.GetDimensions(matte_size.x, matte_size.y);
    CompletedDisplayBand.GetDimensions(snapshot_size.x, snapshot_size.y);
    BaseParallax.GetDimensions(field_size.x, field_size.y);
    uint4 crop = uint4(ContourIdentity[8u], ContourIdentity[9u],
                       ContourIdentity[10u], ContourIdentity[11u]);
    uint4 content = uint4(ContourIdentity[16u], ContourIdentity[17u],
                          ContourIdentity[18u], ContourIdentity[19u]);
    if (ContourIdentity[0u] != V2_SUBTITLE_CONTOUR_SCHEMA ||
        ContourIdentity[1u] != V2_SUBTITLE_CONTOUR_HISTORY_TAG ||
        ContourIdentity[2u] != 1u || ContourIdentity[23u] != 1u ||
        ContourIdentity[22u] != V2_SUBTITLE_CONTOUR_RECIPE ||
        all(uint2(ContourIdentity[4u], ContourIdentity[5u]) == 0u) ||
        any(uint2(ContourIdentity[6u], ContourIdentity[7u]) != subtitle_render_control.zw) ||
        any(crop.xy != analysis_size) || crop.w == 0u ||
        crop.z > crop.y || crop.w > crop.y - crop.z ||
        any(matte_size != uint2(crop.x, crop.w)) || any(snapshot_size != matte_size) ||
        any(field_size != uint2(ContourIdentity[12u], ContourIdentity[13u])) ||
        any(content.xy >= content.zw) || any(content.zw > field_size))
        return false;
    [unroll]
    for (uint word = V2_SUBTITLE_CONTOUR_HISTORY_RESERVED_OFFSET;
         word < V2_SUBTITLE_CONTOUR_HISTORY_WORD_COUNT; ++word) {
        if (ContourIdentity[word] != 0u) return false;
    }
    uint2 owner_clock = uint2(ContourIdentity[20u], ContourIdentity[21u]);
    float4 clock_state = ParallaxState[V2_STATE_VECTOR_GAIN_LAST_OBSERVATION_LOW];
    float4 control_state = ParallaxState[V2_STATE_VECTOR_CONFIRMED_CUT_COUNT];
    if (all(owner_clock == 0u) || any(owner_clock != asuint(clock_state.xy)) ||
        ContourIdentity[3u] != asuint(V2_STATE_CONFIRMED_CUT_COUNT(control_state)))
        return false;
    if (video_roi_active == 1.0f) {
        if (!VideoRoiRectValid() || !VideoRoiContentValid(field_size) ||
            any(content != video_roi_tensor_content_rect))
            return false;
        // Integer source geometry and normalized Base embedding must name the same rectangle.
        float4 expected_roi = float4(subtitle_render_source.xy,
            subtitle_render_source.xy + analysis_size) / float4(source_size, source_size);
        if (any(abs(expected_roi - video_roi_source_uv) * float4(source_size, source_size) > 0.25f))
            return false;
    } else if (any(subtitle_render_source.xy != 0u) || any(analysis_size != source_size)) {
        return false;
    }
    uint count = SubtitleCondition[V2_SUBTITLE_CONDITION_PARAM_CURRENT_COUNT_WORD];
    uint kinds = SubtitleCondition[V2_SUBTITLE_CONDITION_PARAM_CURRENT_KIND_MASK_WORD];
    float plane = asfloat(SubtitleCondition[V2_SUBTITLE_CONDITION_PARAM_TARGET_WORD]);
    if (SubtitleCondition[0u] != V2_SUBTITLE_CONDITION_PARAM_SCHEMA ||
        SubtitleCondition[1u] != V2_SUBTITLE_CONDITION_PARAM_TAG ||
        count == 0u || count > V2_SUBTITLE_PLANE_COVER_CAPACITY ||
        (kinds & ~((1u << count) - 1u)) != 0u ||
        SubtitleCondition[V2_SUBTITLE_CONDITION_PARAM_ACTIVE_WORD] !=
            V2_SUBTITLE_CONDITION_PARAM_FULLY_ACTIVE_VALUE ||
        !isfinite(plane) || plane < 0.0f || plane > V2_DIRECT_CONTAINER_LIMIT)
        return false;
    full_source_plane = plane * ((float)analysis_size.x / (float)source_size.x);
    return true;
}

float3 SubtitleDisplayRoundTrip(float3 native_color) {
    float3 display = DepthColorToSrgb(native_color, subtitle_render_control.y);
    return float3(f16tof32(f32tof16(display.r)),
                  f16tof32(f32tof16(display.g)),
                  f16tof32(f32tof16(display.b)));
}

bool CurrentGlyphContributor(int2 full_pixel) {
    int2 local = full_pixel - int2(subtitle_render_source.xy) -
        int2(0, (int)ContourIdentity[10u]);
    int2 size = int2(ContourIdentity[8u], ContourIdentity[11u]);
    if (any(local < 0) || any(local >= size) ||
        NativeGlyphMatte.Load(int3(local, 0)) != 1u)
        return false;
    float4 reference = CompletedDisplayBand.Load(int3(local, 0));
    float3 current = SourceColor.Load(int3(full_pixel, 0)).rgb;
    if (reference.a != 1.0f || any(!isfinite(reference.rgb)) ||
        any(reference.rgb < 0.0f) || any(reference.rgb > 1.0f) ||
        any(!isfinite(current)))
        return false;
    float3 display = SubtitleDisplayRoundTrip(current);
    return all(isfinite(display)) &&
        all(abs(display - reference.rgb) <= V2_SUBTITLE_CONTOUR_HISTORY_RGB_DISTANCE);
}

bool GlyphBilinearFootprint(float2 candidate_uv, uint2 source_size) {
    // The final color sample saturates U and clamps both axes to the native image boundary.
    // A zero interpolation weight cannot contribute color and must not grow the matte.
    float2 position = saturate(candidate_uv) * float2(source_size) - 0.5f;
    int2 low = int2(floor(position));
    float2 fraction = position - float2(low);
    int2 last = int2(source_size) - 1;
    [unroll]
    for (int y = 0; y < 2; ++y) {
        [unroll]
        for (int x = 0; x < 2; ++x) {
            float weight_x = x == 0 ? 1.0f - fraction.x : fraction.x;
            float weight_y = y == 0 ? 1.0f - fraction.y : fraction.y;
            if (weight_x > 0.0f && weight_y > 0.0f &&
                CurrentGlyphContributor(clamp(low + int2(x, y), int2(0, 0), last)))
                return true;
        }
    }
    return false;
}

float2 SelectSourceUV(float2 destination_uv, float eye_sign) {
    float2 base_uv = ReprojectIfAuthorized(destination_uv, eye_sign);
    if (subtitle_render_control.x != 1u) return base_uv;
    // This early band rejection may only withdraw selection. Full identity validation still
    // follows for every potentially contributing row; malformed/unbound inputs remain Base.
    uint2 native_size;
    SourceColor.GetDimensions(native_size.x, native_size.y);
    uint crop_height = ContourIdentity[11u];
    if (any(native_size == 0u) || crop_height == 0u) return base_uv;
    float native_y = destination_uv.y * (float)native_size.y - 0.5f;
    float crop_top = (float)subtitle_render_source.y + (float)ContourIdentity[10u];
    float crop_bottom = crop_top + (float)crop_height;
    if (native_y <= crop_top - 1.0f || native_y >= crop_bottom) return base_uv;
    uint2 source_size;
    float plane;
    if (!SubtitleRenderAuthorized(source_size, plane)) return base_uv;
    float2 ui_uv = float2(destination_uv.x + eye_sign * plane, destination_uv.y);
    return GlyphBilinearFootprint(base_uv, source_size) ||
           GlyphBilinearFootprint(ui_uv, source_size) ? ui_uv : base_uv;
}
