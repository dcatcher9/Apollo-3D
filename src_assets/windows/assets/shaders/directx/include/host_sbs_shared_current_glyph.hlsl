// Current shared-warp source authority. No legacy selector, alternate UI map,
// background reconstruction or replacement RGB is defined in this owner.
#ifndef HOST_SBS_SHARED_CURRENT_GLYPH_HLSL
#define HOST_SBS_SHARED_CURRENT_GLYPH_HLSL

#include "depth_color.hlsl"

Texture2D<uint> NativeGlyphMatte : register(t3);
Texture2D<float4> CompletedDisplayBand : register(t4);
StructuredBuffer<uint> SubtitleCondition : register(t5);
StructuredBuffer<uint> ContourIdentity : register(t6);

cbuffer HostSbsSubtitleRender : register(b3) {
    uint4 subtitle_render_control; // enabled, color mode, analysis generation lo/hi
    uint4 subtitle_render_source;  // exact analysis source left/top/width/height
};

bool SelectedMatteAuthorized();

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
    NativeGlyphMatte.GetDimensions(matte_size.x,matte_size.y);
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
    if(any(local<0)||any(local>=size))return false;
    uint matte=NativeGlyphMatte.Load(int3(local,0));
    if(matte!=1u)return false;
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

// Exact current selected receipt, source identity and optional queued token guard.
StructuredBuffer<uint> SelectedMatteReceipt : register(t7);
bool SelectedMatteAuthorized() {
    uint n,s;SelectedMatteReceipt.GetDimensions(n,s);
    if(n!=64u||s!=4u||SelectedMatteReceipt[0u]!=0x31534d53u||SelectedMatteReceipt[1u]!=1u||
       SelectedMatteReceipt[2u]<1u||SelectedMatteReceipt[2u]>3u||SelectedMatteReceipt[3u]!=0u||
       !((SelectedMatteReceipt[62u]==1u)||(SelectedMatteReceipt[62u]==2u&&SelectedMatteReceipt[2u]==3u))||SelectedMatteReceipt[63u]!=0u)return false;
    [loop]for(uint i=0u;i<32u;++i)if(SelectedMatteReceipt[28u+i]!=ContourIdentity[i])return false;
    // Exact current contourCB geometry/model/clock mirrors; receipt itself grants no presence.
    [loop]for(uint j=0u;j<12u;++j)if(SelectedMatteReceipt[4u+j]!=ContourIdentity[8u+j])return false;
    [unroll]for(uint k=0u;k<4u;++k)if(SelectedMatteReceipt[16u+k]!=ContourIdentity[4u+k])return false;
    if(any(uint2(SelectedMatteReceipt[20u],SelectedMatteReceipt[21u])!=uint2(ContourIdentity[20u],ContourIdentity[21u]))||
       SelectedMatteReceipt[22u]>1u||SelectedMatteReceipt[23u]!=0u||
       any(uint4(SelectedMatteReceipt[24u],SelectedMatteReceipt[25u],SelectedMatteReceipt[26u],SelectedMatteReceipt[27u])!=uint4(1u,27u,0u,0u))||
       all(uint2(SelectedMatteReceipt[60u],SelectedMatteReceipt[61u])==0u))return false;
#if defined(HOST_SBS_SHARED_PIN_QUEUED)
    // Existing QRG separately proves post-child completion/order, work, token and owner clock.
    if(queued_control.x!=1u||any(uint2(SelectedMatteReceipt[60u],SelectedMatteReceipt[61u])!=queued_root.xy))return false;
#endif
    return true;
}

float4 BackgroundOriginalSample(float2 uv) {
    uv.x=saturate(uv.x);return SourceColor.Sample(LinearSampler,uv);
}
#endif
