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

// PRIVATE algorithm2 bridge: original seven operands stay available; t8 is original M.
Texture2D<uint> OriginalLegacyGlyphMatte : register(t8);
bool SelectedMatteAuthorized();

bool SubtitleRenderAuthorized(out uint2 source_size, out float full_source_plane, bool use_selected) {
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
    if(use_selected)NativeGlyphMatte.GetDimensions(matte_size.x,matte_size.y);
    else OriginalLegacyGlyphMatte.GetDimensions(matte_size.x,matte_size.y);
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

bool CurrentGlyphContributor(int2 full_pixel, bool use_selected) {
    int2 local = full_pixel - int2(subtitle_render_source.xy) -
        int2(0, (int)ContourIdentity[10u]);
    int2 size = int2(ContourIdentity[8u], ContourIdentity[11u]);
    if(any(local<0)||any(local>=size))return false;
    uint matte=use_selected?NativeGlyphMatte.Load(int3(local,0)):OriginalLegacyGlyphMatte.Load(int3(local,0));
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

bool GlyphBilinearFootprint(float2 candidate_uv, uint2 source_size, bool use_selected) {
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
                CurrentGlyphContributor(clamp(low + int2(x, y), int2(0, 0), last),use_selected))
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
    bool use_selected=SelectedMatteAuthorized();
    if (!SubtitleRenderAuthorized(source_size,plane,use_selected)) return base_uv;
    float2 ui_uv = float2(destination_uv.x + eye_sign * plane, destination_uv.y);
    return GlyphBilinearFootprint(base_uv,source_size,use_selected) ||
           GlyphBilinearFootprint(ui_uv,source_size,use_selected) ? ui_uv : base_uv;
}


// PRIVATE ABI1/algorithm2 sidecar. Invalid/unavailable receipt selects literal original M t8.
// Valid algorithm1 kinds1/2/3 preserve old behavior. New algorithm2 accepts ONLY kind3.
// Original SCT/SCP/source/clock authorization and native-RGB revocation remain unchanged.
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
#if SELECTED_MATTE_QUEUED
    // Existing QRG separately proves post-child completion/order, work, token and owner clock.
    if(queued_control.x!=1u||any(uint2(SelectedMatteReceipt[60u],SelectedMatteReceipt[61u])!=queued_root.xy))return false;
#endif
    return true;
}

// Private BG1 ABI. UI/protection M remains the original current selected M.
Texture2D<float4> CurrentErasedBackground : register(t9);
StructuredBuffer<uint> CurrentBackgroundProof : register(t11);
cbuffer CurrentBackgroundDrawLease : register(b4) { uint4 BackgroundExpected[4]; };

float4 BackgroundOriginalSample(float2 uv) {
    uv.x=saturate(uv.x);return SourceColor.Sample(LinearSampler,uv);
}
float4 BackgroundOldSelectedSample(float2 uv,float eye_sign) {
    return BackgroundOriginalSample(SelectSourceUV(uv,eye_sign));
}
bool CurrentBackgroundAuthorized(out uint2 source_size) {
    source_size=0u;
    if(!SelectedMatteAuthorized()||SelectedMatteReceipt[2u]!=3u||SelectedMatteReceipt[62u]!=2u)return false;
    uint n,stride;CurrentBackgroundProof.GetDimensions(n,stride);
    if(n!=96u||stride!=4u||CurrentBackgroundProof[0u]!=0x31504742u||CurrentBackgroundProof[1u]!=1u||
       CurrentBackgroundProof[2u]!=1u||CurrentBackgroundProof[3u]!=0u||CurrentBackgroundProof[62u]!=1u||CurrentBackgroundProof[63u]!=0u)return false;
    // Immutable b4 is returned with the same strong input lease by the BG owner. Null b4
    // cannot authenticate; no pointer identity is inferred from pixel values in a shader.
    if(all(BackgroundExpected[0u].xy==0u))return false;
    [unroll]for(uint i=0u;i<16u;++i)if(CurrentBackgroundProof[4u+i]!=BackgroundExpected[i/4u][i%4u])return false;
    [loop]for(uint j=0u;j<32u;++j)if(CurrentBackgroundProof[20u+j]!=ContourIdentity[j])return false;
    [unroll]for(uint k=0u;k<4u;++k)if(CurrentBackgroundProof[52u+k]!=subtitle_render_control[k]||CurrentBackgroundProof[56u+k]!=subtitle_render_source[k])return false;
    if(any(uint2(CurrentBackgroundProof[60u],CurrentBackgroundProof[61u])!=uint2(SelectedMatteReceipt[60u],SelectedMatteReceipt[61u])))return false;
    SourceColor.GetDimensions(source_size.x,source_size.y);
    uint2 bg_size;CurrentErasedBackground.GetDimensions(bg_size.x,bg_size.y);
    if(any(source_size==0u)||any(bg_size!=uint2(ContourIdentity[8u],ContourIdentity[11u]))||
       any(uint2(CurrentBackgroundProof[64u],CurrentBackgroundProof[65u])!=source_size)||
       any(uint2(CurrentBackgroundProof[66u],CurrentBackgroundProof[67u])!=bg_size)||
       CurrentBackgroundProof[68u]!=ContourIdentity[10u]||CurrentBackgroundProof[69u]!=subtitle_render_source.x||
       CurrentBackgroundProof[70u]!=subtitle_render_source.y||CurrentBackgroundProof[71u]!=subtitle_render_control.y)return false;
    if(CurrentBackgroundProof[77u]!=bg_size.x*bg_size.y||CurrentBackgroundProof[80u]!=CurrentBackgroundProof[77u]||CurrentBackgroundProof[81u]!=1u)return false;
    [unroll]for(uint r=83u;r<96u;++r)if(CurrentBackgroundProof[r]!=0u)return false;
    return true;
}

// Outside the native band is untouched. A=-1/unknown never becomes a valid background.
bool LoadCurrentBackground(int2 pixel,out float3 rgb,out bool erased) {
    rgb=0.0f;erased=false;
    int2 local=pixel-int2(subtitle_render_source.xy)-int2(0,(int)ContourIdentity[10u]);
    int2 size=int2(ContourIdentity[8u],ContourIdentity[11u]);
    if(any(local<0)||any(local>=size))return true;
    float4 value=CurrentErasedBackground.Load(int3(local,0));
    if(value.a==0.0f)return true;
    if(value.a!=1.0f||any(!isfinite(value.rgb)))return false;
    erased=true;rgb=value.rgb;return true;
}

// stats=(BaseU,UIU,route,BaseV), 0=Base,1=current-M UI present,2=BG repair,
// 3=literal old selected behavior on missing/stale/unknown background. Route1 RGB
// is the original UI sample at full coverage, or marked UI over repaired Base at
// fractional coverage; BaseU/UIU remain candidate coordinates, not one RGB owner.
float4 SelectPushPullColour(float2 source_uv,float eye_sign,out float4 stats) {
    float2 base_uv=ReprojectIfAuthorized(source_uv,eye_sign);
    stats=float4(saturate(base_uv.x),saturate(base_uv.x),3.0f,base_uv.y);
    // Same horizontal-renderer band early-out as the qualified old selector. Erasure is
    // band-clamped, so no positive Base/UI tap can change colour outside these limits.
    if(subtitle_render_control.x!=1u){stats.z=0.0f;return BackgroundOriginalSample(base_uv);}
    uint2 native_size;SourceColor.GetDimensions(native_size.x,native_size.y);
    uint crop_height=ContourIdentity[11u];
    if(any(native_size==0u)||crop_height==0u){stats.z=0.0f;return BackgroundOriginalSample(base_uv);}
    float native_y=source_uv.y*(float)native_size.y-0.5f;
    float crop_top=(float)subtitle_render_source.y+(float)ContourIdentity[10u];
    if(native_y<=crop_top-1.0f||native_y>=crop_top+(float)crop_height){stats.z=0.0f;return BackgroundOriginalSample(base_uv);}
    uint2 source_size;
    if(!CurrentBackgroundAuthorized(source_size))return BackgroundOldSelectedSample(source_uv,eye_sign);
    uint2 checked_size;float plane;
    if(!SubtitleRenderAuthorized(checked_size,plane,true))return BackgroundOldSelectedSample(source_uv,eye_sign);
    float2 ui_uv=float2(source_uv.x+eye_sign*plane,source_uv.y);stats.y=saturate(ui_uv.x);
    // Only marked native taps belong to the UI layer. Unmarked taps must not carry
    // the shifted UI background into a fractional glyph footprint.
    float2 ui_position=saturate(ui_uv)*float2(source_size)-0.5f;
    int2 ui_low=int2(floor(ui_position)),ui_last=int2(source_size)-1;
    float2 ui_fraction=ui_position-float2(ui_low);
    float ui_coverage=0.0f;float3 ui_premultiplied=0.0f;
    bool ui_any=false,ui_all=true;
    [unroll]for(int uy=0;uy<2;++uy)[unroll]for(int ux=0;ux<2;++ux) {
        float wx=ux==0?1.0f-ui_fraction.x:ui_fraction.x;
        float wy=uy==0?1.0f-ui_fraction.y:ui_fraction.y;
        if(wx<=0.0f||wy<=0.0f)continue;
        int2 pixel=clamp(ui_low+int2(ux,uy),int2(0,0),ui_last);
        if(!CurrentGlyphContributor(pixel,true)){ui_all=false;continue;}
        // Contributor authorization already checked this immutable native RGB for
        // finiteness and current display agreement. Never load an unmarked RGB here:
        // even zero times a nonfinite unmarked source value would contaminate P.
        float weight=wx*wy;float3 rgb=SourceColor.Load(int3(pixel,0)).rgb;
        ui_coverage+=weight;ui_premultiplied+=weight*rgb;ui_any=true;
    }
    // All positive-weight native taps belong to the UI: retain the exact qualified
    // hardware sample, including its alpha, without touching the hidden Base RGB.
    if(ui_any&&ui_all){stats.z=1.0f;return BackgroundOriginalSample(ui_uv);}
    float2 position=saturate(base_uv)*float2(source_size)-0.5f;
    int2 low=int2(floor(position)),last=int2(source_size)-1;
    float2 fraction=position-float2(low);float3 reconstructed=0.0f;bool any_erased=false;
    [unroll]for(int y=0;y<2;++y)[unroll]for(int x=0;x<2;++x) {
        float wx=x==0?1.0f-fraction.x:fraction.x,wy=y==0?1.0f-fraction.y:fraction.y;
        if(wx<=0.0f||wy<=0.0f)continue;
        int2 pixel=clamp(low+int2(x,y),int2(0,0),last);
        float3 replacement=0.0f;bool erased=false;
        if(!LoadCurrentBackground(pixel,replacement,erased))return BackgroundOldSelectedSample(source_uv,eye_sign);
        float3 rgb=SourceColor.Load(int3(pixel,0)).rgb;
        // Explicit branch: the out-RGB helper is never allowed to overwrite a clear texel.
        if(erased){rgb=replacement;any_erased=true;}
        // An ordinary nonfinite tap must not contaminate a mixed repaired footprint. A finite
        // A1 replacement remains usable even when that erased texel's original raw was nonfinite.
        if(any(!isfinite(rgb)))return BackgroundOldSelectedSample(source_uv,eye_sign);
        reconstructed+=(wx*wy)*rgb;
    }
    if(!any_erased) {
        if(!ui_any){stats.z=0.0f;return BackgroundOriginalSample(base_uv);}
        // A fractional UI layer still needs the exact clear Base sample underneath.
        reconstructed=BackgroundOriginalSample(base_uv).rgb;
    }
    if(any(!isfinite(reconstructed)))return BackgroundOldSelectedSample(source_uv,eye_sign);
    if(ui_any) {
        // A=coverage and P=marked native RGB already premultiplied by its bilinear
        // weight. Using A*Sample(original,UI) would wrongly include unmarked UI RGB
        // and apply A a second time to the marked tap weights.
        float3 composite=ui_premultiplied+(1.0f-saturate(ui_coverage))*reconstructed;
        if(any(!isfinite(composite)))return BackgroundOldSelectedSample(source_uv,eye_sign);
        // The old any-positive UI branch owned alpha even at fractional coverage.
        // RGB-only compositing preserves that alpha contract; unmarked UI RGB is
        // never part of the composite.
        stats.z=1.0f;return float4(composite,BackgroundOriginalSample(ui_uv).a);
    }
    // Background repair changes RGB only. Preserve the qualified old selector's output alpha,
    // which can be UI alpha when an original-M Base footprint hit caused old UI selection.
    bool old_base_hit=GlyphBilinearFootprint(base_uv,source_size,true);
    float original_alpha=BackgroundOriginalSample(old_base_hit?ui_uv:base_uv).a;
    stats.z=2.0f;return float4(reconstructed,original_alpha);
}
