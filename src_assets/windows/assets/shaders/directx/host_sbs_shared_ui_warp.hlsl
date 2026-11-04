// PRIVATE comparison adapter. The pin band itself is the exact shared Game 3D pass.
// Original source/Base/SCP/SCT/matte/receipt operands remain unchanged. No RGB fill.
#define main_ps BaselineCompletedMainPs
#define background_diagnostic_ps BaselineCompletedDiagnosticPs
#include "current_outline/completed/sbs_reprojection_v2_live_ps.hlsl"
#undef main_ps
#undef background_diagnostic_ps

cbuffer SharedPinGeometry : register(b5) {
    uint4 PinSourceShape; // full width, full height, native band top, native band height
};
Texture2D<float> SharedPinWeights : register(t12);
Texture2D<float> SharedPinField : register(t13);
Texture2D<uint> SharedPinRows : register(t14);
RWTexture2D<float> SunshineHostFinalStore : register(u0);
RWTexture2D<float> SharedPinWeightStore : register(u1);
RWTexture2D<uint> SharedPinRowStore : register(u2);

float SharedNativeBaseSample(float2 uv) {
    // Building a second sampled field needs FP32 coefficients: hardware bilinear's
    // quantized fractions can make a source 0.5-pixel slope become >0.5 on this grid.
    uint2 size;BaseParallax.GetDimensions(size.x,size.y);
    float2 position=saturate(uv)*float2(size)-0.5f;
    int2 low=int2(floor(position)),last=int2(size)-1;float2 fraction=position-float2(low);
    float a=BaseParallax.Load(int3(clamp(low,int2(0,0),last),0));
    float b=BaseParallax.Load(int3(clamp(low+int2(1,0),int2(0,0),last),0));
    float c=BaseParallax.Load(int3(clamp(low+int2(0,1),int2(0,0),last),0));
    float d=BaseParallax.Load(int3(clamp(low+int2(1,1),int2(0,0),last),0));
    return lerp(lerp(a,b,fraction.x),lerp(c,d,fraction.x),fraction.y);
}
float SharedOriginalParallax(float2 uv) {
    if (!WarpAuthorized()) return 0.0f;
    if (video_roi_active == 0.0f) return SharedNativeBaseSample(uv);
    uint2 field_size; BaseParallax.GetDimensions(field_size.x, field_size.y);
    if (!VideoRoiRectValid() || !VideoRoiContentValid(field_size)) return 0.0f;
    float2 extent = video_roi_source_uv.zw - video_roi_source_uv.xy;
    uint2 content_extent = video_roi_tensor_content_rect.zw - video_roi_tensor_content_rect.xy;
    float2 content_min = float2(video_roi_tensor_content_rect.xy) / float2(field_size);
    float2 content_max = float2(video_roi_tensor_content_rect.zw) / float2(field_size);
    float2 scale = (content_max - content_min) / extent;
    float2 bias = content_min - video_roi_source_uv.xy * scale;
    float projected_y = clamp(uv.y, video_roi_source_uv.y, video_roi_source_uv.w);
    float budget_per_v = V2_MAX_VERTICAL_SHEAR * (extent.x / extent.y) *
        ((float)content_extent.y / (float)content_extent.x);
    float projected_x=clamp(uv.x,video_roi_source_uv.x,video_roi_source_uv.z);
    float collar=VIDEO_ROI_MAX_HORIZONTAL_SLOPE*abs(uv.x-projected_x)+budget_per_v*abs(uv.y-projected_y);
    float p=extent.x*SharedNativeBaseSample(float2(projected_x*scale.x+bias.x,projected_y*scale.y+bias.y));
    float magnitude=abs(p);
    if(collar<magnitude){float retained=magnitude-collar;return p<0.0f?-retained:retained;}
    return 0.0f;
}

groupshared uint SharedPinGroupAuthorized;
groupshared uint SharedPinGroupSelectedReceipt;
[numthreads(16,16,1)]
void initialize_main(uint3 id : SV_DispatchThreadID, uint3 lane : SV_GroupThreadID) {
    if (all(lane.xy == 0u)) {
        uint2 source_size; float plane;
        SharedPinGroupSelectedReceipt = SelectedMatteAuthorized() ? 1u : 0u;
        SharedPinGroupAuthorized = SubtitleRenderAuthorized(source_size, plane, true) &&
            SharedPinGroupSelectedReceipt != 0u && all(source_size == PinSourceShape.xy) ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    if (id.x >= PinSourceShape.x || id.y >= PinSourceShape.w) return;
    // One owned per-draw header avoids repeating the long immutable receipt check in the PS.
    if (all(id.xy == 0u)) SharedPinRowStore[uint2(0u,PinSourceShape.w)] = SharedPinGroupSelectedReceipt;
    uint2 pixel = uint2(id.x, id.y + PinSourceShape.z);
    float2 uv = (float2(pixel) + 0.5f) / float2(PinSourceShape.xy);
    SunshineHostFinalStore[id.xy] = SharedOriginalParallax(uv);
    float weight = SharedPinGroupAuthorized != 0u && CurrentGlyphContributor(int2(pixel), true) ? 1.0f : 0.0f;
    SharedPinWeightStore[id.xy] = weight;
    if (weight > 0.0f) InterlockedOr(SharedPinRowStore[uint2(0u,id.y)], 1u);
}

#define BUFFER_WIDTH PinSourceShape.x
#define BUFFER_HEIGHT PinSourceShape.w
static const uint Sunshine_SourceAlphaUI = 1u;
float SunshineUIPinWeight(uint x, uint y) { return SharedPinWeights.Load(int3(x,y,0)); }
float SunshineUIPlaneParallax() {
    return asfloat(SubtitleCondition[5u]) * ((float)subtitle_render_source.z / (float)PinSourceShape.x);
}
#include "include/ui_pin_band_shared.hlsl"

bool SharedPinRowContributes(float source_v) {
    float y = saturate(source_v) * (float)PinSourceShape.y - 0.5f;
    int low = (int)floor(y); float fraction = y - (float)low;
    int first = (int)PinSourceShape.z, end = first + (int)PinSourceShape.w;
    int a = clamp(low, 0, (int)PinSourceShape.y - 1);
    int b = clamp(low + 1, 0, (int)PinSourceShape.y - 1);
    if (1.0f-fraction > 0.0f && a >= first && a < end && SharedPinRows.Load(int3(0,a-first,0)) != 0u) return true;
    return fraction > 0.0f && b >= first && b < end && SharedPinRows.Load(int3(0,b-first,0)) != 0u;
}
float2 SharedPinSourceUV(float2 destination_uv, float eye_sign) {
    if (SharedPinRows.Load(int3(0,PinSourceShape.w,0)) == 0u) return SelectSourceUV(destination_uv,eye_sign);
    // Empty rows and missing/stale/revoked masks retain the exact original Base inverse.
    if (!SharedPinRowContributes(destination_uv.y)) return ReprojectIfAuthorized(destination_uv,eye_sign);
    float x = destination_uv.x;
    float v = (destination_uv.y * (float)PinSourceShape.y - (float)PinSourceShape.z) / (float)PinSourceShape.w;
    [loop] for (int iteration=0; iteration<11; ++iteration) {
        float next = destination_uv.x + eye_sign * SharedPinField.SampleLevel(LinearSampler,float2(x,v),0);
        bool settled = asuint(next) == asuint(x); x = next; if (settled) break;
    }
    return float2(x,destination_uv.y);
}
float4 shared_pin_main_ps(PS_INPUT input) : SV_TARGET {
    float2 uv; float eye_sign;
    if (!PackedToSource(input.TexCoord,uv,eye_sign)) return 0.0f;
    float native_y=uv.y*(float)PinSourceShape.y-0.5f;
    if (native_y <= (float)PinSourceShape.z-1.0f || native_y >= (float)(PinSourceShape.z+PinSourceShape.w))
        return BackgroundOriginalSample(ReprojectIfAuthorized(uv,eye_sign));
    // Preserve the production literal legacy selector when selected receipt authority fails.
    if (SharedPinRows.Load(int3(0,PinSourceShape.w,0)) == 0u) return BaselineCompletedMainPs(input);
    return BackgroundOriginalSample(SharedPinSourceUV(uv,eye_sign));
}
float4 shared_pin_mapping_ps(PS_INPUT input) : SV_TARGET {
    float2 uv; float eye_sign;
    if (!PackedToSource(input.TexCoord,uv,eye_sign)) return float4(-1,-1,0,0);
    float2 source_uv = SharedPinSourceUV(uv,eye_sign);
    return float4(source_uv,SharedPinRowContributes(uv.y)?1.0f:0.0f,1.0f);
}
float4 shared_pin_coverage_ps(PS_INPUT input) : SV_TARGET {
    float2 uv; float eye_sign;
    if (!PackedToSource(input.TexCoord,uv,eye_sign)) return 0.0f;
    uint2 source_size; float plane;
    if (!SubtitleRenderAuthorized(source_size,plane,true) || !SelectedMatteAuthorized()) return 0.0f;
    float2 ui_uv=float2(uv.x+eye_sign*plane,uv.y);
    float2 position=saturate(ui_uv)*float2(source_size)-0.5f;
    int2 low=int2(floor(position));float2 fraction=position-float2(low);
    float coverage=0.0f;bool any_marked=false,all_marked=true;
    [unroll]for(int y=0;y<2;++y)[unroll]for(int x=0;x<2;++x) {
        float weight=(x==0?1.0f-fraction.x:fraction.x)*(y==0?1.0f-fraction.y:fraction.y);
        if(weight<=0.0f)continue;
        if(CurrentGlyphContributor(clamp(low+int2(x,y),int2(0,0),int2(source_size)-1),true)) {coverage+=weight;any_marked=true;}
        else all_marked=false;
    }
    return float4(coverage,any_marked&&all_marked?1.0f:0.0f,0.0f,1.0f);
}
