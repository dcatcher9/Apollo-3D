// Harness-only depth-ordered forward coverage. One thread represents one source pixel at the
// per-eye output resolution and atomically records the nearest source x for both eyes. The
// diagnostic mask uses empty destinations to measure disocclusion before Apollo's gather fills it.

Texture2D<float4> LeftColorTexture : register(t0);
Texture2D<float> DepthTexture : register(t1);
StructuredBuffer<float4> SubjectState : register(t2);
RWTexture2D<uint> CoverageTexture : register(u0);
RWTexture2D<float> ParallaxTexture : register(u1);
SamplerState LinearSampler : register(s0);

#include "include/sbs_warp_common.hlsl"

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint full_w, eye_h;
    CoverageTexture.GetDimensions(full_w, eye_h);
    uint eye_w = full_w / 2u;
    uint source_w, source_h;
    LeftColorTexture.GetDimensions(source_w, source_h);
    if (id.x >= eye_w || id.y >= eye_h) {
        return;
    }

    float2 output_uv = (float2(id.xy) + 0.5f) / float2(eye_w, eye_h);
    float2 uv;
    if (!ContentToSourceUV(output_uv, uv)) {
        return;
    }
    float d = DepthTexture.SampleLevel(LinearSampler, uv, 0);
    float4 s0 = SubjectState[0];
    float4 s1 = SubjectState[1];
    float4 s2 = SubjectState[2];
    bool shaped = s0.w > 0.5f;
    float parallax = 0.0f;
    if (shaped) {
        Bestv2Params params = MakeBestv2Params(
            s0, s1, s2, (float)source_w, (float)source_h, subject_stretch > 0.5f);
        parallax = DepthParallax(
            d, s0, s1, params, subject_stretch > 0.5f);
    }

    float shaped_depth = Bestv2WarpDepth(d, s0, s1, shaped, subject_stretch > 0.5f);
    uint depth_key = 1u + (uint)round(saturate(shaped_depth) * 65533.0f);
    uint packed = (depth_key << 16u) | (id.x & 0xffffu);

    float output_shift = parallax * content_scale_x * (float)eye_w;
    int left_x = clamp((int)round((float)id.x + output_shift), 0, (int)eye_w - 1);
    int right_x = clamp((int)round((float)id.x - output_shift), 0, (int)eye_w - 1);
    InterlockedMax(CoverageTexture[uint2((uint)left_x, id.y)], packed);
    InterlockedMax(CoverageTexture[uint2(eye_w + (uint)right_x, id.y)], packed);
}

// Training-data export: write Apollo's exact current full binocular disparity at the same pixel
// centers as one rendered output eye. Letterbox/pillarbox pixels are zero and are excluded by the
// evaluator's centered content-valid mask. The shipping warp applies half of this value with
// opposite signs to the two eyes.
[numthreads(16, 16, 1)]
void parallax_main(uint3 id : SV_DispatchThreadID) {
    uint output_w, output_h;
    ParallaxTexture.GetDimensions(output_w, output_h);
    if (id.x >= output_w || id.y >= output_h) {
        return;
    }
    uint source_w, source_h;
    LeftColorTexture.GetDimensions(source_w, source_h);
    float2 output_uv = (float2(id.xy) + 0.5f) / float2(output_w, output_h);
    float2 uv;
    if (!ContentToSourceUV(output_uv, uv)) {
        ParallaxTexture[id.xy] = 0.0f;
        return;
    }
    float d = DepthTexture.SampleLevel(LinearSampler, uv, 0);
    float4 s0 = SubjectState[0];
    float4 s1 = SubjectState[1];
    float4 s2 = SubjectState[2];
    float full_disparity = 0.0f;
    if (s0.w > 0.5f) {
        Bestv2Params params = MakeBestv2Params(
            s0, s1, s2, (float)source_w, (float)source_h, subject_stretch > 0.5f);
        full_disparity = 2.0f * content_scale_x * DepthParallax(
            d, s0, s1, params, subject_stretch > 0.5f);
    }
    ParallaxTexture[id.xy] = full_disparity;
}

// Training-only unclamped baseline field. Remove the current artistic multiplier so an offline
// loss can reconstruct the shipping nonlinear contract exactly for any candidate scale:
// clamp(raw_baseline * scale, +/- full_production_clamp). The independent +/-3% perceived
// disparity comfort limits are evaluator gates, not this renderer clamp. The production renderer
// never consumes this texture.
[numthreads(16, 16, 1)]
void raw_parallax_main(uint3 id : SV_DispatchThreadID) {
    uint output_w, output_h;
    ParallaxTexture.GetDimensions(output_w, output_h);
    if (id.x >= output_w || id.y >= output_h) {
        return;
    }
    uint source_w, source_h;
    LeftColorTexture.GetDimensions(source_w, source_h);
    float2 output_uv = (float2(id.xy) + 0.5f) / float2(output_w, output_h);
    float2 uv;
    if (!ContentToSourceUV(output_uv, uv)) {
        ParallaxTexture[id.xy] = 0.0f;
        return;
    }
    float d = DepthTexture.SampleLevel(LinearSampler, uv, 0);
    float4 s0 = SubjectState[0];
    float4 s1 = SubjectState[1];
    float4 s2 = SubjectState[2];
    float full_disparity = 0.0f;
    if (s0.w > 0.5f) {
        Bestv2Params params = MakeBestv2Params(
            s0, s1, s2, (float)source_w, (float)source_h, subject_stretch > 0.5f);
        float artistic_ratio = s2.w > 0.0f ? s2.w : 1.0f;
        params.output_scale /= artistic_ratio;
        full_disparity = 2.0f * content_scale_x * DepthParallaxUnclamped(
            d, s0, s1, params, subject_stretch > 0.5f);
    }
    ParallaxTexture[id.xy] = full_disparity;
}
