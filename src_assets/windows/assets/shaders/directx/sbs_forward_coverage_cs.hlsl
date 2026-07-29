// Harness-only depth-ordered forward coverage. One thread represents one source pixel at the
// per-eye output resolution and atomically records the nearest source x for both eyes. The
// diagnostic mask uses empty destinations to measure disocclusion before Apollo's gather fills it.

Texture2D<float4> LeftColorTexture : register(t0);
Texture2D<float> DepthTexture : register(t1);
StructuredBuffer<float4> SubjectState : register(t2);
// Keep the retained-depth ownership resource on the pixel shader's next-free slot.
StructuredBuffer<uint4> FrameRoiTransform : register(t5);
RWTexture2D<uint> CoverageTexture : register(u0);
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

    uint depth_w, depth_h;
    DepthTexture.GetDimensions(depth_w, depth_h);
    SbsFrameRoiTransformData transform = FrameRoiTransformLoad();
    uint roi_mode = SbsWarpRoiMode(
        transform,
        uint2(source_w, source_h),
        uint2(depth_w, depth_h));
    if (roi_mode == SBS_WARP_ROI_MODE_INERT) {
        // This pass consumes occupancy only. Mark the one-to-one identity destinations without
        // inventing a depth value that could participate in foreground ordering.
        uint identity_coverage = 1u;
        InterlockedMax(
            CoverageTexture[uint2(id.x, id.y)],
            identity_coverage);
        InterlockedMax(
            CoverageTexture[uint2(eye_w + id.x, id.y)],
            identity_coverage);
        return;
    }

    if (roi_mode == SBS_WARP_ROI_MODE_ACTIVE) {
        float2 depth_uv;
        float feather_weight;
        if (!SbsWarpActiveSourceToDepthUv(
                transform,
                uv,
                uint2(depth_w, depth_h),
                depth_uv,
                feather_weight)) {
            // Outside the exact focus the effective mapping is identity. As above, coverage needs
            // only a nonzero occupancy marker and must not synthesize an ordering depth.
            uint identity_coverage = 1u;
            InterlockedMax(
                CoverageTexture[uint2(id.x, id.y)],
                identity_coverage);
            InterlockedMax(
                CoverageTexture[uint2(eye_w + id.x, id.y)],
                identity_coverage);
            return;
        }

        float d = DepthTexture.SampleLevel(LinearSampler, depth_uv, 0);
        float4 s0 = SubjectState[SBS_STATE_VECTOR_SUBJECT_RECENTER_DELTA];
        float4 s1 = SubjectState[SBS_STATE_VECTOR_STRETCH_LO];
        float4 s2 = SubjectState[SBS_STATE_VECTOR_ZERO_ANCHOR_SHIFT_PX];
        bool shaped = SBS_STATE_INITIALIZED(s0) > 0.5f;
        float parallax = 0.0f;
        if (shaped) {
            Bestv2Params params = MakeBestv2Params(
                s0,
                s1,
                s2,
                (float)source_w,
                (float)source_h,
                subject_stretch > 0.5f);
            parallax = SbsWarpActiveParallax(
                d,
                feather_weight,
                s0,
                s1,
                params,
                subject_stretch > 0.5f);
        }

        float shaped_depth = Bestv2WarpDepth(
            d,
            s0,
            s1,
            shaped,
            subject_stretch > 0.5f);
        uint depth_key =
            1u + (uint)round(saturate(shaped_depth) * 65533.0f);
        uint packed = (depth_key << 16u) | (id.x & 0xffffu);

        float output_shift =
            parallax * content_scale_x * (float)eye_w;
        int left_x = clamp(
            (int)round((float)id.x + output_shift),
            0,
            (int)eye_w - 1);
        int right_x = clamp(
            (int)round((float)id.x - output_shift),
            0,
            (int)eye_w - 1);
        InterlockedMax(
            CoverageTexture[uint2((uint)left_x, id.y)],
            packed);
        InterlockedMax(
            CoverageTexture[uint2(eye_w + (uint)right_x, id.y)],
            packed);
        return;
    }

    // Explicit all-zero/canonical-full legacy mode. Keep this arithmetic identical to the
    // pre-ROI path so an inactive controller cannot move a pixel or a coverage decision.
    float d = DepthTexture.SampleLevel(LinearSampler, uv, 0);
    float4 s0 = SubjectState[SBS_STATE_VECTOR_SUBJECT_RECENTER_DELTA];
    float4 s1 = SubjectState[SBS_STATE_VECTOR_STRETCH_LO];
    float4 s2 = SubjectState[SBS_STATE_VECTOR_ZERO_ANCHOR_SHIFT_PX];
    bool shaped = SBS_STATE_INITIALIZED(s0) > 0.5f;
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
