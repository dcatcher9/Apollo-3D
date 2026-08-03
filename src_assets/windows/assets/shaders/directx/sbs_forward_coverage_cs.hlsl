// Harness-only depth-ordered forward coverage. One thread represents one source sample at the
// per-eye output resolution and atomically records the nearest source for both eyes. Live V2 is
// contractive and does not execute this pass; direct final-parallax replay uses it only to report
// discrete raster coverage, never as rendering authority.

Texture2D<float4> LeftColorTexture : register(t0);
Texture2D<float> DepthTexture : register(t1);
#if !defined(SBS_DIRECT_PARALLAX)
StructuredBuffer<float4> SubjectState : register(t2);
#endif
#if defined(SBS_DIRECT_PARALLAX)
// Canonical order is diagnostic metadata for collisions in the forward-coverage mask. The
// contractive inverse renderer itself consumes only final parallax.
Texture2D<float> DirectOrderTexture : register(t5);
#endif
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
#if defined(SBS_DIRECT_PARALLAX)
    // Direct replay has no legacy subject, zero-plane, stretch, or adaptive-pop resource.
    float d = DepthTexture.SampleLevel(LinearSampler, uv, 0);
    float parallax = DepthParallax(d);
    float order = DirectOrderTexture.SampleLevel(LinearSampler, uv, 0);
#else
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
#endif

#if defined(SBS_DIRECT_PARALLAX)
    // IEEE-754 finite floats can be transformed into an unsigned lexicographic order. Retaining
    // the high 16 bits matches the existing packed atomic format without clipping an unbounded
    // canonical coordinate into an arbitrary numeric range. Zero remains reserved for no cover.
    // Canonicalize -0 so numerically equal zero values remain a tie in the atomic key.
    uint order_bits = asuint(order == 0.0f ? 0.0f : order);
    uint ordered = (order_bits & 0x80000000u) != 0u ? ~order_bits :
                                                            (order_bits ^ 0x80000000u);
    uint depth_key = 1u + min(ordered >> 16u, 65534u);
#else
    float shaped_depth = Bestv2WarpDepth(d, s0, s1, shaped, subject_stretch > 0.5f);
    uint depth_key = 1u + (uint)round(saturate(shaped_depth) * 65533.0f);
#endif
    // Store normalized source-U rather than id.x. The source and output grids differ under aspect
    // fit, and silently treating their integer columns as identical would move the selected color.
    // Quantization is at most 1/65535 source-U (well below one source pixel at supported sizes).
    uint source_u_key = (uint)round(saturate(uv.x) * 65535.0f);
    uint packed = (depth_key << 16u) | source_u_key;

    float output_shift = parallax * content_scale_x * (float)eye_w;
    int left_x = clamp((int)round((float)id.x + output_shift), 0, (int)eye_w - 1);
    int right_x = clamp((int)round((float)id.x - output_shift), 0, (int)eye_w - 1);
    InterlockedMax(CoverageTexture[uint2((uint)left_x, id.y)], packed);
    InterlockedMax(CoverageTexture[uint2(eye_w + (uint)right_x, id.y)], packed);
}
