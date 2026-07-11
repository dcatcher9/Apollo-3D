// VisionDepth3D depth-ordered forward splat. One thread represents one source pixel at the
// per-eye output resolution and atomically records the nearest source x for both eyes.
// The resolve shader performs directional horizontal hole fill and applies the configured
// forward weight (quality profile 0.35; fidelity profile 0.65).

Texture2D<float4> LeftColorTexture : register(t0);
Texture2D<float> DepthTexture : register(t1);
StructuredBuffer<float4> SubjectState : register(t2);
Texture2D<float> PlaneLockTexture : register(t4);
RWTexture2D<uint> WinnerTexture : register(u0);
SamplerState LinearSampler : register(s0);

#include "include/sbs_warp_common.hlsl"

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint full_w, eye_h;
    WinnerTexture.GetDimensions(full_w, eye_h);
    uint eye_w = full_w / 2u;
    uint source_w, source_h;
    LeftColorTexture.GetDimensions(source_w, source_h);
    if (id.x >= eye_w || id.y >= eye_h) {
        return;
    }

    float2 output_uv = (float2(id.xy) + 0.5f) / float2(eye_w, eye_h);
    float2 uv;
    if (!ContentToSourceUV(output_uv, uv)) {
        return;  // leave the per-eye letterbox/pillarbox winner empty (black in resolve)
    }
    float d = DepthTexture.SampleLevel(LinearSampler, uv, 0);
    float4 s0 = SubjectState[0];
    float4 s1 = SubjectState[1];
    float4 s2 = SubjectState[2];
    bool shaped = (subject_track > 0.5f) && (s0.w > 0.5f);
    float plane_mask = PlaneLockTexture.SampleLevel(LinearSampler, uv, 0);
    float parallax = DepthParallax(d, plane_mask, uv.x, s0, s1, s2, shaped, (float)source_w);

    // Apollo is high=near. Store a 16-bit near priority plus the 16-bit source x; max wins.
    uint depth_key = 1u + (uint)round(saturate(WarpDepth(d, s0, s1, shaped)) * 65533.0f);
    uint packed = (depth_key << 16u) | (id.x & 0xffffu);

    float output_shift = parallax * content_scale_x * (float)eye_w;
    int left_x = clamp((int)round((float)id.x + output_shift), 0, (int)eye_w - 1);
    int right_x = clamp((int)round((float)id.x - output_shift), 0, (int)eye_w - 1);
    InterlockedMax(WinnerTexture[uint2((uint)left_x, id.y)], packed);
    InterlockedMax(WinnerTexture[uint2(eye_w + (uint)right_x, id.y)], packed);
}
