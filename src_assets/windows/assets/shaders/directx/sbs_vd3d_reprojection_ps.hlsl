// VisionDepth3D hybrid geometry, translated from Bestv2's pixel_shift_cuda:
//   classic backward grid sample (35%) + depth-ordered forward splat (65%).
// Forward holes are filled from the nearest valid horizontal pixel, preferring the background
// side for ties (right for the left eye, left for the right eye), for up to 96 pixels in Bestv2.

Texture2D<float4> LeftColorTexture : register(t0);
Texture2D<float> DepthTexture : register(t1);
StructuredBuffer<float4> SubjectState : register(t2);
Texture2D<uint> WinnerTexture : register(t3);
Texture2D<float> PlaneLockTexture : register(t4);
SamplerState LinearSampler : register(s0);

#include "include/sbs_warp_common.hlsl"

struct PS_INPUT {
    float4 Pos : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

uint WinnerAt(int local_x, uint y, bool right_eye, uint eye_w) {
    if (local_x < 0 || local_x >= (int)eye_w) {
        return 0u;
    }
    uint full_x = (right_eye ? eye_w : 0u) + (uint)local_x;
    return WinnerTexture.Load(int3(full_x, y, 0));
}

float4 ForwardColor(uint local_x, uint y, bool right_eye, uint eye_w, uint eye_h, float2 fallback_uv) {
    uint winner = WinnerAt((int)local_x, y, right_eye, eye_w);
    int radius = clamp((int)round(vd3d_fill_radius), 0, 256);

    // Equivalent to VD3D's iterative nearest-valid fill. At equal distance the first lookup is
    // the preferred background side: +x for left eye, -x for right eye.
    [loop]
    for (int r = 1; winner == 0u && r <= radius; ++r) {
        int preferred_x = (int)local_x + (right_eye ? -r : r);
        int other_x = (int)local_x + (right_eye ? r : -r);
        winner = WinnerAt(preferred_x, y, right_eye, eye_w);
        if (winner == 0u) {
            winner = WinnerAt(other_x, y, right_eye, eye_w);
        }
    }

    if (winner == 0u) {
        return LeftColorTexture.Sample(LinearSampler, fallback_uv);
    }
    uint source_x = winner & 0xffffu;
    float2 source_uv = float2(((float)source_x + 0.5f) / (float)eye_w,
                              ((float)y + 0.5f) / (float)eye_h);
    return LeftColorTexture.Sample(LinearSampler, source_uv);
}

float4 main_ps(PS_INPUT input) : SV_TARGET {
    uint full_w, eye_h;
    WinnerTexture.GetDimensions(full_w, eye_h);
    uint eye_w = full_w / 2u;
    uint2 output_px = (uint2)input.Pos.xy;
    bool right_eye = output_px.x >= eye_w;
    uint local_x = right_eye ? output_px.x - eye_w : output_px.x;

    float2 uv = float2(((float)local_x + 0.5f) / (float)eye_w,
                       ((float)output_px.y + 0.5f) / (float)eye_h);
    float d = DepthTexture.SampleLevel(LinearSampler, uv, 0);
    float4 s0 = SubjectState[0];
    float4 s1 = SubjectState[1];
    float4 s2 = SubjectState[2];
    bool shaped = (subject_track > 0.5f) && (s0.w > 0.5f);
    float plane_mask = PlaneLockTexture.SampleLevel(LinearSampler, uv, 0);
    float parallax = DepthParallax(d, plane_mask, uv.x, s0, s1, s2, shaped, (float)eye_w);

    // torch grid_sample uses normalized [-1,1] coordinates; DepthParallax is UV [0,1], hence
    // this is the same left:+shift/right:-shift convention after polarity translation.
    float eye_sign = right_eye ? 1.0f : -1.0f;
    float2 backward_uv = float2(saturate(uv.x + eye_sign * parallax), uv.y);
    float4 backward = LeftColorTexture.Sample(LinearSampler, backward_uv);
    float4 forward = ForwardColor(local_x, output_px.y, right_eye, eye_w, eye_h, uv);
    return lerp(backward, forward, saturate(vd3d_forward_blend));
}
