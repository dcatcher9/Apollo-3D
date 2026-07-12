// VisionDepth3D hybrid geometry, translated from Bestv2's pixel_shift_cuda. The blend is profile
// controlled (shipping VD3D: 65% classic backward + 35% depth-ordered forward). Bestv2's forward
// holes use a nearest-valid horizontal search of up to 96 source pixels.

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

float4 ForwardColor(uint local_x, uint y, bool right_eye, uint eye_w, uint eye_h,
                    float2 fallback_uv, out float raw_hole, out float unresolved_hole) {
    uint winner = WinnerAt((int)local_x, y, right_eye, eye_w);
    raw_hole = winner == 0u ? 1.0f : 0.0f;
    uint source_w, source_h;
    LeftColorTexture.GetDimensions(source_w, source_h);
    // Bestv2's fill radius is calibrated in source pixels. Scale it into the output-eye grid so
    // a 5120->4096 capped stream uses 96*.8 ~= 77 output pixels rather than filling 25% farther.
    int radius = clamp((int)round(96.0f * source_to_output *
                                 Bestv2AspectScale(
                                     (float)source_w, (float)source_h,
                                     literal_bestv2)), 0, 256);

    // Equivalent to VD3D's iterative nearest-valid fill. At equal distance the first lookup is
    // the preferred side from the reference implementation.
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
        unresolved_hole = 1.0f;
        return LeftColorTexture.Sample(LinearSampler, fallback_uv);
    }
    unresolved_hole = 0.0f;
    uint source_x = winner & 0xffffu;
    float2 winner_output_uv = float2(((float)source_x + 0.5f) / (float)eye_w,
                                     ((float)y + 0.5f) / (float)eye_h);
    float2 source_uv;
    if (!ContentToSourceUV(winner_output_uv, source_uv)) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    return LeftColorTexture.Sample(LinearSampler, source_uv);
}

float4 main_ps(PS_INPUT input) : SV_TARGET {
    uint full_w, eye_h;
    WinnerTexture.GetDimensions(full_w, eye_h);
    uint eye_w = full_w / 2u;
    uint source_w, source_h;
    LeftColorTexture.GetDimensions(source_w, source_h);
    uint2 output_px = (uint2)input.Pos.xy;
    bool right_eye = output_px.x >= eye_w;
    uint local_x = right_eye ? output_px.x - eye_w : output_px.x;

    float2 output_uv = float2(((float)local_x + 0.5f) / (float)eye_w,
                              ((float)output_px.y + 0.5f) / (float)eye_h);
    float2 uv;
    if (!ContentToSourceUV(output_uv, uv)) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    float d = DepthTexture.SampleLevel(LinearSampler, uv, 0);
    float4 s0 = SubjectState[0];
    float4 s1 = SubjectState[1];
    float4 s2 = SubjectState[2];
    bool shaped = s0.w > 0.5f;
    float plane_mask = 0.0f;
    if (subject_plane_lock > 0.0f) {
        plane_mask = PlaneLockTexture.SampleLevel(LinearSampler, uv, 0);
    }
    float parallax = DepthParallax(
        d, plane_mask, s0, s1, s2, shaped, (float)source_w, (float)source_h);

    // torch grid_sample uses normalized [-1,1] coordinates; DepthParallax is UV [0,1], hence
    // this is the same left:+shift/right:-shift convention after polarity translation.
    float eye_sign = right_eye ? 1.0f : -1.0f;
    float2 backward_uv = float2(saturate(uv.x + eye_sign * parallax), uv.y);
    float4 backward = LeftColorTexture.Sample(LinearSampler, backward_uv);
    float raw_hole, unresolved_hole;
    float4 forward = ForwardColor(local_x, output_px.y, right_eye, eye_w, eye_h, uv,
                                  raw_hole, unresolved_hole);
    return lerp(backward, forward, saturate(vd3d_forward_blend));
}

// Harness-only diagnostic output. R is the forward-splat hole before the nearest-valid search;
// G is the subset still unresolved after the resolution-scaled Bestv2 search radius. The normal
// live entry point does not execute this second pass.
float4 mask_ps(PS_INPUT input) : SV_TARGET {
    uint full_w, eye_h;
    WinnerTexture.GetDimensions(full_w, eye_h);
    uint eye_w = full_w / 2u;
    uint2 output_px = (uint2)input.Pos.xy;
    bool right_eye = output_px.x >= eye_w;
    uint local_x = right_eye ? output_px.x - eye_w : output_px.x;
    float2 output_uv = float2(((float)local_x + 0.5f) / (float)eye_w,
                              ((float)output_px.y + 0.5f) / (float)eye_h);
    float2 uv;
    if (!ContentToSourceUV(output_uv, uv)) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    float raw_hole, unresolved_hole;
    ForwardColor(local_x, output_px.y, right_eye, eye_w, eye_h, uv,
                 raw_hole, unresolved_hole);
    return float4(raw_hole, unresolved_hole, 0.0f, 1.0f);
}
