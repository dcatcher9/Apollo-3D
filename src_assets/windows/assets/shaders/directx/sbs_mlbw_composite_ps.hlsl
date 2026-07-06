// SBS 3D synthesis via the learned multi-layer backward warp (iw3 MLBW): composite each
// eye from the per-eye field textures (DeltaTex = per-layer horizontal offsets, WeightTex =
// softmax blend weights, up to 4 layers). Per output pixel: two bilinear field taps + L
// bilinear color taps -- vs ~100 taps for the probe-search reprojection this replaces.
//
// Math validated offline against iw3's own composite (tools/warpsim/mlbwsim.cpp,
// byte-match): fields are upsampled align_corners-style, delta is in FIELD-grid pixels,
// and the source offset in normalized u is delta * delta_to_u (see cbuffer).
//
// Inputs: t0 = mono color, t1/t2 = left delta/weight, t3/t4 = right delta/weight,
// s0 = linear clamp sampler.

Texture2D<float4> ColorTexture : register(t0);
Texture2D<float4> DeltaLeft    : register(t1);
Texture2D<float4> WeightLeft   : register(t2);
Texture2D<float4> DeltaRight   : register(t3);
Texture2D<float4> WeightRight  : register(t4);
SamplerState      LinearSampler : register(s0);

struct PS_INPUT {
    float4 Pos      : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

cbuffer Constants : register(b2) {
    float eye_w;       // per-eye output width in pixels (the fitted SBS intermediate / 2)
    float eye_h;       // output height in pixels
    float field_w;     // field texture dims (the model grid)
    float field_h;
    float delta_to_u;  // source-u offset per delta unit: (eye_w-1) / (2*(floor(field_w/2)-1)) / eye_w
    float layers;      // model layer count (<= 4)
    float pad0;
    float pad1;
};

float4 main_ps(PS_INPUT input) : SV_TARGET {
    float2 uv = input.TexCoord;

    // Full SBS: left half = left eye, right half = right eye.
    bool is_right_eye = uv.x > 0.5f;
    float2 src_uv = float2(is_right_eye ? (uv.x - 0.5f) * 2.0f : uv.x * 2.0f, uv.y);

    // align_corners bilinear field lookup: output pixel ex maps to field coord
    // fx = ex*(FW-1)/(EW-1) (texel centers at integers).
    float ex = src_uv.x * eye_w - 0.5f;
    float ey = src_uv.y * eye_h - 0.5f;
    float2 field_uv = float2(
        (ex * (field_w - 1.0f) / (eye_w - 1.0f) + 0.5f) / field_w,
        (ey * (field_h - 1.0f) / (eye_h - 1.0f) + 0.5f) / field_h);

    float4 d = is_right_eye ? DeltaRight.SampleLevel(LinearSampler, field_uv, 0)
                            : DeltaLeft.SampleLevel(LinearSampler, field_uv, 0);
    float4 w = is_right_eye ? WeightRight.SampleLevel(LinearSampler, field_uv, 0)
                            : WeightLeft.SampleLevel(LinearSampler, field_uv, 0);

    // Multi-layer composite; grid_sample border padding == clamped sample coords. No output
    // clamp: the weights are a softmax (>=0, sum 1), so the blend is a convex combination
    // of source values -- and scRGB HDR sources legitimately exceed [0,1].
    float3 c = w.x * ColorTexture.Sample(LinearSampler, float2(saturate(src_uv.x + d.x * delta_to_u), src_uv.y)).rgb
             + w.y * ColorTexture.Sample(LinearSampler, float2(saturate(src_uv.x + d.y * delta_to_u), src_uv.y)).rgb;
    if (layers > 2.5f) {
        c += w.z * ColorTexture.Sample(LinearSampler, float2(saturate(src_uv.x + d.z * delta_to_u), src_uv.y)).rgb
           + w.w * ColorTexture.Sample(LinearSampler, float2(saturate(src_uv.x + d.w * delta_to_u), src_uv.y)).rgb;
    }
    return float4(c, 1.0f);
}
