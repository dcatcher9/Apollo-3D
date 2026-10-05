// sbs_packed_resample_ps for a PQ (Rec.2020 ST 2084) export: each tap is decoded to linear
// scRGB before it is weighted, so the output is FP16 scRGB filtered in linear light exactly
// as the FP16 export was. The per-eye clamp is unchanged.

Texture2D PackedSource : register(t0);

#include "include/pq_input.hlsl"

struct PS_INPUT {
    float4 Pos : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

float4 main_ps(PS_INPUT input) : SV_TARGET {
    uint width, height;
    PackedSource.GetDimensions(width, height);
    const float half_texel = 0.5f / (float)width;
    const bool right_eye = input.TexCoord.x > 0.5f;
    const float lo = right_eye ? 0.5f + half_texel : half_texel;
    const float hi = right_eye ? 1.0f - half_texel : 0.5f - half_texel;
    return SamplePQInputAsScRGB(PackedSource, float2(clamp(input.TexCoord.x, lo, hi), input.TexCoord.y));
}
