Texture2D source_texture : register(t0);
SamplerState source_sampler : register(s0);

#include "include/base_vs_types.hlsl"
#include "include/common.hlsl"

float4 main_ps(vertex_t input) : SV_Target {
    const float4 source = source_texture.Sample(source_sampler, input.tex_coord, 0);
    return float4(ApplySRGBCurve(saturate(source.rgb)), source.a);
}
