Texture2D source_texture : register(t0);
SamplerState source_sampler : register(s0);

cbuffer rgb_present_sdr_white_cbuffer : register(b1) {
    float source_sdr_white_scrgb;
    float3 rgb_present_sdr_white_padding;
};

#include "include/base_vs_types.hlsl"
#include "include/common.hlsl"

float4 main_ps(vertex_t input) : SV_Target {
    const float4 source = source_texture.Sample(source_sampler, input.tex_coord, 0);
    // Windows HDR composition uses linear scRGB, where 1.0 is 80 nits. A BGRA capture is
    // display-referred sRGB, so decode it and restore the source display's SDR reference white.
    return float4(
        RemoveSRGBCurve(source.rgb) * source_sdr_white_scrgb,
        source.a
    );
}
