// Explicit PQ (Rec.2020 ST 2084) Game 3D export presented on an SDR local output. It follows
// rgb_present_hdr_to_srgb_ps after decoding each tap to scRGB.
Texture2D source_texture : register(t0);

cbuffer rgb_present_sdr_white_cbuffer : register(b1) {
    float source_sdr_white_scrgb;
    float3 rgb_present_sdr_white_padding;
};

#include "include/base_vs_types.hlsl"
#include "include/common.hlsl"
#include "include/pq_input.hlsl"

float4 main_ps(vertex_t input) : SV_Target {
    const float4 source = SamplePQInputAsScRGB(source_texture, input.tex_coord);
    return float4(ApplySRGBCurve(ToneMapScRgbToSdr(source.rgb, source_sdr_white_scrgb)), source.a);
}
