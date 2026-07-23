Texture2D cursor : register(t0);
SamplerState def_sampler : register(s0);

cbuffer normalize_white_cbuffer : register(b1) {
    float white_multiplier;
};

#include "include/base_vs_types.hlsl"
#include "include/common.hlsl"

float4 main_ps(vertex_t input) : SV_Target
{
    float4 output = cursor.Sample(def_sampler, input.tex_coord, 0);

    // Win32 cursor bitmaps are display-referred sRGB, while an HDR Desktop Duplication
    // target is linear scRGB. Decode before scaling to the user's SDR reference white.
    output.rgb = RemoveSRGBCurve(output.rgb) * white_multiplier;

    return output;
}
