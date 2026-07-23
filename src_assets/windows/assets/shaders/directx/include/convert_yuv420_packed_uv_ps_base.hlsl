Texture2D image : register(t0);
SamplerState def_sampler : register(s0);

cbuffer color_matrix_cbuffer : register(b0) {
    float4 color_vec_y;
    float4 color_vec_u;
    float4 color_vec_v;
    float2 range_y;
    float2 range_uv;
};

#include "include/base_vs_types.hlsl"

float3 SampleChromaInput(float2 tex_coord)
{
    // 4:2:0 is defined over target-encoded RGB/YUV. Convert every source tap into the
    // declared transfer function and gamut before filtering; nonlinear transfer or gamut
    // conversion after RGB averaging produces incorrect chroma at color and luminance edges.
    return CONVERT_FUNCTION(image.Sample(def_sampler, tex_coord).rgb);
}

float2 main_ps(vertex_t input) : SV_Target
{
#if defined(LEFT_SUBSAMPLING)
    float3 rgb_left = SampleChromaInput(input.tex_right_left_center.xz);
    float3 rgb_right = SampleChromaInput(input.tex_right_left_center.yz);
    float3 rgb = (rgb_left + rgb_right) * 0.5;
#elif defined(LEFT_SUBSAMPLING_SCALE)
    float3 rgb = SampleChromaInput(input.tex_right_center_left_top.yw); // top-center
    rgb += SampleChromaInput(input.tex_right_center_left_bottom.yw); // bottom-center
    rgb *= 2;
    rgb += SampleChromaInput(input.tex_right_center_left_top.xw); // top-right
    rgb += SampleChromaInput(input.tex_right_center_left_top.zw); // top-left
    rgb += SampleChromaInput(input.tex_right_center_left_bottom.xw); // bottom-right
    rgb += SampleChromaInput(input.tex_right_center_left_bottom.zw); // bottom-left
    rgb *= (1./8);
#elif defined(TOPLEFT_SUBSAMPLING)
    float3 rgb_top_left = SampleChromaInput(input.tex_right_left_top.xz);
    float3 rgb_top_right = SampleChromaInput(input.tex_right_left_top.yz);
    float3 rgb_bottom_left = SampleChromaInput(input.tex_right_left_bottom.xz);
    float3 rgb_bottom_right = SampleChromaInput(input.tex_right_left_bottom.yz);
    float3 rgb = (rgb_top_left + rgb_top_right + rgb_bottom_left + rgb_bottom_right) * 0.25;
#endif

    float u = dot(color_vec_u.xyz, rgb) + color_vec_u.w;
    float v = dot(color_vec_v.xyz, rgb) + color_vec_v.w;

    u = u * range_uv.x + range_uv.y;
    v = v * range_uv.x + range_uv.y;

    return float2(u, v);
}
