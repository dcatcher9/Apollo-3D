Texture2D image : register(t0);
SamplerState def_sampler : register(s0);

cbuffer color_matrix_cbuffer : register(b0) {
    float4 color_vec_y;
    float4 color_vec_u;
    float4 color_vec_v;
    float2 range_y;
    float2 range_uv;
};

// The VS consumes the first two words as its subsample offset. Source extent enables exact,
// unrotated 1:1 texel filtering. A nonzero eye width additionally isolates a packed Host-SBS pair.
cbuffer chroma_sampling_cbuffer : register(b2) {
    float2 chroma_subsample_offset;
    uint chroma_source_width;
    uint chroma_source_height;
    uint packed_eye_width;
    uint3 chroma_sampling_padding;
};

#include "include/base_vs_types.hlsl"

float3 SampleChromaInput(float2 tex_coord)
{
    // General 2D scaling retains its sampled-source convention. Packed Host SBS instead uses
    // exact texel loads below: a linear Sample at a chroma center already mixes source light,
    // so wrapping that Sample in the transfer function cannot implement encoded-domain filtering.
    return CONVERT_FUNCTION(image.Sample(def_sampler, tex_coord).rgb);
}

float3 LoadChromaTexel(int2 position, int left, int right)
{
    position = clamp(position, int2(left, 0),
        int2(right - 1, (int)chroma_source_height - 1));
    return CONVERT_FUNCTION(image.Load(int3(position, 0)).rgb);
}

float3 TexelChromaInput(float2 destination_position)
{
    int2 center = int2(destination_position) * 2;
    int left = packed_eye_width != 0u && center.x >= (int)packed_eye_width ?
        (int)packed_eye_width : 0;
    int right = left + (int)(packed_eye_width != 0u ? packed_eye_width : chroma_source_width);
    // Left-sited 4:2:0: horizontal [1,2,1]/4, vertical [1,1]/2. Transfer/gamut conversion
    // precedes every weighted sum, and each eye extends only its own first/last source column.
    float3 rgb = LoadChromaTexel(center, left, right);
    rgb += LoadChromaTexel(center + int2(0, 1), left, right);
    rgb *= 2.0f;
    rgb += LoadChromaTexel(center + int2(-1, 0), left, right);
    rgb += LoadChromaTexel(center + int2(1, 0), left, right);
    rgb += LoadChromaTexel(center + int2(-1, 1), left, right);
    rgb += LoadChromaTexel(center + int2(1, 1), left, right);
    return rgb * (1.0f / 8.0f);
}

float2 main_ps(vertex_t input) : SV_Target
{
    float3 rgb;
    if (chroma_source_width != 0u && chroma_source_height != 0u) {
        rgb = TexelChromaInput(input.viewpoint_pos.xy);
    } else {
#if defined(LEFT_SUBSAMPLING)
    float3 rgb_left = SampleChromaInput(input.tex_right_left_center.xz);
    float3 rgb_right = SampleChromaInput(input.tex_right_left_center.yz);
    rgb = (rgb_left + rgb_right) * 0.5;
#elif defined(LEFT_SUBSAMPLING_SCALE)
    rgb = SampleChromaInput(input.tex_right_center_left_top.yw); // top-center
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
    rgb = (rgb_top_left + rgb_top_right + rgb_bottom_left + rgb_bottom_right) * 0.25;
#endif
    }

    float u = dot(color_vec_u.xyz, rgb) + color_vec_u.w;
    float v = dot(color_vec_v.xyz, rgb) + color_vec_v.w;

    u = u * range_uv.x + range_uv.y;
    v = v * range_uv.x + range_uv.y;

    return float2(u, v);
}
