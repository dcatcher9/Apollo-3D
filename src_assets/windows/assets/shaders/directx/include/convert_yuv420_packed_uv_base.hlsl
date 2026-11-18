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

float3 LoadChromaTexel(int2 position, int left, int right)
{
    position = clamp(position, int2(left, 0),
        int2(right - 1, (int)chroma_source_height - 1));
    return CONVERT_FUNCTION(image.Load(int3(position, 0)).rgb);
}

// Transfer/gamut conversion precedes this left-sited 4:2:0 filter. Keep the addition order
// shared by raster texel loads and the tile's converted source samples.
float3 CombineLeftChromaSamples(float3 top_center, float3 bottom_center,
    float3 top_left, float3 top_right, float3 bottom_left, float3 bottom_right)
{
    float3 rgb = top_center;
    rgb += bottom_center;
    rgb *= 2.0f;
    rgb += top_left;
    rgb += top_right;
    rgb += bottom_left;
    rgb += bottom_right;
    return rgb * (1.0f / 8.0f);
}

float2 EncodeChroma(float3 rgb)
{
    float u = dot(color_vec_u.xyz, rgb) + color_vec_u.w;
    float v = dot(color_vec_v.xyz, rgb) + color_vec_v.w;

    u = u * range_uv.x + range_uv.y;
    v = v * range_uv.x + range_uv.y;

    return float2(u, v);
}
