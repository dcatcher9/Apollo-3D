// Optional Host SBS HDR fast path: render the authenticated packed RGB raster and its P010
// luma plane in one full-resolution draw. Chroma deliberately remains on the established
// intermediate-based pass so this optimization cannot change 4:2:0 siting or filtering.

#include "sbs_reprojection_v2_live_ps.hlsl"
#include "include/common.hlsl"

cbuffer color_matrix_cbuffer : register(b0) {
    float4 color_vec_y;
    float4 color_vec_u;
    float4 color_vec_v;
    float2 range_y;
    float2 range_uv;
};

struct P010_Y_OUTPUT {
    float4 packed_rgb : SV_Target0;
    float luma : SV_Target1;
};

float RoundTripFp16(float value) {
    return f16tof32(f32tof16(value));
}

float3 RoundTripFp16(float3 value) {
    return float3(
        RoundTripFp16(value.r),
        RoundTripFp16(value.g),
        RoundTripFp16(value.b)
    );
}

P010_Y_OUTPUT main_p010_y_ps(PS_INPUT input) {
    P010_Y_OUTPUT output;
    output.packed_rgb = main_ps(input);

    // The reference path first stores the warp in R16G16B16A16_FLOAT, then samples that texture
    // at exact texel centers before scRGB -> Rec.2100 PQ conversion. Reproduce that half-float
    // boundary explicitly so the fused luma plane remains byte-equivalent to the reference draw.
    float3 pq_rgb = scRGBTo2100PQ(RoundTripFp16(output.packed_rgb.rgb));
    float y = dot(color_vec_y.xyz, pq_rgb) + color_vec_y.w;
    output.luma = y * range_y.x + range_y.y;
    return output;
}
