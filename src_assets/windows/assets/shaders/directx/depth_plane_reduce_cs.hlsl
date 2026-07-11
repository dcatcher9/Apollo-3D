// Measure Bestv2's mask-weighted mean raw subject shift. The frame-global anchor is
// resolved separately so both warp geometries consume exactly the same correction.
Texture2D<float> DepthTexture : register(t0);
Texture2D<float> SmoothedBand : register(t1);
StructuredBuffer<float4> SubjectState : register(t2);
RWStructuredBuffer<float2> GroupSums : register(u0);

#include "include/depth_plane_constants.hlsl"
#include "include/bestv2_curve.hlsl"

groupshared float ShiftSum[256];
groupshared float WeightSum[256];

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
    float weighted_shift = 0.0f;
    float weight = 0.0f;
    if (id.x < plane_w && id.y < plane_h) {
        float4 s0 = SubjectState[0];
        float4 s1 = SubjectState[1];
        float d = saturate(DepthTexture.Load(int3(id.xy, 0)));
        float shaped_depth = Bestv2WarpDepth(
            d, s0, s1, true, plane_subject_stretch != 0u);
        weight = saturate(SmoothedBand.Load(int3(id.xy, 0)) * plane_strength);
        weighted_shift = Bestv2RawShiftPx(shaped_depth) * weight;
    }
    ShiftSum[gi] = weighted_shift;
    WeightSum[gi] = weight;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (gi < stride) {
            ShiftSum[gi] += ShiftSum[gi + stride];
            WeightSum[gi] += WeightSum[gi + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi == 0u) {
        uint groups_x = (plane_w + 15u) / 16u;
        GroupSums[gid.y * groups_x + gid.x] = float2(ShiftSum[0], WeightSum[0]);
    }
}
