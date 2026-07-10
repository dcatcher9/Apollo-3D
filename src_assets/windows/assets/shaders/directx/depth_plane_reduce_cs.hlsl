// Measure Bestv2's mask-weighted mean raw subject shift. The frame-global anchor is
// resolved separately so both warp geometries consume exactly the same correction.
Texture2D<float> DepthTexture : register(t0);
Texture2D<float> SmoothedBand : register(t1);
StructuredBuffer<float4> SubjectState : register(t2);
RWStructuredBuffer<float2> GroupSums : register(u0);

#include "include/depth_plane_constants.hlsl"

groupshared float ShiftSum[256];
groupshared float WeightSum[256];

float Bestv2RawShiftPxPlane(float d) {
    float wn = exp(-0.5f * ((d - 0.85f) / 0.24f) * ((d - 0.85f) / 0.24f));
    float wm = exp(-0.5f * ((d - 0.50f) / 0.28f) * ((d - 0.50f) / 0.28f));
    float wf = exp(-0.5f * ((d - 0.15f) / 0.24f) * ((d - 0.15f) / 0.24f));
    return (wn * 9.99f + wm * 3.0f - wf * 2.52f) / (wn + wm + wf + 1e-6f);
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
    float weighted_shift = 0.0f;
    float weight = 0.0f;
    if (id.x < plane_w && id.y < plane_h) {
        float4 s0 = SubjectState[0];
        float4 s1 = SubjectState[1];
        float d = saturate(DepthTexture.Load(int3(id.xy, 0)));
        float d_stretched = saturate((d - s1.x) * s1.y);
        float shaped_depth = saturate(d_stretched + s0.x);
        weight = saturate(SmoothedBand.Load(int3(id.xy, 0)) * plane_strength);
        weighted_shift = Bestv2RawShiftPxPlane(shaped_depth) * weight;
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
