RWStructuredBuffer<float2> GroupSums : register(u0);
RWStructuredBuffer<float4> SubjectState : register(u1);

#include "include/depth_plane_constants.hlsl"

[numthreads(1, 1, 1)]
void main() {
    float shift_sum = 0.0f;
    float weight_sum = 0.0f;
    for (uint i = 0u; i < plane_group_count; ++i) {
        float2 group_sum = GroupSums[i];
        shift_sum += group_sum.x;
        weight_sum += group_sum.y;
    }
    SubjectState[2] = float4(weight_sum > 1e-6f ? shift_sum / weight_sum : 0.0f,
                             weight_sum > 1e-6f ? 1.0f : 0.0f, 0.0f, 0.0f);
}
