// Count the acquisition frame's extreme-near canonical tail entirely on GPU. The frame mean is
// already known here, so raw > mean + probe_u * fixed_scale is exactly u > probe_u for the camera
// that state_resolve will latch. One atomic add per threadgroup avoids a contended per-texel
// counter. State resolve consumes and clears NearTailCount on every path.

StructuredBuffer<float> InputBuffer : register(t0);
StructuredBuffer<float4> FrameStats : register(t1);
StructuredBuffer<float4> ShadowState : register(t2);
StructuredBuffer<float4> LegacyState : register(t3);
RWStructuredBuffer<uint> NearTailCount : register(u0);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"

#define GROUP_SIZE 256
groupshared uint g_near_count[GROUP_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID,
          uint3 tid : SV_GroupThreadID) {
    float4 active = ShadowState[0];
    float4 control = ShadowState[1];
    float4 shoulder = ShadowState[2];
    bool contract_matches =
        asuint(V2_STATE_CONTRACT_TAG_BITS(control)) == V2_CONTRACT_TAG;
    float4 legacy_cut = LegacyState[SBS_STATE_VECTOR_HARD_CUT_PULSE];
    float4 legacy_health = LegacyState[SBS_STATE_VECTOR_HARD_CUT_COUNT];
    uint current_cut_count = asuint(SBS_STATE_HARD_CUT_COUNT(legacy_health));
    uint previous_cut_count = contract_matches ?
        asuint(V2_STATE_CONFIRMED_CUT_COUNT(control)) : current_cut_count;
    bool confirmed_cut = SBS_STATE_HARD_CUT_PULSE(legacy_cut) > 0.5f ||
        (contract_matches && current_cut_count != previous_cut_count);
    bool needs_acquisition_evidence = confirmed_cut ||
        !V2CameraStateValid(active, control, shoulder, target_w * target_h);
    // This branch is frame-uniform. Once a shot owns a valid camera, ordinary completions avoid
    // all tensor reads, group barriers, and atomics; only acquisition or a confirmed cut scans.
    if (!needs_acquisition_evidence) {
        return;
    }

    float4 frame0 = FrameStats[V2_FRAME_STATS_VECTOR_MEAN];
    float4 frame1 = FrameStats[V2_FRAME_STATS_VECTOR_VALID_COUNT];
    bool frame_valid = V2_FRAME_STATS_VALID(frame1) > 0.5f;
    float threshold = V2_FRAME_STATS_MEAN(frame0) +
        v2_near_tail_probe_u * v2_raw_coordinate_scale;
    uint element_count = target_w * target_h;
    uint count = 0u;

    if (frame_valid && !isnan(threshold) && !isinf(threshold)) {
        [loop]
        for (uint index = dtid.x; index < element_count; index += reduce_threads) {
            float value = InputBuffer[index];
            if (!isnan(value) && !isinf(value) && value > threshold) {
                count++;
            }
        }
    }
    g_near_count[tid.x] = count;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = GROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid.x < stride) {
            g_near_count[tid.x] += g_near_count[tid.x + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid.x == 0u && g_near_count[0] > 0u) {
        InterlockedAdd(NearTailCount[0], g_near_count[0]);
    }
}
