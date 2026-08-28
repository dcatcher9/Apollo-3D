// Common-path fusion for the calibrated RGB-to-NCHW producer and the GPU-only near-identical
// tile evidence. The included producer remains the sole owner of resize, transfer, normalization,
// appearance, and exclusion semantics. This wrapper only observes its committed FP32 NCHW words.

#include "rgb_to_nchw_cs.hlsl"
#include "include/host_sbs_near_identical_compare.hlsl"

StructuredBuffer<float> PreviousModelInput : register(t1);
RWStructuredBuffer<NearIdenticalTileEvidence> NearIdenticalTileOutput : register(u3);

cbuffer NearIdenticalFusedMode : register(b1) {
    uint NearIdenticalEvidenceEnabled;
    uint3 NearIdenticalFusedModeReserved;
};

groupshared uint4 FusedNearIdenticalGroupPrimary[NEAR_IDENTICAL_TILE_THREADS];

uint FusedNearIdenticalTileAxisGroups(uint dimension) {
    return dimension / 16u + (dimension % 16u != 0u ? 1u : 0u);
}

[numthreads(16, 16, 1)]
void fused_main(
    uint3 dispatch_thread : SV_DispatchThreadID,
    uint3 group_thread : SV_GroupThreadID,
    uint3 group_id : SV_GroupID) {
    bool in_bounds = dispatch_thread.x < target_w && dispatch_thread.y < target_h;
    if (in_bounds) {
        // Calling the calibrated entry point preserves its output bits. Out-of-bounds lanes skip
        // the call but still participate in the group barrier below.
        main(dispatch_thread);
    }

    // Force-infer/bootstrap/debug frames share this one calibrated producer binary but do not
    // pay for comparison or reduction. Write one zero record per group before the uniform return:
    // force ignores it, while an accidental disabled-mode resolve must fail open to inference
    // instead of consuming stale evidence from a prior frame.
    if (NearIdenticalEvidenceEnabled == 0u) {
        if (group_thread.x == 0u && group_thread.y == 0u) {
            NearIdenticalTileEvidence disabled_evidence;
            disabled_evidence.admitted = 0u;
            disabled_evidence.medium_changed = 0u;
            disabled_evidence.strong_changed = 0u;
            disabled_evidence.nonfinite = 0u;
            uint disabled_tile_group_width = FusedNearIdenticalTileAxisGroups(target_w);
            NearIdenticalTileOutput[
                group_id.y * disabled_tile_group_width + group_id.x] =
                disabled_evidence;
        }
        return;
    }

    // The comparison must observe the same FP32 words that TensorRT will consume, rather than a
    // compiler-local pre-store expression. Synchronize the UAV writes before reading them back.
    DeviceMemoryBarrierWithGroupSync();

    uint linear_thread = group_thread.y * 16u + group_thread.x;
    uint4 primary = 0u;
    // Read the producer's committed exclusion bit rather than reconstructing admission only from
    // tensor geometry. This also fails malformed source-region b2 evidence open to inference:
    // canonical preprocessing writes exclusion=1 when the retained physical rectangle is invalid.
    bool admitted = in_bounds && OutputTensorExclusion[dispatch_thread.xy] == 0u;
    if (admitted) {
        uint plane = target_w * target_h;
        uint index = dispatch_thread.y * target_w + dispatch_thread.x;
        float3 current_nchw = float3(
            OutputBuffer[index],
            OutputBuffer[index + plane],
            OutputBuffer[index + 2u * plane]);
        float3 previous_nchw = float3(
            PreviousModelInput[index],
            PreviousModelInput[index + plane],
            PreviousModelInput[index + 2u * plane]);
        primary = NearIdenticalClassifyModelInput(
            current_nchw, previous_nchw, true);
    }
    FusedNearIdenticalGroupPrimary[linear_thread] = primary;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = NEAR_IDENTICAL_TILE_THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (linear_thread < stride) {
            FusedNearIdenticalGroupPrimary[linear_thread] +=
                FusedNearIdenticalGroupPrimary[linear_thread + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (linear_thread == 0u) {
        NearIdenticalTileEvidence evidence;
        evidence.admitted = FusedNearIdenticalGroupPrimary[0].x;
        evidence.medium_changed = FusedNearIdenticalGroupPrimary[0].y;
        evidence.strong_changed = FusedNearIdenticalGroupPrimary[0].z;
        evidence.nonfinite = FusedNearIdenticalGroupPrimary[0].w;
        uint tile_group_width = FusedNearIdenticalTileAxisGroups(target_w);
        NearIdenticalTileOutput[group_id.y * tile_group_width + group_id.x] = evidence;
    }
}
