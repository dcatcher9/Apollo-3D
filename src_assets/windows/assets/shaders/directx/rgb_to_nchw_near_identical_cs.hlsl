// Common-path fusion for the calibrated RGB-to-NCHW producer and the GPU-only near-identical
// tile evidence. The included producer owns resize, transfer, normalization, and exclusion.
// The authored entries observe committed NCHW and exclusion words. Compact entries
// use the producer's local admission predicate and publish its derived domain once.

#include "rgb_to_nchw_cs.hlsl"
#include "include/host_sbs_near_identical_compare.hlsl"

StructuredBuffer<float> PreviousModelInput : register(t1);
RWStructuredBuffer<NearIdenticalTileEvidence> NearIdenticalTileOutput : register(u2);

cbuffer NearIdenticalFusedMode : register(b1) {
    uint NearIdenticalEvidenceEnabled;
    uint3 NearIdenticalFusedModeReserved;
};

groupshared uint4 FusedNearIdenticalGroupPrimary[NEAR_IDENTICAL_TILE_THREADS];

uint FusedNearIdenticalTileAxisGroups(uint dimension) {
    return dimension / 16u + (dimension % 16u != 0u ? 1u : 0u);
}

// Bounds mirror the canonical full-grid and letterbox footprints. Clip the group to the output
// before projecting it so every nonempty edge group fits the same sampling contract.
bool ModelSourceTileBounds(uint2 group, out int2 origin, out uint2 extent, out uint2 source_size) {
    uint2 texture_size;
    InputTexture.GetDimensions(texture_size.x, texture_size.y);
    source_size = DepthSourceRegionSize();
    origin = 0;
    extent = 0u;
    uint2 target_size = uint2(target_w, target_h);
    uint2 group_first = group * 16u;
    if (!DepthSourceRegionValid(texture_size) || !DepthAnalysisContentValid() ||
        any(group_first >= target_size)) {
        return false;
    }

    uint2 group_end = min(group_first + 16u, target_size);
    bool full_content = analysis_content_left == 0u && analysis_content_top == 0u &&
        analysis_content_right == target_w && analysis_content_bottom == target_h;
    float2 source_lo = 0.0f;
    float2 source_hi = 0.0f;
    if (full_content) {
        if (any(source_size < target_size)) return false;
        float2 source_scale = float2(source_size) / float2(target_size);
        source_lo = float2(group_first) * source_scale;
        source_hi = float2(group_end) * source_scale;
    } else {
        float2 content_min = DepthAnalysisContentMinUv();
        float2 content_scale = DepthAnalysisContentScale();
        if (any(float2(source_size) < float2(target_size) * content_scale)) return false;
        uint2 first_cell = DepthAnalysisClampCell(group_first);
        uint2 last_cell = DepthAnalysisClampCell(group_end - 1u);
        source_lo = ((float2(first_cell) / float2(target_size)) - content_min) /
            content_scale * float2(source_size);
        source_hi = ((float2(last_cell + 1u) / float2(target_size)) - content_min) /
            content_scale * float2(source_size);
    }
    origin = max(int2(floor(source_lo)), int2(0, 0));
    int2 end = min(int2(ceil(source_hi)), int2(source_size));
    if (any(end <= origin) || any(end - origin > (int)ModelSourceTileSide)) return false;
    extent = uint2(end - origin);
    return true;
}

void WriteNearIdenticalTileEvidence(
    uint3 dispatch_thread,
    uint3 group_thread,
    uint3 group_id,
    bool in_bounds,
    bool compact_domain,
    bool producer_admitted) {
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
    // Both paths include actual retained-source validity, not only the content rectangle.
    // A compact entry uses this thread's producer result, never lane zero's new descriptor:
    // there is no cross-group synchronization inside this dispatch.
    bool admitted = producer_admitted;
    if (!compact_domain) admitted = in_bounds && OutputTensorExclusion[dispatch_thread.xy] == 0u;
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

[numthreads(16, 16, 1)]
void fused_main(
    uint3 dispatch_thread : SV_DispatchThreadID,
    uint3 group_thread : SV_GroupThreadID,
    uint3 group_id : SV_GroupID) {
    bool in_bounds = dispatch_thread.x < target_w && dispatch_thread.y < target_h;
    if (in_bounds) main(dispatch_thread);
    WriteNearIdenticalTileEvidence(dispatch_thread, group_thread, group_id, in_bounds, false, false);
}

void WriteFusedTiledInput(
    uint3 dispatch_thread,
    uint3 group_thread,
    uint3 group_id,
    bool compact_domain) {
    bool in_bounds = dispatch_thread.x < target_w && dispatch_thread.y < target_h;
    int2 tile_origin;
    uint2 tile_extent, source_size;
    bool tile_valid = ModelSourceTileBounds(group_id.xy, tile_origin, tile_extent, source_size);
    [branch]
    if (tile_valid) {
        // Let fringe lanes load the source too. No lane may bypass this group's barrier.
        [loop]
        for (uint y = group_thread.y; y < tile_extent.y; y += 16u) {
            [loop]
            for (uint x = group_thread.x; x < tile_extent.x; x += 16u) {
                ModelSourceTile[y * ModelSourceTileSide + x] =
                    LoadModelColorDirect(tile_origin + int2(x, y), source_size);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    bool admitted = false;
    if (in_bounds) {
        admitted = WriteModelInput(dispatch_thread, tile_valid, tile_origin, tile_extent, compact_domain);
    }
    WriteNearIdenticalTileEvidence(dispatch_thread, group_thread, group_id, in_bounds, compact_domain, admitted);
}

[numthreads(16, 16, 1)]
void fused_tiled_main(
    uint3 dispatch_thread : SV_DispatchThreadID,
    uint3 group_thread : SV_GroupThreadID,
    uint3 group_id : SV_GroupID) {
    WriteFusedTiledInput(dispatch_thread, group_thread, group_id, false);
}

[numthreads(16, 16, 1)]
void fused_domain_main(
    uint3 dispatch_thread : SV_DispatchThreadID,
    uint3 group_thread : SV_GroupThreadID,
    uint3 group_id : SV_GroupID) {
    bool in_bounds = dispatch_thread.x < target_w && dispatch_thread.y < target_h;
    bool admitted = false;
    if (in_bounds) admitted = WriteModelInput(dispatch_thread, false, int2(0, 0), uint2(0u, 0u), true);
    WriteNearIdenticalTileEvidence(dispatch_thread, group_thread, group_id, in_bounds, true, admitted);
}

[numthreads(16, 16, 1)]
void fused_tiled_domain_main(
    uint3 dispatch_thread : SV_DispatchThreadID,
    uint3 group_thread : SV_GroupThreadID,
    uint3 group_id : SV_GroupID) {
    WriteFusedTiledInput(dispatch_thread, group_thread, group_id, true);
}
