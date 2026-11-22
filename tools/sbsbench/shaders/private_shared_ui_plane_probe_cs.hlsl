// SPDX-License-Identifier: GPL-3.0-only
// Private diagnostic: compare identical saved Base under cover and actual selected
// native-footprint populations. This shader never publishes SCP/UIP, edits a mask,
// or authorizes rendering. Reuse the real production source/receipt/RGB gates.
// The fixture stages this exact entry beside an exact canonical DirectX source
// tree. Production nested includes intentionally resolve from that logical root.
#include "host_sbs_shared_ui_warp.hlsl"

StructuredBuffer<uint4> ProbeCovers : register(t15);
RWTexture2D<uint> ProbeFootprint : register(u3);
RWStructuredBuffer<uint> ProbeResults : register(u4);
cbuffer PrivatePlaneProbe : register(b7) {
    uint4 ProbeShape; // field width/height, analysis-source width/height
    uint4 ProbeControl; // cover count, native horizontal collar (0/1), zero, zero
};

bool ProbeSourcesValid() {
    uint2 source_size, field_size, footprint_size;
    float plane;
    uint covers, stride;
    BaseParallax.GetDimensions(field_size.x, field_size.y);
    ProbeFootprint.GetDimensions(footprint_size.x, footprint_size.y);
    ProbeCovers.GetDimensions(covers, stride);
    if (!SubtitleRenderAuthorized(source_size, plane, true) || !SelectedMatteAuthorized() ||
        any(ProbeShape.xy != field_size) || any(ProbeShape.xy != footprint_size) ||
        any(ProbeShape.zw != subtitle_render_source.zw) || ProbeShape.z > 5120u ||
        ProbeControl.x == 0u || ProbeControl.x > 8u || covers != ProbeControl.x || stride != 16u ||
        ProbeControl.x != SubtitleCondition[2u] || ProbeControl.y > 1u || any(ProbeControl.zw != 0u)) return false;
    uint4 content = uint4(ContourIdentity[16u], ContourIdentity[17u],
                         ContourIdentity[18u], ContourIdentity[19u]);
    [loop] for (uint i = 0u; i < ProbeControl.x; ++i) {
        uint4 cover = ProbeCovers[i];
        if (any(cover.xy >= cover.zw) || any(cover.xy < content.xy) || any(cover.zw > content.zw)) return false;
    }
    return true;
}

void ProbeMarkCenter(int2 native_center) {
    uint2 source_size;SourceColor.GetDimensions(source_size.x, source_size.y);
    int2 full_center = clamp(native_center + int2(subtitle_render_source.xy), int2(0, 0), int2(source_size) - 1);
    float2 uv = (float2(full_center) + 0.5f) / float2(source_size);
    // Preserve the production SharedOriginalParallax operation order, rather
    // than replacing its ROI projection with an algebraic content-ratio map.
    if (video_roi_active != 0.0f) {
        float2 extent = video_roi_source_uv.zw - video_roi_source_uv.xy;
        float2 content_min = float2(video_roi_tensor_content_rect.xy) / float2(ProbeShape.xy);
        float2 content_max = float2(video_roi_tensor_content_rect.zw) / float2(ProbeShape.xy);
        float2 scale = (content_max - content_min) / extent;
        float2 bias = content_min - video_roi_source_uv.xy * scale;
        float projected_y = clamp(uv.y, video_roi_source_uv.y, video_roi_source_uv.w);
        float projected_x = clamp(uv.x, video_roi_source_uv.x, video_roi_source_uv.z);
        uv = float2(projected_x * scale.x + bias.x, projected_y * scale.y + bias.y);
    }
    // SharedNativeBaseSample clamps taps to the entire tensor. Near a content
    // edge, positive taps can belong to padding and must remain observable.
    float2 position = saturate(uv) * float2(ProbeShape.xy) - 0.5f;
    int2 low = int2(floor(position));
    float2 fraction = position - float2(low);
    [unroll] for (int y = 0; y < 2; ++y) [unroll] for (int x = 0; x < 2; ++x) {
        float wx = x == 0 ? 1.0f - fraction.x : fraction.x;
        float wy = y == 0 ? 1.0f - fraction.y : fraction.y;
        if (wx > 0.0f && wy > 0.0f) {
            int2 at = clamp(low + int2(x, y), int2(0, 0), int2(ProbeShape.xy) - 1);
            InterlockedOr(ProbeFootprint[at], 1u);
        }
    }
}

groupshared uint ProbeGroupAuthorized;
[numthreads(16, 16, 1)]
void footprint_main(uint3 id : SV_DispatchThreadID, uint3 lane : SV_GroupThreadID) {
    if (all(lane.xy == 0u)) ProbeGroupAuthorized = ProbeSourcesValid() ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    if (ProbeGroupAuthorized == 0u || id.x >= ContourIdentity[8u] || id.y >= ContourIdentity[11u]) return;
    int2 native_center = int2(id.xy) + int2(0, (int)ContourIdentity[10u]);
    int2 full_center = native_center + int2(subtitle_render_source.xy);
    if (!CurrentGlyphContributor(full_center, true)) return;
    ProbeMarkCenter(native_center);
    if (ProbeControl.y != 0u) {
        ProbeMarkCenter(native_center + int2(-1, 0));
        ProbeMarkCenter(native_center + int2(1, 0));
    }
}

// Copied candidate arithmetic only for this independent diagnostic; production
// policy stays in host_sbs_subtitle_locator_cs.hlsl. No temporal state is advanced.
float ProbeCandidate(uint index, uint maximum_index) {
    return index == maximum_index ? V2_DIRECT_CONTAINER_LIMIT :
        min((4.0f * (float)index) / (float)ProbeShape.z, V2_DIRECT_CONTAINER_LIMIT);
}

uint ProbeFirstNonconflicting(float desired, uint maximum_index) {
    if (desired <= 0.0f) return 0u;
    if (desired > V2_DIRECT_CONTAINER_LIMIT) return maximum_index + 1u;
    uint index = min((uint)ceil(desired * (float)ProbeShape.z * 0.25f), maximum_index);
    if (index != 0u && desired <= ProbeCandidate(index - 1u, maximum_index)) --index;
    if (desired > ProbeCandidate(index, maximum_index)) ++index;
    return index;
}

groupshared uint ProbeHistogram[54];
groupshared uint ProbeCovered, ProbeInvalid, ProbeAppliedConflicts, ProbePadding;
[numthreads(256, 1, 1)]
void reduce_main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    // Exactly two groups: independent cover and selected-native-footprint populations.
    uint mode = group.x;
    if (mode >= 2u) return;
    if (lane < 54u) ProbeHistogram[lane] = 0u;
    if (lane == 0u) {
        ProbeCovered = ProbeInvalid = ProbeAppliedConflicts = ProbePadding = 0u;
        ProbeGroupAuthorized = ProbeSourcesValid() ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    uint maximum_index = min((ProbeShape.z + 99u) / 100u, 52u);
    if (ProbeGroupAuthorized != 0u) {
        uint covered = 0u, invalid = 0u, applied_conflicts = 0u, padding = 0u;
        uint4 content = uint4(ContourIdentity[16u], ContourIdentity[17u],
                             ContourIdentity[18u], ContourIdentity[19u]);
        float applied = asfloat(SubtitleCondition[5u]);
        float clearance = 2.0f / (float)ProbeShape.z;
        [loop] for (uint cell = lane; cell < ProbeShape.x * ProbeShape.y; cell += 256u) {
            uint2 at = uint2(cell % ProbeShape.x, cell / ProbeShape.x);
            bool member = false;
            if (mode == 0u) {
                [loop] for (uint i = 0u; i < ProbeControl.x; ++i) {
                    uint4 cover = ProbeCovers[i];
                    member = member || (all(at >= cover.xy) && all(at < cover.zw));
                }
            } else member = ProbeFootprint[at] != 0u;
            if (!member) continue;
            ++covered;
            if (any(at < content.xy) || any(at >= content.zw)) ++padding;
            float base = BaseParallax.Load(int3(at, 0));
            if (!isfinite(base) || abs(base) > V2_DIRECT_CONTAINER_LIMIT) { ++invalid; continue; }
            precise float desired = base + clearance;
            if (desired > applied) ++applied_conflicts;
            uint first_clear = ProbeFirstNonconflicting(desired, maximum_index);
            InterlockedAdd(ProbeHistogram[first_clear], 1u);
        }
        InterlockedAdd(ProbeCovered, covered);
        InterlockedAdd(ProbeInvalid, invalid);
        InterlockedAdd(ProbeAppliedConflicts, applied_conflicts);
        InterlockedAdd(ProbePadding, padding);
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane != 0u) return;
    uint offset = mode * 64u;
    [unroll] for (uint word = 0u; word < 64u; ++word) ProbeResults[offset + word] = 0u;
    ProbeResults[offset] = 0x31504255u; // UBP1, private diagnostic schema
    ProbeResults[offset + 1u] = 1u;
    ProbeResults[offset + 2u] = ProbeGroupAuthorized;
    ProbeResults[offset + 3u] = ProbeCovered;
    ProbeResults[offset + 4u] = ProbeInvalid;
    ProbeResults[offset + 8u] = ProbeAppliedConflicts;
    ProbeResults[offset + 9u] = maximum_index;
    ProbeResults[offset + 63u] = ProbePadding;
    if (ProbeGroupAuthorized == 0u || ProbeInvalid != 0u) return;
    uint content_area = (ContourIdentity[18u] - ContourIdentity[16u]) *
                        (ContourIdentity[19u] - ContourIdentity[17u]);
    uint conflicts = ProbeHistogram[maximum_index + 1u];
    [loop] for (int level = (int)maximum_index; level >= 0; --level) {
        ProbeResults[offset + 10u + (uint)level] = conflicts;
        conflicts += ProbeHistogram[(uint)level];
    }
    uint required = maximum_index;
    bool found = false;
    [loop] for (uint candidate = 0u; candidate <= maximum_index; ++candidate) {
        uint bad = ProbeResults[offset + 10u + candidate];
        if (!found && bad * 5u <= ProbeCovered && bad * 50u <= content_area) {
            required = candidate; found = true;
        }
    }
    ProbeResults[offset + 5u] = required;
    ProbeResults[offset + 6u] = asuint(ProbeCandidate(required, maximum_index));
    ProbeResults[offset + 7u] = found ? 0u : 1u;
}
