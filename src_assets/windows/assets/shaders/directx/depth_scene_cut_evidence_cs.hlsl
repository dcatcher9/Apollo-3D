// Live Host SBS V2 scene-cut evidence. This is the cut-only counterpart of
// depth_subject_hist_cs: it preserves the calibrated depth, appearance, ordinal, exposure, and
// structural-support evidence exactly, but does not build subject/stretch histograms or measure
// legacy adaptive-pop edge risk.

Texture2D<float>         DepthTexture : register(t0);  // normalized cut-analysis depth
Texture2D<float>         PreviousDepth : register(t1);  // last structurally reliable endpoint
StructuredBuffer<float>  CurrentModelInput : register(t2);  // completed NCHW ImageNet input
StructuredBuffer<float>  PreviousModelInput : register(t3);
StructuredBuffer<float4> MinMaxEma : register(t4);  // w = current-frame validity
StructuredBuffer<float>  CurrentAppearanceOrdinal : register(t5);  // capture-domain maxRGB
StructuredBuffer<float>  PreviousAppearanceOrdinal : register(t6);
RWStructuredBuffer<uint> SceneCutEvidence : register(u0);

#include "include/depth_constants.hlsl"
#include "include/depth_scene_cut_evidence_contract.hlsl"

#define STRUCTURE_TILE_WIDTH 18
#define STRUCTURE_TILE_TEXELS (STRUCTURE_TILE_WIDTH * STRUCTURE_TILE_WIDTH)
static const int2 STRUCTURE_ORDINAL_OFFSETS[5] = {
    int2(0, 0), int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)
};

groupshared uint g_evidence[CUT_EVIDENCE_WORD_COUNT];
// A 16x16 group plus a one-pixel halo of the pre-tone-map point maxRGB ordinal signal.
groupshared float g_current_appearance_ordinal[STRUCTURE_TILE_TEXELS];
groupshared float g_previous_appearance_ordinal[STRUCTURE_TILE_TEXELS];

float3 CurrentModelColor(uint2 p) {
    uint plane = target_w * target_h;
    uint idx = p.y * target_w + p.x;
    return float3(
        CurrentModelInput[idx] * 0.229f + 0.485f,
        CurrentModelInput[idx + plane] * 0.224f + 0.456f,
        CurrentModelInput[idx + 2u * plane] * 0.225f + 0.406f);
}

float3 PreviousModelColor(uint2 p) {
    uint plane = target_w * target_h;
    uint idx = p.y * target_w + p.x;
    return float3(
        PreviousModelInput[idx] * 0.229f + 0.485f,
        PreviousModelInput[idx + plane] * 0.224f + 0.456f,
        PreviousModelInput[idx + 2u * plane] * 0.225f + 0.406f);
}

[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID,
          uint3 gid : SV_GroupID) {
    uint lin = tid.y * 16u + tid.x;
    if (lin < CUT_EVIDENCE_WORD_COUNT) {
        g_evidence[lin] = 0u;
    }

    // Cooperatively load the same 18x18 ordinal tile as the legacy evaluator. The final 68 halo
    // samples are handled by the first 68 threads' second iteration.
    for (uint tile_idx = lin; tile_idx < STRUCTURE_TILE_TEXELS; tile_idx += 256u) {
        uint tile_x = tile_idx % STRUCTURE_TILE_WIDTH;
        uint tile_y = tile_idx / STRUCTURE_TILE_WIDTH;
        int source_x = clamp((int)(gid.x * 16u + tile_x) - 1, 0, (int)target_w - 1);
        int source_y = clamp((int)(gid.y * 16u + tile_y) - 1, 0, (int)target_h - 1);
        uint source_index = (uint)source_y * target_w + (uint)source_x;
        g_current_appearance_ordinal[tile_idx] = CurrentAppearanceOrdinal[source_index];
        g_previous_appearance_ordinal[tile_idx] = PreviousAppearanceOrdinal[source_index];
    }
    GroupMemoryBarrierWithGroupSync();

    if (dtid.x < target_w && dtid.y < target_h && MinMaxEma[0].w > 0.5f) {
        InterlockedAdd(g_evidence[CUT_EVIDENCE_TOTAL], 1u);

        float d = DepthTexture[dtid.xy];
        if (abs(d - PreviousDepth[dtid.xy]) >= 0.05f) {
            InterlockedAdd(g_evidence[CUT_EVIDENCE_DEPTH_CHANGE], 1u);
        }

        uint tile_center = (tid.y + 1u) * STRUCTURE_TILE_WIDTH + tid.x + 1u;
        float3 current_color = CurrentModelColor(dtid.xy);
        float3 previous_color = PreviousModelColor(dtid.xy);
        float3 raw_rgb_delta = abs(current_color - previous_color);
        bool broad_rgb_changed =
            max(raw_rgb_delta.r, max(raw_rgb_delta.g, raw_rgb_delta.b)) >=
                RAW_RGB_PIXEL_DELTA;
        if (broad_rgb_changed) {
            InterlockedAdd(g_evidence[CUT_EVIDENCE_RAW_RGB_CHANGE], 1u);
            float current_ordinal = g_current_appearance_ordinal[tile_center];
            float previous_ordinal = g_previous_appearance_ordinal[tile_center];
            if (current_ordinal > previous_ordinal) {
                InterlockedAdd(g_evidence[CUT_EVIDENCE_BRIGHTNESS_RISE], 1u);
            } else if (current_ordinal < previous_ordinal) {
                InterlockedAdd(g_evidence[CUT_EVIDENCE_BRIGHTNESS_FALL], 1u);
            }
        }

        float current_samples[5];
        float previous_samples[5];
        [unroll]
        for (int sample_index = 0; sample_index < 5; ++sample_index) {
            int2 offset = STRUCTURE_ORDINAL_OFFSETS[sample_index];
            int tile_index =
                (int)tile_center + offset.y * STRUCTURE_TILE_WIDTH + offset.x;
            current_samples[sample_index] = g_current_appearance_ordinal[tile_index];
            previous_samples[sample_index] = g_previous_appearance_ordinal[tile_index];
        }
        uint current_comparisons = 0u;
        uint previous_comparisons = 0u;
        uint common_comparisons = 0u;
        uint ordering_flips = 0u;
        [unroll]
        for (int first = 0; first < 4; ++first) {
            [unroll]
            for (int second = first + 1; second < 5; ++second) {
                float current_delta = current_samples[first] - current_samples[second];
                float previous_delta = previous_samples[first] - previous_samples[second];
                bool current_reliable =
                    abs(current_delta) >= STRUCTURAL_ORDINAL_CONTRAST_FLOOR;
                bool previous_reliable =
                    abs(previous_delta) >= STRUCTURAL_ORDINAL_CONTRAST_FLOOR;
                current_comparisons += current_reliable ? 1u : 0u;
                previous_comparisons += previous_reliable ? 1u : 0u;
                bool common_reliable = current_reliable && previous_reliable;
                if (common_reliable) {
                    ++common_comparisons;
                    if ((current_delta < 0.0f) != (previous_delta < 0.0f)) {
                        ++ordering_flips;
                    }
                }
            }
        }
        if (current_comparisons >= STRUCTURAL_ORDINAL_MIN_COMMON) {
            InterlockedAdd(g_evidence[CUT_EVIDENCE_CURRENT_STRUCTURAL_SUPPORT], 1u);
        }
        if (previous_comparisons >= STRUCTURAL_ORDINAL_MIN_COMMON) {
            InterlockedAdd(g_evidence[CUT_EVIDENCE_PREVIOUS_STRUCTURAL_SUPPORT], 1u);
        }
        if (common_comparisons >= STRUCTURAL_ORDINAL_MIN_COMMON) {
            InterlockedAdd(g_evidence[CUT_EVIDENCE_COMMON_STRUCTURAL_SUPPORT], 1u);
        }
        if (common_comparisons >= STRUCTURAL_ORDINAL_MIN_COMMON &&
            ordering_flips >= STRUCTURAL_ORDINAL_MIN_FLIPS &&
            ordering_flips * 2u >= common_comparisons) {
            InterlockedAdd(g_evidence[CUT_EVIDENCE_STRUCTURAL_CHANGE], 1u);
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if (lin < CUT_EVIDENCE_WORD_COUNT && g_evidence[lin] > 0u) {
        InterlockedAdd(SceneCutEvidence[lin], g_evidence[lin]);
    }
}
