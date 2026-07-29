StructuredBuffer<float> SceneRgb : register(t0);
StructuredBuffer<float> SceneOrdinal : register(t1);
StructuredBuffer<float> PreviousLayoutHistory : register(t2);
StructuredBuffer<uint4> PreviousRuleState : register(t3);
RWStructuredBuffer<float> AnalysisGrid : register(u0);

#include "include/sbs_scene_controller_constants.hlsl"

uint2 ClampAppearancePixel(int2 pixel) {
    uint2 viewport_origin;
    uint2 viewport_size;
    SceneViewport(
        SBS_SCENE_APPEARANCE_CANVAS_SIZE,
        viewport_origin,
        viewport_size);
    return uint2(clamp(
        pixel,
        int2(viewport_origin),
        int2(viewport_origin + viewport_size - 1u)));
}

float3 LoadSceneRgb(int2 pixel) {
    uint2 p = ClampAppearancePixel(pixel);
    return float3(
        SceneRgb[SceneAppearanceIndex(0u, p)],
        SceneRgb[SceneAppearanceIndex(1u, p)],
        SceneRgb[SceneAppearanceIndex(2u, p)]);
}

float LoadOrdinal(int2 pixel) {
    uint2 p = ClampAppearancePixel(pixel);
    return SceneOrdinal[
        p.y * SBS_SCENE_APPEARANCE_CANVAS_SIZE + p.x];
}

float Luminance(float3 rgb) {
    return dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
}

float PreviousRuleStateWord(uint word) {
    uint4 value = PreviousRuleState[word / 4u];
    return asfloat(value[word & 3u]);
}

uint PreviousRuleStateUint(uint word) {
    uint4 value = PreviousRuleState[word / 4u];
    return value[word & 3u];
}

uint EffectiveResetFlags() {
    uint flags = scene_reset_flags;
    if (PreviousRuleStateWord(
            SBS_SCENE_RULE_STATE_WORD_OUTPUT_VALID) <= 0.5f) {
        flags |= PreviousRuleStateUint(
            SBS_SCENE_RULE_STATE_WORD_RESET_FLAGS);
    }
    return flags;
}

float CurrentRoiCoverage(float2 viewport_uv) {
    const uint roi_invalidating_resets =
        SBS_SCENE_RESET_FLAGS_LAYOUT |
        SBS_SCENE_RESET_FLAGS_GEOMETRY |
        SBS_SCENE_RESET_FLAGS_BACKEND |
        SBS_SCENE_RESET_FLAGS_DISPLAY_OR_HDR;
    if ((EffectiveResetFlags() & roi_invalidating_resets) != 0u) {
        return 1.0f;
    }
    uint state_flags = PreviousRuleStateUint(
        SBS_SCENE_RULE_STATE_WORD_STATE_FLAGS);
    if ((state_flags & SBS_SCENE_STATE_FLAGS_ROI_LOCKED) == 0u) {
        return 1.0f;
    }

    float4 committed_roi = float4(
        PreviousRuleStateWord(
            SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_X0),
        PreviousRuleStateWord(
            SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_Y0),
        PreviousRuleStateWord(
            SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_X1),
        PreviousRuleStateWord(
            SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_Y1));
    bool covered =
        all(viewport_uv >= committed_roi.xy) &&
        all(viewport_uv < committed_roi.zw);
    return covered ? 1.0f : 0.0f;
}

uint CensusOrdinal(int2 center, out uint reliable_comparisons) {
    static const int2 offsets[8] = {
        int2(-2, 0), int2(2, 0), int2(0, -2), int2(0, 2),
        int2(-2, -2), int2(2, -2), int2(-2, 2), int2(2, 2)
    };
    float center_value = LoadOrdinal(center);
    uint signature = 0u;
    reliable_comparisons = 0u;
    [unroll]
    for (uint index = 0u; index < 8u; ++index) {
        float neighbor = LoadOrdinal(center + offsets[index]);
        float contrast = abs(center_value - neighbor);
        float floor_value =
            max(1e-5f, 0.005f * max(abs(center_value), abs(neighbor)));
        if (contrast >= floor_value) {
            reliable_comparisons += 1u;
            if (center_value > neighbor) {
                signature |= 1u << index;
            }
        }
    }
    return signature;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    const uint canvas = SBS_SCENE_ANALYSIS_CANVAS_SIZE;
    if (dispatch_id.x >= canvas || dispatch_id.y >= canvas) {
        return;
    }

    uint2 cell = dispatch_id.xy;
    float2 viewport_uv;
    bool valid = SceneViewportUv(cell, canvas, viewport_uv);
    [unroll]
    for (uint channel = 0u;
         channel < SBS_SCENE_ANALYSIS_GRID_CHANNEL_COUNT;
         ++channel) {
        AnalysisGrid[SceneAnalysisIndex(channel, cell)] = 0.0f;
    }
    if (!valid) {
        return;
    }

    uint2 appearance_origin;
    uint2 appearance_size;
    SceneViewport(
        SBS_SCENE_APPEARANCE_CANVAS_SIZE,
        appearance_origin,
        appearance_size);
    int2 center = int2(
        appearance_origin +
        min(
            uint2(viewport_uv * float2(appearance_size)),
            appearance_size - 1u));
    bool source_finite = true;
    static const int2 validity_neighborhood[9] = {
        int2(0, 0),
        int2(-2, 0), int2(2, 0), int2(0, -2), int2(0, 2),
        int2(-2, -2), int2(2, -2), int2(-2, 2), int2(2, 2)
    };
    [unroll]
    for (uint index = 0u; index < 9u; ++index) {
        float3 rgb = LoadSceneRgb(center + validity_neighborhood[index]);
        float ordinal =
            LoadOrdinal(center + validity_neighborhood[index]);
        source_finite =
            source_finite &&
            !any(isnan(rgb)) && !any(isinf(rgb)) &&
            !isnan(ordinal) && !isinf(ordinal);
    }
    if (!source_finite) {
        // A negative validity sentinel survives the finite evidence pass and makes the global
        // resolver abstain. It is distinct from ordinary letterbox padding (exact zero).
        AnalysisGrid[SceneAnalysisIndex(
            SBS_SCENE_ANALYSIS_GRID_VIEWPORT_VALID, cell)] = -1.0f;
        return;
    }
    float3 center_rgb = LoadSceneRgb(center);
    float center_luma = Luminance(center_rgb);
    static const int2 neighborhood[8] = {
        int2(-2, 0), int2(2, 0), int2(0, -2), int2(0, 2),
        int2(-2, -2), int2(2, -2), int2(-2, 2), int2(2, 2)
    };
    float sum = center_luma;
    float sum_sq = center_luma * center_luma;
    float edge_sum = 0.0f;
    float edge_max = 0.0f;
    [unroll]
    for (uint index = 0u; index < 8u; ++index) {
        float neighbor_luma = Luminance(
            LoadSceneRgb(center + neighborhood[index]));
        sum += neighbor_luma;
        sum_sq += neighbor_luma * neighbor_luma;
        float edge = abs(center_luma - neighbor_luma);
        edge_sum += edge;
        edge_max = max(edge_max, edge);
    }
    float mean = sum / 9.0f;
    float variance = max(sum_sq / 9.0f - mean * mean, 0.0f);
    float texture_evidence = saturate(
        2.5f * sqrt(variance) + 1.5f * edge_sum / 8.0f +
        0.5f * edge_max);
    float chroma =
        max(center_rgb.r, max(center_rgb.g, center_rgb.b)) -
        min(center_rgb.r, min(center_rgb.g, center_rgb.b));

    uint reliable = 0u;
    uint signature = CensusOrdinal(center, reliable);
    float signature_value = (float)signature / 255.0f;
    float activity = 0.0f;
    uint effective_reset_flags = EffectiveResetFlags();
    if (scene_history_valid != 0u &&
        (effective_reset_flags & SBS_SCENE_RESET_FLAGS_LAYOUT) == 0u &&
        reliable >= 4u) {
        uint previous_signature = (uint)round(saturate(
            PreviousLayoutHistory[SceneAnalysisIndex(
                SBS_SCENE_LAYOUT_HISTORY_PREVIOUS_LUMINANCE_ORDINAL,
                cell)]) * 255.0f);
        activity =
            (float)countbits(signature ^ previous_signature) / 8.0f;
    }

    AnalysisGrid[SceneAnalysisIndex(
        SBS_SCENE_ANALYSIS_GRID_VIEWPORT_VALID, cell)] = 1.0f;
    AnalysisGrid[SceneAnalysisIndex(
        SBS_SCENE_ANALYSIS_GRID_LUMINANCE_ORDINAL, cell)] =
        signature_value;
    AnalysisGrid[SceneAnalysisIndex(
        SBS_SCENE_ANALYSIS_GRID_LOCAL_LUMINANCE_VARIANCE, cell)] =
        saturate(4.0f * sqrt(variance));
    AnalysisGrid[SceneAnalysisIndex(
        SBS_SCENE_ANALYSIS_GRID_DENSE_EDGE_TEXTURE, cell)] =
        texture_evidence;
    AnalysisGrid[SceneAnalysisIndex(
        SBS_SCENE_ANALYSIS_GRID_CHROMA_SATURATION, cell)] =
        saturate(2.0f * chroma);
    AnalysisGrid[SceneAnalysisIndex(
        SBS_SCENE_ANALYSIS_GRID_TEMPORAL_ACTIVITY_OCCUPANCY, cell)] =
        activity;
    AnalysisGrid[SceneAnalysisIndex(
        SBS_SCENE_ANALYSIS_GRID_CURRENT_ROI_COVERAGE, cell)] =
        CurrentRoiCoverage(viewport_uv);
    AnalysisGrid[SceneAnalysisIndex(
        SBS_SCENE_ANALYSIS_GRID_VIEWPORT_X, cell)] = viewport_uv.x;
    AnalysisGrid[SceneAnalysisIndex(
        SBS_SCENE_ANALYSIS_GRID_VIEWPORT_Y, cell)] = viewport_uv.y;
    AnalysisGrid[SceneAnalysisIndex(
        SBS_SCENE_ANALYSIS_GRID_INTERACTION_HEATMAP, cell)] = 0.0f;
}
