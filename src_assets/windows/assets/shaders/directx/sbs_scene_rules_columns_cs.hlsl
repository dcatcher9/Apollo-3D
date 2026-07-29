StructuredBuffer<float> AnalysisGrid : register(t0);
StructuredBuffer<float> DenseOutput : register(t1);
StructuredBuffer<float> CurrentDepthHistory : register(t3);
RWStructuredBuffer<float> RuleSummary : register(u0);

#include "include/sbs_scene_controller_constants.hlsl"
#include "include/sbs_scene_rules_summary.hlsl"

float Dense(uint channel, uint2 cell) {
    return DenseOutput[SceneAnalysisIndex(channel, cell)];
}

float Analysis(uint channel, uint2 cell) {
    return AnalysisGrid[SceneAnalysisIndex(channel, cell)];
}

[numthreads(128, 1, 1)]
void main(uint3 thread_id : SV_GroupThreadID) {
    const uint x = thread_id.x;
    const uint canvas = SBS_SCENE_ANALYSIS_CANVAS_SIZE;

    float viewport_valid = 0.0f;
    float gutter_cells = 0.0f;
    float video_support = 0.0f;
    float video_weight = 0.0f;

    [loop]
    for (uint y = 0u; y < canvas; ++y) {
        const uint2 cell = uint2(x, y);
        const float viewport_flag = Analysis(
            SBS_SCENE_ANALYSIS_GRID_VIEWPORT_VALID,
            cell);
        if (viewport_flag <= 0.5f) {
            continue;
        }

        viewport_valid += 1.0f;
        const float unsafe = max(
            Dense(
                SBS_SCENE_DENSE_OUT_ADVERTISEMENT_COMPETITOR_UNSAFE,
                cell),
            Dense(
                SBS_SCENE_DENSE_OUT_BROWSER_SYSTEM_CHROME,
                cell));
        const float video = Dense(
            SBS_SCENE_DENSE_OUT_PRIMARY_PLAYING_VIDEO,
            cell);
        gutter_cells +=
            Dense(SBS_SCENE_DENSE_OUT_STABLE_GUTTER, cell) > 0.70f ?
                1.0f :
                0.0f;
        if (video > 0.18f && unsafe < 0.65f) {
            video_support += 1.0f;
            video_weight += video * (1.0f - unsafe);
        }
    }

    RuleSummary[SbsRuleColumnIndex(x, SBS_RULE_COLUMN_VIEWPORT_VALID)] =
        viewport_valid;
    RuleSummary[SbsRuleColumnIndex(x, SBS_RULE_COLUMN_GUTTER)] =
        gutter_cells;
    RuleSummary[SbsRuleColumnIndex(x, SBS_RULE_COLUMN_VIDEO_SUPPORT)] =
        video_support;
    RuleSummary[SbsRuleColumnIndex(x, SBS_RULE_COLUMN_VIDEO_WEIGHT)] =
        video_weight;

    // This row-major fold is the sole authority for validation, depth, and scroll aggregates.
    const uint row_y = thread_id.x;
    float row_main_valid = 0.0f;
    float row_required_invalid = 0.0f;
    float row_depth_coverage = 0.0f;
    float row_depth_confidence = 0.0f;
    float row_scroll_weight = 0.0f;
    float row_scroll_signed = 0.0f;
    float row_scroll_support = 0.0f;
    [loop]
    for (uint row_x = 0u; row_x < canvas; ++row_x) {
        const uint2 cell = uint2(row_x, row_y);
        const float viewport_flag = Analysis(
            SBS_SCENE_ANALYSIS_GRID_VIEWPORT_VALID,
            cell);
        if (!isfinite(viewport_flag) || viewport_flag < 0.0f) {
            row_required_invalid += 1.0f;
            continue;
        }
        if (viewport_flag <= 0.5f) {
            continue;
        }
        const float output_sentinel = Dense(
            SBS_SCENE_DENSE_OUT_UNKNOWN_BACKGROUND,
            cell);
        if (!isfinite(output_sentinel) || output_sentinel < 0.0f) {
            row_required_invalid += 1.0f;
            continue;
        }
        const float depth_confidence =
            CurrentDepthHistory[SceneAnalysisIndex(
                SBS_SCENE_DEPTH_HISTORY_VALID_DEPTH_CONFIDENCE,
                cell)];
        if (!isfinite(depth_confidence)) {
            row_required_invalid += 1.0f;
            continue;
        }
        const float roi_coverage = Analysis(
            SBS_SCENE_ANALYSIS_GRID_CURRENT_ROI_COVERAGE,
            cell);
        if (!isfinite(roi_coverage)) {
            row_required_invalid += 1.0f;
            continue;
        }
        row_main_valid += 1.0f;
        if (roi_coverage > 0.5f) {
            row_depth_coverage += 1.0f;
            row_depth_confidence += saturate(depth_confidence);
        }
        const float support = Dense(
            SBS_SCENE_DENSE_OUT_SCROLL_SUPPORT,
            cell);
        const float motion_y = Dense(
            SBS_SCENE_DENSE_OUT_LOCAL_MOTION_Y,
            cell);
        if (!isfinite(support) || !isfinite(motion_y)) {
            row_required_invalid += 1.0f;
            continue;
        }
        row_scroll_weight += support * abs(motion_y);
        row_scroll_signed += support * motion_y;
        row_scroll_support += support > 0.20f ? 1.0f : 0.0f;
    }
    RuleSummary[SbsRuleGlobalRowIndex(
        row_y,
        SBS_RULE_GLOBAL_ROW_MAIN_VALID)] = row_main_valid;
    RuleSummary[SbsRuleGlobalRowIndex(
        row_y,
        SBS_RULE_GLOBAL_ROW_REQUIRED_INVALID)] = row_required_invalid;
    RuleSummary[SbsRuleGlobalRowIndex(
        row_y,
        SBS_RULE_GLOBAL_ROW_DEPTH_COVERAGE)] = row_depth_coverage;
    RuleSummary[SbsRuleGlobalRowIndex(
        row_y,
        SBS_RULE_GLOBAL_ROW_DEPTH_CONFIDENCE)] = row_depth_confidence;
    RuleSummary[SbsRuleGlobalRowIndex(
        row_y,
        SBS_RULE_GLOBAL_ROW_SCROLL_WEIGHT)] = row_scroll_weight;
    RuleSummary[SbsRuleGlobalRowIndex(
        row_y,
        SBS_RULE_GLOBAL_ROW_SCROLL_SIGNED)] = row_scroll_signed;
    RuleSummary[SbsRuleGlobalRowIndex(
        row_y,
        SBS_RULE_GLOBAL_ROW_SCROLL_SUPPORT)] = row_scroll_support;
}
