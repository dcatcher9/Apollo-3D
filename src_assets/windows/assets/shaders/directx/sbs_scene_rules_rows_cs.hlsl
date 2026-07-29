StructuredBuffer<float> AnalysisGrid : register(t0);
StructuredBuffer<float> DenseOutput : register(t1);
StructuredBuffer<float> CurrentLayoutHistory : register(t2);
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
void main(
    uint3 group_id : SV_GroupID,
    uint3 thread_id : SV_GroupThreadID)
{
    const uint region = group_id.x;
    const uint y = thread_id.x;
    const bool active =
        region < 2u &&
        SbsRuleLoadUint(
            RuleSummary,
            SbsRuleRegionIndex(region, SBS_RULE_REGION_ACTIVE)) != 0u;
    const uint first_x =
        active ?
            SbsRuleLoadUint(
                RuleSummary,
                SbsRuleRegionIndex(region, SBS_RULE_REGION_FIRST_X)) :
            0u;
    const uint end_x =
        active ?
            SbsRuleLoadUint(
                RuleSummary,
                SbsRuleRegionIndex(region, SBS_RULE_REGION_END_X)) :
            0u;

    float valid_count = 0.0f;
    float video_support = 0.0f;
    float video_weight = 0.0f;
    float content_sum = 0.0f;
    float unsafe_sum = 0.0f;
    float photo_sum = 0.0f;
    float content_cells = 0.0f;
    float2 content_min = 1.0f.xx;
    float2 content_max = 0.0f.xx;
    if (active) {
        [loop]
        for (uint x = first_x; x < end_x; ++x) {
            const uint2 cell = uint2(x, y);
            if (Analysis(
                    SBS_SCENE_ANALYSIS_GRID_VIEWPORT_VALID,
                    cell) <= 0.5f) {
                continue;
            }
            valid_count += 1.0f;
            const float content = Dense(
                SBS_SCENE_DENSE_OUT_PHOTOGRAPHIC_CONTENT_COLLAGE,
                cell);
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
            const float photo =
                CurrentLayoutHistory[SceneAnalysisIndex(
                    SBS_SCENE_LAYOUT_HISTORY_PHOTOGRAPHIC_DENSITY_EMA,
                    cell)];
            content_sum += content;
            unsafe_sum += unsafe;
            photo_sum += photo;
            if (content > 0.22f && unsafe < 0.72f) {
                const float2 uv = float2(
                    Analysis(SBS_SCENE_ANALYSIS_GRID_VIEWPORT_X, cell),
                    Analysis(SBS_SCENE_ANALYSIS_GRID_VIEWPORT_Y, cell));
                content_min = min(content_min, uv);
                content_max = max(content_max, uv);
                content_cells += 1.0f;
            }
            if (video > 0.18f && unsafe < 0.65f) {
                video_support += 1.0f;
                video_weight += video * (1.0f - unsafe);
            }
        }
    }

    RuleSummary[SbsRuleRowIndex(region, y, SBS_RULE_ROW_VALID)] =
        valid_count;
    RuleSummary[SbsRuleRowIndex(
        region,
        y,
        SBS_RULE_ROW_VIDEO_SUPPORT)] = video_support;
    RuleSummary[SbsRuleRowIndex(
        region,
        y,
        SBS_RULE_ROW_VIDEO_WEIGHT)] = video_weight;
    RuleSummary[SbsRuleRowIndex(
        region,
        y,
        SBS_RULE_ROW_CONTENT_SUM)] = content_sum;
    RuleSummary[SbsRuleRowIndex(
        region,
        y,
        SBS_RULE_ROW_UNSAFE_SUM)] = unsafe_sum;
    RuleSummary[SbsRuleRowIndex(
        region,
        y,
        SBS_RULE_ROW_PHOTO_SUM)] = photo_sum;
    RuleSummary[SbsRuleRowIndex(
        region,
        y,
        SBS_RULE_ROW_CONTENT_CELLS)] = content_cells;
    RuleSummary[SbsRuleRowIndex(
        region,
        y,
        SBS_RULE_ROW_CONTENT_MIN_X)] = content_min.x;
    RuleSummary[SbsRuleRowIndex(
        region,
        y,
        SBS_RULE_ROW_CONTENT_MIN_Y)] = content_min.y;
    RuleSummary[SbsRuleRowIndex(
        region,
        y,
        SBS_RULE_ROW_CONTENT_MAX_X)] = content_max.x;
    RuleSummary[SbsRuleRowIndex(
        region,
        y,
        SBS_RULE_ROW_CONTENT_MAX_Y)] = content_max.y;
}
