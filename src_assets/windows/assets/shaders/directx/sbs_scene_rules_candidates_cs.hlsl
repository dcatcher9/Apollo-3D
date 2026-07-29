StructuredBuffer<float> AnalysisGrid : register(t0);
StructuredBuffer<float> DenseOutput : register(t1);
StructuredBuffer<float> CurrentLayoutHistory : register(t2);
RWStructuredBuffer<float> RuleSummary : register(u0);

#include "include/sbs_scene_controller_constants.hlsl"
#include "include/sbs_scene_rules_summary.hlsl"

groupshared float RowVideoSum[128];
groupshared float RowUnsafeSum[128];
groupshared float RowPhotoSum[128];
groupshared float RowValidCount[128];
groupshared float RowVideoCells[128];
groupshared float RowMinX[128];
groupshared float RowMinY[128];
groupshared float RowMaxX[128];
groupshared float RowMaxY[128];

float Dense(uint channel, uint2 cell) {
    return DenseOutput[SceneAnalysisIndex(channel, cell)];
}

float Analysis(uint channel, uint2 cell) {
    return AnalysisGrid[SceneAnalysisIndex(channel, cell)];
}

bool ValidBounds(float4 bounds) {
    return all(bounds.zw > bounds.xy);
}

float BoundsArea(float4 bounds) {
    const float2 extent = max(bounds.zw - bounds.xy, 0.0f.xx);
    return extent.x * extent.y;
}

float2 FullViewportCellHalo() {
    uint2 viewport_origin;
    uint2 viewport_size;
    SceneViewport(
        SBS_SCENE_ANALYSIS_CANVAS_SIZE,
        viewport_origin,
        viewport_size);
    return 0.5f / max(float2(viewport_size), 1.0f.xx);
}

[numthreads(128, 1, 1)]
void main(
    uint3 group_id : SV_GroupID,
    uint3 thread_id : SV_GroupThreadID)
{
    const uint candidate = group_id.x;
    const uint lane = thread_id.x;
    const bool active =
        candidate < SBS_RULE_CANDIDATE_MAX &&
        SbsRuleLoadUint(
            RuleSummary,
            SbsRuleCandidateDescIndex(
                candidate,
                SBS_RULE_CANDIDATE_ACTIVE)) != 0u;

    const uint first_x = SbsRuleLoadUint(
        RuleSummary,
        SbsRuleCandidateDescIndex(
            candidate,
            SBS_RULE_CANDIDATE_FIRST_X));
    const uint end_x = SbsRuleLoadUint(
        RuleSummary,
        SbsRuleCandidateDescIndex(
            candidate,
            SBS_RULE_CANDIDATE_END_X));
    const uint first_y = SbsRuleLoadUint(
        RuleSummary,
        SbsRuleCandidateDescIndex(
            candidate,
            SBS_RULE_CANDIDATE_FIRST_Y));
    const uint end_y = SbsRuleLoadUint(
        RuleSummary,
        SbsRuleCandidateDescIndex(
            candidate,
            SBS_RULE_CANDIDATE_END_Y));

    float video_sum = 0.0f;
    float unsafe_sum = 0.0f;
    float photo_sum = 0.0f;
    float valid_count = 0.0f;
    float video_cells = 0.0f;
    float2 bounds_min = 1.0f.xx;
    float2 bounds_max = 0.0f.xx;
    if (active && lane >= first_y && lane < end_y) {
        [loop]
        for (uint x = first_x; x < end_x; ++x) {
            const uint2 cell = uint2(x, lane);
            if (Analysis(
                    SBS_SCENE_ANALYSIS_GRID_VIEWPORT_VALID,
                    cell) <= 0.5f) {
                continue;
            }
            valid_count += 1.0f;
            const float video = Dense(
                SBS_SCENE_DENSE_OUT_PRIMARY_PLAYING_VIDEO,
                cell);
            const float unsafe = max(
                Dense(
                    SBS_SCENE_DENSE_OUT_ADVERTISEMENT_COMPETITOR_UNSAFE,
                    cell),
                Dense(
                    SBS_SCENE_DENSE_OUT_BROWSER_SYSTEM_CHROME,
                    cell));
            const float photo =
                CurrentLayoutHistory[SceneAnalysisIndex(
                    SBS_SCENE_LAYOUT_HISTORY_PHOTOGRAPHIC_DENSITY_EMA,
                    cell)];
            video_sum += video;
            unsafe_sum += unsafe;
            photo_sum += photo;
            if (video > 0.18f && unsafe < 0.65f) {
                const float2 uv = float2(
                    Analysis(SBS_SCENE_ANALYSIS_GRID_VIEWPORT_X, cell),
                    Analysis(SBS_SCENE_ANALYSIS_GRID_VIEWPORT_Y, cell));
                bounds_min = min(bounds_min, uv);
                bounds_max = max(bounds_max, uv);
                video_cells += 1.0f;
            }
        }
    }

    RowVideoSum[lane] = video_sum;
    RowUnsafeSum[lane] = unsafe_sum;
    RowPhotoSum[lane] = photo_sum;
    RowValidCount[lane] = valid_count;
    RowVideoCells[lane] = video_cells;
    RowMinX[lane] = bounds_min.x;
    RowMinY[lane] = bounds_min.y;
    RowMaxX[lane] = bounds_max.x;
    RowMaxY[lane] = bounds_max.y;
    GroupMemoryBarrierWithGroupSync();

    if (lane != 0u) {
        return;
    }
    if (!active) {
        [unroll]
        for (uint field = 0u;
             field < SBS_RULE_CANDIDATE_RESULT_STRIDE;
             ++field) {
            RuleSummary[
                SbsRuleCandidateResultIndex(candidate, field)] = 0.0f;
        }
        return;
    }

    float total_video = 0.0f;
    float total_unsafe = 0.0f;
    float total_photo = 0.0f;
    float total_valid = 0.0f;
    float total_video_cells = 0.0f;
    float4 bounds = float4(1.0f, 1.0f, 0.0f, 0.0f);
    [loop]
    for (uint row = 0u; row < SBS_SCENE_ANALYSIS_CANVAS_SIZE; ++row) {
        total_video += RowVideoSum[row];
        total_unsafe += RowUnsafeSum[row];
        total_photo += RowPhotoSum[row];
        total_valid += RowValidCount[row];
        total_video_cells += RowVideoCells[row];
        if (RowVideoCells[row] > 0.0f) {
            bounds.xy = min(bounds.xy, float2(RowMinX[row], RowMinY[row]));
            bounds.zw = max(bounds.zw, float2(RowMaxX[row], RowMaxY[row]));
        }
    }

    float score = 0.0f;
    float coverage = 0.0f;
    if (total_valid > 0.0f && ValidBounds(bounds)) {
        const float2 halo = FullViewportCellHalo();
        bounds = clamp(bounds + float4(-halo, halo), 0.0f, 1.0f);
        coverage = total_video_cells / total_valid;
        const float area = BoundsArea(bounds);
        const float unsafe_fraction = total_unsafe / total_valid;
        const float edge_penalty =
            area < 0.25f &&
            (bounds.x < 0.08f || bounds.z > 0.92f) ?
                0.35f :
                1.0f;
        score =
            area >= 0.05f && coverage >= 0.08f ?
                area *
                (0.35f + 0.40f * (total_video / total_valid) +
                 0.15f * (total_photo / total_valid) +
                 0.10f * coverage) *
                edge_penalty *
                (1.0f - 0.65f * unsafe_fraction) :
                0.0f;
    }

    RuleSummary[SbsRuleCandidateResultIndex(
        candidate,
        SBS_RULE_CANDIDATE_SCORE)] = score;
    RuleSummary[SbsRuleCandidateResultIndex(
        candidate,
        SBS_RULE_CANDIDATE_MIN_X)] = bounds.x;
    RuleSummary[SbsRuleCandidateResultIndex(
        candidate,
        SBS_RULE_CANDIDATE_MIN_Y)] = bounds.y;
    RuleSummary[SbsRuleCandidateResultIndex(
        candidate,
        SBS_RULE_CANDIDATE_MAX_X)] = bounds.z;
    RuleSummary[SbsRuleCandidateResultIndex(
        candidate,
        SBS_RULE_CANDIDATE_MAX_Y)] = bounds.w;
    RuleSummary[SbsRuleCandidateResultIndex(
        candidate,
        SBS_RULE_CANDIDATE_COVERAGE)] = coverage;
    RuleSummary[SbsRuleCandidateResultIndex(candidate, 6u)] = 0.0f;
    RuleSummary[SbsRuleCandidateResultIndex(candidate, 7u)] = 0.0f;
}
