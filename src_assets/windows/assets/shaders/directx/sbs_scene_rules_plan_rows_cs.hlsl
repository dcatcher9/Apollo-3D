RWStructuredBuffer<float> RuleSummary : register(u0);

#include "include/sbs_scene_controller_constants.hlsl"
#include "include/sbs_scene_rules_summary.hlsl"

static const uint SBS_SCENE_MAX_VIDEO_RUNS = 4u;
static const uint SBS_SCENE_VIDEO_RUN_GAP_CELLS = 1u;

struct AxisRuns {
    uint4 starts;
    uint4 ends;
    uint count;
    uint overflow;
};

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

void AppendAxisRun(
    uint start,
    uint end,
    uint minimum_length,
    inout AxisRuns runs)
{
    if (end <= start || end - start < minimum_length) {
        return;
    }
    if (runs.count >= SBS_SCENE_MAX_VIDEO_RUNS) {
        runs.overflow = 1u;
        return;
    }
    if (runs.count == 0u) {
        runs.starts.x = start;
        runs.ends.x = end;
    } else if (runs.count == 1u) {
        runs.starts.y = start;
        runs.ends.y = end;
    } else if (runs.count == 2u) {
        runs.starts.z = start;
        runs.ends.z = end;
    } else {
        runs.starts.w = start;
        runs.ends.w = end;
    }
    runs.count += 1u;
}

uint AxisRunStart(AxisRuns runs, uint index) {
    if (index == 0u) {
        return runs.starts.x;
    }
    if (index == 1u) {
        return runs.starts.y;
    }
    return index == 2u ? runs.starts.z : runs.starts.w;
}

uint AxisRunEnd(AxisRuns runs, uint index) {
    if (index == 0u) {
        return runs.ends.x;
    }
    if (index == 1u) {
        return runs.ends.y;
    }
    return index == 2u ? runs.ends.z : runs.ends.w;
}

AxisRuns CollectVideoRowRuns(uint region) {
    AxisRuns runs = (AxisRuns)0;
    uint2 viewport_origin;
    uint2 viewport_size;
    SceneViewport(
        SBS_SCENE_ANALYSIS_CANVAS_SIZE,
        viewport_origin,
        viewport_size);
    const uint minimum_length = max(3u, viewport_size.y / 20u);
    uint run_start = 0u;
    uint last_supported = 0u;
    uint gap_cells = 0u;
    bool in_run = false;
    [loop]
    for (
        uint y = viewport_origin.y;
        y < viewport_origin.y + viewport_size.y;
        ++y
    ) {
        const float valid_count =
            RuleSummary[SbsRuleRowIndex(
                region,
                y,
                SBS_RULE_ROW_VALID)];
        const float support_count =
            RuleSummary[SbsRuleRowIndex(
                region,
                y,
                SBS_RULE_ROW_VIDEO_SUPPORT)];
        const float support_weight =
            RuleSummary[SbsRuleRowIndex(
                region,
                y,
                SBS_RULE_ROW_VIDEO_WEIGHT)];
        const bool supported =
            valid_count > 0.0f &&
            support_count >= 2.0f &&
            support_count >= 0.08f * valid_count &&
            support_weight >= 0.025f * valid_count;
        if (supported) {
            if (!in_run) {
                run_start = y;
                in_run = true;
            }
            last_supported = y;
            gap_cells = 0u;
        } else if (in_run) {
            gap_cells += 1u;
            if (gap_cells > SBS_SCENE_VIDEO_RUN_GAP_CELLS) {
                AppendAxisRun(
                    run_start,
                    last_supported + 1u,
                    minimum_length,
                    runs);
                in_run = false;
                gap_cells = 0u;
            }
        }
    }
    if (in_run) {
        AppendAxisRun(
            run_start,
            last_supported + 1u,
            minimum_length,
            runs);
    }
    return runs;
}

void StoreContentEvidence(uint region) {
    float4 bounds = float4(1.0f, 1.0f, 0.0f, 0.0f);
    float content_sum = 0.0f;
    float unsafe_sum = 0.0f;
    float photo_sum = 0.0f;
    float valid_count = 0.0f;
    float content_cells = 0.0f;
    [loop]
    for (uint y = 0u; y < SBS_SCENE_ANALYSIS_CANVAS_SIZE; ++y) {
        valid_count += RuleSummary[SbsRuleRowIndex(
            region,
            y,
            SBS_RULE_ROW_VALID)];
        content_sum += RuleSummary[SbsRuleRowIndex(
            region,
            y,
            SBS_RULE_ROW_CONTENT_SUM)];
        unsafe_sum += RuleSummary[SbsRuleRowIndex(
            region,
            y,
            SBS_RULE_ROW_UNSAFE_SUM)];
        photo_sum += RuleSummary[SbsRuleRowIndex(
            region,
            y,
            SBS_RULE_ROW_PHOTO_SUM)];
        const float row_content_cells =
            RuleSummary[SbsRuleRowIndex(
                region,
                y,
                SBS_RULE_ROW_CONTENT_CELLS)];
        content_cells += row_content_cells;
        if (row_content_cells > 0.0f) {
            bounds.xy = min(
                bounds.xy,
                float2(
                    RuleSummary[SbsRuleRowIndex(
                        region,
                        y,
                        SBS_RULE_ROW_CONTENT_MIN_X)],
                    RuleSummary[SbsRuleRowIndex(
                        region,
                        y,
                        SBS_RULE_ROW_CONTENT_MIN_Y)]));
            bounds.zw = max(
                bounds.zw,
                float2(
                    RuleSummary[SbsRuleRowIndex(
                        region,
                        y,
                        SBS_RULE_ROW_CONTENT_MAX_X)],
                    RuleSummary[SbsRuleRowIndex(
                        region,
                        y,
                        SBS_RULE_ROW_CONTENT_MAX_Y)]));
        }
    }
    if (ValidBounds(bounds)) {
        const float2 halo = FullViewportCellHalo();
        bounds = clamp(bounds + float4(-halo, halo), 0.0f, 1.0f);
    }

    float coverage = 0.0f;
    float unsafe_fraction = 0.0f;
    float score = 0.0f;
    if (valid_count > 0.0f) {
        const float mean_content = content_sum / valid_count;
        const float mean_photo = photo_sum / valid_count;
        unsafe_fraction = unsafe_sum / valid_count;
        coverage = content_cells / valid_count;
        const float area = BoundsArea(bounds);
        const float edge_penalty =
            area < 0.18f &&
            (bounds.x < 0.08f || bounds.z > 0.92f) ?
                0.30f :
                1.0f;
        score =
            area >= 0.08f ?
                area *
                (0.45f + 0.35f * mean_content +
                 0.10f * mean_photo + 0.10f * coverage) *
                edge_penalty *
                (1.0f - 0.75f * unsafe_fraction) :
                0.0f;
    }

    RuleSummary[SbsRuleRegionIndex(
        region,
        SBS_RULE_REGION_CONTENT_MIN_X)] = bounds.x;
    RuleSummary[SbsRuleRegionIndex(
        region,
        SBS_RULE_REGION_CONTENT_MIN_Y)] = bounds.y;
    RuleSummary[SbsRuleRegionIndex(
        region,
        SBS_RULE_REGION_CONTENT_MAX_X)] = bounds.z;
    RuleSummary[SbsRuleRegionIndex(
        region,
        SBS_RULE_REGION_CONTENT_MAX_Y)] = bounds.w;
    RuleSummary[SbsRuleRegionIndex(
        region,
        SBS_RULE_REGION_CONTENT_SCORE)] = score;
    RuleSummary[SbsRuleRegionIndex(
        region,
        SBS_RULE_REGION_CONTENT_COVERAGE)] = coverage;
    RuleSummary[SbsRuleRegionIndex(
        region,
        SBS_RULE_REGION_UNSAFE_FRACTION)] = unsafe_fraction;
}

[numthreads(1, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    [unroll]
    for (uint candidate = 0u; candidate < SBS_RULE_CANDIDATE_MAX; ++candidate) {
        SbsRuleStoreUint(
            RuleSummary,
            SbsRuleCandidateDescIndex(
                candidate,
                SBS_RULE_CANDIDATE_ACTIVE),
            0u);
    }

    uint candidate_count = 0u;
    bool ambiguous = false;
    const uint region_count = SbsRuleLoadUint(
        RuleSummary,
        SbsRulePlanIndex(SBS_RULE_PLAN_REGION_COUNT));
    [unroll]
    for (uint region = 0u; region < 2u; ++region) {
        if (region >= region_count) {
            continue;
        }
        StoreContentEvidence(region);
        const AxisRuns rows = CollectVideoRowRuns(region);
        const uint column_count = SbsRuleLoadUint(
            RuleSummary,
            SbsRuleRegionIndex(
                region,
                SBS_RULE_REGION_COLUMN_RUN_COUNT));
        const uint column_overflow = SbsRuleLoadUint(
            RuleSummary,
            SbsRuleRegionIndex(
                region,
                SBS_RULE_REGION_COLUMN_OVERFLOW));
        SbsRuleStoreUint(
            RuleSummary,
            SbsRuleRegionIndex(
                region,
                SBS_RULE_REGION_ROW_RUN_COUNT),
            rows.count);
        SbsRuleStoreUint(
            RuleSummary,
            SbsRuleRegionIndex(
                region,
                SBS_RULE_REGION_ROW_OVERFLOW),
            rows.overflow);
        [unroll]
        for (uint run = 0u; run < SBS_SCENE_MAX_VIDEO_RUNS; ++run) {
            SbsRuleStoreUint(
                RuleSummary,
                SbsRuleRegionIndex(
                    region,
                    SBS_RULE_REGION_ROW_START_0 + run),
                AxisRunStart(rows, run));
            SbsRuleStoreUint(
                RuleSummary,
                SbsRuleRegionIndex(
                    region,
                    SBS_RULE_REGION_ROW_END_0 + run),
                AxisRunEnd(rows, run));
        }
        SbsRuleStoreUint(
            RuleSummary,
            SbsRuleRegionIndex(
                region,
                SBS_RULE_REGION_CANDIDATE_FIRST),
            candidate_count);

        const uint region_candidate_first = candidate_count;
        [loop]
        for (uint row = 0u; row < rows.count; ++row) {
            [loop]
            for (uint column = 0u; column < column_count; ++column) {
                if (candidate_count >= SBS_RULE_CANDIDATE_MAX) {
                    ambiguous = true;
                    continue;
                }
                SbsRuleStoreUint(
                    RuleSummary,
                    SbsRuleCandidateDescIndex(
                        candidate_count,
                        SBS_RULE_CANDIDATE_ACTIVE),
                    1u);
                SbsRuleStoreUint(
                    RuleSummary,
                    SbsRuleCandidateDescIndex(
                        candidate_count,
                        SBS_RULE_CANDIDATE_FIRST_X),
                    SbsRuleLoadUint(
                        RuleSummary,
                        SbsRuleRegionIndex(
                            region,
                            SBS_RULE_REGION_COLUMN_START_0 + column)));
                SbsRuleStoreUint(
                    RuleSummary,
                    SbsRuleCandidateDescIndex(
                        candidate_count,
                        SBS_RULE_CANDIDATE_END_X),
                    SbsRuleLoadUint(
                        RuleSummary,
                        SbsRuleRegionIndex(
                            region,
                            SBS_RULE_REGION_COLUMN_END_0 + column)));
                SbsRuleStoreUint(
                    RuleSummary,
                    SbsRuleCandidateDescIndex(
                        candidate_count,
                        SBS_RULE_CANDIDATE_FIRST_Y),
                    AxisRunStart(rows, row));
                SbsRuleStoreUint(
                    RuleSummary,
                    SbsRuleCandidateDescIndex(
                        candidate_count,
                        SBS_RULE_CANDIDATE_END_Y),
                    AxisRunEnd(rows, row));
                SbsRuleStoreUint(
                    RuleSummary,
                    SbsRuleCandidateDescIndex(
                        candidate_count,
                        SBS_RULE_CANDIDATE_REGION),
                    region);
                candidate_count += 1u;
            }
        }
        SbsRuleStoreUint(
            RuleSummary,
            SbsRuleRegionIndex(
                region,
                SBS_RULE_REGION_CANDIDATE_COUNT),
            candidate_count - region_candidate_first);
        ambiguous =
            ambiguous || column_overflow != 0u || rows.overflow != 0u;
    }
    RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_VIDEO_AMBIGUOUS)] =
        ambiguous ? 1.0f : 0.0f;
}
