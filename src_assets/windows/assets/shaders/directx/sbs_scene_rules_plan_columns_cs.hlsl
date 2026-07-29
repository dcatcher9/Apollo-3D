StructuredBuffer<float4> AdaptiveState : register(t4);
StructuredBuffer<float> Meta : register(t5);
RWStructuredBuffer<float> RuleSummary : register(u0);

#include "include/sbs_scene_controller_constants.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"
#include "include/sbs_scene_rules_summary.hlsl"

static const uint SBS_SCENE_MAX_VIDEO_RUNS = 4u;
static const uint SBS_SCENE_VIDEO_RUN_GAP_CELLS = 1u;

struct AxisRuns {
    uint4 starts;
    uint4 ends;
    uint count;
    uint overflow;
};

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

AxisRuns CollectVideoColumnRuns(uint first_x, uint end_x) {
    AxisRuns runs = (AxisRuns)0;
    uint2 viewport_origin;
    uint2 viewport_size;
    SceneViewport(
        SBS_SCENE_ANALYSIS_CANVAS_SIZE,
        viewport_origin,
        viewport_size);
    const uint minimum_length = max(3u, viewport_size.x / 20u);
    uint run_start = 0u;
    uint last_supported = 0u;
    uint gap_cells = 0u;
    bool in_run = false;
    [loop]
    for (uint x = first_x; x < end_x; ++x) {
        const float valid_count = RuleSummary[
            SbsRuleColumnIndex(x, SBS_RULE_COLUMN_VIEWPORT_VALID)];
        const float support_count = RuleSummary[
            SbsRuleColumnIndex(x, SBS_RULE_COLUMN_VIDEO_SUPPORT)];
        const float support_weight = RuleSummary[
            SbsRuleColumnIndex(x, SBS_RULE_COLUMN_VIDEO_WEIGHT)];
        const bool supported =
            valid_count > 0.0f &&
            support_count >= 2.0f &&
            support_count >= 0.08f * valid_count &&
            support_weight >= 0.025f * valid_count;
        if (supported) {
            if (!in_run) {
                run_start = x;
                in_run = true;
            }
            last_supported = x;
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

void StoreRegion(uint region, uint first_x, uint end_x) {
    const AxisRuns columns = CollectVideoColumnRuns(first_x, end_x);
    SbsRuleStoreUint(
        RuleSummary,
        SbsRuleRegionIndex(region, SBS_RULE_REGION_ACTIVE),
        1u);
    SbsRuleStoreUint(
        RuleSummary,
        SbsRuleRegionIndex(region, SBS_RULE_REGION_FIRST_X),
        first_x);
    SbsRuleStoreUint(
        RuleSummary,
        SbsRuleRegionIndex(region, SBS_RULE_REGION_END_X),
        end_x);
    [unroll]
    for (uint field = SBS_RULE_REGION_CONTENT_MIN_X;
         field <= SBS_RULE_REGION_UNSAFE_FRACTION;
         ++field) {
        RuleSummary[SbsRuleRegionIndex(region, field)] = 0.0f;
    }
    SbsRuleStoreUint(
        RuleSummary,
        SbsRuleRegionIndex(region, SBS_RULE_REGION_COLUMN_RUN_COUNT),
        columns.count);
    SbsRuleStoreUint(
        RuleSummary,
        SbsRuleRegionIndex(region, SBS_RULE_REGION_COLUMN_OVERFLOW),
        columns.overflow);
    [unroll]
    for (uint run = 0u; run < SBS_SCENE_MAX_VIDEO_RUNS; ++run) {
        SbsRuleStoreUint(
            RuleSummary,
            SbsRuleRegionIndex(
                region,
                SBS_RULE_REGION_COLUMN_START_0 + run),
            AxisRunStart(columns, run));
        SbsRuleStoreUint(
            RuleSummary,
            SbsRuleRegionIndex(
                region,
                SBS_RULE_REGION_COLUMN_END_0 + run),
            AxisRunEnd(columns, run));
    }
    SbsRuleStoreUint(
        RuleSummary,
        SbsRuleRegionIndex(region, SBS_RULE_REGION_ROW_RUN_COUNT),
        0u);
    SbsRuleStoreUint(
        RuleSummary,
        SbsRuleRegionIndex(region, SBS_RULE_REGION_ROW_OVERFLOW),
        0u);
    SbsRuleStoreUint(
        RuleSummary,
        SbsRuleRegionIndex(region, SBS_RULE_REGION_CANDIDATE_FIRST),
        0u);
    SbsRuleStoreUint(
        RuleSummary,
        SbsRuleRegionIndex(region, SBS_RULE_REGION_CANDIDATE_COUNT),
        0u);
}

[numthreads(1, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    const uint canvas = SBS_SCENE_ANALYSIS_CANVAS_SIZE;
    float valid_cells = 0.0f;
    float required_invalid = 0.0f;
    float depth_coverage = 0.0f;
    float depth_confidence = 0.0f;
    float scroll_weight = 0.0f;
    float scroll_signed = 0.0f;
    float scroll_support = 0.0f;
    uint viewport_first_x = canvas;
    uint viewport_end_x = 0u;
    [loop]
    for (uint x = 0u; x < canvas; ++x) {
        const float column_valid = RuleSummary[
            SbsRuleColumnIndex(x, SBS_RULE_COLUMN_VIEWPORT_VALID)];
        if (column_valid > 0.0f) {
            viewport_first_x = min(viewport_first_x, x);
            viewport_end_x = max(viewport_end_x, x + 1u);
        }
    }
    [loop]
    for (uint y = 0u; y < canvas; ++y) {
        valid_cells += RuleSummary[SbsRuleGlobalRowIndex(
            y,
            SBS_RULE_GLOBAL_ROW_MAIN_VALID)];
        required_invalid += RuleSummary[SbsRuleGlobalRowIndex(
            y,
            SBS_RULE_GLOBAL_ROW_REQUIRED_INVALID)];
        depth_coverage += RuleSummary[SbsRuleGlobalRowIndex(
            y,
            SBS_RULE_GLOBAL_ROW_DEPTH_COVERAGE)];
        depth_confidence += RuleSummary[SbsRuleGlobalRowIndex(
            y,
            SBS_RULE_GLOBAL_ROW_DEPTH_CONFIDENCE)];
        scroll_weight += RuleSummary[SbsRuleGlobalRowIndex(
            y,
            SBS_RULE_GLOBAL_ROW_SCROLL_WEIGHT)];
        scroll_signed += RuleSummary[SbsRuleGlobalRowIndex(
            y,
            SBS_RULE_GLOBAL_ROW_SCROLL_SIGNED)];
        scroll_support += RuleSummary[SbsRuleGlobalRowIndex(
            y,
            SBS_RULE_GLOBAL_ROW_SCROLL_SUPPORT)];
    }

    bool meta_finite = true;
    bool meta_reserved_zero = true;
    [unroll]
    for (uint word = 0u; word < SBS_SCENE_META_WORD_COUNT; ++word) {
        const float value = Meta[word];
        meta_finite = meta_finite && isfinite(value);
        if (word >= 28u) {
            meta_reserved_zero =
                meta_reserved_zero && abs(value) <= 1e-8f;
        }
    }
    const bool adaptive_finite =
        all(isfinite(AdaptiveState[SBS_STATE_VECTOR_RANGE_COLLAPSED])) &&
        all(isfinite(AdaptiveState[SBS_STATE_VECTOR_ANALYSIS_FLAGS])) &&
        all(isfinite(AdaptiveState[
            SBS_STATE_VECTOR_ADAPTIVE_POP_RATIO])) &&
        all(isfinite(AdaptiveState[
            SBS_STATE_VECTOR_VALID_DEPTH_FRACTION])) &&
        all(isfinite(AdaptiveState[SBS_STATE_VECTOR_CUT_FLAGS]));
    const bool input_valid =
        SceneAbiMatches() &&
        required_invalid <= 0.0f &&
        valid_cells > 0.10f * canvas * canvas &&
        meta_finite &&
        adaptive_finite &&
        meta_reserved_zero;

    RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_INPUT_VALID)] =
        input_valid ? 1.0f : 0.0f;
    SbsRuleStoreUint(
        RuleSummary,
        SbsRulePlanIndex(SBS_RULE_PLAN_VIEWPORT_FIRST_X),
        viewport_first_x);
    SbsRuleStoreUint(
        RuleSummary,
        SbsRulePlanIndex(SBS_RULE_PLAN_VIEWPORT_END_X),
        viewport_end_x);
    RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_VALID_CELLS)] =
        valid_cells;
    RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_DEPTH_COVERAGE)] =
        depth_coverage;
    RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_DEPTH_CONFIDENCE)] =
        depth_confidence;
    RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_SCROLL_WEIGHT)] =
        scroll_weight;
    RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_SCROLL_SIGNED)] =
        scroll_signed;
    RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_SCROLL_SUPPORT)] =
        scroll_support;
    RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_VIDEO_AMBIGUOUS)] = 0.0f;
    [unroll]
    for (uint region = 0u; region < 2u; ++region) {
        SbsRuleStoreUint(
            RuleSummary,
            SbsRuleRegionIndex(region, SBS_RULE_REGION_ACTIVE),
            0u);
    }
    if (!input_valid) {
        SbsRuleStoreUint(
            RuleSummary,
            SbsRulePlanIndex(SBS_RULE_PLAN_REGION_COUNT),
            0u);
        return;
    }

    uint best_gutter_start = 0u;
    uint best_gutter_end = 0u;
    uint run_start = 0u;
    bool in_run = false;
    [loop]
    for (uint x = viewport_first_x; x < viewport_end_x; ++x) {
        const float valid_in_column = RuleSummary[
            SbsRuleColumnIndex(x, SBS_RULE_COLUMN_VIEWPORT_VALID)];
        const float gutter_in_column = RuleSummary[
            SbsRuleColumnIndex(x, SBS_RULE_COLUMN_GUTTER)];
        const bool gutter_column =
            valid_in_column > 0.0f &&
            gutter_in_column >= 0.76f * valid_in_column;
        if (gutter_column && !in_run) {
            run_start = x;
            in_run = true;
        }
        if ((!gutter_column || x + 1u == viewport_end_x) && in_run) {
            const uint run_end =
                gutter_column && x + 1u == viewport_end_x ?
                    x + 1u :
                    x;
            if (
                run_end - run_start >= 3u &&
                run_end - run_start >
                    best_gutter_end - best_gutter_start
            ) {
                best_gutter_start = run_start;
                best_gutter_end = run_end;
            }
            in_run = false;
        }
    }

    const bool split =
        best_gutter_end > best_gutter_start &&
        best_gutter_start > viewport_first_x &&
        best_gutter_end < viewport_end_x;
    SbsRuleStoreUint(
        RuleSummary,
        SbsRulePlanIndex(SBS_RULE_PLAN_GUTTER_START),
        best_gutter_start);
    SbsRuleStoreUint(
        RuleSummary,
        SbsRulePlanIndex(SBS_RULE_PLAN_GUTTER_END),
        best_gutter_end);
    SbsRuleStoreUint(
        RuleSummary,
        SbsRulePlanIndex(SBS_RULE_PLAN_SPLIT),
        split ? 1u : 0u);
    SbsRuleStoreUint(
        RuleSummary,
        SbsRulePlanIndex(SBS_RULE_PLAN_REGION_COUNT),
        split ? 2u : 1u);

    StoreRegion(
        0u,
        viewport_first_x,
        split ? best_gutter_start : viewport_end_x);
    if (split) {
        StoreRegion(1u, best_gutter_end, viewport_end_x);
    }
}
