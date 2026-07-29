StructuredBuffer<float> AnalysisGrid : register(t0);
StructuredBuffer<float> DenseOutput : register(t1);
StructuredBuffer<float> CurrentLayoutHistory : register(t2);
StructuredBuffer<float> CurrentDepthHistory : register(t3);
StructuredBuffer<float4> AdaptiveState : register(t4);
StructuredBuffer<float> Meta : register(t5);
StructuredBuffer<uint4> PreviousRuleState : register(t6);
RWStructuredBuffer<float> EvidenceGlobal : register(u0);

#include "include/sbs_scene_controller_constants.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"

float Dense(uint channel, uint2 cell) {
    return DenseOutput[SceneAnalysisIndex(channel, cell)];
}

float Analysis(uint channel, uint2 cell) {
    return AnalysisGrid[SceneAnalysisIndex(channel, cell)];
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

struct RegionEvidence {
    float4 content_bounds;
    float content_score;
    float content_coverage;
    float unsafe_fraction;
};

struct VideoEvidence {
    float4 bounds;
    float score;
    float coverage;
};

// Test-only serial equivalence oracle. Four runs per axis cover the common 2x2 and 3x3
// equal-tile layouts without allowing an
// unbounded component search in this deliberately serial reduction pass. Overflow is an
// explicit ambiguity signal rather than permission to merge unrelated videos.
static const uint SBS_SCENE_MAX_VIDEO_RUNS = 4u;
static const uint SBS_SCENE_VIDEO_RUN_GAP_CELLS = 1u;

struct AxisRuns {
    uint4 starts;
    uint4 ends;
    uint count;
    uint overflow;
};

float4 EmptyBounds() {
    return float4(1.0f, 1.0f, 0.0f, 0.0f);
}

bool ValidBounds(float4 bounds) {
    return all(bounds.zw > bounds.xy);
}

float BoundsArea(float4 bounds) {
    float2 extent = max(bounds.zw - bounds.xy, 0.0f.xx);
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

RegionEvidence EvaluateRegion(uint first_x, uint end_x) {
    RegionEvidence result;
    result.content_bounds = EmptyBounds();
    result.content_score = 0.0f;
    result.content_coverage = 0.0f;
    result.unsafe_fraction = 0.0f;

    float content_sum = 0.0f;
    float unsafe_sum = 0.0f;
    float photo_sum = 0.0f;
    float valid_count = 0.0f;
    float content_cells = 0.0f;
    const uint canvas = SBS_SCENE_ANALYSIS_CANVAS_SIZE;
    [loop]
    for (uint y = 0u; y < canvas; ++y) {
        [loop]
        for (uint x = first_x; x < min(end_x, canvas); ++x) {
            uint2 cell = uint2(x, y);
            if (Analysis(
                    SBS_SCENE_ANALYSIS_GRID_VIEWPORT_VALID,
                    cell) <= 0.5f) {
                continue;
            }
            valid_count += 1.0f;
            float content = Dense(
                SBS_SCENE_DENSE_OUT_PHOTOGRAPHIC_CONTENT_COLLAGE,
                cell);
            float unsafe = max(
                Dense(
                    SBS_SCENE_DENSE_OUT_ADVERTISEMENT_COMPETITOR_UNSAFE,
                    cell),
                Dense(
                    SBS_SCENE_DENSE_OUT_BROWSER_SYSTEM_CHROME,
                    cell));
            float photo = CurrentLayoutHistory[SceneAnalysisIndex(
                SBS_SCENE_LAYOUT_HISTORY_PHOTOGRAPHIC_DENSITY_EMA,
                cell)];
            content_sum += content;
            unsafe_sum += unsafe;
            photo_sum += photo;
            float2 uv = float2(
                Analysis(SBS_SCENE_ANALYSIS_GRID_VIEWPORT_X, cell),
                Analysis(SBS_SCENE_ANALYSIS_GRID_VIEWPORT_Y, cell));
            if (content > 0.22f && unsafe < 0.72f) {
                result.content_bounds.xy = min(result.content_bounds.xy, uv);
                result.content_bounds.zw = max(result.content_bounds.zw, uv);
                content_cells += 1.0f;
            }
        }
    }

    if (valid_count <= 0.0f) {
        return result;
    }
    // Analysis UV is normalized over the complete valid viewport, not over this split region.
    // A split-region denominator over-expanded the old bounds at a stable gutter.
    float2 cell_halo = FullViewportCellHalo();
    if (ValidBounds(result.content_bounds)) {
        result.content_bounds =
            clamp(
                result.content_bounds + float4(-cell_halo, cell_halo),
                0.0f,
                1.0f);
    }

    float mean_content = content_sum / valid_count;
    float mean_photo = photo_sum / valid_count;
    result.unsafe_fraction = unsafe_sum / valid_count;
    result.content_coverage = content_cells / valid_count;
    float content_area = BoundsArea(result.content_bounds);
    float content_edge_penalty =
        content_area < 0.18f &&
        (result.content_bounds.x < 0.08f ||
         result.content_bounds.z > 0.92f) ? 0.30f : 1.0f;
    result.content_score =
        content_area >= 0.08f ?
            content_area *
            (0.45f + 0.35f * mean_content + 0.10f * mean_photo +
             0.10f * result.content_coverage) *
            content_edge_penalty *
            (1.0f - 0.75f * result.unsafe_fraction) :
            0.0f;
    return result;
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

bool IsVideoSupportCell(uint2 cell, out float video, out float unsafe) {
    if (Analysis(
            SBS_SCENE_ANALYSIS_GRID_VIEWPORT_VALID,
            cell) <= 0.5f) {
        video = 0.0f;
        unsafe = 1.0f;
        return false;
    }
    video = Dense(
        SBS_SCENE_DENSE_OUT_PRIMARY_PLAYING_VIDEO,
        cell);
    unsafe = max(
        Dense(
            SBS_SCENE_DENSE_OUT_ADVERTISEMENT_COMPETITOR_UNSAFE,
            cell),
        Dense(
            SBS_SCENE_DENSE_OUT_BROWSER_SYSTEM_CHROME,
            cell));
    return video > 0.18f && unsafe < 0.65f;
}

AxisRuns CollectVideoColumnRuns(uint first_x, uint end_x) {
    AxisRuns runs = (AxisRuns)0;
    const uint canvas = SBS_SCENE_ANALYSIS_CANVAS_SIZE;
    uint2 viewport_origin;
    uint2 viewport_size;
    SceneViewport(canvas, viewport_origin, viewport_size);
    uint minimum_length = max(3u, viewport_size.x / 20u);
    uint run_start = 0u;
    uint last_supported = 0u;
    uint gap_cells = 0u;
    bool in_run = false;
    [loop]
    for (uint x = first_x; x < min(end_x, canvas); ++x) {
        float valid_count = 0.0f;
        float support_count = 0.0f;
        float support_weight = 0.0f;
        [loop]
        for (uint y = viewport_origin.y;
             y < viewport_origin.y + viewport_size.y;
             ++y) {
            uint2 cell = uint2(x, y);
            if (Analysis(
                    SBS_SCENE_ANALYSIS_GRID_VIEWPORT_VALID,
                    cell) <= 0.5f) {
                continue;
            }
            valid_count += 1.0f;
            float video;
            float unsafe;
            if (IsVideoSupportCell(cell, video, unsafe)) {
                support_count += 1.0f;
                support_weight += video * (1.0f - unsafe);
            }
        }
        bool supported =
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

AxisRuns CollectVideoRowRuns(uint first_x, uint end_x) {
    AxisRuns runs = (AxisRuns)0;
    const uint canvas = SBS_SCENE_ANALYSIS_CANVAS_SIZE;
    uint2 viewport_origin;
    uint2 viewport_size;
    SceneViewport(canvas, viewport_origin, viewport_size);
    uint minimum_length = max(3u, viewport_size.y / 20u);
    uint run_start = 0u;
    uint last_supported = 0u;
    uint gap_cells = 0u;
    bool in_run = false;
    [loop]
    for (uint y = viewport_origin.y;
         y < viewport_origin.y + viewport_size.y;
         ++y) {
        float valid_count = 0.0f;
        float support_count = 0.0f;
        float support_weight = 0.0f;
        [loop]
        for (uint x = first_x; x < min(end_x, canvas); ++x) {
            uint2 cell = uint2(x, y);
            if (Analysis(
                    SBS_SCENE_ANALYSIS_GRID_VIEWPORT_VALID,
                    cell) <= 0.5f) {
                continue;
            }
            valid_count += 1.0f;
            float video;
            float unsafe;
            if (IsVideoSupportCell(cell, video, unsafe)) {
                support_count += 1.0f;
                support_weight += video * (1.0f - unsafe);
            }
        }
        bool supported =
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

VideoEvidence EvaluateVideoCandidate(
    uint first_x,
    uint end_x,
    uint first_y,
    uint end_y)
{
    VideoEvidence result;
    result.bounds = EmptyBounds();
    result.score = 0.0f;
    result.coverage = 0.0f;
    float video_sum = 0.0f;
    float unsafe_sum = 0.0f;
    float photo_sum = 0.0f;
    float valid_count = 0.0f;
    float video_cells = 0.0f;
    [loop]
    for (uint y = first_y; y < end_y; ++y) {
        [loop]
        for (uint x = first_x; x < end_x; ++x) {
            uint2 cell = uint2(x, y);
            if (Analysis(
                    SBS_SCENE_ANALYSIS_GRID_VIEWPORT_VALID,
                    cell) <= 0.5f) {
                continue;
            }
            valid_count += 1.0f;
            float video = Dense(
                SBS_SCENE_DENSE_OUT_PRIMARY_PLAYING_VIDEO,
                cell);
            float unsafe = max(
                Dense(
                    SBS_SCENE_DENSE_OUT_ADVERTISEMENT_COMPETITOR_UNSAFE,
                    cell),
                Dense(
                    SBS_SCENE_DENSE_OUT_BROWSER_SYSTEM_CHROME,
                    cell));
            float photo = CurrentLayoutHistory[SceneAnalysisIndex(
                SBS_SCENE_LAYOUT_HISTORY_PHOTOGRAPHIC_DENSITY_EMA,
                cell)];
            video_sum += video;
            unsafe_sum += unsafe;
            photo_sum += photo;
            if (video > 0.18f && unsafe < 0.65f) {
                float2 uv = float2(
                    Analysis(SBS_SCENE_ANALYSIS_GRID_VIEWPORT_X, cell),
                    Analysis(SBS_SCENE_ANALYSIS_GRID_VIEWPORT_Y, cell));
                result.bounds.xy = min(result.bounds.xy, uv);
                result.bounds.zw = max(result.bounds.zw, uv);
                video_cells += 1.0f;
            }
        }
    }
    if (valid_count <= 0.0f || !ValidBounds(result.bounds)) {
        return result;
    }
    float2 cell_halo = FullViewportCellHalo();
    result.bounds = clamp(
        result.bounds + float4(-cell_halo, cell_halo),
        0.0f,
        1.0f);
    result.coverage = video_cells / valid_count;
    float area = BoundsArea(result.bounds);
    float unsafe_fraction = unsafe_sum / valid_count;
    float edge_penalty =
        area < 0.25f &&
        (result.bounds.x < 0.08f || result.bounds.z > 0.92f) ?
            0.35f :
            1.0f;
    result.score =
        area >= 0.05f && result.coverage >= 0.08f ?
            area *
            (0.35f + 0.40f * (video_sum / valid_count) +
             0.15f * (photo_sum / valid_count) +
             0.10f * result.coverage) *
            edge_penalty *
            (1.0f - 0.65f * unsafe_fraction) :
            0.0f;
    return result;
}

void ConsiderCandidate(
    float score,
    float4 bounds,
    uint layout,
    float coverage,
    inout float best_score,
    inout float second_score,
    inout float4 best_bounds,
    inout uint best_layout,
    inout float best_coverage)
{
    if (score <= 0.0f || !ValidBounds(bounds)) {
        return;
    }
    if (score > best_score) {
        second_score = best_score;
        best_score = score;
        best_bounds = bounds;
        best_layout = layout;
        best_coverage = coverage;
    } else if (score > second_score) {
        second_score = score;
    }
}

void ConsiderVideoRunCandidates(
    uint first_x,
    uint end_x,
    inout bool segmentation_ambiguous,
    inout float best_video_score,
    inout float second_video_score,
    inout float best_score,
    inout float second_score,
    inout float4 best_bounds,
    inout uint best_layout,
    inout float best_coverage)
{
    AxisRuns columns = CollectVideoColumnRuns(first_x, end_x);
    AxisRuns rows = CollectVideoRowRuns(first_x, end_x);
    [loop]
    for (uint row = 0u; row < rows.count; ++row) {
        [loop]
        for (uint column = 0u; column < columns.count; ++column) {
            VideoEvidence candidate = EvaluateVideoCandidate(
                AxisRunStart(columns, column),
                AxisRunEnd(columns, column),
                AxisRunStart(rows, row),
                AxisRunEnd(rows, row));
            if (candidate.score > best_video_score) {
                second_video_score = best_video_score;
                best_video_score = candidate.score;
            } else if (candidate.score > second_video_score) {
                second_video_score = candidate.score;
            }
            ConsiderCandidate(
                candidate.score,
                candidate.bounds,
                SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO,
                candidate.coverage,
                best_score,
                second_score,
                best_bounds,
                best_layout,
                best_coverage);
        }
    }
    if (columns.overflow != 0u || rows.overflow != 0u) {
        // More components than the bounded reducer can represent is not a unique primary video.
        segmentation_ambiguous = true;
    }
}

[numthreads(1, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    [unroll]
    for (uint word = 0u;
         word < SBS_SCENE_GLOBAL_OUT_WORD_COUNT;
         ++word) {
        EvidenceGlobal[word] = 0.0f;
    }

    const uint canvas = SBS_SCENE_ANALYSIS_CANVAS_SIZE;
    uint viewport_first_x = canvas;
    uint viewport_end_x = 0u;
    float valid_cells = 0.0f;
    float depth_coverage_cells = 0.0f;
    float depth_confidence_sum = 0.0f;
    float scroll_weight = 0.0f;
    float scroll_signed = 0.0f;
    float scroll_support_cells = 0.0f;
    bool required_inputs_finite = true;
    [loop]
    for (uint y = 0u; y < canvas; ++y) {
        [loop]
        for (uint x = 0u; x < canvas; ++x) {
            uint2 cell = uint2(x, y);
            float viewport_flag = Analysis(
                SBS_SCENE_ANALYSIS_GRID_VIEWPORT_VALID,
                cell);
            if (!isfinite(viewport_flag) || viewport_flag < 0.0f) {
                required_inputs_finite = false;
                continue;
            }
            if (viewport_flag <= 0.5f) {
                continue;
            }
            float output_sentinel = Dense(
                SBS_SCENE_DENSE_OUT_UNKNOWN_BACKGROUND,
                cell);
            if (!isfinite(output_sentinel) || output_sentinel < 0.0f) {
                required_inputs_finite = false;
                continue;
            }
            float depth_confidence =
                CurrentDepthHistory[SceneAnalysisIndex(
                    SBS_SCENE_DEPTH_HISTORY_VALID_DEPTH_CONFIDENCE,
                    cell)];
            if (!isfinite(depth_confidence)) {
                required_inputs_finite = false;
                continue;
            }
            float roi_coverage = Analysis(
                SBS_SCENE_ANALYSIS_GRID_CURRENT_ROI_COVERAGE,
                cell);
            if (!isfinite(roi_coverage)) {
                required_inputs_finite = false;
                continue;
            }
            valid_cells += 1.0f;
            if (roi_coverage > 0.5f) {
                depth_coverage_cells += 1.0f;
                depth_confidence_sum += saturate(depth_confidence);
            }
            viewport_first_x = min(viewport_first_x, x);
            viewport_end_x = max(viewport_end_x, x + 1u);
            float support = Dense(
                SBS_SCENE_DENSE_OUT_SCROLL_SUPPORT,
                cell);
            float motion_y = Dense(
                SBS_SCENE_DENSE_OUT_LOCAL_MOTION_Y,
                cell);
            if (!isfinite(support) || !isfinite(motion_y)) {
                required_inputs_finite = false;
                continue;
            }
            scroll_weight += support * abs(motion_y);
            scroll_signed += support * motion_y;
            scroll_support_cells += support > 0.20f ? 1.0f : 0.0f;
        }
    }

    bool meta_finite = true;
    bool meta_reserved_zero = true;
    [unroll]
    for (uint word = 0u; word < SBS_SCENE_META_WORD_COUNT; ++word) {
        float value = Meta[word];
        meta_finite = meta_finite && isfinite(value);
        if (word >= 28u) {
            meta_reserved_zero =
                meta_reserved_zero && abs(value) <= 1e-8f;
        }
    }
    bool adaptive_finite =
        all(isfinite(AdaptiveState[SBS_STATE_VECTOR_RANGE_COLLAPSED])) &&
        all(isfinite(AdaptiveState[SBS_STATE_VECTOR_ANALYSIS_FLAGS])) &&
        all(isfinite(AdaptiveState[
            SBS_STATE_VECTOR_ADAPTIVE_POP_RATIO])) &&
        all(isfinite(AdaptiveState[
            SBS_STATE_VECTOR_VALID_DEPTH_FRACTION])) &&
        all(isfinite(AdaptiveState[SBS_STATE_VECTOR_CUT_FLAGS]));
    bool input_valid = SceneAbiMatches() &&
                       required_inputs_finite &&
                       valid_cells > 0.10f * canvas * canvas &&
                       meta_finite &&
                       adaptive_finite &&
                       meta_reserved_zero;
    if (!input_valid) {
        EvidenceGlobal[
            SBS_SCENE_GLOBAL_OUT_ZERO_PLANE_NEUTRAL_ABSTAIN] =
            SceneLogit(0.98f);
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_OOD] = 1.0f;
        EvidenceGlobal[
            SBS_SCENE_GLOBAL_OUT_BACKEND_OUTPUT_VALID] = 0.0f;
        return;
    }

    // Locate the widest persistent quiet vertical gutter. It is a conservative segmentation
    // primitive: if no stable separator exists, rules evaluate one full viewport envelope.
    uint best_gutter_start = 0u;
    uint best_gutter_end = 0u;
    uint run_start = 0u;
    bool in_run = false;
    [loop]
    for (uint x = viewport_first_x; x < viewport_end_x; ++x) {
        float valid_in_column = 0.0f;
        float gutter_in_column = 0.0f;
        [loop]
        for (uint y = 0u; y < canvas; ++y) {
            uint2 cell = uint2(x, y);
            if (Analysis(
                    SBS_SCENE_ANALYSIS_GRID_VIEWPORT_VALID,
                    cell) <= 0.5f) {
                continue;
            }
            valid_in_column += 1.0f;
            gutter_in_column +=
                Dense(SBS_SCENE_DENSE_OUT_STABLE_GUTTER, cell) > 0.70f ?
                    1.0f :
                    0.0f;
        }
        bool gutter_column =
            valid_in_column > 0.0f &&
            gutter_in_column >= 0.76f * valid_in_column;
        if (gutter_column && !in_run) {
            run_start = x;
            in_run = true;
        }
        if ((!gutter_column || x + 1u == viewport_end_x) && in_run) {
            uint run_end =
                gutter_column && x + 1u == viewport_end_x ? x + 1u : x;
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

    float best_score = 0.0f;
    float second_score = 0.0f;
    float4 best_bounds = float4(0.0f, 0.0f, 1.0f, 1.0f);
    uint best_layout = SBS_SCENE_LAYOUT_DECISION_NO_TARGET;
    float best_coverage = 0.0f;
    bool video_segmentation_ambiguous = false;
    float best_video_score = 0.0f;
    float second_video_score = 0.0f;
    bool split =
        best_gutter_end > best_gutter_start &&
        best_gutter_start > viewport_first_x &&
        best_gutter_end < viewport_end_x;
    RegionEvidence first = EvaluateRegion(
        viewport_first_x,
        split ? best_gutter_start : viewport_end_x);
    ConsiderVideoRunCandidates(
        viewport_first_x,
        split ? best_gutter_start : viewport_end_x,
        video_segmentation_ambiguous,
        best_video_score,
        second_video_score,
        best_score,
        second_score,
        best_bounds,
        best_layout,
        best_coverage);
    ConsiderCandidate(
        first.content_score,
        first.content_bounds,
        SBS_SCENE_LAYOUT_DECISION_CONTENT_COLLAGE,
        first.content_coverage,
        best_score,
        second_score,
        best_bounds,
        best_layout,
        best_coverage);
    if (split) {
        RegionEvidence second = EvaluateRegion(
            best_gutter_end,
            viewport_end_x);
        ConsiderVideoRunCandidates(
            best_gutter_end,
            viewport_end_x,
            video_segmentation_ambiguous,
            best_video_score,
            second_video_score,
            best_score,
            second_score,
            best_bounds,
            best_layout,
            best_coverage);
        ConsiderCandidate(
            second.content_score,
            second.content_bounds,
            SBS_SCENE_LAYOUT_DECISION_CONTENT_COLLAGE,
            second.content_coverage,
            best_score,
            second_score,
            best_bounds,
            best_layout,
            best_coverage);
    }

    float winner_ratio =
        second_score > 1e-6f ? best_score / second_score : 99.0f;
    float winner_margin =
        best_score > 1e-6f ?
            saturate((best_score - second_score) / best_score) :
            0.0f;
    bool ambiguous =
        best_score > 0.0f &&
        (video_segmentation_ambiguous ||
         (second_video_score > 0.0f &&
          best_video_score / second_video_score < 1.25f) ||
         (second_score > 0.0f && winner_ratio < 1.25f));
    bool insufficient =
        best_score <= 0.0f || best_coverage < 0.02f;
    float roi_confidence =
        !input_valid || ambiguous || insufficient ?
            0.0f :
            saturate(0.45f + 0.35f * winner_margin +
                     0.20f * min(best_score * 3.0f, 1.0f));
    float mask_confidence =
        !input_valid ? 0.0f :
        saturate(0.5f * best_coverage + 0.5f * winner_margin);
    float ood =
        !input_valid ? 1.0f :
        ambiguous ? 0.75f :
        insufficient ? 0.60f :
        saturate(0.35f - winner_margin);
    bool identity_target =
        BoundsArea(best_bounds) >= 0.90f &&
        best_bounds.x <= 0.05f && best_bounds.y <= 0.05f &&
        best_bounds.z >= 0.95f && best_bounds.w >= 0.95f;
    if (identity_target && !ambiguous && !insufficient) {
        best_layout = SBS_SCENE_LAYOUT_DECISION_IDENTITY_FULLSCREEN;
        best_bounds = float4(0.0f, 0.0f, 1.0f, 1.0f);
    }

    float scroll_agreement =
        scroll_weight > 1e-6f ?
            abs(scroll_signed) / scroll_weight :
            0.0f;
    float scroll_fraction =
        valid_cells > 0.0f ? scroll_support_cells / valid_cells : 0.0f;
    float scroll_confidence =
        scroll_fraction >= 0.24f && scroll_agreement >= 0.82f ?
            saturate(scroll_fraction * scroll_agreement) :
            0.0f;
    float scroll_y =
        scroll_weight > 1e-6f ?
            clamp(scroll_signed / scroll_weight, -1.0f, 1.0f) :
            0.0f;

    float no_target_probability =
        saturate(1.0f - roi_confidence);
    float video_probability =
        best_layout == SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO ?
            roi_confidence :
            0.02f;
    float content_probability =
        best_layout == SBS_SCENE_LAYOUT_DECISION_CONTENT_COLLAGE ?
            roi_confidence :
            0.02f;
    float identity_probability =
        identity_target ?
            roi_confidence :
            0.02f;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_LAYOUT_NO_TARGET] =
        SceneLogit(max(no_target_probability, 0.02f));
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_LAYOUT_PRIMARY_VIDEO] =
        SceneLogit(max(video_probability, 0.02f));
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_LAYOUT_CONTENT_COLLAGE] =
        SceneLogit(max(content_probability, 0.02f));
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_LAYOUT_IDENTITY_FULLSCREEN] =
        SceneLogit(max(identity_probability, 0.02f));
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_LAYOUT_AMBIGUOUS] =
        SceneLogit(ambiguous ? 0.90f : 0.05f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_LAYOUT_SCROLL] =
        SceneLogit(max(scroll_confidence, 0.02f));

    float4 adaptive_flags =
        AdaptiveState[SBS_STATE_VECTOR_RANGE_COLLAPSED];
    float4 adaptive_analysis =
        AdaptiveState[SBS_STATE_VECTOR_ANALYSIS_FLAGS];
    uint analysis_flags =
        (uint)SBS_STATE_ANALYSIS_FLAGS(adaptive_analysis);
    bool geometry_reset =
        (EffectiveResetFlags() & SBS_SCENE_RESET_FLAGS_GEOMETRY) != 0u;
    bool hard_cut =
        SBS_STATE_HARD_CUT_PULSE(adaptive_flags) > 0.5f &&
        !geometry_reset;
    bool exposure =
        (analysis_flags & ANALYSIS_FLAG_EXPOSURE_LIKE) != 0u;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_EVENT_SAME_SHOT] =
        SceneLogit(!hard_cut && !exposure ? 0.95f : 0.05f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_EVENT_HARD_CUT] =
        SceneLogit(hard_cut ? 0.98f : 0.02f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_EVENT_FADE_OR_DISSOLVE] =
        SceneLogit(0.02f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_EVENT_FLASH_OR_EXPOSURE] =
        SceneLogit(exposure ? 0.95f : 0.02f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_EVENT_SCROLL] =
        SceneLogit(max(scroll_confidence, 0.02f));
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_EVENT_GEOMETRY_RESET] =
        SceneLogit(geometry_reset ? 0.995f : 0.02f);

    float4 adaptive_profile =
        AdaptiveState[SBS_STATE_VECTOR_ADAPTIVE_POP_RATIO];
    float4 adaptive_valid_depth =
        AdaptiveState[SBS_STATE_VECTOR_VALID_DEPTH_FRACTION];
    float history_depth_fraction =
        depth_coverage_cells > 0.0f ?
            saturate(depth_confidence_sum / depth_coverage_cells) :
            0.0f;
    float valid_depth_fraction = min(
        saturate(
            SBS_STATE_VALID_DEPTH_FRACTION(adaptive_valid_depth)),
        history_depth_fraction);
    float depth_reliability = smoothstep(
        0.05f,
        0.35f,
        valid_depth_fraction);
    float resolved_strength = clamp(
        scene_pop_floor *
            max(SBS_STATE_ADAPTIVE_POP_RATIO(adaptive_profile), 1.0f),
        scene_pop_floor,
        max(scene_pop_ceiling, scene_pop_floor));
    [unroll]
    for (uint action = 0u; action < SBS_SCENE_POP_ACTION_COUNT; ++action) {
        float action_strength = lerp(
            scene_pop_floor,
            scene_pop_ceiling,
            (float)action / (float)(SBS_SCENE_POP_ACTION_COUNT - 1u));
        EvidenceGlobal[
            SBS_SCENE_GLOBAL_OUT_POP_ACTION_SAFE_0 + action] =
            SceneLogit(
                action_strength <= resolved_strength + 1e-5f ?
                    0.90f :
                    0.10f);
    }

    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_ZERO_PLANE_SUBJECT] =
        SceneLogit(scene_zero_plane_mode < 1.5f ? 0.95f : 0.02f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_ZERO_PLANE_MEDIAN] =
        SceneLogit(
            abs(scene_zero_plane_mode - 2.0f) < 0.5f ?
                0.95f :
                0.02f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_ZERO_PLANE_BACKGROUND] =
        SceneLogit(scene_zero_plane_mode > 2.5f ? 0.95f : 0.02f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_ZERO_PLANE_NEUTRAL_ABSTAIN] =
        SceneLogit(input_valid ? 0.02f : 0.98f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_ROI] = roi_confidence;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_MASK] = mask_confidence;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_EVENT] =
        lerp(0.25f, 0.90f, depth_reliability);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_POP] =
        lerp(0.25f, 0.90f, depth_reliability);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_ZERO_PLANE] =
        lerp(0.25f, 0.90f, depth_reliability);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_OOD] = ood;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_GLOBAL_SCROLL_X] = 0.0f;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_GLOBAL_SCROLL_Y] = scroll_y;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_GLOBAL_SCROLL_CONFIDENCE] =
        scroll_confidence;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_BACKEND_OUTPUT_VALID] =
        input_valid ? 1.0f : 0.0f;

    // Internal resolver scratch. The shared resolver consumes it and writes exact zero to the
    // public reserved output range, so the externally visible ABI remains fail-closed.
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_35] = best_bounds.x;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_36] = best_bounds.y;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_37] = best_bounds.z;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_38] = best_bounds.w;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_39] = second_score;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_40] =
        history_depth_fraction;
}
