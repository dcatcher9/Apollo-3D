StructuredBuffer<float> AnalysisGrid : register(t0);
StructuredBuffer<float> DenseOutput : register(t1);
StructuredBuffer<float> PreviousLayoutHistory : register(t2);
StructuredBuffer<float> CurrentDepthHistory : register(t3);
StructuredBuffer<float4> AdaptiveState : register(t4);
StructuredBuffer<float> Meta : register(t5);
StructuredBuffer<uint4> PreviousRuleState : register(t6);
RWStructuredBuffer<float> EvidenceGlobal : register(u0);
RWStructuredBuffer<float> RuleSummary : register(u1);

#include "include/sbs_scene_controller_constants.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"
#include "include/sbs_scene_rule_state.hlsl"
#include "include/sbs_scene_rules_summary.hlsl"

float Analysis(uint channel, uint2 cell) {
    return AnalysisGrid[SceneAnalysisIndex(channel, cell)];
}

float Dense(uint channel, uint2 cell) {
    return DenseOutput[SceneAnalysisIndex(channel, cell)];
}

#include "include/sbs_scene_rules_media.hlsl"

#define PreviousRuleStateWord SbsRulePreviousStateWord
#define PreviousRuleStateUint SbsRulePreviousStateUint
#define EffectiveResetFlags SbsRuleEffectiveResetFlags

bool ValidBounds(float4 bounds) {
    return all(bounds.zw > bounds.xy);
}

float BoundsArea(float4 bounds) {
    const float2 extent = max(bounds.zw - bounds.xy, 0.0f.xx);
    return extent.x * extent.y;
}

#include "include/sbs_scene_rules_finalize.hlsl"

struct SbsSerialRun {
    uint first;
    uint end;
    float mass;
    float fresh_mass;
};

float SerialProjectionValue(
    uint coordinate,
    uint base,
    uint stride,
    uint field)
{
    const float value =
        RuleSummary[base + coordinate * stride + field];
    return isfinite(value) ? max(value, 0.0f) : 0.0f;
}

float SerialCanonicalTemporalMass(float stored_temporal) {
    return SbsRuleTemporalMass(stored_temporal);
}

float SerialCanonicalFreshLiveness(
    float fresh_activity,
    float stored_temporal)
{
    const float fresh =
        isfinite(fresh_activity) ?
            saturate(fresh_activity) :
            0.0f;
    return
        SerialCanonicalTemporalMass(stored_temporal) > 0.0f ?
            fresh :
            0.0f;
}

bool SerialRunValid(SbsSerialRun run) {
    return run.mass > 0.0f && run.end > run.first;
}

bool SerialRunAmbiguous(
    SbsSerialRun best,
    SbsSerialRun runner)
{
    return
        runner.mass > 0.0f &&
        runner.end - runner.first >=
            SBS_RULE_TEMPORAL_RUN_AMBIGUITY_FRACTION *
            (float)(best.end - best.first);
}

void SerialConsiderRun(
    SbsSerialRun candidate,
    inout SbsSerialRun best,
    inout SbsSerialRun runner)
{
    if (!SerialRunValid(candidate)) {
        return;
    }
    const uint candidate_span = candidate.end - candidate.first;
    const uint best_span = best.end - best.first;
    const uint runner_span = runner.end - runner.first;
    const bool beats_best =
        candidate_span > best_span ||
        (
            candidate_span == best_span &&
            candidate.mass > best.mass
        );
    const bool beats_runner =
        candidate_span > runner_span ||
        (
            candidate_span == runner_span &&
            candidate.mass > runner.mass
        );
    if (beats_best) {
        runner = best;
        best = candidate;
    } else if (beats_runner) {
        runner = candidate;
    }
}

void SerialFindGlobalRun(
    uint first,
    uint end,
    float threshold,
    float opposite_extent,
    uint base,
    uint stride,
    uint mass_field,
    uint gutter_field,
    uint fresh_field,
    out SbsSerialRun best,
    out SbsSerialRun runner)
{
    best = (SbsSerialRun)0;
    runner = (SbsSerialRun)0;
    SbsSerialRun current = (SbsSerialRun)0;
    uint last_confirmed = 0u;
    uint unsupported_hole_cells = 0u;
    bool in_run = false;
    [loop]
    for (uint coordinate = first; coordinate < end; ++coordinate) {
        const float mass = SerialProjectionValue(
            coordinate,
            base,
            stride,
            mass_field);
        const float gutter_mass = SerialProjectionValue(
            coordinate,
            base,
            stride,
            gutter_field);
        const bool strong_gutter =
            opposite_extent > 0.0f &&
            gutter_mass >=
                SBS_RULE_TEMPORAL_GUTTER_AXIS_MEAN *
                opposite_extent;
        if (strong_gutter) {
            if (in_run) {
                current.end = last_confirmed + 1u;
                if (current.fresh_mass > 0.0f) {
                    SerialConsiderRun(current, best, runner);
                }
                current = (SbsSerialRun)0;
                in_run = false;
                unsupported_hole_cells = 0u;
            }
            continue;
        }
        if (mass >= threshold) {
            if (!in_run) {
                current.first = coordinate;
                in_run = true;
            }
            current.mass += mass;
            current.fresh_mass += SerialProjectionValue(
                coordinate,
                base,
                stride,
                fresh_field);
            last_confirmed = coordinate;
            unsupported_hole_cells = 0u;
        } else if (in_run) {
            unsupported_hole_cells += 1u;
            if (
                unsupported_hole_cells >
                    SBS_RULE_TEMPORAL_MAX_UNSUPPORTED_HOLE_CELLS
            ) {
                current.end = last_confirmed + 1u;
                if (current.fresh_mass > 0.0f) {
                    SerialConsiderRun(current, best, runner);
                }
                current = (SbsSerialRun)0;
                in_run = false;
                unsupported_hole_cells = 0u;
            }
        }
    }
    if (in_run) {
        current.end = last_confirmed + 1u;
        if (current.fresh_mass > 0.0f) {
            SerialConsiderRun(current, best, runner);
        }
    }
}

float2 SerialConditionalMediaGutterMass(
    uint coordinate,
    bool expand_x,
    SbsSerialRun orthogonal_seed)
{
    float2 result = 0.0f.xx;
    if (!SerialRunValid(orthogonal_seed)) {
        return result;
    }
    [loop]
    for (
        uint orthogonal = orthogonal_seed.first;
        orthogonal < orthogonal_seed.end;
        ++orthogonal
    ) {
        const uint2 cell = expand_x ?
            uint2(coordinate, orthogonal) :
            uint2(orthogonal, coordinate);
        bool media_valid;
        const float media = SbsRuleStoredMediaWeight(
            Dense(
                SBS_SCENE_DENSE_OUT_PHOTOGRAPHIC_CONTENT_COLLAGE,
                cell),
            media_valid);
        const float gutter = Dense(
            SBS_SCENE_DENSE_OUT_STABLE_GUTTER,
            cell);
        if (media_valid) {
            result.x += max(media, 0.0f);
        }
        if (isfinite(gutter)) {
            result.y += saturate(gutter);
        }
    }
    return result;
}

void SerialExpandRunThroughConditionalMedia(
    uint viewport_first,
    uint viewport_end,
    bool expand_x,
    SbsSerialRun orthogonal_seed,
    inout SbsSerialRun run)
{
    if (!SerialRunValid(run) || !SerialRunValid(orthogonal_seed)) {
        return;
    }
    const float opposite_extent =
        (float)(orthogonal_seed.end - orthogonal_seed.first);
    const float continuation_threshold =
        SBS_RULE_TEMPORAL_CONTINUATION_AXIS_FRACTION *
        opposite_extent;
    [loop]
    while (run.first > viewport_first) {
        const uint coordinate = run.first - 1u;
        const float2 evidence = SerialConditionalMediaGutterMass(
            coordinate,
            expand_x,
            orthogonal_seed);
        if (
            evidence.y >=
                    SBS_RULE_TEMPORAL_GUTTER_AXIS_MEAN *
                    opposite_extent ||
            evidence.x < continuation_threshold
        ) {
            break;
        }
        run.first = coordinate;
    }
    [loop]
    while (run.end < viewport_end) {
        const uint coordinate = run.end;
        const float2 evidence = SerialConditionalMediaGutterMass(
            coordinate,
            expand_x,
            orthogonal_seed);
        if (
            evidence.y >=
                    SBS_RULE_TEMPORAL_GUTTER_AXIS_MEAN *
                    opposite_extent ||
            evidence.x < continuation_threshold
        ) {
            break;
        }
        run.end = coordinate + 1u;
    }
}

void SerialStoreGlobalRun(
    uint first_field,
    SbsSerialRun best,
    bool ambiguous)
{
    RuleSummary[SbsRuleAttributionMetaIndex(first_field + 0u)] =
        (float)best.first;
    RuleSummary[SbsRuleAttributionMetaIndex(first_field + 1u)] =
        (float)best.end;
    RuleSummary[SbsRuleAttributionMetaIndex(first_field + 2u)] =
        SerialRunValid(best) ? 1.0f : 0.0f;
    RuleSummary[SbsRuleAttributionMetaIndex(first_field + 3u)] =
        ambiguous ? 1.0f : 0.0f;
}

void SerialBuildProjections(
    uint2 viewport_origin,
    uint2 viewport_size,
    out float valid_cells,
    out float required_invalid)
{
    const uint canvas = SBS_SCENE_ANALYSIS_CANVAS_SIZE;
    const uint2 viewport_end = viewport_origin + viewport_size;
    valid_cells = 0.0f;
    required_invalid = 0.0f;
    [loop]
    for (uint x = 0u; x < canvas; ++x) {
        float media_mass = 0.0f;
        float temporal_mass = 0.0f;
        float gutter_mass = 0.0f;
        float fresh_mass = 0.0f;
        float invalid = 0.0f;
        if (x >= viewport_origin.x && x < viewport_end.x) {
            [loop]
            for (uint y = viewport_origin.y; y < viewport_end.y; ++y) {
                const uint2 cell = uint2(x, y);
                const float viewport_flag = Analysis(
                    SBS_SCENE_ANALYSIS_GRID_VIEWPORT_VALID,
                    cell);
                const float stored_temporal = Dense(
                    SBS_SCENE_DENSE_OUT_PRIMARY_PLAYING_VIDEO,
                    cell);
                const float stored_unknown = Dense(
                    SBS_SCENE_DENSE_OUT_UNKNOWN_BACKGROUND,
                    cell);
                const float fresh = Analysis(
                    SBS_SCENE_ANALYSIS_GRID_TEMPORAL_ACTIVITY_OCCUPANCY,
                    cell);
                const float gutter = Dense(
                    SBS_SCENE_DENSE_OUT_STABLE_GUTTER,
                    cell);
                bool media_valid;
                const float media = SbsRuleStoredMediaWeight(
                    Dense(
                        SBS_SCENE_DENSE_OUT_PHOTOGRAPHIC_CONTENT_COLLAGE,
                        cell),
                    media_valid);
                const bool viewport_inputs_valid =
                    isfinite(viewport_flag) &&
                    viewport_flag >= 0.0f;
                const bool output_sentinel_valid =
                    isfinite(stored_unknown) &&
                    stored_unknown >= 0.0f &&
                    stored_unknown <= 1.0f;
                const bool temporal_inputs_valid =
                    isfinite(stored_temporal) &&
                    stored_temporal >= 0.0f &&
                    stored_temporal <= 1.0f;
                const bool fresh_inputs_valid =
                    isfinite(fresh) &&
                    fresh >= 0.0f &&
                    fresh <= 1.0f;
                const bool gutter_inputs_valid =
                    isfinite(gutter) &&
                    gutter >= 0.0f &&
                    gutter <= 1.0f;
                const bool cell_valid =
                    viewport_inputs_valid &&
                    viewport_flag > 0.5f &&
                    output_sentinel_valid &&
                    temporal_inputs_valid &&
                    fresh_inputs_valid &&
                    gutter_inputs_valid &&
                    media_valid;
                if (!viewport_inputs_valid) {
                    invalid += 1.0f;
                } else if (viewport_flag > 0.5f) {
                    invalid += cell_valid ? 0.0f : 1.0f;
                    valid_cells += cell_valid ? 1.0f : 0.0f;
                    media_mass += media_valid ? max(media, 0.0f) : 0.0f;
                    temporal_mass +=
                        temporal_inputs_valid ?
                            SerialCanonicalTemporalMass(stored_temporal) :
                            0.0f;
                    gutter_mass +=
                        gutter_inputs_valid ? gutter : 0.0f;
                    fresh_mass +=
                        temporal_inputs_valid && fresh_inputs_valid ?
                            SerialCanonicalFreshLiveness(
                                fresh,
                                stored_temporal) :
                            0.0f;
                }
            }
        }
        RuleSummary[SbsRuleColumnIndex(
            x,
            SBS_RULE_COLUMN_MEDIA_MASS)] = media_mass;
        RuleSummary[SbsRuleColumnIndex(
            x,
            SBS_RULE_COLUMN_MEDIA_INVALID)] = invalid;
        RuleSummary[SbsRuleColumnIndex(
            x,
            SBS_RULE_COLUMN_TEMPORAL_MASS)] = temporal_mass;
        RuleSummary[SbsRuleColumnIndex(
            x,
            SBS_RULE_COLUMN_GUTTER_MASS)] = gutter_mass;
        RuleSummary[SbsRuleColumnIndex(
            x,
            SBS_RULE_COLUMN_FRESH_ACTIVITY_MASS)] = fresh_mass;
        required_invalid += invalid;
    }

    [loop]
    for (uint y = 0u; y < canvas; ++y) {
        float media_mass = 0.0f;
        float temporal_mass = 0.0f;
        float gutter_mass = 0.0f;
        float fresh_mass = 0.0f;
        if (y >= viewport_origin.y && y < viewport_end.y) {
            [loop]
            for (uint x = viewport_origin.x; x < viewport_end.x; ++x) {
                const uint2 cell = uint2(x, y);
                const float stored_temporal = Dense(
                    SBS_SCENE_DENSE_OUT_PRIMARY_PLAYING_VIDEO,
                    cell);
                const float fresh = Analysis(
                    SBS_SCENE_ANALYSIS_GRID_TEMPORAL_ACTIVITY_OCCUPANCY,
                    cell);
                bool media_valid;
                media_mass += max(
                    SbsRuleStoredMediaWeight(
                        Dense(
                            SBS_SCENE_DENSE_OUT_PHOTOGRAPHIC_CONTENT_COLLAGE,
                            cell),
                        media_valid),
                    0.0f);
                const bool temporal_inputs_valid =
                    isfinite(stored_temporal) &&
                    stored_temporal >= 0.0f &&
                    stored_temporal <= 1.0f;
                const bool fresh_inputs_valid =
                    isfinite(fresh) &&
                    fresh >= 0.0f &&
                    fresh <= 1.0f;
                const float gutter = Dense(
                    SBS_SCENE_DENSE_OUT_STABLE_GUTTER,
                    cell);
                const bool gutter_inputs_valid =
                    isfinite(gutter) &&
                    gutter >= 0.0f &&
                    gutter <= 1.0f;
                temporal_mass +=
                    temporal_inputs_valid ?
                        SerialCanonicalTemporalMass(stored_temporal) :
                        0.0f;
                gutter_mass +=
                    gutter_inputs_valid ? gutter : 0.0f;
                fresh_mass +=
                    temporal_inputs_valid && fresh_inputs_valid ?
                        SerialCanonicalFreshLiveness(
                            fresh,
                            stored_temporal) :
                        0.0f;
            }
        }
        RuleSummary[SbsRuleMediaRowIndex(y)] = media_mass;
        RuleSummary[SbsRuleTemporalRowIndex(
            y,
            SBS_RULE_TEMPORAL_ROW_ACTIVE_MASS)] = temporal_mass;
        RuleSummary[SbsRuleTemporalRowIndex(
            y,
            SBS_RULE_TEMPORAL_ROW_FRESH_ACTIVITY_MASS)] = fresh_mass;
        RuleSummary[SbsRuleTemporalRowIndex(
            y,
            SBS_RULE_TEMPORAL_ROW_GUTTER_MASS)] = gutter_mass;
    }
}

void SerialBuildCandidateMeta(
    uint2 viewport_origin,
    uint2 viewport_size)
{
    [unroll]
    for (uint field = 0u;
         field < SBS_RULE_ATTRIBUTION_META_FLOAT_COUNT;
         ++field) {
        RuleSummary[SbsRuleAttributionMetaIndex(field)] = 0.0f;
    }
    const uint2 viewport_end = viewport_origin + viewport_size;
    SbsSerialRun rows;
    SbsSerialRun row_runner;
    SbsSerialRun columns;
    SbsSerialRun column_runner;
    SerialFindGlobalRun(
        viewport_origin.y,
        viewport_end.y,
        max(1.0f, ceil(
            SBS_RULE_TEMPORAL_EVENT_AXIS_FILL *
            (float)viewport_size.x)),
        (float)viewport_size.x,
        SBS_RULE_TEMPORAL_ROW_BASE,
        SBS_RULE_TEMPORAL_ROW_STRIDE,
        SBS_RULE_TEMPORAL_ROW_ACTIVE_MASS,
        SBS_RULE_TEMPORAL_ROW_GUTTER_MASS,
        SBS_RULE_TEMPORAL_ROW_FRESH_ACTIVITY_MASS,
        rows,
        row_runner);
    SerialFindGlobalRun(
        viewport_origin.x,
        viewport_end.x,
        max(1.0f, ceil(
            SBS_RULE_TEMPORAL_EVENT_AXIS_FILL *
            (float)viewport_size.y)),
        (float)viewport_size.y,
        SBS_RULE_COLUMN_BASE,
        SBS_RULE_COLUMN_STRIDE,
        SBS_RULE_COLUMN_TEMPORAL_MASS,
        SBS_RULE_COLUMN_GUTTER_MASS,
        SBS_RULE_COLUMN_FRESH_ACTIVITY_MASS,
        columns,
        column_runner);
    const bool rows_ambiguous =
        SerialRunAmbiguous(rows, row_runner);
    const bool columns_ambiguous =
        SerialRunAmbiguous(columns, column_runner);
    const SbsSerialRun seed_rows = rows;
    const SbsSerialRun seed_columns = columns;
    SerialExpandRunThroughConditionalMedia(
        viewport_origin.y,
        viewport_end.y,
        false,
        seed_columns,
        rows);
    SerialExpandRunThroughConditionalMedia(
        viewport_origin.x,
        viewport_end.x,
        true,
        seed_rows,
        columns);
    SerialStoreGlobalRun(
        SBS_RULE_ATTRIBUTION_GLOBAL_X_FIRST,
        columns,
        columns_ambiguous);
    SerialStoreGlobalRun(
        SBS_RULE_ATTRIBUTION_GLOBAL_Y_FIRST,
        rows,
        rows_ambiguous);
    RuleSummary[SbsRuleAttributionMetaIndex(
        SBS_RULE_ATTRIBUTION_SEED_X_FIRST)] =
        (float)seed_columns.first;
    RuleSummary[SbsRuleAttributionMetaIndex(
        SBS_RULE_ATTRIBUTION_SEED_X_END)] =
        (float)seed_columns.end;
    RuleSummary[SbsRuleAttributionMetaIndex(
        SBS_RULE_ATTRIBUTION_SEED_Y_FIRST)] =
        (float)seed_rows.first;
    RuleSummary[SbsRuleAttributionMetaIndex(
        SBS_RULE_ATTRIBUTION_SEED_Y_END)] =
        (float)seed_rows.end;
    RuleSummary[SbsRuleAttributionMetaIndex(
        SBS_RULE_ATTRIBUTION_SEED_VALID)] =
        SerialRunValid(seed_columns) &&
        SerialRunValid(seed_rows) ? 1.0f : 0.0f;

    float selected_temporal_mass = 0.0f;
    float selected_fresh_mass = 0.0f;
    if (
        SerialRunValid(seed_columns) &&
        SerialRunValid(seed_rows)
    ) {
        [loop]
        for (uint y = seed_rows.first; y < seed_rows.end; ++y) {
            [loop]
            for (uint x = seed_columns.first;
                 x < seed_columns.end;
                 ++x) {
                const uint2 cell = uint2(x, y);
                const float fresh = saturate(Analysis(
                    SBS_SCENE_ANALYSIS_GRID_TEMPORAL_ACTIVITY_OCCUPANCY,
                    cell));
                const float stored_temporal = Dense(
                    SBS_SCENE_DENSE_OUT_PRIMARY_PLAYING_VIDEO,
                    cell);
                selected_temporal_mass +=
                    SerialCanonicalTemporalMass(stored_temporal);
                selected_fresh_mass += SerialCanonicalFreshLiveness(
                    fresh,
                    stored_temporal);
            }
        }
    }
    RuleSummary[SbsRuleAttributionMetaIndex(
        SBS_RULE_ATTRIBUTION_GLOBAL_RECT_MASS)] =
        selected_temporal_mass;
    RuleSummary[SbsRuleAttributionMetaIndex(
        SBS_RULE_ATTRIBUTION_GLOBAL_RECT_FRESH)] =
        selected_fresh_mass;

    float coherence_histogram[
        SBS_RULE_TEMPORAL_COHERENCE_BINS *
        SBS_RULE_TEMPORAL_COHERENCE_BINS];
    [unroll]
    for (uint bin = 0u;
         bin < SBS_RULE_TEMPORAL_COHERENCE_BINS *
                   SBS_RULE_TEMPORAL_COHERENCE_BINS;
         ++bin) {
        coherence_histogram[bin] = 0.0f;
    }
    if (SerialRunValid(columns) && SerialRunValid(rows)) {
        const uint coherence_width = columns.end - columns.first;
        const uint coherence_height = rows.end - rows.first;
        [loop]
        for (uint y = rows.first; y < rows.end; ++y) {
            const uint y_bin = min(
                SBS_RULE_TEMPORAL_COHERENCE_BINS - 1u,
                ((y - rows.first) *
                 SBS_RULE_TEMPORAL_COHERENCE_BINS) /
                    coherence_height);
            [loop]
            for (uint x = columns.first; x < columns.end; ++x) {
                const uint2 cell = uint2(x, y);
                const float temporal =
                    SerialCanonicalTemporalMass(Dense(
                        SBS_SCENE_DENSE_OUT_PRIMARY_PLAYING_VIDEO,
                        cell));
                bool media_valid;
                const float media = SbsRuleStoredMediaWeight(
                    Dense(
                        SBS_SCENE_DENSE_OUT_PHOTOGRAPHIC_CONTENT_COLLAGE,
                        cell),
                    media_valid);
                const uint x_bin = min(
                    SBS_RULE_TEMPORAL_COHERENCE_BINS - 1u,
                    ((x - columns.first) *
                     SBS_RULE_TEMPORAL_COHERENCE_BINS) /
                        coherence_width);
                coherence_histogram[
                    x_bin * SBS_RULE_TEMPORAL_COHERENCE_BINS +
                    y_bin] += max(
                        temporal,
                        media_valid ? media : 0.0f);
            }
        }
    }
    float x_marginal[SBS_RULE_TEMPORAL_COHERENCE_BINS];
    float y_marginal[SBS_RULE_TEMPORAL_COHERENCE_BINS];
    [unroll]
    for (uint bin = 0u;
         bin < SBS_RULE_TEMPORAL_COHERENCE_BINS;
         ++bin) {
        x_marginal[bin] = 0.0f;
        y_marginal[bin] = 0.0f;
    }
    float coherence_total = 0.0f;
    [unroll]
    for (uint x_bin = 0u;
         x_bin < SBS_RULE_TEMPORAL_COHERENCE_BINS;
         ++x_bin) {
        [unroll]
        for (uint y_bin = 0u;
             y_bin < SBS_RULE_TEMPORAL_COHERENCE_BINS;
             ++y_bin) {
            const float mass = max(
                coherence_histogram[
                    x_bin * SBS_RULE_TEMPORAL_COHERENCE_BINS +
                    y_bin],
                0.0f);
            x_marginal[x_bin] += mass;
            y_marginal[y_bin] += mass;
            coherence_total += mass;
        }
    }
    float total_variation = 1.0f;
    if (coherence_total > 0.0f) {
        total_variation = 0.0f;
        const float inverse_total = 1.0f / coherence_total;
        [unroll]
        for (uint x_bin = 0u;
             x_bin < SBS_RULE_TEMPORAL_COHERENCE_BINS;
             ++x_bin) {
            [unroll]
            for (uint y_bin = 0u;
                 y_bin < SBS_RULE_TEMPORAL_COHERENCE_BINS;
                 ++y_bin) {
                const float observed =
                    coherence_histogram[
                        x_bin * SBS_RULE_TEMPORAL_COHERENCE_BINS +
                        y_bin] *
                    inverse_total;
                const float separable =
                    x_marginal[x_bin] *
                    y_marginal[y_bin] *
                    inverse_total *
                    inverse_total;
                total_variation +=
                    0.5f * abs(observed - separable);
            }
        }
    }
    RuleSummary[SbsRuleAttributionMetaIndex(
        SBS_RULE_ATTRIBUTION_GLOBAL_RECT_COHERENCE)] =
        saturate(1.0f - total_variation);
}

void SerialBuildEventAggregate(
    uint2 viewport_origin,
    uint2 viewport_size,
    inout float required_invalid)
{
    [loop]
    for (uint group = 0u; group < SBS_RULE_EVENT_GROUP_COUNT; ++group) {
        [loop]
        for (uint field = 0u;
             field < SBS_RULE_EVENT_GROUP_STRIDE;
             ++field) {
            RuleSummary[SbsRuleEventGroupIndex(group, field)] = 0.0f;
        }
    }

    const int2 viewport_first = int2(viewport_origin);
    const int2 viewport_end =
        int2(viewport_origin + viewport_size);
    const bool layout_history_valid =
        scene_history_valid != 0u &&
        SbsRulePreviousTargetsUsable();
    float4 previous_video_bounds;
    const bool previous_video_target =
        SbsRulePreviousVideoTarget(previous_video_bounds);
    float4 event_sum = 0.0f.xxxx;
    float fresh_activity_mass = 0.0f;
    float fresh_activity_inside_previous_target = 0.0f;
    float3 scroll_votes_x = 0.0f.xxx;
    float3 scroll_votes_y = 0.0f.xxx;

    [loop]
    for (uint y = viewport_origin.y;
         y < (uint)viewport_end.y;
         ++y) {
        [loop]
        for (uint x = viewport_origin.x;
             x < (uint)viewport_end.x;
             ++x) {
            const uint2 cell = uint2(x, y);
            const float roi_coverage = Analysis(
                SBS_SCENE_ANALYSIS_GRID_CURRENT_ROI_COVERAGE,
                cell);
            const float depth_confidence =
                CurrentDepthHistory[SceneAnalysisIndex(
                    SBS_SCENE_DEPTH_HISTORY_VALID_DEPTH_CONFIDENCE,
                    cell)];
            const float structural = Dense(
                SBS_SCENE_DENSE_OUT_STRUCTURAL_CONTENT_CUT_SUPPORT,
                cell);
            const float exposure = Dense(
                SBS_SCENE_DENSE_OUT_EXPOSURE_ONLY_CHANGE_SUPPORT,
                cell);
            const float activity = Analysis(
                SBS_SCENE_ANALYSIS_GRID_TEMPORAL_ACTIVITY_OCCUPANCY,
                cell);
            const float texture_evidence = Analysis(
                SBS_SCENE_ANALYSIS_GRID_DENSE_EDGE_TEXTURE,
                cell);
            const float2 viewport_uv = float2(
                Analysis(
                    SBS_SCENE_ANALYSIS_GRID_VIEWPORT_X,
                    cell),
                Analysis(
                    SBS_SCENE_ANALYSIS_GRID_VIEWPORT_Y,
                    cell));
            if (
                !isfinite(roi_coverage) ||
                !isfinite(depth_confidence) ||
                !isfinite(structural) ||
                !isfinite(exposure) ||
                !isfinite(activity) ||
                !isfinite(texture_evidence) ||
                !all(isfinite(viewport_uv))
            ) {
                required_invalid += 1.0f;
                continue;
            }
            if (roi_coverage > 0.5f) {
                const float depth_weight =
                    saturate(depth_confidence);
                event_sum += float4(
                    1.0f,
                    depth_weight,
                    saturate(structural) * depth_weight,
                    saturate(exposure) * depth_weight);
            }
            const float fresh = SerialCanonicalFreshLiveness(
                activity,
                Dense(
                    SBS_SCENE_DENSE_OUT_PRIMARY_PLAYING_VIDEO,
                    cell));
            fresh_activity_mass += fresh;
            const bool inside_previous_video_target =
                previous_video_target &&
                all(viewport_uv >= previous_video_bounds.xy) &&
                all(viewport_uv <= previous_video_bounds.zw);
            if (inside_previous_video_target) {
                fresh_activity_inside_previous_target += fresh;
            }
            const bool scroll_eligible =
                layout_history_valid &&
                saturate(texture_evidence) >=
                    SBS_RULE_SCROLL_MIN_TEXTURE &&
                !inside_previous_video_target;
            const bool observable_horizontal =
                scroll_eligible &&
                (int)x >= viewport_first.x + 3 &&
                (int)x + 3 < viewport_end.x;
            const bool observable_vertical =
                scroll_eligible &&
                (int)y >= viewport_first.y + 3 &&
                (int)y + 3 < viewport_end.y;
            if (!observable_horizontal && !observable_vertical) {
                continue;
            }
            const uint current_signature =
                (uint)round(saturate(Analysis(
                    SBS_SCENE_ANALYSIS_GRID_LUMINANCE_ORDINAL,
                    cell)) * 255.0f);
            float scroll_costs_x[7];
            float scroll_costs_y[7];
            [unroll]
            for (uint shift_index = 0u;
                 shift_index < 7u;
                 ++shift_index) {
                scroll_costs_x[shift_index] = 0.0f;
                scroll_costs_y[shift_index] = 0.0f;
            }
            [unroll]
            for (int shift = -3; shift <= 3; ++shift) {
                if (observable_horizontal) {
                    const uint2 shifted = uint2(
                        int2(cell) + int2(shift, 0));
                    const uint previous_signature =
                        (uint)round(saturate(
                            PreviousLayoutHistory[
                                SceneAnalysisIndex(
                                    SBS_SCENE_LAYOUT_HISTORY_PREVIOUS_LUMINANCE_ORDINAL,
                                    shifted)]) * 255.0f);
                    scroll_costs_x[shift + 3] =
                        (float)countbits(
                            current_signature ^ previous_signature);
                }
                if (observable_vertical) {
                    const uint2 shifted = uint2(
                        int2(cell) + int2(0, shift));
                    const uint previous_signature =
                        (uint)round(saturate(
                            PreviousLayoutHistory[
                                SceneAnalysisIndex(
                                    SBS_SCENE_LAYOUT_HISTORY_PREVIOUS_LUMINANCE_ORDINAL,
                                    shifted)]) * 255.0f);
                    scroll_costs_y[shift + 3] =
                        (float)countbits(
                            current_signature ^ previous_signature);
                }
            }
            scroll_votes_x += SbsRuleClassifyScrollVote(
                observable_horizontal,
                scroll_costs_x[0],
                scroll_costs_x[1],
                scroll_costs_x[2],
                scroll_costs_x[3],
                scroll_costs_x[4],
                scroll_costs_x[5],
                scroll_costs_x[6]);
            scroll_votes_y += SbsRuleClassifyScrollVote(
                observable_vertical,
                scroll_costs_y[0],
                scroll_costs_y[1],
                scroll_costs_y[2],
                scroll_costs_y[3],
                scroll_costs_y[4],
                scroll_costs_y[5],
                scroll_costs_y[6]);
        }
    }

    RuleSummary[SbsRuleEventGroupIndex(
        0u,
        SBS_RULE_EVENT_GROUP_ROI_CELLS)] = event_sum.x;
    RuleSummary[SbsRuleEventGroupIndex(
        0u,
        SBS_RULE_EVENT_GROUP_DEPTH_WEIGHT)] = event_sum.y;
    RuleSummary[SbsRuleEventGroupIndex(
        0u,
        SBS_RULE_EVENT_GROUP_ROI_STRUCTURAL_WEIGHT)] = event_sum.z;
    RuleSummary[SbsRuleEventGroupIndex(
        0u,
        SBS_RULE_EVENT_GROUP_ROI_EXPOSURE_WEIGHT)] = event_sum.w;
    RuleSummary[SbsRuleEventGroupIndex(
        0u,
        SBS_RULE_EVENT_GROUP_SCROLL_Y_WEIGHT)] = scroll_votes_y.x;
    RuleSummary[SbsRuleEventGroupIndex(
        0u,
        SBS_RULE_EVENT_GROUP_SCROLL_Y_SIGNED)] = scroll_votes_y.y;
    RuleSummary[SbsRuleEventGroupIndex(
        0u,
        SBS_RULE_EVENT_GROUP_SCROLL_Y_SUPPORT)] = scroll_votes_y.z;
    RuleSummary[SbsRuleEventGroupIndex(
        0u,
        SBS_RULE_EVENT_GROUP_SCROLL_X_WEIGHT)] = scroll_votes_x.x;
    RuleSummary[SbsRuleEventGroupIndex(
        0u,
        SBS_RULE_EVENT_GROUP_SCROLL_X_SIGNED)] = scroll_votes_x.y;
    RuleSummary[SbsRuleEventGroupIndex(
        0u,
        SBS_RULE_EVENT_GROUP_SCROLL_X_SUPPORT)] = scroll_votes_x.z;
    RuleSummary[SbsRuleEventGroupIndex(
        0u,
        SBS_RULE_EVENT_GROUP_FRESH_ACTIVITY_MASS)] =
        fresh_activity_mass;
    RuleSummary[SbsRuleEventGroupIndex(
        0u,
        SBS_RULE_EVENT_GROUP_FRESH_ACTIVITY_INSIDE_PREVIOUS_TARGET)] =
        fresh_activity_inside_previous_target;
    RuleSummary[SBS_RULE_EVENT_SOURCE_MARKER] =
        SBS_RULE_EVENT_SOURCE_PREAGGREGATED;
}

#define SBS_SCENE_RULES_POLICY_ONLY
#include "sbs_scene_rules_reduce_cs.hlsl"
#undef SBS_SCENE_RULES_POLICY_ONLY

[numthreads(1, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    [loop]
    for (uint field = 0u;
         field < SBS_RULE_SUMMARY_FLOAT_COUNT;
         ++field) {
        RuleSummary[field] = 0.0f;
    }
    uint2 viewport_origin;
    uint2 viewport_size;
    SceneViewport(
        SBS_SCENE_ANALYSIS_CANVAS_SIZE,
        viewport_origin,
        viewport_size);
    float valid_cells;
    float required_invalid;
    SerialBuildProjections(
        viewport_origin,
        viewport_size,
        valid_cells,
        required_invalid);
    SerialBuildCandidateMeta(
        viewport_origin,
        viewport_size);
    SerialBuildEventAggregate(
        viewport_origin,
        viewport_size,
        required_invalid);
    RuleSummary[SbsRuleReduceAggregateIndex(
        SBS_RULE_REDUCE_AGGREGATE_VALID_CELLS)] =
        valid_cells;
    RuleSummary[SbsRuleReduceAggregateIndex(
        SBS_RULE_REDUCE_AGGREGATE_REQUIRED_INVALID)] =
        required_invalid;
    SbsRuleEmitGlobalEvidence();
}
