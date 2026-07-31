StructuredBuffer<float> AnalysisGrid : register(t0);
StructuredBuffer<float> DenseOutput : register(t1);
RWStructuredBuffer<float> RuleSummary : register(u0);

#include "include/sbs_scene_controller_constants.hlsl"
#include "include/sbs_scene_rules_summary.hlsl"
#include "include/sbs_scene_rules_media.hlsl"

// One workgroup builds both axis projections from evidence's canonical media, temporal, and
// stable-gutter lanes. A 32-row tile lets the owning column lane accumulate its column while one
// row lane folds the same media/gutter values from group memory.
static const uint SBS_RULE_MEDIA_TILE_ROWS = 32u;
groupshared float2 MediaGutterTile[
    SBS_RULE_MEDIA_TILE_ROWS * SBS_SCENE_ANALYSIS_CANVAS_SIZE];

float Analysis(uint channel, uint2 cell) {
    return AnalysisGrid[SceneAnalysisIndex(channel, cell)];
}

float Dense(uint channel, uint2 cell) {
    return DenseOutput[SceneAnalysisIndex(channel, cell)];
}

struct SbsColumnsRun {
    uint first;
    uint end;
    float mass;
    float fresh_mass;
};

float ColumnsProjectionValue(
    uint coordinate,
    uint base,
    uint stride,
    uint field)
{
    const float value =
        RuleSummary[base + coordinate * stride + field];
    return isfinite(value) ? max(value, 0.0f) : 0.0f;
}

bool ColumnsRunValid(SbsColumnsRun run) {
    return run.mass > 0.0f && run.end > run.first;
}

bool ColumnsRunAmbiguous(
    SbsColumnsRun best,
    SbsColumnsRun runner)
{
    return
        runner.mass > 0.0f &&
        runner.end - runner.first >=
            SBS_RULE_TEMPORAL_RUN_AMBIGUITY_FRACTION *
            (float)(best.end - best.first);
}

void ColumnsConsiderRun(
    SbsColumnsRun candidate,
    inout SbsColumnsRun best,
    inout SbsColumnsRun runner)
{
    if (!ColumnsRunValid(candidate)) {
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

void ColumnsFindGlobalRun(
    uint first,
    uint end,
    float threshold,
    float opposite_extent,
    uint base,
    uint stride,
    uint mass_field,
    uint gutter_field,
    uint fresh_field,
    out SbsColumnsRun best,
    out SbsColumnsRun runner)
{
    best = (SbsColumnsRun)0;
    runner = (SbsColumnsRun)0;
    SbsColumnsRun current = (SbsColumnsRun)0;
    uint last_confirmed = 0u;
    uint unsupported_hole_cells = 0u;
    bool in_run = false;
    [loop]
    for (uint coordinate = first; coordinate < end; ++coordinate) {
        const float mass = ColumnsProjectionValue(
            coordinate,
            base,
            stride,
            mass_field);
        const float gutter_mass = ColumnsProjectionValue(
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
                    ColumnsConsiderRun(current, best, runner);
                }
                current = (SbsColumnsRun)0;
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
            current.fresh_mass += ColumnsProjectionValue(
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
                    ColumnsConsiderRun(current, best, runner);
                }
                current = (SbsColumnsRun)0;
                in_run = false;
                unsupported_hole_cells = 0u;
            }
        }
    }
    if (in_run) {
        current.end = last_confirmed + 1u;
        if (current.fresh_mass > 0.0f) {
            ColumnsConsiderRun(current, best, runner);
        }
    }
}

void ColumnsExpandRunThroughConditionalMedia(
    uint viewport_first,
    uint viewport_end,
    float opposite_extent,
    uint projection_base,
    inout SbsColumnsRun run)
{
    if (!ColumnsRunValid(run) || opposite_extent <= 0.0f) {
        return;
    }
    const float continuation_threshold =
        SBS_RULE_TEMPORAL_CONTINUATION_AXIS_FRACTION *
        opposite_extent;
    [loop]
    while (run.first > viewport_first) {
        const uint coordinate = run.first - 1u;
        const float2 media_gutter = max(
            MediaGutterTile[projection_base + coordinate],
            0.0f.xx);
        if (
            media_gutter.y >=
                SBS_RULE_TEMPORAL_GUTTER_AXIS_MEAN * opposite_extent ||
            media_gutter.x < continuation_threshold
        ) {
            break;
        }
        run.first = coordinate;
    }
    [loop]
    while (run.end < viewport_end) {
        const uint coordinate = run.end;
        const float2 media_gutter = max(
            MediaGutterTile[projection_base + coordinate],
            0.0f.xx);
        if (
            media_gutter.y >=
                SBS_RULE_TEMPORAL_GUTTER_AXIS_MEAN * opposite_extent ||
            media_gutter.x < continuation_threshold
        ) {
            break;
        }
        run.end = coordinate + 1u;
    }
}

void ColumnsStoreGlobalRun(
    uint first_field,
    SbsColumnsRun best,
    bool ambiguous)
{
    RuleSummary[SbsRuleAttributionMetaIndex(first_field + 0u)] =
        (float)best.first;
    RuleSummary[SbsRuleAttributionMetaIndex(first_field + 1u)] =
        (float)best.end;
    RuleSummary[SbsRuleAttributionMetaIndex(first_field + 2u)] =
        ColumnsRunValid(best) ? 1.0f : 0.0f;
    RuleSummary[SbsRuleAttributionMetaIndex(first_field + 3u)] =
        ambiguous ? 1.0f : 0.0f;
}

[numthreads(128, 1, 1)]
void main(uint3 thread_id : SV_GroupThreadID) {
    const uint lane = thread_id.x;
    const uint canvas = SBS_SCENE_ANALYSIS_CANVAS_SIZE;
    uint2 viewport_origin;
    uint2 viewport_size;
    SceneViewport(canvas, viewport_origin, viewport_size);
    const uint2 viewport_end = viewport_origin + viewport_size;
    const bool active_column =
        lane >= viewport_origin.x && lane < viewport_end.x;

    if (lane < SBS_RULE_ATTRIBUTION_META_FLOAT_COUNT) {
        RuleSummary[SbsRuleAttributionMetaIndex(lane)] = 0.0f;
    }

    float column_media_mass = 0.0f;
    float column_media_invalid = 0.0f;
    float column_temporal_mass = 0.0f;
    float column_gutter_mass = 0.0f;
    float column_fresh_activity_mass = 0.0f;
    float column_main_valid = 0.0f;

    // Classify one 32-row tile at a time. Every persistent row lane is written, including rows
    // outside the current viewport, so aspect changes cannot expose stale scratch.
    [unroll]
    for (uint tile = 0u; tile < 4u; ++tile) {
        [unroll]
        for (uint offset = 0u;
             offset < SBS_RULE_MEDIA_TILE_ROWS;
             ++offset) {
            const uint y = tile * SBS_RULE_MEDIA_TILE_ROWS + offset;
            float media_weight = 0.0f;
            float gutter_weight = 0.0f;
            if (
                active_column &&
                y >= viewport_origin.y &&
                y < viewport_end.y
            ) {
                const uint2 cell = uint2(lane, y);
                const float viewport_flag = Analysis(
                    SBS_SCENE_ANALYSIS_GRID_VIEWPORT_VALID,
                    cell);
                bool cell_inputs_valid =
                    isfinite(viewport_flag) && viewport_flag >= 0.0f;
                if (cell_inputs_valid && viewport_flag > 0.5f) {
                    const float stored_temporal = Dense(
                        SBS_SCENE_DENSE_OUT_PRIMARY_PLAYING_VIDEO,
                        cell);
                    const float stored_unknown = Dense(
                        SBS_SCENE_DENSE_OUT_UNKNOWN_BACKGROUND,
                        cell);
                    const bool output_sentinel_valid =
                        isfinite(stored_unknown) &&
                        stored_unknown >= 0.0f &&
                        stored_unknown <= 1.0f;
                    const bool temporal_inputs_valid =
                        isfinite(stored_temporal) &&
                        stored_temporal >= 0.0f &&
                        stored_temporal <= 1.0f;
                    const float stored_fresh_activity = Analysis(
                        SBS_SCENE_ANALYSIS_GRID_TEMPORAL_ACTIVITY_OCCUPANCY,
                        cell);
                    const bool fresh_activity_inputs_valid =
                        isfinite(stored_fresh_activity) &&
                        stored_fresh_activity >= 0.0f &&
                        stored_fresh_activity <= 1.0f;
                    const float fresh_activity =
                        fresh_activity_inputs_valid ?
                            stored_fresh_activity :
                            0.0f;
                    const float temporal_mass =
                        temporal_inputs_valid ?
                            SbsRuleTemporalMass(stored_temporal) :
                            0.0f;
                    column_temporal_mass += temporal_mass;
                    column_fresh_activity_mass +=
                        temporal_mass > 0.0f ? fresh_activity : 0.0f;
                    bool media_inputs_valid;
                    media_weight = SbsRuleStoredMediaWeight(
                        Dense(
                            SBS_SCENE_DENSE_OUT_PHOTOGRAPHIC_CONTENT_COLLAGE,
                            cell),
                        media_inputs_valid);
                    const float stored_gutter = Dense(
                        SBS_SCENE_DENSE_OUT_STABLE_GUTTER,
                        cell);
                    const bool gutter_inputs_valid =
                        isfinite(stored_gutter) &&
                        stored_gutter >= 0.0f &&
                        stored_gutter <= 1.0f;
                    gutter_weight =
                        gutter_inputs_valid ? stored_gutter : 0.0f;
                    cell_inputs_valid =
                        output_sentinel_valid &&
                        temporal_inputs_valid &&
                        fresh_activity_inputs_valid &&
                        media_inputs_valid &&
                        gutter_inputs_valid;
                }
                column_media_invalid +=
                    cell_inputs_valid ? 0.0f : 1.0f;
                column_main_valid +=
                    cell_inputs_valid && viewport_flag > 0.5f ?
                        1.0f :
                        0.0f;
                column_media_mass += max(media_weight, 0.0f);
                column_gutter_mass += max(gutter_weight, 0.0f);
            }
            MediaGutterTile[
                offset * SBS_SCENE_ANALYSIS_CANVAS_SIZE + lane] =
                max(float2(media_weight, gutter_weight), 0.0f.xx);
        }
        GroupMemoryBarrierWithGroupSync();

        if (lane < SBS_RULE_MEDIA_TILE_ROWS) {
            const uint y = tile * SBS_RULE_MEDIA_TILE_ROWS + lane;
            float2 row_media_gutter_mass = 0.0f.xx;
            if (y >= viewport_origin.y && y < viewport_end.y) {
                [loop]
                for (uint x = viewport_origin.x;
                     x < viewport_end.x;
                     ++x) {
                    row_media_gutter_mass += MediaGutterTile[
                        lane * SBS_SCENE_ANALYSIS_CANVAS_SIZE + x];
                }
            }
            RuleSummary[SbsRuleMediaRowIndex(y)] =
                max(row_media_gutter_mass.x, 0.0f);
            RuleSummary[SbsRuleTemporalRowIndex(
                y,
                SBS_RULE_TEMPORAL_ROW_GUTTER_MASS)] =
                max(row_media_gutter_mass.y, 0.0f);
        }
        GroupMemoryBarrierWithGroupSync();

        // The media/gutter tile is dead after its row projection. Reuse it
        // for temporal/fresh row projection instead of rescanning every cell
        // later for validity, scroll, and temporal inputs.
        [unroll]
        for (uint temporal_offset = 0u;
             temporal_offset < SBS_RULE_MEDIA_TILE_ROWS;
             ++temporal_offset) {
            const uint temporal_y =
                tile * SBS_RULE_MEDIA_TILE_ROWS + temporal_offset;
            float2 temporal_fresh = 0.0f.xx;
            if (
                active_column &&
                temporal_y >= viewport_origin.y &&
                temporal_y < viewport_end.y
            ) {
                const uint2 temporal_cell = uint2(lane, temporal_y);
                const float stored_temporal = Dense(
                    SBS_SCENE_DENSE_OUT_PRIMARY_PLAYING_VIDEO,
                    temporal_cell);
                const float stored_fresh_activity = Analysis(
                    SBS_SCENE_ANALYSIS_GRID_TEMPORAL_ACTIVITY_OCCUPANCY,
                    temporal_cell);
                if (
                    isfinite(stored_temporal) &&
                    stored_temporal >= 0.0f &&
                    stored_temporal <= 1.0f &&
                    isfinite(stored_fresh_activity) &&
                    stored_fresh_activity >= 0.0f &&
                    stored_fresh_activity <= 1.0f
                ) {
                    const float temporal_mass =
                        SbsRuleTemporalMass(stored_temporal);
                    temporal_fresh = float2(
                        temporal_mass,
                        temporal_mass > 0.0f ?
                            stored_fresh_activity :
                            0.0f);
                }
            }
            MediaGutterTile[
                temporal_offset * canvas + lane] =
                temporal_fresh;
        }
        GroupMemoryBarrierWithGroupSync();

        if (lane < SBS_RULE_MEDIA_TILE_ROWS) {
            const uint temporal_y =
                tile * SBS_RULE_MEDIA_TILE_ROWS + lane;
            float2 row_temporal_fresh = 0.0f.xx;
            if (
                temporal_y >= viewport_origin.y &&
                temporal_y < viewport_end.y
            ) {
                [loop]
                for (uint x = viewport_origin.x;
                     x < viewport_end.x;
                     ++x) {
                    row_temporal_fresh += MediaGutterTile[
                        lane * canvas + x];
                }
            }
            RuleSummary[SbsRuleTemporalRowIndex(
                temporal_y,
                SBS_RULE_TEMPORAL_ROW_ACTIVE_MASS)] =
                max(row_temporal_fresh.x, 0.0f);
            RuleSummary[SbsRuleTemporalRowIndex(
                temporal_y,
                SBS_RULE_TEMPORAL_ROW_FRESH_ACTIVITY_MASS)] =
                max(row_temporal_fresh.y, 0.0f);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    RuleSummary[SbsRuleColumnIndex(
        lane,
        SBS_RULE_COLUMN_MEDIA_MASS)] = column_media_mass;
    RuleSummary[SbsRuleColumnIndex(
        lane,
        SBS_RULE_COLUMN_MEDIA_INVALID)] = column_media_invalid;
    RuleSummary[SbsRuleColumnIndex(
        lane,
        SBS_RULE_COLUMN_TEMPORAL_MASS)] = column_temporal_mass;
    RuleSummary[SbsRuleColumnIndex(
        lane,
        SBS_RULE_COLUMN_GUTTER_MASS)] = column_gutter_mass;
    RuleSummary[SbsRuleColumnIndex(
        lane,
        SBS_RULE_COLUMN_FRESH_ACTIVITY_MASS)] =
        column_fresh_activity_mass;

    // Validity is already known from the column-owned first pass. Reduce it
    // in shared memory so the scalar finalizer reads one pair, not 256 UAV
    // entries. Scroll is reduced where it originates, in the evidence groups.
    MediaGutterTile[lane] = float2(
        column_main_valid,
        column_media_invalid);
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = canvas / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            MediaGutterTile[lane] +=
                MediaGutterTile[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // The global projections are already resident in this workgroup's UAV
    // writes. Discover both first-axis runs here rather than spending a
    // serialized stage and barrier in the attribution pass.
    DeviceMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        const float2 validity = MediaGutterTile[0u];
        RuleSummary[SbsRuleReduceAggregateIndex(
            SBS_RULE_REDUCE_AGGREGATE_VALID_CELLS)] =
            validity.x;
        RuleSummary[SbsRuleReduceAggregateIndex(
            SBS_RULE_REDUCE_AGGREGATE_REQUIRED_INVALID)] =
            validity.y;

        SbsColumnsRun best_rows;
        SbsColumnsRun runner_rows;
        SbsColumnsRun best_columns;
        SbsColumnsRun runner_columns;
        ColumnsFindGlobalRun(
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
            best_rows,
            runner_rows);
        ColumnsFindGlobalRun(
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
            best_columns,
            runner_columns);
        const bool rows_ambiguous =
            ColumnsRunAmbiguous(best_rows, runner_rows);
        const bool columns_ambiguous =
            ColumnsRunAmbiguous(best_columns, runner_columns);
        const SbsColumnsRun seed_rows = best_rows;
        const SbsColumnsRun seed_columns = best_columns;
        ColumnsStoreGlobalRun(
            SBS_RULE_ATTRIBUTION_GLOBAL_X_FIRST,
            seed_columns,
            columns_ambiguous);
        ColumnsStoreGlobalRun(
            SBS_RULE_ATTRIBUTION_GLOBAL_Y_FIRST,
            seed_rows,
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
            ColumnsRunValid(seed_columns) &&
            ColumnsRunValid(seed_rows) ? 1.0f : 0.0f;
    }
    DeviceMemoryBarrierWithGroupSync();

    // Expand a temporal seed only through photographic support connected in
    // two dimensions. A viewport-wide row projection lets an unrelated side
    // animation extend the player's top/bottom edge; conditioning each axis
    // on the seed span of the other axis prevents that cross-axis leak while
    // still recovering a mostly static player border around localized motion.
    const uint expansion_seed_first_x = (uint)round(max(
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_SEED_X_FIRST)],
        0.0f));
    const uint expansion_seed_end_x = (uint)round(max(
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_SEED_X_END)],
        0.0f));
    const uint expansion_seed_first_y = (uint)round(max(
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_SEED_Y_FIRST)],
        0.0f));
    const uint expansion_seed_end_y = (uint)round(max(
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_SEED_Y_END)],
        0.0f));
    const bool expansion_seed_valid =
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_SEED_VALID)] > 0.5f &&
        expansion_seed_end_x > expansion_seed_first_x &&
        expansion_seed_end_y > expansion_seed_first_y;
    float2 conditional_column_media_gutter = 0.0f.xx;
    if (
        expansion_seed_valid &&
        lane >= viewport_origin.x &&
        lane < viewport_end.x
    ) {
        [loop]
        for (uint y = expansion_seed_first_y;
             y < expansion_seed_end_y;
             ++y) {
            const uint2 cell = uint2(lane, y);
            bool media_valid;
            const float media = SbsRuleStoredMediaWeight(
                Dense(
                    SBS_SCENE_DENSE_OUT_PHOTOGRAPHIC_CONTENT_COLLAGE,
                    cell),
                media_valid);
            const float gutter = Dense(
                SBS_SCENE_DENSE_OUT_STABLE_GUTTER,
                cell);
            conditional_column_media_gutter += float2(
                media_valid ? max(media, 0.0f) : 0.0f,
                isfinite(gutter) ? saturate(gutter) : 0.0f);
        }
    }
    float2 conditional_row_media_gutter = 0.0f.xx;
    if (
        expansion_seed_valid &&
        lane >= viewport_origin.y &&
        lane < viewport_end.y
    ) {
        [loop]
        for (uint x = expansion_seed_first_x;
             x < expansion_seed_end_x;
             ++x) {
            const uint2 cell = uint2(x, lane);
            bool media_valid;
            const float media = SbsRuleStoredMediaWeight(
                Dense(
                    SBS_SCENE_DENSE_OUT_PHOTOGRAPHIC_CONTENT_COLLAGE,
                    cell),
                media_valid);
            const float gutter = Dense(
                SBS_SCENE_DENSE_OUT_STABLE_GUTTER,
                cell);
            conditional_row_media_gutter += float2(
                media_valid ? max(media, 0.0f) : 0.0f,
                isfinite(gutter) ? saturate(gutter) : 0.0f);
        }
    }
    MediaGutterTile[lane] = conditional_column_media_gutter;
    MediaGutterTile[canvas + lane] = conditional_row_media_gutter;
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u && expansion_seed_valid) {
        SbsColumnsRun expanded_columns = (SbsColumnsRun)0;
        expanded_columns.first = expansion_seed_first_x;
        expanded_columns.end = expansion_seed_end_x;
        expanded_columns.mass = 1.0f;
        SbsColumnsRun expanded_rows = (SbsColumnsRun)0;
        expanded_rows.first = expansion_seed_first_y;
        expanded_rows.end = expansion_seed_end_y;
        expanded_rows.mass = 1.0f;
        ColumnsExpandRunThroughConditionalMedia(
            viewport_origin.x,
            viewport_end.x,
            (float)(expansion_seed_end_y - expansion_seed_first_y),
            0u,
            expanded_columns);
        ColumnsExpandRunThroughConditionalMedia(
            viewport_origin.y,
            viewport_end.y,
            (float)(expansion_seed_end_x - expansion_seed_first_x),
            canvas,
            expanded_rows);
        ColumnsStoreGlobalRun(
            SBS_RULE_ATTRIBUTION_GLOBAL_X_FIRST,
            expanded_columns,
            RuleSummary[SbsRuleAttributionMetaIndex(
                SBS_RULE_ATTRIBUTION_GLOBAL_X_AMBIGUOUS)] > 0.5f);
        ColumnsStoreGlobalRun(
            SBS_RULE_ATTRIBUTION_GLOBAL_Y_FIRST,
            expanded_rows,
            RuleSummary[SbsRuleAttributionMetaIndex(
                SBS_RULE_ATTRIBUTION_GLOBAL_Y_AMBIGUOUS)] > 0.5f);
    }
    DeviceMemoryBarrierWithGroupSync();

    // The selected global X/Y envelope is the only 2-D temporal hypothesis.
    // Reduce its exact mass here, while all 128 column lanes are still active,
    // so the scalar finalizer can reject incoherent cross-axis projections
    // without another GPU dispatch.
    const uint selected_first_x = (uint)round(max(
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_SEED_X_FIRST)],
        0.0f));
    const uint selected_end_x = (uint)round(max(
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_SEED_X_END)],
        0.0f));
    const uint selected_first_y = (uint)round(max(
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_SEED_Y_FIRST)],
        0.0f));
    const uint selected_end_y = (uint)round(max(
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_SEED_Y_END)],
        0.0f));
    const bool selected_valid =
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_SEED_VALID)] > 0.5f &&
        selected_end_x > selected_first_x &&
        selected_end_y > selected_first_y;
    const uint coherence_first_x = (uint)round(max(
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_GLOBAL_X_FIRST)],
        0.0f));
    const uint coherence_end_x = (uint)round(max(
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_GLOBAL_X_END)],
        0.0f));
    const uint coherence_first_y = (uint)round(max(
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_GLOBAL_Y_FIRST)],
        0.0f));
    const uint coherence_end_y = (uint)round(max(
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_GLOBAL_Y_END)],
        0.0f));
    const bool coherence_bounds_valid =
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_GLOBAL_X_VALID)] > 0.5f &&
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_GLOBAL_Y_VALID)] > 0.5f &&
        coherence_end_x > coherence_first_x &&
        coherence_end_y > coherence_first_y;
    float selected_temporal_mass = 0.0f;
    float selected_fresh_mass = 0.0f;
    if (
        selected_valid &&
        lane >= selected_first_x &&
        lane < selected_end_x
    ) {
        [loop]
        for (uint y = selected_first_y; y < selected_end_y; ++y) {
            const uint2 cell = uint2(lane, y);
            const float fresh = saturate(Analysis(
                SBS_SCENE_ANALYSIS_GRID_TEMPORAL_ACTIVITY_OCCUPANCY,
                cell));
            const float temporal = SbsRuleTemporalMass(Dense(
                SBS_SCENE_DENSE_OUT_PRIMARY_PLAYING_VIDEO,
                cell));
            selected_temporal_mass += temporal;
            selected_fresh_mass += temporal > 0.0f ? fresh : 0.0f;
        }
    }
    MediaGutterTile[lane] = float2(
        selected_temporal_mass,
        selected_fresh_mass);
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = canvas / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            MediaGutterTile[lane] +=
                MediaGutterTile[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) {
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_GLOBAL_RECT_MASS)] =
            MediaGutterTile[0u].x;
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_GLOBAL_RECT_FRESH)] =
            MediaGutterTile[0u].y;
    }
    GroupMemoryBarrierWithGroupSync();

    // A pair of independent axis projections is not enough to prove one rectangle: two crossed
    // regions can provide the widest X and Y runs while referring to different activity. Fold
    // temporal occupancy and the canonical static-media envelope into one fixed 4x4 joint
    // histogram. Its distance from the separable distribution implied by its marginals is a
    // cheap low-frequency 2-D coherence signal, not another candidate or route classifier.
    float coherence_bins[SBS_RULE_TEMPORAL_COHERENCE_BINS];
    [unroll]
    for (uint y_bin = 0u;
         y_bin < SBS_RULE_TEMPORAL_COHERENCE_BINS;
         ++y_bin) {
        coherence_bins[y_bin] = 0.0f;
    }
    if (
        coherence_bounds_valid &&
        lane >= coherence_first_x &&
        lane < coherence_end_x
    ) {
        const uint coherence_height =
            coherence_end_y - coherence_first_y;
        [loop]
        for (uint y = coherence_first_y; y < coherence_end_y; ++y) {
            const uint2 cell = uint2(lane, y);
            const float temporal = SbsRuleTemporalMass(Dense(
                SBS_SCENE_DENSE_OUT_PRIMARY_PLAYING_VIDEO,
                cell));
            bool media_valid;
            const float media = SbsRuleStoredMediaWeight(
                Dense(
                    SBS_SCENE_DENSE_OUT_PHOTOGRAPHIC_CONTENT_COLLAGE,
                    cell),
                media_valid);
            const uint y_bin = min(
                SBS_RULE_TEMPORAL_COHERENCE_BINS - 1u,
                ((y - coherence_first_y) *
                 SBS_RULE_TEMPORAL_COHERENCE_BINS) /
                    coherence_height);
            coherence_bins[y_bin] +=
                max(temporal, media_valid ? media : 0.0f);
        }
    }
    [unroll]
    for (uint y_bin = 0u;
         y_bin < SBS_RULE_TEMPORAL_COHERENCE_BINS;
         ++y_bin) {
        MediaGutterTile[
            lane * SBS_RULE_TEMPORAL_COHERENCE_BINS + y_bin] =
            float2(coherence_bins[y_bin], 0.0f);
    }
    GroupMemoryBarrierWithGroupSync();

    const uint coherence_cell_count =
        SBS_RULE_TEMPORAL_COHERENCE_BINS *
        SBS_RULE_TEMPORAL_COHERENCE_BINS;
    float coherence_bin_mass = 0.0f;
    if (lane < coherence_cell_count && coherence_bounds_valid) {
        const uint x_bin =
            lane / SBS_RULE_TEMPORAL_COHERENCE_BINS;
        const uint y_bin =
            lane % SBS_RULE_TEMPORAL_COHERENCE_BINS;
        const uint coherence_width =
            coherence_end_x - coherence_first_x;
        [loop]
        for (uint x = coherence_first_x; x < coherence_end_x; ++x) {
            const uint cell_x_bin = min(
                SBS_RULE_TEMPORAL_COHERENCE_BINS - 1u,
                ((x - coherence_first_x) *
                 SBS_RULE_TEMPORAL_COHERENCE_BINS) /
                    coherence_width);
            if (cell_x_bin == x_bin) {
                coherence_bin_mass += MediaGutterTile[
                    x * SBS_RULE_TEMPORAL_COHERENCE_BINS + y_bin].x;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < coherence_cell_count) {
        MediaGutterTile[lane] = float2(coherence_bin_mass, 0.0f);
    }
    GroupMemoryBarrierWithGroupSync();

    if (lane == 0u) {
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
                    MediaGutterTile[
                        x_bin * SBS_RULE_TEMPORAL_COHERENCE_BINS +
                        y_bin].x,
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
                        MediaGutterTile[
                            x_bin * SBS_RULE_TEMPORAL_COHERENCE_BINS +
                            y_bin].x *
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
}
