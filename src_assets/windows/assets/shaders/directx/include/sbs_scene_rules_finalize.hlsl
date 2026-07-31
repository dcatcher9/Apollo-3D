#ifndef SBS_SCENE_RULES_FINALIZE_HLSL
#define SBS_SCENE_RULES_FINALIZE_HLSL

// Deterministic scalar policy over the parallel projection pass. Candidate
// selection intentionally has only two paths:
//   1. one unique global temporal rectangle;
//   2. one quiet, global media hull.
// Anything else abstains and leaves the resolver on full-frame or its
// previously committed target.

float SbsRuleMediaColumnMass(uint column) {
    return max(
        RuleSummary[SbsRuleColumnIndex(
            column,
            SBS_RULE_COLUMN_MEDIA_MASS)],
        0.0f);
}

float SbsRuleMediaRowMass(uint row) {
    return max(RuleSummary[SbsRuleMediaRowIndex(row)], 0.0f);
}

float SbsRuleTemporalColumnMass(uint column) {
    return max(
        RuleSummary[SbsRuleColumnIndex(
            column,
            SBS_RULE_COLUMN_TEMPORAL_MASS)],
        0.0f);
}

uint SbsRuleCandidateMetaUint(uint field) {
    return (uint)round(max(
        RuleSummary[SbsRuleAttributionMetaIndex(field)],
        0.0f));
}

bool SbsRuleCandidateMetaFlag(uint field) {
    return
        RuleSummary[SbsRuleAttributionMetaIndex(field)] > 0.5f;
}

void SbsRuleQualifyGlobalScrollVotes(
    float3 votes,
    out float weight,
    out float signed_direction,
    out float support)
{
    weight = 0.0f;
    signed_direction = 0.0f;
    support = 0.0f;
    if (
        !all(isfinite(votes)) ||
        votes.x <= 0.0f ||
        votes.z <= 0.0f
    ) {
        return;
    }

    const float direction_consensus = abs(votes.y) / votes.z;
    if (direction_consensus >= SBS_RULE_CANDIDATE_MIN_DOMINANCE) {
        weight = votes.x;
        signed_direction = votes.y < 0.0f ? -1.0f : 1.0f;
        support = votes.z;
    }
}

void SbsRuleFinalizeInputPlan() {
    [unroll]
    for (uint field = 0u;
         field < SBS_RULE_SUMMARY_PLAN_FLOAT_COUNT;
         ++field) {
        RuleSummary[SbsRulePlanIndex(field)] = 0.0f;
    }

    float valid_cells = RuleSummary[SbsRuleReduceAggregateIndex(
        SBS_RULE_REDUCE_AGGREGATE_VALID_CELLS)];
    float required_invalid = RuleSummary[SbsRuleReduceAggregateIndex(
        SBS_RULE_REDUCE_AGGREGATE_REQUIRED_INVALID)];
    float depth_coverage = 0.0f;
    float depth_confidence = 0.0f;
    float roi_structural_weight = 0.0f;
    float roi_exposure_weight = 0.0f;
    float fresh_activity_mass = 0.0f;
    float fresh_activity_inside_previous_target = 0.0f;
    float3 scroll_votes_y = 0.0f.xxx;
    float3 scroll_votes_x = 0.0f.xxx;

    const bool event_preaggregated =
        RuleSummary[SBS_RULE_EVENT_SOURCE_MARKER] ==
            SBS_RULE_EVENT_SOURCE_PREAGGREGATED;
    if (event_preaggregated) {
        [unroll]
        for (uint group = 0u;
             group < SBS_RULE_EVENT_GROUP_COUNT;
             ++group) {
            const float4 event_sum = float4(
                RuleSummary[SbsRuleEventGroupIndex(
                    group,
                    SBS_RULE_EVENT_GROUP_ROI_CELLS)],
                RuleSummary[SbsRuleEventGroupIndex(
                    group,
                    SBS_RULE_EVENT_GROUP_DEPTH_WEIGHT)],
                RuleSummary[SbsRuleEventGroupIndex(
                    group,
                    SBS_RULE_EVENT_GROUP_ROI_STRUCTURAL_WEIGHT)],
                RuleSummary[SbsRuleEventGroupIndex(
                    group,
                    SBS_RULE_EVENT_GROUP_ROI_EXPOSURE_WEIGHT)]);
            const float3 group_scroll_y = float3(
                RuleSummary[SbsRuleEventGroupIndex(
                    group,
                    SBS_RULE_EVENT_GROUP_SCROLL_Y_WEIGHT)],
                RuleSummary[SbsRuleEventGroupIndex(
                    group,
                    SBS_RULE_EVENT_GROUP_SCROLL_Y_SIGNED)],
                RuleSummary[SbsRuleEventGroupIndex(
                    group,
                    SBS_RULE_EVENT_GROUP_SCROLL_Y_SUPPORT)]);
            const float3 group_scroll_x = float3(
                RuleSummary[SbsRuleEventGroupIndex(
                    group,
                    SBS_RULE_EVENT_GROUP_SCROLL_X_WEIGHT)],
                RuleSummary[SbsRuleEventGroupIndex(
                    group,
                    SBS_RULE_EVENT_GROUP_SCROLL_X_SIGNED)],
                RuleSummary[SbsRuleEventGroupIndex(
                    group,
                    SBS_RULE_EVENT_GROUP_SCROLL_X_SUPPORT)]);
            const float2 fresh_activity = float2(
                RuleSummary[SbsRuleEventGroupIndex(
                    group,
                    SBS_RULE_EVENT_GROUP_FRESH_ACTIVITY_MASS)],
                RuleSummary[SbsRuleEventGroupIndex(
                    group,
                    SBS_RULE_EVENT_GROUP_FRESH_ACTIVITY_INSIDE_PREVIOUS_TARGET)]);
            if (
                !all(isfinite(event_sum)) ||
                !all(isfinite(group_scroll_y)) ||
                !all(isfinite(group_scroll_x)) ||
                !all(isfinite(fresh_activity)) ||
                event_sum.x < 0.0f ||
                group_scroll_y.x < 0.0f ||
                group_scroll_y.z < 0.0f ||
                group_scroll_y.z > group_scroll_y.x ||
                abs(group_scroll_y.y) > group_scroll_y.z ||
                group_scroll_x.x < 0.0f ||
                group_scroll_x.z < 0.0f ||
                group_scroll_x.z > group_scroll_x.x ||
                abs(group_scroll_x.y) > group_scroll_x.z ||
                any(fresh_activity < 0.0f.xx) ||
                fresh_activity.y > fresh_activity.x
            ) {
                required_invalid += 1.0f;
                continue;
            }
            depth_coverage += event_sum.x;
            depth_confidence += event_sum.y;
            roi_structural_weight += event_sum.z;
            roi_exposure_weight += event_sum.w;
            scroll_votes_y += group_scroll_y;
            scroll_votes_x += group_scroll_x;
            fresh_activity_mass += fresh_activity.x;
            fresh_activity_inside_previous_target += fresh_activity.y;
        }
    } else {
        required_invalid += 1.0f;
    }

    float global_scroll_weight;
    float global_scroll_signed;
    float global_scroll_support;
    SbsRuleQualifyGlobalScrollVotes(
        scroll_votes_y,
        global_scroll_weight,
        global_scroll_signed,
        global_scroll_support);
    float global_horizontal_scroll_weight;
    float global_horizontal_scroll_signed;
    float global_horizontal_scroll_support;
    SbsRuleQualifyGlobalScrollVotes(
        scroll_votes_x,
        global_horizontal_scroll_weight,
        global_horizontal_scroll_signed,
        global_horizontal_scroll_support);

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
    uint2 viewport_origin;
    uint2 viewport_size;
    SceneViewport(
        SBS_SCENE_ANALYSIS_CANVAS_SIZE,
        viewport_origin,
        viewport_size);
    const float expected_viewport_cells =
        (float)(viewport_size.x * viewport_size.y);
    const bool input_valid =
        SceneAbiMatches() &&
        required_invalid <= 0.0f &&
        expected_viewport_cells > 0.0f &&
        valid_cells >= 0.99f * expected_viewport_cells &&
        meta_finite &&
        adaptive_finite &&
        meta_reserved_zero;

    RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_INPUT_VALID)] =
        input_valid ? 1.0f : 0.0f;
    RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_VALID_CELLS)] =
        valid_cells;
    RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_DEPTH_COVERAGE)] =
        depth_coverage;
    RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_DEPTH_CONFIDENCE)] =
        depth_confidence;
    // Local-scroll fields remain allocated for ABI stability but are no
    // longer produced or consumed.
    RuleSummary[SbsRulePlanIndex(
        SBS_RULE_PLAN_GLOBAL_SCROLL_WEIGHT)] =
        global_scroll_weight;
    RuleSummary[SbsRulePlanIndex(
        SBS_RULE_PLAN_GLOBAL_SCROLL_SIGNED)] =
        global_scroll_signed;
    RuleSummary[SbsRulePlanIndex(
        SBS_RULE_PLAN_GLOBAL_SCROLL_SUPPORT)] =
        global_scroll_support;
    RuleSummary[SbsRulePlanIndex(
        SBS_RULE_PLAN_GLOBAL_HORIZONTAL_SCROLL_WEIGHT)] =
        global_horizontal_scroll_weight;
    RuleSummary[SbsRulePlanIndex(
        SBS_RULE_PLAN_GLOBAL_HORIZONTAL_SCROLL_SIGNED)] =
        global_horizontal_scroll_signed;
    RuleSummary[SbsRulePlanIndex(
        SBS_RULE_PLAN_GLOBAL_HORIZONTAL_SCROLL_SUPPORT)] =
        global_horizontal_scroll_support;
    RuleSummary[SbsRulePlanIndex(
        SBS_RULE_PLAN_ROI_STRUCTURAL_WEIGHT)] =
        roi_structural_weight;
    RuleSummary[SbsRulePlanIndex(
        SBS_RULE_PLAN_ROI_EXPOSURE_WEIGHT)] =
        roi_exposure_weight;
    RuleSummary[SbsRulePlanIndex(
        SBS_RULE_PLAN_FRESH_ACTIVITY_MASS)] =
        fresh_activity_mass;
    RuleSummary[SbsRulePlanIndex(
        SBS_RULE_PLAN_FRESH_ACTIVITY_INSIDE_PREVIOUS_TARGET)] =
        fresh_activity_inside_previous_target;
}

void SbsRuleFinalizeTemporalProbe() {
    [unroll]
    for (uint field = 0u;
         field < SBS_RULE_TEMPORAL_PROBE_FLOAT_COUNT;
         ++field) {
        RuleSummary[SbsRuleTemporalProbeIndex(field)] = 0.0f;
    }
    if (
        RuleSummary[SbsRulePlanIndex(
            SBS_RULE_PLAN_INPUT_VALID)] <= 0.5f
        || !SbsRulePreviousTargetsUsable()
    ) {
        return;
    }

    uint2 viewport_origin;
    uint2 viewport_size;
    SceneViewport(
        SBS_SCENE_ANALYSIS_CANVAS_SIZE,
        viewport_origin,
        viewport_size);
    const uint2 viewport_end = viewport_origin + viewport_size;
    const uint first_x = SbsRuleCandidateMetaUint(
        SBS_RULE_ATTRIBUTION_GLOBAL_X_FIRST);
    const uint end_x = SbsRuleCandidateMetaUint(
        SBS_RULE_ATTRIBUTION_GLOBAL_X_END);
    const uint first_y = SbsRuleCandidateMetaUint(
        SBS_RULE_ATTRIBUTION_GLOBAL_Y_FIRST);
    const uint end_y = SbsRuleCandidateMetaUint(
        SBS_RULE_ATTRIBUTION_GLOBAL_Y_END);
    const bool x_valid =
        SbsRuleCandidateMetaFlag(
            SBS_RULE_ATTRIBUTION_GLOBAL_X_VALID) &&
        first_x >= viewport_origin.x &&
        end_x <= viewport_end.x &&
        end_x > first_x;
    const bool y_valid =
        SbsRuleCandidateMetaFlag(
            SBS_RULE_ATTRIBUTION_GLOBAL_Y_VALID) &&
        first_y >= viewport_origin.y &&
        end_y <= viewport_end.y &&
        end_y > first_y;
    const bool unique_axes =
        x_valid &&
        y_valid &&
        !SbsRuleCandidateMetaFlag(
            SBS_RULE_ATTRIBUTION_GLOBAL_X_AMBIGUOUS) &&
        !SbsRuleCandidateMetaFlag(
            SBS_RULE_ATTRIBUTION_GLOBAL_Y_AMBIGUOUS);
    const uint seed_first_x = SbsRuleCandidateMetaUint(
        SBS_RULE_ATTRIBUTION_SEED_X_FIRST);
    const uint seed_end_x = SbsRuleCandidateMetaUint(
        SBS_RULE_ATTRIBUTION_SEED_X_END);
    const uint seed_first_y = SbsRuleCandidateMetaUint(
        SBS_RULE_ATTRIBUTION_SEED_Y_FIRST);
    const uint seed_end_y = SbsRuleCandidateMetaUint(
        SBS_RULE_ATTRIBUTION_SEED_Y_END);
    const bool seed_valid =
        SbsRuleCandidateMetaFlag(
            SBS_RULE_ATTRIBUTION_SEED_VALID) &&
        seed_first_x >= viewport_origin.x &&
        seed_end_x <= viewport_end.x &&
        seed_end_x > seed_first_x &&
        seed_first_y >= viewport_origin.y &&
        seed_end_y <= viewport_end.y &&
        seed_end_y > seed_first_y;
    // Conditional photographic support can connect an interior temporal
    // seed to unrelated browser content all the way at a viewport edge.
    // Reaching that edge is not evidence that the player owns it. Preserve
    // a true full-viewport result (which is canonical identity), but make
    // every partial envelope fail closed when expansion alone claimed an
    // edge that the temporal seed never observed.
    const bool expanded_is_full_viewport =
        x_valid &&
        y_valid &&
        first_x == viewport_origin.x &&
        end_x == viewport_end.x &&
        first_y == viewport_origin.y &&
        end_y == viewport_end.y;
    const bool expansion_claims_unowned_viewport_edge =
        seed_valid &&
        ((first_x == viewport_origin.x &&
          seed_first_x > viewport_origin.x) ||
         (end_x == viewport_end.x &&
          seed_end_x < viewport_end.x) ||
         (first_y == viewport_origin.y &&
          seed_first_y > viewport_origin.y) ||
         (end_y == viewport_end.y &&
          seed_end_y < viewport_end.y));
    const bool expanded_envelope_safe =
        expanded_is_full_viewport ||
        !expansion_claims_unowned_viewport_edge;

    float total_temporal_mass = 0.0f;
    [loop]
    for (uint x = viewport_origin.x; x < viewport_end.x; ++x) {
        total_temporal_mass += SbsRuleTemporalColumnMass(x);
    }
    const float selected_temporal_mass = max(
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_GLOBAL_RECT_MASS)],
        0.0f);
    const float selected_fresh_mass = max(
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_GLOBAL_RECT_FRESH)],
        0.0f);
    const float rectangle_coherence =
        RuleSummary[SbsRuleAttributionMetaIndex(
            SBS_RULE_ATTRIBUTION_GLOBAL_RECT_COHERENCE)];
    const bool rectangle_coherence_valid =
        isfinite(rectangle_coherence) &&
        rectangle_coherence >= 0.0f &&
        rectangle_coherence <= 1.0f;
    const float rectangle_cells =
        unique_axes && seed_valid ?
            (float)(
                (seed_end_x - seed_first_x) *
                (seed_end_y - seed_first_y)) :
            0.0f;
    const float viewport_cells =
        max((float)(viewport_size.x * viewport_size.y), 1.0f);
    const float event_fill =
        rectangle_cells > 0.0f ?
            selected_temporal_mass / rectangle_cells :
            0.0f;
    const float event_containment =
        total_temporal_mass > 0.0f ?
            selected_temporal_mass / total_temporal_mass :
            0.0f;
    const float fresh_activity_mass = max(
        RuleSummary[SbsRulePlanIndex(
            SBS_RULE_PLAN_FRESH_ACTIVITY_MASS)],
        0.0f);
    const float material_activity_threshold =
        SBS_RULE_TEMPORAL_MATERIAL_MIN_VIEWPORT_FRACTION *
        viewport_cells;
    const bool fresh_material_activity =
        fresh_activity_mass >= material_activity_threshold;
    // A stable accumulated rectangle may coexist with a transient animated
    // sidebar or ad. Require the ordinary frame-level material floor inside
    // the candidate itself; outside activity cannot help a weak candidate
    // pass. Accumulated temporal containment remains the competitor test.
    const bool candidate_fresh_material_activity =
        selected_fresh_mass >= material_activity_threshold;
    const bool event_valid =
        unique_axes &&
        seed_valid &&
        expanded_envelope_safe &&
        rectangle_cells / viewport_cells >=
            SBS_RULE_TEMPORAL_EVENT_MIN_AREA &&
        event_fill >= SBS_RULE_TEMPORAL_EVENT_MIN_FILL &&
        rectangle_coherence_valid &&
        rectangle_coherence >=
            SBS_RULE_TEMPORAL_MIN_RECT_COHERENCE &&
        event_containment >= SBS_RULE_CANDIDATE_MIN_DOMINANCE &&
        candidate_fresh_material_activity;

    RuleSummary[SbsRuleTemporalProbeIndex(
        SBS_RULE_TEMPORAL_PROBE_EVENT_FILL)] =
        saturate(event_fill);
    RuleSummary[SbsRuleTemporalProbeIndex(
        SBS_RULE_TEMPORAL_PROBE_EVENT_CONTAINMENT)] =
        saturate(event_containment);
    // The event statistics are measured on the temporal seed. The columns
    // pass then expands that seed through media conditioned on the opposite
    // seed span, recovering static player borders without letting unrelated
    // activity on another axis stretch the proposal.
    if (x_valid && y_valid) {
        RuleSummary[SbsRuleTemporalProbeIndex(
            SBS_RULE_TEMPORAL_PROBE_EVENT_X0)] =
            (float)(first_x - viewport_origin.x) /
                max((float)viewport_size.x, 1.0f);
        RuleSummary[SbsRuleTemporalProbeIndex(
            SBS_RULE_TEMPORAL_PROBE_EVENT_Y0)] =
            (float)(first_y - viewport_origin.y) /
                max((float)viewport_size.y, 1.0f);
        RuleSummary[SbsRuleTemporalProbeIndex(
            SBS_RULE_TEMPORAL_PROBE_EVENT_X1)] =
            (float)(end_x - viewport_origin.x) /
                max((float)viewport_size.x, 1.0f);
        RuleSummary[SbsRuleTemporalProbeIndex(
            SBS_RULE_TEMPORAL_PROBE_EVENT_Y1)] =
            (float)(end_y - viewport_origin.y) /
                max((float)viewport_size.y, 1.0f);
    }
    RuleSummary[SbsRuleTemporalProbeIndex(
        SBS_RULE_TEMPORAL_PROBE_EVENT_VALID)] =
        event_valid ? 1.0f : 0.0f;
    RuleSummary[SbsRuleTemporalProbeIndex(
        SBS_RULE_TEMPORAL_PROBE_UNRESOLVED)] =
        fresh_material_activity && !event_valid ? 1.0f : 0.0f;
    RuleSummary[SbsRuleTemporalProbeIndex(
        SBS_RULE_TEMPORAL_PROBE_AMBIGUOUS)] =
        fresh_material_activity && !event_valid ? 1.0f : 0.0f;

    float4 previous_video_bounds;
    const bool previous_video_target =
        SbsRulePreviousVideoTarget(previous_video_bounds);
    const float previous_target_containment =
        fresh_activity_mass > 0.0f ?
            max(
                RuleSummary[SbsRulePlanIndex(
                    SBS_RULE_PLAN_FRESH_ACTIVITY_INSIDE_PREVIOUS_TARGET)],
                0.0f) /
                fresh_activity_mass :
            0.0f;
    const bool hold_observed =
        previous_video_target &&
        fresh_material_activity;
    const bool hold_valid =
        previous_video_target &&
        hold_observed &&
        previous_target_containment >=
            SBS_RULE_CANDIDATE_MIN_DOMINANCE;
    RuleSummary[SbsRuleTemporalProbeIndex(
        SBS_RULE_TEMPORAL_PROBE_HOLD_OBSERVED)] =
        hold_observed ? 1.0f : 0.0f;
    RuleSummary[SbsRuleTemporalProbeIndex(
        SBS_RULE_TEMPORAL_PROBE_HOLD_VALID)] =
        hold_valid ? 1.0f : 0.0f;
    RuleSummary[SbsRuleTemporalProbeIndex(
        SBS_RULE_TEMPORAL_PROBE_HOLD_CONTAINMENT)] =
        saturate(previous_target_containment);
}

float SbsRuleMediaAxisMass(uint axis, uint coordinate) {
    return axis == 0u ?
        SbsRuleMediaColumnMass(coordinate) :
        SbsRuleMediaRowMass(coordinate);
}

bool SbsRuleFindRetainedMediaBounds(
    uint axis,
    uint first,
    uint end,
    float total_mass,
    out uint retained_first,
    out uint retained_end)
{
    retained_first = first;
    retained_end = first;
    if (
        end <= first ||
        !isfinite(total_mass) ||
        total_mass <= 0.0f
    ) {
        return false;
    }

    float peak_mass = 0.0f;
    [loop]
    for (uint coordinate = first; coordinate < end; ++coordinate) {
        peak_mass = max(
            peak_mass,
            SbsRuleMediaAxisMass(axis, coordinate));
    }
    const float trim_mass =
        SBS_RULE_MEDIA_AXIS_TRIM_MASS * total_mass;
    float cumulative = 0.0f;
    bool found_first = false;
    [loop]
    for (uint coordinate = first; coordinate < end; ++coordinate) {
        const float mass =
            SbsRuleMediaAxisMass(axis, coordinate);
        if (cumulative + mass > trim_mass) {
            retained_first = coordinate;
            found_first = true;
            break;
        }
        cumulative += mass;
    }
    cumulative = 0.0f;
    bool found_end = false;
    [loop]
    for (int coordinate = (int)end - 1;
         coordinate >= (int)first;
         --coordinate) {
        const float mass =
            SbsRuleMediaAxisMass(axis, (uint)coordinate);
        if (cumulative + mass > trim_mass) {
            retained_end = (uint)coordinate + 1u;
            found_end = true;
            break;
        }
        cumulative += mass;
    }
    if (!found_first || !found_end || retained_end <= retained_first) {
        return false;
    }

    const float supported_edge_mass =
        SBS_RULE_MEDIA_AXIS_EDGE_SUPPORT_FRACTION * peak_mass;
    [loop]
    while (
        retained_first > first &&
        SbsRuleMediaAxisMass(axis, retained_first - 1u) >=
            supported_edge_mass
    ) {
        retained_first -= 1u;
    }
    [loop]
    while (
        retained_end < end &&
        SbsRuleMediaAxisMass(axis, retained_end) >=
            supported_edge_mass
    ) {
        retained_end += 1u;
    }
    return retained_end > retained_first;
}

void SbsRuleBuildContentCandidate(
    out float score,
    out float4 bounds,
    out float coverage)
{
    score = 0.0f;
    bounds = 0.0f.xxxx;
    coverage = 0.0f;
    if (
        RuleSummary[SbsRulePlanIndex(
            SBS_RULE_PLAN_INPUT_VALID)] <= 0.5f
    ) {
        return;
    }
    float4 previous_committed_video_bounds;
    if (SbsRulePreviousCommittedVideoTarget(
            previous_committed_video_bounds)) {
        return;
    }

    uint2 viewport_origin;
    uint2 viewport_size;
    SceneViewport(
        SBS_SCENE_ANALYSIS_CANVAS_SIZE,
        viewport_origin,
        viewport_size);
    const uint2 viewport_end = viewport_origin + viewport_size;
    const float viewport_cells =
        max((float)(viewport_size.x * viewport_size.y), 1.0f);
    float column_media_mass = 0.0f;
    float row_media_mass = 0.0f;
    [loop]
    for (uint x = viewport_origin.x; x < viewport_end.x; ++x) {
        column_media_mass += SbsRuleMediaColumnMass(x);
    }
    [loop]
    for (uint y = viewport_origin.y; y < viewport_end.y; ++y) {
        row_media_mass += SbsRuleMediaRowMass(y);
    }
    uint first_x;
    uint end_x;
    uint first_y;
    uint end_y;
    const bool x_valid = SbsRuleFindRetainedMediaBounds(
        0u,
        viewport_origin.x,
        viewport_end.x,
        column_media_mass,
        first_x,
        end_x);
    const bool y_valid = SbsRuleFindRetainedMediaBounds(
        1u,
        viewport_origin.y,
        viewport_end.y,
        row_media_mass,
        first_y,
        end_y);
    if (!x_valid || !y_valid) {
        return;
    }

    bounds = float4(
        (float)(first_x - viewport_origin.x) /
            max((float)viewport_size.x, 1.0f),
        (float)(first_y - viewport_origin.y) /
            max((float)viewport_size.y, 1.0f),
        (float)(end_x - viewport_origin.x) /
            max((float)viewport_size.x, 1.0f),
        (float)(end_y - viewport_origin.y) /
            max((float)viewport_size.y, 1.0f));
    const float2 halo =
        1.0f / max(float2(viewport_size), 1.0f.xx);
    bounds = clamp(
        bounds + float4(-halo, halo),
        0.0f,
        1.0f);

    float retained_column_mass = 0.0f;
    float retained_row_mass = 0.0f;
    [loop]
    for (uint x = first_x; x < end_x; ++x) {
        retained_column_mass += SbsRuleMediaColumnMass(x);
    }
    [loop]
    for (uint y = first_y; y < end_y; ++y) {
        retained_row_mass += SbsRuleMediaRowMass(y);
    }
    const float total_media_mass =
        max(column_media_mass, row_media_mass);
    // Inclusion/exclusion is a conservative lower bound on media mass in the
    // rectangular hull. Cross-shaped or detached projections therefore
    // abstain instead of manufacturing a content target.
    const float retained_media_mass = max(
        retained_column_mass +
            retained_row_mass -
            total_media_mass,
        0.0f);
    const float envelope_cells = max(
        BoundsArea(bounds) * viewport_cells,
        1.0f);
    coverage = ValidBounds(bounds) ?
        saturate(retained_media_mass / envelope_cells) :
        0.0f;
    const float viewport_media_fraction =
        retained_media_mass / viewport_cells;
    score =
        ValidBounds(bounds) &&
        viewport_media_fraction >=
            SBS_RULE_MEDIA_MIN_VIEWPORT_MASS &&
        coverage >= SBS_RULE_MEDIA_MIN_HULL_DENSITY ?
            viewport_media_fraction :
            0.0f;
}

#endif
