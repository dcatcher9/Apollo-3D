#ifndef SBS_SCENE_RULES_POLICY_ONLY
StructuredBuffer<float4> AdaptiveState : register(t4);
StructuredBuffer<float> Meta : register(t5);
StructuredBuffer<uint4> PreviousRuleState : register(t6);
RWStructuredBuffer<float> RuleSummary : register(u0);
RWStructuredBuffer<float> EvidenceGlobal : register(u1);

#include "include/sbs_scene_controller_constants.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"
#include "include/sbs_scene_rule_state.hlsl"
#include "include/sbs_scene_rules_summary.hlsl"

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
#endif

void SbsRuleEmitGlobalEvidence() {
    [unroll]
    for (uint word = 0u;
         word < SBS_SCENE_GLOBAL_OUT_WORD_COUNT;
         ++word) {
        EvidenceGlobal[word] = 0.0f;
    }

    SbsRuleFinalizeInputPlan();
    SbsRuleFinalizeTemporalProbe();

    const bool input_valid =
        RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_INPUT_VALID)] > 0.5f;
    if (!input_valid) {
        EvidenceGlobal[
            SBS_SCENE_GLOBAL_OUT_ZERO_PLANE_NEUTRAL_ABSTAIN] =
            SceneLogit(0.98f);
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_OOD] = 1.0f;
        EvidenceGlobal[
            SBS_SCENE_GLOBAL_OUT_BACKEND_OUTPUT_VALID] = 0.0f;
        return;
    }

    float best_score = 0.0f;
    float4 best_bounds = float4(0.0f, 0.0f, 1.0f, 1.0f);
    uint best_layout = SBS_SCENE_LAYOUT_DECISION_NO_TARGET;
    float best_coverage = 0.0f;
    const bool previous_targets_usable =
        SbsRulePreviousTargetsUsable();
    const bool previous_counter_state_valid =
        SbsRulePreviousCounterStateValid();
    const bool temporal_event_valid =
        RuleSummary[SbsRuleTemporalProbeIndex(
            SBS_RULE_TEMPORAL_PROBE_EVENT_VALID)] > 0.5f;
    const bool temporal_ambiguous =
        RuleSummary[SbsRuleTemporalProbeIndex(
            SBS_RULE_TEMPORAL_PROBE_AMBIGUOUS)] > 0.5f;
    const bool temporal_hold_valid =
        RuleSummary[SbsRuleTemporalProbeIndex(
            SBS_RULE_TEMPORAL_PROBE_HOLD_VALID)] > 0.5f;
    const bool temporal_hold_observed =
        RuleSummary[SbsRuleTemporalProbeIndex(
            SBS_RULE_TEMPORAL_PROBE_HOLD_OBSERVED)] > 0.5f;
    const float4 raw_temporal_event_bounds = float4(
        RuleSummary[SbsRuleTemporalProbeIndex(
            SBS_RULE_TEMPORAL_PROBE_EVENT_X0)],
        RuleSummary[SbsRuleTemporalProbeIndex(
            SBS_RULE_TEMPORAL_PROBE_EVENT_Y0)],
        RuleSummary[SbsRuleTemporalProbeIndex(
            SBS_RULE_TEMPORAL_PROBE_EVENT_X1)],
        RuleSummary[SbsRuleTemporalProbeIndex(
            SBS_RULE_TEMPORAL_PROBE_EVENT_Y1)]);
    float4 temporal_event_bounds = 0.0f.xxxx;
    const bool temporal_event_bounds_valid =
        temporal_event_valid &&
        SbsRuleValidateUnitBounds(
            raw_temporal_event_bounds,
            temporal_event_bounds);
    float4 previous_temporal_probe_bounds;
    const bool previous_temporal_probe =
        SbsRulePreviousProvisionalVideoTarget(
            previous_temporal_probe_bounds);
    float4 previous_committed_video_bounds;
    const bool previous_committed_video =
        SbsRulePreviousCommittedVideoTarget(
            previous_committed_video_bounds);
    float4 previous_video_challenger_bounds;
    const bool previous_video_challenger =
        SbsRulePreviousVideoChallengerTarget(
            previous_video_challenger_bounds);
    const float temporal_event_area =
        temporal_event_bounds_valid ?
            BoundsArea(temporal_event_bounds) :
            0.0f;
    const bool video_relocation_challenge =
        previous_committed_video &&
        !previous_video_challenger &&
        temporal_event_bounds_valid &&
        temporal_event_area >
            SBS_RULE_TEMPORAL_COMPONENT_AMBIGUITY_AREA_FRACTION *
                BoundsArea(previous_committed_video_bounds) &&
        SceneIntersectionOverMinArea(
            temporal_event_bounds,
            previous_committed_video_bounds) <
                SBS_RULE_TEMPORAL_COMPONENT_AMBIGUITY_AREA_FRACTION &&
        SceneIou(
            temporal_event_bounds,
            previous_committed_video_bounds) <
                SBS_SCENE_ROI_GEOMETRY_IDENTITY_MIN_IOU;
    const bool freeze_temporal_target =
        previous_video_challenger ||
        (previous_committed_video &&
         !video_relocation_challenge);
    // A challenger of committed geometry is frozen: later activity may
    // corroborate it through containment, but cannot drag it toward a moving
    // subject or animated sidebar.
    const bool temporal_probe_authoritative =
        (previous_video_challenger ?
            temporal_hold_valid :
         video_relocation_challenge ?
            true :
         previous_committed_video ?
            temporal_hold_valid :
             (temporal_event_bounds_valid ||
              (previous_temporal_probe && temporal_hold_valid)));
    float content_score;
    float4 content_bounds;
    float content_coverage;
    SbsRuleBuildContentCandidate(
        content_score,
        content_bounds,
        content_coverage);
    const float temporal_candidate_area =
        temporal_event_bounds_valid ?
            BoundsArea(temporal_event_bounds) :
            0.0f;
    const float content_candidate_area =
        ValidBounds(content_bounds) ?
            BoundsArea(content_bounds) :
            0.0f;
    // Motion can prove that a region is live without proving the exact
    // boundary of a larger mixed-media surface. When one simple static hull
    // safely contains a much smaller first temporal seed, publish the whole
    // hull as CONTENT rather than mislabeling or cropping to the moving inset.
    const bool content_union_authoritative =
        temporal_probe_authoritative &&
        temporal_event_bounds_valid &&
        !previous_temporal_probe &&
        !previous_video_challenger &&
        !previous_committed_video &&
        content_score > 0.0f &&
        content_candidate_area < SBS_RULE_IDENTITY_MIN_AREA &&
        content_candidate_area > 0.0f &&
        temporal_candidate_area <=
            SBS_RULE_CANDIDATE_MIN_DOMINANCE *
                content_candidate_area &&
        SceneIntersectionOverMinArea(
            temporal_event_bounds,
            content_bounds) >= SBS_RULE_CANDIDATE_MIN_DOMINANCE;

    if (temporal_probe_authoritative && !content_union_authoritative) {
        const float4 temporal_bounds =
            previous_video_challenger ?
                previous_video_challenger_bounds :
            video_relocation_challenge ?
                temporal_event_bounds :
            previous_committed_video ?
                previous_committed_video_bounds :
            temporal_event_bounds_valid ?
                temporal_event_bounds :
                previous_temporal_probe_bounds;
        const float temporal_coverage =
            !freeze_temporal_target && temporal_event_bounds_valid ?
                RuleSummary[SbsRuleTemporalProbeIndex(
                    SBS_RULE_TEMPORAL_PROBE_EVENT_FILL)] :
                RuleSummary[SbsRuleTemporalProbeIndex(
                    SBS_RULE_TEMPORAL_PROBE_HOLD_CONTAINMENT)];
        if (ValidBounds(temporal_bounds)) {
            best_score = 1.0f;
            best_bounds = temporal_bounds;
            best_layout =
                SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO;
            best_coverage = temporal_coverage;
        }
    } else if (
        content_union_authoritative ||
        (
            !temporal_ambiguous &&
            !previous_temporal_probe &&
            !previous_video_challenger &&
            !(previous_committed_video && !temporal_hold_observed) &&
            content_score > 0.0f &&
            ValidBounds(content_bounds)
        )
    ) {
        best_score = content_score;
        best_bounds = content_bounds;
        best_layout =
            SBS_SCENE_LAYOUT_DECISION_CONTENT_COLLAGE;
        best_coverage = content_coverage;
    }

    RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_BEST_SCORE)] =
        best_score;
    RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_SECOND_SCORE)] =
        0.0f;
    RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_WINNER_MARGIN)] =
        0.0f;
    RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_CANDIDATE_COVERAGE)] =
        best_coverage;
    const bool ambiguous =
        temporal_ambiguous &&
        best_score <= 0.0f;
    const bool insufficient =
        best_score <= 0.0f;
    const float roi_confidence =
        ambiguous || insufficient ?
            0.0f :
            saturate(0.65f + 0.35f * best_coverage);
    const float mask_confidence =
        insufficient ? 0.0f : saturate(best_coverage);
    const float ood =
        ambiguous ? 0.75f :
        insufficient ? 0.60f :
        saturate(1.0f - roi_confidence);
    // A crop covering at least 75% of the viewport improves linear sampling
    // density by less than 16%. That is too little to justify changing model
    // shape and convergence geometry, regardless of which edge supplied the
    // small omission.
    const bool identity_target =
        BoundsArea(best_bounds) >= SBS_RULE_IDENTITY_MIN_AREA;
    if (identity_target && !ambiguous && !insufficient) {
        best_layout = SBS_SCENE_LAYOUT_DECISION_IDENTITY_FULLSCREEN;
        best_bounds = float4(0.0f, 0.0f, 1.0f, 1.0f);
    }

    const float global_scroll_weight =
        RuleSummary[SbsRulePlanIndex(
            SBS_RULE_PLAN_GLOBAL_SCROLL_WEIGHT)];
    const float global_scroll_signed =
        RuleSummary[SbsRulePlanIndex(
            SBS_RULE_PLAN_GLOBAL_SCROLL_SIGNED)];
    const float global_scroll_support_cells =
        RuleSummary[SbsRulePlanIndex(
            SBS_RULE_PLAN_GLOBAL_SCROLL_SUPPORT)];
    const float global_horizontal_scroll_weight =
        RuleSummary[SbsRulePlanIndex(
            SBS_RULE_PLAN_GLOBAL_HORIZONTAL_SCROLL_WEIGHT)];
    const float global_horizontal_scroll_signed =
        RuleSummary[SbsRulePlanIndex(
            SBS_RULE_PLAN_GLOBAL_HORIZONTAL_SCROLL_SIGNED)];
    const float global_horizontal_scroll_support_cells =
        RuleSummary[SbsRulePlanIndex(
            SBS_RULE_PLAN_GLOBAL_HORIZONTAL_SCROLL_SUPPORT)];
    // The evidence pass already excludes a retained player and counts only
    // textured cells with a complete seven-position comparison. Normalize by
    // that exact observable population; sparse browser pages must not be
    // penalized for blank or unobservable viewport cells.
    const float global_scroll_fraction =
        global_scroll_weight > 0.0f ?
            global_scroll_support_cells / global_scroll_weight :
            0.0f;
    const float global_horizontal_scroll_fraction =
        global_horizontal_scroll_weight > 0.0f ?
            global_horizontal_scroll_support_cells /
                global_horizontal_scroll_weight :
            0.0f;
    const float global_scroll_confidence =
        global_scroll_fraction >=
            SBS_RULE_SCROLL_MIN_SUPPORT_FRACTION_OF_OBSERVABLE &&
        global_scroll_weight > 0.0f ?
            1.0f :
            0.0f;
    const float global_horizontal_scroll_confidence =
        global_horizontal_scroll_fraction >=
            SBS_RULE_SCROLL_MIN_SUPPORT_FRACTION_OF_OBSERVABLE &&
        global_horizontal_scroll_weight > 0.0f ?
            1.0f :
            0.0f;
    const float scroll_confidence_y =
        global_scroll_confidence;
    float scroll_y =
        global_scroll_confidence > 0.0f &&
        global_scroll_weight > 1e-6f ?
            clamp(global_scroll_signed, -1.0f, 1.0f) :
            0.0f;
    const float scroll_confidence_x =
        global_horizontal_scroll_confidence;
    float scroll_x =
        global_horizontal_scroll_confidence > 0.0f &&
        global_horizontal_scroll_weight > 1e-6f ?
            clamp(global_horizontal_scroll_signed, -1.0f, 1.0f) :
            0.0f;
    if (scroll_confidence_x <= 0.0f) {
        scroll_x = 0.0f;
    }
    if (scroll_confidence_y <= 0.0f) {
        scroll_y = 0.0f;
    }
    const float scroll_vector_length = length(float2(
        scroll_x,
        scroll_y));
    if (scroll_vector_length > 1.0f) {
        const float reciprocal_length = rcp(scroll_vector_length);
        scroll_x *= reciprocal_length;
        scroll_y *= reciprocal_length;
    }
    const float scroll_confidence = max(
        scroll_confidence_x,
        scroll_confidence_y);

    // Ambiguity is an explicit public decision, not a no-target subtype. If no-target retained
    // its ordinary 1.0 fallback probability here, its logit would always beat the 0.90
    // ambiguity lane and the resolver could never publish AMBIGUOUS.
    const float no_target_probability =
        ambiguous ? 0.05f : saturate(1.0f - roi_confidence);
    const float video_probability =
        best_layout == SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO ?
            roi_confidence :
            0.02f;
    const float content_probability =
        best_layout == SBS_SCENE_LAYOUT_DECISION_CONTENT_COLLAGE ?
            roi_confidence :
            0.02f;
    const float identity_probability =
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

    const float4 adaptive_flags =
        AdaptiveState[SBS_STATE_VECTOR_RANGE_COLLAPSED];
    const float4 adaptive_analysis =
        AdaptiveState[SBS_STATE_VECTOR_ANALYSIS_FLAGS];
    const uint analysis_flags =
        (uint)SBS_STATE_ANALYSIS_FLAGS(adaptive_analysis);
    const uint effective_reset_flags = EffectiveResetFlags();
    const bool geometry_reset =
        (effective_reset_flags & SBS_SCENE_RESET_FLAGS_GEOMETRY) != 0u;
    const float roi_event_cells =
        RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_DEPTH_COVERAGE)];
    const float roi_event_depth_weight =
        RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_DEPTH_CONFIDENCE)];
    const float roi_event_depth_coverage =
        roi_event_cells > 0.0f ?
            saturate(roi_event_depth_weight / roi_event_cells) :
            0.0f;
    const float roi_structural_support =
        roi_event_depth_weight > 0.0f ?
            saturate(
                RuleSummary[
                    SbsRulePlanIndex(
                        SBS_RULE_PLAN_ROI_STRUCTURAL_WEIGHT)] /
                roi_event_depth_weight) :
            0.0f;
    const bool previous_roi_locked =
        previous_targets_usable &&
        (PreviousRuleStateUint(SBS_SCENE_RULE_STATE_WORD_STATE_FLAGS) &
         SBS_SCENE_STATE_FLAGS_ROI_LOCKED) != 0u;
    const bool roi_event_evidence_reliable =
        roi_event_cells >= 4.0f &&
        roi_event_depth_coverage >=
            SBS_RULE_ROI_EVENT_MIN_DEPTH_COVERAGE;
    const uint current_detector_cut_count = asuint(
        SBS_STATE_HARD_CUT_COUNT(
            AdaptiveState[SBS_STATE_VECTOR_HARD_CUT_COUNT]));
    const uint previous_detector_cut_count =
        PreviousRuleStateUint(
            SBS_SCENE_RULE_STATE_WORD_LAST_DETECTOR_CUT_COUNT);
    const bool detector_cut_count_advanced =
        previous_counter_state_valid &&
        current_detector_cut_count > previous_detector_cut_count;
    const bool detector_reset_supersedes =
        (effective_reset_flags &
            (SBS_SCENE_RESET_FLAGS_DEPTH_SHOT |
             SBS_SCENE_RESET_FLAGS_GEOMETRY |
             SBS_SCENE_RESET_FLAGS_BACKEND |
             SBS_SCENE_RESET_FLAGS_DISPLAY_OR_HDR)) != 0u;
    // Once exact counter state exists, the monotonic delta is authoritative. A stale/held
    // one-frame pulse must not retrigger after that count was already consumed.
    const bool detector_cut_signal =
        previous_counter_state_valid ?
            detector_cut_count_advanced :
            SBS_STATE_HARD_CUT_PULSE(adaptive_flags) > 0.5f;
    const bool adaptive_hard_cut =
        detector_cut_signal &&
        !detector_reset_supersedes;
    const bool adaptive_exposure =
        (analysis_flags & ANALYSIS_FLAG_EXPOSURE_LIKE) != 0u;
    const bool hard_cut =
        adaptive_hard_cut &&
        (!previous_roi_locked ||
         (roi_event_evidence_reliable &&
          roi_structural_support >=
              SBS_RULE_ROI_STRUCTURAL_CUT_MIN_SUPPORT));
    // Exposure is a frame-global photometric event and veto. ROI-local exposure support remains
    // diagnostic state, but cannot suppress a valid global detector result merely because the
    // locked content region was quiet or excluded.
    const bool exposure = adaptive_exposure;
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

    const float4 adaptive_profile =
        AdaptiveState[SBS_STATE_VECTOR_ADAPTIVE_POP_RATIO];
    const float4 adaptive_valid_depth =
        AdaptiveState[SBS_STATE_VECTOR_VALID_DEPTH_FRACTION];
    const float depth_coverage_cells =
        RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_DEPTH_COVERAGE)];
    const float history_depth_fraction =
        depth_coverage_cells > 0.0f ?
            saturate(
                RuleSummary[
                    SbsRulePlanIndex(SBS_RULE_PLAN_DEPTH_CONFIDENCE)] /
                depth_coverage_cells) :
            0.0f;
    const float valid_depth_fraction = min(
        saturate(SBS_STATE_VALID_DEPTH_FRACTION(adaptive_valid_depth)),
        history_depth_fraction);
    const float depth_reliability = smoothstep(
        0.05f,
        0.35f,
        valid_depth_fraction);
    const float resolved_strength = clamp(
        scene_pop_floor *
            max(SBS_STATE_ADAPTIVE_POP_RATIO(adaptive_profile), 1.0f),
        scene_pop_floor,
        max(scene_pop_ceiling, scene_pop_floor));
    [unroll]
    for (uint action = 0u; action < SBS_SCENE_POP_ACTION_COUNT; ++action) {
        const float action_strength = lerp(
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
        SceneLogit(0.02f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_ROI] = roi_confidence;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_MASK] = mask_confidence;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_EVENT] =
        lerp(0.25f, 0.90f, depth_reliability);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_POP] =
        lerp(0.25f, 0.90f, depth_reliability);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_ZERO_PLANE] =
        lerp(0.25f, 0.90f, depth_reliability);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_OOD] = ood;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_GLOBAL_SCROLL_X] = scroll_x;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_GLOBAL_SCROLL_Y] = scroll_y;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_GLOBAL_SCROLL_CONFIDENCE] =
        scroll_confidence;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_BACKEND_OUTPUT_VALID] = 1.0f;

    // Internal resolver scratch. The common resolver clears these words from the public output.
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_35] = best_bounds.x;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_36] = best_bounds.y;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_37] = best_bounds.z;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_38] = best_bounds.w;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_39] = 0.0f;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_40] =
        history_depth_fraction;
}

#ifndef SBS_SCENE_RULES_POLICY_ONLY
[numthreads(1, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    SbsRuleEmitGlobalEvidence();
}
#endif
