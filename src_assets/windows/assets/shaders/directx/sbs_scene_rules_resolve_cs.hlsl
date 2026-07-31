StructuredBuffer<float> EvidenceGlobal : register(t0);
StructuredBuffer<uint4> PreviousRuleState : register(t1);
StructuredBuffer<float> Meta : register(t2);
StructuredBuffer<float4> AdaptiveState : register(t3);
StructuredBuffer<float> RuleSummary : register(t4);
RWStructuredBuffer<float> GlobalOutput : register(u0);
RWStructuredBuffer<uint4> NextRuleState : register(u1);

#include "include/sbs_scene_controller_constants.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"
#include "include/sbs_scene_rule_state.hlsl"
#include "include/sbs_scene_rules_summary.hlsl"

float StateWord(float state[SBS_SCENE_RULE_STATE_WORD_COUNT], uint word) {
    return state[word];
}

void ClearAcquisitionTarget(
    inout float state[SBS_SCENE_RULE_STATE_WORD_COUNT])
{
    state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_ROI_X0] = 0.0f;
    state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_ROI_Y0] = 0.0f;
    state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_ROI_X1] = 1.0f;
    state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_ROI_Y1] = 1.0f;
    state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_SCORE] = 0.0f;
    state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_DWELL_S] = 0.0f;
    state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_LAYOUT] =
        (float)SBS_SCENE_LAYOUT_DECISION_NO_TARGET;
    state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_VALID] = 0.0f;
}

void ClearChallengerTarget(
    inout float state[SBS_SCENE_RULE_STATE_WORD_COUNT])
{
    state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_ROI_X0] = 0.0f;
    state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_ROI_Y0] = 0.0f;
    state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_ROI_X1] = 1.0f;
    state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_ROI_Y1] = 1.0f;
    state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_SCORE] = 0.0f;
    state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_DWELL_S] = 0.0f;
    state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_LAYOUT] =
        (float)SBS_SCENE_LAYOUT_DECISION_NO_TARGET;
    state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_VALID] = 0.0f;
}

#define PreviousStateUint SbsRulePreviousStateUint

float4 CandidateBounds() {
    return float4(
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_35],
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_36],
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_37],
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_38]);
}

uint EvidenceLayout() {
    uint decision = SBS_SCENE_LAYOUT_DECISION_NO_TARGET;
    float best =
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_LAYOUT_NO_TARGET];
    [unroll]
    for (uint index = 1u; index < 6u; ++index) {
        float value =
            EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_LAYOUT_NO_TARGET + index];
        if (value > best) {
            best = value;
            decision = index;
        }
    }
    return decision;
}

uint EvidenceEvent() {
    uint decision = SBS_SCENE_EVENT_DECISION_SAME_SHOT;
    float best =
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_EVENT_SAME_SHOT];
    [unroll]
    for (uint index = 1u; index < 6u; ++index) {
        float value =
            EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_EVENT_SAME_SHOT + index];
        if (value > best) {
            best = value;
            decision = index;
        }
    }
    return decision;
}

[numthreads(1, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    float state[SBS_SCENE_RULE_STATE_WORD_COUNT];
    uint effective_reset_flags = SbsRuleEffectiveResetFlags();
    uint previous_geometry_state_kind;
    float4 previous_committed_bounds;
    float previous_committed_confidence;
    const bool previous_geometry_state_valid =
        SbsRulePreviousResolvedGeometryState(
            previous_geometry_state_kind,
            previous_committed_bounds,
            previous_committed_confidence);
    uint previous_acquisition_layout;
    float4 previous_acquisition_bounds;
    const bool previous_acquisition_valid =
        SbsRulePreviousAcquisitionTarget(
            previous_acquisition_layout,
            previous_acquisition_bounds);
    uint previous_challenger_layout;
    float4 previous_challenger_bounds;
    const bool previous_challenger_valid =
        SbsRulePreviousChallengerTarget(
            previous_challenger_layout,
            previous_challenger_bounds);
    bool controller_reset =
        !SbsRulePreviousStateIdentityValid() ||
        !SbsRulePreviousWordIsWellFormedBool(
            SBS_SCENE_RULE_STATE_WORD_OUTPUT_VALID) ||
        !previous_geometry_state_valid ||
        (effective_reset_flags &
            (SBS_SCENE_RESET_FLAGS_LAYOUT |
             SBS_SCENE_RESET_FLAGS_GEOMETRY |
             SBS_SCENE_RESET_FLAGS_BACKEND |
             SBS_SCENE_RESET_FLAGS_DISPLAY_OR_HDR)) != 0u;
    // Even if an auxiliary record is independently well formed, controller reset debt owns the
    // transition atomically. Never let an old acquisition/challenger resume after the committed
    // geometry or lifecycle identity has been rejected.
    const bool previous_acquisition_reusable =
        !controller_reset && previous_acquisition_valid;
    const bool previous_challenger_reusable =
        !controller_reset && previous_challenger_valid;
    const uint current_detector_cut_count = asuint(
        SBS_STATE_HARD_CUT_COUNT(
            AdaptiveState[SBS_STATE_VECTOR_HARD_CUT_COUNT]));
    const uint current_external_cut_count = asuint(
        SBS_STATE_EXTERNAL_CUT_COUNT(
            AdaptiveState[SBS_STATE_VECTOR_EXTERNAL_CUT_COUNT]));
    const bool previous_counter_state_valid =
        SbsRulePreviousCounterStateValid();
    uint last_detector_cut_count =
        previous_counter_state_valid ?
            PreviousStateUint(
                SBS_SCENE_RULE_STATE_WORD_LAST_DETECTOR_CUT_COUNT) :
            current_detector_cut_count;
    bool previous_detector_cut_pending =
        previous_counter_state_valid &&
        (PreviousStateUint(
            SBS_SCENE_RULE_STATE_WORD_STATE_FLAGS) &
         SBS_SCENE_STATE_FLAGS_DETECTOR_CUT_PENDING) != 0u;
    // The adaptive detector state is rebuilt independently of the scene controller. A lower
    // count therefore begins a new counter epoch rather than becoming permanently older than
    // the retained baseline. Baseline it immediately, including on invalid/frozen updates.
    const bool detector_counter_epoch_changed =
        previous_counter_state_valid &&
        current_detector_cut_count < last_detector_cut_count;
    if (detector_counter_epoch_changed) {
        last_detector_cut_count = current_detector_cut_count;
        previous_detector_cut_pending = false;
    }
    const bool detector_depth_reset_supersedes =
        (effective_reset_flags &
            (SBS_SCENE_RESET_FLAGS_DEPTH_SHOT |
             SBS_SCENE_RESET_FLAGS_GEOMETRY |
             SBS_SCENE_RESET_FLAGS_BACKEND |
             SBS_SCENE_RESET_FLAGS_DISPLAY_OR_HDR)) != 0u;
    [unroll]
    for (uint vector_index = 0u;
         vector_index < SBS_SCENE_RULE_STATE_VECTOR_COUNT;
         ++vector_index) {
        float4 vector_value =
            controller_reset ?
                SbsSceneRuleStateInitialVector(vector_index) :
                asfloat(PreviousRuleState[vector_index]);
        state[vector_index * 4u] = vector_value.x;
        state[vector_index * 4u + 1u] = vector_value.y;
        state[vector_index * 4u + 2u] = vector_value.z;
        state[vector_index * 4u + 3u] = vector_value.w;
    }
    // A rejected auxiliary role must not remain externally marked live, including while backend
    // evidence is invalid and the resolver takes the strict-hold return below. Canonical payloads
    // also prevent NaNs or nonintegral enums from surviving indefinitely in published state.
    if (!previous_acquisition_valid) {
        ClearAcquisitionTarget(state);
    }
    if (!previous_challenger_valid) {
        ClearChallengerTarget(state);
    }
    uint roi_generation =
        controller_reset ?
            0u :
            PreviousStateUint(SBS_SCENE_RULE_STATE_WORD_ROI_GENERATION);
    uint update_count =
        controller_reset ?
            0u :
            PreviousStateUint(SBS_SCENE_RULE_STATE_WORD_UPDATE_COUNT);
    uint last_external_cut_count =
        previous_counter_state_valid ?
            PreviousStateUint(
                SBS_SCENE_RULE_STATE_WORD_LAST_EXTERNAL_CUT_COUNT) :
            current_external_cut_count;
    // External and detector counters share the independently rebuilt adaptive state.
    // Rebase a lower value immediately so the next increment in the new epoch is observable.
    if (
        previous_counter_state_valid &&
        current_external_cut_count < last_external_cut_count
    ) {
        last_external_cut_count = current_external_cut_count;
    }

    bool evidence_valid =
        EvidenceGlobal[
            SBS_SCENE_GLOBAL_OUT_BACKEND_OUTPUT_VALID] > 0.5f;
    if (!evidence_valid) {
        // Partial/invalid output is a strict hold. Do not age release timers, move ROI state,
        // consume cut latches, or alter controller decisions. Reset debt is retained in the
        // exact uint lane until a later valid frame atomically seeds fresh histories.
        state[SBS_SCENE_RULE_STATE_WORD_OUTPUT_VALID] = 0.0f;
        state[SBS_SCENE_RULE_STATE_WORD_OOD_CONFIDENCE] = 1.0f;
        state[SBS_SCENE_RULE_STATE_WORD_BACKEND_OUTPUT_AGE_S] =
            StateWord(
                state,
                SBS_SCENE_RULE_STATE_WORD_BACKEND_OUTPUT_AGE_S) +
            max(scene_elapsed_seconds, 0.0f);
        state[SBS_SCENE_RULE_STATE_WORD_REJECTION_REASON] =
            (float)SBS_SCENE_REJECTION_REASON_INVALID_INPUT;
        state[
            SBS_SCENE_RULE_STATE_WORD_ROI_STRUCTURAL_CUT_SUPPORT] =
            0.0f;
        state[
            SBS_SCENE_RULE_STATE_WORD_ROI_EXPOSURE_ONLY_SUPPORT] =
            0.0f;
        state[
            SBS_SCENE_RULE_STATE_WORD_ROI_EVENT_DEPTH_COVERAGE] =
            0.0f;

        update_count = min(update_count + 1u, 0xfffffffeu);
        uint held_state_flags =
            controller_reset ?
                SBS_SCENE_STATE_FLAGS_INITIALIZED :
                PreviousStateUint(SBS_SCENE_RULE_STATE_WORD_STATE_FLAGS);
        uint held_state_kind =
            controller_reset ?
                SBS_SCENE_STATE_KIND_FULL_FRAME :
                previous_geometry_state_kind;
        if (!controller_reset) {
            state[SBS_SCENE_RULE_STATE_WORD_STATE_KIND] =
                (float)held_state_kind;
            state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_X0] =
                previous_committed_bounds.x;
            state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_Y0] =
                previous_committed_bounds.y;
            state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_X1] =
                previous_committed_bounds.z;
            state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_Y1] =
                previous_committed_bounds.w;
            state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_CONFIDENCE] =
                previous_committed_confidence;
        }
        // Temporal lockout is retired. Preserve the strict invalid-evidence hold for every live
        // decision while still publishing the reserved flag in its canonical zero state.
        held_state_flags &= ~SBS_SCENE_STATE_FLAGS_TEMPORAL_LOCKOUT_ACTIVE;
        if (held_state_kind == SBS_SCENE_STATE_KIND_FULL_FRAME) {
            held_state_flags &= ~SBS_SCENE_STATE_FLAGS_ROI_LOCKED;
        }
        held_state_flags &=
            ~SBS_SCENE_STATE_FLAGS_DETECTOR_CUT_PENDING;
        if (
            previous_detector_cut_pending &&
            !detector_depth_reset_supersedes
        ) {
            held_state_flags |=
                SBS_SCENE_STATE_FLAGS_DETECTOR_CUT_PENDING;
        }
        held_state_flags |= SBS_SCENE_STATE_FLAGS_FALLBACK_ACTIVE;
        uint held_promotion_flags = 0u;
        uint held_history_flags =
            SBS_SCENE_HISTORY_FLAGS_LAYOUT_READ_BANK |
            SBS_SCENE_HISTORY_FLAGS_DEPTH_READ_BANK;
        uint held_diagnostic_flags =
            SBS_SCENE_DIAGNOSTIC_FLAGS_OOD |
            SBS_SCENE_DIAGNOSTIC_FLAGS_DEPTH_INVALID;

        [unroll]
        for (uint word = 0u;
             word < SBS_SCENE_GLOBAL_OUT_WORD_COUNT;
             ++word) {
            GlobalOutput[word] = 0.0f;
        }
        GlobalOutput[
            SBS_SCENE_GLOBAL_OUT_ZERO_PLANE_NEUTRAL_ABSTAIN] =
            SceneLogit(0.98f);
        GlobalOutput[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_OOD] = 1.0f;
        GlobalOutput[SBS_SCENE_GLOBAL_OUT_BACKEND_OUTPUT_VALID] = 0.0f;

        [unroll]
        for (uint vector_index = 0u;
             vector_index < SBS_SCENE_RULE_STATE_VECTOR_COUNT;
             ++vector_index) {
            uint4 resolved_bits = asuint(float4(
                state[vector_index * 4u],
                state[vector_index * 4u + 1u],
                state[vector_index * 4u + 2u],
                state[vector_index * 4u + 3u]));
            if (vector_index ==
                SBS_SCENE_RULE_STATE_VECTOR_BACKEND_GENERATION) {
                resolved_bits.w = scene_backend_generation;
            }
            if (vector_index ==
                SBS_SCENE_RULE_STATE_VECTOR_ROI_GENERATION) {
                resolved_bits.x = roi_generation;
                resolved_bits.y = update_count;
                resolved_bits.z = held_state_flags;
                resolved_bits.w = effective_reset_flags;
            }
            if (vector_index ==
                SBS_SCENE_RULE_STATE_VECTOR_PROMOTION_FLAGS) {
                resolved_bits.y = held_promotion_flags;
                resolved_bits.z = held_history_flags;
                resolved_bits.w = held_diagnostic_flags;
            }
            if (vector_index ==
                SBS_SCENE_RULE_STATE_VECTOR_LAST_EXTERNAL_CUT_COUNT) {
                resolved_bits[
                    SBS_SCENE_RULE_STATE_WORD_LAST_EXTERNAL_CUT_COUNT &
                    3u] = last_external_cut_count;
            }
            if (vector_index ==
                SBS_SCENE_RULE_STATE_VECTOR_LAST_DETECTOR_CUT_COUNT) {
                resolved_bits.x = last_detector_cut_count;
                resolved_bits.yzw = uint3(0u, 0u, 0u);
            }
            NextRuleState[vector_index] = resolved_bits;
        }
        return;
    }
    float roi_confidence = saturate(
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_ROI]);
    float mask_confidence = saturate(
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_MASK]);
    float ood_confidence = saturate(
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_OOD]);
    uint candidate_layout = EvidenceLayout();
    bool ambiguity_evidence =
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_LAYOUT_AMBIGUOUS] > 0.0f;
    float4 candidate_bounds = CandidateBounds();
    bool candidate_bounds_valid =
        all(candidate_bounds.zw > candidate_bounds.xy) &&
        all(candidate_bounds >= 0.0f) &&
        all(candidate_bounds <= 1.0f);
    bool candidate_is_roi =
        candidate_layout == SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO ||
        candidate_layout == SBS_SCENE_LAYOUT_DECISION_CONTENT_COLLAGE;
    const bool candidate_is_identity =
        candidate_layout == SBS_SCENE_LAYOUT_DECISION_IDENTITY_FULLSCREEN;
    bool candidate_valid =
        evidence_valid &&
        (candidate_is_roi || candidate_is_identity) &&
        candidate_bounds_valid &&
        !ambiguity_evidence;
    const uint previous_state_kind =
        controller_reset ?
            SBS_SCENE_STATE_KIND_FULL_FRAME :
            previous_geometry_state_kind;
    const float raw_previous_layout_evidence_seconds = StateWord(
        state,
        SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE);
    const bool previous_layout_evidence_seconds_valid =
        isfinite(raw_previous_layout_evidence_seconds) &&
        raw_previous_layout_evidence_seconds >= 0.0f;
    if (!previous_layout_evidence_seconds_valid) {
        state[SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE] =
            0.0f;
    }

    uint state_kind = previous_state_kind;
    float4 committed_bounds =
        controller_reset ?
            float4(0.0f, 0.0f, 1.0f, 1.0f) :
            previous_committed_bounds;
    float committed_confidence =
        controller_reset ?
            0.0f :
            previous_committed_confidence;
    bool committed_this_update = false;
    uint promotion_flags =
        SBS_SCENE_PROMOTION_FLAGS_LAYOUT_HISTORY |
        SBS_SCENE_PROMOTION_FLAGS_DEPTH_HISTORY;

    // Repeated broad coherent page translation freezes ROI/depth decisions.
    // Ambiguous motion is deliberately not special-cased into an acquisition:
    // the safe answer during page motion is to retain committed geometry and
    // discard every uncommitted target.
    float scroll_confidence = saturate(
        EvidenceGlobal[
            SBS_SCENE_GLOBAL_OUT_GLOBAL_SCROLL_CONFIDENCE]);
    const float2 scroll_direction = float2(
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_GLOBAL_SCROLL_X],
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_GLOBAL_SCROLL_Y]);
    const float previous_scroll_confidence = controller_reset ? 0.0f :
        saturate(SbsRulePreviousStateWord(
            SBS_SCENE_RULE_STATE_WORD_SCROLL_CONFIDENCE));
    const float2 previous_scroll_direction = controller_reset ?
        0.0f.xx :
        float2(
            SbsRulePreviousStateWord(
                SBS_SCENE_RULE_STATE_WORD_SCROLL_X),
            SbsRulePreviousStateWord(
                SBS_SCENE_RULE_STATE_WORD_SCROLL_Y));
    const bool previous_scroll_hold_active =
        !controller_reset &&
        (PreviousStateUint(
             SBS_SCENE_RULE_STATE_WORD_STATE_FLAGS) &
         SBS_SCENE_STATE_FLAGS_SCROLL_HOLD_ACTIVE) != 0u;
    const bool raw_scroll_observed = scroll_confidence > 0.0f;
    const bool consistent_scroll_observation =
        raw_scroll_observed &&
        previous_scroll_confidence > 0.0f &&
        dot(scroll_direction, previous_scroll_direction) > 0.5f;
    const float elapsed_seconds = max(scene_elapsed_seconds, 0.0f);
    const float scroll_enter_seconds =
        isfinite(scene_scroll_enter_seconds) &&
        scene_scroll_enter_seconds >= 0.0f ?
            scene_scroll_enter_seconds :
            0.05f;

    float scroll_dwell = StateWord(
        state,
        SBS_SCENE_RULE_STATE_WORD_SCROLL_DWELL_S);
    const float previous_scroll_hold_seconds = max(
        StateWord(
            state,
            SBS_SCENE_RULE_STATE_WORD_SCROLL_HOLD_S),
        0.0f);
    float scroll_hold = previous_scroll_hold_seconds;
    const bool post_scroll_release_resync =
        !previous_scroll_hold_active &&
        previous_scroll_hold_seconds >= 0.12f;
    bool scroll_hold_active = false;
    if (previous_scroll_hold_active && raw_scroll_observed) {
        // Once active, any continuing broad translation retains the safety
        // hold immediately, including a direction reversal.
        scroll_dwell = max(scroll_dwell, scroll_enter_seconds);
        scroll_hold = 0.0f;
        scroll_hold_active = true;
    } else if (!previous_scroll_hold_active && raw_scroll_observed) {
        // Entry is measured on the captured-stream presentation clock. A
        // 50-ms proof therefore takes the same wall time at 72 Hz, 90 Hz, or
        // with dropped inference opportunities.
        scroll_dwell =
            consistent_scroll_observation ?
                scroll_dwell + elapsed_seconds :
                0.0f;
        scroll_hold = 0.0f;
        scroll_hold_active =
            scroll_dwell >= scroll_enter_seconds;
    } else {
        scroll_dwell = 0.0f;
        if (previous_scroll_hold_active) {
            scroll_hold += elapsed_seconds;
            scroll_hold_active = scroll_hold < 0.12f;
            if (!scroll_hold_active) {
                // Retain the completed quiet duration for exactly one state
                // update. It marks the following update as the bounded
                // post-release detector quarantine without adding another
                // controller state, flag, or timer.
                scroll_hold = max(scroll_hold, 0.12f);
            }
        } else {
            scroll_hold = 0.0f;
        }
    }
    const bool exposure_blocked =
        Meta[SBS_SCENE_META_EXPOSURE_CHANGE_EVIDENCE] > 0.5f;
    const bool geometry_blocked =
        scroll_hold_active || exposure_blocked;
    if (geometry_blocked) {
        candidate_valid = false;
    }
    if (scroll_hold_active) {
        // Scroll invalidates every uncommitted target. Exposure is only a photometric veto and
        // therefore freezes the pending state instead of destroying useful geometry evidence.
        ClearAcquisitionTarget(state);
        ClearChallengerTarget(state);
        state[SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE] =
            0.0f;
    }

    bool committed_roi =
        state_kind == SBS_SCENE_STATE_KIND_VIDEO ||
        state_kind == SBS_SCENE_STATE_KIND_CONTENT;
    if (!geometry_blocked) {
        if (!committed_roi) {
            if (candidate_valid && candidate_is_identity) {
                // Full-frame is already the current geometry. It is valid evidence, but it
                // must not create a synthetic content acquisition.
                ClearAcquisitionTarget(state);
                ClearChallengerTarget(state);
                state[
                    SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE] =
                    0.0f;
            } else if (candidate_valid) {
                bool same_acquisition =
                    previous_acquisition_reusable &&
                    previous_acquisition_layout == candidate_layout &&
                    SceneIou(
                        previous_acquisition_bounds,
                        candidate_bounds) >=
                        SBS_SCENE_ROI_GEOMETRY_IDENTITY_MIN_IOU;
                const float acquisition_gap =
                    same_acquisition &&
                    previous_acquisition_layout ==
                        SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO ?
                        max(
                            StateWord(
                                state,
                                SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE),
                            0.0f) :
                        0.0f;
                float acquisition_dwell =
                    same_acquisition ?
                        StateWord(
                            state,
                            SBS_SCENE_RULE_STATE_WORD_ACQUISITION_DWELL_S) +
                            acquisition_gap +
                            max(scene_elapsed_seconds, 0.0f) :
                        0.0f;
                state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_ROI_X0] =
                    candidate_bounds.x;
                state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_ROI_Y0] =
                    candidate_bounds.y;
                state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_ROI_X1] =
                    candidate_bounds.z;
                state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_ROI_Y1] =
                    candidate_bounds.w;
                state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_SCORE] =
                    0.0f;
                state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_DWELL_S] =
                    acquisition_dwell;
                state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_LAYOUT] =
                    (float)candidate_layout;
                state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_VALID] = 1.0f;
                state[
                    SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE] =
                    0.0f;
                ClearChallengerTarget(state);
                const float required_acquisition_dwell =
                    candidate_layout ==
                        SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO ?
                        max(
                            scene_acquire_seconds,
                            SBS_RULE_INITIAL_VIDEO_ACQUIRE_MIN_SECONDS) :
                        max(
                            scene_acquire_seconds,
                            SBS_RULE_INITIAL_CONTENT_ACQUIRE_MIN_SECONDS);
                if (
                    acquisition_dwell >= required_acquisition_dwell
                ) {
                    state_kind =
                        candidate_layout ==
                            SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO ?
                            SBS_SCENE_STATE_KIND_VIDEO :
                            SBS_SCENE_STATE_KIND_CONTENT;
                    committed_bounds = candidate_bounds;
                    committed_confidence = roi_confidence;
                    state[
                        SBS_SCENE_RULE_STATE_WORD_COMMITTED_MASK_CONFIDENCE] =
                        mask_confidence;
                    state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_LAYOUT] =
                        (float)candidate_layout;
                    state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_AGE_S] =
                        0.0f;
                    ClearAcquisitionTarget(state);
                    ClearChallengerTarget(state);
                    state[
                        SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE] =
                        0.0f;
                    roi_generation =
                        min(roi_generation + 1u, 0xfffffffeu);
                    promotion_flags |= SBS_SCENE_PROMOTION_FLAGS_ROI;
                    committed_this_update = true;
                }
            } else {
                // Visual content may update more slowly than the accepted stream cadence.
                // Preserve only a temporal acquisition across that bounded stream-time gap;
                // dwell advances only when matching fresh evidence returns, at which point the
                // elapsed stream time is charged once. Static, ambiguous, and genuinely missing
                // evidence still clear without another state machine or threshold family.
                const bool temporal_cadence_gap =
                    !ambiguity_evidence &&
                    previous_acquisition_reusable &&
                    previous_acquisition_layout ==
                        SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO;
                float missing = temporal_cadence_gap ?
                    max(
                        StateWord(
                            state,
                            SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE),
                        0.0f) +
                        max(scene_elapsed_seconds, 0.0f) :
                    0.0f;
                const bool retain_temporal_acquisition =
                    temporal_cadence_gap &&
                    missing <=
                        SBS_SCENE_TEMPORAL_CADENCE_GAP_SECONDS;
                if (!retain_temporal_acquisition) {
                    ClearAcquisitionTarget(state);
                    missing = 0.0f;
                }
                ClearChallengerTarget(state);
                state[
                    SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE] =
                    missing;
            }
        } else {
            ClearAcquisitionTarget(state);
            bool candidate_is_incumbent =
                candidate_valid &&
                ((state_kind == SBS_SCENE_STATE_KIND_VIDEO &&
                  candidate_layout ==
                      SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO) ||
                 (state_kind == SBS_SCENE_STATE_KIND_CONTENT &&
                  candidate_layout ==
                      SBS_SCENE_LAYOUT_DECISION_CONTENT_COLLAGE)) &&
                SceneIou(committed_bounds, candidate_bounds) >=
                    SBS_SCENE_ROI_GEOMETRY_IDENTITY_MIN_IOU;
            if (candidate_is_incumbent) {
                // Stable geometry is intentionally anchored. A candidate must become a decisive
                // challenger before its rectangle can move, preventing one-cell ROI breathing.
                committed_confidence = roi_confidence;
                state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_AGE_S] +=
                    max(scene_elapsed_seconds, 0.0f);
                ClearChallengerTarget(state);
                state[
                    SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE] =
                    0.0f;
            } else if (candidate_valid) {
                bool same_challenger =
                    previous_challenger_reusable &&
                    previous_challenger_layout == candidate_layout &&
                    SceneIou(
                        previous_challenger_bounds,
                        candidate_bounds) >=
                        SBS_SCENE_ROI_GEOMETRY_IDENTITY_MIN_IOU;
                const float challenger_gap =
                    same_challenger &&
                    previous_challenger_layout ==
                        SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO ?
                        max(
                            StateWord(
                                state,
                                SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE),
                            0.0f) :
                        0.0f;
                float challenger_dwell =
                    same_challenger ?
                        StateWord(
                            state,
                            SBS_SCENE_RULE_STATE_WORD_CHALLENGER_DWELL_S) +
                            challenger_gap +
                            max(scene_elapsed_seconds, 0.0f) :
                        0.0f;
                state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_ROI_X0] =
                    candidate_bounds.x;
                state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_ROI_Y0] =
                    candidate_bounds.y;
                state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_ROI_X1] =
                    candidate_bounds.z;
                state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_ROI_Y1] =
                    candidate_bounds.w;
                state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_SCORE] =
                    0.0f;
                state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_DWELL_S] =
                    challenger_dwell;
                state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_LAYOUT] =
                    (float)candidate_layout;
                state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_VALID] = 1.0f;
                state[
                    SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE] =
                    0.0f;
                const float required_challenger_dwell =
                    max(scene_challenger_seconds, 0.75f);
                if (
                    challenger_dwell >= required_challenger_dwell
                ) {
                    state_kind =
                        candidate_is_identity ?
                            SBS_SCENE_STATE_KIND_FULL_FRAME :
                        candidate_layout ==
                            SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO ?
                            SBS_SCENE_STATE_KIND_VIDEO :
                            SBS_SCENE_STATE_KIND_CONTENT;
                    committed_bounds =
                        candidate_is_identity ?
                            float4(0.0f, 0.0f, 1.0f, 1.0f) :
                            candidate_bounds;
                    committed_confidence =
                        candidate_is_identity ? 0.0f : roi_confidence;
                    state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_LAYOUT] =
                        (float)candidate_layout;
                    state[
                        SBS_SCENE_RULE_STATE_WORD_COMMITTED_MASK_CONFIDENCE] =
                        candidate_is_identity ? 0.0f : mask_confidence;
                    state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_AGE_S] =
                        0.0f;
                    ClearChallengerTarget(state);
                    ClearAcquisitionTarget(state);
                    state[
                        SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE] =
                        0.0f;
                    roi_generation =
                        min(roi_generation + 1u, 0xfffffffeu);
                    promotion_flags |= SBS_SCENE_PROMOTION_FLAGS_ROI;
                    committed_this_update = true;
                }
            } else {
                const bool temporal_cadence_gap =
                    !ambiguity_evidence &&
                    previous_challenger_reusable &&
                    previous_challenger_layout ==
                        SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO;
                float missing = temporal_cadence_gap ?
                    max(
                        StateWord(
                            state,
                            SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE),
                        0.0f) +
                        max(scene_elapsed_seconds, 0.0f) :
                    0.0f;
                const bool retain_temporal_challenger =
                    temporal_cadence_gap &&
                    missing <=
                        SBS_SCENE_TEMPORAL_CADENCE_GAP_SECONDS;
                if (!retain_temporal_challenger) {
                    ClearChallengerTarget(state);
                    missing = 0.0f;
                }
                if (retain_temporal_challenger) {
                    state[
                        SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE] =
                        missing;
                } else if (state_kind == SBS_SCENE_STATE_KIND_VIDEO) {
                    // Playback history is the only evidence that distinguishes a paused player
                    // from a static image. Once proven, quiet time cannot revoke that evidence:
                    // retain the exact ROI until a stable replacement or invalidating reset.
                    state[
                        SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE] =
                        0.0f;
                } else {
                    // Generic static content has no paused-playback contract. Its ordinary
                    // no-evidence ceiling still returns the pipeline to canonical full frame.
                    float missing = StateWord(
                        state,
                        SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE);
                    missing += max(scene_elapsed_seconds, 0.0f);
                    state[
                        SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE] =
                        missing;
                    if (missing >= scene_release_seconds) {
                        state_kind = SBS_SCENE_STATE_KIND_FULL_FRAME;
                        committed_bounds =
                            float4(0.0f, 0.0f, 1.0f, 1.0f);
                        committed_confidence = 0.0f;
                        state[
                            SBS_SCENE_RULE_STATE_WORD_COMMITTED_MASK_CONFIDENCE] =
                            0.0f;
                        state[
                            SBS_SCENE_RULE_STATE_WORD_COMMITTED_LAYOUT] =
                            (float)SBS_SCENE_LAYOUT_DECISION_NO_TARGET;
                        state[
                            SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_AGE_S] =
                            0.0f;
                        ClearAcquisitionTarget(state);
                        ClearChallengerTarget(state);
                        roi_generation =
                            min(roi_generation + 1u, 0xfffffffeu);
                        promotion_flags |= SBS_SCENE_PROMOTION_FLAGS_ROI;
                        committed_this_update = true;
                    }
                }
            }
        }
    }

    state[SBS_SCENE_RULE_STATE_WORD_SCHEMA_VERSION] =
        (float)SBS_SCENE_SCHEMA_VERSION;
    state[SBS_SCENE_RULE_STATE_WORD_STATE_KIND] = (float)state_kind;
    state[SBS_SCENE_RULE_STATE_WORD_OUTPUT_VALID] =
        evidence_valid ? 1.0f : 0.0f;
    update_count = min(update_count + 1u, 0xfffffffeu);
    // The first update that releases the scroll hold still compares against
    // depth history from before the page translation. Treat that update as a
    // history-resynchronization boundary: keep externally visible decisions
    // frozen for one update, but allow the current depth bank to be promoted
    // below. Otherwise the translated page can be misreported as a hard cut.
    const bool scroll_release_resync =
        previous_scroll_hold_active && !scroll_hold_active;
    bool decisions_frozen =
        scroll_hold_active || scroll_release_resync;
    const bool previous_roi_locked =
        !controller_reset &&
        (PreviousStateUint(
            SBS_SCENE_RULE_STATE_WORD_STATE_FLAGS) &
         SBS_SCENE_STATE_FLAGS_ROI_LOCKED) != 0u;
    const bool detector_cut_count_unconsumed =
        current_detector_cut_count > last_detector_cut_count;
    const bool detector_cut_evidence_accepted =
        detector_cut_count_unconsumed &&
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_EVENT_HARD_CUT] > 0.0f;
    bool detector_cut_pending =
        previous_detector_cut_pending &&
        detector_cut_count_unconsumed;
    if (
        scroll_hold_active &&
        previous_roi_locked &&
        detector_cut_evidence_accepted
    ) {
        // Only a cut structurally corroborated inside an already-proven ROI
        // may survive the hold. In FULL_FRAME, a detector count during scroll
        // has no trustworthy geometry attribution and is consumed by the
        // release quarantine instead of becoming a delayed false cut.
        detector_cut_pending = true;
    }
    const float detector_roi_event_cells =
        RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_DEPTH_COVERAGE)];
    const float detector_roi_depth_weight =
        RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_DEPTH_CONFIDENCE)];
    const float detector_roi_depth_coverage =
        detector_roi_event_cells > 0.0f ?
            saturate(
                detector_roi_depth_weight /
                detector_roi_event_cells) :
            0.0f;
    // A reliable same-frame rejection belongs to that frame even while geometry decisions are
    // scroll-frozen. Consume it immediately; otherwise a later unrelated in-ROI change could
    // corroborate the stale count. Insufficient/invalid evidence remains durable.
    const bool detector_cut_rejection_definitive =
        detector_cut_count_unconsumed &&
        !detector_cut_pending &&
        !detector_cut_evidence_accepted &&
        !detector_depth_reset_supersedes &&
        previous_roi_locked &&
        detector_roi_event_cells >= 4.0f &&
        detector_roi_depth_coverage >=
            SBS_RULE_ROI_EVENT_MIN_DEPTH_COVERAGE;
    float history_depth_fraction = saturate(
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_40]);
    uint state_flags =
        SBS_SCENE_STATE_FLAGS_INITIALIZED |
        SBS_SCENE_STATE_FLAGS_LAYOUT_HISTORY_VALID;
    if (
        evidence_valid &&
        history_depth_fraction > 0.05f
    ) {
        state_flags |= SBS_SCENE_STATE_FLAGS_DEPTH_HISTORY_VALID;
    }
    if (
        state_kind == SBS_SCENE_STATE_KIND_VIDEO ||
        state_kind == SBS_SCENE_STATE_KIND_CONTENT
    ) {
        state_flags |= SBS_SCENE_STATE_FLAGS_ROI_LOCKED;
    }
    if (scroll_hold_active) {
        state_flags |= SBS_SCENE_STATE_FLAGS_SCROLL_HOLD_ACTIVE;
    }
    if (detector_cut_pending) {
        state_flags |= SBS_SCENE_STATE_FLAGS_DETECTOR_CUT_PENDING;
    }
    if (!evidence_valid) {
        state_flags |= SBS_SCENE_STATE_FLAGS_FALLBACK_ACTIVE;
    }
    float4 adaptive_cut_state =
        AdaptiveState[SBS_STATE_VECTOR_CUT_FLAGS];
    uint adaptive_cut_flags =
        (uint)max(SBS_STATE_CUT_FLAGS(adaptive_cut_state), 0.0f);
    bool geometry_cut_armed =
        (adaptive_cut_flags & CUT_FLAG_GEOMETRY_ARMED) != 0u;
    bool appearance_cut_armed =
        (adaptive_cut_flags & CUT_FLAG_APPEARANCE_ARMED) != 0u;
    bool shot_latched =
        (adaptive_cut_flags & CUT_FLAG_LATCHED) != 0u;
    if (decisions_frozen) {
        uint previous_state_flags =
            PreviousStateUint(SBS_SCENE_RULE_STATE_WORD_STATE_FLAGS);
        state_flags |= previous_state_flags &
            (SBS_SCENE_STATE_FLAGS_CUT_ARMED |
             SBS_SCENE_STATE_FLAGS_CUT_ACCEPTED_EVER);
        if (scroll_release_resync) {
            // The hold flag is already clear, so do not leave telemetry
            // claiming that the page is still scrolling during the resync
            // update. Geometry remains FULL_FRAME unless an already-proven
            // incumbent exists; only the candidate attribution is refreshed.
            state[SBS_SCENE_RULE_STATE_WORD_LAYOUT_DECISION] =
                (float)candidate_layout;
            state[SBS_SCENE_RULE_STATE_WORD_LAYOUT_CONFIDENCE] =
                roi_confidence;
            state[SBS_SCENE_RULE_STATE_WORD_EVENT_DECISION] =
                (float)SBS_SCENE_EVENT_DECISION_SAME_SHOT;
            state[SBS_SCENE_RULE_STATE_WORD_EVENT_CONFIDENCE] = 1.0f;
        }
    } else {
        if (geometry_cut_armed || appearance_cut_armed) {
            state_flags |= SBS_SCENE_STATE_FLAGS_CUT_ARMED;
        }
        if (shot_latched) {
            state_flags |= SBS_SCENE_STATE_FLAGS_CUT_ACCEPTED_EVER;
        }
    }
    state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_X0] =
        committed_bounds.x;
    state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_Y0] =
        committed_bounds.y;
    state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_X1] =
        committed_bounds.z;
    state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_Y1] =
        committed_bounds.w;
    state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_CONFIDENCE] =
        committed_confidence;
    state[SBS_SCENE_RULE_STATE_WORD_INCUMBENT_SCORE] =
        RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_BEST_SCORE)];
    state[SBS_SCENE_RULE_STATE_WORD_RUNNER_UP_SCORE] =
        RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_SECOND_SCORE)];
    state[SBS_SCENE_RULE_STATE_WORD_WINNER_MARGIN] =
        RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_WINNER_MARGIN)];
    state[SBS_SCENE_RULE_STATE_WORD_CANDIDATE_COVERAGE] =
        RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_CANDIDATE_COVERAGE)];
    state[SBS_SCENE_RULE_STATE_WORD_SCROLL_X] =
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_GLOBAL_SCROLL_X];
    state[SBS_SCENE_RULE_STATE_WORD_SCROLL_Y] =
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_GLOBAL_SCROLL_Y];
    state[SBS_SCENE_RULE_STATE_WORD_SCROLL_CONFIDENCE] =
        scroll_confidence;
    state[SBS_SCENE_RULE_STATE_WORD_SCROLL_DWELL_S] = scroll_dwell;
    state[SBS_SCENE_RULE_STATE_WORD_SCROLL_HOLD_S] = scroll_hold;
    const bool external_cut_request =
        Meta[SBS_SCENE_META_EXTERNAL_CUT_REQUEST] > 0.5f &&
        current_external_cut_count > last_external_cut_count;
    const bool detector_reset_completed =
        detector_depth_reset_supersedes ||
        committed_this_update;
    bool external_cut_accepted = false;
    bool detector_cut_accepted = false;
    const bool suppress_post_scroll_cut =
        post_scroll_release_resync &&
        !previous_detector_cut_pending &&
        !previous_roi_locked;
    if (!decisions_frozen) {
        state[SBS_SCENE_RULE_STATE_WORD_LAYOUT_DECISION] =
            (float)candidate_layout;
        state[SBS_SCENE_RULE_STATE_WORD_LAYOUT_CONFIDENCE] =
            roi_confidence;

        // The adaptive detector can publish a cut count one pipeline update
        // after the controller has released scroll. With no cut already
        // corroborated during the hold, that single delayed edge is
        // indistinguishable from scroll settling and is consumed as
        // ambiguous. A previously pending cut remains authoritative.
        detector_cut_accepted =
            !suppress_post_scroll_cut &&
            !committed_this_update &&
            !detector_reset_completed &&
            detector_cut_count_unconsumed &&
            (detector_cut_pending ||
             detector_cut_evidence_accepted);
        uint event_decision =
            committed_this_update ?
                SBS_SCENE_EVENT_DECISION_GEOMETRY_RESET :
            external_cut_request ?
                SBS_SCENE_EVENT_DECISION_HARD_CUT :
            detector_cut_accepted ?
                SBS_SCENE_EVENT_DECISION_HARD_CUT :
            suppress_post_scroll_cut ?
                SBS_SCENE_EVENT_DECISION_SAME_SHOT :
                EvidenceEvent();
        external_cut_accepted =
            external_cut_request &&
            !committed_this_update;
        state[SBS_SCENE_RULE_STATE_WORD_EVENT_DECISION] =
            (float)event_decision;
        state[SBS_SCENE_RULE_STATE_WORD_EVENT_CONFIDENCE] =
            external_cut_accepted || detector_cut_accepted ?
                1.0f :
                EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_EVENT];
        if (event_decision == SBS_SCENE_EVENT_DECISION_HARD_CUT) {
            state[SBS_SCENE_RULE_STATE_WORD_SCENE_AGE_S] = 0.0f;
            promotion_flags |= SBS_SCENE_PROMOTION_FLAGS_SHOT;
        } else {
            state[SBS_SCENE_RULE_STATE_WORD_SCENE_AGE_S] +=
                max(scene_elapsed_seconds, 0.0f);
        }
        state[SBS_SCENE_RULE_STATE_WORD_CUT_COOLDOWN_S] =
            shot_latched &&
            !(geometry_cut_armed || appearance_cut_armed) ?
                StateWord(
                    state,
                    SBS_SCENE_RULE_STATE_WORD_CUT_COOLDOWN_S) +
                  max(scene_elapsed_seconds, 0.0f) :
                0.0f;
        state[SBS_SCENE_RULE_STATE_WORD_CUT_REARM_EVIDENCE] =
            0.5f *
            ((geometry_cut_armed ? 1.0f : 0.0f) +
             (appearance_cut_armed ? 1.0f : 0.0f));

        uint pop_action = 0u;
        [unroll]
        for (uint action = 0u;
             action < SBS_SCENE_POP_ACTION_COUNT;
             ++action) {
            if (
                EvidenceGlobal[
                    SBS_SCENE_GLOBAL_OUT_POP_ACTION_SAFE_0 + action] >= 0.0f
            ) {
                pop_action = action;
            }
        }
        state[SBS_SCENE_RULE_STATE_WORD_POP_ACTION] = (float)pop_action;
        state[SBS_SCENE_RULE_STATE_WORD_POP_STRENGTH] = lerp(
            scene_pop_floor,
            scene_pop_ceiling,
            (float)pop_action /
                (float)(SBS_SCENE_POP_ACTION_COUNT - 1u));
        state[SBS_SCENE_RULE_STATE_WORD_POP_CONFIDENCE] =
            EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_POP];

        uint zero_decision =
            scene_zero_plane_mode < 1.5f ?
                SBS_SCENE_ZERO_PLANE_DECISION_SUBJECT :
            scene_zero_plane_mode > 2.5f ?
                SBS_SCENE_ZERO_PLANE_DECISION_BACKGROUND :
                SBS_SCENE_ZERO_PLANE_DECISION_MEDIAN;
        state[SBS_SCENE_RULE_STATE_WORD_ZERO_PLANE_DECISION] =
            (float)zero_decision;
        state[SBS_SCENE_RULE_STATE_WORD_ZERO_PLANE_CONFIDENCE] =
            EvidenceGlobal[
                SBS_SCENE_GLOBAL_OUT_CONFIDENCE_ZERO_PLANE];
    }
    state[SBS_SCENE_RULE_STATE_WORD_OOD_CONFIDENCE] = ood_confidence;
    state[SBS_SCENE_RULE_STATE_WORD_BACKEND_OUTPUT_AGE_S] =
        evidence_valid ? 0.0f :
            StateWord(
                state,
                SBS_SCENE_RULE_STATE_WORD_BACKEND_OUTPUT_AGE_S) +
            max(scene_elapsed_seconds, 0.0f);
    state[SBS_SCENE_RULE_STATE_WORD_REJECTION_REASON] =
        ambiguity_evidence ?
            (float)SBS_SCENE_REJECTION_REASON_AMBIGUOUS :
            (float)SBS_SCENE_REJECTION_REASON_NONE;
    if (detector_reset_completed) {
        // A valid reset has already reseeded every shot-dependent history, so both cumulative
        // requests are consumed even when scroll freezes ordinary controller decisions.
        last_external_cut_count =
            max(last_external_cut_count, current_external_cut_count);
        last_detector_cut_count =
            max(
                last_detector_cut_count,
                current_detector_cut_count);
        detector_cut_pending = false;
        state_flags &=
            ~SBS_SCENE_STATE_FLAGS_DETECTOR_CUT_PENDING;
    } else if (!decisions_frozen) {
        last_external_cut_count =
            max(last_external_cut_count, current_external_cut_count);
        // A valid, unfrozen ROI decision is final for this detector count. If local
        // corroboration rejected it, baseline the count now so unrelated future structure
        // cannot resurrect the old cut. Invalid/frozen frames deliberately retain the delta.
        const bool detector_cut_rejected =
            detector_cut_count_unconsumed &&
            !detector_cut_pending &&
            (!detector_cut_evidence_accepted ||
             suppress_post_scroll_cut);
        if (
            detector_reset_completed ||
            detector_cut_accepted ||
            external_cut_accepted ||
            detector_cut_rejected
        ) {
            last_detector_cut_count =
                max(
                    last_detector_cut_count,
                    current_detector_cut_count);
            detector_cut_pending = false;
            state_flags &=
                ~SBS_SCENE_STATE_FLAGS_DETECTOR_CUT_PENDING;
        }
    } else if (detector_cut_rejection_definitive) {
        last_detector_cut_count =
            max(
                last_detector_cut_count,
                current_detector_cut_count);
        detector_cut_pending = false;
        state_flags &=
            ~SBS_SCENE_STATE_FLAGS_DETECTOR_CUT_PENDING;
    }
    if (!evidence_valid) {
        promotion_flags &=
            ~(SBS_SCENE_PROMOTION_FLAGS_LAYOUT_HISTORY |
              SBS_SCENE_PROMOTION_FLAGS_DEPTH_HISTORY);
    } else if (scroll_hold_active) {
        // Keep the appearance endpoint and motion channels moving so cumulative scrolling
        // remains measurable. Freeze the entire depth bank from the first coherent scroll
        // observation and throughout its bounded quiet hold. The release-resync update keeps
        // decisions frozen above but deliberately writes current depth here.
        promotion_flags &= ~SBS_SCENE_PROMOTION_FLAGS_DEPTH_HISTORY;
    }
    uint history_flags =
        SBS_SCENE_HISTORY_FLAGS_LAYOUT_READ_BANK |
        SBS_SCENE_HISTORY_FLAGS_DEPTH_READ_BANK;
    if ((promotion_flags &
         SBS_SCENE_PROMOTION_FLAGS_LAYOUT_HISTORY) != 0u) {
        history_flags |= SBS_SCENE_HISTORY_FLAGS_LAYOUT_WRITE_BANK;
    }
    if ((promotion_flags &
         SBS_SCENE_PROMOTION_FLAGS_DEPTH_HISTORY) != 0u) {
        history_flags |= SBS_SCENE_HISTORY_FLAGS_DEPTH_WRITE_BANK;
    }
    uint diagnostic_flags = 0u;
    if (ood_confidence >= 0.65f) {
        diagnostic_flags |= SBS_SCENE_DIAGNOSTIC_FLAGS_OOD;
    }
    if (ambiguity_evidence) {
        diagnostic_flags |= SBS_SCENE_DIAGNOSTIC_FLAGS_AMBIGUOUS;
    }
    if (history_depth_fraction <= 0.05f) {
        diagnostic_flags |= SBS_SCENE_DIAGNOSTIC_FLAGS_DEPTH_INVALID;
    }
    const float roi_event_cells =
        RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_DEPTH_COVERAGE)];
    const float roi_event_depth_weight =
        RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_DEPTH_CONFIDENCE)];
    state[SBS_SCENE_RULE_STATE_WORD_ROI_STRUCTURAL_CUT_SUPPORT] =
        roi_event_depth_weight > 0.0f ?
            saturate(
                RuleSummary[SbsRulePlanIndex(
                    SBS_RULE_PLAN_ROI_STRUCTURAL_WEIGHT)] /
                roi_event_depth_weight) :
            0.0f;
    state[SBS_SCENE_RULE_STATE_WORD_ROI_EXPOSURE_ONLY_SUPPORT] =
        roi_event_depth_weight > 0.0f ?
            saturate(
                RuleSummary[SbsRulePlanIndex(
                    SBS_RULE_PLAN_ROI_EXPOSURE_WEIGHT)] /
                roi_event_depth_weight) :
            0.0f;
    state[SBS_SCENE_RULE_STATE_WORD_ROI_EVENT_DEPTH_COVERAGE] =
        roi_event_cells > 0.0f ?
            saturate(roi_event_depth_weight / roi_event_cells) :
            0.0f;

    [unroll]
    for (uint word = 0u; word < 35u; ++word) {
        GlobalOutput[word] =
            isfinite(EvidenceGlobal[word]) ?
                EvidenceGlobal[word] :
                0.0f;
    }
    GlobalOutput[SBS_SCENE_GLOBAL_OUT_BACKEND_OUTPUT_VALID] =
        evidence_valid ? 1.0f : 0.0f;
    [unroll]
    for (uint word = 35u;
         word < SBS_SCENE_GLOBAL_OUT_WORD_COUNT;
         ++word) {
        GlobalOutput[word] = 0.0f;
    }
    const bool forced_post_scroll_same_shot =
        post_scroll_release_resync &&
        !committed_this_update &&
        !external_cut_accepted &&
        !detector_cut_accepted;
    if (
        scroll_release_resync ||
        forced_post_scroll_same_shot ||
        committed_this_update ||
        external_cut_accepted ||
        detector_cut_accepted
    ) {
        const uint forced_event_decision =
            (scroll_release_resync || forced_post_scroll_same_shot) ?
                SBS_SCENE_EVENT_DECISION_SAME_SHOT :
            committed_this_update ?
                SBS_SCENE_EVENT_DECISION_GEOMETRY_RESET :
                SBS_SCENE_EVENT_DECISION_HARD_CUT;
        [unroll]
        for (uint decision = 0u; decision < 6u; ++decision) {
            GlobalOutput[
                SBS_SCENE_GLOBAL_OUT_EVENT_SAME_SHOT + decision] =
                SceneLogit(
                    decision == forced_event_decision ?
                        0.995f :
                        0.02f);
        }
        if (
            scroll_release_resync ||
            forced_post_scroll_same_shot ||
            external_cut_accepted ||
            detector_cut_accepted
        ) {
            GlobalOutput[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_EVENT] = 1.0f;
        }
    }

    [unroll]
    for (uint vector_index = 0u;
         vector_index < SBS_SCENE_RULE_STATE_VECTOR_COUNT;
         ++vector_index) {
        float4 resolved_vector = float4(
            state[vector_index * 4u],
            state[vector_index * 4u + 1u],
            state[vector_index * 4u + 2u],
            state[vector_index * 4u + 3u]);
        uint4 resolved_bits = asuint(resolved_vector);
        // Keep uint-bit fields as direct output bitcasts. Routing tiny integer bit patterns
        // through the temporary float array lets some D3D11 compiler paths flush the subnormal
        // representation to numeric zero.
        if (vector_index ==
            SBS_SCENE_RULE_STATE_VECTOR_BACKEND_GENERATION) {
            resolved_bits.w = scene_backend_generation;
        }
        if (vector_index ==
            SBS_SCENE_RULE_STATE_VECTOR_ROI_GENERATION) {
            resolved_bits.x = roi_generation;
            resolved_bits.y = update_count;
            resolved_bits.z = state_flags;
            resolved_bits.w = effective_reset_flags;
        }
        if (vector_index ==
            SBS_SCENE_RULE_STATE_VECTOR_PROMOTION_FLAGS) {
            resolved_bits.y = promotion_flags;
            resolved_bits.z = history_flags;
            resolved_bits.w = diagnostic_flags;
        }
        if (vector_index ==
            SBS_SCENE_RULE_STATE_VECTOR_LAST_EXTERNAL_CUT_COUNT) {
            resolved_bits[
                SBS_SCENE_RULE_STATE_WORD_LAST_EXTERNAL_CUT_COUNT &
                3u] = last_external_cut_count;
        }
        if (vector_index ==
            SBS_SCENE_RULE_STATE_VECTOR_LAST_DETECTOR_CUT_COUNT) {
            resolved_bits.x = last_detector_cut_count;
            resolved_bits.yzw = uint3(0u, 0u, 0u);
        }
        NextRuleState[vector_index] = resolved_bits;
    }
}
