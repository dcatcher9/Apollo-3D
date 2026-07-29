StructuredBuffer<float> EvidenceGlobal : register(t0);
StructuredBuffer<uint4> PreviousRuleState : register(t1);
StructuredBuffer<float> Meta : register(t2);
StructuredBuffer<float4> AdaptiveState : register(t3);
RWStructuredBuffer<float> GlobalOutput : register(u0);
RWStructuredBuffer<uint4> NextRuleState : register(u1);

#include "include/sbs_scene_controller_constants.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"

float StateWord(float state[SBS_SCENE_RULE_STATE_WORD_COUNT], uint word) {
    return state[word];
}

uint PreviousStateUint(uint word) {
    uint4 value = PreviousRuleState[word / 4u];
    return value[word & 3u];
}

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
    bool previous_schema_valid =
        asfloat(PreviousRuleState[0].x) ==
            (float)SBS_SCENE_SCHEMA_VERSION;
    uint previous_backend_generation =
        PreviousStateUint(SBS_SCENE_RULE_STATE_WORD_BACKEND_GENERATION);
    uint effective_reset_flags = scene_reset_flags;
    if (asfloat(PreviousRuleState[
            SBS_SCENE_RULE_STATE_VECTOR_OUTPUT_VALID].z) <= 0.5f) {
        effective_reset_flags |=
            PreviousStateUint(SBS_SCENE_RULE_STATE_WORD_RESET_FLAGS);
    }
    bool controller_reset =
        !previous_schema_valid ||
        (previous_backend_generation != 0u &&
         previous_backend_generation != scene_backend_generation) ||
        (effective_reset_flags &
            (SBS_SCENE_RESET_FLAGS_LAYOUT |
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
    uint roi_generation =
        controller_reset ?
            0u :
            PreviousStateUint(SBS_SCENE_RULE_STATE_WORD_ROI_GENERATION);
    uint update_count =
        controller_reset ?
            0u :
            PreviousStateUint(SBS_SCENE_RULE_STATE_WORD_UPDATE_COUNT);
    uint last_external_cut_count =
        controller_reset ?
            0u :
            PreviousStateUint(
                SBS_SCENE_RULE_STATE_WORD_LAST_EXTERNAL_CUT_COUNT);

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
        state[SBS_SCENE_RULE_STATE_WORD_RESERVED_1] = 0.0f;
        state[SBS_SCENE_RULE_STATE_WORD_RESERVED_2] = 0.0f;
        state[SBS_SCENE_RULE_STATE_WORD_RESERVED_3] = 0.0f;

        update_count = min(update_count + 1u, 0xfffffffeu);
        uint held_state_flags =
            controller_reset ?
                SBS_SCENE_STATE_FLAGS_INITIALIZED :
                PreviousStateUint(SBS_SCENE_RULE_STATE_WORD_STATE_FLAGS);
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
    bool candidate_valid =
        evidence_valid && candidate_is_roi && candidate_bounds_valid &&
        roi_confidence >= 0.55f && ood_confidence < 0.65f;

    uint state_kind = (uint)round(StateWord(
        state,
        SBS_SCENE_RULE_STATE_WORD_STATE_KIND));
    float4 committed_bounds = float4(
        StateWord(state, SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_X0),
        StateWord(state, SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_Y0),
        StateWord(state, SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_X1),
        StateWord(state, SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_Y1));
    float committed_confidence = saturate(StateWord(
        state,
        SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_CONFIDENCE));
    bool committed_this_update = false;
    uint promotion_flags =
        SBS_SCENE_PROMOTION_FLAGS_LAYOUT_HISTORY |
        SBS_SCENE_PROMOTION_FLAGS_DEPTH_HISTORY;

    // Broad coherent page translation pauses ROI/depth decisions. A player-local pan has low
    // global support because the video evidence suppresses its scroll cells in the evidence pass.
    float scroll_confidence = saturate(
        EvidenceGlobal[
            SBS_SCENE_GLOBAL_OUT_GLOBAL_SCROLL_CONFIDENCE]);
    float scroll_dwell = StateWord(
        state,
        SBS_SCENE_RULE_STATE_WORD_SCROLL_DWELL_S);
    float scroll_hold = StateWord(
        state,
        SBS_SCENE_RULE_STATE_WORD_SCROLL_HOLD_S);
    if (scroll_confidence >= 0.45f) {
        scroll_dwell += max(scene_elapsed_seconds, 0.0f);
        scroll_hold = 0.0f;
        if (scroll_dwell >= scene_scroll_enter_seconds) {
            state_kind = SBS_SCENE_STATE_KIND_SCROLL_HOLD;
        }
    } else {
        scroll_dwell = 0.0f;
        if (state_kind == SBS_SCENE_STATE_KIND_SCROLL_HOLD) {
            scroll_hold += max(scene_elapsed_seconds, 0.0f);
            if (scroll_hold >= 0.12f) {
                uint retained_layout = (uint)round(StateWord(
                    state,
                    SBS_SCENE_RULE_STATE_WORD_COMMITTED_LAYOUT));
                state_kind =
                    retained_layout ==
                        SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO ?
                        SBS_SCENE_STATE_KIND_VIDEO :
                    retained_layout ==
                        SBS_SCENE_LAYOUT_DECISION_CONTENT_COLLAGE ?
                        SBS_SCENE_STATE_KIND_CONTENT :
                        SBS_SCENE_STATE_KIND_FULL_FRAME;
                scroll_hold = 0.0f;
            }
        }
    }

    bool committed_roi =
        state_kind == SBS_SCENE_STATE_KIND_VIDEO ||
        state_kind == SBS_SCENE_STATE_KIND_CONTENT;
    if (state_kind != SBS_SCENE_STATE_KIND_SCROLL_HOLD) {
        if (!committed_roi) {
            if (candidate_valid) {
                float4 acquisition_bounds = float4(
                    StateWord(
                        state,
                        SBS_SCENE_RULE_STATE_WORD_ACQUISITION_ROI_X0),
                    StateWord(
                        state,
                        SBS_SCENE_RULE_STATE_WORD_ACQUISITION_ROI_Y0),
                    StateWord(
                        state,
                        SBS_SCENE_RULE_STATE_WORD_ACQUISITION_ROI_X1),
                    StateWord(
                        state,
                        SBS_SCENE_RULE_STATE_WORD_ACQUISITION_ROI_Y1));
                uint acquisition_layout = (uint)round(StateWord(
                    state,
                    SBS_SCENE_RULE_STATE_WORD_ACQUISITION_LAYOUT));
                bool same_acquisition =
                    StateWord(
                        state,
                        SBS_SCENE_RULE_STATE_WORD_ACQUISITION_VALID) > 0.5f &&
                    acquisition_layout == candidate_layout &&
                    SceneIou(acquisition_bounds, candidate_bounds) >= 0.92f;
                float acquisition_dwell =
                    same_acquisition ?
                        StateWord(
                            state,
                            SBS_SCENE_RULE_STATE_WORD_ACQUISITION_DWELL_S) +
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
                    roi_confidence;
                state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_DWELL_S] =
                    acquisition_dwell;
                state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_LAYOUT] =
                    (float)candidate_layout;
                state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_VALID] = 1.0f;
                if (acquisition_dwell >= scene_acquire_seconds) {
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
                    state[
                        SBS_SCENE_RULE_STATE_WORD_ACQUISITION_VALID] = 0.0f;
                    roi_generation =
                        min(roi_generation + 1u, 0xfffffffeu);
                    promotion_flags |= SBS_SCENE_PROMOTION_FLAGS_ROI;
                    committed_this_update = true;
                }
            } else {
                state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_VALID] = 0.0f;
                state[SBS_SCENE_RULE_STATE_WORD_ACQUISITION_DWELL_S] =
                    0.0f;
            }
        } else {
            bool candidate_is_incumbent =
                candidate_valid &&
                ((state_kind == SBS_SCENE_STATE_KIND_VIDEO &&
                  candidate_layout ==
                      SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO) ||
                 (state_kind == SBS_SCENE_STATE_KIND_CONTENT &&
                  candidate_layout ==
                      SBS_SCENE_LAYOUT_DECISION_CONTENT_COLLAGE)) &&
                SceneIou(committed_bounds, candidate_bounds) >= 0.60f;
            if (candidate_is_incumbent) {
                // Stable geometry is intentionally anchored. A candidate must become a decisive
                // challenger before its rectangle can move, preventing one-cell ROI breathing.
                committed_confidence =
                    lerp(committed_confidence, roi_confidence, 0.20f);
                state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_AGE_S] +=
                    max(scene_elapsed_seconds, 0.0f);
                state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_VALID] = 0.0f;
                state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_DWELL_S] =
                    0.0f;
                state[
                    SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE] =
                    0.0f;
            } else if (
                candidate_valid &&
                roi_confidence >=
                    max(committed_confidence * 1.15f, 0.65f)
            ) {
                float4 challenger_bounds = float4(
                    StateWord(
                        state,
                        SBS_SCENE_RULE_STATE_WORD_CHALLENGER_ROI_X0),
                    StateWord(
                        state,
                        SBS_SCENE_RULE_STATE_WORD_CHALLENGER_ROI_Y0),
                    StateWord(
                        state,
                        SBS_SCENE_RULE_STATE_WORD_CHALLENGER_ROI_X1),
                    StateWord(
                        state,
                        SBS_SCENE_RULE_STATE_WORD_CHALLENGER_ROI_Y1));
                uint challenger_layout = (uint)round(StateWord(
                    state,
                    SBS_SCENE_RULE_STATE_WORD_CHALLENGER_LAYOUT));
                bool same_challenger =
                    StateWord(
                        state,
                        SBS_SCENE_RULE_STATE_WORD_CHALLENGER_VALID) > 0.5f &&
                    challenger_layout == candidate_layout &&
                    SceneIou(challenger_bounds, candidate_bounds) >= 0.92f;
                float challenger_dwell =
                    same_challenger ?
                        StateWord(
                            state,
                            SBS_SCENE_RULE_STATE_WORD_CHALLENGER_DWELL_S) +
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
                    roi_confidence;
                state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_DWELL_S] =
                    challenger_dwell;
                state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_LAYOUT] =
                    (float)candidate_layout;
                state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_VALID] = 1.0f;
                if (challenger_dwell >= scene_challenger_seconds) {
                    state_kind =
                        candidate_layout ==
                            SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO ?
                            SBS_SCENE_STATE_KIND_VIDEO :
                            SBS_SCENE_STATE_KIND_CONTENT;
                    committed_bounds = candidate_bounds;
                    committed_confidence = roi_confidence;
                    state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_LAYOUT] =
                        (float)candidate_layout;
                    state[
                        SBS_SCENE_RULE_STATE_WORD_COMMITTED_MASK_CONFIDENCE] =
                        mask_confidence;
                    state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_AGE_S] =
                        0.0f;
                    state[SBS_SCENE_RULE_STATE_WORD_CHALLENGER_VALID] =
                        0.0f;
                    roi_generation =
                        min(roi_generation + 1u, 0xfffffffeu);
                    promotion_flags |= SBS_SCENE_PROMOTION_FLAGS_ROI;
                    committed_this_update = true;
                }
            } else {
                // A quiet/paused player remains locked. Only a decisive challenger or the long
                // release ceiling may replace it; an animated sidebar cannot win by persistence.
                float missing = StateWord(
                    state,
                    SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE);
                missing += max(scene_elapsed_seconds, 0.0f);
                state[
                    SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE] =
                    missing;
                if (missing >= scene_release_seconds) {
                    state_kind = SBS_SCENE_STATE_KIND_FULL_FRAME;
                    committed_bounds = float4(0.0f, 0.0f, 1.0f, 1.0f);
                    committed_confidence = 0.0f;
                    state[SBS_SCENE_RULE_STATE_WORD_COMMITTED_LAYOUT] =
                        (float)SBS_SCENE_LAYOUT_DECISION_NO_TARGET;
                    roi_generation =
                        min(roi_generation + 1u, 0xfffffffeu);
                    promotion_flags |= SBS_SCENE_PROMOTION_FLAGS_ROI;
                    committed_this_update = true;
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
    bool decisions_frozen =
        state_kind == SBS_SCENE_STATE_KIND_SCROLL_HOLD;
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
        state_kind == SBS_SCENE_STATE_KIND_CONTENT ||
        state_kind == SBS_SCENE_STATE_KIND_SCROLL_HOLD
    ) {
        state_flags |= SBS_SCENE_STATE_FLAGS_ROI_LOCKED;
    }
    if (state_kind == SBS_SCENE_STATE_KIND_SCROLL_HOLD) {
        state_flags |= SBS_SCENE_STATE_FLAGS_SCROLL_HOLD_ACTIVE;
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
        committed_confidence;
    state[SBS_SCENE_RULE_STATE_WORD_RUNNER_UP_SCORE] =
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_39];
    state[SBS_SCENE_RULE_STATE_WORD_WINNER_MARGIN] =
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_MASK];
    state[SBS_SCENE_RULE_STATE_WORD_CANDIDATE_COVERAGE] =
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_MASK];
    state[SBS_SCENE_RULE_STATE_WORD_SCROLL_X] =
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_GLOBAL_SCROLL_X];
    state[SBS_SCENE_RULE_STATE_WORD_SCROLL_Y] =
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_GLOBAL_SCROLL_Y];
    state[SBS_SCENE_RULE_STATE_WORD_SCROLL_CONFIDENCE] =
        scroll_confidence;
    state[SBS_SCENE_RULE_STATE_WORD_SCROLL_DWELL_S] = scroll_dwell;
    state[SBS_SCENE_RULE_STATE_WORD_SCROLL_HOLD_S] = scroll_hold;
    if (!decisions_frozen) {
        state[SBS_SCENE_RULE_STATE_WORD_LAYOUT_DECISION] =
            (float)candidate_layout;
        state[SBS_SCENE_RULE_STATE_WORD_LAYOUT_CONFIDENCE] =
            roi_confidence;

        uint event_decision =
            committed_this_update ?
                SBS_SCENE_EVENT_DECISION_GEOMETRY_RESET :
                EvidenceEvent();
        state[SBS_SCENE_RULE_STATE_WORD_EVENT_DECISION] =
            (float)event_decision;
        state[SBS_SCENE_RULE_STATE_WORD_EVENT_CONFIDENCE] =
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
    if (!decisions_frozen) {
        uint current_external_cut_count = asuint(
            SBS_STATE_EXTERNAL_CUT_COUNT(
                AdaptiveState[SBS_STATE_VECTOR_EXTERNAL_CUT_COUNT]));
        last_external_cut_count =
            max(last_external_cut_count, current_external_cut_count);
    }
    if (!evidence_valid) {
        promotion_flags &=
            ~(SBS_SCENE_PROMOTION_FLAGS_LAYOUT_HISTORY |
              SBS_SCENE_PROMOTION_FLAGS_DEPTH_HISTORY);
    } else if (state_kind == SBS_SCENE_STATE_KIND_SCROLL_HOLD) {
        // Keep the appearance endpoint and motion channels moving so cumulative scrolling
        // remains measurable. The history-commit pass freezes all semantic layout channels and
        // the entire depth bank while the hold is active.
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
    state[SBS_SCENE_RULE_STATE_WORD_RESERVED_1] = 0.0f;
    state[SBS_SCENE_RULE_STATE_WORD_RESERVED_2] = 0.0f;
    state[SBS_SCENE_RULE_STATE_WORD_RESERVED_3] = 0.0f;

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
    if (committed_this_update) {
        [unroll]
        for (uint decision = 0u; decision < 6u; ++decision) {
            GlobalOutput[
                SBS_SCENE_GLOBAL_OUT_EVENT_SAME_SHOT + decision] =
                SceneLogit(
                    decision == SBS_SCENE_EVENT_DECISION_GEOMETRY_RESET ?
                        0.995f :
                        0.02f);
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
        NextRuleState[vector_index] = resolved_bits;
    }
}
