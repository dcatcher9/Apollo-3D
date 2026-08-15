// Live Host SBS V2 cut-state resolver. It writes the authenticated 32-word CutBridgeState
// transport consumed by telemetry, Dump 3D, and offline boundary planning. Reserved compatibility
// slots retain their declared defaults and are never consumed by V2 geometry.

RWStructuredBuffer<float4> CutBridgeState : register(u0);
RWStructuredBuffer<uint> SceneCutEvidence : register(u1);

#include "include/depth_constants.hlsl"
#include "include/depth_scene_cut_evidence_contract.hlsl"
#include "include/depth_scene_cut_state_contract.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"

[numthreads(1, 1, 1)]
void main() {
    float4 s = CutBridgeState[SBS_STATE_VECTOR_SCENE_AGE];
    float4 s1 = CutBridgeState[SBS_STATE_VECTOR_DEPTH_CHANGE_BASELINE_EMA];
    float4 s2 = CutBridgeState[SBS_STATE_VECTOR_CUT_FLAGS];
    float4 telemetry = CutBridgeState[SBS_STATE_VECTOR_CURRENT_DEPTH_CHANGE_FRACTION];
    uint4 health_counters = asuint(CutBridgeState[SBS_STATE_VECTOR_HARD_CUT_COUNT]);
    float4 telemetry_flags = CutBridgeState[SBS_STATE_VECTOR_RANGE_COLLAPSED];
    if (asuint(SBS_STATE_CUT_CONTRACT_TAG_BITS(s)) != SBS_CUT_CONTRACT_TAG) {
        s = SbsAdaptiveStateInitialVector(0u);
        s1 = SbsAdaptiveStateInitialVector(1u);
        s2 = SbsAdaptiveStateInitialVector(2u);
        telemetry = SbsAdaptiveStateInitialVector(3u);
        health_counters = asuint(SbsAdaptiveStateInitialVector(4u));
        telemetry_flags = SbsAdaptiveStateInitialVector(5u);
    }
    SBS_STATE_CUT_CONTRACT_TAG_BITS(s) = asfloat(SBS_CUT_CONTRACT_TAG);
    float4 current_diagnostics = 0.0f;
    SBS_STATE_STRUCTURAL_CHANGE_FRACTION(current_diagnostics) = -1.0f;
    SBS_STATE_RAW_RGB_CHANGE_FRACTION(current_diagnostics) = -1.0f;
    float4 analysis_diagnostics = 0.0f;
    SBS_STATE_CURRENT_STRUCTURAL_SUPPORT_FRACTION(analysis_diagnostics) = -1.0f;
    SBS_STATE_PREVIOUS_STRUCTURAL_SUPPORT_FRACTION(analysis_diagnostics) = -1.0f;
    SBS_STATE_COMMON_STRUCTURAL_SUPPORT_FRACTION(analysis_diagnostics) = -1.0f;
    SBS_STATE_CURRENT_DEPTH_CHANGE_FRACTION(telemetry) = 0.0f;
    SBS_STATE_HARD_CUT_PULSE(telemetry_flags) = 0.0f;

    float total = (float)SceneCutEvidence[CUT_EVIDENCE_TOTAL];
    if (total > 0.5f) {
        bool initialized = SBS_STATE_INITIALIZED(s) > 0.5f;
        float previous_scene_age = SBS_STATE_SCENE_AGE(s);
        uint stream_frame_delta = clamp(
            SceneCutEvidence[CUT_EVIDENCE_STREAM_FRAME_DELTA], 1u, 65535u);
        float scene_age = initialized ?
            min(previous_scene_age + (float)stream_frame_delta, 65535.0f) : 0.0f;
        float change_fraction =
            (float)SceneCutEvidence[CUT_EVIDENCE_DEPTH_CHANGE] / total;
        float structural_change_fraction =
            (float)SceneCutEvidence[CUT_EVIDENCE_STRUCTURAL_CHANGE] / total;
        uint raw_rgb_change_count = SceneCutEvidence[CUT_EVIDENCE_RAW_RGB_CHANGE];
        float raw_rgb_change_fraction = (float)raw_rgb_change_count / total;
        float current_structural_support_fraction =
            (float)SceneCutEvidence[CUT_EVIDENCE_CURRENT_STRUCTURAL_SUPPORT] / total;
        float previous_structural_support_fraction =
            (float)SceneCutEvidence[CUT_EVIDENCE_PREVIOUS_STRUCTURAL_SUPPORT] / total;
        float common_structural_support_fraction =
            (float)SceneCutEvidence[CUT_EVIDENCE_COMMON_STRUCTURAL_SUPPORT] / total;
        uint brightness_rise_count = SceneCutEvidence[CUT_EVIDENCE_BRIGHTNESS_RISE];
        uint brightness_fall_count = SceneCutEvidence[CUT_EVIDENCE_BRIGHTNESS_FALL];

        bool model_input_history_valid =
            SBS_STATE_MODEL_INPUT_HISTORY_STATE(s2) > 0.5f;
        SBS_STATE_CURRENT_STRUCTURAL_SUPPORT_FRACTION(analysis_diagnostics) =
            current_structural_support_fraction;
        SBS_STATE_PREVIOUS_STRUCTURAL_SUPPORT_FRACTION(analysis_diagnostics) =
            model_input_history_valid ? previous_structural_support_fraction : -1.0f;
        SBS_STATE_COMMON_STRUCTURAL_SUPPORT_FRACTION(analysis_diagnostics) =
            model_input_history_valid ? common_structural_support_fraction : -1.0f;
        SBS_STATE_STRUCTURAL_CHANGE_FRACTION(current_diagnostics) =
            model_input_history_valid ? structural_change_fraction : -1.0f;
        SBS_STATE_RAW_RGB_CHANGE_FRACTION(current_diagnostics) =
            model_input_history_valid ? raw_rgb_change_fraction : -1.0f;

        bool model_input_history_gap =
            SBS_STATE_MODEL_INPUT_HISTORY_STATE(s2) > 1.5f &&
            SBS_STATE_MODEL_INPUT_HISTORY_STATE(s2) < 2.5f;
        bool low_structure_scene =
            SBS_STATE_MODEL_INPUT_HISTORY_STATE(s2) > 2.5f &&
            SBS_STATE_MODEL_INPUT_HISTORY_STATE(s2) < 3.5f;
        bool current_structure_reliable =
            current_structural_support_fraction >= STRUCTURAL_COLOR_MIN_SUPPORT;
        bool previous_structure_reliable =
            previous_structural_support_fraction >= STRUCTURAL_COLOR_MIN_SUPPORT;
        bool common_structure_reliable =
            common_structural_support_fraction >= STRUCTURAL_COLOR_MIN_SUPPORT;
        bool common_structure_representative =
            common_structure_reliable &&
            common_structural_support_fraction >=
                STRUCTURAL_COLOR_EXPOSURE_MIN_COMMON_RATIO *
                min(current_structural_support_fraction,
                    previous_structural_support_fraction);
        float appearance_change_baseline = initialized ?
            saturate(SBS_STATE_APPEARANCE_CHANGE_BASELINE_EMA(telemetry_flags)) :
            0.0f;

        bool broad_appearance_proposal =
            model_input_history_valid &&
            raw_rgb_change_fraction >= RAW_RGB_CUT_HIGH &&
            structural_change_fraction >= STRUCTURAL_COLOR_CUT_HIGH;
        bool localized_appearance_proposal =
            model_input_history_valid &&
            current_structure_reliable &&
            previous_structure_reliable &&
            raw_rgb_change_fraction >= LOCALIZED_RGB_CUT_FLOOR &&
            structural_change_fraction >= STRUCTURAL_COLOR_CUT_HIGH &&
            raw_rgb_change_fraction >=
                appearance_change_baseline + LOCALIZED_RGB_CUT_BASELINE_MARGIN &&
            raw_rgb_change_fraction >=
                max(LOCALIZED_RGB_CUT_FLOOR,
                    appearance_change_baseline *
                    LOCALIZED_RGB_CUT_BASELINE_MULTIPLIER);
        bool appearance_proposal =
            broad_appearance_proposal || localized_appearance_proposal;
        bool broad_rgb_transition =
            model_input_history_valid &&
            raw_rgb_change_fraction >= RAW_RGB_CUT_HIGH;
        bool brightness_direction_consistent =
            raw_rgb_change_count > 0u &&
            (float)max(brightness_rise_count, brightness_fall_count) >=
                STRUCTURAL_COLOR_EXPOSURE_MIN_DIRECTIONAL_SHARE *
                (float)raw_rgb_change_count;
        bool exposure_structure_preserved =
            structural_change_fraction < STRUCTURAL_COLOR_EXPOSURE_QUIET ||
            (structural_change_fraction <=
                 STRUCTURAL_COLOR_EXPOSURE_MAX_FLIP_TO_RGB_RATIO *
                 raw_rgb_change_fraction &&
             brightness_direction_consistent);
        bool exposure_like_transition =
            broad_rgb_transition &&
            current_structure_reliable &&
            previous_structure_reliable &&
            common_structure_representative &&
            exposure_structure_preserved;

        bool structureless_candidate =
            model_input_history_valid &&
            previous_structure_reliable &&
            !current_structure_reliable;
        bool structureless_transition =
            structureless_candidate && !model_input_history_gap;
        bool persistent_structureless_transition =
            structureless_candidate && model_input_history_gap;
        bool same_scene_gap_return =
            model_input_history_gap &&
            current_structure_reliable &&
            raw_rgb_change_fraction < STRUCTURELESS_RETURN_RGB_SAME_MAX &&
            structural_change_fraction < STRUCTURAL_COLOR_EXPOSURE_QUIET;
        bool base_appearance_veto =
            exposure_like_transition ||
            structureless_transition ||
            same_scene_gap_return;
        bool photometric_recovery_veto =
            exposure_like_transition || same_scene_gap_return;

        uint cut_flags = (uint)max(SBS_STATE_CUT_FLAGS(s2), 0.0f);
        bool geometry_armed = (cut_flags & CUT_FLAG_GEOMETRY_ARMED) != 0u;
        bool appearance_armed = (cut_flags & CUT_FLAG_APPEARANCE_ARMED) != 0u;
        bool cut_latched = (cut_flags & CUT_FLAG_LATCHED) != 0u;
        bool appearance_recovery =
            (cut_flags & CUT_FLAG_APPEARANCE_RECOVERY) != 0u;
        bool geometry_confirmation_pending =
            (cut_flags & CUT_FLAG_GEOMETRY_CONFIRMATION_PENDING) != 0u;
        bool appearance_recovery_tail =
            appearance_recovery && !appearance_proposal;
        bool appearance_veto = base_appearance_veto || appearance_recovery_tail;
        float depth_change_baseline = initialized ?
            saturate(SBS_STATE_DEPTH_CHANGE_BASELINE_EMA(s1)) :
            change_fraction;

        bool relative_geometry_spike =
            cut_latched && !geometry_armed &&
            !appearance_veto &&
            scene_age >= CUT_ARM_SETTLE_STREAM_FRAMES &&
            change_fraction >= DEPTH_CUT_RELATIVE_FLOOR &&
            (change_fraction >= depth_change_baseline + DEPTH_CUT_RELATIVE_MARGIN ||
             change_fraction >=
                 depth_change_baseline * DEPTH_CUT_RELATIVE_MULTIPLIER);
        bool immediate_appearance_cut =
            appearance_armed &&
            appearance_proposal &&
            !appearance_veto &&
            change_fraction >= DEPTH_CUT_CORROBORATE;
        // A structureless reference cannot corroborate structure by construction: every ordinal
        // pair against a uniform slate has zero contrast, so structural change is exactly zero
        // no matter how different the returning scene is. Requiring it here made the return cut
        // after a bridged flash/slate unreachable and held the slate-acquired camera forever.
        // The waiver is confined to the structureless-bridge scene state and the return arm
        // still demands reliable current structure plus the full geometry-change bar.
        bool reference_structureless =
            model_input_history_valid && !previous_structure_reliable;
        bool geometry_structure_corroborated =
            persistent_structureless_transition ||
            reference_structureless ||
            structural_change_fraction >= STRUCTURAL_GEOMETRY_CUT_FLOOR;
        bool geometry_confirmation_candidate =
            !appearance_veto &&
            geometry_structure_corroborated &&
            ((geometry_armed && change_fraction >= DEPTH_CUT_HIGH) ||
             (low_structure_scene && current_structure_reliable &&
              change_fraction >= DEPTH_CUT_HIGH) ||
             (geometry_confirmation_pending && current_structure_reliable &&
              change_fraction >= DEPTH_CUT_HIGH) ||
             relative_geometry_spike);
        bool structureless_candidate_already_confirmed =
            persistent_structureless_transition &&
            geometry_confirmation_candidate;
        bool confirmed_geometry_cut =
            geometry_confirmation_pending &&
            geometry_confirmation_candidate;
        bool shot_cut =
            initialized &&
            (immediate_appearance_cut ||
             structureless_candidate_already_confirmed ||
             confirmed_geometry_cut);
        bool start_geometry_confirmation =
            initialized &&
            !shot_cut &&
            !geometry_confirmation_pending &&
            geometry_confirmation_candidate &&
            !persistent_structureless_transition;

        uint analysis_flags =
            (appearance_proposal ? ANALYSIS_FLAG_APPEARANCE_PROPOSAL : 0u) |
            (exposure_like_transition ? ANALYSIS_FLAG_EXPOSURE_LIKE : 0u) |
            ((structureless_transition || persistent_structureless_transition) ?
                ANALYSIS_FLAG_STRUCTURELESS : 0u) |
            (same_scene_gap_return ? ANALYSIS_FLAG_SAME_RETURN : 0u) |
            (appearance_veto ? ANALYSIS_FLAG_VETO : 0u) |
            (relative_geometry_spike ? ANALYSIS_FLAG_RELATIVE_SPIKE : 0u) |
            (localized_appearance_proposal ?
                ANALYSIS_FLAG_LOCALIZED_APPEARANCE_PROPOSAL : 0u) |
            (geometry_confirmation_candidate ?
                ANALYSIS_FLAG_GEOMETRY_CONFIRMATION_CANDIDATE : 0u);
        SBS_STATE_ANALYSIS_FLAGS(analysis_diagnostics) = (float)analysis_flags;

        if (!initialized) {
            scene_age = 0.0f;
            cut_flags = 0u;
            depth_change_baseline = change_fraction;
            appearance_change_baseline = 0.0f;
        } else if (shot_cut) {
            scene_age = 0.0f;
            cut_flags = CUT_FLAG_LATCHED;
            depth_change_baseline = change_fraction;
            appearance_change_baseline = 0.0f;
        } else {
            if (!structureless_transition &&
                !appearance_recovery_tail &&
                !start_geometry_confirmation) {
                depth_change_baseline =
                    lerp(depth_change_baseline,
                         change_fraction,
                         DEPTH_CUT_BASELINE_ALPHA);
            }
            if (model_input_history_valid && !start_geometry_confirmation) {
                appearance_change_baseline =
                    lerp(appearance_change_baseline,
                         raw_rgb_change_fraction,
                         APPEARANCE_CUT_BASELINE_ALPHA);
            }
            if (!cut_latched) {
                if (scene_age >= CUT_ARM_SETTLE_STREAM_FRAMES) {
                    cut_flags = CUT_FLAG_GEOMETRY_ARMED | CUT_FLAG_APPEARANCE_ARMED;
                }
            } else {
                if (!geometry_armed) {
                    if (change_fraction < DEPTH_CUT_LOW) {
                        if ((cut_flags & CUT_FLAG_GEOMETRY_LOW_ONCE) != 0u) {
                            cut_flags |= CUT_FLAG_GEOMETRY_ARMED;
                            cut_flags &= ~CUT_FLAG_GEOMETRY_LOW_ONCE;
                        } else {
                            cut_flags |= CUT_FLAG_GEOMETRY_LOW_ONCE;
                        }
                    } else {
                        cut_flags &= ~CUT_FLAG_GEOMETRY_LOW_ONCE;
                    }
                }
                if (!appearance_armed) {
                    if (!appearance_proposal) {
                        if ((cut_flags & CUT_FLAG_APPEARANCE_QUIET_ONCE) != 0u) {
                            cut_flags |= CUT_FLAG_APPEARANCE_ARMED;
                            cut_flags &= ~CUT_FLAG_APPEARANCE_QUIET_ONCE;
                        } else {
                            cut_flags |= CUT_FLAG_APPEARANCE_QUIET_ONCE;
                        }
                    } else {
                        cut_flags &= ~CUT_FLAG_APPEARANCE_QUIET_ONCE;
                    }
                }
            }
            if (photometric_recovery_veto) {
                cut_flags |= CUT_FLAG_APPEARANCE_RECOVERY;
            } else {
                cut_flags &= ~CUT_FLAG_APPEARANCE_RECOVERY;
            }
            cut_flags &= ~CUT_FLAG_GEOMETRY_CONFIRMATION_PENDING;
            if (start_geometry_confirmation) {
                cut_flags |= CUT_FLAG_GEOMETRY_CONFIRMATION_PENDING;
            }
        }

        float next_model_input_history_state =
            start_geometry_confirmation ? 4.0f :
            (structureless_transition ? 2.0f :
             ((persistent_structureless_transition ||
               (low_structure_scene && !current_structure_reliable)) ? 3.0f :
              1.0f));

        SBS_STATE_SCENE_AGE(s) = scene_age;
        SBS_STATE_INITIALIZED(s) = 1.0f;
        SBS_STATE_DEPTH_CHANGE_BASELINE_EMA(s1) = depth_change_baseline;
        SBS_STATE_CUT_FLAGS(s2) = (float)cut_flags;
        SBS_STATE_MODEL_INPUT_HISTORY_STATE(s2) = next_model_input_history_state;
        SBS_STATE_CURRENT_DEPTH_CHANGE_FRACTION(telemetry) = change_fraction;
        SBS_STATE_APPEARANCE_CHANGE_BASELINE_EMA(telemetry_flags) =
            appearance_change_baseline;
        if (shot_cut) {
            SBS_STATE_HARD_CUT_COUNT(health_counters) =
                min(SBS_STATE_HARD_CUT_COUNT(health_counters) + 1u,
                    SBS_ADAPTIVE_STATE_COUNTER_MAX);
            SBS_STATE_HARD_CUT_PULSE(telemetry_flags) = 1.0f;
        }
    }

    CutBridgeState[SBS_STATE_VECTOR_SCENE_AGE] = s;
    CutBridgeState[SBS_STATE_VECTOR_DEPTH_CHANGE_BASELINE_EMA] = s1;
    CutBridgeState[SBS_STATE_VECTOR_CUT_FLAGS] = s2;
    CutBridgeState[SBS_STATE_VECTOR_CURRENT_DEPTH_CHANGE_FRACTION] = telemetry;
    CutBridgeState[SBS_STATE_VECTOR_HARD_CUT_COUNT] = asfloat(health_counters);
    CutBridgeState[SBS_STATE_VECTOR_RANGE_COLLAPSED] = telemetry_flags;
    CutBridgeState[SBS_STATE_VECTOR_STRUCTURAL_CHANGE_FRACTION] = current_diagnostics;
    CutBridgeState[SBS_STATE_VECTOR_CURRENT_STRUCTURAL_SUPPORT_FRACTION] =
        analysis_diagnostics;

    [unroll]
    for (uint index = 0u; index < CUT_EVIDENCE_WORD_COUNT; ++index) {
        SceneCutEvidence[index] = 0u;
    }
}
