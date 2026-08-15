// Optional, shadow-only current-frame motion evidence. This root is deliberately outside the
// authenticated Host SBS producer closure and has no geometry, history, or inference authority.

StructuredBuffer<float> CurrentModelInput : register(t0);
StructuredBuffer<float> PreviousModelInput : register(t1);
StructuredBuffer<float> CurrentAppearanceOrdinal : register(t2);
StructuredBuffer<float> PreviousAppearanceOrdinal : register(t3);
Texture2D<uint> CurrentTensorExclusion : register(t4);
Texture2D<uint> PreviousTensorExclusion : register(t5);
StructuredBuffer<float4> CutBridgeState : register(t6);
RWStructuredBuffer<uint> ProbeWords : register(u0);

#include "include/depth_constants.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"

cbuffer ProbeConstants : register(b1) {
    uint probe_current_frame_low;
    uint probe_current_frame_high;
    uint probe_baseline_frame_low;
    uint probe_baseline_frame_high;
    uint probe_bottom_top;
    uint probe_bottom_bottom;
    uint2 probe_reserved;
};

#define MOTION_PROBE_CONTRACT_TAG 0x314D4643u
#define MOTION_PROBE_MIN_SCENE_AGE 8u
#define MOTION_PROBE_MAX_EXACT_NUMERIC_COUNTER 16777215.0f
#define MOTION_PROBE_KNOWN_CUT_FLAGS \
    (CUT_FLAG_GEOMETRY_ARMED | CUT_FLAG_APPEARANCE_ARMED | \
     CUT_FLAG_GEOMETRY_LOW_ONCE | CUT_FLAG_APPEARANCE_QUIET_ONCE | \
     CUT_FLAG_LATCHED | CUT_FLAG_APPEARANCE_RECOVERY | \
     CUT_FLAG_GEOMETRY_CONFIRMATION_PENDING)
#define MOTION_PROBE_KNOWN_ANALYSIS_FLAGS \
    (ANALYSIS_FLAG_APPEARANCE_PROPOSAL | ANALYSIS_FLAG_EXPOSURE_LIKE | \
     ANALYSIS_FLAG_STRUCTURELESS | ANALYSIS_FLAG_SAME_RETURN | ANALYSIS_FLAG_VETO | \
     ANALYSIS_FLAG_RELATIVE_SPIKE | ANALYSIS_FLAG_LOCALIZED_APPEARANCE_PROPOSAL | \
     ANALYSIS_FLAG_GEOMETRY_CONFIRMATION_CANDIDATE)

#define PROBE_WORD_CONTRACT_TAG 0u
#define PROBE_WORD_PRIOR_STATE_FLAGS 1u
#define PROBE_WORD_CURRENT_FRAME_LOW 2u
#define PROBE_WORD_CURRENT_FRAME_HIGH 3u
#define PROBE_WORD_BASELINE_FRAME_LOW 4u
#define PROBE_WORD_BASELINE_FRAME_HIGH 5u
#define PROBE_WORD_HARD_CUT_COUNT 6u
#define PROBE_WORD_SCENE_AGE 7u
#define PROBE_WORD_CUT_FLAGS 8u
#define PROBE_WORD_ANALYSIS_FLAGS 9u
#define PROBE_WORD_HISTORY_STATE 10u
#define PROBE_WORD_ADMITTED 11u
#define PROBE_WORD_EXCLUSION_MISMATCH 12u
#define PROBE_WORD_EXACT_CHANGED 13u
#define PROBE_WORD_RGB_1_OVER_1024 14u
#define PROBE_WORD_RGB_1_OVER_256 15u
#define PROBE_WORD_RGB_1_OVER_64 16u
#define PROBE_WORD_MAX_RGB_DELTA_BITS 17u
#define PROBE_WORD_MAX_TILE_EXACT_CHANGED 18u
#define PROBE_WORD_APPEARANCE_1_OVER_1024 19u
#define PROBE_WORD_MAX_APPEARANCE_DELTA_BITS 20u
#define PROBE_WORD_BOTTOM_ADMITTED 21u
#define PROBE_WORD_BOTTOM_EXACT_CHANGED 22u
#define PROBE_WORD_BOTTOM_RGB_1_OVER_1024 23u
#define PROBE_WORD_BOTTOM_MAX_RGB_DELTA_BITS 24u

#define LOCAL_ADMITTED 0u
#define LOCAL_EXCLUSION_MISMATCH 1u
#define LOCAL_EXACT_CHANGED 2u
#define LOCAL_RGB_1_OVER_1024 3u
#define LOCAL_RGB_1_OVER_256 4u
#define LOCAL_RGB_1_OVER_64 5u
#define LOCAL_MAX_RGB_DELTA_BITS 6u
#define LOCAL_APPEARANCE_1_OVER_1024 7u
#define LOCAL_MAX_APPEARANCE_DELTA_BITS 8u
#define LOCAL_BOTTOM_ADMITTED 9u
#define LOCAL_BOTTOM_EXACT_CHANGED 10u
#define LOCAL_BOTTOM_RGB_1_OVER_1024 11u
#define LOCAL_BOTTOM_MAX_RGB_DELTA_BITS 12u
#define LOCAL_WORD_COUNT 13u

groupshared uint GroupWords[LOCAL_WORD_COUNT];

bool ProbeCanonicalBoolean(float value) {
    return isfinite(value) && (value == 0.0f || value == 1.0f);
}

bool ProbeFiniteWholeInRange(float value, float maximum) {
    return isfinite(value) && value >= 0.0f && value <= maximum &&
           value == floor(value);
}

float3 CurrentModelColor(uint index, uint plane) {
    return float3(
        CurrentModelInput[index] * 0.229f + 0.485f,
        CurrentModelInput[index + plane] * 0.224f + 0.456f,
        CurrentModelInput[index + 2u * plane] * 0.225f + 0.406f);
}

float3 PreviousModelColor(uint index, uint plane) {
    return float3(
        PreviousModelInput[index] * 0.229f + 0.485f,
        PreviousModelInput[index + plane] * 0.224f + 0.456f,
        PreviousModelInput[index + 2u * plane] * 0.225f + 0.406f);
}

[numthreads(16, 16, 1)]
void main(
    uint3 dtid : SV_DispatchThreadID,
    uint3 tid : SV_GroupThreadID,
    uint3 gid : SV_GroupID) {
    uint linear_thread = tid.y * 16u + tid.x;
    if (linear_thread < LOCAL_WORD_COUNT) {
        GroupWords[linear_thread] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    if (dtid.x < target_w && dtid.y < target_h) {
        uint current_exclusion = CurrentTensorExclusion[dtid.xy];
        uint previous_exclusion = PreviousTensorExclusion[dtid.xy];
        if (current_exclusion != previous_exclusion) {
            InterlockedAdd(GroupWords[LOCAL_EXCLUSION_MISMATCH], 1u);
        }

        if (current_exclusion == 0u && previous_exclusion == 0u) {
            uint plane = target_w * target_h;
            uint index = dtid.y * target_w + dtid.x;
            bool exact_changed =
                asuint(CurrentModelInput[index]) != asuint(PreviousModelInput[index]) ||
                asuint(CurrentModelInput[index + plane]) !=
                    asuint(PreviousModelInput[index + plane]) ||
                asuint(CurrentModelInput[index + 2u * plane]) !=
                    asuint(PreviousModelInput[index + 2u * plane]);
            float3 current_color = CurrentModelColor(index, plane);
            float3 previous_color = PreviousModelColor(index, plane);
            float3 rgb_delta = abs(current_color - previous_color);
            float maximum_rgb_delta = max(rgb_delta.r, max(rgb_delta.g, rgb_delta.b));
            float appearance_delta = abs(
                CurrentAppearanceOrdinal[index] - PreviousAppearanceOrdinal[index]);
            bool bottom_band = dtid.y >= probe_bottom_top &&
                               dtid.y < probe_bottom_bottom;

            InterlockedAdd(GroupWords[LOCAL_ADMITTED], 1u);
            if (exact_changed) {
                InterlockedAdd(GroupWords[LOCAL_EXACT_CHANGED], 1u);
            }
            if (maximum_rgb_delta >= (1.0f / 1024.0f)) {
                InterlockedAdd(GroupWords[LOCAL_RGB_1_OVER_1024], 1u);
            }
            if (maximum_rgb_delta >= (1.0f / 256.0f)) {
                InterlockedAdd(GroupWords[LOCAL_RGB_1_OVER_256], 1u);
            }
            if (maximum_rgb_delta >= (1.0f / 64.0f)) {
                InterlockedAdd(GroupWords[LOCAL_RGB_1_OVER_64], 1u);
            }
            InterlockedMax(GroupWords[LOCAL_MAX_RGB_DELTA_BITS], asuint(maximum_rgb_delta));
            if (appearance_delta >= (1.0f / 1024.0f)) {
                InterlockedAdd(GroupWords[LOCAL_APPEARANCE_1_OVER_1024], 1u);
            }
            InterlockedMax(
                GroupWords[LOCAL_MAX_APPEARANCE_DELTA_BITS], asuint(appearance_delta));

            if (bottom_band) {
                InterlockedAdd(GroupWords[LOCAL_BOTTOM_ADMITTED], 1u);
                if (exact_changed) {
                    InterlockedAdd(GroupWords[LOCAL_BOTTOM_EXACT_CHANGED], 1u);
                }
                if (maximum_rgb_delta >= (1.0f / 1024.0f)) {
                    InterlockedAdd(GroupWords[LOCAL_BOTTOM_RGB_1_OVER_1024], 1u);
                }
                InterlockedMax(
                    GroupWords[LOCAL_BOTTOM_MAX_RGB_DELTA_BITS],
                    asuint(maximum_rgb_delta));
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if (linear_thread == 0u) {
        InterlockedAdd(ProbeWords[PROBE_WORD_ADMITTED], GroupWords[LOCAL_ADMITTED]);
        InterlockedAdd(
            ProbeWords[PROBE_WORD_EXCLUSION_MISMATCH],
            GroupWords[LOCAL_EXCLUSION_MISMATCH]);
        InterlockedAdd(
            ProbeWords[PROBE_WORD_EXACT_CHANGED], GroupWords[LOCAL_EXACT_CHANGED]);
        InterlockedAdd(
            ProbeWords[PROBE_WORD_RGB_1_OVER_1024], GroupWords[LOCAL_RGB_1_OVER_1024]);
        InterlockedAdd(
            ProbeWords[PROBE_WORD_RGB_1_OVER_256], GroupWords[LOCAL_RGB_1_OVER_256]);
        InterlockedAdd(
            ProbeWords[PROBE_WORD_RGB_1_OVER_64], GroupWords[LOCAL_RGB_1_OVER_64]);
        InterlockedMax(
            ProbeWords[PROBE_WORD_MAX_RGB_DELTA_BITS],
            GroupWords[LOCAL_MAX_RGB_DELTA_BITS]);
        InterlockedMax(
            ProbeWords[PROBE_WORD_MAX_TILE_EXACT_CHANGED],
            GroupWords[LOCAL_EXACT_CHANGED]);
        InterlockedAdd(
            ProbeWords[PROBE_WORD_APPEARANCE_1_OVER_1024],
            GroupWords[LOCAL_APPEARANCE_1_OVER_1024]);
        InterlockedMax(
            ProbeWords[PROBE_WORD_MAX_APPEARANCE_DELTA_BITS],
            GroupWords[LOCAL_MAX_APPEARANCE_DELTA_BITS]);
        InterlockedAdd(
            ProbeWords[PROBE_WORD_BOTTOM_ADMITTED], GroupWords[LOCAL_BOTTOM_ADMITTED]);
        InterlockedAdd(
            ProbeWords[PROBE_WORD_BOTTOM_EXACT_CHANGED],
            GroupWords[LOCAL_BOTTOM_EXACT_CHANGED]);
        InterlockedAdd(
            ProbeWords[PROBE_WORD_BOTTOM_RGB_1_OVER_1024],
            GroupWords[LOCAL_BOTTOM_RGB_1_OVER_1024]);
        InterlockedMax(
            ProbeWords[PROBE_WORD_BOTTOM_MAX_RGB_DELTA_BITS],
            GroupWords[LOCAL_BOTTOM_MAX_RGB_DELTA_BITS]);

        if (gid.x == 0u && gid.y == 0u) {
            float scene_age_value =
                SBS_STATE_SCENE_AGE(CutBridgeState[SBS_STATE_VECTOR_SCENE_AGE]);
            float cut_flags_value =
                SBS_STATE_CUT_FLAGS(CutBridgeState[SBS_STATE_VECTOR_CUT_FLAGS]);
            float analysis_flags_value = SBS_STATE_ANALYSIS_FLAGS(
                CutBridgeState[SBS_STATE_VECTOR_ANALYSIS_FLAGS]);
            float history_state_value = SBS_STATE_MODEL_INPUT_HISTORY_STATE(
                CutBridgeState[SBS_STATE_VECTOR_MODEL_INPUT_HISTORY_STATE]);
            float initialized_value = SBS_STATE_INITIALIZED(
                CutBridgeState[SBS_STATE_VECTOR_INITIALIZED]);
            float depth_ready_value = SBS_STATE_DEPTH_READY(
                CutBridgeState[SBS_STATE_VECTOR_DEPTH_READY]);
            float collapsed_value = SBS_STATE_RANGE_COLLAPSED(
                CutBridgeState[SBS_STATE_VECTOR_RANGE_COLLAPSED]);
            float hard_cut_pulse_value = SBS_STATE_HARD_CUT_PULSE(
                CutBridgeState[SBS_STATE_VECTOR_HARD_CUT_PULSE]);
            // The health vector is a uint4 transported through float4 bits, not numeric floats.
            uint hard_cut_count_value = asuint(SBS_STATE_HARD_CUT_COUNT(
                CutBridgeState[SBS_STATE_VECTOR_HARD_CUT_COUNT]));
            bool hard_cut_count_valid =
                hard_cut_count_value <= SBS_ADAPTIVE_STATE_COUNTER_MAX;
            bool scene_age_valid = ProbeFiniteWholeInRange(
                scene_age_value, MOTION_PROBE_MAX_EXACT_NUMERIC_COUNTER);
            bool cut_flags_valid = ProbeFiniteWholeInRange(
                cut_flags_value, (float)MOTION_PROBE_KNOWN_CUT_FLAGS);
            bool analysis_flags_valid = ProbeFiniteWholeInRange(
                analysis_flags_value, (float)MOTION_PROBE_KNOWN_ANALYSIS_FLAGS);
            bool history_state_valid = ProbeFiniteWholeInRange(history_state_value, 4.0f);
            bool state_fields_valid =
                scene_age_valid && cut_flags_valid && analysis_flags_valid &&
                history_state_valid && ProbeCanonicalBoolean(initialized_value) &&
                ProbeCanonicalBoolean(depth_ready_value) &&
                ProbeCanonicalBoolean(collapsed_value) &&
                ProbeCanonicalBoolean(hard_cut_pulse_value);
            uint scene_age = scene_age_valid ? (uint)scene_age_value : 0xFFFFFFFFu;
            uint cut_flags = cut_flags_valid ? (uint)cut_flags_value : 0xFFFFFFFFu;
            uint analysis_flags = analysis_flags_valid ?
                (uint)analysis_flags_value : 0xFFFFFFFFu;
            uint history_state = history_state_valid ?
                (uint)history_state_value : 0xFFFFFFFFu;
            uint prior_flags = 0u;
            prior_flags |=
                asuint(SBS_STATE_CUT_CONTRACT_TAG_BITS(CutBridgeState[0])) ==
                    SBS_CUT_CONTRACT_TAG ? 1u << 0u : 0u;
            prior_flags |= initialized_value > 0.5f ? 1u << 1u : 0u;
            prior_flags |= depth_ready_value > 0.5f ? 1u << 2u : 0u;
            prior_flags |= collapsed_value < 0.5f ? 1u << 3u : 0u;
            prior_flags |= history_state == 1u ? 1u << 4u : 0u;
            prior_flags |= scene_age >= MOTION_PROBE_MIN_SCENE_AGE ? 1u << 5u : 0u;
            prior_flags |= cut_flags == 3u || cut_flags == 19u ? 1u << 6u : 0u;
            prior_flags |=
                hard_cut_pulse_value < 0.5f && analysis_flags == 0u ? 1u << 7u : 0u;
            prior_flags |= hard_cut_count_valid ? 1u << 8u : 0u;
            prior_flags |= state_fields_valid ? 1u << 9u : 0u;

            ProbeWords[PROBE_WORD_CONTRACT_TAG] = MOTION_PROBE_CONTRACT_TAG;
            ProbeWords[PROBE_WORD_PRIOR_STATE_FLAGS] = prior_flags;
            ProbeWords[PROBE_WORD_CURRENT_FRAME_LOW] = probe_current_frame_low;
            ProbeWords[PROBE_WORD_CURRENT_FRAME_HIGH] = probe_current_frame_high;
            ProbeWords[PROBE_WORD_BASELINE_FRAME_LOW] = probe_baseline_frame_low;
            ProbeWords[PROBE_WORD_BASELINE_FRAME_HIGH] = probe_baseline_frame_high;
            ProbeWords[PROBE_WORD_HARD_CUT_COUNT] = hard_cut_count_value;
            ProbeWords[PROBE_WORD_SCENE_AGE] = scene_age;
            ProbeWords[PROBE_WORD_CUT_FLAGS] = cut_flags;
            ProbeWords[PROBE_WORD_ANALYSIS_FLAGS] = analysis_flags;
            ProbeWords[PROBE_WORD_HISTORY_STATE] = history_state;
        }
    }
}
