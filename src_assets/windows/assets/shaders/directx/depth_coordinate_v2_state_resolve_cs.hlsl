// Latch the shot center from the first usable frame and resolve only frame-local hard safety.
// Scale is the fixed authenticated model/shape coordinate scale. Unusable depth publishes flat
// without moving the scene camera; only an authenticated cut may clear a camera without replacing
// it from a usable field on the same update.

StructuredBuffer<float4> FrameStats : register(t0);
StructuredBuffer<float4> LegacyState : register(t1);
RWStructuredBuffer<float4> ShadowState : register(u0);
RWStructuredBuffer<uint> NearTailCount : register(u1);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"

uint IncrementExactCounter(uint value) {
    // Keep 0xffffffff reserved as a conspicuous corrupt/uninitialized sentinel.
    return min(value, 0xfffffffdu) + 1u;
}

void ResetNearShoulder(inout float4 shoulder) {
    shoulder = float4(0.0f, v2_near_log_tau, asfloat(0u), asfloat(0u));
}

void StoreCameraState(float4 active, float4 control, inout float4 shoulder) {
    V2SealCameraCenter(active, control, shoulder);
    ShadowState[0] = active;
    ShadowState[1] = control;
    ShadowState[2] = shoulder;
}

void PublishUnavailable(
    inout float4 active,
    inout float4 control,
    inout float4 shoulder,
    bool clear_camera
) {
    if (clear_camera) {
        active = float4(0.0f, 0.0f, v2_convergence_curve_default, 1.0f);
        ResetNearShoulder(shoulder);
    } else {
        V2_STATE_CONTAINER_SCALE(active) = 1.0f;
    }
    V2_STATE_FRAME_VALID(control) = 0.0f;
}

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    // The count UAV is single-use frame evidence. Consume and clear it before any early return so
    // an invalid/cut frame cannot leak coverage into a later camera acquisition.
    uint observed_near_tail_count = NearTailCount[0];
    NearTailCount[0] = 0u;
    float4 active = ShadowState[0];
    float4 control = ShadowState[1];
    float4 shoulder = ShadowState[2];
    bool contract_matches = asuint(V2_STATE_CONTRACT_TAG_BITS(control)) == V2_CONTRACT_TAG;

    // Observe both signals. A persistent generation change recovers a cut if the transient pulse
    // fell between shadow updates; a pulse alone remains sufficient on the frame it is emitted.
    float4 legacy_cut = LegacyState[SBS_STATE_VECTOR_HARD_CUT_PULSE];
    float4 legacy_health = LegacyState[SBS_STATE_VECTOR_HARD_CUT_COUNT];
    uint current_cut_count = asuint(SBS_STATE_HARD_CUT_COUNT(legacy_health));
    uint previous_cut_count = contract_matches ?
        asuint(V2_STATE_CONFIRMED_CUT_COUNT(control)) : current_cut_count;
    bool confirmed_cut = SBS_STATE_HARD_CUT_PULSE(legacy_cut) > 0.5f ||
        (contract_matches && current_cut_count != previous_cut_count);
    if (!contract_matches) {
        // Never inherit counters or calibration state from an unknown same-sized buffer. The
        // current observed cut generation becomes the new authenticated baseline.
        active = float4(0.0f, 0.0f, v2_convergence_curve_default, 1.0f);
        control = float4(asfloat(0u), 0.0f, asfloat(current_cut_count),
                         asfloat(V2_CONTRACT_TAG));
        ResetNearShoulder(shoulder);
    }
    bool camera_initialized = V2CameraStateValid(
        active, control, shoulder, target_w * target_h);
    bool empty_camera = V2EmptyCameraStateValid(active, control, shoulder);
    if (contract_matches && !camera_initialized && !empty_camera) {
        // Same-tag corruption must not become a permanent pseudo-camera. Coverage has used the
        // same strict predicate and therefore collected fresh acquisition evidence this frame.
        active = float4(0.0f, 0.0f, v2_convergence_curve_default, 1.0f);
        control = float4(asfloat(0u), 0.0f, asfloat(current_cut_count),
                         asfloat(V2_CONTRACT_TAG));
        ResetNearShoulder(shoulder);
    }
    V2_STATE_CONTRACT_TAG_BITS(control) = asfloat(V2_CONTRACT_TAG);
    V2_STATE_CONFIRMED_CUT_COUNT(control) = asfloat(current_cut_count);

    float4 frame0 = FrameStats[V2_FRAME_STATS_VECTOR_MEAN];
    float4 frame1 = FrameStats[V2_FRAME_STATS_VECTOR_VALID_COUNT];
    float observed_std = V2_FRAME_STATS_POPULATION_STD(frame0);
    bool frame_valid = V2_FRAME_STATS_VALID(frame1) > 0.5f &&
        V2Finite(observed_std) && observed_std > v2_collapse_abs_epsilon;
    if (!frame_valid) {
        PublishUnavailable(
            active, control, shoulder, confirmed_cut || !camera_initialized);
        StoreCameraState(active, control, shoulder);
        return;
    }

    bool acquiring = !camera_initialized || confirmed_cut;
    if (acquiring) {
        float inverse_scale = 1.0f / v2_raw_coordinate_scale;
        uint texel_count = (uint)V2_FRAME_STATS_TEXEL_COUNT(frame1);
        bool coverage_count_valid = texel_count == target_w * target_h &&
            texel_count > 0u && observed_near_tail_count <= texel_count;
        float coverage = coverage_count_valid ?
            (float)observed_near_tail_count / (float)texel_count : 0.0f;
        float effective_near_tau = V2ExpectedNearTau(coverage);
        if (!V2Finite(inverse_scale) || inverse_scale <= 0.0f ||
            !coverage_count_valid || !V2Finite(coverage) ||
            !V2Finite(effective_near_tau) || effective_near_tau <= 0.0f) {
            PublishUnavailable(active, control, shoulder, true);
            StoreCameraState(active, control, shoulder);
            return;
        }
        active = float4(
            V2_FRAME_STATS_MEAN(frame0),
            inverse_scale,
            v2_convergence_curve_default,
            1.0f);
        shoulder = float4(
            coverage,
            effective_near_tau,
            asfloat(observed_near_tail_count),
            asfloat(0u));
        V2_STATE_CALIBRATION_REVISION(control) = asfloat(
            IncrementExactCounter(asuint(V2_STATE_CALIBRATION_REVISION(control))));
    }

    // V2Curve is monotone, so the raw frame endpoints are the exact current curve extrema. This
    // hard representation cap is deliberately frame-local: a transient extreme shrinks only its
    // own field and ordinary following frames recover to full requested gain.
    float minimum_curve = V2Curve(
        (V2_FRAME_STATS_MINIMUM(frame0) - V2_STATE_CENTER(active)) *
        V2_STATE_INVERSE_SCALE(active)) - V2_STATE_CONVERGENCE_CURVE(active);
    float maximum_curve = V2Curve(
        (V2_FRAME_STATS_MAXIMUM(frame0) - V2_STATE_CENTER(active)) *
        V2_STATE_INVERSE_SCALE(active)) - V2_STATE_CONVERGENCE_CURVE(active);
    float maximum_requested = v2_requested_gain * max(abs(minimum_curve), abs(maximum_curve));
    float container_scale = maximum_requested > 0.0f ?
        min(1.0f, v2_direct_container_limit / maximum_requested) : 1.0f;
    if (!V2Finite(minimum_curve) || !V2Finite(maximum_curve) ||
        !V2Finite(maximum_requested) || !V2Finite(container_scale) ||
        container_scale <= 0.0f) {
        PublishUnavailable(active, control, shoulder, confirmed_cut || acquiring);
    } else {
        V2_STATE_CONTAINER_SCALE(active) = container_scale;
        V2_STATE_FRAME_VALID(control) = 1.0f;
    }

    StoreCameraState(active, control, shoulder);
}
