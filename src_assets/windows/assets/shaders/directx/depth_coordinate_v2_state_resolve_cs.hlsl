// Continuous Host mean zero and robust depth amplitude. Only authenticated infer
// publication observes the controller; cuts retain geometry and reuse holds the complete tuple.

StructuredBuffer<float4> FrameStats : register(t0);
StructuredBuffer<float4> CutBridgeState : register(t1);
// Publish the mixed ABI as integer words. A floating typed UAV store may flush small mode
// tokens even after integer local computation; numeric camera words use explicit bit casts.
RWStructuredBuffer<uint4> ShadowState : register(u0);

#include "include/depth_coordinate_v2.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"

uint IncrementExactCounter(uint value) {
    // Keep 0xffffffff reserved as a conspicuous corrupt/uninitialized sentinel.
    return min(value, 0xfffffffdu) + 1u;
}

void ResetMappingState(inout uint4 mapping_state) {
    // This row contains only integer payloads. Keep the mode token as an integer through local
    // reset/sealing; treating bit pattern 1 as a floating-point temporary can flush it to zero.
    mapping_state = uint4(0u, 0u,
        V2_ADAPTIVE_POLICY_ID, 0u);
}

void ClearAdaptiveCameraState(inout V2AdaptiveCameraState camera_state) {
    camera_state.clock = 0u;
    camera_state.target = 0u;
    camera_state.seed_times = 0u;
    camera_state.seed_values = 0u;
}

void DisarmAdaptiveCamera(inout V2AdaptiveCameraState camera_state) {
    camera_state.clock.z = 0u;
    camera_state.target.xyz = 0u;
    if (camera_state.clock.w == 0u) {
        camera_state.clock.w = 0u;
        camera_state.seed_times = 0u;
        camera_state.seed_values = 0u;
    }
}

void StoreCameraState(float4 active, float4 control, inout uint4 mapping_state, V2AdaptiveCameraState camera_state) {
    V2_STATE_CAMERA_CENTER_INTEGRITY_BITS(mapping_state) =
        V2CameraIntegrityWithAdaptiveState(active, control, mapping_state, camera_state);
    // Seal the fully validated, current-frame-ready decision once. Per-texel producers and the
    // packed renderer consume this versioned token instead of repeating the full state checksum.
    V2_STATE_RENDERER_AUTHORIZATION_BITS(mapping_state) =
        V2_STATE_FRAME_VALID(control) > 0.5f ? V2_CONTRACT_TAG : 0u;
    ShadowState[0] = asuint(active);
    ShadowState[1] = asuint(control);
    ShadowState[2] = mapping_state;
    ShadowState[3] = camera_state.clock;
    ShadowState[4] = camera_state.target;
    ShadowState[5] = camera_state.seed_times;
    ShadowState[6] = camera_state.seed_values;
}

// Host mean zero and robust amplitude, observed only on authenticated infer publication.
// A presentation/reuse never enters this function and cannot earn catch-up adaptation time.
void ObserveAdaptiveCamera(inout float4 active, inout float4 control, inout uint4 mapping,
                     inout V2AdaptiveCameraState camera_state, float4 frame0, float4 frame1,
                     bool initialized) {
    V2_STATE_FRAME_VALID(control) = 0.0f;
    // A cut remains authenticated frame metadata; it does not latch or disarm adaptation.

    uint2 now = uint2(v2_joint_observation_timestamp_low, v2_joint_observation_timestamp_high);
    uint2 previous = camera_state.clock.xy;
    if (all(now == 0u) || (!all(previous == 0u) && !V2ClockAfter(now, previous))) {
        DisarmAdaptiveCamera(camera_state);
        return;
    }
    bool can_move = initialized && camera_state.clock.z == 1u &&
        V2ClockWithin(now, previous, (uint)v2_adaptive_max_tick_gap_ms * 1000u);
    uint2 elapsed = V2ClockSubtract(now, previous);
    camera_state.clock.xy = now;

    float budget = V2DisplayBudget();
    float recorded_budget = asfloat(camera_state.target.w);
    // Artistic strength is stream configuration, not another adaptive normalization variable.
    if (!V2Finite(budget) || budget <= 0.0f ||
        (recorded_budget != 0.0f && recorded_budget != budget)) {
        DisarmAdaptiveCamera(camera_state);
        return;
    }
    camera_state.target.w = asuint(budget);

    float minimum = V2_FRAME_STATS_MINIMUM(frame0);
    float maximum = V2_FRAME_STATS_MAXIMUM(frame0);
    float mean = V2_FRAME_STATS_MEAN(frame0);
    float observed_std = V2_FRAME_STATS_POPULATION_STD(frame0);
    float4 quantiles = FrameStats[V2_FRAME_STATS_VECTOR_PERCENTILE_LOW];
    float low = V2_FRAME_STATS_PERCENTILE_LOW(quantiles);
    float high = V2_FRAME_STATS_PERCENTILE_HIGH(quantiles);
    precise float half_span = 0.5f * (high - low);
    // The model's calibrated raw scale is an artistic amplitude prior, not a noise floor or a
    // physical inverse-distance conversion. Tiny raw contrast cannot earn full disparity.
    float target_reference = max(half_span, v2_raw_coordinate_scale);
    precise float target_zero = mean;
    precise float target_inverse = 1.0f / target_reference;
    bool usable = V2_FRAME_STATS_VALID(frame1) == 1.0f &&
        V2Finite(minimum) && V2Finite(maximum) && V2Finite(mean) &&
        maximum > minimum && mean >= minimum && mean <= maximum &&
        V2Finite(observed_std) && observed_std > v2_collapse_abs_epsilon &&
        V2_FRAME_STATS_PERCENTILE_VALID(quantiles) == 1.0f &&
        V2Finite(low) && V2Finite(high) && low >= minimum && high <= maximum && low <= high &&
        V2Finite(V2_FRAME_STATS_PERCENTILE_BIN_WIDTH(quantiles)) &&
        V2_FRAME_STATS_PERCENTILE_BIN_WIDTH(quantiles) >= 0.0f &&
        V2Finite(half_span) && V2Finite(v2_raw_coordinate_scale) && v2_raw_coordinate_scale > 0.0f &&
        V2Finite(target_reference) && target_reference > 0.0f &&
        V2Finite(target_inverse) && target_inverse > 0.0f;
    if (!usable) {
        DisarmAdaptiveCamera(camera_state);
        return;
    }

    // Historical state-tail field names retain their ABI; the nearest reference carries Host D.
    camera_state.target.xyz = asuint(float3(target_zero, target_inverse, target_reference));
    camera_state.clock.z = 1u;
    if (!initialized) {
        // Acquire on the first genuine observation, before DDup can freeze a static stream.
        camera_state.clock.w = 1u;
        camera_state.seed_times = uint4(now, now);
        camera_state.seed_values = uint4(asuint(float2(target_reference, target_zero)), 0u, 0u);
        active = float4(target_zero, target_inverse, v2_convergence_curve_default, 1.0f);
    } else if (can_move) {
        float seconds = float(elapsed.x) * 0.000001f;
        float alpha = 1.0f - exp(-seconds / v2_adaptive_time_constant_seconds);
        float old_inverse = V2_STATE_INVERSE_SCALE(active);
        // Subtract logarithms instead of forming an overflowing target/old ratio.
        float log_delta = alpha * (log(target_inverse) - log(old_inverse));
        float log_bound = v2_adaptive_log_scale_rate * seconds;
        precise float next_inverse = old_inverse * exp(clamp(log_delta, -log_bound, log_bound));
        float zero_bound = v2_adaptive_zero_budget_per_second * seconds / next_inverse;
        float old_zero = V2_STATE_CENTER(active);
        float zero_delta = alpha * target_zero - alpha * old_zero;
        precise float next_zero = old_zero + clamp(zero_delta, -zero_bound, zero_bound);
        if (!V2Finite(next_inverse) || next_inverse <= 0.0f ||
            !V2Finite(zero_bound) || !V2Finite(next_zero)) {
            DisarmAdaptiveCamera(camera_state);
            return;
        }
        active = float4(next_zero, next_inverse, v2_convergence_curve_default, 1.0f);
    }
    V2_STATE_CALIBRATION_REVISION(control) = asfloat(
        IncrementExactCounter(asuint(V2_STATE_CALIBRATION_REVISION(control))));
    V2_STATE_FRAME_VALID(control) = 1.0f;
}

void PublishUnavailable(
    inout float4 active,
    inout float4 control,
    inout uint4 mapping_state,
    bool clear_camera
) {
    if (clear_camera) {
        active = float4(0.0f, 0.0f, v2_convergence_curve_default, 1.0f);
        ResetMappingState(mapping_state);
    } else {
        V2_STATE_CONTAINER_SCALE(active) = 1.0f;
    }
    V2_STATE_FRAME_VALID(control) = 0.0f;
}

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float4 active = asfloat(ShadowState[0]);
    float4 control = asfloat(ShadowState[1]);
    uint4 mapping_state = ShadowState[2];
    V2AdaptiveCameraState camera_state;
    camera_state.clock = ShadowState[3];
    camera_state.target = ShadowState[4];
    camera_state.seed_times = ShadowState[5];
    camera_state.seed_values = ShadowState[6];
    bool contract_matches = asuint(V2_STATE_CONTRACT_TAG_BITS(control)) == V2_CONTRACT_TAG;

    float4 cut_header = CutBridgeState[SBS_STATE_VECTOR_CUT_CONTRACT_TAG_BITS];
    bool cut_contract_matches =
        asuint(SBS_STATE_CUT_CONTRACT_TAG_BITS(cut_header)) == SBS_CUT_CONTRACT_TAG;
    float4 cut_health = 0.0f;
    uint current_cut_count = 0u;
    if (cut_contract_matches) {
        cut_health = CutBridgeState[SBS_STATE_VECTOR_HARD_CUT_COUNT];
        current_cut_count = asuint(SBS_STATE_HARD_CUT_COUNT(cut_health));
    }
    uint previous_cut_count = contract_matches ?
        asuint(V2_STATE_CONFIRMED_CUT_COUNT(control)) : current_cut_count;
    if (!contract_matches) {
        // Never inherit counters or calibration state from an unknown same-sized buffer. The
        // current authenticated cut generation becomes the new baseline when it is available.
        active = float4(0.0f, 0.0f, v2_convergence_curve_default, 1.0f);
        control = float4(asfloat(0u), 0.0f, asfloat(previous_cut_count),
                         asfloat(V2_CONTRACT_TAG));
        ResetMappingState(mapping_state);
        ClearAdaptiveCameraState(camera_state);
    }
    bool camera_initialized = V2CameraStateWithAdaptiveStateValid(active, control, mapping_state, camera_state);
    bool empty_camera = V2EmptyCameraStateWithAdaptiveStateValid(active, control, mapping_state, camera_state);
    if (contract_matches && !camera_initialized && !empty_camera) {
        // Same-tag corruption must not become a permanent pseudo-camera.
        active = float4(0.0f, 0.0f, v2_convergence_curve_default, 1.0f);
        control = float4(asfloat(0u), 0.0f, asfloat(previous_cut_count),
                         asfloat(V2_CONTRACT_TAG));
        ResetMappingState(mapping_state);
        ClearAdaptiveCameraState(camera_state);
    }
    V2_STATE_CONTRACT_TAG_BITS(control) = asfloat(V2_CONTRACT_TAG);

    // The sole adaptive policy is authenticated, never inferred from arbitrary state bits.
    // Malformed constant padding or a foreign policy cannot authorize a parallax field.
    if (!V2JointPlaneConstantsValid()) {
        PublishUnavailable(active, control, mapping_state, !camera_initialized);
        DisarmAdaptiveCamera(camera_state);
        StoreCameraState(active, control, mapping_state, camera_state);
        return;
    }

    // The cut buffer is an independently authenticated producer. A stale or same-sized foreign
    // buffer makes this frame unavailable, but it must not erase the established adaptive camera.
    if (!cut_contract_matches) {
        V2_STATE_CONFIRMED_CUT_COUNT(control) = asfloat(previous_cut_count);
        PublishUnavailable(active, control, mapping_state, false);
        DisarmAdaptiveCamera(camera_state);
        StoreCameraState(active, control, mapping_state, camera_state);
        return;
    }

    // Keep the authenticated cut epoch for shared frame/subtitle ownership.
    V2_STATE_CONFIRMED_CUT_COUNT(control) = asfloat(current_cut_count);

    float4 frame0 = FrameStats[V2_FRAME_STATS_VECTOR_MEAN];
    float4 frame1 = FrameStats[V2_FRAME_STATS_VECTOR_VALID_COUNT];
    ObserveAdaptiveCamera(active, control, mapping_state, camera_state, frame0, frame1,
        camera_initialized);
    StoreCameraState(active, control, mapping_state, camera_state);
}
