// Acquire the shot center from the first usable depth field, then hold it until the next
// authenticated cut. Scale is the fixed authenticated model/shape coordinate scale. Unusable
// depth publishes flat without moving the scene camera; only an authenticated cut may clear a
// camera without replacing it from a usable field on the same update.

StructuredBuffer<float4> FrameStats : register(t0);
StructuredBuffer<float4> CutBridgeState : register(t1);
RWStructuredBuffer<float4> ShadowState : register(u0);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"

uint IncrementExactCounter(uint value) {
    // Keep 0xffffffff reserved as a conspicuous corrupt/uninitialized sentinel.
    return min(value, 0xfffffffdu) + 1u;
}

void ResetMappingState(inout float4 mapping_state) {
    mapping_state = float4(asfloat(0u), asfloat(0u), asfloat(0u), asfloat(0u));
}

void StoreCameraState(float4 active, float4 control, inout float4 mapping_state) {
    V2SealCameraCenter(active, control, mapping_state);
    // Seal the fully validated, current-frame-ready decision once. Per-texel producers and the
    // packed renderer consume this versioned token instead of repeating the full state checksum.
    V2SealRendererAuthorization(control, mapping_state);
    ShadowState[0] = active;
    ShadowState[1] = control;
    ShadowState[2] = mapping_state;
}

void PublishUnavailable(
    inout float4 active,
    inout float4 control,
    inout float4 mapping_state,
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
    float4 active = ShadowState[0];
    float4 control = ShadowState[1];
    float4 mapping_state = ShadowState[2];
    bool contract_matches = asuint(V2_STATE_CONTRACT_TAG_BITS(control)) == V2_CONTRACT_TAG;

    float4 cut_header = CutBridgeState[SBS_STATE_VECTOR_CUT_CONTRACT_TAG_BITS];
    bool cut_contract_matches =
        asuint(SBS_STATE_CUT_CONTRACT_TAG_BITS(cut_header)) == SBS_CUT_CONTRACT_TAG;
    float4 cut_pulse = 0.0f;
    float4 cut_health = 0.0f;
    uint current_cut_count = 0u;
    if (cut_contract_matches) {
        cut_pulse = CutBridgeState[SBS_STATE_VECTOR_HARD_CUT_PULSE];
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
    }
    bool camera_initialized = V2CameraStateValid(active, control, mapping_state);
    bool empty_camera = V2EmptyCameraStateValid(active, control, mapping_state);
    if (contract_matches && !camera_initialized && !empty_camera) {
        // Same-tag corruption must not become a permanent pseudo-camera.
        active = float4(0.0f, 0.0f, v2_convergence_curve_default, 1.0f);
        control = float4(asfloat(0u), 0.0f, asfloat(previous_cut_count),
                         asfloat(V2_CONTRACT_TAG));
        ResetMappingState(mapping_state);
    }
    V2_STATE_CONTRACT_TAG_BITS(control) = asfloat(V2_CONTRACT_TAG);

    // The cut buffer is an independently authenticated producer. A stale or same-sized foreign
    // buffer makes this frame unavailable, but it must not erase a previously valid scene camera.
    if (!cut_contract_matches) {
        V2_STATE_CONFIRMED_CUT_COUNT(control) = asfloat(previous_cut_count);
        PublishUnavailable(active, control, mapping_state, false);
        StoreCameraState(active, control, mapping_state);
        return;
    }

    // Observe both authenticated signals. A persistent generation change recovers a cut if the
    // transient pulse fell between shadow updates; a pulse alone remains sufficient on its frame.
    bool confirmed_cut = SBS_STATE_HARD_CUT_PULSE(cut_pulse) > 0.5f ||
        (contract_matches && current_cut_count != previous_cut_count);
    V2_STATE_CONFIRMED_CUT_COUNT(control) = asfloat(current_cut_count);

    float4 frame0 = FrameStats[V2_FRAME_STATS_VECTOR_MEAN];
    float4 frame1 = FrameStats[V2_FRAME_STATS_VECTOR_VALID_COUNT];
    float observed_std = V2_FRAME_STATS_POPULATION_STD(frame0);
    bool frame_valid = V2_FRAME_STATS_VALID(frame1) > 0.5f &&
        V2Finite(observed_std) && observed_std > v2_collapse_abs_epsilon;
    if (!frame_valid) {
        PublishUnavailable(
            active, control, mapping_state, confirmed_cut || !camera_initialized);
        StoreCameraState(active, control, mapping_state);
        return;
    }

    bool acquiring = !camera_initialized || confirmed_cut;
    if (acquiring) {
        float inverse_scale = 1.0f / v2_raw_coordinate_scale;
        if (!V2Finite(inverse_scale) || inverse_scale <= 0.0f) {
            PublishUnavailable(active, control, mapping_state, true);
            StoreCameraState(active, control, mapping_state);
            return;
        }
        // The arithmetic mean provides the occupancy behavior we need without a discrete scene
        // classifier: a small near object remains prominent, while a large near region naturally
        // pulls the zero plane toward itself. The wider calibrated coordinate scale keeps framing
        // and letterbox regions from collapsing the rest of the scene onto the far shelf.
        float acquired_center = V2_FRAME_STATS_MEAN(frame0);
        active = float4(
            acquired_center, inverse_scale, v2_convergence_curve_default, 1.0f);
        ResetMappingState(mapping_state);
        V2_STATE_CALIBRATION_REVISION(control) = asfloat(
            IncrementExactCounter(asuint(V2_STATE_CALIBRATION_REVISION(control))));
    }

    // Keep the ABI field fixed at identity. The map pass applies the strict representation bound
    // independently per texel, so one raw outlier can no longer pump the whole frame's pop.
    V2_STATE_CONTAINER_SCALE(active) = 1.0f;
    V2_STATE_FRAME_VALID(control) = 1.0f;

    StoreCameraState(active, control, mapping_state);
}
