// Acquire the shot center from the first usable depth field, then hold it until the next
// authenticated cut. Scale is the fixed authenticated model/shape coordinate scale. Unusable
// depth publishes flat without moving the scene camera; only an authenticated cut may clear a
// camera without replacing it from a usable field on the same update.

StructuredBuffer<float4> FrameStats : register(t0);
StructuredBuffer<float4> CutBridgeState : register(t1);
StructuredBuffer<uint> RawHistogram256 : register(t2);
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

uint StageBinCount(uint bin) {
    return RawHistogram256[bin * 2u] + RawHistogram256[bin * 2u + 1u];
}

float SmoothedStageBin(uint bin) {
    // Exactly one zero-padded `same` [0.25, 0.50, 0.25] pass. Smoothing is deliberately
    // acceptance-only: Otsu and all class masses use the original paired counts.
    float value = 0.5f * (float)StageBinCount(bin);
    if (bin > 0u) {
        value += 0.25f * (float)StageBinCount(bin - 1u);
    }
    if (bin + 1u < V2_STAGE_HISTOGRAM_BIN_COUNT) {
        value += 0.25f * (float)StageBinCount(bin + 1u);
    }
    return value;
}

bool ResolveStageBoundary(
    float minimum,
    float maximum,
    uint expected_count,
    float arithmetic_mean,
    out float center
) {
    center = arithmetic_mean;
    float raw_range = maximum - minimum;
    if (expected_count == 0u || !V2Finite(raw_range) || raw_range <= 0.0f) {
        return false;
    }

    uint histogram_count = 0u;
    float total_moment = 0.0f;
    [loop]
    for (uint bin = 0u; bin < V2_STAGE_HISTOGRAM_BIN_COUNT; ++bin) {
        uint count = StageBinCount(bin);
        histogram_count += count;
        float x = ((float)bin + 0.5f) / (float)V2_STAGE_HISTOGRAM_BIN_COUNT;
        total_moment += (float)count * x;
    }
    // depth_hist_cs intentionally ignores negative raw values while the Welford pass accepts all
    // finite values. Never infer a boundary unless both independent passes counted the same exact
    // field; calibrated DAV2 output satisfies this contract.
    if (histogram_count != expected_count) {
        return false;
    }

    float global_mean = total_moment / (float)histogram_count;
    float best_score = -1.0f;
    uint best_first_split = 0u;
    uint best_upper_split = 0u;
    bool found = false;
    uint far_count = 0u;
    float far_moment = 0.0f;
    [loop]
    for (uint first_split = 0u;
         first_split + 2u < V2_STAGE_HISTOGRAM_BIN_COUNT;
         ++first_split) {
        uint first_count = StageBinCount(first_split);
        far_count += first_count;
        far_moment += (float)first_count *
            (((float)first_split + 0.5f) / (float)V2_STAGE_HISTOGRAM_BIN_COUNT);

        uint middle_count = 0u;
        float middle_moment = 0.0f;
        [loop]
        for (uint upper_split = first_split + 1u;
             upper_split + 1u < V2_STAGE_HISTOGRAM_BIN_COUNT;
             ++upper_split) {
            uint current_count = StageBinCount(upper_split);
            middle_count += current_count;
            middle_moment += (float)current_count *
                (((float)upper_split + 0.5f) /
                 (float)V2_STAGE_HISTOGRAM_BIN_COUNT);
            uint near_count = histogram_count - far_count - middle_count;
            // Each original-count class must carry at least N/128 mass. Use multiplied integer
            // comparisons so non-divisible N has the exact mathematical ceil behavior.
            if (far_count * V2_STAGE_HISTOGRAM_BIN_COUNT < histogram_count ||
                middle_count * V2_STAGE_HISTOGRAM_BIN_COUNT < histogram_count ||
                near_count * V2_STAGE_HISTOGRAM_BIN_COUNT < histogram_count) {
                continue;
            }

            float near_moment = total_moment - far_moment - middle_moment;
            float far_delta = far_moment / (float)far_count - global_mean;
            float middle_delta = middle_moment / (float)middle_count - global_mean;
            float near_delta = near_moment / (float)near_count - global_mean;
            float score = (float)far_count * far_delta * far_delta +
                (float)middle_count * middle_delta * middle_delta +
                (float)near_count * near_delta * near_delta;
            // Ascending loops plus a strict comparison specify the tie break: retain the first
            // lexicographic (first_split, upper_split) pair with the maximum score.
            if (V2Finite(score) && score > best_score) {
                best_score = score;
                best_first_split = first_split;
                best_upper_split = upper_split;
                found = true;
            }
        }
    }
    if (!found) {
        return false;
    }

    float middle_peak = 0.0f;
    [loop]
    for (uint bin = best_first_split + 1u; bin <= best_upper_split; ++bin) {
        middle_peak = max(middle_peak, SmoothedStageBin(bin));
    }
    float near_peak = 0.0f;
    [loop]
    for (uint bin = best_upper_split + 1u;
         bin < V2_STAGE_HISTOGRAM_BIN_COUNT;
         ++bin) {
        near_peak = max(near_peak, SmoothedStageBin(bin));
    }
    float weaker_peak = min(middle_peak, near_peak);
    float valley = max(
        SmoothedStageBin(best_upper_split),
        SmoothedStageBin(best_upper_split + 1u));
    if (weaker_peak <= 0.0f ||
        valley / weaker_peak > v2_stage_valley_ratio_max) {
        return false;
    }

    float threshold = minimum +
        (((float)best_upper_split + 0.5f) /
         (float)V2_STAGE_HISTOGRAM_BIN_COUNT) * raw_range;
    // Preserve the conservative adoption predicate: the former knee placement must still lie
    // above the arithmetic mean. Once accepted, however, the separated upper valley itself is
    // the zero plane. This keeps a dominant middle subject near the screen while an isolated
    // near stage (for example a reaching hand) retains its relative relief.
    float acceptance_center = threshold - v2_raw_coordinate_scale;
    if (!V2Finite(acceptance_center) || acceptance_center <= arithmetic_mean) {
        return false;
    }
    center = threshold;
    return true;
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
        float acquired_center = V2_FRAME_STATS_MEAN(frame0);
        ResolveStageBoundary(
            V2_FRAME_STATS_MINIMUM(frame0),
            V2_FRAME_STATS_MAXIMUM(frame0),
            (uint)V2_FRAME_STATS_VALID_COUNT(frame1),
            V2_FRAME_STATS_MEAN(frame0),
            acquired_center);
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
