#ifndef DEPTH_COORDINATE_V2_HLSL
#define DEPTH_COORDINATE_V2_HLSL

#include "include/depth_coordinate_v2_contract.generated.hlsl"

bool V2Finite(float value) {
    return !isnan(value) && !isinf(value);
}

bool V2ApproximatelyEqual(float left, float right) {
    return V2Finite(left) && V2Finite(right) && abs(left - right) <= 1.0e-6f;
}

bool V2CalibrationRevisionValid(uint revision) {
    // IncrementExactCounter deliberately reserves all ones as corrupt/uninitialized evidence.
    return revision > 0u && revision != 0xffffffffu;
}

float V2NearDenseWeight(float coverage) {
    return smoothstep(
        v2_near_tail_coverage_low,
        v2_near_tail_coverage_high,
        coverage);
}

float V2ExpectedNearTau(float coverage) {
    return lerp(
        v2_near_log_tau,
        v2_near_log_tau_dense,
        V2NearDenseWeight(coverage));
}

bool V2NearShoulderValid(float4 shoulder, uint texel_count) {
    float coverage = V2_STATE_LATCHED_NEAR_TAIL_COVERAGE(shoulder);
    float effective_tau = V2_STATE_EFFECTIVE_NEAR_LOG_TAU(shoulder);
    uint count = asuint(V2_STATE_LATCHED_NEAR_TAIL_COUNT(shoulder));
    float expected_coverage = texel_count > 0u ?
        (float)count / (float)texel_count : 0.0f;
    return texel_count > 0u && count <= texel_count &&
        V2Finite(coverage) && coverage >= 0.0f && coverage <= 1.0f &&
        V2ApproximatelyEqual(coverage, expected_coverage) &&
        V2Finite(effective_tau) &&
        effective_tau >= v2_near_log_tau_dense &&
        effective_tau <= v2_near_log_tau &&
        V2ApproximatelyEqual(effective_tau, V2ExpectedNearTau(coverage)) &&
        asuint(V2_STATE_NEAR_SHOULDER_RESERVED(shoulder)) == 0u;
}

bool V2CameraStateValid(
    float4 active,
    float4 control,
    float4 shoulder,
    uint texel_count
) {
    float frame_valid = V2_STATE_FRAME_VALID(control);
    float container_scale = V2_STATE_CONTAINER_SCALE(active);
    float expected_inverse_scale = 1.0f / v2_raw_coordinate_scale;
    uint revision = asuint(V2_STATE_CALIBRATION_REVISION(control));
    return asuint(V2_STATE_CONTRACT_TAG_BITS(control)) == V2_CONTRACT_TAG &&
        V2Finite(v2_raw_coordinate_scale) && v2_raw_coordinate_scale > 0.0f &&
        V2Finite(V2_STATE_CENTER(active)) &&
        V2ApproximatelyEqual(V2_STATE_INVERSE_SCALE(active), expected_inverse_scale) &&
        V2ApproximatelyEqual(
            V2_STATE_CONVERGENCE_CURVE(active),
            v2_convergence_curve_default) &&
        V2Finite(container_scale) && container_scale > 0.0f && container_scale <= 1.0f &&
        (frame_valid == 0.0f || frame_valid == 1.0f) &&
        (frame_valid > 0.5f || container_scale == 1.0f) &&
        V2CalibrationRevisionValid(revision) &&
        V2NearShoulderValid(shoulder, texel_count);
}

bool V2EmptyCameraStateValid(
    float4 active,
    float4 control,
    float4 shoulder
) {
    uint revision = asuint(V2_STATE_CALIBRATION_REVISION(control));
    return asuint(V2_STATE_CONTRACT_TAG_BITS(control)) == V2_CONTRACT_TAG &&
        V2_STATE_CENTER(active) == 0.0f &&
        V2_STATE_INVERSE_SCALE(active) == 0.0f &&
        V2_STATE_CONVERGENCE_CURVE(active) == v2_convergence_curve_default &&
        V2_STATE_CONTAINER_SCALE(active) == 1.0f &&
        (revision == 0u || V2CalibrationRevisionValid(revision)) &&
        V2_STATE_FRAME_VALID(control) == 0.0f &&
        V2_STATE_LATCHED_NEAR_TAIL_COVERAGE(shoulder) == 0.0f &&
        V2_STATE_EFFECTIVE_NEAR_LOG_TAU(shoulder) == v2_near_log_tau &&
        asuint(V2_STATE_LATCHED_NEAR_TAIL_COUNT(shoulder)) == 0u &&
        asuint(V2_STATE_NEAR_SHOULDER_RESERVED(shoulder)) == 0u;
}

// D3D shader model 5 has exp/log but not expm1/log1p. Preserve precision at both C1 joins with
// short Taylor branches; ordinary values use the native transcendental instructions.
float V2Expm1(float value) {
    if (abs(value) < 1.0e-3f) {
        float value2 = value * value;
        return value + 0.5f * value2 + value2 * value * (1.0f / 6.0f);
    }
    return exp(value) - 1.0f;
}

float V2Log1p(float value) {
    if (abs(value) < 1.0e-3f) {
        float value2 = value * value;
        return value - 0.5f * value2 + value2 * value * (1.0f / 3.0f);
    }
    return log(1.0f + value);
}

float V2CurveWithNearTau(float coordinate, float near_tau) {
    if (coordinate < 0.0f) {
        return v2_far_tau * V2Expm1(coordinate / v2_far_tau);
    }
    if (coordinate <= 1.0f) {
        return coordinate;
    }
    return 1.0f + near_tau * V2Log1p((coordinate - 1.0f) / near_tau);
}

// The base curve remains the hard-container authority. Occupancy adaptation may only choose a
// smaller positive near tau in the map pass, so it cannot relax the container and amplify the
// unchanged far/middle branches.
float V2Curve(float coordinate) {
    return V2CurveWithNearTau(coordinate, v2_near_log_tau);
}

#endif
