#ifndef DEPTH_COORDINATE_V2_HLSL
#define DEPTH_COORDINATE_V2_HLSL

#include "include/depth_coordinate_v2_contract.generated.hlsl"

#define V2_STAGE_HISTOGRAM_SOURCE_BIN_COUNT 256u
#define V2_STAGE_HISTOGRAM_BIN_COUNT 128u

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

// Authenticate both scene-camera coordinates. The fixed inverse scale and revision are included
// so a same-tag state assembled from different camera generations cannot accidentally validate.
// Starting from zero deliberately keeps the generated empty-state word zero while still making
// every acquired camera carry a deterministic non-zero checksum in the ordinary case. Integer
// overflow is the specified modulo-2^32 checksum behavior.
uint V2CameraCenterIntegrityBits(
    float4 active,
    float4 control
) {
    uint checksum = 0u;
    checksum = (checksum ^ asuint(V2_STATE_CENTER(active))) * 16777619u;
    checksum = (checksum ^ asuint(V2_STATE_INVERSE_SCALE(active))) * 16777619u;
    checksum = (checksum ^ asuint(V2_STATE_CONVERGENCE_CURVE(active))) * 16777619u;
    checksum = (checksum ^ asuint(V2_STATE_CALIBRATION_REVISION(control))) * 16777619u;
    return checksum;
}

bool V2CameraCenterIntegrityValid(float4 active, float4 control, float4 mapping_state) {
    return asuint(V2_STATE_CAMERA_CENTER_INTEGRITY_BITS(mapping_state)) ==
        V2CameraCenterIntegrityBits(active, control);
}

void V2SealCameraCenter(float4 active, float4 control, inout float4 mapping_state) {
    V2_STATE_CAMERA_CENTER_INTEGRITY_BITS(mapping_state) = asfloat(
        V2CameraCenterIntegrityBits(active, control));
}

bool V2MappingStateValid(float4 mapping_state) {
    return asuint(V2_STATE_MAPPING_STATE_RESERVED_0(mapping_state)) == 0u &&
        asuint(V2_STATE_MAPPING_STATE_RESERVED_1(mapping_state)) == 0u &&
        asuint(V2_STATE_MAPPING_STATE_RESERVED_2(mapping_state)) == 0u;
}

bool V2ConvergenceCurveValid(float value) {
    return value == v2_convergence_curve_default;
}

bool V2CameraStateValid(
    float4 active,
    float4 control,
    float4 mapping_state
) {
    float frame_valid = V2_STATE_FRAME_VALID(control);
    float container_scale = V2_STATE_CONTAINER_SCALE(active);
    float expected_inverse_scale = 1.0f / v2_raw_coordinate_scale;
    uint revision = asuint(V2_STATE_CALIBRATION_REVISION(control));
    return asuint(V2_STATE_CONTRACT_TAG_BITS(control)) == V2_CONTRACT_TAG &&
        V2Finite(v2_raw_coordinate_scale) && v2_raw_coordinate_scale > 0.0f &&
        V2Finite(V2_STATE_CENTER(active)) &&
        V2ApproximatelyEqual(V2_STATE_INVERSE_SCALE(active), expected_inverse_scale) &&
        V2ConvergenceCurveValid(V2_STATE_CONVERGENCE_CURVE(active)) &&
        V2Finite(container_scale) && container_scale > 0.0f && container_scale <= 1.0f &&
        (frame_valid == 0.0f || frame_valid == 1.0f) &&
        (frame_valid > 0.5f || container_scale == 1.0f) &&
        V2CalibrationRevisionValid(revision) &&
        V2CameraCenterIntegrityValid(active, control, mapping_state) &&
        V2MappingStateValid(mapping_state);
}

bool V2EmptyCameraStateValid(
    float4 active,
    float4 control,
    float4 mapping_state
) {
    uint revision = asuint(V2_STATE_CALIBRATION_REVISION(control));
    return asuint(V2_STATE_CONTRACT_TAG_BITS(control)) == V2_CONTRACT_TAG &&
        V2_STATE_CENTER(active) == 0.0f &&
        V2_STATE_INVERSE_SCALE(active) == 0.0f &&
        V2_STATE_CONVERGENCE_CURVE(active) == v2_convergence_curve_default &&
        V2_STATE_CONTAINER_SCALE(active) == 1.0f &&
        (revision == 0u || V2CalibrationRevisionValid(revision)) &&
        V2_STATE_FRAME_VALID(control) == 0.0f &&
        V2CameraCenterIntegrityValid(active, control, mapping_state) &&
        V2MappingStateValid(mapping_state);
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

// The fixed curve is shared by the hard-container resolver and the map pass. Keeping one curve
// prevents content occupancy from changing the relative depth of an otherwise identical scene.
float V2Curve(float coordinate) {
    return V2CurveWithNearTau(coordinate, v2_near_log_tau);
}

#endif
