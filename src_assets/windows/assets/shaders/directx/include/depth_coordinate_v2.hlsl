#ifndef DEPTH_COORDINATE_V2_HLSL
#define DEPTH_COORDINATE_V2_HLSL

#include "include/depth_coordinate_v2_contract.generated.hlsl"

bool V2Finite(float value) {
    return !isnan(value) && !isinf(value);
}

bool V2CalibrationRevisionValid(uint revision) {
    // IncrementExactCounter deliberately reserves all ones as corrupt/uninitialized evidence.
    return revision > 0u && revision != 0xffffffffu;
}

bool V2JointPlaneModeValid(uint mode) {
    return mode == V2_ADAPTIVE_POLICY_ID;
}

bool V2JointPlaneConstantsValid() {
    return V2JointPlaneModeValid(v2_joint_plane_mode) &&
        v2_joint_plane_reserved0 == 0u && v2_joint_plane_reserved1 == 0u &&
        v2_joint_plane_reserved2 == 0u &&
        v2_joint_observation_reserved0 == 0u && v2_joint_observation_reserved1 == 0u &&
        v2_coordinate_reserved0 == 0.0f && v2_coordinate_reserved1 == 0.0f &&
        v2_direct_container_limit == V2_DIRECT_CONTAINER_LIMIT &&
        v2_convergence_curve_default == V2_CONVERGENCE_CURVE_DEFAULT;
}

float V2DisplayBudget() {
    return v2_direct_container_limit;
}

float V2BoundDisplayParallax(float value) {
    return clamp(value, -V2DisplayBudget(), V2DisplayBudget());
}

// All controller payloads remain integers, including mixed float words, until UAV publication.
// This avoids losing small uint tokens through floating-point temporaries on native GPUs.
struct V2AdaptiveCameraState {
    uint4 clock;
    uint4 target;
    uint4 seed_times;
    uint4 seed_values;
};

bool V2ClockAfter(uint2 left, uint2 right) {
    return left.y > right.y || (left.y == right.y && left.x > right.x);
}

bool V2ClockAtLeast(uint2 left, uint2 right) {
    return all(left == right) || V2ClockAfter(left, right);
}

uint2 V2ClockSubtract(uint2 left, uint2 right) {
    return uint2(left.x - right.x, left.y - right.y - (left.x < right.x ? 1u : 0u));
}

bool V2ClockWithin(uint2 later, uint2 earlier, uint maximum_us) {
    uint2 difference = V2ClockSubtract(later, earlier);
    return V2ClockAtLeast(later, earlier) && difference.y == 0u && difference.x <= maximum_us;
}

uint V2CameraIntegrityBase(float4 active, float4 control, uint mode) {
    uint checksum = 0u;
    checksum = (checksum ^ asuint(V2_STATE_CENTER(active))) * 16777619u;
    checksum = (checksum ^ asuint(V2_STATE_INVERSE_SCALE(active))) * 16777619u;
    checksum = (checksum ^ asuint(V2_STATE_CONVERGENCE_CURVE(active))) * 16777619u;
    checksum = (checksum ^ asuint(V2_STATE_CALIBRATION_REVISION(control))) * 16777619u;
    return (checksum ^ mode) * 16777619u;
}

uint V2CameraIntegrityWithAdaptiveState(float4 active, float4 control, uint4 mapping, V2AdaptiveCameraState camera_state) {
    uint checksum = V2CameraIntegrityBase(active, control, V2_STATE_JOINT_PLANE_MODE_BITS(mapping));
    [unroll] for (uint component = 0u; component < 4u; ++component)
            checksum = (checksum ^ camera_state.clock[component]) * 16777619u;
        [unroll] for (uint component = 0u; component < 4u; ++component)
            checksum = (checksum ^ camera_state.target[component]) * 16777619u;
        [unroll] for (uint component = 0u; component < 4u; ++component)
            checksum = (checksum ^ camera_state.seed_times[component]) * 16777619u;
        [unroll] for (uint component = 0u; component < 4u; ++component)
            checksum = (checksum ^ camera_state.seed_values[component]) * 16777619u;
    return checksum;
}

bool V2AdaptiveCameraTailValid(uint mode, V2AdaptiveCameraState camera_state) {
    float4 target = asfloat(camera_state.target);
    float2 seed = asfloat(camera_state.seed_values.xy);
    if (mode != V2_ADAPTIVE_POLICY_ID || camera_state.clock.z > 1u || camera_state.clock.w > 1u ||
        any(camera_state.seed_values.zw != 0u) ||
        (camera_state.clock.z != 0u && all(camera_state.clock.xy == 0u)) ||
        !V2Finite(target.x) || !V2Finite(target.y) || !V2Finite(target.z) ||
        !V2Finite(target.w) || !V2Finite(seed.x) || !V2Finite(seed.y) ||
        target.w < 0.0f || target.w > v2_direct_container_limit) return false;
    bool no_target = all(target.xyz == 0.0f);
    bool valid_target = target.y > 0.0f && target.z > 0.0f &&
        abs(target.y * target.z - 1.0f) <= 2.0e-6f;
    if ((!no_target && !valid_target) || (camera_state.clock.z != 0u && !valid_target)) return false;
    if ((!no_target && target.z < v2_raw_coordinate_scale) ||
        (seed.x != 0.0f && seed.x < v2_raw_coordinate_scale)) return false;
    if (camera_state.clock.w == 0u)
        return camera_state.clock.z == 0u && all(camera_state.seed_times == 0u) && all(seed == 0.0f);
    if (all(camera_state.seed_times.xy == 0u) ||
        !V2ClockAtLeast(camera_state.seed_times.zw, camera_state.seed_times.xy) ||
        !V2ClockAtLeast(camera_state.clock.xy, camera_state.seed_times.zw) ||
        seed.x <= 0.0f || target.w <= 0.0f) return false;
    return all(camera_state.seed_times.zw == camera_state.seed_times.xy);
}

bool V2MappingStateValid(float4 mapping_state) {
    return V2JointPlaneModeValid(asuint(V2_STATE_JOINT_PLANE_MODE_BITS(mapping_state))) &&
        asuint(V2_STATE_MAPPING_STATE_RESERVED_2(mapping_state)) == 0u;
}

bool V2RendererAuthorizationValid(float4 control, float4 mapping_state) {
    uint expected = V2_STATE_FRAME_VALID(control) > 0.5f ? V2_CONTRACT_TAG : 0u;
    return asuint(V2_STATE_RENDERER_AUTHORIZATION_BITS(mapping_state)) == expected;
}

void V2SealRendererAuthorization(float4 control, inout float4 mapping_state) {
    V2_STATE_RENDERER_AUTHORIZATION_BITS(mapping_state) = asfloat(
        V2_STATE_FRAME_VALID(control) > 0.5f ? V2_CONTRACT_TAG : 0u);
}

bool V2ConvergenceCurveValid(float value) {
    return value == v2_convergence_curve_default;
}

bool V2CameraParametersValid(
    float4 active,
    float4 control,
    float4 mapping_state
) {
    float frame_valid = V2_STATE_FRAME_VALID(control);
    float container_scale = V2_STATE_CONTAINER_SCALE(active);
    float expected_inverse_scale = 1.0f / v2_raw_coordinate_scale;
    float inverse_scale = V2_STATE_INVERSE_SCALE(active);
    bool scale_valid = V2Finite(inverse_scale) && inverse_scale > 0.0f &&
        inverse_scale <= expected_inverse_scale + 2.0e-6f;
    uint revision = asuint(V2_STATE_CALIBRATION_REVISION(control));
    return asuint(V2_STATE_CONTRACT_TAG_BITS(control)) == V2_CONTRACT_TAG &&
        V2Finite(v2_raw_coordinate_scale) && v2_raw_coordinate_scale > 0.0f &&
        V2Finite(V2_STATE_CENTER(active)) &&
        scale_valid &&
        V2ConvergenceCurveValid(V2_STATE_CONVERGENCE_CURVE(active)) &&
        V2Finite(container_scale) && container_scale == 1.0f &&
        (frame_valid == 0.0f || frame_valid == 1.0f) &&
        V2CalibrationRevisionValid(revision) &&
        V2RendererAuthorizationValid(control, mapping_state) &&
        V2MappingStateValid(mapping_state);
}

bool V2CameraStateWithAdaptiveStateValid(float4 active, float4 control, uint4 mapping, V2AdaptiveCameraState camera_state) {
    return V2CameraParametersValid(active, control, asfloat(mapping)) &&
        V2AdaptiveCameraTailValid(V2_STATE_JOINT_PLANE_MODE_BITS(mapping), camera_state) &&
        mapping.x == V2CameraIntegrityWithAdaptiveState(active, control, mapping, camera_state) &&
        camera_state.clock.w == 1u &&
        (V2_STATE_FRAME_VALID(control) == 0.0f || camera_state.clock.z == 1u);
}

bool V2EmptyCameraParametersValid(
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
        V2RendererAuthorizationValid(control, mapping_state) &&
        V2MappingStateValid(mapping_state);
}

bool V2EmptyCameraStateWithAdaptiveStateValid(float4 active, float4 control, uint4 mapping, V2AdaptiveCameraState camera_state) {
    return V2EmptyCameraParametersValid(active, control, asfloat(mapping)) &&
        V2AdaptiveCameraTailValid(mapping.z, camera_state) &&
        camera_state.clock.w == 0u &&
        mapping.x == V2CameraIntegrityWithAdaptiveState(active, control, mapping, camera_state);
}

#endif
