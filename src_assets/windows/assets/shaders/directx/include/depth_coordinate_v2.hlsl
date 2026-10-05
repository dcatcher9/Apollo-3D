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

bool V2JointPlaneModeValid(uint mode) {
    return mode == 0u || mode == 3u;
}

bool V2JointPlaneConstantsValid() {
    return V2JointPlaneModeValid(v2_joint_plane_mode) &&
        v2_joint_plane_reserved0 == 0u && v2_joint_plane_reserved1 == 0u &&
        v2_joint_plane_reserved2 == 0u &&
        v2_joint_observation_reserved0 == 0u && v2_joint_observation_reserved1 == 0u;
}

float V2DisplayBudget() {
    return v2_direct_container_limit;
}

float V2BoundDisplayParallax(float value) {
    return v2_joint_plane_mode == 3u ? clamp(value, -V2DisplayBudget(), V2DisplayBudget()) : value;
}

// All controller payloads remain integers, including mixed float words, until UAV publication.
// This avoids losing small uint tokens through floating-point temporaries on native GPUs.
struct V2GainState {
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
    return mode != 0u ? (checksum ^ mode) * 16777619u : checksum;
}

uint V2CameraIntegrityWithGain(float4 active, float4 control, uint4 mapping, V2GainState gain) {
    uint checksum = V2CameraIntegrityBase(active, control, V2_STATE_JOINT_PLANE_MODE_BITS(mapping));
    if (V2_STATE_JOINT_PLANE_MODE_BITS(mapping) == 3u) {
        [unroll] for (uint component = 0u; component < 4u; ++component)
            checksum = (checksum ^ gain.clock[component]) * 16777619u;
        [unroll] for (uint component = 0u; component < 4u; ++component)
            checksum = (checksum ^ gain.target[component]) * 16777619u;
        [unroll] for (uint component = 0u; component < 4u; ++component)
            checksum = (checksum ^ gain.seed_times[component]) * 16777619u;
        [unroll] for (uint component = 0u; component < 4u; ++component)
            checksum = (checksum ^ gain.seed_values[component]) * 16777619u;
    }
    return checksum;
}

bool V2GainTailValid(uint mode, V2GainState gain) {
    if (mode == 0u)
        return all(gain.clock == 0u) && all(gain.target == 0u) &&
            all(gain.seed_times == 0u) && all(gain.seed_values == 0u);
    float4 target = asfloat(gain.target);
    float2 seed = asfloat(gain.seed_values.xy);
    if (mode != 3u || gain.clock.z > 1u || gain.clock.w > 1u ||
        any(gain.seed_values.zw != 0u) ||
        (gain.clock.z != 0u && all(gain.clock.xy == 0u)) ||
        !V2Finite(target.x) || !V2Finite(target.y) || !V2Finite(target.z) ||
        !V2Finite(target.w) || !V2Finite(seed.x) || !V2Finite(seed.y) ||
        target.w < 0.0f || target.w > v2_direct_container_limit) return false;
    bool no_target = all(target.xyz == 0.0f);
    bool valid_target = target.y > 0.0f && target.z > 0.0f &&
        abs(target.y * target.z - 1.0f) <= 2.0e-6f;
    if ((!no_target && !valid_target) || (gain.clock.z != 0u && !valid_target)) return false;
    if ((!no_target && target.z < v2_raw_coordinate_scale) ||
        (seed.x != 0.0f && seed.x < v2_raw_coordinate_scale)) return false;
    if (gain.clock.w == 0u)
        return gain.clock.z == 0u && all(gain.seed_times == 0u) && all(seed == 0.0f);
    if (all(gain.seed_times.xy == 0u) ||
        !V2ClockAtLeast(gain.seed_times.zw, gain.seed_times.xy) ||
        !V2ClockAtLeast(gain.clock.xy, gain.seed_times.zw) ||
        seed.x <= 0.0f || target.w <= 0.0f) return false;
    return all(gain.seed_times.zw == gain.seed_times.xy);
}

// Authenticate both scene-camera coordinates. The inverse scale and revision are included
// so a same-tag state assembled from different camera generations cannot accidentally validate.
// Starting from zero deliberately keeps the generated empty-state word zero while still making
// every acquired camera carry a deterministic non-zero checksum in the ordinary case. Integer
// overflow is the specified modulo-2^32 checksum behavior.
uint V2CameraCenterIntegrityBits(
    float4 active,
    float4 control,
    float4 mapping_state
) {
    uint mode = asuint(V2_STATE_JOINT_PLANE_MODE_BITS(mapping_state));
    // Preserve every legacy mode-zero seal, including the generated zero empty-state word.
    return V2CameraIntegrityBase(active, control, mode);
}

bool V2CameraCenterIntegrityValid(float4 active, float4 control, float4 mapping_state) {
    return asuint(V2_STATE_CAMERA_CENTER_INTEGRITY_BITS(mapping_state)) ==
        V2CameraCenterIntegrityBits(active, control, mapping_state);
}

void V2SealCameraCenter(float4 active, float4 control, inout float4 mapping_state) {
    V2_STATE_CAMERA_CENTER_INTEGRITY_BITS(mapping_state) = asfloat(
        V2CameraCenterIntegrityBits(active, control, mapping_state));
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
    uint mode = asuint(V2_STATE_JOINT_PLANE_MODE_BITS(mapping_state));
    float inverse_scale = V2_STATE_INVERSE_SCALE(active);
    bool scale_valid = mode == 0u ?
        V2ApproximatelyEqual(inverse_scale, expected_inverse_scale) :
        V2Finite(inverse_scale) && inverse_scale > 0.0f &&
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

bool V2CameraStateValid(float4 active, float4 control, float4 mapping_state) {
    return asuint(V2_STATE_JOINT_PLANE_MODE_BITS(mapping_state)) == 0u &&
        V2CameraParametersValid(active, control, mapping_state) &&
        V2CameraCenterIntegrityValid(active, control, mapping_state);
}

bool V2CameraStateWithGainValid(float4 active, float4 control, uint4 mapping, V2GainState gain) {
    return V2CameraParametersValid(active, control, asfloat(mapping)) &&
        V2GainTailValid(V2_STATE_JOINT_PLANE_MODE_BITS(mapping), gain) &&
        mapping.x == V2CameraIntegrityWithGain(active, control, mapping, gain) &&
        (mapping.z == 0u || (gain.clock.w == 1u &&
          (V2_STATE_FRAME_VALID(control) == 0.0f || gain.clock.z == 1u)));
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

bool V2EmptyCameraStateValid(float4 active, float4 control, float4 mapping_state) {
    return asuint(V2_STATE_JOINT_PLANE_MODE_BITS(mapping_state)) == 0u &&
        V2EmptyCameraParametersValid(active, control, mapping_state) &&
        V2CameraCenterIntegrityValid(active, control, mapping_state);
}

bool V2EmptyCameraStateWithGainValid(float4 active, float4 control, uint4 mapping, V2GainState gain) {
    return V2EmptyCameraParametersValid(active, control, asfloat(mapping)) &&
        V2GainTailValid(mapping.z, gain) &&
        (mapping.z == 0u || gain.clock.w == 0u) &&
        mapping.x == V2CameraIntegrityWithGain(active, control, mapping, gain);
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

// The production reference applies this fixed curve. Adaptive Host mode is linear with an
// independent hard representation bound.
float V2Curve(float coordinate) {
    return V2CurveWithNearTau(coordinate, v2_near_log_tau);
}

// Bound each texel independently instead of shrinking the whole frame to accommodate one raw
// outlier.  The fourth-order soft container is odd, monotone, has unit derivative at the zero
// plane, and approaches (but does not reach) the representation limit:
//
//   contained(x) = x / fourth_root(1 + (x / limit)^4)
//
// Its derivative is positive and no greater than one, so it cannot introduce a steeper cliff.
// The final clamp is only a floating-point safety belt for the strict packed-field contract.
float V2PointwiseContainer(float requested) {
    if (!V2Finite(requested) ||
        !V2Finite(v2_direct_container_limit) ||
        v2_direct_container_limit <= 0.0f) {
        return 0.0f;
    }
    // Factor the larger magnitude out of the fourth root. This is algebraically identical
    // to the expression above but never forms an unbounded normalized^4 intermediate.
    float requested_magnitude = abs(requested);
    float smaller = min(requested_magnitude, v2_direct_container_limit);
    float larger = max(requested_magnitude, v2_direct_container_limit);
    float ratio = smaller / larger;
    float ratio_squared = ratio * ratio;
    float inverse_fourth_root = rsqrt(sqrt(
        1.0f + ratio_squared * ratio_squared));
    float contained_magnitude = smaller * inverse_fourth_root;
    float contained = requested < 0.0f ? -contained_magnitude : contained_magnitude;
    return clamp(
        contained,
        -v2_direct_container_limit,
        v2_direct_container_limit
    );
}

#endif
