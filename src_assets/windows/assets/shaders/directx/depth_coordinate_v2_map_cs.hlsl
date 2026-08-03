// Map raw DAV2 into the desired signed one-eye source-U candidate. The production entrypoint
// writes only that field. coordinate_main is a Dump-3D-only diagnostic entrypoint over the same
// authenticated source and is never dispatched by the ordinary live path.

StructuredBuffer<float> InputBuffer : register(t0);
StructuredBuffer<float4> ShadowState : register(t1);
RWTexture2D<float> Output : register(u0);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2.hlsl"

bool CanonicalCoordinate(uint index, out float coordinate) {
    coordinate = 0.0f;
    float4 active = ShadowState[0];
    float4 control = ShadowState[1];
    float4 shoulder = ShadowState[2];
    if (!V2CameraStateValid(active, control, shoulder, target_w * target_h)) {
        return false;
    }
    // Current-frame validity is independent of the retained scene camera. Unusable depth maps
    // flat, while the next usable no-cut field resumes the same coordinate without a gauge jump.
    if (V2_STATE_FRAME_VALID(control) < 0.5f) {
        return false;
    }
    // State resolve has already made validity/collapse atomic with the camera update. Re-reading
    // FrameStats here would duplicate that policy per texel and risk drift.
    float raw = InputBuffer[index];
    bool valid = !isnan(raw) && !isinf(raw);
    if (!valid) {
        return false;
    }
    coordinate = (raw - V2_STATE_CENTER(active)) * V2_STATE_INVERSE_SCALE(active);
    if (!V2Finite(coordinate)) {
        coordinate = 0.0f;
        return false;
    }
    return true;
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= target_w || id.y >= target_h) {
        return;
    }
    uint index = id.y * target_w + id.x;
    float coordinate;
    if (!CanonicalCoordinate(index, coordinate)) {
        Output[id.xy] = 0.0f;
        return;
    }
    float4 active = ShadowState[0];
    float4 shoulder = ShadowState[2];
    float requested = v2_requested_gain *
        (V2CurveWithNearTau(
            coordinate,
            V2_STATE_EFFECTIVE_NEAR_LOG_TAU(shoulder)) -
         V2_STATE_CONVERGENCE_CURVE(active));
    float candidate = requested * V2_STATE_CONTAINER_SCALE(active);
    if (!V2Finite(coordinate) || !V2Finite(candidate)) {
        candidate = 0.0f;
    }
    Output[id.xy] = candidate;
}

[numthreads(16, 16, 1)]
void coordinate_main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= target_w || id.y >= target_h) {
        return;
    }
    float coordinate;
    CanonicalCoordinate(id.y * target_w + id.x, coordinate);
    Output[id.xy] = coordinate;
}
