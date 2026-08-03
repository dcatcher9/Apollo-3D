// Map raw DAV2 into the canonical unbounded coordinate and desired signed one-eye source-U
// candidate. Shadow-only mode records both; explicit live V2 binds CandidateOut for position and
// CoordinateOut independently for visibility.

StructuredBuffer<float> InputBuffer : register(t0);
StructuredBuffer<float4> ShadowState : register(t1);
RWTexture2D<float> CoordinateOut : register(u0);
RWTexture2D<float> CandidateOut : register(u1);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2.hlsl"

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= target_w || id.y >= target_h) {
        return;
    }
    uint index = id.y * target_w + id.x;
    float4 active = ShadowState[0];
    float4 control = ShadowState[1];
    float4 shoulder = ShadowState[2];
    if (!V2CameraStateValid(active, control, shoulder, target_w * target_h)) {
        CoordinateOut[id.xy] = 0.0f;
        CandidateOut[id.xy] = 0.0f;
        return;
    }
    // Current-frame validity is independent of the retained scene camera. Unusable depth maps
    // flat, while the next usable no-cut field resumes the same coordinate without a gauge jump.
    if (V2_STATE_FRAME_VALID(control) < 0.5f) {
        CoordinateOut[id.xy] = 0.0f;
        CandidateOut[id.xy] = 0.0f;
        return;
    }
    // State resolve has already made validity/collapse atomic with the camera update. Re-reading
    // FrameStats here would duplicate that policy per texel and risk drift.
    float raw = InputBuffer[index];
    bool valid = !isnan(raw) && !isinf(raw);
    if (!valid) {
        CoordinateOut[id.xy] = 0.0f;
        CandidateOut[id.xy] = 0.0f;
        return;
    }
    float coordinate = (raw - V2_STATE_CENTER(active)) * V2_STATE_INVERSE_SCALE(active);
    float requested = v2_requested_gain *
        (V2CurveWithNearTau(
            coordinate,
            V2_STATE_EFFECTIVE_NEAR_LOG_TAU(shoulder)) -
         V2_STATE_CONVERGENCE_CURVE(active));
    float candidate = requested * V2_STATE_CONTAINER_SCALE(active);
    if (!V2Finite(coordinate) || !V2Finite(candidate)) {
        coordinate = 0.0f;
        candidate = 0.0f;
    }
    CoordinateOut[id.xy] = coordinate;
    CandidateOut[id.xy] = candidate;
}
