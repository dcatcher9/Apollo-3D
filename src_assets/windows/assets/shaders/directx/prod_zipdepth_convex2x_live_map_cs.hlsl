// Stage-2 history-free raw refined-depth mapping.  Coarse DAV2 has already resolved moments,
// range, cuts, camera state and the complete history tuple.  This pass consumes only the fused
// exact-2x raw output plus that authenticated coarse camera state, then enters the existing
// nonlinear V2Curve before ownership and both spatial limiters.

StructuredBuffer<float> RefinedDepth : register(t0);
StructuredBuffer<float4> CoarseShadowState : register(t1);
Texture2D<uint> FieldExclusion : register(t2);
RWTexture2D<float> CandidateOut : register(u0);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2.hlsl"

bool RefinedCanonicalCoordinate(uint index, out float coordinate) {
    coordinate = 0.0f;
    float4 mapping_state = CoarseShadowState[2];
    if (asuint(V2_STATE_RENDERER_AUTHORIZATION_BITS(mapping_state)) != V2_CONTRACT_TAG) {
        return false;
    }
    float raw = RefinedDepth[index];
    if (!V2Finite(raw)) {
        return false;
    }
    float4 active = CoarseShadowState[0];
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

    // The guidance preprocess generated this exclusion texture from the exact doubled content
    // rectangle.  Clamp synthetic support to its nearest real field cell without a second fit.
    uint2 sample_position = FieldExclusion[id.xy] != 0u ?
        DepthAnalysisClampCell(id.xy) : id.xy;
    float coordinate;
    if (!RefinedCanonicalCoordinate(
            sample_position.y * target_w + sample_position.x, coordinate)) {
        CandidateOut[id.xy] = 0.0f;
        return;
    }

    float4 active = CoarseShadowState[0];
    float requested = v2_requested_gain *
        (V2Curve(coordinate) - V2_STATE_CONVERGENCE_CURVE(active));
    float candidate = V2PointwiseContainer(requested);
    CandidateOut[id.xy] = V2Finite(candidate) ? candidate : 0.0f;
}
