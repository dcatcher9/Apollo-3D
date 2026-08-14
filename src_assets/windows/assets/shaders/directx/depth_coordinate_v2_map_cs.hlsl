// Map raw DAV2 into the desired signed one-eye source-U candidate. The production entrypoint
// writes only that field. coordinate_main is a Dump-3D-only diagnostic entrypoint over the same
// authenticated source and is never dispatched by the ordinary live path.

StructuredBuffer<float> InputBuffer : register(t0);
StructuredBuffer<float4> ShadowState : register(t1);
Texture2D<uint> TensorExclusion : register(t2);
RWTexture2D<float> Output : register(u0);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2.hlsl"

bool CanonicalCoordinate(uint index, out float coordinate) {
    coordinate = 0.0f;
    float4 mapping_state = ShadowState[2];
    if (asuint(V2_STATE_RENDERER_AUTHORIZATION_BITS(mapping_state)) != V2_CONTRACT_TAG) {
        return false;
    }
    float4 active = ShadowState[0];
    // State resolve has already authenticated the current scene camera and made current-frame
    // validity atomic with it. The versioned authorization token above is the only state guard
    // needed in this per-texel pass.
    float raw = InputBuffer[index];
    if (!V2Finite(raw)) {
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
    // Synthetic letterbox cells are a nearest-boundary extension of the real analysis domain.
    // Sanitizing before ownership and both limiters prevents arbitrary model padding output from
    // entering a valid row/column envelope.
    uint2 sample_position = TensorExclusion[id.xy] != 0u ?
        DepthAnalysisClampCell(id.xy) : id.xy;
    uint index = sample_position.y * target_w + sample_position.x;
    float coordinate;
    if (!CanonicalCoordinate(index, coordinate)) {
        Output[id.xy] = 0.0f;
        return;
    }
    float4 active = ShadowState[0];
    float requested = v2_requested_gain *
        (V2Curve(coordinate) -
         V2_STATE_CONVERGENCE_CURVE(active));
    float candidate = V2PointwiseContainer(requested);
    if (!V2Finite(candidate)) {
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
    uint2 sample_position = TensorExclusion[id.xy] != 0u ?
        DepthAnalysisClampCell(id.xy) : id.xy;
    CanonicalCoordinate(sample_position.y * target_w + sample_position.x, coordinate);
    Output[id.xy] = coordinate;
}
