// Continuous Host mean zero and robust depth amplitude. Only authenticated infer
// publication observes the controller; reuse holds the complete tuple.

StructuredBuffer<float4> FrameStats : register(t0);
StructuredBuffer<float4> DepthHealthState : register(t1);
// Publish the mixed ABI as integer words. A floating typed UAV store may flush small mode
// tokens even after integer local computation; numeric camera words use explicit bit casts.
RWStructuredBuffer<uint4> ShadowState : register(u0);

#include "include/depth_coordinate_v2_state_resolve_common.hlsl"

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    ResolveDepthCoordinateV2State();
}
