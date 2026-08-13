// Exact least row-wise Lipschitz majorant of the orientation-selective vertical field. For each
// row this computes
//
//   q(x) = max_s VerticalConditioned(s) - step * abs(x - s),
//   step = max_horizontal_slope / target_w.
//
// The vertical share may raise or lower the immutable candidate, but it already satisfies the
// vertical shear bound. This pure horizontal majorant preserves that bound, enforces the
// contractive inverse-warp slope, and avoids the lateral lowering introduced by a horizontal
// minorant component. One GPU thread owns a complete row.

Texture2D<float> VerticalConditioned : register(t0);
RWTexture2D<float> FinalOut : register(u0);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2.hlsl"

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint y = id.x;
    if (y >= target_h || target_w == 0u) {
        return;
    }

    float max_step = v2_max_horizontal_slope / DepthAnalysisContentWidthCells();
    float value = VerticalConditioned[uint2(0u, y)];
    FinalOut[uint2(0u, y)] = value;
    [loop]
    for (uint x = 1u; x < target_w; ++x) {
        value = max(VerticalConditioned[uint2(x, y)], value - max_step);
        FinalOut[uint2(x, y)] = value;
    }
    DeviceMemoryBarrier();

    value = FinalOut[uint2(target_w - 1u, y)];
    [loop]
    for (int x = (int)target_w - 2; x >= 0; --x) {
        const uint2 position = uint2((uint)x, y);
        value = max(FinalOut[position], value - max_step);
        FinalOut[position] = value;
    }
}
