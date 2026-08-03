// Exact least row-wise Lipschitz majorant of the column-conditioned VerticalMajorant input. For
// each row this computes
//
//   q(x) = max_s VerticalMajorant(s) - step * abs(x - s).
//
// Composed after the exact column majorant, this is the least anisotropic 2D majorant under the
// selected vertical-shear and horizontal-slope bounds. Therefore q never lowers the immutable
// candidate, and adjacent horizontal texels differ by at most step. Near peaks keep their requested
// disparity while the minimum necessary transition is assigned to the less important far/background
// side. One GPU thread owns a complete row so the serial dependency never crosses a thread boundary.
// FinalOut is the live position authority.

Texture2D<float> VerticalMajorant : register(t0);
RWTexture2D<float> FinalOut : register(u0);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2.hlsl"

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint y = id.x;
    if (y >= target_h || target_w == 0u) {
        return;
    }
    float max_step = v2_max_horizontal_slope / (float)target_w;
    float value = VerticalMajorant[uint2(0u, y)];
    FinalOut[uint2(0u, y)] = value;
    [loop]
    for (uint x = 1u; x < target_w; ++x) {
        value = max(VerticalMajorant[uint2(x, y)], value - max_step);
        FinalOut[uint2(x, y)] = value;
    }
    DeviceMemoryBarrier();

    value = FinalOut[uint2(target_w - 1u, y)];
    [loop]
    for (int x = (int)target_w - 2; x >= 0; --x) {
        value = max(FinalOut[uint2((uint)x, y)], value - max_step);
        FinalOut[uint2((uint)x, y)] = value;
    }
}
