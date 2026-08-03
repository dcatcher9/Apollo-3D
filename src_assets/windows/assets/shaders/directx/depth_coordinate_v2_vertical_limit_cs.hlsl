// Exact least column-wise Lipschitz majorant used to bound vertical disparity shear before the
// live row-wise invertibility projection. For each column this computes
//
//   q(y) = max_s Candidate(s) - step * abs(y - s),
//   step = 2.0 / target_w.
//
// The 2.0 bound is horizontal disparity pixels per source-image vertical pixel when the model and
// source grids are aspect matched. Dividing by target_w expresses each vertical depth-texel step in
// normalized source-U. Candidate remains immutable evidence. VerticalOut is an explicit
// diagnostic/intermediate; the existing row majorant consumes it and remains the sole live
// position authority.

Texture2D<float> Candidate : register(t0);
RWTexture2D<float> VerticalOut : register(u0);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2.hlsl"

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x;
    if (x >= target_w || target_h == 0u || target_w == 0u) {
        return;
    }
    float max_step = v2_max_vertical_shear / (float)target_w;
    float value = Candidate[uint2(x, 0u)];
    VerticalOut[uint2(x, 0u)] = value;
    [loop]
    for (uint y = 1u; y < target_h; ++y) {
        value = max(Candidate[uint2(x, y)], value - max_step);
        VerticalOut[uint2(x, y)] = value;
    }
    DeviceMemoryBarrier();

    value = VerticalOut[uint2(x, target_h - 1u)];
    [loop]
    for (int y = (int)target_h - 2; y >= 0; --y) {
        value = max(VerticalOut[uint2(x, (uint)y)], value - max_step);
        VerticalOut[uint2(x, (uint)y)] = value;
    }
}
