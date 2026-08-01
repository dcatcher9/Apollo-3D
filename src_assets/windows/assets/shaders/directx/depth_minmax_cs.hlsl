// Parallel reduction: min/max of the raw disparity buffer (the previous frame's TensorRT
// output) into two uint bits. Fully GPU-resident so no CPU readback / flush is introduced
// (which would stall the pipeline and can starve DWM -> TDR).
//
// Depth Anything V2's relative output is affine-invariant (scale/shift arbitrary, drifting),
// so it must be min/max-normalized before use. For non-negative floats, asuint() preserves
// ordering, so InterlockedMin/InterlockedMax on the raw bit pattern yields the correct min/max.

StructuredBuffer<float> InputBuffer : register(t0);
RWByteAddressBuffer     MinMaxOut   : register(u0);  // [0]=min bits, [4]=max bits, [8]=valid count

#include "include/depth_constants.hlsl"

#define GROUP_SIZE 256
groupshared uint g_min[GROUP_SIZE];
groupshared uint g_max[GROUP_SIZE];
groupshared uint g_valid[GROUP_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID) {
    uint count = target_w * target_h;
    uint lmin = 0xFFFFFFFFu;
    uint lmax = 0u;
    uint lvalid = 0u;

    // Grid-stride: each thread folds in every reduce_threads-th element.
    [loop]
    for (uint idx = dtid.x; idx < count; idx += reduce_threads) {
        float v = InputBuffer[idx];
        if (!isnan(v) && !isinf(v) && v >= 0.0f) {
            uint u = asuint(v);
            lmin = min(lmin, u);
            lmax = max(lmax, u);
            lvalid++;
        }
    }

    g_min[tid.x] = lmin;
    g_max[tid.x] = lmax;
    g_valid[tid.x] = lvalid;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = GROUP_SIZE / 2; s > 0; s >>= 1) {
        if (tid.x < s) {
            g_min[tid.x] = min(g_min[tid.x], g_min[tid.x + s]);
            g_max[tid.x] = max(g_max[tid.x], g_max[tid.x + s]);
            g_valid[tid.x] += g_valid[tid.x + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid.x == 0) {
        if (g_valid[0] > 0u) {
            MinMaxOut.InterlockedMin(0, g_min[0]);
            MinMaxOut.InterlockedMax(4, g_max[0]);
            MinMaxOut.InterlockedAdd(8, g_valid[0]);
        }
    }
}
