// 256-bin histogram of the raw disparity buffer, binned over THIS frame's raw min/max
// (already reduced by depth_minmax_cs into MinMaxRaw). depth_minmax_ema_cs then scans the
// histogram for the permanent robust P2/P98 bounds instead of using the raw min/max
// directly -- a handful of outlier pixels (DA-V2's near-spike tail) can otherwise stretch
// the normalization range and squeeze the whole scene's parallax. Fully GPU-resident, no
// CPU readback; dispatched only when percentile normalization is enabled.

StructuredBuffer<float>  InputBuffer : register(t0);
RWStructuredBuffer<uint> Histogram   : register(u0);  // 256 bins, reset by depth_minmax_ema_cs
RWByteAddressBuffer      MinMaxRaw   : register(u1);  // read-only here: [0]=min bits, [4]=max bits

// Shared depth-pass cbuffer (slots 11-12 = the percentile bounds consumed by depth_minmax_ema_cs).
#include "include/depth_constants.hlsl"

#define GROUP_SIZE 256
#define NUM_BINS 256
groupshared uint g_hist[NUM_BINS];

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID) {
    g_hist[tid.x] = 0u;  // GROUP_SIZE == NUM_BINS: one bin per thread
    GroupMemoryBarrierWithGroupSync();

    // Bin over this frame's raw range (from the preceding depth_minmax_cs dispatch).
    float vmin = asfloat(MinMaxRaw.Load(0));
    float vmax = asfloat(MinMaxRaw.Load(4));
    float inv_range = (float)NUM_BINS / max(vmax - vmin, 1e-12f);

    uint count = target_w * target_h;
    [loop]
    for (uint idx = dtid.x; idx < count; idx += reduce_threads) {
        float v = InputBuffer[idx];
        if (isnan(v) || v < 0.0f) v = 0.0f;
        // Must match the exact transform in depth_minmax_cs / buffer_to_tex_cs.
        if (output_transform == 1) v = 1.0f / (v + depth_shift);
        uint bin = min((uint)((v - vmin) * inv_range), NUM_BINS - 1u);
        InterlockedAdd(g_hist[bin], 1u);
    }

    GroupMemoryBarrierWithGroupSync();
    if (g_hist[tid.x] > 0u) {
        InterlockedAdd(Histogram[tid.x], g_hist[tid.x]);
    }
}
