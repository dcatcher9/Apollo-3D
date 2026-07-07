// Parallel reduction: min/max of the raw disparity buffer (the previous frame's TensorRT
// output) into two uint bits. Fully GPU-resident so no CPU readback / flush is introduced
// (which would stall the pipeline and can starve DWM -> TDR).
//
// Depth Anything V2's relative output is affine-invariant (scale/shift arbitrary, drifting),
// so it must be min/max-normalized before use. For non-negative floats, asuint() preserves
// ordering, so InterlockedMin/InterlockedMax on the raw bit pattern yields the correct min/max.

StructuredBuffer<float> InputBuffer : register(t0);
RWByteAddressBuffer     MinMaxOut   : register(u0);  // [0]=min bits, [4]=max bits

cbuffer Constants : register(b0) {
    uint target_w;
    uint target_h;
    uint is_hdr;
    float ema_alpha;
    float minmax_alpha;
    uint reduce_threads;  // total threads in this dispatch (groups * GROUP_SIZE)
    uint output_transform;  // 0 = identity (DA-V2 disparity); 1 = shifted reciprocal (DA-V3 depth -> disparity)
    float depth_shift;  // shift in 1/(depth + depth_shift) when output_transform == 1
    float snap_ratio;       // unused here; shared 48-byte cbuffer layout parity
    float floor_frac;       // unused here; layout parity
    float floor_ref_alpha;  // unused here; layout parity
    float pad0;
};

#define GROUP_SIZE 256
groupshared uint g_min[GROUP_SIZE];
groupshared uint g_max[GROUP_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID) {
    uint count = target_w * target_h;
    uint lmin = 0xFFFFFFFFu;
    uint lmax = 0u;

    // Grid-stride: each thread folds in every reduce_threads-th element.
    [loop]
    for (uint idx = dtid.x; idx < count; idx += reduce_threads) {
        float v = InputBuffer[idx];
        if (isnan(v) || v < 0.0f) v = 0.0f;
        // DA-V3 emits scale-shift-invariant DEPTH (larger = farther); convert to disparity
        // (larger = closer) with a SHIFTED reciprocal 1/(depth + shift). The shift keeps the
        // near end finite (depth->0 gives 1/shift, not a ~1e6 spike), so no outlier hijacks the
        // min/max -- the clean alternative to 1/depth + robust clipping (iw3's approach). Stays
        // non-negative, so asuint() ordering still holds.
        if (output_transform == 1) v = 1.0f / (v + depth_shift);
        uint u = asuint(v);
        lmin = min(lmin, u);
        lmax = max(lmax, u);
    }

    g_min[tid.x] = lmin;
    g_max[tid.x] = lmax;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = GROUP_SIZE / 2; s > 0; s >>= 1) {
        if (tid.x < s) {
            g_min[tid.x] = min(g_min[tid.x], g_min[tid.x + s]);
            g_max[tid.x] = max(g_max[tid.x], g_max[tid.x + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid.x == 0) {
        MinMaxOut.InterlockedMin(0, g_min[0]);
        MinMaxOut.InterlockedMax(4, g_max[0]);
    }
}
