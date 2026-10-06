// Resolve GPU histogram into the Host adaptive quantile tail. P05 uses the lower crossing-bin
// edge and P95 the upper edge, clipped to the true range. These are FP32 256-bin bounds,
// not exact interpolated percentiles. All original Welford statistics remain untouched.
// The same resolve overwrites the independent normalization histogram. Its existing P02/P98
// EMA consumer and reset stay unchanged; only the duplicate full-depth traversal is removed.
StructuredBuffer<uint4> HistogramPartials : register(t0);
RWStructuredBuffer<float4> FrameStats : register(u0);
RWStructuredBuffer<uint> NormalizationHistogram : register(u1);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2_contract.generated.hlsl"

groupshared uint histogram[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID) {
    uint count = 0u;
    uint normalization_count = 0u;
    [loop] for (uint group = 0u; group < max(reduce_threads / 256u, 1u); ++group) {
        count += HistogramPartials[group * 128u + tid.x / 4u][tid.x % 4u];
        normalization_count += HistogramPartials[group * 128u + 64u + tid.x / 4u][tid.x % 4u];
    }
    histogram[tid.x] = count;
    NormalizationHistogram[tid.x] = normalization_count;
    GroupMemoryBarrierWithGroupSync();
    if (tid.x != 0u) return;
    float4 frame0 = FrameStats[V2_FRAME_STATS_VECTOR_MEAN];
    float4 frame1 = FrameStats[V2_FRAME_STATS_VECTOR_VALID_COUNT];
    float minimum = V2_FRAME_STATS_MINIMUM(frame0);
    float maximum = V2_FRAME_STATS_MAXIMUM(frame0);
    float range = maximum - minimum;
    float bin_width = range / v2_host_percentile_bin_count;
    uint total = 0u;
    bool low_found = false, high_found = false;
    float low = 0.0f, high = 0.0f;
    [loop] for (uint bin = 0u; bin < 256u; ++bin) {
        total += histogram[bin];
        if (!low_found && (float)total >=
            v2_host_percentile_low * V2_FRAME_STATS_VALID_COUNT(frame1)) {
            low = clamp(minimum + float(bin) * bin_width, minimum, maximum);
            low_found = true;
        }
        if (!high_found && (float)total >=
            v2_host_percentile_high * V2_FRAME_STATS_VALID_COUNT(frame1)) {
            high = clamp(minimum + float(bin + 1u) * bin_width, minimum, maximum);
            high_found = true;
        }
    }
    bool valid = v2_joint_plane_mode == V2_ADAPTIVE_POLICY_ID && V2_FRAME_STATS_VALID(frame1) == 1.0f &&
        !isnan(bin_width) && !isinf(bin_width) && bin_width >= 0.0f &&
        total > 0u && (float)total == V2_FRAME_STATS_VALID_COUNT(frame1) &&
        low_found && high_found && low <= high;
    float4 tail = 0.0f;
    if (valid) {
        V2_FRAME_STATS_PERCENTILE_LOW(tail) = low;
        V2_FRAME_STATS_PERCENTILE_HIGH(tail) = high;
        V2_FRAME_STATS_PERCENTILE_VALID(tail) = 1.0f;
        V2_FRAME_STATS_PERCENTILE_BIN_WIDTH(tail) = bin_width;
    }
    FrameStats[V2_FRAME_STATS_VECTOR_PERCENTILE_LOW] = tail;
}
