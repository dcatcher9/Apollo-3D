// Mode-3 raw-depth scan, shared by geometry P05/P95 and cut-normalization P02/P98.
// Every reduction group overwrites two independent 256-bin populations. Only the raw/exclusion
// traversal is shared: signed finite geometry bounds and nonnegative normalization bounds retain
// their exact, distinct validity and bin arithmetic. No clear, readback or reuse dispatch is needed.
StructuredBuffer<float> InputBuffer : register(t0);
Texture2D<uint> TensorExclusion : register(t1);
StructuredBuffer<float4> FrameStats : register(t2);
RWStructuredBuffer<uint4> HistogramPartials : register(u0);
RWByteAddressBuffer MinMaxRaw : register(u1); // read-only existing normalization reduction

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2_contract.generated.hlsl"

groupshared uint histogram[256];
groupshared uint normalization_histogram[256];

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID,
          uint3 gid : SV_GroupID) {
    histogram[tid.x] = 0u;
    normalization_histogram[tid.x] = 0u;
    GroupMemoryBarrierWithGroupSync();
    float4 frame0 = FrameStats[V2_FRAME_STATS_VECTOR_MEAN];
    float4 frame1 = FrameStats[V2_FRAME_STATS_VECTOR_VALID_COUNT];
    float minimum = V2_FRAME_STATS_MINIMUM(frame0);
    float range = V2_FRAME_STATS_MAXIMUM(frame0) - minimum;
    bool valid = v2_joint_plane_mode == 3u && V2_FRAME_STATS_VALID(frame1) == 1.0f &&
        !isnan(range) && !isinf(range) && range >= 0.0f;
    float inverse_range = range > 0.0f ? 256.0f / range : 0.0f;
    valid = valid && !isnan(inverse_range) && !isinf(inverse_range);
    // These are the old normalization shader's unsigned-bit extrema, including signed zero.
    // Do not substitute the signed finite geometry extrema or its unfloored inverse range.
    float normalization_minimum = asfloat(MinMaxRaw.Load(0));
    float normalization_maximum = asfloat(MinMaxRaw.Load(4));
    uint normalization_valid_count = MinMaxRaw.Load(8);
    uint eligible_count = MinMaxRaw.Load(12);
    bool normalization_valid = eligible_count > 0u && normalization_valid_count == eligible_count &&
        !isnan(normalization_minimum) && !isinf(normalization_minimum) &&
        !isnan(normalization_maximum) && !isinf(normalization_maximum) &&
        normalization_maximum >= normalization_minimum;
    float normalization_inverse_range = normalization_valid ?
        256.0f / max(normalization_maximum - normalization_minimum, 1e-12f) : 0.0f;
    uint position_y = dtid.x / target_w;
    uint position_x = dtid.x - position_y * target_w;
    uint stride_y = reduce_threads / target_w;
    uint stride_x = reduce_threads - stride_y * target_w;
    [loop] for (uint index = dtid.x; index < target_w * target_h; index += reduce_threads) {
        if (TensorExclusion[uint2(position_x, position_y)] == 0u) {
            float value = InputBuffer[index];
            if (!isnan(value) && !isinf(value)) {
                if (valid) {
                    uint bin = range > 0.0f ?
                        min((uint)max((value - minimum) * inverse_range, 0.0f), 255u) : 0u;
                    InterlockedAdd(histogram[bin], 1u);
                }
                if (normalization_valid && value >= 0.0f) {
                    uint bin = min((uint)((value - normalization_minimum) *
                        normalization_inverse_range), 255u);
                    InterlockedAdd(normalization_histogram[bin], 1u);
                }
            }
        }
        position_x += stride_x;
        position_y += stride_y;
        if (position_x >= target_w) {
            position_x -= target_w;
            position_y++;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // One lane owns one whole uint4 record; concurrent component writes cannot race.
    if (tid.x < 64u) {
        uint bin = tid.x * 4u;
        HistogramPartials[gid.x * 128u + tid.x] = uint4(
            histogram[bin], histogram[bin + 1u], histogram[bin + 2u], histogram[bin + 3u]);
        HistogramPartials[gid.x * 128u + 64u + tid.x] = uint4(
            normalization_histogram[bin], normalization_histogram[bin + 1u],
            normalization_histogram[bin + 2u], normalization_histogram[bin + 3u]);
    }
}
