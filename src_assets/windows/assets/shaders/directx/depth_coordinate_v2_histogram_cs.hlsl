// Mode-3 robust raw-depth histogram. Every reduction group overwrites its 256 bins;
// no clear, global atomic accumulator, CPU readback or reuse dispatch is required.
// Unlike the private cut-normalization histogram, all finite signed eligible values count.
StructuredBuffer<float> InputBuffer : register(t0);
Texture2D<uint> TensorExclusion : register(t1);
StructuredBuffer<float4> FrameStats : register(t2);
RWStructuredBuffer<uint4> HistogramPartials : register(u0);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2_contract.generated.hlsl"

groupshared uint histogram[256];

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID,
          uint3 gid : SV_GroupID) {
    histogram[tid.x] = 0u;
    GroupMemoryBarrierWithGroupSync();
    float4 frame0 = FrameStats[V2_FRAME_STATS_VECTOR_MEAN];
    float4 frame1 = FrameStats[V2_FRAME_STATS_VECTOR_VALID_COUNT];
    float minimum = V2_FRAME_STATS_MINIMUM(frame0);
    float range = V2_FRAME_STATS_MAXIMUM(frame0) - minimum;
    bool valid = v2_joint_plane_mode == 3u && V2_FRAME_STATS_VALID(frame1) == 1.0f &&
        !isnan(range) && !isinf(range) && range >= 0.0f;
    float inverse_range = range > 0.0f ? 256.0f / range : 0.0f;
    valid = valid && !isnan(inverse_range) && !isinf(inverse_range);
    uint position_y = dtid.x / target_w;
    uint position_x = dtid.x - position_y * target_w;
    uint stride_y = reduce_threads / target_w;
    uint stride_x = reduce_threads - stride_y * target_w;
    [loop] for (uint index = dtid.x; index < target_w * target_h; index += reduce_threads) {
        if (valid && TensorExclusion[uint2(position_x, position_y)] == 0u) {
            float value = InputBuffer[index];
            if (!isnan(value) && !isinf(value)) {
                uint bin = range > 0.0f ?
                    min((uint)max((value - minimum) * inverse_range, 0.0f), 255u) : 0u;
                InterlockedAdd(histogram[bin], 1u);
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
        HistogramPartials[gid.x * 64u + tid.x] = uint4(
            histogram[bin], histogram[bin + 1u], histogram[bin + 2u], histogram[bin + 3u]);
    }
}
