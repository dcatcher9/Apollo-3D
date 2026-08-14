// Merge per-group Welford records into two frame-stat float4 vectors:
//   [0] = {mean, population_std, minimum, maximum}
//   [1] = {valid_count, eligible_count, valid, reserved}
// `eligible_count` is the real tensor-content population: the full tensor for ordinary full-frame
// analysis, or the unpadded content population for a contain-fit ROI. Every admitted texel must be
// finite before the frame can publish.

StructuredBuffer<float4> Partials : register(t0);
RWStructuredBuffer<float4> FrameStats : register(u0);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2_contract.generated.hlsl"

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint group_count = max(reduce_threads / 256u, 1u);
    float mean = 0.0f;
    float m2 = 0.0f;
    float minimum = 3.402823466e+38f;
    float maximum = -3.402823466e+38f;
    uint count = 0u;
    uint eligible_count = 0u;

    [loop]
    for (uint group = 0u; group < group_count; ++group) {
        float4 record = Partials[group * 2u];
        float4 range_record = Partials[group * 2u + 1u];
        eligible_count += (uint)max(range_record.y, 0.0f);
        uint right_count = (uint)max(record.z, 0.0f);
        if (right_count == 0u) {
            continue;
        }
        float right_mean = record.x;
        float right_m2 = record.y;
        if (count == 0u) {
            mean = right_mean;
            m2 = right_m2;
            minimum = record.w;
            maximum = range_record.x;
            count = right_count;
        } else {
            uint merged_count = count + right_count;
            float delta = right_mean - mean;
            mean += delta * ((float)right_count / (float)merged_count);
            m2 += right_m2 + delta * delta *
                ((float)count * (float)right_count / (float)merged_count);
            minimum = min(minimum, record.w);
            maximum = max(maximum, range_record.x);
            count = merged_count;
        }
    }

    uint texel_count = eligible_count;
    float variance = count > 0u ? max(m2 / (float)count, 0.0f) : 0.0f;
    float stddev = sqrt(variance);
    bool valid = count == texel_count && texel_count > 0u && !isnan(mean) && !isinf(mean) &&
        !isnan(stddev) && !isinf(stddev) &&
        !isnan(minimum) && !isinf(minimum) &&
        !isnan(maximum) && !isinf(maximum) && maximum >= minimum;
    float4 frame0 = 0.0f;
    float4 frame1 = 0.0f;
    if (valid) {
        V2_FRAME_STATS_MEAN(frame0) = mean;
        V2_FRAME_STATS_POPULATION_STD(frame0) = stddev;
        V2_FRAME_STATS_MINIMUM(frame0) = minimum;
        V2_FRAME_STATS_MAXIMUM(frame0) = maximum;
    }
    V2_FRAME_STATS_VALID_COUNT(frame1) = (float)count;
    V2_FRAME_STATS_TEXEL_COUNT(frame1) = (float)texel_count;
    V2_FRAME_STATS_VALID(frame1) = valid ? 1.0f : 0.0f;
    V2_FRAME_STATS_RESERVED(frame1) = 0.0f;
    FrameStats[V2_FRAME_STATS_VECTOR_MEAN] = frame0;
    FrameStats[V2_FRAME_STATS_VECTOR_VALID_COUNT] = frame1;
}
