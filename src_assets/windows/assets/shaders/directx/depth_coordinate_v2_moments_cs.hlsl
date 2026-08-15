// Numerically stable per-group Welford moments over raw DAV2 output. Three uint4 records per group:
//   [3*g+0] = {asuint(mean), asuint(M2), finite_count, asuint(finite_minimum)}
//   [3*g+1] = {asuint(finite_maximum), eligible_count,
//              normalization_minimum_bits, normalization_maximum_bits}
//   [3*g+2] = {normalization_valid_count, 0, 0, 0}
// V2 frame statistics admit every finite value. The normalization reduction retains the older,
// narrower finite && value >= 0 contract and its unsigned-bit extrema exactly, including -0.
// Every group overwrites its records, so no accumulator clear or CPU readback is required.

StructuredBuffer<float> InputBuffer : register(t0);
Texture2D<uint> TensorExclusion : register(t1);
RWStructuredBuffer<uint4> Partials : register(u0);

#include "include/depth_constants.hlsl"

#define GROUP_SIZE 256
groupshared float g_mean[GROUP_SIZE];
groupshared float g_m2[GROUP_SIZE];
groupshared float g_minimum[GROUP_SIZE];
groupshared float g_maximum[GROUP_SIZE];
groupshared uint g_count[GROUP_SIZE];
groupshared uint g_eligible[GROUP_SIZE];
groupshared uint g_normalization_minimum[GROUP_SIZE];
groupshared uint g_normalization_maximum[GROUP_SIZE];
groupshared uint g_normalization_valid_count[GROUP_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID,
          uint3 tid : SV_GroupThreadID,
          uint3 gid : SV_GroupID) {
    uint element_count = target_w * target_h;
    float mean = 0.0f;
    float m2 = 0.0f;
    float minimum = 3.402823466e+38f;
    float maximum = -3.402823466e+38f;
    uint count = 0u;
    uint eligible = 0u;
    uint normalization_minimum = 0xFFFFFFFFu;
    uint normalization_maximum = 0u;
    uint normalization_valid_count = 0u;

    [loop]
    for (uint index = dtid.x; index < element_count; index += reduce_threads) {
        uint2 position = uint2(index % target_w, index / target_w);
        if (TensorExclusion[position] != 0u) {
            continue;
        }
        eligible++;
        float value = InputBuffer[index];
        if (!isnan(value) && !isinf(value)) {
            count++;
            float delta = value - mean;
            mean += delta / (float)count;
            float delta2 = value - mean;
            m2 += delta * delta2;
            minimum = min(minimum, value);
            maximum = max(maximum, value);
            if (value >= 0.0f) {
                uint normalization_bits = asuint(value);
                normalization_minimum = min(normalization_minimum, normalization_bits);
                normalization_maximum = max(normalization_maximum, normalization_bits);
                normalization_valid_count++;
            }
        }
    }

    g_mean[tid.x] = mean;
    g_m2[tid.x] = m2;
    g_minimum[tid.x] = minimum;
    g_maximum[tid.x] = maximum;
    g_count[tid.x] = count;
    g_eligible[tid.x] = eligible;
    g_normalization_minimum[tid.x] = normalization_minimum;
    g_normalization_maximum[tid.x] = normalization_maximum;
    g_normalization_valid_count[tid.x] = normalization_valid_count;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = GROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid.x < stride) {
            uint right_count = g_count[tid.x + stride];
            g_eligible[tid.x] += g_eligible[tid.x + stride];
            g_normalization_minimum[tid.x] = min(
                g_normalization_minimum[tid.x],
                g_normalization_minimum[tid.x + stride]);
            g_normalization_maximum[tid.x] = max(
                g_normalization_maximum[tid.x],
                g_normalization_maximum[tid.x + stride]);
            g_normalization_valid_count[tid.x] +=
                g_normalization_valid_count[tid.x + stride];
            if (right_count > 0u) {
                uint left_count = g_count[tid.x];
                if (left_count == 0u) {
                    g_mean[tid.x] = g_mean[tid.x + stride];
                    g_m2[tid.x] = g_m2[tid.x + stride];
                    g_minimum[tid.x] = g_minimum[tid.x + stride];
                    g_maximum[tid.x] = g_maximum[tid.x + stride];
                    g_count[tid.x] = right_count;
                } else {
                    uint merged_count = left_count + right_count;
                    float delta = g_mean[tid.x + stride] - g_mean[tid.x];
                    g_mean[tid.x] += delta * ((float)right_count / (float)merged_count);
                    g_m2[tid.x] += g_m2[tid.x + stride] + delta * delta *
                        ((float)left_count * (float)right_count / (float)merged_count);
                    g_minimum[tid.x] = min(g_minimum[tid.x], g_minimum[tid.x + stride]);
                    g_maximum[tid.x] = max(g_maximum[tid.x], g_maximum[tid.x + stride]);
                    g_count[tid.x] = merged_count;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid.x == 0u) {
        Partials[gid.x * 3u] = uint4(
            asuint(g_mean[0]), asuint(g_m2[0]), g_count[0], asuint(g_minimum[0]));
        Partials[gid.x * 3u + 1u] = uint4(
            asuint(g_maximum[0]),
            g_eligible[0],
            g_normalization_minimum[0],
            g_normalization_maximum[0]);
        Partials[gid.x * 3u + 2u] =
            uint4(g_normalization_valid_count[0], 0u, 0u, 0u);
    }
}
