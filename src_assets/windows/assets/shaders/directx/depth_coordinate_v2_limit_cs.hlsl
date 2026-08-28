// Conservative row-wise Lipschitz majorant of the orientation-selective vertical share. For each
// row this computes the Q30 approximation of
//
//   q(x) = max_s VerticalShare(s) - step * abs(x - s),
//   step = max_horizontal_slope / content_width.
//
// The vertical share may raise or lower the immutable candidate, but it already satisfies the
// vertical shear bound. This pure horizontal majorant preserves that bound, enforces the
// contractive inverse-warp slope, and avoids the lateral lowering introduced by a horizontal
// minorant component. One 32-thread group owns a row. Q30 makes chunk transfer composition exact
// and associative; flooring the step keeps the resulting envelope conservatively inside the
// authenticated slope bound.

Texture2D<float> VerticalShare : register(t0);
RWTexture2D<float> FinalOut : register(u0);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2.hlsl"

#define V2_LIMIT_THREADS V2_LIMITER_GROUP_THREADS
#define V2_LIMIT_Q30_SCALE V2_LIMITER_Q_SCALE
#define V2_LIMIT_HORIZONTAL_STEP_Q30_NUMERATOR \
    V2_LIMITER_HORIZONTAL_STEP_Q_NUMERATOR
// Initial convex-2x live authority is landscape-only.  Its largest row is exactly twice the
// calibrated coarse maximum (2072), while its 868-cell columns continue to fit the unchanged
// vertical limiter.  These two int arrays consume about 16 KiB at the expanded bound.
#define V2_LIMIT_MAX_HORIZONTAL_DIMENSION (2u * V2_MODEL_CALIBRATED_MAX_DIMENSION)

groupshared int LineCandidateQ30[V2_LIMIT_MAX_HORIZONTAL_DIMENSION];
groupshared int ForwardMajorantQ30[V2_LIMIT_MAX_HORIZONTAL_DIMENSION];
// {forward end, backward end}
groupshared int2 LocalEndsQ30[V2_LIMIT_THREADS];
// {incoming forward value, incoming backward value}
groupshared int2 ChunkCarriesQ30[V2_LIMIT_THREADS];

int V2LimitUpperQ30(float value) {
    value = V2Finite(value) ?
        clamp(value, -v2_direct_container_limit, v2_direct_container_limit) : 0.0f;
    return (int)ceil(value * V2_LIMIT_Q30_SCALE);
}

float V2LimitFromQ30(int value) {
    return (float)value / V2_LIMIT_Q30_SCALE;
}

int V2LimitStepQ30(uint content_width, int max_decay_q30) {
    uint exact_step_q30 = V2_LIMIT_HORIZONTAL_STEP_Q30_NUMERATOR / content_width;
    uint bounded_step_q30 = min(exact_step_q30, (uint)max_decay_q30);
    return max(1, (int)bounded_step_q30);
}

int V2LimitDecayQ30(int step_q30, uint distance, int max_decay_q30) {
    if (distance == 0u) {
        return 0;
    }
    int signed_distance = (int)distance;
    return step_q30 > max_decay_q30 / signed_distance ?
        max_decay_q30 : step_q30 * signed_distance;
}

[numthreads(V2_LIMIT_THREADS, 1, 1)]
void main(
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID) {
    uint y = group_id.x;
    if (y >= target_h || target_w == 0u || target_h == 0u ||
        target_w > V2_LIMIT_MAX_HORIZONTAL_DIMENSION) {
        return;
    }

    float max_step = v2_max_horizontal_slope / DepthAnalysisContentWidthCells();
    uint lane = group_thread_id.x;

    // Preserve the tiny diagnostic/unit-test path exactly and avoid empty chunks.
    if (target_w <= V2_LIMIT_THREADS) {
        if (lane == 0u) {
            float value = VerticalShare[uint2(0u, y)];
            FinalOut[uint2(0u, y)] = value;
            [loop]
            for (uint serial_x = 1u; serial_x < target_w; ++serial_x) {
                value = max(VerticalShare[uint2(serial_x, y)], value - max_step);
                FinalOut[uint2(serial_x, y)] = value;
            }
            DeviceMemoryBarrier();
            value = FinalOut[uint2(target_w - 1u, y)];
            [loop]
            for (int serial_back_x = (int)target_w - 2;
                 serial_back_x >= 0;
                 --serial_back_x) {
                const uint2 position = uint2((uint)serial_back_x, y);
                value = max(FinalOut[position], value - max_step);
                FinalOut[position] = value;
            }
        }
        return;
    }

    int max_decay_q30 = 2 * V2_LIMITER_CONTAINER_Q_LIMIT;
    uint content_width = analysis_content_right > analysis_content_left ?
        analysis_content_right - analysis_content_left : 1u;
    int max_step_q30 = V2LimitStepQ30(content_width, max_decay_q30);
    [loop]
    for (uint load_x = lane; load_x < target_w; load_x += V2_LIMIT_THREADS) {
        LineCandidateQ30[load_x] = V2LimitUpperQ30(VerticalShare[uint2(load_x, y)]);
    }
    GroupMemoryBarrierWithGroupSync();

    uint chunk_start = lane * target_w / V2_LIMIT_THREADS;
    uint chunk_end = (lane + 1u) * target_w / V2_LIMIT_THREADS;
    int forward_q30 = LineCandidateQ30[chunk_start];
    [loop]
    for (uint local_forward_x = chunk_start + 1u;
         local_forward_x < chunk_end;
         ++local_forward_x) {
        forward_q30 = max(
            LineCandidateQ30[local_forward_x],
            forward_q30 - max_step_q30);
    }
    int backward_q30 = LineCandidateQ30[chunk_end - 1u];
    [loop]
    for (int local_backward_x = (int)chunk_end - 2;
         local_backward_x >= (int)chunk_start;
         --local_backward_x) {
        backward_q30 = max(
            LineCandidateQ30[(uint)local_backward_x],
            backward_q30 - max_step_q30);
    }
    LocalEndsQ30[lane] = int2(forward_q30, backward_q30);
    GroupMemoryBarrierWithGroupSync();

    if (lane == 0u) {
        forward_q30 = LocalEndsQ30[0u].x;
        [unroll]
        for (uint forward_chunk = 1u;
             forward_chunk < V2_LIMIT_THREADS;
             ++forward_chunk) {
            ChunkCarriesQ30[forward_chunk].x = forward_q30;
            uint forward_start = forward_chunk * target_w / V2_LIMIT_THREADS;
            uint forward_end = (forward_chunk + 1u) * target_w / V2_LIMIT_THREADS;
            int forward_span_q30 = V2LimitDecayQ30(
                max_step_q30,
                forward_end - forward_start,
                max_decay_q30);
            forward_q30 = max(
                LocalEndsQ30[forward_chunk].x,
                forward_q30 - forward_span_q30);
        }

        backward_q30 = LocalEndsQ30[V2_LIMIT_THREADS - 1u].y;
        [unroll]
        for (int backward_chunk = (int)V2_LIMIT_THREADS - 2;
             backward_chunk >= 0;
             --backward_chunk) {
            ChunkCarriesQ30[(uint)backward_chunk].y = backward_q30;
            uint backward_start = (uint)backward_chunk * target_w / V2_LIMIT_THREADS;
            uint backward_end = ((uint)backward_chunk + 1u) * target_w / V2_LIMIT_THREADS;
            int backward_span_q30 = V2LimitDecayQ30(
                max_step_q30,
                backward_end - backward_start,
                max_decay_q30);
            backward_q30 = max(
                LocalEndsQ30[(uint)backward_chunk].y,
                backward_q30 - backward_span_q30);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    forward_q30 = LineCandidateQ30[chunk_start];
    if (lane != 0u) {
        forward_q30 = max(
            forward_q30,
            ChunkCarriesQ30[lane].x - max_step_q30);
    }
    ForwardMajorantQ30[chunk_start] = forward_q30;
    [loop]
    for (uint replay_forward_x = chunk_start + 1u;
         replay_forward_x < chunk_end;
         ++replay_forward_x) {
        forward_q30 = max(
            LineCandidateQ30[replay_forward_x],
            forward_q30 - max_step_q30);
        ForwardMajorantQ30[replay_forward_x] = forward_q30;
    }
    GroupMemoryBarrierWithGroupSync();

    backward_q30 = LineCandidateQ30[chunk_end - 1u];
    if (lane + 1u != V2_LIMIT_THREADS) {
        backward_q30 = max(
            backward_q30,
            ChunkCarriesQ30[lane].y - max_step_q30);
    }
    uint write_x = chunk_end - 1u;
    int final_q30 = max(ForwardMajorantQ30[write_x], backward_q30);
    FinalOut[uint2(write_x, y)] = V2LimitFromQ30(final_q30);
    [loop]
    for (int scan_x = (int)chunk_end - 2; scan_x >= (int)chunk_start; --scan_x) {
        write_x = (uint)scan_x;
        backward_q30 = max(
            LineCandidateQ30[write_x],
            backward_q30 - max_step_q30);
        final_q30 = max(ForwardMajorantQ30[write_x], backward_q30);
        FinalOut[uint2(write_x, y)] = V2LimitFromQ30(final_q30);
    }
}
