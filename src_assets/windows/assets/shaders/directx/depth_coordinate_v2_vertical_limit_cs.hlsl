// Produce conservative Q30 column-wise upper/lower envelopes and the fixed orientation-selective
// vertical share used by Host SBS V2:
//
//   Upper(y) = max_s OwnershipRefined(s) - step * abs(y - s)
//   Lower(y) = min_s OwnershipRefined(s) + step * abs(y - s)
//   Conditioned(y) = a * Upper(y) + (1 - a) * Lower(y)
//
// where step = max_vertical_shear / content_width and a is the authenticated contract constant.
// One 32-thread group owns a column. Q30 makes chunk transfer composition exact and associative;
// upper inputs round upward, lower inputs round downward, and the step rounds downward so neither
// envelope can exceed the authenticated shear bound because of a chunk-boundary float mismatch.

Texture2D<float> OwnershipRefined : register(t0);
RWTexture2D<float> VerticalMajorant : register(u0);
RWTexture2D<float> VerticalConditioned : register(u1);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2.hlsl"

#define V2_LIMIT_THREADS V2_LIMITER_GROUP_THREADS
#define V2_LIMIT_Q30_SCALE V2_LIMITER_Q_SCALE
#define V2_LIMIT_VERTICAL_STEP_Q30_NUMERATOR \
    V2_LIMITER_VERTICAL_STEP_Q_NUMERATOR

groupshared int LineUpperQ30[V2_MODEL_CALIBRATED_MAX_DIMENSION];
groupshared int LineLowerQ30[V2_MODEL_CALIBRATED_MAX_DIMENSION];
groupshared int ForwardUpperQ30[V2_MODEL_CALIBRATED_MAX_DIMENSION];
groupshared int ForwardLowerQ30[V2_MODEL_CALIBRATED_MAX_DIMENSION];
// {forward upper end, forward lower end, backward upper end, backward lower end}
groupshared int4 LocalEndsQ30[V2_LIMIT_THREADS];
// {incoming forward upper, incoming forward lower, incoming backward upper, incoming backward lower}
groupshared int4 ChunkCarriesQ30[V2_LIMIT_THREADS];

int V2LimitUpperQ30(float value) {
    value = V2Finite(value) ?
        clamp(value, -v2_direct_container_limit, v2_direct_container_limit) : 0.0f;
    return (int)ceil(value * V2_LIMIT_Q30_SCALE);
}

int V2LimitLowerQ30(float value) {
    value = V2Finite(value) ?
        clamp(value, -v2_direct_container_limit, v2_direct_container_limit) : 0.0f;
    return (int)floor(value * V2_LIMIT_Q30_SCALE);
}

float V2LimitFromQ30(int value) {
    return (float)value / V2_LIMIT_Q30_SCALE;
}

int V2LimitStepQ30(uint content_width, int max_decay_q30) {
    uint exact_step_q30 = V2_LIMIT_VERTICAL_STEP_Q30_NUMERATOR / content_width;
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

float share_vertical_envelopes(float upper, float lower) {
    // Keep the persisted float32 evaluation order and forbid contraction/reassociation.
    precise float majorant_share = v2_vertical_majorant_share;
    precise float minorant_share = 1.0f - majorant_share;
    precise float majorant_term = majorant_share * upper;
    precise float minorant_term = minorant_share * lower;
    precise float conditioned = majorant_term + minorant_term;
    return conditioned;
}

[numthreads(V2_LIMIT_THREADS, 1, 1)]
void main(
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID) {
    uint x = group_id.x;
    if (x >= target_w || target_h == 0u || target_w == 0u ||
        target_h > V2_MODEL_CALIBRATED_MAX_DIMENSION) {
        return;
    }

    float max_step = v2_max_vertical_shear / DepthAnalysisContentWidthCells();
    uint lane = group_thread_id.x;

    // Preserve the tiny diagnostic/unit-test path exactly and avoid empty chunks.
    if (target_h <= V2_LIMIT_THREADS) {
        if (lane == 0u) {
            float candidate = OwnershipRefined[uint2(x, 0u)];
            float upper = candidate;
            float lower = candidate;
            VerticalMajorant[uint2(x, 0u)] = upper;
            VerticalConditioned[uint2(x, 0u)] = lower;
            [loop]
            for (uint serial_y = 1u; serial_y < target_h; ++serial_y) {
                candidate = OwnershipRefined[uint2(x, serial_y)];
                upper = max(candidate, upper - max_step);
                lower = min(candidate, lower + max_step);
                VerticalMajorant[uint2(x, serial_y)] = upper;
                VerticalConditioned[uint2(x, serial_y)] = lower;
            }
            DeviceMemoryBarrier();
            const uint last_y = target_h - 1u;
            upper = VerticalMajorant[uint2(x, last_y)];
            lower = VerticalConditioned[uint2(x, last_y)];
            VerticalConditioned[uint2(x, last_y)] = share_vertical_envelopes(upper, lower);
            [loop]
            for (int serial_back_y = (int)target_h - 2;
                 serial_back_y >= 0;
                 --serial_back_y) {
                const uint2 position = uint2(x, (uint)serial_back_y);
                upper = max(VerticalMajorant[position], upper - max_step);
                lower = min(VerticalConditioned[position], lower + max_step);
                VerticalMajorant[position] = upper;
                VerticalConditioned[position] = share_vertical_envelopes(upper, lower);
            }
        }
        return;
    }

    int max_decay_q30 = 2 * V2_LIMITER_CONTAINER_Q_LIMIT;
    uint content_width = analysis_content_right > analysis_content_left ?
        analysis_content_right - analysis_content_left : 1u;
    int max_step_q30 = V2LimitStepQ30(content_width, max_decay_q30);
    [loop]
    for (uint load_y = lane; load_y < target_h; load_y += V2_LIMIT_THREADS) {
        float candidate = OwnershipRefined[uint2(x, load_y)];
        LineUpperQ30[load_y] = V2LimitUpperQ30(candidate);
        LineLowerQ30[load_y] = V2LimitLowerQ30(candidate);
    }
    GroupMemoryBarrierWithGroupSync();

    uint chunk_start = lane * target_h / V2_LIMIT_THREADS;
    uint chunk_end = (lane + 1u) * target_h / V2_LIMIT_THREADS;
    int forward_upper_q30 = LineUpperQ30[chunk_start];
    int forward_lower_q30 = LineLowerQ30[chunk_start];
    [loop]
    for (uint local_forward_y = chunk_start + 1u;
         local_forward_y < chunk_end;
         ++local_forward_y) {
        forward_upper_q30 = max(
            LineUpperQ30[local_forward_y],
            forward_upper_q30 - max_step_q30);
        forward_lower_q30 = min(
            LineLowerQ30[local_forward_y],
            forward_lower_q30 + max_step_q30);
    }
    int backward_upper_q30 = LineUpperQ30[chunk_end - 1u];
    int backward_lower_q30 = LineLowerQ30[chunk_end - 1u];
    [loop]
    for (int local_backward_y = (int)chunk_end - 2;
         local_backward_y >= (int)chunk_start;
         --local_backward_y) {
        backward_upper_q30 = max(
            LineUpperQ30[(uint)local_backward_y],
            backward_upper_q30 - max_step_q30);
        backward_lower_q30 = min(
            LineLowerQ30[(uint)local_backward_y],
            backward_lower_q30 + max_step_q30);
    }
    LocalEndsQ30[lane] = int4(
        forward_upper_q30,
        forward_lower_q30,
        backward_upper_q30,
        backward_lower_q30);
    GroupMemoryBarrierWithGroupSync();

    if (lane == 0u) {
        [unroll]
        for (uint clear_chunk = 0u; clear_chunk < V2_LIMIT_THREADS; ++clear_chunk) {
            ChunkCarriesQ30[clear_chunk] = int4(0, 0, 0, 0);
        }

        forward_upper_q30 = LocalEndsQ30[0u].x;
        forward_lower_q30 = LocalEndsQ30[0u].y;
        [unroll]
        for (uint forward_chunk = 1u;
             forward_chunk < V2_LIMIT_THREADS;
             ++forward_chunk) {
            ChunkCarriesQ30[forward_chunk].xy = int2(
                forward_upper_q30,
                forward_lower_q30);
            uint forward_start = forward_chunk * target_h / V2_LIMIT_THREADS;
            uint forward_end = (forward_chunk + 1u) * target_h / V2_LIMIT_THREADS;
            int forward_span_q30 = V2LimitDecayQ30(
                max_step_q30,
                forward_end - forward_start,
                max_decay_q30);
            forward_upper_q30 = max(
                LocalEndsQ30[forward_chunk].x,
                forward_upper_q30 - forward_span_q30);
            forward_lower_q30 = min(
                LocalEndsQ30[forward_chunk].y,
                forward_lower_q30 + forward_span_q30);
        }

        backward_upper_q30 = LocalEndsQ30[V2_LIMIT_THREADS - 1u].z;
        backward_lower_q30 = LocalEndsQ30[V2_LIMIT_THREADS - 1u].w;
        [unroll]
        for (int backward_chunk = (int)V2_LIMIT_THREADS - 2;
             backward_chunk >= 0;
             --backward_chunk) {
            ChunkCarriesQ30[(uint)backward_chunk].zw = int2(
                backward_upper_q30,
                backward_lower_q30);
            uint backward_start = (uint)backward_chunk * target_h / V2_LIMIT_THREADS;
            uint backward_end = ((uint)backward_chunk + 1u) * target_h / V2_LIMIT_THREADS;
            int backward_span_q30 = V2LimitDecayQ30(
                max_step_q30,
                backward_end - backward_start,
                max_decay_q30);
            backward_upper_q30 = max(
                LocalEndsQ30[(uint)backward_chunk].z,
                backward_upper_q30 - backward_span_q30);
            backward_lower_q30 = min(
                LocalEndsQ30[(uint)backward_chunk].w,
                backward_lower_q30 + backward_span_q30);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    forward_upper_q30 = LineUpperQ30[chunk_start];
    forward_lower_q30 = LineLowerQ30[chunk_start];
    if (lane != 0u) {
        forward_upper_q30 = max(
            forward_upper_q30,
            ChunkCarriesQ30[lane].x - max_step_q30);
        forward_lower_q30 = min(
            forward_lower_q30,
            ChunkCarriesQ30[lane].y + max_step_q30);
    }
    ForwardUpperQ30[chunk_start] = forward_upper_q30;
    ForwardLowerQ30[chunk_start] = forward_lower_q30;
    [loop]
    for (uint replay_forward_y = chunk_start + 1u;
         replay_forward_y < chunk_end;
         ++replay_forward_y) {
        forward_upper_q30 = max(
            LineUpperQ30[replay_forward_y],
            forward_upper_q30 - max_step_q30);
        forward_lower_q30 = min(
            LineLowerQ30[replay_forward_y],
            forward_lower_q30 + max_step_q30);
        ForwardUpperQ30[replay_forward_y] = forward_upper_q30;
        ForwardLowerQ30[replay_forward_y] = forward_lower_q30;
    }
    GroupMemoryBarrierWithGroupSync();

    backward_upper_q30 = LineUpperQ30[chunk_end - 1u];
    backward_lower_q30 = LineLowerQ30[chunk_end - 1u];
    if (lane + 1u != V2_LIMIT_THREADS) {
        backward_upper_q30 = max(
            backward_upper_q30,
            ChunkCarriesQ30[lane].z - max_step_q30);
        backward_lower_q30 = min(
            backward_lower_q30,
            ChunkCarriesQ30[lane].w + max_step_q30);
    }
    uint write_y = chunk_end - 1u;
    int final_upper_q30 = max(ForwardUpperQ30[write_y], backward_upper_q30);
    int final_lower_q30 = min(ForwardLowerQ30[write_y], backward_lower_q30);
    float final_upper = V2LimitFromQ30(final_upper_q30);
    float final_lower = V2LimitFromQ30(final_lower_q30);
    VerticalMajorant[uint2(x, write_y)] = final_upper;
    VerticalConditioned[uint2(x, write_y)] = clamp(
        share_vertical_envelopes(final_upper, final_lower),
        final_lower,
        final_upper);
    [loop]
    for (int scan_y = (int)chunk_end - 2; scan_y >= (int)chunk_start; --scan_y) {
        write_y = (uint)scan_y;
        backward_upper_q30 = max(
            LineUpperQ30[write_y],
            backward_upper_q30 - max_step_q30);
        backward_lower_q30 = min(
            LineLowerQ30[write_y],
            backward_lower_q30 + max_step_q30);
        final_upper_q30 = max(ForwardUpperQ30[write_y], backward_upper_q30);
        final_lower_q30 = min(ForwardLowerQ30[write_y], backward_lower_q30);
        final_upper = V2LimitFromQ30(final_upper_q30);
        final_lower = V2LimitFromQ30(final_lower_q30);
        VerticalMajorant[uint2(x, write_y)] = final_upper;
        VerticalConditioned[uint2(x, write_y)] = clamp(
            share_vertical_envelopes(final_upper, final_lower),
            final_lower,
            final_upper);
    }
}
