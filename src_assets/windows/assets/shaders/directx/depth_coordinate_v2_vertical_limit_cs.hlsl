// Produce conservative Q30 column-wise upper/lower envelopes and the fixed orientation-selective
// vertical share used by Host SBS V2:
//
//   Upper(y) = max_s OwnershipRefined(s) - step * abs(y - s)
//   Lower(y) = min_s OwnershipRefined(s) + step * abs(y - s)
//   Conditioned(y) = a * Upper(y) + (1 - a) * Lower(y)
//
// where step = max_vertical_shear / content_width and a is the authenticated contract constant.
// One group owns V2_LIMITER_GROUP_LINES adjacent columns, each split into V2_LIMITER_LINE_CHUNKS
// row chunks, so neighboring threads read and write neighboring texels. Q30 makes chunk transfer
// composition exact and associative; upper inputs round upward, lower inputs round downward, and
// the step rounds downward so neither envelope can exceed the authenticated shear bound because of
// a chunk-boundary float mismatch. The result is bit-identical to the serial Q30 recurrence.

Texture2D<float> OwnershipRefined : register(t0);
RWTexture2D<float> VerticalMajorant : register(u0);
RWTexture2D<float> VerticalConditioned : register(u1);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2.hlsl"

#define V2_LIMIT_SERIAL_MAX_LINES V2_LIMITER_SERIAL_MAX_LINES
#define V2_LIMIT_GROUP_LINES V2_LIMITER_GROUP_LINES
#define V2_LIMIT_LINE_CHUNKS V2_LIMITER_LINE_CHUNKS
#define V2_LIMIT_Q30_SCALE V2_LIMITER_Q_SCALE
#define V2_LIMIT_VERTICAL_STEP_Q30_NUMERATOR \
    V2_LIMITER_VERTICAL_STEP_Q_NUMERATOR
// The single-high fused runtime supports portrait fields up to 868x2072. Larger requests are
// outside the authenticated contract and leave the outputs unwritten.
#define V2_LIMIT_MAX_VERTICAL_DIMENSION (2u * V2_MODEL_CALIBRATED_MAX_DIMENSION)

// {forward upper end, forward lower end, backward upper start, backward lower start}
groupshared int4 LocalEndsQ30[V2_LIMIT_LINE_CHUNKS][V2_LIMIT_GROUP_LINES];
// {incoming forward upper, incoming forward lower, incoming backward upper, incoming backward lower}
groupshared int4 ChunkCarriesQ30[V2_LIMIT_LINE_CHUNKS][V2_LIMIT_GROUP_LINES];

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

// Equal to V2LimitDecayQ30 for every distance, with saturation_distance = max_decay / step
// computed once instead of one integer division per texel.
int V2LimitSaturatedDecayQ30(int step_q30, uint distance, uint saturation_distance,
                             int max_decay_q30) {
    return distance <= saturation_distance ? step_q30 * (int)distance : max_decay_q30;
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

[numthreads(V2_LIMIT_GROUP_LINES, V2_LIMIT_LINE_CHUNKS, 1)]
void main(
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID) {
    if (target_h == 0u || target_w == 0u ||
        target_h > V2_LIMIT_MAX_VERTICAL_DIMENSION) {
        return;
    }
    uint column = group_thread_id.x;
    uint lane = group_thread_id.y;
    uint x = group_id.x * V2_LIMIT_GROUP_LINES + column;
    // A partial last group keeps its idle threads until the final group barrier.
    bool active = x < target_w;

    float max_step = v2_max_vertical_shear / DepthAnalysisContentWidthCells();

    // Preserve the tiny diagnostic/unit-test path exactly and avoid empty chunks.
    if (target_h <= V2_LIMIT_SERIAL_MAX_LINES) {
        if (active && lane == 0u) {
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
    uint saturation_distance = (uint)(max_decay_q30 / max_step_q30);

    uint chunk_start = lane * target_h / V2_LIMIT_LINE_CHUNKS;
    uint chunk_end = (lane + 1u) * target_h / V2_LIMIT_LINE_CHUNKS;
    int forward_upper_q30 = 0;
    int forward_lower_q30 = 0;
    int backward_upper_q30 = 0;
    int backward_lower_q30 = 0;
    if (active) {
        // One forward traversal stores this chunk's local forward envelopes and forms both chunk
        // ends. The backward value at chunk_start is the closed form
        // max_s(Candidate(s) - decay(s - chunk_start)). Every Q30 input lies in
        // [-container, container] and the decay saturates at twice that limit, so a saturated
        // term can never exceed Candidate(chunk_start); the serial recurrence gives the same value.
        float candidate = OwnershipRefined[uint2(x, chunk_start)];
        forward_upper_q30 = V2LimitUpperQ30(candidate);
        forward_lower_q30 = V2LimitLowerQ30(candidate);
        backward_upper_q30 = forward_upper_q30;
        backward_lower_q30 = forward_lower_q30;
        VerticalMajorant[uint2(x, chunk_start)] = V2LimitFromQ30(forward_upper_q30);
        VerticalConditioned[uint2(x, chunk_start)] = V2LimitFromQ30(forward_lower_q30);
        [loop]
        for (uint local_forward_y = chunk_start + 1u;
             local_forward_y < chunk_end;
             ++local_forward_y) {
            candidate = OwnershipRefined[uint2(x, local_forward_y)];
            int upper_q30 = V2LimitUpperQ30(candidate);
            int lower_q30 = V2LimitLowerQ30(candidate);
            forward_upper_q30 = max(upper_q30, forward_upper_q30 - max_step_q30);
            forward_lower_q30 = min(lower_q30, forward_lower_q30 + max_step_q30);
            int decay_q30 = V2LimitSaturatedDecayQ30(
                max_step_q30,
                local_forward_y - chunk_start,
                saturation_distance,
                max_decay_q30);
            backward_upper_q30 = max(backward_upper_q30, upper_q30 - decay_q30);
            backward_lower_q30 = min(backward_lower_q30, lower_q30 + decay_q30);
            VerticalMajorant[uint2(x, local_forward_y)] = V2LimitFromQ30(forward_upper_q30);
            VerticalConditioned[uint2(x, local_forward_y)] = V2LimitFromQ30(forward_lower_q30);
        }
    }
    LocalEndsQ30[lane][column] = int4(
        forward_upper_q30,
        forward_lower_q30,
        backward_upper_q30,
        backward_lower_q30);
    GroupMemoryBarrierWithGroupSync();

    if (lane == 0u) {
        forward_upper_q30 = LocalEndsQ30[0u][column].x;
        forward_lower_q30 = LocalEndsQ30[0u][column].y;
        [unroll]
        for (uint forward_chunk = 1u;
             forward_chunk < V2_LIMIT_LINE_CHUNKS;
             ++forward_chunk) {
            ChunkCarriesQ30[forward_chunk][column].xy = int2(
                forward_upper_q30,
                forward_lower_q30);
            uint forward_start = forward_chunk * target_h / V2_LIMIT_LINE_CHUNKS;
            uint forward_end = (forward_chunk + 1u) * target_h / V2_LIMIT_LINE_CHUNKS;
            int forward_span_q30 = V2LimitDecayQ30(
                max_step_q30,
                forward_end - forward_start,
                max_decay_q30);
            forward_upper_q30 = max(
                LocalEndsQ30[forward_chunk][column].x,
                forward_upper_q30 - forward_span_q30);
            forward_lower_q30 = min(
                LocalEndsQ30[forward_chunk][column].y,
                forward_lower_q30 + forward_span_q30);
        }

        backward_upper_q30 = LocalEndsQ30[V2_LIMIT_LINE_CHUNKS - 1u][column].z;
        backward_lower_q30 = LocalEndsQ30[V2_LIMIT_LINE_CHUNKS - 1u][column].w;
        [unroll]
        for (int backward_chunk = (int)V2_LIMIT_LINE_CHUNKS - 2;
             backward_chunk >= 0;
             --backward_chunk) {
            ChunkCarriesQ30[(uint)backward_chunk][column].zw = int2(
                backward_upper_q30,
                backward_lower_q30);
            uint backward_start = (uint)backward_chunk * target_h / V2_LIMIT_LINE_CHUNKS;
            uint backward_end = ((uint)backward_chunk + 1u) * target_h / V2_LIMIT_LINE_CHUNKS;
            int backward_span_q30 = V2LimitDecayQ30(
                max_step_q30,
                backward_end - backward_start,
                max_decay_q30);
            backward_upper_q30 = max(
                LocalEndsQ30[(uint)backward_chunk][column].z,
                backward_upper_q30 - backward_span_q30);
            backward_lower_q30 = min(
                LocalEndsQ30[(uint)backward_chunk][column].w,
                backward_lower_q30 + backward_span_q30);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (!active) {
        return;
    }

    // One backward traversal completes each texel. With the incoming carries,
    //   Forward(y) = max(LocalForward(y), carry - decay(y - chunk_start + 1))
    //   Backward(y) = max(LocalBackward(y), carry - decay(chunk_end - y))
    // (min/plus for the lower envelope). LocalForward was stored as float; Q30-to-float
    // conversion is monotonic, so max/min of converted values equals converting the Q30 max/min.
    // Each thread rereads only texels it wrote itself.
    int4 carry = ChunkCarriesQ30[lane][column];
    bool forward_carry = lane != 0u;
    bool backward_carry = lane + 1u != V2_LIMIT_LINE_CHUNKS;
    [loop]
    for (int scan_y = (int)chunk_end - 1; scan_y >= (int)chunk_start; --scan_y) {
        uint write_y = (uint)scan_y;
        float candidate = OwnershipRefined[uint2(x, write_y)];
        int upper_q30 = V2LimitUpperQ30(candidate);
        int lower_q30 = V2LimitLowerQ30(candidate);
        if (write_y + 1u == chunk_end) {
            backward_upper_q30 = upper_q30;
            backward_lower_q30 = lower_q30;
        } else {
            backward_upper_q30 = max(upper_q30, backward_upper_q30 - max_step_q30);
            backward_lower_q30 = min(lower_q30, backward_lower_q30 + max_step_q30);
        }
        int complete_upper_q30 = backward_upper_q30;
        int complete_lower_q30 = backward_lower_q30;
        if (backward_carry) {
            int decay_q30 = V2LimitSaturatedDecayQ30(
                max_step_q30, chunk_end - write_y, saturation_distance, max_decay_q30);
            complete_upper_q30 = max(complete_upper_q30, carry.z - decay_q30);
            complete_lower_q30 = min(complete_lower_q30, carry.w + decay_q30);
        }
        if (forward_carry) {
            int decay_q30 = V2LimitSaturatedDecayQ30(
                max_step_q30, write_y - chunk_start + 1u, saturation_distance, max_decay_q30);
            complete_upper_q30 = max(complete_upper_q30, carry.x - decay_q30);
            complete_lower_q30 = min(complete_lower_q30, carry.y + decay_q30);
        }
        float final_upper = max(
            VerticalMajorant[uint2(x, write_y)],
            V2LimitFromQ30(complete_upper_q30));
        float final_lower = min(
            VerticalConditioned[uint2(x, write_y)],
            V2LimitFromQ30(complete_lower_q30));
        VerticalMajorant[uint2(x, write_y)] = final_upper;
        VerticalConditioned[uint2(x, write_y)] = clamp(
            share_vertical_envelopes(final_upper, final_lower),
            final_lower,
            final_upper);
    }
}
