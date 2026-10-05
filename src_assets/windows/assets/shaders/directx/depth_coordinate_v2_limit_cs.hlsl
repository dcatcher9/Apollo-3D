// Conservative row-wise Lipschitz majorant of the orientation-selective vertical share. For each
// row this computes the Q30 approximation of
//
//   q(x) = max_s VerticalShare(s) - step * abs(x - s),
//   step = max_horizontal_slope / content_width.
//
// The vertical share may raise or lower the immutable candidate, but it already satisfies the
// vertical shear bound. This pure horizontal majorant preserves that bound, enforces the
// contractive inverse-warp slope, and avoids the lateral lowering introduced by a horizontal
// minorant component. One group owns V2_LIMITER_GROUP_LINES adjacent rows, each split into
// V2_LIMITER_LINE_CHUNKS column chunks. Q30 makes chunk transfer composition exact and
// associative; flooring the step keeps the resulting envelope conservatively inside the
// authenticated slope bound. The result is bit-identical to the serial Q30 recurrence.

Texture2D<float> VerticalShare : register(t0);
RWTexture2D<float> FinalOut : register(u0);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2.hlsl"

#define V2_LIMIT_SERIAL_MAX_LINES V2_LIMITER_SERIAL_MAX_LINES
#define V2_LIMIT_GROUP_LINES V2_LIMITER_GROUP_LINES
#define V2_LIMIT_LINE_CHUNKS V2_LIMITER_LINE_CHUNKS
#define V2_LIMIT_Q30_SCALE V2_LIMITER_Q_SCALE
#define V2_LIMIT_HORIZONTAL_STEP_Q30_NUMERATOR \
    V2_LIMITER_HORIZONTAL_STEP_Q_NUMERATOR
// Every authenticated single-high orientation is exactly 2x a calibrated DAV2 profile. Its
// largest row is therefore twice the generated calibrated maximum (2072). Larger requests are
// outside the authenticated contract and leave the output unwritten.
#define V2_LIMIT_MAX_HORIZONTAL_DIMENSION (2u * V2_MODEL_CALIBRATED_MAX_DIMENSION)

// {forward end, backward start}
groupshared int2 LocalEndsQ30[V2_LIMIT_LINE_CHUNKS][V2_LIMIT_GROUP_LINES];
// {incoming forward value, incoming backward value}
groupshared int2 ChunkCarriesQ30[V2_LIMIT_LINE_CHUNKS][V2_LIMIT_GROUP_LINES];

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

// Equal to V2LimitDecayQ30 for every distance, with saturation_distance = max_decay / step
// computed once instead of one integer division per texel.
int V2LimitSaturatedDecayQ30(int step_q30, uint distance, uint saturation_distance,
                             int max_decay_q30) {
    return distance <= saturation_distance ? step_q30 * (int)distance : max_decay_q30;
}

[numthreads(V2_LIMIT_GROUP_LINES, V2_LIMIT_LINE_CHUNKS, 1)]
void main(
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID) {
    if (target_w == 0u || target_h == 0u ||
        target_w > V2_LIMIT_MAX_HORIZONTAL_DIMENSION) {
        return;
    }
    uint row = group_thread_id.x;
    uint lane = group_thread_id.y;
    uint y = group_id.x * V2_LIMIT_GROUP_LINES + row;
    // A partial last group keeps its idle threads until the final group barrier.
    bool active = y < target_h;

    float max_step = v2_max_horizontal_slope / DepthAnalysisContentWidthCells();

    // Preserve the tiny diagnostic/unit-test path exactly and avoid empty chunks.
    if (target_w <= V2_LIMIT_SERIAL_MAX_LINES) {
        if (active && lane == 0u) {
            float value = VerticalShare[uint2(0u, y)];
            FinalOut[uint2(0u, y)] = V2BoundDisplayParallax(value);
            [loop]
            for (uint serial_x = 1u; serial_x < target_w; ++serial_x) {
                value = max(VerticalShare[uint2(serial_x, y)], value - max_step);
                FinalOut[uint2(serial_x, y)] = V2BoundDisplayParallax(value);
            }
            DeviceMemoryBarrier();
            value = FinalOut[uint2(target_w - 1u, y)];
            [loop]
            for (int serial_back_x = (int)target_w - 2;
                 serial_back_x >= 0;
                 --serial_back_x) {
                const uint2 position = uint2((uint)serial_back_x, y);
                value = max(FinalOut[position], value - max_step);
                FinalOut[position] = V2BoundDisplayParallax(value);
            }
        }
        return;
    }

    int max_decay_q30 = 2 * V2_LIMITER_CONTAINER_Q_LIMIT;
    uint content_width = analysis_content_right > analysis_content_left ?
        analysis_content_right - analysis_content_left : 1u;
    int max_step_q30 = V2LimitStepQ30(content_width, max_decay_q30);
    uint saturation_distance = (uint)(max_decay_q30 / max_step_q30);

    uint chunk_start = lane * target_w / V2_LIMIT_LINE_CHUNKS;
    uint chunk_end = (lane + 1u) * target_w / V2_LIMIT_LINE_CHUNKS;
    int forward_q30 = 0;
    int backward_q30 = 0;
    if (active) {
        // One forward traversal stores this chunk's local forward majorant and forms both chunk
        // ends. The backward value at chunk_start is the closed form
        // max_s(VerticalShare(s) - decay(s - chunk_start)). Every Q30 input lies in
        // [-container, container] and the decay saturates at twice that limit, so a saturated
        // term can never exceed VerticalShare(chunk_start); the serial recurrence gives the same
        // value.
        forward_q30 = V2LimitUpperQ30(VerticalShare[uint2(chunk_start, y)]);
        backward_q30 = forward_q30;
        FinalOut[uint2(chunk_start, y)] = V2LimitFromQ30(forward_q30);
        [loop]
        for (uint local_forward_x = chunk_start + 1u;
             local_forward_x < chunk_end;
             ++local_forward_x) {
            int value_q30 = V2LimitUpperQ30(VerticalShare[uint2(local_forward_x, y)]);
            forward_q30 = max(value_q30, forward_q30 - max_step_q30);
            int decay_q30 = V2LimitSaturatedDecayQ30(
                max_step_q30,
                local_forward_x - chunk_start,
                saturation_distance,
                max_decay_q30);
            backward_q30 = max(backward_q30, value_q30 - decay_q30);
            FinalOut[uint2(local_forward_x, y)] = V2LimitFromQ30(forward_q30);
        }
    }
    LocalEndsQ30[lane][row] = int2(forward_q30, backward_q30);
    GroupMemoryBarrierWithGroupSync();

    if (lane == 0u) {
        forward_q30 = LocalEndsQ30[0u][row].x;
        [unroll]
        for (uint forward_chunk = 1u;
             forward_chunk < V2_LIMIT_LINE_CHUNKS;
             ++forward_chunk) {
            ChunkCarriesQ30[forward_chunk][row].x = forward_q30;
            uint forward_start = forward_chunk * target_w / V2_LIMIT_LINE_CHUNKS;
            uint forward_end = (forward_chunk + 1u) * target_w / V2_LIMIT_LINE_CHUNKS;
            int forward_span_q30 = V2LimitDecayQ30(
                max_step_q30,
                forward_end - forward_start,
                max_decay_q30);
            forward_q30 = max(
                LocalEndsQ30[forward_chunk][row].x,
                forward_q30 - forward_span_q30);
        }

        backward_q30 = LocalEndsQ30[V2_LIMIT_LINE_CHUNKS - 1u][row].y;
        [unroll]
        for (int backward_chunk = (int)V2_LIMIT_LINE_CHUNKS - 2;
             backward_chunk >= 0;
             --backward_chunk) {
            ChunkCarriesQ30[(uint)backward_chunk][row].y = backward_q30;
            uint backward_start = (uint)backward_chunk * target_w / V2_LIMIT_LINE_CHUNKS;
            uint backward_end = ((uint)backward_chunk + 1u) * target_w / V2_LIMIT_LINE_CHUNKS;
            int backward_span_q30 = V2LimitDecayQ30(
                max_step_q30,
                backward_end - backward_start,
                max_decay_q30);
            backward_q30 = max(
                LocalEndsQ30[(uint)backward_chunk][row].y,
                backward_q30 - backward_span_q30);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (!active) {
        return;
    }

    // One backward traversal completes each texel from the stored local forward majorant, the
    // local backward recurrence and both incoming carries. Q30-to-float conversion is monotonic,
    // so max of converted values equals converting the Q30 max. Each thread rereads only texels
    // it wrote itself.
    int2 carry = ChunkCarriesQ30[lane][row];
    bool forward_carry = lane != 0u;
    bool backward_carry = lane + 1u != V2_LIMIT_LINE_CHUNKS;
    [loop]
    for (int scan_x = (int)chunk_end - 1; scan_x >= (int)chunk_start; --scan_x) {
        uint write_x = (uint)scan_x;
        int value_q30 = V2LimitUpperQ30(VerticalShare[uint2(write_x, y)]);
        backward_q30 = write_x + 1u == chunk_end ?
            value_q30 : max(value_q30, backward_q30 - max_step_q30);
        int complete_q30 = backward_q30;
        if (backward_carry) {
            int decay_q30 = V2LimitSaturatedDecayQ30(
                max_step_q30, chunk_end - write_x, saturation_distance, max_decay_q30);
            complete_q30 = max(complete_q30, carry.y - decay_q30);
        }
        if (forward_carry) {
            int decay_q30 = V2LimitSaturatedDecayQ30(
                max_step_q30, write_x - chunk_start + 1u, saturation_distance, max_decay_q30);
            complete_q30 = max(complete_q30, carry.x - decay_q30);
        }
        FinalOut[uint2(write_x, y)] = V2BoundDisplayParallax(max(
            FinalOut[uint2(write_x, y)],
            V2LimitFromQ30(complete_q30)));
    }
}
