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
// Exact completed Q30 values at the first and last admitted columns.
groupshared int2 EdgesQ30[V2_LIMIT_GROUP_LINES];

#define V2_LIMIT_BLOCK_CELLS 16u
#define V2_LIMIT_BLOCK_COUNT \
    ((V2_LIMIT_MAX_HORIZONTAL_DIMENSION + V2_LIMIT_LINE_CHUNKS * V2_LIMIT_BLOCK_CELLS - 1u) / \
     (V2_LIMIT_LINE_CHUNKS * V2_LIMIT_BLOCK_CELLS))

// The first checkpoint is the first value. Later checkpoints precede their blocks.
// Each lane owns its row and chunk in both arrays.
groupshared int ForwardCheckpointQ30[V2_LIMIT_BLOCK_COUNT][V2_LIMIT_LINE_CHUNKS][V2_LIMIT_GROUP_LINES];
groupshared int2 ReverseBlockQ30[V2_LIMIT_BLOCK_CELLS][V2_LIMIT_LINE_CHUNKS][V2_LIMIT_GROUP_LINES];

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

// Compute saturation_distance = max_decay / step once for the whole scan.
int V2LimitSaturatedDecayQ30(int step_q30, uint distance, uint saturation_distance,
                             int max_decay_q30) {
    return distance <= saturation_distance ? step_q30 * (int)distance : max_decay_q30;
}

float LoadVerticalShare(uint x, uint y, bool compact, uint2 columns) {
    return VerticalShare[uint2(compact ? clamp(x, columns.x, columns.y) : x, y)];
}

void V2LimitMain(uint3 group_id, uint3 group_thread_id, bool compact) {
    if (target_w == 0u || target_h == 0u ||
        target_w > V2_LIMIT_MAX_HORIZONTAL_DIMENSION) {
        return;
    }
    uint row = group_thread_id.x;
    uint lane = group_thread_id.y;
    uint y = group_id.x * V2_LIMIT_GROUP_LINES + row;
    // A partial last group keeps its idle threads until the final group barrier.
    bool active = y < target_h;
    uint2 columns = compact && DepthAnalysisContentValid() ?
        uint2(analysis_content_left, analysis_content_right - 1u) : uint2(0u, target_w - 1u);

    float max_step = v2_max_horizontal_slope / DepthAnalysisContentWidthCells();

    // Preserve the tiny diagnostic/unit-test path exactly and avoid empty chunks.
    if (target_w <= V2_LIMIT_SERIAL_MAX_LINES) {
        if (active && lane == 0u) {
            float value = LoadVerticalShare(0u, y, compact, columns);
            FinalOut[uint2(0u, y)] = V2BoundDisplayParallax(value);
            [loop]
            for (uint serial_x = 1u; serial_x < target_w; ++serial_x) {
                value = max(LoadVerticalShare(serial_x, y, compact, columns), value - max_step);
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

    uint domain_width = columns.y - columns.x + 1u;
    bool restricted = compact && domain_width > V2_LIMIT_SERIAL_MAX_LINES && domain_width < target_w;
    uint scan_width = restricted ? domain_width : target_w;
    uint scan_left = restricted ? columns.x : 0u;
    uint chunk_start = scan_left + lane * scan_width / V2_LIMIT_LINE_CHUNKS;
    uint chunk_end = scan_left + (lane + 1u) * scan_width / V2_LIMIT_LINE_CHUNKS;
    int forward_q30 = 0;
    int backward_q30 = 0;
    if (active) {
        // One forward traversal records the checkpoints and forms both chunk ends.
        // The backward value at chunk_start is the closed form
        // max_s(VerticalShare(s) - decay(s - chunk_start)). Every Q30 input lies in
        // [-container, container] and the decay saturates at twice that limit, so a saturated
        // term can never exceed VerticalShare(chunk_start); the serial recurrence gives the same
        // value.
        forward_q30 = V2LimitUpperQ30(LoadVerticalShare(chunk_start, y, compact, columns));
        ForwardCheckpointQ30[0u][lane][row] = forward_q30;
        backward_q30 = forward_q30;
        [loop]
        for (uint local_forward_x = chunk_start + 1u;
             local_forward_x < chunk_end;
             ++local_forward_x) {
            int value_q30 = V2LimitUpperQ30(LoadVerticalShare(local_forward_x, y, compact, columns));
            uint relative_x = local_forward_x - chunk_start;
            if (relative_x % V2_LIMIT_BLOCK_CELLS == 0u) {
                ForwardCheckpointQ30[relative_x / V2_LIMIT_BLOCK_CELLS][lane][row] = forward_q30;
            }
            forward_q30 = max(value_q30, forward_q30 - max_step_q30);
            int decay_q30 = V2LimitSaturatedDecayQ30(
                max_step_q30,
                relative_x,
                saturation_distance,
                max_decay_q30);
            backward_q30 = max(backward_q30, value_q30 - decay_q30);
        }
    }
    LocalEndsQ30[lane][row] = int2(forward_q30, backward_q30);
    GroupMemoryBarrierWithGroupSync();

    if (lane == 0u) {
        ChunkCarriesQ30[0u][row].x = 0;
        ChunkCarriesQ30[V2_LIMIT_LINE_CHUNKS - 1u][row].y = 0;
        forward_q30 = LocalEndsQ30[0u][row].x;
        [unroll]
        for (uint forward_chunk = 1u;
             forward_chunk < V2_LIMIT_LINE_CHUNKS;
             ++forward_chunk) {
            ChunkCarriesQ30[forward_chunk][row].x = forward_q30;
            uint forward_start = forward_chunk * scan_width / V2_LIMIT_LINE_CHUNKS;
            uint forward_end = (forward_chunk + 1u) * scan_width / V2_LIMIT_LINE_CHUNKS;
            int forward_span_q30 = V2LimitSaturatedDecayQ30(
                max_step_q30,
                forward_end - forward_start,
                saturation_distance,
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
            uint backward_start = (uint)backward_chunk * scan_width / V2_LIMIT_LINE_CHUNKS;
            uint backward_end = ((uint)backward_chunk + 1u) * scan_width / V2_LIMIT_LINE_CHUNKS;
            int backward_span_q30 = V2LimitSaturatedDecayQ30(
                max_step_q30,
                backward_end - backward_start,
                saturation_distance,
                max_decay_q30);
            backward_q30 = max(
                LocalEndsQ30[(uint)backward_chunk][row].y,
                backward_q30 - backward_span_q30);
        }
        if (restricted) {
            EdgesQ30[row] = int2(backward_q30, forward_q30);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (!active) {
        return;
    }

    // Visit the blocks right to left. Rebuild the local forward values from the checkpoint and
    // keep the backward recurrence across blocks. Each lane reads only its own block scratch.
    int2 carry = ChunkCarriesQ30[lane][row];
    bool forward_carry = lane != 0u;
    bool backward_carry = lane + 1u != V2_LIMIT_LINE_CHUNKS;
    uint block_count = (chunk_end - chunk_start + V2_LIMIT_BLOCK_CELLS - 1u) / V2_LIMIT_BLOCK_CELLS;
    // This starting value is below the smallest possible clamped sample.
    backward_q30 = -max_decay_q30;
    [loop]
    for (int block = (int)block_count - 1; block >= 0; --block) {
        uint block_start = chunk_start + (uint)block * V2_LIMIT_BLOCK_CELLS;
        uint block_end = min(block_start + V2_LIMIT_BLOCK_CELLS, chunk_end);
        forward_q30 = ForwardCheckpointQ30[(uint)block][lane][row];
        [loop]
        for (uint x = block_start; x < block_end; ++x) {
            int value_q30 = V2LimitUpperQ30(LoadVerticalShare(x, y, compact, columns));
            forward_q30 = max(value_q30, forward_q30 - max_step_q30);
            ReverseBlockQ30[x - block_start][lane][row] = int2(value_q30, forward_q30);
        }
        [loop]
        for (int scan_x = (int)block_end - 1; scan_x >= (int)block_start; --scan_x) {
            uint x = (uint)scan_x;
            int2 local = ReverseBlockQ30[x - block_start][lane][row];
            backward_q30 = max(local.x, backward_q30 - max_step_q30);
            int complete_q30 = max(local.y, backward_q30);
            if (backward_carry) {
                int decay_q30 = V2LimitSaturatedDecayQ30(
                    max_step_q30, chunk_end - x, saturation_distance, max_decay_q30);
                complete_q30 = max(complete_q30, carry.y - decay_q30);
            }
            if (forward_carry) {
                int decay_q30 = V2LimitSaturatedDecayQ30(
                    max_step_q30, x - chunk_start + 1u, saturation_distance, max_decay_q30);
                complete_q30 = max(complete_q30, carry.x - decay_q30);
            }
            // Q30-to-float conversion is monotonic, so one conversion after max gives the same bits.
            FinalOut[uint2(x, y)] = V2BoundDisplayParallax(V2LimitFromQ30(complete_q30));
        }
    }

    if (restricted) {
        // Outside the admitted interval the input is the nearest edge value. Interior
        // contributions decay away from the edge at the same rate as the majorant.
        const int2 complete_edge_q30 = EdgesQ30[row];
        if (columns.x > 0u) {
            const int edge_q30 = V2LimitUpperQ30(VerticalShare[uint2(columns.x, y)]);
            [loop]
            for (uint distance = lane + 1u; distance <= columns.x; distance += V2_LIMIT_LINE_CHUNKS) {
                int decay = V2LimitSaturatedDecayQ30(max_step_q30, distance, saturation_distance, max_decay_q30);
                FinalOut[uint2(columns.x - distance, y)] = V2BoundDisplayParallax(
                    V2LimitFromQ30(max(edge_q30, complete_edge_q30.x - decay)));
            }
        }
        if (columns.y + 1u < target_w) {
            const int edge_q30 = V2LimitUpperQ30(VerticalShare[uint2(columns.y, y)]);
            [loop]
            for (uint distance = lane + 1u; distance < target_w - columns.y; distance += V2_LIMIT_LINE_CHUNKS) {
                int decay = V2LimitSaturatedDecayQ30(max_step_q30, distance, saturation_distance, max_decay_q30);
                FinalOut[uint2(columns.y + distance, y)] = V2BoundDisplayParallax(
                    V2LimitFromQ30(max(edge_q30, complete_edge_q30.y - decay)));
            }
        }
    }
}

// Only current proposal rows use the shared horizontal core. Rows whose complete proposal
// equals immutable Base retain every Base bit rather than acquiring another Q30 rounding.
StructuredBuffer<uint> GaussianSmoothingControl : register(t2);
Texture2D<float> GaussianBase : register(t3);
groupshared uint SubtitleRowChanged[V2_LIMIT_GROUP_LINES];

bool GaussianSmoothingEnabled() {
    uint words, stride;
    GaussianSmoothingControl.GetDimensions(words, stride);
    if (words != 1u || stride != 4u) return false;
    return GaussianSmoothingControl[0u] == 1u;
}

[numthreads(V2_LIMIT_GROUP_LINES, V2_LIMIT_LINE_CHUNKS, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
    V2LimitMain(group_id, group_thread_id, false);
}

[numthreads(V2_LIMIT_GROUP_LINES, V2_LIMIT_LINE_CHUNKS, 1)]
void subtitle_main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
    // Every lane follows the shared core and its barriers. Per-row copy decisions occur only
    // after all core writes complete, so independent rows cannot invalidate group barriers.
    if (GaussianSmoothingEnabled()) {
        uint row = group_thread_id.x;
        uint y = group_id.x * V2_LIMIT_GROUP_LINES + row;
        if (group_thread_id.y == 0u) SubtitleRowChanged[row] = 0u;
        GroupMemoryBarrierWithGroupSync();
        if (y < target_h) {
            [loop]
            for (uint x = group_thread_id.y; x < target_w; x += V2_LIMIT_LINE_CHUNKS) {
                uint2 position = uint2(x, y);
                if (asuint(VerticalShare.Load(int3(position, 0))) != asuint(GaussianBase.Load(int3(position, 0))))
                    InterlockedOr(SubtitleRowChanged[row], 1u);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        V2LimitMain(group_id, group_thread_id, false);
        DeviceMemoryBarrierWithGroupSync();
        if (y < target_h && SubtitleRowChanged[row] == 0u) {
            [loop]
            for (uint copy_x = group_thread_id.y; copy_x < target_w; copy_x += V2_LIMIT_LINE_CHUNKS)
                FinalOut[uint2(copy_x, y)] = GaussianBase.Load(int3(copy_x, y, 0));
        }
    } else {
        uint y = group_id.x * V2_LIMIT_GROUP_LINES + group_thread_id.x;
        if (y < target_h) {
            [loop]
            for (uint x = group_thread_id.y; x < target_w; x += V2_LIMIT_LINE_CHUNKS) {
                uint2 position = uint2(x, y);
                FinalOut[position] = GaussianBase.Load(int3(position, 0));
            }
        }
    }
}

[numthreads(V2_LIMIT_GROUP_LINES, V2_LIMIT_LINE_CHUNKS, 1)]
void compact_main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
    V2LimitMain(group_id, group_thread_id, true);
}
