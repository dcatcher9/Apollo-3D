// Eye-specific horizontal limiter. Always recorded after the production horizontal majorant.
//
// The production horizontal majorant (depth_coordinate_v2_limit_cs.hlsl) uses one symmetric slope
// for both eyes. The geometry at a foreground edge is not symmetric. The renderer solves
//
//   out(x) = x - eye_sign * parallax(x)
//
// so for each eye the background on one side of a nearer object is covered by it, and the
// background on the other side has to be stretched over the uncovered region. This pass builds one
// field per eye from the same vertical share:
//
//   covered side  steep slope `eye_occlusion_slope`
//   hole side     `eye_hole_slope` (the production slope keeps today's stretch).
//
// Before either ramp starts, the foreground value is carried unchanged over a few cells
// (`eye_extension_cells`: low 16 bits on the covered side, high 16 bits on the hole side). A depth
// edge from a monocular model usually lies a few cells inside the object it belongs to, most of all
// at tips and soft fringes. A ramp that starts there squeezes the object's own pixels in one eye and
// stretches them in the other. With the extension the object is drawn rigid and the ramp lies on
// the background next to it.
//
//   left  eye (eye_sign -1):  max(forward steep, backward gentle)
//   right eye (eye_sign +1):  max(forward gentle, backward steep)
//
// Both slopes stay below one, so each eye map keeps a unique contractive inverse, and a max-plus
// dilation along rows preserves the vertical shear bound of its input. The output is one R32F
// texture of twice the field height: rows [0, H) hold the left-eye field, rows [H, 2H) the
// right-eye field. The production field is not touched.
//
// One group owns V2_LIMITER_GROUP_LINES rows, each split into V2_EYE_LINE_CHUNKS chunks, one thread
// per chunk. A max-plus recurrence composes across chunks through one value per sweep: the true
// envelope at a chunk's border is the larger of that chunk's own border value and the previous
// border value decayed over the chunk's length.
//
// Each thread first sweeps its chunk as if it stood alone and writes every texel once: the forward
// sweep writes nothing and keeps one checkpoint per block of V2_EYE_BLOCK_CELLS cells; the backward
// sweep takes the blocks in reverse order, rebuilds a block's forward values from its checkpoint
// and writes the larger of the two directions. The pass's time depends on writing each texel once
// without reading it back, and on short serial chains: on a 1540x868 field the whole limiter stage
// (the production limiters take 0.07 ms of it) measured 0.67 ms with one serial scan per row and a
// read-back, 0.32 ms with eight chunks and a read-back, 0.28 ms with eight chunks written once,
// and 0.10 ms with these 32 chunks of 16-cell blocks (2026-11-21, RTX 5080).
// The thread then publishes its two border values, the group resolves the borders, and each thread
// raises its cells to the envelope that enters from the neighbouring chunk. That envelope falls by
// one step per cell and the chunk's own envelope by at most one step, so once it is no higher than
// the input at a cell it can never win again and the fix-up stops there. The result equals the
// serial recurrence up to the rounding of the decay products.

Texture2D<float> VerticalShare : register(t0);
RWTexture2D<float> EyeOut : register(u0);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2.hlsl"

cbuffer EyeLimiter : register(b3) {
    float eye_occlusion_slope;
    float eye_hole_slope;
    uint eye_extension_cells;
    uint eye_compact;
};

static const uint V2_EYE_EXTENSION_LIMIT = 8u;

// The newest V2_EYE_EXTENSION_LIMIT inputs of a sweep, nearest first.
struct EyeHistory {
    float cell[V2_EYE_EXTENSION_LIMIT];
};

// Largest of `value` and the inputs before it: .x over `cells.x` of them, .y over `cells.y`.
float2 EyeExtended(EyeHistory history, float value, uint2 cells) {
    float2 extended = value;
    for (uint index = 0u; index < V2_EYE_EXTENSION_LIMIT; ++index) {
        float cell = history.cell[index];
        extended = float2(index < cells.x ? max(extended.x, cell) : extended.x,
                          index < cells.y ? max(extended.y, cell) : extended.y);
    }
    return extended;
}

EyeHistory EyeHistoryPush(EyeHistory history, float value) {
    for (uint index = V2_EYE_EXTENSION_LIMIT - 1u; index > 0u; --index) {
        history.cell[index] = history.cell[index - 1u];
    }
    history.cell[0] = value;
    return history;
}

float LoadShare(uint x, uint y, uint2 columns) {
    float value = VerticalShare[uint2(clamp(x, columns.x, columns.y), y)];
    return V2Finite(value) ? V2BoundDisplayParallax(value) : 0.0f;
}

#define V2_EYE_LINE_CHUNKS 32u
#define V2_EYE_BLOCK_CELLS 16u
// Chunks of up to 256 cells: fields up to 8192 cells wide, four times the widest authenticated one.
#define V2_EYE_MAX_BLOCKS 16u
static const float V2_EYE_NO_CARRY = -1.0e30f;

// .x gentle (hole-side slope), .y steep (covered-side slope)
groupshared float2 ForwardEnds[V2_EYE_LINE_CHUNKS][V2_LIMITER_GROUP_LINES];
groupshared float2 BackwardBegins[V2_EYE_LINE_CHUNKS][V2_LIMITER_GROUP_LINES];

// The inputs that precede `x` in sweep direction `direction`, nearest first. Before the first cell
// of a sweep they are the first cell itself, as in the serial recurrence.
EyeHistory EyeHistoryAt(uint x, int direction, uint y, uint2 columns) {
    EyeHistory history;
    for (uint index = 0u; index < V2_EYE_EXTENSION_LIMIT; ++index) {
        int source = (int)x - direction * (int)(index + 1u);
        history.cell[index] = LoadShare((uint)clamp(source, 0, (int)target_w - 1), y, columns);
    }
    return history;
}

[numthreads(V2_LIMITER_GROUP_LINES, V2_EYE_LINE_CHUNKS, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
    uint line_index = group_thread_id.x;
    uint chunk = group_thread_id.y;
    uint y = group_id.x * V2_LIMITER_GROUP_LINES + line_index;
    uint chunk_cells = (target_w + V2_EYE_LINE_CHUNKS - 1u) / V2_EYE_LINE_CHUNKS;
    uint begin = min(chunk * chunk_cells, target_w);
    uint end = min(begin + chunk_cells, target_w);
    uint blocks = (end - begin + V2_EYE_BLOCK_CELLS - 1u) / V2_EYE_BLOCK_CELLS;
    bool active = target_w != 0u && target_h != 0u && y < target_h && begin < end && blocks <= V2_EYE_MAX_BLOCKS;
    uint2 columns = eye_compact != 0u && DepthAnalysisContentValid() ?
        uint2(analysis_content_left, analysis_content_right - 1u) : uint2(0u, max(target_w, 1u) - 1u);
    float width = max(DepthAnalysisContentWidthCells(), 1.0f);
    // Out-of-range requests are pulled back inside the contractive interval instead of producing
    // an unusable field. .x gentle (hole side), .y steep (covered side).
    float2 step = float2(clamp(eye_hole_slope, 0.05f, 0.95f), clamp(eye_occlusion_slope, 0.05f, 0.95f)) / width;
    uint2 cells = uint2(min(eye_extension_cells >> 16u, V2_EYE_EXTENSION_LIMIT),
                        min(eye_extension_cells & 0xffffu, V2_EYE_EXTENSION_LIMIT));
    uint left_y = y;
    uint right_y = y + target_h;

    // This chunk alone. left eye (-1): forward steep, backward gentle; right eye (+1): forward
    // gentle, backward steep.
    float2 forward = V2_EYE_NO_CARRY;
    float2 backward = V2_EYE_NO_CARRY;
    if (active) {
        // Forward: no writes, the state before each block is kept.
        float2 checkpoint[V2_EYE_MAX_BLOCKS];
        EyeHistory history = EyeHistoryAt(begin, 1, y, columns);
        [loop]
        for (uint block = 0u; block < blocks; ++block) {
            checkpoint[block] = forward;
            uint block_begin = begin + block * V2_EYE_BLOCK_CELLS;
            for (uint cell = 0u; cell < V2_EYE_BLOCK_CELLS; ++cell) {
                uint x = block_begin + cell;
                if (x < end) {
                    float value = LoadShare(x, y, columns);
                    forward = max(EyeExtended(history, value, cells), forward - step);
                    history = EyeHistoryPush(history, value);
                }
            }
        }

        // Backward, blocks in reverse order: rebuild the block's forward values, then write the
        // larger of the two directions once.
        EyeHistory reverse_history = EyeHistoryAt(end - 1u, -1, y, columns);
        [loop]
        for (uint remaining = blocks; remaining > 0u; --remaining) {
            uint block = remaining - 1u;
            uint block_begin = begin + block * V2_EYE_BLOCK_CELLS;
            float input[V2_EYE_BLOCK_CELLS];
            float2 rebuilt[V2_EYE_BLOCK_CELLS];
            float2 state = checkpoint[block];
            history = EyeHistoryAt(block_begin, 1, y, columns);
            for (uint cell = 0u; cell < V2_EYE_BLOCK_CELLS; ++cell) {
                uint x = block_begin + cell;
                float value = x < end ? LoadShare(x, y, columns) : 0.0f;
                if (x < end) {
                    state = max(EyeExtended(history, value, cells), state - step);
                    history = EyeHistoryPush(history, value);
                }
                input[cell] = value;
                rebuilt[cell] = state;
            }
            for (uint countdown = V2_EYE_BLOCK_CELLS; countdown > 0u; --countdown) {
                uint cell = countdown - 1u;
                uint x = block_begin + cell;
                if (x < end) {
                    float value = input[cell];
                    backward = max(EyeExtended(reverse_history, value, cells), backward - step);
                    reverse_history = EyeHistoryPush(reverse_history, value);
                    EyeOut[uint2(x, left_y)] = V2BoundDisplayParallax(max(rebuilt[cell].y, backward.x));
                    EyeOut[uint2(x, right_y)] = V2BoundDisplayParallax(max(rebuilt[cell].x, backward.y));
                }
            }
        }
    }
    ForwardEnds[chunk][line_index] = forward;
    BackwardBegins[chunk][line_index] = backward;
    GroupMemoryBarrierWithGroupSync();

    if (active) {
        // The true envelopes just outside this chunk, resolved from the chunks before and after it.
        float2 carry = V2_EYE_NO_CARRY;
        float decay = (float)chunk_cells;
        [loop]
        for (uint before = 0u; before < chunk; ++before) {
            carry = max(ForwardEnds[before][line_index], carry - step * decay);
        }
        bool2 alive = true;
        [loop]
        for (uint x = begin; x < end && any(alive); ++x) {
            carry -= step;
            alive = alive && (carry > LoadShare(x, y, columns));
            if (alive.y) {
                uint2 position = uint2(x, left_y);
                EyeOut[position] = V2BoundDisplayParallax(max(EyeOut[position], carry.y));
            }
            if (alive.x) {
                uint2 position = uint2(x, right_y);
                EyeOut[position] = V2BoundDisplayParallax(max(EyeOut[position], carry.x));
            }
        }

        carry = V2_EYE_NO_CARRY;
        [loop]
        for (uint after = V2_EYE_LINE_CHUNKS; after > chunk + 1u; --after) {
            uint other = after - 1u;
            uint other_begin = min(other * chunk_cells, target_w);
            uint other_cells = min(other_begin + chunk_cells, target_w) - other_begin;
            carry = max(BackwardBegins[other][line_index], carry - step * (float)other_cells);
        }
        alive = true;
        [loop]
        for (uint reverse = end; reverse > begin && any(alive); --reverse) {
            uint x = reverse - 1u;
            carry -= step;
            alive = alive && (carry > LoadShare(x, y, columns));
            if (alive.x) {
                uint2 position = uint2(x, left_y);
                EyeOut[position] = V2BoundDisplayParallax(max(EyeOut[position], carry.x));
            }
            if (alive.y) {
                uint2 position = uint2(x, right_y);
                EyeOut[position] = V2BoundDisplayParallax(max(EyeOut[position], carry.y));
            }
        }
    }
}
