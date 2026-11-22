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
// right-eye field. Each thread owns one row of both halves, writes its forward envelopes there and
// combines them in place during the backward sweep. The production field is not touched.

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

EyeHistory EyeHistoryStart(float value) {
    EyeHistory history;
    for (uint index = 0u; index < V2_EYE_EXTENSION_LIMIT; ++index) {
        history.cell[index] = value;
    }
    return history;
}

// Largest of `value` and the `cells` inputs before it.
float EyeExtended(EyeHistory history, float value, uint cells) {
    float extended = value;
    for (uint index = 0u; index < V2_EYE_EXTENSION_LIMIT; ++index) {
        extended = index < cells ? max(extended, history.cell[index]) : extended;
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

[numthreads(V2_LIMITER_GROUP_LINES, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
    if (target_w == 0u || target_h == 0u) {
        return;
    }
    uint y = group_id.x * V2_LIMITER_GROUP_LINES + group_thread_id.x;
    if (y >= target_h) {
        return;
    }
    uint2 columns = eye_compact != 0u && DepthAnalysisContentValid() ?
        uint2(analysis_content_left, analysis_content_right - 1u) : uint2(0u, target_w - 1u);
    float width = DepthAnalysisContentWidthCells();
    // Out-of-range requests are pulled back inside the contractive interval instead of producing
    // an unusable field.
    float hole_step = clamp(eye_hole_slope, 0.05f, 0.95f) / width;
    float covered_step = clamp(eye_occlusion_slope, 0.05f, 0.95f) / width;
    uint covered_cells = min(eye_extension_cells & 0xffffu, V2_EYE_EXTENSION_LIMIT);
    uint hole_cells = min(eye_extension_cells >> 16u, V2_EYE_EXTENSION_LIMIT);
    uint left_y = y;
    uint right_y = y + target_h;

    float first = LoadShare(0u, y, columns);
    EyeHistory history = EyeHistoryStart(first);
    float gentle = first;
    float steep = first;
    EyeOut[uint2(0u, left_y)] = steep;
    EyeOut[uint2(0u, right_y)] = gentle;
    [loop]
    for (uint forward_x = 1u; forward_x < target_w; ++forward_x) {
        float value = LoadShare(forward_x, y, columns);
        gentle = max(EyeExtended(history, value, hole_cells), gentle - hole_step);
        steep = max(EyeExtended(history, value, covered_cells), steep - covered_step);
        EyeOut[uint2(forward_x, left_y)] = steep;
        EyeOut[uint2(forward_x, right_y)] = gentle;
        history = EyeHistoryPush(history, value);
    }
    DeviceMemoryBarrier();

    first = LoadShare(target_w - 1u, y, columns);
    history = EyeHistoryStart(first);
    gentle = first;
    steep = first;
    [loop]
    for (int backward_x = (int)target_w - 1; backward_x >= 0; --backward_x) {
        uint x = (uint)backward_x;
        if (x + 1u < target_w) {
            float value = LoadShare(x, y, columns);
            gentle = max(EyeExtended(history, value, hole_cells), gentle - hole_step);
            steep = max(EyeExtended(history, value, covered_cells), steep - covered_step);
            history = EyeHistoryPush(history, value);
        }
        uint2 left_position = uint2(x, left_y);
        uint2 right_position = uint2(x, right_y);
        EyeOut[left_position] = V2BoundDisplayParallax(max(EyeOut[left_position], gentle));
        EyeOut[right_position] = V2BoundDisplayParallax(max(EyeOut[right_position], steep));
    }
}
