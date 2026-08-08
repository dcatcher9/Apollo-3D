// Final-field burned-in overlay conditioning for Depth Coordinate V2.
//
// This pass runs only for an exact-frame authenticated overlay plan and only after the ordinary
// horizontal limiter. Every loose half-open analysis-pixel rectangle becomes an exact-zero core.
// The original slope-limited field is clamped by the anisotropic distance from that core, so the
// necessary transition is entirely outside the rectangle and is the shortest collar admitted by
// the existing horizontal/vertical contracts. The pass never reads or writes source color.

Texture2D<float> UnconditionedFinalParallax : register(t0);
StructuredBuffer<uint4> OverlayLooseRects : register(t1);
RWTexture2D<float> ConditionedFinalParallax : register(u0);

cbuffer OverlayZeroPlaneConstants : register(b2) {
    uint overlay_analysis_width;
    uint overlay_analysis_height;
    uint overlay_loose_rect_count;
    uint overlay_reserved;
};

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2.hlsl"

// Slot-for-slot mirror of models::host_sbs_overlay_max_loose_rects. The production source-closure
// test binds this shader and the CPU plan definition together; neither value is renderer state.
#define HOST_SBS_OVERLAY_MAX_LOOSE_RECTS 8u

bool OverlayPlanValid() {
    if (overlay_reserved != 0u || overlay_analysis_width == 0u ||
        overlay_analysis_height == 0u || overlay_loose_rect_count == 0u ||
        overlay_loose_rect_count > HOST_SBS_OVERLAY_MAX_LOOSE_RECTS) {
        return false;
    }

    uint rect_capacity;
    uint rect_stride;
    OverlayLooseRects.GetDimensions(rect_capacity, rect_stride);
    if (rect_stride != 16u || overlay_loose_rect_count > rect_capacity) {
        return false;
    }

    [loop]
    for (uint index = 0u; index < overlay_loose_rect_count; ++index) {
        const uint4 rect = OverlayLooseRects[index];
        if (rect.x >= rect.z || rect.y >= rect.w ||
            rect.z > overlay_analysis_width || rect.w > overlay_analysis_height) {
            return false;
        }
    }
    return true;
}

float OverlayDistanceBudget(const float2 field_uv) {
    const float2 analysis_size = float2(
        (float)overlay_analysis_width,
        (float)overlay_analysis_height
    );
    // One complete final-field footprint outside the authored loose rectangle guarantees that
    // bilinear sampling anywhere in that rectangle can only mix exact-zero field texels.
    const float2 sampling_guard = 1.0f / float2((float)target_w, (float)target_h);
    const float vertical_slope_per_source_v =
        v2_max_vertical_shear * ((float)target_h / (float)target_w);
    float minimum_budget = 3.402823466e+38f;

    [loop]
    for (uint index = 0u; index < overlay_loose_rect_count; ++index) {
        const uint4 rect = OverlayLooseRects[index];
        const float2 core_min = (float2(rect.xy) / analysis_size) - sampling_guard;
        const float2 core_max = (float2(rect.zw) / analysis_size) + sampling_guard;
        const float2 outside_distance = max(
            max(core_min - field_uv, field_uv - core_max),
            float2(0.0f, 0.0f)
        );
        const float budget =
            v2_max_horizontal_slope * outside_distance.x +
            vertical_slope_per_source_v * outside_distance.y;
        minimum_budget = min(minimum_budget, budget);
    }
    return minimum_budget;
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= target_w || id.y >= target_h) {
        return;
    }

    const uint2 position = id.xy;
    const float original = UnconditionedFinalParallax[position];
    if (overlay_loose_rect_count == 0u) {
        // The ordinary absent-plan path is an exact field copy.
        ConditionedFinalParallax[position] = original;
        return;
    }
    if (!OverlayPlanValid()) {
        // Mismatched or malformed products disable overlay treatment for this frame.
        ConditionedFinalParallax[position] = original;
        return;
    }

    if (!V2Finite(original)) {
        ConditionedFinalParallax[position] = 0.0f;
        return;
    }
    const float2 field_uv =
        (float2(position) + float2(0.5f, 0.5f)) /
        float2((float)target_w, (float)target_h);
    const float budget = OverlayDistanceBudget(field_uv);
    ConditionedFinalParallax[position] = clamp(original, -budget, budget);
}
