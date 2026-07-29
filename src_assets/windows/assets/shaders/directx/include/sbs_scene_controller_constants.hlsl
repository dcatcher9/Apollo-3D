#ifndef SBS_SCENE_CONTROLLER_CONSTANTS_HLSL
#define SBS_SCENE_CONTROLLER_CONSTANTS_HLSL

#include "include/sbs_scene_controller_contract.generated.hlsl"

// The 96-byte upload-ring element in sbs_scene_controller_gpu.cpp. The buffer is deliberately
// separate from the depth constants: scene preparation and completed-depth resolution can be
// queued back-to-back without rewriting a constant buffer that an earlier dispatch still owns.
cbuffer SceneControllerConstants : register(b0) {
    uint scene_source_width;
    uint scene_source_height;
    uint scene_depth_width;
    uint scene_depth_height;

    uint scene_color_mode;
    uint scene_backend_generation;
    uint scene_history_valid;
    uint scene_reset_flags;

    float scene_elapsed_seconds;
    float scene_pop_floor;
    float scene_pop_ceiling;
    float scene_zero_plane_mode;

    float scene_acquire_seconds;
    float scene_challenger_seconds;
    float scene_release_seconds;
    float scene_scroll_enter_seconds;

    uint4 scene_ordered_abi_hash_words_0;
    uint4 scene_ordered_abi_hash_words_1;
};

bool SceneAbiMatches() {
    return
        scene_ordered_abi_hash_words_0.x ==
            SBS_SCENE_ORDERED_ABI_HASH_WORD_0 &&
        scene_ordered_abi_hash_words_0.y ==
            SBS_SCENE_ORDERED_ABI_HASH_WORD_1 &&
        scene_ordered_abi_hash_words_0.z ==
            SBS_SCENE_ORDERED_ABI_HASH_WORD_2 &&
        scene_ordered_abi_hash_words_0.w ==
            SBS_SCENE_ORDERED_ABI_HASH_WORD_3 &&
        scene_ordered_abi_hash_words_1.x ==
            SBS_SCENE_ORDERED_ABI_HASH_WORD_4 &&
        scene_ordered_abi_hash_words_1.y ==
            SBS_SCENE_ORDERED_ABI_HASH_WORD_5 &&
        scene_ordered_abi_hash_words_1.z ==
            SBS_SCENE_ORDERED_ABI_HASH_WORD_6 &&
        scene_ordered_abi_hash_words_1.w ==
            SBS_SCENE_ORDERED_ABI_HASH_WORD_7;
}

uint ScenePlanarIndex(uint channel, uint2 pixel, uint canvas_size) {
    return channel * canvas_size * canvas_size +
           pixel.y * canvas_size + pixel.x;
}

uint SceneAnalysisIndex(uint channel, uint2 pixel) {
    return ScenePlanarIndex(
        channel,
        pixel,
        SBS_SCENE_ANALYSIS_CANVAS_SIZE);
}

uint SceneAppearanceIndex(uint channel, uint2 pixel) {
    return ScenePlanarIndex(
        channel,
        pixel,
        SBS_SCENE_APPEARANCE_CANVAS_SIZE);
}

// Return the centered, aspect-preserving viewport on a square canonical canvas. Padding is a
// separate invalid region; it must never be interpreted as black content or numeric zero depth.
void SceneViewport(
    uint canvas_size,
    out uint2 viewport_origin,
    out uint2 viewport_size)
{
    uint source_w = max(scene_source_width, 1u);
    uint source_h = max(scene_source_height, 1u);
    if (source_w >= source_h) {
        viewport_size.x = canvas_size;
        viewport_size.y = max(
            1u,
            min(
                canvas_size,
                (uint)round(
                    (float)canvas_size * (float)source_h / (float)source_w)));
    } else {
        viewport_size.y = canvas_size;
        viewport_size.x = max(
            1u,
            min(
                canvas_size,
                (uint)round(
                    (float)canvas_size * (float)source_w / (float)source_h)));
    }
    viewport_origin = (canvas_size.xx - viewport_size) / 2u;
}

bool SceneViewportUv(
    uint2 pixel,
    uint canvas_size,
    out float2 viewport_uv)
{
    uint2 origin;
    uint2 size;
    SceneViewport(canvas_size, origin, size);
    if (any(pixel < origin) || any(pixel >= origin + size)) {
        viewport_uv = 0.0f;
        return false;
    }
    viewport_uv =
        (float2(pixel - origin) + 0.5f) / max(float2(size), 1.0f.xx);
    return true;
}

float SceneEmaAlpha(float elapsed_seconds, float time_constant_seconds) {
    return 1.0f - exp(
        -max(elapsed_seconds, 0.0f) /
        max(time_constant_seconds, 1e-4f));
}

float SceneLogit(float probability) {
    float p = clamp(probability, 1e-5f, 1.0f - 1e-5f);
    return clamp(log(p / (1.0f - p)), -12.0f, 12.0f);
}

float SceneIou(float4 a, float4 b) {
    float2 lo = max(a.xy, b.xy);
    float2 hi = min(a.zw, b.zw);
    float2 extent = max(hi - lo, 0.0f.xx);
    float intersection = extent.x * extent.y;
    float2 a_extent = max(a.zw - a.xy, 0.0f.xx);
    float2 b_extent = max(b.zw - b.xy, 0.0f.xx);
    float union_area =
        a_extent.x * a_extent.y + b_extent.x * b_extent.y - intersection;
    return union_area > 1e-6f ? intersection / union_area : 0.0f;
}

#endif
