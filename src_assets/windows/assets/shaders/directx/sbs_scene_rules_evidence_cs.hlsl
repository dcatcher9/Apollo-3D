StructuredBuffer<float> AnalysisGrid : register(t0);
StructuredBuffer<float> PreviousLayoutHistory : register(t1);
Texture2D<float> NormalizedDepth : register(t2);
StructuredBuffer<float> PreviousDepthHistory : register(t3);
StructuredBuffer<float4> DepthFrameState : register(t4);
StructuredBuffer<uint4> PreviousRuleState : register(t5);
StructuredBuffer<float> RawDepth : register(t6);
StructuredBuffer<float4> AdaptiveState : register(t7);

RWStructuredBuffer<float> DenseOutput : register(u0);
RWStructuredBuffer<float> NextLayoutHistory : register(u1);
RWStructuredBuffer<float> NextDepthHistory : register(u2);
RWStructuredBuffer<float> Meta : register(u3);

#include "include/sbs_scene_controller_constants.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"

float Analysis(uint channel, uint2 cell) {
    return AnalysisGrid[SceneAnalysisIndex(channel, cell)];
}

float LayoutHistory(uint channel, uint2 cell) {
    return PreviousLayoutHistory[SceneAnalysisIndex(channel, cell)];
}

float DepthHistory(uint channel, uint2 cell) {
    return PreviousDepthHistory[SceneAnalysisIndex(channel, cell)];
}

float RuleStateWord(uint word) {
    uint4 value = PreviousRuleState[word / 4u];
    return asfloat(value[word & 3u]);
}

uint RuleStateUint(uint word) {
    uint4 value = PreviousRuleState[word / 4u];
    return value[word & 3u];
}

uint EffectiveResetFlags() {
    uint flags = scene_reset_flags;
    if (RuleStateWord(SBS_SCENE_RULE_STATE_WORD_OUTPUT_VALID) <= 0.5f) {
        flags |= RuleStateUint(SBS_SCENE_RULE_STATE_WORD_RESET_FLAGS);
    }
    return flags;
}

float LoadDepth(float2 viewport_uv) {
    uint2 dimensions = uint2(
        max(scene_depth_width, 1u),
        max(scene_depth_height, 1u));
    uint2 pixel = min(
        uint2(saturate(viewport_uv) * float2(dimensions)),
        dimensions - 1u);
    float value = NormalizedDepth.Load(int3(pixel, 0));
    return (isnan(value) || isinf(value)) ? 0.0f : saturate(value);
}

bool NormalizedDepthValid(float2 viewport_uv) {
    uint2 dimensions = uint2(
        max(scene_depth_width, 1u),
        max(scene_depth_height, 1u));
    uint2 pixel = min(
        uint2(saturate(viewport_uv) * float2(dimensions)),
        dimensions - 1u);
    float value = NormalizedDepth.Load(int3(pixel, 0));
    return !isnan(value) && !isinf(value);
}

float LoadRawDepth(float2 viewport_uv, out bool valid) {
    uint2 dimensions = uint2(
        max(scene_depth_width, 1u),
        max(scene_depth_height, 1u));
    uint2 pixel = min(
        uint2(saturate(viewport_uv) * float2(dimensions)),
        dimensions - 1u);
    float value = RawDepth[pixel.y * dimensions.x + pixel.x];
    valid = !isnan(value) && !isinf(value) && value >= 0.0f;
    return valid ? value : 0.0f;
}

float LoadCurrentSignature(int2 cell) {
    int canvas = (int)SBS_SCENE_ANALYSIS_CANVAS_SIZE;
    uint2 clamped_cell = uint2(clamp(cell, int2(0, 0), canvas - 1));
    return Analysis(
        SBS_SCENE_ANALYSIS_GRID_LUMINANCE_ORDINAL,
        clamped_cell);
}

float LoadPreviousSignature(uint2 cell) {
    return LayoutHistory(
        SBS_SCENE_LAYOUT_HISTORY_PREVIOUS_LUMINANCE_ORDINAL,
        cell);
}

void WriteMeta(bool depth_resources_ready) {
    [unroll]
    for (uint word = 0u; word < SBS_SCENE_META_WORD_COUNT; ++word) {
        Meta[word] = 0.0f;
    }

    float state_kind = RuleStateWord(
        SBS_SCENE_RULE_STATE_WORD_STATE_KIND);
    Meta[SBS_SCENE_META_ELAPSED_SECONDS] =
        max(scene_elapsed_seconds, 0.0f);
    Meta[SBS_SCENE_META_SOURCE_ASPECT_RATIO] =
        (float)max(scene_source_width, 1u) /
        (float)max(scene_source_height, 1u);
    Meta[SBS_SCENE_META_ROI_X0] = RuleStateWord(
        SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_X0);
    Meta[SBS_SCENE_META_ROI_Y0] = RuleStateWord(
        SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_Y0);
    Meta[SBS_SCENE_META_ROI_X1] = RuleStateWord(
        SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_X1);
    Meta[SBS_SCENE_META_ROI_Y1] = RuleStateWord(
        SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_Y1);
    Meta[SBS_SCENE_META_STATE_FULL_FRAME] =
        state_kind < 0.5f ? 1.0f : 0.0f;
    Meta[SBS_SCENE_META_STATE_VIDEO] =
        abs(state_kind - 1.0f) < 0.5f ? 1.0f : 0.0f;
    Meta[SBS_SCENE_META_STATE_CONTENT] =
        abs(state_kind - 2.0f) < 0.5f ? 1.0f : 0.0f;
    Meta[SBS_SCENE_META_STATE_SCROLL_HOLD] =
        abs(state_kind - 3.0f) < 0.5f ? 1.0f : 0.0f;
    float valid_depth_fraction =
        SBS_STATE_VALID_DEPTH_FRACTION(
            AdaptiveState[SBS_STATE_VECTOR_VALID_DEPTH_FRACTION]);
    Meta[SBS_SCENE_META_DEPTH_INPUT_VALID] =
        depth_resources_ready &&
        isfinite(valid_depth_fraction) &&
        valid_depth_fraction > 0.05f ?
            1.0f :
            0.0f;
    Meta[SBS_SCENE_META_ROI_GENERATION_CHANGED] = 0.0f;
    uint effective_reset_flags = EffectiveResetFlags();
    Meta[SBS_SCENE_META_LAYOUT_RESET_REQUESTED] =
        (effective_reset_flags & SBS_SCENE_RESET_FLAGS_LAYOUT) != 0u ?
            1.0f :
            0.0f;
    Meta[SBS_SCENE_META_DEPTH_SHOT_RESET_REQUESTED] =
        (effective_reset_flags & SBS_SCENE_RESET_FLAGS_DEPTH_SHOT) != 0u ?
            1.0f :
            0.0f;
    Meta[SBS_SCENE_META_SCROLL_HOLD_ACTIVE] =
        Meta[SBS_SCENE_META_STATE_SCROLL_HOLD];
    Meta[SBS_SCENE_META_HDR_SCRGB_SOURCE] =
        scene_color_mode == 2u ? 1.0f : 0.0f;
    uint adaptive_analysis_flags =
        (uint)max(
            SBS_STATE_ANALYSIS_FLAGS(
                AdaptiveState[SBS_STATE_VECTOR_ANALYSIS_FLAGS]),
            0.0f);
    Meta[SBS_SCENE_META_EXPOSURE_CHANGE_EVIDENCE] =
        (adaptive_analysis_flags & ANALYSIS_FLAG_EXPOSURE_LIKE) != 0u ?
            1.0f :
            0.0f;
    Meta[SBS_SCENE_META_ADAPTIVE_POP_ENABLED] =
        scene_pop_ceiling > scene_pop_floor + 1e-5f ? 1.0f : 0.0f;
    Meta[SBS_SCENE_META_POP_FLOOR] = scene_pop_floor;
    Meta[SBS_SCENE_META_POP_CEILING] = scene_pop_ceiling;
    Meta[SBS_SCENE_META_ZERO_PLANE_SUBJECT_ALLOWED] =
        scene_zero_plane_mode < 1.5f ? 1.0f : 0.0f;
    Meta[SBS_SCENE_META_ZERO_PLANE_MEDIAN_ALLOWED] =
        abs(scene_zero_plane_mode - 2.0f) < 0.5f ? 1.0f : 0.0f;
    Meta[SBS_SCENE_META_ZERO_PLANE_BACKGROUND_ALLOWED] =
        scene_zero_plane_mode > 2.5f ? 1.0f : 0.0f;
    uint current_external_cut_count = asuint(
        SBS_STATE_EXTERNAL_CUT_COUNT(
            AdaptiveState[SBS_STATE_VECTOR_EXTERNAL_CUT_COUNT]));
    uint previous_external_cut_count =
        RuleStateUint(
            SBS_SCENE_RULE_STATE_WORD_LAST_EXTERNAL_CUT_COUNT);
    Meta[SBS_SCENE_META_EXTERNAL_CUT_REQUEST] =
        effective_reset_flags == 0u &&
        current_external_cut_count > previous_external_cut_count ?
                1.0f :
                0.0f;
    Meta[SBS_SCENE_META_SCENE_AGE] = saturate(
        RuleStateWord(SBS_SCENE_RULE_STATE_WORD_SCENE_AGE_S) / 10.0f);
    Meta[SBS_SCENE_META_SECONDS_SINCE_INTERACTION] = 1.0f;
    Meta[SBS_SCENE_META_SOURCE_COLOR_MODE] = (float)scene_color_mode;
    Meta[SBS_SCENE_META_PREVIOUS_BACKEND_OUTPUT_VALID] =
        RuleStateWord(SBS_SCENE_RULE_STATE_WORD_OUTPUT_VALID);
    // Reserved meta[28..31] remain exactly zero.
}

[numthreads(16, 16, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    const uint canvas = SBS_SCENE_ANALYSIS_CANVAS_SIZE;
    if (dispatch_id.x >= canvas || dispatch_id.y >= canvas) {
        return;
    }
    uint2 cell = dispatch_id.xy;
    bool viewport_valid =
        Analysis(SBS_SCENE_ANALYSIS_GRID_VIEWPORT_VALID, cell) > 0.5f;
    float frame_state = DepthFrameState[0].w;
    float2 viewport_uv = float2(
        Analysis(SBS_SCENE_ANALYSIS_GRID_VIEWPORT_X, cell),
        Analysis(SBS_SCENE_ANALYSIS_GRID_VIEWPORT_Y, cell));
    float2 depth_texel = 1.0f / max(
        float2(scene_depth_width, scene_depth_height),
        1.0f.xx);
    bool raw_depth_valid = false;
    LoadRawDepth(viewport_uv, raw_depth_valid);
    bool normalized_depth_valid =
        NormalizedDepthValid(viewport_uv) &&
        NormalizedDepthValid(viewport_uv + float2(depth_texel.x, 0.0f)) &&
        NormalizedDepthValid(viewport_uv - float2(depth_texel.x, 0.0f)) &&
        NormalizedDepthValid(viewport_uv + float2(0.0f, depth_texel.y)) &&
        NormalizedDepthValid(viewport_uv - float2(0.0f, depth_texel.y));
    bool depth_valid =
        viewport_valid && raw_depth_valid && normalized_depth_valid &&
        frame_state > 0.5f &&
        scene_depth_width > 0u && scene_depth_height > 0u;

    float activity = viewport_valid ?
        saturate(Analysis(
            SBS_SCENE_ANALYSIS_GRID_TEMPORAL_ACTIVITY_OCCUPANCY,
            cell)) :
        0.0f;
    float texture_evidence = viewport_valid ?
        saturate(Analysis(
            SBS_SCENE_ANALYSIS_GRID_DENSE_EDGE_TEXTURE,
            cell)) :
        0.0f;
    float variance = viewport_valid ?
        saturate(Analysis(
            SBS_SCENE_ANALYSIS_GRID_LOCAL_LUMINANCE_VARIANCE,
            cell)) :
        0.0f;
    float chroma = viewport_valid ?
        saturate(Analysis(
            SBS_SCENE_ANALYSIS_GRID_CHROMA_SATURATION,
            cell)) :
        0.0f;
    float photographic = viewport_valid ?
        saturate(
            0.55f * texture_evidence +
            0.25f * variance +
            0.20f * chroma) :
        0.0f;

    uint effective_reset_flags = EffectiveResetFlags();
    bool layout_history_valid =
        scene_history_valid != 0u &&
        (effective_reset_flags & SBS_SCENE_RESET_FLAGS_LAYOUT) == 0u;
    bool depth_history_valid =
        scene_history_valid != 0u &&
        (effective_reset_flags &
            (SBS_SCENE_RESET_FLAGS_DEPTH_SHOT |
             SBS_SCENE_RESET_FLAGS_GEOMETRY |
             SBS_SCENE_RESET_FLAGS_BACKEND |
             SBS_SCENE_RESET_FLAGS_DISPLAY_OR_HDR)) == 0u;
    float fast_alpha = SceneEmaAlpha(scene_elapsed_seconds, 0.12f);
    float slow_alpha = SceneEmaAlpha(scene_elapsed_seconds, 1.00f);
    float dwell_alpha = SceneEmaAlpha(scene_elapsed_seconds, 0.60f);
    float previous_fast = LayoutHistory(
        SBS_SCENE_LAYOUT_HISTORY_FAST_ACTIVITY_EMA,
        cell);
    float previous_slow = LayoutHistory(
        SBS_SCENE_LAYOUT_HISTORY_SLOW_ACTIVITY_EMA,
        cell);
    float previous_photo = LayoutHistory(
        SBS_SCENE_LAYOUT_HISTORY_PHOTOGRAPHIC_DENSITY_EMA,
        cell);
    float previous_chroma = LayoutHistory(
        SBS_SCENE_LAYOUT_HISTORY_CHROMA_EMA,
        cell);
    float fast_activity =
        layout_history_valid ?
            lerp(previous_fast, activity, fast_alpha) :
            activity;
    float slow_activity =
        layout_history_valid ?
            lerp(previous_slow, activity, slow_alpha) :
            activity;
    float photo_ema =
        layout_history_valid ?
            lerp(previous_photo, photographic, slow_alpha) :
            photographic;
    float chroma_ema =
        layout_history_valid ?
            lerp(previous_chroma, chroma, slow_alpha) :
            chroma;
    float prior_dwell = layout_history_valid ?
        LayoutHistory(
            SBS_SCENE_LAYOUT_HISTORY_STABLE_OCCUPANCY_DWELL,
            cell) :
        0.0f;
    float occupied = max(photo_ema, slow_activity);
    float dwell = viewport_valid ?
        lerp(prior_dwell, occupied, dwell_alpha) :
        0.0f;

    float best_mismatch = 9.0f;
    float second_mismatch = 9.0f;
    int best_shift = 0;
    uint valid_shift_count = 0u;
    if (viewport_valid && layout_history_valid) {
        uint current_signature = (uint)round(saturate(
            LoadCurrentSignature(int2(cell))) * 255.0f);
        [unroll]
        for (int shift = -3; shift <= 3; ++shift) {
            int2 shifted_cell = int2(cell) + int2(0, shift);
            if (any(shifted_cell < 0) ||
                any(shifted_cell >=
                    (int)SBS_SCENE_ANALYSIS_CANVAS_SIZE)) {
                continue;
            }
            float2 shifted_uv;
            if (!SceneViewportUv(
                    uint2(shifted_cell),
                    SBS_SCENE_ANALYSIS_CANVAS_SIZE,
                    shifted_uv)) {
                continue;
            }
            uint previous_signature = (uint)round(saturate(
                LoadPreviousSignature(uint2(shifted_cell))) *
                255.0f);
            float mismatch =
                (float)countbits(current_signature ^ previous_signature);
            valid_shift_count += 1u;
            if (mismatch < best_mismatch) {
                second_mismatch = best_mismatch;
                best_mismatch = mismatch;
                best_shift = shift;
            } else if (mismatch < second_mismatch) {
                second_mismatch = mismatch;
            }
        }
    }
    float motion_confidence =
        viewport_valid && layout_history_valid &&
        valid_shift_count >= 2u ?
            saturate((second_mismatch - best_mismatch) / 4.0f) :
            0.0f;
    float vertical_motion =
        motion_confidence > 0.10f ? (float)best_shift / 3.0f : 0.0f;

    float depth = depth_valid ? LoadDepth(viewport_uv) :
        DepthHistory(SBS_SCENE_DEPTH_HISTORY_LAST_NORMALIZED_DEPTH, cell);
    float2 depth_dxdy = 0.0f;
    if (depth_valid) {
        depth_dxdy.x = abs(
            LoadDepth(viewport_uv + float2(depth_texel.x, 0.0f)) -
            LoadDepth(viewport_uv - float2(depth_texel.x, 0.0f)));
        depth_dxdy.y = abs(
            LoadDepth(viewport_uv + float2(0.0f, depth_texel.y)) -
            LoadDepth(viewport_uv - float2(0.0f, depth_texel.y)));
    }
    float previous_depth =
        DepthHistory(SBS_SCENE_DEPTH_HISTORY_LAST_NORMALIZED_DEPTH, cell);
    float depth_change =
        depth_valid && depth_history_valid ?
            abs(depth - previous_depth) :
            0.0f;
    float depth_fast = depth_valid ?
        (depth_history_valid ?
            lerp(
                DepthHistory(
                    SBS_SCENE_DEPTH_HISTORY_FAST_DEPTH_EMA,
                    cell),
                depth,
                fast_alpha) :
            depth) :
        DepthHistory(SBS_SCENE_DEPTH_HISTORY_FAST_DEPTH_EMA, cell);
    float depth_slow = depth_valid ?
        (depth_history_valid ?
            lerp(
                DepthHistory(
                    SBS_SCENE_DEPTH_HISTORY_SLOW_DEPTH_EMA,
                    cell),
                depth,
                slow_alpha) :
            depth) :
        DepthHistory(SBS_SCENE_DEPTH_HISTORY_SLOW_DEPTH_EMA, cell);
    float depth_variance = depth_valid ?
        (depth_history_valid ? lerp(
            DepthHistory(
                SBS_SCENE_DEPTH_HISTORY_DEPTH_VARIANCE_EMA,
                cell),
            (depth - depth_slow) * (depth - depth_slow),
            slow_alpha) : 0.0f) :
        DepthHistory(SBS_SCENE_DEPTH_HISTORY_DEPTH_VARIANCE_EMA, cell);
    float depth_change_ema = depth_valid ?
        (depth_history_valid ? lerp(
            DepthHistory(
                SBS_SCENE_DEPTH_HISTORY_ABSOLUTE_DEPTH_CHANGE_EMA,
                cell),
            depth_change,
            fast_alpha) : 0.0f) :
        DepthHistory(
            SBS_SCENE_DEPTH_HISTORY_ABSOLUTE_DEPTH_CHANGE_EMA,
            cell);
    float depth_valid_confidence = depth_valid ?
        (depth_history_valid ? lerp(
            DepthHistory(
                SBS_SCENE_DEPTH_HISTORY_VALID_DEPTH_CONFIDENCE,
                cell),
            1.0f,
            fast_alpha) : 1.0f) :
        (depth_history_valid ? lerp(
            DepthHistory(
                SBS_SCENE_DEPTH_HISTORY_VALID_DEPTH_CONFIDENCE,
                cell),
            0.0f,
            slow_alpha) : 0.0f);

    float edge_attached = viewport_valid &&
        (viewport_uv.x < 0.14f || viewport_uv.x > 0.86f ||
         viewport_uv.y < 0.10f) ? 1.0f : 0.0f;
    float stable_gutter = viewport_valid ?
        saturate(
            (0.18f - photo_ema) * 4.0f +
            (0.10f - slow_activity) * 3.0f) :
        0.0f;
    float chrome =
        edge_attached * saturate(texture_evidence - 0.10f) *
        saturate(1.0f - 1.5f * photo_ema);
    float exterior_competitor =
        edge_attached *
        saturate(0.65f * fast_activity + 0.35f * photo_ema) *
        saturate(1.1f - dwell);
    float video_evidence =
        viewport_valid *
        saturate((fast_activity - 0.05f) * 2.5f) *
        saturate((photo_ema - 0.05f) * 2.0f) *
        saturate(0.35f + dwell) *
        (1.0f - 0.75f * exterior_competitor);
    float content_evidence =
        viewport_valid *
        saturate((photo_ema - 0.16f) * 2.2f) *
        saturate(0.25f + dwell) *
        (1.0f - 0.80f * chrome);
    float focus = max(video_evidence, content_evidence);
    float focus_boundary = saturate(texture_evidence * focus);
    float scroll_support =
        motion_confidence * saturate(abs(vertical_motion) * 1.5f) *
        saturate(1.0f - video_evidence);
    float depth_reference_texel_scale =
        (float)min(scene_depth_width, scene_depth_height) / 434.0f;
    depth_dxdy *= depth_reference_texel_scale;

    [unroll]
    for (uint channel = 0u;
         channel < SBS_SCENE_DENSE_OUT_CHANNEL_COUNT;
         ++channel) {
        DenseOutput[SceneAnalysisIndex(channel, cell)] = 0.0f;
    }
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_UNKNOWN_BACKGROUND, cell)] =
        viewport_valid ? saturate(1.0f - max(focus, chrome)) : 1.0f;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_PRIMARY_PLAYING_VIDEO, cell)] =
        video_evidence;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_PHOTOGRAPHIC_CONTENT_COLLAGE, cell)] =
        content_evidence;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_BROWSER_SYSTEM_CHROME, cell)] = chrome;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_ADVERTISEMENT_COMPETITOR_UNSAFE, cell)] =
        exterior_competitor;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_FOCUS_BOUNDARY, cell)] = focus_boundary;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_STABLE_GUTTER, cell)] = stable_gutter;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_SUBJECT_SALIENCY, cell)] =
        depth_valid ?
            saturate(0.6f * depth + 2.0f * max(depth_dxdy.x, depth_dxdy.y)) :
            0.0f;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_LOCAL_MOTION_X, cell)] = 0.0f;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_LOCAL_MOTION_Y, cell)] = vertical_motion;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_LOCAL_MOTION_CONFIDENCE, cell)] =
        motion_confidence;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_STRUCTURAL_CONTENT_CUT_SUPPORT, cell)] =
        saturate(depth_change * 3.0f) * activity;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_EXPOSURE_ONLY_CHANGE_SUPPORT, cell)] =
        saturate(activity - depth_change);
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_SCROLL_SUPPORT, cell)] = scroll_support;

    NextLayoutHistory[SceneAnalysisIndex(
        SBS_SCENE_LAYOUT_HISTORY_PREVIOUS_LUMINANCE_ORDINAL, cell)] =
        viewport_valid ?
            Analysis(SBS_SCENE_ANALYSIS_GRID_LUMINANCE_ORDINAL, cell) :
            0.0f;
    NextLayoutHistory[SceneAnalysisIndex(
        SBS_SCENE_LAYOUT_HISTORY_FAST_ACTIVITY_EMA, cell)] = fast_activity;
    NextLayoutHistory[SceneAnalysisIndex(
        SBS_SCENE_LAYOUT_HISTORY_SLOW_ACTIVITY_EMA, cell)] = slow_activity;
    NextLayoutHistory[SceneAnalysisIndex(
        SBS_SCENE_LAYOUT_HISTORY_PHOTOGRAPHIC_DENSITY_EMA, cell)] =
        photo_ema;
    NextLayoutHistory[SceneAnalysisIndex(
        SBS_SCENE_LAYOUT_HISTORY_CHROMA_EMA, cell)] = chroma_ema;
    NextLayoutHistory[SceneAnalysisIndex(
        SBS_SCENE_LAYOUT_HISTORY_HORIZONTAL_MOTION, cell)] = 0.0f;
    NextLayoutHistory[SceneAnalysisIndex(
        SBS_SCENE_LAYOUT_HISTORY_VERTICAL_MOTION, cell)] = vertical_motion;
    NextLayoutHistory[SceneAnalysisIndex(
        SBS_SCENE_LAYOUT_HISTORY_MOTION_CONFIDENCE, cell)] =
        motion_confidence;
    NextLayoutHistory[SceneAnalysisIndex(
        SBS_SCENE_LAYOUT_HISTORY_STABLE_OCCUPANCY_DWELL, cell)] = dwell;
    NextLayoutHistory[SceneAnalysisIndex(
        SBS_SCENE_LAYOUT_HISTORY_ACCEPTED_FOCUS_EVIDENCE_EMA, cell)] =
        layout_history_valid ?
            lerp(
                LayoutHistory(
                    SBS_SCENE_LAYOUT_HISTORY_ACCEPTED_FOCUS_EVIDENCE_EMA,
                    cell),
                focus,
                slow_alpha) :
            focus;
    NextLayoutHistory[SceneAnalysisIndex(
        SBS_SCENE_LAYOUT_HISTORY_EXCLUSION_UI_AD_EVIDENCE_EMA, cell)] =
        layout_history_valid ?
            lerp(
                LayoutHistory(
                    SBS_SCENE_LAYOUT_HISTORY_EXCLUSION_UI_AD_EVIDENCE_EMA,
                    cell),
                max(chrome, exterior_competitor),
                slow_alpha) :
            max(chrome, exterior_competitor);
    NextLayoutHistory[SceneAnalysisIndex(
        SBS_SCENE_LAYOUT_HISTORY_SECONDS_SINCE_VALID_LAYOUT, cell)] =
        viewport_valid ? 0.0f :
            min(
                LayoutHistory(
                    SBS_SCENE_LAYOUT_HISTORY_SECONDS_SINCE_VALID_LAYOUT,
                    cell) + max(scene_elapsed_seconds, 0.0f),
                10.0f);

    NextDepthHistory[SceneAnalysisIndex(
        SBS_SCENE_DEPTH_HISTORY_LAST_NORMALIZED_DEPTH, cell)] =
        depth_valid ? depth : previous_depth;
    NextDepthHistory[SceneAnalysisIndex(
        SBS_SCENE_DEPTH_HISTORY_FAST_DEPTH_EMA, cell)] = depth_fast;
    NextDepthHistory[SceneAnalysisIndex(
        SBS_SCENE_DEPTH_HISTORY_SLOW_DEPTH_EMA, cell)] = depth_slow;
    NextDepthHistory[SceneAnalysisIndex(
        SBS_SCENE_DEPTH_HISTORY_SETTLED_SHOT_REFERENCE_DEPTH, cell)] =
        depth_history_valid ?
            DepthHistory(
                SBS_SCENE_DEPTH_HISTORY_SETTLED_SHOT_REFERENCE_DEPTH,
                cell) :
            depth_slow;
    NextDepthHistory[SceneAnalysisIndex(
        SBS_SCENE_DEPTH_HISTORY_DEPTH_VARIANCE_EMA, cell)] = depth_variance;
    NextDepthHistory[SceneAnalysisIndex(
        SBS_SCENE_DEPTH_HISTORY_ABSOLUTE_DEPTH_CHANGE_EMA, cell)] =
        depth_change_ema;
    NextDepthHistory[SceneAnalysisIndex(
        SBS_SCENE_DEPTH_HISTORY_HORIZONTAL_DEPTH_GRADIENT_EMA, cell)] =
        depth_valid ?
            (depth_history_valid ? lerp(
                DepthHistory(
                    SBS_SCENE_DEPTH_HISTORY_HORIZONTAL_DEPTH_GRADIENT_EMA,
                    cell),
                depth_dxdy.x,
                fast_alpha) : depth_dxdy.x) :
            DepthHistory(
                SBS_SCENE_DEPTH_HISTORY_HORIZONTAL_DEPTH_GRADIENT_EMA,
                cell);
    NextDepthHistory[SceneAnalysisIndex(
        SBS_SCENE_DEPTH_HISTORY_VERTICAL_DEPTH_GRADIENT_EMA, cell)] =
        depth_valid ?
            (depth_history_valid ? lerp(
                DepthHistory(
                    SBS_SCENE_DEPTH_HISTORY_VERTICAL_DEPTH_GRADIENT_EMA,
                    cell),
                depth_dxdy.y,
                fast_alpha) : depth_dxdy.y) :
            DepthHistory(
                SBS_SCENE_DEPTH_HISTORY_VERTICAL_DEPTH_GRADIENT_EMA,
                cell);
    NextDepthHistory[SceneAnalysisIndex(
        SBS_SCENE_DEPTH_HISTORY_VALID_DEPTH_CONFIDENCE, cell)] =
        depth_valid_confidence;
    NextDepthHistory[SceneAnalysisIndex(
        SBS_SCENE_DEPTH_HISTORY_SECONDS_SINCE_VALID_DEPTH, cell)] =
        depth_valid ? 0.0f :
            min(
                DepthHistory(
                    SBS_SCENE_DEPTH_HISTORY_SECONDS_SINCE_VALID_DEPTH,
                    cell) + max(scene_elapsed_seconds, 0.0f),
                10.0f);

    // The reducer is deliberately serial, so validate each cell's parallel pass outputs here.
    // A negative sentinel lets the single reduction lane fail closed without rescanning every
    // channel. Missing depth is not numerical corruption: its confidence/history age already
    // degrades locally while appearance/layout evidence remains usable.
    bool outputs_finite = true;
    [loop]
    for (uint channel = 0u;
         channel < SBS_SCENE_DENSE_OUT_CHANNEL_COUNT;
         ++channel) {
        outputs_finite =
            outputs_finite &&
            isfinite(DenseOutput[SceneAnalysisIndex(channel, cell)]);
    }
    [loop]
    for (uint channel = 0u;
         channel < SBS_SCENE_LAYOUT_HISTORY_CHANNEL_COUNT;
         ++channel) {
        outputs_finite =
            outputs_finite &&
            isfinite(NextLayoutHistory[SceneAnalysisIndex(channel, cell)]);
    }
    [loop]
    for (uint channel = 0u;
         channel < SBS_SCENE_DEPTH_HISTORY_CHANNEL_COUNT;
         ++channel) {
        outputs_finite =
            outputs_finite &&
            isfinite(NextDepthHistory[SceneAnalysisIndex(channel, cell)]);
    }
    if (!outputs_finite) {
        DenseOutput[SceneAnalysisIndex(
            SBS_SCENE_DENSE_OUT_UNKNOWN_BACKGROUND, cell)] = -1.0f;
    }

    if (cell.x == 0u && cell.y == 0u) {
        WriteMeta(frame_state > 0.5f);
    }
}
