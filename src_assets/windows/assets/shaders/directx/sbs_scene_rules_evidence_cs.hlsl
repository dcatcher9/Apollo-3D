StructuredBuffer<float> AnalysisGrid : register(t0);
StructuredBuffer<float> PreviousLayoutHistory : register(t1);
Texture2D<float> NormalizedDepth : register(t2);
StructuredBuffer<float> PreviousDepthHistory : register(t3);
StructuredBuffer<float4> DepthFrameState : register(t4);
StructuredBuffer<uint4> PreviousRuleState : register(t5);
StructuredBuffer<float> RawDepth : register(t6);
StructuredBuffer<float4> AdaptiveState : register(t7);
StructuredBuffer<uint4> FrameRoiTransform : register(t8);

RWStructuredBuffer<float> DenseOutput : register(u0);
RWStructuredBuffer<float> NextLayoutHistory : register(u1);
RWStructuredBuffer<float> NextDepthHistory : register(u2);
RWStructuredBuffer<float> Meta : register(u3);
RWStructuredBuffer<float> RuleSummary : register(u4);

#include "include/sbs_scene_controller_constants.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"
#include "include/sbs_frame_roi_transform.hlsl"
#include "include/sbs_scene_rule_state.hlsl"
#include "include/sbs_scene_rules_summary.hlsl"
#include "include/sbs_scene_rules_media.hlsl"

groupshared float4 RoiEventGroupSums[
    SBS_RULE_EVIDENCE_GROUP_THREAD_COUNT];
groupshared float4 ScrollVoteGroupSumsY[
    SBS_RULE_EVIDENCE_GROUP_THREAD_COUNT];
groupshared float4 ScrollVoteGroupSumsX[
    SBS_RULE_EVIDENCE_GROUP_THREAD_COUNT];
groupshared float2 FreshActivityGroupSums[
    SBS_RULE_EVIDENCE_GROUP_THREAD_COUNT];

float Analysis(uint channel, uint2 cell) {
    return AnalysisGrid[SceneAnalysisIndex(channel, cell)];
}

float LayoutHistory(uint channel, uint2 cell) {
    return PreviousLayoutHistory[SceneAnalysisIndex(channel, cell)];
}

float DepthHistory(uint channel, uint2 cell) {
    return PreviousDepthHistory[SceneAnalysisIndex(channel, cell)];
}

float LoadNormalizedDepth(float2 depth_uv, out bool valid) {
    uint2 dimensions = uint2(
        max(scene_depth_width, 1u),
        max(scene_depth_height, 1u));
    uint2 pixel = min(
        uint2(saturate(depth_uv) * float2(dimensions)),
        dimensions - 1u);
    float value = NormalizedDepth.Load(int3(pixel, 0));
    valid = !isnan(value) && !isinf(value);
    return valid ? saturate(value) : 0.0f;
}

float LoadRawDepth(float2 depth_uv, out bool valid) {
    uint2 dimensions = uint2(
        max(scene_depth_width, 1u),
        max(scene_depth_height, 1u));
    uint2 pixel = min(
        uint2(saturate(depth_uv) * float2(dimensions)),
        dimensions - 1u);
    float value = RawDepth[pixel.y * dimensions.x + pixel.x];
    valid = !isnan(value) && !isinf(value) && value >= 0.0f;
    return valid ? value : 0.0f;
}

bool ResolveFrameDepthTransform(
    out SbsFrameRoiTransformData transform,
    out bool explicit_legacy)
{
    transform = FrameRoiDecode(
        0u.xxxx,
        0u.xxxx,
        0u.xxxx,
        0u.xxxx,
        0u.xxxx,
        0u.xxxx,
        0u.xxxx,
        0u.xxxx);
    explicit_legacy = false;

    uint transform_vectors = 0u;
    uint transform_stride = 0u;
    FrameRoiTransform.GetDimensions(
        transform_vectors,
        transform_stride);
    if (transform_vectors < SBS_FRAME_ROI_VECTOR_COUNT) {
        return false;
    }
    transform = FrameRoiTransformLoad();

    bool exact_resource_shape =
        transform_vectors == SBS_FRAME_ROI_VECTOR_COUNT &&
        transform_stride == 16u;
    explicit_legacy =
        exact_resource_shape &&
        FrameRoiDataUnboundZero(transform);
    if (explicit_legacy) {
        // The all-zero eight-vector resource is the original full-frame ABI. Keep its UV and
        // resource-validity behavior byte-for-byte; only nonzero owned transforms opt into the
        // stricter identity and ROI checks below.
        return true;
    }

    uint normalized_width = 0u;
    uint normalized_height = 0u;
    NormalizedDepth.GetDimensions(
        normalized_width,
        normalized_height);
    uint raw_elements = 0u;
    uint raw_stride = 0u;
    RawDepth.GetDimensions(raw_elements, raw_stride);
    bool model_pixel_count_safe =
        scene_depth_width > 0u &&
        scene_depth_height > 0u &&
        scene_depth_width <=
            0xffffffffu / max(scene_depth_height, 1u);
    bool resources_match_model =
        model_pixel_count_safe &&
        normalized_width == scene_depth_width &&
        normalized_height == scene_depth_height &&
        raw_stride == 4u &&
        raw_elements == scene_depth_width * scene_depth_height;

    return
        exact_resource_shape &&
        resources_match_model &&
        FrameRoiDataValid(transform) &&
        all(FrameRoiDataSourceFrameId(transform) ==
            uint2(
                scene_source_frame_id_low,
                scene_source_frame_id_high)) &&
        FrameRoiDataBackendGeneration(transform) ==
            scene_backend_generation &&
        all(FrameRoiDataSourceDimensions(transform) ==
            uint2(scene_source_width, scene_source_height)) &&
        all(FrameRoiDataModelDimensions(transform) ==
            uint2(scene_depth_width, scene_depth_height));
}

bool SourceToDepthUv(
    SbsFrameRoiTransformData transform,
    bool explicit_legacy,
    float2 source_uv,
    out float2 depth_uv)
{
    if (explicit_legacy) {
        depth_uv = source_uv;
        return true;
    }
    if (!FrameRoiDataSourceInsideFocus(transform, source_uv) ||
        !FrameRoiDataSourceToModelUv(
            transform,
            source_uv,
            depth_uv)) {
        depth_uv = 0.0f.xx;
        return false;
    }
    uint2 dimensions = uint2(scene_depth_width, scene_depth_height);
    uint2 pixel = min(
        uint2(saturate(depth_uv) * float2(dimensions)),
        dimensions - 1u);
    return FrameRoiDataAcceptedModelPixel(transform, pixel);
}

bool DepthUvAccepted(
    SbsFrameRoiTransformData transform,
    bool explicit_legacy,
    float2 depth_uv)
{
    if (explicit_legacy) {
        return true;
    }
    if (any(depth_uv < 0.0f.xx) || any(depth_uv >= 1.0f.xx)) {
        return false;
    }
    uint2 dimensions = uint2(scene_depth_width, scene_depth_height);
    uint2 pixel = uint2(depth_uv * float2(dimensions));
    return FrameRoiDataAcceptedModelPixel(transform, pixel);
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

void WriteMeta(
    bool depth_resources_ready,
    bool roi_generation_changed)
{
    [unroll]
    for (uint word = 0u; word < SBS_SCENE_META_WORD_COUNT; ++word) {
        Meta[word] = 0.0f;
    }

    float state_kind = SbsRulePreviousStateWord(
        SBS_SCENE_RULE_STATE_WORD_STATE_KIND);
    const uint state_flags = SbsRulePreviousStateUint(
        SBS_SCENE_RULE_STATE_WORD_STATE_FLAGS);
    Meta[SBS_SCENE_META_ELAPSED_SECONDS] =
        max(scene_elapsed_seconds, 0.0f);
    Meta[SBS_SCENE_META_SOURCE_ASPECT_RATIO] =
        (float)max(scene_source_width, 1u) /
        (float)max(scene_source_height, 1u);
    Meta[SBS_SCENE_META_ROI_X0] = SbsRulePreviousStateWord(
        SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_X0);
    Meta[SBS_SCENE_META_ROI_Y0] = SbsRulePreviousStateWord(
        SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_Y0);
    Meta[SBS_SCENE_META_ROI_X1] = SbsRulePreviousStateWord(
        SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_X1);
    Meta[SBS_SCENE_META_ROI_Y1] = SbsRulePreviousStateWord(
        SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_Y1);
    Meta[SBS_SCENE_META_STATE_FULL_FRAME] =
        state_kind < 0.5f ? 1.0f : 0.0f;
    Meta[SBS_SCENE_META_STATE_VIDEO] =
        abs(state_kind - 1.0f) < 0.5f ? 1.0f : 0.0f;
    Meta[SBS_SCENE_META_STATE_CONTENT] =
        abs(state_kind - 2.0f) < 0.5f ? 1.0f : 0.0f;
    float valid_depth_fraction =
        SBS_STATE_VALID_DEPTH_FRACTION(
            AdaptiveState[SBS_STATE_VECTOR_VALID_DEPTH_FRACTION]);
    Meta[SBS_SCENE_META_DEPTH_INPUT_VALID] =
        depth_resources_ready &&
        isfinite(valid_depth_fraction) &&
        valid_depth_fraction > 0.05f ?
            1.0f :
            0.0f;
    Meta[SBS_SCENE_META_ROI_GENERATION_CHANGED] =
        roi_generation_changed ? 1.0f : 0.0f;
    uint effective_reset_flags = SbsRuleEffectiveResetFlags();
    Meta[SBS_SCENE_META_LAYOUT_RESET_REQUESTED] =
        (effective_reset_flags & SBS_SCENE_RESET_FLAGS_LAYOUT) != 0u ?
            1.0f :
            0.0f;
    Meta[SBS_SCENE_META_DEPTH_SHOT_RESET_REQUESTED] =
        (effective_reset_flags & SBS_SCENE_RESET_FLAGS_DEPTH_SHOT) != 0u ?
            1.0f :
            0.0f;
    Meta[SBS_SCENE_META_SCROLL_HOLD_ACTIVE] =
        (state_flags & SBS_SCENE_STATE_FLAGS_SCROLL_HOLD_ACTIVE) != 0u ?
            1.0f :
            0.0f;
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
        SbsRulePreviousStateUint(
            SBS_SCENE_RULE_STATE_WORD_LAST_EXTERNAL_CUT_COUNT);
    Meta[SBS_SCENE_META_EXTERNAL_CUT_REQUEST] =
        effective_reset_flags == 0u &&
        current_external_cut_count > previous_external_cut_count ?
                1.0f :
                0.0f;
    Meta[SBS_SCENE_META_SCENE_AGE] = saturate(
        SbsRulePreviousStateWord(
            SBS_SCENE_RULE_STATE_WORD_SCENE_AGE_S) / 10.0f);
    Meta[SBS_SCENE_META_SECONDS_SINCE_INTERACTION] = 1.0f;
    Meta[SBS_SCENE_META_SOURCE_COLOR_MODE] = (float)scene_color_mode;
    Meta[SBS_SCENE_META_PREVIOUS_BACKEND_OUTPUT_VALID] =
        SbsRulePreviousStateWord(
            SBS_SCENE_RULE_STATE_WORD_OUTPUT_VALID);
    // Reserved meta[9, 28..31] remain exactly zero.
}

[numthreads(
    SBS_RULE_EVIDENCE_GROUP_SIZE,
    SBS_RULE_EVIDENCE_GROUP_SIZE,
    1)]
void main(
    uint3 dispatch_id : SV_DispatchThreadID,
    uint3 group_thread_id : SV_GroupThreadID,
    uint3 group_id : SV_GroupID) {
    const uint canvas = SBS_SCENE_ANALYSIS_CANVAS_SIZE;
    uint2 cell = dispatch_id.xy;
    uint2 viewport_origin;
    uint2 viewport_size;
    SceneViewport(canvas, viewport_origin, viewport_size);
    const int2 viewport_first = int2(viewport_origin);
    const int2 viewport_end = int2(viewport_origin + viewport_size);
    const float viewport_flag = Analysis(
        SBS_SCENE_ANALYSIS_GRID_VIEWPORT_VALID,
        cell);
    bool viewport_valid =
        isfinite(viewport_flag) && viewport_flag > 0.5f;
    float frame_state = DepthFrameState[0].w;
    float2 viewport_uv = float2(
        Analysis(SBS_SCENE_ANALYSIS_GRID_VIEWPORT_X, cell),
        Analysis(SBS_SCENE_ANALYSIS_GRID_VIEWPORT_Y, cell));
    SbsFrameRoiTransformData frame_roi_transform;
    bool explicit_legacy_transform = false;
    bool frame_depth_transform_valid = ResolveFrameDepthTransform(
        frame_roi_transform,
        explicit_legacy_transform);
    float2 depth_uv = viewport_uv;
    bool depth_location_valid =
        frame_depth_transform_valid &&
        SourceToDepthUv(
            frame_roi_transform,
            explicit_legacy_transform,
            viewport_uv,
            depth_uv);
    float2 depth_texel = 1.0f / max(
        float2(scene_depth_width, scene_depth_height),
        1.0f.xx);
    bool raw_depth_valid = false;
    if (depth_location_valid) {
        LoadRawDepth(depth_uv, raw_depth_valid);
    }
    const float2 depth_offset_x = float2(depth_texel.x, 0.0f);
    const float2 depth_offset_y = float2(0.0f, depth_texel.y);
    bool normalized_depth_region_accepted =
        depth_location_valid &&
        DepthUvAccepted(
            frame_roi_transform,
            explicit_legacy_transform,
            depth_uv) &&
        DepthUvAccepted(
            frame_roi_transform,
            explicit_legacy_transform,
            depth_uv + depth_offset_x) &&
        DepthUvAccepted(
            frame_roi_transform,
            explicit_legacy_transform,
            depth_uv - depth_offset_x) &&
        DepthUvAccepted(
            frame_roi_transform,
            explicit_legacy_transform,
            depth_uv + depth_offset_y) &&
        DepthUvAccepted(
            frame_roi_transform,
            explicit_legacy_transform,
            depth_uv - depth_offset_y);
    float depth_center = 0.0f;
    float depth_left = 0.0f;
    float depth_right = 0.0f;
    float depth_up = 0.0f;
    float depth_down = 0.0f;
    bool depth_center_valid = false;
    bool depth_left_valid = false;
    bool depth_right_valid = false;
    bool depth_up_valid = false;
    bool depth_down_valid = false;
    if (normalized_depth_region_accepted) {
        depth_center = LoadNormalizedDepth(depth_uv, depth_center_valid);
        depth_left = LoadNormalizedDepth(
            depth_uv - depth_offset_x,
            depth_left_valid);
        depth_right = LoadNormalizedDepth(
            depth_uv + depth_offset_x,
            depth_right_valid);
        depth_up = LoadNormalizedDepth(
            depth_uv - depth_offset_y,
            depth_up_valid);
        depth_down = LoadNormalizedDepth(
            depth_uv + depth_offset_y,
            depth_down_valid);
    }
    bool normalized_depth_valid =
        depth_center_valid &&
        depth_left_valid &&
        depth_right_valid &&
        depth_up_valid &&
        depth_down_valid;
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

    uint effective_reset_flags = SbsRuleEffectiveResetFlags();
    bool layout_history_valid =
        scene_history_valid != 0u &&
        SbsRulePreviousTargetsUsable();
    const bool previous_scroll_hold_active =
        layout_history_valid &&
        (SbsRulePreviousStateUint(
             SBS_SCENE_RULE_STATE_WORD_STATE_FLAGS) &
         SBS_SCENE_STATE_FLAGS_SCROLL_HOLD_ACTIVE) != 0u;
    // Keep the ordinal endpoint valid so continuing page translation remains
    // observable, but never blend semantic/spatial attribution from before a
    // scroll into the candidate that may be promoted on release.
    const bool semantic_layout_history_valid =
        layout_history_valid && !previous_scroll_hold_active;
    // Normalized depth is only comparable while both frames use the same normalization
    // geometry. A newly accepted ROI carries reset debt until its first valid depth promotes the
    // matching transform, and MinMaxEma.frame_state == 2 marks that same direct reseed even for
    // the explicit legacy/full-frame path. Rebase the depth histories on this frame, but never
    // interpret its new normalization coordinates as scene structure.
    bool depth_geometry_reseed =
        frame_state > 1.5f ||
        (!explicit_legacy_transform &&
         FrameRoiDataResetDebt(frame_roi_transform));
    bool depth_history_valid =
        scene_history_valid != 0u &&
        !depth_geometry_reseed &&
        (effective_reset_flags &
            (SBS_SCENE_RESET_FLAGS_DEPTH_SHOT |
             SBS_SCENE_RESET_FLAGS_GEOMETRY |
             SBS_SCENE_RESET_FLAGS_BACKEND |
             SBS_SCENE_RESET_FLAGS_DISPLAY_OR_HDR)) == 0u;
    const float previous_depth_valid_confidence = saturate(
        DepthHistory(
            SBS_SCENE_DEPTH_HISTORY_VALID_DEPTH_CONFIDENCE,
            cell));
    const bool depth_history_comparable =
        depth_history_valid &&
        !previous_scroll_hold_active &&
        previous_depth_valid_confidence >= 0.5f;
    float fast_alpha = SceneEmaAlpha(scene_elapsed_seconds, 0.12f);
    float activity_alpha = SceneEmaAlpha(scene_elapsed_seconds, 0.50f);
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
        semantic_layout_history_valid ?
            lerp(previous_fast, activity, fast_alpha) :
            activity;
    float slow_activity =
        semantic_layout_history_valid ?
            lerp(previous_slow, activity, activity_alpha) :
            activity;
    float photo_ema =
        semantic_layout_history_valid ?
            lerp(previous_photo, photographic, slow_alpha) :
            photographic;
    float chroma_ema =
        semantic_layout_history_valid ?
            lerp(previous_chroma, chroma, slow_alpha) :
            chroma;
    float prior_dwell = semantic_layout_history_valid ?
        LayoutHistory(
            SBS_SCENE_LAYOUT_HISTORY_STABLE_OCCUPANCY_DWELL,
            cell) :
        0.0f;
    float occupied = max(photo_ema, slow_activity);
    float dwell = viewport_valid ?
        lerp(prior_dwell, occupied, dwell_alpha) :
        0.0f;
    if (previous_scroll_hold_active) {
        // Page translation is not playback occupancy. Keep the luminance
        // signature advancing so scroll remains observable, but erase the
        // activity/dwell history that would otherwise turn a translated page
        // into a full-frame VIDEO proposal after the hold releases.
        fast_activity = 0.0f;
        slow_activity = 0.0f;
        dwell = 0.0f;
    }

    float scroll_shift_costs_x[7];
    float scroll_shift_costs_y[7];
    [unroll]
    for (uint shift_index = 0u; shift_index < 7u; ++shift_index) {
        scroll_shift_costs_x[shift_index] = 0.0f;
        scroll_shift_costs_y[shift_index] = 0.0f;
    }
    const bool scroll_search_eligible =
        viewport_valid &&
        layout_history_valid &&
        texture_evidence >= SBS_RULE_SCROLL_MIN_TEXTURE;
    float4 previous_video_bounds;
    const bool previous_video_target =
        SbsRulePreviousVideoTarget(previous_video_bounds);
    const bool inside_previous_video_target =
        previous_video_target &&
        all(viewport_uv >= previous_video_bounds.xy) &&
        all(viewport_uv <= previous_video_bounds.zw);
    // Scroll is a frame-level safety overlay, not a per-cell classifier. Only cells with a
    // complete seven-position comparison contribute to the global X/Y translation fit. Player
    // interiors are excluded using the previous accepted target, so ordinary motion inside a
    // video cannot freeze its own ROI while coherent page motion outside it still can.
    const bool horizontal_scroll_observable =
        scroll_search_eligible &&
        !inside_previous_video_target &&
        int(cell.x) >= viewport_first.x + 3 &&
        int(cell.x) + 3 < viewport_end.x;
    const bool vertical_scroll_observable =
        scroll_search_eligible &&
        !inside_previous_video_target &&
        int(cell.y) >= viewport_first.y + 3 &&
        int(cell.y) + 3 < viewport_end.y;
    if (horizontal_scroll_observable || vertical_scroll_observable) {
        const uint current_signature = (uint)round(saturate(
            LoadCurrentSignature(int2(cell))) * 255.0f);
        [unroll]
        for (int shift = -3; shift <= 3; ++shift) {
            if (horizontal_scroll_observable) {
                const int2 shifted_cell_x =
                    int2(cell) + int2(shift, 0);
                const uint previous_signature_x =
                    (uint)round(saturate(
                        LoadPreviousSignature(
                            uint2(shifted_cell_x))) *
                        255.0f);
                const float mismatch_x =
                    (float)countbits(
                        current_signature ^ previous_signature_x);
                scroll_shift_costs_x[shift + 3] = mismatch_x;
            }

            if (vertical_scroll_observable) {
                const int2 shifted_cell_y =
                    int2(cell) + int2(0, shift);
                const uint previous_signature_y =
                    (uint)round(saturate(
                        LoadPreviousSignature(
                            uint2(shifted_cell_y))) *
                        255.0f);
                const float mismatch_y =
                    (float)countbits(
                        current_signature ^ previous_signature_y);
                scroll_shift_costs_y[shift + 3] = mismatch_y;
            }
        }
    }

    float depth = depth_valid ? depth_center :
        DepthHistory(SBS_SCENE_DEPTH_HISTORY_LAST_NORMALIZED_DEPTH, cell);
    float2 depth_dxdy = 0.0f;
    if (depth_valid) {
        depth_dxdy.x = abs(depth_right - depth_left);
        depth_dxdy.y = abs(depth_down - depth_up);
    }
    float previous_depth =
        DepthHistory(SBS_SCENE_DEPTH_HISTORY_LAST_NORMALIZED_DEPTH, cell);
    float depth_change =
        depth_valid && depth_history_comparable ?
            abs(depth - previous_depth) :
            0.0f;
    float depth_fast = depth_valid ?
        (depth_history_comparable ?
            lerp(
                DepthHistory(
                    SBS_SCENE_DEPTH_HISTORY_FAST_DEPTH_EMA,
                    cell),
                depth,
                fast_alpha) :
            depth) :
        DepthHistory(SBS_SCENE_DEPTH_HISTORY_FAST_DEPTH_EMA, cell);
    float depth_slow = depth_valid ?
        (depth_history_comparable ?
            lerp(
                DepthHistory(
                    SBS_SCENE_DEPTH_HISTORY_SLOW_DEPTH_EMA,
                    cell),
                depth,
                slow_alpha) :
            depth) :
        DepthHistory(SBS_SCENE_DEPTH_HISTORY_SLOW_DEPTH_EMA, cell);
    float depth_variance = depth_valid ?
        (depth_history_comparable ? lerp(
            DepthHistory(
                SBS_SCENE_DEPTH_HISTORY_DEPTH_VARIANCE_EMA,
                cell),
            (depth - depth_slow) * (depth - depth_slow),
            slow_alpha) : 0.0f) :
        DepthHistory(SBS_SCENE_DEPTH_HISTORY_DEPTH_VARIANCE_EMA, cell);
    float depth_change_ema = depth_valid ?
        (depth_history_comparable ? lerp(
            DepthHistory(
                SBS_SCENE_DEPTH_HISTORY_ABSOLUTE_DEPTH_CHANGE_EMA,
                cell),
            depth_change,
            fast_alpha) : 0.0f) :
        DepthHistory(
            SBS_SCENE_DEPTH_HISTORY_ABSOLUTE_DEPTH_CHANGE_EMA,
            cell);
    float depth_valid_confidence = depth_valid ?
        (depth_history_comparable ? lerp(
            previous_depth_valid_confidence,
            1.0f,
            fast_alpha) : 1.0f) :
        (depth_history_comparable ? lerp(
            previous_depth_valid_confidence,
            0.0f,
            slow_alpha) : 0.0f);

    float stable_gutter = viewport_valid ?
        saturate(
            (0.18f - photo_ema) * 4.0f +
            (0.10f - slow_activity) * 3.0f) :
        0.0f;
    // UI-like texture and short-lived competing activity are deliberately geometry-neutral.
    // A left/right/top browser-band prior made identical media depend on orientation and treated
    // the bottom edge differently. These are soft semantic cues only: persistent photographic
    // support can overcome them at every location, including all four viewport edges.
    float ui_like =
        (viewport_valid ? 1.0f : 0.0f) *
        saturate(texture_evidence - 0.10f) *
        saturate((0.45f - photo_ema) * 2.0f) *
        saturate(0.40f + 0.60f * (1.0f - slow_activity));
    const float transient_activity =
        saturate((fast_activity - slow_activity) * 2.0f);
    float transient_competitor =
        (viewport_valid ? 1.0f : 0.0f) *
        transient_activity *
        saturate(1.0f - dwell) *
        saturate(0.25f + 0.75f * ui_like);
    bool static_media_inputs_valid;
    const float static_media_weight = SbsRuleStaticMediaWeight(
        viewport_flag,
        transient_competitor,
        ui_like,
        variance,
        chroma,
        photo_ema,
        chroma_ema,
        static_media_inputs_valid);
    // Playing video often changes only a small fraction of its pixels on any one capture
    // (especially after screen-recording cadence conversion).  The old fast-EMA threshold
    // therefore saw the moving subject but never the stable player envelope.  Treat the slow
    // activity EMA as temporal occupancy: infrequent changes accumulate across the panel while
    // static browser chrome remains dark.  Photographic support still has to corroborate every
    // cell, and the reducer still requires a coherent two-axis run before it can propose an ROI.
    // This lane remains an intentionally heuristic temporal proposal; unlike the static-media
    // lane below, it is not the authoritative scalar classifier for a reducer mass.
    float temporal_video_occupancy =
        saturate((slow_activity - 0.0025f) * 120.0f);
    float video_evidence =
        viewport_valid *
        temporal_video_occupancy *
        saturate((photo_ema - 0.04f) * 4.0f) *
        saturate(0.45f + dwell) *
        (1.0f - 0.75f * transient_competitor);
    // Focus-boundary diagnostics may remain permissive; they do not own static ROI mass.
    float static_focus_heuristic =
        viewport_valid *
        saturate((photo_ema - 0.16f) * 2.2f) *
        saturate(0.25f + dwell) *
        (1.0f - 0.80f * ui_like);
    float focus = max(video_evidence, static_focus_heuristic);
    float focus_boundary = saturate(texture_evidence * focus);
    float structural_cut_support =
        depth_valid ? saturate(depth_change * 3.0f) * activity : 0.0f;
    float exposure_only_support =
        depth_valid ? saturate(activity - depth_change) : 0.0f;
    float roi_coverage = Analysis(
        SBS_SCENE_ANALYSIS_GRID_CURRENT_ROI_COVERAGE,
        cell);
    float roi_depth_weight =
        isfinite(roi_coverage) && roi_coverage > 0.5f ?
            saturate(depth_valid_confidence) :
            0.0f;
    // A large negative group sum is a compact corruption sentinel. The plan pass converts it
    // back into required-invalid evidence before any controller decision is allowed.
    float roi_cell_weight =
        !isfinite(roi_coverage) ? -65536.0f :
        roi_coverage > 0.5f ? 1.0f :
        0.0f;
    float depth_reference_texel_scale =
        (float)min(scene_depth_width, scene_depth_height) / 434.0f;
    depth_dxdy *= depth_reference_texel_scale;

    const uint group_lane =
        group_thread_id.y * SBS_RULE_EVIDENCE_GROUP_SIZE +
        group_thread_id.x;
    RoiEventGroupSums[group_lane] = float4(
        roi_cell_weight,
        roi_depth_weight,
        saturate(structural_cut_support) * roi_depth_weight,
        saturate(exposure_only_support) * roi_depth_weight);
    ScrollVoteGroupSumsY[group_lane] = float4(
        SbsRuleClassifyScrollVote(
            vertical_scroll_observable,
            scroll_shift_costs_y[0],
            scroll_shift_costs_y[1],
            scroll_shift_costs_y[2],
            scroll_shift_costs_y[3],
            scroll_shift_costs_y[4],
            scroll_shift_costs_y[5],
            scroll_shift_costs_y[6]),
        0.0f);
    ScrollVoteGroupSumsX[group_lane] = float4(
        SbsRuleClassifyScrollVote(
            horizontal_scroll_observable,
            scroll_shift_costs_x[0],
            scroll_shift_costs_x[1],
            scroll_shift_costs_x[2],
            scroll_shift_costs_x[3],
            scroll_shift_costs_x[4],
            scroll_shift_costs_x[5],
            scroll_shift_costs_x[6]),
        0.0f);
    // The slow activity EMA owns envelope discovery and retention, but it cannot prove that a
    // provisional player is still moving. Preserve current ordinal activity only where the
    // canonical, photographic temporal classifier also has support. Raw repaint activity must
    // never bypass that classifier and turn animated text or browser chrome into VIDEO.
    const float classified_fresh_activity =
        SbsRuleTemporalMass(video_evidence) > 0.0f ?
            activity :
            0.0f;
    FreshActivityGroupSums[group_lane] = float2(
        classified_fresh_activity,
        inside_previous_video_target ?
            classified_fresh_activity :
            0.0f);
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = SBS_RULE_EVIDENCE_GROUP_THREAD_COUNT / 2u;
         stride > 0u;
         stride >>= 1u) {
        if (group_lane < stride) {
            RoiEventGroupSums[group_lane] +=
                RoiEventGroupSums[group_lane + stride];
            ScrollVoteGroupSumsY[group_lane] +=
                ScrollVoteGroupSumsY[group_lane + stride];
            ScrollVoteGroupSumsX[group_lane] +=
                ScrollVoteGroupSumsX[group_lane + stride];
            FreshActivityGroupSums[group_lane] +=
                FreshActivityGroupSums[group_lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (group_lane == 0u) {
        const uint group_index =
            group_id.y * SBS_RULE_EVIDENCE_GROUPS_PER_AXIS +
            group_id.x;
        const float4 group_sum = RoiEventGroupSums[0];
        const float3 scroll_y = ScrollVoteGroupSumsY[0].xyz;
        const float3 scroll_x = ScrollVoteGroupSumsX[0].xyz;
        const float2 fresh_activity = FreshActivityGroupSums[0];
        RuleSummary[SbsRuleEventGroupIndex(
            group_index,
            SBS_RULE_EVENT_GROUP_ROI_CELLS)] = group_sum.x;
        RuleSummary[SbsRuleEventGroupIndex(
            group_index,
            SBS_RULE_EVENT_GROUP_DEPTH_WEIGHT)] = group_sum.y;
        RuleSummary[SbsRuleEventGroupIndex(
            group_index,
            SBS_RULE_EVENT_GROUP_ROI_STRUCTURAL_WEIGHT)] = group_sum.z;
        RuleSummary[SbsRuleEventGroupIndex(
            group_index,
            SBS_RULE_EVENT_GROUP_ROI_EXPOSURE_WEIGHT)] = group_sum.w;
        RuleSummary[SbsRuleEventGroupIndex(
            group_index,
            SBS_RULE_EVENT_GROUP_SCROLL_Y_WEIGHT)] = scroll_y.x;
        RuleSummary[SbsRuleEventGroupIndex(
            group_index,
            SBS_RULE_EVENT_GROUP_SCROLL_Y_SIGNED)] = scroll_y.y;
        RuleSummary[SbsRuleEventGroupIndex(
            group_index,
            SBS_RULE_EVENT_GROUP_SCROLL_Y_SUPPORT)] = scroll_y.z;
        RuleSummary[SbsRuleEventGroupIndex(
            group_index,
            SBS_RULE_EVENT_GROUP_SCROLL_X_WEIGHT)] = scroll_x.x;
        RuleSummary[SbsRuleEventGroupIndex(
            group_index,
            SBS_RULE_EVENT_GROUP_SCROLL_X_SIGNED)] = scroll_x.y;
        RuleSummary[SbsRuleEventGroupIndex(
            group_index,
            SBS_RULE_EVENT_GROUP_SCROLL_X_SUPPORT)] = scroll_x.z;
        RuleSummary[SbsRuleEventGroupIndex(
            group_index,
            SBS_RULE_EVENT_GROUP_FRESH_ACTIVITY_MASS)] =
            fresh_activity.x;
        RuleSummary[SbsRuleEventGroupIndex(
            group_index,
            SBS_RULE_EVENT_GROUP_FRESH_ACTIVITY_INSIDE_PREVIOUS_TARGET)] =
            fresh_activity.y;
        if (group_index == 0u) {
            RuleSummary[SBS_RULE_EVENT_SOURCE_MARKER] =
                SBS_RULE_EVENT_SOURCE_PREAGGREGATED;
        }
    }

    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_UNKNOWN_BACKGROUND, cell)] =
        viewport_valid ? saturate(1.0f - max(focus, ui_like)) : 1.0f;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_PRIMARY_PLAYING_VIDEO, cell)] =
        video_evidence;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_PHOTOGRAPHIC_CONTENT_COLLAGE, cell)] =
        static_media_inputs_valid ? static_media_weight : -1.0f;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_BROWSER_SYSTEM_CHROME, cell)] = ui_like;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_ADVERTISEMENT_COMPETITOR_UNSAFE, cell)] =
        transient_competitor;
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
        SBS_SCENE_DENSE_OUT_LOCAL_MOTION_X, cell)] =
        0.0f;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_LOCAL_MOTION_Y, cell)] =
        0.0f;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_LOCAL_MOTION_CONFIDENCE, cell)] =
        0.0f;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_STRUCTURAL_CONTENT_CUT_SUPPORT, cell)] =
        structural_cut_support;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_EXPOSURE_ONLY_CHANGE_SUPPORT, cell)] =
        exposure_only_support;
    DenseOutput[SceneAnalysisIndex(
        SBS_SCENE_DENSE_OUT_SCROLL_SUPPORT, cell)] = 0.0f;

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
        SBS_SCENE_LAYOUT_HISTORY_HORIZONTAL_MOTION, cell)] =
        0.0f;
    NextLayoutHistory[SceneAnalysisIndex(
        SBS_SCENE_LAYOUT_HISTORY_VERTICAL_MOTION, cell)] =
        0.0f;
    NextLayoutHistory[SceneAnalysisIndex(
        SBS_SCENE_LAYOUT_HISTORY_MOTION_CONFIDENCE, cell)] =
        0.0f;
    NextLayoutHistory[SceneAnalysisIndex(
        SBS_SCENE_LAYOUT_HISTORY_STABLE_OCCUPANCY_DWELL, cell)] = dwell;
    NextLayoutHistory[SceneAnalysisIndex(
        SBS_SCENE_LAYOUT_HISTORY_ACCEPTED_FOCUS_EVIDENCE_EMA, cell)] =
        semantic_layout_history_valid ?
            lerp(
                LayoutHistory(
                    SBS_SCENE_LAYOUT_HISTORY_ACCEPTED_FOCUS_EVIDENCE_EMA,
                    cell),
                focus,
                slow_alpha) :
            focus;
    NextLayoutHistory[SceneAnalysisIndex(
        SBS_SCENE_LAYOUT_HISTORY_EXCLUSION_UI_AD_EVIDENCE_EMA, cell)] =
        semantic_layout_history_valid ?
            lerp(
                LayoutHistory(
                    SBS_SCENE_LAYOUT_HISTORY_EXCLUSION_UI_AD_EVIDENCE_EMA,
                    cell),
                max(ui_like, transient_competitor),
                slow_alpha) :
            max(ui_like, transient_competitor);
    NextLayoutHistory[SceneAnalysisIndex(
        SBS_SCENE_LAYOUT_HISTORY_SECONDS_SINCE_VALID_LAYOUT, cell)] =
        viewport_valid ? 0.0f :
            semantic_layout_history_valid ?
                min(
                    LayoutHistory(
                        SBS_SCENE_LAYOUT_HISTORY_SECONDS_SINCE_VALID_LAYOUT,
                        cell) + max(scene_elapsed_seconds, 0.0f),
                    10.0f) :
                0.0f;

    NextDepthHistory[SceneAnalysisIndex(
        SBS_SCENE_DEPTH_HISTORY_LAST_NORMALIZED_DEPTH, cell)] =
        depth_valid ? depth : previous_depth;
    NextDepthHistory[SceneAnalysisIndex(
        SBS_SCENE_DEPTH_HISTORY_FAST_DEPTH_EMA, cell)] = depth_fast;
    NextDepthHistory[SceneAnalysisIndex(
        SBS_SCENE_DEPTH_HISTORY_SLOW_DEPTH_EMA, cell)] = depth_slow;
    NextDepthHistory[SceneAnalysisIndex(
        SBS_SCENE_DEPTH_HISTORY_SETTLED_SHOT_REFERENCE_DEPTH, cell)] =
        depth_history_comparable ?
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
            (depth_history_comparable ? lerp(
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
            (depth_history_comparable ? lerp(
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
    // Viewport coordinates are consumed by target containment and scroll attribution but are not
    // reproduced in DenseOutput. Fold them into the same fail-closed sentinel so the production
    // reducer and the serial oracle reject identical malformed inputs.
    bool outputs_finite = all(isfinite(viewport_uv));
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
        WriteMeta(
            frame_state > 0.5f &&
                frame_depth_transform_valid,
            frame_depth_transform_valid &&
                !explicit_legacy_transform &&
                FrameRoiDataResetDebt(frame_roi_transform));
    }
}
