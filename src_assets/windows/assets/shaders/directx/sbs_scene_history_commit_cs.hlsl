StructuredBuffer<float> PreviousLayoutHistory : register(t0);
StructuredBuffer<float> PreviousDepthHistory : register(t1);
StructuredBuffer<uint4> ResolvedRuleState : register(t2);

RWStructuredBuffer<float> NextLayoutHistory : register(u0);
RWStructuredBuffer<float> NextDepthHistory : register(u1);

#include "include/sbs_scene_controller_constants.hlsl"

float RuleStateWord(uint word) {
    uint4 value = ResolvedRuleState[word / 4u];
    return asfloat(value[word & 3u]);
}

uint RuleStateUint(uint word) {
    uint4 value = ResolvedRuleState[word / 4u];
    return value[word & 3u];
}

[numthreads(16, 16, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    const uint canvas = SBS_SCENE_ANALYSIS_CANVAS_SIZE;
    if (dispatch_id.x >= canvas || dispatch_id.y >= canvas) {
        return;
    }

    uint2 cell = dispatch_id.xy;
    bool output_valid =
        RuleStateWord(SBS_SCENE_RULE_STATE_WORD_OUTPUT_VALID) > 0.5f;
    uint promotion_flags =
        RuleStateUint(SBS_SCENE_RULE_STATE_WORD_PROMOTION_FLAGS);
    uint reset_flags =
        RuleStateUint(SBS_SCENE_RULE_STATE_WORD_RESET_FLAGS);
    bool scroll_hold =
        output_valid &&
        abs(
            RuleStateWord(SBS_SCENE_RULE_STATE_WORD_STATE_KIND) -
            (float)SBS_SCENE_STATE_KIND_SCROLL_HOLD) < 0.5f;
    bool layout_promote =
        output_valid &&
        (promotion_flags & SBS_SCENE_PROMOTION_FLAGS_LAYOUT_HISTORY) != 0u;
    bool depth_promote =
        output_valid &&
        (promotion_flags & SBS_SCENE_PROMOTION_FLAGS_DEPTH_HISTORY) != 0u;
    bool shot_reset =
        (promotion_flags & SBS_SCENE_PROMOTION_FLAGS_SHOT) != 0u ||
        (reset_flags & SBS_SCENE_RESET_FLAGS_DEPTH_SHOT) != 0u;
    bool roi_reset =
        (promotion_flags & SBS_SCENE_PROMOTION_FLAGS_ROI) != 0u ||
        (reset_flags & SBS_SCENE_RESET_FLAGS_GEOMETRY) != 0u;

    // A hard content cut does not invalidate stable page/video layout. Display/backend layout
    // resets are different: their freshly prepared candidate is the new canonical reference.
    bool preserve_layout_for_shot =
        (promotion_flags & SBS_SCENE_PROMOTION_FLAGS_SHOT) != 0u &&
        (reset_flags &
            (SBS_SCENE_RESET_FLAGS_LAYOUT |
             SBS_SCENE_RESET_FLAGS_DISPLAY_OR_HDR |
             SBS_SCENE_RESET_FLAGS_BACKEND)) == 0u;
    for (uint channel = 0u;
         channel < SBS_SCENE_LAYOUT_HISTORY_CHANNEL_COUNT;
         ++channel) {
        uint index = SceneAnalysisIndex(channel, cell);
        float next_value = NextLayoutHistory[index];
        if (!layout_promote) {
            next_value = PreviousLayoutHistory[index];
        } else if (scroll_hold) {
            bool advance_for_scroll =
                channel ==
                    SBS_SCENE_LAYOUT_HISTORY_PREVIOUS_LUMINANCE_ORDINAL ||
                channel == SBS_SCENE_LAYOUT_HISTORY_HORIZONTAL_MOTION ||
                channel == SBS_SCENE_LAYOUT_HISTORY_VERTICAL_MOTION ||
                channel == SBS_SCENE_LAYOUT_HISTORY_MOTION_CONFIDENCE;
            if (!advance_for_scroll) {
                next_value = PreviousLayoutHistory[index];
            }
        } else if (preserve_layout_for_shot) {
            if (channel ==
                SBS_SCENE_LAYOUT_HISTORY_PREVIOUS_LUMINANCE_ORDINAL) {
                // Advance the exact appearance endpoint so the next frame compares against the
                // new shot instead of retriggering the same cut.
            } else if (channel ==
                SBS_SCENE_LAYOUT_HISTORY_FAST_ACTIVITY_EMA) {
                next_value = 0.0f;
            } else {
                next_value = PreviousLayoutHistory[index];
            }
        }
        NextLayoutHistory[index] = next_value;
    }

    if (!depth_promote) {
        for (uint channel = 0u;
             channel < SBS_SCENE_DEPTH_HISTORY_CHANNEL_COUNT;
             ++channel) {
            uint index = SceneAnalysisIndex(channel, cell);
            NextDepthHistory[index] = PreviousDepthHistory[index];
        }
        return;
    }

    if (shot_reset || roi_reset) {
        uint last_index = SceneAnalysisIndex(
            SBS_SCENE_DEPTH_HISTORY_LAST_NORMALIZED_DEPTH,
            cell);
        float current_depth = NextDepthHistory[last_index];
        float valid_confidence = NextDepthHistory[SceneAnalysisIndex(
            SBS_SCENE_DEPTH_HISTORY_VALID_DEPTH_CONFIDENCE,
            cell)];
        float seconds_since_valid = NextDepthHistory[SceneAnalysisIndex(
            SBS_SCENE_DEPTH_HISTORY_SECONDS_SINCE_VALID_DEPTH,
            cell)];
        NextDepthHistory[SceneAnalysisIndex(
            SBS_SCENE_DEPTH_HISTORY_FAST_DEPTH_EMA, cell)] =
            current_depth;
        NextDepthHistory[SceneAnalysisIndex(
            SBS_SCENE_DEPTH_HISTORY_SLOW_DEPTH_EMA, cell)] =
            current_depth;
        NextDepthHistory[SceneAnalysisIndex(
            SBS_SCENE_DEPTH_HISTORY_SETTLED_SHOT_REFERENCE_DEPTH,
            cell)] = current_depth;
        NextDepthHistory[SceneAnalysisIndex(
            SBS_SCENE_DEPTH_HISTORY_DEPTH_VARIANCE_EMA,
            cell)] = 0.0f;
        NextDepthHistory[SceneAnalysisIndex(
            SBS_SCENE_DEPTH_HISTORY_ABSOLUTE_DEPTH_CHANGE_EMA,
            cell)] = 0.0f;
        NextDepthHistory[SceneAnalysisIndex(
            SBS_SCENE_DEPTH_HISTORY_VALID_DEPTH_CONFIDENCE,
            cell)] = valid_confidence;
        NextDepthHistory[SceneAnalysisIndex(
            SBS_SCENE_DEPTH_HISTORY_SECONDS_SINCE_VALID_DEPTH,
            cell)] = seconds_since_valid;
    }
}
