StructuredBuffer<float4> AdaptiveState : register(t4);
StructuredBuffer<uint4> PreviousRuleState : register(t6);
RWStructuredBuffer<float> RuleSummary : register(u0);
RWStructuredBuffer<float> EvidenceGlobal : register(u1);

#include "include/sbs_scene_controller_constants.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"
#include "include/sbs_scene_rules_summary.hlsl"

float PreviousRuleStateWord(uint word) {
    const uint4 value = PreviousRuleState[word / 4u];
    return asfloat(value[word & 3u]);
}

uint PreviousRuleStateUint(uint word) {
    const uint4 value = PreviousRuleState[word / 4u];
    return value[word & 3u];
}

uint EffectiveResetFlags() {
    uint flags = scene_reset_flags;
    if (PreviousRuleStateWord(
            SBS_SCENE_RULE_STATE_WORD_OUTPUT_VALID) <= 0.5f) {
        flags |= PreviousRuleStateUint(
            SBS_SCENE_RULE_STATE_WORD_RESET_FLAGS);
    }
    return flags;
}

bool ValidBounds(float4 bounds) {
    return all(bounds.zw > bounds.xy);
}

float BoundsArea(float4 bounds) {
    const float2 extent = max(bounds.zw - bounds.xy, 0.0f.xx);
    return extent.x * extent.y;
}

void ConsiderCandidate(
    float score,
    float4 bounds,
    uint layout,
    float coverage,
    inout float best_score,
    inout float second_score,
    inout float4 best_bounds,
    inout uint best_layout,
    inout float best_coverage)
{
    if (score <= 0.0f || !ValidBounds(bounds)) {
        return;
    }
    if (score > best_score) {
        second_score = best_score;
        best_score = score;
        best_bounds = bounds;
        best_layout = layout;
        best_coverage = coverage;
    } else if (score > second_score) {
        second_score = score;
    }
}

[numthreads(1, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    [unroll]
    for (uint word = 0u;
         word < SBS_SCENE_GLOBAL_OUT_WORD_COUNT;
         ++word) {
        EvidenceGlobal[word] = 0.0f;
    }

    const bool input_valid =
        RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_INPUT_VALID)] > 0.5f;
    if (!input_valid) {
        EvidenceGlobal[
            SBS_SCENE_GLOBAL_OUT_ZERO_PLANE_NEUTRAL_ABSTAIN] =
            SceneLogit(0.98f);
        EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_OOD] = 1.0f;
        EvidenceGlobal[
            SBS_SCENE_GLOBAL_OUT_BACKEND_OUTPUT_VALID] = 0.0f;
        return;
    }

    float best_score = 0.0f;
    float second_score = 0.0f;
    float4 best_bounds = float4(0.0f, 0.0f, 1.0f, 1.0f);
    uint best_layout = SBS_SCENE_LAYOUT_DECISION_NO_TARGET;
    float best_coverage = 0.0f;
    float best_video_score = 0.0f;
    float second_video_score = 0.0f;
    bool video_segmentation_ambiguous =
        RuleSummary[
            SbsRulePlanIndex(SBS_RULE_PLAN_VIDEO_AMBIGUOUS)] > 0.5f;
    const uint region_count = SbsRuleLoadUint(
        RuleSummary,
        SbsRulePlanIndex(SBS_RULE_PLAN_REGION_COUNT));
    [unroll]
    for (uint region = 0u; region < 2u; ++region) {
        if (region >= region_count) {
            continue;
        }
        const uint candidate_first = SbsRuleLoadUint(
            RuleSummary,
            SbsRuleRegionIndex(
                region,
                SBS_RULE_REGION_CANDIDATE_FIRST));
        const uint candidate_count = SbsRuleLoadUint(
            RuleSummary,
            SbsRuleRegionIndex(
                region,
                SBS_RULE_REGION_CANDIDATE_COUNT));
        [loop]
        for (uint index = 0u; index < candidate_count; ++index) {
            const uint candidate = candidate_first + index;
            const float score = RuleSummary[
                SbsRuleCandidateResultIndex(
                    candidate,
                    SBS_RULE_CANDIDATE_SCORE)];
            const float4 bounds = float4(
                RuleSummary[SbsRuleCandidateResultIndex(
                    candidate,
                    SBS_RULE_CANDIDATE_MIN_X)],
                RuleSummary[SbsRuleCandidateResultIndex(
                    candidate,
                    SBS_RULE_CANDIDATE_MIN_Y)],
                RuleSummary[SbsRuleCandidateResultIndex(
                    candidate,
                    SBS_RULE_CANDIDATE_MAX_X)],
                RuleSummary[SbsRuleCandidateResultIndex(
                    candidate,
                    SBS_RULE_CANDIDATE_MAX_Y)]);
            const float coverage = RuleSummary[
                SbsRuleCandidateResultIndex(
                    candidate,
                    SBS_RULE_CANDIDATE_COVERAGE)];
            if (score > best_video_score) {
                second_video_score = best_video_score;
                best_video_score = score;
            } else if (score > second_video_score) {
                second_video_score = score;
            }
            ConsiderCandidate(
                score,
                bounds,
                SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO,
                coverage,
                best_score,
                second_score,
                best_bounds,
                best_layout,
                best_coverage);
        }

        const float content_score = RuleSummary[
            SbsRuleRegionIndex(
                region,
                SBS_RULE_REGION_CONTENT_SCORE)];
        const float4 content_bounds = float4(
            RuleSummary[SbsRuleRegionIndex(
                region,
                SBS_RULE_REGION_CONTENT_MIN_X)],
            RuleSummary[SbsRuleRegionIndex(
                region,
                SBS_RULE_REGION_CONTENT_MIN_Y)],
            RuleSummary[SbsRuleRegionIndex(
                region,
                SBS_RULE_REGION_CONTENT_MAX_X)],
            RuleSummary[SbsRuleRegionIndex(
                region,
                SBS_RULE_REGION_CONTENT_MAX_Y)]);
        const float content_coverage = RuleSummary[
            SbsRuleRegionIndex(
                region,
                SBS_RULE_REGION_CONTENT_COVERAGE)];
        ConsiderCandidate(
            content_score,
            content_bounds,
            SBS_SCENE_LAYOUT_DECISION_CONTENT_COLLAGE,
            content_coverage,
            best_score,
            second_score,
            best_bounds,
            best_layout,
            best_coverage);
    }

    const float winner_ratio =
        second_score > 1e-6f ? best_score / second_score : 99.0f;
    const float winner_margin =
        best_score > 1e-6f ?
            saturate((best_score - second_score) / best_score) :
            0.0f;
    const bool ambiguous =
        best_score > 0.0f &&
        (video_segmentation_ambiguous ||
         (second_video_score > 0.0f &&
          best_video_score / second_video_score < 1.25f) ||
         (second_score > 0.0f && winner_ratio < 1.25f));
    const bool insufficient =
        best_score <= 0.0f || best_coverage < 0.02f;
    const float roi_confidence =
        ambiguous || insufficient ?
            0.0f :
            saturate(0.45f + 0.35f * winner_margin +
                     0.20f * min(best_score * 3.0f, 1.0f));
    const float mask_confidence =
        saturate(0.5f * best_coverage + 0.5f * winner_margin);
    const float ood =
        ambiguous ? 0.75f :
        insufficient ? 0.60f :
        saturate(0.35f - winner_margin);
    const bool identity_target =
        BoundsArea(best_bounds) >= 0.90f &&
        best_bounds.x <= 0.05f && best_bounds.y <= 0.05f &&
        best_bounds.z >= 0.95f && best_bounds.w >= 0.95f;
    if (identity_target && !ambiguous && !insufficient) {
        best_layout = SBS_SCENE_LAYOUT_DECISION_IDENTITY_FULLSCREEN;
        best_bounds = float4(0.0f, 0.0f, 1.0f, 1.0f);
    }

    const float valid_cells =
        RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_VALID_CELLS)];
    const float scroll_weight =
        RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_SCROLL_WEIGHT)];
    const float scroll_signed =
        RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_SCROLL_SIGNED)];
    const float scroll_support_cells =
        RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_SCROLL_SUPPORT)];
    const float scroll_agreement =
        scroll_weight > 1e-6f ?
            abs(scroll_signed) / scroll_weight :
            0.0f;
    const float scroll_fraction =
        valid_cells > 0.0f ?
            scroll_support_cells / valid_cells :
            0.0f;
    const float scroll_confidence =
        scroll_fraction >= 0.24f && scroll_agreement >= 0.82f ?
            saturate(scroll_fraction * scroll_agreement) :
            0.0f;
    const float scroll_y =
        scroll_weight > 1e-6f ?
            clamp(scroll_signed / scroll_weight, -1.0f, 1.0f) :
            0.0f;

    const float no_target_probability = saturate(1.0f - roi_confidence);
    const float video_probability =
        best_layout == SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO ?
            roi_confidence :
            0.02f;
    const float content_probability =
        best_layout == SBS_SCENE_LAYOUT_DECISION_CONTENT_COLLAGE ?
            roi_confidence :
            0.02f;
    const float identity_probability =
        identity_target ?
            roi_confidence :
            0.02f;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_LAYOUT_NO_TARGET] =
        SceneLogit(max(no_target_probability, 0.02f));
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_LAYOUT_PRIMARY_VIDEO] =
        SceneLogit(max(video_probability, 0.02f));
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_LAYOUT_CONTENT_COLLAGE] =
        SceneLogit(max(content_probability, 0.02f));
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_LAYOUT_IDENTITY_FULLSCREEN] =
        SceneLogit(max(identity_probability, 0.02f));
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_LAYOUT_AMBIGUOUS] =
        SceneLogit(ambiguous ? 0.90f : 0.05f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_LAYOUT_SCROLL] =
        SceneLogit(max(scroll_confidence, 0.02f));

    const float4 adaptive_flags =
        AdaptiveState[SBS_STATE_VECTOR_RANGE_COLLAPSED];
    const float4 adaptive_analysis =
        AdaptiveState[SBS_STATE_VECTOR_ANALYSIS_FLAGS];
    const uint analysis_flags =
        (uint)SBS_STATE_ANALYSIS_FLAGS(adaptive_analysis);
    const bool geometry_reset =
        (EffectiveResetFlags() & SBS_SCENE_RESET_FLAGS_GEOMETRY) != 0u;
    const bool hard_cut =
        SBS_STATE_HARD_CUT_PULSE(adaptive_flags) > 0.5f &&
        !geometry_reset;
    const bool exposure =
        (analysis_flags & ANALYSIS_FLAG_EXPOSURE_LIKE) != 0u;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_EVENT_SAME_SHOT] =
        SceneLogit(!hard_cut && !exposure ? 0.95f : 0.05f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_EVENT_HARD_CUT] =
        SceneLogit(hard_cut ? 0.98f : 0.02f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_EVENT_FADE_OR_DISSOLVE] =
        SceneLogit(0.02f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_EVENT_FLASH_OR_EXPOSURE] =
        SceneLogit(exposure ? 0.95f : 0.02f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_EVENT_SCROLL] =
        SceneLogit(max(scroll_confidence, 0.02f));
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_EVENT_GEOMETRY_RESET] =
        SceneLogit(geometry_reset ? 0.995f : 0.02f);

    const float4 adaptive_profile =
        AdaptiveState[SBS_STATE_VECTOR_ADAPTIVE_POP_RATIO];
    const float4 adaptive_valid_depth =
        AdaptiveState[SBS_STATE_VECTOR_VALID_DEPTH_FRACTION];
    const float depth_coverage_cells =
        RuleSummary[SbsRulePlanIndex(SBS_RULE_PLAN_DEPTH_COVERAGE)];
    const float history_depth_fraction =
        depth_coverage_cells > 0.0f ?
            saturate(
                RuleSummary[
                    SbsRulePlanIndex(SBS_RULE_PLAN_DEPTH_CONFIDENCE)] /
                depth_coverage_cells) :
            0.0f;
    const float valid_depth_fraction = min(
        saturate(SBS_STATE_VALID_DEPTH_FRACTION(adaptive_valid_depth)),
        history_depth_fraction);
    const float depth_reliability = smoothstep(
        0.05f,
        0.35f,
        valid_depth_fraction);
    const float resolved_strength = clamp(
        scene_pop_floor *
            max(SBS_STATE_ADAPTIVE_POP_RATIO(adaptive_profile), 1.0f),
        scene_pop_floor,
        max(scene_pop_ceiling, scene_pop_floor));
    [unroll]
    for (uint action = 0u; action < SBS_SCENE_POP_ACTION_COUNT; ++action) {
        const float action_strength = lerp(
            scene_pop_floor,
            scene_pop_ceiling,
            (float)action / (float)(SBS_SCENE_POP_ACTION_COUNT - 1u));
        EvidenceGlobal[
            SBS_SCENE_GLOBAL_OUT_POP_ACTION_SAFE_0 + action] =
            SceneLogit(
                action_strength <= resolved_strength + 1e-5f ?
                    0.90f :
                    0.10f);
    }

    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_ZERO_PLANE_SUBJECT] =
        SceneLogit(scene_zero_plane_mode < 1.5f ? 0.95f : 0.02f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_ZERO_PLANE_MEDIAN] =
        SceneLogit(
            abs(scene_zero_plane_mode - 2.0f) < 0.5f ?
                0.95f :
                0.02f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_ZERO_PLANE_BACKGROUND] =
        SceneLogit(scene_zero_plane_mode > 2.5f ? 0.95f : 0.02f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_ZERO_PLANE_NEUTRAL_ABSTAIN] =
        SceneLogit(0.02f);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_ROI] = roi_confidence;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_MASK] = mask_confidence;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_EVENT] =
        lerp(0.25f, 0.90f, depth_reliability);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_POP] =
        lerp(0.25f, 0.90f, depth_reliability);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_ZERO_PLANE] =
        lerp(0.25f, 0.90f, depth_reliability);
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_CONFIDENCE_OOD] = ood;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_GLOBAL_SCROLL_X] = 0.0f;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_GLOBAL_SCROLL_Y] = scroll_y;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_GLOBAL_SCROLL_CONFIDENCE] =
        scroll_confidence;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_BACKEND_OUTPUT_VALID] = 1.0f;

    // Internal resolver scratch. The common resolver clears these words from the public output.
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_35] = best_bounds.x;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_36] = best_bounds.y;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_37] = best_bounds.z;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_38] = best_bounds.w;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_39] = second_score;
    EvidenceGlobal[SBS_SCENE_GLOBAL_OUT_RESERVED_40] =
        history_depth_fraction;
}
