#ifndef SBS_SCENE_RULE_STATE_HLSL
#define SBS_SCENE_RULE_STATE_HLSL

// Shared access and validity rules for the previous scene-controller state. Every consumer
// declares the structured buffer as `PreviousRuleState` before including this file.
float SbsRulePreviousStateWord(uint word) {
    const uint4 value = PreviousRuleState[word / 4u];
    return asfloat(value[word & 3u]);
}

uint SbsRulePreviousStateUint(uint word) {
    const uint4 value = PreviousRuleState[word / 4u];
    return value[word & 3u];
}

bool SbsRulePreviousWordIsWellFormedBool(uint word) {
    const float value = SbsRulePreviousStateWord(word);
    // Zero is a legitimate held/invalid output. A positive flag must be unambiguously true;
    // fractional false values are malformed state, not a second spelling of false.
    return
        isfinite(value) &&
        (
            value == 0.0f ||
            (value > 0.5f && value <= 1.0f)
        );
}

bool SbsRulePreviousStateIdentityValid() {
    const float schema =
        SbsRulePreviousStateWord(SBS_SCENE_RULE_STATE_WORD_SCHEMA_VERSION);
    const uint backend_generation = SbsRulePreviousStateUint(
        SBS_SCENE_RULE_STATE_WORD_BACKEND_GENERATION);
    return
        isfinite(schema) &&
        schema == (float)SBS_SCENE_SCHEMA_VERSION &&
        (backend_generation == 0u ||
         backend_generation == scene_backend_generation) &&
        SbsRulePreviousWordIsWellFormedBool(
            SBS_SCENE_RULE_STATE_WORD_OUTPUT_VALID) &&
        (SbsRulePreviousStateUint(
             SBS_SCENE_RULE_STATE_WORD_STATE_FLAGS) &
         SBS_SCENE_STATE_FLAGS_INITIALIZED) != 0u;
}

uint SbsRuleEffectiveResetFlags() {
    uint flags = scene_reset_flags;
    const float previous_output_valid = SbsRulePreviousStateWord(
        SBS_SCENE_RULE_STATE_WORD_OUTPUT_VALID);
    // Invalid output retains its reset debt. Treat NaN/Inf as invalid rather than allowing a
    // malformed state to bypass the reset gate through an unordered floating-point comparison.
    if (
        !isfinite(previous_output_valid) ||
        previous_output_valid <= 0.5f ||
        previous_output_valid > 1.0f
    ) {
        flags |= SbsRulePreviousStateUint(
            SBS_SCENE_RULE_STATE_WORD_RESET_FLAGS);
    }
    return flags;
}

bool SbsRulePreviousTargetsUsable() {
    const uint target_reset_mask =
        SBS_SCENE_RESET_FLAGS_LAYOUT |
        SBS_SCENE_RESET_FLAGS_GEOMETRY |
        SBS_SCENE_RESET_FLAGS_BACKEND |
        SBS_SCENE_RESET_FLAGS_DISPLAY_OR_HDR;
    return
        SbsRulePreviousStateIdentityValid() &&
        (SbsRuleEffectiveResetFlags() & target_reset_mask) == 0u;
}

// Counter epochs are independent of geometry/layout reset debt. Keeping this predicate separate
// prevents a target reset from losing an external/detector cut that still needs to be consumed.
bool SbsRulePreviousCounterStateValid() {
    return SbsRulePreviousStateIdentityValid();
}

static const float
    SBS_RULE_CONTENT_PROMOTION_DROPOUT_TOLERANCE_SECONDS = 0.75f;

bool SbsRulePreviousWordIsExactUint(uint word, uint expected) {
    const float value = SbsRulePreviousStateWord(word);
    return
        isfinite(value) &&
        value == (float)expected;
}

bool SbsRulePreviousWordIsCanonicalZero(uint word) {
    return SbsRulePreviousStateUint(word) == 0u;
}

bool SbsRulePreviousWordIsTrueFlag(uint word) {
    const float value = SbsRulePreviousStateWord(word);
    return
        isfinite(value) &&
        value > 0.5f &&
        value <= 1.0f;
}

bool SbsRulePreviousWordIsNormalizedPositive(uint word) {
    const float value = SbsRulePreviousStateWord(word);
    return
        isfinite(value) &&
        value > 0.0f &&
        value <= 1.0f;
}

bool SbsRulePreviousWordIsNormalized(uint word) {
    const float value = SbsRulePreviousStateWord(word);
    return
        isfinite(value) &&
        value >= 0.0f &&
        value <= 1.0f;
}

bool SbsRulePreviousWordIsNonnegativeFinite(uint word) {
    const float value = SbsRulePreviousStateWord(word);
    return isfinite(value) && value >= 0.0f;
}

bool SbsRulePreviousRawStateKind(out uint state_kind) {
    state_kind = SBS_SCENE_STATE_KIND_FULL_FRAME;
    const float value = SbsRulePreviousStateWord(
        SBS_SCENE_RULE_STATE_WORD_STATE_KIND);
    if (
        !SbsRulePreviousStateIdentityValid() ||
        !isfinite(value) ||
        (
            value != (float)SBS_SCENE_STATE_KIND_FULL_FRAME &&
            value != (float)SBS_SCENE_STATE_KIND_VIDEO &&
            value != (float)SBS_SCENE_STATE_KIND_CONTENT
        )
    ) {
        return false;
    }
    state_kind = (uint)value;
    return true;
}

bool SbsRulePreviousOutputValid() {
    return SbsRulePreviousWordIsTrueFlag(
        SBS_SCENE_RULE_STATE_WORD_OUTPUT_VALID);
}

bool SbsRuleValidateUnitBounds(
    float4 raw_bounds,
    out float4 bounds)
{
    bounds = 0.0f.xxxx;
    if (
        !all(isfinite(raw_bounds)) ||
        !all(raw_bounds >= 0.0f.xxxx) ||
        !all(raw_bounds <= 1.0f.xxxx) ||
        !all(raw_bounds.zw > raw_bounds.xy)
    ) {
        return false;
    }
    bounds = raw_bounds;
    return true;
}

bool SbsRuleReadPreviousBounds(
    uint word_x0,
    uint word_y0,
    uint word_x1,
    uint word_y1,
    out float4 bounds)
{
    const float4 raw_bounds = float4(
        SbsRulePreviousStateWord(word_x0),
        SbsRulePreviousStateWord(word_y0),
        SbsRulePreviousStateWord(word_x1),
        SbsRulePreviousStateWord(word_y1));
    return SbsRuleValidateUnitBounds(raw_bounds, bounds);
}

bool SbsRulePreviousAcquisitionTarget(
    out uint layout,
    out float4 bounds)
{
    layout = SBS_SCENE_LAYOUT_DECISION_NO_TARGET;
    bounds = 0.0f.xxxx;
    const float layout_word = SbsRulePreviousStateWord(
        SBS_SCENE_RULE_STATE_WORD_ACQUISITION_LAYOUT);
    const bool layout_valid =
        isfinite(layout_word) &&
        (
            layout_word ==
                (float)SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO ||
            layout_word ==
                (float)SBS_SCENE_LAYOUT_DECISION_CONTENT_COLLAGE
        );
    const float cadence_gap = SbsRulePreviousStateWord(
        SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE);
    const bool cadence_gap_valid =
        isfinite(cadence_gap) &&
        cadence_gap >= 0.0f &&
        cadence_gap <= SBS_SCENE_TEMPORAL_CADENCE_GAP_SECONDS &&
        (
            layout_word ==
                (float)SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO ||
            cadence_gap == 0.0f
        );
    if (
        !SbsRulePreviousTargetsUsable() ||
        !SbsRulePreviousWordIsExactUint(
            SBS_SCENE_RULE_STATE_WORD_STATE_KIND,
            SBS_SCENE_STATE_KIND_FULL_FRAME) ||
        !SbsRulePreviousWordIsTrueFlag(
            SBS_SCENE_RULE_STATE_WORD_ACQUISITION_VALID) ||
        !layout_valid ||
        !SbsRulePreviousWordIsCanonicalZero(
            SBS_SCENE_RULE_STATE_WORD_ACQUISITION_SCORE) ||
        !SbsRulePreviousWordIsNonnegativeFinite(
            SBS_SCENE_RULE_STATE_WORD_ACQUISITION_DWELL_S) ||
        !cadence_gap_valid
    ) {
        return false;
    }
    if (!SbsRuleReadPreviousBounds(
        SBS_SCENE_RULE_STATE_WORD_ACQUISITION_ROI_X0,
        SBS_SCENE_RULE_STATE_WORD_ACQUISITION_ROI_Y0,
        SBS_SCENE_RULE_STATE_WORD_ACQUISITION_ROI_X1,
        SBS_SCENE_RULE_STATE_WORD_ACQUISITION_ROI_Y1,
        bounds)) {
        return false;
    }
    layout = (uint)layout_word;
    return true;
}

bool SbsRulePreviousProvisionalVideoTarget(out float4 bounds) {
    uint layout;
    return
        SbsRulePreviousAcquisitionTarget(layout, bounds) &&
        layout == SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO;
}

bool SbsRulePreviousChallengerTarget(
    out uint layout,
    out float4 bounds)
{
    layout = SBS_SCENE_LAYOUT_DECISION_NO_TARGET;
    bounds = 0.0f.xxxx;
    const float state_kind_word = SbsRulePreviousStateWord(
        SBS_SCENE_RULE_STATE_WORD_STATE_KIND);
    const bool state_kind_valid =
        isfinite(state_kind_word) &&
        (
            state_kind_word == (float)SBS_SCENE_STATE_KIND_VIDEO ||
            state_kind_word == (float)SBS_SCENE_STATE_KIND_CONTENT
        );
    const float layout_word = SbsRulePreviousStateWord(
        SBS_SCENE_RULE_STATE_WORD_CHALLENGER_LAYOUT);
    const bool layout_valid =
        isfinite(layout_word) &&
        (
            layout_word ==
                (float)SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO ||
            layout_word ==
                (float)SBS_SCENE_LAYOUT_DECISION_CONTENT_COLLAGE ||
            layout_word ==
                (float)SBS_SCENE_LAYOUT_DECISION_IDENTITY_FULLSCREEN
        );
    const float cadence_gap = SbsRulePreviousStateWord(
        SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE);
    const bool cadence_gap_valid =
        isfinite(cadence_gap) &&
        cadence_gap >= 0.0f &&
        cadence_gap <= SBS_SCENE_TEMPORAL_CADENCE_GAP_SECONDS &&
        (
            layout_word ==
                (float)SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO ||
            cadence_gap == 0.0f
        );
    if (
        !SbsRulePreviousTargetsUsable() ||
        !state_kind_valid ||
        !SbsRulePreviousWordIsTrueFlag(
            SBS_SCENE_RULE_STATE_WORD_CHALLENGER_VALID) ||
        !layout_valid ||
        !SbsRulePreviousWordIsCanonicalZero(
            SBS_SCENE_RULE_STATE_WORD_CHALLENGER_SCORE) ||
        !SbsRulePreviousWordIsNonnegativeFinite(
            SBS_SCENE_RULE_STATE_WORD_CHALLENGER_DWELL_S) ||
        !cadence_gap_valid
    ) {
        return false;
    }
    if (!SbsRuleReadPreviousBounds(
        SBS_SCENE_RULE_STATE_WORD_CHALLENGER_ROI_X0,
        SBS_SCENE_RULE_STATE_WORD_CHALLENGER_ROI_Y0,
        SBS_SCENE_RULE_STATE_WORD_CHALLENGER_ROI_X1,
        SBS_SCENE_RULE_STATE_WORD_CHALLENGER_ROI_Y1,
        bounds)) {
        return false;
    }
    layout = (uint)layout_word;
    return true;
}

bool SbsRulePreviousVideoChallengerTarget(out float4 bounds) {
    uint layout;
    return
        SbsRulePreviousChallengerTarget(layout, bounds) &&
        layout == SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO;
}

bool SbsRulePreviousCommittedTarget(
    out uint state_kind,
    out uint layout,
    out float4 bounds,
    out float confidence)
{
    state_kind = SBS_SCENE_STATE_KIND_FULL_FRAME;
    layout = SBS_SCENE_LAYOUT_DECISION_NO_TARGET;
    bounds = 0.0f.xxxx;
    confidence = 0.0f;
    uint raw_state_kind;
    const bool raw_state_kind_valid =
        SbsRulePreviousRawStateKind(raw_state_kind);
    const float layout_word = SbsRulePreviousStateWord(
        SBS_SCENE_RULE_STATE_WORD_COMMITTED_LAYOUT);
    const bool layout_valid =
        isfinite(layout_word) &&
        (
            layout_word ==
                (float)SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO ||
            layout_word ==
                (float)SBS_SCENE_LAYOUT_DECISION_CONTENT_COLLAGE
        );
    const uint resolved_state_kind =
        layout_word ==
            (float)SBS_SCENE_LAYOUT_DECISION_PRIMARY_VIDEO ?
                SBS_SCENE_STATE_KIND_VIDEO :
                SBS_SCENE_STATE_KIND_CONTENT;
    const bool role_matches = raw_state_kind == resolved_state_kind;
    if (
        !SbsRulePreviousTargetsUsable() ||
        !raw_state_kind_valid ||
        (
            raw_state_kind != SBS_SCENE_STATE_KIND_VIDEO &&
            raw_state_kind != SBS_SCENE_STATE_KIND_CONTENT
        ) ||
        !layout_valid ||
        !role_matches ||
        (SbsRulePreviousStateUint(
             SBS_SCENE_RULE_STATE_WORD_STATE_FLAGS) &
         SBS_SCENE_STATE_FLAGS_ROI_LOCKED) == 0u ||
        !SbsRulePreviousWordIsNormalizedPositive(
            SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_CONFIDENCE) ||
        !SbsRulePreviousWordIsNormalized(
            SBS_SCENE_RULE_STATE_WORD_COMMITTED_MASK_CONFIDENCE) ||
        !SbsRulePreviousWordIsNonnegativeFinite(
            SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_AGE_S) ||
        !SbsRulePreviousWordIsNonnegativeFinite(
            SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE)
    ) {
        return false;
    }
    if (!SbsRuleReadPreviousBounds(
        SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_X0,
        SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_Y0,
        SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_X1,
        SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_Y1,
        bounds)) {
        return false;
    }
    state_kind = resolved_state_kind;
    layout = (uint)layout_word;
    confidence = SbsRulePreviousStateWord(
        SBS_SCENE_RULE_STATE_WORD_COMMITTED_ROI_CONFIDENCE);
    return true;
}

bool SbsRulePreviousResolvedGeometryState(
    out uint state_kind,
    out float4 committed_bounds,
    out float committed_confidence)
{
    state_kind = SBS_SCENE_STATE_KIND_FULL_FRAME;
    committed_bounds = float4(0.0f, 0.0f, 1.0f, 1.0f);
    committed_confidence = 0.0f;
    uint raw_state_kind;
    if (!SbsRulePreviousRawStateKind(raw_state_kind)) {
        return false;
    }
    if (raw_state_kind == SBS_SCENE_STATE_KIND_FULL_FRAME) {
        return true;
    }
    uint committed_layout;
    return SbsRulePreviousCommittedTarget(
        state_kind,
        committed_layout,
        committed_bounds,
        committed_confidence);
}

bool SbsRulePreviousCommittedVideoTarget(out float4 bounds) {
    uint state_kind;
    uint layout;
    float confidence;
    return
        SbsRulePreviousCommittedTarget(
            state_kind,
            layout,
            bounds,
            confidence) &&
        state_kind == SBS_SCENE_STATE_KIND_VIDEO;
}

bool SbsRulePreviousFreshCommittedContentTarget(
    out float4 bounds)
{
    bounds = 0.0f.xxxx;
    const float dropout_seconds = SbsRulePreviousStateWord(
        SBS_SCENE_RULE_STATE_WORD_SECONDS_SINCE_LAYOUT_EVIDENCE);
    uint state_kind;
    uint layout;
    float confidence;
    if (
        !SbsRulePreviousCommittedTarget(
            state_kind,
            layout,
            bounds,
            confidence) ||
        state_kind != SBS_SCENE_STATE_KIND_CONTENT ||
        dropout_seconds >
            SBS_RULE_CONTENT_PROMOTION_DROPOUT_TOLERANCE_SECONDS
    ) {
        return false;
    }
    return true;
}

// The scroll evidence mask and its eligible-area denominator must describe the same target.
// This covers a provisional full-frame probe, a committed-state video challenger, or the
// committed player itself. Keeping it here prevents either reducer pass from drifting.
bool SbsRulePreviousVideoTarget(out float4 bounds) {
    bounds = 0.0f.xxxx;
    float4 candidate_bounds;
    if (SbsRulePreviousProvisionalVideoTarget(candidate_bounds)) {
        bounds = candidate_bounds;
        return true;
    }
    if (SbsRulePreviousVideoChallengerTarget(candidate_bounds)) {
        bounds = candidate_bounds;
        return true;
    }
    if (SbsRulePreviousCommittedVideoTarget(candidate_bounds)) {
        bounds = candidate_bounds;
        return true;
    }
    return false;
}

#endif
