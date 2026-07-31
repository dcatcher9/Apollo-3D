#ifndef SBS_SCENE_RULES_SUMMARY_HLSL
#define SBS_SCENE_RULES_SUMMARY_HLSL

// Private fixed-layout scratch shared by the two production rule-reduction passes. This is not
// part of the public controller ABI and never leaves the GPU.
#include "include/sbs_scene_rules_summary_layout.shared.h"

#define SBS_RULE_COLUMN_COUNT SBS_RULE_SUMMARY_COLUMN_COUNT
#define SBS_RULE_COLUMN_STRIDE SBS_RULE_SUMMARY_COLUMN_STRIDE
#define SBS_RULE_COLUMN_BASE SBS_RULE_SUMMARY_COLUMN_BASE
#define SBS_RULE_COLUMN_MEDIA_MASS 0u
#define SBS_RULE_COLUMN_MEDIA_INVALID 1u
#define SBS_RULE_COLUMN_TEMPORAL_MASS 2u
#define SBS_RULE_COLUMN_GUTTER_MASS 3u
#define SBS_RULE_COLUMN_FRESH_ACTIVITY_MASS 4u

#define SBS_RULE_PLAN_BASE SBS_RULE_SUMMARY_PLAN_BASE
#define SBS_RULE_PLAN_INPUT_VALID SBS_RULE_SUMMARY_PLAN_INPUT_VALID
#define SBS_RULE_PLAN_VALID_CELLS SBS_RULE_SUMMARY_PLAN_VALID_CELLS
#define SBS_RULE_PLAN_DEPTH_COVERAGE \
    SBS_RULE_SUMMARY_PLAN_DEPTH_COVERAGE
#define SBS_RULE_PLAN_DEPTH_CONFIDENCE \
    SBS_RULE_SUMMARY_PLAN_DEPTH_CONFIDENCE
#define SBS_RULE_PLAN_ROI_STRUCTURAL_WEIGHT \
    SBS_RULE_SUMMARY_PLAN_ROI_STRUCTURAL_WEIGHT
#define SBS_RULE_PLAN_ROI_EXPOSURE_WEIGHT \
    SBS_RULE_SUMMARY_PLAN_ROI_EXPOSURE_WEIGHT
#define SBS_RULE_PLAN_BEST_SCORE SBS_RULE_SUMMARY_PLAN_BEST_SCORE
#define SBS_RULE_PLAN_SECOND_SCORE SBS_RULE_SUMMARY_PLAN_SECOND_SCORE
#define SBS_RULE_PLAN_WINNER_MARGIN \
    SBS_RULE_SUMMARY_PLAN_WINNER_MARGIN
#define SBS_RULE_PLAN_CANDIDATE_COVERAGE \
    SBS_RULE_SUMMARY_PLAN_CANDIDATE_COVERAGE
#define SBS_RULE_PLAN_GLOBAL_SCROLL_WEIGHT \
    SBS_RULE_SUMMARY_PLAN_GLOBAL_SCROLL_WEIGHT
#define SBS_RULE_PLAN_GLOBAL_SCROLL_SIGNED \
    SBS_RULE_SUMMARY_PLAN_GLOBAL_SCROLL_SIGNED
#define SBS_RULE_PLAN_GLOBAL_SCROLL_SUPPORT \
    SBS_RULE_SUMMARY_PLAN_GLOBAL_SCROLL_SUPPORT
#define SBS_RULE_PLAN_FRESH_ACTIVITY_MASS \
    SBS_RULE_SUMMARY_PLAN_FRESH_ACTIVITY_MASS
#define SBS_RULE_PLAN_FRESH_ACTIVITY_INSIDE_PREVIOUS_TARGET \
    SBS_RULE_SUMMARY_PLAN_FRESH_ACTIVITY_INSIDE_PREVIOUS_TARGET
#define SBS_RULE_PLAN_GLOBAL_HORIZONTAL_SCROLL_WEIGHT \
    SBS_RULE_SUMMARY_PLAN_GLOBAL_HORIZONTAL_SCROLL_WEIGHT
#define SBS_RULE_PLAN_GLOBAL_HORIZONTAL_SCROLL_SIGNED \
    SBS_RULE_SUMMARY_PLAN_GLOBAL_HORIZONTAL_SCROLL_SIGNED
#define SBS_RULE_PLAN_GLOBAL_HORIZONTAL_SCROLL_SUPPORT \
    SBS_RULE_SUMMARY_PLAN_GLOBAL_HORIZONTAL_SCROLL_SUPPORT

#define SBS_RULE_MEDIA_ROW_BASE SBS_RULE_SUMMARY_MEDIA_ROW_BASE
#define SBS_RULE_MEDIA_ROW_COUNT SBS_RULE_SUMMARY_MEDIA_ROW_COUNT

// The evidence shader reduces ROI-local event inputs once while those values are already in
// registers. One float4 per 16x16 evidence group avoids rereading three full 128x128 planes in
// the later column pass.
#define SBS_RULE_TEMPORAL_ROW_BASE SBS_RULE_SUMMARY_TEMPORAL_ROW_BASE
#define SBS_RULE_TEMPORAL_ROW_COUNT SBS_RULE_SUMMARY_TEMPORAL_ROW_COUNT
#define SBS_RULE_TEMPORAL_ROW_STRIDE SBS_RULE_SUMMARY_TEMPORAL_ROW_STRIDE
// Each temporal row stores continuous effective active mass, exact current activity, and the
// stable-gutter confidence projected across the row:
//   active_mass: sum of soft temporal weights in the row
//   fresh_activity_mass: current ordinal-change activity in the row
//   gutter_mass: sum of the geometry-neutral stable-gutter dense lane in the row
#define SBS_RULE_TEMPORAL_ROW_ACTIVE_MASS 0u
#define SBS_RULE_TEMPORAL_ROW_FRESH_ACTIVITY_MASS 1u
#define SBS_RULE_TEMPORAL_ROW_GUTTER_MASS 2u

#define SBS_RULE_EVENT_GROUP_BASE SBS_RULE_SUMMARY_EVENT_GROUP_BASE
#define SBS_RULE_EVENT_GROUP_COUNT SBS_RULE_SUMMARY_EVENT_GROUP_COUNT
#define SBS_RULE_EVENT_GROUP_STRIDE SBS_RULE_SUMMARY_EVENT_GROUP_STRIDE
#define SBS_RULE_EVENT_GROUP_ROI_CELLS 0u
#define SBS_RULE_EVENT_GROUP_DEPTH_WEIGHT 1u
#define SBS_RULE_EVENT_GROUP_ROI_STRUCTURAL_WEIGHT 2u
#define SBS_RULE_EVENT_GROUP_ROI_EXPOSURE_WEIGHT 3u
#define SBS_RULE_EVENT_GROUP_SCROLL_Y_WEIGHT \
    SBS_RULE_SUMMARY_EVENT_GROUP_SCROLL_Y_WEIGHT
#define SBS_RULE_EVENT_GROUP_SCROLL_Y_SIGNED \
    SBS_RULE_SUMMARY_EVENT_GROUP_SCROLL_Y_SIGNED
#define SBS_RULE_EVENT_GROUP_SCROLL_Y_SUPPORT \
    SBS_RULE_SUMMARY_EVENT_GROUP_SCROLL_Y_SUPPORT
#define SBS_RULE_EVENT_GROUP_SCROLL_X_WEIGHT \
    SBS_RULE_SUMMARY_EVENT_GROUP_SCROLL_X_WEIGHT
#define SBS_RULE_EVENT_GROUP_SCROLL_X_SIGNED \
    SBS_RULE_SUMMARY_EVENT_GROUP_SCROLL_X_SIGNED
#define SBS_RULE_EVENT_GROUP_SCROLL_X_SUPPORT \
    SBS_RULE_SUMMARY_EVENT_GROUP_SCROLL_X_SUPPORT
#define SBS_RULE_EVENT_GROUP_FRESH_ACTIVITY_MASS \
    SBS_RULE_SUMMARY_EVENT_GROUP_FRESH_ACTIVITY_MASS
#define SBS_RULE_EVENT_GROUP_FRESH_ACTIVITY_INSIDE_PREVIOUS_TARGET \
    SBS_RULE_SUMMARY_EVENT_GROUP_FRESH_ACTIVITY_INSIDE_PREVIOUS_TARGET
#define SBS_RULE_EVENT_SOURCE_MARKER SBS_RULE_SUMMARY_EVENT_SOURCE_MARKER
#define SBS_RULE_EVENT_SOURCE_PREAGGREGATED 1.0f

// Production publishes one global temporal seed and its media-expanded envelope.
#define SBS_RULE_ATTRIBUTION_META_BASE \
    SBS_RULE_SUMMARY_ATTRIBUTION_META_BASE
#define SBS_RULE_ATTRIBUTION_META_FLOAT_COUNT \
    SBS_RULE_SUMMARY_ATTRIBUTION_META_FLOAT_COUNT
#define SBS_RULE_ATTRIBUTION_GLOBAL_X_FIRST 0u
#define SBS_RULE_ATTRIBUTION_GLOBAL_X_END 1u
#define SBS_RULE_ATTRIBUTION_GLOBAL_X_VALID 2u
#define SBS_RULE_ATTRIBUTION_GLOBAL_X_AMBIGUOUS 3u
#define SBS_RULE_ATTRIBUTION_GLOBAL_Y_FIRST 4u
#define SBS_RULE_ATTRIBUTION_GLOBAL_Y_END 5u
#define SBS_RULE_ATTRIBUTION_GLOBAL_Y_VALID 6u
#define SBS_RULE_ATTRIBUTION_GLOBAL_Y_AMBIGUOUS 7u
#define SBS_RULE_ATTRIBUTION_SEED_X_FIRST 8u
#define SBS_RULE_ATTRIBUTION_SEED_X_END 9u
#define SBS_RULE_ATTRIBUTION_SEED_Y_FIRST 10u
#define SBS_RULE_ATTRIBUTION_SEED_Y_END 11u
#define SBS_RULE_ATTRIBUTION_SEED_VALID 12u
#define SBS_RULE_ATTRIBUTION_GLOBAL_RECT_COHERENCE 13u
#define SBS_RULE_ATTRIBUTION_GLOBAL_RECT_MASS 22u
#define SBS_RULE_ATTRIBUTION_GLOBAL_RECT_FRESH 23u

// Private final-reducer scratch follows cross-axis attribution.
#define SBS_RULE_TEMPORAL_PROBE_BASE \
    SBS_RULE_SUMMARY_TEMPORAL_PROBE_BASE
#define SBS_RULE_TEMPORAL_PROBE_EVENT_VALID 0u
#define SBS_RULE_TEMPORAL_PROBE_EVENT_X0 1u
#define SBS_RULE_TEMPORAL_PROBE_EVENT_Y0 2u
#define SBS_RULE_TEMPORAL_PROBE_EVENT_X1 3u
#define SBS_RULE_TEMPORAL_PROBE_EVENT_Y1 4u
#define SBS_RULE_TEMPORAL_PROBE_EVENT_FILL 5u
#define SBS_RULE_TEMPORAL_PROBE_EVENT_CONTAINMENT 6u
#define SBS_RULE_TEMPORAL_PROBE_HOLD_OBSERVED 7u
#define SBS_RULE_TEMPORAL_PROBE_HOLD_VALID 8u
#define SBS_RULE_TEMPORAL_PROBE_HOLD_CONTAINMENT 9u
#define SBS_RULE_TEMPORAL_PROBE_UNRESOLVED 10u
#define SBS_RULE_TEMPORAL_PROBE_AMBIGUOUS 11u
#define SBS_RULE_TEMPORAL_PROBE_FLOAT_COUNT \
    SBS_RULE_SUMMARY_TEMPORAL_PROBE_FLOAT_COUNT
#define SBS_RULE_REDUCE_AGGREGATE_BASE \
    SBS_RULE_SUMMARY_REDUCE_AGGREGATE_BASE
#define SBS_RULE_REDUCE_AGGREGATE_VALID_CELLS 0u
#define SBS_RULE_REDUCE_AGGREGATE_REQUIRED_INVALID 1u
#define SBS_RULE_REDUCE_AGGREGATE_FLOAT_COUNT \
    SBS_RULE_SUMMARY_REDUCE_AGGREGATE_FLOAT_COUNT
#define SBS_RULE_SUMMARY_REQUIRED_FLOAT_COUNT \
    (SBS_RULE_REDUCE_AGGREGATE_BASE + \
     SBS_RULE_REDUCE_AGGREGATE_FLOAT_COUNT)

// Shadow-rules calibration thresholds. A locked ROI corroborates an already-armed adaptive hard
// cut, but it never originates a cut by itself. Exposure is intentionally a frame-global
// photometric event/veto; ROI-local exposure support remains diagnostic telemetry only. Raw
// support and depth coverage are retained in rule state so whole-clip traces can explain event
// attribution and calibrate the hard-cut threshold.
static const float SBS_RULE_ROI_EVENT_MIN_DEPTH_COVERAGE = 0.05f;
static const float SBS_RULE_ROI_STRUCTURAL_CUT_MIN_SUPPORT = 0.02f;

// Quiet content uses one global central-mass hull. Detached islands are not
// selected or reclassified; a broad or sparse result abstains.
static const float SBS_RULE_MEDIA_AXIS_TRIM_MASS = 0.10f;
static const float SBS_RULE_MEDIA_AXIS_EDGE_SUPPORT_FRACTION = 0.04f;
static const float SBS_RULE_MEDIA_MIN_VIEWPORT_MASS = 0.01f;
static const float SBS_RULE_MEDIA_MIN_HULL_DENSITY = 0.06f;
// Below this omitted-area gain, changing model shape and convergence geometry
// costs more stability than the crop returns in linear sampling density.
static const float SBS_RULE_IDENTITY_MIN_AREA = 0.75f;
// Temporal evidence is continuous. The evidence pass first accumulates ordinal activity over
// time and requires photographic support; this projection then suppresses its weak tail. Using
// that classified occupancy (rather than a single frame's raw bit flips) lets ordinary playback
// reveal the stable player envelope without making one scene cut the only acquisition chance.
static const float SBS_RULE_TEMPORAL_EVENT_CELL_LOW = 1.0f / 8.0f;
static const float SBS_RULE_TEMPORAL_EVENT_CELL_HIGH = 3.0f / 8.0f;
static const float SBS_RULE_TEMPORAL_EVENT_AXIS_FILL = 0.10f;
static const float SBS_RULE_TEMPORAL_EVENT_MIN_AREA = 0.035f;
static const float SBS_RULE_TEMPORAL_EVENT_MIN_FILL = 0.32f;
// Independent X/Y projections can otherwise manufacture the bounding rectangle of crossed or
// corner activity. A fixed 4x4 joint histogram compares the actual temporal/media support with
// the separable distribution implied by its row and column marginals. This is deliberately a
// coarse, low-frequency coherence check: irregular or disconnected evidence abstains, while a
// stable media envelope may corroborate localized motion inside an ordinary player.
static const uint SBS_RULE_TEMPORAL_COHERENCE_BINS = 4u;
static const float SBS_RULE_TEMPORAL_MIN_RECT_COHERENCE = 0.85f;
// One dominance threshold governs axis competition, retained/current
// activity containment, and challenger size. Cases that cannot clear this
// common margin abstain instead of entering a special-case route.
static const float SBS_RULE_CANDIDATE_MIN_DOMINANCE = 0.72f;
// A close second run on either axis is evidence for two independently moving regions, not a
// unique player. Candidates rank by supported span before mass so a narrow high-change ad cannot
// outrank a wider, sparsely changing player. Once a partition clears the axis support floor,
// close span alone is ambiguous; intensity must not choose between equally sized players. The
// rule remains identical for horizontal and vertical layouts.
static const float SBS_RULE_TEMPORAL_RUN_AMBIGUITY_FRACTION =
    SBS_RULE_CANDIDATE_MIN_DOMINANCE;
// Relocation compares two-dimensional candidate area using the square of the
// shared linear dominance margin.
static const float SBS_RULE_TEMPORAL_COMPONENT_AMBIGUITY_AREA_FRACTION =
    SBS_RULE_TEMPORAL_RUN_AMBIGUITY_FRACTION *
    SBS_RULE_TEMPORAL_RUN_AMBIGUITY_FRACTION;
// A boundary lane must be quiet/non-photographic over most of the opposite viewport axis. This
// recognizes whitespace/dividers at any orientation without naming an edge, aspect ratio, or
// component count.
static const float SBS_RULE_TEMPORAL_GUTTER_AXIS_MEAN = 0.62f;
// After temporal discovery, each axis expands only through contiguous
// photographic support. Discovery itself may bridge one analysis cell.
static const float SBS_RULE_TEMPORAL_CONTINUATION_AXIS_FRACTION = 0.18f;
static const uint SBS_RULE_TEMPORAL_MAX_UNSUPPORTED_HOLE_CELLS = 1u;
// Material classified motion which cannot produce one coherent rectangle is contradictory
// evidence, not permission to reinterpret the static media hull as a collage. The same floor
// also defines whether activity is observable around an existing video target.
static const float SBS_RULE_TEMPORAL_MATERIAL_MIN_VIEWPORT_FRACTION = 0.002f;
// Initial playback acquisition may react after a short, stable observation.
// Cold static content waits through a bounded playback-observation window:
// motion gets the first opportunity to prove VIDEO, while an image or genuinely
// paused-at-entry page still becomes generic CONTENT before the 800 ms gate.
// Replacing a proven target remains deliberately slower because it changes an
// already-visible crop.
static const float SBS_RULE_INITIAL_VIDEO_ACQUIRE_MIN_SECONDS =
    SBS_SCENE_TEMPORAL_CADENCE_GAP_SECONDS;
static const float SBS_RULE_INITIAL_CONTENT_ACQUIRE_MIN_SECONDS = 0.70f;
// Scroll is a frame-global consensus over independent per-cell translation votes. Only textured
// ordinal cells for which all seven shifts are observable vote. Each cell must beat zero shift
// and the opposite direction; the reducer then requires one dominant direction and a minimum
// supporter fraction among observable cells. That fraction is deliberately not viewport area:
// it keeps sparse pages observable, while cold coherent camera pans conservatively abstain.
static const float SBS_RULE_SCROLL_MIN_TEXTURE = 0.08f;
static const float SBS_RULE_SCROLL_MIN_ZERO_GAIN = 0.08f;
static const float SBS_RULE_SCROLL_MIN_RUNNER_MARGIN = 0.05f;
static const float
    SBS_RULE_SCROLL_MIN_SUPPORT_FRACTION_OF_OBSERVABLE = 0.12f;

float SbsRuleTemporalMass(float activity) {
    return isfinite(activity) ?
        saturate(
            (activity - SBS_RULE_TEMPORAL_EVENT_CELL_LOW) /
            (SBS_RULE_TEMPORAL_EVENT_CELL_HIGH -
             SBS_RULE_TEMPORAL_EVENT_CELL_LOW)) :
        0.0f;
}

float3 SbsRuleClassifyScrollVote(
    bool observable,
    float cost_neg3,
    float cost_neg2,
    float cost_neg1,
    float cost_zero,
    float cost_pos1,
    float cost_pos2,
    float cost_pos3)
{
    if (!observable) {
        return 0.0f.xxx;
    }
    const float negative_cost =
        min(cost_neg3, min(cost_neg2, cost_neg1));
    const float positive_cost =
        min(cost_pos1, min(cost_pos2, cost_pos3));
    const bool negative_wins = negative_cost < positive_cost;
    const float best_cost =
        negative_wins ? negative_cost : positive_cost;
    const float opposite_cost =
        negative_wins ? positive_cost : negative_cost;
    const float zero_gain =
        (cost_zero - best_cost) / max(cost_zero, 1.0f);
    const float opposite_margin =
        (opposite_cost - best_cost) / max(opposite_cost, 1.0f);
    if (
        best_cost < cost_zero &&
        zero_gain >= SBS_RULE_SCROLL_MIN_ZERO_GAIN &&
        opposite_margin >= SBS_RULE_SCROLL_MIN_RUNNER_MARGIN
    ) {
        return float3(
            1.0f,
            negative_wins ? -1.0f : 1.0f,
            1.0f);
    }
    // Keep the observable-cell denominator even when this cell abstains.
    // Scroll coverage is a fraction of textured cells that could make a
    // complete comparison, not a fraction of the entire viewport.
    return float3(1.0f, 0.0f, 0.0f);
}

uint SbsRuleColumnIndex(uint column, uint field) {
    return SBS_RULE_COLUMN_BASE + column * SBS_RULE_COLUMN_STRIDE + field;
}

uint SbsRulePlanIndex(uint field) {
    return SBS_RULE_PLAN_BASE + field;
}

uint SbsRuleMediaRowIndex(uint row) {
    return SBS_RULE_MEDIA_ROW_BASE + row;
}

uint SbsRuleEventGroupIndex(uint group, uint field) {
    return SBS_RULE_EVENT_GROUP_BASE +
           group * SBS_RULE_EVENT_GROUP_STRIDE + field;
}

uint SbsRuleTemporalRowIndex(uint row, uint field) {
    return SBS_RULE_TEMPORAL_ROW_BASE +
           row * SBS_RULE_TEMPORAL_ROW_STRIDE + field;
}

uint SbsRuleAttributionMetaIndex(uint field) {
    return SBS_RULE_ATTRIBUTION_META_BASE + field;
}

uint SbsRuleTemporalProbeIndex(uint field) {
    return SBS_RULE_TEMPORAL_PROBE_BASE + field;
}

uint SbsRuleReduceAggregateIndex(uint field) {
    return SBS_RULE_REDUCE_AGGREGATE_BASE + field;
}

void SbsRuleStoreUint(
    RWStructuredBuffer<float> summary,
    uint index,
    uint value)
{
    summary[index] = (float)value;
}

// If this fails, the shared native/HLSL allocation constant was not updated with this layout.
#if SBS_RULE_PLAN_BASE != \
    (SBS_RULE_COLUMN_BASE + \
     SBS_RULE_COLUMN_COUNT * SBS_RULE_COLUMN_STRIDE)
#error SBS rule-summary plan base does not follow its column storage
#endif
#if SBS_RULE_MEDIA_ROW_BASE != \
    (SBS_RULE_PLAN_BASE + SBS_RULE_SUMMARY_PLAN_FLOAT_COUNT)
#error SBS rule-summary media-row base does not follow its plan storage
#endif
#if SBS_RULE_TEMPORAL_ROW_BASE != \
    (SBS_RULE_MEDIA_ROW_BASE + SBS_RULE_MEDIA_ROW_COUNT)
#error SBS rule-summary temporal-row base does not follow media rows
#endif
#if SBS_RULE_EVENT_GROUP_BASE != SBS_RULE_SUMMARY_EVENT_GROUP_BASE
#error SBS rule-summary event-group base does not match its shared layout
#endif
#if SBS_RULE_EVENT_GROUP_BASE != \
    (SBS_RULE_TEMPORAL_ROW_BASE + \
     SBS_RULE_TEMPORAL_ROW_COUNT * SBS_RULE_TEMPORAL_ROW_STRIDE)
#error SBS rule-summary event groups do not follow temporal rows
#endif
#if SBS_RULE_EVENT_SOURCE_MARKER != \
    (SBS_RULE_EVENT_GROUP_BASE + \
     SBS_RULE_EVENT_GROUP_COUNT * SBS_RULE_EVENT_GROUP_STRIDE)
#error SBS rule-summary event marker does not follow its group storage
#endif
#if SBS_RULE_ATTRIBUTION_META_BASE != \
    (SBS_RULE_EVENT_SOURCE_MARKER + 1u)
#error SBS rule-summary attribution metadata does not follow the event marker
#endif
#if SBS_RULE_TEMPORAL_PROBE_BASE != \
    (SBS_RULE_ATTRIBUTION_META_BASE + \
     SBS_RULE_ATTRIBUTION_META_FLOAT_COUNT)
#error SBS rule-summary temporal scratch does not follow attribution
#endif
#if SBS_RULE_REDUCE_AGGREGATE_BASE != \
    (SBS_RULE_TEMPORAL_PROBE_BASE + \
     SBS_RULE_TEMPORAL_PROBE_FLOAT_COUNT)
#error SBS rule-summary reduce aggregate does not follow temporal scratch
#endif
#if SBS_RULE_SUMMARY_FLOAT_COUNT != SBS_RULE_SUMMARY_REQUIRED_FLOAT_COUNT
#error SBS rule-summary layout does not match its shared allocation size
#endif
#if (SBS_RULE_SUMMARY_EVENT_GROUP_COUNT * \
     SBS_RULE_EVIDENCE_GROUP_THREAD_COUNT) != \
    (SBS_RULE_SUMMARY_COLUMN_COUNT * \
     SBS_RULE_SUMMARY_COLUMN_COUNT)
#error SBS rule-summary event-group storage does not match the evidence dispatch topology
#endif

#endif
