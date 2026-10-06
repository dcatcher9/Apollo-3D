// Compact SLR14 lower-text authority with one source-timed adaptive shared UI plane.
//
// OCR8 is the sole geometry source.  A coherent subtitle stack plus any bottom ribbon candidates
// need two distinct, exact-frame observations before becoming one shared-plane owner. Only tight
// cores copied from the current OCR8 record may authorize ownership; only their paired
// same-frame covers may probe or condition the BaseField. Cached owner, pending, target, and death-grace
// state never manufacture current geometry. Ordinary covers retain the established four-sided
// analytic collar; a canonical full-width/bottom ribbon cover exposes only its top collar.

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2_ocr_assert.generated.hlsl"
#include "include/depth_coordinate_v2.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"

StructuredBuffer<uint4> GeometryState : register(t0);
StructuredBuffer<float4> CutBridge : register(t1);
Texture2D<float> BaseField : register(t2);
StructuredBuffer<uint> LocatorStateRead : register(t3);
StructuredBuffer<uint> ConditionParams : register(t4);
StructuredBuffer<uint> OcrRecord : register(t7);

RWStructuredBuffer<uint> LocatorState : register(u2);
RWTexture2D<float> ConditionedField : register(u3);
RWStructuredBuffer<uint> ConditionParamsOut : register(u4);

cbuffer SubtitleLocatorConstants : register(b2) {
    uint4 locator_field;   // width, height, ROI top, ROI bottom
    uint4 locator_source;  // analysis source width, height, enabled, input-domain reset
    uint4 locator_frame;   // matched frame lo/hi, analysis generation lo/hi
    uint4 locator_content; // integer half-open real-source rectangle in locator_field
    uint4 locator_observation; // exact source timestamp us lo/hi, two zero reserved words
};

static const uint FLAG_OWNER = 1u;
static const uint FLAG_PENDING = 2u;
static const uint FLAG_TARGET_VALID = 4u;
static const uint FLAG_TARGET_RESET = 8u;
static const uint FLAG_PROVISIONAL_CURRENT = V2_SUBTITLE_LOCATOR_PROVISIONAL_CURRENT_FLAG;
static const uint KNOWN_FLAGS = FLAG_OWNER | FLAG_PENDING | FLAG_TARGET_VALID |
    FLAG_TARGET_RESET | FLAG_PROVISIONAL_CURRENT;

static const uint EVENT_NONE = 0u;
static const uint EVENT_BIRTH = 1u;
static const uint EVENT_DEATH = 2u;
static const uint EVENT_HANDOFF = 3u;

static const uint MAX_LINES = V2_SUBTITLE_LOCATOR_RECTANGLE_CAPACITY;
static const uint STACK_BASE = 0u;
static const uint OLD_OWNER_BASE = 4u;
static const uint OLD_PENDING_BASE = 8u;
static const uint NEW_OWNER_BASE = 12u;
static const uint NEW_PENDING_BASE = 16u;
static const uint NEW_CURRENT_BASE = 20u;
static const uint MATCHED_BASE = 24u;
static const uint DEATH_GRACE_OBSERVATIONS =
    V2_SUBTITLE_LOCATOR_DEATH_GRACE_OBSERVATIONS;

groupshared uint PreviousState[V2_SUBTITLE_LOCATOR_STATE_WORD_COUNT];
groupshared uint ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_STATE_WORD_COUNT];
groupshared uint4 QualifiedCores[V2_OCR_FINAL_BOX_CAPACITY];
groupshared uint4 QualifiedCovers[V2_OCR_FINAL_BOX_CAPACITY];
groupshared uint QualifiedKinds[V2_OCR_FINAL_BOX_CAPACITY];
groupshared uint QualifiedMasks[V2_OCR_FINAL_BOX_CAPACITY];
groupshared uint4 WorkRects[28];
groupshared uint WorkKinds[28];
groupshared uint4 StackCovers[MAX_LINES];
groupshared uint4 MatchedCovers[MAX_LINES];

// The adaptive controller is owned by the versioned SLR tail. These are within-group scratch,
// not another persistent authority source. Counts cover the complete current-cover union.
static const uint ADAPTIVE = V2_SUBTITLE_LOCATOR_ADAPTIVE_OFFSET;
static const uint ADAPTIVE_APPROACH = 1u;
static const uint ADAPTIVE_RETREAT = 2u;
static const uint ADAPTIVE_CAPPED = 4u;
static const uint ADAPTIVE_CLOCK = 8u;
groupshared uint AdaptiveReset;
groupshared uint AdaptiveDistinct;
groupshared uint AdaptiveOcrValid;
groupshared uint AdaptiveDomainValid;
groupshared uint AdaptiveGeometryReady;
// The adaptive plane retains four-source-pixel levels through the full representation cap. Supported Host
// sources are at most 5120 pixels wide: ceil(5120/100)=52 is the last level, with one extra
// histogram bucket for samples no candidate can clear.
static const uint ADAPTIVE_HOST_MAX_SOURCE_WIDTH = 5120u;
static const uint ADAPTIVE_COUNT_WORDS = 56u;
groupshared uint AdaptiveCounts[ADAPTIVE_COUNT_WORDS];
groupshared uint AdaptiveCurrentCount;
groupshared uint4 AdaptiveCurrentCovers[MAX_LINES];

static const uint CONDITION_PARAM_SCHEMA_WORD = 0u;
static const uint CONDITION_PARAM_TAG_WORD = 1u;
static const uint CONDITION_PARAM_CURRENT_COUNT_WORD = 2u;
static const uint CONDITION_PARAM_CURRENT_KINDS_WORD = 3u;
static const uint CONDITION_PARAM_FADE_STEP_WORD = 4u;
static const uint CONDITION_PARAM_TARGET_WORD = 5u;

bool ProvisionalSingleLineReplacementRects(uint4 old_core, uint4 current_core);

bool FiniteFloat(float value) {
    return (asuint(value) & 0x7f800000u) != 0x7f800000u;
}

bool AdaptivePolicyValid() {
    return V2JointPlaneConstantsValid() &&
        FiniteFloat(v2_requested_gain) && v2_requested_gain > 0.0f &&
        all(uint2(v2_joint_observation_timestamp_low,
                  v2_joint_observation_timestamp_high) == locator_observation.xy);
}

// The existing raw-coordinate publication owns Base. Flat/failed geometry is not an empty
// scene observation: it cannot contribute conflict counts, spend UI dwell, or lower the plane.
// This check runs only in the observation/condition verdict resolver, never per conditioned pixel.
bool GeometryPublicationReady() {
    uint vector_count, stride;
    GeometryState.GetDimensions(vector_count, stride);
    if (vector_count * 4u != V2_SHADOW_STATE_WORD_COUNT || stride != 16u ||
        !AdaptivePolicyValid()) return false;
    float4 active = asfloat(GeometryState[0u]);
    float4 control = asfloat(GeometryState[1u]);
    uint4 mapping = GeometryState[2u];
    V2AdaptiveCameraState camera;
    camera.clock = GeometryState[3u];
    camera.target = GeometryState[4u];
    camera.seed_times = GeometryState[5u];
    camera.seed_values = GeometryState[6u];
    uint2 now = uint2(v2_joint_observation_timestamp_low,
                     v2_joint_observation_timestamp_high);
    return V2CameraStateWithAdaptiveStateValid(active, control, mapping, camera) &&
        V2_STATE_FRAME_VALID(control) == 1.0f &&
        mapping.y == V2_CONTRACT_TAG && mapping.z == V2_ADAPTIVE_POLICY_ID &&
        any(now != 0u) && all(camera.clock.xy == now) &&
        asuint(SBS_STATE_CUT_CONTRACT_TAG_BITS(
            CutBridge[SBS_STATE_VECTOR_CUT_CONTRACT_TAG_BITS])) == SBS_CUT_CONTRACT_TAG &&
        asuint(V2_STATE_CONFIRMED_CUT_COUNT(control)) ==
            asuint(SBS_STATE_HARD_CUT_COUNT(CutBridge[SBS_STATE_VECTOR_HARD_CUT_COUNT]));
}

uint AdaptiveMaximumIndex() {
    // .04 source U / (4/source_width) equals source_width/100. Integer ceil keeps the level
    // bound independent of SM5 division rounding; the last candidate returns the cap exactly.
    return (locator_source.x + 99u) / 100u;
}

uint AdaptiveRead(bool previous, uint word) {
    return previous ? PreviousState[ADAPTIVE + word] : ConditionStateSnapshot[ADAPTIVE + word];
}

bool AdaptiveTailValid(bool previous) {
    bool all_zero = true;
    [unroll]
    for (uint word = 0u; word < V2_SUBTITLE_LOCATOR_ADAPTIVE_WORD_COUNT; ++word) {
        all_zero = all_zero && AdaptiveRead(previous, word) == 0u;
    }
    // An inactive publication can authenticate only an empty current mask. Previous inactive
    // state reacquires ownership before it can grant any current cover authority.
    if (all_zero) return !previous && ConditionStateSnapshot[20u] == 0u;
    uint flags = AdaptiveRead(previous, 1u);
    uint2 clock = uint2(AdaptiveRead(previous, 4u), AdaptiveRead(previous, 5u));
    uint2 approach = uint2(AdaptiveRead(previous, 6u), AdaptiveRead(previous, 7u));
    uint2 retreat = uint2(AdaptiveRead(previous, 8u), AdaptiveRead(previous, 9u));
    float applied = asfloat(AdaptiveRead(previous, 3u));
    float limit = asfloat(AdaptiveRead(previous, 12u));
    uint covered = AdaptiveRead(previous, 13u);
    uint content_area = (locator_content.z - locator_content.x) *
        (locator_content.w - locator_content.y);
    uint durable_target = previous ? PreviousState[18u] : ConditionStateSnapshot[18u];
    uint durable_fade = previous ? PreviousState[24u] : ConditionStateSnapshot[24u];
    uint owner_flags = previous ? PreviousState[2u] : ConditionStateSnapshot[2u];
    uint provisional_target = previous ? PreviousState[V2_SUBTITLE_LOCATOR_PROVISIONAL_TARGET_WORD] :
        ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_PROVISIONAL_TARGET_WORD];
    uint provisional_fade = previous ? PreviousState[V2_SUBTITLE_LOCATOR_PROVISIONAL_FADE_WORD] :
        ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_PROVISIONAL_FADE_WORD];
    if ((owner_flags & (FLAG_OWNER | FLAG_TARGET_VALID)) ==
            (FLAG_OWNER | FLAG_TARGET_VALID) &&
        (durable_target != asuint(applied) || durable_fade != 2u ||
        ((owner_flags & FLAG_PROVISIONAL_CURRENT) != 0u &&
         (provisional_target != asuint(applied) || provisional_fade != 2u)))) return false;
    return AdaptiveRead(previous, 0u) == 1u && (flags & ~15u) == 0u &&
        (flags & (ADAPTIVE_APPROACH | ADAPTIVE_RETREAT)) !=
            (ADAPTIVE_APPROACH | ADAPTIVE_RETREAT) &&
        AdaptiveRead(previous, 2u) <= AdaptiveMaximumIndex() &&
        AdaptiveRead(previous, 10u) <= AdaptiveMaximumIndex() &&
        AdaptiveRead(previous, 11u) <= AdaptiveMaximumIndex() &&
        FiniteFloat(applied) && FiniteFloat(limit) && applied >= 0.0f &&
        limit >= 0.0f && limit == V2DisplayBudget() && applied <= limit &&
        (((flags & ADAPTIVE_CLOCK) != 0u) == any(clock != 0u)) &&
        (((flags & ADAPTIVE_APPROACH) != 0u) == any(approach != 0u)) &&
        (((flags & ADAPTIVE_RETREAT) != 0u) == any(retreat != 0u)) &&
        ((flags & ADAPTIVE_APPROACH) == 0u || !V2ClockAfter(approach, clock)) &&
        ((flags & ADAPTIVE_RETREAT) == 0u || !V2ClockAfter(retreat, clock)) &&
        covered <= content_area && AdaptiveRead(previous, 14u) <= covered &&
        (((flags & ADAPTIVE_CAPPED) != 0u) ==
         (AdaptiveRead(previous, 14u) * 5u > covered ||
          AdaptiveRead(previous, 14u) * 50u > content_area)) &&
        AdaptiveRead(previous, 15u) <= covered;
}

bool ZeroRect(uint4 rectangle) {
    return all(rectangle == uint4(0u, 0u, 0u, 0u));
}

uint LocatorContentWidth() {
    return locator_content.z - locator_content.x;
}

bool SubtitleTargetIsValid(float target) {
    return FiniteFloat(target) && FiniteFloat(v2_direct_container_limit) &&
        v2_direct_container_limit > 0.0f && target >= 0.0f &&
        target <= v2_direct_container_limit;
}

// The active field/ROI is carried by the existing runtime cbuffer and repeated in OCR8.  The host
// enables this path only after authenticating the DAV2 tensor shape; SLR13 independently checks
// finite ABI-sized geometry, the actual BaseField dispatch dimensions, and that the ROI is a
// non-empty subset of the exact bottom-6:1 detector crop. Any disagreement publishes no current
// authority; the complete writer publishes BaseField exactly, while full-content live production
// emits zero indirect groups and leaves the final UAV untouched.
bool LocatorDomainGeometryValid() {
    if (!AdaptivePolicyValid() ||
        any(locator_observation.zw != 0u) ||
        locator_source.x == 0u || locator_source.y == 0u || locator_source.z > 1u ||
        locator_source.x > ADAPTIVE_HOST_MAX_SOURCE_WIDTH ||
        locator_field.x == 0u || locator_field.y == 0u ||
        locator_field.x > 0xffffu || locator_field.y > 0xffffu ||
        locator_field.z >= locator_field.w || locator_field.w > locator_field.y ||
        target_w != locator_field.x || target_h != locator_field.y ||
        locator_content.x >= locator_content.z || locator_content.y >= locator_content.w ||
        locator_content.z > locator_field.x || locator_content.w > locator_field.y ||
        locator_field.z < locator_content.y || locator_field.w > locator_content.w ||
        any(locator_content != DepthAnalysisContentCells())) {
        return false;
    }

    if (!V2SubtitleOcrFieldIsCalibrated(locator_field.x, locator_field.y)) return false;

    uint expected_roi_top;
    uint expected_roi_bottom;
    if (!V2SubtitleOcrProjectContentRowCeil(
            locator_source.x, locator_source.y, locator_field.x, locator_field.y,
            locator_content, V2_OCR_SAFE_ROW_TOP, expected_roi_top) ||
        !V2SubtitleOcrProjectContentRowCeil(
            locator_source.x, locator_source.y, locator_field.x, locator_field.y,
            locator_content, V2_OCR_SAFE_ROW_BOTTOM, expected_roi_bottom)) return false;
    return locator_field.z == expected_roi_top && locator_field.w == expected_roi_bottom;
}

bool LocatorGeometryValid() {
    return locator_source.z == 1u && LocatorDomainGeometryValid();
}

bool ValidRoiRect(uint4 rectangle) {
    return rectangle.x < rectangle.z && rectangle.y < rectangle.w &&
        rectangle.x >= locator_content.x && rectangle.z <= locator_content.z &&
        rectangle.y >= locator_field.z && rectangle.w <= locator_field.w;
}

bool ValidFieldRect(uint4 rectangle) {
    return rectangle.x < rectangle.z && rectangle.y < rectangle.w &&
        rectangle.x >= locator_content.x && rectangle.y >= locator_content.y &&
        rectangle.z <= locator_content.z && rectangle.w <= locator_content.w;
}

uint RectArea(uint4 rectangle) {
    return (rectangle.z - rectangle.x) * (rectangle.w - rectangle.y);
}

float RectIou(uint4 a, uint4 b) {
    uint left = max(a.x, b.x);
    uint top = max(a.y, b.y);
    uint right = min(a.z, b.z);
    uint bottom = min(a.w, b.w);
    uint intersection = right > left && bottom > top ? (right - left) * (bottom - top) : 0u;
    uint union_area = RectArea(a) + RectArea(b) - intersection;
    return union_area == 0u ? 0.0f : (float)intersection / (float)union_area;
}

uint4 RectSummary(uint base, uint count) {
    if (count == 0u) return uint4(0u, 0u, 0u, 0u);
    uint4 summary = WorkRects[base];
    [unroll]
    for (uint index = 1u; index < MAX_LINES; ++index) {
        if (index < count) {
            uint4 rectangle = WorkRects[base + index];
            summary.x = min(summary.x, rectangle.x);
            summary.y = min(summary.y, rectangle.y);
            summary.z = max(summary.z, rectangle.z);
            summary.w = max(summary.w, rectangle.w);
        }
    }
    return summary;
}

uint RectAreaSum(uint base, uint count) {
    uint area = 0u;
    [unroll]
    for (uint index = 0u; index < MAX_LINES; ++index) {
        if (index < count) area += RectArea(WorkRects[base + index]);
    }
    return area;
}

void CopyRects(uint destination, uint source, uint count) {
    // Keep this bounded integer copy loop intact: forced unrolling across ownership branches
    // triggers the SM5 compiler's register-type emitter failure after pipeline specialization.
    [loop]
    for (uint index = 0u; index < MAX_LINES; ++index) {
        WorkRects[destination + index] = index < count ? WorkRects[source + index] :
            uint4(0u, 0u, 0u, 0u);
        WorkKinds[destination + index] = index < count ? WorkKinds[source + index] : 0u;
    }
}

bool CoherentLines(uint4 a, uint4 b) {
    uint width_a = a.z - a.x;
    uint width_b = b.z - b.x;
    uint height_a = a.w - a.y;
    uint height_b = b.w - b.y;
    uint overlap = min(a.z, b.z) > max(a.x, b.x) ?
        min(a.z, b.z) - max(a.x, b.x) : 0u;
    uint center_x_delta_twice = (a.x + a.z) > (b.x + b.z) ?
        (a.x + a.z) - (b.x + b.z) : (b.x + b.z) - (a.x + a.z);
    uint left_delta = a.x > b.x ? a.x - b.x : b.x - a.x;
    uint right_delta = a.z > b.z ? a.z - b.z : b.z - a.z;
    uint center_y_delta_twice = (a.y + a.w) > (b.y + b.w) ?
        (a.y + a.w) - (b.y + b.w) : (b.y + b.w) - (a.y + a.w);
    uint gap = 0u;
    if (a.w <= b.y) gap = b.y - a.w;
    else if (b.w <= a.y) gap = a.y - b.w;
    return overlap * 2u >= min(width_a, width_b) &&
        (center_x_delta_twice <= max(height_a, height_b) ||
         left_delta <= max(height_a, height_b) || right_delta <= max(height_a, height_b)) &&
        max(height_a, height_b) <= 2u * min(height_a, height_b) &&
        center_y_delta_twice >= min(height_a, height_b) &&
        gap * 2u <= max(height_a, height_b);
}

bool SameBaselineSegments(uint4 a, uint4 b) {
    // OCR deliberately keeps ordinary covers independent when a horizontal gap exceeds the text
    // join limit. Re-associate only strongly aligned, nearby core segments for owner selection; do
    // not union their geometry. The strict vertical relation prevents a nearby upper/lower scene
    // text line from bridging otherwise separate components.
    bool a_before_b = a.z <= b.x;
    bool b_before_a = b.z <= a.x;
    if (!a_before_b && !b_before_a) return false;
    uint height_a = a.w - a.y;
    uint height_b = b.w - b.y;
    uint shorter_height = min(height_a, height_b);
    uint taller_height = max(height_a, height_b);
    uint vertical_overlap = min(a.w, b.w) > max(a.y, b.y) ?
        min(a.w, b.w) - max(a.y, b.y) : 0u;
    uint center_y_delta_twice = (a.y + a.w) > (b.y + b.w) ?
        (a.y + a.w) - (b.y + b.w) : (b.y + b.w) - (a.y + a.w);
    uint horizontal_gap = a_before_b ? b.x - a.z : a.x - b.z;
    uint combined_span = max(a.z, b.z) - min(a.x, b.x);
    uint maximum_width = LocatorContentWidth() * V2_SUBTITLE_LOCATOR_MAX_WIDTH_NUMERATOR /
        V2_SUBTITLE_LOCATOR_MAX_WIDTH_DENOMINATOR;
    return vertical_overlap * 4u >= shorter_height * 3u &&
        taller_height <= 2u * shorter_height &&
        center_y_delta_twice <= shorter_height &&
        horizontal_gap <= 8u * taller_height &&
        combined_span <= maximum_width;
}

bool ValidOcrBoxPayload(uint offset) {
    uint4 rectangle = uint4(
        OcrRecord[offset + 0u], OcrRecord[offset + 1u],
        OcrRecord[offset + 2u], OcrRecord[offset + 3u]);
    float score = asfloat(OcrRecord[offset + 4u]);
    return ValidFieldRect(rectangle) && FiniteFloat(score) &&
        score >= V2_OCR_MIN_MEAN_SCORE && score <= 1.0f &&
        (OcrRecord[offset + 5u] & ~V2_OCR_BOX_KNOWN_FLAGS) == 0u &&
        OcrRecord[offset + 6u] != 0u && OcrRecord[offset + 6u] <= V2_OCR_OUTPUT_WIDTH &&
        OcrRecord[offset + 7u] < OcrRecord[offset + 6u];
}

bool ValidOcrPair(uint slot) {
    uint raw_offset = V2_OCR_RAW_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
    uint final_offset = V2_OCR_FINAL_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
    if (!ValidOcrBoxPayload(raw_offset) || !ValidOcrBoxPayload(final_offset)) return false;
    uint4 core = uint4(
        OcrRecord[raw_offset + 0u], OcrRecord[raw_offset + 1u],
        OcrRecord[raw_offset + 2u], OcrRecord[raw_offset + 3u]);
    uint4 cover = uint4(
        OcrRecord[final_offset + 0u], OcrRecord[final_offset + 1u],
        OcrRecord[final_offset + 2u], OcrRecord[final_offset + 3u]);
    if (!ValidRoiRect(core) || cover.x > core.x || cover.y > core.y ||
        cover.z < core.z || cover.w < core.w ||
        OcrRecord[raw_offset + 4u] != OcrRecord[final_offset + 4u] ||
        OcrRecord[raw_offset + 5u] != OcrRecord[final_offset + 5u] ||
        OcrRecord[raw_offset + 6u] != OcrRecord[final_offset + 6u] ||
        OcrRecord[raw_offset + 7u] != OcrRecord[final_offset + 7u]) {
        return false;
    }
    bool ribbon = (OcrRecord[raw_offset + 5u] & V2_OCR_BOX_FLAG_RIBBON) != 0u;
    if (ribbon) {
        uint width = core.z - core.x;
        uint minimum_bottom = 0u;
        bool projected = V2SubtitleOcrProjectContentRowCeil(
            locator_source.x, locator_source.y, locator_field.x, locator_field.y,
            locator_content,
            V2_OCR_SAFE_ROW_BOTTOM - V2_OCR_RIBBON_BOTTOM_TOLERANCE_PIXELS,
            minimum_bottom);
        if (width * V2_OCR_RIBBON_MIN_WIDTH_DENOMINATOR <
                LocatorContentWidth() * V2_OCR_RIBBON_MIN_WIDTH_NUMERATOR ||
            !projected || core.w < minimum_bottom || core.w > locator_field.w ||
            OcrRecord[raw_offset + 6u] <= OcrRecord[raw_offset + 7u] ||
            OcrRecord[raw_offset + 7u] < V2_OCR_RIBBON_MIN_STRUCTURAL_GAPS ||
            cover.x != locator_content.x || cover.z != locator_content.z ||
            cover.w != locator_content.w ||
            cover.y < locator_field.z) {
            return false;
        }
    } else if (!ValidRoiRect(cover)) return false;
    return true;
}

bool ZeroOcrBox(uint offset) {
    [unroll]
    for (uint word = 0u; word < V2_OCR_BOX_WORD_COUNT; ++word) {
        if (OcrRecord[offset + word] != 0u) return false;
    }
    return true;
}

bool ValidateOcrRecord(out uint final_count) {
    final_count = 0u;
    if (!LocatorGeometryValid()) {
        return false;
    }
    uint raw_count = OcrRecord[3u];
    final_count = OcrRecord[4u];
    if (OcrRecord[0u] != V2_OCR_RECORD_SCHEMA || OcrRecord[1u] != V2_OCR_RECORD_TAG ||
        OcrRecord[2u] != 1u || raw_count > V2_OCR_RAW_BOX_CAPACITY ||
        final_count > V2_OCR_FINAL_BOX_CAPACITY || raw_count != final_count ||
        OcrRecord[5u] != locator_frame.x || OcrRecord[6u] != locator_frame.y ||
        OcrRecord[7u] != locator_frame.z || OcrRecord[8u] != locator_frame.w ||
        OcrRecord[9u] != locator_source.x || OcrRecord[10u] != locator_source.y ||
        OcrRecord[11u] != locator_field.x || OcrRecord[12u] != locator_field.y ||
        OcrRecord[13u] != locator_field.z || OcrRecord[14u] != locator_field.w ||
        OcrRecord[15u] != 0u) {
        return false;
    }
    [loop]
    for (uint slot = 0u; slot < V2_OCR_RAW_BOX_CAPACITY; ++slot) {
        uint offset = V2_OCR_RAW_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
        if (slot < raw_count ? !ValidOcrPair(slot) : !ZeroOcrBox(offset)) return false;
    }
    [loop]
    for (uint final_slot = 0u; final_slot < V2_OCR_FINAL_BOX_CAPACITY; ++final_slot) {
        uint offset = V2_OCR_FINAL_BOX_OFFSET + final_slot * V2_OCR_BOX_WORD_COUNT;
        if (final_slot >= final_count && !ZeroOcrBox(offset)) return false;
    }
    return true;
}

uint BuildCurrentStack(uint final_count) {
    uint qualified_count = 0u;
    uint ribbon_mask = 0u;
    [loop]
    for (uint slot = 0u; slot < V2_OCR_FINAL_BOX_CAPACITY; ++slot) {
        if (slot < final_count) {
            uint raw_offset = V2_OCR_RAW_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
            uint final_offset = V2_OCR_FINAL_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
            uint4 core = uint4(
                OcrRecord[raw_offset + 0u], OcrRecord[raw_offset + 1u],
                OcrRecord[raw_offset + 2u], OcrRecord[raw_offset + 3u]);
            uint4 cover = uint4(
                OcrRecord[final_offset + 0u], OcrRecord[final_offset + 1u],
                OcrRecord[final_offset + 2u], OcrRecord[final_offset + 3u]);
            uint kind = (OcrRecord[raw_offset + 5u] & V2_OCR_BOX_FLAG_RIBBON) != 0u ? 1u : 0u;
            uint width = core.z - core.x;
            uint height = core.w - core.y;
            uint field_cell_scale =
                V2SubtitleLocatorFieldCellScale(locator_field.x, locator_field.y);
            // Generic subtitle-line geometry.  In particular, square badges/logos fail the
            // aspect gate without a position- or brand-specific exclusion.
            // Coarse DAV2 tensors have a 434-cell short side. Exact convex-2x live fields keep
            // square cells at twice that density, so fixed field-cell thresholds scale by two
            // to preserve their source-space footprint. Relative width/aspect rules are unchanged.
            // A ribbon has independent detector topology evidence and may span the field.
            uint maximum_width = LocatorContentWidth() * V2_SUBTITLE_LOCATOR_MAX_WIDTH_NUMERATOR /
                V2_SUBTITLE_LOCATOR_MAX_WIDTH_DENOMINATOR;
            uint corner_edge_threshold =
                LocatorContentWidth() / V2_SUBTITLE_LOCATOR_CORNER_EDGE_DIVISOR;
            uint left_clearance = core.x - locator_content.x;
            uint right_clearance = locator_content.z - core.z;
            // Persistent player branding is commonly both bottom-attached and tucked into a
            // horizontal corner. Reject only ordinary cores strictly inside that symmetric edge
            // threshold and within the final ROI rows. Equality at the edge threshold remains
            // eligible, and ribbon topology is never filtered by this ordinary-core rule.
            bool bottom_corner_ordinary = kind == 0u &&
                min(left_clearance, right_clearance) < corner_edge_threshold &&
                core.w + V2_SUBTITLE_LOCATOR_CORNER_BOTTOM_ROWS * field_cell_scale >=
                    locator_field.w;
            if (field_cell_scale != 0u &&
                width >= V2_SUBTITLE_LOCATOR_MIN_WIDTH_CELLS * field_cell_scale &&
                !bottom_corner_ordinary &&
                (kind != 0u || width <= maximum_width) &&
                height >= V2_SUBTITLE_LOCATOR_MIN_HEIGHT_CELLS * field_cell_scale &&
                width * V2_SUBTITLE_LOCATOR_MIN_ASPECT_DENOMINATOR >=
                    V2_SUBTITLE_LOCATOR_MIN_ASPECT_NUMERATOR * height) {
                QualifiedCores[qualified_count] = core;
                QualifiedCovers[qualified_count] = cover;
                QualifiedKinds[qualified_count] = kind;
                if (kind != 0u) ribbon_mask |= 1u << qualified_count;
                ++qualified_count;
            }
        }
    }
    if (qualified_count == 0u) return 0u;

    [loop]
    for (uint index = 0u; index < V2_OCR_FINAL_BOX_CAPACITY; ++index) {
        QualifiedMasks[index] = index < qualified_count && QualifiedKinds[index] == 0u ?
            (1u << index) : 0u;
    }
    [loop]
    for (uint closure_pass = 0u; closure_pass < V2_OCR_FINAL_BOX_CAPACITY; ++closure_pass) {
        [loop]
        for (uint a = 0u; a < V2_OCR_FINAL_BOX_CAPACITY; ++a) {
            if (a < qualified_count) {
                uint expanded = QualifiedMasks[a];
                [loop]
                for (uint b = 0u; b < V2_OCR_FINAL_BOX_CAPACITY; ++b) {
                    if (b < qualified_count && QualifiedKinds[b] == 0u &&
                        (expanded & (1u << b)) != 0u) {
                        [loop]
                        for (uint c = 0u; c < V2_OCR_FINAL_BOX_CAPACITY; ++c) {
                            if (c < qualified_count && QualifiedKinds[c] == 0u &&
                                (CoherentLines(QualifiedCores[b], QualifiedCores[c]) ||
                                 SameBaselineSegments(QualifiedCores[b], QualifiedCores[c]))) {
                                expanded |= 1u << c;
                            }
                        }
                    }
                }
                QualifiedMasks[a] = expanded;
            }
        }
    }

    uint best_mask = 0u;
    uint best_area = 0u;
    uint4 best_bbox = uint4(0u, 0u, 0u, 0u);
    [loop]
    for (uint root = 0u; root < V2_OCR_FINAL_BOX_CAPACITY; ++root) {
        if (root < qualified_count && QualifiedKinds[root] == 0u) {
            uint mask = QualifiedMasks[root];
            // Evaluate each connected component once and reject unsupported >4-line blocks.
            if ((mask & ((1u << root) - 1u)) == 0u) {
                uint count = 0u;
                uint area = 0u;
                uint4 bbox = uint4(0xffffffffu, 0xffffffffu, 0u, 0u);
                [loop]
                for (uint index = 0u; index < V2_OCR_FINAL_BOX_CAPACITY; ++index) {
                    if (index < qualified_count && (mask & (1u << index)) != 0u) {
                        uint4 rectangle = QualifiedCores[index];
                        ++count;
                        area += RectArea(rectangle);
                        bbox.x = min(bbox.x, rectangle.x);
                        bbox.y = min(bbox.y, rectangle.y);
                        bbox.z = max(bbox.z, rectangle.z);
                        bbox.w = max(bbox.w, rectangle.w);
                    }
                }
                uint maximum_width = LocatorContentWidth() *
                    V2_SUBTITLE_LOCATOR_MAX_WIDTH_NUMERATOR /
                    V2_SUBTITLE_LOCATOR_MAX_WIDTH_DENOMINATOR;
                bool better = count <= MAX_LINES && bbox.z - bbox.x <= maximum_width &&
                    (best_mask == 0u || area > best_area ||
                    (area == best_area && (bbox.w > best_bbox.w ||
                    (bbox.w == best_bbox.w && (bbox.y > best_bbox.y ||
                    (bbox.y == best_bbox.y && bbox.x < best_bbox.x))))));
                if (better) {
                    best_mask = mask;
                    best_area = area;
                    best_bbox = bbox;
                }
            }
        }
    }
    uint selected_mask = best_mask | ribbon_mask;
    if (selected_mask == 0u) return 0u;

    uint selected_count = 0u;
    [loop]
    for (uint selected_index = 0u; selected_index < V2_OCR_FINAL_BOX_CAPACITY; ++selected_index) {
        if (selected_index < qualified_count && (selected_mask & (1u << selected_index)) != 0u) {
            ++selected_count;
        }
    }
    // The compact authenticated state has four rectangles.  Never silently discard a detected
    // ribbon or an ordinary line to make a mixed owner fit: an over-capacity observation abstains.
    if (selected_count > MAX_LINES) return 0u;

    uint stack_count = 0u;
    [loop]
    for (uint collect_index = 0u; collect_index < V2_OCR_FINAL_BOX_CAPACITY; ++collect_index) {
        if (collect_index < qualified_count && (selected_mask & (1u << collect_index)) != 0u) {
            WorkRects[STACK_BASE + stack_count] = QualifiedCores[collect_index];
            WorkKinds[STACK_BASE + stack_count] = QualifiedKinds[collect_index];
            StackCovers[stack_count] = QualifiedCovers[collect_index];
            ++stack_count;
        }
    }
    // Canonical top/left order makes temporal comparison and dumps deterministic.
    [unroll]
    for (uint sort_a = 0u; sort_a < MAX_LINES; ++sort_a) {
        [unroll]
        for (uint sort_b = sort_a + 1u; sort_b < MAX_LINES; ++sort_b) {
            if (sort_b < stack_count) {
                uint4 left = WorkRects[STACK_BASE + sort_a];
                uint4 right = WorkRects[STACK_BASE + sort_b];
                if (right.y < left.y || (right.y == left.y && right.x < left.x)) {
                    WorkRects[STACK_BASE + sort_a] = right;
                    WorkRects[STACK_BASE + sort_b] = left;
                    uint4 cover_swap = StackCovers[sort_a];
                    StackCovers[sort_a] = StackCovers[sort_b];
                    StackCovers[sort_b] = cover_swap;
                    uint kind_swap = WorkKinds[STACK_BASE + sort_a];
                    WorkKinds[STACK_BASE + sort_a] = WorkKinds[STACK_BASE + sort_b];
                    WorkKinds[STACK_BASE + sort_b] = kind_swap;
                }
            }
        }
    }
    return stack_count;
}

bool ValidateAndAccumulateRoiRectangle(
    uint4 rectangle, uint slot, uint count, inout uint4 bbox, inout uint area) {
    if (slot >= count) return ZeroRect(rectangle);
    if (!ValidRoiRect(rectangle)) return false;
    if (slot == 0u) bbox = rectangle;
    else {
        bbox.x = min(bbox.x, rectangle.x);
        bbox.y = min(bbox.y, rectangle.y);
        bbox.z = max(bbox.z, rectangle.z);
        bbox.w = max(bbox.w, rectangle.w);
    }
    area += RectArea(rectangle);
    return true;
}

bool ValidateCurrentRectangle(uint4 rectangle, uint slot, uint count, uint kinds) {
    if (slot >= count) return ZeroRect(rectangle);
    bool ribbon = ((kinds >> slot) & 1u) != 0u;
    return ribbon ?
        (ValidFieldRect(rectangle) && rectangle.x == locator_content.x &&
         rectangle.z == locator_content.z && rectangle.w == locator_content.w &&
        rectangle.y >= locator_field.z) : ValidRoiRect(rectangle);
}

bool CanonicalCoreOrder(uint4 previous, uint4 current) {
    return current.y > previous.y || (current.y == previous.y && current.x >= previous.x);
}

bool ValidatePreviousRectBlock(uint offset, uint count, uint4 expected_bbox, uint expected_area) {
    uint4 bbox = uint4(0u, 0u, 0u, 0u);
    uint area = 0u;
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        uint base = offset + slot * 4u;
        uint4 rectangle = uint4(
            PreviousState[base + 0u], PreviousState[base + 1u],
            PreviousState[base + 2u], PreviousState[base + 3u]);
        if (!ValidateAndAccumulateRoiRectangle(
                rectangle, slot, count, bbox, area)) return false;
        if (slot > 0u && slot < count) {
            uint previous_base = base - 4u;
            uint4 previous = uint4(
                PreviousState[previous_base + 0u], PreviousState[previous_base + 1u],
                PreviousState[previous_base + 2u], PreviousState[previous_base + 3u]);
            if (!CanonicalCoreOrder(previous, rectangle)) return false;
        }
    }
    return all(bbox == expected_bbox) && area == expected_area;
}

bool ValidatePreviousCurrent(uint count) {
    uint kinds = (PreviousState[V2_SUBTITLE_LOCATOR_KIND_WORD] >>
                  V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT) & V2_SUBTITLE_LOCATOR_KIND_MASK;
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        uint base = V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + slot * 4u;
        uint4 rectangle = uint4(
            PreviousState[base + 0u], PreviousState[base + 1u],
            PreviousState[base + 2u], PreviousState[base + 3u]);
        if (!ValidateCurrentRectangle(rectangle, slot, count, kinds)) return false;
    }
    return true;
}

uint KindMaskForCount(uint count) {
    return count == 0u ? 0u : (1u << count) - 1u;
}

bool PackedKindsValid(uint packed, uint shift, uint count) {
    uint kinds = (packed >> shift) & V2_SUBTITLE_LOCATOR_KIND_MASK;
    return (kinds & ~KindMaskForCount(count)) == 0u;
}

bool ValidatePreviousState() {
    if (!AdaptiveTailValid(true)) return false;
    uint flags = PreviousState[2u];
    uint owner_count = PreviousState[4u];
    uint pending_count = PreviousState[12u];
    uint current_count = PreviousState[20u];
    bool owner = (flags & FLAG_OWNER) != 0u;
    bool pending = (flags & FLAG_PENDING) != 0u;
    bool target_valid = (flags & FLAG_TARGET_VALID) != 0u;
    bool target_reset = (flags & FLAG_TARGET_RESET) != 0u;
    bool provisional_current = (flags & FLAG_PROVISIONAL_CURRENT) != 0u;
    uint packed_kinds = PreviousState[V2_SUBTITLE_LOCATOR_KIND_WORD];
    uint known_kind_bits =
        (V2_SUBTITLE_LOCATOR_KIND_MASK << V2_SUBTITLE_LOCATOR_OWNER_KIND_SHIFT) |
        (V2_SUBTITLE_LOCATOR_KIND_MASK << V2_SUBTITLE_LOCATOR_PENDING_KIND_SHIFT) |
        (V2_SUBTITLE_LOCATOR_KIND_MASK << V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT);
    if (PreviousState[0u] != V2_SUBTITLE_LOCATOR_STATE_SCHEMA ||
        PreviousState[1u] != V2_SUBTITLE_LOCATOR_STATE_TAG || (flags & ~KNOWN_FLAGS) != 0u ||
        owner_count > MAX_LINES || pending_count > MAX_LINES || current_count > MAX_LINES ||
        PreviousState[21u] > EVENT_HANDOFF || PreviousState[24u] > 2u ||
        PreviousState[27u] != locator_field.x || PreviousState[28u] != locator_field.y ||
        (packed_kinds & ~known_kind_bits) != 0u ||
        !PackedKindsValid(packed_kinds, V2_SUBTITLE_LOCATOR_OWNER_KIND_SHIFT, owner_count) ||
        !PackedKindsValid(packed_kinds, V2_SUBTITLE_LOCATOR_PENDING_KIND_SHIFT, pending_count) ||
        !PackedKindsValid(packed_kinds, V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT, current_count) ||
        owner != (owner_count != 0u) ||
        pending != (pending_count != 0u) || owner != (PreviousState[3u] != 0u) ||
        (current_count != 0u && (!owner || !target_valid || current_count > owner_count))) {
        return false;
    }
    if (!ValidatePreviousRectBlock(
            V2_SUBTITLE_LOCATOR_OWNER_OFFSET, owner_count,
            uint4(PreviousState[5u], PreviousState[6u], PreviousState[7u], PreviousState[8u]),
            PreviousState[9u]) ||
        !ValidatePreviousRectBlock(
            V2_SUBTITLE_LOCATOR_PENDING_OFFSET, pending_count,
            uint4(PreviousState[13u], PreviousState[14u], PreviousState[15u], PreviousState[16u]),
            PreviousState[17u]) || !ValidatePreviousCurrent(current_count)) {
        return false;
    }
    float target = asfloat(PreviousState[18u]);
    uint lifetime_count = PreviousState[25u];
    if (owner) {
        if (lifetime_count != 0u) return false;
        if (provisional_current) {
            uint owner_kinds = (packed_kinds >> V2_SUBTITLE_LOCATOR_OWNER_KIND_SHIFT) &
                V2_SUBTITLE_LOCATOR_KIND_MASK;
            uint pending_kinds = (packed_kinds >> V2_SUBTITLE_LOCATOR_PENDING_KIND_SHIFT) &
                V2_SUBTITLE_LOCATOR_KIND_MASK;
            uint current_kinds = (packed_kinds >> V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT) &
                V2_SUBTITLE_LOCATOR_KIND_MASK;
            uint4 pending_core = uint4(
                PreviousState[V2_SUBTITLE_LOCATOR_PENDING_OFFSET + 0u],
                PreviousState[V2_SUBTITLE_LOCATOR_PENDING_OFFSET + 1u],
                PreviousState[V2_SUBTITLE_LOCATOR_PENDING_OFFSET + 2u],
                PreviousState[V2_SUBTITLE_LOCATOR_PENDING_OFFSET + 3u]);
            uint4 current_cover = uint4(
                PreviousState[V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + 0u],
                PreviousState[V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + 1u],
                PreviousState[V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + 2u],
                PreviousState[V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + 3u]);
            uint4 owner_core = uint4(
                PreviousState[V2_SUBTITLE_LOCATOR_OWNER_OFFSET + 0u],
                PreviousState[V2_SUBTITLE_LOCATOR_OWNER_OFFSET + 1u],
                PreviousState[V2_SUBTITLE_LOCATOR_OWNER_OFFSET + 2u],
                PreviousState[V2_SUBTITLE_LOCATOR_OWNER_OFFSET + 3u]);
            if (!pending || !target_valid || target_reset || owner_count != 1u ||
                pending_count != 1u || current_count != 1u || owner_kinds != 0u ||
                pending_kinds != 0u || current_kinds != 0u ||
                PreviousState[21u] != EVENT_NONE || PreviousState[24u] != 2u ||
                lifetime_count != 0u || current_cover.x > pending_core.x ||
                current_cover.y > pending_core.y || current_cover.z < pending_core.z ||
                current_cover.w < pending_core.w ||
                !ProvisionalSingleLineReplacementRects(owner_core, pending_core) ||
                !SubtitleTargetIsValid(asfloat(
                    PreviousState[V2_SUBTITLE_LOCATOR_PROVISIONAL_TARGET_WORD])) ||
                PreviousState[V2_SUBTITLE_LOCATOR_PROVISIONAL_FADE_WORD] != 2u) {
                return false;
            }
        } else if (PreviousState[V2_SUBTITLE_LOCATOR_PROVISIONAL_TARGET_WORD] != 0u ||
                   PreviousState[V2_SUBTITLE_LOCATOR_PROVISIONAL_FADE_WORD] != 0u) {
            return false;
        }
        if (target_valid) {
            if (target_reset || PreviousState[19u] != PreviousState[3u] ||
                !SubtitleTargetIsValid(target) ||
                PreviousState[24u] != 2u) return false;
        } else if (target_reset) {
            if (lifetime_count != 0u || PreviousState[18u] != 0u ||
                PreviousState[19u] != 0u || current_count != 0u ||
                PreviousState[24u] != 0u) return false;
        } else if (lifetime_count != 0u || PreviousState[18u] != 0u ||
                   PreviousState[19u] != 0u || current_count != 0u ||
                   PreviousState[24u] != 0u) return false;
    } else if (lifetime_count == 0u) {
        if (provisional_current || target_valid || target_reset || PreviousState[18u] != 0u ||
            PreviousState[19u] != 0u || PreviousState[29u] != 0u ||
            PreviousState[30u] != 0u || current_count != 0u || PreviousState[24u] != 0u) {
            return false;
        }
    } else {
        if (provisional_current || lifetime_count > DEATH_GRACE_OBSERVATIONS) return false;
        uint4 bounds = uint4(
            PreviousState[29u] & 0xffffu, PreviousState[30u] & 0xffffu,
            PreviousState[29u] >> 16u, PreviousState[30u] >> 16u);
        if (target_valid || target_reset || PreviousState[19u] != 0u ||
            !SubtitleTargetIsValid(target) ||
            !ValidRoiRect(bounds) || current_count != 0u || PreviousState[24u] != 0u) return false;
    }
    return true;
}

void LoadPreviousRects(uint state_offset, uint kind_shift, uint work_base, uint count) {
    uint kinds = (PreviousState[V2_SUBTITLE_LOCATOR_KIND_WORD] >> kind_shift) &
        V2_SUBTITLE_LOCATOR_KIND_MASK;
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        uint offset = state_offset + slot * 4u;
        WorkRects[work_base + slot] = slot < count ? uint4(
            PreviousState[offset + 0u], PreviousState[offset + 1u],
            PreviousState[offset + 2u], PreviousState[offset + 3u]) :
            uint4(0u, 0u, 0u, 0u);
        WorkKinds[work_base + slot] = slot < count ? ((kinds >> slot) & 1u) : 0u;
    }
}

bool StackCompatible(uint first_base, uint first_count, uint second_base, uint second_count) {
    if (first_count == 0u || first_count != second_count) return false;
    [unroll]
    for (uint index = 0u; index < MAX_LINES; ++index) {
        if (index < first_count) {
            if (WorkKinds[first_base + index] != WorkKinds[second_base + index]) return false;
            // A stack transaction confirms every member, not merely enough aggregate overlap.
            // Otherwise two unchanged lines can lend their IoU to a newly disjoint third line and
            // give that one-observation geometry immediate owner authority.
            if (RectIou(WorkRects[first_base + index], WorkRects[second_base + index]) <
                V2_SUBTITLE_LOCATOR_MATCH_IOU_THRESHOLD) {
                return false;
            }
        }
    }
    return true;
}

bool StackExactlyEqual(uint first_base, uint first_count, uint second_base, uint second_count) {
    if (first_count == 0u || first_count != second_count) return false;
    [unroll]
    for (uint index = 0u; index < MAX_LINES; ++index) {
        if (index < first_count &&
            (WorkKinds[first_base + index] != WorkKinds[second_base + index] ||
             any(WorkRects[first_base + index] != WorkRects[second_base + index]))) {
            return false;
        }
    }
    return true;
}

bool StackCoverExactlyEqualsPreviousCurrent(uint stack_index, uint current_index) {
    uint base = V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + current_index * 4u;
    return all(StackCovers[stack_index] == uint4(
        PreviousState[base + 0u], PreviousState[base + 1u],
        PreviousState[base + 2u], PreviousState[base + 3u]));
}

bool ProvisionalSingleLineReplacementRects(uint4 old_core, uint4 current_core) {
    uint old_height = old_core.w - old_core.y;
    uint current_height = current_core.w - current_core.y;
    uint shorter_height = min(old_height, current_height);
    uint taller_height = max(old_height, current_height);
    uint vertical_overlap = min(old_core.w, current_core.w) >
            max(old_core.y, current_core.y) ?
        min(old_core.w, current_core.w) - max(old_core.y, current_core.y) : 0u;
    uint center_y_delta_twice = (old_core.y + old_core.w) >
            (current_core.y + current_core.w) ?
        (old_core.y + old_core.w) - (current_core.y + current_core.w) :
        (current_core.y + current_core.w) - (old_core.y + old_core.w);
    uint old_center_x_twice = old_core.x + old_core.z;
    uint current_center_x_twice = current_core.x + current_core.z;
    return RectIou(old_core, current_core) < V2_SUBTITLE_LOCATOR_MATCH_IOU_THRESHOLD &&
        vertical_overlap *
            V2_SUBTITLE_LOCATOR_PROVISIONAL_MIN_VERTICAL_OVERLAP_DENOMINATOR >=
        shorter_height * V2_SUBTITLE_LOCATOR_PROVISIONAL_MIN_VERTICAL_OVERLAP_NUMERATOR &&
        taller_height <=
            V2_SUBTITLE_LOCATOR_PROVISIONAL_MAX_HEIGHT_RATIO * shorter_height &&
        center_y_delta_twice <=
            V2_SUBTITLE_LOCATOR_PROVISIONAL_MAX_CENTER_Y_DELTA_SHORTER_HEIGHT *
                shorter_height &&
        old_center_x_twice >= 2u * current_core.x &&
        old_center_x_twice < 2u * current_core.z &&
        current_center_x_twice >= 2u * old_core.x &&
        current_center_x_twice < 2u * old_core.z;
}

bool ProvisionalSingleLineReplacement(uint old_base, uint current_base) {
    return WorkKinds[old_base] == 0u && WorkKinds[current_base] == 0u &&
        ProvisionalSingleLineReplacementRects(
            WorkRects[old_base], WorkRects[current_base]);
}

uint MatchCurrentToOwner(uint current_count, uint owner_count) {
    uint used = 0u;
    uint matched = 0u;
    [unroll]
    for (uint current_index = 0u; current_index < MAX_LINES; ++current_index) {
        if (current_index < current_count) {
            float best_iou = V2_SUBTITLE_LOCATOR_MATCH_IOU_THRESHOLD;
            uint best_owner = MAX_LINES;
            [unroll]
            for (uint owner_index = 0u; owner_index < MAX_LINES; ++owner_index) {
                if (owner_index < owner_count && (used & (1u << owner_index)) == 0u) {
                    if (WorkKinds[STACK_BASE + current_index] !=
                        WorkKinds[OLD_OWNER_BASE + owner_index]) continue;
                    float iou = RectIou(
                        WorkRects[STACK_BASE + current_index], WorkRects[OLD_OWNER_BASE + owner_index]);
                    if (iou >= best_iou) {
                        best_iou = iou;
                        best_owner = owner_index;
                    }
                }
            }
            if (best_owner < MAX_LINES) {
                used |= 1u << best_owner;
                WorkRects[MATCHED_BASE + matched] = WorkRects[STACK_BASE + current_index];
                WorkKinds[MATCHED_BASE + matched] = WorkKinds[STACK_BASE + current_index];
                MatchedCovers[matched] = StackCovers[current_index];
                ++matched;
            }
        }
    }
    [unroll]
    for (uint index = 0u; index < MAX_LINES; ++index) {
        if (index >= matched) {
            WorkRects[MATCHED_BASE + index] = uint4(0u, 0u, 0u, 0u);
            WorkKinds[MATCHED_BASE + index] = 0u;
            MatchedCovers[index] = uint4(0u, 0u, 0u, 0u);
        }
    }
    return matched;
}

void CopyStackCoversToCurrent(uint count) {
    [unroll]
    for (uint index = 0u; index < MAX_LINES; ++index) {
        WorkRects[NEW_CURRENT_BASE + index] = index < count ? StackCovers[index] :
            uint4(0u, 0u, 0u, 0u);
        WorkKinds[NEW_CURRENT_BASE + index] = index < count ?
            WorkKinds[STACK_BASE + index] : 0u;
    }
}

void CopyMatchedCoversToCurrent(uint count) {
    [unroll]
    for (uint index = 0u; index < MAX_LINES; ++index) {
        WorkRects[NEW_CURRENT_BASE + index] = index < count ? MatchedCovers[index] :
            uint4(0u, 0u, 0u, 0u);
        WorkKinds[NEW_CURRENT_BASE + index] = index < count ?
            WorkKinds[MATCHED_BASE + index] : 0u;
    }
}

uint NextGeneration(uint value) {
    return value == 0u || value >= 0xfffffffdu ? 1u : value + 1u;
}

void StoreRectBlock(uint offset, uint base, uint count) {
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        uint4 rectangle = slot < count ? WorkRects[base + slot] : uint4(0u, 0u, 0u, 0u);
        LocatorState[offset + slot * 4u + 0u] = rectangle.x;
        LocatorState[offset + slot * 4u + 1u] = rectangle.y;
        LocatorState[offset + slot * 4u + 2u] = rectangle.z;
        LocatorState[offset + slot * 4u + 3u] = rectangle.w;
    }
}

uint PackKinds(uint base, uint count, uint shift) {
    uint kinds = 0u;
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        if (slot < count) kinds |= (WorkKinds[base + slot] & 1u) << slot;
    }
    return kinds << shift;
}

void PublishState(
    uint owner_generation, uint owner_count, uint pending_count, uint current_count,
    float target, bool target_valid, bool target_reset, uint fade_step, uint lifetime_count,
    uint4 grace_bounds, uint event, uint scene_epoch, bool provisional_current,
    float provisional_target, uint provisional_fade
) {
    uint flags = (owner_count != 0u ? FLAG_OWNER : 0u) |
        (pending_count != 0u ? FLAG_PENDING : 0u) |
        (target_valid ? FLAG_TARGET_VALID : 0u) | (target_reset ? FLAG_TARGET_RESET : 0u) |
        (provisional_current ? FLAG_PROVISIONAL_CURRENT : 0u);
    uint4 owner_bbox = RectSummary(NEW_OWNER_BASE, owner_count);
    uint4 pending_bbox = RectSummary(NEW_PENDING_BASE, pending_count);
    LocatorState[0u] = V2_SUBTITLE_LOCATOR_STATE_SCHEMA;
    LocatorState[1u] = V2_SUBTITLE_LOCATOR_STATE_TAG;
    LocatorState[2u] = flags;
    LocatorState[3u] = owner_count != 0u ? owner_generation : 0u;
    LocatorState[4u] = owner_count;
    LocatorState[5u] = owner_bbox.x;
    LocatorState[6u] = owner_bbox.y;
    LocatorState[7u] = owner_bbox.z;
    LocatorState[8u] = owner_bbox.w;
    LocatorState[9u] = RectAreaSum(NEW_OWNER_BASE, owner_count);
    LocatorState[10u] = locator_frame.z;
    LocatorState[11u] = locator_frame.w;
    LocatorState[12u] = pending_count;
    LocatorState[13u] = pending_bbox.x;
    LocatorState[14u] = pending_bbox.y;
    LocatorState[15u] = pending_bbox.z;
    LocatorState[16u] = pending_bbox.w;
    LocatorState[17u] = RectAreaSum(NEW_PENDING_BASE, pending_count);
    LocatorState[18u] = asuint(target);
    LocatorState[19u] = target_valid ? owner_generation : 0u;
    LocatorState[20u] = target_valid ? current_count : 0u;
    LocatorState[21u] = event;
    LocatorState[22u] = locator_frame.x;
    LocatorState[23u] = locator_frame.y;
    LocatorState[24u] = target_valid ? fade_step : 0u;
    LocatorState[25u] = lifetime_count;
    LocatorState[26u] = scene_epoch;
    LocatorState[27u] = locator_field.x;
    LocatorState[28u] = locator_field.y;
    LocatorState[V2_SUBTITLE_LOCATOR_PROVISIONAL_TARGET_WORD] = provisional_current ?
        asuint(provisional_target) :
        (owner_count == 0u && lifetime_count != 0u ?
            (grace_bounds.z << 16u) | grace_bounds.x : 0u);
    LocatorState[V2_SUBTITLE_LOCATOR_PROVISIONAL_FADE_WORD] = provisional_current ?
        provisional_fade :
        (owner_count == 0u && lifetime_count != 0u ?
            (grace_bounds.w << 16u) | grace_bounds.y : 0u);
    LocatorState[V2_SUBTITLE_LOCATOR_KIND_WORD] =
        PackKinds(NEW_OWNER_BASE, owner_count, V2_SUBTITLE_LOCATOR_OWNER_KIND_SHIFT) |
        PackKinds(NEW_PENDING_BASE, pending_count, V2_SUBTITLE_LOCATOR_PENDING_KIND_SHIFT) |
        PackKinds(NEW_CURRENT_BASE, target_valid ? current_count : 0u,
                  V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT);
    StoreRectBlock(V2_SUBTITLE_LOCATOR_OWNER_OFFSET, NEW_OWNER_BASE, owner_count);
    StoreRectBlock(V2_SUBTITLE_LOCATOR_PENDING_OFFSET, NEW_PENDING_BASE, pending_count);
    StoreRectBlock(
        V2_SUBTITLE_LOCATOR_CURRENT_OFFSET, NEW_CURRENT_BASE,
        target_valid ? current_count : 0u);
    [unroll]
    for (uint word = 0u; word < V2_SUBTITLE_LOCATOR_ADAPTIVE_WORD_COUNT; ++word) {
        LocatorState[ADAPTIVE + word] = AdaptiveReset == 0u ?
            PreviousState[ADAPTIVE + word] : 0u;
    }
}

void ResolveObservation() {
    AdaptiveReset = 1u;
    AdaptiveDistinct = 1u;
    AdaptiveOcrValid = 0u;
    AdaptiveDomainValid = 0u;
    AdaptiveGeometryReady = 0u;
    [loop]
    for (uint index = 0u; index < V2_SUBTITLE_LOCATOR_STATE_WORD_COUNT; ++index) {
        PreviousState[index] = LocatorState[index];
    }

    bool cut_valid = asuint(SBS_STATE_CUT_CONTRACT_TAG_BITS(
        CutBridge[SBS_STATE_VECTOR_CUT_CONTRACT_TAG_BITS])) == SBS_CUT_CONTRACT_TAG;
    uint scene_epoch = cut_valid ? asuint(SBS_STATE_HARD_CUT_COUNT(
        CutBridge[SBS_STATE_VECTOR_HARD_CUT_COUNT])) : 0u;
    // A no-submit/abstaining observation still has a valid source/field domain in which the
    // previous owner lifetime can age. Only current OCR authority requires locator_source.z == 1.
    bool locator_domain_valid = LocatorDomainGeometryValid();
    bool old_valid = locator_domain_valid && ValidatePreviousState();
    uint old_owner_count = old_valid ? PreviousState[4u] : 0u;
    uint old_pending_count = old_valid ? PreviousState[12u] : 0u;
    if (old_valid) {
        LoadPreviousRects(
            V2_SUBTITLE_LOCATOR_OWNER_OFFSET, V2_SUBTITLE_LOCATOR_OWNER_KIND_SHIFT,
            OLD_OWNER_BASE, old_owner_count);
        LoadPreviousRects(
            V2_SUBTITLE_LOCATOR_PENDING_OFFSET, V2_SUBTITLE_LOCATOR_PENDING_KIND_SHIFT,
            OLD_PENDING_BASE, old_pending_count);
    }
    bool distinct_observation = !old_valid || PreviousState[22u] != locator_frame.x ||
        PreviousState[23u] != locator_frame.y || PreviousState[10u] != locator_frame.z ||
        PreviousState[11u] != locator_frame.w;
    // The one-frame pulse is delivery-local. Infer advances CutBridge and SLR in one coherent
    // publication while reuse freezes both; the durable hard-cut authority is therefore the
    // authenticated epoch transition. Re-reading a retained pulse must never restart the scene.
    bool hard_cut = cut_valid && old_valid && PreviousState[26u] != scene_epoch;
    AdaptiveReset = !old_valid || locator_source.w != 0u || hard_cut ||
        PreviousState[ADAPTIVE] != 1u ? 1u : 0u;
    AdaptiveDistinct = distinct_observation ? 1u : 0u;
    AdaptiveDomainValid = cut_valid && locator_domain_valid ? 1u : 0u;
    AdaptiveGeometryReady = AdaptiveDomainValid != 0u && GeometryPublicationReady() ? 1u : 0u;

    bool old_target_valid = old_valid && (PreviousState[2u] & FLAG_TARGET_VALID) != 0u;
    float old_target = old_target_valid ? asfloat(PreviousState[18u]) : 0.0f;
    uint old_grace = old_valid && old_owner_count == 0u ? PreviousState[25u] : 0u;
    bool old_provisional_current = old_valid &&
        (PreviousState[2u] & FLAG_PROVISIONAL_CURRENT) != 0u;
    uint4 old_grace_bounds = old_grace != 0u ? uint4(
        PreviousState[29u] & 0xffffu, PreviousState[30u] & 0xffffu,
        PreviousState[29u] >> 16u, PreviousState[30u] >> 16u) :
        uint4(0u, 0u, 0u, 0u);
    float old_cached_target = old_grace != 0u ? asfloat(PreviousState[18u]) : 0.0f;

    uint final_count = 0u;
    bool ocr_valid = ValidateOcrRecord(final_count);
    AdaptiveOcrValid = ocr_valid ? 1u : 0u;
    if (!cut_valid || !locator_domain_valid ||
        ((locator_source.w != 0u || hard_cut) && !ocr_valid)) {
        PublishState(0u, 0u, 0u, 0u, 0.0f, false, false, 0u, 0u,
                     uint4(0u, 0u, 0u, 0u), EVENT_NONE, scene_epoch,
                     false, 0.0f, 0u);
        return;
    }
    if (!ocr_valid) {
        // Missing, stale, abstaining, and malformed OCR have no current geometry and can never
        // confirm pending state. They are nevertheless a missed observation of an otherwise valid
        // owner lifetime: clear pending, cache the bounded target, and age its ordinary six-step
        // death grace. A hard cut above remains an explicit scene boundary, so invalid OCR on that
        // boundary cannot carry the old plane into the new scene.
        uint grace = 0u;
        uint4 grace_bounds = uint4(0u, 0u, 0u, 0u);
        float cached_target = 0.0f;
        uint event = EVENT_NONE;
        if (old_owner_count != 0u && old_target_valid) {
            grace = DEATH_GRACE_OBSERVATIONS;
            grace_bounds = RectSummary(OLD_OWNER_BASE, old_owner_count);
            cached_target = old_target;
            event = EVENT_DEATH;
        } else if (old_grace != 0u) {
            grace = distinct_observation ? old_grace - 1u : old_grace;
            if (grace != 0u) {
                grace_bounds = old_grace_bounds;
                cached_target = old_cached_target;
            }
        }
        PublishState(0u, 0u, 0u, 0u, cached_target, false, false, 0u, grace,
                     grace_bounds, event, scene_epoch, false, 0.0f, 0u);
        return;
    }
    uint stack_count = BuildCurrentStack(final_count);

    uint new_owner_count = 0u;
    uint new_pending_count = 0u;
    uint authority_count = 0u;
    uint owner_generation = 0u;
    uint event = EVENT_NONE;
    bool provisional_candidate = false;
    float inherited_target = 0.0f;
    uint grace = 0u;
    uint4 grace_bounds = uint4(0u, 0u, 0u, 0u);

    if (locator_source.w != 0u) {
        if (stack_count != 0u) {
            CopyRects(NEW_PENDING_BASE, STACK_BASE, stack_count);
            new_pending_count = stack_count;
        }
    } else if (hard_cut) {
        if (old_owner_count != 0u && stack_count != 0u) {
            uint matched = MatchCurrentToOwner(stack_count, old_owner_count);
            if (matched != 0u) {
                CopyRects(NEW_OWNER_BASE, MATCHED_BASE, matched);
                CopyMatchedCoversToCurrent(matched);
                new_owner_count = matched;
                authority_count = matched;
                // A cut starts a new owner generation even when geometry survives. The old scene
                // UI filter resets below; current covers immediately pin to the reset plane.
                owner_generation = NextGeneration(PreviousState[3u]);
                if (matched < stack_count) {
                    CopyRects(NEW_PENDING_BASE, STACK_BASE, stack_count);
                    new_pending_count = stack_count;
                }
            } else {
                CopyRects(NEW_PENDING_BASE, STACK_BASE, stack_count);
                new_pending_count = stack_count;
                event = EVENT_DEATH;
            }
        } else {
            if (stack_count != 0u) {
                CopyRects(NEW_PENDING_BASE, STACK_BASE, stack_count);
                new_pending_count = stack_count;
            }
            if (old_owner_count != 0u) event = EVENT_DEATH;
        }
    } else if (stack_count == 0u) {
        if (old_owner_count != 0u) {
            event = EVENT_DEATH;
            if (old_target_valid) {
                grace = DEATH_GRACE_OBSERVATIONS;
                grace_bounds = RectSummary(OLD_OWNER_BASE, old_owner_count);
                inherited_target = old_target;
            }
        } else if (old_grace != 0u) {
            grace = distinct_observation ? old_grace - 1u : old_grace;
            if (grace != 0u) {
                grace_bounds = old_grace_bounds;
                inherited_target = old_cached_target;
            }
        }
    } else if (old_owner_count != 0u) {
        uint matched = MatchCurrentToOwner(stack_count, old_owner_count);
        // Established owner continuity has priority over a stale pending transaction. Otherwise a
        // current box that overlaps both tracks can confirm the pending geometry and needlessly
        // bump generation even though every current member still belongs to owner.
        if (matched == stack_count && stack_count <= old_owner_count) {
            CopyRects(NEW_OWNER_BASE, STACK_BASE, stack_count);
            CopyStackCoversToCurrent(stack_count);
            new_owner_count = stack_count;
            authority_count = stack_count;
            owner_generation = PreviousState[3u];
        } else if (old_pending_count != 0u && distinct_observation &&
                   StackCompatible(
                       OLD_PENDING_BASE, old_pending_count, STACK_BASE, stack_count)) {
            CopyRects(NEW_OWNER_BASE, STACK_BASE, stack_count);
            CopyStackCoversToCurrent(stack_count);
            new_owner_count = stack_count;
            authority_count = stack_count;
            owner_generation = NextGeneration(PreviousState[3u]);
            event = EVENT_HANDOFF;
        } else if (old_provisional_current && !distinct_observation &&
                   old_pending_count == 1u && stack_count == 1u &&
                   StackExactlyEqual(OLD_PENDING_BASE, 1u, STACK_BASE, 1u) &&
                   StackCoverExactlyEqualsPreviousCurrent(0u, 0u)) {
            // Re-reading the exact observation that created the provisional bridge neither
            // confirms it nor drops its one-frame verdict. The durable owner remains unchanged.
            CopyRects(NEW_OWNER_BASE, OLD_OWNER_BASE, old_owner_count);
            CopyRects(NEW_PENDING_BASE, STACK_BASE, stack_count);
            CopyStackCoversToCurrent(stack_count);
            CopyRects(MATCHED_BASE, STACK_BASE, stack_count);
            new_owner_count = old_owner_count;
            new_pending_count = stack_count;
            authority_count = stack_count;
            owner_generation = PreviousState[3u];
            provisional_candidate = true;
        } else {
            CopyRects(NEW_OWNER_BASE, OLD_OWNER_BASE, old_owner_count);
            CopyRects(NEW_PENDING_BASE, STACK_BASE, stack_count);
            CopyMatchedCoversToCurrent(matched);
            new_owner_count = old_owner_count;
            new_pending_count = stack_count;
            authority_count = matched;
            owner_generation = PreviousState[3u];
            if (distinct_observation && matched == 0u && old_pending_count == 0u &&
                old_target_valid && PreviousState[20u] == 1u &&
                PreviousState[21u] == EVENT_NONE && PreviousState[24u] == 2u &&
                old_owner_count == 1u && stack_count == 1u &&
                ProvisionalSingleLineReplacement(OLD_OWNER_BASE, STACK_BASE)) {
                // A centered same-baseline one-line replacement may condition only its exact
                // current cover while the ordinary two-observation handoff remains pending.
                CopyStackCoversToCurrent(stack_count);
                CopyRects(MATCHED_BASE, STACK_BASE, stack_count);
                authority_count = stack_count;
                provisional_candidate = true;
            }
        }
    } else {
        bool confirmed = old_pending_count != 0u && distinct_observation &&
            StackCompatible(OLD_PENDING_BASE, old_pending_count, STACK_BASE, stack_count);
        if (confirmed) {
            CopyRects(NEW_OWNER_BASE, STACK_BASE, stack_count);
            CopyStackCoversToCurrent(stack_count);
            new_owner_count = stack_count;
            authority_count = stack_count;
            owner_generation = NextGeneration(0u);
            event = EVENT_BIRTH;
        } else {
            CopyRects(NEW_PENDING_BASE, STACK_BASE, stack_count);
            new_pending_count = stack_count;
            grace = old_grace;
            if (grace != 0u && distinct_observation) --grace;
            if (grace != 0u) {
                grace_bounds = old_grace_bounds;
                inherited_target = old_cached_target;
            }
        }
    }

    bool target_valid = false;
    bool target_reset = false;
    float target = 0.0f;
    uint fade_step = 0u;
    bool provisional_current = false;
    float provisional_target = 0.0f;
    uint provisional_fade = 0u;
    if (new_owner_count != 0u) {
        // OCR ownership/current covers are unchanged. Plane selection is independent of the
        // local-support probe: an authorized cover must not disappear when that probe disagrees.
        target = AdaptiveReset == 0u ? asfloat(PreviousState[ADAPTIVE + 3u]) : 0.0f;
        target_valid = SubtitleTargetIsValid(target);
        fade_step = target_valid ? 2u : 0u;
        provisional_current = provisional_candidate && target_valid && authority_count != 0u;
        provisional_target = provisional_current ? target : 0.0f;
        provisional_fade = provisional_current ? 2u : 0u;
        if (!target_valid) {
            target = 0.0f;
            target_reset = true;
            authority_count = 0u;
        }
    } else if (grace != 0u) {
        target = inherited_target;
    }

    PublishState(
        owner_generation, new_owner_count, new_pending_count, authority_count,
        target, target_valid, target_reset, fade_step,
        new_owner_count != 0u ? 0u : grace,
        grace_bounds, event, scene_epoch, provisional_current,
        provisional_target, provisional_fade);
}

bool ValidateConditionRectBlock(uint offset, uint count, out uint4 bbox, out uint area) {
    bbox = uint4(0u, 0u, 0u, 0u);
    area = 0u;
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        uint base = offset + slot * 4u;
        uint4 rectangle = uint4(
            ConditionStateSnapshot[base + 0u], ConditionStateSnapshot[base + 1u],
            ConditionStateSnapshot[base + 2u], ConditionStateSnapshot[base + 3u]);
        if (!ValidateAndAccumulateRoiRectangle(
                rectangle, slot, count, bbox, area)) return false;
        if (slot > 0u && slot < count) {
            uint previous_base = base - 4u;
            uint4 previous = uint4(
                ConditionStateSnapshot[previous_base + 0u],
                ConditionStateSnapshot[previous_base + 1u],
                ConditionStateSnapshot[previous_base + 2u],
                ConditionStateSnapshot[previous_base + 3u]);
            if (!CanonicalCoreOrder(previous, rectangle)) return false;
        }
    }
    return true;
}

bool ValidateConditionCurrentBlock(uint count) {
    uint kinds = (ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_KIND_WORD] >>
                  V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT) & V2_SUBTITLE_LOCATOR_KIND_MASK;
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        uint base = V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + slot * 4u;
        uint4 rectangle = uint4(
            ConditionStateSnapshot[base + 0u], ConditionStateSnapshot[base + 1u],
            ConditionStateSnapshot[base + 2u], ConditionStateSnapshot[base + 3u]);
        if (!ValidateCurrentRectangle(rectangle, slot, count, kinds)) return false;
    }
    return true;
}

bool ConditionStateValid(out uint current_count, out float target, out uint fade_step) {
    current_count = ConditionStateSnapshot[20u];
    uint flags = ConditionStateSnapshot[2u];
    bool provisional_current = (flags & FLAG_PROVISIONAL_CURRENT) != 0u;
    float durable_target = asfloat(ConditionStateSnapshot[18u]);
    uint durable_fade = ConditionStateSnapshot[24u];
    target = provisional_current ?
        asfloat(ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_PROVISIONAL_TARGET_WORD]) :
        durable_target;
    fade_step = provisional_current ?
        ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_PROVISIONAL_FADE_WORD] : durable_fade;
    uint owner_count = ConditionStateSnapshot[4u];
    uint pending_count = ConditionStateSnapshot[12u];
    uint packed_kinds = ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_KIND_WORD];
    uint known_kind_bits =
        (V2_SUBTITLE_LOCATOR_KIND_MASK << V2_SUBTITLE_LOCATOR_OWNER_KIND_SHIFT) |
        (V2_SUBTITLE_LOCATOR_KIND_MASK << V2_SUBTITLE_LOCATOR_PENDING_KIND_SHIFT) |
        (V2_SUBTITLE_LOCATOR_KIND_MASK << V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT);
    bool cut_valid = asuint(SBS_STATE_CUT_CONTRACT_TAG_BITS(
        CutBridge[SBS_STATE_VECTOR_CUT_CONTRACT_TAG_BITS])) == SBS_CUT_CONTRACT_TAG;
    uint scene_epoch = cut_valid ? asuint(SBS_STATE_HARD_CUT_COUNT(
        CutBridge[SBS_STATE_VECTOR_HARD_CUT_COUNT])) : 0u;
    if (!LocatorGeometryValid() || !GeometryPublicationReady() || !AdaptiveTailValid(false) ||
        !cut_valid || ConditionStateSnapshot[26u] != scene_epoch ||
        ConditionStateSnapshot[0u] != V2_SUBTITLE_LOCATOR_STATE_SCHEMA ||
        ConditionStateSnapshot[1u] != V2_SUBTITLE_LOCATOR_STATE_TAG ||
        (flags & ~KNOWN_FLAGS) != 0u || (flags & FLAG_OWNER) == 0u ||
        (flags & FLAG_TARGET_VALID) == 0u || (flags & FLAG_TARGET_RESET) != 0u ||
        ((flags & FLAG_PENDING) != 0u) != (pending_count != 0u) ||
        owner_count == 0u || owner_count > MAX_LINES || current_count == 0u ||
        current_count > owner_count || pending_count > MAX_LINES ||
        ConditionStateSnapshot[3u] == 0u ||
        ConditionStateSnapshot[19u] != ConditionStateSnapshot[3u] ||
        ConditionStateSnapshot[10u] != locator_frame.z ||
        ConditionStateSnapshot[11u] != locator_frame.w ||
        ConditionStateSnapshot[22u] != locator_frame.x ||
        ConditionStateSnapshot[23u] != locator_frame.y ||
        ConditionStateSnapshot[21u] > EVENT_HANDOFF ||
        durable_fade != 2u ||
        fade_step != 2u ||
        ConditionStateSnapshot[25u] != 0u ||
        ConditionStateSnapshot[27u] != locator_field.x ||
        ConditionStateSnapshot[28u] != locator_field.y ||
        (packed_kinds & ~known_kind_bits) != 0u ||
        !PackedKindsValid(packed_kinds, V2_SUBTITLE_LOCATOR_OWNER_KIND_SHIFT, owner_count) ||
        !PackedKindsValid(packed_kinds, V2_SUBTITLE_LOCATOR_PENDING_KIND_SHIFT, pending_count) ||
        !PackedKindsValid(packed_kinds, V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT, current_count) ||
        !SubtitleTargetIsValid(durable_target) || !SubtitleTargetIsValid(target) ||
        !FiniteFloat(v2_max_horizontal_slope) || v2_max_horizontal_slope < 0.0f ||
        !FiniteFloat(v2_max_vertical_shear) || v2_max_vertical_shear < 0.0f) {
        return false;
    }
    if (ConditionStateSnapshot[ADAPTIVE + 15u] != 0u || durable_fade != 2u ||
         fade_step != 2u || asuint(target) != ConditionStateSnapshot[ADAPTIVE + 3u] ||
         asuint(durable_target) != ConditionStateSnapshot[ADAPTIVE + 3u]) return false;
    uint owner_kinds = (packed_kinds >> V2_SUBTITLE_LOCATOR_OWNER_KIND_SHIFT) &
        V2_SUBTITLE_LOCATOR_KIND_MASK;
    uint pending_kinds = (packed_kinds >> V2_SUBTITLE_LOCATOR_PENDING_KIND_SHIFT) &
        V2_SUBTITLE_LOCATOR_KIND_MASK;
    uint current_kinds = (packed_kinds >> V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT) &
        V2_SUBTITLE_LOCATOR_KIND_MASK;
    if (provisional_current) {
        if ((flags & FLAG_PENDING) == 0u || owner_count != 1u || pending_count != 1u ||
            current_count != 1u || owner_kinds != 0u || pending_kinds != 0u ||
            current_kinds != 0u || ConditionStateSnapshot[21u] != EVENT_NONE ||
            durable_fade != 2u || ConditionStateSnapshot[25u] != 0u) {
            return false;
        }
    } else if (ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_PROVISIONAL_TARGET_WORD] != 0u ||
               ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_PROVISIONAL_FADE_WORD] != 0u) {
        return false;
    }
    uint4 owner_bbox;
    uint owner_area;
    uint4 pending_bbox;
    uint pending_area;
    if (!ValidateConditionRectBlock(
            V2_SUBTITLE_LOCATOR_OWNER_OFFSET, owner_count, owner_bbox, owner_area) ||
        !ValidateConditionRectBlock(
            V2_SUBTITLE_LOCATOR_PENDING_OFFSET, pending_count, pending_bbox, pending_area) ||
        !ValidateConditionCurrentBlock(current_count) ||
        any(owner_bbox != uint4(
            ConditionStateSnapshot[5u], ConditionStateSnapshot[6u],
            ConditionStateSnapshot[7u], ConditionStateSnapshot[8u])) ||
        owner_area != ConditionStateSnapshot[9u] ||
        any(pending_bbox != uint4(
            ConditionStateSnapshot[13u], ConditionStateSnapshot[14u],
            ConditionStateSnapshot[15u], ConditionStateSnapshot[16u])) ||
        pending_area != ConditionStateSnapshot[17u]) {
        return false;
    }
    if (provisional_current) {
        // A provisional verdict is allowed to render only the exact raw-core/final-cover pair
        // selected from this same authenticated OCR8 record. Replaying selection here prevents a
        // corrupted or widened state cover from borrowing the pending core's temporary authority.
        uint final_count = 0u;
        if (!ValidateOcrRecord(final_count) || BuildCurrentStack(final_count) != 1u) return false;
        uint4 pending_core = uint4(
            ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_PENDING_OFFSET + 0u],
            ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_PENDING_OFFSET + 1u],
            ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_PENDING_OFFSET + 2u],
            ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_PENDING_OFFSET + 3u]);
        uint4 owner_core = uint4(
            ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_OWNER_OFFSET + 0u],
            ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_OWNER_OFFSET + 1u],
            ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_OWNER_OFFSET + 2u],
            ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_OWNER_OFFSET + 3u]);
        uint4 current_cover = uint4(
            ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + 0u],
            ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + 1u],
            ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + 2u],
            ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + 3u]);
        if (!ProvisionalSingleLineReplacementRects(owner_core, pending_core) ||
            WorkKinds[STACK_BASE] != 0u || any(WorkRects[STACK_BASE] != pending_core) ||
            any(StackCovers[0u] != current_cover)) {
            return false;
        }
    }
    return true;
}

bool ConditionContentValid() {
    return target_w != 0u && target_h != 0u &&
        target_w == locator_field.x && target_h == locator_field.y &&
        locator_content.x < locator_content.z && locator_content.y < locator_content.w &&
        locator_content.z <= target_w && locator_content.w <= target_h &&
        all(locator_content == DepthAnalysisContentCells());
}

// Authenticate one immutable state snapshot, then publish only the six verdict words consumed by
// the complete out-of-place writer. Invalid authority publishes canonical zero words; the writer
// still visits the full field and copies Base exactly.
void PublishConditionParamsFromSnapshot() {
    uint current_count;
    float target;
    uint fade_step;
    bool state_valid = ConditionStateValid(current_count, target, fade_step);
    uint current_kinds = state_valid ?
        ((ConditionStateSnapshot[V2_SUBTITLE_LOCATOR_KIND_WORD] >>
          V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT) & V2_SUBTITLE_LOCATOR_KIND_MASK) : 0u;

    ConditionParamsOut[CONDITION_PARAM_SCHEMA_WORD] =
        state_valid ? V2_SUBTITLE_CONDITION_PARAM_SCHEMA : 0u;
    ConditionParamsOut[CONDITION_PARAM_TAG_WORD] =
        state_valid ? V2_SUBTITLE_CONDITION_PARAM_TAG : 0u;
    ConditionParamsOut[CONDITION_PARAM_CURRENT_COUNT_WORD] =
        state_valid ? current_count : 0u;
    ConditionParamsOut[CONDITION_PARAM_CURRENT_KINDS_WORD] = current_kinds;
    ConditionParamsOut[CONDITION_PARAM_FADE_STEP_WORD] = state_valid ? fade_step : 0u;
    ConditionParamsOut[CONDITION_PARAM_TARGET_WORD] = state_valid ? asuint(target) : 0u;
}

// The source-timed controller probes the current immutable Base cover union.
float AdaptiveMaximumPlane() {
    // Reserve the representation range for the independently selected UI plane.
    return V2DisplayBudget();
}

float AdaptiveCandidate(uint index, float limit) {
    if (index == AdaptiveMaximumIndex()) return limit;
    return min((4.0f * (float)index) / (float)locator_source.x, limit);
}

bool AdaptiveBudgetPass(uint bad, uint covered, uint content, bool release) {
    if (covered == 0u) return true;
    // Calibrated field sizes keep these integer products below 2^32. Equality passes entry;
    // release deliberately requires both budgets to be strictly below their thresholds.
    return release ? bad * 20u < covered * 3u && bad * 200u < content * 3u :
        bad * 5u <= covered && bad * 50u <= content;
}

bool SourceTimeElapsed(uint2 now, uint2 start, uint minimum_us) {
    if (V2ClockAfter(start, now)) return false;
    uint2 elapsed = V2ClockSubtract(now, start);
    return elapsed.y != 0u || elapsed.x >= minimum_us;
}

void AdaptiveDisarm(inout uint flags, inout uint tail[16]) {
    flags &= ~(ADAPTIVE_APPROACH | ADAPTIVE_RETREAT);
    tail[6u] = 0u;
    tail[7u] = 0u;
    tail[8u] = 0u;
    tail[9u] = 0u;
    tail[10u] = 0u;
    tail[11u] = 0u;
}

void AdaptivePublish(uint tail[16]) {
    // Resolve a complete private tuple before writing persistent UAV state. Timer decisions must
    // never consume intermediate UAV stores from another branch of this same observation.
    [unroll]
    for (uint word = 0u; word < 16u; ++word) LocatorState[ADAPTIVE + word] = tail[word];
    if ((LocatorState[2u] & (FLAG_OWNER | FLAG_TARGET_VALID)) ==
            (FLAG_OWNER | FLAG_TARGET_VALID)) {
        LocatorState[18u] = tail[3u];
        LocatorState[24u] = 2u;
        if ((LocatorState[2u] & FLAG_PROVISIONAL_CURRENT) != 0u) {
            LocatorState[V2_SUBTITLE_LOCATOR_PROVISIONAL_TARGET_WORD] = tail[3u];
            LocatorState[V2_SUBTITLE_LOCATOR_PROVISIONAL_FADE_WORD] = 2u;
        }
    }
}

void AdaptiveAdvance() {
    // Exact redispatches and the ordinary no-dispatch reuse path consume no clock or evidence.
    if (AdaptiveDistinct == 0u && AdaptiveReset == 0u) return;
    uint tail[16];
    [unroll]
    for (uint word = 0u; word < 16u; ++word) tail[word] = LocatorState[ADAPTIVE + word];
    if (AdaptiveDomainValid == 0u) {
        // No calibrated source/cut domain exists in which to label a cap or retain timed evidence.
        // Publish a canonical inactive tail; the empty current mask remains exact Base.
        [unroll]
        for (uint word = 0u; word < V2_SUBTITLE_LOCATOR_ADAPTIVE_WORD_COUNT; ++word) {
            tail[word] = 0u;
        }
        AdaptivePublish(tail);
        return;
    }
    tail[0u] = 1u;
    float limit = AdaptiveMaximumPlane();
    float applied = min(asfloat(tail[3u]), limit);
    uint flags = tail[1u];
    if (asfloat(tail[12u]) != limit) AdaptiveDisarm(flags, tail);
    tail[12u] = asuint(limit);
    tail[3u] = asuint(applied);
    if (AdaptiveGeometryReady == 0u) {
        // Preserve the applied plane/goal and OCR transaction, but discard timing authority.
        // Clearing the accepted clock makes the next ready observation rearm without spending
        // a missing/collapsed/stale geometry interval or carrying a retreat dwell through it.
        AdaptiveDisarm(flags, tail);
        flags &= ~ADAPTIVE_CLOCK;
        tail[4u] = 0u;
        tail[5u] = 0u;
        tail[1u] = flags;
        AdaptivePublish(tail);
        return;
    }
    if (AdaptiveOcrValid == 0u) {
        // An abstention/malformed record supplies no probe, not an invented empty UI mask.
        AdaptiveDisarm(flags, tail);
        tail[1u] = flags;
        AdaptivePublish(tail);
        return;
    }

    uint covered = AdaptiveCounts[0u];
    uint content_area = LocatorContentWidth() * (locator_content.w - locator_content.y);
    uint maximum_index = AdaptiveMaximumIndex();
    uint required = maximum_index;
    uint release_index = maximum_index;
    bool required_found = false;
    bool release_found = false;
    [loop]
    for (uint level = 0u; level <= maximum_index; ++level) {
        uint bad = AdaptiveCounts[2u + level];
        if (!required_found && AdaptiveBudgetPass(bad, covered, content_area, false)) {
            required = level;
            required_found = true;
        }
        if (!release_found && AdaptiveBudgetPass(bad, covered, content_area, true)) {
            release_index = level;
            release_found = true;
        }
    }
    flags = required_found ? flags & ~ADAPTIVE_CAPPED : flags | ADAPTIVE_CAPPED;
    tail[13u] = covered;
    tail[14u] = AdaptiveCounts[2u + maximum_index];
    tail[15u] = AdaptiveCounts[1u];
    uint2 now = locator_observation.xy;
    uint2 last = uint2(tail[4u], tail[5u]);
    bool have_clock = (flags & ADAPTIVE_CLOCK) != 0u;
    if (AdaptiveCounts[1u] != 0u || all(now == 0u) ||
        (have_clock && !V2ClockAfter(now, last))) {
        // No frame count/FPS clock is substituted. Keep the last accepted source time and plane.
        AdaptiveDisarm(flags, tail);
        tail[1u] = flags;
        AdaptivePublish(tail);
        return;
    }

    uint goal = tail[2u];
    if (required > goal) {
        flags &= ~ADAPTIVE_RETREAT;
        tail[8u] = 0u;
        tail[9u] = 0u;
        tail[11u] = 0u;
        if ((flags & ADAPTIVE_APPROACH) == 0u) {
            flags |= ADAPTIVE_APPROACH;
            tail[6u] = now.x;
            tail[7u] = now.y;
            tail[10u] = required;
        } else {
            tail[10u] = min(tail[10u], required);
        }
        uint2 start = uint2(tail[6u], tail[7u]);
        if (SourceTimeElapsed(now, start, 100000u)) {
            goal = tail[10u];
            AdaptiveDisarm(flags, tail);
        }
    } else {
        flags &= ~ADAPTIVE_APPROACH;
        tail[6u] = 0u;
        tail[7u] = 0u;
        tail[10u] = 0u;
        if (release_found && release_index < goal &&
            applied == AdaptiveCandidate(goal, limit)) {
            if ((flags & ADAPTIVE_RETREAT) == 0u) {
                flags |= ADAPTIVE_RETREAT;
                tail[8u] = now.x;
                tail[9u] = now.y;
                tail[11u] = release_index;
            } else {
                tail[11u] = max(tail[11u], release_index);
            }
            uint2 start = uint2(tail[8u], tail[9u]);
            if (SourceTimeElapsed(now, start, 1500000u)) {
                goal = tail[11u];
                AdaptiveDisarm(flags, tail);
            }
        } else {
            flags &= ~ADAPTIVE_RETREAT;
            tail[8u] = 0u;
            tail[9u] = 0u;
            tail[11u] = 0u;
        }
    }
    float target = AdaptiveCandidate(goal, limit);
    if (have_clock) {
        uint2 elapsed = V2ClockSubtract(now, last);
        // Game's live rate integration is capped to 250ms; this does not discard dwell evidence
        // or expire a legitimate slower inference cadence. Both rates are one-eye source U/sec.
        float seconds = elapsed.y != 0u ? 0.25f : min((float)elapsed.x * 1.0e-6f, 0.25f);
        float step = seconds * (target > applied ? 0.03f : 0.005f);
        applied = target > applied ? min(target, applied + step) : max(target, applied - step);
    }
    tail[2u] = goal;
    tail[4u] = now.x;
    tail[5u] = now.y;
    tail[1u] = flags | ADAPTIVE_CLOCK;
    tail[3u] = asuint(applied);
    AdaptivePublish(tail);
}

uint AdaptiveFirstNonconflictingIndex(float desired, float limit, uint maximum_index) {
    if (desired <= 0.0f) return 0u;
    if (desired > limit) return maximum_index + 1u;
    uint index = min((uint)ceil(desired * (float)locator_source.x * 0.25f), maximum_index);
    // The arithmetic estimate can straddle a level by one FP32 ULP. Compare against the actual
    // shader candidates so histogram bins retain the strict conflict predicate.
    if (index != 0u && desired <= AdaptiveCandidate(index - 1u, limit)) --index;
    if (desired > AdaptiveCandidate(index, limit)) ++index;
    return index;
}

void AdaptiveResolveHistogram() {
    uint maximum_index = AdaptiveMaximumIndex();
    uint conflicts = AdaptiveCounts[1u] + AdaptiveCounts[3u + maximum_index];
    // Bucket k names the first candidate that can clear a sample; only buckets above a level
    // conflict with it. The extra bucket above the last level is explicitly unclearable.
    [loop]
    for (int level = (int)maximum_index; level >= 0; --level) {
        uint offset = 2u + (uint)level;
        uint bucket = AdaptiveCounts[offset];
        AdaptiveCounts[offset] = conflicts;
        conflicts += bucket;
    }
}

void AdaptiveProbe(uint lane) {
    uint covered = 0u;
    uint invalid = 0u;
    float limit = AdaptiveMaximumPlane();
    float clearance = 2.0f / (float)locator_source.x;
    uint maximum_index = AdaptiveMaximumIndex();
    uint pending_bucket = 0xffffffffu;
    uint pending_count = 0u;
    uint current_count = AdaptiveCurrentCount;
    [loop]
    for (uint slot = 0u; slot < current_count; ++slot) {
        uint4 rectangle = AdaptiveCurrentCovers[slot];
        uint width = rectangle.z - rectangle.x;
        uint area = width * (rectangle.w - rectangle.y);
        // Keep each lane's original cell, cell+256, ... row-major visits. Derive the initial
        // coordinate and stride once; add/carry avoids dynamic division for every covered cell.
        uint position_y = lane / width;
        uint position_x = lane - position_y * width;
        uint stride_y = 256u / width;
        uint stride_x = 256u - stride_y * width;
        [loop]
        for (uint cell = lane; cell < area; cell += 256u) {
            uint2 position = rectangle.xy + uint2(position_x, position_y);
            position_x += stride_x;
            position_y += stride_y;
            if (position_x >= width) {
                position_x -= width;
                ++position_y;
            }
            bool already_counted = false;
            [unroll]
            for (uint earlier = 0u; earlier < MAX_LINES; ++earlier) {
                if (earlier < slot) {
                    uint4 prior = AdaptiveCurrentCovers[earlier];
                    already_counted = already_counted ||
                        (position.x >= prior.x && position.y >= prior.y &&
                         position.x < prior.z && position.y < prior.w);
                }
            }
            if (already_counted) continue;
            float base = BaseField.Load(int3(position, 0));
            bool finite = FiniteFloat(base) && abs(base) <= v2_direct_container_limit;
            ++covered;
            if (!finite) ++invalid;
            if (finite) {
                precise float desired = base + clearance;
                uint bucket = AdaptiveFirstNonconflictingIndex(desired, limit, maximum_index);
                if (bucket != pending_bucket && pending_count != 0u) {
                    InterlockedAdd(AdaptiveCounts[2u + pending_bucket], pending_count);
                    pending_count = 0u;
                }
                pending_bucket = bucket;
                ++pending_count;
            }
        }
    }
    InterlockedAdd(AdaptiveCounts[0u], covered);
    InterlockedAdd(AdaptiveCounts[1u], invalid);
    if (pending_count != 0u) {
        InterlockedAdd(AdaptiveCounts[2u + pending_bucket], pending_count);
    }
}

// The helper ensures every reset/abstention return comes back here before the UAV visibility barrier.
// The complete just-written state is then snapshotted and authenticated without aliasing u2 as an
// SRV in the same dispatch.
[numthreads(256, 1, 1)]
void resolve_main(uint3 dispatch_id : SV_DispatchThreadID, uint lane : SV_GroupIndex) {
    if (lane == 0u) ResolveObservation();
    AllMemoryBarrierWithGroupSync();
    if (lane < ADAPTIVE_COUNT_WORDS) AdaptiveCounts[lane] = 0u;
    // Snapshot the just-published authenticated current covers once per group. Reuse and
    // redispatch retain their existing observation gates; this scratch grants no authority.
    if (lane == 0u) AdaptiveCurrentCount = LocatorState[20u];
    if (lane < MAX_LINES) {
        uint offset = V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + lane * 4u;
        AdaptiveCurrentCovers[lane] = uint4(
            LocatorState[offset], LocatorState[offset + 1u],
            LocatorState[offset + 2u], LocatorState[offset + 3u]);
    }
    GroupMemoryBarrierWithGroupSync();
    if (AdaptiveDomainValid != 0u && AdaptiveGeometryReady != 0u && AdaptiveOcrValid != 0u &&
        (AdaptiveDistinct != 0u || AdaptiveReset != 0u)) AdaptiveProbe(lane);
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        if (AdaptiveDomainValid != 0u && AdaptiveGeometryReady != 0u && AdaptiveOcrValid != 0u &&
            (AdaptiveDistinct != 0u || AdaptiveReset != 0u)) AdaptiveResolveHistogram();
        AdaptiveAdvance();
    }
    AllMemoryBarrierWithGroupSync();
    [loop]
    for (uint index = lane; index < V2_SUBTITLE_LOCATOR_STATE_WORD_COUNT; index += 256u) {
        ConditionStateSnapshot[index] = LocatorState[index];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) PublishConditionParamsFromSnapshot();
}

bool ConditionParamsValid(
    out uint current_count,
    out uint current_kinds,
    out uint fade_step,
    out float target) {
    current_count = ConditionParams[CONDITION_PARAM_CURRENT_COUNT_WORD];
    current_kinds = ConditionParams[CONDITION_PARAM_CURRENT_KINDS_WORD];
    fade_step = ConditionParams[CONDITION_PARAM_FADE_STEP_WORD];
    target = asfloat(ConditionParams[CONDITION_PARAM_TARGET_WORD]);
    return AdaptivePolicyValid() &&
        ConditionParams[CONDITION_PARAM_SCHEMA_WORD] == V2_SUBTITLE_CONDITION_PARAM_SCHEMA &&
        ConditionParams[CONDITION_PARAM_TAG_WORD] == V2_SUBTITLE_CONDITION_PARAM_TAG &&
        current_count != 0u && current_count <= MAX_LINES &&
        (current_kinds & ~V2_SUBTITLE_LOCATOR_KIND_MASK) == 0u &&
        PackedKindsValid(current_kinds, 0u, current_count) &&
        fade_step == 2u &&
        SubtitleTargetIsValid(target) &&
        target <= AdaptiveMaximumPlane();
}

float EvaluateConditionedBase(
    uint2 condition_position,
    float base,
    bool state_valid,
    uint current_count,
    uint current_kinds,
    float condition_target,
    out bool changed) {
    changed = false;
    if (!state_valid || !FiniteFloat(base) || abs(base) > v2_direct_container_limit) {
        return base;
    }
    float best_distance = 3.402823466e+38f;
    precise float horizontal_step = v2_max_horizontal_slope /
        (float)LocatorContentWidth();
    precise float vertical_step = v2_max_vertical_shear /
        (float)LocatorContentWidth();
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        if (slot < current_count) {
            uint offset = V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + slot * 4u;
            uint4 rectangle = uint4(
                LocatorStateRead[offset + 0u], LocatorStateRead[offset + 1u],
                LocatorStateRead[offset + 2u], LocatorStateRead[offset + 3u]);
            bool ribbon = ((current_kinds >> slot) & 1u) != 0u;
            // A canonical ribbon cover spans the full field from its corrected top to the bottom.
            // Make the one-edge policy explicit: the strip and everything below its top use core
            // budget, while only rows above it receive the vertical analytic collar. Ordinary
            // subtitle covers retain the established four-sided Manhattan collar.
            uint dx = ribbon ? 0u :
                (condition_position.x < rectangle.x ? rectangle.x - condition_position.x :
                 (condition_position.x >= rectangle.z ?
                    condition_position.x - (rectangle.z - 1u) : 0u));
            uint dy = ribbon ?
                (condition_position.y < rectangle.y ? rectangle.y - condition_position.y : 0u) :
                (condition_position.y < rectangle.y ? rectangle.y - condition_position.y :
                 (condition_position.y >= rectangle.w ?
                    condition_position.y - (rectangle.w - 1u) : 0u));
            precise float horizontal_distance = (float)dx * horizontal_step;
            precise float vertical_distance = (float)dy * vertical_step;
            precise float distance = horizontal_distance + vertical_distance;
            if (distance < best_distance) {
                best_distance = distance;
            }
        }
    }
    if (best_distance == 0.0f) {
        // Current coarse OCR covers are the admitted UI mask. Pin every covered cell to exactly
        // one plane, including signed zero, rather than retaining a per-glyph depth slack.
        changed = asuint(base) != asuint(condition_target);
        return condition_target;
    }
    precise float budget = best_distance;
    precise float delta = base - condition_target;
    // Exact Base is a semantic branch, not an algebraic coincidence: bypassing reconstruction
    // avoids changing an already-safe R32_FLOAT bit pattern through target + (base - target).
    if (abs(delta) <= budget) {
        return base;
    }
    precise float full = condition_target + (delta < 0.0f ? -budget : budget);
    changed = true;
    return full;
}

// Sole production writer. It always reads immutable BaseField out of place and writes every output
// cell, so padding extends the nearest real content cell, unchanged Base bits remain exact, and an
// invalid condition verdict still publishes Base without a preparatory texture copy.
[numthreads(16, 16, 1)]
void condition_main(uint3 dispatch_id : SV_DispatchThreadID) {
    if (dispatch_id.x >= target_w || dispatch_id.y >= target_h) return;
    bool content_valid = ConditionContentValid();
    uint2 condition_position = content_valid ?
        DepthAnalysisClampCell(dispatch_id.xy) : dispatch_id.xy;
    uint current_count;
    uint current_kinds;
    uint fade_step;
    float condition_target;
    bool state_valid = ConditionParamsValid(
        current_count, current_kinds, fade_step, condition_target);
    float base = BaseField.Load(int3(condition_position, 0));
    bool changed;
    float conditioned = EvaluateConditionedBase(
        condition_position, base, state_valid, current_count, current_kinds,
        condition_target, changed);
    ConditionedField[dispatch_id.xy] = conditioned;
}
