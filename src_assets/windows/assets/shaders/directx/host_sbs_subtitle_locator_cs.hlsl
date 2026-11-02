// Direct current OCR8 coverage with an independent source-timed shared UI plane.
// Every producer-admitted current cover authorizes the native RGB search and the same Base
// union probe. Pixel confirmation and Gaussian presentation remain separate stages.
#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2_ocr_assert.generated.hlsl"
#include "include/depth_coordinate_v2.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"
#include "include/host_sbs_subtitle_history_compatibility.hlsl"

StructuredBuffer<uint4> GeometryState : register(t0);
StructuredBuffer<float4> CutBridge : register(t1);
Texture2D<float> BaseField : register(t2);
StructuredBuffer<uint> LocatorStateRead : register(t3);
StructuredBuffer<uint> ConditionParams : register(t4);
Texture2D<float> ContourDistance : register(t5);
StructuredBuffer<uint> ContourIdentity : register(t6);
StructuredBuffer<uint> OcrRecord : register(t7);
StructuredBuffer<float> CurrentProbability : register(t8);
StructuredBuffer<float> PreviousProbability : register(t9);
RWStructuredBuffer<uint> LocatorState : register(u2);
RWTexture2D<float> ConditionedField : register(u3);
RWStructuredBuffer<uint> ConditionParamsOut : register(u4);
RWTexture2D<float> GaussianWeightOut : register(u5);

cbuffer SubtitleLocatorConstants : register(b2) {
    uint4 locator_field; // width, height, ROI top, ROI bottom
    uint4 locator_source; // source width, height, current record enabled, domain reset
    uint4 locator_frame; // exact source frame lo/hi, analysis generation lo/hi
    uint4 locator_content;
    uint4 locator_observation; // exact source time lo/hi, zero reserved words
};

static const uint MAX_LINES = V2_SUBTITLE_PLANE_COVER_CAPACITY;
static const uint ADAPTIVE = V2_SUBTITLE_PLANE_ADAPTIVE_OFFSET;
static const uint ADAPTIVE_CAPPED = 4u;
static const uint ADAPTIVE_CLOCK = 8u;
// Survives abstention, unlike the current mask: clock rearm is not a new cue.
static const uint ADAPTIVE_PRESENTED = V2_SUBTITLE_PLANE_ADAPTIVE_PRESENTED_FLAG;
static const uint ADAPTIVE_HOST_MAX_SOURCE_WIDTH = 5120u;
static const uint ADAPTIVE_COUNT_WORDS = 56u;
groupshared uint PreviousState[V2_SUBTITLE_PLANE_STATE_WORD_COUNT];
groupshared uint ConditionStateSnapshot[V2_SUBTITLE_PLANE_STATE_WORD_COUNT];
groupshared uint AdaptiveReset, AdaptiveDistinct, AdaptiveOcrValid;
groupshared uint AdaptiveDomainValid, AdaptiveGeometryReady, AdaptiveCurrentCount;
groupshared uint AdaptiveCounts[ADAPTIVE_COUNT_WORDS];
groupshared uint4 AdaptiveCurrentCovers[MAX_LINES];
groupshared uint ProbabilityResetGate;
groupshared uint ProbabilityResetMask;
groupshared uint4 ProbabilitySource;
groupshared uint4 ProbabilityCoreBounds;
groupshared float2 ProbabilityMass;
groupshared float2 ProbabilitySums[256];
groupshared uint ProbabilityInvalid[256];

static const uint CONDITION_PARAM_SCHEMA_WORD = 0u;
static const uint CONDITION_PARAM_TAG_WORD = 1u;
static const uint CONDITION_PARAM_CURRENT_COUNT_WORD = V2_SUBTITLE_CONDITION_PARAM_CURRENT_COUNT_WORD;
static const uint CONDITION_PARAM_CURRENT_KINDS_WORD = V2_SUBTITLE_CONDITION_PARAM_CURRENT_KIND_MASK_WORD;
static const uint CONDITION_PARAM_ACTIVE_WORD = V2_SUBTITLE_CONDITION_PARAM_ACTIVE_WORD;
static const uint CONDITION_PARAM_TARGET_WORD = V2_SUBTITLE_CONDITION_PARAM_TARGET_WORD;


bool FiniteFloat(float value) {
    return (asuint(value) & 0x7f800000u) != 0x7f800000u;
}

bool AdaptivePolicyValid() {
    return V2JointPlaneConstantsValid() &&
        FiniteFloat(v2_requested_gain) && v2_requested_gain > 0.0f &&
        all(uint2(v2_joint_observation_timestamp_low,
                  v2_joint_observation_timestamp_high) == locator_observation.xy);
}

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

uint StateRead(bool previous, uint word) {
    return previous ? PreviousState[word] : ConditionStateSnapshot[word];
}
uint SlotMask(uint count) { return (1u << count) - 1u; }

bool AdaptiveTailValid(bool previous) {
    bool all_zero = true;
    [unroll]
    for (uint word = 0u; word < V2_SUBTITLE_PLANE_ADAPTIVE_WORD_COUNT; ++word)
        all_zero = all_zero && AdaptiveRead(previous, word) == 0u;
    if (all_zero) return StateRead(previous, V2_SUBTITLE_PLANE_CURRENT_SLOT_MASK_WORD) == 0u;
    uint flags = AdaptiveRead(previous, 1u);
    uint2 clock = uint2(AdaptiveRead(previous, 4u), AdaptiveRead(previous, 5u));
    float applied = asfloat(AdaptiveRead(previous, 3u));
    float limit = asfloat(AdaptiveRead(previous, 12u));
    uint covered = AdaptiveRead(previous, 13u);
    uint content_area = (locator_content.z - locator_content.x) *
        (locator_content.w - locator_content.y);
    return AdaptiveRead(previous, 0u) == 1u && (flags & ~V2_SUBTITLE_PLANE_ADAPTIVE_KNOWN_FLAGS) == 0u &&
        AdaptiveRead(previous, 6u) == 0u && AdaptiveRead(previous, 7u) == 0u &&
        AdaptiveRead(previous, 8u) == 0u && AdaptiveRead(previous, 9u) == 0u &&
        AdaptiveRead(previous, 10u) == 0u && AdaptiveRead(previous, 11u) == 0u &&
        AdaptiveRead(previous, 2u) <= AdaptiveMaximumIndex() &&
        FiniteFloat(applied) && FiniteFloat(limit) && applied >= 0.0f &&
        limit == V2DisplayBudget() && applied <= limit &&
        (((flags & ADAPTIVE_CLOCK) != 0u) == any(clock != 0u)) &&
        covered <= content_area && AdaptiveRead(previous, 14u) <= covered &&
        (((flags & ADAPTIVE_CAPPED) != 0u) ==
          (AdaptiveRead(previous, 14u) * 5u > covered ||
           AdaptiveRead(previous, 14u) * 50u > content_area)) &&
        AdaptiveRead(previous, 15u) <= covered;
}

uint LocatorContentWidth() {
    return locator_content.z - locator_content.x;
}

bool SubtitleTargetIsValid(float target) {
    return FiniteFloat(target) && FiniteFloat(v2_direct_container_limit) &&
        v2_direct_container_limit > 0.0f && target >= 0.0f &&
        target <= v2_direct_container_limit;
}

bool LocatorSourceGeometryValid() {
    if (!AdaptivePolicyValid() ||
        any(locator_observation.zw != 0u) ||
        locator_source.x == 0u || locator_source.y == 0u || locator_source.z > 1u ||
        locator_source.x > ADAPTIVE_HOST_MAX_SOURCE_WIDTH ||
        locator_field.x == 0u || locator_field.y == 0u ||
        locator_field.x > 0xffffu || locator_field.y > 0xffffu ||
        target_w != locator_field.x || target_h != locator_field.y ||
        locator_content.x >= locator_content.z || locator_content.y >= locator_content.w ||
        locator_content.z > locator_field.x || locator_content.w > locator_field.y ||
        any(locator_content != DepthAnalysisContentCells())) {
        return false;
    }

    return V2SubtitleOcrFieldIsCalibrated(locator_field.x, locator_field.y);
}

bool LocatorDomainGeometryValid() {
    if (!LocatorSourceGeometryValid() ||
        locator_field.z >= locator_field.w || locator_field.w > locator_field.y ||
        locator_field.z < locator_content.y || locator_field.w > locator_content.w) return false;

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
    if (!LocatorGeometryValid() || all(locator_frame.xy == 0u)) {
        return false;
    }
    uint words, stride;
    OcrRecord.GetDimensions(words, stride);
    if (words != V2_OCR_RECORD_WORD_COUNT || stride != 4u) return false;
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

uint CurrentKinds(uint count) {
    uint kinds = 0u;
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        uint offset = V2_OCR_FINAL_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
        if (slot < count && (OcrRecord[offset + 5u] & V2_OCR_BOX_FLAG_RIBBON) != 0u)
            kinds |= 1u << slot;
    }
    return kinds;
}

bool StateHeaderValid(bool previous, bool exact_frame) {
    uint flags = StateRead(previous, V2_SUBTITLE_PLANE_FLAGS_WORD);
    uint mask = StateRead(previous, V2_SUBTITLE_PLANE_CURRENT_SLOT_MASK_WORD);
    uint reset_mask = StateRead(previous, V2_SUBTITLE_PLANE_HISTORY_RESET_MASK_WORD);
    uint warmup = StateRead(previous, V2_SUBTITLE_PLANE_WARMUP_WORD);
    if (StateRead(previous, 0u) != V2_SUBTITLE_PLANE_STATE_SCHEMA ||
        StateRead(previous, 1u) != V2_SUBTITLE_PLANE_STATE_TAG ||
        (flags & ~V2_SUBTITLE_PLANE_CURRENT_VALID_FLAG) != 0u ||
        mask > SlotMask(MAX_LINES) || mask != SlotMask(countbits(mask)) ||
        (flags == 0u && mask != 0u) || (reset_mask & ~mask) != 0u ||
        (reset_mask != 0u && (flags != V2_SUBTITLE_PLANE_CURRENT_VALID_FLAG ||
         AdaptiveRead(previous, 0u) != 1u || AdaptiveRead(previous, 15u) != 0u)) ||
        warmup != 0u) return false;
    [unroll]
    for (uint word = V2_SUBTITLE_PLANE_RESERVED_OFFSET;
         word < V2_SUBTITLE_PLANE_HEADER_WORD_COUNT; ++word)
        if (StateRead(previous, word) != 0u) return false;
    if (all(uint2(StateRead(previous, V2_SUBTITLE_PLANE_FRAME_OFFSET),
                  StateRead(previous, V2_SUBTITLE_PLANE_FRAME_OFFSET + 1u)) == 0u)) return false;
    [unroll]
    for (uint component = 0u; component < 2u; ++component) {
        if (StateRead(previous, V2_SUBTITLE_PLANE_ANALYSIS_GENERATION_OFFSET + component) != locator_frame[component + 2u] ||
            StateRead(previous, V2_SUBTITLE_PLANE_SOURCE_EXTENT_OFFSET + component) != locator_source[component] ||
            StateRead(previous, V2_SUBTITLE_PLANE_FIELD_EXTENT_OFFSET + component) != locator_field[component] ||
            StateRead(previous, V2_SUBTITLE_PLANE_ROI_OFFSET + component) != locator_field[component + 2u]) return false;
        if (exact_frame &&
            (StateRead(previous, V2_SUBTITLE_PLANE_FRAME_OFFSET + component) != locator_frame[component] ||
             StateRead(previous, V2_SUBTITLE_PLANE_OBSERVATION_TIMESTAMP_OFFSET + component) != locator_observation[component])) return false;
    }
    [unroll]
    for (uint component = 0u; component < 4u; ++component)
        if (StateRead(previous, V2_SUBTITLE_PLANE_CONTENT_OFFSET + component) != locator_content[component]) return false;
    return AdaptiveTailValid(previous);
}

void ResolveObservation() {
    [loop]
    for (uint word = 0u; word < V2_SUBTITLE_PLANE_STATE_WORD_COUNT; ++word)
        PreviousState[word] = LocatorState[word];
    bool cut_valid = asuint(SBS_STATE_CUT_CONTRACT_TAG_BITS(
        CutBridge[SBS_STATE_VECTOR_CUT_CONTRACT_TAG_BITS])) == SBS_CUT_CONTRACT_TAG;
    uint epoch = cut_valid ? asuint(SBS_STATE_HARD_CUT_COUNT(
        CutBridge[SBS_STATE_VECTOR_HARD_CUT_COUNT])) : 0u;
    bool domain_valid = LocatorDomainGeometryValid();
    bool old_valid = domain_valid && StateHeaderValid(true, false);
    bool hard_cut = old_valid && cut_valid &&
        PreviousState[V2_SUBTITLE_PLANE_CUT_EPOCH_WORD] != epoch;
    AdaptiveReset = !old_valid || locator_source.w != 0u || hard_cut ||
        PreviousState[ADAPTIVE] != 1u ? 1u : 0u;
    AdaptiveDistinct = !old_valid ||
        PreviousState[V2_SUBTITLE_PLANE_FRAME_OFFSET] != locator_frame.x ||
        PreviousState[V2_SUBTITLE_PLANE_FRAME_OFFSET + 1u] != locator_frame.y ? 1u : 0u;
    AdaptiveDomainValid = domain_valid && cut_valid ? 1u : 0u;
    AdaptiveGeometryReady = AdaptiveDomainValid != 0u && GeometryPublicationReady() ? 1u : 0u;
    uint count;
    AdaptiveOcrValid = ValidateOcrRecord(count) ? 1u : 0u;
    bool current_valid = AdaptiveOcrValid != 0u && AdaptiveGeometryReady != 0u &&
        any(locator_frame.xy != 0u);
    AdaptiveCurrentCount = current_valid ? count : 0u;
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        uint offset = V2_OCR_FINAL_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
        AdaptiveCurrentCovers[slot] = slot < AdaptiveCurrentCount ?
            uint4(OcrRecord[offset], OcrRecord[offset + 1u],
                  OcrRecord[offset + 2u], OcrRecord[offset + 3u]) : uint4(0u, 0u, 0u, 0u);
    }
    [loop]
    for (uint word = 0u; word < V2_SUBTITLE_PLANE_STATE_WORD_COUNT; ++word) LocatorState[word] = 0u;
    LocatorState[0u] = V2_SUBTITLE_PLANE_STATE_SCHEMA;
    LocatorState[1u] = V2_SUBTITLE_PLANE_STATE_TAG;
    LocatorState[V2_SUBTITLE_PLANE_FLAGS_WORD] = current_valid ? V2_SUBTITLE_PLANE_CURRENT_VALID_FLAG : 0u;
    LocatorState[V2_SUBTITLE_PLANE_CURRENT_SLOT_MASK_WORD] = SlotMask(AdaptiveCurrentCount);
    // A calibrated depth source can have an unprojectable OCR crop. Preserve its exact
    // observation identity for current abstention without granting any subtitle-domain
    // authority: domain_valid still gates covers, controller state and conditioning.
    if (LocatorSourceGeometryValid()) {
        [unroll]
        for (uint component = 0u; component < 2u; ++component) {
            LocatorState[V2_SUBTITLE_PLANE_FRAME_OFFSET + component] = locator_frame[component];
            LocatorState[V2_SUBTITLE_PLANE_ANALYSIS_GENERATION_OFFSET + component] = locator_frame[component + 2u];
            LocatorState[V2_SUBTITLE_PLANE_OBSERVATION_TIMESTAMP_OFFSET + component] = locator_observation[component];
            LocatorState[V2_SUBTITLE_PLANE_SOURCE_EXTENT_OFFSET + component] = locator_source[component];
            LocatorState[V2_SUBTITLE_PLANE_FIELD_EXTENT_OFFSET + component] = locator_field[component];
            LocatorState[V2_SUBTITLE_PLANE_ROI_OFFSET + component] = locator_field[component + 2u];
        }
        [unroll]
        for (uint component = 0u; component < 4u; ++component)
            LocatorState[V2_SUBTITLE_PLANE_CONTENT_OFFSET + component] = locator_content[component];
        LocatorState[V2_SUBTITLE_PLANE_CUT_EPOCH_WORD] = epoch;
    }
    [unroll]
    for (uint word = 0u; word < V2_SUBTITLE_PLANE_ADAPTIVE_WORD_COUNT; ++word)
        LocatorState[ADAPTIVE + word] = AdaptiveReset == 0u ? PreviousState[ADAPTIVE + word] : 0u;
}

bool ConditionStateValid(out uint count, out uint kinds, out float target) {
    count = 0u; kinds = 0u; target = 0.0f;
    if (!LocatorGeometryValid() || !GeometryPublicationReady() ||
        !StateHeaderValid(false, true) ||
        ConditionStateSnapshot[V2_SUBTITLE_PLANE_FLAGS_WORD] != V2_SUBTITLE_PLANE_CURRENT_VALID_FLAG ||
        ConditionStateSnapshot[V2_SUBTITLE_PLANE_CUT_EPOCH_WORD] !=
            asuint(SBS_STATE_HARD_CUT_COUNT(CutBridge[SBS_STATE_VECTOR_HARD_CUT_COUNT]))) return false;
    uint current_count;
    if (!ValidateOcrRecord(current_count) ||
        ConditionStateSnapshot[V2_SUBTITLE_PLANE_CURRENT_SLOT_MASK_WORD] != SlotMask(current_count)) return false;
    if (ConditionStateSnapshot[ADAPTIVE + 15u] != 0u ||
        !FiniteFloat(v2_max_horizontal_slope) || v2_max_horizontal_slope < 0.0f ||
        !FiniteFloat(v2_max_vertical_shear) || v2_max_vertical_shear < 0.0f) return false;
    count = current_count;
    kinds = CurrentKinds(count);
    target = count != 0u ? asfloat(ConditionStateSnapshot[ADAPTIVE + 3u]) : 0.0f;
    return SubtitleTargetIsValid(target);
}

void PublishConditionParamsFromSnapshot() {
    uint count, kinds;
    float target;
    bool valid = ConditionStateValid(count, kinds, target) && count != 0u;
    ConditionParamsOut[CONDITION_PARAM_SCHEMA_WORD] = V2_SUBTITLE_CONDITION_PARAM_SCHEMA;
    ConditionParamsOut[CONDITION_PARAM_TAG_WORD] = V2_SUBTITLE_CONDITION_PARAM_TAG;
    ConditionParamsOut[CONDITION_PARAM_CURRENT_COUNT_WORD] = valid ? count : 0u;
    ConditionParamsOut[CONDITION_PARAM_CURRENT_KINDS_WORD] = valid ? kinds : 0u;
    ConditionParamsOut[CONDITION_PARAM_ACTIVE_WORD] = valid ? V2_SUBTITLE_CONDITION_PARAM_FULLY_ACTIVE_VALUE : 0u;
    ConditionParamsOut[CONDITION_PARAM_TARGET_WORD] = valid ? asuint(target) : 0u;
}

float AdaptiveMaximumPlane() {
    // Reserve the representation range for the independently selected UI plane.
    return V2DisplayBudget();
}

float AdaptiveCandidate(uint index, float limit) {
    if (index == AdaptiveMaximumIndex()) return limit;
    return min((4.0f * (float)index) / (float)locator_source.x, limit);
}

bool AdaptiveBudgetPass(uint bad, uint covered, uint content) {
    if (covered == 0u) return true;
    // Calibrated field sizes keep these integer products below 2^32. Equality passes.
    return bad * 5u <= covered && bad * 50u <= content;
}

void AdaptiveDisarm(inout uint flags, inout uint tail[16]) {
    // Former approach/retreat flags and dwell words cannot retain authority.
    flags &= ~3u;
    tail[6u] = 0u;
    tail[7u] = 0u;
    tail[8u] = 0u;
    tail[9u] = 0u;
    tail[10u] = 0u;
    tail[11u] = 0u;
}

void AdaptivePublish(uint tail[16]) {
    [unroll]
    for (uint word = 0u; word < V2_SUBTITLE_PLANE_ADAPTIVE_WORD_COUNT; ++word)
        LocatorState[ADAPTIVE + word] = tail[word];
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
        for (uint word = 0u; word < V2_SUBTITLE_PLANE_ADAPTIVE_WORD_COUNT; ++word) {
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
        flags &= ~ADAPTIVE_CLOCK;
        tail[4u] = 0u;
        tail[5u] = 0u;
        tail[1u] = flags;
        AdaptivePublish(tail);
        return;
    }

    uint covered = AdaptiveCounts[0u];
    uint content_area = LocatorContentWidth() * (locator_content.w - locator_content.y);
    uint maximum_index = AdaptiveMaximumIndex();
    uint required = maximum_index;
    bool required_found = false;
    [loop]
    for (uint level = 0u; level <= maximum_index; ++level) {
        uint bad = AdaptiveCounts[2u + level];
        if (!required_found && AdaptiveBudgetPass(bad, covered, content_area)) {
            required = level;
            required_found = true;
        }
    }
    flags = required_found ? flags & ~ADAPTIVE_CAPPED : flags | ADAPTIVE_CAPPED;
    tail[13u] = covered;
    tail[14u] = AdaptiveCounts[2u + maximum_index];
    tail[15u] = AdaptiveCounts[1u];
    uint2 now = locator_observation.xy;
    uint2 last = uint2(tail[4u], tail[5u]);
    bool have_clock = (flags & ADAPTIVE_CLOCK) != 0u;
    if (AdaptiveCurrentCount == 0u) {
        // Authoritative EMPTY removes protection now, even without a usable
        // integration clock. Its hidden presentation cannot enter the next cue.
        AdaptiveDisarm(flags, tail);
        flags &= ~(ADAPTIVE_CAPPED | ADAPTIVE_PRESENTED | ADAPTIVE_CLOCK);
        bool accept_empty_clock = any(now != 0u) && (!have_clock || V2ClockAfter(now, last));
        tail[2u] = 0u;
        tail[3u] = 0u;
        tail[4u] = accept_empty_clock ? now.x : 0u;
        tail[5u] = accept_empty_clock ? now.y : 0u;
        tail[1u] = accept_empty_clock ? flags | ADAPTIVE_CLOCK : flags;
        tail[13u] = 0u;
        tail[14u] = 0u;
        tail[15u] = 0u;
        AdaptivePublish(tail);
        return;
    }
    if (AdaptiveCounts[1u] != 0u || all(now == 0u) ||
        (have_clock && !V2ClockAfter(now, last))) {
        // No frame count/FPS clock is substituted. Hold presentation and discard
        // timing authority, so the next valid observation only rearms the clock.
        AdaptiveDisarm(flags, tail);
        flags &= ~ADAPTIVE_CLOCK;
        tail[4u] = 0u;
        tail[5u] = 0u;
        tail[1u] = flags;
        AdaptivePublish(tail);
        return;
    }

    // Current clearance is a diagnostic goal, not a presentation lower bound.
    uint goal = required;
    AdaptiveDisarm(flags, tail);
    precise float current_goal = AdaptiveCandidate(required, limit);
    if ((flags & ADAPTIVE_PRESENTED) == 0u) {
        // No frame confirmation or temporal wait on first visible entry/reentry.
        applied = current_goal;
    } else if (have_clock) {
        uint2 elapsed = V2ClockSubtract(now, last);
        precise float cap = (float)V2_SUBTITLE_PLANE_MAX_TICK_GAP_MS * 0.001f;
        precise float seconds = elapsed.y != 0u ? cap : min((float)elapsed.x * 1.0e-6f, cap);
        precise float tau = (float)V2_SUBTITLE_PLANE_SMOOTHING_TIME_CONSTANT_MS * 0.001f;
        precise float alpha = 1.0f - exp(-seconds / tau);
        precise float delta = current_goal - applied;
        precise float maximum_step = (float)V2_SUBTITLE_PLANE_SLEW_SOURCE_PIXELS_PER_SECOND *
            seconds / (float)locator_source.x;
        precise float update = applied + clamp(delta * alpha, -maximum_step, maximum_step);
        applied = clamp(update, min(applied, current_goal), max(applied, current_goal));
    }
    tail[2u] = goal;
    tail[4u] = now.x;
    tail[5u] = now.y;
    tail[1u] = flags | ADAPTIVE_CLOCK | ADAPTIVE_PRESENTED;
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

bool ProbabilityUnitValid(float value) {
    return FiniteFloat(value) && value >= 0.0f && value <= 1.0f;
}

bool ProbabilityPairAtSourceCenter(uint2 position, out float current, out float previous) {
    current = 0.0f;
    previous = 0.0f;
    float2 detector = (float2(position) + 0.5f) *
        float2((float)V2_OCR_OUTPUT_WIDTH / (float)ProbabilitySource.x,
               (float)V2_OCR_OUTPUT_HEIGHT / (float)ProbabilitySource.w) - 0.5f;
    int2 lo = int2(floor(detector));
    float2 blend = frac(detector);
    int2 a = clamp(lo, int2(0, 0), int2(V2_OCR_OUTPUT_WIDTH - 1u, V2_OCR_OUTPUT_HEIGHT - 1u));
    int2 b = clamp(lo + int2(1, 1), int2(0, 0),
                  int2(V2_OCR_OUTPUT_WIDTH - 1u, V2_OCR_OUTPUT_HEIGHT - 1u));
    uint4 indices = uint4((uint)a.y * V2_OCR_OUTPUT_WIDTH + (uint)a.x,
                         (uint)a.y * V2_OCR_OUTPUT_WIDTH + (uint)b.x,
                         (uint)b.y * V2_OCR_OUTPUT_WIDTH + (uint)a.x,
                         (uint)b.y * V2_OCR_OUTPUT_WIDTH + (uint)b.x);
    float4 now = float4(CurrentProbability[indices.x], CurrentProbability[indices.y],
                       CurrentProbability[indices.z], CurrentProbability[indices.w]);
    float4 old = float4(PreviousProbability[indices.x], PreviousProbability[indices.y],
                       PreviousProbability[indices.z], PreviousProbability[indices.w]);
    if (!ProbabilityUnitValid(now.x) || !ProbabilityUnitValid(now.y) ||
        !ProbabilityUnitValid(now.z) || !ProbabilityUnitValid(now.w) ||
        !ProbabilityUnitValid(old.x) || !ProbabilityUnitValid(old.y) ||
        !ProbabilityUnitValid(old.z) || !ProbabilityUnitValid(old.w)) return false;
    current = lerp(lerp(now.x, now.y, blend.x), lerp(now.z, now.w, blend.x), blend.y);
    previous = lerp(lerp(old.x, old.y, blend.x), lerp(old.z, old.w, blend.x), blend.y);
    return true;
}

uint ProbabilitySourceEdge(uint edge, uint content_begin, uint content_extent, uint source_extent) {
    // First source center at or beyond this field edge, with exact half-open membership.
    // Source/field extents are bounded before this integer arithmetic is reached.
    uint scaled_edge = 2u * (edge - content_begin) * source_extent;
    uint denominator = 2u * content_extent;
    if (scaled_edge <= content_extent) return 0u;
    return min((scaled_edge - content_extent + denominator - 1u) / denominator, source_extent);
}

uint4 ProbabilityCoreSourceBounds(uint slot) {
    uint offset = V2_OCR_RAW_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
    uint4 core = uint4(OcrRecord[offset], OcrRecord[offset + 1u],
                      OcrRecord[offset + 2u], OcrRecord[offset + 3u]);
    uint content_width = locator_content.z - locator_content.x;
    uint content_height = locator_content.w - locator_content.y;
    uint left = ProbabilitySourceEdge(core.x, locator_content.x, content_width, ProbabilitySource.x);
    uint right = ProbabilitySourceEdge(core.z, locator_content.x, content_width, ProbabilitySource.x);
    uint top = ProbabilitySourceEdge(core.y, locator_content.y, content_height, ProbabilitySource.y);
    uint bottom = ProbabilitySourceEdge(core.w, locator_content.y, content_height, ProbabilitySource.y);
    top = clamp(top, ProbabilitySource.z, ProbabilitySource.z + ProbabilitySource.w);
    bottom = clamp(bottom, ProbabilitySource.z, ProbabilitySource.z + ProbabilitySource.w);
    return uint4(left, top - ProbabilitySource.z, right, bottom - ProbabilitySource.z);
}

void PrepareProbabilityReset() {
    ProbabilityResetGate = 0u;
    ProbabilityResetMask = 0u;
    ProbabilitySource = uint4(0u, 0u, 0u, 0u);
    uint count, kinds, crop_top, crop_height;
    float target;
    if (AdaptiveReset != 0u || !ConditionStateValid(count, kinds, target) || count == 0u ||
        any(locator_source.xy > V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT) ||
        !V2SubtitleOcrComputeCrop(locator_source.x, locator_source.y, crop_top, crop_height)) return;
    uint now_words, now_stride, old_words, old_stride;
    CurrentProbability.GetDimensions(now_words, now_stride);
    PreviousProbability.GetDimensions(old_words, old_stride);
    if (now_words != V2_OCR_OUTPUT_WIDTH * V2_OCR_OUTPUT_HEIGHT || now_stride != 4u ||
        old_words != now_words || old_stride != 4u) return;
    uint4 source = uint4(locator_source.xy, crop_top, crop_height);
    bool duplicate;
    if (!V2SubtitleHistoryCompatible(ContourIdentity, source, locator_field, locator_content,
            locator_frame, locator_observation.xy, locator_source.w,
            ConditionStateSnapshot[V2_SUBTITLE_PLANE_CUT_EPOCH_WORD], duplicate) || duplicate ||
        ContourIdentity[2u] != 1u) return;
    ProbabilitySource = source;
    ProbabilityResetGate = 1u;
}

void ResolveProbabilityReset(uint lane) {
    if (lane == 0u) PrepareProbabilityReset();
    GroupMemoryBarrierWithGroupSync();
    // The gate and count are uniform for this single 256-lane observation group.
    if (ProbabilityResetGate == 0u) return;
    [loop]
    for (uint slot = 0u; slot < AdaptiveCurrentCount; ++slot) {
        if (lane == 0u) ProbabilityCoreBounds = ProbabilityCoreSourceBounds(slot);
        GroupMemoryBarrierWithGroupSync();
        uint width = ProbabilityCoreBounds.z - ProbabilityCoreBounds.x;
        uint area = width * (ProbabilityCoreBounds.w - ProbabilityCoreBounds.y);
        precise float current_mass = 0.0f;
        precise float previous_mass = 0.0f;
        uint invalid = 0u;
        if (width != 0u) {
            uint position_y = lane / width;
            uint position_x = lane - position_y * width;
            uint stride_y = 256u / width;
            uint stride_x = 256u - stride_y * width;
            [loop]
            for (uint cell = lane; cell < area; cell += 256u) {
                uint2 position = ProbabilityCoreBounds.xy + uint2(position_x, position_y);
                position_x += stride_x;
                position_y += stride_y;
                if (position_x >= width) {
                    position_x -= width;
                    ++position_y;
                }
                float current, previous;
                if (!ProbabilityPairAtSourceCenter(position, current, previous)) {
                    invalid = 1u;
                } else {
                    current_mass += current;
                    previous_mass += previous;
                }
            }
        }
        ProbabilitySums[lane] = float2(current_mass, previous_mass);
        ProbabilityInvalid[lane] = invalid;
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint step = 128u; step != 0u; step >>= 1u) {
            if (lane < step) {
                ProbabilitySums[lane] += ProbabilitySums[lane + step];
                ProbabilityInvalid[lane] |= ProbabilityInvalid[lane + step];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0u) ProbabilityMass = ProbabilitySums[0u];
        GroupMemoryBarrierWithGroupSync();
        // Every lane takes the same mass-validity branch. Normalize only positive complete
        // masses, then revisit exactly the same native centers and bilinear contributors.
        bool mass_valid = ProbabilityInvalid[0u] == 0u &&
            FiniteFloat(ProbabilityMass.x) && FiniteFloat(ProbabilityMass.y) &&
            ProbabilityMass.x >= 0.0f && ProbabilityMass.y >= 0.0f;
        // Capture the shared validity in every lane before the second pass reuses its scratch.
        GroupMemoryBarrierWithGroupSync();
        if (mass_valid && ProbabilityMass.x > 0.0f && ProbabilityMass.y > 0.0f) {
            precise float difference = 0.0f;
            invalid = 0u;
            uint position_y = lane / width;
            uint position_x = lane - position_y * width;
            uint stride_y = 256u / width;
            uint stride_x = 256u - stride_y * width;
            [loop]
            for (uint cell = lane; cell < area; cell += 256u) {
                uint2 position = ProbabilityCoreBounds.xy + uint2(position_x, position_y);
                position_x += stride_x;
                position_y += stride_y;
                if (position_x >= width) {
                    position_x -= width;
                    ++position_y;
                }
                float current, previous;
                if (!ProbabilityPairAtSourceCenter(position, current, previous)) {
                    invalid = 1u;
                } else {
                    precise float current_unit = current / ProbabilityMass.x;
                    precise float previous_unit = previous / ProbabilityMass.y;
                    difference += abs(current_unit - previous_unit);
                }
            }
            ProbabilitySums[lane] = float2(difference, 0.0f);
            ProbabilityInvalid[lane] = invalid;
            GroupMemoryBarrierWithGroupSync();
            [unroll]
            for (uint step = 128u; step != 0u; step >>= 1u) {
                if (lane < step) {
                    ProbabilitySums[lane].x += ProbabilitySums[lane + step].x;
                    ProbabilityInvalid[lane] |= ProbabilityInvalid[lane + step];
                }
                GroupMemoryBarrierWithGroupSync();
            }
            if (lane == 0u && ProbabilityInvalid[0u] == 0u &&
                FiniteFloat(ProbabilitySums[0u].x)) {
                precise float score = 0.5f * ProbabilitySums[0u].x;
                if (score > V2_SUBTITLE_CONTOUR_PROBABILITY_CHANGE_RATIO)
                    ProbabilityResetMask |= 1u << slot;
            }
        } else if (lane == 0u && mass_valid &&
                   (ProbabilityMass.x > 0.0f || ProbabilityMass.y > 0.0f)) {
            // Exactly one zero mass has total variation one; both zero masses set no bit.
            if (1.0f > V2_SUBTITLE_CONTOUR_PROBABILITY_CHANGE_RATIO)
                ProbabilityResetMask |= 1u << slot;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) {
        LocatorState[V2_SUBTITLE_PLANE_HISTORY_RESET_MASK_WORD] = ProbabilityResetMask;
        ConditionStateSnapshot[V2_SUBTITLE_PLANE_HISTORY_RESET_MASK_WORD] = ProbabilityResetMask;
    }
}

void ResolveWarmup() {
    // Current native text evidence owns protection on this observation. A cue or a
    // retained rectangle cannot extend protection after that evidence disappears.
    LocatorState[V2_SUBTITLE_PLANE_WARMUP_WORD] = 0u;
    ConditionStateSnapshot[V2_SUBTITLE_PLANE_WARMUP_WORD] = 0u;
}

[numthreads(256, 1, 1)]
void resolve_main(uint3 dispatch_id : SV_DispatchThreadID, uint lane : SV_GroupIndex) {
    if (lane == 0u) ResolveObservation();
    AllMemoryBarrierWithGroupSync();
    if (lane < ADAPTIVE_COUNT_WORDS) AdaptiveCounts[lane] = 0u;
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
    if (lane < V2_SUBTITLE_PLANE_STATE_WORD_COUNT)
        ConditionStateSnapshot[lane] = LocatorState[lane];
    GroupMemoryBarrierWithGroupSync();
    ResolveProbabilityReset(lane);
    AllMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        ResolveWarmup();
        PublishConditionParamsFromSnapshot();
    }
}

bool ConditionParamsValid(out uint count, out uint kinds, out uint active, out float target) {
    uint words, stride;
    ConditionParams.GetDimensions(words, stride);
    count = 0u; kinds = 0u; active = 0u; target = 0.0f;
    if (words != V2_SUBTITLE_CONDITION_PARAM_WORD_COUNT || stride != 4u ||
        !ConditionStateValid(count, kinds, target) || count == 0u) return false;
    active = ConditionParams[CONDITION_PARAM_ACTIVE_WORD];
    return ConditionParams[0u] == V2_SUBTITLE_CONDITION_PARAM_SCHEMA &&
        ConditionParams[1u] == V2_SUBTITLE_CONDITION_PARAM_TAG &&
        ConditionParams[CONDITION_PARAM_CURRENT_COUNT_WORD] == count &&
        ConditionParams[CONDITION_PARAM_CURRENT_KINDS_WORD] == kinds &&
        active == V2_SUBTITLE_CONDITION_PARAM_FULLY_ACTIVE_VALUE &&
        ConditionParams[CONDITION_PARAM_TARGET_WORD] == asuint(target);
}

bool ContourIdentityValid(uint current_count, uint current_kinds) {
    uint distance_width, distance_height, words, stride;
    ContourDistance.GetDimensions(distance_width, distance_height);
    ContourIdentity.GetDimensions(words, stride);
    if (words != V2_SUBTITLE_CONTOUR_HISTORY_WORD_COUNT || stride != 4u ||
        distance_width != locator_field.x || distance_height != locator_field.y ||
        ContourIdentity[0u] != V2_SUBTITLE_CONTOUR_SCHEMA ||
        ContourIdentity[1u] != V2_SUBTITLE_CONTOUR_HISTORY_TAG || ContourIdentity[2u] != 1u ||
        ContourIdentity[22u] != V2_SUBTITLE_CONTOUR_RECIPE ||
        ContourIdentity[V2_SUBTITLE_CONTOUR_HISTORY_SNAPSHOT_VALID_WORD] != 1u ||
        ContourIdentity[3u] != ConditionStateSnapshot[V2_SUBTITLE_PLANE_CUT_EPOCH_WORD] ||
        ContourIdentity[8u] != locator_source.x || ContourIdentity[9u] != locator_source.y ||
        ContourIdentity[20u] != locator_observation.x || ContourIdentity[21u] != locator_observation.y)
        return false;
    uint crop_top, crop_height;
    if (!V2SubtitleOcrComputeCrop(locator_source.x, locator_source.y, crop_top, crop_height) ||
        ContourIdentity[10u] != crop_top || ContourIdentity[11u] != crop_height) return false;
    [unroll]
    for (uint component = 0u; component < 4u; ++component) {
        if (ContourIdentity[4u + component] != locator_frame[component] ||
            ContourIdentity[12u + component] != locator_field[component] ||
            ContourIdentity[16u + component] != locator_content[component]) return false;
    }
    [unroll]
    for (uint reserved = 24u; reserved < V2_SUBTITLE_CONTOUR_HISTORY_WORD_COUNT; ++reserved)
        if (ContourIdentity[reserved] != 0u) return false;
    return true;
}

bool ConditionContentValid() {
    return target_w != 0u && target_h != 0u &&
        target_w == locator_field.x && target_h == locator_field.y &&
        locator_content.x < locator_content.z && locator_content.y < locator_content.w &&
        locator_content.z <= target_w && locator_content.w <= target_h &&
        all(locator_content == DepthAnalysisContentCells());
}

bool EffectiveContourDistance(uint2 position, float raw, uint count, out float distance) {
    distance = raw;
    return FiniteFloat(raw) && raw >= 0.0f && asuint(raw) != 0x80000000u;
}

float EvaluateConditionedBase(
    uint2 condition_position,
    float base,
    bool state_valid,
    uint current_count,
    uint current_kinds,
    float condition_target,
    out float weight) {
    weight = 0.0f;
    if (!state_valid || !FiniteFloat(base) || abs(base) > v2_direct_container_limit) {
        return base;
    }
    float budget;
    if (!EffectiveContourDistance(condition_position,
        ContourDistance.Load(int3(condition_position, 0)), current_count, budget)) return base;
    // Each supported footprint is exactly on the UI plane. Elsewhere retain immutable Base
    // whenever it is inside the nearest pin's slope-feasible interval; no soft plateau or
    // Gaussian tail can unnecessarily pull additional scenery toward the caption.
    if (budget == 0.0f) {
        weight = 1.0f;
        return condition_target;
    }
    precise float difference = condition_target - base;
    precise float gap_pixels = abs(difference) * (float)locator_source.x;
    if (gap_pixels == 0.0f || budget >= gap_pixels) return base;
    precise float remaining = budget / gap_pixels;
    weight = 1.0f - remaining;
    if (weight <= 0.0f) {
        weight = 0.0f;
        return base;
    }
    if (weight >= 1.0f) {
        weight = 1.0f;
        return condition_target;
    }
    precise float normalized_budget = budget / (float)locator_source.x;
    precise float proposal = difference > 0.0f ?
        condition_target - normalized_budget : condition_target + normalized_budget;
    // Independent FP32 division/subtraction can round a near-gap proposal beyond Base.
    // Preserve the closed Base/UI interval and copy endpoint bits, including signed zero.
    if (difference > 0.0f) {
        if (proposal <= base) return base;
        if (proposal >= condition_target) return condition_target;
    } else {
        if (proposal >= base) return base;
        if (proposal <= condition_target) return condition_target;
    }
    return proposal;
}

groupshared uint ConditionGroupValid;
groupshared uint ConditionGroupContentValid;
groupshared uint ConditionGroupCount;
groupshared uint ConditionGroupKinds;
groupshared float ConditionGroupTarget;

[numthreads(16, 16, 1)]
void condition_main(uint3 dispatch_id : SV_DispatchThreadID, uint group_index : SV_GroupIndex) {
    // Identity/state are immutable for the entire dispatch. Authenticate once per group while
    // every thread still participates in the barrier, including partial edge-group threads.
    if (group_index == 0u) {
        uint words, stride;
        LocatorStateRead.GetDimensions(words, stride);
        [unroll]
        for (uint word = 0u; word < V2_SUBTITLE_PLANE_STATE_WORD_COUNT; ++word)
            ConditionStateSnapshot[word] = words == V2_SUBTITLE_PLANE_STATE_WORD_COUNT && stride == 4u ?
                LocatorStateRead[word] : 0u;
        uint count, kinds, active;
        float target;
        ConditionGroupContentValid = ConditionContentValid() ? 1u : 0u;
        ConditionGroupValid = ConditionParamsValid(count, kinds, active, target) &&
            ContourIdentityValid(count, kinds) ? 1u : 0u;
        ConditionGroupCount = count;
        ConditionGroupKinds = kinds;
        ConditionGroupTarget = target;
    }
    GroupMemoryBarrierWithGroupSync();
    if (all(dispatch_id.xy == 0u)) {
        // A current support slab has finite horizontal budget across that content row.
        // Scan the first content column only here; no individual group may mistake an
        // unrelated empty top row for absence of every current glyph footprint.
        uint enabled = 0u;
        if (ConditionGroupValid != 0u && ConditionGroupContentValid != 0u) {
            [loop]
            for (uint row = locator_content.y; row < locator_content.w; ++row) {
                uint2 probe = uint2(locator_content.x, row);
                float row_budget;
                if (EffectiveContourDistance(probe, ContourDistance.Load(int3(probe, 0)),
                    ConditionGroupCount, row_budget)) {
                    enabled = 1u;
                    break;
                }
            }
        }
        ConditionParamsOut[0u] = enabled;
    }
    if (dispatch_id.x >= target_w || dispatch_id.y >= target_h) return;
    // Clamp only an actual finite-budget proposal. Empty rows retain original Base bits,
    // including padding; they must not inherit a neighboring content row or subtitle ramp.
    uint2 condition_position = ConditionGroupContentValid != 0u && ConditionGroupValid != 0u ?
        DepthAnalysisClampCell(dispatch_id.xy) : dispatch_id.xy;
    bool local_valid = false;
    if (ConditionGroupContentValid != 0u && ConditionGroupValid != 0u) {
        float local_budget;
        local_valid = EffectiveContourDistance(condition_position,
            ContourDistance.Load(int3(condition_position, 0)), ConditionGroupCount, local_budget);
    }
    float base = BaseField.Load(int3(local_valid ? condition_position : dispatch_id.xy, 0));
    float weight;
    float conditioned = EvaluateConditionedBase(
        condition_position, base, local_valid, ConditionGroupCount,
        ConditionGroupKinds, ConditionGroupTarget, weight);
    ConditionedField[dispatch_id.xy] = conditioned;
    GaussianWeightOut[dispatch_id.xy] = weight;
}
