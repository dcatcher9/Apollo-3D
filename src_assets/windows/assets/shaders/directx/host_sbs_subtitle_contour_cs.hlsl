// Native subtitle contours: four genuine fixed-anchor RGB confirmations inside current OCR covers.
// OCR selects the current search region; temporal stability is not semantic text truth.
// Support is binary0/1. Arbitrary colors use the same fixed-anchor RGB stability predicate.
// Shared exact UI pins, margin and slope collar remain the downstream depth policy.

#include "include/depth_coordinate_v2_ocr_assert.generated.hlsl"
#include "include/depth_coordinate_v2.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"
#include "include/host_sbs_subtitle_history_compatibility.hlsl"

cbuffer SubtitleContourConstants : register(b0) {
    uint4 contour_source;      // analysis source width/height, bottom crop top/height
    uint4 contour_field;       // field width/height, authenticated OCR ROI top/bottom
    uint4 contour_content;     // integer half-open real-content rectangle in the field
    uint4 contour_frame;       // exact current frame lo/hi, analysis generation lo/hi
    uint4 contour_observation; // exact source timestamp us lo/hi, input-domain reset, reserved zero
    uint4 contour_options;     // enabled 1, generated recipe schema, two canonical zero reserved words
};

Texture2D<float4> CurrentDisplayBand : register(t0); // R16G16B16A16_FLOAT; alpha is finite-valid 0/1
StructuredBuffer<float> ProbabilityMap : register(t1);
StructuredBuffer<uint> CurrentPlane : register(t2);
StructuredBuffer<uint> CurrentConditionParams : register(t3);
Texture2D<uint> CurrentSupportRead : register(t5);
Texture2D<float4> PreviousFeatures : register(t6); // fixed RGB anchor and exact valid/count/mask code
StructuredBuffer<uint> PreviousIdentity : register(t7);
StructuredBuffer<float4> CurrentCutBridge : register(t8);
StructuredBuffer<uint> CurrentOcrRecord : register(t9);
StructuredBuffer<uint4> CurrentGeometryState : register(t10);
Texture2D<float4> NextFeaturesRead : register(t11); // immutable staged result before old history overwrite

RWTexture2D<uint> NativeSupportWrite : register(u0);
RWTexture2D<uint> FieldPinWrite : register(u1);
RWTexture2D<float4> HistoryFeaturesWrite : register(u2);
RWStructuredBuffer<uint> HistoryIdentityWrite : register(u3);
RWTexture2D<float4> CompletedDisplayBandWrite : register(u4); // same RGBA16F as input
RWStructuredBuffer<float> CompletedProbabilityWrite : register(u5);

static const uint CONTOUR_SCHEMA = V2_SUBTITLE_CONTOUR_SCHEMA;
static const uint RECIPE_SCHEMA = V2_SUBTITLE_CONTOUR_RECIPE;
static const uint HISTORY_TAG = V2_SUBTITLE_CONTOUR_HISTORY_TAG;
static const uint HISTORY_WORDS = V2_SUBTITLE_CONTOUR_HISTORY_WORD_COUNT;
static const uint MAX_SOURCE_EXTENT = V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT;
static const int HISTORY_JITTER_RADIUS = V2_SUBTITLE_CONTOUR_HISTORY_JITTER_RADIUS_SOURCE_PIXELS;

groupshared uint CurrentGate;
groupshared uint CurrentCount;
groupshared uint CurrentEpoch;
groupshared uint CurrentResetMask;
groupshared uint4 CurrentCovers[V2_SUBTITLE_PLANE_COVER_CAPACITY];
groupshared uint4 CurrentCores[V2_SUBTITLE_PLANE_COVER_CAPACITY];
groupshared uint HistoryGate;
groupshared uint HistoryDuplicate;
groupshared uint SnapshotGate;

bool ContourFinite(float value) {
    return (asuint(value) & 0x7f800000u) != 0x7f800000u;
}

bool ContourFinite3(float3 value) {
    return ContourFinite(value.x) && ContourFinite(value.y) && ContourFinite(value.z);
}

bool ClockLess(uint2 a, uint2 b) {
    return V2SubtitleHistoryClockLess(a, b);
}

uint2 SourceDispatchStride() {
    // Every entry is dispatched on the existing 16x16 field indirect grid. Source passes
    // grid-stride over the larger native band; no CPU flip or new indirect argument ABI.
    return max(((contour_field.xy + 15u) / 16u) * 16u, uint2(16u, 16u));
}

bool ValidCurrentRect(uint4 r) {
    return r.x < r.z && r.y < r.w && r.x >= contour_content.x &&
        r.y >= contour_content.y && r.z <= contour_content.z && r.w <= contour_content.w;
}

uint4 ReadCurrentRect(uint slot, uint box_offset) {
    uint offset = box_offset + slot * V2_OCR_BOX_WORD_COUNT;
    return uint4(CurrentOcrRecord[offset], CurrentOcrRecord[offset + 1u],
                 CurrentOcrRecord[offset + 2u], CurrentOcrRecord[offset + 3u]);
}

bool OcrPairPayloadValid(uint slot) {
    uint core_offset = V2_OCR_RAW_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
    uint cover_offset = V2_OCR_FINAL_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
    uint4 core = uint4(CurrentOcrRecord[core_offset], CurrentOcrRecord[core_offset + 1u],
                       CurrentOcrRecord[core_offset + 2u], CurrentOcrRecord[core_offset + 3u]);
    uint4 cover = uint4(CurrentOcrRecord[cover_offset], CurrentOcrRecord[cover_offset + 1u],
                        CurrentOcrRecord[cover_offset + 2u], CurrentOcrRecord[cover_offset + 3u]);
    float score = asfloat(CurrentOcrRecord[core_offset + 4u]);
    if (!ValidCurrentRect(core) || !ValidCurrentRect(cover) ||
        core.y < contour_field.z || core.w > contour_field.w ||
        cover.x > core.x || cover.y > core.y || cover.z < core.z || cover.w < core.w ||
        !ContourFinite(score) || score < V2_OCR_MIN_MEAN_SCORE || score > 1.0f ||
        (CurrentOcrRecord[core_offset + 5u] & ~V2_OCR_BOX_KNOWN_FLAGS) != 0u ||
        CurrentOcrRecord[core_offset + 6u] == 0u ||
        CurrentOcrRecord[core_offset + 6u] > V2_OCR_OUTPUT_WIDTH ||
        CurrentOcrRecord[core_offset + 7u] >= CurrentOcrRecord[core_offset + 6u]) return false;
    [unroll]
    for (uint word = 4u; word < V2_OCR_BOX_WORD_COUNT; ++word) {
        if (CurrentOcrRecord[core_offset + word] != CurrentOcrRecord[cover_offset + word]) return false;
    }
    bool ribbon = (CurrentOcrRecord[core_offset + 5u] & V2_OCR_BOX_FLAG_RIBBON) != 0u;
    return ribbon ? cover.x == contour_content.x && cover.z == contour_content.z &&
                        cover.w == contour_content.w && cover.y >= contour_field.z :
                    cover.y >= contour_field.z && cover.w <= contour_field.w;
}

bool CurrentGeometryReady() {
    uint vector_count, stride;
    CurrentGeometryState.GetDimensions(vector_count, stride);
    if (vector_count * 4u != V2_SHADOW_STATE_WORD_COUNT || stride != 16u ||
        !V2JointPlaneConstantsValid() || !ContourFinite(v2_requested_gain) || v2_requested_gain <= 0.0f ||
        any(uint2(v2_joint_observation_timestamp_low, v2_joint_observation_timestamp_high) !=
            contour_observation.xy)) return false;
    float4 active = asfloat(CurrentGeometryState[0u]);
    float4 control = asfloat(CurrentGeometryState[1u]);
    uint4 mapping = CurrentGeometryState[2u];
    V2AdaptiveCameraState camera;
    camera.clock = CurrentGeometryState[3u];
    camera.target = CurrentGeometryState[4u];
    camera.seed_times = CurrentGeometryState[5u];
    camera.seed_values = CurrentGeometryState[6u];
    uint2 now = uint2(v2_joint_observation_timestamp_low, v2_joint_observation_timestamp_high);
    // A compact UI-plane timestamp is not an integrity seal for the current raw/Base publication.
    // Use the same canonical state/clock/cut admission as the UI resolver and conditioner.
    return V2CameraStateWithAdaptiveStateValid(active, control, mapping, camera) &&
        V2_STATE_FRAME_VALID(control) == 1.0f &&
        mapping.y == V2_CONTRACT_TAG && mapping.z == V2_ADAPTIVE_POLICY_ID &&
        any(now != 0u) && all(camera.clock.xy == now) &&
        asuint(SBS_STATE_CUT_CONTRACT_TAG_BITS(
            CurrentCutBridge[SBS_STATE_VECTOR_CUT_CONTRACT_TAG_BITS])) == SBS_CUT_CONTRACT_TAG &&
        asuint(V2_STATE_CONFIRMED_CUT_COUNT(control)) ==
            asuint(SBS_STATE_HARD_CUT_COUNT(CurrentCutBridge[SBS_STATE_VECTOR_HARD_CUT_COUNT]));
}

bool CurrentSnapshotEvidenceValid() {
    // Current detector evidence includes a legal empty record. Plane/condition state is not
    // read here: this independent source/probability snapshot cannot authorize pins.
    uint band_w, band_h, probability_count, probability_stride;
    uint ocr_words, ocr_stride, cut_vectors, cut_stride;
    CurrentDisplayBand.GetDimensions(band_w, band_h);
    ProbabilityMap.GetDimensions(probability_count, probability_stride);
    CurrentOcrRecord.GetDimensions(ocr_words, ocr_stride);
    CurrentCutBridge.GetDimensions(cut_vectors, cut_stride);
    if (contour_options.x != 1u || contour_options.y != RECIPE_SCHEMA ||
        any(contour_options.zw != 0u) || contour_observation.z > 1u ||
        contour_observation.w != 0u || !any(contour_frame.xy != 0u) ||
        !any(contour_observation.xy != 0u) ||
        any(contour_source.xy == 0u) || any(contour_source.xy > MAX_SOURCE_EXTENT) ||
        any(contour_field.xy == 0u) || any(contour_field.xy > MAX_SOURCE_EXTENT) ||
        contour_source.w == 0u || contour_source.z > contour_source.y ||
        contour_source.w > contour_source.y - contour_source.z ||
        band_w != contour_source.x || band_h != contour_source.w ||
        contour_content.x >= contour_content.z || contour_content.y >= contour_content.w ||
        contour_content.z > contour_field.x || contour_content.w > contour_field.y ||
        contour_field.z >= contour_field.w || contour_field.z < contour_content.y ||
        contour_field.w > contour_content.w ||
        probability_count != V2_OCR_OUTPUT_WIDTH * V2_OCR_OUTPUT_HEIGHT || probability_stride != 4u ||
        ocr_words != V2_OCR_RECORD_WORD_COUNT || ocr_stride != 4u ||
        cut_vectors != SBS_ADAPTIVE_STATE_VECTOR_COUNT || cut_stride != 16u) return false;
    uint crop_top, crop_height, roi_top, roi_bottom;
    if (!V2SubtitleOcrComputeCrop(contour_source.x, contour_source.y, crop_top, crop_height) ||
        crop_top != contour_source.z || crop_height != contour_source.w ||
        !V2SubtitleOcrProjectContentRowCeil(contour_source.x, contour_source.y,
            contour_field.x, contour_field.y, contour_content, V2_OCR_SAFE_ROW_TOP, roi_top) ||
        !V2SubtitleOcrProjectContentRowCeil(contour_source.x, contour_source.y,
            contour_field.x, contour_field.y, contour_content, V2_OCR_SAFE_ROW_BOTTOM, roi_bottom) ||
        roi_top != contour_field.z || roi_bottom != contour_field.w ||
        !CurrentGeometryReady()) return false;
    uint count = CurrentOcrRecord[4u];
    if (CurrentOcrRecord[0u] != V2_OCR_RECORD_SCHEMA || CurrentOcrRecord[1u] != V2_OCR_RECORD_TAG ||
        CurrentOcrRecord[2u] != 1u || count > V2_OCR_FINAL_BOX_CAPACITY ||
        CurrentOcrRecord[3u] != count ||
        any(uint4(CurrentOcrRecord[5u], CurrentOcrRecord[6u],
                  CurrentOcrRecord[7u], CurrentOcrRecord[8u]) != contour_frame) ||
        CurrentOcrRecord[9u] != contour_source.x || CurrentOcrRecord[10u] != contour_source.y ||
        CurrentOcrRecord[11u] != contour_field.x || CurrentOcrRecord[12u] != contour_field.y ||
        CurrentOcrRecord[13u] != contour_field.z || CurrentOcrRecord[14u] != contour_field.w ||
        CurrentOcrRecord[15u] != 0u) return false;
    [loop]
    for (uint slot = 0u; slot < V2_OCR_RAW_BOX_CAPACITY; ++slot) {
        if (slot < count) {
            if (!OcrPairPayloadValid(slot)) return false;
            uint core_offset = V2_OCR_RAW_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
            if ((CurrentOcrRecord[core_offset + 5u] & V2_OCR_BOX_FLAG_RIBBON) != 0u) {
                uint4 core = uint4(CurrentOcrRecord[core_offset], CurrentOcrRecord[core_offset + 1u],
                    CurrentOcrRecord[core_offset + 2u], CurrentOcrRecord[core_offset + 3u]);
                uint minimum_bottom;
                if ((core.z - core.x) * V2_OCR_RIBBON_MIN_WIDTH_DENOMINATOR <
                        (contour_content.z - contour_content.x) * V2_OCR_RIBBON_MIN_WIDTH_NUMERATOR ||
                    !V2SubtitleOcrProjectContentRowCeil(contour_source.x, contour_source.y,
                        contour_field.x, contour_field.y, contour_content,
                        V2_OCR_SAFE_ROW_BOTTOM - V2_OCR_RIBBON_BOTTOM_TOLERANCE_PIXELS, minimum_bottom) ||
                    core.w < minimum_bottom ||
                    CurrentOcrRecord[core_offset + 7u] < V2_OCR_RIBBON_MIN_STRUCTURAL_GAPS) return false;
            }
        } else {
            uint offset = V2_OCR_RAW_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
            [unroll]
            for (uint word = 0u; word < V2_OCR_BOX_WORD_COUNT; ++word) {
                if (CurrentOcrRecord[offset + word] != 0u) return false;
            }
        }
    }
    [loop]
    for (uint slot = count; slot < V2_OCR_FINAL_BOX_CAPACITY; ++slot) {
        uint offset = V2_OCR_FINAL_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
        [unroll]
        for (uint word = 0u; word < V2_OCR_BOX_WORD_COUNT; ++word) {
            if (CurrentOcrRecord[offset + word] != 0u) return false;
        }
    }
    return true;
}

[numthreads(16, 16, 1)]
void snapshot_main(uint3 dispatch_id : SV_DispatchThreadID, uint group_index : SV_GroupIndex) {
    if (group_index == 0u) SnapshotGate = CurrentSnapshotEvidenceValid() ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    uint width, height, words, word_stride;
    CompletedDisplayBandWrite.GetDimensions(width, height);
    CompletedProbabilityWrite.GetDimensions(words, word_stride);
    bool valid = SnapshotGate != 0u && width == contour_source.x && height == contour_source.w &&
        words == V2_OCR_OUTPUT_WIDTH * V2_OCR_OUTPUT_HEIGHT && word_stride == 4u;
    uint2 stride = SourceDispatchStride();
    [loop]
    for (uint y = dispatch_id.y; y < height; y += stride.y) {
        [loop]
        for (uint x = dispatch_id.x; x < width; x += stride.x) {
            float4 value = 0.0f;
            if (valid) {
                value = CurrentDisplayBand.Load(int3(x, y, 0));
            }
            // Preprocess already owns alpha0/zero for invalid source pixels. Preserve that
            // RGBA16F policy verbatim instead of silently replacing diagnostic input evidence.
            CompletedDisplayBandWrite[uint2(x, y)] = value;
        }
    }
    uint flat_thread = dispatch_id.y * stride.x + dispatch_id.x;
    uint flat_stride = stride.x * stride.y;
    [loop]
    for (uint word = flat_thread; word < words; word += flat_stride) {
        float value = 0.0f;
        if (valid) {
            // No float arithmetic: NaN/Inf are model-fault evidence, not probability zero.
            value = asfloat(asuint(ProbabilityMap[word]));
        }
        CompletedProbabilityWrite[word] = value;
    }
}

bool CurrentAdaptiveTailValid() {
    uint base = V2_SUBTITLE_PLANE_ADAPTIVE_OFFSET;
    uint flags = CurrentPlane[base + 1u];
    uint2 clock = uint2(CurrentPlane[base + 4u], CurrentPlane[base + 5u]);
    uint2 approach = uint2(CurrentPlane[base + 6u], CurrentPlane[base + 7u]);
    uint2 retreat = uint2(CurrentPlane[base + 8u], CurrentPlane[base + 9u]);
    float applied = asfloat(CurrentPlane[base + 3u]);
    uint maximum_level = (contour_source.x + 99u) / 100u;
    uint area = (contour_content.z - contour_content.x) *
        (contour_content.w - contour_content.y);
    uint covered = CurrentPlane[base + 13u];
    uint conflicts = CurrentPlane[base + 14u];
    // Accepted controller time may be held or disarmed independently of this source tuple.
    return CurrentPlane[base] == 1u && (flags & ~15u) == 0u && (flags & 3u) != 3u &&
        CurrentPlane[base + 2u] <= maximum_level &&
        CurrentPlane[base + 10u] <= maximum_level && CurrentPlane[base + 11u] <= maximum_level &&
        ContourFinite(applied) && applied >= 0.0f && applied <= V2DisplayBudget() &&
        CurrentPlane[base + 12u] == asuint(V2DisplayBudget()) &&
        (((flags & 8u) != 0u) == any(clock != 0u)) &&
        (((flags & 1u) != 0u) == any(approach != 0u)) &&
        (((flags & 2u) != 0u) == any(retreat != 0u)) &&
        !ClockLess(clock, approach) && !ClockLess(clock, retreat) &&
        covered <= area && conflicts <= covered && CurrentPlane[base + 15u] <= covered &&
        (((flags & 4u) != 0u) == (conflicts * 5u > covered || conflicts * 50u > area));
}

bool CurrentAuthorityValid(out uint current_count, out uint epoch) {
    current_count = 0u;
    epoch = 0u;
    uint plane_words, plane_stride, param_words, param_stride;
    CurrentPlane.GetDimensions(plane_words, plane_stride);
    CurrentConditionParams.GetDimensions(param_words, param_stride);
    if (SnapshotGate == 0u ||
        plane_words != V2_SUBTITLE_PLANE_STATE_WORD_COUNT || plane_stride != 4u ||
        param_words != V2_SUBTITLE_CONDITION_PARAM_WORD_COUNT || param_stride != 4u) return false;
    // SnapshotGate already authenticates every current OCR8 pair, source/crop/ROI, and the
    // sealed current geometry/cut publication. No historical rectangle grants authority.
    uint ocr_count = CurrentOcrRecord[4u];
    if (ocr_count == 0u || ocr_count > V2_SUBTITLE_PLANE_COVER_CAPACITY) return false;
    epoch = asuint(SBS_STATE_HARD_CUT_COUNT(CurrentCutBridge[SBS_STATE_VECTOR_HARD_CUT_COUNT]));
    uint kinds = 0u;
    [loop]
    for (uint slot = 0u; slot < ocr_count; ++slot) {
        uint offset = V2_OCR_FINAL_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
        if ((CurrentOcrRecord[offset + 5u] & V2_OCR_BOX_FLAG_RIBBON) != 0u) kinds |= 1u << slot;
    }
    uint target_bits = CurrentConditionParams[V2_SUBTITLE_CONDITION_PARAM_TARGET_WORD];
    float target = asfloat(target_bits);
    uint adaptive = V2_SUBTITLE_PLANE_ADAPTIVE_OFFSET;
    if (CurrentPlane[0u] != V2_SUBTITLE_PLANE_STATE_SCHEMA ||
        CurrentPlane[1u] != V2_SUBTITLE_PLANE_STATE_TAG ||
        CurrentPlane[V2_SUBTITLE_PLANE_FLAGS_WORD] != V2_SUBTITLE_PLANE_CURRENT_VALID_FLAG ||
        CurrentPlane[V2_SUBTITLE_PLANE_CURRENT_SLOT_MASK_WORD] != (1u << ocr_count) - 1u ||
        (CurrentPlane[V2_SUBTITLE_PLANE_HISTORY_RESET_MASK_WORD] & ~((1u << ocr_count) - 1u)) != 0u ||
        CurrentPlane[V2_SUBTITLE_PLANE_CUT_EPOCH_WORD] != epoch ||
        !CurrentAdaptiveTailValid() ||
        CurrentPlane[adaptive + 15u] != 0u ||
        CurrentPlane[adaptive + 3u] != target_bits ||
        CurrentPlane[adaptive + 12u] != asuint(V2DisplayBudget()) ||
        CurrentConditionParams[0u] != V2_SUBTITLE_CONDITION_PARAM_SCHEMA ||
        CurrentConditionParams[1u] != V2_SUBTITLE_CONDITION_PARAM_TAG ||
        CurrentConditionParams[V2_SUBTITLE_CONDITION_PARAM_CURRENT_COUNT_WORD] != ocr_count ||
        CurrentConditionParams[V2_SUBTITLE_CONDITION_PARAM_CURRENT_KIND_MASK_WORD] != kinds ||
        CurrentConditionParams[V2_SUBTITLE_CONDITION_PARAM_ACTIVE_WORD] !=
            V2_SUBTITLE_CONDITION_PARAM_FULLY_ACTIVE_VALUE ||
        !ContourFinite(target) || target < 0.0f || target > V2DisplayBudget()) return false;
    [unroll]
    for (uint component = 0u; component < 2u; ++component) {
        if (CurrentPlane[V2_SUBTITLE_PLANE_FRAME_OFFSET + component] != contour_frame[component] ||
            CurrentPlane[V2_SUBTITLE_PLANE_ANALYSIS_GENERATION_OFFSET + component] != contour_frame[2u + component] ||
            CurrentPlane[V2_SUBTITLE_PLANE_OBSERVATION_TIMESTAMP_OFFSET + component] != contour_observation[component] ||
            CurrentPlane[V2_SUBTITLE_PLANE_SOURCE_EXTENT_OFFSET + component] != contour_source[component] ||
            CurrentPlane[V2_SUBTITLE_PLANE_FIELD_EXTENT_OFFSET + component] != contour_field[component] ||
            CurrentPlane[V2_SUBTITLE_PLANE_ROI_OFFSET + component] != contour_field[2u + component]) return false;
    }
    // Provisional presentation never changes the four-observation RGB detector. Its packed
    // countdown/cue latch is nevertheless part of the authenticated current plane header.
    uint warmup = CurrentPlane[V2_SUBTITLE_PLANE_WARMUP_WORD];
    if ((warmup & ~(V2_SUBTITLE_PLANE_WARMUP_REMAINING_MASK | V2_SUBTITLE_PLANE_WARMUP_CUE_BIT)) != 0u ||
        (warmup & V2_SUBTITLE_PLANE_WARMUP_REMAINING_MASK) > V2_SUBTITLE_PLANE_WARMUP_OBSERVATIONS) return false;
    [unroll]
    for (uint component = 0u; component < 4u; ++component) {
        if (CurrentPlane[V2_SUBTITLE_PLANE_CONTENT_OFFSET + component] != contour_content[component]) return false;
    }
    [unroll]
    for (uint word = V2_SUBTITLE_PLANE_RESERVED_OFFSET;
         word < V2_SUBTITLE_PLANE_RESERVED_OFFSET + V2_SUBTITLE_PLANE_RESERVED_WORD_COUNT; ++word) {
        if (CurrentPlane[word] != 0u) return false;
    }
    current_count = ocr_count;
    return true;
}

void PrepareCurrentGate(uint group_index) {
    if (group_index == 0u) {
        uint count, epoch;
        SnapshotGate = CurrentSnapshotEvidenceValid() ? 1u : 0u;
        // Pins and their published snapshot must share the same global source/OCR verdict.
        // In particular, a rejected ribbon in epoch zero cannot borrow prior valid RGB.
        CurrentGate = CurrentAuthorityValid(count, epoch) && SnapshotGate != 0u ? 1u : 0u;
        CurrentCount = CurrentGate != 0u ? count : 0u;
        CurrentResetMask = CurrentGate != 0u ?
            CurrentPlane[V2_SUBTITLE_PLANE_HISTORY_RESET_MASK_WORD] : 0u;
        CurrentEpoch = SnapshotGate != 0u ?
            asuint(SBS_STATE_HARD_CUT_COUNT(CurrentCutBridge[SBS_STATE_VECTOR_HARD_CUT_COUNT])) : 0u;
        [unroll]
        for (uint slot = 0u; slot < V2_SUBTITLE_PLANE_COVER_CAPACITY; ++slot) {
            CurrentCovers[slot] = CurrentGate != 0u && slot < count ?
                ReadCurrentRect(slot, V2_OCR_FINAL_BOX_OFFSET) : uint4(0u, 0u, 0u, 0u);
            CurrentCores[slot] = CurrentGate != 0u && slot < count ?
                ReadCurrentRect(slot, V2_OCR_RAW_BOX_OFFSET) : uint4(0u, 0u, 0u, 0u);
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

bool PreviousAuthorityValid(out bool duplicate) {
    duplicate = false;
    uint feature_w, feature_h;
    PreviousFeatures.GetDimensions(feature_w, feature_h);
    return CurrentGate != 0u && feature_w == contour_source.x && feature_h == contour_source.w &&
        V2SubtitleHistoryCompatible(PreviousIdentity, contour_source, contour_field, contour_content,
            contour_frame, contour_observation.xy, contour_observation.z, CurrentEpoch, duplicate);
}

void PrepareHistoryGate(uint group_index) {
    if (group_index == 0u) {
        bool duplicate;
        HistoryGate = PreviousAuthorityValid(duplicate) ? 1u : 0u;
        HistoryDuplicate = HistoryGate != 0u && duplicate ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
}

bool BandPositionValid(int2 p) {
    return all(p >= int2(0, 0)) && p.x < (int)contour_source.x && p.y < (int)contour_source.w;
}

bool LoadDisplayColor(int2 p, out float3 color) {
    color = 0.0f;
    if (!BandPositionValid(p)) return false;
    float4 value = CurrentDisplayBand.Load(int3(p, 0));
    if (value.a != 1.0f || !ContourFinite3(value.rgb) || any(value.rgb < 0.0f) || any(value.rgb > 1.0f)) return false;
    color = value.rgb;
    return true;
}

bool SourcePixelInRect(int2 p, uint4 r) {
    if (CurrentGate == 0u || !BandPositionValid(p)) return false;
    // Current source centers and half-open field edges are compared as integer rationals.
    // Covers grant search authority; flagged raw/cores localize count invalidation only.
    uint2 scaled = uint2(
        2u * contour_content.x * contour_source.x +
            (2u * (uint)p.x + 1u) * (contour_content.z - contour_content.x),
        2u * contour_content.y * contour_source.y +
            (2u * ((uint)p.y + contour_source.z) + 1u) * (contour_content.w - contour_content.y));
    return scaled.x >= 2u * r.x * contour_source.x && scaled.x < 2u * r.z * contour_source.x &&
        scaled.y >= 2u * r.y * contour_source.y && scaled.y < 2u * r.w * contour_source.y;
}

bool CoveredSourcePixel(int2 p) {
    [unroll]
    for (uint slot = 0u; slot < V2_SUBTITLE_PLANE_COVER_CAPACITY; ++slot) {
        if (slot < CurrentCount) {
            if (SourcePixelInRect(p, CurrentCovers[slot])) return true;
        }
    }
    return false;
}

bool ChangedCurrentCorePixel(int2 p) {
    [unroll]
    for (uint slot = 0u; slot < V2_SUBTITLE_PLANE_COVER_CAPACITY; ++slot) {
        // Overlap uses only the union of flagged current raw/cores. An expanded cover or an
        // unflagged core cannot recursively extend reset scope through another current box.
        if (slot < CurrentCount && (CurrentResetMask & (1u << slot)) != 0u &&
            SourcePixelInRect(p, CurrentCores[slot])) return true;
    }
    return false;
}


bool CurrentProbabilityEvidence(int2 p, out float probability) {
    probability = 0.0f;
    float2 detector = (float2(p) + 0.5f) *
        float2((float)V2_OCR_OUTPUT_WIDTH / (float)contour_source.x,
               (float)V2_OCR_OUTPUT_HEIGHT / (float)contour_source.w) - 0.5f;
    int2 lo = int2(floor(detector));
    float2 blend = frac(detector);
    int2 a = clamp(lo, int2(0, 0), int2(V2_OCR_OUTPUT_WIDTH - 1u, V2_OCR_OUTPUT_HEIGHT - 1u));
    int2 b = clamp(lo + int2(1, 1), int2(0, 0), int2(V2_OCR_OUTPUT_WIDTH - 1u, V2_OCR_OUTPUT_HEIGHT - 1u));
    float tl = ProbabilityMap[(uint)a.y * V2_OCR_OUTPUT_WIDTH + (uint)a.x];
    float tr = ProbabilityMap[(uint)a.y * V2_OCR_OUTPUT_WIDTH + (uint)b.x];
    float bl = ProbabilityMap[(uint)b.y * V2_OCR_OUTPUT_WIDTH + (uint)a.x];
    float br = ProbabilityMap[(uint)b.y * V2_OCR_OUTPUT_WIDTH + (uint)b.x];
    // Global map validity already participates in OCR8's resolver verdict. A locally invalid
    // probability is still never interpolated into a positive contour candidate.
    if (!ContourFinite(tl) || !ContourFinite(tr) || !ContourFinite(bl) || !ContourFinite(br) ||
        tl < 0.0f || tl > 1.0f || tr < 0.0f || tr > 1.0f ||
        bl < 0.0f || bl > 1.0f || br < 0.0f || br > 1.0f) return false;
    probability = lerp(lerp(tl, tr, blend.x), lerp(bl, br, blend.x), blend.y);
    return true;
}



float ColorDistance(float3 a, float3 b) {
    float3 d = abs(a - b);
    return max(d.x, max(d.y, d.z));
}

bool DecodeFeature(float4 feature, out uint count) {
    count = 0u;
    if (!ContourFinite3(feature.rgb) || !ContourFinite(feature.w) ||
        any(feature.rgb < 0.0f) || any(feature.rgb > 1.0f) || feature.w < 0.0f ||
        feature.w > (float)((V2_SUBTITLE_CONTOUR_HISTORY_CONFIRMATION_OBSERVATIONS <<
            V2_SUBTITLE_CONTOUR_HISTORY_FEATURE_COUNT_SHIFT) |
            V2_SUBTITLE_CONTOUR_HISTORY_FEATURE_VALID_BIT | V2_SUBTITLE_CONTOUR_HISTORY_FEATURE_MASK_BIT))
        return false;
    uint code = (uint)feature.w;
    count = code >> V2_SUBTITLE_CONTOUR_HISTORY_FEATURE_COUNT_SHIFT;
    uint expected = V2_SUBTITLE_CONTOUR_HISTORY_FEATURE_VALID_BIT |
        (count << V2_SUBTITLE_CONTOUR_HISTORY_FEATURE_COUNT_SHIFT) |
        (count == V2_SUBTITLE_CONTOUR_HISTORY_CONFIRMATION_OBSERVATIONS ?
            V2_SUBTITLE_CONTOUR_HISTORY_FEATURE_MASK_BIT : 0u);
    return count <= V2_SUBTITLE_CONTOUR_HISTORY_CONFIRMATION_OBSERVATIONS && feature.w == (float)expected;
}

float4 AdvanceFeature(int2 p, float3 color, bool candidate) {
    float4 reset = float4(color, (float)V2_SUBTITLE_CONTOUR_HISTORY_FEATURE_VALID_BIT);
    if (!candidate) return reset;
    uint count = 0u;
    float3 anchor = color;
    if (HistoryDuplicate != 0u) {
        float4 same = PreviousFeatures.Load(int3(p, 0));
        uint old_count;
        // Repeated timestamps/frame IDs never count as a new confirmation or discover a pixel.
        if (PreviousIdentity[2u] == 1u && DecodeFeature(same, old_count) && old_count != 0u &&
            ColorDistance(color, same.rgb) <= V2_SUBTITLE_CONTOUR_HISTORY_RGB_DISTANCE &&
            (old_count > 1u || all(asuint(same.rgb) == asuint(color))))
            return same;
        return reset;
    }
    count = 1u;
    // Only a genuine newer observation reaches here. A qualifying same-pixel fixed reference
    // wins even in a flagged current raw/core; duplicate/reuse consumed no confirmation above.
    if (HistoryGate != 0u && PreviousIdentity[2u] == 1u) {
        bool found = false;
        float4 same = PreviousFeatures.Load(int3(p, 0));
        uint same_count;
        // Keep a valid fixed reference at this pixel. A closer colour in a newly admitted
        // neighbour must not replace an already stable reference and restart its streak.
        bool same_counted = DecodeFeature(same, same_count) && same_count != 0u;
        if (same_counted && ColorDistance(color, same.rgb) <= V2_SUBTITLE_CONTOUR_HISTORY_RGB_DISTANCE) {
            found = true;
            anchor = same.rgb;
            count = min(same_count + 1u, V2_SUBTITLE_CONTOUR_HISTORY_CONFIRMATION_OBSERVATIONS);
        }
        float best_color = 2.0f;
        int best_space = 0;
        // A coordinate previously outside current authority (count0) or invalid must acquire
        // its own first observation. Nearby mature colors cannot preconfirm newly added covers.
        // A flagged current raw/core mismatch starts at count1/current RGB without borrowing
        // a mature neighbour. Otherwise counted coordinates retain bounded jitter correspondence.
        if (!found && same_counted && !ChangedCurrentCorePixel(p)) {
          [loop]
          for (int dy = -HISTORY_JITTER_RADIUS; dy <= HISTORY_JITTER_RADIUS; ++dy) {
            [loop]
            for (int dx = -HISTORY_JITTER_RADIUS; dx <= HISTORY_JITTER_RADIUS; ++dx) {
                int2 q = p + int2(dx, dy);
                if (!BandPositionValid(q)) continue;
                float4 prior = PreviousFeatures.Load(int3(q, 0));
                uint prior_count;
                if (!DecodeFeature(prior, prior_count) || prior_count == 0u) continue;
                float distance = ColorDistance(color, prior.rgb);
                int space = dx * dx + dy * dy;
                // Select one correspondence, never pool counts or prefer the oldest candidate.
                if (distance <= V2_SUBTITLE_CONTOUR_HISTORY_RGB_DISTANCE &&
                    (!found || distance < best_color || (distance == best_color && space < best_space))) {
                    found = true;
                    best_color = distance;
                    best_space = space;
                    anchor = prior.rgb;
                    count = min(prior_count + 1u, V2_SUBTITLE_CONTOUR_HISTORY_CONFIRMATION_OBSERVATIONS);
                }
            }
          }
        }
    }
    uint code = V2_SUBTITLE_CONTOUR_HISTORY_FEATURE_VALID_BIT |
        (count << V2_SUBTITLE_CONTOUR_HISTORY_FEATURE_COUNT_SHIFT) |
        (count == V2_SUBTITLE_CONTOUR_HISTORY_CONFIRMATION_OBSERVATIONS ?
            V2_SUBTITLE_CONTOUR_HISTORY_FEATURE_MASK_BIT : 0u);
    return float4(anchor, (float)code);
}

[numthreads(16, 16, 1)]
void temporal_main(uint3 dispatch_id : SV_DispatchThreadID, uint group_index : SV_GroupIndex) {
    PrepareCurrentGate(group_index);
    PrepareHistoryGate(group_index);
    uint width, height, feature_w, feature_h;
    NativeSupportWrite.GetDimensions(width, height);
    HistoryFeaturesWrite.GetDimensions(feature_w, feature_h);
    bool valid = SnapshotGate != 0u && width == contour_source.x && height == contour_source.w &&
        feature_w == width && feature_h == height;
    uint2 stride = SourceDispatchStride();
    [loop]
    for (uint y = dispatch_id.y; y < height; y += stride.y) {
        [loop]
        for (uint x = dispatch_id.x; x < width; x += stride.x) {
            int2 p = int2(x, y);
            float3 color;
            float probability;
            float4 feature = 0.0f;
            uint mask = 0u;
            if (valid && LoadDisplayColor(p, color)) {
                bool candidate = false;
                if (CurrentGate != 0u) {
                    // Current-frame OCR8 covers authorize each pixel independently.
                    // Box count, ordering, splits and merges are not RGB-history reset keys.
                    candidate = CoveredSourcePixel(p) &&
                        CurrentProbabilityEvidence(p, probability);
                }
                feature = AdvanceFeature(p, color, candidate);
                mask = (uint)feature.w & V2_SUBTITLE_CONTOUR_HISTORY_FEATURE_MASK_BIT;
            }
            NativeSupportWrite[uint2(x, y)] = mask;
            if (x < feature_w && y < feature_h) HistoryFeaturesWrite[uint2(x, y)] = feature;
        }
    }
}

int FloorDivide(int numerator, int denominator) {
    int quotient = numerator / denominator;
    return quotient - (numerator < 0 && numerator % denominator != 0 ? 1 : 0);
}

int CeilDivide(int numerator, int denominator) {
    int quotient = numerator / denominator;
    return quotient + (numerator > 0 && numerator % denominator != 0 ? 1 : 0);
}

[numthreads(16, 16, 1)]
void project_main(uint3 dispatch_id : SV_DispatchThreadID, uint group_index : SV_GroupIndex) {
    PrepareCurrentGate(group_index);
    uint width, height, native_width, native_height;
    FieldPinWrite.GetDimensions(width, height);
    CurrentSupportRead.GetDimensions(native_width, native_height);
    if (dispatch_id.x >= width || dispatch_id.y >= height) return;
    uint pin = 0u;
    if (CurrentGate != 0u && width == contour_field.x && height == contour_field.y &&
        native_width == contour_source.x && native_height == contour_source.w &&
        all(dispatch_id.xy >= contour_content.xy) && all(dispatch_id.xy < contour_content.zw)) {
        int2 local = int2(dispatch_id.xy) - int2(contour_content.xy);
        int2 content_size = int2(contour_content.zw - contour_content.xy);
        // A source pixel occupies [L+x*C/S-.5,L+(x+1)*C/S-.5] in field-node coordinates.
        // Gather every admitted pixel area intersecting this node's open +/-1 bilinear support.
        // Signed integer floor/ceil preserves conservative bounds without SM5 reciprocal error.
        int begin_x = max(0, FloorDivide((2 * local.x - 1) * (int)contour_source.x, 2 * content_size.x));
        int end_x = min((int)contour_source.x,
            CeilDivide((2 * local.x + 3) * (int)contour_source.x, 2 * content_size.x));
        int begin_y = max((int)contour_source.z,
            FloorDivide((2 * local.y - 1) * (int)contour_source.y, 2 * content_size.y));
        int end_y = min((int)(contour_source.z + contour_source.w),
            CeilDivide((2 * local.y + 3) * (int)contour_source.y, 2 * content_size.y));
        [loop]
        for (int y = begin_y; y < end_y && pin == 0u; ++y) {
            [loop]
            for (int x = begin_x; x < end_x; ++x) {
                uint support = CurrentSupportRead.Load(int3(x, y - (int)contour_source.z, 0));
                if (support == 1u) {
                    pin = 1u;
                    break;
                }
            }
        }
    }
    FieldPinWrite[dispatch_id.xy] = pin;
}



uint HistoryWord(uint word, bool valid) {
    if (word == 0u) return CONTOUR_SCHEMA;
    if (word == 1u) return HISTORY_TAG;
    if (word == 2u) return valid && CurrentGate != 0u ? 1u : 0u;
    if (word == 3u) return valid ? CurrentEpoch : 0u;
    if (word >= 4u && word < 8u) return contour_frame[word - 4u];
    if (word >= 8u && word < 12u) return contour_source[word - 8u];
    if (word >= 12u && word < 16u) return contour_field[word - 12u];
    if (word >= 16u && word < 20u) return contour_content[word - 16u];
    if (word == 20u) return contour_observation.x;
    if (word == 21u) return contour_observation.y;
    if (word == 22u) return RECIPE_SCHEMA;
    if (word == V2_SUBTITLE_CONTOUR_HISTORY_SNAPSHOT_VALID_WORD) return valid ? 1u : 0u;
    // SCT5 carries source/clock/cut identity only. Reserved24..31 are canonical zero;
    // changing current OCR count, order, kinds or cover segmentation is not a history reset.
    return 0u;
}

[numthreads(16, 16, 1)]
void history_main(uint3 dispatch_id : SV_DispatchThreadID, uint group_index : SV_GroupIndex) {
    // Unbind both old history SRVs before writing the persistent history UAVs.
    PrepareCurrentGate(group_index);
    uint width, height, native_width, native_height, words, word_stride, next_w, next_h;
    HistoryFeaturesWrite.GetDimensions(width, height);
    NextFeaturesRead.GetDimensions(next_w, next_h);
    CurrentSupportRead.GetDimensions(native_width, native_height);
    HistoryIdentityWrite.GetDimensions(words, word_stride);
    bool valid = SnapshotGate != 0u && width == contour_source.x && height == contour_source.w &&
        native_width == width && native_height == height && next_w == width && next_h == height &&
        words == HISTORY_WORDS && word_stride == 4u;
    uint2 stride = SourceDispatchStride();
    [loop]
    for (uint y = dispatch_id.y; y < height; y += stride.y) {
        [loop]
        for (uint x = dispatch_id.x; x < width; x += stride.x) {
            float4 feature = 0.0f;
            float3 color;
            if (valid && LoadDisplayColor(int2(x, y), color)) {
                // The immutable next texture carries the chosen reference/count. Reading old
                // neighbor history while overwriting it would otherwise race across threadgroups.
                feature = NextFeaturesRead.Load(int3(x, y, 0));
            }
            HistoryFeaturesWrite[uint2(x, y)] = feature;
        }
    }
    if (dispatch_id.y == 0u) {
        [loop]
        for (uint word = dispatch_id.x; word < words; word += stride.x)
            HistoryIdentityWrite[word] = words == HISTORY_WORDS && word_stride == 4u ? HistoryWord(word, valid) : 0u;
    }
}
