// Native subtitle contours use current detector evidence inside current OCR covers.
// Fixed-anchor RGB confirmation remains diagnostic history, not presentation authority.
// Detector evidence learns a current text palette; it is not a glyph mask. Current native
// stroke/outline evidence selects binary support within each current paired cover.
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
RWTexture2D<uint> DenseGlyphMatteWrite : register(u6); // native R32_UINT literal 0/1, never aliases t5

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
groupshared int CurrentReferenceCenterTwice;
groupshared uint4 CurrentCovers[V2_SUBTITLE_PLANE_COVER_CAPACITY];
groupshared uint4 CurrentCores[V2_SUBTITLE_PLANE_COVER_CAPACITY];
groupshared uint4 CurrentSupportSearch[V2_SUBTITLE_PLANE_COVER_CAPACITY];
groupshared uint HistoryGate;
groupshared uint HistoryDuplicate;
groupshared uint SnapshotGate;
groupshared uint PaletteNeeded;
groupshared uint PaletteHistogramCount[V2_SUBTITLE_CONTOUR_PALETTE_BIN_COUNT];
groupshared uint PaletteHistogramRepresentative[V2_SUBTITLE_CONTOUR_PALETTE_BIN_COUNT];
groupshared uint PaletteAnchorPeak;
groupshared uint PaletteCount[V2_SUBTITLE_PLANE_COVER_CAPACITY];
groupshared uint4 PaletteMode[V2_SUBTITLE_PLANE_COVER_CAPACITY * V2_SUBTITLE_CONTOUR_PALETTE_MAX_MODES];

// The dense current glyph matte is a native-color membership resource, independent of the
// diagnostic sparse contour/distance field. Each output tile owns its complete dependency
// halo, so all bounded same-frame growth happens in one dispatch without inter-group reads.
#define GLYPH_TILE_SIDE 16u
#define GLYPH_TILE_HALO (V2_SUBTITLE_GLYPH_MATTE_STEPS + V2_SUBTITLE_GLYPH_MATTE_COLLAR_RADIUS_SOURCE_PIXELS)
#define GLYPH_TILE_EXTENT (GLYPH_TILE_SIDE + 2u * GLYPH_TILE_HALO)
#define GLYPH_TILE_PIXELS (GLYPH_TILE_EXTENT * GLYPH_TILE_EXTENT)
groupshared float4 GlyphTileColors[GLYPH_TILE_PIXELS];
groupshared uint GlyphTileMask[GLYPH_TILE_PIXELS];
groupshared uint GlyphTileNext[GLYPH_TILE_PIXELS];
groupshared uint2 GlyphTileGroupStride;

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

bool CurrentClassifierReferenceValid(out int reference_center) {
    reference_center = (int)contour_source.x;
    uint base = V2_OCR_CLASSIFIER_HISTORY_OFFSET;
    bool empty = true;
    [loop]
    for (uint word = 0u; word < V2_OCR_CLASSIFIER_HISTORY_WORD_COUNT; ++word)
        empty = empty && CurrentOcrRecord[base + word] == 0u;
    if (empty) return true;
    uint2 clock = uint2(CurrentOcrRecord[base + 4u], CurrentOcrRecord[base + 5u]);
    if (CurrentOcrRecord[base] != V2_OCR_CLASSIFIER_HISTORY_TAG ||
        CurrentOcrRecord[base + 1u] != V2_OCR_CLASSIFIER_HISTORY_SCHEMA ||
        CurrentOcrRecord[base + 2u] != V2_OCR_CLASSIFIER_HISTORY_KNOWN_FLAGS ||
        CurrentOcrRecord[base + 3u] > V2_OCR_CLASSIFIER_PRIOR_CAP ||
        !any(clock != 0u) || any(clock != contour_observation.xy)) return false;
    [unroll]
    for (uint lane = 0u; lane < 4u; ++lane) {
        if (CurrentOcrRecord[base + 6u + lane] != contour_frame[lane] ||
            CurrentOcrRecord[base + 10u + lane] != contour_source[lane] ||
            CurrentOcrRecord[base + 16u + lane] != contour_content[lane]) return false;
    }
    if (CurrentOcrRecord[base + 14u] != contour_field.x ||
        CurrentOcrRecord[base + 15u] != contour_field.y) return false;
    uint reference_width = CurrentOcrRecord[base + 21u];
    int center = asint(CurrentOcrRecord[base + 20u]);
    int search_top = asint(CurrentOcrRecord[base + 22u]);
    if (reference_width == 0u || reference_width > MAX_SOURCE_EXTENT ||
        center < -(int)reference_width || center > (int)reference_width ||
        search_top < -(int)MAX_SOURCE_EXTENT || search_top > (int)MAX_SOURCE_EXTENT ||
        CurrentOcrRecord[base + 23u] > 1u || CurrentOcrRecord[base + 24u] > 1u ||
        CurrentOcrRecord[base + 30u] != 0u || CurrentOcrRecord[base + 31u] != 0u) return false;
    uint2 gap = uint2(CurrentOcrRecord[base + 25u], CurrentOcrRecord[base + 26u]);
    if (CurrentOcrRecord[base + 24u] == 0u) {
        if (any(gap != 0u)) return false;
    } else if (!any(gap != 0u) || ClockLess(clock, gap)) return false;
    float trusted_height = asfloat(CurrentOcrRecord[base + 27u]);
    uint trusted_top = CurrentOcrRecord[base + 28u], trusted_bottom = CurrentOcrRecord[base + 29u];
    if (!ContourFinite(trusted_height) || trusted_height < 0.0f ||
        trusted_height > (float)(V2_OCR_SAFE_ROW_BOTTOM - V2_OCR_SAFE_ROW_TOP)) return false;
    if (trusted_height == 0.0f) {
        if (trusted_top != 0u || trusted_bottom != 0u) return false;
    } else if (trusted_top < V2_OCR_SAFE_ROW_TOP || trusted_top >= trusted_bottom ||
        trusted_bottom > V2_OCR_SAFE_ROW_BOTTOM) return false;
    bool votes = false;
    [unroll]
    for (uint bin = 0u; bin < V2_OCR_CLASSIFIER_PRIOR_BIN_COUNT; ++bin) {
        uint vote = CurrentOcrRecord[base + 32u + bin];
        if (vote > V2_OCR_CLASSIFIER_PRIOR_CAP) return false;
        votes = votes || vote != 0u;
    }
    if (votes && (trusted_height <= 0.0f || CurrentOcrRecord[base + 3u] == 0u)) return false;
    reference_center = center;
    return true;
}

bool CurrentSnapshotEvidenceValid(out int reference_center) {
    reference_center = (int)contour_source.x;
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
    return CurrentClassifierReferenceValid(reference_center);
}

[numthreads(16, 16, 1)]
void snapshot_main(uint3 dispatch_id : SV_DispatchThreadID, uint group_index : SV_GroupIndex) {
    if (group_index == 0u) {
        int reference_center;
        SnapshotGate = CurrentSnapshotEvidenceValid(reference_center) ? 1u : 0u;
    }
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
    uint2 retreat = uint2(CurrentPlane[base + 8u], CurrentPlane[base + 9u]);
    float applied = asfloat(CurrentPlane[base + 3u]);
    uint maximum_level = (contour_source.x + 99u) / 100u;
    uint area = (contour_content.z - contour_content.x) *
        (contour_content.w - contour_content.y);
    uint covered = CurrentPlane[base + 13u];
    uint conflicts = CurrentPlane[base + 14u];
    // Accepted controller time may be held or disarmed independently of this source tuple.
    return CurrentPlane[base] == 1u && (flags & ~V2_SUBTITLE_PLANE_ADAPTIVE_KNOWN_FLAGS) == 0u &&
        CurrentPlane[base + 6u] == 0u && CurrentPlane[base + 7u] == 0u &&
        CurrentPlane[base + 8u] == 0u && CurrentPlane[base + 9u] == 0u &&
        CurrentPlane[base + 10u] == 0u && CurrentPlane[base + 11u] == 0u &&
        CurrentPlane[base + 2u] <= maximum_level &&
        CurrentPlane[base + 10u] <= maximum_level && CurrentPlane[base + 11u] <= maximum_level &&
        ContourFinite(applied) && applied >= 0.0f && applied <= V2DisplayBudget() &&
        CurrentPlane[base + 12u] == asuint(V2DisplayBudget()) &&
        (((flags & 8u) != 0u) == any(clock != 0u)) &&
        (((flags & 2u) != 0u) == any(retreat != 0u)) &&
        !ClockLess(clock, retreat) &&
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
    // Current probability owns presentation. The retired provisional countdown word is zero.
    if (CurrentPlane[V2_SUBTITLE_PLANE_WARMUP_WORD] != 0u) return false;
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

bool WideGreater(uint a_low, uint a_high, uint b_low, uint b_high) {
    return a_high > b_high || (a_high == b_high && a_low > b_low);
}

void CurrentWideProduct(uint a, uint b, out uint low, out uint high) {
    // FXC/SM5 has no umulExtended intrinsic. Split both operands into 16-bit limbs;
    // every intermediate fits uint32, including carries, and the result is exact.
    uint p0 = (a & 65535u) * (b & 65535u);
    uint t = (a >> 16u) * (b & 65535u) + (p0 >> 16u);
    uint cross = (a & 65535u) * (b >> 16u) + (t & 65535u);
    low = (cross << 16u) | (p0 & 65535u);
    high = (a >> 16u) * (b >> 16u) + (t >> 16u) + (cross >> 16u);
}

bool CurrentSlotRibbon(uint slot) {
    return (CurrentOcrRecord[V2_OCR_RAW_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT + 5u] &
        V2_OCR_BOX_FLAG_RIBBON) != 0u;
}

uint4 CurrentOrdinarySupportSearch(uint slot) {
    uint4 cover = CurrentCovers[slot], core = CurrentCores[slot];
    // Ribbon covers have their own producer meaning; none of the ordinary expansion is
    // applied to them. Group only against this fixed core, never a growing/transitive union.
    if (CurrentSlotRibbon(slot)) return cover;
    uint left = core.x, right = core.z;
    [loop]
    for (uint member = 0u; member < CurrentCount; ++member) {
        if (CurrentSlotRibbon(member)) continue;
        uint4 other = CurrentCores[member];
        uint overlap_top = max(core.y, other.y), overlap_bottom = min(core.w, other.w);
        if (overlap_bottom <= overlap_top) continue;
        uint intersection = overlap_bottom - overlap_top;
        uint union_height = max(core.w, other.w) - min(core.y, other.y);
        if (intersection * V2_OCR_LINE_GROUP_MIN_VERTICAL_IOU_DENOMINATOR <
            union_height * V2_OCR_LINE_GROUP_MIN_VERTICAL_IOU_NUMERATOR) continue;
        left = min(left, other.x);
        right = max(right, other.z);
    }
    uint content_width = contour_content.z - contour_content.x;
    uint center_field = left + right - 2u * contour_content.x;
    uint line_low, line_high;
    CurrentWideProduct(center_field * contour_source.x, V2_OCR_OUTPUT_WIDTH, line_low, line_high);
    int reference_scaled = CurrentReferenceCenterTwice * (int)V2_OCR_OUTPUT_WIDTH;
    int margin = (int)(2u * V2_SUBTITLE_CONTOUR_ORDINARY_SEARCH_CENTER_BIAS_DETECTOR_PIXELS * contour_source.x);
    int upper = reference_scaled + margin, lower = reference_scaled - margin;
    bool right_biased = upper < 0;
    if (upper >= 0) {
        uint upper_low, upper_high;
        CurrentWideProduct((uint)upper, content_width, upper_low, upper_high);
        right_biased = WideGreater(line_low, line_high, upper_low, upper_high);
    }
    bool left_biased = false;
    if (lower > 0) {
        uint lower_low, lower_high;
        CurrentWideProduct((uint)lower, content_width, lower_low, lower_high);
        left_biased = WideGreater(lower_low, lower_high, line_low, line_high);
    }
    uint inset = V2_SUBTITLE_CONTOUR_ORDINARY_SEARCH_INSET_DETECTOR_PIXELS * content_width / V2_OCR_OUTPUT_WIDTH;
    // Preserve the current core and any producer side already clamped to actual content.
    if (cover.x != contour_content.x && !right_biased) cover.x = min(core.x, cover.x + inset);
    if (cover.z != contour_content.z && !left_biased) cover.z = max(core.z, cover.z - min(inset, cover.z));
    uint pad = ((core.w - core.y) * V2_SUBTITLE_CONTOUR_ORDINARY_SEARCH_PAD_HEIGHT_NUMERATOR +
        V2_SUBTITLE_CONTOUR_ORDINARY_SEARCH_PAD_HEIGHT_DENOMINATOR - 1u) /
        V2_SUBTITLE_CONTOUR_ORDINARY_SEARCH_PAD_HEIGHT_DENOMINATOR;
    cover.y = max(cover.y, core.y - min(pad, core.y));
    return cover;
}

void PrepareCurrentGate(uint group_index) {
    if (group_index == 0u) {
        uint count, epoch;
        int reference_center;
        SnapshotGate = CurrentSnapshotEvidenceValid(reference_center) ? 1u : 0u;
        CurrentReferenceCenterTwice = reference_center;
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
            CurrentSupportSearch[slot] = CurrentCovers[slot];
        }
        [loop]
        for (uint active_slot = 0u; active_slot < CurrentCount; ++active_slot)
            CurrentSupportSearch[active_slot] = CurrentOrdinarySupportSearch(active_slot);
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
    // The supplied rectangle is a current coarse cover, paired core, or derived support search.
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

bool SupportSearchSourcePixel(int2 p) {
    [unroll]
    for (uint slot = 0u; slot < V2_SUBTITLE_PLANE_COVER_CAPACITY; ++slot) {
        if (slot < CurrentCount) {
            if (SourcePixelInRect(p, CurrentSupportSearch[slot])) return true;
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
    // This strict rendering threshold is also authenticated by saved-state readers. Keep
    // each FP32 operation and its rounding point explicit; fused lerp/MAD can cross it.
    precise float2 ratio = float2((float)V2_OCR_OUTPUT_WIDTH / (float)contour_source.x,
        (float)V2_OCR_OUTPUT_HEIGHT / (float)contour_source.w);
    precise float2 centered = float2(p) + 0.5f;
    precise float2 scaled = centered * ratio;
    precise float2 detector = scaled - 0.5f;
    int2 lo = int2(floor(detector));
    precise float2 blend = detector - floor(detector);
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
    precise float upper_delta = tr - tl;
    precise float upper_product = upper_delta * blend.x;
    precise float upper = tl + upper_product;
    precise float lower_delta = br - bl;
    precise float lower_product = lower_delta * blend.x;
    precise float lower = bl + lower_product;
    precise float vertical_delta = lower - upper;
    precise float vertical_product = vertical_delta * blend.y;
    precise float sampled = upper + vertical_product;
    probability = sampled;
    return true;
}



float ColorDistance(float3 a, float3 b) {
    float3 d = abs(a - b);
    return max(d.x, max(d.y, d.z));
}

bool CurrentContrastAt(int2 p, float3 color) {
    float3 neighbor;
    if (!LoadDisplayColor(p, neighbor)) return false;
    return ColorDistance(color, neighbor) > V2_SUBTITLE_CONTOUR_SPATIAL_EDGE_RGB_DISTANCE;
}

bool2 CurrentSpatialEdges(int2 p, float3 color) {
    const int radius = (int)V2_SUBTITLE_CONTOUR_SPATIAL_EDGE_RADIUS_SOURCE_PIXELS;
    bool x = CurrentContrastAt(p + int2(radius, 0), color) || CurrentContrastAt(p - int2(radius, 0), color);
    bool y = CurrentContrastAt(p + int2(0, radius), color) || CurrentContrastAt(p - int2(0, radius), color);
    return bool2(x, y);
}

bool CurrentSpatialEdge(int2 p, float3 color) {
    return any(CurrentSpatialEdges(p, color));
}

uint MaxChannel(uint3 value) { return max(value.x, max(value.y, value.z)); }

uint3 PaletteQuantize(float3 color) {
    precise float3 scaled = color * (float)V2_SUBTITLE_CONTOUR_PALETTE_CHANNEL_MAX;
    precise float3 rounded = scaled + 0.5f;
    return (uint3)floor(rounded);
}

bool SourceTileTouchesCurrentCover(uint2 first) {
    if (any(first >= uint2(contour_source.x, contour_source.w))) return false;
    uint2 last = min(first + 15u, uint2(contour_source.x - 1u, contour_source.w - 1u));
    uint2 content = contour_content.zw - contour_content.xy;
    uint2 start = uint2(2u * contour_content.x * contour_source.x + (2u * first.x + 1u) * content.x,
        2u * contour_content.y * contour_source.y + (2u * (first.y + contour_source.z) + 1u) * content.y);
    uint2 finish = uint2(2u * contour_content.x * contour_source.x + (2u * last.x + 1u) * content.x,
        2u * contour_content.y * contour_source.y + (2u * (last.y + contour_source.z) + 1u) * content.y);
    [loop]
    for (uint slot = 0u; slot < CurrentCount; ++slot) {
        uint4 r = CurrentCovers[slot];
        if (finish.x >= 2u * r.x * contour_source.x && start.x < 2u * r.z * contour_source.x &&
            finish.y >= 2u * r.y * contour_source.y && start.y < 2u * r.w * contour_source.y) return true;
    }
    return false;
}

int2 CurrentPaletteSamplePosition(uint4 core, uint sample_index) {
    uint gx = sample_index % V2_SUBTITLE_CONTOUR_PALETTE_GRID_COLUMNS;
    uint gy = sample_index / V2_SUBTITLE_CONTOUR_PALETTE_GRID_COLUMNS;
    uint content_width = contour_content.z - contour_content.x;
    uint content_height = contour_content.w - contour_content.y;
    // Integer field-core grid centers, then floor to a native source pixel. At the
    // bounded 5120 extents even the 64-column numerator fits an unsigned 32-bit word.
    uint x = (((core.x - contour_content.x) * (2u * V2_SUBTITLE_CONTOUR_PALETTE_GRID_COLUMNS) +
        (2u * gx + 1u) * (core.z - core.x)) * contour_source.x) /
        (2u * V2_SUBTITLE_CONTOUR_PALETTE_GRID_COLUMNS * content_width);
    uint absolute_y = (((core.y - contour_content.y) * (2u * V2_SUBTITLE_CONTOUR_PALETTE_GRID_ROWS) +
        (2u * gy + 1u) * (core.w - core.y)) * contour_source.y) /
        (2u * V2_SUBTITLE_CONTOUR_PALETTE_GRID_ROWS * content_height);
    return int2(min(x, contour_source.x - 1u),
        clamp((int)absolute_y - (int)contour_source.z, 0, (int)contour_source.w - 1));
}

uint3 PaletteHue(uint3 color) {
    uint peak = max(MaxChannel(color), 1u);
    return (color * V2_SUBTITLE_CONTOUR_PALETTE_HUE_CHANNEL_MAX + peak / 2u) / peak;
}

void PrepareCurrentPalettes(uint group_index, uint2 dispatch_position) {
    if (group_index < V2_SUBTITLE_PLANE_COVER_CAPACITY) {
        PaletteCount[group_index] = 0u;
    }
    if (group_index == 0u) {
        PaletteNeeded = 0u;
        uint2 first = dispatch_position - uint2(group_index % 16u, group_index / 16u);
        uint2 stride = SourceDispatchStride();
        if (CurrentGate != 0u) {
            // Include every grid-stride tile owned by this group, including a second native
            // tile to the right of the field grid. Outside groups still publish normal history.
            [loop]
            for (uint y = first.y; y < contour_source.w && PaletteNeeded == 0u; y += stride.y) {
                [loop]
                for (uint x = first.x; x < contour_source.x; x += stride.x) {
                    if (SourceTileTouchesCurrentCover(uint2(x, y))) { PaletteNeeded = 1u; break; }
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (PaletteNeeded == 0u) return;
    [loop]
    for (uint slot = 0u; slot < CurrentCount; ++slot) {
        if (group_index < V2_SUBTITLE_CONTOUR_PALETTE_BIN_COUNT) {
            PaletteHistogramCount[group_index] = 0u;
            PaletteHistogramRepresentative[group_index] = 0u;
        }
        if (group_index == 0u) PaletteAnchorPeak = 0u;
        GroupMemoryBarrierWithGroupSync();
        uint4 core = CurrentCores[slot];
        [loop]
        for (uint sample_index = group_index;
             sample_index < V2_SUBTITLE_CONTOUR_PALETTE_GRID_COLUMNS * V2_SUBTITLE_CONTOUR_PALETTE_GRID_ROWS;
             sample_index += 256u) {
            int2 p = CurrentPaletteSamplePosition(core, sample_index);
            float3 color;
            float probability;
            bool color_valid = LoadDisplayColor(p, color);
            bool probability_valid = CurrentProbabilityEvidence(p, probability);
            bool anchor = false;
            if (color_valid && probability_valid) {
                if (probability > V2_OCR_ACTIVE_PROBABILITY_THRESHOLD) anchor = CurrentSpatialEdge(p, color);
            }
            if (anchor) {
                uint3 q = PaletteQuantize(color);
                // Bin from actual RGB, not q>>14: RGB.75 rounds to q49151.
                uint3 bin = min((uint3)(color * 4.0f), uint3(3u, 3u, 3u));
                uint code = bin.x * 16u + bin.y * 4u + bin.z;
                uint peak = MaxChannel(q);
                uint key = (peak << V2_SUBTITLE_CONTOUR_PALETTE_REPRESENTATIVE_INDEX_BITS) |
                    (V2_SUBTITLE_CONTOUR_PALETTE_REPRESENTATIVE_INDEX_MASK - sample_index);
                InterlockedAdd(PaletteHistogramCount[code], 1u);
                InterlockedMax(PaletteHistogramRepresentative[code], key);
                InterlockedMax(PaletteAnchorPeak, peak);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (group_index == 0u) {
            // Scalar bitsets avoid FXC forcing an unroll for a dynamically written vector lane.
            uint visited_low = 0u, visited_high = 0u;
            [loop]
            for (uint attempt = 0u; attempt < V2_SUBTITLE_CONTOUR_PALETTE_BIN_COUNT; ++attempt) {
                uint best = V2_SUBTITLE_CONTOUR_PALETTE_BIN_COUNT;
                uint best_peak = 0u, best_count = 1u;
                [loop]
                for (uint bin = 0u; bin < V2_SUBTITLE_CONTOUR_PALETTE_BIN_COUNT; ++bin) {
                    uint count = PaletteHistogramCount[bin];
                    uint peak = PaletteHistogramRepresentative[bin] >> V2_SUBTITLE_CONTOUR_PALETTE_REPRESENTATIVE_INDEX_BITS;
                    uint visited_word = bin < 32u ? visited_low : visited_high;
                    if (count == 0u || (visited_word & (1u << (bin % 32u))) != 0u ||
                        V2_SUBTITLE_CONTOUR_PALETTE_BRIGHTNESS_DENOMINATOR * peak <
                            V2_SUBTITLE_CONTOUR_PALETTE_BRIGHTNESS_NUMERATOR * PaletteAnchorPeak) continue;
                    if (best == V2_SUBTITLE_CONTOUR_PALETTE_BIN_COUNT || peak > best_peak ||
                        (peak == best_peak && (count > best_count || (count == best_count && bin < best)))) {
                        best = bin; best_peak = peak; best_count = count;
                    }
                }
                if (best == V2_SUBTITLE_CONTOUR_PALETTE_BIN_COUNT) break;
                if (best < 32u) visited_low |= 1u << best;
                else visited_high |= 1u << (best - 32u);
                uint representative_index = V2_SUBTITLE_CONTOUR_PALETTE_REPRESENTATIVE_INDEX_MASK -
                    (PaletteHistogramRepresentative[best] & V2_SUBTITLE_CONTOUR_PALETTE_REPRESENTATIVE_INDEX_MASK);
                float3 representative_color;
                bool representative_valid = LoadDisplayColor(CurrentPaletteSamplePosition(core, representative_index), representative_color);
                if (!representative_valid) continue;
                uint3 representative = PaletteQuantize(representative_color);
                uint3 hue = PaletteHue(representative);
                bool represented = false;
                [loop]
                for (uint prior_mode = 0u; prior_mode < PaletteCount[slot]; ++prior_mode) {
                    uint4 prior = PaletteMode[slot * V2_SUBTITLE_CONTOUR_PALETTE_MAX_MODES + prior_mode];
                    uint3 prior_rounded = (prior.xyz + prior.w / 2u) / prior.w;
                    uint3 prior_hue = PaletteHue(prior_rounded);
                    uint distance = MaxChannel((uint3)abs(int3(hue) - int3(prior_hue)));
                    if (V2_SUBTITLE_CONTOUR_PALETTE_MERGE_HUE_DENOMINATOR * distance <=
                        V2_SUBTITLE_CONTOUR_PALETTE_MERGE_HUE_NUMERATOR * V2_SUBTITLE_CONTOUR_PALETTE_HUE_CHANNEL_MAX)
                        represented = true;
                }
                if (represented) continue;
                uint selected_mode = PaletteCount[slot];
                // Keep the existing exact rational matcher representation: virtual sum/count.
                PaletteMode[slot * V2_SUBTITLE_CONTOUR_PALETTE_MAX_MODES + selected_mode] =
                    uint4(representative * best_count, best_count);
                PaletteCount[slot] = selected_mode + 1u;
                if (PaletteCount[slot] == V2_SUBTITLE_CONTOUR_PALETTE_MAX_MODES) break;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

bool CurrentPaletteMatches(uint slot, uint3 color) {
    [loop]
    for (uint mode = 0u; mode < PaletteCount[slot]; ++mode) {
        uint4 value = PaletteMode[slot * V2_SUBTITLE_CONTOUR_PALETTE_MAX_MODES + mode];
        int3 delta = int3(color * value.w) - int3(value.xyz);
        uint distance = MaxChannel((uint3)abs(delta));
        uint scale = V2_SUBTITLE_CONTOUR_PALETTE_CHANNEL_MAX * value.w;
        if (V2_SUBTITLE_CONTOUR_PALETTE_MATCH_RGB_DENOMINATOR * distance <=
            V2_SUBTITLE_CONTOUR_PALETTE_MATCH_RGB_NUMERATOR * scale) return true;
    }
    return false;
}

bool CurrentShortTangent(int2 p, float3 color, int2 direction, uint maximum) {
    uint length = 1u;
    if (maximum == 0u) return false;
    [unroll]
    for (int sign = -1; sign <= 1; sign += 2) {
        [loop]
        for (uint distance = 1u; distance <= maximum; ++distance) {
            float3 neighbor;
            bool neighbor_valid = LoadDisplayColor(p + direction * sign * (int)distance, neighbor);
            if (!neighbor_valid) break;
            float contrast = ColorDistance(color, neighbor);
            if (contrast > V2_SUBTITLE_CONTOUR_SPATIAL_EDGE_RGB_DISTANCE) break;
            ++length;
            if (length > maximum) return false;
        }
    }
    return true;
}

bool CurrentOutlineNeighborMatches(int2 p, float3 color, uint slot) {
    float3 neighbor;
    bool neighbor_valid = LoadDisplayColor(p, neighbor);
    if (!neighbor_valid) return false;
    float contrast = ColorDistance(color, neighbor);
    if (contrast <= V2_SUBTITLE_CONTOUR_SPATIAL_EDGE_RGB_DISTANCE) return false;
    uint3 neighbor_quantized = PaletteQuantize(neighbor);
    return CurrentPaletteMatches(slot, neighbor_quantized);
}

bool CurrentGlyphEvidence(int2 p, float3 color) {
    bool2 edges = CurrentSpatialEdges(p, color);
    if (!any(edges)) return false;
    uint3 q = PaletteQuantize(color);
    uint maximum = MaxChannel(q), minimum = min(q.x, min(q.y, q.z));
    [loop]
    for (uint slot = 0u; slot < CurrentCount; ++slot) {
        if (!SourcePixelInRect(p, CurrentSupportSearch[slot]) || PaletteCount[slot] == 0u) continue;
        if (CurrentPaletteMatches(slot, q)) {
            uint limit = (V2_SUBTITLE_CONTOUR_LONG_LINE_CORE_HEIGHT_NUMERATOR *
                (CurrentCores[slot].w - CurrentCores[slot].y) * contour_source.y) /
                (V2_SUBTITLE_CONTOUR_LONG_LINE_CORE_HEIGHT_DENOMINATOR * (contour_content.w - contour_content.y));
            if ((edges.x && CurrentShortTangent(p, color, int2(0, 1), limit)) ||
                (edges.y && CurrentShortTangent(p, color, int2(1, 0), limit))) return true;
        }
        // The first retained mode has the greatest mean peak by construction. Read the
        // authoritative retained payload directly instead of a redundant shared cache.
        uint4 brightest_mode = PaletteMode[slot * V2_SUBTITLE_CONTOUR_PALETTE_MAX_MODES];
        uint2 peak = uint2(MaxChannel(brightest_mode.xyz), brightest_mode.w);
        if (V2_SUBTITLE_CONTOUR_OUTLINE_VALUE_DENOMINATOR * maximum * peak.y >
                V2_SUBTITLE_CONTOUR_OUTLINE_VALUE_NUMERATOR * peak.x ||
            V2_SUBTITLE_CONTOUR_OUTLINE_CHROMA_DENOMINATOR * (maximum - minimum) >
                V2_SUBTITLE_CONTOUR_OUTLINE_CHROMA_NUMERATOR * V2_SUBTITLE_CONTOUR_PALETTE_CHANNEL_MAX) continue;
        const int radius = (int)V2_SUBTITLE_CONTOUR_OUTLINE_PROBE_RADIUS_SOURCE_PIXELS;
        // Explicit calls preserve the exact eight-point set without nested unrolled
        // continue/early-return lowering in FXC.
        if (CurrentOutlineNeighborMatches(p + int2(-radius, -radius), color, slot)) return true;
        if (CurrentOutlineNeighborMatches(p + int2(0, -radius), color, slot)) return true;
        if (CurrentOutlineNeighborMatches(p + int2(radius, -radius), color, slot)) return true;
        if (CurrentOutlineNeighborMatches(p + int2(-radius, 0), color, slot)) return true;
        if (CurrentOutlineNeighborMatches(p + int2(radius, 0), color, slot)) return true;
        if (CurrentOutlineNeighborMatches(p + int2(-radius, radius), color, slot)) return true;
        if (CurrentOutlineNeighborMatches(p + int2(0, radius), color, slot)) return true;
        if (CurrentOutlineNeighborMatches(p + int2(radius, radius), color, slot)) return true;
    }
    return false;
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
    PrepareCurrentPalettes(group_index, dispatch_id.xy);
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
            float4 feature = 0.0f;
            uint mask = 0u;
            if (valid && LoadDisplayColor(p, color)) {
                bool candidate = false;
                float probability = 0.0f;
                if (CurrentGate != 0u) {
                    // Current-frame OCR8 covers authorize each pixel independently.
                    // Box count, ordering, splits and merges are not RGB-history reset keys.
                    candidate = CoveredSourcePixel(p) &&
                        CurrentProbabilityEvidence(p, probability);
                }
                feature = AdvanceFeature(p, color, candidate);
                // Current learned-color stroke evidence admits and withdraws support here.
                // Probability is palette-anchor evidence, never a filled word/glyph bitmap.
                // Positive count keeps duplicate admission fail-closed;
                // every genuine eligible observation starts at one without an acquisition wait.
                uint count = (uint)feature.w >> V2_SUBTITLE_CONTOUR_HISTORY_FEATURE_COUNT_SHIFT;
                mask = candidate && count != 0u && SupportSearchSourcePixel(p) &&
                    CurrentGlyphEvidence(p, color) ? 1u : 0u;
            }
            NativeSupportWrite[uint2(x, y)] = mask;
            if (x < feature_w && y < feature_h) HistoryFeaturesWrite[uint2(x, y)] = feature;
        }
    }
}

// Current sparse support seeds a bounded color-geodesic body fill. The final small native
// collar preserves the full nonzero-alpha outline/AA footprint, including RGB-rounded AA.
// Neither probability values nor history counts supply new membership or a frame wait.
[numthreads(16, 16, 1)]
void dense_main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID,
                uint group_index : SV_GroupIndex) {
    if (group_index == 0u) GlyphTileGroupStride = max((contour_field.xy + 15u) / 16u, uint2(1u, 1u));
    GroupMemoryBarrierWithGroupSync();
    uint2 group_stride = GlyphTileGroupStride;
    // The group has consumed its immutable loop controls before PrepareCurrentGate's lane-0
    // reads of the same constant buffer. This also makes their uniform lifetime explicit to
    // SM5's barrier/race analysis rather than propagating lane-local gate control into loops.
    GroupMemoryBarrierWithGroupSync();
    PrepareCurrentGate(group_index);
    uint width, height, support_w, support_h;
    DenseGlyphMatteWrite.GetDimensions(width, height);
    CurrentSupportRead.GetDimensions(support_w, support_h);
    bool valid = CurrentGate != 0u && CurrentCount != 0u &&
        width == contour_source.x && height == contour_source.w &&
        support_w == width && support_h == height;
    // Authority is a data gate, never a barrier/loop-control gate. All lanes execute the
    // same texture-dimension tile loops even for current empty/invalid or canonical 1x1
    // resources. Invalid authority loads zero membership and overwrites every output zero.
    [loop]
    for (uint tile_y = group_id.y * GLYPH_TILE_SIDE; tile_y < height;
         tile_y += group_stride.y * GLYPH_TILE_SIDE) {
        [loop]
        for (uint tile_x = group_id.x * GLYPH_TILE_SIDE; tile_x < width;
             tile_x += group_stride.x * GLYPH_TILE_SIDE) {
            [loop]
            for (uint index = group_index; index < GLYPH_TILE_PIXELS; index += 256u) {
                int2 p = int2(tile_x, tile_y) +
                    int2(index % GLYPH_TILE_EXTENT, index / GLYPH_TILE_EXTENT) - (int)GLYPH_TILE_HALO;
                float3 color;
                bool color_valid = LoadDisplayColor(p, color) && valid;
                GlyphTileColors[index] = color_valid ? float4(color, 1.0f) : 0.0f;
                GlyphTileMask[index] = color_valid && CurrentSupportRead.Load(int3(p, 0)) == 1u ? 1u : 0u;
            }
            GroupMemoryBarrierWithGroupSync();
            [loop]
            for (uint step = 0u; step < V2_SUBTITLE_GLYPH_MATTE_STEPS; ++step) {
                [loop]
                for (uint index = group_index; index < GLYPH_TILE_PIXELS; index += 256u) {
                    uint admitted = GlyphTileMask[index];
                    float4 color = GlyphTileColors[index];
                    if (admitted == 0u && color.a == 1.0f) {
                        int2 p = int2(index % GLYPH_TILE_EXTENT, index / GLYPH_TILE_EXTENT);
                        bool bright = max(color.r, max(color.g, color.b)) >= V2_SUBTITLE_GLYPH_MATTE_BRIGHTNESS_SPLIT;
                        [unroll]
                        for (int dy = -1; dy <= 1; ++dy) {
                            [unroll]
                            for (int dx = -1; dx <= 1; ++dx) {
                                int2 q = p + int2(dx, dy);
                                if ((dx == 0 && dy == 0) || any(q < 0) ||
                                    any(q >= (int)GLYPH_TILE_EXTENT)) continue;
                                uint neighbor = (uint)q.y * GLYPH_TILE_EXTENT + (uint)q.x;
                                float4 other = GlyphTileColors[neighbor];
                                bool other_bright = max(other.r, max(other.g, other.b)) >= V2_SUBTITLE_GLYPH_MATTE_BRIGHTNESS_SPLIT;
                                if (GlyphTileMask[neighbor] == 1u && other.a == 1.0f && bright == other_bright &&
                                    ColorDistance(color.rgb, other.rgb) <= V2_SUBTITLE_GLYPH_MATTE_RGB_DISTANCE) admitted = 1u;
                            }
                        }
                    }
                    GlyphTileNext[index] = admitted;
                }
                GroupMemoryBarrierWithGroupSync();
                [loop]
                for (uint index = group_index; index < GLYPH_TILE_PIXELS; index += 256u)
                    GlyphTileMask[index] = GlyphTileNext[index];
                GroupMemoryBarrierWithGroupSync();
            }
            uint2 output = uint2(tile_x, tile_y) + thread_id.xy;
            if (output.x < width && output.y < height) {
                uint center = (thread_id.y + GLYPH_TILE_HALO) * GLYPH_TILE_EXTENT + thread_id.x + GLYPH_TILE_HALO;
                uint admitted = 0u;
                if (GlyphTileColors[center].a == 1.0f) {
                    const int radius = (int)V2_SUBTITLE_GLYPH_MATTE_COLLAR_RADIUS_SOURCE_PIXELS;
                    [unroll]
                    for (int dy = -radius; dy <= radius; ++dy) {
                        [unroll]
                        for (int dx = -radius; dx <= radius; ++dx) {
                            if (dx * dx + dy * dy <= (int)V2_SUBTITLE_GLYPH_MATTE_COLLAR_RADIUS_SQUARED) {
                                uint neighbor = (uint)((int)center + dy * (int)GLYPH_TILE_EXTENT + dx);
                                admitted |= GlyphTileMask[neighbor];
                            }
                        }
                    }
                }
                DenseGlyphMatteWrite[output] = admitted != 0u ? 1u : 0u;
            }
            // Every lane finishes reading this tile before any lane loads the next one.
            GroupMemoryBarrierWithGroupSync();
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
