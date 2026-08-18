// GPU-only near-identical proposal, authenticated branch receipt, and postprocess gating. The
// mandatory GPU-undecided preprocess has already compared its committed calibrated NCHW output
// with independently owned state-1 history. The CPU never maps this evidence or decision.

#include "include/host_sbs_near_identical_compare.hlsl"

StructuredBuffer<NearIdenticalTileEvidence> NearIdenticalTileInput : register(t2);
StructuredBuffer<uint> NearIdenticalHistoryOwner : register(t3);
StructuredBuffer<float4> NearIdenticalCutBridgeState : register(t4);
Texture2D<float> NearIdenticalPreviousDepth : register(t5);
StructuredBuffer<float4> NearIdenticalMinMaxEma : register(t6);
RWStructuredBuffer<uint> NearIdenticalOcrRecord : register(u1);
RWStructuredBuffer<uint> NearIdenticalHistoryOwnerOutput : register(u2);
RWByteAddressBuffer NearIdenticalDecision : register(u3);
RWTexture2D<float> NearIdenticalReuseDepthOutput : register(u4);
RWStructuredBuffer<uint> NearIdenticalSceneCutEvidence : register(u5);

#include "include/depth_constants.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"
#include "include/depth_coordinate_v2_contract.generated.hlsl"
#include "include/depth_valid_history_contract.hlsl"

cbuffer NearIdenticalConstants : register(b1) {
    uint near_identical_request_flags;
    uint near_identical_tile_group_width;
    uint near_identical_tile_group_height;
    uint near_identical_tile_group_count;
    uint near_identical_current_frame_low;
    uint near_identical_current_frame_high;
    uint near_identical_baseline_frame_low;
    uint near_identical_baseline_frame_high;
    uint near_identical_domain_tag_low;
    uint near_identical_domain_tag_high;
    uint near_identical_request_token_low;
    uint near_identical_request_token_high;
    uint near_identical_reduce_groups;
    uint near_identical_stream_frame_delta;
    uint near_identical_expected_work;
    uint near_identical_expected_work_cookie;
    uint near_identical_observation_timestamp_low;
    uint near_identical_observation_timestamp_high;
    uint near_identical_timestamp_padding0;
    uint near_identical_timestamp_padding1;
};

// Bound only for the detector finalizer's exact-current OCR8 abstention publication. Keeping this
// publication in the independently authenticated detector closure avoids changing the canonical
// OCR producer closure.
cbuffer NearIdenticalOcrConstants : register(b2) {
    uint near_identical_ocr_frame_low;
    uint near_identical_ocr_frame_high;
    uint near_identical_ocr_analysis_generation_low;
    uint near_identical_ocr_analysis_generation_high;
    uint near_identical_ocr_source_width;
    uint near_identical_ocr_source_height;
    uint near_identical_ocr_field_width;
    uint near_identical_ocr_field_height;
    uint near_identical_ocr_crop_top;
    uint near_identical_ocr_crop_height;
    uint near_identical_ocr_roi_top;
    uint near_identical_ocr_roi_bottom;
    uint4 near_identical_ocr_tensor_content;
};

#define NEAR_IDENTICAL_REQUEST_AUTHORIZED (1u << 0u)
#define NEAR_IDENTICAL_REQUEST_FORCE_INFER (1u << 1u)
#define NEAR_IDENTICAL_RESOLVE_THREADS 64u
#define NEAR_IDENTICAL_REDUCE_ELEMENTS_PER_GROUP 256u
#define NEAR_IDENTICAL_RESOLVE_FLAG_MALFORMED (1u << 0u)
#define NEAR_IDENTICAL_RESOLVE_FLAG_LOCAL_VETO (1u << 1u)
#define NEAR_IDENTICAL_MAX_TILE_GROUP_COUNT 2048u
#define NEAR_IDENTICAL_MAX_INFER_OWNER_AGE 4u
#define NEAR_IDENTICAL_MAX_INFER_OWNER_OBSERVATION_AGE_US 100000u

#define NEAR_IDENTICAL_HISTORY_OWNER_TAG 0x3142484Eu
#define NEAR_IDENTICAL_HISTORY_OWNER_SCHEMA 2u

#define NEAR_IDENTICAL_DECISION_REUSE 0u
#define NEAR_IDENTICAL_DECISION_INFER 1u
#define NEAR_IDENTICAL_DECISION_COOKIE 0xD1EC15A5u
#define NEAR_IDENTICAL_TOKEN_LOW_COOKIE 0xA3756C91u
#define NEAR_IDENTICAL_TOKEN_HIGH_COOKIE 0x5C8A936Eu
#define NEAR_IDENTICAL_PROPOSAL_MAGIC 0x504F5250u
#define NEAR_IDENTICAL_RECEIPT_MAGIC 0x47524243u
#define NEAR_IDENTICAL_REQUEST_MAGIC 0x54535152u
#define NEAR_IDENTICAL_OPTIONAL_RECEIPT_MAGIC 0x52434F4Fu
#define NEAR_IDENTICAL_WORK_FLAGS_COOKIE 0x6F435257u
#define NEAR_IDENTICAL_WORK_OPTIONAL_OCR (1u << 0u)
#define NEAR_IDENTICAL_WORK_SUBTITLE_OBSERVATION (1u << 1u)
#define NEAR_IDENTICAL_WORK_OPTIONAL_OCR_DUE (1u << 3u)
#define NEAR_IDENTICAL_WORK_SUBTITLE_OBSERVATION_DUE (1u << 4u)

#define NEAR_IDENTICAL_DECISION_OFFSET 0u
#define NEAR_IDENTICAL_DECISION_COOKIE_OFFSET 4u
#define NEAR_IDENTICAL_DECISION_TOKEN_LOW_OFFSET 8u
#define NEAR_IDENTICAL_DECISION_TOKEN_HIGH_OFFSET 12u
#define NEAR_IDENTICAL_DECISION_TOKEN_LOW_COOKIE_OFFSET 16u
#define NEAR_IDENTICAL_DECISION_TOKEN_HIGH_COOKIE_OFFSET 20u
#define NEAR_IDENTICAL_DECISION_MAGIC_OFFSET 24u
#define NEAR_IDENTICAL_DECISION_RESERVED_OFFSET 28u
#define NEAR_IDENTICAL_REQUEST_TOKEN_LOW_OFFSET 32u
#define NEAR_IDENTICAL_REQUEST_TOKEN_HIGH_OFFSET 36u
#define NEAR_IDENTICAL_REQUEST_TOKEN_LOW_COOKIE_OFFSET 40u
#define NEAR_IDENTICAL_REQUEST_TOKEN_HIGH_COOKIE_OFFSET 44u
#define NEAR_IDENTICAL_REQUEST_MAGIC_OFFSET 48u
#define NEAR_IDENTICAL_REQUEST_WORK_FLAGS_OFFSET 52u
#define NEAR_IDENTICAL_REQUEST_WORK_FLAGS_COOKIE_OFFSET 56u
#define NEAR_IDENTICAL_REQUEST_RESERVED_OFFSET 60u
#define NEAR_IDENTICAL_INFER_REDUCE_OFFSET 64u
#define NEAR_IDENTICAL_INFER_ONE_OFFSET 80u
#define NEAR_IDENTICAL_INFER_GRID16_OFFSET 96u
#define NEAR_IDENTICAL_INFER_GRID8_OFFSET 112u
#define NEAR_IDENTICAL_INFER_COLUMNS_OFFSET 128u
#define NEAR_IDENTICAL_INFER_ROWS_OFFSET 144u
#define NEAR_IDENTICAL_REUSE_GRID16_OFFSET 160u
#define NEAR_IDENTICAL_OPTIONAL_PREPROCESS_OFFSET 176u
#define NEAR_IDENTICAL_SUBTITLE_CONDITION_GRID16_OFFSET 176u
#define NEAR_IDENTICAL_OPTIONAL_CELLS_OFFSET 192u
#define NEAR_IDENTICAL_OPTIONAL_ONE_OFFSET 208u
#define NEAR_IDENTICAL_SUBTITLE_RECORD_ONE_OFFSET 224u
#define NEAR_IDENTICAL_OBSERVATION_ONE_OFFSET 240u

#define NEAR_IDENTICAL_HISTORY_WORD_CONTRACT_TAG 0u
#define NEAR_IDENTICAL_HISTORY_WORD_CONTRACT_SCHEMA 1u
#define NEAR_IDENTICAL_HISTORY_WORD_FRAME_LOW 2u
#define NEAR_IDENTICAL_HISTORY_WORD_FRAME_HIGH 3u
#define NEAR_IDENTICAL_HISTORY_WORD_DOMAIN_TAG_LOW 4u
#define NEAR_IDENTICAL_HISTORY_WORD_DOMAIN_TAG_HIGH 5u
#define NEAR_IDENTICAL_HISTORY_WORD_TARGET_WIDTH 6u
#define NEAR_IDENTICAL_HISTORY_WORD_TARGET_HEIGHT 7u
#define NEAR_IDENTICAL_HISTORY_WORD_OBSERVATION_TIMESTAMP_LOW 8u
#define NEAR_IDENTICAL_HISTORY_WORD_OBSERVATION_TIMESTAMP_HIGH 9u

groupshared uint4 NearIdenticalResolvePrimary[NEAR_IDENTICAL_RESOLVE_THREADS];
groupshared uint NearIdenticalResolveFlags[NEAR_IDENTICAL_RESOLVE_THREADS];

bool NearIdenticalOwnerIsNewer() {
    return near_identical_current_frame_high > near_identical_baseline_frame_high ||
        (near_identical_current_frame_high == near_identical_baseline_frame_high &&
         near_identical_current_frame_low > near_identical_baseline_frame_low);
}

bool NearIdenticalHistoryOwnerMatchesRequest() {
    uint owner_frame_low =
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_FRAME_LOW];
    uint owner_frame_high =
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_FRAME_HIGH];
    bool owner_at_or_before_host_baseline =
        owner_frame_high < near_identical_baseline_frame_high ||
        (owner_frame_high == near_identical_baseline_frame_high &&
         owner_frame_low <= near_identical_baseline_frame_low);
    // PreviousModelInput and this owner advance only on an authenticated infer. The host baseline
    // names the immediately preceding opaque root, so allow the GPU-owned infer owner to be older
    // while the cumulative current-vs-owner comparison is still bounded to four frame steps and
    // strictly less than 100 ms of source observation time.
    uint low_delta = near_identical_current_frame_low - owner_frame_low;
    uint borrow = near_identical_current_frame_low < owner_frame_low ? 1u : 0u;
    uint high_delta = near_identical_current_frame_high - owner_frame_high - borrow;
    bool owner_age_valid =
        (owner_frame_low != 0u || owner_frame_high != 0u) &&
        owner_at_or_before_host_baseline && high_delta == 0u && low_delta > 0u &&
        low_delta <= NEAR_IDENTICAL_MAX_INFER_OWNER_AGE;
    uint owner_timestamp_low = NearIdenticalHistoryOwner[
        NEAR_IDENTICAL_HISTORY_WORD_OBSERVATION_TIMESTAMP_LOW];
    uint owner_timestamp_high = NearIdenticalHistoryOwner[
        NEAR_IDENTICAL_HISTORY_WORD_OBSERVATION_TIMESTAMP_HIGH];
    bool observation_not_regressed =
        near_identical_observation_timestamp_high > owner_timestamp_high ||
        (near_identical_observation_timestamp_high == owner_timestamp_high &&
         near_identical_observation_timestamp_low >= owner_timestamp_low);
    uint observation_borrow =
        near_identical_observation_timestamp_low < owner_timestamp_low ? 1u : 0u;
    uint observation_delta_low =
        near_identical_observation_timestamp_low - owner_timestamp_low;
    uint observation_delta_high =
        near_identical_observation_timestamp_high - owner_timestamp_high - observation_borrow;
    bool observation_age_valid =
        (owner_timestamp_low != 0u || owner_timestamp_high != 0u) &&
        (near_identical_observation_timestamp_low != 0u ||
         near_identical_observation_timestamp_high != 0u) &&
        observation_not_regressed && observation_delta_high == 0u &&
        observation_delta_low < NEAR_IDENTICAL_MAX_INFER_OWNER_OBSERVATION_AGE_US;
    return
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_CONTRACT_TAG] ==
            NEAR_IDENTICAL_HISTORY_OWNER_TAG &&
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_CONTRACT_SCHEMA] ==
            NEAR_IDENTICAL_HISTORY_OWNER_SCHEMA &&
        owner_age_valid && observation_age_valid &&
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_DOMAIN_TAG_LOW] ==
            near_identical_domain_tag_low &&
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_DOMAIN_TAG_HIGH] ==
            near_identical_domain_tag_high &&
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_TARGET_WIDTH] == target_w &&
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_TARGET_HEIGHT] == target_h;
}

uint NearIdenticalExpectedReduceGroups() {
    uint target_elements = target_w * target_h;
    uint groups = target_elements / NEAR_IDENTICAL_REDUCE_ELEMENTS_PER_GROUP;
    groups += target_elements % NEAR_IDENTICAL_REDUCE_ELEMENTS_PER_GROUP != 0u ? 1u : 0u;
    return min(NEAR_IDENTICAL_RESOLVE_THREADS, max(1u, groups));
}

void NearIdenticalWriteDispatchArgs(uint byte_offset, uint x, uint y, uint z) {
    NearIdenticalDecision.Store4(byte_offset, uint4(x, y, z, 0u));
}

bool NearIdenticalWorkValid(uint flags) {
    return flags == 0u ||
        flags == NEAR_IDENTICAL_WORK_OPTIONAL_OCR ||
        flags == NEAR_IDENTICAL_WORK_SUBTITLE_OBSERVATION ||
        flags == NEAR_IDENTICAL_WORK_OPTIONAL_OCR_DUE ||
        flags == NEAR_IDENTICAL_WORK_SUBTITLE_OBSERVATION_DUE;
}

bool NearIdenticalSubtitleConstantsValid() {
    return
        (near_identical_request_flags == NEAR_IDENTICAL_REQUEST_AUTHORIZED ||
         near_identical_request_flags == NEAR_IDENTICAL_REQUEST_FORCE_INFER) &&
        (near_identical_request_token_low != 0u ||
         near_identical_request_token_high != 0u) &&
        (near_identical_current_frame_low != 0u ||
         near_identical_current_frame_high != 0u) &&
        NearIdenticalWorkValid(near_identical_expected_work) &&
        near_identical_expected_work_cookie ==
            (near_identical_expected_work == 0u ? 0u :
             near_identical_expected_work ^ NEAR_IDENTICAL_WORK_FLAGS_COOKIE);
}

// Authenticate the complete subtitle request/receipt pair without CPU readback. Callers use the
// returned work only after this succeeds. A CPU-known force transaction may still publish an
// exact-current abstention when this proof is malformed.
bool NearIdenticalSubtitleReceiptValid(
    out uint decision,
    out uint request_work,
    out bool optional_ocr) {
    decision = NearIdenticalDecision.Load(NEAR_IDENTICAL_DECISION_OFFSET);
    request_work = NearIdenticalDecision.Load(
        NEAR_IDENTICAL_REQUEST_WORK_FLAGS_OFFSET);
    optional_ocr = false;
    uint decision_token_low = NearIdenticalDecision.Load(
        NEAR_IDENTICAL_DECISION_TOKEN_LOW_OFFSET);
    uint decision_token_high = NearIdenticalDecision.Load(
        NEAR_IDENTICAL_DECISION_TOKEN_HIGH_OFFSET);
    uint request_token_low = NearIdenticalDecision.Load(
        NEAR_IDENTICAL_REQUEST_TOKEN_LOW_OFFSET);
    uint request_token_high = NearIdenticalDecision.Load(
        NEAR_IDENTICAL_REQUEST_TOKEN_HIGH_OFFSET);
    uint optional_marker = NearIdenticalDecision.Load(
        NEAR_IDENTICAL_DECISION_RESERVED_OFFSET);
    bool request_valid = NearIdenticalSubtitleConstantsValid() &&
        request_token_low == near_identical_request_token_low &&
        request_token_high == near_identical_request_token_high &&
        request_work == near_identical_expected_work &&
        NearIdenticalWorkValid(request_work) &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_MAGIC_OFFSET) ==
            NEAR_IDENTICAL_REQUEST_MAGIC &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_TOKEN_LOW_COOKIE_OFFSET) ==
            (request_token_low ^ NEAR_IDENTICAL_TOKEN_LOW_COOKIE) &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_TOKEN_HIGH_COOKIE_OFFSET) ==
            (request_token_high ^ NEAR_IDENTICAL_TOKEN_HIGH_COOKIE) &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_WORK_FLAGS_COOKIE_OFFSET) ==
            (request_work == 0u ?
                0u : request_work ^ NEAR_IDENTICAL_WORK_FLAGS_COOKIE) &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_RESERVED_OFFSET) == 0u;
    bool receipt_valid = request_valid &&
        decision_token_low == request_token_low &&
        decision_token_high == request_token_high &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_DECISION_MAGIC_OFFSET) ==
            NEAR_IDENTICAL_RECEIPT_MAGIC &&
        (decision == NEAR_IDENTICAL_DECISION_REUSE ||
         decision == NEAR_IDENTICAL_DECISION_INFER) &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_DECISION_COOKIE_OFFSET) ==
            (decision ^ NEAR_IDENTICAL_DECISION_COOKIE ^ optional_marker) &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_DECISION_TOKEN_LOW_COOKIE_OFFSET) ==
            (decision_token_low ^ NEAR_IDENTICAL_TOKEN_LOW_COOKIE) &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_DECISION_TOKEN_HIGH_COOKIE_OFFSET) ==
            (decision_token_high ^ NEAR_IDENTICAL_TOKEN_HIGH_COOKIE) &&
        (optional_marker == 0u ||
         (optional_marker == NEAR_IDENTICAL_OPTIONAL_RECEIPT_MAGIC &&
          ((request_work == NEAR_IDENTICAL_WORK_OPTIONAL_OCR &&
            decision == NEAR_IDENTICAL_DECISION_INFER) ||
           request_work == NEAR_IDENTICAL_WORK_OPTIONAL_OCR_DUE)));
    optional_ocr = receipt_valid &&
        optional_marker == NEAR_IDENTICAL_OPTIONAL_RECEIPT_MAGIC;
    return receipt_valid;
}

// Depth authorization is deliberately stronger than subtitle authorization. The depth branch
// consumes the owner/tile/reduction authority that made this transaction GPU-undecided, while the
// independently clocked subtitle branch authenticates only its exact request/receipt disposition.
bool NearIdenticalDepthReceiptValid(out uint decision) {
    decision = NearIdenticalDecision.Load(NEAR_IDENTICAL_DECISION_OFFSET);
    uint decision_token_low = NearIdenticalDecision.Load(
        NEAR_IDENTICAL_DECISION_TOKEN_LOW_OFFSET);
    uint decision_token_high = NearIdenticalDecision.Load(
        NEAR_IDENTICAL_DECISION_TOKEN_HIGH_OFFSET);
    uint request_token_low = NearIdenticalDecision.Load(
        NEAR_IDENTICAL_REQUEST_TOKEN_LOW_OFFSET);
    uint request_token_high = NearIdenticalDecision.Load(
        NEAR_IDENTICAL_REQUEST_TOKEN_HIGH_OFFSET);
    bool constants_valid =
        near_identical_request_flags == NEAR_IDENTICAL_REQUEST_AUTHORIZED &&
        near_identical_tile_group_width == (target_w + 15u) / 16u &&
        near_identical_tile_group_height == (target_h + 15u) / 16u &&
        near_identical_tile_group_count ==
            near_identical_tile_group_width * near_identical_tile_group_height &&
        near_identical_reduce_groups == NearIdenticalExpectedReduceGroups() &&
        near_identical_stream_frame_delta > 0u &&
        near_identical_stream_frame_delta <= 65535u &&
        NearIdenticalOwnerIsNewer() &&
        NearIdenticalWorkValid(near_identical_expected_work) &&
        near_identical_expected_work_cookie ==
            (near_identical_expected_work == 0u ? 0u :
             near_identical_expected_work ^ NEAR_IDENTICAL_WORK_FLAGS_COOKIE) &&
        request_token_low == near_identical_request_token_low &&
        request_token_high == near_identical_request_token_high;
    uint request_work_flags = NearIdenticalDecision.Load(
        NEAR_IDENTICAL_REQUEST_WORK_FLAGS_OFFSET);
    bool request_work_valid = NearIdenticalWorkValid(request_work_flags) &&
        request_work_flags == near_identical_expected_work &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_WORK_FLAGS_COOKIE_OFFSET) ==
            (request_work_flags == 0u ?
                0u : request_work_flags ^ NEAR_IDENTICAL_WORK_FLAGS_COOKIE) &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_RESERVED_OFFSET) == 0u;
    uint receipt_optional = NearIdenticalDecision.Load(
        NEAR_IDENTICAL_DECISION_RESERVED_OFFSET);
    return constants_valid &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_DECISION_MAGIC_OFFSET) ==
            NEAR_IDENTICAL_RECEIPT_MAGIC &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_MAGIC_OFFSET) ==
            NEAR_IDENTICAL_REQUEST_MAGIC &&
        (decision == NEAR_IDENTICAL_DECISION_REUSE ||
         decision == NEAR_IDENTICAL_DECISION_INFER) &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_DECISION_COOKIE_OFFSET) ==
            (decision ^ NEAR_IDENTICAL_DECISION_COOKIE ^ receipt_optional) &&
        decision_token_low == request_token_low &&
        decision_token_high == request_token_high &&
        (request_token_low != 0u || request_token_high != 0u) &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_DECISION_TOKEN_LOW_COOKIE_OFFSET) ==
            (decision_token_low ^ NEAR_IDENTICAL_TOKEN_LOW_COOKIE) &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_DECISION_TOKEN_HIGH_COOKIE_OFFSET) ==
            (decision_token_high ^ NEAR_IDENTICAL_TOKEN_HIGH_COOKIE) &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_TOKEN_LOW_COOKIE_OFFSET) ==
            (request_token_low ^ NEAR_IDENTICAL_TOKEN_LOW_COOKIE) &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_TOKEN_HIGH_COOKIE_OFFSET) ==
            (request_token_high ^ NEAR_IDENTICAL_TOKEN_HIGH_COOKIE) &&
        request_work_valid &&
        (receipt_optional == 0u ||
         (receipt_optional == NEAR_IDENTICAL_OPTIONAL_RECEIPT_MAGIC &&
          ((request_work_flags == NEAR_IDENTICAL_WORK_OPTIONAL_OCR &&
            decision == NEAR_IDENTICAL_DECISION_INFER) ||
           request_work_flags == NEAR_IDENTICAL_WORK_OPTIONAL_OCR_DUE)));
}

void NearIdenticalWriteOcrAbstention() {
    [loop]
    for (uint word = 0u; word < V2_OCR_RECORD_WORD_COUNT; ++word) {
        NearIdenticalOcrRecord[word] = 0u;
    }
    NearIdenticalOcrRecord[0u] = V2_OCR_RECORD_SCHEMA;
    NearIdenticalOcrRecord[1u] = V2_OCR_RECORD_TAG;
    NearIdenticalOcrRecord[5u] = near_identical_ocr_frame_low;
    NearIdenticalOcrRecord[6u] = near_identical_ocr_frame_high;
    NearIdenticalOcrRecord[7u] = near_identical_ocr_analysis_generation_low;
    NearIdenticalOcrRecord[8u] = near_identical_ocr_analysis_generation_high;
    NearIdenticalOcrRecord[9u] = near_identical_ocr_source_width;
    NearIdenticalOcrRecord[10u] = near_identical_ocr_source_height;
    NearIdenticalOcrRecord[11u] = near_identical_ocr_field_width;
    NearIdenticalOcrRecord[12u] = near_identical_ocr_field_height;
    NearIdenticalOcrRecord[13u] = near_identical_ocr_roi_top;
    NearIdenticalOcrRecord[14u] = near_identical_ocr_roi_bottom;
}

[numthreads(1, 1, 1)]
void history_owner_main(uint3 dispatch_thread : SV_DispatchThreadID) {
    NearIdenticalHistoryOwnerOutput[NEAR_IDENTICAL_HISTORY_WORD_CONTRACT_TAG] = 0u;
    float history_state = SBS_STATE_MODEL_INPUT_HISTORY_STATE(
        NearIdenticalCutBridgeState[SBS_STATE_VECTOR_MODEL_INPUT_HISTORY_STATE]);
    bool state_valid =
        asuint(SBS_STATE_CUT_CONTRACT_TAG_BITS(NearIdenticalCutBridgeState[0])) ==
            SBS_CUT_CONTRACT_TAG &&
        NearIdenticalFinite(history_state) &&
        DepthValidHistoryAdvances(NearIdenticalMinMaxEma[0].w, history_state);
    bool owner_valid =
        state_valid &&
        (near_identical_current_frame_low != 0u ||
         near_identical_current_frame_high != 0u) &&
        (near_identical_domain_tag_low != 0u ||
         near_identical_domain_tag_high != 0u) &&
        (near_identical_observation_timestamp_low != 0u ||
         near_identical_observation_timestamp_high != 0u) &&
        DepthAnalysisContentValid() &&
        near_identical_stream_frame_delta == 0u &&
        NearIdenticalWorkValid(near_identical_expected_work) &&
        near_identical_expected_work_cookie ==
            (near_identical_expected_work == 0u ? 0u :
             near_identical_expected_work ^ NEAR_IDENTICAL_WORK_FLAGS_COOKIE);
    NearIdenticalHistoryOwnerOutput[NEAR_IDENTICAL_HISTORY_WORD_CONTRACT_SCHEMA] =
        NEAR_IDENTICAL_HISTORY_OWNER_SCHEMA;
    NearIdenticalHistoryOwnerOutput[NEAR_IDENTICAL_HISTORY_WORD_FRAME_LOW] =
        near_identical_current_frame_low;
    NearIdenticalHistoryOwnerOutput[NEAR_IDENTICAL_HISTORY_WORD_FRAME_HIGH] =
        near_identical_current_frame_high;
    NearIdenticalHistoryOwnerOutput[NEAR_IDENTICAL_HISTORY_WORD_DOMAIN_TAG_LOW] =
        near_identical_domain_tag_low;
    NearIdenticalHistoryOwnerOutput[NEAR_IDENTICAL_HISTORY_WORD_DOMAIN_TAG_HIGH] =
        near_identical_domain_tag_high;
    NearIdenticalHistoryOwnerOutput[NEAR_IDENTICAL_HISTORY_WORD_TARGET_WIDTH] = target_w;
    NearIdenticalHistoryOwnerOutput[NEAR_IDENTICAL_HISTORY_WORD_TARGET_HEIGHT] = target_h;
    NearIdenticalHistoryOwnerOutput[
        NEAR_IDENTICAL_HISTORY_WORD_OBSERVATION_TIMESTAMP_LOW] =
            near_identical_observation_timestamp_low;
    NearIdenticalHistoryOwnerOutput[
        NEAR_IDENTICAL_HISTORY_WORD_OBSERVATION_TIMESTAMP_HIGH] =
            near_identical_observation_timestamp_high;
    DeviceMemoryBarrier();
    if (owner_valid) {
        NearIdenticalHistoryOwnerOutput[NEAR_IDENTICAL_HISTORY_WORD_CONTRACT_TAG] =
            NEAR_IDENTICAL_HISTORY_OWNER_TAG;
    }
}

// Seed CutBridge's frame age from the last receipt-authorized infer postprocess, not from the last
// branch the CPU can name. history_owner_main writes words 1..9 for an accepted infer observation
// and uses word 0 only to authenticate the narrower model-history baseline. Ignoring word 0 here
// therefore yields B-A for opaque infer B, while opaque reuse leaves A in place and yields C-A.
[numthreads(1, 1, 1)]
void scene_seed_main(uint3 dispatch_thread : SV_DispatchThreadID) {
    uint owner_frame_low =
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_FRAME_LOW];
    uint owner_frame_high =
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_FRAME_HIGH];
    bool owner_older =
        near_identical_current_frame_high > owner_frame_high ||
        (near_identical_current_frame_high == owner_frame_high &&
         near_identical_current_frame_low > owner_frame_low);
    bool owner_valid =
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_CONTRACT_SCHEMA] ==
            NEAR_IDENTICAL_HISTORY_OWNER_SCHEMA &&
        (owner_frame_low != 0u || owner_frame_high != 0u) &&
        owner_older &&
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_DOMAIN_TAG_LOW] ==
            near_identical_domain_tag_low &&
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_DOMAIN_TAG_HIGH] ==
            near_identical_domain_tag_high &&
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_TARGET_WIDTH] == target_w &&
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_TARGET_HEIGHT] == target_h &&
        (near_identical_current_frame_low != 0u ||
         near_identical_current_frame_high != 0u) &&
        (near_identical_domain_tag_low != 0u || near_identical_domain_tag_high != 0u) &&
        DepthAnalysisContentValid() &&
        near_identical_stream_frame_delta == 0u &&
        NearIdenticalWorkValid(near_identical_expected_work) &&
        near_identical_expected_work_cookie ==
            (near_identical_expected_work == 0u ? 0u :
             near_identical_expected_work ^ NEAR_IDENTICAL_WORK_FLAGS_COOKIE);

    uint frame_delta = 1u;
    if (owner_valid) {
        uint borrow = near_identical_current_frame_low < owner_frame_low ? 1u : 0u;
        uint delta_low = near_identical_current_frame_low - owner_frame_low;
        uint delta_high = near_identical_current_frame_high - owner_frame_high - borrow;
        frame_delta = delta_high != 0u ? 65535u : min(delta_low, 65535u);
    }
    [unroll]
    for (uint word = 0u; word < 9u; ++word) {
        NearIdenticalSceneCutEvidence[word] = 0u;
    }
    NearIdenticalSceneCutEvidence[9u] = frame_delta;
    DeviceMemoryBarrier();
}

// The CPU has already published the complete current RQST and cleared this proposal/args record
// in one ordered UpdateSubresource. CUDA must replace the tag-last PROP below with a CBRG receipt
// after the actual branch.
[numthreads(NEAR_IDENTICAL_RESOLVE_THREADS, 1, 1)]
void resolve_main(uint3 group_thread : SV_GroupThreadID) {
    uint linear_thread = group_thread.x;
    uint4 primary = 0u;
    uint expected_tile_width = (target_w + 15u) / 16u;
    uint expected_tile_height = (target_h + 15u) / 16u;
    bool expected_tile_count_valid = expected_tile_width != 0u &&
        expected_tile_height <=
            NEAR_IDENTICAL_MAX_TILE_GROUP_COUNT / max(expected_tile_width, 1u);
    uint expected_tile_count = expected_tile_count_valid ?
        expected_tile_width * expected_tile_height : 0u;
    uint resolve_flags = !expected_tile_count_valid ||
        near_identical_tile_group_count != expected_tile_count ?
            NEAR_IDENTICAL_RESOLVE_FLAG_MALFORMED : 0u;
    uint tile_limit = min(
        near_identical_tile_group_count,
        expected_tile_count);
    for (uint tile_index = linear_thread;
         tile_index < tile_limit;
         tile_index += NEAR_IDENTICAL_RESOLVE_THREADS) {
        NearIdenticalTileEvidence tile = NearIdenticalTileInput[tile_index];
        primary += uint4(
            tile.admitted,
            tile.medium_changed,
            tile.strong_changed,
            tile.nonfinite);
        bool tile_well_formed =
            tile.admitted <= NEAR_IDENTICAL_TILE_THREADS &&
            tile.medium_changed <= tile.admitted &&
            tile.strong_changed <= tile.admitted &&
            tile.nonfinite <= tile.admitted;
        if (!tile_well_formed) {
            resolve_flags |= NEAR_IDENTICAL_RESOLVE_FLAG_MALFORMED;
        }
        // Unsupported edge slivers own no local veto. For every supported tile, the policy only
        // needs to know whether its strong-change ratio exceeds the inclusive 75% limit; finding
        // the maximum ratio adds no decision information.
        if (tile_well_formed && tile.admitted >= 64u &&
            tile.strong_changed * 4u > tile.admitted * 3u) {
            resolve_flags |= NEAR_IDENTICAL_RESOLVE_FLAG_LOCAL_VETO;
        }
    }
    NearIdenticalResolvePrimary[linear_thread] = primary;
    NearIdenticalResolveFlags[linear_thread] = resolve_flags;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = NEAR_IDENTICAL_RESOLVE_THREADS / 2u;
         stride > 0u;
         stride >>= 1u) {
        if (linear_thread < stride) {
            NearIdenticalResolvePrimary[linear_thread] +=
                NearIdenticalResolvePrimary[linear_thread + stride];
            NearIdenticalResolveFlags[linear_thread] |=
                NearIdenticalResolveFlags[linear_thread + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (linear_thread == 0u) {
        NearIdenticalDecision.Store(NEAR_IDENTICAL_DECISION_MAGIC_OFFSET, 0u);
        uint expected_admitted = DepthAnalysisContentValid() ?
            (analysis_content_right - analysis_content_left) *
                (analysis_content_bottom - analysis_content_top) :
            0u;
        bool constants_valid =
            near_identical_request_flags == NEAR_IDENTICAL_REQUEST_AUTHORIZED &&
            (near_identical_request_token_low != 0u ||
             near_identical_request_token_high != 0u) &&
            near_identical_reduce_groups == NearIdenticalExpectedReduceGroups() &&
            near_identical_stream_frame_delta == 0u &&
            NearIdenticalWorkValid(near_identical_expected_work) &&
            near_identical_expected_work_cookie ==
                (near_identical_expected_work == 0u ? 0u :
                 near_identical_expected_work ^ NEAR_IDENTICAL_WORK_FLAGS_COOKIE);
        bool request_record_valid =
            NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_MAGIC_OFFSET) ==
                NEAR_IDENTICAL_REQUEST_MAGIC &&
            NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_TOKEN_LOW_OFFSET) ==
                near_identical_request_token_low &&
            NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_TOKEN_HIGH_OFFSET) ==
                near_identical_request_token_high &&
            NearIdenticalDecision.Load(
                NEAR_IDENTICAL_REQUEST_TOKEN_LOW_COOKIE_OFFSET) ==
                (near_identical_request_token_low ^ NEAR_IDENTICAL_TOKEN_LOW_COOKIE) &&
            NearIdenticalDecision.Load(
                NEAR_IDENTICAL_REQUEST_TOKEN_HIGH_COOKIE_OFFSET) ==
                (near_identical_request_token_high ^ NEAR_IDENTICAL_TOKEN_HIGH_COOKIE) &&
            NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_WORK_FLAGS_OFFSET) ==
                near_identical_expected_work &&
            NearIdenticalWorkValid(
                NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_WORK_FLAGS_OFFSET)) &&
            NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_WORK_FLAGS_COOKIE_OFFSET) ==
                (NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_WORK_FLAGS_OFFSET) == 0u ?
                    0u :
                    (NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_WORK_FLAGS_OFFSET) ^
                     NEAR_IDENTICAL_WORK_FLAGS_COOKIE)) &&
            NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_RESERVED_OFFSET) == 0u;
        bool tile_shape_valid =
            near_identical_tile_group_width == (target_w + 15u) / 16u &&
            near_identical_tile_group_height == (target_h + 15u) / 16u &&
            near_identical_tile_group_count ==
                near_identical_tile_group_width * near_identical_tile_group_height &&
            near_identical_tile_group_count <= NEAR_IDENTICAL_MAX_TILE_GROUP_COUNT;
        bool request_authorized = constants_valid && request_record_valid;
        bool owner_valid =
            request_authorized && NearIdenticalOwnerIsNewer() &&
            (near_identical_domain_tag_low != 0u ||
             near_identical_domain_tag_high != 0u) &&
            NearIdenticalHistoryOwnerMatchesRequest();
        bool evidence_complete =
            owner_valid && tile_shape_valid && expected_admitted > 0u &&
            NearIdenticalResolvePrimary[0].x == expected_admitted;
        bool counters_well_formed =
            (NearIdenticalResolveFlags[0] &
             NEAR_IDENTICAL_RESOLVE_FLAG_MALFORMED) == 0u &&
            NearIdenticalResolvePrimary[0].y <= NearIdenticalResolvePrimary[0].x &&
            NearIdenticalResolvePrimary[0].z <= NearIdenticalResolvePrimary[0].x &&
            NearIdenticalResolvePrimary[0].w <= NearIdenticalResolvePrimary[0].x;
        bool all_finite = evidence_complete && counters_well_formed &&
            NearIdenticalResolvePrimary[0].w == 0u;
        bool global_quiet =
            NearIdenticalResolvePrimary[0].z * 40u <=
                NearIdenticalResolvePrimary[0].x &&
            NearIdenticalResolvePrimary[0].y * 10u <=
                NearIdenticalResolvePrimary[0].x;
        bool supported_tiles_quiet =
            (NearIdenticalResolveFlags[0] &
             NEAR_IDENTICAL_RESOLVE_FLAG_LOCAL_VETO) == 0u;
        uint proposal = all_finite && global_quiet && supported_tiles_quiet ?
            NEAR_IDENTICAL_DECISION_REUSE : NEAR_IDENTICAL_DECISION_INFER;
        uint requested_work = NearIdenticalDecision.Load(
            NEAR_IDENTICAL_REQUEST_WORK_FLAGS_OFFSET);
        // Ordinary OCR remains infer-coupled. A cadence-due request is the bounded independent
        // clock and preprocesses on either depth branch.
        bool optional_preprocess = request_authorized &&
            ((proposal == NEAR_IDENTICAL_DECISION_INFER &&
              requested_work == NEAR_IDENTICAL_WORK_OPTIONAL_OCR) ||
             requested_work == NEAR_IDENTICAL_WORK_OPTIONAL_OCR_DUE);
        NearIdenticalWriteDispatchArgs(
            NEAR_IDENTICAL_OPTIONAL_PREPROCESS_OFFSET,
            optional_preprocess ? (V2_OCR_INPUT_WIDTH + 15u) / 16u : 0u,
            optional_preprocess ? (V2_OCR_INPUT_HEIGHT + 15u) / 16u : 1u,
            1u);
        // Publish the 32-byte proposal as two naturally aligned vectors, with the tag lane zero.
        // This is also the exact pair of 128-bit records consumed by the CUDA bridge.
        NearIdenticalDecision.Store4(
            NEAR_IDENTICAL_DECISION_OFFSET,
            uint4(
                proposal,
                proposal ^ NEAR_IDENTICAL_DECISION_COOKIE,
                near_identical_request_token_low,
                near_identical_request_token_high));
        NearIdenticalDecision.Store4(
            NEAR_IDENTICAL_DECISION_TOKEN_LOW_COOKIE_OFFSET,
            uint4(
                near_identical_request_token_low ^ NEAR_IDENTICAL_TOKEN_LOW_COOKIE,
                near_identical_request_token_high ^ NEAR_IDENTICAL_TOKEN_HIGH_COOKIE,
                0u,
                0u));
        DeviceMemoryBarrier();
        // CUDA cannot revalidate the detector constants that authorized optional preprocessing.
        // Publish PROP only when that same authority was valid; otherwise the missing tag makes
        // the setter fail depth open to infer while keeping the optional OCR handle off.
        if (request_authorized) {
            NearIdenticalDecision.Store(
                NEAR_IDENTICAL_DECISION_MAGIC_OFFSET,
                NEAR_IDENTICAL_PROPOSAL_MAGIC);
        }
    }
}

// The joined CUDA root has completed and every interop resource is unmapped. One direct finalizer
// independently validates depth and subtitle authority, publishes every postprocess shape, and
// writes an exact-current abstention whenever no authenticated optional OCR receipt may be
// consumed. Force-infer depth work remains host-authored and therefore skips the indirect shapes.
[numthreads(1, 1, 1)]
void finalize_main(uint3 dispatch_thread : SV_DispatchThreadID) {
    uint decision;
    uint request_work_flags;
    bool optional_ocr;
    bool receipt_valid = NearIdenticalSubtitleReceiptValid(
        decision, request_work_flags, optional_ocr);
    bool force_exact_publication =
        near_identical_request_flags == NEAR_IDENTICAL_REQUEST_FORCE_INFER &&
        NearIdenticalSubtitleConstantsValid() &&
        near_identical_expected_work != 0u;
    bool branch_independent =
        near_identical_expected_work == NEAR_IDENTICAL_WORK_OPTIONAL_OCR_DUE ||
        near_identical_expected_work == NEAR_IDENTICAL_WORK_SUBTITLE_OBSERVATION_DUE;
    bool run_observation = near_identical_expected_work != 0u &&
        ((receipt_valid &&
          (decision == NEAR_IDENTICAL_DECISION_INFER || branch_independent)) ||
         force_exact_publication);
    bool run_optional = run_observation &&
        (near_identical_expected_work == NEAR_IDENTICAL_WORK_OPTIONAL_OCR ||
         near_identical_expected_work == NEAR_IDENTICAL_WORK_OPTIONAL_OCR_DUE) &&
        receipt_valid && optional_ocr;
    bool run_record = run_observation && !run_optional;
    // This slot held the pre-CUDA OCR-preprocess args. Republish it now as the full-field
    // conditioner grid only for the authenticated infer observation.
    NearIdenticalWriteDispatchArgs(
        NEAR_IDENTICAL_SUBTITLE_CONDITION_GRID16_OFFSET,
        run_observation ? (target_w + 15u) / 16u : 0u,
        run_observation ? (target_h + 15u) / 16u : 1u,
        1u);
    NearIdenticalWriteDispatchArgs(
        NEAR_IDENTICAL_OPTIONAL_CELLS_OFFSET,
        run_optional ? (V2_OCR_OUTPUT_WIDTH / 8u + 31u) / 32u : 0u,
        run_optional ? (V2_OCR_OUTPUT_HEIGHT + 3u) / 4u : 1u,
        1u);
    NearIdenticalWriteDispatchArgs(
        NEAR_IDENTICAL_OPTIONAL_ONE_OFFSET,
        run_optional ? 1u : 0u,
        1u,
        1u);
    NearIdenticalWriteDispatchArgs(
        NEAR_IDENTICAL_SUBTITLE_RECORD_ONE_OFFSET,
        run_record ? 1u : 0u,
        1u,
        1u);
    NearIdenticalWriteDispatchArgs(
        NEAR_IDENTICAL_OBSERVATION_ONE_OFFSET,
        run_observation ? 1u : 0u,
        1u,
        1u);
    if (run_record) {
        NearIdenticalWriteOcrAbstention();
    }

    // A force-infer root runs every depth postprocess stage with host-authored direct dispatches.
    // Only an opaque GPU-undecided root may replace these records with receipt-gated arguments.
    if (near_identical_request_flags == NEAR_IDENTICAL_REQUEST_AUTHORIZED) {
        uint depth_decision;
        bool depth_receipt_valid = NearIdenticalDepthReceiptValid(depth_decision);
        bool run_infer =
            depth_receipt_valid && depth_decision == NEAR_IDENTICAL_DECISION_INFER;
        // The depth textures are rotated before these arguments are consumed. A malformed opaque
        // receipt cannot authorize current inference state, but it must still restore the retained
        // normalized field into the new target. This bounded hold advances no inference, scene,
        // history, OCR, SLR, or ownership state.
        bool hold_previous_depth = !run_infer;
        NearIdenticalWriteDispatchArgs(
            NEAR_IDENTICAL_INFER_REDUCE_OFFSET,
            run_infer ? near_identical_reduce_groups : 0u,
            1u,
            1u);
        NearIdenticalWriteDispatchArgs(
            NEAR_IDENTICAL_INFER_ONE_OFFSET, run_infer ? 1u : 0u, 1u, 1u);
        NearIdenticalWriteDispatchArgs(
            NEAR_IDENTICAL_INFER_GRID16_OFFSET,
            run_infer ? (target_w + 15u) / 16u : 0u,
            run_infer ? (target_h + 15u) / 16u : 1u,
            1u);
        NearIdenticalWriteDispatchArgs(
            NEAR_IDENTICAL_INFER_GRID8_OFFSET,
            run_infer ? (target_w + 7u) / 8u : 0u,
            run_infer ? (target_h + 7u) / 8u : 1u,
            1u);
        NearIdenticalWriteDispatchArgs(
            NEAR_IDENTICAL_INFER_COLUMNS_OFFSET,
            run_infer ? target_w : 0u,
            1u,
            1u);
        NearIdenticalWriteDispatchArgs(
            NEAR_IDENTICAL_INFER_ROWS_OFFSET,
            run_infer ? target_h : 0u,
            1u,
            1u);
        NearIdenticalWriteDispatchArgs(
            NEAR_IDENTICAL_REUSE_GRID16_OFFSET,
            hold_previous_depth ? (target_w + 15u) / 16u : 0u,
            hold_previous_depth ? (target_h + 15u) / 16u : 1u,
            1u);
    }
    DeviceMemoryBarrier();
}

[numthreads(16, 16, 1)]
void reuse_depth_main(uint3 dispatch_thread : SV_DispatchThreadID) {
    if (dispatch_thread.x < target_w && dispatch_thread.y < target_h) {
        NearIdenticalReuseDepthOutput[dispatch_thread.xy] =
            NearIdenticalPreviousDepth[dispatch_thread.xy];
    }
}
