// GPU-only near-identical proposal, authenticated branch receipt, and postprocess gating.
// The calibrated RGB-to-NCHW producer remains untouched: this required closure reads its output
// and an independently owned state-1 history. The CPU never maps this evidence or decision.

StructuredBuffer<float> CurrentModelInput : register(t0);
StructuredBuffer<float> PreviousModelInput : register(t1);

struct NearIdenticalTileEvidence {
    uint admitted;
    uint medium_changed;
    uint strong_changed;
    uint nonfinite;
};

StructuredBuffer<NearIdenticalTileEvidence> NearIdenticalTileInput : register(t2);
StructuredBuffer<uint> NearIdenticalHistoryOwner : register(t3);
StructuredBuffer<float4> NearIdenticalCutBridgeState : register(t4);
Texture2D<float> NearIdenticalPreviousDepth : register(t5);
RWStructuredBuffer<NearIdenticalTileEvidence> NearIdenticalTileOutput : register(u0);
RWStructuredBuffer<uint> NearIdenticalHistoryOwnerOutput : register(u2);
RWByteAddressBuffer NearIdenticalDecision : register(u3);
RWTexture2D<float> NearIdenticalReuseDepthOutput : register(u4);
RWStructuredBuffer<uint> NearIdenticalSceneCutEvidence : register(u5);

#include "include/depth_constants.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"

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
    uint2 near_identical_reserved;
};

#define NEAR_IDENTICAL_REQUEST_AUTHORIZED (1u << 0u)
#define NEAR_IDENTICAL_MEDIUM_DELTA (1.0f / 64.0f)
#define NEAR_IDENTICAL_STRONG_DELTA 0.20f
#define NEAR_IDENTICAL_TILE_THREADS 256u
#define NEAR_IDENTICAL_RESOLVE_THREADS 64u
#define NEAR_IDENTICAL_REDUCE_ELEMENTS_PER_GROUP 256u
#define NEAR_IDENTICAL_RESOLVE_FLAG_MALFORMED (1u << 0u)
#define NEAR_IDENTICAL_RESOLVE_FLAG_LOCAL_VETO (1u << 1u)

#define NEAR_IDENTICAL_HISTORY_OWNER_TAG 0x3142484Eu
#define NEAR_IDENTICAL_HISTORY_OWNER_SCHEMA 1u

#define NEAR_IDENTICAL_DECISION_REUSE 0u
#define NEAR_IDENTICAL_DECISION_INFER 1u
#define NEAR_IDENTICAL_DECISION_COOKIE 0xD1EC15A5u
#define NEAR_IDENTICAL_TOKEN_LOW_COOKIE 0xA3756C91u
#define NEAR_IDENTICAL_TOKEN_HIGH_COOKIE 0x5C8A936Eu
#define NEAR_IDENTICAL_PROPOSAL_MAGIC 0x504F5250u
#define NEAR_IDENTICAL_RECEIPT_MAGIC 0x47524243u
#define NEAR_IDENTICAL_REQUEST_MAGIC 0x54535152u

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
#define NEAR_IDENTICAL_REQUEST_RESERVED_0_OFFSET 52u
#define NEAR_IDENTICAL_REQUEST_RESERVED_1_OFFSET 56u
#define NEAR_IDENTICAL_REQUEST_RESERVED_2_OFFSET 60u
#define NEAR_IDENTICAL_INFER_REDUCE_OFFSET 64u
#define NEAR_IDENTICAL_INFER_ONE_OFFSET 80u
#define NEAR_IDENTICAL_INFER_GRID16_OFFSET 96u
#define NEAR_IDENTICAL_INFER_GRID8_OFFSET 112u
#define NEAR_IDENTICAL_INFER_COLUMNS_OFFSET 128u
#define NEAR_IDENTICAL_INFER_ROWS_OFFSET 144u
#define NEAR_IDENTICAL_REUSE_GRID16_OFFSET 160u

#define NEAR_IDENTICAL_HISTORY_WORD_CONTRACT_TAG 0u
#define NEAR_IDENTICAL_HISTORY_WORD_CONTRACT_SCHEMA 1u
#define NEAR_IDENTICAL_HISTORY_WORD_FRAME_LOW 2u
#define NEAR_IDENTICAL_HISTORY_WORD_FRAME_HIGH 3u
#define NEAR_IDENTICAL_HISTORY_WORD_DOMAIN_TAG_LOW 4u
#define NEAR_IDENTICAL_HISTORY_WORD_DOMAIN_TAG_HIGH 5u
#define NEAR_IDENTICAL_HISTORY_WORD_TARGET_WIDTH 6u
#define NEAR_IDENTICAL_HISTORY_WORD_TARGET_HEIGHT 7u

groupshared uint4 NearIdenticalGroupPrimary[NEAR_IDENTICAL_TILE_THREADS];
groupshared uint4 NearIdenticalResolvePrimary[NEAR_IDENTICAL_RESOLVE_THREADS];
groupshared uint NearIdenticalResolveFlags[NEAR_IDENTICAL_RESOLVE_THREADS];

bool NearIdenticalFinite(float value) {
    return (asuint(value) & 0x7f800000u) != 0x7f800000u;
}

bool NearIdenticalOwnerIsNewer() {
    return near_identical_current_frame_high > near_identical_baseline_frame_high ||
        (near_identical_current_frame_high == near_identical_baseline_frame_high &&
         near_identical_current_frame_low > near_identical_baseline_frame_low);
}

bool NearIdenticalHistoryOwnerMatchesRequest() {
    return
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_CONTRACT_TAG] ==
            NEAR_IDENTICAL_HISTORY_OWNER_TAG &&
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_CONTRACT_SCHEMA] ==
            NEAR_IDENTICAL_HISTORY_OWNER_SCHEMA &&
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_FRAME_LOW] ==
            near_identical_baseline_frame_low &&
        NearIdenticalHistoryOwner[NEAR_IDENTICAL_HISTORY_WORD_FRAME_HIGH] ==
            near_identical_baseline_frame_high &&
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

[numthreads(16, 16, 1)]
void compare_main(
    uint3 dispatch_thread : SV_DispatchThreadID,
    uint3 group_thread : SV_GroupThreadID,
    uint3 group_id : SV_GroupID) {
    uint linear_thread = group_thread.y * 16u + group_thread.x;
    uint4 primary = 0u;
    if (dispatch_thread.x < target_w && dispatch_thread.y < target_h &&
        DepthAnalysisContentValid() &&
        DepthAnalysisCellIsContent(dispatch_thread.xy)) {
        uint plane = target_w * target_h;
        uint index = dispatch_thread.y * target_w + dispatch_thread.x;
        float3 current_nchw = float3(
            CurrentModelInput[index],
            CurrentModelInput[index + plane],
            CurrentModelInput[index + 2u * plane]);
        float3 previous_nchw = float3(
            PreviousModelInput[index],
            PreviousModelInput[index + plane],
            PreviousModelInput[index + 2u * plane]);
        bool finite =
            all(bool3(
                NearIdenticalFinite(current_nchw.r),
                NearIdenticalFinite(current_nchw.g),
                NearIdenticalFinite(current_nchw.b))) &&
            all(bool3(
                NearIdenticalFinite(previous_nchw.r),
                NearIdenticalFinite(previous_nchw.g),
                NearIdenticalFinite(previous_nchw.b)));
        float max_delta = 0.0f;
        if (finite) {
            // ImageNet means cancel. Scale the NCHW delta directly back to model RGB.
            float3 delta = abs(current_nchw - previous_nchw) *
                float3(0.229f, 0.224f, 0.225f);
            max_delta = max(delta.r, max(delta.g, delta.b));
        }
        primary = uint4(
            1u,
            finite && max_delta >= NEAR_IDENTICAL_MEDIUM_DELTA ? 1u : 0u,
            finite && max_delta >= NEAR_IDENTICAL_STRONG_DELTA ? 1u : 0u,
            finite ? 0u : 1u);
    }
    NearIdenticalGroupPrimary[linear_thread] = primary;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = NEAR_IDENTICAL_TILE_THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (linear_thread < stride) {
            NearIdenticalGroupPrimary[linear_thread] +=
                NearIdenticalGroupPrimary[linear_thread + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (linear_thread == 0u) {
        NearIdenticalTileEvidence evidence;
        evidence.admitted = NearIdenticalGroupPrimary[0].x;
        evidence.medium_changed = NearIdenticalGroupPrimary[0].y;
        evidence.strong_changed = NearIdenticalGroupPrimary[0].z;
        evidence.nonfinite = NearIdenticalGroupPrimary[0].w;
        uint tile_index = group_id.y * near_identical_tile_group_width + group_id.x;
        if (tile_index < near_identical_tile_group_count) {
            NearIdenticalTileOutput[tile_index] = evidence;
        }
    }
}

[numthreads(1, 1, 1)]
void history_owner_main(uint3 dispatch_thread : SV_DispatchThreadID) {
    NearIdenticalHistoryOwnerOutput[NEAR_IDENTICAL_HISTORY_WORD_CONTRACT_TAG] = 0u;
    float history_state = SBS_STATE_MODEL_INPUT_HISTORY_STATE(
        NearIdenticalCutBridgeState[SBS_STATE_VECTOR_MODEL_INPUT_HISTORY_STATE]);
    bool state_valid =
        asuint(SBS_STATE_CUT_CONTRACT_TAG_BITS(NearIdenticalCutBridgeState[0])) ==
            SBS_CUT_CONTRACT_TAG &&
        NearIdenticalFinite(history_state) && history_state == 1.0f;
    bool owner_valid =
        state_valid &&
        (near_identical_current_frame_low != 0u ||
         near_identical_current_frame_high != 0u) &&
        (near_identical_domain_tag_low != 0u ||
         near_identical_domain_tag_high != 0u) &&
        DepthAnalysisContentValid() &&
        near_identical_stream_frame_delta == 0u &&
        all(near_identical_reserved == 0u);
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
    DeviceMemoryBarrier();
    if (owner_valid) {
        NearIdenticalHistoryOwnerOutput[NEAR_IDENTICAL_HISTORY_WORD_CONTRACT_TAG] =
            NEAR_IDENTICAL_HISTORY_OWNER_TAG;
    }
}

// The CPU has already published the complete current RQST and cleared this proposal/args record
// in one ordered UpdateSubresource. CUDA must replace the tag-last PROP below with a CBRG receipt
// after the actual branch.
[numthreads(NEAR_IDENTICAL_RESOLVE_THREADS, 1, 1)]
void resolve_main(uint3 group_thread : SV_GroupThreadID) {
    uint linear_thread = group_thread.x;
    uint4 primary = 0u;
    uint resolve_flags = 0u;
    for (uint tile_index = linear_thread;
         tile_index < near_identical_tile_group_count;
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
            all(near_identical_reserved == 0u);
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
            NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_RESERVED_0_OFFSET) == 0u &&
            NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_RESERVED_1_OFFSET) == 0u &&
            NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_RESERVED_2_OFFSET) == 0u;
        bool tile_shape_valid =
            near_identical_tile_group_width == (target_w + 15u) / 16u &&
            near_identical_tile_group_height == (target_h + 15u) / 16u &&
            near_identical_tile_group_count ==
                near_identical_tile_group_width * near_identical_tile_group_height;
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
        NearIdenticalDecision.Store(
            NEAR_IDENTICAL_DECISION_MAGIC_OFFSET,
            NEAR_IDENTICAL_PROPOSAL_MAGIC);
    }
}

// After CUDA unmap, enable inference postprocess only for a matching authenticated CBRG receipt.
[numthreads(1, 1, 1)]
void postprocess_args_main(uint3 dispatch_thread : SV_DispatchThreadID) {
    uint decision = NearIdenticalDecision.Load(NEAR_IDENTICAL_DECISION_OFFSET);
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
        all(near_identical_reserved == 0u) &&
        request_token_low == near_identical_request_token_low &&
        request_token_high == near_identical_request_token_high;
    bool receipt_valid = constants_valid &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_DECISION_MAGIC_OFFSET) ==
            NEAR_IDENTICAL_RECEIPT_MAGIC &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_MAGIC_OFFSET) ==
            NEAR_IDENTICAL_REQUEST_MAGIC &&
        (decision == NEAR_IDENTICAL_DECISION_REUSE ||
         decision == NEAR_IDENTICAL_DECISION_INFER) &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_DECISION_COOKIE_OFFSET) ==
            (decision ^ NEAR_IDENTICAL_DECISION_COOKIE) &&
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
        NearIdenticalDecision.Load(NEAR_IDENTICAL_DECISION_RESERVED_OFFSET) == 0u &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_RESERVED_0_OFFSET) == 0u &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_RESERVED_1_OFFSET) == 0u &&
        NearIdenticalDecision.Load(NEAR_IDENTICAL_REQUEST_RESERVED_2_OFFSET) == 0u;
    bool run_infer = receipt_valid && decision == NEAR_IDENTICAL_DECISION_INFER;
    if (run_infer) {
        [unroll]
        for (uint word = 0u; word < 9u; ++word) {
            NearIdenticalSceneCutEvidence[word] = 0u;
        }
        NearIdenticalSceneCutEvidence[9u] = near_identical_stream_frame_delta;
    }
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
        run_infer ? 0u : (target_w + 15u) / 16u,
        run_infer ? 1u : (target_h + 15u) / 16u,
        1u);
    DeviceMemoryBarrier();
}

[numthreads(16, 16, 1)]
void reuse_depth_main(uint3 dispatch_thread : SV_DispatchThreadID) {
    if (dispatch_thread.x < target_w && dispatch_thread.y < target_h) {
        NearIdenticalReuseDepthOutput[dispatch_thread.xy] =
            NearIdenticalPreviousDepth[dispatch_thread.xy];
    }
}
