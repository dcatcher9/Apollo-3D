// Map raw DAV2 into the desired signed one-eye source-U candidate. The production entrypoint also
// commits the complete raw padded comparison/depth tuple and its authenticated owner. Completion of
// the direct or indirect dispatch followed by UAV unbinding is the publication boundary; the tag
// written by thread zero is not a same-dispatch synchronization point. coordinate_main is a
// Dump-3D-only diagnostic entrypoint over the same authenticated source and has no history or owner
// side effects.

StructuredBuffer<float> InputBuffer : register(t0);
StructuredBuffer<float4> ShadowState : register(t1);
Texture2D<uint> TensorExclusion : register(t2);
StructuredBuffer<float4> MinMaxEma : register(t3);
StructuredBuffer<float> CurrentModelInput : register(t4);
StructuredBuffer<float> CurrentAppearanceOrdinal : register(t5);
StructuredBuffer<float4> CutBridgeState : register(t6);
Texture2D<float> CurrentDepth : register(t7);
RWTexture2D<float> Output : register(u0);
RWStructuredBuffer<float> PreviousModelInput : register(u1);
RWStructuredBuffer<float> PreviousAppearanceOrdinal : register(u2);
RWTexture2D<float> PreviousReliableDepth : register(u3);
RWTexture2D<uint> PreviousTensorExclusion : register(u4);
RWStructuredBuffer<uint> NearIdenticalHistoryOwnerOutput : register(u5);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"
#include "include/depth_valid_history_contract.hlsl"
#define HOST_SBS_NEAR_IDENTICAL_CONSTANTS_REGISTER b2
#include "include/host_sbs_near_identical_constants.hlsl"
#include "include/host_sbs_near_identical_history_owner.hlsl"

bool DepthHistoryAdvances(out float history_state) {
    history_state = SBS_STATE_MODEL_INPUT_HISTORY_STATE(
        CutBridgeState[SBS_STATE_VECTOR_MODEL_INPUT_HISTORY_STATE]);
    return DepthValidHistoryAdvances(MinMaxEma[0].w, history_state);
}

void PublishNearIdenticalHistoryOwner(bool history_advances, float history_state) {
    NearIdenticalHistoryOwnerOutput[NEAR_IDENTICAL_HISTORY_WORD_CONTRACT_TAG] = 0u;
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

    bool owner_valid =
        history_advances &&
        asuint(SBS_STATE_CUT_CONTRACT_TAG_BITS(CutBridgeState[0])) ==
            SBS_CUT_CONTRACT_TAG &&
        V2Finite(history_state) &&
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
    DeviceMemoryBarrier();
    if (owner_valid) {
        NearIdenticalHistoryOwnerOutput[NEAR_IDENTICAL_HISTORY_WORD_CONTRACT_TAG] =
            NEAR_IDENTICAL_HISTORY_OWNER_TAG;
    }
}

bool CanonicalCoordinate(uint index, out float coordinate) {
    coordinate = 0.0f;
    float4 mapping_state = ShadowState[2];
    if (asuint(V2_STATE_RENDERER_AUTHORIZATION_BITS(mapping_state)) != V2_CONTRACT_TAG) {
        return false;
    }
    float4 active = ShadowState[0];
    // State resolve has already authenticated the current scene camera and made current-frame
    // validity atomic with it. The versioned authorization token above is the only state guard
    // needed in this per-texel pass.
    float raw = InputBuffer[index];
    if (!V2Finite(raw)) {
        return false;
    }
    coordinate = (raw - V2_STATE_CENTER(active)) * V2_STATE_INVERSE_SCALE(active);
    if (!V2Finite(coordinate)) {
        coordinate = 0.0f;
        return false;
    }
    return true;
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float history_state;
    bool history_advances = DepthHistoryAdvances(history_state);

    // Preserve the raw padded inference domain before any coordinate-map clamp or early exit.
    // Padded NCHW/depth cells are edge-replicated by their producers and exclusion remains 1.
    // The tuple is target-sized, so every in-bounds texel advances all six values together.
    if (id.x < target_w && id.y < target_h && history_advances) {
        uint plane = target_w * target_h;
        uint raw_index = id.y * target_w + id.x;
        PreviousModelInput[raw_index] = CurrentModelInput[raw_index];
        PreviousModelInput[raw_index + plane] = CurrentModelInput[raw_index + plane];
        PreviousModelInput[raw_index + 2u * plane] = CurrentModelInput[raw_index + 2u * plane];
        PreviousAppearanceOrdinal[raw_index] = CurrentAppearanceOrdinal[raw_index];
        PreviousReliableDepth[id.xy] = CurrentDepth[id.xy];
        PreviousTensorExclusion[id.xy] = TensorExclusion[id.xy];
    }

    if (all(id == uint3(0u, 0u, 0u))) {
        PublishNearIdenticalHistoryOwner(history_advances, history_state);
    }

    if (id.x >= target_w || id.y >= target_h) {
        return;
    }
    // Synthetic letterbox cells are a nearest-boundary extension of the real analysis domain.
    // Sanitizing before ownership and both limiters prevents arbitrary model padding output from
    // entering a valid row/column envelope.
    uint2 sample_position = TensorExclusion[id.xy] != 0u ?
        DepthAnalysisClampCell(id.xy) : id.xy;
    uint index = sample_position.y * target_w + sample_position.x;
    float coordinate;
    if (!CanonicalCoordinate(index, coordinate)) {
        Output[id.xy] = 0.0f;
        return;
    }
    float4 active = ShadowState[0];
    float requested = v2_requested_gain *
        (V2Curve(coordinate) -
         V2_STATE_CONVERGENCE_CURVE(active));
    float candidate = V2PointwiseContainer(requested);
    if (!V2Finite(candidate)) {
        candidate = 0.0f;
    }
    Output[id.xy] = candidate;
}

[numthreads(16, 16, 1)]
void coordinate_main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= target_w || id.y >= target_h) {
        return;
    }
    float coordinate;
    uint2 sample_position = TensorExclusion[id.xy] != 0u ?
        DepthAnalysisClampCell(id.xy) : id.xy;
    CanonicalCoordinate(sample_position.y * target_w + sample_position.x, coordinate);
    Output[id.xy] = coordinate;
}
