// Preserve the exact preprocessed NCHW color only for frames whose TensorRT output contained a
// valid depth sample and whose resolve pass selected history advance. State 2 holds the last
// structurally reliable color/ordinal/depth tuple for one black or fully clipped update, allowing
// an immediate supported return to be compared with the last visible scene. State 3 advances a
// persistent-low endpoint, so persistent content cannot remain vetoed indefinitely. State 4
// holds a geometry-only candidate's pre-change endpoint for one confirmation update.
StructuredBuffer<float4> MinMaxEma : register(t0);  // w = current-frame validity
StructuredBuffer<float> CurrentModelInput : register(t1);
StructuredBuffer<float> CurrentAppearanceOrdinal : register(t2);
StructuredBuffer<float4> SubjectState : register(t3);  // [2].w: 0 empty, 1 advance, 2/4 hold, 3 low
Texture2D<float> CurrentDepth : register(t4);
StructuredBuffer<float> CurrentRawDepth : register(t5);
StructuredBuffer<uint4> FrameRoiTransform : register(t6);
RWStructuredBuffer<float> PreviousModelInput : register(u0);
RWStructuredBuffer<float> PreviousAppearanceOrdinal : register(u1);
RWTexture2D<float> PreviousReliableDepth : register(u2);
RWStructuredBuffer<uint4> PreviousFrameRoiTransform : register(u3);
RWTexture2D<uint> PreviousReliableValidity : register(u4);

#include "include/depth_constants.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"
#include "include/sbs_frame_roi_transform.hlsl"

bool TryTargetPlane(out uint plane) {
    plane = 0u;
    if (target_w == 0u ||
        target_h == 0u ||
        target_w > 0xffffffffu / target_h) {
        return false;
    }
    plane = target_w * target_h;
    return true;
}

bool FrameRoiAcceptedBoundsContains(
    uint4 bounds,
    uint2 pixel)
{
    return all(pixel >= bounds.xy) &&
           all(pixel < bounds.zw);
}

[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint minmax_count;
    uint minmax_stride;
    MinMaxEma.GetDimensions(minmax_count, minmax_stride);
    uint current_model_count;
    uint current_model_stride;
    CurrentModelInput.GetDimensions(
        current_model_count,
        current_model_stride);
    uint current_ordinal_count;
    uint current_ordinal_stride;
    CurrentAppearanceOrdinal.GetDimensions(
        current_ordinal_count,
        current_ordinal_stride);
    uint subject_state_count;
    uint subject_state_stride;
    SubjectState.GetDimensions(
        subject_state_count,
        subject_state_stride);
    uint current_depth_width;
    uint current_depth_height;
    CurrentDepth.GetDimensions(
        current_depth_width,
        current_depth_height);
    uint current_raw_count;
    uint current_raw_stride;
    CurrentRawDepth.GetDimensions(
        current_raw_count,
        current_raw_stride);
    uint current_transform_vectors;
    uint current_transform_stride;
    FrameRoiTransform.GetDimensions(
        current_transform_vectors,
        current_transform_stride);
    uint previous_model_count;
    uint previous_model_stride;
    PreviousModelInput.GetDimensions(
        previous_model_count,
        previous_model_stride);
    uint previous_ordinal_count;
    uint previous_ordinal_stride;
    PreviousAppearanceOrdinal.GetDimensions(
        previous_ordinal_count,
        previous_ordinal_stride);
    uint previous_depth_width;
    uint previous_depth_height;
    PreviousReliableDepth.GetDimensions(
        previous_depth_width,
        previous_depth_height);
    uint previous_transform_vectors;
    uint previous_transform_stride;
    PreviousFrameRoiTransform.GetDimensions(
        previous_transform_vectors,
        previous_transform_stride);
    uint previous_validity_width;
    uint previous_validity_height;
    PreviousReliableValidity.GetDimensions(
        previous_validity_width,
        previous_validity_height);

    bool state_resources_safe =
        minmax_count >= 1u &&
        minmax_stride == 16u &&
        subject_state_count >= SBS_ADAPTIVE_STATE_VECTOR_COUNT &&
        subject_state_stride == 16u;
    float history_state =
        state_resources_safe ?
        SBS_STATE_MODEL_INPUT_HISTORY_STATE(
            SubjectState[SBS_STATE_VECTOR_MODEL_INPUT_HISTORY_STATE]) :
        0.0f;
    bool hold_history =
        (history_state > 1.5f &&
         history_state < 2.5f) ||
        (history_state > 3.5f &&
         history_state < 4.5f);
    bool promote_history =
        state_resources_safe &&
        MinMaxEma[0].w >= 0.5f &&
        !hold_history;
    SbsFrameRoiTransformData transform =
        FrameRoiTransformLoad();
    // The explicit zero current transform is sufficient to select the legacy path. Never read
    // PreviousFrameRoiTransform in this dispatch: lane (0,0) promotes that same UAV for active
    // frames and D3D11 provides no cross-group ordering between the read and eight-vector write.
    // Legacy mode intentionally leaves the transform UAV untouched, so it needs no prior-value
    // check.
    bool legacy_unbound_pair =
        FrameRoiDataUnboundZero(transform);
    uint plane;
    bool target_plane_safe =
        TryTargetPlane(plane);
    bool legacy_resources_safe =
        target_plane_safe &&
        current_model_stride == 4u &&
        plane <= current_model_count / 3u &&
        current_ordinal_stride == 4u &&
        current_ordinal_count >= plane &&
        current_depth_width >= target_w &&
        current_depth_height >= target_h &&
        previous_model_stride == 4u &&
        plane <= previous_model_count / 3u &&
        previous_ordinal_stride == 4u &&
        previous_ordinal_count >= plane &&
        previous_depth_width >= target_w &&
        previous_depth_height >= target_h;

    // Exact backward-compatible path. It touches only the three historical tuple UAVs; the
    // transform and per-pixel-validity outputs may remain completely unbound.
    if (legacy_unbound_pair) {
        if (dtid.x >= target_w ||
            dtid.y >= target_h ||
            !promote_history ||
            !legacy_resources_safe) {
            return;
        }
        uint legacy_idx =
            dtid.y * target_w + dtid.x;
        PreviousModelInput[legacy_idx] =
            CurrentModelInput[legacy_idx];
        PreviousModelInput[legacy_idx + plane] =
            CurrentModelInput[legacy_idx + plane];
        PreviousModelInput[legacy_idx + 2u * plane] =
            CurrentModelInput[legacy_idx + 2u * plane];
        PreviousAppearanceOrdinal[legacy_idx] =
            CurrentAppearanceOrdinal[legacy_idx];
        PreviousReliableDepth[dtid.xy] =
            CurrentDepth[dtid.xy];
        return;
    }

    bool current_transform_valid =
        current_transform_vectors >= SBS_FRAME_ROI_VECTOR_COUNT &&
        current_transform_stride == 16u &&
        FrameRoiDataValid(transform);
    bool transform_dimensions_match =
        current_transform_valid &&
        all(FrameRoiDataModelDimensions(transform) ==
            uint2(target_w, target_h));
    bool active_resources_safe =
        legacy_resources_safe &&
        current_raw_stride == 4u &&
        current_raw_count >= plane &&
        previous_transform_vectors >= SBS_FRAME_ROI_VECTOR_COUNT &&
        previous_transform_stride == 16u &&
        previous_validity_width >= target_w &&
        previous_validity_height >= target_h;
    bool promote_active_history =
        promote_history &&
        transform_dimensions_match &&
        active_resources_safe;

    // One lane promotes the exact eight-vector transform with the NCHW/ordinal/depth tuple.
    // D3D11 makes the completed dispatch visible as a unit to later passes; an invalid completion
    // or one-update structureless/geometry-confirmation hold writes neither the tuple nor its
    // ownership metadata.
    if (dtid.x == 0u && dtid.y == 0u &&
        promote_active_history) {
        [unroll]
        for (uint vector_index = 0u;
             vector_index < SBS_FRAME_ROI_VECTOR_COUNT;
             ++vector_index) {
            PreviousFrameRoiTransform[vector_index] =
                FrameRoiTransform[vector_index];
        }
    }

    if (dtid.x >= target_w || dtid.y >= target_h ||
        !promote_active_history)
        return;

    bool canonical_full_frame =
        FrameRoiDataCanonicalFullFrame(transform);
    uint4 accepted_bounds =
        FrameRoiDataAcceptedModelBounds(transform);
    uint idx = dtid.y * target_w + dtid.x;
    float raw_depth = CurrentRawDepth[idx];
    float normalized_depth =
        CurrentDepth[dtid.xy];
    float model_r = CurrentModelInput[idx];
    float model_g = CurrentModelInput[idx + plane];
    float model_b = CurrentModelInput[idx + 2u * plane];
    float ordinal =
        CurrentAppearanceOrdinal[idx];
    bool accepted =
        canonical_full_frame ||
        FrameRoiAcceptedBoundsContains(
            accepted_bounds,
            dtid.xy);
    bool reliable =
        accepted &&
        !isnan(raw_depth) &&
        !isinf(raw_depth) &&
        raw_depth >= 0.0f &&
        !isnan(normalized_depth) &&
        !isinf(normalized_depth) &&
        !isnan(model_r) &&
        !isinf(model_r) &&
        !isnan(model_g) &&
        !isinf(model_g) &&
        !isnan(model_b) &&
        !isinf(model_b) &&
        !isnan(ordinal) &&
        !isinf(ordinal);
    if (!reliable) {
        // The transform is promoted for this frame, so stale values from its predecessor cannot
        // remain addressable under the new ownership metadata. Clear the whole tuple and mark the
        // pixel invalid; every temporal/ordinal consumer abstains on a zero mask.
        PreviousModelInput[idx] = 0.0f;
        PreviousModelInput[idx + plane] = 0.0f;
        PreviousModelInput[idx + 2u * plane] = 0.0f;
        PreviousAppearanceOrdinal[idx] = 0.0f;
        PreviousReliableDepth[dtid.xy] = 0.0f;
        PreviousReliableValidity[dtid.xy] = 0u;
        return;
    }
    PreviousModelInput[idx] = model_r;
    PreviousModelInput[idx + plane] = model_g;
    PreviousModelInput[idx + 2u * plane] = model_b;
    PreviousAppearanceOrdinal[idx] = ordinal;
    PreviousReliableDepth[dtid.xy] = normalized_depth;
    PreviousReliableValidity[dtid.xy] = 1u;
}
