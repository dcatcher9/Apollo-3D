// Resolve the GPU-only scene-controller state into the exact transform owned by the next
// accepted DA-V2 inference. One lane is intentional: this writes eight uint4 values and performs
// only scalar rectangle arithmetic. It never scans a feature/depth grid.
//
// The CPU reserves `frame_roi_transform_version` and `frame_roi_bank_identity` before dispatch.
// Those values are written into the GPU record and remain unchanged through prepared -> pending
// -> completed ownership. Submission failure rolls the reservation back on the CPU; the shader
// never invents or advances lifecycle identity.

StructuredBuffer<uint4> CurrentRuleState : register(t0);
// The transform paired with the last VALID normalized depth surface. It is deliberately not the
// last submitted transform: reset debt must survive one or more invalid new-generation results.
StructuredBuffer<uint4> FrameRoiTransform : register(t1);
RWStructuredBuffer<uint4> OutputFrameRoiTransform : register(u0);

#include "include/sbs_scene_controller_contract.generated.hlsl"
#include "include/sbs_frame_roi_transform.hlsl"
#include "include/sbs_roi_shape_request.hlsl"

// 96-byte upload-ring element. Integer fields must be uploaded as integers, not float values.
cbuffer FrameRoiBuilderConstants : register(b0) {
    uint frame_roi_source_width;
    uint frame_roi_source_height;
    uint frame_roi_model_width;
    uint frame_roi_model_height;

    uint frame_roi_source_frame_id_low;
    uint frame_roi_source_frame_id_high;
    uint frame_roi_active_rules;
    uint frame_roi_expected_backend_generation;

    uint frame_roi_transform_version_low;
    uint frame_roi_transform_version_high;
    uint frame_roi_bank_identity;
    uint frame_roi_lifecycle_reserved;

    float frame_roi_quiet_halo_cells;
    float frame_roi_feather_cells;
    float frame_roi_min_focus_cells;
    uint frame_roi_expected_request_authority;

    uint frame_roi_analysis_canvas_size;
    uint frame_roi_expected_roi_generation;
    uint frame_roi_shape_request_id;
    uint frame_roi_expected_rule_update_count;

    uint4 frame_roi_expected_committed_roi_bits;
};

bool RuleControllerHeaderValid(
    uint4 header,
    uint4 identity,
    out uint roi_generation,
    out uint rule_update_count)
{
    float output_valid = asfloat(header.z);
    uint backend_generation = header.w;
    uint state_flags = identity.z;
    roi_generation = identity.x;
    rule_update_count = identity.y;
    return header.x ==
               SBS_ROI_SHAPE_CONTROLLER_SCHEMA_FLOAT_BITS &&
           isfinite(output_valid) &&
           output_valid > 0.5f &&
           backend_generation == frame_roi_expected_backend_generation &&
           backend_generation != 0u &&
           roi_generation == frame_roi_expected_roi_generation &&
           roi_generation != 0u &&
           frame_roi_expected_rule_update_count != 0u &&
           rule_update_count >= frame_roi_expected_rule_update_count &&
           (state_flags & SBS_SCENE_STATE_FLAGS_INITIALIZED) != 0u &&
           (state_flags & SBS_SCENE_STATE_FLAGS_FALLBACK_ACTIVE) == 0u;
}

SbsFrameRoiTransformData MakeTransformData(
    uint flags,
    uint roi_generation,
    uint accepted_focus_pixels,
    uint4 accepted_bounds,
    float4 focus,
    float4 crop,
    float4 feather,
    uint fallback_reason)
{
    uint backend_generation =
        frame_roi_active_rules != 0u ?
            frame_roi_expected_backend_generation :
            0u;
    return FrameRoiDecode(
        uint4(
            SBS_FRAME_ROI_SCHEMA_VERSION,
            flags,
            frame_roi_source_frame_id_low,
            frame_roi_source_frame_id_high),
        uint4(
            roi_generation,
            backend_generation,
            frame_roi_source_width,
            frame_roi_source_height),
        uint4(
            frame_roi_model_width,
            frame_roi_model_height,
            accepted_focus_pixels,
            frame_roi_shape_request_id),
        asuint(focus),
        asuint(crop),
        accepted_bounds,
        asuint(feather),
        uint4(
            frame_roi_transform_version_low,
            frame_roi_transform_version_high,
            frame_roi_bank_identity,
            fallback_reason));
}

void StoreTransform(SbsFrameRoiTransformData value) {
    OutputFrameRoiTransform[SBS_FRAME_ROI_VECTOR_HEADER] = value.header;
    OutputFrameRoiTransform[SBS_FRAME_ROI_VECTOR_IDENTITY] = value.identity;
    OutputFrameRoiTransform[SBS_FRAME_ROI_VECTOR_MODEL] = value.model;
    OutputFrameRoiTransform[SBS_FRAME_ROI_VECTOR_FOCUS] = value.focus_bits;
    OutputFrameRoiTransform[SBS_FRAME_ROI_VECTOR_CROP] = value.crop_bits;
    OutputFrameRoiTransform[SBS_FRAME_ROI_VECTOR_ACCEPTED_BOUNDS] =
        value.accepted_bounds;
    OutputFrameRoiTransform[SBS_FRAME_ROI_VECTOR_FEATHER] =
        value.feather_bits;
    OutputFrameRoiTransform[SBS_FRAME_ROI_VECTOR_DIAGNOSTICS] =
        value.diagnostics;
}

[numthreads(1, 1, 1)]
void main() {
    const float4 canonical_full =
        float4(0.0f, 0.0f, 1.0f, 1.0f);
    uint2 source_size =
        uint2(frame_roi_source_width, frame_roi_source_height);
    uint2 model_size =
        uint2(frame_roi_model_width, frame_roi_model_height);
    bool dimensions_valid =
        all(source_size > 0u.xx) &&
        all(model_size > 0u.xx) &&
        all((model_size % SBS_FRAME_ROI_MODEL_PATCH_SIZE) == 0u.xx) &&
        model_size.x <= 0xffffffffu / max(model_size.y, 1u);
    bool lifecycle_valid =
        (frame_roi_transform_version_low != 0u ||
         frame_roi_transform_version_high != 0u) &&
        frame_roi_bank_identity < SBS_FRAME_ROI_BANK_COUNT &&
        frame_roi_lifecycle_reserved == 1u;
    uint full_pixel_count =
        frame_roi_model_width * frame_roi_model_height;
    bool full_frame_shape_valid =
        dimensions_valid &&
        FrameRoiPhysicalAspectMatches(
            canonical_full,
            source_size,
            model_size,
            SBS_FRAME_ROI_LEGACY_ASPECT_REL_TOLERANCE);
    uint flags =
        dimensions_valid && lifecycle_valid && full_frame_shape_valid ?
            SBS_FRAME_ROI_FLAG_VALID | SBS_FRAME_ROI_FLAG_FULL_FRAME :
            0u;
    uint fallback_reason =
        !dimensions_valid ?
            SBS_FRAME_ROI_FALLBACK_INVALID_DIMENSIONS :
        !full_frame_shape_valid ?
            SBS_FRAME_ROI_FALLBACK_FULL_FRAME_SHAPE_MISMATCH :
            SBS_FRAME_ROI_FALLBACK_INACTIVE;
    uint roi_generation = 0u;
    uint rule_update_count = 0u;
    float4 focus = canonical_full;
    float4 crop = canonical_full;
    float4 feather = 0.0f.xxxx;
    uint4 accepted_bounds =
        uint4(0u, 0u, model_size);
    uint accepted_focus_pixels = full_pixel_count;

    if (dimensions_valid && lifecycle_valid && frame_roi_active_rules != 0u) {
        uint4 rule_header =
            CurrentRuleState[SBS_SCENE_RULE_STATE_VECTOR_SCHEMA_VERSION];
        uint4 rule_identity =
            CurrentRuleState[SBS_SCENE_RULE_STATE_VECTOR_ROI_GENERATION];
        bool controller_valid =
            RuleControllerHeaderValid(
                rule_header,
                rule_identity,
                roi_generation,
                rule_update_count);
        if (!controller_valid) {
            fallback_reason = SBS_FRAME_ROI_FALLBACK_CONTROLLER_INVALID;
            roi_generation = 0u;
        } else if (
            (rule_identity.z & SBS_SCENE_STATE_FLAGS_ROI_LOCKED) == 0u
        ) {
            fallback_reason = SBS_FRAME_ROI_FALLBACK_NO_LOCKED_ROI;
        } else {
            uint4 candidate_focus_bits =
                CurrentRuleState[
                    SBS_SCENE_RULE_STATE_VECTOR_COMMITTED_ROI_X0];
            uint expected_shape_request_id =
                SbsRoiShapeRequestIdFromFields(
                    SBS_ROI_SHAPE_REQUEST_SCHEMA_VERSION,
                    frame_roi_expected_backend_generation,
                    frame_roi_expected_roi_generation,
                    frame_roi_expected_rule_update_count,
                    source_size,
                    model_size,
                    frame_roi_expected_committed_roi_bits);
            uint expected_request_flags =
                frame_roi_expected_request_authority & 0xffffu;
            uint expected_request_reason =
                frame_roi_expected_request_authority >> 16u;
            bool request_authorizes_active_roi =
                (expected_request_flags &
                 SBS_ROI_SHAPE_REQUEST_FLAG_ACTIVE_ROI) != 0u &&
                (expected_request_flags &
                 (SBS_ROI_SHAPE_REQUEST_FLAG_FULL_FRAME |
                  SBS_ROI_SHAPE_REQUEST_FLAG_FALLBACK)) == 0u &&
                expected_request_reason ==
                    SBS_ROI_SHAPE_REQUEST_REASON_NONE;
            bool shape_request_matches =
                request_authorizes_active_roi &&
                frame_roi_shape_request_id != 0u &&
                frame_roi_shape_request_id ==
                    expected_shape_request_id &&
                all(candidate_focus_bits ==
                    frame_roi_expected_committed_roi_bits);
            float4 candidate_focus = asfloat(candidate_focus_bits);
            uint analysis_canvas =
                max(frame_roi_analysis_canvas_size, 1u);
            float min_focus_extent =
                max(frame_roi_min_focus_cells, 1.0f) /
                (float)analysis_canvas;
            if (!shape_request_matches) {
                fallback_reason =
                    SBS_FRAME_ROI_FALLBACK_SHAPE_REQUEST_MISMATCH;
            } else if (!FrameRoiRectValid(candidate_focus)) {
                fallback_reason = SBS_FRAME_ROI_FALLBACK_MALFORMED_ROI;
            } else if (
                candidate_focus.z - candidate_focus.x < min_focus_extent ||
                candidate_focus.w - candidate_focus.y < min_focus_extent
            ) {
                fallback_reason = SBS_FRAME_ROI_FALLBACK_ROI_TOO_SMALL;
            } else {
                focus = candidate_focus;
                float halo =
                    max(frame_roi_quiet_halo_cells, 0.0f) /
                    (float)analysis_canvas;
                float4 halo_bounds = float4(
                    max(focus.xy - halo.xx, 0.0f.xx),
                    min(focus.zw + halo.xx, 1.0f.xx));
                float2 halo_size = halo_bounds.zw - halo_bounds.xy;

                // The fixed tensor is never padded. Expand the accepted focus+quiet halo to the
                // smallest enclosing SOURCE rectangle whose physical aspect exactly matches the
                // tensor, then shift that rectangle inside the source while preserving enclosure.
                float normalized_target_ratio =
                    ((float)frame_roi_model_width /
                     (float)frame_roi_model_height) *
                    ((float)frame_roi_source_height /
                     (float)frame_roi_source_width);
                float crop_width;
                float crop_height;
                if (
                    halo_size.x / halo_size.y >
                    normalized_target_ratio
                ) {
                    crop_width = halo_size.x;
                    crop_height =
                        crop_width / normalized_target_ratio;
                } else {
                    crop_height = halo_size.y;
                    crop_width =
                        crop_height * normalized_target_ratio;
                }

                bool aspect_envelope_possible =
                    isfinite(crop_width) &&
                    isfinite(crop_height) &&
                    crop_width > 0.0f &&
                    crop_height > 0.0f &&
                    crop_width <= 1.0f + 1e-6f &&
                    crop_height <= 1.0f + 1e-6f;
                if (!aspect_envelope_possible) {
                    fallback_reason =
                        SBS_FRAME_ROI_FALLBACK_ASPECT_ENVELOPE_IMPOSSIBLE;
                    focus = canonical_full;
                } else {
                    crop_width = min(crop_width, 1.0f);
                    crop_height = min(crop_height, 1.0f);
                    float2 crop_size = float2(crop_width, crop_height);
                    float2 desired_lo =
                        0.5f * (halo_bounds.xy + halo_bounds.zw - crop_size);
                    float2 allowed_lo = max(
                        0.0f.xx,
                        halo_bounds.zw - crop_size);
                    float2 allowed_hi = min(
                        halo_bounds.xy,
                        1.0f.xx - crop_size);
                    bool placement_possible =
                        all(allowed_lo <= allowed_hi + 1e-6f.xx);
                    if (!placement_possible) {
                        fallback_reason =
                            SBS_FRAME_ROI_FALLBACK_ASPECT_ENVELOPE_IMPOSSIBLE;
                        focus = canonical_full;
                    } else {
                        float2 crop_lo =
                            clamp(desired_lo, allowed_lo, allowed_hi);
                        crop = float4(crop_lo, crop_lo + crop_size);
                        // Remove insignificant arithmetic spill at the source edge while keeping
                        // the enclosing aspect construction untouched in the interior.
                        crop.xy = max(crop.xy, 0.0f.xx);
                        crop.zw = min(crop.zw, 1.0f.xx);

                        accepted_bounds =
                            FrameRoiExpectedAcceptedModelBounds(
                                focus,
                                crop,
                                model_size);
                        accepted_focus_pixels =
                            FrameRoiAcceptedBoundsValid(
                                accepted_bounds,
                                model_size) ?
                                FrameRoiAcceptedBoundsPixelCount(
                                    accepted_bounds) :
                                0u;
                        if (accepted_focus_pixels == 0u) {
                            fallback_reason =
                                SBS_FRAME_ROI_FALLBACK_EMPTY_FOCUS;
                            focus = canonical_full;
                            crop = canonical_full;
                            accepted_bounds =
                                uint4(0u, 0u, model_size);
                            accepted_focus_pixels = full_pixel_count;
                        } else {
                            float2 focus_size = focus.zw - focus.xy;
                            float feather_width =
                                max(frame_roi_feather_cells, 0.0f) /
                                (float)analysis_canvas;
                            float2 feather_xy = min(
                                feather_width.xx,
                                0.25f * focus_size);
                            feather = float4(
                                feather_xy.x,
                                feather_xy.y,
                                feather_xy.x,
                                feather_xy.y);
                            flags =
                                SBS_FRAME_ROI_FLAG_VALID |
                                SBS_FRAME_ROI_FLAG_ACTIVE_ROI;
                            fallback_reason = SBS_FRAME_ROI_FALLBACK_NONE;
                        }
                    }
                }
            }
        }
    }

    if ((flags & SBS_FRAME_ROI_FLAG_ACTIVE_ROI) == 0u) {
        focus = canonical_full;
        crop = canonical_full;
        feather = 0.0f.xxxx;
        accepted_bounds = uint4(0u, 0u, model_size);
        accepted_focus_pixels = full_pixel_count;
        flags =
            dimensions_valid &&
            lifecycle_valid &&
            full_frame_shape_valid ?
                SBS_FRAME_ROI_FLAG_VALID |
                    SBS_FRAME_ROI_FLAG_FULL_FRAME :
                0u;
        if (
            dimensions_valid &&
            lifecycle_valid &&
            !full_frame_shape_valid
        ) {
            fallback_reason =
                SBS_FRAME_ROI_FALLBACK_FULL_FRAME_SHAPE_MISMATCH;
        }
    }

    SbsFrameRoiTransformData candidate = MakeTransformData(
        flags,
        roi_generation,
        accepted_focus_pixels,
        accepted_bounds,
        focus,
        crop,
        feather,
        fallback_reason);

    // Compare with the transform paired with the last VALID depth. Since invalid completions do
    // not promote that resource, this bit remains set until a real depth for the new geometry has
    // seeded normalization, EMA, adaptive state, and rendering.
    if (
        FrameRoiDataValid(candidate) &&
        !FrameRoiDataSameGeometry(candidate, FrameRoiTransformLoad())
    ) {
        candidate.header.y |= SBS_FRAME_ROI_FLAG_RESET_DEBT;
    }
    StoreTransform(candidate);
}
