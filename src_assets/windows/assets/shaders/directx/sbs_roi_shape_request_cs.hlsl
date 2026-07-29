// Convert the current GPU rule state into one exact, deterministic model-shape request.
//
// This pass performs only scalar validation and shape arithmetic. It never scans scene pixels,
// reads back to the CPU, pads a model tensor, or changes controller ownership.

StructuredBuffer<uint4> CurrentRuleState : register(t0);
RWStructuredBuffer<uint4> OutputRoiShapeRequest : register(u0);

#include "include/sbs_scene_controller_contract.generated.hlsl"
#include "include/sbs_roi_shape_request.hlsl"

// 64-byte upload-ring element. Canonical dimensions are the already validated full-frame DA-V2
// shape for this session. Profile maxima include engine, configured, and native-source caps.
cbuffer SbsRoiShapeRequestConstants : register(b0) {
    uint roi_shape_source_width;
    uint roi_shape_source_height;
    uint roi_shape_canonical_model_width;
    uint roi_shape_canonical_model_height;

    uint roi_shape_target_pixel_budget;
    uint roi_shape_profile_max_width;
    uint roi_shape_profile_max_height;
    uint roi_shape_expected_backend_generation;

    float roi_shape_quiet_halo_cells;
    uint roi_shape_analysis_canvas_size;
    float roi_shape_max_model_aspect;
    uint roi_shape_active_rules;

    uint4 roi_shape_reserved;
};

uint AlignDownToPatch(uint value) {
    return
        (value / SBS_ROI_SHAPE_REQUEST_PATCH_SIZE) *
        SBS_ROI_SHAPE_REQUEST_PATCH_SIZE;
}

uint AlignNearestToPatch(float value) {
    float patch = (float)SBS_ROI_SHAPE_REQUEST_PATCH_SIZE;
    return max(
        SBS_ROI_SHAPE_REQUEST_PATCH_SIZE,
        (uint)floor(value / patch + 0.5f) *
            SBS_ROI_SHAPE_REQUEST_PATCH_SIZE);
}

bool ShapeConfigurationValid() {
    // Canonical full-frame dimensions may legitimately exceed the active ROI aspect envelope.
    // CPU canonical sizing preserves the source aspect and lowers the short side to honor the
    // same pixel budget; only DeriveActiveModelShape applies max_model_aspect as a shape cap.
    return roi_shape_source_width > 0u &&
           roi_shape_source_height > 0u &&
           roi_shape_target_pixel_budget >=
               SBS_ROI_SHAPE_REQUEST_PATCH_SIZE *
               SBS_ROI_SHAPE_REQUEST_PATCH_SIZE &&
           roi_shape_target_pixel_budget <=
               SBS_ROI_SHAPE_REQUEST_ENGINE_MAX_DIMENSION *
               SBS_ROI_SHAPE_REQUEST_ENGINE_MAX_DIMENSION &&
           SbsRoiShapeRequestPatchAligned(
               roi_shape_canonical_model_width) &&
           SbsRoiShapeRequestPatchAligned(
               roi_shape_canonical_model_height) &&
           roi_shape_canonical_model_width <= roi_shape_source_width &&
           roi_shape_canonical_model_height <= roi_shape_source_height &&
           roi_shape_canonical_model_width <=
               SBS_ROI_SHAPE_REQUEST_ENGINE_MAX_DIMENSION &&
           roi_shape_canonical_model_height <=
               SBS_ROI_SHAPE_REQUEST_ENGINE_MAX_DIMENSION &&
           roi_shape_canonical_model_width <=
               roi_shape_profile_max_width &&
           roi_shape_canonical_model_height <=
               roi_shape_profile_max_height &&
           roi_shape_profile_max_width >=
               SBS_ROI_SHAPE_REQUEST_PATCH_SIZE &&
           roi_shape_profile_max_height >=
               SBS_ROI_SHAPE_REQUEST_PATCH_SIZE &&
           isfinite(roi_shape_quiet_halo_cells) &&
           roi_shape_quiet_halo_cells >= 0.0f &&
           roi_shape_analysis_canvas_size > 0u &&
           roi_shape_quiet_halo_cells <=
               (float)roi_shape_analysis_canvas_size &&
           isfinite(roi_shape_max_model_aspect) &&
           roi_shape_max_model_aspect >= 1.0f &&
           roi_shape_max_model_aspect <=
               SBS_ROI_SHAPE_REQUEST_MAX_ASPECT_LIMIT;
}

bool RuleSchemaValid(uint rule_schema_bits) {
    return rule_schema_bits == asuint((float)SBS_SCENE_SCHEMA_VERSION);
}

float4 ExpandedCommittedRoi(uint4 committed_roi_bits) {
    float4 roi = asfloat(committed_roi_bits);
    float halo =
        roi_shape_quiet_halo_cells /
        (float)roi_shape_analysis_canvas_size;
    return float4(
        max(roi.xy - halo.xx, 0.0f.xx),
        min(roi.zw + halo.xx, 1.0f.xx));
}

bool DeriveActiveModelShape(
    uint4 committed_roi_bits,
    out uint2 model_dimensions,
    out uint derived_flags)
{
    model_dimensions = uint2(
        roi_shape_canonical_model_width,
        roi_shape_canonical_model_height);
    derived_flags = 0u;

    float4 expanded = ExpandedCommittedRoi(committed_roi_bits);
    float2 physical_extent =
        (expanded.zw - expanded.xy) *
        float2(roi_shape_source_width, roi_shape_source_height);
    if (!all(isfinite(physical_extent)) ||
        any(physical_extent < float2(
            SBS_ROI_SHAPE_REQUEST_PATCH_SIZE,
            SBS_ROI_SHAPE_REQUEST_PATCH_SIZE))) {
        return false;
    }

    float physical_aspect = physical_extent.x / physical_extent.y;
    float minimum_aspect = 1.0f / roi_shape_max_model_aspect;
    float fitted_aspect = clamp(
        physical_aspect,
        minimum_aspect,
        roi_shape_max_model_aspect);
    if (fitted_aspect != physical_aspect) {
        derived_flags |= SBS_ROI_SHAPE_REQUEST_FLAG_ASPECT_CLAMPED;
    }

    uint max_width = AlignDownToPatch(min(
        min(
            roi_shape_profile_max_width,
            SBS_ROI_SHAPE_REQUEST_ENGINE_MAX_DIMENSION),
        min(
            roi_shape_source_width,
            (uint)floor(physical_extent.x))));
    uint max_height = AlignDownToPatch(min(
        min(
            roi_shape_profile_max_height,
            SBS_ROI_SHAPE_REQUEST_ENGINE_MAX_DIMENSION),
        min(
            roi_shape_source_height,
            (uint)floor(physical_extent.y))));
    if (max_width < SBS_ROI_SHAPE_REQUEST_PATCH_SIZE ||
        max_height < SBS_ROI_SHAPE_REQUEST_PATCH_SIZE) {
        return false;
    }

    float ideal_height = sqrt(
        (float)roi_shape_target_pixel_budget / fitted_aspect);
    float ideal_width = ideal_height * fitted_aspect;
    float fit_scale = min(
        1.0f,
        min(
            (float)max_width / ideal_width,
            (float)max_height / ideal_height));
    ideal_width *= fit_scale;
    ideal_height *= fit_scale;

    uint base_width = min(AlignNearestToPatch(ideal_width), max_width);
    uint base_height = min(AlignNearestToPatch(ideal_height), max_height);
    if (base_width < SBS_ROI_SHAPE_REQUEST_PATCH_SIZE ||
        base_height < SBS_ROI_SHAPE_REQUEST_PATCH_SIZE) {
        return false;
    }

    // Search the small patch-aligned neighborhood around the analytic solution. Aspect error is
    // primary because the transform must resize only real pixels; area error breaks close ties.
    float best_score = 3.402823466e+38f;
    uint2 best = uint2(base_width, base_height);
    bool found = false;
    [unroll]
    for (int y_offset = -2; y_offset <= 2; ++y_offset) {
        int candidate_height =
            (int)base_height +
            y_offset * (int)SBS_ROI_SHAPE_REQUEST_PATCH_SIZE;
        if (candidate_height <
                (int)SBS_ROI_SHAPE_REQUEST_PATCH_SIZE ||
            candidate_height > (int)max_height) {
            continue;
        }

        uint aspect_width = AlignNearestToPatch(
            (float)candidate_height * fitted_aspect);
        [unroll]
        for (int x_offset = -1; x_offset <= 1; ++x_offset) {
            int candidate_width =
                (int)aspect_width +
                x_offset * (int)SBS_ROI_SHAPE_REQUEST_PATCH_SIZE;
            if (candidate_width <
                    (int)SBS_ROI_SHAPE_REQUEST_PATCH_SIZE ||
                candidate_width > (int)max_width) {
                continue;
            }

            float candidate_aspect =
                (float)candidate_width / (float)candidate_height;
            if (candidate_aspect < minimum_aspect ||
                candidate_aspect > roi_shape_max_model_aspect) {
                continue;
            }
            float aspect_error =
                abs(candidate_aspect - fitted_aspect) /
                max(fitted_aspect, 1e-6f);
            float area_error =
                abs(
                    (float)(candidate_width * candidate_height) -
                    (float)roi_shape_target_pixel_budget) /
                (float)roi_shape_target_pixel_budget;
            float score = aspect_error * 4.0f + area_error;
            uint2 candidate =
                uint2((uint)candidate_width, (uint)candidate_height);
            bool lexicographically_smaller =
                candidate.y < best.y ||
                (candidate.y == best.y && candidate.x < best.x);
            if (score < best_score ||
                (score == best_score && lexicographically_smaller)) {
                best_score = score;
                best = candidate;
                found = true;
            }
        }
    }

    if (!found) {
        return false;
    }
    model_dimensions = best;
    bool profile_limited =
        fit_scale < 1.0f ||
        model_dimensions.x != AlignNearestToPatch(
            sqrt(
                (float)roi_shape_target_pixel_budget *
                fitted_aspect)) ||
        model_dimensions.y != AlignNearestToPatch(
            sqrt(
                (float)roi_shape_target_pixel_budget /
                fitted_aspect));
    if (profile_limited) {
        derived_flags |= SBS_ROI_SHAPE_REQUEST_FLAG_PROFILE_CLAMPED;
    }
    return true;
}

SbsRoiShapeRequestData MakeRequest(
    uint flags,
    uint reason,
    uint4 rule_header,
    uint4 rule_identity,
    uint4 committed_roi_bits,
    uint2 model_dimensions)
{
    SbsRoiShapeRequestData value = SbsRoiShapeRequestDecode(
        uint4(
            SBS_ROI_SHAPE_REQUEST_SCHEMA_VERSION,
            flags,
            reason,
            0u),
        uint4(
            rule_header.w,
            rule_identity.x,
            rule_identity.y,
            rule_header.x),
        uint4(
            roi_shape_source_width,
            roi_shape_source_height,
            model_dimensions.x,
            model_dimensions.y),
        committed_roi_bits);
    value.header.w = SbsRoiShapeRequestId(value);
    return value;
}

void StoreRequest(SbsRoiShapeRequestData value) {
    OutputRoiShapeRequest[SBS_ROI_SHAPE_REQUEST_VECTOR_HEADER] =
        value.header;
    OutputRoiShapeRequest[SBS_ROI_SHAPE_REQUEST_VECTOR_IDENTITY] =
        value.identity;
    OutputRoiShapeRequest[SBS_ROI_SHAPE_REQUEST_VECTOR_SHAPE] =
        value.shape;
    OutputRoiShapeRequest[
        SBS_ROI_SHAPE_REQUEST_VECTOR_COMMITTED_ROI] =
        value.committed_roi_bits;
}

[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
    if (any(dispatch_thread_id != 0u.xxx)) {
        return;
    }

    uint4 rule_header = CurrentRuleState[0];
    uint4 rule_identity = CurrentRuleState[1];
    uint4 committed_roi_bits = CurrentRuleState[2];
    uint reason = SBS_ROI_SHAPE_REQUEST_REASON_NONE;

    if (roi_shape_active_rules == 0u) {
        reason = SBS_ROI_SHAPE_REQUEST_REASON_INACTIVE;
    } else if (!ShapeConfigurationValid()) {
        reason =
            SBS_ROI_SHAPE_REQUEST_REASON_INVALID_SOURCE_OR_PROFILE;
    } else if (!RuleSchemaValid(rule_header.x)) {
        reason =
            SBS_ROI_SHAPE_REQUEST_REASON_CONTROLLER_SCHEMA_MISMATCH;
    } else if (
        rule_header.w != roi_shape_expected_backend_generation ||
        roi_shape_expected_backend_generation == 0u) {
        reason =
            SBS_ROI_SHAPE_REQUEST_REASON_BACKEND_GENERATION_MISMATCH;
    } else if (
        !isfinite(asfloat(rule_header.z)) ||
        asfloat(rule_header.z) <= 0.5f) {
        reason =
            SBS_ROI_SHAPE_REQUEST_REASON_CONTROLLER_OUTPUT_INVALID;
    } else if (
        (rule_identity.z & SBS_SCENE_STATE_FLAGS_INITIALIZED) == 0u) {
        reason =
            SBS_ROI_SHAPE_REQUEST_REASON_CONTROLLER_UNINITIALIZED;
    } else if (
        (rule_identity.z & SBS_SCENE_STATE_FLAGS_ROI_LOCKED) == 0u) {
        reason = SBS_ROI_SHAPE_REQUEST_REASON_ROI_NOT_LOCKED;
    } else if (
        (rule_identity.z &
         SBS_SCENE_STATE_FLAGS_FALLBACK_ACTIVE) != 0u) {
        reason =
            SBS_ROI_SHAPE_REQUEST_REASON_CONTROLLER_FALLBACK_ACTIVE;
    } else if (
        rule_identity.x == 0u ||
        rule_identity.y == 0u) {
        reason =
            SBS_ROI_SHAPE_REQUEST_REASON_INVALID_RULE_IDENTITY;
    } else if (!SbsRoiShapeRequestRectValid(committed_roi_bits)) {
        reason =
            SBS_ROI_SHAPE_REQUEST_REASON_MALFORMED_COMMITTED_ROI;
    }

    uint2 model_dimensions = uint2(
        roi_shape_canonical_model_width,
        roi_shape_canonical_model_height);
    uint derived_flags = 0u;
    // HLSL does not guarantee short-circuit evaluation around a function with out parameters.
    // Keep fallback control flow structurally separate so DeriveActiveModelShape cannot mutate
    // canonical dimensions for no-lock, malformed, or otherwise rejected controller states.
    if (reason == SBS_ROI_SHAPE_REQUEST_REASON_NONE) {
        if (!DeriveActiveModelShape(
                committed_roi_bits,
                model_dimensions,
                derived_flags)) {
            reason =
                SBS_ROI_SHAPE_REQUEST_REASON_INVALID_SOURCE_OR_PROFILE;
            model_dimensions = uint2(
                roi_shape_canonical_model_width,
                roi_shape_canonical_model_height);
            derived_flags = 0u;
        }
    }

    uint flags =
        SBS_ROI_SHAPE_REQUEST_FLAG_VALID |
        (reason == SBS_ROI_SHAPE_REQUEST_REASON_NONE ?
            SBS_ROI_SHAPE_REQUEST_FLAG_ACTIVE_ROI | derived_flags :
            SBS_ROI_SHAPE_REQUEST_FLAG_FULL_FRAME |
                SBS_ROI_SHAPE_REQUEST_FLAG_FALLBACK);
    StoreRequest(MakeRequest(
        flags,
        reason,
        rule_header,
        rule_identity,
        committed_roi_bits,
        model_dimensions));
}
