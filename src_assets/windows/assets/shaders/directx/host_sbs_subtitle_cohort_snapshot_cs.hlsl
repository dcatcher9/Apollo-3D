// PRIVATE RESEARCH snapshot only. Independent of the frozen three-entry cohort closure.
// Dispatch after every reader of the old private tuple, using the canonical subtitle
// condition grid (zero groups for reuse). All four private destinations are paired.
// Raw numeric faults are preserved for next Prepare's validation; an invalid current
// context publishes zero SCT identity and can never certify a clean reference.
#include "include/depth_coordinate_v2_ocr_assert.generated.hlsl"
#include "include/host_sbs_subtitle_cohort_contract.h"

cbuffer SubtitleCohortSnapshotConstants : register(b0) {
    uint4 cohort_source;
    uint4 cohort_field;
    uint4 cohort_content;
    uint4 cohort_frame;
    uint4 cohort_observation;
    uint4 cohort_control;
};

Texture2D<float4> CompletedSource : register(t0);
StructuredBuffer<float> CompletedP : register(t1);
StructuredBuffer<uint> CurrentSCT : register(t2);
Texture2D<uint> CurrentRawBaselineS : register(t3);
RWTexture2D<float4> PrivatePreviousSource : register(u0);
RWStructuredBuffer<float> PrivatePreviousP : register(u1);
RWStructuredBuffer<uint> PrivatePreviousSCT : register(u2);
RWTexture2D<uint> PrivatePreviousRawS : register(u3);

groupshared uint SnapshotIdentityValid;

bool CurrentSnapshotIdentityValid() {
    uint words, word_stride;
    CurrentSCT.GetDimensions(words, word_stride);
    if (words != V2_SUBTITLE_CONTOUR_HISTORY_WORD_COUNT || word_stride != 4u ||
        cohort_control.x != 1u || cohort_control.y != SBC_SCHEMA || any(cohort_control.zw != 0u) ||
        cohort_observation.z > 1u || cohort_observation.w != 0u ||
        !any(cohort_frame.xy != 0u) || !any(cohort_observation.xy != 0u) ||
        any(cohort_source.xy == 0u) || any(cohort_source.xy > V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT) ||
        any(cohort_field.xy == 0u) || any(cohort_field.xy > V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT) ||
        !V2SubtitleOcrFieldIsCalibrated(cohort_field.x, cohort_field.y) ||
        cohort_content.x >= cohort_content.z || cohort_content.y >= cohort_content.w ||
        any(cohort_content.zw > cohort_field.xy)) return false;
    uint crop_top, crop_height, roi_top, roi_bottom;
    if (!V2SubtitleOcrComputeCrop(cohort_source.x, cohort_source.y, crop_top, crop_height) ||
        crop_top != cohort_source.z || crop_height != cohort_source.w ||
        !V2SubtitleOcrProjectContentRowCeil(cohort_source.x, cohort_source.y,
            cohort_field.x, cohort_field.y, cohort_content, V2_OCR_SAFE_ROW_TOP, roi_top) ||
        !V2SubtitleOcrProjectContentRowCeil(cohort_source.x, cohort_source.y,
            cohort_field.x, cohort_field.y, cohort_content, V2_OCR_SAFE_ROW_BOTTOM, roi_bottom) ||
        roi_top != cohort_field.z || roi_bottom != cohort_field.w || roi_top >= roi_bottom) return false;
    // word2 is current pin authority, not OCR empty or snapshot readiness.
    if (CurrentSCT[0u] != V2_SUBTITLE_CONTOUR_SCHEMA ||
        CurrentSCT[1u] != V2_SUBTITLE_CONTOUR_HISTORY_TAG || CurrentSCT[2u] > 1u ||
        CurrentSCT[22u] != V2_SUBTITLE_CONTOUR_RECIPE ||
        CurrentSCT[V2_SUBTITLE_CONTOUR_HISTORY_SNAPSHOT_VALID_WORD] != 1u) return false;
    [unroll]
    for (uint snapshot_component = 0u; snapshot_component < 4u; ++snapshot_component) {
        if (CurrentSCT[4u + snapshot_component] != cohort_frame[snapshot_component] ||
            CurrentSCT[8u + snapshot_component] != cohort_source[snapshot_component] ||
            CurrentSCT[12u + snapshot_component] != cohort_field[snapshot_component] ||
            CurrentSCT[16u + snapshot_component] != cohort_content[snapshot_component]) return false;
    }
    if (any(uint2(CurrentSCT[20u], CurrentSCT[21u]) != cohort_observation.xy)) return false;
    [unroll]
    for (uint snapshot_reserved = V2_SUBTITLE_CONTOUR_HISTORY_RESERVED_OFFSET;
         snapshot_reserved < V2_SUBTITLE_CONTOUR_HISTORY_WORD_COUNT; ++snapshot_reserved) {
        if (CurrentSCT[snapshot_reserved] != 0u) return false;
    }
    // Cut word3 is already GPU-owned SCT evidence. There is no CPU cut mirror in b0.
    return true;
}

[numthreads(16, 16, 1)]
void snapshot_main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint group_index : SV_GroupIndex) {
    uint sw, sh, rw, rh, ow, oh, bw, bh, pn, ps, opn, ops, on, os;
    CompletedSource.GetDimensions(sw, sh);
    CurrentRawBaselineS.GetDimensions(rw, rh);
    PrivatePreviousSource.GetDimensions(ow, oh);
    PrivatePreviousRawS.GetDimensions(bw, bh);
    CompletedP.GetDimensions(pn, ps);
    PrivatePreviousP.GetDimensions(opn, ops);
    PrivatePreviousSCT.GetDimensions(on, os);
    bool planes_valid = sw == cohort_source.x && sh == cohort_source.w &&
        rw == sw && rh == sh && ow == sw && oh == sh && bw == sw && bh == sh &&
        pn == V2_OCR_OUTPUT_WIDTH * V2_OCR_OUTPUT_HEIGHT && ps == 4u && opn == pn && ops == 4u &&
        on == V2_SUBTITLE_CONTOUR_HISTORY_WORD_COUNT && os == 4u;
    if (group_index == 0u) SnapshotIdentityValid = planes_valid && CurrentSnapshotIdentityValid() ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();

    // An invalid CB cannot define a trustworthy dispatch stride. Exactly group0
    // invalidates/copies the tuple with a fixed stride, avoiding overlapping writes
    // when a malformed field disagrees with the caller's canonical dispatch grid.
    if (SnapshotIdentityValid == 0u && any(group_id != 0u)) return;
    uint2 stride = SnapshotIdentityValid != 0u ?
        max(((cohort_field.xy + 15u) / 16u) * 16u, uint2(16u, 16u)) : uint2(16u, 16u);
    [loop]
    for (uint snapshot_y = dispatch_id.y; snapshot_y < oh; snapshot_y += stride.y) {
        [loop]
        for (uint snapshot_x = dispatch_id.x; snapshot_x < ow; snapshot_x += stride.x) {
            float4 source_value = 0.0f;
            if (planes_valid) source_value = CompletedSource.Load(int3(snapshot_x, snapshot_y, 0));
            PrivatePreviousSource[uint2(snapshot_x, snapshot_y)] = source_value;
        }
    }
    [loop]
    for (uint raw_y = dispatch_id.y; raw_y < bh; raw_y += stride.y) {
        [loop]
        for (uint raw_x = dispatch_id.x; raw_x < bw; raw_x += stride.x) {
            uint raw_value = 0u;
            if (planes_valid) raw_value = CurrentRawBaselineS.Load(int3(raw_x, raw_y, 0));
            PrivatePreviousRawS[uint2(raw_x, raw_y)] = raw_value;
        }
    }
    uint flat_thread = dispatch_id.y * stride.x + dispatch_id.x;
    uint flat_stride = stride.x * stride.y;
    [loop]
    for (uint probability_word = flat_thread; probability_word < opn; probability_word += flat_stride) {
        float probability_value = 0.0f;
        if (planes_valid) probability_value = asfloat(asuint(CompletedP[probability_word]));
        PrivatePreviousP[probability_word] = probability_value;
    }
    [loop]
    for (uint identity_word = flat_thread; identity_word < on; identity_word += flat_stride) {
        uint identity_value = 0u;
        if (SnapshotIdentityValid != 0u) identity_value = CurrentSCT[identity_word];
        PrivatePreviousSCT[identity_word] = identity_value;
    }
}
