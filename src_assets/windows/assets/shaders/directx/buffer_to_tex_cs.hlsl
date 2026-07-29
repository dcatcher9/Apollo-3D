StructuredBuffer<float>  InputBuffer : register(t0);
StructuredBuffer<float4> MinMaxEma   : register(t1);  // [0]={P2,P98,initialized,frame_state}
Texture2D<float>          PreviousDepth : register(t2);
Texture2D<uint>           EmaMotionMask : register(t3);
StructuredBuffer<uint4>   FrameRoiTransform : register(t4);
StructuredBuffer<uint4>   PreviousDepthSurfaceTransform : register(t5);
RWTexture2D<float>       OutputTexture : register(u0);
// This is a distinct ping-pong bank. It must never alias either transform SRV above.
RWStructuredBuffer<uint4> NextDepthSurfaceTransform : register(u1);

#include "include/depth_constants.hlsl"
#include "include/sbs_frame_roi_transform.hlsl"

SbsFrameRoiTransformData LoadPreviousDepthSurfaceTransform() {
    return SBS_FRAME_ROI_DECODE_RESOURCE(
        PreviousDepthSurfaceTransform);
}

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

void CopyCurrentTransformToNext() {
    [unroll]
    for (uint vector_index = 0u;
         vector_index < SBS_FRAME_ROI_VECTOR_COUNT;
         ++vector_index) {
        NextDepthSurfaceTransform[vector_index] =
            FrameRoiTransform[vector_index];
    }
}

void CopyPreviousTransformToNext() {
    [unroll]
    for (uint vector_index = 0u;
         vector_index < SBS_FRAME_ROI_VECTOR_COUNT;
         ++vector_index) {
        NextDepthSurfaceTransform[vector_index] =
            PreviousDepthSurfaceTransform[vector_index];
    }
}

void ClearNextTransform() {
    [unroll]
    for (uint vector_index = 0u;
         vector_index < SBS_FRAME_ROI_VECTOR_COUNT;
         ++vector_index) {
        NextDepthSurfaceTransform[vector_index] =
            0u.xxxx;
    }
}

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= target_w || DTid.y >= target_h)
        return;

    SbsFrameRoiTransformData transform = FrameRoiTransformLoad();
    SbsFrameRoiTransformData previous_surface_transform =
        LoadPreviousDepthSurfaceTransform();
    bool legacy_unbound_pair =
        FrameRoiDataUnboundZero(transform) &&
        FrameRoiDataUnboundZero(previous_surface_transform);

    // Preserve the historical shader exactly while transform resources are absent. In
    // particular, the new transform UAV may remain unbound during a staged rollout.
    if (legacy_unbound_pair) {
        uint legacy_idx =
            DTid.y * target_w + DTid.x;
        float4 legacy_scale = MinMaxEma[0];
        if (legacy_scale.w < 0.5f) {
            OutputTexture[DTid.xy] =
                PreviousDepth[DTid.xy];
            return;
        }
        float legacy_raw = InputBuffer[legacy_idx];
        if (isnan(legacy_raw) ||
            isinf(legacy_raw) ||
            legacy_raw < 0.0f) {
            OutputTexture[DTid.xy] =
                PreviousDepth[DTid.xy];
            return;
        }
        float2 legacy_mm = legacy_scale.xy;
        if (any(isnan(legacy_mm)) ||
            any(isinf(legacy_mm)) ||
            legacy_mm.y < legacy_mm.x) {
            legacy_mm = float2(0.0f, 1.0f);
        }
        float legacy_mapped = saturate(
            (max(legacy_raw, 0.0f) - legacy_mm.x) /
            max(legacy_mm.y - legacy_mm.x, 1e-6f));
        float legacy_old_depth =
            PreviousDepth[DTid.xy];
        if (isnan(legacy_old_depth) ||
            isinf(legacy_old_depth)) {
            legacy_old_depth = legacy_mapped;
        }
        float legacy_frame_alpha =
            legacy_scale.w > 1.5f ?
            1.0f :
            ema_alpha;
        float legacy_filtered = lerp(
            legacy_old_depth,
            legacy_mapped,
            legacy_frame_alpha);
        OutputTexture[DTid.xy] =
            EmaMotionMask[DTid.xy] != 0u ?
            lerp(
                legacy_filtered,
                legacy_mapped,
                ema_edge_strength) :
            legacy_filtered;
        return;
    }

    uint input_count;
    uint input_stride;
    InputBuffer.GetDimensions(input_count, input_stride);
    uint minmax_count;
    uint minmax_stride;
    MinMaxEma.GetDimensions(minmax_count, minmax_stride);
    uint previous_depth_width;
    uint previous_depth_height;
    PreviousDepth.GetDimensions(
        previous_depth_width,
        previous_depth_height);
    uint motion_width;
    uint motion_height;
    EmaMotionMask.GetDimensions(
        motion_width,
        motion_height);
    uint output_width;
    uint output_height;
    OutputTexture.GetDimensions(
        output_width,
        output_height);
    uint current_transform_vectors;
    uint current_transform_stride;
    FrameRoiTransform.GetDimensions(
        current_transform_vectors,
        current_transform_stride);
    uint previous_transform_vectors;
    uint previous_transform_stride;
    PreviousDepthSurfaceTransform.GetDimensions(
        previous_transform_vectors,
        previous_transform_stride);
    uint next_transform_vectors;
    uint next_transform_stride;
    NextDepthSurfaceTransform.GetDimensions(
        next_transform_vectors,
        next_transform_stride);

    uint plane;
    bool target_plane_safe =
        TryTargetPlane(plane);
    bool current_transform_valid =
        current_transform_vectors >= SBS_FRAME_ROI_VECTOR_COUNT &&
        current_transform_stride == 16u &&
        FrameRoiDataValid(transform);
    bool transform_dimensions_match =
        current_transform_valid &&
        all(FrameRoiDataModelDimensions(transform) ==
            uint2(target_w, target_h));
    bool previous_transform_valid =
        previous_transform_vectors >= SBS_FRAME_ROI_VECTOR_COUNT &&
        previous_transform_stride == 16u &&
        FrameRoiDataValid(previous_surface_transform);
    uint2 previous_model_size =
        FrameRoiDataModelDimensions(previous_surface_transform);
    bool previous_surface_valid =
        previous_transform_valid &&
        previous_model_size.x <= previous_depth_width &&
        previous_model_size.y <= previous_depth_height &&
        previous_model_size.x <= output_width &&
        previous_model_size.y <= output_height;
    bool next_transform_safe =
        next_transform_vectors >= SBS_FRAME_ROI_VECTOR_COUNT &&
        next_transform_stride == 16u;
    bool current_resources_safe =
        target_plane_safe &&
        input_stride == 4u &&
        input_count >= plane &&
        minmax_count >= 1u &&
        minmax_stride == 16u &&
        motion_width >= target_w &&
        motion_height >= target_h &&
        output_width >= target_w &&
        output_height >= target_h &&
        next_transform_safe;
    float4 scale =
        minmax_count >= 1u &&
        minmax_stride == 16u ?
        MinMaxEma[0] :
        0.0f.xxxx;
    bool promote_current =
        transform_dimensions_match &&
        current_resources_safe &&
        scale.w >= 0.5f;

    if (!promote_current) {
        if (DTid.x == 0u &&
            DTid.y == 0u &&
            next_transform_safe) {
            if (previous_surface_valid) {
                CopyPreviousTransformToNext();
            } else {
                ClearNextTransform();
            }
        }
        if (previous_surface_valid) {
            // The output texture already owns this previous surface. Copy explicitly when the
            // active extents match; otherwise leave the larger/smaller retained surface untouched.
            if (all(previous_model_size == uint2(target_w, target_h)) &&
                DTid.x < previous_depth_width &&
                DTid.y < previous_depth_height &&
                DTid.x < output_width &&
                DTid.y < output_height) {
                OutputTexture[DTid.xy] =
                    PreviousDepth[DTid.xy];
            }
        } else if (DTid.x < output_width &&
                   DTid.y < output_height) {
            OutputTexture[DTid.xy] = 0.0f;
        }
        return;
    }

    if (DTid.x == 0u && DTid.y == 0u) {
        CopyCurrentTransformToNext();
    }

    bool canonical_full_frame =
        FrameRoiDataCanonicalFullFrame(transform);
    uint4 accepted_bounds =
        FrameRoiDataAcceptedModelBounds(transform);
    bool same_surface_geometry =
        previous_surface_valid &&
        all(previous_model_size == uint2(target_w, target_h)) &&
        !FrameRoiDataGeometryReseedRequired(
            transform,
            previous_surface_transform);
    if (!canonical_full_frame &&
        !FrameRoiAcceptedBoundsContains(
            accepted_bounds,
            DTid.xy)) {
        // Only the exact focus ROI contributes geometry. The enclosing aspect crop exists to feed
        // DA-V2 real pixels without padding; reprojection keeps its exterior exactly flat.
        OutputTexture[DTid.xy] = 0.0f;
        return;
    }

    uint idx = DTid.y * target_w + DTid.x;
    float raw = InputBuffer[idx];
    if (isnan(raw) || isinf(raw) || raw < 0.0f) {
        // A missing prediction is not evidence for the far plane. Hold the last valid depth for
        // this texel. Across a geometry reseed the previous texel belongs to a different source
        // mapping, so it is not valid evidence and the inert value is used until a real sample.
        OutputTexture[DTid.xy] =
            scale.w > 1.5f || !same_surface_geometry ?
            0.0f :
            PreviousDepth[DTid.xy];
        return;
    }

    // The validated permanent order is range->pixel: normalize using the current P2/P98 bounds,
    // then temporally smooth the normalized depth.
    float2 mm = scale.xy;
    if (any(isnan(mm)) || any(isinf(mm)) || mm.y < mm.x) mm = float2(0.0f, 1.0f);
    float mapped = saturate((max(raw, 0.0f) - mm.x) / max(mm.y - mm.x, 1e-6f));
    float old_depth =
        same_surface_geometry ?
        PreviousDepth[DTid.xy] :
        mapped;
    if (isnan(old_depth) || isinf(old_depth)) old_depth = mapped;
    float frame_alpha =
        scale.w > 1.5f || !same_surface_geometry ?
        1.0f :
        ema_alpha;
    float filtered = lerp(old_depth, mapped, frame_alpha);
    OutputTexture[DTid.xy] = EmaMotionMask[DTid.xy] != 0u ?
                              lerp(filtered, mapped, ema_edge_strength) : filtered;
}
