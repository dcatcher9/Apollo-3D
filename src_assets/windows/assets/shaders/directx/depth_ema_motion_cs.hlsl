// Mark moving depth-transition bands where the per-pixel EMA should trust the current frame.
// Both current mapped depth and previous normalized depth are immutable SRVs.
StructuredBuffer<float>  InputBuffer : register(t0);
StructuredBuffer<float4> MinMaxEma   : register(t1);  // [0]={P2,P98,initialized,frame_state}
Texture2D<float>          PreviousDepth : register(t2);
StructuredBuffer<uint4>   FrameRoiTransform : register(t3);
StructuredBuffer<uint4>   PreviousDepthSurfaceTransform : register(t4);
RWTexture2D<uint>         MotionMask : register(u0);

#include "include/depth_constants.hlsl"
#include "include/sbs_frame_roi_transform.hlsl"

int2 ClampPixel(int2 p) {
    return clamp(p, int2(0, 0), int2((int)target_w - 1, (int)target_h - 1));
}

float CurrentDepth(int2 p) {
    p = ClampPixel(p);
    float2 mm = MinMaxEma[0].xy;
    float raw = InputBuffer[(uint)p.y * target_w + (uint)p.x];
    if (isnan(raw) || isinf(raw) || raw < 0.0f) {
        // Match buffer_to_tex_cs: missing current evidence holds history. Treating it as raw zero
        // would manufacture a far-depth edge and could snap adjacent valid texels through the mask.
        return PreviousDepth[p];
    }
    if (any(isnan(mm)) || any(isinf(mm)) || mm.y < mm.x) mm = float2(0.0f, 1.0f);
    return saturate((max(raw, 0.0f) - mm.x) / max(mm.y - mm.x, 1e-6f));
}

SbsFrameRoiTransformData LoadPreviousDepthSurfaceTransform() {
    return SBS_FRAME_ROI_DECODE_RESOURCE(
        PreviousDepthSurfaceTransform);
}

bool TargetPlaneFitsInput(uint input_count, out uint plane) {
    plane = 0u;
    if (target_w == 0u ||
        target_h == 0u ||
        target_w > 0xffffffffu / target_h) {
        return false;
    }
    plane = target_w * target_h;
    return plane <= input_count;
}

bool FrameRoiAcceptedBoundsContains(
    uint4 bounds,
    uint2 pixel)
{
    return all(pixel >= bounds.xy) &&
           all(pixel < bounds.zw);
}

float CurrentNeighborDepth(
    int2 p,
    float center_depth,
    uint4 accepted_bounds,
    bool canonical_full_frame)
{
    p = ClampPixel(p);
    if (!canonical_full_frame &&
        !FrameRoiAcceptedBoundsContains(
            accepted_bounds,
            (uint2)p)) {
        // The exact focus boundary is not a depth edge. Outside-focus model pixels belong to the
        // enclosing aspect crop only and must not contribute to pop-risk or motion decisions.
        return center_depth;
    }
    if (canonical_full_frame) {
        return CurrentDepth(p);
    }
    uint idx = (uint)p.y * target_w + (uint)p.x;
    float raw = InputBuffer[idx];
    if (isnan(raw) || isinf(raw) || raw < 0.0f) {
        return center_depth;
    }
    return CurrentDepth(p);
}

bool IsMovingEdge(
    int2 p,
    uint4 accepted_bounds,
    bool canonical_full_frame)
{
    p = ClampPixel(p);
    float current = CurrentDepth(p);
    float change = abs(current - PreviousDepth[p]);
    float gradient = 0.0f;
    gradient = max(gradient, abs(current - CurrentNeighborDepth(
        p + int2(-1, 0), current, accepted_bounds, canonical_full_frame)));
    gradient = max(gradient, abs(current - CurrentNeighborDepth(
        p + int2(1, 0), current, accepted_bounds, canonical_full_frame)));
    gradient = max(gradient, abs(current - CurrentNeighborDepth(
        p + int2(0, -1), current, accepted_bounds, canonical_full_frame)));
    gradient = max(gradient, abs(current - CurrentNeighborDepth(
        p + int2(0, 1), current, accepted_bounds, canonical_full_frame)));
    // ema_edge_gradient is specified in the same 434-reference-texel units as the subject and
    // adaptive-pop spatial thresholds. Keep the motion mask stable when a profile/native cap
    // resolves a 392- or 420-short-side grid.
    float reference_gradient = gradient * DepthReferenceTexelScale();
    return change >= ema_edge_change && reference_gradient >= ema_edge_gradient;
}

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= target_w || DTid.y >= target_h)
        return;

    uint input_count;
    uint input_stride;
    InputBuffer.GetDimensions(input_count, input_stride);
    uint minmax_count;
    uint minmax_stride;
    MinMaxEma.GetDimensions(minmax_count, minmax_stride);
    uint previous_width;
    uint previous_height;
    PreviousDepth.GetDimensions(previous_width, previous_height);
    uint mask_width;
    uint mask_height;
    MotionMask.GetDimensions(mask_width, mask_height);
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
    if (DTid.x >= mask_width || DTid.y >= mask_height) {
        return;
    }
    uint plane;
    bool current_resources_safe =
        input_stride == 4u &&
        TargetPlaneFitsInput(input_count, plane) &&
        minmax_count >= 1u &&
        minmax_stride == 16u &&
        previous_width >= target_w &&
        previous_height >= target_h &&
        mask_width >= target_w &&
        mask_height >= target_h;

    float frame_state =
        minmax_count >= 1u &&
        minmax_stride == 16u ?
        MinMaxEma[0].w :
        0.0f;
    if (!current_resources_safe ||
        frame_state < 0.5f ||
        frame_state > 1.5f) {
        MotionMask[DTid.xy] = 0u;
        return;
    }

    SbsFrameRoiTransformData transform = FrameRoiTransformLoad();
    SbsFrameRoiTransformData previous_surface_transform =
        LoadPreviousDepthSurfaceTransform();
    bool current_unbound =
        FrameRoiDataUnboundZero(transform);
    bool previous_unbound =
        FrameRoiDataUnboundZero(previous_surface_transform);
    bool legacy_unbound_pair =
        current_unbound &&
        previous_unbound;
    bool current_transform_valid =
        current_transform_vectors >= SBS_FRAME_ROI_VECTOR_COUNT &&
        current_transform_stride == 16u &&
        FrameRoiDataValid(transform);
    bool previous_transform_valid =
        previous_transform_vectors >= SBS_FRAME_ROI_VECTOR_COUNT &&
        previous_transform_stride == 16u &&
        FrameRoiDataValid(previous_surface_transform);
    bool transform_dimensions_match =
        current_transform_valid &&
        all(FrameRoiDataModelDimensions(transform) ==
            uint2(target_w, target_h));
    bool transform_usable =
        legacy_unbound_pair ||
        transform_dimensions_match;
    bool canonical_full_frame =
        legacy_unbound_pair ||
        (transform_dimensions_match &&
         FrameRoiDataCanonicalFullFrame(transform));
    uint4 accepted_bounds =
        current_transform_valid ?
        FrameRoiDataAcceptedModelBounds(transform) :
        0u.xxxx;
    bool same_surface_geometry =
        legacy_unbound_pair ||
        (transform_dimensions_match &&
         previous_transform_valid &&
         !FrameRoiDataGeometryReseedRequired(
             transform,
             previous_surface_transform));
    if (!transform_usable ||
        (!canonical_full_frame &&
         !FrameRoiAcceptedBoundsContains(
             accepted_bounds,
             DTid.xy))) {
        MotionMask[DTid.xy] = 0u;
        return;
    }
    uint idx = DTid.y * target_w + DTid.x;
    float raw = InputBuffer[idx];
    if (!same_surface_geometry ||
        (!canonical_full_frame &&
         (isnan(raw) || isinf(raw) || raw < 0.0f))) {
        // PreviousDepth is meaningful only in the source mapping recorded beside that surface.
        // A geometry change snaps in buffer_to_tex_cs; the motion stencil must not compare the
        // same model coordinate across unrelated crops.
        MotionMask[DTid.xy] = 0u;
        return;
    }

    MotionMask[DTid.xy] =
        IsMovingEdge(
            int2(DTid.xy),
            accepted_bounds,
            canonical_full_frame) ?
        1u :
        0u;
}
