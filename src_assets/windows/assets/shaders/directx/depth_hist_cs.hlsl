// 256-bin histogram of the raw disparity buffer, binned over THIS frame's raw min/max
// (already reduced by depth_minmax_cs into MinMaxRaw). depth_minmax_ema_cs then scans the
// histogram for the permanent robust P2/P98 bounds instead of using the raw min/max
// directly -- a handful of outlier pixels (DA-V2's near-spike tail) can otherwise stretch
// the normalization range and squeeze the whole scene's parallax. Fully GPU-resident, no
// CPU readback; dispatched only when percentile normalization is enabled.

StructuredBuffer<float>  InputBuffer : register(t0);
StructuredBuffer<uint4>  FrameRoiTransform : register(t1);
RWStructuredBuffer<uint> Histogram   : register(u0);  // 256 bins, reset by depth_minmax_ema_cs
RWByteAddressBuffer      MinMaxRaw   : register(u1);  // read-only: min bits, max bits, valid count

// Shared depth-pass cbuffer (slots 11-12 = the percentile bounds consumed by depth_minmax_ema_cs).
#include "include/depth_constants.hlsl"
#include "include/sbs_frame_roi_transform.hlsl"

#define GROUP_SIZE 256
#define NUM_BINS 256
groupshared uint g_hist[NUM_BINS];

bool TargetPlaneFitsInput(uint input_count, out uint count) {
    count = 0u;
    if (target_w == 0u ||
        target_h == 0u ||
        target_w > 0xffffffffu / target_h) {
        return false;
    }
    count = target_w * target_h;
    return count <= input_count && reduce_threads > 0u;
}

bool FrameRoiAcceptedBoundsContains(
    uint4 bounds,
    uint2 pixel)
{
    return all(pixel >= bounds.xy) &&
           all(pixel < bounds.zw);
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID) {
    g_hist[tid.x] = 0u;  // GROUP_SIZE == NUM_BINS: one bin per thread
    GroupMemoryBarrierWithGroupSync();

    uint input_count;
    uint input_stride;
    InputBuffer.GetDimensions(input_count, input_stride);
    uint histogram_count;
    uint histogram_stride;
    Histogram.GetDimensions(histogram_count, histogram_stride);
    uint accumulator_bytes;
    MinMaxRaw.GetDimensions(accumulator_bytes);
    uint transform_vectors;
    uint transform_stride;
    FrameRoiTransform.GetDimensions(
        transform_vectors,
        transform_stride);
    uint count;
    bool target_plane_safe =
        input_stride == 4u &&
        TargetPlaneFitsInput(input_count, count);
    bool histogram_safe =
        histogram_count >= NUM_BINS &&
        histogram_stride == 4u;

    SbsFrameRoiTransformData transform = FrameRoiTransformLoad();
    bool legacy_unbound =
        FrameRoiDataUnboundZero(transform);
    bool current_transform_valid =
        transform_vectors >= SBS_FRAME_ROI_VECTOR_COUNT &&
        transform_stride == 16u &&
        FrameRoiDataValid(transform);
    bool transform_dimensions_match =
        current_transform_valid &&
        all(FrameRoiDataModelDimensions(transform) ==
            uint2(target_w, target_h));
    bool transform_usable =
        legacy_unbound ||
        transform_dimensions_match;
    bool accumulator_safe =
        accumulator_bytes >=
            (current_transform_valid ? 16u : 12u);
    bool canonical_full_frame =
        legacy_unbound ||
        (transform_dimensions_match &&
         FrameRoiDataCanonicalFullFrame(transform));
    uint4 accepted_bounds =
        current_transform_valid ?
        FrameRoiDataAcceptedModelBounds(transform) :
        0u.xxxx;

    // Bin over this frame's raw range (from the preceding depth_minmax_cs dispatch).
    float vmin = asfloat(MinMaxRaw.Load(0));
    float vmax = asfloat(MinMaxRaw.Load(4));
    uint valid_count = MinMaxRaw.Load(8);
    bool valid_bounds =
        target_plane_safe &&
        histogram_safe &&
        accumulator_safe &&
        transform_usable &&
        valid_count > 0u &&
        !isnan(vmin) &&
        !isinf(vmin) &&
        !isnan(vmax) &&
        !isinf(vmax) &&
        vmax >= vmin;
    float inv_range = valid_bounds ? (float)NUM_BINS / max(vmax - vmin, 1e-12f) : 0.0f;

    [loop]
    for (uint idx = dtid.x;
         valid_bounds && idx < count;
         idx += reduce_threads) {
        uint2 pixel = uint2(idx % target_w, idx / target_w);
        if (!canonical_full_frame &&
            !FrameRoiAcceptedBoundsContains(
                accepted_bounds,
                pixel)) {
            continue;
        }
        float v = InputBuffer[idx];
        if (valid_bounds && !isnan(v) && !isinf(v) && v >= 0.0f) {
            uint bin = min((uint)((v - vmin) * inv_range), NUM_BINS - 1u);
            InterlockedAdd(g_hist[bin], 1u);
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if (histogram_safe && g_hist[tid.x] > 0u) {
        InterlockedAdd(Histogram[tid.x], g_hist[tid.x]);
    }
}
