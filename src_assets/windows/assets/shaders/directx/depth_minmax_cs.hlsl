// Parallel reduction: min/max of the raw disparity buffer (the previous frame's TensorRT
// output) into two uint bits. Fully GPU-resident so no CPU readback / flush is introduced
// (which would stall the pipeline and can starve DWM -> TDR).
//
// Depth Anything V2's relative output is affine-invariant (scale/shift arbitrary, drifting),
// so it must be min/max-normalized before use. For non-negative floats, asuint() preserves
// ordering, so InterlockedMin/InterlockedMax on the raw bit pattern yields the correct min/max.

StructuredBuffer<float> InputBuffer : register(t0);
StructuredBuffer<uint4> FrameRoiTransform : register(t1);
RWByteAddressBuffer     MinMaxOut   : register(u0);  // min/max, valid count, accepted-focus count

#include "include/depth_constants.hlsl"
#include "include/sbs_frame_roi_transform.hlsl"

#define GROUP_SIZE 256
groupshared uint g_min[GROUP_SIZE];
groupshared uint g_max[GROUP_SIZE];
groupshared uint g_valid[GROUP_SIZE];
groupshared uint g_accepted[GROUP_SIZE];

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
    uint input_count;
    uint input_stride;
    InputBuffer.GetDimensions(input_count, input_stride);
    uint accumulator_bytes;
    MinMaxOut.GetDimensions(accumulator_bytes);
    uint transform_vectors;
    uint transform_stride;
    FrameRoiTransform.GetDimensions(
        transform_vectors,
        transform_stride);

    uint count;
    bool target_plane_safe =
        input_stride == 4u &&
        TargetPlaneFitsInput(input_count, count);
    uint lmin = 0xFFFFFFFFu;
    uint lmax = 0u;
    uint lvalid = 0u;
    uint laccepted = 0u;
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
    bool reduction_usable =
        target_plane_safe &&
        transform_usable &&
        accumulator_safe;
    bool canonical_full_frame =
        legacy_unbound ||
        (transform_dimensions_match &&
         FrameRoiDataCanonicalFullFrame(transform));
    uint4 accepted_bounds =
        current_transform_valid ?
        FrameRoiDataAcceptedModelBounds(transform) :
        0u.xxxx;

    // Grid-stride: each thread folds in every reduce_threads-th element.
    [loop]
    for (uint idx = dtid.x;
         reduction_usable && idx < count;
         idx += reduce_threads) {
        uint2 pixel = uint2(idx % target_w, idx / target_w);
        if (!canonical_full_frame &&
            !FrameRoiAcceptedBoundsContains(
                accepted_bounds,
                pixel)) {
            continue;
        }
        // Offset 12 is an ABI extension used only once transform wiring is active. An all-zero
        // legacy binding keeps the historical 12-byte accumulator path byte-for-byte intact.
        laccepted += current_transform_valid ? 1u : 0u;
        float v = InputBuffer[idx];
        if (!isnan(v) && !isinf(v) && v >= 0.0f) {
            uint u = asuint(v);
            lmin = min(lmin, u);
            lmax = max(lmax, u);
            lvalid++;
        }
    }

    g_min[tid.x] = lmin;
    g_max[tid.x] = lmax;
    g_valid[tid.x] = lvalid;
    g_accepted[tid.x] = laccepted;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = GROUP_SIZE / 2; s > 0; s >>= 1) {
        if (tid.x < s) {
            g_min[tid.x] = min(g_min[tid.x], g_min[tid.x + s]);
            g_max[tid.x] = max(g_max[tid.x], g_max[tid.x + s]);
            g_valid[tid.x] += g_valid[tid.x + s];
            g_accepted[tid.x] += g_accepted[tid.x + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid.x == 0 && reduction_usable) {
        if (g_valid[0] > 0u) {
            MinMaxOut.InterlockedMin(0, g_min[0]);
            MinMaxOut.InterlockedMax(4, g_max[0]);
            MinMaxOut.InterlockedAdd(8, g_valid[0]);
        }
        if (g_accepted[0] > 0u) {
            MinMaxOut.InterlockedAdd(12, g_accepted[0]);
        }
    }
}
