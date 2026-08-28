// Standalone frozen ZipDepth convex-2x reconstruction. This shader is authenticated for the
// experimental logits boundary but is not part of the live estimator or production prewarm set.
//
// CoarseDepth is contiguous FP32 [H,W]. ConvexLogits is contiguous FP32 NCHW [36,H,W], where
// channel = neighbor * 4 + subpixel. Neighbors and 2x2 subpixels are both row-major.
cbuffer Convex2xConstants : register(b0) {
    uint coarse_width;
    uint coarse_height;
    uint2 convex2x_reserved;
};

StructuredBuffer<float> CoarseDepth  : register(t0);
StructuredBuffer<float> ConvexLogits : register(t1);
RWTexture2D<float>       RefinedDepth : register(u0);

static const int2 neighbor_offsets[9] = {
    int2(-1, -1), int2(0, -1), int2(1, -1),
    int2(-1,  0), int2(0,  0), int2(1,  0),
    int2(-1,  1), int2(0,  1), int2(1,  1)
};

uint CoarseIndex(uint2 cell) {
    return cell.y * coarse_width + cell.x;
}

float LoadLogit(uint2 cell, uint neighbor, uint subpixel) {
    const uint channel = neighbor * 4u + subpixel;
    return ConvexLogits[channel * coarse_width * coarse_height + CoarseIndex(cell)];
}

float LoadReplicatedNeighbor(uint2 cell, uint neighbor) {
    const int2 upper = int2((int)coarse_width - 1, (int)coarse_height - 1);
    const int2 sample_cell = clamp(int2(cell) + neighbor_offsets[neighbor], int2(0, 0), upper);
    return CoarseDepth[(uint)sample_cell.y * coarse_width + (uint)sample_cell.x];
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= coarse_width || DTid.y >= coarse_height)
        return;

    const uint2 cell = DTid.xy;
    [unroll]
    for (uint subpixel = 0u; subpixel < 4u; ++subpixel) {
        // Subtracting the per-subpixel maximum is the stable softmax transform. The denominator
        // is strictly positive for the finite logits admitted by the experimental boundary.
        float maximum_logit = LoadLogit(cell, 0u, subpixel);
        [unroll]
        for (uint neighbor = 1u; neighbor < 9u; ++neighbor) {
            maximum_logit = max(maximum_logit, LoadLogit(cell, neighbor, subpixel));
        }

        float weighted_depth = 0.0f;
        float weight_sum = 0.0f;
        [unroll]
        for (uint neighbor = 0u; neighbor < 9u; ++neighbor) {
            const float weight = exp(LoadLogit(cell, neighbor, subpixel) - maximum_logit);
            weighted_depth += weight * LoadReplicatedNeighbor(cell, neighbor);
            weight_sum += weight;
        }

        const uint2 subpixel_offset = uint2(subpixel & 1u, subpixel >> 1u);
        RefinedDepth[cell * 2u + subpixel_offset] = max(weighted_depth / weight_sum, 0.0f);
    }
}
