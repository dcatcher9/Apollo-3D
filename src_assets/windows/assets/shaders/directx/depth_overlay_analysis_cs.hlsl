// Burned-in overlay preparation for Host SBS analysis.
//
// The full-resolution source mask is authored/detected before DAV2 downscaling. sanitize_main
// writes a private inference-only color texture; unmasked texels are copied exactly and masked
// texels receive a bounded nearest-background fill. exclusion_main independently max-pools the
// same 0/255 mask into the authenticated tensor grid. Neither output is ever displayed.

Texture2D<float4> SourceColor : register(t0);
Texture2D<float> TightDilatedMask : register(t1);
RWTexture2D<float4> SanitizedColor : register(u0);
RWTexture2D<uint> TensorExclusion : register(u1);

#include "include/depth_constants.hlsl"

#define OVERLAY_FILL_MAX_RADIUS 32

bool OverlaySourcePixelIsMasked(int2 position, uint2 source_size) {
    if (position.x < 0 || position.y < 0 ||
        position.x >= (int)source_size.x || position.y >= (int)source_size.y) {
        return true;
    }
    return TightDilatedMask.Load(int3(position, 0)) > 0.5f;
}

float4 OverlayLocalBackground(int2 position, uint2 source_size) {
    static const int2 directions[8] = {
        int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1),
        int2(-1, -1), int2(1, -1), int2(-1, 1), int2(1, 1)
    };
    float4 weighted_sum = 0.0f;
    float weight_sum = 0.0f;

    // Stop independently at the first unmasked sample in each direction. The detector's
    // dilation is responsible for covering glyph outlines; this pass only removes those known
    // edges and makes no claim to recover the hidden picture.
    [unroll]
    for (int direction_index = 0; direction_index < 8; ++direction_index) {
        [loop]
        for (int radius = 1; radius <= OVERLAY_FILL_MAX_RADIUS; ++radius) {
            int2 sample_position = position + directions[direction_index] * radius;
            if (sample_position.x < 0 || sample_position.y < 0 ||
                sample_position.x >= (int)source_size.x ||
                sample_position.y >= (int)source_size.y) {
                break;
            }
            if (!OverlaySourcePixelIsMasked(sample_position, source_size)) {
                float weight = rcp((float)radius);
                weighted_sum +=
                    SourceColor.Load(int3(sample_position, 0)) * weight;
                weight_sum += weight;
                break;
            }
        }
    }

    // A valid tight/dilated subtitle mask normally has background within a few pixels. A malformed
    // oversized component must still never leak its original overlay pixels into DAV2: use a
    // deterministic neutral capture-domain value, while the corresponding tensor cells remain
    // excluded from every statistic. Encoded SDR uses middle code value; linear SDR/scRGB use a
    // neutral diffuse gray in linear units.
    float neutral = color_mode == 0u ? 0.5f : 0.18f;
    return weight_sum > 0.0f ?
        weighted_sum / weight_sum : float4(neutral, neutral, neutral, 1.0f);
}

[numthreads(16, 16, 1)]
void sanitize_main(uint3 id : SV_DispatchThreadID) {
    uint source_width = 0u;
    uint source_height = 0u;
    SourceColor.GetDimensions(source_width, source_height);
    uint mask_width = 0u;
    uint mask_height = 0u;
    TightDilatedMask.GetDimensions(mask_width, mask_height);
    if (id.x >= source_width || id.y >= source_height ||
        source_width != mask_width || source_height != mask_height) {
        return;
    }

    int2 position = int2(id.xy);
    SanitizedColor[id.xy] = OverlaySourcePixelIsMasked(
        position, uint2(source_width, source_height)) ?
        OverlayLocalBackground(position, uint2(source_width, source_height)) :
        SourceColor.Load(int3(position, 0));
}

[numthreads(16, 16, 1)]
void exclusion_main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= target_w || id.y >= target_h) {
        return;
    }

    uint source_width = 0u;
    uint source_height = 0u;
    TightDilatedMask.GetDimensions(source_width, source_height);
    uint2 source_size = uint2(source_width, source_height);
    uint2 target_size = uint2(target_w, target_h);
    bool excluded = source_width == 0u || source_height == 0u;

    // Exact positive-overlap footprint used by rgb_to_nchw_cs.hlsl's authenticated area resize.
    // If any contributing full-resolution texel is masked, the tensor cell cannot vote in any
    // depth/color/ordinal reduction.
    float2 source_scale = float2(source_size) / float2(target_size);
    float2 source_lo = float2(id.xy) * source_scale;
    float2 source_hi = float2(id.xy + 1u) * source_scale;
    int2 first = max(int2(floor(source_lo)), int2(0, 0));
    int2 end = min(int2(ceil(source_hi)), int2(source_size));
    [loop]
    for (int source_y = first.y; source_y < end.y && !excluded; ++source_y) {
        [loop]
        for (int source_x = first.x; source_x < end.x; ++source_x) {
            float x_coverage =
                min(source_hi.x, (float)(source_x + 1)) -
                max(source_lo.x, (float)source_x);
            float y_coverage =
                min(source_hi.y, (float)(source_y + 1)) -
                max(source_lo.y, (float)source_y);
            if (x_coverage > 0.0f && y_coverage > 0.0f &&
                TightDilatedMask.Load(int3(source_x, source_y, 0)) > 0.5f) {
                excluded = true;
                break;
            }
        }
    }
    TensorExclusion[id.xy] = excluded ? 1u : 0u;
}
