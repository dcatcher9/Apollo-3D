// Weighted depth histogram for Bestv2-derived subject tracking:
// every texel of the NORMALIZED depth map votes for its depth bin with weight
//   center Gaussian (favor the frame center, where the subject usually is)
//   x smoothness (downweight depth edges, so silhouette ramps don't skew the estimate).
// depth_subject_resolve_cs then takes the weighted 35th-percentile-from-NEAR as the
// subject depth. Runs at depth cadence on the low-res depth grid -- negligible cost.

Texture2D<float>         DepthTexture : register(t0);  // normalized depth, high = near
RWStructuredBuffer<uint> SubjectHist  : register(u0);  // 256 bins, weight in 1/1024 units
RWStructuredBuffer<uint> PlainHist    : register(u1);  // bins + edge/depth/ordinal/raw-RGB counts
Texture2D<float>         PreviousDepth : register(t1);
StructuredBuffer<float>  CurrentModelInput : register(t2);  // completed frame, NCHW ImageNet
StructuredBuffer<float>  PreviousModelInput : register(t3);
StructuredBuffer<float4> MinMaxEma : register(t4);  // w = current-frame validity
StructuredBuffer<float>  CurrentAppearanceOrdinal : register(t5);  // pre-tone-map point maxRGB
StructuredBuffer<float>  PreviousAppearanceOrdinal : register(t6);

// Shared depth-pass cbuffer (only target_w/target_h are used here).
#include "include/depth_constants.hlsl"

#define NUM_BINS 256
#define STRUCTURE_TILE_WIDTH 18
#define STRUCTURE_TILE_TEXELS (STRUCTURE_TILE_WIDTH * STRUCTURE_TILE_WIDTH)
static const int2 STRUCTURE_ORDINAL_OFFSETS[5] = {
    int2(0, 0), int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)
};

groupshared uint g_hist[NUM_BINS];
groupshared uint g_plain[NUM_BINS];
groupshared uint g_edge_count;
groupshared uint g_change_count;
groupshared uint g_structural_change_count;
groupshared uint g_raw_rgb_change_count;
// A 16x16 group plus a one-pixel halo of the pre-tone-map point-sampled ordinal signal.
groupshared float g_current_appearance_ordinal[STRUCTURE_TILE_TEXELS];
groupshared float g_previous_appearance_ordinal[STRUCTURE_TILE_TEXELS];

float3 CurrentModelColor(uint2 p) {
    uint plane = target_w * target_h;
    uint idx = p.y * target_w + p.x;
    return float3(
        CurrentModelInput[idx] * 0.229f + 0.485f,
        CurrentModelInput[idx + plane] * 0.224f + 0.456f,
        CurrentModelInput[idx + 2u * plane] * 0.225f + 0.406f);
}

float3 PreviousModelColor(uint2 p) {
    uint plane = target_w * target_h;
    uint idx = p.y * target_w + p.x;
    return float3(
        PreviousModelInput[idx] * 0.229f + 0.485f,
        PreviousModelInput[idx + plane] * 0.224f + 0.456f,
        PreviousModelInput[idx + 2u * plane] * 0.225f + 0.406f);
}

[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID,
          uint3 gid : SV_GroupID) {
    uint lin = tid.y * 16 + tid.x;  // 256 threads/group: one shared bin each
    g_hist[lin] = 0u;
    g_plain[lin] = 0u;
    if (lin == 0u) {
        g_edge_count = 0u;
        g_change_count = 0u;
        g_structural_change_count = 0u;
        g_raw_rgb_change_count = 0u;
    }

    // Cooperatively load the 18x18 ordinal tile. The final 68 halo samples are handled by the
    // first 68 threads' second loop iteration; clamping exactly matches the depth-gradient rule.
    for (uint tile_idx = lin; tile_idx < STRUCTURE_TILE_TEXELS; tile_idx += 256u) {
        uint tile_x = tile_idx % STRUCTURE_TILE_WIDTH;
        uint tile_y = tile_idx / STRUCTURE_TILE_WIDTH;
        int source_x = clamp((int)(gid.x * 16u + tile_x) - 1, 0, (int)target_w - 1);
        int source_y = clamp((int)(gid.y * 16u + tile_y) - 1, 0, (int)target_h - 1);
        uint source_index = (uint)source_y * target_w + (uint)source_x;
        g_current_appearance_ordinal[tile_idx] =
            CurrentAppearanceOrdinal[source_index];
        g_previous_appearance_ordinal[tile_idx] =
            PreviousAppearanceOrdinal[source_index];
    }
    GroupMemoryBarrierWithGroupSync();

    if (dtid.x < target_w && dtid.y < target_h && MinMaxEma[0].w > 0.5f) {
        float d = DepthTexture[dtid.xy];

        // Forward-difference gradient (clamped at the far edges).
        uint xn = min(dtid.x + 1, target_w - 1);
        uint yn = min(dtid.y + 1, target_h - 1);
        float gx = DepthTexture[uint2(xn, dtid.y)] - d;
        float gy = DepthTexture[uint2(dtid.x, yn)] - d;
        float grad = sqrt(gx * gx + gy * gy);
        // Express every spatial-gradient threshold on the 434-short-side calibration grid.
        // The scale comes from target_w/target_h, which are the resolved TensorRT dimensions
        // after patch alignment and profile/native caps, rather than the requested config value.
        float reference_texel_scale = DepthReferenceTexelScale();
        float reference_grad = grad * reference_texel_scale;
        // Fixed controller thresholds: changing the independent EMA ablation knobs must not
        // silently alter scene classification.
        if (reference_grad >= EDGE_GRADIENT_THRESHOLD) {
            // Weight each edge texel by how far past the threshold it is, instead of counting it
            // once. Warp stress scales with the disparity STEP a silhouette produces, so a few
            // violent discontinuities outrank many gentle ones -- a distinction a threshold count
            // cannot make, and the reason edge-dense-but-soft scenes were classified alongside
            // sharp ones. Capped so a handful of extreme texels cannot dominate the frame. Scale
            // BOTH the linear weight and the cap: a one-texel discontinuity occupies a fraction
            // inverse to grid resolution, including after the linear term saturates.
            float weight = min(
                reference_grad * (1.0f / EDGE_GRADIENT_THRESHOLD),
                EDGE_WEIGHT_MAX * reference_texel_scale);
            InterlockedAdd(g_edge_count, (uint)(weight * EDGE_WEIGHT_SCALE + 0.5f));
        }
        if (abs(d - PreviousDepth[dtid.xy]) >= 0.05f) {
            InterlockedAdd(g_change_count, 1u);
        }

        // Broad replacement is measured on the exact display-referred model inputs that produced
        // the matched depths. Structural order comes from the separately preserved capture-domain
        // point maxRGB signal: it precedes HDR Reinhard and avoids nonlinear-then-bilinear mixing.
        // Under an identical monotone curve on every channel, pair order can only stay ordered or
        // collapse to a rejected tie. Codec noise, color matrices, and local tone mapping remain
        // outside that exposure model. Depth geometry is still the shot-reset authority.
        uint tile_center = (tid.y + 1u) * STRUCTURE_TILE_WIDTH + tid.x + 1u;
        float3 current_color = CurrentModelColor(dtid.xy);
        float3 previous_color = PreviousModelColor(dtid.xy);
        float3 raw_rgb_delta = abs(current_color - previous_color);
        if (max(raw_rgb_delta.r, max(raw_rgb_delta.g, raw_rgb_delta.b)) >=
            RAW_RGB_PIXEL_DELTA) {
            InterlockedAdd(g_raw_rgb_change_count, 1u);
        }

        float current_samples[5];
        float previous_samples[5];
        [unroll]
        for (int sample_index = 0; sample_index < 5; ++sample_index) {
            int2 offset = STRUCTURE_ORDINAL_OFFSETS[sample_index];
            int tile_index =
                (int)tile_center + offset.y * STRUCTURE_TILE_WIDTH + offset.x;
            current_samples[sample_index] =
                g_current_appearance_ordinal[tile_index];
            previous_samples[sample_index] =
                g_previous_appearance_ordinal[tile_index];
        }
        uint common_comparisons = 0u;
        uint ordering_flips = 0u;
        [unroll]
        for (int first = 0; first < 4; ++first) {
            [unroll]
            for (int second = first + 1; second < 5; ++second) {
                float current_delta = current_samples[first] - current_samples[second];
                float previous_delta = previous_samples[first] - previous_samples[second];
                bool common_reliable =
                    abs(current_delta) >= STRUCTURAL_ORDINAL_CONTRAST_FLOOR &&
                    abs(previous_delta) >= STRUCTURAL_ORDINAL_CONTRAST_FLOOR;
                if (common_reliable) {
                    ++common_comparisons;
                    if ((current_delta < 0.0f) != (previous_delta < 0.0f)) {
                        ++ordering_flips;
                    }
                }
            }
        }
        bool ordinal_changed =
            common_comparisons >= STRUCTURAL_ORDINAL_MIN_COMMON &&
            ordering_flips >= STRUCTURAL_ORDINAL_MIN_FLIPS &&
            ordering_flips * 2u >= common_comparisons;
        if (ordinal_changed) {
            InterlockedAdd(g_structural_change_count, 1u);
        }

        // Subject smoothness uses the same reference-texel gradient. Otherwise changing the depth
        // grid would silently change which silhouette ramps can vote for recenter/stretch even if
        // adaptive-pop risk itself were normalized.
        // smooth_w = 1 - sigmoid(10 * (reference_grad - 0.025)).
        float smooth_w =
            1.0f - 1.0f / (1.0f + exp(-10.0f * (reference_grad - 0.025f)));

        // Center Gaussian in [-1,1] frame coords (Bestv2 sigmas: y 0.55, x 0.70).
        float nx = (float)dtid.x / (float)max(target_w - 1, 1u) * 2.0f - 1.0f;
        float ny = (float)dtid.y / (float)max(target_h - 1, 1u) * 2.0f - 1.0f;
        float center_w = exp(-0.5f * ((ny / 0.55f) * (ny / 0.55f) + (nx / 0.70f) * (nx / 0.70f)));

        float w = center_w * smooth_w;
        uint bin = min((uint)(saturate(d) * (float)NUM_BINS), NUM_BINS - 1u);
        InterlockedAdd(g_hist[bin], (uint)(w * 1024.0f + 0.5f));
        InterlockedAdd(g_plain[bin], 1u);  // unweighted, for the stretch 5/95 percentiles
    }

    GroupMemoryBarrierWithGroupSync();
    if (g_hist[lin] > 0u) {
        InterlockedAdd(SubjectHist[lin], g_hist[lin]);
    }
    if (g_plain[lin] > 0u) {
        InterlockedAdd(PlainHist[lin], g_plain[lin]);
    }
    if (lin == 0u) {
        InterlockedAdd(PlainHist[NUM_BINS], g_edge_count);
        InterlockedAdd(PlainHist[NUM_BINS + 1], g_change_count);
        InterlockedAdd(PlainHist[NUM_BINS + 2], g_structural_change_count);
        InterlockedAdd(PlainHist[NUM_BINS + 3], g_raw_rgb_change_count);
    }
}
