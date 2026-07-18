// Weighted depth histogram for Bestv2-derived subject tracking:
// every texel of the NORMALIZED depth map votes for its depth bin with weight
//   center Gaussian (favor the frame center, where the subject usually is)
//   x smoothness (downweight depth edges, so silhouette ramps don't skew the estimate).
// depth_subject_resolve_cs then takes the weighted 35th-percentile-from-NEAR as the
// subject depth. Runs at depth cadence on the low-res depth grid -- negligible cost.

Texture2D<float>         DepthTexture : register(t0);  // normalized depth, high = near
RWStructuredBuffer<uint> SubjectHist  : register(u0);  // 256 bins, weight in 1/1024 units
RWStructuredBuffer<uint> PlainHist    : register(u1);  // bins + edge/depth-change/color-change counts
Texture2D<float>         PreviousDepth : register(t1);
StructuredBuffer<float>  CurrentModelInput : register(t2);  // completed frame, NCHW ImageNet
StructuredBuffer<float>  PreviousModelInput : register(t3);
StructuredBuffer<float4> MinMaxEma : register(t4);  // w = current-frame validity

// Shared depth-pass cbuffer (only target_w/target_h are used here).
#include "include/depth_constants.hlsl"

#define NUM_BINS 256
groupshared uint g_hist[NUM_BINS];
groupshared uint g_plain[NUM_BINS];
groupshared uint g_edge_count;
groupshared uint g_change_count;
groupshared uint g_color_change_count;

[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID) {
    uint lin = tid.y * 16 + tid.x;  // 256 threads/group: one shared bin each
    g_hist[lin] = 0u;
    g_plain[lin] = 0u;
    if (lin == 0u) {
        g_edge_count = 0u;
        g_change_count = 0u;
        g_color_change_count = 0u;
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
        // Fixed controller thresholds: changing the independent EMA ablation knobs must not
        // silently alter scene classification.
        if (grad >= 0.02f) {
            InterlockedAdd(g_edge_count, 1u);
        }
        if (abs(d - PreviousDepth[dtid.xy]) >= 0.05f) {
            InterlockedAdd(g_change_count, 1u);
        }

        // Detect hard cuts whose depth field is coincidentally similar. Compare the exact
        // production-preprocessed model input that produced this depth against the prior one;
        // reconstructing sRGB makes the threshold independent of ImageNet channel scales.
        uint channel_stride = target_w * target_h;
        uint input_idx = dtid.y * target_w + dtid.x;
        float3 current_color = float3(
            CurrentModelInput[input_idx] * 0.229f + 0.485f,
            CurrentModelInput[input_idx + channel_stride] * 0.224f + 0.456f,
            CurrentModelInput[input_idx + 2u * channel_stride] * 0.225f + 0.406f);
        float3 previous_color = float3(
            PreviousModelInput[input_idx] * 0.229f + 0.485f,
            PreviousModelInput[input_idx + channel_stride] * 0.224f + 0.456f,
            PreviousModelInput[input_idx + 2u * channel_stride] * 0.225f + 0.406f);
        if (max(max(abs(current_color.r - previous_color.r),
                    abs(current_color.g - previous_color.g)),
                abs(current_color.b - previous_color.b)) >= 0.20f) {
            InterlockedAdd(g_color_change_count, 1u);
        }

        // smooth_w = 1 - sigmoid(10 * (grad - 0.025)): flat regions vote, edges mostly don't.
        float smooth_w = 1.0f - 1.0f / (1.0f + exp(-10.0f * (grad - 0.025f)));

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
        InterlockedAdd(PlainHist[NUM_BINS + 2], g_color_change_count);
    }
}
