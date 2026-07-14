// Weighted depth histogram for Bestv2-derived subject tracking:
// every texel of the NORMALIZED depth map votes for its depth bin with weight
//   center Gaussian (favor the frame center, where the subject usually is)
//   x smoothness (downweight depth edges, so silhouette ramps don't skew the estimate).
// depth_subject_resolve_cs then takes the weighted 35th-percentile-from-NEAR as the
// subject depth. Runs at depth cadence on the low-res depth grid -- negligible cost.

Texture2D<float>         DepthTexture : register(t0);  // normalized depth, high = near
RWStructuredBuffer<uint> SubjectHist  : register(u0);  // 256 bins, weight in 1/1024 units
RWStructuredBuffer<uint> PlainHist    : register(u1);  // 256 bins + edge/change counts at 256/257
Texture2D<float>         PreviousDepth : register(t1);

// Shared depth-pass cbuffer (only target_w/target_h are used here).
#include "include/depth_constants.hlsl"

#define NUM_BINS 256
groupshared uint g_hist[NUM_BINS];
groupshared uint g_plain[NUM_BINS];
groupshared uint g_edge_count;
groupshared uint g_change_count;

[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID) {
    uint lin = tid.y * 16 + tid.x;  // 256 threads/group: one shared bin each
    g_hist[lin] = 0u;
    g_plain[lin] = 0u;
    if (lin == 0u) {
        g_edge_count = 0u;
        g_change_count = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    if (dtid.x < target_w && dtid.y < target_h) {
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
    }
}
