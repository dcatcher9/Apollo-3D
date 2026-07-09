// 1-thread pass: resolve the tracked subject depth from the weighted histogram
// (depth_subject_hist_cs), EMA it for stability, and precompute everything the
// reprojection needs per pixel:
//   SubjectState[0] = { recenter_delta, subject_curve, subject_depth_ema, initialized }
// The reprojection's shaped path then evaluates BandCurve(saturate(d + recenter_delta))
// and subtracts subject_lock * subject_curve -- anchoring the subject at the screen
// plane. Because a global depth-scale drift moves the subject's parallax too, the
// subtraction also cancels normalization drift at the subject (VD3D's stability trick).
// Resets the histogram for the next frame's accumulation.

RWStructuredBuffer<uint>   SubjectHist  : register(u0);  // 256 bins from depth_subject_hist_cs
RWStructuredBuffer<float4> SubjectState : register(u1);  // {delta, scurve, subj_ema, init}

// Shared depth-pass cbuffer; subject_recenter (slot 15) is consumed here.
cbuffer Constants : register(b0) {
    uint target_w;
    uint target_h;
    uint is_hdr;
    float ema_alpha;
    float minmax_alpha;
    uint reduce_threads;
    uint output_transform;
    float depth_shift;
    float snap_ratio;
    float floor_frac;
    float floor_ref_alpha;
    float pct_lo;
    float pct_hi;
    float lock_frames;
    float locked_alpha;
    float subject_recenter;
};

#define NUM_BINS 256

// VD3D's near/mid/far Gaussian disparity bands, translated to Apollo's high=near depth
// (band centers mirrored) and normalized so the near band peaks at +1 (positive = pops
// out; divergence is the master gain). Amplitudes from the Bestv2 preset:
// fg -9*1.11, mg -3, bg +2.4*1.05 px in VD3D's negative=pop convention -> +1 / +0.300 /
// -0.252 here. MUST stay identical to BandCurve in sbs_reprojection_ps.hlsl (no #include
// support in the runtime-compiled shaders) and to band_curve in tools/warpsim/warpsim.cpp.
float BandCurve(float d) {
    float wn = exp(-0.5f * ((d - 0.85f) / 0.24f) * ((d - 0.85f) / 0.24f));
    float wm = exp(-0.5f * ((d - 0.50f) / 0.28f) * ((d - 0.50f) / 0.28f));
    float wf = exp(-0.5f * ((d - 0.15f) / 0.24f) * ((d - 0.15f) / 0.24f));
    return (wn * 1.0f + wm * 0.300f + wf * -0.252f) / (wn + wm + wf + 1e-6f);
}

[numthreads(1, 1, 1)]
void main() {
    // Total weighted votes.
    float total = 0.0f;
    for (uint b = 0; b < NUM_BINS; b++) {
        total += (float)SubjectHist[b];
    }

    float4 s = SubjectState[0];
    if (total > 0.5f) {
        // Weighted 35th percentile from the NEAR side (bin 255 = nearest): the subject is
        // usually among the nearer smooth regions but not the extreme foreground.
        float target = 0.35f * total;
        float cum = 0.0f;
        float subj_raw = 0.5f;
        for (int nb = NUM_BINS - 1; nb >= 0; nb--) {
            cum += (float)SubjectHist[nb];
            if (cum >= target) {
                subj_raw = ((float)nb + 0.5f) / (float)NUM_BINS;
                break;
            }
        }

        // Heavy EMA (VD3D SubjectDepthEMA alpha 0.95): the anchor must move slowly or the
        // whole scene appears to breathe with the subject estimate.
        float subj = (s.w > 0.5f) ? lerp(s.z, subj_raw, 0.05f) : subj_raw;

        float delta = (0.5f - subj) * subject_recenter;
        float scurve = BandCurve(saturate(subj + delta));
        s = float4(delta, scurve, subj, 1.0f);
    }
    // total == 0 (uninitialized depth): keep previous state.

    SubjectState[0] = s;

    for (uint rb = 0; rb < NUM_BINS; rb++) {
        SubjectHist[rb] = 0u;
    }
}
