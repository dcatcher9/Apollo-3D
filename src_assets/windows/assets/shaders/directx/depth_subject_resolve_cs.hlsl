// 1-thread pass: resolve the tracked subject depth from the weighted histogram
// (depth_subject_hist_cs), EMA it for stability, and precompute everything the
// reprojection needs per pixel:
//   SubjectState[0] = { recenter_delta, subject_curve, subject_depth_ema, initialized }
//   SubjectState[1] = { stretch_lo, stretch_inv_range, Bestv2 convergence EMA, 0 }
//   SubjectState[2] = { exact plane-lock weighted mean raw shift, initialized, 0, 0 }
// The reprojection's shaped path then evaluates BandCurve(saturate(d + recenter_delta))
// and subtracts subject_lock * subject_curve -- anchoring the subject at the screen
// plane. Because a global depth-scale drift moves the subject's parallax too, the
// subtraction also cancels normalization drift at the subject (VD3D's stability trick).
// Resets the histogram for the next frame's accumulation.

RWStructuredBuffer<uint>   SubjectHist  : register(u0);  // 256 weighted bins (subject estimate)
RWStructuredBuffer<float4> SubjectState : register(u1);  // [0..2], see header above
RWStructuredBuffer<uint>   PlainHist    : register(u2);  // 256 unweighted bins (stretch 5/95 pct)

#include "include/depth_constants.hlsl"

#define NUM_BINS 256

// BandCurve() -- the near/mid/far disparity shaping profile, shared with
// sbs_reprojection_ps.hlsl (keep tools/warpsim/warpsim.cpp band_curve in sync).
#include "include/band_curve.hlsl"

[numthreads(1, 1, 1)]
void main() {
    // Total weighted votes.
    float total = 0.0f;
    for (uint b = 0; b < NUM_BINS; b++) {
        total += (float)SubjectHist[b];
    }

    float4 s = SubjectState[0];
    float4 s1 = SubjectState[1];
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

        // EMA (VD3D SubjectDepthEMA alpha=0.80 => new-value weight 0.20; verified against a
        // real Bestv2 render log 2026-07-09). The anchor moves slowly so the scene doesn't
        // breathe with the subject estimate, but not so slowly it lags cuts/motion.
        bool initialized = s.w > 0.5f;
        float subj = initialized ? lerp(s.z, subj_raw, 0.20f) : subj_raw;

        // Disparity stretch (VD3D shape_depth_for_pop): rescale the [lo,hi] percentile band of
        // the (unweighted) depth distribution to full [0,1] so the mid-range uses the whole
        // parallax budget. lo=0, inv_range=1 when off -> the recenter path below is unchanged.
        float lo_val = 0.0f, inv_range = 1.0f;
        if (subject_stretch > 0.5f) {
            float ptotal = 0.0f;
            for (uint pb = 0; pb < NUM_BINS; pb++) ptotal += (float)PlainHist[pb];
            if (ptotal > 0.5f) {
                float lo_c = stretch_lo_pct * ptotal, hi_c = stretch_hi_pct * ptotal;
                float pc = 0.0f, hv = 1.0f;
                bool got_lo = false;
                for (uint qb = 0; qb < NUM_BINS; qb++) {
                    pc += (float)PlainHist[qb];
                    if (!got_lo && pc >= lo_c) { lo_val = ((float)qb + 0.5f) / (float)NUM_BINS; got_lo = true; }
                    if (pc >= hi_c) { hv = ((float)qb + 0.5f) / (float)NUM_BINS; break; }
                }
                inv_range = 1.0f / max(hv - lo_val, 1e-4f);
            }
        }

        // Recenter is computed in STRETCHED space (VD3D: delta = (0.5 - subj_stretched)*recenter).
        float subj_str = saturate((subj - lo_val) * inv_range);
        float delta = (0.5f - subj_str) * subject_recenter;
        float scurve = BandCurve(saturate(subj_str + delta));
        s = float4(delta, scurve, subj, 1.0f);
        // VD3D ConvergenceEMA(alpha=.90), driven by its low-near subject convention. Stored even
        // when the Apollo profile is selected so switching profiles cannot expose stale state.
        float conv_target = (1.0f - subj) * 0.006f;
        float conv_ema = initialized ? lerp(s1.z, conv_target, 0.10f) : conv_target;
        s1 = float4(lo_val, inv_range, conv_ema, 0.0f);
    }
    // total == 0 (uninitialized depth): keep previous state.

    SubjectState[0] = s;
    SubjectState[1] = s1;

    for (uint rb = 0; rb < NUM_BINS; rb++) {
        SubjectHist[rb] = 0u;
        PlainHist[rb] = 0u;
    }
}
