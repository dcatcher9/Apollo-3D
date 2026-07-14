// 1-thread pass: resolve the tracked subject depth from the weighted histogram
// (depth_subject_hist_cs), EMA it for stability, and precompute everything the
// reprojection needs per pixel:
//   SubjectState[0] = { recenter_delta, scene_age, subject_depth_ema, initialized }
//   SubjectState[1] = { stretch_lo, stretch_inv_range, Bestv2 convergence EMA,
//                       adaptive pop ratio }
// The reprojection then evaluates the permanent Bestv2 pixel-calibrated field.
// Resets the histogram for the next frame's accumulation.

RWStructuredBuffer<uint>   SubjectHist  : register(u0);  // 256 weighted bins (subject estimate)
RWStructuredBuffer<float4> SubjectState : register(u1);  // [0..1], see header above
RWStructuredBuffer<uint>   PlainHist    : register(u2);  // 256 unweighted bins (stretch 5/95 pct)

#include "include/depth_constants.hlsl"

#define NUM_BINS 256

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
        float previous_scene_age = s.y;
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

        // EMA (Bestv2 SubjectDepthEMA alpha=0.80 => new-value weight 0.20; verified against a
        // real Bestv2 render log 2026-07-09). The anchor moves slowly so the scene doesn't
        // breathe with the subject estimate, but not so slowly it lags cuts/motion.
        bool initialized = s.w > 0.5f;
        float subj = initialized ? lerp(s.z, subj_raw, 0.20f) : subj_raw;

        // Disparity stretch (Bestv2 shape_depth_for_pop): rescale the [lo,hi] percentile band of
        // the (unweighted) depth distribution to full [0,1] so the mid-range uses the whole
        // parallax budget. lo=0, inv_range=1 when off -> the recenter path below is unchanged.
        float lo_val = 0.0f, inv_range = 1.0f;
        float ptotal = 0.0f;
        if (subject_stretch > 0.5f || adaptive_pop > 0.5f) {
            for (uint pb = 0; pb < NUM_BINS; pb++) ptotal += (float)PlainHist[pb];
        }
        if (subject_stretch > 0.5f) {
            if (ptotal > 0.5f) {
                float lo_c = 0.05f * ptotal, hi_c = 0.95f * ptotal;
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

        // Recenter is computed in stretched space: delta = (0.5 - subj_stretched)*recenter.
        float subj_str = saturate((subj - lo_val) * inv_range);
        float delta = (0.5f - subj_str) * subject_recenter;
        s = float4(delta, 0.0f, subj, 1.0f);
        // Bestv2 ConvergenceEMA(alpha=.90), driven by its low-near subject convention. Stored even
        // when the Apollo profile is selected so switching profiles cannot expose stale state.
        float conv_target = (1.0f - subj) * 0.006f;
        float conv_ema = initialized ? lerp(s1.z, conv_target, 0.10f) : conv_target;
        // Scene-adaptive pop is deliberately latched. Depth-edge density predicts warp risk, but
        // continuously responding to it makes disparity breathe even over a narrow strength band.
        // Choose once at startup and again only on a hard depth cut, where the content itself is
        // discontinuous. Between cuts the multiplier is bit-stable. The base remains the floor,
        // and the configured ceiling is never exceeded.
        float pop_ratio = max(s1.w, 1.0f);
        if (adaptive_pop > 0.5f && ptotal > 0.5f) {
            float edge_fraction = (float)PlainHist[NUM_BINS] / ptotal;
            float change_fraction = (float)PlainHist[NUM_BINS + 1] / ptotal;
            // Normalization settling can change 50-60% of depth texels on the first few frames.
            // The committed scene-cut clip reaches 66.8%, so 65% separates that cut from ordinary
            // startup/motion in the current core suite.
            float scene_age = initialized ? min(previous_scene_age + 1.0f, 65535.0f) : 0.0f;
            bool hard_cut = initialized && scene_age >= 8.0f && change_fraction >= 0.65f;
            if (!initialized || hard_cut) {
                // Full extra pop is safe for low-complexity depth fields (<=0.7% edge texels).
                // Fade to the base strength by 1.6%; the extended suite validated the 1.30
                // endpoint itself, so this classification only decides where the gain is useful.
                float confidence = 1.0f - smoothstep(0.007f, 0.016f, edge_fraction);
                pop_ratio = lerp(1.0f, max(adaptive_pop_max_ratio, 1.0f), confidence);
                scene_age = 0.0f;
            }
            s.y = scene_age;
        } else {
            pop_ratio = 1.0f;
        }
        s1 = float4(lo_val, inv_range, conv_ema, pop_ratio);
    }
    // total == 0 (uninitialized depth): keep previous state.

    SubjectState[0] = s;
    SubjectState[1] = s1;

    for (uint rb = 0; rb < NUM_BINS; rb++) {
        SubjectHist[rb] = 0u;
        PlainHist[rb] = 0u;
    }
    PlainHist[NUM_BINS] = 0u;
    PlainHist[NUM_BINS + 1] = 0u;
}
