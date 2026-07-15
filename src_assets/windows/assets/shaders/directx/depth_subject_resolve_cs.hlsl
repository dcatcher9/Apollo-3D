// 1-thread pass: resolve the tracked subject depth from the weighted histogram
// (depth_subject_hist_cs), EMA it for stability, and precompute everything the
// reprojection needs per pixel:
//   SubjectState[0] = { recenter_delta, scene_age, subject_depth_ema, initialized }
//   SubjectState[1] = { stretch_lo, stretch_inv_range, Bestv2 convergence EMA,
//                       adaptive pop ratio }
//   SubjectState[2] = { shot-latched zero-plane anchor shift in source pixels, valid, mode,
//                       confidence-gated artistic camera multiplier (0 = disabled) }
// The reprojection then evaluates the permanent Bestv2 pixel-calibrated field.
// Resets the histogram for the next frame's accumulation.

RWStructuredBuffer<uint>   SubjectHist  : register(u0);  // 256 weighted bins (subject estimate)
RWStructuredBuffer<float4> SubjectState : register(u1);  // [0..2], see header above
RWStructuredBuffer<uint>   PlainHist    : register(u2);  // 256 bins + edge/change counters
StructuredBuffer<float>    ArtisticGlobal : register(t0);  // [safe scale ceiling, confidence], optional

#include "include/depth_constants.hlsl"
#include "include/bestv2_curve.hlsl"

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
    float4 s2 = SubjectState[2];
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
        float background_val = 0.25f, median_val = 0.5f;
        float ptotal = 0.0f;
        if (subject_stretch > 0.5f || adaptive_pop > 0.5f || zero_plane_mode > 0.5f ||
            artistic_policy > 0.5f) {
            for (uint pb = 0; pb < NUM_BINS; pb++) ptotal += (float)PlainHist[pb];
        }
        if (ptotal > 0.5f && (subject_stretch > 0.5f || zero_plane_mode > 0.5f)) {
            float lo_c = 0.05f * ptotal, bg_c = 0.25f * ptotal;
            float med_c = 0.50f * ptotal, hi_c = 0.95f * ptotal;
            float pc = 0.0f, hv = 1.0f;
            bool got_lo = false, got_bg = false, got_med = false;
            for (uint qb = 0; qb < NUM_BINS; qb++) {
                pc += (float)PlainHist[qb];
                float qv = ((float)qb + 0.5f) / (float)NUM_BINS;
                if (!got_lo && pc >= lo_c) { lo_val = qv; got_lo = true; }
                if (!got_bg && pc >= bg_c) { background_val = qv; got_bg = true; }
                if (!got_med && pc >= med_c) { median_val = qv; got_med = true; }
                if (pc >= hi_c) { hv = qv; break; }
            }
            if (subject_stretch > 0.5f) {
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
        // Scene camera parameters are deliberately latched. Continuously responding to depth
        // statistics makes disparity breathe. Choose once at startup and again only on a hard
        // depth cut, where the content itself is already discontinuous.
        bool scene_control = adaptive_pop > 0.5f || zero_plane_mode > 0.5f ||
                             artistic_policy > 0.5f;
        float scene_age = initialized ? min(previous_scene_age + 1.0f, 65535.0f) : 0.0f;
        float change_fraction = ptotal > 0.5f ? (float)PlainHist[NUM_BINS + 1] / ptotal : 0.0f;
        // Normalization settling can change 50-60% of depth texels on the first few frames.
        // The committed scene-cut clip reaches 66.8%, so 65% separates that cut from ordinary
        // startup/motion in the current core suite.
        bool hard_cut = scene_control && initialized && scene_age >= 8.0f &&
                        change_fraction >= 0.65f;
        if (!initialized || hard_cut) {
            scene_age = 0.0f;
        }

        // Depth-edge density predicts warp risk. Between cuts the multiplier remains bit-stable;
        // the base is the floor and the configured ceiling is never exceeded.
        float pop_ratio = max(s1.w, 1.0f);
        if (adaptive_pop > 0.5f && ptotal > 0.5f) {
            float edge_fraction = (float)PlainHist[NUM_BINS] / ptotal;
            if (!initialized || hard_cut) {
                // Full extra pop is safe for low-complexity depth fields (<=0.7% edge texels).
                // Fade to the base strength by 1.6%; the extended suite validated the 1.30
                // endpoint itself, so this classification only decides where the gain is useful.
                float confidence = 1.0f - smoothstep(0.007f, 0.016f, edge_fraction);
                pop_ratio = lerp(1.0f, max(adaptive_pop_max_ratio, 1.0f), confidence);
            }
        } else {
            pop_ratio = 1.0f;
        }
        s.y = scene_control && ptotal > 0.5f ? scene_age : 0.0f;

        // Explicit artistic zero plane. Resolve the chosen raw anchor through this frame's
        // stretch/recenter/Bestv2 curve and latch the resulting source-pixel shift. Storing the
        // final shift rather than raw depth prevents later percentile/recenter motion from making
        // convergence breathe. Subject, median, and far/mid-background correspond to the paper's
        // shot-level affine offset t.
        float zero_anchor_shift = s2.x;
        float zero_valid = s2.y;
        if (zero_plane_mode > 0.5f && ptotal > 0.5f) {
            if (!initialized || hard_cut || zero_valid < 0.5f) {
                float zero_anchor_depth = zero_plane_mode < 1.5f ? subj :
                                          zero_plane_mode < 2.5f ? median_val : background_val;
                float zero_anchor_shaped = subject_stretch > 0.5f ?
                    saturate((zero_anchor_depth - lo_val) * inv_range) : zero_anchor_depth;
                zero_anchor_shaped = saturate(zero_anchor_shaped + delta);
                zero_anchor_shift = Bestv2RawShiftPxFast(zero_anchor_shaped);
                zero_valid = 1.0f;
            }
        } else {
            zero_valid = 0.0f;
        }
        // The learned output is the highest render-validated safe multiplier of Apollo's complete
        // baseline disparity, including its existing adaptive-pop decision. The configured style
        // consumes none, half, or all of that available gain. Confidence can reject the learned
        // ceiling but can never raise it. Latch only at scene start/cuts so prediction noise cannot
        // make the camera breathe within a shot. Mode 2 is the offline harness's exact-scale grid.
        float artistic_multiplier = artistic_policy > 0.5f ? s2.w : 0.0f;
        if (artistic_policy > 0.5f && (!initialized || hard_cut || artistic_multiplier <= 0.0f)) {
            if (artistic_policy > 1.5f) {
                artistic_multiplier = clamp(artistic_style_or_override, 0.5f, 1.5f);
            } else {
                float safe_ceiling = ArtisticGlobal[0];
                float ceiling_confidence = ArtisticGlobal[1];
                if (!isfinite(safe_ceiling) || !isfinite(ceiling_confidence)) {
                    safe_ceiling = 1.0f;
                    ceiling_confidence = 0.0f;
                }
                safe_ceiling = clamp(safe_ceiling, 1.0f, 1.5f);
                ceiling_confidence = saturate(ceiling_confidence);
                float style_mix = saturate(artistic_style_or_override);
                float requested_scale = lerp(1.0f, safe_ceiling, style_mix);
                artistic_multiplier = ceiling_confidence >= 0.5f ?
                                      min(requested_scale, safe_ceiling) : 1.0f;
            }
        }

        s1 = float4(lo_val, inv_range, conv_ema, pop_ratio);
        s2 = float4(zero_anchor_shift, zero_valid, zero_plane_mode, artistic_multiplier);
    }
    // total == 0 (uninitialized depth): keep previous state.

    SubjectState[0] = s;
    SubjectState[1] = s1;
    SubjectState[2] = s2;

    for (uint rb = 0; rb < NUM_BINS; rb++) {
        SubjectHist[rb] = 0u;
        PlainHist[rb] = 0u;
    }
    PlainHist[NUM_BINS] = 0u;
    PlainHist[NUM_BINS + 1] = 0u;
}
