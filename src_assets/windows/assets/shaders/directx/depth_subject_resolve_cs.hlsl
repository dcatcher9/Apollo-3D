// 1-thread pass: resolve the tracked subject depth from the weighted histogram
// (depth_subject_hist_cs), EMA it for stability, and precompute everything the
// reprojection needs per pixel:
//   SubjectState[0] = { recenter_delta, scene_age, subject_depth_ema, initialized }
//   SubjectState[1] = { stretch_lo, stretch_inv_range, Bestv2 convergence EMA,
//                       adaptive pop ratio }
//   SubjectState[2] = { shot-latched zero-plane anchor shift in source pixels, valid,
//                       depth-cut state (-1 cooldown, 0 startup, 1 ready), color history valid }
// The reprojection then evaluates the permanent Bestv2 pixel-calibrated field.
// Resets the histogram for the next frame's accumulation.

RWStructuredBuffer<uint>   SubjectHist  : register(u0);  // 256 weighted bins (subject estimate)
RWStructuredBuffer<float4> SubjectState : register(u1);  // [0..2], see header above
RWStructuredBuffer<uint>   PlainHist    : register(u2);  // 256 unweighted bins (stretch 5/95 pct)

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

        bool initialized = s.w > 0.5f;

        // Disparity stretch (Bestv2 shape_depth_for_pop): rescale the [lo,hi] percentile band of
        // the (unweighted) depth distribution to full [0,1] so the mid-range uses the whole
        // parallax budget. lo=0, inv_range=1 when off -> the recenter path below is unchanged.
        float lo_val = 0.0f, inv_range = 1.0f;
        float background_val = 0.25f, median_val = 0.5f;
        float ptotal = 0.0f;
        for (uint pb = 0; pb < NUM_BINS; pb++) ptotal += (float)PlainHist[pb];
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

        // Subject and convergence histories always need cut detection. Adaptive-pop and explicit
        // zero-plane camera parameters are optionally latched below, but disabling both must not
        // allow the preceding shot's subject/convergence EMA to bleed into a new scene.
        float scene_age = initialized ? min(previous_scene_age + 1.0f, 65535.0f) : 0.0f;
        float change_fraction = ptotal > 0.5f ? (float)PlainHist[NUM_BINS + 1] / ptotal : 0.0f;
        float color_change_fraction = ptotal > 0.5f ?
                                      (float)PlainHist[NUM_BINS + 2] / ptotal : 0.0f;
        // Normalization settling can change 50-60% of depth texels on the first few frames.
        // The committed scene-cut clip reaches 66.8%, so 65% separates that cut from ordinary
        // startup/motion in the current core suite. Depth-change detection therefore waits for
        // settling, but color-change detection can safely arm as soon as one real prior NCHW
        // input exists. This avoids an eight-update blind window for similar-depth shot cuts.
        float depth_cut_state = s2.z;
        bool depth_cut_ready = depth_cut_state > 0.5f;
        bool color_history_valid = s2.w > 0.5f;
        bool hard_cut = initialized &&
                        ((depth_cut_ready && change_fraction >= 0.65f) ||
                         (color_history_valid && color_change_fraction >= 0.70f));
        if (!depth_cut_ready && initialized && scene_age >= 8.0f) {
            // Become ready for the *next* update; the last startup-settling frame cannot
            // retroactively classify itself as a depth cut.
            depth_cut_state = 1.0f;
            depth_cut_ready = true;
        }
        if (!initialized || hard_cut) {
            scene_age = 0.0f;
        }
        if (hard_cut) {
            depth_cut_state = -1.0f;
            depth_cut_ready = false;
        } else if (depth_cut_state < -0.5f &&
                   ((change_fraction < 0.35f && color_change_fraction < 0.50f) ||
                    scene_age >= 2.0f)) {
            // Prefer a genuinely stable frame, but never stay blind indefinitely during sustained
            // motion: after two updates the detector is rearmed regardless.
            depth_cut_state = 1.0f;
            depth_cut_ready = true;
        }

        // Reset temporal subject/convergence state on a detected cut. Otherwise the previous
        // scene bleeds into the first frames of the new shot even though pop/zero-plane relatch.
        // Between cuts retain the validated Bestv2 SubjectDepthEMA (new-value weight 0.20).
        float subj = (!initialized || hard_cut) ? subj_raw : lerp(s.z, subj_raw, 0.20f);
        float subj_str = saturate((subj - lo_val) * inv_range);
        float delta = (0.5f - subj_str) * subject_recenter;
        s = float4(delta, 0.0f, subj, 1.0f);
        float conv_target = (1.0f - subj) * 0.006f;
        float conv_ema = (!initialized || hard_cut) ? conv_target :
                         lerp(s1.z, conv_target, 0.10f);

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
        // Keep the detector's settling/cooldown clock even when optional scene-camera controls are
        // disabled; otherwise depth-only cuts can never arm after their eight-update settle time.
        s.y = scene_age;

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
        s1 = float4(lo_val, inv_range, conv_ema, pop_ratio);
        s2 = float4(zero_anchor_shift, zero_valid,
                    depth_cut_state,
                    1.0f); // current NCHW input is copied to history after this dispatch
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
    PlainHist[NUM_BINS + 2] = 0u;
}
