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

// Updates to wait after a cut before classifying scene risk for adaptive pop. Matches the depth
// cut detector's settling window below, which exists because normalization settling perturbs
// 50-60% of depth texels on the first few frames. Until then the floor is held, so an unsettled
// field can never win the ceiling for a whole shot.
#define POP_CLASSIFY_SETTLE_FRAMES 8.0f

// Scene-risk endpoints, calibrated against the MEASURED weighted edge fraction of the committed
// suites rather than against synthetic content. Across all 23 clips the three synthetic ones
// (fast_motion 0.0001, flat_transition 0.0048, flat_page 0.0087) sit far below real footage, which
// spans 0.038-0.245 with a median near 0.10. The previous 0.007/0.016 endpoints therefore
// saturated on every real clip, pinning the controller to its floor and making the adaptive band
// inert -- only the synthetic clips ever reached the ceiling. These endpoints span roughly the
// 10th-90th percentile of real content so the band is actually exercised.
// New-value weight for the P5/P95 stretch band EMA. Matches sbs_3d_minmax_ema (0.18), which
// smooths the same kind of quantity -- a depth-domain range -- rather than the convergence EMA,
// which smooths an anchor. Reset on a cut like every other temporal state here.
#define STRETCH_BAND_EMA 0.18f
// Band percentile tail. The band clips by design; at 0.05 (P5/P95) the measured plateau reached
// 15.8%/22.6% of pixels because the EMA lags the live distribution. Widening reduces the CLIPPING
// itself rather than softening its edge.
#define STRETCH_BAND_TAIL 0.02f

#define POP_RISK_LOW 0.04f
#define POP_RISK_HIGH 0.20f

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
            float lo_c = STRETCH_BAND_TAIL * ptotal, bg_c = 0.25f * ptotal;
            float med_c = 0.50f * ptotal, hi_c = (1.0f - STRETCH_BAND_TAIL) * ptotal;
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

        // Damp the stretch band, the last per-frame adaptive gain in the depth domain that had no
        // smoothing at all (subject depth, convergence and the normalization min/max are all EMA'd).
        // lo/inv_range form a MULTIPLICATIVE gain, so an unsmoothed band makes the depth mapping
        // breathe between cuts and that wobble is then multiplied by the pop strength.
        // Consistently positive on real content: core-real jitter -2.0%, extended -4.3%, and -6.6%
        // on a 240-frame native clip, with stereo volume flat. It DOES regress the synthetic
        // async-depth-ghost probe (fast_motion jitter 1.72 -> 3.15); see the roadmap -- the likely
        // mechanism is that band smoothing compounds an existing depth/color temporal misalignment,
        // which that clip exists to expose. Revisit if async-depth ghosting is ever chased directly.
        // Attack fast, release slow, and smooth in (lo, hi) space rather than on the reciprocal.
        // A symmetric EMA lags the live percentiles, and any frame whose band is narrower than the
        // live P2/P98 clips the difference in Bestv2WarpDepth -- that lag is what kept the plateau
        // above the 4% this band nominally implies. Expanding to cover the live percentiles
        // immediately removes lag-induced clipping; contraction still decays at STRETCH_BAND_EMA,
        // which is the direction that actually causes the mapping to breathe.
        if (initialized && !hard_cut && subject_stretch > 0.5f) {
            float hi_live = lo_val + 1.0f / max(inv_range, 1e-4f);
            float prev_hi = s1.x + 1.0f / max(s1.y, 1e-4f);
            lo_val = min(lerp(s1.x, lo_val, STRETCH_BAND_EMA), lo_val);
            float hi_val = max(lerp(prev_hi, hi_live, STRETCH_BAND_EMA), hi_live);
            inv_range = 1.0f / max(hi_val - lo_val, 1e-4f);
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
            // PlainHist[NUM_BINS] accumulates gradient-magnitude-weighted edge texels in fixed
            // point (EDGE_WEIGHT_SCALE is shared via include/depth_constants.hlsl). Dividing by it yields
            // a threshold-equivalent edge fraction: identical to the historical count when every
            // edge sits at the threshold, and proportionally larger when edges are more violent.
            // The POP_RISK_LOW/HIGH endpoints below are calibrated to measured content.
            float edge_fraction = (float)PlainHist[NUM_BINS] / (ptotal * EDGE_WEIGHT_SCALE);
            if (!initialized || hard_cut) {
                // Classify on a SETTLED depth field, never on the cut frame. Normalization
                // settling changes 50-60% of depth texels on the first few frames (see the cut
                // detector above, which waits the same 8 updates for exactly this reason). An
                // unsettled field reads smoother than the shot really is, so a busy scene can be
                // classified as clean and hold the full ceiling for its entire duration -- which
                // is what gave the opening shot of tartanair_house_easy maximum pop and the worst
                // cross-row shear in the suite. Hold the conservative floor until it settles:
                // when the risk is not yet measurable, do not grant the bonus.
                pop_ratio = 1.0f;
            } else if (scene_age == POP_CLASSIFY_SETTLE_FRAMES) {
                // One classification per shot, on the first settled field, then bit-stable until
                // the next cut. Equality rather than >= keeps that single-shot latch exact.
                // Full extra pop for low-complexity depth fields (<= POP_RISK_LOW weighted
                // edge fraction), fading to the floor by POP_RISK_HIGH. The classification decides
                // only where the configured gain is useful, not what the endpoints are.
                float confidence = 1.0f - smoothstep(POP_RISK_LOW, POP_RISK_HIGH, edge_fraction);
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
            // Keep RE-RESOLVING the anchor until the depth field settles, then freeze it for the
            // shot. Latching on the cut frame itself is the same defect that was fixed above for
            // the pop classifier: normalization settling perturbs 50-60% of depth texels on the
            // first frames, and lo_val/inv_range/delta feeding the anchor below are raw cut-frame
            // values too (the stretch band's EMA is reset on a cut by design). A bad latch here is
            // worse than a bad pop class, because the whole point of a shot-latched zero plane is
            // that it does not move -- so it is unrecoverable until the next cut.
            // Resolve TWICE, not continuously: once immediately so the new shot never renders on
            // the previous shot's plane, then once more when the field has settled. Tracking every
            // frame through the settle window was measured and is worse -- it converts one
            // correction into ~8 frames of drift, and scene_cut (the clip built to probe
            // normalization swim across cuts) regressed 4.90 -> 8.19 on static_jitter_p95.
            if (!initialized || hard_cut || zero_valid < 0.5f ||
                scene_age == POP_CLASSIFY_SETTLE_FRAMES) {
                float zero_anchor_depth = zero_plane_mode < 1.5f ? subj :
                                          zero_plane_mode < 2.5f ? median_val : background_val;
                // Must shape identically to Bestv2WarpDepth, or the anchor stops describing the
                // plane the warp actually renders: one soft clamp over the recentred band value.
                float zero_anchor_shaped = subject_stretch > 0.5f ?
                    (zero_anchor_depth - lo_val) * inv_range : zero_anchor_depth;
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
