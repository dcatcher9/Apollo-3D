// 1-thread pass: resolve the tracked subject depth from the weighted histogram
// (depth_subject_hist_cs), EMA it for stability, and precompute everything the
// reprojection needs per pixel:
//   SubjectState[0] = { recenter_delta, scene_age, subject_depth_ema, initialized }
//   SubjectState[1] = { stretch_lo, stretch_inv_range, depth-change baseline,
//                       adaptive pop ratio }
//   SubjectState[2] = { shot-latched zero-plane anchor shift in source pixels, valid,
//                       cut-state flags (stored exactly as a uint-valued float),
//                       model-input history state:
//                         0 empty, 1 normal advance, 2 one-update hold,
//                         3 persistent-low endpoint (advance + return authority) }
// Append-only telemetry; the production warp never reads these elements:
//   SubjectState[3] = { settle-latched edge fraction, current depth-change fraction,
//                       valid-depth fraction, effective raw-range width }
//   SubjectState[4] = uint-bit counters { hard cuts, external cuts, empty raw, collapsed raw }
//   SubjectState[5] = { current range-collapsed flag, depth-ready flag, hard-cut pulse, reserved }
//   SubjectState[6] = { current edge fraction, current zero-anchor candidate shift in source px,
//                       structural-change fraction, raw-RGB-change fraction }
//   SubjectState[7] = { current/previous/common structural support fractions,
//                       analysis flag bitmask stored as an exact uint-valued float }
// The reprojection then evaluates the permanent Bestv2 pixel-calibrated field.
// Resets the histogram for the next frame's accumulation.

RWStructuredBuffer<uint>   SubjectHist  : register(u0);  // 256 weighted bins (subject estimate)
RWStructuredBuffer<float4> SubjectState : register(u1);  // [0..7], see header above
RWStructuredBuffer<uint>   PlainHist    : register(u2);  // 256 bins + seven evidence counters
StructuredBuffer<uint4>    FrameRoiTransform : register(t0);
StructuredBuffer<uint4>    PreviousFrameRoiTransform : register(t1);

#include "include/depth_constants.hlsl"
#include "include/bestv2_curve.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"
#include "include/sbs_frame_roi_transform.hlsl"

#define NUM_BINS 256

// Updates to wait after a cut before classifying scene risk for adaptive pop. Matches the depth
// cut detector's settling window below, which exists because normalization settling perturbs
// 50-60% of depth texels on the first few frames. Until then the floor is held, so an unsettled
// field can never win the ceiling for a whole shot.
#define POP_CLASSIFY_SETTLE_FRAMES 8.0f

// Independent proposal arms prevent one signal from starving the other. A cut clears both arms.
// Geometry rearms after two low-depth updates; appearance rearms after two updates without the
// qualified broad-RGB + ordinal proposal. CUT_FLAG_LATCHED is a permanent "has accepted a cut"
// phase marker for this estimator lifetime: independent branch rearm deliberately never clears
// it. CUT_FLAG_APPEARANCE_RECOVERY marks the first update after a photometric veto. Geometry is
// blocked only if that update is a quiet, well-supported repeat of the same endpoint: neural-depth
// normalization can remain unsettled one update after a flash has returned even though RGB and
// ordinal evidence are already quiet. A disjoint or correlated new endpoint remains authoritative
// during that guard. This asymmetry distinguishes post-cut recovery from startup, where no weak
// or relative cut path is allowed.
// Scene-risk endpoints, calibrated against the MEASURED weighted edge fraction of the committed
// suites. The three stable-shot synthetic probes used for calibration (fast_motion 0.0001,
// flat_transition 0.0048, flat_page 0.0087) sit far below the remaining non-probe measurements,
// which span 0.038-0.245 with a median near 0.10. Those measurements mix declared real-capture,
// animation, simulation, ai-generated, anime and unclassified content. The previous 0.007/0.016
// endpoints saturated across that mixed set, pinning the controller to its floor and making the
// adaptive band inert. These endpoints span roughly its
// 10th-90th percentile so the band is actually exercised.
// New-value weight for the P5/P95 stretch band EMA. Matches sbs_3d_minmax_ema (0.18), which
// smooths the same kind of quantity -- a depth-domain range -- rather than an anchor smoother.
// Reset on a cut like every other temporal state here.
#define STRETCH_BAND_EMA 0.18f
// Band percentile tail. The band clips by design; at 0.05 (P5/P95) the measured plateau reached
// 15.8%/22.6% of pixels because the EMA lags the live distribution. Widening reduces the CLIPPING
// itself rather than softening its edge.
#define STRETCH_BAND_TAIL 0.02f

#define POP_RISK_LOW 0.04f
#define POP_RISK_HIGH 0.20f

SbsFrameRoiTransformData LoadPreviousFrameRoiTransform() {
    return SBS_FRAME_ROI_DECODE_RESOURCE(
        PreviousFrameRoiTransform);
}

bool TargetDimensionsSafe() {
    return target_w > 0u &&
           target_h > 0u &&
           target_w <= 0xffffffffu / target_h;
}

[numthreads(1, 1, 1)]
void main() {
    uint subject_hist_count;
    uint subject_hist_stride;
    SubjectHist.GetDimensions(
        subject_hist_count,
        subject_hist_stride);
    uint subject_state_count;
    uint subject_state_stride;
    SubjectState.GetDimensions(
        subject_state_count,
        subject_state_stride);
    uint plain_hist_count;
    uint plain_hist_stride;
    PlainHist.GetDimensions(
        plain_hist_count,
        plain_hist_stride);
    uint current_transform_vectors;
    uint current_transform_stride;
    FrameRoiTransform.GetDimensions(
        current_transform_vectors,
        current_transform_stride);
    uint previous_transform_vectors;
    uint previous_transform_stride;
    PreviousFrameRoiTransform.GetDimensions(
        previous_transform_vectors,
        previous_transform_stride);
    bool histogram_resources_safe =
        subject_hist_count >= NUM_BINS &&
        subject_hist_stride == 4u &&
        plain_hist_count >= NUM_BINS + 7u &&
        plain_hist_stride == 4u;
    bool subject_state_safe =
        subject_state_count >= SBS_ADAPTIVE_STATE_VECTOR_COUNT &&
        subject_state_stride == 16u;
    if (!histogram_resources_safe ||
        !subject_state_safe) {
        return;
    }

    // Total weighted votes.
    float total = 0.0f;
    for (uint b = 0; b < NUM_BINS; b++) {
        total += (float)SubjectHist[b];
    }

    float4 s = SubjectState[SBS_STATE_VECTOR_SUBJECT_RECENTER_DELTA];
    float4 s1 = SubjectState[SBS_STATE_VECTOR_STRETCH_LO];
    float4 s2 = SubjectState[SBS_STATE_VECTOR_ZERO_ANCHOR_SHIFT_PX];
    float4 telemetry = SubjectState[SBS_STATE_VECTOR_LATCHED_EDGE_FRACTION];
    uint4 health_counters =
        asuint(SubjectState[SBS_STATE_VECTOR_HARD_CUT_COUNT]);
    float4 telemetry_flags = SubjectState[SBS_STATE_VECTOR_RANGE_COLLAPSED];
    float4 current_diagnostics = 0.0f;
    SBS_STATE_CURRENT_EDGE_FRACTION(current_diagnostics) = -1.0f;
    SBS_STATE_CURRENT_ZERO_ANCHOR_CANDIDATE_SHIFT_PX(current_diagnostics) = -1.0f;
    SBS_STATE_STRUCTURAL_CHANGE_FRACTION(current_diagnostics) = -1.0f;
    SBS_STATE_RAW_RGB_CHANGE_FRACTION(current_diagnostics) = -1.0f;
    float4 analysis_diagnostics = 0.0f;
    SBS_STATE_CURRENT_STRUCTURAL_SUPPORT_FRACTION(analysis_diagnostics) = -1.0f;
    SBS_STATE_PREVIOUS_STRUCTURAL_SUPPORT_FRACTION(analysis_diagnostics) = -1.0f;
    SBS_STATE_COMMON_STRUCTURAL_SUPPORT_FRACTION(analysis_diagnostics) = -1.0f;
    // One-update values must never look live merely because an invalid frame preserved state.
    SBS_STATE_CURRENT_DEPTH_CHANGE_FRACTION(telemetry) = 0.0f;
    SBS_STATE_HARD_CUT_PULSE(telemetry_flags) = 0.0f;
    SbsFrameRoiTransformData current_transform =
        FrameRoiTransformLoad();
    SbsFrameRoiTransformData previous_transform =
        LoadPreviousFrameRoiTransform();
    bool current_transform_valid =
        current_transform_vectors >= SBS_FRAME_ROI_VECTOR_COUNT &&
        current_transform_stride == 16u &&
        FrameRoiDataValid(current_transform);
    bool current_transform_dimensions_match =
        current_transform_valid &&
        TargetDimensionsSafe() &&
        all(FrameRoiDataModelDimensions(current_transform) ==
            uint2(target_w, target_h));
    bool previous_transform_valid =
        previous_transform_vectors >= SBS_FRAME_ROI_VECTOR_COUNT &&
        previous_transform_stride == 16u &&
        FrameRoiDataValid(previous_transform);
    bool legacy_unbound_pair =
        FrameRoiDataUnboundZero(current_transform) &&
        FrameRoiDataUnboundZero(previous_transform);
    bool current_transform_usable =
        legacy_unbound_pair ||
        current_transform_dimensions_match;
    bool geometry_reseed =
        current_transform_dimensions_match &&
        !legacy_unbound_pair &&
        (!previous_transform_valid ||
         FrameRoiDataGeometryReseedRequired(
             current_transform,
             previous_transform));
    if (total > 0.5f &&
        current_transform_usable) {
        float previous_scene_age = SBS_STATE_SCENE_AGE(s);
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

        // A transform/generation change is a geometry reseed, not a content cut. Treat the first
        // valid completion as a fresh adaptive field so normalization, subject EMA, stretch, pop,
        // and zero plane snap without emitting or incrementing the hard-cut signal.
        bool initialized =
            SBS_STATE_INITIALIZED(s) > 0.5f &&
            !geometry_reseed;

        // Disparity stretch (Bestv2 shape_depth_for_pop): rescale the [lo,hi] percentile band of
        // the (unweighted) depth distribution to full [0,1] so the mid-range uses the whole
        // parallax budget. lo=0, inv_range=1 when off -> the recenter path below is unchanged.
        float lo_val = 0.0f, inv_range = 1.0f;
        float background_val = 0.25f, median_val = 0.5f;
        float ptotal = 0.0f;
        for (uint pb = 0; pb < NUM_BINS; pb++) ptotal += (float)PlainHist[pb];
        if (ptotal > 0.5f) {
            float lo_c = STRETCH_BAND_TAIL * ptotal, bg_c = 0.25f * ptotal;
            float med_c = 0.50f * ptotal, hi_c = (1.0f - STRETCH_BAND_TAIL) * ptotal;
            float pc = 0.0f, hv = 1.0f;
            bool got_lo = false, got_bg = false, got_med = false;
            for (uint qb = 0; qb < NUM_BINS; qb++) {
                pc += (float)PlainHist[qb];
                // Band BOUNDS take the crossing bin's outer edge so a large atom in that bin is
                // not cut through and clipped (see depth_minmax_ema_cs for the same reasoning).
                // background/median are point estimates, not bounds, so they keep the bin center.
                float qv = ((float)qb + 0.5f) / (float)NUM_BINS;
                if (!got_lo && pc >= lo_c) { lo_val = (float)qb / (float)NUM_BINS; got_lo = true; }
                if (!got_bg && pc >= bg_c) { background_val = qv; got_bg = true; }
                if (!got_med && pc >= med_c) { median_val = qv; got_med = true; }
                if (pc >= hi_c) { hv = ((float)qb + 1.0f) / (float)NUM_BINS; break; }
            }
            if (subject_stretch > 0.5f) {
                inv_range = 1.0f / max(hv - lo_val, 1e-4f);
            }
        }

        // The subject history always needs cut detection. Adaptive-pop and explicit zero-plane
        // camera parameters are optionally latched below, but disabling both must not allow the
        // preceding shot's subject EMA to bleed into a new scene.
        float scene_age = initialized ? min(previous_scene_age + 1.0f, 65535.0f) : 0.0f;
        float change_fraction = ptotal > 0.5f ? (float)PlainHist[NUM_BINS + 1] / ptotal : 0.0f;
        float structural_change_fraction = ptotal > 0.5f ?
                                           (float)PlainHist[NUM_BINS + 2] / ptotal : 0.0f;
        float raw_rgb_change_fraction = ptotal > 0.5f ?
                                        (float)PlainHist[NUM_BINS + 3] / ptotal : 0.0f;
        float current_structural_support_fraction = ptotal > 0.5f ?
                                                    (float)PlainHist[NUM_BINS + 4] / ptotal : 0.0f;
        float previous_structural_support_fraction = ptotal > 0.5f ?
                                                     (float)PlainHist[NUM_BINS + 5] / ptotal : 0.0f;
        float common_structural_support_fraction = ptotal > 0.5f ?
                                                   (float)PlainHist[NUM_BINS + 6] / ptotal : 0.0f;
        // Normalization settling can change 50-60% of depth texels on the first few frames.
        // The measured committed cuts reach 62.5% and 63.1%, while ordinary core motion stays
        // below 19.7%; 60% is therefore the armed geometry authority. The weaker 25% path requires
        // BOTH broad RGB replacement and ordinal structural change. A flash passes broad RGB but
        // fails ordinal structure; detailed motion can pass ordinal but stays far below broad RGB.
        bool model_input_history_valid =
            !geometry_reseed &&
            SBS_STATE_MODEL_INPUT_HISTORY_STATE(s2) > 0.5f;
        SBS_STATE_CURRENT_STRUCTURAL_SUPPORT_FRACTION(analysis_diagnostics) =
            current_structural_support_fraction;
        SBS_STATE_PREVIOUS_STRUCTURAL_SUPPORT_FRACTION(analysis_diagnostics) =
            model_input_history_valid ?
                previous_structural_support_fraction :
                -1.0f;
        SBS_STATE_COMMON_STRUCTURAL_SUPPORT_FRACTION(analysis_diagnostics) =
            model_input_history_valid ?
                common_structural_support_fraction :
                -1.0f;
        SBS_STATE_STRUCTURAL_CHANGE_FRACTION(current_diagnostics) =
            model_input_history_valid ?
                structural_change_fraction :
                -1.0f;
        SBS_STATE_RAW_RGB_CHANGE_FRACTION(current_diagnostics) =
            model_input_history_valid ?
                raw_rgb_change_fraction :
                -1.0f;
        bool model_input_history_gap =
            !geometry_reseed &&
            SBS_STATE_MODEL_INPUT_HISTORY_STATE(s2) > 1.5f &&
            SBS_STATE_MODEL_INPUT_HISTORY_STATE(s2) < 2.5f;
        bool low_structure_scene =
            !geometry_reseed &&
            SBS_STATE_MODEL_INPUT_HISTORY_STATE(s2) > 2.5f;
        bool appearance_proposal =
            model_input_history_valid &&
            raw_rgb_change_fraction >= RAW_RGB_CUT_HIGH &&
            structural_change_fraction >= STRUCTURAL_COLOR_CUT_HIGH;
        bool current_structure_reliable =
            current_structural_support_fraction >= STRUCTURAL_COLOR_MIN_SUPPORT;
        bool previous_structure_reliable =
            previous_structural_support_fraction >= STRUCTURAL_COLOR_MIN_SUPPORT;
        bool common_structure_reliable =
            common_structural_support_fraction >= STRUCTURAL_COLOR_MIN_SUPPORT;
        bool common_structure_representative =
            common_structure_reliable &&
            common_structural_support_fraction >=
                STRUCTURAL_COLOR_EXPOSURE_MIN_COMMON_RATIO *
                min(current_structural_support_fraction,
                    previous_structural_support_fraction);
        bool broad_rgb_transition =
            model_input_history_valid &&
            raw_rgb_change_fraction >= RAW_RGB_CUT_HIGH;
        // Quiet ordinal change is evidence of exposure only when both frames and their common
        // comparison sites carry enough usable structure. Without these support checks, a black
        // or fully clipped frame vacuously looks exposure-like because every ordinal pair abstains.
        bool exposure_like_transition =
            broad_rgb_transition &&
            current_structure_reliable &&
            previous_structure_reliable &&
            common_structure_representative &&
            structural_change_fraction < STRUCTURAL_COLOR_EXPOSURE_QUIET;
        // A transition from reliable structure into a structureless frame is intentionally
        // deferred and the last reliable appearance/depth history is retained below. Raw RGB
        // cannot qualify the flat frame itself: a uniform value can match the previous scene's
        // dominant color. The next supported frame resolves the deferred relation against the
        // preserved endpoint.
        bool structureless_candidate =
            model_input_history_valid &&
            previous_structure_reliable &&
            !current_structure_reliable;
        // One update is enough to distinguish a flash from persistent low-detail content. Hold A
        // exactly once. On a second consecutive low-structure update, preserved A-vs-current
        // geometry is authoritative and history advances afterward whether or not it crosses the
        // cut threshold; that prevents an indefinite veto and prevents periodic retriggers.
        bool structureless_transition =
            structureless_candidate &&
            !model_input_history_gap;
        bool persistent_structureless_transition =
            structureless_candidate &&
            model_input_history_gap;
        // The history-copy pass holds depth and appearance together. Suppress only a near-exact
        // supported endpoint return. The former broad-RGB (<70%) veto classified many visibly
        // different endpoints as "same"; a low-appearance B could then overrule authoritative
        // A-vs-B depth. A strict 1% endpoint-equivalence bound still bridges an exact clipped flash
        // while allowing photometrically similar, geometrically different content to cut.
        bool same_scene_gap_return =
            model_input_history_gap &&
            current_structure_reliable &&
            raw_rgb_change_fraction < STRUCTURELESS_RETURN_RGB_SAME_MAX &&
            structural_change_fraction < STRUCTURAL_COLOR_EXPOSURE_QUIET;
        bool base_appearance_veto =
            exposure_like_transition ||
            structureless_transition ||
            same_scene_gap_return;
        bool photometric_recovery_veto =
            exposure_like_transition || same_scene_gap_return;
        bool quiet_supported_repeat =
            model_input_history_valid &&
            current_structure_reliable &&
            previous_structure_reliable &&
            common_structure_representative &&
            raw_rgb_change_fraction < STRUCTURELESS_RETURN_RGB_SAME_MAX &&
            structural_change_fraction < STRUCTURAL_COLOR_EXPOSURE_QUIET;
        uint cut_flags = (uint)max(SBS_STATE_CUT_FLAGS(s2), 0.0f);
        bool geometry_armed = (cut_flags & CUT_FLAG_GEOMETRY_ARMED) != 0u;
        bool appearance_armed = (cut_flags & CUT_FLAG_APPEARANCE_ARMED) != 0u;
        bool cut_latched = (cut_flags & CUT_FLAG_LATCHED) != 0u;
        bool appearance_recovery =
            (cut_flags & CUT_FLAG_APPEARANCE_RECOVERY) != 0u;
        bool appearance_recovery_tail =
            appearance_recovery && quiet_supported_repeat;
        bool appearance_veto =
            base_appearance_veto || appearance_recovery_tail;
        float depth_change_baseline = initialized ?
            saturate(SBS_STATE_DEPTH_CHANGE_BASELINE_EMA(s1)) :
            change_fraction;

        // A relative geometry spike is the no-starvation escape while absolute geometry remains
        // latched high. The EMA is evaluated from the PREVIOUS update, reset to every accepted cut,
        // and then follows steady evidence. A constant high signal therefore cannot periodically
        // retrigger. The normal settling window is also a refractory: a cut-frame normalization
        // jump cannot masquerade as a second cut. After settling, a sufficiently large new rise
        // remains detectable without a periodic cooldown escape.
        bool relative_geometry_spike =
            cut_latched && !geometry_armed &&
            !appearance_veto &&
            scene_age >= POP_CLASSIFY_SETTLE_FRAMES &&
            change_fraction >= DEPTH_CUT_RELATIVE_FLOOR &&
            (change_fraction >= depth_change_baseline + DEPTH_CUT_RELATIVE_MARGIN ||
             change_fraction >=
                 depth_change_baseline * DEPTH_CUT_RELATIVE_MULTIPLIER);
        uint analysis_flags =
            (appearance_proposal ? ANALYSIS_FLAG_APPEARANCE_PROPOSAL : 0u) |
            (exposure_like_transition ? ANALYSIS_FLAG_EXPOSURE_LIKE : 0u) |
            ((structureless_transition || persistent_structureless_transition) ?
                ANALYSIS_FLAG_STRUCTURELESS : 0u) |
            (same_scene_gap_return ? ANALYSIS_FLAG_SAME_RETURN : 0u) |
            (appearance_veto ? ANALYSIS_FLAG_VETO : 0u) |
            (relative_geometry_spike ? ANALYSIS_FLAG_RELATIVE_SPIKE : 0u);
        // Do not bit-cast small uint masks into IEEE subnormals. GPU FTZ mode
        // legally collapses those bit patterns to zero before readback.
        SBS_STATE_ANALYSIS_FLAGS(analysis_diagnostics) = (float)analysis_flags;
        bool shot_cut =
            initialized &&
            ((geometry_armed && !appearance_veto &&
              change_fraction >= DEPTH_CUT_HIGH) ||
             (appearance_armed && appearance_proposal &&
              change_fraction >= DEPTH_CUT_CORROBORATE) ||
             (low_structure_scene && current_structure_reliable &&
              change_fraction >= DEPTH_CUT_HIGH) ||
             relative_geometry_spike);

        if (!initialized) {
            scene_age = 0.0f;
            cut_flags = 0u;
            depth_change_baseline = change_fraction;
        } else if (shot_cut) {
            scene_age = 0.0f;
            cut_flags = CUT_FLAG_LATCHED;
            depth_change_baseline = change_fraction;
        } else {
            // The structureless frame has no trustworthy geometry. Its comparison to the
            // preserved endpoint must not pull the relative-cut baseline upward and hide the
            // supported A-vs-B return that resolves the deferred transition.
            if (!structureless_transition && !appearance_recovery_tail) {
                depth_change_baseline =
                    lerp(depth_change_baseline, change_fraction, DEPTH_CUT_BASELINE_ALPHA);
            }
            if (!cut_latched) {
                if (scene_age >= POP_CLASSIFY_SETTLE_FRAMES) {
                    // Arm for the NEXT update. Startup normalization and appearance changes can
                    // never fire either weak branch on the update that completes settling.
                    cut_flags = CUT_FLAG_GEOMETRY_ARMED | CUT_FLAG_APPEARANCE_ARMED;
                }
            } else {
                if (!geometry_armed) {
                    if (change_fraction < DEPTH_CUT_LOW) {
                        if ((cut_flags & CUT_FLAG_GEOMETRY_LOW_ONCE) != 0u) {
                            cut_flags |= CUT_FLAG_GEOMETRY_ARMED;
                            cut_flags &= ~CUT_FLAG_GEOMETRY_LOW_ONCE;
                        } else {
                            cut_flags |= CUT_FLAG_GEOMETRY_LOW_ONCE;
                        }
                    } else {
                        cut_flags &= ~CUT_FLAG_GEOMETRY_LOW_ONCE;
                    }
                }
                if (!appearance_armed) {
                    if (!appearance_proposal) {
                        if ((cut_flags & CUT_FLAG_APPEARANCE_QUIET_ONCE) != 0u) {
                            cut_flags |= CUT_FLAG_APPEARANCE_ARMED;
                            cut_flags &= ~CUT_FLAG_APPEARANCE_QUIET_ONCE;
                        } else {
                            cut_flags |= CUT_FLAG_APPEARANCE_QUIET_ONCE;
                        }
                    } else {
                        cut_flags &= ~CUT_FLAG_APPEARANCE_QUIET_ONCE;
                    }
                }
            }
            if (photometric_recovery_veto) {
                cut_flags |= CUT_FLAG_APPEARANCE_RECOVERY;
            } else {
                // `appearance_recovery` above was read from the previous update, so the first
                // quiet update remains guarded and only arms geometry again for the next frame.
                cut_flags &= ~CUT_FLAG_APPEARANCE_RECOVERY;
            }
        }
        float next_model_input_history_state =
            geometry_reseed ? 1.0f :
            (structureless_transition ? 2.0f :
             ((persistent_structureless_transition ||
               (low_structure_scene && !current_structure_reliable)) ? 3.0f :
              1.0f));

        // Damp the stretch band, the last per-frame adaptive gain in the depth domain that had no
        // smoothing at all (subject depth and the normalization min/max are both EMA'd).
        // lo/inv_range form a MULTIPLICATIVE gain, so an unsmoothed band makes the depth mapping
        // breathe between cuts and that wobble is then multiplied by the pop strength.
        // Aggregate jitter changed -2.0% on the historical seven-clip non-synthetic core grouping,
        // -4.3% on the mixed-content public extended suite, and -6.6% on a 240-frame native clip,
        // with stereo volume flat. It DOES regress the synthetic
        // async-depth-ghost probe (fast_motion jitter 1.72 -> 3.15); see the roadmap -- the likely
        // mechanism is that band smoothing compounds an existing depth/color temporal misalignment,
        // which that clip exists to expose. Revisit if async-depth ghosting is ever chased directly.
        // Attack fast, release slow, and smooth in (lo, hi) space rather than on the reciprocal.
        // A symmetric EMA lags the live percentiles, and any frame whose band is narrower than the
        // live P2/P98 clips the difference in Bestv2WarpDepth -- that lag is what kept the plateau
        // above the 4% this band nominally implies. Expanding to cover the live percentiles
        // immediately removes lag-induced clipping; contraction still decays at STRETCH_BAND_EMA,
        // which is the direction that actually causes the mapping to breathe.
        if (initialized && !shot_cut && subject_stretch > 0.5f) {
            float hi_live = lo_val + 1.0f / max(inv_range, 1e-4f);
            float prev_hi =
                SBS_STATE_STRETCH_LO(s1) +
                1.0f / max(SBS_STATE_STRETCH_INV_RANGE(s1), 1e-4f);
            lo_val = min(
                lerp(SBS_STATE_STRETCH_LO(s1), lo_val, STRETCH_BAND_EMA),
                lo_val);
            float hi_val = max(lerp(prev_hi, hi_live, STRETCH_BAND_EMA), hi_live);
            inv_range = 1.0f / max(hi_val - lo_val, 1e-4f);
        }

        // Reset temporal subject state on a detected cut. Otherwise the previous
        // scene bleeds into the first frames of the new shot even though pop/zero-plane relatch.
        // Between cuts retain the validated Bestv2 SubjectDepthEMA (new-value weight 0.20).
        float subj = (!initialized || shot_cut) ?
            subj_raw :
            lerp(SBS_STATE_SUBJECT_DEPTH_EMA(s), subj_raw, 0.20f);
        float subj_str = saturate((subj - lo_val) * inv_range);
        float delta = (0.5f - subj_str) * subject_recenter;
        SBS_STATE_SUBJECT_RECENTER_DELTA(s) = delta;
        SBS_STATE_SCENE_AGE(s) = 0.0f;
        SBS_STATE_SUBJECT_DEPTH_EMA(s) = subj;
        SBS_STATE_INITIALIZED(s) = 1.0f;

        // Depth-edge density predicts warp risk. Between cuts the multiplier remains bit-stable;
        // the base is the floor and the configured ceiling is never exceeded.
        float pop_ratio = max(SBS_STATE_ADAPTIVE_POP_RATIO(s1), 1.0f);
        float latched_edge_fraction =
            SBS_STATE_LATCHED_EDGE_FRACTION(telemetry);
        float current_edge_fraction = ptotal > 0.5f ?
            (float)PlainHist[NUM_BINS] / (ptotal * EDGE_WEIGHT_SCALE) :
            -1.0f;
        SBS_STATE_CURRENT_EDGE_FRACTION(current_diagnostics) =
            current_edge_fraction;
        if (adaptive_pop > 0.5f && ptotal > 0.5f) {
            // PlainHist[NUM_BINS] accumulates 434-reference-texel
            // gradient-magnitude-weighted edges in fixed point (the producer also scales its
            // saturation cap). EDGE_WEIGHT_SCALE is shared via include/depth_constants.hlsl.
            // Dividing by it yields a resolution-stable threshold-equivalent edge fraction:
            // identical to the historical count at the 434 grid and proportionally larger when
            // edges are more violent. The POP_RISK_LOW/HIGH endpoints below remain tied to their
            // measured 434-short-side calibration.
            if (!initialized || shot_cut) {
                // Classify on a SETTLED depth field, never on the cut frame. Normalization
                // settling changes 50-60% of depth texels on the first few frames (see the cut
                // detector above, which waits the same 8 updates for exactly this reason). An
                // unsettled field reads smoother than the shot really is, so a busy scene can be
                // classified as clean and hold the full ceiling for its entire duration -- which
                // is what gave the opening shot of tartanair_house_easy maximum pop and the worst
                // cross-row shear in the suite. Hold the conservative floor until it settles:
                // when the risk is not yet measurable, do not grant the bonus.
                pop_ratio = 1.0f;
                latched_edge_fraction = -1.0f;
            } else if (scene_age == POP_CLASSIFY_SETTLE_FRAMES) {
                // One classification per shot, on the first settled field, then bit-stable until
                // the next cut. Equality rather than >= keeps that single-shot latch exact.
                // Full extra pop for low-complexity depth fields (<= POP_RISK_LOW weighted
                // edge fraction), fading to the floor by POP_RISK_HIGH. The classification decides
                // only where the configured gain is useful, not what the endpoints are.
                float confidence = 1.0f - smoothstep(
                    POP_RISK_LOW, POP_RISK_HIGH, current_edge_fraction);
                pop_ratio = lerp(1.0f, max(adaptive_pop_max_ratio, 1.0f), confidence);
                latched_edge_fraction = current_edge_fraction;
            }
        } else {
            pop_ratio = 1.0f;
            latched_edge_fraction = -1.0f;
        }
        // Keep the detector's settling/cooldown clock even when optional scene-camera controls are
        // disabled; otherwise depth-only cuts can never arm after their eight-update settle time.
        SBS_STATE_SCENE_AGE(s) = scene_age;

        // Explicit artistic zero plane. Resolve the chosen raw anchor through this frame's
        // stretch/recenter/Bestv2 curve and latch the resulting source-pixel shift. Storing the
        // final shift rather than raw depth prevents later percentile/recenter motion from making
        // convergence breathe. Subject, median, and far/mid-background correspond to the paper's
        // shot-level affine offset t.
        // Resolved unconditionally. It used to be skipped for the removed `legacy` mode and for a
        // degenerate histogram, and the warp then fell back to a subject-anchor path. That path is
        // gone, so the anchor must always exist by the time s0.w marks the state initialized --
        // which it does, since this block and the s0 write share the same `total > 0.5f` guard.
        // With no histogram the percentile defaults (median 0.5, background 0.25, lo 0 / range 1)
        // still yield a sane plane, so there is nothing left to branch on.
        float zero_anchor_shift = SBS_STATE_ZERO_ANCHOR_SHIFT_PX(s2);
        float zero_valid = SBS_STATE_ZERO_ANCHOR_VALID(s2);
        {
            float zero_anchor_depth = zero_plane_mode < 1.5f ? subj :
                                      zero_plane_mode < 2.5f ? median_val : background_val;
            // Evaluate the current candidate every valid update even when the shot latch is
            // frozen. Offline two-sided boundary refinement can then aggregate the same exact
            // stretch/recenter/Bestv2 mapping without duplicating it outside the shader.
            float zero_anchor_shaped = subject_stretch > 0.5f ?
                (zero_anchor_depth - lo_val) * inv_range : zero_anchor_depth;
            zero_anchor_shaped = saturate(zero_anchor_shaped + delta);
            float current_zero_anchor_candidate_shift =
                Bestv2RawShiftPxFast(zero_anchor_shaped);
            SBS_STATE_CURRENT_ZERO_ANCHOR_CANDIDATE_SHIFT_PX(current_diagnostics) =
                current_zero_anchor_candidate_shift;
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
            if (!initialized || shot_cut || zero_valid < 0.5f ||
                scene_age == POP_CLASSIFY_SETTLE_FRAMES) {
                zero_anchor_shift = current_zero_anchor_candidate_shift;
                zero_valid = 1.0f;
            }
        }
        SBS_STATE_STRETCH_LO(s1) = lo_val;
        SBS_STATE_STRETCH_INV_RANGE(s1) = inv_range;
        SBS_STATE_DEPTH_CHANGE_BASELINE_EMA(s1) = depth_change_baseline;
        SBS_STATE_ADAPTIVE_POP_RATIO(s1) = pop_ratio;
        SBS_STATE_ZERO_ANCHOR_SHIFT_PX(s2) = zero_anchor_shift;
        SBS_STATE_ZERO_ANCHOR_VALID(s2) = zero_valid;
        SBS_STATE_CUT_FLAGS(s2) = (float)cut_flags;
        SBS_STATE_MODEL_INPUT_HISTORY_STATE(s2) =
            next_model_input_history_state;
        SBS_STATE_LATCHED_EDGE_FRACTION(telemetry) = latched_edge_fraction;
        SBS_STATE_CURRENT_DEPTH_CHANGE_FRACTION(telemetry) = change_fraction;
        if (shot_cut) {
            SBS_STATE_HARD_CUT_COUNT(health_counters) =
                min(SBS_STATE_HARD_CUT_COUNT(health_counters) + 1u, 0xfffffffeu);
            SBS_STATE_HARD_CUT_PULSE(telemetry_flags) = 1.0f;
        }
        // depth_valid_history_cs holds only state 2. State 3 advances the persistent-low endpoint
        // while retaining one supported-return authority.
    }
    // total == 0 (uninitialized depth): keep previous state.

    SubjectState[SBS_STATE_VECTOR_SUBJECT_RECENTER_DELTA] = s;
    SubjectState[SBS_STATE_VECTOR_STRETCH_LO] = s1;
    SubjectState[SBS_STATE_VECTOR_ZERO_ANCHOR_SHIFT_PX] = s2;
    SubjectState[SBS_STATE_VECTOR_LATCHED_EDGE_FRACTION] = telemetry;
    SubjectState[SBS_STATE_VECTOR_HARD_CUT_COUNT] = asfloat(health_counters);
    SubjectState[SBS_STATE_VECTOR_RANGE_COLLAPSED] = telemetry_flags;
    SubjectState[SBS_STATE_VECTOR_CURRENT_EDGE_FRACTION] = current_diagnostics;
    SubjectState[SBS_STATE_VECTOR_CURRENT_STRUCTURAL_SUPPORT_FRACTION] =
        analysis_diagnostics;

    for (uint rb = 0; rb < NUM_BINS; rb++) {
        SubjectHist[rb] = 0u;
        PlainHist[rb] = 0u;
    }
    PlainHist[NUM_BINS] = 0u;
    PlainHist[NUM_BINS + 1] = 0u;
    PlainHist[NUM_BINS + 2] = 0u;
    PlainHist[NUM_BINS + 3] = 0u;
    PlainHist[NUM_BINS + 4] = 0u;
    PlainHist[NUM_BINS + 5] = 0u;
    PlainHist[NUM_BINS + 6] = 0u;
}
