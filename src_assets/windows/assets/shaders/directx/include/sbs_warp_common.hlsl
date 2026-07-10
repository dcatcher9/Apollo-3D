#ifndef SBS_WARP_COMMON_HLSL
#define SBS_WARP_COMMON_HLSL

// Shared disparity field for both geometry implementations. Keeping this in one include is what
// makes the warp A/B meaningful: Apollo-probe and VD3D-hybrid see identical depth shaping,
// subject anchoring, parallax, and border behavior.
cbuffer Constants : register(b2) {
    float max_divergence;
    float focal_plane;
    float parallax_steps;
    float border_fade;
    float depth_floor;
    float subject_track;
    float subject_lock;
    float subject_stretch;
    float subject_plane_lock;
    float subject_plane_width;
    float dof_strength;
    float dof_focus_width;
    float vd3d_forward_blend;
    float vd3d_fill_radius;
    float bestv2_shift_profile;
    float warp_pad1;
};

#include "include/band_curve.hlsl"

float BorderFade(float x) {
    return (border_fade <= 0.0f) ? 1.0f : saturate(min(x, 1.0f - x) / border_fade);
}

// Depth after the Bestv2 percentile stretch and subject recenter. Apollo stores high=near, the
// polarity opposite VD3D's render tensor, so the mirrored operation remains high=near here.
float WarpDepth(float d, float4 s0, float4 s1, bool shaped) {
    if (!shaped) {
        return d;
    }
    float d_str = (subject_stretch > 0.5f) ? saturate((d - s1.x) * s1.y) : d;
    return saturate(d_str + s0.x);
}

// Bestv2's shift field in Apollo's sign convention (positive = pop). VisionDepth3D specifies
// the bands in SOURCE PIXELS: fg=-9*1.11, mg=-3, bg=+2.4*1.05 with low=near and negative=pop.
// Apollo is high=near and positive=pop, hence the mirrored centers and signs below. Keeping the
// result in pixels until the final division is essential: a fixed normalized divergence cannot
// reproduce the preset at more than one source resolution.
float Bestv2RawShiftPx(float d) {
    float wn = exp(-0.5f * ((d - 0.85f) / 0.24f) * ((d - 0.85f) / 0.24f));
    float wm = exp(-0.5f * ((d - 0.50f) / 0.28f) * ((d - 0.50f) / 0.28f));
    float wf = exp(-0.5f * ((d - 0.15f) / 0.24f) * ((d - 0.15f) / 0.24f));
    return (wn * 9.99f + wm * 3.0f + wf * -2.52f) / (wn + wm + wf + 1e-6f);
}

// Bestv2's cinematic_window_depth_sculpt, mirrored from VD3D low=near depth into Apollo's
// high=near convention. The preset activates this even with depth_pop_gamma=1: its
// bg_push_multiplier=1.05 produces window_strength=(1.05-1)*1.20=.06. The remaining values are
// derived directly from Bestv2: subject_hold=.35+.20*.28=.406, near_pull=.06+.06*.11=.0666,
// subject_width=.12. This is disparity shaping only; the saved pre-warp depth stays untouched.
float Bestv2WindowDepth(float d, float raw_subject_depth) {
    float rel = d - raw_subject_depth;  // high=near: positive is in front of the subject
    float far_mask = smoothstep(0.04f, 0.42f, -rel);
    float near_mask = smoothstep(0.04f, 0.32f, rel);
    float subject_mask = exp(-0.5f * (rel / 0.12f) * (rel / 0.12f));

    // Mirrored VD3D operations: pushing low=near depth upward moves high=near depth downward;
    // pulling it downward moves high=near depth upward.
    float far_boost = far_mask * 0.06f * d * 0.65f;
    float near_boost = near_mask * 0.0666f * (1.0f - d) * 0.45f;
    float sculpted = saturate(d - far_boost + near_boost);
    float hold = saturate(subject_mask * 0.406f);
    return lerp(sculpted, d, min(hold, 0.95f));
}

float Bestv2Parallax(float d, float4 s0, float4 s1, float source_width) {
    float shaped_depth = Bestv2WindowDepth(WarpDepth(d, s0, s1, true), s0.z);
    float shift_px = Bestv2RawShiftPx(shaped_depth);
    // VD3D computes its subject anchor from subject_depth_track directly, not from the
    // stretched/recentered/sculpted copy used for the per-pixel disparity field.
    float subject_shift_px = Bestv2RawShiftPx(s0.z);

    // Bestv2 apply_subject_plane_lock(.28, width=.12), expressed on the shift field. Apollo's
    // current depth-band mask is the real-time equivalent; concealment/silhouette morphology is
    // deliberately a later controlled step.
    if (subject_plane_lock > 0.0f) {
        float t = (d - s0.z) / max(subject_plane_width, 1e-4f);
        shift_px = lerp(shift_px, subject_shift_px, subject_plane_lock * exp(-0.5f * t * t));
    }

    // Exact Bestv2 preset values: parallax_balance=.35, subject_lock=.95 (runtime parameter),
    // zero_parallax_strength=.008, convergence_strength=.006 with dynamic convergence enabled.
    float parallax = (shift_px - subject_lock * subject_shift_px) * 0.35f / source_width;
    parallax -= 0.008f * 0.5f;
    // s1.z is VD3D's ConvergenceEMA(alpha=.90) of (low-near subject depth * .006).
    parallax += s1.z * 4.0f / source_width;
    return clamp(parallax, -0.071f, 0.071f);
}

float Bestv2SearchRadius(float source_width) {
    // Conservative bound for the pixel bands + zero-parallax trim + convergence. The preset's
    // 7.1% clamp is a safety limit, not the normal search span; using it directly would make the
    // fixed probe count too coarse at high resolution.
    // Worst case is one extreme band minus an oppositely signed subject band: approximately
    // 9.99 - .95*(-2.52) = 12.384 px, not merely the 9.99 px foreground amplitude.
    return 0.004f + (12.51f * 0.35f + 0.006f * 4.0f) / source_width;
}

// Signed parallax in source UV units. Positive values move near content right in the left eye and
// left in the right eye. SubjectState[0] is {recenter_delta, subject_curve, subject_depth, init};
// SubjectState[1] is {stretch_lo, stretch_inv_range, _, _}.
float DepthParallax(float d, float x, float4 s0, float4 s1, bool shaped, float source_width) {
    if (shaped) {
        if (bestv2_shift_profile > 0.5f) {
            return Bestv2Parallax(d, s0, s1, source_width) * BorderFade(x);
        }
        float c = BandCurve(WarpDepth(d, s0, s1, true));
        if (subject_plane_lock > 0.0f) {
            float t = (d - s0.z) / max(subject_plane_width, 1e-4f);
            float band = exp(-0.5f * t * t);
            c = lerp(c, s0.y, subject_plane_lock * band);
        }
        c -= subject_lock * s0.y;
        return clamp(c, -1.0f, 1.0f) * max_divergence * BorderFade(x);
    }
    float df = depth_floor + (1.0f - depth_floor) * d;
    return (df - focal_plane) * max_divergence * BorderFade(x);
}

#endif
