#ifndef SBS_WARP_COMMON_HLSL
#define SBS_WARP_COMMON_HLSL

#include "include/bestv2_curve.hlsl"

// The validated core clips and Bestv2-derived profile were calibrated at 854 source pixels wide.
// The reference preset expresses disparity as literal render pixels, which becomes imperceptible on a 5120px
// desktop. Preserve exact behavior at and below the calibration raster, but scale wider sources
// so disparity remains a constant percentage of each eye instead of a constant pixel count.
// Non-literal production geometry also applies the independent reference-aspect correction below.
// Consequently, a sub-854 non-16:9 source preserves Bestv2's literal pixel shift and then scales
// that shift for its aspect; this is deliberate and must be changed only with a dedicated 4:3 A/B.
static const float BESTV2_CALIBRATION_WIDTH = 854.0f;
// The approved headset look was tuned with a 5120x2160 stream request. Normalize source-UV
// disparity by reference/current per-eye aspect so the same scene keeps its calibrated stereo
// strength when the client requests a different image shape. This is image geometry; it does not
// depend on the client's physical panel size or placement.
static const float BESTV2_REFERENCE_ASPECT = 5120.0f / 2160.0f;

float Bestv2AspectScale(float source_width, float source_height, float literal_mode) {
    if (literal_mode > 0.5f) {
        return 1.0f;
    }
    float aspect = max(source_width / max(source_height, 1.0f), 1e-4f);
    // Bound pathological custom modes while covering the validated portrait-like 3552x3840 case.
    return clamp(BESTV2_REFERENCE_ASPECT / aspect, 0.5f, 3.0f);
}

float Bestv2ParallaxWidth(float source_width, float literal_mode) {
    if (literal_mode > 0.5f) {
        return max(source_width, 1.0f);
    }
    return min(max(source_width, 1.0f), BESTV2_CALIBRATION_WIDTH);
}

// Apollo's shared depth-shaping and disparity contract.
cbuffer Constants : register(b2) {
    float reserved0;
    float reserved1;
    float reserved2;
    float reserved3;
    float reserved4;
    float _subject_track_reserved;  // layout compatibility; Bestv2 subject shaping is permanent
    float subject_lock;
    float subject_stretch;
    float subject_plane_lock;
    float subject_plane_width;
    float content_scale_x;       // source content width / output-eye width (per-eye letterbox)
    float content_scale_y;       // source content height / output-eye height
    float reserved12;
    float pop_strength;          // final production stereo-parallax multiplier
    float literal_bestv2;       // harness-only: bypass production resolution/aspect/pop scaling
    float source_to_output;      // output-content pixels per mono-source pixel
};

// Map one eye's output UV into the mono source. Letterbox/pillarbox is applied independently in
// each eye, preventing a packed-frame viewport offset from becoming a false stereo disparity.
bool ContentToSourceUV(float2 output_uv, out float2 source_uv) {
    float2 scale = max(float2(content_scale_x, content_scale_y), float2(1e-6f, 1e-6f));
    float2 lo = 0.5f * (float2(1.0f, 1.0f) - scale);
    float2 hi = lo + scale;
    if (any(output_uv < lo) || any(output_uv > hi)) {
        source_uv = float2(0.0f, 0.0f);
        return false;
    }
    source_uv = saturate((output_uv - lo) / scale);
    return true;
}

// Depth after the Bestv2 percentile stretch and subject recenter. Apollo stores high=near.
float WarpDepth(float d, float4 s0, float4 s1, bool shaped) {
    return Bestv2WarpDepth(d, s0, s1, shaped, subject_stretch > 0.5f);
}

float Bestv2Parallax(float d, float plane_mask, float4 s0, float4 s1, float4 s2,
                     float source_width, float source_height, bool use_plane_lock) {
    float parallax_width = Bestv2ParallaxWidth(source_width, literal_bestv2);
    float aspect_scale = Bestv2AspectScale(source_width, source_height, literal_bestv2);
    float shaped_depth = WarpDepth(d, s0, s1, true);
    float subject_depth = WarpDepth(s0.z, s0, s1, true);
    // The live probe loop calls this once per search sample and output pixel. Its bounded
    // polynomial form avoids three exponentials per probe while remaining deeply subpixel.
    float shift_px = Bestv2RawShiftPxFast(shaped_depth);
    float subject_shift_px = Bestv2RawShiftPxFast(subject_depth);

    // Fallback used only if exact morphology could not initialize. The normal Bestv2 path below
    // consumes its center-weighted, closed and smoothed silhouette plus weighted mean shift.
    // Keep this uniform condition as a real branch. D3DCompiler otherwise flattens it and
    // executes the Gaussian exponential for every full-resolution search probe even when the
    // shipping profile has plane lock disabled.
    [branch]
    if (use_plane_lock && s2.y <= 0.5f) {
        float t = (d - s0.z) / max(subject_plane_width, 1e-4f);
        shift_px = lerp(shift_px, subject_shift_px, subject_plane_lock * exp(-0.5f * t * t));
    }

    // Exact Bestv2 preset values: parallax_balance=.35, subject_lock=.95 (runtime parameter),
    // zero_parallax_strength=.008, convergence_strength=.006 with dynamic convergence enabled.
    float parallax = (shift_px - subject_lock * subject_shift_px) * 0.35f / parallax_width;
    parallax -= 0.008f * 0.5f;
    [branch]
    if (use_plane_lock && s2.y > 0.5f) {
        float correction_mask = pow(saturate(plane_mask * subject_plane_lock), 0.75f);
        float subject_mean = (s2.x - subject_lock * subject_shift_px) * 0.35f / parallax_width;
        subject_mean -= 0.008f * 0.5f;
        parallax -= subject_mean * correction_mask;
    }
    // s1.z is Bestv2's ConvergenceEMA(alpha=.90) of (low-near subject depth * .006).
    parallax += s1.z * 4.0f / parallax_width;
    // Scale the safety bound with the same factor: 7.1% was a physical-angle limit at the
    // reference aspect, not a universal percentage of differently sized panel widths.
    float strength = literal_bestv2 > 0.5f ? 1.0f : pop_strength;
    return clamp(parallax * strength * aspect_scale,
                 -0.071f * aspect_scale, 0.071f * aspect_scale);
}

// Loop-invariant values for the shipping plane-lock-off specialization. Keeping the original
// operation groups here avoids recomputing source geometry, subject shift and convergence for
// every search probe while retaining the same Bestv2 field and safety bound.
struct Bestv2NoPlaneParams {
    float subject_shift_px;
    float parallax_scale;
    float convergence_bias;
    float output_scale;
    float clamp_abs;
};

Bestv2NoPlaneParams MakeBestv2NoPlaneParams(float4 s0, float4 s1,
                                             float source_width, float source_height) {
    Bestv2NoPlaneParams p;
    float parallax_width = Bestv2ParallaxWidth(source_width, literal_bestv2);
    float subject_depth = WarpDepth(s0.z, s0, s1, true);
    p.subject_shift_px = Bestv2RawShiftPxFast(subject_depth);
    p.parallax_scale = 0.35f / parallax_width;
    p.convergence_bias = -0.008f * 0.5f + s1.z * 4.0f / parallax_width;
    float aspect_scale = Bestv2AspectScale(source_width, source_height, literal_bestv2);
    float strength = literal_bestv2 > 0.5f ? 1.0f : pop_strength;
    p.output_scale = strength * aspect_scale;
    p.clamp_abs = 0.071f * aspect_scale;
    return p;
}

float DepthParallaxNoPlane(float d, float4 s0, float4 s1, Bestv2NoPlaneParams p) {
    float shaped_depth = WarpDepth(d, s0, s1, true);
    float shift_px = Bestv2RawShiftPxFast(shaped_depth);
    float parallax = (shift_px - subject_lock * p.subject_shift_px) * p.parallax_scale;
    parallax += p.convergence_bias;
    return clamp(parallax * p.output_scale, -p.clamp_abs, p.clamp_abs);
}

float Bestv2SearchRadius(float source_width, float source_height) {
    // Conservative bound for the pixel bands + zero-parallax trim + convergence. The preset's
    // 7.1% clamp is a safety limit, not the normal search span; using it directly would make the
    // fixed probe count too coarse at high resolution.
    // Worst case is one extreme band minus an oppositely signed subject band: approximately
    // 9.99 - .95*(-2.52) = 12.384 px, not merely the 9.99 px foreground amplitude.
    float strength = literal_bestv2 > 0.5f ? 1.0f : pop_strength;
    return Bestv2AspectScale(source_width, source_height, literal_bestv2) * strength *
           (0.004f + (12.51f * 0.35f + 0.006f * 4.0f) /
                       Bestv2ParallaxWidth(source_width, literal_bestv2));
}

// Signed Bestv2 parallax in source UV units. Before subject state initializes, return zero rather
// than falling back to the removed legacy divergence/focal-plane field.
float DepthParallax(float d, float plane_mask, float4 s0, float4 s1, float4 s2,
                    bool shaped, float source_width, float source_height, bool use_plane_lock) {
    return shaped ? Bestv2Parallax(
        d, plane_mask, s0, s1, s2, source_width, source_height, use_plane_lock) : 0.0f;
}

#endif
