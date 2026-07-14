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
    float subject_lock;
    float subject_stretch;
    float content_scale_x;       // source content width / output-eye width (per-eye letterbox)
    float content_scale_y;       // source content height / output-eye height
    float pop_strength;          // final production stereo-parallax multiplier
    float literal_bestv2;        // harness-only: bypass production resolution/aspect/pop scaling
    float padding0;
    float padding1;
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

// Loop-invariant values for the production warp. Keeping the original
// operation groups here avoids recomputing source geometry, subject shift and convergence for
// every search probe while retaining the same Bestv2 field and safety bound.
struct Bestv2Params {
    float subject_shift_px;
    float parallax_scale;
    float convergence_bias;
    float output_scale;
    float clamp_abs;
};

Bestv2Params MakeBestv2Params(float4 s0, float4 s1,
                              float source_width, float source_height,
                              bool use_subject_stretch) {
    Bestv2Params p;
    float parallax_width = Bestv2ParallaxWidth(source_width, literal_bestv2);
    float subject_depth = Bestv2WarpDepth(s0.z, s0, s1, true, use_subject_stretch);
    p.subject_shift_px = Bestv2RawShiftPxFast(subject_depth);
    p.parallax_scale = 0.35f / parallax_width;
    p.convergence_bias = -0.008f * 0.5f + s1.z * 4.0f / parallax_width;
    float aspect_scale = Bestv2AspectScale(source_width, source_height, literal_bestv2);
    float strength = literal_bestv2 > 0.5f ? 1.0f : pop_strength;
    p.output_scale = strength * aspect_scale;
    p.clamp_abs = 0.071f * aspect_scale;
    return p;
}

float DepthParallax(float d, float4 s0, float4 s1, Bestv2Params p,
                    bool use_subject_stretch) {
    float shaped_depth = Bestv2WarpDepth(d, s0, s1, true, use_subject_stretch);
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

#endif
