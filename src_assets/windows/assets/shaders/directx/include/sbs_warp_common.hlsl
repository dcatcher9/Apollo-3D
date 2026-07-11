#ifndef SBS_WARP_COMMON_HLSL
#define SBS_WARP_COMMON_HLSL

#include "include/bestv2_curve.hlsl"

// Shared disparity field for both geometry implementations. Keeping this in one include is what
// makes the warp A/B meaningful: Apollo-probe and VD3D-hybrid see identical depth shaping,
// subject anchoring, parallax, and border behavior.
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
    float vd3d_forward_blend;
    float reserved13;
    float reserved14;
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

// Depth after the Bestv2 percentile stretch and subject recenter. Apollo stores high=near, the
// polarity opposite VD3D's render tensor, so the mirrored operation remains high=near here.
float WarpDepth(float d, float4 s0, float4 s1, bool shaped) {
    return Bestv2WarpDepth(d, s0, s1, shaped, subject_stretch > 0.5f);
}

float Bestv2Parallax(float d, float plane_mask, float4 s0, float4 s1, float4 s2, float source_width) {
    float shaped_depth = WarpDepth(d, s0, s1, true);
    float subject_depth = WarpDepth(s0.z, s0, s1, true);
    float shift_px = Bestv2RawShiftPx(shaped_depth);
    float subject_shift_px = Bestv2RawShiftPx(subject_depth);

    // Fallback used only if exact morphology could not initialize. The normal Bestv2 path below
    // consumes its center-weighted, closed and smoothed silhouette plus weighted mean shift.
    if (subject_plane_lock > 0.0f && s2.y <= 0.5f) {
        float t = (d - s0.z) / max(subject_plane_width, 1e-4f);
        shift_px = lerp(shift_px, subject_shift_px, subject_plane_lock * exp(-0.5f * t * t));
    }

    // Exact Bestv2 preset values: parallax_balance=.35, subject_lock=.95 (runtime parameter),
    // zero_parallax_strength=.008, convergence_strength=.006 with dynamic convergence enabled.
    float parallax = (shift_px - subject_lock * subject_shift_px) * 0.35f / source_width;
    parallax -= 0.008f * 0.5f;
    if (subject_plane_lock > 0.0f && s2.y > 0.5f) {
        float correction_mask = pow(saturate(plane_mask * subject_plane_lock), 0.75f);
        float subject_mean = (s2.x - subject_lock * subject_shift_px) * 0.35f / source_width;
        subject_mean -= 0.008f * 0.5f;
        parallax -= subject_mean * correction_mask;
    }
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

// Signed Bestv2 parallax in source UV units. Before subject state initializes, return zero rather
// than falling back to the removed legacy divergence/focal-plane field.
float DepthParallax(float d, float plane_mask, float x, float4 s0, float4 s1, float4 s2,
                    bool shaped, float source_width) {
    return shaped ? Bestv2Parallax(d, plane_mask, s0, s1, s2, source_width) : 0.0f;
}

#endif
