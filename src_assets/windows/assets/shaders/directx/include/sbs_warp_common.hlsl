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
    float warp_pad0;
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

// Signed parallax in source UV units. Positive values move near content right in the left eye and
// left in the right eye. SubjectState[0] is {recenter_delta, subject_curve, subject_depth, init};
// SubjectState[1] is {stretch_lo, stretch_inv_range, _, _}.
float DepthParallax(float d, float x, float4 s0, float4 s1, bool shaped) {
    if (shaped) {
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
