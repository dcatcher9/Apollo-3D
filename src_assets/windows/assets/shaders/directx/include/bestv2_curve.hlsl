#ifndef BESTV2_CURVE_HLSL
#define BESTV2_CURVE_HLSL

#include "include/sbs_adaptive_state_contract.generated.hlsl"

// Cbuffer-independent Bestv2 shaping primitives shared by the live warp and evaluator coverage.
//
// ONE clamp, not two. saturate(saturate(x) + a) is not saturate(x + a): the inner clamp collapses
// every x < 0 onto the constant a, whereas the folded form still ramps over [-a, 0]. Because
// DepthParallax is a pure function of shaped depth plus frame-uniform scalars, each clipped group
// receives an IDENTICAL disparity -- a flat plateau with no stereo relief (cardboarding).
//
// Folding is free but only relocates the clipped window; its width is 1 either way. The clipped
// FRACTION is set by the band percentiles in depth_subject_resolve_cs (STRETCH_BAND_TAIL), and
// that is the knob that actually reduces cardboarding. A soft-knee variant of this function was
// measured and REJECTED: softening keeps the same over-clipping and charges the band interior for
// it, costing 6.3-6.4% of p10-p90 mid-scene relief and 3.6-4.1% of gated stereo volume, whereas
// widening the band cost ~0. Do not reintroduce a knee here -- widen the band instead.
float Bestv2WarpDepth(float d, float4 s0, float4 s1, bool shaped, bool stretch_enabled) {
    if (!shaped) {
        return d;
    }
    float x = stretch_enabled ?
        (d - SBS_STATE_STRETCH_LO(s1)) * SBS_STATE_STRETCH_INV_RANGE(s1) :
        d;
    return saturate(x + SBS_STATE_SUBJECT_RECENTER_DELTA(s0));
}

// Degree-7 polynomial approximation of Bestv2RawShiftPx over normalized depth [0, 1].
// Its maximum raw-shift error is below 0.01 source pixel (and below 0.014 output pixel at
// the maximum supported aspect scaling). The live full-resolution probe loop uses this form
// to replace three transcendental operations per probe.
// No input clamp: every call site already delivers [0,1]. Bestv2WarpDepth returns saturate(...)
// on its shaped path, depth_subject_resolve_cs saturates before calling, and the one unshaped
// passthrough (sbs_forward_coverage_cs) never reaches here -- DepthParallax is inside `if (shaped)`
// and the raw value it produces feeds saturate(shaped_depth) instead.
float Bestv2RawShiftPxFast(float d) {
    return -1.39635933f + d * (2.776208766f + d * (21.04503417f + d *
           (-94.6673759f + d * (376.6610774f + d * (-645.141824f + d *
           (482.8701123f - 133.5645677f * d))))));
}

#endif
