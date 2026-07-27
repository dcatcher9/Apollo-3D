#ifndef DEPTH_CONSTANTS_HLSL
#define DEPTH_CONSTANTS_HLSL

// Shared depth-pass constant buffer (register b0). ONE canonical layout for every depth-stage
// compute shader (rgb_to_nchw, buffer_to_tex, depth_minmax_cs, depth_minmax_ema_cs,
// depth_hist_cs, depth_subject_hist_cs, depth_subject_resolve_cs). MUST match the cb[0..15]
// fill in ensure_cbuffers() in src/video_depth_estimator.cpp slot-for-slot. Each shader reads
// only the fields it needs; the rest are inert. The 16 scalars occupy exactly 4 float4 registers.
// Adding a field: append here + set the matching C++ slot.
cbuffer Constants : register(b0) {
    uint  target_w;
    uint  target_h;
    uint  color_mode;        // 0=sRGB SDR, 1=linear SDR, 2=linear scRGB HDR
    float ema_alpha;         // per-pixel depth temporal EMA blend (buffer_to_tex)
    float minmax_alpha;      // normalized min/max temporal EMA blend (depth_minmax_ema)
    uint  reduce_threads;    // total threads in the reduction dispatch (grid-stride)
    float ema_edge_change;   // >0 enables motion-edge snap; normalized depth delta threshold
    float ema_edge_gradient; // minimum current normalized-depth gradient
    float ema_edge_strength; // extra blend toward current depth inside the motion mask
    float subject_recenter;  // subject recenter strength (depth_subject_resolve)
    float subject_stretch;   // > 0.5 = apply the disparity stretch (depth_subject_resolve)
    float adaptive_pop;      // > 0.5 = maintain a scene-risk pop multiplier in SubjectState[1].w
    float adaptive_pop_max_ratio; // absolute configured ceiling / base pop strength
    float zero_plane_mode;   // 1 subject, 2 median, 3 background (shot-latched); 0 was legacy
    float padding1;
    float padding2;
};

// Spatial depth-gradient thresholds are calibrated on the patch-aligned 434-texel short side.
// Derive the transfer scale from the ACTUAL resolved tensor dimensions, not depth_short_side:
// profile/native caps can resolve a different grid (for example 392 or 420). Multiplying a
// finite difference by current/reference expresses it in 434-reference-texel units. The 434 grid
// is exactly identity, preserving its calibrated shader arithmetic.
#define DEPTH_GRADIENT_REFERENCE_SHORT_SIDE 434.0f
float DepthReferenceTexelScale() {
    return (float)min(target_w, target_h) / DEPTH_GRADIENT_REFERENCE_SHORT_SIDE;
}

// Adaptive-pop edge risk is accumulated in fixed point because InterlockedAdd is integer-only.
// Each qualifying texel contributes min(reference_grad / 0.02,
// EDGE_WEIGHT_MAX * current/reference) * EDGE_WEIGHT_SCALE. Below the cap, the smaller weight of
// a coarser grid cancels its larger one-texel silhouette fraction. The cap needs the same scale or
// saturated discontinuities would still change risk with resolution. The consumer recovers a
// threshold-equivalent edge fraction by dividing the sum by EDGE_WEIGHT_SCALE * texel count.
// Shared here because the producer (depth_subject_hist_cs) and the consumer
// (depth_subject_resolve_cs) must agree exactly: a mismatch silently rescales scene risk and
// changes which scenes receive extra pop. Do not duplicate these values.
#define EDGE_GRADIENT_THRESHOLD 0.02f
#define EDGE_WEIGHT_SCALE 256.0f
#define EDGE_WEIGHT_MAX 8.0f

// Scene-cut evidence contract. A broad RGB delta finds frame-wide appearance replacement, while
// the ordinal producer compares all ten pairwise orderings in a center/left/right/up/down stencil
// of capture-domain, point-sampled max(R,G,B). Sampling before the model's luminance-coupled HDR
// tonemapper and before spatial filtering is part of the contract: maxRGB then commutes with an
// identical monotone curve on every channel. Clipping creates ties, which abstain because a
// comparison must have this much contrast in BOTH frames. Requiring BOTH frame-wide RGB
// replacement and ordinal structure rejects flashes/exposure and ordinary detailed motion
// respectively. Depth geometry still corroborates the proposal. Conversely, broad RGB
// replacement with ordinal evidence inside a narrower QUIET band explicitly vetoes
// standalone/relative depth authority on that transition: a neural-depth jump caused by exposure
// or HDR tone mapping is not scene geometry and must not reset the zero plane. That exposure
// classification is evidence-backed only when enough texels have reliable structure in the
// current frame, history frame, and their intersection. Reliable structure disappearing is held
// out separately regardless of raw color distance: preserve the last reliable appearance/depth
// history for one update. Only a near-exact supported endpoint return is suppressed, so a clipped
// flash returning to the same shot cannot reset state. A second low-structure update releases the
// veto and lets preserved A-vs-current depth decide, while a photometrically similar but
// geometrically different supported return also remains visible. The band between quiet and
// proposal entry remains ambiguous and preserves standalone geometry authority.
#define RAW_RGB_PIXEL_DELTA 0.20f
#define RAW_RGB_CUT_HIGH 0.70f
#define STRUCTURAL_ORDINAL_CONTRAST_FLOOR 0.01f
#define STRUCTURAL_ORDINAL_MIN_COMMON 4u
#define STRUCTURAL_ORDINAL_MIN_FLIPS 2u
#define STRUCTURAL_COLOR_MIN_SUPPORT 0.01f
#define STRUCTURAL_COLOR_EXPOSURE_QUIET 0.01f
#define STRUCTURAL_COLOR_CUT_HIGH 0.03f
// Fraction of model texels allowed to differ by RAW_RGB_PIXEL_DELTA while still calling a
// structureless-gap return the same visible endpoint. This is deliberately far below the broad
// 70% proposal: low appearance evidence must not veto independently preserved A-vs-B geometry.
#define STRUCTURELESS_RETURN_RGB_SAME_MAX 0.01f
#define DEPTH_CUT_HIGH 0.60f
#define DEPTH_CUT_CORROBORATE 0.25f
#define DEPTH_CUT_LOW 0.10f
#define DEPTH_CUT_BASELINE_ALPHA 0.125f
#define DEPTH_CUT_RELATIVE_FLOOR 0.30f
#define DEPTH_CUT_RELATIVE_MARGIN 0.20f
#define DEPTH_CUT_RELATIVE_MULTIPLIER 2.0f

#endif
