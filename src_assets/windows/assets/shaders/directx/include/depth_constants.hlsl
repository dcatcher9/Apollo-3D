#ifndef DEPTH_CONSTANTS_HLSL
#define DEPTH_CONSTANTS_HLSL

// Shared depth-pass constant buffer (register b0). ONE canonical layout for every depth-stage
// compute shader (rgb_to_nchw, buffer_to_tex, depth_minmax_cs, depth_minmax_ema_cs,
// depth_hist_cs, depth_subject_hist_cs, depth_subject_resolve_cs). MUST match the cb[0..19]
// fill in ensure_cbuffers() in src/video_depth_estimator.cpp slot-for-slot. Each shader reads
// only the fields it needs; the rest are inert. All 20 are 4-byte scalars = exactly 5 float4
// registers, no straddle. Adding a field: append here + set the matching cbf[N] in the C++.
cbuffer Constants : register(b0) {
    uint  target_w;
    uint  target_h;
    uint  is_hdr;
    float ema_alpha;         // per-pixel depth temporal EMA blend (buffer_to_tex)
    float minmax_alpha;      // normalized min/max temporal EMA blend (depth_minmax_ema)
    uint  reduce_threads;    // total threads in the reduction dispatch (grid-stride)
    uint  output_transform;  // 0 = identity (DA-V2 disparity); 1 = shifted reciprocal (DA-V3)
    float depth_shift;       // shift in 1/(depth + depth_shift) when output_transform == 1
    float snap_ratio;        // A1 scene-cut snap ratio (depth_minmax_ema; 0 = off)
    float floor_frac;        // A3 range-floor fraction (depth_minmax_ema; 0 = off)
    float floor_ref_alpha;   // A3 reference-range decay
    float pct_lo;            // robust normalization low percentile fraction (0 = raw min)
    float pct_hi;            // robust normalization high percentile fraction (1 = raw max)
    float lock_frames;       // scene-lock: updates before the bounds freeze (0 = off)
    float locked_alpha;      // blend rate while scene-locked
    float subject_recenter;  // subject recenter strength (depth_subject_resolve)
    float stretch_lo_pct;    // shape_depth_for_pop stretch bounds (depth_subject_resolve)
    float stretch_hi_pct;
    float subject_stretch;   // > 0.5 = apply the disparity stretch (depth_subject_resolve)
    float pixel_ema_first;   // > 0.5 = pixel->range EMA order (buffer_to_tex): smooth raw disparity
                             // before normalizing, instead of EMA'ing the normalized depth
};

#endif
