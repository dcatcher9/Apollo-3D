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
    float zero_plane_mode;   // 0 legacy, 1 subject, 2 median, 3 background (shot-latched)
    float padding1;
    float padding2;
};

#endif
