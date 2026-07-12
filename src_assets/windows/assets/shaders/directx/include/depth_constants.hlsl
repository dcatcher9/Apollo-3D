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
    uint  color_mode;        // 0=sRGB SDR, 1=linear SDR, 2=linear scRGB HDR
    float ema_alpha;         // per-pixel depth temporal EMA blend (buffer_to_tex)
    float minmax_alpha;      // normalized min/max temporal EMA blend (depth_minmax_ema)
    uint  reduce_threads;    // total threads in the reduction dispatch (grid-stride)
    float reserved6;
    float reserved7;
    float reserved8;
    float reserved9;
    float reserved10;
    float reserved11;
    float reserved12;
    float reserved0;
    float reserved1;
    float subject_recenter;  // subject recenter strength (depth_subject_resolve)
    float reserved16;
    float reserved17;
    float subject_stretch;   // > 0.5 = apply the disparity stretch (depth_subject_resolve)
    float reserved19;
};

#endif
