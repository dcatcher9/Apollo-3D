#ifndef DEPTH_CONSTANTS_HLSL
#define DEPTH_CONSTANTS_HLSL

// Shared depth-pass constant buffer (register b0). This is the single layout used by inference
// preprocessing, private normalized depth, and the coordinate-V2 producer. It must match the
// 16 words uploaded by ensure_cbuffers() in src/video_depth_estimator.cpp slot-for-slot.
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
    uint  analysis_content_left;
    uint  analysis_content_top;
    uint  analysis_content_right;
    uint  analysis_content_bottom;
    uint3 analysis_content_reserved;
};

bool DepthAnalysisContentValid() {
    return target_w > 0u && target_h > 0u &&
        analysis_content_left < analysis_content_right &&
        analysis_content_top < analysis_content_bottom &&
        analysis_content_right <= target_w && analysis_content_bottom <= target_h &&
        all(analysis_content_reserved == uint3(0u, 0u, 0u));
}

float2 DepthAnalysisContentScale() {
    return float2(
        analysis_content_right - analysis_content_left,
        analysis_content_bottom - analysis_content_top) /
        max(float2(target_w, target_h), float2(1.0f, 1.0f));
}

float2 DepthAnalysisContentMinUv() {
    return float2(analysis_content_left, analysis_content_top) /
        max(float2(target_w, target_h), float2(1.0f, 1.0f));
}

uint4 DepthAnalysisContentCells() {
    if (!DepthAnalysisContentValid() || target_w == 0u || target_h == 0u) {
        return uint4(0u, 0u, 0u, 0u);
    }
    return uint4(
        analysis_content_left,
        analysis_content_top,
        analysis_content_right,
        analysis_content_bottom);
}

bool DepthAnalysisCellIsContent(uint2 position) {
    uint4 content = DepthAnalysisContentCells();
    return position.x >= content.x && position.y >= content.y &&
        position.x < content.z && position.y < content.w;
}

uint2 DepthAnalysisClampCell(uint2 position) {
    uint4 content = DepthAnalysisContentCells();
    return clamp(position, content.xy, max(content.zw, 1u) - 1u);
}

float DepthAnalysisContentWidthCells() {
    return max((float)(analysis_content_right - analysis_content_left), 1.0f);
}

// Spatial depth-gradient thresholds are calibrated on the patch-aligned 434-texel short side.
// Derive the transfer scale from the ACTUAL resolved tensor dimensions, not depth_short_side:
// profile/native caps can resolve a different grid (for example 392 or 420). Multiplying a
// finite difference by current/reference expresses it in 434-reference-texel units. The 434 grid
// is exactly identity, preserving its calibrated shader arithmetic.
#define DEPTH_GRADIENT_REFERENCE_SHORT_SIDE 434.0f
float DepthReferenceTexelScale() {
    uint4 content = DepthAnalysisContentCells();
    return (float)min(content.z - content.x, content.w - content.y) /
        DEPTH_GRADIENT_REFERENCE_SHORT_SIDE;
}

#endif
