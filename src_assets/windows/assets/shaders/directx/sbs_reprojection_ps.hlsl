// SBS 3D synthesis: reproject a mono color image + per-pixel depth into a
// full side-by-side stereo pair.
//
// Instead of the classic single blind horizontal shift (which duplicates
// foreground edges -> "double images" and can't resolve occlusion), this does an
// occlusion-aware backward search per eye: for each output pixel we scan the
// horizontal line, find every source pixel whose depth-driven parallax reprojects
// it onto this pixel, and keep the FRONTMOST one. Foreground therefore correctly
// hides background instead of ghosting, and disoccluded gaps fall back to the
// nearest edge instead of smearing a duplicate.
//
// Inputs (unchanged contract with display_vram.cpp):
//   t0 = mono color, t1 = normalized depth in [0,1) (1-exp(-raw*0.1), high = near),
//   s0 = linear clamp sampler, b2 = tuning constants below.

Texture2D<float4> LeftColorTexture : register(t0);
Texture2D<float>  DepthTexture      : register(t1);
// Subject-tracking state from depth_subject_resolve_cs. [0] = {recenter_delta, reserved,
// subject_depth_ema, initialized}; [1] = {stretch_lo_val, stretch_inv_range, _, _}.
StructuredBuffer<float4> SubjectState : register(t2);
// Bound only by the offline harness mask pass. It is produced by forward-splatting the exact
// shared parallax field, exposing holes that this backward gather necessarily paints over.
Texture2D<uint> ForwardCoverageTexture : register(t3);
Texture2D<float> PlaneLockTexture : register(t4);
SamplerState      LinearSampler     : register(s0);

struct PS_INPUT {
    float4 Pos      : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

#include "include/sbs_warp_common.hlsl"

// Silhouette-stable depth read: a 2x2 spread of taps so any depth step spans ~2.5 probe
// steps of the reprojection search, in BOTH axes. The normalized model depth is texel-sharp;
// read directly, a silhouette transition is only ~1 probe step wide horizontally -- the zero-
// crossing detection then flips with probe phase pixel to pixel -- and a diagonal silhouette
// is a texel staircase vertically, so adjacent ROWS resolve the contested strip differently
// (the dotted/mesh fringe along occlusion edges). Widening the transition in x keeps g()
// continuous and the lerped crossings stable; widening in y turns the row staircase into a
// smooth diagonal. Costs ~6px of silhouette softness -- still ~10x sharper than pre-guided.
// ofs = 0.75 / depth texture dims, computed once per pixel in Reproject.
float SampleDepth(float sx, float sy, float2 ofs) {
    return 0.25f * (DepthTexture.SampleLevel(LinearSampler, float2(sx - ofs.x, sy - ofs.y), 0)
                  + DepthTexture.SampleLevel(LinearSampler, float2(sx + ofs.x, sy - ofs.y), 0)
                  + DepthTexture.SampleLevel(LinearSampler, float2(sx - ofs.x, sy + ofs.y), 0)
                  + DepthTexture.SampleLevel(LinearSampler, float2(sx + ofs.x, sy + ofs.y), 0));
}

// Find the source U coordinate that reprojects onto `uv` for one eye, choosing the
// nearest (frontmost) surface so foreground occludes rather than duplicates.
// eyeSign = +1 right eye, -1 left eye.
float2 Reproject(float2 uv, float eyeSign) {
    // Subject anchoring is live this frame only if configured AND the resolve pass has
    // produced state (init != 0 -- it is 0 for the first frames). Mandatory shader/resource
    // initialization is validated before the estimator is published. Decide it ONCE here
    // from the runtime state, and read the frame-uniform SubjectState once, so the probe loop
    // below issues no per-probe buffer loads (DepthParallax gets s0/s1 as args). `shaped` also
    // gates searchRadius, so the search span and the parallax mapping can never disagree.
    float4 s0 = SubjectState[0];
    float4 s1 = SubjectState[1];
    float4 s2 = SubjectState[2];
    bool shaped = s0.w > 0.5f;
    uint dw, dh;
    DepthTexture.GetDimensions(dw, dh);
    // Bestv2's calibrated bands are SOURCE-COLOR pixel shifts. Normalizing by the smaller
    // inference-depth width amplified Apollo whenever the model texture was downscaled, while
    // VD3D correctly used the eye/source width. Depth dimensions remain correct for tap offsets.
    uint sourceWidth, sourceHeight;
    LeftColorTexture.GetDimensions(sourceWidth, sourceHeight);

    float aspectScale = Bestv2AspectScale(
        (float)sourceWidth, (float)sourceHeight, literal_bestv2);
    float searchRadius = shaped ? Bestv2SearchRadius((float)sourceWidth, (float)sourceHeight) : 0.0f;
    if (searchRadius <= 1e-6f) {
        return uv;  // subject state is not initialized yet
    }

    int steps = clamp((int)round(24.0f * aspectScale), 12, 72);
    float startX = uv.x - searchRadius;
    float stepX  = (2.0f * searchRadius) / (float)steps;

    // Depth-read tap spread (see SampleDepth), hoisted out of the probe loop.
    float2 ofs = float2(0.75f / (float) dw, 0.75f / (float) dh);

    // A source at position x forward-warps to out(x) = x - eyeSign * parallax(depth(x)).
    // We want out(x) == uv.x, i.e. g(x) = (x - uv.x) - eyeSign * parallax(depth(x)) == 0.
    // Marching x and watching g() cross zero locates each reprojecting source; we keep
    // the one with the greatest depth (closest to the viewer).
    float bestX     = uv.x;
    float bestDepth = -1.0f;

    // Track the FARTHEST (lowest depth = background) sample in the window as well. When
    // no source reprojects onto uv (a disocclusion / hole revealed by a moving foreground
    // edge), we fill from this nearest background instead of sampling straight through,
    // which would smear the foreground colour into the gap -- the bright "white edge"
    // that shimmers on motion.
    float bgX     = uv.x;
    float bgDepth = 2.0f;  // above any normalized depth (<= 1)

    float prevX = startX;
    float prevD = SampleDepth(prevX, uv.y, ofs);
    float planeMask = 0.0f;
    if (subject_plane_lock > 0.0f) {
        // The exact mask is 13x13-smoothed and intentionally low-frequency. Sample it once at
        // the destination rather than once per probe; the uniform branch removes all fetches
        // from both shipping profiles where plane lock is disabled.
        planeMask = PlaneLockTexture.SampleLevel(LinearSampler, uv, 0);
    }
    float prevG = (prevX - uv.x) - eyeSign * DepthParallax(
        prevD, planeMask, s0, s1, s2, shaped, (float)sourceWidth, (float)sourceHeight);
    if (prevD < bgDepth) { bgDepth = prevD; bgX = prevX; }

    [loop]
    for (int i = 1; i <= steps; i++) {
        float x = startX + stepX * i;
        float d = SampleDepth(x, uv.y, ofs);
        float g = (x - uv.x) - eyeSign * DepthParallax(
            d, planeMask, s0, s1, s2, shaped, (float)sourceWidth, (float)sourceHeight);

        // Zero crossing between prevX and x => a source in this span reprojects onto uv.
        if ((prevG <= 0.0f && g >= 0.0f) || (prevG >= 0.0f && g <= 0.0f)) {
            float denom = g - prevG;
            float t = (abs(denom) > 1e-6f) ? saturate(-prevG / denom) : 0.0f;
            float crossX = lerp(prevX, x, t);
            float crossD = lerp(prevD, d, t);
            if (crossD > bestDepth) {
                bestDepth = crossD;
                bestX     = crossX;
            }
        }

        if (d < bgDepth) { bgDepth = d; bgX = x; }

        prevX = x;
        prevD = d;
        prevG = g;
    }

    // Valid surface found -> use it; otherwise fill the hole with nearest background.
    float outX = (bestDepth >= 0.0f) ? bestX : bgX;
    return float2(outX, uv.y);
}

float4 main_ps(PS_INPUT input) : SV_TARGET {
    float2 uv = input.TexCoord;

    // Full SBS: left half (0..0.5) = left eye, right half (0.5..1) = right eye.
    bool is_right_eye = uv.x > 0.5f;
    float eyeSign = is_right_eye ? 1.0f : -1.0f;

    // Map this eye's half into its own aspect-fitted source rectangle. Bars are identical in both
    // eyes, so aspect conversion cannot introduce a global stereo offset.
    float2 output_uv = uv;
    output_uv.x = is_right_eye ? (uv.x - 0.5f) * 2.0f : uv.x * 2.0f;
    float2 src_uv;
    if (!ContentToSourceUV(output_uv, src_uv)) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float2 sample_uv = Reproject(src_uv, eyeSign);

    // Disoccluded regions clamp to the nearest valid column instead of wrapping.
    sample_uv.x = saturate(sample_uv.x);

    float4 col = LeftColorTexture.Sample(LinearSampler, sample_uv);

    return col;
}

// Harness-only diagnostic output. R marks exact forward-coverage disocclusion before this
// backward gather paints over it. G is zero because Apollo always returns a sampled color rather
// than leaving an output pixel unresolved. Bars are not content and remain unmarked. Compiling
// this separate entry point adds no live-stream work.
float4 mask_ps(PS_INPUT input) : SV_TARGET {
    float2 uv = input.TexCoord;
    bool is_right_eye = uv.x > 0.5f;
    float eyeSign = is_right_eye ? 1.0f : -1.0f;
    float2 output_uv = uv;
    output_uv.x = is_right_eye ? (uv.x - 0.5f) * 2.0f : uv.x * 2.0f;
    float2 src_uv;
    if (!ContentToSourceUV(output_uv, src_uv)) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    uint full_w, full_h;
    ForwardCoverageTexture.GetDimensions(full_w, full_h);
    uint2 output_px = min((uint2)input.Pos.xy, uint2(full_w - 1u, full_h - 1u));
    float hole = ForwardCoverageTexture.Load(int3(output_px, 0)) == 0u ? 1.0f : 0.0f;
    return float4(hole, 0.0f, 0.0f, 1.0f);
}
