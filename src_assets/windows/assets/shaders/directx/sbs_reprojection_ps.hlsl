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
// Subject-tracking state from depth_subject_resolve_cs. [0] = {recenter_delta, subject_curve,
// subject_depth_ema, initialized}; [1] = {stretch_lo_val, stretch_inv_range, _, _}. Only read
// when subject_track is on; an unbound/zero buffer (initialized == 0) falls back to linear.
StructuredBuffer<float4> SubjectState : register(t2);
Texture2D<float> PlaneLockTexture : register(t4);
SamplerState      LinearSampler     : register(s0);

struct PS_INPUT {
    float4 Pos      : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

#include "include/sbs_warp_common.hlsl"

// Silhouette-stable depth read: a 2x2 spread of taps so any depth step spans ~2.5 probe
// steps of the reprojection search, in BOTH axes. The guided-upsampled depth is texel-sharp;
// read raw, a silhouette transition is only ~1 probe step wide horizontally -- the zero-
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
    // produced state (init != 0 -- it is 0 for the first frames, or when the subject shaders
    // failed to compile so the SubjectState SRV is unbound and reads 0). Decide it ONCE here
    // from the runtime state, and read the frame-uniform SubjectState once, so the probe loop
    // below issues no per-probe buffer loads (DepthParallax gets s0/s1 as args). `shaped` also
    // gates searchRadius, so the search span and the parallax mapping can never disagree.
    float4 s0 = SubjectState[0];
    float4 s1 = SubjectState[1];
    float4 s2 = SubjectState[2];
    bool shaped = (subject_track > 0.5f) && (s0.w > 0.5f);
    uint dw, dh;
    DepthTexture.GetDimensions(dw, dh);

    // Widest distance any surface can travel (near or far side of the focal plane). BorderFade
    // only shrinks parallax, so this (fade=1) remains a valid upper bound on the search span.
    // The shaped path clamps its curve to [-1, 1], so its bound is the full max_divergence
    // (2x the linear path's -- probe spacing coarsens accordingly).
    float searchRadius = shaped && bestv2_shift_profile > 0.5f
        ? Bestv2SearchRadius((float)dw)
        : shaped
        ? max_divergence
        : max_divergence * max(focal_plane, 1.0f - focal_plane);
    if (searchRadius <= 1e-6f) {
        return uv;  // divergence 0 -> flat passthrough, both eyes identical
    }

    int steps = (parallax_steps >= 1.0f) ? (int)parallax_steps : 24;
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
    float prevMask = PlaneLockTexture.SampleLevel(LinearSampler, float2(prevX, uv.y), 0);
    float prevG = (prevX - uv.x) - eyeSign * DepthParallax(prevD, prevMask, prevX, s0, s1, s2, shaped, (float)dw);
    if (prevD < bgDepth) { bgDepth = prevD; bgX = prevX; }

    [loop]
    for (int i = 1; i <= steps; i++) {
        float x = startX + stepX * i;
        float d = SampleDepth(x, uv.y, ofs);
        float planeMask = PlaneLockTexture.SampleLevel(LinearSampler, float2(x, uv.y), 0);
        float g = (x - uv.x) - eyeSign * DepthParallax(d, planeMask, x, s0, s1, s2, shaped, (float)dw);

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

    // Map this eye's half back into the source image's [0,1] horizontal range.
    float2 src_uv = uv;
    src_uv.x = is_right_eye ? (uv.x - 0.5f) * 2.0f : uv.x * 2.0f;

    float2 sample_uv = Reproject(src_uv, eyeSign);

    // Disoccluded regions clamp to the nearest valid column instead of wrapping.
    sample_uv.x = saturate(sample_uv.x);

    float4 col = LeftColorTexture.Sample(LinearSampler, sample_uv);

    // Depth-of-field (VD3D apply_dof_cuda, post-warp): blur the reprojected color by how far the
    // sampled surface sits from the focal plane -- the tracked subject depth when subject
    // anchoring is live, else the near plane (depth 1.0). Focusing the near plane keeps the
    // nearest content sharp and blurs the background (standard bokeh); focusing focal_plane (0.5)
    // instead would blur the near subject, the opposite of what DOF is for. In focus stays sharp;
    // off-focus gets a bokeh gather. Softens off-focus disocclusion fills and adds separation.
    if (dof_strength > 0.0f) {
        float4 s0 = SubjectState[0];
        float focal = (subject_track > 0.5f && s0.w > 0.5f) ? s0.z : 1.0f;
        float dsamp = DepthTexture.SampleLevel(LinearSampler, sample_uv, 0);
        float defocus = saturate(abs(dsamp - focal) / max(dof_focus_width, 1e-3f));
        if (defocus > 0.01f) {
            // Two-ring disc gather (inner 6 @ 0.5r + outer 6 @ r + center = 13 taps) for a
            // smoother bokeh than a single ring. Round via the source aspect so the radius is
            // isotropic. r is a fraction of source width; per-eye the source is sampled directly.
            uint cw, ch;
            LeftColorTexture.GetDimensions(cw, ch);
            float r = defocus * dof_strength;
            float2 rad = float2(r, r * (float) cw / (float) ch);
            float4 acc = col;
            [unroll] for (int k = 0; k < 6; k++) {
                float a = 6.2831853f * (k / 6.0f);
                float2 dir = float2(cos(a), sin(a));
                acc += LeftColorTexture.Sample(LinearSampler, sample_uv + dir * rad * 0.5f);
                acc += LeftColorTexture.Sample(LinearSampler, sample_uv + dir * rad);
            }
            col = acc / 13.0f;
        }
    }

    return col;
}
