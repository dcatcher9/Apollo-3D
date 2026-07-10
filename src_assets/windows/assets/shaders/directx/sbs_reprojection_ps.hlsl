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
SamplerState      LinearSampler     : register(s0);

struct PS_INPUT {
    float4 Pos      : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

cbuffer Constants : register(b2) {
    float max_divergence;   // Parallax gain as a fraction of source width: signed parallax =
                            // (depth - focal_plane) * max_divergence. 0 = flat.
    float focal_plane;      // Zero-parallax depth in [0,1] (e.g. 0.5).
    float parallax_steps;   // Horizontal probes per eye; fewer = big GPU saving on this full-res pass.
    float border_fade;      // Ramp parallax to 0 within this fraction of the L/R edges (0 = off).
    float depth_floor;      // Far-depth compression: d' = floor + (1-floor)*d. Shrinks the
                            // foreground-vs-background parallax gap at silhouettes, and with it
                            // the disocclusion band that gets stretch-filled there. 0 = off.
    float subject_track;    // > 0.5 = VD3D-style shaped disparity (band curve + subject anchor)
                            // instead of the linear (depth - focal_plane) mapping.
    float subject_lock;     // Fraction of the subject's own parallax subtracted everywhere
                            // (~1 pins the tracked subject to the screen plane).
    float subject_stretch;  // > 0.5 = apply the disparity stretch using SubjectState[1] bounds.
    float subject_plane_lock;  // Flatten residual disparity within the subject band (0 = off).
    float subject_plane_width; // Half-width (normalized depth) of that subject band.
    float dof_strength;     // Depth-of-field: blur radius (fraction of width) at full defocus. 0 = off.
    float dof_focus_width;  // In-focus depth half-width around the focal plane (beyond it -> full blur).
};

// Border fade: parallax ramps to zero within border_fade of the LEFT/RIGHT source edges, so
// an object touching a side edge doesn't pop out of the screen -- a "window violation" that
// causes binocular rivalry (one eye sees it clipped by the frame, the other doesn't). Only
// the L/R edges need this: parallax is purely horizontal, so the top/bottom edges clip both
// eyes identically (ordinary occlusion, no rivalry).
float BorderFade(float x) {
    return (border_fade <= 0.0f) ? 1.0f : saturate(min(x, 1.0f - x) / border_fade);
}

// BandCurve() -- the near/mid/far disparity shaping profile, shared with
// depth_subject_resolve_cs.hlsl (keep tools/warpsim/warpsim.cpp band_curve in sync).
#include "include/band_curve.hlsl"

// Signed horizontal parallax for a normalized depth sample at source position x, in
// source-UV units. Positive => nearer than the focal plane => pops out of the screen.
//
// Linear path (default): depth_floor first compresses the far range (d' = floor +
// (1-floor)*d): the visible cost of a depth cliff is the disocclusion band, whose width
// scales with |d_near - d_far|; lifting the far floor narrows that band without touching
// foreground pop.
//
// Shaped path (shaped == true): recenter depth around the tracked subject, map it through
// the near/mid/far band curve, and subtract subject_lock x the subject's own curve value so
// the subject sits at the screen plane. The residual is clamped to [-1, 1], making
// max_divergence the hard parallax bound (the probe search radius relies on this).
//
// s0 = SubjectState[0] {delta, subject_curve, subj_ema, init}; s1 = SubjectState[1]
// {stretch_lo, stretch_inv_range, _, _}. Both are frame-uniform and are read ONCE in
// Reproject (not per probe) and passed in; `shaped` is decided there too so the search
// radius and this mapping can never disagree about which path is live.
float DepthParallax(float d, float x, float4 s0, float4 s1, bool shaped) {
    if (shaped) {
        // Optional disparity stretch (VD3D shape_depth_for_pop): rescale the [lo,hi] percentile
        // band to full [0,1] so the mid-range uses the whole parallax budget. lo=0, inv_range=1
        // (stretch off) makes this an identity.
        float d_str = (subject_stretch > 0.5f) ? saturate((d - s1.x) * s1.y) : d;
        float d_shaped = saturate(d_str + s0.x);  // recenter (delta in stretched space)
        float c = BandCurve(d_shaped);
        // Local subject-plane lock (VD3D apply_subject_plane_lock): flatten the near-subject
        // surround toward the subject's own parallax, so the subject band reads as one plane.
        if (subject_plane_lock > 0.0f) {
            float t = (d - s0.z) / max(subject_plane_width, 1e-4f);
            float band = exp(-0.5f * t * t);
            c = lerp(c, s0.y, subject_plane_lock * band);
        }
        c -= subject_lock * s0.y;   // global anchor: subject to the screen plane
        return clamp(c, -1.0f, 1.0f) * max_divergence * BorderFade(x);
    }
    float df = depth_floor + (1.0f - depth_floor) * d;
    return (df - focal_plane) * max_divergence * BorderFade(x);
}

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
    bool shaped = (subject_track > 0.5f) && (s0.w > 0.5f);

    // Widest distance any surface can travel (near or far side of the focal plane). BorderFade
    // only shrinks parallax, so this (fade=1) remains a valid upper bound on the search span.
    // The shaped path clamps its curve to [-1, 1], so its bound is the full max_divergence
    // (2x the linear path's -- probe spacing coarsens accordingly).
    float searchRadius = shaped
        ? max_divergence
        : max_divergence * max(focal_plane, 1.0f - focal_plane);
    if (searchRadius <= 1e-6f) {
        return uv;  // divergence 0 -> flat passthrough, both eyes identical
    }

    int steps = (parallax_steps >= 1.0f) ? (int)parallax_steps : 24;
    float startX = uv.x - searchRadius;
    float stepX  = (2.0f * searchRadius) / (float)steps;

    // Depth-read tap spread (see SampleDepth), hoisted out of the probe loop.
    uint dw, dh;
    DepthTexture.GetDimensions(dw, dh);
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
    float prevG = (prevX - uv.x) - eyeSign * DepthParallax(prevD, prevX, s0, s1, shaped);
    if (prevD < bgDepth) { bgDepth = prevD; bgX = prevX; }

    [loop]
    for (int i = 1; i <= steps; i++) {
        float x = startX + stepX * i;
        float d = SampleDepth(x, uv.y, ofs);
        float g = (x - uv.x) - eyeSign * DepthParallax(d, x, s0, s1, shaped);

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
