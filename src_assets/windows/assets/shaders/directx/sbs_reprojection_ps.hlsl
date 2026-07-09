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
// Subject-tracking state from depth_subject_resolve_cs: {recenter_delta, subject_curve,
// subject_depth_ema, initialized}. Only read when subject_track is on; an unbound/zero
// buffer (initialized == 0) falls back to the linear mapping below.
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
    float pad2;
};

// Border fade: parallax ramps to zero within border_fade of the LEFT/RIGHT source edges, so
// an object touching a side edge doesn't pop out of the screen -- a "window violation" that
// causes binocular rivalry (one eye sees it clipped by the frame, the other doesn't). Only
// the L/R edges need this: parallax is purely horizontal, so the top/bottom edges clip both
// eyes identically (ordinary occlusion, no rivalry).
float BorderFade(float x) {
    return (border_fade <= 0.0f) ? 1.0f : saturate(min(x, 1.0f - x) / border_fade);
}

// VD3D's near/mid/far Gaussian disparity bands, translated to Apollo's high=near depth
// (band centers mirrored) and normalized so the near band peaks at +1 (positive = pops out;
// divergence is the master gain). Amplitudes from the VD3D Bestv2 preset: fg -9*1.11,
// mg -3, bg +2.4*1.05 px in VD3D's negative=pop convention -> +1 / +0.300 / -0.252 here.
// MUST stay identical to BandCurve in depth_subject_resolve_cs.hlsl (no #include support
// in the runtime-compiled shaders) and to band_curve in tools/warpsim/warpsim.cpp.
float BandCurve(float d) {
    float wn = exp(-0.5f * ((d - 0.85f) / 0.24f) * ((d - 0.85f) / 0.24f));
    float wm = exp(-0.5f * ((d - 0.50f) / 0.28f) * ((d - 0.50f) / 0.28f));
    float wf = exp(-0.5f * ((d - 0.15f) / 0.24f) * ((d - 0.15f) / 0.24f));
    return (wn * 1.0f + wm * 0.300f + wf * -0.252f) / (wn + wm + wf + 1e-6f);
}

// Signed horizontal parallax for a normalized depth sample at source position x, in
// source-UV units. Positive => nearer than the focal plane => pops out of the screen.
//
// Linear path (default): depth_floor first compresses the far range (d' = floor +
// (1-floor)*d): the visible cost of a depth cliff is the disocclusion band, whose width
// scales with |d_near - d_far|; lifting the far floor narrows that band without touching
// foreground pop.
//
// Shaped path (subject_track): recenter depth around the tracked subject, map it through
// the near/mid/far band curve, and subtract subject_lock x the subject's own curve value
// so the subject sits at the screen plane. The residual is clamped to [-1, 1], making
// max_divergence the hard parallax bound (the probe search radius relies on this).
float DepthParallax(float d, float x) {
    if (subject_track > 0.5f) {
        float4 s = SubjectState[0];
        if (s.w > 0.5f) {
            float c = BandCurve(saturate(d + s.x)) - subject_lock * s.y;
            return clamp(c, -1.0f, 1.0f) * max_divergence * BorderFade(x);
        }
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
    // Widest distance any surface can travel (near or far side of the focal plane).
    // BorderFade only shrinks parallax, so this (fade=1) remains a valid upper bound on
    // the search span. The shaped path clamps its curve to [-1, 1], so its bound is the
    // full max_divergence (2x the linear path's -- probe spacing coarsens accordingly).
    float searchRadius = (subject_track > 0.5f)
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
    float prevG = (prevX - uv.x) - eyeSign * DepthParallax(prevD, prevX);
    if (prevD < bgDepth) { bgDepth = prevD; bgX = prevX; }

    [loop]
    for (int i = 1; i <= steps; i++) {
        float x = startX + stepX * i;
        float d = SampleDepth(x, uv.y, ofs);
        float g = (x - uv.x) - eyeSign * DepthParallax(d, x);

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

    return LeftColorTexture.Sample(LinearSampler, sample_uv);
}
