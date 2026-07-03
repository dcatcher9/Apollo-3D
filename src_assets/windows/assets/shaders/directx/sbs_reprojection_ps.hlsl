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
SamplerState      LinearSampler     : register(s0);

struct PS_INPUT {
    float4 Pos      : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

cbuffer Constants : register(b2) {
    float max_divergence;   // Parallax budget as a fraction of source width (e.g. 0.03). 0 = flat.
    float focal_plane;      // Zero-parallax depth in [0,1] (e.g. 0.5).
    float depth_scale;      // Linear depth-contrast gain (e.g. 1.5).
    float parallax_steps;   // Horizontal probes per eye; fewer = big GPU saving on this full-res pass.
    float border_fade;      // Ramp parallax to 0 within this fraction of the L/R edges (0 = off).
    float depth_floor;      // Far-depth compression: d' = floor + (1-floor)*d. Shrinks the
                            // foreground-vs-background parallax gap at silhouettes, and with it
                            // the disocclusion band that gets stretch-filled there. 0 = off.
    float pad1;
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

// Signed horizontal parallax for a normalized depth sample at source position x, in
// source-UV units. Positive => nearer than the focal plane => pops out of the screen.
// depth_floor first compresses the far range (d' = floor + (1-floor)*d): the visible cost of a
// depth cliff is the disocclusion band, whose width scales with |d_near - d_far|; lifting the
// far floor narrows that band without touching foreground pop.
float DepthParallax(float d, float x) {
    d = depth_floor + (1.0f - depth_floor) * d;
    return (d - focal_plane) * depth_scale * max_divergence * BorderFade(x);
}

// Silhouette-stable depth read: a 2x2 spread of taps so any depth step spans ~2.5 probe
// steps of the reprojection search, in BOTH axes. The guided-upsampled depth is texel-sharp;
// read raw, a silhouette transition is only ~1 probe step wide horizontally -- the zero-
// crossing detection then flips with probe phase pixel to pixel -- and a diagonal silhouette
// is a texel staircase vertically, so adjacent ROWS resolve the contested strip differently
// (the dotted/mesh fringe along occlusion edges). Widening the transition in x keeps g()
// continuous and the lerped crossings stable; widening in y turns the row staircase into a
// smooth diagonal. Costs ~6px of silhouette softness -- still ~10x sharper than pre-guided.
float SampleDepth(float sx, float sy) {
    uint dw, dh;
    DepthTexture.GetDimensions(dw, dh);
    float ox = 0.75f / (float) dw;
    float oy = 0.75f / (float) dh;
    return 0.25f * (DepthTexture.SampleLevel(LinearSampler, float2(sx - ox, sy - oy), 0)
                  + DepthTexture.SampleLevel(LinearSampler, float2(sx + ox, sy - oy), 0)
                  + DepthTexture.SampleLevel(LinearSampler, float2(sx - ox, sy + oy), 0)
                  + DepthTexture.SampleLevel(LinearSampler, float2(sx + ox, sy + oy), 0));
}

// Find the source U coordinate that reprojects onto `uv` for one eye, choosing the
// nearest (frontmost) surface so foreground occludes rather than duplicates.
// eyeSign = +1 right eye, -1 left eye.
float2 Reproject(float2 uv, float eyeSign) {
    // Widest distance any surface can travel (near or far side of the focal plane).
    // Must include depth_scale so the search covers the full parallax range. BorderFade only
    // shrinks parallax, so this (fade=1) remains a valid upper bound on the search span.
    float searchRadius = max_divergence * depth_scale * max(focal_plane, 1.0f - focal_plane);
    if (searchRadius <= 1e-6f) {
        return uv;  // divergence 0 -> flat passthrough, both eyes identical
    }

    int steps = (parallax_steps >= 1.0f) ? (int)parallax_steps : 8;
    float startX = uv.x - searchRadius;
    float stepX  = (2.0f * searchRadius) / (float)steps;

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
    float prevD = SampleDepth(prevX, uv.y);
    float prevG = (prevX - uv.x) - eyeSign * DepthParallax(prevD, prevX);
    if (prevD < bgDepth) { bgDepth = prevD; bgX = prevX; }

    [loop]
    for (int i = 1; i <= steps; i++) {
        float x = startX + stepX * i;
        float d = SampleDepth(x, uv.y);
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
