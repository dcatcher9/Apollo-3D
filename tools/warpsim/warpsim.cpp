// Offline simulator of Apollo's sbs_reprojection_ps.hlsl, fed by the real dumped
// depth (guided, 1596x672) and source (5120x2160 crop) of a problem frame.
// Renders the right-eye arm-edge region and compares warp formulations:
//   mode 0: exact replica of the current shader (validates the sim against the dump)
//   mode 1: no 4-tap depth smoothing (raw bilinear reads)
//   mode 2: cliff-step model (bisect cliff position; emit front-side step, no phantom lerp)
//   mode 3: splat/coverage (per-probe footprint, frontmost with hysteresis, no bracketing)
// Usage: warpsim <mode> <out.raw>   (writes 640x320 RGB24)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

static const int DW = 1596, DH = 672;          // guided depth dims
static const int SW = 800, SH = 400;           // source crop dims
static const int SOX = 2000, SOY = 900;        // source crop origin in the 5120x2160 frame
static const int OW = 640, OH = 320;           // output (right-eye crop) dims
static const int EOX = 1600, EOY = 720;        // output origin in eye coords (4096x1728)

static std::vector<unsigned char> g_depth, g_src;

// --- shader constants (from sunshine.conf / config defaults) ---
// divergence folded the old depth_scale in (0.015 * 0.9 = 0.0135); the shader no longer
// has a separate depth_scale.
static const float DIVERGENCE = 0.0135f, FOCAL = 0.5f;
static const float BORDER_FADE = 0.02f, DEPTH_FLOOR = 0.25f;
static const int   STEPS = 24;
static const float EYE_SIGN = 1.0f;  // right eye

static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Bilinear sample of the depth texture (normalized uv, CLAMP addressing), like SampleLevel.
static float depth_bilinear(float u, float v) {
    float fx = u * DW - 0.5f, fy = v * DH - 0.5f;
    int x0 = (int) floorf(fx), y0 = (int) floorf(fy);
    float tx = fx - x0, ty = fy - y0;
    int x1 = x0 + 1, y1 = y0 + 1;
    x0 = std::max(0, std::min(DW - 1, x0)); x1 = std::max(0, std::min(DW - 1, x1));
    y0 = std::max(0, std::min(DH - 1, y0)); y1 = std::max(0, std::min(DH - 1, y1));
    float d00 = g_depth[y0 * DW + x0] / 255.0f, d10 = g_depth[y0 * DW + x1] / 255.0f;
    float d01 = g_depth[y1 * DW + x0] / 255.0f, d11 = g_depth[y1 * DW + x1] / 255.0f;
    return (d00 * (1 - tx) + d10 * tx) * (1 - ty) + (d01 * (1 - tx) + d11 * tx) * ty;
}

// The shader's SampleDepth: 2x2 spread of bilinear taps at +-0.75 texel.
static bool g_smooth = true;
static float sample_depth(float u, float v) {
    if (!g_smooth) return depth_bilinear(u, v);
    float ox = 0.75f / DW, oy = 0.75f / DH;
    return 0.25f * (depth_bilinear(u - ox, v - oy) + depth_bilinear(u + ox, v - oy)
                  + depth_bilinear(u - ox, v + oy) + depth_bilinear(u + ox, v + oy));
}

static float border_fade(float x) {
    return clampf(std::min(x, 1.0f - x) / BORDER_FADE, 0.0f, 1.0f);
}
// Shaped-disparity variant (sbs_3d_subject_track, 2026-07-09): the shader can replace the
// linear mapping below with BandCurve(saturate(d + delta)) - subject_lock * scurve, clamped
// to [-1,1], times DIVERGENCE (searchRadius then = DIVERGENCE, not DIVERGENCE/2). delta and
// scurve come from depth_subject_resolve_cs's SubjectState. To study that path here, set
// SUBJECT_DELTA/SUBJECT_SCURVE from a dump and route depth_parallax through band_curve.
static float band_curve(float d) {
    // MUST match BandCurve in sbs_reprojection_ps.hlsl / depth_subject_resolve_cs.hlsl.
    float wn = expf(-0.5f * ((d - 0.85f) / 0.24f) * ((d - 0.85f) / 0.24f));
    float wm = expf(-0.5f * ((d - 0.50f) / 0.28f) * ((d - 0.50f) / 0.28f));
    float wf = expf(-0.5f * ((d - 0.15f) / 0.24f) * ((d - 0.15f) / 0.24f));
    return (wn * 1.0f + wm * 0.300f + wf * -0.252f) / (wn + wm + wf + 1e-6f);
}
static const bool  SUBJECT_TRACK = false;   // flip + fill the two constants to sim the shaped path
static const float SUBJECT_DELTA = 0.0f;    // SubjectState.x from a dump
static const float SUBJECT_SCURVE = 0.0f;   // SubjectState.y from a dump
static const float SUBJECT_LOCK = 0.95f;

static float depth_parallax(float d, float x) {
    if (SUBJECT_TRACK) {
        float c = band_curve(clampf(d + SUBJECT_DELTA, 0.0f, 1.0f)) - SUBJECT_LOCK * SUBJECT_SCURVE;
        return clampf(c, -1.0f, 1.0f) * DIVERGENCE * border_fade(x);
    }
    d = DEPTH_FLOOR + (1.0f - DEPTH_FLOOR) * d;
    return (d - FOCAL) * DIVERGENCE * border_fade(x);
}

// Bilinear sample of the source crop (uv in full-frame normalized coords).
static void src_sample(float u, float v, unsigned char *rgb) {
    float fx = u * 5120.0f - SOX - 0.5f, fy = v * 2160.0f - SOY - 0.5f;
    int x0 = (int) floorf(fx), y0 = (int) floorf(fy);
    float tx = fx - x0, ty = fy - y0;
    int x1 = x0 + 1, y1 = y0 + 1;
    x0 = std::max(0, std::min(SW - 1, x0)); x1 = std::max(0, std::min(SW - 1, x1));
    y0 = std::max(0, std::min(SH - 1, y0)); y1 = std::max(0, std::min(SH - 1, y1));
    for (int c = 0; c < 3; c++) {
        float p00 = g_src[(y0 * SW + x0) * 3 + c], p10 = g_src[(y0 * SW + x1) * 3 + c];
        float p01 = g_src[(y1 * SW + x0) * 3 + c], p11 = g_src[(y1 * SW + x1) * 3 + c];
        float p = (p00 * (1 - tx) + p10 * tx) * (1 - ty) + (p01 * (1 - tx) + p11 * tx) * ty;
        rgb[c] = (unsigned char) lroundf(clampf(p, 0.0f, 255.0f));
    }
}

// ---- mode 0/1/2: the marching search with variants ----
static float reproject_march(float ux, float vy, int mode) {
    float searchRadius = SUBJECT_TRACK ? DIVERGENCE : DIVERGENCE * std::max(FOCAL, 1.0f - FOCAL);
    float startX = ux - searchRadius;
    float stepX = 2.0f * searchRadius / STEPS;

    float bestX = ux, bestDepth = -1.0f;
    float bgX = ux, bgDepth = 2.0f;

    float prevX = startX;
    float prevD = sample_depth(prevX, vy);
    float prevG = (prevX - ux) - EYE_SIGN * depth_parallax(prevD, prevX);
    if (prevD < bgDepth) { bgDepth = prevD; bgX = prevX; }

    for (int i = 1; i <= STEPS; i++) {
        float x = startX + stepX * i;
        float d = sample_depth(x, vy);
        float g = (x - ux) - EYE_SIGN * depth_parallax(d, x);

        if ((prevG <= 0 && g >= 0) || (prevG >= 0 && g <= 0)) {
            float crossX, crossD;
            if (mode == 2 && fabsf(d - prevD) > 0.15f) {
                // cliff-step: bisect the depth edge, emit the front side at the edge
                float frontD = std::max(prevD, d);
                float midD = 0.5f * (prevD + d);
                bool frontIsPrev = prevD > d;
                float aX = prevX, bX = x;
                for (int k = 0; k < 4; k++) {
                    float mX = 0.5f * (aX + bX);
                    float mD = sample_depth(mX, vy);
                    if ((mD > midD) == frontIsPrev) aX = mX; else bX = mX;
                }
                crossX = frontIsPrev ? aX : bX;
                crossD = frontD;
            } else {
                float denom = g - prevG;
                float t = (fabsf(denom) > 1e-6f) ? clampf(-prevG / denom, 0, 1) : 0.0f;
                crossX = prevX + (x - prevX) * t;
                crossD = prevD + (d - prevD) * t;
            }
            if (crossD > bestDepth) { bestDepth = crossD; bestX = crossX; }
        }
        if (d < bgDepth) { bgDepth = d; bgX = x; }
        prevX = x; prevD = d; prevG = g;
    }
    return (bestDepth >= 0.0f) ? bestX : bgX;
}

// ---- mode 3: splat/coverage ----
static float reproject_splat(float ux, float vy) {
    float searchRadius = SUBJECT_TRACK ? DIVERGENCE : DIVERGENCE * std::max(FOCAL, 1.0f - FOCAL);
    float startX = ux - searchRadius;
    float stepX = 2.0f * searchRadius / STEPS;
    float h = 0.75f * stepX;          // footprint half-width
    const float MARGIN = 0.02f;       // frontmost hysteresis

    float bestD = -1.0f, bestP = 0.0f;
    bool covered = false;
    float bgX = ux, bgDepth = 2.0f;

    for (int i = 0; i <= STEPS; i++) {
        float x = startX + stepX * i;
        float d = sample_depth(x, vy);
        float p = EYE_SIGN * depth_parallax(d, x);
        float m = x - p;
        if (fabsf(m - ux) <= h && d > bestD + MARGIN) { bestD = d; bestP = p; covered = true; }
        if (d < bgDepth) { bgDepth = d; bgX = x; }
    }
    return covered ? (ux + bestP) : bgX;
}

// ---- mode 4: two-layer edge blend ----
// The march is unchanged, but when the winning crossing lies on a depth RAMP (an edge band,
// between the local front and back plateaus), don't sample the ramp's stretched source strip
// (whose content varies row to row -> the serrated fringe). Instead cross-dissolve between
// the FRONT surface continued at its own parallax and the BACK surface at its parallax,
// weighted by the ramp fraction. Both layers sample coherent, unstretched texture.
static void reproject_blend(float ux, float vy, unsigned char *rgb) {
    float searchRadius = SUBJECT_TRACK ? DIVERGENCE : DIVERGENCE * std::max(FOCAL, 1.0f - FOCAL);
    float startX = ux - searchRadius;
    float stepX = 2.0f * searchRadius / STEPS;

    float bestX = ux, bestDepth = -1.0f, bestGrad = 1.0f;
    float bgX = ux, bgDepth = 2.0f;
    float dmax = 0.0f;
    float maxStep = 0.0f, xSteep = ux, steepGrad = 1.0f;  // steepest depth change = silhouette位置
    float sxA = ux, sxB = ux, sdA = 0.0f, sdB = 0.0f;     // the steepest bracket's endpoints
    float tv = 0.0f;

    float prevX = startX;
    float prevD = sample_depth(prevX, vy);
    float prevG = (prevX - ux) - EYE_SIGN * depth_parallax(prevD, prevX);
    if (prevD < bgDepth) { bgDepth = prevD; bgX = prevX; }
    dmax = std::max(dmax, prevD);

    for (int i = 1; i <= STEPS; i++) {
        float x = startX + stepX * i;
        float d = sample_depth(x, vy);
        float g = (x - ux) - EYE_SIGN * depth_parallax(d, x);
        if ((prevG <= 0 && g >= 0) || (prevG >= 0 && g <= 0)) {
            float denom = g - prevG;
            float t = (fabsf(denom) > 1e-6f) ? clampf(-prevG / denom, 0, 1) : 0.0f;
            float crossX = prevX + (x - prevX) * t;
            float crossD = prevD + (d - prevD) * t;
            if (crossD > bestDepth) {
                bestDepth = crossD; bestX = crossX;
                bestGrad = (d >= prevD) ? 1.0f : -1.0f;  // front side is toward +grad
            }
        }
        float ds = fabsf(d - prevD);
        tv += ds;
        if (ds > maxStep) {
            maxStep = ds;
            sxA = prevX; sxB = x; sdA = prevD; sdB = d;  // winning bracket, refined after loop
            steepGrad = (d >= prevD) ? 1.0f : -1.0f;
        }
        if (d < bgDepth) { bgDepth = d; bgX = x; }
        dmax = std::max(dmax, d);
        prevX = x; prevD = d; prevG = g;
    }

    if (bestDepth < 0.0f) { src_sample(clampf(bgX, 0, 1), vy, rgb); return; }

    // Sub-pixel silhouette position: where the (bilinear-continuous) depth crosses the
    // global front/back midpoint inside the steepest bracket. The bracket midpoint alone is
    // probe-quantized (+-half step), and everything clamped against xSteep inherits that
    // per-row jitter, rendering the edge's contact line as broken dashes.
    if (maxStep > 0.0f && fabsf(sdB - sdA) > 1e-6f) {
        float gmid = 0.5f * (dmax + bgDepth);
        float tmid = clampf((gmid - sdA) / (sdB - sdA), 0.0f, 1.0f);
        xSteep = sxA + (sxB - sxA) * tmid;
    }

    float range = dmax - bgDepth;
    const float EDGE_T = 0.15f;  // treat as an edge band only across a real cliff
    extern float g_tv_gate;
    bool cleanEdge = (tv < g_tv_gate * range);
    if (cleanEdge && range > EDGE_T && bestDepth < dmax - 0.02f && bestDepth > bgDepth + 0.02f) {
        float w = clampf((bestDepth - bgDepth) / range, 0.0f, 1.0f);
        extern float g_blend_lo;   // dissolve window = [lo, hi] of the ramp fraction
        extern float g_blend_hi;
        w = clampf((w - g_blend_lo) / (g_blend_hi - g_blend_lo), 0.0f, 1.0f);
        w = w * w * (3.0f - 2.0f * w);  // smoothstep
        // Each layer samples PARALLAX-CONTINUED (seamless with the adjacent normal-path
        // pixels at both band boundaries), but CLAMPED at the silhouette so it can never
        // read the other side's content: the front layer never reaches past the edge (which
        // painted a phantom second edge), the back layer never reaches into the object's
        // contact-shadow strip (which painted a dark channel).
        extern float g_inset_f, g_inset_b;  // clamp margins, in source px
        float insetF = g_inset_f / 5120.0f, insetB = g_inset_b / 5120.0f;
        float xF = ux + EYE_SIGN * depth_parallax(dmax, ux);
        float xB = ux + EYE_SIGN * depth_parallax(bgDepth, ux);
        if ((xF - xSteep) * steepGrad < insetF) xF = xSteep + steepGrad * insetF;
        if ((xSteep - xB) * steepGrad < insetB) xB = xSteep - steepGrad * insetB;
        xF = clampf(xF, 0, 1); xB = clampf(xB, 0, 1);
        unsigned char cF[3], cB[3];
        src_sample(xF, vy, cF);
        src_sample(xB, vy, cB);
        for (int c = 0; c < 3; c++) rgb[c] = (unsigned char) lroundf(cB[c] + (cF[c] - cB[c]) * w);
        return;
    }
    // Contact-shadow exclusion: a background pixel near a silhouette may sample source columns
    // immediately on the bg side of the edge -- the object's contact shadow / halo. That strip
    // carries BACKGROUND depth, so it detaches from the object and floats in the middle of the
    // background as a dark line tracking the contour. Push such samples out past the strip.
    extern float g_exclude, g_excl_min;  // source px
    if (cleanEdge && g_exclude > 0 && maxStep > EDGE_T && bestDepth <= bgDepth + 0.05f) {
        float excl = g_exclude / 5120.0f;
        float exclMin = g_excl_min / 5120.0f;
        float rel = bestX - xSteep;
        float dist = -rel * ((steepGrad > 0) ? 1.0f : -1.0f);  // distance onto the bg side
        // Only on the BACKGROUND side of the edge. Smoothly remap the sampling window
        // [0, excl] -> [exclMin, excl]: skips the shadow strip near the edge but stays
        // continuous at the outer boundary (a hard clamp replicates one column -> stripes).
        if (dist >= 0 && dist < excl) {
            float distR = exclMin + dist * (excl - exclMin) / excl;
            bestX = xSteep - steepGrad * distR;
        }
    }
    src_sample(clampf(bestX, 0, 1), vy, rgb);
}

float g_blend_lo = 0.0f;
float g_blend_hi = 1.0f;
float g_inset_f = 4.0f;
float g_inset_b = 8.0f;
float g_exclude = 0.0f;
float g_excl_min = 10.0f;
float g_tv_gate = 999.0f;

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: warpsim <mode> <out.raw> [blend_lo] [blend_hi] [inset_f] [inset_b]\n"); return 2; }
    int mode = atoi(argv[1]);
    if (argc > 3) g_blend_lo = (float) atof(argv[3]);
    if (argc > 4) g_blend_hi = (float) atof(argv[4]);
    if (argc > 5) g_inset_f = (float) atof(argv[5]);
    if (argc > 6) g_inset_b = (float) atof(argv[6]);
    if (argc > 7) g_exclude = (float) atof(argv[7]);
    if (argc > 8) g_tv_gate = (float) atof(argv[8]);

    auto load = [](const char *p, std::vector<unsigned char> &v, size_t n) {
        FILE *f = fopen(p, "rb");
        if (!f) { fprintf(stderr, "missing %s\n", p); exit(1); }
        v.resize(n);
        if (fread(v.data(), 1, n, f) != n) { fprintf(stderr, "short read %s\n", p); exit(1); }
        fclose(f);
    };
    load("depth_1596x672.bin", g_depth, (size_t) DW * DH);
    load("source_x2000y900_800x400.bin", g_src, (size_t) SW * SH * 3);

    g_smooth = (mode != 1);

    std::vector<unsigned char> out((size_t) OW * OH * 3);
    for (int py = 0; py < OH; py++) {
        float vy = (EOY + py + 0.5f) / 1728.0f;
        for (int px = 0; px < OW; px++) {
            float ux = (EOX + px + 0.5f) / 4096.0f;
            if (mode == 4) {
                reproject_blend(ux, vy, &out[((size_t) py * OW + px) * 3]);
            } else {
                float outX = (mode == 3) ? reproject_splat(ux, vy) : reproject_march(ux, vy, mode);
                outX = clampf(outX, 0.0f, 1.0f);
                src_sample(outX, vy, &out[((size_t) py * OW + px) * 3]);
            }
        }
    }
    FILE *f = fopen(argv[2], "wb");
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    printf("mode %d -> %s\n", mode, argv[2]);
    return 0;
}
