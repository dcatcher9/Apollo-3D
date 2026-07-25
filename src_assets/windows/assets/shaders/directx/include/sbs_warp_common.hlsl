#ifndef SBS_WARP_COMMON_HLSL
#define SBS_WARP_COMMON_HLSL

#include "include/bestv2_curve.hlsl"

// The validated core clips and Bestv2-derived profile were calibrated at 854 source pixels wide.
// The reference preset expresses disparity as literal render pixels, which becomes imperceptible on a 5120px
// desktop. Preserve exact behavior at and below the calibration raster, but scale wider sources
// so disparity remains a constant percentage of each eye instead of a constant pixel count.
// Non-literal production geometry also applies the independent reference-aspect correction below.
// Consequently, a sub-854 non-16:9 source preserves Bestv2's literal pixel shift and then scales
// that shift for its aspect; this is deliberate and must be changed only with a dedicated 4:3 A/B.
static const float BESTV2_CALIBRATION_WIDTH = 854.0f;
// The approved headset look was tuned with a 5120x2160 stream request. Normalize source-UV
// disparity by reference/current per-eye aspect so the same scene keeps its calibrated stereo
// strength when the client requests a different image shape. This is image geometry; it does not
// depend on the client's physical panel size or placement.
static const float BESTV2_REFERENCE_ASPECT = 5120.0f / 2160.0f;

float Bestv2AspectScale(float source_width, float source_height, float literal_mode) {
    if (literal_mode > 0.5f) {
        return 1.0f;
    }
    float aspect = max(source_width / max(source_height, 1.0f), 1e-4f);
    // Bound pathological custom modes while covering the validated portrait-like 3552x3840 case.
    return clamp(BESTV2_REFERENCE_ASPECT / aspect, 0.5f, 3.0f);
}

float Bestv2ParallaxWidth(float source_width, float literal_mode) {
    if (literal_mode > 0.5f) {
        return max(source_width, 1.0f);
    }
    return min(max(source_width, 1.0f), BESTV2_CALIBRATION_WIDTH);
}

// Apollo's shared depth-shaping and disparity contract.
cbuffer Constants : register(b2) {
    float subject_lock;
    float subject_stretch;
    float content_scale_x;       // source content width / output-eye width (per-eye letterbox)
    float content_scale_y;       // source content height / output-eye height
    float pop_strength;          // final production stereo-parallax multiplier
    float literal_bestv2;        // harness-only: bypass production resolution/aspect/pop scaling
    float adaptive_pop;          // use SubjectState[1].w scene-risk multiplier
    float adaptive_pop_max;      // configured ceiling; the resolved ratio arrives in SubjectState
};

// Map one eye's output UV into the mono source. Letterbox/pillarbox is applied independently in
// each eye, preventing a packed-frame viewport offset from becoming a false stereo disparity.
bool ContentToSourceUV(float2 output_uv, out float2 source_uv) {
    float2 scale = max(float2(content_scale_x, content_scale_y), float2(1e-6f, 1e-6f));
    float2 lo = 0.5f * (float2(1.0f, 1.0f) - scale);
    float2 hi = lo + scale;
    if (any(output_uv < lo) || any(output_uv > hi)) {
        source_uv = float2(0.0f, 0.0f);
        return false;
    }
    source_uv = saturate((output_uv - lo) / scale);
    return true;
}

// Loop-invariant values for the production warp. Keeping the original
// operation groups here avoids recomputing source geometry, subject shift and convergence for
// every search probe while retaining the same Bestv2 field.
struct Bestv2Params {
    float subject_shift_px;
    float zero_anchor_shift_px;
    float explicit_zero_plane;
    float anchor_shift_px;
    float parallax_scale;
    float convergence_bias;
    float output_scale;
};

Bestv2Params MakeBestv2Params(float4 s0, float4 s1, float4 s2,
                              float source_width, float source_height,
                              bool use_subject_stretch) {
    Bestv2Params p;
    float parallax_width = Bestv2ParallaxWidth(source_width, literal_bestv2);
    float subject_depth = Bestv2WarpDepth(s0.z, s0, s1, true, use_subject_stretch);
    p.subject_shift_px = Bestv2RawShiftPxFast(subject_depth);
    p.zero_anchor_shift_px = s2.x;
    p.explicit_zero_plane = s2.y;
    p.parallax_scale = 0.35f / parallax_width;
    // Legacy retains the Bestv2 trim/convergence exactly. Explicit modes instead place their
    // shot-latched anchor on the screen plane, so adding the legacy offset would defeat t.
    p.convergence_bias = p.explicit_zero_plane > 0.5f ? 0.0f :
                         -0.008f * 0.5f + s1.z * 4.0f / parallax_width;
    // Frame-uniform: hoisted out of the probe loop, and read directly by the search bound below.
    p.anchor_shift_px = p.explicit_zero_plane > 0.5f ? p.zero_anchor_shift_px :
                        subject_lock * p.subject_shift_px;
    float aspect_scale = Bestv2AspectScale(source_width, source_height, literal_bestv2);
    float adaptive_ratio = adaptive_pop > 0.5f ? max(s1.w, 1.0f) : 1.0f;
    float strength = literal_bestv2 > 0.5f ? 1.0f : pop_strength * adaptive_ratio;
    p.output_scale = strength * aspect_scale;
    return p;
}

float DepthParallax(float d, float4 s0, float4 s1, Bestv2Params p,
                    bool use_subject_stretch) {
    float shaped_depth = Bestv2WarpDepth(d, s0, s1, true, use_subject_stretch);
    float shift_px = Bestv2RawShiftPxFast(shaped_depth);
    float parallax = (shift_px - p.anchor_shift_px) * p.parallax_scale;
    parallax += p.convergence_bias;
    // No safety clamp: it was provably unreachable. reach = 9.979 * (0.35/854) * strength *
    // aspect_scale and the old bound was 0.071 * aspect_scale, so aspect cancels and binding needs
    // strength > 17.36, while config.cpp validates BOTH pop_strength and adaptive_pop_max into
    // [0.25, 2.0]. 8.7x margin, enforced rather than incidental.
    return parallax * p.output_scale;
}

// Probe SPACING is what the historical calibration actually pinned; the step count was only ever
// the number of probes needed to achieve that spacing across an oversized radius. The former
// formulation derived spacing as 2*radius/step_count, in which the aspect scale
// and the search strength appeared in both factors purely so they would cancel --
// a now-removed calibrated-strength constant and the step count's strength renormalization
// existed only to make that cancellation exact. Expressing spacing directly collapses all of it, and it is what makes
// the search window sizeable per frame: the probes then sit on a global lattice instead of being
// positioned relative to the output pixel, so narrowing the window removes probes rather than
// moving every one of them.
//
// The remaining bracket is the legacy radius geometry, retained verbatim as the calibration
// reference it now is rather than as a bound.
//
// 13.4 places probes ~1.0 depth texel apart. The historical 19.4 (and the 24 before it) sampled at
// ~0.69 texels, i.e. 1.46x finer than the depth signal can vary: SampleDepth is bilinear over the
// 770-wide map, so g(x) is piecewise-linear between depth texels and one probe interval generically
// contains at most one breakpoint at unit spacing. A ladder at 0.69 / 0.80 / 1.00 / 1.20 texels
// found stretch improving monotonically as spacing coarsened (0.5059 -> 0.4964 -> 0.4822 -> 0.4741)
// with fold 0.000, stereo volume and jitter unchanged, and warp time -7.8% / -16.1% / -22.2%.
// Visual inspection at 1.0 across every clip: worst case 0.060% of pixels differ by more than 8
// gray levels (c525), the thin-structure anime clip 0.035%, and the largest differences are in
// smooth sky rather than at silhouettes -- so the coarser search is not missing crossings.
//
// STOP AT 1.0 rather than taking 1.2's better numbers. Unit spacing is where the probe grid matches
// the resolution of the signal it samples; beyond that the grid is coarser than the data and the
// one-breakpoint-per-interval argument no longer holds, so the continued stretch improvement is
// more likely the metric rewarding a smoother mapping than the warp resolving better.
static const float BESTV2_CALIBRATED_STEPS = 13.4f;
// Significant bits kept in the returned spacing. This quantization is load-bearing, not cosmetic.
// The shared lattice is only shared if two runs sampling the same index k land on the same
// position, and the compiler strength-reduces the loop's (float)(probeStart + i) * spacing into an
// accumulator seeded at each run's OWN starting index -- so in general they do not. Forcing
// spacing to an exact multiple of a power of two removes the difference instead of forbidding the
// transform: every partial sum k * spacing is then exactly representable (|k| stays in the
// hundreds and the mantissa product well under 2^24), so accumulating and multiplying agree
// bit-for-bit. Marking the position `precise` also works and needs no quantization, but it blocks
// the strength reduction and measured about 18% more warp time. The quantization itself moves
// spacing by at most 2^-11 relative.
static const float BESTV2_PROBE_SPACING_BITS = 10.0f;

// Target probe spacing expressed in DEPTH texels, which is the quantity that actually governs the
// search: SampleDepth is bilinear over the depth map, so g(x) is piecewise-linear between depth
// texels and the probe grid's job is to bracket at most one breakpoint per interval. Spacing was
// previously derived purely from source-COLOR geometry and never referenced the depth map, so its
// value in depth texels held only by coincidence at one depth resolution -- changing
// sbs_3d_depth_short_side silently moved it (finer if lowered, wasting work; coarser if raised,
// risking missed crossings). Expressing it in depth texels makes it self-calibrating.
//
// 1.22 is the empirically validated point, not a derived one. A ladder measured 0.843 / 0.979 /
// 1.220 / 1.460 depth texels: stretch improved monotonically as spacing coarsened, fold stayed
// 0.000, stereo volume and jitter were unchanged, and warp time fell 7.8% / 16.1% / 22.2%. 1.220
// was visually inspected across every core clip -- worst case 0.060% of pixels differ by more than
// 8 gray levels, and on the thin structures most at risk (a warrior's staff, wispy hair, silhouette
// edges) 23, 18 and 22 pixels respectively, with no breakage. 1.460 was measured but NOT visually
// inspected, so it is not adopted.
static const float BESTV2_TARGET_DEPTH_TEXELS = 1.22f;

float Bestv2ProbeSpacing(float source_width, float depth_width) {
    // literal_bestv2 is a fixed-strength comparison reference whose spacing never varies, so it
    // keeps its historical unrenormalized count rather than the calibrated-strength one.
    // literal_bestv2 is a fixed-strength comparison reference; keep it on the historical
    // source-geometry formulation so its spacing is unaffected by depth-resolution changes.
    float spacing;
    if (literal_bestv2 > 0.5f) {
        spacing = (2.0f * 1.0f / BESTV2_CALIBRATED_STEPS) *
                  (0.004f + (10.1f * 0.35f + 0.006f * 4.0f) /
                              Bestv2ParallaxWidth(source_width, literal_bestv2));
    } else {
        spacing = BESTV2_TARGET_DEPTH_TEXELS / max(depth_width, 1.0f);
    }
    float ulp = exp2(floor(log2(spacing)) - BESTV2_PROBE_SPACING_BITS);
    return round(spacing / ulp) * ulp;
}

// Extrema of Bestv2RawShiftPxFast. The polynomial is evaluated only on [0,1] -- Bestv2WarpDepth
// saturates, and the stretch and recenter stages saturate before it -- and on that interval its
// extrema are the endpoints: S(0) and S(1). Verified numerically over the closed interval; the
// polynomial has no interior extremum outside [S(0), S(1)].
static const float BESTV2_SHIFT_PX_MIN = -1.39635933f;
static const float BESTV2_SHIFT_PX_MAX = 8.58230571f;
// The bound below is exact rather than estimated, so this only has to keep float evaluation slack
// from putting a root exactly on the window edge, where its outer bracketing probe would be one
// lattice cell away. 10% is far more than the polynomial's <0.01 px error needs.
static const float BESTV2_SEARCH_MARGIN = 1.10f;

// Every root of the backward search satisfies (x - uv.x) = eyeSign * DepthParallax(depth(x)), so
// the window only has to cover max|DepthParallax|. DepthParallax's single free input is the shaped
// depth, which is confined to [0,1]; everything else it touches is frame-uniform and already
// resolved in Bestv2Params. The reachable interval is therefore computed exactly:
//
//   [ (S_MIN - anchor)*parallax_scale + convergence_bias,
//     (S_MAX - anchor)*parallax_scale + convergence_bias ] * output_scale
//
// The previous formulation bounded the same quantity globally instead, and was about six times
// wider than any displacement that can occur at the shipped defaults, because it budgeted for
// three things simultaneously:
//   (a) a convergence bias worth ~49% of the radius, which is identically zero whenever an
//       explicit zero plane is selected -- the default since sbs_3d_zero_plane became `median`;
//   (b) the most adverse anchor ANY frame could produce (|S| span 9.979 px) rather than this
//       frame's, though the anchor is a resolved per-frame scalar sitting in Bestv2Params; and
//   (c) adaptive_pop_max rather than the strength the adaptive controller actually resolved --
//       so scenes the controller declined to boost paid the widest window and received none of
//       the gain, and widening the pop band made every scene's probe more expensive.
// The old comment claimed the resolved ratio was "unavailable to this loop-bound helper". It was
// not: SubjectState[1] is read at the only call site, and MakeBestv2Params already consumes it.
//
// Against the pre-change renderer this cuts mean probes per output pixel from 42.7 to about 16.0
// on core and 45.0 to about 15.1 on extended (reconstructed from the exported warp maps, since
// the count is not exported), for measured warp time -45.9% and -49.2%. Every rendered, depth and
// warp-map artifact on both suites stayed byte-identical to a run that kept the old window on
// the same lattice, which is the real gate here: the gated metrics are depth-side and cannot see
// a warp change at all.
float Bestv2SearchRadius(Bestv2Params p) {
    float lo = (BESTV2_SHIFT_PX_MIN - p.anchor_shift_px) * p.parallax_scale + p.convergence_bias;
    float hi = (BESTV2_SHIFT_PX_MAX - p.anchor_shift_px) * p.parallax_scale + p.convergence_bias;
    // output_scale is strictly positive, so scaling the interval preserves it.
    float reach = max(abs(lo), abs(hi)) * p.output_scale;
    return reach * BESTV2_SEARCH_MARGIN;
}

// Bound worst-case tall/high-resolution work. The live packed target contains two source-sized
// eyes; ~0.875G depth samples caps 3552x3840 at 32 samples per output pixel instead of allowing
// ~1.7G. With spacing fixed and the window sized from the frame's real displacement bound, normal
// rasters land far below this and never reach it; the five-sample floor is only meaningful far
// beyond the validated raster envelope and preserves a functional fallback there.
static const float BESTV2_MAX_DEPTH_PROBES = 8.75e8f;

// Total depth samples one output pixel may take, including the loop's initial sample.
int Bestv2MaxProbes(float source_width, float source_height) {
    float packed_pixels = max(2.0f * source_width * source_height, 1.0f);
    return max((int)floor(BESTV2_MAX_DEPTH_PROBES / packed_pixels), 5);
}

// Probe intervals along the global lattice k * spacing. The run starts at the lattice cell holding
// uv_x - radius (see Bestv2ProbeStart), so its phase varies by up to one cell per output pixel;
// one extra interval absorbs that and guarantees the covered span always contains
// [uv_x - radius, uv_x + radius]. A root at the very edge of the window therefore still has both
// of its bracketing probes sampled, in a narrowed window exactly as in a wide one.
//
// The count is frame-uniform rather than derived per pixel from ceil((uv_x + radius)/spacing).
// Both were measured at the shipped narrow radius on the same clip: uniform 0.0366 ms against
// 0.0381 ms for the per-pixel form, whose trip count diverges inside a wave by the wave's own
// width in lattice cells. At the old oversized radius the measured ranking was the other way
// round, for reasons that were not chased down -- so treat this as measured at the shipped
// window, not as a general rule, and re-measure if the window ever widens again.
//
// `spacing` is scaled in place when the work budget cannot afford the full lattice. The factor is
// integral so the thinned probes remain a sub-lattice of the same grid and stay exactly
// representable, and it is frame-uniform like the count itself. Only rasters far beyond the
// validated envelope reach it, and there two runs with different radii may land on different
// lattices and lose the byte-exactness argument.
int Bestv2ProbeIntervals(float radius, int max_probes, inout float spacing) {
    int intervals = (int)ceil((2.0f * radius) / spacing) + 1;
    int decimate = max((intervals + max_probes) / max_probes, 1);
    if (decimate > 1) {
        spacing *= (float)decimate;
        intervals = (int)ceil((2.0f * radius) / spacing) + 1;
    }
    return intervals;
}

// First lattice index of the run. Monotone in uv_x and independent of everything else, so two runs
// that share `spacing` sample bit-identical positions wherever their windows overlap.
int Bestv2ProbeStart(float uv_x, float radius, float spacing) {
    return (int)floor((uv_x - radius) / spacing);
}

#endif
