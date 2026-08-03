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
// External final-parallax replay and live V2 consume a certified sub-unit-slope field. Each eye
// therefore has one root and an inexpensive fixed-point inverse. SBS_DIRECT_PARALLAX is the
// encoded interchange contract; SBS_LIVE_V2_SIGNED_PARALLAX binds the signed field directly.
//
// Legacy inputs (unchanged contract with display_vram.cpp):
//   t0 = mono color, t1 = robust P2/P98-normalized and temporally filtered depth (high = near),
//   t2 = legacy subject state, t4 = legacy frame state, s0 = linear clamp sampler,
//   b2 = tuning constants below.
// Live V2 replaces t1 with authenticated final source-U and t2 with its state.

Texture2D<float4> LeftColorTexture : register(t0);
Texture2D<float>  DepthTexture      : register(t1);
#if defined(SBS_LIVE_V2_SIGNED_PARALLAX)
StructuredBuffer<float4> ParallaxV2State : register(t2);
#elif !defined(SBS_DIRECT_PARALLAX)
// Subject-tracking state from depth_subject_resolve_cs. [0] = {recenter_delta, scene_age,
// subject_depth_ema, initialized}; [1] = {stretch_lo_val, stretch_inv_range,
// depth_change_baseline, adaptive_pop_ratio}. Direct replay does not declare this resource.
StructuredBuffer<float4> SubjectState : register(t2);
#endif
// Bound only by the legacy/offline diagnostic mask pass. A live V2 map is contractive and has no
// internal continuous-geometry holes. Its dump-only mask instead attributes the finite-image
// boundary samples that main_ps clamps to the nearest source column.
Texture2D<uint> ForwardCoverageTexture : register(t3);
#if !defined(SBS_DIRECT_PARALLAX) && !defined(SBS_LIVE_V2_SIGNED_PARALLAX)
// {min, max, initialized, frame_state}; frame_state == 0 means this TensorRT completion was
// entirely invalid and every depth/subject output deliberately held its preceding value.
StructuredBuffer<float4> DepthFrameState : register(t4);
#endif
SamplerState      LinearSampler     : register(s0);

struct PS_INPUT {
    float4 Pos      : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

#ifdef SBS_LIVE_V2_SIGNED_PARALLAX
#include "include/depth_coordinate_v2_contract.generated.hlsl"
#endif
#include "include/sbs_warp_common.hlsl"

#ifdef SBS_LIVE_V2_SIGNED_PARALLAX
bool ParallaxV2Finite(float value) {
    return !isnan(value) && !isinf(value);
}

bool ParallaxV2CameraInitialized(float4 active, float4 control) {
    uint revision = asuint(V2_STATE_CALIBRATION_REVISION(control));
    return ParallaxV2Finite(V2_STATE_CENTER(active)) &&
        ParallaxV2Finite(V2_STATE_INVERSE_SCALE(active)) &&
        ParallaxV2Finite(V2_STATE_CONVERGENCE_CURVE(active)) &&
        V2_STATE_INVERSE_SCALE(active) > 0.0f &&
        revision > 0u && revision != 0xffffffffu;
}
#endif

// depth_warp_prefilter_cs folds the former four-tap silhouette-width kernel once per completed
// model depth. Each full-resolution search probe therefore needs only this single lookup.
float SampleDepth(float sx, float sy) {
    return DepthTexture.SampleLevel(LinearSampler, float2(sx, sy), 0);
}

// Find the source U coordinate that reprojects onto `uv` for one eye.
// eyeSign = +1 right eye, -1 left eye.
float2 Reproject(
    float2 uv,
    float eyeSign,
    bool use_subject_stretch
) {
#ifdef SBS_CONTRACTIVE_PARALLAX
    // The final direct-parallax field has an authenticated source-U slope <= 0.5. Therefore
    // x -> uv.x + eyeSign * parallax(x) is a contraction for either eye and its fixed point is
    // the unique inverse of out(x) = x - eyeSign * parallax(x). Starting at the destination,
    // twelve iterations leave at most 0.5^12 of the initial error (under 0.04 source pixel at
    // 3840 pixels for the 0.04 source-U container). There is no visibility choice: monotonicity
    // rules out multiple roots, so canonical order, lattice search, and hole filling do not belong
    // in this rendering path. The separate harness coverage pass remains a diagnostic only.
    float sourceX = uv.x;
    [unroll]
    for (int iteration = 0; iteration < 12; ++iteration) {
        sourceX = uv.x + eyeSign * DepthParallax(SampleDepth(sourceX, uv.y));
    }
    return float2(sourceX, uv.y);
#else
    // Subject anchoring is live this frame only if configured AND the resolve pass has
    // produced state (init != 0 -- it is 0 for the first frames). Mandatory shader/resource
    // initialization is validated before the estimator is published. Decide it ONCE here
    // from the runtime state, and read the frame-uniform SubjectState once, so the probe loop
    // below issues no per-probe buffer loads (DepthParallax gets s0/s1 as args). The search span is
    // then derived from the same Bestv2Params the mapping uses, so the two cannot disagree.
    float4 s0 = SubjectState[SBS_STATE_VECTOR_SUBJECT_RECENTER_DELTA];
    float4 s1 = SubjectState[SBS_STATE_VECTOR_STRETCH_LO];
    float4 s2 = SubjectState[SBS_STATE_VECTOR_ZERO_ANCHOR_SHIFT_PX];
    bool shaped = SBS_STATE_INITIALIZED(s0) > 0.5f;
    // Bestv2's calibrated bands are SOURCE-COLOR pixel shifts. Normalizing by the smaller
    // inference-depth width amplified Apollo whenever the model texture was downscaled, while
    // Parallax uses the eye/source width. Depth dimensions remain correct for tap offsets.
    uint sourceWidth, sourceHeight;
    LeftColorTexture.GetDimensions(sourceWidth, sourceHeight);

    if (!shaped) {
        return uv;  // subject state is not initialized yet
    }
    Bestv2Params params = MakeBestv2Params(
        s0, s1, s2, (float)sourceWidth, (float)sourceHeight, use_subject_stretch);
    // Sized from THIS frame's resolved anchor, strength and convergence rather than from a global
    // worst case, so the window matches the displacement that can actually occur.
    float searchRadius = Bestv2SearchRadius(params);
    if (searchRadius <= 1e-6f) {
        return uv;  // this frame's parallax field is degenerate; the root is uv.x itself
    }

    // Probes sit on the global lattice k * spacing rather than at uv.x +- i * step, so the window
    // can be narrowed without moving any probe that survives.
    uint depthWidth, depthHeight;
    DepthTexture.GetDimensions(depthWidth, depthHeight);
    float spacing = Bestv2ProbeSpacing((float)sourceWidth, (float)depthWidth);
    int intervals = Bestv2ProbeIntervals(
        searchRadius, Bestv2MaxProbes((float)sourceWidth, (float)sourceHeight), spacing);
    int probeStart = Bestv2ProbeStart(uv.x, searchRadius, spacing);

    // A source at position x forward-warps to out(x) = x - eyeSign * parallax(depth(x)).
    // We want out(x) == uv.x, i.e. g(x) = (x - uv.x) - eyeSign * parallax(depth(x)) == 0.
    // Marching x and watching g() cross zero locates each reprojecting source; we keep
    // the one with the greatest depth (closest to the viewer).
    float bestX = uv.x;
    float bestDepth = -1.0f;

    // Track the FARTHEST (lowest depth = background) sample in the window as well. When
    // no source reprojects onto uv (a disocclusion / hole revealed by a moving foreground
    // edge), we fill from this nearest background instead of sampling straight through,
    // which would smear the foreground colour into the gap -- the bright "white edge"
    // that shimmers on motion.
    float bgX = uv.x;
    float bgDepth = 2.0f;  // above any normalized depth (<= 1)

    // Positions are exact multiples of the quantized spacing (see Bestv2ProbeSpacing), so they
    // depend only on the lattice index and not on where this run's window happened to start.
    float prevX = (float)probeStart * spacing;
    float prevD = SampleDepth(prevX, uv.y);
    float prevG = (prevX - uv.x) - eyeSign * DepthParallax(
        prevD, s0, s1, params, use_subject_stretch);
    if (prevD < bgDepth) { bgDepth = prevD; bgX = prevX; }

    [loop]
    for (int i = 1; i <= intervals; i++) {
        float x = (float)(probeStart + i) * spacing;
        float d = SampleDepth(x, uv.y);
        float g = (x - uv.x) - eyeSign * DepthParallax(
            d, s0, s1, params, use_subject_stretch);

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
#endif
}

float4 main_ps(PS_INPUT input) : SV_TARGET {
#ifdef SBS_LIVE_V2_SIGNED_PARALLAX
    // The CPU authenticates the producer and shader closure before selecting this renderer. Keep
    // the GPU boundary fail-closed as well: a same-sized unknown state buffer must never drive a
    // signed displacement field. A tag mismatch holds the already-rendered flat/last good pair.
    float4 v2_active = ParallaxV2State[V2_STATE_VECTOR_CENTER];
    float4 v2_control = ParallaxV2State[V2_STATE_VECTOR_CALIBRATION_REVISION];
    if (asuint(V2_STATE_CONTRACT_TAG_BITS(v2_control)) != V2_CONTRACT_TAG) {
        discard;
    }

    bool v2_camera_initialized = ParallaxV2CameraInitialized(v2_active, v2_control);
    bool v2_frame_valid = V2_STATE_FRAME_VALID(v2_control) > 0.5f;
    // An invalid completion retains the preceding camera and final field. Hold the corresponding
    // matched color as well. If no camera exists (including an invalid confirmed-cut acquisition),
    // draw the current matched color flat rather than preserving stale scene geometry.
    if (!v2_frame_valid && v2_camera_initialized) {
        discard;
    }
    bool v2_warp_current_frame = v2_frame_valid && v2_camera_initialized;
#elif !defined(SBS_DIRECT_PARALLAX)
    // Once real depth exists, an all-invalid TensorRT completion leaves depth/subject state
    // untouched. Discarding every pixel likewise holds the last matched color/depth pair without
    // a CPU readback or GPU flush. Before the first valid depth, do draw this completion: the
    // uninitialized subject state makes Reproject() return the identity mapping, keeping flat SBS
    // color live instead of freezing the first invalid frame forever.
    if (preserve_previous_on_invalid > 0.5f
            && DepthFrameState[0].w < 0.5f
            && DepthFrameState[0].z >= 0.5f) {
        discard;
    }
#endif

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

    // Compile the stretch-on and stretch-off loops separately and select once per pixel, avoiding
    // a per-probe runtime select in the production path. Direct replay is already shaped, so it
    // must not branch on the legacy subject-stretch control.
    float2 sample_uv;
#ifdef SBS_LIVE_V2_SIGNED_PARALLAX
    [branch]
    if (v2_warp_current_frame) {
        sample_uv = Reproject(src_uv, eyeSign, false);
    } else {
        sample_uv = src_uv;
    }
#elif defined(SBS_DIRECT_PARALLAX)
    sample_uv = Reproject(src_uv, eyeSign, false);
#else
    [branch]
    if (subject_stretch > 0.5f) {
        sample_uv = Reproject(src_uv, eyeSign, true);
    } else {
        sample_uv = Reproject(src_uv, eyeSign, false);
    }
#endif

    // Disoccluded regions clamp to the nearest valid column instead of wrapping.
    sample_uv.x = saturate(sample_uv.x);

    float4 col = LeftColorTexture.Sample(LinearSampler, sample_uv);

    return col;
}

// Harness-only exact warp mapping. The diagnostic pass calls the same Reproject function as
// main_ps and runs outside production timing. Its R32_FLOAT target preserves Reproject's RAW
// normalized source U before main_ps clamps it to [0, 1]. Geometry/comfort metrics therefore see
// the requested disparity instead of silently losing off-screen motion; source-fidelity metrics
// apply the live clamp before reconstructing the sampled color. Aspect-content validity derives
// from the shared shape contract; forward-coverage validity remains in the existing warp mask,
// avoiding three redundant float channels per output pixel.
float mapping_ps(PS_INPUT input) : SV_TARGET {
    float2 packed_uv = input.TexCoord;
    bool is_right_eye = packed_uv.x > 0.5f;
    float eyeSign = is_right_eye ? 1.0f : -1.0f;
    float2 output_uv = packed_uv;
    output_uv.x = is_right_eye ? (packed_uv.x - 0.5f) * 2.0f : packed_uv.x * 2.0f;
    float2 src_uv;
    if (!ContentToSourceUV(output_uv, src_uv)) {
        return 0.0f;
    }

    float2 sample_uv;
#if defined(SBS_DIRECT_PARALLAX) || defined(SBS_LIVE_V2_SIGNED_PARALLAX)
    sample_uv = Reproject(src_uv, eyeSign, false);
#else
    [branch]
    if (subject_stretch > 0.5f) {
        sample_uv = Reproject(src_uv, eyeSign, true);
    } else {
        sample_uv = Reproject(src_uv, eyeSign, false);
    }
#endif
    return sample_uv.x;
}

// Harness-only diagnostic output. R marks legacy forward-coverage disocclusion before its gather
// paints over it. Live V2 has no internal holes, but its monotone map is defined on a finite source
// interval: R marks inverse samples outside [0, 1] that main_ps clamps to the nearest boundary
// column. Bars are not content and remain unmarked. Compiling this separate entry point adds no
// live-stream work.
float4 mask_ps(PS_INPUT input) : SV_TARGET {
    float2 uv = input.TexCoord;
    bool is_right_eye = uv.x > 0.5f;
    float2 output_uv = uv;
    output_uv.x = is_right_eye ? (uv.x - 0.5f) * 2.0f : uv.x * 2.0f;
    float2 src_uv;
    if (!ContentToSourceUV(output_uv, src_uv)) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
#ifdef SBS_LIVE_V2_SIGNED_PARALLAX
    float eyeSign = is_right_eye ? 1.0f : -1.0f;
    float sourceX = Reproject(src_uv, eyeSign, false).x;
    float boundary_extrapolation = (sourceX < 0.0f || sourceX > 1.0f) ? 1.0f : 0.0f;
    return float4(boundary_extrapolation, 0.0f, 0.0f, 1.0f);
#else
    uint full_w, full_h;
    ForwardCoverageTexture.GetDimensions(full_w, full_h);
    uint2 output_px = min((uint2)input.Pos.xy, uint2(full_w - 1u, full_h - 1u));
    float hole = ForwardCoverageTexture.Load(int3(output_px, 0)) == 0u ? 1.0f : 0.0f;
    return float4(hole, 0.0f, 0.0f, 1.0f);
#endif
}
