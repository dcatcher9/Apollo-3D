// SPDX-License-Identifier: GPL-3.0-only
// Native GPU implementation of the current Sunshine Game 3D effect.
// The add-on owns input capture, resources, scheduling and export; all image work
// below remains on the GPU. No external FX shader or ReShade preset is needed.
// Ported from the frozen 2026-09-18 three-file Game 3D source. Geometry authority:
// docs/host-sbs.md and the Depth Coordinate V2 vertical/horizontal limit shaders.
// Optional source-alpha UI pinning reuses the horizontal pass after conditioning.
// Live Auto selects each frame's UI mask on the GPU (SunshineUIDetection*CS);
// explicit replay retains its captured selection. Neither path adds color-layer
// blending.
// Nearest-covered-depth mode first resolves one global UI plane on this queue.
#define SUNSHINE_UI_NEAREST_PLANE 1
#define SUNSHINE_UI_FRONT_LIMIT_PLANE 1
#define SUNSHINE_UI_SHALLOW_FRONT_PLANE 1
#define SUNSHINE_UI_DISPLAY_FRACTION_PLANE 1
#define SUNSHINE_UI_ABSOLUTE_LEVEL_PROBE 1
#define SUNSHINE_UI_MASK_CHANNEL 1
#define SUNSHINE_UI_AUTOMATIC_DETECTION 1
#define SUNSHINE_LINEAR_DISTANCE_DEPTH 1
// Limiter groups own eight adjacent lines; UI pinning is a separate pass.
#define SUNSHINE_LIMITER_LINE_GROUPS 8
// One pass renders both eyes straight into the side-by-side target.
#define SUNSHINE_PACKED_EYES 1
// The mono pack (SunshineAutomaticMono) and the depth preview read no depth
// conditioning, so the renderer may skip it for a pack that shows either.
#define SUNSHINE_MONO_SKIPS_CONDITIONING 1
// An HDR10 (PQ) source is decoded per tap where the pack interpolates it
// (SunshineSourceLinear): no full-frame linearization pass and no FP16 linear
// texture. SunshineRenderPackedPQPS writes the 10-bit PQ export (Rec.2020,
// ST 2084) encoded exactly as the host's scRGBTo2100PQ (docs/reshade-sbs.md,
// PQ wire transfer).
#define SUNSHINE_PQ_PER_TAP 1
// UI pinning groups own eight adjacent rows.
#define SUNSHINE_UI_PIN_LINE_GROUPS 8
// UI mask alpha pins with weight saturate(gain * alpha) through a soft band
// (docs/reshade-sbs.md, UI pin band).
#define SUNSHINE_UI_SOFT_PIN_GAIN 8
// Automatic UI detection writes this many decision texels; its statistics
// rows hold the cells of this many scene-evidence images (docs/reshade-sbs.md,
// UI detection flags and decision texels).
#define SUNSHINE_UI_DECISION_TEXELS 17
#define SUNSHINE_UI_SCENE_EVIDENCE_IMAGES 2
// Each of the 16x16 statistics tiles is counted by this many groups (the
// tiles pass's group z); part k of tile (x, y) stores its counts at
// statistics column x + 16k, and the reduce sums the parts.
#define SUNSHINE_UI_DETECTION_TILE_PARTS 4
// Candidate layout 2 (UI framework S1, E1), mirrored from
// game3d_ui_detection_contract.h: every candidate has its own slot and bit in
// Sunshine_UICandidates and Sunshine_UIAcceptedCandidates; the offscreen UI
// layer (t7) never shares the UI color tag's slot (t12) and decides as source
// 10. SunshineUIDetectionReduceCS ports ui_selection::decide
// (game3d_ui_selection.h) line for line, the T1 grace and its hold store and
// the H1 override included; this revision must equal ui_selection::revision.
// Revision 7 decided exactly as revision 5: revision 6 (fix 3 and fix 4,
// removed by user decision) added candidate bit 0x100 and source 12, which
// stay reserved. Revision 8 adds the empty change set; revision 9 removes
// H2's still screen (source 11, reserved) and S3's frame identity (decision
// texel 12, reserved). Revision 10 keeps T1's held decision across HUD-less
// re-offers, judges the declared alphas one way (decision texel 16) on
// sample frames only, bounds the layer's premultiplied test by its covered
// pixels and lets H1 override every S1 winner.
#define SUNSHINE_UI_CANDIDATE_LAYOUT 2
#define SUNSHINE_UI_SELECTION_REVISION 10
#define SUNSHINE_UI_CANDIDATE_UI_ALPHA 0x1u
#define SUNSHINE_UI_CANDIDATE_UI_COLOR 0x2u
#define SUNSHINE_UI_CANDIDATE_BACKBUFFER 0x4u
#define SUNSHINE_UI_CANDIDATE_CURRENT 0x8u
#define SUNSHINE_UI_CANDIDATE_HUDLESS 0x10u
#define SUNSHINE_UI_CANDIDATE_EXACT 0x20u
#define SUNSHINE_UI_CANDIDATE_LAYER 0x40u
#define SUNSHINE_UI_SOURCE_LAYER 10
// Hidden-scene evidence D (docs/reshade-sbs.md, hidden-scene evidence),
// mirrored from game3d_ui_detection_contract.h: a grid of cells over the
// frame, the parallax step of a depth edge in pixels per 2160 rows, the edge
// cells valid evidence needs, the hidden and visible bounds of D in percent,
// the fixed point of the cell sums, and H1's shares of nearly opaque pixels
// (opaque-full) and of changed pixels (a HUD-less pre-UI claim) in percent.
#define SUNSHINE_UI_SCENE_CELLS_X 256
#define SUNSHINE_UI_SCENE_CELLS_Y 144
#define SUNSHINE_UI_SCENE_EDGE_PX 4
#define SUNSHINE_UI_SCENE_MIN_EDGES 128
#define SUNSHINE_UI_SCENE_HIDDEN_PERCENT 15
#define SUNSHINE_UI_SCENE_VISIBLE_PERCENT 25
#define SUNSHINE_UI_SCENE_LUMA_SCALE 1048576
#define SUNSHINE_UI_SCENE_PARALLAX_SCALE 4096
#define SUNSHINE_UI_SCENE_OPAQUE_PERCENT 99
#define SUNSHINE_UI_SCENE_HUDLESS_CHANGED_PERCENT 90
// The first statistics row of the one-way judgment counts (A2: 16 rows of
// strong pixels, then 16 of contradicted ones, of UIAlpha, the UI color tag,
// Backbuffer and current alpha), of the scene compare groups' partial sums,
// below the 112 rows of per-tile detection counts, of the per-tile pre-UI
// pixel counts (H1 d: the offscreen UI layer against the presented frame,
// rows 128-143), and of the layer's per-tile pixels beyond the premultiplied
// bound (V1, rows 208-223; rows 144-207 are reserved).
#define SUNSHINE_UI_JUDGMENT_ROW 80
#define SUNSHINE_UI_SCENE_PARTIAL_ROW 112
#define SUNSHINE_UI_PRE_UI_ROW 128
#define SUNSHINE_UI_LAYER_BOUND_ROW 208
// Exact per-session UI counters (docs/reshade-sbs.md, UI counters), mirrored
// from game3d_ui_counters.h: the detection reduce adds every detection frame
// to these words of SunshineUICountersStore. The no-mask reasons are offsets
// from SUNSHINE_UI_COUNTER_NONE. The decided words of sources 11 and 12 and
// word 30 (H2 and fix 3, removed) are reserved and stay zero, and so are
// words 14 and 18 (S1's invariants, zero by construction) since revision 10.
#define SUNSHINE_UI_COUNTER_WORDS 31
#define SUNSHINE_UI_COUNTER_DETECTION_FRAMES 0
#define SUNSHINE_UI_COUNTER_DECIDED 1
#define SUNSHINE_UI_COUNTER_INEXACT_DIFFERENCE 15
#define SUNSHINE_UI_COUNTER_DEPTH_NOT_CURRENT 16
#define SUNSHINE_UI_COUNTER_CONTRADICTED 17
#define SUNSHINE_UI_COUNTER_NONE 19
#define SUNSHINE_UI_COUNTER_FULL_ALPHA 28
#define SUNSHINE_UI_COUNTER_REUSED 29
#define SUNSHINE_UI_NONE_LAYER_ASIDE 0
#define SUNSHINE_UI_NONE_TRUSTED_INVALID 1
#define SUNSHINE_UI_NONE_PRESENTED_BLOCKED 2
#define SUNSHINE_UI_NONE_AMBIGUOUS 3
#define SUNSHINE_UI_NONE_DIFFERENCE_FAILED 4
#define SUNSHINE_UI_NONE_GATE_NO_HOLD 5
#define SUNSHINE_UI_NONE_NO_CANDIDATE 6
#define SUNSHINE_UI_NONE_OTHER 7
#define SUNSHINE_UI_NONE_UNACCEPTED 8
//
// Specialize BUFFER_WIDTH, BUFFER_HEIGHT and BUFFER_COLOR_SPACE at compile time.
// This preserves the original per-resolution group-memory footprint. Color-space
// values are 1=SDR sRGB, 2=linear scRGB, 3=HDR10 PQ. All SRVs/RTVs are non-sRGB.
#if !defined(BUFFER_WIDTH) || !defined(BUFFER_HEIGHT) || !defined(BUFFER_COLOR_SPACE)
#error Native Game 3D requires source dimensions and color-space specialization.
#endif
#if BUFFER_WIDTH <= 0 || BUFFER_HEIGHT <= 0 || BUFFER_WIDTH > 8192 || BUFFER_HEIGHT > 8192 || (BUFFER_WIDTH & 1) || (BUFFER_HEIGHT & 1)
#error Native Game 3D requires even source dimensions up to 8192x8192.
#endif
#if BUFFER_COLOR_SPACE != 1 && BUFFER_COLOR_SPACE != 2 && BUFFER_COLOR_SPACE != 3
#error Native Game 3D requires a known source color space.
#endif
#if BUFFER_WIDTH <= 3840 && BUFFER_HEIGHT <= 3840
#define _SUNSHINE_HOST_WARP_SUPPORTED 1
#else
#define _SUNSHINE_HOST_WARP_SUPPORTED 0
#endif

// Stable 80-byte ABI. Readiness flags are 32-bit integers, never C++ bools.
cbuffer SunshineGame3DConstants : register(b0)
{
    float Depth_Adjustment : packoffset(c0.x);
    int Depth_Map_View : packoffset(c0.y);
    uint Sunshine_DepthReady : packoffset(c0.z);
    uint Sunshine_CameraDepthReady : packoffset(c0.w);
    int Sunshine_CameraCoordinateBasis : packoffset(c1.x);
    float Sunshine_CameraDepthScale : packoffset(c1.y);
    float Sunshine_CameraStrengthBlend : packoffset(c1.z);
    float Sunshine_DisparityLimitUv : packoffset(c1.w);
    float2 Sunshine_CameraProjection : packoffset(c2.x);
    float2 Sunshine_CameraRawDepthRange : packoffset(c2.z);
    float2 Sunshine_CameraConvergence : packoffset(c3.x);
    float2 Sunshine_DepthJitter : packoffset(c3.z);
    float4 Sunshine_CameraDepthRect : packoffset(c4);
};

// Independent UI contract; preserve the existing 80-byte geometry ABI.
// Live coverage policy supplies the effective switch; replay uses its frozen
// selection without re-running the temporal observation policy.
cbuffer SunshineUIConstants : register(b1)
{
    uint Sunshine_SourceAlphaUI;
    uint Sunshine_UIPlaneMode;
    float Sunshine_UIPlaneInverseDepth;
    uint Sunshine_UIMaskChannel; // 0 = alpha; 1 = red from an explicit single-channel UIAlpha tag.
};

Texture2D<float4> SunshineSourceSampler : register(t0);
Texture2D<float> DepthBuffer : register(t1);
Texture2D<float4> SunshineLinearClamp : register(t2);
Texture2D<float> SunshineHostCandidateSampler : register(t3);
Texture2D<float> SunshineHostVerticalConditionedSampler : register(t4);
Texture2D<float> SunshineHostFinalSampler : register(t5);
// The current presented color as copied, before PQ linearization; t0 may be
// the color a HUD-less image is paired with. Read by scene evidence and the
// detection tiles' pre-UI pixel counts only.
Texture2D<float4> SunshinePresentedColor : register(t6);
// The offscreen UI layer candidate (SUNSHINE_UI_CANDIDATE_LAYER); read by
// automatic detection (its alpha, and its colour against the presented
// color) and, as a pre-UI scene image, by scene evidence only.
Texture2D<float4> SunshineUILayer : register(t7);
Texture2D<float> SunshineUIPlaneTilesSampler : register(t8);
Texture2D<float> SunshineUIPlaneResolvedSampler : register(t9);
Texture2D<uint4> SunshineUIDetectionSampler : register(t10);
Texture2D<float4> SunshineUIDedicatedAlpha : register(t11);
Texture2D<float4> SunshineUIColorAlpha : register(t12);
Texture2D<float4> SunshineUIBackbufferAlpha : register(t13);
Texture2D<float4> SunshineHUDless : register(t14);
// Sunshine_UIDetectionFlags (docs/reshade-sbs.md, UI detection flags and
// decision texels), mirrored from game3d_ui_detection_contract.h. Stored bits
// describe the offscreen UI layer slot (t7): it must pass the premultiplied
// bound, with a float layer's HDR headroom, and is the one-frame-late copy.
// Per-frame bits ride in one render's pushed word only: the offered UIAlpha
// or UI color tag was not captured in the exact pair's tag batch (its
// candidate bit from bit SUNSHINE_UI_PER_FRAME_UNALIGNED_SHIFT; the one-way
// test does not judge it), a status sample,
// whose decision texels the CPU reads (only it counts the one-way judgment
// and the pre-UI pixels), the offered HUD-less snapshot is the previous
// render's, re-offered (T1), the consumed depth is not this frame's, an
// accepted candidate of the previous real frame is missing (T1), no previous
// real decision exists in this chain, so the hold store reads as none (T1),
// the CPU holds a hidden verdict of D on the presented frame (H1), its
// samples read the pre-UI scene image visible (H1), and, from bit
// SUNSHINE_UI_PER_FRAME_REFUTED_SHIFT, the candidate bits whose full claims a
// visible verdict refuted (H1), and the offered layer's signature is proven
// the pre-UI scene image (H1 d, the acceptance ledger's pre-UI proof). Every
// layer is the late copy (stored 0x4), which no pass judges (A2).
#define SUNSHINE_UI_STORED_PREMULTIPLIED 0x1u
#define SUNSHINE_UI_STORED_HDR_HEADROOM 0x2u
#define SUNSHINE_UI_STORED_LATE_LAYER 0x4u
// 0x8u (stored), 0x10000u, 0x20000u, 0x80000u and 0x20000000u (per-frame)
// are reserved and never reused.
#define SUNSHINE_UI_PER_FRAME_UNALIGNED_SHIFT 12
#define SUNSHINE_UI_PER_FRAME_SAMPLE 0x4000u
#define SUNSHINE_UI_PER_FRAME_REOFFER 0x8000u
#define SUNSHINE_UI_PER_FRAME_DEPTH_NOT_CURRENT 0x40000u
#define SUNSHINE_UI_PER_FRAME_ACCEPTED_MISSING 0x100000u
#define SUNSHINE_UI_PER_FRAME_HOLD_RESET 0x200000u
#define SUNSHINE_UI_PER_FRAME_SCENE_HIDDEN 0x400000u
#define SUNSHINE_UI_PER_FRAME_PRE_UI_VISIBLE 0x800000u
#define SUNSHINE_UI_PER_FRAME_REFUTED_SHIFT 24
#define SUNSHINE_UI_PER_FRAME_PRE_UI_PROVEN 0x80000000u
// H1, mirrored from game3d_ui_detection_contract.h: the pre-UI scene image's
// claim bit beside the candidate bits, the h1 word's applied bit above the
// S1 winner's source, and the pre-UI image of decision texel 6 .w.
#define SUNSHINE_UI_CLAIM_PRE_UI 0x80u
#define SUNSHINE_UI_H1_APPLIED 0x100u
#define SUNSHINE_UI_PRE_UI_IMAGE_HUDLESS 1
#define SUNSHINE_UI_PRE_UI_IMAGE_LAYER 2
// The T1 hold store (SunshineUIHoldStore), mirrored from
// game3d_ui_detection_contract.h: texels {state, applied source, applied
// covered} of the last real detection, state none, own or spent; and the
// frame reason word of decision texel 9 .w (the own decision's no-mask
// reason, or DECIDED, plus REUSED when the grace applied the stored one).
#define SUNSHINE_UI_HOLD_NONE 0
#define SUNSHINE_UI_HOLD_OWN 1
#define SUNSHINE_UI_HOLD_SPENT 2
#define SUNSHINE_UI_HOLD_STORE_TEXELS 3
#define SUNSHINE_UI_FRAME_REASON_DECIDED 0xFFu
#define SUNSHINE_UI_FRAME_REASON_REUSED 0x10000u
cbuffer SunshineUIDetectionConstants : register(b2)
{
    // SUNSHINE_UI_CANDIDATE_* bits: 0x1 UIAlpha (t11), 0x2 UI color tag (t12),
    // 0x4 Backbuffer (t13), 0x8 current alpha (t0), 0x10 paired HUD-less
    // (t14), 0x20 exact pair, 0x40 offscreen UI layer (t7).
    uint Sunshine_UICandidates;
    float Sunshine_UIDifferenceThreshold;
    // The candidates the game session accepts (A1), in the same bit positions.
    uint Sunshine_UIAcceptedCandidates;
    uint Sunshine_UIDetectionFlags; // SUNSHINE_UI_STORED_* and SUNSHINE_UI_PER_FRAME_* bits.
    // The pair threshold of the offscreen UI layer (t7) and the presented
    // color (t6) from their own encodings; zero without a layer, when the
    // two are not comparable or on a frame that is not a detection sample
    // (H1 d, the pre-UI pixel counts, which the CPU reads from samples only).
    float Sunshine_UIPreUIThreshold;
    // Reserved, pushed as zero: H2's still-screen flag (0x1) and fix 3 and
    // fix 4's bits 0x2-0x200 used it, all removed.
    uint Sunshine_UIReserved;
};
RWTexture2D<float> SunshineHostCandidateStore : register(u0);
RWTexture2D<float> SunshineHostVerticalMajorantStore : register(u1);
RWTexture2D<float> SunshineHostVerticalConditionedStore : register(u2);
RWTexture2D<float> SunshineHostFinalStore : register(u3);
RWTexture2D<float> SunshineUIPlaneTilesStore : register(u4);
RWTexture2D<float> SunshineUIPlaneResolvedStore : register(u5);
RWTexture2D<uint4> SunshineAlphaCoverageStore : register(u6);
RWTexture2D<uint4> SunshineUIConflictStore : register(u7);
// Bound at u7 for the detection reduce only (SunshineUIConflictCS binds its
// statistics there); unbound, as in offline replay, its adds are dropped.
RWTexture2D<uint> SunshineUICountersStore : register(u7);
// The T1 hold store, SUNSHINE_UI_HOLD_STORE_TEXELS R32_UINT texels bound at
// u5 for the detection reduce only (SunshineUINearestReduceCS binds the
// resolved UI plane there); unbound, as in offline replay, it reads
// SUNSHINE_UI_HOLD_NONE and its writes are dropped.
RWTexture2D<uint> SunshineUIHoldStore : register(u5);
SamplerState SunshinePointClamp : register(s0);
SamplerState SunshineLinearClampState : register(s1);
SamplerState SunshinePointBorder : register(s2);

float2 SunshineDepthAllocationSize()
{
    uint width, height;
    DepthBuffer.GetDimensions(width, height);
    return float2(width, height);
}
bool SunshineCameraFinite(float value)
{
    return (asuint(value) & 0x7f800000u) != 0x7f800000u;
}
float SunshineSelectedUIAlpha(uint2 coordinate)
{
    float4 value = SunshineSourceSampler.Load(int3(int2(coordinate), 0));
    return Sunshine_UIMaskChannel == 1u ? value.r : value.a;
}

// Auto validates each CURRENT pair on the GPU. No prior CPU observation can
// authorize a changed candidate: session acceptance only names the sources
// that carry UI coverage, and this frame's own pixels are the mask. The
// tiles pass reads UIAlpha (t11 .r), the UI color tag (t12 .a), the
// Backbuffer (t13 .a) and the current alpha (t0 .a).
float SunshineHUDlessDifferenceOf(float3 a, float3 b, out bool valid)
{
    valid = all(isfinite(a)) && all(isfinite(b));
    // scRGB is unbounded linear light; use a relative tolerance above one.
    float scale = BUFFER_COLOR_SPACE == 2 ? max(1.0, max(max(abs(a.r), abs(a.g)), abs(a.b))) : 1.0;
    return max(max(abs(a.r-b.r), abs(a.g-b.g)), abs(a.b-b.b)) / scale;
}
float SunshineHUDlessDifference(uint2 xy, out bool valid)
{
    return SunshineHUDlessDifferenceOf(SunshineSourceSampler.Load(int3(xy, 0)).rgb, SunshineHUDless.Load(int3(xy, 0)).rgb,
        valid);
}
// The pre-UI pixel counts of one pixel (H1 d): the offscreen UI layer's
// colour against the presented colour at a coarse threshold, eight times
// their pair threshold as the lit test of a HUD-less image: {matching, lit
// layer, 0, 0} (.z and .w held the removed lit-presented shadow statistics),
// all zero when the pair is not comparable (threshold zero) or either colour
// is not finite. scRGB is compared with the same relative tolerance above one
// as SunshineHUDlessDifference.
uint4 SunshinePreUIPixel(uint2 xy, float3 layer)
{
    const float coarse = Sunshine_UIPreUIThreshold * 8.0;
    if (!(coarse > 0.0)) return 0u;
    float3 presented = SunshinePresentedColor.Load(int3(xy, 0)).rgb;
    if (!all(isfinite(presented)) || !all(isfinite(layer))) return 0u;
    float presentedPeak = max(max(abs(presented.r), abs(presented.g)), abs(presented.b));
    float layerPeak = max(max(abs(layer.r), abs(layer.g)), abs(layer.b));
    float scale = BUFFER_COLOR_SPACE == 2 ? max(1.0, presentedPeak) : 1.0;
    float3 difference = abs(presented - layer);
    float delta = max(max(difference.r, difference.g), difference.b) / scale;
    return uint4(delta <= coarse ? 1u : 0u, layerPeak > coarse ? 1u : 0u, 0u, 0u);
}
// Each of the 16x16 tiles is counted by SUNSHINE_UI_DETECTION_TILE_PARTS
// groups of 256 threads (group z: part k takes the tile's runs of 16 rows
// whose index is k modulo the parts); the integer counts do not depend on how
// pixels are split among them, and the reduce adds the parts.
groupshared uint4 SunshineUIDetectionCoverage[256];
groupshared uint4 SunshineUIDetectionInvalid[256];
groupshared uint4 SunshineUIDetectionDifference[256];
// Lit HUD-less pixels, then nearly opaque pixels of UIAlpha, the UI color tag
// and the Backbuffer.
groupshared uint4 SunshineUIDetectionLit[256];
// The offscreen UI layer's covered pixels within the premultiplied bound, its
// out-of-range pixels and its nearly opaque pixels within the bound, then the
// current alpha's nearly opaque pixels.
groupshared uint4 SunshineUIDetectionLayer[256];
// The layer's pixels beyond the premultiplied bound (V1).
groupshared uint SunshineUIDetectionBound[256];
// Whether a layer pixel lies beyond the premultiplied bound (V1). An
// offscreen layer is UI only when blended over transparent black: its color
// stays within a small multiple of its alpha. UI tinted brighter than white
// (Stellar Blade's pulsing markers, up to twice its alpha) qualifies; the
// saturated pixels of a scene buffer whose alpha is not coverage (Dead Space:
// a luma-like alpha, mean 11/255) and a scene image drawn into the cleared
// target (Stellar Blade in SDR, alpha 0) do not, though the dark or grey
// pixels of a luma-like alpha stay within it. scRGB UI may be up to 10000
// nits. Only a cleared layer (SUNSHINE_UI_STORED_PREMULTIPLIED) is bounded.
bool SunshineUILayerBeyondBound(float4 layer)
{
    const bool premultiplied = (Sunshine_UIDetectionFlags & SUNSHINE_UI_STORED_PREMULTIPLIED) != 0u;
    const float headroom = Sunshine_UIDetectionFlags & SUNSHINE_UI_STORED_HDR_HEADROOM ? 125.0 : 2.0;
    return premultiplied && max(max(layer.r, layer.g), layer.b) > layer.a * headroom + 4.0 / 255.0;
}
[numthreads(16, 16, 1)]
void SunshineUIDetectionTilesCS(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint2 first = group.xy * uint2(BUFFER_WIDTH, BUFFER_HEIGHT) / 16u;
    uint2 last = (group.xy + 1u) * uint2(BUFFER_WIDTH, BUFFER_HEIGHT) / 16u;
    // This part's statistics column.
    const uint2 column = uint2(group.x + 16u * group.z, group.y);
    uint4 coverage = 0u, invalid = 0u, difference = 0u, lit = 0u, layerCounts = 0u;
    uint beyondBound = 0u;
    // The sample phase's sums: the one-way judgment and the pre-UI pixels.
    uint4 strong = 0u, contradicted = 0u, preUI = 0u;
    // Presented, Backbuffer and tagged alpha are checked for range only; a
    // layer pixel of in-range alpha beyond the premultiplied bound counts
    // apart, never as covered or opaque, and the reduce decides whether such
    // pixels invalidate the whole layer (V1).
    // A2 counts on a status sample (SUNSHINE_UI_PER_FRAME_SAMPLE) with an
    // exact pair only, the one judge that reads them: UIAlpha, the UI color
    // tag, Backbuffer and current alpha. The one-frame-late layer copy is not
    // same-sample evidence (E2) and is never judged, nor is a declared tag
    // captured in another tag batch than the pair (unaligned): neither
    // counts a strong pixel.
    const uint pairBits = SUNSHINE_UI_CANDIDATE_HUDLESS | SUNSHINE_UI_CANDIDATE_EXACT;
    const bool sample = (Sunshine_UIDetectionFlags & SUNSHINE_UI_PER_FRAME_SAMPLE) != 0u;
    const bool judging = sample && (Sunshine_UICandidates & pairBits) == pairBits;
    const uint unaligned = Sunshine_UIDetectionFlags >> SUNSHINE_UI_PER_FRAME_UNALIGNED_SHIFT;
    const uint4 judged = uint4((unaligned & SUNSHINE_UI_CANDIDATE_UI_ALPHA) ? 0u : 1u,
        (unaligned & SUNSHINE_UI_CANDIDATE_UI_COLOR) ? 0u : 1u, 1u, 1u);
    // A candidate that is not offered is not bound: uniform branches skip its
    // loads and use the zero an unbound view reads. The presented color (t0)
    // is always bound and always read.
    const uint offered = Sunshine_UICandidates;
    [loop] for (uint y = first.y + thread.y + 16u * group.z; y < last.y; y += 16u * SUNSHINE_UI_DETECTION_TILE_PARTS)
    [loop] for (uint x = first.x + thread.x; x < last.x; x += 16u) {
        const int3 at = int3(x, y, 0);
        const float4 presentedPair = SunshineSourceSampler.Load(at);
        float4 a = float4(0.0, 0.0, 0.0, presentedPair.a);
        [branch] if (offered & SUNSHINE_UI_CANDIDATE_UI_ALPHA) a.x = SunshineUIDedicatedAlpha.Load(at).r;
        [branch] if (offered & SUNSHINE_UI_CANDIDATE_UI_COLOR) a.y = SunshineUIColorAlpha.Load(at).a;
        [branch] if (offered & SUNSHINE_UI_CANDIDATE_BACKBUFFER) a.z = SunshineUIBackbufferAlpha.Load(at).a;
        bool4 okay = isfinite(a) && a >= 0.0 && a <= 1.0;
        coverage += uint4(okay && a > 0.0);
        invalid += uint4(!okay);
        // H1's opaque-full test: UIAlpha, UI color, Backbuffer and current
        // alpha of at least 254/255 (finite and in range).
        uint4 opaque = uint4(okay && a >= 254.0 / 255.0);
        lit.yzw += opaque.xyz;
        layerCounts.w += opaque.w;
        float4 layer = 0.0;
        [branch] if (offered & SUNSHINE_UI_CANDIDATE_LAYER) layer = SunshineUILayer.Load(at);
        bool layerOkay = isfinite(layer.a) && layer.a >= 0.0 && layer.a <= 1.0;
        bool beyond = layerOkay && SunshineUILayerBeyondBound(layer);
        // Covered: in (0, 1], which is finite and in range, and within the
        // bound. FXC folds the scalar "layerOkay && layer.a > 0.0" into
        // layerOkay alone.
        layerCounts.x += layer.a > 0.0 && layer.a <= 1.0 && !beyond ? 1u : 0u;
        layerCounts.y += layerOkay ? 0u : 1u;
        layerCounts.z += layerOkay && layer.a >= 254.0 / 255.0 && !beyond ? 1u : 0u;
        beyondBound += beyond ? 1u : 0u;
        float3 hudless = 0.0;
        [branch] if (offered & SUNSHINE_UI_CANDIDATE_HUDLESS) hudless = SunshineHUDless.Load(at).rgb;
        bool finite;
        float delta = SunshineHUDlessDifferenceOf(presentedPair.rgb, hudless, finite);
        bool unchanged = finite && delta <= Sunshine_UIDifferenceThreshold * .5;
        difference.x += finite && delta > Sunshine_UIDifferenceThreshold ? 1u : 0u;
        difference.y += finite ? 0u : 1u;
        difference.z += unchanged ? 1u : 0u;
        difference.w += 1u;
        // A HUD-less pixel that shows scene content rather than black.
        bool litPixel = finite && max(max(abs(hudless.r), abs(hudless.g)), abs(hudless.b)) > Sunshine_UIDifferenceThreshold * 8.0;
        lit.x += litPixel ? 1u : 0u;
        // A2, one way: a strong pixel (alpha in [1/2, 1], so finite)
        // contradicts its source where the exact HUD-less image is lit and
        // unchanged, the scene shown without UI. Real UI changes the pixels it
        // covers, and dims and tints over dark or changed pixels never do.
        [branch] if (judging) {
            const uint4 strongPixel = uint4(a >= .5 && a <= 1.0) & judged;
            strong += strongPixel;
            contradicted += litPixel && unchanged ? strongPixel : uint4(0u, 0u, 0u, 0u);
        }
        // H1 (d): the layer's colour against the presented colour (zero
        // without a pre-UI threshold, which only a sample frame pushes).
        preUI += SunshinePreUIPixel(uint2(x, y), layer.rgb);
    }
    uint lane = thread.y * 16u + thread.x;
    SunshineUIDetectionCoverage[lane] = coverage;
    SunshineUIDetectionInvalid[lane] = invalid;
    SunshineUIDetectionDifference[lane] = difference;
    SunshineUIDetectionLit[lane] = lit;
    SunshineUIDetectionLayer[lane] = layerCounts;
    SunshineUIDetectionBound[lane] = beyondBound;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint step = 128u; step; step >>= 1u) {
        if (lane < step) {
            SunshineUIDetectionCoverage[lane] += SunshineUIDetectionCoverage[lane+step];
            SunshineUIDetectionInvalid[lane] += SunshineUIDetectionInvalid[lane+step];
            SunshineUIDetectionDifference[lane] += SunshineUIDetectionDifference[lane+step];
            SunshineUIDetectionLit[lane] += SunshineUIDetectionLit[lane+step];
            SunshineUIDetectionLayer[lane] += SunshineUIDetectionLayer[lane+step];
            SunshineUIDetectionBound[lane] += SunshineUIDetectionBound[lane+step];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (!lane) {
        SunshineAlphaCoverageStore[column] = SunshineUIDetectionCoverage[0];
        SunshineAlphaCoverageStore[column + uint2(0,16)] = SunshineUIDetectionInvalid[0];
        SunshineAlphaCoverageStore[column + uint2(0,32)] = SunshineUIDetectionDifference[0];
        SunshineAlphaCoverageStore[column + uint2(0,48)] = SunshineUIDetectionLit[0];
        SunshineAlphaCoverageStore[column + uint2(0,64)] = SunshineUIDetectionLayer[0];
        SunshineAlphaCoverageStore[column + uint2(0,SUNSHINE_UI_LAYER_BOUND_ROW)] = uint4(SunshineUIDetectionBound[0], 0u, 0u, 0u);
    }
    // The sample phase: on a status sample a second pass sums the one-way
    // judgment and the pre-UI pixel counts in the coverage, invalid and
    // difference arrays once lane 0 stored their sums (group-shared memory is
    // near the cs_5_0 limit). Every other frame skips it and stores zero
    // counts, which its reduce never reads.
    if (sample) {
        GroupMemoryBarrierWithGroupSync();
        SunshineUIDetectionCoverage[lane] = strong;
        SunshineUIDetectionInvalid[lane] = contradicted;
        SunshineUIDetectionDifference[lane] = preUI;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint half_step = 128u; half_step; half_step >>= 1u) {
            if (lane < half_step) {
                SunshineUIDetectionCoverage[lane] += SunshineUIDetectionCoverage[lane+half_step];
                SunshineUIDetectionInvalid[lane] += SunshineUIDetectionInvalid[lane+half_step];
                SunshineUIDetectionDifference[lane] += SunshineUIDetectionDifference[lane+half_step];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (!lane) {
            SunshineAlphaCoverageStore[column + uint2(0,SUNSHINE_UI_JUDGMENT_ROW)] = SunshineUIDetectionCoverage[0];
            SunshineAlphaCoverageStore[column + uint2(0,SUNSHINE_UI_JUDGMENT_ROW + 16u)] = SunshineUIDetectionInvalid[0];
            SunshineAlphaCoverageStore[column + uint2(0,SUNSHINE_UI_PRE_UI_ROW)] = SunshineUIDetectionDifference[0];
        }
    } else if (!lane) {
        SunshineAlphaCoverageStore[column + uint2(0,SUNSHINE_UI_JUDGMENT_ROW)] = 0u;
        SunshineAlphaCoverageStore[column + uint2(0,SUNSHINE_UI_JUDGMENT_ROW + 16u)] = 0u;
        SunshineAlphaCoverageStore[column + uint2(0,SUNSHINE_UI_PRE_UI_ROW)] = 0u;
    }
}
// One tile's counts in a statistics row: the sum of its parts (a statistics
// texture only 16 columns wide reads zero beyond them).
uint4 SunshineUIDetectionTileCounts(uint2 tile, uint row)
{
    uint4 sum = 0u;
    [unroll] for (uint part = 0u; part < SUNSHINE_UI_DETECTION_TILE_PARTS; ++part)
        sum += SunshineUIDetectionSampler.Load(int3(tile.x + 16u * part, tile.y + row, 0));
    return sum;
}
// One thread per tile, then an exact group sum; thread 0 decides. The
// decision is ui_selection::decide (game3d_ui_selection.h) line for line, in
// the same uint32 arithmetic (counts stay below 3840 x 3840, so no product
// overflows); test_game3d_ui_selection_contract runs this pass against it.
groupshared uint SunshineUIDetectionMatching[256];
// The first candidate of a bit set in draw order (ui_selection::draw_order):
// UIAlpha, UI color tag, layer, Backbuffer, current, HUD-less.
uint SunshineUIFirstCandidate(uint bits)
{
    return (bits & SUNSHINE_UI_CANDIDATE_UI_ALPHA) ? SUNSHINE_UI_CANDIDATE_UI_ALPHA :
        (bits & SUNSHINE_UI_CANDIDATE_UI_COLOR) ? SUNSHINE_UI_CANDIDATE_UI_COLOR :
        (bits & SUNSHINE_UI_CANDIDATE_LAYER) ? SUNSHINE_UI_CANDIDATE_LAYER :
        (bits & SUNSHINE_UI_CANDIDATE_BACKBUFFER) ? SUNSHINE_UI_CANDIDATE_BACKBUFFER :
        (bits & SUNSHINE_UI_CANDIDATE_CURRENT) ? SUNSHINE_UI_CANDIDATE_CURRENT :
        (bits & SUNSHINE_UI_CANDIDATE_HUDLESS);
}
[numthreads(256, 1, 1)]
void SunshineUIDetectionReduceCS(uint3 thread : SV_GroupThreadID)
{
    uint lane = thread.x;
    uint2 tile = uint2(lane % 16u, lane / 16u);
    uint4 d = SunshineUIDetectionTileCounts(tile, 32u);
    SunshineUIDetectionCoverage[lane] = SunshineUIDetectionTileCounts(tile, 0u);
    SunshineUIDetectionInvalid[lane] = SunshineUIDetectionTileCounts(tile, 16u);
    SunshineUIDetectionDifference[lane] = d;
    SunshineUIDetectionLit[lane] = SunshineUIDetectionTileCounts(tile, 48u);
    SunshineUIDetectionLayer[lane] = SunshineUIDetectionTileCounts(tile, 64u);
    SunshineUIDetectionBound[lane] = SunshineUIDetectionTileCounts(tile, SUNSHINE_UI_LAYER_BOUND_ROW).x;
    SunshineUIDetectionMatching[lane] = d.w && d.z * 100u >= d.w * 99u ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint step = 128u; step; step >>= 1u) {
        if (lane < step) {
            SunshineUIDetectionCoverage[lane] += SunshineUIDetectionCoverage[lane+step];
            SunshineUIDetectionInvalid[lane] += SunshineUIDetectionInvalid[lane+step];
            SunshineUIDetectionDifference[lane] += SunshineUIDetectionDifference[lane+step];
            SunshineUIDetectionLit[lane] += SunshineUIDetectionLit[lane+step];
            SunshineUIDetectionLayer[lane] += SunshineUIDetectionLayer[lane+step];
            SunshineUIDetectionBound[lane] += SunshineUIDetectionBound[lane+step];
            SunshineUIDetectionMatching[lane] += SunshineUIDetectionMatching[lane+step];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    // The sample phase sums the one-way judgment (A2) and the pre-UI pixel
    // counts (H1 d) in the coverage, invalid and difference arrays once every
    // lane holds their sums; a frame that is not a status sample skips it,
    // and its one-way and pre-UI counts are zero.
    const uint4 coverage_sum = SunshineUIDetectionCoverage[0], invalid_sum = SunshineUIDetectionInvalid[0];
    const uint4 difference_sum = SunshineUIDetectionDifference[0];
    uint4 strong = 0u, contradicted = 0u, pre_ui_sum = 0u;
    if (Sunshine_UIDetectionFlags & SUNSHINE_UI_PER_FRAME_SAMPLE) {
        GroupMemoryBarrierWithGroupSync();
        SunshineUIDetectionCoverage[lane] = SunshineUIDetectionTileCounts(tile, SUNSHINE_UI_JUDGMENT_ROW);
        SunshineUIDetectionInvalid[lane] = SunshineUIDetectionTileCounts(tile, SUNSHINE_UI_JUDGMENT_ROW + 16u);
        SunshineUIDetectionDifference[lane] = SunshineUIDetectionTileCounts(tile, SUNSHINE_UI_PRE_UI_ROW);
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint half_step = 128u; half_step; half_step >>= 1u) {
            if (lane < half_step) {
                SunshineUIDetectionCoverage[lane] += SunshineUIDetectionCoverage[lane+half_step];
                SunshineUIDetectionInvalid[lane] += SunshineUIDetectionInvalid[lane+half_step];
                SunshineUIDetectionDifference[lane] += SunshineUIDetectionDifference[lane+half_step];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        strong = SunshineUIDetectionCoverage[0];
        contradicted = SunshineUIDetectionInvalid[0];
        pre_ui_sum = SunshineUIDetectionDifference[0];
    }
    if (lane) return;
    uint4 coverage = coverage_sum, invalid = invalid_sum;
    uint4 difference = difference_sum;
    uint matching_tiles = SunshineUIDetectionMatching[0], lit = SunshineUIDetectionLit[0].x;
    uint2 opaque = SunshineUIDetectionLit[0].yz;
    const uint4 layer_sum = SunshineUIDetectionLayer[0];
    const uint beyond = SunshineUIDetectionBound[0];
    // Nearly opaque pixels of the Backbuffer and the current alpha (H1).
    const uint opaque_backbuffer = SunshineUIDetectionLit[0].w, opaque_current = layer_sum.w;
    const uint pixels = difference.w;
    // The layer's covered pixels (within the premultiplied bound), invalid
    // pixels and opaque pixels (within the bound). V1 (revision 10): its
    // pixels beyond the bound invalidate it when they lie on more than 1% of
    // the frame and either on more than 5% of it or on more pixels than its
    // opaque ones (within the bound), as a scene image in the target does
    // (Stellar Blade's SDR scene, or a luma-like alpha whose pixels are
    // rarely opaque); below that they are uncovered, not UI (an additive glow
    // beside an opaque HUD).
    const bool scene_like = beyond * 100u > pixels && (beyond * 20u > pixels || beyond > layer_sum.z);
    const uint3 layer = uint3(layer_sum.x, layer_sum.y + (scene_like ? beyond : 0u), layer_sum.z);
    // The layer against the presented frame (H1 d): {matching, lit layer, 0,
    // 0} (words 46 and 47 are reserved zeros); zero without an offered layer
    // or a pre-UI threshold.
    const uint4 pre_ui_pixels = (Sunshine_UICandidates & SUNSHINE_UI_CANDIDATE_LAYER) && Sunshine_UIPreUIThreshold > 0.0 ?
        uint4(pre_ui_sum.xy, 0u, 0u) : 0u;
    const uint offered = Sunshine_UICandidates, accepted = Sunshine_UIAcceptedCandidates;
    const uint alpha_bits = SUNSHINE_UI_CANDIDATE_UI_ALPHA | SUNSHINE_UI_CANDIDATE_UI_COLOR |
        SUNSHINE_UI_CANDIDATE_BACKBUFFER | SUNSHINE_UI_CANDIDATE_CURRENT | SUNSHINE_UI_CANDIDATE_LAYER;
    const uint declared_alpha_bits = SUNSHINE_UI_CANDIDATE_UI_ALPHA | SUNSHINE_UI_CANDIDATE_UI_COLOR;
    const uint inferred_alpha_bits = alpha_bits & ~declared_alpha_bits;
    const uint candidate_bits = alpha_bits | SUNSHINE_UI_CANDIDATE_HUDLESS;
    // V1: an alpha candidate is valid with at most 1% invalid pixels, accepted
    // or not; selective is some coverage below 90% of the frame.
    uint valid = 0u, selective = 0u;
    if (invalid.x * 100u <= pixels) valid |= SUNSHINE_UI_CANDIDATE_UI_ALPHA;
    if (invalid.y * 100u <= pixels) valid |= SUNSHINE_UI_CANDIDATE_UI_COLOR;
    if (layer.y * 100u <= pixels) valid |= SUNSHINE_UI_CANDIDATE_LAYER;
    if (invalid.z * 100u <= pixels) valid |= SUNSHINE_UI_CANDIDATE_BACKBUFFER;
    if (invalid.w * 100u <= pixels) valid |= SUNSHINE_UI_CANDIDATE_CURRENT;
    if (coverage.x && coverage.x * 10u < pixels * 9u) selective |= SUNSHINE_UI_CANDIDATE_UI_ALPHA;
    if (coverage.y && coverage.y * 10u < pixels * 9u) selective |= SUNSHINE_UI_CANDIDATE_UI_COLOR;
    if (layer.x && layer.x * 10u < pixels * 9u) selective |= SUNSHINE_UI_CANDIDATE_LAYER;
    if (coverage.z && coverage.z * 10u < pixels * 9u) selective |= SUNSHINE_UI_CANDIDATE_BACKBUFFER;
    if (coverage.w && coverage.w * 10u < pixels * 9u) selective |= SUNSHINE_UI_CANDIDATE_CURRENT;
    // V2: difference is changed-color support, not uniquely recovered
    // opacity. A partial change set needs broad unchanged scene evidence in
    // clean tiles as well as bounded total coverage. UI covering the whole
    // frame (menus, title screens) changes nearly every pixel while the
    // HUD-less image still shows a lit scene; only an exact pair qualifies
    // there, since a mismatched pair can also differ everywhere, and a black
    // HUD-less image is no scene. A lit pair without any changed pixel in
    // clean tiles, exact or not, is a valid empty set (no UI on screen), which
    // an accepted pair decides as an empty mask of its own (source 5).
    const bool partial_set = !difference.y && difference.x && difference.x * 4u < pixels &&
        difference.z * 100u >= pixels * 75u && matching_tiles >= 128u;
    const bool full_set = (offered & SUNSHINE_UI_CANDIDATE_EXACT) && !difference.y &&
        difference.x * 100u >= pixels * 98u && lit * 2u >= pixels;
    const bool empty_set = pixels && !difference.y && !difference.x && lit * 2u >= pixels && matching_tiles >= 128u;
    if (partial_set || full_set || empty_set) valid |= SUNSHINE_UI_CANDIDATE_HUDLESS;
    if (partial_set) selective |= SUNSHINE_UI_CANDIDATE_HUDLESS;
    valid &= offered & candidate_bits;
    // S1: among offered, accepted and valid candidates the first in draw
    // order decides at any coverage: nothing is no UI, everything a
    // full-screen menu (P1). An unaccepted or invalid candidate never blocks
    // another, except that an offered, accepted declared alpha (UIAlpha or the
    // UI color tag) keeps inferred alpha from deciding even while it is
    // invalid itself: that frame then has no alpha mask rather than a
    // scene-wide one.
    const bool block = (offered & accepted & declared_alpha_bits) != 0u;
    const uint eligible = offered & accepted & valid & (block ? ~inferred_alpha_bits : 0xffffffffu);
    uint source = 0u, covered = 0u, winner = 0u;
    if (eligible & SUNSHINE_UI_CANDIDATE_UI_ALPHA) { source = 1u; covered = coverage.x; winner = SUNSHINE_UI_CANDIDATE_UI_ALPHA; }
    else if (eligible & SUNSHINE_UI_CANDIDATE_UI_COLOR) { source = 2u; covered = coverage.y; winner = SUNSHINE_UI_CANDIDATE_UI_COLOR; }
    else if (eligible & SUNSHINE_UI_CANDIDATE_LAYER) {
        source = SUNSHINE_UI_SOURCE_LAYER; covered = layer.x; winner = SUNSHINE_UI_CANDIDATE_LAYER;
    }
    else if (eligible & SUNSHINE_UI_CANDIDATE_BACKBUFFER) { source = 3u; covered = coverage.z; winner = SUNSHINE_UI_CANDIDATE_BACKBUFFER; }
    else if (eligible & SUNSHINE_UI_CANDIDATE_CURRENT) { source = 4u; covered = coverage.w; winner = SUNSHINE_UI_CANDIDATE_CURRENT; }
    else if (eligible & SUNSHINE_UI_CANDIDATE_HUDLESS) {
        if (full_set) { source = 6u; covered = pixels; }
        else { source = 5u; covered = difference.x; }
    }
    // H1, a hidden scene (docs/reshade-sbs.md, hidden-scene evidence): the
    // informative full claims of this frame are (a) an alpha S1 may select
    // (eligible: accepted, valid and not kept out by the declared-alpha
    // block) that is opaque-full (alpha of at least 254/255 on 99% of
    // pixels), (b)
    // the valid, opaque-full offscreen UI layer, a target proven cleared to
    // transparent, accepted or not, (c) an exact full change set, accepted or
    // not, and (d) a pre-UI scene image: the HUD-less image changed on 90% of
    // pixels, or else an offered layer without coverage whose signature the
    // CPU's acceptance ledger proved the pre-UI scene image by its pixels
    // (SUNSHINE_UI_PER_FRAME_PRE_UI_PROVEN). An unaccepted UIAlpha, UI color
    // tag, Backbuffer or current alpha is never informative. A claim acts
    // unless the CPU refuted its signature, and (d) only while the CPU's
    // samples read the pre-UI image visible. While the CPU holds a hidden
    // verdict of D on the presented frame and the depth is this frame's, an
    // acting claim shows the frame flat (source 8), whatever S1 selected; a
    // winner already flat everywhere is relabelled 8 too, since UI pins at
    // weight 1 either way (P1). The evidence passes only measure.
    const uint s1_source = source;
    const uint opaque_needed = pixels * SUNSHINE_UI_SCENE_OPAQUE_PERCENT;
    uint claims = 0u;
    if (pixels) {
        if ((eligible & SUNSHINE_UI_CANDIDATE_UI_ALPHA) && opaque.x * 100u >= opaque_needed)
            claims |= SUNSHINE_UI_CANDIDATE_UI_ALPHA;
        if ((eligible & SUNSHINE_UI_CANDIDATE_UI_COLOR) && opaque.y * 100u >= opaque_needed)
            claims |= SUNSHINE_UI_CANDIDATE_UI_COLOR;
        if ((valid & SUNSHINE_UI_CANDIDATE_LAYER) && layer.z * 100u >= opaque_needed) claims |= SUNSHINE_UI_CANDIDATE_LAYER;
        if ((eligible & SUNSHINE_UI_CANDIDATE_BACKBUFFER) && opaque_backbuffer * 100u >= opaque_needed)
            claims |= SUNSHINE_UI_CANDIDATE_BACKBUFFER;
        if ((eligible & SUNSHINE_UI_CANDIDATE_CURRENT) && opaque_current * 100u >= opaque_needed)
            claims |= SUNSHINE_UI_CANDIDATE_CURRENT;
    }
    if (full_set && (valid & SUNSHINE_UI_CANDIDATE_HUDLESS)) claims |= SUNSHINE_UI_CANDIDATE_HUDLESS;
    const uint image = (offered & SUNSHINE_UI_CANDIDATE_HUDLESS) ? SUNSHINE_UI_PRE_UI_IMAGE_HUDLESS :
        (offered & SUNSHINE_UI_CANDIDATE_LAYER) ? SUNSHINE_UI_PRE_UI_IMAGE_LAYER : 0u;
    if (pixels && ((image == SUNSHINE_UI_PRE_UI_IMAGE_HUDLESS &&
            difference.x * 100u >= pixels * SUNSHINE_UI_SCENE_HUDLESS_CHANGED_PERCENT) ||
            (image == SUNSHINE_UI_PRE_UI_IMAGE_LAYER && !layer.x &&
             (Sunshine_UIDetectionFlags & SUNSHINE_UI_PER_FRAME_PRE_UI_PROVEN))))
        claims |= SUNSHINE_UI_CLAIM_PRE_UI;
    const uint refuted = (Sunshine_UIDetectionFlags >> SUNSHINE_UI_PER_FRAME_REFUTED_SHIFT) & candidate_bits;
    const uint acting = (claims & candidate_bits & ~refuted) |
        ((Sunshine_UIDetectionFlags & SUNSHINE_UI_PER_FRAME_PRE_UI_VISIBLE) ? (claims & SUNSHINE_UI_CLAIM_PRE_UI) : 0u);
    const bool h1 = acting && (Sunshine_UIDetectionFlags & SUNSHINE_UI_PER_FRAME_SCENE_HIDDEN) &&
        !(Sunshine_UIDetectionFlags & SUNSHINE_UI_PER_FRAME_DEPTH_NOT_CURRENT);
    if (h1) { source = 8u; covered = pixels; }
    const uint unaccepted = offered & ~accepted;
    // F1: without a source, the first reason that applies and the candidate
    // it refused, the first in draw order that matches it. An acting claim
    // that H1 did not apply refuses its first claimant, the pre-UI image
    // alone its image (HUD-less when offered, else the layer).
    uint none_reason = SUNSHINE_UI_NONE_OTHER, refused = 0u;
    if (acting) {
        none_reason = SUNSHINE_UI_NONE_GATE_NO_HOLD;
        refused = SunshineUIFirstCandidate(acting & candidate_bits);
        if (!refused) refused = (offered & SUNSHINE_UI_CANDIDATE_HUDLESS) ? SUNSHINE_UI_CANDIDATE_HUDLESS : SUNSHINE_UI_CANDIDATE_LAYER;
    } else if (block && (offered & accepted & valid & inferred_alpha_bits)) {
        none_reason = SUNSHINE_UI_NONE_PRESENTED_BLOCKED;
        refused = SunshineUIFirstCandidate(offered & accepted & valid & inferred_alpha_bits);
    } else if (offered & accepted & alpha_bits & ~valid) {
        none_reason = SUNSHINE_UI_NONE_TRUSTED_INVALID;
        refused = SunshineUIFirstCandidate(offered & accepted & alpha_bits & ~valid);
    } else if ((offered & SUNSHINE_UI_CANDIDATE_LAYER) && !(valid & SUNSHINE_UI_CANDIDATE_LAYER)) {
        none_reason = SUNSHINE_UI_NONE_LAYER_ASIDE;
        refused = SUNSHINE_UI_CANDIDATE_LAYER;
    } else if (unaccepted & valid & selective) {
        none_reason = SUNSHINE_UI_NONE_UNACCEPTED;
        refused = SunshineUIFirstCandidate(unaccepted & valid & selective);
    } else if (offered & SUNSHINE_UI_CANDIDATE_HUDLESS) {
        none_reason = SUNSHINE_UI_NONE_DIFFERENCE_FAILED;
        refused = SUNSHINE_UI_CANDIDATE_HUDLESS;
    } else if (unaccepted & valid & alpha_bits & ~selective) {
        none_reason = SUNSHINE_UI_NONE_AMBIGUOUS;
        refused = SunshineUIFirstCandidate(unaccepted & valid & alpha_bits & ~selective);
    } else if (!(offered & alpha_bits)) none_reason = SUNSHINE_UI_NONE_NO_CANDIDATE;
    if (source) refused = 0u;
    const uint frame_reason = source ? SUNSHINE_UI_FRAME_REASON_DECIDED : none_reason;
    // A2: a valid exact change set judges the offered, valid UIAlpha, UI
    // color tag, Backbuffer and current alpha; at least a tenth of a source's
    // strong pixels where the HUD-less image is lit and unchanged contradicts
    // it. Its counts exist on status samples only.
    const uint pair_bits = SUNSHINE_UI_CANDIDATE_HUDLESS | SUNSHINE_UI_CANDIDATE_EXACT;
    const bool exact_judge = (offered & pair_bits) == pair_bits && (valid & SUNSHINE_UI_CANDIDATE_HUDLESS);
    uint contradicted_bits = 0u;
    if (exact_judge) {
        if ((valid & SUNSHINE_UI_CANDIDATE_UI_ALPHA) && strong.x && contradicted.x * 10u >= strong.x)
            contradicted_bits |= SUNSHINE_UI_CANDIDATE_UI_ALPHA;
        if ((valid & SUNSHINE_UI_CANDIDATE_UI_COLOR) && strong.y && contradicted.y * 10u >= strong.y)
            contradicted_bits |= SUNSHINE_UI_CANDIDATE_UI_COLOR;
        if ((valid & SUNSHINE_UI_CANDIDATE_BACKBUFFER) && strong.z && contradicted.z * 10u >= strong.z)
            contradicted_bits |= SUNSHINE_UI_CANDIDATE_BACKBUFFER;
        if ((valid & SUNSHINE_UI_CANDIDATE_CURRENT) && strong.w && contradicted.w * 10u >= strong.w)
            contradicted_bits |= SUNSHINE_UI_CANDIDATE_CURRENT;
    }
    // T1: a real frame that decided no source while an accepted candidate is
    // missing or offered but invalid has no decision of its own. It applies
    // the previous real frame's own decision once, from the hold store, and
    // then has no mask; SUNSHINE_UI_PER_FRAME_HOLD_RESET means no previous
    // real decision exists in this chain. A HUD-less re-offer without a
    // decision of its own (SUNSHINE_UI_PER_FRAME_REOFFER) leaves the store as
    // it is, so it applies the held decision for as long as the snapshot is
    // re-offered. The store is read before it is written.
    const bool no_own = !source && ((Sunshine_UIDetectionFlags & SUNSHINE_UI_PER_FRAME_ACCEPTED_MISSING) ||
        (offered & accepted & candidate_bits & ~valid));
    const uint prior = (Sunshine_UIDetectionFlags & SUNSHINE_UI_PER_FRAME_HOLD_RESET) ? SUNSHINE_UI_HOLD_NONE :
        SunshineUIHoldStore[uint2(0u, 0u)];
    const uint stored_source = SunshineUIHoldStore[uint2(1u, 0u)], stored_covered = SunshineUIHoldStore[uint2(2u, 0u)];
    const bool reoffer = (Sunshine_UIDetectionFlags & SUNSHINE_UI_PER_FRAME_REOFFER) && prior != SUNSHINE_UI_HOLD_NONE;
    const bool reused = no_own && (prior == SUNSHINE_UI_HOLD_OWN || (reoffer && stored_source));
    const uint held = reused ? stored_source : source, held_covered = reused ? stored_covered : covered;
    if (!(no_own && reoffer)) {
        SunshineUIHoldStore[uint2(0u, 0u)] = no_own ? SUNSHINE_UI_HOLD_SPENT : SUNSHINE_UI_HOLD_OWN;
        SunshineUIHoldStore[uint2(1u, 0u)] = held;
        SunshineUIHoldStore[uint2(2u, 0u)] = held_covered;
    }
    // The applied decision: this frame's own, or the one T1 reused.
    const uint applied = held, applied_covered = held_covered;
    // Exact per-session counters (game3d_ui_counters.h): this detection frame,
    // its applied decision and, without one, the own decision's reason; the
    // own decision's judgments. Counting only; nothing on the GPU reads them.
    const bool alpha = (applied >= 1u && applied <= 4u) || applied == SUNSHINE_UI_SOURCE_LAYER;
    const bool full_alpha = alpha && applied_covered * 100u >= pixels * 99u;
    InterlockedAdd(SunshineUICountersStore[uint2(SUNSHINE_UI_COUNTER_DETECTION_FRAMES, 0u)], 1u);
    if (applied <= SUNSHINE_UI_SOURCE_LAYER)
        InterlockedAdd(SunshineUICountersStore[uint2(SUNSHINE_UI_COUNTER_DECIDED + applied, 0u)], 1u);
    if (!applied) InterlockedAdd(SunshineUICountersStore[uint2(SUNSHINE_UI_COUNTER_NONE + none_reason, 0u)], 1u);
    if (source == 5u && (offered & pair_bits) == SUNSHINE_UI_CANDIDATE_HUDLESS)
        InterlockedAdd(SunshineUICountersStore[uint2(SUNSHINE_UI_COUNTER_INEXACT_DIFFERENCE, 0u)], 1u);
    if (Sunshine_UIDetectionFlags & SUNSHINE_UI_PER_FRAME_DEPTH_NOT_CURRENT)
        InterlockedAdd(SunshineUICountersStore[uint2(SUNSHINE_UI_COUNTER_DEPTH_NOT_CURRENT, 0u)], 1u);
    // The S1 winner (not overridden by H1) is an accepted alpha the one-way
    // test contradicts.
    if (!h1 && (accepted & contradicted_bits & winner))
        InterlockedAdd(SunshineUICountersStore[uint2(SUNSHINE_UI_COUNTER_CONTRADICTED, 0u)], 1u);
    if (full_alpha)
        InterlockedAdd(SunshineUICountersStore[uint2(SUNSHINE_UI_COUNTER_FULL_ALPHA, 0u)], 1u);
    if (reused)
        InterlockedAdd(SunshineUICountersStore[uint2(SUNSHINE_UI_COUNTER_REUSED, 0u)], 1u);
    // Texel 0 is the applied decision the mask pass resolves.
    SunshineAlphaCoverageStore[uint2(0,0)] = uint4(applied, applied_covered, pixels, matching_tiles);
    // Diagnostic evidence for the CPU readback; nothing reads it on the GPU.
    SunshineAlphaCoverageStore[uint2(1,0)] = uint4(offered, difference.x, difference.z, difference.y);
    SunshineAlphaCoverageStore[uint2(2,0)] = coverage;
    SunshineAlphaCoverageStore[uint2(3,0)] = invalid;
    SunshineAlphaCoverageStore[uint2(4,0)] = uint4(lit, accepted, opaque);
    // No scene evidence unless the evidence passes run after this one.
    SunshineAlphaCoverageStore[uint2(5,0)] = 0u;
    SunshineAlphaCoverageStore[uint2(6,0)] = 0u;
    SunshineAlphaCoverageStore[uint2(7,0)] = uint4(layer, valid);
    // The one-way counts of Backbuffer and current alpha; .x (the layer's
    // before revision 10) is a reserved zero.
    SunshineAlphaCoverageStore[uint2(8,0)] = uint4(0u, strong.zw, refused);
    // The mask pass keeps the previous real frame's mask when reused.
    SunshineAlphaCoverageStore[uint2(9,0)] = uint4(0u, contradicted.zw, frame_reason | (reused ? SUNSHINE_UI_FRAME_REASON_REUSED : 0u));
    // H1: the opaque Backbuffer and current counts, the raw claims before
    // refutation, and the S1 winner with the applied bit.
    SunshineAlphaCoverageStore[uint2(10,0)] = uint4(opaque_backbuffer, opaque_current, claims,
        s1_source | (h1 ? SUNSHINE_UI_H1_APPLIED : 0u));
    // H1 (d): the pre-UI pixel counts the CPU's acceptance ledger proves the
    // layer from; nothing on the GPU reads them.
    SunshineAlphaCoverageStore[uint2(11,0)] = pre_ui_pixels;
    // Texels 12-15 are reserved zeros (removed H2, S3, fix 3 and fix 4 words).
    [unroll] for (uint reserved = 12u; reserved < 16u; ++reserved) SunshineAlphaCoverageStore[uint2(reserved, 0u)] = 0u;
    // A2 of the declared alphas: strong UIAlpha and UI color tag, then their
    // contradicted pixels.
    SunshineAlphaCoverageStore[uint2(16,0)] = uint4(strong.xy, contradicted.xy);
}
[numthreads(8, 8, 1)]
void SunshineUIDetectionMaskCS(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= BUFFER_WIDTH || id.y >= BUFFER_HEIGHT) return;
    // T1: a frame that reused the previous real frame's decision keeps the
    // mask that decision resolved, which no later pass overwrites.
    if (SunshineUIDetectionSampler.Load(int3(9,0,0)).w & SUNSHINE_UI_FRAME_REASON_REUSED) return;
    uint source = SunshineUIDetectionSampler.Load(int3(0,0,0)).x;
    float mask = 0.0;
    int3 at = int3(id.xy, 0);
    // Full-frame UI (a full change set or H1): the whole frame pins.
    if (source == 6u || source == 8u) mask = 1.0;
    // Otherwise load only the selected candidate's raw mask (the same channel
    // the tiles pass reads).
    else if (source == 1u) mask = SunshineUIDedicatedAlpha.Load(at).r;
    else if (source == 2u) mask = SunshineUIColorAlpha.Load(at).a;
    else if (source == 3u) mask = SunshineUIBackbufferAlpha.Load(at).a;
    else if (source == 4u) mask = SunshineSourceSampler.Load(at).a;
    else if (source == SUNSHINE_UI_SOURCE_LAYER) mask = SunshineUILayer.Load(at).a;
    else if (source == 5u) {
        bool finite;
        mask = SunshineHUDlessDifference(id.xy, finite) > Sunshine_UIDifferenceThreshold && finite ? 1.0 : 0.0;
    }
    SunshineHostCandidateStore[id.xy] = mask;
}

float2 SunshineCameraRawDepthRange()
{
    // Zero-initialized storage preserves [0,1] for older integrations without
    // specializing this writable uniform in ReShade Performance Mode.
    return Sunshine_CameraCoordinateBasis == 1 || all(Sunshine_CameraRawDepthRange == 0.0) ?
        float2(0.0, 1.0) : Sunshine_CameraRawDepthRange;
}

bool SunshineCameraActive()
{
    if (!Sunshine_CameraDepthReady)
        return false;
    if (!SunshineCameraFinite(Sunshine_DisparityLimitUv) ||
        Sunshine_DisparityLimitUv <= 0.0 || Sunshine_DisparityLimitUv > 0.04)
        return false;
    float4 rect = Sunshine_CameraDepthRect;
    if (!SunshineCameraFinite(rect.x) || !SunshineCameraFinite(rect.y) ||
        !SunshineCameraFinite(rect.z) || !SunshineCameraFinite(rect.w) ||
        any(rect.xy < 0.0) || any(rect.zw <= 0.0) || any(rect.xy + rect.zw > 1.000001))
        return false;
    float A = Sunshine_CameraProjection.x, inverseB = Sunshine_CameraProjection.y;
    if (Sunshine_CameraCoordinateBasis != 0 && Sunshine_CameraCoordinateBasis != 1 && Sunshine_CameraCoordinateBasis != 2)
        return false;
    if (Sunshine_CameraCoordinateBasis == 1)
    {
        if (!((A == 0.0 && inverseB == 1.0) || (A == 1.0 && inverseB == -1.0)))
            return false;
    }
    float K = Sunshine_CameraDepthScale;
    float referenceZPD = Sunshine_CameraConvergence.x, zeroInverseDistance = Sunshine_CameraConvergence.y;
    if (!SunshineCameraFinite(A) || !SunshineCameraFinite(inverseB) ||
        !SunshineCameraFinite(K) || !SunshineCameraFinite(referenceZPD) ||
        !SunshineCameraFinite(zeroInverseDistance))
        return false;
    if (inverseB == 0.0 || K <= 0.0 || referenceZPD <= 0.0 || referenceZPD > 1.0 || zeroInverseDistance < 0.0)
        return false;
    if (Sunshine_CameraCoordinateBasis == 2)
    {
        // Positive linear distances have no camera near endpoint. Validate
        // each reciprocal and bound displacement before its multiplication.
        float gain = referenceZPD * K;
        return SunshineCameraFinite(gain) && SunshineCameraFinite(gain * zeroInverseDistance);
    }
    // Resource precision may store the hardware-depth [0,1] interval in a
    // different raw range. Decode those actual endpoints without clipping.
    float2 rawRange = SunshineCameraRawDepthRange();
    if (!SunshineCameraFinite(rawRange.x) || !SunshineCameraFinite(rawRange.y) || rawRange.x >= rawRange.y)
        return false;
    float2 endpointInverseDistance = (rawRange - A) * inverseB;
    float largestInverseDistance = max(endpointInverseDistance.x, endpointInverseDistance.y);
    if (!SunshineCameraFinite(endpointInverseDistance.x) || !SunshineCameraFinite(endpointInverseDistance.y) ||
        !all(endpointInverseDistance >= 0.0) || largestInverseDistance <= 0.0)
        return false;
    // Geometry is now R32 source-U, not reciprocal FP16 intermediates.
    // Match CPU admission to the actual displacement expression over the raw
    // interval. Keep overflow rejection, without obsolete half-float caps.
    float gain = referenceZPD * K;
    return SunshineCameraFinite(gain) && SunshineCameraFinite(gain * zeroInverseDistance) &&
        SunshineCameraFinite(gain * (zeroInverseDistance - largestInverseDistance));
}

float2 SunshineCameraDepthCoordinates(float2 coordinate, float2 allocationSize)
{
    float2 halfTexel = 0.5 / allocationSize;
    float2 first = Sunshine_CameraDepthRect.xy + halfTexel;
    float2 last = Sunshine_CameraDepthRect.xy + Sunshine_CameraDepthRect.zw - halfTexel;
    // Point sampling must not cross into neighboring allocation padding through
    // floating-point rounding at the last active pixel.
    return clamp(Sunshine_CameraDepthRect.xy + coordinate * Sunshine_CameraDepthRect.zw + Sunshine_DepthJitter, first, last);
}

float SunshineAutomaticStrengthBlend()
{
    // A missing or invalid source-owned transition must never enable stereo.
    return SunshineCameraFinite(Sunshine_CameraStrengthBlend)
        ? saturate(Sunshine_CameraStrengthBlend) : 0.0;
}

bool SunshineAutomaticMono()
{
    if (!_SUNSHINE_HOST_WARP_SUPPORTED || !Sunshine_DepthReady || !SunshineCameraActive())
        return true;
    // A ready depth diagnostic remains visible when stereo strength is zero.
    if (Depth_Map_View == 2)
        return false;
    return !SunshineCameraFinite(Depth_Adjustment) || Depth_Adjustment <= 0.0 ||
        SunshineAutomaticStrengthBlend() <= 0.0;
}
bool SunshineCameraInverseDistance(float rawDepth, out float inverseDistance)
{
    // Geometry passes a precise output. Keep visualization non-precise here:
    // otherwise FXC propagates it through sampled UVs into the eye warp.
    float affine = (rawDepth - Sunshine_CameraProjection.x) * Sunshine_CameraProjection.y;
    inverseDistance = affine;
    if (Sunshine_CameraCoordinateBasis != 2)
        return SunshineCameraFinite(inverseDistance) && inverseDistance >= 0.0;
    if (!SunshineCameraFinite(rawDepth) || !SunshineCameraFinite(affine) || affine <= 0.0 || asuint(affine) < 0x00800000u)
        return false;
    inverseDistance = 1.0 / affine;
    return SunshineCameraFinite(inverseDistance) && inverseDistance > 0.0 && asuint(inverseDistance) >= 0x00800000u;
}

float SunshineCameraDepth(float rawDepth)
{
    // Device, relative raw and reciprocal linear bases share the same
    // rational visualization; geometry uses decoded inverse depth directly.
    float inverseDistance = (rawDepth - Sunshine_CameraProjection.x) * Sunshine_CameraProjection.y;
    if (Sunshine_CameraCoordinateBasis == 2)
        if (!SunshineCameraInverseDistance(rawDepth, inverseDistance)) return 1.0;
    return rcp(1.0 + Sunshine_CameraDepthScale * inverseDistance);
}

// Hidden-scene evidence (docs/reshade-sbs.md, hidden-scene evidence). It
// measures whether the presented color, and the pre-UI scene image (a
// HUD-less image when one is offered, else the offscreen UI layer's colour),
// show the consumed depth's edges; it never decides a mask. The renderer
// dispatches these passes after detection on sample frames only, the CPU
// holds the verdict (game3d_scene_guard.h), and SunshineUIDetectionReduceCS
// applies H1 under a held verdict while an informative claim acts.
// inspect_game3d_dump.py holds the same statistic as a CPU oracle.
//
// Output pixel x lies in cell column floor(x * 256 / width), row y in cell row
// floor(y * 144 / height). Each cell sums its pixels' perceptual luma and
// strength-1 parallax in fixed point, so its means and every comparison below
// are exact integers whatever order the pixels were added in. Three passes:
// cell sums into their own texture, per-block comparisons into the statistics
// rows below the detection tiles (from SUNSHINE_UI_SCENE_PARTIAL_ROW), and
// their sum into decision texels 5 and 6.
// A depth edge's cell-mean parallax step in fixed point: SUNSHINE_UI_SCENE_EDGE_PX
// pixels per 2160 output rows, rounded up, so the integer test is exact and
// cannot overflow.
static const uint SunshineSceneEdgeStep =
    (SUNSHINE_UI_SCENE_EDGE_PX * SUNSHINE_UI_SCENE_PARALLAX_SCALE * BUFFER_HEIGHT + 2159u) / 2160u;
bool SunshineSceneDepthActive()
{
    // A frame smaller than the grid would leave cells without a pixel. Above
    // the renderer's 3840 x 3840 detection bound a cell's fixed-point sums
    // could overflow 32 bits.
    return BUFFER_WIDTH >= SUNSHINE_UI_SCENE_CELLS_X && BUFFER_HEIGHT >= SUNSHINE_UI_SCENE_CELLS_Y &&
        BUFFER_WIDTH <= 3840 && BUFFER_HEIGHT <= 3840 && Sunshine_DepthReady && SunshineCameraActive();
}
// Perceptual code luma: PQ code for HDR10 and sRGB code for SDR as copied,
// PQ of luminance for scRGB. A non-finite component counts as zero.
uint SunshineSceneLuma(float3 rgb)
{
    rgb = isfinite(rgb) ? rgb : 0.0;
#if BUFFER_COLOR_SPACE == 3
    float luma = dot(rgb, float3(0.2627, 0.6780, 0.0593));
#elif BUFFER_COLOR_SPACE == 2
    const float m1 = 2610.0 / 16384.0, m2 = 2523.0 / 32.0;
    const float c1 = 3424.0 / 4096.0, c2 = 2413.0 / 128.0, c3 = 2392.0 / 128.0;
    float power = pow(clamp(dot(rgb, float3(0.2126, 0.7152, 0.0722)) * 80.0, 0.0, 10000.0) / 10000.0, m1);
    float luma = pow((c1 + c2 * power) / (1.0 + c3 * power), m2);
#else
    float luma = dot(rgb, float3(0.2126, 0.7152, 0.0722));
#endif
    return uint(saturate(luma) * SUNSHINE_UI_SCENE_LUMA_SCALE + 0.5);
}
// The parallax SunshineHostCandidateCS decodes at this pixel for strength and
// blend one, in output pixels, clamped at the disparity limit; zero where depth
// does not decode. Stereo strength never changes the evidence. The depth texel
// is floor(uv * size) rather than the sampler's, whose subtexel rounding is
// the GPU's own, so the CPU oracle reproduces it exactly.
int SunshineSceneParallax(uint2 xy)
{
    float2 coordinate = (float2(xy) + 0.5) / float2(BUFFER_WIDTH, BUFFER_HEIGHT);
    float2 size = SunshineDepthAllocationSize();
    float2 uv = SunshineCameraDepthCoordinates(coordinate, size);
    float raw = DepthBuffer.Load(int3(clamp(int2(floor(uv * size)), 0, int2(size) - 1), 0));
    float q = (raw - Sunshine_CameraProjection.x) * Sunshine_CameraProjection.y;
    bool validDepth = true;
    if (Sunshine_CameraCoordinateBasis == 2) validDepth = SunshineCameraInverseDistance(raw, q);
    float displacement = Sunshine_CameraConvergence.x * Sunshine_CameraDepthScale * (Sunshine_CameraConvergence.y - q);
    if (Sunshine_CameraCoordinateBasis == 2)
    {
        float gain = Sunshine_CameraConvergence.x * Sunshine_CameraDepthScale;
        displacement = gain > 0.0 ? clamp(Sunshine_CameraConvergence.y - q, -1.5 / gain, 2.5 / gain) * gain : 0.0;
    }
    if (!validDepth || !SunshineCameraFinite(displacement)) return 0;
    float limit = Sunshine_DisparityLimitUv * BUFFER_WIDTH;
    float parallax = clamp(-clamp(displacement, -1.5, 2.5) * (BUFFER_HEIGHT / 2160.0 * 100.0), -limit, limit);
    return int(floor(parallax * SUNSHINE_UI_SCENE_PARALLAX_SCALE + 0.5));
}
// The pre-UI scene image the evidence measures beside the presented colour
// (H1 d, ui_selection::pre_ui_image_of): the HUD-less image (t14) when one is
// offered, else the offscreen UI layer's colour (t7), else none.
uint SunshinePreUIImage()
{
    return (Sunshine_UICandidates & SUNSHINE_UI_CANDIDATE_HUDLESS) ? SUNSHINE_UI_PRE_UI_IMAGE_HUDLESS :
        (Sunshine_UICandidates & SUNSHINE_UI_CANDIDATE_LAYER) ? SUNSHINE_UI_PRE_UI_IMAGE_LAYER : 0u;
}
// Cell sums {presented luma, pre-UI image luma, parallax, pixels} in fixed
// point, one texel per cell of the scene cell texture (u6 here, t10 when
// compared). A group sums sixteen cells of one cell row with sixteen lanes
// per cell: lane i sums column i of its cell (and every sixteenth column
// after it), so neighbouring lanes read neighbouring pixels.
groupshared uint3 SunshineSceneCellSums[16][16];
[numthreads(16, 16, 1)]
void SunshineSceneCellsCS(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    const uint2 cells = uint2(SUNSHINE_UI_SCENE_CELLS_X, SUNSHINE_UI_SCENE_CELLS_Y);
    const uint2 size = uint2(BUFFER_WIDTH, BUFFER_HEIGHT);
    const uint2 cell = uint2(group.x * 16u + thread.y, group.y);
    // The first pixel of this cell and of the next: ceil(cell * size / cells).
    const uint2 first = (cell * size + cells - 1u) / cells, last = ((cell + 1u) * size + cells - 1u) / cells;
    const bool active = SunshineSceneDepthActive();
    const uint image = SunshinePreUIImage();
    uint3 sums = 0u; // Parallax adds as two's complement bits.
    if (active) {
        [loop] for (uint y = first.y; y < last.y; ++y)
        [loop] for (uint x = first.x + thread.x; x < last.x; x += 16u) {
            sums.x += SunshineSceneLuma(SunshinePresentedColor.Load(int3(x, y, 0)).rgb);
            if (image == SUNSHINE_UI_PRE_UI_IMAGE_HUDLESS) sums.y += SunshineSceneLuma(SunshineHUDless.Load(int3(x, y, 0)).rgb);
            else if (image == SUNSHINE_UI_PRE_UI_IMAGE_LAYER) sums.y += SunshineSceneLuma(SunshineUILayer.Load(int3(x, y, 0)).rgb);
            sums.z += asuint(SunshineSceneParallax(uint2(x, y)));
        }
    }
    SunshineSceneCellSums[thread.y][thread.x] = sums;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint step = 8u; step; step >>= 1u) {
        if (thread.x < step) SunshineSceneCellSums[thread.y][thread.x] += SunshineSceneCellSums[thread.y][thread.x + step];
        GroupMemoryBarrierWithGroupSync();
    }
    const uint2 extent = last - first;
    if (active && !thread.x) SunshineAlphaCoverageStore[cell] = uint4(SunshineSceneCellSums[thread.y][0], extent.x * extent.y);
}
// Cell means, rounded half away from zero: luma of both images (presented,
// pre-UI), parallax.
void SunshineSceneCellMeans(uint2 cell, out uint2 luma, out int parallax)
{
    uint4 sums = SunshineUIDetectionSampler.Load(int3(cell, 0));
    uint count = max(sums.w, 1u), half_count = sums.w / 2u;
    luma = (sums.xy + half_count) / count;
    int sum = asint(sums.z);
    uint magnitude = (uint(abs(sum)) + half_count) / count;
    parallax = sum < 0 ? -int(magnitude) : int(magnitude);
}
// The larger step of the cell means to the right and to the lower cell, per
// image (activity) and of parallax; the last column and row lack one of them.
void SunshineSceneSteps(uint2 cell, out uint2 activity, out uint depthStep)
{
    uint2 luma, next;
    int parallax, nextParallax;
    SunshineSceneCellMeans(cell, luma, parallax);
    activity = 0u; depthStep = 0u;
    if (cell.x + 1u < SUNSHINE_UI_SCENE_CELLS_X) {
        SunshineSceneCellMeans(cell + uint2(1, 0), next, nextParallax);
        activity = max(activity, max(luma, next) - min(luma, next));
        depthStep = max(depthStep, uint(abs(parallax - nextParallax)));
    }
    if (cell.y + 1u < SUNSHINE_UI_SCENE_CELLS_Y) {
        SunshineSceneCellMeans(cell + uint2(0, 1), next, nextParallax);
        activity = max(activity, max(luma, next) - min(luma, next));
        depthStep = max(depthStep, uint(abs(parallax - nextParallax)));
    }
}
// Null cells (columns, rows), wrapped: beyond the one-cell stencil, not
// collinear, and close enough to see the same local texture.
static const int2 SunshineSceneNulls[4] = {int2(5, 3), int2(-7, 4), int2(8, -2), int2(-4, -6)};
// Edge cells, wins - losses of each image, and the presented image's decided
// (untied) comparisons.
groupshared int SunshineSceneTotals[4];
// One thread per cell, one group per 16x16 cells: at each edge cell, each
// image's activity against its activity at the null cells. The group's sums
// {n, wins - losses presented, pre-UI image, presented decided} go to statistics
// texel (group.x, SUNSHINE_UI_SCENE_PARTIAL_ROW + group.y).
[numthreads(16, 16, 1)]
void SunshineSceneCompareCS(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    const uint lane = thread.y * 16u + thread.x;
    const bool active = SunshineSceneDepthActive();
    if (lane < 4u) SunshineSceneTotals[lane] = 0;
    GroupMemoryBarrierWithGroupSync();
    if (active && all(id.xy < uint2(SUNSHINE_UI_SCENE_CELLS_X, SUNSHINE_UI_SCENE_CELLS_Y))) {
        uint2 activity;
        uint depthStep;
        SunshineSceneSteps(id.xy, activity, depthStep);
        if (depthStep >= SunshineSceneEdgeStep) {
            int2 balance = 0;
            int decided = 0;
            [unroll] for (uint k = 0u; k < 4u; ++k) {
                uint2 wrapped = uint2(int2(id.xy) + int2(SUNSHINE_UI_SCENE_CELLS_X, SUNSHINE_UI_SCENE_CELLS_Y) + SunshineSceneNulls[k]) %
                    uint2(SUNSHINE_UI_SCENE_CELLS_X, SUNSHINE_UI_SCENE_CELLS_Y);
                uint2 other;
                uint unused;
                SunshineSceneSteps(wrapped, other, unused);
                balance += int2(activity > other) - int2(activity < other);
                decided += activity.x != other.x ? 1 : 0;
            }
            InterlockedAdd(SunshineSceneTotals[0], 1);
            InterlockedAdd(SunshineSceneTotals[1], balance.x);
            InterlockedAdd(SunshineSceneTotals[2], balance.y);
            InterlockedAdd(SunshineSceneTotals[3], decided);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (active && !lane) {
        SunshineAlphaCoverageStore[uint2(group.x, SUNSHINE_UI_SCENE_PARTIAL_ROW + group.y)] =
            uint4(asuint(SunshineSceneTotals[0]), asuint(SunshineSceneTotals[1]), asuint(SunshineSceneTotals[2]),
                asuint(SunshineSceneTotals[3]));
    }
}
// Sums the compare groups' partial sums: D = (wins - losses) / (4 n); ties
// count 0, so a black or flat image reads as hidden. Writes decision texels 5
// (presented, with its verdict and decided comparisons) and 6 (the pre-UI
// scene image, with which image it is, when one is offered).
[numthreads(16, 16, 1)]
void SunshineSceneEvidenceCS(uint3 thread : SV_GroupThreadID)
{
    const uint lane = thread.y * 16u + thread.x;
    const bool active = SunshineSceneDepthActive();
    if (lane < 4u) SunshineSceneTotals[lane] = 0;
    GroupMemoryBarrierWithGroupSync();
    if (active && thread.y < SUNSHINE_UI_SCENE_CELLS_Y / 16u) {
        int4 partial = asint(SunshineUIDetectionSampler.Load(int3(thread.x, SUNSHINE_UI_SCENE_PARTIAL_ROW + thread.y, 0)));
        InterlockedAdd(SunshineSceneTotals[0], partial.x);
        InterlockedAdd(SunshineSceneTotals[1], partial.y);
        InterlockedAdd(SunshineSceneTotals[2], partial.z);
        InterlockedAdd(SunshineSceneTotals[3], partial.w);
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane) return;
    const int n = SunshineSceneTotals[0], pairs = 4 * n;
    const int2 balance = int2(SunshineSceneTotals[1], SunshineSceneTotals[2]);
    // Reused or generated depth is not this frame's: no evidence.
    const bool valid = active && n >= SUNSHINE_UI_SCENE_MIN_EDGES &&
        !(Sunshine_UIDetectionFlags & SUNSHINE_UI_PER_FRAME_DEPTH_NOT_CURRENT);
    const float2 d = n ? float2(balance) / float(pairs) : 0.0;
    // Verdict 0 none, 1 hidden, 2 ambiguous, 3 visible.
    const uint verdict = !valid ? 0u : balance.x * 100 < SUNSHINE_UI_SCENE_HIDDEN_PERCENT * pairs ? 1u :
        balance.x * 100 >= SUNSHINE_UI_SCENE_VISIBLE_PERCENT * pairs ? 3u : 2u;
    // State bit 0 valid, bit 1 ran, bits 2-3 the presented verdict; w how many
    // of the presented image's comparisons were decided rather than tied.
    SunshineAlphaCoverageStore[uint2(5, 0)] =
        uint4(uint(n), asuint(d.x), uint(valid) | 2u | verdict << 2, uint(SunshineSceneTotals[3]));
    const uint image = SunshinePreUIImage();
    SunshineAlphaCoverageStore[uint2(6, 0)] = image ? uint4(uint(n), asuint(d.y), uint(valid) | 2u, image) : 0u;
}

#if BUFFER_COLOR_SPACE == 3
float3 SunshineDecodePQ(float3 code)
{
    // ST.2084 absolute luminance in Rec.2020 -> signed linear Rec.709/scRGB.
    // The 125 factor maps 10000 nits to scRGB's 80-nit reference unit.
    const float m1 = 2610.0 / 16384.0, m2 = 2523.0 / 32.0;
    const float c1 = 3424.0 / 4096.0, c2 = 2413.0 / 128.0, c3 = 2392.0 / 128.0;
    float3 power = pow(saturate(code), 1.0 / m2);
    float3 linear_2020 = pow(max(power - c1, 0.0) /
        max(c2 - c3 * power, 1.0e-6), 1.0 / m1);
    const float3x3 rec2020_to_rec709 = float3x3(
        1.6604910021, -0.5876411388, -0.0728498633,
        -0.1245504745, 1.1328998971, -0.0083494226,
        -0.0181507634, -0.1005788980, 1.1187296614);
    return mul(rec2020_to_rec709, linear_2020) * 125.0;
}

// The texel coordinate of a normalized coordinate, in 1/256 texels, as the
// sampler forms it for bilinear filtering: the coordinate truncated to 21
// fractional bits, then rounded to 1/256 texel (measured on NVIDIA; an exact
// reproduction of that hardware's weights). size * q / 8192 in integers.
int SunshineFilterCoordinate(float coordinate, uint size)
{
    uint q = uint(floor(saturate(coordinate) * 2097152.0));
    return int((q >> 13u) * size + (((q & 8191u) * size + 4096u) >> 13u)) - 128;
}

// The PQ source decoded to scRGB at uv as the former linearization pass and a
// linear-filter sample of its FP16 texture gave it: each tap decoded and held
// in FP16, then interpolated with the sampler's weights. The pack reads rows
// at texel centres, where the vertical weight is zero, so only the
// horizontal tap pair is read.
float3 SunshineSourceLinear(float2 uv)
{
    int x = SunshineFilterCoordinate(uv.x, BUFFER_WIDTH);
    int row = clamp(SunshineFilterCoordinate(uv.y, BUFFER_HEIGHT) >> 8, 0, BUFFER_HEIGHT - 1);
    int x0 = x >> 8;
    float weight = float(x & 255) / 256.0;
    float3 left = f16tof32(f32tof16(SunshineDecodePQ(
        SunshineSourceSampler.Load(int3(clamp(x0, 0, BUFFER_WIDTH - 1), row, 0)).rgb)));
    float3 right = f16tof32(f32tof16(SunshineDecodePQ(
        SunshineSourceSampler.Load(int3(clamp(x0 + 1, 0, BUFFER_WIDTH - 1), row, 0)).rgb)));
    return lerp(left, right, weight);
}

// The host's scRGBTo2100PQ (src_assets/.../include/common.hlsl: Rec709toRec2020,
// 80 nits per scRGB unit, NitsToPQ), in the same operations and order.
float3 SunshineHostPQ(float3 rgb)
{
    static const float3x3 rec709_to_rec2020 = {
        0.627402, 0.329292, 0.043306,
        0.069095, 0.919544, 0.011360,
        0.016394, 0.088028, 0.895578
    };
    rgb = mul(rec709_to_rec2020, rgb);
    rgb *= 80;
    static const float m1 = 2610.0 / 4096.0 / 4;
    static const float m2 = 2523.0 / 4096.0 * 128;
    static const float c1 = 3424.0 / 4096.0;
    static const float c2 = 2413.0 / 4096.0 * 32;
    static const float c3 = 2392.0 / 4096.0 * 32;
    float3 Lp = pow(saturate(rgb / 10000.0), m1);
    return pow((c1 + c2 * Lp) / (1 + c3 * Lp), m2);
}

#endif
float4 SunshineNativeSource(float2 uv)
{
    float4 color = float4(SunshineSourceSampler.SampleLevel(SunshinePointClamp, (float4(uv, 0, 0)).xy, (float4(uv, 0, 0)).w).rgb, 1.0);
    // This helper is used only for the native mono endpoint. Rendered PQ color
    // was decoded before interpolation; native scRGB and SDR need no transform.
#if BUFFER_COLOR_SPACE == 3
    color.rgb = SunshineDecodePQ(color.rgb);
#endif
    return color;
}

#if _SUNSHINE_HOST_WARP_SUPPORTED
static const float SunshineHostContainer = 0.04;
static const float SunshineHostQScale = 1073741824.0;
static const int SunshineHostContainerQ = 42949672;

float SunshineBoundFinalParallax(float value)
{
    // Reapply the same uniform interval after fixed-point conditioning so its
    // outward rounding cannot exceed the actual per-frame display budget.
    float limit = SunshineCameraFinite(Sunshine_DisparityLimitUv) && SunshineCameraFinite(Depth_Adjustment) ?
        clamp(Sunshine_DisparityLimitUv, 0.0, SunshineHostContainer) *
        clamp(Depth_Adjustment, 0.0, 100.0) * 0.01 * SunshineAutomaticStrengthBlend() : 0.0;
    return clamp(value, -limit, limit);
}

bool SunshineHostWarpActive()
{
    return Sunshine_DepthReady && SunshineCameraActive() &&
        SunshineCameraFinite(Depth_Adjustment) && Depth_Adjustment > 0 && SunshineAutomaticStrengthBlend() > 0;
}

bool SunshineSourceUI(uint x, uint y) {
    float alpha = SunshineSelectedUIAlpha(uint2(x, y));
    return SunshineCameraFinite(alpha) && alpha > 0.0;
}

// How firmly a texel pins: zero exactly where SunshineSourceUI is false, one
// from alpha 1/SUNSHINE_UI_SOFT_PIN_GAIN up, in proportion below.
float SunshineUIPinWeight(uint x, uint y) {
    float alpha = SunshineSelectedUIAlpha(uint2(x, y));
    return SunshineCameraFinite(alpha) && alpha > 0.0 ? saturate(alpha * SUNSHINE_UI_SOFT_PIN_GAIN) : 0.0;
}

bool SunshineUINearestActive() {
    return Sunshine_SourceAlphaUI != 0u && Sunshine_UIPlaneMode == 2u && SunshineHostWarpActive() &&
        SunshineCameraFinite(Sunshine_UIPlaneInverseDepth) && Sunshine_UIPlaneInverseDepth >= 0.0;
}

// Full source-pixel coverage, including soft positive alpha. Depth coordinates
// and decoding match the scene candidate, before its displacement clamp. Every
// tile and the final scalar are overwritten on each enabled render; no history,
// CPU readback or source-plane smoothing participates in the covered maximum.
groupshared float SunshineUINearestScratch[256];
[numthreads(16, 16, 1)]
void SunshineUINearestTilesCS(uint3 id : SV_DispatchThreadID,
    uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID) {
    uint lane = thread_id.y * 16u + thread_id.x;
    float nearest = 0.0;
    if (SunshineUINearestActive() && id.x < BUFFER_WIDTH && id.y < BUFFER_HEIGHT && SunshineSourceUI(id.x, id.y)) {
        float2 coordinate = (float2(id.xy) + 0.5) / float2(BUFFER_WIDTH, BUFFER_HEIGHT);
        float2 uv = SunshineCameraDepthCoordinates(coordinate, SunshineDepthAllocationSize());
        float raw = DepthBuffer.SampleLevel(SunshinePointBorder, uv, 0).x;
        precise float q = (raw - Sunshine_CameraProjection.x) * Sunshine_CameraProjection.y;
        bool validDepth = true;
        if (Sunshine_CameraCoordinateBasis == 2) validDepth = SunshineCameraInverseDistance(raw, q);
        if (validDepth && SunshineCameraFinite(q) && q >= 0.0) nearest = q;
    }
    SunshineUINearestScratch[lane] = nearest;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint step = 128u; step != 0u; step >>= 1u) {
        if (lane < step) SunshineUINearestScratch[lane] = max(SunshineUINearestScratch[lane], SunshineUINearestScratch[lane + step]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) SunshineUIPlaneTilesStore[int2(group_id.xy)] = SunshineUINearestScratch[0];
}

[numthreads(256, 1, 1)]
void SunshineUINearestReduceCS(uint3 thread_id : SV_GroupThreadID) {
    uint lane = thread_id.x;
    const uint tiles_x = (BUFFER_WIDTH + 15u) / 16u;
    const uint tiles_y = (BUFFER_HEIGHT + 15u) / 16u;
    float nearest = 0.0;
    if (SunshineUINearestActive()) {
        [loop]
        for (uint index = lane; index < tiles_x * tiles_y; index += 256u)
            nearest = max(nearest, SunshineUIPlaneTilesSampler.Load(int3(int2(index % tiles_x, index / tiles_x), 0)));
    }
    SunshineUINearestScratch[lane] = nearest;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint step = 128u; step != 0u; step >>= 1u) {
        if (lane < step) SunshineUINearestScratch[lane] = max(SunshineUINearestScratch[lane], SunshineUINearestScratch[lane + step]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u)
        SunshineUIPlaneResolvedStore[int2(0, 0)] = SunshineUINearestActive() ?
            max(Sunshine_UIPlaneInverseDepth, SunshineUINearestScratch[0]) : 0.0;
}

[numthreads(8, 8, 1)]
void SunshineHostCandidateCS(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= BUFFER_WIDTH || id.y >= BUFFER_HEIGHT) return;
    float parallax = 0.0;
    float displayLimit = 0.0;
    if (SunshineHostWarpActive())
    {
        float2 coordinate = (float2(id.xy) + 0.5) / float2(BUFFER_WIDTH, BUFFER_HEIGHT);
        float2 uv = SunshineCameraDepthCoordinates(coordinate, SunshineDepthAllocationSize());
        float raw = DepthBuffer.SampleLevel(SunshinePointBorder, (float4(uv, 0, 0)).xy, (float4(uv, 0, 0)).w).x;
        precise float q = (raw - Sunshine_CameraProjection.x) * Sunshine_CameraProjection.y;
        bool validDepth = true;
        if (Sunshine_CameraCoordinateBasis == 2) validDepth = SunshineCameraInverseDistance(raw, q);
        precise float strength = clamp(Depth_Adjustment, 0.0, 100.0) * 0.01 * SunshineAutomaticStrengthBlend();
        displayLimit = Sunshine_DisparityLimitUv * strength;
        precise float displacement = Sunshine_CameraConvergence.x * Sunshine_CameraDepthScale *
            (Sunshine_CameraConvergence.y - q) * strength;
        if (Sunshine_CameraCoordinateBasis == 2)
        {
            float gain = Sunshine_CameraConvergence.x * Sunshine_CameraDepthScale * strength;
            float delta = Sunshine_CameraConvergence.y - q;
            // Bound the product using the existing field limits; do not clip
            // decoded q or overflow before the final display-parallax clamp.
            displacement = gain > 0.0 ? clamp(delta, -1.5 / gain, 2.5 / gain) * gain : 0.0;
        }
        if (validDepth && SunshineCameraFinite(displacement))
            parallax = -clamp(displacement, -1.5, 2.5) * (float(BUFFER_HEIGHT) * rcp(2160.0) * 100.0) / BUFFER_WIDTH;
    }
    // Same signed source-U safety domain as the current Host conditioner.
    // This bound applies to CURRENT pixels, including newly revealed near
    // geometry that the asynchronous CPU extrema have not observed yet.
    // It never rewrites/clips raw depth and never moves the screen plane.
    SunshineHostCandidateStore[int2(id.xy)] = clamp(parallax, -displayLimit, displayLimit);
}

// Mechanical ReShade binding translation of depth_coordinate_v2_vertical_limit_cs.hlsl
// Original SHA256: fbcad9fe485ea5c82904ea3587afaac26b7bc04bdb61d88f115d405c5ffca5bf
// {forward upper end, forward lower end, backward upper start, backward lower start}
groupshared int4 V_LocalEndsQ30[8u][8u];
// {incoming forward upper, incoming forward lower, incoming backward upper, incoming backward lower}
groupshared int4 V_ChunkCarriesQ30[8u][8u];

int V_V2LimitUpperQ30(float value) {
    value = SunshineCameraFinite(value) ?
        clamp(value, -SunshineHostContainer, SunshineHostContainer) : 0.0f;
    return (int)ceil(value * SunshineHostQScale);
}

int V_V2LimitLowerQ30(float value) {
    value = SunshineCameraFinite(value) ?
        clamp(value, -SunshineHostContainer, SunshineHostContainer) : 0.0f;
    return (int)floor(value * SunshineHostQScale);
}

float V_V2LimitFromQ30(int value) {
    return (float)value / SunshineHostQScale;
}

int V_V2LimitStepQ30(uint content_width, int max_decay_q30) {
    uint exact_step_q30 = 2147483648u / content_width;
    uint bounded_step_q30 = min(exact_step_q30, (uint)max_decay_q30);
    return max(1, (int)bounded_step_q30);
}

int V_V2LimitDecayQ30(int step_q30, uint distance, int max_decay_q30) {
    if (distance == 0u) {
        return 0;
    }
    int signed_distance = (int)distance;
    return step_q30 > max_decay_q30 / signed_distance ?
        max_decay_q30 : step_q30 * signed_distance;
}

// Equal to V_V2LimitDecayQ30 for every distance, with saturation_distance = max_decay / step
// computed once instead of one integer division per texel.
int V_V2LimitSaturatedDecayQ30(int step_q30, uint distance, uint saturation_distance,
                             int max_decay_q30) {
    return distance <= saturation_distance ? step_q30 * (int)distance : max_decay_q30;
}

float V_share_vertical_envelopes(float upper, float lower) {
    // Keep the persisted float32 evaluation order and forbid contraction/reassociation.
    precise float majorant_share = 0.75;
    precise float minorant_share = 1.0f - majorant_share;
    precise float majorant_term = majorant_share * upper;
    precise float minorant_term = minorant_share * lower;
    precise float conditioned = majorant_term + minorant_term;
    return conditioned;
}

[numthreads(8u, 8u, 1)]
void SunshineHostVerticalCS(
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID) {
    uint column = group_thread_id.x;
    uint lane = group_thread_id.y;
    uint x = group_id.x * 8u + column;
    // A partial last group keeps its idle threads until the final group barrier.
    bool active = x < BUFFER_WIDTH;

    float max_step = 2.0 / float(BUFFER_WIDTH);

    // Preserve the tiny diagnostic/unit-test path exactly and avoid empty chunks.
    if (BUFFER_HEIGHT <= 32u) {
        if (active && lane == 0u) {
            float candidate = SunshineHostCandidateSampler.Load(int3(int2(uint2(x, 0u)), 0));
            float upper = candidate;
            float lower = candidate;
            SunshineHostVerticalMajorantStore[int2(uint2(x, 0u))] = upper;
            SunshineHostVerticalConditionedStore[int2(uint2(x, 0u))] = lower;
            [loop]
            for (uint serial_y = 1u; serial_y < BUFFER_HEIGHT; ++serial_y) {
                candidate = SunshineHostCandidateSampler.Load(int3(int2(uint2(x, serial_y)), 0));
                upper = max(candidate, upper - max_step);
                lower = min(candidate, lower + max_step);
                SunshineHostVerticalMajorantStore[int2(uint2(x, serial_y))] = upper;
                SunshineHostVerticalConditionedStore[int2(uint2(x, serial_y))] = lower;
            }
            AllMemoryBarrier();
            const uint last_y = BUFFER_HEIGHT - 1u;
            upper = SunshineHostVerticalMajorantStore[int2(uint2(x, last_y))];
            lower = SunshineHostVerticalConditionedStore[int2(uint2(x, last_y))];
            SunshineHostVerticalConditionedStore[int2(uint2(x, last_y))] = V_share_vertical_envelopes(upper, lower);
            [loop]
            for (int serial_back_y = (int)BUFFER_HEIGHT - 2;
                 serial_back_y >= 0;
                 --serial_back_y) {
                const uint2 position = uint2(x, (uint)serial_back_y);
                upper = max(SunshineHostVerticalMajorantStore[int2(position)], upper - max_step);
                lower = min(SunshineHostVerticalConditionedStore[int2(position)], lower + max_step);
                SunshineHostVerticalMajorantStore[int2(position)] = upper;
                SunshineHostVerticalConditionedStore[int2(position)] = V_share_vertical_envelopes(upper, lower);
            }
        }
        return;
    }

    int max_decay_q30 = 2 * SunshineHostContainerQ;
    uint content_width = BUFFER_WIDTH;
    int max_step_q30 = V_V2LimitStepQ30(content_width, max_decay_q30);
    uint saturation_distance = (uint)(max_decay_q30 / max_step_q30);

    uint chunk_start = lane * BUFFER_HEIGHT / 8u;
    uint chunk_end = (lane + 1u) * BUFFER_HEIGHT / 8u;
    int forward_upper_q30 = 0;
    int forward_lower_q30 = 0;
    int backward_upper_q30 = 0;
    int backward_lower_q30 = 0;
    if (active) {
        // One forward traversal stores this chunk's local forward envelopes and forms both chunk
        // ends. The backward value at chunk_start is the closed form
        // max_s(Candidate(s) - decay(s - chunk_start)). Every Q30 input lies in
        // [-container, container] and the decay saturates at twice that limit, so a saturated
        // term can never exceed Candidate(chunk_start); the serial recurrence gives the same value.
        float candidate = SunshineHostCandidateSampler.Load(int3(int2(uint2(x, chunk_start)), 0));
        forward_upper_q30 = V_V2LimitUpperQ30(candidate);
        forward_lower_q30 = V_V2LimitLowerQ30(candidate);
        backward_upper_q30 = forward_upper_q30;
        backward_lower_q30 = forward_lower_q30;
        SunshineHostVerticalMajorantStore[int2(uint2(x, chunk_start))] = V_V2LimitFromQ30(forward_upper_q30);
        SunshineHostVerticalConditionedStore[int2(uint2(x, chunk_start))] = V_V2LimitFromQ30(forward_lower_q30);
        [loop]
        for (uint local_forward_y = chunk_start + 1u;
             local_forward_y < chunk_end;
             ++local_forward_y) {
            candidate = SunshineHostCandidateSampler.Load(int3(int2(uint2(x, local_forward_y)), 0));
            int upper_q30 = V_V2LimitUpperQ30(candidate);
            int lower_q30 = V_V2LimitLowerQ30(candidate);
            forward_upper_q30 = max(upper_q30, forward_upper_q30 - max_step_q30);
            forward_lower_q30 = min(lower_q30, forward_lower_q30 + max_step_q30);
            int decay_q30 = V_V2LimitSaturatedDecayQ30(
                max_step_q30,
                local_forward_y - chunk_start,
                saturation_distance,
                max_decay_q30);
            backward_upper_q30 = max(backward_upper_q30, upper_q30 - decay_q30);
            backward_lower_q30 = min(backward_lower_q30, lower_q30 + decay_q30);
            SunshineHostVerticalMajorantStore[int2(uint2(x, local_forward_y))] = V_V2LimitFromQ30(forward_upper_q30);
            SunshineHostVerticalConditionedStore[int2(uint2(x, local_forward_y))] = V_V2LimitFromQ30(forward_lower_q30);
        }
    }
    V_LocalEndsQ30[lane][column] = int4(
        forward_upper_q30,
        forward_lower_q30,
        backward_upper_q30,
        backward_lower_q30);
    GroupMemoryBarrierWithGroupSync();

    if (lane == 0u) {
        forward_upper_q30 = V_LocalEndsQ30[0u][column].x;
        forward_lower_q30 = V_LocalEndsQ30[0u][column].y;
        [unroll]
        for (uint forward_chunk = 1u;
             forward_chunk < 8u;
             ++forward_chunk) {
            V_ChunkCarriesQ30[forward_chunk][column].xy = int2(
                forward_upper_q30,
                forward_lower_q30);
            uint forward_start = forward_chunk * BUFFER_HEIGHT / 8u;
            uint forward_end = (forward_chunk + 1u) * BUFFER_HEIGHT / 8u;
            int forward_span_q30 = V_V2LimitDecayQ30(
                max_step_q30,
                forward_end - forward_start,
                max_decay_q30);
            forward_upper_q30 = max(
                V_LocalEndsQ30[forward_chunk][column].x,
                forward_upper_q30 - forward_span_q30);
            forward_lower_q30 = min(
                V_LocalEndsQ30[forward_chunk][column].y,
                forward_lower_q30 + forward_span_q30);
        }

        backward_upper_q30 = V_LocalEndsQ30[8u - 1u][column].z;
        backward_lower_q30 = V_LocalEndsQ30[8u - 1u][column].w;
        [unroll]
        for (int backward_chunk = (int)8u - 2;
             backward_chunk >= 0;
             --backward_chunk) {
            V_ChunkCarriesQ30[(uint)backward_chunk][column].zw = int2(
                backward_upper_q30,
                backward_lower_q30);
            uint backward_start = (uint)backward_chunk * BUFFER_HEIGHT / 8u;
            uint backward_end = ((uint)backward_chunk + 1u) * BUFFER_HEIGHT / 8u;
            int backward_span_q30 = V_V2LimitDecayQ30(
                max_step_q30,
                backward_end - backward_start,
                max_decay_q30);
            backward_upper_q30 = max(
                V_LocalEndsQ30[(uint)backward_chunk][column].z,
                backward_upper_q30 - backward_span_q30);
            backward_lower_q30 = min(
                V_LocalEndsQ30[(uint)backward_chunk][column].w,
                backward_lower_q30 + backward_span_q30);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (!active) {
        return;
    }

    // One backward traversal completes each texel. With the incoming carries,
    //   Forward(y) = max(LocalForward(y), carry - decay(y - chunk_start + 1))
    //   Backward(y) = max(LocalBackward(y), carry - decay(chunk_end - y))
    // (min/plus for the lower envelope). LocalForward was stored as float; Q30-to-float
    // conversion is monotonic, so max/min of converted values equals converting the Q30 max/min.
    // Each thread rereads only texels it wrote itself.
    int4 carry = V_ChunkCarriesQ30[lane][column];
    bool forward_carry = lane != 0u;
    bool backward_carry = lane + 1u != 8u;
    [loop]
    for (int scan_y = (int)chunk_end - 1; scan_y >= (int)chunk_start; --scan_y) {
        uint write_y = (uint)scan_y;
        float candidate = SunshineHostCandidateSampler.Load(int3(int2(uint2(x, write_y)), 0));
        int upper_q30 = V_V2LimitUpperQ30(candidate);
        int lower_q30 = V_V2LimitLowerQ30(candidate);
        if (write_y + 1u == chunk_end) {
            backward_upper_q30 = upper_q30;
            backward_lower_q30 = lower_q30;
        } else {
            backward_upper_q30 = max(upper_q30, backward_upper_q30 - max_step_q30);
            backward_lower_q30 = min(lower_q30, backward_lower_q30 + max_step_q30);
        }
        int complete_upper_q30 = backward_upper_q30;
        int complete_lower_q30 = backward_lower_q30;
        if (backward_carry) {
            int decay_q30 = V_V2LimitSaturatedDecayQ30(
                max_step_q30, chunk_end - write_y, saturation_distance, max_decay_q30);
            complete_upper_q30 = max(complete_upper_q30, carry.z - decay_q30);
            complete_lower_q30 = min(complete_lower_q30, carry.w + decay_q30);
        }
        if (forward_carry) {
            int decay_q30 = V_V2LimitSaturatedDecayQ30(
                max_step_q30, write_y - chunk_start + 1u, saturation_distance, max_decay_q30);
            complete_upper_q30 = max(complete_upper_q30, carry.x - decay_q30);
            complete_lower_q30 = min(complete_lower_q30, carry.y + decay_q30);
        }
        float final_upper = max(
            SunshineHostVerticalMajorantStore[int2(uint2(x, write_y))],
            V_V2LimitFromQ30(complete_upper_q30));
        float final_lower = min(
            SunshineHostVerticalConditionedStore[int2(uint2(x, write_y))],
            V_V2LimitFromQ30(complete_lower_q30));
        SunshineHostVerticalMajorantStore[int2(uint2(x, write_y))] = final_upper;
        SunshineHostVerticalConditionedStore[int2(uint2(x, write_y))] = clamp(
            V_share_vertical_envelopes(final_upper, final_lower),
            final_lower,
            final_upper);
    }
}

// Mechanical ReShade binding translation of depth_coordinate_v2_limit_cs.hlsl
// Original SHA256: 8f9330ea245ef46fa4a7e238355b5a365afe419869fcc2c061e6798cc169a572
// {forward end, backward start}
groupshared int2 H_LocalEndsQ30[8u][8u];
// {incoming forward value, incoming backward value}
groupshared int2 H_ChunkCarriesQ30[8u][8u];

int H_V2LimitUpperQ30(float value) {
    value = SunshineCameraFinite(value) ?
        clamp(value, -SunshineHostContainer, SunshineHostContainer) : 0.0f;
    return (int)ceil(value * SunshineHostQScale);
}

float H_V2LimitFromQ30(int value) {
    return (float)value / SunshineHostQScale;
}

int H_V2LimitStepQ30(uint content_width, int max_decay_q30) {
    uint exact_step_q30 = 536870912u / content_width;
    uint bounded_step_q30 = min(exact_step_q30, (uint)max_decay_q30);
    return max(1, (int)bounded_step_q30);
}

int H_V2LimitDecayQ30(int step_q30, uint distance, int max_decay_q30) {
    if (distance == 0u) {
        return 0;
    }
    int signed_distance = (int)distance;
    return step_q30 > max_decay_q30 / signed_distance ?
        max_decay_q30 : step_q30 * signed_distance;
}

// Equal to H_V2LimitDecayQ30 for every distance, with saturation_distance = max_decay / step
// computed once instead of one integer division per texel.
int H_V2LimitSaturatedDecayQ30(int step_q30, uint distance, uint saturation_distance,
                             int max_decay_q30) {
    return distance <= saturation_distance ? step_q30 * (int)distance : max_decay_q30;
}

float SunshineUIPlaneParallax() {
    // Adaptive live selection is resolved outside the geometry pipeline. Replay
    // receives this exact applied fraction, including intermediate ramp values.
    if (Sunshine_UIPlaneMode == 5u) {
        if (!SunshineHostWarpActive() || !SunshineCameraFinite(Sunshine_UIPlaneInverseDepth) ||
            Sunshine_UIPlaneInverseDepth < 0.0 || Sunshine_UIPlaneInverseDepth > 0.75) return 0.0;
        return Sunshine_UIPlaneInverseDepth * SunshineBoundFinalParallax(SunshineHostContainer);
    }
    // A fixed shallow plane at one quarter of the authoritative display cap.
    // The inverse-depth word is unused; scene geometry only gates admission.
    if (Sunshine_UIPlaneMode == 4u) {
        if (!SunshineHostWarpActive()) return 0.0;
        return 0.25f * SunshineBoundFinalParallax(SunshineHostContainer);
    }
    // A fixed display-space plane at the current permitted front limit. Scene
    // depth, gain and zero affect admission but never position this UI plane.
    // The inverse-depth word is intentionally unused in this explicit mode.
    if (Sunshine_UIPlaneMode == 3u) {
        if (!SunshineHostWarpActive()) return 0.0;
        // Reuse the final scene clamp, including its float operation order.
        return SunshineBoundFinalParallax(SunshineHostContainer);
    }
    // Mode zero preserves historical screen-plane UI. Malformed UI metadata
    // falls back to that plane without disabling otherwise valid scene depth.
    if ((Sunshine_UIPlaneMode != 1u && Sunshine_UIPlaneMode != 2u) || !SunshineHostWarpActive() ||
        !SunshineCameraFinite(Sunshine_UIPlaneInverseDepth) || Sunshine_UIPlaneInverseDepth < 0.0)
        return 0.0;
    float inverse_depth = Sunshine_UIPlaneInverseDepth;
    if (Sunshine_UIPlaneMode == 2u) {
        float resolved = SunshineUIPlaneResolvedSampler.Load(int3(0, 0, 0));
        if (!SunshineCameraFinite(resolved) || resolved < 0.0) return 0.0;
        inverse_depth = max(inverse_depth, resolved);
    }
    // Same native-depth adapter and display budget as the scene candidate,
    // evaluated at the independently resolved UI depth rather than a texel.
    precise float strength = clamp(Depth_Adjustment, 0.0, 100.0) * 0.01 * SunshineAutomaticStrengthBlend();
    precise float displacement = Sunshine_CameraConvergence.x * Sunshine_CameraDepthScale *
        (Sunshine_CameraConvergence.y - inverse_depth) * strength;
    if (!SunshineCameraFinite(displacement)) return 0.0;
    float parallax = -clamp(displacement, -1.5, 2.5) *
        (float(BUFFER_HEIGHT) * rcp(2160.0) * 100.0) / BUFFER_WIDTH;
    // Match the final scene field's bound, including float operation order.
    return SunshineBoundFinalParallax(parallax);
}

// UI pinning runs as its own pass after the scene field h. Each texel u with
// pin weight w(u) > 0 has slack r(u) = (1 - w(u))|h(u) - p| around the UI plane
// p, and the field becomes clamp(h, p - b, p + b) with
// b(x) = min_u r(u) + 0.5*max(|x - u| - 1, 0)/W (docs/reshade-sbs.md, UI pin
// band). Bilinear color reaches one texel beyond a positive-alpha texel
// center, so its neighbors (the collar) share its slack, and a neighboring
// output cannot pull a faint copy of an antialiased glyph into otherwise
// unmasked background. A bound is kept as an anchor texel a and a slack r and
// evaluated in closed form, r + 0.5*|x - a|/W, so a binary mask reproduces the
// distance rule 0.5*max(d - 1, 0)/W exactly.
// Each group owns eight adjacent rows split into eight chunks; no row is held
// in group memory. Per chunk: the anchor and slack reaching furthest right and
// left, then the incoming {left, right} anchors and slacks; slack -1 is none.
groupshared int2 SunshineUIPinAnchors[8u][8u];
groupshared float2 SunshineUIPinSlack[8u][8u];
groupshared int2 SunshineUIPinCarryAnchors[8u][8u];
groupshared float2 SunshineUIPinCarrySlack[8u][8u];
// Each pinning scan loads this many texels of its chunk (weights and field
// values) before it walks them in order: the loads of a block are independent
// and overlap, while the walk does exactly the arithmetic, in exactly the
// order, of a texel-by-texel scan. A block's field values are the ones the
// texel-by-texel scan would read: each texel is read before the scan writes it,
// and the scan writes only texels it has passed. Finer chunks would shorten
// the scans but move the carries' float comparisons, so the field would not
// stay bit-identical.
#define SUNSHINE_UI_PIN_BLOCK 16u

float SunshineUIPinRamp(uint distance) {
    return 0.5 * float(distance) / float(BUFFER_WIDTH);
}

// The bound an anchor with this slack puts on a texel this far away.
float SunshineUIPinBound(float slack, uint distance) {
    return slack + SunshineUIPinRamp(distance);
}

// The smaller of two slacks, either of which may be none.
float SunshineUIPinMinSlack(float a, float b) {
    return a < 0.0 ? b : b < 0.0 ? a : min(a, b);
}

// Zero slack is the binary distance rule in its own arithmetic, so a binary
// mask pins bit for bit as before; a zero bound is the plane itself.
float SunshineUIPinClamp(float value, float plane, float slack, uint distance) {
    float ramp = SunshineUIPinRamp(distance);
    if (slack <= 0.0) return distance == 0u ? plane : clamp(value, plane - ramp, plane + ramp);
    return clamp(value, (plane - slack) - ramp, (plane + slack) + ramp);
}

// The slack of one texel of the original field, or -1 without UI.
float SunshineUIPinSlackOf(float value, float weight, float plane) {
    return weight > 0.0 ? (1.0 - weight) * abs(value - plane) : -1.0;
}

[numthreads(8u, 8u, 1)]
void SunshineHostHorizontalCS(
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID) {
    uint row = group_thread_id.x;
    uint lane = group_thread_id.y;
    uint y = group_id.x * 8u + row;
    // A partial last group keeps its idle threads until the final group barrier.
    bool active = y < BUFFER_HEIGHT;

    float max_step = 0.5 / float(BUFFER_WIDTH);

    // Preserve the tiny diagnostic/unit-test path exactly and avoid empty chunks.
    if (BUFFER_WIDTH <= 32u) {
        if (active && lane == 0u) {
            float value = SunshineHostVerticalConditionedSampler.Load(int3(int2(uint2(0u, y)), 0));
            SunshineHostFinalStore[int2(uint2(0u, y))] = SunshineBoundFinalParallax(value);
            [loop]
            for (uint serial_x = 1u; serial_x < BUFFER_WIDTH; ++serial_x) {
                value = max(SunshineHostVerticalConditionedSampler.Load(int3(int2(uint2(serial_x, y)), 0)), value - max_step);
                SunshineHostFinalStore[int2(uint2(serial_x, y))] = SunshineBoundFinalParallax(value);
            }
            AllMemoryBarrier();
            value = SunshineHostFinalStore[int2(uint2(BUFFER_WIDTH - 1u, y))];
            [loop]
            for (int serial_back_x = (int)BUFFER_WIDTH - 2;
                 serial_back_x >= 0;
                 --serial_back_x) {
                const uint2 position = uint2((uint)serial_back_x, y);
                value = max(SunshineHostFinalStore[int2(position)], value - max_step);
                SunshineHostFinalStore[int2(position)] = SunshineBoundFinalParallax(value);
            }
        }
        return;
    }

    int max_decay_q30 = 2 * SunshineHostContainerQ;
    uint content_width = BUFFER_WIDTH;
    int max_step_q30 = H_V2LimitStepQ30(content_width, max_decay_q30);
    uint saturation_distance = (uint)(max_decay_q30 / max_step_q30);

    uint chunk_start = lane * BUFFER_WIDTH / 8u;
    uint chunk_end = (lane + 1u) * BUFFER_WIDTH / 8u;
    int forward_q30 = 0;
    int backward_q30 = 0;
    if (active) {
        // One forward traversal stores this chunk's local forward majorant and forms both chunk
        // ends. The backward value at chunk_start is the closed form
        // max_s(VerticalShare(s) - decay(s - chunk_start)). Every Q30 input lies in
        // [-container, container] and the decay saturates at twice that limit, so a saturated
        // term can never exceed VerticalShare(chunk_start); the serial recurrence gives the same
        // value.
        forward_q30 = H_V2LimitUpperQ30(SunshineHostVerticalConditionedSampler.Load(int3(int2(uint2(chunk_start, y)), 0)));
        backward_q30 = forward_q30;
        SunshineHostFinalStore[int2(uint2(chunk_start, y))] = H_V2LimitFromQ30(forward_q30);
        [loop]
        for (uint local_forward_x = chunk_start + 1u;
             local_forward_x < chunk_end;
             ++local_forward_x) {
            int value_q30 = H_V2LimitUpperQ30(SunshineHostVerticalConditionedSampler.Load(int3(int2(uint2(local_forward_x, y)), 0)));
            forward_q30 = max(value_q30, forward_q30 - max_step_q30);
            int decay_q30 = H_V2LimitSaturatedDecayQ30(
                max_step_q30,
                local_forward_x - chunk_start,
                saturation_distance,
                max_decay_q30);
            backward_q30 = max(backward_q30, value_q30 - decay_q30);
            SunshineHostFinalStore[int2(uint2(local_forward_x, y))] = H_V2LimitFromQ30(forward_q30);
        }
    }
    H_LocalEndsQ30[lane][row] = int2(forward_q30, backward_q30);
    GroupMemoryBarrierWithGroupSync();

    if (lane == 0u) {
        forward_q30 = H_LocalEndsQ30[0u][row].x;
        [unroll]
        for (uint forward_chunk = 1u;
             forward_chunk < 8u;
             ++forward_chunk) {
            H_ChunkCarriesQ30[forward_chunk][row].x = forward_q30;
            uint forward_start = forward_chunk * BUFFER_WIDTH / 8u;
            uint forward_end = (forward_chunk + 1u) * BUFFER_WIDTH / 8u;
            int forward_span_q30 = H_V2LimitDecayQ30(
                max_step_q30,
                forward_end - forward_start,
                max_decay_q30);
            forward_q30 = max(
                H_LocalEndsQ30[forward_chunk][row].x,
                forward_q30 - forward_span_q30);
        }

        backward_q30 = H_LocalEndsQ30[8u - 1u][row].y;
        [unroll]
        for (int backward_chunk = (int)8u - 2;
             backward_chunk >= 0;
             --backward_chunk) {
            H_ChunkCarriesQ30[(uint)backward_chunk][row].y = backward_q30;
            uint backward_start = (uint)backward_chunk * BUFFER_WIDTH / 8u;
            uint backward_end = ((uint)backward_chunk + 1u) * BUFFER_WIDTH / 8u;
            int backward_span_q30 = H_V2LimitDecayQ30(
                max_step_q30,
                backward_end - backward_start,
                max_decay_q30);
            backward_q30 = max(
                H_LocalEndsQ30[(uint)backward_chunk][row].y,
                backward_q30 - backward_span_q30);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (!active) {
        return;
    }

    // One backward traversal completes each texel from the stored local forward majorant, the
    // local backward recurrence and both incoming carries. Q30-to-float conversion is monotonic,
    // so max of converted values equals converting the Q30 max. Each thread rereads only texels
    // it wrote itself.
    int2 carry = H_ChunkCarriesQ30[lane][row];
    bool forward_carry = lane != 0u;
    bool backward_carry = lane + 1u != 8u;
    [loop]
    for (int scan_x = (int)chunk_end - 1; scan_x >= (int)chunk_start; --scan_x) {
        uint write_x = (uint)scan_x;
        int value_q30 = H_V2LimitUpperQ30(SunshineHostVerticalConditionedSampler.Load(int3(int2(uint2(write_x, y)), 0)));
        backward_q30 = write_x + 1u == chunk_end ?
            value_q30 : max(value_q30, backward_q30 - max_step_q30);
        int complete_q30 = backward_q30;
        if (backward_carry) {
            int decay_q30 = H_V2LimitSaturatedDecayQ30(
                max_step_q30, chunk_end - write_x, saturation_distance, max_decay_q30);
            complete_q30 = max(complete_q30, carry.y - decay_q30);
        }
        if (forward_carry) {
            int decay_q30 = H_V2LimitSaturatedDecayQ30(
                max_step_q30, write_x - chunk_start + 1u, saturation_distance, max_decay_q30);
            complete_q30 = max(complete_q30, carry.x - decay_q30);
        }
        // Game 3D bounds the completed field by its display cap; the stored
        // forward majorant above is an unbounded intermediate.
        SunshineHostFinalStore[int2(uint2(write_x, y))] = SunshineBoundFinalParallax(max(
            SunshineHostFinalStore[int2(uint2(write_x, y))],
            H_V2LimitFromQ30(complete_q30)));
    }
}

// UI pinning runs after the scene field is complete. A probe frame observes
// the unpinned field first, so it never measures its own pinned field.
[numthreads(8, 8, 1)]
void SunshineApplyUICS(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID) {
    if (Sunshine_SourceAlphaUI == 0u) return;
    uint row = thread_id.x, lane = thread_id.y;
    uint y = group_id.x * 8u + row;
    // A partial last group keeps its idle threads until the final barrier.
    bool active = y < BUFFER_HEIGHT;
    uint chunk_start = lane * BUFFER_WIDTH / 8u;
    uint chunk_end = (lane + 1u) * BUFFER_WIDTH / 8u;
    float plane = SunshineUIPlaneParallax();
    // Beyond its chunk a texel u bounds the field from its collar: anchor u + 1
    // to the right and u - 1 to the left. Keep the one with the least bound
    // where the next chunk starts and where the previous one ends.
    int2 anchors = int2(0, 0);
    float2 slack = float2(-1.0, -1.0), reach = float2(0.0, 0.0);
    if (active) {
        [loop]
        for (uint block = chunk_start; block < chunk_end; block += SUNSHINE_UI_PIN_BLOCK) {
            float weights[SUNSHINE_UI_PIN_BLOCK], values[SUNSHINE_UI_PIN_BLOCK];
            [unroll] for (uint i = 0u; i < SUNSHINE_UI_PIN_BLOCK; ++i) {
                const uint at = min(block + i, chunk_end - 1u);
                weights[i] = SunshineUIPinWeight(at, y);
                values[i] = SunshineHostFinalStore[int2(uint2(at, y))];
            }
            [unroll] for (uint j = 0u; j < SUNSHINE_UI_PIN_BLOCK; ++j) {
                const uint x = block + j;
                float weight = weights[j];
                if (x >= chunk_end || weight <= 0.0) continue;
                // Fully weighted UI has no slack whatever the field.
                float r = weight >= 1.0 ? 0.0 : SunshineUIPinSlackOf(values[j], weight, plane);
                float right = SunshineUIPinBound(r, chunk_end - 1u - x);
                if (slack.x < 0.0 || right <= reach.x) { anchors.x = (int)x + 1; slack.x = r; reach.x = right; }
                float left = SunshineUIPinBound(r, x - chunk_start);
                if (slack.y < 0.0 || left < reach.y) { anchors.y = (int)x - 1; slack.y = r; reach.y = left; }
            }
        }
    }
    SunshineUIPinAnchors[lane][row] = anchors;
    SunshineUIPinSlack[lane][row] = slack;
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        // Each chunk's left carry is the best rightward anchor of the chunks
        // before it, compared where it starts; ties keep the nearer anchor.
        int anchor = 0;
        float best = -1.0;
        [unroll]
        for (uint chunk = 0u; chunk < 8u; ++chunk) {
            SunshineUIPinCarryAnchors[chunk][row].x = anchor;
            SunshineUIPinCarrySlack[chunk][row].x = best;
            int next = (int)((chunk + 1u) * BUFFER_WIDTH / 8u);
            int offered_anchor = SunshineUIPinAnchors[chunk][row].x;
            float offered = SunshineUIPinSlack[chunk][row].x;
            if (offered >= 0.0 && (best < 0.0 ||
                    SunshineUIPinBound(offered, (uint)(next - offered_anchor)) <= SunshineUIPinBound(best, (uint)(next - anchor)))) {
                anchor = offered_anchor; best = offered;
            }
        }
        // The right carry: the best leftward anchor of the chunks after it,
        // compared at its last texel.
        anchor = 0; best = -1.0;
        [unroll]
        for (int chunk = 7; chunk >= 0; --chunk) {
            SunshineUIPinCarryAnchors[(uint)chunk][row].y = anchor;
            SunshineUIPinCarrySlack[(uint)chunk][row].y = best;
            int previous = (int)((uint)chunk * BUFFER_WIDTH / 8u) - 1;
            int offered_anchor = SunshineUIPinAnchors[(uint)chunk][row].y;
            float offered = SunshineUIPinSlack[(uint)chunk][row].y;
            if (offered >= 0.0 && (best < 0.0 ||
                    SunshineUIPinBound(offered, (uint)(offered_anchor - previous)) <= SunshineUIPinBound(best, (uint)(anchor - previous)))) {
                anchor = offered_anchor; best = offered;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    int2 carry_anchor = SunshineUIPinCarryAnchors[lane][row];
    float2 carry_slack = SunshineUIPinCarrySlack[lane][row];
    // A row without UI is left bit-for-bit intact.
    if (!active || (carry_slack.x < 0.0 && carry_slack.y < 0.0 && slack.x < 0.0)) return;
    // The same plane again: FXC marks everything evaluated before this precise
    // arithmetic precise, and the clamps below must stay the multiply-adds of
    // the binary distance rule.
    plane = SunshineUIPlaneParallax();

    // Backward: bound each texel by the best anchor at or right of it. In the
    // chunk a texel's collar slack is the least slack of itself and its
    // neighbors; neighbors in other chunks arrive through the carries.
    int anchor = carry_anchor.y;
    float best = carry_slack.y;
    float value = 0.0, r_here = -1.0, r_next = -1.0;
    bool flat = true;
    if (chunk_start < chunk_end) {
        value = SunshineHostFinalStore[int2(uint2(chunk_end - 1u, y))];
        r_here = SunshineUIPinSlackOf(value, SunshineUIPinWeight(chunk_end - 1u, y), plane);
    }
    // Blocks from the chunk's end: texel top - j reads its left neighbor
    // top - j - 1, which the scan writes only after it.
    [loop]
    for (int top = (int)chunk_end - 1; top >= (int)chunk_start; top -= (int)SUNSHINE_UI_PIN_BLOCK) {
        float previous_weights[SUNSHINE_UI_PIN_BLOCK], previous_values[SUNSHINE_UI_PIN_BLOCK];
        [unroll] for (uint i = 0u; i < SUNSHINE_UI_PIN_BLOCK; ++i) {
            const uint at = (uint)max(top - (int)i - 1, (int)chunk_start);
            previous_weights[i] = SunshineUIPinWeight(at, y);
            previous_values[i] = SunshineHostFinalStore[int2(uint2(at, y))];
        }
        [unroll] for (uint j = 0u; j < SUNSHINE_UI_PIN_BLOCK; ++j) {
            const int scan_x = top - (int)j;
            if (scan_x < (int)chunk_start) break;
            uint x = (uint)scan_x;
            float previous = 0.0, r_previous = -1.0;
            if (x > chunk_start) {
                previous = previous_values[j];
                r_previous = SunshineUIPinSlackOf(previous, previous_weights[j], plane);
            }
            float collar = SunshineUIPinMinSlack(SunshineUIPinMinSlack(r_previous, r_here), r_next);
            if (collar >= 0.0 && (best < 0.0 || collar <= SunshineUIPinBound(best, (uint)(anchor - scan_x)))) {
                anchor = scan_x; best = collar;
            }
            flat = flat && best == 0.0 && anchor == scan_x;
            if (best >= 0.0) {
                float pinned = SunshineUIPinClamp(value, plane, best, (uint)(anchor - scan_x));
                if (asuint(pinned) != asuint(value)) SunshineHostFinalStore[int2(uint2(x, y))] = pinned;
            }
            r_next = r_here; r_here = r_previous; value = previous;
        }
    }
    // A chunk that is the plane throughout has nothing left to bound.
    if (flat) return;

    // Forward: bound each texel by the best anchor at or left of it. The left
    // carry was taken from the original field. The backward pass replaced the
    // field in this chunk, so a collar texel offers |pinned - p|, never more
    // than its collar slack; whatever it adds beyond that is no tighter than
    // the backward bound or than h itself (h changes by at most 0.5/W a texel).
    anchor = carry_anchor.x;
    best = carry_slack.x;
    float w_previous = 0.0, w_here = chunk_start < chunk_end ? SunshineUIPinWeight(chunk_start, y) : 0.0;
    // Blocks from the chunk's start: texel x is read (as the backward pass
    // left it) before the scan writes it, and the scan writes only behind.
    [loop]
    for (uint block = chunk_start; block < chunk_end; block += SUNSHINE_UI_PIN_BLOCK) {
        float next_weights[SUNSHINE_UI_PIN_BLOCK], currents[SUNSHINE_UI_PIN_BLOCK];
        [unroll] for (uint i = 0u; i < SUNSHINE_UI_PIN_BLOCK; ++i) {
            next_weights[i] = SunshineUIPinWeight(min(block + i + 1u, chunk_end - 1u), y);
            currents[i] = SunshineHostFinalStore[int2(uint2(min(block + i, chunk_end - 1u), y))];
        }
        [unroll] for (uint j = 0u; j < SUNSHINE_UI_PIN_BLOCK; ++j) {
            const uint x = block + j;
            if (x >= chunk_end) break;
            float w_next = x + 1u < chunk_end ? next_weights[j] : 0.0;
            bool collar = w_previous > 0.0 || w_here > 0.0 || w_next > 0.0;
            // Next to fully weighted UI the backward pass left the plane itself.
            if (max(max(w_previous, w_here), w_next) >= 1.0) {
                anchor = (int)x; best = 0.0;
            } else if (collar || best >= 0.0) {
                float current = currents[j];
                if (collar) {
                    float offered = abs(current - plane);
                    if (best < 0.0 || offered <= SunshineUIPinBound(best, (uint)((int)x - anchor))) { anchor = (int)x; best = offered; }
                }
                float pinned = SunshineUIPinClamp(current, plane, best, (uint)((int)x - anchor));
                if (asuint(pinned) != asuint(current)) SunshineHostFinalStore[int2(uint2(x, y))] = pinned;
            }
            w_previous = w_here; w_here = w_next;
        }
    }
}

// Exact tile counts from the selected mask and the unprotected, conditioned
// scene. Only texels the mask pins exactly (pin weight one) count: fainter
// tails blend toward scene depth and never place the plane. Each tile writes
// two uint4 rows: coverage, invalid, five conflict counts, and pixel count.
// The absolute candidate planes remain within half the current scene cap.
// All 16x16 tiles fit in an 8 KiB statistics texture.
// Edge tiles publish only their pixel partition; placement evidence and source
// reads are restricted to the exact central 75% rectangle.
groupshared uint4 SunshineUIConflictA[64];
groupshared uint4 SunshineUIConflictB[64];
[numthreads(8, 8, 1)]
void SunshineUIConflictCS(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID) {
    uint lane = thread_id.y * 8u + thread_id.x;
    uint2 first = group_id.xy * uint2(BUFFER_WIDTH, BUFFER_HEIGHT) / 16u;
    uint2 last = (group_id.xy + 1u) * uint2(BUFFER_WIDTH, BUFFER_HEIGHT) / 16u;
    if (group_id.x < 2u || group_id.x >= 14u || group_id.y < 2u || group_id.y >= 14u) {
        if (lane == 0u) {
            SunshineUIConflictStore[int2(group_id.x, group_id.y * 2u)] = uint4(0u, 0u, 0u, 0u);
            SunshineUIConflictStore[int2(group_id.x, group_id.y * 2u + 1u)] =
                uint4(0u, 0u, 0u, (last.x - first.x) * (last.y - first.y));
        }
        return;
    }
    uint4 a = 0u, b = 0u;
    bool active = SunshineHostWarpActive();
    float cap = SunshineBoundFinalParallax(SunshineHostContainer);
    float uiLimit = 0.5 * cap;
    [loop]
    for (uint y = first.y + thread_id.y; y < last.y; y += 8u) {
        [loop]
        for (uint x = first.x + thread_id.x; x < last.x; x += 8u) {
            b.w += 1u;
            if (SunshineUIPinWeight(x, y) < 1.0) continue;
            a.x += 1u;
            float p = SunshineHostFinalSampler.Load(int3(int2(x, y), 0));
            if (!active || !SunshineCameraFinite(p) || cap <= 0.0) { a.y += 1u; continue; }
            float required = p + 0.05 * cap;
            a.z += required > 0.0 ? 1u : 0u;
            a.w += required > min(0.001, uiLimit) ? 1u : 0u;
            b.x += required > min(0.002, uiLimit) ? 1u : 0u;
            b.y += required > min(0.003, uiLimit) ? 1u : 0u;
            b.z += required > min(0.0035, uiLimit) ? 1u : 0u;
        }
    }
    SunshineUIConflictA[lane] = a;
    SunshineUIConflictB[lane] = b;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint step = 32u; step != 0u; step >>= 1u) {
        if (lane < step) {
            SunshineUIConflictA[lane] += SunshineUIConflictA[lane + step];
            SunshineUIConflictB[lane] += SunshineUIConflictB[lane + step];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) {
        // No covered pixels does not prove clearance if scene admission failed.
        // An impossible count invalidates the whole bounded observation.
        if ((!active || cap <= 0.0) && all(group_id.xy == 2u)) SunshineUIConflictA[0].y = 0xffffffffu;
        SunshineUIConflictStore[int2(group_id.x, group_id.y * 2u)] = SunshineUIConflictA[0];
        SunshineUIConflictStore[int2(group_id.x, group_id.y * 2u + 1u)] = SunshineUIConflictB[0];
    }
}

float4 SunshineHostRender(float2 eyeUV, bool rightEye)
{
    float eyeSign = rightEye ? 1.0 : -1.0;
    float sourceX = eyeUV.x;
    // FX translates for-loops to while-loops; forcing unroll fails FXC for this
    // exact-equality break. A dynamic loop keeps the same 11-step upper bound.
    [loop]
    for (int iteration = 0; iteration < 11; ++iteration)
    {
        float nextSourceX = eyeUV.x + eyeSign * SunshineHostFinalSampler.SampleLevel(SunshineLinearClampState, (float4(sourceX, eyeUV.y, 0, 0)).xy, (float4(sourceX, eyeUV.y, 0, 0)).w);
        bool settled = asuint(nextSourceX) == asuint(sourceX);
        sourceX = nextSourceX;
        if (settled) break;
    }
    // Diagnostic corresponds to the actual conditioned field, near bright.
    if (Depth_Map_View == 1)
    {
        float p = SunshineHostFinalSampler.SampleLevel(SunshineLinearClampState, (float4(eyeUV, 0, 0)).xy, (float4(eyeUV, 0, 0)).w);
        return float4((0.5 + 0.5 * p / SunshineHostContainer).xxx, 1.0);
    }
    float2 sourceUV = float2(saturate(sourceX), eyeUV.y);
    #if BUFFER_COLOR_SPACE == 3
        float4 color = float4(SunshineSourceLinear(sourceUV), 1.0);
    #else
        float4 color = SunshineSourceSampler.SampleLevel(SunshineLinearClampState, (float4(sourceUV, 0, 0)).xy, (float4(sourceUV, 0, 0)).w);
    #endif
    // Native SDR/scRGB or already-decoded PQ: do not re-decode or expand here.
    // Host has no synthetic internal fill. Only finite-source clamping lacks coverage.
    color.a = sourceX < 0.0 || sourceX > 1.0 ? 0.0 : 1.0;
    return color;
}

#endif
void PostProcessVS(uint id : SV_VertexID, out float4 position : SV_Position,
    out float2 texcoord : TEXCOORD0)
{
    texcoord = float2(id == 2 ? 2.0 : 0.0, id == 1 ? 2.0 : 0.0);
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}
float4 SunshineRenderEye(float2 eyeUV, bool rightEye)
{
    if (SunshineAutomaticMono())
        return SunshineNativeSource(eyeUV);
    if (Depth_Map_View == 2) {
        float2 uv = SunshineCameraDepthCoordinates(eyeUV, SunshineDepthAllocationSize());
        float depth = SunshineCameraDepth(DepthBuffer.SampleLevel(SunshinePointBorder, (float4(uv, 0, 0)).xy, (float4(uv, 0, 0)).w).x);
        depth = SunshineCameraFinite(depth) ? saturate(depth) : 0.0;
        return float4(depth.xxx, 1.0);
    }
#if _SUNSHINE_HOST_WARP_SUPPORTED
    return SunshineHostRender(eyeUV, rightEye);
#else
    return SunshineNativeSource(eyeUV);
#endif
}
float4 SunshineRenderPackedPS(float4 position : SV_Position, float2 texcoord : TEXCOORD0) : SV_Target
{
    uint packedX = (uint)position.x;
    // Missing depth and zero strength export the current source.
    if (SunshineAutomaticMono()) {
        float2 monoUV = float2((packedX % BUFFER_WIDTH) + 0.5, position.y) /
            float2(BUFFER_WIDTH, BUFFER_HEIGHT);
        return SunshineNativeSource(monoUV);
    }
    // Each branch sees a literal eye, as the former per-eye pass did, so the
    // compiler folds the pixel-center scale into the warp's first add exactly
    // as before (one rounding). Subtracting the integer width is exact and
    // yields that eye's rasterized pixel center.
    float4 color;
    [branch] if (packedX < BUFFER_WIDTH) {
        color = SunshineRenderEye(position.xy / float2(BUFFER_WIDTH, BUFFER_HEIGHT), false);
    } else {
        float2 center = float2(position.x - float(BUFFER_WIDTH), position.y);
        color = SunshineRenderEye(center / float2(BUFFER_WIDTH, BUFFER_HEIGHT), true);
    }
#if BUFFER_COLOR_SPACE == 1
    // Keep the FP16 eye representation the former intermediate held before
    // RGB10A2 packing.
    return f16tof32(f32tof16(color));
#else
    // The HDR export is itself FP16, the representation the eyes held.
    return color;
#endif
}
#if BUFFER_COLOR_SPACE == 3
// The 10-bit PQ export of an HDR10 source (R10G10B10A2_UNORM target): each
// eye's scRGB value, held in FP16 as the FP16 export holds it, encoded as the
// host encodes that export (SunshineHostPQ); the mono pack writes the source's
// own code values. Alpha is the same 0/1 coverage.
float4 SunshineRenderPackedPQPS(float4 position : SV_Position, float2 texcoord : TEXCOORD0) : SV_Target
{
    uint packedX = (uint)position.x;
    if (SunshineAutomaticMono())
        return float4(saturate(SunshineSourceSampler.Load(int3(int(packedX % BUFFER_WIDTH), int(position.y), 0)).rgb), 1.0);
    float4 color;
    [branch] if (packedX < BUFFER_WIDTH) {
        color = SunshineRenderEye(position.xy / float2(BUFFER_WIDTH, BUFFER_HEIGHT), false);
    } else {
        float2 center = float2(position.x - float(BUFFER_WIDTH), position.y);
        color = SunshineRenderEye(center / float2(BUFFER_WIDTH, BUFFER_HEIGHT), true);
    }
    return float4(SunshineHostPQ(f16tof32(f32tof16(color.rgb))), color.a);
}
#endif
