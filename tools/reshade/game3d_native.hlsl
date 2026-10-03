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
// UI pinning groups own eight adjacent rows.
#define SUNSHINE_UI_PIN_LINE_GROUPS 8
// UI mask alpha pins with weight saturate(gain * alpha) through a soft band
// (docs/reshade-sbs.md, UI pin band).
#define SUNSHINE_UI_SOFT_PIN_GAIN 8
// Automatic UI detection writes this many decision texels; its statistics
// rows hold the cells of this many scene-evidence images (docs/reshade-sbs.md,
// UI detection flags and decision texels).
#define SUNSHINE_UI_DECISION_TEXELS 7
#define SUNSHINE_UI_SCENE_EVIDENCE_IMAGES 2
// Hidden-scene evidence D (docs/reshade-sbs.md, hidden-scene evidence),
// mirrored from game3d_ui_detection_contract.h: a grid of cells over the
// frame, the parallax step of a depth edge in pixels per 2160 rows, the edge
// cells valid evidence needs, the hidden and visible bounds of D in percent,
// the fixed point of the cell sums, and the gates' shares of nearly opaque
// and of changed pixels in percent.
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
// Exact per-session UI counters (docs/reshade-sbs.md, UI counters), mirrored
// from game3d_ui_counters.h: the detection reduce adds every detection frame
// to these words of SunshineUICountersStore. The no-mask reasons are offsets
// from SUNSHINE_UI_COUNTER_NONE.
#define SUNSHINE_UI_COUNTER_WORDS 25
#define SUNSHINE_UI_COUNTER_DETECTION_FRAMES 0
#define SUNSHINE_UI_COUNTER_DECIDED 1
#define SUNSHINE_UI_COUNTER_UNTRUSTED_INFERRED 11
#define SUNSHINE_UI_COUNTER_INEXACT_DIFFERENCE 12
#define SUNSHINE_UI_COUNTER_DEPTH_NOT_CURRENT 13
#define SUNSHINE_UI_COUNTER_TRUSTED_FULL 14
#define SUNSHINE_UI_COUNTER_PRESENTED_OVER_DEDICATED 15
#define SUNSHINE_UI_COUNTER_NONE 16
#define SUNSHINE_UI_COUNTER_FULL_ALPHA 24
#define SUNSHINE_UI_NONE_LAYER_ASIDE 0
#define SUNSHINE_UI_NONE_TRUSTED_INVALID 1
#define SUNSHINE_UI_NONE_PRESENTED_BLOCKED 2
#define SUNSHINE_UI_NONE_AMBIGUOUS 3
#define SUNSHINE_UI_NONE_DIFFERENCE_FAILED 4
#define SUNSHINE_UI_NONE_GATE_NO_HOLD 5
#define SUNSHINE_UI_NONE_NO_CANDIDATE 6
#define SUNSHINE_UI_NONE_OTHER 7
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
// the color a HUD-less image is paired with. Read by scene evidence only.
Texture2D<float4> SunshinePresentedColor : register(t6);
Texture2D<float> SunshineUIPlaneTilesSampler : register(t8);
Texture2D<float> SunshineUIPlaneResolvedSampler : register(t9);
Texture2D<uint4> SunshineUIDetectionSampler : register(t10);
Texture2D<float4> SunshineUIDedicatedAlpha : register(t11);
Texture2D<float4> SunshineUIColorAlpha : register(t12);
Texture2D<float4> SunshineUIBackbufferAlpha : register(t13);
Texture2D<float4> SunshineHUDless : register(t14);
// Sunshine_UIDetectionFlags (docs/reshade-sbs.md, UI detection flags and
// decision texels), mirrored from game3d_ui_detection_contract.h. Stored bits
// describe the UI color slot's source: it must be premultiplied, with a float
// layer's HDR headroom, and is the one-frame-late offscreen UI layer (which
// selects the hidden-scene layer route and is trusted per source).
// Per-frame bits ride in one render's pushed word only: the CPU holds a
// hidden-scene verdict for the layer or the HUD-less route, or the consumed
// depth is not this frame's.
#define SUNSHINE_UI_STORED_PREMULTIPLIED 0x1u
#define SUNSHINE_UI_STORED_HDR_HEADROOM 0x2u
#define SUNSHINE_UI_STORED_LATE_LAYER 0x4u
// 0x8u (stored) and 0x20000u (per-frame) are reserved and never reused.
#define SUNSHINE_UI_PER_FRAME_SCENE_HOLD 0x10000u
#define SUNSHINE_UI_PER_FRAME_DEPTH_NOT_CURRENT 0x40000u
#define SUNSHINE_UI_PER_FRAME_SCENE_HOLD_HUDLESS 0x80000u
cbuffer SunshineUIDetectionConstants : register(b2)
{
    uint Sunshine_UICandidates; // bit0..2 captured alpha, bit3 current alpha, bit4 paired HUDless, bit5 exact pair.
    float Sunshine_UIDifferenceThreshold;
    uint Sunshine_UITrustedAlpha; // Bit i: the game session trusts alpha candidate i as UI coverage.
    uint Sunshine_UIDetectionFlags; // SUNSHINE_UI_STORED_* and SUNSHINE_UI_PER_FRAME_* bits.
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
// authorize a changed candidate: session trust only names the channel that
// carries UI coverage, and this frame's own pixels are the mask.
float4 SunshineUIDetectionAlpha(uint2 xy)
{
    int3 at = int3(xy, 0);
    return float4(SunshineUIDedicatedAlpha.Load(at).r, SunshineUIColorAlpha.Load(at).a,
        SunshineUIBackbufferAlpha.Load(at).a, SunshineSourceSampler.Load(at).a);
}
float SunshineHUDlessDifference(uint2 xy, out bool valid)
{
    float3 a = SunshineSourceSampler.Load(int3(xy, 0)).rgb;
    float3 b = SunshineHUDless.Load(int3(xy, 0)).rgb;
    valid = all(isfinite(a)) && all(isfinite(b));
    // scRGB is unbounded linear light; use a relative tolerance above one.
    float scale = BUFFER_COLOR_SPACE == 2 ? max(1.0, max(max(abs(a.r), abs(a.g)), abs(a.b))) : 1.0;
    return max(max(abs(a.r-b.r), abs(a.g-b.g)), abs(a.b-b.b)) / scale;
}
// Each of the 16x16 tiles is one group of 256 threads; the integer counts do
// not depend on how pixels are split among them.
groupshared uint4 SunshineUIDetectionCoverage[256];
groupshared uint4 SunshineUIDetectionInvalid[256];
groupshared uint4 SunshineUIDetectionDifference[256];
// Lit HUD-less pixels, then nearly opaque pixels of alpha candidates 0 and 1.
groupshared uint3 SunshineUIDetectionLit[256];
[numthreads(16, 16, 1)]
void SunshineUIDetectionTilesCS(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint2 first = group.xy * uint2(BUFFER_WIDTH, BUFFER_HEIGHT) / 16u;
    uint2 last = (group.xy + 1u) * uint2(BUFFER_WIDTH, BUFFER_HEIGHT) / 16u;
    uint4 coverage = 0u, invalid = 0u, difference = 0u;
    uint3 lit = 0u;
    [loop] for (uint y = first.y + thread.y; y < last.y; y += 16u)
    [loop] for (uint x = first.x + thread.x; x < last.x; x += 16u) {
        float4 a = SunshineUIDetectionAlpha(uint2(x,y));
        bool4 okay = isfinite(a) && a >= 0.0 && a <= 1.0;
        coverage += uint4(okay && a > 0.0);
        invalid += uint4(!okay);
        // The hidden-scene gate's opacity: UI alpha or UI color alpha of at
        // least 254/255.
        lit.yz += uint2(okay.xy && a.xy >= 254.0 / 255.0);
        // An offscreen layer is UI only when blended over transparent black:
        // its color stays within a small multiple of its alpha. UI tinted
        // brighter than white (Stellar Blade's pulsing markers, up to twice
        // its alpha) qualifies; a scene buffer whose alpha is not coverage
        // (Dead Space: alpha near 11/255 under saturated color) does not.
        // scRGB UI may be up to 10000 nits.
        if (Sunshine_UIDetectionFlags & SUNSHINE_UI_STORED_PREMULTIPLIED) {
            float4 layer = SunshineUIColorAlpha.Load(int3(x, y, 0));
            float headroom = Sunshine_UIDetectionFlags & SUNSHINE_UI_STORED_HDR_HEADROOM ? 125.0 : 2.0;
            invalid.y += max(max(layer.r, layer.g), layer.b) > layer.a * headroom + 4.0 / 255.0 ? 1u : 0u;
        }
        bool finite;
        float delta = SunshineHUDlessDifference(uint2(x,y), finite);
        difference.x += finite && delta > Sunshine_UIDifferenceThreshold ? 1u : 0u;
        difference.y += finite ? 0u : 1u;
        difference.z += finite && delta <= Sunshine_UIDifferenceThreshold * .5 ? 1u : 0u;
        difference.w += 1u;
        // A HUD-less pixel that shows scene content rather than black.
        float3 hudless = SunshineHUDless.Load(int3(x, y, 0)).rgb;
        lit.x += finite && max(max(abs(hudless.r), abs(hudless.g)), abs(hudless.b)) > Sunshine_UIDifferenceThreshold * 8.0 ? 1u : 0u;
    }
    uint lane = thread.y * 16u + thread.x;
    SunshineUIDetectionCoverage[lane] = coverage;
    SunshineUIDetectionInvalid[lane] = invalid;
    SunshineUIDetectionDifference[lane] = difference;
    SunshineUIDetectionLit[lane] = lit;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint step = 128u; step; step >>= 1u) {
        if (lane < step) {
            SunshineUIDetectionCoverage[lane] += SunshineUIDetectionCoverage[lane+step];
            SunshineUIDetectionInvalid[lane] += SunshineUIDetectionInvalid[lane+step];
            SunshineUIDetectionDifference[lane] += SunshineUIDetectionDifference[lane+step];
            SunshineUIDetectionLit[lane] += SunshineUIDetectionLit[lane+step];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (!lane) {
        SunshineAlphaCoverageStore[group.xy] = SunshineUIDetectionCoverage[0];
        SunshineAlphaCoverageStore[group.xy + uint2(0,16)] = SunshineUIDetectionInvalid[0];
        SunshineAlphaCoverageStore[group.xy + uint2(0,32)] = SunshineUIDetectionDifference[0];
        SunshineAlphaCoverageStore[group.xy + uint2(0,48)] = uint4(SunshineUIDetectionLit[0], 0u);
    }
}
// One thread per tile, then an exact group sum; thread 0 decides.
groupshared uint SunshineUIDetectionMatching[256];
[numthreads(256, 1, 1)]
void SunshineUIDetectionReduceCS(uint3 thread : SV_GroupThreadID)
{
    uint lane = thread.x;
    uint2 tile = uint2(lane % 16u, lane / 16u);
    uint4 d = SunshineUIDetectionSampler.Load(int3(tile.x, tile.y + 32u, 0));
    SunshineUIDetectionCoverage[lane] = SunshineUIDetectionSampler.Load(int3(tile, 0));
    SunshineUIDetectionInvalid[lane] = SunshineUIDetectionSampler.Load(int3(tile.x, tile.y + 16u, 0));
    SunshineUIDetectionDifference[lane] = d;
    SunshineUIDetectionLit[lane] = SunshineUIDetectionSampler.Load(int3(tile.x, tile.y + 48u, 0)).xyz;
    SunshineUIDetectionMatching[lane] = d.w && d.z * 100u >= d.w * 99u ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint step = 128u; step; step >>= 1u) {
        if (lane < step) {
            SunshineUIDetectionCoverage[lane] += SunshineUIDetectionCoverage[lane+step];
            SunshineUIDetectionInvalid[lane] += SunshineUIDetectionInvalid[lane+step];
            SunshineUIDetectionDifference[lane] += SunshineUIDetectionDifference[lane+step];
            SunshineUIDetectionLit[lane] += SunshineUIDetectionLit[lane+step];
            SunshineUIDetectionMatching[lane] += SunshineUIDetectionMatching[lane+step];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane) return;
    uint4 coverage = SunshineUIDetectionCoverage[0], invalid = SunshineUIDetectionInvalid[0];
    uint4 difference = SunshineUIDetectionDifference[0];
    uint matching_tiles = SunshineUIDetectionMatching[0], lit = SunshineUIDetectionLit[0].x;
    uint2 opaque = SunshineUIDetectionLit[0].yz;
    uint source = 0u, covered = 0u;
    // A channel the game session trusts as UI coverage is the mask whatever it
    // covers this frame: nothing is no UI, everything a full-screen menu. A few
    // invalid pixels (at most 1%, such as additive glow in a UI layer) do not
    // disqualify a trusted channel. While a trusted dedicated UI channel (0 UI
    // alpha, 1 UI color) is offered, presented alpha (2 Backbuffer, 3 current)
    // never decides: when that channel fails, this frame has no alpha mask
    // rather than a scene-wide one.
    // A layer without alpha is no layer (game3d_ui_detection_contract.h,
    // layer_without_alpha): an offscreen UI layer copy with no alpha anywhere
    // but color on more than 1% of pixels (Stellar Blade's SDR scene image in
    // its cleared UI target) is set aside. It neither decides, blocks presented
    // alpha, opens the layer route nor counts as an overlay. Tagged UI colors
    // keep blocking: their failure must not let presented alpha flatten the
    // scene. Beside a set-aside layer, presented alpha decides only below the
    // full-frame bound, so a channel trusted in another output mode cannot
    // flatten the frame while the layer holds only glow.
    uint candidates = Sunshine_UICandidates;
    const bool aside = (Sunshine_UIDetectionFlags & SUNSHINE_UI_STORED_LATE_LAYER) && !coverage.y &&
        invalid.y * 100u > difference.w;
    if (aside) candidates &= ~2u;
    bool dedicated = false, trusted_decided = false;
    [unroll] for (uint trusted = 0u; trusted < 4u; ++trusted) {
        const bool offered = (candidates & Sunshine_UITrustedAlpha & (1u << trusted)) != 0u;
        if (!source && offered && !(dedicated && trusted >= 2u) && invalid[trusted] * 100u <= difference.w &&
            !(aside && trusted >= 2u && coverage[trusted] * 10u >= difference.w * 9u)) {
            source = trusted + 1u; covered = coverage[trusted];
            trusted_decided = true;
        }
        if (offered && trusted < 2u) dedicated = true;
    }
    // An untrusted channel must look selective. Empty and nearly full-scene
    // alpha are ambiguous: an opaque channel may carry no UI at all.
    [unroll] for (uint candidate = 0u; candidate < 4u; ++candidate) {
        if (!source && !(dedicated && candidate >= 2u) && (candidates & (1u << candidate)) && !invalid[candidate] &&
            coverage[candidate] && coverage[candidate] * 10u < difference.w * 9u) {
            source = candidate + 1u; covered = coverage[candidate];
        }
    }
    // Difference is changed-color support, not uniquely recovered opacity.
    // Require broad unchanged scene evidence as well as bounded total coverage.
    if (!source && (Sunshine_UICandidates & 16u) && !difference.y && difference.x &&
        difference.x * 4u < difference.w && difference.z * 100u >= difference.w * 75u && matching_tiles >= 128u) {
        source = 5u; covered = difference.x;
    }
    // UI covering the whole frame (menus, title screens) changes nearly every
    // pixel while the HUD-less image still shows a lit scene: keep it flat.
    // Only an exact pair qualifies; a mismatched pair can also differ
    // everywhere, and a black HUD-less image is no scene.
    if (!source && (Sunshine_UICandidates & 48u) == 48u && !difference.y &&
        difference.x * 100u >= difference.w * 98u && lit * 2u >= difference.w) {
        source = 6u; covered = difference.w;
    }
    // A hidden scene (docs/reshade-sbs.md, hidden-scene evidence). When no
    // source decided, an untrusted UIAlpha or offscreen UI layer (never a
    // tagged UI color) nearly opaque almost everywhere without an invalid
    // pixel opens the layer route, and a HUD-less image differing from the
    // frame almost everywhere the HUD-less route. Either is full-frame UI only
    // while the CPU holds that route's verdict that the presented frame lacks
    // the depth's edges (8 layer, 9 HUD-less). The evidence passes only
    // measure; this frame's own gate still has to be open.
    const uint pixels = difference.w;
    const uint untrusted = candidates & ~Sunshine_UITrustedAlpha;
    const bool layer_gate = !source &&
        (((untrusted & 1u) && !invalid.x && opaque.x * 100u >= pixels * SUNSHINE_UI_SCENE_OPAQUE_PERCENT) ||
         ((untrusted & 2u) && (Sunshine_UIDetectionFlags & SUNSHINE_UI_STORED_LATE_LAYER) &&
          !invalid.y && opaque.y * 100u >= pixels * SUNSHINE_UI_SCENE_OPAQUE_PERCENT));
    const bool hudless_gate = !source && (Sunshine_UICandidates & 16u) &&
        difference.x * 100u >= pixels * SUNSHINE_UI_SCENE_HUDLESS_CHANGED_PERCENT;
    if (layer_gate && (Sunshine_UIDetectionFlags & SUNSHINE_UI_PER_FRAME_SCENE_HOLD)) {
        source = 8u; covered = pixels;
    } else if (hudless_gate && (Sunshine_UIDetectionFlags & SUNSHINE_UI_PER_FRAME_SCENE_HOLD_HUDLESS)) {
        source = 9u; covered = pixels;
    }
    // Exact per-session counters (game3d_ui_counters.h): this detection frame,
    // its decision and, without one, the first reason that applies. Counting
    // only; nothing on the GPU reads them.
    bool trusted_invalid = false, ambiguous = false;
    [unroll] for (uint channel = 0u; channel < 4u; ++channel) {
        const bool usable = (candidates & (1u << channel)) != 0u && !(dedicated && channel >= 2u);
        const bool channel_trusted = (Sunshine_UITrustedAlpha & (1u << channel)) != 0u;
        trusted_invalid = trusted_invalid || (usable && channel_trusted && invalid[channel] * 100u > pixels);
        ambiguous = ambiguous || (usable && !channel_trusted && !invalid[channel] &&
            (!coverage[channel] || coverage[channel] * 10u >= pixels * 9u));
    }
    uint none_reason = SUNSHINE_UI_NONE_OTHER;
    if (layer_gate || hudless_gate) none_reason = SUNSHINE_UI_NONE_GATE_NO_HOLD;
    else if (trusted_invalid) none_reason = SUNSHINE_UI_NONE_TRUSTED_INVALID;
    else if (dedicated && (candidates & 12u)) none_reason = SUNSHINE_UI_NONE_PRESENTED_BLOCKED;
    else if (aside) none_reason = SUNSHINE_UI_NONE_LAYER_ASIDE;
    else if (Sunshine_UICandidates & 16u) none_reason = SUNSHINE_UI_NONE_DIFFERENCE_FAILED;
    else if (ambiguous) none_reason = SUNSHINE_UI_NONE_AMBIGUOUS;
    else if (!(candidates & 15u)) none_reason = SUNSHINE_UI_NONE_NO_CANDIDATE;
    InterlockedAdd(SunshineUICountersStore[uint2(SUNSHINE_UI_COUNTER_DETECTION_FRAMES, 0u)], 1u);
    InterlockedAdd(SunshineUICountersStore[uint2(SUNSHINE_UI_COUNTER_DECIDED + source, 0u)], 1u);
    if (!source) InterlockedAdd(SunshineUICountersStore[uint2(SUNSHINE_UI_COUNTER_NONE + none_reason, 0u)], 1u);
    if (!trusted_decided && (source == 3u || source == 4u ||
            (source == 2u && (Sunshine_UIDetectionFlags & SUNSHINE_UI_STORED_LATE_LAYER))))
        InterlockedAdd(SunshineUICountersStore[uint2(SUNSHINE_UI_COUNTER_UNTRUSTED_INFERRED, 0u)], 1u);
    if ((source == 5u || source == 9u) && (Sunshine_UICandidates & 48u) == 16u)
        InterlockedAdd(SunshineUICountersStore[uint2(SUNSHINE_UI_COUNTER_INEXACT_DIFFERENCE, 0u)], 1u);
    if (Sunshine_UIDetectionFlags & SUNSHINE_UI_PER_FRAME_DEPTH_NOT_CURRENT)
        InterlockedAdd(SunshineUICountersStore[uint2(SUNSHINE_UI_COUNTER_DEPTH_NOT_CURRENT, 0u)], 1u);
    if (trusted_decided && covered * 100u >= pixels * 99u && (Sunshine_UICandidates & 48u) == 48u && !difference.y &&
            difference.z * 2u >= pixels)
        InterlockedAdd(SunshineUICountersStore[uint2(SUNSHINE_UI_COUNTER_TRUSTED_FULL, 0u)], 1u);
    if ((source == 3u || source == 4u) && dedicated)
        InterlockedAdd(SunshineUICountersStore[uint2(SUNSHINE_UI_COUNTER_PRESENTED_OVER_DEDICATED, 0u)], 1u);
    if (source >= 1u && source <= 4u && covered * 100u >= pixels * 99u)
        InterlockedAdd(SunshineUICountersStore[uint2(SUNSHINE_UI_COUNTER_FULL_ALPHA, 0u)], 1u);
    SunshineAlphaCoverageStore[uint2(0,0)] = uint4(source, covered, difference.w, matching_tiles);
    // Diagnostic evidence for the CPU readback; nothing reads it on the GPU.
    SunshineAlphaCoverageStore[uint2(1,0)] = uint4(Sunshine_UICandidates, difference.x, difference.z, difference.y);
    SunshineAlphaCoverageStore[uint2(2,0)] = coverage;
    SunshineAlphaCoverageStore[uint2(3,0)] = invalid;
    SunshineAlphaCoverageStore[uint2(4,0)] = uint4(lit, Sunshine_UITrustedAlpha, opaque);
    // No scene evidence unless the evidence passes run after this one.
    SunshineAlphaCoverageStore[uint2(5,0)] = 0u;
    SunshineAlphaCoverageStore[uint2(6,0)] = 0u;
}
[numthreads(8, 8, 1)]
void SunshineUIDetectionMaskCS(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= BUFFER_WIDTH || id.y >= BUFFER_HEIGHT) return;
    uint source = SunshineUIDetectionSampler.Load(int3(0,0,0)).x;
    float mask = 0.0;
    int3 at = int3(id.xy, 0);
    // Full-frame UI: the whole frame pins.
    if (source == 6u || source == 8u || source == 9u) mask = 1.0;
    // Otherwise load only the selected candidate's raw mask (the same channel
    // SunshineUIDetectionAlpha reads), the offscreen UI layer included.
    else if (source == 1u) mask = SunshineUIDedicatedAlpha.Load(at).r;
    else if (source == 2u) mask = SunshineUIColorAlpha.Load(at).a;
    else if (source == 3u) mask = SunshineUIBackbufferAlpha.Load(at).a;
    else if (source == 4u) mask = SunshineSourceSampler.Load(at).a;
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
// measures whether the presented color, and a HUD-less image when one is
// offered, show the consumed depth's edges; it never decides a mask. The
// renderer dispatches these passes after detection on sample frames only, the
// CPU owns the verdict, and SunshineUIDetectionReduceCS acts on a held verdict
// while its gate is open. inspect_game3d_dump.py holds the same statistic as
// a CPU oracle.
//
// Output pixel x lies in cell column floor(x * 256 / width), row y in cell row
// floor(y * 144 / height). Each cell sums its pixels' perceptual luma and
// strength-1 parallax in fixed point, so its means and every comparison below
// are exact integers whatever order the pixels were added in. Three passes:
// cell sums into their own texture, per-block comparisons into the statistics
// rows below the detection tiles, and their sum into decision texels 5 and 6.
static const uint SunshineScenePartialRow = 64u;
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
// Cell sums {presented luma, HUD-less luma, parallax, pixels} in fixed point,
// one texel per cell of the scene cell texture (u6 here, t10 when compared).
// A group sums sixteen cells of one cell row with sixteen lanes per cell: lane
// i sums column i of its cell (and every sixteenth column after it), so
// neighbouring lanes read neighbouring pixels.
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
    const bool hudless = (Sunshine_UICandidates & 16u) != 0u;
    uint3 sums = 0u; // Parallax adds as two's complement bits.
    if (active) {
        [loop] for (uint y = first.y; y < last.y; ++y)
        [loop] for (uint x = first.x + thread.x; x < last.x; x += 16u) {
            sums.x += SunshineSceneLuma(SunshinePresentedColor.Load(int3(x, y, 0)).rgb);
            if (hudless) sums.y += SunshineSceneLuma(SunshineHUDless.Load(int3(x, y, 0)).rgb);
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
// Cell means, rounded half away from zero: luma of both images, parallax.
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
// {n, wins - losses presented, HUD-less, presented decided} go to statistics
// texel (group.x, SunshineScenePartialRow + group.y).
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
    if (active && !lane)
        SunshineAlphaCoverageStore[uint2(group.x, SunshineScenePartialRow + group.y)] =
            uint4(asuint(SunshineSceneTotals[0]), asuint(SunshineSceneTotals[1]), asuint(SunshineSceneTotals[2]),
                asuint(SunshineSceneTotals[3]));
}
// Sums the compare groups' partial sums: D = (wins - losses) / (4 n); ties
// count 0, so a black or flat image reads as hidden. Writes decision texels 5
// (presented, with its verdict and decided comparisons) and 6 (HUD-less, when
// offered).
[numthreads(16, 16, 1)]
void SunshineSceneEvidenceCS(uint3 thread : SV_GroupThreadID)
{
    const uint lane = thread.y * 16u + thread.x;
    const bool active = SunshineSceneDepthActive();
    if (lane < 4u) SunshineSceneTotals[lane] = 0;
    GroupMemoryBarrierWithGroupSync();
    if (active && thread.y < SUNSHINE_UI_SCENE_CELLS_Y / 16u) {
        int4 partial = asint(SunshineUIDetectionSampler.Load(int3(thread.x, SunshineScenePartialRow + thread.y, 0)));
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
    SunshineAlphaCoverageStore[uint2(6, 0)] = (Sunshine_UICandidates & 16u) ?
        uint4(uint(n), asuint(d.y), uint(valid) | 2u, 0u) : 0u;
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

float4 SunshinePreparePQPS(float4 position : SV_Position, float2 texcoord : TEXCOORD0) : SV_Target
{
    float4 color = SunshineSourceSampler.Load(int3(int2(position.xy), 0));
    color.rgb = SunshineDecodePQ(color.rgb);
    return color;
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
        for (uint x = chunk_start; x < chunk_end; ++x) {
            float weight = SunshineUIPinWeight(x, y);
            if (weight <= 0.0) continue;
            // Fully weighted UI has no slack whatever the field.
            float r = weight >= 1.0 ? 0.0 : SunshineUIPinSlackOf(SunshineHostFinalStore[int2(uint2(x, y))], weight, plane);
            float right = SunshineUIPinBound(r, chunk_end - 1u - x);
            if (slack.x < 0.0 || right <= reach.x) { anchors.x = (int)x + 1; slack.x = r; reach.x = right; }
            float left = SunshineUIPinBound(r, x - chunk_start);
            if (slack.y < 0.0 || left < reach.y) { anchors.y = (int)x - 1; slack.y = r; reach.y = left; }
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
    [loop]
    for (int scan_x = (int)chunk_end - 1; scan_x >= (int)chunk_start; --scan_x) {
        uint x = (uint)scan_x;
        float previous = 0.0, r_previous = -1.0;
        if (x > chunk_start) {
            previous = SunshineHostFinalStore[int2(uint2(x - 1u, y))];
            r_previous = SunshineUIPinSlackOf(previous, SunshineUIPinWeight(x - 1u, y), plane);
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
    [loop]
    for (uint x = chunk_start; x < chunk_end; ++x) {
        float w_next = x + 1u < chunk_end ? SunshineUIPinWeight(x + 1u, y) : 0.0;
        bool collar = w_previous > 0.0 || w_here > 0.0 || w_next > 0.0;
        // Next to fully weighted UI the backward pass left the plane itself.
        if (max(max(w_previous, w_here), w_next) >= 1.0) {
            anchor = (int)x; best = 0.0;
        } else if (collar || best >= 0.0) {
            float current = SunshineHostFinalStore[int2(uint2(x, y))];
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
        float4 color = SunshineLinearClamp.SampleLevel(SunshineLinearClampState, (float4(sourceUV, 0, 0)).xy, (float4(sourceUV, 0, 0)).w);
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
