// SPDX-License-Identifier: GPL-3.0-only
// Native GPU implementation of the current Sunshine Game 3D effect.
// The add-on owns input capture, resources, scheduling and export; all image work
// below remains on the GPU. No external FX shader or ReShade preset is needed.
// Ported from the frozen 2026-09-18 three-file Game 3D source. Geometry authority:
// docs/host-sbs.md and the Depth Coordinate V2 vertical/horizontal limit shaders.
// Optional source-alpha UI pinning reuses the horizontal pass after conditioning.
// Live UI selection uses bounded alpha-coverage observations; explicit replay
// retains its captured selection. Neither path adds color-layer blending.
// Nearest-covered-depth mode first resolves one global UI plane on this queue.
#define SUNSHINE_UI_NEAREST_PLANE 1
#define SUNSHINE_UI_FRONT_LIMIT_PLANE 1
#define SUNSHINE_UI_SHALLOW_FRONT_PLANE 1
#define SUNSHINE_UI_DISPLAY_FRACTION_PLANE 1
#define SUNSHINE_UI_CONFLICT_PROBE 1
#define SUNSHINE_UI_ABSOLUTE_LEVEL_PROBE 1
#define SUNSHINE_UI_ALPHA_COVERAGE 1
#define SUNSHINE_UI_MASK_CHANNEL 1
#define SUNSHINE_UI_AUTOMATIC_DETECTION 1
#define SUNSHINE_LINEAR_DISTANCE_DEPTH 1
// Limiter groups own eight adjacent lines; UI pinning is a separate pass.
#define SUNSHINE_LIMITER_LINE_GROUPS 8
// One pass renders both eyes straight into the side-by-side target.
#define SUNSHINE_PACKED_EYES 1
// UI pinning groups own eight adjacent rows.
#define SUNSHINE_UI_PIN_LINE_GROUPS 8
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
Texture2D<float> SunshineUIPlaneTilesSampler : register(t8);
Texture2D<float> SunshineUIPlaneResolvedSampler : register(t9);
Texture2D<uint4> SunshineUIDetectionSampler : register(t10);
Texture2D<float4> SunshineUIDedicatedAlpha : register(t11);
Texture2D<float4> SunshineUIColorAlpha : register(t12);
Texture2D<float4> SunshineUIBackbufferAlpha : register(t13);
Texture2D<float4> SunshineHUDless : register(t14);
cbuffer SunshineUIDetectionConstants : register(b2)
{
    uint Sunshine_UICandidates; // bit0..2 captured alpha, bit3 current alpha, bit4 paired HUDless, bit5 exact pair.
    float Sunshine_UIDifferenceThreshold;
    uint Sunshine_UITrustedAlpha; // Bit i: the game session trusts alpha candidate i as UI coverage.
    uint Sunshine_UIDetectionFlags; // bit0 UI color+alpha must be premultiplied, bit1 with HDR headroom.
};
RWTexture2D<float> SunshineHostCandidateStore : register(u0);
RWTexture2D<float> SunshineHostVerticalMajorantStore : register(u1);
RWTexture2D<float> SunshineHostVerticalConditionedStore : register(u2);
RWTexture2D<float> SunshineHostFinalStore : register(u3);
RWTexture2D<float> SunshineUIPlaneTilesStore : register(u4);
RWTexture2D<float> SunshineUIPlaneResolvedStore : register(u5);
RWTexture2D<uint4> SunshineAlphaCoverageStore : register(u6);
RWTexture2D<uint4> SunshineUIConflictStore : register(u7);
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
groupshared uint SunshineUIDetectionLit[256];
[numthreads(16, 16, 1)]
void SunshineUIDetectionTilesCS(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint2 first = group.xy * uint2(BUFFER_WIDTH, BUFFER_HEIGHT) / 16u;
    uint2 last = (group.xy + 1u) * uint2(BUFFER_WIDTH, BUFFER_HEIGHT) / 16u;
    uint4 coverage = 0u, invalid = 0u, difference = 0u;
    uint lit = 0u;
    [loop] for (uint y = first.y + thread.y; y < last.y; y += 16u)
    [loop] for (uint x = first.x + thread.x; x < last.x; x += 16u) {
        float4 a = SunshineUIDetectionAlpha(uint2(x,y));
        bool4 okay = isfinite(a) && a >= 0.0 && a <= 1.0;
        coverage += uint4(okay && a > 0.0);
        invalid += uint4(!okay);
        // An offscreen layer is UI only when blended over transparent black:
        // no color above its alpha (scRGB UI may be up to 10000 nits).
        if (Sunshine_UIDetectionFlags & 1u) {
            float4 layer = SunshineUIColorAlpha.Load(int3(x, y, 0));
            float headroom = Sunshine_UIDetectionFlags & 2u ? 125.0 : 1.0;
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
        lit += finite && max(max(abs(hudless.r), abs(hudless.g)), abs(hudless.b)) > Sunshine_UIDifferenceThreshold * 8.0 ? 1u : 0u;
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
        SunshineAlphaCoverageStore[group.xy + uint2(0,48)] = uint4(SunshineUIDetectionLit[0], 0u, 0u, 0u);
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
    SunshineUIDetectionLit[lane] = SunshineUIDetectionSampler.Load(int3(tile.x, tile.y + 48u, 0)).x;
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
    uint matching_tiles = SunshineUIDetectionMatching[0], lit = SunshineUIDetectionLit[0];
    uint source = 0u, covered = 0u;
    // A channel the game session trusts as UI coverage is the mask whatever it
    // covers this frame: nothing is no UI, everything a full-screen menu.
    [unroll] for (uint trusted = 0u; trusted < 4u; ++trusted) {
        if (!source && (Sunshine_UICandidates & Sunshine_UITrustedAlpha & (1u << trusted)) && !invalid[trusted]) {
            source = trusted + 1u; covered = coverage[trusted];
        }
    }
    // An untrusted channel must look selective. Empty and nearly full-scene
    // alpha are ambiguous: an opaque channel may carry no UI at all.
    [unroll] for (uint candidate = 0u; candidate < 4u; ++candidate) {
        if (!source && (Sunshine_UICandidates & (1u << candidate)) && !invalid[candidate] &&
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
    SunshineAlphaCoverageStore[uint2(0,0)] = uint4(source, covered, difference.w, matching_tiles);
    // Diagnostic evidence for the CPU readback; nothing reads it on the GPU.
    SunshineAlphaCoverageStore[uint2(1,0)] = uint4(Sunshine_UICandidates, difference.x, difference.z, difference.y);
    SunshineAlphaCoverageStore[uint2(2,0)] = coverage;
    SunshineAlphaCoverageStore[uint2(3,0)] = invalid;
    SunshineAlphaCoverageStore[uint2(4,0)] = uint4(lit, Sunshine_UITrustedAlpha, 0u, 0u);
}
[numthreads(8, 8, 1)]
void SunshineUIDetectionMaskCS(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= BUFFER_WIDTH || id.y >= BUFFER_HEIGHT) return;
    uint source = SunshineUIDetectionSampler.Load(int3(0,0,0)).x;
    float mask = 0.0;
    int3 at = int3(id.xy, 0);
    // Load only the selected candidate (the same channel SunshineUIDetectionAlpha reads).
    if (source == 1u) mask = SunshineUIDedicatedAlpha.Load(at).r;
    else if (source == 2u) mask = SunshineUIColorAlpha.Load(at).a;
    else if (source == 3u) mask = SunshineUIBackbufferAlpha.Load(at).a;
    else if (source == 4u) mask = SunshineSourceSampler.Load(at).a;
    else if (source == 6u) mask = 1.0;
    else if (source == 5u) {
        bool finite;
        mask = SunshineHUDlessDifference(id.xy, finite) > Sunshine_UIDifferenceThreshold && finite ? 1.0 : 0.0;
    }
    SunshineHostCandidateStore[id.xy] = mask;
}

// Fixed-size exact coverage observation. Every source texel participates; the
// small result is read asynchronously, independently of whether UI is enabled.
groupshared uint4 SunshineAlphaCoverageScratch[64];
[numthreads(8, 8, 1)]
void SunshineAlphaCoverageCS(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID)
{
    uint2 first = group_id.xy * uint2(BUFFER_WIDTH, BUFFER_HEIGHT) / 16u;
    uint2 last = (group_id.xy + 1u) * uint2(BUFFER_WIDTH, BUFFER_HEIGHT) / 16u;
    uint4 counts = 0u;
    [loop]
    for (uint y = first.y + thread_id.y; y < last.y; y += 8u) {
        [loop]
        for (uint x = first.x + thread_id.x; x < last.x; x += 8u) {
            float alpha = SunshineSelectedUIAlpha(uint2(x, y));
            bool finite = SunshineCameraFinite(alpha);
            counts.x += finite && alpha > 0.0 ? 1u : 0u;
            counts.y += 1u;
            counts.z += finite ? 0u : 1u;
        }
    }
    uint lane = thread_id.y * 8u + thread_id.x;
    SunshineAlphaCoverageScratch[lane] = counts;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint step = 32u; step != 0u; step >>= 1u) {
        if (lane < step) SunshineAlphaCoverageScratch[lane] += SunshineAlphaCoverageScratch[lane + step];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) SunshineAlphaCoverageStore[int2(group_id.xy)] = SunshineAlphaCoverageScratch[0];
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

// UI pinning runs as its own pass after the scene field. Each group owns
// eight adjacent rows split into eight chunks; a look-ahead cursor finds the
// nearest selected-UI texel on either side of every texel, so no row is held
// in group memory.
// {last UI index, first UI index} per chunk, and incoming {left, right}.
groupshared int2 SunshineUIPinEnds[8u][8u];
groupshared int2 SunshineUIPinCarries[8u][8u];

void SunshinePinSourceUI(uint x, uint y, int distance_pixels, float plane) {
    // No UI exists in this row: leave the original field bit-for-bit intact.
    if (distance_pixels >= BUFFER_WIDTH) return;
    // Bilinear color reaches one texel beyond a positive-alpha texel center.
    // Pin that support too, so a neighboring output cannot pull a faint copy
    // of an antialiased glyph into otherwise unmasked background.
    float bound = 0.5 * float(max(distance_pixels - 1, 0)) / float(BUFFER_WIDTH);
    float value = SunshineHostFinalStore[int2(uint2(x, y))];
    SunshineHostFinalStore[int2(uint2(x, y))] = distance_pixels <= 1 ? plane : clamp(value, plane - bound, plane + bound);
}

// The first selected-UI texel in [x, end), or end when there is none.
uint SunshineNextSourceUI(uint x, uint end, uint y) {
    [loop]
    while (x < end && !SunshineSourceUI(x, y)) ++x;
    return x;
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
    int first_ui = BUFFER_WIDTH, last_ui = -1;
    if (active) {
        [loop]
        for (uint x = chunk_start; x < chunk_end; ++x) {
            if (SunshineSourceUI(x, y)) {
                first_ui = min(first_ui, (int)x);
                last_ui = (int)x;
            }
        }
    }
    SunshineUIPinEnds[lane][row] = int2(last_ui, first_ui);
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        int left = -1;
        [unroll]
        for (uint chunk = 0u; chunk < 8u; ++chunk) {
            SunshineUIPinCarries[chunk][row].x = left;
            left = max(left, SunshineUIPinEnds[chunk][row].x);
        }
        int right = BUFFER_WIDTH;
        [unroll]
        for (int chunk = 7; chunk >= 0; --chunk) {
            SunshineUIPinCarries[(uint)chunk][row].y = right;
            right = min(right, SunshineUIPinEnds[(uint)chunk][row].y);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    int2 carry = SunshineUIPinCarries[lane][row];
    // A row without UI is left bit-for-bit intact.
    if (!active || (carry.x < 0 && carry.y >= BUFFER_WIDTH && first_ui >= BUFFER_WIDTH)) return;
    float plane = SunshineUIPlaneParallax();
    int left = carry.x;
    int right = first_ui < BUFFER_WIDTH ? first_ui : carry.y;
    [loop]
    for (uint x = chunk_start; x < chunk_end; ++x) {
        if ((int)x == right) {
            left = right;
            uint next = SunshineNextSourceUI(x + 1u, chunk_end, y);
            right = next < chunk_end ? (int)next : carry.y;
        }
        int distance_left = left >= 0 ? (int)x - left : BUFFER_WIDTH;
        int distance_right = right < BUFFER_WIDTH ? right - (int)x : BUFFER_WIDTH;
        SunshinePinSourceUI(x, y, min(distance_left, distance_right), plane);
    }
}

// Exact tile counts from the selected mask and the unprotected, conditioned
// scene. Each tile writes two uint4 rows: coverage, invalid, five conflict
// counts, and pixel count. The absolute candidate planes remain within half
// the current scene cap. All 16x16 tiles fit in an 8 KiB statistics texture.
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
            float alpha = SunshineSelectedUIAlpha(uint2(x, y));
            b.w += 1u;
            if (!SunshineCameraFinite(alpha) || alpha <= 0.0) continue;
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
