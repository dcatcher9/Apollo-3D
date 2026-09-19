// SPDX-License-Identifier: GPL-3.0-only
// Native GPU implementation of the current Sunshine Game 3D effect.
// The add-on owns input capture, resources, scheduling and export; all image work
// below remains on the GPU. No external FX shader or ReShade preset is needed.
// Ported from the frozen 2026-09-18 three-file Game 3D source. Geometry authority:
// docs/host-sbs.md and the Depth Coordinate V2 vertical/horizontal limit shaders.
// Keep this pass graph and its floating-point precision unchanged during migration.
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
    float Sunshine_Reserved : packoffset(c1.w);
    float2 Sunshine_CameraProjection : packoffset(c2.x);
    float2 Sunshine_CameraRawDepthRange : packoffset(c2.z);
    float2 Sunshine_CameraConvergence : packoffset(c3.x);
    float2 Sunshine_DepthJitter : packoffset(c3.z);
    float4 Sunshine_CameraDepthRect : packoffset(c4);
};

Texture2D<float4> SunshineSourceSampler : register(t0);
Texture2D<float> DepthBuffer : register(t1);
Texture2D<float4> SunshineLinearClamp : register(t2);
Texture2D<float> SunshineHostCandidateSampler : register(t3);
Texture2D<float> SunshineHostVerticalConditionedSampler : register(t4);
Texture2D<float> SunshineHostFinalSampler : register(t5);
Texture2D<float4> SunshineEyeLeftSampler : register(t6);
Texture2D<float4> SunshineEyeRightSampler : register(t7);
RWTexture2D<float> SunshineHostCandidateStore : register(u0);
RWTexture2D<float> SunshineHostVerticalMajorantStore : register(u1);
RWTexture2D<float> SunshineHostVerticalConditionedStore : register(u2);
RWTexture2D<float> SunshineHostFinalStore : register(u3);
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
    float4 rect = Sunshine_CameraDepthRect;
    if (!SunshineCameraFinite(rect.x) || !SunshineCameraFinite(rect.y) ||
        !SunshineCameraFinite(rect.z) || !SunshineCameraFinite(rect.w) ||
        any(rect.xy < 0.0) || any(rect.zw <= 0.0) || any(rect.xy + rect.zw > 1.000001))
        return false;
    float A = Sunshine_CameraProjection.x, inverseB = Sunshine_CameraProjection.y;
    if (Sunshine_CameraCoordinateBasis != 0 && Sunshine_CameraCoordinateBasis != 1)
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
float SunshineCameraDepth(float rawDepth)
{
    // Camera basis: q=(raw-A)/B. Raw basis: t=raw or 1-raw. Both share the
    // same rational visualization only; geometry uses inverse depth directly.
    float inverseDistance = (rawDepth - Sunshine_CameraProjection.x) * Sunshine_CameraProjection.y;
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

bool SunshineHostWarpActive()
{
    return Sunshine_DepthReady && SunshineCameraActive() &&
        SunshineCameraFinite(Depth_Adjustment) && Depth_Adjustment > 0 && SunshineAutomaticStrengthBlend() > 0;
}

[numthreads(8, 8, 1)]
void SunshineHostCandidateCS(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= BUFFER_WIDTH || id.y >= BUFFER_HEIGHT) return;
    float parallax = 0.0;
    if (SunshineHostWarpActive())
    {
        float2 coordinate = (float2(id.xy) + 0.5) / float2(BUFFER_WIDTH, BUFFER_HEIGHT);
        float2 uv = SunshineCameraDepthCoordinates(coordinate, SunshineDepthAllocationSize());
        float raw = DepthBuffer.SampleLevel(SunshinePointBorder, (float4(uv, 0, 0)).xy, (float4(uv, 0, 0)).w).x;
        precise float q = (raw - Sunshine_CameraProjection.x) * Sunshine_CameraProjection.y;
        precise float strength = clamp(Depth_Adjustment, 0.0, 100.0) * 0.01 * SunshineAutomaticStrengthBlend();
        precise float displacement = Sunshine_CameraConvergence.x * Sunshine_CameraDepthScale *
            (Sunshine_CameraConvergence.y - q) * strength;
        if (SunshineCameraFinite(displacement))
            parallax = -clamp(displacement, -1.5, 2.5) * (float(BUFFER_HEIGHT) * rcp(2160.0) * 100.0) / BUFFER_WIDTH;
    }
    // Same signed source-U safety domain as the current Host conditioner.
    SunshineHostCandidateStore[int2(id.xy)] = clamp(parallax, -SunshineHostContainer, SunshineHostContainer);
}

// Mechanical ReShade binding translation of depth_coordinate_v2_vertical_limit_cs.hlsl
// Original SHA256: 68392c96466d4b3b165dac71740adbec05da7ede9d900d9569b0efa4ebcb2841
groupshared int V_ForwardUpperQ30[BUFFER_HEIGHT];
groupshared int V_ForwardLowerQ30[BUFFER_HEIGHT];
// {forward upper end, forward lower end, backward upper end, backward lower end}
groupshared int4 V_LocalEndsQ30[32u];
// {incoming forward upper, incoming forward lower, incoming backward upper, incoming backward lower}
groupshared int4 V_ChunkCarriesQ30[32u];

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

float V_share_vertical_envelopes(float upper, float lower) {
    // Keep the persisted float32 evaluation order and forbid contraction/reassociation.
    precise float majorant_share = 0.75;
    precise float minorant_share = 1.0f - majorant_share;
    precise float majorant_term = majorant_share * upper;
    precise float minorant_term = minorant_share * lower;
    precise float conditioned = majorant_term + minorant_term;
    return conditioned;
}

[numthreads(32, 1, 1)]
void SunshineHostVerticalCS(
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID) {
    uint x = group_id.x;
    if (x >= BUFFER_WIDTH) {
        return;
    }

    float max_step = 2.0 / float(BUFFER_WIDTH);
    uint lane = group_thread_id.x;

    // Preserve the tiny diagnostic/unit-test path exactly and avoid empty chunks.
    if (BUFFER_HEIGHT <= 32u) {
        if (lane == 0u) {
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

    uint chunk_start = lane * BUFFER_HEIGHT / 32u;
    uint chunk_end = (lane + 1u) * BUFFER_HEIGHT / 32u;
    float candidate = SunshineHostCandidateSampler.Load(int3(int2(uint2(x, chunk_start)), 0));
    int forward_upper_q30 = V_V2LimitUpperQ30(candidate);
    int forward_lower_q30 = V_V2LimitLowerQ30(candidate);
    [loop]
    for (uint local_forward_y = chunk_start + 1u;
         local_forward_y < chunk_end;
         ++local_forward_y) {
        candidate = SunshineHostCandidateSampler.Load(int3(int2(uint2(x, local_forward_y)), 0));
        forward_upper_q30 = max(
            V_V2LimitUpperQ30(candidate),
            forward_upper_q30 - max_step_q30);
        forward_lower_q30 = min(
            V_V2LimitLowerQ30(candidate),
            forward_lower_q30 + max_step_q30);
    }
    candidate = SunshineHostCandidateSampler.Load(int3(int2(uint2(x, chunk_end - 1u)), 0));
    int backward_upper_q30 = V_V2LimitUpperQ30(candidate);
    int backward_lower_q30 = V_V2LimitLowerQ30(candidate);
    [loop]
    for (int local_backward_y = (int)chunk_end - 2;
         local_backward_y >= (int)chunk_start;
         --local_backward_y) {
        candidate = SunshineHostCandidateSampler.Load(int3(int2(uint2(x, (uint)local_backward_y)), 0));
        backward_upper_q30 = max(
            V_V2LimitUpperQ30(candidate),
            backward_upper_q30 - max_step_q30);
        backward_lower_q30 = min(
            V_V2LimitLowerQ30(candidate),
            backward_lower_q30 + max_step_q30);
    }
    V_LocalEndsQ30[lane] = int4(
        forward_upper_q30,
        forward_lower_q30,
        backward_upper_q30,
        backward_lower_q30);
    GroupMemoryBarrierWithGroupSync();

    if (lane == 0u) {
        forward_upper_q30 = V_LocalEndsQ30[0u].x;
        forward_lower_q30 = V_LocalEndsQ30[0u].y;
        [unroll]
        for (uint forward_chunk = 1u;
             forward_chunk < 32u;
             ++forward_chunk) {
            V_ChunkCarriesQ30[forward_chunk].xy = int2(
                forward_upper_q30,
                forward_lower_q30);
            uint forward_start = forward_chunk * BUFFER_HEIGHT / 32u;
            uint forward_end = (forward_chunk + 1u) * BUFFER_HEIGHT / 32u;
            int forward_span_q30 = V_V2LimitDecayQ30(
                max_step_q30,
                forward_end - forward_start,
                max_decay_q30);
            forward_upper_q30 = max(
                V_LocalEndsQ30[forward_chunk].x,
                forward_upper_q30 - forward_span_q30);
            forward_lower_q30 = min(
                V_LocalEndsQ30[forward_chunk].y,
                forward_lower_q30 + forward_span_q30);
        }

        backward_upper_q30 = V_LocalEndsQ30[32u - 1u].z;
        backward_lower_q30 = V_LocalEndsQ30[32u - 1u].w;
        [unroll]
        for (int backward_chunk = (int)32u - 2;
             backward_chunk >= 0;
             --backward_chunk) {
            V_ChunkCarriesQ30[(uint)backward_chunk].zw = int2(
                backward_upper_q30,
                backward_lower_q30);
            uint backward_start = (uint)backward_chunk * BUFFER_HEIGHT / 32u;
            uint backward_end = ((uint)backward_chunk + 1u) * BUFFER_HEIGHT / 32u;
            int backward_span_q30 = V_V2LimitDecayQ30(
                max_step_q30,
                backward_end - backward_start,
                max_decay_q30);
            backward_upper_q30 = max(
                V_LocalEndsQ30[(uint)backward_chunk].z,
                backward_upper_q30 - backward_span_q30);
            backward_lower_q30 = min(
                V_LocalEndsQ30[(uint)backward_chunk].w,
                backward_lower_q30 + backward_span_q30);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    candidate = SunshineHostCandidateSampler.Load(int3(int2(uint2(x, chunk_start)), 0));
    forward_upper_q30 = V_V2LimitUpperQ30(candidate);
    forward_lower_q30 = V_V2LimitLowerQ30(candidate);
    if (lane != 0u) {
        forward_upper_q30 = max(
            forward_upper_q30,
            V_ChunkCarriesQ30[lane].x - max_step_q30);
        forward_lower_q30 = min(
            forward_lower_q30,
            V_ChunkCarriesQ30[lane].y + max_step_q30);
    }
    V_ForwardUpperQ30[chunk_start] = forward_upper_q30;
    V_ForwardLowerQ30[chunk_start] = forward_lower_q30;
    [loop]
    for (uint replay_forward_y = chunk_start + 1u;
         replay_forward_y < chunk_end;
         ++replay_forward_y) {
        candidate = SunshineHostCandidateSampler.Load(int3(int2(uint2(x, replay_forward_y)), 0));
        forward_upper_q30 = max(
            V_V2LimitUpperQ30(candidate),
            forward_upper_q30 - max_step_q30);
        forward_lower_q30 = min(
            V_V2LimitLowerQ30(candidate),
            forward_lower_q30 + max_step_q30);
        V_ForwardUpperQ30[replay_forward_y] = forward_upper_q30;
        V_ForwardLowerQ30[replay_forward_y] = forward_lower_q30;
    }
    GroupMemoryBarrierWithGroupSync();

    candidate = SunshineHostCandidateSampler.Load(int3(int2(uint2(x, chunk_end - 1u)), 0));
    backward_upper_q30 = V_V2LimitUpperQ30(candidate);
    backward_lower_q30 = V_V2LimitLowerQ30(candidate);
    if (lane + 1u != 32u) {
        backward_upper_q30 = max(
            backward_upper_q30,
            V_ChunkCarriesQ30[lane].z - max_step_q30);
        backward_lower_q30 = min(
            backward_lower_q30,
            V_ChunkCarriesQ30[lane].w + max_step_q30);
    }
    uint write_y = chunk_end - 1u;
    int final_upper_q30 = max(V_ForwardUpperQ30[write_y], backward_upper_q30);
    int final_lower_q30 = min(V_ForwardLowerQ30[write_y], backward_lower_q30);
    float final_upper = V_V2LimitFromQ30(final_upper_q30);
    float final_lower = V_V2LimitFromQ30(final_lower_q30);
    SunshineHostVerticalMajorantStore[int2(uint2(x, write_y))] = final_upper;
    SunshineHostVerticalConditionedStore[int2(uint2(x, write_y))] = clamp(
        V_share_vertical_envelopes(final_upper, final_lower),
        final_lower,
        final_upper);
    [loop]
    for (int scan_y = (int)chunk_end - 2; scan_y >= (int)chunk_start; --scan_y) {
        write_y = (uint)scan_y;
        candidate = SunshineHostCandidateSampler.Load(int3(int2(uint2(x, write_y)), 0));
        backward_upper_q30 = max(
            V_V2LimitUpperQ30(candidate),
            backward_upper_q30 - max_step_q30);
        backward_lower_q30 = min(
            V_V2LimitLowerQ30(candidate),
            backward_lower_q30 + max_step_q30);
        final_upper_q30 = max(V_ForwardUpperQ30[write_y], backward_upper_q30);
        final_lower_q30 = min(V_ForwardLowerQ30[write_y], backward_lower_q30);
        final_upper = V_V2LimitFromQ30(final_upper_q30);
        final_lower = V_V2LimitFromQ30(final_lower_q30);
        SunshineHostVerticalMajorantStore[int2(uint2(x, write_y))] = final_upper;
        SunshineHostVerticalConditionedStore[int2(uint2(x, write_y))] = clamp(
            V_share_vertical_envelopes(final_upper, final_lower),
            final_lower,
            final_upper);
    }
}

// Mechanical ReShade binding translation of depth_coordinate_v2_limit_cs.hlsl
// Original SHA256: d039457f8d5db27a801523dc220ab8610657185d28c399a93690e2f1c187820e
groupshared int H_LineCandidateQ30[BUFFER_WIDTH];
groupshared int H_ForwardMajorantQ30[BUFFER_WIDTH];
// {forward end, backward end}
groupshared int2 H_LocalEndsQ30[32u];
// {incoming forward value, incoming backward value}
groupshared int2 H_ChunkCarriesQ30[32u];

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

[numthreads(32, 1, 1)]
void SunshineHostHorizontalCS(
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID) {
    uint y = group_id.x;
    if (y >= BUFFER_HEIGHT) {
        return;
    }

    float max_step = 0.5 / float(BUFFER_WIDTH);
    uint lane = group_thread_id.x;

    // Preserve the tiny diagnostic/unit-test path exactly and avoid empty chunks.
    if (BUFFER_WIDTH <= 32u) {
        if (lane == 0u) {
            float value = SunshineHostVerticalConditionedSampler.Load(int3(int2(uint2(0u, y)), 0));
            SunshineHostFinalStore[int2(uint2(0u, y))] = value;
            [loop]
            for (uint serial_x = 1u; serial_x < BUFFER_WIDTH; ++serial_x) {
                value = max(SunshineHostVerticalConditionedSampler.Load(int3(int2(uint2(serial_x, y)), 0)), value - max_step);
                SunshineHostFinalStore[int2(uint2(serial_x, y))] = value;
            }
            AllMemoryBarrier();
            value = SunshineHostFinalStore[int2(uint2(BUFFER_WIDTH - 1u, y))];
            [loop]
            for (int serial_back_x = (int)BUFFER_WIDTH - 2;
                 serial_back_x >= 0;
                 --serial_back_x) {
                const uint2 position = uint2((uint)serial_back_x, y);
                value = max(SunshineHostFinalStore[int2(position)], value - max_step);
                SunshineHostFinalStore[int2(position)] = value;
            }
        }
        return;
    }

    int max_decay_q30 = 2 * SunshineHostContainerQ;
    uint content_width = BUFFER_WIDTH;
    int max_step_q30 = H_V2LimitStepQ30(content_width, max_decay_q30);
    [loop]
    for (uint load_x = lane; load_x < BUFFER_WIDTH; load_x += 32u) {
        H_LineCandidateQ30[load_x] = H_V2LimitUpperQ30(SunshineHostVerticalConditionedSampler.Load(int3(int2(uint2(load_x, y)), 0)));
    }
    GroupMemoryBarrierWithGroupSync();

    uint chunk_start = lane * BUFFER_WIDTH / 32u;
    uint chunk_end = (lane + 1u) * BUFFER_WIDTH / 32u;
    int forward_q30 = H_LineCandidateQ30[chunk_start];
    [loop]
    for (uint local_forward_x = chunk_start + 1u;
         local_forward_x < chunk_end;
         ++local_forward_x) {
        forward_q30 = max(
            H_LineCandidateQ30[local_forward_x],
            forward_q30 - max_step_q30);
    }
    int backward_q30 = H_LineCandidateQ30[chunk_end - 1u];
    [loop]
    for (int local_backward_x = (int)chunk_end - 2;
         local_backward_x >= (int)chunk_start;
         --local_backward_x) {
        backward_q30 = max(
            H_LineCandidateQ30[(uint)local_backward_x],
            backward_q30 - max_step_q30);
    }
    H_LocalEndsQ30[lane] = int2(forward_q30, backward_q30);
    GroupMemoryBarrierWithGroupSync();

    if (lane == 0u) {
        forward_q30 = H_LocalEndsQ30[0u].x;
        [unroll]
        for (uint forward_chunk = 1u;
             forward_chunk < 32u;
             ++forward_chunk) {
            H_ChunkCarriesQ30[forward_chunk].x = forward_q30;
            uint forward_start = forward_chunk * BUFFER_WIDTH / 32u;
            uint forward_end = (forward_chunk + 1u) * BUFFER_WIDTH / 32u;
            int forward_span_q30 = H_V2LimitDecayQ30(
                max_step_q30,
                forward_end - forward_start,
                max_decay_q30);
            forward_q30 = max(
                H_LocalEndsQ30[forward_chunk].x,
                forward_q30 - forward_span_q30);
        }

        backward_q30 = H_LocalEndsQ30[32u - 1u].y;
        [unroll]
        for (int backward_chunk = (int)32u - 2;
             backward_chunk >= 0;
             --backward_chunk) {
            H_ChunkCarriesQ30[(uint)backward_chunk].y = backward_q30;
            uint backward_start = (uint)backward_chunk * BUFFER_WIDTH / 32u;
            uint backward_end = ((uint)backward_chunk + 1u) * BUFFER_WIDTH / 32u;
            int backward_span_q30 = H_V2LimitDecayQ30(
                max_step_q30,
                backward_end - backward_start,
                max_decay_q30);
            backward_q30 = max(
                H_LocalEndsQ30[(uint)backward_chunk].y,
                backward_q30 - backward_span_q30);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    forward_q30 = H_LineCandidateQ30[chunk_start];
    if (lane != 0u) {
        forward_q30 = max(
            forward_q30,
            H_ChunkCarriesQ30[lane].x - max_step_q30);
    }
    H_ForwardMajorantQ30[chunk_start] = forward_q30;
    [loop]
    for (uint replay_forward_x = chunk_start + 1u;
         replay_forward_x < chunk_end;
         ++replay_forward_x) {
        forward_q30 = max(
            H_LineCandidateQ30[replay_forward_x],
            forward_q30 - max_step_q30);
        H_ForwardMajorantQ30[replay_forward_x] = forward_q30;
    }
    GroupMemoryBarrierWithGroupSync();

    backward_q30 = H_LineCandidateQ30[chunk_end - 1u];
    if (lane + 1u != 32u) {
        backward_q30 = max(
            backward_q30,
            H_ChunkCarriesQ30[lane].y - max_step_q30);
    }
    uint write_x = chunk_end - 1u;
    int final_q30 = max(H_ForwardMajorantQ30[write_x], backward_q30);
    SunshineHostFinalStore[int2(uint2(write_x, y))] = H_V2LimitFromQ30(final_q30);
    [loop]
    for (int scan_x = (int)chunk_end - 2; scan_x >= (int)chunk_start; --scan_x) {
        write_x = (uint)scan_x;
        backward_q30 = max(
            H_LineCandidateQ30[write_x],
            backward_q30 - max_step_q30);
        final_q30 = max(H_ForwardMajorantQ30[write_x], backward_q30);
        SunshineHostFinalStore[int2(uint2(write_x, y))] = H_V2LimitFromQ30(final_q30);
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
void SunshineRenderEyesPS(float4 position : SV_Position, float2 texcoord : TEXCOORD0,
    out float4 left : SV_Target0, out float4 right : SV_Target1)
{
    float2 eyeUV = position.xy / float2(BUFFER_WIDTH, BUFFER_HEIGHT);
    left = SunshineRenderEye(eyeUV, false);
    right = SunshineRenderEye(eyeUV, true);
}
float4 SunshinePackEyesPS(float4 position : SV_Position, float2 texcoord : TEXCOORD0) : SV_Target
{
    // Derive the same pixel center in either half, avoiding packed-UV rounding
    // and an extra bilinear resample (visible on 4K edges).
    uint packedX = (uint)position.x;
    float2 eyeUV = float2((packedX % BUFFER_WIDTH) + 0.5, position.y) /
        float2(BUFFER_WIDTH, BUFFER_HEIGHT);
    // Missing depth and zero strength export the current source, never stale eyes.
    if (SunshineAutomaticMono())
        return SunshineNativeSource(eyeUV);
    int2 eyePixel = int2(eyeUV * float2(BUFFER_WIDTH, BUFFER_HEIGHT));
    float4 color;
    if (packedX < BUFFER_WIDTH)
        color = SunshineEyeLeftSampler.Load(int3(eyePixel, 0));
    else
        color = SunshineEyeRightSampler.Load(int3(eyePixel, 0));
#if BUFFER_COLOR_SPACE == 1
    // Preserve the former FP16 intermediate before RGB10A2 packing.
    return f16tof32(f32tof16(color));
#else
    // The HDR export itself rounds to the same FP16 representation.
    return color;
#endif
}
