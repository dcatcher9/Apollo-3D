// Production Host SBS renderer for the authenticated Depth Coordinate V2 final field.
//
// The producer supplies a signed, one-eye displacement in source-U units whose horizontal slope
// is strictly below one. Each eye map therefore has one inverse; a short fixed-point solve is the
// complete visibility rule. Invalid or unauthenticated state renders the current color frame
// through identity so a producer fault can never expose stale pixels or legacy geometry.

Texture2D<float4> SourceColor : register(t0);
Texture2D<float> FinalParallax : register(t1);
StructuredBuffer<float4> ParallaxState : register(t2);
Texture2D<float> CandidateParallax : register(t3);
SamplerState LinearSampler : register(s0);

cbuffer HostSbsV2Geometry : register(b2) {
    float content_scale_x;
    float content_scale_y;
    float2 geometry_reserved;
};

#include "include/depth_coordinate_v2_contract.generated.hlsl"

struct PS_INPUT {
    float4 Pos : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

bool Finite(float value) {
    return !isnan(value) && !isinf(value);
}

bool CameraInitialized(float4 active, float4 control, float4 shoulder) {
    uint revision = asuint(V2_STATE_CALIBRATION_REVISION(control));
    uint integrity = 0u;
    integrity = (integrity ^ asuint(V2_STATE_CENTER(active))) * 16777619u;
    integrity = (integrity ^ asuint(V2_STATE_INVERSE_SCALE(active))) * 16777619u;
    integrity = (integrity ^ revision) * 16777619u;
    return asuint(V2_STATE_CONTRACT_TAG_BITS(control)) == V2_CONTRACT_TAG &&
        Finite(V2_STATE_CENTER(active)) &&
        Finite(V2_STATE_INVERSE_SCALE(active)) &&
        Finite(V2_STATE_CONVERGENCE_CURVE(active)) &&
        V2_STATE_INVERSE_SCALE(active) > 0.0f &&
        revision > 0u && revision != 0xffffffffu &&
        asuint(V2_STATE_CAMERA_CENTER_INTEGRITY_BITS(shoulder)) == integrity;
}

bool WarpAvailable() {
    float4 active = ParallaxState[V2_STATE_VECTOR_CENTER];
    float4 control = ParallaxState[V2_STATE_VECTOR_CALIBRATION_REVISION];
    float4 shoulder = ParallaxState[V2_STATE_VECTOR_LATCHED_NEAR_TAIL_COVERAGE];
    return V2_STATE_FRAME_VALID(control) > 0.5f &&
        CameraInitialized(active, control, shoulder);
}

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

float SampleParallax(float source_x, float source_y) {
    return FinalParallax.SampleLevel(LinearSampler, float2(source_x, source_y), 0);
}

// Experimental collar defocus selected for the C75 headset trial. Values are measured in current
// source-color pixels: this is independent of the smaller depth-grid resolution, but deliberately
// is not claimed to preserve the same angular footprint when the stream resolution changes. Only
// positive conditioner deviation is eligible; negative deviation belongs to compressed foreground
// rows and must keep the original sharp color sample.
static const float COLLAR_DEFOCUS_ONSET_SOURCE_PX = 4.0f;
static const float COLLAR_DEFOCUS_FULL_SOURCE_PX = 20.0f;
static const float COLLAR_DEFOCUS_MAX_SIGMA_SOURCE_PX = 6.0f;

float4 SampleSourceColor(float2 source_uv) {
    return SourceColor.SampleLevel(LinearSampler, saturate(source_uv), 0);
}

float4 SampleWithCollarDefocus(float2 source_uv) {
    uint source_width;
    uint source_height;
    SourceColor.GetDimensions(source_width, source_height);
    if (source_width == 0u || source_height == 0u) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float final_parallax = FinalParallax.SampleLevel(LinearSampler, source_uv, 0);
    float candidate_parallax = CandidateParallax.SampleLevel(LinearSampler, source_uv, 0);
    float positive_deviation_source_px =
        max(final_parallax - candidate_parallax, 0.0f) * (float)source_width;

    // Preserve the previous one-tap renderer bit-for-bit outside the measured collar support.
    [branch]
    if (positive_deviation_source_px <= COLLAR_DEFOCUS_ONSET_SOURCE_PX) {
        return SourceColor.SampleLevel(LinearSampler, source_uv, 0);
    }

    float response = smoothstep(
        COLLAR_DEFOCUS_ONSET_SOURCE_PX,
        COLLAR_DEFOCUS_FULL_SOURCE_PX,
        positive_deviation_source_px
    );

    // Use the smallest separable binomial kernel that preserves the trial's sigma-6 second
    // moment. A [1,2,1]/4 kernel has variance radius^2/2, so its tap radius is sqrt(2)*sigma.
    // The denser 5x5 trial was visually indistinguishable on the crown witness while requiring
    // 25 rather than 9 color reads; retain the cheaper policy until headset evidence justifies a
    // larger footprint. The same source-space taps and weights are used for both eyes. Samples
    // remain in the existing render color space (including linear scRGB HDR); do not clamp,
    // tone-map, or gamma-convert here.
    static const float SQRT_TWO = 1.4142135623730951f;
    float2 offset = (SQRT_TWO * COLLAR_DEFOCUS_MAX_SIGMA_SOURCE_PX) /
        float2(source_width, source_height);
    const float binomial_weights[3] = {1.0f, 2.0f, 1.0f};
    float4 center = SampleSourceColor(source_uv);
    float4 blurred = (binomial_weights[1] * binomial_weights[1]) * center;
    [unroll]
    for (int tap_y = -1; tap_y <= 1; ++tap_y) {
        [unroll]
        for (int tap_x = -1; tap_x <= 1; ++tap_x) {
            if (tap_x == 0 && tap_y == 0) {
                continue;
            }
            float weight = binomial_weights[tap_x + 1] * binomial_weights[tap_y + 1];
            blurred += weight * SampleSourceColor(
                source_uv + float2((float)tap_x, (float)tap_y) * offset
            );
        }
    }
    blurred *= (1.0f / 16.0f);
    return lerp(center, blurred, response);
}

float2 Reproject(float2 destination_uv, float eye_sign) {
    // out(x) = x - eye_sign * parallax(x). The authenticated field is contractive, so this
    // iteration converges to its unique inverse without ownership search or hole filling.
    float source_x = destination_uv.x;
    [unroll]
    for (int iteration = 0; iteration < 12; ++iteration) {
        source_x = destination_uv.x + eye_sign *
            SampleParallax(source_x, destination_uv.y);
    }
    return float2(source_x, destination_uv.y);
}

bool PackedToSource(float2 packed_uv, out float2 source_uv, out float eye_sign) {
    bool right_eye = packed_uv.x > 0.5f;
    eye_sign = right_eye ? 1.0f : -1.0f;
    float2 eye_uv = packed_uv;
    eye_uv.x = right_eye ? (packed_uv.x - 0.5f) * 2.0f : packed_uv.x * 2.0f;
    return ContentToSourceUV(eye_uv, source_uv);
}

float4 main_ps(PS_INPUT input) : SV_TARGET {
    float2 source_uv;
    float eye_sign;
    if (!PackedToSource(input.TexCoord, source_uv, eye_sign)) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Null/uninitialized buffers read as an invalid contract. Keep the exact current color frame
    // live while the producer starts or after it fails terminally.
    bool warp_available = WarpAvailable();
    float2 sample_uv = warp_available ? Reproject(source_uv, eye_sign) : source_uv;
    sample_uv.x = saturate(sample_uv.x);
    return warp_available ?
        SampleWithCollarDefocus(sample_uv) :
        SourceColor.Sample(LinearSampler, sample_uv);
}
