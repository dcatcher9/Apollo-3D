// Production Host SBS renderer for the authenticated Depth Coordinate V2 final field.
//
// The producer supplies a signed, one-eye displacement in source-U units whose horizontal slope
// is strictly below one. Each eye map therefore has one inverse; a short fixed-point solve is the
// complete visibility rule. Invalid or unauthenticated state renders the current color frame
// through identity so a producer fault can never expose stale pixels or legacy geometry.

Texture2D<float4> SourceColor : register(t0);
Texture2D<float> FinalParallax : register(t1);
StructuredBuffer<float4> ParallaxState : register(t2);
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
    float2 sample_uv = WarpAvailable() ? Reproject(source_uv, eye_sign) : source_uv;
    sample_uv.x = saturate(sample_uv.x);
    return SourceColor.Sample(LinearSampler, sample_uv);
}
