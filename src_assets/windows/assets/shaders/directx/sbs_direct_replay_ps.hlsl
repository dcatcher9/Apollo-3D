// Harness-only renderer for an authenticated external final-parallax artifact.
//
// The uploaded R32 field uses the finite [0,1] interchange encoding. The native V2 evaluator
// and live stream do not use this shader; they compile the pinned production renderer instead.

Texture2D<float4> SourceColor : register(t0);
Texture2D<float> EncodedParallax : register(t1);
SamplerState LinearSampler : register(s0);

cbuffer OutputGeometry : register(b2) {
    float content_scale_x;
    float content_scale_y;
    float2 geometry_reserved;
};

cbuffer DirectParallaxConstants : register(b4) {
    float direct_parallax_source_u_limit;
    float3 direct_parallax_reserved;
};

struct PS_INPUT {
    float4 Pos : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

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
    float encoded = EncodedParallax.SampleLevel(
        LinearSampler,
        float2(source_x, source_y),
        0
    );
    return (encoded * 2.0f - 1.0f) * direct_parallax_source_u_limit;
}

float2 Reproject(float2 destination_uv, float eye_sign) {
    float source_x = destination_uv.x;
    [unroll]
    for (int iteration = 0; iteration < 11; ++iteration) {
        float next_source_x = destination_uv.x + eye_sign *
            SampleParallax(source_x, destination_uv.y);
        bool exactly_settled = asuint(next_source_x) == asuint(source_x);
        source_x = next_source_x;
#if !defined(HOST_SBS_TEST_FIXED_ELEVEN_REFERENCE)
        if (exactly_settled) {
            break;
        }
#endif
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
    float2 sample_uv = Reproject(source_uv, eye_sign);
    sample_uv.x = saturate(sample_uv.x);
    return SourceColor.Sample(LinearSampler, sample_uv);
}

// R32_FLOAT preserves the inverse map before finite-image clamping.
float mapping_ps(PS_INPUT input) : SV_TARGET {
    float2 source_uv;
    float eye_sign;
    if (!PackedToSource(input.TexCoord, source_uv, eye_sign)) {
        return 0.0f;
    }
    return Reproject(source_uv, eye_sign).x;
}

// Red marks samples that the rendered output clamps to the nearest source column.
float4 mask_ps(PS_INPUT input) : SV_TARGET {
    float2 source_uv;
    float eye_sign;
    if (!PackedToSource(input.TexCoord, source_uv, eye_sign)) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    float source_x = Reproject(source_uv, eye_sign).x;
    float boundary = (source_x < 0.0f || source_x > 1.0f) ? 1.0f : 0.0f;
    return float4(boundary, 0.0f, 0.0f, 1.0f);
}
