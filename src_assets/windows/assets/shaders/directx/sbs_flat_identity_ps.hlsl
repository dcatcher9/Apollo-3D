// Independent fail-flat Host SBS renderer.
//
// This shader deliberately has no V2 contract include, depth texture, state buffer, or
// authenticated-source dependency. It duplicates the current mono frame into both packed eyes,
// preserving the source transfer function and the same per-eye aspect-fit convention as V2.

Texture2D<float4> SourceColor : register(t0);
SamplerState LinearSampler : register(s0);

cbuffer HostSbsFlatGeometry : register(b2) {
    float content_scale_x;
    float content_scale_y;
    float2 geometry_reserved;
};

struct PS_INPUT {
    float4 Pos : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

float4 main_ps(PS_INPUT input) : SV_TARGET {
    const bool right_eye = input.TexCoord.x > 0.5f;
    float2 eye_uv = input.TexCoord;
    eye_uv.x = right_eye ? (input.TexCoord.x - 0.5f) * 2.0f :
                           input.TexCoord.x * 2.0f;

    const float2 scale = max(
        float2(content_scale_x, content_scale_y),
        float2(1e-6f, 1e-6f)
    );
    const float2 lo = 0.5f * (float2(1.0f, 1.0f) - scale);
    const float2 hi = lo + scale;
    if (any(eye_uv < lo) || any(eye_uv > hi)) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    return SourceColor.Sample(LinearSampler, saturate((eye_uv - lo) / scale));
}
