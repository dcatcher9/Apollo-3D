// Resamples a full side-by-side texture authored at another size into the packed stream raster.
//
// The packed YUV conversion reads exact output-sized texels, so any resizing must happen before it.
// Both rasters have the same per-eye aspect: each output eye maps linearly onto its own source half.
// Sampling is clamped to that half's texel centers, so linear filtering never mixes the two eyes.
// Values pass through in their own transfer (linear or encoded); nothing here converts color.

Texture2D<float4> PackedSource : register(t0);
SamplerState LinearSampler : register(s0);

struct PS_INPUT {
    float4 Pos : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

float4 main_ps(PS_INPUT input) : SV_TARGET {
    uint width, height;
    PackedSource.GetDimensions(width, height);
    const float half_texel = 0.5f / (float)width;
    // Output pixel centers never fall on the seam; the output width is even.
    const bool right_eye = input.TexCoord.x > 0.5f;
    const float lo = right_eye ? 0.5f + half_texel : half_texel;
    const float hi = right_eye ? 1.0f - half_texel : 0.5f - half_texel;
    return PackedSource.Sample(LinearSampler, float2(clamp(input.TexCoord.x, lo, hi), input.TexCoord.y));
}
