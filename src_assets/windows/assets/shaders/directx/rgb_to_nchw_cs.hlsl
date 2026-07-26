Texture2D<float4> InputTexture : register(t0);
RWStructuredBuffer<float> OutputBuffer : register(u0);
RWStructuredBuffer<float> OutputAppearanceOrdinal : register(u1);
SamplerState LinearSampler : register(s0);

#include "include/depth_constants.hlsl"
#include "include/depth_color.hlsl"

// Compute shader to bilinearly resize RGB interleaved image to NCHW Float32 with ImageNet normalization
[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= target_w || DTid.y >= target_h)
        return;

    // Calculate normalized UV coordinates based on target dimensions (center of the pixel)
    float2 uv = float2((DTid.x + 0.5f) / (float)target_w, (DTid.y + 0.5f) / (float)target_h);
    uint base_idx = DTid.y * target_w + DTid.x;

    // Sample with linear filtering (SampleLevel bypasses mipmapping)
    float4 pixel = InputTexture.SampleLevel(LinearSampler, uv, 0);

    // Preserve one exposure-ordinal signal BEFORE the model's HDR tonemapper. Reinhard uses a
    // per-pixel luminance divisor, so maxRGB reconstructed from the post-tone-map NCHW tensor is
    // not rank invariant: a global exposure change can reverse two differently coloured pixels.
    //
    // Point sampling is deliberate. A nonlinear exposure curve followed by bilinear mixing can
    // also reverse the ordering of two mixtures even when the curve is globally monotone. At a
    // fixed capture texel, however, maxRGB commutes with an identical monotone channel curve; a
    // pair can retain its order or collapse to a tie, never reverse it. The later census requires
    // reliable contrast in both frames, so ties abstain.
    uint source_w, source_h;
    InputTexture.GetDimensions(source_w, source_h);
    uint2 source_point = min(
        uint2(uv * float2(source_w, source_h)),
        uint2(source_w - 1u, source_h - 1u));
    float3 capture_rgb = InputTexture.Load(int3(source_point, 0)).rgb;
    OutputAppearanceOrdinal[base_idx] =
        max(capture_rgb.r, max(capture_rgb.g, capture_rgb.b));

    // HDR capture is scRGB: LINEAR light with Rec.709 primaries (so primaries already match
    // the SDR-trained model; only the transfer function differs). Compress highlights with
    // luminance-preserving Reinhard (1.0 scRGB = 80 nits) so highlights don't blind the first conv,
    // then gamma-encode to sRGB so midtones land where ImageNet normalization expects them
    // (feeding linear light makes the image far too dark and degrades the depth estimate).
    pixel.rgb = DepthColorToSrgb(pixel.rgb, color_mode);

    // ImageNet Normalization
    float r = (pixel.r - 0.485f) / 0.229f;
    float g = (pixel.g - 0.456f) / 0.224f;
    float b = (pixel.b - 0.406f) / 0.225f;

    // NCHW Layout mapping: Output shape is [1, 3, target_h, target_w]
    uint channel_stride = target_w * target_h;
    
    // Write R, G, B channels
    OutputBuffer[base_idx] = r;
    OutputBuffer[base_idx + channel_stride] = g;
    OutputBuffer[base_idx + 2 * channel_stride] = b;
}
