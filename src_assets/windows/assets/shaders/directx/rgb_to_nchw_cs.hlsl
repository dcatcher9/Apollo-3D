Texture2D<float4> InputTexture : register(t0);
RWStructuredBuffer<float> OutputBuffer : register(u0);
SamplerState LinearSampler : register(s0);

cbuffer Constants : register(b0) {
    uint target_w;
    uint target_h;
    uint is_hdr;
};

// Linear -> sRGB OETF (gamma encode). Depth Anything V2 is trained on sRGB-encoded
// images, so a linear-light signal must be gamma-encoded before ImageNet normalization.
float3 linear_to_srgb(float3 c) {
    c = saturate(c);
    float3 lo = c * 12.92f;
    float3 hi = 1.055f * pow(c, 1.0f / 2.4f) - 0.055f;
    return (c <= 0.0031308f) ? lo : hi;
}

// Compute shader to bilinearly resize RGB interleaved image to NCHW Float32 with ImageNet normalization
[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= target_w || DTid.y >= target_h)
        return;

    // Calculate normalized UV coordinates based on target dimensions (center of the pixel)
    float2 uv = float2((DTid.x + 0.5f) / (float)target_w, (DTid.y + 0.5f) / (float)target_h);

    // Sample with linear filtering (SampleLevel bypasses mipmapping)
    float4 pixel = InputTexture.SampleLevel(LinearSampler, uv, 0);

    // HDR capture is scRGB: LINEAR light with Rec.709 primaries (so primaries already match
    // the SDR-trained model; only the transfer function differs). Compress highlights with
    // Reinhard (x/(1+x); 1.0 = SDR white) so values > 1.0 don't blind the first conv layer,
    // then gamma-encode to sRGB so midtones land where ImageNet normalization expects them
    // (feeding linear light makes the image far too dark and degrades the depth estimate).
    if (is_hdr != 0) {
        float3 c = max(pixel.rgb, 0.0f);  // scRGB can be slightly negative (out of gamut)
        c = c / (1.0f + c);
        pixel.rgb = linear_to_srgb(c);
    }

    // ImageNet Normalization
    float r = (pixel.r - 0.485f) / 0.229f;
    float g = (pixel.g - 0.456f) / 0.224f;
    float b = (pixel.b - 0.406f) / 0.225f;

    // NCHW Layout mapping: Output shape is [1, 3, target_h, target_w]
    uint channel_stride = target_w * target_h;
    uint base_idx = DTid.y * target_w + DTid.x;
    
    // Write R, G, B channels
    OutputBuffer[base_idx] = r;
    OutputBuffer[base_idx + channel_stride] = g;
    OutputBuffer[base_idx + 2 * channel_stride] = b;
}
