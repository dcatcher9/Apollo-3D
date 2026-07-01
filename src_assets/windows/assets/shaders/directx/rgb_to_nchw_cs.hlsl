Texture2D<float4> InputTexture : register(t0);
RWStructuredBuffer<float> OutputBuffer : register(u0);
SamplerState LinearSampler : register(s0);

cbuffer Constants : register(b0) {
    uint target_w;
    uint target_h;
    uint is_hdr;
};

// Compute shader to bilinearly resize RGB interleaved image to NCHW Float32 with ImageNet normalization
[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= target_w || DTid.y >= target_h)
        return;

    // Calculate normalized UV coordinates based on target dimensions (center of the pixel)
    float2 uv = float2((DTid.x + 0.5f) / (float)target_w, (DTid.y + 0.5f) / (float)target_h);
    
    // Sample with linear filtering (SampleLevel bypasses mipmapping)
    float4 pixel = InputTexture.SampleLevel(LinearSampler, uv, 0);

    // Apply Reinhard tonemapping to compress HDR highlights before feeding to the SDR-trained AI
    // (x / (1 + x)) - this prevents the network's first conv layer from being blinded by values > 1.0
    if (is_hdr != 0) {
        pixel.rgb = pixel.rgb / (1.0f + pixel.rgb);
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
