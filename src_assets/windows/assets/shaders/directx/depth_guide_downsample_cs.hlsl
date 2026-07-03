// Guided depth upsampling, pass 1 of 2: downsample the full-res captured color to the
// depth map's grid, producing the low-res "guide" image whose per-texel colors the
// joint-bilateral pass (depth_guided_upsample_cs) compares against. Tonemapping brings
// HDR scRGB into [0,1] so the color-distance sigma means the same thing in SDR and HDR.

Texture2D<float4> ColorTexture : register(t0);  // full-res captured frame (SDR sRGB / HDR scRGB)
SamplerState      LinearSampler : register(s0);
RWTexture2D<float4> GuideOut    : register(u0);  // in_w x in_h, RGBA16F

cbuffer Params : register(b0) {
    uint in_w;           // low-res depth map dims
    uint in_h;
    uint out_w;          // guided output dims (unused in this pass)
    uint out_h;
    float inv2sig_sp2;   // (unused in this pass)
    float inv2sig_r2;    // (unused in this pass)
    uint is_hdr;
    float radius;        // (unused in this pass)
};

float3 Tonemap(float3 c) {
    if (is_hdr) {
        c = c / (1.0f + c);  // Reinhard: linear scRGB -> [0,1)
    }
    return saturate(c);
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= in_w || id.y >= in_h) {
        return;
    }
    float2 uv = (float2(id.xy) + 0.5f) / float2(in_w, in_h);
    float3 c = ColorTexture.SampleLevel(LinearSampler, uv, 0).rgb;
    GuideOut[id.xy] = float4(Tonemap(c), 1.0f);
}
