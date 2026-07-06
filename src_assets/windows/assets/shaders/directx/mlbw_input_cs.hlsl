// Learned warp (iw3 MLBW), pass 1 of 3: build the model's input tensors for BOTH eyes.
// Each input is [3, FH, FW] fp32 (planar): ch0 = depth resampled onto the model grid,
// ch1 = divergence feature, ch2 = convergence feature (constant planes, ramped to zero
// near the L/R borders -- iw3's preserve_screen_border -- which replaces the old
// shader-side BorderFade). The right eye runs on a HORIZONTALLY FLIPPED depth (iw3's
// shift>0 convention); the feature planes are edge-symmetric so both eyes share values.

Texture2D<float>  DepthTexture  : register(t0);  // estimator depth (normalized, EMA'd)
SamplerState      LinearSampler : register(s0);
RWStructuredBuffer<float> InLeft  : register(u0);  // 3*FH*FW
RWStructuredBuffer<float> InRight : register(u1);  // 3*FH*FW (flipped depth)

cbuffer Params : register(b0) {
    uint fw;             // model grid dims (from the engine's fixed input shape)
    uint fh;
    float div_feat;      // divergence_pix / 32 (iw3 make_divergence_feature_value)
    float conv_feat;     // -divergence_pix * convergence / 32
    float border_texels; // preserve-screen-border ramp width in grid texels (0 = off)
    float pad0;
    float pad1;
    float pad2;
};

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= fw || id.y >= fh) {
        return;
    }
    float v = (id.y + 0.5f) / (float) fh;
    float dl = DepthTexture.SampleLevel(LinearSampler, float2((id.x + 0.5f) / (float) fw, v), 0);
    float dr = DepthTexture.SampleLevel(LinearSampler, float2((fw - id.x - 0.5f) / (float) fw, v), 0);

    // Border ramp (iw3: linspace(0,1,border_pix) from each L/R edge). Symmetric, so the
    // flipped (right-eye) input shares the same feature planes.
    float ramp = 1.0f;
    if (border_texels > 1.0f) {
        float e = min((float) id.x, (float) (fw - 1 - id.x));
        ramp = saturate(e / (border_texels - 1.0f));
    }

    uint plane = fw * fh;
    uint idx = id.y * fw + id.x;
    InLeft[idx] = dl;
    InLeft[plane + idx] = div_feat * ramp;
    InLeft[2 * plane + idx] = conv_feat * ramp;
    InRight[idx] = dr;
    InRight[plane + idx] = div_feat * ramp;
    InRight[2 * plane + idx] = conv_feat * ramp;
}
