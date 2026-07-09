// Learned warp (iw3 MLBW), pass 1 of 3: build the model's input tensors for BOTH eyes.
// Each input is [3, FH, FW] fp32 (planar): ch0 = depth resampled onto the model grid,
// ch1 = divergence feature, ch2 = convergence feature (constant planes, ramped to zero
// near the L/R borders -- iw3's preserve_screen_border -- which replaces the old
// shader-side BorderFade). The right eye runs on a HORIZONTALLY FLIPPED depth (iw3's
// shift>0 convention); the feature planes are edge-symmetric so both eyes share values.

Texture2D<float>  DepthTexture  : register(t0);  // estimator depth (normalized, EMA'd)
// Subject-tracking state from depth_subject_resolve_cs: {delta, scurve, subject_depth_ema,
// initialized}. Only read when subject_track is on; unbound/uninitialized -> config conv.
StructuredBuffer<float4> SubjectState : register(t1);
SamplerState      LinearSampler : register(s0);
RWStructuredBuffer<float> InLeft  : register(u0);  // 3*FH*FW
RWStructuredBuffer<float> InRight : register(u1);  // 3*FH*FW (flipped depth)

cbuffer Params : register(b0) {
    uint fw;             // model grid dims (from the engine's fixed input shape)
    uint fh;
    float div_feat;      // divergence_pix / 32 (iw3 make_divergence_feature_value)
    float conv_feat;     // -divergence_pix * convergence / 32
    float border_texels; // preserve-screen-border ramp width in grid texels (0 = off)
    float subject_track; // > 0.5 = drive the convergence plane from the tracked subject depth
    float subject_lock;  // blend from the config convergence (0) to the subject depth (1)
    float conv_cfg;      // config convergence = (focal - floor) / (1 - floor), for the blend
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

    // Subject anchoring under MLBW: convergence IS iw3's "which depth sits at the screen
    // plane", so anchoring the tracked subject = moving the convergence plane to the
    // subject's (EMA'd) depth. Stays in the model's training distribution -- convergence
    // is a user knob in iw3 -- unlike remapping the depth plane itself.
    float cf = conv_feat;
    if (subject_track > 0.5f) {
        float4 s = SubjectState[0];
        if (s.w > 0.5f) {
            float conv = lerp(conv_cfg, saturate(s.z), subject_lock);
            cf = -div_feat * conv;  // conv_feat = -div_pix*conv/32 and div_feat = div_pix/32
        }
    }

    uint plane = fw * fh;
    uint idx = id.y * fw + id.x;
    InLeft[idx] = dl;
    InLeft[plane + idx] = div_feat * ramp;
    InLeft[2 * plane + idx] = cf * ramp;
    InRight[idx] = dr;
    InRight[plane + idx] = div_feat * ramp;
    InRight[2 * plane + idx] = cf * ramp;
}
