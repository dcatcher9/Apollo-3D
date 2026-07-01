StructuredBuffer<float>  InputBuffer : register(t0);
StructuredBuffer<float4> MinMaxEma   : register(t1);  // {min, max, initialized, pad}
RWTexture2D<float>       OutputTexture : register(u0);

cbuffer Constants : register(b0) {
    uint target_w;
    uint target_h;
    uint is_hdr;
    float ema_alpha;
    uint normalize;
    float depth_gamma;
    float minmax_alpha;
    uint reduce_threads;
};

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= target_w || DTid.y >= target_h)
        return;

    uint idx = DTid.y * target_w + DTid.x;
    float raw = InputBuffer[idx];

    float mapped;
    if (normalize != 0) {
        // Per-frame min/max normalization. Depth Anything V2 relative output is
        // affine-invariant (scale/shift arbitrary and drifting), so a fixed curve
        // either collapses or saturates depending on scene content. The min/max is
        // EMA-smoothed across frames (see depth_minmax_ema_cs) for a stable scale.
        float2 mm = MinMaxEma[0].xy;
        float d = saturate((max(raw, 0.0f) - mm.x) / max(mm.y - mm.x, 1e-6f));

        // Optional shaping on the normalized [0,1] disparity. gamma = 1.0 -> linear
        // (screen-space parallax stays proportional to normalized disparity).
        if (depth_gamma != 1.0f) {
            d = pow(d, depth_gamma);
        }
        mapped = d;
    } else {
        // Legacy path: fixed exponential assuming raw disparity ~[0,30]. Kept as an
        // A/B toggle (sbs_3d_normalize=false); fragile because the range is not fixed.
        mapped = 1.0f - exp(-raw * 0.1f);
    }

    // Light EMA temporal smoothing. Depth Anything is scale/shift-variant frame to
    // frame, so with no smoothing the depth at object edges shimmers on motion.
    // ema_alpha (config) blends new vs previous; 1.0 disables smoothing entirely.
    float old_depth = OutputTexture[DTid.xy];
    OutputTexture[DTid.xy] = lerp(old_depth, mapped, ema_alpha);
}
