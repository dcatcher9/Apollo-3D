StructuredBuffer<float>  InputBuffer : register(t0);
StructuredBuffer<float4> MinMaxEma   : register(t1);  // [0]={min,max,init,ref_range}, [1]={range_scale,_,_,_}
RWTexture2D<float>       OutputTexture : register(u0);

#include "include/depth_constants.hlsl"

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= target_w || DTid.y >= target_h)
        return;

    uint idx = DTid.y * target_w + DTid.x;
    float raw = InputBuffer[idx];

    // DA-V3 emits DEPTH (larger = farther); shifted reciprocal -> disparity (larger = closer)
    // so the normalization below (tuned for DA-V2 disparity) is unchanged. Must match the exact
    // transform in depth_minmax_cs so min/max and this map agree. The shift bounds the near end.
    if (output_transform == 1) raw = 1.0f / (raw + depth_shift);

    // Per-frame min/max normalization. Depth Anything V2 relative output is
    // affine-invariant (scale/shift arbitrary and drifting), so a fixed curve
    // either collapses or saturates depending on scene content. The min/max is
    // EMA-smoothed across frames (see depth_minmax_ema_cs) for a stable scale.
    float2 mm = MinMaxEma[0].xy;
    float mapped = saturate((max(raw, 0.0f) - mm.x) / max(mm.y - mm.x, 1e-6f));

    // A3 range floor: on near-flat content (current range << reference), range_scale < 1;
    // compress the depth contrast toward the ~0.5 focal plane so the scene's parallax shrinks
    // instead of min/max stretching a hallucinated flat-page structure to full separation.
    float range_scale = MinMaxEma[1].x;
    mapped = 0.5f + (mapped - 0.5f) * range_scale;

    // Light EMA temporal smoothing. Depth Anything is scale/shift-variant frame to
    // frame, so with no smoothing the depth at object edges shimmers on motion.
    // ema_alpha (config) blends new vs previous; 1.0 disables smoothing entirely.
    float old_depth = OutputTexture[DTid.xy];
    OutputTexture[DTid.xy] = lerp(old_depth, mapped, ema_alpha);
}
