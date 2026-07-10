StructuredBuffer<float>  InputBuffer : register(t0);
StructuredBuffer<float4> MinMaxEma   : register(t1);  // [0]={min,max,init,ref_range}, [1]={range_scale,_,_,_}
RWTexture2D<float>       OutputTexture : register(u0);
RWTexture2D<float>       RawEmaTexture : register(u1);  // pixel->range mode: per-pixel history in RAW disparity units

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

    // Per-pixel temporal EMA order (see config sbs_3d_ema_pixel_first):
    //  * pixel->range (VD3D order): smooth the RAW disparity first, then normalize the smoothed
    //    value with the current bounds -- so a drifting normalization range re-maps a
    //    self-consistent history each frame (the whole history moves with the range).
    //  * range->pixel (default): normalize first, then EMA the normalized depth -- the history
    //    is stored already-normalized, so it lags when the bounds drift.
    // (Approximation vs VD3D: the min/max/percentile bounds are still reduced from the raw
    //  disparity, not the smoothed disparity; the two differ negligibly at steady state.)
    float2 mm = MinMaxEma[0].xy;
    float range_scale = MinMaxEma[1].x;  // A3 range floor: compress contrast on near-flat content

    if (pixel_ema_first > 0.5f) {
        float raw_old = RawEmaTexture[DTid.xy];
        float raw_sm = lerp(raw_old, raw, ema_alpha);
        RawEmaTexture[DTid.xy] = raw_sm;
        float mapped = saturate((max(raw_sm, 0.0f) - mm.x) / max(mm.y - mm.x, 1e-6f));
        mapped = 0.5f + (mapped - 0.5f) * range_scale;
        OutputTexture[DTid.xy] = mapped;
    } else {
        float mapped = saturate((max(raw, 0.0f) - mm.x) / max(mm.y - mm.x, 1e-6f));
        mapped = 0.5f + (mapped - 0.5f) * range_scale;
        float old_depth = OutputTexture[DTid.xy];
        OutputTexture[DTid.xy] = lerp(old_depth, mapped, ema_alpha);
    }
}
