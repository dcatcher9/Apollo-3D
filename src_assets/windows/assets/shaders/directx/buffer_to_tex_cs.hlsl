StructuredBuffer<float>  InputBuffer : register(t0);
StructuredBuffer<float4> MinMaxEma   : register(t1);  // [0]={P2,P98,initialized,_}
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

    // The validated permanent order is range->pixel: normalize using the current P2/P98 bounds,
    // then temporally smooth the normalized depth.
    float2 mm = MinMaxEma[0].xy;
    float mapped = saturate((max(raw, 0.0f) - mm.x) / max(mm.y - mm.x, 1e-6f));
    float old_depth = OutputTexture[DTid.xy];
    OutputTexture[DTid.xy] = lerp(old_depth, mapped, ema_alpha);
}
