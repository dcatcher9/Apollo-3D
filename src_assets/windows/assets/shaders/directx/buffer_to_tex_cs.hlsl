StructuredBuffer<float>  InputBuffer : register(t0);
StructuredBuffer<float4> MinMaxEma   : register(t1);  // [0]={P2,P98,initialized,frame_state}
Texture2D<float>          PreviousDepth : register(t2);
Texture2D<uint>           EmaMotionMask : register(t3);
RWTexture2D<float>       OutputTexture : register(u0);

#include "include/depth_constants.hlsl"

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= target_w || DTid.y >= target_h)
        return;

    uint idx = DTid.y * target_w + DTid.x;
    float4 scale = MinMaxEma[0];
    if (scale.w < 0.5f) {
        // All model outputs for this frame were invalid. Hold the last real depth without
        // initializing normalization/history or feeding synthetic geometry downstream.
        OutputTexture[DTid.xy] = PreviousDepth[DTid.xy];
        return;
    }
    float raw = InputBuffer[idx];
    if (isnan(raw) || isinf(raw) || raw < 0.0f) {
        // A missing prediction is not evidence for the far plane. Hold the last valid depth for
        // this texel; on the first frame PreviousDepth is the inert cleared value.
        OutputTexture[DTid.xy] = PreviousDepth[DTid.xy];
        return;
    }

    // The validated permanent order is range->pixel: normalize using the current P2/P98 bounds,
    // then temporally smooth the normalized depth.
    float2 mm = scale.xy;
    if (any(isnan(mm)) || any(isinf(mm)) || mm.y < mm.x) mm = float2(0.0f, 1.0f);
    float mapped = saturate((max(raw, 0.0f) - mm.x) / max(mm.y - mm.x, 1e-6f));
    float old_depth = PreviousDepth[DTid.xy];
    if (isnan(old_depth) || isinf(old_depth)) old_depth = mapped;
    float frame_alpha = scale.w > 1.5f ? 1.0f : ema_alpha;
    float filtered = lerp(old_depth, mapped, frame_alpha);
    OutputTexture[DTid.xy] = EmaMotionMask[DTid.xy] != 0u ?
                              lerp(filtered, mapped, ema_edge_strength) : filtered;
}
