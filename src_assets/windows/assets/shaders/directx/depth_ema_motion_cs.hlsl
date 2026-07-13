// Mark moving depth-transition bands where the per-pixel EMA should trust the current frame.
// Both current mapped depth and previous normalized depth are immutable SRVs, so dilation is
// deterministic and cannot race neighboring writes in buffer_to_tex_cs.
StructuredBuffer<float>  InputBuffer : register(t0);
StructuredBuffer<float4> MinMaxEma   : register(t1);  // [0]={P2,P98,initialized,_}
Texture2D<float>          PreviousDepth : register(t2);
RWTexture2D<uint>         MotionMask : register(u0);

#include "include/depth_constants.hlsl"

int2 ClampPixel(int2 p) {
    return clamp(p, int2(0, 0), int2((int)target_w - 1, (int)target_h - 1));
}

float CurrentDepth(int2 p) {
    p = ClampPixel(p);
    float2 mm = MinMaxEma[0].xy;
    float raw = InputBuffer[(uint)p.y * target_w + (uint)p.x];
    return saturate((max(raw, 0.0f) - mm.x) / max(mm.y - mm.x, 1e-6f));
}

bool IsMovingEdge(int2 p) {
    p = ClampPixel(p);
    float current = CurrentDepth(p);
    float change = abs(current - PreviousDepth[p]);
    float gradient = 0.0f;
    gradient = max(gradient, abs(current - CurrentDepth(p + int2(-1, 0))));
    gradient = max(gradient, abs(current - CurrentDepth(p + int2( 1, 0))));
    gradient = max(gradient, abs(current - CurrentDepth(p + int2(0, -1))));
    gradient = max(gradient, abs(current - CurrentDepth(p + int2(0,  1))));
    return change >= ema_edge_change && gradient >= ema_edge_gradient;
}

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= target_w || DTid.y >= target_h)
        return;

    uint radius = min((uint)(ema_edge_dilation + 0.5f), 2u);
    bool moving = false;
    [unroll]
    for (int y = -2; y <= 2; ++y) {
        [unroll]
        for (int x = -2; x <= 2; ++x) {
            if (abs(x) <= (int)radius && abs(y) <= (int)radius)
                moving = moving || IsMovingEdge(int2(DTid.xy) + int2(x, y));
        }
    }
    MotionMask[DTid.xy] = moving ? 1u : 0u;
}
