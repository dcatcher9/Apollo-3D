// Normalize raw DAV2 depth, apply the temporal EMA, and publish the exact moving-edge diagnostic
// mask in one pass. The mask remains independently observable, but the depth decision consumes
// the same boolean in-register instead of round-tripping it through a second texture dispatch.
StructuredBuffer<float>  InputBuffer : register(t0);
StructuredBuffer<float4> MinMaxEma   : register(t1);  // [0]={P2,P98,initialized,frame_state}
Texture2D<float>          PreviousDepth : register(t2);
Texture2D<uint>           TensorExclusion : register(t3);
RWTexture2D<float>        OutputTexture : register(u0);
RWTexture2D<uint>         MotionMask : register(u1);

#include "include/depth_constants.hlsl"

int2 ClampPixel(int2 p) {
    return clamp(p, int2(0, 0), int2((int)target_w - 1, (int)target_h - 1));
}

float CurrentDepth(int2 p, float2 mm) {
    p = ClampPixel(p);
    float raw = InputBuffer[(uint)p.y * target_w + (uint)p.x];
    if (isnan(raw) || isinf(raw) || raw < 0.0f) {
        // Missing current evidence holds history. Treating it as raw zero would manufacture a
        // far-depth edge and could snap adjacent valid texels through the motion mask.
        return PreviousDepth[p];
    }
    return saturate((raw - mm.x) / max(mm.y - mm.x, 1e-6f));
}

bool IsMovingEdge(int2 p, float current, float2 mm) {
    p = ClampPixel(p);

    static const int2 neighbor_offsets[4] = {
        int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)
    };
    [unroll]
    for (int neighbor_index = 0; neighbor_index < 4; ++neighbor_index) {
        int2 neighbor = ClampPixel(p + neighbor_offsets[neighbor_index]);
        if (TensorExclusion[neighbor] != 0u) {
            // The moving-edge classifier is a five-cell stencil. An artificial sanitizer fill
            // boundary must not make an adjacent admitted texel snap toward current depth.
            return false;
        }
    }

    float change = abs(current - PreviousDepth[p]);
    float gradient = 0.0f;
    gradient = max(gradient, abs(current - CurrentDepth(p + int2(-1, 0), mm)));
    gradient = max(gradient, abs(current - CurrentDepth(p + int2( 1, 0), mm)));
    gradient = max(gradient, abs(current - CurrentDepth(p + int2(0, -1), mm)));
    gradient = max(gradient, abs(current - CurrentDepth(p + int2(0,  1), mm)));
    // ema_edge_gradient is specified in 434-reference-texel units. Keep the motion mask stable
    // when the native depth shape resolves a 392- or 420-short-side grid.
    float reference_gradient = gradient * DepthReferenceTexelScale();
    return change >= ema_edge_change && reference_gradient >= ema_edge_gradient;
}

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= target_w || DTid.y >= target_h)
        return;

    // Synthetic contain-fit padding has no diagnostic authority. Its normalized depth is filled
    // by pad_main after every admitted content texel has been published. Keeping the padding path
    // out of this pass avoids recomputing the same five-cell boundary stencil for every padded
    // texel on narrow or tall analysis ROIs.
    if (TensorExclusion[DTid.xy] != 0u) {
        MotionMask[DTid.xy] = 0u;
        return;
    }

    float4 scale = MinMaxEma[0];
    uint2 sample_position = DTid.xy;

    // An all-invalid completion holds history and publishes the same all-zero diagnostic mask as
    // the former first pass. Do this before touching raw model output.
    if (scale.w < 0.5f) {
        MotionMask[DTid.xy] = 0u;
        OutputTexture[DTid.xy] = PreviousDepth[sample_position];
        return;
    }

    float2 mm = scale.xy;
    if (any(isnan(mm)) || any(isinf(mm)) || mm.y < mm.x) mm = float2(0.0f, 1.0f);

    uint idx = sample_position.y * target_w + sample_position.x;
    float raw = InputBuffer[idx];
    bool raw_valid = !isnan(raw) && !isinf(raw) && raw >= 0.0f;
    float previous = PreviousDepth[sample_position];
    float mapped = raw_valid ?
        saturate((raw - mm.x) / max(mm.y - mm.x, 1e-6f)) : previous;

    // The old mask pass ran only when both thresholds were positive and only for a history frame
    // (frame_state == 1). Preserve its comparison form so a non-finite state behaves identically.
    bool moving = false;
    if (ema_edge_change > 0.0f && ema_edge_gradient > 0.0f &&
        !(scale.w < 0.5f || scale.w > 1.5f)) {
        moving = IsMovingEdge(int2(sample_position), mapped, mm);
    }

    MotionMask[DTid.xy] = moving ? 1u : 0u;

    if (!raw_valid) {
        // A missing prediction is not evidence for the far plane. Hold the last valid depth for
        // this texel; on the first frame PreviousDepth is the inert cleared value.
        OutputTexture[DTid.xy] = previous;
        return;
    }

    float old_depth = previous;
    if (isnan(old_depth) || isinf(old_depth)) old_depth = mapped;
    float frame_alpha = scale.w > 1.5f ? 1.0f : ema_alpha;
    float filtered = lerp(old_depth, mapped, frame_alpha);
    OutputTexture[DTid.xy] = moving ?
        lerp(filtered, mapped, ema_edge_strength) : filtered;
}

// The first dispatch has completed before this entry point runs, so every clamped source below is
// an admitted content texel already written by main. D3D11 serializes UAV dispatch accesses; this
// pass performs only one exclusion lookup and one depth load/store per synthetic padding texel.
[numthreads(16, 16, 1)]
void pad_main(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= target_w || DTid.y >= target_h ||
        TensorExclusion[DTid.xy] == 0u)
        return;

    OutputTexture[DTid.xy] = OutputTexture[DepthAnalysisClampCell(DTid.xy)];
}
