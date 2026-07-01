// Edge-aware foreground dilation of the low-res depth map (in the spirit of iw3's
// dilate_edge). The depth map is much smaller than the color frame, so upsampling its
// silhouettes produces a stair-stepped parallax boundary -> the jaggy fringe ("毛刺")
// seen at character edges. Here, at depth discontinuities we blend the depth toward a
// smoothed, slightly foreground-biased value (smoothing kills the jaggies; the
// foreground bias keeps background from haloing onto the object). Flat interior regions
// are left untouched so per-object depth stays crisp.

Texture2D<float>   DepthIn  : register(t0);
RWTexture2D<float> DepthOut : register(u0);

cbuffer Constants : register(b0) {
    uint target_w;
    uint target_h;
    uint is_hdr;
    float ema_alpha;
    uint normalize;
    float depth_gamma;
    float minmax_alpha;
    uint reduce_threads;
    float edge_dilation;  // 0 = passthrough; scales the edge blend strength
    float pad0;
    float pad1;
    float pad2;
};

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= target_w || DTid.y >= target_h)
        return;

    int2 p = int2(DTid.xy);
    float c = DepthIn[p];

    if (edge_dilation <= 0.0f) {
        DepthOut[p] = c;
        return;
    }

    // 3x3 neighborhood: local min/max (edge detection) + gaussian blur.
    float mx = c;
    float mn = c;
    float blur = 0.0f;
    float wsum = 0.0f;
    const float gw[3] = {1.0f, 2.0f, 1.0f};
    int2 lo = int2(0, 0);
    int2 hi = int2((int)target_w - 1, (int)target_h - 1);

    [unroll]
    for (int dy = -1; dy <= 1; dy++) {
        [unroll]
        for (int dx = -1; dx <= 1; dx++) {
            int2 q = clamp(p + int2(dx, dy), lo, hi);
            float s = DepthIn[q];
            mx = max(mx, s);
            mn = min(mn, s);
            float w = gw[dx + 1] * gw[dy + 1];
            blur += s * w;
            wsum += w;
        }
    }
    blur /= wsum;

    // Smoothed value with a mild bias toward the local foreground (near) depth.
    float fg_smooth = 0.75f * blur + 0.25f * mx;

    // Edge weight: 0 in flat regions, ->1 where the local depth range is large.
    // edge_dilation scales how aggressively edges are smoothed.
    float range = mx - mn;
    float w = saturate(range * 8.0f * edge_dilation);

    DepthOut[p] = lerp(c, fg_smooth, w);
}
