// 1-thread pass: fold this frame's raw min/max into an EMA'd min/max (temporal scale
// stability, à la iw3's IncrementalEMAScaler), then reset the raw accumulator for the
// next frame's reduction. Runs on the D3D timeline right after depth_minmax_cs.

RWStructuredBuffer<float4> MinMaxEma : register(u0);  // {min, max, initialized, pad}
RWByteAddressBuffer        MinMaxRaw : register(u1);  // [0]=min bits, [4]=max bits

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

[numthreads(1, 1, 1)]
void main() {
    float new_min = asfloat(MinMaxRaw.Load(0));
    float new_max = asfloat(MinMaxRaw.Load(4));

    float4 s = MinMaxEma[0];
    if (s.z < 0.5f) {
        // First frame: seed the EMA with the raw values instead of blending from zero.
        s.x = new_min;
        s.y = new_max;
        s.z = 1.0f;
    } else {
        s.x = lerp(s.x, new_min, minmax_alpha);
        s.y = lerp(s.y, new_max, minmax_alpha);
    }
    MinMaxEma[0] = s;

    // Reset accumulator so next frame's InterlockedMin/Max start from the identity.
    MinMaxRaw.Store(0, 0xFFFFFFFFu);
    MinMaxRaw.Store(4, 0u);
}
