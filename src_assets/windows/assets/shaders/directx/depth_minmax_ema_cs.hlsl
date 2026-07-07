// 1-thread pass: fold this frame's raw min/max into an EMA'd min/max (temporal scale
// stability, a la iw3's IncrementalEMAScaler), then reset the raw accumulator for the
// next frame's reduction. Runs on the D3D timeline right after depth_minmax_cs.
//
// A1 (scene-cut snap): on a hard content cut the raw range jumps, and a slow EMA would make
// the depth scale "swim" for ~0.2-0.7 s. When the raw range/center departs the EMA hard, SNAP
// to raw instead of blending. Gated by snap_ratio (0 = off).
// A3 (range floor): track a slow-max REFERENCE range; when the current range drops well below
// it (flat desktop/page content), emit a range_scale < 1 so buffer_to_tex compresses the depth
// contrast -- otherwise min/max normalization stretches a near-flat scene to full parallax and
// amplifies the depth model's hallucinated structure. Gated by floor_frac (0 = off).

RWStructuredBuffer<float4> MinMaxEma : register(u0);  // [0]={min,max,initialized,ref_range}, [1]={range_scale,_,_,_}
RWByteAddressBuffer        MinMaxRaw : register(u1);  // [0]=min bits, [4]=max bits

cbuffer Constants : register(b0) {
    uint target_w;
    uint target_h;
    uint is_hdr;
    float ema_alpha;
    float minmax_alpha;
    uint reduce_threads;
    uint output_transform;  // unused here; shared layout
    float depth_shift;      // unused here; shared layout
    float snap_ratio;       // A1: raw-vs-EMA range ratio (or center shift) that triggers a snap; 0 = off
    float floor_frac;       // A3: current range below ref*floor_frac -> scale parallax down; 0 = off
    float floor_ref_alpha;  // A3: reference-range decay toward smaller ranges
    float pad0;
};

[numthreads(1, 1, 1)]
void main() {
    float new_min = asfloat(MinMaxRaw.Load(0));
    float new_max = asfloat(MinMaxRaw.Load(4));

    float4 s = MinMaxEma[0];
    float ref = s.w;
    if (s.z < 0.5f) {
        // First frame: seed the EMA (and the reference range) with the raw values.
        s.x = new_min;
        s.y = new_max;
        s.z = 1.0f;
        ref = max(new_max - new_min, 1e-6f);
    } else {
        float ema_range = max(s.y - s.x, 1e-6f);
        float raw_range = max(new_max - new_min, 1e-6f);
        float rr = raw_range / ema_range;
        float center_shift = abs(0.5f * (new_min + new_max) - 0.5f * (s.x + s.y)) / ema_range;
        bool cut = (snap_ratio > 0.0f) &&
                   (rr > snap_ratio || rr < 1.0f / snap_ratio || center_shift > 0.5f);
        if (cut) {
            s.x = new_min;  // A1: snap the scale to the new scene immediately
            s.y = new_max;
        } else {
            s.x = lerp(s.x, new_min, minmax_alpha);
            s.y = lerp(s.y, new_max, minmax_alpha);
        }
    }

    // A3: slow-max reference range (snaps up to a new max, decays slowly toward smaller ranges).
    float range = max(s.y - s.x, 1e-6f);
    ref = (range > ref) ? range : lerp(ref, range, floor_ref_alpha);
    s.w = ref;

    float range_scale = 1.0f;
    if (floor_frac > 0.0f) {
        range_scale = saturate(range / max(ref * floor_frac, 1e-6f));
    }

    MinMaxEma[0] = s;
    MinMaxEma[1] = float4(range_scale, 0.0f, 0.0f, 0.0f);

    // Reset accumulator so next frame's InterlockedMin/Max start from the identity.
    MinMaxRaw.Store(0, 0xFFFFFFFFu);
    MinMaxRaw.Store(4, 0u);
}
