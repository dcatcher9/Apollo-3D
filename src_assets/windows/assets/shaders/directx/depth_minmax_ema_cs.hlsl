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
RWStructuredBuffer<uint>   Histogram : register(u2);  // 256 bins from depth_hist_cs (percentile mode only)

#include "include/depth_constants.hlsl"

#define NUM_BINS 256

[numthreads(1, 1, 1)]
void main() {
    float new_min = asfloat(MinMaxRaw.Load(0));
    float new_max = asfloat(MinMaxRaw.Load(4));

    // Robust percentile bounds: replace the raw min/max with the norm_pct_lo/hi percentiles
    // scanned from the 256-bin histogram (depth_hist_cs, binned over the raw range). Outlier
    // pixels -- DA-V2's near-spike tail is the classic case -- land in the extreme bins and
    // are excluded from the normalization range instead of squeezing the whole scene's
    // parallax. Bin centers quantize the bound to ~0.4% of the frame range, far below the
    // temporal EMA's smoothing. Values outside [lo,hi] saturate in buffer_to_tex_cs.
    bool use_percentile = (pct_lo > 0.0f) || (pct_hi < 1.0f);
    if (use_percentile) {
        float bin_w = max(new_max - new_min, 1e-12f) / (float)NUM_BINS;
        float total = (float)(target_w * target_h);
        float lo_count = pct_lo * total;
        float hi_count = pct_hi * total;
        float pct_min = new_min;
        float pct_max = new_max;
        bool found_lo = (pct_lo <= 0.0f);   // disabled -> keep the raw min
        bool found_hi = (pct_hi >= 1.0f);   // disabled -> keep the raw max
        float cum = 0.0f;
        [loop]
        for (uint b = 0; b < NUM_BINS; b++) {
            cum += (float)Histogram[b];
            if (!found_lo && cum >= lo_count) {
                pct_min = new_min + ((float)b + 0.5f) * bin_w;
                found_lo = true;
            }
            if (!found_hi && cum >= hi_count) {
                pct_max = new_min + ((float)b + 0.5f) * bin_w;
                found_hi = true;
            }
            Histogram[b] = 0u;  // reset for the next frame's accumulation
        }
        if (pct_max - pct_min > 1e-9f) {
            new_min = pct_min;
            new_max = pct_max;
        }
    }

    float4 s = MinMaxEma[0];
    float ref = s.w;
    // Scene-lock state: how many updates the bounds have been converging since the last
    // cut/seed. Once past lock_frames, the blend rate drops to locked_alpha (near-freeze) --
    // the realtime stand-in for VD3D's whole-video locked percentile range: within a scene
    // the depth scale stops breathing entirely; a scene cut (below) restarts the learning.
    float frames_settled = MinMaxEma[1].y;
    if (s.z < 0.5f) {
        // First frame: seed the EMA (and the reference range) with the raw values.
        s.x = new_min;
        s.y = new_max;
        s.z = 1.0f;
        ref = max(new_max - new_min, 1e-6f);
        frames_settled = 0.0f;
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
            frames_settled = 0.0f;  // relearn the new scene's bounds before re-locking
        } else {
            bool locked = (lock_frames > 0.5f) && (frames_settled >= lock_frames);
            float alpha = locked ? locked_alpha : minmax_alpha;
            s.x = lerp(s.x, new_min, alpha);
            s.y = lerp(s.y, new_max, alpha);
            frames_settled = min(frames_settled + 1.0f, 16777000.0f);  // clamp below fp32 integer limit
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
    MinMaxEma[1] = float4(range_scale, frames_settled, 0.0f, 0.0f);

    // Reset accumulator so next frame's InterlockedMin/Max start from the identity.
    MinMaxRaw.Store(0, 0xFFFFFFFFu);
    MinMaxRaw.Store(4, 0u);
}
