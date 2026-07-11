// 1-thread pass: fold this frame's raw min/max into an EMA'd min/max (temporal scale
// stability, a la iw3's IncrementalEMAScaler), then reset the raw accumulator for the
// next frame's reduction. Runs on the D3D timeline right after depth_minmax_cs.
//
RWStructuredBuffer<float4> MinMaxEma : register(u0);  // [0]={min,max,initialized,_}
RWByteAddressBuffer        MinMaxRaw : register(u1);  // [0]=min bits, [4]=max bits
RWStructuredBuffer<uint>   Histogram : register(u2);  // 256 bins from depth_hist_cs (percentile mode only)

#include "include/depth_constants.hlsl"

#define NUM_BINS 256

[numthreads(1, 1, 1)]
void main() {
    float new_min = asfloat(MinMaxRaw.Load(0));
    float new_max = asfloat(MinMaxRaw.Load(4));

    // Robust percentile bounds: replace the raw min/max with the permanent P2/P98 percentiles
    // scanned from the 256-bin histogram (depth_hist_cs, binned over the raw range). Outlier
    // pixels -- DA-V2's near-spike tail is the classic case -- land in the extreme bins and
    // are excluded from the normalization range instead of squeezing the whole scene's
    // parallax. Bin centers quantize the bound to ~0.4% of the frame range, far below the
    // temporal EMA's smoothing. Values outside [lo,hi] saturate in buffer_to_tex_cs.
    {
        float bin_w = max(new_max - new_min, 1e-12f) / (float)NUM_BINS;
        float total = (float)(target_w * target_h);
        float lo_count = 0.02f * total;
        float hi_count = 0.98f * total;
        float pct_min = new_min;
        float pct_max = new_max;
        bool found_lo = false;
        bool found_hi = false;
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
    if (s.z < 0.5f) {
        // First frame: seed the EMA with the percentile values.
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
