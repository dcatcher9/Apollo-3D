// 1-thread pass: fold this frame's raw min/max into an EMA'd min/max (temporal scale
// stability, a la iw3's IncrementalEMAScaler), then reset the raw accumulator for the
// next frame's reduction. Runs on the D3D timeline right after depth_minmax_cs.
//
// w is this-frame validity: 0 = invalid/hold, 1 = valid with history, 2 = first valid frame.
RWStructuredBuffer<float4> MinMaxEma : register(u0);  // [0]={min,max,initialized,frame_state}
RWByteAddressBuffer        MinMaxRaw : register(u1);  // min bits, max bits, valid count
RWStructuredBuffer<uint>   Histogram : register(u2);  // permanent P2/P98 histogram from depth_hist_cs

#include "include/depth_constants.hlsl"

#define NUM_BINS 256

[numthreads(1, 1, 1)]
void main() {
    float new_min = asfloat(MinMaxRaw.Load(0));
    float new_max = asfloat(MinMaxRaw.Load(4));
    uint valid_count = MinMaxRaw.Load(8);
    bool valid_bounds = valid_count > 0u && !isnan(new_min) && !isinf(new_min) &&
                        !isnan(new_max) && !isinf(new_max) && new_max >= new_min;

    // Robust percentile bounds: replace the raw min/max with the permanent P2/P98 percentiles
    // scanned from the 256-bin histogram (depth_hist_cs, binned over the raw range). Outlier
    // pixels -- DA-V2's near-spike tail is the classic case -- land in the extreme bins and
    // are excluded from the normalization range instead of squeezing the whole scene's
    // parallax. Bin centers quantize the bound to ~0.4% of the frame range, far below the
    // temporal EMA's smoothing. Values outside [lo,hi] saturate in buffer_to_tex_cs.
    {
        float bin_w = valid_bounds ? max(new_max - new_min, 1e-12f) / (float)NUM_BINS : 0.0f;
        float total = (float)valid_count;
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
        if (valid_bounds && pct_max - pct_min > 1e-9f) {
            new_min = pct_min;
            new_max = pct_max;
        }
    }

    float4 s = MinMaxEma[0];
    if (!valid_bounds) {
        // Preserve the last real normalization state and mark this frame as a hold. In particular,
        // do not manufacture an initialized [0,1] range when the first TensorRT result is invalid.
        s.w = 0.0f;
    } else if (s.z < 0.5f || any(isnan(s.xy)) || any(isinf(s.xy))) {
        // First frame: seed the EMA with the percentile values.
        s.x = new_min;
        s.y = new_max;
        s.z = 1.0f;
        s.w = 2.0f;
    } else {
        s.x = lerp(s.x, new_min, minmax_alpha);
        s.y = lerp(s.y, new_max, minmax_alpha);
        s.w = 1.0f;
    }
    MinMaxEma[0] = s;

    // Reset accumulator so next frame's InterlockedMin/Max start from the identity.
    MinMaxRaw.Store(0, 0xFFFFFFFFu);
    MinMaxRaw.Store(4, 0u);
    MinMaxRaw.Store(8, 0u);
}
