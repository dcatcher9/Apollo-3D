// 1-thread pass: fold this frame's raw min/max into an EMA'd min/max (temporal scale
// stability, a la iw3's IncrementalEMAScaler), then reset the raw accumulator for the
// next frame's reduction. Runs on the D3D timeline right after depth_minmax_cs.
//
// w is this-frame validity: 0 = invalid/hold, 1 = valid with history, 2 = first valid frame.
RWStructuredBuffer<float4> MinMaxEma : register(u0);  // [0]={min,max,initialized,frame_state}
RWByteAddressBuffer        MinMaxRaw : register(u1);  // min bits, max bits, valid count
RWStructuredBuffer<uint>   Histogram : register(u2);  // permanent P2/P98 histogram from depth_hist_cs
// Append-only diagnostics. SubjectState[0..2] remain the production warp contract:
//   [3].zw = valid-depth fraction, effective EMA range width
//   [4].zw = empty-raw count, collapsed-raw count (stored as uint bits)
//   [5].xy = current range-collapsed flag, depth-ready flag
RWStructuredBuffer<float4> DiagnosticState : register(u3);

#include "include/depth_constants.hlsl"

#define NUM_BINS 256

[numthreads(1, 1, 1)]
void main() {
    float new_min = asfloat(MinMaxRaw.Load(0));
    float new_max = asfloat(MinMaxRaw.Load(4));
    uint valid_count = MinMaxRaw.Load(8);
    bool valid_bounds = valid_count > 0u && !isnan(new_min) && !isinf(new_min) &&
                        !isnan(new_max) && !isinf(new_max) && new_max >= new_min;
    float4 telemetry = DiagnosticState[3];
    uint4 health_counters = asuint(DiagnosticState[4]);
    float4 telemetry_flags = DiagnosticState[5];

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
            // Take the crossing bin's OUTER EDGE, not its center. A percentile only excludes its
            // nominal 2% when the distribution is smooth across the crossing bin; when a large
            // atom sits there -- sky, a far wall, DA-V2's flat background -- a bin-center bound
            // cuts through the atom and clips a large share of the frame to a single depth. c525
            // puts 66.1% of its pixels in bin 0, and the centered bound saturated 49.9% of the
            // frame to exactly 0. An outer edge can only widen the range, so it clips at most the
            // nominal fraction, and costs at most one bin (~0.4%) of range precision.
            if (!found_lo && cum >= lo_count) {
                pct_min = new_min + (float)b * bin_w;
                found_lo = true;
            }
            if (!found_hi && cum >= hi_count) {
                pct_max = new_min + ((float)b + 1.0f) * bin_w;
                found_hi = true;
            }
            Histogram[b] = 0u;  // reset for the next frame's accumulation
        }
        if (valid_bounds && pct_max - pct_min > 1e-9f) {
            new_min = pct_min;
            new_max = pct_max;
        }
    }

    float collapse_scale = max(1.0f, max(abs(new_min), abs(new_max)));
    bool range_collapsed = valid_bounds &&
                           new_max - new_min <= collapse_scale * 1.0e-5f;
    if (!valid_bounds) {
        health_counters.z = min(health_counters.z + 1u, 0xfffffffeu);
    } else if (range_collapsed) {
        health_counters.w = min(health_counters.w + 1u, 0xfffffffeu);
    }
    telemetry.z = (float)valid_count /
                  (float)max(target_w * target_h, 1u);

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
        // Attack fast, release slow. A symmetric EMA lags the live percentiles, and whenever the
        // smoothed range is NARROWER than the frame's, buffer_to_tex_cs saturates the difference
        // away -- lag becomes clipped depth. Expanding immediately to cover [new_min, new_max]
        // makes that impossible, while contraction still decays at minmax_alpha. Expansion is also
        // the safe direction for stability: the range is a multiplicative gain, so growing it
        // LOWERS the gain, and it is fast SHRINKING that makes the depth scale breathe.
        s.x = min(lerp(s.x, new_min, minmax_alpha), new_min);
        s.y = max(lerp(s.y, new_max, minmax_alpha), new_max);
        s.w = 1.0f;
    }
    MinMaxEma[0] = s;
    telemetry.w = s.z > 0.5f ? max(s.y - s.x, 0.0f) : 0.0f;
    telemetry_flags.x = range_collapsed ? 1.0f : 0.0f;
    telemetry_flags.y = s.z > 0.5f ? 1.0f : 0.0f;
    DiagnosticState[3] = telemetry;
    DiagnosticState[4] = asfloat(health_counters);
    DiagnosticState[5] = telemetry_flags;

    // Reset accumulator so next frame's InterlockedMin/Max start from the identity.
    MinMaxRaw.Store(0, 0xFFFFFFFFu);
    MinMaxRaw.Store(4, 0u);
    MinMaxRaw.Store(8, 0u);
}
