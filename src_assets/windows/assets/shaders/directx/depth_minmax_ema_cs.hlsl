// 1-thread pass: fold this frame's raw min/max into an EMA'd min/max (temporal scale
// stability, a la iw3's IncrementalEMAScaler), then reset the raw accumulator for the
// next frame's fused moments/range reduction. Runs after the shared histogram/quantile resolve on the D3D timeline.
//
// w is this-frame validity: 0 = invalid/hold, 1 = valid with history, 2 = first valid frame.
RWStructuredBuffer<float4> MinMaxEma : register(u0);  // [0]={min,max,initialized,frame_state}
RWByteAddressBuffer        MinMaxRaw : register(u1);  // min/max bits, valid + eligible counts
RWStructuredBuffer<uint>   Histogram : register(u2);  // permanent P2/P98 histogram from the shared quantile resolve
// Tagged depth-health state. Geometry does not consume this state directly, and this pass is
// its only producer: the tag and reserved word keep their CPU-initialized values.
//   [0].yz = empty-raw count, collapsed-raw count (stored as uint bits)
//   [1]    = valid-depth fraction, effective EMA range width, range-collapsed flag,
//            depth-ready flag
RWStructuredBuffer<float4> DepthHealthState : register(u3);

uint LoadNormalizationHistogram(uint bin) {
    return Histogram[bin];
}

void ResetNormalizationHistogramBin(uint bin) {
    Histogram[bin] = 0u;
}

#include "include/depth_minmax_ema_common.hlsl"

[numthreads(1, 1, 1)]
void main() {
    ResolveDepthNormalization();
}
