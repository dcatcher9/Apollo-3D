#ifndef BAND_CURVE_HLSL
#define BAND_CURVE_HLSL

// VD3D's near/mid/far Gaussian disparity bands, translated to Apollo's high=near depth (band
// centers mirrored). Band amplitudes from the VD3D Bestv2 preset: fg -9*1.11, mg -3,
// bg +2.4*1.05 px in VD3D's negative=pop convention -> +1 / +0.300 / -0.252 here (positive =
// pops out; divergence is the master gain). Note the weighted-average curve PEAKS at ~0.86
// (d=1), not literally +1 -- the mid band dilutes the near amplitude -- so do not tighten the
// probe search radius assuming a +-1 range.
//
// Shared by sbs_reprojection_ps.hlsl (the per-pixel parallax) and depth_subject_resolve_cs.hlsl
// (the subject-anchor precompute) so the two can never drift; keep the C++ replica in
// tools/warpsim/warpsim.cpp (band_curve) in sync with any change here.
float BandCurve(float d) {
    float wn = exp(-0.5f * ((d - 0.85f) / 0.24f) * ((d - 0.85f) / 0.24f));
    float wm = exp(-0.5f * ((d - 0.50f) / 0.28f) * ((d - 0.50f) / 0.28f));
    float wf = exp(-0.5f * ((d - 0.15f) / 0.24f) * ((d - 0.15f) / 0.24f));
    return (wn * 1.0f + wm * 0.300f + wf * -0.252f) / (wn + wm + wf + 1e-6f);
}

#endif
