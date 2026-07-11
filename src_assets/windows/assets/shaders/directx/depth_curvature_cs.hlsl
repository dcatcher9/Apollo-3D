// Foreground curvature (VD3D enhance_foreground_curvature, GPU port): add a rounded bulge to
// the near/foreground region so the subject reads as VOLUME instead of a flat cutout -- Apollo's
// documented "no true small-feature relief (nose reads flat)" residual. Reshapes the NORMALIZED
// depth in place, BEFORE subject tracking and either retained warp path, so both see the same
// shaped depth. VD3D fits an ellipse to the foreground centroid/spread; this port uses a centered
// ellipse (subjects sit near center) gated by the foreground mask, which concentrates the bulge
// on the foreground regardless -- a cheaper approximation with the same visible effect.

RWTexture2D<float> DepthTexture : register(u0);  // normalized depth, high = near; reshaped in place

cbuffer CurveConstants : register(b0) {
    uint  cv_w;
    uint  cv_h;
    float cv_strength;     // config foreground_curvature (push toward the viewer)
    float cv_near_start;   // foreground threshold (VD3D near_start 0.60)
    float cv_gamma;        // bump roundness (VD3D shape_gamma 1.35)
    float cv_spread;       // ellipse half-extent in normalized [-1,1] frame coords
    float cv_pad0;
    float cv_pad1;
};

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= cv_w || id.y >= cv_h) {
        return;
    }
    float d = DepthTexture[id.xy];

    // Soft foreground mask: 0 below near_start, ramps to 1 at the near end.
    float w = saturate((d - cv_near_start) / max(1e-6f, 1.0f - cv_near_start));
    if (w <= 0.0f) {
        return;  // background untouched
    }

    // Centered elliptical bump in [-1,1] frame coords, rounded by pow(gamma).
    float nx = ((float)id.x / (float)max(cv_w - 1, 1u)) * 2.0f - 1.0f;
    float ny = ((float)id.y / (float)max(cv_h - 1, 1u)) * 2.0f - 1.0f;
    float s = max(cv_spread, 1e-6f);
    float radial = saturate(1.0f - ((nx / s) * (nx / s) + (ny / s) * (ny / s)));
    float bump = pow(radial, cv_gamma);

    // Push the foreground interior toward the viewer (VD3D: out = d + bump*w*strength).
    DepthTexture[id.xy] = saturate(d + bump * w * cv_strength);
}
