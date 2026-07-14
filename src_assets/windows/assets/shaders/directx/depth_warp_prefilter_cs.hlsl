// Precompute the silhouette-width depth filter used by the SBS reprojection search.
//
// The former pixel-shader path averaged four bilinear samples at (+/-0.75, +/-0.75)
// depth texels for every search probe. At a depth-texel center that operation is the
// separable 3x3 kernel [0.375, 0.25, 0.375] in each axis. Computing that kernel once
// per completed depth map lets the full-resolution warp use one bilinear sample per
// probe while preserving the original normalized depth for metrics and subject state.

Texture2D<float> InputDepth : register(t0);
RWTexture2D<float> OutputDepth : register(u0);

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint width, height;
    InputDepth.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    int2 p = int2(tid.xy);
    int2 hi = int2((int) width - 1, (int) height - 1);
    static const float weights[3] = {0.375f, 0.25f, 0.375f};
    float filtered = 0.0f;
    [unroll]
    for (int y = -1; y <= 1; ++y) {
        [unroll]
        for (int x = -1; x <= 1; ++x) {
            filtered += weights[x + 1] * weights[y + 1] *
                        InputDepth.Load(int3(clamp(p + int2(x, y), int2(0, 0), hi), 0));
        }
    }
    OutputDepth[tid.xy] = filtered;
}
