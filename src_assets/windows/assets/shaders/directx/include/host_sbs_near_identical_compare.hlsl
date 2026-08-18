#ifndef HOST_SBS_NEAR_IDENTICAL_COMPARE_HLSL
#define HOST_SBS_NEAR_IDENTICAL_COMPARE_HLSL

struct NearIdenticalTileEvidence {
    uint admitted;
    uint medium_changed;
    uint strong_changed;
    uint nonfinite;
};

#define NEAR_IDENTICAL_MEDIUM_DELTA (1.0f / 64.0f)
#define NEAR_IDENTICAL_STRONG_DELTA 0.20f
#define NEAR_IDENTICAL_TILE_THREADS 256u

bool NearIdenticalFinite(float value) {
    return (asuint(value) & 0x7f800000u) != 0x7f800000u;
}

uint4 NearIdenticalClassifyModelInput(
    float3 current_nchw,
    float3 previous_nchw,
    bool admitted) {
    if (!admitted)
        return 0u;

    bool finite =
        all(bool3(
            NearIdenticalFinite(current_nchw.r),
            NearIdenticalFinite(current_nchw.g),
            NearIdenticalFinite(current_nchw.b))) &&
        all(bool3(
            NearIdenticalFinite(previous_nchw.r),
            NearIdenticalFinite(previous_nchw.g),
            NearIdenticalFinite(previous_nchw.b)));
    float max_delta = 0.0f;
    if (finite) {
        // ImageNet means cancel. Scale the NCHW delta directly back to model RGB.
        float3 delta = abs(current_nchw - previous_nchw) *
            float3(0.229f, 0.224f, 0.225f);
        max_delta = max(delta.r, max(delta.g, delta.b));
    }
    return uint4(
        1u,
        finite && max_delta >= NEAR_IDENTICAL_MEDIUM_DELTA ? 1u : 0u,
        finite && max_delta >= NEAR_IDENTICAL_STRONG_DELTA ? 1u : 0u,
        finite ? 0u : 1u);
}

#endif
