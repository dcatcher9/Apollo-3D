#ifndef BESTV2_CURVE_HLSL
#define BESTV2_CURVE_HLSL

// Cbuffer-independent Bestv2 shaping primitives shared by the live warps and the exact
// subject-plane reducer. Keep the preset constants in one place so ablations cannot drift.
float Bestv2WarpDepth(float d, float4 s0, float4 s1, bool shaped, bool stretch_enabled) {
    if (!shaped) {
        return d;
    }
    float d_str = stretch_enabled ? saturate((d - s1.x) * s1.y) : d;
    return saturate(d_str + s0.x);
}

float Bestv2RawShiftPx(float d) {
    float wn = exp(-0.5f * ((d - 0.85f) / 0.24f) * ((d - 0.85f) / 0.24f));
    float wm = exp(-0.5f * ((d - 0.50f) / 0.28f) * ((d - 0.50f) / 0.28f));
    float wf = exp(-0.5f * ((d - 0.15f) / 0.24f) * ((d - 0.15f) / 0.24f));
    return (wn * 9.99f + wm * 3.0f - wf * 2.52f) / (wn + wm + wf + 1e-6f);
}

#endif
