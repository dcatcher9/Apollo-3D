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

// Degree-7 polynomial approximation of Bestv2RawShiftPx over normalized depth [0, 1].
// Its maximum raw-shift error is below 0.01 source pixel (and below 0.014 output pixel at
// the maximum supported aspect scaling). The live full-resolution probe loop uses this form
// to replace three transcendental operations per probe; low-resolution subject reductions
// retain the exact curve above.
float Bestv2RawShiftPxFast(float d) {
    d = saturate(d);
    return -1.39635933f + d * (2.776208766f + d * (21.04503417f + d *
           (-94.6673759f + d * (376.6610774f + d * (-645.141824f + d *
           (482.8701123f - 133.5645677f * d))))));
}

#endif
