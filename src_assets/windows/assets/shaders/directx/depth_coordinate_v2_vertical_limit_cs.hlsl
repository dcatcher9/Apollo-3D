// Produce the exact column-wise upper envelope and the fixed orientation-selective vertical
// share used by Host SBS V2:
//
//   Upper(y) = max_s Candidate(s) - step * abs(y - s)
//   Lower(y) = min_s Candidate(s) + step * abs(y - s)
//   Conditioned(y) = a * Upper(y) + (1 - a) * Lower(y)
//
// where step = max_vertical_shear / target_w and a is the authenticated contract constant.
// VerticalConditioned temporarily stores the forward lower scan, then is overwritten with the
// completed share during the backward scan. VerticalMajorant remains a diagnostic upper field;
// the row pass consumes only VerticalConditioned.

Texture2D<float> Candidate : register(t0);
RWTexture2D<float> VerticalMajorant : register(u0);
RWTexture2D<float> VerticalConditioned : register(u1);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2.hlsl"

float share_vertical_envelopes(float upper, float lower) {
    // Keep the persisted float32 evaluation order and forbid contraction/reassociation.
    precise float majorant_share = v2_vertical_majorant_share;
    precise float minorant_share = 1.0f - majorant_share;
    precise float majorant_term = majorant_share * upper;
    precise float minorant_term = minorant_share * lower;
    precise float conditioned = majorant_term + minorant_term;
    return conditioned;
}

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x;
    if (x >= target_w || target_h == 0u || target_w == 0u) {
        return;
    }

    float max_step = v2_max_vertical_shear / (float)target_w;
    float candidate = Candidate[uint2(x, 0u)];
    float upper = candidate;
    float lower = candidate;
    VerticalMajorant[uint2(x, 0u)] = upper;
    VerticalConditioned[uint2(x, 0u)] = lower;

    [loop]
    for (uint y = 1u; y < target_h; ++y) {
        candidate = Candidate[uint2(x, y)];
        upper = max(candidate, upper - max_step);
        lower = min(candidate, lower + max_step);
        VerticalMajorant[uint2(x, y)] = upper;
        VerticalConditioned[uint2(x, y)] = lower;
    }
    DeviceMemoryBarrier();

    const uint last_y = target_h - 1u;
    upper = VerticalMajorant[uint2(x, last_y)];
    lower = VerticalConditioned[uint2(x, last_y)];
    VerticalConditioned[uint2(x, last_y)] = share_vertical_envelopes(upper, lower);

    [loop]
    for (int y = (int)target_h - 2; y >= 0; --y) {
        const uint2 position = uint2(x, (uint)y);
        upper = max(VerticalMajorant[position], upper - max_step);
        lower = min(VerticalConditioned[position], lower + max_step);
        VerticalMajorant[position] = upper;
        VerticalConditioned[position] = share_vertical_envelopes(upper, lower);
    }
}
