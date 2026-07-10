// Separable rectangular max/min/average filters. Two passes reproduce PyTorch's
// 21x21 max pool, 15x15 min pool (closing), and 13x13 average pool exactly.
Texture2D<float> InputMask : register(t0);
RWTexture2D<float> OutputMask : register(u0);

#include "include/depth_plane_constants.hlsl"

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= plane_w || id.y >= plane_h) return;

    float value = filter_op == 0u ? 0.0f : (filter_op == 1u ? 1.0f : 0.0f);
    float sum = 0.0f;
    [loop]
    for (int o = -(int)filter_radius; o <= (int)filter_radius; ++o) {
        int2 p = int2(id.xy);
        if (filter_axis == 0u) p.x += o; else p.y += o;
        // PyTorch pooling padding uses -inf for max, +inf for min, and zero for average.
        bool inside = p.x >= 0 && p.y >= 0 && p.x < (int)plane_w && p.y < (int)plane_h;
        float sample_value = inside ? InputMask.Load(int3(p, 0)) :
            (filter_op == 0u ? -3.402823466e+38f : (filter_op == 1u ? 3.402823466e+38f : 0.0f));
        if (filter_op == 0u) value = max(value, sample_value);
        else if (filter_op == 1u) value = min(value, sample_value);
        else sum += sample_value;
    }
    if (filter_op == 2u) value = sum / (float)(2u * filter_radius + 1u);
    OutputMask[id.xy] = saturate(value);
}
