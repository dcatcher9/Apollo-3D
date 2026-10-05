// Explicit PQ (Rec.2020 ST 2084) Game 3D export presented on a linear scRGB local output.
Texture2D source_texture : register(t0);

#include "include/base_vs_types.hlsl"
#include "include/pq_input.hlsl"

float4 main_ps(vertex_t input) : SV_Target {
    return SamplePQInputAsScRGB(source_texture, input.tex_coord);
}
