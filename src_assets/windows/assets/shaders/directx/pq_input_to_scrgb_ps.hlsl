// Decodes PQ (Rec.2020 ST 2084) texels 1:1 to linear scRGB. The Game 3D cursor patch uses it so
// the Windows cursor is blended in linear light over a PQ export.
Texture2D source_texture : register(t0);

#include "include/base_vs_types.hlsl"
#include "include/pq_input.hlsl"

float4 main_ps(vertex_t input) : SV_Target {
    const float4 code = source_texture.Load(int3(input.viewpoint_pos.xy, 0));
    return float4(PQInputToScRGB(code.rgb), code.a);
}
