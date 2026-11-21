// Exact-raster FP16/scRGB-to-P010 chroma. The caller supplies two even-width eyes,
// an even source height and a matching native chroma-plane UAV; each Z slice owns one eye.
#include "include/convert_perceptual_quantizer_base.hlsl"
#include "include/convert_yuv420_packed_uv_base.hlsl"

RWTexture2D<float2> OutputChroma : register(u0);

#define UV_TILE_WIDTH 16u
#define UV_TILE_HEIGHT 8u
#define SOURCE_TILE_WIDTH (2u * UV_TILE_WIDTH + 2u)
#define SOURCE_TILE_HEIGHT (2u * UV_TILE_HEIGHT)

groupshared float3 gamma_tile[SOURCE_TILE_WIDTH * SOURCE_TILE_HEIGHT];

[numthreads(UV_TILE_WIDTH, UV_TILE_HEIGHT, 1)]
void main_cs(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    // The two halo columns independently clamp to this eye.
    int eye_left = int(group.z * packed_eye_width);
    int eye_right = eye_left + int(packed_eye_width);
    uint start_x = group.x * UV_TILE_WIDTH;
    uint start_y = group.y * UV_TILE_HEIGHT;
    uint thread_index = lane.y * UV_TILE_WIDTH + lane.x;

    [loop]
    for (uint i = thread_index; i < SOURCE_TILE_WIDTH * SOURCE_TILE_HEIGHT;
         i += UV_TILE_WIDTH * UV_TILE_HEIGHT) {
        int x = eye_left + int(2u * start_x + i % SOURCE_TILE_WIDTH) - 1;
        int y = int(2u * start_y + i / SOURCE_TILE_WIDTH);
        gamma_tile[i] = LoadChromaTexel(int2(x, y), eye_left, eye_right);
    }
    GroupMemoryBarrierWithGroupSync();

    uint x = start_x + lane.x;
    uint y = start_y + lane.y;
    if (x >= packed_eye_width / 2u || y >= chroma_source_height / 2u) return;

    uint center = lane.y * 2u * SOURCE_TILE_WIDTH + lane.x * 2u + 1u;
    float3 rgb = CombineLeftChromaSamples(
        gamma_tile[center],
        gamma_tile[center + SOURCE_TILE_WIDTH],
        gamma_tile[center - 1u],
        gamma_tile[center + 1u],
        gamma_tile[center + SOURCE_TILE_WIDTH - 1u],
        gamma_tile[center + SOURCE_TILE_WIDTH + 1u]);
    OutputChroma[uint2(uint(eye_left) / 2u + x, y)] = EncodeChroma(rgb);
}
