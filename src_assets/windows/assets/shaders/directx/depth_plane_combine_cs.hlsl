// Blend the filled silhouette back with the original band: max(original, closed*.70).
Texture2D<float> OriginalBand : register(t0);
Texture2D<float> ClosedBand : register(t1);
RWTexture2D<float> OutputMask : register(u0);

#include "include/depth_plane_constants.hlsl"

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= plane_w || id.y >= plane_h) return;
    OutputMask[id.xy] = max(OriginalBand.Load(int3(id.xy, 0)),
                            ClosedBand.Load(int3(id.xy, 0)) * 0.70f);
}
