// Bestv2 apply_subject_plane_lock: depth band and center weighting.
Texture2D<float> DepthTexture : register(t0);
StructuredBuffer<float4> SubjectState : register(t1);
RWTexture2D<float> PlaneBand : register(u0);

#include "include/depth_plane_constants.hlsl"

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= plane_w || id.y >= plane_h) return;

    float d = saturate(DepthTexture.Load(int3(id.xy, 0)));
    float subject_depth = saturate(SubjectState[0].z);
    float t = (d - subject_depth) / max(plane_width, 1e-4f);
    float band = saturate(exp(-0.5f * t * t));

    // torch.linspace includes both endpoints. Match Bestv2's fixed center_bias=.35,
    // vertical sigma=.70 and horizontal sigma=.85 exactly.
    float x = plane_w > 1u ? -1.0f + 2.0f * (float)id.x / (float)(plane_w - 1u) : 0.0f;
    float y = plane_h > 1u ? -1.0f + 2.0f * (float)id.y / (float)(plane_h - 1u) : 0.0f;
    float center_weight = exp(-0.5f * ((y / 0.70f) * (y / 0.70f) +
                                      (x / 0.85f) * (x / 0.85f)));
    PlaneBand[id.xy] = band * (0.65f + 0.35f * center_weight);
}
