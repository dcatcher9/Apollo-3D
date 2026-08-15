// Dump-only views of the exact authenticated Host SBS V2 inverse map.
//
// Keep these entry points out of the production renderer root. They are compiled lazily after a
// Dump 3D request, so a missing or invalid diagnostic shader can never disable live streaming.
#include "sbs_reprojection_v2_live_ps.hlsl"

// R32_FLOAT preserves off-image source U.
float mapping_ps(PS_INPUT input) : SV_TARGET {
    float2 source_uv;
    float eye_sign;
    if (!PackedToSource(input.TexCoord, source_uv, eye_sign)) {
        return 0.0f;
    }
    return ReprojectIfAuthorized(source_uv, eye_sign).x;
}

// V2 has no internal holes; red marks finite-image boundary clamping.
float4 mask_ps(PS_INPUT input) : SV_TARGET {
    float2 source_uv;
    float eye_sign;
    if (!PackedToSource(input.TexCoord, source_uv, eye_sign)) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    float source_x = ReprojectIfAuthorized(source_uv, eye_sign).x;
    float boundary = (source_x < 0.0f || source_x > 1.0f) ? 1.0f : 0.0f;
    return float4(boundary, 0.0f, 0.0f, 1.0f);
}
