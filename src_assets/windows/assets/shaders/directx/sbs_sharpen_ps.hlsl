// VisionDepth3D Bestv2 SDR tensor_sharpen(factor=.2), applied after the completed warp.
// Each eye is filtered independently; F.conv2d(padding=1) uses zeros beyond its boundary.
Texture2D<float4> SbsTexture : register(t0);

#include "include/sbs_warp_common.hlsl"

struct PS_INPUT {
    float4 Pos : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

float4 EyeLoad(int local_x, int y, bool right_eye, int eye_w, int h) {
    if (local_x < 0 || local_x >= eye_w || y < 0 || y >= h) return 0.0f;
    return SbsTexture.Load(int3(local_x + (right_eye ? eye_w : 0), y, 0));
}

float4 EyeSample(float local_x, float y, bool right_eye, int eye_w, int h) {
    int x0 = (int)floor(local_x);
    int y0 = (int)floor(y);
    float2 f = frac(float2(local_x, y));
    return lerp(lerp(EyeLoad(x0, y0, right_eye, eye_w, h),
                     EyeLoad(x0 + 1, y0, right_eye, eye_w, h), f.x),
                lerp(EyeLoad(x0, y0 + 1, right_eye, eye_w, h),
                     EyeLoad(x0 + 1, y0 + 1, right_eye, eye_w, h), f.x), f.y);
}

float4 main_ps(PS_INPUT input) : SV_TARGET {
    uint full_w_u, h_u;
    SbsTexture.GetDimensions(full_w_u, h_u);
    int eye_w = (int)(full_w_u / 2u);
    int h = (int)h_u;
    int2 px = int2(input.Pos.xy);
    bool right_eye = px.x >= eye_w;
    int x = right_eye ? px.x - eye_w : px.x;
    float2 output_uv = (float2(x, px.y) + 0.5f) / float2(eye_w, h);
    float2 unused_source_uv;
    if (!ContentToSourceUV(output_uv, unused_source_uv)) return float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Bestv2's kernel is calibrated at one SOURCE pixel. Fractional sampling keeps the same
    // footprint when the encoder scales the SBS output (for example 5120 -> 4096).
    float tap = max(source_to_output, 1e-3f);
    float4 out_color = EyeLoad(x, px.y, right_eye, eye_w, h) * 5.2f
                     - EyeSample((float)x - tap, (float)px.y, right_eye, eye_w, h)
                     - EyeSample((float)x + tap, (float)px.y, right_eye, eye_w, h)
                     - EyeSample((float)x, (float)px.y - tap, right_eye, eye_w, h)
                     - EyeSample((float)x, (float)px.y + tap, right_eye, eye_w, h);
    return saturate(out_color);
}
