// VisionDepth3D Bestv2 SDR tensor_sharpen(factor=.2), applied after the completed warp.
// Each eye is filtered independently; F.conv2d(padding=1) uses zeros beyond its boundary.
Texture2D<float4> SbsTexture : register(t0);

struct PS_INPUT {
    float4 Pos : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

float4 EyeLoad(int local_x, int y, bool right_eye, int eye_w, int h) {
    if (local_x < 0 || local_x >= eye_w || y < 0 || y >= h) return 0.0f;
    return SbsTexture.Load(int3(local_x + (right_eye ? eye_w : 0), y, 0));
}

float4 main_ps(PS_INPUT input) : SV_TARGET {
    uint full_w_u, h_u;
    SbsTexture.GetDimensions(full_w_u, h_u);
    int eye_w = (int)(full_w_u / 2u);
    int h = (int)h_u;
    int2 px = int2(input.Pos.xy);
    bool right_eye = px.x >= eye_w;
    int x = right_eye ? px.x - eye_w : px.x;

    float4 out_color = EyeLoad(x, px.y, right_eye, eye_w, h) * 5.2f
                     - EyeLoad(x - 1, px.y, right_eye, eye_w, h)
                     - EyeLoad(x + 1, px.y, right_eye, eye_w, h)
                     - EyeLoad(x, px.y - 1, right_eye, eye_w, h)
                     - EyeLoad(x, px.y + 1, right_eye, eye_w, h);
    return saturate(out_color);
}
