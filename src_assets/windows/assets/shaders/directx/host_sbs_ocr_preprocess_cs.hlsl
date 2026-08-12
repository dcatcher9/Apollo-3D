Texture2D<float4> InputTexture : register(t0);
RWStructuredBuffer<float> OcrInput : register(u0);

#include "include/depth_color.hlsl"
#include "include/depth_coordinate_v2_contract.generated.hlsl"

// OCR preprocessing is authenticated only as part of the generated V2 contract.  Never let a
// stale generated header silently fall back to a same-shaped but unauthenticated local ABI.
#if !defined(V2_OCR_INPUT_N) || !defined(V2_OCR_INPUT_C) || \
    !defined(V2_OCR_INPUT_WIDTH) || !defined(V2_OCR_INPUT_HEIGHT) || \
    !defined(V2_OCR_IMAGENET_MEAN_B) || !defined(V2_OCR_IMAGENET_MEAN_G) || \
    !defined(V2_OCR_IMAGENET_MEAN_R) || !defined(V2_OCR_IMAGENET_STD_B) || \
    !defined(V2_OCR_IMAGENET_STD_G) || !defined(V2_OCR_IMAGENET_STD_R)
#error "Generated V2 OCR preprocess contract macros are required"
#endif

cbuffer OcrPreprocessConstants : register(b0) {
    uint source_w;
    uint source_h;
    uint crop_top;
    uint crop_height;
    uint color_mode;
    uint3 reserved;
};

#define OCR_WIDTH V2_OCR_INPUT_WIDTH
#define OCR_HEIGHT V2_OCR_INPUT_HEIGHT

float3 LoadDisplayColor(int2 pixel) {
    pixel = clamp(pixel, int2(0, (int)crop_top),
                  int2((int)source_w - 1, (int)(crop_top + crop_height) - 1));
    return DepthColorToSrgb(InputTexture.Load(int3(pixel, 0)).rgb, color_mode);
}

float3 SampleCropBilinear(uint2 output_pixel) {
    float2 source_position = float2(
        ((float)output_pixel.x + 0.5f) * (float)source_w / (float)OCR_WIDTH - 0.5f,
        (float)crop_top +
            ((float)output_pixel.y + 0.5f) * (float)crop_height / (float)OCR_HEIGHT - 0.5f);
    int2 lo = int2(floor(source_position));
    float2 blend = frac(source_position);
    float3 top = lerp(LoadDisplayColor(lo), LoadDisplayColor(lo + int2(1, 0)), blend.x);
    float3 bottom = lerp(
        LoadDisplayColor(lo + int2(0, 1)),
        LoadDisplayColor(lo + int2(1, 1)),
        blend.x);
    return lerp(top, bottom, blend.y);
}

// PP-OCR's DecodeImage stage supplies BGR and NormalizeImage applies the listed mean/std in that
// order. The capture texture is RGB, so channel 0 intentionally receives blue.
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= OCR_WIDTH || id.y >= OCR_HEIGHT || source_w == 0u || source_h == 0u ||
        crop_height == 0u || crop_top + crop_height > source_h) {
        return;
    }
    float3 rgb = SampleCropBilinear(id.xy);
    uint index = id.y * OCR_WIDTH + id.x;
    uint stride = OCR_WIDTH * OCR_HEIGHT;
    OcrInput[index] = (rgb.b - V2_OCR_IMAGENET_MEAN_B) / V2_OCR_IMAGENET_STD_B;
    OcrInput[index + stride] =
        (rgb.g - V2_OCR_IMAGENET_MEAN_G) / V2_OCR_IMAGENET_STD_G;
    OcrInput[index + 2u * stride] =
        (rgb.r - V2_OCR_IMAGENET_MEAN_R) / V2_OCR_IMAGENET_STD_R;
}
