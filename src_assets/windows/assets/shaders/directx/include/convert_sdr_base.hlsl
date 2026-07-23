#include "include/common.hlsl"

cbuffer sdr_color_transform_cbuffer : register(b1) {
    uint target_bt2020;
    uint source_is_hdr;
    float source_sdr_white_scrgb;
    float sdr_color_transform_padding;
};

float3 ConvertLinearToTargetSdr(float3 input, bool input_is_hdr)
{
    float3 linear_rgb = input_is_hdr ? ToneMapScRgbToSdr(input, source_sdr_white_scrgb) :
                                      max(input, 0.0);

    if (target_bt2020) {
        return saturate(ApplyBT2020Curve(Rec709toRec2020(linear_rgb)));
    }

    // Rec. 601/SMPTE-170M and Rec. 709 use the same camera OETF constants here.
    return saturate(ApplyBT709Curve(linear_rgb));
}
