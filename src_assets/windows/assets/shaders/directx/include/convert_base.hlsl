#include "include/convert_sdr_base.hlsl"

float3 CONVERT_FUNCTION(float3 input)
{
    // The overwhelmingly common BGRA8 path targets Rec. 601/709 SDR. Use the bounded
    // no-pow composite transfer regardless of the physical display's HDR state; BGRA is
    // categorically display-referred SDR. Only BT.2020 gamut conversion needs the slow path.
    if (!target_bt2020) {
        return SRGBCodeToBT709Code(input);
    }

    // BGRA8 desktop capture is display-referred sRGB even when the physical display has
    // Advanced Color enabled (notably WGC SDR capture). Decode it before producing the
    // transfer function and gamut declared in the encoded video VUI, but never apply the
    // FP16/scRGB HDR-to-SDR tone map to this path.
    return ConvertLinearToTargetSdr(RemoveSRGBCurve(input), false);
}
