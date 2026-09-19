#include "include/convert_sdr_base.hlsl"

float3 CONVERT_FUNCTION(float3 input)
{
    // The overwhelmingly common BGRA8 path targets Rec. 601/709 SDR. Use the bounded
    // no-pow composite transfer regardless of the physical display's HDR state; BGRA is
    // categorically display-referred SDR. Only BT.2020 gamut conversion needs the slow path.
    // Initialize the single return value explicitly. FXC's flow analysis reports X4000 for the
    // equivalent early-return form when this include is compiled through the three YUV entry
    // points, even though both branches return. The optimizer removes this initialization.
    float3 converted = float3(0.0, 0.0, 0.0);
    if (target_is_hdr) {
        // An SDR game (or WGC SDR frame) may feed an HDR stream. Restore its configured
        // reference white in absolute scRGB before encoding Rec.2020/ST2084.
        converted = scRGBTo2100PQ(RemoveSRGBCurve(input) * source_sdr_white_scrgb);
    } else if (!target_bt2020) {
        converted = SRGBCodeToBT709Code(input);
    } else {
        // BGRA8 desktop capture is display-referred sRGB even when the physical display has
        // Advanced Color enabled (notably WGC SDR capture). Decode it before producing the
        // transfer function and gamut declared in the encoded video VUI, but never apply the
        // FP16/scRGB HDR-to-SDR tone map to this path.
        converted = ConvertLinearToTargetSdr(RemoveSRGBCurve(input), false);
    }
    return converted;
}
