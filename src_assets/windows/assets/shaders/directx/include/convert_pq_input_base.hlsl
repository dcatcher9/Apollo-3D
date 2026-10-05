// Final stream conversion of a "pq" Game 3D export (Rec.2020 ST 2084 code values).
// Define PQ_INPUT_HDR_OUTPUT for a PQ (P010 HDR) stream; otherwise the stream is SDR.
#include "include/convert_sdr_base.hlsl"
#include "include/pq_input.hlsl"

// StreamGammaChannel keeps its isfinite() guard for FP16 input; a decoded PQ code is always
// finite, which FXC reports as X3577 for every channel.
#pragma warning(disable : 3577)

float3 CONVERT_FUNCTION(float3 input)
{
    // FXC reports X4000 for an early-return form when compiled through the YUV entry points.
    float3 converted = float3(0.0, 0.0, 0.0);
#if defined(PQ_INPUT_HDR_OUTPUT)
    // Windows default: the wire already carries the stream's Rec.2020 ST 2084 code values.
    converted = input;
    if (stream_gamma_mode != 0) {
        // Same correction as StreamGammaScRGBToPQ, which applies it in the Rec.2020 wire basis.
        float3 linear2020 = PQInputToLinear2020(input);
        float white = max(stream_gamma_white_scrgb, 0.5);  // host bounds W to 40..1000 nits
        float gamma = stream_gamma_mode == 1 ? 2.2 : 2.4;
        float3 corrected = float3(
            StreamGammaChannel(linear2020.r, white, gamma),
            StreamGammaChannel(linear2020.g, white, gamma),
            StreamGammaChannel(linear2020.b, white, gamma)
        );
        converted = NitsToPQ(corrected * 80.0);
    }
#else
    // An HDR export in an SDR stream follows the scRGB tone map of the FP16 export.
    converted = ConvertLinearToTargetSdr(PQInputToScRGB(input), source_is_hdr);
#endif
    return converted;
}
