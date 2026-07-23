#include "include/convert_sdr_base.hlsl"

float3 CONVERT_FUNCTION(float3 input)
{
    // This shader is selected from the actual frame texture format, so only the linear FP16
    // path may consume the display-derived HDR-to-SDR flag.
    return ConvertLinearToTargetSdr(input, source_is_hdr);
}
