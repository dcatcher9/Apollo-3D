// Stream-local gamma follows Gloam's uncalibrated HDR gamma path (LutGenerator.cs,
// commit 79d4c471bc5e8802fd4288712014776af023b4d2). Evaluate the analytical curve
// here instead of sampling a display's finite hardware gamma ramp.
// common.hlsl must be included first. This is deliberately outside the authenticated
// depth/warp shader closures: correction belongs to the final stream conversion.
cbuffer stream_gamma_cbuffer: register(b4) {
  uint stream_gamma_mode;
  float stream_gamma_white_scrgb;
  float2 stream_gamma_padding;
};

float StreamGammaChannel(float linear_value, float white, float gamma) {
  if (!isfinite(linear_value)) {
    return 0.0;
  }
  // Gloam's LUT is separable in the HDR wire's Rec.2020 primaries. A bright
  // channel passes through even when another channel in the same pixel is dark.
  if (linear_value >= white) {
    return linear_value;
  }
  float x = saturate(linear_value / white);
  // Use the exact IEC sRGB OETF. ApplySRGBCurve is a fast approximation intended
  // for other paths and would change this correction, especially near black.
  float encoded = x <= 0.0031308 ? 12.92 * x :
                                   1.055 * pow(x, 1.0 / 2.4) - 0.055;
  return white * pow(max(encoded, 0.0), gamma);
}

float3 StreamGammaScRGBToPQ(float3 rgb) {
  // Keep the legacy operation order exactly for Windows default.
  if (stream_gamma_mode == 0) {
    return scRGBTo2100PQ(rgb);
  }
  float3 linear2020 = Rec709toRec2020(rgb);
  float white = max(stream_gamma_white_scrgb, 0.5);  // host bounds W to 40..1000 nits
  float gamma = stream_gamma_mode == 1 ? 2.2 : 2.4;
  float3 corrected = float3(
    StreamGammaChannel(linear2020.r, white, gamma),
    StreamGammaChannel(linear2020.g, white, gamma),
    StreamGammaChannel(linear2020.b, white, gamma)
  );
  return NitsToPQ(corrected * 80.0);
}
