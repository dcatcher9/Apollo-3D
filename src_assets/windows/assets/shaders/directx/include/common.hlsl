// This is a fast sRGB approximation from Microsoft's ColorSpaceUtility.hlsli
float3 ApplySRGBCurve(float3 x)
{
    return x < 0.0031308 ? 12.92 * x : 1.13005 * sqrt(x - 0.00228) - 0.13448 * x + 0.005719;
}

float3 RemoveSRGBCurve(float3 x)
{
    x = saturate(x);
    float3 lo = x / 12.92;
    float3 hi = pow((x + 0.055) / 1.055, 2.4);
    return x <= 0.04045 ? lo : hi;
}

float3 ApplyBT709Curve(float3 x)
{
    x = max(x, 0.0);
    float3 lo = 4.5 * x;
    float3 hi = 1.099 * pow(x, 0.45) - 0.099;
    return x < 0.018 ? lo : hi;
}

float3 ApplyBT2020Curve(float3 x)
{
    // BT.2020 10-bit uses alpha=1.0993 and beta=0.0181.
    x = max(x, 0.0);
    float3 lo = 4.5 * x;
    float3 hi = 1.0993 * pow(x, 0.45) - 0.0993;
    return x < 0.0181 ? lo : hi;
}

float SRGBCodeToBT709Code(float x)
{
    x = saturate(x);

    // Compose the sRGB EOTF with the BT.709 OETF without an SFU power operation on every
    // BGRA8 luma/chroma tap. These fourth-degree minimax polynomials use normalized domains
    // and have a dense-domain maximum code-value error below 3.7e-5 (< 0.01 of an 8-bit
    // code step). That is substantially cheaper than up to six transfer evaluations per
    // output UV pixel in the 7680x2160 type0s path and avoids the extra fetches of a 1D LUT.
    if (x <= 0.04045) {
        return x * (4.5 / 12.92);
    }

    if (x < 0.142825681303039) { // sRGB code value whose decoded linear value is 0.018
        float t = (2.0 * x - 0.183275681303039) / 0.102375681303039;
        return mad(
            mad(
                mad(
                    mad(-0.0000203208382756466, t, 0.000379028747322516),
                    t,
                    0.00808284888652290
                ),
                t,
                0.0330765198090362
            ),
            t,
            0.0394817883386249
        );
    }

    float t = (2.0 * x - 1.142825681303039) / 0.857174318696961;
    return saturate(
        mad(
            mad(
                mad(
                    mad(0.00127867013357397, t, -0.00322804757744502),
                    t,
                    0.0124427001451592
                ),
                t,
                0.462640801365181
            ),
            t,
            0.526902601731455
        )
    );
}

float3 SRGBCodeToBT709Code(float3 x)
{
    return float3(
        SRGBCodeToBT709Code(x.r),
        SRGBCodeToBT709Code(x.g),
        SRGBCodeToBT709Code(x.b)
    );
}

float3 NitsToPQ(float3 L)
{
    // Constants from SMPTE 2084 PQ
    static const float m1 = 2610.0 / 4096.0 / 4;
    static const float m2 = 2523.0 / 4096.0 * 128;
    static const float c1 = 3424.0 / 4096.0;
    static const float c2 = 2413.0 / 4096.0 * 32;
    static const float c3 = 2392.0 / 4096.0 * 32;

    float3 Lp = pow(saturate(L / 10000.0), m1);
    return pow((c1 + c2 * Lp) / (1 + c3 * Lp), m2);
}

float3 Rec709toRec2020(float3 rec709)
{
    static const float3x3 ConvMat =
    {
        0.627402, 0.329292, 0.043306,
        0.069095, 0.919544, 0.011360,
        0.016394, 0.088028, 0.895578
    };
    return mul(ConvMat, rec709);
}

float3 ToneMapScRgbToSdr(float3 rgb, float sdr_white_scrgb)
{
    rgb = max(rgb, 0.0);

    // Normalize absolute scRGB (1.0 = 80 nits) to the source display's configured SDR
    // reference white. Preserve shadows and midtones through a 0.5-linear knee, then use a
    // smooth asymptotic shoulder so HDR highlights retain detail instead of hard-clipping.
    rgb /= max(sdr_white_scrgb, 1.0);
    float luminance = max(dot(rgb, float3(0.2126, 0.7152, 0.0722)), 0.0);
    if (luminance > 0.5) {
        float mapped_luminance = 0.5 + 0.5 * (1.0 - exp(-2.0 * (luminance - 0.5)));
        rgb *= mapped_luminance / max(luminance, 1e-6);
    }

    // Uniformly fit rare out-of-gamut peaks without changing hue.
    float peak = max(rgb.r, max(rgb.g, rgb.b));
    rgb /= max(peak, 1.0);
    return rgb;
}

float3 scRGBTo2100PQ(float3 rgb)
{
    // Convert from Rec 709 primaries (used by scRGB) to Rec 2020 primaries (used by Rec 2100)
    rgb = Rec709toRec2020(rgb);

    // 1.0f is defined as 80 nits in the scRGB colorspace
    rgb *= 80;

    // Apply the PQ transfer function on the raw color values in nits
    return NitsToPQ(rgb);
}
