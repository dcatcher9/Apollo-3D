// Decoding of the Game 3D "pq" wire transfer: Rec.2020 primaries, SMPTE ST 2084 code values,
// code 1.0 = 10000 cd/m2 (R10G10B10A2). These are the exact inverses of common.hlsl NitsToPQ
// and Rec709toRec2020, so a decoded value re-enters the established scRGB paths (1.0 = 80 nits).
// Kept apart from common.hlsl, which belongs to authenticated renderer source closures.

float3 PQToNits(float3 code)
{
    static const float m1 = 2610.0 / 4096.0 / 4;
    static const float m2 = 2523.0 / 4096.0 * 128;
    static const float c1 = 3424.0 / 4096.0;
    static const float c2 = 2413.0 / 4096.0 * 32;
    static const float c3 = 2392.0 / 4096.0 * 32;

    float3 Np = pow(saturate(code), 1.0 / m2);
    return 10000.0 * pow(max(Np - c1, 0.0) / (c2 - c3 * Np), 1.0 / m1);
}

float3 Rec2020toRec709(float3 rec2020)
{
    // Inverse of the host's Rec709toRec2020 matrix (computed from its rounded coefficients).
    static const float3x3 ConvMat =
    {
         1.6604955963, -0.5876564706, -0.0728397133,
        -0.1245461991,  1.1328951150, -0.0083477830,
        -0.0181543227, -0.1005969899,  1.1187512120
    };
    return mul(ConvMat, rec2020);
}

// Linear Rec.2020 in scRGB units (1.0 = 80 nits).
float3 PQInputToLinear2020(float3 code)
{
    return PQToNits(code) / 80.0;
}

// Linear Rec.709 scRGB. Rec.2020 colors outside Rec.709 become negative, as in scRGB.
float3 PQInputToScRGB(float3 code)
{
    return Rec2020toRec709(PQInputToLinear2020(code));
}

float4 LoadPQInputAsScRGB(Texture2D source, int2 position, int2 size)
{
    float4 code = source.Load(int3(clamp(position, int2(0, 0), size - 1), 0));
    return float4(PQInputToScRGB(code.rgb), code.a);
}

// Bilinear filtering in linear light with clamp addressing: each tap is decoded before it is
// weighted, as the FP16 scRGB export was filtered by the sampler.
float4 SamplePQInputAsScRGB(Texture2D source, float2 uv)
{
    uint width, height;
    source.GetDimensions(width, height);
    int2 size = int2(width, height);
    float2 position = uv * float2(width, height) - 0.5;
    float2 base = floor(position);
    float2 weight = position - base;
    int2 texel = int2(base);
    float4 top = lerp(LoadPQInputAsScRGB(source, texel, size), LoadPQInputAsScRGB(source, texel + int2(1, 0), size), weight.x);
    float4 bottom = lerp(LoadPQInputAsScRGB(source, texel + int2(0, 1), size), LoadPQInputAsScRGB(source, texel + int2(1, 1), size), weight.x);
    return lerp(top, bottom, weight.y);
}
