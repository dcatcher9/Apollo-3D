#ifndef DIRECT_GLYPH_CANONICAL_SAMPLING
#define DIRECT_GLYPH_CANONICAL_SAMPLING

// Integrate an encoded RGB8 cell using the height of the entire extended OCR rectangle.
// The fractional column at the right edge repeats the last native pixel.
uint3 CanonicalRgb(Texture2D<float4> source, uint4 rectangle, uint2 cell, uint targetHeight)
{
    uint2 extent = rectangle.zw - rectangle.xy;
    uint height = extent.y;
    uint2 begin = cell * height;
    uint2 end = begin + height;
    uint2 first = begin / targetHeight;
    uint2 limit = (end + targetHeight - 1u) / targetHeight;
    limit.x = min(limit.x, extent.x);
    uint3 integral = 0u, remainder = 0u;
    [loop] for (uint y = first.y; y < limit.y; ++y)
    {
        uint dy = min(end.y, (y + 1u) * targetHeight) - max(begin.y, y * targetHeight);
        uint3 row = 0u;
        [loop] for (uint x = first.x; x < limit.x; ++x)
        {
            uint right = x + 1u == extent.x ? end.x : min(end.x, (x + 1u) * targetHeight);
            uint dx = right - max(begin.x, x * targetHeight);
            uint2 pixel = rectangle.xy + uint2(x, y);
            row += dx * uint3(floor(saturate(source.Load(int3(pixel, 0)).rgb) * 255.0f + .5f));
        }
        // Divide each row before the vertical sum to avoid 32-bit overflow for tall inputs.
        uint3 quotient = row / height;
        integral += quotient * dy;
        remainder += (row - quotient * height) * dy;
    }
    uint area = height * height;
    return integral / height + ((integral % height) * height + remainder + area / 2u) / area;
}

void CanonicalNativeAxis(uint pixel, uint nativeHeight, uint targetHeight, uint limit, out uint first,
                         out uint second, out float fraction)
{
    int numerator = int((2u * pixel + 1u) * targetHeight) - int(nativeHeight);
    int denominator = int(2u * nativeHeight);
    int low = numerator >= 0 ? numerator / denominator : -((-numerator + denominator - 1) / denominator);
    fraction = float(numerator - low * denominator) / float(denominator);
    first = uint(clamp(low, 0, int(limit) - 1));
    second = uint(clamp(low + 1, 0, int(limit) - 1));
}
#endif
