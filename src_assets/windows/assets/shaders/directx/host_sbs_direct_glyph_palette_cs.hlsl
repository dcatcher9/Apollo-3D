// Mixed Direct Glyph RGB encoder and native fill-plus-outline decoder.
// Every lattice uses the height of the same native, extended OCR rectangle.
#include "include/host_sbs_direct_glyph_normalization.hlsl"
#include "include/host_sbs_direct_glyph_palette_weights.generated.hlsl"
#include "include/host_sbs_direct_glyph_line.hlsl"

static const float Gaussian[7] = {1.452672208e-05f, 5.394577810e-03f, 1.878587213e-01f, 6.134643483e-01f,
                                  1.878587213e-01f, 5.394577810e-03f, 1.452672208e-05f};
cbuffer Setup : register(b0)
{
    uint Slot;
    uint Width28;
    uint Width84;
    uint StageValue; // Canonical height, tile offset, filter radius or reduction pass.
    uint HistScale;
}
StructuredBuffer<GlyphLine> LinePlan : register(t11);
#define Box (LinePlan[Slot].box)
Texture2D<float4> Source : register(t0);
StructuredBuffer<float4> Roles : register(t1);
StructuredBuffer<float4> LocalMax : register(t2);
StructuredBuffer<float4> LocalMean : register(t3);
StructuredBuffer<float4> Palette : register(t4);
StructuredBuffer<uint> Winners : register(t5);
StructuredBuffer<float4> NearbyBackground : register(t6);
StructuredBuffer<uint> Rgb28 : register(t7);
StructuredBuffer<uint> Rgb84 : register(t8);
StructuredBuffer<uint> Histogram : register(t9);
StructuredBuffer<float4> PreviousHistogram : register(t10);

StructuredBuffer<float> Scores : register(t12);
StructuredBuffer<float4> MomentA : register(t12);
StructuredBuffer<float4> MomentB : register(t13);
StructuredBuffer<float4> LargeBackground : register(t14);
StructuredBuffer<float4> Proto : register(t15);

RWStructuredBuffer<uint> UintOut : register(u0);
RWStructuredBuffer<float4> Float4Out : register(u0);
RWStructuredBuffer<float4> SecondOut : register(u1);
RWStructuredBuffer<float> FloatOut : register(u0);

uint2 NativeSize()
{
    return Box.zw - Box.xy;
}
uint Pack(uint3 rgb)
{
    return rgb.x + (rgb.y << 8) + (rgb.z << 16);
}
float3 Unpack(uint rgb)
{
    return float3(rgb & 255u, (rgb >> 8) & 255u, (rgb >> 16) & 255u);
}
float3 Source8(int2 p)
{
    uint2 local = uint2(clamp(p, 0, int2(NativeSize()) - 1));
    return floor(saturate(Source.Load(int3(local + Box.xy, 0)).rgb) * 255 + .5f);
}
uint At(int2 p, uint width, uint height)
{
    uint2 xy = uint2(clamp(p, 0, int2(width, height) - 1));
    return xy.y * width + xy.x;
}
struct Taps
{
    uint4 i;
    float2 weight;
};
Taps GetTaps(uint2 pixel, uint h, uint height, uint width)
{
    uint2 first, second;
    float2 f;
    CanonicalNativeAxis(pixel.x, h, height, width, first.x, second.x, f.x);
    CanonicalNativeAxis(pixel.y, h, height, height, first.y, second.y, f.y);
    Taps taps;
    taps.i = uint4(first.y * width + first.x, first.y * width + second.x, second.y * width + first.x,
                   second.y * width + second.x);
    taps.weight = f;
    return taps;
}
float4 Blend(float4 a, float4 b, float4 c, float4 d, float2 f)
{
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}
float4 Lift(StructuredBuffer<float4> field, Taps t)
{
    return Blend(field[t.i.x], field[t.i.y], field[t.i.z], field[t.i.w], t.weight);
}

[numthreads(16, 8, 1)] void pack_rgb(uint3 id : SV_DispatchThreadID) {
    uint height = StageValue, width = height == 28u ? Width28 : Width84;
    if (id.x >= width || id.y >= height)
        return;
    UintOut[id.y * width + id.x] = Pack(CanonicalRgb(Source, Box, id.xy, height));
}[numthreads(16, 8, 1)] void coarse(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width28 || id.y >= 28u)
        return;
    float4 peak = 0, sum = 0;
    [unroll] for (int j = -2; j <= 2; ++j)[unroll] for (int i = -2; i <= 2; ++i)
    {
        float4 value = Roles[At(int2(id.xy) + int2(i, j), Width28, 28)];
        peak = max(peak, value);
        sum += value;
    }
    Float4Out[id.y * Width28 + id.x] = peak;
    SecondOut[id.y * Width28 + id.x] = sum / 25;
}
[numthreads(16, 8, 1)] void make_hist(uint3 id : SV_DispatchThreadID) {
    uint2 size = NativeSize();
    uint step = max(1u, uint(round(float(size.y) / 96)));
    uint2 p = id.xy * step;
    if (any(p >= size))
        return;
    uint3 bin = uint3(round(Source8(int2(p)) / 16));
    uint base = (bin.x * 289 + bin.y * 17 + bin.z) * 4;
    float4 roles = Lift(Roles, GetTaps(p, size.y, 28, Width28));
    uint4 v = uint4(round(saturate(roles) * HistScale));
    [unroll] for (uint k = 0; k < 4; ++k) InterlockedAdd(UintOut[base + k], v[k]);
} uint PalIndex(int3 p)
{
    return uint(p.x * 289 + p.y * 17 + p.z);
}
[numthreads(128, 1, 1)] void smooth_hist(uint3 id : SV_DispatchThreadID) {
    uint at = id.x;
    if (at >= 4913)
        return;
    int3 p = int3(at / 289, (at / 17) % 17, at % 17);
    float4 sum = 0;
    [unroll] for (int k = -3; k <= 3; ++k)
    {
        int3 other = p;
        if (StageValue == 0)
            other.x += k;
        else if (StageValue == 1)
            other.y += k;
        else
            other.z += k;
        if (any(other < 0) || any(other >= 17))
            continue;
        uint ix = PalIndex(other);
        float4 v;
        if (StageValue == 0)
        {
            uint start = ix * 4;
            v = float4(Histogram[start], Histogram[start + 1], Histogram[start + 2], Histogram[start + 3]) /
                HistScale;
        }
        else
            v = PreviousHistogram[ix];
        sum += v * Gaussian[k + 3];
    }
    Float4Out[at] = sum;
    if (StageValue == 2)
        SecondOut[at] = (sum + .12f) / (dot(sum, 1.0f) + .48f);
}
// Break exact ties by the first RGB bin, matching the training palette.
groupshared float2 Rank[256];
groupshared uint2 RankId[256];
[numthreads(256, 1, 1)] void choose_colors(uint lane : SV_GroupIndex) {
    float2 best = -1;
    uint2 bestId = 0;
    [loop] for (uint p = lane; p < 4913; p += 256)
    {
        float2 rank = PreviousHistogram[p].yz * Palette[p].yz;
        if (rank.x > best.x)
        {
            best.x = rank.x;
            bestId.x = p;
        }
        if (rank.y > best.y)
        {
            best.y = rank.y;
            bestId.y = p;
        }
    }
    Rank[lane] = best;
    RankId[lane] = bestId;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = 128; stride > 0; stride >>= 1)
    {
        if (lane < stride)
        {
            float2 other = Rank[lane + stride];
            uint2 idx = RankId[lane + stride];
            [unroll] for (uint k = 0; k < 2; ++k) if (other[k] > Rank[lane][k] ||
                                                      (other[k] == Rank[lane][k] && idx[k] < RankId[lane][k]))
            {
                Rank[lane][k] = other[k];
                RankId[lane][k] = idx[k];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0)
    {
        UintOut[0] = RankId[0].x;
        UintOut[1] = RankId[0].y;
    }
} float Range(float3 v)
{
    return max(v.x, max(v.y, v.z));
}
float4 ColorOdds(float3 source)
{
    float3 pos = source / 16;
    int3 first = int3(floor(pos));
    float3 frac = pos - float3(first);
    float4 odds = 0;
    [unroll] for (int i = 0; i < 8; ++i)
    {
        int3 bit = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        float3 weight = lerp(1 - frac, frac, float3(bit));
        odds += Palette[PalIndex(min(first + bit, 16))] * (weight.x * weight.y * weight.z);
    }
    return odds;
}
float3 PalColor(uint i)
{
    return min(255.0f, 16 * float3(i / 289, (i / 17) % 17, i % 17));
}
[numthreads(16, 8, 1)] void pack_tile(uint3 p : SV_DispatchThreadID) {
    if (p.x >= 512 || p.y >= 144)
        return;
    int2 pos = int2(int(StageValue) + int(p.x) - 36, int(p.y) - 56);
    float3 rgb = Unpack(Rgb28[At(pos, Width28, 28)]) / 255;
    uint index = p.y * 512 + p.x;
    uint cells = 512 * 144;
    FloatOut[index] = rgb.x;
    FloatOut[index + cells] = rgb.y;
    FloatOut[index + 2 * cells] = rgb.z;
}[numthreads(16, 8, 1)] void export_roles(uint3 p : SV_DispatchThreadID)
{
    uint x = StageValue + p.x;
    if (x >= Width28 || p.x >= 440 || p.y >= 28)
        return;
    uint i = (p.y + 56) * 512 + p.x + 36, cells = 512 * 144;
    float4 v = float4(Scores[i], Scores[i + cells], Scores[i + 2 * cells], Scores[i + 3 * cells]);
    float4 mass = exp(v - max(max(v.x, v.y), max(v.z, v.w)));
    Float4Out[p.y * Width28 + x] = mass / dot(mass, 1.0f);
}

[numthreads(16, 8, 1)] void make_moments(uint3 p : SV_DispatchThreadID) {
    if (p.x >= Width84 || p.y >= 84)
        return;
    uint idx = p.y * Width84 + p.x;
    float3 color = Unpack(Rgb84[idx]) / 255;
    float bg = Lift(Roles, GetTaps(p.xy, 84, 28, Width28)).x;
    float mass = .015f + bg * bg * bg;
    Float4Out[idx] = float4(color * mass, mass);
    SecondOut[idx] = float4(color * color * mass, 0);
}[numthreads(16, 8, 1)] void smooth_moments(uint3 p : SV_DispatchThreadID)
{
    if (p.x >= Width84 || p.y >= 84)
        return;
    bool vertical = (StageValue & 64u) != 0;
    int radius = int(StageValue & 63u);
    float4 a = 0, b = 0;
    [loop] for (int i = -radius; i <= radius; ++i)
    {
        uint ix = At(int2(p.xy) + (vertical ? int2(0, i) : int2(i, 0)), Width84, 84);
        a += MomentA[ix];
        b += MomentB[ix];
    }
    uint at = p.y * Width84 + p.x;
    if (vertical)
    {
        float3 mu = a.xyz / a.w;
        float3 variance = max(0, b.xyz / a.w - mu * mu);
        Float4Out[at] = float4(mu, sqrt(dot(variance, 1.0f / 3)));
    }
    else
    {
        float n = 2 * radius + 1;
        Float4Out[at] = a / n;
        SecondOut[at] = b / n;
    }
}
groupshared float4 TempFill[128], TempOutline[128];
[numthreads(128, 1, 1)] void prototypes(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    const uint count = Width84 * 84;
    uint n = (count + 127) / 128;
    uint part = group.x + 65535u * group.y;
    float4 a = 0, b = 0;
    if (StageValue == 0)
    {
        uint i = part * 128 + lane;
        if (i < count)
        {
            float2 weights = Lift(Roles, GetTaps(uint2(i % Width84, i / Width84), 84, 28, Width28)).yz;
            float4 rgb = float4(Unpack(Rgb84[i]) / 255, 1);
            a = rgb * (weights.x * weights.x);
            b = rgb * (weights.y * weights.y);
        }
    }
    else
    {
        [loop] for (uint i = lane; i < n; i += 128)
        {
            a += Proto[2 * i];
            b += Proto[2 * i + 1];
        }
    }
    TempFill[lane] = a;
    TempOutline[lane] = b;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = 64; stride > 0; stride >>= 1)
    {
        if (lane < stride)
        {
            TempFill[lane] += TempFill[lane + stride];
            TempOutline[lane] += TempOutline[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0 && part < n)
    {
        float4 fill = TempFill[0], outline = TempOutline[0];
        Float4Out[2 * part] = StageValue == 0 ? fill : fill / (fill.w + 1e-5f);
        Float4Out[2 * part + 1] = StageValue == 0 ? outline : outline / (outline.w + 1e-5f);
    }
}

float2 InkMixEvidence(float3 color, float3 ink, float4 background)
{
    float3 inkOffset = ink - background.xyz;
    float3 pixelOffset = color - background.xyz;
    float contrast = dot(inkOffset, inkOffset);
    float noise = .025f + background.w;
    float fraction = saturate(dot(pixelOffset, inkOffset) / (contrast + .0001f));
    float3 residual = pixelOffset - fraction * inkOffset;
    float reliability = contrast / (contrast + 3.0f * noise * noise);
    return float2(fraction, min(6.0f, sqrt(dot(residual, residual) / 3.0f) / noise)) * reliability;
}

[numthreads(16, 8, 1)] void decode_palette(uint3 id : SV_DispatchThreadID)
{
    uint2 size = NativeSize();
    if (any(id.xy >= size))
        return;
    int2 p = int2(id.xy);
    float3 rgb = Source8(p), color = rgb / 255;
    Taps low = GetTaps(id.xy, size.y, 28, Width28), hi = GetTaps(id.xy, size.y, 84, Width84);
    float4 role = Lift(Roles, low);
    float2 peak = Lift(LocalMax, low).yz;
    float3 avg = Lift(LocalMean, low).yzw;
    float3 soft = Blend(float4(Unpack(Rgb84[hi.i.x]), 0), float4(Unpack(Rgb84[hi.i.y]), 0),
                        float4(Unpack(Rgb84[hi.i.z]), 0), float4(Unpack(Rgb84[hi.i.w]), 0), hi.weight)
                      .rgb /
                  255;
    float4 nearby = Lift(NearbyBackground, hi), wide = Lift(LargeBackground, hi);
    float3 odds = ColorOdds(rgb).yzw;
    // Encoder roles and context, native RGB detail, two estimates of background,
    // shared line colors, local gradients and color posterior, in training order.
    float f[50];
    [unroll] for (int i = 0; i < 4; ++i) f[i] = role[i];
    f[4] = peak.x;
    f[5] = peak.y;
    [unroll] for (int j = 0; j < 3; ++j)
    {
        f[6 + j] = avg[j];
        f[9 + j] = color[j];
        f[12 + j] = (color[j] - soft[j]) * 6;
        f[15 + j] = (color[j] - nearby[j]) * 5;
        f[18 + j] = (color[j] - nearby[j]) / (.025f + nearby.w);
        f[22 + j] = (color[j] - wide[j]) * 5;
        f[25 + j] = (color[j] - wide[j]) / (.025f + wide.w);
        f[29 + j] = (color[j] - Proto[0][j]) * 3;
        f[32 + j] = (color[j] - Proto[1][j]) * 3;
        f[37 + j] = odds[j];
    }
    f[21] = nearby.w * 5;
    f[28] = wide.w * 5;
    int step = max(1, int(round(float(size.y) / 84)));
    f[35] = Range(abs(Source8(p + int2(step, 0)) - Source8(p - int2(step, 0)))) / 255;
    f[36] = Range(abs(Source8(p + int2(0, step)) - Source8(p - int2(0, step)))) / 255;
    f[40] = dot(abs(rgb - PalColor(Winners[0])), 1.0f / 192.0f);
    f[41] = dot(abs(rgb - PalColor(Winners[1])), 1.0f / 192.0f);
    [unroll] for (int k = 0; k < 42; ++k) f[k] = clamp(f[k], -12, 12);
    // Compare the observed pixel with blends of each recurring ink and the nearby background.
    // Low contrast reduces the influence of both the opacity estimate and its color error.
    float3 fillInk = PalColor(Winners[0]) / 255.0f;
    float3 outlineInk = PalColor(Winners[1]) / 255.0f;
    float2 nearFill = InkMixEvidence(color, fillInk, nearby);
    float2 nearOutline = InkMixEvidence(color, outlineInk, nearby);
    float2 wideFill = InkMixEvidence(color, fillInk, wide);
    float2 wideOutline = InkMixEvidence(color, outlineInk, wide);
    f[42] = nearFill.x;    f[43] = nearFill.y;
    f[44] = nearOutline.x; f[45] = nearOutline.y;
    f[46] = wideFill.x;    f[47] = wideFill.y;
    f[48] = wideOutline.x; f[49] = wideOutline.y;
    FloatOut[id.y * size.x + id.x] = PaletteMask(f);
}
