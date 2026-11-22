// Phase-aware RGB tile and q8 resolve for the supported fixed score profiles.
#define TileCapacity 6
// The full-resolution heads depend on at most 36 RGB pixels on either side,
// across all four phases of the model's quarter-resolution pooling grid.
#define Halo 36
// With four-pixel aligned origins, the serialized model has
// the full input domain on every index [33, extent - 33).
#define RescueHalo 34
// CUDA consumes only the four header words. The first tile's reserved word
// identifies the lattice of every tile in this observation.
#define ScaleWord 10u

cbuffer Geometry : register(b0) { uint4 Band; uint4 Tile; }
Texture2D<float4> SourceRGB : register(t0);
StructuredBuffer<float> LogScores : register(t0);
StructuredBuffer<int> Placement : register(t2);
RWStructuredBuffer<int> PlacementOut : register(u0);
RWStructuredBuffer<int> ResolveProof : register(u1);
RWByteAddressBuffer Work : register(u1);
RWStructuredBuffer<float> RGBTensor : register(u0);
RWTexture2D<uint> Candidate : register(u0);
RWByteAddressBuffer PackResolveArgs : register(u2);

int2 SnapDown(int2 v, int2 phase) { return ((v + phase) & ~3) - phase; }

bool SupportedTile() {
    return Tile.z <= 1u && Tile.w == 0u && (all(Tile.xy == uint2(512u, 144u)) ||
                                 all(Tile.xy == uint2(384u, 104u)));
}

bool ValidScale(int scale) {
    return scale == 0 || (Tile.z == 1u && (scale == 2 || scale == 4));
}

int2 ScaleSize(int2 size, int scale) { return (size + scale - 1) / scale; }

int4 ProjectCover(int4 box, int2 size, int scale) {
    if (scale == 1) return box;
    // Include both pixel-center bilinear taps at either end of the native interval.
    return int4(max(box.xy / scale - 1, 0), min(ScaleSize(box.zw, scale) + 1, size));
}

void SetTile(uint tile, int2 origin, int2 lo, int2 hi) {
    uint at = 4u + 8u * tile;
    PlacementOut[at + 0u] = origin.x;
    PlacementOut[at + 1u] = origin.y;
    PlacementOut[at + 2u] = lo.x;
    PlacementOut[at + 3u] = lo.y;
    PlacementOut[at + 4u] = hi.x;
    PlacementOut[at + 5u] = hi.y;
}

bool IntersectsCover(int2 lo, int2 hi, uint slots, int2 size, int scale) {
    [loop] for (uint i = 0u; i < slots; ++i) {
        int4 box = ProjectCover(int4(Work.Load4(16u + 16u * i)), size, scale);
        if (all(lo < box.zw) && all(box.xy < hi)) return true;
    }
    return false;
}

uint PlanTiles(int2 minBox, int2 maxBox, int2 size, int2 phase, int2 bandBegin, int2 bandEnd,
               uint slots, int2 tileSize, int scale, int halo) {
    if (any(bandEnd - bandBegin < tileSize)) return 0u;
    int2 lastOrigin = bandEnd - tileSize;
    uint count = 0u;
    int y = minBox.y;
    [loop] while (y < maxBox.y) {
        int originY = clamp(SnapDown(int2(0, y - halo), phase).y, bandBegin.y, lastOrigin.y);
        int endY = min(maxBox.y, originY == lastOrigin.y ? size.y : originY + tileSize.y - halo);
        if (endY <= y) return 0u;
        int x = minBox.x;
        [loop] while (x < maxBox.x) {
            int2 lo = int2(x, y);
            int originX = clamp(SnapDown(int2(x - halo, 0), phase).x, bandBegin.x, lastOrigin.x);
            int2 hi = int2(min(maxBox.x, originX == lastOrigin.x ? size.x : originX + tileSize.x - halo), endY);
            if (hi.x <= x) return 0u;
            if (IntersectsCover(lo, hi, slots, size, scale)) {
                if (count >= TileCapacity) return 0u;
                SetTile(count, int2(originX, originY), lo, hi);
                ++count;
            }
            x = hi.x;
        }
        y = endY;
    }
    return count;
}

int2 ModelPhase(int scale) { return int2(0, int((Band.z / uint(scale)) & 3u)); }

bool ModelBandFits(int scale) {
    int2 phase = ModelPhase(scale);
    int2 size = ScaleSize(int2(Band.xy), scale);
    return all(SnapDown(size + 3, phase) - SnapDown(0, phase) >= int2(Tile.xy));
}

uint PlanOnLattice(int4 nativeCover, int scale, uint slots, bool padded, int halo) {
    int2 size = ScaleSize(int2(Band.xy), scale);
    int4 cover = ProjectCover(nativeCover, size, scale);
    int2 minBox = cover.xy, maxBox = cover.zw;
    int2 phase = ModelPhase(scale);
    int2 bandBegin = SnapDown(0, phase), bandEnd = SnapDown(size + 3, phase);
    int2 tileSize = int2(Tile.xy);
    // Extend only the model's read lattice. Score windows still use the real extent.
    if (padded) bandEnd = max(bandEnd, bandBegin + tileSize);
    int2 readBegin = SnapDown(max(minBox - halo, 0), phase);
    int2 readEnd = SnapDown(min(maxBox + halo, size) + 3, phase);
    int2 tileBegin = max(bandBegin, readEnd - tileSize);
    int2 tileEnd = min(readBegin, bandEnd - tileSize);
    if (all(tileBegin <= tileEnd)) {
        int2 centered = SnapDown((minBox + maxBox - tileSize) / 2 + 2, phase);
        SetTile(0u, clamp(centered, tileBegin, tileEnd), minBox, maxBox);
        return 1u;
    }
    return PlanTiles(minBox, maxBox, size, phase, bandBegin, bandEnd, slots, tileSize, scale, halo);
}

[numthreads(1, 1, 1)]
void plan_tile() {
    [unroll] for (uint word = 0u; word < 4u + 8u * TileCapacity; ++word) PlacementOut[word] = 0;
    PackResolveArgs.Store4(0u, 0u);
    // CUDA alone writes the completed-tile mask in word 1.
    uint2 modeCount = Work.Load2(0u);
    if (modeCount.x != 2u) return;
    if (modeCount.y == 0u || modeCount.y > 8u || !SupportedTile()) {
        Work.Store(0u, 0u); Work.Store(8u, 4u); return;
    }
    uint2 sourceSize;
    SourceRGB.GetDimensions(sourceSize.x,sourceSize.y);
    if (any(sourceSize != Band.xy)) { Work.Store(0u,0u); Work.Store(8u,5u); return; }
    int2 size = int2(Band.xy);
    int2 minBox = size, maxBox = 0;
    int tallest = 0;
    for (uint i = 0; i < modeCount.y; ++i) {
        int4 box = int4(Work.Load4(16u + 16u*i));
        if (any(box.xy < 0) || any(box.zw > size) || any(box.xy >= box.zw)) {
            Work.Store(0u, 0u); Work.Store(8u, 4u); return;
        }
        minBox = min(minBox, box.xy);
        maxBox = max(maxBox, box.zw);
        tallest = max(tallest, box.w - box.y);
    }
    uint count = 0u;
    bool adaptive = Tile.z == 1u;
    int desired = !adaptive ? 1 : tallest >= 152 ? 4 : tallest >= 80 ? 2 : 1;
    int scale = desired;
    int maxScale = adaptive ? 4 : 1;
    bool rescued = false;
    int4 nativeCover = int4(minBox, maxBox);
    [loop] while (scale > 1 && !ModelBandFits(scale)) scale /= 2;
    [loop] for (; scale <= maxScale; scale *= 2) {
        bool fits = ModelBandFits(scale);
        if (fits) count = PlanOnLattice(nativeCover, scale, modeCount.y, false, Halo);
        if (count != 0u) break;
        if (adaptive && desired == 1 && scale == 1) {
            count = PlanOnLattice(nativeCover, 1, modeCount.y, true, RescueHalo);
            if (count != 0u) { rescued = true; break; }
        }
        if (!fits) break;
    }
    if (count == 0u && adaptive) {
        scale = 4;
        count = PlanOnLattice(nativeCover, scale, modeCount.y, true, RescueHalo);
        rescued = count != 0u;
    }
    if (count == 0u) { Work.Store(0u, 0u); Work.Store(8u, 4u); return; }
    if (scale > 1 || rescued) {
        // An earlier attempt can have filled records before hitting the cap.
        [loop] for (uint word = 4u + count * 8u; word < 4u + 8u * TileCapacity; ++word)
            PlacementOut[word] = 0;
    }
    if (scale > 1) PlacementOut[ScaleWord] = scale;
    PlacementOut[0] = int(count);
    PackResolveArgs.Store4(0u, uint4(Tile.x / 16u, Tile.y / 8u, count, 0u));
}

[numthreads(16, 8, 1)]
void pack_rgb(uint3 p : SV_DispatchThreadID) {
    [branch] if (p.z >= TileCapacity || p.z >= uint(Placement[0])) return;
    uint at = 4u + 8u * p.z;
    uint cells = Tile.x * Tile.y;
    uint id = 3u * cells * p.z + p.y * Tile.x + p.x;
    int scaleWord = Placement[ScaleWord];
    if (!ValidScale(scaleWord)) return;
    int scale = max(1, scaleWord);
    int2 sourceSize = int2(Band.xy);
    int2 q = clamp(int2(p.xy) + int2(Placement[at], Placement[at + 1u]), 0, ScaleSize(sourceSize, scale) - 1);
    float3 rgb;
    [branch] if (scale == 1) {
        rgb = floor(saturate(SourceRGB.Load(int3(q, 0)).rgb) * 255.0f + 0.5f) / 255.0f;
    } else {
        uint3 sum = 0u;
        [loop] for (int y = 0; y < scale; ++y) {
            [loop] for (int x = 0; x < scale; ++x) {
                int2 source = min(q * scale + int2(x, y), sourceSize - 1);
                sum += uint3(floor(saturate(SourceRGB.Load(int3(source, 0)).rgb) * 255.0f + 0.5f));
            }
        }
        uint count = uint(scale * scale);
        rgb = float3((sum + count / 2u) / count) / 255.0f;
    }
    RGBTensor[id] = rgb.r;
    RGBTensor[cells+id] = rgb.g;
    RGBTensor[2u*cells+id] = rgb.b;
}

[numthreads(16, 8, 1)]
void resolve_matte(uint3 p : SV_DispatchThreadID) {
    uint count = uint(ResolveProof[0]);
    [branch] if (count == 0u || count > TileCapacity || p.z >= count ||
                 uint(ResolveProof[1]) != (1u << count) - 1u) return;
    uint at = 4u + 8u * p.z;
    int scaleWord = ResolveProof[ScaleWord];
    if (!ValidScale(scaleWord)) return;
    int2 pixel = int2(p.xy) + int2(ResolveProof[at], ResolveProof[at+1u]);
    if (any(pixel < 0) || any(pixel >= ScaleSize(int2(Band.xy), max(1, scaleWord)))) return;
    // Each new inference saves its predecessor into the next slice. Only the
    // last tile still resides in the fixed TensorRT binding at slice zero.
    uint slice = p.z + 1u < count ? p.z + 1u : 0u;
    uint cells = Tile.x * Tile.y;
    uint id = 4u * cells * slice + p.y * Tile.x + p.x;
    float4 scores = float4(LogScores[id],LogScores[id+cells],LogScores[id+2u*cells],LogScores[id+3u*cells]);
    if (!all(isfinite(scores))) {
        int previous;
        InterlockedOr(ResolveProof[2],1,previous);
        return;
    }
    // Each selected pixel belongs to exactly one window. Only true band edges
    // may put a window inside a tile's context margin.
    if (any(pixel < int2(ResolveProof[at+2u],ResolveProof[at+3u])) ||
        any(pixel >= int2(ResolveProof[at+4u],ResolveProof[at+5u]))) return;
    float4 odds = exp(scores-max(max(scores.x,scores.y),max(scores.z,scores.w)));
    float ink = odds.y+odds.z+odds.w;
    Candidate[pixel] = scaleWord == 0 ? uint(round(255.0f*ink/(odds.x+ink))) : asuint(ink/(odds.x+ink));
}

[numthreads(1, 1, 1)]
void finish_score() {
    if (Work.Load(0u) != 2u) return;
    uint count = uint(Placement[0]);
    int scaleWord = Placement[ScaleWord];
    if (count == 0u || count > TileCapacity || Placement[2] != 0 ||
        Placement[3] != 0 || !ValidScale(scaleWord) || uint(Placement[1]) != (1u << count) - 1u) {
        Work.Store(0u, 0u);
        Work.Store(8u, 4u);
        return;
    }
    // Only the retired response to this plan can authorize a score read.
    Work.Store(12u, uint(max(1, scaleWord)));
}
