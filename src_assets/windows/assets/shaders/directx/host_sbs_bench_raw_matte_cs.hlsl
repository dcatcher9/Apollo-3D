/* File-fed benchmark only. Spatial RawMatte, frozen alpha8 cutoff, current source authority.
 * The GPU joins the actual Rule receipt, C24/SCT, OCR8 and condition tuple.
 */
cbuffer ModelRequest : register(b0) {
    uint4 Shape; // native band width/height, cutoff, reserved
    uint4 Owner; // frame lo/hi, analysis generation lo/hi
    uint4 ClockToken; // observation clock lo/hi, transaction token lo/hi
}
cbuffer ActualContour : register(b1) {
    uint4 Source; uint4 Field; uint4 Content; uint4 Model; uint4 Observation; uint4 Options;
}
Texture2D<float4> RGB : register(t0);
Texture2D<uint> OracleFacts : register(t1);
Texture2D<uint> CoverBits : register(t2);
Texture2D<uint> Body : register(t3);
Texture2D<uint> SelectedBits : register(t4);
Texture2D<uint> RawM : register(t5);
StructuredBuffer<float> Weights : register(t6);
ByteAddressBuffer Work : register(t7);
Texture2D<uint> Geometry : register(t8);
Texture2D<uint> Trusted : register(t9);
Texture2D<uint> Rule : register(t10);
StructuredBuffer<uint> RuleReceipt : register(t11);
StructuredBuffer<uint> OCR : register(t12);
StructuredBuffer<uint> SCT : register(t13);
StructuredBuffer<uint> Condition : register(t14);
Texture2D<uint> Candidate : register(t15);
RWTexture2D<uint> OutputMask : register(u0);

RWByteAddressBuffer WorkStore : register(u2);
RWTexture2D<uint> OutputAlpha8 : register(u3);
RWTexture2D<uint> GeometryStore : register(u4);
RWStructuredBuffer<uint> Receipt : register(u5);
RWByteAddressBuffer Evidence : register(u6);

static const uint TileX = 16u, TileY = 8u;
static const uint StemX = 24u, StemY = 16u, SumX = 22u, SumY = 14u;
static const uint HaloX = 22u, HaloY = 14u;
static const uint RgbValid = 1u << 24u, RawBit = 1u << 25u, HighBit = 1u << 26u;
static const uint StemWeight = 0u, StemBias = 96u;
static const uint LocalWeight = 104u, LocalBias = 176u;
static const uint WideWeight = 184u, WideBias = 256u;
static const uint JoinWeight = 264u, JoinBias = 392u;
static const uint ReadoutWeight = 400u, ReadoutBias = 432u;
static const uint Rgb8Lookup = 436u, CountLookup = 692u;

groupshared uint Tile[22u * 14u];
groupshared uint Rows[2u * 14u];
groupshared float Stem[24u * 16u * 8u];
groupshared float Sum[22u * 14u * 8u];

bool InBounds(int2 p) { return all(p >= 0) && all(uint2(p) < Shape.xy); }
bool Active(uint i) { return (Work.Load(68u) & (1u << i)) != 0u; }
uint CurrentSlots(int2 p) { return CoverBits.Load(int3(p, 0)) & Work.Load(68u); }

uint4 PaddedCover(uint4 cover, uint margin) {
    return uint4(uint2(max(int2(cover.xy) - int(margin), int2(0, 0))),
                 min(cover.zw + margin, Shape.xy));
}
bool TileIntersects(uint4 tile, uint margin) {
    [unroll] for (uint i = 0u; i < 8u; ++i) {
        if (Active(i)) {
            uint4 cover = PaddedCover(Work.Load4(80u + 16u * i), margin);
            if (all(tile.xy < cover.zw) && all(cover.xy < tile.zw)) return true;
        }
    }
    return false;
}
uint4 Args(uint4 roi) {
    return uint4((roi.zw - roi.xy + uint2(TileX - 1u, TileY - 1u)) / uint2(TileX, TileY), 1u, 0u);
}

uint CBWord(uint i) {
    const uint4 words[6] = {Source, Field, Content, Model, Observation, Options};
    return words[i / 4u][i % 4u];
}
uint ProjectCenter(uint boundary, uint start, uint extent, uint span) {
    // For an in-content boundary, ceil((2*(boundary-start)*extent-span)/(2*span)).
    // The numerator can be negative only for a boundary whose native result is zero.
    uint twice=2u*(boundary-start)*extent;
    return twice<=span?0u:(twice+span-1u)/(2u*span);
}
uint4 Rect(uint offset) { return uint4(OCR[offset], OCR[offset+1u], OCR[offset+2u], OCR[offset+3u]); }
uint4 NativeCover(uint4 r) {
    uint dx=Content.z-Content.x, dy=Content.w-Content.y;
    uint top=ProjectCenter(r.y,Content.y,Source.y,dy);
    uint bottom=ProjectCenter(r.w,Content.y,Source.y,dy);
    return min(uint4(ProjectCenter(r.x,Content.x,Source.x,dx),
                     top>Source.z?top-Source.z:0u,
                     ProjectCenter(r.z,Content.x,Source.x,dx),
                     bottom>Source.z?bottom-Source.z:0u), uint4(Shape.xy,Shape.xy));
}
bool RuleCurrent() {
    if (any(Source.xw != Shape.xy) || any(Shape.xy==0u) || any(Source.xy>5120u) || Source.z>=Source.y ||
        Source.w!=Source.y-Source.z || any(Model!=Owner) || any(Observation.xy!=ClockToken.xy) ||
        all(Owner.xy==0u) || all(ClockToken.xy==0u) || all(ClockToken.zw==0u) ||
        Observation.z>1u || Observation.w!=0u || any(Options!=uint4(1u,27u,0u,0u))) return false;
    if (any(Field.xy==0u) || any(Field.xy>5120u) || Field.z>=Field.w || Field.w>Field.y ||
        any(Content.xy>=Content.zw) || any(Content.zw>Field.xy)) return false;
    if (any(uint4(RuleReceipt[0],RuleReceipt[1],RuleReceipt[2],RuleReceipt[3])!=uint4(0x31534d53u,1u,3u,0u)) ||
        any(uint4(RuleReceipt[60],RuleReceipt[61],RuleReceipt[62],RuleReceipt[63])!=uint4(ClockToken.zw,2u,0u))) return false;
    if (SCT[0]!=5u || SCT[1]!=0x35544353u || SCT[2]>1u || SCT[22]!=27u || SCT[23]!=1u ||
        any(uint4(SCT[4],SCT[5],SCT[6],SCT[7])!=Model) ||
        any(uint4(SCT[8],SCT[9],SCT[10],SCT[11])!=Source) ||
        any(uint4(SCT[12],SCT[13],SCT[14],SCT[15])!=Field) ||
        any(uint4(SCT[16],SCT[17],SCT[18],SCT[19])!=Content) ||
        any(uint2(SCT[20],SCT[21])!=Observation.xy)) return false;
    [loop] for (uint i=0u; i<24u; ++i) if (RuleReceipt[4u+i]!=CBWord(i)) return false;
    [loop] for (uint j=0u; j<32u; ++j) if (RuleReceipt[28u+j]!=SCT[j] || (j>=24u && SCT[j]!=0u)) return false;
    return true;
}

bool OcrCurrent(uint count) {
    if (OCR[0]!=4u || OCR[1]!=0x3852434fu || OCR[2]!=1u || OCR[3]!=count || count>8u || OCR[15]!=0u ||
        any(uint4(OCR[5],OCR[6],OCR[7],OCR[8])!=Model) ||
        any(uint2(OCR[9],OCR[10])!=Source.xy) ||
        any(uint4(OCR[11],OCR[12],OCR[13],OCR[14])!=Field)) return false;
    if (Condition[0]!=4u || Condition[1]!=0x34504353u || Condition[2]!=count ||
        (Condition[3] & ~((1u<<count)-1u))!=0u) return false;
    return count==0u ? SCT[2]==0u : SCT[2]==1u && Condition[4]==2u;
}

[numthreads(1, 1, 1)]
void prepare_main() {
    [unroll] for (uint start=0u; start<240u; start+=16u) WorkStore.Store4(start, 0u);
    Evidence.Store4(0u,0u);
    WorkStore.Store(76u,Shape.z);
    if (Shape.w != 0u) { WorkStore.Store(72u,5u); return; }
    if (!RuleCurrent()) { WorkStore.Store(72u,1u); return; }
    uint count = OCR[4];
    if (!OcrCurrent(count)) { WorkStore.Store(72u,2u); return; }
    if (count == 0u) { WorkStore.Store(212u,1u); return; }
    uint4 roi = uint4(Shape.xy,0u,0u);
    // Invert the native pixel-center test used by CBC for the authenticated final OCR cover.
    [loop] for (uint slot=0u; slot<count; ++slot) {
        uint4 core = Rect(16u+8u*slot);
        uint4 raw = Rect(144u+8u*slot);
        if (any(core.xy>=core.zw) || any(raw.xy>=raw.zw) || any(raw.xy<Content.xy) ||
            any(raw.zw>Content.zw) || any(core.xy<raw.xy) || any(core.zw>raw.zw)) {
            WorkStore.Store(72u,3u); return;
        }
        uint4 box = NativeCover(raw);
        if (any(box.xy>=box.zw)) { WorkStore.Store(72u,3u); return; }
        WorkStore.Store4(80u+16u*slot,box);
        roi = uint4(min(roi.xy,box.xy),max(roi.zw,box.zw));
    }
    uint4 padded=PaddedCover(roi,4u);
    WorkStore.Store4(0u,Args(roi));
    WorkStore.Store4(16u,roi);
    WorkStore.Store4(32u,Args(padded));
    WorkStore.Store4(48u,padded);
    WorkStore.Store(68u,(1u<<count)-1u);
    WorkStore.Store(212u,2u);
}

void Fault(uint bit) {
    uint previous;
    Evidence.InterlockedOr(8u,bit,previous);
}
uint3 RgbCode(int2 p) {
    float4 rgba=RGB.Load(int3(p,0));
    bool valid=rgba.a==1.0f && all(isfinite(rgba.rgb)) && all(rgba.rgb>=0.0f) && all(rgba.rgb<=1.0f);
    if (!valid) Fault(1u);
    float3 safe=valid?rgba.rgb:float3(0.0f,0.0f,0.0f);
    return uint3(floor(safe*255.0f+0.5f));
}
uint PackTile(int2 p) {
    if (!InBounds(p)) return 0u;
    uint3 code = RgbCode(p);
    uint raw = RawM.Load(int3(p,0)), owners=CoverBits.Load(int3(p,0));
    uint selected=SelectedBits.Load(int3(p,0));
    if (raw>1u || (owners & ~Work.Load(68u))!=0u || (selected>>16u)!=0u ||
        ((selected&255u) & ~owners)!=0u) Fault(2u);
    uint flags = (raw == 1u ? RawBit : 0u) |
                 (((SelectedBits.Load(int3(p, 0)) >> 8u) & CurrentSlots(p)) != 0u ? HighBit : 0u);
    return code.r | (code.g << 8u) | (code.b << 16u) | RgbValid | flags;
}

uint Distance(uint base, uint2 p) {
    uint c = Rows[base + p.y + 3u];
    if ((c & (1u << (p.x + 3u))) != 0u) return 0u;
    c |= Rows[base + p.y + 2u] | Rows[base + p.y + 4u];
    if ((c & (7u << (p.x + 2u))) != 0u) return 1u;
    c |= Rows[base + p.y + 1u] | Rows[base + p.y + 5u];
    if ((c & (31u << (p.x + 1u))) != 0u) return 2u;
    c |= Rows[base + p.y] | Rows[base + p.y + 6u];
    return (c & (127u << p.x)) != 0u ? 3u : 4u;
}

uint DeltaRgb(uint a, uint b) {
    if (((a & b) & RgbValid) == 0u) return 0u;
    int3 ca = int3(a & 255u, (a >> 8u) & 255u, (a >> 16u) & 255u);
    int3 cb = int3(b & 255u, (b >> 8u) & 255u, (b >> 16u) & 255u);
    int3 dif = abs(ca - cb);
    return uint(max(max(dif.r, dif.g), dif.b));
}

[numthreads(TileX, TileY, 1)]
void features_main(uint3 tile : SV_GroupID, uint3 thread : SV_GroupThreadID, uint lane : SV_GroupIndex) {
    if (lane==0u) { uint previous; Evidence.InterlockedAdd(0u,1u,previous); }
    uint4 roi = Work.Load4(48u);
    uint2 begin = roi.xy + tile.xy * uint2(TileX, TileY);
    if (!TileIntersects(uint4(begin, min(begin + uint2(TileX, TileY), roi.zw)), 4u)) return;
    for (uint n = lane; n < HaloX * HaloY; n += TileX * TileY) {
        Tile[n] = PackTile(int2(begin) + int2(n % HaloX, n / HaloX) - 3);
    }
    GroupMemoryBarrierWithGroupSync();

    if (lane < HaloY) {
        uint raw = 0u, high = 0u;
        [unroll] for (uint x = 0u; x < HaloX; ++x) {
            uint word = Tile[lane * HaloX + x];
            raw |= ((word & RawBit) != 0u ? 1u : 0u) << x;
            high |= ((word & HighBit) != 0u ? 1u : 0u) << x;
        }
        Rows[lane] = raw;
        Rows[lane + HaloY] = high;
    }
    GroupMemoryBarrierWithGroupSync();

    uint2 t = thread.xy;
    uint2 p = begin + t;
    if (any(p >= roi.zw)) return;
    uint take = 127u << t.x;
    uint count = 0u;
    [unroll] for (uint y = 0u; y < 7u; ++y) count += countbits(Rows[t.y + y] & take);
    uint at = (t.y + 3u) * HaloX + t.x + 3u;
    uint query = Tile[at];
    uint dx = max(DeltaRgb(query, Tile[at - 1u]), DeltaRgb(query, Tile[at + 1u]));
    uint dy = max(DeltaRgb(query, Tile[at - HaloX]), DeltaRgb(query, Tile[at + HaloX]));
    GeometryStore[p] = Distance(0u, t) | (Distance(HaloY, t) << 3u) | (count << 6u) |
                       (dx << 12u) | (dy << 20u) | (1u << 31u);
}

void ReadInput(int2 p, out float feature[12]) {
    uint3 code = RgbCode(p);
    feature[0] = Weights[Rgb8Lookup + code.r];
    feature[1] = Weights[Rgb8Lookup + code.g];
    feature[2] = Weights[Rgb8Lookup + code.b];
    uint owners = CurrentSlots(p);
    feature[3] = owners != 0u ? 1.0f : 0.0f;
    feature[4] = Body.Load(int3(p, 0)) != 0u ? 1.0f : 0.0f;
    feature[5] = ((SelectedBits.Load(int3(p, 0)) >> 8u) & owners) != 0u ? 1.0f : 0.0f;
    feature[6] = RawM.Load(int3(p, 0)) != 0u ? 1.0f : 0.0f;
    uint packed = Geometry.Load(int3(p, 0));
    feature[7] = float(packed & 7u) * 0.25f;
    feature[8] = float((packed >> 3u) & 7u) * 0.25f;
    feature[9] = Weights[CountLookup + ((packed >> 6u) & 63u)];
    feature[10] = Weights[Rgb8Lookup + ((packed >> 12u) & 255u)];
    feature[11] = Weights[Rgb8Lookup + ((packed >> 20u) & 255u)];
}

[numthreads(TileX, TileY, 1)]
void score_main(uint3 tile : SV_GroupID, uint3 thread : SV_GroupThreadID, uint lane : SV_GroupIndex) {
    if (lane==0u) { uint previous; Evidence.InterlockedAdd(4u,1u,previous); }
    uint4 roi = Work.Load4(16u);
    uint2 begin = roi.xy + tile.xy * uint2(TileX, TileY);
    if (!TileIntersects(uint4(begin, min(begin + uint2(TileX, TileY), roi.zw)), 0u)) return;

    for (uint cell = lane; cell < StemX * StemY; cell += TileX * TileY) {
        int2 p = int2(begin) + int2(cell % StemX, cell / StemX) - 4;
        bool valid = InBounds(p);
        float feature[12];
        if (valid) ReadInput(p, feature);
        [unroll] for (uint k = 0u; k < 8u; ++k) {
            precise float x = 0.0f;
            if (valid) {
                [unroll] for (uint c = 0u; c < 12u; ++c) x += feature[c] * Weights[StemWeight + k * 12u + c];
                x = max(0.0f, x + Weights[StemBias + k]);
            }
            Stem[cell * 8u + k] = x;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint sum_cell = lane; sum_cell < SumX * SumY; sum_cell += TileX * TileY) {
        uint2 t = uint2(sum_cell % SumX, sum_cell / SumX);
        bool valid = InBounds(int2(begin + t) - 3);
        uint center = ((t.y + 1u) * StemX + t.x + 1u) * 8u;
        [unroll] for (uint k = 0u; k < 8u; ++k) {
            precise float x = 0.0f;
            if (valid) {
                [unroll] for (uint y = 0u; y < 3u; ++y) {
                    [unroll] for (uint x1 = 0u; x1 < 3u; ++x1) {
                        uint at = ((t.y + y) * StemX + t.x + x1) * 8u + k;
                        x += Stem[at] * Weights[LocalWeight + k * 9u + y * 3u + x1];
                    }
                }
                x = Stem[center + k] + max(0.0f, x + Weights[LocalBias + k]);
            }
            Sum[sum_cell * 8u + k] = x;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint2 p = begin + thread.xy;
    if (any(p >= roi.zw) || CurrentSlots(int2(p)) == 0u) return;
    OutputMask[p] = 0u;
    uint trusted=Trusted.Load(int3(p,0));
    if (trusted>1u || Body.Load(int3(p,0))>1u) Fault(4u);
    if (trusted == 1u || (OracleFacts.Load(int3(p, 0)) & (1u << 11u)) != 0u) {
        OutputMask[p] = 1u;
        return;
    }
    uint packed = Geometry.Load(int3(p, 0));
    if ((packed & 0x80000000u) == 0u) { Fault(8u); return; }
    if ((packed & 7u) > 3u) return;
    uint center = ((thread.y + 3u) * SumX + thread.x + 3u) * 8u;
    float readout[16];
    [unroll] for (uint c = 0u; c < 8u; ++c) {
        precise float wide = 0.0f;
        [unroll] for (uint y = 0u; y < 3u; ++y) {
            [unroll] for (uint x = 0u; x < 3u; ++x) {
                uint at = ((thread.y + 3u*y) * SumX + thread.x + 3u*x) * 8u + c;
                wide += Sum[at] * Weights[WideWeight + c * 9u + y * 3u + x];
            }
        }
        readout[c] = Sum[center + c];
        readout[8u + c] = max(0.0f, wide + Weights[WideBias + c]);
    }
    float joined[8];
    [unroll] for (uint j = 0u; j < 8u; ++j) {
        precise float value = 0.0f;
        [unroll] for (uint f = 0u; f < 16u; ++f) value += readout[f] * Weights[JoinWeight + j * 16u + f];
        joined[j] = max(0.0f, value + Weights[JoinBias + j]);
    }
    precise float4 logits;
    [unroll] for (uint h = 0u; h < 4u; ++h) {
        precise float value = 0.0f;
        [unroll] for (uint a = 0u; a < 8u; ++a) value += joined[a] * Weights[ReadoutWeight + h * 8u + a];
        logits[h] = value + Weights[ReadoutBias + h];
    }
    float high = max(max(logits.x, logits.y), max(logits.z, logits.w));
    float4 e = exp(logits - high);
    precise float alpha = 1.0f - e.x / (e.x + e.y + e.z + e.w);
    precise float scaled = saturate(alpha) * 255.0f;
    uint alpha8 = uint(round(scaled)); // round_ne, inclusive threshold after quantization
    OutputAlpha8[p] = alpha8;
    OutputMask[p] = alpha8 >= Shape.z ? 1u : 0u;
}




bool CoveredForOutput(uint2 p, uint slots) {
    uint owners = CoverBits.Load(int3(p,0)) & slots;
    [unroll] for (uint i=0u;i<8u;++i) {
        if ((owners & (1u<<i)) != 0u) {
            uint4 box=WorkStore.Load4(80u+16u*i);
            if (all(p>=box.xy)&&all(p<box.zw)) return true;
        }
    }
    return false;
}

[numthreads(TileX,TileY,1)]
void publish_main(uint3 id:SV_DispatchThreadID) {
    if (any(id.xy>=Shape.xy)) return;
    uint planned=WorkStore.Load(212u);
    uint2 geometry=WorkStore.Load2(32u), score=WorkStore.Load2(0u);
    uint3 completed=Evidence.Load3(0u);
    bool pipeline=all(completed==uint3(geometry.x*geometry.y,score.x*score.y,0u));
    uint accepted=planned==1u || (planned==2u && pipeline) ? planned : 0u;
    uint selected=Rule.Load(int3(id.xy,0));
    [branch] if (accepted!=0u) {
        selected=0u;
        if (accepted==2u && CoveredForOutput(id.xy,WorkStore.Load(68u)))
            selected=Candidate.Load(int3(id.xy,0));
    }
    OutputMask[id.xy]=selected;
    if (accepted!=2u) OutputAlpha8[id.xy]=0u;
    if (all(id.xy==0u)) {
        [loop] for (uint i=0u;i<64u;++i) Receipt[i]=RuleReceipt[i];
        if (accepted!=0u) { Receipt[2u]=4u; Receipt[62u]=3u; }
        WorkStore.Store(64u,accepted);
        if (planned==2u && !pipeline) WorkStore.Store(72u,4u);
        WorkStore.Store3(208u+8u,completed); // actual geometry groups, score groups, fault bits
    }
}


