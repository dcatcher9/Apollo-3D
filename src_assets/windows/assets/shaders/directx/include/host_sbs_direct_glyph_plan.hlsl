#ifndef HOST_SBS_DIRECT_GLYPH_PLAN_HLSL
#define HOST_SBS_DIRECT_GLYPH_PLAN_HLSL
// Bindings and source-authority joins of the Direct Glyph plan and publish passes.
// The GPU joins the actual root, C24/SCT, OCR8, condition and source tuple before planning.
cbuffer ModelRequest : register(b0) {
    uint4 Shape; // native band width/height, cutoff, reserved
    uint4 Owner; // frame lo/hi, analysis generation lo/hi
    uint4 ClockToken; // observation clock lo/hi, transaction token lo/hi
}
cbuffer ActualContour : register(b1) {
    uint4 Source; uint4 Field; uint4 Content; uint4 Model; uint4 Observation; uint4 Options;
}
ByteAddressBuffer Work : register(t7);
StructuredBuffer<uint> OCR : register(t12);
StructuredBuffer<uint> SCT : register(t13);
StructuredBuffer<uint> Condition : register(t14);
Texture2D<uint> Candidate : register(t15);
RWTexture2D<uint> OutputMask : register(u0);

RWByteAddressBuffer WorkStore : register(u2);
RWTexture2D<uint> OutputAlpha8 : register(u3);
RWStructuredBuffer<uint> Receipt : register(u5);
RWByteAddressBuffer Evidence : register(u6);

static const uint TileX = 16u, TileY = 8u;

uint4 PaddedCover(uint4 cover, uint margin) {
    return uint4(uint2(max(int2(cover.xy) - int(margin), int2(0, 0))),
                 min(cover.zw + margin, Shape.xy));
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

bool OcrCurrent(uint count) {
    if (OCR[0]!=4u || OCR[1]!=0x3852434fu || OCR[2]!=1u || OCR[3]!=count || count>8u || OCR[15]!=0u ||
        any(uint4(OCR[5],OCR[6],OCR[7],OCR[8])!=Model) ||
        any(uint2(OCR[9],OCR[10])!=Source.xy) ||
        any(uint4(OCR[11],OCR[12],OCR[13],OCR[14])!=Field)) return false;
    if (Condition[0]!=4u || Condition[1]!=0x34504353u || Condition[2]!=count ||
        (Condition[3] & ~((1u<<count)-1u))!=0u) return false;
    return count==0u ? SCT[2]==0u : SCT[2]==1u && Condition[4]==2u;
}

ByteAddressBuffer ActualRoot : register(t16);
StructuredBuffer<uint> SourceEnvelope : register(t17);
StructuredBuffer<uint> EmptyWitness : register(t21);
RWByteAddressBuffer PublishArgs : register(u7);

static const uint HasEmptyWitness = 8u;

uint4 SourceWord(uint i) {
    return uint4(SourceEnvelope[i],SourceEnvelope[i+1u],SourceEnvelope[i+2u],SourceEnvelope[i+3u]);
}
uint4 SCTWord(uint i) { return uint4(SCT[i],SCT[i+1u],SCT[i+2u],SCT[i+3u]); }

bool EarlyContextCurrent() {
    if (any(Source.xw!=Shape.xy)||any(Shape.xy==0u)||any(Source.xy>5120u)||Source.z>=Source.y||
        Source.w!=Source.y-Source.z||any(Model!=Owner)||any(Observation.xy!=ClockToken.xy)||
        all(Owner.xy==0u)||all(ClockToken.xy==0u)||all(ClockToken.zw==0u)||
        Observation.z>1u||Observation.w!=0u||any(Options!=uint4(1u,27u,0u,0u))) return false;
    if (any(Field.xy==0u)||any(Field.xy>5120u)||Field.z>=Field.w||Field.w>Field.y||
        any(Content.xy>=Content.zw)||any(Content.zw>Field.xy)) return false;
    if (SCT[0]!=5u||SCT[1]!=0x35544353u||SCT[2]>1u||SCT[22]!=27u||SCT[23]!=1u||
        any(SCTWord(4u)!=Model)||any(SCTWord(8u)!=Source)||any(SCTWord(12u)!=Field)||
        any(SCTWord(16u)!=Content)||any(uint2(SCT[20],SCT[21])!=Observation.xy)) return false;
    [unroll] for (uint i=24u;i<32u;++i) if (SCT[i]!=0u) return false;
    return true;
}

bool EarlyRootCurrent(uint mode, uint expectedWork) {
    uint2 token=ClockToken.zw, cookie=token^uint2(0xa3756c91u,0x5c8a936eu);
    uint marker=ActualRoot.Load(28u);
    if (marker!=0u&&(marker!=0x52434f4fu||expectedWork!=1u)) return false;
    if (any(ActualRoot.Load4(0u)!=uint4(1u,1u^0xd1ec15a5u^marker,token))||
        any(ActualRoot.Load4(16u)!=uint4(cookie,0x47524243u,marker))||
        any(ActualRoot.Load4(32u)!=uint4(token,cookie))||
        any(ActualRoot.Load4(48u)!=uint4(0x54535152u,expectedWork,expectedWork!=0u?(expectedWork^0x6f435257u):0u,0u)))
        return false;
    uint4 head=uint4(1u,1u,1u,0u);
    return all(ActualRoot.Load4(80u)==(mode==2u?uint4(0u,0u,0u,0u):head))&&
           all(ActualRoot.Load4(240u)==head)&&
           all(ActualRoot.Load4(marker==0x52434f4fu?208u:224u)==head);
}

bool SourceCurrent(uint proof_tag, uint features, uint4 formats) {
    uint4 uip=SourceWord(0u);
    if (any(uip.xyz!=uint3(4u,0x34504955u,1u))||uip.w>255u||
        any(SourceWord(4u)!=Model)||any(SourceWord(8u)!=uint4(Observation.xy,Source.xy))||
        any(SourceWord(12u)!=Field)||any(SourceWord(16u)!=Content)||
        SourceEnvelope[20u]!=SCT[3u]||SourceEnvelope[22u]!=0u||SourceEnvelope[23u]!=0u) return false;
    if (any(SourceWord(40u)!=uint4(proof_tag,1u,5u,1u))||
        any(SourceWord(44u)!=uint4(Model.xy,Observation.xy))||
        any(SourceWord(48u)!=uint4(Model.zw,ClockToken.zw))||any(SourceWord(52u)!=Source)) return false;
    uint4 roi=SourceWord(56u), raw=SourceWord(60u);
    if (any(raw.xy==0u)||any(raw.xy>16384u)||any(raw.zw==0u)||any(roi.xy>=roi.zw)||
        any(roi.zw>raw.xy)||any(roi.zw-roi.xy!=Source.xy)) return false;
    if ((SourceEnvelope[64u]&8u)==0u||SourceEnvelope[65u]!=4u||SourceEnvelope[66u]!=1u||
        SourceEnvelope[67u]!=1u||SourceEnvelope[68u]!=1u||SourceEnvelope[69u]!=0u||
        SourceEnvelope[70u]>2u||SourceEnvelope[71u]!=features) return false;
    uint n=Shape.x*Shape.y, g=((Shape.x+15u)/16u)*((Shape.y+15u)/16u);
    return all(SourceWord(72u)==uint4(n,n,0u,n))&&all(SourceWord(76u)==uint4(g,g,0u,SCT[3u]))&&
           all(SourceWord(80u)==Field)&&all(SourceWord(84u)==Content)&&all(SourceWord(88u)==Options)&&
           all(SourceWord(92u)==Observation)&&all(SourceWord(96u)==ActualRoot.Load4(240u))&&
           all(SourceWord(100u)==formats);
}

uint4 FieldGrid() { return uint4((Field.xy+15u)/16u,1u,0u); }

// Rejoin the GPU classifier's receipt against the original Root/C24/SCT and owner.
// Called only after EarlyContextCurrent and EarlyRootCurrent have verified that tuple.
bool EmptyWitnessCurrent(uint mode, uint expectedWork) {
    if ((Shape.w&HasEmptyWitness)==0u||expectedWork!=1u||SCT[2u]!=0u||SCT[3u]!=0u||
        ActualRoot.Load(28u)!=0x52434f4fu||any(ActualRoot.Load4(176u)!=FieldGrid())||
        any(ActualRoot.Load4(224u)!=uint4(0u,1u,1u,0u))) return false;
    uint words, stride; EmptyWitness.GetDimensions(words,stride);
    if (words!=32u||stride!=4u||
        any(uint4(EmptyWitness[0u],EmptyWitness[1u],EmptyWitness[2u],EmptyWitness[3u])!=uint4(0x31504d45u,1u,1u,0u))||
        any(uint4(EmptyWitness[28u],EmptyWitness[29u],EmptyWitness[30u],EmptyWitness[31u])!=uint4(ClockToken.zw,mode,expectedWork)))
        return false;
    [loop] for (uint i=0u;i<24u;++i) if (EmptyWitness[4u+i]!=CBWord(i)) return false;
    return true;
}

void PublishEarlyReceipt(uint kind, uint algorithm, bool current) {
    Receipt[0u]=0x31534d53u; Receipt[1u]=1u; Receipt[2u]=kind; Receipt[3u]=current?0u:1u;
    [loop] for (uint i=0u;i<24u;++i) Receipt[4u+i]=CBWord(i);
    [loop] for (uint j=0u;j<32u;++j) Receipt[28u+j]=current?SCT[j]:0u;
    Receipt[60u]=ClockToken.z; Receipt[61u]=ClockToken.w; Receipt[62u]=algorithm; Receipt[63u]=0u;
}
#endif
