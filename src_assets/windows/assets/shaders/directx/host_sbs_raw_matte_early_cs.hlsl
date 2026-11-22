// Reduced Source5/CBC5/RawM/Inside producer authority for the frozen spatial RawMatte.
// The geometry and score shaders are the same binaries as the file-fed benchmark.
#include "host_sbs_bench_raw_matte_cs.hlsl"

ByteAddressBuffer ActualRoot : register(t16);
StructuredBuffer<uint> SourceEnvelope : register(t17);
StructuredBuffer<uint> CBCMeta : register(t18);
StructuredBuffer<uint> RawMeta : register(t19);
ByteAddressBuffer FloorData : register(t20);
StructuredBuffer<uint> EmptyWitness : register(t21);
RWByteAddressBuffer LegacyArgs : register(u1);
RWByteAddressBuffer PublishArgs : register(u7);

static const uint DescriptorFault = 1u, HasLegacy = 2u, HasLazyLegacy = 4u, HasEmptyWitness = 8u;
static const uint Held = 3u;

uint4 SourceWord(uint i) {
    return uint4(SourceEnvelope[i],SourceEnvelope[i+1u],SourceEnvelope[i+2u],SourceEnvelope[i+3u]);
}
uint4 CBCWord(uint i) { return uint4(CBCMeta[i],CBCMeta[i+1u],CBCMeta[i+2u],CBCMeta[i+3u]); }
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

bool FloorCurrent(uint mode, uint count) {
    uint n=Shape.x*Shape.y;
    if (any(CBCWord(0u)!=uint4(0x31434e45u,5u,1u,0u))||CBCMeta[4u]!=1u||CBCMeta[5u]!=1u||
        CBCMeta[9u]!=1u||CBCMeta[80u]!=SCT[3u]||
        any(CBCWord(40u)!=uint4(0x31434e45u,1u,1u,mode))||
        any(CBCWord(44u)!=uint4(Shape.xy,28u,1u))||any(CBCWord(48u)!=uint4(Source.xy,28u,1u))||
        any(CBCWord(52u)!=uint4(ClockToken.zw,1u,1u^0x6f435257u))||
        any(CBCWord(56u)!=uint4(Model.xy,Observation.xy))||any(CBCWord(60u)!=uint4(Model.zw,0u,0u))||
        any(CBCWord(64u)!=uint4(0u,0u,Source.xy))||any(CBCWord(68u)!=uint4(5u,0u,4u,136u))) return false;
    [unroll] for (uint c=72u;c<80u;++c) if (CBCMeta[c]!=0u) return false;
    if (any(uint4(RawMeta[0u],RawMeta[1u],RawMeta[2u],RawMeta[3u])!=uint4(0x31474e52u,1u,1u,0u))||
        RawMeta[5u]!=1u||RawMeta[7u]>n||RawMeta[8u]!=(RawMeta[7u]!=0u?2u:1u)||
        ((RawMeta[4u]|RawMeta[6u])&~((1u<<count)-1u))!=0u||(count==0u&&RawMeta[7u]!=0u)) return false;
    [unroll] for (uint r=0u;r<16u;++r) if (RawMeta[36u+r]!=CBCMeta[r]) return false;
    if (any(FloorData.Load4(0u)!=uint4(0x37424643u,7u,1u,0u))||
        any(FloorData.Load4(192u)!=uint4(Shape.xy,n,68608u))||
        any(FloorData.Load3(208u)!=uint3(64u,128u,16u))||
        any(FloorData.Load4(308u).xy!=uint2(0x31444e49u,1u))||
        FloorData.Load(316u)>255u||FloorData.Load(320u)>255u||
        FloorData.Load(324u)>n||FloorData.Load(328u)>n||FloorData.Load(332u)!=1u) return false;
    [unroll] for (uint f=32u;f<48u;f+=4u) if (any(FloorData.Load4(4u*f)!=0u)) return false;
    [loop] for (uint i=0u;i<24u;++i) {
        uint cb=CBWord(i);
        if (CBCMeta[16u+i]!=cb||RawMeta[12u+i]!=cb||FloorData.Load(4u*(8u+i))!=cb) return false;
    }
    return true;
}

bool CurrentLegacy() {
    if ((Shape.w&HasLegacy)==0u||
        any(uint4(RuleReceipt[0u],RuleReceipt[1u],RuleReceipt[2u],RuleReceipt[3u])!=uint4(0x31534d53u,1u,1u,0u))||
        any(uint4(RuleReceipt[60u],RuleReceipt[61u],RuleReceipt[62u],RuleReceipt[63u])!=uint4(ClockToken.zw,1u,0u)))
        return false;
    [loop] for (uint i=0u;i<24u;++i) if (RuleReceipt[4u+i]!=CBWord(i)) return false;
    [loop] for (uint j=0u;j<32u;++j) if (RuleReceipt[28u+j]!=SCT[j]) return false;
    return true;
}

uint4 LegacyFieldGrid() { return uint4((Field.xy+15u)/16u,1u,0u); }

// Rejoin the GPU classifier's receipt against the original Root/C24/SCT and owner.
// Called only after EarlyContextCurrent and EarlyRootCurrent have verified that tuple.
bool EmptyWitnessCurrent(uint mode, uint expectedWork) {
    if ((Shape.w&HasEmptyWitness)==0u||expectedWork!=1u||SCT[2u]!=0u||SCT[3u]!=0u||
        ActualRoot.Load(28u)!=0x52434f4fu||any(ActualRoot.Load4(176u)!=LegacyFieldGrid())||
        any(ActualRoot.Load4(224u)!=uint4(0u,1u,1u,0u))) return false;
    uint words, stride; EmptyWitness.GetDimensions(words,stride);
    if (words!=32u||stride!=4u||
        any(uint4(EmptyWitness[0u],EmptyWitness[1u],EmptyWitness[2u],EmptyWitness[3u])!=uint4(0x31504d45u,1u,1u,0u))||
        any(uint4(EmptyWitness[28u],EmptyWitness[29u],EmptyWitness[30u],EmptyWitness[31u])!=uint4(ClockToken.zw,mode,expectedWork)))
        return false;
    [loop] for (uint i=0u;i<24u;++i) if (EmptyWitness[4u+i]!=CBWord(i)) return false;
    return true;
}

[numthreads(1,1,1)]
void prepare_early_main() {
    [unroll] for (uint start=0u;start<240u;start+=16u) WorkStore.Store4(start,0u);
    Evidence.Store4(0u,0u);
    WorkStore.Store(76u,Shape.z);
    const bool lazyLegacy=(Shape.w&HasLazyLegacy)!=0u;
    if (lazyLegacy) LegacyArgs.Store4(0u,0u);
    bool hold=any(ActualRoot.Load3(240u)==0u);
    PublishArgs.Store4(0u,uint4((Shape.xy+uint2(TileX-1u,TileY-1u))/uint2(TileX,TileY),hold?0u:1u,0u));
    if (hold) { WorkStore.Store(64u,Held); return; }

    uint mode=(Shape.w>>8u)&255u, expectedWork=(Shape.w>>16u)&255u;
    if (!EarlyContextCurrent()||!EarlyRootCurrent(mode,expectedWork)) { WorkStore.Store(72u,1u); return; }
    if (EmptyWitnessCurrent(mode,expectedWork)) {
        WorkStore.Store(16u,1u); // Fast-empty marker. Score ROI is otherwise unused on EMPTY.
        WorkStore.Store(212u,1u);
        return;
    }
    // The legacy dense shader grid-strides at field pitch over the native band.
    bool legacy=lazyLegacy?all(ActualRoot.Load4(176u)==LegacyFieldGrid()):CurrentLegacy();
    WorkStore.Store(208u,legacy?1u:0u);
    if ((Shape.w&DescriptorFault)!=0u) { WorkStore.Store(72u,5u); return; }
    uint count=OCR[4u];
    if (!OcrCurrent(count)) { WorkStore.Store(72u,2u); return; }
    if (expectedWork!=1u||!SourceCurrent(0x35534743u,63u,uint4(10u,28u,10u,1u))||!FloorCurrent(mode,count)) { WorkStore.Store(72u,6u); return; }
    if (count==0u) { WorkStore.Store(212u,1u); return; }

    uint4 roi=uint4(Shape.xy,0u,0u);
    [loop] for (uint slot=0u;slot<count;++slot) {
        uint4 core=Rect(16u+8u*slot), raw=Rect(144u+8u*slot);
        if (any(core.xy>=core.zw)||any(raw.xy>=raw.zw)||any(raw.xy<Content.xy)||any(raw.zw>Content.zw)||
            any(core.xy<raw.xy)||any(core.zw>raw.zw)) { WorkStore.Store(72u,3u); return; }
        uint4 box=NativeCover(raw);
        if (any(box.xy>=box.zw)) { WorkStore.Store(72u,3u); return; }
        WorkStore.Store4(80u+16u*slot,box);
        roi=uint4(min(roi.xy,box.xy),max(roi.zw,box.zw));
    }
    uint4 padded=PaddedCover(roi,4u);
    WorkStore.Store4(0u,Args(roi));
    WorkStore.Store4(16u,roi);
    WorkStore.Store4(32u,Args(padded));
    WorkStore.Store4(48u,padded);
    WorkStore.Store(68u,(1u<<count)-1u);
    WorkStore.Store(212u,2u);
}

void PublishEarlyReceipt(uint kind, uint algorithm, bool current) {
    Receipt[0u]=0x31534d53u; Receipt[1u]=1u; Receipt[2u]=kind; Receipt[3u]=current?0u:1u;
    [loop] for (uint i=0u;i<24u;++i) Receipt[4u+i]=CBWord(i);
    [loop] for (uint j=0u;j<32u;++j) Receipt[28u+j]=current?SCT[j]:0u;
    Receipt[60u]=ClockToken.z; Receipt[61u]=ClockToken.w; Receipt[62u]=algorithm; Receipt[63u]=0u;
}

[numthreads(TileX,TileY,1)]
void publish_early_main(uint3 id:SV_DispatchThreadID) {
    if (any(id.xy>=Shape.xy)) return;
    uint planned=WorkStore.Load(212u);
    uint2 geometry=WorkStore.Load2(32u), score=WorkStore.Load2(0u);
    uint3 completed=Evidence.Load3(0u);
    bool pipeline=all(completed==uint3(geometry.x*geometry.y,score.x*score.y,0u));
    uint accepted=planned==1u||(planned==2u&&pipeline)?planned:0u;
    bool legacy=accepted==0u&&WorkStore.Load(208u)==1u;
    bool lazyLegacy=legacy&&(Shape.w&HasLazyLegacy)!=0u;
    bool covered=false;
    // Producer outputs may retain older pixel data on an authenticated fast empty.
    [branch] if (accepted==2u) covered=CoveredForOutput(id.xy,WorkStore.Load(68u));
    if (!lazyLegacy) {
        uint selected=0u;
        if (covered) selected=Candidate.Load(int3(id.xy,0));
        if (legacy) selected=Rule.Load(int3(id.xy,0));
        OutputMask[id.xy]=selected;
    }
    // Score has written alpha only at non-floor candidates with RawM <= 3. Fill all other pixels.
    bool clearAlpha=true;
    [branch] if (covered)
        clearAlpha=Trusted.Load(int3(id.xy,0))!=0u||(Geometry.Load(int3(id.xy,0))&7u)>3u;
    if (clearAlpha) OutputAlpha8[id.xy]=0u;
    if (all(id.xy==0u)) {
        if (lazyLegacy) {
            uint4 grid=LegacyFieldGrid();
            LegacyArgs.Store4(0u,grid);
            WorkStore.Store3(228u,grid.xyz);
            PublishEarlyReceipt(1u,1u,true);
        } else if (legacy) { [loop] for (uint i=0u;i<64u;++i) Receipt[i]=RuleReceipt[i]; }
        else PublishEarlyReceipt(accepted!=0u?4u:1u,3u,accepted!=0u);
        WorkStore.Store(64u,accepted);
        if (planned==2u&&!pipeline) WorkStore.Store(72u,4u);
        WorkStore.Store3(216u,completed);
    }
}
