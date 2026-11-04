// PRIVATE CFB5: authoritative BODY plus positively certified collar/black/AA/closure; independent Raw fault fallback.
// PRIVATE BBSC1 adds an earlier measured-style snapshot; no old M upload, oracle, frame wait or rendering authority.
cbuffer ActualContourCB : register(b0) {
    uint4 CSource;uint4 CField;uint4 CContent;
    uint4 CFrame;uint4 CObservation;uint4 COptions;
};
cbuffer BandHook : register(b1) { uint4 Hook; };
cbuffer BandShape : register(b2) { uint4 Shape; };
Texture2D<float4> CurrentRGB : register(t0);
StructuredBuffer<uint> CurrentOCR : register(t1);
Texture2D<uint> SelectedBits : register(t2);
Texture2D<uint> GlobalBody : register(t3);
Texture2D<uint> CoverBits : register(t4);
StructuredBuffer<uint> CBCMeta : register(t5);
StructuredBuffer<uint> Parent : register(t6);
StructuredBuffer<uint> RootFlags : register(t7);
Texture2D<uint> Passable : register(t8);
Texture2D<uint> InnerEligible : register(t9);
Texture2D<uint> RawM : register(t10);
StructuredBuffer<uint> RawMeta : register(t11);
globallycoherent RWByteAddressBuffer Data : register(u0);
RWTexture2D<uint> V3M : register(u1); // Packed ownership/AA/closure temp until final binary publication.
globallycoherent RWTexture2D<uint> PassableCache : register(u2);
// Current own Raw-root witness scratch. Canonical second-graph init clears the
// borrowed mode-final texture over EVERY ROI pixel before this unit's reduction.
// Entry5 only accumulates; ordered entry6 consumes before any next-unit reuse.
globallycoherent RWTexture2D<uint> RawComponentWitness : register(u3);
Texture2D<uint> Code1M : register(t12);
// PRIVATE BBSC1 extension. These optional views never grant current calibration authority.
Texture2D<uint> StabilityImageRead : register(t13); // RGB24 | SelectedLOW8, one committed epoch
Texture2D<uint> StabilityHeaderRead : register(t14); // 128x1 R32_UINT
Texture2D<uint> CurrentInsideBits : register(t15); // Existing current independent component authority, local anchors only.
RWTexture2D<uint> StabilityHeaderWrite : register(u4);
RWTexture2D<uint> StabilityImageWrite : register(u5);
RWTexture2D<uint> AAHistory : register(u6); // Paired strong-snapshot construction/accepted-AA bits.
RWStructuredBuffer<uint> WeakAAState : register(u7); // PRIVATE WAA1 independent observation header+RGB/AA words.
bool WeakAAPlan();

static const uint ROOT_MASK=0x3fffffffu,ENCLOSED=0x80000000u,BAD_RIM=0x40000000u;
static const uint BAND_MASK=255u,OLD_AA=256u,SUPPORT_AA=512u,CLOSURE=1024u,UNKNOWN_RAW=2048u,CERTIFIED_COLLAR=4096u,MEASURED_RIM=8192u;
static const uint F_CACHE=8u;
static const uint TAG=0x37424643u, WORDS=68608u, HIST=1024u, BANDS=66560u;
static const uint F_SHAPE=1u,F_GRAPH=2u,F_COUNTS=4u;
static const uint ST_NO_BODY=0u,ST_KNOWN=1u,ST_SOURCE=2u,ST_RESOURCE=3u;
static const uint ST_NO_INFORMATION=4u,ST_TIED=5u,ST_FEW=6u,ST_MAJORITY=7u,ST_NO_REPEAT=8u,ST_T_CAP=9u,ST_FAULT=10u,ST_PREPARED=11u;
uint U(uint unit) { return 128u+64u*unit; }
uint BlackModelWord(uint word) { return CBCMeta[word]; }
// CBC5 HIGH8 carries trusted colour/closed-BODY support. Every geometric/body consumer
// below sees the exact original128 LOW8 selected ownership.
uint SelectedGeometryBits(uint word) { return word&255u; }
uint SelectedForegroundBits(uint word) { return (word>>8u)&255u; }
uint SelectedGeometryAt(int2 p) { return SelectedGeometryBits(SelectedBits.Load(int3(p,0))); }
#include "physical_black_model_shared.hlsl"
#include "cbc_unit_state_shared.hlsl"
#include "current_stroke_math_shared.hlsl"
// CBC-only exact singleton: local opaque Raw4CC authority, with no permission
// to infer foreground, nonblack geometry, antialiasing or retained evidence.
bool OpaqueBlackOnly(uint unit) { return CBCMeta[PhysicalModelBank(unit)+1u]==4u; }
uint R(uint word) { return Data.Load(word*4u); }
void W(uint word,uint value) { Data.Store(word*4u,value); }
// Internal per-unit word36: fresh Rawbegin clears it, and calibration resets it.
// A usable shared extent may add support without authorizing replacement of Raw.
// Zero preserves legacy known-width authority; every other value retains Raw.
static const uint UNIT_WIDTH_POLICY=36u,WIDTH_REPLACE_RAW=0u,WIDTH_ADD_ONLY=1u;
bool CurrentWidthAvailable(uint unit) {
    uint cu=U(unit),t=R(cu+11u);
    return R(cu+13u)==ST_KNOWN&&t>0u&&t<=16u;
}
bool CurrentWidthMayReplaceRaw(uint unit) {
    return CurrentWidthAvailable(unit)&&R(U(unit)+UNIT_WIDTH_POLICY)==WIDTH_REPLACE_RAW;
}
void Fault(uint bits) { uint ignored;Data.InterlockedOr(12u,bits,ignored); }
bool DataShape() { uint bytes;Data.GetDimensions(bytes);return bytes==WORDS*4u; }
uint H(uint unit,uint band,uint extent) { return HIST+(unit*64u+band)*128u+extent; }
uint B(uint unit,uint band) { return BANDS+(unit*64u+band)*4u; }
uint CBWord(uint i) {
    if(i<4u)return CSource[i];if(i<8u)return CField[i-4u];if(i<12u)return CContent[i-8u];
    if(i<16u)return CFrame[i-12u];if(i<20u)return CObservation[i-16u];return COptions[i-20u];
}
bool Native(uint2 p) { return all(p<Shape.xy); }
bool InputReady() {
    if(!DataShape()||R(0u)!=TAG||R(1u)!=7u||R(2u)!=1u||R(3u)!=0u)return false;
    uint n,s;CBCMeta.GetDimensions(n,s);if(n!=512u||s!=4u)return false;
    RawMeta.GetDimensions(n,s);if(n!=512u||s!=4u)return false;
    if(CBCMeta[0u]!=0x31434e45u||CBCMeta[1u]!=5u||CBCMeta[2u]!=1u||CBCMeta[3u]!=0u)return false;
    if(RawMeta[0u]!=0x31474e52u||RawMeta[1u]!=1u||RawMeta[2u]!=1u||RawMeta[3u]!=0u)return false;
    [loop]for(uint i=0u;i<24u;++i)if(CBCMeta[16u+i]!=CBWord(i)||RawMeta[12u+i]!=CBWord(i)||R(8u+i)!=CBWord(i))return false;
    return all(CSource.xw==Shape.xy)&&Shape.z==Shape.x*Shape.y&&Shape.w==0u;
}
bool ROI(uint2 local,out uint2 p) {
    p=0u;if(Hook.x>=8u||any(Hook.yzw))return false;
    uint ub=128u+32u*Hook.x;uint4 r=uint4(CBCMeta[ub+24u],CBCMeta[ub+25u],CBCMeta[ub+26u],CBCMeta[ub+27u]);
    if(any(r.xy>=r.zw)||any(r.zw>Shape.xy)||any(local>=r.zw-r.xy))return false;
    p=r.xy+local;return true;
}
bool OwnCover(int2 p) {
    if(any(p<0)||any(p>=int2(Shape.xy)))return false;
    return (CoverBits.Load(int3(p,0))&(1u<<Hook.x))!=0u;
}
bool OwnBody(int2 p) {
    return OwnCover(p)&&(SelectedGeometryAt(p)&(1u<<Hook.x))!=0u;
}
bool ValidPoint(int2 p) { return OwnCover(p)&&CurrentRGB.Load(int3(p,0)).a==1.0f; }
uint3 RGB8(int2 p) { return (uint3)floor(CurrentRGB.Load(int3(p,0)).rgb*255.0f+0.5f); }
bool Black(int2 p) {
    float3 v=CurrentRGB.Load(int3(p,0)).rgb;
    return PhysicalBlackRGB(v,Hook.x);
}
bool HasAdoptedPhysicalRefinement(uint unit) {
    uint u=128u+32u*unit;
    return CurrentCBCPhysicalRefinementAdopted(CBCMeta[u],CBCMeta[u+1u],CBCMeta[u+3u],CBCMeta[u+6u],CBCMeta[u+29u]);
}
bool AnyPhysicalBlack(int2 p,uint3 code) {
    // CBCunit and Rawbegin already validated snapshots. Do not consult mutable
    // future ST_KNOWN: raw/band/AA stages interleave across these same units.
    if(max(code.r,max(code.g,code.b))<=1u)return true;
    uint cover=CoverBits.Load(int3(p,0));
    [unroll]for(uint unit=0u;unit<8u;++unit)
        if((cover&(1u<<unit))!=0u&&!OpaqueBlackOnly(unit)&&
           HasAdoptedPhysicalRefinement(unit)&&PhysicalBlackCode(code,unit))return true;
    return false;
}
bool GraphReady() {
    uint n,s,nf,sf;Parent.GetDimensions(n,s);RootFlags.GetDimensions(nf,sf);
    if(n!=Shape.z+1u||s!=4u||nf!=n||sf!=4u){Fault(F_SHAPE);return false;}
    if(Parent[0u]!=0u){Fault(F_GRAPH);return false;}
    return true;
}
uint FindStable(uint id) {
    uint node=id;
    [loop]while(true) {
        uint next=Parent[node];
        if(next==0u||next==0xffffffffu||next>node){Fault(F_GRAPH);return 0u;}
        if(next==node)return node;node=next;
    }
    return 0u;
}
bool Exterior(int2 p) {
    if(!OwnCover(p)||Passable.Load(int3(p,0))!=1u)return false;
    uint root=FindStable((uint)p.y*Shape.x+(uint)p.x+1u);
    return root!=0u&&(RootFlags[root]&2u)!=0u;
}
// PROFILE endpoint only, while Parent/RootFlags still name the first passable
// graph. Keep both original boundary classes2/3, add closed unseeded class0,
// and reject closed seeded text interior class1. AA Exterior stays unchanged.
bool ProfileBackground(int2 p) {
    if(!OwnCover(p)||Passable.Load(int3(p,0))!=1u)return false;
    uint root=FindStable((uint)p.y*Shape.x+(uint)p.x+1u);
    return root!=0u&&(RootFlags[root]&3u)!=1u;
}
int CeilDiv(int n,int d) { return n>=0?(n+d-1)/d:n/d; }
uint NativeCoreHeight(uint unit) {
    uint first=16u+8u*unit;
    int dy=(int)(CContent.w-CContent.y);
    if(dy<=0)return 0u;
    int top=CeilDiv(2*((int)CurrentOCR[first+1u]-(int)CContent.y)*(int)CSource.y-dy,2*dy);
    int bottom=CeilDiv(2*((int)CurrentOCR[first+3u]-(int)CContent.y)*(int)CSource.y-dy,2*dy);
    return (uint)max(1,bottom-top);
}
[numthreads(8,1,1)]
void prepare_profiles_main(uint3 group:SV_GroupID,uint3 tid:SV_GroupThreadID) {
    if(any(group)||tid.x>=8u||!DataShape())return;
    uint unit=tid.x,cu=U(unit),count=R(cu);
    if(count==0u){W(cu+13u,ST_NO_BODY);return;}
    W(cu+13u,ST_SOURCE);
    if(!InputReady())return;
    uint n,s;CurrentOCR.GetDimensions(n,s);
    if(n!=256u||s!=4u||CurrentOCR[0u]!=4u||CurrentOCR[1u]!=0x3852434fu||CurrentOCR[2u]!=1u||unit>=CurrentOCR[4u])return;
    uint ub=128u+32u*unit;
    W(cu+15u,CBCMeta[ub+6u]==1u?1u:0u);
    if(CBCMeta[4u]!=1u||CBCMeta[5u]!=1u||CBCMeta[ub+1u]!=1u)return;
    uint xmin=R(cu+1u),ymin=R(cu+2u),xmax=R(cu+3u),ymax=R(cu+4u);
    if(xmin>=xmax||ymin>=ymax||xmax>Shape.x||ymax>Shape.y)return;
    uint bandw=ymax-ymin,bandcount=(xmax-xmin+bandw-1u)/bandw,Rmax=NativeCoreHeight(unit);
    W(cu+5u,bandw);W(cu+6u,Rmax);W(cu+7u,bandcount);
    uint resource=(bandcount>64u?1u:0u)|(Rmax>128u?2u:0u);
    if(resource!=0u){W(cu+13u,ST_RESOURCE);W(cu+23u,resource);uint ignored;Data.InterlockedOr(24u,1u<<unit,ignored);return;}
    W(cu+13u,ST_PREPARED);W(cu+14u,1u);uint ignored;Data.InterlockedOr(16u,1u<<unit,ignored);
}
bool AcceptedProfile(int2 p,int2 dir,uint Rmax,out uint extent) {
    extent=0u;
    if(InnerEligible.Load(int3(p,0))!=1u)return false;
    int2 tangent=int2(-dir.y,dir.x);
    if(!ValidPoint(p)||!ValidPoint(p+tangent)||!ValidPoint(p-tangent)||
       !ValidPoint(p+dir)||!ValidPoint(p+dir+tangent)||!ValidPoint(p+dir-tangent))return false;
    if(!OwnBody(p+tangent)||!OwnBody(p-tangent)||OwnBody(p+dir)||OwnBody(p+dir+tangent)||OwnBody(p+dir-tangent))return false;
    uint3 F=RGB8(p);uint start=0u;
    [loop]for(uint j=1u;j<=Rmax;++j) {
        int2 q=p+(int)j*dir;
        if(!ValidPoint(q)||GlobalBody.Load(int3(q,0))!=0u)return false;
        uint3 v=RGB8(q);bool black=PhysicalBlackCode(v,Hook.x);
        if(start==0u) {
            if(black) {
                if(j>2u)return false;
                int3 delta=abs(int3(F)-int3(v));
                if(max(delta.r,max(delta.g,delta.b))<=20.4f)return false;
                start=j;
            } else if(j>=2u)return false;
            continue;
        }
        if(black)continue;
        if(!ProfileBackground(q))return false;
        // Endpoint distance semantics remain CFB2. The first nonblack q is at j, so the last
        // physical-black sample is j-1 source pixels from this BODY origin.
        // CFB2 histograms store that support extent, not physical run length.
        extent=j-1u;return extent>0u;
    }
    return false;
}
[numthreads(16,16,1)]
void profile_main(uint3 d:SV_DispatchThreadID) {
    uint2 p;if(d.z!=0u||!InputReady()||!ROI(d.xy,p))return;
    uint cu=U(Hook.x);if(R(cu+14u)!=1u||!OwnBody(int2(p)))return;
    if(!GraphReady()){W(cu+13u,ST_FAULT);return;}
    uint xmin=R(cu+1u),bandw=R(cu+5u),Rmax=R(cu+6u);
    if(bandw==0u||p.x<xmin)return;
    uint band=(p.x-xmin)/bandw;
    if(band>=64u||band>=R(cu+7u)){Fault(F_COUNTS);return;}
    const int2 dirs[4]={int2(1,0),int2(-1,0),int2(0,1),int2(0,-1)};
    [unroll]for(uint k=0u;k<4u;++k)if(!OwnBody(int2(p)+dirs[k])) {
        uint t;if(AcceptedProfile(int2(p),dirs[k],Rmax,t)) {
            if(t==0u||t>=Rmax||t>=128u){Fault(F_COUNTS);return;}
            uint ignored;Data.InterlockedAdd(H(Hook.x,band,t)*4u,1u,ignored);
            Data.InterlockedAdd((cu+8u)*4u,1u,ignored);Data.InterlockedAdd(55u*4u,1u,ignored);
        }
    }
}
void SetBit(inout uint2 mask,uint b) { if(b<32u)mask.x|=1u<<b;else mask.y|=1u<<(b-32u); }
bool HasBit(uint2 mask,uint b) { return b<32u?(mask.x&(1u<<b))!=0u:(mask.y&(1u<<(b-32u)))!=0u; }
// Adjacent maximal windows share exactly one physical extent. Require two
// distinct modal bands at that extent and a strict-majority union of support.
// The shared extent is conservative; neither outer endpoint is authorized.
bool ResolveAdjacentSharedExtent(uint lowWindow,uint highWindow,uint2 common,uint2 united,uint informativeCount,out uint extent) {
    extent=0u;
    if(highWindow!=lowWindow+1u||countbits(common.x)+countbits(common.y)<2u||
       2u*(countbits(united.x)+countbits(united.y))<=informativeCount)return false;
    extent=highWindow;return true;
}
void CalibrateCurrentWidth(uint3 group) {
    if(any(group)||Hook.x>=8u||any(Hook.yzw)||!DataShape())return;
    uint unit=Hook.x,cu=U(unit);
    W(cu+UNIT_WIDTH_POLICY,WIDTH_REPLACE_RAW);
    if(R(cu+14u)!=1u)return;
    if(!InputReady()){W(cu+13u,ST_FAULT);return;}
    uint bandcount=R(cu+7u),Rmax=R(cu+6u);uint2 informative=0u;uint total=0u;
    [loop]for(uint band=0u;band<bandcount;++band) {
        uint sum=0u,maximum=0u,minimumMode=0u,maximumMode=0u;
        [loop]for(uint t=0u;t<128u;++t) {
            uint n=R(H(unit,band,t));sum+=n;
            if(n!=0u&&(t==0u||t>=Rmax)){Fault(F_COUNTS);W(cu+13u,ST_FAULT);return;}
            if(n>maximum){maximum=n;minimumMode=maximumMode=t;}
            else if(n!=0u&&n==maximum){minimumMode=min(minimumMode,t);maximumMode=max(maximumMode,t);}
        }
        uint b=B(unit,band);W(b,sum);W(b+1u,maximum);W(b+2u,minimumMode);W(b+3u,maximumMode);total+=sum;
        if(sum!=0u&&maximumMode-minimumMode<=1u)SetBit(informative,band);
    }
    if(total!=R(cu+8u)){Fault(F_COUNTS);W(cu+13u,ST_FAULT);return;}
    uint informativeCount=countbits(informative.x)+countbits(informative.y);
    W(cu+9u,informativeCount);W(cu+20u,informative.x);W(cu+21u,informative.y);
    if(informativeCount==0u){W(cu+13u,ST_NO_INFORMATION);return;}
    uint maximum=0u,lowWindow=0u,highWindow=0u;uint2 winner=0u,common=0u,united=0u;bool tied=false;
    [loop]for(uint k=0u;k<Rmax;++k) {
        uint2 support=0u;
        [loop]for(uint band=0u;band<bandcount;++band)if(HasBit(informative,band)) {
            uint b=B(unit,band);
            if(R(b+2u)>=k&&R(b+3u)<=k+1u)SetBit(support,band);
        }
        uint count=countbits(support.x)+countbits(support.y);if(count==0u)continue;
        if(count>maximum){maximum=count;winner=common=united=support;lowWindow=highWindow=k;tied=false;}
        else if(count==maximum) {
            // Keep every maximal window. Distinct adjacent windows can express
            // one repeated extent with a strict-majority common bandset.
            if(any(support!=winner))tied=true;
            common&=support;united|=support;
            lowWindow=min(lowWindow,k);highWindow=max(highWindow,k);
        }
    }
    bool unresolved=tied,sharedAddOnly=false;uint repeatedLow=lowWindow,repeatedHigh=highWindow+1u;
    if(tied) {
        uint commonCount=countbits(common.x)+countbits(common.y);
        if(highWindow-lowWindow<=1u&&commonCount>=2u&&2u*commonCount>informativeCount) {
            winner=united;maximum=countbits(winner.x)+countbits(winner.y);
            unresolved=false;
        } else {
            uint sharedExtent;
            if(ResolveAdjacentSharedExtent(lowWindow,highWindow,common,united,informativeCount,sharedExtent)) {
                winner=united;maximum=countbits(winner.x)+countbits(winner.y);
                repeatedLow=repeatedHigh=sharedExtent;unresolved=false;sharedAddOnly=true;
            }
        }
    }
    W(cu+10u,maximum);W(cu+16u,winner.x);W(cu+17u,winner.y);W(cu+18u,lowWindow);W(cu+19u,highWindow);
    if(unresolved){W(cu+13u,ST_TIED);return;}
    if(maximum<2u){W(cu+13u,ST_FEW);return;}
    if(2u*maximum<=informativeCount){W(cu+13u,ST_MAJORITY);return;}
    uint learned=0u,donors=0u;
    [loop]for(uint t=repeatedLow;t<=repeatedHigh&&t<Rmax&&t<128u;++t) {
        uint count=0u;
        [loop]for(uint band=0u;band<bandcount;++band)if(HasBit(winner,band)&&R(H(unit,band,t))!=0u)++count;
        if(count>=2u){learned=t;donors=count;}
    }
    W(cu+22u,learned);W(cu+24u,donors);
    if(learned==0u){W(cu+13u,ST_NO_REPEAT);return;}
    if(learned>16u){W(cu+13u,ST_T_CAP);W(cu+23u,R(cu+23u)|4u);uint ignored;Data.InterlockedOr(24u,1u<<unit,ignored);return;}
    // Publish add-only confidence only after all repeat, source and extent-cap checks.
    W(cu+UNIT_WIDTH_POLICY,sharedAddOnly?WIDTH_ADD_ONLY:WIDTH_REPLACE_RAW);
    W(cu+11u,learned);W(cu+12u,learned-1u);W(cu+13u,ST_KNOWN);uint ignored;Data.InterlockedOr(20u,1u<<unit,ignored);
}
static const uint TILE=46u,HALO=15u;
groupshared uint TileA[2116];
groupshared uint TileB[2116];
// GLOBAL full3x3 lookup: t=1 includes an isolated seed center as required by
// Chebyshev dilation. Halo-edge seeds can read one source pixel beyond the tile.
bool SeedAt(int2 p) {
    [unroll]for(int dy=-1;dy<=1;++dy)[unroll]for(int dx=-1;dx<=1;++dx) {
        int2 q=p+int2(dx,dy);
        if(OwnCover(q)&&(InnerEligible.Load(int3(q,0))==1u||
           (SelectedGeometryAt(q)&(1u<<Hook.x))!=0u))return true;
    }
    return false;
}
// PRIVATE BBSC1, packed OFFER extension16384. This is not a public CFB7 semantic.
// Only final publication can accept the offer, after a separate complete geometry validation.
// LOW8 BAND and every AA/closure donor predicate remain unchanged.
static const uint PRIVATE_CACHED_BLACK_OFFER=16384u,PRIVATE_LOCAL_CONFLICT_BLACK_OFFER=32768u;
static const uint PRIVATE_OFFER_MASK=PRIVATE_CACHED_BLACK_OFFER|PRIVATE_LOCAL_CONFLICT_BLACK_OFFER;
static const uint PRIVATE_AA_HISTORY_OFFER=1u<<24u;
// Current own-unit known/adopted opaque Raw4CC only. This separate bit cannot
// become ordinary AA/cache construction or borrow shared UNKNOWN_RAW/BAND evidence.
static const uint PRIVATE_OPAQUE_BLACK_OFFER=1u<<25u;
static const uint PRIVATE_PUBLISH_STRIP_MASK=PRIVATE_OFFER_MASK|PRIVATE_AA_HISTORY_OFFER|PRIVATE_OPAQUE_BLACK_OFFER;
static const uint STABILITY_AA_OLD_STRONG_PLAN=153u;
// Independent lifetimes: intact strong reference versus old global-reuse permission.
static const uint STABILITY_LEGACY_GLOBAL_VALID=152u,UNIT_CACHED_T_DONORS=37u;
static const uint STABILITY_TAG=0x31435342u,STABILITY_SCHEMA=5u,STABILITY_WORDS=160u;
// Header:0tag1schema2strongReferenceValid3snapshotFault4oldActive5legacyReuseFramePlan6geometryFault
// 7freshCurrentPlan;8..31committedCB24;32W33H34N35actualOriginalSRVformat
// 36..43committedT;44..67lastSeenCB24(every valid fresh call, not only commit)
// 68..99committedROI;100acceptedOfferPixels101newOfferPixels102commitPlan
// 103revocationReasons104epoch105privateOfferBit106privatePackedMask107lastSeenValid
// 108..123committedStaticProducer16;124committedHardCutEpoch;125..126reserved
// 127lastSeenHardCutEpoch;128..143lastSeenStaticProducer16
// 144completedFramePlan145completedReuseGate146completedCommit147snapshotWriteCount
// 148freshAdvancePlan149exactDuplicatePlan150completedFresh151completedDuplicate;152legacyGlobalValid;153oldStrongAAPlan; others reserved.
uint SW(uint i){return StabilityHeaderWrite[uint2(i,0u)];}
void SS(uint i,uint v){StabilityHeaderWrite[uint2(i,0u)]=v;}
uint SR(uint i){return StabilityHeaderRead.Load(int3(i,0u,0));}
bool StabilityHeaderROShape(){uint w,h;StabilityHeaderRead.GetDimensions(w,h);return w==STABILITY_WORDS&&h==1u;}
bool StabilityHeaderRWShape(){uint w,h;StabilityHeaderWrite.GetDimensions(w,h);return w==STABILITY_WORDS&&h==1u;}
bool StabilityImageROShape(){uint w,h;StabilityImageRead.GetDimensions(w,h);return all(uint2(w,h)==Shape.xy);}
bool StabilityImageRWShape(){uint w,h;StabilityImageWrite.GetDimensions(w,h);return all(uint2(w,h)==Shape.xy);}
bool StabilityAAHistoryShape(){uint w,h;AAHistory.GetDimensions(w,h);return all(uint2(w,h)==Shape.xy);}
bool StabilityValidRGB(float4 v){return v.a==1.0f&&all(isfinite(v.rgb))&&all(v.rgb>=0.0f)&&all(v.rgb<=1.0f);}
uint StabilityPackRGB(uint3 c){return c.r|(c.g<<8u)|(c.b<<16u);}
bool StabilityLater64(uint2 a,uint2 b){return a.y>b.y||(a.y==b.y&&a.x>b.x);}
bool CurrentCanonicalRibbonUnit(uint unit,out uint4 core) {
    core=0u;
    uint words,stride;CurrentOCR.GetDimensions(words,stride);
    if(words!=256u||stride!=4u||CurrentOCR[0u]!=4u||CurrentOCR[1u]!=0x3852434fu||CurrentOCR[2u]!=1u||
       CurrentOCR[3u]!=CurrentOCR[4u]||CurrentOCR[4u]>8u)return false;
    if(any(uint4(CurrentOCR[5u],CurrentOCR[6u],CurrentOCR[7u],CurrentOCR[8u])!=CFrame)||
       any(uint2(CurrentOCR[9u],CurrentOCR[10u])!=CSource.xy)||
       any(uint4(CurrentOCR[11u],CurrentOCR[12u],CurrentOCR[13u],CurrentOCR[14u])!=CField)||
       any(CContent.xy>=CContent.zw)||any(CContent.zw>CField.xy))return false;
    if(unit>=CurrentOCR[4u])return false;
    uint raw=16u+unit*8u;
    uint final=144u+unit*8u;
    if(CurrentOCR[raw+5u]!=1u||CurrentOCR[final+5u]!=1u)return false;
    core=uint4(CurrentOCR[raw],CurrentOCR[raw+1u],CurrentOCR[raw+2u],CurrentOCR[raw+3u]);
    uint4 cover=uint4(CurrentOCR[final],CurrentOCR[final+1u],CurrentOCR[final+2u],CurrentOCR[final+3u]);
    if(any(core.xy>=core.zw)||any(core.xy<CContent.xy)||any(core.zw>CContent.zw)||
       any(cover.xy>core.xy)||any(cover.zw<core.zw)||cover.x!=CContent.x||cover.z!=CContent.z||
       cover.w!=CContent.w||cover.y<CField.z)return false;
    if(any(uint4(CurrentOCR[raw+4u],CurrentOCR[raw+5u],CurrentOCR[raw+6u],CurrentOCR[raw+7u])!=
           uint4(CurrentOCR[final+4u],CurrentOCR[final+5u],CurrentOCR[final+6u],CurrentOCR[final+7u])))return false;
    uint ub=128u+32u*unit;
    if(CBCMeta[ub]!=1u||CBCMeta[ub+1u]!=1u||
       any(uint4(CBCMeta[ub+16u],CBCMeta[ub+17u],CBCMeta[ub+18u],CBCMeta[ub+19u])!=core))return false;
    return true;
}

bool StabilityCurrentUnits(out uint mask){
    mask=0u;uint n,stride;CurrentOCR.GetDimensions(n,stride);
    if(n!=256u||stride!=4u||CurrentOCR[0u]!=4u||CurrentOCR[1u]!=0x3852434fu||CurrentOCR[2u]!=1u)return false;
    uint count=CurrentOCR[4u];if(count==0u||count>8u)return false;
    [unroll]for(uint u=0u;u<8u;++u)if(u<count){
        uint4 uiCore;if(CurrentCanonicalRibbonUnit(u,uiCore))continue;
        uint b=128u+32u*u;
        if(CBCMeta[b]!=1u||CBCMeta[b+1u]!=1u||CBCMeta[b+5u]==0u)return false;
        mask|=1u<<u;
    }
    return mask!=0u;
}
uint StabilityStaticProducerWord(uint i){
    if(i==0u)return 43u; // postprocess mode
    if(i<7u)return 45u+i; // 46SRVformat,47transfer,48/49originWH,50resourceFormat,51samples
    if(i==7u)return 62u; // producer inputKnown
    return 56u+i; // 64..71 analysisROI and recipe/mip/viewdimension/bindFlags
}
bool StabilityStaticRW(uint first){
    [unroll]for(uint i=0u;i<16u;++i)if(SW(first+i)!=CBCMeta[StabilityStaticProducerWord(i)])return false;
    // CBC80 is the authenticated SCT/UIP durable hard-cut epoch, independent of C24 input reset.
    if(first==108u)return SW(124u)==CBCMeta[80u];
    return first==128u&&SW(127u)==CBCMeta[80u];
}
bool StabilityStaticRO(uint first){
    [unroll]for(uint i=0u;i<16u;++i)if(SR(first+i)!=CBCMeta[StabilityStaticProducerWord(i)])return false;
    if(first==108u)return SR(124u)==CBCMeta[80u];
    return first==128u&&SR(127u)==CBCMeta[80u];
}
bool StabilityLastTupleRW(){
    [unroll]for(uint i=0u;i<24u;++i)if(SW(44u+i)!=CBWord(i))return false;
    return StabilityStaticRW(128u);
}
bool StabilityFramePlanRW(){
    if(!StabilityHeaderRWShape()||SW(0u)!=STABILITY_TAG||SW(1u)!=STABILITY_SCHEMA||SW(7u)!=1u)return false;
    [unroll]for(uint i=0u;i<24u;++i)if(SW(44u+i)!=CBWord(i))return false;
    return StabilityStaticRW(128u);
}
bool StabilityFramePlanRO(){
    if(!StabilityHeaderROShape()||SR(0u)!=STABILITY_TAG||SR(1u)!=STABILITY_SCHEMA||SR(7u)!=1u)return false;
    [unroll]for(uint i=0u;i<24u;++i)if(SR(44u+i)!=CBWord(i))return false;
    return StabilityStaticRO(128u);
}
bool StabilityScopeRW(){
    [unroll]for(uint i=0u;i<12u;++i)if(SW(8u+i)!=CBWord(i))return false;
    if(SW(22u)!=CFrame.z||SW(23u)!=CFrame.w)return false;
    [unroll]for(uint k=21u;k<24u;++k)if(SW(8u+k)!=CBWord(k))return false;
    return SW(32u)==Shape.x&&SW(33u)==Shape.y&&SW(34u)==Shape.z&&SW(35u)==CBCMeta[46u]&&StabilityStaticRW(108u);
}
bool StabilityLastSeenScopeRW(){
    [unroll]for(uint i=0u;i<12u;++i)if(SW(44u+i)!=CBWord(i))return false;
    if(SW(58u)!=CFrame.z||SW(59u)!=CFrame.w)return false;
    [unroll]for(uint k=21u;k<24u;++k)if(SW(44u+k)!=CBWord(k))return false;
    return StabilityStaticRW(128u);
}
bool StabilityScopeRO(){
    [unroll]for(uint i=0u;i<12u;++i)if(SR(8u+i)!=CBWord(i))return false;
    if(SR(22u)!=CFrame.z||SR(23u)!=CFrame.w)return false;
    [unroll]for(uint k=21u;k<24u;++k)if(SR(8u+k)!=CBWord(k))return false;
    return SR(32u)==Shape.x&&SR(33u)==Shape.y&&SR(34u)==Shape.z&&SR(35u)==CBCMeta[46u]&&StabilityStaticRO(108u);
}
bool StabilityCompleteKnown(){
    uint mask;if(!InputReady()||!StabilityCurrentUnits(mask))return false;
    [unroll]for(uint u=0u;u<8u;++u)if((mask&(1u<<u))!=0u){
        uint b=U(u),k=PhysicalModelBank(u),r=128u+32u*u;
        if(OpaqueBlackOnly(u)||CBCMeta[k+2u]!=1u||!HasAdoptedPhysicalRefinement(u)||
           RawMeta[r]!=1u||RawMeta[r+1u]!=1u||R(b+14u)!=1u||!CurrentWidthMayReplaceRaw(u)||
           R(b+8u)==0u)return false;
    }
    return true;
}
bool StabilityOfferUnit(uint unit,out uint t){
    t=0u;if(unit>=8u||!StabilityHeaderROShape()||!StabilityImageROShape())return false;
    if(OpaqueBlackOnly(unit))return false;
    if(!StabilityFramePlanRO()||SR(5u)!=1u||SR(STABILITY_LEGACY_GLOBAL_VALID)!=1u||!StabilityScopeRO())return false;
    uint mask;if(!StabilityCurrentUnits(mask)||SR(4u)!=mask||(mask&(1u<<unit))==0u)return false;
    uint b=U(unit),r=128u+32u*unit;
    if(R(b+13u)!=ST_NO_INFORMATION||R(b+8u)!=0u||R(b+14u)!=1u||
       RawMeta[r]!=1u||!HasAdoptedPhysicalRefinement(unit)||CBCMeta[PhysicalModelBank(unit)+2u]!=1u)return false;
    t=SR(36u+unit);return t>0u&&t<=16u;
}
// Current conflict proof is computed once by calibration, not per grow texel.
// It only reuses an actually committed strong t, never derives a new extent.
bool StabilityStrongReferenceUnit(uint unit,out uint t){
    t=0u;if(unit>=8u||!StabilityHeaderROShape()||!StabilityImageROShape())return false;
    if(OpaqueBlackOnly(unit))return false;
    if(!StabilityFramePlanRO()||SR(2u)!=1u||!StabilityScopeRO())return false;
    uint mask;if(!StabilityCurrentUnits(mask)||SR(4u)!=mask||(mask&(1u<<unit))==0u)return false;
    uint b=U(unit),r=128u+32u*unit;
    if(R(b+14u)!=1u||RawMeta[r]!=1u||
       !HasAdoptedPhysicalRefinement(unit)||CBCMeta[PhysicalModelBank(unit)+2u]!=1u)return false;
    t=SR(36u+unit);return t>0u&&t<=16u;
}
bool CurrentSoftNonemptyConflict(uint unit){
    uint b=U(unit),status=R(b+13u);
    return R(b+14u)==1u&&R(b+8u)!=0u&&status>=ST_NO_INFORMATION&&status<=ST_NO_REPEAT;
}
[numthreads(1,1,1)]
void calibrate_main(uint3 group:SV_GroupID){
    CalibrateCurrentWidth(group);
    if(any(group)||Hook.x>=8u||any(Hook.yzw)||!DataShape())return;
    uint unit=Hook.x,b=U(unit);W(b+UNIT_CACHED_T_DONORS,0u);
    if(!InputReady()||!CurrentSoftNonemptyConflict(unit))return;
    uint t;if(!StabilityStrongReferenceUnit(unit,t))return;
    uint donors=0u;
    [loop]for(uint band=0u;band<R(b+7u);++band)if(R(H(unit,band,t))!=0u)++donors;
    W(b+UNIT_CACHED_T_DONORS,donors);
}
bool StabilityLocalConflictUnit(uint unit,out uint t){
    t=0u;return CurrentSoftNonemptyConflict(unit)&&R(U(unit)+UNIT_CACHED_T_DONORS)>=2u&&
        StabilityStrongReferenceUnit(unit,t);
}
bool StabilityInnerAt(int2 p){
    uint u=128u+32u*Hook.x;
    int4 roi=int4(CBCMeta[u+24u],CBCMeta[u+25u],CBCMeta[u+26u],CBCMeta[u+27u]);
    return all(p>=roi.xy)&&all(p<roi.zw)&&OwnCover(p)&&InnerEligible.Load(int3(p,0))==1u;
}
bool StabilityOfferPixel(int2 p){
    if(!ValidPoint(p)||!Black(p))return false;
    float4 v=CurrentRGB.Load(int3(p,0));if(!StabilityValidRGB(v))return false;
    uint old=StabilityImageRead.Load(int3(p,0));
    return (old&0x00ffffffu)==StabilityPackRGB(RGB8(p));
}
bool StabilityCurrentInsideAt(int2 p){
    uint w,h;CurrentInsideBits.GetDimensions(w,h);
    if(any(uint2(w,h)!=Shape.xy)||Hook.x>=8u)return false;
    uint unitBit=1u<<Hook.x,good=R(79u)&~R(80u)&255u;
    uint u=128u+32u*Hook.x;
    int4 roi=int4(CBCMeta[u+24u],CBCMeta[u+25u],CBCMeta[u+26u],CBCMeta[u+27u]);
    return (good&unitBit)!=0u&&all(p>=roi.xy)&&all(p<roi.zw)&&OwnCover(p)&&
        (CurrentInsideBits.Load(int3(p,0))&unitBit)!=0u;
}
bool StabilityLocalInnerAt(int2 p){
    bool trustedInside=StabilityCurrentInsideAt(p);
    if(!StabilityInnerAt(p)&&!trustedInside)return false;
    float4 v=CurrentRGB.Load(int3(p,0));if(!StabilityValidRGB(v))return false;
    uint old=StabilityImageRead.Load(int3(p,0));
    return ((old>>24u)&(1u<<Hook.x))!=0u&&
        (trustedInside||(old&0x00ffffffu)==StabilityPackRGB(RGB8(p)));
}
uint StabilityLocalDirectHorizontal(int2 p,uint radius){
    uint best=radius*radius+1u;
    [loop]for(int dx=-(int)radius;dx<=(int)radius;++dx)
        if(StabilityLocalInnerAt(p+int2(dx,0)))best=min(best,(uint)(dx*dx));
    return best;
}
uint StabilityDirectHorizontal(int2 p,uint radius){
    uint best=radius*radius+1u;
    [loop]for(int dx=-(int)radius;dx<=(int)radius;++dx)
        if(StabilityInnerAt(p+int2(dx,0)))best=min(best,(uint)(dx*dx));
    return best;
}
bool StabilityOfferAcceptedRW(){
    return StabilityFramePlanRW()&&SW(5u)==1u&&SW(STABILITY_LEGACY_GLOBAL_VALID)==1u&&SW(6u)==0u;
}
bool StabilityLocalOfferAcceptedRW(){
    // Local offers carry their own stable current/old seed proof. A global LOW
    // mismatch revokes legacy reuse, not the intact strong RGB/LOW reference.
    return StabilityFramePlanRW()&&SW(2u)==1u&&StabilityScopeRW();
}
bool StabilitySnapshotWritePlan(){
    // No mutable Data fault/publication word participates in the full-image write predicate.
    return StabilityFramePlanRW()&&StabilityImageRWShape()&&StabilityAAHistoryShape()&&SW(148u)==1u&&SW(102u)==1u;
}
// This authorization is frozen by begin BEFORE validate invalidates the old
// strong-valid words. It is independent of legacy LOW-geometry reuse.
bool StabilityAAHistoryPlan(){
    return StabilityHeaderRWShape()&&StabilityAAHistoryShape()&&
        SW(STABILITY_AA_OLD_STRONG_PLAN)==1u&&StabilityFramePlanRW();
}
bool StabilityAAHistoryOffer(uint2 q){
    if(!StabilityAAHistoryPlan()||!StabilityImageROShape()||(AAHistory[q]&2u)==0u)return false;
    if(!WeakAAPlan()&&(V3M[q]&(OLD_AA|SUPPORT_AA))!=0u)return false; // Confidence shares actual admitted AA at publication, not raw candidates.
    int2 center=int2(q);
    // Exact current/old RGB at ALL nine samples; construction comparison omits
    // q itself and never includes private offers or retained-AA output.
    [unroll]for(int dy=-1;dy<=1;++dy)[unroll]for(int dx=-1;dx<=1;++dx){
        int2 p=center+int2(dx,dy);
        if(any(p<0)||any(p>=int2(Shape.xy)))return false;
        float4 v=CurrentRGB.Load(int3(p,0));
        if(!StabilityValidRGB(v)||
           (StabilityImageRead.Load(int3(p,0))&0x00ffffffu)!=StabilityPackRGB(RGB8(p)))return false;
        if(dx!=0||dy!=0){
            bool oldConstruction=(AAHistory[(uint2)p]&1u)!=0u;
            bool nowConstruction=(V3M[(uint2)p]&0x3fffu)!=0u;
            if(oldConstruction!=nowConstruction)return false;
        }
    }
    uint owners=CoverBits.Load(int3(center,0))&SW(4u)&255u;
    [unroll]for(uint unit=0u;unit<8u;++unit)if((owners&(1u<<unit))!=0u){
        if(OpaqueBlackOnly(unit))continue;
        uint ru=128u+32u*unit,bit=1u<<unit;
        if(CBCMeta[ru]!=1u||CBCMeta[ru+1u]!=1u||!HasAdoptedPhysicalRefinement(unit)||
           CBCMeta[PhysicalModelBank(unit)+2u]!=1u||RawMeta[ru]!=1u||RawMeta[ru+1u]!=1u||R(U(unit)+14u)!=1u)continue;
        int4 roi=int4(CBCMeta[ru+24u],CBCMeta[ru+25u],CBCMeta[ru+26u],CBCMeta[ru+27u]);
        if(any(roi.xy<0)||any(roi.xy>=roi.zw)||any(roi.zw>int2(Shape.xy))||
           any(center<roi.xy)||any(center>=roi.zw))continue;
        [unroll]for(int sy=-1;sy<=1;++sy)[unroll]for(int sx=-1;sx<=1;++sx)if(sx!=0||sy!=0){
            int2 p=center+int2(sx,sy);
            if(any(p<roi.xy)||any(p>=roi.zw)||(CoverBits.Load(int3(p,0))&bit)==0u||
               (V3M[(uint2)p]&bit)==0u||RawM.Load(int3(p,0))==0u)continue;
            // RawM presence alone is not physical-black authority. The same
            // current owner must also classify this distinct measured donor.
            if(PhysicalBlackRGB(CurrentRGB.Load(int3(p,0)).rgb,unit))return true;
        }
    }
    return false;
}
// PRIVATE WAA2: restrict only a conservatively observed-appearance episode; cold evidence stays legacy.
// Header0..255, then N same-q RGB24/witness/candidate/admitted/covered/non-UI words.
static const uint WEAK_TAG=0x31414157u,WEAK_SCHEMA=2u,WEAK_HEADER=256u;
static const uint WEAK_WITNESS=1u<<24u,WEAK_CANDIDATE=1u<<25u,WEAK_ADMITTED=1u<<26u,WEAK_RGB_VALID=1u<<27u,
    WEAK_COVERED=1u<<28u,WEAK_NON_UI=1u<<29u;
bool WeakAAShape(){uint n,s;WeakAAState.GetDimensions(n,s);return s==4u&&n==WEAK_HEADER+Shape.z;}
bool WeakAAPlan(){
    if(!WeakAAShape()||WeakAAState[0u]!=WEAK_TAG||WeakAAState[1u]!=WEAK_SCHEMA||WeakAAState[4u]==0u)return false;
    [unroll]for(uint i=0u;i<24u;++i)if(WeakAAState[16u+i]!=CBWord(i))return false;
    return true;
}
bool WeakAAUnits(out uint mask){
    mask=0u;uint n,s;CurrentOCR.GetDimensions(n,s);
    if(n!=256u||s!=4u||CurrentOCR[0u]!=4u||CurrentOCR[1u]!=0x3852434fu||CurrentOCR[2u]!=1u||
       CurrentOCR[3u]!=CurrentOCR[4u]||CurrentOCR[4u]>8u)return false;
    if(any(uint4(CurrentOCR[5u],CurrentOCR[6u],CurrentOCR[7u],CurrentOCR[8u])!=CFrame)||
       any(uint2(CurrentOCR[9u],CurrentOCR[10u])!=CSource.xy)||
       any(uint4(CurrentOCR[11u],CurrentOCR[12u],CurrentOCR[13u],CurrentOCR[14u])!=CField))return false;
    [unroll]for(uint u=0u;u<8u;++u)if(u<CurrentOCR[4u]){
        uint4 core;if(CurrentCanonicalRibbonUnit(u,core))continue;
        uint b=128u+32u*u;
        if(CBCMeta[b]!=1u||CBCMeta[b+1u]!=1u)return false;
        mask|=1u<<u;
    }
    return true; // Valid EMPTY/UI-only is a real RGB anchor, with no glyph authority.
}
bool WeakAAAnchorScope(){
    [unroll]for(uint i=0u;i<12u;++i)if(WeakAAState[40u+i]!=CBWord(i))return false;
    if(WeakAAState[54u]!=CFrame.z||WeakAAState[55u]!=CFrame.w)return false;
    [unroll]for(uint i=21u;i<24u;++i)if(WeakAAState[40u+i]!=CBWord(i))return false;
    [unroll]for(uint i=0u;i<16u;++i)if(WeakAAState[64u+i]!=CBCMeta[StabilityStaticProducerWord(i)])return false;
    return WeakAAState[80u]==CBCMeta[80u];
}
bool WeakAAEpisode(uint mask){
    if(WeakAAState[81u]!=mask)return false;
    [unroll]for(uint u=0u;u<8u;++u)if((mask&(1u<<u))!=0u){
        // Episode continuity tolerates observed OCR box jitter without a distance threshold.
        // Same-q flags were zero outside the old cover; current use also requires current cover.
        [unroll]for(uint role=0u;role<2u;++role){
            uint oldBase=88u+16u*u+8u*role,currentBase=(role==0u?16u:144u)+8u*u;
            if(WeakAAState[oldBase+5u]!=CurrentOCR[currentBase+5u])return false;
            uint4 oldRect=uint4(WeakAAState[oldBase],WeakAAState[oldBase+1u],WeakAAState[oldBase+2u],WeakAAState[oldBase+3u]);
            uint4 currentRect=uint4(CurrentOCR[currentBase],CurrentOCR[currentBase+1u],CurrentOCR[currentBase+2u],CurrentOCR[currentBase+3u]);
            if(any(oldRect.xy>=oldRect.zw)||any(currentRect.xy>=currentRect.zw)||
               any(max(oldRect.xy,currentRect.xy)>=min(oldRect.zw,currentRect.zw)))return false;
        }
    }
    return true;
}
void WeakAABegin(){
    if(!WeakAAShape())return;
    bool header=WeakAAState[0u]==WEAK_TAG&&WeakAAState[1u]==WEAK_SCHEMA;
    uint mask=0u;bool proof=InputReady()&&WeakAAUnits(mask)&&any(CFrame.xy!=0u)&&any(CObservation.xy!=0u);
    bool seen=header&&WeakAAState[11u]==1u,duplicate=proof&&seen;
    [unroll]for(uint i=0u;i<24u;++i)if(seen&&WeakAAState[16u+i]!=CBWord(i))duplicate=false;
    bool sameScope=header&&WeakAAAnchorScope();
    duplicate=duplicate&&sameScope; // Same24CB alone cannot authenticate changed producer/cut/scope.
    bool newer=!seen||(StabilityLater64(CObservation.xy,uint2(WeakAAState[32u],WeakAAState[33u]))&&
        (StabilityLater64(CFrame.xy,uint2(WeakAAState[28u],WeakAAState[29u]))||!sameScope||CObservation.z!=0u));
    bool fresh=proof&&newer;
    bool anchor=header&&WeakAAState[2u]==1u&&sameScope&&(!fresh||CObservation.z==0u);
    bool episode=anchor&&WeakAAEpisode(mask);
    uint qualified=episode?(WeakAAState[232u]&mask):0u;
    uint sampleMask=anchor&&WeakAAState[81u]==0u?mask:0u;
    if(!header){[unroll]for(uint i=0u;i<WEAK_HEADER;++i)WeakAAState[i]=0u;}
    WeakAAState[0u]=WEAK_TAG;WeakAAState[1u]=WEAK_SCHEMA;
    WeakAAState[4u]=fresh?1u:(duplicate?2u:0u);WeakAAState[5u]=anchor?1u:0u;
    WeakAAState[6u]=episode?1u:0u;WeakAAState[7u]=mask;
    WeakAAState[233u]=qualified;WeakAAState[234u]=sampleMask;
    if(!proof){WeakAAState[2u]=0u;WeakAAState[232u]=0u;return;}
    // No stale/duplicate call advances the epoch or changes a pixel. Plan fields are transient.
    if(!fresh)return;
    WeakAAState[2u]=0u;WeakAAState[3u]=0u;WeakAAState[8u]=0u;WeakAAState[9u]=Shape.z;
    WeakAAState[10u]=mask==0u?1u:0u;WeakAAState[13u]=0u;WeakAAState[14u]=0u;WeakAAState[15u]=0u;
    [unroll]for(uint i=216u;i<232u;++i)WeakAAState[i]=0u;
    WeakAAState[235u]=0u;
    [unroll]for(uint i=0u;i<24u;++i)WeakAAState[16u+i]=CBWord(i);
    WeakAAState[11u]=1u;
}
// Exactly64-or-fewer native sites per unit, only at a genuine EMPTY/UI-only -> glyph
// transition with an authenticated old full-image RGB anchor. This is evidence of
// changed trusted foreground, not a claim OCR-empty proved physical absence.
// Old pixel words remain immutable throughout this prepublication dispatch.
void WeakAASampleAppearance(uint2 lane){
    if(any(lane>=8u)||!WeakAAPlan()||WeakAAState[4u]!=1u||WeakAAState[234u]==0u||!InputReady())return;
    uint mask=WeakAAState[234u];
    [unroll]for(uint unit=0u;unit<8u;++unit)if((mask&(1u<<unit))!=0u){
        uint b=128u+32u*unit,bit=1u<<unit;
        uint4 roi=uint4(CBCMeta[b+24u],CBCMeta[b+25u],CBCMeta[b+26u],CBCMeta[b+27u]);
        if(any(roi.xy>=roi.zw)||any(roi.zw>Shape.xy))continue;
        uint2 size=roi.zw-roi.xy,n=min(uint2(8u,8u),size);
        if(any(lane>=n))continue;
        uint2 q=roi.xy+((2u*lane+1u)*size)/(2u*n);
        if((SelectedForegroundBits(SelectedBits.Load(int3(q,0)))&CoverBits.Load(int3(q,0))&bit)==0u)continue;
        float4 current=CurrentRGB.Load(int3(q,0));uint old=WeakAAState[WEAK_HEADER+q.y*Shape.x+q.x];
        if(!StabilityValidRGB(current)||(old&WEAK_RGB_VALID)==0u)continue;
        uint ignored;InterlockedAdd(WeakAAState[216u+2u*unit],1u,ignored);
        if((old&0x00ffffffu)!=StabilityPackRGB(RGB8(int2(q))))InterlockedAdd(WeakAAState[217u+2u*unit],1u,ignored);
    }
}
bool WeakAAAdmitAndAdvance(uint2 q,bool directAA,bool independentCurrent,bool retainedAA,bool actualNonUi){
    if(!WeakAAPlan())return directAA; // Unbound/unavailable/invalid owner preserves legacy behavior.
    uint index=WEAK_HEADER+q.y*Shape.x+q.x,old=WeakAAState[index];
    bool anchor=WeakAAState[5u]==1u&&(old&WEAK_RGB_VALID)!=0u;
    bool episode=anchor&&WeakAAState[6u]==1u;
    if(WeakAAState[4u]==2u)return !episode?directAA:(directAA&&(old&WEAK_ADMITTED)!=0u);
    float4 source=CurrentRGB.Load(int3(q,0));bool rgbValid=StabilityValidRGB(source);
    uint rgb=rgbValid?StabilityPackRGB(RGB8(int2(q))):0u;
    uint owners=CoverBits.Load(int3(q,0))&WeakAAState[7u]&255u;
    bool glyphHere=owners!=0u;
    bool same=anchor&&(old&0x00ffffffu)==rgb;
    bool witness=glyphHere&&rgbValid&&anchor&&(!same||(episode&&same&&(old&WEAK_WITNESS)!=0u));
    // All overlapping current owners must belong to the previously qualified episode.
    // New coverage or prior actual non-UI protection may never acquire a new wait.
    bool qualified=episode&&glyphHere&&(owners&~WeakAAState[233u])==0u;
    bool fallback=!rgbValid||!anchor||!episode||!glyphHere||!qualified||(old&WEAK_COVERED)==0u;
    bool admit=directAA&&(independentCurrent||fallback||witness||
        (episode&&(old&(WEAK_CANDIDATE|WEAK_ADMITTED|WEAK_NON_UI))!=0u));
    uint next=rgbValid?(rgb|WEAK_RGB_VALID):0u;
    if(glyphHere){next|=WEAK_COVERED;if(witness)next|=WEAK_WITNESS;if(directAA)next|=WEAK_CANDIDATE;
        if(admit||retainedAA)next|=WEAK_ADMITTED;if(actualNonUi)next|=WEAK_NON_UI;}
    WeakAAState[index]=next;
    uint ignored;if(!rgbValid)InterlockedOr(WeakAAState[3u],1u,ignored);
    if(directAA&&!admit)InterlockedAdd(WeakAAState[13u],1u,ignored);
    // Disjoint categories; the one existing commit folds fallback into total admitted.
    if(admit)InterlockedAdd(WeakAAState[fallback?15u:14u],1u,ignored);
    if(all((q&15u)==0u)){
        uint covered=min(16u,Shape.x-q.x)*min(16u,Shape.y-q.y);
        InterlockedAdd(WeakAAState[8u],covered,ignored);
    }
    return admit;
}
void WeakAACommit(){
    if(!WeakAAPlan()||WeakAAState[4u]!=1u)return;
    // Fold even a faulted/partial publication, preserving the post-helper QA counts.
    WeakAAState[14u]+=WeakAAState[15u];
    bool valid=InputReady()&&WeakAAState[3u]==0u&&WeakAAState[8u]==Shape.z;
    WeakAAState[2u]=0u;WeakAAState[232u]=0u;
    if(valid){
        uint qualified=WeakAAState[233u]&WeakAAState[7u];
        [unroll]for(uint unit=0u;unit<8u;++unit)if((WeakAAState[234u]&(1u<<unit))!=0u){
            uint samples=WeakAAState[216u+2u*unit],changed=WeakAAState[217u+2u*unit];
            if(samples>=4u&&changed>samples/2u)qualified|=1u<<unit;
        }
        WeakAAState[232u]=qualified;WeakAAState[235u]=qualified&~WeakAAState[233u];
        [unroll]for(uint i=0u;i<24u;++i)WeakAAState[40u+i]=CBWord(i);
        [unroll]for(uint i=0u;i<16u;++i)WeakAAState[64u+i]=CBCMeta[StabilityStaticProducerWord(i)];
        WeakAAState[80u]=CBCMeta[80u];WeakAAState[81u]=WeakAAState[7u];
        [unroll]for(uint u=0u;u<8u;++u)[unroll]for(uint k=0u;k<8u;++k){
            WeakAAState[88u+16u*u+k]=CurrentOCR[16u+8u*u+k];
            WeakAAState[96u+16u*u+k]=CurrentOCR[144u+8u*u+k];
        }
        WeakAAState[12u]+=1u;WeakAAState[2u]=1u;
    }
}

[numthreads(1,1,1)]
void private_cache_begin_main(uint3 g:SV_GroupID){
    if(any(g))return;
    WeakAABegin();
    if(!StabilityHeaderRWShape())return;
    bool header=SW(0u)==STABILITY_TAG&&SW(1u)==STABILITY_SCHEMA;
    bool wasValid=header&&SW(2u)==1u;
    bool legacyValid=wasValid&&SW(STABILITY_LEGACY_GLOBAL_VALID)==1u;
    uint reasons=0u,mask=0u;bool proof=InputReady()&&StabilityImageROShape();
    bool current=proof&&StabilityCurrentUnits(mask);
    bool hasLast=header&&SW(107u)==1u;
    bool sameDomain=hasLast&&StabilityLastSeenScopeRW();
    bool exactDuplicate=proof&&hasLast&&StabilityLastTupleRW();
    bool observationNew=!hasLast||StabilityLater64(CObservation.xy,uint2(SW(60u),SW(61u)));
    bool frameNew=!hasLast||StabilityLater64(CFrame.xy,uint2(SW(56u),SW(57u)));
    bool reset=CObservation.z!=0u;
    // A domain/reset may restart frame IDs, but never bypass the material observation-clock high water.
    bool freshAdvance=proof&&(!hasLast||(observationNew&&(frameNew||!sameDomain||reset)));
    bool authorizedTuple=proof&&(freshAdvance||exactDuplicate);
    bool authorizedCurrent=authorizedTuple&&current;
    bool scope=wasValid&&StabilityScopeRW();
    if(!proof||!current)reasons|=1u;
    if(wasValid&&authorizedTuple&&!scope)reasons|=2u;
    if(proof&&!authorizedTuple)reasons|=4u;
    if(wasValid&&authorizedCurrent&&SW(4u)!=mask)reasons|=8u;
    if(freshAdvance&&reset)reasons|=16u;
    // Proof-valid stale calls have no authority to reset/EMPTY/re-scope a newer committed cache.
    // Exact duplicate reset1 is the same event, not a fresh domain reset.
    if(!proof)wasValid=false;
    else if(authorizedTuple&&(!current||(freshAdvance&&reset)||(wasValid&&(!scope||SW(4u)!=mask))))wasValid=false;
    SS(0u,STABILITY_TAG);SS(1u,STABILITY_SCHEMA);SS(2u,wasValid?1u:0u);
    SS(STABILITY_LEGACY_GLOBAL_VALID,wasValid&&legacyValid?1u:0u);
    if(!header)SS(107u,0u);
    SS(3u,0u);SS(5u,wasValid&&legacyValid&&authorizedCurrent&&scope?1u:0u);SS(6u,0u);
    SS(7u,authorizedCurrent?1u:0u);SS(100u,0u);SS(101u,0u);SS(102u,0u);SS(103u,reasons);SS(147u,0u);
    SS(148u,freshAdvance?1u:0u);SS(149u,exactDuplicate?1u:0u);
    SS(STABILITY_AA_OLD_STRONG_PLAN,wasValid&&authorizedCurrent&&scope&&SW(4u)==mask&&StabilityAAHistoryShape()?1u:0u);
    SS(105u,PRIVATE_PUBLISH_STRIP_MASK);SS(106u,65535u|PRIVATE_AA_HISTORY_OFFER|PRIVATE_OPAQUE_BLACK_OFFER);
    // Fresh valid EMPTY advances LastSeen too. Duplicates and stale tuples never advance it.
    if(freshAdvance){
        [unroll]for(uint j=0u;j<24u;++j)SS(44u+j,CBWord(j));
        [unroll]for(uint k=0u;k<16u;++k)SS(128u+k,CBCMeta[StabilityStaticProducerWord(k)]);
        SS(127u,CBCMeta[80u]);SS(107u,1u);
    }
}
[numthreads(16,16,1)]
void private_cache_validate_main(uint3 d:SV_DispatchThreadID){
    if(d.z!=0u||!Native(d.xy))return;
    WeakAASampleAppearance(d.xy);
    if(!StabilityHeaderRWShape())return;
    bool ready=StabilityFramePlanRW()&&InputReady();
    if(all(d.xy==0u)){
        bool complete=ready&&SW(148u)==1u&&StabilityCompleteKnown()&&StabilityImageROShape()&&StabilityAAHistoryShape();
        SS(102u,complete?1u:0u);
        // Ordered before final snapshot writes: the old valid header must not cover a partial new image.
        if(complete){SS(2u,0u);SS(STABILITY_LEGACY_GLOBAL_VALID,0u);}
    }
    // AA reads old RGB/history only in this phase. Atomic writes touch bit24;
    // the compared lower construction bits are immutable throughout dispatch.
    if(ready&&StabilityAAHistoryOffer(d.xy)){uint ignored;InterlockedOr(V3M[d.xy],PRIVATE_AA_HISTORY_OFFER,ignored);}
    // The begin-plan is immutable during this dispatch; never gate geometry on mutable committedValid.
    if(!ready||SW(5u)!=1u||!StabilityImageROShape())return;
    uint old=StabilityImageRead.Load(int3(d.xy,0))>>24u;
    uint now=SelectedGeometryAt(int2(d.xy));uint nearOld=0u,nearNow=0u;
    [unroll]for(int dy=-1;dy<=1;++dy)[unroll]for(int dx=-1;dx<=1;++dx){
        int2 q=int2(d.xy)+int2(dx,dy);if(any(q<0)||any(q>=int2(Shape.xy)))continue;
        nearOld|=StabilityImageRead.Load(int3(q,0))>>24u;nearNow|=SelectedGeometryAt(q);
    }
    if((old&~nearNow)!=0u||(now&~nearOld)!=0u){uint ignored;InterlockedOr(StabilityHeaderWrite[uint2(6u,0u)],1u,ignored);}
}
[numthreads(1,1,1)]
void private_cache_commit_main(uint3 g:SV_GroupID){
    if(any(g))return;
    WeakAACommit();
    if(!StabilityHeaderRWShape())return;
    uint mask=0u;bool current=InputReady()&&StabilityCurrentUnits(mask);
    bool plan=StabilityFramePlanRW(),fresh=SW(148u)==1u,duplicate=SW(149u)==1u;
    bool known=plan&&current&&StabilityCompleteKnown()&&StabilityImageROShape()&&StabilityAAHistoryShape();
    bool snapshotPlanned=SW(102u)==1u;
    bool commit=plan&&fresh&&snapshotPlanned&&SW(3u)==0u&&SW(147u)==Shape.z&&known;
    SS(144u,plan?1u:0u);SS(145u,plan&&SW(5u)==1u&&SW(6u)==0u?1u:0u);SS(146u,commit?1u:0u);
    SS(150u,fresh?1u:0u);SS(151u,duplicate?1u:0u);
    if(commit){
        SS(2u,0u);SS(4u,mask);
        [unroll]for(uint j=0u;j<24u;++j)SS(8u+j,CBWord(j));
        SS(32u,Shape.x);SS(33u,Shape.y);SS(34u,Shape.z);SS(35u,CBCMeta[46u]);
        [unroll]for(uint k=0u;k<16u;++k)SS(108u+k,CBCMeta[StabilityStaticProducerWord(k)]);
        SS(124u,CBCMeta[80u]);
        [unroll]for(uint u=0u;u<8u;++u){
            SS(36u+u,(mask&(1u<<u))!=0u?R(U(u)+11u):0u);
            [unroll]for(uint z=0u;z<4u;++z)SS(68u+4u*u+z,(mask&(1u<<u))!=0u?CBCMeta[128u+32u*u+24u+z]:0u);
        }
        SS(104u,SW(104u)+1u);SS(STABILITY_LEGACY_GLOBAL_VALID,1u);SS(2u,1u); // Valid only after the preceding full-image dispatch completed.
    }else {
        bool incompleteSnapshot=snapshotPlanned&&(!plan||!fresh||!known||SW(147u)!=Shape.z);
        // Begin already handles invalid proof and authorized fresh EMPTY/reset/scope changes.
        // Valid stale tuples and valid duplicate replays must not destroy a newer persistent cache.
        if(SW(3u)!=0u||incompleteSnapshot||(plan&&!current)){
            SS(2u,0u);SS(STABILITY_LEGACY_GLOBAL_VALID,0u);
        }else if(SW(6u)!=0u)SS(STABILITY_LEGACY_GLOBAL_VALID,0u);
    }
    // Consume transient authorization. Exact replay authorization is rebuilt by the next begin.
    SS(7u,0u);SS(5u,0u);SS(102u,0u);SS(148u,0u);SS(149u,0u);SS(STABILITY_AA_OLD_STRONG_PLAN,0u);
}


[numthreads(16,16,1)]
void grow_tile_main(uint3 group:SV_GroupID,uint3 tid:SV_GroupThreadID,uint lane:SV_GroupIndex,uint3 d:SV_DispatchThreadID) {
    bool input=InputReady()&&Hook.x<8u,known=false,cached=false,localConflict=false;uint t=1u,cacheT=0u;
    // Raw's own black4CC does not exist yet at this stage. K4 may not replace
    // that proof with a BAND/N8 envelope or an old width/snapshot offer.
    // Keep every lane in the shared work/barriers: FXC treats resource-backed
    // readiness as varying even when the serialized owner keeps it immutable.
    if(input&&OpaqueBlackOnly(Hook.x))input=false;
    uint2 origin=0u,end=0u;
    if(input){uint cu=U(Hook.x);t=R(cu+11u);known=CurrentWidthAvailable(Hook.x);
        cached=!known&&StabilityOfferUnit(Hook.x,cacheT);
        if(!known&&!cached)localConflict=StabilityLocalConflictUnit(Hook.x,cacheT);
        if(cached||localConflict)t=cacheT;
        if(known||cached||localConflict){uint ub=128u+32u*Hook.x;origin=uint2(CBCMeta[ub+24u],CBCMeta[ub+25u]);end=uint2(CBCMeta[ub+26u],CBCMeta[ub+27u]);}
    }
    bool active=known||cached||localConflict;uint radius=known?t-1u:(cached||localConflict)?t:0u;
    int2 tileOrigin=int2(origin+group.xy*16u)-int2(HALO,HALO);
    [loop]for(uint i=lane;i<2116u;i+=256u){int2 p=tileOrigin+int2(i%TILE,i/TILE);TileA[i]=known?(SeedAt(p)?1u:0u):localConflict?(StabilityLocalInnerAt(p)?1u:0u):cached&&StabilityInnerAt(p)?1u:0u;}
    GroupMemoryBarrierWithGroupSync();
    [loop]for(uint i=lane;i<2116u;i+=256u){
        int x=(int)(i%TILE);uint row=(i/TILE)*TILE;uint value=(cached||localConflict)?t*t+1u:0u;
        [loop]for(int dx=-(int)radius;dx<=(int)radius;++dx){
            int q=x+dx;
            if(known){if(q>=0&&q<(int)TILE)value|=TileA[row+(uint)q];}
            else if(cached||localConflict){
                // Cachedt may16: a single outer column lies beyond the unchanged HALO15 tile.
                bool inner=false;
                [branch]if(q>=0&&q<(int)TILE)inner=TileA[row+(uint)q]!=0u;
                else inner=localConflict?StabilityLocalInnerAt(tileOrigin+int2(q,(int)(i/TILE))):StabilityInnerAt(tileOrigin+int2(q,(int)(i/TILE)));
                if(inner)value=min(value,(uint)(dx*dx));
            }
        }
        TileB[i]=value;
    }
    GroupMemoryBarrierWithGroupSync();
    uint2 p=origin+d.xy;
    if(active&&d.z==0u&&all(p<end)&&Native(p)&&ValidPoint(int2(p))&&Black(int2(p))){
        uint x=tid.x+HALO;int y=(int)(tid.y+HALO);uint value=(cached||localConflict)?t*t+1u:0u;
        [loop]for(int dy=-(int)radius;dy<=(int)radius;++dy){
            if(known)value|=TileB[(uint)(y+dy)*TILE+x];
            else {int row=y+dy;uint horizontal=t*t+1u;
                [branch]if(row>=0&&row<(int)TILE)horizontal=TileB[(uint)row*TILE+x];
                else horizontal=localConflict?StabilityLocalDirectHorizontal(int2(p)+int2(0,dy),t):StabilityDirectHorizontal(int2(p)+int2(0,dy),t);
                value=min(value,horizontal+(uint)(dy*dy));}
        }
        if(known&&value!=0u){uint ignored;InterlockedOr(V3M[p],1u<<Hook.x,ignored);}
        if((cached||localConflict)&&value<=t*t&&StabilityOfferPixel(int2(p))){uint ignored;
            InterlockedOr(V3M[p],localConflict?PRIVATE_LOCAL_CONFLICT_BLACK_OFFER:PRIVATE_CACHED_BLACK_OFFER,ignored);
        }
    }
}
// CFB5 final-only local Code1-collar certificates (called only for valid residuals). Immutable current source/BODY/model only.
// No parent graph, mutable packed neighbour, alpha oracle or history is read.
bool CertValid(int2 p,uint bit) {
    if(any(p<0)||any(p>=int2(Shape.xy))||(CoverBits.Load(int3(p,0))&bit)==0u)return false;
    float4 v=CurrentRGB.Load(int3(p,0));
    return v.a==1.0f&&all(isfinite(v.rgb))&&all(v.rgb>=0.0f)&&all(v.rgb<=1.0f);
}
bool CertOwnBody(int2 p,uint bit) {
    return CertValid(p,bit)&&(SelectedGeometryAt(p)&bit)!=0u;
}
bool CertBlack(int2 p,uint unit,uint bit) {
    return CertValid(p,bit)&&PhysicalBlackRGB(CurrentRGB.Load(int3(p,0)).rgb,unit);
}
uint3 CertCode(int2 p) { return PhysicalRGB8(CurrentRGB.Load(int3(p,0)).rgb); }
uint3 CertCellLow(uint3 c) { return 2u*c-(uint3)(c>0u); }
uint3 CertCellHigh(uint3 c) { return min(uint3(510u,510u,510u),2u*c+1u); }
// Exact signed A*s>=B interval intersection on s in[0,1]. Cross-products <=260100.
bool CertConstrain(int coefficient,int rhs,inout uint ln,inout uint ld,inout uint un,inout uint ud) {
    if(coefficient>0) {
        if(rhs>0) {uint n=(uint)rhs,d=(uint)coefficient;if(n*ld>ln*d){ln=n;ld=d;}}
    } else if(coefficient<0) {
        if(rhs>0)return false;
        uint n=(uint)(-rhs),d=(uint)(-coefficient);
        if(n*ud<un*d){un=n;ud=d;}
    } else if(rhs>0)return false;
    return ln*ud<=un*ld;
}
// One exact all-feasible interval supplies both existing visibility tests.
// s is endpoint fraction; 1-s is black fraction. No new tolerance is introduced.
static const uint CERT_ENDPOINT_VISIBLE=1u, CERT_BLACK_VISIBLE=2u;
uint CertBlendVisibility(uint3 q,uint3 endpoint,uint3 blackLow,uint3 blackHigh) {
    uint3 qlo=CertCellLow(q),qhi=CertCellHigh(q),elo=CertCellLow(endpoint),ehi=CertCellHigh(endpoint);
    uint3 blo=CertCellLow(blackLow),bhi=CertCellHigh(blackHigh);
    uint ln=0u,ld=1u,un=1u,ud=1u;
    [unroll]for(uint c=0u;c<3u;++c) {
        if(!CertConstrain((int)ehi[c]-(int)bhi[c],(int)qlo[c]-(int)bhi[c],ln,ld,un,ud))return 0u;
        if(!CertConstrain((int)blo[c]-(int)elo[c],(int)blo[c]-(int)qhi[c],ln,ld,un,ud))return 0u;
    }
    if(ln*ud>un*ld)return 0u;
    return (ln*255u>=ld*5u?CERT_ENDPOINT_VISIBLE:0u)|
           (un*255u<=ud*250u?CERT_BLACK_VISIBLE:0u);
}
bool CertInterior(int2 q,uint unit,uint bit,uint3 qc,uint3 low,uint3 high,bool requireInside) {
    [unroll]for(int dy=-1;dy<=1;++dy)[unroll]for(int dx=-1;dx<=1;++dx)if(dx!=0||dy!=0) {
        int2 dir=int2(dx,dy),fp=q;bool foundF=false,foundK=false;
        // Stop on the physical black boundary before seeking BODY on its far side.
        [loop]for(int distance=1;distance<=3;++distance) {
            int2 p=q+distance*dir;
            if(!CertValid(p,bit))break;
            if(CertOwnBody(p,bit)){fp=p;foundF=true;break;}
            if(CertBlack(p,unit,bit))break;
        }
        if(!foundF)continue;
        // Keep the original first-LOW donor and ray stops; only validate that donor.
        if(requireInside){
            uint u=128u+32u*unit;
            int4 roi=int4(CBCMeta[u+24u],CBCMeta[u+25u],CBCMeta[u+26u],CBCMeta[u+27u]);
            if(any(fp<roi.xy)||any(fp>=roi.zw)||(CurrentInsideBits.Load(int3(fp,0))&bit)==0u)continue;
        }
        [loop]for(int distance=1;distance<=3;++distance) {
            int2 p=q-distance*dir;
            if(!CertValid(p,bit))break;
            if(CertOwnBody(p,bit))break;
            if(CertBlack(p,unit,bit)){foundK=true;break;}
        }
        if(foundK) {
            uint visibility=CertBlendVisibility(qc,CertCode(fp),low,high);
            if((visibility&CERT_ENDPOINT_VISIBLE)!=0u) {
                bool trustedForeground=(SelectedForegroundBits(SelectedBits.Load(int3(fp,0)))&bit)!=0u;
                bool adoptedKnownBlack=CBCMeta[PhysicalModelBank(unit)+2u]==1u&&HasAdoptedPhysicalRefinement(unit);
                if(trustedForeground||!adoptedKnownBlack||(visibility&CERT_BLACK_VISIBLE)!=0u)return true;
            }
        }
    }
    return false;
}
bool CertExterior(int2 q,uint unit,uint bit,uint3 qc,uint3 low,uint3 high) {
    [unroll]for(int dy=-1;dy<=1;++dy)[unroll]for(int dx=-1;dx<=1;++dx)if(dx!=0||dy!=0) {
        int2 dir=int2(dx,dy);
        if(!CertBlack(q-dir,unit,bit))continue;
        [loop]for(int distance=1;distance<=3;++distance) {
            int2 p=q+distance*dir;
            if(!CertValid(p,bit))break;
            // No crossing any accepted BODY or this unit's physical black rim.
            if(SelectedGeometryAt(p)!=0u||CertBlack(p,unit,bit))break;
            if(Code1M.Load(int3(p,0))==0u&&(CertBlendVisibility(qc,CertCode(p),low,high)&CERT_BLACK_VISIBLE)!=0u)return true;
        }
    }
    return false;
}
bool CertCode1Collar(int2 q,uint geometryCandidates) {
    if(Code1M.Load(int3(q,0))==0u)return false;
    bool interiorOnly=geometryCandidates!=0u;
    uint insideUnits=0u;
    if(interiorOnly){
        if(R(77u)!=0x31444e49u||R(78u)!=1u)return false;
        insideUnits=R(79u)&~R(80u)&255u;
        if((insideUnits&geometryCandidates)==0u)return false;
        // Current Inside already guarantees this pixel in the existing final OR.
        if((CurrentInsideBits.Load(int3(q,0))&insideUnits&CoverBits.Load(int3(q,0)))!=0u)return false;
    }
    float4 qv=CurrentRGB.Load(int3(q,0));
    if(qv.a!=1.0f||any(!isfinite(qv.rgb))||any(qv.rgb<0.0f)||any(qv.rgb>1.0f))return false;
    uint cover=CoverBits.Load(int3(q,0)),nearby=0u;
    // Existing BOX3 ownership requires a distinct current candidate.
    [loop]for(int dy=-3;dy<=3;++dy)[loop]for(int dx=-3;dx<=3;++dx) {
        if(dx==0&&dy==0)continue;
        int2 p=q+int2(dx,dy);
        if(any(p<0)||any(p>=int2(Shape.xy)))continue;
        float4 pv=CurrentRGB.Load(int3(p,0));
        if(pv.a!=1.0f||any(!isfinite(pv.rgb))||any(pv.rgb<0.0f)||any(pv.rgb>1.0f))continue;
        nearby|=SelectedGeometryAt(p)&CoverBits.Load(int3(p,0));
    }
    uint owners=cover&nearby;
    if(interiorOnly)owners&=geometryCandidates&insideUnits;
    if(owners==0u)return false;
    uint3 qc=CertCode(q);
    [loop]for(uint unit=0u;unit<8u;++unit) {
        uint bit=1u<<unit,u=(128u+32u*unit);
        if((owners&bit)==0u||CBCMeta[u]!=1u||CBCMeta[u+1u]!=1u||!CertValid(q,bit))continue;
        if(OpaqueBlackOnly(unit))continue;
        // ParentValid already validated every evaluated immutable model record.
        uint bank=PhysicalModelBank(unit);uint3 low=0u,high=1u;
        bool knownBlack=CBCMeta[bank+2u]==1u;
        if(interiorOnly&&(!knownBlack||!HasAdoptedPhysicalRefinement(unit)))continue;
        if(knownBlack){low=PhysicalUnpack(CBCMeta[bank+11u]);high=PhysicalUnpack(CBCMeta[bank+12u]);}
        // Demoted LOW can use only the existing interior proof, backed by current Inside.
        // Its physical-black/exterior shortcuts remain excluded. Non-LOW policy is unchanged.
        if((!interiorOnly&&CertBlack(q,unit,bit))||CertInterior(q,unit,bit,qc,low,high,interiorOnly)||
           (!interiorOnly&&CertExterior(q,unit,bit,qc,low,high)))return true;
    }
    return false;
}

// Final-only ownership141; geometry/packed/fallback remain LOW8 legacy. q is an existing legacy own-BODY candidate.
// For KNOWNt>=1, every valid own-ROI physical-black N8 neighbour is already
// in the immutable learned BAND (q seeds its Chebyshev envelope). Do not read
// neighbouring V3M here: publish_final_main overwrites that UAV in-place.
bool RetainLegacyOutlineAA(int2 q,uint unit,uint bit) {
    uint u=128u+32u*unit;
    int4 roi=int4(CBCMeta[u+24u],CBCMeta[u+25u],CBCMeta[u+26u],CBCMeta[u+27u]);
    [unroll]for(int dy=-1;dy<=1;++dy)[unroll]for(int dx=-1;dx<=1;++dx)if(dx!=0||dy!=0) {
        int2 p=q+int2(dx,dy);
        if(any(p<roi.xy)||any(p>=roi.zw))continue;
        if(CertBlack(p,unit,bit))return true;
    }
    return false;
}
bool FinalBodyAuthority(int2 q,uint selectedWord) {
    uint legacy=SelectedGeometryBits(selectedWord),foreground=SelectedForegroundBits(selectedWord);
    if((legacy|foreground)==0u)return false;
    [loop]for(uint unit=0u;unit<8u;++unit) {
        uint bit=1u<<unit;
        bool oldBody=(legacy&bit)!=0u,fg=(foreground&bit)!=0u;
        if(!oldBody&&!fg)continue;
        uint cu=U(unit),bank=PhysicalModelBank(unit);
        if(OpaqueBlackOnly(unit)) {
            if(oldBody)return true; // Keep the original independently admitted BODY only.
            continue;
        }
        // Unknown black OR unadopted CBC topology retains the old BODY fallback.
        // A known colour model alone does not prove an available glyph boundary.
        // Only adopted current physical refinement may demote exterior LOW8;
        // CBC5 HIGH8 carries trusted colour/closed-BODY evidence independently of t.
        if(CBCMeta[bank+2u]!=1u||!HasAdoptedPhysicalRefinement(unit)) {
            if(oldBody)return true;
            continue;
        }
        if(fg)return true;
        if(CurrentWidthAvailable(unit)&&oldBody&&RetainLegacyOutlineAA(q,unit,bit))return true;
    }
    return false;
}


// Typed RIBBON is a current UI policy, not a failed glyph/black model.
// Keep its strip out of LOW/BODY/AA/closure/cache evidence. Only final M is ORed.
bool CurrentRibbonUiAt(uint2 p) {
    uint words,stride;CurrentOCR.GetDimensions(words,stride);
    if(words!=256u||stride!=4u||CurrentOCR[0u]!=4u||CurrentOCR[1u]!=0x3852434fu||CurrentOCR[2u]!=1u||
       CurrentOCR[3u]!=CurrentOCR[4u]||CurrentOCR[4u]>8u)return false;
    if(any(uint4(CurrentOCR[5u],CurrentOCR[6u],CurrentOCR[7u],CurrentOCR[8u])!=CFrame)||
       any(uint2(CurrentOCR[9u],CurrentOCR[10u])!=CSource.xy)||
       any(uint4(CurrentOCR[11u],CurrentOCR[12u],CurrentOCR[13u],CurrentOCR[14u])!=CField)||
       any(CContent.xy>=CContent.zw)||any(CContent.zw>CField.xy))return false;
    // Exact existing native-center/half-open projection. Never use expanded cover.y.
    uint2 scaled=uint2(2u*CContent.x*CSource.x+(2u*p.x+1u)*(CContent.z-CContent.x),
        2u*CContent.y*CSource.y+(2u*(p.y+CSource.z)+1u)*(CContent.w-CContent.y));
    [loop]for(uint unit=0u;unit<CurrentOCR[4u];++unit) {
        uint4 core;if(!CurrentCanonicalRibbonUnit(unit,core))continue;
        if(scaled.x>=2u*CContent.x*CSource.x&&scaled.x<2u*CContent.z*CSource.x&&
           scaled.y>=2u*core.y*CSource.y&&scaled.y<2u*CContent.w*CSource.y)return true;
    }
    return false;
}

uint ResolvePublishedGlyphProtection(bool publicationInputsValid,
                                    bool verifiedBodySupport,
                                    uint packedProtectionEvidence,
                                    bool residualCollarCertificate,
                                    uint literalRawFallback) {
    return publicationInputsValid?
        ((verifiedBodySupport||packedProtectionEvidence!=0u||residualCollarCertificate)?1u:0u):
        (literalRawFallback!=0u?1u:0u);
}

[numthreads(16,16,1)]
void publish_final_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||!Native(d.xy))return;
    uint insideW,insideH;CurrentInsideBits.GetDimensions(insideW,insideH);
    bool insideShapeValid=all(uint2(insideW,insideH)==Shape.xy);
    if(!insideShapeValid&&DataShape())Fault(F_SHAPE);
    uint literalRawFallback=RawM.Load(int3(d.xy,0));bool dataShapeValid=DataShape();
    bool publicationInputsValid=dataShapeValid&&RawMeta[2u]==1u&&RawMeta[3u]==0u&&R(3u)==0u;
    uint privatePackedWord=publicationInputsValid?V3M[d.xy]:0u;
    uint packedProtectionEvidence=privatePackedWord&~PRIVATE_PUBLISH_STRIP_MASK;
    bool opaqueCurrentBlack=publicationInputsValid&&(privatePackedWord&PRIVATE_OPAQUE_BLACK_OFFER)!=0u;
    bool retainedAA=publicationInputsValid&&(privatePackedWord&PRIVATE_AA_HISTORY_OFFER)!=0u&&StabilityAAHistoryPlan();
    bool privateCachedBlack=publicationInputsValid&&(
        ((privatePackedWord&PRIVATE_CACHED_BLACK_OFFER)!=0u&&StabilityOfferAcceptedRW())||
        ((privatePackedWord&PRIVATE_LOCAL_CONFLICT_BLACK_OFFER)!=0u&&StabilityLocalOfferAcceptedRW()));
    uint hasMeasuredBlackBand=(packedProtectionEvidence&BAND_MASK)!=0u?1u:0u;
    uint selectedWord=SelectedBits.Load(int3(d.xy,0));
    bool verifiedBodySupport=false;
    // Invalid publication must take literal RawM without any model/Data read.
    [branch]if(publicationInputsValid)verifiedBodySupport=FinalBodyAuthority(int2(d.xy),selectedWord);
    uint selectedGeometryCandidates=SelectedGeometryBits(selectedWord);
    bool residualCollarCertificate=false;
    // Preserve the non-LOW path; a demoted LOW may recheck its validated inside witness.
    [branch]if(publicationInputsValid&&(selectedGeometryCandidates==0u||!verifiedBodySupport)&&packedProtectionEvidence==0u&&Code1M.Load(int3(d.xy,0))!=0u) {
        residualCollarCertificate=CertCode1Collar(int2(d.xy),selectedGeometryCandidates);
    }
    uint legacyReferenceSupport=(literalRawFallback!=0u||hasMeasuredBlackBand!=0u||(packedProtectionEvidence&OLD_AA)!=0u)?1u:0u;
    // Valid publication combines admitted BODY, packed protection and the
    // residual candidate-collar certificate. Packed UNKNOWN_RAW also preserves
    // nonKNOWN-unit overlap. Invalid publication uses literal RawM alone.
    uint publishedProtection=ResolvePublishedGlyphProtection(publicationInputsValid,verifiedBodySupport,
        packedProtectionEvidence,residualCollarCertificate,literalRawFallback);
    bool currentRibbonUi=false;
    [branch]if(publicationInputsValid)currentRibbonUi=CurrentRibbonUiAt(d.xy);
    uint originalPublishedProtection=publishedProtection;
    bool directCurrentAA=publicationInputsValid&&(packedProtectionEvidence&(OLD_AA|SUPPORT_AA))!=0u;
    bool independentCurrent=verifiedBodySupport||(packedProtectionEvidence&~(OLD_AA|SUPPORT_AA))!=0u||
        residualCollarCertificate||privateCachedBlack||retainedAA;
    bool actualNonUi=publicationInputsValid&&independentCurrent;
    // Preserve the existing AA exemption's cleanup semantics. Its remembered
    // actual protection additionally needs current finalize-ready identity.
    bool normalizedInside=publicationInputsValid&&insideShapeValid&&
        (CurrentInsideBits.Load(int3(d.xy,0))&(R(79u)&~R(80u)&255u)&CoverBits.Load(int3(d.xy,0)))!=0u;
    if(normalizedInside)independentCurrent=true;
    if(normalizedInside&&R(77u)==0x31444e49u&&R(78u)==1u&&InputReady())actualNonUi=true;
    bool admittedCurrentAA=directCurrentAA;
    [branch]if(publicationInputsValid)admittedCurrentAA=WeakAAAdmitAndAdvance(d.xy,directCurrentAA,independentCurrent,retainedAA,actualNonUi);
    if(directCurrentAA&&!independentCurrent&&!admittedCurrentAA)publishedProtection=0u;
    if(privateCachedBlack)publishedProtection=1u;
    if(retainedAA)publishedProtection=1u;
    if(StabilityHeaderRWShape()&&privateCachedBlack){uint ignored;
        InterlockedAdd(StabilityHeaderWrite[uint2(100u,0u)],1u,ignored);
        if(originalPublishedProtection==0u)InterlockedAdd(StabilityHeaderWrite[uint2(101u,0u)],1u,ignored);
    }
    if(currentRibbonUi)publishedProtection=1u;
    // Snapshot-write eligibility is an immutable plan set by the completed validation dispatch.
    // Only snapshotFault changes here; it never controls another thread's write predicate.
    if(StabilitySnapshotWritePlan()){
        float4 v=CurrentRGB.Load(int3(d.xy,0));
        if(StabilityValidRGB(v)){
            StabilityImageWrite[d.xy]=StabilityPackRGB(RGB8(int2(d.xy)))|((SelectedGeometryAt(int2(d.xy))&255u)<<24u);
            AAHistory[d.xy]=(packedProtectionEvidence!=0u?1u:0u)|
                ((admittedCurrentAA||retainedAA)?2u:0u);
        }else {StabilityImageWrite[d.xy]=0u;AAHistory[d.xy]=0u;uint ignored;InterlockedOr(StabilityHeaderWrite[uint2(3u,0u)],1u,ignored);}
        // One analytical coverage count per16x16 group; all native lanes above unconditionally write RGB or zero+fault.
        // The descriptor/current-frame/commit plan is uniform and immutable; count/fault never control writes.
        if(all((d.xy&15u)==0u)){
            uint covered=min(16u,Shape.x-d.x)*min(16u,Shape.y-d.y),written;
            InterlockedAdd(StabilityHeaderWrite[uint2(147u,0u)],covered,written);
        }
    }
    // Keep the current opaque Raw offer out of AA decisions, observer exemptions
    // and snapshot/history construction above. It joins positive protection only now.
    if(opaqueCurrentBlack)publishedProtection=1u;
    V3M[d.xy]=publishedProtection;
    if(publicationInputsValid){uint ignored;
        if((packedProtectionEvidence&OLD_AA)!=0u)Data.InterlockedAdd(67u*4u,1u,ignored);
        if((packedProtectionEvidence&SUPPORT_AA)!=0u)Data.InterlockedAdd(68u*4u,1u,ignored);
        if((packedProtectionEvidence&CLOSURE)!=0u)Data.InterlockedAdd(69u*4u,1u,ignored);
        if((packedProtectionEvidence&UNKNOWN_RAW)!=0u)Data.InterlockedAdd(72u*4u,1u,ignored);
        if((packedProtectionEvidence&MEASURED_RIM)!=0u)Data.InterlockedAdd(84u*4u,1u,ignored); // Unique valid pre-Inside pixels.
        if(literalRawFallback!=0u&&publishedProtection==0u)Data.InterlockedAdd(73u*4u,1u,ignored);
        if((packedProtectionEvidence&CERTIFIED_COLLAR)!=0u||residualCollarCertificate)Data.InterlockedAdd(75u*4u,1u,ignored);
        if(Code1M.Load(int3(d.xy,0))!=0u&&publishedProtection==0u)Data.InterlockedAdd(76u*4u,1u,ignored);
        if(publishedProtection!=0u&&legacyReferenceSupport==0u)Data.InterlockedAdd(70u*4u,1u,ignored);
    }
    if(dataShapeValid&&publicationInputsValid&&literalRawFallback==0u&&publishedProtection!=0u){uint ignored;Data.InterlockedAdd(28u,1u,ignored);}
    if(dataShapeValid&&all(d.xy==0u)) {
        W(2u,publicationInputsValid?1u:0u);
        [unroll]for(uint k=0u;k<16u;++k)W(32u+k,RawMeta[k]);
        if(!publicationInputsValid)W(7u,0u);
    }
}

bool TrueCoverEdge(int2 p) {
    uint ub=128u+32u*Hook.x;
    int4 r=int4(CBCMeta[ub+12u],CBCMeta[ub+13u],CBCMeta[ub+14u],CBCMeta[ub+15u]);
    return p.x==r.x||p.y==r.y||p.x==r.z-1||p.y==r.w-1;
}
bool NativeEdge(int2 p) { return p.x==0||p.y==0||p.x==(int)Shape.x-1||p.y==(int)Shape.y-1; }
bool CacheRootValid(uint root,bool enclosed,out uint2 pixel,out uint rootTag) {
    pixel=0u;rootTag=0u;
    if(root==0u||root>Shape.z||root>=0x40000000u){Fault(F_CACHE);return false;}
    pixel=uint2((root-1u)%Shape.x,(root-1u)/Shape.x);
    if(!OwnCover(int2(pixel))){Fault(F_CACHE);return false;}
    rootTag=PassableCache[pixel];
    if((rootTag&ROOT_MASK)!=root||((rootTag&ENCLOSED)!=0u)!=enclosed){Fault(F_CACHE);return false;}
    return true;
}
// AA background classes mirror ProfileBackground: exterior or closed-unseeded.
// InnerEligible is the current first-passable-graph seeded-interior bit.
// Root proof checks only immutable identity/enclosure bits, never BAD_RIM.
bool CachedAABackground(int2 p,bool knownBlack) {
    if(!OwnCover(p))return false;
    uint tag=PassableCache[(uint2)p],root=tag&ROOT_MASK;
    if(root==0u)return false;
    bool enclosed=(tag&ENCLOSED)!=0u;
    [branch]if(enclosed) {
        // UNKNOWN black keeps the exact old exterior-only region policy.
        if(!knownBlack)return false;
        if(InnerEligible.Load(int3(p,0))!=0u)return false;
    }
    uint2 rp;uint rt;return CacheRootValid(root,enclosed,rp,rt);
}
bool OwnBand(int2 p) { return OwnCover(p)&&(V3M[(uint2)p]&(1u<<Hook.x))!=0u; }
// Exact geometric append membership, NOT the stricter AA/source predicate.
// Parent/RootFlags name only the SECOND own-unit physical-black graph here.
bool RawAppendMember(int2 p) {
    if(!OwnCover(p)||Passable.Load(int3(p,0))!=1u)return false;
    uint root=FindStable((uint)p.y*Shape.x+(uint)p.x+1u);
    return root!=0u&&(RootFlags[root]&3u)==1u;
}
bool RawSelected(int2 p) {
    // Keep every original AA validity/black test, then bound its donor. Never
    // consult mutable AA/closure/UNKNOWN bits or future units' KNOWN status.
    if(!ValidPoint(p)||!Black(p)||(!OwnBand(p)&&Code1M.Load(int3(p,0))==0u))return false;
    return RawAppendMember(p);
}
// Called only for a validated current own-unit Raw member. The existing known
// style width bounds the search; no BODY-seeded interior can supply background.
// Black traversal cannot jump to a different physical component, including a
// diagonal neighbour. First nonblack ends that ray whether it is usable or not.
bool OpaqueRawHasBackgroundWitness(int2 p) {
    if(OpaqueBlackOnly(Hook.x)||!CurrentWidthMayReplaceRaw(Hook.x)||
       BlackModelWord(PhysicalModelBank(Hook.x)+2u)!=1u||!HasAdoptedPhysicalRefinement(Hook.x))return false;
    if(!ValidPoint(p)||!Black(p))return false;
    if(!RawAppendMember(p))return false;
    uint extent=R(U(Hook.x)+11u); // CurrentWidthMayReplaceRaw proved KNOWN1..16.
    uint root=FindStable((uint)p.y*Shape.x+(uint)p.x+1u);
    if(root==0u)return false;
    const int2 directions[8]={int2(-1,-1),int2(0,-1),int2(1,-1),int2(-1,0),
                             int2(1,0),int2(-1,1),int2(0,1),int2(1,1)};
    [unroll]for(uint ray=0u;ray<8u;++ray) {
        [loop]for(uint distance=1u;distance<=extent;++distance) {
            int2 q=p+(int)distance*directions[ray];
            if(!ValidPoint(q)||GlobalBody.Load(int3(q,0))!=0u)break;
            if(!Black(q)) {
                if(CachedAABackground(q,true))return true;
                break;
            }
            if(Passable.Load(int3(q,0))!=1u)break;
            if(FindStable((uint)q.y*Shape.x+(uint)q.x+1u)!=root)break;
        }
    }
    return false;
}
// One bit in dead mode-final scratch, indexed by the ACTUAL current Raw root.
// This is private opaque permission, never a mathematical RootFlags bit, BAND,
// AA donor, closure certificate or retained/history construction evidence.
static const uint RAW_COMPONENT_BACKGROUND_WITNESS=1u;
// Separate private first-source-root reduction; index cells may alias Raw roots.
static const uint SOURCE_ROOT_ENVELOPE_WITNESS=2u;
static const uint CURRENT_ROOT_WITNESS_ALLOWED_MASK=RAW_COMPONENT_BACKGROUND_WITNESS|SOURCE_ROOT_ENVELOPE_WITNESS;
bool RawComponentWitnessShapeKnown() {
    uint w,h;RawComponentWitness.GetDimensions(w,h);
    return all(uint2(w,h)==Shape.xy);
}
bool OpaqueRawRootPixel(int2 p,out uint2 rootPixel) {
    rootPixel=0u;
    if(!RawAppendMember(p))return false;
    uint root=FindStable((uint)p.y*Shape.x+(uint)p.x+1u);
    if(root==0u)return false;
    rootPixel=uint2((root-1u)%Shape.x,(root-1u)/Shape.x);
    if(!OwnCover(int2(rootPixel))){Fault(F_GRAPH);return false;}
    return true;
}
bool OpaqueRawComponentHasBackgroundWitness(int2 p) {
    if(OpaqueBlackOnly(Hook.x)||!CurrentWidthMayReplaceRaw(Hook.x)||
       BlackModelWord(PhysicalModelBank(Hook.x)+2u)!=1u||!HasAdoptedPhysicalRefinement(Hook.x))return false;
    if(!ValidPoint(p)||!Black(p))return false;
    uint2 rootPixel;if(!OpaqueRawRootPixel(p,rootPixel))return false;
    uint witnesses=RawComponentWitness[rootPixel];
    return (witnesses&~CURRENT_ROOT_WITNESS_ALLOWED_MASK)==0u&&
           (witnesses&RAW_COMPONENT_BACKGROUND_WITNESS)!=0u;
}
// Grow has completed the current unit's LOW8 BAND before aggregation. All
// aggregate writes are atomic ORs to OTHER bits; never write a BAND bit here.
// In particular MEASURED_RIM/AA/closure cannot seed this one-hop certificate.
bool AdjacentMeasuredRawRim(int2 p,bool band) {
    if(band||!ValidPoint(p)||!Black(p)||!HasAdoptedPhysicalRefinement(Hook.x)||
       BlackModelWord(PhysicalModelBank(Hook.x)+2u)!=1u||!RawAppendMember(p))return false;
    [unroll]for(int dy=-1;dy<=1;++dy)[unroll]for(int dx=-1;dx<=1;++dx)if(dx!=0||dy!=0) {
        if(OwnBand(p+int2(dx,dy)))return true;
    }
    return false;
}
bool Support(int2 p) { return OwnCover(p)&&(OwnBand(p)||RawSelected(p)); }
bool IntervalVisible(uint3 q,uint3 r) {
    // Half-code rounding cells, with clipped endpoints closed. No unsigned -1 at c0.
    uint3 qlo=2u*q-(uint3)(q>0u),qhi=min(uint3(510u,510u,510u),2u*q+1u);
    uint3 rlo=2u*r-(uint3)(r>0u),rhi=min(uint3(510u,510u,510u),2u*r+1u);
    uint ln=0u,ld=1u,un=1u,ud=1u;
    [unroll]for(uint c=0u;c<3u;++c) {
        if(qlo[c]*ld>ln*rhi[c]){ln=qlo[c];ld=rhi[c];}
        if(rlo[c]>0u&&qhi[c]*ud<un*rlo[c]){un=qhi[c];ud=rlo[c];}
    }
    // Every product <=260100; uint32 is exact. All feasible alpha must be >=5/255.
    return ln*ud<=un*ld&&un*255u<=ud*250u;
}
// Signed A*s >= B constraint; s is the background fraction 1-alpha.
// All endpoints are doubled RGB8 cells. Nonnegative rational cross-products
// are <=510*510; negative numerators are handled before conversion to uint.
bool ConstrainBackgroundFraction(int coefficient,int rhs,inout uint ln,inout uint ld,inout uint un,inout uint ud) {
    return StrokeConstrainBackgroundFraction(coefficient,rhs,ln,ld,un,ud);
}
bool KnownBlackBoxIntervalVisible(uint3 q,uint3 r,uint3 blackLow,uint3 blackHigh) {
    uint3 qlo=2u*q-(uint3)(q>0u),qhi=min(uint3(510u,510u,510u),2u*q+1u);
    uint3 rlo=2u*r-(uint3)(r>0u),rhi=min(uint3(510u,510u,510u),2u*r+1u);
    uint3 blo=2u*blackLow-(uint3)(blackLow>0u),bhi=min(uint3(510u,510u,510u),2u*blackHigh+1u);
    return StrokeRgbIntervalVisibleCells(qlo,qhi,blo,bhi,(int3)rlo,(int3)rhi);
}
// Portable exact SM5 32x32->64 multiply, identical arithmetic to the
// existing CurrentWideProduct. FXC/SM5 has no umulExtended intrinsic.
void AAWideProduct(uint a,uint b,out uint low,out uint high) {
    uint p0=(a&65535u)*(b&65535u);
    uint t=(a>>16u)*(b&65535u)+(p0>>16u);
    uint cross=(a&65535u)*(b>>16u)+(t&65535u);
    low=(cross<<16u)|(p0&65535u);
    high=(a>>16u)*(b>>16u)+(t>>16u)+(cross>>16u);
}
bool AAProductsLE(uint a,uint b,uint c,uint d) {
    uint al,ah,bl,bh;AAWideProduct(a,b,al,ah);AAWideProduct(c,d,bl,bh);
    return ah<bh||(ah==bh&&al<=bl);
}
bool ConstrainLumaFraction(int coefficient,int rhs,inout uint ln,inout uint ld,inout uint un,inout uint ud) {
    if(coefficient>0) {
        if(rhs>0) {
            uint n=(uint)rhs,d=(uint)coefficient;
            if(!AAProductsLE(n,ld,ln,d)){ln=n;ld=d;}
        }
    } else if(coefficient<0) {
        if(rhs>0)return false;
        uint n=(uint)(-rhs),d=(uint)(-coefficient);
        if(!AAProductsLE(un,d,n,ud)){un=n;ud=d;}
    } else if(rhs>0)return false;
    return AAProductsLE(ln,ud,un,ld);
}
uint3 AAHalfLow(uint3 v) { return StrokeHalfLow(v); }
uint3 AAHalfHigh(uint3 v) { return StrokeHalfHigh(v); }
uint AALuma709(uint3 v) { return v.r*2126u+v.g*7152u+v.b*722u; }
bool KnownBlackLumaVisible(uint3 q,uint3 r,uint3 blackLow,uint3 blackHigh) {
    // Exact canonical-RGB-derived BT709 bounds, NOT recovered decoder planes.
    // Doubled endpoints <=5100000; products <=26010000000000 require45bits.
    uint qlo=AALuma709(AAHalfLow(q)),qhi=AALuma709(AAHalfHigh(q));
    uint rlo=AALuma709(AAHalfLow(r)),rhi=AALuma709(AAHalfHigh(r));
    uint blo=AALuma709(AAHalfLow(blackLow)),bhi=AALuma709(AAHalfHigh(blackHigh));
    uint ln=0u,ld=1u,un=1u,ud=1u;
    if(!ConstrainLumaFraction((int)rhi-(int)bhi,(int)qlo-(int)bhi,ln,ld,un,ud))return false;
    if(!ConstrainLumaFraction((int)blo-(int)rlo,(int)blo-(int)qhi,ln,ld,un,ud))return false;
    // Preserve ALL-feasible-alpha>=5/255; max numerator*255 <=1300500000.
    return AAProductsLE(ln,ud,un,ld)&&un*255u<=ud*250u;
}
bool KnownChromaSubtractionVisible(uint3 q,uint3 r,uint3 blackLow,uint3 blackHigh,uint3 prototype) {
    // q=clip(r-D*1+e,0,255), D>=0, e in learnedBlackBox-prototype.
    // One common achromatic subtraction; per-channel residual is bounded by
    // THIS caption's observed black spread, never a tuned chroma tolerance.
    int3 qlo=(int3)AAHalfLow(q),qhi=(int3)AAHalfHigh(q);
    int3 rlo=(int3)AAHalfLow(r),rhi=(int3)AAHalfHigh(r);
    int3 elo=(int3)AAHalfLow(blackLow)-2*(int3)prototype;
    int3 ehi=(int3)AAHalfHigh(blackHigh)-2*(int3)prototype;
    // R+e <=1020 doubled codes; larger D gives only all-zero clipped RGB.
    int lower=0,upper=1020;
    [unroll]for(uint c=0u;c<3u;++c) {
        if(qhi[c]<510)lower=max(lower,rlo[c]+elo[c]-qhi[c]);
        if(qlo[c]>0)upper=min(upper,rhi[c]+ehi[c]-qlo[c]);
        if(lower>upper)return false;
    }
    if(!KnownBlackLumaVisible(q,r,blackLow,blackHigh))return false;
    return true;
}
// EXTERNAL P10 whole-root withdrawal hypothesis. Exact original grow envelope:
// SeedAt's3x3 LOW|Inner support plus known(t-1) grow equals distance<=t.
// BODY/HIGH can preserve their own recipient, but cannot manufacture this root bit.
bool ExactCurrentOriginalEnvelope(int2 q,uint extent) {
    if(!OwnCover(q))return false;
    if(SeedAt(q))return true;
    uint bit=1u<<Hook.x;
    [loop]for(int dy=-(int)extent;dy<=(int)extent;++dy) {
        [loop]for(int dx=-(int)extent;dx<=(int)extent;++dx) {
            if(abs(dx)<=1&&abs(dy)<=1)continue;
            int2 p=q+int2(dx,dy);
            if(!OwnCover(p))continue;
            if(InnerEligible.Load(int3(p,0))==1u)return true;
            if((SelectedGeometryAt(p)&bit)!=0u)return true;
        }
    }
    return false;
}
// CALL ONLY from the ordered closure dispatch after this current unit's entire
// aggregate reduction. Absence of bit2 is not authority during its accumulation.
bool CurrentKnownClosedRecipientAllowed(int2 q) {
    if(!CurrentWidthAvailable(Hook.x))return true;
    uint ru=128u+32u*Hook.x;
    // Without an issued current Raw graph, no completed absence is available.
    // Preserve the old policy rather than borrow stale scratch from another unit.
    if(RawMeta[ru]!=1u||RawMeta[ru+1u]!=1u)return true;
    if(!OwnCover(q))return false;
    uint bit=1u<<Hook.x,selected=SelectedBits.Load(int3(q,0));
    if(GlobalBody.Load(int3(q,0))!=0u)return true;
    if((selected&(bit|(bit<<8u)))!=0u)return true;
    if(InnerEligible.Load(int3(q,0))==1u)return true;
    uint tag=PassableCache[(uint2)q];
    if((tag&ENCLOSED)==0u)return true;
    uint root=tag&ROOT_MASK;
    if(root==0u)return false;
    uint2 rootPixel;uint rootTag;
    if(!CacheRootValid(root,true,rootPixel,rootTag))return false;
    uint witnesses=RawComponentWitness[rootPixel];
    return (witnesses&~CURRENT_ROOT_WITNESS_ALLOWED_MASK)==0u&&
           (witnesses&SOURCE_ROOT_ENVELOPE_WITNESS)!=0u;
}

// EXTERNAL P10 B4: preserve a PREVIOUSLY possible AA offer when its original
// physical-black donor independently reaches current EXTERIOR background.
// Same bounded pointwise C2 eligibility, traversal and first-nonblack stop.
// A closed unseeded recipient cannot serve as this exterior witness.
bool OpaqueRawHasExteriorBackgroundWitness(int2 p) {
    if(OpaqueBlackOnly(Hook.x)||!CurrentWidthMayReplaceRaw(Hook.x)||
       BlackModelWord(PhysicalModelBank(Hook.x)+2u)!=1u||!HasAdoptedPhysicalRefinement(Hook.x))return false;
    if(!ValidPoint(p)||!Black(p))return false;
    if(!RawAppendMember(p))return false;
    uint extent=R(U(Hook.x)+11u);
    uint root=FindStable((uint)p.y*Shape.x+(uint)p.x+1u);
    if(root==0u)return false;
    const int2 directions[8]={int2(-1,-1),int2(0,-1),int2(1,-1),int2(-1,0),
                             int2(1,0),int2(-1,1),int2(0,1),int2(1,1)};
    [loop]for(uint ray=0u;ray<8u;++ray) {
        [loop]for(uint distance=1u;distance<=extent;++distance) {
            int2 q=p+(int)distance*directions[ray];
            if(!ValidPoint(q)||GlobalBody.Load(int3(q,0))!=0u)break;
            if(!Black(q)) {
                if(CachedAABackground(q,true)) {
                    if((PassableCache[(uint2)q]&ENCLOSED)==0u)return true;
                }
                break;
            }
            if(Passable.Load(int3(q,0))!=1u)break;
            if(FindStable((uint)q.y*Shape.x+(uint)q.x+1u)!=root)break;
        }
    }
    return false;
}
// Only AA recipients use this exception. Closure still uses the strict whole
// source-root guard. No newly offered M/AA or component witness qualifies p.
bool CurrentKnownClosedAARecipientAllowed(int2 q,int2 donor) {
    if(CurrentKnownClosedRecipientAllowed(q))return true;
    return OpaqueRawHasExteriorBackgroundWitness(donor);
}

// Current-only geometric gap fill, separate from the original unrolled colour rays.
// Four forward directions visit each unordered black bracket once; newly added AA
// never qualifies as a BAND or current RawSelected donor.
void CurrentBlackGapRays(int2 p,uint cu) {
    const int2 directions[4]={int2(1,0),int2(-1,1),int2(0,1),int2(1,1)};
    [loop]for(uint k=0u;k<4u;++k) {
        int2 dir=directions[k],q=p+dir,r=p+2*dir;
        if(!ValidPoint(q)||!ValidPoint(r)||!OwnBand(r))continue;
        if(!RawSelected(r))continue;
        uint3 qc=RGB8(q);
        if(AnyPhysicalBlack(q,qc)||GlobalBody.Load(int3(q,0))!=0u)continue;
        if(!CurrentKnownClosedAARecipientAllowed(q,p))continue;
        // New geometric candidate/acceptance, not an old colour-interval success.
        uint ignored;Data.InterlockedAdd((cu+30u)*4u,1u,ignored);
        InterlockedOr(V3M[(uint2)q],SUPPORT_AA|OLD_AA,ignored);
        Data.InterlockedAdd((cu+31u)*4u,1u,ignored);
        Data.InterlockedAdd((cu+32u)*4u,1u,ignored);
    }
}
// Separate one-pixel gap experiment. Caller proves the EXACT pointwise C2
// background witness and p outside measured BAND. Component-wide opaque offers
// never grant this permission. The opposite endpoint must be original measured
// BAND and RawSelected in the SAME immutable current own-unit Raw4CC.
void ObservedRawToMeasuredBandGapRays(int2 p,uint cu) {
    uint root=FindStable((uint)p.y*Shape.x+(uint)p.x+1u);
    if(root==0u)return;
    [loop]for(int dy=-1;dy<=1;++dy)[loop]for(int dx=-1;dx<=1;++dx)if(dx!=0||dy!=0) {
        int2 dir=int2(dx,dy),q=p+dir,r=p+2*dir;
        if(!ValidPoint(q)||!ValidPoint(r))continue;
        if(!OwnBand(r))continue;
        if(!RawSelected(r))continue;
        // RawSelected proved r is an accepted current Raw member before this
        // side-effecting graph query. No HLSL short-circuit assumption is used.
        uint oppositeRoot=FindStable((uint)r.y*Shape.x+(uint)r.x+1u);
        if(oppositeRoot==0u||oppositeRoot!=root)continue;
        uint3 qc=RGB8(q);
        if(AnyPhysicalBlack(q,qc)||GlobalBody.Load(int3(q,0))!=0u)continue;
        if(!CachedAABackground(q,true))continue;
        if(!CurrentKnownClosedAARecipientAllowed(q,p))continue;
        // Ordinary AA only: normal current observer/history admission still runs.
        // No OLD_AA bit, independent-foreground exemption or recursive donor.
        uint ignored;Data.InterlockedAdd((cu+30u)*4u,1u,ignored);
        InterlockedOr(V3M[(uint2)q],SUPPORT_AA,ignored);
        Data.InterlockedAdd((cu+31u)*4u,1u,ignored);
        Data.InterlockedAdd((cu+33u)*4u,1u,ignored);
    }
}
// PRIVATE NEW EXPERIMENT: the short-gap cap3 and KNOWN-width-minus-one
// coupling are explicit candidate policy, not derived from a canonical radius.
// Keep the existing one-pixel helper and its behavior independently unchanged.
static const uint POINTWISE_RAW_SHORT_GAP_MAX_NONBLACK_PIXELS=3u;
bool ShortGapCurrentBackgroundPoint(int2 q,int2 donor,out uint backgroundTag) {
    backgroundTag=0u;
    if(!ValidPoint(q))return false;
    uint3 qc=RGB8(q);
    if(AnyPhysicalBlack(q,qc)||GlobalBody.Load(int3(q,0))!=0u)return false;
    if(!CachedAABackground(q,true))return false;
    if(!CurrentKnownClosedAARecipientAllowed(q,donor))return false;
    // Root identity/enclosure are immutable here; mutable BAD_RIM is excluded.
    backgroundTag=PassableCache[(uint2)q]&(ROOT_MASK|ENCLOSED);
    return true;
}
void ObservedRawToMeasuredBandShortGapRays(int2 p,uint cu) {
    // Caller proves ONLY p's exact pointwise C2 witness and p outside BAND.
    // Component-wide opaque offers and existing AA/M never supply a donor.
    if(!CurrentWidthMayReplaceRaw(Hook.x))return;
    uint thickness=R(U(Hook.x)+11u);
    if(thickness<=2u)return;
    uint maximumGap=min(POINTWISE_RAW_SHORT_GAP_MAX_NONBLACK_PIXELS,thickness-1u);
    uint root=FindStable((uint)p.y*Shape.x+(uint)p.x+1u);
    if(root==0u)return;
    [loop]for(int dy=-1;dy<=1;++dy)[loop]for(int dx=-1;dx<=1;++dx)if(dx!=0||dy!=0) {
        int2 dir=int2(dx,dy);
        [loop]for(uint gap=2u;gap<=maximumGap;++gap) {
            int2 r=p+(int)(gap+1u)*dir;
            if(!ValidPoint(r))continue;
            if(!OwnBand(r))continue;
            if(!RawSelected(r))continue;
            uint oppositeRoot=FindStable((uint)r.y*Shape.x+(uint)r.x+1u);
            if(oppositeRoot==0u||oppositeRoot!=root)continue;
            bool allCurrent=true;uint sharedBackgroundTag=0u;
            [loop]for(uint step=1u;step<=gap;++step) {
                int2 q=p+(int)step*dir;uint backgroundTag;
                // A separate call/branch avoids assuming HLSL short-circuits
                // side-effecting cache checks. Validate ALL points before ORs.
                if(!ShortGapCurrentBackgroundPoint(q,p,backgroundTag)){allCurrent=false;break;}
                if(step==1u)sharedBackgroundTag=backgroundTag;
                else if(backgroundTag!=sharedBackgroundTag){allCurrent=false;break;}
            }
            if(!allCurrent)continue;
            uint ignored;Data.InterlockedAdd((cu+30u)*4u,gap,ignored);
            [loop]for(uint writeStep=1u;writeStep<=gap;++writeStep) {
                int2 q=p+(int)writeStep*dir;
                InterlockedOr(V3M[(uint2)q],SUPPORT_AA,ignored);
            }
            Data.InterlockedAdd((cu+31u)*4u,gap,ignored);
            Data.InterlockedAdd((cu+33u)*4u,gap,ignored);
        }
    }
}
void OuterRays(int2 p,bool physicalBand,bool rawSelectedDonor) {
    uint cu=U(Hook.x);uint3 pc=RGB8(p);
    // Current unit model is already immutable/validated by CBCunit and Rawbegin.
    // Load its endpoint bounds ONCE per seed, never once per ray or pixel mode.
    uint bank=PhysicalModelBank(Hook.x);bool knownBlack=BlackModelWord(bank+2u)==1u;
    uint3 blackLow=0u,blackHigh=0u,blackPrototype=0u;
    if(knownBlack){blackLow=PhysicalUnpack(BlackModelWord(bank+11u));blackHigh=PhysicalUnpack(BlackModelWord(bank+12u));blackPrototype=PhysicalUnpack(BlackModelWord(bank+5u));}
    [branch]if(knownBlack&&physicalBand&&rawSelectedDonor&&HasAdoptedPhysicalRefinement(Hook.x))
        CurrentBlackGapRays(p,cu);
    [loop]for(int dy=-1;dy<=1;++dy)[loop]for(int dx=-1;dx<=1;++dx)if(dx!=0||dy!=0) {
        int2 dir=int2(dx,dy),q=p+dir,r=p+2*dir;
        if(!ValidPoint(q)||!ValidPoint(r)||!CachedAABackground(q,knownBlack)||!CachedAABackground(r,knownBlack))continue;
        uint3 qc=RGB8(q),rc=RGB8(r);
        if(AnyPhysicalBlack(q,qc)||AnyPhysicalBlack(r,rc))continue;
        if(GlobalBody.Load(int3(q,0))!=0u||GlobalBody.Load(int3(r,0))!=0u)continue;
        if(!CurrentKnownClosedAARecipientAllowed(q,p))continue;
        // Global exclusion: every raw/band addition is physical black under
        // one potentially eligible current unit, even a future interleaved unit.
        // Code1M plus globalAny exclude pre-AA/pre-closure global V3 only;
        // new AA/closure pixels never recursively seed this one-hop operation.
        // Retain every old AA acceptance. Only its rejected rays may consult
        // the current learned black endpoint box; UNKNOWN stays exact old.
        // CFB5: inherited collar is a candidate; it no longer bypasses AA validation.
        int3 delta=abs(int3(rc)-int3(pc));
        [branch]if(max(delta.r,max(delta.g,delta.b))<21) {
            // UNKNOWN black preserves the original unconditional gap rejection.
            if(!knownBlack)continue;
            // A bounded current enclosed-black donor and existing q candidate
            // may reach the unchanged RGB interval tests despite the low gap.
            if(!physicalBand||!rawSelectedDonor||Code1M.Load(int3(q,0))==0u)continue;
        }
        uint ignored;Data.InterlockedAdd((cu+30u)*4u,1u,ignored);
        if(!IntervalVisible(qc,rc)) {
            if(!knownBlack)continue;
            if(!KnownBlackBoxIntervalVisible(qc,rc,blackLow,blackHigh)) {
                if(!KnownChromaSubtractionVisible(qc,rc,blackLow,blackHigh,blackPrototype))continue;
            }
        }
        InterlockedOr(V3M[(uint2)q],SUPPORT_AA|(physicalBand?OLD_AA:0u),ignored);
        Data.InterlockedAdd((cu+31u)*4u,1u,ignored);
        Data.InterlockedAdd((cu+(physicalBand?32u:33u))*4u,1u,ignored);
    }
}
void RejectRoot(uint root,uint2 rootPixel) {
    uint ignored;InterlockedOr(PassableCache[rootPixel],BAD_RIM,ignored);
}
[numthreads(16,16,1)]
void aggregate_support_main(uint3 d:SV_DispatchThreadID) {
    uint2 p;if(d.z!=0u||!InputReady()||!ROI(d.xy,p))return;
    uint cu=U(Hook.x),ru=128u+32u*Hook.x;
    // Appending and this observation are serial, before the next unit reuses
    // Parent/RootFlags. Unissued units must never consult an older graph.
    if(RawMeta[ru]!=1u||RawMeta[ru+1u]!=1u)return;
    if(!GraphReady()){W(cu+13u,ST_FAULT);return;}
    if(OpaqueBlackOnly(Hook.x)) {
        // The second graph is THIS unit's current physical-black4CC. Neither
        // the global RawM bitmap nor Code1 carries this membership authority.
        if(ValidPoint(int2(p))&&Black(int2(p))&&RawAppendMember(int2(p))) {
            uint ignored;InterlockedOr(V3M[p],PRIVATE_OPAQUE_BLACK_OFFER,ignored);
        }
        return;
    }
    if(!RawComponentWitnessShapeKnown()){Fault(F_SHAPE);return;}
    // Preserve the exact local C2 witness and its width bound for AA donors.
    // Component-wide opaque permission is consumed only in the NEXT dispatch;
    // no thread may publish based on another group's unfinished accumulation.
    bool observedOpaqueRaw=OpaqueRawHasBackgroundWitness(int2(p));
    if(observedOpaqueRaw) {
        uint ignored;InterlockedOr(V3M[p],PRIVATE_OPAQUE_BLACK_OFFER,ignored);
        uint2 rootPixel;
        if(OpaqueRawRootPixel(int2(p),rootPixel))
            InterlockedOr(RawComponentWitness[rootPixel],RAW_COMPONENT_BACKGROUND_WITNESS,ignored);
    }
    if(Code1M.Load(int3(p,0))!=0u&&SelectedGeometryAt(p)==0u&&
       ValidPoint(int2(p))&&InnerEligible.Load(int3(p,0))==1u) {
        uint ignored;InterlockedOr(V3M[p],CERTIFIED_COLLAR,ignored);
    }
    bool widthAvailable=CurrentWidthAvailable(Hook.x);
    if(!CurrentWidthMayReplaceRaw(Hook.x)) {
        // Preserve EXACT Rawappend geometry for unavailable or add-only widths.
        // No added alpha/ValidPoint test: original append has none.
        if(RawAppendMember(int2(p))){uint ignored;InterlockedOr(V3M[p],UNKNOWN_RAW,ignored);}
        // Shared consensus still contributes measured BAND, AA and closure below.
        if(!widthAvailable)return;
    }
    if(all(d.xy==0u)){uint ignored;Data.InterlockedOr(58u*4u,1u<<Hook.x,ignored);}
    bool band=OwnBand(int2(p)),selected=RawSelected(int2(p));uint ignored;
    bool measuredRim=AdjacentMeasuredRawRim(int2(p),band);
    if(measuredRim)InterlockedOr(V3M[p],MEASURED_RIM,ignored);
    if(band)Data.InterlockedAdd((cu+35u)*4u,1u,ignored);
    if(selected)Data.InterlockedAdd((cu+34u)*4u,1u,ignored);
    uint tag=PassableCache[p],root=tag&ROOT_MASK;
    if(root==0u||(tag&ENCLOSED)==0u)return;
    uint2 rp;uint rt;if(!CacheRootValid(root,true,rp,rt))return;
    // Current closed source-root reduction. Pure original LOW|Inner envelope;
    // independent BODY/HIGH fastpaths do not create a whole-root witness.
    if(CurrentWidthAvailable(Hook.x)) {
        if(ExactCurrentOriginalEnvelope(int2(p),R(cu+11u))) {
            uint ignored;InterlockedOr(RawComponentWitness[rp],SOURCE_ROOT_ENVELOPE_WITNESS,ignored);
        }
    }
    bool bad=!ValidPoint(int2(p))||NativeEdge(int2(p))||TrueCoverEdge(int2(p));
    const int2 dirs[4]={int2(1,0),int2(-1,0),int2(0,1),int2(0,-1)};
    [unroll]for(uint k=0u;k<4u;++k) {
        int2 q=int2(p)+dirs[k];
        if(!ValidPoint(q)){bad=true;continue;}
        uint qtag=PassableCache[(uint2)q],qr=qtag&ROOT_MASK;
        if(qr==root) {
            if((qtag&ENCLOSED)==0u){Fault(F_CACHE);bad=true;}
            continue;
        }
        // A different adjacent passable label contradicts the captured original4CC.
        if(qr!=0u){Fault(F_CACHE);bad=true;continue;}
        if(NativeEdge(q)||TrueCoverEdge(q)||!Black(q)||!Support(q))bad=true;
    }
    if(bad){RejectRoot(root,rp);Data.InterlockedAdd((cu+29u)*4u,1u,ignored);}
}
[numthreads(16,16,1)]
void publish_closure_main(uint3 d:SV_DispatchThreadID) {
    uint2 p;if(d.z!=0u||!InputReady()||!ROI(d.xy,p))return;
    if(OpaqueBlackOnly(Hook.x))return;
    uint cu=U(Hook.x);
    if(!RawComponentWitnessShapeKnown()){Fault(F_SHAPE);return;}
    uint ru=128u+32u*Hook.x;
    if(RawMeta[ru]==1u&&RawMeta[ru+1u]==1u) {
        if(!GraphReady()){W(cu+13u,ST_FAULT);return;}
        // Aggregate has completed for THIS unit. Deep black joins may reuse an
        // independent background witness from the SAME accepted current Raw4CC.
        // Separate counters have no such witness. No AA or other support is born.
        if(OpaqueRawComponentHasBackgroundWitness(int2(p))) {
            uint ignored;InterlockedOr(V3M[p],PRIVATE_OPAQUE_BLACK_OFFER,ignored);
        }
        // Ordered after every source-root reduction write. Recompute the exact
        // current-unit immutable donors; never borrow global packed proof bits.
        if(CurrentWidthAvailable(Hook.x)) {
            bool band=OwnBand(int2(p)),selected=RawSelected(int2(p));
            bool measuredRim=AdjacentMeasuredRawRim(int2(p),band);
            bool observedOpaqueRaw=OpaqueRawHasBackgroundWitness(int2(p));
            if(observedOpaqueRaw&&!band) {
                ObservedRawToMeasuredBandGapRays(int2(p),cu); // Existing one-pixel behavior stays exact.
                ObservedRawToMeasuredBandShortGapRays(int2(p),cu);
            }
            // The additional donor has its own current Raw/background proof. It does
            // not widen shared RawSelected/Support, acquire measured BAND, or seed the
            // original measured-band gap/closure/cache paths. The isolated gap
            // helpers above are separate ordinary-AA paths. Existing OuterRays colour
            // and background guards stay unchanged.
            if((band||selected||measuredRim||observedOpaqueRaw)&&ValidPoint(int2(p))&&Black(int2(p)))
                OuterRays(int2(p),band,selected||observedOpaqueRaw);
        }
    }
    if(!CurrentWidthAvailable(Hook.x))return;
    uint tag=PassableCache[p],root=tag&ROOT_MASK;
    if(root==0u||(tag&ENCLOSED)==0u)return;
    uint2 rp;uint rt;if(!CacheRootValid(root,true,rp,rt))return;
    bool accepted=(rt&BAD_RIM)==0u;uint ignored;
    if(root==p.y*Shape.x+p.x+1u) {
        Data.InterlockedAdd((cu+25u)*4u,1u,ignored);
        Data.InterlockedAdd((cu+(accepted?26u:27u))*4u,1u,ignored);
    }
    if(accepted) {
        // Root acceptance/counts stay unchanged; count only actual pixel writes.
        if(CurrentKnownClosedRecipientAllowed(int2(p))) {
            InterlockedOr(V3M[p],CLOSURE,ignored);Data.InterlockedAdd((cu+28u)*4u,1u,ignored);
        }
    }
}
