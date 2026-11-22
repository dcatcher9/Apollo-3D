// PRIVATE overlap-style original-K experiment. The exact-equality prior remains
// a separate unchanged control. This file is not wired into production/tests.
// Producer inputs are immutable. No F2/KNOWN/header, RGB or outline is generated.
//
// ABI inherited from the strict prior: actual C24 b0, native Shape b1,
// t0RGB,t1OCR,t2Selected,t3Cover,t4CBC,t5F2,t6Families,t7SCT,t8PriorRead,
// u0PriorWrite,u1CandidateWrite. Additional t9OriginalCurrentM,u2UnionMWrite.
// New table shape is8x20 (NOT the strict control's8x16): rows0..15 retain the
// documented per-unit diagnostic layout; rows16..18 contain exact C24, row19
// contains the role/shape header. Clear/overwrite fresh per frame. The future
// owner MUST byte-join the complete immutable consumed source tuple between
// prior and candidate dispatch, and export exact inputs, table and both masks.
#include "host_sbs_shared_width_prior_research_cs.hlsl"
Texture2D<uint> OriginalCurrentM : register(t9);
RWTexture2D<uint> UnionMWrite : register(u2);

static const uint OVERLAP_TAG=0x3157504fu; // OPW1 private role, schema1
static const uint OP_UNAVAILABLE=0u,OP_LOCAL_STRONG=1u,OP_BORROWED=2u;
static const uint OP_CONFLICTING_WIDTHS=3u,OP_NO_DONOR=4u,OP_NO_STYLE=5u;
static const uint OP_NO_FILL=6u,OP_BAD_RECEIVER=7u,OP_AMBIGUOUS_FAMILIES=8u,OP_LOCAL_CONFLICT=9u;
// Pair diagnostics:1notstrong,2recipientgeometry,4disjointK,8unmatchedfamilies,
//16self,32unavailablecurrentreceiver,64ambiguousFGoverlap. No native-height or
// exact K/FG-box equality is used as evidence of font identity or font scale.
bool OverlapBox(uint3 al,uint3 ah,uint3 bl,uint3 bh) {
    return all(max(al,bl)<=min(ah,bh));
}
bool OverlapK(uint a,uint b) {
    uint ka=PhysicalModelBank(a),kb=PhysicalModelBank(b);
    return OverlapBox(PhysicalUnpack(CBCMeta[ka+11u]),PhysicalUnpack(CBCMeta[ka+12u]),
                      PhysicalUnpack(CBCMeta[kb+11u]),PhysicalUnpack(CBCMeta[kb+12u]));
}
//0unmatched,1one unambiguous bijection,2ambiguous. Both directional degree
// checks are mandatory: equal counts or a one-way peer search is insufficient.
uint QualifiedOverlapMatch(uint a,uint b) {
    uint ma=(FamilyModels.Load(int3(0u,a,0))>>8u)&255u,mb=(FamilyModels.Load(int3(0u,b,0))>>8u)&255u;
    if(ma==0u||countbits(ma)!=countbits(mb))return 0u;
    bool unmatched=false,ambiguous=false;
    [loop]for(uint direction=0u;direction<2u;++direction) {
        uint from=direction==0u?a:b,to=direction==0u?b:a;
        uint fm=direction==0u?ma:mb,tm=direction==0u?mb:ma;
        [loop]for(uint i=0u;i<8u;++i)if((fm&(1u<<i))!=0u) {
            uint3 lo=PhysicalUnpack(FamilyModels.Load(int3(1u+2u*i,from,0))),hi=PhysicalUnpack(FamilyModels.Load(int3(2u+2u*i,from,0)));uint degree=0u;
            [loop]for(uint j=0u;j<8u;++j)if((tm&(1u<<j))!=0u) {
                uint3 tl=PhysicalUnpack(FamilyModels.Load(int3(1u+2u*j,to,0))),th=PhysicalUnpack(FamilyModels.Load(int3(2u+2u*j,to,0)));
                if(OverlapBox(lo,hi,tl,th))++degree;
            }
            unmatched=unmatched||degree==0u;ambiguous=ambiguous||degree>1u;
        }
    }
    return ambiguous?2u:(unmatched?0u:1u);
}
bool BorrowedGeometry(uint unit,uint extent) {
    uint cu=CU(unit),height=NativeHeight(unit),u=UB(unit);
    uint4 rect=uint4(CBCMeta[u+24u],CBCMeta[u+25u],CBCMeta[u+26u],CBCMeta[u+27u]);
    // The canonical profile search requires t<recipient's own native core
    // height. It is a bound, not a cross-line style/scale equality test.
    return extent>0u&&extent<=16u&&extent<height&&R(cu+14u)==1u&&R(cu+6u)==height&&
           all(rect.xy<rect.zw)&&all(rect.zw<=Shape.xy)&&all(rect.zw-rect.xy>extent);
}
bool WeakReceiver(uint unit) {
    uint cu=CU(unit),status=R(cu+13u);
    // Source/resource/fault/unprepared states never borrow. A weaker local
    // KNOWN measurement can agree with a donor, but it is never overwritten.
    return R(cu)!=0u&&R(cu+14u)==1u&&(status==1u||(status>=4u&&status<=9u))&&StrongDonorBands(unit)==0u;
}
uint OverlapHeaderWord(uint x) {
    if(x==0u)return OVERLAP_TAG;if(x==1u)return 1u;if(x==2u)return Shape.x;if(x==3u)return Shape.y;
    if(x==4u)return CSource.y;if(x==5u)return CurrentOCR[4u];if(x==6u)return 1u;return 0u;
}
[numthreads(8,1,1)]
void overlap_prior_main(uint3 group:SV_GroupID,uint3 tid:SV_GroupThreadID) {
    if(any(group))return;uint w,h;PriorWrite.GetDimensions(w,h);if(w!=8u||h!=20u)return;uint unit=tid.x;
    [unroll]for(uint row=0u;row<20u;++row)PriorWrite[uint2(unit,row)]=0u;
    if(!FrameReady())return;
    [unroll]for(uint row=0u;row<3u;++row)PriorWrite[uint2(unit,16u+row)]=CBWord(unit+8u*row);
    PriorWrite[uint2(unit,19u)]=OverlapHeaderWord(unit);if(unit>=CurrentOCR[4u])return;
    uint cu=CU(unit);PriorWrite[uint2(unit,3u)]=R(cu+13u);PriorWrite[uint2(unit,4u)]=NativeHeight(unit);PriorWrite[uint2(unit,6u)]=R(cu+11u);
    if(!StyleReady(unit)){PriorWrite[uint2(unit,1u)]=OP_NO_STYLE;PriorWrite[uint2(unit,7u)]=32u;return;}
    if(R(cu)==0u){PriorWrite[uint2(unit,1u)]=OP_NO_FILL;return;}
    uint own=StrongDonorBands(unit);
    if(own!=0u){PriorWrite[uint2(unit,0u)]=R(cu+11u);PriorWrite[uint2(unit,1u)]=OP_LOCAL_STRONG;PriorWrite[uint2(unit,2u)]=1u<<unit;PriorWrite[uint2(unit,5u)]=own;return;}
    if(!WeakReceiver(unit)){PriorWrite[uint2(unit,1u)]=OP_BAD_RECEIVER;PriorWrite[uint2(unit,7u)]=32u;return;}
    uint width=0u,donors=0u,totalBands=0u,reasons=0u;bool conflict=false,ambiguous=false;
    [loop]for(uint peer=0u;peer<CurrentOCR[4u];++peer) {
        uint rejection=0u,bands=0u;
        if(peer==unit)rejection=16u;
        else {
            bands=StrongDonorBands(peer);
            if(bands==0u)rejection=1u;
            else {
                uint t=R(CU(peer)+11u);
                if(!BorrowedGeometry(unit,t))rejection|=2u;
                if(!OverlapK(unit,peer))rejection|=4u;
                uint match=QualifiedOverlapMatch(unit,peer);
                if(match==0u)rejection|=8u;if(match==2u)rejection|=64u;
                // Do not cherry-pick a different donor through a genuinely
                // ambiguous compatible-K foreground-family correspondence.
                if(match==2u&&(rejection&4u)==0u)ambiguous=true;
            }
        }
        PriorWrite[uint2(unit,8u+peer)]=rejection;if(peer!=unit)reasons|=rejection;
        if(rejection!=0u)continue;
        uint t=R(CU(peer)+11u);if(width!=0u&&width!=t)conflict=true;
        width=t;donors|=1u<<peer;totalBands+=bands;
    }
    PriorWrite[uint2(unit,7u)]=reasons;PriorWrite[uint2(unit,2u)]=donors;PriorWrite[uint2(unit,5u)]=totalBands;
    if(ambiguous){PriorWrite[uint2(unit,1u)]=OP_AMBIGUOUS_FAMILIES;return;}
    if(conflict){PriorWrite[uint2(unit,1u)]=OP_CONFLICTING_WIDTHS;return;}
    if(donors==0u){PriorWrite[uint2(unit,1u)]=OP_NO_DONOR;return;}
    if(R(cu+13u)==1u&&R(cu+11u)!=width){PriorWrite[uint2(unit,1u)]=OP_LOCAL_CONFLICT;return;}
    PriorWrite[uint2(unit,0u)]=width;PriorWrite[uint2(unit,1u)]=OP_BORROWED;
}
bool OverlapSourceValid(float4 value) {
    return value.a==1.0f&&all(isfinite(value.rgb))&&all(value.rgb>=0.0f)&&all(value.rgb<=1.0f);
}
bool CurrentOverlapTable() {
    uint w,h;PriorRead.GetDimensions(w,h);if(w!=8u||h!=20u)return false;
    [unroll]for(uint i=0u;i<24u;++i)if(PriorRead.Load(int3(i%8u,16u+i/8u,0))!=CBWord(i))return false;
    [unroll]for(uint x=0u;x<8u;++x)if(PriorRead.Load(int3(x,19u,0))!=OverlapHeaderWord(x))return false;
    return true;
}
bool OverlapOwnDomain(int2 p,uint unit) {
    if(any(p<0)||any(p>=int2(Shape.xy)))return false;uint u=UB(unit);
    uint4 rect=uint4(CBCMeta[u+24u],CBCMeta[u+25u],CBCMeta[u+26u],CBCMeta[u+27u]);
    return all(p>=int2(rect.xy))&&all(p<int2(rect.zw))&&(CoverBits.Load(int3(p,0))&(1u<<unit))!=0u;
}
bool CurrentExteriorRowMatch(uint2 p,uint3 code) {
    [loop]for(uint x=0u;x<Shape.x;++x) {
        int2 q=int2(x,p.y);if(CoverBits.Load(int3(q,0))!=0u)continue;
        float4 v=CurrentRGB.Load(int3(q,0));if(OverlapSourceValid(v)&&all(PhysicalRGB8(v.rgb)==code))return true;
    }
    return false;
}
[numthreads(16,16,1)]
void overlap_candidate_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||any(d.xy>=Shape.xy))return;uint w,h;
    CandidateWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    UnionMWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;CandidateWrite[d.xy]=0u;
    // A malformed original-M shape has no literal output to publish. The
    // future owner must reject that job instead of interpreting UnionM bytes.
    OriginalCurrentM.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    uint original=OriginalCurrentM.Load(int3(d.xy,0));UnionMWrite[d.xy]=original;
    if(original!=0u||!FrameReady()||!CurrentOverlapTable())return;
    float4 value=CurrentRGB.Load(int3(d.xy,0));if(!OverlapSourceValid(value))return;uint3 code=PhysicalRGB8(value.rgb);
    uint candidates=0u;
    [loop]for(uint unit=0u;unit<CurrentOCR[4u];++unit) {
        if(PriorRead.Load(int3(unit,1u,0))!=OP_BORROWED||!StyleReady(unit)||!WeakReceiver(unit)||!OverlapOwnDomain(int2(d.xy),unit)||!PhysicalBlackRGB(value.rgb,unit))continue;
        uint extent=PriorRead.Load(int3(unit,0u,0));if(!BorrowedGeometry(unit,extent))continue;bool seeded=false;
        [loop]for(int dy=-(int)extent;dy<=(int)extent&&!seeded;++dy)[loop]for(int dx=-(int)extent;dx<=(int)extent&&!seeded;++dx) {
            int2 q=int2(d.xy)+int2(dx,dy);
            if(OverlapOwnDomain(q,unit)&&OwnCurrentLow(q,unit))seeded=true;
        }
        if(seeded)candidates|=1u<<unit;
    }
    // Only prospective additions perform the full-row scan. A source-identical
    // exterior background observation vetoes every unit's addition at q.
    if(candidates!=0u&&CurrentExteriorRowMatch(d.xy,code))candidates=0u;
    CandidateWrite[d.xy]=candidates;if(candidates!=0u)UnionMWrite[d.xy]=1u;
}
