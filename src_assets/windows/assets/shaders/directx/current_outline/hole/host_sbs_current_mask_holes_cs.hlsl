#ifndef CURRENT_MASK_HOLES_C7_V4_HLSL
#define CURRENT_MASK_HOLES_C7_V4_HLSL
// PRIVATE C7 v4 EXPERIMENT. Original final-M topology plus current owner proofs.
// This is a bounded completion hypothesis, not text alpha or scene recognition.
// Every donor/topology read is immutable original evidence. No new M is read.
// clear -> per-unit capture -> prepare/freeze -> per-unit init/union/cheap reduce/
// width reduce/counter -> publish(pending only) -> Hook8 single final commit.
// reduce_main reuses entry6: Hook.y=0 completes cheap root vetoes, Hook.y=1
// evaluates width only after those immutable component-wide vetoes are known.
cbuffer ActualContourCB : register(b0) {
    uint4 CSource;uint4 CField;uint4 CContent;
    uint4 CFrame;uint4 CObservation;uint4 COptions;
};
cbuffer HoleHook : register(b1) { uint4 Hook; };
cbuffer HoleShape : register(b2) { uint4 Shape; };
Texture2D<float4> CurrentRGB : register(t0);
StructuredBuffer<uint> CurrentOCR : register(t1);
Texture2D<uint> SelectedBits : register(t2);
Texture2D<uint> CoverBits : register(t3);
StructuredBuffer<uint> CBCMeta : register(t4);
StructuredBuffer<uint> RawMeta : register(t5);
StructuredBuffer<uint> CurrentRawParent : register(t6);
StructuredBuffer<uint> CurrentRawFlags : register(t7);
Texture2D<uint> CurrentRawPassable : register(t8);
Texture2D<uint> CurrentInner : register(t9);
Texture2D<uint> CurrentRawWitness : register(t10);
Texture2D<uint> OriginalFinalM : register(t11); // freeze only; NEVER bound at commit
Texture2D<uint> RawM : register(t12);
Texture2D<uint> ProofRead : register(t13); // never bound with ProofWrite
Texture2D<uint> Code1M : register(t14);
Texture2D<uint> FamilyModels : register(t15); // exact immutable Inside 17x8 bank
RWTexture2D<uint> ProofWrite : register(u0); // clear/capture/freeze only
globallycoherent RWStructuredBuffer<uint> Parent : register(u1);
globallycoherent RWStructuredBuffer<uint> Flags : register(u2);
RWByteAddressBuffer HoleArgs : register(u3); // 8 * uint4 = 128 bytes
globallycoherent RWTexture2D<uint> MaskWrite : register(u4); // final commit only
globallycoherent RWByteAddressBuffer Data : register(u5); // F2; no simultaneous SRV

static const uint INVALID=0xffffffffu,WORDS=68608u;
static const uint ORIGINAL_FINAL_M=1u<<16u,PROOF_ALLOWED=0x1ffffffu;
// Low31 bits have ROOT semantics. Pending31 has PIXEL semantics, preserved
// across unit graph reuse and never interpreted as any root certificate.
static const uint ROOT_MEMBER_WIDTH=1u,ROOT_BOUNDARY=2u,ROOT_BOUNDARY_WIDTH=4u;
static const uint ROOT_C3_BOUNDARY=8u,ROOT_STRICT_HIGH_COUNTER=16u,ROOT_BAD_COLOUR=32u;
static const uint ROOT_BAD_LOCAL_VARIATION=64u,ROOT_ALLOWED=127u;
static const uint PIXEL_PENDING=0x80000000u;
// Flags[0] is a separate frame phase header, never a component root.
static const uint PHASE_CLEAR=1u<<24u,PHASE_FROZEN=1u<<25u;
static const uint PHASE_ALLOWED=PHASE_CLEAR|PHASE_FROZEN|0xffffu;
uint UB(uint unit) { return 128u+32u*unit; }
uint CU(uint unit) { return 128u+64u*unit; }
uint R(uint word) { return Data.Load(word*4u); }
uint CBWord(uint i) {
    if(i<4u)return CSource[i];if(i<8u)return CField[i-4u];
    if(i<12u)return CContent[i-8u];if(i<16u)return CFrame[i-12u];
    if(i<20u)return CObservation[i-16u];return COptions[i-20u];
}
uint BlackModelWord(uint word) { return CBCMeta[word]; }
#include "physical_black_model_shared.hlsl"
#include "cbc_unit_state_shared.hlsl"
#include "current_stroke_math_shared.hlsl"
bool Native(int2 p) { return all(p>=0)&&all(p<int2(Shape.xy)); }
uint Id(uint2 p) { return p.y*Shape.x+p.x+1u; }
bool GraphShape() {
    uint np,sp,nf,sf;Parent.GetDimensions(np,sp);Flags.GetDimensions(nf,sf);
    return np==Shape.z+1u&&nf==np&&sp==4u&&sf==4u;
}
void Fault() {
    uint n,s;Parent.GetDimensions(n,s);
    if(n!=0u&&s==4u){uint ignored;InterlockedExchange(Parent[0u],INVALID,ignored);}
}
bool ShapeKnown() {
    return all(Shape.xy>0u)&&all(Shape.xy<=5120u)&&Shape.z==Shape.x*Shape.y&&
           Shape.w==0u&&all(CSource.xw==Shape.xy)&&CSource.y>0u&&CSource.y<=5120u&&
           CSource.z<=CSource.y&&CSource.w<=CSource.y-CSource.z&&
           all(CField.xy>0u)&&all(CField.xy<=5120u)&&all(CContent.xy<CContent.zw)&&all(CContent.zw<=CField.xy);
}
bool SourceValid(int2 p) {
    if(!Native(p))return false;
    float4 v=CurrentRGB.Load(int3(p,0));
    return v.a==1.0f&&all(isfinite(v.rgb))&&all(v.rgb>=0.0f)&&all(v.rgb<=1.0f);
}
bool OwnCover(int2 p,uint unit) {
    return Native(p)&&(CoverBits.Load(int3(p,0))&(1u<<unit))!=0u;
}
bool UnitRect(uint unit,out uint4 rect) {
    rect=0u;if(unit>=8u)return false;uint u=UB(unit);
    rect=uint4(CBCMeta[u+24u],CBCMeta[u+25u],CBCMeta[u+26u],CBCMeta[u+27u]);
    return all(rect.xy<rect.zw)&&all(rect.zw<=Shape.xy);
}
bool Domain(int2 p,uint unit) {
    uint4 rect;if(!UnitRect(unit,rect)||!OwnCover(p,unit))return false;
    return all(p>=int2(rect.xy))&&all(p<int2(rect.zw));
}
bool PixelFromROIPhase(uint2 local,uint phase,out uint2 p) {
    p=0u;uint4 rect;if(Hook.x>=8u||Hook.y!=phase||any(Hook.zw)||!UnitRect(Hook.x,rect))return false;
    if(any(local>=rect.zw-rect.xy))return false;p=rect.xy+local;return true;
}
bool PixelFromROI(uint2 local,out uint2 p) { return PixelFromROIPhase(local,0u,p); }
bool CommonShapes() {
    if(!ShapeKnown()||!GraphShape())return false;
    uint w,h,n,s,bytes;CurrentRGB.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    SelectedBits.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    CoverBits.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    CBCMeta.GetDimensions(n,s);if(n!=512u||s!=4u)return false;
    RawMeta.GetDimensions(n,s);if(n!=512u||s!=4u)return false;
    CurrentOCR.GetDimensions(n,s);if(n!=256u||s!=4u)return false;
    FamilyModels.GetDimensions(w,h);if(w!=17u||h!=8u)return false;
    Data.GetDimensions(bytes);return bytes==WORDS*4u;
}
bool FrameReady(bool requireInside) {
    if(!CommonShapes())return false;
    if(any(uint4(CBCMeta[0u],CBCMeta[1u],CBCMeta[2u],CBCMeta[3u])!=uint4(0x31434e45u,5u,1u,0u)))return false;
    if(CBCMeta[4u]!=1u||CBCMeta[5u]!=1u||CBCMeta[9u]!=1u)return false;
    if(CBCMeta[81u]!=0x31424649u||CBCMeta[82u]!=2u||CBCMeta[83u]!=17u||CBCMeta[84u]!=8u||
       CBCMeta[85u]>255u||CBCMeta[86u]>255u||(CBCMeta[86u]&~CBCMeta[85u])!=0u)return false;
    if(any(uint4(RawMeta[0u],RawMeta[1u],RawMeta[2u],RawMeta[3u])!=uint4(0x31474e52u,1u,1u,0u)))return false;
    if(any(Data.Load4(0u)!=uint4(0x37424643u,7u,1u,0u)))return false;
    [loop]for(uint i=0u;i<24u;++i)
        if(CBCMeta[16u+i]!=CBWord(i)||RawMeta[12u+i]!=CBWord(i)||R(8u+i)!=CBWord(i))return false;
    if(any(uint2(CBCMeta[44u],CBCMeta[45u])!=Shape.xy))return false;
    if(CurrentOCR[0u]!=4u||CurrentOCR[1u]!=0x3852434fu||CurrentOCR[2u]!=1u||
       CurrentOCR[3u]!=CurrentOCR[4u]||CurrentOCR[4u]>8u)return false;
    if(any(uint4(CurrentOCR[5u],CurrentOCR[6u],CurrentOCR[7u],CurrentOCR[8u])!=CFrame)||
       any(uint2(CurrentOCR[9u],CurrentOCR[10u])!=CSource.xy)||
       any(uint4(CurrentOCR[11u],CurrentOCR[12u],CurrentOCR[13u],CurrentOCR[14u])!=CField))return false;
    if(requireInside&&(R(77u)!=0x31444e49u||R(78u)!=1u||R(83u)!=1u))return false;
    return true;
}
bool UnitKnown(uint unit) {
    if(unit>=8u||unit>=CurrentOCR[4u])return false;uint u=UB(unit),cu=CU(unit),bank=PhysicalModelBank(unit);
    uint first=16u+8u*unit;
    uint4 core=uint4(CurrentOCR[first],CurrentOCR[first+1u],CurrentOCR[first+2u],CurrentOCR[first+3u]);
    if(any(core.xy>=core.zw)||any(core.xy<CContent.xy)||any(core.zw>CContent.zw))return false;
    if(RawMeta[u]!=1u||RawMeta[u+1u]!=1u)return false;
    if(!CurrentCBCPhysicalRefinementAdopted(CBCMeta[u],CBCMeta[u+1u],CBCMeta[u+3u],CBCMeta[u+6u],CBCMeta[u+29u]))return false;
    if(!PhysicalModelRecordValid(unit)||CBCMeta[bank+1u]==4u||CBCMeta[bank+2u]!=1u)return false;
    // Exact existing Inside family roles: every observed family box is ordered
    // and disjoint from K; only the qualified foreground submask is classified.
    uint masks=FamilyModels.Load(int3(0u,unit,0)),allMask=masks&255u,mask=(masks>>8u)&255u;
    if((masks&0xffff0000u)!=0u||(mask&~allMask)!=0u||mask==0u||(CBCMeta[86u]&(1u<<unit))==0u)return false;
    uint3 kl=PhysicalUnpack(CBCMeta[bank+11u]),kh=PhysicalUnpack(CBCMeta[bank+12u]);
    [loop]for(uint mode=0u;mode<8u;++mode) {
        uint lo=FamilyModels.Load(int3(1u+2u*mode,unit,0)),hi=FamilyModels.Load(int3(2u+2u*mode,unit,0));
        if((allMask&(1u<<mode))==0u){if(lo!=0u||hi!=0u)return false;continue;}
        uint3 low=PhysicalUnpack(lo),high=PhysicalUnpack(hi);
        if(((lo|hi)&0xff000000u)!=0u||any(low>high)||!(any(high<kl)||any(kh<low)))return false;
    }
    uint4 rect;if(!UnitRect(unit,rect))return false;
    uint t=R(cu+11u);
    return R(cu+14u)==1u&&R(cu+13u)==1u&&t>0u&&t<=16u&&R(cu+36u)==0u;
}
uint BoxDistance(uint3 q,uint3 low,uint3 high) { return StrokeBoxDistance(q,low,high); }
bool CurrentBlackLike(float3 rgb,uint unit) {
    // Exact Inside Classify comparison; a NEW C7 recipient role only. It never
    // changes K, physical Black(), Raw membership, AA or any retained cache.
    uint bank=PhysicalModelBank(unit);uint3 code=(uint3)floor(rgb*255.0f+0.5f);
    uint3 kl=PhysicalUnpack(CBCMeta[bank+11u]),kh=PhysicalUnpack(CBCMeta[bank+12u]);
    uint dK=BoxDistance(code,kl,kh),mask=(FamilyModels.Load(int3(0u,unit,0))>>8u)&255u,dF=255u;
    [loop]for(uint mode=0u;mode<8u;++mode)if((mask&(1u<<mode))!=0u) {
        uint3 low=PhysicalUnpack(FamilyModels.Load(int3(1u+2u*mode,unit,0)));
        uint3 high=PhysicalUnpack(FamilyModels.Load(int3(2u+2u*mode,unit,0)));
        dF=min(dF,BoxDistance(code,low,high));
    }
    return StrokeBlackLikeDecision(rgb,code,kl,kh,dK,dF);
}
bool ProofROShape() { uint w,h;ProofRead.GetDimensions(w,h);return all(uint2(w,h)==Shape.xy); }
bool ProofRWShape() { uint w,h;ProofWrite.GetDimensions(w,h);return all(uint2(w,h)==Shape.xy); }
bool HeaderKnown() { return (Flags[0u]&~PHASE_ALLOWED)==0u&&(Flags[0u]&PHASE_CLEAR)!=0u; }
bool LateReady() {
    return FrameReady(true)&&ProofROShape()&&HeaderKnown()&&
           (Flags[0u]&PHASE_FROZEN)!=0u&&Parent[0u]==0u;
}
bool PlannedUnitPhase(uint phase) {
    return Hook.x<8u&&Hook.y==phase&&!any(Hook.zw)&&LateReady()&&UnitKnown(Hook.x)&&
           (Flags[0u]&(1u<<(8u+Hook.x)))!=0u;
}
bool PlannedUnit() { return PlannedUnitPhase(0u); }
uint FindRaw(uint node) {
    [loop]for(uint hops=0u;hops<Shape.z;++hops) {
        uint next=CurrentRawParent[node];
        if(next==node)return node;
        if(next==0u||next==INVALID||next>node){Fault();return 0u;}node=next;
    }
    Fault();return 0u;
}
bool CurrentC3Opaque(int2 p,uint unit) {
    if(!SourceValid(p)||!OwnCover(p,unit)||!PhysicalBlackRGB(CurrentRGB.Load(int3(p,0)).rgb,unit))return false;
    if(CurrentRawPassable.Load(int3(p,0))!=1u)return false;
    uint root=FindRaw(Id((uint2)p));if(root==0u)return false;
    uint flags=CurrentRawFlags[root];
    if((flags&~3u)!=0u){Fault();return false;}
    if((flags&3u)!=1u)return false;
    uint2 rp=uint2((root-1u)%Shape.x,(root-1u)/Shape.x);
    if(!OwnCover(int2(rp),unit)){Fault();return false;}
    uint witness=CurrentRawWitness.Load(int3(rp,0));
    if((witness&~3u)!=0u){Fault();return false;}
    return (witness&1u)!=0u;
}

[numthreads(16,16,1)]
void clear_main(uint3 d:SV_DispatchThreadID) {
    bool valid=ShapeKnown()&&GraphShape()&&ProofRWShape();
    if(!valid){if(all(d==0u))Fault();return;}
    if(all(d==0u)){Parent[0u]=0u;Flags[0u]=PHASE_CLEAR;}
    if(d.z!=0u||!Native(int2(d.xy)))return;
    ProofWrite[d.xy]=0u;Parent[Id(d.xy)]=INVALID;Flags[Id(d.xy)]=0u;
}

[numthreads(16,16,1)]
void capture_unit_main(uint3 d:SV_DispatchThreadID) {
    uint2 p;if(d.z!=0u||!FrameReady(false)||!ProofRWShape()||!HeaderKnown()||
       (Flags[0u]&PHASE_FROZEN)!=0u||!UnitKnown(Hook.x)||!PixelFromROI(d.xy,p))return;
    uint n,s,nf,sf,w,h;CurrentRawParent.GetDimensions(n,s);CurrentRawFlags.GetDimensions(nf,sf);
    bool shape=n==Shape.z+1u&&nf==n&&s==4u&&sf==4u;
    CurrentRawPassable.GetDimensions(w,h);shape=shape&&all(uint2(w,h)==Shape.xy);
    CurrentInner.GetDimensions(w,h);shape=shape&&all(uint2(w,h)==Shape.xy);
    CurrentRawWitness.GetDimensions(w,h);shape=shape&&all(uint2(w,h)==Shape.xy);
    if(!shape){Fault();return;}
    if(CurrentRawParent[0u]!=0u){Fault();return;}
    uint ignored;if(all(d.xy==0u))InterlockedOr(Flags[0u],1u<<Hook.x,ignored);
    uint bit=1u<<Hook.x,colourBit=1u<<(17u+Hook.x),value=0u;
    if(OwnCover(int2(p),Hook.x)&&SourceValid(int2(p))) {
        uint inner=CurrentInner.Load(int3(p,0));
        if(inner>1u){Fault();return;}
        if((SelectedBits.Load(int3(p,0))&bit)!=0u||inner==1u)value|=bit;
        if(CurrentC3Opaque(int2(p),Hook.x))value|=bit<<8u;
        if(CurrentBlackLike(CurrentRGB.Load(int3(p,0)).rgb,Hook.x))value|=colourBit;
    }
    // Units are serialized. Each replaces only its own two capture bits.
    uint old=ProofWrite[p];ProofWrite[p]=(old&~(bit|(bit<<8u)|colourBit))|value;
}

uint NativeCoreHeight(uint unit);
bool NoShortWidthHistogram(uint unit) {
    uint cu=CU(unit),lower=R(cu+18u),upper=R(cu+19u),rmax=R(cu+6u),bands=R(cu+7u);
    if(rmax==0u||rmax>128u||rmax!=NativeCoreHeight(unit)||bands==0u||bands>64u||lower>upper||upper>=rmax){Fault();return false;}
    uint total=0u;bool shortProfile=false;
    [loop]for(uint band=0u;band<64u;++band) {
        [loop]for(uint extent=0u;extent<128u;++extent) {
            uint count=R(1024u+(unit*64u+band)*128u+extent);
            if(count==0u)continue;
            if(band>=bands||extent==0u||extent>=rmax||count>0xffffffffu-total){Fault();return false;}
            total+=count;if(extent<lower)shortProfile=true;
        }
    }
    if(total!=R(cu+8u)){Fault();return false;}
    return !shortProfile;
}

[numthreads(8,1,1)]
void prepare_holes_main(uint3 group:SV_GroupID,uint3 tid:SV_GroupThreadID) {
    if(any(group))return;uint bytes;HoleArgs.GetDimensions(bytes);if(bytes!=128u){Fault();return;}
    uint unit=tid.x;if(unit>=8u)return;HoleArgs.Store4(unit*16u,0u);
    if(!FrameReady(true)||!ProofROShape()||!HeaderKnown()||Parent[0u]!=0u||!UnitKnown(unit))return;
    if((Flags[0u]&(1u<<unit))==0u||!NoShortWidthHistogram(unit))return;
    uint4 rect;if(!UnitRect(unit,rect))return;
    uint ignored;InterlockedOr(Flags[0u],1u<<(8u+unit),ignored);
    HoleArgs.Store4(unit*16u,uint4((rect.zw-rect.xy+15u)/16u,1u,0u));
}

[numthreads(16,16,1)]
void freeze_final_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||!Native(int2(d.xy))||!FrameReady(true)||!ProofRWShape()||!HeaderKnown())return;
    uint w,h;OriginalFinalM.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy)){Fault();return;}
    uint original=OriginalFinalM.Load(int3(d.xy,0)),proof=ProofWrite[d.xy];
    if(original>1u||(proof&~(PROOF_ALLOWED&~ORIGINAL_FINAL_M))!=0u){Fault();return;}
    ProofWrite[d.xy]=proof|(original!=0u?ORIGINAL_FINAL_M:0u);
    if(all(d.xy==0u)){uint ignored;InterlockedOr(Flags[0u],PHASE_FROZEN,ignored);}
}

bool OriginalM0(int2 p,uint unit) { return Domain(p,unit)&&(ProofRead.Load(int3(p,0))&ORIGINAL_FINAL_M)==0u; }
uint AtomicParent(uint id) { uint value;InterlockedCompareExchange(Parent[id],0u,0u,value);return value; }
uint FindAtomic(uint start) {
    uint node=start;
    [allow_uav_condition]while(true) {
        uint parent=AtomicParent(node);
        if(parent==0u||parent==INVALID||parent>node){Fault();return INVALID;}
        if(parent==node){uint ignored;InterlockedMin(Parent[start],parent,ignored);return parent;}
        uint grandparent=AtomicParent(parent);
        if(grandparent==0u||grandparent==INVALID||grandparent>parent){Fault();return INVALID;}
        uint ignored;InterlockedMin(Parent[node],grandparent,ignored);node=parent;
    }
    Fault();return INVALID;
}
void Join(uint first,uint second) {
    [allow_uav_condition]while(true) {
        uint a=FindAtomic(first),b=FindAtomic(second);
        if(a==INVALID||b==INVALID||a==b)return;
        uint high=max(a,b),low=min(a,b),observed;
        InterlockedCompareExchange(Parent[high],high,low,observed);if(observed==high)return;
    }
}

[numthreads(16,16,1)]
void init_main(uint3 d:SV_DispatchThreadID) {
    uint2 p;if(d.z!=0u||!PlannedUnit()||!PixelFromROI(d.xy,p))return;
    if((ProofRead.Load(int3(p,0))&~PROOF_ALLOWED)!=0u){Fault();return;}
    uint id=Id(p);Parent[id]=OriginalM0(int2(p),Hook.x)?id:INVALID;
    Flags[id]&=PIXEL_PENDING; // Preserve earlier units' pending PIXEL offers.
}

[numthreads(16,16,1)]
void union_main(uint3 d:SV_DispatchThreadID) {
    uint2 p;if(d.z!=0u||!PlannedUnit()||!PixelFromROI(d.xy,p)||!OriginalM0(int2(p),Hook.x))return;
    uint id=Id(p);
    if(p.x!=0u&&OriginalM0(int2(p)-int2(1,0),Hook.x))Join(id,id-1u);
    if(p.y!=0u&&OriginalM0(int2(p)-int2(0,1),Hook.x))Join(id,id-Shape.x);
}

bool OriginalEnvelope(int2 p,uint unit,uint extent) {
    uint bit=1u<<unit;
    if(!Domain(p,unit))return false;
    if((ProofRead.Load(int3(p,0))&bit)!=0u)return true;
    [loop]for(int dy=-(int)extent;dy<=(int)extent;++dy) {
        [loop]for(int dx=-(int)extent;dx<=(int)extent;++dx) {
            int2 q=p+int2(dx,dy);
            if(Domain(q,unit)&&(ProofRead.Load(int3(q,0))&bit)!=0u)return true;
        }
    }
    return false;
}
int CeilDiv(int n,int d) { return n>=0?(n+d-1)/d:n/d; }
uint NativeCoreHeight(uint unit) {
    int dy=(int)(CContent.w-CContent.y);if(dy<=0)return 0u;uint first=16u+8u*unit;
    int top=CeilDiv(2*((int)CurrentOCR[first+1u]-(int)CContent.y)*(int)CSource.y-dy,2*dy);
    int bottom=CeilDiv(2*((int)CurrentOCR[first+3u]-(int)CContent.y)*(int)CSource.y-dy,2*dy);
    return(uint)max(1,bottom-top);
}
bool StrictHighCounter(int2 p,uint unit,uint height) {
    uint highBit=1u<<(8u+unit);
    const int2 dirs[8]={int2(0,-1),int2(1,-1),int2(1,0),int2(1,1),
                       int2(0,1),int2(-1,1),int2(-1,0),int2(-1,-1)};
    uint mask=0u;
    [loop]for(uint ray=0u;ray<8u;++ray) {
        [loop]for(uint step=1u;step<=height;++step) {
            int2 q=p+(int)step*dirs[ray];if(!OwnCover(q,unit)||!SourceValid(q))break;
            // Full original-HIGH ray: LOW/Inner/black/newM never stop it.
            if((SelectedBits.Load(int3(q,0))&highBit)!=0u){mask|=1u<<ray;break;}
        }
    }
    if(countbits(mask)<3u)return false;
    // Strict surround: no closed180-degree halfplane contains every hit.
    [loop]for(uint start=0u;start<8u;++start) {
        uint closedHalfplane=0u;
        [loop]for(uint step=0u;step<5u;++step)closedHalfplane|=1u<<((start+step)&7u);
        if((mask&~closedHalfplane)==0u)return false;
    }
    return true;
}
bool DirectCurrentVariationDonor(int2 p,uint unit) {
    uint bank=PhysicalModelBank(unit);
    int3 span=int3(PhysicalUnpack(CBCMeta[bank+12u]))-int3(PhysicalUnpack(CBCMeta[bank+11u]));
    uint3 code=(uint3)floor(CurrentRGB.Load(int3(p,0)).rgb*255.0f+0.5f);
    uint opaqueBit=1u<<(8u+unit);
    [loop]for(int dy=-1;dy<=1;++dy)[loop]for(int dx=-1;dx<=1;++dx)if(dx!=0||dy!=0) {
        int2 q=p+int2(dx,dy);if(!Domain(q,unit)||!SourceValid(q))continue;
        uint proof=ProofRead.Load(int3(q,0));
        if((proof&(ORIGINAL_FINAL_M|opaqueBit))!=(ORIGINAL_FINAL_M|opaqueBit))continue;
        uint3 donor=(uint3)floor(CurrentRGB.Load(int3(q,0)).rgb*255.0f+0.5f);
        if(all(abs(int3(code)-int3(donor))<=span))return true;
    }
    return false;
}

uint FindStable(uint node);
[numthreads(16,16,1)]
void reduce_main(uint3 d:SV_DispatchThreadID) {
    uint2 p;
    if(d.z!=0u||Hook.y>1u||!PlannedUnitPhase(Hook.y)||
       !PixelFromROIPhase(d.xy,Hook.y,p)||!OriginalM0(int2(p),Hook.x))return;
    [branch]if(Hook.y==0u) {
        uint root=FindAtomic(Id(p));if(root==INVALID)return;
        uint flags=0u;
        // Complete every original proof/source check even for a rejected root.
        // Width queries are pure and cannot discover a new fault; defer them.
        if((ProofRead.Load(int3(p,0))&(1u<<(17u+Hook.x)))==0u)flags|=ROOT_BAD_COLOUR;
        if(!SourceValid(int2(p))||p.x==0u||p.y==0u||p.x+1u==Shape.x||p.y+1u==Shape.y)flags|=ROOT_BOUNDARY;
        const int2 cardinal[4]={int2(1,0),int2(-1,0),int2(0,1),int2(0,-1)};
        [loop]for(uint k=0u;k<4u;++k)if(!Domain(int2(p)+cardinal[k],Hook.x))flags|=ROOT_BOUNDARY;
        uint opaqueBit=1u<<(8u+Hook.x);
        [loop]for(int dy=-1;dy<=1;++dy)[loop]for(int dx=-1;dx<=1;++dx)if(dx!=0||dy!=0) {
            int2 q=int2(p)+int2(dx,dy);if(!Domain(q,Hook.x))continue;
            uint proof=ProofRead.Load(int3(q,0));if((proof&~PROOF_ALLOWED)!=0u){Fault();return;}
            if((proof&ORIGINAL_FINAL_M)==0u)continue;
            if(!SourceValid(q)){flags|=ROOT_BOUNDARY;continue;}
            // M0N4 / original-M1N8 is the explicit digital-topology dual.
            if((proof&opaqueBit)!=0u)flags|=ROOT_C3_BOUNDARY;
        }
        if(flags!=0u){uint ignored;InterlockedOr(Flags[root],flags,ignored);}
        return;
    }
    // Dispatch ordering makes the cheap component flags complete and immutable.
    // Width bits written here never control another lane's execution.
    uint root=FindStable(Id(p));if(root==0u)return;
    uint word=Flags[root];if((word&~(ROOT_ALLOWED|PIXEL_PENDING))!=0u){Fault();return;}
    if((word&(ROOT_BOUNDARY|ROOT_BAD_COLOUR))!=0u||(word&ROOT_C3_BOUNDARY)==0u)return;
    uint extent=R(CU(Hook.x)+11u),flags=0u;
    if(OriginalEnvelope(int2(p),Hook.x,extent))flags|=ROOT_MEMBER_WIDTH;
    [loop]for(int dy=-1;dy<=1;++dy)[loop]for(int dx=-1;dx<=1;++dx)if(dx!=0||dy!=0) {
        int2 q=int2(p)+int2(dx,dy);if(!Domain(q,Hook.x))continue;
        uint proof=ProofRead.Load(int3(q,0));if((proof&~PROOF_ALLOWED)!=0u){Fault();return;}
        if((proof&ORIGINAL_FINAL_M)==0u||!SourceValid(q))continue;
        if(OriginalEnvelope(q,Hook.x,extent))flags|=ROOT_BOUNDARY_WIDTH;
    }
    if(flags!=0u){uint ignored;InterlockedOr(Flags[root],flags,ignored);}
}

uint FindStable(uint node) {
    [loop]for(uint hops=0u;hops<Shape.z;++hops) {
        uint next=Parent[node];if(next==node)return node;
        if(next==0u||next==INVALID||next>node){Fault();return 0u;}node=next;
    }
    Fault();return 0u;
}

[numthreads(16,16,1)]
void publish_main(uint3 d:SV_DispatchThreadID) {
    if(Hook.x<8u) {
        uint2 p;if(d.z!=0u||!PlannedUnit()||!PixelFromROI(d.xy,p)||!OriginalM0(int2(p),Hook.x))return;
        uint root=FindStable(Id(p));if(root==0u)return;
        uint word=Flags[root];if((word&~(ROOT_ALLOWED|PIXEL_PENDING))!=0u){Fault();return;}
        uint flags=word&ROOT_ALLOWED;
        bool member=(flags&ROOT_MEMBER_WIDTH)!=0u;
        bool counterVeto=(flags&ROOT_STRICT_HIGH_COUNTER)!=0u&&(flags&ROOT_BAD_LOCAL_VARIATION)!=0u;
        bool width=(member||(flags&ROOT_BOUNDARY_WIDTH)!=0u)&&!counterVeto;
        if((flags&(ROOT_BOUNDARY|ROOT_BAD_COLOUR))==0u&&(flags&ROOT_C3_BOUNDARY)!=0u&&width) {
            uint ignored;InterlockedOr(Flags[Id(p)],PIXEL_PENDING,ignored);
        }
        return;
    }
    // One commit ONLY after all eight units' pending traversals have finished.
    // No graph traversal, Fault call or mutable counter relation controls writes.
    if(Hook.x!=8u||any(Hook.yzw)||d.z!=0u||!Native(int2(d.xy))||!LateReady())return;
    uint w,h;MaskWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    RawM.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    Code1M.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    if((ProofRead.Load(int3(d.xy,0))&ORIGINAL_FINAL_M)!=0u||(Flags[Id(d.xy)]&PIXEL_PENDING)==0u)return;
    uint before;InterlockedOr(MaskWrite[d.xy],1u,before);
    if(before==0u) {
        uint ignored;
        if(RawM.Load(int3(d.xy,0))==0u) {
            Data.InterlockedAdd(7u*4u,1u,ignored);Data.InterlockedAdd(70u*4u,1u,ignored);
        } else Data.InterlockedAdd(73u*4u,0xffffffffu,ignored);
        if(Code1M.Load(int3(d.xy,0))!=0u)Data.InterlockedAdd(76u*4u,0xffffffffu,ignored);
    }
}

// Entry8 is appended so original entry0..7 ABI indices stay unchanged.
// It runs AFTER both completed reductions and BEFORE this unit's pending publication.
// Only immutable eligibility bits1/2/4/8/32 control this dispatch. Concurrent
// counter writes16/64 are never read to decide whether another pixel executes.
[numthreads(16,16,1)]
void counter_main(uint3 d:SV_DispatchThreadID) {
    uint2 p;if(d.z!=0u||!PlannedUnit()||!PixelFromROI(d.xy,p)||!OriginalM0(int2(p),Hook.x))return;
    uint root=FindStable(Id(p));if(root==0u)return;
    uint word=Flags[root];if((word&~(ROOT_ALLOWED|PIXEL_PENDING))!=0u){Fault();return;}
    uint completed=word&ROOT_ALLOWED;
    // A member-width witness bounds geometry; it never exempts a missing
    // component from the same original-HIGH counter check as a boundary witness.
    if((completed&(ROOT_BOUNDARY|ROOT_BAD_COLOUR))!=0u||
       (completed&(ROOT_MEMBER_WIDTH|ROOT_BOUNDARY_WIDTH))==0u||
       (completed&ROOT_C3_BOUNDARY)==0u)return;
    uint flags=0u;
    if(StrictHighCounter(int2(p),Hook.x,NativeCoreHeight(Hook.x)))flags|=ROOT_STRICT_HIGH_COUNTER;
    if(!DirectCurrentVariationDonor(int2(p),Hook.x))flags|=ROOT_BAD_LOCAL_VARIATION;
    if(flags!=0u){uint ignored;InterlockedOr(Flags[root],flags,ignored);}
}
#endif
