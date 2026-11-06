// FG-edge original-source refinement. Permission functions match the verified
// fg-edge-rgb-interval-v1 probe; publication uses an immutable POST-C7/G1 mask.
// No physical-K/Raw/BAND/history change, generated colour or recursive donor.
cbuffer ActualContourCB : register(b0) {
    uint4 CSource;uint4 CField;uint4 CContent;
    uint4 CFrame;uint4 CObservation;uint4 COptions;
};
cbuffer ForegroundEdgeShape : register(b1) { uint4 Shape; };
Texture2D<float4> CurrentRGB : register(t0);
StructuredBuffer<uint> CurrentOCR : register(t1);
Texture2D<uint> SelectedBits : register(t2);
Texture2D<uint> CoverBits : register(t3);
StructuredBuffer<uint> CBCMeta : register(t4);
// t5 is NULL: F2 is never simultaneously bound as SRV and UAV.
Texture2D<uint> FamilyModels : register(t6);
StructuredBuffer<uint> SCT : register(t7);
Texture2D<uint> TrustedInnerBits : register(t8); // Independent current modal/seeded components, unit bits.
Texture2D<uint> CurrentInsideBits : register(t9);
Texture2D<uint> CurrentBody : register(t10);
Texture2D<uint> CurrentFinalM : register(t11); // immutable post-completion copy
Texture2D<uint> RawM : register(t12); // accounting only, never recipient permission
Texture2D<uint> Code1M : register(t13); // accounting only
globallycoherent RWTexture2D<uint> MaskWrite : register(u4);
globallycoherent RWByteAddressBuffer Data : register(u5);
uint UB(uint unit) { return 128u+32u*unit; }
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
bool FrameReady() {
    if(any(Shape.xy==0u)||any(Shape.xy>5120u)||Shape.z!=Shape.x*Shape.y||Shape.w!=0u||any(CSource.xw!=Shape.xy))return false;
    if(CSource.y==0u||CSource.y>5120u||CSource.z>CSource.y||CSource.w!=CSource.y-CSource.z)return false;
    if(any(CField.xy==0u)||any(CField.xy>5120u)||CField.z>=CField.w||CField.w>CField.y||any(CContent.xy>=CContent.zw)||any(CContent.zw>CField.xy))return false;
    if(any(COptions!=uint4(1u,27u,0u,0u))||!any(CFrame.xy)||!any(CObservation.xy))return false;
    uint w,h,n,s,bytes;
    CurrentRGB.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    SelectedBits.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    CoverBits.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    FamilyModels.GetDimensions(w,h);if(w!=17u||h!=8u)return false;
    CBCMeta.GetDimensions(n,s);if(n!=512u||s!=4u)return false;
    Data.GetDimensions(bytes);if(bytes!=68608u*4u)return false;
    CurrentOCR.GetDimensions(n,s);if(n!=256u||s!=4u)return false;
    SCT.GetDimensions(n,s);if(n!=32u||s!=4u)return false;
    if(any(uint4(CBCMeta[0u],CBCMeta[1u],CBCMeta[2u],CBCMeta[3u])!=uint4(0x31434e45u,5u,1u,0u))||CBCMeta[4u]!=1u||CBCMeta[5u]!=1u||CBCMeta[9u]!=1u)return false;
    if(any(Data.Load4(0u)!=uint4(0x37424643u,7u,1u,0u))||R(77u)!=0x31444e49u||R(78u)!=1u||R(83u)!=1u)return false;
    if(CBCMeta[81u]!=0x31424649u||CBCMeta[82u]!=2u||CBCMeta[83u]!=17u||CBCMeta[84u]!=8u||CBCMeta[85u]>255u||CBCMeta[86u]>255u||(CBCMeta[86u]&~CBCMeta[85u])!=0u)return false;
    [loop]for(uint i=0u;i<24u;++i)if(CBCMeta[16u+i]!=CBWord(i)||R(8u+i)!=CBWord(i))return false;
    if(CurrentOCR[0u]!=4u||CurrentOCR[1u]!=0x3852434fu||CurrentOCR[2u]!=1u||CurrentOCR[3u]!=CurrentOCR[4u]||CurrentOCR[4u]>8u)return false;
    if(SCT[0u]!=5u||SCT[1u]!=0x35544353u||SCT[2u]!=1u||CBCMeta[80u]!=SCT[3u]||SCT[22u]!=27u||SCT[23u]!=1u)return false;
    [unroll]for(uint j=0u;j<4u;++j) {
        if(CurrentOCR[5u+j]!=CFrame[j]||SCT[4u+j]!=CFrame[j]||SCT[8u+j]!=CSource[j]||SCT[12u+j]!=CField[j]||SCT[16u+j]!=CContent[j])return false;
        if(CurrentOCR[11u+j]!=CField[j])return false;
    }
    if(CurrentOCR[9u]!=CSource.x||CurrentOCR[10u]!=CSource.y||any(uint2(SCT[20u],SCT[21u])!=CObservation.xy))return false;
    [unroll]for(uint k=24u;k<32u;++k)if(SCT[k]!=0u)return false;
    return true;
}
static const uint FG_ORIGINAL_PRESENT=1u, FG_FRAME_UNAVAILABLE=2u;
static const uint FG_MALFORMED_OPERAND=4u, FG_UNCOVERED=8u;
static const uint FG_INVALID_SOURCE=16u, FG_COLOUR_MODEL_UNAVAILABLE=32u;
static const uint FG_OUTSIDE_OWN_ROI=64u, FG_NO_OWNED_SEED=128u;
static const uint FG_ALPHA_EMPTY=256u, FG_ALPHA_VISIBILITY_UNPROVED=512u;
static const uint FG_NO_VALID_EXTERIOR=1024u, FG_EXTERIOR_COLOUR_PRESENT=2048u;
static const uint FG_RECOVERED=4096u, FG_MALFORMED_ROW_COVER=8192u;
static const uint FG_AMBIGUOUS_SEED_FAMILY=16384u, FG_INPUT_SHAPE_UNAVAILABLE=32768u;

bool FgSourceValid(float4 value) {
    return value.a==1.0f&&all(isfinite(value.rgb))&&
           all(value.rgb>=0.0f)&&all(value.rgb<=1.0f);
}
bool FgWordValid(uint selected,uint cover,uint active) {
    uint low=selected&255u,high=(selected>>8u)&255u;
    // HIGH is not assumed to be a subset of LOW. Only LOW owns a seed here.
    return (selected&0xffff0000u)==0u&&(cover&~active)==0u&&
           ((low|high)&~active)==0u&&(low&~cover)==0u;
}
bool FgOwnRoi(int2 p,uint unit) {
    uint u=UB(unit);
    return all(p>=int2(CBCMeta[u+24u],CBCMeta[u+25u]))&&
           all(p<int2(CBCMeta[u+26u],CBCMeta[u+27u]));
}
// Positive current colour-model readiness, deliberately separate from
// CurrentCBCPhysicalRefinementAdopted/StyleReady, width and Inside(q).
bool FgColourReady(uint unit) {
    if(unit>=CurrentOCR[4u])return false;
    uint u=UB(unit),bank=PhysicalModelBank(unit);
    if(CBCMeta[u]!=1u||CBCMeta[u+1u]!=1u||CBCMeta[u+29u]!=1u||
       !PhysicalModelRecordValid(unit)||CBCMeta[bank+2u]!=1u||
       (CBCMeta[bank+1u]!=1u&&CBCMeta[bank+1u]!=2u))return false;
    uint4 rect=uint4(CBCMeta[u+24u],CBCMeta[u+25u],CBCMeta[u+26u],CBCMeta[u+27u]);
    if(any(rect.xy>=rect.zw)||any(rect.zw>Shape.xy))return false;
    uint first=16u+8u*unit;
    uint4 core=uint4(CurrentOCR[first],CurrentOCR[first+1u],CurrentOCR[first+2u],CurrentOCR[first+3u]);
    if(any(core.xy>=core.zw)||any(core.xy<CContent.xy)||any(core.zw>CContent.zw))return false;
    uint masks=FamilyModels.Load(int3(0u,unit,0)),allMask=masks&255u,qualified=(masks>>8u)&255u;
    if((masks&0xffff0000u)!=0u||qualified==0u||(qualified&~allMask)!=0u||
       (CBCMeta[86u]&(1u<<unit))==0u)return false;
    uint3 kl=PhysicalUnpack(CBCMeta[bank+11u]),kh=PhysicalUnpack(CBCMeta[bank+12u]);
    [loop]for(uint mode=0u;mode<8u;++mode) {
        uint lo=FamilyModels.Load(int3(1u+2u*mode,unit,0));
        uint hi=FamilyModels.Load(int3(2u+2u*mode,unit,0));
        if((allMask&(1u<<mode))==0u) {if(lo!=0u||hi!=0u)return false;continue;}
        uint3 fl=PhysicalUnpack(lo),fh=PhysicalUnpack(hi);
        if(((lo|hi)&0xff000000u)!=0u||any(fl>fh)||!(any(fh<kl)||any(kh<fl)))return false;
    }
    return true;
}
uint FgFamilyMatches(uint3 code,uint unit) {
    uint matches=0u,qualified=(FamilyModels.Load(int3(0u,unit,0))>>8u)&255u;
    [loop]for(uint mode=0u;mode<8u;++mode)if((qualified&(1u<<mode))!=0u) {
        uint3 lo=PhysicalUnpack(FamilyModels.Load(int3(1u+2u*mode,unit,0)));
        uint3 hi=PhysicalUnpack(FamilyModels.Load(int3(2u+2u*mode,unit,0)));
        if(all(code>=lo)&&all(code<=hi))++matches;
    }
    return matches;
}
// Exact CertBlendVisibility arithmetic from the canonical band shader, requiring
// BOTH existing bits. a is the single FG fraction, not a per-channel fitted alpha.
// Doubled RGB8 quantization cells and 5/255 visibility are unchanged arithmetic;
// no estimated tolerance or alpha raster is supplied to the shader.
uint FgIntervalReason(uint3 query,uint3 foreground,uint3 blackLow,uint3 blackHigh) {
    uint3 qlo=StrokeHalfLow(query),qhi=StrokeHalfHigh(query);
    uint3 flo=StrokeHalfLow(foreground),fhi=StrokeHalfHigh(foreground);
    uint3 klo=StrokeHalfLow(blackLow),khi=StrokeHalfHigh(blackHigh);
    uint ln=0u,ld=1u,un=1u,ud=1u;
    [unroll]for(uint channel=0u;channel<3u;++channel) {
        if(!StrokeConstrainBackgroundFraction((int)fhi[channel]-(int)khi[channel],
             (int)qlo[channel]-(int)khi[channel],ln,ld,un,ud))return FG_ALPHA_EMPTY;
        if(!StrokeConstrainBackgroundFraction((int)klo[channel]-(int)flo[channel],
             (int)klo[channel]-(int)qhi[channel],ln,ld,un,ud))return FG_ALPHA_EMPTY;
    }
    if(ln*ud>un*ld)return FG_ALPHA_EMPTY;
    return ln*255u>=ld*5u&&un*255u<=ud*250u?0u:FG_ALPHA_VISIBILITY_UNPROVED;
}
void FgExteriorRow(uint y,uint3 code,uint active,
                   out bool available,out bool present,out bool malformed) {
    available=false;present=false;malformed=false;
    [loop]for(uint x=0u;x<Shape.x;++x) {
        uint cover=CoverBits.Load(int3(x,y,0));
        if((cover&~active)!=0u) {malformed=true;return;}
        if(cover!=0u)continue;
        float4 value=CurrentRGB.Load(int3(x,y,0));
        if(!FgSourceValid(value))continue;
        available=true;
        if(all(PhysicalRGB8(value.rgb)==code))present=true;
    }
}

void PublishCurrentMaskAddition(uint2 p,uint raw,uint code1) {
    uint before;InterlockedOr(MaskWrite[p],1u,before);
    if(before==0u) {
        uint ignored;
        if(raw==0u) {
            Data.InterlockedAdd(7u*4u,1u,ignored);Data.InterlockedAdd(70u*4u,1u,ignored);
        } else Data.InterlockedAdd(73u*4u,0xffffffffu,ignored);
        if(code1!=0u)Data.InterlockedAdd(76u*4u,0xffffffffu,ignored);
    }
}

bool TrustedCurrentCover(uint2 p,uint unit) {
    uint first=144u+8u*unit;
    uint4 r=uint4(CurrentOCR[first],CurrentOCR[first+1u],CurrentOCR[first+2u],CurrentOCR[first+3u]);
    if(any(r.xy>=r.zw)||any(r.xy<CContent.xy)||any(r.zw>CContent.zw))return false;
    uint2 scaled=2u*CContent.xy*CSource.xy+
        (2u*uint2(p.x,p.y+CSource.z)+1u)*(CContent.zw-CContent.xy);
    return all(scaled>=2u*r.xy*CSource.xy)&&all(scaled<2u*r.zw*CSource.xy);
}

// Runs after every original completion stage. A trusted inner component neither supplies
// physical-black adoption nor becomes an outline/AA/history donor for the original pipeline.
[numthreads(16,16,1)]
void trusted_inner_publish_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||any(d.xy>=Shape.xy))return;
    uint w,h;TrustedInnerBits.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    uint bits=TrustedInnerBits.Load(int3(d.xy,0));if(bits==0u)return;
    if(!FrameReady()||R(3u)!=0u)return;
    uint active=(1u<<CurrentOCR[4u])-1u;
    if((bits&~active)!=0u)return;
    [loop]for(uint unit=0u;unit<CurrentOCR[4u];++unit)if((bits&(1u<<unit))!=0u) {
        uint u=UB(unit);
        if(CBCMeta[u]!=1u||CBCMeta[u+1u]!=1u||!TrustedCurrentCover(d.xy,unit))return;
    }
    if(!FgSourceValid(CurrentRGB.Load(int3(d.xy,0))))return;
    MaskWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    RawM.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    Code1M.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    uint raw=RawM.Load(int3(d.xy,0)),code1=Code1M.Load(int3(d.xy,0));
    if(raw>1u||code1>1u)return;
    PublishCurrentMaskAddition(d.xy,raw,code1);
}


[numthreads(16,16,1)]
void foreground_edge_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||any(d.xy>=Shape.xy))return;
    uint w,h;
    CurrentFinalM.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    if(CurrentFinalM.Load(int3(d.xy,0))!=0u)return;
    if(!FrameReady()||R(3u)!=0u)return;
    CurrentInsideBits.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    CurrentBody.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    MaskWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    RawM.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    Code1M.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    uint raw=RawM.Load(int3(d.xy,0)),code1=Code1M.Load(int3(d.xy,0));
    if(raw>1u||code1>1u)return;
    uint active=(1u<<CurrentOCR[4u])-1u;
    uint cover=CoverBits.Load(int3(d.xy,0)),selected=SelectedBits.Load(int3(d.xy,0));
    if(!FgWordValid(selected,cover,active)||(CurrentInsideBits.Load(int3(d.xy,0))&~active)!=0u||
       CurrentBody.Load(int3(d.xy,0))>1u||cover==0u)return;
    float4 query=CurrentRGB.Load(int3(d.xy,0));if(!FgSourceValid(query))return;
    [loop]for(uint unit=0u;unit<CurrentOCR[4u];++unit)if((cover&(1u<<unit))!=0u) {
        if(!FgColourReady(unit)||!FgOwnRoi(int2(d.xy),unit))return;
    }
    uint3 qc=PhysicalRGB8(query.rgb);uint offered=0u;
    [loop]for(uint owner=0u;owner<CurrentOCR[4u];++owner) {
        uint bit=1u<<owner;if((cover&bit)==0u)continue;
        uint bank=PhysicalModelBank(owner);
        uint3 kl=PhysicalUnpack(CBCMeta[bank+11u]),kh=PhysicalUnpack(CBCMeta[bank+12u]);
        [unroll]for(int dy=-1;dy<=1;++dy)[unroll]for(int dx=-1;dx<=1;++dx)if(dx!=0||dy!=0) {
            int2 p=int2(d.xy)+int2(dx,dy);
            if(any(p<0)||any(p>=int2(Shape.xy))||!FgOwnRoi(p,owner))continue;
            uint pc=CoverBits.Load(int3(p,0)),ps=SelectedBits.Load(int3(p,0));
            if(!FgWordValid(ps,pc,active))continue;
            if((pc&bit)==0u||(ps&bit)==0u||CurrentFinalM.Load(int3(p,0))!=1u)continue;
            float4 donor=CurrentRGB.Load(int3(p,0));if(!FgSourceValid(donor))continue;
            uint3 fc=PhysicalRGB8(donor.rgb);if(FgFamilyMatches(fc,owner)!=1u)continue;
            if(FgIntervalReason(qc,fc,kl,kh)==0u)offered|=bit;
        }
    }
    if(offered==0u)return;
    bool available,present,malformed;FgExteriorRow(d.y,qc,active,available,present,malformed);
    if(malformed||!available||present)return;
    // Same unique-add accounting as current Inside/C7. Counts are not read by
    // FrameReady or the permission functions. No output M is ever read as a donor.
    PublishCurrentMaskAddition(d.xy,raw,code1);
}
