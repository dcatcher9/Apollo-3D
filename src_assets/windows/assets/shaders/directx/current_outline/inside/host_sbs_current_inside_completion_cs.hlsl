#include "current_stroke_math_shared.hlsl"
// Independent current Inside graph runs before Raw; its final OR remains after publication.
// Existing classifier/components are unchanged. Only local cached anchors reuse its current bits.
cbuffer ActualContourCB : register(b0) {
    uint4 CSource;uint4 CField;uint4 CContent;
    uint4 CFrame;uint4 CObservation;uint4 COptions;
};
cbuffer InsideHook : register(b1) {uint4 Hook;};
cbuffer InsideShape : register(b2) {uint4 Shape;};
Texture2D<float4> CurrentRGB : register(t0);
Texture2D<uint> OriginalBody : register(t1);
Texture2D<uint> CoverBits : register(t3);
StructuredBuffer<uint> CBCMeta : register(t4);
Texture2D<uint> Eligible : register(t5);
StructuredBuffer<uint> Parent : register(t6);
StructuredBuffer<uint> RootFlags : register(t7);
Texture2D<uint> RawM : register(t8);
Texture2D<uint> Code1M : register(t9);
ByteAddressBuffer ProfileRead : register(t10);
Texture2D<uint> Models : register(t13); // 17x8 R32_UINT, immutable afterCBC
Texture2D<uint> InsideBits : register(t14); // neverbound withInsideWrite
RWTexture2D<uint> MaskWrite : register(u0); // emit:Eligible; finalize:ownV3M
RWTexture2D<uint> SeedWrite : register(u1);
RWTexture2D<uint> DomainWrite : register(u2);
RWTexture2D<uint> CoverWrite : register(u3);
globallycoherent RWStructuredBuffer<uint> ParentWrite : register(u4);
globallycoherent RWStructuredBuffer<uint> FlagsWrite : register(u5);
RWTexture2D<uint> InsideWrite : register(u6);
globallycoherent RWByteAddressBuffer StageWords : register(u7); // prepare:16Bargs; publish/final:F2Data
static const uint INVALID=0xffffffffu,WORDS=68608u,FAMILY_TAG=0x31424649u;
uint UB(uint unit){return 128u+32u*unit;}
uint CBWord(uint i){
    if(i<4u)return CSource[i];if(i<8u)return CField[i-4u];
    if(i<12u)return CContent[i-8u];if(i<16u)return CFrame[i-12u];
    if(i<20u)return CObservation[i-16u];return COptions[i-20u];
}
bool Native(uint2 p){return all(p<Shape.xy);}
bool DataROShape(){uint n;ProfileRead.GetDimensions(n);return n==WORDS*4u;}
bool DataRWShape(){uint n;StageWords.GetDimensions(n);return n==WORDS*4u;}
bool CommonFrameShape(){
    if(any(Shape.xy==0u)||any(Shape.xy>5120u)||Shape.z!=Shape.x*Shape.y||Shape.w!=0u||any(CSource.xw!=Shape.xy))return false;
    uint n,s;CBCMeta.GetDimensions(n,s);if(n!=512u||s!=4u)return false;
    if(any(uint4(CBCMeta[0u],CBCMeta[1u],CBCMeta[2u],CBCMeta[3u])!=uint4(0x31434e45u,5u,1u,0u)))return false;
    if(CBCMeta[9u]!=1u||CBCMeta[4u]!=1u||CBCMeta[5u]!=1u)return false;
    if(CBCMeta[81u]!=FAMILY_TAG||CBCMeta[82u]!=2u||CBCMeta[83u]!=17u||CBCMeta[84u]!=8u||CBCMeta[85u]>255u||CBCMeta[86u]>255u||(CBCMeta[86u]&~CBCMeta[85u])!=0u)return false;
    [unroll]for(uint i=0u;i<24u;++i)if(CBCMeta[16u+i]!=CBWord(i))return false;
    uint w,h;CurrentRGB.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    OriginalBody.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    CoverBits.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    Models.GetDimensions(w,h);return w==17u&&h==8u;
}
bool FrameReadyRO(){
    if(!DataROShape()||!CommonFrameShape())return false;
    if(any(ProfileRead.Load4(0u)!=uint4(0x37424643u,7u,1u,0u)))return false;
    if(ProfileRead.Load(77u*4u)!=0x31444e49u||ProfileRead.Load(78u*4u)!=1u)return false;
    [unroll]for(uint i=0u;i<24u;++i)if(ProfileRead.Load((8u+i)*4u)!=CBWord(i))return false;
    return true;
}
bool FrameReadyRW(){
    if(!DataRWShape()||!CommonFrameShape())return false;
    if(any(StageWords.Load4(0u)!=uint4(0x37424643u,7u,1u,0u)))return false;
    if(StageWords.Load(77u*4u)!=0x31444e49u||StageWords.Load(78u*4u)!=1u)return false;
    [unroll]for(uint i=0u;i<24u;++i)if(StageWords.Load((8u+i)*4u)!=CBWord(i))return false;
    return true;
}
uint3 Unpack(uint v){return uint3(v&255u,(v>>8u)&255u,(v>>16u)&255u);}
uint BoxDistance(uint3 q,uint3 low,uint3 high){
    return StrokeBoxDistance(q,low,high);
}
bool UnitReady(uint unit){
    if(unit>=8u)return false;uint u=UB(unit),bank=384u+16u*unit;
    if(CBCMeta[bank+1u]==4u)return false; // Opaque Raw4CC is not an Inside/style model.
    if(CBCMeta[u]!=1u||CBCMeta[u+1u]!=1u||CBCMeta[u+29u]!=1u||CBCMeta[bank+2u]!=1u)return false;
    uint masks=Models.Load(int3(0u,unit,0)),allMask=masks&255u,mask=(masks>>8u)&255u;
    if((masks&0xffff0000u)!=0u||(mask&~allMask)!=0u||mask==0u||(CBCMeta[86u]&(1u<<unit))==0u)return false;
    uint3 kl=Unpack(CBCMeta[bank+11u]),kh=Unpack(CBCMeta[bank+12u]);
    [unroll]for(uint mode=0u;mode<8u;++mode){
        uint lo=Models.Load(int3(1u+2u*mode,unit,0)),hi=Models.Load(int3(2u+2u*mode,unit,0));
        if((allMask&(1u<<mode))==0u){if(lo!=0u||hi!=0u)return false;continue;}
        uint3 low=Unpack(lo),high=Unpack(hi);
        if(((lo|hi)&0xff000000u)!=0u||any(low>high)||!(any(high<kl)||any(kh<low)))return false;
    }
    return true;
}
bool ROI(uint2 local,out uint2 p){
    p=0u;if(Hook.x>=8u||any(Hook.yzw))return false;uint u=UB(Hook.x);
    uint4 r=uint4(CBCMeta[u+24u],CBCMeta[u+25u],CBCMeta[u+26u],CBCMeta[u+27u]);
    if(any(r.xy>=r.zw)||any(r.zw>Shape.xy)||any(local>=r.zw-r.xy))return false;p=r.xy+local;return true;
}
bool ColourValid(float4 v){return v.a==1.0f&&all(isfinite(v.rgb))&&all(v.rgb>=0.0f)&&all(v.rgb<=1.0f);}
void Classify(float3 rgb,uint unit,out bool barrier,out uint dF){
    uint bank=384u+16u*unit;uint3 code=(uint3)floor(rgb*255.0f+0.5f);
    uint3 kl=Unpack(CBCMeta[bank+11u]),kh=Unpack(CBCMeta[bank+12u]);
    uint dK=BoxDistance(code,kl,kh),mask=(Models.Load(int3(0u,unit,0))>>8u)&255u;dF=255u;
    [unroll]for(uint mode=0u;mode<8u;++mode)if((mask&(1u<<mode))!=0u){
        uint3 low=Unpack(Models.Load(int3(1u+2u*mode,unit,0))),high=Unpack(Models.Load(int3(2u+2u*mode,unit,0)));
        dF=min(dF,BoxDistance(code,low,high));
    }
    barrier=StrokeBlackLikeDecision(rgb,code,kl,kh,dK,dF);
}
[numthreads(1,1,1)]
void prepare_unit_main(uint3 g:SV_GroupID){
    if(any(g))return;uint bytes;StageWords.GetDimensions(bytes);if(bytes!=16u)return;StageWords.Store4(0u,0u);
    if(Hook.x>=8u||any(Hook.yzw)||!FrameReadyRO())return;
    if(!UnitReady(Hook.x))return;uint u=UB(Hook.x);
    uint4 r=uint4(CBCMeta[u+24u],CBCMeta[u+25u],CBCMeta[u+26u],CBCMeta[u+27u]);
    if(any(r.xy>=r.zw)||any(r.zw>Shape.xy))return;
    StageWords.Store4(0u,uint4((r.zw-r.xy+15u)/16u,1u,0u));
}
[numthreads(16,16,1)]
void emit_init_main(uint3 d:SV_DispatchThreadID){
    uint2 p;if(d.z!=0u||!FrameReadyRO()||!ROI(d.xy,p))return;
    uint np,sp,nf,sf;ParentWrite.GetDimensions(np,sp);FlagsWrite.GetDimensions(nf,sf);
    if(np!=Shape.z+1u||nf!=np||sp!=4u||sf!=4u)return;
    if(all(d.xy==0u)){ParentWrite[0u]=0u;FlagsWrite[0u]=0u;}
    uint bit=1u<<Hook.x;bool cover=(CoverBits.Load(int3(p,0))&bit)!=0u;
    float4 v=CurrentRGB.Load(int3(p,0));bool barrier=true;uint dF=255u;
    if(cover&&ColourValid(v))Classify(v.rgb,Hook.x,barrier,dF);
    bool passable=cover&&ColourValid(v)&&!barrier;
    MaskWrite[p]=passable?1u:0u;
    SeedWrite[p]=passable&&(OriginalBody.Load(int3(p,0))&bit)!=0u&&dF==0u?1u:0u;
    DomainWrite[p]=cover?Hook.x+1u:0u;CoverWrite[p]=cover?1u:0u;
    uint id=p.y*Shape.x+p.x+1u;ParentWrite[id]=passable?id:INVALID;FlagsWrite[id]=0u;
}
bool GraphShape(){uint n,s,nf,sf;Parent.GetDimensions(n,s);RootFlags.GetDimensions(nf,sf);return n==Shape.z+1u&&nf==n&&s==4u&&sf==4u;}
uint FindStable(uint id){
    [loop]for(uint hops=0u;hops<Shape.z;++hops){uint parent=Parent[id];if(parent==id)return id;if(parent==0u||parent==INVALID||parent>id)return 0u;id=parent;}
    return 0u;
}
void FailUnit(){uint ignored;StageWords.InterlockedOr(80u*4u,1u<<Hook.x,ignored);}
[numthreads(16,16,1)]
void publish_inside_main(uint3 d:SV_DispatchThreadID){
    uint2 p;if(d.z!=0u||!FrameReadyRW()||!ROI(d.xy,p))return;
    uint w,h;InsideWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy)){FailUnit();return;}
    // Each serialized unit replaces its own bit at every current ROI texel.
    // Compute the final word once, so unchanged glyph/background needs no write.
    uint bit=1u<<Hook.x,old=InsideWrite[p],updated=old&~bit;
    if(!GraphShape()||Parent[0u]!=0u)FailUnit();
    else {
        uint ignored;StageWords.InterlockedOr(79u*4u,bit,ignored);
        if(Eligible.Load(int3(p,0))==1u){
            uint root=FindStable(p.y*Shape.x+p.x+1u);
            if(root==0u)FailUnit();
            else {
                uint flags=RootFlags[root];
                if((flags&~3u)!=0u)FailUnit();
                else if((flags&3u)==1u)updated|=bit;
            }
        }
    }
    if(updated!=old)InsideWrite[p]=updated;
}
[numthreads(16,16,1)]
void finalize_main(uint3 d:SV_DispatchThreadID){
    if(d.z!=0u||!Native(d.xy))return;
    uint w,h;InsideWrite.GetDimensions(w,h);
    bool insideShapeValid=all(uint2(w,h)==Shape.xy);uint kept=0u;
    // Normalize the early current bits before any frame-readiness return.
    // This preserves the old publication cleanup, including its lack of77/78 gates.
    if(insideShapeValid){
        uint old=InsideWrite[d.xy];
        if(old!=0u){
            bool publicationValid=DataRWShape();
            if(publicationValid)publicationValid=StageWords.Load(2u*4u)==1u&&StageWords.Load(3u*4u)==0u;
            uint good=publicationValid?(StageWords.Load(79u*4u)&~StageWords.Load(80u*4u)&255u):0u;
            uint cover=publicationValid?CoverBits.Load(int3(d.xy,0)):0u;
            kept=old&good&cover;
            if(kept!=old)InsideWrite[d.xy]=kept;
        }
    }
    bool ready=FrameReadyRW()&&insideShapeValid;
    if(all(d.xy==0u)&&DataRWShape())StageWords.Store(83u*4u,ready?1u:0u);
    if(!ready)return; // Existing141 currentpixel remains byte-exact.
    uint inside=kept;uint before=MaskWrite[d.xy];uint ignored;
    if(inside!=0u)StageWords.InterlockedAdd(81u*4u,1u,ignored);
    if(before==0u&&inside!=0u){
        StageWords.InterlockedAdd(82u*4u,1u,ignored);
        if(RawM.Load(int3(d.xy,0))==0u){
            StageWords.InterlockedAdd(7u*4u,1u,ignored);
            // A valid141 zero implies packed0/localCert0, so its legacy
            // reference support is alsozero when RawM iszero.
            StageWords.InterlockedAdd(70u*4u,1u,ignored);
        }else StageWords.InterlockedAdd(73u*4u,0xffffffffu,ignored);
        if(Code1M.Load(int3(d.xy,0))!=0u)StageWords.InterlockedAdd(76u*4u,0xffffffffu,ignored);
    }
    MaskWrite[d.xy]=before!=0u||inside!=0u?1u:0u;
}
