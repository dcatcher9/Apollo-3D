// TEST-ONLY shared original-outline width prior. This is a geometric/style
// hypothesis, never a production mask, local KNOWN measurement or alpha proof.
// Both entrypoints run AFTER the unchanged current-outline helper. All producer
// resources are read-only. The prior and candidate have independent UAVs.
cbuffer ActualContourCB : register(b0) {
    uint4 CSource; uint4 CField; uint4 CContent;
    uint4 CFrame; uint4 CObservation; uint4 COptions;
};
cbuffer ResearchShape : register(b1) { uint4 Shape; };
Texture2D<float4> CurrentRGB : register(t0);
StructuredBuffer<uint> CurrentOCR : register(t1);
Texture2D<uint> SelectedBits : register(t2);
Texture2D<uint> CoverBits : register(t3);
StructuredBuffer<uint> CBCMeta : register(t4);
ByteAddressBuffer F2 : register(t5);
Texture2D<uint> FamilyModels : register(t6);
StructuredBuffer<uint> SCT : register(t7);
Texture2D<uint> PriorRead : register(t8);
RWTexture2D<uint> PriorWrite : register(u0); // 8x16; x=unit, y=field
RWTexture2D<uint> CandidateWrite : register(u1); // per-unit bits, original K only

uint BlackModelWord(uint word) { return CBCMeta[word]; }
#include "../../../src_assets/windows/assets/shaders/directx/current_outline/outline/physical_black_model_shared.hlsl"
#include "../../../src_assets/windows/assets/shaders/directx/current_outline/outline/cbc_unit_state_shared.hlsl"

// Table fields: width, status, donor bits, original local status, native core
// height, independently supporting donor bands, original local width, reason OR.
// Rows8..15 contain per-donor rejection bits:1notstrong,2scale,4K,8foreground,
//16self,32receiver-style-unavailable. These diagnostics grant no permission.
static const uint UNAVAILABLE=0u,LOCAL=1u,SHARED=2u,CONFLICT=3u,NO_DONOR=4u,NO_STYLE=5u,NO_FILL=6u;
uint CU(uint unit) { return 128u+64u*unit; }
uint UB(uint unit) { return 128u+32u*unit; }
uint R(uint word) { return F2.Load(word*4u); }
uint CBWord(uint i) {
    if(i<4u)return CSource[i];if(i<8u)return CField[i-4u];
    if(i<12u)return CContent[i-8u];if(i<16u)return CFrame[i-12u];
    if(i<20u)return CObservation[i-16u];return COptions[i-20u];
}
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
    F2.GetDimensions(bytes);if(bytes!=68608u*4u)return false;
    CurrentOCR.GetDimensions(n,s);if(n!=256u||s!=4u)return false;
    SCT.GetDimensions(n,s);if(n!=32u||s!=4u)return false;
    if(any(uint4(CBCMeta[0u],CBCMeta[1u],CBCMeta[2u],CBCMeta[3u])!=uint4(0x31434e45u,5u,1u,0u))||CBCMeta[4u]!=1u||CBCMeta[5u]!=1u||CBCMeta[9u]!=1u)return false;
    if(any(F2.Load4(0u)!=uint4(0x37424643u,7u,1u,0u))||R(77u)!=0x31444e49u||R(78u)!=1u||R(83u)!=1u)return false;
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
int CeilDiv(int n,int d) { return n>=0?(n+d-1)/d:n/d; }
uint NativeHeight(uint unit) {
    uint first=16u+8u*unit;int dy=(int)(CContent.w-CContent.y);
    int top=CeilDiv(2*((int)CurrentOCR[first+1u]-(int)CContent.y)*(int)CSource.y-dy,2*dy);
    int bottom=CeilDiv(2*((int)CurrentOCR[first+3u]-(int)CContent.y)*(int)CSource.y-dy,2*dy);
    return(uint)max(1,bottom-top);
}
bool StyleReady(uint unit) {
    if(unit>=CurrentOCR[4u])return false;uint u=UB(unit),bank=PhysicalModelBank(unit);
    if(!CurrentCBCPhysicalRefinementAdopted(CBCMeta[u],CBCMeta[u+1u],CBCMeta[u+3u],CBCMeta[u+6u],CBCMeta[u+29u])||!PhysicalModelRecordValid(unit)||CBCMeta[bank+1u]==4u||CBCMeta[bank+2u]!=1u)return false;
    uint first=16u+8u*unit;uint4 core=uint4(CurrentOCR[first],CurrentOCR[first+1u],CurrentOCR[first+2u],CurrentOCR[first+3u]);
    if(any(core.xy>=core.zw)||any(core.xy<CContent.xy)||any(core.zw>CContent.zw))return false;
    uint4 rect=uint4(CBCMeta[u+24u],CBCMeta[u+25u],CBCMeta[u+26u],CBCMeta[u+27u]);
    if(any(rect.xy>=rect.zw)||any(rect.zw>Shape.xy))return false;
    uint masks=FamilyModels.Load(int3(0u,unit,0)),allMask=masks&255u,mask=(masks>>8u)&255u;
    if((masks&0xffff0000u)!=0u||(mask&~allMask)!=0u||mask==0u||(CBCMeta[86u]&(1u<<unit))==0u)return false;
    uint3 kl=PhysicalUnpack(CBCMeta[bank+11u]),kh=PhysicalUnpack(CBCMeta[bank+12u]);
    [loop]for(uint mode=0u;mode<8u;++mode) {
        uint lo=FamilyModels.Load(int3(1u+2u*mode,unit,0)),hi=FamilyModels.Load(int3(2u+2u*mode,unit,0));
        if((allMask&(1u<<mode))==0u){if(lo!=0u||hi!=0u)return false;continue;}
        uint3 low=PhysicalUnpack(lo),high=PhysicalUnpack(hi);
        if(((lo|hi)&0xff000000u)!=0u||any(low>high)||!(any(high<kl)||any(kh<low)))return false;
        // Exact-peer matching below assumes unique qualified family boxes.
        // A duplicated X must not make {X,X} look equal to {X,Y}.
        if((mask&(1u<<mode))!=0u) {
            [loop]for(uint prior=0u;prior<mode;++prior)
                if((mask&(1u<<prior))!=0u&&lo==FamilyModels.Load(int3(1u+2u*prior,unit,0))&&hi==FamilyModels.Load(int3(2u+2u*prior,unit,0)))return false;
        }
    }
    return true;
}
bool SameForeground(uint a,uint b) {
    uint ma=(FamilyModels.Load(int3(0u,a,0))>>8u)&255u,mb=(FamilyModels.Load(int3(0u,b,0))>>8u)&255u;
    if(countbits(ma)!=countbits(mb))return false;
    // Palette slot order is not style. Every qualified box needs an exact peer.
    [loop]for(uint i=0u;i<8u;++i)if((ma&(1u<<i))!=0u) {
        uint lo=FamilyModels.Load(int3(1u+2u*i,a,0)),hi=FamilyModels.Load(int3(2u+2u*i,a,0));bool found=false;
        [loop]for(uint j=0u;j<8u;++j)if((mb&(1u<<j))!=0u&&lo==FamilyModels.Load(int3(1u+2u*j,b,0))&&hi==FamilyModels.Load(int3(2u+2u*j,b,0)))found=true;
        if(!found)return false;
    }
    return true;
}
bool SameStyle(uint a,uint b) {
    uint ka=PhysicalModelBank(a),kb=PhysicalModelBank(b);
    return NativeHeight(a)==NativeHeight(b)&&CBCMeta[ka+11u]==CBCMeta[kb+11u]&&CBCMeta[ka+12u]==CBCMeta[kb+12u]&&SameForeground(a,b);
}
uint StrongDonorBands(uint unit) {
    if(!StyleReady(unit))return 0u;uint cu=CU(unit),t=R(cu+11u),bands=R(cu+7u);
    if(R(cu+13u)!=1u||R(cu+14u)!=1u||R(cu+36u)!=0u||t==0u||t>16u||bands==0u||bands>64u||R(cu+9u)<2u||R(cu+10u)<2u||2u*R(cu+10u)<=R(cu+9u)||R(cu+24u)<2u)return 0u;
    uint2 winners=uint2(R(cu+16u),R(cu+17u));uint count=0u;
    [loop]for(uint band=0u;band<bands;++band) {
        bool winner=band<32u?(winners.x&(1u<<band))!=0u:(winners.y&(1u<<(band-32u)))!=0u;
        if(winner&&R(1024u+(unit*64u+band)*128u+t)!=0u)++count;
    }
    return count>=2u?count:0u;
}
[numthreads(8,1,1)]
void prior_main(uint3 group:SV_GroupID,uint3 tid:SV_GroupThreadID) {
    if(any(group))return;uint w,h;PriorWrite.GetDimensions(w,h);if(w!=8u||h!=16u)return;
    uint unit=tid.x;[unroll]for(uint field=0u;field<16u;++field)PriorWrite[uint2(unit,field)]=0u;
    if(!FrameReady()||unit>=CurrentOCR[4u])return;
    uint cu=CU(unit);PriorWrite[uint2(unit,3u)]=R(cu+13u);PriorWrite[uint2(unit,4u)]=NativeHeight(unit);PriorWrite[uint2(unit,6u)]=R(cu+11u);
    if(!StyleReady(unit)){PriorWrite[uint2(unit,1u)]=NO_STYLE;PriorWrite[uint2(unit,7u)]=32u;return;}
    if(R(cu)==0u){PriorWrite[uint2(unit,1u)]=NO_FILL;return;}
    uint reasons=0u;
    [loop]for(uint peer=0u;peer<CurrentOCR[4u];++peer) {
        uint rejection=0u;
        if(peer==unit)rejection=16u;
        else {
            if(StrongDonorBands(peer)==0u)rejection|=1u;
            if(NativeHeight(unit)!=NativeHeight(peer))rejection|=2u;
            uint ka=PhysicalModelBank(unit),kb=PhysicalModelBank(peer);
            if(CBCMeta[ka+11u]!=CBCMeta[kb+11u]||CBCMeta[ka+12u]!=CBCMeta[kb+12u])rejection|=4u;
            if(!SameForeground(unit,peer))rejection|=8u;
        }
        PriorWrite[uint2(unit,8u+peer)]=rejection;if(peer!=unit)reasons|=rejection;
    }
    PriorWrite[uint2(unit,7u)]=reasons;
    uint own=StrongDonorBands(unit);
    if(own!=0u){PriorWrite[uint2(unit,0u)]=R(cu+11u);PriorWrite[uint2(unit,1u)]=LOCAL;PriorWrite[uint2(unit,2u)]=1u<<unit;PriorWrite[uint2(unit,5u)]=own;return;}
    uint width=0u,donors=0u,totalBands=0u;bool conflict=false;
    [loop]for(uint donor=0u;donor<CurrentOCR[4u];++donor) {
        if(donor==unit)continue;uint support=StrongDonorBands(donor);
        if(support==0u||!SameStyle(unit,donor))continue;
        uint t=R(CU(donor)+11u);if(width!=0u&&width!=t)conflict=true;
        width=t;donors|=1u<<donor;totalBands+=support;
    }
    PriorWrite[uint2(unit,1u)]=conflict?CONFLICT:(donors!=0u?SHARED:NO_DONOR);
    PriorWrite[uint2(unit,2u)]=donors;PriorWrite[uint2(unit,5u)]=totalBands;
    if(!conflict&&donors!=0u)PriorWrite[uint2(unit,0u)]=width;
}
bool OwnCurrentLow(int2 p,uint unit) {
    if(any(p<0)||any(p>=int2(Shape.xy)))return false;uint bit=1u<<unit;
    if((CoverBits.Load(int3(p,0))&bit)==0u||(SelectedBits.Load(int3(p,0))&bit)==0u)return false;
    float4 value=CurrentRGB.Load(int3(p,0));if(value.a!=1.0f||!all(isfinite(value.rgb))||any(value.rgb<0.0f)||any(value.rgb>1.0f))return false;
    uint3 code=PhysicalRGB8(value.rgb);uint mask=(FamilyModels.Load(int3(0u,unit,0))>>8u)&255u;
    // Current colour can revoke a retained LOW candidate. No new fill is inferred.
    [loop]for(uint mode=0u;mode<8u;++mode)if((mask&(1u<<mode))!=0u) {
        uint3 lo=PhysicalUnpack(FamilyModels.Load(int3(1u+2u*mode,unit,0))),hi=PhysicalUnpack(FamilyModels.Load(int3(2u+2u*mode,unit,0)));
        if(all(code>=lo)&&all(code<=hi))return true;
    }
    return false;
}
[numthreads(16,16,1)]
void candidate_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||any(d.xy>=Shape.xy))return;uint w,h;CandidateWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    CandidateWrite[d.xy]=0u;if(!FrameReady())return;PriorRead.GetDimensions(w,h);if(w!=8u||h!=16u)return;
    float4 value=CurrentRGB.Load(int3(d.xy,0));if(value.a!=1.0f||!all(isfinite(value.rgb))||any(value.rgb<0.0f)||any(value.rgb>1.0f))return;
    uint result=0u;uint3 code=PhysicalRGB8(value.rgb);
    [loop]for(uint unit=0u;unit<CurrentOCR[4u];++unit) {
        uint bit=1u<<unit,t=PriorRead.Load(int3(unit,0u,0));
        if(PriorRead.Load(int3(unit,1u,0))!=SHARED||t==0u||t>16u||!StyleReady(unit)||(CoverBits.Load(int3(d.xy,0))&bit)==0u||(SelectedBits.Load(int3(d.xy,0))&bit)!=0u||!PhysicalBlackCode(code,unit))continue;
        bool found=false;
        [loop]for(int dy=-(int)t;dy<=(int)t&&!found;++dy)[loop]for(int dx=-(int)t;dx<=(int)t&&!found;++dx)
            if(OwnCurrentLow(int2(d.xy)+int2(dx,dy),unit))found=true;
        if(found)result|=bit;
    }
    CandidateWrite[d.xy]=result;
}
