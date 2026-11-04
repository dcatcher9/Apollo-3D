#include "include/depth_coordinate_v2_ocr_assert.generated.hlsl"
// Current colour-component evidence plus optional physical-black enclosure support.
// No history/temporal component records, source-age decisions, UI or EMPTY capability.
cbuffer ActualContourCB : register(b0) {
    uint4 CSource; uint4 CField; uint4 CContent;
    uint4 CFrame; uint4 CObservation; uint4 COptions;
};
cbuffer HookCB : register(b1) { uint4 Hook; };
cbuffer ProducerCB : register(b2) {
    uint4 PControl; uint4 PShape; uint4 POrigin; uint4 PRoot; uint4 PModel;
    uint4 PGeneration; uint4 PROI; uint4 PRecipe; uint4 PReserved0; uint4 PReserved1;
};
Texture2D<float4> CurrentRGB : register(t0);
StructuredBuffer<uint> CurrentOCR : register(t1);
StructuredBuffer<uint> BaseValidation : register(t2);
StructuredBuffer<uint> Palette : register(t3);
Texture2D<uint> ModeBody : register(t4); // Original modeBody, later exterior scratch.
Texture2D<uint> ModeCover : register(t5);
StructuredBuffer<uint> GraphParent : register(t6);
StructuredBuffer<uint> GraphFlags : register(t7);
Texture2D<uint> AllBodyBits : register(t8);
Texture2D<uint> AllCoverBits : register(t9);
StructuredBuffer<uint> CurrentSCT : register(t10);
StructuredBuffer<uint> CurrentUIP : register(t11);
ByteAddressBuffer CurrentRoot : register(t12); // Exact copied current dispatchArguments256.
Texture2D<float4> OriginalSource : register(t13); // Actual leased encoded origin SRV.
Texture2D<uint> DebugBaselineBody : register(t14);
Texture2D<uint> DebugBaselineM : register(t15);

globallycoherent RWStructuredBuffer<uint> Meta : register(u0);
RWTexture2D<uint> UnitBitsWrite : register(u1);
RWTexture2D<uint> CoverBitsWrite : register(u2);
RWTexture2D<uint> SelectedBitsWrite : register(u3);
RWTexture2D<uint> CandidateBody : register(u4);
RWTexture2D<uint> CandidateM : register(u5);
RWTexture2D<uint> StageAux : register(u6); // begin/unit_meta:17x8 models; other entries:native exterior. UnitArgs stays rawu7.
RWByteAddressBuffer UnitArgs : register(u7);

// CBC5: Selected LOW8 is geometry; HIGH8 is trusted colour OR known-black-closed nonblack BODY.
// AllBody HIGH8 remains per-mode style potential; do not confuse the two roles.
// ProducerCB remains the existing input schema1; no helper/constant-buffer ABI change.
static const uint TAG=0x31434e45u, SCHEMA=5u, INPUT_SCHEMA=1u, LIMIT=5120u;
static const uint GEOMETRY_BITS=255u;
static const uint ALL_BODY_STYLE_SHIFT=8u, SELECTED_FOREGROUND_SHIFT=8u;
// Phase-local AllBody fields; selection clears them before publication.
static const uint ALL_BODY_RESTORED_SHIFT=16u, ALL_BODY_PENDING_FOREGROUND_SHIFT=24u;
// Independent phase-only seedless offers, never original BODY or its override.
static const uint SELECTED_UNSEEDED_SHIFT=16u;
bool SelectedHasUnseededOffer(uint word,uint bit) { return (word&(bit<<SELECTED_UNSEEDED_SHIFT))!=0u; }
bool AllBodyHasNormalGeometry(uint word,uint bit) { return (word&bit)!=0u; }
bool AllBodyHasStylePotential(uint word,uint bit) { return (word&(bit<<ALL_BODY_STYLE_SHIFT))!=0u; }
bool AllBodyHasRestoredGeometry(uint word,uint bit) { return ((word>>ALL_BODY_RESTORED_SHIFT)&bit)!=0u; }
bool AllBodyHasPendingForeground(uint word,uint bit) { return ((word>>ALL_BODY_PENDING_FOREGROUND_SHIFT)&bit)!=0u; }
bool AllBodyUnitWasUndiscovered(uint word,uint bit) { return ((word|(word>>ALL_BODY_RESTORED_SHIFT))&bit)==0u; }
uint SelectedGeometryBits(uint word) { return word&GEOMETRY_BITS; }
bool SelectedHasGeometry(uint word,uint bit) { return (word&bit)!=0u; }
bool SelectedHasForeground(uint word,uint bit) { return (word&(bit<<SELECTED_FOREGROUND_SHIFT))!=0u; }
static const uint F_CONTEXT=1u, F_BASE=2u, F_SHAPE=4u, F_PALETTE=8u, F_GRAPH=16u, F_BODY_DOMAIN=32u;
bool Has64(uint2 x) { return any(x!=0u); }
bool Native(uint2 p) { return p.x<PShape.x && p.y<PShape.y; }
uint UB(uint slot) { return 128u+32u*slot; }
void Store(uint i,uint v) { uint ignored;InterlockedExchange(Meta[i],v,ignored); }
void Store4(uint i,uint4 v) { Store(i,v.x);Store(i+1u,v.y);Store(i+2u,v.z);Store(i+3u,v.w); }
void Fault(uint f) { uint ignored;InterlockedOr(Meta[3u],f,ignored); }
bool MetaShape() { uint n,s;Meta.GetDimensions(n,s);return n==512u&&s==4u; }
bool PrivateShape() { return all(PShape.xy>0u)&&all(PShape.xy<=LIMIT); }
uint BlackModelWord(uint word) { return Meta[word]; }
#include "physical_black_model_shared.hlsl"
uint CBWord(uint i) {
    if(i==0u)return CSource.x;if(i==1u)return CSource.y;if(i==2u)return CSource.z;if(i==3u)return CSource.w;
    if(i==4u)return CField.x;if(i==5u)return CField.y;if(i==6u)return CField.z;if(i==7u)return CField.w;
    if(i==8u)return CContent.x;if(i==9u)return CContent.y;if(i==10u)return CContent.z;if(i==11u)return CContent.w;
    if(i==12u)return CFrame.x;if(i==13u)return CFrame.y;if(i==14u)return CFrame.z;if(i==15u)return CFrame.w;
    if(i==16u)return CObservation.x;if(i==17u)return CObservation.y;if(i==18u)return CObservation.z;if(i==19u)return CObservation.w;
    if(i==20u)return COptions.x;if(i==21u)return COptions.y;if(i==22u)return COptions.z;return COptions.w;
}
bool BaseProof() {
    uint n,s;BaseValidation.GetDimensions(n,s);if(n!=32u||s!=4u)return false;
    if(any(uint4(BaseValidation[0u],BaseValidation[1u],BaseValidation[2u],BaseValidation[3u])!=uint4(0x31564c49u,1u,1u,0u)))return false;
    [loop]for(uint i=0u;i<24u;++i)if(BaseValidation[4u+i]!=CBWord(i))return false;
    return all(uint4(BaseValidation[28u],BaseValidation[29u],BaseValidation[30u],BaseValidation[31u])==0u);
}
bool TypedCurrent() {
    uint n,s,bytes,w,h;
    if(!PrivateShape())return false;
    CurrentRGB.GetDimensions(w,h);if(any(uint2(w,h)!=PShape.xy)||any(CSource.xw!=PShape.xy))return false;
    if(any(CSource.xy==0u)||any(CSource.xy>LIMIT)||CSource.z>=CSource.y||CSource.w!=CSource.y-CSource.z)return false;
    if(any(CField.xy==0u)||any(CField.xy>LIMIT)||CField.z>=CField.w||CField.w>CField.y||any(CContent.xy>=CContent.zw)||any(CContent.zw>CField.xy))return false;
    if(any(COptions!=uint4(1u,27u,0u,0u))||CObservation.z>1u||CObservation.w!=0u)return false;
    if(any(PControl.xyz!=uint3(TAG,INPUT_SCHEMA,1u))||(PControl.w!=1u&&PControl.w!=2u&&PControl.w!=4u))return false;
    if(!Has64(PModel.xy)||!Has64(PModel.zw)||any(PModel!=uint4(CFrame.xy,CObservation.xy))||any(PGeneration.xy!=CFrame.zw)||PGeneration.z>1u||PGeneration.w!=0u)return false;
    if(any(PROI.zw!=CSource.xy)||any(PReserved0)||any(PReserved1))return false;
    if(PRoot.z!=1u||PRoot.w!=(1u^0x6f435257u)||!Has64(PRoot.xy))return false;
    CurrentRoot.GetDimensions(bytes);if(bytes!=256u)return false;
    uint2 cookies=PRoot.xy^uint2(0xa3756c91u,0x5c8a936eu);
    uint4 d0=CurrentRoot.Load4(0u),d1=CurrentRoot.Load4(16u);
    uint4 r0=CurrentRoot.Load4(32u),r1=CurrentRoot.Load4(48u);
    if(any(r0!=uint4(PRoot.xy,cookies))||any(r1!=uint4(0x54535152u,1u,PRoot.w,0u)))return false;
    if(any(d0!=uint4(1u,1u^0xd1ec15a5u^0x52434f4fu,PRoot.xy))||any(d1!=uint4(cookies,0x47524243u,0x52434f4fu)))return false;
    uint4 infer=CurrentRoot.Load4(80u);
    if(PControl.w==2u) { if(any(infer!=0u))return false; }
    else { if(any(infer!=uint4(1u,1u,1u,0u)))return false; }
    if(any(CurrentRoot.Load4(208u)!=uint4(1u,1u,1u,0u))||any(CurrentRoot.Load3(240u)!=uint3(1u,1u,1u)))return false;
    CurrentOCR.GetDimensions(n,s);if(n!=256u||s!=4u)return false;
    if(any(uint3(CurrentOCR[0u],CurrentOCR[1u],CurrentOCR[2u])!=uint3(4u,0x3852434fu,1u))||CurrentOCR[3u]!=CurrentOCR[4u]||CurrentOCR[4u]>8u||CurrentOCR[15u]!=0u)return false;
    if(any(uint4(CurrentOCR[5u],CurrentOCR[6u],CurrentOCR[7u],CurrentOCR[8u])!=CFrame))return false;
    if(any(uint2(CurrentOCR[9u],CurrentOCR[10u])!=CSource.xy)||any(uint4(CurrentOCR[11u],CurrentOCR[12u],CurrentOCR[13u],CurrentOCR[14u])!=CField))return false;
    CurrentUIP.GetDimensions(n,s);if(n!=(PRecipe.x==5u?104u:40u)||s!=4u)return false;
    if(any(uint3(CurrentUIP[0u],CurrentUIP[1u],CurrentUIP[2u])!=uint3(4u,0x34504955u,1u))||CurrentUIP[3u]>255u||CurrentUIP[22u]!=0u||CurrentUIP[23u]!=0u)return false;
    uint validSlots=(1u<<CurrentOCR[4u])-1u;
    if(CurrentUIP[3u]!=validSlots||(CurrentUIP[21u]&~validSlots)!=0u)return false;
    if(any(uint4(CurrentUIP[4u],CurrentUIP[5u],CurrentUIP[6u],CurrentUIP[7u])!=CFrame)||any(uint2(CurrentUIP[8u],CurrentUIP[9u])!=CObservation.xy))return false;
    if(any(uint2(CurrentUIP[10u],CurrentUIP[11u])!=CSource.xy)||any(uint4(CurrentUIP[12u],CurrentUIP[13u],CurrentUIP[14u],CurrentUIP[15u])!=CField)||any(uint4(CurrentUIP[16u],CurrentUIP[17u],CurrentUIP[18u],CurrentUIP[19u])!=CContent))return false;
    CurrentSCT.GetDimensions(n,s);if(n!=32u||s!=4u)return false;
    if(CurrentSCT[0u]!=5u||CurrentSCT[1u]!=0x35544353u||CurrentSCT[2u]>1u||CurrentSCT[22u]!=27u||CurrentSCT[23u]!=1u)return false;
    if(any(uint4(CurrentSCT[4u],CurrentSCT[5u],CurrentSCT[6u],CurrentSCT[7u])!=CFrame)||any(uint4(CurrentSCT[8u],CurrentSCT[9u],CurrentSCT[10u],CurrentSCT[11u])!=CSource))return false;
    if(any(uint4(CurrentSCT[12u],CurrentSCT[13u],CurrentSCT[14u],CurrentSCT[15u])!=CField)||any(uint4(CurrentSCT[16u],CurrentSCT[17u],CurrentSCT[18u],CurrentSCT[19u])!=CContent))return false;
    if(any(uint2(CurrentSCT[20u],CurrentSCT[21u])!=CObservation.xy)||CurrentUIP[20u]!=CurrentSCT[3u])return false;
    [unroll]for(uint j=24u;j<32u;++j)if(CurrentSCT[j]!=0u)return false;
    return true;
}
bool LegacyFormatKnown() {
    if(PGeneration.z!=1u||PShape.w!=1u||PRecipe.x!=1u||PRecipe.y!=0u||PRecipe.z!=4u||(PRecipe.w&8u)==0u||POrigin.w!=1u)return false;
    bool rgba=PShape.z==28u&&(POrigin.z==28u||POrigin.z==27u);
    bool bgra=PShape.z==87u&&(POrigin.z==87u||POrigin.z==90u);
    if(!rgba&&!bgra)return false;
    uint w,h;OriginalSource.GetDimensions(w,h);
    if(any(uint2(w,h)!=POrigin.xy)||any(POrigin.xy==0u))return false;
    if(any(PROI.xy>POrigin.xy)||any(PROI.zw>POrigin.xy-PROI.xy))return false;
    return true;
}
// Recipe5 source availability. It never substitutes for TypedCurrent model/mode authority.
// CurrentUIP first40 is actualUIP; appended64 is the exact CGS5 lower-band source record.
bool DisplaySnapshotSourceProof() {
    if(PGeneration.z!=0u||PShape.w!=1u||PRecipe.x!=5u||PRecipe.y!=0u||PRecipe.z!=4u||PRecipe.w!=136u||POrigin.w!=1u)return false;
    if(PShape.z!=28u||POrigin.z!=28u||any(PROI.xy)||any(PROI.zw!=CSource.xy))return false;
    uint w,h;OriginalSource.GetDimensions(w,h);
    if(any(uint2(w,h)!=CSource.xy)||any(POrigin.xy!=CSource.xy))return false;
    uint n,s;CurrentUIP.GetDimensions(n,s);if(n!=104u||s!=4u)return false;
    if(any(uint4(CurrentUIP[40u],CurrentUIP[41u],CurrentUIP[42u],CurrentUIP[43u])!=uint4(0x35534743u,1u,5u,1u)))return false;
    if(any(uint4(CurrentUIP[44u],CurrentUIP[45u],CurrentUIP[46u],CurrentUIP[47u])!=PModel))return false;
    if(any(uint4(CurrentUIP[48u],CurrentUIP[49u],CurrentUIP[50u],CurrentUIP[51u])!=uint4(PGeneration.xy,PRoot.xy)))return false;
    if(any(uint4(CurrentUIP[52u],CurrentUIP[53u],CurrentUIP[54u],CurrentUIP[55u])!=CSource))return false;
    uint4 parentROI=uint4(CurrentUIP[56u],CurrentUIP[57u],CurrentUIP[58u],CurrentUIP[59u]);
    uint4 rawShape=uint4(CurrentUIP[60u],CurrentUIP[61u],CurrentUIP[62u],CurrentUIP[63u]);
    // Parent RAW descriptor/transfer are distinct from the encoded analysis carrier.
    if(any(rawShape.xy==0u)||any(rawShape.xy>16384u)||any(rawShape.zw==0u))return false;
    if(any(parentROI.xy>=parentROI.zw)||any(parentROI.zw>rawShape.xy)||any(parentROI.zw-parentROI.xy!=CSource.xy))return false;
    if((CurrentUIP[64u]&8u)==0u||CurrentUIP[65u]!=4u||CurrentUIP[66u]!=1u||CurrentUIP[67u]!=1u)return false;
    if(CurrentUIP[68u]!=1u||CurrentUIP[69u]!=0u||CurrentUIP[70u]>2u||CurrentUIP[71u]!=63u)return false;
    uint pixels=CSource.x*CSource.w,groups=((CSource.x+15u)/16u)*((CSource.w+15u)/16u);
    if(any(uint4(CurrentUIP[72u],CurrentUIP[73u],CurrentUIP[74u],CurrentUIP[75u])!=uint4(pixels,pixels,0u,pixels)))return false;
    if(any(uint4(CurrentUIP[76u],CurrentUIP[77u],CurrentUIP[78u],CurrentUIP[79u])!=uint4(groups,groups,0u,CurrentSCT[3u])))return false;
    if(any(uint4(CurrentUIP[80u],CurrentUIP[81u],CurrentUIP[82u],CurrentUIP[83u])!=CField))return false;
    if(any(uint4(CurrentUIP[84u],CurrentUIP[85u],CurrentUIP[86u],CurrentUIP[87u])!=CContent))return false;
    if(any(uint4(CurrentUIP[88u],CurrentUIP[89u],CurrentUIP[90u],CurrentUIP[91u])!=COptions))return false;
    if(any(uint4(CurrentUIP[92u],CurrentUIP[93u],CurrentUIP[94u],CurrentUIP[95u])!=CObservation))return false;
    if(any(CurrentRoot.Load4(176u)!=uint4((CField.xy+15u)/16u,1u,0u)))return false;
    if(any(CurrentRoot.Load4(224u)!=uint4(0u,1u,1u,0u)))return false;
    if(any(CurrentRoot.Load4(240u)!=uint4(1u,1u,1u,0u)))return false;
    if(any(uint4(CurrentUIP[96u],CurrentUIP[97u],CurrentUIP[98u],CurrentUIP[99u])!=CurrentRoot.Load4(240u)))return false;
    if(any(uint4(CurrentUIP[100u],CurrentUIP[101u],CurrentUIP[102u],CurrentUIP[103u])!=uint4(10u,28u,10u,1u)))return false;
    return true;
}
bool FormatKnown() {
    return PRecipe.x==5u?DisplaySnapshotSourceProof():LegacyFormatKnown();
}

int CeilDiv(int n,int d) { return n>=0?(n+d-1)/d:n/d; }
int4 NativeCover(uint4 r) {
    int dx=(int)(CContent.z-CContent.x),dy=(int)(CContent.w-CContent.y);
    int l=CeilDiv(2*((int)r.x-(int)CContent.x)*(int)CSource.x-dx,2*dx);
    int rr=CeilDiv(2*((int)r.z-(int)CContent.x)*(int)CSource.x-dx,2*dx);
    int t=CeilDiv(2*((int)r.y-(int)CContent.y)*(int)CSource.y-dy,2*dy)-(int)CSource.z;
    int b=CeilDiv(2*((int)r.w-(int)CContent.y)*(int)CSource.y-dy,2*dy)-(int)CSource.z;
    return int4(l,t,rr,b);
}
uint4 Rect(uint first) { return uint4(Palette[first],Palette[first+1u],Palette[first+2u],Palette[first+3u]); }
bool PaletteReady(bool requireMode) {
    uint n,s;Palette.GetDimensions(n,s);if(n!=128u||s!=4u)return false;
    if(Hook.x>=8u||Hook.y>=8u||any(Hook.zw))return false;
    if(Palette[0u]!=0x35434c49u||Palette[1u]!=3u||Palette[2u]!=1u||Palette[3u]!=Hook.x||Palette[4u]>8u||Palette[7u]>8u)return false;
    [loop]for(uint i=0u;i<24u;++i)if(Palette[24u+i]!=CBWord(i))return false;
    if(requireMode) {
        if(Palette[22u]!=2u||Hook.x>=Palette[7u]||Hook.y>=Palette[4u])return false;
    }
    return true;
}
bool PixelFromROI(uint2 local,out uint2 p) {
    p=0u;if(Hook.x>=8u)return false;
    uint u=UB(Hook.x);uint4 r=uint4(Meta[u+24u],Meta[u+25u],Meta[u+26u],Meta[u+27u]);
    if(any(r.xy>=r.zw)||any(r.zw>PShape.xy)||any(local>=r.zw-r.xy))return false;
    p=r.xy+local;return true;
}

// Per-unit immutable family snapshot, captured while this Palette is current.
// col0 LOW8=allvalid-disjoint modes, HIGH8=repeatconfident subset.
// Faithful145 consumes only HIGH8; LOW8 is capture-only future evidence.
// StageAux is always a R32_UINT Texture2D; it is never a buffer-type alias.
static const uint FAMILY_TAG=0x31424649u;
uint PackFamily(uint3 c) { return c.r|(c.g<<8u)|(c.b<<16u); }
bool FamilyShape() {uint w,h;StageAux.GetDimensions(w,h);return w==17u&&h==8u;}
bool CaptureFamilies(uint unit,uint3 blackLow,uint3 blackHigh,bool knownK) {
    if(!FamilyShape()){Fault(F_SHAPE);return false;}
    [unroll]for(uint col=0u;col<17u;++col)StageAux[uint2(col,unit)]=0u;
    uint allMask=0u,confidentMask=0u;
    [loop]for(uint mode=0u;mode<Palette[4u];++mode) {
        uint first=64u+8u*mode;
        uint3 rep16=uint3(Palette[first],Palette[first+1u],Palette[first+2u]);
        uint peak=max(rep16.r,max(rep16.g,rep16.b));uint control=Palette[first+7u]>>24u,radius=control&127u;
        if(peak==0u||any(rep16>65535u)||Palette[first+3u]!=peak||Palette[first+4u]>=64u||
           Palette[first+5u]==0u||Palette[first+5u]>512u||Palette[first+6u]>=512u||radius<1u||radius>65u){Fault(F_PALETTE);return false;}
        if(!knownK)continue;
        uint3 modal=PhysicalUnpack(Palette[first+7u]);
        uint3 rep=(rep16*255u+32767u)/65535u;
        uint3 low=min((uint3)max(int3(modal)-int(radius),0),rep);
        uint3 high=max(min(modal+radius,255u),rep);
        bool disjoint=any(high<blackLow)||any(blackHigh<low);
        if(!disjoint)continue;
        StageAux[uint2(1u+2u*mode,unit)]=PackFamily(low);
        StageAux[uint2(2u+2u*mode,unit)]=PackFamily(high);allMask|=1u<<mode;
        if((control&128u)!=0u)confidentMask|=1u<<mode;
    }
    StageAux[uint2(0u,unit)]=allMask|(confidentMask<<8u);
    if(allMask!=0u){uint ignored;InterlockedOr(Meta[85u],1u<<unit,ignored);}
    if(confidentMask!=0u){uint ignored;InterlockedOr(Meta[86u],1u<<unit,ignored);}
    return true;
}

[numthreads(1,1,1)]
void begin_main(uint3 g:SV_GroupID) {
    if(any(g)||!MetaShape())return;
    [loop]for(uint i=0u;i<512u;++i)Meta[i]=0u;
    Store(0u,TAG);Store(1u,SCHEMA);Store(11u,PControl.w);
    Store(81u,FAMILY_TAG);Store(82u,2u);Store(83u,17u);Store(84u,8u);
    if(!FamilyShape()){Fault(F_SHAPE);return;}
    [loop]for(uint row=0u;row<8u;++row)[loop]for(uint col=0u;col<17u;++col)StageAux[uint2(col,row)]=0u;
    Store4(12u,uint4(CFrame.xy,CObservation.xy));
    Store4(16u,CSource);Store4(20u,CField);Store4(24u,CContent);
    Store4(28u,CFrame);Store4(32u,CObservation);Store4(36u,COptions);
    Store4(40u,PControl);Store4(44u,PShape);Store4(48u,POrigin);Store4(52u,PRoot);Store4(56u,PModel);
    Store4(60u,PGeneration);Store4(64u,PROI);Store4(68u,PRecipe);Store4(72u,PReserved0);Store4(76u,PReserved1);
    bool typed=TypedCurrent();bool base=BaseProof();bool format=false;
    if(typed)format=FormatKnown();
    Store(9u,typed?1u:0u);Store(4u,format?1u:0u);Store(5u,(typed&&base&&format)?1u:0u);
    if(typed)Store(80u,CurrentSCT[3u]);
    if(!typed)Fault(F_CONTEXT);if(!base)Fault(F_BASE);if(!PrivateShape())Fault(F_SHAPE);
}

[numthreads(16,16,1)]
void clear_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||!Native(d.xy))return;
    UnitBitsWrite[d.xy]=0u;CoverBitsWrite[d.xy]=0u;SelectedBitsWrite[d.xy]=0u;
    CandidateBody[d.xy]=0u;CandidateM[d.xy]=0u;
}

uint3 CanonicalHalfBits(uint3 code) {
    precise float3 v=float3(code)/255.0f;
    uint3 h=f32tof16(v)&0xffffu;float3 rounded=f16tof32(h);
    return h-(uint3)(rounded>v);
}
[numthreads(16,16,1)]
void validate_encoding_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||!Native(d.xy)||!MetaShape())return;
    if(Meta[9u]==0u||Meta[3u]!=0u)return;
    // Stage aliases t1=P as raw structured uint bits, t5=currentS. One parallel
    // scan strengthens numeric faults without changing the baseline classifier.
    uint pn,ps,sw,sh;CurrentOCR.GetDimensions(pn,ps);ModeCover.GetDimensions(sw,sh);
    if(pn!=153600u||ps!=4u||any(uint2(sw,sh)!=PShape.xy)){Fault(F_SHAPE);return;}
    float4 actual=CurrentRGB.Load(int3(d.xy,0));
    uint3 actualBits=asuint(actual.rgb);
    if(any((actualBits&0x7f800000u)==0x7f800000u)||any(actual.rgb<0.0f)||any(actual.rgb>1.0f)||asuint(actual.a)!=0x3f800000u||ModeCover.Load(int3(d.xy,0))>1u)Fault(F_BASE);
    uint nativeCount=PShape.x*PShape.y;
    uint native_linear_index=d.y*PShape.x+d.x;
    [loop]for(uint pi=native_linear_index;pi<153600u;pi+=nativeCount) {
        uint bits=CurrentOCR[pi];float probability=asfloat(bits);
        if((bits&0x7f800000u)==0x7f800000u||probability<0.0f||probability>1.0f)Fault(F_BASE);
    }
    if(Meta[4u]==0u)return;
    uint2 original=PROI.xy+uint2(d.x,CSource.z+d.y);
    if(PRecipe.x==5u) {
        // The fullROI carrier has NO proof above cropTop. No recipe5 load may escape band.
        if(any(PROI.xy)||original.x>=CSource.x||original.y<CSource.z||original.y>=CSource.y){Fault(F_SHAPE);return;}
    }
    float4 v=OriginalSource.Load(int3(original,0));
    uint3 code=(uint3)floor(v.rgb*255.0f+0.5f);
    uint3 h=CanonicalHalfBits(code);float3 expected=f16tof32(h);
    bool same=all(asuint(actual.rgb)==asuint(expected))&&asuint(actual.a)==0x3f800000u&&asuint(v.a)==0x3f800000u;
    if(!same){uint ignored;InterlockedAnd(Meta[5u],0u,ignored);}
}

[numthreads(1,1,1)]
void unit_meta_main(uint3 g:SV_GroupID) {
    if(any(g)||!MetaShape()||Hook.x>=8u)return;
    uint u=UB(Hook.x);Store(u+28u,Hook.x);Store(u+31u,0u);
    uint bank=PhysicalModelBank(Hook.x);
    [unroll]for(uint w=0u;w<16u;++w)Store(bank+w,0u);
    if(!PaletteReady(false)||!BaseProof()){Fault(F_PALETTE);return;}
    if(Hook.x>=Palette[7u]||Palette[22u]!=2u)return;
    uint4 a=uint4(Palette[48u],Palette[49u],Palette[50u],Palette[51u]);
    uint4 b=uint4(Palette[52u],Palette[53u],Palette[54u],Palette[55u]);
    uint4 c=uint4(Palette[56u],Palette[57u],Palette[58u],Palette[59u]);
    uint4 d=uint4(Palette[60u],Palette[61u],Palette[62u],Palette[63u]);
    // K4 belongs only to completed CBC publication, never to an upstream palette.
    if(a.y==4u||!PhysicalModelWordsValid(a,b,c,d)){Fault(F_PALETTE);return;}
    Store4(bank,a);Store4(bank+4u,b);Store4(bank+8u,c);Store4(bank+12u,d);
    if(!CaptureFamilies(Hook.x,PhysicalUnpack(c.w),PhysicalUnpack(d.x),a.z==1u))return;
    uint4 cover=Rect(12u),core=Rect(16u),search=Rect(8u);
    if(any(cover.xy>=cover.zw)||any(cover.xy<CContent.xy)||any(cover.zw>CContent.zw)){Fault(F_PALETTE);return;}
    int4 native=NativeCover(cover);int4 clip=int4(max(native.xy,int2(0,0)),min(native.zw,int2(PShape.xy)));
    bool observed=all(native.xy>=0)&&all(native.zw<=int2(PShape.xy))&&all(native.xy<native.zw);
    Store(u,1u);Store(u+1u,observed?1u:0u);Store(u+7u,Palette[4u]);
    Store4(u+8u,cover);Store4(u+12u,asuint(native));Store4(u+16u,core);Store4(u+20u,search);Store4(u+24u,asuint(clip));
    uint ignored;InterlockedOr(Meta[6u],1u<<Hook.x,ignored);
}

uint FindStable(uint start) {
    uint node=start;
    [loop]while(true) {
        uint parent=GraphParent[node];
        if(parent==0u||parent==0xffffffffu||parent>node){Fault(F_GRAPH);return 0u;}
        if(parent==node)return node;
        node=parent;
    }
    return 0u;
}
// Existing native BOX3/AA support scale, reconstructed inside each original4CC.
// collect_main alone aliases DebugBaselineBody/t14 to the CURRENT per-mode Seeds.
// publish_main retains the original accumulated-baseline BODY binding at t14.
// Current per-mode Palette exists only inside the original collect loop. Later
// select_unit_main must never read the palette left behind by another unit.
bool CurrentCollectCore(uint2 p) {
    uint u=UB(Hook.x);
    if(Meta[3u]!=0u||Meta[9u]!=1u||Meta[4u]!=1u||Meta[5u]!=1u||
       Meta[u+1u]!=1u||!Native(p))return false;
    uint4 core=uint4(Meta[u+16u],Meta[u+17u],Meta[u+18u],Meta[u+19u]);
    if(any(core.xy>=core.zw)||any(core.xy<CContent.xy)||any(core.zw>CContent.zw))return false;
    int4 nativeCore=NativeCover(core);
    if(any(int2(p)<nativeCore.xy)||any(int2(p)>=nativeCore.zw))return false;
    return true;
}
bool CurrentCoreOwnAdaptiveRGB8(uint2 p) {
    // Reuse the exact current Meta/core predicate at every call; never cache it.
    if(!CurrentCollectCore(p))return false;
    // CurrentRGB is already the same joined encoded display-band input used by
    // PaletteQuantize; do not convert its RGB again or read an unproved origin.
    float4 source=CurrentRGB.Load(int3(p,0));
    if(source.a!=1.0f||any(!isfinite(source.rgb))||any(source.rgb<0.0f)||any(source.rgb>1.0f))return false;
    uint start=64u+8u*Hook.y;
    uint3 representative=uint3(Palette[start],Palette[start+1u],Palette[start+2u]);
    uint peak=max(representative.r,max(representative.g,representative.b));
    if(peak==0u||any(representative>65535u)||Palette[start+3u]!=peak||
       Palette[start+4u]>=64u||Palette[start+5u]==0u||Palette[start+5u]>512u||
       Palette[start+6u]>=512u||((Palette[start+7u]>>24u)&128u)==0u||((Palette[start+7u]>>24u)&127u)<1u||((Palette[start+7u]>>24u)&127u)>65u)return false;
    // Only the additive current-core exception uses the data-derived repeated-shell radius.
    // Modal evidence is independent of P/edge, but the mode/bin/OriginalBODY
    // and this candidate's ownership remain derived by the original palette.
    uint modeControl=Palette[start+7u]>>24u;
    uint radius=modeControl&127u;
    uint3 modal=PhysicalUnpack(Palette[start+7u]);
    uint3 delta=(uint3)abs(int3(PhysicalRGB8(source.rgb))-int3(modal));
    return max(delta.r,max(delta.g,delta.b))<=radius;
}

// Current per-mode ownership parameters are captured before Palette scratch changes.
// Products in this exact rational interval are bounded by255*255; int32 is exact.
groupshared uint CollectStyleReady,CollectRepeatConfident;
groupshared int3 CollectFLo,CollectFHi,CollectKLo,CollectKHi;
void OwnershipLower(inout int2 bound,int n,int d) {
    if(n*bound.y>bound.x*d)bound=int2(n,d);
}
void OwnershipUpper(inout int2 bound,int n,int d) {
    if(n*bound.y<bound.x*d)bound=int2(n,d);
}
bool OwnershipStyleCode(uint3 code) {
    if(CollectStyleReady==0u)return false;
    int2 lower=int2(0,1),upper=int2(1,1);
    [unroll]for(uint c=0u;c<3u;++c) {
        int a=CollectFLo[c]-CollectKLo[c],b=CollectFHi[c]-CollectKHi[c],q=(int)code[c];
        if(a>0)OwnershipUpper(upper,q-CollectKLo[c],a);
        else if(a<0)OwnershipLower(lower,CollectKLo[c]-q,-a);
        else if(q<CollectKLo[c])return false;
        if(b>0)OwnershipLower(lower,q-CollectKHi[c],b);
        else if(b<0)OwnershipUpper(upper,CollectKHi[c]-q,-b);
        else if(q>CollectKHi[c])return false;
    }
    return lower.x*upper.y<=upper.x*lower.y;
}
void InitCollectOwnership() {
    uint start=64u+8u*Hook.y;
    uint3 rep=uint3(Palette[start],Palette[start+1u],Palette[start+2u]);
    uint peak=max(rep.r,max(rep.g,rep.b)),control=Palette[start+7u]>>24u,radius=control&127u;
    if(peak==0u||any(rep>65535u)||Palette[start+3u]!=peak||Palette[start+4u]>=64u||
       Palette[start+5u]==0u||Palette[start+5u]>512u||Palette[start+6u]>=512u||radius<1u||radius>65u)return;
    CollectRepeatConfident=(control&128u)!=0u?1u:0u;
    uint bank=PhysicalModelBank(Hook.x);
    if(!PhysicalModelRecordValid(Hook.x)||Meta[bank+2u]!=1u)return;
    int3 modal=int3(PhysicalUnpack(Palette[start+7u]));
    int3 representative=int3((rep*255u+32767u)/65535u);
    CollectFLo=min(max(modal-(int)radius,0),representative);
    CollectFHi=max(min(modal+(int)radius,255),representative);
    CollectKLo=int3(PhysicalUnpack(Meta[bank+11u]));
    CollectKHi=int3(PhysicalUnpack(Meta[bank+12u]));
    CollectStyleReady=1u;
}
bool CurrentOwnStyle(uint2 p) {
    if(CollectStyleReady==0u||!Native(p))return false;
    float4 value=CurrentRGB.Load(int3(p,0));
    if(value.a!=1.0f||any(!isfinite(value.rgb))||any(value.rgb<0.0f)||any(value.rgb>1.0f))return false;
    return OwnershipStyleCode(PhysicalRGB8(value.rgb));
}

// Eligible closed, unseeded current-mode components can offer geometry for validation.
// It can only offer temporary pending; physical admission remains in select_unit_main.
uint UnseededColourRoot(uint id) {
    uint first=GraphParent[id];
    if(first==0u||first==0xffffffffu||first>id)return 0u;
    uint node=id;
    [loop]for(uint hops=0u;hops<PShape.x*PShape.y;++hops) {
        uint next=GraphParent[node];
        if(next==0u||next==0xffffffffu||next>node)return 0u;
        if(next==node)return GraphFlags[node]==0u?node:0u;
        node=next;
    }
    return 0u;
}
bool CurrentOwnTrustedFill(uint2 p) {
    if(CollectStyleReady==0u||CollectRepeatConfident==0u)return false;
    float4 value=CurrentRGB.Load(int3(p,0));
    if(value.a!=1.0f||any(!isfinite(value.rgb))||any(value.rgb<0.0f)||any(value.rgb>1.0f))return false;
    int3 code=int3(PhysicalRGB8(value.rgb));
    return all(code>=CollectFLo)&&all(code<=CollectFHi);
}

static const uint COLLECT_HALO=3u, COLLECT_TILE=22u, COLLECT_PIXELS=484u;
static const uint COLLECT_REACH_SOURCE_SEED=0x1u, COLLECT_REACH_MODAL_SEED=0x2u;
// Exact existing native contrast/tangent coherence, float32(0.08), not a new threshold.
static const float COLLECT_COHERENCE=0.08f;
groupshared uint CollectReady;
groupshared uint CollectAnyRoot;
// Any original accepted ModeBody1 is availability, independently of strong-root reconstruction.
groupshared uint CollectOriginalBodyAvailable;
groupshared uint4 CollectOldSeedSearch;
groupshared uint CollectRoot[484];
groupshared uint CollectReachA[484];
groupshared uint CollectReachB[484];
// Reach0x1/0x2 retain exact128 geometry;0x4 is separate style-modal ownership.
// CollectCore bit0 is nativecore, bit1 is own-mode exact F/K style membership.
groupshared float3 CollectRGB[484];
groupshared uint CollectCore[484];
// Exact original colour pixel-centre projection, not conservative cover bounds.
bool PixelInOldSeedSearch(int2 p,uint4 r) {
    if(any(p<0)||any(p>=int2(PShape.xy)))return false;
    uint2 scaled=uint2(2u*CContent.x*CSource.x+(2u*(uint)p.x+1u)*(CContent.z-CContent.x),
        2u*CContent.y*CSource.y+(2u*((uint)p.y+CSource.z)+1u)*(CContent.w-CContent.y));
    return scaled.x>=2u*r.x*CSource.x&&scaled.x<2u*r.z*CSource.x&&
        scaled.y>=2u*r.y*CSource.y&&scaled.y<2u*r.w*CSource.y;
}
uint4 OldSeedSearch() {
    uint4 search=Rect(8u),core=Rect(16u);
    bool ribbon=(CurrentOCR[V2_OCR_RAW_BOX_OFFSET+Hook.x*V2_OCR_BOX_WORD_COUNT+5u]&V2_OCR_BOX_FLAG_RIBBON)!=0u;
    if(!ribbon) {
        uint pad=((core.w-core.y)*V2_SUBTITLE_CONTOUR_ORDINARY_SEARCH_PAD_HEIGHT_NUMERATOR+
            V2_SUBTITLE_CONTOUR_ORDINARY_SEARCH_PAD_HEIGHT_DENOMINATOR-1u)/
            V2_SUBTITLE_CONTOUR_ORDINARY_SEARCH_PAD_HEIGHT_DENOMINATOR;
        search.y=max(search.y,core.y-min(pad,core.y));
    }
    return search;
}
bool RestoredColourRoot(uint root) {
    if(root==0u)return false;
    uint4 growth=Rect(8u);
    if(CollectOldSeedSearch.y<=growth.y)return false; // Ribbons and unchanged tops.
    uint index=root-1u;int2 up=int2(index%PShape.x,index/PShape.x)-int2(0,1);
    return !PixelInOldSeedSearch(up,CollectOldSeedSearch);
}
[numthreads(16,16,1)]
void collect_main(uint3 d:SV_DispatchThreadID,uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    if(lane==0u) {
        CollectReady=0u;CollectAnyRoot=0u;CollectOriginalBodyAvailable=0u;CollectOldSeedSearch=0u;
        CollectStyleReady=0u;CollectRepeatConfident=0u;
        CollectFLo=0;CollectFHi=0;CollectKLo=0;CollectKHi=0;
        if(MetaShape()) {
            uint n,s,nf,sf,bw,bh,cw,ch,sw,sh;
            GraphParent.GetDimensions(n,s);GraphFlags.GetDimensions(nf,sf);
            ModeBody.GetDimensions(bw,bh);ModeCover.GetDimensions(cw,ch);
            DebugBaselineBody.GetDimensions(sw,sh); // Current original Seeds alias.
            if(!PaletteReady(true)||!BaseProof())Fault(F_PALETTE);
            else if(n!=PShape.x*PShape.y+1u||s!=4u||nf!=n||sf!=4u||
                    any(uint2(bw,bh)!=PShape.xy)||any(uint2(cw,ch)!=PShape.xy)||
                    any(uint2(sw,sh)!=PShape.xy))Fault(F_SHAPE);
            else if(GraphParent[0u]!=0u)Fault(F_GRAPH);
            else {CollectReady=1u;CollectOldSeedSearch=OldSeedSearch();InitCollectOwnership();}
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // Z and readiness are uniform across each16x16x1 group. Partial native groups
    // participate in every barrier; only their final publication is clipped.
    if(group.z!=0u||CollectReady==0u)return;
    int2 origin=int2(group.xy*16u)-int2(COLLECT_HALO,COLLECT_HALO);
    uint anyRoot=0u,anyOriginalBody=0u;
    [loop]for(uint i=lane;i<COLLECT_PIXELS;i+=256u) {
        int2 p=origin+int2(i%COLLECT_TILE,i/COLLECT_TILE);
        uint root=0u,reach=0u,core=0u;float3 joinedRGB=0.0f;
        if(all(p>=0)&&all(p<int2(PShape.xy))) {
            uint body=ModeBody.Load(int3(p,0));
            if(body>1u)Fault(F_BODY_DOMAIN);
            if(body==1u) {
                anyOriginalBody=1u;
                uint current=FindStable((uint)p.y*PShape.x+(uint)p.x+1u);
                if(current!=0u&&(GraphFlags[current]&4u)!=0u) {
                    root=current;
                    uint seed=DebugBaselineBody.Load(int3(p,0));
                    if(seed>3u||seed==2u)Fault(F_BODY_DOMAIN);
                    else reach=(seed&1u)!=0u?COLLECT_REACH_SOURCE_SEED:0u;
                    // Only the NEW channel uses joined current RGB/core evidence.
                    // Old reach above remains independent of these predicates.
                    if(CurrentCollectCore((uint2)p)) {
                        float4 joined=CurrentRGB.Load(int3(p,0));
                        if(joined.a==1.0f && all(isfinite(joined.rgb)) &&
                           all(joined.rgb>=0.0f) && all(joined.rgb<=1.0f)) {
                            core=1u;joinedRGB=joined.rgb;
                            if(OwnershipStyleCode(PhysicalRGB8(joined.rgb)))core|=2u;
                            [branch]if(ModeCover.Load(int3(p,0))==1u && CurrentCoreOwnAdaptiveRGB8((uint2)p)) {
                                reach|=COLLECT_REACH_MODAL_SEED;
                                if((core&2u)!=0u)reach|=4u;
                            }
                        }
                    }
                }
            }
        }
        CollectRoot[i]=root;CollectReachA[i]=reach;CollectReachB[i]=0u;
        CollectRGB[i]=joinedRGB;CollectCore[i]=core;anyRoot|=root;
    }
    if(anyRoot!=0u){uint ignored;InterlockedOr(CollectAnyRoot,1u,ignored);}
    if(anyOriginalBody!=0u){uint ignored;InterlockedOr(CollectOriginalBodyAvailable,1u,ignored);}
    GroupMemoryBarrierWithGroupSync();
    // Every collect completes before a consumer reads this availability flag.
    if(lane==0u&&CollectOriginalBodyAvailable!=0u){uint ignored;InterlockedOr(Meta[UB(Hook.x)+31u],1u,ignored);}
    // Three synchronous N8 geodesic steps. Immutable original4CC roots prevent
    // diagonal shortcuts into another accepted component. The3pixel halo is exact
    // for all central16x16 outputs, including paths crossing workgroup boundaries.
    // The shared verdict is uniform. Empty/weak-only tiles leave initialized B0
    // and skip all reconstruction work, while publishing original AllBODY/Cover.
    if(CollectAnyRoot!=0u) {
    [unroll]for(uint step=0u;step<3u;++step) {
        bool fromA=(step&1u)==0u;
        [loop]for(uint i=lane;i<COLLECT_PIXELS;i+=256u) {
            uint root=CollectRoot[i];int2 p=int2(i%COLLECT_TILE,i/COLLECT_TILE);
            uint reach=fromA?CollectReachA[i]:CollectReachB[i];
            if(root!=0u) {
                [unroll]for(int y=-1;y<=1;++y)[unroll]for(int x=-1;x<=1;++x) {
                    int2 q=p+int2(x,y);
                    if(all(q>=0)&&all(q<int2(COLLECT_TILE,COLLECT_TILE))) {
                        uint at=(uint)q.y*COLLECT_TILE+(uint)q.x;
                        if(CollectRoot[at]==root) {
                            uint neighbour=fromA?CollectReachA[at]:CollectReachB[at];
                            reach|=neighbour&COLLECT_REACH_SOURCE_SEED; // Original any-seed channel: exact old N8 rule.
                            if((CollectCore[i]&1u)!=0u && (CollectCore[at]&1u)!=0u) {
                                float3 delta=abs(CollectRGB[i]-CollectRGB[at]);
                                if(max(delta.r,max(delta.g,delta.b))<=COLLECT_COHERENCE)reach|=neighbour&COLLECT_REACH_MODAL_SEED;
                            }
                            if((CollectCore[i]&3u)==3u && (CollectCore[at]&3u)==3u) {
                                reach|=neighbour&4u;
                            }
                        }
                    }
                }
            }
            else reach=0u;
            if(fromA)CollectReachB[i]=reach;else CollectReachA[i]=reach;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    }
    if(!Native(d.xy))return;
    uint body=ModeBody.Load(int3(d.xy,0)),cover=ModeCover.Load(int3(d.xy,0));
    if(body>1u||cover>1u){Fault(F_BODY_DOMAIN);return;}
    uint bit=1u<<Hook.x,ignored;
    if(cover==1u)InterlockedOr(CoverBitsWrite[d.xy],bit,ignored);
    // No direct LOW8/HIGH8, original BODY or physical-graph override is granted here.
    // Selected bits16..23 are phase-only offers, normalized after known physical closure.
    if(body==0u&&cover==1u&&Meta[9u]==1u&&Meta[4u]==1u&&Meta[5u]==1u&&
       Meta[UB(Hook.x)+1u]==1u&&Meta[PhysicalModelBank(Hook.x)+2u]==1u&&
       CurrentOwnTrustedFill(d.xy)&&UnseededColourRoot(d.y*PShape.x+d.x+1u)!=0u) {
        uint previous;
        InterlockedOr(SelectedBitsWrite[d.xy],bit<<SELECTED_UNSEEDED_SHIFT,previous);
        if(!SelectedHasUnseededOffer(previous,bit)&&AllBodyUnitWasUndiscovered(UnitBitsWrite[d.xy],bit))
            InterlockedAdd(Meta[UB(Hook.x)+4u],1u,ignored);
        // This offer never altered baseline BODY. Accepted closure raises existing Meta7;
        // rejected offers leave baseline fallback unchanged. Old top-restore Meta87 stays literal.
    }
    if(body==1u) {
        if(cover!=1u)Fault(F_BODY_DOMAIN);
        uint local=(d.y-group.y*16u+COLLECT_HALO)*COLLECT_TILE+
                   d.x-group.x*16u+COLLECT_HALO;
        uint component=CollectRoot[local];
        // Weak-only accepted colour components have no reconstruction-cache root.
        if(component==0u)component=FindStable(d.y*PShape.x+d.x+1u);
        bool restored=RestoredColourRoot(component);
        if(restored) {
            InterlockedOr(Meta[87u],bit,ignored); // Existing restored components request filtered BODY/BOX3 publication.
            if(component==0u||(GraphFlags[component]&4u)==0u||CollectRepeatConfident==0u||
               Meta[PhysicalModelBank(Hook.x)+2u]!=1u)return; // Restores never use UNKNOWN black fallback.
        }
        bool style=CurrentOwnStyle(d.xy);
        uint stylePotentialBit=bit<<ALL_BODY_STYLE_SHIFT,foregroundBit=bit<<SELECTED_FOREGROUND_SHIFT;
        // HIGH8 records an accepted OWN-MODE style potential, not geometry.
        if(!restored) {
            uint old;InterlockedOr(UnitBitsWrite[d.xy],bit|(style?stylePotentialBit:0u),old);
            if(AllBodyUnitWasUndiscovered(old,bit)&&!SelectedHasUnseededOffer(SelectedBitsWrite[d.xy],bit))
                InterlockedAdd(Meta[UB(Hook.x)+4u],1u,ignored);
        }
        bool preserve=(CollectReachB[local]&3u)!=0u;
        // Only residual current per-mode OriginalBODY enters the O(1) RGB test.
        // Existing selected evidence, outside-core behavior and BODY0 stay exact.
        bool directHold=false;
        // For KNOWN style, a modal hold is already inside Fbox at alpha1.
        // Re-evaluate it only when residual geometry or UNKNOWN style needs it.
        [branch]if((!preserve||CollectStyleReady==0u)&&cover==1u)directHold=CurrentCoreOwnAdaptiveRGB8(d.xy);
        preserve=preserve||directHold;
        // Unified positive foreground: existing trusted style evidence OR
        // existing coherent modal-seeded reach. Original any-source reach
        // alone is never foreground authority; LOW8 discovery stays exact.
        bool foreground=((CollectReachB[local]&4u)!=0u)||
            ((CollectReachB[local]&COLLECT_REACH_MODAL_SEED)!=0u)||directHold;
        if(restored) {
            uint pending=bit<<ALL_BODY_RESTORED_SHIFT;
            if(foreground||style)pending|=bit<<ALL_BODY_PENDING_FOREGROUND_SHIFT;
            uint old;InterlockedOr(UnitBitsWrite[d.xy],pending,old);
            if(AllBodyUnitWasUndiscovered(old,bit)&&!SelectedHasUnseededOffer(SelectedBitsWrite[d.xy],bit))
                InterlockedAdd(Meta[UB(Hook.x)+4u],1u,ignored);
            return; // No restored LOW8/HIGH8 authority before physical closure.
        }
        if(foreground)InterlockedOr(SelectedBitsWrite[d.xy],foregroundBit,ignored);
        if(preserve) {
            uint previous;InterlockedOr(SelectedBitsWrite[d.xy],bit,previous);
            if((previous&bit)==0u) {
                InterlockedAdd(Meta[UB(Hook.x)+5u],1u,ignored);
            }
        }
    }
}

[numthreads(16,16,1)]
void clear_scratch_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||!Native(d.xy))return;
    // Stage aliases: eligible,seeds,domain,cover. No unit bitplane is bound here.
    UnitBitsWrite[d.xy]=0u;CoverBitsWrite[d.xy]=0u;SelectedBitsWrite[d.xy]=0u;StageAux[d.xy]=0u;
}

// A provisional exact singleton does not change the palette, its original
// BODY, or the published physical model. Unit+2 is private phase state only:
// bit1 marks preparation; bit0 records any true-cover/native-boundary black;
// bit2 rejects OriginalBODY that is itself in the candidate black partition.
static const uint UNIFORM_BLACK_PREPARED=2u,UNIFORM_BLACK_BOUNDARY=1u,UNIFORM_BLACK_BODY=4u;
bool UniformBlackCandidate(uint unit) {
    uint s=PhysicalModelBank(unit);
    if(Meta[s+1u]!=1u||Meta[s+2u]!=0u||Meta[s+3u]!=5u)return false;
    uint n=Meta[s+4u],prototype=Meta[s+5u],xs=Meta[s+7u];
    uint3 p=PhysicalUnpack(prototype);
    return Meta[s]==PHYSICAL_MODEL_TAG&&n>=2u&&n<=512u&&Meta[s+6u]==n&&
        xs>=2u&&xs<=64u&&xs<=n&&Meta[s+8u]==xs&&(prototype&0xff000000u)==0u&&
        max(p.r,max(p.g,p.b))>1u&&
        all(uint4(Meta[s+9u],Meta[s+10u],Meta[s+11u],Meta[s+12u])==0u)&&
        Meta[s+13u]==512u&&Meta[s+14u]==0u&&Meta[s+15u]==0u;
}
bool PartitionBlackRGB(float3 rgb,uint unit,bool provisional) {
    if(!provisional)return PhysicalBlackRGB(rgb,unit);
    // Preserve the physical predicate's literal union. The exact candidate is
    // only a partition hypothesis until every original BODY peer is enclosed.
    return max(rgb.r,max(rgb.g,rgb.b))<=1.0f/255.0f||
        all(PhysicalRGB8(rgb)==PhysicalUnpack(Meta[PhysicalModelBank(unit)+5u]));
}
bool UniformBlackQualified(uint unit) {
    if(!UniformBlackCandidate(unit))return false;
    uint u=UB(unit);
    return Meta[3u]==0u&&Meta[9u]==1u&&Meta[4u]==1u&&Meta[5u]==1u&&
        Meta[u+1u]==1u&&Meta[u+29u]==1u&&Meta[u+2u]==UNIFORM_BLACK_PREPARED&&
        Meta[u+4u]>0u&&Meta[u+30u]==Meta[u+4u]&&Meta[u+3u]==Meta[u+30u];
}
bool TrueCoverBoundary(uint2 p,uint bit) {
    if(p.x==0u||p.y==0u||p.x+1u==PShape.x||p.y+1u==PShape.y)return true;
    return (AllCoverBits.Load(int3(p+uint2(1u,0u),0))&bit)==0u||
        (AllCoverBits.Load(int3(p-uint2(1u,0u),0))&bit)==0u||
        (AllCoverBits.Load(int3(p+uint2(0u,1u),0))&bit)==0u||
        (AllCoverBits.Load(int3(p-uint2(0u,1u),0))&bit)==0u;
}
// Known/literal behavior stays unchanged. An exact nonliteral singleton can
// run the existing graph provisionally, without granting refinement authority.
bool BlackRefinementReady(uint unit) {
    uint s=PhysicalModelBank(unit);
    if(Meta[s+2u]==1u)return true;
    uint3 p=PhysicalUnpack(Meta[s+5u]);
    return (Meta[s+3u]==5u&&max(p.r,max(p.g,p.b))<=1u&&
        2u*Meta[s+6u]>Meta[s+4u]&&Meta[s+8u]>=2u&&2u*Meta[s+8u]>Meta[s+7u])||
        UniformBlackCandidate(unit);
}

[numthreads(1,1,1)]
void prepare_unit_main(uint3 g:SV_GroupID) {
    if(any(g)||!MetaShape()||Hook.x>=8u)return;
    uint bytes;UnitArgs.GetDimensions(bytes);if(bytes!=64u){Fault(F_SHAPE);return;}
    UnitArgs.Store4(0u,0u);UnitArgs.Store4(16u,0u);UnitArgs.Store4(32u,0u);UnitArgs.Store4(48u,0u);
    uint u=UB(Hook.x);Store(u+2u,0u);
    uint4 clip=uint4(Meta[u+24u],Meta[u+25u],Meta[u+26u],Meta[u+27u]);
    if(Meta[u]==0u||Meta[u+4u]==0u)return;
    if(any(clip.xy>=clip.zw)||any(clip.zw>PShape.xy)){Fault(F_SHAPE);return;}
    uint2 groups=(clip.zw-clip.xy+15u)/16u;
    bool ready=Meta[3u]==0u&&Meta[9u]==1u&&Meta[4u]==1u&&Meta[5u]==1u&&Meta[u+1u]==1u&&BlackRefinementReady(Hook.x);
    if(ready&&UniformBlackCandidate(Hook.x))Store(u+2u,UNIFORM_BLACK_PREPARED);
    Store(u+29u,ready?1u:0u);
    UnitArgs.Store4(0u,uint4(groups,ready?1u:0u,0u));
    UnitArgs.Store4(16u,uint4(groups,1u,0u));
}

[numthreads(16,16,1)]
void emit_exterior_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||!MetaShape())return;uint2 p;
    bool inside=PixelFromROI(d.xy,p);if(!inside)return;
    uint u=UB(Hook.x);if(Meta[u+29u]!=1u)return;
    uint bit=1u<<Hook.x;
    bool cover=(AllCoverBits.Load(int3(p,0))&bit)!=0u;
    bool body=AllBodyHasNormalGeometry(AllBodyBits.Load(int3(p,0)),bit);
    float3 rgb=CurrentRGB.Load(int3(p,0)).rgb;
    bool provisional=(Meta[u+2u]&UNIFORM_BLACK_PREPARED)!=0u;
    bool black=PartitionBlackRGB(rgb,Hook.x,provisional);
    bool passable=cover&&(!black||body);
    // This deliberately rejects the whole candidate if ANY black component
    // reaches the true domain edge, including a disconnected background peer.
    if(provisional&&cover&&black&&TrueCoverBoundary(p,bit)) {
        uint ignored;InterlockedOr(Meta[u+2u],UNIFORM_BLACK_BOUNDARY,ignored);
    }
    int4 r=int4(Meta[u+12u],Meta[u+13u],Meta[u+14u],Meta[u+15u]);
    bool border=(int)p.x==r.x||(int)p.x==r.z-1||(int)p.y==r.y||(int)p.y==r.w-1;
    UnitBitsWrite[p]=passable?1u:0u;CoverBitsWrite[p]=passable&&border?1u:0u;
    SelectedBitsWrite[p]=cover?Hook.x+1u:0u;StageAux[p]=cover?1u:0u;
}
[numthreads(16,16,1)]
void resolve_exterior_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||!MetaShape())return;uint2 p;
    bool inside=PixelFromROI(d.xy,p);if(!inside)return;
    if(Meta[UB(Hook.x)+29u]!=1u)return;
    uint np,sp,nf,sf;GraphParent.GetDimensions(np,sp);GraphFlags.GetDimensions(nf,sf);
    if(np!=PShape.x*PShape.y+1u||sp!=4u||nf!=np||sf!=4u){Fault(F_SHAPE);StageAux[p]=0u;return;}
    if(GraphParent[0u]!=0u){Fault(F_GRAPH);StageAux[p]=0u;return;}
    uint value=0u;
    if((AllCoverBits.Load(int3(p,0))&(1u<<Hook.x))!=0u) {
        // Eligible scratch is stage t4 for this entry (not original/exterior ModeBody).
        if(ModeBody.Load(int3(p,0))==1u) {
            uint root=FindStable(p.y*PShape.x+p.x+1u);
            if(root!=0u)value=(GraphFlags[root]&1u)!=0u?1u:0u;
        }
    }
    StageAux[p]=value;
}

// This exact CBC physical component must be closed and carry current S.
// Original binary border seeds retain their meanings; Raw never binds S.
bool UnseededPhysicalSourceAvailable(uint2 p) {
    uint root=FindStable(p.y*PShape.x+p.x+1u);
    if(root==0u)return false;
    uint flags=GraphFlags[root];
    return (flags&3u)==0u&&(flags&4u)!=0u;
}

[numthreads(16,16,1)]
void count_enclosed_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||!MetaShape())return;uint2 p;
    bool inside=PixelFromROI(d.xy,p);if(!inside)return;
    uint u=UB(Hook.x),bit=1u<<Hook.x;
    if(Meta[u+29u]!=1u)return;
    uint word=AllBodyBits.Load(int3(p,0));
    bool normal=AllBodyHasNormalGeometry(word,bit),restored=AllBodyHasRestoredGeometry(word,bit);
    if(normal&&(Meta[u+2u]&UNIFORM_BLACK_PREPARED)!=0u&&
       PartitionBlackRGB(CurrentRGB.Load(int3(p,0)).rgb,Hook.x,true)) {
        uint ignored;InterlockedOr(Meta[u+2u],UNIFORM_BLACK_BODY,ignored);
    }
    bool enclosed=ModeBody.Load(int3(p,0))==0u;
    if(enclosed) {
        uint ignored;
        if(normal)InterlockedAdd(Meta[u+30u],1u,ignored); // OLD-only refinement trigger.
        if(normal||(restored&&Meta[PhysicalModelBank(Hook.x)+2u]==1u&&
            !PhysicalBlackRGB(CurrentRGB.Load(int3(p,0)).rgb,Hook.x)))
            InterlockedAdd(Meta[u+3u],1u,ignored); // Combined admitted closure candidates.
    }
}

[numthreads(16,16,1)]
void select_unit_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||!MetaShape())return;uint2 p;
    bool inside=PixelFromROI(d.xy,p);if(!inside)return;
    uint u=UB(Hook.x),bit=1u<<Hook.x;
    bool format=Meta[4u]==1u&&Meta[5u]==1u;
    // A missing preparation marker must not let an exact singleton take the
    // ordinary known-model route through otherwise nonzero graph counters.
    bool provisional=(Meta[u+2u]&UNIFORM_BLACK_PREPARED)!=0u||UniformBlackCandidate(Hook.x);
    bool qualified=provisional&&UniformBlackQualified(Hook.x);
    bool ready=Meta[3u]==0u&&Meta[9u]==1u&&format&&Meta[u+1u]==1u&&Meta[u+29u]==1u&&
        (!provisional||qualified);
    bool knownBlack=Meta[PhysicalModelBank(Hook.x)+2u]==1u||qualified;
    bool refine=ready&&Meta[u+3u]>0u,normalRefine=ready&&Meta[u+30u]>0u;
    if(all(d.xy==0u)) {
        uint status=Meta[3u]!=0u?5u:!format?4u:Meta[u+1u]==0u?3u:refine?1u:2u;
        Store(u+6u,status);uint ignored;InterlockedOr(Meta[refine?7u:8u],bit,ignored);
    }
    // AllBody t8 is unbound; temporary fields are accessed at this pixel only.
    uint word=UnitBitsWrite[p],foregroundBit=bit<<SELECTED_FOREGROUND_SHIFT;
    bool normal=AllBodyHasNormalGeometry(word,bit),oldPending=AllBodyHasRestoredGeometry(word,bit);
    bool newPending=SelectedHasUnseededOffer(SelectedBitsWrite[p],bit);
    bool exterior=ModeBody.Load(int3(p,0))!=0u;
    bool physical=ready&&knownBlack&&
        !PartitionBlackRGB(CurrentRGB.Load(int3(p,0)).rgb,Hook.x,provisional)&&!exterior;
    bool oldKeep=oldPending&&physical,newKeep=false;
    [branch]if(newPending&&physical&&Meta[u+31u]!=0u)newKeep=UnseededPhysicalSourceAvailable(p);
    bool keep=normal&&(!normalRefine||SelectedHasGeometry(SelectedBitsWrite[p],bit)||!exterior);
    if(!provisional&&normal&&normalRefine&&!exterior&&AllBodyHasStylePotential(word,bit)) {
        uint ignored;InterlockedOr(SelectedBitsWrite[p],foregroundBit,ignored);
    }
    // A provisional K4 partition does not grant a new foreground/style role.
    // Preserve only HIGH evidence already supplied independently upstream.
    [branch]if(normal&&ready&&!exterior&&Meta[PhysicalModelBank(Hook.x)+2u]==1u&&
               !SelectedHasForeground(SelectedBitsWrite[p],bit)) {
        if(!PartitionBlackRGB(CurrentRGB.Load(int3(p,0)).rgb,Hook.x,provisional)) {
            uint ignored;InterlockedOr(SelectedBitsWrite[p],foregroundBit,ignored);
        }
    }
    // Request BODY/BOX3 publication after admission. This is not permission to
    // change the old physical-refinement count, status or fallback authority.
    if(newKeep) { uint ignored;InterlockedOr(Meta[87u],bit,ignored); }
    if((oldKeep&&AllBodyHasPendingForeground(word,bit))||newKeep) {
        uint ignored;InterlockedOr(SelectedBitsWrite[p],foregroundBit,ignored);
    }
    if(keep||oldKeep||newKeep) {
        uint old;InterlockedOr(SelectedBitsWrite[p],bit,old);
        if((old&bit)==0u){uint ignored;InterlockedAdd(Meta[u+5u],1u,ignored);}
    }
    uint normalized=word&~((bit<<ALL_BODY_RESTORED_SHIFT)|(bit<<ALL_BODY_PENDING_FOREGROUND_SHIFT));
    if(oldKeep||newKeep) {
        normalized|=bit;
        if((oldKeep&&AllBodyHasPendingForeground(word,bit))||newKeep)normalized|=foregroundBit;
    }
    else if((oldPending||newPending)&&!normal) {
        normalized&=~foregroundBit;
        uint ignored;InterlockedAdd(Meta[u+4u],0xffffffffu,ignored);
    }
    UnitBitsWrite[p]=normalized;
    // Offset16 still executes when physical readiness is false: cleanup is unconditional.
    [branch]if(newPending) { uint ignored;InterlockedAnd(SelectedBitsWrite[p],~(bit<<SELECTED_UNSEEDED_SHIFT),ignored); }
}

[numthreads(16,16,1)]
void publish_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||!Native(d.xy)||!MetaShape())return;
    bool valid=Meta[3u]==0u&&Meta[9u]==1u;
    uint body=DebugBaselineBody.Load(int3(d.xy,0)),matte=DebugBaselineM.Load(int3(d.xy,0));
    if(valid&&(Meta[7u]|Meta[87u])!=0u) {
        uint bits=SelectedGeometryBits(SelectedBitsWrite[d.xy]);body=bits!=0u?1u:0u;matte=0u;
        uint cover=AllCoverBits.Load(int3(d.xy,0))&GEOMETRY_BITS;
        if(cover!=0u) {
            [loop]for(int y=-3;y<=3&&matte==0u;++y)[loop]for(int x=-3;x<=3&&matte==0u;++x) {
                int2 q=int2(d.xy)+int2(x,y);
                if(all(q>=0)&&all(q<int2(PShape.xy))) {
                    if((SelectedGeometryBits(SelectedBitsWrite[(uint2)q])&cover)!=0u)matte=1u;
                }
            }
        }
    }
    CandidateBody[d.xy]=body;CandidateM[d.xy]=matte;
    if(all(d.xy==0u)) {
        // All selectors have completed. No other thread in this publication
        // reads the model bank, so K4 cannot be observed partially by a consumer.
        [unroll]for(uint unit=0u;unit<8u;++unit) {
            uint u=UB(unit),s=PhysicalModelBank(unit);
            if((Meta[u+2u]&UNIFORM_BLACK_PREPARED)!=0u||UniformBlackCandidate(unit)) {
                bool promote=valid&&UniformBlackQualified(unit)&&Meta[u+6u]==1u&&
                    (Meta[7u]&(1u<<unit))!=0u;
                if(promote) {
                    uint n=Meta[s+4u],prototype=Meta[s+5u];uint3 p=PhysicalUnpack(prototype);
                    Store(s+1u,4u);Store(s+2u,1u);Store(s+3u,1u);
                    Store(s+9u,n);Store(s+10u,max(p.r,max(p.g,p.b)));
                    Store(s+11u,prototype);Store(s+12u,prototype);
                } else {
                    // A valid geometric rejection publishes the same UNKNOWN
                    // bank/readiness/counts as the original skipped graph.
                    Store(u+29u,0u);Store(u+3u,0u);Store(u+30u,0u);
                }
            }
            Store(u+2u,0u);
        }
        // Every select has completed; clear phase availability before CBC512 publication.
        [unroll]for(uint unit=0u;unit<8u;++unit)Store(UB(unit)+31u,0u);
        Store(2u,valid?1u:0u);Store(10u,!valid?0u:(Meta[7u]|Meta[87u])!=0u?2u:1u);
    }
}
