// PRIVATE recipe5. Already-display-referred actual FP16 snapshot, no second transfer.
// Lower-band availability only. No OCR/NCHW/P/UIP/EMPTY or renderer publication.
cbuffer ActualContourCB : register(b0) {
    uint4 CSource;uint4 CField;uint4 CContent;
    uint4 CFrame;uint4 CObservation;uint4 COptions;
};
cbuffer Request : register(b1) {
    uint4 Control;   // CGS5, schema1, recipe5, actual parent transfer mode0/1/2.
    uint4 Shape;     // Full ROI W/H, cropTop, bandH. Lower band ends at fullH.
    uint4 ROI;       // Actual pending parent raw-source L/T/R/B.
    uint4 Frame;     // Pending frame64, observation64.
    uint4 Identity;  // Pending generation64, actual transaction token64.
    uint4 RawShape;  // Actual parent raw W/H, view format, storage format.
    uint4 RawStorage;// Actual raw bind flags, SRV dimension, mips, array size.
    uint4 Sample;    // Actual raw samples/quality, actual snapshot view/storage format.
    uint4 Reserved;
};
Texture2D<float4> ActualDisplaySnapshot : register(t0);
ByteAddressBuffer ActualFinal : register(t1);
StructuredBuffer<uint> ActualSCT : register(t2);
StructuredBuffer<uint> ActualUIP : register(t3);
StructuredBuffer<uint> ActualProof : register(t4);
Texture2D<float4> ActualParentRaw : register(t5); // Dimensions only; no source-pixel verdict.
RWTexture2D<float4> ReferenceRGB8 : register(u0); // Full ROI carrier; ONLY lower band is written.
RWTexture2D<float4> AnalysisBand : register(u1); // Native band, explicit RTZ-half RGB8/255.
globallycoherent RWStructuredBuffer<uint> SourceProof : register(u2);
RWByteAddressBuffer CopyArguments : register(u3);
RWStructuredBuffer<uint> UIPEnvelope : register(u4);

static const uint TAG=0x35534743u,SCHEMA=1u,RECIPE=5u,FEATURES=63u;
static const uint F_JOIN=1u,F_COVERAGE=2u,F_INVALID=4u;

bool Has64(uint2 v) { return any(v!=0u); }
bool CurrentInputsKnown() {
    if(any(Control.xyz!=uint3(TAG,SCHEMA,RECIPE))||Control.w>2u||any(Reserved))return false;
    if(any(Shape.xy==0u)||any(Shape.xy>5120u)||Shape.z>=Shape.y||Shape.w!=Shape.y-Shape.z)return false;
    if(any(Shape!=CSource)||any(CField.xy==0u)||any(CField.xy>5120u)||CField.z>=CField.w||CField.w>CField.y)return false;
    if(any(CContent.xy>=CContent.zw)||any(CContent.zw>CField.xy))return false;
    if(any(COptions!=uint4(1u,27u,0u,0u))||CObservation.z>1u||CObservation.w!=0u)return false;
    if(!Has64(Frame.xy)||!Has64(Frame.zw)||!Has64(Identity.zw))return false;
    if(any(Frame!=uint4(CFrame.xy,CObservation.xy))||any(Identity.xy!=CFrame.zw))return false;
    uint sw,sh,rw,rh;ActualDisplaySnapshot.GetDimensions(sw,sh);ActualParentRaw.GetDimensions(rw,rh);
    if(any(uint2(sw,sh)!=Shape.xw)||any(uint2(rw,rh)!=RawShape.xy))return false;
    if(any(RawShape.xy==0u)||any(RawShape.zw==0u)||RawStorage.y!=4u||(RawStorage.x&8u)==0u)return false;
    if(any(RawStorage.zw!=1u)||any(Sample!=uint4(1u,0u,10u,10u)))return false;
    if(any(ROI.xy>=ROI.zw)||any(ROI.zw>RawShape.xy)||any(ROI.zw-ROI.xy!=Shape.xy))return false;
    uint n,s;ActualSCT.GetDimensions(n,s);if(n!=32u||s!=4u)return false;
    if(ActualSCT[0u]!=5u||ActualSCT[1u]!=0x35544353u||ActualSCT[2u]>1u||ActualSCT[22u]!=27u||ActualSCT[23u]!=1u)return false;
    [unroll]for(uint i=0u;i<4u;++i) {
        if(ActualSCT[4u+i]!=CFrame[i]||ActualSCT[8u+i]!=CSource[i]||
           ActualSCT[12u+i]!=CField[i]||ActualSCT[16u+i]!=CContent[i])return false;
    }
    if(any(uint2(ActualSCT[20u],ActualSCT[21u])!=CObservation.xy))return false;
    [unroll]for(uint j=24u;j<32u;++j)if(ActualSCT[j]!=0u)return false;
    uint bytes;ActualFinal.GetDimensions(bytes);if(bytes!=256u)return false;
    uint2 cookies=Identity.zw^uint2(0xa3756c91u,0x5c8a936eu);
    // A fresh real snapshot requires the actual optional-preprocess/OOCR receipt.
    // work2 abstention has OCR.valid0 and cannot legitimately publish SCT23=1.
    uint work=ActualFinal.Load(52u),marker=0x52434f4fu;
    if(work!=1u)return false;
    if(any(ActualFinal.Load4(0u)!=uint4(1u,1u^0xd1ec15a5u^marker,Identity.zw)))return false;
    if(any(ActualFinal.Load4(16u)!=uint4(cookies,0x47524243u,marker)))return false;
    if(any(ActualFinal.Load4(32u)!=uint4(Identity.zw,cookies)))return false;
    if(any(ActualFinal.Load4(48u)!=uint4(0x54535152u,work,work^0x6f435257u,0u)))return false;
    // Source availability is mode-agnostic: completed direct mode2 keeps infer80 zero;
    // authorized/queued modes publish1110. Downstream CBC independently joins its mode.
    uint4 infer=ActualFinal.Load4(80u);
    if(any(infer!=0u)&&any(infer!=uint4(1u,1u,1u,0u)))return false;
    if(any(ActualFinal.Load4(176u)!=uint4((CField.xy+15u)/16u,1u,0u)))return false;
    if(any(ActualFinal.Load4(208u)!=uint4(1u,1u,1u,0u)))return false;
    if(any(ActualFinal.Load4(224u)!=uint4(0u,1u,1u,0u)))return false;
    return all(ActualFinal.Load4(240u)==uint4(1u,1u,1u,0u));
}
bool SourceOutputsKnown() {
    uint rw,rh,bw,bh,n,s;
    ReferenceRGB8.GetDimensions(rw,rh);AnalysisBand.GetDimensions(bw,bh);SourceProof.GetDimensions(n,s);
    return all(uint2(rw,rh)==Shape.xy)&&all(uint2(bw,bh)==Shape.xw)&&n==64u&&s==4u;
}
uint ExpectedPixels() { return Shape.x*Shape.w; }
uint ExpectedGroups() { return ((Shape.x+15u)/16u)*((Shape.w+15u)/16u); }

[numthreads(1,1,1)]
void prepare_main(uint3 g:SV_GroupID) {
    if(any(g))return;
    uint bytes,n,s;CopyArguments.GetDimensions(bytes);SourceProof.GetDimensions(n,s);
    if(bytes==16u)CopyArguments.Store4(0u,uint4(1u,1u,0u,0u));
    ActualFinal.GetDimensions(bytes);
    // Actual root0 is commonly (0,1,1), not Z0. Hold every published output/proof.
    if(bytes==256u&&any(ActualFinal.Load3(240u)==0u))return;
    if(n!=64u||s!=4u){if(n>0u)SourceProof[0u]=0u;return;}
    [unroll]for(uint i=0u;i<64u;++i)SourceProof[i]=0u;
    SourceProof[1u]=SCHEMA;SourceProof[2u]=RECIPE;
    [unroll]for(uint j=0u;j<4u;++j) {
        SourceProof[4u+j]=Frame[j];SourceProof[8u+j]=Identity[j];
        SourceProof[12u+j]=CSource[j];SourceProof[16u+j]=ROI[j];
        SourceProof[20u+j]=RawShape[j];SourceProof[24u+j]=RawStorage[j];
        SourceProof[40u+j]=CField[j];SourceProof[44u+j]=CContent[j];
        SourceProof[48u+j]=COptions[j];SourceProof[52u+j]=CObservation[j];
        SourceProof[56u+j]=bytes==256u?ActualFinal.Load(240u+4u*j):0u;
    }
    SourceProof[28u]=Sample.x;SourceProof[29u]=Sample.y;SourceProof[30u]=Control.w;
    SourceProof[31u]=FEATURES;SourceProof[32u]=ExpectedPixels();SourceProof[36u]=ExpectedGroups();
    uint sn,ss;ActualSCT.GetDimensions(sn,ss);SourceProof[39u]=sn==32u&&ss==4u?ActualSCT[3u]:0u;
    SourceProof[60u]=10u;SourceProof[61u]=28u;SourceProof[62u]=10u;
    SourceProof[63u]=sn==32u&&ss==4u?ActualSCT[23u]:0u;
    CopyArguments.GetDimensions(bytes);
    if(bytes!=16u||!CurrentInputsKnown()||!SourceOutputsKnown()){SourceProof[38u]=F_JOIN;return;}
    CopyArguments.Store4(0u,uint4((Shape.x+15u)/16u,(Shape.w+15u)/16u,1u,0u));
}

uint RTZHalfCode(uint code) {
    // Exact floor-half of code/255. For1..254, e=highestbit(code)-8; all are normal.
    // significand=floor(code*2^(10-e)/255). Integer numerator never exceeds520192.
    if(code==0u)return 0u;if(code==255u)return 0x3c00u;
    uint high=(uint)firstbithigh(code);
    uint significand=(code<<(18u-high))/255u;
    return ((high+7u)<<10u)|(significand-1024u);
}
[numthreads(16,16,1)]
void copy_main(uint3 group:SV_GroupID,uint3 id:SV_DispatchThreadID,uint lane:SV_GroupIndex) {
    bool inside=id.z==0u&&all(id.xy<Shape.xw);
    if(inside) {
        float4 snapshot=ActualDisplaySnapshot.Load(int3(id.xy,0));
        bool valid=snapshot.a==1.0f&&all(isfinite(snapshot))&&all(snapshot.rgb>=0.0f)&&all(snapshot.rgb<=1.0f);
        float4 reference=0.0f,band=0.0f;
        if(valid) {
            uint3 code=(uint3)floor(snapshot.rgb*255.0f+0.5f);
            reference=float4(float3(code)/255.0f,1.0f);
            band=float4(f16tof32(RTZHalfCode(code.r)),f16tof32(RTZHalfCode(code.g)),f16tof32(RTZHalfCode(code.b)),1.0f);
        } else {
            uint ignored;InterlockedAdd(SourceProof[34u],1u,ignored);
        }
        ReferenceRGB8[uint2(id.x,Shape.z+id.y)]=reference;
        AnalysisBand[id.xy]=band;
    }
    // No early return or varying barrier flow. Coverage is independent of source validity.
    DeviceMemoryBarrierWithGroupSync();
    if(lane==0u) {
        int2 remaining=int2(Shape.xw)-int2(group.xy*16u);
        uint2 extent=uint2(clamp(remaining,int2(0,0),int2(16,16)));
        uint written=group.z==0u?extent.x*extent.y:0u,ignored;
        InterlockedAdd(SourceProof[33u],written,ignored);
        InterlockedAdd(SourceProof[37u],1u,ignored);
    }
}

[numthreads(1,1,1)]
void finalize_main(uint3 g:SV_GroupID) {
    if(any(g))return;
    uint n,s;SourceProof.GetDimensions(n,s);if(n!=64u||s!=4u)return;
    uint written=SourceProof[33u],invalid=SourceProof[34u];
    SourceProof[35u]=invalid<=written?written-invalid:0u;
    uint fault=SourceProof[38u];
    if(SourceProof[1u]!=SCHEMA||SourceProof[2u]!=RECIPE||!CurrentInputsKnown()||!SourceOutputsKnown())fault|=F_JOIN;
    if(SourceProof[32u]!=ExpectedPixels()||written!=ExpectedPixels()||invalid>written||
       SourceProof[36u]!=ExpectedGroups()||SourceProof[37u]!=ExpectedGroups())fault|=F_COVERAGE;
    if(invalid!=0u)fault|=F_INVALID;
    SourceProof[38u]=fault;
    if(fault!=0u||SourceProof[35u]!=ExpectedPixels()){SourceProof[0u]=0u;SourceProof[3u]=0u;return;}
    DeviceMemoryBarrier();SourceProof[3u]=1u;
    DeviceMemoryBarrier();SourceProof[0u]=TAG; // Source availability only, tag LAST.
}

bool AcceptedPendingJoin() {
    if(!CurrentInputsKnown())return false;
    uint n,s;ActualProof.GetDimensions(n,s);if(n!=64u||s!=4u)return false;
    if(any(uint4(ActualProof[0u],ActualProof[1u],ActualProof[2u],ActualProof[3u])!=uint4(TAG,SCHEMA,RECIPE,1u)))return false;
    [unroll]for(uint i=0u;i<4u;++i) {
        if(ActualProof[4u+i]!=Frame[i]||ActualProof[8u+i]!=Identity[i]||ActualProof[12u+i]!=CSource[i]||
           ActualProof[16u+i]!=ROI[i]||ActualProof[20u+i]!=RawShape[i]||ActualProof[24u+i]!=RawStorage[i]||
           ActualProof[40u+i]!=CField[i]||ActualProof[44u+i]!=CContent[i]||ActualProof[48u+i]!=COptions[i]||
           ActualProof[52u+i]!=CObservation[i]||ActualProof[56u+i]!=ActualFinal.Load(240u+4u*i))return false;
    }
    if(any(uint4(ActualProof[28u],ActualProof[29u],ActualProof[30u],ActualProof[31u])!=uint4(Sample.xy,Control.w,FEATURES)))return false;
    if(any(uint4(ActualProof[32u],ActualProof[33u],ActualProof[34u],ActualProof[35u])!=uint4(ExpectedPixels(),ExpectedPixels(),0u,ExpectedPixels())))return false;
    if(any(uint4(ActualProof[36u],ActualProof[37u],ActualProof[38u],ActualProof[39u])!=uint4(ExpectedGroups(),ExpectedGroups(),0u,ActualSCT[3u])))return false;
    if(any(uint4(ActualProof[60u],ActualProof[61u],ActualProof[62u],ActualProof[63u])!=uint4(10u,28u,10u,1u)))return false;
    ActualUIP.GetDimensions(n,s);if(n!=40u||s!=4u)return false;
    if(any(uint3(ActualUIP[0u],ActualUIP[1u],ActualUIP[2u])!=uint3(4u,0x34504955u,1u))||ActualUIP[3u]>255u||ActualUIP[22u]!=0u||ActualUIP[23u]!=0u)return false;
    return all(uint4(ActualUIP[4u],ActualUIP[5u],ActualUIP[6u],ActualUIP[7u])==CFrame)&&
           all(uint2(ActualUIP[8u],ActualUIP[9u])==CObservation.xy)&&all(uint2(ActualUIP[10u],ActualUIP[11u])==CSource.xy)&&
           all(uint4(ActualUIP[12u],ActualUIP[13u],ActualUIP[14u],ActualUIP[15u])==CField)&&
           all(uint4(ActualUIP[16u],ActualUIP[17u],ActualUIP[18u],ActualUIP[19u])==CContent)&&ActualUIP[20u]==ActualSCT[3u];
}
[numthreads(128,1,1)]
void envelope_main(uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    if(any(group))return;
    uint n,s,m,t,o,k;ActualUIP.GetDimensions(n,s);ActualProof.GetDimensions(m,t);UIPEnvelope.GetDimensions(o,k);
    if(n!=40u||s!=4u||m!=64u||t!=4u||o!=104u||k!=4u)return;
    if(lane<40u)UIPEnvelope[lane]=ActualUIP[lane];
    else if(lane<104u) {
        uint value=ActualProof[lane-40u];
        if(lane==43u&&!AcceptedPendingJoin())value=0u;
        UIPEnvelope[lane]=value;
    }
}
