// Fresh EMPTY parks only reduced producer work. The actual transaction Root remains the publisher authority.
#include "include/depth_coordinate_v2_contract.generated.hlsl"

cbuffer ActualContour : register(b0) {
    uint4 Source; uint4 Field; uint4 Content; uint4 Model; uint4 Observation; uint4 Options;
};
cbuffer ExpectedRequest : register(b2) {
    uint4 Native_Mode_Work; // native width, band height, mode (1,2,4), requested work
    uint4 Owner;            // frame lo/hi, analysis generation lo/hi
    uint4 ClockToken;       // clock lo/hi, current root token lo/hi
};
ByteAddressBuffer ActualRoot : register(t0);
StructuredBuffer<uint> OCR : register(t1);
StructuredBuffer<uint> SCT : register(t2);
StructuredBuffer<uint> SCP : register(t3);
StructuredBuffer<uint> UIP : register(t4);

RWByteAddressBuffer ProducerSchedule : register(u0);
RWStructuredBuffer<uint> Witness : register(u1);

static const uint EmptyTag=0x31504d45u, EmptySchema=1u;
static const uint PassThrough=0u, FreshEmpty=1u, RootHeld=2u;
static const uint OocrMarker=0x52434f4fu;

uint CBWord(uint i) {
    const uint4 words[6]={Source,Field,Content,Model,Observation,Options};
    return words[i/4u][i%4u];
}

bool RootCurrent() {
    uint bytes; ActualRoot.GetDimensions(bytes);
    if (bytes!=256u||Native_Mode_Work.w!=1u||
        (Native_Mode_Work.z!=1u&&Native_Mode_Work.z!=2u&&Native_Mode_Work.z!=4u)||
        all(ClockToken.zw==0u)) return false;
    const uint4 one=uint4(1u,1u,1u,0u);
    const uint2 token=ClockToken.zw, cookie=token^uint2(0xa3756c91u,0x5c8a936eu);
    return all(ActualRoot.Load4(0u)==uint4(1u,1u^0xd1ec15a5u^OocrMarker,token))&&
        all(ActualRoot.Load4(16u)==uint4(cookie,0x47524243u,OocrMarker))&&
        all(ActualRoot.Load4(32u)==uint4(token,cookie))&&
        all(ActualRoot.Load4(48u)==uint4(0x54535152u,1u,1u^0x6f435257u,0u))&&
        all(ActualRoot.Load4(80u)==(Native_Mode_Work.z==2u?uint4(0u,0u,0u,0u):one))&&
        all(ActualRoot.Load4(176u)==uint4((Field.xy+15u)/16u,1u,0u))&&
        all(ActualRoot.Load4(208u)==one)&&all(ActualRoot.Load4(224u)==uint4(0u,1u,1u,0u))&&
        all(ActualRoot.Load4(240u)==one);
}

bool ContourCurrentEmpty() {
    if (any(Source.xw!=Native_Mode_Work.xy)||any(Source.xy==0u)||
        any(Source.xy>V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT)||
        any(Field.xy==0u)||any(Field.xy>V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT)||
        any(Content.xy>=Content.zw)||any(Content.zw>Field.xy)||
        Field.z>=Field.w||Field.z<Content.y||Field.w>Content.w||
        any(Model!=Owner)||all(Owner.xy==0u)||any(Observation.xy!=ClockToken.xy)||
        all(ClockToken.xy==0u)||Observation.z>1u||Observation.w!=0u||
        any(Options!=uint4(1u,V2_SUBTITLE_CONTOUR_RECIPE,0u,0u))) return false;
    uint top,height,start,end;
    if (!V2SubtitleOcrComputeCrop(Source.x,Source.y,top,height)||
        any(Source.zw!=uint2(top,height))||
        !V2SubtitleOcrProjectContentRowCeil(Source.x,Source.y,Field.x,Field.y,Content,V2_OCR_SAFE_ROW_TOP,start)||
        !V2SubtitleOcrProjectContentRowCeil(Source.x,Source.y,Field.x,Field.y,Content,V2_OCR_SAFE_ROW_BOTTOM,end)||
        any(Field.zw!=uint2(start,end))) return false;

    uint count,stride; SCT.GetDimensions(count,stride);
    if (count!=V2_SUBTITLE_CONTOUR_HISTORY_WORD_COUNT||stride!=4u||
        any(uint4(SCT[0u],SCT[1u],SCT[2u],SCT[3u])!=
             uint4(V2_SUBTITLE_CONTOUR_SCHEMA,V2_SUBTITLE_CONTOUR_HISTORY_TAG,0u,0u))||
        any(uint4(SCT[20u],SCT[21u],SCT[22u],SCT[23u])!=uint4(Observation.xy,V2_SUBTITLE_CONTOUR_RECIPE,1u)))
        return false;
    [unroll] for (uint i=0u;i<4u;++i)
        if (SCT[4u+i]!=Model[i]||SCT[8u+i]!=Source[i]||SCT[12u+i]!=Field[i]||SCT[16u+i]!=Content[i]) return false;
    [unroll] for (uint j=24u;j<32u;++j) if (SCT[j]!=0u) return false;
    return true;
}

bool EmptyDetectorCurrent() {
    uint n,s; OCR.GetDimensions(n,s);
    if (n!=V2_OCR_RECORD_WORD_COUNT||s!=4u||
        any(uint4(OCR[0u],OCR[1u],OCR[2u],OCR[3u])!=uint4(V2_OCR_RECORD_SCHEMA,V2_OCR_RECORD_TAG,1u,0u))||
        OCR[4u]!=0u||OCR[15u]!=0u||
        any(uint4(OCR[5u],OCR[6u],OCR[7u],OCR[8u])!=Model)||
        any(uint2(OCR[9u],OCR[10u])!=Source.xy)||
        any(uint4(OCR[11u],OCR[12u],OCR[13u],OCR[14u])!=Field)) return false;
    // The final 48 words are separate classifier history and can be nonzero on a valid empty.
    [loop] for (uint i=V2_OCR_RAW_BOX_OFFSET;i<V2_OCR_CURRENT_PREFIX_WORD_COUNT;++i)
        if (OCR[i]!=0u) return false;
    return true;
}

bool EmptyConditionCurrent() {
    uint n,s; SCP.GetDimensions(n,s);
    if (n!=V2_SUBTITLE_CONDITION_PARAM_WORD_COUNT||s!=4u||
        any(uint4(SCP[0u],SCP[1u],SCP[2u],SCP[3u])!=
             uint4(V2_SUBTITLE_CONDITION_PARAM_SCHEMA,V2_SUBTITLE_CONDITION_PARAM_TAG,0u,0u))||
        any(uint2(SCP[4u],SCP[5u])!=0u)) return false;
    UIP.GetDimensions(n,s);
    if (n!=V2_SUBTITLE_PLANE_STATE_WORD_COUNT||s!=4u||
        any(uint4(UIP[0u],UIP[1u],UIP[2u],UIP[3u])!=
            uint4(V2_SUBTITLE_PLANE_STATE_SCHEMA,V2_SUBTITLE_PLANE_STATE_TAG,V2_SUBTITLE_PLANE_CURRENT_VALID_FLAG,0u))||
        any(uint4(UIP[4u],UIP[5u],UIP[6u],UIP[7u])!=Model)||
        any(uint4(UIP[8u],UIP[9u],UIP[10u],UIP[11u])!=uint4(Observation.xy,Source.xy))||
        any(uint4(UIP[12u],UIP[13u],UIP[14u],UIP[15u])!=Field)||
        any(uint4(UIP[16u],UIP[17u],UIP[18u],UIP[19u])!=Content)||
        any(uint4(UIP[20u],UIP[21u],UIP[22u],UIP[23u])!=0u)) return false;
    return true;
}

[numthreads(1,1,1)]
void prepare_empty_main() {
    uint rootBytes, scheduleBytes, n, s;
    ActualRoot.GetDimensions(rootBytes);
    ProducerSchedule.GetDimensions(scheduleBytes);
    Witness.GetDimensions(n,s);
    if (rootBytes!=256u||scheduleBytes!=256u||n!=32u||s!=4u) return;
    [unroll] for (uint i=0u;i<256u;i+=16u) ProducerSchedule.Store4(i,ActualRoot.Load4(i));
    Witness[0u]=EmptyTag; Witness[1u]=EmptySchema; Witness[2u]=PassThrough; Witness[3u]=0u;
    [unroll] for (uint j=4u;j<32u;++j) Witness[j]=0u;
    if (any(ActualRoot.Load3(240u)==0u)) { Witness[2u]=RootHeld; return; }
    if (!RootCurrent()||!ContourCurrentEmpty()||!EmptyDetectorCurrent()||!EmptyConditionCurrent()) return;

    // Change only this private scheduling twin. Consumers still bind the real Root.
    ProducerSchedule.Store(240u,0u);
    [loop] for (uint k=0u;k<24u;++k) Witness[4u+k]=CBWord(k);
    Witness[28u]=ClockToken.z; Witness[29u]=ClockToken.w;
    Witness[30u]=Native_Mode_Work.z; Witness[31u]=Native_Mode_Work.w;
    Witness[2u]=FreshEmpty;
}
