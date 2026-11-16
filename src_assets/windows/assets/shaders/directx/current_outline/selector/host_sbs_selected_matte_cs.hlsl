// PRIVATE selected-matte publication, algorithm version1. Not an EMPTY/PRESENT classifier.
// Dispatch Z is copied from the actual observationOne.X. Root0 executes no threads and
// preserves this independently allocated selected tuple across temporal epoch replacement.
cbuffer ActualContour : register(b0) {
    uint4 Source; uint4 Field; uint4 Content; uint4 Model; uint4 Observation; uint4 Options;
};
cbuffer SelectionRequest : register(b1) {
    uint4 Request; // typed postprocess class1/2/4, expected work0/1/2, temporal supplied, version1
    uint4 Token;   // actual transaction token lo/hi,0,0
};
Texture2D<uint> LegacyM : register(t0);
Texture2D<uint> BaselineM : register(t1);
Texture2D<uint> TemporalM : register(t2);
StructuredBuffer<uint> NumericProof : register(t3);
ByteAddressBuffer TemporalReceipt : register(t4);
StructuredBuffer<uint> SCT : register(t5);
ByteAddressBuffer Root : register(t6);
StructuredBuffer<uint> CurrentFrame : register(t7);
StructuredBuffer<uint> SourceEnvelope : register(t8);
StructuredBuffer<uint> CBCMeta : register(t9);
StructuredBuffer<uint> RawMeta : register(t10);
ByteAddressBuffer F2Data : register(t11);
Texture2D<uint> CurrentOutlineM : register(t12);
RWTexture2D<uint> SelectedM : register(u0);
RWStructuredBuffer<uint> SelectedReceipt : register(u1);
RWByteAddressBuffer NativeArguments : register(u2);

uint CBWord(uint i) {
    if(i==0u)return Source.x;if(i==1u)return Source.y;if(i==2u)return Source.z;if(i==3u)return Source.w;
    if(i==4u)return Field.x;if(i==5u)return Field.y;if(i==6u)return Field.z;if(i==7u)return Field.w;
    if(i==8u)return Content.x;if(i==9u)return Content.y;if(i==10u)return Content.z;if(i==11u)return Content.w;
    if(i==12u)return Model.x;if(i==13u)return Model.y;if(i==14u)return Model.z;if(i==15u)return Model.w;
    if(i==16u)return Observation.x;if(i==17u)return Observation.y;if(i==18u)return Observation.z;if(i==19u)return Observation.w;
    if(i==20u)return Options.x;if(i==21u)return Options.y;if(i==22u)return Options.z;return Options.w;
}
bool SizeMatches(Texture2D<uint> image) {uint w,h;image.GetDimensions(w,h);return w==Source.x&&h==Source.w;}
bool RootTyped() {
    uint n;Root.GetDimensions(n);if(n!=256u||all(Token.xy==0u)||any(Token.zw)||(Request.w!=1u&&Request.w!=2u)||Request.z>1u||Request.y>2u)return false;
    if(Request.x!=1u&&Request.x!=2u&&Request.x!=4u)return false;
    uint2 cookie=Token.xy^uint2(0xa3756c91u,0x5c8a936eu);
    uint marker=Root.Load(28u);
    if(marker!=0u&&(marker!=0x52434f4fu||Request.y!=1u))return false;
    if(any(Root.Load4(0u)!=uint4(1u,1u^0xd1ec15a5u^marker,Token.xy))||
       any(Root.Load4(16u)!=uint4(cookie,0x47524243u,marker))||
       any(Root.Load4(32u)!=uint4(Token.xy,cookie))||
       any(Root.Load4(48u)!=uint4(0x54535152u,Request.y,Request.y!=0u?(Request.y^0x6f435257u):0u,0u)))return false;
    // FORCE2 uses host-direct depth dispatches. It must NOT be required to carry infer80=1110.
    if(Request.x==2u){if(any(Root.Load4(80u)!=0u))return false;}
    else {if(any(Root.Load4(80u)!=uint4(1u,1u,1u,0u)))return false;}
    if(any(Root.Load4(240u)!=uint4(1u,1u,1u,0u)))return false;
    if(marker==0x52434f4fu&&any(Root.Load4(208u)!=uint4(1u,1u,1u,0u)))return false;
    if(marker==0u&&any(Root.Load4(224u)!=uint4(1u,1u,1u,0u)))return false;
    return true;
}
bool LegacyContextTyped() {
    uint n,s;SelectedReceipt.GetDimensions(n,s);if(n!=64u||s!=4u)return false;
    SCT.GetDimensions(n,s);if(n!=32u||s!=4u)return false;
    uint w,h;SelectedM.GetDimensions(w,h);
    if(any(Source.xy==0u)||Source.x>5120u||Source.y>5120u||Source.z>Source.y||Source.w==0u||Source.w>Source.y-Source.z||
       w!=Source.x||h!=Source.w||!SizeMatches(LegacyM))return false;
    if(any(Field.xy==0u)||Field.z>=Field.w||Field.w>Field.y||any(Content.xy>=Content.zw)||any(Content.zw>Field.xy)||
       all(Model.xy==0u)||all(Observation.xy==0u)||Observation.z>1u||Observation.w!=0u||any(Options!=uint4(1u,27u,0u,0u)))return false;
    // snapshot-valid is independent of active CurrentGate. Authenticated current empty may
    // have SCT2=0/SCT23=1; the existing SCP/UI gate (not this receipt) controls presentation.
    if(SCT[0u]!=5u||SCT[1u]!=0x35544353u||SCT[2u]>1u||SCT[22u]!=27u||SCT[23u]!=1u)return false;
    if(any(uint4(SCT[4u],SCT[5u],SCT[6u],SCT[7u])!=Model)||
       any(uint4(SCT[8u],SCT[9u],SCT[10u],SCT[11u])!=Source)||
       any(uint4(SCT[12u],SCT[13u],SCT[14u],SCT[15u])!=Field)||
       any(uint4(SCT[16u],SCT[17u],SCT[18u],SCT[19u])!=Content)||
       any(uint2(SCT[20u],SCT[21u])!=Observation.xy))return false;
    [unroll]for(uint i=24u;i<32u;++i)if(SCT[i]!=0u)return false;
    return true;
}
// Current authority is the legacy context proof plus the typed transaction root.
bool CurrentTyped() { return LegacyContextTyped() && RootTyped(); }

bool BaselineTyped() {
    uint n,s;NumericProof.GetDimensions(n,s);if(n!=32u||s!=4u||!SizeMatches(BaselineM))return false;
    if(any(uint4(NumericProof[0],NumericProof[1],NumericProof[2],NumericProof[3])!=uint4(0x31564c49u,1u,1u,0u)))return false;
    [loop]for(uint i=0u;i<24u;++i)if(NumericProof[4u+i]!=CBWord(i))return false;
    [unroll]for(uint k=28u;k<32u;++k)if(NumericProof[k]!=0u)return false;
    return true;
}
bool TemporalTyped() {
    if(Request.z!=1u||!SizeMatches(TemporalM))return false;
    uint n,s;TemporalReceipt.GetDimensions(n);if(n!=1024u)return false;
    CurrentFrame.GetDimensions(n,s);if(n!=64u||s!=4u)return false;
    if(any(TemporalReceipt.Load4(0u)!=uint4(0x31545742u,1u,1u,0u)))return false;
    if(any(uint4(CurrentFrame[0],CurrentFrame[1],CurrentFrame[2],CurrentFrame[3])!=uint4(Source.x,Source.w,Source.x*Source.w,0u))||
       CurrentFrame[9u]!=1u||CurrentFrame[10u]!=1u||CurrentFrame[11u]!=0u||
       CurrentFrame[22u]!=SCT[3u]||CurrentFrame[23u]!=1u)return false;
    if(any(TemporalReceipt.Load4(16u)!=uint4(CurrentFrame[4],CurrentFrame[5],CurrentFrame[6],CurrentFrame[7])))return false;
    [loop]for(uint i=0u;i<32u;++i)if(TemporalReceipt.Load(4u*(20u+i))!=CurrentFrame[16u+i])return false;
    [loop]for(uint j=0u;j<24u;++j)if(TemporalReceipt.Load(4u*(52u+j))!=CBWord(j))return false;
    if(any(TemporalReceipt.Load4(304u)!=uint4(Model.xy,Observation.xy))||
       any(TemporalReceipt.Load4(320u)!=uint4(CurrentFrame[52],CurrentFrame[53],CurrentFrame[54],CurrentFrame[55])))return false;
    return true;
}

uint F(uint word) { return F2Data.Load(word*4u); }
bool CurrentOutlineTyped() {
    if(Request.w!=2u||Request.y!=1u||Request.z!=0u||!SizeMatches(CurrentOutlineM))return false;
    if(Source.z>=Source.y||Source.w!=Source.y-Source.z)return false;
    uint n,z;SourceEnvelope.GetDimensions(n,z);if(n!=104u||z!=4u)return false;
    CBCMeta.GetDimensions(n,z);if(n!=512u||z!=4u)return false;
    RawMeta.GetDimensions(n,z);if(n!=512u||z!=4u)return false;
    F2Data.GetDimensions(n);if(n!=68608u*4u)return false;
    if(any(uint4(CBCMeta[0u],CBCMeta[1u],CBCMeta[2u],CBCMeta[3u])!=uint4(0x31434e45u,5u,1u,0u))||
       CBCMeta[4u]!=1u||CBCMeta[5u]!=1u||CBCMeta[9u]!=1u||CBCMeta[80u]!=SCT[3u])return false;
    [loop]for(uint i=0u;i<24u;++i)if(CBCMeta[16u+i]!=CBWord(i)||RawMeta[12u+i]!=CBWord(i)||F(8u+i)!=CBWord(i))return false;
    if(any(uint4(CBCMeta[40u],CBCMeta[41u],CBCMeta[42u],CBCMeta[43u])!=uint4(0x31434e45u,1u,1u,Request.x)))return false;
    if(any(uint4(CBCMeta[44u],CBCMeta[45u],CBCMeta[46u],CBCMeta[47u])!=uint4(Source.x,Source.w,28u,1u))||
       any(uint4(CBCMeta[48u],CBCMeta[49u],CBCMeta[50u],CBCMeta[51u])!=uint4(Source.xy,28u,1u)))return false;
    if(any(uint4(CBCMeta[52u],CBCMeta[53u],CBCMeta[54u],CBCMeta[55u])!=uint4(Token.xy,1u,1u^0x6f435257u))||
       any(uint4(CBCMeta[56u],CBCMeta[57u],CBCMeta[58u],CBCMeta[59u])!=uint4(Model.xy,Observation.xy))||
       any(uint4(CBCMeta[60u],CBCMeta[61u],CBCMeta[62u],CBCMeta[63u])!=uint4(Model.zw,0u,0u)))return false;
    if(any(uint4(CBCMeta[64u],CBCMeta[65u],CBCMeta[66u],CBCMeta[67u])!=uint4(0u,0u,Source.xy))||
       any(uint4(CBCMeta[68u],CBCMeta[69u],CBCMeta[70u],CBCMeta[71u])!=uint4(5u,0u,4u,136u)))return false;
    [unroll]for(uint c=72u;c<80u;++c)if(CBCMeta[c]!=0u)return false;
    // SourceEnvelope is the ACTUAL producer-owned literal UIP40+CGS5 proof64.
    if(any(uint3(SourceEnvelope[0u],SourceEnvelope[1u],SourceEnvelope[2u])!=uint3(4u,0x34504955u,1u))||
       SourceEnvelope[3u]>255u||SourceEnvelope[20u]!=SCT[3u]||SourceEnvelope[22u]!=0u||SourceEnvelope[23u]!=0u)return false;
    if(any(uint4(SourceEnvelope[4u],SourceEnvelope[5u],SourceEnvelope[6u],SourceEnvelope[7u])!=Model)||
       any(uint2(SourceEnvelope[8u],SourceEnvelope[9u])!=Observation.xy)||
       any(uint2(SourceEnvelope[10u],SourceEnvelope[11u])!=Source.xy)||
       any(uint4(SourceEnvelope[12u],SourceEnvelope[13u],SourceEnvelope[14u],SourceEnvelope[15u])!=Field)||
       any(uint4(SourceEnvelope[16u],SourceEnvelope[17u],SourceEnvelope[18u],SourceEnvelope[19u])!=Content))return false;
    if(any(uint4(SourceEnvelope[40u],SourceEnvelope[41u],SourceEnvelope[42u],SourceEnvelope[43u])!=uint4(0x35534743u,1u,5u,1u))||
       any(uint4(SourceEnvelope[44u],SourceEnvelope[45u],SourceEnvelope[46u],SourceEnvelope[47u])!=uint4(Model.xy,Observation.xy))||
       any(uint4(SourceEnvelope[48u],SourceEnvelope[49u],SourceEnvelope[50u],SourceEnvelope[51u])!=uint4(Model.zw,Token.xy))||
       any(uint4(SourceEnvelope[52u],SourceEnvelope[53u],SourceEnvelope[54u],SourceEnvelope[55u])!=Source))return false;
    uint4 roi=uint4(SourceEnvelope[56u],SourceEnvelope[57u],SourceEnvelope[58u],SourceEnvelope[59u]);
    uint4 raw=uint4(SourceEnvelope[60u],SourceEnvelope[61u],SourceEnvelope[62u],SourceEnvelope[63u]);
    if(any(raw.xy==0u)||any(raw.xy>16384u)||any(raw.zw==0u)||any(roi.xy>=roi.zw)||any(roi.zw>raw.xy)||any(roi.zw-roi.xy!=Source.xy))return false;
    if((SourceEnvelope[64u]&8u)==0u||SourceEnvelope[65u]!=4u||SourceEnvelope[66u]!=1u||SourceEnvelope[67u]!=1u||
       SourceEnvelope[68u]!=1u||SourceEnvelope[69u]!=0u||SourceEnvelope[70u]>2u||SourceEnvelope[71u]!=63u)return false;
    uint N=Source.x*Source.w,G=((Source.x+15u)/16u)*((Source.w+15u)/16u);
    if(any(uint4(SourceEnvelope[72u],SourceEnvelope[73u],SourceEnvelope[74u],SourceEnvelope[75u])!=uint4(N,N,0u,N))||
       any(uint4(SourceEnvelope[76u],SourceEnvelope[77u],SourceEnvelope[78u],SourceEnvelope[79u])!=uint4(G,G,0u,SCT[3u])))return false;
    if(any(uint4(SourceEnvelope[80u],SourceEnvelope[81u],SourceEnvelope[82u],SourceEnvelope[83u])!=Field)||
       any(uint4(SourceEnvelope[84u],SourceEnvelope[85u],SourceEnvelope[86u],SourceEnvelope[87u])!=Content)||
       any(uint4(SourceEnvelope[88u],SourceEnvelope[89u],SourceEnvelope[90u],SourceEnvelope[91u])!=Options)||
       any(uint4(SourceEnvelope[92u],SourceEnvelope[93u],SourceEnvelope[94u],SourceEnvelope[95u])!=Observation)||
       any(uint4(SourceEnvelope[96u],SourceEnvelope[97u],SourceEnvelope[98u],SourceEnvelope[99u])!=Root.Load4(240u))||
       any(uint4(SourceEnvelope[100u],SourceEnvelope[101u],SourceEnvelope[102u],SourceEnvelope[103u])!=uint4(10u,28u,10u,1u)))return false;
    if(any(Root.Load4(176u)!=uint4((Field.xy+15u)/16u,1u,0u))||any(Root.Load4(224u)!=uint4(0u,1u,1u,0u)))return false;
    if(any(uint4(RawMeta[0u],RawMeta[1u],RawMeta[2u],RawMeta[3u])!=uint4(0x31474e52u,1u,1u,0u)))return false;
    [unroll]for(uint r=0u;r<16u;++r)if(RawMeta[36u+r]!=CBCMeta[r])return false;
    if(any(uint4(F(0u),F(1u),F(2u),F(3u))!=uint4(0x37424643u,7u,1u,0u)))return false;
    // Final-publish copies RawMeta0..15 to Data32..47; begin initializes this region0.
    // Same owned wrapper interval is necessary: raw/F2 do NOT contain an independent token.
    [unroll]for(uint f=0u;f<16u;++f)if(F(32u+f)!=RawMeta[f])return false;
    if(any(uint4(F(48u),F(49u),F(50u),F(51u))!=uint4(Source.x,Source.w,N,68608u))||
       any(uint3(F(52u),F(53u),F(54u))!=uint3(64u,128u,16u))||
       any(uint4(F(59u),F(60u),F(61u),F(62u))!=uint4(N,0x3fffffffu,0x80000000u,0x40000000u))||
       any(uint4(F(63u),F(64u),F(65u),F(66u))!=uint4(255u,256u,512u,1024u)))return false;
    if(F(71u)!=2048u||F(72u)>N||F(73u)>N||F(74u)!=4096u||F(75u)>N||F(76u)>N)return false;
    if(F(77u)!=0x31444e49u||F(78u)!=1u||F(79u)>255u||F(80u)>255u||F(81u)>N||F(82u)>N||F(83u)!=1u)return false;
    // CFB5 certified collar plus UNKNOWN packed membership and independent literal RawM fault fallback.
    return true;
}
[numthreads(1,1,1)]
void prepare_main(uint3 group:SV_GroupID) {
    if(any(group))return;
    uint bytes,w,h;NativeArguments.GetDimensions(bytes);SelectedM.GetDimensions(w,h);
    if(bytes!=16u)return;
    Root.GetDimensions(bytes);
    bool hold=bytes==256u&&any(Root.Load3(240u)==0u);
    NativeArguments.Store4(0u,uint4((w+15u)/16u,(h+15u)/16u,hold?0u:1u,0u));
}
groupshared uint GroupCurrent,GroupKind;
[numthreads(16,16,1)]
void select_main(uint3 id:SV_DispatchThreadID,uint lane:SV_GroupIndex) {
    if(lane==0u) {
        if(Request.w==2u) {
            GroupCurrent=LegacyContextTyped()?1u:0u;GroupKind=1u;
            if(GroupCurrent!=0u&&CurrentTyped()&&BaselineTyped()&&CurrentOutlineTyped())GroupKind=3u;
        } else {
            GroupCurrent=CurrentTyped()?1u:0u;GroupKind=1u;
            if(GroupCurrent!=0u&&BaselineTyped()){GroupKind=2u;if(TemporalTyped())GroupKind=3u;}
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint w,h;SelectedM.GetDimensions(w,h);if(id.z!=0u||id.x>=w||id.y>=h)return;
    uint value=0u;
    if(Request.w==2u) {
        // Literal LEGACY data is retained on every source/readiness failure. New PS also keeps
        // the original legacy SRV at t8 for receipt-unavailable/startup/root0 fallback.
        value=GroupKind==3u?CurrentOutlineM.Load(int3(id.xy,0)):LegacyM.Load(int3(id.xy,0));
        SelectedM[id.xy]=value;
    } else {
        if(GroupCurrent!=0u){value=LegacyM.Load(int3(id.xy,0));if(GroupKind==2u)value=BaselineM.Load(int3(id.xy,0));if(GroupKind==3u)value=TemporalM.Load(int3(id.xy,0));}
        SelectedM[id.xy]=value==1u?1u:0u;
    }
    if(all(id.xy==0u)) {
        SelectedReceipt[0u]=0x31534d53u;SelectedReceipt[1u]=1u;SelectedReceipt[2u]=GroupKind;SelectedReceipt[3u]=GroupCurrent!=0u?0u:1u;
        [loop]for(uint i=0u;i<24u;++i)SelectedReceipt[4u+i]=CBWord(i);
        [loop]for(uint j=0u;j<32u;++j)SelectedReceipt[28u+j]=GroupCurrent!=0u?SCT[j]:0u;
        SelectedReceipt[60u]=Token.x;SelectedReceipt[61u]=Token.y;SelectedReceipt[62u]=Request.w==2u?2u:1u;SelectedReceipt[63u]=0u;
    }
}
