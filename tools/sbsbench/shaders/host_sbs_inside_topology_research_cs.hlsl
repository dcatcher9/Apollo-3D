// TEST-ONLY observer of the ACTUAL canonical Inside Parent/RootFlags graph.
// No connectivity is reconstructed and neither canonical graph is writable.
// The caller first replays canonical Inside shaders/ROI union/reduce on fresh
// scratch, and must compare completed replay InsideBits with production bits
// before interpreting this diagnostic. These labels grant no render authority.
//
// Per unit, use the canonical prepared InsideArgs for all three entrypoints:
// prepare_main -> ordered reduce_invalid_main -> ordered classify_main.
// Clear RootBad before each unit; clear all four output textures and UnitHeader
// once per frame. Unissued units retain header0. Header1 means prepared; any
// ORed bit2 revokes that UNIT'S entire topology interpretation, including class
// bits written by other lanes before a malformed graph was discovered.
cbuffer ActualContourCB : register(b0) {
    uint4 CSource; uint4 CField; uint4 CContent;
    uint4 CFrame; uint4 CObservation; uint4 COptions;
};
cbuffer UnitHook : register(b1) { uint4 Hook; };
cbuffer ResearchShape : register(b2) { uint4 Shape; };
Texture2D<float4> CurrentRGB : register(t0);
Texture2D<uint> CurrentEligible : register(t1);
StructuredBuffer<uint> CurrentParent : register(t2);
StructuredBuffer<uint> CurrentRootFlags : register(t3);
Texture2D<uint> CurrentCoverBits : register(t4);
StructuredBuffer<uint> CurrentCBC : register(t5);
StructuredBuffer<uint> CurrentOCR : register(t6);
RWTexture2D<uint> ClosedUnseededWrite : register(u0);
RWTexture2D<uint> ExteriorWrite : register(u1);
RWTexture2D<uint> SeededWrite : register(u2);
RWTexture2D<uint> UnknownWrite : register(u3);
RWStructuredBuffer<uint> RootBad : register(u4);
RWStructuredBuffer<uint> UnitHeader : register(u5);

static const uint HEADER_READY=1u,HEADER_FAULT=2u;
static const uint ROOT_INVALID_NEIGHBOR=1u;

uint CBWord(uint i) {
    if(i<4u)return CSource[i];if(i<8u)return CField[i-4u];
    if(i<12u)return CContent[i-8u];if(i<16u)return CFrame[i-12u];
    if(i<20u)return CObservation[i-16u];return COptions[i-20u];
}
uint UB(uint unit) { return 128u+32u*unit; }
bool Native(int2 p) { return all(p>=0)&&all(p<int2(Shape.xy)); }
bool HeaderShape() { uint n,s;UnitHeader.GetDimensions(n,s);return n==8u&&s==4u; }
void Fault() {
    if(Hook.x<8u&&HeaderShape()){uint ignored;InterlockedOr(UnitHeader[Hook.x],HEADER_FAULT,ignored);}
}
bool OutputsKnown() {
    uint w,h,n,s;
    ClosedUnseededWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    ExteriorWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    SeededWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    UnknownWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    RootBad.GetDimensions(n,s);return n==Shape.z+1u&&s==4u&&HeaderShape();
}
bool FrameKnown() {
    if(Hook.x>=8u||any(Hook.yzw)||any(Shape.xy==0u)||any(Shape.xy>5120u)||Shape.z!=Shape.x*Shape.y||Shape.w!=0u||any(CSource.xw!=Shape.xy))return false;
    if(CSource.y==0u||CSource.y>5120u||CSource.z>=CSource.y||CSource.w!=CSource.y-CSource.z)return false;
    if(any(CField.xy==0u)||any(CField.xy>5120u)||CField.z>=CField.w||CField.w>CField.y||any(CContent.xy>=CContent.zw)||any(CContent.zw>CField.xy))return false;
    if(any(COptions!=uint4(1u,27u,0u,0u))||CObservation.z>1u||CObservation.w!=0u||!any(CFrame.xy)||!any(CObservation.xy))return false;
    uint w,h,n,s;
    CurrentRGB.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    CurrentEligible.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    CurrentCoverBits.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    CurrentCBC.GetDimensions(n,s);if(n!=512u||s!=4u)return false;
    CurrentOCR.GetDimensions(n,s);if(n!=256u||s!=4u)return false;
    if(any(uint4(CurrentCBC[0u],CurrentCBC[1u],CurrentCBC[2u],CurrentCBC[3u])!=uint4(0x31434e45u,5u,1u,0u))||CurrentCBC[4u]!=1u||CurrentCBC[5u]!=1u||CurrentCBC[9u]!=1u)return false;
    [loop]for(uint i=0u;i<24u;++i)if(CurrentCBC[16u+i]!=CBWord(i))return false;
    if(any(uint2(CurrentCBC[44u],CurrentCBC[45u])!=Shape.xy))return false;
    if(any(uint3(CurrentOCR[0u],CurrentOCR[1u],CurrentOCR[2u])!=uint3(4u,0x3852434fu,1u))||CurrentOCR[3u]!=CurrentOCR[4u]||CurrentOCR[4u]>8u||Hook.x>=CurrentOCR[4u]||CurrentOCR[15u]!=0u)return false;
    if(any(uint4(CurrentOCR[5u],CurrentOCR[6u],CurrentOCR[7u],CurrentOCR[8u])!=CFrame)||any(uint2(CurrentOCR[9u],CurrentOCR[10u])!=CSource.xy)||any(uint4(CurrentOCR[11u],CurrentOCR[12u],CurrentOCR[13u],CurrentOCR[14u])!=CField))return false;
    uint u=UB(Hook.x),bank=384u+16u*Hook.x;
    // These are the independent canonical Inside role gates. Physical CBC
    // selection/adoption status and enclosed-body count are NOT prerequisites.
    // Family validation belongs to canonical prepare, whose actual indirect
    // arguments control issuance of this observer as well.
    if(CurrentCBC[u]!=1u||CurrentCBC[u+1u]!=1u||CurrentCBC[u+29u]!=1u||CurrentCBC[bank+1u]==4u||CurrentCBC[bank+2u]!=1u)return false;
    uint4 rect=uint4(CurrentCBC[u+24u],CurrentCBC[u+25u],CurrentCBC[u+26u],CurrentCBC[u+27u]);
    return all(rect.xy<rect.zw)&&all(rect.zw<=Shape.xy)&&OutputsKnown();
}
bool OwnDomain(int2 p) {
    if(!Native(p))return false;uint u=UB(Hook.x);
    uint4 rect=uint4(CurrentCBC[u+24u],CurrentCBC[u+25u],CurrentCBC[u+26u],CurrentCBC[u+27u]);
    return all(p>=int2(rect.xy))&&all(p<int2(rect.zw))&&(CurrentCoverBits.Load(int3(p,0))&(1u<<Hook.x))!=0u;
}
bool PixelFromROI(uint2 local,out uint2 p) {
    p=0u;uint u=UB(Hook.x);uint4 rect=uint4(CurrentCBC[u+24u],CurrentCBC[u+25u],CurrentCBC[u+26u],CurrentCBC[u+27u]);
    if(any(local>=rect.zw-rect.xy))return false;p=rect.xy+local;return all(p<Shape.xy);
}
bool SourceValid(int2 p) {
    if(!Native(p))return false;float4 v=CurrentRGB.Load(int3(p,0));
    return v.a==1.0f&&all(isfinite(v.rgb))&&all(v.rgb>=0.0f)&&all(v.rgb<=1.0f);
}
bool GraphKnown() {
    uint np,sp,nf,sf;CurrentParent.GetDimensions(np,sp);CurrentRootFlags.GetDimensions(nf,sf);
    return np==Shape.z+1u&&nf==np&&sp==4u&&sf==4u&&CurrentParent[0u]==0u;
}
uint FindStable(uint start) {
    uint node=start;
    [loop]for(uint hop=0u;hop<Shape.z;++hop) {
        if(node==0u||node>Shape.z){Fault();return 0u;}
        int2 p=int2((node-1u)%Shape.x,(node-1u)/Shape.x);
        if(!OwnDomain(p)||CurrentEligible.Load(int3(p,0))!=1u){Fault();return 0u;}
        uint next=CurrentParent[node];
        if(next==node)return node;
        if(next==0u||next==0xffffffffu||next>node){Fault();return 0u;}node=next;
    }
    Fault();return 0u;
}
[numthreads(1,1,1)]
void prepare_main(uint3 group:SV_GroupID) {
    if(any(group)||Hook.x>=8u||!HeaderShape())return;
    UnitHeader[Hook.x]=FrameKnown()&&GraphKnown()?HEADER_READY:HEADER_FAULT;
}
[numthreads(16,16,1)]
void reduce_invalid_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||!FrameKnown()||!GraphKnown()){if(all(d==0u))Fault();return;}
    if(UnitHeader[Hook.x]!=HEADER_READY)return;
    uint2 p;if(!PixelFromROI(d.xy,p)||!OwnDomain(int2(p)))return;
    uint eligible=CurrentEligible.Load(int3(p,0));if(eligible>1u){Fault();return;}if(eligible==0u)return;
    uint root=FindStable(p.y*Shape.x+p.x+1u);if(root==0u)return;
    if((CurrentRootFlags[root]&~3u)!=0u){Fault();return;}
    if(!SourceValid(int2(p))){uint ignored;InterlockedOr(RootBad[root],ROOT_INVALID_NEIGHBOR,ignored);return;}
    const int2 cardinal[4]={int2(1,0),int2(-1,0),int2(0,1),int2(0,-1)};
    // An invalid immediate native neighbour is an unavailable digital wall,
    // even when the canonical emit made it nonpassable. It must not certify
    // closure of this root. Native/cover/ROI boundary flags remain separate.
    [unroll]for(uint k=0u;k<4u;++k) {
        int2 q=int2(p)+cardinal[k];
        if(Native(q)&&!SourceValid(q)){uint ignored;InterlockedOr(RootBad[root],ROOT_INVALID_NEIGHBOR,ignored);}
    }
}
[numthreads(16,16,1)]
void classify_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||!FrameKnown())return;
    uint2 p;if(!PixelFromROI(d.xy,p)||!OwnDomain(int2(p)))return;
    uint bit=1u<<Hook.x;
    if(UnitHeader[Hook.x]!=HEADER_READY||!GraphKnown()||!SourceValid(int2(p))) {
        UnknownWrite[p]|=bit;return;
    }
    uint eligible=CurrentEligible.Load(int3(p,0));
    if(eligible!=1u){if(eligible>1u)Fault();UnknownWrite[p]|=bit;return;}
    uint root=FindStable(p.y*Shape.x+p.x+1u);if(root==0u){UnknownWrite[p]|=bit;return;}
    uint flags=CurrentRootFlags[root],bad=RootBad[root];
    if((flags&~3u)!=0u||(bad&~ROOT_INVALID_NEIGHBOR)!=0u){Fault();UnknownWrite[p]|=bit;return;}
    if(bad!=0u){UnknownWrite[p]|=bit;return;}
    if(flags==0u)ClosedUnseededWrite[p]|=bit;
    if((flags&2u)!=0u)ExteriorWrite[p]|=bit;
    if((flags&1u)!=0u)SeededWrite[p]|=bit;
}
