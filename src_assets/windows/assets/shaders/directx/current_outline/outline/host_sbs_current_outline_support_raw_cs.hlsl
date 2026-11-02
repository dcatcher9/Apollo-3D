// PRIVATE ONE shared learned physical-black topology trial; cache/selection math retained.
// Current validated per-unit CBC model changes the geometric black population.
// Private exact CPU4023 raw black-ring completion. Current-only semantic UNKNOWN.
cbuffer ActualContourCB : register(b0) {
    uint4 CSource; uint4 CField; uint4 CContent;
    uint4 CFrame; uint4 CObservation; uint4 COptions;
};
cbuffer RingHook : register(b1) { uint4 Hook; };
cbuffer RingShape : register(b2) { uint4 Shape; };
Texture2D<float4> CurrentRGB : register(t0);
Texture2D<uint> AllBodyBits : register(t1);
Texture2D<uint> SelectedBits : register(t2);
Texture2D<uint> CoverBits : register(t3);
StructuredBuffer<uint> EnclosureMeta : register(t4);
Texture2D<uint> Eligible : register(t5);
StructuredBuffer<uint> Parent : register(t6);
StructuredBuffer<uint> RootFlags : register(t7);
Texture2D<uint> InnerEligible : register(t8);
Texture2D<uint> Code1M : register(t9);
globallycoherent RWStructuredBuffer<uint> Meta : register(u0);
RWTexture2D<uint> EligibleWrite : register(u1);
RWTexture2D<uint> SeedWrite : register(u2);
RWTexture2D<uint> DomainWrite : register(u3);
RWTexture2D<uint> CoverWrite : register(u4);
RWTexture2D<uint> InnerWrite : register(u5);
RWTexture2D<uint> RingM : register(u6);
RWByteAddressBuffer UnitArgs : register(u7);
static const uint TAG=0x31474e52u, F_PARENT=1u, F_SHAPE=2u, F_GRAPH=4u;
uint UB(uint unit) { return 128u+32u*unit; }
bool Native(uint2 p) { return all(p<Shape.xy); }
void Fault(uint bit) { uint ignored;InterlockedOr(Meta[3u],bit,ignored); }
uint CBWord(uint i) {
    if(i<4u)return CSource[i];if(i<8u)return CField[i-4u];
    if(i<12u)return CContent[i-8u];if(i<16u)return CFrame[i-12u];
    if(i<20u)return CObservation[i-16u];return COptions[i-20u];
}
bool MetaShape() { uint n,s;Meta.GetDimensions(n,s);return n==512u&&s==4u; }
uint BlackModelWord(uint word) { return EnclosureMeta[word]; }
#include "physical_black_model_shared.hlsl"
#include "cbc_unit_state_shared.hlsl"
uint SelectedGeometryBits(uint word) { return word&255u; }
bool AllBodyHasGeometry(uint word,uint bit) { return (word&bit)!=0u; }
bool SelectedHasGeometry(uint word,uint bit) { return (word&bit)!=0u; }
bool ParentValid() {
    uint n,s;EnclosureMeta.GetDimensions(n,s);
    if(n!=512u||s!=4u)return false;
    if(any(uint4(EnclosureMeta[0u],EnclosureMeta[1u],EnclosureMeta[2u],EnclosureMeta[3u])!=uint4(0x31434e45u,5u,1u,0u)))return false;
    [loop]for(uint i=0u;i<24u;++i)if(EnclosureMeta[16u+i]!=CBWord(i))return false;
    // One validation per evaluated unit at Rawbegin, not per texel. Bank stays
    // immutable in this same serialized owner interval, including future units.
    [unroll]for(uint unit=0u;unit<8u;++unit) {
        uint evaluated=EnclosureMeta[UB(unit)];
        if(evaluated>1u||(evaluated==1u&&!PhysicalModelRecordValid(unit)))return false;
    }
    return all(uint2(EnclosureMeta[44u],EnclosureMeta[45u])==Shape.xy)&&all(CSource.xw==Shape.xy);
}
bool InputShapes() {
    if(any(Shape.xy==0u)||any(Shape.xy>5120u)||Shape.z!=Shape.x*Shape.y||Shape.w!=0u)return false;
    uint w,h;CurrentRGB.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    AllBodyBits.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    SelectedBits.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    CoverBits.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    Code1M.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return false;
    return true;
}
bool Good() { return Meta[0u]==TAG&&Meta[1u]==1u&&Meta[2u]==1u&&Meta[3u]==0u; }
bool PixelFromROI(uint2 local,out uint2 p) {
    p=0u;if(Hook.x>=8u||any(Hook.yzw))return false;
    uint u=UB(Hook.x);uint4 r=uint4(EnclosureMeta[u+24u],EnclosureMeta[u+25u],EnclosureMeta[u+26u],EnclosureMeta[u+27u]);
    if(any(r.xy>=r.zw)||any(r.zw>Shape.xy)||any(local>=r.zw-r.xy))return false;
    p=r.xy+local;return true;
}
bool InOwnCover(int2 p) {
    if(any(p<0)||any(p>=int2(Shape.xy)))return false;
    return (CoverBits.Load(int3(p,0))&(1u<<Hook.x))!=0u;
}
bool IsBlack(uint2 p) {
    float3 c=CurrentRGB.Load(int3(p,0)).rgb;
    return PhysicalBlackRGB(c,Hook.x);
}
uint FindStable(uint id) {
    uint node=id;
    [loop]while(true) {
        uint next=Parent[node];
        if(next==0u||next==0xffffffffu||next>node){Fault(F_GRAPH);return 0u;}
        if(next==node)return node;
        node=next;
    }
    return 0u;
}
bool GraphValid() {
    uint n,s,nf,sf;Parent.GetDimensions(n,s);RootFlags.GetDimensions(nf,sf);
    if(n!=Shape.z+1u||s!=4u||nf!=n||sf!=4u){Fault(F_SHAPE);return false;}
    if(Parent[0u]!=0u){Fault(F_GRAPH);return false;}
    return true;
}
[numthreads(1,1,1)]
void begin_main(uint3 g:SV_GroupID) {
    if(any(g)||!MetaShape())return;
    [loop]for(uint i=0u;i<512u;++i)Meta[i]=0u;
    Meta[0u]=TAG;Meta[1u]=1u;
    [loop]for(uint j=0u;j<24u;++j)Meta[12u+j]=CBWord(j);
    uint n,s;EnclosureMeta.GetDimensions(n,s);
    if(n==512u&&s==4u) {
        Meta[5u]=EnclosureMeta[2u];
        [unroll]for(uint k=0u;k<16u;++k)Meta[36u+k]=EnclosureMeta[k];
    }
    bool valid=ParentValid()&&InputShapes();Meta[2u]=valid?1u:0u;
    if(!valid)Meta[3u]=F_PARENT;
    // V3 u7 aliases the full GPU-only calibration/histogram raw buffer for this entry.
    uint bytes;UnitArgs.GetDimensions(bytes);
    if(bytes!=68608u*4u){Fault(F_SHAPE);return;}
    [loop]for(uint z=0u;z<1024u;++z)UnitArgs.Store(4u*z,0u);
    UnitArgs.Store(0u,0x37424643u);UnitArgs.Store(4u,7u);UnitArgs.Store(8u,valid?1u:0u);
    UnitArgs.Store(12u,valid?0u:1u);
    [loop]for(uint c=0u;c<24u;++c)UnitArgs.Store((8u+c)*4u,CBWord(c));
    UnitArgs.Store(48u*4u,Shape.x);UnitArgs.Store(49u*4u,Shape.y);UnitArgs.Store(50u*4u,Shape.z);
    UnitArgs.Store(51u*4u,68608u);UnitArgs.Store(52u*4u,64u);UnitArgs.Store(53u*4u,128u);UnitArgs.Store(54u*4u,16u);
    UnitArgs.Store(59u*4u,Shape.z);UnitArgs.Store(60u*4u,0x3fffffffu);UnitArgs.Store(61u*4u,0x80000000u);UnitArgs.Store(62u*4u,0x40000000u);
    UnitArgs.Store(63u*4u,255u);UnitArgs.Store(64u*4u,256u);UnitArgs.Store(65u*4u,512u);UnitArgs.Store(66u*4u,1024u);
    UnitArgs.Store(71u*4u,2048u); // Exact UNKNOWN_RAW geometry retained.
    UnitArgs.Store(77u*4u,0x31444e49u);UnitArgs.Store(78u*4u,1u);
    UnitArgs.Store(74u*4u,4096u); // CFB5 collar-certificate policy; counters75/76 start0.
    [unroll]for(uint unit=0u;unit<8u;++unit){uint cu=128u+64u*unit;UnitArgs.Store((cu+1u)*4u,Shape.x);UnitArgs.Store((cu+2u)*4u,Shape.y);}
}
[numthreads(16,16,1)]
void reset_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||!Native(d.xy))return;
    EligibleWrite[d.xy]=0u;SeedWrite[d.xy]=0u;DomainWrite[d.xy]=0u;CoverWrite[d.xy]=0u;InnerWrite[d.xy]=0u;
    RingM[d.xy]=Code1M.Load(int3(d.xy,0));
    // EligibleWrite/u1 clears packed temp; SeedWrite/u2 clears the one-N cache in this reset. Domain still clears
    // globally, and every subsequent raw graph overwrites Eligible inside its entire ROI.
    uint bytes;UnitArgs.GetDimensions(bytes);if(bytes!=68608u*4u){Fault(F_SHAPE);return;}
    uint pixel=d.y*Shape.x+d.x;
    [loop]for(uint i=pixel;i<67584u;i+=Shape.z)UnitArgs.Store((1024u+i)*4u,0u);
    uint bits=SelectedGeometryBits(SelectedBits.Load(int3(d.xy,0))); // LOW8 immutable geometry; HIGH8 is final-only foreground.
    [unroll]for(uint unit=0u;unit<8u;++unit)if((bits&(1u<<unit))!=0u){
        uint cu=128u+64u*unit,ignored;
        UnitArgs.InterlockedAdd((cu+0u)*4u,1u,ignored);
        UnitArgs.InterlockedMin((cu+1u)*4u,d.x,ignored);UnitArgs.InterlockedMin((cu+2u)*4u,d.y,ignored);
        UnitArgs.InterlockedMax((cu+3u)*4u,d.x+1u,ignored);UnitArgs.InterlockedMax((cu+4u)*4u,d.y+1u,ignored);
    }
}
[numthreads(1,1,1)]
void prepare_unit_main(uint3 g:SV_GroupID) {
    if(any(g)||!MetaShape()||Hook.x>=8u||any(Hook.yzw))return;
    uint bytes;UnitArgs.GetDimensions(bytes);if(bytes!=64u){Fault(F_SHAPE);return;}
    UnitArgs.Store4(0u,0u);UnitArgs.Store4(16u,0u);UnitArgs.Store4(32u,0u);UnitArgs.Store4(48u,0u);
    if(!Good())return;
    uint u=UB(Hook.x);
    // Exact parent CPU guard: evaluated, fully observed, some enclosed original Body.
    if(!CurrentCBCPhysicalRefinementAdopted(EnclosureMeta[u],EnclosureMeta[u+1u],EnclosureMeta[u+3u],EnclosureMeta[u+6u],EnclosureMeta[u+29u]))return;
    uint4 r=uint4(EnclosureMeta[u+24u],EnclosureMeta[u+25u],EnclosureMeta[u+26u],EnclosureMeta[u+27u]);
    if(any(r.xy>=r.zw)||any(r.zw>Shape.xy)){Fault(F_SHAPE);return;}
    uint2 groups=(r.zw-r.xy+15u)/16u;UnitArgs.Store4(0u,uint4(groups,1u,0u));
    Meta[u]=1u;uint ignored;InterlockedOr(Meta[4u],1u<<Hook.x,ignored);
}
[numthreads(16,16,1)]
void emit_passable_main(uint3 d:SV_DispatchThreadID) {
    uint2 p;if(d.z!=0u||!Good()||!PixelFromROI(d.xy,p))return;
    uint bit=1u<<Hook.x;bool cover=(CoverBits.Load(int3(p,0))&bit)!=0u;
    bool body=AllBodyHasGeometry(AllBodyBits.Load(int3(p,0)),bit);
    bool keep=SelectedHasGeometry(SelectedBits.Load(int3(p,0)),bit);
    EligibleWrite[p]=cover&&(!IsBlack(p)||body)?1u:0u;
    SeedWrite[p]=keep?1u:0u;DomainWrite[p]=cover?Hook.x+1u:0u;CoverWrite[p]=cover?1u:0u;InnerWrite[p]=0u;
}
[numthreads(16,16,1)]
void resolve_inner_main(uint3 d:SV_DispatchThreadID) {
    uint2 p;if(d.z!=0u||!Good()||!PixelFromROI(d.xy,p))return;
    // u4 snapshots the FIRST passable graph before blackCC reuses Parent/RootFlags.
    // Root ID low30; bit31 means enclosed; bit30 starts0 and later aggregates bad rims.
    CoverWrite[p]=0u;
    if(all(d.xy==0u)){uint ignored;UnitArgs.Store(56u*4u,Hook.x+1u);UnitArgs.InterlockedAdd(57u*4u,1u,ignored);}
    if(!GraphValid()){InnerWrite[p]=0u;return;}
    uint value=0u;
    if(Eligible.Load(int3(p,0))==1u&&InOwnCover(int2(p))) {
        uint root=FindStable(p.y*Shape.x+p.x+1u);
        if(root!=0u){
            if(root>=0x40000000u){Fault(F_SHAPE);InnerWrite[p]=0u;return;}
            uint flags=RootFlags[root];value=(flags&3u)==1u?1u:0u;
            CoverWrite[p]=root|((flags&2u)==0u?0x80000000u:0u);
        }
    }
    InnerWrite[p]=value;
}
bool AdjacentInner(int2 p) { return InOwnCover(p)&&InnerEligible.Load(int3(p,0))==1u; }
[numthreads(16,16,1)]
void emit_black_main(uint3 d:SV_DispatchThreadID) {
    uint2 p;if(d.z!=0u||!Good()||!PixelFromROI(d.xy,p))return;
    bool cover=InOwnCover(int2(p));bool black=cover&&IsBlack(p);
    bool adjacent=AdjacentInner(int2(p)+int2(-1,0))||AdjacentInner(int2(p)+int2(1,0))||AdjacentInner(int2(p)+int2(0,-1))||AdjacentInner(int2(p)+int2(0,1));
    EligibleWrite[p]=black?1u:0u;SeedWrite[p]=black&&adjacent?1u:0u;
    DomainWrite[p]=cover?Hook.x+1u:0u;CoverWrite[p]=cover?1u:0u;
}
[numthreads(16,16,1)]
void append_main(uint3 d:SV_DispatchThreadID) {
    uint2 p;if(d.z!=0u||!Good()||!PixelFromROI(d.xy,p))return;
    if(!GraphValid())return;
    if(all(d.xy==0u)){Meta[UB(Hook.x)+1u]=1u;uint ignored;InterlockedOr(Meta[6u],1u<<Hook.x,ignored);}
    if(Eligible.Load(int3(p,0))!=1u||!InOwnCover(int2(p)))return;
    uint root=FindStable(p.y*Shape.x+p.x+1u);if(root==0u)return;
    // Whole black4CC, seeded by eligible interior adjacency, rejecting true-cover boundary.
    if((RootFlags[root]&3u)!=1u)return;
    uint old;InterlockedOr(RingM[p],1u,old);
    if(old==0u){uint ignored;InterlockedAdd(Meta[7u],1u,ignored);InterlockedAdd(Meta[UB(Hook.x)+2u],1u,ignored);}
}
[numthreads(16,16,1)]
void publish_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||!Native(d.xy)||!MetaShape())return;
    bool valid=Meta[3u]==0u&&Meta[5u]==1u;
    if(!valid)RingM[d.xy]=Code1M.Load(int3(d.xy,0));
    if(all(d.xy==0u)){
        if(!valid){Meta[7u]=0u;[unroll]for(uint unit=0u;unit<8u;++unit)Meta[UB(unit)+2u]=0u;}
        Meta[2u]=valid?1u:0u;Meta[8u]=!valid?0u:Meta[7u]!=0u?2u:1u;
    }
}
