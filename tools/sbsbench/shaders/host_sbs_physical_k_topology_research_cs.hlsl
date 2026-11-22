// PRIVATE non-K connectivity experiment. The canonical Inside replay remains
// a separate byte-exact prerequisite. Only this fresh graph's barrier changes:
// exact current physical K, never the broader nearest-family blacklike rule.
// Actual canonical ROI union/reduce and the invalid-neighbour classifier consume
// this graph. No production resource, mask, palette or header is writable.
#include "../../../src_assets/windows/assets/shaders/directx/current_outline/inside/host_sbs_current_inside_completion_cs.hlsl"
uint BlackModelWord(uint word) { return CBCMeta[word]; }
#include "../../../src_assets/windows/assets/shaders/directx/current_outline/outline/physical_black_model_shared.hlsl"
Texture2D<uint> CurrentFaultFreeInside : register(t15);

bool PhysicalGraphReady() {
    if(Hook.x>=8u||any(Hook.yzw)||!FrameReadyRO()||!UnitReady(Hook.x))return false;
    uint bank=PhysicalModelBank(Hook.x);
    if(!PhysicalModelRecordValid(Hook.x)||CBCMeta[bank+2u]!=1u||CBCMeta[bank+1u]==4u)return false;
    uint bit=1u<<Hook.x;
    if(ProfileRead.Load(83u*4u)!=1u||(ProfileRead.Load(79u*4u)&bit)==0u||(ProfileRead.Load(80u*4u)&bit)!=0u)return false;
    uint w,h;CurrentFaultFreeInside.GetDimensions(w,h);return all(uint2(w,h)==Shape.xy);
}
[numthreads(1,1,1)]
void physical_prepare_main(uint3 g:SV_GroupID) {
    if(any(g))return;uint bytes;StageWords.GetDimensions(bytes);if(bytes!=16u)return;StageWords.Store4(0u,0u);
    uint np,sp,nf,sf;ParentWrite.GetDimensions(np,sp);FlagsWrite.GetDimensions(nf,sf);
    if(np!=Shape.z+1u||nf!=np||sp!=4u||sf!=4u)return;
    ParentWrite[0u]=0u;FlagsWrite[0u]=0u;
    if(!PhysicalGraphReady())return;uint u=UB(Hook.x);
    uint4 r=uint4(CBCMeta[u+24u],CBCMeta[u+25u],CBCMeta[u+26u],CBCMeta[u+27u]);
    if(any(r.xy>=r.zw)||any(r.zw>Shape.xy))return;
    StageWords.Store4(0u,uint4((r.zw-r.xy+15u)/16u,1u,0u));
}
[numthreads(16,16,1)]
void physical_emit_main(uint3 d:SV_DispatchThreadID) {
    uint2 p;if(d.z!=0u||!PhysicalGraphReady()||!ROI(d.xy,p))return;
    uint np,sp,nf,sf;ParentWrite.GetDimensions(np,sp);FlagsWrite.GetDimensions(nf,sf);
    if(np!=Shape.z+1u||nf!=np||sp!=4u||sf!=4u)return;
    uint bit=1u<<Hook.x;bool cover=(CoverBits.Load(int3(p,0))&bit)!=0u;
    float4 v=CurrentRGB.Load(int3(p,0));bool passable=cover&&ColourValid(v)&&!PhysicalBlackRGB(v.rgb,Hook.x);
    uint distanceF=255u;bool unusedBarrier=true;
    if(passable)Classify(v.rgb,Hook.x,unusedBarrier,distanceF);
    uint inside=CurrentFaultFreeInside.Load(int3(p,0));
    uint good=ProfileRead.Load(79u*4u)&~ProfileRead.Load(80u*4u)&255u;
    // Malformed positive bits cannot silently become an absent keep witness.
    if((inside&~good)!=0u){uint ignored;InterlockedExchange(ParentWrite[0u],INVALID,ignored);}
    MaskWrite[p]=passable?1u:0u;
    // Qualified palette boxes only (canonical Classify distanceF), plus current
    // fault-free Inside. Whole BODY/LOW membership never supplies a seed here.
    SeedWrite[p]=passable&&(distanceF==0u||(inside&bit)!=0u)?1u:0u;
    DomainWrite[p]=cover?Hook.x+1u:0u;CoverWrite[p]=cover?1u:0u;
    uint id=p.y*Shape.x+p.x+1u;ParentWrite[id]=passable?id:INVALID;FlagsWrite[id]=0u;
}
