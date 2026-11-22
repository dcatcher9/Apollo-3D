// PRIVATE physical-K + measured-width guard. The existing physical graph only
// proposes erasure. Preserve original M near current physical K, or whenever
// any owning unit lacks a strong CURRENT LOCAL width or a complete source/ROI
// neighbourhood. Remaining proposals also need a positive same-row current
// exterior colour observation. These are detection guards, not outline or alpha.
#define unanchored_body_topology_main PhysicalKProposalAt
#include "host_sbs_unanchored_body_topology_research_cs.hlsl"
#undef unanchored_body_topology_main

// Additional diagnostic bits (the existing common reason bits stay unchanged).
static const uint KEEP_LOCAL_WIDTH_UNKNOWN=16384u,KEEP_MEASURED_K_COLLAR=32768u;
static const uint KEEP_WIDTH_ROI_UNKNOWN=65536u;
static const uint KEEP_NO_CURRENT_ROW_COLOUR_WITNESS=131072u;
bool WidthSourceValid(float4 v) {
    return v.a==1.0f&&all(isfinite(v.rgb))&&all(v.rgb>=0.0f)&&all(v.rgb<=1.0f);
}
uint LocalWidthKeepReason(int2 p,uint unit) {
    // No inherited/shared estimate: ambiguous or weak local measurements abstain.
    if(StrongDonorBands(unit)==0u)return KEEP_UNKNOWN|KEEP_LOCAL_WIDTH_UNKNOWN;
    uint t=R(CU(unit)+11u);
    if(t==0u||t>16u)return KEEP_UNKNOWN|KEEP_LOCAL_WIDTH_UNKNOWN;
    uint u=UB(unit),bit=1u<<unit;
    uint4 rect=uint4(CBCMeta[u+24u],CBCMeta[u+25u],CBCMeta[u+26u],CBCMeta[u+27u]);
    if(any(rect.xy>=rect.zw)||any(rect.zw>Shape.xy)||any(p<int2(rect.xy))||any(p>=int2(rect.zw))||
       (CoverBits.Load(int3(p,0))&bit)==0u)return KEEP_UNKNOWN|KEEP_WIDTH_ROI_UNKNOWN;
    // Same native Chebyshev extent as the existing measured-width prior. A
    // clipped/invalid sample cannot prove absence of nearby original physical K.
    [loop]for(int dy=-(int)t;dy<=(int)t;++dy)[loop]for(int dx=-(int)t;dx<=(int)t;++dx) {
        int2 q=p+int2(dx,dy);
        if(any(q<0)||any(q>=int2(Shape.xy))||any(q<int2(rect.xy))||any(q>=int2(rect.zw)))return KEEP_UNKNOWN|KEEP_WIDTH_ROI_UNKNOWN;
        if((CoverBits.Load(int3(q,0))&bit)==0u)return KEEP_UNKNOWN|KEEP_WIDTH_ROI_UNKNOWN;
        float4 v=CurrentRGB.Load(int3(q,0));
        if(!WidthSourceValid(v))return KEEP_UNKNOWN|KEEP_WIDTH_ROI_UNKNOWN;
        if(PhysicalBlackRGB(v.rgb,unit))return KEEP_MEASURED_K_COLLAR;
    }
    return 0u;
}
[numthreads(16,16,1)]
void unanchored_body_physical_k_main(uint3 d:SV_DispatchThreadID) {
    PhysicalKProposalAt(d);
    if(d.z!=0u||any(d.xy>=Shape.xy))return;
    uint w,h;RejectedBodyWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    FilteredMWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    KeepReasonsWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    uint proposal=RejectedBodyWrite[d.xy];if(proposal==0u)return;
    uint keep=0u;
    if(!FrameReady()||!WidthSourceValid(CurrentRGB.Load(int3(d.xy,0))))keep|=KEEP_UNKNOWN|KEEP_WIDTH_ROI_UNKNOWN;
    else {
        uint owners=SelectedBits.Load(int3(d.xy,0))&255u,active=(1u<<CurrentOCR[4u])-1u;
        if(owners==0u||proposal!=owners||(owners&~active)!=0u)keep|=KEEP_UNKNOWN|KEEP_WIDTH_ROI_UNKNOWN;
        // Every owning unit must permit erasure. One valid keep or unavailable
        // unit withdraws the entire proposal and retains literal original M.
        [loop]for(uint unit=0u;unit<CurrentOCR[4u]&&keep==0u;++unit)
            if((owners&(1u<<unit))!=0u)keep|=LocalWidthKeepReason(int2(d.xy),unit);
    }
    if(keep!=0u) {
        RejectedBodyWrite[d.xy]=0u;FilteredMWrite[d.xy]=CurrentFinalM.Load(int3(d.xy,0));
        KeepReasonsWrite[d.xy]=(KeepReasonsWrite[d.xy]&~REJECT)|keep;
        return;
    }
    // Positive CURRENT background-colour evidence, never original M0 or a
    // presumed absence of foreground. Exact RGB8 equality may still coincide
    // by chance; this private veto grants no semantic alpha certainty.
    uint3 code=PhysicalRGB8(CurrentRGB.Load(int3(d.xy,0)).rgb);bool witnessed=false;
    [loop]for(uint x=0u;x<Shape.x&&!witnessed;++x) {
        int2 q=int2(x,d.y);
        if(CoverBits.Load(int3(q,0))!=0u)continue; // outside ALL current covers
        float4 v=CurrentRGB.Load(int3(q,0));
        if(WidthSourceValid(v)&&all(PhysicalRGB8(v.rgb)==code))witnessed=true;
    }
    if(!witnessed) {
        RejectedBodyWrite[d.xy]=0u;FilteredMWrite[d.xy]=CurrentFinalM.Load(int3(d.xy,0));
        KeepReasonsWrite[d.xy]=(KeepReasonsWrite[d.xy]&~REJECT)|KEEP_NO_CURRENT_ROW_COLOUR_WITNESS;
    }
}
