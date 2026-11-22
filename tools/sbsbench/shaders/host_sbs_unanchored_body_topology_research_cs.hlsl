// PRIVATE stricter successor to the separately retained body-witness diagnostic.
// The common pixel predicate only proposes. Actual canonical Inside topology,
// reproduced byte-exactly on fresh scratch, can only WITHDRAW that proposal.
#include "host_sbs_unanchored_body_research_cs.hlsl"
cbuffer TopologyInterpretation : register(b2) { uint4 TopologyControl; };
Texture2D<uint> ClosedUnseededBits : register(t12);
Texture2D<uint> UnknownTopologyBits : register(t13);
StructuredBuffer<uint> TopologyUnitHeader : register(t14);
Texture2D<uint> ExteriorTopologyBits : register(t15);
Texture2D<uint> SeededTopologyBits : register(t16);
static const uint KEEP_NOT_CLOSED=512u,KEEP_TOPOLOGY_UNKNOWN=1024u;
static const uint KEEP_TOPOLOGY_EXTERIOR=2048u,KEEP_TOPOLOGY_SEEDED=4096u,TOPOLOGY_CERTIFIED=8192u;

[numthreads(16,16,1)]
void unanchored_body_topology_main(uint3 d:SV_DispatchThreadID) {
    BodyWitnessAt(d);
    if(d.z!=0u||any(d.xy>=Shape.xy))return;
    uint w,h;RejectedBodyWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    FilteredMWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    KeepReasonsWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    uint proposal=RejectedBodyWrite[d.xy];if(proposal==0u)return;
    uint reasons=KeepReasonsWrite[d.xy],failure=0u,n,s;
    if(any(TopologyControl!=uint4(0x31504f54u,1u,1u,0u)))failure|=KEEP_TOPOLOGY_UNKNOWN;
    ClosedUnseededBits.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))failure|=KEEP_TOPOLOGY_UNKNOWN;
    UnknownTopologyBits.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))failure|=KEEP_TOPOLOGY_UNKNOWN;
    ExteriorTopologyBits.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))failure|=KEEP_TOPOLOGY_UNKNOWN;
    SeededTopologyBits.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))failure|=KEEP_TOPOLOGY_UNKNOWN;
    TopologyUnitHeader.GetDimensions(n,s);if(n!=8u||s!=4u)failure|=KEEP_TOPOLOGY_UNKNOWN;
    if(failure==0u) {
        uint closed=ClosedUnseededBits.Load(int3(d.xy,0)),unknown=UnknownTopologyBits.Load(int3(d.xy,0));
        uint exterior=ExteriorTopologyBits.Load(int3(d.xy,0)),seeded=SeededTopologyBits.Load(int3(d.xy,0));
        uint active=(1u<<CurrentOCR[4u])-1u;
        if(((closed|unknown|exterior|seeded)&~active)!=0u)failure|=KEEP_TOPOLOGY_UNKNOWN;
        [loop]for(uint unit=0u;unit<CurrentOCR[4u];++unit)if((proposal&(1u<<unit))!=0u) {
            uint bit=1u<<unit;
            if(TopologyUnitHeader[unit]!=1u||(unknown&bit)!=0u)failure|=KEEP_TOPOLOGY_UNKNOWN;
            if((closed&bit)==0u)failure|=KEEP_NOT_CLOSED;
            if((exterior&bit)!=0u)failure|=KEEP_TOPOLOGY_EXTERIOR;
            if((seeded&bit)!=0u)failure|=KEEP_TOPOLOGY_SEEDED;
        }
    }
    if(failure!=0u) {
        RejectedBodyWrite[d.xy]=0u;FilteredMWrite[d.xy]=CurrentFinalM.Load(int3(d.xy,0));
        KeepReasonsWrite[d.xy]=(reasons&~REJECT)|failure;
    } else KeepReasonsWrite[d.xy]=reasons|TOPOLOGY_CERTIFIED;
}
