// PRIVATE TEST-ONLY soft-layer recovery hypothesis. No production registration.
// Compile only soft_layer_main (cs_5_0), after the unchanged current producer.
// All input resources remain immutable. This is not alpha, clean-background,
// semantic text, synthesized RGB or outline authority.
//
// Bindings deliberately match host_sbs_unanchored_body_research_cs.hlsl:
// b0 actual C24; b1 Shape=(native width, band height, pixel count, 0).
// Current SRVs: t0 RGB, t1 OCR256, t2 selected LOW/HIGH, t3 CoverBits,
// t4 CBC512, t5 F2 raw68608, t6 family17x8, t7 SCT32,
// t9 current Inside bits, t10 original BODY, t11 immutable original final M.
// t8 and u0/u1 belong to the included width-prior observer and are UNUSED by
// soft_layer_main. The three outputs below have native Shape.xy dimensions:
// u2 per-unit recovery bits; u3 literal original M plus recovery; u4 reasons.
// The renames share validators/layout without changing the existing observer.
#define RejectedBodyWrite RecoveryBitsWrite
#define FilteredMWrite RecoveredUnionMWrite
#define KeepReasonsWrite SoftReasonsWrite
#include "host_sbs_unanchored_body_research_cs.hlsl"
#undef RejectedBodyWrite
#undef FilteredMWrite
#undef KeepReasonsWrite

static const uint SOFT_ORIGINAL_PRESENT=1u;
static const uint SOFT_FRAME_UNAVAILABLE=2u;
static const uint SOFT_MALFORMED_PIXEL=4u;
static const uint SOFT_NOT_BODY=8u;
static const uint SOFT_NO_OWNED_LOW=16u;
static const uint SOFT_STYLE_UNAVAILABLE=32u;
static const uint SOFT_INVALID_SOURCE=64u;
static const uint SOFT_PHYSICAL_K=128u;
static const uint SOFT_NOT_BLACKLIKE=256u;
static const uint SOFT_EXACT_FOREGROUND=512u;
static const uint SOFT_NO_VALID_EXTERIOR=1024u;
static const uint SOFT_ROW_COLOUR_PRESENT=2048u;
static const uint SOFT_RECOVERED=4096u;
static const uint SOFT_MALFORMED_ROW_COVER=8192u;
static const uint SOFT_BLACKLIKE=16384u;
static const uint SOFT_OWN_INSIDE=32768u;

bool SoftValidSource(float4 value) {
    return value.a==1.0f&&all(isfinite(value.rgb))&&
           all(value.rgb>=0.0f)&&all(value.rgb<=1.0f);
}

// Exhaust the actual native row, not a radius, vertical neighbourhood or a
// selected sample subset. Invalid source samples supply neither a match nor
// exterior availability. A malformed cover word makes absence uninterpretable.
void SoftExteriorRow(uint y,uint3 code,uint active,
                     out bool available,out bool present,out bool malformed) {
    available=false;present=false;malformed=false;
    [loop]for(uint x=0u;x<Shape.x;++x) {
        uint cover=CoverBits.Load(int3(x,y,0));
        if((cover&~active)!=0u) { malformed=true;return; }
        if(cover!=0u)continue; // Exterior to ALL current paired covers.
        float4 value=CurrentRGB.Load(int3(x,y,0));
        if(!SoftValidSource(value))continue;
        available=true;
        if(all(PhysicalRGB8(value.rgb)==code))present=true;
    }
}

[numthreads(16,16,1)]
void soft_layer_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||any(d.xy>=Shape.xy))return;
    uint w,h;
    RecoveryBitsWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    RecoveredUnionMWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    SoftReasonsWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    RecoveryBitsWrite[d.xy]=0u;
    RecoveredUnionMWrite[d.xy]=0u;
    SoftReasonsWrite[d.xy]=SOFT_FRAME_UNAVAILABLE;
    // The caller must bind the original M and all three output shapes exactly.
    // If original M itself has no corresponding texel, no literal can be read.
    CurrentFinalM.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    uint original=CurrentFinalM.Load(int3(d.xy,0));
    RecoveredUnionMWrite[d.xy]=original; // Preserve even noncanonical M literally.
    if(original>1u) { SoftReasonsWrite[d.xy]=SOFT_MALFORMED_PIXEL;return; }
    if(original==1u) { SoftReasonsWrite[d.xy]=SOFT_ORIGINAL_PRESENT;return; }
    if(!FrameReady())return;
    CurrentBody.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    CurrentInsideBits.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;

    uint active=(1u<<CurrentOCR[4u])-1u;
    uint body=CurrentBody.Load(int3(d.xy,0));
    uint bits=SelectedBits.Load(int3(d.xy,0));
    uint cover=CoverBits.Load(int3(d.xy,0));
    uint inside=CurrentInsideBits.Load(int3(d.xy,0));
    uint owners=bits&255u,high=(bits>>8u)&255u;
    if(body>1u||(bits&0xffff0000u)!=0u||(cover&~active)!=0u||
       (inside&~active)!=0u||(high&~active)!=0u||
       (owners&~active)!=0u||(owners&~cover)!=0u) {
        SoftReasonsWrite[d.xy]=SOFT_MALFORMED_PIXEL;return;
    }
    if(body==0u) { SoftReasonsWrite[d.xy]=SOFT_NOT_BODY;return; }
    if(owners==0u) { SoftReasonsWrite[d.xy]=SOFT_NO_OWNED_LOW;return; }
    float4 value=CurrentRGB.Load(int3(d.xy,0));
    if(!SoftValidSource(value)) { SoftReasonsWrite[d.xy]=SOFT_INVALID_SOURCE;return; }
    uint3 code=PhysicalRGB8(value.rgb);

    // Missing/unknown completion of any overlapping current owner abstains.
    // InsideCurrent is completion/fault state, not a demand for Inside(q)==1.
    // In particular no new width, K, Raw, graph, HIGH or Inside permission is made.
    [loop]for(uint unit=0u;unit<CurrentOCR[4u];++unit) {
        uint bit=1u<<unit;if((cover&bit)==0u)continue;
        if(!StyleReady(unit)||!InsideCurrent(unit)) {
            SoftReasonsWrite[d.xy]=SOFT_STYLE_UNAVAILABLE;return;
        }
        // Never recover a physical-K sample under another covered unit either.
        if(PhysicalBlackCode(code,unit)) {
            SoftReasonsWrite[d.xy]=SOFT_PHYSICAL_K;return;
        }
    }

    uint candidates=owners,reasons=0u;
    [loop]for(uint owner=0u;owner<CurrentOCR[4u];++owner) {
        uint bit=1u<<owner;if((owners&bit)==0u)continue;
        bool exactForeground,blacklike;
        CurrentFamilyEvidence(value.rgb,code,owner,exactForeground,blacklike);
        // Classification is diagnostic ONLY. It does not narrow this first
        // hypothesis or silently discard foreground-like soft/inner AA.
        reasons|=blacklike?SOFT_BLACKLIKE:SOFT_NOT_BLACKLIKE;
        if(exactForeground)reasons|=SOFT_EXACT_FOREGROUND;
        if((inside&bit)!=0u)reasons|=SOFT_OWN_INSIDE;
    }

    bool available,present,malformed;
    SoftExteriorRow(d.y,code,active,available,present,malformed);
    if(malformed) { SoftReasonsWrite[d.xy]=reasons|SOFT_MALFORMED_ROW_COVER;return; }
    if(!available) { SoftReasonsWrite[d.xy]=reasons|SOFT_NO_VALID_EXTERIOR;return; }
    if(present) { SoftReasonsWrite[d.xy]=reasons|SOFT_ROW_COLOUR_PRESENT;return; }
    RecoveryBitsWrite[d.xy]=candidates;
    RecoveredUnionMWrite[d.xy]=1u;
    SoftReasonsWrite[d.xy]=reasons|SOFT_RECOVERED;
}

// LIMITS: Exact row absence is not semantic separation. A unique covered scene
// colour can satisfy this hypothesis; codec variation can remove an exterior
// exact match, and genuine soft ink can share an exterior RGB8 and be rejected.
// BODY/LOW and style families are existing hypotheses, not alpha GT. Blacklike,
// exact-foreground and per-pixel Inside annotations are diagnostic, not gates.
// Original M is read-only and no recovered point is a donor or row reference.
// Worst-case work is O(native width) per eligible pixel, with no performance
// qualification. No current width is required or extended, and no K is widened.
