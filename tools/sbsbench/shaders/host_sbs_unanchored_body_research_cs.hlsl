// TEST-ONLY current BODY witness observer. It consumes the EXISTING GPU Inside
// graph; it never reconstructs connectivity, manufactures alpha, changes the
// producer's mask or grants rendering authority. The source may still admit
// scene/ink interpretations with identical RGB. Learned families are hypotheses.
#include "host_sbs_shared_width_prior_research_cs.hlsl"
#include "../../../src_assets/windows/assets/shaders/directx/current_outline/hole/current_stroke_math_shared.hlsl"

Texture2D<uint> CurrentInsideBits : register(t9);
Texture2D<uint> CurrentBody : register(t10);
Texture2D<uint> CurrentFinalM : register(t11);
RWTexture2D<uint> RejectedBodyWrite : register(u2); // per-unit original LOW bits
RWTexture2D<uint> FilteredMWrite : register(u3); // private original-M minus rejection
RWTexture2D<uint> KeepReasonsWrite : register(u4); // independent diagnostic only

static const uint BODY_SEEN=1u,KEEP_UNKNOWN=2u,KEEP_EXACT_FG=4u,KEEP_INSIDE=8u;
static const uint KEEP_BLACKLIKE=16u,KEEP_OTHER_UNIT=32u,REJECT=64u;
static const uint ORIGINAL_HIGH=128u,OTHER_UNIT_PRESENT=256u;

bool InsideCurrent(uint unit) {
    uint bit=1u<<unit;
    return (R(79u)&bit)!=0u&&(R(80u)&bit)==0u;
}
void CurrentFamilyEvidence(float3 rgb,uint3 code,uint unit,out bool exactForeground,out bool blacklike) {
    uint bank=PhysicalModelBank(unit),mask=(FamilyModels.Load(int3(0u,unit,0))>>8u)&255u;
    uint3 low=PhysicalUnpack(CBCMeta[bank+11u]),high=PhysicalUnpack(CBCMeta[bank+12u]);
    uint distanceK=StrokeBoxDistance(code,low,high),distanceF=255u;exactForeground=false;
    [loop]for(uint mode=0u;mode<8u;++mode)if((mask&(1u<<mode))!=0u) {
        uint3 fl=PhysicalUnpack(FamilyModels.Load(int3(1u+2u*mode,unit,0)));
        uint3 fh=PhysicalUnpack(FamilyModels.Load(int3(2u+2u*mode,unit,0)));
        uint distance=StrokeBoxDistance(code,fl,fh);distanceF=min(distanceF,distance);
        exactForeground=exactForeground||distance==0u;
    }
    // The same pointwise arithmetic as current Inside; no RGB threshold is added.
    blacklike=StrokeBlackLikeDecision(rgb,code,low,high,distanceK,distanceF);
}
void BodyWitnessAt(uint3 d) {
    if(d.z!=0u||any(d.xy>=Shape.xy))return;
    uint w,h;RejectedBodyWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    FilteredMWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    KeepReasonsWrite.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    RejectedBodyWrite[d.xy]=0u;FilteredMWrite[d.xy]=0u;KeepReasonsWrite[d.xy]=KEEP_UNKNOWN;
    CurrentFinalM.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    uint original=CurrentFinalM.Load(int3(d.xy,0));
    // Fallback remains literal even if the diagnostic cannot validate its input.
    FilteredMWrite[d.xy]=original;
    if(original>1u||!FrameReady())return;
    CurrentBody.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    CurrentInsideBits.GetDimensions(w,h);if(any(uint2(w,h)!=Shape.xy))return;
    uint body=CurrentBody.Load(int3(d.xy,0));if(body>1u)return;
    if(body==0u){KeepReasonsWrite[d.xy]=0u;return;}
    uint bits=SelectedBits.Load(int3(d.xy,0)),cover=CoverBits.Load(int3(d.xy,0));
    uint inside=CurrentInsideBits.Load(int3(d.xy,0)),owners=bits&255u;
    uint active=(1u<<CurrentOCR[4u])-1u,reasons=BODY_SEEN|((bits&0xff00u)!=0u?ORIGINAL_HIGH:0u);
    // HIGH is an annotation, not necessarily a subset of LOW. Inactive HIGH
    // nevertheless cannot be interpreted as a current unit's evidence.
    if((bits&0xffff0000u)!=0u||(cover&~active)!=0u||(inside&~active)!=0u||
       (((bits>>8u)&255u)&~active)!=0u||owners==0u||(owners&~active)!=0u||(owners&~cover)!=0u) {
        KeepReasonsWrite[d.xy]=reasons|KEEP_UNKNOWN;return;
    }
    float4 value=CurrentRGB.Load(int3(d.xy,0));
    if(value.a!=1.0f||!all(isfinite(value.rgb))||any(value.rgb<0.0f)||any(value.rgb>1.0f)) {
        KeepReasonsWrite[d.xy]=reasons|KEEP_UNKNOWN;return;
    }
    uint3 code=PhysicalRGB8(value.rgb);bool keep=false;
    [loop]for(uint unit=0u;unit<CurrentOCR[4u];++unit) {
        uint bit=1u<<unit;if((cover&bit)==0u)continue;bool own=(owners&bit)!=0u;
        if(!own)reasons|=OTHER_UNIT_PRESENT;
        // Inside has its own canonical readiness. Its current fault-free
        // positive result does not require physical-refinement adoption.
        bool witnessedInside=InsideCurrent(unit)&&(inside&bit)!=0u;
        if(witnessedInside) {
            keep=true;reasons|=KEEP_INSIDE;if(!own)reasons|=KEEP_OTHER_UNIT;
        }
        // Unknown COVERED units conservatively preserve the current mask too.
        // Missing a stricter style certificate is not proof of no other keep.
        if(!StyleReady(unit)||!InsideCurrent(unit)) {
            keep=true;reasons|=KEEP_UNKNOWN;if(!own)reasons|=KEEP_OTHER_UNIT;continue;
        }
        bool exactForeground,blacklike;
        CurrentFamilyEvidence(value.rgb,code,unit,exactForeground,blacklike);
        if(exactForeground||blacklike||witnessedInside) {
            keep=true;
            if(exactForeground)reasons|=KEEP_EXACT_FG;
            if(blacklike)reasons|=KEEP_BLACKLIKE;
            if(witnessedInside)reasons|=KEEP_INSIDE;
            if(!own)reasons|=KEEP_OTHER_UNIT;
        }
    }
    if(!keep) {
        RejectedBodyWrite[d.xy]=owners;FilteredMWrite[d.xy]=0u;reasons|=REJECT;
    }
    KeepReasonsWrite[d.xy]=reasons;
}
[numthreads(16,16,1)]
void unanchored_body_main(uint3 d:SV_DispatchThreadID) { BodyWitnessAt(d); }
