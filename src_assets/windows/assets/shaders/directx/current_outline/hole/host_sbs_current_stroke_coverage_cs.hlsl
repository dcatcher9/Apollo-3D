#ifndef HOST_SBS_CURRENT_STROKE_COVERAGE_HLSL
#define HOST_SBS_CURRENT_STROKE_COVERAGE_HLSL
// Optional current stroke coverage. The four original reference samples constrain a
// local affine-background HYPOTHESIS, not independently known background or
// semantic alpha. An unobserved scene spot can be indistinguishable from ink.
// This independent optional group reuses frozen C7 proof and pending/commit;
// it never changes K, Raw, C7 planning/header, original M or donor authority.
#include "host_sbs_current_mask_holes_cs.hlsl"
#include "current_stroke_math_shared.hlsl"
cbuffer StrokeCoverageDiagnostics : register(b3) { uint4 SCDiagnostics; };
RWTexture2D<uint> CoverageOffers : register(u6); // independent original-M0 offer bits

bool SCOffersShape() {
    uint w,h;CoverageOffers.GetDimensions(w,h);return all(uint2(w,h)==Shape.xy);
}
bool SCExportOffers() {
    return SCDiagnostics.x==1u&&!any(SCDiagnostics.yzw)&&SCOffersShape();
}

// Own diagnostic offers only. Native Root1 args clear the WHOLE current image;
// Root0 skips this dispatch and preserves old bytes without claiming freshness.
// No C7 proof/header/root/counter or original/final mask is cleared here.
[numthreads(16,16,1)]
void clear_offers_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||!ShapeKnown()||!SCExportOffers()||!Native(int2(d.xy)))return;
    CoverageOffers[d.xy]=0u;
}

// Slot u3 is this group's OWN 128-byte argument buffer, never C7's buffer.
// The caller clears it before each root-gated prepare dispatch. Root0 skips
// prepare and therefore retains zero support dispatches, never stale work.
bool SCUnitReady(uint unit) {
    return unit<8u&&LateReady()&&UnitKnown(unit)&&
           (Flags[0u]&(1u<<unit))!=0u; // captured, NOT NoShort planned8..15
}

[numthreads(8,1,1)]
void prepare_main(uint3 group:SV_GroupID,uint3 tid:SV_GroupThreadID) {
    if(any(group))return;
    uint bytes;HoleArgs.GetDimensions(bytes);
    if(bytes!=128u)return;
    uint unit=tid.x;if(unit>=8u)return;
    HoleArgs.Store4(unit*16u,0u);
    if(!SCUnitReady(unit))return;
    uint4 rect;if(!UnitRect(unit,rect))return;
    HoleArgs.Store4(unit*16u,uint4((rect.zw-rect.xy+15u)/16u,1u,0u));
}

uint3 SCHalfLow(uint3 v) { return StrokeHalfLow(v); }
uint3 SCHalfHigh(uint3 v) { return StrokeHalfHigh(v); }

// Reference exclusion only. A physical model of another current covered unit
// cannot accidentally become this unit's estimated background. This does not
// grant that other model any foreground/AA or new Raw permission.
bool SCAnyCurrentPhysical(int2 p,uint3 code) {
    if(max(code.r,max(code.g,code.b))<=1u)return true;
    uint cover=CoverBits.Load(int3(p,0));
    [loop]for(uint unit=0u;unit<8u;++unit)if((cover&(1u<<unit))!=0u) {
        uint u=UB(unit);
        if(!CurrentCBCPhysicalRefinementAdopted(CBCMeta[u],CBCMeta[u+1u],CBCMeta[u+3u],CBCMeta[u+6u],CBCMeta[u+29u]))continue;
        if(!PhysicalModelRecordValid(unit))return true; // unavailable reference, no new phase fault
        if(PhysicalBlackCode(code,unit))return true;
    }
    return false;
}

bool SCReference(int2 p,uint unit,out uint3 code) {
    code=0u;
    if(!Domain(p,unit)||!SourceValid(p))return false;
    uint proof=ProofRead.Load(int3(p,0));
    if((proof&~PROOF_ALLOWED)!=0u){Fault();return false;}
    // Original final M and captured LOW|Inner are frozen across ALL units.
    // Selected LOW/HIGH is original; pending31/new M is never consulted.
    if((proof&(ORIGINAL_FINAL_M|255u))!=0u)return false;
    if((SelectedBits.Load(int3(p,0))&65535u)!=0u)return false;
    code=PhysicalRGB8(CurrentRGB.Load(int3(p,0)).rgb);
    if(SCAnyCurrentPhysical(p,code))return false;
    return true;
}

// Necessary condition only: no source/model classification and no Fault here.
// A reserved proof bit retains the old path, so the envelope and original donor
// loop still determine whether/when that corruption is observed. Valid proof
// without any original-M1/current-C3 neighbour cannot supply an old donor.
bool SCPossibleOriginalDonor(int2 q,uint unit) {
    uint required=ORIGINAL_FINAL_M|(1u<<(8u+unit));
    [loop]for(int dy=-1;dy<=1;++dy)[loop]for(int dx=-1;dx<=1;++dx) {
        if(dx==0&&dy==0)continue;
        int2 p=q-int2(dx,dy);if(!Domain(p,unit))continue;
        uint proof=ProofRead.Load(int3(p,0));
        if((proof&~PROOF_ALLOWED)!=0u||(proof&required)==required)return true;
    }
    return false;
}

// Exact per-channel affine feasibility for samples at (1,0),(1,1),(2,0),
// (2,1), in the basis (direction, signed tangent). Doubled half-code cells
// are clipped closed intervals. The common normal slope is shared by both
// rows; tangential slope is unrestricted. All signed intermediates <=1530.
bool SCAffineChannel(uint a,uint b,uint c,uint d,out int low,out int high) {
    int al=(int)(2u*a-(uint)(a>0u)),ah=(int)min(510u,2u*a+1u);
    int bl=(int)(2u*b-(uint)(b>0u)),bh=(int)min(510u,2u*b+1u);
    int cl=(int)(2u*c-(uint)(c>0u)),ch=(int)min(510u,2u*c+1u);
    int dl=(int)(2u*d-(uint)(d>0u)),dh=(int)min(510u,2u*d+1u);
    int slopeLow=max(cl-ah,dl-bh),slopeHigh=min(ch-al,dh-bl);
    low=0;high=0;
    if(slopeLow>slopeHigh)return false;
    low=max(0,max(al-slopeHigh,cl-2*slopeHigh));
    high=min(510,min(ah-slopeLow,ch-2*slopeLow));
    return low<=high;
}

// complete=false: a reference is unavailable/foreground, so no local affine
// hypothesis was supplied. complete=true + false: all references qualified,
// but no physical-range affine continuation fits; abstain for the ENTIRE q.
bool SCStencil(int2 q,int2 direction,int2 tangent,uint unit,
               inout uint referenceState,inout uint3 sharedA,inout uint3 sharedC,
               out bool complete,out int3 low,out int3 high,
               out int3 referenceLow,out int3 referenceHigh) {
    complete=false;low=0;high=0;referenceLow=0;referenceHigh=0;
    uint3 a,b,c,d;
    // Only A and C repeat between this donor's two signed tangents. Cache
    // lazily: the first C query must remain AFTER a qualified B, as before.
    // Reference inputs are immutable and a repeated Fault is idempotent;
    // the first fault-discovering query remains in its original position.
    if((referenceState&1u)==0u) {
        bool valid=SCReference(q+direction,unit,sharedA);
        referenceState|=1u|(valid?2u:0u);
    }
    if((referenceState&2u)==0u)return false;a=sharedA;
    if(!SCReference(q+direction+tangent,unit,b))return false;
    if((referenceState&4u)==0u) {
        bool valid=SCReference(q+2*direction,unit,sharedC);
        referenceState|=4u|(valid?8u:0u);
    }
    if((referenceState&8u)==0u)return false;c=sharedC;
    if(!SCReference(q+2*direction+tangent,unit,d))return false;
    complete=true;
    referenceLow=(int3)min(min(SCHalfLow(a),SCHalfLow(b)),min(SCHalfLow(c),SCHalfLow(d)));
    referenceHigh=(int3)max(max(SCHalfHigh(a),SCHalfHigh(b)),max(SCHalfHigh(c),SCHalfHigh(d)));
    [unroll]for(uint channel=0u;channel<3u;++channel) {
        int l,h;if(!SCAffineChannel(a[channel],b[channel],c[channel],d[channel],l,h))return false;
        low[channel]=l;high[channel]=h;
    }
    return true;
}

// All three uncertainty hulls use the shared exact RGB interval arithmetic.
// Keep the intermediate visibility test as an early rejection before the N8
// references; never carry a narrower fraction interval into an enlarged hull.
[numthreads(16,16,1)]
void support_main(uint3 d:SV_DispatchThreadID) {
    if(d.z!=0u||Hook.x>=8u||any(Hook.yzw)||!SCUnitReady(Hook.x))return;
    uint2 pixel;if(!PixelFromROI(d.xy,pixel))return;int2 q=int2(pixel);
    if(!OriginalM0(q,Hook.x)||!SourceValid(q))return;
    uint original=ProofRead.Load(int3(q,0));
    if((original&~PROOF_ALLOWED)!=0u){Fault();return;}
    // Capture stores this exact qualified-FG/K comparison for every current
    // owned valid ROI pixel. Reuse it without changing source/proof validation.
    if((original&(1u<<(17u+Hook.x)))==0u)return;
    if(!SCPossibleOriginalDonor(q,Hook.x))return;
    uint extent=R(CU(Hook.x)+11u);
    if(!OriginalEnvelope(q,Hook.x,extent))return;
    bool supplied=false;int3 pooledLow=510,pooledHigh=0;
    int3 observedLow=510,observedHigh=0;
    uint donorBit=1u<<(8u+Hook.x);
    [loop]for(int dy=-1;dy<=1;++dy)[loop]for(int dx=-1;dx<=1;++dx) {
        if(dx==0&&dy==0)continue;
        int2 direction=int2(dx,dy),p=q-direction;
        if(!Domain(p,Hook.x)||!SourceValid(p))continue;
        uint proof=ProofRead.Load(int3(p,0));
        if((proof&~PROOF_ALLOWED)!=0u){Fault();return;}
        if((proof&(ORIGINAL_FINAL_M|donorBit))!=(ORIGINAL_FINAL_M|donorBit))continue;
        if(!PhysicalBlackRGB(CurrentRGB.Load(int3(p,0)).rgb,Hook.x))continue;
        int2 tangent=int2(-direction.y,direction.x);
        uint referenceState=0u;uint3 sharedA=0u,sharedC=0u;
        [loop]for(int side=-1;side<=1;side+=2) {
            bool complete;int3 low,high,referenceLow,referenceHigh;
            bool fits=SCStencil(q,direction,side*tangent,Hook.x,referenceState,sharedA,sharedC,
                                complete,low,high,referenceLow,referenceHigh);
            if(!complete)continue;
            if(!fits)return; // Never cherry-pick another stencil after contradiction.
            pooledLow=min(pooledLow,low);pooledHigh=max(pooledHigh,high);supplied=true;
            observedLow=min(observedLow,referenceLow);observedHigh=max(observedHigh,referenceHigh);
        }
    }
    if(!supplied)return;
    uint3 code=PhysicalRGB8(CurrentRGB.Load(int3(q,0)).rgb);
    uint bank=PhysicalModelBank(Hook.x);
    uint3 kl=PhysicalUnpack(CBCMeta[bank+11u]),kh=PhysicalUnpack(CBCMeta[bank+12u]);
    uint3 ql=SCHalfLow(code),qh=SCHalfHigh(code),bl=SCHalfLow(kl),bh=SCHalfHigh(kh);
    if(!StrokeRgbIntervalVisibleCells(ql,qh,bl,bh,pooledLow,pooledHigh))return;
    // Original M0 is not an independent clean-background certificate: a missed
    // AA sample can otherwise manufacture a slope and apparent absorption at q.
    // Retain the raw observed reference colours as background null alternatives
    // alongside every affine continuation. The original affine test remains a
    // conjunct, so this uncertainty guard can only withdraw old permissions.
    // Both hypotheses use the unchanged common-RGB/all-feasible opacity bound.
    if(!StrokeRgbIntervalVisibleCells(ql,qh,bl,bh,min(pooledLow,observedLow),max(pooledHigh,observedHigh)))return;
    // Outward stencils do not exhaust the observed background alternatives.
    // Retain EVERY eligible adjacent original-M0 colour as an additional null,
    // including lateral observations that need not fit the affine hypothesis.
    // The two previous tests remain conjuncts: this can only withdraw offers.
    int3 adjacentLow=min(pooledLow,observedLow),adjacentHigh=max(pooledHigh,observedHigh);
    [loop]for(int ny=-1;ny<=1;++ny)[loop]for(int nx=-1;nx<=1;++nx) {
        if(nx==0&&ny==0)continue;
        uint3 neighbor;
        if(!SCReference(q+int2(nx,ny),Hook.x,neighbor))continue;
        adjacentLow=min(adjacentLow,(int3)SCHalfLow(neighbor));
        adjacentHigh=max(adjacentHigh,(int3)SCHalfHigh(neighbor));
    }
    if(!StrokeRgbIntervalVisibleCells(ql,qh,bl,bh,adjacentLow,adjacentHigh))return;
    // Optional export records this model's OWN original-M0 permission so alpha
    // errors cannot be hidden by union with old M. It never gates selection or
    // supplies a donor. C7's one commit still enforces the shared sticky fault.
    uint ignored;
    if(SCExportOffers())InterlockedOr(CoverageOffers[pixel],1u<<Hook.x,ignored);
    InterlockedOr(Flags[Id(pixel)],PIXEL_PENDING,ignored);
}
#endif
