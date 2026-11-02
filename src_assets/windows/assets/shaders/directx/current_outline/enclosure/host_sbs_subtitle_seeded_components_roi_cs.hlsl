// PRIVATE RESEARCH ONLY. Exact seeded four-connected colour-component primitive.
// One binary colour mode/current-unit domain per invocation. No palette, OCR,
// temporal, glyph, or drawing authority is produced by this mathematical operator.
// Ordered init -> union -> reduce -> resolve -> dilate dispatches are required.
// All inputs, scratch and outputs are distinct resources; zero groups hold them.

cbuffer SubtitleSeededComponentsConstants : register(b0) {
    uint4 component_shape; // width, height, width*height, reserved zero
};

Texture2D<uint> Eligible : register(t0);
Texture2D<uint> Seeds : register(t1);
Texture2D<uint> DomainTag : register(t2);
Texture2D<uint> CoverClip : register(t3);
// PRIVATE CBC physical-graph stage only. Raw/Inside bind explicit NULL at t5.
// This is the already-authenticated CURRENT native S; it never changes binary Seeds.
Texture2D<uint> CurrentPhysicalSupport : register(t5);
globallycoherent RWStructuredBuffer<uint> Parent : register(u0);
globallycoherent RWStructuredBuffer<uint> RootFlags : register(u1);
RWTexture2D<uint> Body : register(u2);
RWTexture2D<uint> Final : register(u3);

static const uint INVALID = 0xffffffffu;
static const uint HAS_SEED = 1u;
static const uint TOUCHES_BOUNDARY = 2u;
groupshared uint ShapeValid;
groupshared uint BodyWidth;
groupshared uint BodyHeight;
groupshared uint FinalWidth;
groupshared uint FinalHeight;
groupshared uint ParentWords;
groupshared uint FlagsWords;
groupshared uint PublicationFault;

void CheckShape() {
    uint ew,eh,sw,sh,dw,dh,cw,ch,bw,bh,fw,fh,pw,ps,rw,rs;
    Eligible.GetDimensions(ew,eh); Seeds.GetDimensions(sw,sh);
    DomainTag.GetDimensions(dw,dh); CoverClip.GetDimensions(cw,ch);
    Body.GetDimensions(bw,bh); Final.GetDimensions(fw,fh);
    Parent.GetDimensions(pw,ps); RootFlags.GetDimensions(rw,rs);
    BodyWidth=bw; BodyHeight=bh; FinalWidth=fw; FinalHeight=fh;
    ParentWords=pw; FlagsWords=rw;
    bool valid=component_shape.x!=0u && component_shape.y!=0u;
    uint count=0u;
    if (valid) {
        valid=component_shape.y<=0xfffffffeu/component_shape.x;
        if (valid) count=component_shape.x*component_shape.y;
    }
    ShapeValid=valid && component_shape.z==count && component_shape.w==0u &&
        ew==component_shape.x && eh==component_shape.y &&
        sw==ew && sh==eh && dw==ew && dh==eh && cw==ew && ch==eh &&
        bw==ew && bh==eh && fw==ew && fh==eh &&
        pw==count+1u && ps==4u && rw==count+1u && rs==4u?1u:0u;
}

bool InBounds(uint2 p) { return p.x<component_shape.x && p.y<component_shape.y; }
uint PixelId(uint2 p) { return p.y*component_shape.x+p.x+1u; }

uint AtomicParent(uint id) {
    uint value; InterlockedCompareExchange(Parent[id],0u,0u,value); return value;
}

void MarkFault() {
    if (ParentWords!=0u) { uint ignored; InterlockedExchange(Parent[0u],INVALID,ignored); }
}

uint FindAtomic(uint start) {
    uint node=start;
    [allow_uav_condition] while (true) {
        uint parent=AtomicParent(node);
        // Root0 is never a valid component and cannot merge disconnected borders.
        if (parent==0u || parent==INVALID || parent>node) { MarkFault();return INVALID; }
        if (parent==node) {
            uint ignored; InterlockedMin(Parent[start],parent,ignored); return parent;
        }
        uint grandparent=AtomicParent(parent);
        if (grandparent==0u || grandparent==INVALID || grandparent>parent) { MarkFault();return INVALID; }
        uint shortened; InterlockedMin(Parent[node],grandparent,shortened); node=parent;
    }
    // FXC requires a syntactic return after an unbounded UAV-conditioned loop.
    MarkFault(); return INVALID;
}

void Join(uint first,uint second) {
    [allow_uav_condition] while (true) {
        uint a=FindAtomic(first),b=FindAtomic(second);
        if (a==INVALID || b==INVALID || a==b) return;
        uint high=max(a,b),low=min(a,b),observed;
        InterlockedCompareExchange(Parent[high],high,low,observed);
        if (observed==high) return;
        // Competing CAS lowered a root. Retry current roots; no lock, group spin,
        // fixed propagation rounds, or assumption about workgroup scheduling.
    }
}

bool AtDomainBoundary(uint2 p,uint domain) {
    if (p.x==0u || p.y==0u || p.x+1u==component_shape.x || p.y+1u==component_shape.y) return true;
    return DomainTag.Load(int3(p+uint2(1u,0u),0))!=domain || DomainTag.Load(int3(p-uint2(1u,0u),0))!=domain ||
        DomainTag.Load(int3(p+uint2(0u,1u),0))!=domain || DomainTag.Load(int3(p-uint2(0u,1u),0))!=domain;
}

void BeginPublication(uint lane) {
    if (lane==0u) {
        CheckShape(); PublicationFault=1u;
        if (ShapeValid!=0u) PublicationFault=AtomicParent(0u)==0u?0u:1u;
    }
}

[numthreads(16,16,1)]
void init_main(uint3 dispatch:SV_DispatchThreadID,uint lane:SV_GroupIndex) {
    if (lane==0u) CheckShape();
    GroupMemoryBarrierWithGroupSync();
    if (all(dispatch==0u)) {
        if (ParentWords!=0u) Parent[0u]=0u;
        if (FlagsWords!=0u) RootFlags[0u]=0u;
    }
    if (ShapeValid==0u || !InBounds(dispatch.xy)) return;
    uint2 p=dispatch.xy; uint id=PixelId(p);
    Parent[id]=Eligible.Load(int3(p,0))==1u && DomainTag.Load(int3(p,0))!=0u?id:INVALID;
    RootFlags[id]=0u; Body[p]=0u; Final[p]=0u;
}

[numthreads(16,16,1)]
void union_main(uint3 dispatch:SV_DispatchThreadID,uint lane:SV_GroupIndex) {
    if (lane==0u) CheckShape();
    GroupMemoryBarrierWithGroupSync();
    if (ShapeValid==0u) { if (all(dispatch==0u)) MarkFault();return; }
    if (!InBounds(dispatch.xy)) return;
    uint2 p=dispatch.xy;
    if (Eligible.Load(int3(p,0))>1u || Seeds.Load(int3(p,0))>1u || CoverClip.Load(int3(p,0))>1u) { MarkFault();return; }
    uint domain=DomainTag.Load(int3(p,0));
    if (domain==0u || Eligible.Load(int3(p,0))!=1u) return;
    uint id=PixelId(p);
    if (p.x!=0u) {
        uint2 left=p-uint2(1u,0u);
        if (DomainTag.Load(int3(left,0))==domain && Eligible.Load(int3(left,0))==1u) Join(id,PixelId(left));
    }
    if (p.y!=0u) {
        uint2 up=p-uint2(0u,1u);
        if (DomainTag.Load(int3(up,0))==domain && Eligible.Load(int3(up,0))==1u) Join(id,PixelId(up));
    }
}

[numthreads(16,16,1)]
void reduce_main(uint3 dispatch:SV_DispatchThreadID,uint lane:SV_GroupIndex) {
    BeginPublication(lane); GroupMemoryBarrierWithGroupSync();
    if (ShapeValid==0u || PublicationFault!=0u || !InBounds(dispatch.xy)) return;
    uint2 p=dispatch.xy; uint domain=DomainTag.Load(int3(p,0));
    if (domain==0u || Eligible.Load(int3(p,0))!=1u) return;
    uint root=FindAtomic(PixelId(p));
    if (root==INVALID) return;
    uint flags=(Seeds.Load(int3(p,0))==1u?HAS_SEED:0u) | (AtDomainBoundary(p,domain)?TOUCHES_BOUNDARY:0u);
    if (flags!=0u) { uint ignored; InterlockedOr(RootFlags[root],flags,ignored); }
}

[numthreads(16,16,1)]
void resolve_main(uint3 dispatch:SV_DispatchThreadID,uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    BeginPublication(lane); GroupMemoryBarrierWithGroupSync();
    if (ShapeValid==0u) {
        if (any(group!=0u)) return;
        [loop] for (uint bodyY=dispatch.y;bodyY<BodyHeight;bodyY+=16u) {
            [loop] for (uint bodyX=dispatch.x;bodyX<BodyWidth;bodyX+=16u) Body[uint2(bodyX,bodyY)]=1u;
        }
        return;
    }
    if (!InBounds(dispatch.xy)) return;
    uint2 p=dispatch.xy; uint value=PublicationFault!=0u?1u:0u;
    if (PublicationFault==0u && DomainTag.Load(int3(p,0))!=0u && Eligible.Load(int3(p,0))==1u) {
        uint root=FindAtomic(PixelId(p));
        if (root!=INVALID) value=(RootFlags[root]&3u)==HAS_SEED?1u:0u;
    }
    Body[p]=value;
}

[numthreads(16,16,1)]
void dilate_main(uint3 dispatch:SV_DispatchThreadID,uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    BeginPublication(lane); GroupMemoryBarrierWithGroupSync();
    if (ShapeValid==0u) {
        if (any(group!=0u)) return;
        [loop] for (uint finalY=dispatch.y;finalY<FinalHeight;finalY+=16u) {
            [loop] for (uint finalX=dispatch.x;finalX<FinalWidth;finalX+=16u) Final[uint2(finalX,finalY)]=1u;
        }
        return;
    }
    if (!InBounds(dispatch.xy)) return;
    uint2 p=dispatch.xy; uint value=PublicationFault!=0u?1u:0u;
    if (PublicationFault==0u && CoverClip.Load(int3(p,0))==1u) {
        // Exact native Euclidean radius3 disk:29 offsets, no repeated growth.
        [unroll] for (int offsetY=-3;offsetY<=3;++offsetY) {
            [unroll] for (int offsetX=-3;offsetX<=3;++offsetX) {
                int2 q=int2(p)+int2(offsetX,offsetY);
                if (offsetX*offsetX+offsetY*offsetY<=9 && all(q>=0) && q.x<int(component_shape.x) && q.y<int(component_shape.y))
                    value|=Body[q];
            }
        }
    }
    Final[p]=value;
}

// Separate fixed separable-filter support geometry. The original circular entry
// above remains byte-identical; callers choose one final pass, never both.
[numthreads(16,16,1)]
void dilate_box_main(uint3 dispatch:SV_DispatchThreadID,uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    BeginPublication(lane); GroupMemoryBarrierWithGroupSync();
    if (ShapeValid==0u) {
        if (any(group!=0u)) return;
        [loop] for (uint boxFinalY=dispatch.y;boxFinalY<FinalHeight;boxFinalY+=16u) {
            [loop] for (uint boxFinalX=dispatch.x;boxFinalX<FinalWidth;boxFinalX+=16u) Final[uint2(boxFinalX,boxFinalY)]=1u;
        }
        return;
    }
    if (!InBounds(dispatch.xy)) return;
    uint2 p=dispatch.xy; uint value=PublicationFault!=0u?1u:0u;
    if (PublicationFault==0u && CoverClip.Load(int3(p,0))==1u) {
        // Exactly49 offsets max(abs(dx),abs(dy))<=3 from selected Body. This is
        // one dilation of actual components, not OCR rectangle filling.
        [unroll] for (int boxOffsetY=-3;boxOffsetY<=3;++boxOffsetY) {
            [unroll] for (int boxOffsetX=-3;boxOffsetX<=3;++boxOffsetX) {
                int2 q=int2(p)+int2(boxOffsetX,boxOffsetY);
                if (all(q>=0) && q.x<int(component_shape.x) && q.y<int(component_shape.y)) value|=Body[q];
            }
        }
    }
    Final[p]=value;
}


// PRIVATE bounded-dispatch derivative. Original six-entry source above is byte-exact.
// Only the local dispatch origin/extent change; graph IDs and four-connected math stay native.
cbuffer EnclosureRoiHook : register(b1) { uint4 enclosure_roi_hook; };
StructuredBuffer<uint> EnclosureMeta : register(t4);
groupshared uint RoiValid;
groupshared uint2 RoiOrigin;
groupshared uint2 RoiExtent;
groupshared uint CurrentPhysicalSupportReady;

void CheckRoiShape() {
    CheckShape();RoiValid=0u;RoiOrigin=0u;RoiExtent=0u;
    uint n,stride;EnclosureMeta.GetDimensions(n,stride);
    if(ShapeValid==0u||n!=512u||stride!=4u)return;
    if(enclosure_roi_hook.x>=8u||any(enclosure_roi_hook.yzw))return;
    if(EnclosureMeta[0u]!=0x31434e45u||EnclosureMeta[1u]!=5u||EnclosureMeta[3u]!=0u||EnclosureMeta[9u]!=1u||EnclosureMeta[4u]!=1u||EnclosureMeta[5u]!=1u)return;
    if(any(uint2(EnclosureMeta[44u],EnclosureMeta[45u])!=component_shape.xy))return;
    uint base=128u+32u*enclosure_roi_hook.x;
    if(EnclosureMeta[base+29u]!=1u)return;
    uint4 r=uint4(EnclosureMeta[base+24u],EnclosureMeta[base+25u],EnclosureMeta[base+26u],EnclosureMeta[base+27u]);
    if(any(r.xy>=r.zw)||any(r.zw>component_shape.xy))return;
    RoiOrigin=r.xy;RoiExtent=r.zw-r.xy;RoiValid=1u;
}

[numthreads(16,16,1)]
void init_roi_main(uint3 dispatch:SV_DispatchThreadID,uint lane:SV_GroupIndex) {
    if(lane==0u)CheckRoiShape();
    GroupMemoryBarrierWithGroupSync();
    if(RoiValid==0u) { if(all(dispatch==0u))MarkFault();return; }
    // Local origin, not native(0,0), owns the per-unit graph-fault reset.
    if(all(dispatch==0u)) { Parent[0u]=0u;RootFlags[0u]=0u; }
    if(dispatch.z!=0u||any(dispatch.xy>=RoiExtent))return;
    uint2 p=dispatch.xy+RoiOrigin;uint id=PixelId(p);
    Parent[id]=Eligible.Load(int3(p,0))==1u&&DomainTag.Load(int3(p,0))!=0u?id:INVALID;
    RootFlags[id]=0u;Body[p]=0u;Final[p]=0u;
}

[numthreads(16,16,1)]
void union_roi_main(uint3 dispatch:SV_DispatchThreadID,uint lane:SV_GroupIndex) {
    if(lane==0u)CheckRoiShape();
    GroupMemoryBarrierWithGroupSync();
    if(RoiValid==0u) { if(all(dispatch==0u))MarkFault();return; }
    if(dispatch.z!=0u||any(dispatch.xy>=RoiExtent))return;
    uint2 p=dispatch.xy+RoiOrigin;
    if(Eligible.Load(int3(p,0))>1u||Seeds.Load(int3(p,0))>1u||CoverClip.Load(int3(p,0))>1u) { MarkFault();return; }
    uint domain=DomainTag.Load(int3(p,0));
    if(domain==0u||Eligible.Load(int3(p,0))!=1u)return;
    uint id=PixelId(p);
    if(p.x!=0u) {
        uint2 left=p-uint2(1u,0u);
        if(DomainTag.Load(int3(left,0))==domain&&Eligible.Load(int3(left,0))==1u)Join(id,PixelId(left));
    }
    if(p.y!=0u) {
        uint2 up=p-uint2(0u,1u);
        if(DomainTag.Load(int3(up,0))==domain&&Eligible.Load(int3(up,0))==1u)Join(id,PixelId(up));
    }
}

[numthreads(16,16,1)]
void reduce_roi_main(uint3 dispatch:SV_DispatchThreadID,uint lane:SV_GroupIndex) {
    if(lane==0u) {
        CheckRoiShape();PublicationFault=1u;
        if(RoiValid!=0u)PublicationFault=AtomicParent(0u)==0u?0u:1u;
        uint sw,sh;CurrentPhysicalSupport.GetDimensions(sw,sh);
        CurrentPhysicalSupportReady=sw==component_shape.x&&sh==component_shape.y?1u:0u;
    }
    GroupMemoryBarrierWithGroupSync();
    if(RoiValid==0u||PublicationFault!=0u||dispatch.z!=0u||any(dispatch.xy>=RoiExtent))return;
    uint2 p=dispatch.xy+RoiOrigin;uint domain=DomainTag.Load(int3(p,0));
    if(domain==0u||Eligible.Load(int3(p,0))!=1u)return;
    uint root=FindAtomic(PixelId(p));if(root==INVALID)return;
    uint flags=(Seeds.Load(int3(p,0))==1u?HAS_SEED:0u)|(AtDomainBoundary(p,domain)?TOUCHES_BOUNDARY:0u);
    // Bit4 is independent evidence, never exterior/boundary authority. The next
    // resolve/count/select dispatch may consume it after this reduction completes.
    if(CurrentPhysicalSupportReady!=0u&&CurrentPhysicalSupport.Load(int3(p,0))==1u)flags|=4u;
    if(flags!=0u) { uint ignored;InterlockedOr(RootFlags[root],flags,ignored); }
}
