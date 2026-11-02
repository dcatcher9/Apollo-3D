// PRIVATE RESEARCH ONLY. Exact independent-domain four-connected hole fill.
// This is a mathematical primitive, not subtitle or drawing authority.
// Dispatch init -> union -> resolve in command order, using the full native grid.
// Inputs and scratch/output resources must not alias. No CPU verdict or round limit.

cbuffer SubtitleTopologyConstants : register(b0) {
    uint4 topology_shape; // width, height, width*height, reserved zero
};

Texture2D<uint> KnownBarrier : register(t0);
Texture2D<uint> DomainTag : register(t1);
globallycoherent RWStructuredBuffer<uint> Parent : register(u0);
RWTexture2D<uint> Enclosed : register(u1);

static const uint BARRIER = 0xffffffffu;
groupshared uint ShapeValid;
groupshared uint OutputWidth;
groupshared uint OutputHeight;
groupshared uint ParentWords;
groupshared uint ResolveFault;

void CheckShape() {
    uint knownWidth,knownHeight,domainWidth,domainHeight,outputWidth,outputHeight,parentWords,stride;
    KnownBarrier.GetDimensions(knownWidth,knownHeight);
    DomainTag.GetDimensions(domainWidth,domainHeight);
    Enclosed.GetDimensions(outputWidth,outputHeight);
    Parent.GetDimensions(parentWords,stride);
    OutputWidth=outputWidth;OutputHeight=outputHeight;ParentWords=parentWords;
    bool productValid=topology_shape.x!=0u && topology_shape.y!=0u;
    uint count=0u;
    if (productValid) {
        productValid=topology_shape.y<=0xfffffffeu/topology_shape.x;
        if (productValid) count=topology_shape.x*topology_shape.y;
    }
    ShapeValid=productValid && topology_shape.z==count && topology_shape.w==0u &&
        knownWidth==topology_shape.x && knownHeight==topology_shape.y &&
        domainWidth==knownWidth && domainHeight==knownHeight &&
        OutputWidth==knownWidth && OutputHeight==knownHeight &&
        ParentWords==count+1u && stride==4u?1u:0u;
}

bool Position(uint2 p) { return p.x<topology_shape.x && p.y<topology_shape.y; }
uint PixelId(uint2 p) { return p.y*topology_shape.x+p.x+1u; }

uint AtomicParent(uint id) {
    uint value;
    InterlockedCompareExchange(Parent[id],0u,0u,value);
    return value;
}

void MarkFault() {
    // Only UNION or an impossible/corrupt parent path sets the invalid sentinel.
    // INIT clears root0 before UNION starts. No other pass clears it afterward.
    if (ParentWords!=0u) {
        uint ignored;
        InterlockedExchange(Parent[0u],BARRIER,ignored);
    }
}

uint FindAtomic(uint start) {
    uint node=start;
    [allow_uav_condition] while (true) {
        uint parent=AtomicParent(node);
        if (parent==BARRIER || parent>node) { MarkFault();return BARRIER; }
        if (parent==0u || parent==node) {
            // root0 is virtual and immutable on valid paths. Do not atomicMin
            // Parent[0]: it also holds the persistent invalid-input sentinel.
            if (start!=0u) {
                uint compressed;
                InterlockedMin(Parent[start],parent,compressed);
            }
            return parent;
        }
        uint grandparent=AtomicParent(parent);
        if (grandparent==BARRIER || grandparent>parent) { MarkFault();return BARRIER; }
        uint shortened;
        InterlockedMin(Parent[node],grandparent,shortened);
        node=parent;
    }
    // FXC does not prove that the loop above returns on every terminating path.
    MarkFault();
    return BARRIER;
}

void Join(uint first,uint second) {
    [allow_uav_condition] while (true) {
        uint firstRoot=FindAtomic(first);
        uint secondRoot=FindAtomic(second);
        if (firstRoot==BARRIER || secondRoot==BARRIER || firstRoot==secondRoot) return;
        uint high=max(firstRoot,secondRoot),low=min(firstRoot,secondRoot);
        uint observed;
        InterlockedCompareExchange(Parent[high],high,low,observed);
        if (observed==high) return;
        // Another operation lowered this root. Retry its current roots rather
        // than waiting for a workgroup, a lock holder, or a fixed iteration count.
    }
}

bool Exterior(uint2 p,uint domain) {
    if (p.x==0u || p.y==0u || p.x+1u==topology_shape.x || p.y+1u==topology_shape.y) return true;
    return DomainTag.Load(int3(p+uint2(1u,0u),0))!=domain ||
        DomainTag.Load(int3(p-uint2(1u,0u),0))!=domain ||
        DomainTag.Load(int3(p+uint2(0u,1u),0))!=domain ||
        DomainTag.Load(int3(p-uint2(0u,1u),0))!=domain;
}

[numthreads(16,16,1)]
void init_main(uint3 dispatch:SV_DispatchThreadID,uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    if (lane==0u) CheckShape();
    GroupMemoryBarrierWithGroupSync();
    if (all(dispatch==0u) && ParentWords!=0u) Parent[0u]=0u;
    if (ShapeValid==0u || !Position(dispatch.xy)) return;
    uint2 p=dispatch.xy;
    uint domain=DomainTag.Load(int3(p,0));
    uint known=KnownBarrier.Load(int3(p,0));
    uint value=0u;
    if (domain!=0u) {
        if (known==1u) value=BARRIER;
        else if (known!=0u) value=BARRIER-1u; // UNION validates every numeric input.
        else if (!Exterior(p,domain)) value=PixelId(p);
    }
    Parent[PixelId(p)]=value;
}

[numthreads(16,16,1)]
void union_main(uint3 dispatch:SV_DispatchThreadID,uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    if (lane==0u) CheckShape();
    GroupMemoryBarrierWithGroupSync();
    if (ShapeValid==0u) {
        if (all(dispatch==0u)) MarkFault();
        return;
    }
    if (!Position(dispatch.xy)) return;
    uint2 p=dispatch.xy;
    uint known=KnownBarrier.Load(int3(p,0));
    if (known>1u) { MarkFault();return; }
    uint domain=DomainTag.Load(int3(p,0));
    if (domain==0u || known==1u) return;
    uint id=PixelId(p);
    if (p.x!=0u) {
        uint2 left=p-uint2(1u,0u);
        if (DomainTag.Load(int3(left,0))==domain && KnownBarrier.Load(int3(left,0))==0u) Join(id,PixelId(left));
    }
    if (p.y!=0u) {
        uint2 up=p-uint2(0u,1u);
        if (DomainTag.Load(int3(up,0))==domain && KnownBarrier.Load(int3(up,0))==0u) Join(id,PixelId(up));
    }
}

[numthreads(16,16,1)]
void resolve_main(uint3 dispatch:SV_DispatchThreadID,uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    if (lane==0u) {
        CheckShape();
        ResolveFault=1u;
        if (ShapeValid!=0u) ResolveFault=AtomicParent(0u)==0u?0u:1u;
    }
    GroupMemoryBarrierWithGroupSync();
    if (ShapeValid==0u) {
        // Invalid geometry cannot choose the output stride or justify pruning.
        // A caller must submit at least one group; an unusable output is rejected.
        if (any(group!=0u)) return;
        [loop] for (uint fallbackY=dispatch.y;fallbackY<OutputHeight;fallbackY+=16u) {
            [loop] for (uint fallbackX=dispatch.x;fallbackX<OutputWidth;fallbackX+=16u)
                Enclosed[uint2(fallbackX,fallbackY)]=1u;
        }
        return;
    }
    if (!Position(dispatch.xy)) return;
    uint2 p=dispatch.xy;
    uint value=1u;
    if (ResolveFault==0u) {
        uint domain=DomainTag.Load(int3(p,0));
        if (domain==0u) value=0u;
        else if (KnownBarrier.Load(int3(p,0))==0u) {
            uint root=FindAtomic(PixelId(p));
            value=root!=0u?1u:0u;
        }
    }
    Enclosed[p]=value;
}
