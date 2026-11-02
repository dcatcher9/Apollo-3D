// PRIVATE ONE CPU-prototype-matched push/pull+erase1 trial. No new protection M.
// SourceColor is the actual immutable draw source in its original working RGB domain.
#include "include/depth_color.hlsl"
cbuffer ActualContour : register(b0) {
    uint4 CSource; uint4 CField; uint4 CContent;
    uint4 CFrame; uint4 CObservation; uint4 COptions;
};
cbuffer BackgroundRequest : register(b1) {
    uint4 Lease[4];
    uint4 Shape; // W, fullH, cropTop, bandH; GPU joined to actual contour CB.
    uint4 PoolInfo; // paddedW, paddedH, total cells, levels.
    uint4 Level; // source level, destination level,0,0.
    uint4 Mips[16]; // width,height,float4 offset,0.
};
cbuffer ActualSubtitleRender : register(b2) {
    uint4 RenderControl; uint4 RenderSource;
};
Texture2D<float4> SourceColor : register(t0);
Texture2D<float> BaseParallax : register(t1);
StructuredBuffer<float4> ParallaxState : register(t2);
Texture2D<uint> NativeGlyphMatte : register(t3);
Texture2D<float4> CompletedDisplayBand : register(t4);
StructuredBuffer<uint> SubtitleCondition : register(t5);
StructuredBuffer<uint> ContourIdentity : register(t6);
StructuredBuffer<uint> SelectedMatteReceipt : register(t7);
RWStructuredBuffer<float4> Pool : register(u0);
RWTexture2D<float4> Background : register(u1);
globallycoherent RWStructuredBuffer<uint> Proof : register(u2);
RWByteAddressBuffer Arguments : register(u3);
static const uint TAG=0x31504742u, SCHEMA=1u;
static const uint F_JOIN=1u,F_COVERAGE=2u,F_NO_BACKGROUND=4u,F_MATTE=8u,F_NUMERIC=16u;

bool Has64(uint2 v){return any(v!=0u);}
uint CBWord(uint i) {
    if(i<4u)return CSource[i];if(i<8u)return CField[i-4u];
    if(i<12u)return CContent[i-8u];if(i<16u)return CFrame[i-12u];
    if(i<20u)return CObservation[i-16u];return COptions[i-20u];
}
bool CurrentKnown(out bool empty,out uint2 sourceSize) {
    empty=false; sourceSize=0u;
    uint n,s;ContourIdentity.GetDimensions(n,s);if(n!=32u||s!=4u)return false;
    SelectedMatteReceipt.GetDimensions(n,s);if(n!=64u||s!=4u)return false;
    SubtitleCondition.GetDimensions(n,s);if(n!=6u||s!=4u)return false;
    ParallaxState.GetDimensions(n,s);if(n<4u||s!=16u)return false;
    if(RenderControl.x!=1u||RenderControl.y>2u||any(RenderControl.zw!=CFrame.zw))return false;
    if(any(Shape!=CSource)||any(CSource.xy==0u)||any(CSource.xy>5120u)||
       CSource.z>=CSource.y||CSource.w!=CSource.y-CSource.z||
       any(CField.xy==0u)||CField.z>=CField.w||CField.w>CField.y||
       any(CContent.xy>=CContent.zw)||any(CContent.zw>CField.xy)||
       !Has64(CFrame.xy)||!Has64(CObservation.xy)||CObservation.z>1u||CObservation.w!=0u||
       any(COptions!=uint4(1u,27u,0u,0u)))return false;
    SourceColor.GetDimensions(sourceSize.x,sourceSize.y);
    if(any(sourceSize==0u)||any(RenderSource.zw!=CSource.xy)||
       any(RenderSource.xy>sourceSize)||any(CSource.xy>sourceSize-RenderSource.xy))return false;
    uint w,h;NativeGlyphMatte.GetDimensions(w,h);if(any(uint2(w,h)!=CSource.xw))return false;
    CompletedDisplayBand.GetDimensions(w,h);if(any(uint2(w,h)!=CSource.xw))return false;
    BaseParallax.GetDimensions(w,h);if(any(uint2(w,h)!=CField.xy))return false;
    Background.GetDimensions(w,h);if(any(uint2(w,h)!=CSource.xw))return false;
    if(ContourIdentity[0u]!=5u||ContourIdentity[1u]!=0x35544353u||ContourIdentity[2u]>1u||
       ContourIdentity[22u]!=27u||ContourIdentity[23u]!=1u)return false;
    [unroll]for(uint j=0u;j<4u;++j) {
        if(ContourIdentity[4u+j]!=CFrame[j]||ContourIdentity[8u+j]!=CSource[j]||
           ContourIdentity[12u+j]!=CField[j]||ContourIdentity[16u+j]!=CContent[j])return false;
    }
    if(any(uint2(ContourIdentity[20u],ContourIdentity[21u])!=CObservation.xy))return false;
    [unroll]for(uint k=24u;k<32u;++k)if(ContourIdentity[k]!=0u)return false;
    if(any(asuint(ParallaxState[3u].xy)!=CObservation.xy)||
       ContourIdentity[3u]!=asuint(ParallaxState[1u].z))return false;
    if(any(uint4(SelectedMatteReceipt[0],SelectedMatteReceipt[1],SelectedMatteReceipt[2],SelectedMatteReceipt[3])!=uint4(0x31534d53u,1u,3u,0u))||
       SelectedMatteReceipt[62u]!=2u||SelectedMatteReceipt[63u]!=0u||
       !Has64(uint2(SelectedMatteReceipt[60u],SelectedMatteReceipt[61u])))return false;
    [loop]for(uint a=0u;a<24u;++a)if(SelectedMatteReceipt[4u+a]!=CBWord(a))return false;
    [unroll]for(uint b=0u;b<32u;++b)if(SelectedMatteReceipt[28u+b]!=ContourIdentity[b])return false;
    if(SubtitleCondition[0]!=4u||SubtitleCondition[1]!=0x34504353u)return false;
    uint count=SubtitleCondition[2],kinds=SubtitleCondition[3],active=SubtitleCondition[4];
    float plane=asfloat(SubtitleCondition[5]);
    if(count>8u||(kinds&~((1u<<count)-1u))!=0u||!isfinite(plane)||plane<0.0f||plane>0.04f)return false;
    empty=ContourIdentity[2u]==0u;
    if(empty){if(count!=0u||kinds!=0u||active!=0u)return false;}
    else if(count==0u||active!=2u)return false;
    if(!Has64(Lease[0].xy)||!Has64(Lease[0].zw)||!Has64(Lease[1].xy)||
       !Has64(Lease[1].zw)||!Has64(Lease[2].xy)||!Has64(Lease[2].zw)||
       !Has64(Lease[3].xy)||!Has64(Lease[3].zw))return false;
    if(PoolInfo.w==0u||PoolInfo.w>16u||Mips[0].x<Shape.x||Mips[0].y<Shape.w||
       any(Mips[0].xy!=PoolInfo.xy)||Mips[0].z!=0u||any(Level.zw))return false;
    uint cells=0u;
    [loop]for(uint m=0u;m<PoolInfo.w;++m) {
        uint4 v=Mips[m];if(any(v.xy==0u)||v.z!=cells||v.w!=0u)return false;
        if(m>0u&&any(v.xy!=max(Mips[m-1u].xy/2u,1u)))return false;
        cells+=v.x*v.y;
    }
    if(any(Mips[PoolInfo.w-1u].xy!=1u)||cells!=PoolInfo.z)return false;
    Pool.GetDimensions(n,s);return n==cells&&s==16u;
}

float3 DisplayRoundTrip(float3 raw) {
    float3 d=DepthColorToSrgb(raw,RenderControl.y);
    return float3(f16tof32(f32tof16(d.r)),f16tof32(f32tof16(d.g)),f16tof32(f32tof16(d.b)));
}
bool CurrentInk(int2 local) {
    if(any(local<0)||any(local>=int2(Shape.xw)))return false;
    if(NativeGlyphMatte.Load(int3(local,0))!=1u)return false;
    float4 ref=CompletedDisplayBand.Load(int3(local,0));
    float3 raw=SourceColor.Load(int3(local+int2(RenderSource.xy)+int2(0,(int)Shape.z),0)).rgb;
    if(ref.a!=1.0f||any(!isfinite(ref.rgb))||any(ref.rgb<0.0f)||any(ref.rgb>1.0f)||any(!isfinite(raw)))return false;
    float3 display=DisplayRoundTrip(raw);
    return all(isfinite(display))&&all(abs(display-ref.rgb)<=0.08f);
}
void StoreArgs(uint slot,uint2 size,bool enabled) {
    Arguments.Store3(slot*12u,enabled?uint3((size+15u)/16u,1u):uint3(0u,0u,0u));
}
[numthreads(1,1,1)]void prepare_main(uint3 id:SV_DispatchThreadID) {
    [unroll]for(uint i=0u;i<96u;++i)Proof[i]=0u;
    [unroll]for(uint a=0u;a<32u;++a)Arguments.Store3(a*12u,uint3(0u,0u,0u));
    Proof[0]=TAG;Proof[1]=SCHEMA;Proof[62]=1u;
    [unroll]for(uint l=0u;l<16u;++l)Proof[4u+l]=Lease[l/4u][l%4u];
    bool empty;uint2 sourceSize;if(!CurrentKnown(empty,sourceSize)){Proof[3]=F_JOIN;return;}
    [unroll]for(uint j=0u;j<32u;++j)Proof[20u+j]=ContourIdentity[j];
    [unroll]for(uint k=0u;k<4u;++k){Proof[52u+k]=RenderControl[k];Proof[56u+k]=RenderSource[k];}
    Proof[60]=SelectedMatteReceipt[60];Proof[61]=SelectedMatteReceipt[61];
    Proof[64]=sourceSize.x;Proof[65]=sourceSize.y;Proof[66]=Shape.x;Proof[67]=Shape.w;
    Proof[68]=Shape.z;Proof[69]=RenderSource.x;Proof[70]=RenderSource.y;Proof[71]=RenderControl.y;
    Proof[77]=Shape.x*Shape.w;Proof[78]=PoolInfo.z;Proof[79]=PoolInfo.w;
    if(empty){Proof[2]=2u;return;}
    Proof[2]=3u; // internal pending; never PS-usable before seal.
    for(uint m=0u;m<PoolInfo.w;++m)StoreArgs(m,Mips[m].xy,true);
    StoreArgs(PoolInfo.w,uint2(1u,1u),true);
    for(uint p=0u;p+1u<PoolInfo.w;++p)StoreArgs(PoolInfo.w+1u+p,Mips[PoolInfo.w-2u-p].xy,true);
}
groupshared uint InitCounts[7];
[numthreads(16,16,1)]void init_main(uint3 id:SV_DispatchThreadID,uint li:SV_GroupIndex) {
    if(li<7u)InitCounts[li]=0u;
    GroupMemoryBarrierWithGroupSync();
    if(Proof[2]==3u&&all(id.xy<Mips[0].xy)) {
    uint index=id.y*Mips[0].x+id.x;
    if(any(id.xy>=Shape.xw))Pool[index]=0.0f;
    else {
    int2 q=int2(id.xy);uint matte=NativeGlyphMatte.Load(int3(q,0));
    if(matte>1u)InterlockedOr(Proof[3],F_MATTE);
    float3 raw=SourceColor.Load(int3(q+int2(RenderSource.xy)+int2(0,(int)Shape.z),0)).rgb;
    bool finite=all(isfinite(raw));bool own=CurrentInk(q);
    bool rawNear=false,erase=false;
    [unroll]for(int dy=-1;dy<=1;++dy)[unroll]for(int dx=-1;dx<=1;++dx) {
        int2 r=q+int2(dx,dy);
        if(all(r>=0)&&all(r<int2(Shape.xw))) {
            rawNear=rawNear||NativeGlyphMatte.Load(int3(r,0))!=0u;
            if(!erase)erase=CurrentInk(r);
        }
    }
    bool unknown=matte!=0u&&!own;
    erase=erase&&!unknown;
    bool valid=!rawNear&&matte==0u&&finite;
    // Only level0 uses negative tag storage. Decode max(w,0) before every P/W operation.
    float tag=valid?1.0f:(unknown?-2.0f:(erase?-1.0f:0.0f));
    Pool[index]=float4(valid?raw:0.0f,tag);
    if(valid)InterlockedAdd(InitCounts[0],1u);
    if(own)InterlockedAdd(InitCounts[1],1u);
    if(erase)InterlockedAdd(InitCounts[2],1u);
    if(unknown)InterlockedAdd(InitCounts[3],1u);
    if(!finite)InterlockedAdd(InitCounts[4],1u);
    if(matte==1u)InterlockedAdd(InitCounts[5],1u);
    }
    }
    GroupMemoryBarrierWithGroupSync();
    if(li<5u&&InitCounts[li]!=0u)InterlockedAdd(Proof[72u+li],InitCounts[li]);
    if(li==5u&&InitCounts[5]!=0u)InterlockedAdd(Proof[82],InitCounts[5]);
}
[numthreads(16,16,1)]void down_main(uint3 id:SV_DispatchThreadID) {
    if(Proof[2]!=3u||Level.x>=PoolInfo.w||Level.y>=PoolInfo.w||Level.y!=Level.x+1u)return;
    uint4 src=Mips[Level.x],dst=Mips[Level.y];if(any(id.xy>=dst.xy))return;
    uint2 scale=uint2(src.x>1u?2u:1u,src.y>1u?2u:1u);
    float4 sum=0.0f;
    [unroll]for(uint y=0u;y<2u;++y)[unroll]for(uint x=0u;x<2u;++x) {
        if(x<scale.x&&y<scale.y) {
            uint2 p=id.xy*scale+uint2(x,y);float4 v=Pool[src.z+p.y*src.x+p.x];
            sum+=float4(v.rgb,max(v.a,0.0f));
        }
    }
    Pool[dst.z+id.y*dst.x+id.x]=sum/(float)(scale.x*scale.y);
}
[numthreads(1,1,1)]void coarse_main(uint3 id:SV_DispatchThreadID) {
    if(Proof[2]!=3u)return;
    uint offset=Mips[PoolInfo.w-1u].z;float4 v=Pool[offset];
    if(v.a<=0.0f||any(!isfinite(v))){InterlockedOr(Proof[3],F_NO_BACKGROUND);Pool[offset]=0.0f;return;}
    Pool[offset]=float4(v.rgb/v.a,1.0f);Proof[81]=1u;
}
float3 CoarseBilinear(uint4 fine,uint4 coarse,uint2 q) {
    float2 pos=(float2(q)+0.5f)*(float2(coarse.xy)/float2(fine.xy))-0.5f;
    int2 low=int2(floor(pos));float2 f=pos-float2(low);int2 hi=int2(coarse.xy)-1;
    int2 a=clamp(low,int2(0,0),hi),b=clamp(low+int2(1,0),int2(0,0),hi);
    int2 c=clamp(low+int2(0,1),int2(0,0),hi),d=clamp(low+int2(1,1),int2(0,0),hi);
    float3 top=Pool[coarse.z+a.y*coarse.x+a.x].rgb*(1.0f-f.x)+Pool[coarse.z+b.y*coarse.x+b.x].rgb*f.x;
    float3 bottom=Pool[coarse.z+c.y*coarse.x+c.x].rgb*(1.0f-f.x)+Pool[coarse.z+d.y*coarse.x+d.x].rgb*f.x;
    return top*(1.0f-f.y)+bottom*f.y;
}
groupshared uint PublishedCount;
[numthreads(16,16,1)]void pull_main(uint3 id:SV_DispatchThreadID,uint li:SV_GroupIndex) {
    if(li==0u)PublishedCount=0u;
    GroupMemoryBarrierWithGroupSync();
    if(Proof[2]==3u&&Proof[81]==1u&&Level.x<PoolInfo.w&&Level.y+1u==Level.x) {
    uint4 fine=Mips[Level.y],coarse=Mips[Level.x];if(all(id.xy<fine.xy)) {
    uint at=fine.z+id.y*fine.x+id.x;float4 local=Pool[at];float W=max(local.a,0.0f);
    float3 colour=local.rgb+(1.0f-W)*CoarseBilinear(fine,coarse,id.xy);
    Pool[at]=float4(colour,1.0f);
    if(Level.y==0u&&all(id.xy<Shape.xw)) {
        int2 p=int2(id.xy)+int2(RenderSource.xy)+int2(0,(int)Shape.z);
        float3 raw=SourceColor.Load(int3(p,0)).rgb;bool erased=local.a==-1.0f;
        bool finite=all(isfinite(colour));
        float flag=erased?(finite?1.0f:-1.0f):0.0f; // unknown rawM1 is E0/original RGB, never whole-fragment fallback.
        Background[id.xy]=float4(erased&&finite?colour:raw,flag);
        if(erased&&!finite)InterlockedOr(Proof[3],F_NUMERIC);
        InterlockedAdd(PublishedCount,1u);
    }
    }
    }
    GroupMemoryBarrierWithGroupSync();
    if(li==0u&&PublishedCount!=0u)InterlockedAdd(Proof[80],PublishedCount);
}
[numthreads(1,1,1)]void seal_main(uint3 id:SV_DispatchThreadID) {
    if(Proof[2]!=3u)return;
    // Native1x1 cannot have both an erasure hole and a BG anchor; it stays unavailable.
    if(Proof[80]!=Proof[77])InterlockedOr(Proof[3],F_COVERAGE);
    Proof[2]=(Proof[3]==0u&&Proof[81]==1u)?1u:0u;
}
