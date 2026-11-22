// Native subtitle score with half-scale context. The early publisher owns the selected matte.
cbuffer ModelRequest : register(b0) { uint4 Shape; uint4 Owner; uint4 ClockToken; }
cbuffer ActualContour : register(b1) {
    uint4 Source; uint4 Field; uint4 Content; uint4 Model; uint4 Observation; uint4 Options;
}
ByteAddressBuffer CurrentWork : register(t7);
RWByteAddressBuffer CurrentWorkOut : register(u7);
RWByteAddressBuffer GeometryArgs : register(u7);
#define BandSize (Shape.xy)
#define BandOrigin (uint2(0,Source.z))
#define ActiveSlots (CurrentWork.Load(68u))
#define Mode (CurrentWork.Load(212u))
#define Cutoff (Shape.z)
static const uint NativeCapacity=98304u;

Texture2D<float4> RGB : register(t0);
Texture2D<uint> CoverBits : register(t2);
Texture2D<uint> Body : register(t3);
Texture2D<uint> SelectedBits : register(t4);
Texture2D<uint> RawM : register(t5);
StructuredBuffer<float> W : register(t6);
ByteAddressBuffer Domain : register(t24);
Texture2D<uint> Geometry : register(t8);
Texture2D<uint> Floor : register(t9);
Texture2D<uint> Axial : register(t22);
StructuredBuffer<float> Native : register(t23);
RWTexture2D<uint> OutputMask : register(u0);
RWTexture2D<uint> AxialStore : register(u1);
RWByteAddressBuffer DomainStore : register(u2);
RWTexture2D<uint> OutputAlpha8 : register(u3);
RWTexture2D<uint> GeometryStore : register(u4);

RWByteAddressBuffer WorkCount : register(u6);
RWStructuredBuffer<float> NativeOut : register(u7);

static const uint Margin = 10u;
static const uint ValidBit=1u<<24u, RawBit=1u<<25u, HighBit=1u<<26u;
groupshared uint Tile[22*14];
groupshared uint Rows[2*14];
groupshared uint X0,Y0,X1,Y1;
groupshared float Stem[18*10*12];
groupshared float Pool[14*10*12];
groupshared float Low1[12*8*12];
groupshared float Low2[10*6*12];

bool Valid(int2 p,uint2 size) { return all(p>=0) && all(uint2(p)<size); }
uint Owners(int2 p) { return CoverBits.Load(int3(p,0))&ActiveSlots; }
uint Address(int2 p,uint2 size) { return uint(p.y)*size.x+uint(p.x); }
bool Covered(int2 p) {
    uint active=Owners(p);
    [unroll] for(uint i=0u;i<8u;++i) {
        if((active&(1u<<i))!=0u) {
            uint4 box=CurrentWork.Load4(80u+16u*i);
            if(all(uint2(p)>=box.xy)&&all(uint2(p)<box.zw))return true;
        }
    }
    return false;
}
uint4 ContextRect(uint4 cover) {
    uint2 lower=uint2(max(int2(cover.xy)-int(Margin),0))&~1u;
    return uint4(lower,min(cover.zw+Margin,BandSize));
}
bool InRect(uint2 p,uint4 box) { return all(p>=box.xy)&&all(p<box.zw); }
uint4 Crop() { return Domain.Load4(32u); }
#define Size (Crop().zw-Crop().xy)
#define Phase ((BandOrigin+Crop().xy)&1u)

void Fault(uint n) {
    uint previous;
    WorkCount.InterlockedOr(8u,n,previous);
}
uint3 Code(int2 p) {
    float4 a=RGB.Load(int3(p,0));
    if(a.a!=1.0f||!all(isfinite(a.rgb))||any(a.rgb<0.0f)||any(a.rgb>1.0f)) {
        Fault(1u);
        return 0;
    }
    return uint3(floor(saturate(a.rgb)*255.0f+0.5f));
}
uint PackTile(int2 p) {
    if(!Valid(p,BandSize))return 0u;
    uint3 c=Code(p);
    uint raw=RawM.Load(int3(p,0));
    uint selected=SelectedBits.Load(int3(p,0));
    uint owned=CoverBits.Load(int3(p,0));
    if(raw>1u||(owned&~ActiveSlots)!=0u||(selected>>16u)!=0u||((selected&255u)&~owned)!=0u)Fault(2u);
    uint high=selected>>8u;
    return c.r|(c.g<<8u)|(c.b<<16u)|ValidBit|(raw?RawBit:0u)|((high&owned&ActiveSlots)?HighBit:0u);
}
uint Dist(uint offset,uint2 p) {
    uint a=Rows[offset+p.y+3u];
    if(a&(1u<<(p.x+3u)))return 0u;
    a|=Rows[offset+p.y+2u]|Rows[offset+p.y+4u];
    if(a&(7u<<(p.x+2u)))return 1u;
    a|=Rows[offset+p.y+1u]|Rows[offset+p.y+5u];
    if(a&(31u<<(p.x+1u)))return 2u;
    a|=Rows[offset+p.y]|Rows[offset+p.y+6u];
    return (a&(127u<<p.x))?3u:4u;
}
int3 PackedRgb(uint a) { return int3(a&255u,(a>>8u)&255u,(a>>16u)&255u); }
uint Difference(uint a,uint b) {
    if((a&b&ValidBit)==0u)return 0u;
    int3 v=abs(PackedRgb(a)-PackedRgb(b));
    return uint(max(v.x,max(v.y,v.z)));
}
int SignedAxial(uint a,uint b,uint c) {
    if((a&b&c&ValidBit)==0u)return 0;
    int d=dot(PackedRgb(b)-PackedRgb(a),int3(1,2,1));
    int e=dot(PackedRgb(b)-PackedRgb(c),int3(1,2,1));
    return ((d>0&&e>0)||(d<0&&e<0))? sign(d)*min(abs(d),abs(e)):0;
}
bool Editable(int2 p,uint geometry) {
    return Covered(p) && Floor.Load(int3(p,0))==0u && (geometry&7u)<=3u && (geometry>>31u);
}
[numthreads(16,8,1)]
void geometry(uint3 group:SV_GroupID,uint3 lane:SV_GroupThreadID,uint index:SV_GroupIndex) {
    if(index==0u) {
        X0=BandSize.x;Y0=BandSize.y;X1=0u;Y1=0u;
        uint previous;WorkCount.InterlockedAdd(0u,1u,previous);
    }
    uint4 box=ContextRect(CurrentWork.Load4(16u));
    int2 origin=int2(box.xy+group.xy*uint2(16,8));
    for(uint cell=index;cell<308;cell+=128)
        Tile[cell]=PackTile(origin+int2(cell%22,cell/22)-3);
    GroupMemoryBarrierWithGroupSync();
    if(index<14u) {
        uint raw=0,high=0;
        for(uint x=0;x<22;++x) {
            uint v=Tile[index*22+x];
            if(v&RawBit)raw|=1u<<x;
            if(v&HighBit)high|=1u<<x;
        }
        Rows[index]=raw;Rows[index+14u]=high;
    }
    GroupMemoryBarrierWithGroupSync();
    int2 p=origin+int2(lane.xy);
    if(InRect(uint2(p),box)) {
        uint at=(lane.y+3u)*22u+lane.x+3u;
        uint c=Tile[at];uint rg=0;
        for(uint j=0;j<7;++j)rg+=countbits(Rows[lane.y+j]&(127u<<lane.x));
        uint dx=max(Difference(c,Tile[at-1u]),Difference(c,Tile[at+1u]));
        uint dy=max(Difference(c,Tile[at-22u]),Difference(c,Tile[at+22u]));
        uint g=Dist(0u,lane.xy)|(Dist(14u,lane.xy)<<3u)|(rg<<6u)|(dx<<12u)|(dy<<20u)|(1u<<31u);
        uint ax=uint(SignedAxial(Tile[at-1u],c,Tile[at+1u])+1020);
        uint ay=uint(SignedAxial(Tile[at-22u],c,Tile[at+22u])+1020);
        GeometryStore[p]=g;
        AxialStore[p]=ax|(ay<<11u);
        uint floor_value = Floor.Load(int3(p,0));
        if(floor_value>1u||Body.Load(int3(p,0))>1u) Fault(4u);
        // The publisher reads Candidate only for matching current slot bit and slot box.
        // One native center belongs to one thread even if slot boxes overlap.
        bool covered = Covered(p);
        if(covered) OutputMask[p] = floor_value==1u?1u:0u;
        if(covered && floor_value==0u && (g&7u)<=3u) {
            InterlockedMin(X0,uint(p.x));InterlockedMin(Y0,uint(p.y));
            InterlockedMax(X1,uint(p.x)+1u);InterlockedMax(Y1,uint(p.y)+1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if(index==0u&&X0<X1&&Y0<Y1) {
        uint previous;
        DomainStore.InterlockedMin(0u,X0,previous);
        DomainStore.InterlockedMin(4u,Y0,previous);
        DomainStore.InterlockedMax(8u,X1,previous);
        DomainStore.InterlockedMax(12u,Y1,previous);
    }
}
[numthreads(1,1,1)]
void init_domain() {
    DomainStore.Store4(0u,uint4(BandSize,0u,0u));
    DomainStore.Store4(16u,uint4(0u,1u,1u,0u));
    DomainStore.Store4(32u,0u);
    uint4 plan=uint4(0u,1u,1u,0u);
    if(Mode==2u&&ActiveSlots!=0u) {
        uint4 roi=CurrentWork.Load4(16u);
        uint4 box=ContextRect(roi);
        bool valid=all(roi.xy<roi.zw)&&all(box.xy<=roi.xy)&&all(box.zw>=roi.zw)&&all(box.zw<=BandSize);
        if(valid) {
            uint2 grid=(box.zw-box.xy+uint2(15u,7u))/uint2(16u,8u);
            plan=uint4(grid,1u,0u);
        }
    }
    DomainStore.Store4(64u,plan);
    GeometryArgs.Store4(0u,plan);
}
[numthreads(1,1,1)]
void extent() {
    if(CurrentWorkOut.Load(212u)!=2u)return;
    uint4 roi=CurrentWorkOut.Load4(16u);
    uint4 geo=ContextRect(roi);
    uint4 plan=DomainStore.Load4(64u);
    CurrentWorkOut.Store4(32u,plan);
    CurrentWorkOut.Store4(48u,geo);
    CurrentWorkOut.Store4(0u,uint4(0u,1u,1u,0u));
    if(any(geo.xy>roi.xy)||any(geo.zw<roi.zw)||any(geo.xy>=geo.zw)||any(geo.zw>BandSize)||
       any(plan.xy!=(geo.zw-geo.xy+uint2(15u,7u))/uint2(16u,8u))||plan.z!=1u) {
        Fault(64u);return;
    }
    uint4 bounds=DomainStore.Load4(0u);
    if(any(bounds.xy>=bounds.zw))return;
    uint4 crop = ContextRect(bounds);
    uint2 size=crop.zw-crop.xy;
    if(any(crop.xy<geo.xy)||any(crop.zw>geo.zw)) {Fault(64u);return;}
    if(size.x*size.y>NativeCapacity) {Fault(16u);return;}
    uint4 args=uint4((size+uint2(15u,7u))/uint2(16u,8u),1u,0u);
    DomainStore.Store4(16u,args);
    CurrentWorkOut.Store4(0u,args);
    DomainStore.Store4(32u,crop);
}
void ReadInput(int2 p,out float f[14]) {
    uint g=Geometry.Load(int3(p,0));
    uint a=Axial.Load(int3(p,0));
    uint3 c=Code(p);
    for(uint ch=0;ch<3u;++ch)f[ch]=float(c[ch])/255.0f;
    uint slots=Owners(p);
    f[3]=slots?1:0;
    f[4]=Body.Load(int3(p,0))?1:0;
    f[5]=((SelectedBits.Load(int3(p,0))>>8u)&slots)?1:0;
    f[6]=RawM.Load(int3(p,0))?1:0;
    f[7]=float(g&7u)*.25f;
    f[8]=float((g>>3u)&7u)*.25f;
    f[9]=float((g>>6u)&63u)/49.0f;
    f[10]=float((g>>12u)&255u)/255.0f;
    f[11]=float((g>>20u)&255u)/255.0f;
    f[12]=float(int(a&2047u)-1020)/1020.0f;
    f[13]=float(int((a>>11u)&2047u)-1020)/1020.0f;
}
[numthreads(16,8,1)]
void trunk(uint3 grp:SV_GroupID, uint3 tid:SV_GroupThreadID,uint lane:SV_GroupIndex) {
    if (lane==0u) {uint unused;WorkCount.InterlockedAdd(12u,1u,unused);}
    int2 origin=int2(grp.xy*uint2(16,8));
    for(uint cell=lane;cell<180;cell+=128) {
        int2 p=origin+int2(cell%18,cell/18)-1;
        bool live=Valid(p,Size);
        float f[14]; if(live)ReadInput(p+int2(Crop().xy),f);
        for(uint ch=0;ch<12;++ch) {
            precise float sum=0;
            if(live) {
                for(uint k=0;k<14;++k) sum+=f[k]*W[ch*14+k];
                sum=max(0,sum+W[168+ch]);
            }
            Stem[cell*12+ch]=sum;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    int2 p=origin+int2(tid.xy);
    if(!Valid(p,Size)) return;
    float local[12];
    for(uint ch=0;ch<12;++ch) {
        precise float sum=0;
        for(uint y=0;y<3;++y)
            for(uint x=0;x<3;++x)
                sum+=Stem[((tid.y+y)*18+tid.x+x)*12+ch]*W[180+ch*9+y*3+x];
        local[ch]=max(0,sum+W[288+ch]);
    }
    for(uint outc=0;outc<12;++outc) {
        precise float sum=0;
        for(uint k=0;k<12;++k) sum+=local[k]*W[300+outc*12+k];
        NativeOut[Address(p,Size)*12+outc]=Stem[((tid.y+1)*18+tid.x+1)*12+outc]+max(0,sum+W[444+outc]);
    }
}

float NativeAt(int2 p,uint ch) { return Valid(p,Size)?Native[Address(p,Size)*12+ch]:0; }

[numthreads(16,8,1)]
void finish(uint3 grp:SV_GroupID,uint3 tid:SV_GroupThreadID,uint lane:SV_GroupIndex) {
    uint2 expected = Domain.Load2(16u);
    bool trunk_done=WorkCount.Load(12u)==expected.x*expected.y;
    if (lane==0u && !trunk_done) Fault(32u);
    if(lane==0u) {uint unused;WorkCount.InterlockedAdd(4u,1u,unused);}
    int2 origin=int2(grp.xy*uint2(16,8));
    int2 loworigin=(origin+int2(Phase))/2-1;
    uint2 lowSize=(Size+Phase+1)/2;
    for(uint cell=lane;cell<140;cell+=128) {
        int2 q=loworigin+int2(cell%14,cell/14)-2;
        int2 p=2*q-int2(Phase);
        bool live=Valid(q,lowSize);
        for(uint ch=0;ch<12;++ch) {
            precise float v=0;
            if(live) {
                v+=NativeAt(p,ch);
                v+=NativeAt(p+int2(1,0),ch);
                v+=NativeAt(p+int2(0,1),ch);
                v+=NativeAt(p+int2(1,1),ch);
            }
            Pool[cell*12+ch]=v*.25;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for(uint cell=lane;cell<96;cell+=128) {
        uint2 r=uint2(cell%12,cell/12);
        bool live=Valid(loworigin+int2(r)-1,lowSize);
        float local[12];
        for(uint ch=0;ch<12;++ch) {
            precise float sum=0;
            if(live)
                for(uint y=0;y<3;++y)
                    for(uint x=0;x<3;++x)
                        sum+=Pool[((r.y+y)*14+r.x+x)*12+ch]*W[456+ch*9+y*3+x];
            local[ch]=max(0,sum+W[564+ch]);
        }
        for(uint ch=0;ch<12;++ch) {
            precise float sum=0;
            if(live)
                for(uint k=0;k<12;++k) sum+=local[k]*W[576+ch*12+k];
            Low1[cell*12+ch]=live?Pool[((r.y+1)*14+r.x+1)*12+ch]+max(0,sum+W[720+ch]):0;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for(uint cell=lane;cell<60;cell+=128) {
        uint2 r=uint2(cell%10,cell/10);
        bool live=Valid(loworigin+int2(r),lowSize);
        for(uint ch=0;ch<12;++ch) {
            precise float sum=0;
            if(live)
                for(uint y=0;y<3;++y)
                    for(uint x=0;x<3;++x)
                        sum+=Low1[((r.y+y)*12+r.x+x)*12+ch]*W[732+ch*9+y*3+x];
            Low2[cell*12+ch]=live?Low1[((r.y+1)*12+r.x+1)*12+ch]+max(0,sum+W[840+ch]):0;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    int2 p=origin+int2(tid.xy);
    if(!trunk_done||!Valid(p,Size))return;
    int2 global=p+int2(Crop().xy);
    if(!Editable(global,Geometry.Load(int3(global,0))))return;
    uint2 nativePhase=(uint2(p)+Phase)&1u;
    int2 low=(p+int2(Phase))/2-(1-int2(nativePhase));
    float2 blend=.75-.5*float2(nativePhase);
    int2 lo=clamp(low,0,int2(lowSize)-1)-loworigin;
    int2 hi=clamp(low+1,0,int2(lowSize)-1)-loworigin;
    float feature[24];
    for(uint ch=0;ch<12;++ch) {
        feature[ch]=Native[Address(p,Size)*12+ch];
        precise float a=(1-blend.x)*Low2[(lo.y*10+lo.x)*12+ch]+blend.x*Low2[(lo.y*10+hi.x)*12+ch];
        precise float b=(1-blend.x)*Low2[(hi.y*10+lo.x)*12+ch]+blend.x*Low2[(hi.y*10+hi.x)*12+ch];
        feature[ch+12]=(1-blend.y)*a+blend.y*b;
    }
    float j[12];
    for(uint ch=0;ch<12;++ch) {
        precise float sum=0;
        for(uint k=0;k<24;++k) sum+=feature[k]*W[852+ch*24+k];
        j[ch]=max(0,sum+W[1140+ch]);
    }
    float4 logit;
    for(uint ch=0;ch<4;++ch) {
        precise float sum=0;
        for(uint k=0;k<12;++k) sum+=j[k]*W[1152+ch*12+k];
        logit[ch]=sum+W[1200+ch];

    }
    float4 e=exp(logit-max(max(logit.x,logit.y),max(logit.z,logit.w)));
    precise float alpha=1-e.x/(e.x+e.y+e.z+e.w);
    uint q8=uint(round(alpha*255.0f));
    OutputAlpha8[global]=q8;
    OutputMask[global]=q8>=Cutoff?1u:0u;
}
