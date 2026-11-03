// PRIVATE upstream seed-only colour model. No CBC/F2 black-predicate changes.
// Include after LegacyBlackNeighborAt; existing CurrentWideProduct is reused.
static const uint BLACK_SEED_TAG=0x314b4c42u;
groupshared uint GBlackPixel[512];
groupshared uint GBlackRGB[512];
groupshared uint GBlackHistogram[256];
groupshared uint GBlackKnown;
groupshared uint3 GBlackMinimum;
groupshared uint3 GBlackMaximum;
uint BlackPack(uint3 c) { return c.x|(c.y<<8u)|(c.z<<16u); }
uint3 BlackUnpack(uint p) { return uint3(p&255u,(p>>8u)&255u,(p>>16u)&255u); }
uint3 BlackRGB8(float3 c) {
    precise float3 scaled=c*255.0f;
    precise float3 rounded=scaled+0.5f;
    return (uint3)floor(rounded);
}
void StoreBlackMinimum(uint index,int2 p,float3 own,uint ownPeak) {
    uint best=0xffffffffu,packed=0u;
    static const int2 directions[4]={int2(-2,0),int2(2,0),int2(0,-2),int2(0,2)};
    [unroll]for(uint d=0u;d<4u;++d) {
        int2 q=p+directions[d];float3 value;
        if(!LegacyBlackNeighborAt(q,own,ownPeak)||!LoadDisplayColor(q,value))continue;
        uint peak=MaxChannel(PaletteQuantize(value));
        if(peak<best){best=peak;packed=BlackPack(BlackRGB8(value));}
    }
    if(best!=0xffffffffu){GBlackPixel[index]=(uint)p.y*contour_source.x+(uint)p.x+1u;GBlackRGB[index]=packed;}
}
bool FirstBlackAnchor(uint i) {
    uint id=GBlackPixel[i];if(id==0u)return false;
    // The fixed grid is a Cartesian monotone x/y product. Equal-pixel runs
    // therefore repeat in the preceding grid-column and/or preceding row.
    if(i%64u!=0u){if(GBlackPixel[i-1u]==id)return false;}
    if(i>=64u){if(GBlackPixel[i-64u]==id)return false;}
    return true;
}
uint3 BlackMultiply64x32(uint2 a,uint b) {
    uint l0,h0,l1,h1;CurrentWideProduct(a.x,b,l0,h0);CurrentWideProduct(a.y,b,l1,h1);
    uint mid=h0+l1;return uint3(l0,mid,h1+(mid<h0?1u:0u));
}
int BlackCompare96(uint3 a,uint3 b) {
    if(a.z!=b.z)return a.z>b.z?1:-1;
    if(a.y!=b.y)return a.y>b.y?1:-1;
    if(a.x!=b.x)return a.x>b.x?1:-1;return 0;
}
// Shared exact Otsu split: 0=no split, 1=ambiguous, 2=one dark cluster.
uint FindBlackDarkCluster(uint n,uint total,out uint bestCount,out uint bestCut) {
    bestCount=0u;bestCut=0u;
    uint n0=0u,s0=0u,bestDenominator=0u;
    uint2 bestNumerator=0u;bool tied=false;
    [loop]for(uint cut=0u;cut<255u;++cut) {
        uint h=GBlackHistogram[cut];
        if(h==0u)continue;
        n0+=h;s0+=cut*h;
        if(n0==n)break;
        uint lhs=s0*n,rhs=total*n0,delta=lhs>rhs?lhs-rhs:rhs-lhs;
        uint lo,hi;CurrentWideProduct(delta,delta,lo,hi);uint2 numerator=uint2(lo,hi);
        uint denominator=n0*(n-n0);
        int comparison=bestDenominator==0u?1:BlackCompare96(BlackMultiply64x32(numerator,bestDenominator),BlackMultiply64x32(bestNumerator,denominator));
        if(comparison>0){bestNumerator=numerator;bestDenominator=denominator;bestCount=n0;bestCut=cut;tied=false;}
        else if(comparison==0&&n0!=bestCount)tied=true;
    }
    return bestDenominator==0u?0u:tied?1u:2u;
}
// Recovery for compressed-video outlines whose exact RGB values vary. The
// existing contrast-qualified current anchors remain the sole sample source.
// One unique dark histogram cluster must cover a strict majority of sampled
// columns; schema2 records minority sample support without adding history.
// The private wire remains 16 words: votes/columns mean cluster supporters here.
bool TryClusterBlackModel(uint n,uint total) {
    uint bestCount,bestCut;
    if(FindBlackDarkCluster(n,total,bestCount,bestCut)!=2u)return false;
    uint clusterX=0u;
    [loop]for(uint gx=0u;gx<64u;++gx) {
        bool supported=false;
        [unroll]for(uint gy=0u;gy<8u;++gy) {
            uint i=gy*64u+gx;
            if(FirstBlackAnchor(i)&&MaxChannel(BlackUnpack(GBlackRGB[i]))<=bestCut)supported=true;
        }
        if(supported)++clusterX;
    }
    if(clusterX<2u||2u*clusterX<=PaletteWrite[55u])return false;
    uint3 low=255u,high=0u;uint prototype=0xffffffffu;
    [loop]for(uint k=0u;k<512u;++k)if(FirstBlackAnchor(k)) {
        uint code=GBlackRGB[k];uint3 c=BlackUnpack(code);
        if(MaxChannel(c)<=bestCut){low=min(low,c);high=max(high,c);prototype=min(prototype,code);}
    }
    if(prototype==0xffffffffu)return false;
    // Schema2 honestly records a column-supported minority dark cluster.
    PaletteWrite[49u]=2u*bestCount<=n?2u:1u;
    PaletteWrite[50u]=1u;PaletteWrite[51u]=1u;PaletteWrite[53u]=prototype;
    PaletteWrite[54u]=bestCount;PaletteWrite[56u]=clusterX;
    PaletteWrite[57u]=bestCount;PaletteWrite[58u]=bestCut;
    PaletteWrite[59u]=BlackPack(low);PaletteWrite[60u]=BlackPack(high);
    return true;
}

void LearnBlackModel() {
    // Lane0 only, after the existing sample-store group barrier. No new pass.
    [unroll]for(uint r=48u;r<64u;++r)PaletteWrite[r]=0u;
    PaletteWrite[48u]=BLACK_SEED_TAG;PaletteWrite[49u]=1u;PaletteWrite[51u]=2u;PaletteWrite[61u]=512u;
    uint n=0u,prototype=0u,balance=0u,total=0u;
    [loop]for(uint i=0u;i<512u;++i)if(FirstBlackAnchor(i)) {
        uint c=GBlackRGB[i],peak=MaxChannel(BlackUnpack(c));
        ++n;total+=peak;++GBlackHistogram[peak];
        if(balance==0u){prototype=c;balance=1u;}
        else if(c==prototype)++balance;else --balance;
    }
    PaletteWrite[52u]=n;if(n==0u)return;
    uint votes=0u,allX=0u,prototypeX=0u;
    [loop]for(uint j=0u;j<512u;++j)if(FirstBlackAnchor(j)&&GBlackRGB[j]==prototype)++votes;
    [loop]for(uint gx=0u;gx<64u;++gx) {
        bool anyX=false,sameX=false;
        [unroll]for(uint gy=0u;gy<8u;++gy){uint i=gy*64u+gx;if(FirstBlackAnchor(i)){anyX=true;if(GBlackRGB[i]==prototype)sameX=true;}}
        if(anyX)++allX;if(sameX)++prototypeX;
    }
    PaletteWrite[53u]=prototype;PaletteWrite[54u]=votes;PaletteWrite[55u]=allX;PaletteWrite[56u]=prototypeX;
    if(2u*votes<=n){PaletteWrite[51u]=3u;TryClusterBlackModel(n,total);return;}
    if(prototypeX<2u||2u*prototypeX<=allX){PaletteWrite[51u]=4u;TryClusterBlackModel(n,total);return;}
    uint bestCount,bestCut;
    uint split=FindBlackDarkCluster(n,total,bestCount,bestCut);
    if(split==0u){
        // Schema3: contrast-qualified identical literal-black anchors need no
        // brighter population. Keep generic uniform dark/chromatic samples UNKNOWN.
        uint literalPeak=MaxChannel(BlackUnpack(prototype));
        if(n>=2u&&votes==n&&allX>=2u&&prototypeX==allX&&literalPeak<=1u){
            PaletteWrite[49u]=3u;PaletteWrite[50u]=1u;PaletteWrite[51u]=1u;
            PaletteWrite[57u]=n;PaletteWrite[58u]=literalPeak;
            PaletteWrite[59u]=prototype;PaletteWrite[60u]=prototype;return;
        }
        PaletteWrite[51u]=5u;return;
    }
    if(split==1u){PaletteWrite[51u]=6u;return;}
    uint3 low=255u,high=0u,proto=BlackUnpack(prototype);
    [loop]for(uint k=0u;k<512u;++k)if(FirstBlackAnchor(k)) {
        uint3 c=BlackUnpack(GBlackRGB[k]);if(MaxChannel(c)<=bestCut){low=min(low,c);high=max(high,c);}
    }
    if(MaxChannel(proto)>bestCut||any(proto<low)||any(proto>high)){PaletteWrite[51u]=7u;return;}
    PaletteWrite[50u]=1u;PaletteWrite[51u]=1u;PaletteWrite[57u]=bestCount;PaletteWrite[58u]=bestCut;
    PaletteWrite[59u]=BlackPack(low);PaletteWrite[60u]=BlackPack(high);
}
bool LoadBlackModel() {
    GBlackKnown=0u;GBlackMinimum=0u;GBlackMaximum=0u;
    uint known=PaletteRead[50u],status=PaletteRead[51u],n=PaletteRead[52u],prototype=PaletteRead[53u],votes=PaletteRead[54u],xs=PaletteRead[55u],px=PaletteRead[56u];
    if(PaletteRead[48u]!=BLACK_SEED_TAG||(PaletteRead[49u]!=1u&&PaletteRead[49u]!=2u&&PaletteRead[49u]!=3u)||known>1u||n>512u||votes>n||xs>64u||px>xs||xs>n||(prototype&0xff000000u)!=0u||PaletteRead[61u]!=512u||PaletteRead[62u]!=0u||PaletteRead[63u]!=0u)return false;
    if(known==0u) {
        if(PaletteRead[49u]!=1u)return false;
        if(status<2u||status>7u)return false;
        if(status==2u){if(n!=0u||prototype!=0u||votes!=0u||xs!=0u||px!=0u)return false;}
        else {
            if(n==0u||votes==0u||xs==0u||px==0u)return false;
            if(status==3u&&2u*votes>n)return false;
            if(status>=4u&&2u*votes<=n)return false;
            if(status==4u&&px>=2u&&2u*px>xs)return false;
            if(status>=5u&&(px<2u||2u*px<=xs))return false;
        }
        if(any(uint4(PaletteRead[57u],PaletteRead[58u],PaletteRead[59u],PaletteRead[60u])!=0u))return false;
        return true;
    }
    uint count=PaletteRead[57u],cut=PaletteRead[58u],low=PaletteRead[59u],high=PaletteRead[60u];
    uint3 p=BlackUnpack(prototype),a=BlackUnpack(low),b=BlackUnpack(high);
    if(PaletteRead[49u]==3u) {
        if(status!=1u||n<2u||xs<2u||px!=xs||votes!=n||count!=n||
           low!=prototype||high!=prototype||MaxChannel(p)>1u||cut!=MaxChannel(p))return false;
    } else if(PaletteRead[49u]==1u) {
    if(status!=1u||n==0u||xs==0u||px<2u||2u*votes<=n||2u*px<=xs||count==0u||count>=n||count<votes||cut>=255u||(low&0xff000000u)!=0u||(high&0xff000000u)!=0u||any(a>b)||any(p<a)||any(p>b)||MaxChannel(b)>cut)return false;
    } else {
        if(status!=1u||n==0u||xs==0u||px<2u||votes<2u||px>votes||2u*votes>n||2u*px<=xs||count!=votes||count>=n||cut>=255u||(low&0xff000000u)!=0u||(high&0xff000000u)!=0u||any(a>b)||any(p<a)||any(p>b)||MaxChannel(b)>cut)return false;
    }
    GBlackKnown=1u;GBlackMinimum=a;GBlackMaximum=b;return true;
}
bool BlackNeighborAt(int2 neighbor,float3 own,uint representativePeak) {
    if(!LegacyBlackNeighborAt(neighbor,own,representativePeak))return false;
    if(GBlackKnown==0u)return true;
    float3 rgb;if(!LoadDisplayColor(neighbor,rgb))return false;
    uint3 code=BlackRGB8(rgb);return all(code>=GBlackMinimum)&&all(code<=GBlackMaximum);
}
