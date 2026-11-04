// ONE private physical-topology union. BlackModelWord reads this invocation's
// immutable validated CBCMeta512. No model learning or new authority occurs here.
static const uint PHYSICAL_MODEL_TAG=0x314b4c42u;
uint PhysicalModelBank(uint unit) { return 384u+16u*unit; }
uint3 PhysicalUnpack(uint c) { return uint3(c&255u,(c>>8u)&255u,(c>>16u)&255u); }
uint3 PhysicalRGB8(float3 c) {
    precise float3 scaled=c*255.0f;
    precise float3 rounded=scaled+0.5f;
    return (uint3)floor(rounded);
}
bool PhysicalModelWordsValid(uint4 a,uint4 b,uint4 c,uint4 d) {
    uint known=a.z,status=a.w,n=b.x,prototype=b.y,votes=b.z,xs=b.w,px=c.x;
    if(a.x!=PHYSICAL_MODEL_TAG||(a.y!=1u&&a.y!=2u&&a.y!=3u&&a.y!=4u)||known>1u||n>512u||votes>n||xs>64u||px>xs||xs>n||
       (prototype&0xff000000u)!=0u||d.y!=512u||d.z!=0u||d.w!=0u)return false;
    if(known==0u) {
        if(a.y!=1u)return false;
        if(status<2u||status>7u)return false;
        if(status==2u){if(n!=0u||prototype!=0u||votes!=0u||xs!=0u||px!=0u)return false;}
        else {
            if(n==0u||votes==0u||xs==0u||px==0u)return false;
            if(status==3u&&2u*votes>n)return false;
            if(status>=4u&&2u*votes<=n)return false;
            if(status==4u&&px>=2u&&2u*px>xs)return false;
            if(status>=5u&&(px<2u||2u*px<=xs))return false;
        }
        return all(uint4(c.yzw,d.x)==0u);
    }
    uint count=c.y,cut=c.z,low=c.w,high=d.x;
    uint3 p=PhysicalUnpack(prototype),lo=PhysicalUnpack(low),hi=PhysicalUnpack(high);
    if(a.y==3u||a.y==4u) {
        // K4 is published only by completed CBC qualification: every original
        // BODY peer is enclosed and no candidate/literal-black touches its true
        // cover/native boundary. Its authority is opaque black in a current
        // own Raw4CC, never foreground/style, AA, closure, Inside or a cache.
        // It is never an upstream palette record.
        uint literalPeak=max(p.r,max(p.g,p.b));
        if(status!=1u||n<2u||xs<2u||px!=xs||votes!=n||count!=n||
           low!=prototype||high!=prototype||cut!=literalPeak||
           (a.y==3u?literalPeak>1u:literalPeak<=1u))return false;
    } else if(a.y==1u) {
    if(status!=1u||n==0u||xs==0u||px<2u||2u*votes<=n||2u*px<=xs||count==0u||count>=n||
       count<votes||cut>=255u||(low&0xff000000u)!=0u||(high&0xff000000u)!=0u||
       any(lo>hi)||any(p<lo)||any(p>hi)||max(hi.r,max(hi.g,hi.b))>cut)return false;
    } else {
        if(status!=1u||n==0u||xs==0u||px<2u||votes<2u||px>votes||2u*votes>n||2u*px<=xs||count!=votes||count>=n||
           cut>=255u||(low&0xff000000u)!=0u||(high&0xff000000u)!=0u||
           any(lo>hi)||any(p<lo)||any(p>hi)||max(hi.r,max(hi.g,hi.b))>cut)return false;
    }
    return true;
}
bool PhysicalModelRecordValid(uint unit) {
    if(unit>=8u)return false;uint s=PhysicalModelBank(unit);
    return PhysicalModelWordsValid(
        uint4(BlackModelWord(s),BlackModelWord(s+1u),BlackModelWord(s+2u),BlackModelWord(s+3u)),
        uint4(BlackModelWord(s+4u),BlackModelWord(s+5u),BlackModelWord(s+6u),BlackModelWord(s+7u)),
        uint4(BlackModelWord(s+8u),BlackModelWord(s+9u),BlackModelWord(s+10u),BlackModelWord(s+11u)),
        uint4(BlackModelWord(s+12u),BlackModelWord(s+13u),BlackModelWord(s+14u),BlackModelWord(s+15u)));
}
// Pure hot-path classification: CBC validates once at current unit snapshot;
// Rawbegin validates every evaluated snapshot once before any child geometry.
// The ordered owner retains this immutable bank through Raw/F2 completion.
bool PhysicalBlackCode(uint3 code,uint unit) {
    if(max(code.r,max(code.g,code.b))<=1u)return true;
    uint s=PhysicalModelBank(unit);
    if(BlackModelWord(s+2u)!=1u)return false; // Valid UNKNOWN, never malformed.
    uint3 low=PhysicalUnpack(BlackModelWord(s+11u)),high=PhysicalUnpack(BlackModelWord(s+12u));
    return all(code>=low)&&all(code<=high);
}
bool PhysicalBlackRGB(float3 rgb,uint unit) {
    // Preserve the exact original literal branch. Canonical RTZ/encoding proof
    // makes the equivalent RGB8 branch used by profiles/global exclusion agree.
    if(max(rgb.r,max(rgb.g,rgb.b))<=(1.0f/255.0f))return true;
    uint s=PhysicalModelBank(unit);
    if(BlackModelWord(s+2u)!=1u)return false;
    uint3 code=PhysicalRGB8(rgb),low=PhysicalUnpack(BlackModelWord(s+11u)),high=PhysicalUnpack(BlackModelWord(s+12u));
    return all(code>=low)&&all(code<=high);
}
