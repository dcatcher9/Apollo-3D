#ifndef HOST_SBS_CURRENT_STROKE_MATH_SHARED_HLSL
#define HOST_SBS_CURRENT_STROKE_MATH_SHARED_HLSL
// Pure arithmetic only. Source quantization, model validation, ownership and
// foreground/reference permissions remain with each current-frame caller.
uint StrokeBoxDistance(uint3 q,uint3 low,uint3 high) {
    int3 below=max(int3(low)-int3(q),0),above=max(int3(q)-int3(high),0);
    int3 d=max(below,above);return(uint)max(d.r,max(d.g,d.b));
}
bool StrokeBlackLikeDecision(float3 rgb,uint3 code,uint3 low,uint3 high,uint dK,uint dF) {
    bool physical=max(rgb.r,max(rgb.g,rgb.b))<=1.0f/255.0f||all(code>=low)&&all(code<=high);
    return physical||dK<=dF;
}
uint3 StrokeHalfLow(uint3 v) { return 2u*v-(uint3)(v>0u); }
uint3 StrokeHalfHigh(uint3 v) { return min(uint3(510u,510u,510u),2u*v+1u); }

// Signed A*s>=B for one common background fraction s=1-alpha. These RGB
// endpoints are doubled code cells in [0,510]; cross-products <=260100.
// Luma's wider endpoints require its separate wide-product implementation.
bool StrokeConstrainBackgroundFraction(int coefficient,int rhs,
                                      inout uint ln,inout uint ld,inout uint un,inout uint ud) {
    if(coefficient>0) {
        if(rhs>0) {
            uint n=(uint)rhs,d=(uint)coefficient;
            if(n*ld>ln*d){ln=n;ld=d;}
        }
    } else if(coefficient<0) {
        if(rhs>0)return false;
        uint n=(uint)(-rhs),d=(uint)(-coefficient);
        if(n*ud<un*d){un=n;ud=d;}
    } else if(rhs>0)return false;
    return ln*ud<=un*ld;
}
bool StrokeRgbIntervalVisibleCells(uint3 qlo,uint3 qhi,uint3 blackLow,uint3 blackHigh,
                                   int3 backgroundLow,int3 backgroundHigh) {
    uint ln=0u,ld=1u,un=1u,ud=1u;
    [unroll]for(uint channel=0u;channel<3u;++channel) {
        if(!StrokeConstrainBackgroundFraction(backgroundHigh[channel]-(int)blackHigh[channel],
             (int)qlo[channel]-(int)blackHigh[channel],ln,ld,un,ud))return false;
        if(!StrokeConstrainBackgroundFraction((int)blackLow[channel]-backgroundLow[channel],
             (int)blackLow[channel]-(int)qhi[channel],ln,ld,un,ud))return false;
    }
    // The original ALL-feasible-opacity>=5/255 policy, including nonempty fit.
    return ln*ud<=un*ld&&un*255u<=ud*250u;
}
#endif
