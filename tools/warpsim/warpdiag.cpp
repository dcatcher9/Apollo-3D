// Diagnostic renderer: for each output pixel of the mode-0 (current shader) warp,
// encode WHICH path produced it:
//   red   = no crossing found -> background fill
//   green = crossing found, brightness ~ bestDepth
//   blue  = number of crossings detected (x64)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
static const int DW=1596,DH=672,SW=800,SH=400,SOX=2000,SOY=900,OW=640,OH=320,EOX=1600,EOY=720;
static std::vector<unsigned char> g_depth;
static const float DIVERGENCE=0.015f,FOCAL=0.5f,DEPTH_SCALE=0.9f,BORDER_FADE=0.02f,DEPTH_FLOOR=0.25f;
static const int STEPS=24; static const float EYE_SIGN=1.0f;
static float clampf(float v,float lo,float hi){return v<lo?lo:(v>hi?hi:v);}
static float depth_bilinear(float u,float v){
  float fx=u*DW-0.5f,fy=v*DH-0.5f;int x0=(int)floorf(fx),y0=(int)floorf(fy);
  float tx=fx-x0,ty=fy-y0;int x1=x0+1,y1=y0+1;
  x0=std::max(0,std::min(DW-1,x0));x1=std::max(0,std::min(DW-1,x1));
  y0=std::max(0,std::min(DH-1,y0));y1=std::max(0,std::min(DH-1,y1));
  float d00=g_depth[y0*DW+x0]/255.0f,d10=g_depth[y0*DW+x1]/255.0f;
  float d01=g_depth[y1*DW+x0]/255.0f,d11=g_depth[y1*DW+x1]/255.0f;
  return (d00*(1-tx)+d10*tx)*(1-ty)+(d01*(1-tx)+d11*tx)*ty;}
static float sample_depth(float u,float v){
  float ox=0.75f/DW,oy=0.75f/DH;
  return 0.25f*(depth_bilinear(u-ox,v-oy)+depth_bilinear(u+ox,v-oy)+depth_bilinear(u-ox,v+oy)+depth_bilinear(u+ox,v+oy));}
static float border_fade(float x){return clampf(std::min(x,1.0f-x)/BORDER_FADE,0,1);}
static float depth_parallax(float d,float x){d=DEPTH_FLOOR+(1.0f-DEPTH_FLOOR)*d;return (d-FOCAL)*DEPTH_SCALE*DIVERGENCE*border_fade(x);}
int main(){
  FILE*f=fopen("depth_1596x672.bin","rb");g_depth.resize((size_t)DW*DH);fread(g_depth.data(),1,g_depth.size(),f);fclose(f);
  std::vector<unsigned char> out((size_t)OW*OH*3);
  for(int py=0;py<OH;py++){
    float vy=(EOY+py+0.5f)/1728.0f;
    for(int px=0;px<OW;px++){
      float ux=(EOX+px+0.5f)/4096.0f;
      float r=DIVERGENCE*DEPTH_SCALE*std::max(FOCAL,1.0f-FOCAL);
      float startX=ux-r,stepX=2.0f*r/STEPS;
      float bestDepth=-1.0f;int ncross=0;
      float prevX=startX,prevD=sample_depth(prevX,vy);
      float prevG=(prevX-ux)-EYE_SIGN*depth_parallax(prevD,prevX);
      for(int i=1;i<=STEPS;i++){
        float x=startX+stepX*i,d=sample_depth(x,vy);
        float g=(x-ux)-EYE_SIGN*depth_parallax(d,x);
        if((prevG<=0&&g>=0)||(prevG>=0&&g<=0)){
          float denom=g-prevG,t=(fabsf(denom)>1e-6f)?clampf(-prevG/denom,0,1):0.0f;
          float crossD=prevD+(d-prevD)*t;
          if(crossD>bestDepth)bestDepth=crossD;
          ncross++;
        }
        prevX=x;prevD=d;prevG=g;
      }
      unsigned char*o=&out[((size_t)py*OW+px)*3];
      o[0]=(bestDepth<0)?255:0;                              // red: bg fill
      o[1]=(bestDepth<0)?0:(unsigned char)lroundf(bestDepth*255); // green: winner depth
      o[2]=(unsigned char)std::min(255,ncross*64);           // blue: crossing count
    }
  }
  f=fopen("diag.raw","wb");fwrite(out.data(),1,out.size(),f);fclose(f);
  printf("diag done\n");return 0;
}
