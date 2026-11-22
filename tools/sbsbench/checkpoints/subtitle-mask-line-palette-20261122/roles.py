import numpy as np
from scipy.ndimage import gaussian_filter
import normalization as norm
TARGET=28

def role_fields(rgb, engine):
    low=norm.normalize_region(rgb,TARGET)
    probs=np.zeros((*low.shape[:2],4),'f4')
    tiles=0
    for start,stop,owned,tile in norm.normalized_tiles(low,512,144):
        tensor=np.ascontiguousarray(tile.transpose(2,0,1)[None],dtype='f4')/255
        z=engine(tensor)[0];exp=np.exp(z-z.max(0));post=exp/exp.sum(0)
        probs[:,start:stop]=post.transpose(1,2,0)[owned];tiles+=1
    return low,probs,tiles

def lift(field,hw,y,x):
    h,w=hw;fh,fw=field.shape[:2]
    x0,x1,wx=norm._native_axis(w,h,TARGET,fw);y0,y1,wy=norm._native_axis(h,h,TARGET,fh)
    a=field[y0[y],x0[x]];b=field[y0[y],x1[x]];c=field[y1[y],x0[x]];d=field[y1[y],x1[x]]
    up=a+wx[x,None]*(b-a);down=c+wx[x,None]*(d-c)
    return up+wy[y,None]*(down-up)

def palette(rgb,probs):
    h,w=rgb.shape[:2]
    step=max(1,round(h/96))
    ys,xs=np.mgrid[0:h:step,0:w:step];y,x=ys.ravel(),xs.ravel()
    color=rgb[y,x].astype('f4')/16
    idx=np.rint(color).astype('i4')@np.array([289,17,1],'i4')
    ps=lift(probs,(h,w),y,x)
    bins=np.stack([np.bincount(idx,weights=p,minlength=4913).reshape(17,17,17) for p in ps.T]).astype('f4')
    bins=gaussian_filter(bins,(0,.65,.65,.65),mode='constant')
    strength=bins.sum(0)
    posterior=(bins+.12)/(strength+.48)
    colors=[]
    for k in (1,2):
        # A frequent color needs both class specificity and evidence from that class.
        ranking=bins[k]*posterior[k]
        idx3=np.unravel_index(np.argmax(ranking),ranking.shape)
        colors.append((np.array(idx3,'f4')*16).clip(0,255))
    return posterior,colors
