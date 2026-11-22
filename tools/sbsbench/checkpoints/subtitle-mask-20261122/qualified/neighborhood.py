import numpy as np
DIRECTIONS=((0,1),(1,0),(1,1),(1,-1))

def features(rgb,alpha):
    h,w=alpha.shape
    color=rgb.astype(np.float32)
    padded_color=np.pad(color,((1,1),(1,1),(0,0)),mode='edge')
    padded_alpha=np.pad(alpha,1,mode='edge')
    for dy,dx in DIRECTIONS:
        sa=np.s_[1+dy:1+dy+h,1+dx:1+dx+w];sb=np.s_[1-dy:1-dy+h,1-dx:1-dx+w]
        a=padded_color[sa];b=padded_color[sb]
        delta=b-a
        coef=np.clip(((color-a)*delta).sum(2)/np.maximum((delta*delta).sum(2),1.),0.,1.)
        error=np.max(np.abs(color-(a+coef[:,:,None]*delta)),axis=2)
        span=np.abs(delta).max(2)
        yield np.stack([np.minimum(padded_alpha[sa],padded_alpha[sb]), np.log1p(span/16), np.log1p(error/4)],axis=2)
