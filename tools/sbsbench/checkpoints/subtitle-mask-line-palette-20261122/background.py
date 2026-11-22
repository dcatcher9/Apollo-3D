import numpy as np
from torch import nn
from scipy.ndimage import maximum_filter, uniform_filter
import roles as d

def smooth(a,n):return uniform_filter(a,(n,n,1),mode='nearest')

def lift(field,y,x,h):
    # Preserve the normalized area sampler's fractional last column.
    H,W=field.shape[:2]
    yy=(y+.5)*(H/h)-.5;xx=(x+.5)*(H/h)-.5
    y0=np.floor(yy).astype('i4');x0=np.floor(xx).astype('i4')
    fy=(yy-y0).astype('f4')[:,None];fx=(xx-x0).astype('f4')[:,None]
    y1=(y0+1).clip(0,H-1);x1=(x0+1).clip(0,W-1);y0=y0.clip(0,H-1);x0=x0.clip(0,W-1)
    a=field[y0,x0];b=field[y1,x0]
    return a+(field[y0,x1]-a)*fx+fy*((b+(field[y1,x1]-b)*fx)-(a+(field[y0,x1]-a)*fx))

class Context:
    def __init__(self,rgb,enc):
        self.rgb=rgb;h,w=rgb.shape[:2];_,self.roles,self.tiles=d.role_fields(rgb,enc)
        fine=d.norm.normalize_region(rgb,84).astype('f4')/255
        H,W=fine.shape[:2];yy,xx=np.indices((H,W))
        u=lift(self.roles,yy.ravel(),xx.ravel(),H).reshape(H,W,4)
        self.coarse=np.concatenate([self.roles,maximum_filter(self.roles[:,:,1:3],(5,5,1),mode='nearest'),smooth(self.roles[:,:,1:4],5)],2)
        # Estimate underlying RGB at two distances using locations the encoder calls background.
        # Bound the weights away from zero so flat letter interiors still have a defined estimate.
        wbg=.015+u[:,:,:1]**3
        fields=[]
        for n in (13,35):
            mass=smooth(wbg,n)
            mu=smooth(fine*wbg,n)/mass
            variance=np.maximum(0,smooth(wbg*fine*fine,n)/mass-mu*mu)
            fields.extend([mu,np.sqrt(variance.mean(2,keepdims=True))])
        self.bg=np.concatenate(fields,2);self.fine=fine
        # Current-line color prototypes preserve color shifts shared by recurring strokes.
        weights=u[:,:,1:3]**2
        total=weights.sum((0,1))+1e-5
        self.proto=np.einsum('hwc,hwk->kc',fine,weights)/total[:,None]
    def features(self,y,x):
        rgb=self.rgb;h,w=rgb.shape[:2];color=rgb[y,x].astype('f4')/255
        co=lift(self.coarse,y,x,h);bg=lift(self.bg,y,x,h);soft=lift(self.fine,y,x,h)
        ds=[]
        for start in (0,4):
            delta=color-bg[:,start:start+3];noise=bg[:,start+3:start+4]
            ds.extend([delta*5,delta/(.025+noise),noise*5])
        for prototype in self.proto:ds.append((color-prototype)*3)
        step=max(1,round(h/84));horizontal=(rgb[y,np.minimum(w-1,x+step)].astype('f4')-rgb[y,np.maximum(0,x-step)].astype('f4'))/255
        vertical=(rgb[np.minimum(h-1,y+step),x].astype('f4')-rgb[np.maximum(0,y-step),x].astype('f4'))/255
        return np.ascontiguousarray(np.column_stack([co,color,(color-soft)*6,*ds,abs(horizontal).max(1),abs(vertical).max(1)]).clip(-12,12),dtype='f4')

class Net(nn.Module):
    def __init__(self,n):
        super().__init__();self.layers=nn.Sequential(nn.Linear(n,48),nn.ReLU(),nn.Linear(48,32),nn.ReLU(),nn.Linear(32,3))
    def forward(self,x):return self.layers(x)
