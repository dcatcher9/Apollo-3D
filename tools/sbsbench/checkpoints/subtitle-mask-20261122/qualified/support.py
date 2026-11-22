import numpy as np
import torch
from torch import nn
import native as d

def guided_context(line):
    high=d.norm.normalize_region(line.rgb,84)
    h,w=high.shape[:2]; yy,xx=np.indices((h,w))
    roles=d.lift(line.p,(h,w),yy.ravel(),xx.ravel()).reshape(h,w,4)[:,:,1:3]
    field=roles.copy()
    high=high.astype('f4')
    for step in (1,2,4,8):
        result=field.copy()
        for dx,dy in ((1,0),(0,1),(1,1),(1,-1)):
            for sign in (-1,1):
                x=np.clip(xx+dx*sign*step,0,w-1);y=np.clip(yy+dy*sign*step,0,h-1)
                mx=np.clip(xx+dx*sign*(step//2),0,w-1);my=np.clip(yy+dy*sign*(step//2),0,h-1)
                dif=np.maximum(np.abs(high-high[y,x]),np.abs(high-high[my,mx])).max(2)
                affinity=np.exp(-(dif/20.)**2)
                np.maximum(result,field[y,x]*affinity[:,:,None],out=result)
        field=result
    return field

class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(21,16),nn.ReLU(),nn.Linear(16,2))
        nn.init.constant_(self.net[-1].bias,-3.)
    def forward(self,x):
        # A learned addition needs positive role evidence arriving along similar-color pixels.
        extra=self.net(x).sigmoid()*x[:,-2:]
        base=x[:,-4:-2]
        return base+(1-base)*extra

class Context:
    def __init__(self,rgb,net,head):
        self.line=d.Line(rgb,net);self.head=head;self.context=guided_context(self.line)
    def features(self,y,x):
        raw=self.line.features(y,x)
        with torch.inference_mode():base=self.head(torch.from_numpy(raw)).sigmoid().numpy()
        rgb=self.line.rgb;h,w=rgb.shape[:2];ch,cw=self.context.shape[:2]
        x0,x1,wx=d.norm._native_axis(w,h,ch,cw);y0,y1,wy=d.norm._native_axis(h,h,ch,ch)
        u=self.context[y0[y],x0[x]];v=self.context[y0[y],x1[x]]
        a=u+wx[x,None]*(v-u);u=self.context[y1[y],x0[x]];v=self.context[y1[y],x1[x]]
        b=u+wx[x,None]*(v-u);e=a+wy[y,None]*(b-a)
        return np.column_stack([raw,base,e])
