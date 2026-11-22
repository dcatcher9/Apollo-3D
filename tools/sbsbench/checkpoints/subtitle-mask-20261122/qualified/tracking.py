import numpy as np
import temporal as prev

def bounds(a,b):
    return min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])

class Tracker:
    def __init__(self,target,hold,history):
        self.target=target;self.hold=hold;self.history=history;self.previous=[];self.shape=None
    def predict(self,rgb,boxes,covers,models):
        if self.shape!=rgb.shape:self.previous=[]
        self.shape=rgb.shape;out=np.zeros(rgb.shape[:2],bool);saved=[];total=0;changes=[]
        for bi,box in enumerate(boxes):
            window=box;accepted=False;n=0;f=0.
            matches=[s for s in self.previous if prev.overlaps(box,s[0]) and sum(prev.overlaps(s[0],b) for b in boxes)==1]
            if len(matches)==1:
                oldbox,oldwin,oldrgb,oldalpha=matches[0]
                x,y,r,b=prev.intersection(oldwin,box)
                v=np.s_[y-oldwin[1]:b-oldwin[1],x-oldwin[0]:r-oldwin[0]]
                n,f=prev.confirmation(oldrgb[v],rgb[y:b,x:r],oldalpha[v])
                accepted=n>=24 and f>=.70
                if accepted and self.hold:
                    # Current detector cover authenticates every pixel, including any carried margin.
                    window=list(prev.intersection(bounds(oldbox,box),covers[bi]))
            l,t,r,b=window;crop=rgb[t:b,l:r]
            assert self.target==28
            _,alpha,tiles=models.score(crop);total+=tiles;fused=alpha.copy()
            if accepted:
                x,y,r,b=prev.intersection(oldwin,window)
                v=np.s_[y-oldwin[1]:b-oldwin[1],x-oldwin[0]:r-oldwin[0]]
                z=np.s_[y-t:b-t,x-l:r-l]
                delta=np.abs(crop[z].astype('f4')-oldrgb[v]).max(2)
                fused[z]+=.95*np.exp(-(delta/24)**2)*(oldalpha[v]-alpha[z])
            out[window[1]:window[3],window[0]:window[2]]|=np.rint(fused*255)>=38
            saved.append((box,window,crop.copy(),alpha));changes.append((accepted,window!=box))
        self.previous=saved if self.history else []
        return out,total,changes
