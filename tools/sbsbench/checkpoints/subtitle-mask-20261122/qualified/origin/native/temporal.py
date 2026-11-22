"""Current RGB confirms native stroke identity before the previous-frame alpha can contribute."""
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import test_model as study
import decoder as d

ROOT=d.ROOT/'temporal'
ARMS=('native_head','temporal')
def size(b):
    return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def intersection(a,b):
    return max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
def overlaps(a,b):
    common=size(intersection(a,b));return common>.5*(size(a)+size(b)-common)

def edges(rgb,step):
    luma=(rgb[:,:,0].astype('i4')+2*rgb[:,:,1].astype('i4')+rgb[:,:,2])//4
    return np.stack([luma[2:-2:step,4::step]-luma[2:-2:step,:-4:step],luma[4::step,2:-2:step]-luma[:-4:step,2:-2:step]],axis=-1)

def confirmation(old,current,previous_alpha):
    h,w=old.shape[:2]
    if min(h,w)<5:return 0,0.
    stride=max(2,w//250)
    a=edges(old,stride);b=edges(current,stride)
    power=(a*a).sum(2)
    ink=previous_alpha[2:-2:stride,2:-2:stride]>=38/255
    samples=(power>=24**2)&ink
    dot=(a*b).sum(2)
    agrees=(dot>0)&(dot.astype('f8')**2>=.7*power*(b*b).sum(2))
    delta=np.abs(current[2:-2:stride,2:-2:stride].astype('i2')-old[2:-2:stride,2:-2:stride]).max(2)
    n=int(samples.sum())
    return n,float((samples&agrees&(delta<=32)).sum()/max(1,n))

class Temporal:
    def __init__(self):
        self.last=[];self.audit=[];self.dims=None
    def __call__(self,rgb,cores,boxes,models):
        enc,head=models
        if self.dims!=rgb.shape:self.last=[]
        self.dims=rgb.shape;out={k:np.zeros(rgb.shape[:2],bool) for k in ARMS};tiles=Counter({k:0 for k in ARMS})
        next_state=[];audit=[]
        for box in boxes:
            l,t,r,b=box;crop=rgb[t:b,l:r];h,w=crop.shape[:2];line=d.Line(crop,enc)
            alpha=np.empty((h,w),'f4')
            for i in range(0,h*w,65536):
                y,x=np.divmod(np.arange(i,min(h*w,i+65536)),w)
                with torch.inference_mode():alpha[y,x]=d.union(head(torch.from_numpy(line.features(y,x)))).numpy()
            fused=alpha.copy();n=0;f=0.;accepted=False
            matches=[s for s in self.last if overlaps(box,s[0]) and sum(overlaps(s[0],other) for other in boxes)==1]
            if len(matches)==1:
                old_box,old_rgb,old_alpha=matches[0];x0,y0,x1,y1=intersection(box,old_box)
                q=np.s_[y0-t:y1-t,x0-l:x1-l];z=np.s_[y0-old_box[1]:y1-old_box[1],x0-old_box[0]:x1-old_box[0]]
                n,f=confirmation(old_rgb[z],crop[q],old_alpha[z])
                if n>=24 and f>=.70:
                    accepted=True
                    delta=np.max(np.abs(crop[q].astype('f4')-old_rgb[z]),axis=2)
                    trust=.95*np.exp(-(delta/24.)**2)
                    fused[q]+=trust*(old_alpha[z]-alpha[q])
            for name,field in zip(ARMS,(alpha,fused),strict=True):
                out[name][t:b,l:r]|=np.rint(255*field)>=38;tiles[name]+=line.tiles
            # Exactly one previous raw prediction, never a recursively retained prediction.
            next_state.append((box,crop.copy(),alpha));audit.append(dict(samples=n,agreement=f,accepted=accepted))
        self.last=next_state;self.audit.append(audit)
        return out,tiles

def main():
    mode=sys.argv[1]
    if mode=='plan':
        ROOT.mkdir(exist_ok=True)
        p=d.m.read(d.ROOT/'extended-plan.json');p.update(single=d.m.bind(d.ROOT/'extended-plan.json'),temporal=d.m.bind(Path(__file__)),method='One-frame native alpha blend only for unique intersecting boxes with 70% matched sampled stroke edges and photometric consistency; no cue metadata used by predictor. Flush on empty or changed source dimensions. Recheck exact one-frame controls and all same-box swaps/disappearances.')
        d.m.dump(ROOT/'extended-plan.json',p);return
    p=d.m.read(ROOT/'extended-plan.json');assert p['runtime']==d.m.runtime();d.m.checked(p['temporal'])
    torch.set_num_threads(2);models=study.models(p);pred=Temporal()
    ref=d.reference;ref.ROOT=ROOT;ref.arms=ARMS;ref.predict=pred
    study.ROOT=ROOT;study.ARMS=ARMS;study.predict=pred
    {'sequence':ref.sequence,'authored':study.authored}[mode](models,p)
    fn='extended-sequence.json' if mode=='sequence' else 'authored.json'
    data=d.m.read(ROOT/fn);control=d.m.read(d.ROOT/fn)
    for row,old,check in zip(data['rows'],control['rows'],pred.audit,strict=True):
        assert row['counts']['native_head']==old['counts']['native_head']
        row['check']=check
    d.m.dump(ROOT/(mode+'-verified.json'),data)
    print('matches',[(r.get('event',r.get('id')),sum(c['accepted'] for c in r['check'])) for r in data['rows'] if r['check']])
if __name__=='__main__':main()
