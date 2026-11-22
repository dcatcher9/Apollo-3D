"""Learn native subtitle opacity from normalized glyph evidence and a local background estimate."""
from pathlib import Path
import sys
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import uniform_filter,maximum_filter,gaussian_filter,distance_transform_edt
ROOT=Path(__file__).resolve().parent;DEV=ROOT.parent
sys.path.insert(0,str(DEV/'rawmatte-final-qualification-1122'))
import qualify as q
m=q.m;es=q.es;d=es.d
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
def plan():
    p=m.read(q.ROOT/'extended-plan.json');fit=m.read(es.s.PLAN)
    m.dump(ROOT/'plan.json',dict(runtime=m.runtime(),baseline=m.bind(q.ROOT/'summary.json'),fit=fit['sources'],encoder=p['engine'],script=m.bind(Path(__file__)),
        seed=44877,steps=3000,design='Frozen mixed H28 CNN, one native shared feature decoder predicting combined protection and two auxiliary visible layers. Train on fit-only native RGB and exact layers, with plain/outline/shadow/both; synthesize low-contrast outline, reverse-polarity fill, translucent decoration, flat, textured, drifting and clean backgrounds. No validation image or label enters training. Evaluate standalone and evidence-gated addition to frozen baseline at q38. All four prior cohorts and genuine glyph counters are retained.'))
def variants(img,lab,clean,palette,rng,idx):
    yield 'original',img,lab
    h,w=clean.shape[:2];Y,X=np.indices((h,w),dtype='f4');Y=Y/max(1,h-1);X=X/max(1,w-1)
    origin=rng.uniform(15,240,3).astype('f4')
    for k in range(2):
        # Alternative backgrounds are fit-only. Partial smooth areas make a faint border identifiable.
        if k==0:
            base=(origin+X[:,:,None]*rng.uniform(-25,25,3)+Y[:,:,None]*rng.uniform(-20,20,3)).clip(0,255).astype('f4')
            base=(base+rng.uniform(.02,.22)*(clean.astype('f4')-clean.mean((0,1)))).clip(0,255)
        else:
            base=np.roll(clean,(int(rng.integers(3,max(5,h))),int(rng.integers(5,max(8,w)))),(0,1)).astype('f4')
        middle=np.median(base.reshape(-1,3)[::7],axis=0)
        # Vary chromaticity independently of luminance and include near-neutral boundaries.
        delta=rng.normal(0,1,3);delta/=max(1e-4,abs(delta).max())
        outline=(middle+delta*rng.uniform(9,110)).clip(0,255)
        fill=(255-middle+rng.uniform(-50,50,3)).clip(0,255) if k==0 else palette[0]
        colors=np.stack([fill,outline,palette[2]])
        role=np.rint(lab*rng.uniform([.48,.34,.26],[1,1,.9])).astype('u1')
        a=role.astype('f4')/255
        render=np.rint((a@colors)+(1-a.sum(2,keepdims=True))*base).clip(0,255).astype('u1')
        if idx%3==0:render=q.r.video(render,rng)
        yield ('faint-smooth','faint-texture')[k],render,role
    if idx%4==0:yield 'blank',clean,np.zeros_like(lab)
def indices(lab,rng):
    f,o,s=np.moveaxis(lab,2,0);pos=(f>=32)|(o>=32);clear=lab.max(2)==0
    near=clear&(distance_transform_edt(~pos)<=max(2,lab.shape[0]/22)) if pos.any() else clear
    ix=[]
    for group,cap in [(np.ones_like(pos),1500),(f>=32,420),((o>=32)&(f<32),650),(s>=32,240),(near,620)]:
        pool=np.flatnonzero(group)
        if len(pool):ix.append(rng.choice(pool,min(len(pool),cap),replace=False))
    return np.unravel_index(np.unique(np.concatenate(ix)),pos.shape)
def train():
    p=m.read(ROOT/'plan.json');assert p['runtime']==m.runtime();m.checked(p['script'])
    torch.set_num_threads(2);torch.manual_seed(p['seed']);rng=np.random.default_rng(p['seed']);enc,_,_=es.load_models()
    xs=[];ys=[];counts=Counter()
    for src in p['fit']:
        for i,row in enumerate(m.read(m.checked(src))['rows']):
            assert row['split']=='fit'
            rgb,labels,clean=q.ref.real.nt.read_native(row)
            palette=np.asarray(row.get('role_colors_rgb8',row.get('glyph_colors_rgb8')),'f4')
            for l,t,r,b in row['work_boxes_ltrb_band']:
                for name,im,lab in variants(rgb[t:b,l:r],labels[t:b,l:r],clean[t:b,l:r],palette,rng,i):
                    y,x=indices(lab,rng);cx=Context(im,enc);xs.append(cx.features(y,x));ys.append(lab[y,x]);counts[row['language'],name]+=1
            if i%16==0:print('source',i,flush=True)
    x=torch.from_numpy(np.concatenate(xs));y=torch.from_numpy(np.concatenate(ys).astype('f4')/255)
    net=Net(x.shape[1]);opt=torch.optim.AdamW(net.parameters(),lr=.0022)
    for it in range(p['steps']):
        idx=rng.integers(0,len(x),3072);z=net(x[idx]);lab=y[idx];tar=lab[:,:2].sum(1).clamp(max=1)
        positive=(lab[:,:2].amax(1)>=32/255).float()
        loss=F.binary_cross_entropy_with_logits(z[:,0],tar)+.80*F.binary_cross_entropy_with_logits(z[:,0]-np.log(38/217),positive)+.35*F.binary_cross_entropy_with_logits(z[:,1:],lab[:,:2])
        opt.zero_grad();loss.backward();opt.step()
        if it==1900:opt.param_groups[0]['lr']=.0008
    path=ROOT/'background-native.pt';torch.save(net.eval().state_dict(),path)
    m.dump(ROOT/'trained.json',dict(runtime=m.runtime(),plan=m.bind(ROOT/'plan.json'),weights=m.bind(path),features=x.shape[1],sites=len(x),counts={str(k):v for k,v in counts.items()}))
def load():
    p=m.read(ROOT/'trained.json');assert p['runtime']==m.runtime() and p['plan']==m.bind(ROOT/'plan.json')
    n=Net(p['features']);n.load_state_dict(torch.load(m.checked(p['weights']),weights_only=True));return n.eval()
def native(rgb,enc,n):
    cx=Context(rgb,enc);h,w=rgb.shape[:2];out=np.empty((h,w),'f4')
    for i in range(0,h*w,65536):
        y,x=np.divmod(np.arange(i,min(h*w,i+65536)),w)
        with torch.inference_mode():out[y,x]=n(torch.from_numpy(cx.features(y,x)))[:,0].sigmoid().numpy()
    return out,cx.tiles
if __name__=='__main__':{'plan':plan,'train':train}[sys.argv[1]]()

