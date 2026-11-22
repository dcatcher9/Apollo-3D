"""Fit a native fill-outline transition from clean and chroma-subsampled subtitle plates."""
from pathlib import Path
import sys
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import uniform_filter,gaussian_filter
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import compare as c
import seam
import evaluate
m=c.m;es=c.es

class Transfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_span=nn.Parameter(torch.tensor(np.log(32.),dtype=torch.float32))
        self.log_noise=nn.Parameter(torch.tensor(np.log(8.),dtype=torch.float32))
    def forward(self,x,base):
        strength=x[:,:,0]*x[:,:,1]/(x[:,:,1]+self.log_span.exp())
        allowed=strength/(1+(x[:,:,2]/self.log_noise.exp()).square())
        return torch.maximum(base,allowed.amax(1))

def features(rgb,alpha):
    for raw in seam.features(rgb,alpha):
        yield raw[:,:,0],np.expm1(raw[:,:,1])*16,np.expm1(raw[:,:,2])*4

def video(rgb,rng):
    # Render the original crop with 4:2:0 chroma sampling and mild luma softening.
    rgb=rgb.astype('f4');y=rgb@np.array([.2126,.7152,.0722],'f4');chroma=rgb-y[:,:,None]
    h,w=y.shape;pad=np.pad(chroma,((0,h%2),(0,w%2),(0,0)),mode='edge')
    low=(pad[::2,::2]+pad[1::2,::2]+pad[::2,1::2]+pad[1::2,1::2])*.25
    from PIL import Image
    res=np.stack([np.asarray(Image.fromarray(low[:,:,i]).resize((pad.shape[1],pad.shape[0]),Image.Resampling.BILINEAR))[:h,:w] for i in range(3)],2)
    y=gaussian_filter(y,rng.uniform(.3,.65),mode='nearest')
    noise=rng.normal(0,1.5,size=(h,w,3)).astype('f4')
    return np.clip(np.rint(res+y[:,:,None]+noise),0,255).astype('u1')

def plan():
    p=m.read(ROOT.parent/'fit-plan.json')
    assert p['runtime']==m.runtime()
    m.dump(ROOT/'fit-plan.json',dict(p,new_code=m.bind(Path(__file__)),features=m.bind(Path(seam.__file__)),seed=12853,
        method='Restrict repair to a contrast transition between two protected neighbors. Two learned scales govern RGB contrast and tolerated native color deviation. Include original and subsampled video variants, sample and upweight real counters explicitly. Same dense paired tests as the one-pixel control.'))

def train():
    p=m.read(ROOT/'fit-plan.json');assert p['runtime']==m.runtime();m.checked(p['new_code']);m.checked(p['features'])
    torch.set_num_threads(2);torch.manual_seed(p['seed']);rng=np.random.default_rng(p['seed']);models=es.load_models()
    xx=[];bb=[];yy=[];ww=[];cohort=Counter()
    from scipy.ndimage import binary_fill_holes
    for receipt in p['fit']:
        for row in m.read(m.checked(receipt))['rows']:
            assert row['split']=='fit';rgb,labels,_=es.d.reference.real.nt.read_native(row)
            for l,t,r,b in row['work_boxes_ltrb_band']:
                native=rgb[t:b,l:r];roles=labels[t:b,l:r]
                target=np.minimum(roles[:,:,:2].astype('f4').sum(2)/255,1)
                on=roles[:,:,:2].max(2)>=32;clear=roles.max(2)==0
                counters=(binary_fill_holes(on)&~on)&clear
                for name,im in [('rgb',native),('video',video(native,rng))]:
                    _,a,_=c.line_original(im,models)
                    feats=np.stack([np.stack(u,2) for u in features(im,a)],2)
                    offer=feats[:,:,:,0].max(2)
                    sites=[]
                    for mask,cap in ((offer>a+.015,1000),(np.ones(on.shape,bool),400),((offer>a+.015)&on,340),((offer>a+.015)&~on,260),(counters,200)):
                        ix=np.flatnonzero(mask)
                        if len(ix):sites.append(rng.choice(ix,min(cap,len(ix)),replace=False))
                    y,x=np.unravel_index(np.unique(np.concatenate(sites)),a.shape)
                    xx.append(feats[y,x]);bb.append(a[y,x]);yy.append(np.stack([target[y,x],on[y,x]],1))
                    ww.append(1.+counters[y,x].astype('f4'));cohort[(row['language'],name)]+=1
    x,base,y,w=[torch.from_numpy(np.concatenate(v)) for v in (xx,bb,yy,ww)]
    net=Transfer();opt=torch.optim.Adam(net.parameters(),lr=.012)
    for step in range(850):
        ix=rng.integers(0,len(x),2048);a=net(x[ix],base[ix]).clamp(1e-6,1-1e-6)
        loss=(w[ix]*(F.binary_cross_entropy(a,y[ix,0],reduction='none')+.7*F.binary_cross_entropy_with_logits(torch.logit(a)-np.log(38/217),y[ix,1],reduction='none'))).mean()
        opt.zero_grad();loss.backward();opt.step()
    file=ROOT/'transition.pt';torch.save(net.state_dict(),file)
    m.dump(ROOT/'trained.json',dict(runtime=m.runtime(),plan=m.bind(ROOT/'fit-plan.json'),weights=m.bind(file),samples=len(x),counts={str(k):v for k,v in cohort.items()},span=float(net.log_span.exp().detach()),noise=float(net.log_noise.exp().detach())))
    print(m.read(ROOT/'trained.json'))

def load():
    p=m.read(ROOT/'trained.json');assert p['plan']==m.bind(ROOT/'fit-plan.json') and p['runtime']==m.runtime()
    net=Transfer();net.load_state_dict(torch.load(m.checked(p['weights']),weights_only=True));return net.eval()

def apply(rgb,alpha,net):
    out=alpha.copy();span=float(net.log_span.exp().detach());noise=float(net.log_noise.exp().detach())
    for support,delta,err in features(rgb,alpha):
        np.maximum(out,support*(delta/(delta+span))/(1+(err/noise)**2),out=out)
    return out

if __name__=='__main__':
    mode=sys.argv[1]
    if mode in ('fit-plan','train'):
        {'fit-plan':plan,'train':train}[mode]()
    else:
        seam.load=load;seam.apply=apply;evaluate.ROOT=ROOT
        if mode=='plan':
            # Use the exact frozen control image, the two outputs live in separate folders.
            m.dump(ROOT/'control.json',m.read(ROOT.parent/'control.json'))
        evaluate.main()

