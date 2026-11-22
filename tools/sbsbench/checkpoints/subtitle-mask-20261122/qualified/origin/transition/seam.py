"""One native-pixel, color-guided reconstruction of the continuous protection field."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from compare import es, m, ROOT, line_original, track

# Both sides have to carry protection. A genuine gap between two white strokes
# has a center RGB outside their color segment, unlike an antialiased transition.
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

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias=nn.Parameter(torch.tensor(1.))
        self.span=nn.Parameter(torch.tensor(.5))
        self.error=nn.Parameter(torch.tensor(1.))
    def forward(self,x,base):
        gain=torch.sigmoid(self.bias+F.softplus(self.span)*x[:,:,1]-F.softplus(self.error)*x[:,:,2])
        return torch.maximum(base,(gain*x[:,:,0]).amax(1))

def apply(rgb,alpha,net):
    # The four directions share three learned coefficients and one bounded gate.
    b=float(net.bias.detach());s=float(F.softplus(net.span.detach()));e=float(F.softplus(net.error.detach()))
    out=alpha.copy()
    for f in features(rgb,alpha):
        g=1/(1+np.exp(np.clip(-b-s*f[:,:,1]+e*f[:,:,2],-50,50)))
        np.maximum(out,g*f[:,:,0],out=out)
    return out

def plan():
    import compare
    p=m.read(track.ROOT/'extended-plan.json')
    assert p['runtime']==m.runtime()
    m.dump(ROOT/'fit-plan.json',dict(runtime=m.runtime(),base=m.bind(track.ROOT/'extended-plan.json'),
          script=m.bind(Path(__file__)),observer=m.bind(Path(compare.__file__)),
          fit=m.read(es.s.PLAN)['sources'],seed=724192,updates=1000,
          design='Frozen H28 mixed backbone and native decoder, same q38. A single shared, bounded gate learns a one-native-pixel interpolation from confident opposite neighbors, RGB segment residual and color span. Fit separate Chinese, English and Japanese synthetic subtitle plates with exact original native layer labels, including native narrow gaps and negative corrections. Validate 24 point captures, 30-frame actual OCR 4K, 24 camera frames and 60 authored multiresolution frames; separately report enclosed gaps on the published crop.'))

def train():
    p=m.read(ROOT/'fit-plan.json');m.checked(p['script']);assert p['runtime']==m.runtime()
    torch.set_num_threads(2);rng=np.random.default_rng(p['seed']);torch.manual_seed(p['seed'])
    models=es.load_models();XX=[];BB=[];YY=[];counts={}
    for receipt in p['fit']:
        for row in m.read(m.checked(receipt))['rows']:
            assert row['split']=='fit'
            rgb,labels,clean=es.d.reference.real.nt.read_native(row)
            for l,t,r,b in row['work_boxes_ltrb_band']:
                im=rgb[t:b,l:r];roles=labels[t:b,l:r]
                _,base,_=line_original(im,models);feat=np.stack(list(features(im,base)),axis=2)
                offered=feat[:,:,:,0].max(2)
                target=np.clip(roles[:,:,:2].astype('f4').sum(2)/255,0,1)
                decision=(roles[:,:,:2].max(2)>=32).astype('f4')
                pool=[]
                for cond,cap in ((offered>base+.015,900),(np.ones(base.shape,bool),400),((offered>base+.015)&(decision==0),250),((offered>base+.015)&(decision==1),250)):
                    ids=np.flatnonzero(cond)
                    if len(ids):pool.append(rng.choice(ids,min(cap,len(ids)),replace=False))
                y,x=np.unravel_index(np.unique(np.concatenate(pool)),base.shape)
                XX.append(feat[y,x]);BB.append(base[y,x]);YY.append(np.stack([target[y,x],decision[y,x]],1))
                counts[row['language']]=counts.get(row['language'],0)+1
    x=torch.from_numpy(np.concatenate(XX));base=torch.from_numpy(np.concatenate(BB));target=torch.from_numpy(np.concatenate(YY))
    net=Decoder();optimizer=torch.optim.Adam(net.parameters(),lr=.025)
    for i in range(p['updates']):
        k=rng.integers(0,len(x),1536);pred=net(x[k],base[k]).clamp(1e-6,1-1e-6)
        cut=torch.logit(pred)-np.log(38/217)
        loss=F.binary_cross_entropy(pred,target[k,0])+.65*F.binary_cross_entropy_with_logits(cut,target[k,1])
        optimizer.zero_grad();loss.backward();optimizer.step()
    weights=ROOT/'decoder.pt';torch.save(net.state_dict(),weights)
    m.dump(ROOT/'trained.json',dict(plan=m.bind(ROOT/'fit-plan.json'),runtime=m.runtime(),weights=m.bind(weights),samples=len(x),lines=counts,
                                    coefficients=[float(t.detach()) for t in (net.bias,F.softplus(net.span),F.softplus(net.error))]))
    print(len(x),counts,net.state_dict())

def load():
    p=m.read(ROOT/'trained.json');assert p['runtime']==m.runtime() and m.bind(ROOT/'fit-plan.json')==p['plan']
    net=Decoder();net.load_state_dict(torch.load(m.checked(p['weights']),weights_only=True));return net.eval()

if __name__=='__main__':{'plan':plan,'train':train}[sys.argv[1]]()

