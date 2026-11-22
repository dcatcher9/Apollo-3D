"""Learn how much evidence can travel within a stroke from current RGB and the learned native matte."""
from pathlib import Path
import sys,importlib.util
from collections import Counter
import numpy as np
from scipy.ndimage import maximum_filter
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'rawmatte-native-role-decoder-1121'
sys.path.insert(0,str(BASE))
import decoder as d
import test_model as study
m=d.m
spec=importlib.util.spec_from_file_location('native_role_training',BASE/'fit.py')
original_fit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(original_fit)
PLAN=ROOT/'fit-plan-v2.json'

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

def bind_plan():
    p=m.read(BASE/'plan.json');orig=m.read(BASE/'extended-plan.json')
    ROOT.mkdir(exist_ok=True)
    m.dump(PLAN,dict(runtime=m.runtime(),sources=p['train'],initial=m.bind(BASE/'extended-plan.json'),head=m.read(BASE/'trained.json')['weights'],
                control=orig['engine'],model_code=m.bind(Path(d.__file__)),code=m.bind(Path(__file__)),
                policy='Fit only. Frozen mixed CNN at extended height 28; frozen native decoder. Learn bounded extra fill and outline confidence from 84px aligned color-connected role evidence (four scales and eight directions). Genuine and thickened native rasters share position and ROI; soft cumulative mass dilation keeps real counters and clean background labeled. Fixed q38, single inference per tile. Evaluate holdouts before any live change.'))

def train():
    p=m.read(PLAN);assert p['runtime']==m.runtime();m.checked(p['code']);m.checked(p['model_code'])
    torch.set_num_threads(2);torch.manual_seed(17941)
    net,head=study.models(m.read(m.checked(p['initial'])));rng=np.random.default_rng(17941)
    X=[];Y=[];audit=Counter()
    for source in p['sources']:
        for idx,row in enumerate(m.read(m.checked(source))['rows']):
            assert row['split']=='fit'
            rgb,roles,clean=d.reference.real.nt.read_native(row)
            colors=np.asarray(row.get('role_colors_rgb8',row.get('glyph_colors_rgb8')),'f4')
            assert colors.shape==(3,3)
            for l,t,r,b in row['work_boxes_ltrb_band']:
                ref=roles[t:b,l:r];cl=clean[t:b,l:r];h=b-t
                for mode in ('original','heavy','very-heavy','blank'):
                    if mode=='blank' and idx%3:continue
                    if mode=='original':im=rgb[t:b,l:r];lab=ref
                    elif mode=='blank':im=cl;lab=np.zeros_like(ref)
                    else:
                        radius=max(1,round(h*(.020 if mode=='heavy' else .042)))
                        cum=ref.cumsum(2,dtype='i4')
                        yy,xx=np.ogrid[-radius:radius+1,-radius:radius+1]
                        footprint=((yy*yy+xx*xx)<=radius*radius)[:,:,None]
                        wide=maximum_filter(cum,footprint=footprint,mode='nearest')
                        lab=np.diff(wide,axis=2,prepend=np.zeros_like(wide[:,:,:1])).astype('u1')
                        weights=lab.astype('f4')/255
                        im=np.rint(weights@colors+(1-weights.sum(2,keepdims=True))*cl).clip(0,255).astype('u1')
                    y,x=original_fit.indices(lab,rng)
                    features=Context(im,net,head).features(y,x)
                    X.append(features);Y.append(lab[y,x,:2]);audit[(row['language'],mode)]+=1
            if idx%20==0:print(source['path'].split('\\')[-2],idx,flush=True)
    X=torch.from_numpy(np.concatenate(X));Y=torch.from_numpy(np.concatenate(Y).astype('f4')/255)
    model=Residual();opt=torch.optim.AdamW(model.parameters(),lr=.002)
    for step in range(1250):
        idx=rng.integers(0,len(X),size=2048);x=X[idx];y=Y[idx]
        p2=model(x).clamp(1e-6,1-1e-6);prob=p2[:,0]+(1-p2[:,0])*p2[:,1]
        decision=torch.logit(prob.clamp(1e-6,1-1e-6))-np.log(38/217)
        loss=F.binary_cross_entropy(p2,y)+.6*F.binary_cross_entropy_with_logits(decision,(y.max(1).values>=32/255).float())
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    file=ROOT/'residual.pt';torch.save(model.eval().state_dict(),file)
    m.dump(ROOT/'trained.json',dict(runtime=m.runtime(),plan=m.bind(PLAN),model=m.bind(file),parameters=sum(p.numel() for p in model.parameters()),fit_pixels=len(X),composition={str(k):v for k,v in audit.items()}))
    print('Trained residual on',len(X),'pixels',flush=True)
if __name__=='__main__':{'plan':bind_plan,'train':train}[sys.argv[1]]()
