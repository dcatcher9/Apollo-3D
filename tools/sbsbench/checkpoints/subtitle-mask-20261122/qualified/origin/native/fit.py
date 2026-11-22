"""Fit the native head with disjoint multilingual libass labels, bold renders and clean plates."""
from pathlib import Path
import sys,re,json
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt
import decoder as d

ROOT=d.ROOT; m=d.m; DEV=d.DEV
SEED=81271

def plan():
    old=m.read(DEV/'rawmatte-extended-crop-fit-1121/plan.json')
    ep=m.read(DEV/'rawmatte-extended-crop-fit-1121/extended-plan.json')
    m.dump(ROOT/'plan.json',dict(runtime=m.runtime(),train=old['train'],engine=ep['control'],
          sampler=m.bind(Path(d.norm.__file__)), net=m.bind(Path(d.__file__)), script=m.bind(Path(__file__)),
          reference=m.bind(DEV/'rawmatte-extended-crop-fit-1121/extended-plan.json'),
          seed=SEED,updates=1350, native_threshold=38,
          design='Frozen mixed RGB model at exact expanded line height 28. Learn native foreground fill and outline probabilities from continuous coarse role/color statistics and exact source colors. Native labels from English, Chinese, and Japanese fit recordings. Original, libass bold, same ink on translated backgrounds, and empty controls; balanced extra deep fill, real glyph holes, boundary and hard background samples. No capture/validation pixels enter fitting. Compare raw28, learned native and live on unchanged cohorts at q38.'))

def bold_data(row,clean):
    sys.path.insert(0,str(DEV/'rawmatte-fit-ocr-clutter-1121'))
    import author_scene_pairs
    adapter,exact8,_,_=author_scene_pairs.helpers()
    script=m.checked(row['ass']).read_text(encoding='utf-8-sig')
    lines=[]
    for line in script.splitlines():
        if line.startswith('Style: '):
            fields=line.split(',');fields[7]='-1';line=','.join(fields)
        lines.append(line)
    w,h=row['source_size_w_h'];top=row['band_ltrb'][1]
    eng=adapter.Renderer(w,h,[m.checked(row['font'])])
    try:
        premul,roles,fragments=eng.render('\n'.join(lines)+'\n')
        assert fragments
        roles=roles[top:];pre=premul[top:]
        rgb=np.rint(np.clip(pre+clean*(1-roles.sum(-1,keepdims=True)),0,255)).astype('u1')
        return rgb,np.moveaxis(exact8(roles),0,-1).copy()
    finally:eng.close()

def indices(label,rng):
    f,o,s=np.moveaxis(label,2,0)
    ink=(f>=32)|(o>=32);clear=label.max(2)==0
    deep=distance_transform_edt(f>=200)>=3
    near=clear&(distance_transform_edt(~ink)<=max(2,label.shape[0]/28*2)) if ink.any() else np.zeros_like(ink)
    flat=[]
    for mask,count in ((np.ones_like(ink),1400),(deep,220),(near,380),((o>=32)&(f<32),220)):
        group=np.flatnonzero(mask)
        if len(group):flat.extend(rng.choice(group,min(count,len(group)),replace=False))
    return np.unravel_index(np.array(flat),ink.shape)

def bank(p):
    rng=np.random.default_rng(SEED)
    model=d.reference.s.ev.standard.model(m.checked(p['engine']))
    xs=[]; ys=[]; xs2=[]; descriptions=[]; counts=Counter()
    for source in p['train']:
        rows=m.read(m.checked(source))['rows']
        for i,row in enumerate(rows):
            assert row['split']=='fit'
            rgb,role,clean=d.reference.real.nt.read_native(row)
            for style in ('original','bold'):
                img,alpha=(rgb,role) if style=='original' else bold_data(row,clean)
                for l,t,r,b in row['work_boxes_ltrb_band']:
                    im=img[t:b,l:r];lab=alpha[t:b,l:r];cl=clean[t:b,l:r]
                    opacity=lab.sum(2,keepdims=True,dtype='f4')/255
                    shifted=np.roll(cl,(int(rng.integers(4,20)),int(rng.integers(5,32))),(0,1))
                    other=np.rint(np.clip(im+(shifted.astype('f4')-cl)*(1-opacity),0,255)).astype('u1')
                    y,x=indices(lab,rng)
                    base=d.Line(im,model);sibling=d.Line(other,model)
                    xs.append(base.features(y,x));xs2.append(sibling.features(y,x))
                    ys.append(lab[y,x,:2]);counts[(row['language'],style)]+=1
                    if style=='original' and i%3==0:
                        no=np.zeros_like(lab)
                        y,x=indices(no,rng);neg=d.Line(cl,model).features(y,x)
                        xs.append(neg);xs2.append(neg);ys.append(no[y,x,:2])
            descriptions.append(row['id'])
            if i%16==0:print('native training sources',len(descriptions),flush=True)
    X,X2,Y=[np.concatenate(z) for z in (xs,xs2,ys)]
    assert np.isfinite(X).all() and X.shape==X2.shape
    return torch.from_numpy(X),torch.from_numpy(X2),torch.from_numpy(Y.astype('f4')/255),dict((str(k),v) for k,v in counts.items())

def train():
    p=m.read(ROOT/'plan.json'); assert p['runtime']==m.runtime()
    for k in ('net','sampler','script'):m.checked(p[k])
    torch.manual_seed(SEED);torch.set_num_threads(3);rng=np.random.default_rng(SEED)
    x,paired,y,counts=bank(p)
    net=d.NativeHead(); opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.0005)
    losses=[]
    for step in range(p['updates']):
        idx=rng.integers(0,len(x),size=1536);target=y[idx];mark=target.max(1).values>=32/255
        a,b=net(x[idx]),net(paired[idx]);a_u,b_u=d.union(a),d.union(b)
        loss=0.
        for logits,prob in ((a,a_u),(b,b_u)):
            # Alpha supervision and the actual native protection decision share one head.
            loss+=.5*F.binary_cross_entropy_with_logits(logits,target)
            decision=torch.logit(prob.clamp(1e-6,1-1e-6))-np.log(38/217)
            loss+=.35*F.binary_cross_entropy_with_logits(decision,mark.float())
        loss+=.35*(a_u-b_u).abs().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        losses.append(float(loss.detach()))
    file=ROOT/'native-head.pt';torch.save(net.eval().state_dict(),file)
    m.dump(ROOT/'trained.json',dict(plan=m.bind(ROOT/'plan.json'),runtime=m.runtime(),parameters=sum(v.numel() for v in net.parameters()), paired_native_sites=len(x),weights=m.bind(file),fit_coverage=counts, first_loss=np.mean(losses[:100]),last_loss=np.mean(losses[-100:])))
    print('trained',len(x),'pairs',counts,flush=True)
if __name__=='__main__':{'plan':plan,'train':train}[sys.argv[1]]()
