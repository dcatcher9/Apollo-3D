"""Resolve each source color against both the OCR line palette and its nearby background."""
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.ndimage import map_coordinates
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"rawmatte-native-neighborhood-1122"))
import neighborhood_fit as previous
z=previous.z
q, es, m = z.q,z.es,z.m


class Context(z.Context):
    def __init__(self, image, encoder):
        super().__init__(image,encoder)
        self.color_posterior,self.inks=es.d.palette(image,self.roles)

    def features(self,y,x):
        local=super().features(y,x)
        color=self.rgb[y,x].astype("f4")
        votes=[map_coordinates(table,(color/16).T,order=1,mode="nearest",prefilter=False)
               for table in self.color_posterior[1:]]
        deltas=[np.abs(color-ink).mean(1)/64 for ink in self.inks]
        return np.ascontiguousarray(np.column_stack((local,*votes,*deltas)),dtype="f4")


def plan():
    src=m.read(previous.ROOT/"plan.json")
    m.dump(ROOT/"plan.json",dict(fit=src["fit"],encoder=src["encoder"],runtime=m.runtime(),
        code=m.bind(Path(__file__)),background=m.bind(Path(z.__file__)),
        palette=m.bind(Path(es.d.__file__)),seed=45218,steps=3800,
        design="A single head with line-wide color-role posterior and local background/texture evidence. Include native bold renderings, fit-only heavy variants, and similar-brightness fill/rim. Consistency loss sees exactly the same native gold on a translated fit-only background. Fixed q38."))


def train():
    p=m.read(ROOT/"plan.json")
    assert p["runtime"]==m.runtime()
    for key in ("encoder","code","background","palette"):m.checked(p[key])
    torch.set_num_threads(3);torch.manual_seed(p["seed"]);rng=np.random.default_rng(p["seed"])
    enc,_,_=es.load_models();X=[];S=[];Y=[];counts=Counter()
    for receipt in p["fit"]:
        for i,row in enumerate(m.read(m.checked(receipt))["rows"]):
            assert row["split"]=="fit"
            rgb,roles,bg=q.ref.real.nt.read_native(row)
            bold=es.s.original_fit.bold_data(row,bg)
            colors=np.asarray(row.get("role_colors_rgb8",row.get("glyph_colors_rgb8")),"f4")
            for l,t,r,b in row["work_boxes_ltrb_band"]:
                image,lab,clean=rgb[t:b,l:r],roles[t:b,l:r],bg[t:b,l:r]
                variants=list(previous.variants(image,lab,clean,colors,rng,i))
                variants.append(("libass-bold",bold[0][t:b,l:r],bold[1][t:b,l:r]))
                a=lab.astype("f4")/255
                common=rng.uniform(30,222,3);near=(common+rng.uniform(-28,28,3)).clip(0,255)
                pair_cols=np.stack([common,near,colors[2]])
                adjacent=np.rint(a@pair_cols+(1-a.sum(2,keepdims=True))*clean).clip(0,255).astype("u1")
                variants.append(("similar-ink-rim",adjacent,lab))
                for name,im,alpha in variants:
                    y,x=z.indices(alpha,rng)
                    idx=es.s.original_fit.indices(alpha,rng)
                    y,x=np.r_[y,idx[0]],np.r_[x,idx[1]]
                    ctx=Context(im,enc)
                    original=ctx.features(y,x)
                    if name in ("original","libass-bold","heavy","similar-ink-rim","blank"):
                        opacity=alpha.sum(2,keepdims=True,dtype="f4")/255
                        rotated=np.roll(clean,(int(rng.integers(3,21)),int(rng.integers(7,35))),(0,1))
                        sibling=np.rint(im+(rotated.astype("f4")-clean)*(1-opacity)).clip(0,255).astype("u1")
                        sibling_data=Context(sibling,enc).features(y,x)
                    else:
                        sibling_data=original
                    X.append(original.astype("f2"));S.append(sibling_data.astype("f2"));Y.append(alpha[y,x,:2]);counts[row["language"],name]+=1
            if i%18==0:print("fit",i,flush=True)
    x,s,y=(torch.from_numpy(np.concatenate(k)) for k in (X,S,Y));del X,S,Y
    net=z.Net(x.shape[1]);opt=torch.optim.AdamW(net.parameters(),lr=.0018)
    for it in range(p["steps"]):
        ix=rng.integers(0,len(x),2048);labels=y[ix].float()/255
        outputs=net(torch.cat([x[ix],s[ix]]).float()).reshape(2,len(ix),3)
        cover=labels.sum(1).clamp(max=1)
        hard=(labels.amax(1)>=32/255).float()
        logits=outputs[:,:,0]
        loss=F.binary_cross_entropy_with_logits(logits,cover.expand_as(logits))
        loss+=.85*F.binary_cross_entropy_with_logits(logits-np.log(38/217),hard.expand_as(logits))
        loss+=.35*F.binary_cross_entropy_with_logits(outputs[:,:,1:],labels.expand_as(outputs[:,:,1:]))
        loss+=.45*(logits[0].sigmoid()-logits[1].sigmoid()).abs().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if it==2600:opt.param_groups[0]["lr"]=.00055
    path=ROOT/"palette.pt";torch.save(net.eval().state_dict(),path)
    m.dump(ROOT/"trained.json",dict(runtime=m.runtime(),plan=m.bind(ROOT/"plan.json"),weights=m.bind(path),features=x.shape[1],sites=len(x),counts={str(k):v for k,v in counts.items()}))


def load():
    p=m.read(ROOT/"trained.json");assert p["plan"]==m.bind(ROOT/"plan.json") and p["runtime"]==m.runtime()
    head=z.Net(p["features"]);head.load_state_dict(torch.load(m.checked(p["weights"]),weights_only=True))
    return head.eval()


def native(rgb,enc,model):
    ctx=Context(rgb,enc);h,w=rgb.shape[:2];a=np.empty((h,w),"f4")
    for i in range(0,h*w,65536):
        y,x=np.divmod(np.arange(i,min(h*w,i+65536)),w)
        with torch.inference_mode():a[y,x]=model(torch.from_numpy(ctx.features(y,x)))[:,0].sigmoid().numpy()
    return a,ctx.tiles


if __name__=="__main__":{"plan":plan,"train":train}[sys.argv[1]]()
