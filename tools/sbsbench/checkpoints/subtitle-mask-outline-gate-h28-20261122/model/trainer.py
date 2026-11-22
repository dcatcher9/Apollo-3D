"""Learn a correction specifically for missing visible borders, anchored to the released decoder."""
from pathlib import Path
import copy,sys
import numpy as np, torch
from torch.nn import functional as F
import vote_gate as v
r,c,m=v.r,v.c,v.m
def run():
    home=r/"head-outline-gate";home.mkdir();seed=437725;torch.manual_seed(seed);torch.set_num_threads(3);rng=np.random.default_rng(seed)
    release=m.read(r/"head-distilled/trained.json");earlier=m.read(r/"head-vote-gate/trained.json");base=c.field.refit.load();base.load_state_dict(torch.load(m.checked(release["model"]),weights_only=True));base.requires_grad_(False);net=v.Correction(base)
    net.gate.load_state_dict(torch.load(m.checked(earlier["palette_gate"]),weights_only=True))
    sources=[r/"head-stock/data.json",r/"head-data-only/data.json",r/"shared-ink/data.json"]
    m.dump(home/"plan.json",dict(runtime=m.runtime(),source=m.bind(Path(__file__)),trunk=m.bind(r/"head-distilled/trained.json"),init=m.bind(r/"head-vote-gate/trained.json"),
        inputs=[m.bind(p) for p in sources],seed=seed,method="Train a local outline vote on the released opacity decoder. Penalize changing published opacity on plain and shadow-only styles, true clear gaps and already-correct fill. Emphasize near-decision real libass outlines where a changed color vote could recover coverage. Calibrate strength on untouched fit styles, ignoring test metrics."))
    train=[];vals=[]
    for path in sources:
        rec=m.read(path);assert rec["runtime"]==m.runtime()
        for part,ref in rec.get("sets",rec.get("files")).items():
            with np.load(m.checked(ref)) as z:x=torch.from_numpy(z["x"]);y=torch.from_numpy(z["y" if "y" in z else "lab"]);groups=z["group"]
            pools=[np.flatnonzero(groups==j) for j in range(len(rec["groups"]))]
            no_rim=np.array([("shadow" in n and "both" not in n) or "plain" in n or "blank" in n for n in rec["groups"]])[groups]
            if part!="fit":vals.append((x,y,pools,no_rim,rec["groups"]));continue
            with torch.inference_mode():
                old=[];potential=[]
                for b in x.split(65536):
                    feat=b.float();old.append(base(feat)[:,0].half());forced=feat.clone();forced[:,37:40]=0;forced[:,38]=1;potential.append(base(forced)[:,0].half())
                old=torch.cat(old);potential=torch.cat(potential)
            rim=(y[:,1]>=32)&(y[:,0]<32)
            shift=(old<c.CUT)&(potential>=c.CUT)
            hard=np.flatnonzero((rim&shift).numpy());neg=np.flatnonzero(((y.amax(1)==0)&shift).numpy());train.append((x,y,pools,old,torch.from_numpy(no_rim),hard,neg))
    opt=torch.optim.AdamW(net.gate.parameters(),lr=.003,weight_decay=.0006)
    for i in range(2500):
        x,y,pools,old,absent,hard,neg=train[i%3]
        ix=np.concatenate([rng.choice(p,max(90,1800//len(pools))) for p in pools]+[rng.choice(hard,260),rng.choice(neg,250)])
        z=net(x[ix].float())[:,0];lab=y[ix];fg=lab[:,:2].amax(1)>=32;rim=(lab[:,1]>=32)&(lab[:,0]<32);cl=lab.amax(1)==0;prior=old[ix].float()
        w=(fg|cl).float()+.15*(~fg&~cl)+2.4*rim+.55*cl
        loss=(w*F.binary_cross_entropy_with_logits(z-c.CUT,fg.float(),reduction="none")).mean()
        preserve=absent[ix]|(cl&(prior<c.CUT))|((lab[:,0]>=32)&(prior>c.CUT))
        loss+=.55*(preserve*F.smooth_l1_loss(z,prior.clamp(-11,11),reduction="none")).mean()
        a=lab[:,:2].sum(1).clamp(max=255).float()/255
        loss+=.30*F.binary_cross_entropy_with_logits(z,a)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.gate.parameters(),2.5);opt.step()
        if i==1600:opt.param_groups[0]["lr"]=.001
    table={};scores={}
    with torch.inference_mode():
        for k in (0,.25,.5,.75,1):
            net.strength=k;scoresets=[]
            for x,y,pools,absent,names in vals:
                sel=torch.cat([net(f.float())[:,0]>=c.CUT for f in x.split(65536)]).numpy();lab=y.numpy();fg=lab[:,:2].max(1)>=32;rim=(lab[:,1]>=32)&(lab[:,0]<32);cl=lab.max(1)==0;rs={}
                for name,ids in zip(names,pools,strict=True):
                    hit=(sel[ids]&fg[ids]).sum();fp=(sel[ids]&cl[ids]).sum()
                    rs[name]=dict(f1=float(2*hit/max(1,fg[ids].sum()+hit+fp)),clear=float(fp/max(1,cl[ids].sum())),rim=float((sel[ids]&rim[ids]).sum()/max(1,rim[ids].sum())),rc=int(rim[ids].sum()))
                scoresets.append(rs)
            table[k]=scoresets;scores[k]=float(np.mean([np.mean([a["f1"] for n,a in rs.items() if n!="blank"])+.30*np.mean([a["rim"] for a in rs.values() if a["rc"]>100])-.45*np.mean([a["clear"] for a in rs.values()]) for rs in scoresets]))
    best=max(scores,key=scores.get);dst=home/"gate.pt";torch.save(net.gate.state_dict(),dst)
    m.dump(home/"trained.json",dict(runtime=m.runtime(),plan=m.bind(home/"plan.json"),model=release["model"],encoder=release["encoder"],palette_gate=m.bind(dst),strength=best,scores=scores,validation=table))
    print(best,scores,flush=True)
if __name__=="__main__":run()
