"""Transfer useful predictions from two training-time encoders into the original one-pass native head."""
from pathlib import Path
import copy,sys
import numpy as np
import torch
from torch.nn import functional as F
import refit
r,c,m=refit.ROOT,refit.c,refit.m
SCHOOLS=[("head-stock","head-trained","validation"),("head-data-only","head-balanced","val")]
def run():
    home=r/"head-distilled";home.mkdir();torch.set_num_threads(3);torch.manual_seed(121122);rng=np.random.default_rng(121122)
    records={n:m.bind(r/n/"trained.json") for pair in SCHOOLS for n in pair[:2]}
    m.dump(home/"plan.json",dict(runtime=m.runtime(),trainer=m.bind(Path(__file__)),parents=records,steps=2900,seed=121122,
      design="One native opacity head uses the original deployed encoder. Train with two role-encoder teachers on the same pixels. Transfer a teacher's confidence only when consistent with the exact libass target. Preserve older correct outputs on native fill, clear and gaps. Select by balanced score across both held-out-phrase cohorts. Teachers run only during training."))
    def head(arm):
        net=c.field.refit.load();doc=m.read(m.checked(records[arm]));assert doc["runtime"]==m.runtime()
        net.load_state_dict(torch.load(m.checked(doc["model"]),weights_only=True));return net
    sets=[];vals=[]
    for student,teacher,valid in SCHOOLS:
        ds,dt=[m.read(m.checked(m.read(m.checked(records[k]))["data"])) for k in (student,teacher)]
        assert ds["sites"]==dt["sites"] and ds["groups"]==dt["groups"]
        for partition in ("fit",valid):
            with np.load(m.checked(ds["sets"][partition])) as s,np.load(m.checked(dt["sets"][partition])) as e:
                label="y" if "y" in s.files else "lab";y=torch.from_numpy(s[label]);group=s["group"];x=torch.from_numpy(s["x"])
                assert np.array_equal(s[label],e[label]) and np.array_equal(s["group"],e["group"])
                groups=ds["groups"];pools=[np.flatnonzero(group==i) for i in range(len(groups))]
                if partition=="fit":
                    net=head(teacher);h=head(student)
                    with torch.inference_mode():
                        prior=torch.cat([net(z.float())[:,0] for z in torch.from_numpy(e["x"]).split(65536)]).half()
                        anchor=torch.cat([h(z.float())[:,0] for z in x.split(65536)]).half()
                    sets.append((x,y,pools,groups,prior,anchor))
                else:vals.append((x,y,pools,groups))
    net=head("head-stock");start=copy.deepcopy(net.state_dict());opt=torch.optim.AdamW(net.parameters(),lr=.00048,weight_decay=.0003)
    for it in range(2900):
        x,y,pools,groups,teach,anchor=sets[it%2];ix=np.concatenate([rng.choice(pool,max(80,2400//len(pools))) for pool in pools])
        lab=y[ix];fg=lab[:,:2].amax(1)>=32;clear=lab.amax(1)==0;z=net(x[ix].float())[:,0]
        opacity=lab[:,:2].sum(1).clamp(max=255).float()/255;prior=teach[ix].float();old=anchor[ix].float()
        confident=((prior>c.CUT+.6)&fg)|((prior<c.CUT-.6)&clear);stable=((old>c.CUT+.8)&fg)|((old<c.CUT-.8)&clear)
        hard=F.binary_cross_entropy_with_logits(z-c.CUT,fg.float(),reduction="none")
        soft=F.binary_cross_entropy_with_logits((z-c.CUT)/1.8,((prior-c.CUT)/1.8).sigmoid(),reduction="none")
        loss=(hard*((fg|clear).float()+.14*(~fg&~clear))).mean()+.56*F.binary_cross_entropy_with_logits(z,opacity)
        loss+=.75*(soft*confident).mean()+.12*(F.smooth_l1_loss(z,old.clamp(-9,8),reduction="none")*stable).mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),4.);opt.step()
        if it==1850:opt.param_groups[0]["lr"]=.00019
    end=net.state_dict();out=c.field.refit.load();tab={};scores={}
    def mix(k):return {n:v+k*(end[n]-v) for n,v in start.items()}
    with torch.inference_mode():
        for k in (0,.25,.50,.75,1):
            out.load_state_dict(mix(k));data={};av=[]
            for cohort,(x,y,pools,groups) in zip(SCHOOLS,vals,strict=True):
                mask=torch.cat([out(v.float())[:,0]>=c.CUT for v in x.split(65536)]).numpy();lab=y.numpy();gold=lab[:,:2].max(1)>=32;clear=lab.max(1)==0;stats={}
                for n,idx in zip(groups,pools,strict=True):
                    sel=mask[idx];g=gold[idx];cl=clear[idx];hit=(g&sel).sum();false=(cl&sel).sum()
                    stats[n]=dict(f1=float(2*hit/max(1,g.sum()+hit+false)),clear=float(false/max(1,cl.sum())),recall=float(hit/max(1,g.sum())))
                fs=[v["f1"] for n,v in stats.items() if n!="blank"];av.append(np.mean(fs)+.3*np.quantile(fs,.15)-.22*stats["blank"]["clear"]);data[cohort[0]]=stats
            scores[k]=float(min(av)+.5*np.mean(av));tab[k]=data
        best=max(scores,key=scores.get);out.load_state_dict(mix(best));p=home/"palette.pt";torch.save(out.state_dict(),p)
    m.dump(home/"trained.json",dict(runtime=m.runtime(),plan=m.bind(home/"plan.json"),encoder=m.read(m.checked(records["head-stock"]))["encoder"],model=m.bind(p),blend=best,scores=scores,validation=tab))
    print("chosen",best,scores)
if __name__=="__main__":run()
