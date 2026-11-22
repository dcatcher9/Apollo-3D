"""Distill an unchanged glyph on a texture-suppressed clean plate into the existing native decoder."""
from pathlib import Path
from collections import Counter
import hashlib,sys,json
from unittest.mock import patch
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch.nn import functional as F
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/"rawmatte-line-palette-1122"))
import palette_fit as p
import cues
q,m,es=p.q,p.m,p.es
cut=float(np.log(37.5/(255-37.5)))
def plan():
    spec=m.read(p.ROOT/"plan.json")
    m.dump(root/"plan.json",dict(runtime=m.runtime(),source=m.bind(Path(__file__)),fit=spec["fit"],base=m.bind(p.ROOT/"trained.json"),
        cue_impl=m.bind(Path(cues.__file__)),seed=318122,iterations=1800,candidates=[0,.25,.5,.75,1],threshold=38,
        change="Train eight new first-layer columns for fill and outline color projection and residual at both background scales. Every original weight is fixed. "
        "Counterfactual pairs share glyph, OCR geometry and encoder output. Only the clean background texture differs. "
        "Supervise confirmed texture-induced errors and preserve already correct glyph sites and background. "
        "Choose interpolation on uniformly sampled held-out fit rows before opening fresh, 4K or marked-capture metrics."))
def tensor(a):return torch.from_numpy(np.asarray(a))
def score(net,a):
    with torch.inference_mode():return net(tensor(a))[:,0].numpy().copy()
def context_with_held_roles(image,ctx):
    def held(*args):return None,ctx.roles,ctx.tiles
    with patch.object(p.es.d,"role_fields",side_effect=held):
        return cues.Context(image,None)
def main():
    spec=m.read(root/"plan.json");assert m.runtime()==spec["runtime"] and m.checked(spec["source"])==Path(__file__)
    m.checked(spec["base"])
    rng=np.random.default_rng(spec["seed"]);torch.manual_seed(spec["seed"]);torch.set_num_threads(2)
    encoder=es.load_models()[0];base=p.load();net=cues.model();m.checked(spec["cue_impl"])
    train=[];valid=[];counts=Counter();holdout=[]
    for manifest in spec["fit"]:
        rows=m.read(m.checked(manifest))["rows"]
        for index,row in enumerate(rows):
            assert row["split"]=="fit"
            # Block whole rows including all their sibling or recolored images.
            isval=index%8==2
            if isval:holdout.append(row["id"])
            rgb,roles,clean=q.ref.real.nt.read_native(row)
            for l,t,r,b in row["work_boxes_ltrb_band"]:
                lab=roles[t:b,l:r];plate=clean[t:b,l:r];original=rgb[t:b,l:r]
                h,w=plate.shape[:2];frac=lab.astype("f4")/255
                trans=1-frac.sum(2,keepdims=True).clip(0,1)
                # Existing renderer and a difficult ink pair chosen from this fit plate.
                flat=plate.reshape(-1,3)
                swatch=flat[rng.integers(0,len(flat),3)].astype("f4")
                swatch[0]=(swatch[0]+rng.uniform(-50,50,3)).clip(0,255)
                swatch[1]=(.65*swatch[0]+.35*swatch[1]+rng.uniform(-15,15,3)).clip(0,255)
                decoy=np.rint(frac@swatch+trans*plate).clip(0,255).astype("u1")
                common=np.median(plate.reshape(-1,3)[::7],axis=0)
                low_outline=np.clip(common+rng.normal(0,10,3),0,255)
                colors=np.stack([np.asarray(row.get("role_colors_rgb8",row.get("glyph_colors_rgb8")))[0],low_outline,swatch[2]])
                faint=np.rint(frac@colors+trans*plate).clip(0,255).astype("u1")
                for variant,img in (("original",original),("scene-ink",decoy),("faint-rim",faint)):
                    ctx=cues.Context(img,encoder)
                    smooth=gaussian_filter(plate.astype("f4"),(max(2,h/8),max(2,h/8),0),mode="nearest")
                    twin=np.rint(img.astype("f4")+trans*(smooth-plate)).clip(0,255).astype("u1")
                    smooth_ctx=context_with_held_roles(twin,ctx)
                    ix=rng.choice(w*h,min(15000,w*h),replace=False)
                    y,x=np.divmod(ix,w);X=ctx.features(y,x);orig=score(base,np.ascontiguousarray(X[:,:42]))
                    soft=score(base,np.ascontiguousarray(smooth_ctx.features(y,x)[:,:42]));f,o,sh=(lab[y,x,k] for k in range(3))
                    positive=(f>=32)|(o>=32);clear=(f==0)&(o==0)&(sh==0)
                    yes=orig>=cut;good=(positive&yes)|(clear&~yes);teacher_ok=positive==(soft>=cut)
                    flip=(positive|clear)&~good&teacher_ok
                    counts[variant,"oracle-corrects"]+=int(flip.sum())
                    if isval:
                        keep=positive|clear
                        valid.append((X[keep].astype("f2"),positive[keep],row["language"]))
                        continue
                    picked=[]
                    for mask,n in ((flip&clear,900),(flip&positive,700),(positive&yes,1800),(clear&~yes,2200),
                                   (clear&yes&~flip,400),(positive&~yes&~flip,400),(~(positive|clear),350)):
                        pool=np.flatnonzero(mask);picked.extend(rng.choice(pool,min(n,len(pool)),replace=False))
                    j=np.asarray(picked,dtype=int)
                    # Reproduce the existing logits at correct sites. At identified errors the oracle
                    # gives a physically achievable direction, with a firm margin across the cutoff.
                    target=orig.copy()
                    # Exact visible layers supervise difficult misses even where smoothing hides faint ink.
                    target[positive&~yes]=np.maximum(orig[positive&~yes],cut+1.25)
                    target[clear&yes]=np.minimum(orig[clear&yes],cut-.85)
                    target[flip&clear]=np.minimum(soft[flip&clear],cut-1.1)
                    target[flip&positive]=np.maximum(soft[flip&positive],cut+1.1)
                    wt=np.where(flip,1.,np.where(good,1.0,.60)).astype("f4")
                    train.append((X[j].astype("f2"),target[j],wt[j],orig[j],positive[j],clear[j]))
            if index%12==0:print("collected",index,len(rows),flush=True)
    X,Y,W,old,pos,clear=[tensor(np.concatenate([v[k] for v in train])) for k in range(6)]
    del train
    for par in net.parameters():par.requires_grad_(False)
    layer=net.layers[0].weight;layer.requires_grad_(True)
    ref=layer.detach().clone();opt=torch.optim.Adam([layer],lr=.003)
    for it in range(spec["iterations"]):
        idx=tensor(rng.integers(0,len(X),1536));z=net(X[idx].float())[:,0]
        target=Y[idx];origin=old[idx]
        # Bound logit drift on true strokes that already cross the original operating point.
        dist=F.smooth_l1_loss(z,target,reduction="none")
        preserve=F.relu(torch.minimum(origin,torch.full_like(z,cut+1.4))-z)
        guard=(pos[idx]&(origin>=cut)).float()
        loss=(W[idx]*dist).mean()+.9*(guard*preserve).mean()+.08*(layer-ref).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();layer.grad[:,:42]=0;opt.step()
        if it==1050:opt.param_groups[0]["lr"]=.001
    delta=layer.detach().clone()-ref
    table=[]
    for strength in spec["candidates"]:
        with torch.no_grad():layer.copy_(ref+strength*delta)
        groups={}
        for feats,truth,lang in valid:
            mask=score(net,feats.astype("f4"))>=cut
            for k in ("all",lang):
                g=groups.setdefault(k,Counter());g.update(tp=int((mask&truth).sum()),fp=int((mask&~truth).sum()),fn=int((~mask&truth).sum()))
        for g in groups.values():g["f1"]=2*g["tp"]/(2*g["tp"]+g["fp"]+g["fn"])
        table.append(dict(strength=strength,groups=groups))
    baseline=table[0]["groups"]["all"]
    eligible=[v for v in table if v["groups"]["all"]["fn"]<=baseline["fn"]+max(2,.03*baseline["fn"])]
    selected=max(eligible,key=lambda a:a["groups"]["all"]["f1"])
    with torch.no_grad():layer.copy_(ref+selected["strength"]*delta)
    path=root/"palette.pt";torch.save(net.eval().state_dict(),path)
    assert torch.equal(layer.detach()[:,:42],ref[:,:42])
    m.dump(root/"trained.json",dict(runtime=m.runtime(),plan=m.bind(root/"plan.json"),model=m.bind(path),rows_fit_holdout=holdout,
        training_sites=len(X),counts={str(k):v for k,v in counts.items()},validation=table,selected=selected["strength"]))
    print("selected",selected["strength"],selected["groups"]["all"],flush=True)
def load():
    rec=m.read(root/"trained.json");assert rec["runtime"]==m.runtime() and rec["plan"]==m.bind(root/"plan.json")
    n=p.z.Net(50);n.load_state_dict(torch.load(m.checked(rec["model"]),weights_only=True));return n.eval()
if __name__=="__main__":{"plan":plan,"train":main}[sys.argv[1]]()
