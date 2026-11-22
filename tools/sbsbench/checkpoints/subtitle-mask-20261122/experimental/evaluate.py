"""Matched comparison to the prior continuous protection model on four archived cohorts."""
import sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
import conditioned as z
q=z.q;es=q.es;ref=q.ref;m=q.m;ROOT=z.ROOT
ARMS=('prior','learned','blend')
def plan():
    p=m.read(q.ROOT/'extended-plan.json')
    p.update(proposed=m.bind(ROOT/'trained.json'),eval_code=m.bind(Path(__file__)),model_code=m.bind(Path(z.__file__)),arms=list(ARMS),
             design='H28 with preserved previous causal history, fixed q38. Compare prior continuous model, direct learned background-conditioned mask, and prior + (1-prior)*learned**2. Record exact counts against known CPU baseline on every frame. Same three independent train-free camera records and 4K OCR observations as previous qualification.')
    m.dump(ROOT/'extended-plan.json',p)
def run(mode):
    torch.set_num_threads(2);p=m.read(ROOT/'extended-plan.json');assert m.runtime()==p['runtime']
    m.checked(p['model_code']);m.checked(p['eval_code'])
    models=es.load_models();enc=models[0];new=z.load();transition=q.r.load()
    trackers=[q.c.track.Tracker(28,True,mode!='captured') for _ in ARMS];state={}
    geom=ref.geometry
    def geometry(rec,dims,content):
        core,work=geom(rec,dims,content);words=np.frombuffer(rec,'<u4')
        state['cover']=[ref.s.expected_rect(words[144+8*i:148+8*i],content,*dims) for i in range(int(words[4]))]
        return core,work
    def predict(rgb,cores,boxes,_):
        cache={};covers=state.pop('cover',boxes);masks={};ts={}
        def fields(im):
            k=(im.ctypes.data,im.shape,im.strides)
            if k not in cache:
                _,old,n=q.c.line_original(im,models);old=q.r.apply(im,old,transition)
                raw,n2=z.native(im,enc,new);assert n2==n
                cache[k]=(old,raw,old+(1-old)*raw*raw,n)
            return cache[k]
        for i,(arm,tr) in enumerate(zip(ARMS,trackers,strict=True)):
            def line(im,_):
                fs=fields(im);return None,fs[i],fs[-1]
            es.run_line=line
            masks[arm],ts[arm],_=tr.predict(rgb,boxes,covers,None)
        return masks,ts
    ref.arms=ARMS;ref.ROOT=ROOT;ref.geometry=geometry;ref.predict=predict
    if mode in ('captured','sequence'):
        {'captured':ref.captured,'sequence':ref.sequence}[mode](None,p)
        file={'captured':'extended-captures.json','sequence':'extended-sequence.json'}[mode]
    else:
        key='camera24' if mode=='camera' else 'multires60'
        fixture=ref.s.ev.load_fixtures(p['cohorts'])[key]
        groups=defaultdict(lambda:defaultdict(Counter));history={};results=[]
        for row in fixture:
            rgb,alpha,_=ref.real.nt.read_native(row)
            boxes=row['work_boxes_ltrb_band'];state['cover']=row.get('proxy_ocr_covers_ltrb_band',boxes)
            masks,tiles=predict(rgb,row['core_boxes_ltrb_band'],boxes,None);truth=m.truth(alpha.transpose(2,0,1));roi=ref.s.ev.standard.mask_for(rgb.shape[:2],boxes)
            event={'steady':'same_cue','replacement':'same_box_swap','clear':'blank_after','blank':'blank_before'}.get(row['event'],row['event'])
            meta=dict(case_id=row['case_id'],cue_id=row['cue_id'],event=event) if mode=='authored' else row
            counts=ref.real.b.measure(groups,history,meta,truth,roi,masks,('all',row['language'],row['style'],meta['event']))
            results.append(dict(id=row['id'],event=row['event'],counts=counts,tiles=tiles))
            if row['id'] in q.VIS:
                y,x=np.nonzero(truth['mask']);cut=np.s_[max(0,y.min()-8):y.max()+9,max(0,x.min()-8):x.max()+9]
                np.savez_compressed(ROOT/(row['id']+'.npz'),rgb=rgb[cut],target=truth['mask'][cut],clear=truth['clear'][cut],**{k:v[cut] for k,v in masks.items()})
        file=mode+'.json'
        m.dump(ROOT/file,dict(runtime=m.runtime(),plan=m.bind(ROOT/'extended-plan.json'),groups={k:{a:m.summary(v) for a,v in gr.items()} for k,gr in groups.items()},rows=results))
    doc=m.read(ROOT/file);control=m.read(q.VIDEO/(mode+'-paired.json'));assert doc['runtime']==control['runtime']==m.runtime()
    for a,b in zip(doc['rows'],control['rows'],strict=True):
        assert all(a.get(k)==b.get(k) for k in ('id','event','ordinal'))
        if mode=='captured':assert all(v['prior']==b['counts'][key]['candidate'] for key,v in a['counts'].items())
        else:assert Counter(a['counts']['prior'])==Counter(b['counts']['candidate'])
    m.dump(ROOT/(mode+'-verified.json'),dict(doc,reference=m.bind(q.VIDEO/(mode+'-paired.json'))))
    if mode=='captured':print(doc['totals'])
    else:print(mode,{k:[round(v['f1'],5),round(v['local_clear_rate'],5),round(v['stable_target_flip_rate'],5)] for k,v in doc['groups']['all'].items()},flush=True)
if __name__=='__main__':
    if sys.argv[1]=='plan':plan()
    else:run(sys.argv[1])

