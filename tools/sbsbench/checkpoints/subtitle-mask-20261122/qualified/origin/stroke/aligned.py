"""Measure canonical detail height separately from current-OCR-bounded temporal crop alignment."""
from pathlib import Path
from collections import Counter,defaultdict
import sys
import numpy as np
import torch
import eval_strokes as es
import camera
prev=es.prev;d=es.d;ref=d.reference;m=d.m
ROOT=es.ROOT/'aligned'
ARMS=('h28','aligned28','aligned40')
def bounds(a,b):
    return min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])
class Tracker:
    def __init__(self,target,hold,history):
        self.target=target;self.hold=hold;self.history=history;self.previous=[];self.shape=None
    def predict(self,rgb,boxes,covers,models):
        if self.shape!=rgb.shape:self.previous=[]
        self.shape=rgb.shape;out=np.zeros(rgb.shape[:2],bool);saved=[];total=0;changes=[]
        for bi,box in enumerate(boxes):
            window=box;accepted=False;n=0;f=0.
            matches=[s for s in self.previous if prev.overlaps(box,s[0]) and sum(prev.overlaps(s[0],b) for b in boxes)==1]
            if len(matches)==1:
                oldbox,oldwin,oldrgb,oldalpha=matches[0]
                x,y,r,b=prev.intersection(oldwin,box)
                v=np.s_[y-oldwin[1]:b-oldwin[1],x-oldwin[0]:r-oldwin[0]]
                n,f=prev.confirmation(oldrgb[v],rgb[y:b,x:r],oldalpha[v])
                accepted=n>=24 and f>=.70
                if accepted and self.hold:
                    # Current detector cover authenticates every pixel, including any carried margin.
                    window=list(prev.intersection(bounds(oldbox,box),covers[bi]))
            l,t,r,b=window;crop=rgb[t:b,l:r]
            d.TARGET=self.target
            _,alpha,tiles=es.run_line(crop,models);total+=tiles;fused=alpha.copy()
            if accepted:
                x,y,r,b=prev.intersection(oldwin,window)
                v=np.s_[y-oldwin[1]:b-oldwin[1],x-oldwin[0]:r-oldwin[0]]
                z=np.s_[y-t:b-t,x-l:r-l]
                delta=np.abs(crop[z].astype('f4')-oldrgb[v]).max(2)
                fused[z]+=.95*np.exp(-(delta/24)**2)*(oldalpha[v]-alpha[z])
            out[window[1]:window[3],window[0]:window[2]]|=np.rint(fused*255)>=38
            saved.append((box,window,crop.copy(),alpha));changes.append((accepted,window!=box))
        self.previous=saved if self.history else []
        return out,total,changes

def main():
    mode=sys.argv[1]
    if mode=='plan':
        ROOT.mkdir(exist_ok=True)
        p=m.read(es.ROOT/'extended-plan.json')
        m.dump(ROOT/'extended-plan.json',dict(p,program=m.bind(Path(__file__)),source=m.bind(es.ROOT/'extended-plan.json'),method='Frozen weights. Same q38 and extended boxes. Compare exact height 28, verified previous/current box union capped by current detector cover at H28, and identical policy at fixed H40. Use no previous frame on individual point samples. When the corpus lacks an actual per-observation detector cover, keep the source Work as the cap.'))
        return
    torch.set_num_threads(2);p=m.read(ROOT/'extended-plan.json');m.checked(p['program']);assert m.runtime()==p['runtime']
    models=es.load_models();track=[Tracker(t,hold,mode!='captured') for t,hold in ((28,False),(28,True),(40,True))]
    ctx={};log=[]
    geom=ref.geometry
    def decode(record,dims,content):
        cores,boxes=geom(record,dims,content)
        words=np.frombuffer(record,'<u4')
        ctx['covers']=[ref.s.expected_rect(words[144+8*i:148+8*i],content,*dims) for i in range(int(words[4]))]
        return cores,boxes
    def predict(rgb,cores,boxes,nets):
        covers=ctx.pop('covers',boxes)
        assert len(covers)==len(boxes)
        for w,c in zip(boxes,covers):assert list(prev.intersection(w,c))==list(w)
        data=[tr.predict(rgb,boxes,covers,nets) for tr in track]
        log.append({k:info[2] for k,info in zip(ARMS,data)})
        return {k:item[0] for k,item in zip(ARMS,data)},Counter({k:item[1] for k,item in zip(ARMS,data)})
    ref.ROOT=ROOT;ref.predict=predict;ref.arms=ARMS;ref.geometry=decode
    if mode in ('sequence','captured'):
        {'sequence':ref.sequence,'captured':ref.captured}[mode](models,p)
        filename='extended-sequence.json' if mode=='sequence' else 'extended-captures.json'
    else:
        common=m.read(d.DEV/'rawmatte-native-guided-refinement-1121/plan.json')
        source=ref.s.ev.load_fixtures(common['cohort'])['camera24' if mode=='camera' else 'multires60']
        hist={};groups=defaultdict(lambda:defaultdict(Counter));rows=[]
        for row in source:
            rgb,alpha,_=ref.real.nt.read_native(row);boxes=row['work_boxes_ltrb_band']
            ctx['covers']=row.get('proxy_ocr_covers_ltrb_band',boxes)
            masks,tiles=predict(rgb,row['core_boxes_ltrb_band'],boxes,models)
            roi=ref.s.ev.standard.mask_for(rgb.shape[:2],boxes)
            event={'steady':'same_cue','replacement':'same_box_swap','clear':'blank_after','blank':'blank_before'}.get(row['event'],row['event'])
            meta=dict(case_id=row['case_id'],cue_id=row['cue_id'],event=event) if mode=='authored' else row
            counts=ref.real.b.measure(groups,hist,meta,m.truth(alpha.transpose(2,0,1)),roi,masks,('all',row['language'],row['style'],meta['event']))
            rows.append(dict(id=row['id'],event=row['event'],counts=counts,tiles=tiles))
        filename=mode+'.json';groups={k:{arm:m.summary(v) for arm,v in data.items()} for k,data in groups.items()}
        m.dump(ROOT/filename,dict(runtime=m.runtime(),plan=m.bind(ROOT/'extended-plan.json'),groups=groups,rows=rows))
        print({k:[round(v['f1'],5),round(v['stable_target_flip_rate'],5),v['clear_flips']] for k,v in groups['all'].items()})
    result=m.read(ROOT/filename);base=m.read(es.ROOT/filename)
    for row,start,l in zip(result['rows'],base['rows'],log,strict=True):
        if mode=='captured':
            assert all(row['counts'][k]['h28']==v['stroke'] for k,v in start['counts'].items())
        else:assert row['counts']['h28']==start['counts']['stroke_temporal']
        row['history']=l
    m.dump(ROOT/(mode+'-verified.json'),result)
if __name__=='__main__':main()
