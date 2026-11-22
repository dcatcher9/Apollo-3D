"""Verify actual published masks for a dense labeled frame and trace the offline reference latency."""
from hashlib import sha256
import time
from collections import Counter
import numpy as np
from PIL import Image
import torch
import qualify as q
m=q.m;ROOT=q.ROOT

def frame4k(p,models):
    recorded=m.read(ROOT/'extended-sequence.json')
    seq=m.read(m.checked(p['sequence_plan']));capture=m.checked(seq['application'])
    native=m.read(capture);fixture=m.read(m.checked(seq['fixture']))
    legacy=m.read(m.checked(p['legacy_rule']));rulepath=m.checked(legacy['sources']['rule'])
    rr=m.read(rulepath);idx=16
    frame=fixture['frames'][idx];assert frame['event']=='onset' and frame['case']=='en-large-factory'
    host=native['frames'][idx];rframe=rr['frames'][idx];rect=host['contour_identity']
    assert frame['frame']['sha256']==host['source_snapshot_sha256']==rframe['source_snapshot_sha256']
    with Image.open(m.checked(frame['frame'])) as im:rgb=np.array(im.crop((0,rect[10],rect[8],rect[9])).convert('RGB'))
    record=(capture.parent/host['ocr']['file']).read_bytes();assert sha256(record).hexdigest()==host['ocr']['sha256']
    cores,boxes=q.ref.geometry(record,(rect[8],rect[9],rect[10]),rect[16:20]);assert boxes==host['work_boxes_ltrb']
    case=next(c for c in fixture['cases'] if c['scenario']==frame['cue_id'])
    with np.load(m.checked(case['visible_layer_artifact'])) as f:alpha=f['visible_alpha']
    assert sha256(alpha.tobytes()).hexdigest()==case['exact_layer_sha256']
    l,t,r,b=case['visible_alpha_ltrb'];full=np.zeros((3,*rgb.shape[:2]),'u1');full[:,t-rect[10]:b-rect[10],l:r]=alpha
    truth=m.truth(full);roi=q.ref.s.ev.standard.mask_for(rgb.shape[:2],boxes)
    masks=dict(live=m.load_mask(capture.parent,host),rule=m.load_mask(rulepath.parent,rframe),previous=np.zeros_like(roi),final=np.zeros_like(roi))
    for l,t,r,b in boxes:
        cut=np.s_[t:b,l:r];_,a,_=models.line(rgb[cut])
        masks['previous'][cut]|=np.rint(a*255)>=38
        masks['final'][cut]|=np.rint(models.final(rgb[cut],a)*255)>=38
    for arm,mask in masks.items():
        count=m.overlap(mask,truth,roi,None,'')
        numbers=legacy['frames'][idx]['counts']['rule'] if arm=='rule' else recorded['rows'][idx]['counts']['current' if arm=='live' else arm]
        assert count==Counter(numbers),(arm,count,numbers)
    yy,xx=np.nonzero(truth['mask']);a=np.s_[max(0,yy.min()-8):yy.max()+9,max(0,xx.min()-8):xx.max()+9]
    np.savez_compressed(ROOT/'native4k-factory.npz',rgb=rgb[a],target=truth['mask'][a],clear=truth['clear'][a],**{k:v[a] for k,v in masks.items()})

def trace(models):
    rgb,_=q.c.sample();raw=q.device.idle;start=time.perf_counter
    results=[];poll=0.
    def observe():
        nonlocal poll
        t=start();raw();poll+=start()-t
    q.device.idle=observe
    try:
        for i in range(17):
            poll=0.;begin=start()
            _,v,n=models.line(rgb);models.final(rgb,v)
            total=start()-begin
            if i>=2:results.append([1000*total,1000*poll,1000*(total-poll)])
    finally:q.device.idle=raw
    m.dump(ROOT/'timing-trace.json',dict(runtime=m.runtime(),samples=len(results),native_shape=list(rgb.shape[:2]),
          scope='Single offline native line including CPU features, synchronous transfers and three engine stages. The middle column times every actual Windows process-list check in this harness. The last column subtracts this measured check duration per observation. Excludes capture, OCR, history and warp.',
          columns=['wall_ms','process_check_ms','wall_minus_check_ms'],median=np.median(results,0).tolist(),p95=np.percentile(results,95,axis=0).tolist()))
if __name__=='__main__':
    torch.set_num_threads(2);p=m.read(ROOT/'extended-plan.json');assert p['runtime']==m.runtime();models=q.Models(p)
    frame4k(p,models);trace(models)

