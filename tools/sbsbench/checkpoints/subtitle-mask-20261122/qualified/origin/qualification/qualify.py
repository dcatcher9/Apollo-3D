"""Score the frozen single-pass candidate on the RTX 5080 and retain matched display pixels."""
from collections import Counter,defaultdict
import ctypes as C
from pathlib import Path
import sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
DEV=ROOT.parent
VIDEO=DEV/'rawmatte-stroke-continuity-1121/video'
sys.path.insert(0,str(VIDEO))
from export_stage import r
from rtx_points import AlphaEngine,device
import review
es=r.es;c=r.c;m=r.m;ref=es.d.reference
ARMS=('previous','final')
VIS=['navcam-480-dark-onset','parachute-720-pale-onset','landing-2160-color-shadow-onset']
def plan():
    p=m.read(VIDEO/'extended-plan.json');assert p['runtime']==m.runtime()
    plans=m.read(es.ROOT/'rtx/extended-plan.json');trans=m.read(VIDEO/'rtx-parity.json')
    references={k:m.bind(device.HERE/(k+'.json')) for k in ('multires60','camera24')}
    p.update(cnn=plans['cnn_engine'],native=plans['alpha_engine'],transition=trans['engine'],trained_transition=m.bind(VIDEO/'trained.json'),
             run=m.bind(Path(__file__)),cohorts=m.read(DEV/'rawmatte-native-guided-refinement-1121/plan.json')['cohort'],
             baseline=references,legacy_rule=m.bind(DEV/'rawmatte-native-union-rule-comparison-1121/union-rule-vs-direct.json'),
             sample_ids=VIS,design='Previous and final H28, unchanged expanded current OCR Work, q38. Actual TensorRT CNN, native decoder and transition; native feature generation and causal tracking on CPU. Capture and 4K archived actual OCR. Camera and authored use recorded fixture boxes. Controls use the matching archived host or reproduced Direct Glyph GPU evaluation. Same-frame source, labels and predictions exported on first appearances. No training.')
    m.dump(ROOT/'extended-plan.json',p)

class Transition:
    def __init__(self,g,path):
        self.g=g;self.engine=g.runtime.deserialize_cuda_engine(path.read_bytes())
        self.ctx=self.engine.create_execution_context();self.storage={}
        for name,n in [('native_rgb8',3),('input_alpha',1),('alpha',1)]:
            capacity=256*4096*n;dev=C.c_uint64();pin=C.c_void_p()
            g.malloc(C.byref(dev),4*capacity);g.alloc_host(C.byref(pin),4*capacity)
            arr=np.ctypeslib.as_array((C.c_float*capacity).from_address(pin.value))
            assert self.ctx.set_tensor_address(name,dev.value)
            self.storage[name]=(dev,pin,arr)
    def __call__(self,rgb,a):
        h,w=a.shape;assert h<=256 and w<=4096
        g=self.g;device.idle()
        for name,values in [('native_rgb8',np.ascontiguousarray(rgb.transpose(2,0,1)[None],dtype='f4')),('input_alpha',a[None,None])]:
            dst,pin,buf=self.storage[name];assert self.ctx.set_input_shape(name,values.shape)
            buf[:values.size]=values.ravel();g.upload(dst,pin,values.nbytes,g.stream)
        assert tuple(self.ctx.get_tensor_shape('alpha'))==(1,1,h,w)
        assert self.ctx.execute_async_v3(g.stream.value)
        dst,pin,buf=self.storage['alpha'];g.download(pin,dst,a.nbytes,g.stream);g.sync(g.stream)
        return buf[:h*w].reshape(h,w).copy()

class Models:
    def __init__(self,p):
        self.gpu=device.Gpu()
        self.enc=device.Engine(self.gpu,m.checked(p['cnn']),144,512)
        self.native=AlphaEngine(self.gpu,m.checked(p['native']))
        self.final=Transition(self.gpu,m.checked(p['transition']))
        _,self.head,_=es.load_models()
    def line(self,rgb):
        ctx=es.s.Context(rgb,self.enc,self.head);h,w=rgb.shape[:2]
        mask=np.empty((h,w),'f4')
        for i in range(0,h*w,65536):
            y,x=np.divmod(np.arange(i,min(h*w,i+65536)),w);mask[y,x]=self.native(ctx.features(y,x))
        return None,mask,ctx.line.tiles

def run(mode,p,models):
    trackers=[c.track.Tracker(28,True,mode!='captured') for _ in ARMS]
    state={};ref.arms=ARMS;ref.ROOT=ROOT
    geom=ref.geometry
    def geometry(data,dims,content):
        cores,boxes=geom(data,dims,content);words=np.frombuffer(data,'<u4')
        state['covers']=[ref.s.expected_rect(words[144+8*i:148+8*i],content,*dims) for i in range(int(words[4]))]
        return cores,boxes
    def predict(rgb,cores,boxes,_):
        covers=state.pop('covers',boxes);cache={};masks={};tiles={}
        def score(crop,unused):
            k=(crop.ctypes.data,crop.shape,crop.strides)
            if k not in cache:cache[k]=models.line(crop)
            return cache[k]
        def revised(crop,unused):
            _,a,n=score(crop,None);return None,models.final(crop,a),n
        for name,tracker,step in zip(ARMS,trackers,(score,revised),strict=True):
            es.run_line=step
            masks[name],tiles[name],_=tracker.predict(rgb,boxes,covers,None)
        return masks,tiles
    ref.geometry=geometry;ref.predict=predict
    try:
        if mode in ('sequence','captured'):
            {'sequence':ref.sequence,'captured':ref.captured}[mode](None,p)
            return
        refbase=m.read(m.checked(p['baseline'][mode]));assert refbase['runtime']==m.runtime()
        assert refbase['dataset']==p['cohorts'][mode]
        rows=ref.s.ev.load_fixtures(p['cohorts'])[mode];groups=defaultdict(lambda:defaultdict(Counter));hist={};details=[]
        for row,old in zip(rows,refbase['rows'],strict=True):
            assert old['id']==row['id'];rgb,alpha,_=ref.real.nt.read_native(row);truth=m.truth(alpha.transpose(2,0,1))
            boxes=row['work_boxes_ltrb_band'];state['covers']=row.get('proxy_ocr_covers_ltrb_band',boxes)
            masks,tiles=predict(rgb,row['core_boxes_ltrb_band'],boxes,None)
            plan=review.ev.standard.stock_plan(row,rgb.shape[:2],152);assert plan['boxes']==boxes
            roi=review.ev.standard.mask_for(rgb.shape[:2],boxes)
            if plan['status']=='active':
                field=review.arrays({'live':models.enc},review.ev.nt.area_rgb(rgb,plan['unit']),plan['tiles'],(144,512))['live']
                live=review.ev.nt.native_publish(field,rgb.shape[:2],plan)>=38
            else:live=np.zeros(rgb.shape[:2],bool)
            masks['live']=live
            event={'steady':'same_cue','replacement':'same_box_swap','clear':'blank_after','blank':'blank_before'}.get(row['event'],row['event'])
            meta=dict(case_id=row['case_id'],cue_id=row['cue_id'],event=event) if mode=='multires60' else row
            cnt=ref.real.b.measure(groups,hist,meta,truth,roi,masks,('all',row['language'],row['style'],meta['event']))
            assert cnt['live']==old['counts']['half-conv']['metrics'],row['id']
            details.append(dict(id=row['id'],event=row['event'],counts=cnt,tiles=tiles))
            if row['id'] in p['sample_ids']:
                yy,xx=np.nonzero(truth['mask']);area=np.s_[max(0,yy.min()-8):yy.max()+9,max(0,xx.min()-8):xx.max()+9]
                np.savez_compressed(ROOT/(row['id']+'.npz'),rgb=rgb[area],target=truth['mask'][area],clear=truth['clear'][area],**{k:v[area] for k,v in masks.items()})
        m.dump(ROOT/(mode+'.json'),dict(runtime=m.runtime(),plan=m.bind(ROOT/'extended-plan.json'),groups={k:{arm:m.summary(v) for arm,v in g.items()} for k,g in groups.items()},rows=details))
    finally:ref.geometry=geom

def real_crop(models):
    rgb,view=c.sample()
    _,a,_=models.line(rgb);final=models.final(rgb,a)
    refimpl=ref.real;source=m.read(refimpl.ROOT/'plan.json');key='b142-f2126-s0'
    item=next(row for row in m.read(m.checked(source['source']))['samples'] if row['id']==key)
    loc=next(row for row in m.read(refimpl.ROOT/'points.json')['rows'] if row['id']==key)
    raw=(Path(item['source_dump'])/item['source_band_file']).read_bytes()
    from hashlib import sha256
    assert sha256(raw).hexdigest()==item['source_band_sha256']
    w,fh=item['analysis_source_wh'];top,h=item['native_band_top_height'];band=np.floor(np.frombuffer(raw,'<f2').reshape(h,w,4)[:,:,:3].astype('f4')*255+.5).astype('u1')
    work=loc['work'];l,t,rr,b=work[0];core=item['native_core_band_rect']
    domain=ref.s.ev.standard.mask_for((h,w),work)
    wide=ref.s.ev.standard.mask_for((h,w),[refimpl.bounded(work[0],item['native_band_rect'],(core[3]-core[1]+7)//8)])
    geom=ref.s.ev.standard.stock_plan(dict(core_boxes_ltrb_band=[core],work_boxes_ltrb_band=work,band_ltrb=[0,top,w,fh]),(h,w),152)
    live=refimpl.field_codes(band,geom,domain,wide,models.enc)[t:b,l:rr]
    assert np.array_equal(band[t:b,l:rr],rgb)
    np.savez_compressed(ROOT/'marked-glyph.npz',rgb=rgb[view],live=live[view],previous=a[view],final=final[view])
    results=[]
    for i in range(17):
        start=time.perf_counter();_,v,n=models.line(rgb);stop=time.perf_counter()
        out=models.final(rgb,v);end=time.perf_counter()
        if i>=2:results.append([(stop-start)*1000,(end-stop)*1000,(end-start)*1000])
    assert np.array_equal(out,final)
    m.dump(ROOT/'timing.json',dict(runtime=m.runtime(),source=key,native_shape=list(rgb.shape[:2]),tiles=n,n=len(results),
          p50_ms=np.median(results,0).tolist(),p95_ms=np.percentile(results,95,axis=0).tolist(),
          columns=['mixed CNN plus CPU feature construction plus GPU native decoder','TensorRT transition including Python, copies and synchronization','entire one-line offline inference'],
          scope='Idle host, 2 warmups, 15 sequential samples on actual marked line. No OCR, causal tracking, warp or capture. This is the offline reference pipeline.'))

def main():
    if sys.argv[1]=='plan':return plan()
    p=m.read(ROOT/'extended-plan.json');assert p['runtime']==m.runtime();m.checked(p['run']);torch.set_num_threads(2)
    models=Models(p)
    for mode in sys.argv[1:]:
        assert p['runtime']==m.runtime()
        if mode=='real':real_crop(models)
        else:run(mode,p,models)
        print(mode,'finished',flush=True)
if __name__=='__main__':main()

