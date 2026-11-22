import sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
import stroke_support as s
import temporal as prev

ROOT=s.ROOT;d=s.d;m=s.m;study=s.study
ARMS=('native_head','stroke','stroke_temporal')

def run_line(rgb,models):
    enc,head,residual=models;ctx=s.Context(rgb,enc,head)
    h,w=rgb.shape[:2];base=np.empty((h,w),'f4');score=base.copy()
    for i in range(0,h*w,65536):
        y,x=np.divmod(np.arange(i,min(h*w,i+65536)),w);f=torch.from_numpy(ctx.features(y,x))
        b=f[:,-4:-2];base[y,x]=(b[:,0]+(1-b[:,0])*b[:,1]).numpy()
        with torch.inference_mode():
            p=residual(f);score[y,x]=(p[:,0]+(1-p[:,0])*p[:,1]).numpy()
    return base,score,ctx.line.tiles

class Predictor:
    def __init__(self,use_history):
        self.last=[];self.audit=[];self.shape=None;self.use_history=use_history
    def __call__(self,rgb,cores,boxes,models):
        if self.shape!=rgb.shape:self.last=[]
        self.shape=rgb.shape
        out={k:np.zeros(rgb.shape[:2],bool) for k in ARMS};tiles=Counter({k:0 for k in ARMS})
        nexts=[];log=[]
        for box in boxes:
            l,t,r,b=box;im=rgb[t:b,l:r];base,alpha,n=run_line(im,models)
            fused=alpha.copy();accepted=False
            matching=[s for s in self.last if prev.overlaps(box,s[0]) and sum(prev.overlaps(s[0],k) for k in boxes)==1]
            if len(matching)==1:
                p,old,was=matching[0];a,c,e,f=prev.intersection(p,box)
                here=np.s_[c-t:f-t,a-l:e-l];there=np.s_[c-p[1]:f-p[1],a-p[0]:e-p[0]]
                count,agreement=prev.confirmation(old[there],im[here],was[there])
                if count>=24 and agreement>=.70:
                    accepted=True
                    delta=np.max(np.abs(im[here].astype('f4')-old[there]),axis=2)
                    fused[here]+=.95*np.exp(-(delta/24.)**2)*(was[there]-alpha[here])
            nexts.append((box,im.copy(),alpha));log.append(accepted)
            for k,a in zip(ARMS,(base,alpha,fused),strict=True):
                out[k][t:b,l:r]|=np.rint(a*255)>=38;tiles[k]+=n
        self.last=nexts if self.use_history else [];self.audit.append(log);return out,tiles

def plan():
    p=m.read(s.BASE/'extended-plan.json');trained=m.read(ROOT/'trained.json');assert m.bind(s.PLAN)==trained['plan']
    p.update(weights=m.bind(ROOT/'trained.json'),runtime=m.runtime(),scoring=m.bind(Path(__file__)),temporal_method=m.bind(Path(prev.__file__)),arms=list(ARMS))
    m.dump(ROOT/'extended-plan.json',p)

def load_models():
    p=m.read(ROOT/'extended-plan.json')
    assert p['runtime']==m.runtime()
    for k in ('weights','scoring','temporal_method'):m.checked(p[k])
    enc,head=study.models(p)
    residual=s.Residual();residual.load_state_dict(torch.load(m.checked(m.read(m.checked(p['weights']))['model']),weights_only=True));residual.eval()
    return enc,head,residual

def visual(models,p):
    real=d.reference.real
    src=m.read(real.ROOT/'plan.json');key='b142-f2126-s0'
    sample=next(r for r in real.read(m.checked(src['source']))['samples'] if r['id']==key)
    row=next(r for r in m.read(real.ROOT/'points.json')['rows'] if r['id']==key)
    from hashlib import sha256
    raw=(Path(sample['source_dump'])/sample['source_band_file']).read_bytes();assert sha256(raw).hexdigest()==sample['source_band_sha256']
    w,_=sample['analysis_source_wh'];top,h=sample['native_band_top_height']
    rgb=np.floor(np.frombuffer(raw,'<f2').reshape(h,w,4)[:,:,:3].astype('f4')*255+.5).astype('u1')
    l,t,r,b=row['work'][0];cover=sample['native_band_rect'];native=rgb[t:b,l:r]
    old,new,_=run_line(native,models)
    wide=d.reference.s.ev.standard.mask_for((h,w),[real.bounded(row['work'][0],cover,(sample['native_core_band_rect'][3]-sample['native_core_band_rect'][1]+7)//8)])
    domain=d.reference.s.ev.standard.mask_for((h,w),row['work'])
    meta=dict(core_boxes_ltrb_band=[sample['native_core_band_rect']],work_boxes_ltrb_band=row['work'],band_ltrb=[0,top,w,sample['analysis_source_wh'][1]])
    geom=d.reference.s.ev.standard.stock_plan(meta,(h,w),152)
    baseline=real.field_codes(rgb,geom,domain,wide,models[0])[t:b,l:r]
    cx,cy=cover[0]+741-l,cover[1]+138-t
    vals=[int(baseline[cy,cx]),int(np.rint(old[cy,cx]*255)),int(np.rint(new[cy,cx]*255))]
    xyxy=[cover[0]+670-l,max(0,cover[1]+36-t),min(r-l,cover[0]+780-l),min(b-t,cover[1]+157-t)]
    x0,y0,x1,y1=xyxy;cut=np.s_[y0:y1,x0:x1]
    image=Image.fromarray(native[cut])
    ims=[image,Image.fromarray(((baseline[cut]>=38)*255).astype('u1')).convert('RGB'),Image.fromarray(((np.rint(new[cut]*255)>=38)*255).astype('u1')).convert('RGB')]
    bw,bh=2*image.width,2*image.height;canvas=Image.new('RGB',(3*(bw+15)+20,bh+56),'#f8f8f5');dr=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',17)
    for j,(im,title) in enumerate(zip(ims,('Original glyph','Current mixed build','Learned native mask'),strict=True)):
        left=15+j*(bw+15);dr.text((left,7),title,font=font,fill='#14232a')
        canvas.paste(im.resize((bw,bh),Image.Resampling.NEAREST),(left,38));x=left+2*(cx-x0)+1;y=38+2*(cy-y0)+1
        dr.ellipse((x-9,y-9,x+9,y+9),outline='#ff9900',width=2)
    file=ROOT/'marked-stroke.png';canvas.save(file)
    m.dump(ROOT/'visual.json',dict(runtime=m.runtime(),plan=m.bind(ROOT/'extended-plan.json'),source=key,native_xy=[cx+l,cy+t],q8=vals,cutoff=38,image=m.bind(file)))
    print(vals)

def main():
    mode=sys.argv[1]
    if mode=='plan':return plan()
    torch.set_num_threads(2);models=load_models();p=m.read(ROOT/'extended-plan.json')
    if mode=='visual':return visual(models,p)
    pred=Predictor(mode!='captured');ref=d.reference;ref.ROOT=ROOT;ref.arms=ARMS;ref.predict=pred
    study.ROOT=ROOT;study.ARMS=ARMS;study.predict=pred
    {'sequence':ref.sequence,'authored':study.authored,'captured':ref.captured}[mode](models,p)
    file={'sequence':'extended-sequence.json','authored':'authored.json','captured':'extended-captures.json'}[mode]
    result=m.read(ROOT/file);old=m.read(s.BASE/file)
    for row,control,matches in zip(result['rows'],old['rows'],pred.audit,strict=True):
        if mode=='captured':
            assert all(row['counts'][role]['native_head']==val['native_head'] for role,val in control['counts'].items())
        else:assert row['counts']['native_head']==control['counts']['native_head']
        row['matches']=matches
    m.dump(ROOT/(mode+'-paired.json'),result)
if __name__=='__main__':main()
