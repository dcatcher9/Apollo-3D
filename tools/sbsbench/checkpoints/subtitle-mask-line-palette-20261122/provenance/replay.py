"""Feed every evaluated OCR crop through the actual repository shader and the mixed engine."""
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import os
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"rawmatte-detector-margin-fine-1122"))
import study
q,p,m,es=study.q,study.p,study.m,study.es
ref=q.ref
FRESH=ROOT.parent/"rawmatte-fresh-temporal-1122"
ARMS=("native",)

def plan():
    doc=m.read(study.ROOT/"extended-plan.json")
    doc.update(runtime=m.runtime(),score=m.bind(Path(__file__)),native_binary=m.bind(ROOT/"roi.exe"),
       native_source=m.bind(ROOT/"roi.cpp"),shader_receipt=m.bind(ROOT.parent/"rawmatte-palette-host-review-1122/verified.json"),
       arms=list(ARMS),fresh=m.bind(FRESH/"manifest.json"),
       quality={k:m.bind(study.ROOT/(k+"-verified.json")) for k in ("sequence","captured")},
       fresh_quality=m.bind(FRESH/"palette-sixteenth.json"))
    m.dump(ROOT/"extended-plan.json",doc)

def replay(kind,phase):
    torch.set_num_threads(2);cfg=m.read(ROOT/"extended-plan.json");assert cfg["runtime"]==m.runtime()
    for k in ("score","native_binary","native_source","shader_receipt"):m.checked(cfg[k])
    folder=ROOT/(kind+"-"+phase);folder.mkdir()
    m.dump(folder/"extended-plan.json",cfg)
    ref.ROOT=folder;ref.arms=ARMS
    previous=ref.geometry;state={};records=[]
    tracker=q.c.track.Tracker(28,True,kind!="captured")
    if phase=="control":
        enc,*_=es.load_models();net=p.load()
    else:
        control=m.read(ROOT/(kind+"-control")/"inputs.json")
        assert control["runtime"]==m.runtime() and control["plan"]==m.bind(ROOT/"extended-plan.json")
    def geometry(data,dims,content):
        cores,work=previous(data,dims,content);words=np.frombuffer(data,"<u4")
        state["cover"]=[ref.s.expected_rect(words[144+8*i:148+8*i],content,*dims) for i in range(int(words[4]))]
        return cores,work
    ref.geometry=geometry
    def predict(rgb,cores,boxes,_):
        caps=state.pop("cover");work=study.windows(boxes,caps,"small")
        source_path=None
        def line(crop,_):
            nonlocal source_path
            i=len(records)
            key=f"{kind}-{i:03d}"
            off=crop.ctypes.data-rgb.ctypes.data;top,xbytes=divmod(off,rgb.strides[0]);left=xbytes//rgb.strides[1]
            h,w=crop.shape[:2];full_h,full_w=rgb.shape[:2]
            assert np.array_equal(rgb[top:top+h,left:left+w],crop)
            box=[left,top,left+w,top+h]
            source_hash=sha256(rgb.tobytes()).hexdigest()
            target=ROOT/key
            w28=(w*28+h-1)//h;w84=(w*84+h-1)//h
            record=dict(name=key,width=full_w,height=full_h,box=box,w28=w28,w84=w84,source_rgb8=source_hash)
            if phase=="control":
                target.mkdir();name=target/"source.bin"
                if source_path is None:
                    rgba=np.ones((*rgb.shape[:2],4),"f2");rgba[:,:,:3]=rgb.astype("f4")/255
                    rgba.tofile(name);source_path=name
                else:os.link(source_path,name)
                alpha,n=p.native(crop,enc,net)
                np.save(target/"alpha-reference.npy",alpha,allow_pickle=False)
                record.update(source=m.bind(name),reference=m.bind(target/"alpha-reference.npy"))
            else:
                stamp=control["cases"][i]
                assert all(record[k]==stamp[k] for k in record)
                m.checked(stamp["source"])
                expected=np.load(m.checked(stamp["reference"]),allow_pickle=False)
                alpha=np.fromfile(target/"alpha.f32","<f4").reshape(h,w)
                assert np.isfinite(alpha).all()
                delta=abs(alpha-expected)
                record.update(native=m.bind(target/"alpha.f32"),pixels=h*w,mean_absolute=float(delta.mean()),
                    max_absolute=float(delta.max()),raw_mask_differences=int(np.count_nonzero((np.rint(255*alpha)>=38)!=(np.rint(255*expected)>=38))))
                n=(w28+439)//440
            records.append(record)
            return None,alpha,n
        es.run_line=line
        mask,n,_=tracker.predict(rgb,work,caps,None)
        return {"native":mask},{"native":n}
    ref.predict=predict
    if kind in ("sequence","captured"):
        {"sequence":ref.sequence,"captured":ref.captured}[kind](None,cfg)
        file="extended-sequence.json" if kind=="sequence" else "extended-captures.json"
        scores=m.read(folder/file)
        original=m.read(m.checked(cfg["quality"][kind]))
        if phase=="control":
            for a,b in zip(scores["rows"],original["rows"],strict=True):
                if kind=="sequence":assert Counter(a["counts"]["native"])==Counter(b["counts"]["small"])
                else:assert all(v["native"]==b["counts"][k]["small"] for k,v in a["counts"].items())
    else:
        groups=defaultdict(lambda:defaultdict(Counter));history={};events=[]
        for row in m.read(m.checked(cfg["fresh"]))["rows"]:
            if row["event"]=="blank":tracker=q.c.track.Tracker(28,True,True)
            rgb,roles,_=ref.real.nt.read_native(row);state["cover"]=row["proxy_ocr_covers_ltrb_band"]
            boxes=row["work_boxes_ltrb_band"]
            masks,tiles=predict(rgb,row["core_boxes_ltrb_band"],boxes,None)
            roi=ref.s.ev.standard.mask_for(rgb.shape[:2],boxes)
            counts=ref.real.b.measure(groups,history,row,m.truth(roles.transpose(2,0,1)),roi,masks,("all",row["language"],row["case_id"],row["event"]))
            events.append(dict(id=row["id"],event=row["event"],counts=counts))
        scores=dict(runtime=m.runtime(),groups={k:{a:m.summary(v) for a,v in group.items()} for k,group in groups.items()},rows=events)
        if phase=="control":
            original=m.read(m.checked(cfg["fresh_quality"]))
            for a,b in zip(events,original["rows"],strict=True):assert Counter(a["counts"]["native"])==Counter(b["counts"])
        m.dump(folder/"fresh.json",scores)
    m.dump(folder/"inputs.json",dict(runtime=m.runtime(),plan=m.bind(ROOT/"extended-plan.json"),cases=records,frames=len(scores["rows"])))
    print(kind,phase,len(scores["rows"]),len(records),flush=True)

def schedule():
    with (ROOT/"cases.txt").open("x") as f:
        for cohort in ("fresh","captured","sequence"):
            cfg=m.read(ROOT/(cohort+"-control")/"inputs.json")
            for c in cfg["cases"]:print(c["name"],c["width"],c["height"],*c["box"],c["w28"],c["w84"],file=f)

if __name__=="__main__":
    if sys.argv[1]=="plan":plan()
    elif sys.argv[1]=="schedule":schedule()
    else:replay(*sys.argv[1:])
