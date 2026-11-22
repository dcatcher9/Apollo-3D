"""Replay all three cohorts with the refitted head and the frozen geometry/history."""
from collections import Counter,defaultdict
from pathlib import Path
import sys
import numpy as np
import torch
import importlib.util
spec=importlib.util.spec_from_file_location("matte_projection_refit",Path(__file__).with_name("train.py"))
refit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(refit)
import cues
ROOT=Path(__file__).resolve().parent
prev=ROOT.parent/"rawmatte-host-sequence-1122"
sys.path.insert(0,str(prev))
import replay
p,m,ref,es,q=replay.p,replay.m,replay.ref,replay.es,replay.q
def run(kind):
    torch.set_num_threads(2)
    folder=ROOT/kind;folder.mkdir()
    cfg=m.read(prev/"extended-plan.json")
    assert m.runtime()==cfg["runtime"]==m.read(ROOT/"trained.json")["runtime"]
    net=refit.load();encoder=es.load_models()[0]
    ref.ROOT=folder;ref.arms=("fit",)
    tracker=q.c.track.Tracker(28,True,kind!="captured")
    geometry=ref.geometry;state={}
    def read(data,dims,content):
        core,boxes=geometry(data,dims,content);u=np.frombuffer(data,"<u4")
        state["cap"]=[ref.s.expected_rect(u[144+8*i:148+8*i],content,*dims) for i in range(int(u[4]))]
        return core,boxes
    def line(crop,_):
        alpha,n=cues.native(crop,encoder,net)
        return None,alpha,n
    def predict(rgb,cores,boxes,_):
        cap=state.pop("cap");es.run_line=line
        mask,n,decisions=tracker.predict(rgb,replay.study.windows(boxes,cap,"small"),cap,None)
        state["history"]=decisions
        return {"fit":mask},{"fit":n}
    ref.predict=predict;ref.geometry=read
    m.dump(folder/"extended-plan.json",cfg)
    if kind=="fresh":
        groups,memory,rows=defaultdict(lambda:defaultdict(Counter)),{},[]
        for row in m.read(m.checked(cfg["fresh"]))["rows"]:
            if row["event"]=="blank":tracker=q.c.track.Tracker(28,True,True)
            rgb,roles,_=ref.real.nt.read_native(row)
            boxes=row["work_boxes_ltrb_band"];state["cap"]=row["proxy_ocr_covers_ltrb_band"]
            masks,_=predict(rgb,row["core_boxes_ltrb_band"],boxes,None)
            roi=ref.s.ev.standard.mask_for(rgb.shape[:2],boxes)
            values=ref.real.b.measure(groups,memory,row,m.truth(roles.transpose(2,0,1)),roi,masks,
                ("all",row["language"],row["case_id"],row["event"]))
            rows.append(dict(id=row["id"],counts=values["fit"],history=state["history"]))
        totals={k:m.summary(v["fit"]) for k,v in groups.items()}
        m.dump(folder/"scores.json",dict(runtime=m.runtime(),groups=totals,rows=rows))
        print(kind,totals["all"])
    else:{"captured":ref.captured,"sequence":ref.sequence}[kind](None,cfg)
    m.dump(folder/"verified.json",dict(runtime=m.runtime(),model=m.bind(ROOT/"trained.json"),
        control=m.bind(prev/(kind+"-control")/"inputs.json"),script=m.bind(Path(__file__))))
if __name__=="__main__":run(sys.argv[1])
