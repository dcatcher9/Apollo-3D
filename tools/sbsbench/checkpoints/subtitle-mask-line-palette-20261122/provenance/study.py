"""Test actual OCR-bounded changes in the native support rectangle on unchanged observations."""
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"rawmatte-line-palette-1122"))
import palette_fit as p
q,es,m=p.q,p.es,p.m
ref=q.ref
ARMS=("checkpoint","original","small","medium","medium_y","eighth")


def plan():
    doc=m.read(p.ROOT/"extended-plan.json")
    doc.update(arms=list(ARMS),geometry_script=m.bind(Path(__file__)),checkpoint=m.bind(q.ROOT/"summary.json"),
        model=m.bind(p.ROOT/"trained.json"),diagnosis=m.bind(p.ROOT/"marked-diagnosis.json"),
        controls={name:m.bind(p.ROOT/(name+"-verified.json")) for name in ("captured","sequence")},
        design="One H28 canonical scale of the entire working rectangle for each arm. Test 1/16 or 1/12 of the work height on every side, 1/12 vertically, and the previous 1/8 control. All marked points outside the original work are within 7.61 percent of its edge. Never cross that cover. Score from zero history for points and the same causal tracker for sequences. Only actual OCR observations, identical threshold and model weights. Keep the original mask ROI as the denominator; whole-band FP counts include any extra protected background.")
    m.dump(ROOT/"extended-plan.json",doc)


def windows(boxes,covers,arm):
    if arm in ("original","checkpoint"):return boxes
    if arm=="detector":return covers
    divisor={"small":16,"medium":12,"medium_y":12,"eighth":8}[arm]
    result=[]
    for (l,t,r,b),(cl,ct,cr,cb) in zip(boxes,covers,strict=True):
        pad=(b-t+divisor-1)//divisor
        pad_x=0 if arm=="medium_y" else pad
        result.append([max(cl,l-pad_x),max(ct,t-pad),min(cr,r+pad_x),min(cb,b+pad)])
    return result


def run(mode):
    torch.set_num_threads(2)
    doc=m.read(ROOT/"extended-plan.json");assert doc["runtime"]==m.runtime()
    for name in ("geometry_script","model","diagnosis"):m.checked(doc[name])
    models=es.load_models();net=p.load();transfer=q.r.load()
    trackers={arm:q.c.track.Tracker(28,True,mode!="captured") for arm in ARMS}
    geometry=ref.geometry;state={}
    def geometry_with_cap(record,dims,content):
        core,boxes=geometry(record,dims,content);words=np.frombuffer(record,"<u4")
        state["cap"]=[ref.s.expected_rect(words[144+8*i:148+8*i],content,*dims) for i in range(int(words[4]))]
        return core,boxes
    def predict(rgb,cores,boxes,_):
        cache={};covers=state.pop("cap");out={};tiles={}
        for arm,tracker in trackers.items():
            older=arm=="checkpoint"
            def line(image,_):
                key=(image.ctypes.data,image.shape,image.strides,older)
                if key not in cache:
                    if older:
                        _,alpha,tiles=q.c.line_original(image,models)
                        alpha=q.r.apply(image,alpha,transfer)
                    else:
                        alpha,tiles=p.native(image,models[0],net)
                    cache[key]=(None,alpha,tiles)
                return cache[key]
            es.run_line=line
            work=windows(boxes,covers,arm)
            out[arm],tiles[arm],_=tracker.predict(rgb,work,covers,None)
        return out,tiles
    ref.ROOT=ROOT;ref.arms=ARMS;ref.geometry=geometry_with_cap;ref.predict=predict
    {"captured":ref.captured,"sequence":ref.sequence}[mode](None,doc)
    filename="extended-captures.json" if mode=="captured" else "extended-sequence.json"
    actual=m.read(ROOT/filename);control=m.read(m.checked(doc["controls"][mode]))
    assert actual["runtime"]==control["runtime"]==m.runtime()
    for row,old in zip(actual["rows"],control["rows"],strict=True):
        assert all(row.get(k)==old.get(k) for k in ("id","event","ordinal"))
        if mode=="captured":
            for label,d in row["counts"].items():
                assert (d["checkpoint"],d["original"])==(old["counts"][label]["prior"],old["counts"][label]["learned"])
        else:
            for arm,previous in (("checkpoint","prior"),("original","learned")):
                assert Counter(row["counts"][arm])==Counter(old["counts"][previous])
    m.dump(ROOT/(mode+"-verified.json"),dict(actual,matched_controls=doc["controls"][mode]))
    if mode=="captured":print(actual["totals"])
    else:print({k:[round(v["f1"],5),v["clear_fp"],v["erase_selected"],v["swap_selected"]] for k,v in actual["groups"]["all"].items()})


if __name__=="__main__":
    if sys.argv[1]=="plan":plan()
    else:run(sys.argv[1])
