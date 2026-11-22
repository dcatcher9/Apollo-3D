"""Export the separate native protection transition to ONNX with dynamic crop dimensions."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import onnx,onnxruntime as ort
import importlib.util
_spec=importlib.util.spec_from_file_location('native_video_transition',Path(__file__).with_name('refine.py'))
r=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r)

class Transition(nn.Module):
    def __init__(self,span,noise):
        super().__init__()
        self.register_buffer('span',torch.tensor(span))
        self.register_buffer('noise',torch.tensor(noise))
    def forward(self,rgb,alpha):
        color=F.pad(rgb,(1,1,1,1),mode='replicate')
        coverage=F.pad(alpha,(1,1,1,1),mode='replicate')
        out=alpha
        for dy,dx in r.seam.DIRECTIONS:
            def offset(oy,ox):
                return (slice(None),slice(None),slice(1+oy,(oy-1) or None),slice(1+ox,(ox-1) or None))
            a=color[offset(dy,dx)];b=color[offset(-dy,-dx)]
            d=b-a
            t=(((rgb-a)*d).sum(1,True)/(d*d).sum(1,True).clamp_min(1.)).clamp(0.,1.)
            error=(rgb-a-t*d).abs().amax(1,True)
            span=d.abs().amax(1,True)
            support=torch.minimum(coverage[offset(dy,dx)],coverage[offset(-dy,-dx)])
            update=support*span/(span+self.span)/(1.+(error/self.noise).square())
            out=torch.maximum(out,update)
        return out

def main():
    spec=r.m.read(r.ROOT/'trained.json');assert spec['runtime']==r.m.runtime()
    model=Transition(spec['span'],spec['noise']).eval()
    path=r.ROOT/'glyph_transition.onnx';net=r.load()
    axes={k:{2:'height',3:'width'} for k in ('native_rgb8','input_alpha','alpha')}
    torch.onnx.export(model,(torch.zeros(1,3,23,71),torch.zeros(1,1,23,71)),path,input_names=['native_rgb8','input_alpha'],output_names=['alpha'],dynamic_axes=axes,opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(path),full_check=True)
    options=ort.SessionOptions();options.intra_op_num_threads=2
    sess=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    compare=r.m.read(r.ROOT.parent/'control.json')
    with np.load(r.m.checked(compare['inputs'])) as archive:cases=[(archive['rgb'],archive['alpha'])]
    rng=np.random.default_rng(94298)
    for h,w in ((1,1),(1,27),(41,1),(11,19),(80,640)):
        cases.append((rng.integers(256,size=(h,w,3),dtype='u1'),rng.random((h,w),dtype='f4')))
    results=[]
    for rgb,a in cases:
        z=sess.run(None,dict(native_rgb8=np.ascontiguousarray(rgb.transpose(2,0,1)[None],dtype='f4'),input_alpha=a[None,None]))[0][0,0]
        correct=r.apply(rgb,a,net)
        changes=int(((np.rint(z*255)>=38)!=(np.rint(correct*255)>=38)).sum())
        err=float(np.abs(z-correct).max());assert changes==0 and err<2e-5
        results.append(dict(shape=list(a.shape),max_alpha_error=err,changed_mask_pixels=changes))
    r.m.dump(r.ROOT/'export.json',dict(runtime=r.m.runtime(),trained=r.m.bind(r.ROOT/'trained.json'),model=r.m.bind(path),parity=results,
             scope='Single native transition stage, FP32 RGB byte units and continuous alpha in [0, 1]. Mixed backbone and prior native decoder are separate.'))
    print(results)
if __name__=='__main__':main()

