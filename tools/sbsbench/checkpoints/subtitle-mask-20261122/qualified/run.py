"""Restore the qualified H28 model from this directory and run it on an extended RGB subtitle crop."""
from pathlib import Path
from hashlib import sha256
import argparse,json
import numpy as np
import torch
import onnxruntime as ort
from PIL import Image
from native import NativeHead
from support import Context,Residual
from transition import Transfer,apply
from tracking import Tracker

ROOT=Path(__file__).resolve().parent
class Masker:
    def __init__(self):
        model=ROOT/'model'
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(model/'mixed_rgb.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
        self.input=self.session.get_inputs()[0].name;self.output=self.session.get_outputs()[0].name
        self.head=NativeHead();self.residual=Residual();self.transition=Transfer()
        for net,file in [(self.head,'native-head.pt'),(self.residual,'residual.pt'),(self.transition,'transition.pt')]:
            net.load_state_dict(torch.load(model/file,weights_only=True,map_location='cpu'));net.eval()
        self.tracker=Tracker(28,True,True)
    def encode(self,tensor):
        return self.session.run([self.output],{self.input:tensor})[0]
    def score(self,rgb):
        if rgb.ndim!=3 or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8 or not min(rgb.shape[:2]):raise ValueError('Expected a nonempty RGB8 crop')
        cx=Context(rgb,self.encode,self.head);h,w=rgb.shape[:2];alpha=np.empty((h,w),np.float32)
        for i in range(0,h*w,65536):
            y,x=np.divmod(np.arange(i,min(h*w,i+65536)),w)
            with torch.inference_mode():p=self.residual(torch.from_numpy(cx.features(y,x))).numpy()
            alpha[y,x]=p[:,0]+(1-p[:,0])*p[:,1]
        return None,apply(rgb,alpha,self.transition),cx.line.tiles
    def alpha(self,rgb):return self.score(rgb)[1]
    def frame(self,rgb,work,cover):
        """Supply current OCR work/cap rectangles [left,top,right,bottom] in native band coordinates."""
        return self.tracker.predict(rgb,work,cover,self)[0]

def digest(arr):return sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
def vector():
    data=np.load(ROOT/'test-input.npz')
    masker=Masker();out={}
    for name in ('a','b'):
        a=masker.alpha(data[name]);out[name]=digest(np.rint(a*255).astype('u1'))
    h,w=data['a'].shape[:2]
    for idx,(key,margin,enabled) in enumerate([('a',0,True),('a',1,True),('b',0,True),('b',0,False)]):
        boxes=[[margin,0,w-margin,h]] if enabled else []
        mask=masker.frame(data[key],boxes,[[0,0,w,h]] if enabled else [])
        out['frame'+str(idx)]=digest(mask)
    return out
def verify():
    index=json.loads((ROOT/'sha256.json').read_text())
    for name,expected in index.items():
        path=ROOT/name
        assert digest(np.frombuffer(path.read_bytes(),np.uint8))==expected,name
    assert vector()==json.loads((ROOT/'test-expected.json').read_text())
    print('Verified all checkpoint files, two native alpha maps and four temporal mask frames.')
if __name__=='__main__':
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('verify')
    for cmd in ('mask','alpha'):
        p=sub.add_parser(cmd);p.add_argument('image',type=Path);p.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.command=='verify':verify()
    else:
        with Image.open(args.image) as im:rgb=np.asarray(im.convert('RGB'))
        alpha=np.rint(Masker().alpha(rgb)*255).astype('u1')
        Image.fromarray(alpha if args.command=='alpha' else (alpha>=38).astype('u1')*255).save(args.output)

