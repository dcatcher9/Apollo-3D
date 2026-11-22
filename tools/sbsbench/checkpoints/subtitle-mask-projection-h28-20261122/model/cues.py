"""Add a foreground/background compositing cue to the native head without another image pass."""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"rawmatte-line-palette-1122"))
import palette_fit as p
class Context(p.Context):
    def features(self,y,x):
        base=super().features(y,x)
        color=base[:,9:12]
        additions=[]
        # Recover the two local means and noise directly from channels the native decoder already reads.
        for at,noise_at in ((15,21),(22,28)):
            d=base[:,at:at+3]/5
            mu=color-d
            noise=.025+base[:,noise_at]/5
            for ink in self.inks:
                v=ink[None,:]/255-mu
                norm=(v*v).sum(1)
                projection=np.clip((d*v).sum(1)/(norm+.0001),0,1)
                residual=d-v*projection[:,None]
                error=np.sqrt((residual*residual).sum(1)/3)
                reliability=norm/(norm+3*noise**2)
                additions.extend((projection*reliability,np.clip(error/noise,0,6)*reliability))
        return np.ascontiguousarray(np.column_stack((base,*additions)),dtype="f4")
def model():
    old=p.load();new=p.z.Net(50)
    with torch.no_grad():
        new.layers[0].weight.zero_()
        new.layers[0].weight[:,:42].copy_(old.layers[0].weight)
        new.layers[0].bias.copy_(old.layers[0].bias)
        for n in (2,4):new.layers[n].load_state_dict(old.layers[n].state_dict())
    return new.eval()
def native(rgb,encoder,net):
    ctx=Context(rgb,encoder);h,w=rgb.shape[:2];a=np.empty(h*w,"f4")
    with torch.inference_mode():
        for start in range(0,h*w,65536):
            sl=slice(start,min(h*w,start+65536));y,x=np.divmod(np.arange(sl.start,sl.stop),w)
            a[sl]=net(torch.from_numpy(ctx.features(y,x)))[:,0].sigmoid().numpy()
    return a.reshape(h,w),ctx.tiles
