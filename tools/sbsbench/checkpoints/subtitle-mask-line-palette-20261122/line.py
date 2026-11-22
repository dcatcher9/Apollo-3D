import numpy as np
import torch
from scipy.ndimage import map_coordinates
import roles
import background as z

class Context(z.Context):
    def __init__(self, image, encoder):
        super().__init__(image,encoder)
        self.color_posterior,self.inks=roles.palette(image,self.roles)

    def features(self,y,x):
        local=super().features(y,x)
        color=self.rgb[y,x].astype("f4")
        votes=[map_coordinates(table,(color/16).T,order=1,mode="nearest",prefilter=False)
               for table in self.color_posterior[1:]]
        deltas=[np.abs(color-ink).mean(1)/64 for ink in self.inks]
        return np.ascontiguousarray(np.column_stack((local,*votes,*deltas)),dtype="f4")

def native(rgb,enc,model):
    ctx=Context(rgb,enc);h,w=rgb.shape[:2];a=np.empty((h,w),"f4")
    for i in range(0,h*w,65536):
        y,x=np.divmod(np.arange(i,min(h*w,i+65536)),w)
        with torch.inference_mode():a[y,x]=model(torch.from_numpy(ctx.features(y,x)))[:,0].sigmoid().numpy()
    return a,ctx.tiles
