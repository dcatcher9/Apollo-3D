import numpy as np
import torch
from torch import nn
import neighborhood as seam

class Transfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_span=nn.Parameter(torch.tensor(np.log(32.),dtype=torch.float32))
        self.log_noise=nn.Parameter(torch.tensor(np.log(8.),dtype=torch.float32))
    def forward(self,x,base):
        strength=x[:,:,0]*x[:,:,1]/(x[:,:,1]+self.log_span.exp())
        allowed=strength/(1+(x[:,:,2]/self.log_noise.exp()).square())
        return torch.maximum(base,allowed.amax(1))

def features(rgb,alpha):
    for raw in seam.features(rgb,alpha):
        yield raw[:,:,0],np.expm1(raw[:,:,1])*16,np.expm1(raw[:,:,2])*4

def apply(rgb,alpha,net):
    out=alpha.copy();span=float(net.log_span.exp().detach());noise=float(net.log_noise.exp().detach())
    for support,delta,err in features(rgb,alpha):
        np.maximum(out,support*(delta/(delta+span))/(1+(err/noise)**2),out=out)
    return out
