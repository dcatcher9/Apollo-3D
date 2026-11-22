import numpy as np

def size(b):
    return max(0,b[2]-b[0])*max(0,b[3]-b[1])

def intersection(a,b):
    return max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])

def overlaps(a,b):
    common=size(intersection(a,b));return common>.5*(size(a)+size(b)-common)

def edges(rgb,step):
    luma=(rgb[:,:,0].astype('i4')+2*rgb[:,:,1].astype('i4')+rgb[:,:,2])//4
    return np.stack([luma[2:-2:step,4::step]-luma[2:-2:step,:-4:step],luma[4::step,2:-2:step]-luma[:-4:step,2:-2:step]],axis=-1)

def confirmation(old,current,previous_alpha):
    h,w=old.shape[:2]
    if min(h,w)<5:return 0,0.
    stride=max(2,w//250)
    a=edges(old,stride);b=edges(current,stride)
    power=(a*a).sum(2)
    ink=previous_alpha[2:-2:stride,2:-2:stride]>=38/255
    samples=(power>=24**2)&ink
    dot=(a*b).sum(2)
    agrees=(dot>0)&(dot.astype('f8')**2>=.7*power*(b*b).sum(2))
    delta=np.abs(current[2:-2:stride,2:-2:stride].astype('i2')-old[2:-2:stride,2:-2:stride]).max(2)
    n=int(samples.sum())
    return n,float((samples&agrees&(delta<=32)).sum()/max(1,n))
