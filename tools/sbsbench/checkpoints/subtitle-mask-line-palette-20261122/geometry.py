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
