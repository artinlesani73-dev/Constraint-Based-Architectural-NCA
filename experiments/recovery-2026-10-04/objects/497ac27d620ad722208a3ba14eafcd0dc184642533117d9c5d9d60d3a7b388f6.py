"""regularizers_v1: notebook-faithful density/TV plus explicit cantilever variants."""
import torch
import torch.nn.functional as F
from nca.losses import _occupancy,_boolean
VERSION='regularizers_v1'
NAMES=('density_binary','tv','cantilever_boundary','cantilever_historical')


def density_binary(p):
    _occupancy(p)
    return (p*(1-p)).flatten(1).mean(1)


def total_variation(p):
    _occupancy(p)
    if min(p.shape[1:])<2:raise ValueError('TV requires at least two cells on each spatial axis')
    return sum((p.diff(dim=axis)).abs().flatten(1).mean(1) for axis in (1,2,3))


def cantilever_historical(p,max_overhang=3,threshold=.3):
    """Per-scene transcription of notebook cell19, including historical limits."""
    _occupancy(p)
    if isinstance(max_overhang,bool) or not isinstance(max_overhang,int) or max_overhang<1:raise ValueError('Invalid overhang')
    terms=[]
    for z in range(max_overhang,p.shape[1]):
        below=p[:,z-max_overhang:z].max(1).values
        support=F.max_pool2d(below[:,None],3,1,1)[:,0]
        terms.append((p[:,z]*(1-torch.sigmoid(10*(support-threshold)))).flatten(1).mean(1))
    return torch.stack(terms).mean(0) if terms else p.flatten(1).sum(1)*0


def cantilever_boundary(p,support_boundary,depth=3):
    """Local geometric proxy, not engineering safety or a max-span guarantee.

    Every layer participates. Fixed boundary at the cell is supported. Otherwise
    take max strength in previous1..depth layers within a3x3 horizontal stencil,
    with zero outside volume. No sigmoid empty-space support or wraparound.
    """
    _occupancy(p);_boolean(support_boundary,p.shape,p.device)
    if isinstance(depth,bool) or not isinstance(depth,int) or depth<1:raise ValueError('Invalid depth')
    occupied=torch.maximum(p,support_boundary.to(p.dtype))
    b,d,h,w=p.shape
    horizontal=F.max_pool2d(occupied.reshape(b*d,1,h,w),3,1,1).reshape(b,d,h,w)
    layers=[]
    for z in range(d):
        strength=support_boundary[:,z].to(p.dtype)
        if z:strength=torch.maximum(strength,horizontal[:,max(0,z-depth):z].max(1).values)
        layers.append(p[:,z]*(1-strength))
    return torch.stack(layers,dim=1).flatten(1).mean(1)


def regularizer_terms(p,support_boundary):
    return {'density_binary':density_binary(p),'tv':total_variation(p),
            'cantilever_boundary':cantilever_boundary(p,support_boundary),
            'cantilever_historical':cantilever_historical(p)}
