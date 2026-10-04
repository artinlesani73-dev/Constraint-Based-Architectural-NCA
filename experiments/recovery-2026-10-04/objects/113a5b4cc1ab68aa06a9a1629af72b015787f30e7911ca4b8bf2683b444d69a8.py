"""Opt-in signed raw maximin access; no historical objective is changed.

Detached stable z/y/x topology selects a branch at nonsmooth ties, not a unique
classical derivative. An impossible legal graph returns loss one, explicit
infeasibility and zero gradient. CPU float32/64 only.
"""
import numpy as np
import torch
from nca.access import _neighbors, _regions
from nca.losses import _boolean
VERSION = 'raw_component_bottleneck_v3'


def _raw(value):
    if value.ndim != 4 or value.device.type != 'cpu' or value.dtype not in (torch.float32, torch.float64) or min(value.shape) < 1 or not torch.isfinite(value).all():
        raise ValueError('Expected nonempty finite CPU float32/64 [B,D,H,W] raw values')


def raw_component_strength(material, permitted, endpoints):
    """One [D,H,W] CPU field -> differentiable scalar and topology diagnostics."""
    if material.ndim != 3 or material.device.type != 'cpu':
        raise ValueError('Expected CPU [D,H,W] material')
    _raw(material[None]); _boolean(permitted,material.shape,material.device)
    regions=_regions(endpoints,tuple(material.shape));shape=tuple(material.shape)
    legal=permitted.detach().numpy().reshape(-1);values=material.detach().numpy().reshape(-1)
    candidates=np.flatnonzero(legal)
    # Stable sort gives lexicographic z,y,x tie handling.
    order=candidates[np.argsort(-values[candidates],kind='stable')]
    n=len(values);parent=np.full(n,-1,dtype=np.int64);sizes=np.ones(n,dtype=np.int64)
    bits=[0]*n;membership=[0]*n
    for j,(_,region) in enumerate(regions):
        for index in np.flatnonzero(region.reshape(-1)&legal): membership[int(index)] |= 1<<j
    required=(1<<len(regions))-1
    def root(index):
        while parent[index]!=index:
            parent[index]=parent[parent[index]];index=int(parent[index])
        return index
    for entry in order:
        index=int(entry);parent[index]=index;bits[index]=membership[index]
        for neighbor in _neighbors(index,shape):
            if parent[neighbor]<0:continue
            a,b=root(index),root(neighbor)
            if a==b:continue
            if sizes[a]<sizes[b]:a,b=b,a
            parent[b]=a;sizes[a]+=sizes[b];bits[a]|=bits[b]
        if bits[root(index)]==required:
            return material.reshape(-1)[index],{'version':VERSION,'critical_zyx':[int(x) for x in np.unravel_index(index,shape)],
                'legal_route_exists':True,'entrance_ids':[name for name,_ in regions]}
    # Even all legal cells cannot connect the regions; retain a zero gradient.
    return material.reshape(-1)[0]*0.,{'version':VERSION,'critical_zyx':None,'legal_route_exists':False,
        'entrance_ids':[name for name,_ in regions]}



def raw_component_access(raw, permitted, endpoints):
    """Same legal graph/entrances; relu(1-b_raw), with unchanged forward."""
    _raw(raw); _boolean(permitted,raw.shape,raw.device)
    if len(endpoints) != len(raw):
        raise ValueError('One entrance mapping per sample required')
    scored = [raw_component_strength(p, mask, regions) for p, mask, regions in zip(raw, permitted, endpoints)]
    return torch.stack([torch.relu(1-strength) for strength, _ in scored]), [details for _, details in scored]
