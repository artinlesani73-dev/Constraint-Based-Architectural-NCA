"""Experimental component_bottleneck_v2; no existing objective/default is changed.

Strength is the largest voxel threshold at which ONE six-connected component
touches all entrance regions. A descending union-find computes the exact value.
The returned tensor gathers the critical voxel: a piecewise-linear derivative
almost everywhere, with deterministic tie selection. CPU topology selection is
detached; this is not a smooth relaxation or a GPU implementation.
"""
from collections import deque
import numpy as np
import torch
from nca.losses import _occupancy, _boolean

VERSION = 'component_bottleneck_v2'


def _regions(endpoints, shape):
    if len(endpoints) < 2:
        raise ValueError('Need at least two entrance regions')
    used = np.zeros(shape, dtype=bool)
    result = []
    for name, region in sorted(endpoints.items()):
        if isinstance(region, torch.Tensor):
            region = region.detach().cpu().numpy()
        region = np.asarray(region)
        if not isinstance(name,str) or not name or region.dtype != bool or region.shape != shape or not region.any() or (used & region).any():
            raise ValueError('Entrance IDs/masks must be unique, nonempty, disjoint and shape-matched')
        used |= region
        result.append((name,region))
    return result


def _neighbors(index, shape):
    depth,height,width=shape; plane=height*width
    z,rem=divmod(index,plane);y,x=divmod(rem,width)
    if z: yield index-plane
    if z+1<depth: yield index+plane
    if y: yield index-width
    if y+1<height: yield index+width
    if x: yield index-1
    if x+1<width: yield index+1


def component_strength(material, permitted, endpoints):
    """One [D,H,W] CPU field -> differentiable scalar and topology diagnostics."""
    if material.ndim != 3 or material.device.type != 'cpu':
        raise ValueError('Expected CPU [D,H,W] material')
    _occupancy(material[None]); _boolean(permitted,material.shape,material.device)
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
    return material.sum()*0.,{'version':VERSION,'critical_zyx':None,'legal_route_exists':False,
        'entrance_ids':[name for name,_ in regions]}


def component_access(material, permitted, endpoints):
    """Batched per-scene loss. Worst destination and best common component."""
    _occupancy(material);_boolean(permitted,material.shape,material.device)
    if len(endpoints)!=len(material):raise ValueError('One entrance mapping per sample required')
    scored=[component_strength(p,mask,regions) for p,mask,regions in zip(material,permitted,endpoints)]
    return torch.stack([1-strength for strength,_ in scored]),[details for _,details in scored]


def component_connectivity(material, permitted, endpoints, threshold=.5):
    """Independent binary BFS oracle, with no multi-origin flood initialization."""
    material=np.asarray(material);permitted=np.asarray(permitted)
    if material.ndim!=3 or permitted.shape!=material.shape or permitted.dtype!=bool or not np.isfinite(material).all() or ((material<0)|(material>1)).any():
        raise ValueError('Invalid fields')
    if isinstance(threshold,bool) or not np.isfinite(threshold) or not 0<=threshold<1:
        raise ValueError('Threshold must be in [0,1)')
    regions=_regions(endpoints,material.shape)
    occupied=(material>threshold)&permitted;seen=np.zeros_like(occupied);count=0;spanning=0
    for seed_array in np.argwhere(occupied):
        seed=tuple(seed_array)
        if seen[seed]:continue
        count+=1;seen[seed]=True;queue=deque([seed]);touched=set()
        while queue:
            cell=queue.popleft()
            touched.update(name for name,region in regions if region[cell])
            # Deliberately separate coordinate BFS from union-find's flat indexing.
            for axis in range(3):
                for step in (-1,1):
                    q=list(cell);q[axis]+=step;q=tuple(q)
                    if all(0<=q[i]<occupied.shape[i] for i in range(3)) and occupied[q] and not seen[q]:
                        seen[q]=True;queue.append(q)
        spanning+=len(touched)==len(regions)
    return {'version':VERSION,'threshold':threshold,'all_connected':spanning>0,
        'material_components':count,'components_touching_all_entrances':spanning}
