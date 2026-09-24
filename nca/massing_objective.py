"""MO1 CPU piecewise-differentiable massing residuals; historical losses unchanged.

Detached graph ordering selects active branches; gradients gather the original
tensors. Ties are nonsmooth and use deterministic choices, not unique derivatives.
"""
from dataclasses import dataclass
import heapq
import math
import numpy as np
import torch
import torch.nn.functional as F
from nca.access import component_strength, _neighbors
from nca.contract import validate_scene, entrance_masks
from nca.massing_targets import MassingTargetSpec, neighbors, evaluate_targets

VERSION = 'massing_residuals_v1'


def _field(p):
    if p.ndim != 3 or p.device.type != 'cpu' or p.dtype not in (torch.float32,torch.float64) or min(p.shape)<1:
        raise ValueError('Expected CPU float32/64 [Z,Y,X]')
    if not torch.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Occupancy must be finite in [0,1]')


def _mask(mask,p):
    if not isinstance(mask,np.ndarray) or mask.dtype!=bool or tuple(mask.shape)!=tuple(p.shape):
        raise ValueError('Expected shape-matched NumPy boolean mask')


def soft_bulk(p,width):
    """Min over complete cubes, max over their union; exact binary opening."""
    _field(p)
    if type(width)!=int or width<1:raise ValueError('Positive integer width required')
    if width>min(p.shape):return p*0
    eroded=-F.max_pool3d(-p[None,None],width,stride=1)
    pad=(width-1,)*6
    return F.max_pool3d(F.pad(eroded,pad,value=0),width,stride=1)[0,0]


def component_excess(p,allowed):
    """Sum merge deficits of all peaks except the global eldest.

    Descending activation merges six-neighbor components. Each younger peak pays
    peak minus merge-level; surviving disconnected domain components pay peak.
    At binary endpoints this is max(number of occupied components - 1, 0).
    Zero-height components contribute zero. Stable flat-index tie handling.
    """
    _field(p);_mask(allowed,p)
    flat=p.reshape(-1);v=p.detach().numpy().ravel();n=len(v)
    candidates=np.flatnonzero(allowed.ravel())
    order=candidates[np.argsort(-v[candidates],kind='stable')]
    parent=np.full(n,-1,np.int64);peak=np.arange(n);pairs=[]
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    for value in order:
        i=int(value);parent[i]=i
        for j in _neighbors(i,tuple(p.shape)):
            if parent[j]<0:continue
            a,b=root(i),root(j)
            if a==b:continue
            if (-v[peak[a]],peak[a]) > (-v[peak[b]],peak[b]):a,b=b,a
            pairs.append((int(peak[b]),i));parent[b]=a
    roots=[int(i) for i in candidates if parent[i]==i]
    roots.sort(key=lambda i:(-v[peak[i]],peak[i]))
    loss=p.sum()*0
    if pairs:
        births,deaths=zip(*pairs)
        loss=loss+torch.relu(flat[list(births)]-flat[list(deaths)]).sum()
    if len(roots)>1:loss=loss+flat[[int(peak[i]) for i in roots[1:]]].sum()
    return loss


def support_strength(p,support):
    """Exact unrestricted widest-path strength from fixed support cells.

    Fixed support transmits one, even through context, matching geometric_support.
    A critical occupancy index is carried along each selected maximum-bottleneck
    path. Non-support cells of zero occupancy transmit zero, not positive leakage.
    """
    _field(p);_mask(support,p)
    values=p.detach().numpy().ravel();fixed=support.ravel();n=len(values)
    dist=np.full(n,-1.,dtype=float);critical=np.full(n,-1,np.int64);heap=[]
    for j in np.flatnonzero(fixed):
        i=int(j);dist[i]=1.;heap.append((-1.,i))
    heapq.heapify(heap)
    while heap:
        negative,i=heapq.heappop(heap);strength=-negative
        if strength!=dist[i]:continue
        for j in _neighbors(i,tuple(p.shape)):
            local=1. if fixed[j] else float(values[j]);proposed=min(strength,local)
            if proposed>dist[j]:
                dist[j]=proposed
                critical[j]=j if not fixed[j] and local<=strength else critical[i]
                heapq.heappush(heap,(-proposed,j))
    idx=torch.from_numpy(np.maximum(critical,0));gathered=p.reshape(-1)[idx]
    reached=torch.where(torch.from_numpy(critical>=0),gathered,torch.from_numpy((dist>=1.).astype(float)).to(p.dtype))
    return reached.reshape(p.shape)


@dataclass(frozen=True)
class MassingContext:
    scene: dict
    masks: dict
    domain: np.ndarray
    endpoints: dict
    thirds: tuple
    contact: np.ndarray
    width: int
    spec: MassingTargetSpec


def make_context(scene,fields,domain,spec=MassingTargetSpec()):
    scene=validate_scene(scene)
    # Use existing validator for input contracts only; no result-specific masks.
    evaluate_targets(np.zeros_like(domain),scene,fields,domain,spec)
    masks={k:fields[k].copy() for k in ('permitted','protected','existing','support_boundary')}
    endpoints=entrance_masks(scene);legal=masks['permitted'];existing=masks['existing']
    contact=neighbors(existing,diagonal=True)&~existing
    shell=neighbors(existing)&~existing
    for e in scene['entrances']:
        if e['kind']=='facade':contact &= ~(endpoints[e['id']]&shell&legal)
    xs=np.flatnonzero(domain.any(axis=(0,1)));thirds=[]
    for i in range(3):
        lo=xs[0]+(xs[-1]+1-xs[0])*i/3;hi=xs[0]+(xs[-1]+1-xs[0])*(i+1)/3
        part=domain&((np.arange(domain.shape[2])+.5>=lo)&(np.arange(domain.shape[2])+.5<hi))[None,None,:]
        if not part.any():raise ValueError('MO1 requires nonempty fixed X thirds')
        thirds.append(part)
    return MassingContext(scene,masks,domain.copy(),endpoints,tuple(thirds),contact,
                          max(1,math.ceil(spec.min_cube_m/scene['voxel_size_m']-1e-10)),spec)


def massing_residuals(p,context):
    """Nine per-scene residuals. Zero iff binary family passes on valid contexts.

    Continuous residual zero is NOT a thresholded-geometry validity certificate.
    No family weights, requested-volume fitting or optimizer are selected here.
    """
    _field(p);_mask(context.domain,p);c=context;s=c.spec
    mass=p.sum();denom=mass.clamp_min(1);available=c.domain&c.masks['permitted']
    q=p*torch.from_numpy(available);bulk=soft_bulk(q,c.width)
    legal=torch.from_numpy(available)
    raw_strength,_=component_strength(q,legal,c.endpoints)
    bulk_strength,_=component_strength(bulk,legal,c.endpoints)
    connection=(1-raw_strength)+(1-bulk_strength)
    disconnected=component_excess(q,available)+component_excess(bulk,available)
    outside=(p-q).sum()
    support=support_strength(p,c.masks['support_boundary'])
    fraction=mass/int(c.domain.sum())
    coverage=torch.stack([torch.relu(p.new_tensor(s.min_third_fraction)-bulk[torch.from_numpy(part)].sum()/int(part.sum())) for part in c.thirds]).max()
    terms={
        'access':connection+(disconnected+outside)/denom,
        'coverage':coverage,
        'facade':torch.relu(p[torch.from_numpy(c.contact)].sum()/denom-s.max_facade_fraction),
        'ground':p[torch.from_numpy(c.masks['protected'])].sum()/denom,
        'legality':p[torch.from_numpy(~c.masks['permitted'])].sum()/denom,
        'sparsity':torch.relu(s.min_volume_fraction-fraction)+torch.relu(fraction-s.max_volume_fraction),
        'spill':p[torch.from_numpy(~c.domain)].sum()/denom,
        'support':torch.relu(1-mass)+torch.relu(p-support).sum()/denom,
        'thickness':torch.relu(s.min_bulk_fraction-bulk.sum()/denom),
    }
    return terms,bulk


def batched_residuals(p,contexts):
    if p.ndim!=4 or len(p)!=len(contexts) or len(p)==0:raise ValueError('One context per batch member required')
    values=[massing_residuals(a,c)[0] for a,c in zip(p,contexts)]
    return {key:torch.stack([v[key] for v in values]) for key in values[0]}
