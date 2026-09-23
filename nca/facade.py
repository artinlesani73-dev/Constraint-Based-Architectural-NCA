"""facade_endpoint_v1: explicit entrance contact allowance, no production default."""
from dataclasses import replace
import hashlib,json
import numpy as np
import torch
from nca.contract import validate_scene,declared_existing,entrance_masks
from nca.losses import _boolean,_occupancy,face_max,LossSpec
from nca.target_audit import joint_bounds
VERSION='facade_endpoint_v1'


def endpoint_allowance(scene,permitted):
    """Allowed contact = typed facade entrance cells face-adjacent to building.

    Uses frozen scene geometry and named entrances only, never guide, scaffold,
    generated material or loss outcomes. No dilation. Ground entrances do not
    acquire a facade exemption. Preserve strict legality and world-space extent.
    """
    scene=validate_scene(scene);shape=(1,)+(scene['grid_size'],)*3
    _boolean(permitted,shape,permitted.device)
    existing=torch.from_numpy(declared_existing(scene))[None].to(permitted.device)
    shell=(face_max(existing.float())>0)&~existing
    endpoints=entrance_masks(scene);allowance=torch.zeros_like(permitted);patches=[]
    for entrance in sorted(scene['entrances'],key=lambda e:e['id']):
        if entrance['kind']!='facade':continue
        mask=torch.from_numpy(endpoints[entrance['id']])[None].to(permitted.device)&shell&permitted
        allowance |= mask
        cells=torch.nonzero(mask[0],as_tuple=False).cpu().tolist()
        patches.append({'entrance_id':entrance['id'],'cells_zyx':cells,'count':len(cells)})
    raw=json.dumps(scene,sort_keys=True,separators=(',',':')).encode()
    annotation={'version':VERSION,'scene_id':scene['scene_id'],'scene_canonical_sha256':hashlib.sha256(raw).hexdigest(),
                'rule':'facade entrance intersection with six-neighbor building shell and permitted region; no dilation',
                'patches':patches,'allowance_voxels':int(allowance.sum())}
    return allowance,annotation


def facade_term(p,context,allowance,spec=LossSpec()):
    """Excess non-allowlisted contact / ALL material; total mass unchanged.

    This still permits dilution by added non-facade material. It is not an
    attachment requirement or mechanical contact model. Empty is not success.
    """
    _occupancy(p);_boolean(allowance,p.shape,p.device)
    if (allowance & ~(context.facade & context.permitted)).any():
        raise ValueError('Allowance must lie in legal facade region')
    mass=p.flatten(1).sum(1).clamp_min(1)
    ratio=(p*context.facade*~allowance).flatten(1).sum(1)/mass
    return torch.relu(ratio-spec.max_facade_ratio)


def facade_bounds(context,contract,allowance,spec=LossSpec()):
    _boolean(allowance,context.facade.shape,context.facade.device)
    if (allowance & ~(context.facade & context.permitted)).any():raise ValueError('Invalid allowance')
    return joint_bounds(replace(context,facade=context.facade & ~allowance),contract,spec)
