"""budgeted_witness_v1: deterministic constructive baseline inside a fixed envelope."""
import math
import torch
from nca.losses import LossSpec,face_max,zero_padded_erosion
from nca.facade import facade_bounds
VERSION='budgeted_witness_v1'


def build_witness(context,allowance,spec=LossSpec()):
    """Grow a binary guide only to meet existing mass/contact lower bounds.

    B=1 CPU diagnostic. Add face-adjacent uncharged cells in z,y,x order within
    the fixed legal envelope. Refuse additions creating an eroded radius-r core.
    Never expand the region/budget. Return explicit exhaustion/incompatibility;
    success describes construction bounds only, requiring independent evaluation.
    """
    if context.coverage.shape[0]!=1 or context.coverage.device.type!='cpu':raise ValueError('Witness requires one CPU scene')
    p=context.coverage.clone();bounds=facade_bounds(context,'envelope',allowance,spec)
    if not bool(context.route_feasible[0]) or not bool(bounds['joint_necessary_compatible'][0]):
        return {'version':VERSION,'status':'incompatible','material':p,'added_cells':[]}
    lower=float(bounds['joint_minimum_mass'][0]);upper=float(bounds['joint_maximum_mass'][0])
    target=math.ceil(lower-1e-7)
    if target>upper+1e-7 or bool(zero_padded_erosion(p.float(),spec.thickness_radius).any()):
        return {'version':VERSION,'status':'no_binary_zero_core_target','material':p,'added_cells':[]}
    charged=context.facade&~allowance
    available=context.envelope&context.permitted&~charged
    added=[]
    while int(p.sum())<target:
        frontier=(face_max(p.float())>0)&available&~p
        accepted=False
        for cell in torch.nonzero(frontier[0],as_tuple=False).tolist():
            z,y,x=cell;p[0,z,y,x]=True
            if not zero_padded_erosion(p.float(),spec.thickness_radius).any():
                added.append(cell);accepted=True;break
            p[0,z,y,x]=False
        if not accepted:return {'version':VERSION,'status':'frontier_exhausted','material':p,'added_cells':added,'target_mass':target}
    return {'version':VERSION,'status':'constructed','material':p,'added_cells':added,'target_mass':target}
