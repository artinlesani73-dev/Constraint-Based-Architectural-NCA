"""TRAIN-design reference only. Global budget guard, not a trained G3 model."""
import math
import numpy as np
import torch

VERSION='global_volume_budget_reference_v1'

def budget(domain_cells,request,width=3):
    if type(domain_cells) is not int or domain_cells<=0 or type(width) is not int or width<1:
        raise ValueError('Positive integer domain and cube width required')
    # G3 pilot contract is limited to the three existing exact request fractions.
    requests=(.16,.24,.32)
    matches=[r for r in requests if math.isfinite(float(request)) and abs(float(request)-r)<1e-7]
    if len(matches)!=1:raise ValueError('Pilot request must be16,24 or32 percent')
    target=math.ceil(domain_cells*matches[0]);ceiling=min(target+width*width-1,math.floor(.40*domain_cells+1e-10))
    if target>ceiling:raise ValueError('No representable budget band')
    return target,ceiling

def admit(m,eligible,probability,ceiling):
    m=np.asarray(m);eligible=np.asarray(eligible);q=np.asarray(probability)
    if m.dtype!=bool or eligible.dtype!=bool or m.shape!=eligible.shape or m.shape!=q.shape or not np.isfinite(q).all() or ((q<0)|(q>1)).any():
        raise ValueError('Matching Boolean fields and finite probabilities required')
    if type(ceiling) is not int or int(m.sum())>ceiling:raise ValueError('Start exceeds budget; no silent deletion')
    offered=np.flatnonzero(eligible & ~m & (q>.5));room=ceiling-int(m.sum())
    order=np.lexsort((offered,-q.ravel()[offered]))
    selected=offered[order[:room]];born=np.zeros_like(m);born.ravel()[selected]=True
    return born,dict(offered=len(offered),admitted=len(selected),budget_rejected=len(offered)-len(selected))

def feedback(m,domain_cells,target):
    return (target-int(np.asarray(m,dtype=bool).sum()))/domain_cells

def band_loss(logits,m,eligible,domain_cells,target,ceiling):
    # Deliberately use proposals BEFORE hard admission. No gradient through ranking.
    soft_count=m.detach().float().sum()+(eligible.detach().float()*torch.sigmoid(logits)).sum()
    return (torch.relu(target-soft_count)+torch.relu(soft_count-ceiling))/domain_cells
