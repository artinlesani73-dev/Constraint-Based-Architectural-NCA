"""TRAIN-only access ranking supplement; inference and mass loss are unchanged."""
import math
import torch
from torch.nn import functional as F
from nca.paced_generation import block_loss
from nca.access_labels import teacher_graph, priority

VERSION = 'train_access_ranking_v2_one_sided'
MARGIN = 1.0
WEIGHT = 1.0

def access_ranking(logits, eligible_fired, origins, positive, phase):
    if phase not in ('connected','no_teacher_route','seed_access','advance_access'):
        raise ValueError('Unknown supervision phase')
    zero = logits.sum()*0
    if phase in ('connected','no_teacher_route'):
        return zero
    progress = logits[eligible_fired & origins & positive]
    other = logits[eligible_fired & origins & ~positive]
    if not progress.numel() or not other.numel():
        return zero
    # log(1 + mean_{a,b} exp(1 + z_b - z_a)), stable without an A-by-B array.
    # A=advancing teacher origins, B=other teacher origins. Both remain BCE positives.
    # Explicit semi-gradient: other scores are a detached comparison reference.
    gap = MARGIN + torch.logsumexp(other.detach(),0)-math.log(other.numel())
    gap = gap + torch.logsumexp(-progress,0)-math.log(progress.numel())
    return F.softplus(gap)

def ranked_loss(logits,m,eligible_fired,target,origins,seed_phase,D,B,C,positive,phase):
    base,front,volume,band = block_loss(logits,m,eligible_fired,target,origins,seed_phase,D,B,C)
    rank = access_ranking(logits,eligible_fired,origins,positive,phase)
    return base + WEIGHT*rank,front,volume,band,rank
