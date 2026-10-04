"""Proposed G10 TRAIN-only semi-gradient ranking. Not integrated or trained."""
import math
import torch
from torch.nn import functional as F

def access_ranking(logits,eligible_fired,origins,positive,phase):
    if phase not in ('connected','no_teacher_route','seed_access','advance_access'):
        raise ValueError('Unknown supervision phase')
    zero=logits.sum()*0
    if phase in ('connected','no_teacher_route'):return zero
    progress=logits[eligible_fired & origins & positive]
    other=logits[eligible_fired & origins & ~positive]
    if not progress.numel() or not other.numel():return zero
    # Numerically the G9 loss, with other scores a detached comparison reference.
    # This is an explicit semi-gradient, not the full derivative of symmetric ranking.
    gap=1.0+torch.logsumexp(other.detach(),0)-math.log(other.numel())
    gap=gap+torch.logsumexp(-progress,0)-math.log(progress.numel())
    return F.softplus(gap)

