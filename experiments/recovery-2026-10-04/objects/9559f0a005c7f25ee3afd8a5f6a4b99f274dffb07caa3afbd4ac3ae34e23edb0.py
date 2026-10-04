"""TRAIN-only temporal supervision. Never use this teacher graph at inference."""
from collections import deque
import numpy as np
import torch
from torch.nn import functional as F
from nca.paced_generation import full_origins, eligibility, block_loss

VERSION = 'train_access_priority_v1'

def teacher_graph(target, interfaces, *, split):
    if split != 'train':
        raise ValueError('Teacher graph is TRAIN-only')
    target = np.asarray(target, dtype=bool)
    interface = np.asarray(interfaces, dtype=bool)
    points = np.argwhere(interface)
    if target.shape != interface.shape or len(points) == 0 or points[:,2].min() == points[:,2].max():
        raise ValueError('Two X-separated interfaces required')
    x = np.arange(target.shape[2])[None,None,:]
    west = interface & (x == points[:,2].min())
    east = interface & (x == points[:,2].max())
    valid = full_origins(target)
    goals = valid & ~full_origins(~east)
    if not goals.any():
        raise ValueError('Teacher has no full-cube destination')
    distance = np.full(valid.shape, -1, dtype=np.int32)
    distance[goals] = 0
    queue = deque(map(tuple, np.argwhere(goals)))
    while queue:
        p = queue.popleft()
        for axis in range(3):
            for sign in (-1, 1):
                q = list(p); q[axis] += sign; q = tuple(q)
                if all(0 <= q[i] < valid.shape[i] for i in range(3)) and valid[q] and distance[q] < 0:
                    distance[q] = distance[p] + 1
                    queue.append(q)
    return distance, west, east

def priority(field, allowed, graph):
    """Choose labels before stochastic firing. No nearest-route repair/fallback mask."""
    distance, west, east = graph
    eligible, seed_phase = eligibility(field, full_origins(allowed))
    empty = np.zeros_like(eligible)
    if (field & west).any() and (field & east).any():
        return empty, 'connected'
    if seed_phase:
        candidates = eligible & (distance >= 0)
        if candidates.any():
            return candidates & (distance == distance[candidates].min()), 'seed_access'
        return empty, 'no_teacher_route'
    current = full_origins(field) & (distance >= 0)
    if not current.any():
        return empty, 'no_teacher_route'
    best = distance[current].min()
    positive = eligible & (distance >= 0) & (distance < best)
    if positive.any():
        return positive, 'advance_access'
    return empty, 'no_teacher_route'

def temporal_loss(logits, m, eligible_fired, target, origins, seed_phase, D, B, C, positive, phase):
    if phase in ('connected', 'no_teacher_route'):
        return block_loss(logits,m,eligible_fired,target,origins,seed_phase,D,B,C)
    if phase not in ('seed_access', 'advance_access'):
        raise ValueError('Unknown supervision phase')
    # Class-balanced BCE: non-advancing teacher cubes are postponed, not deleted.
    zero = logits.sum()*0
    pos = eligible_fired & positive
    neg = eligible_fired & ~positive
    loss = (F.softplus(-logits[pos]).mean() if pos.any() else zero)
    loss = loss + (F.softplus(logits[neg]).mean() if neg.any() else zero)
    # Volume reward would conflict with postponement. Restore unchanged loss after access.
    return loss, loss, zero, zero
