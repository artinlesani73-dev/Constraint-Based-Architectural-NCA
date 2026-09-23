"""Diagnostic-only exact forward trace; no training default uses this module."""
import torch
from deploy.model_utils import LocalLegalityLoss
from nca.interventions import ARMS, INTERVENTION_VERSION, smooth_clip

def traced_rollout(model, seed, scaffold, arm, steps, generator, beta=20.0):
    """Historical-training forward at epoch60, with one declared material change.

    Fire rate/update scale come from checkpoint config; seed scale is 0.15.
    Hidden channels retain the original clamp. Material legality remains hard.
    Return last raw material candidate for an exact pre-clamp guidance objective.
    """
    if arm not in ARMS or isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('Unknown arm or invalid steps')
    if generator is None:
        raise ValueError('An explicit firing generator is required')
    cfg = model.config
    shape = (seed.shape[0], *seed.shape[-3:])
    if seed.ndim != 5 or tuple(scaffold.shape) != shape or not torch.isfinite(seed).all():
        raise ValueError('Seed/scaffold shapes or values invalid')
    if not torch.isfinite(scaffold).all() or ((scaffold < 0) | (scaffold > 1)).any():
        raise ValueError('Scaffold must be in [0,1]')
    if cfg['ch_structure'] != cfg['n_frozen']:
        raise ValueError('Expected first grown channel to represent material')
    state = seed.clone()
    state[:, cfg['ch_structure']] = (state[:, cfg['ch_structure']] + .15 * scaffold).clamp(0, 1)
    legality = LocalLegalityLoss(cfg).compute_legality_field(seed)
    available = 1 - seed[:, cfg['ch_existing']]
    trajectory = []
    for _ in range(steps):
        delta = model.update_net(model.perceive(state))
        mask = (torch.rand(seed.shape[0], 1, *seed.shape[-3:], device=seed.device,
                           generator=generator) < cfg['fire_rate']).float()
        candidate = state[:, cfg['n_frozen']:] + cfg['update_scale'] * delta * mask
        raw = candidate[:, 0]
        trajectory.append({"raw": raw, "mask": mask, "delta": delta})
        material = smooth_clip(raw, beta) if arm == 'smooth_projected' else raw.clamp(0, 1)
        material = material * available * legality
        state = torch.cat((state[:, :cfg['n_frozen']], material[:, None], candidate[:, 1:].clamp(0, 1)), dim=1)
    return {'state': state, 'raw_material': raw, 'version': INTERVENTION_VERSION,
            'arm': arm, 'steps': steps, 'beta': beta, 'trajectory': trajectory}

