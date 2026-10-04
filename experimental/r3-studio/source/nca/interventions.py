"""Explicit L2 research interventions. Historical model/rollout remain unchanged."""
import math
import torch
import torch.nn.functional as F
from deploy.model_utils import LocalLegalityLoss
from nca.losses import LossSpec, loss_terms, _boolean

INTERVENTION_VERSION = 'material_intervention_v1'
BUDGET_VERSION = 'budget_contract_v2'
ARMS = ('hard_projected', 'hard_preclamp', 'smooth_projected')


def smooth_clip(raw, beta=20.0):
    """Softplus difference; exact derivative of this changed forward function.

    It introduces positive background occupancy near raw=0. This is an explicit
    alternative, not a straight-through approximation of the hard clamp.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('beta must be finite and positive')
    return (F.softplus(beta * raw) - F.softplus(beta * (raw - 1))) / beta


def experimental_rollout(model, seed, scaffold, arm, steps, generator, beta=20.0):
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
    for _ in range(steps):
        delta = model.update_net(model.perceive(state))
        mask = (torch.rand(seed.shape[0], 1, *seed.shape[-3:], device=seed.device,
                           generator=generator) < cfg['fire_rate']).float()
        candidate = state[:, cfg['n_frozen']:] + cfg['update_scale'] * delta * mask
        raw = candidate[:, 0]
        material = smooth_clip(raw, beta) if arm == 'smooth_projected' else raw.clamp(0, 1)
        material = material * available * legality
        state = torch.cat((state[:, :cfg['n_frozen']], material[:, None], candidate[:, 1:].clamp(0, 1)), dim=1)
    return {'state': state, 'raw_material': raw, 'version': INTERVENTION_VERSION,
            'arm': arm, 'steps': steps, 'beta': beta}


def guidance_loss(state, raw, guide, config, arm):
    """Per-scene guide coverage; pre-clamp hinge is a new, exact objective."""
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    _boolean(guide, raw.shape, raw.device)
    if not guide.flatten(1).any(1).all():
        raise ValueError('Guide must be nonempty per scene')
    missing = F.relu(1 - raw) if arm == 'hard_preclamp' else 1 - state[:, config['ch_structure']]
    return (missing * guide).flatten(1).sum(1) / guide.flatten(1).sum(1)


def budget_bounds(context, contract, spec=LossSpec()):
    """Necessary bounds only. Same fractions; explicitly different physical budgets.

    Actual mass is always all non-building material, including spill. Only the
    denominator changes, so moving material outside an envelope cannot evade cap.
    """
    if contract not in ('site', 'envelope'):
        raise ValueError('Contract must be site or envelope')
    counts = lambda x: x.flatten(1).sum(1)
    denominator = counts(context.budget if contract == 'site' else context.envelope & context.permitted)
    guide = counts(context.coverage)
    capacity = counts(context.envelope & context.permitted)
    lower = spec.min_mass_ratio * denominator
    upper = spec.max_mass_ratio * denominator
    return {'contract': contract, 'denominator_voxels': denominator, 'minimum_mass': lower,
            'maximum_mass': upper, 'coverage_voxels': guide, 'envelope_capacity': capacity,
            'budget_compatible': (denominator > 0) & (capacity >= lower) & (guide <= upper)}


def objective_terms(state, raw, context, config, arm, contract, spec=LossSpec()):
    """Nine-family local-test objective. Versioned coverage/budget changes only."""
    p = state[:, config['ch_structure']]
    result = loss_terms(p, context, spec)
    bounds = budget_bounds(context, contract, spec)
    denominator = bounds['denominator_voxels'].to(p.dtype).clamp_min(1)
    ratio = (p * context.budget).flatten(1).sum(1) / denominator
    result['terms']['sparsity'] = 150 * F.relu(ratio - spec.max_mass_ratio).square() + F.relu(spec.min_mass_ratio - ratio)
    result['terms']['coverage'] = guidance_loss(state, raw, context.coverage, config, arm)
    result.update(bounds)
    result.update(version=BUDGET_VERSION, material_arm=arm, mass_ratio=ratio)
    result['context_valid'] = (result['guide_and_envelope_valid'] & result['source_valid'] & bounds['budget_compatible'])
    return result
