"""geometry_losses_v1: per-scene continuous proxies, not architectural certification.

Explicit scaffold/envelope masks and the historical non-building mass denominator.
No production defaults or trainer are changed. See LOSS_PROTOCOL.md for formulas.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
import numpy as np
from nca.contract import fields_from_state, verify_state_matches_scene
from nca.evaluation import flood_fill

LOSS_VERSION = 'geometry_losses_v1'
FAMILIES = ('legality', 'coverage', 'spill', 'ground', 'thickness', 'sparsity',
            'facade', 'access', 'support')


@dataclass(frozen=True)
class LossSpec:
    min_mass_ratio: float = 0.03
    max_mass_ratio: float = 0.12
    max_facade_ratio: float = 0.15
    thickness_radius: int = 2
    reach_hops: int = 64

    def __post_init__(self):
        for value in (self.min_mass_ratio, self.max_mass_ratio, self.max_facade_ratio):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Ratios must be finite and in [0,1]')
        if self.min_mass_ratio > self.max_mass_ratio:
            raise ValueError('Minimum mass exceeds maximum')
        for value in (self.thickness_radius, self.reach_hops):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('Thickness radius and reach hops must be positive integers')


@dataclass(frozen=True)
class LossContext:
    permitted: torch.Tensor
    protected: torch.Tensor
    budget: torch.Tensor
    facade: torch.Tensor
    support: torch.Tensor
    coverage: torch.Tensor
    envelope: torch.Tensor
    source: torch.Tensor
    endpoints: tuple  # one {ID: boolean D,H,W mask} dictionary per scene
    source_ids: tuple
    route_feasible: torch.Tensor


def _occupancy(p):
    if p.ndim != 4 or any(n == 0 for n in p.shape) or not p.is_floating_point():
        raise ValueError('Occupancy must be nonempty floating [B,D,H,W]')
    if not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Occupancy must be finite and in [0,1]')


def _boolean(mask, shape, device):
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(shape) or mask.device != device:
        raise ValueError('Masks must be boolean with matching shape/device')


def face_max(field):
    """Maximum over self and six face neighbors; outside volume equals zero."""
    padded = F.pad(field, (1, 1, 1, 1, 1, 1), value=0)
    return torch.stack((field, padded[:, :-2, 1:-1, 1:-1], padded[:, 2:, 1:-1, 1:-1],
                        padded[:, 1:-1, :-2, 1:-1], padded[:, 1:-1, 2:, 1:-1],
                        padded[:, 1:-1, 1:-1, :-2], padded[:, 1:-1, 1:-1, 2:]), dim=1).max(1).values


def material_envelope(coverage, permitted, radius):
    """Fixed-radius graph dilation in legal space, never adjusted to meet a budget."""
    if coverage.ndim != 4 or any(n == 0 for n in coverage.shape):
        raise ValueError('Expected nonempty [B,D,H,W] masks')
    _boolean(coverage, coverage.shape, coverage.device)
    _boolean(permitted, coverage.shape, coverage.device)
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError('Radius must be a nonnegative integer')
    if (coverage & ~permitted).any():
        raise ValueError('Coverage must lie inside permitted space')
    result = coverage.clone()
    for _ in range(radius):
        result = (face_max(result.float()) > 0) & permitted
    return result


def soft_reach(occupancy, seeds, hops, fixed_seeds=False):
    """Maximum bottleneck path strength within <=hops six-neighbor edges.

    Material source strength includes its occupancy; fixed support boundaries
    have strength one. Zero occupancy transmits no strength. This is neither a
    probability nor exact unrestricted connectivity. Max/min ties use subgradients.
    """
    _occupancy(occupancy)
    _boolean(seeds, occupancy.shape, occupancy.device)
    if isinstance(hops, bool) or not isinstance(hops, int) or hops < 0:
        raise ValueError('Hops must be a nonnegative integer')
    reached = seeds.to(occupancy.dtype) if fixed_seeds else occupancy * seeds
    for _ in range(hops):
        reached = torch.maximum(reached, torch.minimum(face_max(reached), occupancy))
    return reached


def zero_padded_erosion(p, radius):
    """Chebyshev erosion of continuous occupancy; outside-grid occupancy is zero."""
    _occupancy(p)
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
        raise ValueError('Radius must be a positive integer')
    padded = F.pad(p[:, None], (radius,) * 6, value=0)
    return -F.max_pool3d(-padded, 2 * radius + 1, 1)[:, 0]


def context_from_scenes(seed_state, config, scenes, coverage, envelope, route_feasible):
    """Construct detached context; caller must explicitly supply both target masks."""
    if seed_state.ndim != 5 or len(scenes) != seed_state.shape[0]:
        raise ValueError('Expected a scene for every [B,C,D,H,W] sample')
    shape = (seed_state.shape[0], *seed_state.shape[-3:])
    for mask in (coverage, envelope):
        _boolean(mask, shape, seed_state.device)
    _boolean(route_feasible, (shape[0],), seed_state.device)
    collected = {name: [] for name in ('permitted', 'protected', 'budget', 'facade', 'support', 'source')}
    endpoints, source_ids = [], []
    for i, scene in enumerate(scenes):
        problems = verify_state_matches_scene(seed_state[i:i+1], config, scene)
        if problems:
            raise ValueError('; '.join(problems))
        fields = fields_from_state(seed_state[i:i+1], config, scene)
        masks = {name: torch.from_numpy(value).to(device=seed_state.device)
                 for name, value in fields.items() if isinstance(value, np.ndarray)}
        existing = masks['existing']
        facade = (F.max_pool3d(existing.float()[None, None], 3, 1, 1)[0, 0] > 0) & ~existing
        regions = {name: torch.from_numpy(value).to(device=seed_state.device)
                   for name, value in fields['endpoints'].items()}
        sid = sorted(regions)[0]
        choices = torch.nonzero(regions[sid] & masks['permitted'], as_tuple=False)
        source = torch.zeros_like(existing)
        if len(choices):
            source[tuple(choices[0])] = True  # exactly one fixed source cell
        reached = flood_fill(coverage[i].detach().cpu().numpy() & fields['permitted'],
                             source.detach().cpu().numpy())
        connected = all(bool((reached & region).any()) for region in fields['endpoints'].values())
        if bool(route_feasible[i]) and not connected:
            raise ValueError('Feasible flag disagrees with six-neighbor guide connectivity')
        for name, value in (('permitted', masks['permitted']), ('protected', masks['protected']),
                            ('budget', ~existing), ('facade', facade),
                            ('support', masks['support_boundary']), ('source', source)):
            collected[name].append(value)
        endpoints.append(regions)
        source_ids.append(sid)
    return LossContext(**{name: torch.stack(value) for name, value in collected.items()},
                       coverage=coverage.clone(), envelope=envelope.clone(),
                       endpoints=tuple(endpoints), source_ids=tuple(source_ids),
                       route_feasible=route_feasible.clone())


def _validate_context(p, context):
    for name in ('permitted', 'protected', 'budget', 'facade', 'support', 'coverage', 'envelope', 'source'):
        _boolean(getattr(context, name), p.shape, p.device)
    _boolean(context.route_feasible, (len(p),), p.device)
    if len(context.endpoints) != len(p) or len(context.source_ids) != len(p):
        raise ValueError('Expected per-scene endpoints/source IDs')
    if (context.permitted & ~context.budget).any() or (context.protected & context.permitted).any():
        raise ValueError('Inconsistent permitted/protected/budget regions')
    for i, endpoints in enumerate(context.endpoints):
        if len(endpoints) < 2 or context.source_ids[i] not in endpoints:
            raise ValueError('Need at least two endpoint IDs and a designated source')
        used = torch.zeros_like(p[i], dtype=torch.bool)
        for name, region in endpoints.items():
            _boolean(region, p.shape[1:], p.device)
            if not isinstance(name, str) or not name or not region.any() or (region & used).any():
                raise ValueError('Endpoints need nonempty disjoint regions and string IDs')
            used |= region
        if context.source[i].sum() > 1 or (context.source[i] & ~endpoints[context.source_ids[i]]).any():
            raise ValueError('Use at most one source cell within the designated endpoint')
        if (context.source[i] & ~context.permitted[i]).any():
            raise ValueError('Source cell must be permitted')


def loss_terms(p, context, spec=LossSpec()):
    """Return nine [B] terms plus feasibility/applicability flags, never a hidden mean.

    Invalid contexts retain numeric diagnostic terms, but strict reduction refuses
    them. Empty material is flagged separately; it can be a training start state.
    No aggregate architectural-success score is returned.
    """
    _occupancy(p)
    _validate_context(p, context)
    sums = lambda x: x.flatten(1).sum(1)
    counts = lambda x: sums(x).to(p.dtype)
    mass = sums(p)
    available = counts(context.budget)
    guide_size = counts(context.coverage)
    capacity = counts(context.envelope & context.permitted)
    forbidden = ~context.permitted
    valid_geometry = context.route_feasible & (guide_size > 0) & (available > 0)
    valid_geometry &= ~((context.coverage & ~context.permitted).flatten(1).any(1))
    valid_geometry &= ~((context.coverage & ~context.envelope).flatten(1).any(1))
    valid_geometry &= ~((context.envelope & ~context.permitted).flatten(1).any(1))
    contacts_valid = torch.tensor([all(bool((context.coverage[i] & region).any())
                                      for region in endpoints.values())
                                   for i, endpoints in enumerate(context.endpoints)], device=p.device)
    valid_geometry &= contacts_valid
    sources_valid = (counts(context.source) == 1) & ~((context.source & ~context.coverage).flatten(1).any(1))
    compatible = (capacity >= spec.min_mass_ratio * available) & (guide_size <= spec.max_mass_ratio * available)
    context_valid = valid_geometry & sources_valid & compatible
    terms = {}
    terms['legality'] = sums(p * forbidden) / counts(forbidden).clamp_min(1)
    terms['coverage'] = sums((1 - p) * context.coverage) / guide_size.clamp_min(1)
    terms['spill'] = sums(p * context.permitted * ~context.envelope) / available.clamp_min(1)
    terms['ground'] = sums(p * context.protected) / counts(context.protected).clamp_min(1)
    terms['thickness'] = sums(zero_padded_erosion(p, spec.thickness_radius)) / mass.clamp_min(1)
    ratio = sums(p * context.budget) / available.clamp_min(1)
    terms['sparsity'] = 150 * F.relu(ratio - spec.max_mass_ratio).square() + F.relu(spec.min_mass_ratio - ratio)
    terms['facade'] = F.relu(sums(p * context.facade) / mass.clamp_min(1) - spec.max_facade_ratio)
    reached = soft_reach(p * context.permitted, context.source, spec.reach_hops)
    access = []
    for i, endpoints in enumerate(context.endpoints):
        scores = [reached[i][region].max() for name, region in sorted(endpoints.items()) if name != context.source_ids[i]]
        access.append(1 - torch.stack(scores).mean())
    terms['access'] = torch.stack(access)
    supported = soft_reach(p * context.permitted, context.support, spec.reach_hops, fixed_seeds=True)
    terms['support'] = sums(p * (1 - supported)) / mass.clamp_min(1)
    return {'version': LOSS_VERSION, 'terms': terms, 'context_valid': context_valid,
            'route_feasible': context.route_feasible, 'budget_compatible': compatible,
            'guide_and_envelope_valid': valid_geometry, 'source_valid': sources_valid,
            'nonempty_material': mass > 0, 'mass_ratio': ratio,
            'minimum_mass': spec.min_mass_ratio * available,
            'maximum_mass': spec.max_mass_ratio * available,
            'coverage_voxels': guide_size, 'envelope_capacity': capacity,
            'applicable': {'ground': counts(context.protected) > 0,
                           'legality': counts(forbidden) > 0, 'facade': counts(context.facade) > 0},
            'reach_hops': spec.reach_hops}


def mean_terms(result):
    """Strict reduction: fail rather than hiding an infeasible sample in a batch."""
    if not bool(result['context_valid'].all()):
        invalid = torch.nonzero(~result['context_valid']).flatten().tolist()
        raise ValueError(f'Infeasible/invalid objective context in batch indices {invalid}')
    return {name: value.mean() for name, value in result['terms'].items()}
