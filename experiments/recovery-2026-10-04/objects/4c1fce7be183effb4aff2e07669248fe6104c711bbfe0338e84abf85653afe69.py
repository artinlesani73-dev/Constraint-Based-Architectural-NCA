"""Binary massing_targets_v1 pilot contract; not a differentiable training loss."""
from dataclasses import dataclass, asdict
import math
import numpy as np
from nca.volumetric import boolean_grid
from nca.evaluation import flood_fill, geometric_support
from nca.contract import validate_scene, entrance_masks

VERSION = 'massing_targets_v1'
FAMILIES = ('access', 'coverage', 'facade', 'ground', 'legality', 'sparsity', 'spill', 'support', 'thickness')


@dataclass(frozen=True)
class MassingTargetSpec:
    min_volume_fraction: float = .08
    max_volume_fraction: float = .40
    min_cube_m: float = 2.4
    min_bulk_fraction: float = .90
    min_third_fraction: float = .08
    max_facade_fraction: float = .15

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Target parameters must be finite numbers: ' + name)
            if value <= 0 or (name != 'min_cube_m' and value > 1):
                raise ValueError('Positive physical scale and fractions in (0,1] required')
        if self.min_volume_fraction > self.max_volume_fraction:
            raise ValueError('Minimum volume exceeds maximum')


def cube_supported(field, width):
    """Union of ALL fully occupied axis-aligned width^3 cubes (binary opening).

    Supports even widths; outside-grid occupancy is empty. Unlike eroded-center
    fraction, a filled box >=width in all axes retains its boundary voxels.
    """
    field = boolean_grid(field)
    if type(width) is not int or width < 1:
        raise ValueError('Cube width must be a positive integer')
    result = np.zeros_like(field)
    if width > min(field.shape):
        return result
    valid = np.lib.stride_tricks.sliding_window_view(field, (width,) * 3).all(axis=(-3, -2, -1))
    for z in range(width):
        for y in range(width):
            for x in range(width):
                result[z:z+valid.shape[0], y:y+valid.shape[1], x:x+valid.shape[2]] |= valid
    return result


def neighbors(field, diagonal=False):
    p = np.pad(field, 1, constant_values=False); result = np.zeros_like(field)
    for z in range(3):
        for y in range(3):
            for x in range(3):
                if diagonal or abs(z-1)+abs(y-1)+abs(x-1) == 1:
                    result |= p[z:z+field.shape[0], y:y+field.shape[1], x:x+field.shape[2]]
    return result


def reach_from_interface(field, endpoints):
    source = field & endpoints[sorted(endpoints)[0]]
    seed = np.zeros_like(field)
    if source.any(): seed[tuple(np.argwhere(source)[0])] = True
    reached = flood_fill(field, seed)
    hits = {name: bool((reached & region).any()) for name, region in endpoints.items()}
    return reached, hits


def context_interfaces_connectable(field, endpoints):
    """Some available component must touch every interface, not just the first.

    Candidate access still requires all its occupied components to connect. A
    context may contain unused disconnected space, so test all source components.
    """
    remaining = field & endpoints[sorted(endpoints)[0]]
    while remaining.any():
        seed = np.zeros_like(field); seed[tuple(np.argwhere(remaining)[0])] = True
        reached = flood_fill(field, seed)
        if all((reached & mask).any() for mask in endpoints.values()):
            return True
        remaining &= ~reached
    return False


def evaluate_targets(field, scene, fields, domain, spec=MassingTargetSpec()):
    """Nine separate binary checks and explicit conjunction, no weighted score.

    Context flags are necessary feasibility checks, not a promise of a feasible
    massing design. Geometry is evaluated without hiding source violations.
    """
    scene = validate_scene(scene); field = boolean_grid(field); domain = boolean_grid(domain)
    size = scene['voxel_size_m']; shape = (scene['grid_size'],) * 3
    if field.shape != shape or domain.shape != shape or not domain.any():
        raise ValueError('Candidate and nonempty domain must match scene')
    masks = {k: boolean_grid(fields[k]) for k in ('permitted', 'existing', 'protected', 'support_boundary')}
    if any(m.shape != shape for m in masks.values()): raise ValueError('Scene masks must match')
    legal, existing, protected, support = (masks[k] for k in ('permitted','existing','protected','support_boundary'))
    if (domain & (~legal | existing)).any() or (legal & (existing | protected)).any():
        raise ValueError('Domain and scene masks contradict legality')
    endpoints = entrance_masks(scene)
    width = max(1, math.ceil(spec.min_cube_m / size - 1e-10))
    bulk = cube_supported(field & legal & domain, width)
    raw_reached, raw_hits = reach_from_interface(field & legal & domain, endpoints)
    bulk_reached, bulk_hits = reach_from_interface(bulk, endpoints)
    domain_bulk = cube_supported(domain, width)
    domain_connectable = context_interfaces_connectable(domain_bulk, endpoints)
    # Fixed thirds along the declared site's X span, not along candidate extent.
    x = np.flatnonzero(domain.any(axis=(0, 1)))
    thirds = []
    for i in range(3):
        lo = x[0] + (x[-1] + 1 - x[0]) * i / 3
        hi = x[0] + (x[-1] + 1 - x[0]) * (i+1) / 3
        part = domain & ((np.arange(shape[2]) + .5 >= lo) & (np.arange(shape[2]) + .5 < hi))[None, None, :]
        count = int(part.sum()); occupied = int((bulk & part).sum())
        thirds.append({'domain_cells': count, 'bulk_cells': occupied, 'fraction': occupied/count if count else None})
    facade = neighbors(existing, diagonal=True) & ~existing
    face_shell = neighbors(existing) & ~existing
    allowance = np.zeros_like(field)
    for entrance in scene['entrances']:
        if entrance['kind'] == 'facade': allowance |= endpoints[entrance['id']] & face_shell & legal
    mass = int(field.sum()); denom = int(domain.sum()); ratio = mass/denom
    facade_ratio = int((field & facade & ~allowance).sum())/mass if mass else 0.0
    support_report = geometric_support(field, support)
    raw_unreached = int((field & ~raw_reached).sum()); bulk_unreached = int((bulk & ~bulk_reached).sum())
    bulk_fraction = int(bulk.sum())/mass if mass else 0.0
    predicates = {
        'access': mass > 0 and all(raw_hits.values()) and raw_unreached == 0 and all(bulk_hits.values()) and bulk_unreached == 0,
        'coverage': all(t['fraction'] is not None and t['fraction'] + 1e-10 >= spec.min_third_fraction for t in thirds),
        'facade': facade_ratio <= spec.max_facade_fraction + 1e-10,
        'ground': not bool((field & protected).any()),
        'legality': not bool((field & ~legal).any()),
        'sparsity': spec.min_volume_fraction - 1e-10 <= ratio <= spec.max_volume_fraction + 1e-10,
        'spill': not bool((field & ~domain).any()),
        'support': mass > 0 and support_report['unsupported_voxels'] == 0,
        'thickness': mass > 0 and bulk_fraction + 1e-10 >= spec.min_bulk_fraction,
    }
    possible = domain_connectable and all(t['domain_cells'] > 0 for t in thirds)
    report = {'version': VERSION, 'spec': asdict(spec), 'voxel_size_m': size,
        'resolved_cube_width_cells': width, 'resolved_cube_width_m': width*size,
        'occupied_voxels': mass, 'gross_volume_m3': mass*size**3, 'domain_voxels': denom,
        'volume_fraction': ratio, 'bulk_voxels': int(bulk.sum()), 'bulk_fraction': bulk_fraction,
        'thirds': thirds, 'raw_interface_hits': raw_hits, 'bulk_interface_hits': bulk_hits,
        'unreached_occupied_voxels': raw_unreached, 'unreached_bulk_voxels': bulk_unreached,
        'facade_contact_fraction': facade_ratio, 'facade_excess': max(0.0, facade_ratio-spec.max_facade_fraction),
        'illegal_voxels': int((field & ~legal).sum()), 'protected_blocked_voxels': int((field & protected).sum()),
        'outside_domain_voxels': int((field & ~domain).sum()), 'support': support_report,
        'context_bulk_interfaces_connectable': domain_connectable,
        'context_necessary_checks_pass': possible, 'family_pass': predicates,
        'contract_pass': possible and all(predicates.values()),
        'interpretation': 'Pilot binary massing contract, not trained loss, architectural quality, interior circulation or structural certification.'}
    return report, {'bulk': bulk, 'raw_reached': raw_reached, 'bulk_reached': bulk_reached, 'domain_bulk': domain_bulk}
