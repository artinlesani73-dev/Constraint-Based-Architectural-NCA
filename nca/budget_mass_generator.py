"""MG3 budget-checked cube growth. Frozen MG2 route/costs, existing facade limit."""
from dataclasses import asdict, dataclass
import heapq
import math
import time

import numpy as np

from nca.contract import entrance_masks, validate_scene
from nca.volumetric import boolean_grid

VERSION = 'budgeted_contact_growth_v1'


@dataclass(frozen=True)
class BudgetGeneratorSpec:
    cube_m: float = 2.4
    target_fraction: float = .24
    max_seconds: float = 15.0
    contact_weight: float = 12.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or (value < 0 if name == 'contact_weight' else value <= 0):
                raise ValueError('Positive finite parameter required: ' + name)
        if self.target_fraction > 1:
            raise ValueError('Target fraction cannot exceed one')


def cube_windows(field, width):
    return np.lib.stride_tricks.sliding_window_view(field, (width,) * 3)



def cube_delta(field, contact, origin, width):
    """Unique new cells; overlapping contact cells are never counted twice."""
    z, y, x = origin
    region = (slice(z,z+width), slice(y,y+width), slice(x,x+width))
    new = ~field[region]
    return int(new.sum()), int((new & contact[region]).sum())


def grow_budgeted(field, contact, width, target, valid, route, radial, limit, expired):
    """Mutate a completed route via connected legal cubes under its global budget.

    Reconsider deferred origins only after positive geometric growth. Accepted
    zero-delta origins expand once and are never requeued; a finite heap pass
    with no admissible origin ends explicitly. The caller validates the route.
    """
    mass = int(field.sum()); contacts = int((field & contact).sum())
    trace = []; accepted = []; queued = set(map(tuple, route)); frontier = []; deferred = []
    report = {'accepted_origins_zyx': accepted, 'trace': trace,
              'initial_voxels': mass, 'initial_contact_voxels': contacts}
    def finish(status):
        report.update(final_voxels=mass, final_contact_voxels=contacts,
                      deferred_origins_zyx=[list(p) for _,p in sorted(deferred)],
                      frontier_remaining=len(frontier), evaluated_proposals=len(trace))
        return status, report
    if not mass or contacts / mass > limit + 1e-10:
        return finish('route_contact_budget_exceeded')
    def expand(p):
        for axis in range(3):
            for delta in (-1,1):
                q=list(p);q[axis]+=delta;q=tuple(q)
                if 0<=q[axis]<valid.shape[axis] and valid[q] and q not in queued:
                    queued.add(q);heapq.heappush(frontier,(float(radial[q]),q))
    for p in route:expand(tuple(p))
    while mass < target and frontier:
        if expired():return finish('time_limit')
        priority,p=heapq.heappop(frontier)
        added,new_contact=cube_delta(field,contact,p,width)
        ok=(contacts+new_contact)/(mass+added)<=limit+1e-10
        trace.append({'origin_zyx':list(map(int,p)), 'mass_before':mass,
                      'contact_before':contacts,'added_cells':added,
                      'added_contact_cells':new_contact,'accepted':ok})
        if not ok:
            deferred.append((priority,p));continue
        z,y,x=p;field[z:z+width,y:y+width,x:x+width]=True
        accepted.append(list(map(int,p)));mass+=added;contacts+=new_contact
        expand(p)
        if added and deferred:
            for entry in deferred:heapq.heappush(frontier,entry)
            deferred.clear()
    if mass>=target:return finish('target_reached')
    return finish('contact_budget_stalled' if deferred else 'component_exhausted')


def generate_budget_mass(scene, fields, domain, seed, spec=BudgetGeneratorSpec()):
    """Route fully legal cube origins, then grow a connected union toward a budget.

    Two interfaces only. Dijkstra searches all source origins at once but returns
    ONE path to ONE target; disconnected source components are never merged.
    A route requires a cube containing a supported interface cell at
    each end. Failure is a generator limitation, not a proof of site infeasibility.
    The full final field is returned without clipping or postprocessing.
    """
    scene = validate_scene(scene)
    domain = boolean_grid(domain)
    shape = (scene['grid_size'],) * 3
    if domain.shape != shape or not domain.any():
        raise ValueError('Nonempty scene-sized domain required')
    if type(seed) is not int or seed < 0:
        raise ValueError('Nonnegative integer seed required')
    legal, existing, support = (boolean_grid(fields[k]) for k in ('permitted', 'existing', 'support_boundary'))
    if any(a.shape != shape for a in (legal, existing, support)):
        raise ValueError('Masks must match scene')
    if (domain & (~legal | existing)).any():
        raise ValueError('Domain must be legal and exclude context')
    width = max(1, math.ceil(spec.cube_m / scene['voxel_size_m'] - 1e-10))
    field = np.zeros_like(domain)
    route_field = field.copy()
    selected = []
    started = time.perf_counter()
    target = math.ceil(int(domain.sum()) * spec.target_fraction)
    report = {'version': VERSION, 'seed': seed, 'spec': asdict(spec),
              'cube_width_cells': width, 'requested_voxels': target,
              'selected_origins_zyx': selected, 'route_origins_zyx': [],
              'supported_interfaces': len(scene['entrances']) == 2,
              'learned': False, 'postprocessed': False}

    def finish(status):
        count = int(field.sum())
        report.update(status=status, occupied_voxels=count,
                      target_error_voxels=count-target,
                      target_reached=count >= target,
                      wall_seconds=time.perf_counter()-started)
        return field, route_field, report

    def expired():
        return time.perf_counter()-started >= spec.max_seconds

    if len(scene['entrances']) != 2:
        return finish('unsupported_interface_count')
    if width > min(shape):
        return finish('no_legal_cube')
    valid = cube_windows(domain, width).all(axis=(-3, -2, -1))
    coords = np.argwhere(valid)
    if not len(coords):
        return finish('no_legal_cube')
    endpoints = entrance_masks(scene)
    names = sorted(endpoints)
    supported = support.copy()
    for axis in range(3):
        lo = [slice(None)] * 3; hi = lo.copy()
        lo[axis] = slice(None, -1); hi[axis] = slice(1, None)
        supported[tuple(lo)] |= support[tuple(hi)]
        supported[tuple(hi)] |= support[tuple(lo)]
    hits = [cube_windows(endpoints[name] & supported, width).any(axis=(-3, -2, -1)) & valid for name in names]
    if not all(a.any() for a in hits):
        return finish('no_supported_interface_cube')
    rng = np.random.default_rng(seed)
    # Positive stochastic costs diversify shortest legal routes; no MT1 scores.
    # Penalize existing nonexempt facade contact; never remove legal origins.
    from nca.massing_objective import make_context
    context = make_context(scene, fields, domain)
    contact = context.contact
    limit = context.spec.max_facade_fraction
    contact_fraction = cube_windows(contact, width).mean(axis=(-3, -2, -1))
    costs = 1.0 + rng.uniform(0, .75, size=valid.shape) + spec.contact_weight * contact_fraction
    report['contact_policy'] = 'contact_weight times nonexempt contact fraction per cube in route and growth costs; legal domain unchanged'
    report['contact_mask_voxels'] = int((contact & domain).sum())
    distances = np.full(valid.shape, np.inf)
    previous = {}
    heap = []
    for p in map(tuple, np.argwhere(hits[0])):
        distances[p] = costs[p]
        heapq.heappush(heap, (float(costs[p]), p))

    def adjacent(p):
        for axis in range(3):
            for delta in (-1, 1):
                q = list(p); q[axis] += delta; q = tuple(q)
                if 0 <= q[axis] < valid.shape[axis] and valid[q]:
                    yield q

    reached = None
    while heap:
        if expired():
            return finish('time_limit')
        dist, p = heapq.heappop(heap)
        if dist != distances[p]:
            continue
        if hits[1][p]:
            reached = p
            break
        for q in adjacent(p):
            proposed = dist + float(costs[q])
            if proposed < distances[q]:
                distances[q] = proposed
                previous[q] = p
                heapq.heappush(heap, (proposed, q))
    if reached is None:
        return finish('no_cube_route')
    route = [reached]
    while route[-1] in previous:
        route.append(previous[route[-1]])
    route.reverse()
    report['route_origins_zyx'] = [list(map(int, p)) for p in route]

    def add(p):
        z, y, x = p
        field[z:z+width, y:y+width, x:x+width] = True
        selected.append(list(map(int, p)))

    for p in route:
        add(p)
    route_field = field.copy()
    report['max_facade_fraction'] = limit
    if int((field & contact).sum()) / int(field.sum()) > limit + 1e-10:
        return finish('route_contact_budget_exceeded')
    if int(field.sum()) >= target:
        return finish('route_at_or_above_request')
    # Grow radially around the route. Anisotropy varies by seed, with all axes
    # positive; it is a style parameter, not another constraint family.
    scales = rng.uniform(.65, 1.5, size=3)
    report['growth_axis_weights_zyx'] = scales.tolist()
    radial = np.full(valid.shape, np.inf)
    best = np.full(len(coords), np.inf)
    for p in route:
        best = np.minimum(best, (((coords - p) * scales) ** 2).sum(axis=1))
    radial[tuple(coords.T)] = best + rng.uniform(0, 1.5, len(coords)) + spec.contact_weight * contact_fraction[tuple(coords.T)]
    status, growth = grow_budgeted(field, contact, width, target, valid, route,
                                    radial, limit, expired)
    selected.extend(growth['accepted_origins_zyx'])
    report['growth'] = growth
    return finish(status)
