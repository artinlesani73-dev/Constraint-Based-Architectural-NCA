"""MG1 procedural cube-union massing. No learned weights or evaluator feedback."""
from dataclasses import asdict, dataclass
import heapq
import math
import time

import numpy as np

from nca.contract import entrance_masks, validate_scene
from nca.volumetric import boolean_grid

VERSION = 'cube_route_growth_v1'


@dataclass(frozen=True)
class MassGeneratorSpec:
    cube_m: float = 2.4
    target_fraction: float = .24
    max_seconds: float = 15.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Positive finite parameter required: ' + name)
        if self.target_fraction > 1:
            raise ValueError('Target fraction cannot exceed one')


def cube_windows(field, width):
    return np.lib.stride_tricks.sliding_window_view(field, (width,) * 3)


def generate_mass(scene, fields, domain, seed, spec=MassGeneratorSpec()):
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
    costs = 1.0 + rng.uniform(0, .75, size=valid.shape)
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
    radial[tuple(coords.T)] = best + rng.uniform(0, 1.5, len(coords))
    queued = set(route)
    frontier = []

    def expand(p):
        for q in adjacent(p):
            if q not in queued:
                queued.add(q)
                heapq.heappush(frontier, (float(radial[q]), q))

    for p in route:
        expand(p)
    while frontier and int(field.sum()) < target:
        if expired():
            return finish('time_limit')
        _, p = heapq.heappop(frontier)
        add(p)
        expand(p)
    return finish('target_reached' if int(field.sum()) >= target else 'component_exhausted')


def pairwise_diversity(fields):
    """Voxel Jaccard distance for caller-selected VALID alternatives only."""
    arrays = [boolean_grid(a) for a in fields]
    if arrays and any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('Diversity fields must share a domain')
    distances = []
    for i, a in enumerate(arrays):
        for b in arrays[i+1:]:
            union = int((a | b).sum())
            distances.append(1 - int((a & b).sum())/union if union else 0.0)
    unique = len({a.tobytes() for a in arrays})
    return {'valid_alternatives': len(arrays), 'unique_valid_fields': unique,
            'duplicate_fraction': 1-unique/len(arrays) if arrays else None,
            'pair_count': len(distances), 'jaccard_distances': distances,
            'mean_jaccard_distance': float(np.mean(distances)) if distances else None,
            'interpretation': 'Voxel difference among valid outputs; not design quality.'}
