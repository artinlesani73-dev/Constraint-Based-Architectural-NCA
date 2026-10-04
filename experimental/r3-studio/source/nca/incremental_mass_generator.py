"""MG7 incremental cube accounting; unchanged MG5 routing and geometric choices."""
from dataclasses import asdict, dataclass
import heapq
import math
import time

import numpy as np

from nca.contract import entrance_masks, validate_scene
from nca.volumetric import boolean_grid

VERSION = 'incremental_coverage_growth_v1'


@dataclass(frozen=True)
class CoverageGeneratorSpec:
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



def box_counts(field, width):
    """Exact integer sums for every width-cube origin using a summed volume."""
    s=np.pad(np.asarray(field,dtype=np.int64),((1,0),)*3).cumsum(0).cumsum(1).cumsum(2)
    w=width
    return (s[w:,w:,w:]-s[:-w,w:,w:]-s[w:,:-w,w:]-s[w:,w:,:-w]
            +s[:-w,:-w,w:]+s[:-w,w:,:-w]+s[w:,:-w,:-w]-s[:-w,:-w,:-w])


def coverage_parts(domain, fraction):
    """MT1 fixed-X voxel-center thirds and exact integer acceptance minima."""
    x=np.flatnonzero(domain.any(axis=(0,1)))
    if not len(x):raise ValueError('Nonempty domain required')
    parts=[]
    for i in range(3):
        lo=x[0]+(x[-1]+1-x[0])*i/3;hi=x[0]+(x[-1]+1-x[0])*(i+1)/3
        parts.append(domain & ((np.arange(domain.shape[2])+.5>=lo)&(np.arange(domain.shape[2])+.5<hi))[None,None,:])
    counts=np.array([int(p.sum()) for p in parts],dtype=np.int64)
    return parts,np.ceil(counts*(fraction-1e-10)).astype(np.int64)


class IncrementalCubeCounts:
    """Exact cube counts, updated only where distinct newly occupied cells overlap.

    remove_cells accepts cells that were empty in the previous field, once each.
    The growth caller enforces this by reading its cube before setting occupancy.
    """
    def __init__(self, field, contact, width, parts):
        self.width = width
        self.shape = tuple(n-width+1 for n in field.shape)
        self.contact = contact
        self.labels = np.full(field.shape, -1, dtype=np.int8)
        for i, part in enumerate(parts):self.labels[part] = i
        self.by_third = np.stack([box_counts((~field)&part,width).ravel() for part in parts])
        self.added = self.by_third.sum(axis=0)
        self.new_contact = box_counts((~field)&contact,width).ravel()
        self.third_grid = self.by_third.reshape((3,)+self.shape)
        self.added_grid = self.added.reshape(self.shape)
        self.contact_grid = self.new_contact.reshape(self.shape)

    def remove_cells(self, cells):
        for cell in cells:
            point = tuple(int(v) for v in cell)
            region = tuple(slice(max(0,v-self.width+1),min(v+1,n)) for v,n in zip(point,self.shape))
            label = int(self.labels[point])
            if label >= 0:
                self.third_grid[(label,)+region] -= 1
                self.added_grid[region] -= 1
            if self.contact[point]:self.contact_grid[region] -= 1


def grow_coverage(field, contact, width, target, valid, route, radial, limit, expired, parts, minima):
    """Finite connected growth ranked by capped deficit reduction per new cell.

    Every current frontier origin is re-evaluated under the current mass budget.
    Accepted origins expand once, including zero-delta transit. Integer cube
    deltas are decremented locally after positive growth and remain valid during transit.
    """
    from hashlib import sha256
    mass=int(field.sum());contacts=int((field&contact).sum())
    counts=np.array([int((field&p).sum()) for p in parts],dtype=np.int64)
    minima=np.asarray(minima,dtype=np.int64)
    queued=np.zeros_like(valid);frontier=np.zeros_like(valid)
    trace=[];accepted=[];total_proposals=0;total_rejected=0;cache=None
    report={'trace':trace,'accepted_origins_zyx':accepted,'initial_voxels':mass,
            'initial_contact_voxels':contacts,'initial_third_cells':counts.tolist(),
            'minimum_third_cells':minima.tolist(),
            'priority':'maximum sum(min(new_third_cells,remaining_deficit))/new_cells; zero for no new cells; then MG3 radial score; then flat ZYX origin'}
    def finish(status):
        report.update(final_voxels=mass,final_contact_voxels=contacts,final_third_cells=counts.tolist(),
            coverage_minima_met=bool(np.all(counts>=minima) and all(p.any() for p in parts)),
            frontier_remaining=int(frontier.sum()),evaluated_proposals=total_proposals,budget_rejections=total_rejected)
        return status,report
    if not mass or contacts/mass>limit+1e-10:return finish('route_contact_budget_exceeded')
    for p in route:queued[tuple(p)]=True
    def expand(p):
        for axis in range(3):
            for delta in (-1,1):
                q=list(p);q[axis]+=delta;q=tuple(q)
                if 0<=q[axis]<valid.shape[axis] and valid[q] and not queued[q]:
                    queued[q]=True;frontier[q]=True
    for p in route:expand(tuple(p))
    while mass<target and frontier.any():
        if expired():return finish('time_limit')
        if cache is None:
            cache=IncrementalCubeCounts(field,contact,width,parts)
        by_third,added,new_contact=cache.by_third,cache.added,cache.new_contact
        ids=np.flatnonzero(frontier)
        ok=(contacts+new_contact[ids])/(mass+added[ids])<=limit+1e-10
        eligible=ids[ok];deficit=np.maximum(minima-counts,0)
        total_proposals+=len(ids);total_rejected+=int((~ok).sum())
        decision={'mass_before':mass,'contact_before':contacts,'third_cells_before':counts.tolist(),
            'deficits_before':deficit.tolist(),'frontier_count':len(ids),'admissible_count':len(eligible),
            'budget_rejected':int((~ok).sum()),'frontier_sha256':sha256(ids.astype('<i4').tobytes()).hexdigest()}
        if not len(eligible):
            trace.append({**decision,'origin_zyx':None});return finish('contact_budget_stalled')
        gain=np.minimum(by_third[:,eligible],deficit[:,None]).sum(axis=0)
        ratio=gain/np.maximum(added[eligible],1)
        order=np.lexsort((eligible,radial.ravel()[eligible],-ratio));j=int(order[0]);idx=int(eligible[j])
        p=tuple(int(v) for v in np.unravel_index(idx,valid.shape));z,y,x=p
        delta=int(added[idx]);extra_contact=int(new_contact[idx]);extra_thirds=by_third[:,idx].copy()
        trace.append({**decision,'origin_zyx':list(p),'added_cells':delta,'added_contact_cells':extra_contact,
            'added_third_cells':extra_thirds.tolist(),'deficit_reduction_cells':int(gain[j]),'gain_per_new_cell':float(ratio[j]),
            'radial_score':float(radial[p])})
        if delta:
            fresh=np.argwhere(~field[z:z+width,y:y+width,x:x+width])+np.array(p)
            cache.remove_cells(fresh)
        frontier[p]=False;field[z:z+width,y:y+width,x:x+width]=True
        accepted.append(list(p));mass+=delta;contacts+=extra_contact;counts+=extra_thirds;expand(p)
    return finish('target_reached' if mass>=target else 'component_exhausted')


def generate_incremental_mass(scene, fields, domain, seed, spec=CoverageGeneratorSpec()):
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
    parts,minima=coverage_parts(domain,context.spec.min_third_fraction)
    status, growth = grow_coverage(field, contact, width, target, valid, route,
                                  radial, limit, expired, parts, minima)
    selected.extend(growth['accepted_origins_zyx'])
    report['growth'] = growth
    return finish(status)
