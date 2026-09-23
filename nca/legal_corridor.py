"""Deterministic CPU procedural scaffold on the existing permitted voxel graph.

Separate from the isolated bounded-envelope fix. Six face neighbors, explicit
endpoint IDs, shortest legal paths, no endpoint-height clip or iteration cap.
No learned parameters, differentiation, walkability or mechanical safety claim.
"""
from collections import deque
from itertools import combinations
import numpy as np
import torch
import torch.nn.functional as F
from nca.corridor import bounded_vertical_envelope, _radius
from nca.contract import fields_from_state, verify_state_matches_scene
from nca.evaluation import _mask, flood_fill, offsets

LEGAL_VERSION = 'corridor_legal_v1'
NEIGHBORS = tuple(offsets(6))


def _search(allowed, seeds):
    """Multi-source BFS: exact unit-edge distances and deterministic predecessors."""
    distance = np.full(allowed.shape, -1, np.int32)
    parent = np.full(allowed.size, -1, np.int64)
    roots = list(map(tuple, np.argwhere(seeds)))
    queue = deque(roots)
    distance[seeds] = 0
    depth, height, width = allowed.shape
    for_root = lambda p: (p[0] * height + p[1]) * width + p[2]
    while queue:
        z, y, x = queue.popleft()
        for dz, dy, dx in NEIGHBORS:
            nz, ny, nx = z + dz, y + dy, x + dx
            if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
                q = (nz, ny, nx)
                if allowed[q] and distance[q] < 0:
                    distance[q] = distance[z, y, x] + 1
                    parent[for_root(q)] = for_root((z, y, x))
                    queue.append(q)
    return distance, parent


def route_legal_corridor(permitted, endpoints, corridor_width=1, vertical_envelope=1):
    """Return Boolean target/centerline and a JSON-compatible feasibility record.

    Find shortest paths between endpoint regions; a Kruskal minimum spanning
    forest chooses edges by (legal distance, endpoint IDs). Include each legal
    endpoint region so incident paths meet. This minimizes selected edge lengths,
    not the volume of the union or a structural objective. Infeasible scenes
    retain their partial forest with an explicit status, never a success label.
    """
    _radius(corridor_width)
    _radius(vertical_envelope)
    allowed = _mask(permitted)
    if len(endpoints) < 2 or any(not isinstance(k, str) or not k for k in endpoints):
        raise ValueError('Need at least two explicit nonempty string endpoint IDs')
    occupied = np.zeros_like(allowed)
    regions = {}
    for name in sorted(endpoints):
        region = _mask(endpoints[name])
        if region.shape != allowed.shape or not region.any() or (region & occupied).any():
            raise ValueError('Endpoint shapes must match and regions must be nonempty and disjoint')
        occupied |= region
        legal = region & allowed
        if legal.any():
            first = np.zeros_like(allowed)
            first[tuple(np.argwhere(legal)[0])] = True
            if not np.array_equal(flood_fill(legal, first), legal):
                raise ValueError(f'Endpoint {name} has disconnected legal pieces')
        regions[name] = legal
    ids = sorted(regions)
    searches = {name: _search(allowed, region) for name, region in regions.items() if region.any()}
    candidates = []
    for a, b in combinations(ids, 2):
        if a not in searches or b not in searches:
            continue
        distance, parent = searches[a]
        cells = np.flatnonzero(regions[b] & (distance >= 0))
        if not len(cells):
            continue
        end = int(cells[np.argmin(distance.ravel()[cells])])
        candidates.append((int(distance.ravel()[end]), a, b, end))
    components = {name: name for name in ids}
    def root(name):
        while components[name] != name:
            name = components[name]
        return name
    centerline = np.zeros_like(allowed)
    for region in regions.values():
        centerline |= region
    edges = []
    for length, a, b, end in sorted(candidates):
        ra, rb = root(a), root(b)
        if ra == rb:
            continue
        components[rb] = ra
        parent = searches[a][1]
        path = []
        cell = end
        while cell >= 0:
            centerline.ravel()[cell] = True
            path.append(list(map(int, np.unravel_index(cell, allowed.shape))))
            cell = int(parent[cell])
        edges.append({'source': a, 'target': b, 'length_edges': length,
                      'path_zyx': list(reversed(path))})
    field = torch.from_numpy(centerline.astype(np.float32))
    # Isotropic width dilation first; bounded depth dilation second; legal clip
    # last. Keep only thickened cells connected to a routed seed component.
    field = F.max_pool3d(field[None, None], 2 * corridor_width + 1,
                         1, corridor_width)[0, 0]
    field = bounded_vertical_envelope(field, vertical_envelope)
    clipped = (field.numpy() > 0.5) & allowed
    target = flood_fill(clipped, centerline)
    blocked = [name for name in ids if not regions[name].any()]
    feasible = not blocked and len({root(name) for name in ids}) == 1
    report = {'operator': LEGAL_VERSION, 'status': 'connected' if feasible else 'infeasible',
              'all_endpoints_connected': feasible, 'endpoint_ids': ids,
              'blocked_endpoints': blocked, 'neighborhood': 6,
              'corridor_width': corridor_width, 'vertical_envelope': vertical_envelope,
              'height_clip': None, 'edges': edges,
              'centerline_voxels': int(centerline.sum()), 'target_voxels': int(target.sum()),
              'discarded_isolated_dilation_voxels': int((clipped & ~target).sum()),
              'note': 'A legal spatial scaffold; not walking clearance or structural safety.'}
    return {'target': target, 'centerline': centerline, 'report': report}


def compute_legal_corridor_v1(seed_state, config, scenes, corridor_width=1, vertical_envelope=1):
    """Batch adapter; one declared scene per sample, copied CPU routing per scene.

    Targets return on the input device and floating dtype, detached. Config's
    legacy corridor_z_margin is intentionally unused: endpoints at ground level
    must be able to reach the permitted region above the street band.
    """
    if seed_state.ndim != 5 or not seed_state.is_floating_point() or not torch.isfinite(seed_state).all():
        raise ValueError('Expected finite floating (B,C,D,H,W) seed state')
    if len(scenes) != seed_state.shape[0] or not scenes:
        raise ValueError('Pass one scene per batch sample')
    targets, centerlines, reports = [], [], []
    for index, scene in enumerate(scenes):
        state = seed_state[index:index + 1]
        problems = verify_state_matches_scene(state, config, scene)
        if problems:
            raise ValueError('; '.join(problems))
        fields = fields_from_state(state, config, scene)
        result = route_legal_corridor(fields['permitted'], fields['endpoints'],
                                      corridor_width, vertical_envelope)
        targets.append(torch.from_numpy(result['target']).to(seed_state))
        centerlines.append(torch.from_numpy(result['centerline']).to(seed_state))
        reports.append(result['report'])
    return {'target': torch.stack(targets), 'centerline': torch.stack(centerlines), 'reports': reports}
