"""NL0: frozen volume-repair data, never a learned generator or model result."""
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
from nca.contract import entrance_masks
from nca.experiments import digest
from nca.volumetric import boolean_grid

VERSION = 'volume_repair_benchmark_v1'
CHANNELS = ('domain', 'permitted', 'existing', 'protected', 'support_boundary',
            'interfaces', 'requested_fraction')


def field_hash(field):
    a = np.ascontiguousarray(boolean_grid(field))
    return sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def condition(scene, fields, domain, request):
    domain = boolean_grid(domain)
    if not 0 < request <= 1 or not np.isfinite(request):
        raise ValueError('Finite request fraction in (0, 1] required')
    interfaces = np.zeros_like(domain)
    for mask in entrance_masks(scene).values():
        interfaces |= mask
    values = [domain] + [boolean_grid(fields[k]) for k in CHANNELS[1:5]] + [interfaces]
    if any(a.shape != domain.shape for a in values):
        raise ValueError('Context shapes differ')
    return np.stack(values + [np.full(domain.shape, request)], axis=0).astype(np.float32)


def context_hash(scene, fields, domain):
    # Excludes scene names, request and random seed: relabeling cannot hide leakage.
    a = condition(scene, fields, domain, .24)[:6]
    return sha256(str((a.shape, scene['voxel_size_m'])).encode() + a.tobytes()).hexdigest()


def damage(target, kind, case_id, seed=9101):
    target = boolean_grid(target)
    if not target.any():
        raise ValueError('A nonempty valid target is required; blocked sites are guards')
    cut = np.zeros_like(target)
    coords = np.argwhere(target)
    if kind == 'cube5':
        # Exactly one selected occupied center; no favorable-damage rerolls.
        key = sha256(f'{seed}:{case_id}:{kind}'.encode()).digest()
        center = coords[int.from_bytes(key[:8], 'little') % len(coords)]
        region = tuple(slice(max(0, int(v)-2), min(n, int(v)+3))
                       for v, n in zip(center, target.shape))
        cut[region] = True
    elif kind == 'slab2':
        x = int(np.sort(coords[:, 2])[len(coords)//2])
        cut[:, :, x:min(x+2, target.shape[2])] = True
    elif kind != 'intact':
        raise ValueError('Unknown frozen damage kind')
    return target & ~cut, cut


def closing_repair(damaged, domain, permitted, width=3):
    """Nonlearned local comparator: cube closing union surviving input, then legal projection."""
    field = boolean_grid(damaged)
    if type(width) is not int or width < 1 or width % 2 != 1:
        raise ValueError('Odd positive closing width required')
    radius = width//2
    padded = np.pad(field, radius, constant_values=False)
    dilated = np.lib.stride_tricks.sliding_window_view(padded, (width,)*3).any(axis=(-3,-2,-1))
    padded = np.pad(dilated, radius, constant_values=False)
    closed = np.lib.stride_tricks.sliding_window_view(padded, (width,)*3).all(axis=(-3,-2,-1))
    return (field | closed) & boolean_grid(domain) & boolean_grid(permitted)


def repair_metrics(candidate, target, damaged, domain, request):
    candidate, target, damaged = map(boolean_grid, (candidate, target, damaged))
    if any(x.shape != target.shape for x in (candidate, damaged, domain)):
        raise ValueError('Metric shapes differ')
    missing = target & ~damaged
    union = int((candidate | target).sum())
    return {'iou': int((candidate & target).sum())/union if union else 1.,
            'symmetric_difference_cells': int((candidate ^ target).sum()),
            'missing_cells': int(missing.sum()),
            'recovered_cells': int((candidate & missing).sum()),
            'recovery_fraction': float((candidate & missing).sum()/missing.sum()) if missing.any() else None,
            'surviving_cells_removed': int((damaged & ~candidate).sum()),
            'false_positive_cells': int((candidate & ~target).sum()),
            'request_error_cells': int(candidate.sum())-int(np.ceil(request*domain.sum()))}


def assert_split_integrity(rows):
    owners, target_owners = {}, {}
    ids = set()
    for row in rows:
        if row['case'] in ids:
            raise ValueError('Duplicate target case')
        ids.add(row['case'])
        for key, table in [('context_sha256', owners), ('target_sha256', target_owners)]:
            owner = table.setdefault(row[key], row['split'])
            if owner != row['split']:
                raise ValueError('Cross-split leakage: ' + key)
    return {'distinct_contexts': len(owners), 'distinct_targets': len(target_owners)}


def load_example(root, row, *, split='train'):
    """Training callers get training rows only unless explicitly selecting another split.

    A supervisor target is returned separately. No full target, cut mask, teacher
    seed, route, case identifier or checksum is a model input. This API is an
    accidental-leakage guard, not an access-control boundary on local files.
    """
    if row['split'] != split or split not in ('train', 'validation', 'test'):
        raise ValueError('Explicit matching split required')
    root = Path(root).resolve()
    path = (root / row['arrays']).resolve()
    if not path.is_relative_to(root) or digest(path) != row['arrays_sha256']:
        raise ValueError('Invalid or corrupt example')
    with np.load(path, allow_pickle=False) as pack:
        inputs = {'occupancy': pack['damaged'].astype(np.float32),
                  'context': pack['condition'].copy()}
        target = pack['target'].astype(np.float32)
    return inputs, target
