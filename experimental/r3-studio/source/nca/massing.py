"""massing_v1: building-volume interpretation and explicit completion controls.

No trained model, new loss, automatic acceptance gate or construction claim.
All completions preserve source cells; rejected additions stay in the report.
"""
import math
import numpy as np
from nca.contract import validate_scene, declared_existing
from nca.volumetric import boolean_grid, exterior_free, components

VERSION = 'massing_v1'
METHODS = ('identity', 'sealed_cavities', 'axis_span')


def positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def opportunity_region(scene, permitted, padding_m_zyx=(6.4, 6.4, 0.0)):
    """Fixed endpoint bounding box + physical padding, sampled at voxel centers.

    Intersect with supplied historical legality, excluding context. A diagnostic
    denominator only: this introduces no height ceiling or new legal constraint.
    Resolution comparisons must resample the same physical scene and legality.
    """
    scene = validate_scene(scene)
    permitted = boolean_grid(permitted)
    n, size = scene['grid_size'], scene['voxel_size_m']
    if permitted.shape != (n, n, n):
        raise ValueError('Permitted grid must match the declared scene')
    if len(padding_m_zyx) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in padding_m_zyx):
        raise ValueError('Padding must contain three finite nonnegative metre values')
    bounds = []
    box = np.ones_like(permitted)
    centers = (np.arange(n) + .5) * size
    for axis, (key, pad) in enumerate(zip(('z', 'y', 'x'), padding_m_zyx)):
        lo = min(e[key] for e in scene['entrances']) * size - pad
        hi = max(e[key] + e['extent'] for e in scene['entrances']) * size + pad
        bounds.append([max(0.0, lo), min(n * size, hi)])
        shape = [1, 1, 1]; shape[axis] = n
        box &= ((centers >= lo) & (centers < hi)).reshape(shape)
    region = box & permitted & ~declared_existing(scene)
    if not region.any():
        raise ValueError('Opportunity region is empty')
    return region, {'version': VERSION, 'padding_m_zyx': list(padding_m_zyx),
                    'clipped_bounds_m_zyx': bounds, 'domain_voxels': int(region.sum()),
                    'domain_volume_m3': int(region.sum()) * size ** 3,
                    'sampling': 'half-open physical box; voxel centers; historical legality intersection',
                    'role': 'diagnostic domain; not a new legality limit or budget target'}


def complete_mass(source, existing, region, method='sealed_cavities', *, voxel_size_m=.8, axis=0, max_gap_m=6.4):
    """One pass: enclosed cavities or consecutive occupied samples along an axis.

    Cavities use six-neighbor exterior reachability on the full grid, with source
    occupancy alone providing enclosure. Context never supplies closure. Axis span
    fills gaps between consecutive occupied cells if empty gap length <= max_gap_m;
    axis 0 is vertical Z. Context/domain filter additions, never remove source.
    Axis span can bridge intentional gaps: it is an explicit experimental control.
    """
    source, existing, region = map(boolean_grid, (source, existing, region))
    if source.shape != existing.shape or source.shape != region.shape or not region.any():
        raise ValueError('Matching grids and a nonempty region required')
    positive_number(voxel_size_m, 'Voxel size')
    positive_number(max_gap_m, 'Maximum gap')
    if type(axis) is not int or axis not in (0, 1, 2):
        raise ValueError('Axis must be 0, 1 or 2 in z,y,x order')
    if method not in METHODS:
        raise ValueError('Unknown completion method')
    wanted = np.zeros_like(source)
    if method == 'sealed_cavities':
        wanted = ~source & ~exterior_free(source)
    elif method == 'axis_span':
        lines = np.moveaxis(source, axis, 0)
        target = np.moveaxis(wanted, axis, 0)
        for index in np.ndindex(lines.shape[1:]):
            occupied = np.flatnonzero(lines[(slice(None),) + index])
            for left, right in zip(occupied[:-1], occupied[1:]):
                gap = int(right - left - 1)
                if gap and gap * voxel_size_m <= max_gap_m + 1e-10:
                    target[(slice(left + 1, right),) + index] = True
    added = wanted & region & ~existing
    rejected = wanted & ~(region & ~existing)
    result = source | added
    report = {'version': VERSION, 'method': method, 'axis_zyx': axis,
              'voxel_size_m': voxel_size_m, 'max_gap_m': max_gap_m,
              'source_voxels': int(source.sum()), 'requested_additions': int(wanted.sum()),
              'added_voxels': int(added.sum()), 'rejected_additions': int(rejected.sum()),
              'requested_in_context': int((wanted & existing).sum()),
              'requested_outside_domain': int((wanted & ~region).sum()),
              'source_in_context': int((source & existing).sum()),
              'source_outside_domain': int((source & ~region).sum()),
              'result_voxels': int(result.sum()), 'learned': False}
    return result, report, {'requested': wanted, 'added': added, 'rejected': rejected}


def measure_mass(field, existing, region, voxel_size_m=.8):
    field, existing, region = map(boolean_grid, (field, existing, region))
    if field.shape != existing.shape or field.shape != region.shape or not region.any():
        raise ValueError('Matching grids and a nonempty domain required')
    positive_number(voxel_size_m, 'Voxel size')
    coords = np.argwhere(field)
    extents = (coords.max(0) - coords.min(0) + 1).tolist() if len(coords) else [0, 0, 0]
    return {'version': VERSION, 'occupancy_meaning': 'building_volume_interiors_deferred',
            'occupied_voxels': int(field.sum()), 'gross_volume_m3': int(field.sum()) * voxel_size_m ** 3,
            'extent_m_zyx': [v * voxel_size_m for v in extents],
            'components_6': components(field), 'context_collision_voxels': int((field & existing).sum()),
            'outside_domain_voxels': int((field & ~region).sum()),
            'domain_occupancy_fraction': int((field & region).sum()) / int(region.sum()),
            'domain_voxels': int(region.sum()), 'budget_target': None}
