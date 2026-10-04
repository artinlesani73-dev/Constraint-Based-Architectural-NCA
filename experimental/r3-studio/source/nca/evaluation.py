"""Independent binary geometry metrics, version binary_v1.

These evaluate explicit boolean fields, not soft training losses. They do not
certify walkability or mechanical safety. All array coordinates are (z, y, x).
"""
from collections import deque
from itertools import product
import numpy as np

METRIC_VERSION = "binary_v1"


def _mask(value):
    array = np.asarray(value)
    if array.ndim != 3 or array.dtype != np.bool_ or any(n == 0 for n in array.shape):
        raise ValueError("Expected a nonempty 3D boolean field; threshold explicitly first")
    return array


def _matching(*values):
    arrays = tuple(_mask(value) for value in values)
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("Fields must share the same shape")
    return arrays


def offsets(neighborhood=6):
    if neighborhood not in (6, 18, 26):
        raise ValueError("Neighborhood must be 6, 18 or 26")
    return [delta for delta in product((-1, 0, 1), repeat=3)
            if 0 < sum(abs(d) for d in delta) <= {6: 1, 18: 2, 26: 3}[neighborhood]]


def flood_fill(traversable, seeds, neighborhood=6):
    traversable, seeds = _matching(traversable, seeds)
    reached = traversable & seeds
    queue = deque(map(tuple, np.argwhere(reached)))
    neighbors = offsets(neighborhood)
    while queue:
        cell = queue.popleft()
        for delta in neighbors:
            candidate = tuple(i + d for i, d in zip(cell, delta))
            if all(0 <= i < n for i, n in zip(candidate, traversable.shape)):
                if traversable[candidate] and not reached[candidate]:
                    reached[candidate] = True
                    queue.append(candidate)
    return reached


def endpoint_connectivity(traversable, endpoints, source_id, neighborhood=6):
    """Fraction of other endpoint regions reached from one designated region.

    Endpoint identity is explicit. Overlapping masks are rejected rather than
    silently merged. Success requires at least one traversable cell per region;
    this is spatial connectivity, not a headroom/deck or route-width test.
    """
    traversable = _mask(traversable)
    if len(endpoints) < 2 or source_id not in endpoints:
        raise ValueError("Need at least two endpoint IDs and a designated source")
    used = np.zeros_like(traversable)
    checked = {}
    for name, region in endpoints.items():
        _, region = _matching(traversable, region)
        if not name or not region.any() or np.any(used & region):
            raise ValueError("Endpoint regions need unique IDs, nonempty masks and no overlaps")
        used |= region
        checked[name] = region
    source = checked[source_id] & traversable
    # A source region with several disconnected open pieces would silently
    # create multiple origins. Refuse that ambiguity instead of inflating score.
    if source.any():
        first = np.zeros_like(source)
        first[tuple(np.argwhere(source)[0])] = True
        if not np.array_equal(flood_fill(source, first, neighborhood), source):
            raise ValueError("Source endpoint has disconnected traversable pieces")
    reached = flood_fill(traversable, source, neighborhood)
    others = {name: bool(np.any(reached & region))
              for name, region in checked.items() if name != source_id}
    return {"metric_version": METRIC_VERSION, "neighborhood": neighborhood,
            "source_id": source_id, "source_open": bool(source.any()),
            "reached": others, "fraction_reached": sum(others.values()) / len(others),
            "all_connected": bool(source.any()) and all(others.values())}


def material_legality(material, permitted):
    material, permitted = _matching(material, permitted)
    count = int(material.sum())
    illegal = int((material & ~permitted).sum())
    return {"material_voxels": count, "illegal_voxels": illegal,
            "illegal_fraction": illegal / count if count else None,
            "nonempty": count > 0, "zero_illegal_voxels": illegal == 0}


def ground_openness(material, existing, protected):
    material, existing, protected = _matching(material, existing, protected)
    count = int(protected.sum())
    if not count:
        raise ValueError("Protected ground region must be nonempty")
    blocked = int(((material | existing) & protected).sum())
    return {"protected_voxels": count, "blocked_voxels": blocked,
            "open_fraction": 1 - blocked / count}


def eroded_core(material, radius=1):
    """Chebyshev erosion with empty space outside the grid; a thickness proxy.

    This is deliberately not called physical maximum thickness. A result needs
    voxel size and an agreed geometric definition before physical interpretation.
    """
    material = _mask(material)
    if not isinstance(radius, int) or isinstance(radius, bool) or radius < 1:
        raise ValueError("Radius must be a positive integer")
    core = material.copy()
    for _ in range(radius):
        padded = np.pad(core, 1, constant_values=False)
        core = np.logical_and.reduce([
            padded[z:z+material.shape[0], y:y+material.shape[1], x:x+material.shape[2]]
            for z, y, x in product(range(3), repeat=3)
        ])
    count = int(material.sum())
    return {"material_voxels": count, "core_voxels": int(core.sum()),
            "core_fraction": float(core.sum() / count) if count else None,
            "radius_voxels": radius, "nonempty": count > 0}


def geometric_support(material, support_boundary, neighborhood=6):
    """Material connected to declared support cells; no force/stress analysis."""
    material, support_boundary = _matching(material, support_boundary)
    reached = flood_fill(material | support_boundary, support_boundary, neighborhood)
    unsupported = int((material & ~reached).sum())
    count = int(material.sum())
    return {"material_voxels": count, "unsupported_voxels": unsupported,
            "supported_fraction": 1 - unsupported / count if count else None,
            "nonempty": count > 0, "interpretation": "geometric connectivity only"}
