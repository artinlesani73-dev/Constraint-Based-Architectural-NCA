"""Scene and geometry contract, version scene_v1.

This module fixes what a scene *means* so that a rollout, a metric and a render
can be compared without re-guessing conventions. It declares axes, world units,
extent conventions, entrance identity, the legal-material region, the protected
ground void, the declared support region and the binarisation threshold.

Scope limits, stated deliberately:

* This is a naming and validation contract. It adds no constraint family to the
  nine already defined in ``PROJECT_DEFINITION.md`` and introduces no objective.
* ``permitted`` reproduces the historical legality semantics in boolean form so
  that ``nca.evaluation`` can be pointed at the same region the model was masked
  with. Agreement with the legacy float field is asserted by test, not assumed.
* ``support_boundary`` is a declared geometric region. Connection to it is not a
  structural-engineering result.
* No walkability, headroom, clearance or code-compliance claim is made anywhere
  here. Entrance connectivity is spatial connectivity of explicit voxels.

Array coordinates are ``(z, y, x)`` throughout, matching ``nca.evaluation``.
"""

from hashlib import sha256
from pathlib import Path
import json

import numpy as np

CONTRACT_VERSION = "scene_v1"

#: Array axis order for every 3D field in this project.
AXIS_ORDER = ("z", "y", "x")

#: ``z`` increases upward. Index 0 is the ground plane slab. The centre of voxel
#: ``i`` along any axis sits at ``(i + 0.5) * voxel_size_m`` from the grid origin.
UP_AXIS = "z"

#: Extents are half-open ``[start, end)`` voxel index pairs, matching the slicing
#: the historical scene generator performs.
EXTENT_CONVENTION = "half-open [start, end) in voxel indices"

#: Entrance blocks are written from their anchor corner toward increasing index
#: on every axis: ``[z, z+extent) x [y, y+extent) x [x, x+extent)``. The
#: historical generator used a fixed extent of 2 and did not bounds-check it.
ENTRANCE_ANCHOR = "minimum-index corner"
ENTRANCE_EXTENT_DEFAULT = 2

#: Structure is binarised strictly: ``material = field > threshold``. A value
#: exactly equal to the threshold is empty. Thresholding is never implicit.
MATERIAL_THRESHOLD_DEFAULT = 0.5

ENTRANCE_KINDS = ("ground", "facade")
BUILDING_SIDES = ("left", "right")

#: Named, declared exemptions from a validation rule, for scenes that reproduce a
#: historical distribution rather than a designed one. A relaxation must be listed
#: in the scene, is covered by the scene hash, and is refused if unknown. Nothing
#: is ever relaxed by default, and no relaxation weakens a derived region: a
#: relaxed scene is still evaluated by exactly the same rules.
RELAXATIONS = {
    "facade_below_street_band": (
        "Permits a facade entrance to start below street_levels. The historical "
        "training generator placed facade access points from z=3 upward while "
        "street_levels was 6, so an in-distribution replay scene needs this. "
        "Face adjacency to a building is still required."),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SET_DIR = REPO_ROOT / "experiments" / "scenes" / "reference_v1"
MANIFEST_NAME = "manifest.json"

_AXES = ("x", "y", "z")


def _as_numpy(field):
    """Accept a NumPy array or a torch tensor without importing torch."""
    if hasattr(field, "detach"):
        field = field.detach().cpu().numpy()
    return np.asarray(field)


def _integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be a plain integer")
    return value


def _extent(pair, grid_size, name):
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError(f"{name} must be a [start, end) pair")
    start, end = (_integer(v, f"{name} bound") for v in pair)
    if not 0 <= start < end <= grid_size:
        raise ValueError(f"{name} must satisfy 0 <= start < end <= {grid_size}")
    return start, end


def _box(grid_size, spans):
    mask = np.zeros((grid_size,) * 3, bool)
    (x0, x1), (y0, y1), (z0, z1) = spans
    mask[z0:z1, y0:y1, x0:x1] = True
    return mask


def _z_index(grid_size):
    return np.arange(grid_size).reshape(grid_size, 1, 1)


def validate_scene(scene):
    """Return a normalised copy of ``scene`` or raise ``ValueError``.

    Every rule here exists because the historical path accepted the input
    silently and produced a scene that could not be evaluated: an entrance
    truncated at the grid boundary, an entrance buried inside a building, two
    entrances sharing voxels, or a facade anchor implied by a missing ``side``.
    """
    if not isinstance(scene, dict):
        raise ValueError("A scene must be a JSON object")
    if scene.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(f"Scene must declare contract_version {CONTRACT_VERSION!r}")

    scene_id = scene.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ValueError("Scene needs a nonempty scene_id")

    grid_size = _integer(scene.get("grid_size"), "grid_size")
    if grid_size < 4:
        raise ValueError("grid_size must be at least 4")

    voxel_size_m = scene.get("voxel_size_m")
    if not isinstance(voxel_size_m, (int, float)) or isinstance(voxel_size_m, bool):
        raise ValueError("voxel_size_m must be a number")
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be finite and positive")

    street_levels = _integer(scene.get("street_levels"), "street_levels")
    if not 1 <= street_levels <= grid_size:
        raise ValueError(f"street_levels must lie in [1, {grid_size}]")

    relaxations = scene.get("legacy_relaxations", [])
    if not isinstance(relaxations, list):
        raise ValueError("legacy_relaxations must be a list of declared names")
    unknown = [name for name in relaxations if name not in RELAXATIONS]
    if unknown:
        raise ValueError(f"Unknown legacy_relaxations: {unknown}; "
                         f"declared names must be among {sorted(RELAXATIONS)}")
    if len(set(relaxations)) != len(relaxations):
        raise ValueError("legacy_relaxations must not repeat a name")
    relaxations = sorted(relaxations)

    ceiling_z = scene.get("ceiling_z", None)
    if ceiling_z is not None:
        # Reserved. A height ceiling is not one of the nine existing families;
        # enabling it requires a recorded user decision, so nothing derives a
        # region from it yet.
        _integer(ceiling_z, "ceiling_z")
        if not 1 <= ceiling_z <= grid_size:
            raise ValueError(f"ceiling_z must lie in [1, {grid_size}]")

    buildings = scene.get("buildings")
    if not isinstance(buildings, list) or not buildings:
        raise ValueError("A scene needs at least one existing building")
    normalised_buildings = []
    seen_building_ids = set()
    for index, building in enumerate(buildings):
        if not isinstance(building, dict):
            raise ValueError("Each building must be an object")
        identifier = building.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"Building {index} needs a nonempty id")
        if identifier in seen_building_ids:
            raise ValueError(f"Duplicate building id: {identifier}")
        seen_building_ids.add(identifier)
        spans = tuple(_extent(building.get(axis), grid_size, f"{identifier}.{axis}")
                      for axis in _AXES)
        gap_facing_x = building.get("gap_facing_x", None)
        side = building.get("side", None)
        if (gap_facing_x is None) != (side is None):
            # The legacy anchor code reads `side` only when gap_facing_x is set
            # and otherwise falls through to right-side geometry without saying so.
            raise ValueError(f"{identifier}: declare gap_facing_x and side together or neither")
        if gap_facing_x is not None:
            _integer(gap_facing_x, f"{identifier}.gap_facing_x")
            if not 0 <= gap_facing_x <= grid_size:
                raise ValueError(f"{identifier}.gap_facing_x is outside the grid")
            if side not in BUILDING_SIDES:
                raise ValueError(f"{identifier}.side must be one of {BUILDING_SIDES}")
        normalised_buildings.append({
            "id": identifier,
            "x": list(spans[0]), "y": list(spans[1]), "z": list(spans[2]),
            "gap_facing_x": gap_facing_x, "side": side,
        })

    existing = np.zeros((grid_size,) * 3, bool)
    for building in normalised_buildings:
        existing |= _box(grid_size, (building["x"], building["y"], building["z"]))

    entrances = scene.get("entrances")
    if not isinstance(entrances, list) or len(entrances) < 2:
        raise ValueError("A scene needs at least two entrances to evaluate connectivity")
    normalised_entrances = []
    claimed = np.zeros((grid_size,) * 3, bool)
    seen_entrance_ids = set()
    for index, entrance in enumerate(entrances):
        if not isinstance(entrance, dict):
            raise ValueError("Each entrance must be an object")
        identifier = entrance.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"Entrance {index} needs a nonempty id")
        if identifier in seen_entrance_ids:
            raise ValueError(f"Duplicate entrance id: {identifier}")
        seen_entrance_ids.add(identifier)
        kind = entrance.get("kind")
        if kind not in ENTRANCE_KINDS:
            raise ValueError(f"{identifier}.kind must be one of {ENTRANCE_KINDS}")
        extent = _integer(entrance.get("extent", ENTRANCE_EXTENT_DEFAULT), f"{identifier}.extent")
        if extent < 1:
            raise ValueError(f"{identifier}.extent must be at least 1")
        corner = {axis: _integer(entrance.get(axis), f"{identifier}.{axis}") for axis in _AXES}
        for axis, value in corner.items():
            if value < 0 or value + extent > grid_size:
                raise ValueError(
                    f"{identifier}: block [{value}, {value + extent}) on {axis} leaves the grid; "
                    "the historical generator truncated this silently")
        block = _box(grid_size, tuple((corner[axis], corner[axis] + extent) for axis in _AXES))
        if np.any(block & existing):
            raise ValueError(f"{identifier}: entrance block intersects an existing building")
        if np.any(block & claimed):
            raise ValueError(f"{identifier}: entrance blocks overlap; endpoint IDs must be disjoint")
        claimed |= block
        if kind == "ground":
            if corner["z"] + extent > street_levels:
                raise ValueError(
                    f"{identifier}: a ground entrance must lie inside the street band "
                    f"[0, {street_levels})")
        else:
            if corner["z"] < street_levels and "facade_below_street_band" not in relaxations:
                raise ValueError(
                    f"{identifier}: a facade entrance must start at or above z={street_levels}, "
                    "or the scene must declare the facade_below_street_band relaxation")
            if not _touches_existing(block, existing):
                raise ValueError(
                    f"{identifier}: a facade entrance must be face-adjacent to a building")
        normalised_entrances.append({
            "id": identifier, "kind": kind, "extent": extent,
            "x": corner["x"], "y": corner["y"], "z": corner["z"],
        })

    return {
        "contract_version": CONTRACT_VERSION,
        "scene_id": scene_id,
        "description": scene.get("description", ""),
        "grid_size": grid_size,
        "voxel_size_m": float(voxel_size_m),
        "street_levels": street_levels,
        "ceiling_z": ceiling_z,
        "legacy_relaxations": relaxations,
        "buildings": normalised_buildings,
        "entrances": normalised_entrances,
        "notes": list(scene.get("notes", [])),
    }


def _touches_existing(block, existing):
    """Face adjacency (6-neighbourhood) without wrapping across the grid."""
    for axis in range(3):
        for shift in (1, -1):
            shifted = np.zeros_like(block)
            source = [slice(None)] * 3
            target = [slice(None)] * 3
            if shift == 1:
                target[axis] = slice(1, None)
                source[axis] = slice(0, -1)
            else:
                target[axis] = slice(0, -1)
                source[axis] = slice(1, None)
            shifted[tuple(target)] = block[tuple(source)]
            if np.any(shifted & existing):
                return True
    return False


#: Optional keys that are omitted from the canonical form when they hold their
#: empty default. Declaring no relaxations therefore hashes identically to a
#: scene authored before relaxations existed, which is what lets an additive
#: optional field be introduced without revising a frozen set. Any key listed
#: here must be empty-means-absent in meaning, never empty-means-something.
CANONICAL_OMIT_WHEN_EMPTY = ("legacy_relaxations",)


def _canonical_view(scene):
    view = dict(scene)
    for key in CANONICAL_OMIT_WHEN_EMPTY:
        if not view.get(key):
            view.pop(key, None)
    return view


def canonical_json(scene):
    """Byte-stable serialisation; the basis of the scene hash."""
    scene = validate_scene(scene)
    return (json.dumps(_canonical_view(scene), indent=2, sort_keys=True,
                       allow_nan=False, ensure_ascii=True) + "\n").encode("utf-8")


def scene_hash(scene):
    """Hash of the declared scene only. Derived arrays are not covered."""
    return sha256(canonical_json(scene)).hexdigest()


def to_generator_params(scene):
    """Parameters for the historical ``UrbanSceneGenerator``.

    Entrance IDs and extents do not exist in the legacy format, so they are kept
    on the contract side. ``extent`` other than the legacy value of 2 cannot be
    expressed to that generator and is refused here rather than silently ignored.
    """
    scene = validate_scene(scene)
    for entrance in scene["entrances"]:
        if entrance["extent"] != ENTRANCE_EXTENT_DEFAULT:
            raise ValueError(
                f"{entrance['id']}: the historical generator writes a fixed "
                f"{ENTRANCE_EXTENT_DEFAULT}-voxel block and cannot express extent "
                f"{entrance['extent']}")
    return {
        "buildings": [{"x": tuple(b["x"]), "y": tuple(b["y"]), "z": tuple(b["z"]),
                       "gap_facing_x": b["gap_facing_x"], "side": b["side"]}
                      for b in scene["buildings"]],
        "access_points": [{"x": e["x"], "y": e["y"], "z": e["z"], "type": e["kind"]}
                          for e in scene["entrances"]],
    }


def declared_existing(scene):
    scene = validate_scene(scene)
    grid_size = scene["grid_size"]
    existing = np.zeros((grid_size,) * 3, bool)
    for building in scene["buildings"]:
        existing |= _box(grid_size, (building["x"], building["y"], building["z"]))
    return existing


def entrance_masks(scene):
    """Endpoint regions keyed by entrance ID, for ``endpoint_connectivity``."""
    scene = validate_scene(scene)
    grid_size = scene["grid_size"]
    masks = {}
    for entrance in scene["entrances"]:
        extent = entrance["extent"]
        masks[entrance["id"]] = _box(grid_size, tuple(
            (entrance[axis], entrance[axis] + extent) for axis in _AXES))
    return masks


def permitted_region(existing, anchors, street_levels):
    """Legal material region in boolean form.

    Mirrors ``LocalLegalityLoss.compute_legality_field``: space that is not
    occupied by an existing building, and either above the street band or inside
    a declared anchor zone. Equality with the legacy float field is asserted by
    test on the reference scenes.
    """
    existing = _as_numpy(existing).astype(bool)
    anchors = _as_numpy(anchors).astype(bool)
    if existing.shape != anchors.shape or existing.ndim != 3:
        raise ValueError("existing and anchors must be matching 3D fields")
    grid_size = existing.shape[0]
    if not 1 <= street_levels <= grid_size:
        raise ValueError("street_levels is outside the grid")
    above_street = _z_index(grid_size) >= street_levels
    return ~existing & (above_street | anchors)


def protected_void(existing, anchors, street_levels):
    """Street-band space the design is expected to leave open.

    This is the complement of the anchor allowance inside the street band, which
    is the region the historical ground-openness family acted on. It is not the
    explicit pedestrian/no-go geometry of the earlier Step D specification.
    """
    existing = _as_numpy(existing).astype(bool)
    anchors = _as_numpy(anchors).astype(bool)
    grid_size = existing.shape[0]
    street_band = _z_index(grid_size) < street_levels
    band = np.broadcast_to(street_band, existing.shape)
    return band & ~existing & ~anchors


def support_region(existing, anchors, street_levels):
    """Declared support cells: existing buildings and anchored street footprint.

    Geometric only. Connection to this region is not a load-path calculation.
    """
    existing = _as_numpy(existing).astype(bool)
    anchors = _as_numpy(anchors).astype(bool)
    grid_size = existing.shape[0]
    street_band = np.broadcast_to(_z_index(grid_size) < street_levels, existing.shape)
    return existing | (anchors & street_band)


def binarise(field, threshold=MATERIAL_THRESHOLD_DEFAULT):
    """Strict thresholding. ``threshold`` must be explicit and inside (0, 1)."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be a number")
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between 0 and 1")
    return _as_numpy(field) > threshold


def fields_from_state(state, config, scene, threshold=MATERIAL_THRESHOLD_DEFAULT):
    """Boolean evaluation fields read out of one rollout state.

    ``state`` is a single ``(C, G, G, G)`` sample or a ``(1, C, G, G, G)`` batch.
    The frozen channels are read as the model saw them rather than re-derived, so
    a generator that did not realise the declared scene is detectable instead of
    being papered over. ``verify_state_matches_scene`` performs that check.
    """
    scene = validate_scene(scene)
    array = _as_numpy(state)
    if array.ndim == 5:
        if array.shape[0] != 1:
            raise ValueError("Pass one sample at a time; batch results must stay per-scene")
        array = array[0]
    if array.ndim != 4:
        raise ValueError("Expected a (C, G, G, G) state")
    grid_size = scene["grid_size"]
    if array.shape[1:] != (grid_size,) * 3:
        raise ValueError(f"State grid {array.shape[1:]} does not match scene grid {grid_size}")
    street_levels = scene["street_levels"]
    if config.get("street_levels") != street_levels:
        raise ValueError(
            f"street_levels disagree: scene {street_levels}, config {config.get('street_levels')}")

    existing = array[config["ch_existing"]] > 0.5
    anchors = array[config["ch_anchors"]] > 0.5
    material = binarise(array[config["ch_structure"]], threshold)
    return {
        "contract_version": CONTRACT_VERSION,
        "scene_id": scene["scene_id"],
        "threshold": float(threshold),
        "material": material,
        "existing": existing,
        "anchors": anchors,
        "permitted": permitted_region(existing, anchors, street_levels),
        "protected": protected_void(existing, anchors, street_levels),
        "support_boundary": support_region(existing, anchors, street_levels),
        "endpoints": entrance_masks(scene),
    }


def verify_state_matches_scene(state, config, scene):
    """Problems found comparing a realised state against its declared scene.

    An empty list means the frozen channels carry the declared buildings and
    entrance blocks exactly. Anything else is reported rather than raised so a
    caller can record it in a run and continue.
    """
    scene = validate_scene(scene)
    array = _as_numpy(state)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    problems = []
    existing = array[config["ch_existing"]] > 0.5
    declared = declared_existing(scene)
    if not np.array_equal(existing, declared):
        problems.append(
            f"existing channel differs from declared buildings in "
            f"{int(np.sum(existing != declared))} voxels")
    access = array[config["ch_access"]] > 0.5
    union = np.zeros_like(access)
    for mask in entrance_masks(scene).values():
        union |= mask
    if not np.array_equal(access, union):
        problems.append(
            f"access channel differs from declared entrance blocks in "
            f"{int(np.sum(access != union))} voxels")
    ground = array[config["ch_ground"]] > 0.5
    expected_ground = np.zeros_like(ground)
    expected_ground[0] = True
    if not np.array_equal(ground, expected_ground):
        problems.append("ground channel is not exactly the z=0 slab")
    return problems


def load_scene(path):
    return validate_scene(json.loads(Path(path).read_text(encoding="utf-8")))


def load_reference_set(directory=None):
    """Load a frozen scene set, verifying it against its manifest.

    Refuses a manifest hash mismatch, a listed file that is absent, and a scene
    file present in the directory but absent from the manifest. A frozen set that
    drifts must fail loudly; a silently edited reference scene would invalidate
    every comparison made against it.
    """
    directory = Path(directory) if directory is not None else REFERENCE_SET_DIR
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing scene manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(f"Manifest must declare contract_version {CONTRACT_VERSION!r}")
    listed = manifest.get("scenes")
    if not isinstance(listed, list) or not listed:
        raise ValueError("Manifest lists no scenes")

    present = {path.name for path in directory.glob("*.json")} - {MANIFEST_NAME}
    named = {entry["file"] for entry in listed}
    if present != named:
        raise ValueError(
            f"Scene directory and manifest disagree; unlisted {sorted(present - named)}, "
            f"missing {sorted(named - present)}")

    scenes = {}
    for entry in listed:
        path = directory / entry["file"]
        scene = load_scene(path)
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Frozen scene changed on disk: {entry['file']}")
        if scene_hash(scene) != entry["scene_hash"]:
            raise ValueError(f"Canonical scene hash mismatch: {entry['file']}")
        if scene["scene_id"] != entry["scene_id"]:
            raise ValueError(f"scene_id mismatch for {entry['file']}")
        if scene["scene_id"] in scenes:
            raise ValueError(f"Duplicate scene_id in set: {scene['scene_id']}")
        scenes[scene["scene_id"]] = scene
    return scenes


def build_manifest(directory, set_version):
    """Recompute a manifest for a scene directory. Writing it is the caller's job."""
    directory = Path(directory)
    entries = []
    for path in sorted(directory.glob("*.json")):
        if path.name == MANIFEST_NAME:
            continue
        scene = load_scene(path)
        entries.append({
            "file": path.name,
            "scene_id": scene["scene_id"],
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "scene_hash": scene_hash(scene),
        })
    return {
        "contract_version": CONTRACT_VERSION,
        "set_version": set_version,
        "axis_order": list(AXIS_ORDER),
        "extent_convention": EXTENT_CONVENTION,
        "entrance_anchor": ENTRANCE_ANCHOR,
        "material_threshold_default": MATERIAL_THRESHOLD_DEFAULT,
        "scenes": entries,
    }
