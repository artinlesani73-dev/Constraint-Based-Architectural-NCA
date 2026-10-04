"""Sampler for the historical training distribution, set version legacy_easy_v1.

The scenes Model C was trained and evaluated on came from the notebook's own
``UrbanSceneGenerator`` at ``difficulty='easy'``. That generator takes a
difficulty label and lives only in
``notebooks/model_c/NB02_AllConstraints_v3_1_C.ipynb``; the deployed generator
takes explicit parameters instead. Neither can produce the other's scenes, so an
in-distribution replay needs the historical sampling logic transcribed into a
``scene_v1`` scene.

This module is that transcription. It is verified rather than trusted: the test
suite loads the generator class out of the notebook, runs it against this
sampler on shared seeds, and requires the realised frozen channels to be equal
voxel for voxel. The notebook is read only and never executed as a whole.

Faithfulness notes, all consequences of the historical code and none of them
repaired here:

* Every access point is written with ``type: 'facade'``. ``n_ground_access`` is
  counted into the total but never changes the type, so the ground-anchor branch
  never executed and Model C never saw a ground-type access point.
* Facade access points were placed from ``z = 3`` upward while ``street_levels``
  was 6, so some sit below the street band. A scene emitted here declares the
  ``facade_below_street_band`` relaxation when that happens, and only then.
* Buildings always span ``y`` from 0 to a sampled depth, so context is anchored
  to one edge of the grid.
* The historical generator gave access points no identity. IDs are assigned here
  in placement order purely so that endpoint metrics can name them.

The draw order below mirrors the notebook exactly, because a different order
would consume the random stream differently and produce different scenes from
the same seed.
"""

import random

from nca.contract import CONTRACT_VERSION, validate_scene

LEGACY_SET_VERSION = "legacy_easy_v1"
LEGACY_DIFFICULTY = "easy"
LEGACY_PROVENANCE = ("NB02_AllConstraints_v3_1_C.ipynb, UrbanSceneGenerator with "
                     "difficulty='easy'; transcribed, not re-derived")

#: ``_get_difficulty_params('easy')`` from the notebook. ``gap_width`` is drawn.
EASY_PARAMS = {
    "n_buildings": 2,
    "height_range": (14, 18),
    "height_variance": False,
    "width_range": (8, 12),
    "gap_width_range": (14, 18),
    "n_ground_access": 1,
    "n_elevated_access": 1,
    "anchor_budget": 0.10,
}


def sample_legacy_easy(rng, grid_size=32, street_levels=6, voxel_size_m=0.8,
                       scene_id=None):
    """One scene drawn exactly as the historical generator would have drawn it.

    ``rng`` must be a ``random.Random``. ``random.Random(s)`` and a global
    ``random.seed(s)`` share the same generator and method sequence, so the
    stream matches the notebook's for the same seed.

    Returns the raw scene dictionary. It is not validated here: a sampled scene
    may legitimately violate a contract rule that is not covered by a declared
    relaxation, and the caller decides whether to reject it and record why.
    """
    grid = grid_size
    params = dict(EASY_PARAMS)
    params["gap_width"] = rng.randint(*EASY_PARAMS["gap_width_range"])
    gap_center = grid // 2

    buildings = []

    width_one = rng.randint(*params["width_range"])
    depth_one = rng.randint(grid // 2, grid - 2)
    height_one = rng.randint(*params["height_range"])
    x_one_end = gap_center - params["gap_width"] // 2
    x_one_start = max(0, x_one_end - width_one)
    buildings.append({
        "id": "B_left", "x": [x_one_start, x_one_end], "y": [0, depth_one],
        "z": [0, height_one], "gap_facing_x": x_one_end, "side": "left",
    })

    width_two = rng.randint(*params["width_range"])
    depth_two = rng.randint(grid // 2, grid - 2)
    # height_variance is False for 'easy', so the second height is not drawn.
    height_two = height_one if not params["height_variance"] else rng.randint(
        *params["height_range"])
    x_two_start = gap_center + params["gap_width"] // 2
    x_two_end = min(grid, x_two_start + width_two)
    buildings.append({
        "id": "B_right", "x": [x_two_start, x_two_end], "y": [0, depth_two],
        "z": [0, height_two], "gap_facing_x": x_two_start, "side": "right",
    })

    total_access = params["n_ground_access"] + params["n_elevated_access"]
    entrances = []
    used_z = set()
    for index in range(total_access):
        building = rng.choice(buildings)
        z_max_building = building["z"][1]
        is_left = building["side"] == "left"

        z_min = 3
        z_max = max(z_min, z_max_building - 2)

        z = None
        for _ in range(10):
            candidate = rng.randint(z_min, z_max)
            if candidate not in used_z:
                z = candidate
                break
        if z is None:
            available = [value for value in range(z_min, z_max + 1) if value not in used_z]
            z = rng.choice(available) if available else z_min
        used_z.add(z)

        y = rng.randint(building["y"][0],
                        min(building["y"][1] - 2, building["y"][0] + grid // 3))
        x = building["x"][1] if is_left else building["x"][0] - 2
        x = max(0, min(grid - 2, x))

        entrances.append({
            "id": f"E_legacy_{index + 1}", "kind": "facade",
            "x": x, "y": y, "z": z, "extent": 2,
        })

    relaxations = []
    if any(entrance["z"] < street_levels for entrance in entrances):
        relaxations.append("facade_below_street_band")

    return {
        "contract_version": CONTRACT_VERSION,
        "scene_id": scene_id or "legacy-easy-unnamed",
        "description": (f"Historical '{LEGACY_DIFFICULTY}' training scene, gap width "
                        f"{params['gap_width']}, heights {height_one}/{height_two}."),
        "grid_size": grid,
        "voxel_size_m": voxel_size_m,
        "street_levels": street_levels,
        "ceiling_z": None,
        "legacy_relaxations": relaxations,
        "buildings": buildings,
        "entrances": entrances,
        "notes": [
            LEGACY_PROVENANCE,
            "Both access points are typed 'facade' because the historical generator "
            "typed every access point that way; the ground-anchor branch never ran.",
            "Entrance IDs are assigned in placement order; the historical generator "
            "gave access points no identity.",
        ],
    }


def legacy_seed_state(scene, config, device="cpu"):
    """The seed state the *historical* generator would have produced for this scene.

    The deployed generator cannot be used for an in-distribution replay. Its
    anchor rule was changed at deployment: where the notebook wrote ground anchor
    zones only for an access point typed ``'ground'``, the deployed version also
    writes them for any access point below ``street_levels``:

        # notebook
        if ap['type'] == 'ground':
        # deployed
        if ap.get('type') == 'ground' or ap['z'] < sl:

    Since the historical generator typed every access point ``'facade'`` and
    placed some below the street band, the deployed generator produces a wide
    ground anchor footprint for exactly those scenes where training produced
    none. Anchors enter the legality field, so this changes what the model is
    permitted to grow, not merely a bookkeeping field.

    This function reproduces the notebook's rules and is verified against the
    notebook generator voxel for voxel. The deployed generator is left untouched;
    the divergence is measured, not hidden.
    """
    import torch                     # local import keeps this module importable without torch

    scene = validate_scene(scene)
    grid = scene["grid_size"]
    street_levels = scene["street_levels"]
    if config.get("street_levels") != street_levels:
        raise ValueError(
            f"street_levels disagree: scene {street_levels}, config {config.get('street_levels')}")

    state = torch.zeros(1, config["n_channels"], grid, grid, grid, device=device)
    state[:, config["ch_ground"], 0, :, :] = 1.0

    for building in scene["buildings"]:
        x_start, x_end = building["x"]
        y_start, y_end = building["y"]
        z_start, z_end = building["z"]
        state[:, config["ch_existing"], z_start:z_end, y_start:y_end, x_start:x_end] = 1.0

    for entrance in scene["entrances"]:
        x, y, z = entrance["x"], entrance["y"], entrance["z"]
        extent = entrance["extent"]
        state[:, config["ch_access"], z:z + extent, y:y + extent, x:x + extent] = 1.0

    existing_ground = state[:, config["ch_existing"], 0, :, :]
    street_mask = 1.0 - existing_ground
    anchors = torch.zeros(1, 1, grid, grid, grid, device=device)

    # The notebook's ground branch keyed on type 'ground' alone. It is kept here
    # for fidelity even though this distribution never triggers it.
    for entrance in scene["entrances"]:
        if entrance["kind"] == "ground":
            x, y = entrance["x"], entrance["y"]
            for z in range(street_levels):
                anchors[:, 0, z, max(0, y - 2):min(grid, y + 4),
                        max(0, x - 2):min(grid, x + 4)] = 1.0

    for building in scene["buildings"]:
        y_start, y_end = building["y"]
        gap_x = building["gap_facing_x"]
        if gap_x is None:
            continue
        is_left = building["side"] == "left"
        x_start = gap_x if is_left else gap_x - 1
        x_end = gap_x + 1 if is_left else gap_x
        for z in range(street_levels):
            anchors[:, 0, z, y_start:min(y_start + 4, y_end),
                    max(0, x_start):min(grid, x_end)] = 1.0

    for z in range(street_levels):
        anchors[:, 0, z, :, :] *= street_mask

    state[:, config["ch_anchors"]:config["ch_anchors"] + 1, :, :, :] = anchors
    return state


def sample_validated(seed, **kwargs):
    """``(scene, None)`` when the drawn scene satisfies the contract, else ``(None, reason)``.

    Rejections are expected and are not failures of the sampler: the historical
    generator could place two access points whose blocks overlap, which the
    contract refuses because overlapping endpoint regions cannot be scored. The
    reason is returned so that a build records which seeds were discarded and
    why, rather than quietly resampling.
    """
    scene = sample_legacy_easy(random.Random(seed), **kwargs)
    try:
        return validate_scene(scene), None
    except ValueError as error:
        return None, str(error)
