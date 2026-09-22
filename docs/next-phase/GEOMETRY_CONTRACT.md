# Scene and geometry contract, version `scene_v1`

Status: implemented 2026-09-18 as M1 step 1. Module: `nca/contract.py`. Frozen
scene set: `experiments/scenes/reference_v1/`.

This document fixes what a scene means so that a rollout, a metric and a render
can be compared without re-deriving conventions from code. It adds no constraint
family to the nine defined in `PROJECT_DEFINITION.md` and introduces no
objective, loss or training behaviour.

## Declarations

| Item | Value | Note |
|---|---|---|
| Array axis order | `(z, y, x)` | Matches `nca/evaluation.py`; `z` is up |
| Ground plane | `z = 0` slab, full extent | Written under buildings as well, as the historical generator does |
| World units | `voxel_size_m`, reference set uses `0.8` | Voxel `i` centre sits at `(i + 0.5) * voxel_size_m` |
| Extents | half-open `[start, end)` voxel indices | Matches the historical slicing |
| Entrance block | anchor corner toward increasing index, `extent = 2` | The historical generator's fixed `z:z+2, y:y+2, x:x+2` |
| Street band | `z < street_levels`, reference set uses `6` | 4.8 m at 0.8 m per voxel |
| Binarisation | `material = field > threshold`, strict | Threshold must be explicit and inside `(0, 1)` |

`street_levels = 6` follows the configuration embedded in the Model C
checkpoint. The external `notebooks/model_c/config_step_b.json` states `2`; the
embedded value is authoritative and the discrepancy is already recorded in the
change log.

## Derived regions

All are boolean fields of identical shape, read out of one rollout state rather
than re-derived, so a generator that fails to realise a declared scene is
detectable instead of being papered over.

- `permitted` — legal material: not an existing building, and either above the
  street band or inside a declared anchor zone. This reproduces
  `LocalLegalityLoss.compute_legality_field` in boolean form. The historical
  field is already binary, and equality on all six reference scenes is asserted
  by `tests/test_runtime.py`, not assumed.
- `protected` — the street-band space the design is expected to leave open:
  street band minus existing buildings minus anchor zones. This is the region
  the historical ground-openness family acted on.
- `support_boundary` — existing buildings plus the anchored street footprint.
  Declared geometry only; connection to it is not a load-path calculation.
- `endpoints` — one mask per entrance ID, for `endpoint_connectivity`.

## What the contract refuses

Each rule exists because the historical path accepted the input and produced a
scene that could not be evaluated.

1. An entrance block that leaves the grid. The generator wrote it with no bounds
   check, silently yielding a thinner entrance at the boundary.
2. An entrance block intersecting a building — an endpoint that can never be
   open, which would depress every connectivity score without explanation.
3. Overlapping entrance blocks. `endpoint_connectivity` requires disjoint named
   regions and rejects overlap; the contract catches it at scene definition.
4. A ground entrance outside the street band, or a facade entrance below it or
   not face-adjacent to any building.
5. `gap_facing_x` without `side`, or the reverse. The historical anchor code
   reads `side` only when `gap_facing_x` is set and otherwise falls through to
   right-side geometry without saying so.
6. Duplicate or empty building and entrance identifiers, degenerate extents, a
   non-positive voxel size, and a `street_levels` value outside the grid.

## Frozen reference set `reference_v1`

Six scenes, hash-verified through `manifest.json` on every load. A scene whose
bytes change, a listed file that is absent, an unlisted file in the directory,
or a canonical-hash mismatch all fail loudly: a silently edited reference scene
would invalidate every comparison recorded against it.

| Scene | Purpose |
|---|---|
| `ref-01-ground-pair` | Symmetric slabs, 16-voxel gap, ground entrances only |
| `ref-02-facade-pair-and-ground` | Three entrances at distinct heights; primary E0 replay case |
| `ref-03-wide-gap` | 20-voxel span with unequal heights |
| `ref-04-asymmetric-heights` | Entrances offset in `y` as well as `z`; no straight corridor suffices |
| `ref-05-sealed-partition` | Negative control: no legal route exists between the entrances |
| `ref-06-minimal-smoke` | Cheapest scene, for smoke use only |

`scripts/build_reference_scenes.py` regenerates the set and its manifest. It
refuses to replace an existing scene file without `--force`, which is intended
for the initial authoring pass. Changing a scene after results exist against it
requires a new set version and a recorded decision, not an overwrite.

## Reserved and deliberately unused

`ceiling_z` is validated but derives no region and is `null` in every reference
scene. A height ceiling appears in the earlier Step D specification as C3 but is
not one of the nine families Model C was trained against, so enabling it would
add a constraint category and requires a recorded user decision first.

The `protected` region likewise follows Model C's anchor-based street protection,
not Step D's explicit 3 m pedestrian / 12 m no-go strip geometry. The two are
different specifications of ground openness and must not be conflated when
comparing historical scores.

## Divergences this contract makes visible, for E0 to measure

**Superseded 2026-09-18.** The list below was written before the historical
notebook was read. Item 2 is wrong: the z-taper keys are dead in training as
well, not lost in deployment. The list is also incomplete — training adds no
per-step noise at all, and the historical evaluation used no corridor scaffold.
`ROLLOUT_PROFILES.md` carries the corrected and fuller account. The original
wording is kept here rather than edited away.

Reading the historical rollout paths against these declarations surfaced five
places where training, evaluation and serving do not agree. None is repaired
here; M1 step 3 gives each a named profile and step 4 measures it.

1. **Firing.** `UrbanPavilionNCA._step` applies its fire mask to the update
   delta and only while `self.training` is true. `grow()` forces `eval()`, so
   serving applies no internal mask. `deploy/server.py` instead blends whole
   states after the step when `fire_rate < 1.0`. Masking a delta and blending a
   state are not the same operation.
2. **z-taper.** `z_taper_strength` and `z_taper_floor` are present in both
   configurations and are not referenced anywhere in `deploy/model_utils.py`.
   Whatever taper shaped training is absent from serving.
3. **Corridor mask schedule.** Serving reuses `corridor_mask_epochs` and
   `corridor_mask_anneal`, which are training epoch counts, as rollout step
   counts. Twenty epochs and twenty steps are not the same schedule.
4. **Seed scale.** The checkpoint and the external configuration both state
   `corridor_seed_scale = 0.15`; the serving request default is `0.005`, a factor
   of thirty smaller.
5. **Global configuration mutation.** `/generate` writes `request.update_scale`
   into the shared `config` dictionary and restores it afterwards, so concurrent
   requests can observe one another's settings. This is the already-planned
   per-job configuration defect, now with a concrete location.

## Using the contract

```python
from nca.contract import (load_reference_set, to_generator_params,
                          fields_from_state, verify_state_matches_scene)

scenes = load_reference_set()
scene = scenes["ref-02-facade-pair-and-ground"]
state, info = generator.generate(to_generator_params(scene), device="cpu")
assert verify_state_matches_scene(state, config, scene) == []
fields = fields_from_state(final_state, config, scene, threshold=0.5)
```

`fields` feeds `nca.evaluation` directly: `material_legality(fields["material"],
fields["permitted"])`, `ground_openness(...)`, `geometric_support(fields
["material"], fields["support_boundary"])` and `endpoint_connectivity(fields
["material"], fields["endpoints"], source_id)`.

`fields_from_state` accepts one sample at a time and refuses a batch, so
per-scene results cannot be averaged away before they are recorded.
