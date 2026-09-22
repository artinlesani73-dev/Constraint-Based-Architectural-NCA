# Frozen scene sets

Status: `reference_v1` added 2026-09-18 (M1 step 1); `legacy_easy_v1` added
2026-09-18 following the user's decision that E0 should run in-distribution as
well as on designed scenes. Both sets are verified against a hash manifest on
every load.

| Set | Purpose | Scenes | Source |
|---|---|---|---|
| `reference_v1` | Designed for evaluability: span, asymmetry, a sealed-partition negative control | 6 | authored |
| `legacy_easy_v1` | Reproduces the distribution Model C was trained and evaluated on | 12 | historical `easy` sampler, seeds 0-11 |

## Why a second set was needed

The historical scene generator takes a difficulty label and lives only in
`notebooks/model_c/NB02_AllConstraints_v3_1_C.ipynb`. The deployed generator
takes explicit parameters. Neither can produce the other's scenes, so without a
transcription E0 could measure how the rollout profiles differ but could not
separate a weak model from out-of-distribution input.

`nca/legacy_scenes.py` is that transcription, and it is verified rather than
trusted: `tests/test_legacy_scenes.py` extracts the generator class out of the
notebook, executes it in isolation as an oracle, and requires the replayed seed
state to equal the notebook's voxel for voxel on every accepted seed. The
notebook is read only.

Seeds are consumed from zero upward with no cherry-picking. Each is accepted or
recorded with the reason it was refused, and the manifest carries that record.
Seeds 0-11 all passed; rejections do occur further out, where the historical
generator places two access blocks one voxel apart in `z` and they overlap,
which the contract refuses because overlapping endpoint regions cannot be scored.

## Declared relaxations

The historical generator placed facade access points from `z = 3` upward while
`street_levels` was 6, so some sit below the street band — a combination
`scene_v1` refuses by default. Rather than weaken the rule, the contract gained
named relaxations: a scene may declare `facade_below_street_band`, the name is
covered by the scene hash, an unknown or repeated name is refused, and face
adjacency to a building is still required. Four of the twelve legacy scenes
declare it; the designed set declares none, and a test enforces that.

An empty relaxation list is omitted from the canonical form, so declaring no
relaxations hashes identically to a scene authored before relaxations existed.
That is what allowed this additive field without revising `reference_v1`: those
six files are byte-identical to when they were frozen.

## The deployed generator cannot replay this set

Reproducing the distribution surfaced a behaviour change made at deployment. The
notebook wrote ground anchor zones only for an access point typed `'ground'`:

```python
if ap['type'] == 'ground':          # notebook
if ap.get('type') == 'ground' or ap['z'] < sl:   # deployed
```

Since the historical generator typed every access point `'facade'` and placed
some below the street band, the deployed generator writes a wide ground anchor
footprint for exactly the scenes where training produced none. Anchors feed the
legality field, so this widens what the model is permitted to grow rather than
changing a bookkeeping field only. Tests confirm the anchor channels diverge
exactly when an entrance sits below the street band, that the deployed rule only
ever adds anchors, and that the permitted region strictly grows.

`nca.legacy_scenes.legacy_seed_state` is therefore the seed builder for this set,
and the manifest records that. The deployed generator is left untouched; the
divergence is measured, not removed.

## Orientation observation, not a record

Unrecorded, with no run ID: the three profiles run on all twelve legacy scenes at
one shared seed and 50 steps.

| Profile | Mean material voxels | Scenes whose entrances the structure connects |
|---|---|---|
| `historical-training` | 1076.4 | 10 of 12 |
| `historical-evaluation` | 27.7 | 0 of 12 |
| `historical-serving` | 805.8 | 10 of 12 |

Legality was perfect and geometric support complete under every profile. The gap
is the corridor scaffold: training seeds the structure channel at 0.15 and
serving at 0.005, while the historical evaluation profile seeds nothing at all
and produces about 28 voxels out of 32768 — effectively empty geometry.

Two cautions on reading this.

First, it bears on how `v31_evaluation.json` should be read, since that file was
produced by the evaluation profile. Its `avg_coverage` of 0.04 is consistent with
near-empty output. Its `avg_access_reach` of 0.62 is **not** in contradiction with
the zero column above: the historical metric measured reachability through
ground-level void, which an empty design satisfies trivially, whereas the column
above measures connectivity through the grown structure. They answer different
questions, and the historical figure is not wrong — it is measuring something an
empty result scores well on.

Second, none of this is yet evidence. It is one seed, one step count, one
checkpoint, and no run record. E0 is what turns it into a result, per scene, with
full provenance, across both sets.
