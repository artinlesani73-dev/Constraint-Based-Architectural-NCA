# Rollout profiles, version `rollout_v1`

Status: implemented 2026-09-18 as M1 step 3. Module: `nca/rollout.py`. Tests:
`tests/test_rollout.py`. The legacy code paths are untouched and remain the
reference for exact comparison.

This document supersedes the divergence list in `GEOMETRY_CONTRACT.md`, which
was written before the historical notebook was read and contains one claim that
turned out to be wrong. The correction is noted below and in the change log.

## Sources

Each profile is reconstructed from a primary source and records it:

| Profile | Source |
|---|---|
| `historical-training` | `ArchitecturalIntentTrainerV31.train_epoch`, `notebooks/model_c/NB02_AllConstraints_v3_1_C.ipynb` |
| `historical-evaluation` | `ArchitecturalIntentTrainerV31.evaluate`, same notebook; this is what produced `v31_evaluation.json` |
| `historical-serving` | the `/generate` handler in `deploy/server.py`, at its own request defaults |

## What the three disagree about

Every row is confirmed in source, not inferred.

| Axis | training | evaluation | serving |
|---|---|---|---|
| Module mode | `train()` | `eval()` | `eval()` |
| Firing | mask on the update delta, rate 0.65 | none | blend of whole states, only when rate < 1; request default is 1.0 |
| Per-step noise | none | none | `randn_like` on grown channels, request default 0.02 |
| Steps | `random.randint(30, 50)` per epoch | fixed 50 | request default 50 |
| Corridor seed scale | 0.15 | none | request default 0.005 |
| Corridor mask | once on the seed, gated on the epoch index | none | every step, gated on the step index |
| Scene source | procedural `easy` | procedural `easy` | user-supplied parameters |

Three consequences worth stating plainly.

**The recorded historical evaluation had no corridor scaffold at all.**
`evaluate` computes the corridor target and then calls `model.grow(scene,
steps=50)` directly, so the figures in `v31_evaluation.json` come from a rollout
that received neither the 0.15 seeding training relied on nor any mask. Those
figures are therefore not a measurement of the configuration the model was
trained under.

**Firing is a different operation in each place it appears.** Training multiplies
the update delta by a Bernoulli mask before it is applied, which attenuates the
increment. Serving runs the full step and then blends the previous and new
states. The two coincide only in the degenerate cases of rate 0 and rate 1.

**Per-step noise is a deployment invention.** `noise_std` sits in both
configurations, and neither the training loop nor the notebook's `_step` adds
noise anywhere. Only `/generate` injects it.

### Correction to an earlier claim

`GEOMETRY_CONTRACT.md` stated that "whatever taper shaped training is absent
from serving". That was wrong. `z_taper_strength` and `z_taper_floor` appear in
both configurations and in no code path in either the notebook or the
deployment. They are dead configuration keys. Nothing was lost in deployment,
and no taper shaped training.

## Distribution mismatch, and what E0 may therefore claim

The historical scene generator differs from the deployed one in ways that bound
what any replay can establish.

- In the notebook generator, every access point is written with `type:
  'facade'`. `n_ground_access` is counted into the total but never changes the
  type, so the ground-anchor branch of `_generate_anchor_zones` never executed
  during training. Model C never saw a ground-type access point, while the
  deployed interface offers them.
- Training access points were placed between `z = 3` and `building_height - 2`
  with `street_levels = 6`, so some were typed `facade` while sitting below
  street level. The `scene_v1` contract rejects that combination.
- Training buildings always spanned `y` from 0 to a sampled depth, so context
  was anchored to one edge of the grid.
- The deployed generator takes explicit parameters instead of a difficulty
  label, so no user scene is drawn from the training distribution.

The frozen `reference_v1` set was authored for evaluability, not to match the
training distribution, and it does not. E0 on that set can therefore measure how
the three profiles differ from one another on well-defined scenes. It cannot
reproduce the historical aggregate, and a poor score on it is not by itself
evidence that the model is worse than reported — it may be evidence of
out-of-distribution input. Whether to add a second frozen set sampled from the
`easy` generator at fixed seeds is an open decision.

## Behaviour of the shared implementation

`run_rollout(model, seed_state, profile, ...)` returns the final state together
with a record of what was applied: step count and its origin, how many steps
received noise, firing and masking, and whether the seed was scaled or masked.

Guarantees, each covered by a test:

- The serving profile is bitwise identical to the legacy `/generate` loop across
  six request variants, including stochastic firing and altered update scale.
  Without this, a later behaviour change could not be attributed to the change
  rather than to the rewrite.
- The evaluation profile is bitwise identical to `model.grow(seed, steps=50)`.
- The seed state is never mutated, and the shared `config` dictionary is never
  mutated: the update-scale override is scoped and restored in a `finally`
  block. This is narrower than the legacy handler, which leaked on an exception.
  It is still not per-job isolation, since two rollouts sharing one model
  instance would contend; that remains M4 work.
- Module mode is restored even when a step raises.
- A corridor target is required by any profile that uses one and refused by any
  profile that does not, so a corridor that does nothing cannot be passed
  silently.
- `rng_source` must match how the stream is drawn: the historical profiles
  declare `global` and refuse an explicit generator, because reproducing a
  historical stream means drawing from the source the original drew from. A
  variant may declare `explicit` and then becomes reproducible independently of
  global state.
- `profile.replace(...)` renames the result, so a modified profile can never be
  recorded under a historical name.

## Unrecorded observation

For orientation only, with no run ID and no standing as evidence: on
`ref-02-facade-pair-and-ground` with one shared seed and 30 steps, the three
profiles produced 3005, 362 and 1985 material voxels respectively. Legality was
perfect in all three and no profile connected the entrances. E0 is what turns
observations like this into a record, per scene, with full provenance.

## rollout_v2 correction - 2026-09-23

Version v2 applies the profile's `fire_rate` in the scoped model configuration,
passes an explicit torch generator into internal delta masking, and rejects
train/none, train/state_blend and eval/delta_mask combinations. Existing
historical settings preserve their behavior. Tests now compare forward output
against the notebook's model/perception/legality/corridor code and training-loop
prefix, stopping before loss/backward/optimizer operations. The original
notebook is unchanged. Shared-model concurrency remains M4 work.

The earlier open question about a legacy scene set was resolved by D011; see
SCENE_SETS.md. Prior unrecorded orientation numbers are not E0 results.
