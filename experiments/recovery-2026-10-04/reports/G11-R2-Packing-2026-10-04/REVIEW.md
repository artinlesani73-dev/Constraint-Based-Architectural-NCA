# G11-R2 scheduling review — 2026-10-04

## Outcome

R2 is rejected: stability falls from37/45 to21/45 and maximum early volume
error rises from4.344 to6.262 percentage points. All45 still pass nine families,
but the complete TRAIN acceptance gates fail. Preserve R1 as the reference.
R2 is an experimental ordering variant, not a trained model or live replacement.

## Diagnosis before choosing the change

Replayed R1's eight unstable TRAIN cases for64 steps with additional observation
only. All eight birth sequences match their saved originals exactly.
They waste 849 voxel-step allowance in total across
321 non-seed steps with unused capacity.
Only 6 of these steps have a newly exposed
frontier offer that would fit; 1 have a
remaining previously eligible offer that fits after the complete admission pass.
79 steps have a fitting but unoffered cube.
These categories can overlap. Reasons are sequential first-failure counts,
not an independent causal decomposition. Unused allowance is not itself proof
that another ordering could use all of it.

This made intra-step packing a reasonable single hypothesis, not an established
fix. All diagnostic traces and the instrumented code are retained.

## The one tested change

R1 visits reserved cubes lexicographically, then learned proposals by score.
R2 keeps the same two priority groups but repeatedly picks the cube requiring
the fewest new voxels, recalculating after each accepted overlap. Original
ordering breaks ties. Reserved cubes still bypass score/firing explicitly;
learned proposals still need score>0.5 and firing. Frozen eligibility, witness,
checkpoint, seed, quota K, total ceiling C, horizons and evaluator are unchanged.

This remains an explicitly global hybrid. It changes execution trajectories,
including later logits, even though weights do not change.

## Complete paired TRAIN results

One fixed45-case pass; no held-out scenes, optimizer update, threshold tuning,
paid run or second scheduling variant. G10 baseline arrays were reused from
the preceding run and checked for exact array equality in all45 cases.
R1 comparisons use its archived metrics. R2 was newly inferred to128, with64
as the recorded prefix; independent horizon replays were not performed.

| Model | Steps | Nine families | Median volume error (pp) | Maximum error (pp) |
|---|---:|---:|---:|---:|
| G10 | 64 | 41/45 | 0.193 | 1.009 |
| G10 | 128 | 41/45 | 0.193 | 0.263 |
| G11-R1 | 64 | 45/45 | 0.239 | 4.344 |
| G11-R1 | 128 | 45/45 | 0.193 | 0.263 |
| G11-R2 | 64 | 45/45 | 0.954 | 6.262 |
| G11-R2 | 128 | 45/45 | 0.193 | 0.263 |

Stability passes: G10 42/45;
R1 37/45; R2 21/45.
R2 maximum growth64–128: 26.48%.
R2 completed witnesses by64: 45/45.
R2 planner-born median fraction of newly occupied voxels at128:
32.8%.
This is recorded admission attribution, not a causal measure of learned ability.

R2 median128-step CPU rollout: 5.805s;
R1 3.535s, excluding witness planning.
These single-run timings are descriptive.

## Interpretation and next step

Smallest-first does not necessarily maximize filled volume per step. Its early
choices change overlap geometry and subsequent neural proposals, so local
small-delta preference can worsen the longer trajectory. Do not mistake the
packing heuristic for an optimal solver.

Reject R2 as the selected schedule and retain R1.
The next design to assess is an explicit cumulative allowance ledger: unused
capacity from earlier steps remains available later, while the total planned
allowance and hard final volume cap stay fixed. Proposed non-seed envelope:
min(C, 27 + (t-1)*K) at step t, with the original single-cube seed phase.
This CHANGES the per-step rule from min(C,current_mass+K); it must be labelled
and tested as a new version, never presented as the unchanged quota.
The proposed envelope was checked arithmetically against all45 saved R1
trajectories: each fits within it, and its64-step ceiling equals C in every case.
This proves neither usable proposals nor success under the changed trajectory.
Next freeze one cumulative-ledger local experiment with R1 ordering restored,
including late-seed and cap boundary checks. Keep64/128 horizons and all
acceptance thresholds. No extra training or fresh reserved cases yet.

Do not consume new reserved scenes until the selected method passes the
complete TRAIN gates. decision.json records this failed experiment.

## Verification and preservation

All5,760 R2 birth/provenance accounts passed monotonicity, legality, per-step
and global ceilings, witness-union reservation and saved-horizon agreement.
Both output horizons were checked for full-cube depth and connectivity.
The G10 checkpoint hash is unchanged. All eight raw geometry comparison sheets
were inspected; passing family labels do not imply stability or habitability.

This archive includes sources, fixed checkpoint, contexts, witnesses, raw
and hybrid observations, provenance, metrics and the complete eight-case
diagnosis. Earlier failed and successful attempts remain intact. The source
renderer is included. Same-disk archives are not off-device backups.
Repository synchronization is pending. No Drive operation, push, publication
or live-model change occurred. Continue from RESUME.json in this directory.
