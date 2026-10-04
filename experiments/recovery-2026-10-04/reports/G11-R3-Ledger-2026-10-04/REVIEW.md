# G11-R3 cumulative allowance review — 2026-10-04

R3 passes all frozen TRAIN gates; ready for independent evaluation.

## One changed rule

R3 starts from R1, restoring witness-first lexicographic ordering followed by
learned score ordering. R2's smallest-first heuristic is discarded.

Previously each non-seed step allowed total mass up to min(C,current_mass+K);
any unused allowance disappeared. R3 uses min(C,27+(t-1)*K) at step t, where
K=max(9,ceil((C-27)/63)). C and K are unchanged, as are the64/128 evaluation
horizons and acceptance thresholds. At step64 the cumulative ceiling is C.

This explicitly relaxes the instantaneous per-step limit: a later step may add
more than K by using earlier unspent allowance. It neither raises the final cap
nor grants more than the original ideal cumulative allowance. The seed still
admits one full cube only. A cap is a maximum, not a guarantee of growth.

No objective, weights, training distribution or evaluator changed.
This is an inference architecture experiment with fixed G10 final427 weights.
The method remains a hybrid: reserved witness cubes bypass learned score/firing;
other admitted cubes require score>0.5 and Bernoulli0.5 firing.

## Frozen TRAIN comparison

All45 cases are existing TRAIN examples. No new held-out data or optimizer
updates were used. Same deterministic CPU float32 execution, two threads,
firing seed2101. Each rollout runs to128 and reports its64 prefix; these are
not independently replayed horizons. Original G10 output arrays were reused
and checked exactly, including contexts and terminal state.
The R3 witnesses are array-identical to R1, so this comparison isolates the
admission schedule, not a new planner.

| Hybrid version | Steps | Nine families | Median volume error (pp) | Max error (pp) | Stable64–128 |
|---|---:|---:|---:|---:|---:|
| G11-R1 | 64 | 45/45 | 0.239 | 4.344 | 37/45 |
| G11-R1 | 128 | 45/45 | 0.193 | 0.263 | 37/45 |
| G11-R2 | 64 | 45/45 | 0.954 | 6.262 | 21/45 |
| G11-R2 | 128 | 45/45 | 0.193 | 0.263 | 21/45 |
| G11-R3 | 64 | 45/45 | 0.193 | 0.263 | 45/45 |
| G11-R3 | 128 | 45/45 | 0.193 | 0.263 | 45/45 |

Raw G10 on these same45 cases:41/45 nine-family passes at both horizons and
42/45 stability passes. Its earlier69-case results refer to a different set.

R3 maximum growth64–128: 0.000%.
Witnesses fully present by64: 45/45.
R3 median128-step CPU rollout time: 3.668s,
excluding planning. Single local timings do not establish production latency.

Planner-born fraction of added voxels at128:
23.0% minimum,
33.1% median,
51.9% maximum.
Seed excluded. This records the admission path, not causal attribution.
Neither planner-enforced connection/coverage nor hard-cap stability should be
described as something the NCA independently learned.

## Verification

All5,760 hybrid birth accounts pass monotonicity, legality, the NEW cumulative
ceiling, global cap, witness-union reservation and saved-output equality.
Both output horizons pass connectivity/full-cube invariants; all90 saved hybrid
states are finite. Fixed checkpoint hash is unchanged. All45 cached raw G10
cases and witness/context arrays match the prior run exactly.
Pre-inference arithmetic checks covered tiny/saturated caps and the late-seed
single-cube premise. They are not a complete delayed-seed neural replay.

All eight geometry comparison sheets were visually inspected: raw voxel
surfaces, no smoothing or filling, with context wireframes. They show building
massing, not designed interiors or structural certification.
The45 correlated requests and one firing seed are insufficient for a
generalization or diversity claim.

## Decision and next step

Freeze R3 source, checkpoint and protocol. Create one genuinely new geometry set with physical-context disjointness against all prior TRAIN and exposed cases. Evaluate paired G10/R3 once on that new set and all69 old regression cases. Include certificate failures; no tuning or paid training. Only then consider a labelled hybrid preview.

MG7 remains live. A TRAIN pass is preparation for independent assessment,
not deployment acceptance. Keep original NCA outputs distinguishable from
hybrid outputs in any future interface.

## Preservation

All source, protocol, boundary checks, checkpoint, contexts, witnesses, per-case
metrics, birth provenance, states, figures and aggregate decisions are archived.
Previous failures remain intact. Read RESUME.json here next.
Repository synchronization remains pending; original reports are untouched.
No paid run, Drive operation, push, publication or live-model change occurred.
The hash-verified archive is on the same disk, not an off-device backup.
