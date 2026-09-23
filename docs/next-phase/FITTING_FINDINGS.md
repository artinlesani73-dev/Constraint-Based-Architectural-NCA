# F1 findings: repeated training helps, but the baseline is not ready to scale

2026-09-23. Full study `20260923T112646Z_dcd0601d0655`, source `1c61b33`.
Protocol: [FITTING_PROTOCOL.md](FITTING_PROTOCOL.md). Complete tables:
[F1-fitting.md](../../experiments/reports/F1-fitting.md). Every evaluation,
including all nine families, three regularizers and binary diagnostics, is in
[F1-evaluations.json](../../experiments/reports/F1-evaluations.json).

The unchanged NCA can learn connections on these two difficult development
scenes when trained repeatedly on each scene. It has not achieved the joint
connectivity/material-budget target. A source-definition mismatch between
the access loss and geometry evaluator now needs attention before interpreting
access failures as evidence against the architecture.

## What ran

Four independent models: ground-pair and minimal-smoke, each with sparsity
weight30 or3. Every model starts at original Model C, uses the unchanged K2
optimizer step, and receives64 updates on its own scene. Training uses16 growth
steps, Adam0.0001, seed0 and weak0.15 scaffold initialization. There is no solved
field initialization, added objective, architecture change or scene pool.

Evaluations use separate firing seed2 at16 and50 growth steps after updates
0,1,3,8,16,32,64. That is256 updates and56 evaluation records. The full study
completed in860.27 seconds (14.34 minutes), below the1800s cap; every member
finished below600s. No experiment failed or timed out.

F1R `20260923T112316Z_f34a421f8302` passed11 exact restart comparisons across
eight executed updates and14 evaluations. F1P `20260923T112500Z_8e1a30c320b8`
completed eight updates/24 evaluations in82.25s. Its timing-only estimate of
1546.47s admitted the full study. Pilot outcomes did not change the protocol.

## Final outcomes after64 updates

Material percentage is continuous mass divided by the permitted envelope size.
The accepted range remains3%-12%. Binary connectivity uses material>0.5.

| Scene | Sparsity weight | Connected at16 growth steps | Mass at16 | Connected at50 | Mass at50 |
|---|---:|---|---:|---|---:|
| Ground-pair | 30 | No | 12.99% | No | 20.11% |
| Minimal-smoke | 30 | No | 13.14% | Yes | 25.47% |
| Ground-pair | 3 | No | 18.64% | Yes | 28.72% |
| Minimal-smoke | 3 | No | 18.09% | Yes | 30.60% |

At50 steps, the original checkpoint and all K2 models failed on both scenes;
F1 connects three of four final cases. These connections appear only at the
last recorded training boundary (64 updates) and only at50 growth steps.
None of the56 evaluations is both connected and within budget. Several early
checkpoints are in budget but disconnected. All eight final fields have zero
illegal, protected-ground-blocking and geometrically unsupported binary voxels;
this is not a structural-safety certificate.

Guide coverage residuals improve in all eight final comparisons with the original
checkpoint at the same growth horizon. Lower sparsity weight improves coverage
more, while allowing more material. It does not uniformly improve a common
weighted score: for example, minimal-smoke at16 steps is worse under weight30
scoring after mass_3 fitting (43.59 versus41.05 originally), despite its lower
own-recipe score. Do not compare totals that use different coefficient sets.

D1 remains an unequal-effort comparator: its weight30 direct optimizer connected
both scenes within budget, while W1 supplies static zero-nine-family witnesses.
Direct fitting uses independent voxel parameters and a different learning rate;
this does not establish an NCA capacity bound or a matched-cost ranking.

## A concrete access-contract mismatch

The differentiable loss starts propagation from **one fixed legal voxel** in the
first entrance. The binary evaluator starts from the **occupied entrance region**,
after rejecting fragmented source-region occupancy. Both definitions were
explicit in the code, but F1 now demonstrates a consequential disagreement.

All56 evaluations have zero material at the fixed source, so the soft access
loss remains exactly1 throughout. Three final fields are nevertheless connected
under the region-based binary evaluator. For minimal-smoke the fixed source is
`(z,y,x)=(0,15,11)`; for ground-pair it is `(0,14,21)`. In each connected example,
four other source-region voxels are occupied and the region's maximum material
value is1. This explains those metric disagreements directly: the empty fixed
source transmits no soft-reach strength.

This is a stricter point-source requirement than the reported region connectivity,
not proof that the evaluator's connections are imaginary. It also does not prove
that this mismatch alone caused the training failures. Raw source values are
negative in28 evaluations and exactly zero in28; actual parameter-gradient
attribution remains to be measured. Do not infer blocked gradients from a zero
projected field alone, especially at a clamp boundary or an unfired cell.

The post-hoc descriptive audit is preserved in
[F1-outcomes.json](../../experiments/reports/F1-outcomes.json), including all56
source measurements, field hashes, source coordinates, learning boundaries and
parameter movement. No training configuration was changed after seeing this issue.

## What this means for the next phase

Repeated exposure changes the result, so the earlier17-update mixed-scene test
was insufficient evidence for replacing the NCA concept. Conversely, one seed,
two familiar scenes and64 updates do not establish convergence or robust fitting.
The horizon dependence and excess mass still prevent promotion of any checkpoint.

All256 gradients exceeded the existing clipping threshold. Final parameter
movement was4.40%-5.53% of initial parameter L2 norm. These are descriptive
measurements, not proof that clipping or learning rate is wrong; Adam dynamics
and per-family gradient interactions need separate investigation.

Next follow [ACCESS_ALIGNMENT_PLAN.md](ACCESS_ALIGNMENT_PLAN.md): reconcile
the point-versus-region entrance contract, trace access/coverage gradients, and
replay candidate semantics on saved evidence before a one-factor learning test.
Preserve the single-source safeguard; simply allowing many disconnected sources
could create false connectivity. Keep the material budget and all nine families.
Conditioning/perception changes, fresh holdouts and scaling remain later steps.

## Verification and preservation

- Regression `20260923T112028Z_f4b86bdcf231`:151 tests passed, no failures,
  errors or skips; original-checkpoint smoke passed.
- All256 training checkpoints checked; all312 saved training/evaluation fields
  rescored; all56 binary evaluations and both weighted totals verified.
- All eight initial evaluation fields match the saved original K2 baseline
  exactly. All eight final evaluations replay exactly from the four final
  checkpoints, including raw/material arrays and reported metrics.
- 28 training-source hashes match the registered snapshot. F1R/F1P reports and
  earlier experiments remain immutable. The stronger initial-baseline comparison
  was added to the report verifier after their publication and passed for F1.
- Recovery certifies ordinary completed CPU updates, not CUDA, AMP, abrupt
  writes or automatic whole-matrix continuation. Restore exact source snapshots
  if Git line-ending normalization changes checkpoint metadata hashes.

No production default, original checkpoint, deployment, paid job or Drive file
was changed. The earlier local viewer remains a D1/K2 viewer; F1 is documented
in these reports and has not been silently inserted into that artifact.
