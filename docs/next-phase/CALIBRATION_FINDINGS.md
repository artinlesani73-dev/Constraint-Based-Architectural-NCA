# Calibration findings and next experiment

Completed 2026-09-23. K1 measures the original checkpoint under the corrected
experimental objective. It exposes a large material-budget gradient and confirms
that access still has a blocked derivative in the ground-pair scene. We have
prepared a controlled coefficient comparison; we have not established better
trained output or selected final weights.

## What was implemented

`regularizers_v1` ports notebook density/binarization and the sum of three-axis
TV faithfully, with value/gradient and batch parity tests. Density means
mean(p*(1-p)), not an upper material limit. The original cantilever surrogate is
retained diagnostically; a separately named boundary-aware variant accounts for
fixed supports and the lower grid boundary. It is a geometric penalty, not a
structural analysis or physical span limit. See REGULARIZER_AUDIT.md for the exact
historical defects, including its skipped bottom three layers.

`research_objective_v1` composes the nine existing corrected families with the
three retained regularizers. It requires complete explicit coefficients, checks
joint context validity, and keeps the historical cantilever out of the total.
The experimental contract remains radius-six envelope, 3%-12% material budget,
pre-clamp coverage, and facade_endpoint_v1. Original notebook, checkpoint and
production defaults remain intact. No new constraint family was added.

## Measured evidence

K1 run `20260923T092355Z_f657f2f3bdb9` (source f56932c): 71 actual model-gradient
cases and 51 controlled budget probes, zero optimizer updates. All 17 feasible
development scenes use firing seeds 0/1 at 4/16 steps. Three named scenes also
use seed 0 at 50 steps; this small long-horizon subset is not the full benchmark.
The sealed reference remains explicitly infeasible and excluded from calibration.

| Finding | Evidence | Implication |
|---|---|---|
| Material budget becomes dominant | At 16 steps, median unweighted parameter-gradient norm is 12.7824 for sparsity versus 1.4032 for coverage; sparsity is active in 33/34 cases | Historical numeric coefficients cannot be assumed calibrated for corrected formulas |
| Very short rollouts miss this conflict | Sparsity gradients are zero in all 34 four-step cases | Use 16-step training rollouts for the proposed comparison |
| Ground-pair access remains blocked | Access loss is 1 with zero parameter gradient in both seeds at 4/16 steps and the single 50-step case; coverage gradients remain nonzero | Pre-clamp coverage supplies a signal but does not repair the access derivative itself |
| Thickness is inactive throughout K1 | Zero parameter gradients in all 71 model cases | Do not invent inverse-zero weights; retain explicit thick-field controls from earlier audits |
| Long rollouts can worsen excess material | In the three 50-step examples, median sparsity norm is 202.234 | Evaluate beyond the training horizon; do not extrapolate this small sample to all scenes |
| Budget derivative follows the intended branches | All 17 scenes at occupancy .015/.075/.20 have the expected negative/zero/positive sparsity derivative | The measured opposition is not explained by a reversed budget derivative |

Hard legality and ground have zero parameter gradients because their projection
is enforced. Pre-clamp coverage has its intended zero derivative at raw material
>=1; it does not prevent all overshoot or saturation. Zero gradients must be
read with raw values, state and geometric outcomes.

The verifier rechecks every saved parameter-vector norm/cosine, budget derivative
and raw coverage derivative, and recomputes the composed objective on all 71
saved fields. Full per-scene tables: ../../experiments/reports/K1-calibration.md.
The diagnostic took 1631.61 seconds; one four-step case had an unexplained
410.99-second outlier. These timings are not a clean performance benchmark.
A scalar logging warning was harmless; subsequent logging detaches the tensor.
The exact original run source is retained in its registered snapshot.

## Fixed comparison prepared: K2, not yet executed

Two recipes differ only in sparsity weight: mapped_30 uses 30 and mass_3 uses 3.
Other numeric coefficients match the checkpoint table, mapped to the corrected
terms, with density 3, TV 1 and boundary cantilever 5. This is not historical
training parity. Neither recipe is a validated final setting.

At 16 steps, the negative combined raw gradient locally points toward improving
coverage in 2/34 cases for mapped_30 and 18/34 for mass_3. Sparsity-improving cases
change from 31/34 to 18/34. These are infinitesimal directional estimates, not
Adam update predictions or observed learning. They justify an isolated test.

The frozen proposal is experiments/configs/K2-sensitivity.json: two recipes,
two training seeds, 17 updates each with all feasible scenes once in recorded
order (68 logical updates), 16-step rollouts, Adam 1e-4, norm clipping 1 and a
constant scheduler. Evaluate the 17 development scenes at 16/50 steps with
firing seed 2, plus no-update checkpoint and W1 procedural controls. Checkpoint
every update; each run has a 900-second CPU cap. See SENSITIVITY_PLAN.md.

The K2 coordinator/trainer still needs implementation and a restart test of its
actual loop, including scene position, firing RNG, coefficients and constant
scheduler. R2 below verifies the composed objective with the earlier miniature
recovery harness, not this future K2 loop. Any timing cap interruption is retained;
never relabel a partial run as the completed 68-update comparison.

## Recovery and validation

R2 `20260923T095240Z_652e01d22fee` (source 0fca1b8) completed four logical updates,
ten executions across four fresh CPU processes. All seven comparisons pass
exactly: prefix checkpoint, full checkpoint trees, traces and generated fields
for resumed and repeated-resume branches. All three scheduled scenes were used.
Independent artifact verification repeats those comparisons. It tests mass_3
with nine families and three regularizers, not model quality. Completed-update
CPU recovery does not certify CUDA, AMP, sample pools or abrupt mid-write failure.

Regression `20260923T092752Z_fa771e070c3d`: 141 tests, zero failures/errors/skips,
checkpoint smoke exit 0. Earlier 137-test regularizer verification is also kept.
Reports and verification receipts are in experiments/reports/K1-* and R2-*.
All raw inputs, parameter gradients, fields, source snapshots, checkpoints and
worker logs remain in .local-artifacts/runs/<run_id>/.

Exact source-byte snapshots matter: a Windows Git checkout can change Python
line endings and therefore code hashes. Restore the registered source snapshot
into a new recovery workspace when exact checkpoint source hashes are required;
Git commit identity alone is insufficient. Backup verification checks those
snapshot bytes and the frozen K2 config as well as scenes and annotations.

## What follows

Implement and verify the local K2 loop, then execute and compare both recipes
without selecting examples. Continue with matched direct-optimization/NCA
controls and fresh geometry holdouts before architectural or scaling claims.
The existing scenes are development data. The studio redesign remains planned;
no interface or deployment improvement is claimed by this milestone.

No paid compute, remote push, deployment or Drive operation occurred. Work is
committed and locally archived; RESUME.md gives exact continuation instructions.
Local same-disk archives are not an off-device backup. No user setup is needed
for the next local experiment.
