# A3 findings: a partially recoverable access signal

2026-09-23. The pre-clamp maximin loss restores an actual model-parameter gradient
in six of eight failed F3 states. Small access-directed perturbations reduce its
loss in all six, but none repairs binary connectivity. Two states still have no
access gradient through the model. This supports one controlled learning
comparison; it does not justify adopting the candidate or enlarging training.

## Evidence and scope

Source commit `08a6d85`. A3P `20260923T193037Z_0e72e1207fb9`: 25 frozen fields,
two actual gradient cases, 111.09 seconds. Timing-only estimate 1038.85 seconds
admitted the 1500-second study cap. A3 `20260923T193403Z_ac42a8cca390`: 231 fields,
eight actual gradient cases, 336.74 seconds. Every worker and total cap met.
No optimizer updates. Both runs preserve exact scientific source snapshots.

All 231 fields were rescored and their projection identity and independent
binary BFS checks passed. Eight traced forwards exactly match saved F3 outputs.
Verification checks 35 diagnostic source hashes, 48 parameter vectors, 48
last-raw derivative arrays, earlier raw-gradient histories and 12 perturbed
forward fields. Norms/cosines are recomputed from recorded vectors; this is not
independent differentiation of every vector. Initial foundation regression:
180 tests passed, no failures/errors/skips, original-checkpoint smoke passed.
All 37 historical F3 source files remain unchanged.

These are development-scene observations, with repeated horizons and firing
seeds. They are not 231 independent scenes, multiple training seeds or holdout
evidence. The model gradients cover two scenes, two recipes and two horizons
with firing seed 2. No architecture, grid size, budget or constraint family
changed. Existing production and training defaults remain unchanged.

## What the proposed loss changes

The candidate evaluates `relu(1 - b_raw)`, where `b_raw` is the maximum threshold
at which one permitted six-connected component touches every entrance region.
For legally connectable graphs, `b_projected = clamp(b_raw, 0, 1)`. Therefore
the extension exceeds the old saturated loss of one only for negative raw
bottlenecks. Binary connectivity continues to use projected occupancy >0.5.

| Saved source | Fields | Connected | Negative raw bottleneck | Zero raw bottleneck | Candidate loss range |
|---|---:|---:|---:|---:|---:|
| Original checkpoint | 36 | 0 | 23 | 13 | 1â€“1.076005 |
| F2 constant16 | 72 | 59 | 0 | 0 | 0â€“0.807150 |
| F3 mixed16/50 | 72 | 0 | 43 | 29 | 1â€“1.062337 |
| W1 feasible witnesses | 17 | 17 | 0 | 0 | 0 |
| D1 direct voxel solutions | 34 | 34 | 0 | 0 | 0â€“0.027923 |

W1 material is explicitly used as a raw-field control; it is not a network
pre-clamp output. D1 raw fields are direct voxel parameters. All positive-strength
F2/W1/D1 controls retain their access loss value. W1 and D1 are feasibility
controls, not demonstrations that an NCA learned their geometry.

The earlier F3 inspection counted 51 negative/21 zero values at the *projected*
critical voxel. A3 finds 43 negative/29 zero *raw maximin bottlenecks*. These are
different measurements: clipping merges negative values into ties and changes
the selected critical voxel in 68/72 F3 fields. This audit verifies critical
voxels and strengths; it does not extract or compare complete routes.

## Actual learning signals

Every old projected-access parameter gradient is zero in this eight-case matrix.
The raw candidate gives the following unweighted parameter L2 norms:

| Recipe / scene | Steps | Old norm | Raw candidate norm | Candidate descent loss change |
|---|---:|---:|---:|---:|
| mapped_30 / ground pair | 16 | 0 | 0 | No nonzero direction |
| mapped_30 / ground pair | 50 | 0 | 0.642812 | âˆ’0.00006270 |
| mapped_30 / minimal | 16 | 0 | 0.479254 | âˆ’0.00004792 |
| mapped_30 / minimal | 50 | 0 | 0.496186 | âˆ’0.00004971 |
| mass_3 / ground pair | 16 | 0 | 0 | No nonzero direction |
| mass_3 / ground pair | 50 | 0 | 0.409997 | âˆ’0.00004125 |
| mass_3 / minimal | 16 | 0 | 0.483607 | âˆ’0.00004840 |
| mass_3 / minimal | 50 | 0 | 0.495342 | âˆ’0.00004947 |

Each probe moves parameters by L2 length 0.0001 along the normalized candidate
gradient, with the same firing randomness. Both signs are saved; original weights
are restored exactly afterward. Access improves in all six descent probes, but
their projected access losses stay at one: these are small local derivative
checks, not newly connected outputs or training improvements.

The two zero cases explain why a final pre-clamp loss is only a partial repair.
For mapped_30 ground-pair at16, the selected raw critical voxel (z,y,x)=(1,15,9)
is negative at14 and does not fire at15 or16. Its gradient passes through raw15/16
and stops at the earlier clamp; no update-network output receives an access
gradient. For mass_3 ground-pair at16, voxel (0,15,21) is negative at15 and does
not fire at16; the same obstruction occurs one step earlier. Indices are zero
based. This traces the selected branch, not every hypothetical alternate route.

Coverage retains a nonzero parameter signal in both zero-access cases. Therefore
zero access gradient is not proof that the complete model cannot learn. Nor
does restoration in six cases prove it will learn a feasible solution.

## Material tradeoffs remain unresolved

At50 steps, raw-access and sparsity parameter gradients oppose each other in all
four cases: cosine values range from âˆ’0.6041 to âˆ’0.3280. Weighted access norm is
only 0.22%â€“27.48% of the combined other-term norm across these four cases. It
does not dominate the full objective here; in some cases material pressure is
much larger. A norm ratio alone does not specify the optimization direction.

The six access-descent probes reduce total candidate loss in three cases and
increase it in three. The increases are mapped_30 ground-pair at50 (+0.01492),
mapped_30 minimal at50 (+0.18331), and mass_3 minimal at50 (+0.00463). The other
50-step probe, mass_3 ground-pair, slightly improves total loss (âˆ’0.000114).
The two nonzero16-step probes improve it. Do not reweight objectives or widen
the material allowance based on this tiny local test.

## Decision and next experiment

Keep the candidate opt-in. Prepare F4 as a single access-family change against
constant16 F2, with identical original initialization, two scenes, recipes,
64-update exposure and optimizer settings. First require actual-loop baseline
parity, early and trained-state recovery, and a timing pilot. Evaluate the same
six horizons and three firing seeds; count joint connectivity/material-budget
successes as the primary geometric outcome. See RAW_ACCESS_TRAINING_PLAN.md.

Do not combine this with F3 horizons, new weights, a state pool, larger grids or
a conditioning redesign. A learning comparison is justified by semantic checks,
actual restored gradients and directional probes together. It remains possible
that the unchanged coverage bootstrap and candidate access are insufficient.
If so, preserve the negative result and choose one further intervention from
the traced failure. Production UI work remains on the roadmap; these findings
do not certify a model for serving.

## Preserved analysis and robustness details

Reports: `experiments/reports/A3-20260923T193403Z_ac42a8cca390-` followed by
`verification.json`, `evidence.json`, or `outcomes.json`; corresponding A3P
reports retain pilot results. All raw fields, probes, gradients, checkpoints
referenced from prior runs and source ZIPs remain under `.local-artifacts/runs`.

A post-hoc gradient-delta consistency script initially used a tolerance scaled
only to the small access difference. It stopped on residual0.0003354 when the
weighted access delta norm was7.4428, ignoring subtraction of total gradients
around3400. The failed script/receipt is preserved. The final summary reports
the residual and a float32 scale-aware tolerance as a post-hoc diagnostic, not
a preregistered acceptance gate; all eight lie within it. An attempted receipt
write from the Codex directory was access-denied and succeeded from the project
directory. No scientific run, threshold or outcome was altered by this analysis.

After the completed audit, an extreme finite-value check found that the
infeasible-graph zero anchor `raw.sum()*0` could overflow. It was replaced by
`raw.flatten()[0]*0`, with a regression test. This affects only the infeasible
fallback; every audited field has a feasible legal graph. The original audit
snapshot is retained and its source identity must be used for exact repetition.
Follow-up regression20260923T194258Z_d2f3798c93b4 passed181 tests with zero
failures/errors/skips and original-checkpoint smoke0. The final local archive
is recorded in RESUME.md and the milestone receipt.
No Drive operation, paid computation, publication or remote push occurred.
