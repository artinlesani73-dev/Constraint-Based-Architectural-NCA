# Direct-voxel findings: the reference connections are recoverable

Completed 2026-09-23. Direct per-scene optimization connected all17 feasible
scenes with either coefficient recipe, starting from the same weak scaffold as
the NCA. This includes all five reference cases that remained disconnected in K2.
The stronger material-budget weight kept all five reference cases inside the
budget, but failed the minimum-mass floor on five legacy cases. No model or
coefficient is promoted, and architectural quality remains unproven.

## What changed in this comparison

D1 adjusts raw voxel values independently for each scene. It uses no NCA update
network and learns no shared rule. The nine constraint families, retained three
regularizers, legal projection, radius-six envelope, 3%-12% material budget and
facade allowance are unchanged. Forward material is clamp(raw,0,1)*permitted;
raw parameters remain unconstrained and clamp gradients are exact.

Both existing recipes were tested: sparsity30 and3. Initialization is the NCA's
weak0.15 legal scaffold, not the already-solved W1 geometry. Adam0.05 and32 steps
were frozen before results. This step size is in voxel coordinates, not comparable
to the NCA's weight learning rate. Each scene has thousands of independent degrees
of freedom, versus K2's shared network and only17 training updates. This is a useful
control, not an equal-budget architecture contest or a generalizing predictor.

## Recorded execution and checks

| Stage | Run | Scope | Outcome |
|---|---|---|---|
| Regression | 20260923T104700Z_c4dc9a0d6f64 |147 tests and checkpoint smoke |No failures/errors/skips; smoke0 |
| D1R recovery | 20260923T105015Z_484cf36351d7 |Ground-pair; four logical/ten executed updates in four processes |Seven exact checkpoint/trace/field comparisons pass |
| D1P pilot | 20260923T105120Z_4d1e4d2d20d6 |Three scenes x two recipes x8 updates |48 updates; six cases verified;39.84s |
| D1 full | 20260923T105246Z_ab0d4a430b4c |17 scenes x two recipes x32 updates |1088 updates;34 cases;484.44s; no failed/timed-out case |

Source042b7c8. The preregistered timing-only admission estimated799.51s using pilot
p90 update0.33365s and5s startup allowance. It fit the900s cap, so the unchanged full
matrix ran. No quality-based case selection or coefficient/step-size retuning.

All1088 full-run checkpoints, update counters, projections and saved gradient
norms were checked. Every initial/final objective and binary metric was recomputed
(68 scored states). Initial per-family gradients were archived and their norms
rechecked. Intermediate objective values were retained but not all independently
recomputed. Recovery verifies ordinary completed-update CPU restart, not abrupt
mid-write failure, GPU/AMP or cloud recovery. All source snapshots and prior evidence
remain available. Reports: ../../experiments/reports/D1-direct.md, D1P-direct.md,
D1R-direct.md and accompanying verification/aggregate receipts.

## Results that matter

| Method | Connected feasible scenes | Connected AND inside3%-12% budget | Mean material/envelope |
|---|---:|---:|---:|
| Direct weight30,32 optimization updates |17/17 |12/17 |5.41% |
| Direct weight3,32 optimization updates |17/17 |10/17 |8.73% |
| Original NCA,50 growth steps |10/17 |0/17 |28.01% |
| K2 NCA weight30,50 growth steps, both seeds |10/17 |0/17 |21.44% |
| K2 NCA weight3,50 growth steps, seeds0/1 |12/17 and11/17 |0/17 |27.40% and26.80% |
| W1 static procedural control |17/17 |17/17 |3.55% |

Connected-and-in-budget is only a limited conjunction. It is not all-nine-family
success, walkability or structural safety. All final D1 fields have zero illegal,
protected-ground-blocking and geometrically unsupported binary voxels. Continuous
support, coverage and facade penalties can still be positive: their graded
strengths/ratios differ from binary connectivity.

The direct initial fields connect0/17 scenes at threshold0.5 for both recipes;
they are weak scaffolds, not binary solved targets. Both final recipes connect all
five feasible references. Weight30 puts all five inside the budget; weight3 exceeds
the cap on wide-gap and asymmetric-heights. Small coverage/support residuals remain
on some of these cases. Existing18 scenes are development data, and the sealed
negative control remains explicitly infeasible and outside feasible optimization.

## Remaining budget failures, kept visible

Weight30 has **no over-budget cases**, but falls below the3% floor on legacy003,
006,009,010 and011 (ratios1.79%-2.62%). Weight3 has one slightly under-budget case,
legacy000 at2.985%, and six over-budget cases: legacy003,006,010,011,wide-gap and
asymmetric-heights (ratios14.87%-16.49%). The declared1e-6 budget tolerance does not
hide these failures. Details: ../../experiments/reports/D1-budget-failures.json.

Mean raw sparsity penalty is0.00215 for weight30 versus0.07624 for weight3. Mean
coverage penalties are0.00102 and0.000548; mean continuous support penalties are
0.04117 and0.01841. Weight30 has the lower mean total under either common scoring
recipe (0.4588 versus2.5611 under weight30;0.4008 versus0.5026 under weight3).
This does not establish a globally optimal coefficient or convergence after32 steps.
Do not erase the minimum-mass failures by silently changing the accepted floor.

## What the result tells us—and what it does not

The fixed geometry, objectives and weak scaffold provide an optimization route
to connected reference outputs when the solver can directly adjust each voxel.
The earlier failures therefore cannot be attributed simply to impossible reference
connections under this contract. This strengthens the case for testing the NCA's
learning process on those same cases before replacing the concept.

It does not identify one unique cause. D1 changes parameterization, optimizer scale,
number/distribution of updates and per-scene freedom. K2 saw each scene once; it did
not establish the NCA's fitting capacity or convergence. The full route guide is
still supplied, so solving these numerical objectives is not evidence of meaningful
architectural generation. W1 remains the strongest simple numerical baseline,
and direct outputs still need semantic/architectural evaluation.

## Next: a bounded NCA fitting diagnostic

D035: retain D1 as a per-scene comparator; do not promote its coefficients, change
production defaults, or start paid training. Next prepare an actual NCA single-scene
fitting test on ground-pair and minimal-smoke, with the same checkpoint, architecture,
objectives, rollout and both coefficient recipes. Repeated exposure is the intended
change. Keep original and D1 outputs as controls; do not initialize from solved D1
fields or add a new reconstruction constraint.

Profile and preregister a bounded schedule (candidate64 updates per scene/recipe,
with explicit timing cap), check actual-loop recovery, and score connectivity,
budget and residuals at recorded boundaries and beyond the training growth horizon.
If it fits, investigate mixed-scene scheduling/recovery and generalization next.
If it cannot fit under the bounded test, inspect gradients and saturation before
isolating a conditioning/perception change. Neither branch proves convergence or
architecture impossibility. Fresh geometry holdouts remain a later frozen stage.
No new fitting experiment has run in this milestone.

## Local evidence viewer

Added assets/experiment_viewer.html and scripts/build_result_viewer.py. The generated
standalone viewer contains17 scenes x13 result variants (221 fields): original and
all K2 models at16/50 growth steps, both D1 outputs and W1. It offers synchronized
3D/plan views, model/scene selection, building/entrance layers, common scoring and
per-component values. Existing user studio concept is unchanged.

All221 binary voxel-coordinate lists and displayed metrics match their registered
source fields; JavaScript syntax passes. Browser security policy rejected opening
the local file URL, and no workaround was attempted. **Visual and interactive browser
checks remain unverified.** This is a saved-evidence preview, not a deployment or
live inference upgrade. The file and QA receipt are preserved under
.local-artifacts/viewers/D1-20260923T105246Z_ab0d4a430b4c/ and experiments/reports/D1-viewer-*.

All changes/results/decisions are committed and locally archived with verified
payload hashes and restore checks. Private reports remain Git-ignored. No paid
compute, cloud/Drive operation, push or deployment occurred. Local same-disk backup
is not off-device protection; RESUME.md supports continuation after a limit reset.
