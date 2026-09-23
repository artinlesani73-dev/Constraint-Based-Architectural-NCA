# D1 direct-material optimization protocol

Preregistered2026-09-23, before optimizer results. No NCA weight is trained here.
D1 gives each scene its own raw voxel parameters and therefore a much easier,
non-generalizing search space. Objective/geometry meanings remain unchanged.

## Fixed method

Initialize raw material as clamp(seed material +0.15*legal scaffold,0,1), matching
the NCA's weak starting scaffold. Forward material is clamp(raw,0,1)*permitted.
Raw parameters remain unconstrained. This retains exact hard-clamp derivatives,
including dead gradients outside its range; pre-clamp coverage remains the same
explicit hinge. No straight-through gradient, smoothing, changed envelope or
hidden projection back into parameter range. Frozen scene channels stay fixed.

Use the existing two K2 recipes (only sparsity30 versus3 differs), the same nine
families, density-binarization, TV and boundary-cantilever. Adam0.05, betas .9/.999,
eps1e-8, no weight decay/AMSGrad, constant rate, clip norm1. Step size is expressed
in voxel-parameter coordinates and is NOT the same update budget as NCA Adam1e-4.
No random firing, learned rule, sample pool, AMP or scene generalization. Seed0
is recorded for environment/checkpoint consistency, not statistical replication.

W1 is retained as a static procedural control; D1 is never initialized from its
already-solved material. The original and four K2 learned models are matched
scene/objective comparisons using preserved K2 results, not equal-cost methods.
D1 optimizer iterations must never be labeled NCA growth steps.

## Recovery, pilot and cost admission

D1R:ground-pair/mass_3, four logical/ten executed updates across four processes
(whole4,prefix2,resume2,repeated resume2). Compare full checkpoint trees, every
trace value, projected/raw fields and pre-update gradients. Exact source/config/
scene/runtime metadata required. CPU completed-update recovery only.

D1P:legacy008, ground-pair, minimal-smoke x two recipes x8 updates =48 total.
Record initial/final objective terms and binary metrics, initial per-family raw
parameter gradients, every update/checkpoint/field/combined gradient, and timing.
No NCA optimizer run or geometry changes.120-second cap per pilot worker.

Candidate full D1:all17 feasible development scenes x two recipes x32 updates
=1088 local optimizer updates, with original/K2/W1 result controls. No early-stop
or best-looking-iterate selection. Freeze32 steps now; use pilot only for cost
admission, not outcome-based hyperparameter tuning. Estimated full cost is
1.5*(34*max(5,max pilot session setup seconds+3)+34*32*pilot p90 update seconds).
Execute only if that estimate is <=900 seconds. Otherwise retain the profile
and prepare a revised bounded compute plan; do not silently shrink the matrix.
Full run has a hard900-second total cap and120 seconds per case. Timeouts remain
interrupted attempts, with completed boundaries and partial artifacts preserved.

Record fields/gradients/checkpoints at EVERY update. Traces describe the objective
before an update; associated checkpoint/raw/material fields are AFTER it. Initial
and final scored records make this boundary explicit. Initial raw gradient probes
record every family and retained regularizer. Worker recovery can load a saved
checkpoint into a new run/branch with identical metadata; no automatic retry.

## Interpretation

Check all per-family residuals, binary connectivity, budget compliance, legality,
protected ground, geometric support, threshold counts and raw saturation. Compare
both common scoring recipes, never only differently weighted totals. Inference
cost for NCA and per-scene optimization cost for D1 are different quantities.

If D1 recovers reference connectivity, this shows the fixed objectives and weak
scaffold can support recovery with direct per-scene parameters under this optimizer.
It does not prove the NCA architecture cannot learn, isolate a unique failure cause,
or establish architectural quality. K2 used only17 shared-weight training updates.
If D1 fails, retain it: feasibility witnesses do not guarantee useful optimization
gradients. No model is promoted merely because a scalar objective decreases.
No new constraint family, production/default change, paid compute or cloud access.
