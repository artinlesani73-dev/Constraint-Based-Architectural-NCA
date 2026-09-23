# K2 execution protocol

Frozen before the first optimizer update, 2026-09-23. Extends the existing
K2-sensitivity.json proposal without changing its coefficients, scenes or horizons.
All work is local CPU. Original artifacts and production defaults stay unchanged.

## Training and recovery

Four members: mapped_30 and mass_3, each seeds0/1. Each starts from the original
checkpoint, with Adam1e-4 (betas .9/.999, eps1e-8, no weight decay/AMSGrad), norm
clip1, constant learning rate and 16-step hard_preclamp rollouts. No pool or AMP.
17 updates per member, each feasible scene exactly once. NumPy default_rng(seed)
materializes its permutation before outputs; both recipes share that order and
an identically seeded dedicated firing RNG. Full order is checkpoint metadata.

nca/sensitivity.py is the actual loop for both recovery and study. K2R checks
mass_3/seed0 over its first3 actual scheduled updates: uninterrupted3, prefix1,
resume2, repeated resume2 (8 executions). It compares full checkpoint state,
all trace values and material/raw fields. Source/config/runtime/scene/input hashes
must match; training refuses a stale recovery gate. CPU completed-update recovery
only, not CUDA/pools/AMP or abrupt mid-write certification.

Each completed update saves a checkpoint, pre-update material/raw fields, every
objective component, mass ratio and pre-clip norm. The update count indexes the
recorded scene order. The coordinator enforces900 wall-clock seconds per training
worker including initialization/serialization, killing a timed-out worker and
retaining completed boundaries and partial files. A timeout is an interruption,
not a successful partial comparison. Orphan artifacts must be inspected if a kill
occurs during publication; never bypass a failed integrity check.

A linked attempt can import complete recorded boundaries with --resume-run;
imported checkpoints are rehashed and source metadata must still match. No automatic
retry or additional time allowance is implied by interruption. Evaluate only after
all four complete. Raw checkpoint files without their completed training_update
record are retained but not automatically selected as a resume boundary.

## Evaluation and interpretation

4 trained models x17 scenes x2 horizons (16/50) =136 records; original checkpoint
adds34; W1's17 static fields add17, for187 evaluation records total. Each model
case restarts from the same scene/scaffold and firing seed2. W1 has no rollout
horizon/RNG; it is displayed as a static procedural control. Its raw field is the
binary material itself for coverage scoring. Existing17 scenes are development
data; no holdout generalization or architectural-quality claim is permitted.

Save continuous material/raw fields, nine-family residuals, three regularizers,
material ratio, binary metrics and totals under BOTH coefficient recipes. Never
compare only each model's differently weighted total. Report connected/scorable
counts separately, illegal/blocked material, over/under-budget counts, coverage,
access/support and all other residuals. Show all scenes and both training seeds.
Original checkpoint and W1 remain baselines, regardless of which arm looks better.
Each evaluation worker has a separate900-second wall-clock safety cap. W1 loading
time is not construction performance; its historical construction stays in W1.

This tiny17-update pass tests coefficient sensitivity, not training convergence.
A lower objective without maintained connectivity is not success. If neither arm
beats the original model on useful geometry, record it and investigate before
longer/paid training or architecture changes.
