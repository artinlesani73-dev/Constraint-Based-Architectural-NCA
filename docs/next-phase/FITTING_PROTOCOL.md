# F1: repeated single-scene NCA fitting

Preregistered 2026-09-23, before F1 execution. Implements D035/D036.

Question: can the unchanged NCA fit either difficult reference when it gets
repeated optimizer updates on that scene? This is a capacity/optimization
diagnostic on development data, not a generalization test or final recipe choice.

## Frozen design

- Four independent models: ground-pair and minimal-smoke, each with mapped_30
  and mass_3. Training seed 0. Start each from original Model C checkpoint.
- 64 updates per model. Every update resets to the same scene seed plus 0.15
  legal scaffold. No solved D1/W1 field, state pool or reconstruction objective.
- Exactly the K2 optimizer step, inherited without alteration: 16 recurrent
  growth steps, nine corrected families and three retained regularizers,
  hard_preclamp coverage, radius-six envelope, facade_endpoint_v1.
- Adam 0.0001, betas (0.9,0.999), eps 1e-8, no weight decay/AMSGrad/foreach/fused,
  gradient norm clipping 1, constant scheduler. CPU, two threads, deterministic.
- Evaluation after updates 0,1,3,8,16,32,64 at 16 and 50 growth steps, fresh
  explicit firing seed 2. Evaluation never consumes the training firing RNG.
  Preserve every boundary, including regressions; do not select only the best.
- Every update stores pre-update fields/terms, gradient norm, complete post-update
  checkpoint and timing. Evaluations are post-update and name their checkpoint.
  Store raw saturation counts as diagnostics, not proof of gradient causation.
- Compare saved original/K2/D1/W1 controls on these same two scenes. These have
  unequal optimization freedom and effort; present methods and cost explicitly.

## Recovery and cost gates

1. Full regression and checkpoint smoke pass before execution.
2. F1R: ground-pair/mass_3, whole3 versus prefix1+resumed2, repeated twice in
   fresh processes. Require exact traces, full checkpoint trees and saved fields,
   including evaluations at 0,1,3. Eight executed optimizer updates. Each worker
   capped at120 seconds. This certifies ordinary completed CPU update boundaries,
   not abrupt writes, CUDA, AMP or a full-matrix continuation coordinator.
3. F1P: four members x2 updates with evaluation at0,1,2. Pilot is a separate run;
   study restarts from original checkpoint. Each pilot worker capped at120s.
4. Timing-only gate: per-member estimate =1.5*(max(5,max setup+3)
   +64*p90(update seconds)+7*max(evaluation pair seconds)). Four times that is
   the total estimate. Admit only if member<=600s AND total<=1800s. Quality
   results never change the admission, update budget, scenes or coefficients.
5. F1 study: four x64=256 updates,56 evaluations. Enforce600s per process and
   1800s total measured from coordinator start. Stop and retain evidence on
   timeout; never automatically lower targets or extend caps.

Configuration: experiments/configs/F1-fitting.json. New code and snapshots are
hashed in checkpoints. Adding nca/fitting.py changes prior runners' broad source
hash inventories: use their exact source snapshots for old checkpoint recovery.
Worker --resume is usable only in a new run/branch with identical metadata and
verified completed records. Never overwrite finalized runs or bypass source checks.

## Interpretation fixed before outcomes

Report connectivity, continuous mass versus3%-12% budget, all nine terms/three
regularizers, legality/ground/support binary diagnostics and both common weighted
totals at every boundary. Connected+in-budget alone is not architectural success.
One seed and two scenes cannot prove architecture failure or robust fitting.
If fitting improves, investigate mixed-scene scheduling and later fresh holdouts.
If it fails, inspect raw saturation and parameter gradients before choosing one
conditioning/perception intervention. No new constraint family, production change,
paid compute, Drive access, push or deployment is authorized by this protocol.
