# G8: one training-exposure experiment

G7's broader data did not pass acceptance. A TRAIN-only comparison found both
models begin growth at step1, while G7 leaves much more per-step allowance unused
through sparse above-threshold proposals. This supports testing optimization
exposure before changing the model, objective, threshold or admission mechanism.
It does not prove that additional training will solve connection failures.

Change only retained training updates from256 to427. Calculation:
ceil(256*45/27)=427, approximately restoring G6's mean visits per example.
Use the exact45 G7 TRAIN payloads and order, fresh seed1201, original optimizer,
61->64->8 model, G6 paced transition, 64 training steps, batch1, float32,
Adam0.001, clip1, alternating seed/teacher stages, firing0.5, threshold0.5,
32cubed grid and0.8m voxels. No warm start or learned checkpoint selection.
Losses and teacher construction unchanged. Same nine constraint families.

Global B=ceil(request*D), C=min(B+8,floor(.4D)), K=max(9,ceil((C-27)/63)).
Do not change inference horizon or quota. Neural/cube-admission source and all
data bytes remain identical to G7. The427 target is exposure-derived, not tuned
from a checkpoint sweep. Compare final427 with the already-frozen G7 final256.

One fresh Tesla T4 job, at most600 controlled seconds,427 updates64 steps.
Estimated controlled time roughly310-320s from G7, not a guarantee. Setup,
upload/export/download/idle are extra. Stop at600s, never extend or auto-retry.
Expected Python3.13.15,Torch2.11.0+cu130,NumPy2.1.3,CUDA13.0,cuDNN92700.
Keep device probes, full recovery checks at2/3, per-update checkpoints, finite
guards and80% reserved-memory guard. Runtime mismatch stops the run.

After return, verify the427 final and every trace. Also compare update256 model,
optimizer, sampler, trace and RNG numerically against the prior G7 update256,
with run identity kept separate. This is a prefix reproducibility check, not
checkpoint selection. Record any mismatch; do not silently assert equivalence.

Frozen review: final427 only, CPUfloat32, firing2101, single scene seed,
64 and128 steps. Separately report33 exposed regression requests (all prior
G1 development/reserved plus G7 reserved), and12 new frozen G8 reserved requests.
All nine families must pass every request at both horizons. Each cohort/horizon
requires median absolute requested-fraction error<=.02 and maximum<=.04;
each case's mass change must be<=5%. Visually review volumes. No tuning,
rerolls, dropped cases, best-checkpoint choice or postprocessing.
Fresh reserved geometry stays local outside the TRAIN package; no labels or
inference have been produced. Synthetic relatives are not external validation.

Keep APPROVED_G8_JOB=False until this exact one-job paid allowance is approved.
No Drive operation, retry, extra seed, publication, push or live promotion.
MG7 remains live; all G6/G7 evidence remains intact.
