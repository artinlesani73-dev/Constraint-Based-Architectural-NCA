# CGR2: bulk completion and intact stopping

D093, 2026-09-28. One objective revision, not an architecture change.
Preserve CGR1 v2 code, weights and all evidence. Separate bulk_repair module,
semantic identity bulk_constructive_repair_v1; reject CGR1 checkpoint restores.

Keep CGR1 architecture, initialization seed1201, TRAIN81 distribution, Adam.001,
256 updates,32 steps, stochastic firing0.5, monotonic six-face births, no teacher
at inference, detached hard decisions and no deletion. Same nine families and
building-volume semantics. No rooms or blind filling of exterior gaps.

At each step let P=M+eligible*sigmoid(logits), with eligible restricted to fired
legal empty six-face neighbors. Retain frontier positive weight0.5, damaged
negative weight1 and local absolute3-cube mean-volume loss weight0.25.
Change intact negative weight1.5 to3. Add weight0.5 times mean squared deficit
(1-mean_cube(P))^2 over valid3-cubes wholly occupied in the target and touching
eligible cells. Empty selected sets contribute differentiable zero. Full cubes
are determined by fixed convolution sum==27 on binary targets; no average-pool
backward. Use deterministic convolution. This provides direct gradients toward
completion of target bulk without offsetting deficits with excess outside that
cube. It does not differentiate connectivity, enforce a bulk path, or guarantee
all-nine validity. Target cubes are supervision only. Wrong births remain
irreversible; stronger stopping can reduce correct repairs through shared weights.
Coefficients are fixed design choices, not validated optima. This combined loss
trial cannot attribute improvement separately to its two terms.

Local checks: gradient directions, empty masks, target-free inference, attachment,
exact CPU next-update recovery and semantic restore rejection; TRAIN-only audit
at logits0 checks finite bulk gradients and loss scale. Eight-update packaged CPU
rehearsal is engineering evidence only. No validation/TEST tuning.

Frozen evaluation: final256, CPUfloat32,32steps,firing2101, same27 development rows.
Accepted occupancy only, no cleanup. Retain proposals/births/states. Compare CGR1,
NR5 and closing3. Retain original gates: all9intact IoU>=.99 and valid; damaged
valid>=17/18, medianIoU>=.9705768039313023, excess<=325,recovery>=1945,
median absolute request error<=19; zero surviving input removals. Report any
regression relative to CGR1 (.975039 overlap,245excess,1959recovery,15.5error)
even if original gates pass. No automatic live admission or new TEST evaluation.

Proposed one Colab T4 job: seed1201,256updates,600controlledseconds maximum;
setup/download/idle extra. Separate approval needed after local preparation.
No retry, extra seed, Drive access or push. Successful previous CGR1 GPU execution
does not establish CGR2 GPU compatibility or exact GPU recovery.
