# NR3: one trained-model trial

## Current: NR3 single trial completed and reviewed - 2026-09-26

D078 review is complete. Returned GPU run 20260926T073942Z_47ef54dc1d67 completed
256 updates, seed1201, in39.352s controlled wall time,368MiB peak reserved.
All1036 payload hashes and archive SHA256 verified; final checkpoint identity,
checksum and completed cursor verified with weights_only loading. Earlier run
20260926T073835Z_b87bca3faa5d failed GPU admission before any training; preserve both.

Exactly27 VALIDATION examples evaluated, final checkpoint256 only,32 steps,
firing2101, CPU Torch2.8.0+cpu/NumPy2.5.2. No TEST inference, new training,
intermediate checkpoint sweep or formal three-model gate. All27 raw states,
probabilities, binary fields, per-case metrics, source snapshot and helper saved.

Damaged18 medianIoU: model0.912425, unchanged0.874856, closing3 0.972679.
All-nine passes:17/18,6/18,15/18 respectively. Model recovers2286 missing cells,
adds2077 false-positive cells and removes3 surviving cells across these18 inputs.
Intact9: model medianIoU0.905059,8/9 pass;1111 false-positive additions.
All27: model25/27 pass versus unchanged15/27 and closing24/27. Model failures
are access/support. Median absolute requested-volume error119cells versus87
unchanged and6 closing. These are related validation cases, not independent
replications. Better contract pass count does not establish better reconstruction.

Decision D079: close this bounded trial; no more automatic tests/training.
Retain MG7 Studio default; learned model remains exploratory. Next implementation
is an evidence-backed comparison view of damaged input, simple closing, learned
output and target, with false additions/removals and nine-family results visible.
Use saved outputs; do not generate more evaluations. For later learning work,
prioritize preserving intact mass and controlling excess growth within existing
nine families; do not claim a larger grid or a new architecture solves this result.
The mechanism behind excess growth is not established by this one review.

Evidence: Codex outputs/NR3-Single-Trial-Review. Project artifact-copy attempt
was denied by filesystem permissions before copying; use the verified Codex archive. Summary tracked at
experiments/reports/NR3-single-trial-review.json; findings in
docs/next-phase/NR3_SINGLE_TRIAL_FINDINGS.md. No Drive operation, deployment,
push or new GPU job. Same-disk verified copies are not off-device backup.
User may disconnect Colab. Next task: implement the comparison view locally.


## Interpretation

The model restores more missing cells than closing but expands the mass too freely, including when the input is intact. This is evidence of a preservation problem on these examples, not proof of its cause. Training used16 rollout steps and this frozen review uses32; no horizon comparison was performed, so drift from longer rollout is only a hypothesis. Do not change the project concept based on one short seed. Keep the volume brief and nine constraint families. The formal D077 thresholds remain unchanged and were not applied as a three-model gate.
