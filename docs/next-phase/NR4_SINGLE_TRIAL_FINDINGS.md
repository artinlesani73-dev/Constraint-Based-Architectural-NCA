# NR4 final-model review

## Current: NR4 completed; preservation criteria NOT met - 2026-09-26

User explicitly approved the single256-update T4/600s job after preparation.
Returned run20260926T083017Z_51b29cc254ff completed in35.109881s,368MiB reserved.
Receipt/archive SHA25622bb26893e7fa4bba2e93f0110bd2c32a737fc6a0b248079439cbd9a4354b8de
and all1036 payload hashes verified. Final256 checksum, objective, study manifest,
seed, GPU device and optimizer-step/sampler cursors match. Receipt first arrived
with the old NR3 ZIP; waited for matching NR4 ZIP, never treated NR3 as NR4.

Final model only:27 validation rows,32steps,firing2101,CPUfloat32. All raw8-channel
states, probabilities, binary fields and per-case metrics saved; no TEST inference,
new training, intermediate selection or threshold search. Reused NR3 observations
and baselines; all27 case identities and NR3 array checksums verified.

Damaged18: NR3 -> NR4 medianIoU0.912425 -> 0.943557; false-positive cells2077 ->
1354; median absolute requested-volume error117 -> 71cells; recovered missing
cells2286 -> 2224; surviving cells removed3 -> 4. All-nine passes17/18 -> 13/18.
Closing3 remains higher overlap0.972679 and15/18pass on these damaged examples.
Intact9: medianIoU0.905059 -> 0.945415; excess1111 -> 736cells; passes8/9 -> 6/9.
None reaches the .99 intact overlap criterion (best0.971941639).
Overall all-nine passes25/27 -> 19/27. NR4 family pass counts:access19,support21,
thickness26; remaining six families27 each. Output legality/spill are projected,
not learned guarantees. Geometric support is not mechanical certification.

Frozen NR4 criteria: intact overlap FAIL, intact validity FAIL, damaged overlap
PASS, damaged validity FAIL, damaged excess PASS, damaged volume error PASS.
All conditions were required; overall FAIL. This is a tradeoff, not a successful
preservation revision or a demonstrated causal explanation of the failures.
One seed on reused development cases gives no independent generalization claim.
Do not compare numerical NR3 and NR4 loss values as if the objectives were equal.

D081: preserve both models as experimental and keep MG7 live Studio default.
No additional GPU runs, retry, loss retuning, grid/architecture expansion or model
promotion authorized. Next useful local work is to inspect saved failing shapes
and show NR3/NR4 differences in Studio, without running another experiment.
Use those saved fields to distinguish disconnected additions from bulk/interface
failures before proposing another training change. No conclusion yet that larger
models, more steps, or stronger penalties would fix this tradeoff.

Evidence: Codex outputs/NR4-Single-Trial-Review; tracked summary
experiments/reports/NR4-single-trial-review.json and NR4_SINGLE_TRIAL_FINDINGS.md.
Review helper and source snapshot are in evidence folder; archive
Codex outputs/NCA-NR4-Review-2026-09-26.zip with verified member-hash manifest.
Keep previous NR3/NR4 preparation and receipt-only records. No Drive operation,
push or deployment. Same-disk archive is not off-device backup. Colab may be
disconnected now that the returned archive has been locally verified and copied.


## Complete comparison

| Measure | NR3 | NR4 | Simple closing |
|---|---:|---:|---:|
| Damaged median overlap | 0.912425 | 0.943557 | 0.972679 |
| Damaged all-nine passes (18) | 17 | 13 | 15 |
| Damaged excess cells | 2077 | 1354 | 13 |
| Intact median overlap | 0.905059 | 0.945415 | 1.000000 |
| Intact all-nine passes (9) | 8 | 6 | 9 |
| Intact excess cells | 1111 | 736 | 13 |
