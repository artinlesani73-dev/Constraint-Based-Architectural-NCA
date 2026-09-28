# NR5 final-model findings

## Current: NR5 reviewed; bounded training sequence closed - 2026-09-28

User supplied completed run20260928T084850Z_cb574ff5e2be and its receipt. Assistant
launched no paid job. No separate approval reply preceded this attachment; record
user-run execution without inventing approval or inferring further compute access.
GPU job completed256 updates in45.767336s,688MiB peak reserved (560260608 allocated).
All1036 payload hashes, outer receipt checksum, final checkpoint checksum, objective,
32-step identity, seed, study manifest, sampler/trace and all Adam steps verified.

Review: exactly27 development/validation rows, final256 only,32steps,firing2101,
CPUfloat32. All raw states/probabilities/binary fields and per-case metrics retained.
NR3/NR4 saved comparison identities and raw hashes verified; no baseline reruns.
No TEST, longer horizon, threshold changes, additional training or checkpoint choice.

Damaged18 medians: overlap NR3 .912425 -> NR4 .943557 -> NR5 .970577;
closing3 .972679. Excess cells2077 -> 1354 -> 325; closing13. Requested-volume
absolute error median117 -> 71 -> 19cells; closing38. Missing cells recovered
2286 -> 2224 -> 1945; surviving input removed3 -> 4 -> 0. Thus higher overlap and
less excess do not mean more missing volume was recovered. All-nine pass counts
17/18 -> 13/18 -> 11/18; closing15/18.
Intact9 medianIoU .905059 -> .945415 -> .986333. Excess1111 -> 736 -> 172.
NR5 intact validity6/9, and not all examples reachIoU .99. Overall NR5 passes17/27
versus NR4 19/27 and NR3 25/27. NR5 family pass counts:access17,support23,thickness25,
all other families27 each. Failure locations have not been diagnosed in NR5;
do not carry forward NR4's disconnected-outlier explanation as an observed NR5 fact.

Frozen criteria: damaged overlap/excess/volume error PASS; intact overlap,
intact validity, damaged validity FAIL. Overall FAIL. Training-horizon alignment
coincides with substantial improvement in reconstruction in this one comparison;
no causal proof that mismatch alone caused prior excess, no stability beyond32
claim, no independent generalization claim. Compute and firing draws also changed.

D084 model-path decision: keep NR5 as best reconstruction research checkpoint among
NR3/NR4/NR5 on these cases, not a reliable admitted model. NR3 still has highest
all-nine pass count; no single winner on every criterion. Keep MG7 live Studio
and retain all learned variants with experimental labels. Close these incremental
trials: no automatic NR6, repeated penalty adjustments, extra seeds or paid runs.

Roadmap steps1-5 complete; step6 decision recorded above. Next implementation batch
can progress scale/diversity and deployment on the existing procedural path with
explicit method labels, while learned repair remains a separate research result.
This does not redefine the NCA research objective or prove scaling the NCA is safe.
A new substantial NCA training proposal needs an explicit rationale and compute
allowance. Do not present a procedural generator as a trained NCA.

Evidence: Codex outputs/NR5-Single-Trial-Review, including review-script.py,source.zip,
sealed-model.json,imports.json,result.json,comparison.json and27 raw observation pairs.
Summary: experiments/reports/NR5-single-trial-review.json; findings:
docs/next-phase/NR5_SINGLE_TRIAL_FINDINGS.md. Verified local archive:
Codex outputs/NCA-NR5-Review-2026-09-28.zip and receipt. No Drive operation/push/
publishing; same-disk copy is not off-device backup. Returned ZIP verified locally;
user can disconnect the Colab runtime. Historical current entries remain below.

## Model comparison

| Measure | NR3 | NR4 | NR5 | Closing3 |
|---|---:|---:|---:|---:|
| Damaged median IoU | 0.912425 | 0.943557 | 0.970577 | 0.972679 |
| Damaged all-nine passes (18) | 17 | 13 | 11 | 15 |
| Damaged excess cells | 2077 | 1354 | 325 | 13 |
| Missing cells recovered | 2286 | 2224 | 1945 | 1464 |
| Intact median IoU | 0.905059 | 0.945415 | 0.986333 | 1.000000 |
| Intact excess cells | 1111 | 736 | 172 | 13 |
| All-nine passes (27) | 25 | 19 | 17 | 24 |
