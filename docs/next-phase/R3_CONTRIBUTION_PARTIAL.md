# R3 learned-selection contribution — PARTIAL results, October 8, 2026

Status: interrupted by CPU memory allocation failure, NOT a completed comparison.
15 of24 planned method/request outputs evaluated. No automatic retry or scientific tuning.
Frozen protocol: experiments/reports/R3-contribution-protocol-2026-10-08.json.
Full numerical record: experiments/reports/R3-contribution-results-2026-10-08.json.

## Design

Four already-exposed development sites; two firing seeds2102/2103,24% request,128steps,
original route. Raw G10, learned R3, and a neutral constant-score R3 baseline. Neutral
retains R3's planner, witness, firing draws, tie-breaking and admission rules; substitutes
occupancy logit+1 and zero hidden updates. Thus all eligible fired proposals exceed0.5,
with ties resolved by flattened index. It removes learned filtering AND ranking; it is
not a ranking-only ablation, witness-only output, MG7, or a general procedural optimizer.
Neural-looking provenance labels in the inherited trace mean policy-admitted additions
for this baseline, not neural decisions. No model training or production substitution.

## What completed

| Method | Accepted / evaluated requests | Missing planned requests |
|---|---:|---:|
| Raw G10 | 4/5 | 3 |
| Learned R3 | 5/5 | 3 |
| Neutral scores + R3 planner | 5/5 | 3 |

Acceptance requires all nine families at64 AND128, volume error<=4percentage points at
both, and64-to128 growth<=5%. Do not convert this table into full planned8-request scores.
The reference and courtyard have both seeds; the offset-obstacle has seed2102 only.
The blocked site was not evaluated in this study. Its prior studio test is separate evidence.
No missing outputs are counted as successes or silently dropped from the planned denominator.

Across the TWO sites with both seeds, mean128-step Jaccard distance is G10=0.23754,
R3=0.19127, neutral=0.03172. Per-site R3/neutral distances: reference0.23170/0.01854;
courtyard0.15084/0.04489. These are two correlated seed-pair comparisons, not a population
estimate. Larger distance measures occupancy variation; it does not imply useful diversity.
For matching site/seed, R3 versus neutral distances range0.50966–0.59869 (five pairs).

Both final-view plates were inspected. Neutral outputs appear more slab-like and spread
farther in Y; their occupied Z span is7cells, versus10 for R3 on all five completed requests.
These extents do not measure interior quality, uniform thickness or structural adequacy.
All voxels still mean whole building volume; no extra constraint family has been added.

## Interpretation

On these completed cases, learned selection is not demonstrated necessary to pass the
current geometric gates. It substantially affects mass distribution and seed sensitivity.
Do not conclude the neural component is useless, or that more variation makes R3 better.
Retain R3 as experimental; do not adopt the neutral baseline or request new training from
this partial evidence alone. MG7 comparison, broader geometry, route variation and meaningful
architectural diversity assessment remain outstanding.

## Failure and preservation

The controller stopped after118.625seconds. Workers0/1 completed in29.297/54.000seconds;
worker2 stopped after35.328seconds at the start of G10 seed2103 when the CPU allocator
could not allocate1,792,000bytes. Exact error and partial outputs are retained. The failing
allocation size is not a measurement of total peak memory. Host pressure versus capture
retention has not been diagnosed. No user process was killed and no retry was launched.
Each worker had180seconds and the overall allowance was900seconds; this was a memory
failure, not a time-cap stop or a model-quality failure. Worker3 never started.

Frozen inputs/source/model hashes rechecked. Completed worker manifests verified; the
interrupted worker has a separately named post-stop partial-evidence manifest. All existing
seed2102 G10/R3 fields at64/128 matched the prior studio fields byte-for-byte as boolean
arrays. Finite arrays, single births and monotonic final occupancy checked on saved outputs.
No broad runtime or memory-efficiency claim is made from these single timing samples.

## Resume

Do NOT rerun scripts/r3_contribution_study.py run over this attempt; it refuses an existing
controller. Preserve .local-artifacts/r3-contribution-2026-10-08 in full. Next inspect host
memory availability and reduce unnecessary trajectory capture retention in a separately
versioned runner, without changing the frozen sites, seeds, methods or acceptance rules.
Verify exact output parity on already-completed cases before completing the missing nine
method/request slots in a new attempt. Record engineering fixes separately from science.
Do not begin a fresh architecture/training sweep. Current studio8020 remains unchanged.

All source, summaries, raw trajectories, witnesses, contexts, runtime versions, logs and
partial evidence are locally archived under Codex outputs/R3-Contribution-2026-10-08.
Same-disk archive is not an off-device backup. No Drive operation, push or paid compute.
