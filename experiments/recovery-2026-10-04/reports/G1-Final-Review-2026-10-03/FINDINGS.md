# G1 seed-to-volume generation: completed pilot, acceptance failed

2026-10-03. Run `20261003T184158Z_42fc06fd2028`. No model promotion or additional paid run.

## Execution is verified

All 1,034 exported payload hashes, exact archive membership and the original receipt verify. The package manifest matches the corrected portable-path package. Training completed 256 updates, with exactly 128 seed starts and 128 teacher-stage starts verified against every recorded sampler row and stage-depth rule. Final checkpoint trace, sampler cursor and Adam steps agree at 256. Both on-device recovery replays report exact full-payload and state equality. This review verifies the exported reports and checkpoint integrity; it does not independently repeat GPU execution.

Controlled run time: 108.181 seconds; worker: 101.972 seconds; peak reserved GPU memory: 1,436,549,120 bytes (1,370 MiB). Tesla T4 and the prescribed cu130 runtime are recorded in imports.json. These are this run's measurements, not general speed predictions. The earlier Windows-path failure remains separate historical evidence.

## Frozen evaluation

Final checkpoint 256 only. Nine predeclared development requests, fresh single-cell seed per request, CPU float32, firing seed 2101, no cleanup or target input. Both fixed horizons are reported without selecting the more favorable one. Evaluation uses saved source, with the exact packaged model implementation. CPU evaluation is not a cross-device numerical-equivalence claim. Reserved targets were not generated or evaluated.

| Measure | 64 steps: primary | 128 steps: stability |
|---|---:|---:|
| All nine families pass | 0/9 | 0/9 |
| Occupied cells, range | 99–122 | 197–247 |
| Fraction of available domain | 1.83–2.25% | 3.63–4.56% |
| Median absolute request error | 22.00 percentage points | 19.88 percentage points |
| Maximum absolute request error | 30.01 percentage points | 28.20 percentage points |
| Median teacher IoU, diagnostic only | 0.0835 | 0.1407 |

Requests were 16%, 24% and 32%. Every output fails access, coverage, sparsity (too little volume) and thickness. Every output passes facade, ground, legality, spill and geometric support. The cube-supported fraction is 53.85–71.05% at 64 steps and 57.20–72.87% at 128 steps, below the required 90%.

Between the two fixed horizons, occupied cell count increases by 90.57–107.62%; the frozen stability limit was 5%. The field has not stabilized at an adequate volume. All five preregistered acceptance gates fail. The procedural MG7 reference remains 9/9 valid on these requests. Repair results from earlier datasets are not directly comparable to these generation scores.

## What archived training states reveal

No additional model run was required for this diagnostic. Across the last 64 archived training-forward states:

| Start type | Samples | Median starting cells | Median final cells | Median teacher recall | Median false-positive cells |
|---|---:|---:|---:|---:|---:|
| Single seed | 32 | 1 | 209.5 | 20.63% | 4 |
| Teacher stage | 32 | 250 | 759.5 | 92.80% | 100.5 |

These are pre-optimizer forward states from changing weights and different sampled targets, not paired final-model measurements. They show that low completion from a seed is also present in recorded training trajectories; it is not only an observed development-set problem. They do not prove which loss term, schedule or architectural mechanism causes the failure, or rule out a longer training budget.

The current loss supervises the local frontier and local cube-volume discrepancy, while hard growth decisions are detached. Good continuation from supplied teacher mass does not establish successful early growth from a seed. The evidence supports examining the training signal and schedule for that early stage before committing another compute budget. It does not justify claiming that more updates, a larger grid, or a reversible rule will necessarily solve the problem.

## Decision and next step

Do not adopt G1 or relabel partial growth as a successful building mass. Keep MG7 live; preserve G1 as the first verified seed-generation baseline. No automatic retry, threshold relaxation, checkpoint search or extra horizon search.

Next local work: use a fixed final checkpoint and matched TRAIN examples to distinguish seed-start/early-stage behavior from later continuation, and inspect where the current frontier loss supplies useful positive growth gradients. From that diagnosis, specify one focused training change with a preregistered comparison. Keep the nine families, physical scales and reserved-set boundary unchanged. This is proposed diagnostic/design work, not approval for another paid run.

## Preservation

Original ZIP and receipt, final checkpoint, all exported update records, import checks, 18 scored fields and internal states, training diagnostics, exact review script and source are retained here. Original ZIP retains every checkpoint and training state. Review results and resume instructions are in result.json and RESUME.json. The milestone ZIP is hash-verified locally; it is not an off-device backup.

Repository records remain unsynchronized because repository write access was not granted. No repository edit or commit, Drive operation, live-model replacement, push or public deployment occurred. Resume from this folder and the preceding G1 Portable Paths record, not the stale repository D098 record alone.
