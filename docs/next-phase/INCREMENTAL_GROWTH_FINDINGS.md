# MG7: incremental cube accounting

2026-09-25. The equivalence and paired performance gates both pass. All 235 completed reference outputs and growth reports
match exactly after excluding only version and wall time. Both former 64-grid
timeouts now complete with the same recorded prefixes and independently audited
subsequent decisions. All 188 nonpartition cases meet MT1 and the requested volume;
all 49 blocked controls remain failed. These are the saved MG5/MG6 cases, not new
sites or a general reliability estimate.

This is an efficiency change to the procedural volume generator. It does not
train an NCA, change the nine constraint families, add rooms or change physical
resolution. Occupied cells still mean overall building volume (D058). The live
Studio remains MS1/MG3; no existing result or public deployment was changed.

## What changed and why

MG6 found that repeated full-grid cube accounting dominated growth at larger
sizes. The separate `nca/incremental_mass_generator.py` initializes the original
counts once, then decrements only cube origins containing newly occupied cells.
For width w and coordinate q, affected origins on each axis range from
max(0,q-w+1) to min(q+1,number_of_origins), with the upper bound excluded.
Distinct new voxels are applied once; already occupied voxels and zero-addition
transit do not decrement counts. Counts by spatial third, total missing cells
and new facade contact remain exact integers.

The original routing, random draws, radial cost, coverage priority, contact
budget, tie breaks, requested-volume stop and frontier trace are unchanged.
The full frontier is still reconsidered at each step. This is a bounded
optimization of accounting, not a different geometric search. MG5 source and
MG6 partial outputs remain immutable comparators.

## Frozen experiment and evidence

Protocol: INCREMENTAL_GROWTH_PROTOCOL.md and experiments/configs/MG7-incremental.json.
Regression 20260925T082528Z_d941a7a46957: 343 tests pass, smoke exit 0; full regression 162.273 seconds.
Six added tests cover direct cache enumeration across widths/boundaries/contact,
exact growth reports and edge cases, forged full/prefix comparisons and timestamped
resource sampling. Source and protocol were frozen before real-scene results.

Runs: diagnostic 20260925T082841Z_88466161ced3, matrix 20260925T083302Z_6c7ade2f0e16, timing 20260925T084205Z_6fab76a67538.
Each later stage was conditional on the preceding verified gate. No coefficient,
scene, timeout or threshold retuning and no scientific retries were used.

| Stage | Executions | Full comparisons | Prefix comparisons | Admission | Study wall seconds |
|---|---:|---:|---:|---|---:|
| diagnostic | 6 | 4 | 2 | pass | 20.316 |
| matrix | 237 | 235 | 2 | pass | 207.161 |
| timing | 26 | 26 | 0 | pass | 311.162 |

The matrix contains 225 MG5 cases and six MG6 cases at each larger size.
All 188 nonpartition requests are met with 0 to 8 extra voxels,
within the frozen less-than-27-cell allowance. The blocked controls retain
`no_cube_route`. Full comparison includes route, occupied field and all recorded
growth choices; two time-limited references support only prefix comparison.

| Former timeout | Occupied / requested | MG7 matrix generation seconds | Old recorded steps | New full steps |
|---|---:|---:|---:|---:|
| mg64__64__offset_obstacle__s6 | 8223 / 8221 | 2.327 | 2052 | 5046 |
| mg64__64__offset_obstacle__s7 | 8221 / 8221 | 2.225 | 410 | 5051 |

Independent verification rescored every old/new field and bulk mask, replayed
all six diagnostic and 237 matrix outputs without the measurement wrapper,
and directly enumerated every frontier cube for both newly completed 64-grid
traces: 10097 decisions and 12131925 candidate evaluations.
This audit does not call either incremental counts or summed-volume counts.
There are 538 binary rescores across the three stages and 164 exact
Python source matches in each compared source snapshot. Timing outputs are
compared to completed saved references; they are not rerun as another benchmark.

## Paired local timing

Four measured trials per method per case; case order frozen. Method order
alternates reference/optimized, optimized/reference, reference/optimized,
optimized/reference. Two retained 32-grid warmups are excluded from medians.
All 26 executions and raw sample timelines are preserved. The admission rule
requires optimized/reference median wall and CPU ratios at most 0.75, maximum
sampled RSS ratio at most 1.25, and no measured sampling gap over one second.

| Completed-reference case | Median wall s: old / new | Wall speedup | Median CPU s: old / new | CPU speedup | Gate |
|---|---:|---:|---:|---:|---|
| mg4__combined_reverse__v16__s5 | 1.124 / 0.376 | 2.99x | 1.125 / 0.359 | 3.13x | pass |
| 48__offset_obstacle__s6 | 15.723 / 1.242 | 12.66x | 15.859 / 1.227 | 12.93x | pass |
| 64__compact__s6 | 46.563 / 1.356 | 34.34x | 47.375 / 1.367 | 34.65x | pass |

| Case | Maximum sampled RSS MiB: old / new | RSS ratio | Largest measured sample gap s |
|---|---:|---:|---:|
| mg4__combined_reverse__v16__s5 | 269.47 / 270.77 | 1.005 | 0.2153 |
| 48__offset_obstacle__s6 | 261.25 / 249.83 | 0.956 | 0.1797 |
| 64__compact__s6 | 298.34 / 275.60 | 0.924 | 0.2609 |

These ratios describe three matched, completed workloads on this local CPU.
They are not a whole-product speed claim, a confidence interval, or GPU/NCA
training evidence. The old 64-grid timeouts are not valid complete-workload timing
comparators. MG6's 234.526-second wall overrun and wall/CPU discrepancy remain
recorded with unknown cause; MG7 does not explain them retrospectively.

Native sampling requests 10 ms intervals and spans generation plus evaluation;
it can miss brief memory peaks. Same-process trials retain allocator effects.
Generation, growth, evaluation and saving are recorded separately. Study time
also includes input loading, old-field evaluation and evidence persistence.
Cooperative generator limits are not OS-enforced hard deadlines.

## Decision and next step

The equivalence and paired performance gates both pass. Preserve the separate optimized version and all evidence. Read
STUDIO_SCALE_INTEGRATION_NEXT_PLAN.md for conditional integration scope. No
automatic live switch is implied. Arbitrary sites, unseen seeds, finer resolution,
learned generation and public hosting need their own evaluation and decisions.

Exact scene inputs, old/new arrays, traces, resource samples, source snapshots,
protocols and events are under .local-artifacts/runs/<run ID>. Small records and
verification summaries are tracked under experiments/. The local milestone
archive contains raw source and a restore-checked Git bundle plus all MG7 runs.
Keep earlier archives: this archive is incremental. Same-disk storage is not an
off-device backup. Private reports remain ignored and unchanged; Drive untouched.
