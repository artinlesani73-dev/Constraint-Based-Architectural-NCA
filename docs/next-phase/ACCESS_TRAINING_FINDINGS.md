# F2 findings: access improves connectivity, material growth remains excessive

2026-09-23. The access-only change improves actual generated connectivity on
these two development scenes. It does not achieve joint connectivity and material
budget success. Keep it experimental; no production model or recipe is promoted.

## What changed

The four models match F1 in original initialization, architecture, scene exposure,
training seed0,64 updates,16-step rollout, Adam settings and both existing recipes.
Only access changes to component_bottleneck_v2. The other eight families and three
regularizers remain identical. Evaluate all seven fixed boundaries at16/50 growth
steps using separate firing seed2. Both access definitions and both recipe totals
are retained; semantic rescoring alone is never counted as geometry improvement.

F2B proves exact agreement with F1 on12 short training updates and24 evaluations,
including complete optimizer/scheduler/RNG/checkpoint state and voxel arrays.
All28 shared source files match F1 byte-for-byte. Reuse its immutable64-update
controls; do not describe them as freshly retrained for this study.

## Final results after64 updates

Connectivity uses material>0.5 and a single component touching every entrance.
Old and new binary labels agree on all56 F2 evaluations. The budget is continuous
material/envelope3%-12%, tolerance1e-6, not a binary voxel count.

| Recipe / scene | Growth steps | F1 connected | F2 connected | F1 mass | F2 mass |
|---|---:|---|---|---:|---:|
| weight30 / ground-pair |16|No|No|12.99%|13.88%|
| weight30 / minimal-smoke |16|No|No|13.14%|13.32%|
| weight3 / ground-pair |16|No|Yes|18.64%|22.18%|
| weight3 / minimal-smoke |16|No|Yes|18.09%|22.88%|
| weight30 / ground-pair |50|No|Yes|20.11%|23.13%|
| weight30 / minimal-smoke |50|Yes|Yes|25.47%|25.92%|
| weight3 / ground-pair |50|Yes|Yes|28.72%|31.63%|
| weight3 / minimal-smoke |50|Yes|Yes|30.60%|33.42%|

At16 steps, final connectivity improves0/4 to2/4. At50,3/4 to4/4. All eight final
F2 fields contain MORE continuous material than their matched F1 fields; none is
within budget. Across all56 evaluated F2 boundaries, no field meets connectivity
and budget together. There are12/28 in-budget16-step fields and1/28 in-budget
50-step fields, but these are disconnected; the denominators include update0.
All connected F2 evaluations occur at the final64-update boundary. These repeated
measurements are not independent samples or a statistical generalization claim.

All eight final fields have zero binary illegal, blocked-ground and unsupported
voxels; every count is retained in F2-evaluations.json. These proxies do not establish usable
architecture or mechanical safety. Coverage and other continuous residuals remain.

## When the access intervention begins to matter

Exact paired model weights first differ at update51 for weight30/ground,40 for
weight30/minimal,34 for weight3/ground,30 for weight3/minimal. The saved pre-update
fields first differ one update later. Before those boundaries, model weights
match F1 exactly. The first partial candidate access losses occur at those same
updates. This is evidence of changed learning trajectories, not merely rescoring.
Partial loss by itself is not proof of a nonzero parameter derivative on every
later update. All256 recorded gradients exceed the clip threshold, also true of
F1; that observation alone does not establish clipping as the cause of failure.

The two connected16-step F2 fields still have access losses0.3491 and0.2949 because
the continuous bottleneck objective targets strength1, beyond the binary0.5 test.
That is the declared objective, not a scoring error. Whether this extra pressure
contributes to excess growth needs an actual parameter-gradient comparison.

## Interruption, continuation and timing deviation

The timing pilot estimated1334.37s, below the1800s total and600s/member caps.
During the full run, a tool return was delayed929.8s. The third member recorded
1137.74s despite its600s cap; wait(timeout) returned successfully. A system pause
is plausible but unconfirmed. Active CPU time was not measured; do not subtract
the observed delay or present this as a clean performance benchmark.

The original run stopped at the overall cap after254 recorded updates/54
evaluations: three complete models, and update62 of the fourth. Its status remains
interrupted. A separate120-second linked continuation copied those308 records
with matching field/checkpoint hashes, restored update62 and executed only63/64
plus final evaluations. It completed in18.99s including imports; the worker took
12.31s. Cumulative elapsed1820.10s is explicitly NOT original-cap compliant.
The complete scientific matrix is usable; timing compliance is not claimed.
No registered update63 checkpoint existed in the interrupted parent. Any killed
partial computation remains distinct from completed recorded updates.

The runner now checks elapsed time even when wait returns successfully, records
elapsed_cap_exceeded and stops the coordinator before another member starts.
A regression simulates the observed failure. This change happened AFTER all F2
and recovery evidence; historical checkpoints require their original source ZIP.

## Verification and preserved evidence

| Evidence | Run ID | Result |
|---|---|---|
| Initial regression |20260923T134911Z_c3c1cd97dbdb|163 passed; smoke0|
| F2B baseline parity |20260923T135153Z_388dadf703b2|12 training/24 evaluations exact;114.65s|
| F2R early recovery |20260923T135424Z_5e2993d3e18e|8 executed updates/14 evaluations;11 exact checks;70.28s|
| F2P timing pilot |20260923T135623Z_cc3691d8cc81|8 updates/24 evaluations;74.98s; admitted|
| F2 interrupted parent |20260923T135818Z_5520d5d80cec|254 updates/54 evaluations;1801.11s|
| F2 linked completion |20260923T143002Z_55aeaac95580|256/56 total; only2/2 newly executed|
| F2L later recovery |20260923T143228Z_73318ff99c0b|4 final updates and8 evaluations reproduce exactly;38.96s|
| Post-fix regression |20260923T143444Z_3c97b03e720e|164 passed; zero failures/errors/skips; smoke0|

Full verification rescored312 F2 fields and56 F1 controls under both definitions,
checked256 checkpoint boundaries,56 evaluation metrics,32 source hashes and
all308 imported records. Eight initial fields match F1 exactly. All eight final
evaluations replay exactly from the four saved checkpoints. F2L separately proves
exact full-state resume at update63->64 for all four trained models, covering
later behavior beyond the early recovery gate. It does not certify abrupt writes,
CUDA/AMP or arbitrary earlier boundaries. The actual interrupted62->64 completion
and the four63->64 repeat checks are recorded separately.

Reports: experiments/reports/F2-*,F2B-*,F2R-*,F2P-*,F2L-*. All raw fields,
checkpoints, logs, source snapshots and the interrupted attempt remain under
.local-artifacts/runs. Post-hoc scripts/results are preserved under analysis-attempts.
One provenance-archive permission failure was retained and then resolved without
overwriting its already saved report. No scientific evidence was discarded.

Formula rescoring uses shared formulas; independent BFS checks component
connectivity. One seed and two already-seen scenes cannot establish robustness,
generalization, convergence or architectural quality. Read GROWTH_STABILITY_PLAN.md
and D041/D042 next. No Drive access, paid compute, remote push or deployment.
