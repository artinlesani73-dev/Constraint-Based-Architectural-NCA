# MG5: coverage-aware mass growth

2026-09-24. The separately versioned procedural generator passes all 180
nonpartition candidates in the combined MG3/MG4 development set. All 45 complete
partition controls remain failed. The frozen 225-case admission gate passes;
179 earlier valid candidates remain valid and the single MG4 failure is repaired.
The original MG4 result remains 143/144; it has not been overwritten or relabeled.

Diagnostic 20260924T170732Z_b7d16f22121a: 4/4 pass, one changed field, all initial routes identical.
Conditional matrix 20260924T170831Z_264ba0e38124: 225/225 executed, 180/225 overall pass (80%). There
are no execution errors, timeouts or observed resource-cap breaches. Distinguish
expected blocked failures from execution failures and from the nonpartition rate.
Regression 20260924T170440Z_05e362daa48f: 332 tests pass, no skips/errors/failures, smoke exit 0.

## What changed and why

MG3's seeded routing and radial/contact costs stay byte-for-byte preserved in
their historical module. MG5's new coverage_budget_growth_v1 module keeps the
same route construction and changes the order of subsequent whole-cube additions.
Every accepted union remains connected, legal and subject to the existing 15%
façade-contact ratio. All nine constraint families and MT1 thresholds are unchanged.

For each current legal frontier cube, count unique new voxels and their fixed-X-
third memberships. Prefer the greatest fraction of additions that reduce a
remaining third deficit, capping each contribution at that deficit. Original
seeded radial priority breaks ties. Exact integer summed volumes compute deltas;
the full frontier is reconsidered as occupancy/contact totals change. Zero-new-cell
transit is finite: each origin expands once. Stop at the original total request
even if coverage is still unmet; no hidden volume extension or postprocessing.

Counts are usable as substantial bulk because the field is a union of full legal
2.4 m cubes. The independent evaluator verifies every final bulk mask. This does
not justify counting arbitrary occupied cells as bulk in other generators.

## Repaired case and preservation

combined_reverse / 16% / seed 5 requests 1,503 voxels. The new output has exactly
1,503, compared with 1,507 previously. Façade contact is 224/1503 = 14.903526%,
below the unchanged 15% limit. All nine families now pass.

| Fixed domain third | Domain cells | Required bulk | MG4 bulk | MG5 bulk | MG5 coverage |
|---|---:|---:|---:|---:|---:|
| West | 3544 | 284 | 266 | 293 | 8.267494% |
| Middle | 2303 | 185 | 303 | 289 | 12.548849% |
| East | 3544 | 284 | 938 | 921 | 25.987585% |

The west shortfall is resolved through a different growth sequence. Comparing
the final sets gives 27 cells present only in MG5 and 31 present only in MG4;
these are comparison differences, not deletion/postprocessing operations.

All 225 initial routes match the saved baseline exactly. Of 225 final fields,
221 are identical. The four changes are all in MG4 contexts; all 45 MG3 outputs
remain identical. The three previously valid changed outputs remain valid.

| Case | Previous → current voxels | New-only / old-only cells | Outcome |
|---|---:|---:|---|
| vertical_span__v16__s5 | 1019 → 1018 | 1 / 2 | pass → pass |
| combined_reverse__v16__s5 | 1507 → 1503 | 27 / 31 | fail → pass |
| combined_reverse__v24__s5 | 2254 → 2254 | 35 / 35 | pass → pass |
| combined_reverse__v32__s5 | 3007 → 3008 | 46 / 45 | pass → pass |

All 180 nonpartition outputs are 100% cube-qualified bulk and reach the requested
count with 0–8 extra cells, within the frozen bound of fewer than 27 extra cells.
The 45 controls finish no_cube_route with empty output. Their complete separating
walls are known input controls; generic routing failure is not an infeasibility
proof for an arbitrary site. Every failed field and score is retained.

## Diversity and computation

All 60 nonpartition site/request groups have three valid, distinct seeded fields.
The 15 blocked groups have zero valid fields and no diversity estimate. Mean
pairwise Jaccard distance across valid groups ranges 0.077149–0.620989. It measures
voxel-set variation, not architectural quality. Four group summaries change:

| Site / request | Valid seeds before → after | Mean Jaccard distance before → after |
|---|---:|---:|
| vertical_span / 16% | 3 → 3 | 0.408708 → 0.409621 |
| combined_reverse / 16% | 2 → 3 | 0.349067 → 0.333141 |
| combined_reverse / 24% | 3 → 3 | 0.298188 → 0.293368 |
| combined_reverse / 32% | 3 → 3 | 0.269887 → 0.260526 |

The repaired 16% group changes from two to three valid seeds, so its old and new
means have different pair populations (one versus three). The 24% and 32% groups
show small decreases with the same three seeds. No diversity collapse occurred
within this set, but these observations do not establish a quality improvement.

One local CPU matrix at 32³, 0.8 m/cell, two Torch threads, NumPy 2.5.2,
PyTorch 2.8.0+cpu, Python 3.12.14, Windows 11. Generation includes route/growth
and excludes separate binary evaluation and artifact serialization.

| Observation | Value |
|---|---:|
| Nonpartition generation median / p95 / max | 0.720 / 2.914 / 4.244 s |
| All-candidate generation median / p95 / max | 0.583 / 2.757 / 4.244 s |
| Per-case sampled process RSS peak maximum | 345.57 MiB |
| Matrix wall time, including checks and saving | 328.981 s |
| Growth decisions / candidate evaluations | 105,139 / 31,392,998 |
| Rejected candidate evaluations | 1,809,324 |

Candidate counts include repeated frontier evaluations. Requested RSS sample
interval was 10 ms; sampling can miss short peaks, and RSS describes the entire
process. The 15 s candidate cap, 1800 s matrix cap and 2 GiB observed RSS limit
were respected. These are single-run observations, not a latency guarantee.
The new rule performs more accounting. Historical MG3/MG4 timing was not rerun
under matched instrumentation/order here: do not report a causal slowdown factor
or speedup. Profiling and resource admission are needed before larger grids.

## Verification and interpretation

Independent verification rebuilt 25 contexts, rescored all 225 new and 225
previous fields, checked 225 bulk masks and exactly replayed all 225 new fields,
routes and reports (excluding timing). It reconstructed the legal frontier at
every growth step and directly enumerated cube voxels, without the new summed-
volume helper, to audit all 105,139 decisions and 31,392,998 candidate evaluations.
It checked eligibility, unique-cell/third/contact deltas, ranking/ties, frontier
hashes and final unions. All 157 Python files match each source snapshot. The
four-case audit independently covers 2,149 decisions / 589,073 evaluations.
Both admission gates and all valid-only diversity counts/means were recomputed.

This is an inspected development comparison prompted by a known failure. It is
not fresh held-out generalization, trained NCA performance, mechanical safety,
habitability or a proof that MT1's axis-dependent coverage proxy is sufficient.
Passing this set supports this bounded growth repair; it does not require a
whole-project conceptual redesign. D058 building-volume semantics remain current.

The live MS1 Studio still uses its existing MG3 generator and records. No runtime
or UI source was changed, no server restart was required, and no new model was
trained. Larger environments, finer resolution, fresh-site evaluation and product
promotion remain separate work. Read SCALE_READINESS_NEXT_PLAN.md for the next
bounded proposal; do not silently change the deployed generator identity.

All sources, frozen recipe/protocol, old/new fields, routes, accepted origins,
decision traces, context masks, scores, resources and events are retained in
.local-artifacts/runs/<run IDs>. Small records/reports are tracked. Local archive
and restored Git bundle preserve this milestone plus the prior archive chain.
The private reports remain ignored and unchanged. Same-disk backup is not
off-device protection; no Drive operation, paid compute, push or publication ran.
