# MG6: physical-scale mass generation

2026-09-25. The 48³ study passed its admission checks. The 64³ study failed:
both obstructed cases timed out before reaching their requested volume.
This is a small designed scale study using the unchanged
MG5 procedural generator and unchanged MT1 nine-family checks. It does not train
an NCA or establish architectural quality, arbitrary-site reliability or interactive
latency. All old MG5/MG4 sources and outcomes remain preserved.

Regression 20260925T075016Z_c82729a145e4:337 tests pass, smoke0. Five new tests cover scene-sized decoding,
historical context parity, translated-domain/score invariance, increased physical
domains and explicit refusal of resized legacy interfaces. Focused tests1.015s;
full regression181.934s, including infrastructure rather than inference alone.

Study runs: 20260925T075350Z_c677671d8310, 20260925T075945Z_3fc0d812e153. Read SCALE_STUDY_PROTOCOL.md and MG6-scale config/
scenes for exact preregistration. The48 stage preceded any admitted64 generation.
There was no scientific retry, coefficient search or outcome-based scene change.

| Grid | Nonpartition MT1 passes | Nonpartition requests met | Blocked passes | Timeouts | Admission | Total study seconds |
|---|---:|---:|---:|---:|---|---:|
| 48³ | 4/4 | 4/4 | 0/2 | 0 | pass | 61.770 |
| 64³ | 3/4 | 2/4 | 0/2 | 2 | fail | 530.120 |

## Physical meaning and context correctness

The 48³ and 64³ world boxes span 38.4 m and 51.2 m per side at 0.8 m/cell. The building
gap is 25.6 m at 48³ and 38.4 m at 64³. Interface blocks remain 1.6 m per edge, the protected
ground band 4.8 m, and growth cubes 2.4 m. This is a larger physical environment,
not finer resolution. The opportunity region retains6.4m Z/Y endpoint padding.
The exact tested physical domains are:

| Context | Domain voxels | Domain cubic metres |
|---|---:|---:|
| 48__blocked | 8640 | 4423.680 |
| 48__compact | 9216 | 4718.592 |
| 48__offset_obstacle | 22738 | 11641.856 |
| 64__blocked | 13248 | 6782.976 |
| 64__compact | 13824 | 7077.888 |
| 64__offset_obstacle | 34254 | 17538.048 |

At each executed size, the saved MG5 aligned24%,seed0 field was translated in XY
with its original site. Its opportunity domain, gross volume and every independent
MT1 score were unchanged. This embedding control preserves the physical site and
does not count as a larger-site generation success. Seeded generation itself is
not assumed translation-invariant: changing grid size changes RNG array draws.

The new helper explicitly sets context config grid_size and street_levels from
the scene, checks actual state channels against scene geometry, and records exact
masks/config/world units. Historical checkpoint weights are unused; its saved
config supplies existing channel definitions. No model was resized or retrained.
The old builder's two-cell interface limit remains explicit, so finer resolution
will need a separate representation study rather than silent interface shrinkage.

## Candidate results and observed work

Two seeds6/7 per context, one24% volume request, unchanged route cost12. Resource
limits were45s per candidate/420s per48 study and120s/900s at64,2GiB observed RSS,
local CPU with2 Torch threads. These are cooperative checks, not OS hard limits.
Nonpartition does not assert feasibility in advance. Complete partitions are
deliberate failed controls, not defects to hide or reroll. All arrays and traces,
including failed/partial output, are saved in lossless NPZ/JSON with hashed copies.

| Case | Nine-family result | Termination | Generation s | Growth s | Evaluation s | Request error cells |
|---|---|---|---:|---:|---:|---:|
| 48__compact__s6 | pass | target_reached | 5.276 | 4.637 | 0.555 | 4 |
| 48__compact__s7 | pass | target_reached | 5.592 | 5.024 | 0.566 | 2 |
| 48__offset_obstacle__s6 | pass | target_reached | 19.253 | 18.369 | 0.861 | 0 |
| 48__offset_obstacle__s7 | pass | target_reached | 19.983 | 19.125 | 0.863 | 1 |
| 48__blocked__s6 | access, coverage, sparsity, support, thickness | no_cube_route | 0.655 | 0.000 | 0.479 | -2074 |
| 48__blocked__s7 | access, coverage, sparsity, support, thickness | no_cube_route | 0.564 | 0.000 | 0.478 | -2074 |
| 64__compact__s6 | pass | target_reached | 77.241 | 75.103 | 1.994 | 1 |
| 64__compact__s7 | pass | target_reached | 74.886 | 72.759 | 1.854 | 0 |
| 64__offset_obstacle__s6 | pass | time_limit | 120.088 | 117.294 | 2.397 | -4029 |
| 64__offset_obstacle__s7 | coverage, sparsity | time_limit | 234.532 | 231.760 | 3.474 | -6316 |
| 64__blocked__s6 | access, coverage, sparsity, support, thickness | no_cube_route | 3.165 | 0.000 | 1.974 | -3180 |
| 64__blocked__s7 | access, coverage, sparsity, support, thickness | no_cube_route | 1.987 | 0.000 | 1.607 | -3180 |

For failed empty controls, negative request errors report the actual shortfall;
request fidelity is an admission requirement for nonpartition candidates only.
An MT1 pass alone does not imply that the requested24% volume was reached or the
runtime gate passed. There were6/8
fulfilled nonpartition requests and2 recorded generator timeouts.

- 64__offset_obstacle__s6: 4192 cells produced for 8221 requested; the recorded partial field is retained.
- 64__offset_obstacle__s7: 1905 cells produced for 8221 requested; the recorded partial field is retained.

The 64³ offset/seed 7 case reported 234.526 s generation wall time against its 120 s
cooperative setting, with only 38.578 s process CPU for generation plus evaluation.
It retained 1,905/8,221 requested cells and failed coverage and sparsity. Much of
the elapsed time was not recorded as process CPU; the logs do not establish the
cause. Do not attribute that entire wall duration to intrinsic compute cost.
The time_limit status and overrun are preserved. The timer cannot enforce a hard
deadline while its process is unscheduled or blocked inside an unchecked operation.
No timeout setting was increased and no replacement run was substituted.

Generation time in this table is outer wall time, including a scoped timing
wrapper. Growth is measured at its single function boundary. The saved residual
combines setup, Dijkstra and radial initialization; it is not isolated route time.
The wrapper restores the original function and independent replay uses no wrapper.
Growth accounts for97.71% of
generation wall time across nonpartition cases. This locates cost in growth as
a whole; it does not isolate count rebuilding from frontier ranking.
That wall-time share includes the anomalous offset/seed7 interval. Independently,
compact/seed6 spent75.103s in growth of77.238s generator time and recorded80.25s
CPU including evaluation; the expensive growth finding does not rely solely on
the anomalous timeout case. These remain single-run observations.
Context construction and saving/hashing times are retained separately. Study time
includes embedding audit, context construction, scoring and evidence preservation.

Maximum sampled process RSS was352.47MiB; lifetime
peak353.48MiB. Per-case requested10ms sampling spans
generation/evaluation and may miss brief peaks; lifetime peak includes other
process allocations. These are observed working sets, not just tensor estimates.
Nonpartition generation median47.435s,
maximum234.532s. No speedup factor is
inferred from unmatched MG5 historical timing or from unequal48/64 sites. Two
seeds per context are insufficient for a robust tail-latency estimate.

| Context | Valid fields | Unique valid fields | Pairs | Mean Jaccard distance |
|---|---:|---:|---:|---:|
| 48__blocked | 0 | 0 | 0 | undefined |
| 48__compact | 2 | 2 | 1 | 0.21298910851149655 |
| 48__offset_obstacle | 2 | 2 | 1 | 0.6845403060609712 |
| 64__blocked | 0 | 0 | 0 | undefined |
| 64__compact | 2 | 2 | 1 | 0.3328309469982417 |
| 64__offset_obstacle | 1 | 1 | 0 | undefined |

Distances compare only valid voxel sets. One pair per fully passing context is
descriptive variation, not evidence of population diversity or design quality.
Blocked fields have no valid diversity estimate.

## Independent checks and next step

Across the executed stages:10 exact unwrapped generator
replays, 12 independent binary rescores/bulk masks, 6
rebuilt larger-site contexts, 2 exact embedding audits and
13818 accepted-growth-step audits. Every audited cube
is legal; unique additions/contact/third counts and final unions agree. All160
Python sources match each stage/regression snapshot; every registered artifact
and its retained original payload hash verifies. Read MG6-48-verification.json
and, if executed, MG6-64-verification.json. The exhaustive frontier ranking audit
from MG5 remains the algorithm evidence; MG6 does not claim to repeat that whole
audit at every size. Time-limited partial fields, if any, are audited as retained
traces rather than claimed reproducible wall-clock cutoffs.

Next read SCALE_EFFICIENCY_NEXT_PLAN.md. Local count updates are a concrete
candidate for reducing repeated whole-grid work while preserving every choice.
Freeze exact-output and paired-resource comparisons before optimization outcomes.
Do not conflate this with changing constraints, increasing training channels or
relaxing the requested volume. A later live integration must declare supported
sites/sizes and measured latency, and retain old generator replay identities.

No live Studio/runtime change, browser operation, NCA training, paid compute,
Drive operation, push or public hosting occurred. The original private reports
remain unchanged and ignored. All decisions/results and resume instructions are
tracked; local raw-source/Git-bundle archive is verified, on the same disk only.
