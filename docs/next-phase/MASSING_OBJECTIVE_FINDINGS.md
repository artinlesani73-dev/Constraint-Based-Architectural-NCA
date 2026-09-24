# MO1 - Objective admission completed; optimization remains untested

2026-09-24. User liked the MG1 volumes and authorized proceeding. Preserve that
visual/geometric baseline. This milestone implements and checks a separate
piecewise-differentiable CPU objective for the same nine massing families; it
does not change the historical losses, MT1 acceptance or displayed geometry.

## Recorded results

Audit `20260924T115136Z_91b3707a89d3`: all97 archived MT1/MG1 fields retained,873 family
comparisons, zero mismatches and zero bulk-mask differences. Wall time
33.62s includes source snapshot, context preparation and evidence writes.
The full regression ran concurrently, so this is not an isolated performance test.
Both checks read frozen science code; their source snapshots match all142 relevant
Python files. Full regression `20260924T115046Z_0a0c9bfc3d10`:297pass, zero failures/errors/
skips, smoke0, 103.07s. Ten new focused tests also pass. No failed audit or
regression attempt occurred for MO1. No optimizer update or new model was run.

The audit covers48 MT1 analytical controls and49 MG1 fields (45 generated plus
four challenges), not97 independent sites. Family agreement is at binary endpoints;
it does not prove a continuous low-loss field will threshold to a valid design.

## What the new objective does

Min/max cube opening reproduces MT1 bulk exactly at binary endpoints. Interface
bottleneck strengths are combined with a component-excess term on raw and bulk
occupancy, so connected endpoints cannot hide detached satellites. Descending
component merges have live-tensor birth/death gathers for piecewise gradients.
Support uses unrestricted maximum-bottleneck paths from fixed support, rather
than a fixed hop count. Protected/illegal/outside cells remain in raw accounting.
Empty output cannot pass support, thickness, coverage or access merely by avoiding
violations. All definitions and caveats are in MASSING_OBJECTIVE_PROTOCOL.md.

These are nonsmooth graph/min/max operations with deterministic tie branches.
Finite differences were checked on untied operator inputs. Do not claim a unique
gradient at ties, globally smooth dynamics or an efficient GPU implementation.
Residual magnitudes differ; MO1 does not choose a weighted production objective.

## Actual gradient evidence and its limits

Two fixed probes use p=.005+.99*the saved binary field, including nonzero background
outside the domain. This intentionally provides interior probability values for
derivative checks; these diffuse fields are not accepted generated designs.

| Probe | Facade residual | Facade gradient norm | Analytic directional derivative | Numerical derivative |
|---|---:|---:|---:|---:|
| aligned__v24__s0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| partial_obstruction__v24__s0 | 0.02171294 | 0.05215479 | -3.21053634 | -3.21053634 |

All18 family-gradient arrays are finite. The aligned facade term is inactive, so
its zero gradient is expected. The obstructed example has a nonzero descent
direction agreeing with centered finite differences. An independent closed-form
contact-ratio derivative matches its norm and directional derivative as well.
The audit does not claim all nine terms are active or nonzero on these probes.

The facade ratio can decrease by adding non-contact volume as well as removing
contact. This is the retained dilution weakness, not a newly solved problem.
A negative directional derivative is not evidence of better thresholded geometry,
maintained connections, useful diversity or preserved mass. The optimizer must
be evaluated against the full binary contract and requested-volume error.

## Next bounded work

MASSING_DIRECT_PILOT_PLAN.md specifies a proposed four-member MD1 pilot: aligned
and partial-obstruction MG1 cases,24% request, seeds0and1; independent logit fields,
explicit domain projection and32updates/member. Candidate family coefficients,
volume preference and Adam settings are declared in that plan; none is claimed
optimal. They are a new optimizer hypothesis, not part of MO1 endpoint equivalence.

Implement the actual loop and iteration0 parity first; demonstrate exact new-process
checkpoint recovery and profile timing before admission. Retain no-update controls
and prepare a separately versioned contact-aware procedural comparator. Do not
claim optimization beats procedural generation based solely on contact-unaware MG1.
No automatic extra weight sweeps, longer runs, paid compute or NCA training.

## Evidence and preservation

RunStore preserves all97 per-field residual records, the complete two source studies,
two gradient records, source snapshot and result. MO1-verification.json verifies
manifests, source identities, every recorded family comparison and the independent
facade derivative formula; it does not claim a second complete graph audit.
The original reports, old checkpoint/losses, MG1 fields and viewer remain unchanged.
Latest resume and D063 identify current work; older entries are historical.
The milestone has a local commit and verified incremental archive; retain MG1,
Review R2, MT1 and their parent archive chain. Same-disk archive is not off-device.
No Drive operation, paid Colab, server restart, push or publication.
