# MG1 - Procedural building-mass alternatives

2026-09-24. R2-A / D062. Generator cube_route_growth_v1; unchanged binary MT1
evaluation. Occupancy means building volume, with interiors/construction deferred.
This is a procedural development benchmark, not learned growth or held-out evidence.

## Complete retained matrix

Run `20260924T113023Z_24298393f2a5`: 45 generated requests and four separate analytical
challenges, no unexecuted cases. 27/45 generated fields meet all
nine pilot checks. The denominator includes nine fully blocked-context requests.
Construction status, request error and independent validity remain separate.

| Context | Valid / requested | Distinct valid fields across requests | Failed-family counts |
|---|---:|---:|---|
| aligned | 9/9 | 9 | none |
| wide_gap | 9/9 | 9 | none |
| offset_interfaces | 9/9 | 9 | none |
| blocked_gap | 0/9 | 0 | access: 9, coverage: 9, sparsity: 9, support: 9, thickness: 9 |
| partial_obstruction | 0/9 | 0 | facade: 9 |

Study wall time 13.014s, including source snapshot, evaluation
and evidence writes. Total measured generation 1.484s;
candidate range 0.0134-0.0953s. No GPU, NCA or training.
The generator performs seeded graph routing and cube-union growth without evaluator
feedback or acceptance-based rerolling. These timings do not compare against equal-
compute NCA or demonstrate a product-wide speedup.

## Volume control and variation

Requested fractions were 16/24/32%, each with seeds 0/1/2 in every context. Finite
cube additions can overshoot the requested cell count. A too-large initial route,
exhaustion or no route would be retained with explicit status. Full per-candidate
counts, requested counts/errors, selected cube origins, routes and final fields are
in the immutable study and individual records.

Same-request valid-only groups with at least two fields: 9.

- aligned / 16%: 3 valid, 3 unique, mean pairwise Jaccard distance 0.4126.
- aligned / 24%: 3 valid, 3 unique, mean pairwise Jaccard distance 0.3875.
- aligned / 32%: 3 valid, 3 unique, mean pairwise Jaccard distance 0.3090.
- wide_gap / 16%: 3 valid, 3 unique, mean pairwise Jaccard distance 0.4620.
- wide_gap / 24%: 3 valid, 3 unique, mean pairwise Jaccard distance 0.3740.
- wide_gap / 32%: 3 valid, 3 unique, mean pairwise Jaccard distance 0.3118.
- offset_interfaces / 16%: 3 valid, 3 unique, mean pairwise Jaccard distance 0.1842.
- offset_interfaces / 24%: 3 valid, 3 unique, mean pairwise Jaccard distance 0.1085.
- offset_interfaces / 32%: 3 valid, 3 unique, mean pairwise Jaccard distance 0.0597.

All pairwise distances and duplicate fractions are retained, including empty groups
with undefined diversity. Cross-request summaries are separately labeled because
volume changes themselves increase voxel difference. Distinct voxels are not proof
of useful design variation; no aesthetic ranking is inferred.

## Separate evaluator challenges

| Challenge | Pilot pass | Failed families |
|---|---|---|
| thin_appendage | True | none |
| bulky_lattice | False | facade |
| diagonal_band | False | access, spill |
| quarter_turn_field | False | access, coverage, support |

These fields are not generator outputs. The bulky lattice deliberately consists
of full 3-cell bars and highlights that cube-supported depth is not a design-quality
measure. The appendage probes the 90% allowance. The diagonal and quarter-turn
fields keep context fixed: this is not a consistently rotated-site invariance test.
No pass/fail expectation or threshold adjustment was imposed after seeing results.

## Verification and preserved failures

Final regression `20260924T112636Z_ea40cf77f2f4`: 287 tests pass, zero failures/errors/skips,
smoke exit 0, 114.78s. Twelve new tests. Failed parent
20260924T112005Z_e9860ed79bdf tried copying text as an array in the new preservation
test. Linked 20260924T112334Z_588b73843c6d then treated nested endpoint masks as one
array. Both runs have 287 tests, one test error, zero failures, smoke 0. Corrected
the test with recursive type-aware comparison; no generator science changed.
Focused 12-test run passed before the final full regression.

MG1-verification.json records exact deterministic replay of all 45 generated fields,
route fields and generation metadata except wall time; all 49 binary reports and
bulk masks recompute exactly. All 139 relevant Python files match both final
regression and study source snapshots. Run artifact manifests and source ZIP member
hashes verify. Old scientific implementation and studies remain unchanged.

## Product and next decision

The /static/generation gallery displays all saved outcomes, route/final comparison,
three views and full-field diagnostics. It is static evidence presentation; arbitrary
scene submission and durable mass-generation jobs are not yet integrated into Studio.
See local browser QA and final verification receipts for tested interactions.

Do not tune MT1 thresholds to improve this matrix or repeat the old incremental
training sequence. Review failure causes and construct R2-B continuous-objective
tests next, with the same scenes/domains/requests and all failures retained. Before
any optimization run freeze its objective, exact budget and endpoint/gradient tests.
The current algorithm supplies a concrete massing baseline, not a universal planner.
There is no learned-model promotion or architectural-quality certification.

No paid Colab, Drive access, remote push or public hosting. Local archive is on the
same disk; retain the MT1, R2 and earlier archive chain for all prior evidence.


## Browser verification and interpretation update

All49 selector combinations show the correct saved verdict and nine family rows.
Axonometric/vertical/horizontal views, keyboard slice change15to16, cutaway, context
and growth toggles checked.390px viewport: document375px (scrollbar excluded),
tables343.8125px; no horizontal overflow. Final warning/error console list empty.
Two full-page screenshot attempts failed; visible screenshots in a taller viewport
preserve visual checks. Source and screenshots saved under Codex cwd outputs/mg1-qa.

All nine partial-obstruction outputs satisfy access, coverage, ground, legality,
sparsity, spill, support and thickness; facade alone fails. This exposes the
generator's lack of contact-aware routing/growth, not evidence the context has no
feasible solution. A future procedural contact-aware ablation should remain a fair
comparator when testing R2-B. Do not weaken facade limits to claim success.
Thin appendage passes all checks; bulky lattice passes thickness but fails facade.
The pilot has not become an architectural-quality evaluator. Variation shrinks for
the larger offset masses (same-request mean Jaccard0.1842 to0.0597); this points to
limited form variety, rather than evidence of a learned design distribution.
