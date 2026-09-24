# SP1 findings: spatial meaning changes the target

2026-09-24. Run20260924T073056Z_db9cf854c9e3,8.49s CPU, no training.
Read SPATIAL_PLATFORM_SPEC.md for frozen definitions and dimensions.

The aligned platform passes the declared single-plane spatial gate. All five
negative/unsupported cases fail as preregistered. W1 fails that spatial gate in
all six examples. These are hand-designed development cases, not a success-rate
estimate on unseen architecture or a trained-method comparison.

The positive deck has58 material cells,37.12m² floor footprint including a16m²
landing, a2.4m-wide strip and2.4m clear height. It occupies4.8013% of the unchanged
1208-cell permitted envelope, within the original3–12% range. Historical access
and coverage penalties are both1.0; the other seven family penalties are0.0.
W1 on the SAME scene has all nine historical penalties0.0, material ratio3.0629%
and passes historical joint connectivity/budget, but fails the new spatial gate.

This is direct evidence of a semantic mismatch: old guide/entrance targets occupy
the intended approach space. A floor below that space can meet the brief while
failing those targets. Do not fill the intended passage just to recover old scores.
Also do not claim the new gate satisfies the entire old objective or establishes
a structurally safe, accessible or inhabitable building.

| Example | Spatial candidate outcome | Interpretation |
|---|---|---|
| Aligned approaches | Pass | Declared floor, landing and footprint route exist |
| Narrow connecting strips | Fail | 0.8m strips cannot admit2.4m square footprint |
| Missing floor | Fail | Removed cross-section disconnects the surface route |
| Low headroom | Fail |15 floor cells lack required clearance; clear surface falls to27.52m² |
| Blocked span | Fail |15 candidate cells intersect context; no accepted deck |
| Split levels | Unsupported | Method only constructs level decks; not proof of global infeasibility |

The procedural method is intentionally simple: a straight level strip and square
landing, with independent diagnostics and retained failures. Derived surface/void
masks suffice for this case without changing the NCA's channels. This does not
prove that they suffice for multilevel or enclosed architectural spaces.

## Product and evidence

Open http://127.0.0.1:8001/static/spatial/index.html or use the spatial-prototype
link in Studio. Inspect the original scaffold and candidate side by side; select
one of six saved cases, switch plan/section/axonometric, and toggle context or
clear volume. Section is a cut through Y=15 cells (12m), not an elevation projection.
Building boxes remain solid; entrance markers denote outside approach regions.

The gallery is a saved run with an explicit run ID. It does not yet construct
platforms from arbitrary Studio edits or import them into the S2 record library.
Both historical families and new spatial diagnostics are shown separately.

Verification20260924T072445Z_e8fffe7ec74b:236 tests passed,0fail/error/skip,
smoke0. Backend/spec recipe and evaluator unchanged since this pass. Subsequent
gallery HTML/CSS/JS and camera fit were checked in the browser.12 new spatial tests
include independent geometry, width/clearance boundaries, collision retention,
empty/floating material, diagonal rejection and old-metric parity. Saved reports
and masks all recompute exactly; served JSON matches run bytes; source and run
artifacts verify. See experiments/reports/SP1-verification.json and
.local-artifacts/spatial-qa/SP1-20260924. No paid compute or Drive operation.
The QA provenance-verification.json records the effective checkpoint-derived
configuration and checkpoint hash;41 current scientific/input files match the
original run snapshot, whose complete manifest also verifies.

## Next bounded step

Define external approaches versus actual door/interior connections before expanding
this prototype. Then design a versioned spatial access/coverage target that rewards
floor-supported clear passage instead of occupied approach volume. Evaluate it on
the same saved fields first, including the negatives, with the old metrics retained.
Audit feasibility and differentiability before a learned refiner experiment.
Do not start a new training series, merely extend the grid, or treat this one-level
gallery as completion of the larger architectural-generator objective.


## Scope correction - D057, 2026-09-24

This earlier platform/refinement direction is superseded as the research target.
The user clarified volumetric forms with spatial depth and voids, without prescribed
rooms, shelters or functions. Preserve this document as diagnostic/history. Current
scope and next work: VOLUMETRIC_AUDIT_FINDINGS.md and VOLUMETRIC_NEXT_PHASE.md.
