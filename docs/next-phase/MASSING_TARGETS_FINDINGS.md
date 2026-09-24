# MT1: a usable pilot contract, not yet a trained mass generator

2026-09-24. Building mass is now tested against an explicitly versioned binary
contract within the existing nine families. It rewards substantial connected
volume and distribution, instead of interpreting all occupied cells as construction
material whose bulk should be suppressed. Historical losses and model remain intact.

Read MASSING_TARGETS_PROTOCOL.md for exact definitions, numerical pilot choices,
scope and limitations. Code: nca/massing_targets.py; data recipe:
experiments/configs/MT1-targets.json. These settings were declared before the audit;
none was adjusted after observing its outcomes.

## Evidence

Run `20260924T102755Z_e89550a24d8d`:4 development contexts,12 controls each,
48 base records,432 sensitivity records,96 checks all pass;93.23s CPU. Regression
`20260924T102553Z_4ae3c1b346f2`:275 passed,0 failures/errors/skips,smoke0,104.92s
including infrastructure. Fifteen new tests cover physical cube resolution,
boundaries, even-width refinement, disconnected pieces, thin necks, invalid contexts,
volume accounting and representative positive/negative expectations.

| Context | Compact mass | Articulated mass | Other ten controls |
|---|---|---|---|
| Aligned | Pass | Pass | All rejected |
| Wider gap | Pass | Pass | All rejected |
| Offset interfaces | Pass | Pass | All rejected |
| Full blocking partition | Rejected | Rejected | All rejected |

These are intended outcomes of constructed controls, not a6/48 success rate for a
generator. No new model generated the fields. In blocked context, the region itself
lacks a substantial interface connection; the evaluator reports that necessary
condition failure. It retains collisions in the candidate instead of clipping them
away. Passing the context check in another scene is not a full feasibility proof.

## What changed in meaning

Access now requires connected occupied mass and a connected subset made of complete
cubes at least2.4m wide. Both must connect every declared interface; detached mass
cannot be hidden by measuring only a main component. A one-voxel neck can connect
the raw field while failing this substantial-connection check. This is geometric
mass continuity, not walking access or floor/headroom design.

Thickness requires at least90% of occupied volume to belong to such cubes. This
tests local depth; filled boxes retain their surface cells because we measure the
union of whole cubes, not just their eroded centers. Width is rounded upward to
voxel resolution. The aligned refinement test uses3-cell and6-cell cubes at the
same physical scale; it does not prove invariance on arbitrary curved boundaries.

Coverage tests substantial volume in each of three fixed X thirds of a scene-derived
region, at least8% per third. Sparsity separately requires8-40% total occupied
volume relative to that region. Spill rejects volume outside it. Ground and legality
are strict zero-violation checks. Facade retains the historical15% non-exempt contact
ratio and support retains geometric boundary attachment. No tenth family added.

The MA1 region was diagnostic only; MT1 explicitly uses it as a distribution/spill
boundary for this pilot. This is a documented within-family change, not a hidden
new height ceiling. The region and every denominator are fixed before candidate
evaluation. Invalid or outside-region occupied cells remain in volume accounting.

## Sensitivity of the provisional numbers

All nine combinations of cube width1.6/2.4/3.2m and upper volume cap25/40/55% were
evaluated on the same48 fields. Other parameters remained fixed. Full reports are
retained in the run's sensitivity.json; no selected-outcome filtering.

| Upper volume cap | Passes at1.6m | Passes at2.4m | Passes at3.2m |
|---|---:|---:|---:|
|25%|5|5|5|
|40%|6|6|6|
|55%|6|6|6|

Only offset_interfaces/compact_mass changes its verdict: its1440 cells occupy
26.6075% of the5412-cell region. The1274-cell articulated alternative occupies
23.5403% and remains below25%. Thus the volume cap affects acceptable alternatives;
there is no basis here to label40% an optimal budget. No verdict changes from cube
scale alone in these coarse examples; this does not establish the best scale.

Aligned compact/articulated occupy576/470 cells (16.4948%/13.4593%). Wider-gap
controls occupy720/580 cells (16.5289%/13.3150%). All six passing base controls have
100% cube-qualified volume at the2.4m pilot setting.

## Limits and next bounded implementation

The contract establishes geometric meaning and rejects the listed counterexamples.
It does not rank design quality, diversity, articulation, usefulness or habitability.
It remains axis dependent: three X bins suit the tested gap-facing scenes; they are
not a general orientation-independent site partition. Cube-based checks can favor
orthogonal blocks or volumetric lattices;90% bulk permits small thin appendages.
The retained facade ratio still permits dilution by additional non-facade mass.
Geometric attachment is not mechanical support or construction feasibility.

Next, use this binary contract as an independent evaluator for two bounded controls:
1. A parameterized procedural mass generator with multiple retained alternatives
   per scene, measured for feasibility and geometric diversity. Include offset
   interfaces and context obstructions; explicitly report unsupported contexts.
2. A versioned continuous objective for direct voxel optimization. Verify binary
   endpoint agreement, batch behavior and nontrivial gradient flow before fitting;
   measure final thresholded geometry with MT1 rather than trusting soft losses.

Fix comparable domains, requested volume ranges, seeds and compute caps. Include
near-boundary thin links/appendages, volumetric lattices and rotated/diagonal forms
as targeted challenges before claiming general robustness. Do not restart an
unbounded sequence of loss tweaks. A subsequent NCA experiment should have a
specific advantage to test over these controls, such as recovery or fast conditioned
generation. No NCA training or paid Colab run has started under MT1.

## Implementation and preservation

All48 original fields, bulk masks, full context masks, scenes, nine historical
penalties and new family verdicts are saved. All48 facade terms agree numerically
with their old implementation within1e-6. All135 relevant Python files match tested
and audited snapshots. Both source ZIP manifests and run artifact hashes verify;
old SP1/VA1/MA1 study files and checkpoint bytes remain unchanged. Independent
verification checks all bulk masks, counts, record uniqueness and base/sensitivity
parity; it does not claim a second full run of all432 connectivity evaluations.

Viewer /static/targets/index.html compares a compact reference with any selected
control in the same context. All48 selector combinations show the expected verdict
and nine families. True sections, cutaway, context and highlighting were exercised;
desktop/mobile screenshots and final UI sources are retained. The checkbox says
"Highlight other cells" because unqualified volume may be illegal/outside-domain
as well as thin. A first immediate page snapshot showed loading; the loaded state
was verified afterward. No failed scientific or regression attempt occurred.

The viewer uses a compact JSON projection, retaining every displayed case/report
exactly while leaving full context masks and region coordinates in the immutable
run archive. Its source-study hash and export hash are recorded. This reduces
browser transfer without discarding any evidence. It remains a static development
viewer on32³/0.8m geometry, not a general generator. All changes, controls,
sensitivity results and resumption steps are local and archived. No Drive access,
push, publication, old objective replacement or paid training.


## Final review correction and retained first revision

Initial audit20260924T101444Z_3d8c5504cb9e and regression20260924T101120Z_6d8b5d1cc577 passed their
declared controls (274 tests). Review then found a context-feasibility edge case:
the source interface can touch multiple disconnected available regions. Checking
only its first component can falsely reject a site with a later feasible component.
The final implementation examines each source component and requires a single
component to touch all interfaces; disjoint paths cannot be combined to fake it.
A focused regression covers both cases. Final275-test pass and linked audit20260924T102755Z_e89550a24d8d
preserve the unchanged48 records and all432 sensitivity reports exactly. No
threshold, scene, expected outcome or old scientific evidence was altered.
Final verification: experiments/reports/MT1-final-verification.json. The earlier
verification report remains a historical record of the initial code revision.
