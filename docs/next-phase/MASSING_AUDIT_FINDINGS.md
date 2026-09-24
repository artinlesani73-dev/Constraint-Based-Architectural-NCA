# MA1 findings: filling is useful, but not a generation objective

2026-09-24. Decision D058 means occupied cells represent overall building volume,
with interiors and construction deferred. MA1 implements an original/derived
comparison under that interpretation; it does not train or promote a model.

Successful study: `20260924T095133Z_b7015ab62f05`, 11 source fields, 3 operations,
33 retained records, all83 checks pass,14.07s CPU. Parent attempt
`20260924T094943Z_1d6ed517bc60` is retained as failed. Regression
`20260924T094603Z_66da5a9b1158`:260 tests pass,0 failures/errors/skips, smoke0,
103.44s including infrastructure. Twelve new tests cover completion semantics,
context/domain rejection, boundaries, resolution, units and gap counterexamples.

## Results

Counts are occupied voxels at0.8m, not floor area or construction quantities.
All fields and operations are saved, including the unchanged controls.

| Source | Original | Enclosed cavities only | Vertical gaps <=6.4m |
|---|---:|---:|---:|
| Empty |0|0|0|
| W1 scaffold |37|37|37|
| SP1 platform |58|58|58|
| Flat slab |512|512|512|
| Solid block |512|512|512|
| Open-ended shell |512|512|1296|
| Shell with side aperture |487|487|1296|
| Closed shell |610|1296|1296|
| Eight corner cells |8|8|36|
| Detached plates |288|288|1296|
| Separated blocks |864|864|1296|

The closed shell's686-cell cavity is geometrically identifiable. Open shells have
no source-enclosed cavity; context buildings cannot supply that enclosure in this
operation. Vertical completion fills the open shell, but also erases intentional
separation between plates or blocks. These five different sources yield exactly
the same1296-cell final block under that rule. Thus final-output scores alone
cannot explain what the raw generator achieved or what postprocessing removed.

Neither filling method gives the thin platform or W1 path missing volumetric
depth. Completing eight corners produces four disconnected posts, not a coherent
mass. Completion is not a substitute for a suitable mass-generation objective.
These are analytic counterexamples, not learned-model/generalization results.

## Region and objective compatibility

The fixed physical region contains3492 cells/1787.904m³. It comes from interface
extents plus explicit physical padding, historical legality and context exclusion;
it does not adapt to a candidate. The original route envelope remains1208 cells.
Thirty-six cells below the street band are allowed by the old anchor semantics.
The new region is a diagnostic denominator, not a new legality ceiling or an
accepted material/massing budget. Padding remains a study parameter needing
multi-scene review. The aligned refinement test preserves physical volume exactly;
other physical boundaries have center-sampling discretization error.

For the open shell, vertical completion changes512 to1296 cells,262.144 to
663.552m³, preserving outer extents12.8 x7.2 x7.2m. Its fixed-region fraction changes
14.662% to37.113%. Do not adopt37.113% as a budget target from this hand-built shape.

| Family | Open shell -> filled mass (old penalties) | Implication for next contract |
|---|---|---|
| access |1 ->0; occupied interfaces become connected | Useful contact/connectivity evidence, not internal circulation. |
| coverage |1 ->0 | Filling the route solves this old target; it does not establish good mass distribution. |
| facade |0 ->0 | Retain current reports; broader compatibility not established by one scene. |
| ground |0 ->0 | Keep existing protected-space semantics explicit, including anchor exceptions. |
| legality |0 ->0 | Added cells avoid existing context and historical forbidden cells. |
| sparsity |13.84791 ->136.18781 | Old occupied-cell budget strongly rejects this mass; calibrate building volume independently. |
| spill |0.01009 ->0.01631 | Old route-envelope reference remains a conflict to audit. |
| support |0 ->0 | Geometric attachment only, not structural certification. |
| thickness |0 ->0.23148 | Bulk penalty conflicts with filled mass in this example; review within the same family. |

The old ratio increases42.384% to107.285% because it divides all occupied cells by
the1208-cell route envelope. It is not the fraction of that envelope filled, and
is not clipped at100%. Historical3-12% budget results remain unmet. Merely replacing
the denominator must not be presented as having solved the objective.

Every derived case has zero blocked additions in this particular study. Separate
unit tests verify blocked context/domain additions and preservation of original
violations. There is no overall acceptance score or new ninth-plus-one constraint.

## Decision and next work

Use direct building occupancy as the primary representation for the next control
design. Retain cavity/axis completion as explicit ablations, saving raw and final
fields separately. This is an implementation recommendation, not evidence that
direct learned generation is already successful.

Next bounded milestone: define and test a versioned massing objective contract
within the nine families. Establish physical depth/distribution and volume-budget
examples across multiple scenes before training. Audit access as declared interface
connection, coverage as volumetric distribution, and thickness as a mass-scale
control rather than automatically penalizing every filled interior. Challenge it
with thin paths/sheets, fragments, compact and articulated masses, excessive fill,
collisions and unsupported masses. Keep all historical metrics side by side.
Compare procedural and direct-optimization controls before an NCA trial. No larger
grid, new backbone or paid Colab run is justified by MA1 alone.

## Evidence, corrections and viewer

First attempt failed only its expected count:3456 omitted36 allowed anchor cells.
All82 other checks passed. Corrected recipe3492; fresh attempt linked to the failed
one. All33 geometries, operations and metric reports are identical between runs;
no scientific algorithm or threshold was adjusted to improve a result.

Verification replays all33 operations, masks, mass descriptors and nine-family
reports exactly. Parent VA1 data remain byte-identical; parent/source/checkpoint
hashes and run manifests verify. Both run source snapshots verify; all131 relevant
Python files match the tested and audited bytes. See MA1-verification.json.

New gallery: /static/massing/index.html, linked from Studio and historical pages.
Left is original geometry; right is derived building mass. Ochre marks added
volume; turning off highlighting shows one occupancy color. Cutaway and true
vertical/horizontal slices are display controls. No blue mask is treated as a fill
instruction. All33 selector combinations show correct counts; desktop and390px
mobile layouts were visually checked. Mobile tables fit within the viewport.

One screenshot write into the repository was denied by the browser filesystem
scope; screenshots were saved in the authorized workspace and copied unchanged.
An initial verification-helper attempt completed replays but named an incorrect
checkpoint path; the corrected helper uses the authoritative notebooks/model_c
path. The failed helper source is retained. A transient pre-resize DOM read showed
old dimensions; later read and screenshot confirmed390px. One legend label was
corrected for unhighlighted occupancy. Browser navigation can settle after the
immediate snapshot; confirm actual destination before claiming a link passed.

All work remains local. No original loss/model/checkpoint change, training, Drive
access, push or publication. Preserve VA1 and prior archives; the MA1 incremental
archive adds both attempts, regression, final viewer and verification evidence.
