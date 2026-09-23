# T1 findings: objective conflicts and weak success criteria

Completed 2026-09-23. T1_v1 run `20260923T082527Z_845d2aa6aec0`, source `f767975`,
records 432 static geometry cases, 72 necessary-bound checks and 36 direct
occupancy-gradient probes across all 18 frozen scenes. Runtime35.91 seconds CPU;
zero optimizer updates.123 regression tests pass in
`20260923T082336Z_b8c43fcf7723`; checkpoint smoke exit0.

[Full evidence](../../experiments/reports/T1-target-audit.md) includes every case.
Registered hashes, bound arithmetic, target mass/contact/physical volume, saved
norms and pairwise gradient cosines were independently rechecked. A fresh report
render matches the file. All nine recorded zero-loss witness configurations were
recomputed from archived geometry and exactly reproduce their terms.

## 1. The previous budget check was necessary but incomplete

At radius6/envelope,17 of17 feasible scenes passed the earlier capacity checks.
Adding the mandatory facade-contact bound leaves15/17. At radius3/envelope,
17/17 drops to11/17. The sealed reference remains separately invalid.

Coverage requires each guide cell to have unit occupancy. If C of those cells
lie in the facade zone, the15% contact cap requires total material M>=C/0.15.
Zero spill confines material to the envelope, and sparsity caps M at12% of that
envelope. Two radius-six cases cannot satisfy both limits:

| Scene | Mandatory facade cells | Minimum mass from contact | Maximum budget |
|---|---:|---:|---:|
| legacy-easy-seed-007 | 14 | 93.3333 | 92.2800 |
| legacy-easy-seed-008 | 13 | 86.6667 | 64.0800 |

Mass is in fully occupied voxel equivalents; multiply by0.512m3 at0.8m per voxel.
The second conflict is substantial, not rounding. Increasing a loss coefficient
cannot make these exact zero-loss requirements compatible. A weighted compromise
can exist, but must be labeled as such, with per-family residuals visible.

Do not widen envelopes or relax budgets by implication. Also do not call these
architecturally impossible scenes: the conflict belongs to the chosen formulas.
The original site budget retains its earlier envelope-capacity conflicts.

## 2. Zero loss does not yet mean useful architecture

Under radius6/envelope, the legal guide scores zero on all nine terms for
ref-01-ground-pair (36 cells,18.432m3) and ref-06-minimal-smoke (32 cells,16.384m3).
Their routes contain one-voxel-wide segments. A third zero-loss witness is the
radius-one target for ref-02-facade-pair-and-ground. Across all contexts there are
nine zero-loss witness configurations, including repeated geometry under different
budgets/envelopes. These are positive tests of the formulas and counterexamples
to interpreting zero loss as adequate architectural quality.

Current meanings explain this result:

- Thickness minimizes an eroded-core fraction. With radius2, a five-voxel cube
  must be filled to count a core. This discourages bulk; it does not guarantee
  a minimum member thickness, route width or clearance. The cube spans4m at the
  current scale. Legal clipping and grid boundaries can eliminate cores too.
- Facade penalizes contact above15% of total material. It does not require
  attachment, and can encourage adding unrelated material to dilute a ratio.
- Access follows connected material through six-neighbor adjacency. It is not
  traversal through free space, a walking surface, a ramp or headroom check.
- Support describes connection to a prescribed support boundary, not mechanics.
- The frozen procedural route can already satisfy coverage/access/support. NCA
  must demonstrate additional value rather than merely redraw that route.

Empty material never passes: all17 feasible empty controls fail coverage/access
and mass floor. Every nonempty procedural candidate connects all17 feasible
scenes by binary metrics, but thicker geometry frequently violates mass/facade
and radius-six filling violates the bulk penalty in all17 cases.

## 3. Loss magnitudes are not gradient magnitudes

The deterministic radius-six probe has occupancy0.70-0.90 on the guide and
0.01-0.05 elsewhere inside the envelope. It is a diagnostic field, not a model
output or data distribution. On the17 feasible scenes, median thickness value
is0.00565 while its legal-coordinate gradient norm is2.577. Coverage has value
0.201 and norm0.1715; access norm is1.0. Thickness gradients are about15 times
coverage gradients by these median summaries, despite much smaller loss values.

With site budgeting, sparsity and spill gradients oppose each other in17/17
probes (median cosine-0.974); sparsity and support also oppose in17/17
(-0.962). With envelope budgeting, the mass term is inactive in all these probes,
so this particular probe cannot calibrate its weight. Facade/support gradients
oppose in all12 probes where facade has a nonzero derivative. The report shows
all scales and defined pairs without hiding zero norms.

These are direct occupancy derivatives with forbidden coordinates removed, not
NCA parameter derivatives or a full feasible direction at box boundaries.
Max/min ties choose subgradients. They justify further measurement, not inverse-
gradient automatic weighting. Hard legality/ground derivatives on legal coordinates
are expected to be zero; hard projection can enforce them without trainable force.

## Decision and next work

Keep the present model and production defaults intact. Do not calibrate a final
trainer against known contradictory requirements or present unit weights as a
balanced recipe. NEXT_EXPERIMENT_PLAN.md defines the next staged comparison and
its acceptance conditions. First settle material-versus-circulation interpretation
and compare explicit facade/budget semantics; then sample under/in/over-budget
states, actual model parameter gradients and retained regularizers. Preserve nine
families. Paid Colab, larger grids and interface deployment remain separate gates.

The user's intended representation was requested in this task: usable pavilion/
bridge versus abstract connected material. Until answered, keep both branches of
the plan explicit and do not implement an architectural semantic choice by default.
All work and evidence remain local; no Drive access, paid compute or publication.
