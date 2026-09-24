# VA1 findings: material extent and spatial voids

2026-09-24. Corrected target and protocol: D057/VOLUMETRIC_AUDIT_PROTOCOL.md.
Run20260924T085504Z_36c834017633 completed in13.08s CPU. Nine hand-constructed
fields share one context; these are diagnostic probes, not generated designs,
training examples to imitate, or unseen-site generalization evidence.

| Field | Material cells | Bracketed empty cells | Sealed by form / with context | Old material ratio |
|---|---:|---:|---:|---:|
| empty | 0 | 0 | 0 / 0 | 0.00% |
| w1_scaffold | 37 | 0 | 0 / 0 | 3.06% |
| sp1_platform | 58 | 0 | 0 / 0 | 4.80% |
| slab_512 | 512 | 0 | 0 / 0 | 42.38% |
| solid_512 | 512 | 0 | 0 / 0 | 42.38% |
| open_ends_512 | 512 | 784 | 0 / 784 | 42.38% |
| side_aperture_487 | 487 | 634 | 0 / 0 | 40.31% |
| closed_shell_610 | 610 | 686 | 686 / 686 | 50.50% |
| extent_decoy_8 | 8 | 0 | 0 / 0 | 0.66% |

## What the comparison establishes

At512 material cells each, the slab and solid block have no two-axis-bracketed
empty cells; the hollow form has784. Material amount alone therefore does not
describe the tested spatial organization. The extent decoy has the same outer
dimensions as the hollow form but eight components and no bracketed void, so
bounding extent alone is insufficient too. These are specific counterexamples,
not proof that bracketed void is a sufficient measure of architectural quality.

The open-ended form has no cavity sealed by its own material, but the solid context
buildings close its ends, producing784 sealed free cells when context is included.
Removing25 side cells connects the cavity to the exterior; the aperture probe has
634 bracketed empty cells, all exterior-connected, and zero sealed cavities.
The fully closed shell contains686 sealed cells. Open and closed spatial voids
must remain distinct; enclosure is not a requirement imposed by the user's goal.

## Actual objective conflicts

The unchanged envelope contains1208 cells, so its12% cap is144.96 cells (at most144
binary material cells under the implemented tolerance). The three512-cell fields
all have material ratio42.38% and the same sparsity penalty about13.85. The aperture
probe reaches40.31%; the closed shell50.50%. These particular coarse volumetric
probes conflict with the old budget. This is NOT a proof that all volumetric forms
are infeasible under that budget, nor a justification for adopting those ratios
as new thresholds or inflating the denominator after seeing a candidate.

The old thickness term is the fraction of material surviving radius2 (5-cube)
erosion. It is0.125 on the solid8-cube and0 on the slab and all thin-walled hollow
probes. It penalizes filled bulk here, not the overall depth of a hollow form.
Removing thickness alone would not resolve the measured access/coverage and
budget conflicts. No training-gradient or causal-learning claim follows from
this fixed-field comparison.

The open-ended and aperture probes have access=1 and coverage=1: the old targets
ask for occupied material in the approach/route, where those probes have free
space. Closed shell coverage is about0.7143, access1. W1 has all nine penalties0
but zero bracketed spatial void. Facade/legality/ground/support and spill remain
reported individually; equal mass does not imply equal contact or feasibility.

## Representation and next step

A single material field already represents these filled, hollow and open forms.
The immediate issue is the objective contract and physical design domain, rather
than a demonstrated need for more NCA channels. This does not establish that the
existing backbone can learn useful volumetric growth. That is a later experiment.

Follow VOLUMETRIC_NEXT_PHASE.md: define a candidate-independent 3D opportunity
region in world units, audit material-budget compatibility, distinguish free-space
and material roles within access, and revise coverage away from mandatory occupied
centerlines. Keep all nine families, old scores and counterexamples. Do not turn
the current axis-bracket diagnostic into an untested universal loss or impose a
room/shelter program. No training or budget change occurred in VA1.

## Product and evidence

http://127.0.0.1:8001/static/volume/index.html shows nine saved probes with selectable
comparison, cutaway and movable vertical/horizontal slices. Counts use full saved
fields regardless of display clipping. Blue is measured emptiness, not generated
material. The SP1 page now names its brief as superseded and links to VA1; its data
and historical source archives remain intact.

Regression20260924T085027Z_d8c033bd285c:248 passed,0fail/error/skip,smoke0.
12 new analytic geometry tests. Both run archives and source hashes verify; all
nine volume reports, old diagnostic reports and saved masks recompute exactly.
QA: experiments/reports/VA1-verification.json and .local-artifacts/volume-qa/VA1-20260924.
No paid compute, Drive access, public deployment or remote push. Local archive
receipt establishes completion; same-disk storage is not an off-device backup.
