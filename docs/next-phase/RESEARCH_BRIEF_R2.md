# Revised research brief - R2

Accepted 2026-09-24, D061. Evidence cutoff: MT1 commit 968c27c.
This is the tracked implementation brief accompanying the private dated report
addendum. It takes precedence over older material/platform/void-first plans and
outdated milestone labels below the newest PLAN/RESUME entries. Historical evidence
and the original review are preserved, not rewritten.

## Current question

Can a context-conditioned local learned update add useful generation, variation
or recovery behavior beyond procedural and per-scene optimization controls for
overall building mass? NCA is a candidate method, not a required successful answer.
Generate building volume now; interiors and construction come later (D058).
Occupied voxel volume is neither construction material quantity nor floor area.
Retain volumetric depth and meaningful exterior gaps without prescribing rooms.

## Evidence and limits

- F5 closed the old material investigation: 66/72 connected final evaluations,
  0/72 in-budget and 0/72 joint successes. These are repeated development
  evaluations under the old contract, not independent held-out sites or massing
  outcomes. W1 17/17 and D1 12/17 joint outcomes are useful historical controls,
  not matched-compute proof against NCA. See LOCAL_PHASE_CLOSURE.md.
- MA1's 33 saved comparisons show completion can erase intended gaps and cannot
  invent missing depth. Prefer direct mass generation; retain completion as an
  explicit raw/final ablation. See MASSING_AUDIT_FINDINGS.md.
- MT1 is a binary pilot evaluator, not a differentiable loss or trained generator.
  Four contexts, 48 controls, 432 sensitivity reports and 96 checks support its
  intended examples only. Six designed positives pass; 42 intended negative or
  blocked outcomes reject. See MASSING_TARGETS_FINDINGS.md.
- Latest regression: 275 passed in 20260924T102553Z_4ae3c1b346f2. Documentation
  revision adds no new scientific run and needs no repeat model regression.
- Studio S1/S2 delivered local procedural records, comparison, durable jobs,
  cancellation/restart visibility and verified JSON import/export. MT1 is a static
  gallery. Arbitrary-site massing, trained mass generation, full product redesign,
  held-out generalization and larger-grid validation remain unfinished.

## Nine families, explicitly revised meanings

| Family | Building-mass pilot meaning |
|---|---|
| access | Connected mass and substantial-volume connection to every interface; not internal circulation |
| coverage | Distribution of substantial mass in a fixed scene-derived domain |
| thickness | Local volumetric depth, not suppression of all filled bulk |
| sparsity | Lower/upper building-volume fraction of the fixed domain, counting every occupied cell |
| spill | No mass outside the declared opportunity region |
| legality | No occupancy outside the permitted mask |
| ground | No occupancy in the protected ground mask |
| facade | Historical non-exempt contact ratio, with its limitations retained |
| support | Historical geometric boundary attachment, not mechanical safety |

MT1 pilot: 8-40% volume, 2.4 m cube scale, 90% bulk-qualified fraction,
8% bulk occupancy per fixed X third. These are provisional, not architectural
standards. Axis bias, bulky lattices, thin appendages and facade-ratio dilution
remain challenges. Passing geometry does not certify design quality. Preserve
historical metrics; do not compare scores across contracts as if identical.

## Next bounded sequence

### R2-A - Procedural mass alternatives (next)

Specify and implement parameterized mass generation from supported scenes and
interfaces. Freeze scene classes, candidate/seed matrix, volume requests,
evaluation version and local compute cap before benchmarking. Retain every outcome,
including unsupported contexts and failures, with fields/config/source hashes.
Measure nine-family validity, runtime, volume, invalid/duplicate fractions and
geometric diversity among valid outputs. MT1 contexts are development cases;
add targeted rotated/diagonal, appendage, thin-neck and lattice challenges.
The exit is a complete reproducible outcome matrix and a failure-based decision,
not an obligation to make every case pass by adjusting acceptance thresholds.

### R2-B - Direct optimization

Create a separately versioned continuous objective within the nine families.
Verify binary endpoint agreement, batch independence, finite values and meaningful
gradients before fitting. Match scenes/domains/volume requests and independent
final evaluation to the procedural control. Record timing and all raw/thresholded
fields and any projection/completion. Freeze a bounded protocol before execution;
per-scene optimization is not generalizing inference.

### R2-C - Conditional learning pilot

Select one measurable advantage from control results: faster repeated valid
generation, useful valid variation, or local recovery after an edit. Freeze
numerical improvement and validity non-regression targets, fresh site-family splits,
seeds, compute allowance and stop rule before learned results. Planner-initialized,
persistently conditioned NCA remains a candidate, not a committed successful
architecture. W1's thin route is not a valid massing solution by itself.
One preregistered comparison leads to a decision; failure does not automatically
authorize another loss tweak, longer training or more channels. Keep architecture,
objectives and distribution changes distinguishable. Evaluate recovery/locality
as task outcomes, not additional constraint families.

### Product and scale gates

Integrate the verified mass generator with Studio jobs, saved alternatives and
clear procedural/optimized/learned/postprocessed labels. Check real cancellation,
restart/import and responsive error states for integrated jobs. Later implement
direct manipulation, efficient orbitable rendering and unit-verified exports.
Before claiming 10x progress, freeze comparable user tasks/hardware and measure
time to usable alternative, median/tail latency, memory, transfer size and task
completion. MT1's about 17.6x smaller payload is only a transfer-size result.
Profile 32 cubed before 64 cubed; separate larger sites from finer resolution.
Multiscale perception, extra latent channels and a decoder remain conditional.

## Preservation, authority and compute

The private NCA-Next-Phase-Report-Revision-2-2026-09-24.md/.pdf supplement the
unchanged original report and remain ignored. Use this brief, D061 and the latest
RESUME entry to continue; older dated current/in-progress notes are historical.
PLANNER_REFINER_SPEC.md is historical motivation, not a current material contract.
This addendum synthesizes local findings; it is not a new literature review.

Preserve every old result and original checkpoint. New experimental attempts need
unique IDs and parent links. Before paid Colab: frozen experiment and GPU-hour cap,
GPU recovery including optimizer/RNG/pool as applicable, and a verified second
artifact copy. Existing same-disk archives are not off-device backup. Every Drive
operation requires explicit approval within the designated project folder. No
training, upload, push or publication follows from this documentation milestone.
