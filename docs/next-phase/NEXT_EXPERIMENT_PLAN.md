# Next experiment plan after T1

Prepared2026-09-23. This is a staged plan, not a preregistered training run or
approval for Colab. T1 proves objective contradictions in some scenes and weak
zero-loss witnesses in others. Research and product quality both need a better
success definition before changing the model architecture or scaling grids.

## Recommended near-term scope after user clarification

The user is unsure which representation to choose and asked what is more feasible
from the prior plan. Recommend architectural form/material generation first, with
usability evaluated separately. Keep usable pavilion/bridge design as the longer-term
objective. The existing single material field and evaluated routing/support proxies
support the first scope directly. Usable circulation requires explicit floor/void/
clearance interpretation that current evidence does not validate.

This keeps the previous corrected-baseline-first plan: demonstrate learned value,
resolve contradictory facade/material budgets, evaluate recurrent stability and
site diversity, then scale and improve the studio. Do not equate this scope with
arbitrary sculpture or structural certification. Keep spatial scale/entrances/
protected street and attachment context explicit. A zero-loss procedural guide is
a baseline to surpass, not sufficient evidence of learned architectural value.

This is the recommended working direction; the user has not chosen a final design
specification. No user permission is needed to preserve the completed audit or
prepare comparisons, but record later user steering before changing that intent.

## Stage A: choose and test the meaning of existing objectives

Use the recommended material-generation scope for preparation; keep any final
semantic choice explicit and incorporate subsequent user steering. For a usable pavilion/bridge,
access should evaluate traversable space and its relationship to material/decks;
for abstract structural material, retain material connectivity and make later
usability validation explicit. Both stay within the existing access family.
Preserve material connectivity as a diagnostic in either case. Choose widths,
clearance and voxel discretization explicitly; do not infer building-code compliance.

Before implementation, write a new semantic contract and adversarial fixtures:
one-voxel strand, obstructed route, disconnected islands, floating member, facade
anchor versus facade blanket, and an oversized solid volume. A usable-space
contract must distinguish a connected solid strand from usable circulation.
Minimum material thickness, if intended, must be distinguished from the existing
maximum-bulk penalty within the thickness family. One is not a drop-in synonym
for the other. Keep old terms and all T1 scores available for paired comparison.

For the facade/budget contradiction, compare these explicit alternatives:

| Candidate | What changes | Required evidence |
|---|---|---|
| Current formulas | Nothing; baseline with known residuals | Keep all17 feasible scenes and their incompatibility labels |
| Declared anchor exception | Exempt only explicitly annotated required contact zones from excessive-facade penalty; preserve cap outside them | Masks independent of target generator, scene hashes, facade-blanket negative controls, no blanket guide exemption |
| Physical material budget | Express allowance in explicit volume/geometry terms rather than fraction of arbitrary envelope | Per-scene physical specification and rationale, all-nine-term bounds, no post-hoc budget chosen merely to pass |

Recommendation: first test a declared anchor interpretation for architectural
material generation, while retaining material connectivity as access. Required attachment should not automatically count as
unwanted facade occupation. This is a proposal, not an implemented exemption.
Use a one-factor comparison before combining changes. Evaluate all18 existing
scenes, keeping the sealed reference, plus separately versioned adversarial scenes.

Stage A gate: intended failures fail; at least one plausible nontrivial constructive
witness per declared feasible scene meets the agreed semantics or has documented,
accepted tradeoffs; mandatory geometric requirements do not contradict the budget.
Do not demand that all regularizers equal zero. Keep hard constraints distinct from
preferences and report raw residuals, rather than a single opaque success score.

## Stage B: calibrate a fixed small objective recipe

Only after Stage A, freeze the new contract and its rationale. Use direct probes
below, within and above the chosen material budget; include empty, diffuse, thin,
anchored, floating and thick fields. Measure all nine raw values and occupancy
and actual NCA parameter gradient norms/cosines. Use every calibration scene
deterministically, several firing seeds and short/long horizons. Pre-clamp coverage
remains an explicit candidate and needs a saturation/overshoot check.

Audit and port only the retained regularizers under separate version names:
TV, density/binarization as actually defined in the original notebook, and the
support-related cantilever term. Historical fine-tuner density is an upper-density
penalty, not binarization; do not equate them. Verify which notebook recipe trained
the checkpoint before porting. Check ground/support boundary handling; the historical
cantilever expression can penalize material simply because nothing lies below z=0.
Exclude unrelated porosity/surface objectives unless separately requested.

Choose fixed coefficients with a written rationale. Do not divide by zero/inactive
norms or use equal loss values as calibration. Keep a small bounded coefficient
sensitivity comparison, a held-out diagnostic set, per-family residuals and stable
scale across scene sizes. Freeze coefficients before learned-value evaluation.

## Stage C: E2 learned-value comparison

Preregister run size only after runtime/memory profiling. All arms share scene
inputs, interpretation, material budget and independent evaluation:

1. Procedural legal guide and the selected constructive target as no-training
   controls, with their inference cost recorded.
2. Direct occupancy optimization as a per-scene upper-effort comparator. Report
   optimization steps/time and do not present it as generalizing inference.
3. NCA fine-tuning from the original checkpoint with the same objective contract.
   Start with unchanged architecture; record pre-clamp guidance as an explicit arm.
4. A no-update checkpoint baseline under the same rollout controls.

Use paired evaluation seeds and holdout scenes. Existing18 scenes have all been
inspected during development; they are regression/calibration data, not a pristine
test set. Create a separately versioned holdout manifest before viewing outputs.
Evaluate beyond training horizon, missing/damaged seeds, changed geometry and
threshold sensitivity. Compare distributions and per-scene failures, not only means.
No claim of architecture improvement from a single training seed or easier scenes.

Stage C gate: reproducible improvement over procedural and checkpoint controls on
the agreed architectural measures, no hidden legality/ground violations, stable
longer rollouts and transparent cost. If NCA does not add value, report that and
reconsider the model concept before scaling. Architecture changes follow this
baseline, one at a time, with matched resource/evaluation budgets.

## Delivery and compute

Local audits and tiny controlled CPU tests can continue within the existing
scope. Any paid pilot needs a concrete configuration and explicit compute cap.
Extend/check recovery for the real GPU trainer: CUDA RNG, optimizer/scaler,
scheduler, sample/pool state, update counters and resume metadata. Use the exact
approved artifact locations. Every Drive operation still needs specific approval.

Product work should next use these result contracts: distinguish material from
circulation, show raw objective tradeoffs and failed/infeasible cases, and make
baseline comparisons inspectable. The studio redesign can proceed on saved
fixtures once this distinction is decided; it need not wait for a large GPU run.
Archive every attempt and update RESUME.md before ending a milestone.

## A1/W1 implementation update

User accepted proceeding on the recommended material-generation scope. Read
FACADE_FINDINGS.md and D026-D028. The scene-derived facade allowance is implemented
and measured. Radius6/envelope plus facade_endpoint_v1 is selected only as the
experimental baseline for calibration preparation. W1 gives a connected zero-loss
witness on each of17 feasible scenes; the sealed reference remains invalid.
The numerical consistency check is now supported by examples, not only bounds.

Proceed to Stage B's actual parameter-gradient/regularizer measurements, then a
small preregistered matched E2 comparison. Retain the W1 constructive baseline
and original-facade ablation. Architectural quality validation is still required;
a trivial procedural zero-loss output sets a baseline rather than proving value.
Do not restart Stage A or request the same scope approval again unless new evidence
or user steering changes the decision. No paid training has been authorized.


## K1/R2 completed; K2 prepared - 2026-09-23

Read CALIBRATION_FINDINGS.md and D031. Actual gradients and regularizer provenance
are now measured: K1 has 71 model cases and 51 budget probes; 141 tests pass.
The historical cantilever skips the bottom three layers; REGULARIZER_AUDIT.md
supersedes any earlier suggestion that it directly penalized cells at z=0.
Research-objective CPU recovery passes all seven exact checks in R2.

Stage B coefficient selection remains open. K2-sensitivity.json freezes an
isolated sparsity-weight comparison (30 versus 3), two seeds and 17 updates per
run, 16-step rollouts, 900-second CPU cap per run. It is prepared, not executed.
Implement/test the actual K2 loop and run it locally next. Keep no-update and W1
controls, all per-family failures, and separate fresh holdouts for later E2.
No evidence yet warrants scaling, architecture changes or a better-model claim.
Studio work and larger environment diversity remain planned milestones.
