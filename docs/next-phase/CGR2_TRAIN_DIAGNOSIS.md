# CGR1/CGR2 TRAIN trajectory diagnosis — 2026-09-29

D095. CGR2's main observed failure is rejecting reachable missing cells. Its
stronger restraint reduces excess but also loses correct repairs. Retain CGR1
as the experimental reference and MG7 as the live generator.

## Method and scope

Compared final256 checkpoints on all81 TRAIN examples:27intact,27cube5,27slab2.
Both unmodified target-free rollouts used CPUfloat32,32steps and firingseed2101.
Verified source archive and checkpoint hashes plus dataset integrity. Rebuilt
all firing masks independently and asserted every birth and final field matched
the model's own capture. Saved and hash-verified162 arrays with proposals,births,
initial/target/final fields,hidden state and per-cell opportunity/rejection counts.
No parameter updates, new GPU work, heldout evaluation, threshold/horizon search,
or teacher interventions. Targets used only for diagnosis after inference.
Training-set findings are mechanism evidence, not generalization performance.

## Results

| Metric, across TRAIN examples | CGR1 | CGR2 |
|---|---:|---:|
| Correctly recovered cells |3317|3115|
| Still missing |1187|1389|
| Missing, fired eligible at least once but rejected |1010|1151|
| Missing, rejected at least8times |872|1007|
| Missing, never reached a legal growth frontier |168|231|
| Missing, frontier reached but never fired |9|7|
| Unwanted cells, all examples |1117|745|
| Unwanted cells, intact only |364|227|

CGR2's1151/1389=82.9% remaining missing cells had at least one actual firing
opportunity.1007/1389=72.5% were rejected at least8times.1356/1389 remaining
missing cells had proposal probability<=0.5 at the last sampled step;815 had
probability<=0.25. This is more than a collection of near-threshold misses.
These probability bins include cells ineligible at that step; they are not a
counterfactual estimate of what a lower threshold would recover.

Paired outputs:227correct cells in CGR1 only,25in CGR2 only (net202fewer repairs).
385excess cells in CGR1 only,13in CGR2 only (net372fewer excess). Therefore the
revision suppresses both classes, rather than cleanly separating them.

Cube damage dominates: CGR2 leaves1303missing cube5 cells versus86slab2 cells.
In the last8steps CGR2 adds138correct and68wrong cube5 cells, versus33correct
and85wrong slab2 cells; intact examples accumulate61wrong cells in those steps.
Some repair is still active, so this does NOT prove that extra steps never help.
It does show why simply extending all rollouts risks additional unwanted growth.

All targets are reachable by ideal target-only synchronous six-face expansion
within4steps. This is an oracle geometry check, not a learned or stochastic
runtime guarantee. Wrong additions cannot geometrically block a still-empty
target cell in this monotonic occupancy rule: additions only enlarge occupied
neighbor sets. They can change network inputs and future decisions; this audit
contains no intervention establishing that causal effect.

## Interpretation and next design

The evidence supports a discrimination/training problem on these examples,
not lack of grid space or random firing opportunities as the dominant cause.
It does not isolate the two changed loss terms, nor prove a specific architecture
or credit-assignment defect. No coefficient optimization follows from this audit.

Next prepare a TRAIN-only supervision design that distinguishes hard missing
frontier cells from outward excess. Use CGR1 as the reference; isolate ONE change
rather than simultaneously strengthening intact restraint and a bulk term again.
A concrete candidate for specification is supervised intermediate completion
states: expose the model to partial repairs spanning deep cube damage, with
explicit positive and negative frontier labels and fully intact stop examples.
Keep inference target-free and retain the same architecture/nine families.
This is a proposed training-state curriculum, not an implemented or approved job.
Before any GPU request, define how these states are sampled from TRAIN only,
preserve original on-policy examples to measure distribution mismatch, and verify
that no target mask or damage labels leak into model inputs. Freeze the comparison
and budget. The trajectory diagnosis alone does not establish curriculum benefit.

## Preservation

Artifacts: C:/Users/artin/Documents/Codex/outputs/CGR2-Train-Diagnosis-2026-09-29.
Script: scripts/diagnose_bulk_trajectories.py OUTPUT (fresh directory required).
Small report: experiments/reports/CGR2-train-diagnosis.json. Full source snapshot,
request/runtime/checkpoint identities and all per-case histories retained.
Adjacent milestone archive verified; same disk only, no off-device backup claim.
No Drive access, push, new training or live model changes.
