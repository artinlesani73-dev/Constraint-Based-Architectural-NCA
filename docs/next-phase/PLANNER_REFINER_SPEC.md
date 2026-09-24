# Planner and learned refinement — next research specification

2026-09-24. Design proposal following D053; implementation boundary D054.
Architectural material/form generation remains the accepted D026 scope.
No new model is trained or promoted by this document.

## Why the role changes

F5 reached 66/72 connected final evaluations but 0/72 within the material
budget. W1 constructed joint connectivity/budget witnesses on all 17 feasible
development scenes. Direct optimization also outperformed the studied NCA on
joint outcomes, with different compute and parameterization. These are local
development observations, not evidence that an NCA cannot generalize.

The planner already solves much of this particular connectivity problem.
Learning must add measurable value beyond replicating a shortest route. A
beautiful render, lower aggregate loss, or more occupied voxels is insufficient.

## Proposed division of work

1. Scene contract and feasibility: validate scene_v1, build the same permitted,
   protected, support and entrance regions; explicitly preserve infeasible cases.
2. Planner: corridor_legal_v1 centerline, radius-six permitted graph envelope,
   facade_endpoint_v1 allowances and budgeted_witness_v1 construction. Independently
   evaluate the result. Construction status alone is not a validity certificate.
3. Refiner (proposed): evolve a material field plus hidden state while receiving
   fixed scene context, planner field and envelope at every recurrent step.
   Start from the actual evaluated planner material, not a tiny initial bias.
4. Acceptance: evaluate the raw refinement independently. A displayed accepted
   alternative must keep the required connectivity, budget and legality and
   expose all nine family values. If refinement fails, retain the failed raw
   field and its diagnostics, and offer the original planner result separately.
   Never silently replace a failed learned field with a repaired one and call
   it a learned success.

The first architecture candidate should keep the existing grid, scalar material
field and small backbone to isolate initialization plus persistent context.
Represent refinement as local material changes to the planner field, with hard
legality projection and explicit recurrent hidden state. Hard legality does not
guarantee connectivity or budget; evaluation and retention are still necessary.
Do not freeze an immutable centerline in the first design by assumption: that
would preclude rerouting after edits. Test locked-core variants only as named
ablations if the recovery task requires them. Do not combine channel scaling,
new losses, a new scene distribution and a multiscale architecture in one claim.

## A useful learning task

Primary hypothesis: a persistent learned local update can restore a previously
generated material form after localized damage or a small scene edit while
preserving more unaffected material than a fresh solve, at useful latency.
This is a hypothesis, not yet a proven advantage. Planner recomputation is fast;
if it wins on quality and latency, use the planner in the product.

Measure four groups separately:

- Validity: joint connectivity/budget, material legality, explicit feasibility
  and all nine existing family diagnostics. Empty output never passes.
- Recovery: success rate after registered damage/edit types, steps and wall time
  to recovery, retained failures and stability after extra recurrent steps.
- Locality: changed material outside the edited region and disagreement with the
  pre-edit field. This evaluates the refinement task; it is not a tenth constraint.
- Practical cost: median and tail latency, peak memory, recurrent updates,
  planner setup time, training cost and total hardware budget.

Material variety is secondary. Report pairwise material differences only among
outputs meeting the geometric gates. Variation due to disconnection or excess
material is not useful diversity. Visual preference requires a separate human
review and must not be inferred from the proxy scores.

## Comparisons that could justify training

Use the exact same submitted scenes and edits for: unchanged planner output,
planner recomputation, direct voxel optimization initialized from the planner,
and planner-initialized NCA. Include a simple local procedural repair where
applicable. Report both equal wall-clock and equal update comparisons when those
are meaningful; an optimizer update and a CA step are not equal units of work.

Register a small development set of damage and scene-edit cases before training.
Separate material removal (fixed scene contract) from changed entrances/buildings
(new scene hash and newly derived masks). Prevent leaks by splitting site families,
not merely random seeds of the same site. Existing 18 scenes are development
controls. Create fresh held-out site families before any generalization claim.

Admission to a learning pilot requires:

- A frozen edit/task protocol and non-learning control results, including
  infeasible edits. No need to train if controls already satisfy the desired task.
- A declared improvement target relative to those measured controls, alongside
  a no-regression validity gate; select numbers before observing learned results.
- A frozen configuration, multiple training seeds for a finalist, exact restart
  including optimizer/RNG/hidden-state pool if used, and a timing/memory pilot.
- Explicit paid-compute allowance and an approved, verified second artifact copy
  before Colab training. No Drive action has standing approval.

Stop or change the research question if learning repeatedly fails the validity
gate or does not improve recovery/locality/cost over controls. Do not extend the
old loss-tweaking series by renaming it refinement.

## Product path independent of learning

S1 supplies a local procedural workspace with actual geometry, honest diagnostic
labels, editable scene coordinates and immutable saved records. It is deliberately
separate from the historical generator; no trained-model deployment claim.

S2 should add durable asynchronous jobs with restart state and real worker
cancellation, then a revision-aware comparison view. Record submitted/running/
completed/failed/cancelled states; cancellation must stop compute, not merely
discard a browser response. Introduce export/import verification and stable result
schemas before serving long NCA runs. Later add direct manipulation, orbitable
instanced geometry, sections, larger site families and scaled grids after profiling.

This specification authorizes no paid training, cloud storage action or public
deployment. Those are separate concrete milestones.


## S2 follow-up: spatial prerequisite

2026-09-24/D055. S2 product lifecycle/comparison/import is implemented; read
STUDIO_S2.md. User review highlighted that W1 returns thin connected material,
not inhabitable space. SPATIAL_BRIEF.md now precedes the learning-pilot admission
steps above: define the desired surface/void/material semantics and demonstrate
spatially meaningful procedural controls before freezing a refiner task. Existing
scalar-field and budget assumptions above remain hypotheses pending that audit;
no silent redefinition of historical metrics or new constraint family is accepted.
