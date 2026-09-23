# Next diagnostic: align access measurements before changing architecture

Prepared during F1 on 2026-09-23 after observing completed saved checkpoints.
This is a plan, not a new objective version, approved model or executed ablation.
Read FITTING_FINDINGS.md for the complete F1 outcome and uncertainty.

## Concrete issue

`context_from_scenes` chooses the first legal voxel in the first entrance as
the differentiable access source. `endpoint_connectivity` accepts the occupied
part of that entrance region, provided it is one connected component. These
are different success contracts. In completed F1 examples the region is connected,
but the fixed source voxel is empty and the soft access loss remains exactly1.
The source choice originally avoids accidentally starting several independent
routes. Simply seeding every occupied entrance voxel could reintroduce that defect.

## Bounded next work

1. Replay saved F1 fields and classify fixed-source closure, source fragmentation
   and finite-hop limitations separately. Preserve v1 scores; add versioned
   diagnostics. Check the frozen source coordinate, raw/material value, occupied
   region, binary connectivity and soft-reach strength on every evaluation.
2. Trace actual access and coverage parameter gradients at original and final
   checkpoints using identical firing masks. Distinguish a flat projected access
   path from a nonzero pre-clamp coverage path, gradient conflict and recurrent
   saturation. Raw saturation counts alone are insufficient for causation.
3. Freeze a shared entrance contract before changing formulas. Entrance regions
   should have an explicit interpretation. Compare fixed-anchor semantics against
   a single connected source-component interpretation; do not silently mix them.
   Fixtures must cover an empty fixed voxel with a connected remainder, separated
   occupied source pieces, empty endpoints, long detours, multiple destinations,
   and ground/facade entrances. Reject any method that counts several independent
   origins as one connected result. Keep all nine families.
4. Replay candidate semantics on saved D1/W1/K2/F1 fields before training. Report
   which cases change, why, and the gradient consequences. Reconcile evaluation
   and optimization without inflating the material cap or hiding failed cases.
5. Only then freeze one local training intervention with the same architecture,
   scene exposure, seeds and compute policy. Existing F1 remains the old-contract
   comparator; no retroactive renaming of its objective or results.

After this, revisit scheduling, stable longer rollouts and fresh geometry
holdouts. Persistent scene/guide conditioning or multiscale perception remains a
separate architecture experiment if the aligned baseline still warrants it.
Do not combine a loss repair, learning-rate change and added channels in one run.

No paid compute, Drive operation or production change is needed for this audit.
GPU recovery, an explicit compute cap and approved backup operations remain
required before paid Colab work. Larger environments and the design workspace
remain planned milestones; F1 is a diagnostic, not the promised deployment upgrade.
