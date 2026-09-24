# MG5 proposal: account for underfilled thirds during growth

Proposed after MG4 on2026-09-24; not implemented or executed. The143/144 result
narrows the observed gap to one existing coverage check. Preserve all MG3/MG4
fields, seeds, requests, domains and9-family thresholds. No new constraints.

1. Retain the exact completed MG3 route and seeded contact/radial recipe as the
   comparator. Create a separately versioned growth method; do not overwrite it.
2. Compute MT1's three fixed-domain X masks and integer bulk minima using exactly
   the evaluator's voxel-center convention and numerical tolerance. Complete cube
   unions are qualified bulk here; audit this assumption before using counts.
3. Prototype a deterministic priority for facade-feasible additions that reduce
   remaining coverage deficits, with existing seeded radial order as a tie-break.
   Account only for unique new cells. Keep global facade admission and connected
   legal cube growth. Avoid requiring final quotas on the incomplete initial
   route: all three minima may initially be unmet. Preserve finite reconsideration
   and zero-delta transit semantics; never make deficit handling an infinite loop.
4. Keep the requested count and less-than-one-cube overshoot admission unchanged.
   Do not repair coverage by filling arbitrary individual voxels, deleting output
   or silently increasing total volume. Retain stalls and unmet minima explicitly.
5. Before a wider matrix, test exact third-mask/count accounting, overlapping
   cubes, moving priorities, zero-delta transit and finite failure. Freeze a small
   diagnostic comparison including the failed MG4 case, its two passing sibling
   seeds and MG3's repaired partial32%,seed2. No outcome-dependent weight search.
6. If that bounded check supports the hypothesis, freeze a full225-case paired
   replay (MG3's45 plus MG4's180). Require all180 nonpartition validity/fidelity
   passes, preserve all179 earlier valid outputs as valid (geometry need not be
   identical), retain45 blocked failures and all execution/resource bounds.
   Compare seed variation, runtime and memory; do not hide reduced diversity.

MG4 is now inspected development evidence. Passing this combined set would not
be a new blind generalization claim. A subsequent independent site set and the
48/64-grid resource study must be separately specified. Distinguish larger physical
environments at0.8m/cell from finer resolution of the same geometry: a fixed
two-cell interface and rounded cube widths cannot silently carry the same physical
meaning into a resolution change. Audit that representation before such a study.

This proposal does not freeze implementation weights, caps or a runnable MG5
experiment yet. First make the finite priority rule concrete and reviewable,
document it and freeze the evaluation before executing. No paid training, Drive
access or live Studio expansion is required or automatically admitted.
