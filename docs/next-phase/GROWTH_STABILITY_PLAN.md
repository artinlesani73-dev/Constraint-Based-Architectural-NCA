# Next phase: control material growth while retaining the access gain

Prepared from F2 on2026-09-23. This is a plan, not an executed experiment.
Keep the nine families and the3%-12% budget. The two-scene access improvement is
real but incomplete: all final fields exceed budget, and growth duration strongly
changes the result. Larger grids would currently scale an unresolved behavior.

## First: inspect existing models without new training

Freeze a small CPU diagnostic before execution, retaining original/F1/F2 matched
checkpoints on both existing scenes. Profile its cost first using the repaired
elapsed-time guard. Save complete fields and losses at intermediate growth
durations, not just16 and50, and several fixed firing seeds. Suggested durations:
16,24,32,40,50,64; seeds0,1,2. These are proposals until the matrix/caps are fixed.
Do not choose a favorable stopping step after seeing results and call it success.

Measure continuous mass, binary connectivity under both definitions, all remaining
families/regularizers, change in geometry between steps and sensitivity to firing.
Look for a range where connectivity and budget hold together, not a single lucky
frame. Keep disconnected, over-budget and under-floor cases in every denominator.
This tests behavior on development scenes; fresh holdout geometry is still needed.

Then use a bounded set of actual parameter-gradient probes to separate access,
coverage and sparsity contributions. F2's two connected16-step fields still have
positive access loss; test whether further access pressure conflicts with reducing
mass. Raw-field derivatives alone cannot answer that question. Report zero norms,
clamp saturation and clipping without assuming causation. Compare with F1 and the
existing D1/W1 feasible material controls; do not initialize from solved targets.

## Choose one learning change from the evidence

- If later growth consistently destroys acceptable states, test longer/variable
  growth exposure or scene-paired state-pool persistence. Change one training
  mechanism at a time; retain the existing access definition and other objectives.
- If access keeps pushing material after a sufficient connection while mass
  gradients lose the tradeoff, test one explicitly specified access-margin or
  weighting intervention. Preserve the exact bottleneck metric, binary threshold,
  original scores and material budget; do not turn changed scoring into success.
- If the model still cannot fit feasible geometry with correctly exposed losses,
  isolate conditioning or perception capacity. Current64-update results do not
  establish convergence or prove that the NCA concept must be abandoned.

Each chosen trial needs a frozen control, fresh recovery gate, update/compute
accounting and several seeds for a promising finalist. Reuse completed evidence
only with explicit parity/provenance. No simultaneous objective, architecture,
scene-distribution and resolution change. No automatic paid Colab job.

## Keep the larger project moving

The design workspace can develop around the stable scene/result contracts and
saved experiments: show alternatives, failures, budgets, growth history and run
provenance clearly. Preserve NCA-Studio-Concept.html; implement actual new product
files separately. A polished interface must not label the current outputs valid.
Worker cancellation, reproducible jobs and compact instanced rendering remain
the deployment priorities. The existing local viewer has only data/syntax QA;
browser visual/interaction verification remains unresolved.

Scale32 cubed to64 cubed only after profiling and a meaningful joint-validity
baseline. More diverse environments must preserve the same constraint families
and include genuinely new held-out cases. GPU-compatible access and checkpoint
recovery, an explicit Colab budget, and specifically approved Drive operations
are prerequisites for cloud training. No user setup is needed for the next local
growth diagnostic.
