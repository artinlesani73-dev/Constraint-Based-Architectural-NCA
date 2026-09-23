# Next: diagnose and restore the access learning signal

Prepared2026-09-23 after F3. This is an unexecuted diagnostic proposal, not a
new objective default or permission for paid training. Read D046 and
HORIZON_TRAINING_FINDINGS.md. F3 shrank material but lost all final connectivity.

## Question

Can the existing access family guide recovery when clipping makes the selected
bottleneck zero, without changing the definition of valid entrance connectivity?
F3 has51 strictly negative and21 exact-zero selected raw bottlenecks in72 final
fields. Saved values identify a final-clamp obstruction in51; they do not replace
actual parameter-gradient measurements for the whole model.

## First gate: actual gradients and frozen-field controls

Profile and freeze a bounded CPU audit before execution. Proposed initial actual
gradient matrix: all four final F3 models at16/50 steps with firing seed2 (eight
cases), retaining access, coverage, sparsity and total parameter gradients, raw
derivatives, norms and cosines. Require exact replay of saved raw/material fields.
Separate strictly negative from exact-zero critical raw values, trace previous
clamps/firing masks where necessary, and preserve zero signals as measured results.
Reuse original/F2 H1 evidence only after source/formula/hash checks; clearly label
reused vectors rather than calling them freshly computed.

Use the saved original, F2 and F3 evaluations plus existing feasible W1/D1
controls to check loss values, legal-route status and unchanged binary labels.
Freeze the exact replay set, gradient cases, caps and timing-only admission before
running; do not select only convenient failures. No optimizer update in this audit.

## One candidate to investigate, not yet adopt

Let b_raw be the highest raw-field threshold at which one legal six-connected
component touches all entrance regions. Compute it on finite pre-clamp material
values, using the same legal masks, endpoint regions and deterministic tie rule.
The proposed research-only access loss is `max(0, 1 - b_raw)`.

For a legally connectable graph, monotonic clipping implies the existing
projected bottleneck is `clamp(b_raw, 0, 1)`. Therefore:

- For b_raw>=0, the proposed loss equals the current `1-clamp(b_raw,0,1)`.
- For b_raw<0, it extends the current saturated loss1 to `1-b_raw`, allowing a
  gradient to the selected raw bottleneck instead of stopping at the final clamp.
- Binary occupancy on legal cells at threshold0.5 is unchanged by this scoring
  proposal. Forward projection and binary evaluation remain the existing ones.

These are mathematical properties of the proposal, not empirical success. Test
the identity explicitly, including ties, negative fields, saturation, multiple
entrances, endpoint overlap rejection, disconnected legal graphs and masks. For
physically disconnected legal graphs, retain an explicit infeasibility result;
do not invent a useful gradient or connect through forbidden cells.

Different ordering among clipped-to-zero voxels can change the selected route.
Expose this distinction in results. The new loss can exceed1 and its weighted
parameter gradient may dominate other terms. Measure its conflict with sparsity
and coverage, finite behavior, and an explicitly bounded local derivative check.
Nonzero gradients alone are insufficient evidence to train with it.

## Decision gate before another learning comparison

Keep the candidate opt-in. If label/value invariants and actual gradient checks
pass, propose one access-family change against the constant16 F2 control. Keep
initialization, recipes, architecture, scene distribution and update exposure
fixed; do not simultaneously adopt F3 horizons, a new margin, conditioning or a
state pool. Establish baseline parity, real-loop recovery and a measured allowance
before training. Compare joint connectivity/budget, not loss reduction alone.

If useful parameter gradients still cannot reach the failed geometry, use that
evidence to select one conditioning/perception intervention. This diagnostic
does not yet justify an architecture change, larger grid or longer paid run.
Preserve all failures and decisions; private reports remain local/Git-ignored.
Every Drive operation needs its own exact folder-scoped approval.
