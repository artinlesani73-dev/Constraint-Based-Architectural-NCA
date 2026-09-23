# A2 findings: consistent access scoring restores a missing learning signal

2026-09-23. Run `20260923T123641Z_98bf30045a6f`, source `fac46ff`.
Protocol: [ACCESS_AUDIT_PROTOCOL.md](ACCESS_AUDIT_PROTOCOL.md). Full tables:
[A2-access.md](../../experiments/reports/A2-access.md). Detailed fields, sources,
critical coordinates, gradient norms and cosines:
[A2-evidence.json](../../experiments/reports/A2-evidence.json).

The earlier entrance mismatch is now supported by actual parameter-gradient
measurements. The old access loss supplies zero parameter gradient in all12
tested model cases. The experimental component-based definition supplies a
nonzero access gradient in all four fitted models at their16-step training
horizon. This justifies a controlled access-only training comparison before
changing the model architecture. No weights were updated in this audit.

## Candidate contract and its limits

`component_bottleneck_v2` requires ONE six-connected material component to
intersect every entrance region. Disconnected pieces cannot collectively pass.
Its strength is the largest occupancy threshold at which such a component
exists; access loss is1-strength. A separate binary BFS checks the same predicate.

This changes three details within the access family: a region/component source
instead of one fixed voxel, worst-destination strength instead of a mean, and
unbounded spatial reach instead of64 hops. Intermediate replays separate these
effects. Other eight families, three regularizers and coefficients are unchanged.
An unrelated fragment in a source entrance does not invalidate a different
component that itself touches all regions. That is an explicit semantic change
from the old evaluator's refusal to score fragmented source-region occupancy.
No real replay case here had that ambiguity; synthetic tests cover it.

The implementation selects topology using detached CPU union-find, then gathers
the critical voxel from the live tensor. Its derivative is piecewise linear
almost everywhere, with deterministic tie selection. It is not a smooth
relaxation or a GPU implementation. A clipped zero critical voxel may still
transmit no useful parameter gradient. No claim of architectural usability,
walking clearance, strength or structural safety is added.

## All277 saved fields replayed

| Saved results | Cases | Access loss lower | Access loss higher | Binary labels changed |
|---|---:|---:|---:|---:|
| F1 repeated fitting | 56 | 8 | 0 | 0 |
| K2, original checkpoint and W1 controls | 187 | 0 | 30 | 0 |
| D1 final direct fields | 34 | 0 | 0 | 0 |

All eight F1 reductions arise from allowing the connected entrance region rather
than requiring the empty fixed voxel. All30 increases in the earlier comparison
come from worst-destination scoring: a weaker entrance cannot be averaged away.
Removing the hop limit produces no improvement in this saved set. No fixed-point
binary connection required more than64 hops. These are measurements of the
existing fields, not claims of better generated geometry.

The three connected F1 outputs now have zero access loss, while their material
budgets still fail exactly as before. All17 W1 controls retain zero access loss.
Every binary connected/disconnected label agrees with the original evaluator
on these277 fields. Do not relabel historical scores or replace their records.

## Actual gradients: where the correction matters

The audit replays original Model C on both difficult scenes at16/50 steps and
all four final F1 models at16/50 steps, using the same firing seed2. Every raw
and projected field matches its saved result exactly; model weights remain fixed.

| F1 final model,16 growth steps | Old access loss | Candidate loss | Old parameter norm | Candidate parameter norm | Candidate vs coverage cosine |
|---|---:|---:|---:|---:|---:|
| Ground-pair, weight30 | 1 | 0.914526 | 0 | 11.5495 | 0.7124 |
| Minimal-smoke, weight30 | 1 | 0.788892 | 0 | 18.6945 | 0.8030 |
| Ground-pair, weight3 | 1 | 0.560787 | 0 | 7.3706 | 0.7820 |
| Minimal-smoke, weight3 | 1 | 0.508366 | 0 | 11.5769 | 0.7770 |

Positive cosines mean the raw candidate-access and coverage gradients broadly
agree at these checkpoints. They do not predict an Adam update or prove a better
training result. The change is substantial: cosines between the full old and
candidate objective gradients are0.116,-0.174,0.351,-0.054 in the table's order.
Two are negative. Budget interactions and clipping therefore need measurement
in a controlled learning experiment; do not choose a new coefficient from this
single diagnostic.

At all four original-checkpoint cases, BOTH access definitions have zero
parameter gradient; pre-clamp coverage remains nonzero. The candidate therefore
does not solve initialization from a fully disconnected state by itself. Existing
coverage guidance is still the measured way to start growing toward connection.
At the four final50-step cases the candidate parameter gradient is also zero:
three already have zero access loss, while the remaining case is disconnected.

The old access loss can have a nonzero derivative with respect to the last raw
material tensor while its parameter gradient is zero. Those are different
quantities: the recurrent update, firing and clamps lie between the raw tensor
and the weights. We measured both rather than assuming a raw-field gradient is
a model learning signal. This audit does not uniquely attribute every blocked
path to one recurrence step or operation.

## Decision and next experiment

Keep the candidate opt-in and promote no model or production default. Follow
[ACCESS_TRAINING_PLAN.md](ACCESS_TRAINING_PLAN.md): implement an explicitly
versioned access-only training arm with the original architecture, two scenes,
both existing recipes and proposed F1-matched64 updates. Demonstrate exact
baseline parity before reusing F1 controls, verify actual-loop checkpoint recovery,
and admit the run only after a fixed timing pilot. Preserve both definitions in
evaluation and judge joint connectivity/material-budget outcomes on the same
field. A lower loss after rescoring is not a learned improvement.

The measured original-state limitation remains important: the new objective is
most promising once coverage has produced a weak connection. State pools,
conditioning, longer training horizons and larger sites remain separate future
experiments. GPU/Colab implementation and recovery still need preparation.

## Verification and preservation

- Regression `20260923T123338Z_6b51ce24b725`:158 tests passed, zero failures,
  errors or skips, original-checkpoint smoke passed.
- A2 completed277 replays and12 gradient cases in397.78s, below the900s cap.
  No failures/timeouts and zero optimizer updates. Replay worker14.89s; gradient
  workers13.48-50.53s, including process startup and evidence preservation.
- All277 candidate scores and independent binary BFS results recomputed.
  All72 parameter vectors,72 last-raw gradients and432 cosines verified from
  saved arrays;27 source hashes match the registered source snapshot.
- Autograd derivatives were recorded and their numeric summaries checked; this
  is not an independent reimplementation of backpropagation. Synthetic unique
  bottleneck finite differences and randomized threshold/BFS checks also pass.
- Registered source fields, checkpoint hashes, old scores, attribution variants,
  critical coordinates and all outputs remain local and immutable. Adding
  nca/access.py changes older runners' source inventories: use their exact source
  snapshots for checkpoint recovery rather than bypassing metadata checks.

No paid compute, Google Drive operation, production checkpoint change, deployment
or remote push. The earlier local viewer remains unchanged and retains its
previously documented browser visual-QA limitation.
