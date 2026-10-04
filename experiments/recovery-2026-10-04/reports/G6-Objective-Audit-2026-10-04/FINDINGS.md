# Objective and growth-timing audit — 2026-10-04

The evidence shifts the next experiment toward **growth timing**, rather than
another input channel or an immediate loss-weight change. G4 usually already
scores advancing teacher cubes higher than other teacher cubes, but its
admission rule accepts many above-threshold proposals in the same step.

## What was audited

Used the existing final G4 checkpoint on all 27 TRAIN cases, seed-only starts,
firing seed 2101 and a fixed 32-step observation window. Saved 864 step records
and 189 gradient snapshots at steps 1, 4, 8, 12, 16, 24 and 32. Gradients were
decomposed into teacher BCE, weighted local volume and global band terms at
both actual logits and neutral zero logits. State, eligibility and firing were
fixed during each gradient probe. These are instantaneous logit gradients,
not full-rollout parameter updates or causal estimates of training outcomes.
One complete 32-step replay matched the original model's field, hidden state
and count trace exactly. No optimizer updates or held-out evaluation occurred.

“Progress” here means lowering the minimum context cube-graph distance to the
opposite interface. “Other” can include useful target volume; it does not mean
wrong geometry or exclusively sideways movement. Only comparisons with both
teacher-positive groups available, before connection and before the cap, are
used for the matched statistics below.

## Findings

- In 278 matched decision steps, other teacher-positive proposals had a higher
  mean score than progress proposals only 10 times. In 131 steps, both groups'
  mean probabilities exceeded the hard 0.5 acceptance threshold.
- At all 61 matched gradient snapshots, neutral BCE treated both positive
  groups equally. The complete current loss, at actual logits, still encouraged
  the other positive group on average in all 61 snapshots. It teaches eventual
  target membership, not a preferred time to add each target cube.
- Median instantaneous gradient L1 was 0.9723 for BCE, 0.00764 for weighted local
  volume and 0.00653 for global band error. The inspected local gradients do not
  support blaming the global band term alone. These magnitudes are not parameter
  gradients and do not isolate every effect of each term through training.
- Only 2,905 of 17,212 newly added voxels in the matched decision steps belonged
  to strictly advancing cube admissions. This excludes other useful additions
  from the progress category; it is not a fraction of “correct” voxels.
- Across G4's retained training traces, 13,549 of 16,384 steps (82.7%) began at
  the global cap; G5 had 12,301 (75.1%). Growth is impossible in those states,
  although hidden-state updates and gradients still occur. These are not all
  computationally meaningless steps, but training is dominated by capped states.

## One predetermined pacing probe

Set a per-step allowance K=max(9,ceil((C-27)/63)), where C is the unchanged global
ceiling. The first cube still contains 27 cells and follows the original seed
rule. Later steps admit at most K new voxels through the original whole-cube,
overlap-aware score ordering. The floor 9 allows one maximally costly adjacent
cube. Unused allowance does not carry over. The same K applies at 64 and 128
steps; it is never recomputed from the requested evaluation horizon.

In 67 saved states with available progress proposals, pacing increased the
share of added volume assigned to progress in 39 states, tied in 3 and reduced
it in 25. Aggregated progress share changed from681/3994 (17.1%) to386/937
(41.2%), while absolute progress additions fell. A fixed-state replay cannot
establish whole-run success, so the same single policy was also rolled out.
There was no quota search or coefficient tuning.

## Complete TRAIN-only counterfactual

Existing G4 weights were held fixed. Only the admission allowance changed.
Both 64-step and 128-step outputs were retained for every TRAIN case. Original
G4 baseline fields already reached their immutable global cap by step32, so
their saved occupancy is exactly the later baseline occupancy under the
monotone cap. No hidden-state equivalence at later steps is claimed.

| Check | Original G4 | Paced at 64 | Paced at 128 |
|---|---:|---:|---:|
| All nine families pass | 2/27 | 12/27 | 11/27 |
| Access | 2/27 | 22/27 | 22/27 |
| Coverage | 3/27 | 24/27 | 24/27 |
| Facade | 26/27 | 14/27 | 13/27 |
| Each of the other six families | 27/27 | 27/27 | 27/27 |

Only 18/27 paced cases satisfy the 5% mass-stability limit. Maximum mass change
is18.30%. Median volume-fraction error at64 is0.245 percentage points and maximum
3.322 points. All individual scores and geometries, including regressions, are
saved under paced-rollouts. This is not a trained G6 result or generalization
evidence. Neither policy is accepted for deployment.

## Decision and ready next experiment

Prepare G6 as a **pacing-only training change from G4**: same 61-input model,
fresh paired initialization, TRAIN data, teacher stages, losses and firing.
Do not carry G5's extra inputs into this experiment. This isolates the admission
schedule and its induced training-state distribution before adding explicit
task losses. It is a test of a plausible mechanism, not a promised solution.
Facade and stability regressions remain explicit risks.

The new code matches the independent paced rollout exactly for both checked
horizons and all128 step counts. Its loss function is byte-for-byte equal to
G4's function. A packaged CPU rehearsal completed3retained updates and two exact
full-payload/state recovery replays in33.187s. All192step ceilings
and accounting records were verified. Initial weights, row/start schedule and
firing consumption match G4. No paid job was launched during this work.

Next package: G6-Paced-Growth-2026-10-04. One proposed Tesla T4 job, seed1201,256updates64steps,
maximum600controlledseconds; setup/export/download/idle extra. Explicit approval
is still required. No automatic retry. Frozen development gates remain unchanged.
No development or reserved evaluation occurred in this audit. Repeated prior
development use remains a limitation to disclose in the next review.

All work is documented and locally archived. The repository checkout has not
been synchronized and its older RESUME is stale. Use the latest package's
RESUME.json. Archives are on the same disk, not an off-device backup. No Drive,
push, publication or live-model replacement occurred; MG7 remains live.
