# MD1 findings: procedural improvement; direct pilot does not improve geometry

2026-09-24. Run 20260924T123453Z_aaf2d7ad37e3, regression 20260924T123246Z_90e81143f1df. Follow D058/D064 and
MASSING_DIRECT_PILOT_PLAN. Completed experiment, negative primary direct result.
No execution failure, interruption, coefficient sweep or post-result retuning.

## Result

Contact-aware procedural generation passes 4/4, versus 2/4 for original MG1 and
2/4 after 32 direct updates. Both partial-obstruction failures are repaired by the
procedural control; both aligned positives remain valid. This is four development
examples, not evidence of general reliability or a trained-model improvement.

| Saved case | Method | All nine | Cells | Volume / domain | Facade contact / mass |
|---|---|---|---:|---:|---:|
| aligned__v24__s0 | Original | Pass | 847 | 24.255% | 11.334% |
| aligned__v24__s0 | Contact-aware | Pass | 846 | 24.227% | 8.747% |
| aligned__v24__s0 | Direct32 | Pass | 847 | 24.255% | 11.334% |
| aligned__v24__s1 | Original | Pass | 846 | 24.227% | 10.284% |
| aligned__v24__s1 | Contact-aware | Pass | 839 | 24.026% | 6.913% |
| aligned__v24__s1 | Direct32 | Pass | 846 | 24.227% | 10.284% |
| partial_obstruction__v24__s0 | Original | Facade fails | 822 | 24.091% | 19.100% |
| partial_obstruction__v24__s0 | Contact-aware | Pass | 819 | 24.004% | 13.065% |
| partial_obstruction__v24__s0 | Direct32 | Facade fails | 822 | 24.091% | 19.100% |
| partial_obstruction__v24__s1 | Original | Facade fails | 820 | 24.033% | 19.146% |
| partial_obstruction__v24__s1 | Contact-aware | Pass | 819 | 24.004% | 6.838% |
| partial_obstruction__v24__s1 | Direct32 | Facade fails | 820 | 24.033% | 19.146% |

All outputs in this table have 100% binary cube-qualified bulk. The contact-aware
partial outputs each use 819 cells (24.0035% of their 3412-cell domain), so their
contact improvement is not achieved by inflating the mass denominator. Their
nonexempt contact counts are 107 and 56 versus 157 in both original fields.
Seed 0 changes 108 added / 111 removed cells; seed 1 changes 212 added / 213 removed.
The requested volume and all thresholds remain unchanged. Existing MG1 stays intact.

## What failed in direct optimization

The exact thresholded field equals its original at every saved boundary
(0,8,16,24,32) in all four members. We did not save a full probability field at
every intervening update, so this statement is specifically about saved boundaries.
All 128 actual updates and their finite gradient summaries are retained.

For partial-obstruction seeds 0/1 the weighted continuous loss drops from
0.272157/0.273175 to 0.090180/0.080259. Continuous facade residual reaches zero,
while binary facade contact remains 19.100%/19.146%, above 15%. Final requested
continuous fractions are 23.850%/23.849%; at step 8 they were 29.772%/29.728%.
Thus both probability redistribution and transient volume drift matter. Do not
interpret the soft zero as a valid voxel form or report only reduced loss.

Final original occupied cells still have probabilities at least 0.69128/0.69153
in the failing cases; all originally empty cells are at most 0.03292/0.03333.
They remain well away from crossing 0.5. This is consistent with saturated
initialization and a soft objective reducing penalties without discrete edits;
it does not isolate which change would fix the issue. Aligned final losses
increase from 0.113903/0.113651 to 0.125105/0.128207 despite unchanged valid geometry.
The objective is nonsmooth, and these observations do not establish convergence.

MO1's binary-endpoint agreement was useful but insufficient to validate fitting.
It remains correct for its audited endpoints; it never promised soft-to-binary
agreement. This MD1 recipe is not admitted for NCA training. Do not extend steps,
retune weights, change the threshold or replace the final result with a favorite
intermediate as an automatic response. Independent logits are not an NCA;
this negative result neither rejects all NCA architectures nor validates one.

## Mechanics and measured costs

- 303 regression tests pass, zero failures/errors/skips; real checkpoint smoke passes.
 Six new focused tests cover projection, input rejection, actual update replay,
 restore identity rejection, zero-cost original parity and procedural determinism.
- Fresh worker recovery on partial seed0: uninterrupted4 versus 2 + reload + 2 matches
 the entire checkpoint (Adam/scheduler/RNG/logits included), evaluated geometry
 and probability array exactly. Recovery takes 13.048s including process launches.
- Four-update profiles: partial 4.794s, aligned 4.704s external. Doubled extrapolated
 member estimates 22.766/22.823s; conservative pilot estimate 151.178s fits 600s cap.
- Actual 32-update workers take 10.755-11.150s each, including startup, context,
 evaluation and checkpoint writing. Four-case pilot 47.051s, total study 66.880s
 before final evidence attachment/finalization. Timing setup did not overlap regression.
- Contact-aware generation takes 0.154-0.213s/member, including contact-context
 setup, excluding final evaluation. These scopes differ from the full direct
 worker; no precise like-for-like speedup claim. All processes run CPU float64
 for optimization; this is not a GPU/Colab performance result.

## Independent verification and presentation

Run manifests and source ZIP member hashes verify. All 146 relevant Python files
match both regression and experiment snapshots. Verification replays all 20 saved
checkpoints into fresh sessions, matches probability arrays and rescores binary
geometry, and reproduces all four procedural fields/routes/selection traces exactly.
Four no-update fields rescore identically to MG1. This is boundary replay plus
the actual four-step restart experiment, not another full 32-step scientific run.
See experiments/reports/MD1-verification.json for every boundary, metric and delta.

New static gallery: /static/direct/index.html, linked from Studio. All 24 selections
(four originals paired with one procedural and five direct outputs each) are
available. Original on the left, selected method on the right; additions and removed
counts are explicit. Both family tables and volume/contact measures stay visible.
No live arbitrary-scene mass generator or trained-model deployment was added.

Browser checks: 24 selector badges/nine-family rows, vertical/horizontal views,
keyboard slice 16 (13.2m), cutaway and layer controls ; 390px document width 375px,
tables 343.8125px, no horizontal overflow. Browser warn/error logs were empty at
that check. Tall screenshot captures were clipped by the host surface; a settled
1000 × 850 capture displays both volumes. A generic canvas selector failed and was
replaced after refreshing DOM. A checkbox reset after resizing failed; subsequent
DOM retained the prior checked state. Studio navigation link was observed; a later
link/viewport action timed out and reset the browser session. These automation
limitations are retained; no claim that every final navigation action succeeded.

## Bounded next decision

Prefer testing the same contact-aware recipe across MG1's full 45-member matrix
before changing learning architecture. Proposed next phase MG2: same five contexts,
three requests 16/24/32%, seeds 0/1/2, same masks/MT1 and contact weight 12; no rerolls,
no threshold or cost search. Report each baseline-valid regression, every partial
case, blocked context, request error, generation cost and valid-only diversity.
Blocked routing failure remains a preserved limitation, not a feasibility proof.
Freeze acceptance and caps before executing; do not quietly generalize 4/4 to 45/45.

After that evaluation, wire the supported mass generator into a separately labeled
Studio workflow with saved records, cancellation/recovery and these mass semantics.
Keep the old material scaffold and galleries readable as historical experiments.
Revisit a learned method only with a measurable advantage over this stronger
procedural control and an explicit soft-to-binary validation strategy. No paid
Colab task is needed or admitted now. No new constraint family is introduced.
