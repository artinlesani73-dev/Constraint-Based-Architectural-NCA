# G9 TRAIN-only diagnosis and selected next change

The evidence supports two interacting mechanisms: inefficient use of the
whole-cube allowance, and a direct conflict between G9's symmetric ranking term
and useful mass growth. Do not describe score suppression as the sole or proven
dominant cause. Prepare a one-sided ranking gradient as the next controlled
objective change; leave quota, model and data unchanged.

## Scope

Frozen G8/G9 final427 checkpoints on the same45 TRAIN examples, seed-only starts,
64steps,firing2101,CPUfloat32,two threads,unchanged0.5threshold and growth quota.
All90 rollouts and5760 detached admission replays matched the saved fields and
counts exactly. Saved eligible proposal scores and firing at every step.
No optimizer updates, held-out inference, altered generation policy or paid job.

Captured actual logits with a read-only forward hook at every step. At fixed
steps1,8,16,32,48,64, decomposed instantaneous gradients from membership,
weighted volume, band and ranking:540 snapshots total. State, eligibility and
firing are held fixed. These are logit gradients, not parameter gradients,
and not a causal estimate of retraining. G8 and G9 follow different trajectories;
their groups are not identical matched states. Within EACH saved state, base
versus combined-gradient comparisons are paired exactly.

Context-graph progress in allowance-analysis.json and TRAIN-teacher-graph
progress in gradient-records.json have different meanings and are kept separate.
Neither graph or target is passed to inference.

## What slows growth

Both models first grow at step1 in all45 TRAIN cases. Their unchanged quota has
enough theoretical capacity to reach the request by64 after that start.
Connection at64: G8=42/45, G9=41/45; this is a TRAIN diagnostic, not a new
nine-family acceptance benchmark. TRAIN outcomes do not erase G9's previously
recorded fresh-sample gains.

Before the global cap truncates the per-step allowance, excluding seed steps:

| Model | Unused allowance | Fraction | Unused alongside cube rejection | Unused without rejection |
|---|---:|---:|---:|---:|
| G8 | 1166/46460 | 2.51% | 1057 | 109 |
| G9 | 1638/46591 | 3.52% | 1320 | 318 |

G9's1320/1638 unused cells (80.59%) occur on
steps with allowance rejection. This describes co-occurrence: it does not prove
that packing caused80.59% of underfill, or that enlarging the quota would fix it.
Remaining318 occur with no rejection (52 from firing missing the few offered
possibilities,266 after available offers are exhausted). Neither model has a
full-quota step with every eligible score below threshold in this diagnostic.
Unused opportunities across steps are not distinct final missing voxels.

G9 has472 more unused allowance cells;263 of the difference accompanies rejection
and209 accompanies non-rejection. Paths, numbers of full-quota steps and frontier
shapes differ. No single percentage isolates the effect of ranking or quota.

## Direct objective conflict

Among G9's95 sampled pre-connection, below-cap states with both fired
teacher-positive comparison groups, adding the symmetric ranking term changes
the original base gradient from raising to lowering other useful teacher cubes
in94 states. It flips836 of1917 such logit entries (entries across snapshots,
not unique voxels). G8 saved states show1915/2680 flips in113/113 comparable
snapshots when the G9 term is applied counterfactually; G8 was NOT trained with it.

This directly establishes a local loss conflict. It does not establish that all
these cubes should be added now, nor that gradient conflicts explain every
observed underfill or missed endpoint. Ranking was designed to prefer connection
progress; some tradeoff is intentional. G9's learned lower probabilities and
global allocation depend on shared parameters, hidden dynamics and hard admission.

## Selected change: one-sided access ranking

Prototype: access_ranking_one_sided.py. Keep margin1,weight1 and all G8/G9
membership/volume/band terms. Let A be fired advancing teacher origins and O
other fired teacher origins. Use the same numerical ranking value:

    L = softplus(1 + logmeanexp(stop_gradient(z[O]))
                   + logmeanexp(-z[A]))

Only the comparison-group scores O are detached. This is an explicit
semi-gradient learning rule, not the ordinary full derivative of the symmetric
scalar expression. It keeps the G9 push toward advancing cubes without an
auxiliary push downward on O. Base loss still trains all origins. Return zero
for empty groups, connected phase, or no-route fallback as before.

On all540 saved snapshots the prototype exactly retains G9's numerical ranking
value within tolerance, matches its advancing logit gradients, gives zero
auxiliary gradient elsewhere, and preserves base gradients outside A exactly.
No new inference or optimization was needed for this check.

This verifies the intended local gradient change only. Shared network parameters
can still change other scores after training; no promise of preserved volume,
better packing, connection success or generalization follows. Raising useful
scores may increase competition for a fixed quota. The hard-cap limitation
remains: already allocated mass cannot be removed to repair a missed endpoint.

Do not combine this change with a quota increase, longer horizon, new inputs,
larger model or new teacher distribution. Preserving those controls lets one
later comparison test whether the direct suppression was harmful.

## Concrete next implementation

Integrate one-sided ranking as a separate G10 session using the same427 updates,
64-step horizon,45TRAIN data,paired seed1201,initialization and recovery controls.
Use one consolidated local parity/backward/recovery check, then freeze the exact
package and new evaluation protocol before proposing one bounded Colab run.
The existing57 evaluation requests are now regression evidence. Fresh reserved
geometry must be frozen before new training and compared on identical scenes
for the chosen baselines. No new paid allowance is granted by this document.
Do not repeat a coefficient/seed sweep or quietly weaken acceptance thresholds.

G8 remains the stable research comparison; G9 remains evidence of fresh access
gains with timing regressions; MG7 remains live. No candidate is newly promoted.

## Evidence and continuation

Retained raw90 rollouts, exact checkpoints and identities, dataset/source hashes,
5760 per-step records,540 gradient arrays, term summaries, one-sided prototype
and verification. Original artifacts and results remain intact. Read RESUME.json
here next. A first script-generation edit caused a syntax error before any output
directory or model execution; that failed generated script is retained. The
corrected audit completed once, with no training retries.

Repository synchronization is pending. All archives are same-disk preservation,
not off-device backup. No Drive operation, paid training, push, publication or
live replacement occurred.
