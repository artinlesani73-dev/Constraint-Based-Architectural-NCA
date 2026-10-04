# G6 final review — 2026-10-04

**G6 passes all five frozen development gates.** All nine reused development
requests pass all nine constraint families at both64 and128steps. The fields
are identical at the two horizons. Freeze checkpoint256 as the qualified
research candidate; generalization and visual review remain before deployment.
MG7 stays live.

## Verified run

Run `20261004T081739Z_95ddbf523738` completed256updates on Tesla T4 in
199.894controlledseconds, with
192.821workerseconds and
1446MiB peak reserved GPU memory.
All1,035evidence payload hashes,unique archive membership and exact G6package
identity verified. The12original admission probes,8paced probes and union
backward check passed. Full-payload/state recovery matched exactly at updates2
and3. All256saved starts,16,384step ceilings and count records were verified.
This does not mean every GPU training rollout was independently replayed.

G4 and G6 share all initial parameters,dataset hashes,row/start schedules and
final firing-generator state. Runtime versions also match. No new network
parameters or G5distance channels were introduced. This is one seeded pacing
intervention; broader causal or multi-seed reliability is not established.

## Frozen results

Final checkpoint256only,CPUfloat32,firing2101,same9development requests,single
scene-defined seed,threshold0.5. No postprocessing,threshold/horizon/quota search,
checkpoint selection or reserved evaluation. The per-step allowance is fixed
from the64step training schedule and is unchanged in128step review.

| Measure | G4 at64 | G6 at64 | G6 at128 |
|---|---:|---:|---:|
| All nine families pass | 2/9 | **9/9** | **9/9** |
| Access | 2/9 | 9/9 | 9/9 |
| Coverage | 6/9 | 9/9 | 9/9 |
| Facade | 9/9 | 9/9 | 9/9 |
| Each other family | 9/9 | 9/9 | 9/9 |
| Median volume error,percentage points | 0.152 | 0.152 | 0.152 |
| Maximum volume error,percentage points | 0.166 | 0.166 | 0.166 |
| Median teacher IoU | 0.4575 | 0.5407 | 0.5407 |

The five gates are:all-nine validity at64,median error<=2percentage points,
maximum error<=4points,all-nine validity at128,and each mass change<=5%.
All pass. Every field has100%cube-supported mass. Lowest site-third bulk
fraction is0.1050 against0.08minimum;highest facade
contact fraction is0.0731 against0.15maximum.

| Case | Occupied voxels | Per-step allowance | First global-cap step | Facade fraction | Both horizons |
|---|---:|---:|---:|---:|---|
| y0-v16 | 876 | 14 | 64 | 0.073 | PASS |
| y0-v24 | 1310 | 21 | 63 | 0.052 | PASS |
| y0-v32 | 1744 | 28 | 63 | 0.050 | PASS |
| y2-v16 | 874 | 14 | 64 | 0.061 | PASS |
| y2-v24 | 1307 | 21 | 63 | 0.053 | PASS |
| y2-v32 | 1740 | 28 | 63 | 0.056 | PASS |
| y4-v16 | 871 | 14 | 63 | 0.053 | PASS |
| y4-v24 | 1302 | 21 | 62 | 0.051 | PASS |
| y4-v32 | 1733 | 28 | 63 | 0.045 | PASS |

## What changed

G4 exhausted volume early. G6 spreads admissions across the growth sequence.
These development outputs reach the global cap at steps
62–64, after connecting and
distributing mass successfully. The retained training fraction starting at
global capacity fell from82.7% inG4 to
28.7% inG6. This is consistent with the timing
hypothesis,although it does not isolate each effect through learned trajectories.

The prior TRAIN-only pacing diagnostic used fixed G4weights and exposed facade
and stability failures. This result comes from a separate fresh model trained
under pacing. Do not combine its development numbers with that earlier27case
TRAIN diagnostic as if they were one evaluation set.

Column3 of G6admission counts records rejection by the effective per-step
allowance,not solely by the global cap. Quota and every effective ceiling are
saved separately. G4/G6raw rejection counts are therefore not directly
comparable. The all-offer pre-admission candidate remains a diagnostic,not a
no-guard rollout or a causal cap-removal experiment.

## Limits of this milestone

Complete-cube geometry enforces thickness. Budget and saturation enforce size
limits and eventual fixed occupancy; this is not independent evidence of a
learned self-stabilizing dynamical system. Access,coverage and facade passes
are observed under this hybrid transition. The contract concerns overall
building volume and geometric support,not interior layout or structural safety.

The nine development requests have been reused during development. This result
is a pilot gate,not untouched generalization evidence or final deployment
approval. Only one training seed has been evaluated. No visual inspection of
G6geometry is claimed by this numerical review.

## Next phase — frozen generalization and visual review

Keep the exact candidate recorded in candidate-freeze.json. Next evaluate the
four previously reserved scene variants,each at16%,24% and32%requested volume:
12requests,at64 and128steps,with the same seed,threshold,quota and metrics.
No teacher geometry is needed for inference or the nine-family evaluation.
First verify reconstruction of the existing input channels against archived
TRAIN fixtures. Report every case; do not tune or discard failures after seeing
the results. next-generalization-protocol.json records this plan before access.
That evaluation has not been run during this review.

Then inspect saved volumes visually before deciding on research-interface
integration. No additional training or paid GPU job is needed for this next
local evaluation. MG7 remains the live model until that separate decision.

## Preservation

Original evidenceZIP and receipt,checkpoint,source snapshot,all18observations,
pairing checks,training-capacity comparison,candidate hashes and continuation
instructions are saved and archived locally. Repository synchronization remains
pending;the checkout's older RESUME is stale. Use this folder's RESUME.json.
The milestone archive is on the same disk,not an off-device backup. No Drive,
paid retry,push,publication or live-model replacement occurred.
