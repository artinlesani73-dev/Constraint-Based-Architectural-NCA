# G5 final review — 2026-10-04

G5 completed correctly but did not improve the frozen pilot result. It passes
all nine families in **1 of 9** reused development cases, versus **2 of 9** for
G4. Keep G4 as the stronger cube-growth research reference and MG7 as the live
model. G5 is preserved as a rejected candidate, not discarded or promoted.

## Verified execution

Run `20261004T074101Z_d3a90a77c991` completed all 256 updates on Tesla T4 in
179.311 controlled seconds; worker time was
172.731 seconds. Peak reserved memory was
1446 MiB. All 1,035 payload hashes,
unique archive membership and the exact G5 package identity were verified.
The 12 admission reference checks, union backward check and cue device-copy check
passed. Recovery reproduced the full payload and state exactly at updates 2 and 3.

All 256 saved starts and their hashes match the frozen schedule: 128 seed starts
and 128 cube-stage starts. Sampler order, 16,384 step accounts, final optimizer
counts and each retained field's legality, connectivity, seed retention and
cube support were checked. This is not a replay of every training rollout.
The original full ZIP, receipt and every checkpoint remain preserved.

## Matched comparison

G4 and G5 used identical dataset hashes, all 256 row/start choices, core initial
parameters, final firing-generator state and recorded software versions. G5
adds 128 first-layer parameters for its two context channels, initialized to
zero. Those parameters acquired nonzero weights. This confirms participation
in training, not useful guidance. One seed cannot establish a general causal
effect; the input-shape change may also affect numerical kernels.

Evaluation used final checkpoint 256 only, CPU float32, firing seed 2101 and
the same nine development requests at 64 and 128 steps. There was no clipping,
threshold search, checkpoint selection, extra seed or reserved evaluation.

| Measure | G4 at 64 steps | G5 at 64 steps | G5 at 128 steps |
|---|---:|---:|---:|
| All nine families pass | 2/9 | 1/9 | 1/9 |
| Access | 2/9 | 1/9 | 1/9 |
| Coverage | 6/9 | 3/9 | 3/9 |
| Thickness | 9/9 | 9/9 | 9/9 |
| Each of the other six families | 9/9 | 9/9 | 9/9 |
| Median volume error, percentage points | 0.152 | 0.152 | 0.152 |
| Maximum volume error, percentage points | 0.166 | 0.166 | 0.166 |
| Median teacher IoU | 0.4575 | 0.4997 | 0.4997 |

Both candidates pass the volume-error and stability gates and fail all-nine
validity at both horizons. All G5 fields are identical between 64 and 128 steps.
The one valid case is `g1-offset_interfaces-y4-v32`; G5 loses G4's passing
`g1-offset_interfaces-y2-v32` case. Every 16% and 24% request now fails coverage.
Improved teacher overlap therefore did not translate into better task success.

Thickness is imposed by complete-cube additions. Budget control and stability
at full capacity are also enforced by the transition; they are not evidence
of independently learned self-stabilization or general architectural quality.

## Saved-output diagnosis

All nine outputs retain physical west-interface contact. Only one reaches the
east interface. All nine exhaust capacity by steps
11–20. Further steps cannot
redistribute already occupied volume because this transition only adds cubes.

| Case | Occupied voxels | First cap step | Legal voxel moves to east | Failed families |
|---|---:|---:|---:|---|
| y0-v16 | 876 | 11 | 6 | access, coverage |
| y0-v24 | 1310 | 14 | 4 | access, coverage |
| y0-v32 | 1744 | 17 | 2 | access |
| y2-v16 | 874 | 12 | 6 | access, coverage |
| y2-v24 | 1307 | 14 | 4 | access, coverage |
| y2-v32 | 1740 | 17 | 1 | access |
| y4-v16 | 871 | 14 | 6 | access, coverage |
| y4-v24 | 1302 | 17 | 3 | access, coverage |
| y4-v32 | 1733 | 20 | 0 | PASS |

Distances are supplemental legal-voxel graph diagnostics, not whole-cube repair
costs. The frozen evaluator starts reachability at E_east; missing east contact
makes both reachability flags false even when west is physically touched.
Direct contact is recorded separately without altering the frozen score.
The pre-admission all-offer candidate remains a diagnostic, not an unguarded
rollout or a cap-removal experiment.

## What this means for the next step

The distance-input intervention alone is insufficient in this run. Avoid another
paid feature variant or a longer run before examining the training signal.
Code inspection confirms that the active objective contains teacher-origin BCE,
local mean-volume error and global volume-band error. It has no explicit loss
for interface connection or coverage of the fixed site thirds: those outcomes
are encouraged only indirectly through the teacher shapes. This mismatch is
an actionable hypothesis, not proof of the sole failure cause.

Next, audit early growth decisions and objective gradients on TRAIN contexts
using G4 as the reference. Check whether useful progress toward the opposite
interface competes with rapid lateral filling under the current loss. Design
one task-aligned access or allocation objective only after this local audit,
with explicit gradient and failure checks. Preserve the nine families, current
cube representation and budget, and do not encode a teacher route at inference.
No G6 implementation, training configuration or paid allowance is frozen here.

The development set has been reused and must be described that way. Keep
reserved labels unopened for later generalization review; do not tune new
coefficients by repeatedly re-scoring these nine cases. No deployment follows
from a future development-only pass without its planned generalization review.

## Preservation

All observations, source fingerprints, paired checks, decisions and continuation
instructions are saved locally. Earlier G4/G5 preparation and experiment files
are unchanged. Repository synchronization is pending; the checkout's older
RESUME is stale, so use this folder's RESUME.json. The verified archive is on
the same disk, not an off-device backup. No Drive operation, additional paid
job, retry, push, publication or live-model replacement was performed.
