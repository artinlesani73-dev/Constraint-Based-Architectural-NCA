# G6 reserved-scene review — 2026-10-04

G6 passes **10 of 12 cases at both 64 and 128 steps**. The frozen gate requires
12/12 at both horizons, so this candidate does **not** qualify for unrestricted
research generation or live promotion. MG7 remains unchanged.

## Evaluation integrity

The final update-256 checkpoint was frozen before first use of these four scene
variants. Checkpoint SHA256: `883833c5a1171e0969e18ad9c7f9dd8eeb9a80379dba364973684b35531b154b`.
The exact original protocol is copied in next-generalization-protocol.json; its
historical planned status is deliberately retained. execution.json and result.json
record its execution here. All 27 archived TRAIN contexts match reconstructed
seven-channel inputs byte-for-byte, with matching seeds and context hashes.
All frozen Python source fingerprints match. Every 128-step trajectory has an
identical first 64 birth masks to its independently executed 64-step trajectory.

All 12 scene/request pairs were evaluated once per prescribed horizon. CPU
float32, deterministic operations, two threads, firing seed 2101, proposal
threshold 0.5 and fixed quota max(9,ceil((C-27)/63)) were used. No teacher labels,
route input, new training, altered thresholds, retries, dropped failures or
postprocessing were used. No execution errors occurred. All necessary context
feasibility checks pass, which does not prove that every requested design is feasible.

These cases are synthetic relatives of prior scenes, not external architectural
validation. Their first-use results are now consumed evidence: future tuning
against them makes them a regression set, not untouched generalization data.

## Results

| Measure | 64 steps | 128 steps |
|---|---:|---:|
| All nine families | 10/12 | 10/12 |
| Access | 10/12 | 10/12 |
| Each of the other eight families | 12/12 | 12/12 |
| Median absolute volume error, percentage points | 0.161 | 0.167 |
| Maximum absolute volume error, percentage points | 0.180 | 0.180 |

All mass changes are below the frozen 5% tolerance; the largest is
2.056%. Nine fields are identical across horizons and three grow
slightly. All fields have 100% cube-supported mass. The two all-nine gates fail;
the error and stability gates pass. Do not pool these 12 cases with the nine
reused development cases to inflate a generalization claim.

| Case | 64 | 128 | 64-step error (pp) | Mass change |
|---|---|---|---:|---:|
| g1-raised_pair-0-v16 | Pass | Pass | 0.086 | 1.576% |
| g1-raised_pair-0-v24 | Pass | Pass | 0.171 | 0.000% |
| g1-raised_pair-0-v32 | Pass | Pass | 0.157 | 0.000% |
| g1-raised_pair-1-v16 | Pass | Pass | 0.011 | 0.964% |
| g1-raised_pair-1-v24 | Pass | Pass | 0.171 | 0.000% |
| g1-raised_pair-1-v32 | Pass | Pass | 0.157 | 0.000% |
| g1-unequal_building_heights-0-v16 | Pass | Pass | 0.180 | 0.000% |
| g1-unequal_building_heights-0-v24 | Pass | Pass | 0.169 | 0.000% |
| g1-unequal_building_heights-0-v32 | Pass | Pass | 0.178 | 0.000% |
| g1-unequal_building_heights-1-v16 | Access fail | Access fail | 0.157 | 2.056% |
| g1-unequal_building_heights-1-v24 | Access fail | Access fail | 0.157 | 0.000% |
| g1-unequal_building_heights-1-v32 | Pass | Pass | 0.165 | 0.000% |

## Failure interpretation

Both failures are `g1-unequal_building_heights-1`, at 16% and 24% volume. Both
have eight occupied voxels touching the west interface and zero touching the
east interface. They reach across the site's X span but remain below the higher
east connection. The 16% output adds 17 cells after step 64 and reaches its cap
at step 66 without making that contact. The 24% output reaches its cap at step
64 and stays unchanged. More iterations alone cannot repair the saturated
add-only field. The 32% request passes on the same scene.

The frozen access metric starts its flood fill at the alphabetically first
interface, E_east. Therefore both reachability flags are false when east is
untouched. Supplemental direct-contact counts distinguish this from losing
west contact. The metric was not changed.

Every original TRAIN scene has west/east connection origins at z=8/8. Training
varies horizontal position, gap and obstruction but has no vertical connection
offset. That is a verified distribution gap. It supports testing broader
vertical training diversity; it does not prove that this alone will solve the
failure. Data, objective and finite local information propagation may all matter.
The prior destination-cue experiment G5 failed and should not be repeated without
a new mechanism and evidence.

## Visual review

All twelve 64-step geometries were inspected in isometric, front and plan views;
all twelve final 128-step plates were also inspected. The fields are chunky,
stepped building masses with open exterior space around them. They are no longer
one-voxel paths or flat platforms. At 64 steps their vertical bounding extents
range from 5.6 to 12.8 m; these bounds do not
imply uniform thickness. The low-volume failing case is shallower and spreads
sideways, while the 24% failing case grows tall near the west side yet misses the
east connection. Passing geometry still has coarse terraces, irregular additions
and limited demonstrated design diversity. Numerical validity is not architectural
quality, interior circulation, habitability or structural certification.

Final plates are in visuals-final-64/ and visuals-final-128/. The initial plates
in visuals/ had wireframes overlapping captions; they are retained as draft
history, with their rendering script, and superseded by the final layout. The
images use exposed voxel faces and orthographic occupancy projections; they are
not interior sections. Gray wireframes are context, not generated mass.

Thickness, budget and saturation stability are partly enforced by the hybrid
cube-admission algorithm. Do not describe these as independently learned NCA
self-organization. No diversity claim is possible from one firing seed.

## Decision and next implementation

Keep G6 as a frozen research reference. A read-only gallery can show all saved
results with failure labels; unrestricted interactive use and MG7 replacement
are deferred because the admission gate failed.

Next prepare one **training-distribution intervention**: add scene-defined
vertical connection offsets and building-height diversity, keeping the G6 model,
losses, pacing rule, 32-cube resolution and nine families fixed. Use TRAIN-only
teacher construction; freeze the new scene split and compute budget before any
training. Include the original TRAIN cases to limit forgetting. Do not train on
these twelve reserved outputs or use them to select a checkpoint. Freeze fresh
unseen combinations before the next run; label the present twelve as regression.
Check label feasibility and recovery in one consolidated local preparation pass,
then request one explicit Colab allowance. No next training package or paid job
has been launched by this review. Larger grids and deployment polish remain
planned after this specific reliability gap is addressed.

## Preservation and continuation

Saved all 24 fields, hidden states, proposals, birth masks, admission counts,
step ceilings, 12 contexts, exact checkpoint, source/config/split snapshots,
protocol, runtime, metrics and visual plates. The manifest and verified ZIP
preserve this milestone on the same disk; that is not an off-device backup.
No Drive operation was performed. Repository synchronization remains pending;
use this folder's RESUME.json rather than the checkout's older D098 resume.
Original G6 development evidence and the original report remain untouched.
