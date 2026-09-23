# Loss and gradient findings

Completed locally on 2026-09-23. The shared loss package now passes shape,
batching and numerical-gradient checks, but **the training gate is still open**:
we have measured both envelope/budget conflicts and a real dead-gradient case.
Do not launch Colab training yet.

## Implemented

`nca/losses.py` introduces `geometry_losses_v1`, preserving the original notebook,
fine-tuner and checkpoint. Nine named families return separate per-scene terms.
The API requires distinct coverage and material-envelope masks, keeps the
historical 3%-12% mass bounds and denominator, and refuses invalid batches at
strict reduction. Empty material is flagged separately from context validity.

Thickness uses continuous occupancy with exactly zero empty background and
explicit outside-volume padding. Access uses one fixed legal source voxel and
six-neighbor maximum-bottleneck propagation; support uses fixed geometric support
boundaries. Both have an explicit finite hop horizon. These are spatial material
proxies; street void, elevated decks/headroom and structural engineering claims
are not resolved by this implementation. No new constraint family was added.

## Recorded checks

- Initial regression `20260923T002822Z_8f68d5bbcaf7`: 108 tests passed.
- Final regression `20260923T003247Z_0ce597b29bc4`: **109 passed**, zero failures,
  errors/skips, checkpoint smoke exit 0. Includes batch-one/batch-two value and
  gradient equivalence, directional finite differences, path bottlenecks,
  zero-background erosion and scene-adapter connectivity certification.
- L1 `20260923T003413Z_1da1202e4a7f`: all **72 context checks, three real-model
  gradient checks and six historical fine-tuner checks** completed. 29.25 seconds
  CPU. The real-model checks use four updates and seed 0, with no optimizer.
- All recorded artifacts verify. Independent NumPy calculations from saved
  gradient arrays reproduce parameter/occupancy norms and all pairwise cosines.
  Fresh rendering reproduces the report and complete detail JSON.

## Connection guidance needs an explicit material region

The same saved C1 centerlines were tested with four declared envelopes:

| Material envelope | Meets capacity bounds / 18 | Valid context including a feasible route / 18 |
|---|---:|---:|
| C1 legal thick scaffold | 0 | 0 |
| Fixed three-voxel expansion | 3 | 3 |
| Fixed six-voxel expansion | 13 | 12 |
| All permitted space (diagnostic control) | 18 | 17 |

At 0.8m voxels, those graph radii are 2.4m and 4.8m. Neither radius was enlarged
per scene to force a pass. The intentionally sealed reference remains invalid.
Five feasible scenes still have insufficient capacity with radius six:
legacy seeds 000, 001, 007, 008 and ref-06-minimal-smoke.

A broad envelope makes the old volume floor geometrically possible, but using
all permitted space makes spill redundant. It is not a selected design solution.
Necessary capacity checks also do not prove all nine objectives can be satisfied
simultaneously. The next training specification must explicitly define the
material design region and the role/denominator of the lower budget. Do not
silently lower 3% or widen every region to hide the conflict.

## Gradients exist, but not every loss can teach the current model

On legacy-easy-seed-000, the access term has parameter-gradient norm 0.9004 after
four updates. On ref-01-ground-pair, access loss is 1.0 (complete failure), yet
its parameter-gradient norm is **zero**. In the mixed batch, a nonzero aggregate
access gradient hides this failed scene. This is why per-scene diagnostics matter.

A separate post-hoc run, `20260923T003906Z_da05c8eed3f6`, replays that ground case
bit for bit. The two voxels where the access term has a nonzero occupancy
derivative are legal, fired cells at z,y,x = (0,15,9) and (1,15,9). Their final
unclamped structure values are approximately -0.001280 and -0.009346. Clipping to
zero blocks the final-step gradient to the weights. All replay fields and exact
values are retained. This is a demonstrated local mechanism, not a claim that
every connection failure or every later rollout has the same cause.

Legality and ground have zero parameter gradients because hard projection already
removes forbidden material; that is expected. Thickness is zero and has zero
parameter gradient on these early thin states; synthetic thick-volume fixtures
and finite-difference tests check its separate numerical behavior. Zero norms
are recorded, not replaced with artificial signals.

Coverage and spill have locally opposing parameter derivatives (cosines roughly
-0.40 to -0.58 across these three states). Coverage and sparsity align positively
here. These local cosines do not prove global infeasibility and do not justify
adaptive weighting before objective definitions are settled.

The legacy-alone and mixed-batch gradient examples contain a radius-six context
with insufficient capacity; they were inspected only as individual diagnostic
terms. Strict reduction would refuse them for training. No optimizer was run.

## Historical fine-tuner defects confirmed by execution

At both batch sizes one and two:

- Original ThicknessLoss and PorosityLoss raise forward shape errors.
- Original SurfaceAreaLoss returns an output disconnected from occupancy
  gradients. The historical definitions and errors are archived.

These expected historical outcomes are separate from successful new regression
checks. The original fine-tuning script was not imported as a trainer or altered.
Porosity/surface-area objectives were not added to the nine-family loss package.

## Next bounded implementation step

1. Write the next objective contract with an explicit material region, connection
   guide and mass-budget interpretation. Compare stated alternatives on all frozen
   scenes; do not silently choose the convenient denominator or geometry.
2. Run a small, separately versioned gradient intervention on the ground case:
   compare guidance applied to legal pre-clamp material candidates with a smooth
   material-state alternative. Keep hard legality, scenes and existing evidence.
   These are candidates to test, not approved production behavior. Verify that a
   useful derivative reaches the offending cells and model parameters without
   inventing a straight-through gradient that fails finite-difference checks.
3. Keep continuous proxy diagnostics separate from binary connectivity; check
   longer horizons, seeds and broken-route states before claiming useful learning.
4. Only after those decisions, integrate retained regularizers, calibrate weights,
   and test an interrupted/resumed local optimizer step before a paid Colab pilot.

Read the [full report](reports/20260923T003413Z_1da1202e4a7f-L1.md),
[complete details](reports/20260923T003413Z_1da1202e4a7f-L1.details.json) and
[verification](reports/20260923T003413Z_1da1202e4a7f-L1.verification.json).
Raw evidence is under `.local-artifacts/runs/` for the IDs above. No weights,
production defaults or historical scene/checkpoint files changed. Everything
remains local; no Drive access, paid training, push or deployment occurred.
