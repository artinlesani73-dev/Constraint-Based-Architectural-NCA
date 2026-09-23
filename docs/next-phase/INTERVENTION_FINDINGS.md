# Material interventions and recovery: findings

Completed 2026-09-23. The targeted coverage derivative is restored without changing
hard-projected forward states, and a tiny CPU optimizer run resumes exactly across
fresh processes. This is progress in training mechanics, not evidence of better
architectural outputs. All nine constraint families remain represented.

## Evidence

- L2: `20260923T075113Z_d95fabaf3776`, source `95431de`; 108 budget cases,
  54 gradient cases and nine absent-scaffold controls; zero optimizer updates.
- R1: `20260923T075727Z_2233c0e51b9a`, source `6ee2ed8`; four logical updates,
  ten executed updates across four independent process branches; seven exact
  recovery comparisons pass. Every update's loss, fields and checkpoint retained.
- Verification `20260923T074748Z_68688d661ffd`: all 118 tests pass, checkpoint
  smoke exit 0. Its failed predecessor remains archived (fixture type mismatch).
- [Full per-scene and per-case report](../../experiments/reports/L2-R1-interventions.md)
  is reconstructed from hash-verified registered artifacts. Saved gradient norms,
  21 hard-forward pairs, recovery checkpoints/traces/fields were independently
  rechecked. A second in-memory reconstruction matched the saved report exactly.

## What changed and what the comparison means

`material_intervention_v1` adds three explicit experimental arms. The hard control
and pre-clamp guidance have exactly identical forward fields in all 21 matched
pairs. At the two previously blocked ground-case cells, coverage derivatives go
from zero to -1/36. The original projected access derivative remains zero in this
case. Coverage is an auxiliary guide objective within the existing coverage family;
it is not a repaired architectural access metric or a straight-through estimator.

Across the 18 ordinary cases per arm, hard and pre-clamp access gradients are
nonzero in four; smooth-state access gradients are nonzero in all 18. Nevertheless,
all arms connect only three of 18 binary cases, all from the legacy scene at the
longer horizon. Neither reference scene becomes connected. Absent-scaffold hard
coverage has a weight gradient in one of three cases; pre-clamp and smooth have
one in all three. None connects. These are checkpoint forward diagnostics.

Smooth clipping creates positive background occupancy (~0.03466 at raw zero).
For the ground scene, seed zero, four steps, soft mass increases from 51.20 to
2002.80 voxel equivalents while binary material remains empty. With no scaffold,
it increases from 7.35 to 1972.28, again without binary connectivity. Improved soft
access is therefore insufficient reason to select smoothing as a default.

`budget_contract_v2` explicitly compares site and legal-envelope denominators at
unchanged 3%-12% fractions. Radius-six/site passes necessary bounds and routing
on 12/18 scenes; radius-six/envelope on 17/18. The sealed reference stays invalid.
Radius-three/envelope also passes 17/18; radius six is not shown to be optimal.
These checks do not certify thickness, support, facade contact or other objectives.

The radius-six envelope allowance is only 2.04%-10.65% of the old site allowance.
Example: legacy scene 000 changes from 833.28-3333.12 to 22.11-88.44 occupied-voxel
equivalents. This major reduction needs architectural interpretation, not merely
numerical approval. Spill still counts toward total mass. Neither denominator is
selected as the project's final production contract.

## Recovery result and limits

`cpu_training_checkpoint_v1` saves model weights/buffers, Adam state, StepLR,
completed update count, metadata/source/scene identifiers and Python, NumPy,
global PyTorch and explicit firing RNG. Exclusive publication preserves older
checkpoints. Tests reject wrong metadata, overwrite attempts and truncated files.

R1 starts at the historical weights, saves after update two, recreates everything
in a fresh process and repeats updates three/four. Full checkpoint trees, sampled
scenes/horizons, losses and field arrays match the uninterrupted branch exactly;
a second continuation matches too. Three scenes were available, but only two were
sampled in these four updates. Unit weights on nine losses exercise mechanics;
they are not calibrated weights or a meaningful learning curve.

The tested boundary is a completed optimizer update followed by orderly process
exit. CUDA RNG, mixed precision/scaler state, data-loader workers, pool state,
Colab disconnects and abrupt mid-write failure are not certified. A filesystem
without hard-link publication falls back to exclusive copying, which does not
promise atomic visibility under power loss. Before paid training, extend checkpoint
state to the actual trainer and validate its real interruption conditions.

## Next implementation work

1. Audit architectural meaning before choosing the mass contract: compare legal
   guide, thickened scaffold and explicit full-volume candidates against all nine
   terms and independent binary metrics across all 18 scenes. Explain absolute
   material amounts in metres, including thickness/facade/support conflicts.
2. Measure per-term magnitudes and gradient directions on valid contexts, using
   all scenes deterministically rather than relying on random sampling. Review
   retained regularizers and pre-clamp coverage saturation. Keep raw coefficients
   visible; do not hide objective incompatibility with adaptive weighting.
3. Preregister a small corrected-baseline experiment with procedural scaffold,
   direct optimization and NCA controls. Include longer rollouts, perturbations,
   absent-scaffold recovery and held-out scenes. Change one factor at a time.
4. Only after those checks, prepare a bounded Colab pilot, ask for its compute cap
   and exact artifact/Drive operations, and test GPU checkpoint recovery. Bigger
   grids and the redesigned studio remain planned after a measurable baseline.

No paid training, Drive operation, remote push, deployment or production-default
change occurred. Original notebook/checkpoint/scenes remain preserved. Local
archives are same-disk copies; off-device backup remains pending user approval.
