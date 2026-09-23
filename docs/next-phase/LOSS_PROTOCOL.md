# L1_v1: shared loss mechanics and objective compatibility

Frozen before the diagnostic run on 2026-09-23. This is a software/gradient audit,
not a trained-model benchmark. No optimizer steps or production changes.

## Versioned meanings

`geometry_losses_v1` returns nine separate per-scene continuous terms. Occupancy
is [B,D,H,W] in [0,1]. Context is explicit, detached Boolean masks. The connection
coverage mask and material envelope are separate required inputs. Mass uses the
historical non-building denominator and unchanged 3%-12% limits. Ratios are
computed per scene before batch averaging. New formulas mean the historical
loss weights cannot be reused without calibration.

| Term | Formula/meaning |
|---|---|
| Legality | Occupancy in forbidden voxels divided by forbidden-region size. |
| Coverage | Mean missing occupancy on the supplied connection guide. This is an explicit spatial material proxy, not a final walking/deck objective. |
| Spill | Legal mass outside the supplied material envelope divided by non-building volume. Fixed normalization avoids reducing spill merely by adding inside mass. |
| Ground | Occupancy in the protected street region divided by that region's size. |
| Thickness | Continuous zero-padded cubic erosion core mass divided by max(total mass, 1 voxel). Radius 2 means two voxel layers, not a calibrated physical maximum thickness. |
| Sparsity | 150*relu(mass ratio-0.12)^2 + relu(0.03-mass ratio), with mass counted in the non-building region. |
| Facade | relu(contact mass/max(total mass,1)-0.15); preserve the legacy 26-neighbor contact zone. |
| Access | One minus mean bounded-hop maximum-bottleneck material reach at other explicit endpoint regions, seeded at one fixed legal voxel in the first sorted endpoint ID. |
| Support | Mean unsupported mass fraction using the same reach surrogate from fixed existing/anchor support boundaries. This is geometric support only. |

Access/support use six face neighbors. Maximum-bottleneck propagation uses
max/min subgradients; zero material does not transmit. It is not a probability
or exact unlimited connectivity. `reach_hops=64` is explicit in this diagnostic;
it can underestimate longer routes, and ties or broken zero-occupancy routes can
have poor gradients. Those limitations must be measured, not hidden. Binary
connectivity remains the independent evaluator.

No hidden success is assigned to empty material or an empty guide. Empty material
is flagged but allowed as a training start state; empty guides and infeasible
routes invalidate the objective context. A strict mean refuses any invalid batch
member. Context checks include necessary capacity bounds: legal envelope capacity
must accommodate minimum mass; full guide coverage must fit the upper mass bound.
These checks do not prove all nine objectives can simultaneously be zero.
Ground/legality can have zero parameter gradient after hard projection; that is
expected. No arbitrary requirement that every loss must have a nonzero gradient
on every scene. Cantilever, density and TV integration remain subsequent work;
no additional porosity or surface-area family is introduced.

## Recorded diagnostic matrix

Use verified C1 run 20260923T000945Z_3cbdc3603a12, its saved seed/centerline/targets,
18 frozen scenes and original embedded checkpoint configuration.

1. **72 objective-context checks:** each scene with four explicit envelopes:
   C1 legal thick scaffold, centerline expanded by 3 graph steps (2.4m at 0.8m
   voxels), centerline expanded by 6 steps (4.8m), and all permitted space as a
   broad diagnostic control. Fixed radii are not adjusted to make budgets pass.
   Coverage is the saved legal centerline including endpoint regions. Score all
   nine terms on the recorded C1 training-profile/seed-0/legal-target final state.
   Keep infeasible scenes and incompatible capacities as findings, not run errors.
   No envelope is selected for training from this audit alone.
2. **Three real-model gradient checks:** legacy-easy-seed-000 alone,
   ref-01-ground-pair alone, and both in a batch. Original checkpoint, four
   historical-training updates, scale 0.15 on the legal thick scaffold, epoch60,
   explicit firing RNG seed0, two CPU threads. Use the radius-six envelope and
   64-hop proxy. Save full states, occupancy and parameter gradients for every
   term, norms and pairwise cosine similarities (null for zero-norm pairs).
   These are short-horizon diagnostics, not 50-step training/quality estimates.
   Do not alter weights. Record context validity even when inspecting an invalid
   context's individual diagnostic terms; never optimize such a batch silently.
3. **Six historical fine-tuner checks:** extract original ThicknessLoss,
   PorosityLoss and SurfaceAreaLoss definitions without importing the training
   script or running a trainer; record shape/gradient outcomes at B=1 and B=2.
   Preserve errors and disconnected-gradient outcomes as expected historical
   defects, separately from failures in the diagnostic runner.
4. Run synthetic and real foundation regressions. Check per-scene/batch value
   and gradient equivalence, directional finite differences away from nonsmooth
   ties, a unique path bottleneck, empty-background erosion, zero transmission
   through empty space, bounded legal envelopes and strict invalid-batch handling.

All runs have immutable evidence, short safe filenames, config/hashes/seeds,
source snapshots, individual results and explicit failed/interrupted outcomes.
A retry gets a new ID. Keep everything local. Colab preparation follows measured
loss behavior and recovery checks, with a separate explicit compute allowance.
