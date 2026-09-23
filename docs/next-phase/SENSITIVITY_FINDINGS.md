# Sensitivity findings: weights alone did not establish a usable baseline

Completed 2026-09-23. The planned local comparison ran successfully, but neither
weight setting is selected for production or larger training. Lower material-budget
weight improves coverage and a few long-horizon connections, while material remains
far above the allowed budget. All five feasible reference scenes remain disconnected.
This is evidence about a tiny optimization trial, not proof that the NCA concept fails.

## What ran and how it was checked

K2 `20260923T102524Z_f5e1cc169dea`, source33f6858: two recipes x two training seeds,
17 updates each (68 total). Both recipes start from the same original checkpoint
and share each seed's scene order and firing RNG. The only changed coefficient is
sparsity30 versus3. Adam1e-4, norm clipping1, constant learning rate, 16-step growth,
nine corrected constraint families and the three explicitly retained regularizers.
Every feasible development scene is used once in each training run. No pool/AMP.

Evaluation: 136 trained-model cases, 34 original-checkpoint cases and 17 static
W1 controls (187 total). All learned models use identical firing seed2 and growth
horizons16/50. Original production serving behavior is not the baseline here:
all recurrent comparisons use the same experimental rollout/objective contract.
W1 is a static geometric comparator with no growth horizon.

Actual-loop recovery K2R `20260923T102414Z_eaec7bd1510e` passes all seven exact
comparisons over three logical/eight executed updates in fresh processes. The
144-test regression run `20260923T102104Z_15b3cd2f4fd5` has no failures/errors/skips
and checkpoint smoke exit0. All68 K2 checkpoint boundaries, all255 saved training/
evaluation fields, every evaluation metric and both weighted totals were checked.
Source snapshot hashes and report rerender agree. No failed or timed-out run.

Training workers took47.73-51.96 seconds each; the complete study took505.98 seconds
including evaluation and evidence overhead. Timings reflect local contention and
are not a deployment benchmark. Caps were900 wall seconds per worker. No paid
compute, Drive access, push or deployment occurred.

## The important comparison

At16 steps, the original and every trained model connect10/17 scenes. Compared
with weight30, weight3 lowers coverage loss in every scene for both training seeds,
but neither seed gains a binary connection at that horizon.

At50 steps:

| Model | Connected scenes | Scenes over the12% mass cap | Mean material / envelope |
|---|---:|---:|---:|
| Original checkpoint | 10/17 | 17/17 | 28.01% |
| Weight30, seed0 | 10/17 | 15/17 | 21.44% |
| Weight30, seed1 | 10/17 | 15/17 | 21.44% |
| Weight3, seed0 | 12/17 | 17/17 | 27.40% |
| Weight3, seed1 | 11/17 | 17/17 | 26.80% |
| W1 procedural, static | 17/17 | 0/17 | 3.55% |

Weight3 gains legacy008 for both seeds and legacy004 for seed0, losing no original
connections at50 steps. It fails all five feasible reference scenes: ground-pair,
facade-pair-and-ground, wide-gap, asymmetric-heights and minimal-smoke. Weight30
also fails all five. The gains therefore do not address the core reference failures.
The sealed reference remains outside feasible calibration and is still infeasible.

Stronger budget weight helps mass control but sacrifices coverage. At50 steps,
mean coverage loss is0.1722 for weight30 versus0.1416/0.1473 for weight3 and0.1509
for the original. Weight3 still has mean sparsity penalties4.2115/3.7942; weight30
has1.8437/1.8435. Neither satisfies the joint objectives. Mean mass also increases
between16 and50 steps for every recurrent arm; training-horizon results alone
would understate this excess.

Both recipes improve the original's50-step mean total under either common scoring
recipe, but weight30 ranks better there on both totals. At16 steps weight3 ranks
better under its own common scoring column and weight30 under the other. This
ranking dependence is why we retain raw residuals and connectivity rather than
selecting the lowest differently weighted total. Full comparisons and all cases
are in ../../experiments/reports/K2-sensitivity.md.

## What remains wrong

1. **Reference adaptation is unresolved.** Each short run did see all five feasible
   reference geometries, but none becomes connected. Seventeen updates are not a
   convergence study; this result cannot isolate insufficient optimization from a
   weak recurrent update or objective-gradient limitation by itself.
2. **Coverage versus excess mass is a real tradeoff in this test.** The K1 local
   gradient hypothesis translates into better coverage under weight3, but not
   joint success. Simply adopting3 or training longer is not yet justified.
3. **Growth beyond the training horizon still adds excess material.** The 12%
   envelope cap is exceeded in15-17 scenes at50 steps. This is not a physical site
   percentage; the envelope contract and earlier material-allowance change remain
   explicit. No normalization change was made to make these runs look better.
4. **Some metrics are weak success criteria.** All evaluated fields have zero
   illegal, protected-ground-blocking and geometrically unsupported voxels. That
   does not imply endpoint connectivity, walkability or mechanical safety. Training
   thickness stays zero although the independent radius1 erosion finds bulk;
   training's radius2 maximum-bulk proxy is a different measurement, not minimum
   thickness. W1's thin strands remain a strong numerical control and a weak
   architectural design. No new family is needed to acknowledge this limitation.
5. **NCA value beyond a procedural guide remains unproven.** W1 has17 connected,
   in-budget cases and zero nine-family residuals; its TV and boundary-cantilever
   regularizers are small but nonzero. These short NCA runs do not beat that control
   on the measured criteria. Existing scenes have been inspected throughout
   development, so no geometry generalization claim is available.

## Decision and concrete next work

D033: retain both runs as sensitivity evidence; promote neither model nor weight
setting. Preserve the original serving defaults. Keep both coefficient sets as
explicit controls for the next stage rather than declaring an optimum.

Next implement the missing E2 direct-material-optimization control. First profile
it on legacy008, ground-pair and minimal-smoke, using the same legal projection,
scene-derived facade allowance, envelope budget, nine families and regularizers.
Record material initialization, step size, optimizer, update/time budgets and all
residuals before interpreting it. Include the procedural witness as an explicit
initialization/control where relevant, never disguised as NCA generation.

This will test whether the objective can recover those same cases when it directly
controls a field, before attributing failure to the recurrent architecture. A direct
per-scene solve does not generalize and must carry its optimization cost. Freeze
its full matched protocol after local timing/gradient checks, then compare on all17
development scenes. This plan is prepared; no direct optimization has run here.
Do not add a schedule, model architecture and scene distribution change together.

Later: freeze fresh geometry holdouts, test learned seed/edit recovery and growth
stability, then evaluate conditioning/perception changes and larger grids. The
studio redesign remains a separate planned implementation milestone and can use
these saved fields for truthful comparisons. No interface improvement is claimed.

## Evidence and resuming

All run inputs, fields, checkpoints, logs, source snapshots and decisions remain
preserved. Small summaries/reports are in experiments/records and experiments/reports;
raw evidence is in .local-artifacts/runs/<run_id>. RESUME.md records exact next steps.
A new local archive includes Git history and the raw evidence, with verified file
hashes and a fresh restore check. It remains a same-disk copy, not an off-device
backup. The private next-phase report stays Git-ignored.

Interrupted-run imports are implemented but were not exercised by this successful
study; the actual training worker's orderly completed-update restart was exercised
by K2R. Forced timeouts, mid-write failure, CUDA and Colab recovery are not certified.
