# F4 parity: raw-access training

Run `20260923T195731Z_c19895c6c745`.

One access-family change: score raw maximin before clipping. Original initialization, constant16 training, two development scenes, one training seed and two recipes. Both arms use64 updates/model and1024 recurrent training steps. Scoring overhead may differ.

All raw fields, full checkpoint states, schedule cursors, RNG states and source snapshots remain in registered local artifacts. Fields share scoring formulas; component connectivity uses independent BFS. Training fields precede their update, checkpoints follow it. Recovery certifies completed CPU update boundaries, not GPU or abrupt writes.

## Evaluation results

| Arm | Member | Update | Growth | Firing seed | Connected | Mass/envelope | Joint |
|---|---|---:|---:|---:|---|---:|---|
| F4 | mapped_30-r0 | 0 | 16 | 2 | False | 0.1608797 | False |
| F4 | mapped_30-r0 | 0 | 50 | 2 | False | 0.49027273 | False |
| F4 | mapped_30-r0 | 1 | 16 | 2 | False | 0.14891423 | False |
| F4 | mapped_30-r0 | 1 | 50 | 2 | False | 0.46858674 | False |
| F4 | mapped_30-r0 | 3 | 16 | 2 | False | 0.12676497 | False |
| F4 | mapped_30-r0 | 3 | 50 | 2 | False | 0.43086484 | False |
| F4 | mapped_30-r1 | 0 | 16 | 2 | False | 0.12421025 | False |
| F4 | mapped_30-r1 | 0 | 50 | 2 | False | 0.37166923 | False |
| F4 | mapped_30-r1 | 1 | 16 | 2 | False | 0.11849251 | False |
| F4 | mapped_30-r1 | 1 | 50 | 2 | False | 0.36143559 | False |
| F4 | mapped_30-r1 | 3 | 16 | 2 | False | 0.10781872 | False |
| F4 | mapped_30-r1 | 3 | 50 | 2 | False | 0.33185902 | False |
| F4 | mass_3-r0 | 0 | 16 | 2 | False | 0.1608797 | False |
| F4 | mass_3-r0 | 0 | 50 | 2 | False | 0.49027273 | False |
| F4 | mass_3-r0 | 1 | 16 | 2 | False | 0.15131471 | False |
| F4 | mass_3-r0 | 1 | 50 | 2 | False | 0.47703132 | False |
| F4 | mass_3-r0 | 3 | 16 | 2 | False | 0.1331297 | False |
| F4 | mass_3-r0 | 3 | 50 | 2 | False | 0.44970497 | False |
| F4 | mass_3-r1 | 0 | 16 | 2 | False | 0.12421025 | False |
| F4 | mass_3-r1 | 0 | 50 | 2 | False | 0.37166923 | False |
| F4 | mass_3-r1 | 1 | 16 | 2 | False | 0.11887208 | False |
| F4 | mass_3-r1 | 1 | 50 | 2 | False | 0.36307469 | False |
| F4 | mass_3-r1 | 3 | 16 | 2 | False | 0.10835656 | False |
| F4 | mass_3-r1 | 3 | 50 | 2 | False | 0.33250245 | False |
| F2 | mapped_30-r0 | 0 | 16 | 2 | False | 0.1608797 | False |
| F2 | mapped_30-r0 | 0 | 50 | 2 | False | 0.49027273 | False |
| F2 | mapped_30-r0 | 1 | 16 | 2 | False | 0.14891423 | False |
| F2 | mapped_30-r0 | 1 | 50 | 2 | False | 0.46858674 | False |
| F2 | mapped_30-r0 | 3 | 16 | 2 | False | 0.12676497 | False |
| F2 | mapped_30-r0 | 3 | 50 | 2 | False | 0.43086484 | False |
| F2 | mapped_30-r0 | 8 | 16 | 2 | False | 0.092379056 | False |
| F2 | mapped_30-r0 | 8 | 50 | 2 | False | 0.10456944 | False |
| F2 | mapped_30-r0 | 16 | 16 | 2 | False | 0.098932393 | False |
| F2 | mapped_30-r0 | 16 | 50 | 2 | False | 0.1524072 | False |
| F2 | mapped_30-r0 | 32 | 16 | 2 | False | 0.12509046 | False |
| F2 | mapped_30-r0 | 32 | 50 | 2 | False | 0.19409198 | False |
| F2 | mapped_30-r0 | 64 | 16 | 2 | False | 0.13884346 | False |
| F2 | mapped_30-r0 | 64 | 50 | 2 | True | 0.23128161 | False |
| F2 | mapped_30-r1 | 0 | 16 | 2 | False | 0.12421025 | False |
| F2 | mapped_30-r1 | 0 | 50 | 2 | False | 0.37166923 | False |
| F2 | mapped_30-r1 | 1 | 16 | 2 | False | 0.11849251 | False |
| F2 | mapped_30-r1 | 1 | 50 | 2 | False | 0.36143559 | False |
| F2 | mapped_30-r1 | 3 | 16 | 2 | False | 0.10781872 | False |
| F2 | mapped_30-r1 | 3 | 50 | 2 | False | 0.33185902 | False |
| F2 | mapped_30-r1 | 8 | 16 | 2 | False | 0.090065137 | False |
| F2 | mapped_30-r1 | 8 | 50 | 2 | False | 0.14940347 | False |
| F2 | mapped_30-r1 | 16 | 16 | 2 | False | 0.10808117 | False |
| F2 | mapped_30-r1 | 16 | 50 | 2 | False | 0.17179698 | False |
| F2 | mapped_30-r1 | 32 | 16 | 2 | False | 0.12509578 | False |
| F2 | mapped_30-r1 | 32 | 50 | 2 | False | 0.21285172 | False |
| F2 | mapped_30-r1 | 64 | 16 | 2 | False | 0.133159 | False |
| F2 | mapped_30-r1 | 64 | 50 | 2 | True | 0.25915599 | False |
| F2 | mass_3-r0 | 0 | 16 | 2 | False | 0.1608797 | False |
| F2 | mass_3-r0 | 0 | 50 | 2 | False | 0.49027273 | False |
| F2 | mass_3-r0 | 1 | 16 | 2 | False | 0.15131471 | False |
| F2 | mass_3-r0 | 1 | 50 | 2 | False | 0.47703132 | False |
| F2 | mass_3-r0 | 3 | 16 | 2 | False | 0.1331297 | False |
| F2 | mass_3-r0 | 3 | 50 | 2 | False | 0.44970497 | False |
| F2 | mass_3-r0 | 8 | 16 | 2 | False | 0.10306095 | False |
| F2 | mass_3-r0 | 8 | 50 | 2 | False | 0.1562404 | False |
| F2 | mass_3-r0 | 16 | 16 | 2 | False | 0.11510575 | False |
| F2 | mass_3-r0 | 16 | 50 | 2 | False | 0.16833408 | False |
| F2 | mass_3-r0 | 32 | 16 | 2 | False | 0.14790595 | False |
| F2 | mass_3-r0 | 32 | 50 | 2 | False | 0.22935678 | False |
| F2 | mass_3-r0 | 64 | 16 | 2 | True | 0.22180726 | False |
| F2 | mass_3-r0 | 64 | 50 | 2 | True | 0.31632927 | False |
| F2 | mass_3-r1 | 0 | 16 | 2 | False | 0.12421025 | False |
| F2 | mass_3-r1 | 0 | 50 | 2 | False | 0.37166923 | False |
| F2 | mass_3-r1 | 1 | 16 | 2 | False | 0.11887208 | False |
| F2 | mass_3-r1 | 1 | 50 | 2 | False | 0.36307469 | False |
| F2 | mass_3-r1 | 3 | 16 | 2 | False | 0.10835656 | False |
| F2 | mass_3-r1 | 3 | 50 | 2 | False | 0.33250245 | False |
| F2 | mass_3-r1 | 8 | 16 | 2 | False | 0.090319909 | False |
| F2 | mass_3-r1 | 8 | 50 | 2 | False | 0.15081717 | False |
| F2 | mass_3-r1 | 16 | 16 | 2 | False | 0.1079538 | False |
| F2 | mass_3-r1 | 16 | 50 | 2 | False | 0.1715506 | False |
| F2 | mass_3-r1 | 32 | 16 | 2 | False | 0.15540956 | False |
| F2 | mass_3-r1 | 32 | 50 | 2 | False | 0.26212299 | False |
| F2 | mass_3-r1 | 64 | 16 | 2 | True | 0.22884564 | False |
| F2 | mass_3-r1 | 64 | 50 | 2 | True | 0.33423263 | False |

Budget3%-12%, tolerance1e-6. Strict material>0.5 component connectivity. Final grids reuse eight boundary evaluations explicitly; unique counts do not count them twice. H1 historical final-grid controls and all individual losses/metrics are in the JSON report.

## Verification

```json
{
  "baseline_parity": true,
  "training_update_exact_F2_matches": 12,
  "evaluation_record_exact_F2_matches": 24,
  "source_hashes_verified": 41,
  "unique_saved_fields_rescored": 36,
  "unique_evaluations": 24,
  "boundary_evaluations": 24,
  "final_grid_evaluations": 0,
  "checkpoint_schedule_cursors_verified": 16,
  "F2_boundary_controls_rescored": 56,
  "H1_F2_growth_controls_rescored": 72,
  "initial_F2_fields_exact": 8,
  "final_checkpoint_rollouts_exact": 0,
  "elapsed_caps_met": true
}
```

No holdout/generalization, walkability or mechanical safety claim. No automatic model/recipe promotion, paid training or production change.
