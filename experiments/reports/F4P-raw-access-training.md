# F4 pilot: raw-access training

Run `20260923T200402Z_b794e30d69ba`.

One access-family change: score raw maximin before clipping. Original initialization, constant16 training, two development scenes, one training seed and two recipes. Both arms use64 updates/model and1024 recurrent training steps. Scoring overhead may differ.

All raw fields, full checkpoint states, schedule cursors, RNG states and source snapshots remain in registered local artifacts. Fields share scoring formulas; component connectivity uses independent BFS. Training fields precede their update, checkpoints follow it. Recovery certifies completed CPU update boundaries, not GPU or abrupt writes.

## Evaluation results

| Arm | Member | Update | Growth | Firing seed | Connected | Mass/envelope | Joint |
|---|---|---:|---:|---:|---|---:|---|
| F4 | mapped_30-r0 | 0 | 16 | 2 | False | 0.1608797 | False |
| F4 | mapped_30-r0 | 0 | 50 | 2 | False | 0.49027273 | False |
| F4 | mapped_30-r0 | 1 | 16 | 2 | False | 0.14891423 | False |
| F4 | mapped_30-r0 | 1 | 50 | 2 | False | 0.46858674 | False |
| F4 | mapped_30-r0 | 2 | 16 | 0 | False | 0.14064598 | False |
| F4 | mapped_30-r0 | 2 | 24 | 0 | False | 0.2256946 | False |
| F4 | mapped_30-r0 | 2 | 32 | 0 | False | 0.36388379 | False |
| F4 | mapped_30-r0 | 2 | 40 | 0 | False | 0.4329372 | False |
| F4 | mapped_30-r0 | 2 | 50 | 0 | False | 0.44277847 | False |
| F4 | mapped_30-r0 | 2 | 64 | 0 | False | 0.44406763 | False |
| F4 | mapped_30-r0 | 2 | 16 | 1 | False | 0.13797587 | False |
| F4 | mapped_30-r0 | 2 | 24 | 1 | False | 0.2209269 | False |
| F4 | mapped_30-r0 | 2 | 32 | 1 | False | 0.35576445 | False |
| F4 | mapped_30-r0 | 2 | 40 | 1 | False | 0.42618746 | False |
| F4 | mapped_30-r0 | 2 | 50 | 1 | False | 0.44248396 | False |
| F4 | mapped_30-r0 | 2 | 64 | 1 | False | 0.44360086 | False |
| F4 | mapped_30-r0 | 2 | 16 | 2 | False | 0.13741797 | False |
| F4 | mapped_30-r0 | 2 | 24 | 2 | False | 0.21903282 | False |
| F4 | mapped_30-r0 | 2 | 32 | 2 | False | 0.35578766 | False |
| F4 | mapped_30-r0 | 2 | 40 | 2 | False | 0.43133563 | False |
| F4 | mapped_30-r0 | 2 | 50 | 2 | False | 0.4456192 | False |
| F4 | mapped_30-r0 | 2 | 64 | 2 | False | 0.44830155 | False |
| F4 | mapped_30-r1 | 0 | 16 | 2 | False | 0.12421025 | False |
| F4 | mapped_30-r1 | 0 | 50 | 2 | False | 0.37166923 | False |
| F4 | mapped_30-r1 | 1 | 16 | 2 | False | 0.11849251 | False |
| F4 | mapped_30-r1 | 1 | 50 | 2 | False | 0.36143559 | False |
| F4 | mapped_30-r1 | 2 | 16 | 0 | False | 0.11174142 | False |
| F4 | mapped_30-r1 | 2 | 24 | 0 | False | 0.17954694 | False |
| F4 | mapped_30-r1 | 2 | 32 | 0 | False | 0.2852729 | False |
| F4 | mapped_30-r1 | 2 | 40 | 0 | False | 0.32988256 | False |
| F4 | mapped_30-r1 | 2 | 50 | 0 | False | 0.33837318 | False |
| F4 | mapped_30-r1 | 2 | 64 | 0 | False | 0.33996105 | False |
| F4 | mapped_30-r1 | 2 | 16 | 1 | False | 0.10813516 | False |
| F4 | mapped_30-r1 | 2 | 24 | 1 | False | 0.17789808 | False |
| F4 | mapped_30-r1 | 2 | 32 | 1 | False | 0.28375208 | False |
| F4 | mapped_30-r1 | 2 | 40 | 1 | False | 0.32879525 | False |
| F4 | mapped_30-r1 | 2 | 50 | 1 | False | 0.33781463 | False |
| F4 | mapped_30-r1 | 2 | 64 | 1 | False | 0.34139785 | False |
| F4 | mapped_30-r1 | 2 | 16 | 2 | False | 0.11291157 | False |
| F4 | mapped_30-r1 | 2 | 24 | 2 | False | 0.1826711 | False |
| F4 | mapped_30-r1 | 2 | 32 | 2 | False | 0.28136009 | False |
| F4 | mapped_30-r1 | 2 | 40 | 2 | False | 0.32562891 | False |
| F4 | mapped_30-r1 | 2 | 50 | 2 | False | 0.33670533 | False |
| F4 | mapped_30-r1 | 2 | 64 | 2 | False | 0.33778512 | False |
| F4 | mass_3-r0 | 0 | 16 | 2 | False | 0.1608797 | False |
| F4 | mass_3-r0 | 0 | 50 | 2 | False | 0.49027273 | False |
| F4 | mass_3-r0 | 1 | 16 | 2 | False | 0.15131471 | False |
| F4 | mass_3-r0 | 1 | 50 | 2 | False | 0.47703132 | False |
| F4 | mass_3-r0 | 2 | 16 | 0 | False | 0.1456148 | False |
| F4 | mass_3-r0 | 2 | 24 | 0 | False | 0.23608594 | False |
| F4 | mass_3-r0 | 2 | 32 | 0 | False | 0.38046229 | False |
| F4 | mass_3-r0 | 2 | 40 | 0 | False | 0.44641197 | False |
| F4 | mass_3-r0 | 2 | 50 | 0 | False | 0.45566565 | False |
| F4 | mass_3-r0 | 2 | 64 | 0 | False | 0.45604622 | False |
| F4 | mass_3-r0 | 2 | 16 | 1 | False | 0.14306971 | False |
| F4 | mass_3-r0 | 2 | 24 | 1 | False | 0.23186302 | False |
| F4 | mass_3-r0 | 2 | 32 | 1 | False | 0.37144896 | False |
| F4 | mass_3-r0 | 2 | 40 | 1 | False | 0.43954974 | False |
| F4 | mass_3-r0 | 2 | 50 | 1 | False | 0.45407549 | False |
| F4 | mass_3-r0 | 2 | 64 | 1 | False | 0.456545 | False |
| F4 | mass_3-r0 | 2 | 16 | 2 | False | 0.14195691 | False |
| F4 | mass_3-r0 | 2 | 24 | 2 | False | 0.22913195 | False |
| F4 | mass_3-r0 | 2 | 32 | 2 | False | 0.37172136 | False |
| F4 | mass_3-r0 | 2 | 40 | 2 | False | 0.44440722 | False |
| F4 | mass_3-r0 | 2 | 50 | 2 | False | 0.45744714 | False |
| F4 | mass_3-r0 | 2 | 64 | 2 | False | 0.46003008 | False |
| F4 | mass_3-r1 | 0 | 16 | 2 | False | 0.12421025 | False |
| F4 | mass_3-r1 | 0 | 50 | 2 | False | 0.37166923 | False |
| F4 | mass_3-r1 | 1 | 16 | 2 | False | 0.11887208 | False |
| F4 | mass_3-r1 | 1 | 50 | 2 | False | 0.36307469 | False |
| F4 | mass_3-r1 | 2 | 16 | 0 | False | 0.1122648 | False |
| F4 | mass_3-r1 | 2 | 24 | 0 | False | 0.18078057 | False |
| F4 | mass_3-r1 | 2 | 32 | 0 | False | 0.28698456 | False |
| F4 | mass_3-r1 | 2 | 40 | 0 | False | 0.3318823 | False |
| F4 | mass_3-r1 | 2 | 50 | 0 | False | 0.34100166 | False |
| F4 | mass_3-r1 | 2 | 64 | 0 | False | 0.34183949 | False |
| F4 | mass_3-r1 | 2 | 16 | 1 | False | 0.10866508 | False |
| F4 | mass_3-r1 | 2 | 24 | 1 | False | 0.17922564 | False |
| F4 | mass_3-r1 | 2 | 32 | 1 | False | 0.28523561 | False |
| F4 | mass_3-r1 | 2 | 40 | 1 | False | 0.32993391 | False |
| F4 | mass_3-r1 | 2 | 50 | 1 | False | 0.34062007 | False |
| F4 | mass_3-r1 | 2 | 64 | 1 | False | 0.3437511 | False |
| F4 | mass_3-r1 | 2 | 16 | 2 | False | 0.11343396 | False |
| F4 | mass_3-r1 | 2 | 24 | 2 | False | 0.18394187 | False |
| F4 | mass_3-r1 | 2 | 32 | 2 | False | 0.28289258 | False |
| F4 | mass_3-r1 | 2 | 40 | 2 | False | 0.32724422 | False |
| F4 | mass_3-r1 | 2 | 50 | 2 | False | 0.33741227 | False |
| F4 | mass_3-r1 | 2 | 64 | 2 | False | 0.33937123 | False |
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
  "prefix_checkpoint": true,
  "resumed_traces": true,
  "resumed_checkpoint": true,
  "resumed_fields": true,
  "repeat_traces": true,
  "repeat_checkpoint": true,
  "repeat_fields": true,
  "resumed_evaluation_traces": true,
  "resumed_evaluation_fields": true,
  "repeat_evaluation_traces": true,
  "repeat_evaluation_fields": true,
  "source_hashes_verified": 41,
  "unique_saved_fields_rescored": 96,
  "unique_evaluations": 88,
  "boundary_evaluations": 24,
  "final_grid_evaluations": 72,
  "checkpoint_schedule_cursors_verified": 12,
  "F2_boundary_controls_rescored": 56,
  "H1_F2_growth_controls_rescored": 72,
  "initial_F2_fields_exact": 8,
  "final_checkpoint_rollouts_exact": 0,
  "elapsed_caps_met": true,
  "admission": {
    "max_update_seconds": 4.561181299999589,
    "max_evaluation_pair_seconds": 7.6591084000247065,
    "max_extra_grid_seconds": 50.94160180000472,
    "startup_allowance_seconds": 5.111819900019327,
    "safety_factor": 1.5,
    "estimated_member_seconds": 602.374175550256,
    "estimated_total_seconds": 2409.496702201024,
    "admitted": false
  }
}
```

No holdout/generalization, walkability or mechanical safety claim. No automatic model/recipe promotion, paid training or production change.
