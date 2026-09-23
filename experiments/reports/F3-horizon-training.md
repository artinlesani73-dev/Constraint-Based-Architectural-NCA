# F3 study: mixed-horizon training

Run `20260923T160713Z_cc33850561b8`.

One schedule change: alternate16/50 instead of constant16 growth. Original initialization, F2 objectives, two development scenes, one training seed and two recipes. Proposed64 updates/model means2112 recurrent training steps versus1024 for F2: update counts match, compute does not.

All raw fields, full checkpoint states, schedule cursors, RNG states and source snapshots remain in registered local artifacts. Fields share scoring formulas; component connectivity uses independent BFS. Training fields precede their update, checkpoints follow it. Recovery certifies completed CPU update boundaries, not GPU or abrupt writes.

## Evaluation results

| Arm | Member | Update | Growth | Firing seed | Connected | Mass/envelope | Joint |
|---|---|---:|---:|---:|---|---:|---|
| F3 | mapped_30-r0 | 0 | 16 | 2 | False | 0.1608797 | False |
| F3 | mapped_30-r0 | 0 | 50 | 2 | False | 0.49027273 | False |
| F3 | mapped_30-r0 | 1 | 16 | 2 | False | 0.14891423 | False |
| F3 | mapped_30-r0 | 1 | 50 | 2 | False | 0.46858674 | False |
| F3 | mapped_30-r0 | 3 | 16 | 2 | False | 0.12748206 | False |
| F3 | mapped_30-r0 | 3 | 50 | 2 | False | 0.43332034 | False |
| F3 | mapped_30-r0 | 8 | 16 | 2 | False | 0.092271648 | False |
| F3 | mapped_30-r0 | 8 | 50 | 2 | False | 0.10412524 | False |
| F3 | mapped_30-r0 | 16 | 16 | 2 | False | 0.095238201 | False |
| F3 | mapped_30-r0 | 16 | 50 | 2 | False | 0.12442351 | False |
| F3 | mapped_30-r0 | 32 | 16 | 2 | False | 0.10293693 | False |
| F3 | mapped_30-r0 | 32 | 50 | 2 | False | 0.13670896 | False |
| F3 | mapped_30-r0 | 64 | 16 | 0 | False | 0.11818442 | False |
| F3 | mapped_30-r0 | 64 | 24 | 0 | False | 0.12948652 | False |
| F3 | mapped_30-r0 | 64 | 32 | 0 | False | 0.12965249 | False |
| F3 | mapped_30-r0 | 64 | 40 | 0 | False | 0.13034029 | False |
| F3 | mapped_30-r0 | 64 | 50 | 0 | False | 0.13084908 | False |
| F3 | mapped_30-r0 | 64 | 64 | 0 | False | 0.13123645 | False |
| F3 | mapped_30-r0 | 64 | 16 | 1 | False | 0.11661325 | False |
| F3 | mapped_30-r0 | 64 | 24 | 1 | False | 0.12806615 | False |
| F3 | mapped_30-r0 | 64 | 32 | 1 | False | 0.12902895 | False |
| F3 | mapped_30-r0 | 64 | 40 | 1 | False | 0.12984344 | False |
| F3 | mapped_30-r0 | 64 | 50 | 1 | False | 0.1323268 | False |
| F3 | mapped_30-r0 | 64 | 64 | 1 | False | 0.13340564 | False |
| F3 | mapped_30-r0 | 64 | 16 | 2 | False | 0.11439943 | False |
| F3 | mapped_30-r0 | 64 | 24 | 2 | False | 0.12721665 | False |
| F3 | mapped_30-r0 | 64 | 32 | 2 | False | 0.1277221 | False |
| F3 | mapped_30-r0 | 64 | 40 | 2 | False | 0.12888488 | False |
| F3 | mapped_30-r0 | 64 | 50 | 2 | False | 0.13115861 | False |
| F3 | mapped_30-r0 | 64 | 64 | 2 | False | 0.13232104 | False |
| F3 | mapped_30-r1 | 0 | 16 | 2 | False | 0.12421025 | False |
| F3 | mapped_30-r1 | 0 | 50 | 2 | False | 0.37166923 | False |
| F3 | mapped_30-r1 | 1 | 16 | 2 | False | 0.11849251 | False |
| F3 | mapped_30-r1 | 1 | 50 | 2 | False | 0.36143559 | False |
| F3 | mapped_30-r1 | 3 | 16 | 2 | False | 0.10512441 | False |
| F3 | mapped_30-r1 | 3 | 50 | 2 | False | 0.32757798 | False |
| F3 | mapped_30-r1 | 8 | 16 | 2 | False | 0.079831533 | False |
| F3 | mapped_30-r1 | 8 | 50 | 2 | False | 0.13831858 | False |
| F3 | mapped_30-r1 | 16 | 16 | 2 | False | 0.083784111 | False |
| F3 | mapped_30-r1 | 16 | 50 | 2 | False | 0.13982615 | False |
| F3 | mapped_30-r1 | 32 | 16 | 2 | False | 0.099675745 | False |
| F3 | mapped_30-r1 | 32 | 50 | 2 | False | 0.1512468 | False |
| F3 | mapped_30-r1 | 64 | 16 | 0 | False | 0.1177915 | False |
| F3 | mapped_30-r1 | 64 | 24 | 0 | False | 0.14927168 | False |
| F3 | mapped_30-r1 | 64 | 32 | 0 | False | 0.16894437 | False |
| F3 | mapped_30-r1 | 64 | 40 | 0 | False | 0.18055534 | False |
| F3 | mapped_30-r1 | 64 | 50 | 0 | False | 0.19021499 | False |
| F3 | mapped_30-r1 | 64 | 64 | 0 | False | 0.19282942 | False |
| F3 | mapped_30-r1 | 64 | 16 | 1 | False | 0.11132199 | False |
| F3 | mapped_30-r1 | 64 | 24 | 1 | False | 0.14732414 | False |
| F3 | mapped_30-r1 | 64 | 32 | 1 | False | 0.16941366 | False |
| F3 | mapped_30-r1 | 64 | 40 | 1 | False | 0.18214382 | False |
| F3 | mapped_30-r1 | 64 | 50 | 1 | False | 0.19053215 | False |
| F3 | mapped_30-r1 | 64 | 64 | 1 | False | 0.1922043 | False |
| F3 | mapped_30-r1 | 64 | 16 | 2 | False | 0.11682047 | False |
| F3 | mapped_30-r1 | 64 | 24 | 2 | False | 0.14886709 | False |
| F3 | mapped_30-r1 | 64 | 32 | 2 | False | 0.16888103 | False |
| F3 | mapped_30-r1 | 64 | 40 | 2 | False | 0.17777319 | False |
| F3 | mapped_30-r1 | 64 | 50 | 2 | False | 0.1826611 | False |
| F3 | mapped_30-r1 | 64 | 64 | 2 | False | 0.1854005 | False |
| F3 | mass_3-r0 | 0 | 16 | 2 | False | 0.1608797 | False |
| F3 | mass_3-r0 | 0 | 50 | 2 | False | 0.49027273 | False |
| F3 | mass_3-r0 | 1 | 16 | 2 | False | 0.15131471 | False |
| F3 | mass_3-r0 | 1 | 50 | 2 | False | 0.47703132 | False |
| F3 | mass_3-r0 | 3 | 16 | 2 | False | 0.13094252 | False |
| F3 | mass_3-r0 | 3 | 50 | 2 | False | 0.44419494 | False |
| F3 | mass_3-r0 | 8 | 16 | 2 | False | 0.095756486 | False |
| F3 | mass_3-r0 | 8 | 50 | 2 | False | 0.13596414 | False |
| F3 | mass_3-r0 | 16 | 16 | 2 | False | 0.095678262 | False |
| F3 | mass_3-r0 | 16 | 50 | 2 | False | 0.12732558 | False |
| F3 | mass_3-r0 | 32 | 16 | 2 | False | 0.10258132 | False |
| F3 | mass_3-r0 | 32 | 50 | 2 | False | 0.1431042 | False |
| F3 | mass_3-r0 | 64 | 16 | 0 | False | 0.11875428 | False |
| F3 | mass_3-r0 | 64 | 24 | 0 | False | 0.12995075 | False |
| F3 | mass_3-r0 | 64 | 32 | 0 | False | 0.13156924 | False |
| F3 | mass_3-r0 | 64 | 40 | 0 | False | 0.13394602 | False |
| F3 | mass_3-r0 | 64 | 50 | 0 | False | 0.13675365 | False |
| F3 | mass_3-r0 | 64 | 64 | 0 | False | 0.13774404 | False |
| F3 | mass_3-r0 | 64 | 16 | 1 | False | 0.11674662 | False |
| F3 | mass_3-r0 | 64 | 24 | 1 | False | 0.12821867 | False |
| F3 | mass_3-r0 | 64 | 32 | 1 | False | 0.13060984 | False |
| F3 | mass_3-r0 | 64 | 40 | 1 | False | 0.13292557 | False |
| F3 | mass_3-r0 | 64 | 50 | 1 | False | 0.13487846 | False |
| F3 | mass_3-r0 | 64 | 64 | 1 | False | 0.13557483 | False |
| F3 | mass_3-r0 | 64 | 16 | 2 | False | 0.11487868 | False |
| F3 | mass_3-r0 | 64 | 24 | 2 | False | 0.12731744 | False |
| F3 | mass_3-r0 | 64 | 32 | 2 | False | 0.13064124 | False |
| F3 | mass_3-r0 | 64 | 40 | 2 | False | 0.13206637 | False |
| F3 | mass_3-r0 | 64 | 50 | 2 | False | 0.13289666 | False |
| F3 | mass_3-r0 | 64 | 64 | 2 | False | 0.1344353 | False |
| F3 | mass_3-r1 | 0 | 16 | 2 | False | 0.12421025 | False |
| F3 | mass_3-r1 | 0 | 50 | 2 | False | 0.37166923 | False |
| F3 | mass_3-r1 | 1 | 16 | 2 | False | 0.11887208 | False |
| F3 | mass_3-r1 | 1 | 50 | 2 | False | 0.36307469 | False |
| F3 | mass_3-r1 | 3 | 16 | 2 | False | 0.10599212 | False |
| F3 | mass_3-r1 | 3 | 50 | 2 | False | 0.328807 | False |
| F3 | mass_3-r1 | 8 | 16 | 2 | False | 0.08042594 | False |
| F3 | mass_3-r1 | 8 | 50 | 2 | False | 0.13717456 | False |
| F3 | mass_3-r1 | 16 | 16 | 2 | False | 0.090000182 | False |
| F3 | mass_3-r1 | 16 | 50 | 2 | False | 0.14765215 | False |
| F3 | mass_3-r1 | 32 | 16 | 2 | False | 0.10341152 | False |
| F3 | mass_3-r1 | 32 | 50 | 2 | False | 0.15626892 | False |
| F3 | mass_3-r1 | 64 | 16 | 0 | False | 0.1127307 | False |
| F3 | mass_3-r1 | 64 | 24 | 0 | False | 0.1389726 | False |
| F3 | mass_3-r1 | 64 | 32 | 0 | False | 0.14982577 | False |
| F3 | mass_3-r1 | 64 | 40 | 0 | False | 0.15007989 | False |
| F3 | mass_3-r1 | 64 | 50 | 0 | False | 0.15153129 | False |
| F3 | mass_3-r1 | 64 | 64 | 0 | False | 0.15245743 | False |
| F3 | mass_3-r1 | 64 | 16 | 1 | False | 0.10637968 | False |
| F3 | mass_3-r1 | 64 | 24 | 1 | False | 0.13636491 | False |
| F3 | mass_3-r1 | 64 | 32 | 1 | False | 0.1509714 | False |
| F3 | mass_3-r1 | 64 | 40 | 1 | False | 0.15237233 | False |
| F3 | mass_3-r1 | 64 | 50 | 1 | False | 0.15335242 | False |
| F3 | mass_3-r1 | 64 | 64 | 1 | False | 0.15592225 | False |
| F3 | mass_3-r1 | 64 | 16 | 2 | False | 0.11143042 | False |
| F3 | mass_3-r1 | 64 | 24 | 2 | False | 0.13694456 | False |
| F3 | mass_3-r1 | 64 | 32 | 2 | False | 0.14823319 | False |
| F3 | mass_3-r1 | 64 | 40 | 2 | False | 0.15190217 | False |
| F3 | mass_3-r1 | 64 | 50 | 2 | False | 0.15326619 | False |
| F3 | mass_3-r1 | 64 | 64 | 2 | False | 0.15370476 | False |
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
  "source_hashes_verified": 37,
  "unique_saved_fields_rescored": 376,
  "unique_evaluations": 120,
  "boundary_evaluations": 56,
  "final_grid_evaluations": 72,
  "checkpoint_schedule_cursors_verified": 260,
  "F2_boundary_controls_rescored": 56,
  "H1_F2_growth_controls_rescored": 72,
  "initial_F2_fields_exact": 8,
  "final_checkpoint_rollouts_exact": 8,
  "elapsed_caps_met": true,
  "admission": {
    "max_short_update_seconds": 3.280848900001729,
    "max_long_update_seconds": 11.492606699990574,
    "max_evaluation_pair_seconds": 4.402505799982464,
    "max_extra_grid_seconds": 32.473763999994844,
    "startup_allowance_seconds": 5.0,
    "safety_factor": 1.5,
    "estimated_member_seconds": 811.5628256994387,
    "estimated_total_seconds": 3246.2513027977548,
    "admitted": true
  }
}
```

No holdout/generalization, walkability or mechanical safety claim. No automatic model/recipe promotion, paid training or production change.
