# F3 pilot: mixed-horizon training

Run `20260923T160209Z_e292a696d594`.

One schedule change: alternate16/50 instead of constant16 growth. Original initialization, F2 objectives, two development scenes, one training seed and two recipes. Proposed64 updates/model means2112 recurrent training steps versus1024 for F2: update counts match, compute does not.

All raw fields, full checkpoint states, schedule cursors, RNG states and source snapshots remain in registered local artifacts. Fields share scoring formulas; component connectivity uses independent BFS. Training fields precede their update, checkpoints follow it. Recovery certifies completed CPU update boundaries, not GPU or abrupt writes.

## Evaluation results

| Arm | Member | Update | Growth | Firing seed | Connected | Mass/envelope | Joint |
|---|---|---:|---:|---:|---|---:|---|
| F3 | mapped_30-r0 | 0 | 16 | 2 | False | 0.1608797 | False |
| F3 | mapped_30-r0 | 0 | 50 | 2 | False | 0.49027273 | False |
| F3 | mapped_30-r0 | 1 | 16 | 2 | False | 0.14891423 | False |
| F3 | mapped_30-r0 | 1 | 50 | 2 | False | 0.46858674 | False |
| F3 | mapped_30-r0 | 2 | 16 | 0 | False | 0.14113027 | False |
| F3 | mapped_30-r0 | 2 | 24 | 0 | False | 0.22703257 | False |
| F3 | mapped_30-r0 | 2 | 32 | 0 | False | 0.36609405 | False |
| F3 | mapped_30-r0 | 2 | 40 | 0 | False | 0.43425792 | False |
| F3 | mapped_30-r0 | 2 | 50 | 0 | False | 0.44416201 | False |
| F3 | mapped_30-r0 | 2 | 64 | 0 | False | 0.44468546 | False |
| F3 | mapped_30-r0 | 2 | 16 | 1 | False | 0.13846283 | False |
| F3 | mapped_30-r0 | 2 | 24 | 1 | False | 0.22213747 | False |
| F3 | mapped_30-r0 | 2 | 32 | 1 | False | 0.35787615 | False |
| F3 | mapped_30-r0 | 2 | 40 | 1 | False | 0.42813015 | False |
| F3 | mapped_30-r0 | 2 | 50 | 1 | False | 0.44384381 | False |
| F3 | mapped_30-r0 | 2 | 64 | 1 | False | 0.44468546 | False |
| F3 | mapped_30-r0 | 2 | 16 | 2 | False | 0.13787323 | False |
| F3 | mapped_30-r0 | 2 | 24 | 2 | False | 0.22026286 | False |
| F3 | mapped_30-r0 | 2 | 32 | 2 | False | 0.35784966 | False |
| F3 | mapped_30-r0 | 2 | 40 | 2 | False | 0.43337053 | False |
| F3 | mapped_30-r0 | 2 | 50 | 2 | False | 0.44736135 | False |
| F3 | mapped_30-r0 | 2 | 64 | 2 | False | 0.44940412 | False |
| F3 | mapped_30-r1 | 0 | 16 | 2 | False | 0.12421025 | False |
| F3 | mapped_30-r1 | 0 | 50 | 2 | False | 0.37166923 | False |
| F3 | mapped_30-r1 | 1 | 16 | 2 | False | 0.11849251 | False |
| F3 | mapped_30-r1 | 1 | 50 | 2 | False | 0.36143559 | False |
| F3 | mapped_30-r1 | 2 | 16 | 0 | False | 0.11002097 | False |
| F3 | mapped_30-r1 | 2 | 24 | 0 | False | 0.17583166 | False |
| F3 | mapped_30-r1 | 2 | 32 | 0 | False | 0.28109089 | False |
| F3 | mapped_30-r1 | 2 | 40 | 0 | False | 0.32596704 | False |
| F3 | mapped_30-r1 | 2 | 50 | 0 | False | 0.33450061 | False |
| F3 | mapped_30-r1 | 2 | 64 | 0 | False | 0.33659154 | False |
| F3 | mapped_30-r1 | 2 | 16 | 1 | False | 0.10651849 | False |
| F3 | mapped_30-r1 | 2 | 24 | 1 | False | 0.17424601 | False |
| F3 | mapped_30-r1 | 2 | 32 | 1 | False | 0.27917701 | False |
| F3 | mapped_30-r1 | 2 | 40 | 1 | False | 0.32494146 | False |
| F3 | mapped_30-r1 | 2 | 50 | 1 | False | 0.33462375 | False |
| F3 | mapped_30-r1 | 2 | 64 | 1 | False | 0.33602151 | False |
| F3 | mapped_30-r1 | 2 | 16 | 2 | False | 0.11120947 | False |
| F3 | mapped_30-r1 | 2 | 24 | 2 | False | 0.17911072 | False |
| F3 | mapped_30-r1 | 2 | 32 | 2 | False | 0.2777282 | False |
| F3 | mapped_30-r1 | 2 | 40 | 2 | False | 0.32288381 | False |
| F3 | mapped_30-r1 | 2 | 50 | 2 | False | 0.33249235 | False |
| F3 | mapped_30-r1 | 2 | 64 | 2 | False | 0.33491385 | False |
| F3 | mass_3-r0 | 0 | 16 | 2 | False | 0.1608797 | False |
| F3 | mass_3-r0 | 0 | 50 | 2 | False | 0.49027273 | False |
| F3 | mass_3-r0 | 1 | 16 | 2 | False | 0.15131471 | False |
| F3 | mass_3-r0 | 1 | 50 | 2 | False | 0.47703132 | False |
| F3 | mass_3-r0 | 2 | 16 | 0 | False | 0.14420494 | False |
| F3 | mass_3-r0 | 2 | 24 | 0 | False | 0.23331928 | False |
| F3 | mass_3-r0 | 2 | 32 | 0 | False | 0.37633768 | False |
| F3 | mass_3-r0 | 2 | 40 | 0 | False | 0.44261426 | False |
| F3 | mass_3-r0 | 2 | 50 | 0 | False | 0.45149294 | False |
| F3 | mass_3-r0 | 2 | 64 | 0 | False | 0.45227766 | False |
| F3 | mass_3-r0 | 2 | 16 | 1 | False | 0.1415654 | False |
| F3 | mass_3-r0 | 2 | 24 | 1 | False | 0.22878045 | False |
| F3 | mass_3-r0 | 2 | 32 | 1 | False | 0.36746269 | False |
| F3 | mass_3-r0 | 2 | 40 | 1 | False | 0.43499723 | False |
| F3 | mass_3-r0 | 2 | 50 | 1 | False | 0.44888252 | False |
| F3 | mass_3-r0 | 2 | 64 | 1 | False | 0.44997686 | False |
| F3 | mass_3-r0 | 2 | 16 | 2 | False | 0.14062938 | False |
| F3 | mass_3-r0 | 2 | 24 | 2 | False | 0.22638209 | False |
| F3 | mass_3-r0 | 2 | 32 | 2 | False | 0.36786723 | False |
| F3 | mass_3-r0 | 2 | 40 | 2 | False | 0.44067052 | False |
| F3 | mass_3-r0 | 2 | 50 | 2 | False | 0.45255718 | False |
| F3 | mass_3-r0 | 2 | 64 | 2 | False | 0.45336226 | False |
| F3 | mass_3-r1 | 0 | 16 | 2 | False | 0.12421025 | False |
| F3 | mass_3-r1 | 0 | 50 | 2 | False | 0.37166923 | False |
| F3 | mass_3-r1 | 1 | 16 | 2 | False | 0.11887208 | False |
| F3 | mass_3-r1 | 1 | 50 | 2 | False | 0.36307469 | False |
| F3 | mass_3-r1 | 2 | 16 | 0 | False | 0.11079022 | False |
| F3 | mass_3-r1 | 2 | 24 | 0 | False | 0.17763597 | False |
| F3 | mass_3-r1 | 2 | 32 | 0 | False | 0.28335786 | False |
| F3 | mass_3-r1 | 2 | 40 | 0 | False | 0.32834351 | False |
| F3 | mass_3-r1 | 2 | 50 | 0 | False | 0.33639494 | False |
| F3 | mass_3-r1 | 2 | 64 | 0 | False | 0.33907235 | False |
| F3 | mass_3-r1 | 2 | 16 | 1 | False | 0.10726568 | False |
| F3 | mass_3-r1 | 2 | 24 | 1 | False | 0.17615989 | False |
| F3 | mass_3-r1 | 2 | 32 | 1 | False | 0.28175178 | False |
| F3 | mass_3-r1 | 2 | 40 | 1 | False | 0.32670593 | False |
| F3 | mass_3-r1 | 2 | 50 | 1 | False | 0.33625335 | False |
| F3 | mass_3-r1 | 2 | 64 | 1 | False | 0.34005377 | False |
| F3 | mass_3-r1 | 2 | 16 | 2 | False | 0.11198815 | False |
| F3 | mass_3-r1 | 2 | 24 | 2 | False | 0.18094508 | False |
| F3 | mass_3-r1 | 2 | 32 | 2 | False | 0.27994007 | False |
| F3 | mass_3-r1 | 2 | 40 | 2 | False | 0.32491323 | False |
| F3 | mass_3-r1 | 2 | 50 | 2 | False | 0.33556837 | False |
| F3 | mass_3-r1 | 2 | 64 | 2 | False | 0.3373656 | False |
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
