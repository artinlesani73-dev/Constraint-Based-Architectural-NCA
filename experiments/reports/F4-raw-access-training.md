# F4 study: raw-access training

Run `20260923T202649Z_75034cca563c`.

One access-family change: score raw maximin before clipping. Original initialization, constant16 training, two development scenes, one training seed and two recipes. Both arms use64 updates/model and1024 recurrent training steps. Scoring overhead may differ.

All raw fields, full checkpoint states, schedule cursors, RNG states and source snapshots remain in registered local artifacts. Fields share scoring formulas; component connectivity uses independent BFS. Training fields precede their update, checkpoints follow it. Recovery certifies completed CPU update boundaries, not GPU or abrupt writes.

## Evaluation results

| Arm | Member | Update | Growth | Firing seed | Connected | Mass/envelope | Joint |
|---|---|---:|---:|---:|---|---:|---|
| F4 | mapped_30-r0 | 0 | 16 | 2 | False | 0.1608797 | False |
| F4 | mapped_30-r0 | 0 | 50 | 2 | False | 0.49027273 | False |
| F4 | mapped_30-r0 | 1 | 16 | 2 | False | 0.14891423 | False |
| F4 | mapped_30-r0 | 1 | 50 | 2 | False | 0.46858674 | False |
| F4 | mapped_30-r0 | 3 | 16 | 2 | False | 0.12677336 | False |
| F4 | mapped_30-r0 | 3 | 50 | 2 | False | 0.43090522 | False |
| F4 | mapped_30-r0 | 8 | 16 | 2 | False | 0.092456818 | False |
| F4 | mapped_30-r0 | 8 | 50 | 2 | False | 0.1054687 | False |
| F4 | mapped_30-r0 | 16 | 16 | 2 | False | 0.099558905 | False |
| F4 | mapped_30-r0 | 16 | 50 | 2 | False | 0.15323316 | False |
| F4 | mapped_30-r0 | 32 | 16 | 2 | False | 0.12526089 | False |
| F4 | mapped_30-r0 | 32 | 50 | 2 | False | 0.19160235 | False |
| F4 | mapped_30-r0 | 64 | 16 | 0 | False | 0.14273328 | False |
| F4 | mapped_30-r0 | 64 | 24 | 0 | False | 0.17977417 | False |
| F4 | mapped_30-r0 | 64 | 32 | 0 | True | 0.21069361 | False |
| F4 | mapped_30-r0 | 64 | 40 | 0 | True | 0.22944945 | False |
| F4 | mapped_30-r0 | 64 | 50 | 0 | True | 0.23287393 | False |
| F4 | mapped_30-r0 | 64 | 64 | 0 | True | 0.23359072 | False |
| F4 | mapped_30-r0 | 64 | 16 | 1 | False | 0.14086299 | False |
| F4 | mapped_30-r0 | 64 | 24 | 1 | False | 0.17800637 | False |
| F4 | mapped_30-r0 | 64 | 32 | 1 | True | 0.21351652 | False |
| F4 | mapped_30-r0 | 64 | 40 | 1 | True | 0.23073682 | False |
| F4 | mapped_30-r0 | 64 | 50 | 1 | True | 0.23542477 | False |
| F4 | mapped_30-r0 | 64 | 64 | 1 | True | 0.23644252 | False |
| F4 | mapped_30-r0 | 64 | 16 | 2 | False | 0.13935672 | False |
| F4 | mapped_30-r0 | 64 | 24 | 2 | True | 0.17718481 | False |
| F4 | mapped_30-r0 | 64 | 32 | 2 | True | 0.2150636 | False |
| F4 | mapped_30-r0 | 64 | 40 | 2 | True | 0.22990079 | False |
| F4 | mapped_30-r0 | 64 | 50 | 2 | True | 0.23434812 | False |
| F4 | mapped_30-r0 | 64 | 64 | 2 | True | 0.23427722 | False |
| F4 | mapped_30-r1 | 0 | 16 | 2 | False | 0.12421025 | False |
| F4 | mapped_30-r1 | 0 | 50 | 2 | False | 0.37166923 | False |
| F4 | mapped_30-r1 | 1 | 16 | 2 | False | 0.11849251 | False |
| F4 | mapped_30-r1 | 1 | 50 | 2 | False | 0.36143559 | False |
| F4 | mapped_30-r1 | 3 | 16 | 2 | False | 0.10788681 | False |
| F4 | mapped_30-r1 | 3 | 50 | 2 | False | 0.33209592 | False |
| F4 | mapped_30-r1 | 8 | 16 | 2 | False | 0.090215877 | False |
| F4 | mapped_30-r1 | 8 | 50 | 2 | False | 0.14989327 | False |
| F4 | mapped_30-r1 | 16 | 16 | 2 | False | 0.10840274 | False |
| F4 | mapped_30-r1 | 16 | 50 | 2 | False | 0.17210956 | False |
| F4 | mapped_30-r1 | 32 | 16 | 2 | False | 0.12530728 | False |
| F4 | mapped_30-r1 | 32 | 50 | 2 | False | 0.21284087 | False |
| F4 | mapped_30-r1 | 64 | 16 | 0 | False | 0.13648768 | False |
| F4 | mapped_30-r1 | 64 | 24 | 0 | True | 0.19040348 | False |
| F4 | mapped_30-r1 | 64 | 32 | 0 | True | 0.24231373 | False |
| F4 | mapped_30-r1 | 64 | 40 | 0 | True | 0.25527737 | False |
| F4 | mapped_30-r1 | 64 | 50 | 0 | True | 0.25672042 | False |
| F4 | mapped_30-r1 | 64 | 64 | 0 | True | 0.25672042 | False |
| F4 | mapped_30-r1 | 64 | 16 | 1 | False | 0.13087715 | False |
| F4 | mapped_30-r1 | 64 | 24 | 1 | False | 0.18780741 | False |
| F4 | mapped_30-r1 | 64 | 32 | 1 | True | 0.23902635 | False |
| F4 | mapped_30-r1 | 64 | 40 | 1 | True | 0.25556874 | False |
| F4 | mapped_30-r1 | 64 | 50 | 1 | True | 0.25672042 | False |
| F4 | mapped_30-r1 | 64 | 64 | 1 | True | 0.25672042 | False |
| F4 | mapped_30-r1 | 64 | 16 | 2 | False | 0.13583626 | False |
| F4 | mapped_30-r1 | 64 | 24 | 2 | False | 0.19227299 | False |
| F4 | mapped_30-r1 | 64 | 32 | 2 | True | 0.24095896 | False |
| F4 | mapped_30-r1 | 64 | 40 | 2 | True | 0.2555337 | False |
| F4 | mapped_30-r1 | 64 | 50 | 2 | True | 0.25715548 | False |
| F4 | mapped_30-r1 | 64 | 64 | 2 | True | 0.25811318 | False |
| F4 | mass_3-r0 | 0 | 16 | 2 | False | 0.1608797 | False |
| F4 | mass_3-r0 | 0 | 50 | 2 | False | 0.49027273 | False |
| F4 | mass_3-r0 | 1 | 16 | 2 | False | 0.15131471 | False |
| F4 | mass_3-r0 | 1 | 50 | 2 | False | 0.47703132 | False |
| F4 | mass_3-r0 | 3 | 16 | 2 | False | 0.13319834 | False |
| F4 | mass_3-r0 | 3 | 50 | 2 | False | 0.44973588 | False |
| F4 | mass_3-r0 | 8 | 16 | 2 | False | 0.10318568 | False |
| F4 | mass_3-r0 | 8 | 50 | 2 | False | 0.1563925 | False |
| F4 | mass_3-r0 | 16 | 16 | 2 | False | 0.11536168 | False |
| F4 | mass_3-r0 | 16 | 50 | 2 | False | 0.1683626 | False |
| F4 | mass_3-r0 | 32 | 16 | 2 | False | 0.1474496 | False |
| F4 | mass_3-r0 | 32 | 50 | 2 | False | 0.22979306 | False |
| F4 | mass_3-r0 | 64 | 16 | 0 | True | 0.22208007 | False |
| F4 | mass_3-r0 | 64 | 24 | 0 | True | 0.28433868 | False |
| F4 | mass_3-r0 | 64 | 32 | 0 | True | 0.30539247 | False |
| F4 | mass_3-r0 | 64 | 40 | 0 | True | 0.3143037 | False |
| F4 | mass_3-r0 | 64 | 50 | 0 | True | 0.31659204 | False |
| F4 | mass_3-r0 | 64 | 64 | 0 | True | 0.31670281 | False |
| F4 | mass_3-r0 | 64 | 16 | 1 | True | 0.22135995 | False |
| F4 | mass_3-r0 | 64 | 24 | 1 | True | 0.2879945 | False |
| F4 | mass_3-r0 | 64 | 32 | 1 | True | 0.30489078 | False |
| F4 | mass_3-r0 | 64 | 40 | 1 | True | 0.31505239 | False |
| F4 | mass_3-r0 | 64 | 50 | 1 | True | 0.31778741 | False |
| F4 | mass_3-r0 | 64 | 64 | 1 | True | 0.31778741 | False |
| F4 | mass_3-r0 | 64 | 16 | 2 | True | 0.22130166 | False |
| F4 | mass_3-r0 | 64 | 24 | 2 | True | 0.28527018 | False |
| F4 | mass_3-r0 | 64 | 32 | 2 | True | 0.30450928 | False |
| F4 | mass_3-r0 | 64 | 40 | 2 | True | 0.31192863 | False |
| F4 | mass_3-r0 | 64 | 50 | 2 | True | 0.31707752 | False |
| F4 | mass_3-r0 | 64 | 64 | 2 | True | 0.31778741 | False |
| F4 | mass_3-r1 | 0 | 16 | 2 | False | 0.12421025 | False |
| F4 | mass_3-r1 | 0 | 50 | 2 | False | 0.37166923 | False |
| F4 | mass_3-r1 | 1 | 16 | 2 | False | 0.11887208 | False |
| F4 | mass_3-r1 | 1 | 50 | 2 | False | 0.36307469 | False |
| F4 | mass_3-r1 | 3 | 16 | 2 | False | 0.10843865 | False |
| F4 | mass_3-r1 | 3 | 50 | 2 | False | 0.33261329 | False |
| F4 | mass_3-r1 | 8 | 16 | 2 | False | 0.09050478 | False |
| F4 | mass_3-r1 | 8 | 50 | 2 | False | 0.15152413 | False |
| F4 | mass_3-r1 | 16 | 16 | 2 | False | 0.10825745 | False |
| F4 | mass_3-r1 | 16 | 50 | 2 | False | 0.17202988 | False |
| F4 | mass_3-r1 | 32 | 16 | 2 | False | 0.15365547 | False |
| F4 | mass_3-r1 | 32 | 50 | 2 | False | 0.25242296 | False |
| F4 | mass_3-r1 | 64 | 16 | 0 | True | 0.22441679 | False |
| F4 | mass_3-r1 | 64 | 24 | 0 | True | 0.30131036 | False |
| F4 | mass_3-r1 | 64 | 32 | 0 | True | 0.32434362 | False |
| F4 | mass_3-r1 | 64 | 40 | 0 | True | 0.3327933 | False |
| F4 | mass_3-r1 | 64 | 50 | 0 | True | 0.33474225 | False |
| F4 | mass_3-r1 | 64 | 64 | 0 | True | 0.33483976 | False |
| F4 | mass_3-r1 | 64 | 16 | 1 | True | 0.21652006 | False |
| F4 | mass_3-r1 | 64 | 24 | 1 | True | 0.30119988 | False |
| F4 | mass_3-r1 | 64 | 32 | 1 | True | 0.32315099 | False |
| F4 | mass_3-r1 | 64 | 40 | 1 | True | 0.33117467 | False |
| F4 | mass_3-r1 | 64 | 50 | 1 | True | 0.33592701 | False |
| F4 | mass_3-r1 | 64 | 64 | 1 | True | 0.33602151 | False |
| F4 | mass_3-r1 | 64 | 16 | 2 | True | 0.22715679 | False |
| F4 | mass_3-r1 | 64 | 24 | 2 | True | 0.30082935 | False |
| F4 | mass_3-r1 | 64 | 32 | 2 | True | 0.32280982 | False |
| F4 | mass_3-r1 | 64 | 40 | 2 | True | 0.33331552 | False |
| F4 | mass_3-r1 | 64 | 50 | 2 | True | 0.33743343 | False |
| F4 | mass_3-r1 | 64 | 64 | 2 | True | 0.33870968 | False |
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
    "max_update_seconds": 4.672764199989615,
    "max_evaluation_pair_seconds": 5.994843899999978,
    "max_extra_grid_seconds": 46.76010260000476,
    "startup_allowance_seconds": 5.036728100007167,
    "safety_factor": 1.5,
    "estimated_member_seconds": 589.2264701990207,
    "estimated_total_seconds": 2356.905880796083,
    "admitted": true
  }
}
```

No holdout/generalization, walkability or mechanical safety claim. No automatic model/recipe promotion, paid training or production change.
