# F5 recovery: raw-access training

Run `20260923T212734Z_efe016a02413`.

One architecture change: persistent same-scaffold input; unchanged F4 raw-access objective. Original initialization, constant16 training, two development scenes, one training seed and two recipes. Both arms use64 updates/model and1024 recurrent training steps. Scoring overhead may differ.

All raw fields, full checkpoint states, schedule cursors, RNG states and source snapshots remain in registered local artifacts. Fields share scoring formulas; component connectivity uses independent BFS. Training fields precede their update, checkpoints follow it. Recovery certifies completed CPU update boundaries, not GPU or abrupt writes.

## Evaluation results

| Arm | Member | Update | Growth | Firing seed | Connected | Mass/envelope | Joint |
|---|---|---:|---:|---:|---|---:|---|
| F5 | prefix | 0 | 16 | 2 | False | 0.1608797 | False |
| F5 | prefix | 0 | 50 | 2 | False | 0.49027273 | False |
| F5 | prefix | 1 | 16 | 2 | False | 0.15153962 | False |
| F5 | prefix | 1 | 50 | 2 | False | 0.47712043 | False |
| F5 | repeat | 3 | 16 | 2 | False | 0.13389532 | False |
| F5 | repeat | 3 | 50 | 2 | False | 0.44990847 | False |
| F5 | resumed | 3 | 16 | 2 | False | 0.13389532 | False |
| F5 | resumed | 3 | 50 | 2 | False | 0.44990847 | False |
| F5 | whole | 0 | 16 | 2 | False | 0.1608797 | False |
| F5 | whole | 0 | 50 | 2 | False | 0.49027273 | False |
| F5 | whole | 1 | 16 | 2 | False | 0.15153962 | False |
| F5 | whole | 1 | 50 | 2 | False | 0.47712043 | False |
| F5 | whole | 3 | 16 | 2 | False | 0.13389532 | False |
| F5 | whole | 3 | 50 | 2 | False | 0.44990847 | False |
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
| F4 | mapped_30-r0 | 64 | 16 | 2 | False | 0.13935672 | False |
| F4 | mapped_30-r0 | 64 | 50 | 2 | True | 0.23434812 | False |
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
| F4 | mapped_30-r1 | 64 | 16 | 2 | False | 0.13583626 | False |
| F4 | mapped_30-r1 | 64 | 50 | 2 | True | 0.25715548 | False |
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
| F4 | mass_3-r0 | 64 | 16 | 2 | True | 0.22130166 | False |
| F4 | mass_3-r0 | 64 | 50 | 2 | True | 0.31707752 | False |
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
| F4 | mass_3-r1 | 64 | 16 | 2 | True | 0.22715679 | False |
| F4 | mass_3-r1 | 64 | 50 | 2 | True | 0.33743343 | False |

Budget3%-12%, tolerance1e-6. Strict material>0.5 component connectivity. Final grids reuse eight boundary evaluations explicitly; unique counts do not count them twice. F4 historical final-grid controls and all individual losses/metrics are in the JSON report.

## Verification

```json
{
  "baseline_parity": true,
  "training_update_exact_F4_matches": 12,
  "evaluation_record_exact_F4_matches": 24,
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
  "source_hashes_verified": 47,
  "unique_saved_fields_rescored": 22,
  "unique_evaluations": 14,
  "boundary_evaluations": 14,
  "final_grid_evaluations": 0,
  "checkpoint_schedule_cursors_verified": 10,
  "F4_boundary_controls_rescored": 56,
  "F4_final_controls_rescored": 72,
  "initial_F4_fields_exact": 4,
  "final_checkpoint_rollouts_exact": 0,
  "elapsed_caps_met": true
}
```

No holdout/generalization, walkability or mechanical safety claim. No automatic model/recipe promotion, paid training or production change.
