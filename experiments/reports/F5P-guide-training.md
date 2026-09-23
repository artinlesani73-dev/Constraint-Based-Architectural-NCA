# F5 pilot: raw-access training

Run `20260923T213256Z_42130eed3f5d`.

One architecture change: persistent same-scaffold input; unchanged F4 raw-access objective. Original initialization, constant16 training, two development scenes, one training seed and two recipes. Both arms use64 updates/model and1024 recurrent training steps. Scoring overhead may differ.

All raw fields, full checkpoint states, schedule cursors, RNG states and source snapshots remain in registered local artifacts. Fields share scoring formulas; component connectivity uses independent BFS. Training fields precede their update, checkpoints follow it. Recovery certifies completed CPU update boundaries, not GPU or abrupt writes.

## Evaluation results

| Arm | Member | Update | Growth | Firing seed | Connected | Mass/envelope | Joint |
|---|---|---:|---:|---:|---|---:|---|
| F5 | mapped_30-r0 | 0 | 16 | 2 | False | 0.1608797 | False |
| F5 | mapped_30-r0 | 0 | 50 | 2 | False | 0.49027273 | False |
| F5 | mapped_30-r0 | 1 | 16 | 2 | False | 0.14850628 | False |
| F5 | mapped_30-r0 | 1 | 50 | 2 | False | 0.46732646 | False |
| F5 | mapped_30-r0 | 2 | 16 | 0 | False | 0.13981062 | False |
| F5 | mapped_30-r0 | 2 | 24 | 0 | False | 0.22427113 | False |
| F5 | mapped_30-r0 | 2 | 32 | 0 | False | 0.36049953 | False |
| F5 | mapped_30-r0 | 2 | 40 | 0 | False | 0.42978767 | False |
| F5 | mapped_30-r0 | 2 | 50 | 0 | False | 0.44043872 | False |
| F5 | mapped_30-r0 | 2 | 64 | 0 | False | 0.44143167 | False |
| F5 | mapped_30-r0 | 2 | 16 | 1 | False | 0.13710468 | False |
| F5 | mapped_30-r0 | 2 | 24 | 1 | False | 0.21938261 | False |
| F5 | mapped_30-r0 | 2 | 32 | 1 | False | 0.35258391 | False |
| F5 | mapped_30-r0 | 2 | 40 | 1 | False | 0.42314902 | False |
| F5 | mapped_30-r0 | 2 | 50 | 1 | False | 0.43922904 | False |
| F5 | mapped_30-r0 | 2 | 64 | 1 | False | 0.44034708 | False |
| F5 | mapped_30-r0 | 2 | 16 | 2 | False | 0.13665973 | False |
| F5 | mapped_30-r0 | 2 | 24 | 2 | False | 0.21763115 | False |
| F5 | mapped_30-r0 | 2 | 32 | 2 | False | 0.35277721 | False |
| F5 | mapped_30-r0 | 2 | 40 | 2 | False | 0.42797413 | False |
| F5 | mapped_30-r0 | 2 | 50 | 2 | False | 0.44088471 | False |
| F5 | mapped_30-r0 | 2 | 64 | 2 | False | 0.44307825 | False |
| F5 | mapped_30-r1 | 0 | 16 | 2 | False | 0.12421025 | False |
| F5 | mapped_30-r1 | 0 | 50 | 2 | False | 0.37166923 | False |
| F5 | mapped_30-r1 | 1 | 16 | 2 | False | 0.11888804 | False |
| F5 | mapped_30-r1 | 1 | 50 | 2 | False | 0.36161846 | False |
| F5 | mapped_30-r1 | 2 | 16 | 0 | False | 0.11267861 | False |
| F5 | mapped_30-r1 | 2 | 24 | 0 | False | 0.18118106 | False |
| F5 | mapped_30-r1 | 2 | 32 | 0 | False | 0.28669453 | False |
| F5 | mapped_30-r1 | 2 | 40 | 0 | False | 0.33131832 | False |
| F5 | mapped_30-r1 | 2 | 50 | 0 | False | 0.34012747 | False |
| F5 | mapped_30-r1 | 2 | 64 | 0 | False | 0.34131277 | False |
| F5 | mapped_30-r1 | 2 | 16 | 1 | False | 0.10898283 | False |
| F5 | mapped_30-r1 | 2 | 24 | 1 | False | 0.17933798 | False |
| F5 | mapped_30-r1 | 2 | 32 | 1 | False | 0.2852132 | False |
| F5 | mapped_30-r1 | 2 | 40 | 1 | False | 0.33009821 | False |
| F5 | mapped_30-r1 | 2 | 50 | 1 | False | 0.33911335 | False |
| F5 | mapped_30-r1 | 2 | 64 | 1 | False | 0.34408602 | False |
| F5 | mapped_30-r1 | 2 | 16 | 2 | False | 0.11377218 | False |
| F5 | mapped_30-r1 | 2 | 24 | 2 | False | 0.18408149 | False |
| F5 | mapped_30-r1 | 2 | 32 | 2 | False | 0.28225198 | False |
| F5 | mapped_30-r1 | 2 | 40 | 2 | False | 0.32636946 | False |
| F5 | mapped_30-r1 | 2 | 50 | 2 | False | 0.33707827 | False |
| F5 | mapped_30-r1 | 2 | 64 | 2 | False | 0.33870968 | False |
| F5 | mass_3-r0 | 0 | 16 | 2 | False | 0.1608797 | False |
| F5 | mass_3-r0 | 0 | 50 | 2 | False | 0.49027273 | False |
| F5 | mass_3-r0 | 1 | 16 | 2 | False | 0.15153962 | False |
| F5 | mass_3-r0 | 1 | 50 | 2 | False | 0.47712043 | False |
| F5 | mass_3-r0 | 2 | 16 | 0 | False | 0.14624137 | False |
| F5 | mass_3-r0 | 2 | 24 | 0 | False | 0.23674116 | False |
| F5 | mass_3-r0 | 2 | 32 | 0 | False | 0.38109952 | False |
| F5 | mass_3-r0 | 2 | 40 | 0 | False | 0.44668826 | False |
| F5 | mass_3-r0 | 2 | 50 | 0 | False | 0.45595932 | False |
| F5 | mass_3-r0 | 2 | 64 | 0 | False | 0.45667255 | False |
| F5 | mass_3-r0 | 2 | 16 | 1 | False | 0.14370379 | False |
| F5 | mass_3-r0 | 2 | 24 | 1 | False | 0.23262773 | False |
| F5 | mass_3-r0 | 2 | 32 | 1 | False | 0.37221369 | False |
| F5 | mass_3-r0 | 2 | 40 | 1 | False | 0.44055054 | False |
| F5 | mass_3-r0 | 2 | 50 | 1 | False | 0.4540455 | False |
| F5 | mass_3-r0 | 2 | 64 | 1 | False | 0.45643529 | False |
| F5 | mass_3-r0 | 2 | 16 | 2 | False | 0.14241295 | False |
| F5 | mass_3-r0 | 2 | 24 | 2 | False | 0.22964212 | False |
| F5 | mass_3-r0 | 2 | 32 | 2 | False | 0.3721219 | False |
| F5 | mass_3-r0 | 2 | 40 | 2 | False | 0.44526884 | False |
| F5 | mass_3-r0 | 2 | 50 | 2 | False | 0.4577066 | False |
| F5 | mass_3-r0 | 2 | 64 | 2 | False | 0.46006033 | False |
| F5 | mass_3-r1 | 0 | 16 | 2 | False | 0.12421025 | False |
| F5 | mass_3-r1 | 0 | 50 | 2 | False | 0.37166923 | False |
| F5 | mass_3-r1 | 1 | 16 | 2 | False | 0.11937334 | False |
| F5 | mass_3-r1 | 1 | 50 | 2 | False | 0.3632713 | False |
| F5 | mass_3-r1 | 2 | 16 | 0 | False | 0.11339863 | False |
| F5 | mass_3-r1 | 2 | 24 | 0 | False | 0.18257129 | False |
| F5 | mass_3-r1 | 2 | 32 | 0 | False | 0.28853005 | False |
| F5 | mass_3-r1 | 2 | 40 | 0 | False | 0.3331576 | False |
| F5 | mass_3-r1 | 2 | 50 | 0 | False | 0.34192064 | False |
| F5 | mass_3-r1 | 2 | 64 | 0 | False | 0.34274194 | False |
| F5 | mass_3-r1 | 2 | 16 | 1 | False | 0.10969769 | False |
| F5 | mass_3-r1 | 2 | 24 | 1 | False | 0.18096811 | False |
| F5 | mass_3-r1 | 2 | 32 | 1 | False | 0.28712991 | False |
| F5 | mass_3-r1 | 2 | 40 | 1 | False | 0.33219984 | False |
| F5 | mass_3-r1 | 2 | 50 | 1 | False | 0.34395128 | False |
| F5 | mass_3-r1 | 2 | 64 | 1 | False | 0.34646901 | False |
| F5 | mass_3-r1 | 2 | 16 | 2 | False | 0.11445791 | False |
| F5 | mass_3-r1 | 2 | 24 | 2 | False | 0.18553634 | False |
| F5 | mass_3-r1 | 2 | 32 | 2 | False | 0.28390202 | False |
| F5 | mass_3-r1 | 2 | 40 | 2 | False | 0.32799014 | False |
| F5 | mass_3-r1 | 2 | 50 | 2 | False | 0.33781987 | False |
| F5 | mass_3-r1 | 2 | 64 | 2 | False | 0.34005377 | False |
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
  "unique_saved_fields_rescored": 96,
  "unique_evaluations": 88,
  "boundary_evaluations": 24,
  "final_grid_evaluations": 72,
  "checkpoint_schedule_cursors_verified": 12,
  "F4_boundary_controls_rescored": 56,
  "F4_final_controls_rescored": 72,
  "initial_F4_fields_exact": 8,
  "final_checkpoint_rollouts_exact": 0,
  "elapsed_caps_met": true,
  "admission": {
    "max_update_seconds": 4.915810299979057,
    "max_evaluation_pair_seconds": 7.324862000008579,
    "max_extra_grid_seconds": 60.308978600020055,
    "startup_allowance_seconds": 6.461011299979873,
    "safety_factor": 1.5,
    "estimated_member_seconds": 648.9838246480795,
    "estimated_total_seconds": 2595.935298592318,
    "admitted": true
  }
}
```

No holdout/generalization, walkability or mechanical safety claim. No automatic model/recipe promotion, paid training or production change.
