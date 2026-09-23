# F5 study: raw-access training

Run `20260923T214035Z_566da8c507c0`.

One architecture change: persistent same-scaffold input; unchanged F4 raw-access objective. Original initialization, constant16 training, two development scenes, one training seed and two recipes. Both arms use64 updates/model and1024 recurrent training steps. Scoring overhead may differ.

All raw fields, full checkpoint states, schedule cursors, RNG states and source snapshots remain in registered local artifacts. Fields share scoring formulas; component connectivity uses independent BFS. Training fields precede their update, checkpoints follow it. Recovery certifies completed CPU update boundaries, not GPU or abrupt writes.

## Evaluation results

| Arm | Member | Update | Growth | Firing seed | Connected | Mass/envelope | Joint |
|---|---|---:|---:|---:|---|---:|---|
| F5 | mapped_30-r0 | 0 | 16 | 2 | False | 0.1608797 | False |
| F5 | mapped_30-r0 | 0 | 50 | 2 | False | 0.49027273 | False |
| F5 | mapped_30-r0 | 1 | 16 | 2 | False | 0.14850628 | False |
| F5 | mapped_30-r0 | 1 | 50 | 2 | False | 0.46732646 | False |
| F5 | mapped_30-r0 | 3 | 16 | 2 | False | 0.12579265 | False |
| F5 | mapped_30-r0 | 3 | 50 | 2 | False | 0.42672604 | False |
| F5 | mapped_30-r0 | 8 | 16 | 2 | False | 0.091983408 | False |
| F5 | mapped_30-r0 | 8 | 50 | 2 | False | 0.10248319 | False |
| F5 | mapped_30-r0 | 16 | 16 | 2 | False | 0.10012098 | False |
| F5 | mapped_30-r0 | 16 | 50 | 2 | False | 0.14871293 | False |
| F5 | mapped_30-r0 | 32 | 16 | 2 | False | 0.12837498 | False |
| F5 | mapped_30-r0 | 32 | 50 | 2 | False | 0.19429414 | False |
| F5 | mapped_30-r0 | 64 | 16 | 0 | False | 0.14260212 | False |
| F5 | mapped_30-r0 | 64 | 24 | 0 | True | 0.17143071 | False |
| F5 | mapped_30-r0 | 64 | 32 | 0 | True | 0.186351 | False |
| F5 | mapped_30-r0 | 64 | 40 | 0 | True | 0.19521806 | False |
| F5 | mapped_30-r0 | 64 | 50 | 0 | True | 0.20036477 | False |
| F5 | mapped_30-r0 | 64 | 64 | 0 | True | 0.20103602 | False |
| F5 | mapped_30-r0 | 64 | 16 | 1 | False | 0.14088064 | False |
| F5 | mapped_30-r0 | 64 | 24 | 1 | True | 0.17118865 | False |
| F5 | mapped_30-r0 | 64 | 32 | 1 | True | 0.18698001 | False |
| F5 | mapped_30-r0 | 64 | 40 | 1 | True | 0.19515105 | False |
| F5 | mapped_30-r0 | 64 | 50 | 1 | True | 0.19903614 | False |
| F5 | mapped_30-r0 | 64 | 64 | 1 | True | 0.19987245 | False |
| F5 | mapped_30-r0 | 64 | 16 | 2 | False | 0.13826843 | False |
| F5 | mapped_30-r0 | 64 | 24 | 2 | True | 0.16797258 | False |
| F5 | mapped_30-r0 | 64 | 32 | 2 | True | 0.18665691 | False |
| F5 | mapped_30-r0 | 64 | 40 | 2 | True | 0.19518921 | False |
| F5 | mapped_30-r0 | 64 | 50 | 2 | True | 0.19861923 | False |
| F5 | mapped_30-r0 | 64 | 64 | 2 | True | 0.20044036 | False |
| F5 | mapped_30-r1 | 0 | 16 | 2 | False | 0.12421025 | False |
| F5 | mapped_30-r1 | 0 | 50 | 2 | False | 0.37166923 | False |
| F5 | mapped_30-r1 | 1 | 16 | 2 | False | 0.11888804 | False |
| F5 | mapped_30-r1 | 1 | 50 | 2 | False | 0.36161846 | False |
| F5 | mapped_30-r1 | 3 | 16 | 2 | False | 0.10918399 | False |
| F5 | mapped_30-r1 | 3 | 50 | 2 | False | 0.33266109 | False |
| F5 | mapped_30-r1 | 8 | 16 | 2 | False | 0.093683891 | False |
| F5 | mapped_30-r1 | 8 | 50 | 2 | False | 0.15483031 | False |
| F5 | mapped_30-r1 | 16 | 16 | 2 | False | 0.11559954 | False |
| F5 | mapped_30-r1 | 16 | 50 | 2 | False | 0.19905488 | False |
| F5 | mapped_30-r1 | 32 | 16 | 2 | False | 0.12222396 | False |
| F5 | mapped_30-r1 | 32 | 50 | 2 | False | 0.19974941 | False |
| F5 | mapped_30-r1 | 64 | 16 | 0 | False | 0.13748111 | False |
| F5 | mapped_30-r1 | 64 | 24 | 0 | True | 0.1771491 | False |
| F5 | mapped_30-r1 | 64 | 32 | 0 | True | 0.20362473 | False |
| F5 | mapped_30-r1 | 64 | 40 | 0 | True | 0.21549922 | False |
| F5 | mapped_30-r1 | 64 | 50 | 0 | True | 0.21978362 | False |
| F5 | mapped_30-r1 | 64 | 64 | 0 | True | 0.22043011 | False |
| F5 | mapped_30-r1 | 64 | 16 | 1 | False | 0.13211638 | False |
| F5 | mapped_30-r1 | 64 | 24 | 1 | True | 0.17500006 | False |
| F5 | mapped_30-r1 | 64 | 32 | 1 | True | 0.20266806 | False |
| F5 | mapped_30-r1 | 64 | 40 | 1 | True | 0.21355116 | False |
| F5 | mapped_30-r1 | 64 | 50 | 1 | True | 0.21770719 | False |
| F5 | mapped_30-r1 | 64 | 64 | 1 | True | 0.21774194 | False |
| F5 | mapped_30-r1 | 64 | 16 | 2 | False | 0.13639846 | False |
| F5 | mapped_30-r1 | 64 | 24 | 2 | True | 0.17773864 | False |
| F5 | mapped_30-r1 | 64 | 32 | 2 | True | 0.19903497 | False |
| F5 | mapped_30-r1 | 64 | 40 | 2 | True | 0.21237181 | False |
| F5 | mapped_30-r1 | 64 | 50 | 2 | True | 0.216667 | False |
| F5 | mapped_30-r1 | 64 | 64 | 2 | True | 0.21881095 | False |
| F5 | mass_3-r0 | 0 | 16 | 2 | False | 0.1608797 | False |
| F5 | mass_3-r0 | 0 | 50 | 2 | False | 0.49027273 | False |
| F5 | mass_3-r0 | 1 | 16 | 2 | False | 0.15153962 | False |
| F5 | mass_3-r0 | 1 | 50 | 2 | False | 0.47712043 | False |
| F5 | mass_3-r0 | 3 | 16 | 2 | False | 0.13389532 | False |
| F5 | mass_3-r0 | 3 | 50 | 2 | False | 0.44990847 | False |
| F5 | mass_3-r0 | 8 | 16 | 2 | False | 0.10515102 | False |
| F5 | mass_3-r0 | 8 | 50 | 2 | False | 0.15751167 | False |
| F5 | mass_3-r0 | 16 | 16 | 2 | False | 0.11953101 | False |
| F5 | mass_3-r0 | 16 | 50 | 2 | False | 0.17937471 | False |
| F5 | mass_3-r0 | 32 | 16 | 2 | False | 0.15961321 | False |
| F5 | mass_3-r0 | 32 | 50 | 2 | False | 0.24924281 | False |
| F5 | mass_3-r0 | 64 | 16 | 0 | True | 0.23618017 | False |
| F5 | mass_3-r0 | 64 | 24 | 0 | True | 0.29158637 | False |
| F5 | mass_3-r0 | 64 | 32 | 0 | True | 0.31340972 | False |
| F5 | mass_3-r0 | 64 | 40 | 0 | True | 0.31687018 | False |
| F5 | mass_3-r0 | 64 | 50 | 0 | True | 0.31736445 | False |
| F5 | mass_3-r0 | 64 | 64 | 0 | True | 0.32017943 | False |
| F5 | mass_3-r0 | 64 | 16 | 1 | True | 0.23624656 | False |
| F5 | mass_3-r0 | 64 | 24 | 1 | True | 0.2939457 | False |
| F5 | mass_3-r0 | 64 | 32 | 1 | True | 0.31633657 | False |
| F5 | mass_3-r0 | 64 | 40 | 1 | True | 0.31880903 | False |
| F5 | mass_3-r0 | 64 | 50 | 1 | True | 0.32108682 | False |
| F5 | mass_3-r0 | 64 | 64 | 1 | True | 0.32270417 | False |
| F5 | mass_3-r0 | 64 | 16 | 2 | True | 0.23564586 | False |
| F5 | mass_3-r0 | 64 | 24 | 2 | True | 0.29212952 | False |
| F5 | mass_3-r0 | 64 | 32 | 2 | True | 0.31002668 | False |
| F5 | mass_3-r0 | 64 | 40 | 2 | True | 0.3162019 | False |
| F5 | mass_3-r0 | 64 | 50 | 2 | True | 0.31839353 | False |
| F5 | mass_3-r0 | 64 | 64 | 2 | True | 0.32037133 | False |
| F5 | mass_3-r1 | 0 | 16 | 2 | False | 0.12421025 | False |
| F5 | mass_3-r1 | 0 | 50 | 2 | False | 0.37166923 | False |
| F5 | mass_3-r1 | 1 | 16 | 2 | False | 0.11937334 | False |
| F5 | mass_3-r1 | 1 | 50 | 2 | False | 0.3632713 | False |
| F5 | mass_3-r1 | 3 | 16 | 2 | False | 0.10992268 | False |
| F5 | mass_3-r1 | 3 | 50 | 2 | False | 0.33317068 | False |
| F5 | mass_3-r1 | 8 | 16 | 2 | False | 0.094319053 | False |
| F5 | mass_3-r1 | 8 | 50 | 2 | False | 0.15629111 | False |
| F5 | mass_3-r1 | 16 | 16 | 2 | False | 0.11614401 | False |
| F5 | mass_3-r1 | 16 | 50 | 2 | False | 0.20681414 | False |
| F5 | mass_3-r1 | 32 | 16 | 2 | False | 0.16837697 | False |
| F5 | mass_3-r1 | 32 | 50 | 2 | False | 0.27578691 | False |
| F5 | mass_3-r1 | 64 | 16 | 0 | True | 0.24200888 | False |
| F5 | mass_3-r1 | 64 | 24 | 0 | True | 0.31486154 | False |
| F5 | mass_3-r1 | 64 | 32 | 0 | True | 0.33441597 | False |
| F5 | mass_3-r1 | 64 | 40 | 0 | True | 0.33876622 | False |
| F5 | mass_3-r1 | 64 | 50 | 0 | True | 0.34248543 | False |
| F5 | mass_3-r1 | 64 | 64 | 0 | True | 0.34462604 | False |
| F5 | mass_3-r1 | 64 | 16 | 1 | True | 0.2368256 | False |
| F5 | mass_3-r1 | 64 | 24 | 1 | True | 0.31297916 | False |
| F5 | mass_3-r1 | 64 | 32 | 1 | True | 0.33383229 | False |
| F5 | mass_3-r1 | 64 | 40 | 1 | True | 0.33905992 | False |
| F5 | mass_3-r1 | 64 | 50 | 1 | True | 0.34075743 | False |
| F5 | mass_3-r1 | 64 | 64 | 1 | True | 0.3419202 | False |
| F5 | mass_3-r1 | 64 | 16 | 2 | True | 0.24834102 | False |
| F5 | mass_3-r1 | 64 | 24 | 2 | True | 0.31561914 | False |
| F5 | mass_3-r1 | 64 | 32 | 2 | True | 0.3347058 | False |
| F5 | mass_3-r1 | 64 | 40 | 2 | True | 0.33991295 | False |
| F5 | mass_3-r1 | 64 | 50 | 2 | True | 0.34693643 | False |
| F5 | mass_3-r1 | 64 | 64 | 2 | True | 0.35025409 | False |
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
  "unique_saved_fields_rescored": 376,
  "unique_evaluations": 120,
  "boundary_evaluations": 56,
  "final_grid_evaluations": 72,
  "checkpoint_schedule_cursors_verified": 260,
  "F4_boundary_controls_rescored": 56,
  "F4_final_controls_rescored": 72,
  "initial_F4_fields_exact": 8,
  "final_checkpoint_rollouts_exact": 8,
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
