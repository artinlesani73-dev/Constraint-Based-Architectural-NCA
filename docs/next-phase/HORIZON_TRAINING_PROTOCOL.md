# F3 frozen local comparison protocol

2026-09-23. User approved advancing after H1. F3-horizon-training.json is the
machine-readable protocol. Read HORIZON_TRAINING_PLAN.md and D045.

## Fixed comparison

Four models: mapped_30/mass_3 crossed with ground-pair/minimal-smoke. Original
checkpoint, training seed0, weak0.15 scaffold, unchanged architecture, nine F2
families and three regularizers.64 updates/model; odd updates16 growth steps,
even updates50, full backpropagation. No pool, AMP, new margin or extra objective.
F2 control:20260923T143002Z_55aeaac95580; H1 final-grid control:
20260923T150119Z_f7b304516723. Historical failures remain recorded.
Equal update count, unequal compute:2112 versus1024 recurrent steps/model,
different firing-RNG consumption, plus evaluation work.

## Gates and local allowance

CPU, two threads, deterministic operations; no paid/cloud compute. Explicit
elapsed checks apply after successful waits as well as timeouts.

| Phase | Work | Worker cap | Total cap |
|---|---|---:|---:|
| F3B | Four constant16 models, updates1-3 | 180s | 900s |
| F3R | mass_3 ground: whole3, prefix1, resumed2, repeated resumed2 | 180s | 900s |
| F3P | Four mixed models, two updates plus complete final grid | 240s | 1200s |
| F3 | Four mixed models,64 updates plus fixed evaluations | 1200s | 3600s |

Parity requires exact F2 traces, fields and full checkpoint contents except
declared source/protocol/schedule identity. Never resume an old checkpoint under
different metadata. Recovery compares16/50/16 uninterrupted versus1+2 updates in
fresh processes, twice: model/optimizer/scheduler/RNG states, fields/evaluations.
Checkpoint metadata includes full schedule and version; completed_updates derives
the next horizon, additionally saved in a checked cursor record (null after64).
CPU update boundaries are certified, not interrupted writes or GPU recovery.

Pilot admission uses only timing:1.5*(max(5,max(setup)+3)+32*max(update16)
+32*max(update50)+7*max(evaluation_pair)+max(extra_final_grid)) per member.
Require <=1200s/member and <=3600s across four. Includes evidence writes; grid
timing excludes two reused boundary evaluations. Do not admit on quality.
Fresh run IDs/source snapshots for all attempts. No automatic retry or shortened
matrix; revise the protocol explicitly if an allowance is exceeded.

## Evaluation and verification

Boundaries0,1,3,8,16,32,64 at16/50 steps, firing seed2:56 full-study cases.
Final grid:16,24,32,40,50,64 steps with firing seeds0,1,2:72 records, eight reused
from final boundaries. Full study120 unique evaluations +256 pre-update fields.
Pilot24 boundaries/72 grid records,88 unique evaluations +8 training fields.
Parity12 updates/24 evaluations; recovery8 updates/14 evaluations.

Save every checkpoint, trace, source/RNG identity, raw/material field, all loss
terms, regularizers, gradient norm before clipping, learning rate, saturation,
binary metrics, both access definitions and both recipe totals. Verify all
unique fields, source hashes, checkpoint counters/cursors and exact matrix.
Rescore56 F2 boundaries and72 H1 F2 controls; replay eight final F3 rollouts.
Primary outcome: same-field connectivity AND continuous3%-12% mass budget,
tolerance1e-6, strict occupancy>0.5. All scene/recipe/horizon failures retained.
No post-hoc seed/stopping-rule selection. Shared scoring formulas, independent
component BFS; one training seed/two seen scenes, no generalization/promotion.

## Commands and recovery

Project .venv/Scripts/python.exe with scripts/run_horizon_training.py:
`--mode parity`; `--mode recovery --parity-run <B>`;
`--mode pilot --parity-run <B> --recovery-run <R>`;
`--mode study --parity-run <B> --recovery-run <R> --pilot-run <P>`.
After each completed phase: scripts/report_horizon_training.py <run-id>.
Reports refuse overwrite. Parent linkage supports fresh attempts; automatic
partial-study continuation is not implemented. On interruption inspect processes
and evidence, retain failed status, and implement an explicit linked continuation
with strict source checks before executing it. Exact source ZIPs are required
when source identity changes. No Drive operation, push or paid training.
