# F2: access-only learning comparison

Frozen 2026-09-23 before execution. User authorized continuing the local plan.
Configuration: experiments/configs/F2-access-training.json. Read D040 and the
preceding ACCESS_TRAINING_PLAN.md/A2 attribution before interpreting this study.

## Fixed comparison

Four independent models: ground-pair/minimal-smoke times mapped_30/mass_3.
Original Model C weights, training seed0, weak0.15 scaffold, hard_preclamp,
16 recurrent steps, Adam0.0001, clip1, constant scheduler,64 updates. Identical
to F1 in scene exposure, all eight other families, three regularizers, recipes,
architecture and optimizer. Replace only access with component_bottleneck_v2.
This includes endpoint/component semantics, worst destination and unbounded
spatial reach; it is not merely a source-coordinate edit. CPU/two threads only.

Evaluate boundaries0,1,3,8,16,32,64 at16/50 growth steps, independent firing seed2.
Save continuous fields, every checkpoint/update, old and new access definitions,
old binary metrics and independent component BFS, both recipe totals under both
definitions, all remaining families/regularizers, mass and critical-voxel details.
Joint outcome: binary connectivity at material>0.5 AND continuous envelope mass
3%-12% with1e-6 tolerance. Do not select only attractive intermediate boundaries.

## Mandatory gates and caps

1. F2B: old-objective mode, three updates for each of four members. Match F1
   updates1-3 and evaluations0,1,3 at both horizons exactly: trace, arrays, model,
   optimizer, scheduler, all RNG states and counters. Metadata differences only
   for declared protocol/source identity.120 seconds per process. This permits
   reuse of the immutable F1 64-update results, not a claim of fresh full replay.
2. F2R: actual candidate loop, whole3/prefix1/resumed2/repeat2 in fresh processes.
   Compare complete checkpoint trees, traces and fields; evaluations0,1,3 must
   match.120 seconds per process. Early gradients can remain inactive; this tests
   deterministic execution/recovery at completed CPU boundaries, not all later
   active-gradient states, abrupt writes, CUDA or AMP. Separate regression checks
   exercise the live candidate derivative and one-family replacement.
3. F2P: two candidate updates for each member, evaluations0,1,2 at16/50.
   Include checkpoint/field recording, setup and evaluation pair timing.
   Admission per member:1.5*(max(5,max(setup)+3)+64*p90(update)+7*max(eval pair)).
   Both600-second member and1800-second four-member estimates must pass.
   Pilot quality never changes schedule, settings or admission.120 seconds/worker.
4. F2: only after all gates,256 candidate updates/56 evaluations.600 seconds per
   member,1800 seconds total. Terminate at the cap, preserve partial evidence.
   No gate may silently raise these caps; failures/retries get linked new IDs.

## Verification and preservation

Run verifier rechecks artifact hashes, frozen source ZIP, all saved training/
evaluation objectives and projection, checkpoint metadata/counters, all binary
metrics and56 F1 fields under both definitions. Initial fields must match F1.
Replay all eight final study evaluations from the four checkpoints. Shared-formula
rescoring is not independent mathematical verification; candidate binary BFS is
independently implemented. Report every per-scene failure.

Source/config are frozen before the first experiment. Large evidence remains in
.local-artifacts/runs; small records/reports in Git. Exact source snapshots are
needed for historical resume if Git normalizes line endings. Resume workers only
into a NEW run/branch with matching metadata and verified completed checkpoint;
the coordinator does not automatically resume an entire interrupted matrix.
Never overwrite previous outputs or bypass source guards. Update RESUME with IDs.

One seed and two development scenes cannot establish generalization or a final
coefficient choice. Keep original/F1/D1/W1 evidence. No production promotion,
paid compute, Drive operation, architecture change, larger grid or added family.
The next decision follows joint outcomes and horizon behavior, not lower rescored
loss alone. All findings/decisions must be archived locally before handoff.

## F2L supplementary recovery check (registered while F2 is running)

Before inspecting final outcomes, add a fixed recovery diagnostic for ALL four
members: resume each update63 checkpoint in a fresh process and reproduce update64
plus its16/50-step evaluations. Compare full checkpoint trees, traces and fields
exactly against uninterrupted F2. No added exposure or quality-based selection.
Each worker cap120s. This tests trained states beyond the early three-update gate;
it does not prove nonzero access gradients for every replay or certify abrupt
writes/GPU. New immutable run, source snapshot and verification; original F2
training code/config/caps remain frozen. Wrapper check_access_late_recovery.py
adds no nca module and does not change the32 hashed F2 source files.

## Linked completion after the recorded interruption

Parent F220260923T135818Z_5520d5d80cec stopped at its overall cap with254 recorded
updates/54 evaluations: three complete models and fourth update62. Preserve its
interrupted status, logs and any unregistered partial files. The third member's
elapsed1137.74s>600s remains a timing deviation; a long system delay is observed,
not a measured active-CPU correction. Do not rerun completed models for timing.

Freeze a separate120-second continuation for only the fourth model's updates63/64
and final evaluations, using the verified completed62 checkpoint, same source/
metadata/RNG/optimizer. Import254 completed training and54 evaluation records with
hash-checked copies into a NEW child run. Keep the failed parent's process/result
records and cumulative elapsed. Verify the complete256/56 matrix afterwards.
This completes planned scientific evidence; it does not erase the interruption,
raise the original cap or establish a clean timing benchmark. Source snapshot and
wrapper hash retained. No paid compute or extra training exposure. An unrecorded
partial update in the killed worker may be repeated; do not count it as a saved
checkpoint. F2L will then check all four final updates on the assembled child.
