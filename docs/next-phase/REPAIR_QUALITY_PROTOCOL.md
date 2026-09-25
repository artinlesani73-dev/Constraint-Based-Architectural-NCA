# NR3: bounded repair-quality study, frozen before model-quality results

Prepared2026-09-25 after successful NR2 GPU process recovery. Preparation is local
only. GPU runs and every Drive action require separate explicit approval.

## Question and preserved baseline

Can the same small conditioned NCA repair missing volume while preserving intact
acceptable masses? Keep overall building-volume semantics, the same nine MT1
families and MG7 as the deployed procedural comparator. Interiors are not required.
No new architecture, loss coefficients, state pool, AMP, scheduler, resolution,
teacher generation or damage rerolls. Do not promote NR2's8-update checkpoints.

Three fresh models: seeds1201/1202/1203, each256 updates, batch1,16 recurrent steps,
float32, Adam0.001, gradient clip1. Each model uses all81 NL0 TRAIN variants from27
teachers, shuffled without replacement. Private firing seed=model seed+1, sampler
seed=model seed+2. Static conditioning and balanced BCE are the unchanged NR1/NR2
definitions. No target/cut mask/teacher route or seed enters network conditioning.

## Three independently bounded jobs

One explicit seed per invocation, at most600 seconds controlled execution,256
updates. Maximum1800 controlled training seconds across three jobs. The broader
proposal remains at most60 minutes of allocated GPU time including user setup,
uploads/downloads and idle time. The code cannot control provider billing; the
user must track that allocation and disconnect between verified jobs if needed.
Never automatically extend a failed/incomplete run or select a partial checkpoint
as a substitute final model. No automatic retry or continuation entry point.

The admitted stack is the returned Tesla T4 / PyTorch2.11.0+cu128 / CUDA12.8 /
cuDNN91900 / Python3.13.15 / NumPy2.1.3 stack. A change stops for review/preflight.
GPU UUID/host are recorded and checkpoint restore remains bound to their identity;
a type/version match alone does not promise repeatability on every machine.
Run each fresh seed separately. No new long-run GPU recovery guarantee is inferred
from NR2 beyond its observed completed-boundary replay.

Each update saves weights, Adam, complete RNG/sampler state, loss/gradient trace,
timing and pre-update raw rollout state. Boundaries0/64/128/192/256 also save an
independent TRAIN-row0 rollout. That diagnostic is not heldout evaluation. Preserve
all checkpoints, partial writes, failures, logs and process IDs. Maximum reserved
GPU memory80% of capacity; finite gradients/weights/loss required. A parent deadline
and owned-process cleanup bound the worker; EOF monitoring stops orphaned work.

## Backup plan and user handoff

The notebook is disarmed. It uploads only the exact local source/TRAIN ZIP to
Colab runtime storage and runs one explicitly selected approved seed. It never
mounts Drive, installs packages, uploads a full experiment automatically, or starts
the next seed. A per-seed exclusive marker prevents duplicate execution in the
same extraction. Re-extraction is not permission to retry.

After each seed, download its results ZIP and receipt. The assistant verifies
all member hashes and makes a local archive; then separately request permission
to save that exact ZIP/receipt into the sole project Drive folder and read them
back to verify. Start the next seed only after both copies are verified and its
compute allowance is approved. Notebook upload/edits/autosave require their own
explicit Drive scope. Prepare a fresh notebook rather than overwrite NR2.

No entire-VM-loss guarantee exists inside a seed job: local runtime checkpoints
can be lost before export. The proposed loss window is bounded to this one job,
at most256 updates/600 controlled seconds. Exact cross-VM restart is not certified;
retain downloaded evidence, declare incomplete work and obtain a reviewed retry
allowance if needed. This plan needs user acceptance before executing NR3. Never
claim the notebook-only Drive copy backs up all experiments.

## Evaluation separated from paid training

The training ZIP contains no validation/test arrays. Return all three finished
seed archives before model evaluation. The local evaluator verifies every archive
and seals all three update256 checkpoint hashes before opening heldout examples.
The final checkpoints are fixed; no best-checkpoint selection or retuning.

Authoritative geometry evaluation uses CPUfloat32 on the recorded local stack,
PyTorch2.8.0+cpu / NumPy2.5.2, matching the intended current local inference target.
It loads weights only; it does not resume a GPU optimizer on CPU. CPU/GPU outputs
need not be identical. Report this device distinction and do not call CPU timings
GPU deployment performance. NR2 GPU replay remains a separate mechanics result.

Validation diagnostics: all27 validation rows at checkpoints0/64/128/192/256,
32 steps, firing2101 (405 observations across three models). This concretizes the
older proposal's unspecified diagnostic schedule and avoids multiplying every
intermediate checkpoint by all horizon/randomness combinations. No retuning.
Final evaluation: every27 validation and54 test row at16/32/64 steps and firing
2101/2102/2103 (2187 observations). Primary test comparison uses32 steps only:
486 fields across3 models,54 rows,3 firing seeds. Save raw states/probabilities,
strict probability>0.5 fields, every metric and timings, including all failures.
The script checks its local10800s allowance between observations; a single ongoing
observation may finish beyond that deadline. Partial evidence cannot pass admission.
No lengthy heldout evaluation is executed during this preparation milestone.

Reuse all frozen NL0 scenes/targets/damages/baselines, exact request scalars and
MT1 formulas. Threshold and projection are declared, not learned legality.
Report intact/cube5/slab2 separately, every site's/request's results, all nine
families, IoU, missing-cell recovery, removed surviving cells, false additions,
request error,16/64-step sensitivity and model-to-model variation. Firing seeds
are repeated observations, not independently trained model replications. The six
synthetic sites share design ancestry and were already seen in procedural research.

## Frozen primary gate

For EACH trained seed, pool its108 damaged primary test fields (36 damaged rows,
3 firing seeds). Require median IoU minus each comparator's median IoU >=0.02.
Each deterministic comparator is represented by its36 damaged rows; repeating
those rows three times would leave its median and pass rate unchanged. This is
a difference of medians, not a median paired difference. Also require all-nine
pass rate no worse than either unchanged or fixed closing3 comparator.

Every primary intact field must have IoU>=0.99 and retain all-nine validity.
Every damaged field counted as an accepted all-nine repair must have absolute
requested-volume error <=max(8 cells,1% of domain voxels). No omitted cases,
duplicate observations, averaged-away failed training seed or tuned threshold.

The pre-existing NL0 damaged TEST comparator medians are0.9042553191 (unchanged)
and0.9465109229 (closing); pass rates7/36 and14/36. The proposed +0.02 requirement
is arithmetically feasible (below1), not a promise the model can meet it. This
uses already recorded comparator outcomes, not newly observed learned TEST output.
Passing admits a further repair study, not Studio replacement or architectural/
structural certification. Failure closes this recipe pending error analysis.

## Preparation validation

Use synthetic tests for completeness/duplicate protection, each-seed gates,
intact preservation, request-error tolerance, evaluator parity, heldout package
rejection and disarmed command behavior. Full regression follows. Rehearse exactly
one extracted CPU job with8 updates,600s cap, outside the repository, and compare
its checkpoints/states to the prior NR2 CPU control. This checks the new training
driver without executing the256-update study or consuming heldout model results.
The CPU rehearsal is explicitly labeled and the evaluator rejects it as training.

Record effective source/data/config hashes, all results and backup receipts.
Preserve NR2 packages/private reports/history unchanged. No Drive, GPU, push or
Studio operations are part of this local preparation.
