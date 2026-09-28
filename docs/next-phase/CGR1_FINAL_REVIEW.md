# CGR1 final development review — 2026-09-28

The connected constructive model improves the NR5 development results, but fails two frozen acceptance conditions. Keep it experimental; MG7 remains live.

## Evidence and evaluation

User supplied run `20260928T153419Z_44ad70fb1464`: 256 CUDA updates completed, seed 1201, 32 growth steps. Controlled duration 61.469 seconds, peak reserved GPU memory 710 MiB, cleanup exit 0. Verified outer receipt SHA256, all 1,036 payload hashes and exact unique ZIP membership; verified final checkpoint bytes/hash, semantic identity, optimizer/sampler cursor and trace. This establishes successful GPU execution of v2, not exact GPU interruption/recovery equivalence. The original failed v1 attempt remains preserved.

Final checkpoint 256 only; CPU float32, 32 steps, firing seed 2101, all 27 existing validation/development cases. Scored accepted binary occupancy, without cleanup or threshold changes. Saved all proposals, birth masks, hidden state, inputs, outputs and per-case metrics. Verified the 27 prior NR5 array hashes and matching case/damage/baseline identities. No TEST cases inspected. Repeated development cases and one training seed cannot establish generalization.

## Comparison

| Metric | NR5 | CGR1 |
|---|---:|---:|
| All nine checks, all cases | 17/27 | 24/27 |
| All nine checks, damaged cases | 11/18 | 15/18 |
| Damaged median IoU | 0.97058 | 0.97504 |
| Damaged excess voxels | 325 | 245 |
| Damaged correctly recovered voxels | 1,945 | 1,959 |
| Damaged median absolute requested-volume error, cells | 19 | 15.5 |
| Intact all-nine validity | 6/9 | 9/9 |
| Intact median IoU | 0.98633 | 0.99426 |
| Intact excess voxels | 172 | 117 |

The simple closing3 baseline also passes 24/27 overall and 15/18 damaged cases. It recovers fewer damaged cells (1,464) but adds only 13 excess damaged cells. CGR1 is therefore not a universal winner against the simple baseline.

## What remains wrong

All three failed cases are cube5 damage. Each fails both access and thickness: the occupied field is attached, but its thick bulk does not fully connect to the interfaces. All 27 pass support; detached correct and detached excess voxel counts are both zero. Attachment and no deletion are enforced properties, not learned guarantees. Single-cell connectivity is insufficient for volumetric connectivity.

Six of eight frozen conditions pass. Damaged validity is 15/18, below the required 17/18. Three intact examples have IoU below the required 0.99 for every intact example, despite their median exceeding 0.99. Intact inputs still accumulate 117 unwanted cells. Wrong births cannot be undone by this monotonic model.

## Next step

Keep this checkpoint as the connected baseline. Before another paid run, prepare one focused revision of training supervision for connected thick bulk and stopping growth on intact inputs. Retain the nine families and overall-building-volume semantics; do not prescribe rooms or fill exterior gaps. First use existing TRAIN cases to choose a differentiable bulk-aware objective and quantify its gradients and interaction with the existing frontier loss. Freeze the resulting specification, budget and comparison before requesting a single new run. Do not silently relax criteria, select a better horizon, tune on TEST, or launch a seed sweep. This recommendation is a design direction, not evidence that the revised loss will work.

## Preservation and resumption

Full local evidence: `C:/Users/artin/Documents/Codex/outputs/CGR1-Final-Review-2026-09-28`.
Reproducible review: `scripts/review_connected_run.py` (requires a fresh output directory). Small summary: `experiments/reports/CGR1-final-review.json`. Original supplied ZIP/receipt, review source snapshot and all 27 observation arrays retained. Milestone archive is adjacent to the evidence directory; its receipt records hashes. Same-disk copies are not off-device backup. No Drive operations, new training, push or live-model replacement performed.
