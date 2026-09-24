# MD1 - Bounded direct-massing pilot preparation

2026-09-24. Follows MO1/D063. Status: a concrete proposed pilot, not implemented or
executed. MO1's successful endpoint audit admits optimizer mechanics preparation;
it does not establish a good weighted objective or authorize paid training.

## Frozen candidate scope

Use exactly four saved MG1 alternatives from run20260924T113023Z_24298393f2a5:
aligned and partial_obstruction, each at24% requested volume, seeds0and1. Retain
the exact originals as no-update controls. Same scene/domain/MT1 thresholds and
voxel size; no new scene edits or resolution change. This is a development pilot,
not the full45-case comparison or a generalization study.

Candidate parameterization: one independently optimized logit per voxel, with
sigmoid(logit) times the fixed legal domain. Initialize probabilities to0.95 on
MG1 occupancy and0.05 elsewhere in the domain; outside remains0. Record logits,
continuous output and thresholded output at strictly p>0.5. This projection enforces
legality/ground/spill by construction; do not attribute those successes to learning.

Candidate loss: per-scene MO1 residuals, weight1 for eight families and4 for facade.
Within sparsity, add weight1 times ((sum(p)/domain_size -0.24)/0.24)^2 as a requested-
volume preference, while retaining MO1's lower/upper budget term. This explicit
preference is not part of the MT1 binary evaluator or MO1 endpoint-parity claim.
No extra regularizer, repair, threshold tuning or reconstruction target. These
coefficients are one declared hypothesis, not calibrated optimal weights.

Optimizer candidate: CPU float64 Adam, lr0.1, betas(0.9,0.999), eps1e-8,
weight_decay0, amsgrad/foreach/fused false; deterministic seed0, two Torch threads.
32updates/member, evaluate/checkpoint at0/8/16/24/32. Save all boundaries, not just
the most attractive or lowest-loss result. Do not extend steps or try weights
automatically if the candidate fails.

## Required mechanics before execution

1. Implement opt-in module/session and verify iteration0 binary identity to each
   saved MG1 field. Test actual weighted objective and scalar/gradient finiteness.
2. Save source, scene/domain hashes, recipe, requested-volume policy, logits,
   optimizer state, completed update and RNG state. A new process must match
   uninterrupted2+2updates exactly, including optimizer and evaluated geometry.
3. Profile four actual updates on each context. Candidate caps:30seconds/member
   for this profile,120seconds for recovery,120seconds/member for32updates and
  600seconds for the whole four-member pilot. Freeze effective caps and admission
   estimate after timing; no run admitted when projected cost exceeds its cap.
4. Check elapsed time inside the actual loop; retain partial outputs and mark
   timeout/interruption explicitly. Never count an interrupted run as a clean
   performance measurement. Manual continuation gets a linked run ID.

## Comparators and decision

No-update MG1 is mandatory. Prepare a separately versioned contact-aware procedural
control under the same masks/request; report its cost and all outputs. It should
account for the existing facade family in route/growth costs without changing the
generator's old behavior or silently tightening the physical domain. Freeze its
single recipe before observing its results. Do not claim direct optimization is
superior to procedural generation from comparison to contact-unaware MG1 alone.

Primary feasibility question: can any originally failing partial-obstruction
member pass all nine MT1 checks at the final32-update boundary while both originally
passing aligned controls stay valid? Report the count for each context separately.
Report requested-volume error, contact fraction and all family failures; loss
reduction or increased volume alone is not success. Report every intermediate
boundary separately, and do not substitute an early success for final stability.

A positive four-member result admits a larger frozen comparison, not NCA promotion.
If it fails, inspect recorded family tradeoffs once and make a bounded next decision;
do not reopen an indefinite sequence of local loss tweaks. Model training, learned
recovery and multiscale changes remain later questions under RESEARCH_BRIEF_R2.

User liked MG1 geometry. Preserve it and current viewer; future comparisons must
make original, procedural contact-aware and directly optimized outcomes distinct.
No paid Colab, Drive action, push or public deployment.
