# G3 design: budget-aware generation

2026-10-03. Design and local reference implementation only. No trained G3 model or paid-run package exists yet.

## Decision

G2 established growth but overshot requested volume. Introduce explicit whole-volume feedback during learning, with an intrinsic admission guard against irreversible overshoot. The local diagnostic shows that a guard alone is insufficient. Keep the same nine constraint families and building-volume semantics; interiors and rooms remain outside this phase.

This is a hybrid global/local NCA. Counting mass, broadcasting budget and ranking competing births are global operations. Do not describe it as strictly local NCA. The additional information comes from the user request, legal opportunity domain and current generated state, not a teacher volume or route.

## Budget contract and teacher compatibility

For domain size D and requested fraction r, desired count B=ceil(rD). At the present 32-cubed grid and0.8m voxels, physical bulk width w=3. Permit a narrow quantization band from B through C=min(B+w²−1,floor(0.40D)). For the current requests16/24/32%, C=B+8 on the audited TRAIN data.

The allowance has a procedural geometric basis: shifting a width-w cube origin by one face-adjacent cell can introduce at most one w-by-w plane. The final growth addition can therefore overshoot its stopping count by at most w²−1 cells. This reasoning applies to the incremental growth phase, not every possible initial route, generator or geometric representation. Each target must still be audited against the band. All27 existing TRAIN targets fall within it; no teacher labels were trimmed or changed. Their actual excesses above B range from0 to8 cells.

The reference accepts only the existing three pilot requests. It canonicalizes their float32 representations within1e-7 before counting. No arbitrary-request or larger-grid claim is made. Future physical scales require explicit versioned conversion and admission checks. If the initial state exceeds C, reject it with a recorded error; do not delete cells silently.

## Proposed network and training change

Append one scalar feature, (B−M)/D broadcast over the grid at each step, where M is the current occupied count. Recompute after each admitted step. Preserve the existing static request channel. Expand the first pointwise layer from60 to61 inputs, retaining hidden64 and output8. This adds64 weights. For a controlled fresh initialization, first create the G2-size fresh network with the fixed seed, copy those parameters into the expanded network, and initialize the new input weights to zero. Do not warm-start from trained G2 weights. This wiring is a specification, not yet implemented or verified.

Retain G2 frontier loss and local cube-volume term. Proposed additional term, initially coefficient1.0: distance of the one-step soft occupied count from [B,C], divided by D. The soft count is current hard mass plus the sum of sigmoid proposals over fired eligible frontier cells. Compute this term BEFORE hard admission so overshoot remains visible to the loss.

Within the band this term is zero; below the band it encourages additions; above it it discourages additions. It is a one-step surrogate, not an unbiased expected hard count or a differentiable final-volume guarantee. It can encourage geometrically incorrect additions below budget; teacher supervision must supply location information. The hard state and ranking remain detached, so distant credit assignment remains limited. Existing local cube supervision may still be insufficient for thickness.

Use the same planned TRAIN27, fresh seed1201, 50/50 seed/teacher-stage starts,64 steps,256 updates, optimizer and evaluation protocol when preparing a comparison. Adding feedback, band loss and the guard is an integrated architectural candidate, not a pure one-weight ablation like G2. Training-loss scale and recovery must be checked once the full integration exists, before freezing and requesting paid compute.

## Admission rule

The model proposes births only on legal, empty, fired cells adjacent by a face to the previously occupied field. From proposals above0.5, admit at most C−M cells, ranked by descending probability with flattened z/y/x index as a stable tie-break. Preserve all existing cells. At capacity admit none. Save proposed, admitted and budget-rejected birth counts separately at every step.

Every admitted cell touches the prior connected field, so this preserves connectivity from a connected initial seed; it does not guarantee that the field reaches all scene interfaces or that the bulk is connected. Index tie-breaking introduces an orientation bias and must be disclosed. The NumPy/CPU reference is a correctness specification; a GPU implementation and its timing/determinism have not been verified.

The count ceiling is guaranteed by this external rule, not learned. Stability at a filled ceiling is likewise not evidence of learned self-regulation. Score all nine families on the admitted field and retain unconstrained proposal diagnostics. A per-step pre-admission candidate is not a full no-guard rollout; label any later ablation accordingly.

An early wrong addition cannot be removed by the retained irreversible rule. Budget feedback may help avoid it, but cannot guarantee a feasible allocation. Do not conceal incomplete access, coverage or thickness with clipping or a revised threshold.

## Executed local checks

The reference passes deterministic ties, zero remaining capacity, overfull-start rejection,20 seeded randomized no-overflow/retention/eligibility checks, and under-band/inside-band/over-band output-gradient checks. All27 TRAIN teachers fit the proposed band. No optimizer updates occurred.

To test whether a guard alone would suffice, use the saved final G2 model on three preselected TRAIN cases: central-Y,24% request in aligned,wide-gap and partially obstructed families. Run raw and guarded64-step rollouts with firing2101 and zero hidden state. The guarded run applies the admission rule during growth; it is not post-hoc truncation. No new feedback feature or band loss is present in this diagnostic.

| TRAIN case | Desired count B | Ceiling C | Guarded final mass | Failed families |
|---|---:|---:|---:|---|
| Aligned |839|847|847|access,coverage,thickness|
| Wide gap |1046|1054|1054|access,coverage,thickness|
| Partial obstruction |819|827|827|access,coverage,facade,thickness|

Every step retained the original seed, stayed legal and respected C. All three guarded outputs fail overall validity despite reaching the budget. These are limited TRAIN diagnostics, not G3 quality results or a new development benchmark. The comparison supports training with the feedback, not deploying the cap on G2.

## Next implementation milestone

Wire the dynamic feature and band loss into a separately versioned model/session, implement and verify deterministic device admission, and combine the relevant gradient and checkpoint-recovery checks. Preserve raw proposal/admission evidence and compare G3 with G2 under the original nine development requests and gates. Do not automatically launch training or tune several coefficients. Reserved targets stay unopened.

## Preservation

budget_reference.py and check_budget.py retain exact reference code, protocol, teacher compatibility audit, per-step counts and all raw/guarded fields. G2 source and checkpoint remain in the preceding verified review. The local milestone archive includes code, results and resume instructions. It is a same-disk copy, not an off-device backup. Repository synchronization is pending; no repository commit, Drive operation, push, live-model replacement or paid training occurred. MG7 remains live.
