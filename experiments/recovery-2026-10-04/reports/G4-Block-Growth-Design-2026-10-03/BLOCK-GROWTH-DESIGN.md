# G4 design: grow overlapping volumes, not isolated voxel births

2026-10-03. CPU reference and TRAIN feasibility audit complete. No learned G4 model or paid-run package exists yet.

## Why change the growth unit

G3 improved volume matching and produced one valid64-step output, but that output failed thickness after further growth. Adding isolated cells can lower the fraction belonging to complete3-by-3-by-3 cubes. G4 proposes whole legal cubes as the atomic growth unit, aligned with the existing2.4m bulk definition at0.8m spacing. Keep the same nine families and building-volume semantics.

After its first successful addition, the field is a union of complete legal cubes. The current evaluator's cube-support fraction is then100% by construction. This is an enforced geometric property, not a learned achievement or a guarantee of useful architecture. Access, coverage, facade contact, support and budget feasibility must still be evaluated. The unchanged single seed is not itself a valid building volume; failure to place the first cube must remain visible.

## Reference transition

Candidate indices represent cube origins on a30-cubed grid for32-cubed occupancy and width3. Require every cell of the proposed cube to be inside the legal domain; discard no cells by clipping at boundaries or obstacles.

For a single-cell seed, candidates must contain that seed. Among fired proposals above0.5, admit at most one first cube, chosen by descending score with ascending flat z/y/x index ties. The cube must fit the current ceiling. This places27 total cells, retaining the seed, without reading teacher geometry.

After initialization, the current field must already be a complete cube union. Find all cube origins whose cubes are fully occupied. Eligible new origins are legal, not already fully occupied, and six-face-adjacent to at least one such origin. Moving one origin cell overlaps a complete3-cube in an18-cell slab, so each subsequent proposal can introduce at most9 cells. All offered origins are determined from the state at the beginning of the step; admissions do not unlock a new propagation layer until the next step.

Sort fired above-threshold proposals once. Process that order deterministically, recomputing each cube's actual new cells against the field including previous admissions from the same step. Admit the complete cube only if its new cells fit; otherwise retain a budget-rejection record and continue to smaller fitting candidates. Skip already-completed redundant cubes. Never trim a cube to fit.

Use G3's existing desired count B=ceil(rD) and ceiling C=min(B+8,floor(0.40D)) for the current three requests. Starts above C are errors. A remainder smaller than every available increment can cause a stall with unused capacity; do not hide it or raise C. The eight-cell allowance is consistent with the nine-cell maximum increment after initialization, but cannot guarantee a feasible continuation under every obstacle or ordering.

The admission rule preserves the seed, legality, count ceiling, raw connectedness and full cube support when it begins from a connected seed/cube union. It does not repair a disconnected initial field; training and inference adapters must enforce their starting-state invariant. Global ordering and budget accounting remain hybrid operations, not strictly local NCA.

## Local checks and TRAIN representability

The independent checks use the existing `cube_supported` evaluator and flood-fill connectivity, not only the new implementation's own summaries. They check every accepted target-guided step and every proposed teacher stage for retention, legality, connectedness and thickness.

Synthetic cases cover:

- First cube contains the seed and respects capacity.
- No legal cube around the seed preserves the seed and records no growth.
- A ceiling below27 cannot place a complete initial cube.
- Overfull and nonbulk initial states are rejected.
- Overlapping additions count only newly occupied cells: a36-cell initial field plus two proposals adds9 then3 cells, finishing at48 rather than the naive54.
- A smaller remainder stalls without adding a partial cube.

All27 existing TRAIN teacher volumes can be represented exactly by legal3-cube unions connected through the cube-origin graph. No labels or constraints were changed. With target-derived perfect proposal scores, fixed NumPy firing seed2101 and the same ceiling, all27 are reconstructed exactly in22–33 steps, within the proposed64-step horizon. There are no missing teacher voxels at completion. This is a target-guided feasibility bound on these examples, not trained generation performance or a neural training-time estimate.

No development or reserved targets were opened for this audit. Targets, origin masks, stage distances, exact construction histories and final fields are retained for every TRAIN example. The audit took about23.9 seconds locally, excluding earlier synthetic setup; this is not a GPU benchmark.

## Training adapter changes required

The old voxel-distance teacher stages are not generally full cube unions. They must not be silently reused under the new invariant. This audit constructs a separate TRAIN-only curriculum: select the lexicographically first full teacher cube containing the independently selected seed, compute face-neighbor distances among full teacher cube origins, and form each stage as the union of cubes through a chosen distance. All these stages preserve the seed and bulk; the maximum origin distance observed is25. The root and stages use teacher information only in TRAIN supervision. Inference remains a single scene-defined seed and never receives a teacher root, graph or route.

Retain an explicit50/50 mixture of single-seed and cube-stage starts when defining the next session. Freeze the stage sampling and start hashes before paid training; G3 voxel-stage identities are incompatible. Preserve all historical data and generate a separately versioned derived dataset rather than rewriting it.

The proposed neural interface retains a voxel-grid hidden state and G3's static context plus dynamic remaining-budget feature. Interpret the occupancy output at each cube's center as that origin's proposal logit (crop one cell on each side for width3). Supervise origin labels using full teacher cubes. Firing and proposal arrays now use the30-cubed origin grid; record this RNG/state semantic change. The hidden-update firing map must be defined consistently with origin centers during implementation.

Multiple overlapping cube probabilities cannot be summed as independent new voxel counts. For ordinary growth steps, use a differentiable union over eligible cube proposals to derive a soft voxel field, then apply local cube-volume and global budget terms before hard selection. This is a surrogate; budget-coupled admission remains nondifferentiable.

The seed step admits exactly one cube, so the ordinary multi-cube union surrogate is not faithful to that decision. Specify the initialization loss separately before freezing the training protocol—for example, supervise seed-containing cube choice directly without claiming a calibrated soft multi-cube volume estimate. This remains an implementation decision; no paid protocol is frozen in this milestone.

## Next milestone and limits

Implement the cube-proposal model, the derived full-cube stage adapter, overlap-aware soft-volume loss and device admission. Verify gradients, input/label separation, state identity and exact recovery together. Measure device admission cost before approving a training cap; the sequential CPU reference does not establish GPU efficiency. Keep raw proposal, rejection, redundant-proposal and unused-budget records.

Evaluate any trained candidate under the unchanged nine-family protocol. Report thickness and budget compliance as enforced, and learned access/coverage/contact outcomes separately. Cube growth alone can fill the wrong region, make bulky but unhelpful shapes, miss interfaces or stall. The feasibility audit removes one representational concern on TRAIN; it does not establish generalization or guarantee successful learning.

## Preservation

The executable reference, audit, protocol,27 teacher constructions and all failures/stalls from synthetic checks are documented here. Source evaluator dependencies remain in the preceding saved G3 review; their fingerprints are preserved in this milestone. No trained G4 inference, optimizer update, paid compute, Drive action, repository edit/commit, push or deployment occurred. MG7 remains live. Repository synchronization is pending. The verified local archive is a same-disk copy, not an off-device backup.
