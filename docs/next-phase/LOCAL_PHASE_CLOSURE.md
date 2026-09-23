# Local investigation closed: evidence and next decision

2026-09-24 (Berlin; run IDs use UTC). F5 completes the four promised groups of local work: implementation
and regression, checkpoint recovery, bounded timing pilot, and matched training/
evaluation. All 199 automated checks pass. All experiment evidence is preserved.
This closes the local diagnostic/comparison phase, not the whole next-phase report.

## What we now know

- Reproducibility is established for these CPU runs: exact baseline replay,
  explicit configuration/source identities, individual fields, optimizer/RNG
  checkpoints, early and trained-state restarts, verified local archives.
- Historical metric and gradient problems were isolated rather than hidden in
  aggregate scores. F4's corrected raw access recovered some learning signal
  and increased final connections from 59 to 62/72, with 0 joint successes.
- F5 persistent guidance gives 66/72 connected final cases and 0/72
  joint connectivity/budget successes. Its guide weights learn, but the outcome
  must be judged against the material budget and all other recorded families.
- W1 achieved connected/in-budget geometry on 17/17 feasible development scenes;
  D1 direct optimization reached 12/17 with the stronger material recipe.
  These are useful controls, not equal-compute shared-rule learners. They already
  show that solving the selected proxies need not require an NCA.
- Current access is connected material, thickness discourages bulk, and support
  is geometric boundary attachment. They do not yet specify walkable floors,
  useful architectural spaces or mechanical safety. Zero proxy loss can reward
  one-voxel routes. More voxels or a prettier render would not resolve that meaning.

## Decision

Persistent guidance did not produce a final design meeting connectivity and the
material budget together. Close this local sequence of incremental loss and
conditioning experiments. Do not launch another loss tweak, longer run or larger
grid by default. Move to a representation and NCA-role review before new learning.

Write the next design specification around a planner-provided valid scaffold
and an NCA with a narrower refinement/recovery role. Compare that hybrid against
the already strong procedural and direct-optimization controls. Retain the architectural material/form-generation scope already accepted in
D026. Specify what learned refinement would add beyond those controls before
changing representation or starting another training experiment. This is a proposed
direction, not an implemented or proven replacement.

## Next phase boundaries

1. Retain D026's accepted architectural material/form-generation scope and
   define the specific benefit expected from an NCA within that scope. A planner + learned refinement/recovery role is the recommended
   candidate for the design discussion; it is not adopted training code.
2. Design the Studio around honest comparisons: scene editing, saved alternatives,
   per-family diagnostics, visible failure states, cancellable jobs and clear
   distinction between procedural, optimized and learned outputs. Use the user's
   local visual concept as a starting reference. Deployment redesign is outstanding.
3. Before any new training, freeze a new protocol with an explicit success target
   beyond connection alone, fresh validation scenes/seeds, compute allowance and
   recovery/backup requirements. Preserve the existing nine-family boundary;
   any revised semantics must be documented and compared under versioned scoring.
4. Colab is not running and no action from the user is needed to close this phase.
   GPU training and every Drive operation require their separate stated approval.

This is a stopping point for the current local experiment series, not an
indefinite queue of additional local tests. Scope the next phase explicitly before
new learning. M4/M5 deployment, scale and generalization remain unfinished.

## Where to resume

Read PERSISTENT_GUIDE_FINDINGS.md, D053 and RESUME.md. Full F5 `20260923T214035Z_566da8c507c0`;
trained restart `20260923T220548Z_fdd27bb79c92`; source `27416f30141d6648d798380dfd7fcb191889f69f`.
Use experiments/reports/F5-* for results and .local-artifacts/runs for raw evidence.
The local results commit and full archive preserve the history. The archive's
receipt includes a fresh restore and per-file hashes; it remains a same-disk copy,
not an off-device backup. Private NCA-Next-Phase-Report files remain Git-ignored.
