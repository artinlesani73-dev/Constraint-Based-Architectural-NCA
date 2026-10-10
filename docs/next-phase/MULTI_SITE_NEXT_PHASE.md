# Next research phase: context-conditioned growth across multiple sites

User agreement2026-10-10: finish the current local private studio first; the next phase of the whole project is context-conditioned growth of architectural mass across multiple sites. Initial milestone: one small urban block with several existing buildings and multiple opportunity regions. This records direction and sequencing, not authorization to start paid training or cloud transfers.

## Reference and distinction

Read Zhang et al., Outdoor Scene Extrapolation with Hierarchical Generative Cellular Automata, CVPR2024, arXiv2406.08292v1: https://arxiv.org/html/2406.08292v1 . Sections3.1-3.3,4.3,appendicesA.5/C were inspected.

The paper combines sparse local growth, learned bird's-eye-view context and separate geometry refinement. Its training uses paired completion targets; its learned planner is not our procedural route/witness mechanism. Planner ablations show a consistency/diversity tradeoff. Its surface-completion objective and resource measurements do not establish our mass-generation capability. These are adaptation ideas, not a planned transplant of its weights or training recipe.

## Proposed adaptations: decisions to evaluate next phase

1. Define a multi-site scene contract before increasing grid dimensions: rectangular domain/world units, many existing masses, several declared connection regions and separate opportunity regions. Replace the special two-opposing-X-facade representation. Keep current scenes readable and old results unchanged.
2. Start with coarse mass allocation and contextual guidance. Compare explicit contextual fields against a learned overview encoder rather than assuming a learned planner is necessary. Preserve exact procedural/learned attribution and a neutral-score control.
3. Evaluate sparse/active-region computation and boundary communication. Our filled-volume semantics can make active occupancy dense; measure occupancy/frontier fractions and actual memory before choosing sparse kernels. Chunking must exchange state and preserve global budgets; independent cropped generation is not sufficient.
4. Preserve a high-level volumetric representation first. Surface refinement is optional later and must retain its parent voxel mass, units, volume and constraint checks. Smoothing is not better architecture by itself.
5. Resolve training supervision explicitly. Architectural interventions have many acceptable outcomes; a single reference mass should not be treated as the unique truth. Define a frozen diverse synthetic block set and identify what each target teaches. Compare baselines before choosing teacher-guided transitions, longer unrolls or a changed architecture.
6. Evaluate whole-block and per-region results within the same nine constraint families. Inspect useful empty space, resource allocation and interface coordination without silently adding new constraint categories. Measure form variation separately from validity and human review. Include blocked/incompatible requests and record refusals.

## Initial milestone and success questions

One authored small urban block, existing buildings fixed, several opportunity regions and multiple connection choices; no full-neighborhood street generator, room planning or construction claims. Exact extent, voxel size, channels, hierarchy and compute budget remain open until profiling and protocol design. Test whether coherent distinct masses can be generated without consuming meaningful exterior gaps, whether shared budgets/interfaces remain consistent, and whether learning improves useful alternatives beyond the procedural baseline. This milestone is not yet implemented.

## Sequence

Close and preserve the current local studio package with full research-data snapshot, startup/upgrade/rollback instructions and consolidated verification. Then begin a separately versioned block-scale representation/protocol phase. Keep local-only studio, nine families, whole-building-volume semantics, MG7 production and currentR3 evidence intact. No automatic training, Drive operation, push or public deployment.