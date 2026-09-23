# Next experiment: access-only NCA comparison

Prepared from A2, 2026-09-23. This is preparation, not an executed study or
production promotion. Read ACCESS_AUDIT_FINDINGS.md and D039 for measured support.

## Scientific question

Does the component-based access contract improve joint connectivity/material
budget outcomes, given the same architecture, scenes, coefficients and exposure
as F1? A2 changes measurements on frozen fields; it does not answer this question.

Use original Model C initialization, both existing recipes, the same two scenes,
training seed0,16-step growth, weak0.15 scaffold and proposed64 updates per model.
Keep the nine-family objective, replacing only the access family with the fully
specified component_bottleneck_v2 definition. Keep all other eight terms, three
regularizers, optimizer settings and scene exposure unchanged. This one family
change includes region/component semantics, worst destination and unbounded
spatial reach; A2 attribution must remain visible rather than calling it a pure
source-cell change on arbitrary multi-entrance scenes.

Evaluate every F1 boundary (0,1,3,8,16,32,64),16/50 growth steps and firing seed2.
Save BOTH old and candidate access/binary results, all other terms, both common
recipe totals, continuous mass, per-scene failures and computation time. Never
claim improvement just because rescoring lowers a number. Connectivity AND budget
must improve on the same evaluated field; neither alone promotes a model.

## Gates before execution

1. Implement an opt-in objective version and actual resumable loop. Keep original
   F1 code and results immutable. Do not load old training checkpoints through
   mismatched metadata or quietly substitute the candidate into legacy functions.
2. Verify baseline parity using saved original fields and short old-objective
   update traces for all four members. Existing F1 can serve as the old-objective
   control only when this exact parity is demonstrated; otherwise retain the
   discrepancy and resolve it before comparison.
3. Verify restart in fresh processes for the actual candidate objective, including
   topology/tie selection, optimizer, scheduler, firing RNG and intermediate
   evaluations. Ordinary CPU update recovery does not certify abrupt writes/GPU.
4. Profile a small fixed pilot that includes evidence and evaluation costs. Freeze
   exact compute caps and admission before a full comparison; use timing, not
   pilot quality, to admit it. The64-update matrix is a proposal until that gate.
5. Preserve all attempts and source snapshots; new run IDs for failures/retries.

## Interpretation and later work

The original checkpoints still have zero access gradients under both definitions
in A2. Existing pre-clamp coverage provides a measured nonzero signal to start
growth; do not assume the new exact connectivity measure solves disconnected
initialization by itself. On partially connected fitted fields the candidate may
provide an additional signal. Check actual optimization and budget tradeoffs.

If it helps, test stability across firing/training seeds and fresh geometry before
scaling. If it does not, inspect recorded gradient conflict and horizon dependence
before isolating schedule/pool or conditioning changes. Do not combine those with
this access comparison. Candidate CPU union-find is not GPU-ready; Colab training
still requires a profiled implementation, certified recovery and an approved cap.
No paid compute, Drive operation or deployment is authorized by this plan.
