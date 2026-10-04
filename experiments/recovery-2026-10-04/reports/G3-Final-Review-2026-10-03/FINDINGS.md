# G3: volume control improves; thickness is not preserved

2026-10-03. Run `20261003T201425Z_f94f89d7c7ab`. Training completed; frozen generation acceptance failed. No model promotion or new paid run.

## Verified execution

All1,035 exported payload hashes, unique archive membership and receipt verify. The request and final checkpoint match the exact G3 package. All256 updates completed in151.307 controlled seconds (144.619 worker seconds), with1,451,229,184 bytes peak reserved GPU memory (1,384MiB). The eight GPU admission/reference cases passed, and both checkpoint replays report exact full-payload/state equality. These exported checks are verified evidence; GPU execution was not repeated locally.

All256 start selections, final trace, optimizer steps and sampler cursor agree. Independently checked all16,384 recorded training steps for budget-count arithmetic and temporal continuity. All256 archived training states retain their starting cells and stay within the legal domain. Across training,992,823 proposal events were rejected by the budget rule; repeated offers of the same cell count again.

G2/G3 use identical initial base parameter tensors, TRAIN array identities, row/stage order, final firing RNG and runtime versions. G3's extra channel starts at zero. This is a combined feedback/loss/admission architectural comparison, not evidence isolating those three components individually.

## Frozen review

Final256 checkpoint only; nine existing development requests; CPUfloat32; firing2101; single-cell starts;64 primary steps and128 fixed stability steps. The actual61-input BudgetNCA and saved evaluator source are used. No threshold/horizon/checkpoint selection or postprocessing. Reserved targets remain unopened. Reused development and one training seed limit generalization claims.

| Measure | G2,64 steps | G3,64 steps | G3,128 steps |
|---|---:|---:|---:|
| All nine families pass |0/9|1/9|0/9|
| Access |9/9|5/9|5/9|
| Coverage |9/9|7/9|9/9|
| Thickness |1/9|1/9|0/9|
| Sparsity / allowed volume range |0/9|9/9|9/9|
| Facade,ground,legality,spill,support |9/9 each|9/9 each|9/9 each|
| Median absolute request error |25.28pp|4.45pp|0.17pp|
| Maximum absolute request error |31.04pp|8.26pp|4.92pp|
| Median teacher IoU, diagnostic |0.4871|0.5354|0.5298|

Requests are16%,24%,32%. G3 occupies11.55–30.10% of the domain at64steps and16.15–32.16% at128steps. The primary median-error gate is2pp and maximum-error gate4pp; both fail at64steps. Volume grows6.84–39.84% between horizons, exceeding the5% stability limit for every case. All five frozen gates fail overall. Do not replace the primary endpoint with128steps after seeing its smaller median error.

## Budget control versus learned behavior

No development rollout hits its ceiling or has a budget-rejected birth during the first64steps. Thus the improved primary volume range is not produced by directly trimming those nine trajectories at inference. It reflects the trained G3 system as a whole; the separate effects of feedback, band loss and guarded training are not identified.

By128steps, five cases hit the ceiling and248 proposal events have been rejected across the nine rollouts. Those later limits are enforced by construction. Per-step pre-admission candidates are saved and scored separately; they give1/9 validity at the64-step endpoint and0/9 at128, like the admitted fields. They are not a full no-guard ablation.

## Why the one passing shape later fails

The `g1-offset_interfaces-y4-v32` case passes all nine geometry families at64steps, with1,622 cells and93.83% cube-supported mass. By128steps it has1,733 cells but only88.06% cube-supported mass, below the90% requirement. Its volume is closer to the request, yet its geometry no longer passes thickness. This is concrete evidence that adding cells can degrade the thickness fraction.

Across the full set, cube-supported fraction spans71.72–93.83% at64steps and60.62–88.06% at128steps. Thickness fails8/9 and then9/9. Access also remains incomplete in4/9; addressing thickness alone does not guarantee that all interfaces will connect within budget.

## Decision and next step

Keep G3 as the first seed-generation baseline with one valid primary output and substantially improved volume matching. It remains unaccepted. MG7 stays live; no automatic GPU retry, longer rollout selection or threshold relaxation.

Next design work should examine bulk-preserving growth units: proposals that add overlapping legal3-by-3-by-3 regions, accounting for their actual new occupied cells before budget admission. This aligns the growth representation with the existing2.4m bulk definition instead of relying only on individual-voxel births to learn it. The first accepted region must incorporate the seed; connectivity, overlap, boundary legality, ties and finite budget handling need explicit rules. Whole-cube unions can preserve the current cube-support definition by construction, but their ability to reach interfaces and spend the requested budget still requires evidence.

This is a proposed architectural change, not a trained result or a guarantee. Design and audit it against the existing valid teachers and relevant counterexamples before preparing another paid comparison. Preserve the same nine constraint families and physical interpretation. Do not relabel geometric support as structural safety or cube support as architectural quality.

## Preservation

The original full ZIP/receipt, final checkpoint, update records,18 scored fields/states, admission traces, final pre-admission candidates, exact model/evaluator source, review script and G2 comparison are retained here. The original ZIP keeps every training checkpoint and state. `pairing-budget-verification.json` records the additional trace/state audit. `result.json` and `comparison.json` contain all per-case outcomes.

The milestone archive is hash-verified locally and remains a same-disk copy, not an off-device backup. Repository synchronization remains pending because write access was not granted. No repository edit or commit, Drive operation, push, public deployment or reserved evaluation occurred. Resume from this folder rather than the stale repository D098 record.
