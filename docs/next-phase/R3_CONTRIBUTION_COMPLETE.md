# R3 contribution comparison completed — 2026-10-08

All 24 planned method/request slots are accounted for: 20 evaluated outputs and
four explicit no-certified-output records. Original 15 results are preserved;
nine missing slots come from continuation v2. Three extra parity rollouts are
verification only and are not counted as additional study observations.

| Method | Feasible requests accepted | Blocked requests | Mean seed Jaccard distance, feasible sites |
| --- | --- | --- | --- |
| Raw G10 | 4/6 | 2 rejected outputs | 0.23871 |
| R3 learned + planner | 6/6 | 2 no certified outputs | 0.20500 |
| Neutral scores + planner | 6/6 | 2 no certified outputs | 0.02925 |

Acceptance requires all nine families at both64/128, absolute requested-volume
error <=.04, late growth <=.05. Same frozen model, scenes, .24 request, .8m cells,
original route, seeds2102/2103 and CPU float32 settings as the original protocol.
Neutral replaces both learned eligibility filtering and ranking with a constant
positive score and zero hidden updates. It is neither MG7 nor a ranking-only ablation.

## Interpretation and decision

The planner/admission mechanism with neutral scores meets all feasibility gates
on this small sample. Learned selection is therefore not demonstrated necessary
for gate passing here. Learned selection changes the resulting form and increases
seed variation relative to this particular neutral baseline. Average seed Jaccard
distance is a voxel-overlap measure, not architectural quality or useful diversity.

The comparison plates show neutral outputs with seven-cell Z extent throughout;
R3 has ten cells in five requests and eleven in the sixth. Neutral forms appear
flatter and more extended horizontally. Bounding extent and one camera view do
not measure fullness, habitability, structural safety, or architectural superiority.

Retain R3 as the experimental hybrid, retain neutral as a scientific baseline,
and do not replace MG7 or start new training based on this comparison. Next:
freeze one consolidated set of previously unused site geometries within the
supported32/40 grids and opposing-facade representation; compare all three
methods without tuning on those cases. Record all failures, outputs, diversity,
runtime and peak process memory together. Once inspected, that set becomes exposed.
This validates generalization within the supported representation, not arbitrary
architectural sites. Keep the nine families and whole-volume interpretation.

## Memory continuation and verification

Original attempt stopped on CPU allocation failure and remains intact. The old
worker retained output variables across consecutive rollouts and captured unused
proposal/candidate histories. These are concrete retention opportunities; the
precise cause of the original allocation failure is not established.

Version2 runs one method/seed per fresh process and removes only those two unused
raw capture histories. Frozen scientific modules on disk are unchanged. A narrow,
asserted in-memory capture edit is saved as effective-rollout.py in every worker.
Case2 seed2102 G10, R3 and neutral reproduce every saved trajectory array, both
horizon fields, and all metrics exactly. Missing cases then completed without
allocation errors. Total continuation wall time 153.4s includes process
startup and three parity runs; it is not a speedup benchmark. No peak-memory
measurement was collected, so no numerical memory-reduction claim is made.

Original inputs and every contributing result manifest were hash verified.
Provenance for every observation is in the complete results JSON. Separate
controller, worker logs, input lineage, effective capture source, traces, trajectories,
certificates, metric outputs, and comparison plates are preserved.

## Files and resume

- Original immutable attempt: .local-artifacts/r3-contribution-2026-10-08
- Continuation: .local-artifacts/r3-contribution-continuation-v2-2026-10-08
- Summary: experiments/reports/R3-contribution-complete-2026-10-08.json
- Runner: scripts/continue_r3_contribution.py (refuses existing attempt)
- Analysis: scripts/analyze_r3_continuation.py
- Plates: scripts/render_r3_contribution_complete.py
- Archive: outputs/R3-Contribution-Complete-2026-10-08 under Documents/Codex

No training, Drive access, push, studio change or production replacement.
Same-disk archive is not an off-device backup.
