# ED1 environment diversity

## ED1 environment diversity completed - 2026-09-28

Added four designed48-cubed development sites: different connection heights,
staggered obstacles, overhead crossing, asymmetric frontages. Same0.8m cells,
MG7 generator,2.4m growth blocks, MT1 nine-family evaluation and fixed-domain
24% request. Seeds6/7;45s generation cap per candidate; CPU2threads. No new
constraint, loss, model or training. These cases extend environment variety at
one existing scale; no independent generalization or architectural-quality claim.

Run20260928T092021Z_cd4a5b070033 completed8/8 candidates in13.237s including
source capture/context/serialization. All8 reached target and passed all nine
pilot checks. Generation plus evaluation ranged0.960-1.926s. Across seeds the
four sites differ by4544,1506,468,1944cells respectively; occupancy IoU
.437693,.615914,.798796,.661265. Equal volume does not imply identical geometry.

Evidence: .local-artifacts/runs/20260928T092021Z_cd4a5b070033 contains complete
source snapshot, recipe/provenance, four audited scenes and context arrays,
eight raw fields/routes/diagnostics, growth traces, checks and timestamps.
Every registered artifact hash verified. Each raw field checksum checked and
all eight scores independently recomputed from saved arrays, with exact match.
Run record under experiments/records; small summaries ED1-diversity.json,
ED1-seed-differences.json and ED1-studio-check.json under experiments/reports.

Appended four presets to deploy/mass_v2_contexts.json; all11 existing context
objects compare equal, preserving their individual identities and replay inputs.
Catalogue now15 sites /65 evaluated setting combinations. Added only seeds6/7
and request24% for new sites. Existing unsuccessful controls remain selectable.
Extended source provenance to include live-v3 files for new Studio jobs.
Focused preset/bounds/mask/API-origin test passed; existing test expected counts
updated. No generator/evaluator tests rerun because their code is unchanged.

Browser:48 filter lists7 sites. New-site generation saved record
20260928T092244Z_eee2c4098f12 (staggered,seed6); exact field hash and all target
metrics match the study. New record includes live-v3 provenance. No console
errors. Initial mouse activation did not enqueue a job; inspected state, then
keyboard activation completed one job. Screenshot in Codex outputs/
ED1-Diversity-2026-09-28/studio.png. Historical stored records untouched.
Server restarted after checking no active jobs: PID6172/session31618,
python -m uvicorn deploy.studio:app --host127.0.0.1 --port8001 --no-access-log.
Process metadata query was denied; stopped the known owned server session instead.

Local archive destination: Codex outputs/ED1-Diversity-2026-09-28.
No Drive read/write, paid compute, remote push or publishing. Same disk only.
Next batch: unify the Studio entry page and navigation so generation, model
research and evidence are easy to distinguish and find. Avoid further small
training/seed sweeps; learned path remains experimental under D084.
