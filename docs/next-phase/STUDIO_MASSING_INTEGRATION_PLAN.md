# Next product phase: interactive mass generation

2026-09-24. Planned after MG3 meets the frozen36/36 nonblocked development gate.
Not implemented yet. Read D058, D066 and CONTACT_BUDGET_FINDINGS first.

## Product contract

Add an explicitly labeled experimental building-mass mode using the admitted
budgeted_contact_growth_v1. Occupied cells represent overall building volume;
interiors and construction remain later work. Keep historical material-scaffold
records and the old live workflow interpretable and unchanged. Show all nine
MT1 results, requested/actual volume, contact ratio, bulk fraction and actual
termination reason. A completed job can produce a scientifically failed result.

Start with the five frozen contexts, requests16/24/32% and seeded alternatives,
32-cubed grid at0.8m/cell,2.4m cubes and the two supported facade interfaces.
Display the fixed region and unsupported inputs clearly. These development
presets have evidence; edited/unseen sites do not inherit their pass rate.

## Implementation order

1. Inspect existing Studio durable jobs, storage/import schemas and scene request
   paths. Design a versioned mass request/result type; prevent accidental reuse
   of old material metrics or defaults. Bind scene/domain, seed, volume request,
   generator/evaluator versions and source identity to each saved record.
2. Add bounded backend generation and independent MT1 evaluation. Reuse durable
   job submission, cancellation, restart and parameter-preserving retry where
   compatible. Preserve complete route/proposal evidence and failed outputs.
   Never silently substitute another seed or fall back to the old generator.
3. Add the experimental mode to the Studio with clear controls, progress, saved
   alternatives, comparisons and failure states. Use the existing volumetric
   viewer and readable nine-family results. Avoid promising architectural quality
   from the pilot checks. Keep archived comparison galleries available.
4. Add versioned local import/export with hashes, raw evidence and replay. Cross-
   type imports must be explicit, with no material/mass semantic conversion.
5. Test real outputs against MG3 benchmark members, rejected/unsupported inputs,
   durable job cancel/restart/retry, export/import integrity and old-workflow
   compatibility. Run relevant regression once after implementation is coherent.
6. Verify actual browser generation, save/compare/reload and responsive views.
   Restart the local server only after checking no active jobs; preserve records.
   Document outcomes, limitations, exact next actions and verified local archive.

## Admission and boundaries

Do not present the current saved-data gallery as a live workflow. The next mode
is admitted for implementation, not already delivered. Keep greedy stalls and
route-budget failures visible. Do not add constraint families, threshold changes,
new training or paid Colab runs. A learned approach needs a concrete measurable
benefit and its own binary-output admission gate after the MD1 negative finding.
Local development requires no Drive access. Every Drive operation still requires
the user's separate explicit approval within the sole project folder.
