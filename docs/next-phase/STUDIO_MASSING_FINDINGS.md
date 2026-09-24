# MS1: live building-mass Studio

2026-09-24. Implemented the product phase admitted by MG3, D067.
Open http://127.0.0.1:8001/static/live/index.html. The earlier scaffold workspace
remains at / and links to the new page. This is live procedural generation,
not a saved-results simulation and not a newly trained NCA.

## Delivered behavior

Choose one of five frozen development sites,16/24/32% of its fixed region, and
an integer seed0–2147483647. Seeds0–2 have MG3 evidence; other seeds are exploratory.
Scale stays32 cubed at0.8m/cell, with2.4m blocks and two supported interfaces.
The generator and MT1 evaluator are unchanged. Occupied cells represent overall
building volume; interiors and construction are deferred. No new constraint
family, model architecture, loss or threshold was introduced.

Generate queues a real local worker, evaluates all nine families independently,
and automatically saves the result. A completed job can have failed checks.
Cancelled/interrupted/failed execution attempts retain their history and offer
a linked retry with the same site, requested fraction and seed. The new page
provides slices, cutaway, context, a saved-result library, comparison and export.
Records remain available after reload; reopening selects the saved parameters
and evidence for display, while the controls specify the next request.

New mass requests/results use studio_mass_v1, kind mass_result, with separate
.local-artifacts/studio-mass and studio-mass-jobs stores. The shared durable
manager and worker dispatch by record kind. Original studio/studio-jobs remain
separate. Request identity is verified at worker completion and restart recovery,
including source provenance, kind, version and mass parameters. A changed source
file requires a server restart before submitting another job.

Each job preserves request, context/domain identity, generator/evaluator spec,
source ZIP, progress, chained history, worker log and hashed candidate. Results
retain complete raw occupied/route cells, independent metrics and growth trace.
The five fixed contexts are runtime source in deploy/mass_contexts.json, copied
from MG3 and tested against the original context builder. No archived run is
needed merely to launch the page. The startup snapshot now includes the live UI.

## Portable evidence

studio_mass_portable_v1 exports include geometry, all growth decisions and actual
retained source bytes. Imports verify transport hash and source manifest, validate
the typed request, then regenerate and compare all geometry, metrics, input
identity and growth metadata except observed wall time. Existing evidence is
never overwritten; duplicates are recognized. Source origin is not authenticated
by a checksum, and incoming source is retained as data, never executed/extracted.
Material-scaffold exports are rejected by the mass import and vice versa.
Import uses a20MB request limit and a single-import lock. Only currently replayable
results are admitted; a machine-dependent timeout result may not replay exactly.

## Verification

Regression 20260924T144654Z_f7338d28b9ac:320 tests pass,0 failures/errors/skips,
smoke exit0. The ten new integration tests cover real worker/MG3 parity, fixed
inputs, failed scientific outputs, input/type bounds, active cancellation and
parameter-preserving retry, matching/mismatching crash-recovery results, forged
geometry/trace/source rejection, foreign import/re-export/deduplication and exact
comparison differences. The earlier material workflow regression still passes.

Focused test log: outputs/ms1-focused-1.log in Codex cwd,10pass in681.656s.
The host/tool wait was unusually long; no failed test or rerun occurred. Full
suite168.536s; command including smoke174.5641s. Browser generation overlapped
the tail of the suite. These are verification durations, not speed benchmarks.

Independent acceptance 20260924T145347Z_03b4d20a69f7 verifies all three browser-generated
records against MG3, including complete generation metadata (excluding wall time),
route, occupied cells and MT1 scores. All three portable exports and job histories
verify. All152 Python files and four new runtime data/UI files match the full
regression source snapshot. Current source hashes also match all live job archives.
See experiments/reports/MS1-verification.json and both tracked run records.

| Live record | Case | Result |
|---|---|---|
|20260924T144914Z_800512d53ad5|Partial24%,seed2|819 cells; all nine pass|
|20260924T144941Z_e60cce65a38e|Partial32%,seed2|1092 cells; all nine pass|
|20260924T145002Z_6fbc4db10ce8|Blocked32%,seed2|0 cells; no cube route; failed checks retained|

The two partial results compare as273 added cells and0 removed. Browser tests
verified generation, scientific failure reporting,11 comparison rows, library
reload with3 records/jobs, reopening, vertical index16/13.2m, horizontal/axonometric
views, cutaway and export-button preparation. Desktop1100px has1085px document
width; mobile390px has375px document width, neither overflows. No browser warning
or error logs. Native OS file selection for import was not automated; full import
replay/roundtrip behavior was tested through the API. Evidence/captures live in
the acceptance run and outputs/ms1-qa. Temporary viewport was reset.

## Operational findings and boundaries

The four old jobs were terminal and integrity-clean before server restart.
Windows Stop-Process failed with a null-reference error. The first new server
then correctly refused the occupied store lock and exited, preserving ownership.
After rechecking process identity, the old idle PID20784 was terminated using
the runtime; the new server started successfully as PID29824 (session23049).
No jobs or receipts were deleted to resolve the lock. Server state must be
rechecked on resume; PIDs can be reused. Exploratory reads had a nonnumeric
TotalCount typo and a Windows glob mismatch; corrected without file changes.
An unprivileged process query was denied; the scoped approved query succeeded.

The page is local and experimental. Five fixed scenes do not establish unseen-
site reliability. Free site editing and larger grids are not supported in MS1;
they need explicit context/distribution and resource validation. Job history is
single-owner local persistence, not multi-user hosting. The import picker and
all potential seed/site combinations are not claimed as exhaustively tested.
No paid Colab, training, Drive operation, public hosting or push occurred.
Private reports remain ignored and unchanged. Local archive is same-disk only.

## Next phase

The user can now review real generated alternatives directly. The next research
step should freeze a fresh set of site variations beyond these five development
contexts, then profile larger grids at declared physical scales. Keep all nine
families and preserve failures, request fidelity, diversity and CPU/memory costs.
Separate unseen-context generalization from resolution scaling. Define numerical
sample counts/caps and frozen inputs before execution; do not infer them here.
Only after these findings identify a concrete gap should a learned NCA pilot be
specified. Paid training still requires an explicit compute allowance/recovery
plan, and any Drive action requires separate permission in the project folder.
