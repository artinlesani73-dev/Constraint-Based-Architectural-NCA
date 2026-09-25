# MS2: versioned mass Studio with evaluated larger sites

2026-09-25. Integration complete. The new Studio uses MG7 incremental generation
and unchanged MT1 evaluation. All 57 selectable combinations exactly match the
MG7 matrix: 44 valid volumes and 13 blocked controls. Full regression: 351 tests
pass, no failures/errors/skips, real checkpoint smoke exit 0, 225.074 seconds.
No NCA training, constraint changes, finer resolution or public hosting occurred.

Open /static/live-v2/index.html. /static/live/index.html remains the original
MS1/MG3 workflow. The new library reads old records without moving or rewriting
them and labels their version. New jobs use a separate store and explicit generator
dispatch. Imports replay a recognized local version; source bundles remain data.
Original generator/evaluator modules, notebook and checkpoint are unchanged.

## Scope and implementation

Eleven presets expose only evaluated combinations: five at 32 cubed with seeds
0/1/2 and 16/24/32% requests; three each at 48 and 64 cubed with seeds 6/7 and 24%.
Voxel size remains 0.8 m, cube width 2.4 m. World spans are 25.6/38.4/51.2 m.
Lossless context arrays are copied from MG6 and checked against MG7 input masks.
The viewport, cutaway, slice range and physical labels adapt to the selected
geometry. Cross-site comparisons omit meaningless cell-difference counts.

MassV2Jobs reuses the existing durable queue, owned worker tree, cancellation,
parent watchdog and recovery. It adds a 180-second worker wall deadline, checked
at queue ticks while the server runs; this is not hard real-time under suspension.
Four pending jobs are allowed. Generator cooperative limits remain 15/45/120 s.
20 MB imports retain their exact text, validate known formats/source hashes and
bounded unique coordinates, then replay in the same cancellable queue. Repeated
imports create separate attempts. Stop/retry and failures retain history.

## Failures found and fixed

Focused attempt 20260925T090307Z_9e687a560fca: seven tests, one failure. A first-ever import into an
empty store could not create its result directory. Creating missing parent
directories fixed it. The log and failing source snapshot remain retained; that
first attempt used temporary fixture directories. Subsequent MS2 test fixtures
are retained under .local-artifacts/testing/MS2. Focused retry 20260925T090432Z_53e2d71911c5 passed
all eight tests, including real worker parent-death recovery.

Browser attempt 20260925T091257Z_bce5f23dd4ef generated three correct volumes but rejected import.
JavaScript parse/stringify converts integral floats such as 0.0 to 0, invalidating
the Python record checksum. The original API export verifies; its saved browser-
normalized counterpart does not. The correction transports raw export/import
text and keeps strict checksum validation. It also clears stale success status
when choosing a new preview. No backend or numerical source changed afterward.
The original UI remains historical; exports reformatted by that UI may need to be
re-exported from the retained record through version 2. Do not bypass the checksum.

Corrected browser run 20260925T092014Z_06e493b0f94b passes four generated cases and one 64-grid import.
All five fields, routes, target scores and full growth reports (except wall time)
match MG7. Every job source ZIP matches its captured manifest and current deployed
source. The regression, parity and final-browser snapshots each match 168 current
Python sources. Cross-version same-site comparison works; cross-site comparisons
are explicitly descriptive. 390 px viewport has no horizontal overflow. Layer 63,
cutaway, export action, queued file import, local history and zero console errors
were checked. Mobile and desktop screenshots/DOM evidence are retained.

## Observed local timings

| Browser workload | Generation seconds | Queued to saved seconds | Geometric outcome |
|---|---:|---:|---|
| mg5__mg3__partial_obstruction__v24__s2 | 0.224 | 3.442 | pass |
| mg48__48__offset_obstacle__s6 | 1.306 | 5.385 | pass |
| mg64__64__offset_obstacle__s7 | 2.597 | 6.975 | pass |
| mg64__64__blocked__s6 | 0.884 | 4.983 | blocked |
| mg64__64__offset_obstacle__s7 (import) | 2.328 | 8.306 | pass |

Queued-to-saved time comes from durable events and includes scheduling, worker
startup, generation, evaluation and persistence. It excludes the browser's next
poll and painting. UI observation times are retained as upper bounds, including
gaps between tool calls; they are not measured rendering latency or a speedup
benchmark. MG7's earlier paired numerical speedups remain separate evidence.

## Evidence and continuation

Frozen acceptance: STUDIO_V2_PROTOCOL.md. Runs: 20260925T090307Z_9e687a560fca, 20260925T090432Z_53e2d71911c5, 20260925T090806Z_eea8af61e5c0, 20260925T091209Z_181221f5be79, 20260925T091257Z_bce5f23dd4ef, 20260925T092014Z_06e493b0f94b.
Read experiments/reports/MS2-summary.json, MS2-parity.json and MS2-browser.json.
Exact artifacts live in .local-artifacts/runs/<ID>, retained MS2 fixture roots,
studio-mass-v2 and studio-mass-v2-jobs. Old stores remain intact. The milestone
archive includes raw source, all these runs, local Studio state and a verified
Git-bundle restoration. Keep older archives; local copies are not off-device.

Next review the larger generated volumes with the user, then define a versioned
NCA learning baseline against this procedural comparator. Freeze held-out sites,
volume objectives, recovery tests and an explicit Colab compute allowance before
training. Broader seeds/sites/volume choices also need a separate evaluation.
No paid compute or Drive operation is implied. User guide: STUDIO_V2_USER_GUIDE.md.
