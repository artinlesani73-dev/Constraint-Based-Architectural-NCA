# MS2 frozen integration acceptance

2026-09-25. D072. User authorized integration after MG7. MG7 generator and MT1
remain unchanged. Introduce /api/mass-v2 and /static/live-v2 with a separate
durable store; retain MS1 routes, source, records and imports. New interface can
read/export/compare old records, and imported MS1 records replay the recognized
local MS1 code in a worker. Never execute bundled source. Old imported source
bytes remain available for re-export. No automatic overwrite of old results.

Admitted presets: the five existing 32-grid contexts, seeds 0/1/2 and requests
16/24/32%; three MG6 contexts at each of 48 and 64, seeds 6/7 and request 24%.
57 combinations total: 44 expected MT1/request passes and 13 blocked failures.
Exact MG7 matrix comparison required for all 57, including full growth reports
except wall time. Fixed 0.8 m voxels and 2.4 m growth cubes; same nine families.
Large context arrays are copied losslessly from saved MG6 inputs, embedded as
compressed base64 in a UTF-8 manifest so existing source exports retain them.

Keep original per-generator cooperative settings (15/45/120 seconds). New job
manager additionally enforces 180 seconds wall from worker launch, checked each
queue tick (~0.2 s while server runs), terminating the owned process tree. This
is not a hard real-time or host-suspension guarantee. Four pending jobs maximum.
Imported payloads limited to 20 MB before parsing, explicit format/version checks,
source/record hash checks, bounded distinct integer coordinates and known inputs.
Replay runs in the same queue with cancellation, parent-death cleanup and retained
failure evidence. Retries preserve request/payload and parent ID. Repeated imports
may create new attempts; retain each, do not overwrite or silently discard.

Before browser acceptance, test old/new foreign import and re-export, geometry/
trace/source/version rejection, larger input restrictions, request integrity,
queued/running cancellation, linked retry, worker deadline and restart recovery.
Use real workers for admitted 32/48/64 cases plus 64 blocked; retained full suite
must pass. Test deadline enforcement with a deliberately sleeping local fixture
and short injected deadline; record that this is not the production 180 s run.
Parent-death tests must confirm the owned worker stops and recovered attempt is
interrupted. Keep synthetic failures distinct from geometric controls.

Browser acceptance: generate 32 partial/24%/seed2, 48 offset/24%/seed6,
64 offset/24%/seed7 and blocked64/24%/seed6. Verify stored fields against MG7,
scale labels, slice range, cutaway, comparisons, library, export/import queue and
old-result access. Inspect narrow viewport and console errors. Measure actual
submission-to-result wall time separately from generation and first UI display;
no numerical speedup is promoted to an end-to-end speed guarantee.

Save exact source, all results and failures under unique run IDs. If a check fails,
retain it, fix the implementation and link the next attempt. No scientific tuning.
Document findings, decisions, resume steps, local commit and verified raw-source/
Git-bundle archive. No Drive, paid training, push or public hosting.
