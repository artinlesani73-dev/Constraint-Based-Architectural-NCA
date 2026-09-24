# Studio S2: durable local studies

2026-09-24. Extends S1 with background jobs, actual cancellation, explicit restart
state, revision-aware comparison and checked portable import. S1 documentation
remains historical; this document supersedes its job/import/comparison limits.

## Use

Run from the repository: `.venv/Scripts/python.exe -m uvicorn deploy.studio:app
--host 127.0.0.1 --port 8001` (one worker). Open http://127.0.0.1:8001.
Build scaffold submits a durable job and leaves scene editing available. Job
activity displays queued/running/named stages/completed/failed/cancelled/interrupted.
Progress is a named stage, not a fabricated percentage. Each completed result is
saved even if the browser closes or the user edits another scene.

Cancel terminates the worker; Retry creates a new linked attempt without changing
the original. Completion wins a cancellation race if the worker has already exited
and its candidate validates. A cancelled partial candidate is never published.
The local queue allows four pending jobs, with one worker process at a time.

Select two saved results and Compare. Both actual geometries follow the main
view/layer controls. The table displays all nine family values plus connectivity,
budget and material changes. Building/entrance revision details are expandable.
Different scene hashes are explicitly a descriptive comparison, not a controlled
model comparison. Metric definitions and grid/units must match.

Export record downloads the server JSON text, preserving numeric encodings used
by its checksum. Import record sends the original file text without browser
reserialization. Supported checksum envelopes are checked for scene validity,
coordinate bounds and exact procedural replay of geometry and diagnostics.
An existing matching record opens without duplication. New imports preserve their
source envelope and claimed provenance separately from current local provenance.
Old raw S1 exports are accepted only when identical to an existing local record;
otherwise re-export from their original workspace. This is integrity and supported
method verification, not authentication of authorship. Arbitrary NCA results are
not accepted. Import files are limited to2MB; replay is synchronous and bounded
to the existing32-cube procedural method, not the cancellable job queue.

## Persistence and interruption

Jobs live in `.local-artifacts/studio-jobs/<id>`; completed records remain in
`.local-artifacts/studio/<id>`. Immutable request, hash-chained events, exact
source ZIP, progress files, worker log and any candidate remain inspectable.
Record payload and receipt are fsynced; a receipt publishes the saved result.
Partial/corrupt histories are retained and reported, not repaired silently.
Progress files are informational and are not part of the hash-chained history.
Checksums detect accidental corruption, not an attacker rewriting all hashes.

An OS lock prevents two live managers from owning the store. The worker receives
GO only after process ownership and a durable running event. On Windows, a Job
Object kills worker and descendants on cancellation or owner-process death,
including during numerical-library loading. A parent-pipe watchdog starts after
runtime imports; starting it before those imports caused an observed NumPy import
hang and was corrected. Actual Windows owner/child/grandchild death was tested.
The non-Windows fallback stops the direct worker; descendant-tree behavior has
not been established there. Use the tested local Windows path for this milestone.

On restart, an intact already-published result recovers as completed. Other
pending histories become interrupted. No computation is automatically replayed
or resumed from a numerical checkpoint. Retry uses the same submitted scene,
current code/config and a new linked ID. Persisted PIDs are never blindly killed.
Graceful shutdown interrupts unfinished workers. Filesystem exhaustion may
prevent writing status; retained files and server logs then require inspection.

The old synchronous S1 records endpoint remains for compatibility; cancellation
applies to jobs submitted through the new job endpoint and Studio build button.
No paid or long-running NCA job has been admitted through this implementation.

## Meaning and remaining work

The interface explicitly says procedural scaffold, not trained NCA or inhabitable
space. Thin material can connect entrance regions and satisfy these proxies.
Access currently means connected material, not a walkable void or floor system;
thickness is a bulk diagnostic, not a minimum usable width. More voxels or longer
training alone cannot establish the missing spatial meaning. Read SPATIAL_BRIEF.md
before any further learning protocol; retain the existing nine families.

S2 is another M4 product increment, not production hosting or a measured10x gain.
Orbit/sections, direct spatial editing, richer geometry and larger grids remain
later work. No scientific nca file, original model, reference scene or historical
serving path changed. No paid compute, Drive access, remote push or publication.

## Verification

Run20260923T231009Z_672a788a5c0d:224 tests,0failures/errors/skips, smoke exit0.
This covers final backend; the later frontend-only JSON transfer fix was verified
with actual browser downloads/imports.13 S2 tests cover owned processes, crashes,
queue bounds, cancellation/retry, import tampering and geometry comparisons.
Browser checks and retained failures are recorded in
`experiments/reports/S2-studio-verification.json`; screenshots, portable fixtures
and final source are under `.local-artifacts/studio-qa/S2-20260924`.
Keep the new incremental S2 archive together with S1 and the full F5 archive.
Archive receipt under `.local-artifacts/milestones` establishes verified completion;
ZIP existence alone does not. Same-disk copies are not an off-device backup.
