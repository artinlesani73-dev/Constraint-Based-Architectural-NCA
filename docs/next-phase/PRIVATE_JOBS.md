# Local recoverable worker generation — 2026-10-10

Private local-only app, worker overlay over verified storage version. Final app: .local-artifacts/r3-private-jobs-v3-2026-10-10. Research data remains .local-artifacts/r3-research-data-2026-10-09; release1 and storage-version snapshots retained.

Start from repository with existing project Python:

    .venv/Scripts/python.exe -B .local-artifacts/r3-private-jobs-v3-2026-10-10/launch.py --data-dir .local-artifacts/r3-research-data-2026-10-09 --port 8023

One server owns studio.lock for its lifetime. Workers own worker.lock through model loading and generation. These are Windows OS byte-range locks; lock files are retained and crash release is handled by the OS. A new launcher refuses ownership if another studio or worker is active. On startup, while holding worker ownership, queued/running records become interrupted; unique interruption events retain previous statuses, inputs and partial files. Recovery is idempotent. Delayed workers check queued status under the same worker lock and do not revive an interrupted record. Jobs execute the existing generate() function in a separate Python process; neural source/model and rollout logic are unchanged. One job is admitted at a time; this is not a multi-job durable queue. Retry uses a new UUID and optional existing parent_run lineage via the editor. No automatic retry or partial numerical continuation.

A parent server crash does not forcibly terminate its worker. The worker may finish; restarting while it is active refuses access. Wait for completion or explicitly stop the owned worker; never infer that a closed browser means the worker stopped. After worker exit, restarting classifies unfinished evidence. Completed evidence is retained. Watcher finalizes manifest after worker log closes. Graceful job cancellation and multi-job scheduling are not part of this milestone. Older storage-only app does not honor these new locks: do not run it concurrently against this data folder. It was stopped before switching the actual studio.

Verification: scripts/check_r3_private_jobs.py. Synthetic fixture writes partial evidence, is terminated, then recovery preserves it; active worker and duplicate server are rejected; delayed worker cannot resume. One real frozen32-cube comparison is checked against saved run43182d564aa54abeb7d69c1d59752450 at64/128 for both models, plus manifest hashes and untouched historical inventory. This is a regression/recovery check, not a new scientific benchmark. Report: experiments/reports/R3-private-jobs-check-2026-10-10.json.

Attempt history retained:
- First app/check: locked-file probing read the locked byte and raised PermissionError before friendly ownership handling. Corrected probing uses file size, not reading the locked range.
- Second check: immediate reacquisition after termination raced Windows lock release. Fixture now waits at most5 seconds, preserving fail-fast application behavior.
- Check-v3 passed with exact four-field parity and all recovery assertions.
- Final app-v3 removes potential double release of admission after worker-start failure. HTTP caller owns failed-start cleanup; watcher owns successful-start cleanup. Final consolidated check-v4 verifies the final build.

No cloud, Drive, training, paid compute, push, production model change or new constraint families. Next searchable run history, separate labels/notes and comparison workflow, followed by final local package. Do not claim numerical checkpoint resume or remote authentication.
Final live API check: initial sandbox request was denied socket access (its displayed zero count was not a data result). Retried with localhost permission: verified15 saved runs on8023.
