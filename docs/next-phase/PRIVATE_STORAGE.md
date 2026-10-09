# Local private studio: separate research storage

Completed 2026-10-09. Local-only choice confirmed by user; remote hosting is out of scope.

App: .local-artifacts/r3-private-storage-v2-2026-10-09
Research data: .local-artifacts/r3-research-data-2026-10-09
Original release1: .local-artifacts/r3-local-release-2026-10-08 (unchanged)

Build: .venv/Scripts/python.exe -B scripts/build_r3_private_storage.py
Build refuses an existing output. The first build at r3-private-storage-2026-10-09 stopped at a false-positive assertion: ROOT/drafts also matched the tail of DATA_ROOT/drafts. Partial output retained. Corrected check counts the two rewritten draft paths; successful build uses a new v2 attempt directory. Its app identity is R3-private-storage-v1 (first completed storage version).

Start from repository:

    .venv/Scripts/python.exe -B .local-artifacts/r3-private-storage-v2-2026-10-09/launch.py --data-dir .local-artifacts/r3-research-data-2026-10-09 --port 8023

Add --check to verify without serving. Data initialization, performed once, used --import-release .local-artifacts/r3-local-release-2026-10-08 with --check. Never repeat initialization into an existing store: imports refuse overwrite. Use the same explicit data directory on future app versions only when their schema compatibility has been verified. Relative paths resolve from the invoking working directory; absolute paths are recommended for launch shortcuts.

Storage schema1 requires runs/ and drafts/. Import validates release hashes and completed-run manifests, refuses active source jobs and app-overlapping destinations, copies to unique staging, verifies complete before/after inventories and unchanged source, then renames the staging directory. Failures retain staging. store.json records schema; import-receipt.json records provenance and every copied hash. App and model files have a separate private-manifest.json. HTTP clients cannot choose the server data directory. Loopback binding and existing request guards remain. This is a local app, not remote authentication.

One consolidated check passed: scripts/check_r3_private_storage.py, report experiments/reports/R3-private-storage-check-2026-10-09.json, evidence .local-artifacts/r3-private-storage-check-2026-10-09. Imported15 runs/6 drafts; verified all imported hashes, saved one test draft, restarted and found it, reopened the courtyard output, verified every export payload hash, rejected an existing import destination and app-directory store. Source and app remained unchanged. Test draft a04020efb1da43c98001dd0b2f1bd046 exists only in test storage; actual research store starts with original6 drafts.

Actual store imported separately with launcher preflight. Browser on8023 reopened run8f11090ceaef4f618f853acddb7c4988 and showed original15-run/6-draft history and comparison. Screenshot C:/Users/artin/Documents/Codex/outputs/NCA-Private-Storage-2026-10-09/preview.png . Server session64426 may remain running; stop its terminal with Ctrl+C. Earlier release1 on8022 remains separate: new work there does not automatically merge into the new store. Use8023 for new work. Future additional migration must detect conflicts and preserve both stores.

No new generation/training; model and scientific source hashes unchanged. No Drive, push or cloud action. This same-disk store is not off-device backup. Storage integrity hashes detect differences, not authentication against an attacker.

Remaining: worker-process execution and robust interruption recovery, then searchable labels/notes/comparisons and final versioned release. Current inherited startup interruption handling is not yet the new recovery milestone. Do not claim safe concurrent sharing: run a single server per data store; cross-process locking is pending that milestone. Do not roll app versions back across incompatible schema changes without a versioned migration/restore plan.