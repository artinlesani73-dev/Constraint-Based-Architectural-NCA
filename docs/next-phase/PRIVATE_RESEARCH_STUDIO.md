# Private research studio implementation plan

Decision recorded 2026-10-09: user selected a private research studio rather than a public demo. Local preparation is authorized; hosting purchase, external upload, account/access changes and deployment have not been authorized. Local-only versus remote personal access is pending user clarification.

## Baseline and boundaries

Preserve release1 (15 runs,6 drafts) and its manifest/archive unchanged. Build a separately versioned app. Keep Classic, Graphite and Porcelain. Preserve G10/R3 comparison, exact voxel geometry, nine constraint families, failed outputs and model/source identity. R3 remains experimental; MG7 production is unchanged. No new training is needed for this software phase.

Inspected experimental/r3-workflow-studio/server.py and deploy/. The experimental server uses ThreadingHTTPServer, loopback Host/Origin checks, filesystem JSON records and an in-process lock for one job. It is not an authenticated remote service. The existing deploy folder contains FastAPI infrastructure but also historical model paths; do not silently substitute those for the frozen R3 scientific pipeline. Reuse infrastructure selectively after inspecting its contracts.

## Implementation order

1. Persistent storage boundary. Separate immutable app/model files from writable runs,drafts and job records. Configure one explicit data directory. Import release1 by copying and verifying manifests; retain originals. Use UUIDs, atomic writes and schema versions. Provide a migration record and inventory before/after. Do not expose arbitrary filesystem paths.
2. Recoverable execution. Run generation in a separate worker process with one concurrent job initially; persist queued/running/completed/failed/interrupted states. On restart retain partial evidence and label interrupted work; retry creates a new ID with lineage. Preserve deterministic scientific configuration and hashes. Avoid duplicating neural rollout logic.
3. Research workflow. Add searchable run history, user labels/notes, comparison selection and clear running/failed/accepted states. Store annotations separately from immutable experiment evidence. Preserve drafts, scene editing, source attribution and exact export. Model validity and visual preference must remain distinct.
4. Private access. If local-only, retain loopback binding. If remote, choose a hosting/access design after user scope and budget are known; require authenticated access to HTML,assets,API,exports and job submission, HTTPS at ingress, trusted proxy handling, request limits and one-job admission. No public endpoint before access checks pass. Do not turn the current local Host check into a claim of authentication.
5. Recovery and release. Build a versioned package with persistent data mounted outside app updates. Demonstrate one existing-run import/reopen/export and interrupted-worker recovery using a cheap fixture; run one frozen scientific parity check only if generation code changes. Document startup,upgrade,rollback and storage requirements. Off-device backup requires a separately approved destination; Drive operations still require permission each time.

## Completion criteria

All existing run/draft payloads remain accounted for and hash verified. Restart preserves history and labels partial work correctly. No overwrite/delete of scientific evidence. Both model outputs and failures remain traceable to inputs,seed,config and model/code identities. A new package can be rolled back without reverting experiment data. For remote scope, unauthenticated requests cannot read any studio data or start work. Hosting costs and transfer scope are approved before provisioning/upload.

## Next action

Implement the data-directory boundary as a separate version, then consolidate recovery and workflow checks. Remote-access preference is pending; it does not block local storage work. No paid compute, cloud operation, push, model promotion or production change performed by this planning milestone.