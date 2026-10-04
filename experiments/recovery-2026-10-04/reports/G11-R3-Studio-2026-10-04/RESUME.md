# R3 local generation studio — 2026-10-04

Completed: http://127.0.0.1:8014/ . Select one of31 saved site presets, volume16/24/32%, integer firing seed, then Generate comparison. Each request performs actual CPU inference for raw G10 and R3; saved evaluation cases are retained as a separate gallery. One job at a time. MG7 and the original preview at8013 remain unchanged.

## Implementation and evidence
Frozen R3 adapter from the variety milestone (optional firing seed, otherwise unchanged), fixed G10 checkpoint and nine-family evaluator. Same32³ grid. No training, smoothing, post-hoc filling or silent fallback. If the geometry planner cannot certify its witness, the R3 panel states no certified output while raw G10 remains available. Worker errors persist with their traceback. Passing per-run checks covers all nine families at64/128, absolute volume error<=4pp and late growth<=5%; it is not a population median gate or architecture certification.

Runs live in runs/<unique-ID>. Each retains request, exact scene, frozen model/source hashes, runtime versions, condition, certificate attempt, available trajectories/states/provenance, fields64/128, metrics, status and SHA256 manifest. Writes use temporary files then replace for atomic status visibility. Startup marks unfinished jobs interrupted and preserves their previous status. Interrupted inference is not resumable mid-rollout; a new run is required. No deletion endpoint. Runs created after the milestone ZIP require a new archive to be included in backups.

Loopback server; strict local Host and Origin, custom request header, bounded request body, fixed preset/volume options and seed validation. This is a local experimental tool, not hardened public hosting. No external scripts or network dependencies. Static routes are allowlisted and do not expose the model or local filesystem. No arbitrary-scene editor yet.

## Verified
Browser-generated run9ef005ca3b1f451c954331bab59aa52d: wider-gap site,24%,seed2101. All four actual fields exactly match the prior independently saved variety outputs. Result reopened from persisted history after browser reload; no captured browser errors. Source-overlay visualization inspected. History selection retention fixed before completion.

Invalid seed rejected400; foreign Origin rejected403. Synthetic fault-injection checks (separately labelled under verification/, never quality evidence) confirm missing route yields explicit R3 failure with raw comparison retained, and worker exception writes error evidence and releases the job lock. Startup interruption handling is implemented but not exercised with a process kill. Mobile layout, cancellation and sustained-load behavior unverified; do not claim production readiness.

## Next
Use the studio to review repeatable site/seed comparisons and select what meaningful form variation should change. Add controlled geometry editing and/or planner variation only as a separately versioned change, preserving this reference. Do not increase resolution solely for appearance. No new paid run authorized or needed for this milestone.

## Resume
Prior: ../G11-R3-Variety-2026-10-04/RESUME.json . Restart with C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA/.venv/Scripts/python.exe followed by this folder's server.py. Before restarting, check whether8014 is already serving. Records here are current; repository synchronization remains pending. No Drive access, publication, push or live-model promotion. Verified ZIP is a same-disk copy, not off-device backup.
