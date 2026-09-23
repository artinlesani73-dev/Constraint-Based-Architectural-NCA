# Studio S1 — local procedural workspace

2026-09-24. First implemented product increment after the bounded F5 phase.

## Run and use

From PowerShell, run `deploy/run-studio.ps1` in the project, or:

```powershell
.venv/Scripts/python.exe -m uvicorn deploy.studio:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001. The script resolves its own project directory and
works when launched from elsewhere. The historical service remains unchanged.
Use only the loopback binding; S1 has no user accounts or public-hosting setup.

1. Select one of six frozen reference scenes. Source files are never edited.
2. Expand a building or entrance to change voxel coordinates. Building extents
   are half-open; entrance coordinates identify the minimum corner of a 2³ block.
   Gap-facing metadata follows the declared building face when X bounds change.
3. Apply edits validates and saves a new scene revision. Save scene also creates
   a snapshot. Invalid edits produce an explicit error. Unapplied edits are not
   durable; navigating away triggers a browser warning.
4. Build alternative runs the deterministic W1 procedural construction and
   evaluates actual geometry. Rebuilding the identical scene intentionally
   produces identical material, with a separate saved record.
5. Switch axonometric/plan/elevation views and visibility layers. Elevation is a
   projection, not a section cut. Buildings use translucent context surfaces.
6. Select a saved card to restore its scene and evidence. Export record downloads
   the complete JSON. The local copy remains authoritative; import is not yet built.

## Scientific meaning

`budgeted_witness_v1` is a procedural baseline, not an NCA prediction. The checkpoint
is read only for authoritative historical configuration; its weights are unused.
All geometry uses scene_v1, 32³, 0.8 m voxels, street_levels=6 and threshold>0.5.
There are no new constraint families. Evaluation reuses research_objective_v1
(including pre-clamp coverage evaluated on this binary field), its radius-six
fixed permitted envelope and facade_endpoint_v1 allowance. All nine penalties and
three regularizers are saved. The UI exposes the nine family penalties, exact
binary connectivity and material/envelope ratio. It does not aggregate them into
an architectural quality score.

The displayed joint check means compatible context plus binary connectivity and
3–12% material/envelope budget (1e-6 tolerance); it does not mean all nine families
pass. Infeasible scenes retain partial geometry and a failing status. Support is
geometric attachment, access is material connectivity, and thin paths can satisfy
these proxies. No walkability, mechanical safety or minimum thickness is implied.

## Persistence and limits

Each scene/result/failure gets a unique UTC/UUID directory under
`.local-artifacts/studio/<id>/`. The submitted scene/provenance is fsynced to
`request.json` before computation. `record.json` is fsynced before `receipt.json`
publishes its SHA-256. Retrieval verifies the payload; incomplete or corrupt
directories are retained and reported in the library. Writes never overwrite a
record. Results include exact scene/hash, material and guide voxel coordinates,
routing, evaluation, settings, source hashes, config-source checkpoint hash and
runtime versions. These are design-study records, not scientific training runs.

Source files are represented by hashes; use this milestone's source archive/Git
bundle for replay. The record alone does not embed Python source or dependencies.
Filesystem integrity checks detect accidental payload corruption, not deliberate
tampering by someone who can rewrite both files. Local storage is not an
off-device backup. No Drive operation occurs.

One procedural computation runs at a time per server process; excess requests
return 429. Requests are bounded to 100 KB, 12 buildings and eight entrances.
Do not run multiple workers for S1. No asynchronous queue/cancellation is claimed.
A browser disconnect can leave a completed saved study; reopen the library before
retrying. A process crash during computation retains the submitted request as an
incomplete record; there is no automatic resume or replay. S2 must add the full job
lifecycle before expensive work. An incomplete write remains visible as an
integrity issue. Disk-full failure may prevent writing its failure
record and must be diagnosed from server logs.

The canvas draws exposed voxel faces on demand and building boxes without remote
libraries, model assets or one object per voxel. It has three fixed parallel views;
no orbit, picking, clipping/section tool, drag editing, GLB export, saved projects
or side-by-side comparison yet. This is a first product slice, not completion of
M4 or a measured tenfold performance improvement.

Validation is recorded in the S1 verification report and browser QA record.
