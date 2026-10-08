# R3 workflow studio — 2026-10-08

Separate preview on http://127.0.0.1:8021/#skin=graphite. Studio8020 and all
previous evidence remain intact. Classic, Graphite and Porcelain are retained.

Draft saving/restoration and history were already implemented. Added download of
a selected local run, persistent run links and reload reopening. The viewer's
selected result determines export; generation controls remain a separate next-run
setup. Use Edit selected run to bring the saved inputs into that editor.

The ZIP includes four OBJ surfaces where available (G10/R3 at64/128), original
NPZ trajectories/fields, scene, request, seed, config, runtime, model/source
identity, metrics and per-model acceptance. Failed outputs remain labelled;
unavailable R3 produces no fabricated geometry. Every payload is hashed. Source
run hashes must match before export. These hashes detect accidental changes,
not malicious rewriting of both data and manifest.

OBJ coordinates are XYZ in metres, Z up, origin at the grid origin. Faces exactly
follow exposed voxel boundaries; no smoothing or filling. Import as metres since
OBJ has no intrinsic units. Geometry is generated mass only; existing buildings
remain in scene.json. Keep metadata with geometry. This is not a structural or
manifold CAD-solid guarantee. A reopen link works on this computer while this
runtime and its run directory remain available; it is not cloud sharing or a ZIP
import mechanism. Arbitrary archived ZIP import remains outside this milestone.

Validation: one browser-generated40-cube courtyard run, seed2102,24%,original
route:8f11090ceaef4f618f853acddb7c4988. All four occupancy fields reproduce the
preceding site-studio reference exactly. Draft4629bb69 was saved, changed, restored
and used to generate seed2102. Reopening after reload and skin changes verified.
Browser download completed to Downloads/NCA-run-8f11090ceaef4f618f853acddb7c4988.zip.
All25 payload hashes verified. All four exported OBJ surface counts, bounds and
signed enclosed volumes match the original voxel fields and0.8m scale.
Exporting an existing blocked run produces no R3 OBJ; a copied corrupt fixture is
rejected. No additional scientific runs. Scientific source/model hashes unchanged.

Initial issues and fixes: sandbox loopback access required elevated local launch;
Windows default text decoding corrupted overlay labels, corrected using UTF-8
before generation; legacy history() shadowed window.history, fixed using URL hash.
A browser network suspension resolved by a fresh tab. Pre-fix overlay retained in
initial-preview; completed run inputs/identity/manifest remain unchanged.

Source: experimental/r3-workflow-studio (overlay).
Prepare: .venv/Scripts/python.exe scripts/prepare_r3_workflow_studio.py
Start: .venv/Scripts/python.exe .local-artifacts/r3-workflow-studio-2026-10-08/server.py
Runtime already exists; do not rerun preparation over it.
Check: scripts/check_r3_workflow.py RUN_ID; output refuses an existing workflow-check
folder. Do not rerun just to reproduce the recorded pass. Detailed reports are in
experiments/reports/R3-workflow-check-2026-10-08.json and R3-workflow-browser-2026-10-08.json.

Next: reproducible local release packaging with launch instructions and preserved
results/drafts, then final handover. No new training,Drive,push,publication or MG7
replacement. Same-disk archives are not off-device backups.
