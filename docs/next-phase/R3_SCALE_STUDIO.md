# R3 larger-domain studio — October 6, 2026

Implemented a separate experimental studio at http://127.0.0.1:8019/#skin=graphite.
Source: experimental/r3-scale-studio. Frozen parent: experimental/r3-studio.
Existing 32-cube cases remain available; the previously benchmarked scale-40 site is now
selectable and editable. Both use 0.8m voxels: 25.6m versus 32m domain edge. No finer-resolution
mode is implemented. Building heights, street band and interface extent retain the benchmark semantics.

The server accepts only 32/40-cell scenes and requires custom geometry to match the selected
site dimension. Each run saves its effective config. Classic and both experimental skins
decode voxel IDs, center cameras, cull internal faces and draw context using the scene's own
dimension. Editing bounds, saved result labels and draft/revisit behavior follow that dimension.
Historical 32-cell files, original studio and production MG7 are unchanged.

## Verification

One bounded HTTP integration check generated paired G10/R3 outputs on scale-32 and scale-40,
24% request, firing seed2102, original route,128steps,CPU/two threads. R3 occupancy exactly
matches the frozen October4 scale benchmark at BOTH64/128 horizons for BOTH dimensions.
Both runs completed the per-run nine-family, volume and stability gates. Stored voxel IDs,
array dimensions, effective config and run manifest hashes verified. This is integration
regression evidence on exposed geometry, not a new unseen-site evaluation.
Unsupported64-cell and mismatched custom geometry requests return400 without allocating jobs.
An incomplete40-cell draft round-trips exactly through the local API.
JavaScript syntax checks passed. Visually inspected40-cell output in Classic, Graphite and
Porcelain; saved-run editing restores40-cell geometry/seed and editor maximum40.
Frozen scientific source matches the parent byte-for-byte. No trained weights or nine-family
criteria changed. Broader site coverage, route-mode performance at40 and accessibility remain work.

The initial preparation command was blocked by filesystem permissions before destination
creation. The authorized retry succeeded with no partial runtime or scientific run to discard.
Browser label lookup required the observed element ID; this did not change app contents.

## Reproduce and continue

From the repository:
```powershell
.venv/Scripts/python.exe scripts/prepare_r3_scale_studio.py --name r3-scale-studio-new
.venv/Scripts/python.exe .local-artifacts/r3-scale-studio-new/server.py
```
The verified recovered parent must already exist; see the repository recovery instructions.
Check port8019 before starting another process. Preparation refuses to overwrite a runtime.
Current runtime: .local-artifacts/r3-scale-studio-2026-10-06. Historical runs/drafts are copied
unchanged, with their original identities. New runtime identity freezes the updated app and scenes.
The focused checker scripts/check_r3_scale_studio.py targets the dated runtime and refuses to
overwrite its integration evidence. Do not rerun it blindly; use a new version for a new attempt.

Next: broaden supported site geometry in one bounded implementation milestone; preserve
32-cell regression behavior and the nine families. Current editor still supports two facing
buildings plus an optional preset obstacle; it does not support arbitrary urban arrangements.
No further paid training is needed for this completed milestone. No Drive, push or publication.
Evidence archive and screenshot: C:/Users/artin/Documents/Codex/outputs/R3-Scale-Studio-2026-10-06.
Archive captures current evidence only; future generated runs require a new snapshot.
Same-disk archive is not an off-device backup.
