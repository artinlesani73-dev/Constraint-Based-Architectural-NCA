# NCA Studio — local experimental release1

1. Extract the ZIP into a NEW folder. Do not run it from inside the ZIP.
2. On this computer, double-click START-STUDIO.cmd. Keep its window open.
3. Open http://127.0.0.1:8022/#skin=graphite in your browser.
4. Choose a saved local run to review, or a site and seed to generate.
5. Save unfinished edits with Save draft locally. Completed runs save automatically.
6. Download run +3D geometry creates a ZIP with metre-scale OBJ surfaces and evidence.
7. Stop the server with Ctrl+C. Relaunch from this same folder to retain your work.

Classic,Graphite and Porcelain are available. A run link reopens a local saved
result; it is not a cloud-sharing URL. Generation controls describe the NEXT run.
Use Edit selected run to bring an existing result into the geometry editor.

## Environment

This package includes code,model and saved history. It is not a standalone Windows
executable and does not include Python or a virtual environment. On this computer
the launcher uses the existing project .venv (the full path is in START-STUDIO.cmd).
No downloads,dependency installation,cloud access or paid compute occur at launch.
Tested: Python3.12.14,NumPy2.5.2,PyTorch2.8.0+cpu on Windows. Startup checks those
versions and the included files. To use an equivalent environment elsewhere,set
NCA_PYTHON to its python.exe before running the CMD. Another machine still needs
its environment prepared and verified; fresh-machine installation is not tested.

Alternative launch from PowerShell:

    & 'C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA/.venv/Scripts/python.exe' './launch.py'

Verify without serving: append --check.
If8022 is busy,append --port 8023 and open the printed address. Do not stop other
services to free a port. Server binds only to127.0.0.1.

## Your saved work

New results live in app/runs and drafts in app/drafts. Keep the WHOLE extracted
folder. Never replace this folder with another release; extract upgrades beside
it and explicitly preserve/merge history. This release includes the unique saved
runs and drafts from the R3 scale,site and workflow previews; see history-index.json.
Other historical experiments remain in the project archives and are not erased.
Original included history is hash verified on every launch. New UUID-named runs
and drafts can be added normally and are not part of the original release manifest.
Individual runs carry their own evidence manifests. This is change detection,
not a cryptographic signature against hostile rewriting.

Keep a separate backup before moving or upgrading. A ZIP on the same disk does
not protect against disk failure. Google Drive access always needs your explicit
permission; this launcher never accesses it.

## Interpretation

R3 is an experimental hybrid of frozen G10 learned scores,procedural planning and
admission guards. It passed the bounded evaluations documented in evidence/;
this does not establish general architectural reliability. Neutral scores also
passed the tested geometric gates; learning's measured benefit here is influence
on form/variation,not proven necessity for validity. MG7 production is unchanged.
Voxels represent whole building volume,not construction or guaranteed usable rooms.
All nine constraint families remain unchanged. OBJ is an exact voxel surface in
metres,Z up,without smoothing; it is not a certified manifold CAD/structural model.
Failed outputs stay labelled and unavailable R3 has no fabricated mesh.
