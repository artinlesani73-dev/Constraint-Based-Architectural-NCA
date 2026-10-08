# Local release1 handover — 2026-10-08

Ready folder: C:/Users/artin/Documents/Codex/outputs/NCA-Studio-Local-Release-1/NCA-Studio

Double-click START-STUDIO.cmd inside that folder, then open the printed URL
(default http://127.0.0.1:8022/#skin=graphite). Keep the terminal open. Ctrl+C stops
it. Saved work lives in app/runs and app/drafts. The existing8021 studio is unchanged.
If the port is occupied, launch with --port 8023; no existing service is stopped.

The ZIP in the parent folder is10,760,503bytes with402 payloads. It includes15 unique
saved runs and6 drafts merged without changes from the R3 scale/site/workflow
runtimes. Duplicate paths must match hashes; conflicts stop packaging. History-index
records provenance. This is a studio release,not the entire research archive;
all other experiments and older runtimes remain intact.

The package includes code,model,all three skins,reopen links,verified geometry
export,current findings,launcher and instructions. Only server port plumbing changes;
scientific source and model hashes are unchanged. It is loopback-only and performs
no dependency installation,cloud access,paid compute or publication.

This is NOT a standalone executable. The CMD uses the existing project .venv on
this computer. Tested versions: Python3.12.14,NumPy2.5.2,PyTorch2.8.0+cpu. NCA_PYTHON
can select another already prepared equivalent environment. Startup checks versions
and all included files. Fresh-machine environment installation is not tested.
Do not overwrite this folder when upgrading; keep the whole folder,including new
UUID-named results/drafts. New files are allowed but are outside the original release
manifest; saved runs have their own manifests. Back up the whole folder separately.
Same-disk copies are not off-device backup. Every Drive operation still needs approval.

## Verification

Every app source hash and included saved-run manifest verified before packaging.
Fresh extraction passed preflight,listed all15 runs/6 drafts,reopened courtyard run
8f11090ceaef4f618f853acddb7c4988,exported verified payloads and saved a new draft with
the configured8022 origin. No generation. Test server stopped; test copy retained.
START-STUDIO.cmd --check also passed. Delivered ZIP and extracted files hash verified.

Archive SHA256: bc55ca5eaa5ebeb57c852812da21d9cf9a19783a083cf5802a2e64e66e2fc5b6.
An independent same-Windows rebuild reproduces exact archive bytes. Its first
check used string ordering instead of Windows Path ordering and failed before
delivery; correcting the verifier to match builder ordering passed. The original
ZIP was not altered. This is not a cross-platform byte-reproducibility claim.

Build scripts/build_r3_local_release.py refuses an existing output.
Check scripts/check_r3_local_release.py refuses an existing test directory.
Launcher source experimental/r3-local-release.
Frozen build .local-artifacts/r3-local-release-2026-10-08 and matching.zip.
Smoke evidence .local-artifacts/r3-release-smoke-2026-10-08.
Tracked summaries experiments/reports/R3-local-release-{build,check,delivery}-2026-10-08.json.
Do not overwrite these completed attempts or rerun them merely to repeat passes.

## Status and next decision

The bounded evaluation,studio workflow and local-release milestone are complete.
R3 remains an experimental hybrid; MG7 remains production. This does not establish
broad architectural reliability or close all research questions. Neutral scores
also passed the tested geometric gates; learning is not demonstrated necessary for
those gates. Geometry remains whole-volume mass,not rooms or certified construction.

Next is user acceptance of this packaged workflow. Fix specific observed issues;
otherwise define public deployment/production scope separately before publishing.
No further automatic scientific sweep or paid retraining is planned. Standalone
installation,broader architectural evaluation,arbitrary-site support and off-device
backup remain outside this completed local release milestone.
