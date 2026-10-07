# Editable surrounding-site volumes — October 8, 2026

Separate studio: http://127.0.0.1:8020/#skin=graphite . Source lives in
experimental/r3-site-studio; original 8018/8019 source and experiment evidence remain intact.

The editor now supports two facing primary buildings plus zero to four editable surrounding
blocks. Each block has X/Y/Z start/end coordinates, with live context preview, add/remove,
draft persistence (including incomplete coordinates), and restoration from saved runs.
Removing a block changes only the current draft; historical results and saved drafts remain.
Blocks may attach to existing volumes as wings; overlapping context boxes retain existing union
semantics. They are context volumes, not additions to generated mass. Only the first two
buildings declare connection faces. Server rejects more than six buildings, out-of-grid
extents, extra declared connection faces and invalid interfaces before allocating a run.

This is a bounded extension, not arbitrary urban geometry. All boxes remain axis-aligned,
with two X-facing connection interfaces and 32/40-cell grids at0.8m. More interfaces or rotated
faces require a separately versioned planner/input revision. No new constraint family,
model architecture, trained weights, loss or scientific evaluator was introduced.

## Consolidated check

Three new40-cell development fixtures were saved before generation: staggered pair with two
surrounding blocks, offset interfaces with a partial obstacle, and an intentionally blocked
gap. Fixed seed2102,24% requested volume,original route,128steps; CPU/two threads.
Each job had a180-second checker wait cap (not a server cancellation policy).

| Case | Result |
|---|---|
| 32-cell reference | All four G10/R3 fields at64/128 exactly match October6 outputs |
| Staggered courtyard | R3 passes per-run nine-family, volume and stability checks |
| Offset obstacle | R3 passes per-run nine-family, volume and stability checks |
| Full barrier | No certified R3 output; failed-check result and raw G10 retained |

Every run's manifest verified. The blocked failure is expected behavior, not an accepted
volume. All attempted cases are included in experiments/reports/R3-site-studio-2026-10-08.json.
Invalid block count/bounds/extra faces were rejected before creating jobs. JavaScript syntax
checks passed. Browser check preserved/restored an incomplete extra block and reopened the
completed courtyard with its correct context. Graphite/Porcelain visually inspected; no CSS
or renderer changes versus the previously checked32/40 studio. Source scientific modules
compare byte-for-byte to the original snapshot. Limited examples do not establish broad
validity, useful diversity, or a causal benefit from the learned model.

First checker attempt failed before reading scenes because the detached local server was not
available; no generation jobs were launched. Preserved integration/startup-failure.json.
Second attempt used a persistent server and explicit readiness wait; all four jobs completed.
No scientific retries or changing fixtures after results. Git initially did not infer a work
tree in this environment; explicit --work-tree=. worked without changing Git configuration.

## Continue and reproduce

Current runtime: .local-artifacts/r3-site-studio-2026-10-08.
```powershell
.venv/Scripts/python.exe scripts/prepare_r3_site_studio.py --name r3-site-studio-fresh
.venv/Scripts/python.exe .local-artifacts/r3-site-studio-fresh/server.py
```
Preparation verifies the original recovered skin snapshot, builds the scale base and applies
the new overlay/fixtures, freezing a new identity. It requires that recovered local snapshot
and refuses existing destinations. Check port8020 before starting another server. The checker
scripts/check_r3_site_studio.py targets the dated runtime and refuses to overwrite its
integration-v2 directory; inspect prior results rather than rerunning blindly.

Next: freeze one consolidated diversity/model-contribution comparison across the supported
site families. Compare raw G10, R3 and a clearly specified planner-only/procedural baseline
on matching requests before deciding whether to revise or retrain the neural generator.
Preserve blocked cases and separate development from a future untouched final evaluation.
Further geometry generalization remains outstanding; this milestone does not close all site work.

All source, fixtures and summaries are tracked. Full runtime evidence, failed startup record,
drafts and screenshots are archived locally under Codex outputs/R3-Site-Studio-2026-10-08.
Future runs require a new archive. No paid training, Drive action, push, public release or
MG7 replacement. Same-disk archive is not an off-device backup.
