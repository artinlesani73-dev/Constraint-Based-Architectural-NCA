# Repository recovery, October 4, 2026

The repository had remained at October 3 commit 3f129f8 while subsequent development
continued in Codex outputs. This import repairs that source-control gap without replacing
the older production implementation or rewriting historical evidence.

## What is tracked

- Source, notebooks, configuration, reports, decisions and small evidence records from
  64 later development snapshots (including failed attempts and superseded variants).
- Byte-identical text objects up to 512 KiB, deduplicated by SHA256 under
  `experiments/recovery-2026-10-04/objects`. `index.json` maps every original path to its object.
- Readable milestone Markdown under `experiments/recovery-2026-10-04/reports`.
- Approved R3 studio source under `experimental/r3-studio` at its original relative paths.
- Historical orchestration scripts indexed under `tooling`; these may contain absolute
  paths and should not be blindly rerun. Final deployed snapshots override earlier builders.

Every indexed payload has its original size and SHA256. Larger text, checkpoints, arrays,
images and ZIPs remain in the original local output folders; Git records their identity and
location, not their bytes. The private next-phase report is excluded and remains ignored.
Historical documents retain their original pending-sync statements and local URLs; this
document records the later synchronization. No historical experiment is rerun or reclassified.

## Restore the approved studio

From the repository, with the existing project Python environment:

```powershell
.venv/Scripts/python.exe scripts/restore_recovered_snapshot.py G11-R3-Skins-2026-10-04 --check-only
.venv/Scripts/python.exe scripts/restore_recovered_snapshot.py G11-R3-Skins-2026-10-04
.venv/Scripts/python.exe .local-artifacts/recovered/G11-R3-Skins-2026-10-04/server.py
```

The last command starts the experimental loopback server on port 8018. Check that the
port is free first; do not start a duplicate of the currently running studio. The restore
checks all hashes before copying and refuses to overwrite an existing destination.
Use `--artifact-root PATH` if the original output folders have moved. Missing payloads
fail explicitly. A Git clone alone cannot restore checkpoints or full experiment evidence.
New runs/drafts in the restored runtime remain ignored and need separate evidence archiving.
The readable studio source is a frozen import; future development must use a new version
and explicitly regenerate its identity, never silently alter the original snapshot.

## Current scientific position and next step

R3 combines the learned G10 update with explicit route/witness planning and a cumulative
admission budget. Its success must not be attributed entirely to neural learning.
The frozen independent assessment passed 81/81 cases; these are now regression evidence,
not an untouched test set. Route alternatives are experimental, with limited exposed-site coverage.
Classic/Graphite/Porcelain were approved by the user. The 40-cube probe passed on one expanded
physical site at unchanged 0.8m voxel size; observed rollout 8.67s versus 3.94s at 32-cube.
One timing sample per size does not establish typical latency or broad scale generalization.

Next is a dimension-aware experimental 40-cube studio, retaining 32-cube reference behavior.
Do not claim a released larger-grid studio yet. No architecture, weights or evaluator were
changed by this recovery. Production MG7 stays unchanged; no training, Drive access or push.

## Ongoing working rule

Make source and documentation changes in the repository from now on. Commit verified local
milestones there. Store large generated results separately with hashes and tracked summaries.
Continue preserving every failed attempt and decision. Local archives are not off-device backups.
