# Resume the NCA next phase

Last updated: 2026-09-23. Status: rollout controls and E0 reporting checks passed;
E0 run `20260922T230120Z_76f3b4677e8f` completed all 270 cases from commit
`474bf53`, with no execution failures. All registered artifacts verified.
Read E0_FINDINGS.md and its linked detailed report. No training has started.

## Authorization and storage

The user approved next-phase implementation, with research and product quality
both important, preservation of all decisions/results, no new constraint families,
and local archives. The user declined the prepared Drive upload and said to keep
it local (D014). No Drive access is currently authorized. For any future Drive
work, read AGENTS.md: only folder `1fS34Yy0-oMzSxWaYJFiPTkGgrZstgc0H` and actual
descendants are in scope, and EVERY operation, including reads, needs approval.
Do not repeat the declined upload automatically.

Paid Colab is available but no compute cap or training job is authorized. The
private NCA-Next-Phase-Report remains Git-ignored. Preserve the user's untracked
NCA-Studio-Concept.html. No Git push or deployment has been performed.

## Checkout and evidence

- Project: `C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA`.
- Branch: `next-phase/foundations`; original historical baseline: `ac913b9`.
- Foundation milestone: `b841991`; scene contract: `ddc8100`; verified historical
  profiles and legacy scene set: `1dfafa7`.
- Full original 50-file snapshot is under `.local-artifacts/source-snapshots/`.
- Local backup ZIP and checksum: `C:/Users/artin/Documents/Codex/2026-09-06/cre/outputs/NCA-M1-Backup-2026-09-23-1dfafa7.zip`.
  Its Git bundle was restored successfully and both scene manifests verified.
  This archive predates rollout_v2/E0 work. Its receipt is under
  `.local-artifacts/milestones/1dfafa7-backup-receipt.json`.
- Historical notebook, checkpoint and evaluation files remain untouched.
- All run payloads and source snapshots: `.local-artifacts/runs/<run_id>/`;
  tracked summaries: `experiments/records/<run_id>.json`.

## Verified implementation

- `scene_v1`, frozen reference set (6 scenes) and `legacy_easy_v1` (12 scenes).
  Legacy seed state reproduces the notebook generator, including below-street
  facade entrances. Do not replace it with the deployed generator for that set.
- `rollout_v2`: profile firing rates reach internal delta masking; explicit torch
  RNG reaches the model step; incompatible mode/firing combinations are refused.
  Default legacy behavior is preserved; optional generator changes no weights.
- New notebook oracle executes original perception/model/legality/corridor code
  and train_epoch only through the forward result, never loss/backward/optimizer.
  Parity covers full, partial and absent seed-mask phases plus batch size two.
- Local acceptance run `20260922T225856Z_26be76516805`: 82 tests passed, no
  failures/errors/skips, out-of-directory checkpoint smoke exit 0.
- Earlier failed attempt `20260922T225408Z_f67923028c9a` is retained: missing
  helper in the isolated notebook oracle caused three subcase errors. Linked
  retry `20260922T225534Z_1c995e858265` passed 80 tests; the 82-test run adds E0
  reporting checks for empty, failed and unscorable cases.
- `.gitattributes` preserves LF in frozen scene JSON files on fresh checkouts.

## Next actions

1. Implement the versioned bounded vertical-envelope correction described in
   CORRIDOR_FIX_PLAN.md. Keep the original callable for E0 replay and record
   target differences on both frozen scene sets.
2. Correct target legality/routing interactions separately, including ground
   entrances and the endpoint-based height clamp. D016 records why merely
   removing forbidden target voxels is insufficient.
3. Repair loss definitions, tensor shapes and gradients. Then compare the
   corrected NCA with scaffold-only/procedural/direct-optimization controls.
4. Prepare Colab preflight/recovery only after those gates, and agree the compute
   cap and checkpoint backup procedure before any paid training. No user setup
   is needed at the present stage. Keep archives local as requested.
5. Product work can use the stable scene/result records and actual E0 cases to
   build the design workspace; do not imply an improved trained model yet.

## E0 evidence and interpretation

- Run: `20260922T230120Z_76f3b4677e8f`; 270/270 completed, zero failed cases.
  Report generation verified all registered artifacts against their hashes.
- Findings: `docs/next-phase/E0_FINDINGS.md`; detailed report and post-run target
  audit JSON in `docs/next-phase/reports/`, named with the run ID.
- On legacy scenes, training and serving each connect 10/12 scenes at each of
  three seeds; historical evaluation connects 0/12 and yields about 28 voxels
  per scene. The legal corridor target itself connects all 12.
- All main profiles fail to connect the five non-control reference scenes,
  despite permitted-space connectivity. The sixth reference is intentionally
  impossible. Both ground-only legal targets are disconnected, so fix target
  routing as well as the bounded-envelope bug.
- Single-seed ablations are diagnostic only. Serving without noise loses ten
  legacy connections; removing its mask fills far more volume without gaining
  connectivity. Do not promote a different default from these alone.
- The audit uses recorded source-snapshot scenes, preserving exact provenance
  even if a later Git checkout normalizes manifest line endings.
- Source commit for E0: `474bf53`; the later local evidence commit adds reports
  and the handoff. See Git log for its actual hash. A new local E0 backup is
  prepared in the outputs directory as `NCA-E0-Backup-2026-09-23-<commit>.zip`,
  with a checksum and a receipt under `.local-artifacts/milestones/`. Check the
  receipt before claiming the backup complete; the old M1 archive stays intact.

`run_e0.py --parent-run <id>` creates a fresh full retry linked to a retained
attempt; it does not resume midway through a case or overwrite prior results.
If interrupted, inspect run.json, cases/, artifacts/ and result.json first.
No final result means incomplete, even if many case files exist.

## Commands from the project root in PowerShell

```powershell
& .venv/Scripts/python.exe scripts/verify_foundation.py
& .venv/Scripts/python.exe scripts/run_e0.py
& .venv/Scripts/python.exe scripts/experiment.py verify <run_id>
```

The environment is Python 3.12.14, CPU torch 2.8.0, NumPy 2.5.2, recorded in
requirements-cpu.lock.txt. Do not install the CPU lock over Colab's CUDA runtime.
This desktop task can run local commands; older no-local-shell instructions are
superseded. Filesystem/Git write permissions may need renewal in a later task.

## Limits and recovery

These are regression and forward-diagnostic checks, not repaired-model quality
claims. The legacy corridor operator still has its vertical-envelope defect.
Ground openness/legality are enforced by hard masks, not proof of learning.
Geometric connectivity/support do not certify walking clearance or structures.
Shared-model concurrency and the interface redesign remain M4 work.

Read AGENTS.md, this file, PLAN.md, DECISIONS.md, CHANGELOG.md, GEOMETRY_CONTRACT.md,
ROLLOUT_PROFILES.md and SCENE_SETS.md. Inspect Git status and unfinished artifacts
before continuing. Historical handoffs remain recoverable from Git history and
milestone archives. This handoff does not automatically resume a session or
redeem credits when a usage limit resets.

## Active corridor implementation - 2026-09-23

The bounded and legal-routing operators are implemented separately. Read
CORRIDOR_PROTOCOL.md and D017. Preflight 20260923T000140Z_a7ff7de54684 passes 94
checks. Next run scripts/run_corridor_comparison.py locally (54 targets, 108
single-seed forward cases); no training. Inspect its newest run before retrying
and use --parent-run for a fresh linked retry. Preserve all intermediate records.
Then write the comparison report, update this handoff and archive the milestone.

C1 attempt 20260923T000715Z_1e06516e9763 FAILED due to Windows long evidence
filenames after 54 targets/7 recorded rollouts. Retained intact. Runner filenames
now use short stable hashes (full IDs remain in JSON). Next retry with
--parent-run 20260923T000715Z_1e06516e9763; inspect latest run before starting.
