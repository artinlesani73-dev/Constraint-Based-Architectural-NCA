# Resume the NCA next phase

Last updated: 2026-09-23. Status: rollout controls and E0 reporting checks passed;
E0 protocol is ready for a recorded local CPU run. No training has started.

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

1. Run `scripts/run_e0.py` locally and record its run ID. D015 freezes E0_v1:
   270 cases, both scene sets, 50 steps, epoch position 60, three seeds for the
   main profiles and seed 0 for six preliminary ablations. No optimizer updates.
2. Verify the resulting archive and retain all fields, case results and failures.
   Summarize both scene sets separately and compare ablations only to matched
   seed-0 controls. Do not mix three-seed and one-seed denominators.
3. Correct the legacy vertical-envelope dilation using a versioned operator.
   Compare it on the same frozen scenes; do not silently change the E0 baseline.
4. Repair and test loss semantics/gradients and create the corrected baseline.
   Then prepare Colab preflight and recovery, agree a concrete compute cap and
   backup plan, and only then ask the user to launch training.

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
