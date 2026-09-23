# Resume the NCA next phase

Last updated 2026-09-23. **L2 intervention comparisons and R1 exact CPU recovery
are complete.** Read INTERVENTION_FINDINGS.md, D021/D022 and PLAN.md. Next audit
all-nine-term target compatibility and physical budgets before calibrating a
corrected research baseline. No paid training or production change is selected.

## Authorization and storage

The user approved local implementation, preservation of every decision/result,
the nine existing constraint families, and local archives. The user declined
Drive upload (D014). AGENTS.md requires explicit approval BEFORE EVERY Drive
operation including reads, and only within folder
1fS34Yy0-oMzSxWaYJFiPTkGgrZstgc0H and actual descendants. Do not access Drive.

Paid Colab is available; no job or GPU-hour cap has been approved. Private
NCA-Next-Phase-Report files remain Git-ignored. Preserve the user's untracked
NCA-Studio-Concept.html and NCA-M1-Backup-2026-09-23-1dfafa7.zip.sha256.
Local commits are authorized. No remote push, deployment or cloud operation.

## Checkout and durable evidence

- Repo: C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA.
- Branch: next-phase/foundations. Original baseline: ac913b9.
- Milestones: b841991 foundations, ddc8100 scene contract, 1dfafa7 historical
  profiles, 474bf53 rollout/E0 runner, 75402ff E0 evidence, 579031c corridor
  operators, 595c8e0 retry prep, 8501478 C1 evidence, fa66238 shared loss package,
  2e1e510 L1 evidence, 95431de interventions/recovery utilities, 6ee2ed8 L2 evidence
  and R1 runner. The next evidence commit contains this handoff; inspect Git log.
- All raw runs/source snapshots: .local-artifacts/runs/<run_id>/; small tracked
  summaries: experiments/records/<run_id>.json. Original 50-file snapshot remains
  in .local-artifacts/source-snapshots/. Reports do not replace raw artifacts.
- Archive outputs: C:/Users/artin/Documents/Codex/2026-09-06/cre/outputs/.
  Preserve M1, E0, corridor and loss archives. New milestone archive:
  NCA-Intervention-Backup-2026-09-23-<evidence-commit>.zip plus .zip.sha256.
  Verify .local-artifacts/milestones/<commit>-backup-receipt.json before claiming
  completion. Archives are local same-disk copies, not off-device backups.
- Builder: C:/Users/artin/Documents/Codex/2026-09-06/cre/work/package_nca_interventions.py.
  Requires completed L2/R1, creates a new full Git bundle and restore clone,
  checks frozen scene hashes and every ZIP payload hash. Never overwrite archives
  or remove failed/partial evidence to rerun. Use a new attempt name if needed.

## Completed evidence: do not rerun without a reason

- Frozen scene_v1: six reference and 12 legacy scenes with raw/canonical hashes.
  Legacy inputs MUST use legacy_seed_state; deployed generation differs at facade
  anchors below street. Production defaults and original checkpoint remain intact.
- rollout_v2: explicit RNG/fire controls and original notebook-forward parity.
  E0 20260922T230120Z_76f3b4677e8f:270 cases. E0_FINDINGS.md.
- corridor_bounded_v1 and corridor_legal_v1: bounded growth and separately versioned
  legal six-neighbor routing. C1 20260923T000945Z_3cbdc3603a12:54 targets/108 cases.
  CORRIDOR_FINDINGS.md. Failed long-path attempt 20260923T000715Z_1e06516e9763
  retained; short filenames fixed publication, not model behavior.
- geometry_losses_v1: nine shared terms, strict context and batch validation.
  L1 20260923T003413Z_1da1202e4a7f:72 contexts, three model gradients and six
  historical defect checks. LOSS_FINDINGS.md. Post-hoc clamp attribution
  20260923T003906Z_da05c8eed3f6 exactly replays the dead ground derivative.
- L2 20260923T075113Z_d95fabaf3776:108 budget,54 gradient,nine zero-scaffold
  cases; no optimizer. All 21 hard pairs match. Pre-clamp coverage derivatives
  reach the two failed cells; projected access is still blocked in that case.
  Smooth state introduces background mass without binary connectivity gains.
  Radius6/envelope necessary-valid17/18 versus site12/18, but only2.04%-10.65%
  of the site's physical allowance. Radius3/envelope also passes17/18.
- R1 20260923T075727Z_2233c0e51b9a:four logical updates, ten executed across
  four processes; all seven exact recovery checks pass. Full checkpoint, RNG,
  losses and fields match. Nine unit coefficients are mechanics-only. Random
  sampling covered two of the three available scenes; no legacy optimization.
- Verification 20260923T074748Z_68688d661ffd:118 tests,zero failures/errors/skips,
  checkpoint smoke exit0. Linked failed fixture attempt20260923T074523Z_f2efaaf7f135
  retained. Core code unchanged after pass; R1 exercises its separate-process path.
- experiments/reports/L2-R1-interventions.md is reconstructed from registered,
  hash-verified artifacts. Saved gradient norms, hard-pair/frozen fields, recovery
  checkpoint trees/traces/arrays independently rechecked; fresh render equals file.
  Failed first report publication (missing directory) preserved under
  .local-artifacts/analysis-attempts/20260923-l2-r1-report-01; fixed then published.

## Exact next actions

1. Freeze a target compatibility audit across all 18 scenes using legal guide,
   thick scaffold and explicit volumetric candidates. Evaluate all nine terms,
   physical amounts, independent binary metrics and architecture interpretations.
   Necessary capacity bounds alone do not prove joint objective feasibility.
2. Resolve denominator/radius and intended material thickness deliberately. Keep
   site/envelope alternatives explicit; no silent mass reduction or wider regions.
   Pre-clamp coverage is a candidate; it does not repair access semantics by itself.
3. Measure per-family magnitudes and gradient directions, integrate retained
   regularizers and calibrate visible fixed coefficients on deterministic scene
   coverage. Preregister E2 procedural scaffold/direct optimization/NCA controls,
   longer horizons, damage/absent scaffolds and held-out evaluation.
4. Extend checkpoint state to the actual research trainer, including sample pool,
   CUDA RNG and AMP scaler if introduced. Test interruption on that environment.
   R1 only certifies completed-update CPU boundaries after orderly process exit.
   Copy fallback on filesystems without hard links is not atomic under power loss.
5. Prepare a concrete small Colab pilot only after the above; ask for its compute
   cap and any exact Drive operations before use. Bigger grids and the studio
   redesign remain planned. No user setup is required for the next local audit.

## Commands and interruption recovery

From the repo root in PowerShell:

```powershell
& .venv/Scripts/python.exe scripts/experiment.py verify 20260923T075113Z_d95fabaf3776
& .venv/Scripts/python.exe scripts/experiment.py verify 20260923T075727Z_2233c0e51b9a
```

Report writers use exclusive creation. To reverify without overwriting:

```powershell
& .venv/Scripts/python.exe -c "from pathlib import Path; from scripts.report_interventions import render; assert Path('experiments/reports/L2-R1-interventions.md').read_text(encoding='utf-8') == render()"
```

Run scripts/verify_foundation.py only when code changes warrant repeating the
suite; it archives a fresh result. Before any retry inspect processes, result.json
and artifacts. Missing finalization means incomplete, not success. L2 repeats use
`--parent-run 20260923T075113Z_d95fabaf3776`; R1 uses
`--gate-run 20260923T075113Z_d95fabaf3776` and
`--parent-run 20260923T075727Z_2233c0e51b9a`. These create new complete attempts and preserve
old runs; they are not automatic case-skipping resume commands. R1 worker-level
continuation is orchestrated by its coordinator; do not manually overwrite branches.

Python3.12.14, CPU torch2.8.0, NumPy2.5.2; requirements-cpu.lock.txt. Do not install
that CPU lock over Colab CUDA. Local write/Git permissions may need renewal in a
new task. Previous handoffs remain in Git/archives. This record supports resuming
work after a limit reset; it does not automatically resume work or redeem credits.
