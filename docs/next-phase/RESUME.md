# Resume the NCA next phase

Last updated 2026-09-23. **K2 actual-loop recovery and the full local sensitivity
comparison are complete and verified.** Read SENSITIVITY_FINDINGS.md and D033.
144 tests pass;68 optimizer updates and187 evaluation cases are preserved.
Neither coefficient/model is promoted. Next prepare/profile the E2 direct-material
optimization control. No active process at handoff and no user setup required.

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
  and R1 runner. 2c356de L2/R1 evidence, f767975 T1 protocol/runner. The later T1 evidence commit
  contains this handoff; inspect Git log.
- All raw runs/source snapshots: .local-artifacts/runs/<run_id>/; small tracked
  summaries: experiments/records/<run_id>.json. Original 50-file snapshot remains
  in .local-artifacts/source-snapshots/. Reports do not replace raw artifacts.
- Archive outputs: C:/Users/artin/Documents/Codex/2026-09-06/cre/outputs/.
  Preserve M1, E0, corridor and loss archives. Earlier facade milestone archive:
  NCA-Facade-Backup-2026-09-23-<evidence-commit>.zip plus .zip.sha256.
  Verify .local-artifacts/milestones/<commit>-backup-receipt.json before claiming
  completion. Archives are local same-disk copies, not off-device backups.
- Earlier facade builder: C:/Users/artin/Documents/Codex/2026-09-06/cre/work/package_nca_facade.py.
  Requires completed A1/W1, creates a new full Git bundle and restore clone,
  checks frozen scene/annotation hashes and every ZIP payload hash. Never overwrite archives
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

## T1 completion and current decision

- T1 `20260923T082527Z_845d2aa6aec0`, source f767975:432 static targets,72 joint
  bounds,36 direct occupancy gradients. All18 scenes retained, no optimization.
- Regression `20260923T082336Z_b8c43fcf7723`:123 passed,no failures/errors/skips,
  checkpoint smoke exit0. Five added tests verify bound conflicts and candidates.
- Adding mandatory-facade bounds reduces radius6/envelope compatibility from
  17/17 feasible scenes to15/17 (legacy007/008 fail); radius3 reduces to11/17.
- Simple guides in ground-pair/minimal references have zero nine-family penalties
  under radius6/envelope despite one-voxel-wide segments. These are evidence of
  weak architectural success criteria, not production success.
- T1 full report and verification/rerender receipts: experiments/reports/T1-*.
  Recomputed saved norms/cosines, volumes/bounds and all nine zero-loss witness
  records; fresh render matches. No T1 attempt failed. No process remains active.
- User answered they are unsure about usable pavilion/bridge versus abstract
  material and asked which is more feasible from the prior plan. Recommendation:
  architectural material/form generation first, usability evaluated separately,
  usable pavilion/bridge as the longer-term goal. This is advice, not a user-approved
  final architectural specification. Retain material-connectivity access for the
  proposed next comparison; do not claim walkability or mechanical safety.

## A1/W1 completed: current state

- User said 'ok go on' after the material-generation recommendation. Proceed on
  that accepted working scope; do not ask the same representation question again.
  Usability remains a separate longer-term goal; no walkability/safety claims.
- A1 `20260923T084341Z_e37699e31f26`, source2d543de:864 target-arm,144 control-arm,
  144 bound-arm and72 gradient-arm records. Other eight terms unchanged; all18
  facade blankets penalized; radius6/envelope necessary compatibility17/17 feasible.
- `facade_endpoint_v1`: exact named facade entrance cells intersected with direct
  building-face neighbors and permitted space, no dilation, no target dependence.
  Sidecars/manifest in experiments/annotations/facade_endpoint_v1; LF is enforced
  for their hashed bytes. Reconstructed all18 masks independently in report checks.
- W1 `20260923T084933Z_eb2603cd79f7`, source2965502:17 connected zero-loss witnesses,
  one incompatible sealed reference. Ordered legal cell additions replayed; all
  nine terms and binary metrics exactly recomputed. Radius6/envelope unchanged.
- `budgeted_witness_v1`: simple deterministic procedural comparator, no learning.
  Some extra material satisfies a floor/dilutes contact; not architectural quality.
- Latest verification `20260923T084753Z_489a9c4be020`:132 tests,zero failures/errors/
  skips, checkpoint smoke0. Earlier facade-only verification
  `20260923T084059Z_6280ed836d9f`:129 passed. All attempts preserved; none failed.
- Reports and verification/rerender receipts: experiments/reports/A1-* and W1-*.
  All A1 saved gradients independently checked against quotient derivatives.
  No optimizer update in A1/W1, no active processes at this handoff.
- D028 selects radius6/envelope and facade_endpoint_v1 only for experimental
  calibration/baseline preparation. Preserve original facade as ablation and
  original production defaults. All nine families retained; prior physical-budget
  reduction is explicitly accepted for experimentation, not silently relabeled.

## Exact next actions

1. Read SENSITIVITY_FINDINGS.md/D033. Do not rerun completed K1/K2/K2R or restart
   the accepted material-generation scope question. No recipe has been promoted.
2. Prepare the missing E2 direct-material optimizer with identical nine-family,
   regularizer, envelope and facade semantics. Profile/verify a tiny local control
   on legacy008, ground-pair and minimal-smoke. Specify raw/material parameterization,
   initialization, legal projection, step size and recorded per-family gradients.
   Preserve W1 as an explicit procedural/initialization control, never NCA output.
3. Use measured runtime to preregister a bounded matched comparison on all17
   feasible development scenes. Record optimization steps/time per scene; a direct
   per-scene solve is not generalizing inference. Compare original and both K2
   models under common scoring and binary metrics. Keep failed cases.
4. Decide from that evidence whether objective/optimization work or a controlled
   NCA recovery/conditioning experiment comes next. Do not change architecture,
   schedule and geometry distribution together.17-update K2 is not convergence.
5. Freeze fresh geometry holdouts before their outputs. Existing18 scenes remain
   development data. Studio implementation, diversity and grid scaling are planned;
   the studio may use preserved fixtures with truthful material/metric labels.
6. Document, commit and verify a new local archive at each milestone. Paid Colab
   needs a concrete config/cap and GPU recovery. Every Drive action needs explicit
   permission. No agents without authorization; local work can continue directly.

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

## T1 verification commands

```powershell
& .venv/Scripts/python.exe scripts/experiment.py verify 20260923T082527Z_845d2aa6aec0
& .venv/Scripts/python.exe -c "from pathlib import Path; from scripts.report_target_audit import load,render; assert Path('experiments/reports/T1-target-audit.md').read_text(encoding='utf-8') == render(load())"
```

Do not rerun completed T1 without a new reason. If necessary, scripts/run_target_audit.py
accepts `--parent-run 20260923T082527Z_845d2aa6aec0` for a retained fresh attempt.
Report writers refuse overwrite; use in-memory rendering to reverify.

## A1/W1 verification commands

```powershell
& .venv/Scripts/python.exe scripts/experiment.py verify 20260923T084341Z_e37699e31f26
& .venv/Scripts/python.exe scripts/experiment.py verify 20260923T084933Z_eb2603cd79f7
& .venv/Scripts/python.exe -c "from pathlib import Path; from scripts.report_facade_comparison import load,render; assert Path('experiments/reports/A1-facade-comparison.md').read_text(encoding='utf-8') == render(load())"
& .venv/Scripts/python.exe -c "from pathlib import Path; from scripts.report_witnesses import load,render; assert Path('experiments/reports/W1-witnesses.md').read_text(encoding='utf-8') == render(load())"
```

Fresh repeats, only if justified, use scripts/run_facade_comparison.py or
scripts/run_witnesses.py with --parent-run and the relevant previous run ID.
They preserve prior evidence; they do not resume an unfinished case in place.

## K1/R2 completed evidence and current restore commands

- K1 `20260923T092355Z_f657f2f3bdb9`, sourcef56932c:71 model-gradient cases,
  51 budget probes, zero optimizer updates, all independently verified. Recomputed
  composed objective exactly on all71 saved fields. It took1631.61s including one
  unexplained410.99s outlier; do not rerun for a timing benchmark. Logged scalar
  warning was harmless and a later detach-only logging fix is documented.
- R2 `20260923T095240Z_652e01d22fee`, source0fca1b8:four logical/ten executed
  updates across four fresh processes, three scenes, all seven exact recovery
  checks. Independent full checkpoint/trace/array checks pass. CPU orderly
  completed-update boundary only; no CUDA/AMP/pool/abrupt-write or K2-loop claim.
- Verification137tests:20260923T092058Z_bba721fc054d; latest141tests:
  20260923T092752Z_fa771e070c3d,zero failures/errors/skips,checkpoint smoke0.
- Intermediate source commits: f56932c regularizers/K1,82b5485 objective/recovery
  preparation,0fca1b8 K1evidence/K2proposal. Final evidence commit follows; inspect
  Git log and matching backup receipt. All prior milestone history is preserved.
- Main finding: at16steps historical numeric coefficients on corrected formulas
  give coverage-improving negative-raw-gradient directions in2/34 cases versus
  18/34 with only sparsity30->3. This is NOT predicted Adam learning. Ground-pair
  projected access remains blocked; thickness inactive; neither recipe finalized.
- Reports: experiments/reports/K1-calibration.md,R2-recovery.md and verification/
  rerender receipts. K2 config and directional estimates are tracked, not outcomes.
- Earlier calibration archive: outputs/NCA-Calibration-Backup-2026-09-23-<commit>.zip
  in the Codex cwd above, plus .zip.sha256; receipt in repo.local-artifacts/milestones.
  Builder: C:/Users/artin/Documents/Codex/2026-09-06/cre/work/package_nca_calibration.py
  with R2 run ID as argument. It verifies all ZIP payloads, a fresh Git restore,
  scenes/annotations/config hashes and registered R2 exact-source snapshot bytes.
  Preserve earlier backups. Same-disk archive only, not off-device protection.
- A restored Git checkout may normalize Python line endings and fail exact code
  hashes. Use the registered source_snapshot ZIP from the relevant run, extracted
  into a NEW workspace, for exact source-byte recovery. Never replace historical
  artifacts or silently bypass metadata checks. Environment versions also must match.

```powershell
& .venv/Scripts/python.exe scripts/experiment.py verify 20260923T092355Z_f657f2f3bdb9
& .venv/Scripts/python.exe scripts/experiment.py verify 20260923T095240Z_652e01d22fee
```

Report writers refuse overwrite. For revalidation use load/render or verify/render
in memory, comparing with the saved reports. Only repeat an experiment for a new
reason, with a linked new run ID. Inspect processes and partial result records
before any retry. K1 and R2 are complete; neither should be restarted now.

## K2/K2R completed evidence and current restore commands

- K2R20260923T102414Z_eaec7bd1510e, source33f6858:three logical/eight executed
  actual16-step loop updates, all seven exact checkpoint/trace/field checks pass.
  Constant learning rate and complete17-scene order are checkpoint metadata.
- K2 20260923T102524Z_f5e1cc169dea, source33f6858:four17-update members,68 total,
  187 matched evaluations, no failed/timed-out process. Runtime505.98s including
  overhead. All four members initialized from the original checkpoint.
- Latest regression20260923T102104Z_15b3cd2f4fd5:144 tests,zero failures/errors/
  skips,checkpoint smoke0. No hashed training code changed after this pass/gate.
- All68 checkpoint boundaries verified; all255 saved training/evaluation fields
  rescored, all187 binary metrics/common totals match. Snapshot hashes verified.
  Report fresh rerender matches. Reports and verification/aggregate/transition
  receipts: experiments/reports/K2-* and K2R-*.
- Weight3 improves coverage versus30, but at50steps11-12/17 connected and17/17
  over budget. Weight30:10/17 connected,15/17 over budget. Original:10/17 and17/17.
  W1 static:17/17 and0/17. All five feasible reference scenes remain disconnected
  for every recurrent model. No checkpoint or coefficient is selected for promotion.
- K2-sensitivity.json remains the historical frozen proposal (its prepared_not_run
  label is not current execution status). Use immutable new run records for outcomes.
- Latest full archive pattern: outputs/NCA-Sensitivity-Backup-2026-09-23-<commit>.zip
  plus.zip.sha256 in the Codex cwd. Receipt:.local-artifacts/milestones/<commit>-
  backup-receipt.json. Builder:work/package_nca_sensitivity.py in that cwd, arguments
  K2R run ID then K2 run ID. It preserves prior artifacts and Git history, verifies
  every payload hash, fresh Git clone, frozen scenes/annotations/config and exact
  K2R source snapshot bytes. Includes its own builder. Same-disk local archive only.
- Core recovery is CPU completed-update/ordinary-exit only. No timeout occurred.
  Linked interrupted-run imports (--resume-run) are implemented but not exercised
  by this successful study; abrupt/mid-write failure, CUDA and Colab uncertified.
  Never rerun a completed study via resume; never bypass a failed integrity check.

```powershell
& .venv/Scripts/python.exe scripts/experiment.py verify 20260923T102414Z_eaec7bd1510e
& .venv/Scripts/python.exe scripts/experiment.py verify 20260923T102524Z_f5e1cc169dea
```

scripts/report_sensitivity.py refuses overwriting published results. Recheck with
verify(run) and render(run,protocol,training,evaluation) in memory. Preserve all
older reports; repeat experiments only for a concrete new reason with a linked
new run ID. Exact source snapshots are required if Git changed Python line endings.
