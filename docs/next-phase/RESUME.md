# Resume the NCA next phase

## Current F4 preparation - 2026-09-23

Read RAW_ACCESS_TRAINING_PROTOCOL.md/D049. Source implemented, no F4 run yet.
Next full regression, source commit, then scripts/run_raw_access_training.py
--mode parity. Verify with scripts/report_raw_access_training.py <ID>; next
--mode recovery --parity-run <B>; then --mode pilot --parity-run <B>
--recovery-run <R>. Only after verified timing admission use --mode study
--parity-run <B> --recovery-run <R> --pilot-run <P>. Finally verify full and
scripts/check_raw_access_late_recovery.py <full-ID>. Use project .venv Python.
No paid compute/Drive/push. Previous archive878d5f2 receipt exists and verified.

## Previous completed A3 diagnostic - 2026-09-23

Read RAW_ACCESS_FINDINGS.md/D048, then RAW_ACCESS_TRAINING_PLAN.md. A3 full
20260923T193403Z_ac42a8cca390 completed336.74s:231 fields/eight actual F3 gradient
cases/12 reversible probe fields, zero optimizer updates, all caps met. Full
verifier passed35 source hashes,48 parameter vectors/48 last-raw arrays and
all saved-field/trace checks. No active diagnostic/training process.

Pilot20260923T193037Z_0e72e1207fb9 completed111.09s and verified25 fields/two
gradient cases/two probes. Timing1038.85s admitted1500 cap. Source08a6d85.
Initial regression20260923T192815Z_f7cdbff9cc00 passed180 tests/smoke0.

Outcome: old access parameter gradients0/8 nonzero; candidate6/8 nonzero.
All six descent probes reduce candidate access, but repair no binary connection.
Total improves3/worsens3; all four50-step gradients conflict with sparsity.
Two16-step ground-pair cases remain blocked behind earlier clamps/non-firing.
Candidate remains opt-in. Next prepare F4 against constant16 F2, one access
change only, original initialization/scenes/recipes/64 updates; actual-loop
parity, early/trained-state recovery and timing gates before any full training.

Reports experiments/reports/A3-<full-ID>-{verification,evidence,outcomes}.json
and corresponding A3P pilot reports. Raw artifacts .local-artifacts/runs/<ID>.
Post-hoc script/receipt .local-artifacts/analysis-attempts/A3-<full-ID>; failed
first float32 consistency tolerance retained in A3-summary-attempt-1. No
scientific gate/outcome was altered.

After audit, fixed infeasible fallback overflow in nca/raw_access.py only; no
audited case affected. All37 historical F3 files still match. Current diagnostic
source identity differs from the recorded35-file source snapshot in that one
file, so exact repetitions require extracting the registered source ZIP into a
NEW workspace. Do not bypass pilot source equality or historical recovery guards.
Follow-up full regression20260923T194258Z_d2f3798c93b4 passed181 tests, zero
failures/errors/skips, original-checkpoint smoke0,80.57s. No active processes.

Final archive builder in Codex cwd work/package_nca_raw_access.py takes pilot
and full IDs above. Output outputs/NCA-Raw-Access-Backup-2026-09-23-<commit>.zip
and SHA256 sidecar; completion receipt .local-artifacts/milestones/
<commit>-backup-receipt.json. Absence means archive remains to be completed.
Includes all previous evidence/private reports, source snapshots, analysis
attempts, full Git bundle, fresh restore and payload-hash checks. Same-disk copy;
no off-device backup yet. Every Drive operation still needs explicit permission.
No paid compute, remote push, production change or promoted model.

## Previous A3 preparation - 2026-09-23

Read RAW_ACCESS_AUDIT_PROTOCOL.md and D047. Source/config implemented; no pilot
or full audit yet. Next: full foundation regression, local source commit, then
.venv/Scripts/python.exe scripts/run_raw_access_audit.py --mode pilot
Verify with scripts/report_raw_access_audit.py <pilot-ID>. Only after successful
verification and timing admission execute --mode study --pilot-run <pilot-ID>.
No optimizer updates, paid compute, Drive access or remote push. Preserve all
prior outcomes. Prior F3 archive c5183a9 receipt verified at start of this task.

## Previous completed F3 milestone - 2026-09-23

Read HORIZON_TRAINING_FINDINGS.md and D046, then ACCESS_RECOVERY_PLAN.md.
F3 full20260923T160713Z_cc33850561b8 completed1722.77s,256 updates and120 unique
evaluations. All376 fields rescored,260 checkpoint/schedule cursors verified,
37 source hashes checked, eight initial fields/eight final rollouts exact.
No active training/diagnostic process. All elapsed caps met; sourcec0a44b1.

Result: mass decreases in72/72 matched final cases, but all59 F2 connections are
lost, none gained. F3 has0 connected or jointly successful final cases. All12
16-step final cases are in budget; all60 longer ones exceed it. Candidate access
loss1 in72/72; saved critical raw voxel negative in51, exactly0 in21. This is
saved-field/last-clamp evidence, not a new full parameter-gradient experiment.
Do not adopt F3 or scale it. Next: actual failed-state gradient audit, then frozen-
field evaluation of a pre-clamp maximin access-loss extension. Freeze protocol,
cases and allowance first. No further training approved by scientific readiness.

Regression20260923T155421Z_8e4d3bfc6919 passed172 tests/zero errors/failures/skips,
smoke0. F3B20260923T155551Z_dc2ea609facf exact12-update/24-evaluation F2 parity;
F3R20260923T155836Z_10e9dbf37b62 exact11 restart checks plus five intermediate
checkpoint comparisons. F3P20260923T160209Z_e292a696d594 verified96 unique fields,
3246.25s estimate admitted3600s. F3L20260923T184328Z_418e31921193 replays all four
models' updates63/64 and eight evaluations exactly in118.04s;38 hashes/eight
cursors checked. Late wrapper added separately; frozen37 scientific hashes unchanged.

Reports experiments/reports/F3*-{evidence,summary,verification}.json and
F3-horizon-training.md; F3-outcomes.json, F3-critical-cells.json and F3-horizons.png
with figure receipts. Post-hoc scripts/receipts under analysis-attempts. Commands
already completed: scripts/report_horizon_training.py <full-ID>,
scripts/check_horizon_late_recovery.py <full-ID>. Do not rerun for cleaner records.
If recovering another interrupted attempt, inspect processes/result/logs first;
retain its status and use linked new IDs with exact source ZIPs and metadata.

Final local archive: Codex cwd outputs/NCA-Horizon-Training-Backup-2026-09-23-
<results-commit>.zip plus .zip.sha256; authoritative completion receipt at
.local-artifacts/milestones/<results-commit>-backup-receipt.json. Verify receipt;
absence means archival work remains. Builder work/package_nca_horizon.py takes
B,R,P,full IDs above in that order and includes every run, including F3L, and
prior evidence/private reports. Full Git bundle/fresh restore and payload hashes
checked by builder. Same-disk copy, not off-device backup. No Drive access.

The initial F3L escalation timed out before process creation (tool waiting6632.2s),
then a normal authorized local retry succeeded. This did not affect completed
F3 timing. Plot-library access failures resolved with scoped elevated execution.
No paid compute, push, production/default change or promoted model.

## Previous completed H1 milestone - 2026-09-23

H1 full `20260923T150119Z_f7b304516723` completed in 603.78 seconds, all elapsed
caps met; source `3acdefd`. No active training/diagnostic process. Zero optimizer
updates. All 180 fields rescored, 150 transitions checked, 20 historical anchors
exact, 20 gradient forward fields exact, 120 parameter/120 raw vectors and 720
cosines verified, 31 source hashes and 10 unchanged model-weight checks pass.
Eight new F2 gradient cases and 12 verified A2 controls; reused is not rerun.
Pilot `20260923T145912Z_17b1d18f5b10` completed in 73.50s; timing estimate 989.49s
admitted the 1500s cap. Regression `20260923T145719Z_37715b17ba45`: 168 tests,
zero failures/errors/skips, original-checkpoint smoke0. No code changes afterward.

Read GROWTH_AUDIT_FINDINGS.md and D044, then HORIZON_TRAINING_PLAN.md. No joint
connectivity/budget success in 180 fields; all 144 fitted-model fields over budget.
F2 connects all tested seeds from40steps onward; no sampled connection loss.
Long-horizon total gradients align with material reduction, motivating an isolated
alternating16/50 training comparison. This is a proposed F3, not an executed run.
Next: implement fixed schedule, prove constant16 F2 parity, verify actual mixed
16/50/16 restart, profile/freeze caps, then run only if the admission gate passes.
No paid training, model promotion or new architecture/constraint/budget change.

Reports: experiments/reports/H1{,P}-growth.md, corresponding evidence/summary/
verification JSON, H1-outcomes.json and H1-growth.png with figure receipts.
Post-hoc scripts and receipts are under .local-artifacts/analysis-attempts/.
Verifier command (already completed; outputs refuse overwrite):
`.venv/Scripts/python.exe scripts/report_growth_audit.py 20260923T150119Z_f7b304516723`.
Do not rerun completed diagnostics to obtain cleaner records. Inspect processes,
result.json and logs before recovery; failed/new attempts retain linked identities.

Archive builder in Codex cwd: `work/package_nca_growth.py
20260923T145912Z_17b1d18f5b10 20260923T150119Z_f7b304516723`, using project Python.
After the results commit, expect outputs/NCA-Growth-Backup-2026-09-23-<commit>.zip
and .zip.sha256 in the Codex cwd, with .local-artifacts/milestones/<commit>-
backup-receipt.json. Check that receipt for verified completion; absence means
archiving remains pending. Full Git bundle, all prior raw runs, analyses, private
reports and primer are retained. These are same-disk copies. No Drive access.

## Previous completed F2 milestone

Last updated 2026-09-23. **F2 learning comparison, verification and recovery complete.**
Read ACCESS_TRAINING_FINDINGS.md/D041-D042 then GROWTH_STABILITY_PLAN.md.
Final connectivity improves0->2/4 at16steps,3->4/4 at50, but no joint budget success.
Parent20260923T135818Z_5520d5d80cec remains interrupted254/54 with a timing
violation. Child20260923T143002Z_55aeaac95580 imported254/54 and executed remaining
2/2. Complete256/56 matrix rescored; eight final rollouts exact. Cumulative1820.10s
is not cap compliant. F2L20260923T143228Z_73318ff99c0b exactly repeats all four
final updates/eight evaluations. No active training process. No model promoted.
Elapsed-time guard repaired AFTER evidence/recovery; regression
20260923T143444Z_3c97b03e720e passes164 tests, zero failures/errors/skips, smoke0.
Historical32-file source differs now only in scripts/run_access_training.py;
use original source snapshots for optimizer/forward replay with strict metadata.
Next work is the bounded growth diagnostic. Verify the milestone archive receipt
described below; a missing receipt means archival verification still needs completion.
No Drive access, paid compute, production change or remote push.

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

1. Read DIRECT_FINDINGS.md/D035. Do not rerun K1/K2/D1 or reask the accepted
   material-generation scope. Direct reference connectivity is now demonstrated,
   but joint objective/architectural success and NCA fitting remain unresolved.
2. Profile/preregister repeated single-scene NCA fitting on ground-pair and
   minimal-smoke. Keep checkpoint/architecture, objectives, weak initial scaffold
   and both weight recipes fixed. Proposed64 updates per scene/recipe is a candidate,
   not an already executed/approved paid schedule. Use measured timing to freeze
   a bounded CPU cap and intermediate evaluation boundaries before outcomes.
3. Verify actual-loop checkpoint recovery for repeated exposure. Save every update,
   firing RNG, scene position and source hashes. Evaluate at16/50 growth steps and
   compare original/D1/W1 controls. No solved-D1 initialization or new reconstruction
   family. This is fitting capacity, not unseen-geometry or convergence proof.
4. If it fits, isolate mixed-scene scheduling/recovery then fresh holdout evaluation.
   If it does not, inspect saturation/gradients before one conditioning/perception
   change. Do not alter architecture, objectives and distribution together.
5. Viewer:assets/experiment_viewer.html and scripts/build_result_viewer.py. Existing
   generated artifact has verified221 fields/metrics and JS syntax, but browser URL
   policy BLOCKED local-file preview. Do not claim visual/interaction checks passed,
   or bypass that browser-policy restriction. User can inspect the saved artifact.
   Live studio jobs/cancellation/editing/deployment remain separate planned work.
6. Update records, commit and verify a local archive. Paid Colab requires concrete
   config/cap and GPU recovery; EVERY Drive action needs specific approval. Local
   work needs no new scope confirmation. No agents without explicit authorization.

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
- Earlier K2 archive: outputs/NCA-Sensitivity-Backup-2026-09-23-<commit>.zip
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

## D1 completed evidence and current restore commands

- Source042b7c8 adds exact raw-field optimization and fixed D1 protocol. Latest
  regression20260923T104700Z_c4dc9a0d6f64:147 tests,zero failures/errors/skips,smoke0.
  No direct source/config change after the pass/recovery/pilot/full execution.
- D1R20260923T105015Z_484cf36351d7:four logical/ten executed updates,ground-pair,
  mass_3,seven exact full-checkpoint/trace/field-gradient comparisons.19.21s.
- D1P20260923T105120Z_4d1e4d2d20d6:six cases x8 updates=48,39.84s. p90 update
  0.33365s,5s startup allowance,conservative full estimate799.51s<900s cap. Timing,
  not quality, admitted the unchanged32-update/34-case full matrix.
- D1 20260923T105246Z_ab0d4a430b4c:34 cases/1088 updates,484.44s,zero failures or
  timeouts. All1088 checkpoints/projections/norms checked; all68 initial/final
  scored states rescored. Intermediate objectives preserved but not all rescored.
- Both direct recipes connect17/17 from initial0/17 binary-connected weak scaffolds.
  Weight30 has12/17 in-budget,zero over-cap andfive under-floor legacy cases
  (003,006,009,010,011). Weight3 has10/17 in-budget,one under-floor(000),six over-cap
  (003,006,010,011,wide-gap,asymmetric-heights). All five references connect; weight30
  keeps all five in budget. Continuous residuals remain, no architectural certification.
- D1 reports, aggregates, per-case budget failures and verification/rerender receipts:
  experiments/reports/D1-*,D1P-*,D1R-*. Scientific fields/checkpoints/source snapshots
  stay in .local-artifacts/runs/<run_id>. No scientific attempt failed.
- One report-rerender check failed from receipt dictionary order, preserved in
  .local-artifacts/analysis-attempts/D1-viewer-qa-20260923/rerender-attempt-1.json.
  Fresh verifier output rerenders all three reports exactly; no report overwritten.
- Viewer:.local-artifacts/viewers/D1-20260923T105246Z_ab0d4a430b4c/index.html.
  17 scenes/13 variants/221 fields. All exact binary coordinates/metrics verified;
  JS syntax passes. Browser file-URL policy blocked preview; visual/interaction QA
  remains unverified. Manifest and QA in experiments/reports/D1-viewer-*. Original
  user NCA-Studio-Concept.html untouched. Viewer is a local evidence preview only.
- Latest archive:outputs/NCA-Direct-Backup-2026-09-23-<commit>.zip plus.sha256 in
  Codex cwd, receipt:.local-artifacts/milestones/<commit>-backup-receipt.json.
  Builder:work/package_nca_direct.py in Codex cwd; args:D1R ID,D1P ID,D1 full ID.
  Includes all prior runs/history plus viewer/QA, private reports and primer.
  Checks every payload hash,fresh Git clone,scenes/annotations,D1 config and exact
  D1R source snapshot hashes. Same-disk local copy; no off-device backup claimed.
- Recovery boundary is ordinary completed CPU updates. Resume into a NEW run/branch
  with identical source/runtime/scene metadata and a verified checkpoint. Runner
  supports worker --resume, not an automatic full-matrix continuation coordinator.
  Inspect completed update records before retry; never overwrite or bypass integrity.
  CUDA,AMP,abrupt/mid-write timeout recovery remain uncertified.

```powershell
& .venv/Scripts/python.exe scripts/experiment.py verify 20260923T105015Z_484cf36351d7
& .venv/Scripts/python.exe scripts/experiment.py verify 20260923T105120Z_4d1e4d2d20d6
& .venv/Scripts/python.exe scripts/experiment.py verify 20260923T105246Z_ab0d4a430b4c
```

Report writers refuse overwrites. For fresh rerender use scripts.report_direct.verify
then render with that returned verification object (not a key-reordered receipt).
The D1 config remains frozen. To reproduce older K2 recovery, restore its exact source
snapshot: adding a new nca module changes that runner's broad code hash manifest.
No repeating completed experiments without a new reason and linked run ID.


## F1 completed evidence and restore instructions

- Source1c61b33: F1 fixed config/runner and unchanged inherited K2 training step.
  Regression20260923T112028Z_f4b86bdcf231:151 tests, zero failures/errors/skips,
  checkpoint smoke0. No training source/config changes after this pass.
- F1R20260923T112316Z_f34a421f8302:three logical/eight executed updates,14
  evaluations;11 exact CPU recovery comparisons.22 saved fields rescored.71.03s.
- F1P20260923T112500Z_8e1a30c320b8:eight updates/24 evaluations,82.25s; all32
  fields rescored. Estimate1546.47s admitted unchanged1800s total/600s member caps.
- F1 20260923T112646Z_dcd0601d0655:256 updates/56 evaluations,860.27s, zero
  failures/timeouts. All312 fields rescored,256 checkpoints checked,56 metrics
  verified, eight baseline fields and eight final checkpoint rollouts match exactly.
- Reports experiments/reports/F1-*,F1P-*,F1R-*; raw runs/source snapshots remain
  .local-artifacts/runs/<run_id>. F1-outcomes.json is post-hoc descriptive evidence;
  its exact script/result are in .local-artifacts/analysis-attempts/F1-summary-<run>.
  All56 fixed access sources empty, access1; three final regions nevertheless
  connect. Raw source negative28/zero28; gradient causation remains unmeasured.
- Full final50-step mass ratios: mapped30 ground.201082/minimal.254653; mass3
  ground.287250/minimal.305969. Only mapped30 ground remains disconnected. All
  final16-step cases disconnected and over budget. Preserve every earlier boundary.
- Backup builder in Codex cwd:work/package_nca_fitting.py. Args F1R ID,F1P ID,
  F1 study ID. Archive outputs/NCA-Fitting-Backup-2026-09-23-<commit>.zip plus
  .sha256; receipt .local-artifacts/milestones/<commit>-backup-receipt.json.
  Verify receipt before claiming completion. Includes all prior evidence, exact
  snapshots, private reports/primer and Git history. Same-disk only, not off-device.
- Recovery supports completed CPU updates into a NEW run/branch with identical
  metadata, not an automatic whole-study resume coordinator. Before retry inspect
  result.json, logs, complete record/checkpoint boundaries and live processes. Never
  rerun completed work or bypass hashes. CUDA/AMP/abrupt writes remain uncertified.
  To recover historical runs use exact source ZIPs; adding nca/fitting.py changes
  older runners' broad source inventories. Git may normalize Python line endings.

```powershell
& .venv/Scripts/python.exe scripts/experiment.py verify 20260923T112316Z_f34a421f8302
& .venv/Scripts/python.exe scripts/experiment.py verify 20260923T112500Z_8e1a30c320b8
& .venv/Scripts/python.exe scripts/experiment.py verify 20260923T112646Z_dcd0601d0655
```

Report writers refuse overwrites. Current report_fitting.verify can recheck results
in memory. It adds an initial-baseline equality check beyond the immutable F1R/P
publication version. Do not overwrite those reports to add later checks. F1 viewer
not created; previous D1 viewer unchanged, with visual browser QA still blocked.
Next task: ACCESS_ALIGNMENT_PLAN.md; no user Colab action needed yet.


## A2 completed evidence and continuation

- Sourcefac46ff implements opt-in nca/access.py, independent binary BFS, seven
  semantic/derivative tests, fixed A2 config and capped audit runner. No legacy
  objective/evaluator or deployment defaults changed.
- Regression20260923T123338Z_6b51ce24b725:158 tests, zero failures/errors/skips,
  original-checkpoint smoke0. No access/runner/config change after this pass.
- A2 20260923T123641Z_98bf30045a6f:277 replays,12 gradient cases,397.78s, no
  failures/timeouts, zero optimizer updates. Caps600 replay/120 per gradient/900
  total. Every worker completed. Reports experiments/reports/A2-*.
- Verification recomputed277 candidate values and independent binary results;
  checked72 parameter vectors,72 last-raw derivatives and432 cosines;12 saved
  raw/material forwards match.27 source hashes verified. Explicit original model
  config/checkpoint digest matches F1 in A2-provenance-crosscheck.json.
- Access v1 parameter norms zero12/12. Candidate nonzero4/12: all four fitted
  16-step cases, norms7.37-18.69 and positive coverage cosines0.712-0.803. Original
  four cases remain zero; coverage remains nonzero. Final50-step candidate norms
  zero: three already-connected/zero-loss, one disconnected. Do not claim all
  dead gradients fixed, use raw-field gradients as parameter gradients, or infer
  Adam outcomes. Two full objective cosine changes are negative.
- Binary labels match277/277. Source semantics lower8 F1 losses; worst reduction
  raises30 K2/control losses. No measured hop-removal benefit. All17 W1 access
  losses remain zero and all34 D1 access values unchanged. Saved geometry unchanged.
- Next ACCESS_TRAINING_PLAN.md proposes matching F1 with an access-only version.
  Before running: exact baseline parity, actual-loop completed-update recovery,
  profiled timing gate and frozen caps. No new training run has started.
- Candidate uses detached CPU topology selection plus critical-voxel gathering;
  not smooth or GPU-ready. Tie behavior is explicit. Do not simply seed multiple
  disconnected source pieces. Keep nine families and existing budgets.
- Backup builder in Codex cwd:work/package_nca_access.py <A2-run-id>. Outputs
  outputs/NCA-Access-Backup-2026-09-23-<commit>.zip and.sha256; receipt in
  .local-artifacts/milestones/<commit>-backup-receipt.json. Verify receipt before
  claiming completion. Includes all prior local evidence and source snapshots.
  Same-disk archive only, no off-device backup claimed.
- Adding nca/access.py changes historical broad code-hash manifests. Resume old
  optimizers from their exact source snapshots; do not bypass metadata checks.
  A2 loading frozen weights for diagnostics is not checkpoint resume.

```powershell
& .venv/Scripts/python.exe scripts/experiment.py verify 20260923T123641Z_98bf30045a6f
```

Report writers refuse overwrite. report_access_audit.verify(run) rechecks evidence
in memory; render can be used without publishing over existing artifacts. Before
any interrupted-run retry inspect result/status, logs and processes, retain all
partial evidence and use a new linked attempt. There is no automatic A2 partial
coordinator resume. No need to repeat this completed audit without a new reason.

## F2 current execution instructions

Regression20260923T134911Z_c3c1cd97dbdb completed:163 passed, smoke0.
Commit prepared implementation, then use .venv/Scripts/python.exe:
- scripts/run_access_training.py --mode parity
- scripts/report_access_training.py <parity-run>
- scripts/run_access_training.py --mode recovery --parity-run <parity-run>
- scripts/report_access_training.py <recovery-run>
- scripts/run_access_training.py --mode pilot --parity-run <parity-run> --recovery-run <recovery-run>
- scripts/report_access_training.py <pilot-run>
- If admitted: scripts/run_access_training.py --mode study --parity-run <parity-run> --recovery-run <recovery-run> --pilot-run <pilot-run>
- scripts/report_access_training.py <study-run>
Inspect immutable result.json before retry. New attempts use --parent-run; never
overwrite outputs or run a completed matrix again. Report writers are exclusive.
Source snapshot bytes, not a potentially normalized Git checkout, govern resume.

F2 source commit5e57bcc. Parity run20260923T135153Z_388dadf703b2 is active;
inspect result.json/processes before any retry. Regression passed163 tests.

F2B20260923T135153Z_388dadf703b2 completed114.65s:12 updates/24 evaluations
exactly match F1 including full checkpoint state;36 fields rescored,56 controls
rescored under both definitions,32 source hashes and8 original fields verified.
F2R20260923T135424Z_5e2993d3e18e active; inspect before retry. No source changes.

F2R20260923T135424Z_5e2993d3e18e completed70.28s:8 actual updates/14 evaluations,
11 exact recovery checks.22 fields rescored;56 F1 controls and32 source hashes
verified. Pilot20260923T135623Z_cc3691d8cc81 active; caps/config unchanged.

F2P20260923T135623Z_cc3691d8cc81 completed74.98s:8 updates/24 evaluations,
32 saved fields rescored. Estimate1334.37s admits1800s full cap;333.59s/member
below600s. Full F220260923T135818Z_5520d5d80cec active, session2232. Preserve
source5e57bcc/code manifest unchanged. Before retry inspect process and result.

F2L wrapper prepared during active F2, before inspecting final outcomes. It will
repeat update63->64 for all four models, exact full-checkpoint/trace/field and
final16/50 evaluation comparison. No schedule extension or quality selection.
Syntax check passed; execution pending completed F2. Training source manifest
unchanged. Run scripts/check_access_late_recovery.py <completed-F2-run> after
F2 verifier; preserve all output. Backup helper work/package_nca_access_training.py
in Codex cwd takes F2B,F2R,F2P,F2,F2L IDs and includes every earlier local artifact.

F2 timing deviation discovered during execution: mass_3-r0 process completed
with seconds1137.741481, cap600, returncode0, timed_outfalse, coinciding with a
929.8s tool-return delay. System suspend is plausible but unconfirmed; active CPU
seconds not measured, so do not subtract the delay or call it a clean benchmark.
Original wait(timeout) did not enforce elapsed cap across the observed pause.
Keep entire run; remaining member started within original overall allowance.
Reporter now exposes elapsed-cap compliance separately from process completion.
After preserving/verifying original F2 and F2L, repair post-wait cap checks and
regress the failure with a simulated delayed successful process; historical
recovery must use the exact original source ZIP after that fix.

Parent F220260923T135818Z_5520d5d80cec finalized interrupted at1801.11s,
254 recorded updates/54 evaluations. Final member stopped after recorded62.
Linked completion20260923T143002Z_55aeaac95580 imported all254/54 with equal
field/checkpoint hashes and executed only63/64 plus final2 evaluations in18.99s
(coordinator),12.31s worker. Cumulative elapsed1820.10s, NOT original cap compliant.
Scientific matrix complete256/56; full verifier active session24556. Next F2L on
child ID, trajectory analysis on child ID, then timeout fix/regression/docs/archive.
Do not rerun the original matrix or overwrite its interrupted outcome.

## F2 completed evidence, source recovery and archive

- Source5e57bcc implements F2;5b9c19c registers trained-state recovery. Final
  evidence/timeout-fix commit follows; inspect Git log and matching backup receipt.
- F2B20260923T135153Z_388dadf703b2, F2R20260923T135424Z_5e2993d3e18e,
  F2P20260923T135623Z_cc3691d8cc81 completed and verified. Never repeat for no reason.
- Parent20260923T135818Z_5520d5d80cec stays interrupted. Third member elapsed
 1137.74s>600 despite no process timeout; system-delay cause not established.
  Overall cap stopped fourth after62. All partial files/logs preserved.
- Child20260923T143002Z_55aeaac95580 completes only63/64 and final evaluations,
  imports308 previous records with equal hashes, complete256/56.19s continuation,
  cumulative1820.10s. This is not a cap-compliant performance result.
- F2L20260923T143228Z_73318ff99c0b: four63->64 actual updates plus eight evaluation
  replays, exact full checkpoint/trace/fields.38.96s. Early recovery11 checks also
  pass. Abrupt writes/GPU/AMP remain uncertified; actual parent timeout and linked
  completed-boundary continuation are separately evidenced.
- Full verification312 fields/256 checkpoints/56 metrics,56 F1 dual-definition
  controls,32 snapshot hashes,8 original fields and8 final rollouts. Old/new binary
  labels match56/56; all eight final masses increase; final binary illegal,
  blocked-ground and unsupported counts total0. No all-nine-family/architecture claim.
- Post-hoc trajectories differ from F1 weights first at51/40/34/30 by member,
  fields one update later. Both post-hoc helpers and permission-error receipt are
  in .local-artifacts/analysis-attempts/F2-*; reports F2-* preserve every outcome.
- Elapsed-cap fix changes ONLY scripts/run_access_training.py among historical
 32 source hashes. Latest regression20260923T143444Z_3c97b03e720e:164 pass, smoke0.
  For exact historical Session/recovery replay, extract registered source ZIP
  into a NEW workspace and use matching runtime/artifacts. Do not bypass hashes.
  Current report_access_training.verify(run,replay=False) can inspect saved data;
  replay=True needs the original source identity. Earlier F2B/R/P reports predate
  added timing/continuation report fields; preserve their published versions.
- continue_access_training.py is deliberately bounded to one unfinished F2 member,
  at least60 completed updates, at most4 missing,120s worker. It rejects completed
  parents and source drift. It was exercised62->64 before the timeout source edit;
  do not invoke it now on completed work or bypass its historical source guard.
- Backup builder in Codex cwd:work/package_nca_access_training.py F2B F2R F2P F2child
  F2L. It also includes the interrupted parent, all earlier raw runs/source archives,
  analysis, viewer, private reports/primer, full Git bundle and prior receipts.
  Output:outputs/NCA-Access-Training-Backup-2026-09-23-<commit>.zip plus.sha256;
  receipt:.local-artifacts/milestones/<commit>-backup-receipt.json. Verify receipt
  before claiming completion. Same-disk local copy only; no Drive operation.
- Next GROWTH_STABILITY_PLAN.md: profile/freeze a no-training growth/firing audit,
  actual gradient tradeoffs, then choose ONE learning change. No larger/paid run,
  architecture redesign or automatic candidate promotion. Studio remains planned.

H1 source commit3acdefd. Pilot20260923T145912Z_17b1d18f5b10 active, session81037.
Inspect result/log/process state before retry. Diagnostic source/config frozen.

H1P20260923T145912Z_17b1d18f5b10 completed73.50s:12 fields/2 new gradients,
12 reused controls,31 source hashes; all scalar/vector/anchor checks pass.
Estimated989.49s admits1500s cap. H1 full20260923T150119Z_f7b304516723 active,
session10499. All source/config unchanged from3acdefd. After completion run
scripts/report_growth_audit.py on that ID, document findings/next decision,
commit and use work/package_nca_growth.py <pilot ID> <full ID> in Codex cwd.

F3 preparation regression20260923T155421Z_8e4d3bfc6919 passed172 tests, zero failures/errors/skips, smoke0. Source commit precedes first parity run; no code changes after pass.

F3B parity20260923T155551Z_dc2ea609facf active session6198; sourcec0a44b1. After completion verify with scripts/report_horizon_training.py, then recovery with this parity ID. No full training admitted yet.

F3B fully verified37 source hashes/36 fields/16 checkpoint cursors/56 F2 and72 H1 controls. F3R20260923T155836Z_10e9dbf37b62 active session59301; inspect completion before reporter and pilot.

F3R20260923T155836Z_10e9dbf37b62 completed100.23s,8 updates/14 evaluations; all11 exact restart checks pass. Reporter session82087 active. Next F3P with parity20260923T155551Z_dc2ea609facf and this recovery ID; full remains gated.

F3R fully verified. F3P20260923T160209Z_e292a696d594 active session37543; sourcec0a44b1. All31 H1 source files still byte-identical; new objective/evaluation ASTs match F2 exactly. Full training requires pilot admission and report verification.

F3P20260923T160209Z_e292a696d594 completed223.38s,8 updates/24 boundaries/72 grid records. Timing-only estimate3246.25s total/811.56s member admits3600/1200 caps. Reporter session91389 active; full may start only after successful verification.

Supplement F3L registered while full active, before final outcome review: scripts/check_horizon_late_recovery.py <full-ID> repeats62->63->64 for all4models, checks complete checkpoint/trace/field equality and cursors. Run after full verification; no added learned exposure. Wrapper separate hash, core37 hashes unchanged.

Full F3 verification passed376 fields/260 cursors/8 exact final rollouts,37 source hashes. Late launch escalation timed out before process creation (tool reported6632.2s); no study time affected. Normal authorized local retry succeeded: F3L20260923T184328Z_418e31921193 active session66685. Post-hoc F3-outcomes.json saved:72/72 mass reductions,59 connectivity losses,0 gains,0 joint final/boundary successes.

A3 regression20260923T192815Z_f7cdbff9cc00 active session61518. Inspect result and log before retry; no diagnostic run yet.

A3 regression20260923T192815Z_f7cdbff9cc00 completed180 tests, zero failures/errors/skips, smoke0,84.62s. Next source commit then timing pilot; scientific code frozen.

A3P20260923T193037Z_0e72e1207fb9 active session59351; source08a6d85. Inspect result/logs before retry. Full audit remains gated by pilot verification and frozen timing admission.

A3P20260923T193037Z_0e72e1207fb9 completed111.09s and verified35 source hashes/25 fields/2 gradient cases/2 bounded probe fields. Timing estimate1038.85s admits1500 cap. All37 historical F3 code hashes unchanged. Next full A3 with exact pilot identity.

Full A3 20260923T193403Z_ac42a8cca390 active session9272; scientific source08a6d85 and identical pilot protocol hashes. Inspect result/logs before retry; after completion run reporter, work/analyze_a3.py, findings/decision, results commit and work/package_nca_raw_access.py <pilot-ID> <full-ID>.

F4 regression20260923T195529Z_9f4db82ab94f active session43932. Inspect result/logs before retry. Scientific implementation complete, no training gate launched yet.

F4 regression20260923T195529Z_9f4db82ab94f completed186 tests, zero failures/errors/skips, smoke0,83.35s. Scientific code frozen; next source commit then F4B.

F4B20260923T195731Z_c19895c6c745 active session31869; source12357e8. Inspect result before reporter, then recovery. Full training remains gated.

F4B20260923T195731Z_c19895c6c745 completed142.85s,12 exact F2 update matches and24 evaluation matches including full checkpoint/RNG trees. Reporter active session29357. After verified report launch F4R with this parity ID.

F4B reporter passed41 source hashes/36 fields/16 cursors/56 F2 boundary and72 H1 controls. F4R20260923T200110Z_961286a9e4b3 active session51304. Verify before pilot.

F4R20260923T200110Z_961286a9e4b3 completed91.25s,8 actual updates/14 evaluations, all11 restart checks exact. Reporter session10108 active. Next pilot after successful verification.

F4R reporter passed41 source hashes/22 fields/10 cursors, all11 recovery checks. F4P20260923T200402Z_b794e30d69ba active session49762. Scientific source unchanged12357e8. Full requires pilot timing admission and verified report.

First F4P20260923T200402Z_b794e30d69ba completed328.52s; estimate2409.50s exceeds2400s allowance, NOT admitted. No full training started. Reporter session93938 active. Preserve this pilot. Next semantics-preserving removal of redundant raw evaluation loss recomputation, then fresh source/regression/parity/recovery/pilot linked to first gates; caps unchanged.

First pilot fully verified but not admitted. D050 removes redundant evaluation recomputation; no objective/training change. Next new regression/source commit, linked B2/R2/P2. Use reporter prefixes F4B2/F4R2/F4P2; old source runs cannot resume with changed hashes.

Efficiency-revision regression20260923T201206Z_4b81a6dabfa4 active session50349. After pass commit new source, launch parity with --parent-run 20260923T195731Z_c19895c6c745 and use F4B2 report prefix.

Efficiency regression20260923T201206Z_4b81a6dabfa4 passed187 tests, no failures/errors/skips, smoke0,86.46s. No scientific edits after pass. Next preserve revised source and repeat B2/R2/P2 with parent links; independently compare full pilot evidence using work/verify_f4_efficiency.py <old-pilot> <new-pilot> before full training.
