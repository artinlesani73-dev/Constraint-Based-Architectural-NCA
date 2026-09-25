# Next-phase implementation plan

## Current: NR3 notebook uploaded and byte-verified - 2026-09-25

The exact approved folder-verification/notebook-save/readback batch is complete.
New file NCA-NR3-Quality-Study.ipynb, ID1BjiTlIePcBrFUq1FZ4ZZMPuAZEAER1pY,
is directly inside project folder1fS34Yy0-oMzSxWaYJFiPTkGgrZstgc0H. All4310 bytes
match local SHA256 96a667c23675580b5ae326a38b00e0a9c97f6a061147bbd21fb5ba5439fdf1f4.
Verified URL: https://drive.google.com/file/d/1BjiTlIePcBrFUq1FZ4ZZMPuAZEAER1pY/view?usp=drivesdk
Recorded in experiments/reports/NR3-drive-notebook.json and Codex outputs/
NR3-Drive-Notebook-Verification. Notebook code compiles and remains disarmed with
APPROVED_SEED_JOB=False; empty outputs. No training, Colab interaction, overwrite,
new runtime ZIP upload or other cloud files. Existing NR2 notebook is untouched.

NEXT: request explicit seed1201 job/backup approval: one256-update T4 job, at most
600 controlled seconds; setup/download/idle allocation are outside its timer.
The user must accept that whole-VM loss before export can lose this one job.
Then guide the user through this NEW notebook and local NCA-NR3-Quality-Package.zip
from Codex outputs/NR3-Quality-Study. Do not use the older NR2 preflight ZIP.
Assistant Drive/Colab reads, notebook edits or autosave actions need separately
stated authorization. Current upload batch confers no standing Drive permission.

After the approved seed finishes, import its ZIP/receipt, verify/archive locally,
then request exact results ZIP/receipt Drive save+readback. No subsequent seed or
automatic retry before verified backups and its own compute approval. Full experiment
archives remain local; notebook-only Drive verification does not satisfy that backup.
Local documentation commit: Record verified NR3 notebook Drive upload.
Incremental archive outputs/NCA-NR3-Drive-Receipt-<commit7>.zip and adjacent receipt
preserve this action; retain full NR3 archive98c7156 and all previous evidence.
Earlier current entries are historical.


## Current: NR3 quality-study package locally ready - 2026-09-25

Read REPAIR_QUALITY_FINDINGS.md, REPAIR_QUALITY_PROTOCOL.md, D077 and NR3-quality.json.
Package Codex outputs/NR3-Quality-Study: new disarmed notebook, TRAIN-only ZIP, guide,
receipt. No NR3 paid job, Drive operation or model-quality outcome. Three proposed
fresh256-update models, each separate600s job; max1800s controlled/60min allocated
GPU proposal. Keep architecture/loss/16steps/32grid/NL0 splits/nine families.

8 focused tests 20260925T140118Z_93161ba36ba8 and379 full tests 20260925T140149Z_2197b0f9fb6b pass,smoke0.
Extracted CPU rehearsal 20260925T140804Z_7d74952e7f95:8 updates,11.80s,9 learning-state
checkpoint matches and8 raw-state matches to NR2; source identity intentionally new.
One TRAIN initial32-step evaluator probe matches archived NL0 metrics exactly.
All source snapshots/hash receipts verify. No heldout evaluation or long training.

NEXT: request exact save/readback for NCA-NR3-Quality-Study.ipynb directly inside
Drive folder1fS34Yy0-oMzSxWaYJFiPTkGgrZstgc0H. Then present seed1201 compute/backup
approval (256updates,600s cap; setup/idleGPU billed separately). Notebook remains
APPROVED_SEED_JOB=False. Run only an approved seed. Download ZIP/receipt, verify
local copy, ask separately to save/readback that exact evidence in Drive, then
seek the next seed allowance. No automatic mount/sync/retry or standing Drive grant.
Within-job VM loss remains possible; exact cross-VM continuation is not certified.

After all three final model archives are fixed, run the frozen local CPU evaluator:
scripts/evaluate_repair_quality.py --archives <three ZIPs> --manifest-sha256
7234b6d0dcc47959b2a91e59141521ca743b153c06f8c4b6a6f732a678ebb6e7 --mode diagnostics|final --output <freshfolder>.
Adjacent receipts required. Freeze all update256 hashes before heldout inference.
Use the project .venv. Each mode has a10800s between-observation cap; partial results
cannot pass. No retuning based on intermediate validation or test outcomes.

Helpers Codex work/*nr3*.py and raw .local-artifacts/runs/<IDs above> retained.
Do not rerun completed preparation helpers or overwrite packages. Local commit:
Prepare bounded NCA repair quality study and evaluation gates.
Archive outputs/NCA-NR3-Backup-2026-09-25-<commit7>.zip, authoritative repository
receipt .local-artifacts/milestones/<commit7>-nr3-backup-receipt.json. Preserve all
NR2 GPU/Drive/full and earlier archives. Restore source snapshots for exact hashes.
Private reports remain ignored/unchanged; Studio and unrelated files untouched.
Earlier current/in-progress sections are historical.


## Current: NR2 GPU recovery verified from returned Colab artifacts - 2026-09-25

User supplied run 20260925T134015Z_e14a0afc966e. All88 archive payload hashes and whole ZIP hash
verified. Direct local inspection confirms10 exact complete checkpoint pairs,
8 training-state pairs and2 evaluation-boundary pairs, including Adam, sampler
and all saved RNG state. Delivered revision2 manifest and81 training rows match.
Tesla T4, PyTorch2.11.0+cu128, Python3.13.15, NumPy2.1.3. Run27.9814s;
8 unique/16 executed updates; peak reserved368MiB. Read NR2-gpu-verification.json.
Earlier 20260925T131427Z_3726c7292296 failed GPU availability before learning and is retained
in NR2-colab-first-attempt.json and Codex outputs/NR2-Colab-<run-id>/.

This completes the same-runtime GPU process-recovery gate. It does not establish
learned geometry quality, convergence, cross-VM restart, or full off-device backup.
User executed the Colab runs; assistant imported local evidence only. No permission
for additional remote compute or Drive operations is inferred. No Studio change.

NEXT: prepare a frozen, bounded repair-quality study and evaluation/backup plan
using this verified GPU baseline. Retain NR1 math/NL0 splits/nine families and
MG7 comparator; require binary geometry and family metrics as well as loss.
Do not automatically start the proposed60-minute/3-seed/256-update study or retry
preflight. Actual new allowance and exact backup actions require user approval.
The notebook-only Drive copy is verified; full artifacts remain locally archived.
Advise the user to disconnect the idle GPU now that downloads are verified.

Local evidence: Codex outputs/NR2-Colab-20260925T134015Z_e14a0afc966e/ (ZIP, receipt, verification).
SHA256 e5037e8ec634737f20af0dc98196dabc17160bd3140f751c10de190f781b2d8b.
Inspection helper work/verify_nr2_gpu.py; no training executed during verification.
No code changes or new regression tests needed for this data-import milestone.
Local commit: Record successful Colab GPU recovery verification.
Preserve earlier archives and both returned runs. Earlier current entries are historical.


## Current: NR2 notebook saved and byte-verified in project Drive - 2026-09-25

The user explicitly approved folder verification, notebook upload and readback.
That batch is complete; no standing Drive authorization remains. The sole upload
was the current revision2 NCA-NR2-GPU-Preflight.ipynb, file ID
1ZNLINdkrm4f2CEI82ApiTfmKjC7wS0AG, directly under folder
1fS34Yy0-oMzSxWaYJFiPTkGgrZstgc0H. Metadata confirms the parent; all4015 bytes
match local SHA256 1774d516b10ad75819d0ed6078c9ebb86ee18b1fbe771273df2682480312002c.
Drive URL: https://drive.google.com/file/d/1ZNLINdkrm4f2CEI82ApiTfmKjC7wS0AG/view?usp=drivesdk
The raw download transport failed (sandbox denial, then HTTP403); the connector's
default fetch returned the bounded notebook payload and exact verification passed.
No duplicate upload, mutation or deletion. Evidence is in Codex outputs/
NR2-Drive-Notebook-Verification and experiments/reports/NR2-drive-notebook.json.

NEXT: request one GPU preflight allowance (oneGPU,8 unique/16 executed updates,
600s controlled-job cap, setup/download/idle allocation excluded). GPU execution
and any further Drive/Colab notebook reads or autosaved edits need explicit scope
approval before assistant actions. No browser/Colab session was opened this turn;
the uploaded notebook remains disarmed. Current data/source ZIP remains LOCAL in
Codex outputs/NR2-Colab-Preflight-v2 and can be uploaded to runtime only after job
approval. No Drive mount/sync/automatic retry or longer study is authorized.

Only this notebook now has a verified off-device copy. Full experiment artifacts,
checkpoints and archives have NOT been backed up to Drive. Do not conflate this
upload with the outstanding project backup requirement. Preserve all local archive
chains and evidence; prior NR2 milestone receipt certifies the3b2fdb9 archive.
New local documentation commit title: Record verified NR2 notebook Drive upload.
Incremental evidence archive: Codex outputs/NCA-NR2-Drive-Receipt-<commit7>.zip;
its adjacent receipt certifies integrity. Earlier current entries are historical.


## Current: NR2 local package readiness complete - 2026-09-25

Read COLAB_PREFLIGHT_FINDINGS.md, COLAB_PREFLIGHT_PROTOCOL.md and D075.
Use Codex outputs/NR2-Colab-Preflight-v2 (ZIP, disarmed notebook, START-HERE,
receipt). Original package is superseded and retained. No cloud/Drive/GPU execution.
Current config experiments/configs/NR2-readiness-v2.json, linked to failed attempt
20260925T110616Z_d542bd6412fa. Original stalled at NumPy import with blocking stdin watchdog;
reproduced and corrected using Windows pipe polling and monotonic deadline polling.
All originals, diagnostic stacks and failure evidence retained.

Focused 20260925T112534Z_1553fde22b4c:7pass. Regression 20260925T112716Z_bd5f2b1cfe5d:371pass/smoke0.
Successful isolated CPU rehearsal 20260925T113206Z_e84a09a6bda5:10 exact checkpoint pairs,
8 training-state pairs,2 evaluation-boundary pairs; three stopped owned worker trees.
8 unique/16 executed updates,38.91s/600s. GPU remains untested; no quality claim.
Source/exports/checkpoints reverified by work/finalize_nr2.py and NR2-verification.json.

NEXT: present this concrete package; request exact scoped notebook Drive save and
readback/verification if wanted, plus separate one-attempt GPU allowance (oneGPU,
8 unique/16 executed updates,600s execution cap). User must approve exact Drive
actions and compute before remote work. Do not mount Drive, auto-sync, retry or
start the proposed256-update study. Setup/download/idleGPU time is outside job cap.
Download evidence+receipt and verify locally before disconnecting Colab. Whole-VM
loss before download is not protected; extended training requires approved backup.
Review actualGPU evidence before deciding any longer study; retain strict failures.

Local commands already completed: scripts/build_colab_preflight.py --output
<Codex outputs/NR2-Colab-Preflight-v2>; work/rehearse_nr2_v2.py. Do not rerun completed
helpers or overwrite packages. If interrupted, inspect run results/processes first.
Raw successful checkpoints are under Codex outputs/nr2-rehearsals/20260925T113206Z_e84a09a6bda5/runs/;
uninterrupted/resumed end at0008, interrupted at0004. All are also in verified exports.

Commit title: Prepare verified Colab NCA restart preflight package.
Archive: outputs/NCA-NR2-Backup-2026-09-25-<commit7>.zip. Authoritative receipt:
.local-artifacts/milestones/<commit7>-nr2-backup-receipt.json. Keep NR1/NL0 and older
archives; same disk only, off-device pending. Restore exact snapshot bytes for hashes.
Studio was not restarted or changed. Private reports ignored/unchanged. Earlier
current/in-progress sections are historical; two unrelated untracked files untouched.


## Current: NR1 CPU mechanics complete - 2026-09-25

Read NCA_REPAIR_CPU_FINDINGS.md, NCA_REPAIR_CPU_PROTOCOL.md and D074.
Pilot 20260925T100650Z_78bc22b9d584: two8-update CPU members,6 workers,20 exact
checkpoint pairs,16 exact training-state pairs,14 boundaries rescored.61.99s/600s.
All worker trees stopped. Both final binary fields equal their damaged inputs:
loss reduction is not geometric repair. No quality promotion or paid training.
Focused 20260925T100223Z_4f20845cb058:7pass. Full 20260925T100255Z_b00e322820cb:364pass,smoke0.
The final driver ownership correction is validated by the actual pilot; model,
training and test source bytes match focused/full snapshots. Initial driver kept.

Next prepare a versioned Colab package and full multi-example sampler/GPU recovery
preflight, retaining this CPU control and all NL0 training/validation/test separation.
No further local duration/weight sweep or GPU job is admitted automatically.
Present the concrete package/allowance before requesting paid execution. No Drive
mount/read/write without exact separate permission. No user setup is needed yet.

Raw: .local-artifacts/runs/20260925T100650Z_78bc22b9d584/workers/<member>-<branch>/.
Latest verified completed states are checkpoint-0008.pt plus .json manifests in
the uninterrupted/resumed folders; interrupted branches end at0004. Preserve all.
No pilot/regression process remains active. Studio server was not restarted;
MS2 last reported exec session81429, inspect before assuming it is still alive.
Helpers Codex work/*nr1*.py use exclusive writes. Do not rerun completed work.
If explicitly making another CPU attempt, project .venv Python command is
scripts/run_repair_pilot.py --parent-run 20260925T100650Z_78bc22b9d584; this starts
new preserved evidence and repeats the frozen pilot, not a longer training run.

Milestone title: "Implement NCA repair pilot with exact CPU process recovery".
Archive outputs/NCA-NR1-Backup-2026-09-25-<commit7>.zip; repo receipt
.local-artifacts/milestones/<commit7>-nr1-backup-receipt.json is authoritative.
Keep NL0/MS2/MG7 and older archives. Restore exact raw source.zip bytes for
newline-sensitive identity checks; never relax hashes. Same-disk only, Drive pending.
Private reports ignored/unchanged; two unrelated files remain untracked.
Earlier current/in-progress sections are historical.


## Current: NL0 repair benchmark complete - 2026-09-25

Read NCA_REPAIR_BASELINE_FINDINGS.md, NCA_REPAIR_BASELINE_PROTOCOL.md and D073.
Preparation 20260925T094341Z_316cff241020 passes:54 valid teachers,162 examples,
9 blocked guards,6 positive site contexts,54 distinct targets. Regression
20260925T094122Z_b9efabe856ff:357pass,smoke0. All evidence/source hashes verify.
Closing repairs some cuts but invalidates8/18 intact test volumes; do not deploy
automatic closing. No learning occurred; current Studio remains procedural MG7.

Next implement NR1 fresh conditioned NCA/trainer and freeze the complete sampler
and checkpoint config. Run ONLY the proposed two-training-example CPU mechanics
pilot (8 updates/member,600s cap) with real process restart after update4.
The target is a separate supervisor, not model conditioning. Test/validation sites
must remain excluded from learned updates. NR1 training has NOT been implemented
or executed yet. Colab proposal comes after local recovery and package preparation;
no paid budget, Drive operation, push or production promotion is authorized.

No preparation/regression process remains active. No server restart performed;
MS2's last server session81429 may still be active; inspect before starting another.
Artifacts: experiments/reports/NL0-*.json; .local-artifacts/runs/<IDs>/study.json
and lossless NPZ files. Helpers Codex work/*nl0*.py retain creation, verification,
finalization and archive actions. They use exclusive writes; do not rerun blindly.
To make a deliberately new preparation attempt, use project .venv Python with
scripts/prepare_repair_benchmark.py --parent-run 20260925T094341Z_316cff241020;
this creates new evidence and is unnecessary for resuming NR1 implementation.

Milestone title: "Freeze NCA volume repair benchmark and next training protocol".
Archive: Codex outputs/NCA-NL0-Backup-2026-09-25-<commit7>.zip plus verified receipt;
repo .local-artifacts/milestones/<commit7>-nl0-backup-receipt.json is authoritative.
Keep MS2/MG7 and all earlier incremental archives. Off-device backup pending.
Private ignored reports unchanged; two unrelated user files remain untracked.
Earlier active/current sections are historical.


## In progress: NL0 learning-baseline preparation - 2026-09-25

Read NCA_REPAIR_BASELINE_PROTOCOL.md and experiments/configs/NL0-repair.json.
Implementing frozen MG7 teacher/damage data and nonlearned repair controls,
with geometry-level split checks and separate target labels. This is data
admission, not NCA training. Next execute archived regression and NL0 audit,
retain all failures, document outcome, commit and verify local archive.
After NL0 admission, implement NR1 CPU mechanics/recovery pilot before any
Colab request. No paid compute, Drive, promotion or remote push authorized.
Historical milestone sections follow; never rerun a completed audit blindly.



## Active milestone - MS2 complete; review volumes and specify learning baseline

2026-09-25. Versioned Studio integration is admitted: 351 tests, 57 exact preset
comparisons and corrected browser acceptance pass. Read STUDIO_V2_FINDINGS and
STUDIO_V2_USER_GUIDE. Both old and new workflows remain available. Next review
the larger volumes, then freeze a separate learned NCA comparator with held-out
sites, corrected volume meaning, recovery and an explicit Colab allowance.
No training or broader preset admission yet. Earlier active sections are historical.



## Active milestone - MG7 complete; conditional Studio integration next

2026-09-25. The equivalence and paired performance gates both pass.
Read INCREMENTAL_GROWTH_FINDINGS, STUDIO_SCALE_INTEGRATION_NEXT_PLAN and D071.
343 regressions pass; 188 nonpartition and 49 blocked outcomes are preserved.
235 full exact matches; two old timeout prefixes extend to fully audited results.
Next specify a versioned integration with old replay compatibility, bounded worker
lifecycle and limited evaluated scale presets. MS1/MG3 remains the live version.
Earlier active/current sections are historical.


## Active milestone - MG6 complete; exact-output efficiency next

2026-09-25. The48-grid gate passed; the64-grid gate failed and larger live work remains unadmitted. Read SCALE_STUDY_FINDINGS and D070.
337 regressions and independent representation/replay audits pass. New physical
environments retain0.8m voxels and existing nine-family meanings. Next specify a
separate local-count optimization with frozen exact-output and paired-resource
gates, then consider admitted live presets. Finer resolution and training remain
separate. Earlier active/current sections are historical.


## Active milestone - MG5 complete; physical-scale readiness next

2026-09-24. All180 nonpartition cases pass in the combined225-case comparison;
45 blocked fail.179 prior positives preserved,221 fields/all225 routes unchanged.
Known MG4 coverage failure repaired;332 regressions and independent full frontier
audit verify. Read COVERAGE_GROWTH_FINDINGS and SCALE_READINESS_NEXT_PLAN.
Next audit grid/context/physical units, then freeze a bounded48-grid study with
conditional64-grid admission. Larger sites and finer resolution are distinct.
No scale execution or new live version yet. Historical milestone entries follow.

## Active milestone - MG4 complete; coverage-aware growth proposal next

2026-09-24.143/144 nonpartition and0/36 blocked pass; all144 requested volumes met.
One fixed-X-third coverage failure blocks the strict scale admission gate.324
regressions and full independent replay verify. No generator/evaluator changes.
Read SITE_GENERALIZATION_FINDINGS and COVERAGE_GROWTH_NEXT_PLAN. Next specify
finite deficit-aware growth with existing facade/request rules and freeze its
diagnostic comparison; MG5 not implemented yet. Preserve inspected MG4 evidence.
Larger-grid runs, custom live sites and paid training remain outside this milestone.

## Active milestone - MS1 complete; live review and broader evaluation next

2026-09-24. Live mass workflow at /static/live/index.html. D067 and
STUDIO_MASSING_FINDINGS record scope and limitations.320 tests pass; three real
browser jobs exactly match MG3. Durable records, comparisons, replay-checked
portable source/geometry and old-workflow compatibility verified. User guide ready.
Next freeze an unseen-context study separately from larger-grid resource/scaling
checks; preserve nine families, physical units and all failures. Numerical study
spec not yet frozen. No automatic paid NCA training or custom-site UI expansion.

## Active milestone - MG3 complete; interactive mass Studio next, 2026-09-24

36/36 nonblocked and36/45 overall pass with no MG1/MG2 regressions. Two failures
repaired,43 fields unchanged.310 regressions and independent full replay verify.
Read CONTACT_BUDGET_FINDINGS and STUDIO_MASSING_INTEGRATION_PLAN. Implement the
versioned experimental live workflow with durable jobs, honest failures and
replayable saved alternatives. Existing live material workflow remains unchanged
until that implementation. No paid training or new constraints admitted.

## Active milestone - MG2 complete; contact-budget growth next, 2026-09-24

Read CONTACT_GENERATION_FINDINGS.md and D065.34/45 pass vs27/45, no baseline
regressions. Two nonblocked facade failures remain, so the frozen36/36 admission
gate fails. No live mass integration.303 regression passes and full replay verify.
Next bounded proposal: CONTACT_BUDGET_NEXT_PLAN.md, exact global contact accounting
during cube growth with deferred candidates and explicit stalls. Keep thresholds,
cost12 and all nine families. Do not start paid training or a coefficient search.

## Active milestone - MD1 complete; procedural expansion next, 2026-09-24

Read MASSING_DIRECT_FINDINGS.md and D064. Contact-aware generation passes4/4;
direct32 preserves the original2/4 with no binary changes at saved boundaries.
303 regressions and exact fresh-process restart pass. Soft facade zero failed
to predict binary facade validity. Do not promote this objective to NCA training.
Next freeze and test the same contact-aware recipe on the full45-case MG1 matrix,
then integrate mass generation in the live Studio if validated. No automatic
loss/threshold retuning. The new direct gallery presents saved evidence only.

## Active milestone - MO1 objective admission complete, 2026-09-24

Read MASSING_OBJECTIVE_FINDINGS.md and D063.873 family comparisons and97 bulk masks
agree;297 regressions pass. No optimizer updates. Next implement the concrete
MASSING_DIRECT_PILOT_PLAN: four-member direct session, iteration0/recovery/timing
admission and contact-aware procedural comparator before the bounded comparison.
Keep the user-approved MG1 volumes and all old evidence. No new training by default.

## Active milestone - MG1 / R2-A complete, 2026-09-24

Read MASS_GENERATION_FINDINGS.md and D062. Five development contexts,45 generated
requests and4 challenges retained; 27/45 meet MT1 pilot checks.287regressions pass.
Next R2-B: separately versioned continuous-objective endpoint/gradient tests and
bounded direct-optimization protocol. Preserve failures and all old definitions.
Gallery is static evidence; mass-generation Studio job integration remains separate.

## Governing update - R2 / D061, 2026-09-24

Read RESEARCH_BRIEF_R2.md. The user approved a revised review based on findings.
Building mass is the accepted output; prior material/platform/void-first directions
and older milestone status labels below are historical. The original private report
is preserved; its dated Revision 2 addendum is local and ignored.

Next: R2-A procedural mass alternatives with a frozen finite benchmark and complete
retained outcomes; R2-B separate continuous-objective/direct-optimization control;
R2-C learned pilot only after a specific measurable benefit and stop rule are frozen.
Product integration runs alongside this sequence. Scaling and paid Colab remain
gated. MT1 is the latest completed science milestone; this is documentation only.

## Active milestone - MT1 complete, 2026-09-24

D060 and MASSING_TARGETS_FINDINGS.md record the pilot binary nine-family contract:
four contexts,48 controls,432 sensitivity evaluations and275 regression tests.
Next: parameterized mass generation and direct-optimization controls with a
versioned continuous objective, binary-parity/gradient checks and bounded compute.
No new NCA training or architectural-quality claim. Historical entries follow.

## Active direction - D058, 2026-09-24

MA1 completion comparison is now implemented; see MASSING_AUDIT_FINDINGS.md and
D059. Eleven controls/33 records/83 audit checks/260 regression tests; all pass
after a retained expected-domain-count correction. Next: versioned massing
objective acceptance examples across several scenes, before new learning.
The following D058 paragraph records the earlier plan, now partly completed.

Generate overall building mass; develop interiors and construction later.
MASSING_BRIEF.md supersedes the material/void target in historical entries below
and VOLUMETRIC_NEXT_PHASE.md. Next: versioned massing contract, saved original/
derived comparison, then nine-family compatibility audit. No filling implementation,
revised loss or new training is claimed yet.

Owner: Artin Lesani. Started 2026-09-13. Historical baseline: ac913b9.

This tracked plan implements the private next-phase report. It supersedes the old implementation-status document for new work without erasing historical claims. The report itself stays ignored. No new constraint families are planned.

## M0 - Preserve evidence and establish a reliable starting point (complete except the Drive round trip)

- [x] Create a separate implementation branch; preserve the original tracked files and supplied reports in a hash-verified local snapshot.
- [x] Ignore both report formats; retain small plans, decisions, summaries and resume instructions in Git.
- [x] Add append-only run metadata, artifact hashes, source snapshots, explicit outcomes, linked retry IDs and verified backup transfer.
- [x] Add independent binary geometry primitives and synthetic failure tests. These are a first evaluator foundation, not a complete validator for all nine families.
- [x] Load the real Model C checkpoint with its embedded configuration and fail loudly when required assets are absent.
- [x] Repair the missing facade-metadata preview crash; test the actual API path.
- [x] Record final verification and refresh the handoff: run `20260913T194223Z_ad4ee8ee60f3` passed all 16 tests and the out-of-directory smoke check. Milestone commit is recorded in Git history under `Establish reproducible NCA next-phase foundations`.
- [ ] Verify a real Google Drive backup round trip (requires user's Drive location / Colab sign-in).

## M1 - Define the geometry contract and replay the original model (current milestone)

1. [x] Freeze scene schema v1: world units, array axes, entrance IDs, material field, empty-space interpretation, support region and design envelope. Implemented as `scene_v1` in `nca/contract.py` with the frozen set `experiments/scenes/reference_v1/`; see `GEOMETRY_CONTRACT.md` and decisions D007/D008. A height ceiling stays reserved and inert pending a user decision, since it is not one of the nine families.
2. Resolve elevated access semantics. Begin with explicit spatial connectivity; do not claim walking clearance, floor support or structural engineering certification. Partially addressed: facade entrances must sit above the street band and be face-adjacent to a building, and connectivity is documented as spatial only. Clearance, headroom and deck semantics remain undefined.
3. [x] Extract a shared rollout with named historical-training, historical-evaluation, and historical-serving profiles. Preserve the legacy functions for exact comparisons. Implemented as `rollout_v1` in `nca/rollout.py`; see `ROLLOUT_PROFILES.md` and decisions D009/D010. Agreement is bitwise against both legacy paths. Reading the historical notebook corrected one earlier claim and added four findings, including that the recorded historical evaluation used no corridor scaffold and that Model C never saw a ground-type access point.
4. Run E0 on identical frozen scenes/checkpoint/seeds: isolate initial seed scale, firing, noise and masking. Save every per-scene result, continuous fields, binary thresholds, timing, full config and source snapshot. Two frozen sets are now in place, `reference_v1` (designed) and `legacy_easy_v1` (in-distribution, per D011); the legacy set must be seeded with `nca.legacy_scenes.legacy_seed_state`, not the deployed generator. See `SCENE_SETS.md`.
5. Correct corridor dilation with an explicitly versioned operator. Do not silently relabel outputs from the legacy operator as corrected results.

Acceptance: deterministic reference replay works; differences between historical profiles are measured; disconnected endpoints fail; empty output cannot be mistaken for a good design. Gate A is not complete until rollout and gradient checks are finished.

## M2 - Correct the trainable baseline

Move model, scene generator, rollout, differentiable losses and evaluation into the shared package. Repair impossible coverage, ground openness, thickness background, loss tensor shapes and gradient paths. Test batch sizes one and greater than one. Repair actual building/access sampling; all constraints remain active while geometry becomes more diverse.

Run E1 (corrected NCA) and E2 (scaffold-only, procedural, direct voxel optimization) on matched scenes. Use validation cases for settings and a sealed test set for final assessment. Retain failed cases and distinguish original scores from metric v1/v2 results.

## M3 - Learn behavior that merits an NCA

Run E3-E5 separately: scene-paired state pools and edit recovery; context conditioning and 16/32 recurrent channels; improved local or multiscale perception. Match update count and wall-clock budget where practical. Use several training seeds for finalists. Pivot only after documented baseline comparisons, not selected renders.

## M4 - Build the design workspace

Agree on stable scene/result contracts, then implement versioned jobs, real worker cancellation, per-job settings/random generators, compact data transport, cached context, instanced/chunked rendering, resource disposal, scene editing, alternatives, diagnostics and saved projects. GLB/image export follows explicit units and geometry validation. The supplied HTML is a visual concept, not a product implementation.

## M5 - Scale and validate

Profile 32³ before 64³; distinguish larger physical sites from finer voxels. Measure memory/latency and validity before trying coarse/global plus fine/local processing. Test a decoder only after reliable geometry exists. Validate on held-out site families and retain all outputs. A nicer surface must not hide disconnected or unsupported geometry.

## Colab policy

Use Colab for experiments, not permanent hosting. The user chose Google Drive plus local archive. Before training: confirm the GPU-hour cap, freeze the exact config/scenes, verify checkpoint restore including optimizer and RNG states, simulate interruption, and confirm two usable artifact copies. A proposed pilot cap is not spending authorization. No paid training has started.

## Working cadence

Each milestone has a local commit, changelog entry, decision updates, and a refreshed `RESUME.md`. Each experiment gets a unique immutable run record. Review progress by evidence gates; the earlier 8-12-week estimate is not a promise or a reason to skip the baseline.

## Acceptance update - 2026-09-23

M1 step 3 and the legacy scene-set implementation now have a recorded local
regression run: `20260922T223533Z_9c5bfe017c4c` (76 passed; smoke exit 0).
Before M1 step 4, repair/reject the delta-mask fire-rate and explicit-RNG variants,
validate module-mode/firing compatibility, and test historical-training parity
against the notebook loop. These limitations were found in review after the
existing suite passed; see CHANGELOG.md. Do not interpret passing historical
default tests as validation of those ablations. E0 has not run.

## E0 preparation accepted - 2026-09-23

The preconditions above are addressed in rollout_v2 and the notebook-forward
oracle. Run `20260922T225856Z_26be76516805` passes all 82 checks. D015 fixes the
E0_v1 protocol; run it locally next. D014 keeps the prepared backup local and
supersedes the earlier pending Drive-upload action; do not access Drive.

## E0 completed - 2026-09-23

M1 step 4 has a recorded 270-case diagnostic, run
`20260922T230120Z_76f3b4677e8f`. Read E0_FINDINGS.md and D016. Main-profile
comparisons use three seeds; single-seed ablations remain preliminary. The
post-run target audit exposes both target conflicts and a limited scaffold
connectivity advantage. M1 overall is not complete: the versioned corridor fix
and architectural semantics/gradient work remain. Next implement the bounded
operator, then measure legality/routing corrections separately before training.

## Corridor corrections completed - 2026-09-23

M1 step 5 is implemented and measured under corridor_bounded_v1, with the separate
corridor_legal_v1 routing intervention. C1_v1 records 54 targets and 108 matched
cases (20260923T000945Z_3cbdc3603a12); 94 regression checks pass. See D017/D018,
CORRIDOR_FINDINGS.md and the full report. All 17 feasible targets connect legally,
but the original checkpoint still fails on reference scenes. Gate A remains
incomplete. Next follow LOSS_REPAIR_PLAN.md: establish compatible objective
semantics and a versioned shared loss package with batch/gradient checks before
any optimizer experiment. The interface and scaling milestones remain planned.

## Shared loss mechanics measured - 2026-09-23

M2 now has geometry_losses_v1 and L1_v1 diagnostics. All 109 regression checks
pass; L1 records 72 contexts, three model-gradient cases and six historical defect
checks. Read LOSS_FINDINGS.md and D019/D020. Gate A remains open: five feasible
scenes conflict with the tested radius-six envelope/mass floor, and a measured
ground-access gradient is blocked by the lower material clamp. Next explicitly
resolve objective regions/budgets and test a bounded gradient intervention before
optimizer/recovery work. No Colab job or model architecture change is selected.

## L2/R1 completed - 2026-09-23

Read INTERVENTION_FINDINGS.md and D022. L2 measured 108 explicit budget cases,
54 gradient cases and nine absent-scaffold controls. Pre-clamp coverage restores
the known blocked derivative while matching hard forward states exactly; smooth
material adds diffuse occupancy without binary benefit in the tested matrix.
Envelope budgeting is compatible by necessary bounds but cuts absolute material
allowance to 2.04%-10.65% of site budgeting. It is not the production choice.

R1 proves exact completed-update CPU recovery over four logical Adam updates in
fresh processes; this opens mechanics preparation, not paid training or Gate A.
118 tests pass. Next audit all-nine-term feasibility on explicit geometry targets,
calibrate loss/regularizer magnitudes and preregister E2 procedural/direct/NCA
controls. CUDA/Colab recovery and an approved compute cap remain prerequisites.

## T1 completed - 2026-09-23

Run20260923T082527Z_845d2aa6aec0:432 geometry cases,72 joint bounds,36 gradient
probes,123 regression tests pass. Read TARGET_AUDIT_FINDINGS.md and D024.
Radius6/envelope necessary compatibility drops to15/17 feasible scenes after
facade contact; simple guides can score zero on all nine terms. Final coefficient
calibration is deferred until intended architectural semantics and contradictions
are addressed, not replaced with arbitrary weights. NEXT_EXPERIMENT_PLAN.md
specifies semantic fixtures/alternatives, gradient calibration and paired E2 arms.
User is unsure and asked for advice. Recommend material/form generation first,
with usability evaluated separately; keep usable pavilion/bridge as the longer-term
goal. The independent audit is complete; final architectural specification is open.

## A1/W1 completed - 2026-09-23

User accepted architectural material generation as near-term scope. A1
20260923T084341Z_e37699e31f26 isolates a scene-derived facade endpoint allowance;
other eight terms unchanged, all18 facade-blanket controls penalized, radius-six
necessary compatibility17/17 feasible scenes. W1
20260923T084933Z_eb2603cd79f7 provides17 independently replayed constructive
zero-loss witnesses and retains sealed-reference infeasibility.132 tests pass.

Read FACADE_FINDINGS.md and D026-D028. Select the explicit experimental contract
for actual model-gradient/regularizer calibration preparation and matched E2
controls. This closes the measured numerical contradiction, not architectural
quality validation. Production defaults stay unchanged; next work is local.


## K1/R2 completed; K2 prepared - 2026-09-23

Read CALIBRATION_FINDINGS.md and D031. Actual gradients and regularizer provenance
are now measured: K1 has 71 model cases and 51 budget probes; 141 tests pass.
The historical cantilever skips the bottom three layers; REGULARIZER_AUDIT.md
supersedes any earlier suggestion that it directly penalized cells at z=0.
Research-objective CPU recovery passes all seven exact checks in R2.

Stage B coefficient selection remains open. K2-sensitivity.json freezes an
isolated sparsity-weight comparison (30 versus 3), two seeds and 17 updates per
run, 16-step rollouts, 900-second CPU cap per run. It is prepared, not executed.
Implement/test the actual K2 loop and run it locally next. Keep no-update and W1
controls, all per-family failures, and separate fresh holdouts for later E2.
No evidence yet warrants scaling, architecture changes or a better-model claim.
Studio work and larger environment diversity remain planned milestones.


## K2 completed; direct-material comparator next - 2026-09-23

Read SENSITIVITY_FINDINGS.md/D033.144 tests pass; actual-loop recovery has seven
exact passing checks. K2 completes68 optimizer updates and187 matched evaluations.
Neither coefficient setting provides joint success: weight30 retains10/17 connected
and15/17 over budget at50 steps; weight3 gives11-12/17 connected but17/17 over
budget. All five feasible reference scenes remain disconnected. W1 remains17/17
connected/in-budget, while offering no architectural-quality certificate.

Next prepare and profile E2 direct-material optimization under the same semantics
on legacy008, ground-pair and minimal-smoke, before freezing a full17-scene matched
protocol. This separates objective/optimization difficulty from recurrent-model
limitations; per-scene optimization is not generalizing inference. Preserve both
K2 settings as controls, original defaults and all failures.17 updates do not prove
convergence or model-concept failure. No larger/paid run or architecture change is
selected. Fresh holdouts, recovery/conditioning and studio/scaling remain planned.


## D1 completed; NCA fitting capacity diagnostic next - 2026-09-23

Read DIRECT_FINDINGS.md/D035.147 tests pass; exact direct-loop CPU recovery passes.
Full D1:34 cases/1088 updates. Both recipes connect all17, including all five
reference cases. Weight30 is in-budget12/17 (all five references), weight3 in-budget
10/17. Keep under-floor/over-cap and continuous-objective residuals visible.
Direct per-scene fitting has more freedom and different compute; this is not an
architecture-failure proof or learned generalization result. W1 remains the simple
numerical comparator. Neither coefficient/model is promoted.

Next profile/freeze a repeated single-scene NCA fitting test on ground-pair and
minimal-smoke, retaining both recipes, checkpoint/architecture/objectives and weak
initialization. Candidate64 updates, intermediate evaluations and16/50-step growth
checks; exact schedule/compute cap require a recorded timing/recovery gate. No
solved-field initialization or added reconstruction constraint. If fitting works,
test scheduling/recovery/generalization; if not, inspect gradient/saturation then
isolate one conditioning/perception change. Fresh holdouts and scaling remain later.
Local evidence viewer now contains real saved geometry; data/syntax verified, but
browser file-URL policy blocked visual/interactive verification. Live deployment,
worker jobs/cancellation, scene editing and production UI remain planned.


## F1 completed; access-source alignment next - 2026-09-23

Read FITTING_FINDINGS.md/D037. Repeated single-scene fitting completes256 updates
and56 evaluations; three of four final models connect at50 steps, none at16,
and no evaluated boundary is jointly connected/in-budget.151 tests pass,312
fields rescored and eight final checkpoint rollouts replay exactly. No promotion.
The source audit exposes fixed-point access loss versus connected-region binary
evaluation: all56 fixed sources empty, while three final regions connect.
Next follow ACCESS_ALIGNMENT_PLAN.md before architecture changes. Reconcile
semantics without introducing disconnected multiple origins, then trace actual
parameter gradients and isolate one learning intervention. Preserve all previous
contracts/reports, budget limits and nine families. Scaling/holdouts/deployment
remain planned; no Colab setup or paid run is needed for this next local audit.


## A2 completed; access-only learning comparison next - 2026-09-23

Read ACCESS_AUDIT_FINDINGS.md/D039. A2 replays277 fields and12 actual-model
gradient cases without training.158 tests pass. Old access parameter gradients
are zero in12/12 cases; component_bottleneck_v2 restores them in all four fitted
16-step cases, with coverage-aligned directions. Original disconnected cases
still need existing pre-clamp coverage guidance. All277 binary labels match;
eight access-score reductions and30 increases are semantic rescoring, not better
generated geometry. No model or production objective promoted.

Next follow ACCESS_TRAINING_PLAN.md: opt-in access-only objective, baseline parity
against F1, actual-loop CPU recovery and timing admission before proposed matched
64-update comparison. Other eight families, regularizers, recipes, architecture
and scenes unchanged. Evaluate both definitions and joint connectivity/budget.
Candidate is CPU-only; GPU/Colab recovery and spending approval remain pending.

## F2 complete through linked continuation - 2026-09-23

Read ACCESS_TRAINING_FINDINGS.md/D041-D042. Access-only training improves final
connectivity0/4->2/4 at16 growth steps and3/4->4/4 at50. No one of56 evaluations
meets connectivity AND3%-12% mass budget; all final masses increase versus F1.
Retain research candidate, promote no model. Complete256/56 evidence rescored,
eight final rollouts and four final-update recovery checks exact.

A long system delay exposed a timeout-accounting defect; parent study remains
interrupted254/54. Linked completion runs only the last two updates and preserves
all imported hashes and failure records. Timing protocol violated, not a clean
performance benchmark. Explicit elapsed checks and regression added after study.
Next GROWTH_STABILITY_PLAN.md: inspect horizon/firing stability and objective
tradeoffs before selecting one schedule/pool or other intervention. Keep budgets,
architecture and nine families until evidence warrants an isolated change.


## H1 completed; mixed training horizons next - 2026-09-23

Read GROWTH_AUDIT_FINDINGS.md/D044. Frozen-model audit:180 fields, no joint
connectivity/budget success, eight new/12 reused gradient cases, no optimizer.
F2 connections persist across sampled horizons but mass exceeds budget. Long
rollouts expose a useful material-reduction gradient. Verification and168-test
regression pass. Next HORIZON_TRAINING_PLAN.md isolates alternating16/50 training:
implement schedule, prove F2 parity and mixed-horizon restart, profile/freeze
compute caps before the proposed64-update comparison. Hold objectives, recipes,
scenes and architecture fixed. This is not yet training or a model promotion.

## F3 completed; access-gradient recovery diagnostic next - 2026-09-23

F3's64-update mixed16/50 schedule reduces material but loses all59 matched F2
connections. No joint success in120 unique evaluations. Read
HORIZON_TRAINING_FINDINGS.md/D046.172 core tests pass; all376 saved fields and
260 checkpoint cursors verified, final rollouts and trained-state restart exact.
Do not promote F3 or automatically increase training duration/grid size.

Next ACCESS_RECOVERY_PLAN.md: actual gradients at failed bottlenecks, then a
research-only pre-clamp maximin access-loss extension on frozen fields. Selected
critical raw material is negative in51/72 finals, exactly0 in21; distinguish
measured clipping from unmeasured parameter gradients. Preserve binary semantics,
budgets, architecture and9 families. New learning needs its own parity/recovery/
timing gates; larger grids, fresh holdouts and production UI remain later work.


## A3 diagnostic execution - 2026-09-23

Follow RAW_ACCESS_AUDIT_PROTOCOL.md/D047. Run complete regression, preserve source
commit, then pilot and verify it. Full audit is conditional on timing admission
and exact source/config equality. Interpret actual parameter gradients and local
probes before proposing any new learning. Update findings/RESUME and archive all
evidence including failures. Current task adds no optimizer updates.


## A3 completed; F4 preparation next - 2026-09-23

Read RAW_ACCESS_FINDINGS.md/D048: semantic checks231/231 pass; parameter signal
restored6/8 with6/6 favorable access probes, but no repaired connectivity and
two persistent zero states. Preserve partial success and material conflicts.
Next RAW_ACCESS_TRAINING_PLAN.md proposes one access-family change against
constant16 F2, requiring parity, true-loop recovery and timing gates. No new
training has occurred here and no model is promoted. Larger grids, fresh-scene
validation and production Studio remain later work.


## F4 execution - 2026-09-23

Follow RAW_ACCESS_TRAINING_PROTOCOL.md/D049: regression and source freeze, F4B
parity, F4R recovery, F4P timing pilot, conditional bounded F4 then F4L. Preserve
all outputs and judge joint connectivity/budget, not raw-loss improvement.


## F4 complete; persistent conditioning proposed - 2026-09-23

F4 full20260923T202649Z_75034cca563c completed1321.78s within frozen caps.
Raw access gains3 connections, loses0:62/72 versus59/72. Both arms0/72 joint
and0/72 in-budget. Full verifier and early/trained-state recovery passed;
187 regression tests. Read RAW_ACCESS_TRAINING_FINDINGS.md and D051.

Next implement the bounded PERSISTENT_GUIDE_PLAN.md architecture comparison,
starting with parity/gradient/context-binding tests and a frozen protocol.
Use F4 as an unpromoted research control; original initialization for both arms.
No second loss change or bigger grids. No conditioning training has started.
M4 Studio redesign remains outstanding; F4 is a research milestone, not a UI or
production release. Preserve exact evidence and archive receipts before resuming.


## Finish local investigation with F5 - 2026-09-23

Execute PERSISTENT_GUIDE_PROTOCOL.md/D052 through regression, parity, recovery,
timing pilot and conditional full comparison plus trained-state recovery. Then
close with evidence and a decision: fresh-scene validation if promising, or
representation/NCA-role review if joint connectivity/budget still fails.
No automatic follow-on sequence of loss changes or larger-grid training.


## Local investigation closed after F5 - 2026-09-24

Full20260923T214035Z_566da8c507c0 completed1317.67s,256 updates/120 unique evaluations,
all caps met. Outcome66/72 connected,0/72 budget,0/72 joint.
Verifier checks47 source hashes/376 fields/260 cursors/128 F4 controls/eight final
rollouts. F5L20260923T220548Z_fdd27bb79c92 exact8-update/8-evaluation trained restart.199 tests
passed; frozen source27416f3 unchanged. Added PERSISTENT_GUIDE_FINDINGS,
LOCAL_PHASE_CLOSURE and D053. No further incremental local training queued.
Final local results commit/full archive follows; milestone receipt records
completion. All historical artifacts and private ignored reports preserved.


## Current implementation - 2026-09-24, Studio S1

The bounded F5 investigation is closed; earlier running/checklist notes above are
historical. PLANNER_REFINER_SPEC.md is the next research specification. Studio S1
completes the first M4 product slice: real procedural geometry, editable scene
coordinates, saved revisions/results, fixed views and honest nine-family evidence.
211 tests pass and browser interactions/responsive layout were checked. Read
STUDIO_S1.md and S1-studio-verification.json. M4 overall remains incomplete.

Next S2: versioned durable jobs with actual cancellation and restart visibility;
revision-aware comparison; verified portable import. Establish those contracts
before integrating long learned rollouts. New refinement learning remains gated
on frozen edit tasks, control results, success criteria and compute/recovery approval.


## Current implementation - 2026-09-24, Studio S2 complete

STUDIO_S2.md supersedes S1 job/import limits. Background jobs, real cancellation,
restart visibility, linked retries, revision comparison and verified JSON import
are implemented.224 checks pass; browser round trip caught and resolved a numeric
encoding bug. S2 remains local procedural software, not a trained-model deployment.

Next: SPATIAL_BRIEF.md. Define material versus usable surface/void with illustrated
positive/negative examples and a compatibility audit within the same nine families.
The user's thin-element observation makes this a prerequisite to the proposed
planner/refiner learning task. Do not automatically restart F5 or launch Colab.


## Current implementation - 2026-09-24, SP1 spatial prototype complete

Read SPATIAL_PLATFORM_SPEC.md/FINDINGS.md and D056. Local gallery compares six
saved paired examples with actual plan/section/axonometric geometry.236 checks
pass; positive spatial example and five expected negative/unsupported outcomes.
Historical access/coverage conflict with the clear-space interpretation despite
the positive deck meeting the unchanged material budget.

Next: specify approach/door/interior semantics, then a versioned spatial access
and coverage target audit on saved geometry. Preserve nine families and all old
metrics. No training or arbitrary-scene platform integration is claimed yet.


## Current - 2026-09-24: volumetric target corrected; VA1 complete

D057 supersedes SP1 as the design target. The user wants volumetric form and
spatial voids without prescribed architectural functions. Platform and later
room/shelter interpretations are not adopted. Preserve SP1 as a diagnostic.
Read VOLUMETRIC_AUDIT_FINDINGS.md and VOLUMETRIC_NEXT_PHASE.md. Nine saved-field
probes,248 regression tests, exact recomputation and browser checks pass.

Next: candidate-independent 3D physical opportunity region and versioned
access/coverage/budget contract audit, retaining nine families and old metrics.
No automatic model, representation, grid-size or paid-training change.
