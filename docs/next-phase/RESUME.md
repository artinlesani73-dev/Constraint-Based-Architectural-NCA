# Resume the NCA next phase

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


## In progress: NR3 bounded quality study preparation - 2026-09-25

Read REPAIR_QUALITY_PROTOCOL.md and experiments/configs/NR3-quality.json.
New local package Codex outputs/NR3-Quality-Study is disarmed. No paid/Drive work.
Three proposed independent256-update seed jobs, each600s cap, with verified
local and separately approved Drive backup between jobs. No heldout arrays in ZIP.
Local evaluator seals all three final models before heldout inference. Current
step: focused/full tests, then one extracted8-update CPU driver rehearsal and
parity with NR2 CPU control. Preserve every attempt. Final findings/commit/archive
still pending. Do not upload or execute the quality study merely because built.


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


## In progress: NR2 Colab package readiness - 2026-09-25

Read COLAB_PREFLIGHT_PROTOCOL.md and NR2-readiness.json. Local package is in Codex
outputs/NR2-Colab-Preflight. It is disarmed; no GPU or Drive operation occurred.
New portable session/sampler retains NR1 math, with device-bound checkpoint identity.
Next archived focused/full tests, verified extraction outside the original repo,
then one600-second CPU rehearsal of the actual three-worker package. Preserve every
failure and version; never overwrite a delivered package or bypass its hashes.
After findings/local commit/archive, request any exact notebook Drive save and
GPU execution allowance separately. No256-update training entry point is provided.


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


## In progress: NR1 CPU mechanics - 2026-09-25

Read NCA_REPAIR_CPU_PROTOCOL.md and experiments/configs/NR1-cpu.json. New versioned
repair model/trainer and owned-worker replay pilot are implemented; execution and
verification are pending. Next run archived focused tests and full regression,
then ONLY the frozen600-second two-member CPU pilot. Stop owned workers on timeout;
retain partial checkpoints and all failures. Findings, D074, resume, local commit
and verified archive are required. No Colab/Drive/Studio change is authorized.
Check .local-artifacts/runs and running processes before resuming any interruption.


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



## Current: MS2 complete - 2026-09-25

Read STUDIO_V2_FINDINGS.md, STUDIO_V2_USER_GUIDE.md and D072. New local Studio:
http://127.0.0.1:8001/static/live-v2/index.html . Original /static/live retained.
351 tests pass (regression 20260925T090806Z_eea8af61e5c0), 57 exact preset matches (20260925T091209Z_181221f5be79),
four browser generations plus 64-grid import verified (20260925T092014Z_06e493b0f94b). 44 valid and
13 blocked combinations; 11 presets, only studied settings. No NCA training.
Focused failure 20260925T090307Z_9e687a560fca and browser failure 20260925T091257Z_bce5f23dd4ef are preserved; focused
retry 20260925T090432Z_53e2d71911c5 passes. Exact JSON text transport fixes float-sensitive checksum
roundtrip. Do not normalize or bypass checksums; re-export old records in v2.

No experiment/audit process remains active. Local server exec session 81429
was left running; recheck before assuming it exists. Start with project .venv
Python -m uvicorn deploy.studio:app --host 127.0.0.1 --port 8001. Avoid duplicate
server owners; inspect all three job queues before a planned restart. Browser
deliverable is the new route with a 64-grid volume. No need to rerun completed
tests to resume documentation.

Artifacts: experiments/reports/MS2-*.json, .local-artifacts/runs/<IDs>,
.local-artifacts/testing/MS2, studio-mass-v2 and its sibling job store. Old stores
remain unchanged. Codex outputs/ms2-*.log and ms2-qa retain UI evidence; work/*ms2*.py
helpers preserve implementation, failures, audits, finalization and packaging.
Helpers use exclusive writes; inspect partial progress before any rerun.

Next review the larger volumes, then specify a separate learned NCA baseline
against the procedural comparator. Freeze held-out sites, objectives, checkpoints
and recovery before requesting a concrete Colab budget. Broader presets/seeds and
finer resolution require separate tests. No paid compute or external operations.

Local commit title: "Integrate versioned mass Studio with evaluated scale presets".
Archive outputs/NCA-MS2-Backup-2026-09-25-<commit7>.zip and sibling receipt; local
milestone receipt holds exact hashes. Includes current local Studio state and
retained MS2 fixtures; keep earlier incremental archives. Restore raw source ZIP
for newline-sensitive frozen hashes. Private reports remain ignored/unchanged,
two unrelated user files untracked. No Drive, push or hosting; off-device pending.
Earlier current/in-progress sections below are historical.


## In progress: MS2 - 2026-09-25

User authorized versioned Studio scale integration. Read STUDIO_V2_PROTOCOL and
D072. Implementation helpers in Codex work/prepare_ms2.py and outputs/ms2-*.log.
Preserve all failed attempts; inspect processes/runs before retry. Full regression,
57-case parity, bounded worker/import/recovery checks, browser acceptance, then
findings/commit/archive are required. No Drive or paid compute.


## Current: MG7 complete; conditional Studio integration next - 2026-09-25

The equivalence and paired performance gates both pass.
Read INCREMENTAL_GROWTH_FINDINGS.md, STUDIO_SCALE_INTEGRATION_NEXT_PLAN.md and D071.
343 tests pass, smoke exit 0. Runs:
- regression 20260925T082528Z_d941a7a46957
- diagnostic 20260925T082841Z_88466161ced3
- matrix 20260925T083302Z_6c7ade2f0e16
- timing 20260925T084205Z_6fab76a67538

All stages completed and independently verified. No active experiment or audit
process remains; do not rerun to resume documentation. Matrix: 235 exact full
matches, two preserved prefixes followed by complete audited fields, 188 positive
cases and 49 blocked controls. New continuation audit: 10097 steps,
12131925 candidate evaluations. Every measured timing execution is retained.
Read experiments/reports/MG7-summary.json and MG7-<mode>-verification.json for
precise CPU/wall/RSS outcomes and gate limitations. Old MG6 failures remain valid
historical evidence. MG7 is procedural; no new NCA training or live switch.

Next make the conditional integration proposal concrete. Preserve existing
MS1/MG3 replay; reuse separate cancellable workers; verify supported versions,
scale-aware decoding, bounded imports, deadlines and real browser behavior.
Only tested larger scene/seed/request combinations are candidates for admission.
Arbitrary sites, wider seeds/requests and finer resolution need separate studies.

Artifacts: .local-artifacts/runs/<IDs>. Frozen recipe MG7-incremental.json and
INCREMENTAL_GROWTH_PROTOCOL.md. Runner run_incremental_study.py uses --mode and
--admission-run; never rerun blindly. Codex work helpers prepare_mg7_code.py,
freeze_mg7.py, verify_mg7.py, finalize_mg7.py, package_mg7.py and outputs/mg7-*.log
preserve implementation/audits. Helpers use exclusive writes; inspect first.
Local commit title: "Optimize cube accounting with exact-choice verification".
Archive outputs/NCA-MG7-Backup-2026-09-25-<commit7>.zip, sibling receipt and local
milestone receipt contain verified hashes; keep all earlier archives. Raw source
ZIP is authoritative for frozen newline-sensitive hashes after Git checkout.
Private reports ignored/unchanged; two unrelated untracked user files preserved.
No Drive, paid compute, push or public hosting. Same-disk archive is not off-device.
Earlier current/in-progress sections below are historical records.


## In progress: MG7 exact-choice optimization - 2026-09-25

Read INCREMENTAL_GROWTH_PROTOCOL, MG7-incremental config and D071. Separate module
implemented; six synthetic tests and all 343 regressions pass. Regression run
20260925T082528Z_d941a7a46957. Diagnostic 20260925T082841Z_88466161ced3 passes
all six cases and independent verification (MG7-diagnostic-verification.json).
Matrix 20260925T083302Z_6c7ade2f0e16 passes all 237 cases and independent audit:
235 full matches, two prefixes, 10097 new continuation steps / 12131925 proposals.
Timing 20260925T084205Z_6fab76a67538 is running; inspect its state and Codex
outputs/mg7-timing.log before any retry. Next independently verify timing with
work/verify_mg7.py <timing ID> <regression ID>, then finalize_mg7.py <regression>
<diagnostic> <matrix> <timing>. No source/protocol changes after outcomes.
Preserve all failures/old source. Finish findings, local commit and verified archive.


## Current: MG6 complete; exact-output efficiency proposal next - 2026-09-25

Read SCALE_STUDY_FINDINGS.md, SCALE_EFFICIENCY_NEXT_PLAN.md and D070.
The48-grid gate passed; the64-grid gate failed and larger live work remains unadmitted.
Regression 20260925T075016Z_c82729a145e4:337pass,smoke0,181.934s. Studies 20260925T075350Z_c677671d8310, 20260925T075945Z_3fc0d812e153.
7/8 nonpartition MT1 pass;
6/8 requests met,2 timeouts;
4 blocked controls retained.
Source/scene/config frozen in experiments/configs/MG6-scale.json. No scientific
reroll. Full independent reports experiments/reports/MG6-<size>-verification.json
and MG6-summary.json contain case results, units, timing, memory and exact-replay
counts. No active regression/generation/replay process. Do not rerun to resume docs.

64 offset/seed7 returnedtime_limit after234.526s wall despite120s setting, with
38.578s CPU including evaluation. Cause of large wall/CPU gap not established;
do not interpret it as pure compute cost or hide the cooperative-limit overrun.
Both timed-out fields/traces are retained; exact replay covers completed non-timeout
cases, while timed-out results receive recorded-prefix audits only.

The48/64 studies use0.8m/cell: actual wider sites. Exact translated-site embedding
is a separate representation control, not generation equivariance. Same2.4m
cubes,1.6m interfaces,4.8m ground band,24% request,seeds6/7. New context helper
sets scene-sized config explicitly; historical weights unused. Finer resolution
still requires separate interface/ground/mask resampling and is not admitted.

Next freeze exact-output count-cache optimization and paired timing admission.
Keep MG5 as immutable comparator, all old failures, and fixed stopping/coverage/
facade rules. Do not edit nca/coverage_mass_generator.py in place. See next plan
for bounded equivalence tests before broader comparison or live integration.

Run artifacts .local-artifacts/runs/<IDs above>: scenes, lossless NPZ masks/fields/
routes/bulk, JSON traces/results, source/protocol/config and immediate events.
Runner scripts/run_scale_study.py --size 48 and
scripts/run_scale_study.py --size 64 --admission-run <48 run> preserve the gate.
Only justified new attempts
may use --parent-run; inspect existing results first. Old frozen hashes include
line endings; restore raw source ZIP rather than rewriting hashes after checkout.

Codex cwd work/prepare_mg6.py,verify_mg6.py,finalize_mg6.py,package_mg6.py and
outputs/mg6-*.log/mg6-summary.json preserve implementation/audit work. Helpers
use exclusive writes: inspect before rerun. All runtime tests complete before
documentation-only edits; no repeat needed. Local commit title:
"Evaluate bounded physical-scale mass generation". Archive
outputs/NCA-MG6-Backup-2026-09-25-<commit7>.zip and sibling receipt; local milestone
receipt records hash/payloads and restored Git bundle. Preserve MG5/MG4 and prior
archive chain; this remains same-disk storage, not off-device backup.

Live MS1/MG3 unchanged. No server restart/browser action this milestone; previous
PID29824/session23049 is historical and must be rechecked before acting. No paid
compute, NCA training, Drive, push or hosting. Private reports unchanged/ignored;
two unrelated user files remain untracked. Current status supersedes history below.

## Current: MG5 complete; scale-readiness proposal next - 2026-09-24

Read COVERAGE_GROWTH_FINDINGS.md, SCALE_READINESS_NEXT_PLAN.md and D069.
Regression 20260924T170440Z_05e362daa48f:332pass,smoke0. Diagnostic 20260924T170732Z_b7d16f22121a:4/4pass; independent gate
verified before matrix. Matrix 20260924T170831Z_264ba0e38124:225 complete,180/180 nonpartition pass,
45 blocked fail;179 prior positives preserved.221 fields and all225 routes match
baseline. One repair: combined_reverse16%,seed5, west293/3544=8.267494% vs previous
266/3544. Exactly1503 voxels requested/produced;facade14.903526%. Three other valid
fields change slightly and remain valid. Original MG4 failure remains preserved.

No errors/timeouts/resource breaches. All180 request errors0..8,100% bulk. Sampled
RSSmax345.57MiB. Nonpartition generation median0.720s,p952.914s,max4.244s;
study328.981s. These are single-run observations, not matched timing proof.
60 valid diversity groups retain3 distinct seeds each;15 blocked groups have0.
Two same-population groups show small distance decreases; repaired group has
different valid pair counts. See findings for all details and limitations.

Independent225 exact replays,450 scores,225 bulk masks,25 rebuilt contexts,
105139 decisions/31392998 directly enumerated candidate evaluations and157 Python
matches per snapshot verify. Diagnostic separately4 replays,8 scores,2 contexts,
2149 decisions/589073 evaluations. Reports experiments/reports/MG5-*-verification.json.
No active regression/matrix/replay process. Do not rerun just to resume docs.

Next make the physical-unit/context audit and small48/conditional64 resource
study concrete and freeze it before execution. Do not conflate larger physical
environments with finer resolution. MG5 runner intentionally handles fixed32
saved scenes; use a separately versioned size-aware path for scale admission.
Fresh-site evaluation, live promotion and learned pilot remain separate decisions.
No paid compute, new constraints or threshold retuning admitted by this result.

New nca/coverage_mass_generator.py; old budget_mass_generator.py and every prior
evaluator remain unchanged. Live MS1 still uses MG3 at /static/live/index.html;
no UI/runtime edit or restart occurred. Last known serverPID29824/session23049 is
historical: recheck process/queue ownership before any later restart. Added MG5
module is unused by the already running server. New source hashes are frozen in
experiments/configs/MG5-coverage.json. Byte-exact hashes include line endings;
restore retained raw source, do not rewrite hashes after Git newline conversion.

Run evidence: .local-artifacts/runs/<three IDs above>. Every matrix case contains
both previous and current fields, masks, scores, routes, decisions and resources.
Distinct source snapshots/protocol/config/events retained; all225 case files,
25 contexts,no pending. Scripts/run_coverage_generation.py --mode diagnostic or
--mode matrix --diagnostic-run 20260924T170732Z_b7d16f22121a are reproducibility entry points;
execute again only for justified linked attempts with --parent-run <prior ID>.

Codex cwd work/prepare_mg5.py,verify_mg5.py,summarize_mg5.py,finalize_mg5.py,
package_mg5.py and outputs/mg5-*.log,mg5-summary.json retain implementation/audit
work. Helpers use exclusive writes; inspect before rerun. No scientific execution
failed; one bare-python helper preparation command was corrected to project Python.
All runtime tests passed before documentation-only edits; no repeat needed.

Local commit title: "Prioritize underfilled regions during mass growth".
Verified incremental archive outputs/NCA-MG5-Backup-2026-09-24-<commit7>.zip;
sibling receipt and .local-artifacts/milestones/<commit7>-mg5-backup-receipt.json
record hash/bytes/payload count and exact restored Git bundle. Preserve MG4/MS1/
MG3/MG2 and the earlier archive chain. Same-disk only, not off-device protection.
Private report files unchanged/ignored; unrelated user files remain untracked.
No Drive operation, training, paid compute, push or public hosting.

## Current: MG4 complete; one coverage failure retained - 2026-09-24

Read SITE_GENERALIZATION_FINDINGS.md, COVERAGE_GROWTH_NEXT_PLAN.md and D068.
Benchmark 20260924T163831Z_b4e28646a3e0:180 completed,143/144 nonpartition pass,36 blocked
fail,143/180 overall. All144 requests met with0–8 extra cells and100% bulk. No
execution errors/timeouts/resource breach. Strict144/144 scale gate FAILED.
Do not alter the original result, add sites to live Studio or begin larger grids.

Failure combined_reverse__v16__s5:1507 cells for1503 request; west266/3544=7.505643%
vs8%,18-cell arithmetic deficit. Other8 families pass, facade14.930325%. Initial
route45/42/210 west/mid/east; final266/303/938. Last block adds7 cells and total
count stops growth. Seeds3/4 pass on the same site/request. No reroll/repair done.
See MG4-coverage-diagnosis.json; coverage remains a provisional fixed-X-third proxy.

Regression 20260924T163521Z_bd0b9d7571b7:324pass, smoke0.180 exact replays/scores/bulk masks,
20 rebuilt contexts,718654 growth decisions and154 Python source matches
per snapshot verify. Read experiments/reports/MG4-verification.json. No active
regression/benchmark/replay process. Generation median0.218s,max2.254s; sampled
RSS peakmax309.32MiB,lifetimepeak452.63MiB; single-run32-grid observations only.

Next make a separately versioned coverage-aware finite growth priority concrete,
with existing facade admission and unchanged request bound. Freeze a small paired
diagnostic before executing, then conditional225-case MG3/MG4 regression proposal.
MG5 is not implemented/frozen yet. No whole-project/NCA redesign or paid training
admitted. Larger environments and finer resolution remain separate future studies.

Runs .local-artifacts/runs/<IDs> contain complete sources, scenes, traces, fields,
metrics/timing/resources and events.180 case files,20 contexts,no pending. New
execution only when justified; scripts/run_site_generalization.py --parent-run
20260924T163831Z_b4e28646a3e0 creates a linked attempt. Do not rerun just to resume docs.
Core/source hashes frozen in experiments/configs/MG4-sites.json; no nca/deploy
runtime change. Frozen hashes are byte-exact, including line endings: restore
the retained source ZIP or raw repository archive for replay after a fresh checkout;
do not rewrite frozen hashes to accommodate Git line-ending conversion.
Live Studio retains five supported sites and existing records;
last known server PID29824/session23049, recheck before any future restart.

Codex cwd helpers work/:prepare_mg4,verify_mg4,summarize_mg4,diagnose_mg4,
finalize_mg4,package_mg4.py; exclusive writes, inspect before rerun. Summary
outputs/mg4-summary.json and focused log outputs/mg4-focused.log. One inline
diagnostic syntax error corrected without changing experiment outputs.
Local commit title: "Evaluate mass generation across twenty new sites".
Archive outputs/NCA-MG4-Backup-2026-09-24-<commit7>.zip and sibling receipt;
.local-artifacts/milestones/<commit7>-mg4-backup-receipt.json confirms verification
and restored Git bundle. Preserve MS1/MG3/MG2 and earlier archive chain. Same-disk
only; private reports unchanged/ignored; unrelated user files remain untracked.
No Drive, paid compute, push or public hosting.


## Historical MG4 start note - 2026-09-24 (completed above)

Read SITE_GENERALIZATION_PROTOCOL.md, MG4-sites config/scenes and D068.180 planned
candidates, unchanged generator/evaluator. No output inspected at freeze. Runner
and checks pending. Inspect .local-artifacts/runs before retrying; preserve all
failed/interrupted evidence. No Studio change, paid compute or Drive access.


## Current: MS1 live mass Studio complete - 2026-09-24

Read STUDIO_MASSING_FINDINGS.md, STUDIO_MASSING_USER_GUIDE.md and D067.
URL http://127.0.0.1:8001/static/live/index.html. Real procedural mass generation,
fixed five sites,16/24/32% requests, bounded integer seeds. Automatic saving,
durable cancel/retry/restart, nine-family evidence, compare and typed import/export
with replay/source verification. Old material workflow remains at /.

Regression 20260924T144654Z_f7338d28b9ac:320pass, smoke0. New focused tests10pass.
Acceptance 20260924T145347Z_03b4d20a69f7:3 browser-generated records exactly match MG3;
2 positive partial cases and1 blocked failure.273 added/0 removed between24/32%.
All152 Python files plus4 new UI/data files match regression snapshot; all job
source archives and portable exports verify. No active regression/test process.

Records/job evidence: .local-artifacts/studio-mass and studio-mass-jobs.
Live IDs20260924T144914Z_800512d53ad5,20260924T144941Z_e60cce65a38e,
20260924T145002Z_6fbc4db10ce8. All terminal. Four legacy jobs remain terminal.
Source fixture deploy/mass_contexts.json is MG3's exact five contexts; no runtime
dependency on archived runs. Worker snapshots include the new live UI/data.
Source edits require restart before new submissions. Retry keeps all parameters.

Local server restarted after idle check; last known PID29824/session23049,
python -m uvicorn deploy.studio:app --host127.0.0.1 --port8001 --no-access-log.
Recheck identity/queues before stopping; never act on saved PID alone.
First Stop-Process failed; first replacement refused existing store lock. Runtime
termination of verified old PID20784 then successful startup; no evidence deletion.
Browser bindings agent/browser/tab2/viewport survive if session does; viewport
reset and deliverable marked. QA outputs/ms1-qa. API verified import/export; OS
file picker not automated. No Drive, paid compute, push or public deployment.

Next: user reviews live alternatives. Plan and freeze an unseen-context evaluation
separate from a larger-grid resource/scaling study, with explicit physical units,
seeds, caps, request fidelity/diversity metrics and all nine families unchanged.
Do not immediately train or silently loosen the existing site/grid bounds. No new
scientific run is frozen yet; inspect findings before choosing the next experiment.
Do not rerun unchanged regression/acceptance just to resume documentation.

Codex cwd helpers: work/implement_ms1.py,build_ms1_ui.py,verify_ms1.py,
finalize_ms1.py,package_ms1.py. Exclusive writes; do not blindly rerun. Helpers
are authoring history, while committed source and job snapshots are exact state.
Local commit title: "Add live building-mass generation to Studio".
Archive outputs/NCA-MS1-Backup-2026-09-24-<commit7>.zip; completion verified by
sibling receipt and .local-artifacts/milestones/<commit7>-ms1-backup-receipt.json.
Preserve MG3/MG2/MD1/MO1/MG1 and older archives. Same disk, not off-device backup.
Private report hashes unchanged/ignored. Two unrelated user files remain untracked.


## Historical MS1 start note - 2026-09-24 (completed above)

User authorized STUDIO_MASSING_INTEGRATION_PLAN. D067 records scope. Backend and
new mass page under construction. Do not restart old server until source/tests
are coherent and active jobs checked. Full regression/browser QA/archive pending.
Codex helper work/implement_ms1.py creates files exclusively; inspect before rerun.


## Current: MG3 complete; live mass integration next - 2026-09-24

Read CONTACT_BUDGET_FINDINGS.md, STUDIO_MASSING_INTEGRATION_PLAN.md and D066.
Run20260924T140750Z_5d36c2bf3bdc:36/45 pass,36/36 nonblocked. Nine blocked failures retained.
Both MG2 failures repaired;43 fields and all45 initial routes unchanged. No MG1
or MG2 regressions. All36 requests met (0–8 cells extra),100% bulk. Frozen gate met.
This is procedural generation; no new NCA training or live integration yet.

Regression20260924T140546Z_7d07393758df:310pass, smoke0. Independent45 exact replays,135 scores,
135 bulk masks,16,873 growth decisions and150 Python snapshot matches verify.
Regression ran before benchmark. No execution failure, reroll or timeout.
All case fields, contexts, source, route and proposal traces retained under
.local-artifacts/runs/<run-id>. Both experiment records and verification reports
are tracked. No benchmark, regression or verification process remains active.

Next implement the versioned experimental mass mode per the integration plan:
inspect existing durable jobs/schema, bind requests/results and preserve old
material records, add real jobs and independent evaluation, UI save/compare,
typed import/export and replay, then relevant tests/browser QA. Do not rerun
unchanged MG3 science just to resume. Before any new attempt, inspect records;
justified retries use scripts/run_budget_generation.py --parent-run 20260924T140750Z_5d36c2bf3bdc.
Do not start paid training, access Drive or silently use legacy material metrics.

Gallery http://127.0.0.1:8001/static/budget/index.html,45 exact MG2/MG3 pairs.
Desktop/mobile, all selections and slice/layer controls checked. Browser bindings
agent/browser/tab(id2)/viewport exist if session survives; viewport reset. No server
restart this milestone. Existing server continues old live material workflow.
Helpers in Codex cwd work/:implement_mg3,prepare_mg3,verify_mg3,build_mg3_gallery,
finalize_mg3,package_mg3.py. Exclusive writes: inspect before rerunning.
QA outputs/mg3-qa. Some gallery refinements were applied after its initial builder;
use archived gallery-source.zip or committed files for exact recovery.

Local commit title: "Enforce facade contact budget during mass growth".
Archive outputs/NCA-MG3-Backup-2026-09-24-<commit7>.zip; check sibling receipt
and .local-artifacts/milestones/<commit7>-mg3-backup-receipt.json for completion.
Packaging verifies each payload and restores bundled Git HEAD. Preserve all MG2,
MD1,MO1,MG1 and earlier archives. Same disk, not off-device backup. Private reports
stay unchanged and ignored; two unrelated user files remain untracked. No push.


## Historical start note: MG3 budgeted growth - 2026-09-24 (completed above)

User authorized CONTACT_BUDGET_NEXT_PLAN. New separate budgeted_contact_growth_v1,
seven focused tests pass. Read CONTACT_BUDGET_PROTOCOL and MG3-budget.json. Old
generators/evaluator unchanged. Full regression next, then45-case fixed comparison
via scripts/run_budget_generation.py. Inspect live run records before retrying.
All growth decisions saved; deferred candidates reconsider only after positive
growth, zero-delta origins expand once, exhausted feasibility ends explicitly.
No model training, live integration, paid compute or Drive access yet.

## Current: MG2 complete; live promotion held - 2026-09-24

Read CONTACT_GENERATION_FINDINGS.md, CONTACT_GENERATION_PROTOCOL.md and D065.
Run20260924T133657Z_f4c92840c7c0:45 completed,34 pass vs27 MG1, no baseline regressions.
All27 earlier positives preserved;7/9 partial cases repaired;9 blocked remain
failed. Partial24%seed2 and32%seed2 fail facade at15.018315% and16.559927%.
All36 nonblocked request targets met with0–8-cell overshoot; all have100% bulk.
Frozen Studio gate36/36 not met: no live integration, threshold change or reroll.

Verification:45 exact generator/route/trace replays,90 scores and90 bulk masks,
four MD1 matches, diversity exact,147 Python hashes match both snapshots.
Regression20260924T133923Z_4b6c61aedbb5:303pass, smoke0. Benchmark25.689s; later verification
and regression ran concurrently. No failed execution or scientific source edits.
All per-case fields, contexts, original comparisons and failures are in immutable
run directory. experiments/reports/MG2-verification.json retains trace diagnosis.

Both failures begin with completed198-cell routes at7.070707% contact; growth
exceeds the global ratio. Next proposal is CONTACT_BUDGET_NEXT_PLAN.md: separate
MG3 growth admission using actual new-cell contact accounting, bounded deferred
frontier reconsideration and explicit stalls. Same limits/cost, not another weight
sweep. First verify accounting and finite failure, then frozen45-case comparison.
Live Studio integration remains conditional; no training or paid compute admitted.

Gallery http://127.0.0.1:8001/static/contact/index.html.45 pairs, failed cases and
three-decimal contact ratios. Desktop/mobile and slice/layer controls checked.
New browser session restored after prior MD1 timeout; bindings agent/browser/tab2
and viewport available. No server restart; current server still serves old live
material-scaffold workflow. No benchmark/regression from this milestone is active.

Codex cwd helpers: work/verify_mg2.py,build_mg2_gallery.py,finalize_mg2.py,
package_mg2.py. Exclusive writes: inspect before rerunning. QA outputs/mg2-qa.
New attempt only when justified: scripts/run_contact_generation.py --parent-run
20260924T133657Z_f4c92840c7c0. Do not rerun unchanged science just to resume documentation.

Local commit title: "Evaluate contact-aware generation across the full massing matrix".
Archive outputs/NCA-MG2-Backup-2026-09-24-<commit7>.zip and sibling receipt; local
milestone receipt verifies payloads and restored Git bundle. Preserve MD1/MO1/MG1
and older archive chain. Same disk, no off-device backup. Private reports unchanged
and ignored; two unrelated user files untracked. No Drive, paid compute or push.


## In progress: MG2 full matrix - 2026-09-24

User authorized continuing. Read CONTACT_GENERATION_PROTOCOL.md and MG2-contact.json.
Generator/evaluator unchanged; new runner uses exact saved MG1 inputs and cost12.
Studio gate frozen before results: all36 open pass, no27-baseline regressions,
request met with <one cube overshoot, no timeout; retain all9 blocked outputs.
Run scripts/run_contact_generation.py, verify all outputs before conditional
Studio integration. Inspect latest run records/processes before a retry.


## Current: MD1 comparison complete - 2026-09-24

Read MASSING_DIRECT_FINDINGS.md, MASSING_DIRECT_PILOT_PLAN.md and D064.
Run20260924T123453Z_aaf2d7ad37e3: all four32-update direct members completed, primary success FALSE.
Contact-aware procedural4/4 pass; original/direct-final2/4. Both partial failures
are facade-only and remain unchanged by direct optimization. All20 saved direct
binary boundaries exactly equal their originals; soft probabilities/loss changed.
Do not describe this as a successful optimizer or trained-model result.

Regression20260924T123246Z_90e81143f1df:303pass, smoke0. Fresh-process4 vs2+2 recovery exact
for whole checkpoint/evaluation/probabilities. Timing admission passed;47.051s
pilot,66.880s study before final attachment/finalization. Four procedural replays,
20 checkpoint re-evaluations, four original rescores and146 source Python hashes
verify. No scientific failed attempt or hidden reroll. All evidence retained.

Next: freeze MG2 full45-case procedural comparison using same contact cost12,
MG1 scenes/requests/seeds and MT1. Audit baseline regressions and blocked cases;
do not launch another weight/step sweep or NCA training. Then integrate mass
generation into Studio's live workflow if validated. Existing Studio root still
constructs historical material scaffolds; new gallery is saved evidence only.

Gallery http://127.0.0.1:8001/static/direct/index.html.24 combinations and responsive
layout checked. Browser automation session reset on final link/viewport timeout;
inspect tabs/reconnect before using old bindings. Local server not restarted.
No experiment or regression process from this milestone remains running.

Run artifacts: .local-artifacts/runs/20260924T123453Z_aaf2d7ad37e3/, source.zip, original input-study,
per-member checkpoints/probabilities at0/8/16/24/32, updates and comparisons.
Verification: experiments/reports/MD1-verification.json. QA in Codex cwd
outputs/md1-qa. Helpers work/verify_md1.py,build_md1_gallery.py,finalize_md1.py,
package_md1.py; exclusive writes, inspect before rerun. Never overwrite old runs.
New linked attempt only if justified: scripts/run_massing_direct.py --parent-run
20260924T123453Z_aaf2d7ad37e3. This automatically redoes recovery/profile before the pilot.

Local commit title: "Compare direct mass optimization with contact-aware generation".
Archive outputs/NCA-MD1-Backup-2026-09-24-<commit7>.zip, sibling receipt and local
milestone receipt; verify receipt before claiming backup. Preserve MO1/MG1/R2/MT1
and all older parent archives. Same disk, not off-device. Reports stay ignored;
two unrelated user files stay untracked. No Drive, paid compute, push or hosting.


## In progress: MD1 gated comparison - 2026-09-24

User authorized proceeding after MO1. Read MASSING_DIRECT_PILOT_PLAN and
MD1-direct.json. New direct_massing_v1 session, contact_cube_route_growth_v1
control and bounded worker runner are implemented. Six focused tests pass.
Next full regression, then scripts/run_massing_direct.py performs fresh-process
recovery/timing gates and only then the four-case32-update comparison. Inspect
run records and live processes before rerunning. All artifacts append-only; failed
attempts get parent-linked retries. No model training, viewer change or paid compute.


## Current: MO1 objective admission complete - 2026-09-24

Read MASSING_OBJECTIVE_FINDINGS.md/PROTOCOL.md, MASSING_DIRECT_PILOT_PLAN.md and D063.
User liked MG1 geometry; preserve it as the baseline. MO1 audit20260924T115136Z_91b3707a89d3:
97 fields,873 family agreements, zero mismatches,97 exact bulk masks, two finite
gradient probes;33.62s including archive/setup. No optimizer updates. Full regression
20260924T115046Z_0a0c9bfc3d10:297pass, smoke0. Both ran concurrently on unchanged science source.
142 Python files match both snapshots; independent facade derivative formula verifies.

Next implement MD1's four-member direct-massing session, iteration0 MG1 parity,
actual-loop new-process recovery and timing admission, plus separately versioned
contact-aware procedural control. Plan declares candidate coefficients/volume policy;
no optimization result or optimality claim. No threshold change or extra constraints.
Source/evidence: experiments/reports/MO1-verification.json and immutable run dirs.
New audit attempt only: `.venv/Scripts/python.exe scripts/run_massing_objective_audit.py
--parent-run 20260924T115136Z_91b3707a89d3`. Full regression parent is20260924T115046Z_0a0c9bfc3d10.
No reason to repeat either merely to resume documentation.

No benchmark/regression process from this milestone remains active. Server/gallery
were not changed or restarted; inspect live processes before future work. Private
reports remain ignored; original and Revision2 hashes verify unchanged.
Local commit title: "Add verified massing objective and bounded direct-pilot plan".
Archive outputs/NCA-MO1-Backup-2026-09-24-<commit7>.zip with sibling receipt, under
Codex cwd. Helpers work/mo1_progress.py,verify_mo1.py,finalize_mo1.py,package_mo1.py;
exclusive writes require inspection before rerun. Archive is incremental/same-disk;
retain all previous backups. No paid training, Drive access, push or publication.

## In progress: MO1 massing objective admission - 2026-09-24

User authorized direct-optimization preparation after liking MG1 forms. Read
MASSING_OBJECTIVE_PROTOCOL.md / MO1-objective.json. Implemented separate CPU
massing_residuals_v1 and ten tests. Focused10tests pass. Full regression
20260924T115046Z_0a0c9bfc3d10 is running in session29145. MO1 endpoint/gradient
audit may be active; inspect latest run records and processes before restarting.
All source code frozen during the two read-only scientific checks. No optimizer
updates, coefficients or learned model selected; old objectives and gallery intact.
Audit saves97 endpoint comparisons and two derivative probes with explicit caps.
Next depends on results: retain mismatches, or prepare a bounded direct pilot with
recovery/timing gates. No automatic paid compute, Drive, publication or push.

## Current: MG1 procedural comparison complete - 2026-09-24

Read MASS_GENERATION_FINDINGS.md/PROTOCOL.md, RESEARCH_BRIEF_R2.md and D062.
Run20260924T113023Z_24298393f2a5:45 generated requests,4 separate challenges, all retained;
27/45 generator outputs meet unchanged MT1 pilot checks. See per-context/request
breakdown before interpreting the denominator. No NCA training or all-site claim.
Final regression20260924T112636Z_ea40cf77f2f4:287pass, smoke0. Two parent regression errors
were test-fixture metadata handling, preserved with source; generator unchanged.
Verification replays45 routes/final fields and rescored49 fields exactly.

Gallery http://127.0.0.1:8001/static/generation/index.html uses saved study data,
not arbitrary-scene generation. Existing server was not restarted. Inspect live
processes before new compute. No active benchmark/regression is left by completion.
Next: R2-B continuous-objective specification and endpoint/gradient tests before
bounded direct-optimization comparison; preserve failed procedural outcomes and
provisional MT1 limitations. Do not change acceptance thresholds to raise pass rate.

New linked attempts only: `.venv/Scripts/python.exe scripts/run_mass_generation.py
--parent-run 20260924T113023Z_24298393f2a5`; this never publishes viewer data automatically.
Latest full regression: `.venv/Scripts/python.exe scripts/verify_foundation.py
--parent-run 20260924T112636Z_ea40cf77f2f4`. Do not rerun just to resume.
Evidence: experiments/reports/MG1-verification.json, immutable run directory and
Codex cwd outputs/mg1-qa. Helpers in Codex cwd work: verify_mg1.py, finalize_mg1.py,
build_mg1_gallery.py, package_mg1.py. Some writes are exclusive; inspect before retry.
Local commit title: "Add procedural building-mass alternatives and audited MG1 comparison".
Archive outputs/NCA-MG1-Backup-2026-09-24-<commit7>.zip; sibling receipt JSON.
Retain all earlier archives; this is incremental and same-disk, not off-device.
Reports remain ignored; unrelated two user files untracked. No Drive, paid compute,
remote push or publication.

## In progress: MG1 procedural mass alternatives - 2026-09-24

User authorized R2-A. Read MASS_GENERATION_PROTOCOL.md and MG1-procedural.json.
Implemented cube_route_growth_v1, five development contexts, 45 requests and four
separate evaluator probes. No model, old losses or MT1 thresholds changed.
First regression 20260924T112005Z_e9860ed79bdf: 287 tests, one test-fixture error
(copy called on text metadata), zero failures and smoke pass. Fixed test via
deepcopy; preserve failed attempt. Linked regression 20260924T112334Z_588b73843c6d
running in session60580; inspect result before benchmark. Planned command:
`.venv/Scripts/python.exe scripts/run_mass_generation.py`.
Study writes immutable evidence and leaves viewer publishing separate. New static
generation gallery source exists but data/browser QA remain pending. Helper
build_mg1_gallery initially assumed inline CSS and failed before writing files;
corrected shared stylesheet references. Process command-line inspection via CIM
was denied; Get-Process succeeded, existing browser tabs remain available.
No server restart, paid compute, Drive action or push. Completion entry will
supersede this note; inspect current processes/runs before retries.

## Current: review revision R2 adopted - 2026-09-24

Read RESEARCH_BRIEF_R2.md and D061 first, then MT1 protocol/findings below.
The user approved aligning the original review with measured findings and D058's
building-mass meaning. Private Revision 2 markdown/PDF dated 2026-09-24 supplement
the unchanged original files and remain Git-ignored. R2-A procedural mass generation
and a finite benchmark specification are next; no new generator/training started.
R2-B direct optimization and R2-C conditional learning follow explicit gates.
Do not resume older material/platform/void-first plans as current instructions.

Documentation-only milestone: inspect final PDF and integrity receipt under Codex
cwd outputs/review-r2-20260924. Local commit title: "Align research roadmap with
building-mass findings and review revision R2". Archive after commit:
outputs/NCA-Review-R2-2026-09-24-<commit7>.zip, with sibling receipt JSON.
Archive includes changed tracked docs, private originals/addendum, source helper,
PDF QA and commit patch; retain MT1 and all parent experiment archives separately.
Same-disk archive only. Original report hashes recorded in creation-evidence.json.
Latest science tests remain MT1's 275 passes; do not rerun for documentation.
No runtime inspection/restart, paid training, Drive operation, push or publication
in this revision. Inspect live processes before resuming compute or server work.

## Current: MT1 pilot massing contract audited - 2026-09-24

Read MASSING_TARGETS_PROTOCOL.md, MASSING_TARGETS_FINDINGS.md and D060.
D058 building-volume semantics remain current. MT1 adds a separate binary
nine-family evaluator, not a differentiable loss or a trained generator.

Audit20260924T102755Z_e89550a24d8d:4 contexts,48 controls,432 sensitivity reports,
96checks pass,93.23s CPU. Six intended positive controls pass;42 intended negative/
blocked-context outcomes reject. No failed attempt or post-outcome retuning. Final review fixed source-component
selection in context feasibility; initial audit20260924T101444Z_3d8c5504cb9e and
regression20260924T101120Z_6d8b5d1cc577 are preserved. All original48 base
records and432 sensitivity reports are identical after the correction.
Regression20260924T102553Z_4ae3c1b346f2:275pass,0fail/error/skip,smoke0,104.92s.
Fifteen new tests. All135 relevant Python files match tested/audited snapshots.
Run manifests/source hashes, bulk masks/counts and base/sensitivity parity verify.
Old SP1/VA1/MA1 study data, losses and model/checkpoint remain unchanged.

Pilot parameters:8-40% all-occupancy/fixed-domain volume;2.4m local cube scale;
90% bulk-qualified fraction;8% substantial volume per fixed X third. Access
requires both raw and substantial interface-connected mass, with no detached
occupied parts hidden. See protocol for every family and known limitations.
The25% budget sensitivity excludes offset compact mass (26.6075%); articulated
alternative23.5403% remains acceptable. Cube scale1.6-3.2m did not change verdicts
on these examples. These thresholds are experimental, not architectural standards.

Gallery http://127.0.0.1:8001/static/targets/index.html. All48 selector combinations,
display modes, desktop and390px mobile checked. Static viewer data is a verified
compact projection:1,029,993 bytes versus18,147,261-byte full run study; every
displayed record is retained exactly. Full context masks remain in immutable run.
Existing server PID20784/session91597 continues; no restart. No audit/test/training
process active. Inspect process state before restarting or launching another run.

Next bounded work: parameterized procedural mass generation with saved alternatives,
and a separately versioned continuous objective/direct-optimization baseline.
Check binary agreement and gradients before fitting. Use MT1 for independent final
geometry evaluation. Challenge appendages, lattice-like masses and rotated forms;
freeze comparable seeds/scenes/volume ranges and compute caps. Do not start another
unbounded loss-tweaking sequence, NCA training or larger-grid/paid Colab run by default.

Reproduction (only when a new linked attempt is needed):
`.venv/Scripts/python.exe scripts/run_massing_targets.py --parent-run 20260924T102755Z_e89550a24d8d`.
Regression command: `.venv/Scripts/python.exe scripts/verify_foundation.py --parent-run 20260924T102553Z_4ae3c1b346f2`.
Neither command overwrites old evidence or automatically switches viewer data.

Evidence: experiments/reports/MT1-final-verification.json and
.local-artifacts/targets-qa/MT1-20260924. Local commit title: "Add audited pilot
massing targets and multi-context comparison". Post-commit archive in Codex cwd
outputs/NCA-Targets-MT1-Backup-2026-09-24-<commit7>.zip; receipt
.local-artifacts/milestones/<commit7>-targets-mt1-backup-receipt.json. Verify receipt
before claiming archive completion. Helpers in Codex cwd work: build_targets_ui.py,
verify_targets_mt1.py, export_targets_mt1.py, finalize_targets_mt1.py and
package_targets_mt1.py. Preserve MA1 and its full parent archive chain; same-disk
archive is not off-device backup. No Drive operation, push, paid compute or public
deployment. Private reports remain ignored; unrelated user files untracked.

## In progress: MT1 pilot massing targets - 2026-09-24

Read MASSING_TARGETS_PROTOCOL.md. New binary nine-family massing_targets_v1
contract and four-context/12-control audit are implemented, without changing old
losses or starting training. Regression20260924T101120Z_6d8b5d1cc577 running in
owned session74108. Inspect its result/log before further work. Planned audit
scripts/run_massing_targets.py saves48 controls and432 sensitivity reports.
Preserve unexpected outcomes; no automatic threshold retuning. Final completion
entry will supersede this note. No paid compute, Drive access or publication.

## Current: MA1 massing comparison complete - 2026-09-24

D058 remains the brief: building volume now, interiors and construction later.
Read MASSING_AUDIT_FINDINGS.md and MASSING_AUDIT_PROTOCOL.md. MA1 is an analytic
completion comparison, not a trained generator or frozen massing objective.

Successful study20260924T095133Z_b7015ab62f05:11 fields,33 records,83 checks pass,
14.07s CPU. Failed parent20260924T094943Z_1d6ed517bc60 retained: expected domain
count3456 omitted36 old anchor allowances; corrected3492. All geometry and metrics
are identical across attempts. Regression20260924T094603Z_66da5a9b1158:260passed,
0fail/error/skip,smoke0,103.44s. Exact replay of all33 operations/masks/mass reports/
nine-family scores passes.131 science/input Python files match tested source.
Run manifests, source ZIPs, parent VA1 bytes and checkpoint hash verify. No loss,
model or historical scene edits. Private reports remain ignored.

Gallery http://127.0.0.1:8001/static/massing/index.html. Source vs derived mass,
three completion rules,11 controls, additions toggle, cutaway and two true slices.
All33 selector counts checked; desktop/mobile layout checked. Studio and historical
galleries link the new brief; original VA1/SP1 study data are unchanged. Server
PID20784/session91597 remains the existing owned server; no restart was needed.
No study/test/training job active. Inspect process state before restarting it.

Next bounded work: a versioned nine-family massing objective contract and acceptance
examples across several scenes. Prioritize direct mass occupancy; keep completion
as explicit comparison, with raw/final outputs. Address budget, thickness/depth,
coverage/distribution and access/interface meanings before procedural/direct-
optimization baselines and any NCA trial. Do not restart incremental F5 training,
increase grid size or launch Colab by default.

Reproduction: `.venv/Scripts/python.exe scripts/run_massing_audit.py --parent-run
20260924T095133Z_b7015ab62f05` creates a fresh linked attempt; it does not overwrite
history or update the gallery automatically. Full regression command remains
`.venv/Scripts/python.exe scripts/verify_foundation.py --parent-run
20260924T094603Z_66da5a9b1158`. Do not rerun merely to continue documentation.

Evidence: experiments/reports/MA1-verification.json and
.local-artifacts/massing-qa/MA1-20260924. Local commit title: "Add preserved building
mass completion comparison and objective audit". Post-commit archive in Codex cwd
outputs/NCA-Massing-MA1-Backup-2026-09-24-<commit7>.zip; receipt under
.local-artifacts/milestones/<commit7>-massing-ma1-backup-receipt.json. Inspect it
before claiming backup completion. Helpers in Codex cwd work: verify_massing_ma1.py,
verify_massing_ma1_failed_path.py and package_massing_ma1.py. Archive writes are
exclusive; inspect existing files before retrying. Retain VA1 and older archives
for historical raw data/private reports. Same-disk copy is not off-device backup.
No Drive operation, paid compute, push or publication. Unrelated files untracked.

## In progress: MA1 massing comparison - 2026-09-24

Implementing D058 via MASSING_AUDIT_PROTOCOL.md, nca/massing.py and the new
static/massing gallery. Regression20260924T094603Z_66da5a9b1158:260 tests pass,
smoke0, no skips. First MA1 run20260924T094943Z_1d6ed517bc60 failed one expected
domain-count check: actual3492 versus expected3456, due to36 historical anchor
allowances. All other82 checks passed. Recipe expectation corrected; preserve
failed run and link fresh attempt. No science implementation changed after tests.
Viewer implementation is pending browser verification and successful study data.
Before retrying inspect records/processes; do not overwrite run evidence. No
training, Drive operation or push. Completion entry will supersede this note.

## Current: building-mass interpretation confirmed - 2026-09-24

Read D058 and MASSING_BRIEF.md first. The user confirmed generating overall
building mass now, leaving interiors and construction for later. Future occupied
voxels mean building volume, not solid construction material. This supersedes
D057's required material/void target, retaining depth without a room program.

Documentation only: no code, data, model, viewer, objective or old run changed.
VA1 still displays historical material probes and a blue empty-space diagnostic.
No automatic fill is implemented or selected. The last code regression remains
VA1's 248-test pass; it was not rerun for this documentation-only decision.

Next: versioned massing contract and saved original/derived comparison per
MASSING_BRIEF.md, then nine-family checks. Freeze budget/depth criteria before new
training. Do not resume the older void-first plan below as current work.

Preserve VA1 commit48d9c40 and its archive plus all parent archives. This decision
gets a local commit and verified document archive under Codex cwd outputs, with a
receipt recording commit and file hashes. No scientific run, server restart,
training, Drive access, push or publication. Unrelated files remain untracked.

## Current: corrected volumetric target and VA1 audit complete - 2026-09-24

Read D057, VOLUMETRIC_AUDIT_FINDINGS.md and VOLUMETRIC_NEXT_PHASE.md.
User target: volumetric forms with spatial depth and voids, without prescribed
rooms/shelters/functions. SP1 platform and assistant's later room-program framing
are superseded. AGENTS records this. Do not resume platform optimization by default.

VA120260924T085504Z_36c834017633:9 fields,13.08s,no training,all16 checks pass.
Equal512-cell slab/solid have0 bracketed void; hollow has784. Thin hollow forms
have thickness0 but conflict with old material budget/occupied-route targets.
This is a fixed-field diagnostic, not trained/generalization evidence or a
universal space-quality score. Open/closed voids and context closure are separate.
All old nine-family definitions and model files remain unchanged.

Regression20260924T085027Z_d8c033bd285c:248passed,0fail/error/skip,smoke0,
119.23s. All nine saved geometry diagnostics/old scores/masks exactly recompute;
source/checkpoint/config hashes and both RunStore manifests verify. See
experiments/reports/VA1-verification.json and .local-artifacts/volume-qa/VA1-20260924.
Static served study bytes match the retained audit. Final gallery source and
desktop/equal-mass-slice/mobile screenshots are in QA.

Studio http://127.0.0.1:8001/static/volume/index.html. Restarted server PID20784,
session91597; old PID3028/session45515 stopped. Run from project cwd with
`.venv/Scripts/python.exe -m uvicorn deploy.studio:app --host 127.0.0.1 --port 8001 --no-access-log`.
Check process before restart; one worker only. No training/test active. Gallery
uses saved analytical probes, not arbitrary-scene generation or trained NCA.

Next bounded work: a candidate-independent physical3D opportunity region and
versioned access/coverage/material-budget contract audit. Retain nine families,
counterexamples and historical metrics. No automatic new loss-training series,
new backbone, larger grid, paid Colab or Drive operation.

Local commit title: "Document volumetric target and add material-void objective audit".
Post-commit archive: Codex cwd outputs/NCA-Volume-VA1-Backup-2026-09-24-<commit7>.zip.
Completion receipt .local-artifacts/milestones/<commit7>-volume-va1-backup-receipt.json.
Helper Codex cwd work/package_volume_va1.py uses exclusive writes; inspect existing
files/processes before retry. Retain SP1ec8fa1b,S23391fef,S18c51590 and fullF58f2d8d6
archives for prior evidence. Same-disk only. Private report remains ignored;
unrelated concept HTML/M1 sidecar remain untracked. No push/publication/Drive access.


## Current: corrected volumetric target; VA1 audit in progress - 2026-09-24

User clarified the target is volumetric form with spatial depth and voids, without
predefined shelter/room functions. SP1's flat platform is a diagnostic only; the
assistant's subsequent room/shelter framing is also superseded. Follow
VOLUMETRIC_AUDIT_PROTOCOL.md. Implement nca/volumetric.py, fixed probes, regression
and run_volumetric_audit.py; preserve all source and results. New diagnostics are
descriptive, not a tenth constraint or a frozen training loss. No training/Colab.
Check running processes/run directories before continuing. Final result entry
will supersede this in-progress entry when verified.

## Current: SP1 spatial prototype completed - 2026-09-24

Read SPATIAL_PLATFORM_SPEC.md, SPATIAL_PLATFORM_FINDINGS.md, D056 and
experiments/reports/SP1-verification.json. These supersede older current notes.
No training or tests active. Studio server PID3028/session45515 on8001; old
PID19444/session43924 stopped. Current launch from project cwd:
`.venv/Scripts/python.exe -m uvicorn deploy.studio:app --host 127.0.0.1 --port 8001 --no-access-log`.
Inspect owned process/listener before restarting. One server worker only.

Gallery http://127.0.0.1:8001/static/spatial/index.html is linked from Studio.
It displays saved SP1 examples, not arbitrary-scene construction or trained NCA.
Positive level deck2.4m wide with4x4m landing/2.4m clear height meets spatial gate.
Five designed negatives/unsupported cases fail as expected. W1 fails spatial gate
on all six. Positive deck58voxels/37.12m²/4.8013% old-envelope budget, but old access
and coverage penalties1.0 each. Old metrics and original model unchanged.

Study20260924T073056Z_db9cf854c9e3:8.49s,all expected outcomes, exact source and
all paired fields/masks/scenes/diagnostics retained. Served study.json byte-matches.
Regression20260924T072445Z_e8fffe7ec74b:236passed,0fail/error/skip,smoke0,119.32s.
12 independent spatial tests. Subsequent frontend camera/layout changes visually
checked. Both RunStore manifests verify; all saved spatial reports/masks recompute.
QA .local-artifacts/spatial-qa/SP1-20260924 holds screenshots and exact final UI.

Next: clarify external approaches versus doors/interior access, then specify and
audit versioned spatial access/coverage targets on saved positive/negative fields.
No automatic learning, bigger grid, new constraint family or paid Colab launch.
Keep historical losses and comparisons; new gate is a separate geometric contract.

Local commit title: "Add evaluated spatial platform prototype and comparison gallery".
Post-commit incremental archive outputs/NCA-Spatial-SP1-Backup-2026-09-24-<commit7>.zip
under Codex cwd; completion receipt .local-artifacts/milestones/<commit7>-spatial-sp1-backup-receipt.json.
Verify receipt before claiming completion. Helper Codex cwd work/package_spatial_sp1.py
uses exclusive writes. If interrupted inspect files/processes, never overwrite.
Retain S2 archive3391fef, S1 archive8c51590 and full F5 archive8f2d8d6 for older evidence.
Same-disk archive only. No Drive access, push, public hosting or paid training.
Private report ignored; unrelated concept HTML and M1 ZIP sidecar stay untracked.


## Current: Studio S2 implemented and verified - 2026-09-24

Read STUDIO_S2.md, SPATIAL_BRIEF.md, D055 and S2-studio-verification.json.
This entry supersedes historical current/running notes below. No training/test
process is active. Local Studio http://127.0.0.1:8001 runs PID19444/session43924;
inspect before restarting. Launch from project cwd with `.venv/Scripts/python.exe
-m uvicorn deploy.studio:app --host 127.0.0.1 --port 8001`, only one worker.

Final backend regression20260923T231009Z_672a788a5c0d:224pass,0fail/error/skip,
smoke0,111.71s including infrastructure. Final frontend JSON transfer fix followed;
actual browser export/import and checksum rejection pass. Browser cancellation
job20260923T231449Z_133fbe52f872 really entered running then cancelled, no receipt;
retry20260923T231501Z_70f5e39cc675 completed with parent link. Three completed S2
browser results, one cancelled history;13 saved S1/S2 records at milestone.

Preserve .local-artifacts/studio, studio-jobs, studio-s2-probes and studio-qa/S2-20260924.
Four probes retain earlier import-startup hangs and their exact sources. QA includes
screenshots, valid/rejected portable files, final source ZIP and hashes. The original
browser-reencoded download is broken; the filename ending'(1).json' passes checksum.
Server backend matches hashes in jobs. UI transfers original JSON text now.

Jobs survive as histories: incomplete work becomes interrupted on restart, not
automatically resumed. Retry always creates a linked new attempt. Windows tests
prove child/grandchild termination on owner death. No non-Windows tree guarantee.

Next: illustrated spatial contract per SPATIAL_BRIEF.md. User correctly noted the
thin output is not a space. Studio runs procedural W1, never the trained NCA;
F5 is still unpromoted. Specify material/surface/void and within-nine-family metric
meanings with positive/negative examples before more learning. No Colab needed now.

Local milestone commit title: "Add durable Studio jobs, comparison and verified import".
Post-commit incremental archive: Codex cwd outputs/NCA-Studio-S2-Backup-2026-09-24-<commit7>.zip;
receipt .local-artifacts/milestones/<commit7>-studio-s2-backup-receipt.json establishes
verified completion. Packaging helper Codex cwd work/package_studio_s2.py is
write-once: inspect partial files/processes before retry, never overwrite evidence.
Retain S1 archive8c51590 and full F5 archive8f2d8d6 for older evidence. Same-disk only.

Private next-phase report remains ignored. Unrelated user concept HTML/M1 ZIP
sidecar remain untracked. No Drive access, push, publication or paid training.
Every Drive operation still requires separate explicit permission inside the
designated project folder only. Historical resume notes below remain unchanged.


## Current: Studio S1 implemented and verified - 2026-09-24

Read STUDIO_S1.md, PLANNER_REFINER_SPEC.md, D054 and
experiments/reports/S1-studio-verification.json. This heading supersedes older
current/running entries below. No training or verification is active.

New local Studio: http://127.0.0.1:8001. Server started in this session with PID30956,
exec session17539. It may stop when the host/app closes; inspect listener/process
before starting another. Use project cwd and `.venv/Scripts/python.exe -m uvicorn
deploy.studio:app --host 127.0.0.1 --port 8001`, or deploy/run-studio.ps1. A sandbox
launch from another cwd had WinError5 on repository writes; use the project cwd.
Temporary diagnostic port8002 server and earlier servers were stopped.

Final verification20260923T223757Z_6602b2cf16c8:211 passed,0fail/error/skip,
smoke0,96.61s. Parent20260923T223300Z_af097d57765b also211passed. BrowserQA,
screenshots and hash-matched source ZIPs for all saved-study versions are in
.local-artifacts/studio-qa/S1-20260924. Each new study lives in
.local-artifacts/studio/<id>/; five completed browser study records currently.
Final facade result20260923T223755Z_7f580c96ff6a includes durable submitted request.
Earlier four records are preserved and predate request-before-compute persistence.
The JSON download equals20260923T223419Z_6b1b42479c59; it remains in Downloads.

No scientific nca file, historical serving path, original model or reference scene
changed. Reports remain ignored. User-owned NCA-Studio-Concept.html and old ZIP
checksum remain unrelated untracked files. No Drive access, push or deployment.

Next: S2 durable job lifecycle + real cancellation, then revision-aware comparison
and verified import. This is a first Studio product slice, not completion of M4.
Do not automatically resume F5 training or start Colab. The refiner specification
requires task/control evidence and a frozen success criterion before learning.

Local milestone commit is discoverable by title "Add local Studio with evaluated
procedural studies and refinement specification". S1 incremental archive and
receipt under .local-artifacts/milestones are produced after that commit and
include the source bundle, new runs, all Studio data, screenshots and exact files.
It supplements the verified full F5 archive listed below; retain both. Same disk
is not an off-device backup. Check the receipt before claiming archive completion.

## Current: bounded local investigation COMPLETE - 2026-09-24

No active training/verification. Read LOCAL_PHASE_CLOSURE.md,
PERSISTENT_GUIDE_FINDINGS.md and D053. User asked to finish this local phase;
implementation/regression, baseline parity, restart, timing pilot, full comparison
and late recovery are all complete. Do not resume the superseded in-progress
notes below or launch an automatic next local loss/conditioning experiment.

F5 full20260923T214035Z_566da8c507c0: 66/72 connected,0/72 budget,0/72 joint,
gains4/losses0 versus F4. Full256 updates/120 unique
evaluations in1317.67s, all900/member3600/full caps met.
Source27416f30141d6648d798380dfd7fcb191889f69f,47 hashes; all 41 historical F4 files unchanged.
Final regression20260923T212031Z_48dcfddbfd4f:199 tests, no failures/errors/skips,
smoke0. Initial197-test pass preserved. No scientific edits after final pass.

B20260923T212257Z_823a2b09b2dd: exact12 F4 updates/24 evaluations.
R20260923T212734Z_efe016a02413: all11 restart checks; five intermediate checkpoints also match.
P20260923T213256Z_42130eed3f5d: timing gate admits frozen cap, guide gradients nonzero.
L20260923T220548Z_fdd27bb79c92: trained update62->63->64 replay on all four models,
eight updates/eight evaluations exact; no added training exposure.
Full reporter rescored376 fields/260 cursors/128 F4 controls/eight final rollouts.

Next: Write the next design specification around a planner-provided valid scaffold
and an NCA with a narrower refinement/recovery role. Compare that hybrid against
the already strong procedural and direct-optimization controls. Retain the architectural material/form-generation scope already accepted in
D026. Specify what learned refinement would add beyond those controls before
changing representation or starting another training experiment. This is a proposed
direction, not an implemented or proven replacement.

Scope the next phase explicitly. Product Studio redesign remains outstanding;
neither this closure nor a passed regression is a production-quality claim.
No Colab action is needed for this completed phase. No paid training is active.

Evidence: experiments/reports/F5-*, F5B/F5R/F5P and F5L-verification.json;
experiments/records/<ID>.json; .local-artifacts/runs/<ID>. Exact snapshots and
all analysis scripts/receipts preserved. Final figure F5-horizons.png,48 aggregate
points and image/source hashes checked, visually inspected. Source guards remain.

Results commit message: "Close local NCA investigation with verified F5 conditioning results".
Find its ID in Git. Archive in Codex cwd outputs/NCA-Persistent-Guide-Backup-
2026-09-24-<commit7>.zip; sidecar .zip.sha256 and
.local-artifacts/milestones/<commit7>-backup-receipt.json certify completion.
Check receipt/hash; ZIP existence alone does not imply completion. Packaging
script in Codex cwd work/package_nca_guide_training.py takes B,R,P,full IDs above.
If interrupted, inspect process and files before retrying; never overwrite partial
evidence. Fresh Git restore checks6 reference/12 legacy scenes/18 annotations and
config values. Exact47-file source ZIP and working-tree-evidence retain original
byte hashes even if Git normalizes JSON or Python line endings. Use NEW workspaces
for exact historical recovery. No guard bypass or blind rerun of exclusive reports.

Use .venv/Scripts/python.exe, CPU2 threads, Python3.12.14/torch2.8.0+cpu/numpy2.5.2;
no pip in this venv. Plotting uses the existing isolated Codex .plot-deps only.
Branch next-phase/foundations. Preserve unrelated NCA-Studio-Concept.html and
untracked M1 archive sidecar. Private report md/pdf remain ignored. Local archives
are not off-device backups. EVERY Drive operation requires explicit permission,
only folder1fS34Yy0-oMzSxWaYJFiPTkGgrZstgc0H and actual descendants. No push or
publication occurred. Earlier experiment/resume history follows for provenance.


## Current F5 full study RUNNING - 2026-09-23

Full20260923T214035Z_566da8c507c0, active session17706. Frozen source27416f3,
47 scientific hashes. Inspect process/result/logs before any retry. Worker900s,
phase3600s.64 constant16 updates for each of four conditioned models. Do not
edit hashed code/config or change allowances during this experiment.

All gates complete and reporter-verified:
- Final regression20260923T212031Z_48dcfddbfd4f:199 passed/smoke0.
- B20260923T212257Z_823a2b09b2dd:12 updates/24 evals exact F4 parity,141.65s.
- R20260923T212734Z_efe016a02413:11 exact recovery checks,93.50s.
- P20260923T213256Z_42130eed3f5d:351.21s,47 hashes/96 fields/12 cursors,
 128 F4 controls verified. Timing2595.94s admitted3600 cap; all guide gradients
 nonzero in pilot. Full settings unchanged. No outcome-based selection.

After completion, project .venv/Scripts/python.exe:
1. scripts/report_guide_training.py 20260923T214035Z_566da8c507c0
2. scripts/check_guide_late_recovery.py 20260923T214035Z_566da8c507c0
3. Codex cwd work/analyze_f5.py and work/analyze_f5_families.py; then
 work/check_f5_provenance.py fullID. No edits to frozen scientific source.
4. work/plot_f5.py using existing isolated plotting libraries, visually inspect
 PNG then work/verify_f5_figure.py. No pip in project venv.
5. Review full outcomes/limitations, then work/close_f5_phase.py B R P fullID
 (IDs above), which writes findings/phase closure/D053/resume. Review its generated
 interpretation against outcomes before committing. No model auto-promotion.
6. Local results commit and work/package_nca_guide_training.py B R P fullID.
 Builder requires completed F5L verification and verifies bundle/restore/all payloads.

This is the final bounded experiment in the current local investigation. Close
with an evidence-based decision: fresh validation if promising, otherwise review
representation/NCA role before more training. Studio upgrade remains later work.
No paid Colab, Drive operation, push or deployment. Previous4a94dbe archive verified.
All raw evidence and failed attempts retained. Historical state notes follow.


## Current F5 preparation - 2026-09-23

User authorized finishing the local investigation. New opt-in conditioning code
and protocol exist; no F5 experiment has run. Read PERSISTENT_GUIDE_PROTOCOL.md
and D052. Next full regression, source freeze, then run_guide_training.py modes
parity, recovery, pilot, study with matching --parity-run/--recovery-run/--pilot-run.
Verify each with report_guide_training.py <ID>. Full worker900/phase3600 cap,
64 updates x four models, same F4 science. Final check_guide_late_recovery.py <full>.
Do not edit frozen code between gates; preserve every attempt. Close local phase
with results/decision/resume/verified archive. No paid Colab/Drive/push/deployment.
Previous archive4a94dbe is complete and verified; source F4 artifacts remain.


## Current completed F4 milestone - 2026-09-23

No active training or verification process. Read RAW_ACCESS_TRAINING_FINDINGS.md,
D051 and PERSISTENT_GUIDE_PLAN.md before further work. F4 full
20260923T202649Z_75034cca563c completed1321.78s,256 updates/120 unique evaluations,
all900/member and2400/total caps met. Outcome62/72 connected vs59/72 F2,
gains3/losses0, zero in-budget/joint final cases in either arm. No promotion.

Full verifier passed41 source hashes,376 fields,260 cursors,128 F2/H1 controls
and eight final rollouts. Early recovery passed; F4L20260923T205434Z_c8136d2ce8e8
replayed four update62->63->64 checkpoints exactly:8 updates/8 evaluations in
83.58s, all caps met.187-test regression20260923T201206Z_4b81a6dabfa4 passed with
zero failures/errors/skips, smoke0. Scientific source373968d unchanged afterwards.

Final source gates: B2 20260923T201413Z_628ce760d88c, R2
20260923T201744Z_ef6c479ec7f8, P2 20260923T202008Z_206b3f6adb5f. P2 timing
2356.91 admitted2400. First-source12357e8 B/R/P retained, including rejected
P20260923T200402Z_b794e30d69ba forecast2409.50>2400. D050 removes redundant
evaluation work; pilot traces/fields/full state exact between source versions.
All31 historical H1 hashes unchanged. No source/cap/outcome guard bypass.

Evidence: experiments/reports/F4-*.json and F4L-verification.json, F4B2/F4R2/F4P2
reports, experiments/records/<ID>.json; raw .local-artifacts/runs/<ID>.
Post-hoc scripts/receipts in .local-artifacts/analysis-attempts/F4-*.
Final figure F4-horizons-v2.png,48 points checked and visually inspected; v1 kept.

### Exact next step

Implement PERSISTENT_GUIDE_PLAN.md as one bounded architecture test. Keep the F4
raw objective, constant16 horizon, original Model C initialization for both arms,
two scenes, two recipes and64 updates/member. Separate versioned guide branch:
same scaffold, four cached perception features, zero-initialized4->96 projection
before first ReLU (384 added weights); preserve eight total state/four evolving channels.
First implement/test null/zero-branch parity, backbone vs guide gradients, context
binding, RNG preservation, checkpoint migration and actual-loop recovery; freeze
protocol/config/source and compute caps before any learning. Do not train from
F4's final weights, alter losses/weights/scenes, or scale resolution concurrently.
F4 remains a research control, F2 additional reference, neither production-ready.
If bounded conditioning fails, review representation and role vs W1/D1 controls.
No conditioning training is active or already implemented.

Use project .venv/Scripts/python.exe (3.12.14, torch2.8.0+cpu, numpy2.5.2).
This venv has no pip; do not modify it for plotting. Git branch
next-phase/foundations. Preserve unrelated NCA-Studio-Concept.html and the
untracked M1 archive sidecar. Private NCA-Next-Phase-Report md/pdf stay ignored.

### Archive completion and interruption recovery

The final local results commit uses message "Record F4 raw-access learning
results and persistent-guide next step". Determine its ID from Git rather than
this file. Full archive builder in Codex cwd work/package_nca_raw_access_training.py
takes the B2,R2,P2,full IDs above. A successful archive is named
outputs/NCA-Raw-Access-Training-Backup-2026-09-23-<commit7>.zip in Codex cwd,
with .zip.sha256 and .local-artifacts/milestones/<commit7>-backup-receipt.json.
Check the receipt and matching hash before treating archive completion as true.
It verifies a fresh Git restore,6 reference/12 legacy scenes,18 annotations,
41 exact snapshot source hashes and every archived payload. Preserve all old
archives. If interrupted during packaging, inspect process/files; never overwrite
partial evidence. Use a separately named linked packaging attempt if necessary.

For historical optimizer replay extract that run's exact source ZIP into a NEW
workspace; Git checkout line endings may differ. Existing reporters use exclusive
outputs, so do not blindly rerun them. Inspect completed result.json first.

All work local. Same-disk archive is not an off-device backup. Drive requires
explicit approval for EVERY operation, only folder1fS34Yy0-oMzSxWaYJFiPTkGgrZstgc0H.
No paid Colab, Drive access, push, production checkpoint change or deployment.


## Historical in-progress F4 checkpoint - superseded below

Full20260923T202649Z_75034cca563c active session75488. Source373968d,41 frozen
scientific hashes. Inspect result/process/logs before any retry. Worker900s,
full2400s,64 constant16 updates/model across four members. Do not edit hashed
code/config while running, change caps or restart from zero after interruption.

First source12357e8: B20260923T195731Z_c19895c6c745 exact12/24 in142.85s;
R20260923T200110Z_961286a9e4b3 all11 recovery checks in91.25s;
P20260923T200402Z_b794e30d69ba completed328.52s but timing2409.50>2400 NOT
admitted. All first-source reports retained F4B/F4R/F4P. D050 removes one
redundant evaluation recomputation, changes no objective/step/cases/caps.

Revised source373968d: regression20260923T201206Z_4b81a6dabfa4 passed187 tests,
zero failures/errors/skips, smoke0. B2 20260923T201413Z_628ce760d88c exact12/24
in134.59s; R2 20260923T201744Z_ef6c479ec7f8 all11 checks in85.55s;
P2 20260923T202008Z_206b3f6adb5f completed295.22s, timing2356.91 admits2400.
All reporter verifications passed41 hashes and saved fields/controls.
F4-efficiency-parity.json verifies all8 updates/24 boundary/72 grid records
and full checkpoint/RNG states exact between pilots;96 unique fields. Training
step AST unchanged. New gate reports use F4B2/F4R2/F4P2 prefixes.

After full completion, project .venv/Scripts/python.exe:
- scripts/report_raw_access_training.py 20260923T202649Z_75034cca563c
- scripts/check_raw_access_late_recovery.py 20260923T202649Z_75034cca563c
- Codex cwd work/analyze_f4.py (after full evidence report exists)
- work/plot_f4.py, visually inspect figure, then work/verify_f4_figure.py.
Plotting uses isolated Codex cwd .plot-deps; do not install in project venv.
Preserve all failures, document final outcome/next decision and source provenance,
commit locally and archive with work/package_nca_raw_access_training.py B2 R2 P2
fullID (IDs above). Builder requires F4L-verification.json and includes every
previous run/private report; validates bundle/restore/source/payload hashes.

Primary outcome joint component connectivity/material budget, not raw loss.
No paid compute, Drive operation, production promotion or remote push. Previous
full local archive878d5f2 receipt exists; same-disk copy only.

## Previous F4 preparation - 2026-09-23

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

F4B2 20260923T201413Z_628ce760d88c active session30253; revised source373968d. Use report_raw_access_training.py <ID> F4B2. Then fresh recovery linked to oldR and fresh pilot linked to oldP. All caps unchanged; only two of41 scientific source hashes changed (evaluation reuse and report prefix).

F4B2 20260923T201413Z_628ce760d88c completed134.59s with exact12/24 F2 parity. Reporter session37901 active. Next --mode recovery --parity-run thisID --parent-run 20260923T200110Z_961286a9e4b3; report prefix F4R2.

F4B2 fully verified41 hashes/36 fields/16 cursors. F4R2 20260923T201744Z_ef6c479ec7f8 active session28631; next report with F4R2 prefix, then P2 linked to first pilot.

F4R2 20260923T201744Z_ef6c479ec7f8 completed85.55s, all11 restart checks pass. Reporter session59569 active. Next P2 --parity-run 20260923T201413Z_628ce760d88c --recovery-run 20260923T201744Z_ef6c479ec7f8 --parent-run 20260923T200402Z_b794e30d69ba.

F4R2 reporter passed. F4P2 20260923T202008Z_206b3f6adb5f active session11377, revised source373968d. Report with F4P2 prefix, then work/verify_f4_efficiency.py 20260923T200402Z_b794e30d69ba 20260923T202008Z_206b3f6adb5f. Full only if verified and timing-admitted.

F4P2 20260923T202008Z_206b3f6adb5f completed295.22s; estimate2356.91s admits unchanged2400 cap. Cross-version pilot check passed all8 updates/24 boundary/72 grid records and full checkpoint trees, plus identical step AST;96 unique fields. Reporter active session12997. Full may start only after report verifies.

Full F4 20260923T202649Z_75034cca563c active session75488, source373968d,41 frozen hashes. All B2/R2/P2 and cross-pilot equivalence gates passed; full admitted2356.91s under2400 cap. After completed: report_raw_access_training.py <full-ID>, check_raw_access_late_recovery.py <full-ID>, work/analyze_f4.py, findings/decision/provenance/plot and verified archive. Never restart from zero or change caps if interrupted; inspect completed records/processes first.

Full F4 20260923T202649Z_75034cca563c completed1321.78s,256 updates/56 boundary evaluations plus72 final-grid records (8 reused). All phase/worker caps met. Full reporter active session16575. Next F4L after verifier passes; do not rerun training.

Full verifier passed41 source hashes/376 fields/260 cursors/128 controls/eight final rollouts. F4L20260923T205434Z_c8136d2ce8e8 active session16782: four update62-to64 resumes, worker120/total360s. Inspect result before retry.

F5 regression20260923T212031Z_48dcfddbfd4f passed199 tests/smoke0. Source freeze follows, then F5B actual-F4 parity. No scientific edits after this pass.


F5B20260923T212257Z_823a2b09b2dd active session84781; source27416f3. Inspect result before retry. Then report_guide_training.py that ID; only after verification run recovery --parity-run that ID.


F5B report passed. F5R20260923T212734Z_efe016a02413 active session62898; inspect result before retry. Next report_guide_training.py recoveryID, then pilot using B20260923T212257Z_823a2b09b2dd and this R ID.


F5R report verified. F5P20260923T213256Z_42130eed3f5d active session97542, worker240/phase1200. Source27416f3. After completion report_guide_training.py pilotID and inspect timing admission. Full only if admitted <=900/member3600/total with B/R IDs above.


F5 full completed1317.67s within caps. No active training. Full reporter launched; after it passes run check_guide_late_recovery.py fullID before post-hoc analysis and phase closure.


F5 full reporter passed47 source hashes/376 fields/260 cursors/128 controls/eight final replays. Provenance check verifies all41 old F4 files unchanged. Trained-state F5L replay launched; inspect its result before closure.


F5L20260923T220548Z_fdd27bb79c92 passed all trained-state checks in77.01s. Full scientific verification complete. Outcome analysis underway; next plot/QA, close_f5_phase.py, results commit and verified archive. No active training.

