# Next-phase implementation plan

## Connected repair design ready, implementation pending - 2026-09-28

D089: CONNECTED_REPAIR_SPEC.md specifies a separate constructive NCA: synchronous
six-face growth frontier, monotonic accepted occupancy, seven hidden channels,
per-step frontier classification and3-cube local-volume supervision. Decisions
are detached, no claimed gradient through hard births. Preserves input and
prevents new components when input connected; does not ensure disconnected
inputs merge or that all nine checks pass. Wrong births are irreversible.
TRAIN-only81-row hash-verified audit found all inputs subsets of targets and all
missing cells reachable in<=4ideal simultaneous expansions. Not a learned result.
Report: experiments/reports/frontier-feasibility.json; script:
scripts/audit_frontier_feasibility.py. No validation/TEST or training in this audit.
Proposal config CGR1-proposal.json is explicitly disarmed/design-only.
Next implement separate module and one consolidated CPU correctness/recovery
check, then package ONE256-update32-step600s-cap proposed GPU job. Ask for exact
compute approval after preparation; no automatic run/retry. Preserve NR5 and MG7.
Current deliverable is specification and feasibility evidence, not GPU-ready code.


## NR5 saved-output diagnosis complete - 2026-09-28

All27 archived outputs verified and metrics reproduced; raw NR5 stays17/27.
Ten access failures contain27 detached cells:24false additions +3correct newly
reconstructed target cells (none survived from input). Both interfaces and bulk
connect in every output. Support fails4; thickness fails2. Diagnostic pruning
passes26/27 but deletes3correct cells and can improve thickness by denominator
shrinkage. Do not admit it or label it learned repair. Oracle excess removal24/27;
oracle missing restoration18/27. Full detail: NR5_FAILURE_DIAGNOSIS.md.
Outputs: C:/Users/artin/Documents/Codex/outputs/NR5-Failure-Diagnosis-2026-09-28.
Source review: C:/Users/artin/Documents/Codex/2026-09-06/cre/outputs/NR5-Single-Trial-Review.
Next: specify one connectivity-aware growth/volumetric training redesign before
implementation; proposal remains unvalidated. No inference or training this turn.
MG7 stays live. No paid compute, Drive, push or publication. Preserve original
NR5 scores. See per-case experiments/reports/NR5-failure-diagnosis.json and script
scripts/diagnose_nr5.py for exact reproducible analysis.


## Unified Studio home - 2026-09-28

Root / now serves deploy/static/home/index.html with Explore, Research and Archive
navigation. /scaffold serves the original deploy/studio.html byte-for-byte.
Existing live/live-v2/live-v3 explicit scaffold links point there; brand links
return to the home. Generator APIs, geometry, models and stored records unchanged.
No new generation or training run. No push, publishing or Drive access.

Homepage clearly separates procedural MG7 live generation from experimental NCA;
shows dated15-site/three-scale scope, NR3/NR4/NR5 all-nine counts25/19/17 out of27,
ED1 eight-of-eight pilot result and limitations. Downloadable JSON copies match
experiments/reports/NR5-single-trial-review.json and ED1-diversity.json exactly.
These are static dated research snapshots; update them deliberately with future
findings. Original research galleries and historical semantics are preserved.

Checked all12 local href targets with HTTP200, original scaffold response against
source bytes, both research copies against their originals. Browser screenshot
and Research anchor checked; no console errors. Responsive CSS included; no
independent mobile device verification. This completes the current local entry/
navigation batch, not public-hosting readiness or learned-model admission.

Screenshot and verified source/doc archive: Codex outputs/Studio-Home-2026-09-28.
Server PID10740/session96370; local-only127.0.0.1:8001. Restart was performed after
confirming all three job queues had no queued/running tasks. Resume command:
.venv/Scripts/python.exe -m uvicorn deploy.studio:app --host127.0.0.1 --port8001 --no-access-log
(with spaces between option names and values). Same-disk archive, not off-device.

Next: review the consolidated product experience with the user; select the next
substantial research or deployment objective. No automatic additional seed/loss
sweeps or paid training. Current live NCA reliability limitation remains D084.


## ED1 environment diversity completed - 2026-09-28

Added four designed48-cubed development sites: different connection heights,
staggered obstacles, overhead crossing, asymmetric frontages. Same0.8m cells,
MG7 generator,2.4m growth blocks, MT1 nine-family evaluation and fixed-domain
24% request. Seeds6/7;45s generation cap per candidate; CPU2threads. No new
constraint, loss, model or training. These cases extend environment variety at
one existing scale; no independent generalization or architectural-quality claim.

Run20260928T092021Z_cd4a5b070033 completed8/8 candidates in13.237s including
source capture/context/serialization. All8 reached target and passed all nine
pilot checks. Generation plus evaluation ranged0.960-1.926s. Across seeds the
four sites differ by4544,1506,468,1944cells respectively; occupancy IoU
.437693,.615914,.798796,.661265. Equal volume does not imply identical geometry.

Evidence: .local-artifacts/runs/20260928T092021Z_cd4a5b070033 contains complete
source snapshot, recipe/provenance, four audited scenes and context arrays,
eight raw fields/routes/diagnostics, growth traces, checks and timestamps.
Every registered artifact hash verified. Each raw field checksum checked and
all eight scores independently recomputed from saved arrays, with exact match.
Run record under experiments/records; small summaries ED1-diversity.json,
ED1-seed-differences.json and ED1-studio-check.json under experiments/reports.

Appended four presets to deploy/mass_v2_contexts.json; all11 existing context
objects compare equal, preserving their individual identities and replay inputs.
Catalogue now15 sites /65 evaluated setting combinations. Added only seeds6/7
and request24% for new sites. Existing unsuccessful controls remain selectable.
Extended source provenance to include live-v3 files for new Studio jobs.
Focused preset/bounds/mask/API-origin test passed; existing test expected counts
updated. No generator/evaluator tests rerun because their code is unchanged.

Browser:48 filter lists7 sites. New-site generation saved record
20260928T092244Z_eee2c4098f12 (staggered,seed6); exact field hash and all target
metrics match the study. New record includes live-v3 provenance. No console
errors. Initial mouse activation did not enqueue a job; inspected state, then
keyboard activation completed one job. Screenshot in Codex outputs/
ED1-Diversity-2026-09-28/studio.png. Historical stored records untouched.
Server restarted after checking no active jobs: PID6172/session31618,
python -m uvicorn deploy.studio:app --host127.0.0.1 --port8001 --no-access-log.
Process metadata query was denied; stopped the known owned server session instead.

Local archive destination: Codex outputs/ED1-Diversity-2026-09-28.
No Drive read/write, paid compute, remote push or publishing. Same disk only.
Next batch: unify the Studio entry page and navigation so generation, model
research and evidence are easy to distinguish and find. Avoid further small
training/seed sweeps; learned path remains experimental under D084.


## Studio volume explorer - 2026-09-28

Implemented /static/live-v3/index.html, linked from live-v2. This is a frontend
upgrade over the existing /api/mass-v2 service: MG7 remains procedural, NR3/4/5
remain experimental. No model, objective, nine-family evaluator, dataset,
preset geometry, generation budget or stored experiment was changed.

Features: orbit by pointer drag or arrow keys; bounded zoom via buttons, +/- or
Shift-wheel; reset view; boundary-face cache and camera-facing culling; world-size
filter for the EXISTING 11 presets at32/48/64; labeled PNG export with record and
method identity; retained orthogonal slices, fixed-Y cutaway, history, replay
import/export and linked job lifecycle. Shared camera aligns comparison views.
Transparent context uses approximate painter ordering; it is illustrative.
No measured speedup claim, new scale benchmark or learned-model scaling claim.

Verified in the in-app browser against saved64-cubed record
20260925T092140Z_b57cc8b2676f: scale64 filter exposes three sites; record loads
4209.15m3 with original passing checks; pointer rotation, zoom/reset, both slices,
cutaway toggle and different-site comparison work; no browser error logs.
PNG saved to Downloads and copied into Codex outputs/Studio-Explorer-2026-09-28;
881x660 PNG signature, all chunk CRCs and decompressed pixel stream verified.
Screenshot retained there as explorer.png. Existing history was read, not rerun.
Generation/import lifecycle code is carried forward; no new generation job or
import replay was run for this presentation-only change. Responsive rules added,
but small-device interaction has not been independently verified.

Recovery notes: server was stopped; restarted local-only uvicorn deploy.studio:app
on127.0.0.1:8001 (PID6576, tool session7847). First preview hit connection refused;
that stale error tab blocked navigation, so a fresh tab was used. Browser download
wait timed out after successful file creation; filesystem verification confirmed
the PNG. PIL unavailable in project venv; used standard-library PNG validation.

Next: expand environment diversity deliberately using the same nine families,
then evaluate those new settings in one bounded local batch before exposing them
as evaluated choices. Existing32/48/64 support is not newly achieved here. Decide
on a unified landing page once the explorer interaction is reviewed; do not
silently replace historical interfaces. No automatic new paid training.
No Drive operation, push or publication. Local archive is same-disk only.


## Current: NR5 reviewed; bounded training sequence closed - 2026-09-28

User supplied completed run20260928T084850Z_cb574ff5e2be and its receipt. Assistant
launched no paid job. No separate approval reply preceded this attachment; record
user-run execution without inventing approval or inferring further compute access.
GPU job completed256 updates in45.767336s,688MiB peak reserved (560260608 allocated).
All1036 payload hashes, outer receipt checksum, final checkpoint checksum, objective,
32-step identity, seed, study manifest, sampler/trace and all Adam steps verified.

Review: exactly27 development/validation rows, final256 only,32steps,firing2101,
CPUfloat32. All raw states/probabilities/binary fields and per-case metrics retained.
NR3/NR4 saved comparison identities and raw hashes verified; no baseline reruns.
No TEST, longer horizon, threshold changes, additional training or checkpoint choice.

Damaged18 medians: overlap NR3 .912425 -> NR4 .943557 -> NR5 .970577;
closing3 .972679. Excess cells2077 -> 1354 -> 325; closing13. Requested-volume
absolute error median117 -> 71 -> 19cells; closing38. Missing cells recovered
2286 -> 2224 -> 1945; surviving input removed3 -> 4 -> 0. Thus higher overlap and
less excess do not mean more missing volume was recovered. All-nine pass counts
17/18 -> 13/18 -> 11/18; closing15/18.
Intact9 medianIoU .905059 -> .945415 -> .986333. Excess1111 -> 736 -> 172.
NR5 intact validity6/9, and not all examples reachIoU .99. Overall NR5 passes17/27
versus NR4 19/27 and NR3 25/27. NR5 family pass counts:access17,support23,thickness25,
all other families27 each. Failure locations have not been diagnosed in NR5;
do not carry forward NR4's disconnected-outlier explanation as an observed NR5 fact.

Frozen criteria: damaged overlap/excess/volume error PASS; intact overlap,
intact validity, damaged validity FAIL. Overall FAIL. Training-horizon alignment
coincides with substantial improvement in reconstruction in this one comparison;
no causal proof that mismatch alone caused prior excess, no stability beyond32
claim, no independent generalization claim. Compute and firing draws also changed.

D084 model-path decision: keep NR5 as best reconstruction research checkpoint among
NR3/NR4/NR5 on these cases, not a reliable admitted model. NR3 still has highest
all-nine pass count; no single winner on every criterion. Keep MG7 live Studio
and retain all learned variants with experimental labels. Close these incremental
trials: no automatic NR6, repeated penalty adjustments, extra seeds or paid runs.

Roadmap steps1-5 complete; step6 decision recorded above. Next implementation batch
can progress scale/diversity and deployment on the existing procedural path with
explicit method labels, while learned repair remains a separate research result.
This does not redefine the NCA research objective or prove scaling the NCA is safe.
A new substantial NCA training proposal needs an explicit rationale and compute
allowance. Do not present a procedural generator as a trained NCA.

Evidence: Codex outputs/NR5-Single-Trial-Review, including review-script.py,source.zip,
sealed-model.json,imports.json,result.json,comparison.json and27 raw observation pairs.
Summary: experiments/reports/NR5-single-trial-review.json; findings:
docs/next-phase/NR5_SINGLE_TRIAL_FINDINGS.md. Verified local archive:
Codex outputs/NCA-NR5-Review-2026-09-28.zip and receipt. No Drive operation/push/
publishing; same-disk copy is not off-device backup. Returned ZIP verified locally;
user can disconnect the Colab runtime. Historical current entries remain below.


## Current: plan steps1-3 complete; NR5 awaiting paid-run approval - 2026-09-28

User approved the eight-step compact plan. Training code review identifies16-step
supervision vs32-step review, without establishing causality. See
NR5_TRAINING_REVIEW.md. D083 prepares ONE fixed32-step training intervention using
unchanged NR4 loss/model/data/grid/optimizer. No new constraint family or loss term.
Original NR3/NR4 packages/code/evidence and Studio default remain unchanged.

New nca/repair_horizon.py, horizon_package.py, scripts/colab_repair_horizon.py,
build_repair_horizon.py and tests/test_repair_horizon.py. Checkpoint identity binds
train_steps32 and rejects NR4 state. Diagnostic TRAIN boundaries also use32.
Focused2 tests pass, including counted32 updates and exact checkpoint restore.
Extracted-package CPU8-update rehearsal 20260928T082606Z_927a3d1f7632 completes in
24.672s,zero child processes left. All export/member hashes
verified; notebook compiles and remains disarmed. This is readiness, not model
quality or a new GPU recovery proof. No heldout inference or full suite repeat.

Local deliverables: Codex outputs/NR5-Horizon/NCA-NR5-Horizon.ipynb and
NCA-NR5-Horizon-Package.zip. ArchiveSHA a19fcc85131343428cc3a1989dc8619707b212aae8d85ba48a98fc847eaa43e1
ManifestSHA 61c06403120aea05d123c4cf526c54aa566c40cbf5a155a9b4cc189e7ecbb34b
Review proposal: experiments/configs/NR5-horizon.json; readiness:NR5-readiness.json.

NEXT: ask explicit approval for one fresh seed1201,256 optimizer updates,32steps,
T4 job capped600s; setup/export/idle outside timer. Whole-VM loss before download
can lose job; runtime-bound checkpoints do not guarantee cross-VM recovery.
No Drive access or automatic retry authorized. User opens local notebook in Colab,
uploads ZIP, sets APPROVED_SEED_JOB=True only after approval, runs once, returns
ZIP+receipt. No assistant cloud action has occurred. Strict prior T4 software
admission retained; if environment differs, preserve failure and review.

Then review final256 only on27 development rows at32steps/firing2101 CPUfloat32;
compare NR3,NR4,unchanged,closing3. Require all intact IoU>=.99 and9/9valid;
damaged medianIoU>=.9435569333,valid>=17/18,excess<1354,abs request error median<=71.
No TEST, horizon sweep, threshold search or automatic promotion. This is one
bounded intervention; decide model path after outcome before further trials.
Steps6-8 (model choice, larger/diverse volumes, deployment polish) remain pending
and dependent on evidence. ROADMAP.md preserves the compact plan and status.
All artifacts locally archived; same-disk archive is not an off-device backup.


## Current: NR4 failure diagnosis and paired Studio view complete - 2026-09-26

User requested faster work in larger batches and approved diagnosis, comparison
UI and a training revision only if justified. This batch is complete.

All8 NR4 failures retain both raw and bulk interface hits; bulk_unreached=0.
Each has1-2 raw disconnected occupied cells,13 total across8 example observations.
All13 are excess relative to target; none is a target cell. Six examples have
one geometrically unsupported cell each. One cube-damage example has bulk
fraction0.8916667 below0.90. These are per-example counts, not13 independent
spatial events. Main route/bulk connectivity did not fail. Earlier broad wording
about worsened connectivity must be read with this localization. Strict all-nine
failure remains real; do not soften thresholds or relabel these as passes.

Added /static/repair-comparison/index.html with all27 saved cases: input/target,
NR3/NR4 side by side, and closing3. Includes eight-failure filter, NR4 disconnected
cell highlight, common camera/slices/context, recorded metrics and per-case
explanation. Original NR3 study retained; added navigation links only. Download
contains both model identities and source array hashes. MG7 stays default.

Exporter scripts/export_repair_comparison.py validates NR4 raw arrays and
threshold parity, reproduces all27 archived MT1 reports and reconstruction
metrics, and exports diagnosis masks. No model rollout, training, new damage,
TEST evaluation, cleanup operation or parameter search. UI loads without console
errors; eight-option filter and magenta defect highlight visually checked.

D082: no NR5 package yet. Geometry identifies where failure occurs, not the
learning mechanism producing it. Disconnected outliers explain access failures
but not736 false additions on intact examples or sub-.99 intact IoU. Blindly
increasing penalties again is not justified. Deleting islands might address some
symptoms but is not a learned correction and was not performed. No new loss,
architecture, grid size or paid run has been authorized or prepared here.

Next substantial task: review the training design against the preservation
failure before choosing one intervention; distinguish16-step training/32-step
review stability from spatial-error weighting. This batch does not establish
which is causal, and does not authorize a horizon sweep. Continue in coherent
batches, avoid repeated confirmation for local implementation, and still ask
before paid compute or every Drive action. No Drive, push or publication.

Evidence: deploy/static/repair-comparison/diagnosis.json and study.json;
Codex outputs/NR4-Failure-Diagnosis contains preserved diagnosis and verified
batch archive. Old raw NR3/NR4 model evidence is unchanged. Same-disk archive
is not off-device backup. Resume from this entry; prior current entries are history.


## Current: NR4 completed; preservation criteria NOT met - 2026-09-26

User explicitly approved the single256-update T4/600s job after preparation.
Returned run20260926T083017Z_51b29cc254ff completed in35.109881s,368MiB reserved.
Receipt/archive SHA25622bb26893e7fa4bba2e93f0110bd2c32a737fc6a0b248079439cbd9a4354b8de
and all1036 payload hashes verified. Final256 checksum, objective, study manifest,
seed, GPU device and optimizer-step/sampler cursors match. Receipt first arrived
with the old NR3 ZIP; waited for matching NR4 ZIP, never treated NR3 as NR4.

Final model only:27 validation rows,32steps,firing2101,CPUfloat32. All raw8-channel
states, probabilities, binary fields and per-case metrics saved; no TEST inference,
new training, intermediate selection or threshold search. Reused NR3 observations
and baselines; all27 case identities and NR3 array checksums verified.

Damaged18: NR3 -> NR4 medianIoU0.912425 -> 0.943557; false-positive cells2077 ->
1354; median absolute requested-volume error117 -> 71cells; recovered missing
cells2286 -> 2224; surviving cells removed3 -> 4. All-nine passes17/18 -> 13/18.
Closing3 remains higher overlap0.972679 and15/18pass on these damaged examples.
Intact9: medianIoU0.905059 -> 0.945415; excess1111 -> 736cells; passes8/9 -> 6/9.
None reaches the .99 intact overlap criterion (best0.971941639).
Overall all-nine passes25/27 -> 19/27. NR4 family pass counts:access19,support21,
thickness26; remaining six families27 each. Output legality/spill are projected,
not learned guarantees. Geometric support is not mechanical certification.

Frozen NR4 criteria: intact overlap FAIL, intact validity FAIL, damaged overlap
PASS, damaged validity FAIL, damaged excess PASS, damaged volume error PASS.
All conditions were required; overall FAIL. This is a tradeoff, not a successful
preservation revision or a demonstrated causal explanation of the failures.
One seed on reused development cases gives no independent generalization claim.
Do not compare numerical NR3 and NR4 loss values as if the objectives were equal.

D081: preserve both models as experimental and keep MG7 live Studio default.
No additional GPU runs, retry, loss retuning, grid/architecture expansion or model
promotion authorized. Next useful local work is to inspect saved failing shapes
and show NR3/NR4 differences in Studio, without running another experiment.
Use those saved fields to distinguish disconnected additions from bulk/interface
failures before proposing another training change. No conclusion yet that larger
models, more steps, or stronger penalties would fix this tradeoff.

Evidence: Codex outputs/NR4-Single-Trial-Review; tracked summary
experiments/reports/NR4-single-trial-review.json and NR4_SINGLE_TRIAL_FINDINGS.md.
Review helper and source snapshot are in evidence folder; archive
Codex outputs/NCA-NR4-Review-2026-09-26.zip with verified member-hash manifest.
Keep previous NR3/NR4 preparation and receipt-only records. No Drive operation,
push or deployment. Same-disk archive is not off-device backup. Colab may be
disconnected now that the returned archive has been locally verified and copied.


## Current: NR4 preservation revision prepared; GPU run NOT authorized - 2026-09-26

User approved preparation after the proposed focused loss revision. D080 adds a
separate PreservationSession, loss, driver, package verifier/builder and notebook.
NR1/NR2/NR3 source/math and Studio remain unchanged. No GPU or Drive operations.

Old loss B=0.5 positive BCE+0.5 negative BCE. New loss is
B+0.5 negative BCE+1.0 B when occupancy equals target. Damaged-example class
weights are0.5 positive/1.0 negative; intact weights1.0 positive/1.5 negative.
Intact detection is training-only; no target or new flag enters inference.
This also changes total gradient scale and intact-example weighting; it is not
just normalized class balancing. Gradient clipping/Adam can interact with it.
Coefficients are a fixed hypothesis, not tuned or claimed optimal. Stronger
preservation may reduce repair. No tenth constraint family is introduced.

Keep fresh seed1201,32grid,16 training steps,256updates,Adam.001,all81 TRAIN rows,
context,architecture,sampler and output projection. Do not fine-tune NR3 weights.
One proposed T4 job at600s max; setup/export/idle time extra. Strict NR2 software
admission remains. Notebook APPROVED_SEED_JOB=False. Whole-VM loss risk remains;
no cross-VM exact recovery or automatic retry. Separate Drive permission required.

Local readiness:2 focused tests pass; extracted package CPU8-update rehearsal
20260926T075836Z_224ef5b6e97a completes in21.406s with zero active child
processes after cleanup. Full exported evidence hashes verified. This rehearses
wiring, not quality or a new GPU-recovery proof. No heldout inference. No broad
suite repeated because historical implementation is untouched.

Package: Codex outputs/NR4-Preservation/NCA-NR4-Preservation-Package.zip
SHA256 b53abdd586f7346c00712493ff2538a1e7ae75af3aea427526315a0d7b5342f3
Notebook: same folder/NCA-NR4-Preservation.ipynb (disarmed, code compiled).
ManifestSHA fca07109ab66c32658bc30a356721cb724bc3c31fa1086711df7482e723a4a19
Read START-HERE.md and PRESERVATION_PROTOCOL.md. Raw CPU trace and checkpoints,
source hashes, frozen config and readiness report are retained. Same-disk archive
is not a Drive backup. Private reports remain ignored; no push/publish.

NEXT: ask approval for ONE seed1201/256-update T4 job capped600s, including the
stated runtime-only loss risk. User operates local notebook in Colab; no assistant
Drive access implied. After returned ZIP+receipt verification, score only final256
on27 validation rows,32steps,firing2101 CPU. Reuse all prior baselines, no TEST.
Development criteria are frozen in experiments/configs/NR4-preservation.json:
all9 intact rows IoU>=.99 and valid; damaged medianIoU>=NR3,passes>=17/18,
false additions<2077 and median absolute request error<=117cells. All required;
not formal deployment admission. Compare closing3 candidly regardless of outcome.
This split has informed the loss and is now development evidence. No auto retry,
threshold tuning or additional seeds. Document outcome, then decide the next step.


## Current: saved NR3 comparison available in Studio - 2026-09-26

Added /static/repair/index.html, linked from live-v2. All27 D078 validation
examples show input, closing3, final trained NCA and procedural teacher at a shared
scale. Orange highlights excess relative to target; optional blue overlay shows
missing target volume. Axonometric, XZ/XY slices, cutaway, context, metrics and
all-nine results are available. Learned output stays explicitly experimental;
MG7 remains the live generator. No further inference/training/TEST evaluation.

Exporter scripts/export_repair_review.py checks each observation array hash,
complete validation membership, binary threshold parity and reconstructed geometry
against archived repair metrics. Closing3 is reconstructed by the unchanged
baseline function; existing recorded nine-family reports are reused. It exports
portable display data at deploy/static/repair/study.json. Source evidence remains
Codex outputs/NR3-Single-Trial-Review; model/checkpoint and observation hashes are
included. The target is a procedural teacher, not architectural ground truth.

Validation:27 complete exports match recorded scores. Browser loads without
console errors; case selection, missing overlay and vertical slice controls
checked. Visual review confirms common scale and rendered volumes. These are UI
checks, not additional research experiments. No new regression suite was needed
for this isolated static view. No cloud operations, push or publish.

Next: user can inspect learned excess growth alongside the saved target. Before
another learning experiment, propose one focused preservation change and a
bounded compute allowance; do not launch more runs automatically. Keep all
previous evidence and D079 experimental status.


## Current: NR3 single trial completed and reviewed - 2026-09-26

D078 review is complete. Returned GPU run 20260926T073942Z_47ef54dc1d67 completed
256 updates, seed1201, in39.352s controlled wall time,368MiB peak reserved.
All1036 payload hashes and archive SHA256 verified; final checkpoint identity,
checksum and completed cursor verified with weights_only loading. Earlier run
20260926T073835Z_b87bca3faa5d failed GPU admission before any training; preserve both.

Exactly27 VALIDATION examples evaluated, final checkpoint256 only,32 steps,
firing2101, CPU Torch2.8.0+cpu/NumPy2.5.2. No TEST inference, new training,
intermediate checkpoint sweep or formal three-model gate. All27 raw states,
probabilities, binary fields, per-case metrics, source snapshot and helper saved.

Damaged18 medianIoU: model0.912425, unchanged0.874856, closing3 0.972679.
All-nine passes:17/18,6/18,15/18 respectively. Model recovers2286 missing cells,
adds2077 false-positive cells and removes3 surviving cells across these18 inputs.
Intact9: model medianIoU0.905059,8/9 pass;1111 false-positive additions.
All27: model25/27 pass versus unchanged15/27 and closing24/27. Model failures
are access/support. Median absolute requested-volume error119cells versus87
unchanged and6 closing. These are related validation cases, not independent
replications. Better contract pass count does not establish better reconstruction.

Decision D079: close this bounded trial; no more automatic tests/training.
Retain MG7 Studio default; learned model remains exploratory. Next implementation
is an evidence-backed comparison view of damaged input, simple closing, learned
output and target, with false additions/removals and nine-family results visible.
Use saved outputs; do not generate more evaluations. For later learning work,
prioritize preserving intact mass and controlling excess growth within existing
nine families; do not claim a larger grid or a new architecture solves this result.
The mechanism behind excess growth is not established by this one review.

Evidence: Codex outputs/NR3-Single-Trial-Review. Project artifact-copy attempt
was denied by filesystem permissions before copying; use the verified Codex archive. Summary tracked at
experiments/reports/NR3-single-trial-review.json; findings in
docs/next-phase/NR3_SINGLE_TRIAL_FINDINGS.md. No Drive operation, deployment,
push or new GPU job. Same-disk verified copies are not off-device backup.
User may disconnect Colab. Next task: implement the comparison view locally.


## Current: user reduces NR3 to ONE exploratory trial - 2026-09-26

User explicitly requested: "ok too many tests. lets wrap it up in only one quick
test and move forward". D078 and NR3-single-trial.json supersede D077 execution
scope. Run ONLY seed1201 once with existing unchanged NR3 notebook/package:
256 updates,600s maximum job execution, setup/download/idle allocation extra.
The request authorizes the previously proposed single bounded job; do not ask
again for the same compute scope. No assistant Drive/Colab access is inferred.
User can set MODEL_SEED=1201 and APPROVED_SEED_JOB=True and execute once.

Cancel seeds1202/1203, intermediate-checkpoint evaluation, multiple firing/horizon
sweeps and the formal TEST stage. Do not invoke the existing three-model evaluator
or weaken its checks to pass this one-model result. Preserve that protocol as history.
After downloading and verifying seed1201 ZIP/receipt, inspect final checkpoint256
only on all27 VALIDATION rows,32 steps,firing2101,CPUfloat32. Reuse existing predict,
MT1 and repair-metric functions in a separately labeled bounded assessment. Save
raw fields and compare with both frozen unchanged/closing3 baselines. No TEST data.
This is one trial with a results review, not a new training run or parameter sweep.

Report actual geometric improvement, intact damage, all-nine validity and volume
errors candidly. One seed cannot establish replicated reliability or pass D077's
three-model gate. Move to the next implementation decision after this review;
do not automatically extend training/retry/add experiments to obtain a pass.
If repair is poor, retain MG7 and label learned output exploratory instead of
silently promoting it. Preserve success/failure evidence and original plans.

Existing notebook: file1BjiTlIePcBrFUq1FZ4ZZMPuAZEAER1pY in the dedicated Drive folder.
Local ZIP: Codex outputs/NR3-Quality-Study/NCA-NR3-Quality-Package.zip.
No notebook/source/package bytes were changed, no cloud action or training started
by this scope update. New Drive operations still need exact per-action approval.
Decision copy: Codex outputs/NR3-Single-Trial-Decision/decision.json.
Earlier current entries and three-seed guide text are historical where conflicting.


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
