# Next-phase change log

Append meaningful changes, including failures and unresolved limitations. Historical documentation remains intact.

## 2026-09-13 - M0 foundation implementation

- Created `next-phase/foundations` from ac913b9; preserved 50 original files in a hash-verified local archive at `.local-artifacts/source-snapshots/before-next-phase-20260913T192856Z/`.
- Ignored the local next-phase report in Markdown/PDF and local artifact/cache directories. The existing user-supplied studio HTML remains untracked and unchanged.
- Added the working agreement, tracked plan, decision register, experiment policy and durable resume instructions.
- Added append-only experiment creation, artifact copying/hashing, explicit outcomes, linked retries, source snapshots, integrity verification and resumable non-overwriting mirror transfers.
- Added independent `binary_v1` primitives for endpoint connectivity, material legality, ground openness, eroded-core fraction, and geometric support. These do not replace the historical training/evaluation definitions yet.
- Added regressions for disconnected elevated entrances, corner contacts, empty fields, thin/solid material, blocked ground, floating components, corrupt artifacts and interrupted backup transfers.
- Centralized historical checkpoint/config loading using repository-relative paths and `weights_only=True`. Confirmed the real checkpoint embeds `street_levels=6`; external JSON says 2.
- Replaced the smoke test's missing Model A path and silent-success behavior with a real Model C load and explicit tensor/frozen-context/bounds assertions.
- Fixed scene generation when optional `gap_facing_x` is null, matching the UI's newly added buildings. Missing metadata does not create guessed facade anchors.
- Added API and real-checkpoint regressions and a verification runner that records logs, source snapshot, per-test failures and final status.

Validation: final recorded verification and commit are pending in this entry until the run below is appended. No GPU training, objective changes, retraining, or website publishing occurred.

Environment/setup observations: repository writes required narrow filesystem permission; Git metadata required a branch-write approval. CPU PyTorch was installed in an isolated project virtual environment. Early archive attempts failed due to execution-directory permissions; no archive was claimed complete until all 50 file hashes verified. Initial dependency-light tests passed (12 cases); the first full runtime suite passed (15 cases). A backup-interruption regression was subsequently added, requiring a fresh recorded run. FastAPI's test client emits a dependency deprecation notice for httpx; it is retained in the test evidence.

### Recorded acceptance result

Run `20260913T194223Z_ad4ee8ee60f3`: 16 tests, 0 failures, 0 errors, 0 skipped; standalone smoke launched outside the repository root exited 0. The archive includes dirty-source snapshot, complete test output, package/runtime provenance and final metrics. The source hash is the pre-commit state; the local milestone commit contains this verified implementation plus the final handoff documentation. The interrupted-copy test verified that a partial transfer does not publish a partial destination payload and can be retried. Google Drive itself has not yet been connected or verified.

## 2026-09-18 - M1 step 1: scene and geometry contract

- Added `nca/contract.py`, contract version `scene_v1`: axis order `(z, y, x)`, ground plane at `z=0`, world units through `voxel_size_m`, half-open `[start, end)` extents, entrance anchor corner and extent, strict `field > threshold` binarisation, canonical byte-stable serialisation and a scene hash.
- Added derived boolean regions read out of a rollout state rather than re-derived: `permitted`, `protected`, `support_boundary` and per-entrance `endpoints`. `verify_state_matches_scene` reports, rather than raises, when the realised frozen channels differ from the declared scene.
- `permitted` reproduces `LocalLegalityLoss.compute_legality_field` in boolean form. Equality is asserted on all six reference scenes; the historical field was confirmed to be strictly binary, so thresholding it loses nothing.
- Added the frozen set `experiments/scenes/reference_v1/` with six scenes and a hash manifest, plus `scripts/build_reference_scenes.py`, which refuses to replace an existing scene file without `--force`. The set includes `ref-05-sealed-partition`, a negative control in which no legal route between the entrances exists.
- Added `tests/test_contract.py` (28 cases, NumPy only) and three real-checkpoint cases in `tests/test_runtime.py`. Local suite count rises from 16 to 47.
- Documented the contract in `docs/next-phase/GEOMETRY_CONTRACT.md`; recorded decisions D007 and D008.

The contract deliberately follows Model C rather than the earlier Step D specification: `street_levels=6` from the embedded checkpoint configuration, and anchor-based street protection rather than Step D's explicit 3 m pedestrian and 12 m no-go strips. `ceiling_z` is validated but derives no region and is null everywhere, because a height ceiling is not one of the nine existing families. No constraint family, objective or training behaviour was added or changed.

### Divergences found while writing the contract, not repaired here

Recorded now so that M1 steps 3 and 4 measure them instead of rediscovering them. No historical file or result was altered.

1. Firing: `UrbanPavilionNCA._step` masks the update delta and only while `self.training`; `grow()` forces `eval()`, so serving applies no internal mask, while `deploy/server.py` blends whole states after the step when `fire_rate < 1.0`. Masking a delta and blending a state are different operations.
2. `z_taper_strength` and `z_taper_floor` exist in both configurations and are referenced nowhere in `deploy/model_utils.py`; whatever taper shaped training is absent from serving.
3. Serving reuses `corridor_mask_epochs` and `corridor_mask_anneal`, which are training epoch counts, as rollout step counts.
4. `corridor_seed_scale` is 0.15 in the checkpoint and in the external configuration, and 0.005 as the serving request default, a factor of thirty.
5. `/generate` writes `request.update_scale` into the shared `config` dictionary and restores it afterwards, so concurrent requests can observe one another's settings. This locates the already-planned per-job configuration defect.

### Recorded acceptance result

Run `20260917T222043Z_d6902ec4a5dc`: 47 tests, 0 failures, 0 errors, 0 skipped; standalone smoke launched outside the repository root exited 0. Recorded on the project checkout, branch `next-phase/foundations`, parent commit `b841991105ad3c7b4c0e3c8d24726b297cf6f0d9`, Windows 11, Python 3.12.14, torch 2.8.0+cpu, NumPy 2.5.2. Summary at `experiments/records/20260917T222043Z_d6902ec4a5dc.json`; source snapshot and raw test output in the matching `.local-artifacts/runs/` folder. The recorded working-tree status is dirty by design: the source hash is the pre-commit state of this milestone's files, and the commit carrying them follows.

The suite count rose from 16 to 47. This is a regression run over synthetic geometry, archive integrity, real Model C execution and the API path. It establishes no architectural-quality, latency or corrected-training claim.

The same suite was first run to completion in an isolated Linux sandbox mirror on Python 3.12.3 during development. That rehearsal carries no run ID and is not evidence: `scripts/verify_foundation.py` requires real Git provenance and refuses to run outside the repository. The frozen scene set was separately confirmed intact on the checkout, all six files reporting `unchanged`. No GPU training, objective change, retraining, push or publication occurred. Google Drive remains unconnected and unverified.

## 2026-09-18 - M1 step 3: shared rollout with named historical profiles

- Added `nca/rollout.py`, version `rollout_v1`: one rollout implementation with every behavioural axis declared in a `RolloutProfile`, plus `historical_training`, `historical_evaluation` and `historical_serving` constructors, each carrying a provenance string, explicit notes and a version.
- Established correctness by exact agreement rather than inspection. The serving profile is bitwise identical to the legacy `/generate` loop across six request variants (request defaults, stochastic firing, no noise, training seed scale, past the mask schedule, altered update scale); the evaluation profile is bitwise identical to `model.grow(seed, steps=50)`. The legacy paths are unchanged and remain the reference.
- The rollout no longer mutates the caller's seed state or the shared `config` dictionary: the update-scale override is scoped and restored in a `finally` block, where the legacy handler leaked it on an exception. This narrows but does not close the per-job configuration item, which stays M4 work.
- A corridor target is required by any profile that uses one and refused by any profile that does not; `rng_source` must match how the stream is drawn; `profile.replace(...)` renames its result so a variant cannot be recorded under a historical name; module mode is restored even when a step raises.
- Added `tests/test_rollout.py`, 15 cases. Local suite total rises from 47 to 62.
- Documented the profiles in `docs/next-phase/ROLLOUT_PROFILES.md`; recorded decisions D009 and D010.

### Corrections to the 2026-09-18 M1 step 1 entry

Reading `notebooks/model_c/NB02_AllConstraints_v3_1_C.ipynb` contradicted one claim in that entry and showed two others to be incomplete. The original entry is left as written; these supersede it.

1. Divergence 2 was wrong. It stated that a training-time z-taper was absent from serving. `z_taper_strength` and `z_taper_floor` are referenced nowhere in the notebook or the deployment. They are dead configuration keys; nothing was lost at deployment and no taper shaped training.
2. Divergence 1 understated the case. Training adds no per-step noise at all — neither the training loop nor the notebook's `_step` contains any — so serving's per-step noise injection is a deployment addition rather than a difference of degree.
3. Divergence 3 understated the case. The training mask is applied once to the seed before the rollout and gated on the epoch index; serving applies it inside the loop on every step, gated on the step index. The units differ and so do the location and the frequency.

### Further findings from the historical notebook, not repaired

1. The recorded historical evaluation used no corridor scaffold. `evaluate` computes the corridor target for scoring only and then calls `model.grow(scene, steps=50)`, so the figures in `v31_evaluation.json` came from a rollout that received neither training's 0.15 seeding nor any mask. They are not a measurement of the configuration the model was trained under.
2. Model C never saw a ground-type access point. In the notebook's scene generator every access point is written with `type: 'facade'`; `n_ground_access` is counted into the total but never changes the type, so the ground-anchor branch of `_generate_anchor_zones` never executed during training. The deployed interface offers ground access points, which produce anchor geometry absent from training. This pins down the access-sampling defect already listed under M2.
3. Training placed access points from `z=3` upward with `street_levels=6`, so some were typed `facade` while sitting below street level — a combination the `scene_v1` contract rejects. Training buildings also always spanned `y` from 0 to a sampled depth.
4. The deployed scene generator takes explicit parameters where the historical one took a difficulty label, so no user scene is drawn from the training distribution, and neither is the frozen `reference_v1` set. E0 on that set can measure how the profiles differ from one another; it cannot reproduce the historical aggregate, and a poor score on it may indicate out-of-distribution input rather than a worse model. Whether to add a second frozen set sampled from the `easy` generator at fixed seeds is an open decision.

### Validation status

The suite was run to completion in the isolated Linux sandbox mirror: 62 tests, 0 failures, 0 errors, 0 skipped. That is a development rehearsal with no run ID. The authoritative recorded verification for this milestone is pending on the project checkout. An unrecorded orientation observation, carrying no evidential standing: on `ref-02-facade-pair-and-ground` with one shared seed and 30 steps, the three profiles produced 3005, 362 and 1985 material voxels, with perfect legality and no entrance connectivity in any of them. No GPU training, objective change, retraining, push or publication occurred.

## 2026-09-18 - In-distribution scene set and declared relaxations

Follows the user's decision to have E0 run in-distribution as well as on designed scenes. See `SCENE_SETS.md` and decisions D011 and D012.

- Added `nca/legacy_scenes.py`: a transcription of the historical `easy` sampler, plus `legacy_seed_state`, which builds the seed state the notebook generator would have produced. Both are verified against the notebook generator executed as an oracle in `tests/test_legacy_scenes.py`, not against a reading of it. The notebook is read only.
- Added the frozen set `experiments/scenes/legacy_easy_v1/`: twelve scenes from seeds 0-11, consumed in order with no cherry-picking, plus `scripts/build_legacy_scenes.py`. The manifest records the difficulty parameters, the accepted seeds, every rejected seed with its reason, and which seed-state builder the set requires.
- Added named relaxations to the contract. A scene may declare `facade_below_street_band`; the name is covered by the scene hash, an unknown or repeated name is refused, and face adjacency is still required. Four of the twelve legacy scenes declare it; the designed set declares none, enforced by test.
- An empty relaxation list is omitted from the canonical form, so `reference_v1` kept its canonical hashes and its six files are byte-identical to when they were frozen. Verified: the set still loads and the build script still reports `unchanged` for all six.
- Added `tests/test_legacy_scenes.py`, 14 cases. Local suite total rises from 62 to 76.

### Behaviour change found in the deployed scene generator

The notebook wrote ground anchor zones only for an access point typed `'ground'`. The deployed `_generate_anchor_zones` also writes them for any access point with `z < street_levels`. Because the historical generator typed every access point `'facade'` and placed some below the street band, the deployed generator produces a wide ground anchor footprint for exactly the scenes where training produced none. Anchors feed the legality field, so this widens what the model is permitted to grow. Tests confirm the divergence occurs exactly when an entrance sits below the street band, that the deployed rule only ever adds anchors, and that the permitted region strictly grows. Nothing was repaired; `legacy_seed_state` is used for the legacy set and the deployed generator is untouched.

### Orientation observation, not a record

Unrecorded, no run ID, one seed, 50 steps, all twelve legacy scenes: `historical-training` produced a mean 1076.4 material voxels and connected the entrances in 10 of 12 scenes; `historical-serving` 805.8 and 10 of 12; `historical-evaluation` 27.7 and 0 of 12. Legality was perfect and geometric support complete throughout. The difference is the corridor scaffold, which training seeds at 0.15 and serving at 0.005 while the evaluation profile seeds nothing.

This bears on how `v31_evaluation.json` is read, since the evaluation profile produced it: `avg_coverage` of 0.04 is consistent with near-empty output. Its `avg_access_reach` of 0.62 does not contradict the zero above — the historical metric measured reachability through ground-level void, which an empty design satisfies trivially, while the figure above measures connectivity through the grown structure. The historical figure is not wrong; it measures something an empty result scores well on. E0 is what turns any of this into evidence.

### Validation status

Sandbox mirror only: 76 tests, 0 failures, 0 errors, 0 skipped. No run ID; the authoritative recorded verification is pending on the project checkout together with that of M1 step 3. No GPU training, objective change, retraining, push or publication occurred.

## 2026-09-23 - Dedicated Drive folder and approval boundary

- Created the project Drive folder with explicit user authorization; connector returned folder ID `1fS34Yy0-oMzSxWaYJFiPTkGgrZstgc0H` and its URL.
- Added the persistent one-folder-only and ask-before-every-operation rule to `AGENTS.md`; recorded D013 and refreshed the resume handoff.
- Drive reads and writes both need prior approval. No uploads, subsequent remote reads, sharing changes or backup verification were performed. Connector scopes were not changed.
- Validation: successful folder-creation response and local documentation readback. Documentation-only change; model tests were not rerun. Existing implementation work remains untouched.

## 2026-09-23 - Local acceptance of M1 rollout and scene-set milestone

Recorded run `20260922T223533Z_9c5bfe017c4c` completed on the actual Windows checkout: 76 tests, zero failures, errors or skips; out-of-directory Model C smoke exit 0. Parent commit: `ddc81000f44dcdaf9d58c10f23707745803f756c`; dirty-source snapshot captured before testing. UTC run date is September 22; local date in Berlin is September 23. Raw output and source archive are retained under the run directory and the small summary is in `experiments/records/`.

RunStore verification returned no integrity problems. Both frozen manifests loaded without modification: six reference scenes and twelve legacy scenes. Historical Model C files remain unchanged against `ac913b9`; both next-phase report formats remain ignored. No model/optimizer update, paid compute, publication, push, or Drive access occurred.

### Remaining rollout limitations found during acceptance review

These passing regressions cover named historical defaults; they do not validate all parameter variants. `run_rollout` scopes `update_scale` only, while a training-mode `model._step` reads `model.config['fire_rate']` and consumes the global random generator. Consequently a changed profile fire rate is not applied to delta masking, and an explicit generator does not control that path. Also, inconsistent module-mode/firing combinations are currently accepted, so a training-mode profile declaring no firing can still fire internally. Repair or reject these combinations and add targeted behavior tests BEFORE E0 firing/randomness ablations. The exact historical-training path lacks a notebook-loop oracle test; add that before claiming training replay parity. The serving parity test compares against a copied legacy loop, not a live HTTP generation request.

This acceptance establishes local regression behavior only, not improved architectural quality or completion of E0. Existing code and test evidence are preserved in a local milestone with these limitations explicit.

### Checkout integrity fix

Git warned that Windows checkout conversion would turn the frozen JSON scenes
from LF into CRLF. Because manifests hash exact file bytes, this would invalidate
otherwise unchanged scenes after a fresh checkout. Added `.gitattributes` to keep
`experiments/scenes/**/*.json` at LF. Scene contents and manifests were not
regenerated. Verify both sets in the fresh bundle restore before delivering the
backup; this check covers the checkout fix independently of the regression run.

## 2026-09-23 - E0 preparation in progress

- Recorded the user's decision to keep the milestone archive local (D014). No
  Drive access occurred; the existing Drive scope/approval rules still apply.
- Added `rollout_v2`: scoped firing-rate overrides, explicit RNG forwarding for
  delta masks, and rejection of inconsistent firing/module modes. Legacy model
  `_step` gains an optional generator argument; existing callers retain default
  behavior and checkpoint parameter shapes are unchanged.
- Added original-notebook forward parity checks before any losses/optimizer
  operations, across mask phases and batch sizes; added rate/RNG behavior checks.
- Failed verification `20260922T225408Z_f67923028c9a` retained: 80 tests, three
  oracle subcase errors caused by a missing `_compute_mst_edges` helper in the
  isolated notebook namespace. Fixed dependency extraction. A linked retry is
  in progress; this entry is not a claim that the retry has passed.
- Added local E0 protocol/runner with incremental per-case evidence and explicit
  failure records. Protocol D015 is frozen before recorded E0 results.

### E0 preparation acceptance

Linked retry `20260922T225534Z_1c995e858265`: 80 tests passed, no failures/errors/
skips, smoke exit 0. Final expanded suite `20260922T225856Z_26be76516805`: 82
passed with the same clean result, including E0 empty-output and denominator
checks. The failed oracle setup attempt remains intact and linked; no experiment
result was replaced. E0 is prepared but has not produced recorded results yet.

### E0 started

Run `20260922T230120Z_76f3b4677e8f`, source commit `474bf53`, started with the
frozen 270-case E0_v1 matrix. Results are appended per case under its local run
folder. Do not treat it as complete until result.json confirms completion.
The source snapshot was captured before later report-writer/design-note edits.
Added report generation from verified artifacts and a bounded corridor-fix plan;
neither changes the operator used by the running baseline.

## 2026-09-23 - E0 completed and retained locally

Run `20260922T230120Z_76f3b4677e8f` completed all 270 cases, zero execution failures,
in 1219.3 seconds total CPU wall time (including evaluation/storage; not a
latency benchmark). Source commit: `474bf53`. Full states, seed states, corridor
fields, material masks, case metrics, all settings, timing and source snapshot
are retained in the run archive. No weights were updated. Original notebook,
checkpoint, published historical scores and frozen scene files remain unchanged.

Added `scripts/report_e0.py`, `nca/e0_analysis.py`, the detailed report and target
analysis JSON, E0_FINDINGS.md and CORRIDOR_FIX_PLAN.md. The report reads verified
artifact records and the recorded source snapshot rather than trusting current
working files. All artifact hashes passed verification. The post-run target audit
is labeled as such; it did not alter the predeclared E0 matrix. A fresh report
render was reviewed against the raw group totals and per-seed/paired results.

Key findings and next action are in D016. The old assertion that the two firing
forms coincide only at rates 0 or 1 was too strong; a visible clarification now
states their possible equivalence under invariant projection and identical masks.
This documentation correction does not change the recorded experiment.

The 82-test preflight and both earlier verification attempts remain preserved
with verified artifact hashes. No need to rerun model tests for report-only
additions. Historical Model C files have no diff against `ac913b9`. Reports stay
ignored where required; the user studio concept and unrelated checksum file are
not staged. All new evidence is preserved in a local commit and a local backup;
no Drive operation, paid training, push or deployment was performed.

### Final report verification

The detailed report and target-audit JSON exactly match a fresh rendering from
verified artifacts. Full continuous final-state hashes also match across all
three historical-evaluation repeats for each of the 18 scenes. The corresponding
verification receipt is stored beside the report. This confirms deterministic
repeats, not three independent evaluation samples.

## 2026-09-23 - Versioned corridor corrections, preflight

Added nca/corridor.py with corridor_bounded_v1, preserving the original operator
as an unchanged oracle, and the separately versioned nca/legal_corridor.py.
Legal routing operates on explicit entrance IDs and the existing permitted
six-neighbor graph, records paths and infeasibility, and avoids the ground-route
height clipping conflict. scripts/run_corridor_comparison.py implements the
predeclared C1_v1 matrix and immutable per-case evidence (CORRIDOR_PROTOCOL.md).

Regression run 20260923T000140Z_a7ff7de54684 passed all 94 tests with no failures,
errors or skips; checkpoint smoke exit 0. Twelve added checks cover bounded
extent/window oracle/reversal/batch isolation, radius-zero legacy parity,
obstacle detours, paths longer than 64 updates, diagonal disconnection, touching
IDs, blocked/ambiguous endpoints, illegal or isolated dilation, and real-scene
batch parity/input preservation. These are geometry/software checks, not model
training results. Historical source/checkpoint/scenes remain unchanged.

### Retained C1 attempt and Windows evidence-path repair

C1 attempt 20260923T000715Z_1e06516e9763 finalized as failed after all 54 target
records and 7 registered rollout records. A long scene/profile/version filename,
combined with the run directory and immutable-publication suffix, exceeded the
Windows path limit. The attempted eighth case could not publish its case record;
its available artifacts and the outer traceback remain retained. This is an
archive-path failure, not a model-quality result. Changed this runner's payload
and case filenames to short deterministic SHA-256 prefixes; full case IDs stay
inside records. Retry gets a fresh run linked to this retained attempt. No
historical file or record is shortened in place.

## 2026-09-23 - C1 complete and objective conflict documented

Linked retry 20260923T000945Z_3cbdc3603a12 completed all 54 targets and 108 forward
cases from source commit 595c8e0 in 343.54 seconds. All 36 legacy cases match E0
bitwise; all registered artifact hashes verify. Both new target versions are
measured separately. See CORRIDOR_FINDINGS.md, the full per-scene report, target
JSON and verification receipt. Report generation consumes registered copies;
fresh renders exactly match the saved report and audit JSON.

Added scripts/audit_corridor_budget.py and a labeled post-hoc volume audit using
the recorded source notebook and arrays. All 18 legal targets fit less mass than
the historical 3% lower bound, exposing a conflict with zero spill. A first parser
attempt rejected the extra standalone demonstration constructor in notebook cell
20; source/error retained under .local-artifacts/analysis-attempts/20260923-c1-volume-parser-01.
The corrected parser selects the actual trainer initializer (cell 24). This did
not rerun or alter C1. Source-only/geometry findings and measured model limitations
are distinguished in D018 and LOSS_REPAIR_PLAN.md.

The original checkpoint does not benefit in reference-scene connectivity; legal
routing loses one legacy serving connection. No production defaults changed and
no quality gain is claimed for the checkpoint. The 94-test preflight remains the
applicable operator regression result; subsequent edits concern evidence naming
and reporting, exercised by the completed retry and report re-render checks.
Preserved the original notebook/checkpoint and user files; report stays ignored.
Milestone receives a local commit and verified local recovery archive. No Drive
access, paid training, push or deployment occurred.

## 2026-09-23 - Shared loss mechanics and L1 preparation

Added nca/losses.py (geometry_losses_v1), separate from the historical notebook
and fine-tuner. All nine families return [B] terms; context and empty-material
flags are explicit, and strict means refuse incompatible contexts. Distinguishes
coverage guide from material envelope, retains original mass bounds/denominator,
uses zero-padded zero-background thickness and single-source six-neighbor
bottleneck reach. Fixed masks/context are verified before gradients are computed.
No trained checkpoint, production default or new constraint family changed.

Initial verification 20260923T002822Z_8f68d5bbcaf7 passed 108 tests, no failures,
errors or skips and smoke exit 0. A subsequent addition checks scene-adapter
certification of guide connectivity and rejects missing entrance coverage. This
addition passed in 20260923T003247Z_0ce597b29bc4: 109 tests, zero failures/errors/skips, smoke exit 0. Added the preregistered
LOSS_PROTOCOL.md and scripts/run_loss_diagnostics.py: 72 context cases, three
short real-model gradient cases, six expected historical fine-tuner defect checks.
No optimizer updates. D019 records the scope and unresolved semantic limits.

## 2026-09-23 - L1 complete; real gradient limitation isolated

L1 20260923T003413Z_1da1202e4a7f completed the preregistered 72 objective contexts,
three real-model gradient cases and six historical defect checks in 29.25 seconds,
from commit fa66238. No optimizer updates. Final regression
20260923T003247Z_0ce597b29bc4 passed 109 tests, zero errors/failures/skips, smoke 0.

Added scripts/report_loss_diagnostics.py and LOSS_FINDINGS.md with complete
per-scene details. Artifact hashes verify; independent saved-array norm/cosine
recomputation and a fresh report/details render agree. Fixed radius-six envelopes
satisfy necessary bounds on 13/18 scenes; only 12 have valid contexts including
route feasibility. The five other feasible scenes remain capacity conflicts.
Historical fine-tuner shape errors and disconnected surface gradients are
confirmed at B=1 and B=2, recorded as expected historical defects.

Added scripts/trace_access_gradient.py. Its post-hoc run
20260923T003906Z_da05c8eed3f6 exactly replays L1's ground-reference output and
records the two active derivative cells before clamping. Both are legal and fired
but strictly negative before the lower clamp, explaining the zero parameter
gradient in this four-step case. This is local attribution, not a general claim
or an extra preregistered L1 comparison. Source/fields/results are preserved.

D020 keeps training gated on objective compatibility and useful derivatives.
No model changes or optimizer/recovery/training claims. Original notebook,
checkpoint, scenes and production defaults stay intact. Local milestone/archive
and refreshed handoff preserve progress; no cloud access or remote publication.

## 2026-09-23 - L2/R1 preparation and retained verification failure

Added explicit material interventions, budget contracts and CPU checkpoint
recovery utilities; originals remain intact. L2/R1 protocol is in
INTERVENTION_PROTOCOL.md. A first verification, 20260923T074523Z_f2efaaf7f135,
ran 118 tests with one fixture failure and no errors: the immutability assertion
passed a plain dict and an OrderedDict to the deliberately type-exact checkpoint
comparator. The model rejection path had returned before mutation. Corrected
the fixture to deep-copy the original ordered state dictionary, preserving the
strict comparison. Failed evidence remains intact; verification retry is linked.

Linked verification 20260923T074748Z_68688d661ffd passed all 118 tests with no
failures/errors/skips; checkpoint smoke exit 0. The added tests cover hard-forward
and RNG parity, finite differences for both interventions, explicit budget changes,
spill accounting, hard legality, exact optimizer/scheduler/four-RNG continuation,
no-overwrite checkpoints, metadata rejection and truncated-file rejection.

## 2026-09-23 - L2 interventions and R1 recovery completed

L2 20260923T075113Z_d95fabaf3776 completed 108 budget, 54 gradient and nine
absent-scaffold cases in 191.23 seconds CPU, with no optimizer updates. Artifacts
retain all fields/gradients, physical budgets and individual metrics. The gate
permitted only the prescribed recovery test. Source commit 95431de.

Added scripts/run_recovery_smoke.py (6ee2ed8). R1
20260923T075727Z_2233c0e51b9a completed in 23.94 seconds CPU. Four logical
updates (ten executions across uninterrupted, prefix, resumed, repeat branches)
pass seven exact checkpoint/trace/field comparisons. No trained-quality claim.

Added scripts/report_interventions.py and experiments/reports/L2-R1-interventions.md.
It verifies registered artifact hashes and independently recomputes gradient norms,
hard-pair equality and full recovery comparisons. Fresh reconstruction matches the
saved report. First report publication failed because experiments/reports did not
exist; source and failure explanation are retained in
.local-artifacts/analysis-attempts/20260923-l2-r1-report-01. Added explicit directory
creation before exclusive publication; no experiment was repeated or altered.

INTERVENTION_FINDINGS.md, D022 and RESUME record physical-budget implications,
unchanged projected-access failure, smooth background-mass issue, CPU recovery
limits and next actions. Original artifacts and user files remain intact. Local
Git milestone and hash-verified recovery archive preserve progress; no cloud use.

## 2026-09-23 - T1 audit prepared

Added target_candidates and analytical joint bounds in nca/target_audit.py, five
regression tests, frozen TARGET_AUDIT_PROTOCOL.md and scripts/run_target_audit.py.
Verification20260923T082336Z_b8c43fcf7723:123 tests pass,zero failures/errors/skips,
checkpoint smoke exit0. The full audit records static losses and direct occupancy
gradients, not learned quality. Original loss/model code remains intact.

## 2026-09-23 - T1 completed and verified

T1_v1 20260923T082527Z_845d2aa6aec0 (sourcef767975) completed432 static targets,
72 joint-bound records and36 occupancy-gradient cases in35.91 seconds CPU.
No optimization. Every scene, including sealed reference, retained. Added
scripts/report_target_audit.py, full experiments/reports/T1-target-audit.md,
verification and rerender receipts. Hashes, joint arithmetic, candidate volumes,
saved norms/cosines independently checked; nine zero-loss witnesses recomputed
exactly and fresh report rendering matches. No experiment/report attempt failed.

TARGET_AUDIT_FINDINGS.md and D024 document facade/budget conflicts, weak zero-loss
geometry and gradient-scale limitations. NEXT_EXPERIMENT_PLAN.md defines staged
semantic repair, calibration and E2 controls without claiming final weights or
paid-job readiness. Updated handoff and new local archive preserve all history.
Original notebook/checkpoint/losses/production defaults remain unchanged; no cloud.

User clarified uncertainty about the intended representation and requested advice.
D025 and the plan recommend architectural material generation first, usability
separately, with usable pavilion/bridge retained as the longer-term goal. Recorded
as recommendation rather than an invented user decision. No implementation scope
or production contract was changed by that clarification.

## 2026-09-23 - A1 facade allowance prepared

Added nca/facade.py, six regression tests, FACADE_PROTOCOL.md,18 frozen scene-derived
annotation sidecars and manifest, and scripts/run_facade_comparison.py.
Verification20260923T084059Z_6280ed836d9f:129 tests pass,zero failures/errors/skips,
checkpoint smoke exit0. Checks include original parity without allowance, blanket
penalty, finite differences, per-scene batching, metadata-only patch construction
and unchanged budget bounds. No production default or optimizer change.

## 2026-09-23 - A1 evidence verified; W1 prepared

Added report_facade_comparison.py and A1 report/verification/rerender receipts.
All432 target pairs reproduce prior values; other-eight-term equality, annotation
masks, bound arithmetic and72 analytical quotient gradients rechecked. No failures.
Added canonical LF checkout for hashed annotation JSON in .gitattributes.

Added budgeted_witness_v1, three regression tests and W1 protocol/runner.
Verification20260923T084753Z_489a9c4be020 passes132 tests,zero failures/errors/skips,
checkpoint smoke exit0. Determinism, legal adjacency, unchanged guide, infeasible
route refusal and absence of a binary mass within a fractional budget are tested.

## 2026-09-23 - W1 completed; experimental baseline and handoff preserved

W1 20260923T084933Z_eb2603cd79f7 (source2965502) completed18 scenes in4.90s CPU,
17 connected zero-loss witnesses, one retained incompatible sealed reference.
Added report_witnesses.py and W1 report/verification; independent ordered-addition
replay and recomputed nine-family values/binary metrics match all18 records.
FACADE_FINDINGS.md combines A1/W1 outcomes and limits. D028 selects the opt-in
experimental contract for calibration preparation. No optimization or learned-
quality claim. Updated plan/handoff and verified local archive preserve evidence.

## 2026-09-23 - K1 and retained regularizers prepared

Added nca/regularizers.py, five meaningful tests and REGULARIZER_AUDIT.md.
Verification20260923T092058Z_bba721fc054d passes137 tests,zero failures/errors/skips,
checkpoint smoke0. Includes extracted-notebook value/gradient parity, batch checks,
boundary finite differences, ground/floating/no-wrap cases and binarization intent.
Added CALIBRATION_PROTOCOL.md and scripts/run_calibration.py; all raw per-family
parameter gradients and source/coefficients are preserved for K1. No optimizer.

## 2026-09-23 - Research objective composition prepared during K1

Added nca/objective.py and four tests: complete coefficients, explicit regularizer
selection, incompatible-context refusal and pre-clamp saturation. Verification
20260923T092752Z_fa771e070c3d passes141 tests,zero failures/errors/skips,smoke0.
Prepared verified report/proposal writers and gated R2 composed-objective recovery
runner/protocol. These additions are independent of K1's preserved source snapshot.
They will consume completed K1 evidence; no partial outcome is published as final.


## 2026-09-23 - K1 completed, K2 frozen proposal and R2 exact CPU recovery

- Completed K1 20260923T092355Z_f657f2f3bdb9: 71 model-gradient cases and 51 budget
  probes. Independently checked saved vectors/norms/cosines, raw coverage and
  analytic budget derivatives; recomputed all 71 composed objective records.
- Recorded historical regularizer provenance and checkpoint coefficients. Added
  explicit research_objective_v1 with complete nine-family/three-regularizer
  weights and joint context validation. Original training/serving remain intact.
- Preserved the 137-test and 141-test successful runs. K1's harmless scalar logging
  warning is retained; later logging uses detach without changing derivatives.
- Prepared fixed K2 mapped_30 versus mass_3 proposal and directional estimates.
  The future trainer is not implemented or run; the proposal now fixes 16-step
  rollouts because four-step budget gradients were inactive. See D031.
- Completed R2 20260923T095240Z_652e01d22fee at source 0fca1b8: four logical/ten
  executed updates in four CPU processes, all seven exact recovery checks pass.
  Independently verified registered checkpoint trees, traces and field arrays.
- Added CALIBRATION_FINDINGS.md and refreshed resume/plan. Enforced LF for frozen
  experiment configuration JSON so its byte hash survives Windows Git restore.
  Exact Python source bytes remain recoverable from registered run snapshots.
- No paid training, cloud access, push, deployment or performance-improvement
  claim. Preserve all earlier runs and local archives; new verified calibration
  archive receipt is in .local-artifacts/milestones/<commit>-backup-receipt.json.

## 2026-09-23 - K2 actual-loop implementation and preregistration

Added explicit CPU sensitivity Session with shared recipe/seed scene order,
constant learning rate and strict source/config/runtime/input metadata. Added
fresh-process recovery gate, four-member training/evaluation coordinator,
per-update immutable checkpoints, retained logs and linked interrupted-run imports.
K2_PROTOCOL.md preregisters exact evaluation and time caps. New report verifier
will recompute all saved objectives and evaluation geometry metrics.

Verification20260923T102104Z_15b3cd2f4fd5:144 tests,zero failures/errors/skips,
checkpoint smoke0. Actual-loop recovery and study have not run at this entry.
Original proposal, notebook/checkpoint and serving path unchanged.
