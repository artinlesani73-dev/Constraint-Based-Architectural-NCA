# Next-phase change log

Append meaningful changes, including failures and unresolved limitations. Historical documentation remains intact.

## 2026-09-23 - F3 implementation

Added separate horizon_training session, runner, all-field verifier, frozen
proposal/protocol and four schedule/timing/metadata guard tests. Existing F2 code,
original checkpoint and production defaults untouched. Checkpoints identify full
schedule and completed cursor; a separate record states the next horizon.
Final-grid boundary reuse is explicit. Local cap3600s, gated on parity, actual
recovery and timing. No scientific run yet. An initial documentation patch failed
on a mismatched CHANGELOG heading; checked files before applying corrected patch.

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


## 2026-09-23 - Actual K2 sensitivity comparison completed and verified

- K2R20260923T102414Z_eaec7bd1510e, source33f6858:actual16-step training loop,
  constant rate and frozen scene order; three logical/eight executed updates in
  four fresh processes. All seven exact recovery comparisons pass independently.
- K2 20260923T102524Z_f5e1cc169dea:four17-update CPU trials,68 total,187 evaluations
  including original/W1 controls. No timeout/failure. Complete study505.98s.
- Verified all68 checkpoint metadata/update boundaries, recomputed all255 saved
  objective fields and all187 binary evaluation metrics/both common recipe totals.
  Verified exact snapshot hashes and fresh report rendering. Added descriptive
  per-scene transitions, explicitly post-hoc, with no new optimizer run.
- Expanded the report's binary support/bulk/threshold summary; this report-only
  change was outside training source hashes and was checked by actual report
  generation/recomputation. Core training source is unchanged after144-test pass.
- Recorded D033:neither setting promoted. Weight3's long-horizon gains are confined
  to legacy004/008; all five feasible reference scenes still fail connectivity.
  The stronger weight controls mass better but15/17 cases remain over budget.
- Added SENSITIVITY_FINDINGS.md and refreshed handoff/plans. Frozen historical
  K2-sensitivity.json remains unchanged, including prepared_not_run status; actual
  execution/result status lives in the new immutable run records, not a rewritten
  proposal. No direct-material optimization, GPU training or studio change yet.
- All evidence is preserved for a new verified local sensitivity archive. Linked
  interrupted-run imports and forced/mid-write failures were not exercised; K2R
  proves orderly completed-update continuation only. No remote/paid operations.

## 2026-09-23 - D1 direct optimizer and frozen comparison preparation

Added nca/direct.py and separate direct recovery/pilot/full runner, preserving all
original NCA code and objectives. Exact hard projection, unbounded raw variables,
weak-scaffold initialization and explicit Adam settings. DIRECT_PROTOCOL.md and
D1-direct.json freeze the pilot and timing-only full admission. Three field/
gradient regressions added. Run20260923T104700Z_c4dc9a0d6f64 passes147 tests,
zero failures/errors/skips,checkpoint smoke0. Actual direct recovery/pilot pending.
Report verifier prepared for full checkpoint/projection/norm checks and initial/
final objective/geometry rescoring. No NCA optimizer, cloud or paid operation.


## 2026-09-23 - D1 completed and saved-result viewer prepared

- D1R20260923T105015Z_484cf36351d7:four logical/ten executed direct updates,
  seven exact fresh-process CPU recovery checks pass. Source042b7c8.
- D1P20260923T105120Z_4d1e4d2d20d6:six cases/48 updates,39.84s; verified pilot
  p90 timing0.33365s admits frozen full run under799.51s estimate/900s cap.
- D1 20260923T105246Z_ab0d4a430b4c:34 cases/1088 direct updates,484.44s, no failed
  or timed-out cases. Verified every checkpoint/projection/saved gradient norm and
  all68 initial/final scored states. Intermediate objectives retained, not all rescored.
- Both recipes connect17/17. Weight30 has12 in-budget and5 below floor; weight3 has
  10 in-budget,1 below floor,6 above cap. All five references connect; weight30 keeps
  each in budget. Recorded all per-case failures and D035 without promoting a model.
- Added DIRECT_FINDINGS.md, refreshed plans/handoff. No NCA weights, original
  checkpoint or serving defaults changed. No new family, paid training or cloud work.
- Added assets/experiment_viewer.html and scripts/build_result_viewer.py. Generated
  standalone local HTML from221 registered result fields, all17 scenes/13 variants.
  Coordinates/metrics verified and JS syntax passes. Browser URL policy blocked
  file preview; visual/interaction QA explicitly unverified. Original concept unchanged.
- Initial report-rerender assertion failed because sorted JSON receipt key order
  differed from verifier output order. Attempt preserved in analysis-attempts/
  D1-viewer-qa-20260923/rerender-attempt-1.json. Fresh verifier-backed rendering now
  matches D1R/D1P/D1 reports; no scientific result or report was overwritten.
- New full direct archive includes viewer and QA artifacts in addition to all prior
  run/source evidence and Git history. Receipt under.local-artifacts/milestones.


## 2026-09-23 - F1 repeated-scene fitting preparation

Added fitting Session inheriting the unchanged K2 step, strict metadata, separate
evaluation RNG and raw saturation diagnostics. Added fixed F1 protocol/config,
immutable recovery/pilot/study runner with timing admission and process caps, and
guards for unchanged optimizer step, drift, evaluation RNG and timing rejection.
Tests and experiments pending; no scientific outcome presumed.

F1 preparation verification:20260923T112028Z_f4b86bdcf231 passes151 tests,
zero failures/errors/skips and smoke exit0. Source commit1c61b33. F1R
20260923T112316Z_f34a421f8302 passes11 exact comparisons across8 optimizer
executions and14 evaluations. All22 saved fields rescored. F1P
20260923T112500Z_8e1a30c320b8 passes8 updates/24 evaluations; all32 fields
rescored. Its timing-only estimate1546.47s admits the unchanged full256-update
study under1800s total/600s per member. Active run20260923T112646Z_dcd0601d0655.

During full execution, strengthened the report verifier to compare every initial
field and score exactly against the saved K2 original checkpoint control. This
report-only check does not alter training/config/source hashes or admission.
Existing F1R/F1P reports remain immutable; their original checks remain valid.

F1 completed20260923T112646Z_dcd0601d0655:256 updates/56 evaluations,860.27s,
zero failures/timeouts. Verified312 fields,256 checkpoint boundaries,56 metrics,
eight exact original-baseline fields and eight exact final checkpoint rollouts.
Preserved complete F1 reports and a post-hoc descriptive/source audit with its
script under.local-artifacts/analysis-attempts/F1-summary-<run_id>. No optimizer
change in that audit. Three final long-horizon connections, no joint-budget
success; fixed-source/region-source mismatch documented as next work, D037.
All256 updates clipped; parameter movement4.40%-5.53% is descriptive only.
No experiment or analysis attempt failed this milestone. No model promoted.


## 2026-09-23 - A2 access audit preparation

Added opt-in component_bottleneck_v2 and independent binary component BFS; neither
changes existing losses or serving. Added semantic/gradient/threshold fixtures,
a frozen277-field/12-gradient diagnostic and per-process caps. Preserve exact
source fields and parameter vectors, including zero gradients. Results pending.

A2 verification20260923T123338Z_6b51ce24b725 passed158 tests with zero failures,
errors or skips; smoke0. Implementation/protocol committedfac46ff before execution.
A2 20260923T123641Z_98bf30045a6f completed277 replays/12 gradient cases,397.78s,
zero optimizer updates, no failures/timeouts. All277 candidate scores/BFS results
recomputed;72 parameter vectors/72 last-raw derivatives/432 cosines checked.
All12 raw/projected model fields reproduce their saved source results exactly.
27 source hashes match snapshot. Added explicit original-checkpoint/config
crosscheck against F1's recorded metadata. No historical artifact overwritten.

Documented ACCESS_AUDIT_FINDINGS.md/D039 and ACCESS_TRAINING_PLAN.md. Candidate
restores access parameter gradients in four fitted16-step cases but not original
disconnected states. Eight score reductions and30 increases are rescoring only;
all277 binary labels remain equal. No training or candidate promotion followed.

## 2026-09-23 - F2 implementation prepared

Added opt-in access-only Session, strict F1 configuration matching, dual-definition
scoring, exact historical baseline parity gate, actual-loop recovery and fixed
pilot/study coordinator. Added verifier for all fields/checkpoints, common F1
rescoring and final forward replay. Added five isolation/RNG/metadata/cost tests.
F2 protocol/config frozen before execution; no baseline/production code changed.
Results and regression status will be recorded separately after execution.

F2 regression20260923T134911Z_c3c1cd97dbdb passed163 tests, zero failures/errors/
skips and original-checkpoint smoke0. No hashed code change after this pass.

Registered supplementary F2L recovery while F2 was running, before viewing final
outcomes: repeat update63->64 and evaluations for every member. Wrapper uses the
unchanged validated worker and exact checkpoint-tree comparison; syntax checked.
It adds no training exposure and does not alter32 frozen F2 source hashes.

F2 provenance crosscheck:28 shared source files exactly match F1;32 current
source hashes and original config/checkpoint match. One analysis archive mkdir
failed with WinError5 (again after permission renewal). Approved local operation
completed preservation; original report compared unchanged, no overwrite. Both
attempt/script and final receipt retained under analysis-attempts/F2-provenance-
20260923T135818Z_5520d5d80cec. Scientific training unaffected.

F2B12/24 exact parity; F2R8/14 with11 exact recovery checks; F2P8/24 admitted full
matrix on1334.37s estimate. Parent full run hit a timing deviation and overall
interruption at254/54. Linked continuation copied all308 records with equal hashes
and executed2 missing updates/final2 evaluations, producing complete256/56 matrix.
All312 fields rescored,256 checkpoints and56 metrics checked,56 F1 controls dual-
rescored,8 initial fields and8 final checkpoint rollouts exact;32 source hashes.
F2L4 actual final updates/8 evaluations reproduce full checkpoint trees/traces/
fields in38.96s. Post-hoc trajectories/analysis code archived. No model promoted.

Recorded all timing/permission incidents without discarding results. After F2L,
fixed worker timeout accounting: a success return with elapsed time beyond cap
now records elapsed_cap_exceeded and stops the coordinator. Added a regression
for delayed-success and normal completion. Historical source manifests intentionally
differ after this fix; exact archived source is required for old optimizer resume.
Added ACCESS_TRAINING_FINDINGS.md, GROWTH_STABILITY_PLAN.md and decisions D041/D042.

Post-fix regression20260923T143444Z_3c97b03e720e passed164 tests, zero failures/
errors/skips, original-checkpoint smoke0. Final crosscheck verifies all56 binary
labels agree, all8 matched final masses increase, all final binary illegal/blocked/
unsupported voxel counts0, and only the timeout runner differs from historical
32-file source. No further code changes after the passing regression.

## 2026-09-23 - H1 preparation

Added frozen-weight source validation, dual-definition scoring, sampled geometry
transitions, saved parameter/raw-gradient vectors, a fixed timing pilot/full
coordinator and all-field verifier/reporter. Added four tests for transition
semantics, zero/conflicting gradient statistics, pilot admission and delayed-success
timeout enforcement. Original objectives/model/training code unchanged. No H1 run
yet; regression and timing admission required before full execution.

H1 preparation regression20260923T145719Z_37715b17ba45 passed168 tests, zero
failures/errors/skips; original-checkpoint smoke0. Hashed diagnostic code/config
will remain frozen through pilot and full run. No scientific run started yet.

Chart preparation: matplotlib installed in isolated Codex cwd .plot-deps using
bundled Python, leaving repo .venv unchanged. Project Python lacks pip; that first
installer attempt exited before any install. Bundled installer succeeded. Chart
reads report JSON only and records rendering-library versions separately from
scientific environment. No diagnostic source change or training-dependency change.


## 2026-09-23 - H1 completed and findings preserved

Source3acdefd. Pilot20260923T145912Z_17b1d18f5b10 completed73.50s; timing-only
989.49s estimate admitted1500s. Full20260923T150119Z_f7b304516723 completed603.78s,
all worker/full caps met; zero optimizer updates, no failures. Verified31 source
hashes,180 fields,150 transitions,20 exact historical anchors,20 exact gradient
forward fields,120 parameter/120 raw vectors,720 cosines and10 unchanged models.
Eight new F2 gradients,12 reused A2 controls. All no-success results retained.

Added complete H1/H1P reports and raw-evidence references, post-hoc outcome JSON,
four-panel chart with source/image hashes, rendering receipt and visual QA.
Analysis/figure scripts preserved in local analysis-attempts. Added measured
GROWTH_AUDIT_FINDINGS.md, proposed HORIZON_TRAINING_PLAN.md and D044; updated
PLAN/RESUME. No scientific source change after168 passing tests. One documentation
patch used an incorrect DECISIONS context and failed; existing files inspected
before completing updates. No scientific artifact altered by that patch.

Finish with a local results commit and full verified archive; milestone receipt
records archive identity/hash, payload count and restore checks. Archive includes
all prior evidence and private reports while reports remain Git-ignored. No
Drive operation, paid compute, production/default change or remote push.

F3 regression20260923T155421Z_8e4d3bfc6919 passed172 tests, zero failures/errors/skips, original-checkpoint smoke0. Freeze scientific code/config after this pass before parity/recovery/pilot.

F3B20260923T155551Z_dc2ea609facf exact12 updates/24 evaluations in88.22s. F3R20260923T155836Z_10e9dbf37b62 exact11 recovery checks,8 updates/14 evaluations in100.23s. F3P20260923T160209Z_e292a696d594 verified96 unique fields and timing-only3246.25s estimate; admitted3600s cap. Full20260923T160713Z_cc33850561b8 started under sourcec0a44b1. All37 source hashes checked; objective/evaluation ASTs match F2 and31 H1 files unchanged.

While F3 full was running, before reviewing final outcomes, prepared supplementary F3L: replay updates63/64 from all four saved update62 checkpoints. Uses unchanged worker; wrapper identity recorded separately. Cap120s/worker360s total. Syntax checked; actual execution pending study completion. No source file covered by37 F3 hashes changed.

Full F3 completed1722.77s, all elapsed caps met. Reporter verified376 unique fields/260 checkpoint cursors/8 final rollouts. Automatic approval review for F3L launch failed to finish before deadline; tool reported6632.2s waiting, CreateProcess rejected before launch. This was not a training overrun or scientific failure. A normal local retry within granted permissions started F3L20260923T184328Z_418e31921193. System process metadata query was access-denied; no escalation pursued for that query. Post-hoc outcomes and all-intermediate recovery comparisons preserved separately.

F3L20260923T184328Z_418e31921193 completed118.04s: all8 trained-state updates and8 evaluations match full checkpoint/trace/field trees,8 cursors and38 source hashes checked. Saved F3-outcomes and critical-cell inspections with their exact scripts:72 mass reductions,59 connection losses,0 gains;51 negative/21 zero critical raw values, all72 access losses1. No new parameter-gradient experiment in that post-hoc inspection.

Chart attempts: initial normal execution denied reading matplotlib; after explicit read permission two normal attempts imported an incomplete namespace and lacked matplotlib.use. Folder metadata read also denied. No chart artifact had been created. Scoped elevated rendering succeeded using existing isolated libraries; all48 plotted points/hashes verified and chart visually inspected. Training environment unchanged. Added HORIZON_TRAINING_FINDINGS, ACCESS_RECOVERY_PLAN, D046 and final RESUME/PLAN updates. Preserve the negative result; no model promoted. Local results commit and full archive follow; completion recorded in milestone receipt.


A3 preparation 2026-09-23: added signed raw maximin loss, separate exact forward
trace, frozen231-field registry, six-term actual gradient audit, weighted
conflicts and reversible two-sided local probes. Added full artifact verifier,
protocol and regression coverage; historical access/rollout sources unchanged.
A preparation command used unavailable generic python and made no changes;
retried with the project interpreter. No experiment outcome yet.

A3 preparation regression20260923T192815Z_f7cdbff9cc00 passed180 tests with no failures/errors/skips and original-checkpoint smoke0 in84.62s. No scientific code changes after verification.

A3P20260923T193037Z_0e72e1207fb9 passed fixed25-field/2-gradient pilot and complete verifier;111.09s, timing-only1038.85s estimate admits1500s. Exact historical F3 source hashes37/37 preserved. Full audit proceeds without changing selection/caps/code.


A3 full20260923T193403Z_ac42a8cca390 completed336.74s,231 fields/eight gradient
cases/12 probes, zero optimizer updates. Verifier passed35 source hashes,48
parameter vectors/48 last-raw arrays, exact forward parity and all elapsed caps.
Restored6/8 gradients; two zero cases traced through earlier clamps/non-firing.
All six access-descent probes improve access loss; three improve total, three
worsen total. No binary connection repaired. All prior37 F3 sources unchanged.

Post-hoc summary first failed on a float32 cancellation tolerance; original
script/receipt retained in A3-summary-attempt-1. Receipt write from Codex cwd was
access-denied; retry from authorized repo succeeded. Final summary reports
scale-aware residual diagnostics; this is not a changed preregistered gate.
Added RAW_ACCESS_FINDINGS, RAW_ACCESS_TRAINING_PLAN and D048.

Post-audit extreme finite-value check reproduced nonfinite infeasible fallback
from raw.sum()*0. Replaced only that zero anchor with one finite scalar times0;
new regression, no actual audit case affected. Exact35-file diagnostic snapshot
retained; current raw_access.py differs by this fix. Follow-up full regression
20260923T194258Z_d2f3798c93b4 was launched; inspect result before claiming pass.

Follow-up regression20260923T194258Z_d2f3798c93b4 passed181 tests, no failures/errors/skips, smoke0,80.57s. Results commit and full verified local archive follow. No code changes after this pass.


F4 preparation2026-09-23: added explicit raw-objective training identity, fixed16
training path, triple-definition evaluation, baseline parity and early/late
recovery runners, timing admission and full saved-evidence verifier. Existing
F2/F3/A3 source unchanged. Added guards for objective-only change, source/cursor
drift, mismatched-objective recovery and timing completeness. No F4 outcome yet.

F4 preparation regression20260923T195529Z_9f4db82ab94f passed186 tests, no failures/errors/skips, original-checkpoint smoke0,83.35s. Source freeze follows; no scientific edits after passing suite.

F4B20260923T195731Z_c19895c6c745 passed exact12-update/24-evaluation F2 parity in142.85s; reporter verified41 source hashes and128 historical controls. F4R20260923T200110Z_961286a9e4b3 passed all11 actual-loop restart checks in91.25s; reporting before timing pilot.

First F4P20260923T200402Z_b794e30d69ba completed8 updates/24 boundary/72 grid records in328.52s. Timing estimate2409.4967s exceeds frozen2400 cap by9.50s, so no full run admitted. All evidence retained. Identified redundant recomputation of unchanged objective terms for raw scoring; an explicit efficiency revision and new gates will be required before reconsideration.

D050 efficiency revision after verified first pilot: reuse unchanged evaluation terms for raw totals; add exact recomputation regression. Reporter accepts a validated separate prefix to preserve both source versions. Training step, objectives and config unchanged. New gates required.

Efficiency revision regression20260923T201206Z_4b81a6dabfa4 passed187 tests, no failures/errors/skips, smoke0,86.46s. All original gates and first rejected admission retained. New source frozen after this pass.

Revised gates: F4B2 20260923T201413Z_628ce760d88c exact12/24 in134.59s; F4R2 20260923T201744Z_ef6c479ec7f8 all11 recovery checks in85.55s. F4P2 20260923T202008Z_206b3f6adb5f completed295.22s, timing2356.91s admits2400. Cross-version pilot checks all traces/fields/full checkpoint trees exact (8 updates/24 boundary/72 grid,96 unique fields), training step AST identical. Only evaluation reuse/report prefix source files differ. First rejected admission preserved.

Full F4 20260923T202649Z_75034cca563c completed1321.78s,256 updates and full evaluation matrix; all caps met. Scientific source unchanged373968d. Full saved-field/checkpoint verifier started before late recovery and final interpretation.


F4 full verification passed41 source hashes,376 unique fields,260 checkpoint
cursors,56 F2/72 H1 controls and eight final rollouts. F4L trained-state recovery
20260923T205434Z_c8136d2ce8e8 completed83.58s: eight updates/eight evaluations exact,
all caps met. Post-hoc paired analysis checks256 F2/F4 checkpoint pairs and five
early recovery checkpoint pairs; all31 historical H1 source hashes unchanged.
Result62/72 connected vs59/72, gains3/losses0; zero budget/joint success. Added
RAW_ACCESS_TRAINING_FINDINGS, PERSISTENT_GUIDE_PLAN and D051. No code/config change
after187-test regression. Two figure renders retained; v2 separates annotations,
visually inspected and48 aggregates/hash checks passed. Runtime uses existing
isolated plotting dependencies; project training environment unchanged.

Final local commit/full archive preparation follows. Archive success is recorded
in the milestone backup receipt, including all payload hashes, fresh Git restore,
6 reference/12 legacy scenes,18 annotations and41 exact F4 source snapshot hashes.
No Drive/off-device copy, paid training, push or deployment.


F5 preparation: added versioned persistent scaffold branch, owned scene-bound
static perception cache, conditioned rollout/training, F4 parity and early/late
recovery runners, saved-field/source verifier and regression tests. Original
backbone, historical source/defaults and nine families unchanged. Protocol D052
freezes caps and closure decision before F5 outcomes. No F5 run yet.

Initial F5 regression20260923T211811Z_35187d4e6ff3 passed197 tests/smoke0. Review added explicit rejection of legacy forward/grow calls that would omit conditioning, plus conditioned evaluation RNG test. Follow-up regression required before source freeze; first pass retained.

F5 final preparation regression20260923T212031Z_48dcfddbfd4f passed199 tests, no failures/errors/skips, original-checkpoint smoke0,105.27s. Scientific code frozen after this pass. No training outcome yet.


F5B20260923T212257Z_823a2b09b2dd passed exact12-update/24-evaluation F4 parity in141.65s. Reporter verified47 source hashes,36 fields,16 cursors and128 F4 controls. Source27416f3 unchanged. Conditioned recovery is next.


F5R20260923T212734Z_efe016a02413 passed all11 exact restart checks in93.50s; reporter verified47 hashes,22 saved fields,10 cursors and128 F4 controls. Proceeding to fixed timing pilot with unchanged source/caps.


F5P20260923T213256Z_42130eed3f5d completed351.21s. Timing-only estimate648.98/member2595.94/full admits unchanged900/member3600/full caps. All runtime limits met. Saved-evidence reporter underway before full launch.


F5P reporter verified47 source hashes/96 fields/12 cursors/128 F4 controls and nonzero guide gradients. Full F5 20260923T214035Z_566da8c507c0 launched with unchanged source27416f3 and frozen limits. Final outcome not yet available.

F5 full20260923T214035Z_566da8c507c0 completed1317.67s (21.96min),256 updates/56 boundary/72 final-grid records, all runtime caps met. Full saved-field/source/checkpoint verifier launched before trained-state recovery and closure.


Full F5 reporter verified47 source hashes/376 fields/260 cursors/128 controls/eight final rollouts. F5L20260923T220548Z_fdd27bb79c92 completed77.01s with exact8-update/8-evaluation trained-state replay across all four models, all caps met. Post-hoc analysis and closure follow; no active training.



## Local investigation closed after F5 - 2026-09-24

Full20260923T214035Z_566da8c507c0 completed1317.67s,256 updates/120 unique evaluations,
all caps met. Outcome66/72 connected,0/72 budget,0/72 joint.
Verifier checks47 source hashes/376 fields/260 cursors/128 F4 controls/eight final
rollouts. F5L20260923T220548Z_fdd27bb79c92 exact8-update/8-evaluation trained restart.199 tests
passed; frozen source27416f3 unchanged. Added PERSISTENT_GUIDE_FINDINGS,
LOCAL_PHASE_CLOSURE and D053. No further incremental local training queued.
Final local results commit/full archive follows; milestone receipt records
completion. All historical artifacts and private ignored reports preserved.


## 2026-09-24 - Studio S1 and planner/refiner specification

Added separate deploy/studio.py service, studio.html, dedicated CSS/JS and
run-studio.ps1. Warm-paper interface follows the user concept with real procedural
geometry, six scene presets, validated coordinate edits, three orthographic views,
visibility controls, nine-family diagnostics, saved studies and JSON export.
Canvas surface drawing has no remote dependencies and redraws only on changes.
Historical service, original models and all nca scientific source remain unchanged.

Appended design studies under .local-artifacts/studio with unique IDs, scene and
source hashes, full voxel geometry/settings/diagnostics and hashed receipts.
Submitted scene is persisted before computation; incomplete writes are surfaced,
not removed. No delete/overwrite/Drive path. Added12 integration tests covering
six presets, infeasibility, validation, determinism, persistence, corruption,
concurrency rejection and error recording. Wrote PLANNER_REFINER_SPEC.md and
STUDIO_S1.md; D054 records the scope and remaining product work.

Verification20260923T223300Z_af097d57765b:211 pass, smoke0,95.93s.
Final20260923T223757Z_6602b2cf16c8:211 pass, smoke0,96.61s after explicit
failure logging. Browser checks include valid/invalid edits, failed feasibility,
reopening across a server restart, export equality, layers/views and responsive
layout. Full details, retained development observations and artifact/source hashes
are in experiments/reports/S1-studio-verification.json.

One exact-match documentation patch failed without partial edits. Mobile heading
spacing corrected. A browser download-event timeout did not prevent export; the
downloaded JSON was verified equal to the local record. External-cwd sandbox
servers could read but not write the repo: confirmed WinError5. Project-cwd server
works. Added exception logging so failure to preserve a failure is visible in logs;
the denied attempts are retained in the QA report since their file writes failed.
No training, checkpoint promotion, paid compute, Drive action, push or publication.


## 2026-09-24 - Studio S2

Added durable background job requests/source ZIPs/hash-chained state histories,
single-owner queue, cancellable Windows worker trees, shutdown/crash recovery and
linked retry. Added actual side-by-side geometry, nine-metric and scene-revision
comparison; checksummed JSON export/import with supported-method recomputation.
UI names its procedural scaffold and absence of inhabitability explicitly.
Added STUDIO_S2, SPATIAL_BRIEF, D055, QA report and recovery notes.13 S2 regression
tests extend211 to224; all pass including actual forced owner/descendant death.

Two initial direct test attempts timed out in numerical imports with an early
watchdog. Four preserved probes locate and resolve that startup issue by moving
the watcher after runtime loading, while Windows process ownership covers imports.
Browser QA found that JSON parse/stringify changed0.0 to0 and broke checksums;
exact text export/upload fixes the actual round trip. Earlier broken export,
failed probes and observations are retained. Full backend verification preceded
that frontend-only fix; actual browser round trip verifies final frontend.
No scientific objective/model/reference scene/historical serving edits; no paid
training, Drive access, push or publication. Private report remains Git-ignored.


## 2026-09-24 - SP1 spatial platform prototype

Added nca/spatial.py with a deterministic level-deck constructor and independent
floor/headroom/footprint/landing evaluation. Added frozen SP1 recipe, append-only
six-case runner and12 spatial regressions. Existing objectives and models untouched.
Full236-test pass; SP1 expected outcomes all match. All raw candidates, including
collisions and unsupported levels, plus paired W1 fields and nine-family scores
are retained. Old access/coverage conflict with the positive clear-space example.

Added saved-study gallery linked from Studio: three actual geometry views,
material/surface/air distinction, six selectable examples, separate metric tables,
context/clear-volume controls and mobile layout. Initial camera cropped context
tops; corrected before final visual check. A checkbox operation immediately after
viewport reset failed once; fresh-state retry worked. Port inspection was denied
by sandbox; restarted the owned server session successfully. No failed scientific
or regression run. Updated specification, findings, decision and resume records.
No paid training, Drive access, public hosting or push; private reports stay ignored.


## 2026-09-24 - Corrected volumetric target and VA1 audit

Recorded user's rejection of the flat-platform interpretation and clarified that
volumes/spatial voids need no room/shelter program. Updated AGENTS, D057, PLAN,
RESUME and superseded-target notices while retaining all earlier results.
Added separate descriptive volumetric evaluator, nine fixed equal-context probes,
frozen recipe and append-only runner with effective config/checkpoint/source hashes.
Added12 tests; full248 pass. Audit13.08s, all analytic/parity checks pass. Exact
recomputation of all fields/masks/old scores and source verification pass.

Added volume-study gallery with paired selection, cutaway, vertical/horizontal
slices, context/void visibility and separate nine-family table. Updated Studio
navigation; SP1 remains available as historical diagnostic. First SP1 navigation
did not show latest notice; fresh query navigation confirmed new content/link.
Desktop/mobile visuals and controls verified. No failed scientific run, old loss
edit, model promotion, paid training, Drive operation, remote push or publication.


## 2026-09-24 - D058 building-mass interpretation

Recorded user's confirmation: generate overall building mass; leave interiors and
construction for later. Added MASSING_BRIEF, updated AGENTS/PLAN/RESUME, appended
D058 and marked the earlier volumetric next-phase plan superseded. Retained all
historical evidence. No source/model/data/viewer change or scientific run. Checked
documentation diff and private-report ignore rules; no regression rerun needed.
Two initial multi-file patches failed on unmatched final contexts; status checks
confirmed neither applied any edits. Corrected with a guarded document update.
Local commit/document archive only; no Drive operation, push or paid training.

## 2026-09-24 - MA1 building-mass completion comparison

Added separate massing_v1 semantics, fixed physical opportunity-region diagnostics,
source-only cavity filling and bounded vertical-gap completion. Source cells are
preserved; proposed/added/rejected cells and source violations remain explicit.
Added12 tests; all260 regressions pass. Two audit attempts retain all33 records
each. Initial expected3456-cell domain omitted36 old anchor allowances; corrected
recipe3492, linked retry passes83 checks with identical geometry and metrics.
All33 final operations, masks, mass reports and old nine-family scores replay
exactly.131 relevant Python source files match tests and audit. Old models, losses,
checkpoints, reference scenes and VA1/SP1 study JSON remain unchanged.

Added paired gallery:11 sources,3 operations, added-volume highlighting, full/
cutaway geometry and true sections. Browser checked33 selector combinations,
display toggles, desktop and390px mobile; green legend now correctly changes when
highlighting is off. Updated Studio links and historical interpretation notices.
Recorded D059, protocol, findings, verification and resumable next steps.

Browser screenshot write into repo was denied; workspace save and exact copy
preserved evidence. First verification-helper replay used an incorrect checkpoint
path after successful geometry checks; corrected to authoritative notebook path,
retaining failed helper. Immediate post-resize/navigation snapshots can show old
state; destination and final layout were verified after settling. One exploratory
file search used nonexistent paths; corrected by listing repository paths.
No paid training, Drive access, push, publication or historical evidence deletion.

## 2026-09-24 - MT1 pilot massing target contract

Added massing_targets_v1 binary evaluator and separate scene/control constructors.
Reinterpreted access/coverage/thickness/budget/spill explicitly for building mass
inside the same nine families. Historical losses/checkpoint remain unchanged.
Four new development contexts and12 controls each; all48 expected outcomes and
48 facade parity checks pass. Recorded432 sensitivity reports with original fields,
full context masks, complete parameter sets and source snapshots. No failed run or
post-result tuning. Fourteen new tests; full274 regressions and smoke pass.

Added paired compact-reference/candidate gallery with nine-family verdicts, old
scores, four contexts,12 controls and geometry display options. Verified all48
selectors, desktop/mobile layout and displays. Changed checkbox label to "Highlight
other cells" to include illegal/outside-domain cells as well as thin volume.
Initial browser snapshot showed the loading state; loaded state verified later.
Compact browser export reduces18,147,261 bytes to1,029,993 bytes while retaining
every displayed record exactly; full evidence remains in the immutable run.

Added protocol/findings/D060, source/integrity/bulk-mask verification and resume
instructions. No repeated full audit was used for verification; recorded checks
and geometric invariants were verified independently. No new trained generator,
old objective replacement, paid compute, Drive action, push or publication.


MT1 final review update: context feasibility now examines every component touching
the source interface, rather than rejecting on the first disconnected choice.
Added a focused edge-case regression;275 tests pass in20260924T102553Z_4ae3c1b346f2 (104.92s).
Linked audit20260924T102755Z_e89550a24d8d passes96 checks in93.23s. All48 base records and432
sensitivity reports are identical to initial audit20260924T101444Z_3d8c5504cb9e. Both revisions,
source snapshots and verification reports remain preserved. No numerical tuning.
Final report: MT1-final-verification.json. No training, Drive operation or push.


## 2026-09-24 - Review Revision 2 and governing research brief

Created private dated markdown/PDF findings addendum while retaining the original
review unchanged. Added tracked RESEARCH_BRIEF_R2.md and D061; prepended governing
status to PLAN/RESUME and an explicit current-scope notice to the historical
PLANNER_REFINER_SPEC. Preserve all earlier text and experiment history.

Reconciled building-mass semantics, closed material experiments, provisional MT1
checks, delivered Studio work and remaining product/scaling tasks. Recorded bounded
procedural/direct/conditional-NCA sequence and measurable success/stop requirements.
The addendum synthesizes local evidence, without new research/benchmark claims.
Documentation/PDF integrity and visual checks accompany the milestone archive;
latest scientific regression remains 275 passes. No model, code, losses, viewer,
old records or private original report edits. No Drive access, paid training or push.


## 2026-09-24 - MG1 procedural mass alternatives

Added cube_route_growth_v1, fixed45-request matrix/four challenges, immutable run
runner and12meaningful tests. Retained two regression failures due to new test
metadata copying/comparison; corrected type-aware test, final287pass. No science
change after benchmark; full fields, routes, selection traces and scores retained.
Study20260924T113023Z_24298393f2a5: 27/45 generated fields pass unchanged MT1. Independent
verification replayed45 generated outputs and rescored all49 fields. Added protocol,
findings,D062,resume/plan and static gallery linked from Studio. No trained output.

Gallery builder initially assumed inline CSS; failed before writing gallery files,
then used existing shared styles. An exploratory search used nonexistent static
Studio paths; corrected to deploy/studio.html. CIM process inspection was denied;
Get-Process and existing browser tab sufficed without restarting server. Initial
gallery inspection before data publication correctly showed unavailable-study state.
Final browser QA and archive receipts record completed checks. No paid compute,
Drive operation, remote push, historical-model/loss/report deletion or replacement.


MG1 gallery final QA:49 selectors/nine family rows, three views, keyboard slice,
cutaway/growth/context and390px mobile checked. Final warn/error logs empty.
Two full-page screenshot attempts failed; taller viewport screenshots succeeded.
Record the limitation rather than claiming full-page captures. Facade-aware
procedural comparison remains a candidate for fair R2-B controls; no retuning.


## 2026-09-24 - MO1 massing objective admission

Added separate CPU massing_residuals_v1: min/max bulk, interface strengths,
raw/bulk component-excess and fixed-support widest-path residuals within nine
families. No old science or thresholds changed. Ten focused tests and full297pass.
Audit20260924T115136Z_91b3707a89d3:97 fields,873 family agreements, zero bulk differences and
two derivative probes. No failed test/audit attempt and no optimizer update.
Verified142 Python source hashes against regression/audit snapshots and independently
verified facade-ratio gradient formula. Retained full source studies and all records.

Added protocol/findings,D063,concrete MD1 candidate plan and current resume/plan.
User's positive feedback on MG1 volumes is recorded; no viewer change or server
restart. A Windows rg query with literal wildcard paths failed; corrected reads
used exact filenames. Local archive only; no Drive, paid compute, push/publication.
