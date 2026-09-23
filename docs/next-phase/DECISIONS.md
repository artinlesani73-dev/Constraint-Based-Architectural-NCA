# Decision register

Append new decisions or explicit superseding entries; preserve earlier rationale.

## D001 - Keep the original evidence and isolate new implementation

Date: 2026-09-13. Status: accepted. Source: user request.

Work on `next-phase/foundations`. Preserve the original notebook, checkpoint, history and evaluation. Source snapshots include dirty working-tree code so a result can be reconstructed even before a commit. The private report remains ignored. Existing results are historical evidence, not validation of repaired code.

## D002 - Track small records in Git; archive full experiment payloads

Date: 2026-09-13. Status: accepted. Source: user requirement to retain all results and decisions.

Commit summaries under `experiments/records/`; keep full fields, checkpoints, images, logs and snapshots in `.local-artifacts/runs/` or a configured external root. Do not overwrite run metadata or final outcomes. Retry/resume attempts get new IDs linked to the earlier run. Include negative results. Source/control records are append-only by tool convention; file permissions and a remote backup are still needed against manual deletion or device loss.

## D003 - Google Drive plus local archive

Date: 2026-09-13. Status: accepted policy; remote setup pending. Source: user's explicit choice.

Suggested Drive root: `MyDrive/NCA-Next-Phase/`. Keep each complete run folder and verify hashes after copying. Test recovery before paid training. The local before-change snapshot is not an independent backup; cloud synchronization is not assumed to have succeeded merely because a path exists.

## D004 - Do not spend compute before the experiment is defined

Date: 2026-09-13. Status: accepted working rule; numeric allowance pending.

The user has paid Colab, but the pilot cap has not yet been confirmed. Prepare E0 and recovery checks locally. Ask the user to run/sign into Colab only when a concrete notebook and manifest are ready. There is no current paid training job.

## D005 - Independent metrics precede replacement training losses

Date: 2026-09-13. Status: accepted implementation order.

`binary_v1` evaluates explicit boolean arrays with named endpoint IDs and chosen neighborhoods. Empty material yields null material-normalized scores and an explicit nonempty flag. Eroded-core fraction is a voxel-scale proxy, not physical maximum thickness. These primitives are not yet a complete architectural validity gate and do not silently replace historical notebook scores.

## D006 - First serving fix does not silently change the learned rollout

Date: 2026-09-13. Status: accepted.

Fix checkpoint loading and missing optional facade metadata now. Keep current growth schedules, firing/noise, thresholding, legacy corridor operator and UI settings until E0 profiles are explicit. Bounds validation, per-job configuration, actual cancellation and the UI redesign remain planned. This permits attribution rather than combining unrelated behavior changes.

## D007 - The scene contract records Model C's semantics, not Step D's

Date: 2026-09-18. Status: accepted. Source: M1 step 1 implementation.

`scene_v1` in `nca/contract.py` declares axes, world units, half-open extents,
entrance identity and extent, the legal-material region, the protected street
void, the declared support region and strict binarisation. It adds no constraint
family and no objective.

Where the earlier Step D specification and the trained model disagree, the
contract follows the model. `street_levels` is 6 from the embedded checkpoint
configuration, not 2 from the external JSON. Ground protection is the
anchor-based street band Model C was masked with, not Step D's explicit 3 m
pedestrian and 12 m no-go strips. `ceiling_z` is validated but derives no region
and stays null, because a height ceiling is not among the nine existing families
and enabling it would require a new user decision.

`permitted` is asserted equal to the legacy legality field on all six reference
scenes rather than assumed equivalent. Where the contract is stricter than the
historical path it raises instead of accepting: an out-of-grid or building-buried
entrance, overlapping entrance blocks, a facade anchor with a missing `side`.

## D008 - The reference scene set is frozen by hash

Date: 2026-09-18. Status: accepted. Source: M1 step 1 implementation.

`experiments/scenes/reference_v1/` holds six scenes verified against
`manifest.json` on every load. A changed scene file, a canonical-hash mismatch,
a missing listed file or an unlisted present file each fail loudly. Comparisons
recorded against a reference scene assume its bytes never changed, so revising a
scene after results exist requires a new set version and a superseding decision,
not an overwrite. `scripts/build_reference_scenes.py --force` exists for the
initial authoring pass only.

## D009 - Rollout profiles are reconstructed from primary sources and named immutably

Date: 2026-09-18. Status: accepted. Source: M1 step 3 implementation.

`nca/rollout.py` carries one rollout implementation whose every behavioural axis
is declared in a `RolloutProfile`. The three historical profiles are
reconstructed from primary sources and each records its provenance string: the
notebook's `train_epoch` and `evaluate` methods, and the `/generate` handler.

Correctness is established by exact agreement, not by inspection. The serving
profile is bitwise identical to the legacy handler loop across six request
variants and the evaluation profile is bitwise identical to `model.grow(seed,
50)`. Without that, a later behaviour change could not be attributed to the
change rather than to the rewrite.

`profile.replace(...)` renames its result, so a variant can never be recorded
under a historical name. `rng_source` must match how the stream is drawn: the
historical profiles declare `global` and refuse an explicit generator, since
reproducing a historical stream means drawing from the source the original used.

The shared `config` dictionary is no longer mutated per rollout; the update-scale
override is scoped and restored in a `finally` block. This is narrower than the
legacy handler, which leaked on an exception, but it is not per-job isolation and
does not close that M4 item.

## D010 - A wrong recorded claim is corrected by a superseding note, not an edit

Date: 2026-09-18. Status: accepted. Source: M1 step 3 implementation.

`GEOMETRY_CONTRACT.md` claimed that a training-time z-taper was absent from
serving. Reading the notebook showed the z-taper keys are referenced nowhere in
either the notebook or the deployment: they are dead configuration, and nothing
was lost at deployment. The original wording stays in place under a superseding
note, the corrected account lives in `ROLLOUT_PROFILES.md`, and the correction is
recorded in the change log. Wrong claims are retracted visibly, on the same terms
as failed runs.

## D011 - Add an in-distribution scene set alongside the designed one

Date: 2026-09-18. Status: accepted. Source: user's explicit choice when asked.

`reference_v1` was authored for evaluability and is not drawn from the
distribution Model C was trained on. Asked whether E0 should run on designed
scenes only, the user chose to add a legacy-distribution set as well, so that a
poor result can be attributed to the model rather than to out-of-distribution
input.

`experiments/scenes/legacy_easy_v1/` holds twelve scenes from the historical
`easy` sampler at seeds 0-11, consumed in order with no cherry-picking; the
manifest records accepted seeds, rejected seeds with reasons, and the difficulty
parameters. The sampler in `nca/legacy_scenes.py` is a transcription of notebook
code, verified against the notebook generator executed as an oracle rather than
against a reading of it.

`nca.legacy_scenes.legacy_seed_state` is the seed builder for this set, because
the deployed generator cannot reproduce it: a deployment change writes ground
anchor zones for any access point below `street_levels`, where the notebook
required the type `'ground'`. That widens the legality field for precisely the
scenes the historical generator produced. The deployed generator stays untouched
and the divergence is measured.

## D012 - Relaxations are named and declared, never implicit

Date: 2026-09-18. Status: accepted. Source: in-distribution set implementation.

A historical scene can violate a `scene_v1` rule that was written for designed
scenes. Rather than weaken the rule, a scene may declare a named relaxation from
`nca.contract.RELAXATIONS`. A relaxation is covered by the scene hash, is refused
if unknown or repeated, must document what it does not relax, and never changes a
derived region: a relaxed scene is evaluated by exactly the same rules. Only
`facade_below_street_band` exists. The designed set declares none and a test
enforces that.

An empty relaxation list is omitted from the canonical form, listed in
`CANONICAL_OMIT_WHEN_EMPTY`, so declaring no relaxations hashes identically to a
scene authored before the field existed. This is what allowed an additive
optional field without revising `reference_v1`, whose six files remain
byte-identical to when they were frozen. A key may only be listed there when
empty genuinely means absent.

## D013 - One Drive folder; explicit approval for every operation

Date: 2026-09-23. Status: accepted and recorded. Source: explicit user request.

The user authorized creation of `Constraint-Based-Architectural-NCA` in My Drive. The connector returned success, folder ID `1fS34Yy0-oMzSxWaYJFiPTkGgrZstgc0H`, URL https://drive.google.com/drive/folders/1fS34Yy0-oMzSxWaYJFiPTkGgrZstgc0H . This supersedes D003's suggested `NCA-Next-Phase` destination, while retaining local plus Drive backup as the intended policy.

Only this exact folder and its actual descendants are in scope. Ask before every Drive action, including reads/listings/metadata, searches, downloads, verification, uploads, edits, moves, sharing and deletion. Describe exact targets and actions; only an explicitly approved bounded batch may combine operations. No whole-Drive discovery, outside-folder access, shortcut traversal outside the boundary, automatic backup or implicit approval for later verification. This applies to all access routes, including browser, scripts, sync and Colab. The full operational rule is in `AGENTS.md`.

Creation is the only remote change this session. No subsequent folder read was performed because reads now require approval too. No upload or backup round trip is complete. This is a durable behavioral boundary, not a narrowed OAuth permission grant.

## D014 - Keep the prepared milestone backup local

Date: 2026-09-23. Source: explicit user instruction, "no. save them locally".

The user declined uploading/verifying the M1 archive in Drive. Retain the ZIP and
checksum in the local outputs folder. No Drive operation is authorized and do
not repeat the declined upload automatically. This supersedes the pending-upload
next-action wording in earlier handoffs. It does not revoke the dedicated-folder
boundary or ask-before-every-operation rule for any future Drive work.

## D015 - Freeze the first E0 diagnostic before seeing its recorded results

Date: 2026-09-23. Status: implementation decision within the approved E0 plan.

`E0_v1` uses the original Model C checkpoint and unchanged legacy corridor
operator on all 6 reference and 12 legacy scenes. The three historical forward
profiles use seeds 0, 1 and 2; six ablations use predetermined seed 0 only. Total:
270 cases at 50 steps, training schedule position 60, material threshold 0.5,
6-neighbor connectivity. Record counts at thresholds 0.3 and 0.7 as sensitivity
indicators; connectivity is scored only at 0.5. No optimizer updates occur.

The ablations change serving seed scale to 0.15, remove serving noise, remove
serving masking, set training firing rate to 1, and compare serving state-blend
and delta-mask firing at 0.65. The latter changes module mode solely to select
the firing implementation; this model has no dropout or batch normalization.
Single-seed ablations are preliminary diagnostic comparisons, not robust effect
estimates. Historical training step sampling is held at its maximum 50 to match
the other profiles; this is a matched forward replay, not historical aggregate
reproduction. The notebook, historical scores, checkpoint and scene manifests
are unchanged.

All cases, continuous state fields, inputs/corridors, per-case metrics, timing,
failed/unscorable outcomes, configuration and source snapshots are retained
locally. Full learned-value comparisons against procedural/scaffold-only/direct
optimization baselines remain E2. Geometric connectivity is not walkability or
structural certification. No new constraint family is introduced.

## D016 - Use the recorded baseline to repair targets before scaling

Date: 2026-09-23. Basis: E0 `20260922T230120Z_76f3b4677e8f` and its explicit
post-run target audit. This is an engineering priority within the approved plan,
not a change to the project's constraint families or a decision to abandon NCA.

All 270 cases completed. On legacy scenes, training and serving connect 10/12
scenes per seed, evaluation 0/12, while the legal corridor target connects 12/12.
No main profile connects a non-control reference scene, although permitted-space
routes exist for all five. Both ground-only legal targets are disconnected.

First fix the bounded vertical-envelope operation under a new version; then
address legality/routing and height-band clipping as separately measured changes.
Next repair loss semantics and gradients, and make a full comparison with
procedural/scaffold-only/direct-optimization controls. Do not infer that more
voxels, more channels or more training can resolve a disconnected or forbidden
target. The target audit measures a spatial graph property only, not complete
architectural feasibility. Do not change production defaults based on the
single-seed ablations alone. Keep the generated evidence local per D014.

## D017 - Separate the bounded envelope repair from the legal-routing intervention

Date: 2026-09-23. Implementation within D016 and the user's instruction to proceed.

Preserve the original v31 corridor callable and all deployed defaults. Add
corridor_bounded_v1 changing only the vertical expansion; add corridor_legal_v1
as a separate procedural target using the existing permitted field and explicit
entrance IDs. The legal version uses exact six-neighbor paths, a deterministic
minimum spanning forest, bounded thickening, and explicit infeasibility. Remove
its legacy endpoint-height clamp so ground accesses can reach legal elevated
space. No new constraint family or trained weight is introduced. This is spatial
connectivity only; headroom, deck and mechanical semantics remain unresolved.

CORRIDOR_PROTOCOL.md freezes C1_v1: 54 target audits and 108 forward cases, all
18 development scenes, seed 0, 50 steps, original checkpoint, two historical
profiles. Legacy cases must reproduce saved E0 fields exactly. Single-seed
outcomes do not justify a production default or an architectural-quality claim.

## D018 - Resolve objective compatibility before retraining on legal scaffolds

Date: 2026-09-23. Evidence: completed C1_v1 and its labeled post-hoc volume audit.

The legal router connects all 17 feasible frozen scenes with zero forbidden
voxels. The original checkpoint still connects no reference scene under either
forward profile; serving drops from 10/12 to 9/12 on legacy scenes. Do not promote
new production defaults or claim a trained-model improvement from the target fix.

The legal targets' maximum zero-spill mass is 0.405%-2.538% of the notebook's
non-building denominator, below its actual trainer's 3% lower budget on all 18
scenes. The old spill and lower-volume penalties therefore cannot both be zero
with these targets. Do not conceal this by silently weakening the volume floor.
Distinguish the procedural connection scaffold from a material design envelope,
version objective definitions, and check gradients/batching before changing
architecture or paying for training. LOSS_REPAIR_PLAN.md records the next gate.
The audit is post-hoc, not an additional preregistered C1 arm. These remain spatial
proxies within the nine existing families, not verified architectural access or
structural safety. Preserve every failed attempt and keep archives local.

## D019 - Version loss mechanics and diagnose envelopes without selecting a trainer

Date: 2026-09-23. Authorized continuation of D018 and LOSS_REPAIR_PLAN.md.

Add geometry_losses_v1 with explicit coverage and material-envelope inputs,
per-scene reduction, zero-background thickness, six-neighbor single-source
material reach, geometric support, and unchanged 3%-12% mass limits using the
historical non-building denominator. Nine named families remain represented.
The access/coverage meanings are explicitly spatial material diagnostics; this
is not a resolution of architectural void/deck/headroom semantics. No production
trainer or new weights are introduced. Historical files remain intact.

Necessary compatibility checks reject impossible target/envelope/budget contexts
at strict batch reduction. Empty material is flagged separately because it can
be an initial state. Context checks do not prove every objective can be jointly
zero. A source is a single fixed legal entrance voxel, avoiding multi-origin
self-seeding. The scene adapter independently checks six-neighbor guide reach.

L1_v1 compares fixed radius-three and radius-six material envelopes with the C1
scaffold and full permitted-space controls; none is automatically selected for
training or widened to satisfy a budget. Preserve all outcomes, including zero
gradients, tied/finite-horizon surrogate limitations and historical failures.
See LOSS_PROTOCOL.md for formulas and the frozen matrix. Correctness tests and
short real-model derivatives are not evidence of trained architectural quality.

## D020 - Keep the training gate open after measured loss diagnostics

Date: 2026-09-23. Evidence: L1 20260923T003413Z_1da1202e4a7f and the separately
labeled post-hoc clamp replay 20260923T003906Z_da05c8eed3f6.

Shared loss mechanics pass 109 tests, and all 72 contexts/three model-gradient/
six historical checks complete. This is not enough to train responsibly: a fixed
six-voxel envelope still cannot accommodate the old mass floor on five feasible
scenes, and the ground-reference access loss has no parameter gradient. Exact
replay identifies two permitted, fired voxels with negative pre-clamp values;
the lower hard clamp kills the available final-step access derivative there.
A nonzero mixed-batch gradient would hide that failed scene.

Do not select an envelope, lower the volume floor, add a straight-through gradient,
or change the model by implication. Next compare explicit objective-region/budget
alternatives and separately version a local pre-clamp-guidance versus smooth-state
gradient intervention. Preserve hard legality and independent binary metrics.
Only afterward calibrate weights, integrate retained regularizers, test optimizer
recovery and request a bounded Colab allowance. Architectural access semantics
remain open. Full details and limitations are in LOSS_FINDINGS.md.

## D021 - Explicit L2 candidates and a conditional CPU recovery test

Date: 2026-09-23. User authorized the two local experiments followed by a tiny
interruption-and-resume test when the measured gates permit it.

L2_v1 keeps hard-projected forward dynamics as the control, tests legal pre-clamp
coverage guidance with the same forward output, and separately tests a smooth
material update. A changed coverage readout is not a claim that the original
projected access derivative is fixed. Compare site and envelope denominators
explicitly; unchanged 3%-12% fractions mean different absolute physical budgets.
No envelope is silently expanded, no fraction silently lowered, no new family.

INTERVENTION_PROTOCOL.md preregisters 108 budget cases, 54 gradient cases and
nine absent-scaffold controls. Its stated gate may select radius6/envelope and
hard_preclamp for a local recovery-mechanics test only. R1 compares separate
processes, full optimizer/scheduler/model and four RNG streams, plus repeated
continuation. It is not a calibrated nine-family research trainer or approval for
paid compute. Original artifacts and every attempt remain preserved locally.

## D022 - L2/R1 evidence supports mechanics, not a production objective

Date: 2026-09-23. Completed L2 20260923T075113Z_d95fabaf3776 and gated R1
20260923T075727Z_2233c0e51b9a. See INTERVENTION_FINDINGS.md and the full report.
All 21 hard-forward pairs match; pre-clamp coverage restores both measured dead
cell derivatives. Projected access still has the original zero gradient in that
case. Smooth-state derivatives improve but diffuse background mass rises sharply
without improved binary connectivity in this matrix. Keep smoothing experimental.

The radius6/envelope contract is necessary-valid for 17/18 scenes, excluding the
sealed reference; radius3/envelope also passes. Its material allowance is only
2.04%-10.65% of the original site's allowance. Select neither contract nor radius
as the research/production default based on this bound alone. Pre-clamp guidance
and radius6/envelope were used only for the preregistered CPU recovery diagnostic.

R1's four logical updates match exactly across fresh-process restart and repeat
restart, including full checkpoint trees, trace and outputs. Unit loss weights
are mechanics-only. The sample sequence covered two of the three available scenes.
No GPU/Colab or abrupt power-loss certification. Next inspect all-nine-term target
compatibility and architectural material amounts, then calibrate and preregister
E2 controls before a bounded training pilot. Preserve every attempt and archive.

## D023 - Preregister T1 geometry compatibility before calibration

2026-09-23: user authorized continuing with the target audit. TARGET_AUDIT_PROTOCOL.md
freezes432 static geometry cases,72 necessary joint bounds and36 direct occupancy
gradient cases across all18 frozen scenes. Keep nine families and existing formulas.
New facade/coverage/budget bounds expose incompatibility; passing is not sufficient.
All coefficients remain explicit unit values for diagnosis only. No model training,
production default, cloud operation or paid compute is authorized by this protocol.

## D024 - T1 exposes joint conflicts and weak architectural success criteria

2026-09-23. Run20260923T082527Z_845d2aa6aec0 completed432 target cases,
72 bounds and36 direct occupancy probes. All123 tests pass. The joint mandatory
facade bound reduces radius6/envelope necessary-compatible feasible scenes from
17/17 to15/17; radius3/envelope falls to11/17. Never call prior necessary bounds
sufficient. No loss-weight change can eliminate these exact zero-loss conflicts.

Two radius6/envelope reference guides score zero on all nine terms despite
one-voxel-wide segments. Current thickness discourages bulk, facade caps excess
contact, access follows material, and support is geometric. They do not certify
usable architecture. Independent saved-field verification reproduces all nine
zero-loss configurations across the complete matrix.

Keep the existing formulas and defaults unchanged for historical comparison.
NEXT_EXPERIMENT_PLAN.md proposes explicit semantic alternatives before coefficient
calibration and E2 procedural/direct/NCA controls. No new family is added. The
user was asked whether output should primarily be usable pavilion/bridge or abstract
material; do not silently choose while that answer is pending. An anchor exception
is a proposed experiment, not an implemented exemption or approved final contract.
All old scenes are development data now; future holdout scenes must be fresh.

## D025 - Feasible recommended scope after user uncertainty

2026-09-23. User did not choose a representation and asked which direction is more
feasible from the earlier plan. Recommend architectural material/form generation
first, evaluating usability separately and retaining usable pavilion/bridge as the
longer-term goal. Existing state/losses directly represent material. Reinterpreting
it immediately as usable circulation would require explicit floor/void/clearance
semantics and validation beyond current evidence. This is an assistant recommendation,
not a user-approved architectural specification or cancellation of usability goals.

Keep material connectivity for the proposed next controlled comparison; resolve
facade/budget contradictions and show learned value over procedural targets before
scaling. Do not claim structural safety. Update NEXT_EXPERIMENT_PLAN.md and handoff
with any later user steering; no production formulas were changed in this milestone.
