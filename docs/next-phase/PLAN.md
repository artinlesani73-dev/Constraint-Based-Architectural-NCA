# Next-phase implementation plan

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
