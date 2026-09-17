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
3. Extract a shared rollout with named historical-training, historical-evaluation, and historical-serving profiles. Preserve the legacy functions for exact comparisons.
4. Run E0 on identical frozen scenes/checkpoint/seeds: isolate initial seed scale, firing, noise and masking. Save every per-scene result, continuous fields, binary thresholds, timing, full config and source snapshot.
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
