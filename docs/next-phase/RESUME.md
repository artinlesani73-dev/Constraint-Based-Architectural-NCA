# Resume the NCA next phase

Last updated 2026-09-23. **L1 loss mechanics and gradient diagnostics are complete;
training remains gated.** Read LOSS_FINDINGS.md and D019/D020. Next resolve the
material-region/budget contract and test a targeted gradient intervention on the
measured ground-entry failure. No optimizer or paid training has started.

## Authorization and storage

The user approved implementation, preservation of all decisions/results, nine
existing constraint families, research and product improvement, and local
archives. The user declined Drive upload (D014). No Drive access is authorized;
AGENTS.md requires approval BEFORE EVERY operation, even reads, and scope is
only folder 1fS34Yy0-oMzSxWaYJFiPTkGgrZstgc0H and actual descendants.

Paid Colab is available, but a GPU-hour cap and a job have not been approved.
Private NCA-Next-Phase-Report files remain ignored. Preserve the user's untracked
NCA-Studio-Concept.html and NCA-M1-Backup-2026-09-23-1dfafa7.zip.sha256.
No push, deployment or cloud access occurred in the corridor milestone.

## Checkout and durable evidence

- Repo: C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA.
- Branch: next-phase/foundations. Historical baseline: ac913b9.
- Milestones: b841991 foundations, ddc8100 contract, 1dfafa7 historical profiles,
  474bf53 rollout_v2/E0 runner, 75402ff E0 evidence, 579031c corridor operators,
  595c8e0 corridor retry, 8501478 C1 evidence, fa66238 loss package/L1 protocol.
  The later evidence commit adds L1 reports and this handoff; inspect Git log.
- All raw runs/source snapshots: .local-artifacts/runs/<run_id>/;
  small tracked summaries: experiments/records/<run_id>.json.
- Original 50-file snapshot: .local-artifacts/source-snapshots/.
- Local archive outputs: C:/Users/artin/Documents/Codex/2026-09-06/cre/outputs/.
  Keep the M1 (1dfafa7), E0 (75402ff) and corridor (8501478) archives. New archive:
  NCA-Loss-Backup-2026-09-23-<evidence-commit>.zip, with a checksum and a receipt
  .local-artifacts/milestones/<commit>-backup-receipt.json. Verify that receipt
  before claiming backup completion. These are local, same-disk copies.
- Archive builder for this milestone:
  C:/Users/artin/Documents/Codex/2026-09-06/cre/work/package_nca_losses.py.
  It requires completed L1, creates a fresh Git bundle/restore directory, clones
  the committed branch, checks frozen scene hashes and every ZIP payload hash.
  Never rerun over an existing archive or delete a prior one to clear the path.

## Verified implementation and experiments

- scene_v1: 6 reference + 12 legacy scenes, raw/canonical hashes frozen.
  Legacy scenes MUST use nca.legacy_scenes.legacy_seed_state; the deployed
  generator handles below-street facade anchors differently.
- rollout_v2 carries explicit firing/RNG controls and notebook-forward parity.
  E0 20260922T230120Z_76f3b4677e8f completed 270 cases. Read E0_FINDINGS.md.
- corridor_bounded_v1 changes only the cascading depth expansion; original
  centroid/distance/height-clamp/building-mask logic is retained.
- corridor_legal_v1 separately routes explicit IDs through permitted space using
  exact six-neighbor BFS and a deterministic shortest-path spanning forest.
  It retains bounded thickening, rejects ambiguous endpoint pieces, records
  infeasible partial forests and removes the endpoint-based height clip.
- The original corridor callable, notebook/checkpoint and production defaults
  remain unchanged. No model architecture or learned weight changed.
- Regression 20260923T000140Z_a7ff7de54684: 94 passed, no failures/errors/skips,
  checkpoint smoke exit 0. Tests cover real scene batches and geometric defects.
- Failed C1 20260923T000715Z_1e06516e9763 retained: 54 target records and 7
  registered rollouts before Windows long evidence filenames stopped publication.
  Its attempted eighth case could not publish a record. Short stable filenames
  fix the runner; full descriptions remain inside JSON.
- Linked retry **20260923T000945Z_3cbdc3603a12**: 54 target records, 108/108 forward
  cases, no failures/unscorable cases, 343.54 seconds CPU. Source commit 595c8e0.
  36/36 legacy final fields match E0 bitwise; radius-zero parity passes all 18
  scenes. Registered artifacts verify and fresh report/audit renders match.
- One failed post-hoc notebook parser attempt is preserved with its source/error
  under .local-artifacts/analysis-attempts/20260923-c1-volume-parser-01. It counted
  a demonstration constructor as well as the trainer; corrected audit selects
  the actual trainer. This did not modify/re-run the completed experiment.

## What the results mean

Legal targets connect 12/12 legacy + 5/6 reference scenes, with zero forbidden
voxels. The sixth reference is intentionally impossible and flagged accordingly.
Bounded expansion alone retains the two ground-target disconnections.

The old checkpoint still connects 0/6 reference scenes with either profile and
any target. On legacy scenes, training-profile connectivity stays 10/12; serving
falls from 10/12 to 9/12 with legal targets (seed 0). Do not promote defaults or
claim a trained-model improvement. The legal scaffold is a useful procedural
control; spatial connectivity is not walkability or mechanical safety.

The labeled post-hoc volume audit shows all legal targets hold at most
0.405%-2.538% of the historical non-building denominator, below its 3% lower
budget. Zero spill and zero lower-volume penalty cannot coexist with those
narrow targets. Distinguish connection guidance from a material design envelope
before training; do not silently weaken the budget. The underlying penalties
are soft, so this is not proof that no architectural design is feasible.

## Latest loss evidence and interpretation

- geometry_losses_v1: nine per-scene continuous terms, separate coverage/envelope
  masks, original 3%-12% mass limits/non-building denominator, explicit invalid
  contexts and nonempty flags, strict reduction. Synthetic gradient and batch
  checks pass. Do not reuse historical weights without calibration.
- Regression 20260923T002822Z_8f68d5bbcaf7 passed 108 checks; final
  20260923T003247Z_0ce597b29bc4 passed 109, zero failures/errors/skips and smoke 0.
- L1 20260923T003413Z_1da1202e4a7f (source fa66238): 72 context checks, 3 model
  gradient cases, 6 expected historical fine-tuner defects; completed in 29.25s.
  Source/config/input states/full parameter and occupancy gradients are archived.
  All artifact hashes, independent gradient norm/cosine recomputation and fresh
  report/details render checks pass. Reports are in docs/next-phase/reports/.
- Radius-three envelope: 3/18 valid contexts; radius-six: 12/18. Five feasible
  scenes fail the radius-six capacity bound. All permitted space gives 17/18,
  but removes meaningful spill restriction. No envelope is selected for training.
- Legacy-alone and mixed gradient examples include a capacity-invalid scene;
  individual terms were inspected diagnostically, never optimized. A nonzero
  mixed-batch access gradient hides a zero gradient on the ground case.
- Post-hoc 20260923T003906Z_da05c8eed3f6 exactly replays the ground case. The
  two voxels carrying its access derivative have pre-clamp values -0.001280 and
  -0.009346, are permitted and fired, then clamp to zero. The access parameter
  gradient is zero. This is a local four-step attribution, not a general proof.
- Ground/legality zero model gradients are expected after hard projection.
  Finite-hop max/min access/support remain nonsmooth spatial proxies. Architectural
  access semantics and usefulness on other horizons/seeds remain unresolved.

## Exact next actions

1. Define explicit candidate material-region and mass-budget contracts, recording
   the intended denominator and lower bound. Compare their necessary feasibility
   on the frozen scenes. Do not silently lower 3% or widen regions to force a pass.
2. Test a separately versioned local gradient intervention on the recorded dead
   ground case: legal pre-clamp guidance versus a smooth material-state candidate.
   Keep hard legality and historical code intact. Verify useful derivatives at
   the failed cells and weights, finite differences, binary outcomes, and matched
   controls. Do not slip in a straight-through gradient without its own explicit
   justification or treat it as the exact derivative.
3. Check longer horizons, other seeds and damaged/zero-route states, then integrate
   retained regularizers and calibrate objective magnitudes. No adaptive weighting
   should conceal incompatible definitions.
4. After these gates, implement a tiny local optimizer/interruption-recovery test
   saving model/optimizer/scheduler/RNG/scenes/update count. Then complete E2
   procedural/scaffold/direct-optimization controls and agree a paid Colab cap.
5. Product work and larger grids remain planned; no user setup is needed now.

## Commands and interruption recovery

From the repo root in PowerShell:

```powershell
& .venv/Scripts/python.exe scripts/verify_foundation.py
& .venv/Scripts/python.exe scripts/experiment.py verify 20260923T003413Z_1da1202e4a7f
```

Completed L1/C1 do not need rerunning. If new changes justify a loss-diagnostic repeat:

```powershell
& .venv/Scripts/python.exe scripts/run_loss_diagnostics.py --parent-run 20260923T003413Z_1da1202e4a7f
```

This creates a fresh complete attempt; it does not skip/resume cases from an old
run. Inspect result.json and registered artifacts before retrying an interrupted
command. Missing finalization means incomplete, not success. Report writers use
exclusive creation; do not overwrite prior reports to rerender them. The render
and analyze functions can compare outputs in memory.

Python 3.12.14, CPU torch 2.8.0, NumPy 2.5.2; requirements-cpu.lock.txt. Do not
install the CPU lock over Colab CUDA. Local write/Git permissions may need renewal
in a new task. Historical handoffs remain in Git/milestone archives. This handoff
does not automatically resume work or redeem credits after a usage reset.
