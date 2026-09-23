# Resume the NCA next phase

Last updated 2026-09-23. **C1 corridor corrections and comparison are complete.**
Read CORRIDOR_FINDINGS.md, D017/D018, then LOSS_REPAIR_PLAN.md. The next work is
compatible shared losses and batch/gradient checks. No paid training has started.

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
  595c8e0 short artifact paths/retry preparation. The later evidence commit adds
  C1 reports and this handoff; inspect Git log for its actual hash.
- All raw runs/source snapshots: .local-artifacts/runs/<run_id>/;
  small tracked summaries: experiments/records/<run_id>.json.
- Original 50-file snapshot: .local-artifacts/source-snapshots/.
- Local archive outputs: C:/Users/artin/Documents/Codex/2026-09-06/cre/outputs/.
  Keep the M1 archive (1dfafa7) and E0 archive (75402ff). The new archive is named
  NCA-Corridor-Backup-2026-09-23-<evidence-commit>.zip, with a checksum and a receipt
  .local-artifacts/milestones/<commit>-backup-receipt.json. Verify that receipt
  before claiming backup completion. These are local, same-disk copies.
- Archive builder for this milestone:
  C:/Users/artin/Documents/Codex/2026-09-06/cre/work/package_nca_corridor.py.
  It requires completed C1, creates a fresh Git bundle/restore directory, clones
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

## Exact next actions

1. Read LOSS_REPAIR_PLAN.md. Extract a versioned shared loss package with
   per-scene normalization, valid B=1/B>1 shapes, corrected zero-background
   surrogates, explicit infeasible/empty cases and binary evaluation kept separate.
   Freeze coverage/envelope/mass-budget semantics before optimizer experiments.
2. Verify intended gradients, finite differences away from nonsmooth ties,
   model-gradient contribution, and objective conflicts. Do not demand nonzero
   gradients from redundant post-projection penalties.
3. Prepare a tiny local optimizer/recovery check only after those definitions;
   preserve full model/optimizer/scheduler/RNG/scenes/update-count state and test
   interrupted versus uninterrupted continuation. Keep architecture fixed first.
4. Complete E2 scaffold/procedural/direct-optimization comparisons, then agree
   paid Colab cap and an explicitly approved artifact/backup procedure.
5. Product work uses stable scene/result contracts; shared-model concurrency,
   job cancellation and interface redesign remain M4. Larger grids follow
   measured correctness/memory/latency gates. No user setup is needed now.

## Commands and interruption recovery

From the repo root in PowerShell:

```powershell
& .venv/Scripts/python.exe scripts/verify_foundation.py
& .venv/Scripts/python.exe scripts/experiment.py verify 20260923T000945Z_3cbdc3603a12
```

Completed C1 does not need rerunning. If new changes justify a matched repeat:

```powershell
& .venv/Scripts/python.exe scripts/run_corridor_comparison.py --parent-run 20260923T000945Z_3cbdc3603a12
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
