# Resume the NCA next phase

Last updated: 2026-09-13. Status: local foundation verified; Drive backup setup pending. Next implementation task: M1 geometry contract and historical rollout reconciliation (E0).

## Objective and authorization

The user approved planning and implementation of the next-phase recommendations, with complete change/decision/experiment records and recoverable progress. Research and product quality have equal priority. Keep the existing constraint inventory. Paid Colab is available; the user selected Google Drive plus a local archive. The initial compute allowance is pending.

## Current checkout

- Project: `C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA`
- Implementation branch: `next-phase/foundations`
- Historical baseline: `ac913b9abff66f81ec6bf7182d02125cc46cacd3`
- Before implementation the only untracked files were the two next-phase report formats and `NCA-Studio-Concept.html`; preserve them.
- Filesystem permissions may need renewal in a new task. Git metadata writes may require a tool approval because the task originally lives outside this project.

## Completed and verified

- Original 50-file snapshot verified at `.local-artifacts/source-snapshots/before-next-phase-20260913T192856Z/`.
- Report files ignored and still present. Historical notebook/checkpoint/results unchanged.
- Run archive, source snapshot, hash verification, linked retries and interrupted-transfer recovery implemented.
- Independent binary geometry primitives implemented; they are not a complete nine-family architectural validator.
- Real Model C checkpoint loaded with `weights_only=True` and authoritative embedded config. Serving and smoke test share the loader.
- Optional null facade metadata no longer crashes the preview API.
- Verification run: `20260913T194223Z_ad4ee8ee60f3`; 16 tests passed, zero skipped, zero failures/errors, out-of-directory smoke exit code 0. This is a regression run, not a quality or latency benchmark.
- Run summary: `experiments/records/20260913T194223Z_ad4ee8ee60f3.json`; source snapshot and raw test output in the matching `.local-artifacts/runs/` folder.
- Milestone commit title: `Establish reproducible NCA next-phase foundations`. Inspect Git log for its actual hash; do not assume the verification run's parent hash is the finished implementation commit.

## Exact local commands

From this project root in PowerShell:

```powershell
& .venv/Scripts/python.exe scripts/verify_foundation.py
& .venv/Scripts/python.exe scripts/experiment.py verify 20260913T194223Z_ad4ee8ee60f3
& .venv/Scripts/python.exe deploy/test_model.py
```

The isolated environment is Python 3.12.14 with CPU PyTorch 2.8.0 and NumPy 2.5.2. `requirements-cpu.lock.txt` records exact tested packages, including the HTTP test-client dependency notice in the raw log. It is a Windows CPU environment record, not a Colab GPU installation recipe. To recreate on a compatible Windows/Python setup, install the lock with PyPI plus the official `https://download.pytorch.org/whl/cpu` index. Do not replace Colab's CUDA build with the CPU lock.

## Next actions, in order

1. Confirm the user-created Drive folder and preserve a verified off-device copy. The prepared milestone archive can be uploaded through Drive; no Drive connector is configured in this task. Do not mark this complete until the upload and a restore/hash check are verified.
2. Write scene/geometry contract v1 and small frozen reference scenes: axes, units, entrance IDs, legal material, protected void, envelope, support, threshold convention. Preserve existing families.
3. Extract named historical rollout profiles for E0. First reproduce seeding, firing, noise, masking and checkpoint config precisely; save continuous and binary outputs for matched scenes/seeds.
4. Add a corrected bounded vertical-envelope operator with its own version and test it alongside the legacy implementation. Do not silently replace the original notebook or its results.
5. Prepare the short Colab preflight/recovery notebook after E0 and obtain the pilot compute cap before training. Test interrupted/resumed optimizer and pool state before a long run.

Do not run the historical fine-tuner yet: its loss semantics, tensor shapes and gradients remain defective. Bounds validation, global request configuration, actual cancellation and renderer work also remain unresolved. No performance, architectural-quality, or corrected-training claim has been established.

## Open user inputs / pending external actions

- User selected Google Drive plus local archive. Suggested root: `MyDrive/NCA-Next-Phase/`; folder/upload verification pending.
- Pilot allowance question was sent: up to 6 GPU-hours, up to 2 GPU-hours, or decide later. No answer recorded; no paid job is authorized or running.
- No Git push or deployment has been performed. Branch and milestone commit are local.

## Session recovery

1. Read `AGENTS.md`, this file, `PLAN.md`, `DECISIONS.md`, and `CHANGELOG.md`.
2. Inspect Git status and log; preserve uncommitted changes and inspect any partial output before retrying a command.
3. Check experiment metadata under `experiments/` and artifact payloads under `.local-artifacts/`.
4. Resume the first incomplete acceptance item in `PLAN.md`; update this file after each milestone.

This handoff is durable and does not depend on chat memory. It does not automatically restart work or redeem credits when a usage limit resets. Resume by asking the assistant to continue from this file.
