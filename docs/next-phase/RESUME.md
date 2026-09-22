# Resume the NCA next phase

## Latest local acceptance - 2026-09-23

Run `20260922T223533Z_9c5bfe017c4c`: 76 tests passed, no failures/errors/skips,
and out-of-directory Model C smoke exit 0. The run archive verified with no
problems; both frozen sets loaded intact (6 reference, 12 legacy scenes).
This supersedes older pending-verification and no-local-shell statements below.
The local milestone includes the rollout, legacy sampler, declared relaxations,
Drive approval rule, and this run summary; identify its hash from Git history.

Before E0, fix delta-mask profile overrides and RNG routing (the underlying
training step still reads the checkpoint fire rate and global RNG), reject
contradictory mode/firing combinations, and add notebook training-loop parity
coverage. See the 2026-09-23 acceptance entry in CHANGELOG.md. Passing the current
76 regressions does not cover arbitrary firing/RNG variants or prove a quality
gain. E0, corrected training and paid Colab remain pending.

Next user action: approve a specifically named backup upload and verification
batch inside the dedicated Drive folder after the local archive is prepared.
Do not use Drive without that approval. No Colab setup or manual test commands
are needed from the user at this stage. Then repair the E0 preconditions above
and run the recorded baseline. No paid training allowance has been agreed.


## Drive policy update - 2026-09-23

Google Drive is connected. The user authorized creating the dedicated folder
`Constraint-Based-Architectural-NCA`, ID `1fS34Yy0-oMzSxWaYJFiPTkGgrZstgc0H`:
https://drive.google.com/drive/folders/1fS34Yy0-oMzSxWaYJFiPTkGgrZstgc0H .
Only this folder and actual descendants are in scope. Ask and receive explicit
approval before EVERY Drive operation, including reads, lists, metadata checks,
verification and writes. See `AGENTS.md` and D013. Do not run Drive-wide searches
or automatic backup/sync. The creation response confirms success; no files have
been uploaded and no backup round trip has been verified. Earlier statements
below about an unconnected Drive or a proposed `NCA-Next-Phase` folder are
superseded by this update. Implementation and experiment status below is unchanged.


Last updated: 2026-09-18. Status: M0 verified except the Drive round trip; M1
step 1 committed and verified by run `20260917T222043Z_d6902ec4a5dc`; M1 step 3
(shared rollout with named historical profiles) implemented, its recorded
verification run pending, together with the in-distribution scene set the user
asked for. Next implementation task: M1 step 4, E0.

## Objective and authorization

The user approved planning and implementation of the next-phase
recommendations, with complete change/decision/experiment records and
recoverable progress. Research and product quality have equal priority. Keep the
existing constraint inventory. Paid Colab is available; the user selected Google
Drive plus a local archive. The initial compute allowance is pending.

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

## M1 step 3, implemented; verification run outstanding

Added 2026-09-18. See `ROLLOUT_PROFILES.md`, decisions D009 and D010, and the
change log entry of the same date.

- `nca/rollout.py`, version `rollout_v1`: one rollout implementation, every behavioural axis declared in a `RolloutProfile`, with `historical_training`, `historical_evaluation` and `historical_serving` constructors carrying provenance strings and notes.
- Agreement is bitwise, not by inspection: the serving profile matches the legacy `/generate` loop across six request variants, and the evaluation profile matches `model.grow(seed, 50)`. The legacy paths are untouched.
- The seed state and the shared `config` dictionary are no longer mutated; the update-scale override is scoped and restored in a `finally` block. Per-job isolation remains M4 work.
- `tests/test_rollout.py`, 15 cases.
- `nca/legacy_scenes.py` plus the frozen set `experiments/scenes/legacy_easy_v1/` (twelve scenes, seeds 0-11) and `scripts/build_legacy_scenes.py`, per decisions D011 and D012. The sampler is verified against the notebook generator executed as an oracle. Named contract relaxations were added; only `facade_below_street_band` exists and the designed set declares none.
- `tests/test_legacy_scenes.py`, 14 cases. Expected local suite total: 76.
- The legacy set must be seeded with `legacy_seed_state`. The deployed generator writes ground anchor zones for any access point below `street_levels`, where the notebook required the type `'ground'`, which widens the legality field for exactly the scenes the historical generator produced. See `SCENE_SETS.md`.
- Reading the historical notebook corrected one claim recorded on 2026-09-18 and added four findings. The corrections and findings are in the change log; the summary is that training adds no noise at all, the z-taper keys are dead everywhere, the recorded historical evaluation used no corridor scaffold, and Model C never saw a ground-type access point.

## M1 step 1, committed and verified

Added 2026-09-18 and accepted by run `20260917T222043Z_d6902ec4a5dc` (47 tests,
0 failures, 0 errors, 0 skipped, smoke exit 0; parent commit `b841991`, Python
3.12.14, torch 2.8.0+cpu). See `GEOMETRY_CONTRACT.md`, decisions D007 and D008,
and the change log entry of the same date. Committed.

- `nca/contract.py`: contract version `scene_v1`. Axes, ground plane, world units, half-open extents, entrance identity/extent, strict binarisation, canonical serialisation and scene hash; derived `permitted`, `protected`, `support_boundary` and `endpoints`; `verify_state_matches_scene`; `to_generator_params` for driving the historical generator.
- `experiments/scenes/reference_v1/`: six frozen scenes plus a hash manifest, including the `ref-05-sealed-partition` negative control. `scripts/build_reference_scenes.py` rebuilds them and refuses to replace a scene without `--force`.
- `tests/test_contract.py` (28 cases, NumPy only) and three real-checkpoint cases appended to `tests/test_runtime.py`. Expected local suite total: 47.
- The local suite total is 47. The frozen scene set was confirmed intact on the checkout, all six files reporting `unchanged`.

## Exact local commands

From this project root in PowerShell. `verify_foundation.py` records a new run
each time it is invoked and prints its `RUN_ID`.

```powershell
& .venv/Scripts/python.exe scripts/verify_foundation.py
& .venv/Scripts/python.exe -m unittest discover -s tests -p "test_contract.py" -v
& .venv/Scripts/python.exe scripts/build_reference_scenes.py
& .venv/Scripts/python.exe scripts/build_legacy_scenes.py
& .venv/Scripts/python.exe scripts/experiment.py verify 20260917T222043Z_d6902ec4a5dc
& .venv/Scripts/python.exe deploy/test_model.py
```

`build_reference_scenes.py` with no flags is safe: it reports `unchanged` for
every frozen scene and recomputes the manifest. If it reports anything else, a
frozen scene has drifted and the cause must be found before trusting any
comparison recorded against that scene.

The isolated environment is Python 3.12.14 with CPU PyTorch 2.8.0 and NumPy
2.5.2. `requirements-cpu.lock.txt` records exact tested packages, including the
HTTP test-client dependency notice in the raw log. It is a Windows CPU
environment record, not a Colab GPU installation recipe. To recreate on a
compatible Windows/Python setup, install the lock with PyPI plus the official
`https://download.pytorch.org/whl/cpu` index. Do not replace Colab's CUDA build
with the CPU lock.

## Next actions, in order

1. Run `scripts/verify_foundation.py` for M1 step 3, record the run ID in the change log, then commit the step 3 implementation with the run summary and this handoff. Do not stage `NCA-Studio-Concept.html` or the ignored report files.
2. Confirm the user-created Drive folder and preserve a verified off-device copy. The prepared milestone archive can be uploaded through Drive; no Drive connector is configured in this task. Do not mark this complete until the upload and a restore/hash check are verified.
3. Verify the frozen sets load on the checkout: `build_reference_scenes.py` and `build_legacy_scenes.py` should each report `unchanged` for every scene.
4. Run E0 on the frozen reference scenes with the real checkpoint and explicit seeds, isolating seed scale, firing, noise and masking. Save every per-scene result, continuous fields, binary thresholds, timing, full config and source snapshot. `fields_from_state` refuses a batch, so per-scene records cannot be averaged away.
5. Add a corrected bounded vertical-envelope operator with its own version and test it alongside the legacy implementation. The defect is in `compute_corridor_target_v31`: the envelope block assigns into `corridor_dilated[z]` while ascending `z`, so each row reads rows it has already modified and occupancy smears upward without bound instead of dilating by the declared envelope.
6. Prepare the short Colab preflight/recovery notebook after E0 and obtain the pilot compute cap before training. Test interrupted/resumed optimizer and pool state before a long run.

Do not run the historical fine-tuner yet: its loss semantics, tensor shapes and
gradients remain defective. Bounds validation, global request configuration,
actual cancellation and renderer work also remain unresolved. No performance,
architectural-quality, or corrected-training claim has been established.

## Open user inputs / pending external actions

- User selected Google Drive plus local archive. Suggested root: `MyDrive/NCA-Next-Phase/`; folder/upload verification pending.
- Pilot allowance question was sent: up to 6 GPU-hours, up to 2 GPU-hours, or decide later. No answer recorded; no paid job is authorized or running.
- No Git push or deployment has been performed. Branch and milestone commit are local.
- The M1 files were written from a cloud session that can read and write the project folder but has no shell on the device, so it could not run Git or the project environment. Local commands and commits are therefore the user's to run.
- The scene-distribution question is settled: the user chose to add the in-distribution set, recorded as D011 and implemented as `legacy_easy_v1`.
- Worth the user's attention before E0 is designed in detail: an unrecorded orientation run over the twelve legacy scenes shows the `historical-evaluation` profile producing about 28 material voxels per scene and connecting no entrances, against roughly 1076 and 806 voxels and 10-of-12 connectivity for the training and serving profiles. The corridor scaffold accounts for the difference. Since the evaluation profile produced `v31_evaluation.json`, that file describes near-empty geometry; its `avg_access_reach` measured void connectivity, which an empty design satisfies trivially, so it is not in contradiction. None of this is evidence until E0 records it.

## Session recovery

1. Read `AGENTS.md`, this file, `PLAN.md`, `DECISIONS.md`, `GEOMETRY_CONTRACT.md`, `ROLLOUT_PROFILES.md`, `SCENE_SETS.md`, and `CHANGELOG.md`. `ROLLOUT_PROFILES.md` supersedes the divergence list in `GEOMETRY_CONTRACT.md`.
2. Inspect Git status and log; preserve uncommitted changes and inspect any partial output before retrying a command.
3. Check experiment metadata under `experiments/` and artifact payloads under `.local-artifacts/`.
4. Verify the frozen scene set loads before using it: `load_reference_set()` raises on any drift.
5. Resume the first incomplete acceptance item in `PLAN.md`; update this file after each milestone.

This handoff is durable and does not depend on chat memory. It does not
automatically restart work or redeem credits when a usage limit resets. Resume by
asking the assistant to continue from this file.
