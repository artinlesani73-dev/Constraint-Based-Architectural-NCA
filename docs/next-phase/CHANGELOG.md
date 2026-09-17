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
