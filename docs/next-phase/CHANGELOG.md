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
