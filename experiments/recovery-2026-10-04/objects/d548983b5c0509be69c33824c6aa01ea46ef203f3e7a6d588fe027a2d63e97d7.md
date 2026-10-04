# G1 generation pilot ready for approval

The generation training session and Colab package are prepared. This milestone does not demonstrate trained generation quality.

## Verified locally

Run `20261003T162226Z_0089b332210b` completed three retained updates plus two recovery replays in 43.594 controlled seconds. Update 2 uses a teacher stage; update 3 starts from a single seed. Both replays match the full checkpoint payload and output state exactly. This covers model, Adam state, sampler cursor, trace and random states. Final inference uses only scene context and a single seed.

The completed checkpoint loads independently with the matching environment and rejects a changed semantic identity. An initial independent read omitted CUBLAS_WORKSPACE_CONFIG and was correctly refused; restoring that recorded environment resolved the mismatch without changing data or code. All 22 export payload hashes and 51 package payload hashes verify. Supervisor cleanup reports zero active workers. Notebook cells compile. The Colab notebook itself has not been executed in a browser or on GPU.

## Proposed single job

- Tesla T4, one seed, fresh weights, 256 updates of 64 growth steps.
- Exactly 128 single-seed starts and 128 teacher-stage starts. Two replay updates are extra computation and included in the time cap.
- Maximum 600 controlled wall seconds including worker startup, recovery and result writes. Setup, export, download and idle time are additional.
- Exact runtime guard and in-job GPU recovery checks before continuing beyond update 3. Stop on failure; no automatic retry or extra seed.
- Original network dimensions and CGR1 loss retained. New task, data and start schedule are explicitly versioned. Hard births remain detached.
- Full ZIP plus receipt, including interrupted/failed runs. No Drive operation or automatic model promotion.

The frozen review protocol is in PROTOCOL.md. It uses final checkpoint 256 only, nine development requests, single-seed inputs and a fixed 64-step review. A fixed 128-step stability check is additional evidence, not a best-horizon search. Reserved labels remain absent from this package. A successful development pilot is not final generalization or deployment acceptance.

## User action after approval

1. Open NCA-G1-Generation.ipynb in Colab and select a T4 GPU.
2. Run the upload cell and choose NCA-G1-Generation-Package.zip. Filename suffixes are accepted; exact package bytes are checked.
3. After approving the specific job above, set APPROVED_G1_JOB=True and run the training cell once.
4. Run the download cell and return both the full result ZIP and receipt, even after a failure.

No further local rehearsal is required unless code or settings change. If Colab reports a different runtime, return the stopped-run evidence; do not bypass the guard.

## Resume and preservation

Everything remains local under this folder. Repository synchronization remains pending; no repository edits, commits, Drive operations, paid compute, pushes or deployment occurred. Keep the preceding G1 preparation and repair-review evidence. The verified milestone archive is a same-disk copy, not an off-device backup. Checkpoints resume completed updates only; no cross-runtime or mid-rollout recovery claim is made.
