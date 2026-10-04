# G7 — ready for one approved Colab run

Use only the files in this **v2** folder. The original preparation folder is
historical evidence and contains a package that failed before training.

## Files to use

- Notebook: NCA-G7-Vertical.ipynb
- Upload when prompted: NCA-G7-Vertical-Package.zip
- ZIP size: 190,665 bytes
- ZIP SHA256: 030757d49e4557c1e7643e6c68a16611eb8950e0cf508fea22e2ae72833f0ddd
- Manifest SHA256: ca2e0bee4e647a1ad1ada8dddb742f05501cfb5b0a76ef31199112e13de65429

## What this run tests

Broaden training geometry while keeping the G6 model, losses, pacing, grid and
nine constraint families fixed. Retain27 original examples byte-for-byte and
add18 new examples with upward/downward connection offsets, paired elevations
and unequal building heights. All18 new teachers pass the full geometry contract
and budget checks. All45 rows load in both single-seed and teacher-stage modes.

Six new training scenes have west/east connection origins at grid heights
(8,12),(12,8),(10,16),(16,10),(14,14),(16,16). Four fresh reserved scenes are
frozen locally before training, at (9,15),(15,9),(12,18),(18,12), with different
horizontal positions and building heights. No reserved teacher or model output
has been generated. Each scene has16%,24%,32% volume requests. These are related
synthetic variations, not an independent architectural dataset.

Training remains256 updates. Under the frozen shuffle, original examples get
155 updates and new examples101; each row is visited5 or6 times. Thus per-example
exposure changes relative to G6. This is a fixed-compute data-distribution study,
not a perfectly isolated causal estimate or an equal-epochs comparison.

## Local verification

Three retained CPU updates and two exact recovery replays completed in
31.67 controlled seconds. All23 exported payload hashes,
192 transition accounts/ceilings and both complete checkpoint/state recoveries
passed. All initial model parameters match G6 and the neural model/loss source
is byte-identical. No model-quality evaluation was performed. Local CPU recovery
does not certify CUDA; the GPU run retains its device and recovery checks.

The first local package failed at0 updates because the inherited loader expects
the scene seed under `damaged`. Version2 adds that compatibility alias with the
same single-seed values. All geometry arrays and labels remain unchanged. The
failed package, logs and receipt remain in the original folder. No paid compute
was used for either local rehearsal.

## Paid allowance — approval pending

One fresh Tesla T4 run, seed1201,256 updates of64 steps, at most600 controlled
seconds. Includes admission probes, exact recovery checks and per-update evidence.
Setup, upload, export, download and idle time are additional billed runtime.
Expected stack: Python3.13.15, Torch2.11.0+cu130, NumPy2.1.3, CUDA13.0,
cuDNN92700. Stop on mismatch or failed checks; do not bypass guards or retry.

1. After approving the allowance, upload the notebook to Colab and select T4.
2. Run the upload cell and select the package ZIP from this v2 folder.
3. Change APPROVED_G7_JOB=False to True only after this exact run is approved.
4. Run the training cell once. Then download the full evidence ZIP and receipt,
   including after a failure, and share both local files for review.
5. Disconnect the runtime after downloads finish to avoid idle compute use.

No Drive mounting, uploading or other Drive operation is included.

## Frozen review after returned evidence

Verify every artifact and completed update; review only final256. Evaluate the
21 legacy regression requests separately from12 fresh reserved requests, each
at64 and128 steps with firing2101 and fixed threshold/quota. Require every case
to pass all9 families at both horizons, median fraction error<=.02 and maximum
<=.04 in each cohort/horizon, and every mass change<=5%. Inspect geometry.
Do not reroll, tune, select intermediate checkpoints or hide failures.

This milestone prepares training; it does not establish improved G7 quality.
MG7 remains live. Repository synchronization and off-device backup are pending.
Use RESUME.json here; the checkout's old D098 resume is stale. The original
NCA next-phase report stays local and untouched.
