# G3 budget-feedback pilot

G3 implements the documented hybrid budget design. It has not demonstrated improved generated geometry yet. No paid run has started.

## What changed

The network receives one new dynamic input: remaining desired volume divided by domain size. The first layer expands60 to61 inputs, adding64 weights; existing parameters are copied from an identically seeded fresh G2-size initialization, and new weights start at zero. There is no trained-model warm start.

Training retains G2's positive/negative frontier weights1:1 and local cube-volume term0.25. A coefficient1 global budget-band loss is added before the hard admission decision. The requested count uses the existing narrow0–8-voxel quantization allowance, with the40% general maximum retained. Original27 TRAIN arrays,50/50 start schedule,seed1201,256 updates,64 steps and optimizer remain unchanged.

Each step ranks fired legal adjacent proposals above0.5 and admits only what fits in the remaining count allowance. Stable ZYX ties are explicit. Global mass counting, broadcasting and sorting make this a hybrid NCA. The cap enforces the count limit; it does not establish learned volume control or guarantee access, coverage or thickness. Wrong early additions remain irreversible.

## Executed checks

INTEGRATION-VERIFICATION.json records:

- Device-tensor admission matches the independent NumPy reference on eight seeded cases including ties and zero capacity.
- Fresh base parameter copies are exact, new weights are zero, and only64 parameters are added.
- Forced-growth rollouts retain the seed, connected additions and count limit; raw rejected proposals remain measurable.
- Overfull initial states are rejected.
- Budget derivatives encourage additions below the band, discourage them above it and vanish within it; inactive proposal gradients vanish.
- The integrated objective has finite gradients, including a nonzero gradient in the new channel; inference has no teacher dependency.
- G2 checkpoints are rejected as G3 continuations.

The combined CPU rehearsal performs three retained updates plus two exact checkpoint replays covering teacher-stage and seed starts. Final status, hashes and checkpoint checks are in VERIFICATION.json. These are implementation checks; no G3 quality claim follows. The notebook cells compile, but the notebook has not been executed in Colab here.

## Proposed one-job budget

One Tesla T4, seed1201,256 retained updates plus two recovery replays,64 growth steps, batch1,float32. Cap600 controlled wall seconds including startup, eight device/reference admission probes, recovery and checkpoint writes. Setup, export, downloads and idle are additional. No automatic retry or extra seeds.

The job first verifies the fixed cu130 runtime and checks device selection against the reference; recovery must pass before continuing beyond update3. It stops on failure, nonfinite values or the80% reserved-memory cap. GPU admission speed, determinism and G3 recovery are not claimed verified until this job returns its evidence.

After approval for that exact budget:

1. Open NCA-G3-Budget.ipynb in Colab; select a T4 GPU.
2. Upload NCA-G3-Budget-Package.zip in its upload cell.
3. Set APPROVED_G3_JOB=True and run once.
4. Download and return the full ZIP and receipt, including failures.

The frozen review remains final checkpoint256, CPUfloat32, firing2101, nine development requests, single-seed64-step outputs and a fixed128-step stability diagnostic. All nine geometry families and original quality gates remain unchanged. Reserved targets remain unopened. Budget/stability compliance caused by the cap must be labeled as enforced. Report budget rejections and pre-admission candidates alongside admitted geometry; do not confuse a per-step candidate with a no-guard rollout.

## Resume and preservation

The G3 reviewer must instantiate `BudgetNCA` from the exact packaged source (61 inputs), not `ConnectedRepair`. Use the same seed adapter and seven static channels; the dynamic feature is calculated internally. The result includes per-step admission counts and budget tuple. For candidate diagnostics, call rollout with capture=True; teacher labels are not inference arguments. Keep original G1/G2 arrays and metrics as comparisons; reused development is not fresh held-out evidence.

MG7 remains live. No repository edits or commits, Drive operations, pushes or public deployment occurred. Repository sync remains pending because write access was not granted. Source, package, tests, rehearsal, protocol and resume records are preserved in a verified local archive; this remains a same-disk copy rather than an off-device backup.
