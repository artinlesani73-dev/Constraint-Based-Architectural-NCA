# NR1: exact CPU restart passes; binary repair remains unproven

2026-09-25. Pilot `20260925T100650Z_78bc22b9d584` completed in61.99s
against its600s cap. Implemented the proposed4424-parameter, eight-state-channel
conditioned NCA, balanced reconstruction loss and versioned CPU recovery path.
Two independent TRAIN-example members receive8 updates each from the same fresh
seed1201. No old material checkpoint, target mask conditioning, new constraint
family, paid compute or production model replacement.

## What passed

Six separately owned workers performed uninterrupted8, interrupted4 and resumed4
updates for each member. Both deliberate interruptions stopped the complete owned
Windows process tree at a committed update4 checkpoint. Actual worker PIDs differ
from their venv-launcher PIDs; both are retained. All six trees report zero active
processes after completion/termination. Two exit codes1 represent planned kills,
not unreported failed experiments. No unexpected execution failure occurred.

All20 complete checkpoint comparisons match exactly: weights, Adam state,
Python/NumPy/global Torch/private firing RNG, data cursor, loss trace, model mode,
source/config/data/runtime identity. Independent post-run verification also checks
all16 pre-update rollout-state pairs,14 saved boundary records and every saved
source-snapshot member. Boundary arrays and family scores match at update4/8.
NumPy sigmoid/projection agrees with saved probabilities within declared float32
tolerance; replayed PyTorch states/probabilities/fields match exactly.

Unique exposure is16 updates across two members;32 optimizer updates were actually
executed including replay branches. Do not count replay as extra training or the
same-seed/two-example result as independently replicated model-quality evidence.
All4 parameter tensors change in each model, all16 original-update global gradient
norms are positive/finite and none exceeds the clipping threshold1.0. This verifies
that updates occur, not that the learned rule is useful or that clipping is generally
unnecessary. Every model/update/state/checkpoint remains available.

## What the geometry shows

| Member | Evaluation loss: initial / final | IoU: initial / final | Final nine-family result |
|---|---:|---:|---|
| aligned_cube | 0.245131 / 0.216262 | 0.881797 / 0.881797 | Fails |
| partial_slab | 0.222166 / 0.193450 | 0.904762 / 0.904762 | Fails |


Final thresholded fields are exactly the damaged inputs in both members. Missing
cells remain below0.5, and no original-shape repair is achieved. Both final designs
still fail MT1. The full all-family verdicts, raw logits/hidden states, projected
probabilities, missing-cell recovery, false additions, surviving-cell removal and
request error are saved. Read NR1-summary.json for failed families and probability
ranges. Do not equate lower continuous loss with better binary geometry.

The result meets the predeclared **mechanics** gate; this8-update pilot was never
defined as a quality/generalization gate. It neither admits Studio promotion nor
proves a longer run would succeed. It also does not prove this representation
cannot learn repair. No threshold, loss, schedule, example or duration was retuned
after seeing these outcomes. The frozen NR1 quality proposal remains unexecuted.

## Tests and source review

Focused run `20260925T100223Z_4f20845cb058`:7/7 new tests pass. They cover local
perception/initial parity, saturated-logit loss gradients, actual optimizer replay,
source/data mismatch rejection, partial publication and overwrite protection,
corrupt-newest fallback, context mutation and legal projection. Test-created
corrupt/partial files are deliberately retained under .local-artifacts/testing/NR1.

Full regression `20260925T100255Z_b00e322820cb`:364 tests,0 failures/errors/skips,
original-checkpoint smoke0,221.23s. The pilot ran after this suite
finished; no overlapping compute-heavy study. This is a local CPU timing observation,
not a GPU performance forecast or NCA-versus-procedural speedup claim.

Before any pilot worker launched, review caught a Windows ownership issue in the
first driver: killing only a venv launcher could leave its worker child alive.
The final driver reuses the tested WorkerTree helper, waits for an ownership/start
handshake, kills its owned job and verifies no active descendants. The initial
driver/protocol/config are preserved in .local-artifacts/analysis-attempts/NR1-driver-review.
Focused/full snapshots predate that correction; their model/training/test sources
are identical to the pilot. Only the driver and ownership explanation changed.
The real six-worker pilot validates that final driver; the full suite was not
misrepresented as testing its later bytes. No learning code changed after testing.

## Limits and exact next step

Recovery is verified on this CPU environment at completed update boundaries.
Fault-injection checks reject uncommitted/corrupted files; they do not certify
power-loss durability, GPU/AMP, cross-device behavior or arbitrary backprop recovery.
The implementation intentionally rejects unsupported CPU/config/data identities;
there is no Drive/FUSE publication fallback. Boundaries use private evaluation
RNG so evaluation does not change the training sequence.

Next prepare a separately versioned Colab package: portable verified training-only
assets, explicit multi-example sampler, GPU checkpoint/RNG/recovery preflight,
time accounting, evaluation separation and local download/backup instructions.
Retain NR1 CPU as a control. Assess feasibility on the actual GPU before proposing
execution of the earlier60-minute/three-seed/256-update allowance. That allowance
is still a proposal; no cloud work or approved expenditure has begun. Given this
pilot's unchanged binary geometry, retain both loss and strict voxel-quality gates
in the next run. No automatic longer local training or blind weight sweep.

No action is required from the user for package preparation. A concrete ready
package and required steps must be presented before asking to spend Colab compute.
Drive access, including a mount or write/readback, requires exact separate permission.
Studio remains the verified procedural MG7 workflow.

## Durable evidence

Code: nca/repair_training.py; scripts/run_repair_pilot.py; tests/test_repair_training.py.
Frozen settings: experiments/configs/NR1-cpu.json and NCA_REPAIR_CPU_PROTOCOL.md.
Raw study: .local-artifacts/runs/20260925T100650Z_78bc22b9d584; worker folders retain
all checkpoints, SHA manifests, traces, states and interruption markers/logs.
Small reports: experiments/reports/NR1-summary.json and NR1-verification.json.
Post-run helper work/verify_nr1.py rescored fields with shared MT1 formulas; this
does not claim a second independent topology implementation.

Restore exact source-snapshot bytes for byte-hash replay, since Git can normalize
line endings. Do not bypass source guards. RESUME records the next step; the
external archive receipt certifies the local incremental backup. Keep NL0/MS2/MG7
and older archive chain. Same-disk archive is not off-device. Private reports are
unchanged and ignored; unrelated files remain untracked. No Drive, push, publication,
server restart or replacement of the displayed procedural volumes.
