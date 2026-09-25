# NR1: bounded CPU learning mechanics and exact recovery

2026-09-25, fixed before execution. Extends NCA_REPAIR_BASELINE_PROTOCOL's local
gate; the Colab model-quality proposal remains unexecuted and unapproved.

Use NL0 run20260925T094341Z_316cff241020 without rebuilding data. Exactly two
independent members, both fresh seed1201: aligned/v24/s0/cube5 and
partial_obstruction/v24/s0/slab2. Both are TRAIN rows. Each member sees its one
fixed damaged input on all8 updates (batch1). The next-sample cursor is the next
update index; there is no shuffled sampler in this mechanics-only pilot. A future
multi-example training sampler requires a separate frozen configuration. No
validation/test rows are used for learned updates or pilot evaluation.

The model is4424 parameters:15 channels perceived as identity plus central X/Y/Z
differences, in voxel units with zero exterior padding; concatenate32 evolving
and28 static features. Shared60->64->8 pointwise MLP; ReLU; zero final layer;
occupancy logit initialized+2/-2 and7 hidden channels zero. Independent per-cell
Bernoulli0.5 firing using a private CPU generator seeded1202. Exactly16 recurrent
steps/update, fresh state each time. Static perception is cached and immutable;
no trainable graph is cached. Hidden state is zero outside domain AND permission
mask. Raw logits outside that region may evolve and influence boundary cells;
output probability is explicitly zero there. Save raw logits/hidden state too.
There is no life mask, state pool, saturation clamp, scheduler, AMP or GPU path.

Balanced BCE is0.5*mean(softplus(-logit)) over legal positive target cells plus
0.5*mean(softplus(logit)) over legal negatives. Both classes required. Adam
lr0.001,betas0.9/0.999,eps1e-8,weight_decay0,amsgrad/foreach/fused false;
global gradient norm clipped to1. CPUfloat32,2threads,deterministic algorithms.
Target and cut region never enter model conditioning. Binary evaluation remains
strict projected probability>0.5, unmodified MT1 with the same nine families.
Restore exact declared request for metric calculation instead of rounded float32
conditioning. No learned-output postprocessing or soft-loss success criterion.

## Six owned workers, one total allowance

For each member, in order: uninterrupted8; restarted-from-initial worker stopped
after4; fresh worker resumed from that checkpoint through8. At update4 the second
worker saves checkpoint and evaluation, publishes an explicit ready marker with
its PID, and blocks on its input pipe. The parent verifies that marker and terminates
its owned worker tree. Windows venv launchers may have a different PID from the
actual Python worker; both are retained. Reuse the tested WorkerTree job object,
attach ownership before sending the initial GO handshake, and verify zero active
descendants after stopping. This also bounds timeout and parent-death cleanup.
This is deliberate interruption at a completed boundary,
not an arbitrary mid-backprop crash. Distinct PIDs and termination outcomes retained.

The fresh-process comparison certifies20 complete checkpoint pairs (0..4 for the
interrupted branch,4..8 for resumed; across two members), including weights, Adam,
Python/NumPy/global Torch/private firing RNG, model mode, metadata, complete loss
trace and next-sample cursor. Also compare scores and full states/probabilities/
binary fields at4 and8. Unique training exposure is16 updates;32 updates are
actually executed across original and replay branches. Synthetic unit-test updates
are separate and must never be counted as independent trained-model replications.

One600-second external wall deadline covers input checks, preparation and all
six workers/comparisons. Parent timeouts kill owned workers; completed checkpoints,
partial files, worker logs and failed/unfinished members remain. Evidence attachment
and final archive verification follow the bounded experiment and are separately
outside this computation cap. No automatic extension or retry in the same run.
Do not run another compute-heavy experiment alongside this bounded pilot.

## Checkpoint and evidence contract

After every completed optimizer update, atomically publish an exclusive .pt payload
on the local filesystem, then a SHA256/length/version/identity/cursor JSON manifest.
The manifest is the commit marker. Never overwrite a checkpoint. An interruption
before payload publication leaves the partial file; one before manifest publication
leaves an uncommitted payload. Recovery reads only verified manifests; corrupt
newest manifests/payloads fall back to older verified entries with rejection records.
Explicitly requested corrupt checkpoints fail. Fault-injection tests cover both
publication windows, overwrite rejection and corrupt-latest fallback.

Identity binds source hashes, recipe hash, NL0 manifest/member/NPZ identity,
actual occupancy/context/target tensors, model/optimizer identity, CPU/dtype/threads,
deterministic setting, Python/Torch/NumPy versions and both seeds. Mismatches fail.
Only local CPU hard-link publication is supported; do not assume this works on
Drive/FUSE or certifies power-loss persistence or GPU/AMP/cross-device recovery.
The payload loader uses weights_only=True after hash verification. Session tensors
are owned copies and checked for in-place mutation before updates/evaluation/save.

Save every checkpoint, update trace, pre-update rollout state, and boundary0/4/8
raw state/probability/binary array. Boundaries use independent firing seed2101 and
16steps; evaluation must not consume training RNG. Record every MT1 family, IoU,
missing-cell recovery, collateral additions/removals and requested-volume error.
Pre-update training fields belong to the weights BEFORE their numbered update;
boundary fields belong to the checkpoint AFTER the indicated completed updates.

Admission requires finite nonzero global gradients, changed weights, exact initial
binary parity, exact full replay and caps. Geometry quality after8updates is a
descriptive observation, never a promotion gate or generalization claim. Preserve
all failures even if mechanics passes. Studio/checkpoint defaults remain unchanged.

After mechanics admission, prepare the Colab package and GPU-specific preflight
proposal; do not start paid training. Freeze its multi-example sampler, GPU recovery,
time accounting and explicit allowance before asking for execution approval. All
Drive operations remain separately permission-scoped. No automated cloud write.
