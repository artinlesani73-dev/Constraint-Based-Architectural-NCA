# NR2: portable Colab preparation, then a separately approved GPU preflight

Frozen before NR2 local execution, 2026-09-25. Keep NR1 code/weights/evidence and
the nine-family massing contract unchanged. This milestone creates a portable
preflight package, not the proposed three-model quality experiment.

## Exact scope

Include all81 TRAIN variants of27 NL0 teachers, sorted by (case,damage), and only
the Python dependency closure needed for repair and recovery. Do not include
validation/test/blocked examples, private reports, old checkpoints, unrelated
user files, credentials or Git history. Use exact NL0 example bytes and hashes.
Hash every package member, verify the ZIP before extraction, prohibit traversal,
symlinks/duplicates and overwrite, and rehearse the extracted package locally.
Source manifest hashes and exact environment identities guard worker restart.

Reuse the unchanged NR1 model,16-step rollout and balanced BCE/Adam settings.
The deliberate new behavior is sampling: a private CPU generator seeded1203
shuffles all81 rows without replacement; repeat with a new permutation after
each full epoch. Store order, position, epoch, consumed count and sampler RNG.
Seed1201 initializes weights and global RNG;1202 initializes device-local firing.
Fresh state per update; no pool, AMP, scheduler, batch increase or coefficient sweep.
This changes data exposure, not the loss or architecture. Never claim GPU outputs
must equal CPU outputs; compare interrupted/uninterrupted runs on the same device.

Portable checkpoints store CPU copies of weights/Adam, Python/NumPy/CPU Torch
RNG, device firing RNG and all CUDA RNG states, sampler and loss trace. CUDA
restore requires the same host, GPU UUID/driver, model/device/library versions,
precision settings, source and data identity. A new Colab VM or GPU is not
automatically compatible. Atomic payload publication and manifest commit markers
operate only on local runtime storage; no Drive/FUSE checkpoint writes.

Before launching CUDA workers set CUBLAS_WORKSPACE_CONFIG=:4096:8; enable strict
deterministic algorithms, disable cuDNN benchmarking and TF32, and use float32.
An unsupported deterministic operation is a retained failure, never permission
to weaken the check. Record Python/Torch/NumPy/CUDA/cuDNN, GPU UUID/name/capability,
driver, full Torch build configuration, peak allocated/reserved GPU memory and
synchronized update timing. PyTorch warns reproducibility is not guaranteed across
releases/platforms or CPU versus GPU; settings alone are not proof of replay.
[PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Fixed preflight execution

One uninterrupted8-update run; one fresh run deliberately stopped after a durable
update4 checkpoint; one new process resumed for updates5..8. Three owned workers,
8 unique/16 executed updates,10 full checkpoint-pair comparisons and8 complete
training-state-pair comparisons. Boundary4/8 raw states/probabilities also match.
Separate evaluation firing seed2101 and training row0 never consume training RNG.
Replay must be exact, gradients finite/nonzero, and model weights must change.
All completed workers' peak reserved memory must be <=80% of total GPU memory.
These are mechanics/resource gates, not nine-family quality or convergence gates.

Parent enforces one600-second execution deadline across verification, workers and
comparison. Ownership/start handshake precedes learning; Windows uses WorkerTree,
Linux uses a separate process group; a worker stdin-EOF watchdog stops it if the
parent disappears. An intentional interruption and timeout stop only owned work.
Every checkpoint/trace/state/log is retained, including failures and partial files.
The GPU package admits one attempted preflight only: a persistent exclusive marker
blocks rerunning it. A retry requires a new reviewed attempt, not automatic spending.
There is no command for256-update training or automatic continuation after success.

The notebook is disarmed by default. Activating a GPU job requires an explicit
approval flag. The600s cap limits job execution, not provider billing: setup,
uploads/downloads and an idle connected GPU runtime can consume additional time.
Do not promise a fixed dollar cost or that a10-minute runtime allocation is enforced.
User should disconnect the GPU after preserving the downloadable evidence.

## Local validation and limits

Tests cover sampler epoch-boundary recovery, exact same-input CPU math/Adam parity
with NR1, full multiexample restart, split and identity guards, corrupt/uncommitted
checkpoint rejection, overwrite prevention and malicious archive paths. Run the
full regression, then one extracted-package CPU rehearsal of the frozen three
workers. This uses real training inputs for mechanics only and must not be called
a GPU result or a repaired-model benchmark. No CUDA execution can be certified
on the currently installed CPU-only PyTorch.

The finished package includes an empty-output notebook, source/data ZIP with an
embedded expected checksum, integrity receipts and a plain-language run guide.
No internet install is performed automatically; use Colab's existing PyTorch/NumPy,
record actual versions and stop/report an incompatibility before further spending.
Check package extraction from a path outside the original repository so accidental
imports of local source cannot conceal missing dependencies.

## Backup and permission boundary

The preflight exports all evidence and checksums into a downloadable ZIP on success
or ordinary failure. Download it before disconnecting and keep the local package.
This cannot preserve unsent files if the entire VM is deleted mid-run. Colab states
its VMs have enforced lifetimes and may be deleted when idle. Process recovery
therefore does not certify runtime-loss recovery or an off-device backup.
[Colab FAQ](https://research.google.com/colaboratory/faq.html).

No Drive connector operation, mount, automatic sync, notebook upload/publication
or remote GPU job is authorized by preparation. Notebook UI storage is separate
from runtime storage: a 'Save a copy in Drive' action must use the project's sole
approved folder and receive its own explicit permission. Extended training waits
for an approved backup arrangement and GPU preflight evidence. The earlier60-minute
quality-study allowance remains a proposal and is not unlocked by this package.

## Local readiness correction before package revision 2

The first extracted-package CPU rehearsal (20260925T110616Z_d542bd6412fa)
stalled during NumPy native import before creating a model or checkpoint. A
retained subprocess stack trace reproduced the stall with the blocking stdin
watchdog active. Windows now polls the pipe with PeekNamedPipe instead of holding
a blocking CRT stdin read during native imports. A real child-process test checks
that NumPy/PyTorch import, the deadline is detected and closing the parent pipe
terminates the child with exit71. Linux retains the raw pipe EOF watcher.

The first coordinator also exceeded its600s monotonic cap while waiting on a
single Windows process wait. Revision2 polls process state and rechecks the
monotonic deadline every50ms; it does not rely on a single long OS wait. Machine
suspension itself cannot be prevented. All initial package bytes, logs, failed
records and the diagnostic are preserved. The original package is superseded;
use only the separately hashed revision2 after its own readiness results pass.
