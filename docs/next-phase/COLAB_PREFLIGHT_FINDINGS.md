# NR2: Colab preflight prepared; CPU replay verified, GPU pending

2026-09-25. Use **Codex outputs/NR2-Colab-Preflight-v2**. The original
NR2-Colab-Preflight directory is retained as a superseded failed-readiness package.
The current package and disarmed notebook are local files; no Colab, paid GPU,
Drive, deployment or remote Git operation occurred.

## What is ready

The ZIP contains 81 TRAIN examples from 27 NL0 volumes and the required source,
with 96 payload files plus its manifest. Validation/test arrays, private reports,
historical checkpoints, credentials and Git history are excluded. The notebook
checks the whole ZIP before executing its extraction helper; member hashes and
safe paths are checked again. All code cells compile and have empty outputs.

The model, loss and Adam settings retain NR1 mathematics. The new sampler shuffles
the 81 training rows without replacement with its own seed1203 and records its
permutation, cursor, epoch and random state. Checkpoints preserve weights, Adam,
Python/NumPy/Torch/firing/CUDA randomness, sampler, trace and runtime identity.
Same-input CPU tests confirm exact agreement with the prior NR1 implementation.

The fixed prospective GPU preflight has 8 unique updates and 16 executed updates
including replay: uninterrupted8, fresh4 then deliberate stop, and a new process
resuming4. It requires exact replay on the same GPU/runtime, finite positive
gradients, changed weights and reserved memory at most80% of GPU capacity. There
is no longer-training entry point or automatic continuation. Its one-attempt
marker blocks accidental reruns of that extraction.

## Local evidence

Focused 20260925T112534Z_1553fde22b4c: 7 tests pass. Regression 20260925T112716Z_bd5f2b1cfe5d:
371 tests pass, no failures/errors/skips, original-checkpoint smoke exit0.
The final model/driver/test bytes match these snapshots; only the protocol's
failure-correction explanation was appended after focused tests. Full regression
and the successful rehearsal contain the final frozen source bytes.

Extracted-package rehearsal 20260925T113206Z_e84a09a6bda5 completed in 38.91s of its600s
execution allowance. Imports were checked in a separate process and resolved
inside the extraction outside the project repository. All three owned Windows
worker trees have zero active processes after completion/termination. The planned
interruption is retained, including its PID and exit code.

All 10 complete checkpoint pairs, 8 pre-update rollout-state pairs and two
evaluation-boundary pairs match exactly. A separate post-run inspection repeats
these comparisons, checks every source hash and both attempts' evidence exports,
and regenerates the eight sampled indices: [52, 38, 65, 36, 70, 63, 56, 75]. They are distinct TRAIN
rows. The CPU result explicitly reports gpu_recovery_passed=false. No GPU hardware,
Linux process-control path, GPU memory budget or CUDA restart has been validated.

This is training infrastructure evidence, not a repair-quality benchmark. Eight
updates do not measure convergence/generalization or establish usable geometry.
NR1's unchanged binary outputs remain a relevant negative result. No model is
promoted into Studio; the existing procedural MG7 workflow remains the comparator.

## Failed attempt and correction

Initial focused tests passed6 and initial regression passed370. Nevertheless,
the real package rehearsal 20260925T110616Z_d542bd6412fa stalled before its first model/update while
importing NumPy. An isolated diagnostic reproduced this and saved both thread
stacks. The blocking stdin watchdog was active during native module initialization.
Windows now uses nonblocking PeekNamedPipe polling. A child-process regression
imports NumPy/PyTorch with the watchdog running, detects a deadline and verifies
that closing the parent pipe ends the worker with exit71. This test passed.

The initial coordinator's elapsed time was995.19s despite the600s configured cap.
Its single long OS wait did not provide the intended monotonic deadline behavior.
The exact reason for the delay (including any machine suspension) is not certified.
The stalled worker was explicitly stopped after checking its command/PID; the
coordinator reported failure and zero remaining owned processes. There were no
model checkpoints or training updates from that attempt. Revision2 polls process
state and rechecks the monotonic deadline every50ms; the local wrapper also owns
and cleans up its coordinator tree. Machine suspension cannot itself be prevented.

Both ZIP/notebook versions, both configs, all six run records, failed logs/export,
the diagnostic stack and every successful checkpoint remain preserved. No old
result or artifact was overwritten. Version2 is linked to the failed attempt.

## User's next action and remaining limits

Read START-HERE.md in the version2 package directory. Before remote work, ask for
the exact notebook save action inside Drive folder1fS34Yy0-oMzSxWaYJFiPTkGgrZstgc0H
and any required verification as an explicit batch. GPU execution needs separate
approval for one preflight, oneGPU,8 unique/16 executed updates,600s job cap.
The default notebook approval flag remains false. No permission is inferred from
local preparation. The earlier60-minute/three-seed/256-update quality study remains
unapproved and requires separate evaluation and backup planning.

The600s cap covers the controlled job, not Colab setup/download/idle GPU billing.
No fixed monetary cost is promised. Download the evidence ZIP and receipt, verify
their local presence/content and disconnect the GPU. The export records successes
and ordinary failures. Whole-VM deletion before download can still lose runtime
files; process recovery does not establish off-device durability. A new VM/GPU
may fail the strict identity guard; do not bypass it or relax exactness tolerances.
No Drive mount, auto-install, automatic sync, retry or longer run is included.

## Durable record

Frozen configs: NR2-readiness.json (historical) and NR2-readiness-v2.json (current).
Protocol: COLAB_PREFLIGHT_PROTOCOL.md. Reports: experiments/reports/NR2-summary.json
and NR2-verification.json. Raw runs: .local-artifacts/runs/<run-id>. Extracted
packages: Codex outputs/nr2-rehearsals/<run-id>. Diagnostic and helpers:
Codex outputs/nr2-startup-diagnostic.log and work/*nr2*.py.

Current ZIP SHA256: d4979d83f4a78ead3054e84f1983227df9493877bdfcf49c635f7a0ecebd873f.
Current notebook SHA256: 1774d516b10ad75819d0ed6078c9ebb86ee18b1fbe771273df2682480312002c.
Incremental backup completion is certified by its external receipt. Keep NR1/NL0
and all older archives. This is local same-disk preservation, not an off-device
backup. Private reports remain unchanged and ignored; unrelated files stay untracked.
Restore exact source ZIP bytes for hash replay, since Git may normalize newlines.
