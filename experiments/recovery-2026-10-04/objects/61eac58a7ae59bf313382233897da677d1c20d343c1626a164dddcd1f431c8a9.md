# Reversible repair local implementation

The learned reversible update is implemented as a separate local module and has
passed focused CPU checks. It is not trained for model quality, GPU-ready, or
integrated into the live application. Repository write access was not granted;
all code, evidence and resume records are saved in this directory.

## Implemented behavior

reversible_repair.py retains the CGR1 network and hidden-state update. Its fired
active set includes generated occupied cells and empty six-face frontier cells,
excluding immutable original occupancy. Probabilities choose keep/add/remove.
An exact CPU flood fill then removes candidate components that have no path to
any original voxel. This deliberately combines local neural updates with global
connectivity cleanup. It preserves originals but does not merge disconnected
original components or guarantee thick connections.

Training classifies active cells against target occupancy and uses the specified
soft pre-cleanup local-volume loss. Hard choices and cleanup remain detached.
The volume proxy correctly replaces active occupied cells by q rather than
adding q on top of them. Targets are used for supervision only. Captures include
candidate and accepted fields,probabilities,births,direct removals and projection
removals at every step. Semantic identity rejects CGR1 checkpoint restores.

## Verification

Three consolidated tests passed in3.062seconds:
- Exact agreement with the independent reference transition across12 seeded
  stochastic steps,plus explicit deletion and rebirth of generated cells.
- Gradient signs favor retaining a correct added cell,adding a missing cell,and
  deleting an incorrect addition. Empty active sets give zero loss. Target-present
  and target-absent rollouts give identical states under identical firing draws.
- Restoring a saved optimizer-step checkpoint reproduces the next trace,field,
  model,optimizer and RNG payload exactly; prior CGR1 checkpoint identity rejected.

The delete/rebirth probe and training recovery check are separate. This does not
claim verified recovery halfway through a rollout or exact GPU recovery. Learned
rollouts use the original binary/input validation inherited from CGR1. Synthetic
checks establish implementation behavior,not architectural quality.

## Full grid CPU timing

Used the first TRAIN example of each damage type,32cubed,32steps,2CPUthreads,
Torch2.8.0+cpu and NumPy2.5.2. All forward timings include captures and cleanup.
Fresh untrained weights were reset identically for each model and case. A backward
pass tested finite gradients; no optimizer updated these training-example models.

| Example | CGR1 forward seconds | Reversible forward seconds | CGR1 forward and backward | Reversible forward and backward |
|---|---:|---:|---:|---:|
| intact |1.174|1.344|3.796|5.123|
| cube damage |1.238|1.107|3.645|2.882|
| slab damage |.782|1.137|3.017|2.874|

These are single samples,not stable speed comparisons. The timings mix scheduling,
warmup and algorithm cost; do not infer that reversibility speeds training.
A fixed positive-bias stress case grew3492cells in both variants. Forward-only
runtime was.863s for CGR1 and1.156s for reversible repair. Global cleanup is
tractable in these CPU probes,but host/device synchronization on GPU is unmeasured.
Do not extrapolate a paired GPU job duration from these measurements.

## Next gate

Prepare a bounded GPU compatibility and timing package for the reversible update,
including exact recovery and full-grid cleanup cost. It must stop without launching
a quality trial. Only after that result supports a runtime budget should a paired
CGR1 control/reversible comparison be proposed on the same GPU software stack.
The previous CGR3 recovery result does not cover this changed update rule.

The same frozen quality criteria and nine families remain. Keep full ZIP downloads,
all prior experiments and MG7 live. No paid GPU work,Drive access,live changes or
repository commit occurred. All new implementation is local and available for
review; repository synchronization is pending access.
