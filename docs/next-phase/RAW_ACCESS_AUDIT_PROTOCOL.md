# A3: access-gradient recovery audit

Frozen 2026-09-23 before pilot outcomes. Implements ACCESS_RECOVERY_PLAN, D047.
No optimizer update, paid computation, architecture/default change or promotion.

## Fixed evidence matrix

231 frozen fields: all 36 original and 72 F2 fields from H1; all 72 final F3
fields; all 17 feasible W1 witnesses; all 34 final D1 fields. These are repeated
development-scene evaluations, not 231 independent scenes. W1 uses material as
raw explicitly; D1 uses direct voxel parameters, not network outputs. Source
run IDs and artifact hashes are frozen in A3-raw-access.json and run protocols.

Eight newly measured gradient cases: four final F3 models, horizons 16 and 50,
firing seed 2. No reused gradient vector is reported as new. Six full parameter
vectors per case: access v2/v3, coverage, sparsity, total v2/v3; last-raw
derivatives, weighted norms/cosines and the access gradient through each earlier
raw state. Exact forward replay against saved F3 fields is required. Recorded
firing masks and critical-cell histories distinguish final clamp obstruction
from earlier clamps and stochastic non-firing. Trace instrumentation has a real
model forward, parameter-gradient and RNG parity regression.

## Candidate and invariant checks

Opt-in raw_component_bottleneck_v3 computes relu(1-b_raw) using the unchanged
permitted six-connected graph and entrance regions. Stable z/y/x activation
ties select one branch of a nonsmooth maximum/minimum. No unique derivative is
claimed at ties. Impossible legal graphs retain loss one, a false feasibility
flag and zero gradient. The original access.py and rollout remain unchanged.

For a legally connected graph require b_projected=clamp(b_raw,0,1). Require
identical binary BFS results on projected occupancy and thresholded raw. Save
critical-cell changes rather than assuming clipped and raw orderings coincide.
Tests cover signed/zero/saturated fields, ties, masks, multiple entrances,
overlap rejection, infeasible graphs, independent BFS and finite differences.

For each nonzero candidate parameter vector, test both signs of a normalized
parameter perturbation of L2 length 0.0001, with identical firing randomness.
Restore exact original weights in a finally block. Save perturbed forward
fields and access/total/mass values. These are reversible local secant probes,
not optimization updates or evidence of learned improvement. Record failures
to decrease; nonsmooth ties and clamps may invalidate a branch-based prediction.

## Timing and preservation

Pilot: first five records in deterministic source order from each of the five
groups (25 fields); mass_3-r0 at both horizons (2 gradients). Pilot selection is
fixed before results. CPU two threads, deterministic algorithms. Worker caps:
300 seconds for replay, 120 seconds per gradient; total pilot 600 seconds.

Study admission uses ONLY measured timing:
1.5 * (231/25 * replay-worker seconds + 8 * max gradient-worker seconds)
must be <=1500 seconds. Full audit has the same worker caps and 1500-second
total cap. Pilot artifacts must pass verification and scientific code/config/
source registry must match exactly before full launch. Source snapshots, logs,
failed attempts, vectors and individual cases are retained. Do not silently
increase caps or choose a favorable subset after outcomes.

Verifier rescans all registered hashes, rescoring fields and probe fields,
recomputing norms/cosines and checking exact saved-field parity. It does not
independently differentiate every recorded vector. These tests do not certify
GPU/AMP behavior or generalization. Follow any result with a separate decision;
nonzero access gradients alone do not admit training.
