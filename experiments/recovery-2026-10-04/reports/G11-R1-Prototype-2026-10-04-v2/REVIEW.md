# G11-R1 local prototype review — 2026-10-04

G11-R1 repairs all four TRAIN family failures but is NOT accepted: eight cases
fail stability and the largest 64-step volume error is 4.344 percentage points,
above the unchanged 4-point maximum. G10 itself has three TRAIN stability
failures here; its earlier 69-case evaluation was a different set.

## Implemented change

A separate inference adapter reserves a context-derived connected full-cube
witness meeting the existing nine families. Fixed G10 proposal weights grow
around it. Candidate unions must leave enough capacity to complete the witness
and must preserve its final facade ratio. Existing legality, connected cube
growth, quota and total volume ceiling remain in force.

This is explicitly a hybrid planner/NCA. Witness cubes receive priority and
bypass learned scores and stochastic firing. Learned cubes still use score >0.5
and firing probability 0.5. The same random stream continues to update hidden
state. No optimizer update or new paid training occurred.

The witness covers connection and minimum site coverage; it does not prescribe
the full requested mass. Its remaining completion cost is counted by exact voxel
union, including overlaps. Planning uses site geometry, not teacher shapes.
The route and coverage code are deterministic and versioned.

## Frozen local comparison

All 45 existing TRAIN inputs were used, with fixed G10 final checkpoint427,
CPU float32, two threads, firing seed2101 and horizons64/128. Each model was
run once to128;64 is its recorded prefix. These are not independent horizon
replays. No held-out scene was evaluated or selected for this prototype.

45/45 witnesses pass the nine-family certificate and cap
before inference. Witness planning median time: 0.242s.
All 45 paired cases were evaluated if all certificates succeeded; certificate
failures are preserved and must not be omitted from an overall success claim.

| Model | Steps | Nine families | Median volume error (pp) | Max error (pp) |
|---|---:|---:|---:|---:|
| G10 | 64 | 41/45 | 0.193 | 1.009 |
| G10 | 128 | 41/45 | 0.193 | 0.263 |
| G11-R1 | 64 | 45/45 | 0.239 | 4.344 |
| G11-R1 | 128 | 45/45 | 0.193 | 0.263 |

Stability (<=5% growth64–128): G10 42/45;
G11-R1 37/45.
Maximum growth: G10 7.96%;
G11-R1 16.65%.
Reserved witness fully present by64 in 45/45 cases.

Median128-step rollout time, excluding planning:
G10 3.098s;
G11-R1 3.535s.
These single-run local CPU timings are descriptive, not a repeated latency
benchmark or GPU performance estimate.

## Attribution and limitations

Planner-born share of newly occupied voxels at128:
minimum 23.3%, median 33.1%,
maximum 51.9%.
The seed is excluded from the denominator. A cube admitted through the planner
is counted as procedural even if the network might also have proposed it.
These shares record the executed decision path; they are not causal estimates
of how many voxels would be impossible without planning.

The method reserves a particular route and can bias morphology. Passing these
synthetic TRAIN cases does not establish generalization, diversity, architectural
quality or mechanical safety. The witness deliberately builds in some metric
requirements; passing them is not evidence the NCA learned those requirements.
No strict completion deadline is guaranteed for unseen geometry.

## Audit and preservation

Replayed all 5,760 hybrid birth/provenance accounts: no deletion, no illegal
births, unchanged per-step/global ceilings, exact witness-union reservation,
and agreement with saved64/128 output fields. Checkpoint hash is unchanged.
Full-cube depth/connectivity were checked on both output horizons for both models.
Saved four boundary checks cover valid certification, cap shortage, a severed
critical bridge and blocked-plane invalidation; they test certificate predicates,
not completeness of search. Boundary checks ran alongside the paired evaluation.

One first attempt failed before model inference because a NumPy Boolean was
not JSON serializable. Its source and partial output are preserved separately
and copied into this archive. The v2 attempt converts that scalar to bool;
the model/algorithm did not change as a result.

All eight comparison sheets were visually inspected. They show raw exposed
voxel surfaces with existing-context wireframes. They do not show interior
sections. Individual outcomes and failed families remain in case JSON records.

## Decision and next step

Retain G11-R1 as a local experimental hybrid baseline. Do not replace MG7.
Inspect the numerical gates and planner share together; do not describe this
as a newly trained model or automatic G10 improvement.

Next inspect scheduling on the saved TRAIN traces before consuming reserved
cases: distinguish unused per-step capacity, whole-cube packing and proposals
blocked by the witness/facade guards. Current trace records aggregate rejection
only; instrument exact reasons in one bounded diagnostic if necessary. All
witnesses finish by64, so remaining late volume is surrounding mass, not an
unfinished connection. Do not assume planner-first priority alone is causal.
Select one scheduling change with explicit before/after semantics. Keep quotas,
horizons and acceptance thresholds unchanged; no blind sweep or paid training.
Only after TRAIN timing and volume gates pass should a new, disjoint reserved
set be frozen for paired raw/hybrid assessment. No live preview admission yet.

Raw states, birth provenance, per-step accounts, witness paths, contexts, source,
checkpoint and metrics are archived. Original G10 evidence is untouched.
Repository synchronization remains pending; use this RESUME.json.
The verified archive is a same-disk copy, not an off-device backup.
No Drive action, push, publication or paid run occurred.
