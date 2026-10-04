# R3 independent review — 2026-10-04

R3 passes all frozen gates on69 regression cases and12 new cases.

## What was frozen

R3's adapter and G10 final427 checkpoint were frozen by SHA-256 before inference.
The route helper was extracted from the earlier geometry audit and reproduced
all45 saved TRAIN routes exactly before use on new scenes. Planner, admission
ordering, cumulative allowance, score/firing, nine families and thresholds were
unchanged throughout evaluation. There were no training updates or retuning.

The69 previous cases are regression evidence. The12 new cases comprise four
new geometries, each at16%,24%,32% requested volume. Geometry hashes use the six
physical context channels, excluding scene names and requested volume.
They were disjoint from all38 previously known unique TRAIN/evaluation contexts
and from one another before inference. This is a small related synthetic set,
not broad out-of-distribution or real-building validation. It is now consumed;
future work must treat these cases as exposed regression evidence.

Same32-cubed domain resolution,0.8m voxels, CPU float32, deterministic two-thread
execution, firing seed2101. No independent training/firing seeds were added.
R3 and new-case G10 run to128 with64 recorded as a prefix, not a separate run.
Old G10 results were reused only after original hashes and exact context checks.

## Numerical results

| Cohort/model | Steps | All nine | Evaluated outputs | Median volume error (pp) | Max error (pp) |
|---|---:|---:|---:|---:|---:|
| regression_G10 | 64 | 51/69 | 69 | 0.147 | 0.569 |
| regression_G10 | 128 | 51/69 | 69 | 0.139 | 0.180 |
| regression_R3 | 64 | 69/69 | 69 | 0.139 | 0.180 |
| regression_R3 | 128 | 69/69 | 69 | 0.139 | 0.180 |
| fresh_G10 | 64 | 2/12 | 12 | 0.123 | 0.204 |
| fresh_G10 | 128 | 2/12 | 12 | 0.122 | 0.134 |
| fresh_R3 | 64 | 12/12 | 12 | 0.122 | 0.134 |
| fresh_R3 | 128 | 12/12 | 12 | 0.122 | 0.134 |

Required: all nine families on every case; median error<=2 percentage points,
maximum<=4; and mass growth64–128<=5% for every case.
Certificate failures count in the denominator and fail admission; no geometry
was replaced, omitted or procedurally rescued after seeing results.

Stability:
- regression_G10: 69/69; maximum growth 4.597%
- regression_R3: 69/69; maximum growth 0.000%
- fresh_G10: 12/12; maximum growth 1.701%
- fresh_R3: 12/12; maximum growth 0.000%

Certificate failures: 0.
R3's overall frozen numerical acceptance decision: True.
This decision is limited to the assessed pilot domain; it is not automatic
deployment approval or evidence of structural safety.

## Paired changes

- regression at64: 18 family-pass gains, 0 losses versus raw G10.
- regression at128: 18 family-pass gains, 0 losses versus raw G10.
- fresh at64: 10 family-pass gains, 0 losses versus raw G10.
- fresh at128: 10 family-pass gains, 0 losses versus raw G10.

Individual gains, losses, certificate failures and stability failures are
listed in audit.json and the per-case records. Every model uses the same
condition and request for each comparison. Three requests share each geometry
and are correlated; do not interpret12 requests as12 independent sites.

## What is learned and what is procedural

R3 is a hybrid. A geometry planner supplies a legal thick connection and minimum
coverage witness, and mandatory witness additions bypass neural firing/scores.
G10 weights choose other eligible additions subject to global admission guards.
Cumulative allowance carries unused capacity forward, allowing later individual
steps to exceed K while retaining the original cumulative and final ceilings.

Median planner-born share of added voxels at128:
regression 33.0%,
new sample 32.6%.
Seed excluded; admission provenance is not causal attribution.
Hard-cap stability and planner-enforced connections are not independently
learned NCA capabilities. Preserving meaningful volumes does not define rooms,
habitable interiors, program or structural performance.

## Verification and visual review

All frozen input/source hashes remain unchanged. Checkpoint identity and hash
were verified. Every successful hybrid trajectory was checked for no deletion,
legal additions, cumulative/global ceilings, exact witness-union capacity and
agreement with saved trace masses. All saved hybrid horizon states are finite.
Both model outputs were checked for connected full-cube volume at each horizon.

Visual selection is explicit: all fresh cases plus every regression case with
a family or certificate failure in either model, at both horizons.
30 paired cases were rendered and every comparison sheet inspected.
Other regression cases received numerical review, not a claim of exhaustive
visual inspection. Raw voxel surfaces and context wireframes are shown without
smoothing; no interior-section or aesthetic certification is implied.

## Decision and next work

Prepare a separate, clearly labelled local hybrid preview with G10 raw comparison, provenance and honest failure display. Keep MG7 live until the user reviews the preview. Before any general release, extend assessment to more seeds and geometry families and benchmark larger-grid costs separately.

Do not call this a freshly trained model, silently replace MG7, or claim a pure
local NCA system. Preserve raw G10 and hybrid outputs side by side. The nine
constraint families and building-volume concept remain unchanged.
No new paid training was needed.

## Preservation and resume

Source, fixed checkpoint, protocol, all81 contexts and certificate attempts,
324 per-model/horizon records (including failures), successful fields,
hybrid trajectories/provenance, fresh raw G10 trajectories, metrics and figures
are preserved in this folder and a verified archive. The archived original
G10 evidence remains the source for old raw trajectories.

Read RESUME.json here next. Repository synchronization is pending.
No Drive access, publication, remote push or live-model change occurred.
The verified archive is on the same disk, not an off-device backup.
