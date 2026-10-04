# G8 exposure experiment — verified final427 review

**Extra training restores volume accuracy and improves legacy connection results,
but G8 still fails the complete acceptance gate.** Keep it as an experimental
research result. MG7 remains live; no model promotion occurred.

## Run integrity and controlled comparison

Run20261004T111257Z_f5eb0598cf17 completed427 retained updates in
335.864 controlled seconds
(329.176 worker seconds), within600.
Peak reservation:1446MiB.
Verified the original ZIP/receipt and all1719 payload hashes, exact package
manifest, runtime/identity, admission probes and both full-payload/state recovery
checks. All427 start fields, row selections and27328 transition accounts/caps
were verified against saved traces and geometry. All45 reconstructed TRAIN
contexts and seeds match the packaged fixtures.

G8 at update256 is numerically identical to G7 at update256 across model,
optimizer, sampler, training trace, random state and remaining non-identity
payload fields. Run identity is deliberately separate. This establishes a
reproduced common training prefix, rather than a merely similar initialization.
We evaluated only final427; update256 was checked for reproducibility, not
selected as an alternative model. Final checkpoint SHA256:
`2bca26ab350fe6a318d15b12fd0f3ee1da9d6f96d524df103a0b5249743df5ff`.

The remaining171 updates are the only training intervention relative to G7.
This supports attribution along this one deterministic training trajectory,
not a multi-seed estimate that more training always helps. Data, architecture,
loss, hard transition, thresholds, firing and review horizons stayed frozen.

## Separate frozen cohorts

| Cohort | Steps | All nine families | Median size error (pp) | Maximum size error (pp) |
|---|---:|---:|---:|---:|
| regression | 64 | 32/33 | 0.157 | 0.600 |
| regression | 128 | 32/33 | 0.156 | 0.180 |
| fresh_reserved | 64 | 8/12 | 0.119 | 0.387 |
| fresh_reserved | 128 | 8/12 | 0.132 | 0.141 |

The33 regression cases combine the original9 development,12 G6 reserved and12
G7 reserved requests. The12 fresh G8 cases are new synthetic combinations, not
the same cohort used to report G7's earlier fresh score. Do not compare those
two fresh-cohort percentages as if they measured the same cases. G7 was not
evaluated on the new G8 fresh cohort. No teacher geometry or route was provided
at inference. Fresh cases are now consumed evidence for any future work.

Stability within5% holds for33/33 regression and
12/12 fresh requests; maximum growth from64 to128 is
3.959%. Every individual gate is recorded in result.json; passing
the eight other families or matching volume does not excuse access failures.
Full geometric support and thickness are partly enforced by cube admission,
and do not establish structural safety or habitable interior space.

## Paired legacy improvement and remaining failures

On the same33 cases, G7 passed28/33 at64 and
30/33 at128; G8 passes32/33 and
32/33 respectively. Individual improvements and regressions
are preserved in comparison-summary.json and result-audit.json. This is a
substantial improvement in the previously observed size/timing problem, not
universal monotonic improvement in every connection.

The following cases still fail at128:

| Case | Failed families | East-interface occupied cells | Unused volume capacity |
|---|---|---:|---:|
| g7-vertical-reserved-2-v16 | access | 0 | 0 |
| g8-reserved-0-v16 | access | 0 | 0 |
| g8-reserved-0-v24 | access | 0 | 0 |
| g8-reserved-1-v16 | access | 0 | 0 |
| g8-reserved-2-v16 | access | 0 | 0 |

For a cap-exhausted failure, additional monotone growth steps cannot repair the
field: there is no remaining volume allowance and the transition cannot remove
or redistribute occupancy. The remaining problem is where mass is allocated
before saturation. Simply increasing the voxel count, relaxing the threshold,
or accepting a longer horizon is not justified by this result.

## Visual review

Reviewed the full scene plates through64-step and128-step overview sheets.
Outputs have substantial three-dimensional depth, stepped surfaces and exterior
gaps. The failed cases visibly miss a required connection despite bulk elsewhere;
good overall volume is therefore insufficient. The rendered colors and
projections are diagnostic geometry, not interior sections. Plate labels show
the number of passing families; overall acceptance also requires size and
stability gates. Full-resolution plates and all90 saved observations are retained.
This synthetic one-seed study does not establish diversity or architectural
quality across real sites.

## Decision and next action

Close the exposure experiment: the extra updates addressed size/timing but did
not guarantee access. Do not automatically launch another longer training run.
The next focused design task is to make completing required connections before
spending the mass budget an explicit learning priority within the existing
access family. Compare that proposed training objective with the current static
teacher-membership loss using TRAIN-only diagnostics; keep the other eight
families and volume semantics unchanged. Do not silently add a procedural bridge
or relabel repaired output as raw learned generation. Any procedural fallback
would be a separately documented hybrid design decision.

Prepare one concrete access-priority proposal before more paid compute, informed
by the earlier G5 cue failure and G6 objective audit. Preserve G8 as the latest
evaluated reference, with failure labels, without claiming it met the frozen
deployment gate. A saved-result gallery is acceptable for review; unrestricted
live replacement remains deferred.

## Individual case ledger

| Case | Families at64 | Families at128 | Error at64 (pp) | Mass change |
|---|---:|---:|---:|---:|
| g1-offset_interfaces-y0-v16 | 9/9 | 9/9 | 0.150 | 0.000% |
| g1-offset_interfaces-y0-v24 | 9/9 | 9/9 | 0.152 | 0.000% |
| g1-offset_interfaces-y0-v32 | 9/9 | 9/9 | 0.153 | 0.000% |
| g1-offset_interfaces-y2-v16 | 9/9 | 9/9 | 0.020 | 0.807% |
| g1-offset_interfaces-y2-v24 | 9/9 | 9/9 | 0.150 | 0.000% |
| g1-offset_interfaces-y2-v32 | 9/9 | 9/9 | 0.151 | 0.000% |
| g1-offset_interfaces-y4-v16 | 9/9 | 9/9 | 0.166 | 0.000% |
| g1-offset_interfaces-y4-v24 | 9/9 | 9/9 | 0.165 | 0.000% |
| g1-offset_interfaces-y4-v32 | 9/9 | 9/9 | 0.164 | 0.000% |
| g1-raised_pair-0-v16 | 9/9 | 9/9 | 0.105 | 1.699% |
| g1-raised_pair-0-v24 | 9/9 | 9/9 | 0.171 | 0.000% |
| g1-raised_pair-0-v32 | 9/9 | 9/9 | 0.157 | 0.000% |
| g1-raised_pair-1-v16 | 9/9 | 9/9 | 0.201 | 2.320% |
| g1-raised_pair-1-v24 | 9/9 | 9/9 | 0.171 | 0.000% |
| g1-raised_pair-1-v32 | 9/9 | 9/9 | 0.157 | 0.000% |
| g1-unequal_building_heights-0-v16 | 9/9 | 9/9 | 0.180 | 0.000% |
| g1-unequal_building_heights-0-v24 | 9/9 | 9/9 | 0.169 | 0.000% |
| g1-unequal_building_heights-0-v32 | 9/9 | 9/9 | 0.178 | 0.000% |
| g1-unequal_building_heights-1-v16 | 9/9 | 9/9 | 0.425 | 3.813% |
| g1-unequal_building_heights-1-v24 | 9/9 | 9/9 | 0.157 | 0.000% |
| g1-unequal_building_heights-1-v32 | 9/9 | 9/9 | 0.165 | 0.000% |
| g7-vertical-reserved-0-v16 | 9/9 | 9/9 | 0.071 | 1.367% |
| g7-vertical-reserved-0-v24 | 9/9 | 9/9 | 0.147 | 0.000% |
| g7-vertical-reserved-0-v32 | 9/9 | 9/9 | 0.148 | 0.000% |
| g7-vertical-reserved-1-v16 | 9/9 | 9/9 | 0.005 | 0.910% |
| g7-vertical-reserved-1-v24 | 9/9 | 9/9 | 0.162 | 0.000% |
| g7-vertical-reserved-1-v32 | 9/9 | 9/9 | 0.156 | 0.000% |
| g7-vertical-reserved-2-v16 | 8/9 | 8/9 | 0.037 | 1.085% |
| g7-vertical-reserved-2-v24 | 9/9 | 9/9 | 0.087 | 0.196% |
| g7-vertical-reserved-2-v32 | 9/9 | 9/9 | 0.105 | 0.740% |
| g7-vertical-reserved-3-v16 | 9/9 | 9/9 | 0.474 | 3.959% |
| g7-vertical-reserved-3-v24 | 9/9 | 9/9 | 0.482 | 2.614% |
| g7-vertical-reserved-3-v32 | 9/9 | 9/9 | 0.600 | 2.359% |
| g8-reserved-0-v16 | 8/9 | 8/9 | 0.387 | 3.323% |
| g8-reserved-0-v24 | 8/9 | 8/9 | 0.195 | 1.387% |
| g8-reserved-0-v32 | 9/9 | 9/9 | 0.138 | 0.000% |
| g8-reserved-1-v16 | 8/9 | 8/9 | 0.043 | 1.087% |
| g8-reserved-1-v24 | 9/9 | 9/9 | 0.096 | 0.989% |
| g8-reserved-1-v32 | 9/9 | 9/9 | 0.354 | 1.545% |
| g8-reserved-2-v16 | 8/9 | 8/9 | 0.077 | 1.326% |
| g8-reserved-2-v24 | 9/9 | 9/9 | 0.206 | 1.394% |
| g8-reserved-2-v32 | 9/9 | 9/9 | 0.117 | 0.047% |
| g8-reserved-3-v16 | 9/9 | 9/9 | 0.083 | 0.281% |
| g8-reserved-3-v24 | 9/9 | 9/9 | 0.101 | 0.947% |
| g8-reserved-3-v32 | 9/9 | 9/9 | 0.121 | 0.000% |

## Preservation and implementation note

Saved original full ZIP+receipt (including every checkpoint and training array),
final checkpoint,90 evaluated fields/states/proposals/birth traces,45 contexts,
frozen source/config/splits, runtime, comparison and integrity records,30 geometry
plates and overview sheets. The independent128-step runs have exactly matching
64-step birth prefixes, and all final states are finite.

An initial local reviewer adaptation failed before verification or inference
because a broad text substitution changed a hash function name. The failed
script and error note remain in G8-Final-Review-2026-10-04; this corrected review
is in the separate v2 directory. This was a local review-script error, not a
GPU/model failure or scientific retry. Original evidence was never altered.

The verified milestone ZIP is a same-disk archive, not an off-device backup.
Repository synchronization remains pending; resume from this folder's RESUME.json
rather than the checkout's stale D098 record. No Drive operation, further paid
training, push, publication or live-model replacement occurred.
