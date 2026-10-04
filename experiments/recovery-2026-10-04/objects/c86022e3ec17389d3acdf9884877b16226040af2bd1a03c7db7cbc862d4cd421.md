# G7 review — vertical training diversity — 2026-10-04

**The GPU run is valid, but G7 fails the frozen acceptance gates. Do not promote it.**
It fixes one G6 connection failure, yet size accuracy at64 steps and64-to128
stability regress substantially. MG7 remains live; G6 remains the prior research
reference and is itself not generally qualified for deployment.

## Verified evidence

Run20261004T103602Z_1da578905a51 completed256 updates in
189.902 controlled seconds
(183.283 worker seconds), within600.
Peak GPU reservation was1446MiB.
Verified original ZIP SHA256, all1035 payload hashes, package manifest and
runtime/identity, device probes and both full-payload/state recovery checks.
Verified256 row selections and teacher starts, all16384 admission accounts and
temporary caps, finite saved training states and final legal connected cube unions.
The exact original evidence ZIP retains every checkpoint and training array;
the review uses only the frozen final256 checkpoint.

Checkpoint SHA256: `edd2369b8ecdb11fa5a6131bf8472eba4a41664b2e4a532a1c89c7d9dde6fe91`.
All45 reconstructed TRAIN contexts match package input bytes and seeds. Initial
parameters and paced model/loss source match G6 exactly. Data distribution and
per-row exposure differ. All33 independently executed128-step rollouts match
their64-step birth-mask prefixes exactly; all saved hidden states are finite.
These checks establish engineering integrity, not model quality.

## Frozen results, separate cohorts

| Cohort | Steps | All nine families | Median volume error (pp) | Maximum volume error (pp) |
|---|---:|---:|---:|---:|
| regression | 64 | 19/21 | 3.759 | 9.372 |
| regression | 128 | 20/21 | 0.165 | 0.180 |
| fresh_reserved | 64 | 9/12 | 3.631 | 8.125 |
| fresh_reserved | 128 | 10/12 | 0.144 | 0.162 |

The21 regression requests contain9 reused development requests and12 consumed
G6 reserved requests. The12 fresh reserved requests are first-use G7 synthetic
variations. They are now exposed evidence and must not be called untouched in
future tuning. No teachers, routes or labels were used for these inference calls.
No checkpoint, threshold, seed, quota or horizon was selected after inspection.

Access is the only failing family; each of the other eight passes in every case
at both horizons. However, all-nine validity is only one part of the acceptance
gate. The separate requested-volume and stability requirements matter: at64,
both cohorts exceed the median2pp and maximum4pp error limits. At128 those size
errors pass, but only2/21 regression and0/12
fresh cases stay within5% mass change. Maximum mass growth is42.20%.
The visual plates' PASS label denotes **all nine families only**, not complete
acceptance. See this report and result.json for the separate size/stability gates.

## Comparison with G6

On the same21 legacy requests, G6 passed19/21 at both horizons. G7 passes19/21
at64 and20/21 at128. `g1-unequal_building_heights-1-v24` now connects at both
horizons. Its16% request still misses the east interface. A new64-step miss
appears in `g1-offset_interfaces-y4-v16`, which connects by128.
G6 satisfied size/stability limits on these legacy cohorts; G7 does not.
No G6 result on the new G7 reserved scenes is claimed.

Fresh16% requests0 and1 miss the east interface even at128. Fresh16% request2
misses at64 but connects by128; request3 passes both. All fresh24%/32% requests
pass the nine-family conjunction, but their substantial later growth still fails
stability. Three access misses remain at128 across the two cohorts, and all
three have exhausted the volume cap. More monotone growth steps cannot fix
their occupancy. Other failures are delayed growth rather than saturation.

## Visual interpretation

Inspected all33 outputs at both horizons through six overview sheets of the
22 full scene plates. They remain genuine three-dimensional voxel masses with
coarse terraces, protrusions and open exterior space, rather than single-cell
paths. Higher-volume outputs grow noticeably wider and taller after64; the
extra mass is visible, not merely a numerical artifact. The low-volume failing
upward scenes stop below their east connection; the downward case stops above
it. Whole-cube support does not make these finished architectural designs.
Projection views show occupied extents, not interior sections or habitable rooms.

## What this experiment establishes

Broader training data alone, at the same256-update budget, is insufficient for
the frozen target. This does not prove the data change was wrong. Increasing
the dataset from27 to45 lowers per-example exposure to5-6 visits and changes
the sequence of gradients. One seed cannot disentangle diversity, exposure and
optimization. Do not infer that512 updates, a new loss or a larger grid will
necessarily solve this result.

The saved trajectories expose two different issues: delayed filling within64
steps, and allocation that exhausts the cap without completing a connection.
Extending the review horizon would conceal the first problem while leaving the
second. The prescribed64/128 results and thresholds remain unchanged.
Thickness, global budgets and eventual saturation stability are partly enforced
by the hybrid admission algorithm; these are not independently learned behavior.

## Decision and next step

Keep this run as a failed acceptance result with useful partial gains. Do not
replace G6 or MG7, change the acceptance horizon, or launch another paid job.
Next do one consolidated local TRAIN-only diagnosis using existing G6/G7 weights
and training traces: examine seed-start delays, eligible proposal scores and
unused per-step allowance, and compare near-connection allocation before cap.
Separate reduced training exposure from the known lack of temporal ordering in
teacher membership labels before choosing one intervention. Avoid tuning on the
now-consumed reserved cases. Prepare a concrete new frozen proposal only after
that diagnosis; any paid allowance requires explicit approval.

## Individual case ledger

Errors are absolute requested fraction differences in percentage points.

| Case | Nine families at64 | Nine families at128 | Error at64 (pp) | Mass change |
|---|---|---|---:|---:|
| g1-offset_interfaces-y0-v16 | Pass | Pass | 2.633 | 20.83% |
| g1-offset_interfaces-y0-v24 | Pass | Pass | 4.144 | 21.63% |
| g1-offset_interfaces-y0-v32 | Pass | Pass | 8.327 | 35.83% |
| g1-offset_interfaces-y2-v16 | Pass | Pass | 2.622 | 20.72% |
| g1-offset_interfaces-y2-v24 | Pass | Pass | 6.132 | 35.16% |
| g1-offset_interfaces-y2-v32 | Pass | Pass | 9.254 | 41.35% |
| g1-offset_interfaces-y4-v16 | Access fail | Pass | 2.693 | 21.48% |
| g1-offset_interfaces-y4-v24 | Pass | Pass | 3.584 | 18.36% |
| g1-offset_interfaces-y4-v32 | Pass | Pass | 8.596 | 37.43% |
| g1-raised_pair-0-v16 | Pass | Pass | 0.259 | 2.70% |
| g1-raised_pair-0-v24 | Pass | Pass | 3.090 | 15.59% |
| g1-raised_pair-0-v32 | Pass | Pass | 3.682 | 13.56% |
| g1-raised_pair-1-v16 | Pass | Pass | 0.722 | 5.81% |
| g1-raised_pair-1-v24 | Pass | Pass | 3.186 | 16.13% |
| g1-raised_pair-1-v32 | Pass | Pass | 5.167 | 19.84% |
| g1-unequal_building_heights-0-v16 | Pass | Pass | 0.266 | 2.84% |
| g1-unequal_building_heights-0-v24 | Pass | Pass | 5.306 | 29.28% |
| g1-unequal_building_heights-0-v32 | Pass | Pass | 9.372 | 42.20% |
| g1-unequal_building_heights-1-v16 | Access fail | Access fail | 3.759 | 32.08% |
| g1-unequal_building_heights-1-v24 | Pass | Pass | 4.326 | 22.78% |
| g1-unequal_building_heights-1-v32 | Pass | Pass | 8.762 | 38.42% |
| g7-vertical-reserved-0-v16 | Access fail | Access fail | 3.446 | 28.61% |
| g7-vertical-reserved-0-v24 | Pass | Pass | 2.683 | 13.28% |
| g7-vertical-reserved-0-v32 | Pass | Pass | 8.125 | 34.65% |
| g7-vertical-reserved-1-v16 | Access fail | Access fail | 1.397 | 10.60% |
| g7-vertical-reserved-1-v24 | Pass | Pass | 2.314 | 11.42% |
| g7-vertical-reserved-1-v32 | Pass | Pass | 5.307 | 20.46% |
| g7-vertical-reserved-2-v16 | Access fail | Pass | 1.627 | 12.27% |
| g7-vertical-reserved-2-v24 | Pass | Pass | 3.817 | 19.58% |
| g7-vertical-reserved-2-v32 | Pass | Pass | 5.378 | 20.70% |
| g7-vertical-reserved-3-v16 | Pass | Pass | 2.523 | 19.77% |
| g7-vertical-reserved-3-v24 | Pass | Pass | 4.281 | 22.38% |
| g7-vertical-reserved-3-v32 | Pass | Pass | 6.354 | 25.32% |

## Preservation

Original ZIP+receipt, final checkpoint, all66 reviewed output arrays,33 contexts,
source/config/split snapshots, package provenance, exact metrics, timing counts,
comparison audit and visual plates are retained. Local verified milestone ZIP
is a same-disk archive, not an off-device backup. No Drive, new training, push,
publication or live changes occurred. Repository synchronization remains pending;
resume from this folder's RESUME.json instead of the checkout's stale D098 file.
