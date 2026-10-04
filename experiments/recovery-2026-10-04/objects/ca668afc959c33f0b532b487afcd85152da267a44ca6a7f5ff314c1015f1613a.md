# G10 final review — 2026-10-04

G10 passes the volume-error and stability gates but fails overall acceptance:
45/57 regression cases and 6/12 new cases pass all nine families at both horizons.
The one-sided ranking experiment resolves the measured timing problem in this
evaluation, while leaving connection and coverage failures. It is experimental
evidence, not a live replacement. Preserve G8 and G9 for comparison; MG7 remains live.

## Verified experiment

Run 20261004T133117Z_60e4ff105497 completed 427 updates on Tesla T4 in
475.55 controlled seconds (about 7m56s), within 600.
Peak reserved memory: 1448 MiB.
Final checkpoint SHA-256: 8e4ac1f13ff57b2c644cd2bd6fd27c40b25eca7062ec729f002969171a122bb2.

The original returned ZIP and receipt are retained. All 1,719 payload hashes,
unique membership and package identity were verified. Expected Python 3.13.15,
Torch 2.11.0+cu130, NumPy 2.1.3, CUDA 13.0 and cuDNN 92700 matched.
Complete-payload/state recovery checks at updates 2 and 3 passed, along with
12 original and 8 paced device probes. All 427 row choices/start hashes and
saved start arrays were checked, including 27,328 step accounts and ceilings.
The initial numerical payload, all row/start choices and terminal sampler/RNG
states match G9. Pacing/inference source remains byte-identical.
This is one seeded comparison, not independent replication; full training
trajectories were not independently replayed.

Only the ranking auxiliary's gradient through the other-candidate reference
was stopped. Its forward value and advancing-candidate gradient are preserved.
Training phases: 277 seed-access, 15,036 connected, 11,995 advance-access and
20 no-teacher-route steps; all no-route events were below capacity.
Ranking was active on 8,663 steps. These phase counts are descriptive.
The comparison does not prove that all resulting model behavior is caused by
direct gradient suppression: shared parameters and subsequent trajectories interact.

## Frozen evaluation

Final checkpoint 427 only; deterministic CPU float32, two threads, firing seed
2101, horizons 64 and 128, unchanged thresholds and nine constraint families.
The 57 old cases are regression evidence. The four new scenes with three volume
requests each were frozen before training and evaluated identically for G10,
G9 and G8. They are now consumed and must become regression evidence in future work.

| Cohort/model | Steps | All nine | Median volume error (pp) | Max error (pp) |
|---|---:|---:|---:|---:|
| regression | 64 | 45/57 | 0.151 | 0.569 |
| regression | 128 | 45/57 | 0.143 | 0.180 |
| fresh_reserved | 64 | 6/12 | 0.111 | 0.342 |
| fresh_reserved | 128 | 6/12 | 0.121 | 0.141 |
| baseline_g8_fresh | 64 | 4/12 | 0.111 | 0.598 |
| baseline_g8_fresh | 128 | 4/12 | 0.121 | 0.141 |
| baseline_g9_fresh | 64 | 4/12 | 0.590 | 3.598 |
| baseline_g9_fresh | 128 | 4/12 | 0.121 | 0.141 |

Regression and fresh_reserved refer to G10; baseline cohorts name their model.
All volume-error gates pass. G10 passes the 5% growth limit on all 69 cases;
maximum growth from 64 to 128 is 4.597%.
This is stability under a hard volume cap, not independently learned self-stabilization.

On the same old 57 cases, G9 passed 47 at 64 and 49 at 128, versus G10's 45.
On the same new 12, G9 and G8 each pass 4, versus G10's 6 at both horizons.
G8 passes fresh stability; G9 does not. G10 therefore improves the new sample
without dominating either reference across the full evaluation.
Three requests share each scene, and only one training seed was used; this is
not evidence of broad real-site generalization.

## Failure diagnosis and interpretation

There are 18 persistent failures: 15 access failures and three disjoint coverage
failures. The coverage cases are g1-unequal_building_heights-1-v16,
g7-vertical-reserved-2-v16 and g8-reserved-3-v16.
Every persistent failure exhausts its volume allowance by 128.
Additional monotone growth cannot repair those states once the cap is reached.

The seven other families pass all G10 cases at both horizons: facade, ground,
legality, sparsity, spill, support and thickness. Geometric support is not
structural certification. Some gates follow partly from hard admission rules.

The evidence supports a remaining allocation problem: finite volume has been
irreversibly committed without satisfying all spatial requirements. It does
not establish which replacement mechanism will solve that problem.
Restoring fill timing alone is insufficient.

## Visual and numerical audit

All 62 scene plates were inspected through 16 overview sheets: G10's 69 cases
and both references on the new 12, at both horizons. Plates contain raw exposed
voxel surfaces and front/plan projections at 0.8 m spacing, without smoothing
or filling. The projections are not interior sections.
Stepped volumes have depth, but some upward connections terminate below their
destination; coverage failures remain unevenly distributed.
These are massing experiments, not claims of finished architectural spaces.
A 9/9 plate label reports families, not the overall acceptance decision.

All 186 saved states are finite. All 93 independent 64/128 rollout pairs have
identical first-64 birth arrays. No rerolls, checkpoint selection, altered
thresholds or postprocessing were used.

## Decision and next step

Close G10 as a useful but unaccepted experiment. Retain its timing improvement
and fresh-case gains alongside its regressions. Do not promote it or launch
another loss-weight variant automatically.

Next prepare one architecture and budget-allocation decision using TRAIN data:
compare (a) learned reversible occupancy updates, allowing misplaced volume
to be relocated, with (b) explicitly labelled hybrid, budget-aware completion
that reserves capacity for remaining connections before admitting bulk growth.
Assess locality, connectivity/thickness preservation, runtime, training burden
and fidelity to the project's volume concept. Neither proposal is validated yet.

Choose one bounded prototype after that review. Use existing TRAIN failures
and an optimistic reachability/budget feasibility calculation to reject
infeasible designs before paid training. Keep the same nine families and
occupancy-as-volume semantics. Do not add room/shelter targets, silently apply
procedural repair, extend horizons merely to pass, or begin an unbounded sweep.
Any new experiment needs a frozen protocol and genuinely new reserved scenes;
the current 69 cases are regression cases. No paid run is authorized here.

## Paired case changes

### regression_G9, 64 steps

Baseline 47; G10 45. Gains: g8-reserved-1-v24, g8-reserved-2-v16, g9-reserved-0-v16, g9-reserved-2-v24. Losses: g7-vertical-reserved-2-v16, g8-reserved-0-v32, g8-reserved-3-v16, g9-reserved-0-v24, g9-reserved-0-v32, g9-reserved-3-v16.

### regression_G9, 128 steps

Baseline 49; G10 45. Gains: g8-reserved-1-v24, g9-reserved-0-v16, g9-reserved-2-v24. Losses: g1-unequal_building_heights-1-v16, g7-vertical-reserved-2-v16, g8-reserved-0-v32, g8-reserved-3-v16, g9-reserved-0-v24, g9-reserved-0-v32, g9-reserved-3-v16.

### fresh_reserved_G9, 64 steps

Baseline 4; G10 6. Gains: g10-reserved-1-v32, g10-reserved-2-v24. Losses: none.

### fresh_reserved_G9, 128 steps

Baseline 4; G10 6. Gains: g10-reserved-1-v32, g10-reserved-2-v24. Losses: none.

### fresh_reserved_G8, 64 steps

Baseline 4; G10 6. Gains: g10-reserved-2-v24, g10-reserved-2-v32. Losses: none.

### fresh_reserved_G8, 128 steps

Baseline 4; G10 6. Gains: g10-reserved-2-v24, g10-reserved-2-v32. Losses: none.

## Full G10 case ledger

| Case | Families 64 | Families 128 | Growth 64–128 | Failure at 128 |
|---|---:|---:|---:|---|
| g1-offset_interfaces-y0-v16 | 9/9 | 9/9 | 0.57% | none |
| g1-offset_interfaces-y0-v24 | 9/9 | 9/9 | 0.00% | none |
| g1-offset_interfaces-y0-v32 | 9/9 | 9/9 | 0.00% | none |
| g1-offset_interfaces-y2-v16 | 9/9 | 9/9 | 0.69% | none |
| g1-offset_interfaces-y2-v24 | 9/9 | 9/9 | 0.00% | none |
| g1-offset_interfaces-y2-v32 | 9/9 | 9/9 | 0.00% | none |
| g1-offset_interfaces-y4-v16 | 9/9 | 9/9 | 0.00% | none |
| g1-offset_interfaces-y4-v24 | 9/9 | 9/9 | 0.00% | none |
| g1-offset_interfaces-y4-v32 | 9/9 | 9/9 | 0.00% | none |
| g1-raised_pair-0-v16 | 9/9 | 9/9 | 2.20% | none |
| g1-raised_pair-0-v24 | 9/9 | 9/9 | 0.00% | none |
| g1-raised_pair-0-v32 | 9/9 | 9/9 | 0.00% | none |
| g1-raised_pair-1-v16 | 9/9 | 9/9 | 1.95% | none |
| g1-raised_pair-1-v24 | 9/9 | 9/9 | 0.00% | none |
| g1-raised_pair-1-v32 | 9/9 | 9/9 | 0.00% | none |
| g1-unequal_building_heights-0-v16 | 9/9 | 9/9 | 0.00% | none |
| g1-unequal_building_heights-0-v24 | 9/9 | 9/9 | 0.00% | none |
| g1-unequal_building_heights-0-v32 | 9/9 | 9/9 | 0.00% | none |
| g1-unequal_building_heights-1-v16 | 8/9 | 8/9 | 3.43% | coverage |
| g1-unequal_building_heights-1-v24 | 9/9 | 9/9 | 0.00% | none |
| g1-unequal_building_heights-1-v32 | 9/9 | 9/9 | 0.00% | none |
| g7-vertical-reserved-0-v16 | 8/9 | 8/9 | 2.77% | access |
| g7-vertical-reserved-0-v24 | 9/9 | 9/9 | 0.00% | none |
| g7-vertical-reserved-0-v32 | 9/9 | 9/9 | 0.00% | none |
| g7-vertical-reserved-1-v16 | 9/9 | 9/9 | 1.37% | none |
| g7-vertical-reserved-1-v24 | 9/9 | 9/9 | 0.00% | none |
| g7-vertical-reserved-1-v32 | 9/9 | 9/9 | 0.00% | none |
| g7-vertical-reserved-2-v16 | 8/9 | 8/9 | 1.69% | coverage |
| g7-vertical-reserved-2-v24 | 9/9 | 9/9 | 1.25% | none |
| g7-vertical-reserved-2-v32 | 9/9 | 9/9 | 0.54% | none |
| g7-vertical-reserved-3-v16 | 9/9 | 9/9 | 4.60% | none |
| g7-vertical-reserved-3-v24 | 9/9 | 9/9 | 1.93% | none |
| g7-vertical-reserved-3-v32 | 9/9 | 9/9 | 2.26% | none |
| g8-reserved-0-v16 | 8/9 | 8/9 | 2.19% | access |
| g8-reserved-0-v24 | 8/9 | 8/9 | 1.12% | access |
| g8-reserved-0-v32 | 8/9 | 8/9 | 0.00% | access |
| g8-reserved-1-v16 | 8/9 | 8/9 | 1.99% | access |
| g8-reserved-1-v24 | 9/9 | 9/9 | 0.92% | none |
| g8-reserved-1-v32 | 9/9 | 9/9 | 1.44% | none |
| g8-reserved-2-v16 | 9/9 | 9/9 | 1.04% | none |
| g8-reserved-2-v24 | 9/9 | 9/9 | 1.20% | none |
| g8-reserved-2-v32 | 9/9 | 9/9 | 0.33% | none |
| g8-reserved-3-v16 | 8/9 | 8/9 | 0.56% | coverage |
| g8-reserved-3-v24 | 9/9 | 9/9 | 0.63% | none |
| g8-reserved-3-v32 | 9/9 | 9/9 | 0.00% | none |
| g9-reserved-0-v16 | 9/9 | 9/9 | 0.00% | none |
| g9-reserved-0-v24 | 8/9 | 8/9 | 0.48% | access |
| g9-reserved-0-v32 | 8/9 | 8/9 | 0.00% | access |
| g9-reserved-1-v16 | 9/9 | 9/9 | 0.00% | none |
| g9-reserved-1-v24 | 9/9 | 9/9 | 0.00% | none |
| g9-reserved-1-v32 | 9/9 | 9/9 | 0.00% | none |
| g9-reserved-2-v16 | 8/9 | 8/9 | 0.00% | access |
| g9-reserved-2-v24 | 9/9 | 9/9 | 0.00% | none |
| g9-reserved-2-v32 | 9/9 | 9/9 | 1.18% | none |
| g9-reserved-3-v16 | 8/9 | 8/9 | 0.00% | access |
| g9-reserved-3-v24 | 9/9 | 9/9 | 0.00% | none |
| g9-reserved-3-v32 | 9/9 | 9/9 | 0.09% | none |
| g10-reserved-0-v16 | 8/9 | 8/9 | 1.98% | access |
| g10-reserved-0-v24 | 8/9 | 8/9 | 0.92% | access |
| g10-reserved-0-v32 | 8/9 | 8/9 | 0.00% | access |
| g10-reserved-1-v16 | 8/9 | 8/9 | 2.09% | access |
| g10-reserved-1-v24 | 8/9 | 8/9 | 0.99% | access |
| g10-reserved-1-v32 | 9/9 | 9/9 | 1.49% | none |
| g10-reserved-2-v16 | 8/9 | 8/9 | 0.09% | access |
| g10-reserved-2-v24 | 9/9 | 9/9 | 0.00% | none |
| g10-reserved-2-v32 | 9/9 | 9/9 | 0.00% | none |
| g10-reserved-3-v16 | 9/9 | 9/9 | 0.43% | none |
| g10-reserved-3-v24 | 9/9 | 9/9 | 0.00% | none |
| g10-reserved-3-v32 | 9/9 | 9/9 | 0.00% | none |

## Preservation and resume

This directory retains the original full returned ZIP and receipt, all returned
checkpoints inside that ZIP, extracted final checkpoint, training verification,
frozen sources/configuration/splits, 93 contexts, 186 raw observations and birth
traces, metrics, comparisons, 62 plates and 16 overview sheets.
A manifest and verified archive preserve this milestone. The archive is on
the same disk and is not an off-device backup.
Read RESUME.json in this directory to continue. Repository synchronization
remains pending; its older resume record is stale. The original next-phase
report remains untouched. No Drive action, publication, push, paid retry or
live model change occurred.
