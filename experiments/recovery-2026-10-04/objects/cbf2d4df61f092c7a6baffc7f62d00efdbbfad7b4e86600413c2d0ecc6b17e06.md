# G9 final review — 2026-10-04

G9 improves connections on the matched fresh sample but fails the frozen
acceptance criteria. Preserve it as an experimental candidate with timing and
coverage regressions. It is not a live replacement. G8 remains the stable
comparison reference; MG7 remains live. Neither G8 nor G9 passed broad admission.

## Verified run

Run20261004T120338Z_60498d5f0838 completed427 updates on TeslaT4 in
469.45 controlled seconds (7m49s), within600.
Peak reserved memory was1450MiB.
Verified all1719 evidence payload hashes, exact unique ZIP membership and package
identity; final checkpoint hash is 58bdb8d3e05fe8f62dea2b93b4929e4e22aaf5a023d87d7e9f631d797f985a4f.
Expected Python3.13.15,Torch2.11.0+cu130,NumPy2.1.3,CUDA13.0,cuDNN92700 matched.
Both complete-payload/state recovery checks at2/3 passed, as did device probes.

All427 row choices/start hashes and saved start/terminal arrays were checked,
with all27328 step accounts and ceilings. Initial numerical payload, all row/start
choices, final sampler and RNG match G8; model weights diverge as expected.
Original inference/pacing source is byte-identical. This controls one seeded
comparison, not multiple-seed reliability or equal wall-clock compute.
Full training trajectories were not independently replayed.

The ranking term was active on7574/27328 steps. Recorded phases:277 seed-access,
10464 advance-access,16561 connected,26 no-teacher-route. All26 no-route events
were below capacity. These are useful diagnostics, not causal proof of the
remaining failures. Loss decomposition, finite fields and phase/account
consistency passed. No optimizer update or GPU retry was launched in this review.

## Frozen numerical evaluation

Final427 only, CPUfloat32, deterministic two-thread execution, firing seed2101,
unchanged64/128 horizons, thresholds and nine families. The45 previously exposed
cases are regression evidence. The12 new cases were frozen before G9 training
and first evaluated here, identically for G8 and G9; their labels were never
training inputs. They are now consumed for future split accounting.

| Cohort/model | Steps | All nine | Median volume error (pp) | Max error (pp) |
|---|---:|---:|---:|---:|
| regression | 64 | 38/45 | 0.267 | 3.591 |
| regression | 128 | 40/45 | 0.150 | 0.180 |
| fresh_reserved | 64 | 9/12 | 0.138 | 1.230 |
| fresh_reserved | 128 | 9/12 | 0.132 | 0.145 |
| baseline_fresh | 64 | 5/12 | 0.123 | 0.259 |
| baseline_fresh | 128 | 5/12 | 0.132 | 0.145 |

Regression and fresh_reserved rows are G9; baseline_fresh is G8 on the SAME new12.

Paired old45: G8=40/45 at both horizons; G9=38/45 at64,40/45 at128.
At64 G9 gains one case and loses three. At128 it gains two and loses two.
Paired fresh12: G8=5/12, G9=9/12 at both horizons, four improvements and
zero all-nine regressions. The three volume requests share each of four scenes;
these are correlated synthetic examples, not12 independent real sites.

All cohort volume-error gates pass at both horizons (median<=2pp,max<=4pp).
However,12/57 G9 cases exceed5% mass growth between64 and128:
10 regression and2 fresh. Maximum growth is29.99%.
G8 passes stability on all57 corresponding cases, combining prior and fresh
evaluations. These numerical stability failures prevent promotion even where
nine families pass. Stability is growth to a hard cap, not independently learned
self-stabilization.

G9 has one persistent coverage failure, g7-vertical-reserved-0-v16, which also
fails access. Every other family except access passes all57 cases at both
horizons. The seven unaffected families are facade,ground,legality,sparsity,
spill,support,thickness. Geometric support is not structural certification.

## Individual gains, regressions and remaining failures

At64, G9 repairs g7-vertical-reserved-2-v16 but loses
g1-unequal_building_heights-1-v16, g7-vertical-reserved-0-v16 and g8-reserved-1-v24.
At128 the first loss recovers, and g8-reserved-2-v16 becomes an additional gain.
Fresh gains at both horizons are g9-reserved-0-v24, g9-reserved-2-v32,
g9-reserved-3-v16 and g9-reserved-3-v24.

The eight persistent failures are:
- g7-vertical-reserved-0-v16
- g8-reserved-0-v16
- g8-reserved-0-v24
- g8-reserved-1-v16
- g8-reserved-1-v24
- g9-reserved-0-v16
- g9-reserved-2-v16
- g9-reserved-2-v24

All eight retain west contact, miss east contact, and exhaust their volume cap
by128. Extra monotone growth after that cannot repair them. At64 some have
unused volume allowance and later fill; this reveals two distinct problems:
late growth and incorrect final allocation. Extending the display to128 does
not solve all access failures or retroactively satisfy the64/128 stability gate.

## Visual review

Inspected all46 scene plates (three requests each), covering G9's57 cases and
G8's12 fresh counterparts at both horizons. Raw voxel volumes retain depth,
stepped surfaces and exterior gaps. Missed vertical endpoints are visible;
the coverage failure remains skewed toward the starting side. The smaller64
volumes and subsequent additions agree with the recorded growth differences.
These are coarse massing experiments, not interior/architectural-quality claims.
Plates show exposed voxel surfaces plus front/plan projections at0.8m spacing,
without smoothing/filling; projections are not interior sections.
A9/9 plate label indicates families only, not overall stability acceptance.

All138 saved evaluation states are finite. In every one of69 independent
64/128 rollout pairs, the first64 birth arrays match exactly.
No rerolls, best-checkpoint choice, threshold tuning or postprocessing occurred.

## Decision and next concrete work

Close G9 as useful but unqualified evidence. Preserve both G8 and G9 and the
paired fresh comparison; do not automatically choose G9 merely because its
fresh access score is better, and do not discard its gains.

Next perform ONE consolidated TRAIN-only comparison using fixed G8/G9 weights:
record proposal logits, firing, eligible progress/other groups and allowance use
on the same45 TRAIN contexts. At fixed saved states, separate gradients from
membership, volume/band and ranking. Check whether ranking lowers otherwise
useful offers below0.5 or whether step quota rejects well-scored progress.
This is a proposed diagnosis, not a conclusion already established here.

Use that evidence to select one design that preserves connection priority and
timely filling. Do not start an unbounded loss-weight sweep, another blind long
run, change horizon/quota just to pass, or add procedural repair silently.
Keep the nine families and volume semantics. Any next package needs its own
frozen protocol and explicit bounded paid allowance. For product work, a clearly
labelled comparison gallery may use these saved outputs; no live model swap
or claim of deployment readiness follows.

## Case ledger

| Case | Families64 | Families128 | Mass growth64–128 |
|---|---:|---:|---:|
| g1-offset_interfaces-y0-v16 | 9/9 | 9/9 | 4.66% |
| g1-offset_interfaces-y0-v24 | 9/9 | 9/9 | 0.00% |
| g1-offset_interfaces-y0-v32 | 9/9 | 9/9 | 0.00% |
| g1-offset_interfaces-y2-v16 | 9/9 | 9/9 | 3.68% |
| g1-offset_interfaces-y2-v24 | 9/9 | 9/9 | 0.00% |
| g1-offset_interfaces-y2-v32 | 9/9 | 9/9 | 0.00% |
| g1-offset_interfaces-y4-v16 | 9/9 | 9/9 | 7.00% |
| g1-offset_interfaces-y4-v24 | 9/9 | 9/9 | 0.00% |
| g1-offset_interfaces-y4-v32 | 9/9 | 9/9 | 0.00% |
| g1-raised_pair-0-v16 | 9/9 | 9/9 | 3.84% |
| g1-raised_pair-0-v24 | 9/9 | 9/9 | 0.00% |
| g1-raised_pair-0-v32 | 9/9 | 9/9 | 0.18% |
| g1-raised_pair-1-v16 | 9/9 | 9/9 | 2.95% |
| g1-raised_pair-1-v24 | 9/9 | 9/9 | 0.24% |
| g1-raised_pair-1-v32 | 9/9 | 9/9 | 0.00% |
| g1-unequal_building_heights-0-v16 | 9/9 | 9/9 | 4.72% |
| g1-unequal_building_heights-0-v24 | 9/9 | 9/9 | 0.00% |
| g1-unequal_building_heights-0-v32 | 9/9 | 9/9 | 0.00% |
| g1-unequal_building_heights-1-v16 | 8/9 | 9/9 | 9.47% |
| g1-unequal_building_heights-1-v24 | 9/9 | 9/9 | 0.00% |
| g1-unequal_building_heights-1-v32 | 9/9 | 9/9 | 0.00% |
| g7-vertical-reserved-0-v16 | 7/9 | 7/9 | 8.01% |
| g7-vertical-reserved-0-v24 | 9/9 | 9/9 | 0.00% |
| g7-vertical-reserved-0-v32 | 9/9 | 9/9 | 0.00% |
| g7-vertical-reserved-1-v16 | 9/9 | 9/9 | 3.50% |
| g7-vertical-reserved-1-v24 | 9/9 | 9/9 | 0.00% |
| g7-vertical-reserved-1-v32 | 9/9 | 9/9 | 0.00% |
| g7-vertical-reserved-2-v16 | 9/9 | 9/9 | 3.74% |
| g7-vertical-reserved-2-v24 | 9/9 | 9/9 | 2.13% |
| g7-vertical-reserved-2-v32 | 9/9 | 9/9 | 1.24% |
| g7-vertical-reserved-3-v16 | 9/9 | 9/9 | 5.57% |
| g7-vertical-reserved-3-v24 | 9/9 | 9/9 | 4.58% |
| g7-vertical-reserved-3-v32 | 9/9 | 9/9 | 3.45% |
| g8-reserved-0-v16 | 8/9 | 8/9 | 12.38% |
| g8-reserved-0-v24 | 8/9 | 8/9 | 10.75% |
| g8-reserved-0-v32 | 9/9 | 9/9 | 0.00% |
| g8-reserved-1-v16 | 8/9 | 8/9 | 29.99% |
| g8-reserved-1-v24 | 8/9 | 8/9 | 6.62% |
| g8-reserved-1-v32 | 9/9 | 9/9 | 1.85% |
| g8-reserved-2-v16 | 8/9 | 9/9 | 14.44% |
| g8-reserved-2-v24 | 9/9 | 9/9 | 2.89% |
| g8-reserved-2-v32 | 9/9 | 9/9 | 1.14% |
| g8-reserved-3-v16 | 9/9 | 9/9 | 6.69% |
| g8-reserved-3-v24 | 9/9 | 9/9 | 1.65% |
| g8-reserved-3-v32 | 9/9 | 9/9 | 0.00% |
| g9-reserved-0-v16 | 8/9 | 8/9 | 9.24% |
| g9-reserved-0-v24 | 9/9 | 9/9 | 0.27% |
| g9-reserved-0-v32 | 9/9 | 9/9 | 0.00% |
| g9-reserved-1-v16 | 9/9 | 9/9 | 0.00% |
| g9-reserved-1-v24 | 9/9 | 9/9 | 0.07% |
| g9-reserved-1-v32 | 9/9 | 9/9 | 0.00% |
| g9-reserved-2-v16 | 8/9 | 8/9 | 5.68% |
| g9-reserved-2-v24 | 8/9 | 8/9 | 3.02% |
| g9-reserved-2-v32 | 9/9 | 9/9 | 1.97% |
| g9-reserved-3-v16 | 9/9 | 9/9 | 0.00% |
| g9-reserved-3-v24 | 9/9 | 9/9 | 0.00% |
| g9-reserved-3-v32 | 9/9 | 9/9 | 0.27% |

## Preservation and resume

Original full returnedZIP+receipt, all checkpoints in thatZIP, source/config,
split hashes,57 G9 contexts plus12 baseline copies,138 raw evaluations and
birth traces, individual metrics,46 visual plates and12 overview sheets are
saved here and in a verified archive. Archive copies are on the same disk,
not an off-device backup. Read this folder's RESUME.json next.
Repository synchronization remains pending; checkout resume is stale.
No Drive operation, push, publication, new paid run or live-model change occurred.
