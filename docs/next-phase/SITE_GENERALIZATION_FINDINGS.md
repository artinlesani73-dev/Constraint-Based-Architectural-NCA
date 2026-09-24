# MG4: new-site stress evaluation

2026-09-24. The unchanged procedural generator passes143/144 nonpartition
candidates (99.31%). All36 deliberately blocked candidates fail. Full denominator:
143/180 pass (79.44%). Every nonpartition candidate reaches its requested volume.
The predeclared144/144 admission gate is **not met**; no threshold was changed.

Benchmark 20260924T163831Z_b4e28646a3e0; regression 20260924T163521Z_bd0b9d7571b7.
All180 planned cases completed, with no execution errors, generator timeouts,
study time-cap breach or observed resident-memory threshold breach. No reroll,
scene replacement, new constraint family or generator/evaluator edit occurred.

## Population and outcomes

Twenty previously untested designed sites: sixteen nonpartition variations and
four complete partition controls. Three requests16/24/32%, three new seeds3/4/5.
All are32 cubed at0.8m/cell. Static geometry is distinct from the five earlier
sites and within this set. Single-variable groups and combined variations are
predeclared, but this is not an exhaustive factorial design or representative
random sample. These inspected sites are now development evidence, not a fresh
blind population for later tuning. Nonpartition was never assumed feasible.

| Site | Input group | Nine-family passes |
|---|---|---:|
|gap_12|gap|9/9|
|gap_22|gap|9/9|
|gap_8|gap|9/9|
|low_pair|height|9/9|
|low_west|height|9/9|
|low_east|height|9/9|
|lateral_span|interfaces|9/9|
|vertical_span|interfaces|9/9|
|diagonal_rise|interfaces|9/9|
|diagonal_reverse|interfaces|9/9|
|long_obstacle|obstacles|9/9|
|upper_obstacle|obstacles|9/9|
|twin_obstacles|obstacles|9/9|
|combined_tall_east|combined|9/9|
|combined_low_west|combined|9/9|
|combined_reverse|combined|8/9|
|blocked_narrow|blocked|0/9|
|blocked_wide|blocked|0/9|
|blocked_vertical|blocked|0/9|
|blocked_diagonal|blocked|0/9|

The144 nonpartition requests overshoot by0–8 cells, below the unchanged27-cell
block bound. Every nonpartition output is100% cube-qualified bulk. They all pass
access, facade, ground, legality, sparsity, spill, support and thickness. Only
coverage fails once. All36 blocked results terminate no_cube_route and retain
empty outputs and failed checks. Generator routing failure generally is not a
universal infeasibility proof; these four control inputs explicitly contain a
complete separating wall.

## The failure: distribution rather than total volume

combined_reverse__v16__s5 combines a14-cell gap, building heights30/22 cells,
widely offset interfaces (west x9,y23,z18; east x21,y8,z8), and an obstacle
x[14,17),y[10,19),z[6,25). The context's necessary checks pass. Final count1507
versus requested1503, all1507 cells qualified as bulk, facade225/1507=14.930325%.

MT1 coverage splits the fixed legal domain's X span into three parts, and each
requires at least8% qualified bulk. It does not divide the candidate's own extent.

| Fixed third | Domain cells | Required bulk cells | Actual bulk | Coverage |
|---|---:|---:|---:|---:|
| West |3544|284|266|7.505643% — fail|
| Middle |2303|185|303|13.156752%|
| East |3544|284|938|26.467269%|

The west deficit is18 cells. That is arithmetic, not a valid single-cell repair
prescription: new volume must preserve cube scale, contact, connectivity and the
request bound. The initial297-cell route already distributes45/42/210 cells
west/middle/east. Growth adds221/261/728. Its last accepted cube, origin(z,y,x)
(19,18,19), adds7 cells without contact, taking1500 to1507 and triggering the
total-count stop. There are2802 proposal evaluations, including2223 rejected
evaluations over90 distinct origins. Rejection counts include rechecks.

The algorithm uses seeded radial/contact priority and a global facade admission
check; it has no coverage-deficit priority. The trace establishes an underfilled
third at termination. It does not isolate facade deferral as the causal factor
or prove another ordering always succeeds. Seeds3/4 at the same16% request pass:
west316/3544=8.916479% and308/3544=8.690745%. Their contact fractions are14.970060%
and14.950166%. The current recipe can succeed in this site; it is seed-sensitive.
No replacement seed was substituted for the failed member.

The X-third coverage proxy remains provisional and axis-dependent. This result
is not evidence of architectural unfitness. Preserve the accepted definition for
the present comparison; any conceptual metric revision needs a separately
declared study rather than retroactively converting this failure into success.

## Variation and resource observations

There are48 nonpartition site/request diversity groups. Every valid field is
distinct within its group.47 groups contain3 valid seeds; the failed group has2.
Mean pairwise Jaccard distances range0.151876–0.620989. Twelve blocked groups have
zero valid fields and no diversity estimate. Reported differences measure voxel
variation, not design quality. Different seeds/sites prevent treating MG4/MG3
pass rates or timing as matched paired improvements.

| Measurement | Observed value |
|---|---:|
| Generation median /95th percentile /max |0.2176 /0.6048 /2.2542 seconds|
| Generation total,180 candidates |54.3154 seconds|
| Evaluation median /95th percentile /max |0.2033 /0.3741 /0.4550 seconds|
| Evaluation total |42.3080 seconds|
| Context setup total,20 sites |0.5332 seconds|
| Full study before final summary packaging |118.6887 seconds|
| Sampled resident peak,median /max |214.527 /309.320 MiB|
| Maximum sampled increase from case baseline |29.461 MiB|
| Highest recorded process-lifetime peak |452.629 MiB|

Slowest generation: combined_reverse__v32__s4. The10ms sampler requests a polling
interval; Python scheduling/GIL can delay it. Start/end samples are included and
actual sample counts saved. Brief native peaks can be missed. Resident memory
includes Python/PyTorch and retained allocator/cache state, not just the generator.
The lifetime high water also includes earlier setup/serialization and cannot be
assigned to a case. Small baseline deltas do not imply negligible allocations.
These single local observations establish headroom for this32-cubed study only,
not an estimate of48/64-cubed capacity or paid Colab performance. Caps were
cooperative case-boundary checks, not OS resource limits. Regression and replay
verification ran separately from the measured study; no benchmark/test overlap.

## Verification and preserved evidence

324 regression tests pass, no failures/errors/skips, smoke exit0. Four new checks
cover frozen scene uniqueness/partition placement, metadata-independent identity,
native memory sampling/cleanup and valid-only diversity edge cases. Focused4pass
in0.063s; full suite159.329s, complete command164.5098s. Original model smoke remains
an execution check, not new learned-model evidence.

Independent verification rebuilds20 contexts/masks, exactly replays180 generator
outputs/routes/metadata excluding wall time, recomputes180 binary scores and180
bulk masks, audits718,654 individual growth decisions by actual
cube union, recomputes all60 diversity groups and independently rejects the gate.
Both experiment/regression snapshots match154 current Python files; all snapshot
member hashes and run manifests verify. Frozen core hashes are unchanged.
See MG4-verification.json and MG4-coverage-diagnosis.json in experiments/reports.
Replay uses byte-exact frozen sources, including line endings. The source ZIP and
raw repository archive retain those bytes; a fresh Git checkout may normalize
them. Restore the retained bytes rather than changing the frozen hashes.

Run folders retain each scene/domain/mask, complete raw occupied/route/bulk field,
growth trace, all metrics, memory samples summary, wall/CPU time, config, source,
protocol and immediate per-case events.180/180 records present, no pending case.
The study remains completed with a negative admission result; it is not labeled
an execution failure. One read-only inline diagnosis command had a parenthesis
syntax error; corrected helper read the same records without regenerating them.
No live Studio, old gallery, original model or private report change was needed.

## Decision

Keep the current five-site live Studio available with its existing scope. The
broader-scale admission gate failed and stays failed. Do not claim arbitrary-site
reliability, enable custom sites or start larger-grid generation on this evidence.
The next bounded proposal is COVERAGE_GROWTH_NEXT_PLAN.md: change growth priority
within the existing coverage family, retain the facade budget and request bound,
then compare against all preserved positives and this negative case. This is a
focused procedural hypothesis, not a reason to redesign/train the whole NCA yet.
No paid computation, Drive operation, push or publication occurred. Private
reports stay ignored/unchanged. Verified local archive is same-disk only.
