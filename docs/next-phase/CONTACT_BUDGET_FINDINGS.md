# MG3: budget-checked mass growth

2026-09-24. Benchmark 20260924T140750Z_5d36c2bf3bdc; regression
20260924T140546Z_7d07393758df. Protocol and admission rule were frozen first.
The development admission gate passes. No reroll, parameter tuning, relaxed
threshold, execution failure or paid computation was needed.

## Results

| Context | MG1 | MG2 | MG3 |
|---|---:|---:|---:|
| Aligned |9/9|9/9|9/9|
| Wider gap |9/9|9/9|9/9|
| Offset interfaces |9/9|9/9|9/9|
| Partial obstruction |0/9|7/9|9/9|
| Blocked gap |0/9|0/9|0/9|
| **All cases** |**27/45**|**34/45**|**36/45**|

All 36 nonblocked requests pass the same nine MT1 families, reach the requested
count with 0–8 extra cells, and have 100% cube-qualified bulk. Nine blocked
no-route outputs remain failures in the full denominator. This algorithm's
routing failure does not prove universal geometric infeasibility.
Only two final fields changed relative to MG2; all 45 initial routes are identical.
There are no regressions against either MG1 or MG2.

| Repaired partial-obstruction case | MG2 contact | MG3 contact | New total cells | Added / removed vs MG2 |
|---|---:|---:|---:|---:|
|24%, seed2|123/819 =15.018315%|122/819 =14.896215%|819|1 /1|
|32%, seed2|181/1093 =16.559927%|162/1092 =14.835165%|1092|19 /20|

Both now meet the unchanged 15% facade limit, with zero request-count error.
The comparison shows independently regenerated outputs, not deletion-based repair.

## Mechanism and limits

New separate budgeted_contact_growth_v1 preserves MG2 cost12, seeded routing,
radial growth ranking, domain and cube scale. After assembling the complete route,
it checks the existing global contact limit. For every proposed addition it counts
unique new occupied/contact cells, excluding overlaps, and admits the block only
when the resulting global ratio remains compliant. This enforces an existing
family; it adds no new constraint and does not redefine the evaluator.

Deferred origins are reconsidered after positive geometric growth. Accepted
zero-delta origins can expand the frontier once but cannot trigger false progress.
Explicit route-budget, finite-stall, component-exhaustion and timeout outcomes
retain partial results. Greedy growth is not optimal or complete; it can stall
even if a different route/order could work. No nonblocked benchmark case stalled;
stall behavior is supported by focused tests, not a measured broad success rate.

The24% repair evaluated 11 inadmissible proposals over7 distinct origins; the32%
repair evaluated1819 over52 origins. Counts include repeated consideration, not
that many distinct blocks. Respectively58 and88 accepted zero-delta origins were
finite transit steps. All16,873 growth decisions were independently recomputed
from actual cube unions and reconstructed final fields.

## Variation and timing

43 unchanged fields retain their prior variation. Partial-obstruction valid-only
groups each have3 distinct fields. Mean pairwise Jaccard distances are0.299240,
0.201037 and0.151595 at16%,24%,32%. MG2's last two groups each had only2 valid
members, so comparing those means is not a matched population diversity test.
These measurements establish voxel differences, not architectural quality.

Generation totals7.0761s, individual0.139923–0.200554s. Evaluation of new fields
totals6.6466s. Study33.2482s includes prior rescoring and evidence writes before
final summary packaging. Regression ran first, with no overlap. Single local CPU
timings are descriptive observations, not repeated estimates or GPU forecasts.

## Verification and disposition

310 regression tests pass, zero failures/errors/skips, smoke exit0. Seven new
tests cover overlap accounting, reconsideration, stalled growth, zero-delta
transit, invalid completed route, timeout retention and a real partial case.
Regression tests100.529s; full verification command105.7143s.

45 generator/route/complete metadata replays match exactly excluding wall time.
135 binary scores and135 bulk masks (MG1/MG2/MG3) reproduce. Both source snapshots
match150 current Python files; all snapshot hashes and run manifests verify.
Independent growth accounting, diversity and admission-gate checks pass.
See experiments/reports/MG3-verification.json and MG3-final-verification.json.

New /static/budget/index.html presents45 MG2/MG3 pairs, including all failures,
three-decimal contact ratios and exact changed-cell counts. Its projected fields,
scores, generation summaries and comparisons match archived evidence. Browser
checks cover all selections, nine rows, slices, cutaway/context and desktop/mobile
widths without horizontal overflow; no warning/error logs. Saved UI source and
QA captures accompany the local archive. An initial mobile capture showed an
older compositor frame; a fresh capture records the selected repaired case.

The gate admits experimental live Studio integration, not production reliability
on unseen sites. This remains procedural generation, not a newly trained NCA.
Follow STUDIO_MASSING_INTEGRATION_PLAN.md. Existing live generation, historical
model/metrics/galleries and private reports remain unchanged. No server restart,
Drive access, public deployment, push or paid training occurred.
