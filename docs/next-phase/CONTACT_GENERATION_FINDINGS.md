# MG2: full contact-aware generation findings

2026-09-24. Run 20260924T133657Z_f4c92840c7c0; regression
20260924T133923Z_4b6c61aedbb5. Protocol and admission rule were frozen before results.
Execution completed; the Studio promotion gate failed. No coefficient changes,
rerolls, threshold relaxation or failed execution attempt occurred.

## Result and scope

| Context | Original MG1 | Contact-aware MG2 | Change |
|---|---:|---:|---|
| Aligned | 9/9 | 9/9 | All baseline successes preserved |
| Wider gap | 9/9 | 9/9 | All baseline successes preserved |
| Offset interfaces | 9/9 | 9/9 | All baseline successes preserved |
| Partial obstruction | 0/9 | 7/9 | Seven repairs; two facade failures remain |
| Blocked gap | 0/9 | 0/9 | No cube route; empty failed outputs retained |
| **Total** | **27/45** | **34/45** | **No baseline regressions** |

The nonblocked denominator is 36: 34/36 pass. The full denominator stays 45.
All 36 nonblocked runs reach their requested count with 0–8 extra cells; their
full binary volume is cube-qualified bulk. No new family failed in those outputs.
All nine blocked outputs fail access, coverage, sparsity, support and thickness;
this is the generator's routing outcome, not a universal infeasibility proof.

Five contexts, three requests (16%, 24%, 32%), three seeds (0,1,2). Scenes,
domains, MT1 and contact cost 12 exactly match the frozen plan. The four MD1
pilot controls reproduce exactly within this matrix; they are repeated members,
not four additional independent confirmations. These are development examples.

## The two remaining failures

| Case | Final cells | Nonexempt contact cells | Contact fraction | Other eight families |
|---|---:|---:|---:|---|
| Partial obstruction, 24%, seed 2 | 819 | 123 | 15.018315% | Pass |
| Partial obstruction, 32%, seed 2 | 1093 | 181 | 16.559927% | Pass |

Both fail the unchanged 15% facade limit. At their current total masses, at most
122 and 163 contact cells fit that limit: excesses of 1 and 18 respectively.
This arithmetic is not a safe deletion prescription: removing voxels changes
the denominator and can break thickness, connectivity or volume fidelity.
The near-boundary case remains a failure; rounding does not make it pass.

Both share a completed initial route with 198 cells and 14 nonexempt contacts
(7.070707%). The saved cube trace shows the first growth-stage exceedance at
zero-based selected-cube index 203, origin(z,y,x)=(7,19,21): adding three cells,
all contact cells, gives 96 contacts / 633 total =15.165877%. Later additions can
reduce or increase the ratio, so this crossing alone does not establish the
final failure; the complete trace and final counts do. The24% run stops at 819;
the 32% run continues the same seeded growth ordering to 1093.

Partial prefixes while assembling the initial route exceed 15% before the full
route is complete. A future growth check should assess the completed route first,
not treat every initial route-prefix exceedance as an impossible final design.

The present cost penalizes contact fraction of each complete candidate cube.
Overlapping cubes repeatedly include old cells; that ranking does not enforce
the global ratio of unique occupied cells. Recorded traces establish where the
global limit is exceeded. They do not prove that the proposed next algorithm
will reach all volume requests under every constraint.

## Variation and cost

Every valid field remains distinct within its scene/request group. Contact costs
nevertheless reduce mean pairwise voxel difference across seeds in aligned and
wider-gap contexts. This is a useful tradeoff to retain, not architectural quality.

| Context | Request | MG1 mean Jaccard distance | MG2 mean Jaccard distance |
|---|---:|---:|---:|
| Aligned |16%|0.4126|0.2550|
| Aligned |24%|0.3875|0.2159|
| Aligned |32%|0.3090|0.1599|
| Wider gap |16%|0.4620|0.2451|
| Wider gap |24%|0.3740|0.1904|
| Wider gap |32%|0.3118|0.1458|
| Offset interfaces |16%|0.1842|0.1705|
| Offset interfaces |24%|0.1085|0.1122|
| Offset interfaces |32%|0.0597|0.0771|

MG2 partial-obstruction valid-only distances are 0.2992 (3 valid at 16%),0.2463
(2 at 24%),0.1972 (2 at 32%). MG1 had no valid partial alternatives, so no valid-only
MG1 diversity estimate exists there. Failed outputs are excluded only from this
diversity calculation, never from the pass-rate denominator or archive.

Generation totals 7.419s across 45 candidates, individual 0.140–0.230s, including
contact-context setup. New-output evaluation 6.753s; study 25.689s includes original
rescores, source setup and per-case evidence writes before final packaging.
CPU timings are descriptive single-run observations, not repeated benchmark
estimates or GPU predictions. No cap was exceeded. The later replay verification
and regression ran concurrently; their durations are not timing measurements.

## Verification and decision

All 45 generated fields, routes and complete selection metadata replay exactly
(wall-time values excluded). 90 original/new binary scores and 90 bulk masks match.
All four MD1 repeated members match their prior fields and scores. Per-group
valid-only diversity recomputes exactly. Source ZIP members and run manifests
verify; 147 Python files match both benchmark and regression snapshots.
Full regression: 303 tests passed, zero failures/errors/skips, smoke exit 0.

The fixed rule required 36/36 nonblocked passes. It was not met. Do not promote
this generator to the live Studio, lower the gate after seeing the result or
start NCA training. Preserve the demonstrated improvement and failed members.
Read CONTACT_BUDGET_NEXT_PLAN.md for one bounded proposed algorithmic change.
This is not a cost-weight search or a claim that another local trial must succeed.

New saved comparison: /static/contact/index.html, linked from Studio. All 45
original/new pairs and failures remain selectable, with three-decimal contact
percentages so 15.018% does not appear as 15.0%. The original MG1, MD1 and all
historical galleries stay unchanged. This is an evidence viewer, not live mass
generation. Browser checks cover 45 selections, nine family rows, both slices,
keyboard index 16 / 13.2m, cutaway and context. Mobile 390px has 375px document width,
tables 343.8125px; desktop 1000px has 985px document width and two 443.5px panels.
No horizontal overflow or browser warning/error logs observed. Temporary viewport
override reset; screenshots, selector records and projection hashes retained.

No server restart, paid compute, Drive operation, remote push or publication.
The private report remains ignored and unchanged. Local evidence and resume
instructions are archived; same-disk backup is not an off-device backup.
