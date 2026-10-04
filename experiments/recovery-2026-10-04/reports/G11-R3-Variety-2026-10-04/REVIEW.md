# R3 variety assessment — exploratory, 2026-10-04

Decision: follow R3 as the practical experimental path; retain G10 comparison. Test geometry transfer and firing variation before increasing resolution. Nine families and whole-building-volume interpretation unchanged. MG7 is not replaced.

One compact CPU assessment: four constructed geometry changes, three firing seeds each, 24% request and saved horizons 64/128. There are only four sites, not twelve independent sites. These probes are exploratory, not a new held-out release gate; uniqueness against all historical training scenes was not established. Weights are unchanged. The optional firing-seed argument is the only adapter edit, and default-seed births, provenance and both full horizon states exactly match the independent-review saved case.

## Outcomes at 128

- G10: 0/12 all-nine passes; 12/12 within 5% late growth; max absolute volume error 0.163 percentage points.
- R3: 12/12 all-nine passes; 12/12 within 5% late growth; max absolute volume error 0.163 percentage points.

## Variation between firing seeds

Mean pairwise occupied-voxel Jaccard distance at128. Zero means identical occupancy; larger means more different occupied sets. It is not an architectural-quality or useful-diversity metric. The deterministic planner is shared by all seeds on each site.

| Site | Model | Mean distance |
|---|---|---:|
| r3-variety-wide-gap | G10 | 0.277 |
| r3-variety-wide-gap | R3 | 0.205 |
| r3-variety-narrow-gap | G10 | 0.396 |
| r3-variety-narrow-gap | R3 | 0.193 |
| r3-variety-offset-y | G10 | 0.233 |
| r3-variety-offset-y | R3 | 0.159 |
| r3-variety-partial-obstacle | G10 | 0.626 |
| r3-variety-partial-obstacle | R3 | 0.255 |

Per-horizon family failures are preserved in failure-index.json; full metrics, traces, states, birth provenance and contexts are retained. Results do not justify tuning on these four probes and calling the same probes unseen afterward.

## Next

Use these results to set the boundary of the local R3 generation endpoint: report certificate failures explicitly, retain exact source/seed/scene/results per request, and keep raw G10 comparison. A passing result alone does not establish meaningful form diversity. Review the paired geometry plate before deciding whether a new learning objective or planner diversity mechanism is warranted. Do not increase resolution yet or start paid training.

Repository synchronization and off-device backup remain pending. No Drive access, remote push, publication or live-model promotion occurred. This milestone links to ../G11-R3-Preview-2026-10-04/RESUME.md. Original R3 source and all previous results remain unchanged.
