# CGR2 final development review — 2026-09-29

CGR2 completes successfully but does not meet frozen acceptance. Reduced excess
comes with reduced repair recovery; retain CGR1 as the experimental reference.
MG7 remains live. No model promotion or additional training.

## Verified evidence

Run20260929T062933Z_465972e5c250:256 CUDA updates,seed1201,32steps;
78.467s controlled,75.666s worker,714MiB peak reserved memory,cleanup exit0.
Verified outer receipt, all1,036 payload hashes, exact unique ZIP membership,
final checkpoint digest/bytes/semantic identity and optimizer/sampler/trace cursor.
User-supplied execution evidence is not standing permission for another paid run.
GPU execution succeeds; exact GPU interruption/recovery was not tested here.

Evaluation: final256 only, CPUfloat32,32steps,firing2101,27 existing development
cases. Accepted binary occupancy, no cleanup or threshold tuning. No TEST.
All histories, raw proposals, birth masks and state arrays preserved. Verified
54 prior CGR1/NR5 arrays and matching case,damage and baseline records.

| Metric | NR5 | CGR1 | CGR2 | Closing3 |
|---|---:|---:|---:|---:|
| All-nine pass, all27 |17|24|24|24|
| All-nine pass, damaged18 |11|15|15|15|
| Damaged median IoU |.97058|.97504|.97096|.97268|
| Damaged recovered cells |1945|1959|1828|1464|
| Damaged excess cells |325|245|165|13|
| Damaged median absolute volume error, cells |19|15.5|23.5|38|
| Intact excess cells |172|117|83|13|
| Intact median IoU |.98633|.99426|.99316|1.00000|

CGR2 improves excess counts but does not dominate CGR1 or the simple baseline.
Despite fewer total intact excess cells, median intact overlap slightly worsens:
aggregate counts and medians describe different aspects of the distribution.

## Frozen decision

Four of eight gates fail: all-intact overlap (3/9 below.99), damaged validity
(15/18 versus17required), damaged recovery (1828 versus1945required), and median
absolute volume error (23.5 versus19maximum). Intact validity9/9, damaged overlap,
damaged excess and preservation pass. No surviving input cells are removed.

The same three cube5 examples fail access and thickness. Support passes27/27;
zero detached occupied voxels. Enforced attachment does not ensure a thick bulk
connection. The new supervision did not resolve those failures in this one run.
This does not establish that either new loss term individually caused the result:
both changed together, with one training seed and repeatedly used development data.

## Recommended next step

Pause further coefficient trials. Use saved trajectories to inspect where repair
stalls on TRAIN examples and whether missing bridge cells ever become eligible,
remain below threshold, or are blocked by earlier irreversible growth. Compare
CGR1 and CGR2 on identical inputs/firing without optimizing against TEST. This
local diagnosis should distinguish an objective imbalance from the limitations
of detached hard births before proposing another bounded experiment. No claim
that increasing grid size or training duration alone will fix it. Maintain the
nine families and overall-volume concept; this finding does not call for rooms.

## Artifacts and resume

Full evidence: C:/Users/artin/Documents/Codex/outputs/CGR2-Final-Review-2026-09-29.
Review script: scripts/review_bulk_run.py; requires fresh output directory.
Small record: experiments/reports/CGR2-final-review.json. The supplied ZIP and
receipt and every case remain local; an adjacent verified milestone ZIP includes
these and final project records. Same-disk copies are not off-device backup.
No Drive access, push, live replacement, extra seed or paid run performed.
