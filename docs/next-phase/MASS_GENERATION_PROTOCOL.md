# MG1 - First procedural building-mass comparison

Frozen before execution, 2026-09-24, R2-A / D062. Occupancy means building volume;
no rooms, construction or new constraint families. Config MG1-procedural.json.

## Algorithm and supported domain

cube_route_growth_v1 accepts scene_v1, exactly two interfaces, legal scene-sized
domain and explicit nonnegative seed. Enumerate fully contained cubes at 2.4 m
(3 cells at the current 0.8 m resolution). Use positive seeded Dijkstra costs on
the six-neighbor cube-origin graph to find ONE route between cubes intersecting
each interface's cells on or face-adjacent to the support boundary. Multi-source search considers alternative
source components but reconstructs only one connected path. Return explicit failure
when no route exists. This restrictive construction is not a site feasibility proof.

Grow a connected union of legal cubes around the route toward the requested volume.
Seeded axis weights and jitter vary radial growth order. All intermediate selected
origins and the initial route volume are saved. There is no MT1-score feedback,
rejection sampling, final clipping, hidden repair or learned component. The request
is a stopping target, not a new family or guaranteed exact volume. A last cube can
overshoot; a route already larger than requested is returned honestly. Partial
output is retained on cap/exhaustion. Empty output is retained on unsupported cases.

Construction guarantees legal cube-union shape under its domain assumptions, not
all-family acceptance, useful articulation or architectural quality. In particular
facade contact and X-third coverage may fail. Evaluator remains unchanged.

## Frozen matrix and timing

Five development contexts: four MT1 scenes unchanged plus aligned with a partial
obstacle at x[15,17), y[13,18), z[6,14). The obstacle is new context diversity,
not a new constraint. Three requested domain-volume fractions 16/24/32 percent,
three seeds 0/1/2: 45 generated candidates. Every request gets a record, including
blocked gap outcomes. No minimum success rate is assumed; completion means retained
outcomes, verified provenance and a decision based on failures.

CPU only, 2 Torch threads. Generator cooperative cap 15 seconds/candidate,
study cooperative cap 600 seconds including setup after run creation. Check the
study clock between cases; a running evaluator/source snapshot can exceed the cap
before the next boundary. If cap is exceeded, finalize interrupted with retained
partial evidence and mark remaining tasks unexecuted; no silent continuation.
Snapshotting, JSON evidence writes and evaluation are included in study wall time;
per-candidate generation and evaluation are also timed separately. No latency
benchmark against NCA or equal-compute assertion follows.

## Independent evaluation and challenges

Evaluate every field under unchanged massing_targets_v1 and retain bulk masks,
nine-family results, full context masks, physical volumes and source/recipe hashes.
Report target error separately from validity. Compute pairwise Jaccard difference
only among valid fields, per scene AND requested volume; also report all-request
scene summaries separately to expose volume-related differences. Record invalid
fraction, duplicates and every pairwise value, not only best alternatives.

Four analytical challenges on aligned context are evaluated separately from the
45 generator outputs: compact mass with a thin appendage, a bulky lattice made of
3-cell bars, a diagonal band and a quarter-turn of the compact field around Z.
They probe evaluator weaknesses/orientation; no preregistered pass labels and no
claim of a rotated-site invariance test (context stays fixed). Save their fields
even when illegal or disconnected. These are not generator success samples.

## Verification, stopping and recovery

Test deterministic replay, multi-source path reconstruction, cube-union invariants,
blocked graph/unsupported interfaces, target overshoot/exhaustion, explicit caps,
input preservation, validation and valid-only diversity. Archive the full regression.
After the run, verify manifests/source hashes and independently replay all generated
fields and generation metadata excluding wall time. Check count/budget/bulk integrity.
Retain every interrupted/failed attempt and use unique parent-linked IDs for retries.

Preserve original notebook, checkpoint, losses, MT1 fields and all old reports.
No outcome-based threshold tuning or automatic follow-on training. The exit decision
selects a concrete generator limitation to address or admits preparation of R2-B's
continuous objective/direct baseline. A comparison gallery is optional presentation;
Studio integration is a separate product step after these results are reviewed.
No paid Colab, Drive access, push or publication.
