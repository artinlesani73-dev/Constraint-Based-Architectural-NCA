# MG4: frozen new-site stress evaluation

2026-09-24, before any candidate generation. User authorized the next-site study.
This tests the unchanged MG3 procedural generator on previously untested designed
variations. It is neither a trained-model test nor a random held-out population.
Once inspected, these cases become development evidence, not a reusable blind test.

## Frozen design

experiments/scenes/MG4-sites.json defines20 exact32-cubed,0.8m sites. Three gap
variants, three height variants, four interface variants, three obstacle variants,
three combined variants and four full-height partition controls. Groups describe
input variation, not new constraint families. Metadata-independent geometry must
be unique against all five MG1/MG3 sites and within this set. Static scene and
interface validation is allowed before freezing; no generator results used to
select cases. Nonpartition does not imply feasibility or guaranteed success.

Every site uses requests16%,24%,32% and seeds3,4,5:180 candidates,144 nonpartition
and36 deliberate partition controls. No seed search, dropped case, scene repair,
threshold relaxation or parameter sweep after outcomes. Preserve complete fields,
routes, proposal decisions, evaluator masks/scores and errors. Exact scene bytes
and all existing core source hashes are frozen in MG4-sites.json recipe.

Same2.4m growth block, contact cost12,15s cooperative generation cap, two CPU
threads, MT1 defaults and6.4/6.4/0m region padding. Same nine families.1200s study
cap checked at case boundaries;2GiB observed resident-memory stopping threshold
checked between cases. These are cooperative bounds, not OS-enforced limits.
No changes to the live five-site Studio, trained checkpoint or private report.

## Measurements and decision

Report all180 outcomes and the predeclared144/36 partitions separately. Record
family failures, context necessary checks, termination, requested/actual count,
physical volume, local bulk and contact. Request fidelity is0..26 extra cells
(less than one3-cubed block); route overshoot remains a failure if larger. Include
per-site/request unique fields and mean pairwise Jaccard distance across valid
seeds, with sample count; no diversity estimate when fewer than two are valid.

Measure context setup, generation, evaluation and total study separately; CPU
and wall time are descriptive single observations. Memory records use a10ms
resident-set sampler around generation+evaluation, an immediate start/end sample,
and process-lifetime peak working set. Baseline and delta are reported. Native
runtime/allocator/cache memory is included; short peaks can be missed by sampling,
and the OS lifetime peak cannot be assigned to an individual case. Avoid retaining
all full traces in memory across cases. No tracemalloc claim about tensor memory.

Admission for treating the unchanged recipe as ready for broader scale work:
all144 nonpartition candidates pass all nine MT1 checks AND request fidelity;
all36 partition controls fail; no timeouts/errors/resource breach. This strict
designed-set gate is not a population success guarantee. If it fails, diagnose
the preserved fields and contexts before modifying the generator or expanding
the live interface. Larger grids are not executed by this protocol. Separate
larger physical environments from finer resolution in a later frozen study.

Freeze protocol/config/scenes and source snapshot in an immutable run before
generation. Log every completed case immediately. Interrupted reruns get a new
ID linked to their predecessor; never overwrite earlier results. Independently
recompute scores and verify fields/source hashes after the run. No paid compute,
Drive access, public deployment, push or new NCA training is admitted here.
