# Paired reversible repair decision

Do not adopt RGR1. In this matched-runtime paired trial it repairs less accurately
than CGR1 and takes longer. Both fail the frozen acceptance criteria. Retain CGR1
as the research reference and MG7 as the live generator; close this reversible
candidate without an automatic additional tweak or training run.

## Verified experiment

Run20261003T142251Z_20c919d80cee completed256updates for each arm,512total,
within173.191controlled seconds. All2071payload hashes and outer receipt verified,
with exact unique archive membership. Initial and final checkpoint hashes,identity,
Adam update counts,sampler cursors and256trace records per arm were checked.
Initial weights match exactly; row order,ordered data hashes and final RNG payloads
match. Both used T4,Torch2.11.0+cu130,CUDA13.0,cuDNN92700,seed1201,32steps.
This removes the runtime confound from the direct CGR1/RGR1 comparison,not every
possible experimental limitation. The update rule and supervised set both differ.

Final256only,CPUfloat32,32steps,firing2101,original27development inputs; no TEST,
threshold changes,checkpoint selection or extra horizon. Saved and verified54
observation arrays and all case metrics. RGR1 includes its declared anchor projection;
no additional cleanup. Pre-projection diagnostics refer to the FINAL candidate
of the actual projected trajectory,not a separate rollout with projection disabled.

## Results

| Metric | Same-runtime CGR1 | RGR1 |
|---|---:|---:|
| All nine checks,all27 |24|24|
| All nine checks,damaged18 |15|15|
| Damaged median overlap |.97504|.96957|
| Correctly recovered cells |1959|1818|
| Excess cells on damaged inputs |246|227|
| Median absolute requested-volume error,damaged |15.5|18|
| Excess cells on intact inputs |117|116|
| Intact median overlap |.99426|.99085|
| Intact examples below.99overlap |3|4|
| Arm elapsed seconds including its exports |62.051|106.705|
| Peak reserved GPU memory,MiB |710|710|

RGR1 removes19excess damaged cells at the cost of141fewer correctly recovered
cells. Arm time is1.72times the control. The earlier16.5times timing ratio was a
single artificial forward probe,not the full training slowdown. Median CPU review
rollout times were.797s and.906s; these single-run timings are not a hardware benchmark.
The new CGR1 control differs from historical CGR1 by one excess damaged cell
(246versus245),reinforcing why the current paired control should be used.

CGR1 fails2of8gates:intact overlap and damaged validity. RGR1 fails4of8:those two,
damaged overlap and damaged recovery. Both preserve every original input voxel.
The same three cube5 examples fail access/thickness. RGR1 additionally fails
coverage on v16,s0. All other families pass27/27 for both.

## What reversibility actually did

Across27rollouts,RGR1 recorded406direct removals of target-correct additions and
363direct removals of excess additions. Connectivity projection additionally removed
9correct and7excess additions. These are removal EVENTS and may count the same
voxel repeatedly; they are not unique lost cells or permanent error totals.
233case-local cells were born more than once,demonstrating repeated growth after
removal. Original-input removal events were zero. The pre-projection final candidates
also passed24/27; this is not evidence that projection was irrelevant earlier.

This result rejects this particular candidate for adoption under this budget.
It does not prove that every reversible NCA is inferior. One seed,reused development
scenes,detached decisions and a proxy loss limit the conclusion. No pure deletion
ablation was performed,so do not attribute the entire difference to reversibility
alone. The observed deletions show the intended mechanism runs but is insufficient.

## Next phase decision

Stop the sequence of small repair-model variants. Preserve this negative result
and the stronger CGR1 reference. Neither learned model is approved for deployment.
The next planning milestone should return to the user's actual outcome: generation
of volumetric forms from constraints across varied scenes. Define a separate
seed-to-volume generation benchmark and its TRAIN/validation/TEST boundaries before
selecting another training experiment. Repair of an already shaped input is not
evidence for that capability. Reuse the audited nine-family metrics,checkpoint
infrastructure and existing procedural generator as comparison assets; do not
relabel procedural outputs as learned results or loosen criteria silently.

No new paid job,Drive operation,push or live replacement follows from this decision.
Full ZIP exports remain the user's preference. Repository writes remain unavailable;
all source,results and pending resume notes are saved locally for later sync.
