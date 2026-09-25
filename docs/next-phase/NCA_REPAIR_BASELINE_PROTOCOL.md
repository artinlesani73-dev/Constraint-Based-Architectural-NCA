# NL0 / NR1: a learned repair baseline for building volumes

Frozen 2026-09-25 before NL0 benchmark outcomes. NL0 prepares and audits data;
NR1 is the next proposed learning experiment. No trained model is claimed here.

## Why this next step

Keep the overall building-volume brief (D058), MT1's nine families and MG7 as the
current generator. F3 lost connectivity; F4/F5 improved it without satisfying the
old material budget. Those results used historical material semantics and are not
scores under MT1. MD1 lowered continuous penalties without repairing binary forms.
Do not initialize a new massing model from those unpromoted material checkpoints
or admit MD1's objective merely because its derivatives exist.

Test a narrower capability first: recover a partially damaged acceptable volume
while preserving the surviving design. This prepares a hybrid planner/NCA route;
it does not abandon later conditioned generation. Reconstruction supervision is
a training signal, not a tenth architectural constraint. It cannot by itself
prove validity, originality, stability, diversity or architectural quality.

Growing Neural Cellular Automata demonstrates damaged-state training and sample
pools for persistence/regeneration ([Mordvintsev et al., 2020](https://distill.pub/2020/growing-ca/)).
The 3D extension establishes precedent for learned voxel growth
([Sudhakaran et al., 2021](https://arxiv.org/abs/2103.08737)). These motivate a
repair experiment; neither establishes that our site-conditioned masses will work.
Our explicit conditioning, split, loss and caps below are project hypotheses.

## NL0 frozen preparation

Use the exact MG7 matrix run 20260925T083302Z_6c7ade2f0e16, never reroll teachers.
The JSON recipe lists every selected case before outcomes. All are 32 cubed,
0.8m cells, MT1 width 2.4m, volume requests 16/24/32%.

| Split | Sites | Generator seeds | Targets |
|---|---|---|---:|
| Train | aligned, wide_gap, partial_obstruction | 0/1/2 | 27 |
| Validation | offset_interfaces | 0/1/2 | 9 |
| Test | combined_tall_east, combined_reverse | 3/4/5 | 18 |
| Blocked guard | blocked_gap | 0/1/2 | 9 failed procedural records |

These six positive site geometries were already inspected in procedural research.
Validation/test mean excluded from the new model's training; they are not newly
discovered or statistically independent real-world sites. Derivative geometries
share design ancestry. No random voxel, damage or seed split of the same site is
permitted. Hash full context arrays (excluding request/labels) and target occupancy
across splits; reject exact duplicate geometry or target leakage. Report counts.
The blocked site is a separate guard, never an empty positive reconstruction label.

For each of 54 targets retain intact, cube5 and slab2 variants: 162 examples.
Cube5 removes one 5-cell-wide cube centered on a deterministically selected occupied
cell using SHA256(seed 9101, case, kind). Slab2 removes two X slices beginning at
the occupied-cell median X. Clip at world boundary. No rerolls, selection by failure,
minimum damage tuning or post-result thresholds. Save cut region and actual removed
cells separately; this is synthetic missing-volume damage, not an environmental edit.

Compare unchanged damaged input with one fixed nonlearned operation: 3-cube binary
closing, union surviving input, projected to permitted domain. Save and score raw
returned candidates under every MT1 family, target IoU, cell errors, recovery of
missing cells, removed surviving cells, false additions and requested-volume error.
All three variants are separate strata; intact cases cannot inflate repair rates.
Store the intact teacher as a reference upper bound. Restoring its saved bytes is
the correct product action when they exist; an NCA has no demonstrated advantage
over that. MG7 same-seed regeneration also reproduces it and is an information-rich
reference, not a fair target-blind repair algorithm. Benchmark those costs in NR1.

Network inputs may contain damaged occupancy and seven static channels: domain,
permitted, existing, protected, support boundary, union of interfaces, and requested
fraction. Target is a separate supervisor label. Do not feed the cut mask, teacher
occupancy/route, generator seed, case ID or target checksum into the model.
The teacher used to construct synthetic damage is not subsequently available as
conditioning. Conditional reconstruction is ambiguous; do not expect unique recovery
from every partial shape or mistake an alternate valid volume for perfect restoration.

NL0 admission: all 54 archived positive targets rescore identically and pass MT1;
all nine blocked controls retain their failure; all 108 damages remove positive
volume without emptying the target; split integrity, repeatability and exact lossless
reload pass; every example/baseline/result retained. Comparator performance is an
outcome, not an admission threshold. 600s study boundary cap, one process, CPU,
no training/paid compute. Failures get linked new runs; no overwritten evidence.

## NR1 proposed implementation and local gate (not executed by NL0)

Version a fresh small 3D NCA. Eight evolving channels: occupancy logit plus seven
hidden channels. Per-cell local perception: identity and three fixed central
differences for the eight dynamic/seven static channels; 60 features through a
shared 60->64->8 pointwise MLP with ReLU, final layer initialized to zero.
Apply residual updates with independent 0.5 Bernoulli firing per cell. Initialize
logit to +2/-2 from damaged occupancy, hidden state zero. Keep context immutable;
reset hidden state outside permitted domain, project output probability to that
domain. No occupancy clamp in the loss path and no life mask that makes erased
regions permanently unreachable. This is a new repair representation, not a
controlled attribution against F5. Architecture performance must be tested.

First use fresh damaged states each update, batch1, 16 recurrent steps, Adam
lr0.001, clip global gradient norm1.0; CPU float32 with two threads for local
verification. No state pool, mixed precision, scheduler or variable training horizon
in this baseline. After a reproducible baseline, pool/long-horizon training is a
separate matched experiment. Use class-balanced BCEWithLogits within the legal
domain: one half mean positive-target loss plus one half mean negative-target loss.
Both classes must be nonempty. No automatic weight sweeps or constraint-loss mixing.
Fixed >0.5 occupancy is the evaluation rule; no tuned threshold or automatic repair
postprocessing of learned output. Save unprojected logits too so projection effects
remain inspectable. Hard legal projection is declared, not evidence of learned legality.

Before larger training, implement a CPU mechanics pilot on two TRAIN examples only:
8 updates/member, seed1201, 600s total cap. Use an actual new-process interruption
after update4 and compare resumed updates5-8 with uninterrupted weights, Adam,
Python/NumPy/Torch RNG, next data cursor, loss trace and final field exactly.
Checkpoint every completed update with atomic publication plus SHA256 manifest;
retain every previous checkpoint and failed partial-write artifact. Reject mismatched
source/config/dataset/optimizer/device identities. The latest verified completed
update is the restart point, never a partially written file. This pilot tests mechanics
and timing only; it is not a model-quality trial and must not consume test examples.

## Proposed Colab allowance and decision gate

After local recovery passes, prepare a runnable notebook/package with the frozen
dataset, hashes, versioned code and independent verification. Ask for approval of
the concrete package and allowance then. Proposal: one GPU session, at most60 GPU
minutes total including setup/profiling/evaluation, three training seeds1201/1202/1203,
256 updates/model maximum, batch1 and unchanged16 steps. First8 updates are the
timing/memory/recovery pilot; retain and count all work in the allowance. Do not
promise a dollar price or available GPU type. Stop at the earlier step/time cap;
missing final models mean incomplete evidence, not permission to extend the budget.
Profile on the actual GPU; CPU timing cannot set a credible GPU completion estimate.
GPU fresh-process recovery must pass separately; CPU exact replay does not certify
GPU/AMP or reproducibility across devices. No Drive mounting/sync built into the
package. Ask separately for any exact project-folder Drive operation and readback.
The present proposal is NOT authorization to run paid compute or access Drive.

Use final update256 checkpoints only for the primary comparison (no best-step
selection). Validation checkpoints0/64/128/192/256 are diagnostic, not retuning.
Evaluate each completed model with firing seeds2101/2102/2103 at16/32/64 steps;
32 is the frozen primary horizon, other horizons test sensitivity/persistence.
Keep 16-step training even if 64-step evaluation fails. Test set evaluated only
after all three final models are fixed. No 48/64-grid promotion in this phase.

Primary go/no-go for a further repair study: on TEST damaged examples, each of the
three trained seeds, aggregated over its three firing seeds, must improve median
IoU over BOTH unchanged input and fixed closing by at least0.02; all-nine pass rate
must be no worse than either comparator. On TEST intact examples require every
evaluated primary-horizon field to preserve all-nine validity and at least0.99 IoU.
Report every request error; require absolute error <= max(8 cells,1% domain) on
every accepted repaired case, not only the mean. No selective case omission.
These are provisional admission tolerances, not scientific significance or product
certification. Failure closes this recipe; review saved binary errors before a
separately justified revision. Firing seeds are not independently trained models.

Time full loading/conditioning/rollout/evaluation separately and compare against
MG7 regeneration and byte restoration with explicit information access. Speed is
secondary: MG7 is already fast. No quality claim based on BCE, positive gradient,
single attractive example, or the number of evaluated firing seeds. Studio remains
procedural until a separately admitted learned workflow earns integration.
