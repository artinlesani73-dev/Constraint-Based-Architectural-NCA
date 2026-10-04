# G7 TRAIN-only diagnosis and next decision — 2026-10-04

The slowdown is **later proposal scarcity**, not delayed seed activation.
Both models grow their first full cube at step1 in all45 cases. The next bounded
experiment will increase training exposure only; it does not presume that longer
training fixes connection allocation.

## Scope and verification

Compared frozen G6 and G7 final256 weights on all45 G7 TRAIN examples:27 shared
original examples and18 added vertical examples. Those18 were not G6 training
data and are identified separately. Each used one64-step rollout, firing2101,
the unchanged0.5 threshold and quota. No optimizer updates, parameter searches,
new teacher construction, or development/reserved inference occurred.
All5760 detached admission transitions were replayed exactly against captured
fields/counts. Saved all90 rollouts, birth masks, terminal states, per-step
eligible indices, probabilities and firing, summaries, checkpoints and source.
Teacher membership and context distance were analysis labels only, never model
inputs. Connection here means contact with the opposite interface from a
connected legal field; this is not a new nine-family quality benchmark.

## Findings

|64-step TRAIN diagnostic|G6 original27|G7 original27|G6 added18|G7 added18|
|---|---:|---:|---:|---:|
|First cube step, every case|1|1|1|1|
|Opposite interface reached|25/27|21/27|15/18|14/18|
|Median absolute volume error, pp|0.237|2.970|0.157|3.107|
|Median target shortfall, voxels|0|108|0|153.5|

The quota schedule has enough theoretical capacity to reach the requested volume
by64 in every case given the observed step1 start. It does not guarantee that
the model offers the cubes needed to use that capacity.

Across non-seed steps where the full per-step quota applies (before the global
ceiling truncates it), G6 leaves847/46,344
allowance cells unused (1.83%). G7 leaves
8,579/47,352 unused
(18.12%). These are lost step opportunities,
not necessarily distinct missing final voxels.

Of G7's unused allowance,7,243 cells
(84.43%) occur on steps with
no allowance rejection: all eligible probabilities are below threshold, firing
misses the few above-threshold proposals, or the offered cubes are exhausted.
Only the remaining15.57% occurs alongside whole-cube allowance rejection. This
is a descriptive partition, not a causal estimate of changing the cap. It argues
against treating quota packing or delayed startup as the dominant diagnosis.

Before connection, on steps with fired teacher-positive candidates in both
progress and other groups, G7's mean progress probability is<=0.5 in113/720
shared-data steps and127/592 added-data steps, versus11/666 and28/503 for G6.
The models follow different trajectories, so these are conditional diagnostics,
not comparisons on identical hidden states or proof of calibrated confidence.
Progress means lowering minimum context cube-graph distance to the opposite
interface; other growth can still be useful building volume.

The last64 retained training updates also start adding at step1 in both start
modes. Their losses are not directly comparable as causal evidence because
examples and states differ. Reduced exposure is a plausible unresolved factor,
not an established sole cause. Some G6 fields already reach the cap without
connecting, showing that learning to fill faster alone will not guarantee access.

## Single next intervention: G8 exposure

Retain G7's exact45 data payloads, initialization, model, optimizer, teacher
stages, loss, firing, hard transition, nine families and64/128 review horizons.
Change only total retained updates from256 to427:
ceil(256*45/27)=427. This gives9-10 visits per row and approximately restores
G6's mean exposure. The number was derived from dataset sizes, not selected by
a checkpoint sweep. This tests exposure; it is not a promised solution or an
equal-compute comparison with G7.

Use a fresh same-seed run so the full lineage remains unambiguous. Compare the
new update256 numerical payload against the previous G7 update256 after return,
then judge only final427. Package/model provenance can differ; no checkpoint
selection is allowed. The existing model has not been modified or promoted.

G8 package is prepared at `C:\Users\artin\Documents\Codex\outputs\G8-Exposure-Training-2026-10-04`. Its three-update local rehearsal and both
recovery replays passed; update0 andupdate3 match all G7 numerical payload keys,
with identity kept separate. Four fresh reserved scenes (12 requests) were
frozen before any G8 training, with no labels or inference. The33 consumed
legacy requests become regression evidence. Both cohorts retain the existing
all-nine, size and stability requirements. No architecture/loss/threshold change
is bundled in this proposal.

One T4 run of427 updates is proposed, capped600 controlled seconds; setup,
export/download and idle are extra. Based on G7, roughly310-320 controlled
seconds is a planning estimate, not a guarantee. Approval is still pending.
No paid retry, Drive operation, push, publication or MG7 replacement occurred.

## Resume and preservation

Use G8 RESUME.json for the ready package and next user action. This diagnostic,
all raw result arrays, exact dependencies and its verified same-disk archive
remain retained. Repository synchronization and off-device backup remain pending.
The original report and all historical runs are untouched.
