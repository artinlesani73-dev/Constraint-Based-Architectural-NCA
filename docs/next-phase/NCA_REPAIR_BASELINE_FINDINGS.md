# NL0: repair benchmark admitted; learning remains next

2026-09-25. Run `20260925T094341Z_316cff241020`. All54 exact MG7 teacher volumes pass
unchanged MT1 checks, all9 blocked records retain their failure. 162 examples
(54 intact,54 cube cuts,54 slab cuts),108 nonempty damage events,324 nonlearned
candidate comparisons. Six distinct positive context geometries and54 distinct
targets; no geometry/target cross-split duplicates. No learned updates performed.

This advances the accepted building-volume plan by defining a measurable first
role for the next NCA: recover damaged mass while preserving surviving geometry.
The current generator and live Studio remain MG7; the new data are research assets.
Read NCA_REPAIR_BASELINE_PROTOCOL.md for the frozen split, methods, provisional
learning architecture, objective, admission thresholds and proposed compute cap.

## What the simple repair control revealed

| Split | Damage | Unchanged valid | Closing valid | Median IoU: unchanged / closing |
|---|---|---:|---:|---:|
| train | intact | 27/27 | 27/27 | 1.0000 / 0.9976 |
| train | cube5 | 19/27 | 19/27 | 0.9269 / 0.9267 |
| train | slab2 | 0/27 | 27/27 | 0.8951 / 0.9920 |
| validation | intact | 9/9 | 9/9 | 1.0000 / 1.0000 |
| validation | cube5 | 6/9 | 6/9 | 0.9280 / 0.9280 |
| validation | slab2 | 0/9 | 9/9 | 0.8714 / 0.9943 |
| test | intact | 18/18 | 10/18 | 1.0000 / 0.9997 |
| test | cube5 | 7/18 | 6/18 | 0.9606 / 0.9575 |
| test | slab2 | 0/18 | 8/18 | 0.7750 / 0.9277 |


Closing improves all27 train and all9 validation slab cases to validity, but
only8/18 test slab cases. On intact test volumes it reduces18/18 valid to10/18.
The failed-family counts for these intact test outputs are `{'facade': 8}`.
Their aggregate81 added cells are enough to matter even when median IoU is
0.999668. The procedure removes no surviving cells on any example, but can add
undesired cells. Do not deploy it as automatic postprocessing. This is a negative
control outcome, not failure of dataset admission or evidence about a trained NCA.

On test cube cuts, unchanged validity is7/18 and closing6/18. Some incomplete
volumes remain acceptable under MT1; validity alone does not mean original-shape
restoration. Both reconstruction and nine-family checks are required.
Across36 damaged test examples, median IoU is0.904255 unchanged
and0.946511 after closing. NR1's proposed0.02 absolute improvement
threshold over both is below the mathematical IoU ceiling; attainability by an
NCA remains unknown. No thresholds or cases changed after these results.

The six positive site geometries come from already inspected procedural studies;
test/validation are excluded from future learned updates, not unseen to this
research project. Shared site ancestry and small counts limit generalization.
There is no held-out learned performance to report. Future model inputs exclude
target/cut masks, routes, generator seeds, IDs and hashes. Local loader split checks
prevent accidental use; they do not prevent a programmer from opening local files.

## Verification and retention

Regression `20260925T094122Z_b9efabe856ff`:357 tests,0 failures/errors/skips,
original checkpoint smoke0,234.16s. Six new tests cover deterministic
damage, nonmutation/blocked-label rejection, repair exclusions, error metrics,
duplicate-geometry/target leakage and default held-out loader rejection.
NL0 takes95.20s; it overlaps the end of regression,
so neither elapsed duration is an isolated training or inference speed benchmark.
No failed experiment attempt occurred in this milestone.

The post-run verifier checks765 source-snapshot members,54 teacher arrays against
MG7,162 saved examples and324 metric records, plus81 default-loader refusals for
held-out examples. All108 damages replay exactly in the builder.16,800 removed
cells are counted across damage examples, not distinct voxels in one world.
The frozen source hashes match both regression and preparation snapshots.
MT1 score formulas are shared with the existing evaluator; no independent new
topology implementation or GPU/restart certificate is claimed.

Code: nca/repair_benchmark.py and scripts/prepare_repair_benchmark.py. Small evidence:
experiments/reports/NL0-summary.json and NL0-verification.json; raw per-example
arrays, targets, contexts, full family scores, generator traces and source snapshot
are under .local-artifacts/runs/20260925T094341Z_316cff241020. Each NPZ has an archived SHA256.
The separate supervisor target is retained for training/evaluation, never supplied
as a static model channel. Failures/retries must be separate linked records.

## Exact next action

Implement NR1's fresh eight-channel conditioned NCA and trainer under a new version,
then the two-example CPU mechanics test and actual new-process update4 recovery.
Freeze complete sampler/checkpoint identities in machine-readable configuration
before execution. Test all model/optimizer/RNG/data cursor state, not just geometry.
No new weight sweep, larger grid or continuation of F5/MD1 is admitted by NL0.

Only after local mechanics and GPU profiling readiness should a runnable Colab
package be presented for approval. Proposed allowance is at most60 GPU minutes,
three training seeds,256 updates each; this is a proposal, not approved spend or
a claim that it fits an unknown GPU. Paid execution, Drive access and Studio model
promotion remain separate decisions. No user setup is required for the next local
implementation. Drive still requires exact per-operation permission.

Local commit and incremental archive are recorded in RESUME and the external
milestone receipt. Preserve MS2,MG7 and older archive chain. Same disk is not an
off-device backup. Private reports remain ignored/unchanged; unrelated files
remain untracked. No Drive operation, push, publication or server restart.
