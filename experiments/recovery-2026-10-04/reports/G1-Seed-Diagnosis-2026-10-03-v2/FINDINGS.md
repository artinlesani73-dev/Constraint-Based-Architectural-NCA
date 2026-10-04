# G1 diagnosis: early growth is the main observed bottleneck

2026-10-03. Fixed final G1 checkpoint, CPU float32; zero optimizer updates or paid compute.

## Matched diagnostic

Selected the central-Y, 24% request in each of the three TRAIN families before executing the comparison. For every case use the same final model, firing seed 2101, 64 steps and zero hidden state, starting from either the single seed, teacher distance <=3, or half the maximum teacher distance. Teacher-derived starts are diagnostic inputs only, not claimed generation results. No development or reserved examples were used here. These three representative cases are not an exhaustive training-set benchmark.

Instrumentation reproduces all three ordinary seed rollouts exactly, including their hidden state. A target-guided control uses the same initial field and firing pattern but accepts only teacher cells. It fills all three targets within 64 steps from every start. This establishes attainable propagation in these cases, not learnability or generalization.

| Starting stage | Median fraction of initially missing target recovered | Correct-addition acceptance per fired opportunity | Final missing cells, total |
|---|---:|---:|---:|
| One voxel | 12.60% | 3.58% | 2,407 |
| Teacher distance <=3 | 35.19% | 5.98% | 1,787 |
| Half teacher distance | 96.14% | 19.30% | 173 |

Recovery excludes cells already supplied in the starting state. Half-distance is a geometric stage and does not mean half the occupied cells. Repeated offers of the same voxel count as separate opportunities in the acceptance column. The half-stage result still includes only 68.3% remaining recovery on the obstructed case, so later continuation is not universally solved.

For single-seed runs, 2,034 of the 2,407 final missing cells never became an eligible fired frontier cell. A further diagnostic count finds 289 final missing cells rejected at least eight times. Slow early growth prevents exposure to much of the eventual target. Larger rollout horizons alone have not solved the frozen development result: 128 steps still produced 0/9 passes in the prior review.

## Loss inspection

At every diagnostic step, compute the existing loss derivative with respect to the current output logits, holding the current state fixed. Every eligible correct addition has a negative derivative: increasing its logit would reduce the loss. Thus this check found no reversed-sign bug in the immediate positive-growth signal. It is not a full gradient-path audit or proof that the shared network can fit all decisions.

The current frontier objective gives the mean positive-class loss weight 0.5 and the mean negative-class loss weight 1.0. In the 50 seed-start steps containing both classes, the summed logit gradient points toward lower shared output bias in 26 steps. The shared parameter update is more complicated than this bias diagnostic; these counts do not establish a causal explanation by themselves.

Because births are hard and detached, the distant missing mass does not receive a direct gradient through those occupancy decisions. Hidden-state gradients still exist. The combined evidence supports testing a less conservative immediate growth objective before changing model size, grid size or constraint families.

## One focused next comparison: G2 proposal

Change only the positive frontier coefficient from 0.5 to 1.0. Retain negative weight 1.0, local volume weight 0.25, network dimensions, fresh initialization seed 1201, TRAIN27, 50/50 start schedule, 64 steps, 256 updates, optimizer, firing seeds and original G1 evaluation gates. This is an experimentally testable proposal, not a demonstrated fix. Higher growth can also increase false additions or worsen long-rollout stability; retain those outcomes and all nine family scores.

Do not reinterpret a lower training loss as success. Compare G2 with the preserved G1 outputs on the same development requests and exact evaluation protocol, disclose reused development data, and keep reserved targets unopened. No threshold, horizon, checkpoint or seed search. G2 requires a versioned implementation, relevant gradient check, and concrete package before asking approval for any paid comparison. No new paid run is authorized by this document.

For intuition only, the class-weighted BCE term on indistinguishable positive and negative predictions has its optimum at probability 1/3 with weights 0.5:1, and at 1/2 with weights 1:1. This toy calculation excludes the volume term and does not imply that equal weighting will cross the hard >0.5 birth threshold or solve the task.

## Preservation

The first diagnostic script passed a Boolean target into the float convolution loss and stopped before producing a case. Its protocol, script and failure are preserved in the sibling G1-Seed-Diagnosis-2026-10-03 folder. This corrected attempt uses target.float(), matching training; no model or loss was changed.

All nine diagnostic fields, target-guided controls, rejection counts, stepwise loss-gradient measurements, protocol and exact script are saved. The original model/source/package remain in the prior verified G1 review and package archives. A new local archive includes this attempt and the failed diagnostic evidence. Same-disk copies are not off-device backups. Repository sync remains pending; MG7 remains live. No Drive, reserved evaluation, deployment, push or paid training occurred.
